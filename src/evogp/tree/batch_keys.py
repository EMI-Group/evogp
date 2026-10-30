"""Front tables and cache keys of the batch route: what a compiled body is keyed on.

This module owns ONE responsibility for :mod:`evogp.tree.batch_engine` (the
opt-in merged-graph "batch" route): the IDENTITY of a compiled body, and the
tables that hold it.  :mod:`evogp.tree.batch_programs` builds a plan,
:mod:`evogp.tree.batch_run` runs one body, and the facade is the only caller of
the names below -- nothing else in the library spells a batch cache key:

* the three front tables -- :data:`BATCH_PLAN_MEMO` (identity), the
  :class:`_ContentMemo` table :data:`BATCH_CONTENT_MEMO` (content digest) and
  :data:`BATCH_EXEC_MEMO` (compiled bodies);
* :func:`_default_cache_size` (the shared import-time bound
  ``EVOGP_BATCH_CACHE_SIZE``) and the invalidation hook :func:`_clear_tables`;
* :func:`_executable_key` / :func:`_chunk_executable_key` (via
  :func:`_body_key`) -- the memo key of one compiled body;
* :func:`_check_executable_matches` / :func:`_spec_shapes` -- the chunk-aware
  guard that a hit really fits the caller's plan;
* :func:`batch_cache_stats` / :func:`batch_content_stats` and the
  ``batch_content_rejects`` counter behind them.

Import direction: this module imports :mod:`evogp.tree.batch_knobs`,
:mod:`evogp.tree.batch_programs`, :mod:`evogp.tree.dense_memo` and
:mod:`evogp.tree.evoxc_engine`; it never imports
:mod:`evogp.tree.batch_engine` (the facade imports this module, not the other
way round).

Front tables (why a repeated call is cheap)
-------------------------------------------
:data:`BATCH_PLAN_MEMO` maps the three population identity tokens
(:func:`evogp.tree.dense_memo.token`: ``data_ptr``/shape/dtype/device/``_version``,
so an in-place mutation is a miss by construction) plus every resolved plan
parameter to the :class:`BatchExecutable`: a repeated evaluation of an unchanged
population does no re-decode, no re-trace, no re-merge and no re-build.  The
``x``/``labels`` conversion is NOT part of that table -- it runs on every call
(see :func:`evogp.tree.batch_engine._plan_entry`).
:data:`BATCH_CONTENT_MEMO` is the second plan table and answers what the identity
table cannot: a REBUILT population with identical content (a fresh ``Forest``
object carrying the same evolution) hits on
:func:`evogp.tree.batch_programs.plan_digest` and pays only the decode needed to
prove it.  A content hit is verified against the just-decoded programs before it
is served, never trusted: ``batch_content_hits`` counts only entries actually
served, while a refused one is a miss plus a ``batch_content_rejects`` increment.
:data:`BATCH_EXEC_MEMO` keys a compiled body on the ACTUAL program multiset plus
the shape/backend/device/options (and, for a chunked plan, the partition
``(chunk_count, chunk_sizes, chunk_index)``, so a different ``W`` can never hit a
body built for another partition), so the same programs in a fresh population
object still hit -- and such a hit is RE-BOUND to the caller's plan, never returned
with another population's row mapping.  All three tables are dropped by
:func:`evogp.tree.evoxc_engine.clear_cache` (the engine's single invalidation
point) and counted by :func:`batch_cache_stats` / :func:`batch_content_stats` --
separate functions, because ``cache_stats()`` (8 keys) and ``dense_cache_stats()``
are pinned by the suite.

Order sensitivity (why the executable key is never sorted)
---------------------------------------------------------
The function axis of a merged graph is a position, not a multiset: ``row_index``
indexes it and bands are contiguous slices of it, so the same functions in another
order are another compiled body.  The executable key therefore holds the function
tuple IN ORDER (never a sorted digest), and the content key is order-sensitive for
the same reason -- sorting either would serve a graph whose row ``j`` computes a
different function.

Two lifetimes
-------------
``evoxc_engine.clear_cache()`` is the in-memory point: :func:`_clear_tables` is
registered there (as a hook, not via ``dense_memo.register``, so
``dense_cache_stats()`` keeps returning exactly the dense route's keys) and drops
all three tables plus the refusal counter.  Nothing on disk is involved: the batch
route's disk tier is :mod:`evogp.tree.batch_artifacts` (the engine's
``EVOGP_EVOXC_ARTIFACT_DIR`` layer, deleted by ``evoxc_engine.clear_artifacts()``
and counted by ``batch_artifact_stats()``), which this module never touches.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Sequence, Tuple

from . import dense_memo

# The route's vocabulary is batch_knobs'; the records and the digest are
# batch_programs'; the env helper and the invalidation seam are the engine's.
from .batch_knobs import CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE, LOSS_VARIANT, BatchUnsupported
from .batch_programs import BatchChunk, BatchExecutable, BatchPlan
from .evoxc_engine import _env_int, _freeze_options, _register_cache_reset

__all__ = [
    "BATCH_CONTENT_MEMO",
    "BATCH_EXEC_MEMO",
    "BATCH_PLAN_MEMO",
    "batch_cache_stats",
    "batch_content_stats",
]


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
def _default_cache_size() -> int:
    size = _env_int(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE)
    if size < 0:
        raise ValueError(f"{CACHE_SIZE_ENV}={size} must be >= 0")
    return size


class _ContentMemo(dense_memo.DenseMemo):
    """A table whose HIT means an entry was actually SERVED (not merely found).

    Used by BOTH tables that need it: the content table (below) and the executable
    table, whose hits are served either from memory or from the opt-in disk tier
    (:mod:`evogp.tree.batch_artifacts`), so neither may be counted as a compile
    miss.

    The verification (``entry.plan.programs == programs``) happens in the caller,
    AFTER the key lookup, so the plain :meth:`DenseMemo.lookup` cannot tell a
    served entry from a refused one: it would count the hit first and the caller
    would then refuse it.  This subclass splits the two steps -- :meth:`probe`
    reads a value WITHOUT counting, and the caller then records exactly one
    outcome (:meth:`record_hit` / :meth:`record_miss`) -- so
    ``batch_content_hits`` only ever counts an entry that was returned to the
    caller.  The counters stay in the memo (``stats()`` unchanged, five keys via
    :func:`batch_content_stats`) and the LRU touch moves to the hit record, so a
    refused entry does not get promoted either.
    """

    def probe(self, key: Any) -> Any:
        """The value stored under ``key``, or ``None`` -- counters untouched."""
        with self._lock:
            if self.maxsize <= 0:  # disabled table: nothing can be served
                return None
            try:
                return self[key]
            except KeyError:
                return None

    def record_hit(self, key: Any) -> None:
        """Count one SERVED entry and mark ``key`` as most recently used."""
        with self._lock:
            self.hits += 1
            if key in self:
                self.move_to_end(key)

    def record_miss(self) -> None:
        """Count one attempt that has to rebuild (absent, refused or disabled)."""
        with self._lock:
            self.misses += 1

    def discard(self, key: Any) -> None:
        """Drop a refused entry so it can never be probed again."""
        with self._lock:
            self.pop(key, None)


#: Compiled bodies, keyed by the actual program multiset + shape +
#: backend/device/options (see :func:`_executable_key`; a chunked plan adds the
#: partition triple).  The key holds the REAL programs, never a digest: a hash
#: collision would serve wrong numbers.  On the single-body path the value is the
#: whole :class:`BatchExecutable` of the population that FIRST built it, so it must
#: never be handed to another population -- :func:`evogp.tree.batch_engine.build_batch`
#: re-binds a hit to the caller's plan (see the facade's module docstring).  On the
#: chunked path the value is that chunk's raw ``evoxc.Executable`` (nothing of the
#: population's), for the same reason a hit is re-bound.
#: A hit is counted for a body served EITHER from this table or from the opt-in disk
#: tier, so ``batch_exec_misses`` is the number of bodies actually COMPILED (a disk
#: hit is not a compile miss); ``batch_artifact_stats()`` splits the two.
BATCH_EXEC_MEMO = _ContentMemo("batch_exec", _default_cache_size())
#: :class:`BatchExecutable` of an unchanged population, keyed by the three
#: population identity tokens + every resolved plan parameter.
BATCH_PLAN_MEMO = dense_memo.DenseMemo("batch_plan", _default_cache_size())
#: :class:`BatchExecutable` of a population with the same CONTENT, keyed by
#: :func:`evogp.tree.batch_programs.plan_digest` + every resolved plan parameter, so
#: a rebuilt-but-equal population hits after one decode.  The stored entry carries
#: NO pins (its plan outlives the population that built it) and is verified against
#: the just-decoded programs before it is served.
BATCH_CONTENT_MEMO = _ContentMemo("batch_content", _default_cache_size())

#: Content entries refused because the verification failed (a digest collision or a
#: corrupt entry); a refusal always rebuilds and never serves.
_CONTENT_REJECTS = 0
_CONTENT_LOCK = threading.Lock()


def _clear_tables() -> None:
    """Drop all three front tables (registered with the engine's invalidation)."""
    global _CONTENT_REJECTS
    BATCH_EXEC_MEMO.reset()
    BATCH_PLAN_MEMO.reset()
    BATCH_CONTENT_MEMO.reset()
    with _CONTENT_LOCK:
        _CONTENT_REJECTS = 0


#: :func:`evogp.tree.evoxc_engine.clear_cache` drops these tables too -- the
#: engine stays the single in-memory invalidation point.  Registered as a hook
#: (not via ``dense_memo.register``) so ``dense_cache_stats()`` keeps returning
#: exactly the dense route's keys.
_register_cache_reset(_clear_tables)


def _reject_content_entry() -> None:
    """Count one refused content entry (a rebuild follows, never a serve)."""
    global _CONTENT_REJECTS
    with _CONTENT_LOCK:
        _CONTENT_REJECTS += 1


def batch_cache_stats() -> Dict[str, int]:
    """Counters of this route's two front tables (eight keys).

    ``batch_exec_hits`` / ``batch_exec_misses`` / ``batch_exec_entries`` /
    ``batch_exec_maxsize`` describe :data:`BATCH_EXEC_MEMO` (a hit never reaches
    ``evoxc.lower`` / ``compile`` / ``load``, whether it came from memory or from
    the opt-in disk tier of :mod:`evogp.tree.batch_artifacts`; a miss is a body that
    really compiled), ``batch_plan_*`` describe
    :data:`BATCH_PLAN_MEMO` (a hit skips the host decode, the traces and the
    merge).  Deliberately a separate function: ``cache_stats()`` (exactly 8 keys),
    ``dense_cache_stats()`` and ``artifact_cache_stats()`` are pinned by the suite.

    The executable contract those counters describe: an executable is keyed by
    the merged function multiset and shapes -- a hit is REBOUND to the current
    population's plan, never returned with another population's row mapping.  So
    a rising ``batch_exec_hits`` on a differently-sized population is reuse, not
    staleness: the compile is shared, the ``pop_size`` / ``row_index`` /
    ``function_rows`` staging is always the caller's.
    """
    stats: Dict[str, int] = {}
    stats.update(BATCH_EXEC_MEMO.stats())
    stats.update(BATCH_PLAN_MEMO.stats())
    return stats


def batch_content_stats() -> Dict[str, int]:
    """Counters of the CONTENT-addressed plan table (five keys).

    ``batch_content_hits`` counts entries actually SERVED (a verified plan, after
    ONE decode), ``batch_content_misses`` counts attempts that had to rebuild, and
    ``batch_content_rejects`` counts the entries refused by the program
    verification -- a subset of those misses, since a refused entry always
    rebuilds.  ``batch_content_entries`` / ``batch_content_maxsize`` describe the
    table itself.

    Deliberately separate from :func:`batch_cache_stats` (which stays at exactly its
    eight identity-keyed counters), and from ``cache_stats()`` /
    ``dense_cache_stats()`` / ``artifact_cache_stats()``, which are pinned by the
    suite.
    """
    stats: Dict[str, int] = {}
    stats.update(BATCH_CONTENT_MEMO.stats())
    with _CONTENT_LOCK:
        stats["batch_content_rejects"] = int(_CONTENT_REJECTS)
    return stats


# ---------------------------------------------------------------------------
# what ONE compiled body is keyed on
# ---------------------------------------------------------------------------
def _body_key(
    plan: BatchPlan,
    functions: Sequence[Any],
    band_count: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Everything that determines ONE compiled body of ``plan``.

    The function tuple is part of the key IN ORDER and never sorted: the function
    axis is a POSITION (``row_index`` indexes it and bands are contiguous slices),
    so the same functions in another order are another graph.  ``band_count`` is the
    body's own band count (the plan's for the single-body path, the chunk's for a
    chunked one).
    """
    return (
        tuple(functions),
        plan.variant,
        bool(plan.use_mse),
        int(band_count),
        bool(plan.switch_ops),
        plan.alignment,
        int(plan.n_rows),
        int(plan.input_len),
        int(plan.output_len),
        bool(plan.per_row),
        backend_name,
        device,
        _freeze_options(options),
    )


def _executable_key(
    plan: BatchPlan, backend_name: str, device: Any, options: Dict[str, Any]
) -> Tuple[Any, ...]:
    """Memo key of the SINGLE-body executable (unchanged from the original plan).

    Everything that determines the compiled body: the ACTUAL program multiset
    (never a digest -- a collision would serve wrong numbers), the variant, the
    loss metric, the merge shape (band count, ``switch_ops``, alignment, rows), the
    I/O shape and the backend/device/options.  Two populations with the same
    programs therefore share the executable, and any of these changing rebuilds.
    """
    return _body_key(
        plan, plan.functions, plan.band_count, backend_name, device, options
    )


def _chunk_executable_key(
    plan: BatchPlan,
    chunk: BatchChunk,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Memo key of ONE chunk body: the body key plus the partition triple.

    The partition ``(chunk_count, chunk_sizes, chunk_index)`` -- read off
    ``plan.chunks``, the authoritative read surface -- is appended so a body built
    for one ``W`` can never be served to a plan with another ``W``: the two bodies
    would compute the same functions but the readback offsets of the caller's plan
    belong to the caller's partition, and an accidental cross-partition hit would be
    a structural mismatch nothing else checks.
    """
    return _body_key(
        plan, chunk.functions, chunk.band_count, backend_name, device, options
    ) + (
        int(len(plan.chunks)),
        tuple(int(len(other.functions)) for other in plan.chunks),
        int(chunk.index),
    )


def _spec_shapes(specs: Sequence[Any]) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    """``((shape, dtype), ...)`` of a sequence of ``evoxc.TensorSpec`` objects."""
    return tuple(
        (tuple(int(dim) for dim in spec.shape), str(spec.dtype)) for spec in specs
    )


def _check_executable_matches(
    plan: BatchPlan,
    executable: Any,
    *,
    specs: Optional[Sequence[Any]] = None,
    n_functions: Optional[int] = None,
) -> None:
    """Decline when a memoized executable does not fit the CALLER's plan/body.

    Chunk-aware: ``specs`` defaults to the plan's own (single-body) input specs and
    ``n_functions`` to ``len(plan.functions)``; a chunked build passes its CHUNK's
    specs and function count instead, which are what the chunk's merged graph was
    lowered from.

    The executable key (:func:`_body_key`) already pins the compiled body's inputs
    -- program multiset, variant/loss metric, merge shape, rows, input width,
    outputs, backend/device -- so this check is an invariant, not the mechanism: it
    re-derives the two things the caller is about to rely on from the executable's
    own ``evoxc`` signature (:attr:`evoxc.Executable.signature`, recorded at
    ``lower`` time) and compares them with the body's plan:

    * the merged INPUT specs must equal the body's ``evoxc.TensorSpec`` objects
      (shape + dtype) -- on ``per_row=True`` that leading dimension is the merged
      function count;
    * the merged OUTPUT spec must be ``(n_functions, n_rows, output_len)`` for the
      value variant and ``(n_functions,)`` for the loss variant, i.e. the merged
      function axis plus the rows/outputs the plan stages.

    A mismatch means the table served a body built for a different function
    multiset or shape; that is a :class:`BatchUnsupported` decline (the caller
    falls back to the specialized engine) rather than a silently wrong row mapping.
    Nothing is checked when the executable records no signature at all (an
    executable we cannot introspect is still the one the key selected).
    """
    signature = getattr(executable, "signature", None)
    if signature is None:  # pragma: no cover - evoxc always records a signature
        return
    body_specs = plan.specs if specs is None else tuple(specs)
    actual_in = _spec_shapes(getattr(signature, "input_specs", ()) or ())
    expected_in = _spec_shapes(body_specs)
    if actual_in and actual_in != expected_in:
        raise BatchUnsupported(
            f"batch_engine: the memoized executable was compiled for inputs "
            f"{actual_in}, but this population's plan needs {expected_in}"
        )
    count = len(plan.functions) if n_functions is None else int(n_functions)
    if plan.variant == LOSS_VARIANT:
        expected_out: Tuple[int, ...] = (count,)
    else:
        expected_out = (count, int(plan.n_rows), int(plan.output_len))
    actual_out = tuple(shape for shape, _ in _spec_shapes(
        getattr(signature, "output_specs", ()) or ()))
    if len(actual_out) == 1 and actual_out[0] != expected_out:
        raise BatchUnsupported(
            f"batch_engine: the memoized executable has {count} merged "
            f"function(s) and was compiled for output "
            f"{actual_out[0]}, but this population's plan needs {expected_out}"
        )
