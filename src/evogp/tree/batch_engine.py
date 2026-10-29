"""Opt-in "batch" evaluation route: ONE merged evoxc graph per population.

:mod:`evogp.tree.evoxc_engine` specializes one graph per *tree structure* and
:mod:`evogp.tree.dense_engine` compiles one all-ops graph per *shape signature*.
This module is the third route: it traces every **distinct** program of the
population into its own small graph and merges those graphs into ONE executable
with :func:`evoxc.batch_functions`, whose leading axis is the (static) function
axis -- row ``j`` of the merged graph evaluates function ``j``.  One ``evoxc.run``
therefore evaluates the whole population, and only ``evoxc.lower`` /
``evoxc.compile`` / ``evoxc.load`` are ever reached from here (``build_cached`` and
``evoxc.build`` reject a ``Graph``).

The route is reached ONLY through an explicit ``EVOGP_EVAL_MODE=batch``: the
default ``auto`` policy never selects it, and each of the five call sites
(``Tree.forward`` / ``Tree.SR_fitness`` / ``Forest.forward`` /
``Forest.batch_forward`` / ``Forest.SR_fitness``) catches exactly
:class:`BatchUnsupported` and continues on ``Route.fallback_backend``.  A backend
failure from ``lower`` / ``compile`` / ``load`` is NOT converted: there is nothing
safer to fall back to.

Declines and the fallback rule
------------------------------
Every "this route cannot (safely) handle it" answer is :class:`BatchUnsupported`
(a ``RuntimeError``), and it is always safe for the caller to fall back to the
specialized engine on it -- that engine owns the error contract for malformed
trees.  It is raised for: an empty population, a ``per_row``/``x`` shape doubt, an
unknown backend name, ``pop_size < EVOGP_BATCH_MIN_POP``, an op of the merged graph
without a batching rule, and any ``evoxc`` ``TraceError`` / ``TransformError``
raised while tracing or merging (converted, with the original message appended).
Nothing else is converted, and those two exception types never escape.

The four modules of the route (this one is the PUBLIC facade)
------------------------------------------------------------
Callers import ``evogp.tree.batch_engine`` and nothing else; the public surface
below is re-exported, not re-implemented:

* :mod:`evogp.tree.batch_knobs` -- the ``EVOGP_BATCH_*`` constants, the
  ``resolve_*`` helpers, the ``value`` / ``loss`` variant rules and
  :class:`BatchUnsupported`.
* :mod:`evogp.tree.batch_programs` -- host-side program handling: input/label
  conversion, per-function graph bodies, dedup, trace, merge, :func:`_plan_host`,
  :func:`plan_digest` and the :class:`BatchPlan` / :class:`BatchChunk` /
  :class:`BatchExecutable` records.
* :mod:`evogp.tree.batch_chunks` -- the compile-side chunk orchestration: the
  partition arithmetic and the ``lower`` / ``compile`` / ``load`` thread pool.
* :mod:`evogp.tree.batch_stages` -- the per-stage timer surface
  :func:`batch_stage_stats` reports from.
* **this module** -- the three front tables, the plan memo (:func:`_plan_entry`),
  :func:`plan_batch` / :func:`build_batch` / :func:`evaluate_batch` /
  :func:`sr_fitness_batch` / :func:`describe_plan`, the run path, the backend
  resolution and :func:`batch_cache_stats` / :func:`batch_content_stats`.

Chunked compile (``EVOGP_BATCH_COMPILE_WORKERS``)
-------------------------------------------------
The merged compile is the route's dominant cold cost, so the deduped function list
can be split into ``W`` contiguous chunks whose bodies are staged in a thread pool
(:mod:`evogp.tree.batch_chunks`).  ``W <= 1`` -- the default ``0`` and ``1`` both
mean it -- is the ORIGINAL single-executable plan verbatim: one merged graph, one
executable, and ``plan.merged`` / ``plan.graphs`` / ``plan.specs`` /
``plan.report`` describe it.  A chunked plan (``W > 1``, at least
``batch_knobs.MIN_CHUNK_FUNCTIONS`` functions, and neither an explicit band count
nor ``switch_ops``) instead carries one :class:`BatchChunk` per contiguous slice:
``plan.chunks`` is authoritative, ``plan.merged is None``, ``plan.executable is
None`` and ``plan.executables`` holds one body per chunk.  ``chunks`` holds one
record per compiled body in BOTH shapes -- the single-body plan's ``chunks[0]``
mirrors its whole-graph fields object-for-object -- so ``len(plan.chunks)`` is the
plan's chunk count and the build/run paths take the single-body branch exactly when
it is ``1``.  ``row_index`` values stay
GLOBAL indices into ``plan.functions``, runs stay sequential (one device, one
stream, ordered readbacks) and each chunk's rows are written back at its ``offset``,
so the assembled numbers are the single-body plan's.

Front tables (why a repeated call is cheap)
-------------------------------------------
:data:`BATCH_PLAN_MEMO` maps the three population identity tokens
(:func:`evogp.tree.dense_memo.token`: ``data_ptr``/shape/dtype/device/``_version``,
so an in-place mutation is a miss by construction) plus every resolved plan
parameter to the :class:`BatchExecutable`: a repeated evaluation of an unchanged
population does no re-decode, no re-trace, no re-merge and no re-build.  The
``x``/``labels`` conversion is NOT part of that table -- it runs on every call
(see :func:`_plan_entry`).
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
----------------------------------------------------------
The function axis of a merged graph is a position, not a multiset: ``row_index``
indexes it and bands are contiguous slices of it, so the same functions in another
order are another compiled body.  The executable key therefore holds the function
tuple IN ORDER (never a sorted digest), and the content key is order-sensitive for
the same reason -- sorting either would serve a graph whose row ``j`` computes a
different function.

Stage reporting
---------------
:func:`batch_stage_stats` reports the per-stage totals
(:mod:`evogp.tree.batch_stages`: ``decode`` / ``dedup`` / ``trace`` / ``merge`` /
``lower`` / ``compile`` / ``load`` / ``upload`` / ``run`` / ``gather``, plus the
compile region's wall clock and job count) and :func:`batch_stage_reset` zeroes
them; ``BatchPlan.stage_ms`` / ``BatchExecutable.build_stage_ms`` hold the same
split for ONE plan/build.  Timing is measurement only: no stage is reordered and no
number depends on it.

Economics, one line
-------------------
A merged compile is linear in the deduped function count (0.45 s / 2.1 s / 7.1 s
for 4 / 57 / 194 functions) while a warm call is 0.61 / 0.60 / 0.90 / 1.69 ms at
pop 4 / 8 / 64 / 256 -- 1.3x SLOWER than specialized+numpy at pop 4, 6.1x / 13.2x
faster at pop 64 / 256.  So this route pays on repeated evaluation of a stable or
duplicate-rich population, NOT on a generation-per-step GP loop (a new population
is a new program multiset, hence a new compile), and ``DEFAULT_MIN_POP`` is the
measured break-even of that compile.  Design document of record (scheme, the full
grid, band/``switch_ops`` detail, the chunked compile's measured wall reduction,
the on-disk-artifact verdict, the reuse/re-binding contract, open gaps):
``BATCH_EVAL.md`` in this directory.
"""

from __future__ import annotations

import os
import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc

from . import batch_chunks, batch_stages, dense_memo

# The route's whole public vocabulary lives in the siblings -- re-exported here so
# ``evogp.tree.batch_engine`` stays the one import path for callers.
from .batch_knobs import (
    ALIGNMENTS,
    ALIGNMENT_ENV,
    BAND_SIZE_ENV,
    BANDS_ENV,
    CACHE_SIZE_ENV,
    COMPILE_WORKERS_ENV,
    DEFAULT_ALIGNMENT,
    DEFAULT_BAND_SIZE,
    DEFAULT_CACHE_SIZE,
    DEFAULT_COMPILE_WORKERS,
    DEFAULT_MIN_POP,
    EVAL_MODE,
    LOSS_VARIANT,
    MIN_CHUNK_FUNCTIONS,
    MIN_POP_ENV,
    PLAN_VARIANTS,
    SWITCH_OPS_ENV,
    VALUE_VARIANT,
    BatchUnsupported,
    _normalize_variant,
    resolve_alignment,
    resolve_band_count,
    resolve_band_size,
    resolve_bands,
    resolve_compile_workers,
    resolve_min_pop,
    resolve_switch_ops,
)
from .batch_programs import (
    BatchChunk,
    BatchExecutable,
    BatchPlan,
    _host_labels,
    _host_rows,
    _plan_host,
    _pop_size,
    plan_digest,
)
from .evoxc_engine import (
    BACKEND_ENV,
    SUPPORTED_BACKENDS,
    _CPU,
    _env_int,
    _freeze_options,
    _gpu_device_kind,
    _quiet,
    _register_cache_reset,
    _resolve_backend,
    decode_population,
)

__all__ = [
    "BAND_SIZE_ENV",
    "COMPILE_WORKERS_ENV",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_BAND_SIZE",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COMPILE_WORKERS",
    "DEFAULT_MIN_POP",
    "EVAL_MODE",
    "LOSS_VARIANT",
    "MIN_CHUNK_FUNCTIONS",
    "MIN_POP_ENV",
    "PLAN_VARIANTS",
    "VALUE_VARIANT",
    "BatchChunk",
    "BatchExecutable",
    "BatchPlan",
    "BatchUnsupported",
    "batch_cache_stats",
    "batch_content_stats",
    "batch_stage_reset",
    "batch_stage_stats",
    "build_batch",
    "describe_plan",
    "evaluate_batch",
    "plan_batch",
    "resolve_alignment",
    "resolve_band_count",
    "resolve_band_size",
    "resolve_bands",
    "resolve_compile_workers",
    "resolve_min_pop",
    "resolve_switch_ops",
    "sr_fitness_batch",
]

# ---------------------------------------------------------------------------
# front tables
# ---------------------------------------------------------------------------
def _default_cache_size() -> int:
    size = _env_int(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE)
    if size < 0:
        raise ValueError(f"{CACHE_SIZE_ENV}={size} must be >= 0")
    return size


class _ContentMemo(dense_memo.DenseMemo):
    """The content table, where a HIT means an entry was actually SERVED.

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
#: never be handed to another population -- :func:`build_batch` re-binds a hit to
#: the caller's plan (see the module docstring).  On the chunked path the value is
#: that chunk's raw ``evoxc.Executable`` (nothing of the population's), for the same
#: reason a hit is re-bound.
BATCH_EXEC_MEMO = dense_memo.DenseMemo("batch_exec", _default_cache_size())
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
    ``evoxc.lower`` / ``compile`` / ``load``), ``batch_plan_*`` describe
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


def batch_stage_stats() -> Dict[str, Any]:
    """Per-stage totals of this route (32 keys; thin delegation).

    The surface lives in :mod:`evogp.tree.batch_stages`:
    ``batch_<stage>_calls`` / ``batch_<stage>_total_ms`` / ``batch_<stage>_max_ms``
    for the ten stage names, plus ``batch_compile_wall_ms`` (the wall clock of the
    whole parallel lower/compile/load region) and ``batch_compile_jobs`` (chunk
    bodies actually built; 0 when every chunk was a memo hit).

    NOT folded into :func:`batch_cache_stats`: counters describe reuse, these
    describe where the time went, and the specs that pin ``cache_stats()`` /
    ``dense_cache_stats()`` / ``artifact_cache_stats()`` must not learn new keys.
    """
    return batch_stages.stage_stats()


def batch_stage_reset() -> None:
    """Zero the stage counters (timing only -- no cache is touched)."""
    batch_stages.stage_reset()


def plan_batch(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    variant: str = VALUE_VARIANT,
    use_mse: bool = True,
    labels: Any = None,
    band_size: Optional[int] = None,
    bands: Optional[Any] = None,
    switch_ops: Optional[bool] = None,
    alignment: Optional[str] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    compile_workers: Optional[int] = None,
) -> BatchPlan:
    """Decode + dedup + trace + merge one population (host work only, no compile).

    Args:
        batch_node_value / batch_node_type / batch_subtree_size: the population's
            ``(pop_size, max_tree_len)`` tensors (torch on any device or numpy).
        input_len / output_len: the problem's input width and outputs per tree.
        x: the graph input -- ``(pop_size, input_len)`` for ``per_row=True`` (row
            ``i`` of tree ``i``) or ``(N, input_len)`` for ``per_row=False``.
        per_row: selects the two layouts above.
        variant: :data:`VALUE_VARIANT` (default) or :data:`LOSS_VARIANT`
            (``per_row=False`` only).
        use_mse: the loss variant's error metric.
        labels: the loss variant's ``(N, output_len)`` targets (required there).
        band_size / bands / switch_ops / alignment: the merge knobs (see the module
            docstring and the ``resolve_*`` helpers); ``None`` = environment or
            default.
        backend / iree_device_index: resolved here only to record the plan's
            intended route (:func:`build_batch` defaults to it); the merged graph
            itself is backend-agnostic.
        compile_workers: the chunk count request (``EVOGP_BATCH_COMPILE_WORKERS``
            when ``None``); ``0`` / ``1`` plan the original single merged graph and
            ``> 1`` plans contiguous chunks when the population is large enough --
            see :func:`evogp.tree.batch_programs._resolve_workers` for the forcing
            rules.  Nothing is compiled here either way.

    Returns:
        The :class:`BatchPlan`.  Nothing is compiled, no executable exists yet.

    Raises:
        BatchUnsupported: this route cannot handle the population (empty
            population, a shape doubt, a silent op, a trace/merge error).
        ValueError: a malformed knob, an unknown variant, or a malformed tree
            (``decode_population`` owns that message and names row/node).
    """
    if x is None:
        raise BatchUnsupported("batch_engine: an input tensor x is required")
    variant = _normalize_variant(variant, per_row)
    backend_name, device, _ = _resolve_route_backend(backend, iree_device_index)
    size = _pop_size(batch_node_value)
    if size is not None:
        _check_population(size)
    x_host = _host_rows(
        x, int(input_len), "x", per_row=bool(per_row), pop_size=size
    )
    label_host = None
    if variant == LOSS_VARIANT:
        label_host = _host_labels(labels, x_host.shape[0], int(output_len))
    return _plan_host(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        int(input_len),
        int(output_len),
        x_host,
        label_host,
        per_row=bool(per_row),
        variant=variant,
        use_mse=bool(use_mse),
        band_size=band_size,
        bands=bands,
        switch_ops=switch_ops,
        alignment=alignment,
        backend_name=backend_name,
        device=device,
        compile_workers=compile_workers,
    )


def _resolve_route_backend(
    backend: Optional[str], iree_device_index: Optional[int]
) -> Tuple[str, Any, Dict[str, Any]]:
    """Resolve ``(backend_name, device, options)``; a bad NAME is a decline.

    The name is checked before delegating to the engine's resolver, so an unknown
    backend -- which is a "this route does not build that" answer -- surfaces as
    :class:`BatchUnsupported` instead of the resolver's ``ValueError``.  Every
    other resolver error (an unusable ``EVOGP_EVOXC_OPT_LEVEL``, a negative device
    index) is a configuration mistake and keeps propagating as ``ValueError``.
    """
    name = backend if backend else os.environ.get(BACKEND_ENV, "").strip()
    if name and name not in SUPPORTED_BACKENDS:
        raise BatchUnsupported(
            f"batch_engine: unsupported backend {name!r} (this route builds "
            f"{list(SUPPORTED_BACKENDS)})"
        )
    return _resolve_backend(backend, iree_device_index)


def _check_population(pop_size: int, minimum: Optional[int] = None) -> None:
    """Decline for an empty population or one below the threshold."""
    threshold = resolve_min_pop(minimum)
    if pop_size < 1:
        raise BatchUnsupported(
            "batch_engine: empty population (pop_size=0) -- nothing to merge"
        )
    if pop_size < threshold:
        raise BatchUnsupported(
            f"batch_engine: pop_size={pop_size} is below {MIN_POP_ENV}={threshold} "
            "(the merge pays a compile that only pays back at scale)"
        )


# ---------------------------------------------------------------------------
# the executable
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


def build_batch(
    plan: BatchPlan,
    *,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> BatchExecutable:
    """Compile (or reuse) the merged graph of ``plan``; the ONLY build site.

    ``backend`` / ``iree_device_index`` default to the values the plan was resolved
    with, so ``build_batch(plan_batch(...))`` works without repeating them.

    Contract of :data:`BATCH_EXEC_MEMO` (the only table this function reads): a body
    is keyed by the merged function multiset and shapes -- a HIT is REBOUND to the
    current population's plan, never returned with another population's row
    mapping.  Only the compiled body (and the backend / device it was built for)
    crosses between populations; the returned :class:`BatchExecutable` always
    carries ``plan``, so ``pop_size``, ``row_index``, ``function_rows``, ``programs``
    and the staged specs are this call's.  A repeated evaluation of an unchanged
    population therefore still does no re-lower, re-compile or re-load, and a
    same-multiset population of a different size shares the compile WITHOUT sharing
    its staging.

    Two shapes of build, decided by the PLAN (:func:`_plan_host`), never here
    (``len(plan.chunks)`` is the chunk count in both shapes):

    * ``len(plan.chunks) == 1`` -- the original single body: one ``lower``/
      ``compile``/``load`` of ``plan.merged`` (which is that chunk's graph), stored
      in :data:`BATCH_EXEC_MEMO` as the whole :class:`BatchExecutable`, and
      ``executable`` / ``executables`` both set.
    * ``len(plan.chunks) > 1`` -- one body per chunk: every body not already in
      :data:`BATCH_EXEC_MEMO` (keyed with the partition triple, see
      :func:`_chunk_executable_key`) is built by
      :func:`evogp.tree.batch_chunks.build_bodies` in a thread pool of
      ``plan.compile_workers``; each built body is stored raw and every memo hit is
      re-bound to the caller's chunk after :func:`_check_executable_matches`.  Such
      a plan's ``executable`` is ``None`` -- ``executables`` (one body per chunk) is
      authoritative.

    The build path is the explicit evoxc staging sequence --
    ``evoxc.lower`` -> ``evoxc.compile`` -> ``evoxc.load`` -- because
    ``build_cached`` / ``evoxc.build`` reject a ``Graph``.  ``target_backends`` is
    the GPU HAL kind read from torch (:func:`evogp.tree.evoxc_engine._gpu_device_kind`)
    on iree and omitted on numpy; the compile options carry ``opt_level`` only.

    Raises:
        BatchUnsupported: the resolved backend is not one this route builds, or a
            memoized executable does not fit ``plan``
            (:func:`_check_executable_matches`).
        evoxc errors from ``lower`` / ``compile`` / ``load``: propagated (a backend
            failure is not a decline -- there is nothing safer to fall back to);
            on the chunked path the FIRST such failure is re-raised.
    """
    backend_name, device, options = _resolve_route_backend(
        backend if backend is not None else plan.backend_name,
        iree_device_index,
    )
    if len(plan.chunks) <= 1:
        key = _executable_key(plan, backend_name, device, options)
        cached = BATCH_EXEC_MEMO.lookup(key)
        if cached is not None:
            # RE-BIND: the compiled body is shared, the caller's plan is not.  Only
            # ``executable`` / ``backend`` / ``device`` may travel across
            # populations (and ``backend``/``device`` are equal by key); the
            # returned value is a FRESH BatchExecutable around the CURRENT plan, so
            # no downstream read of ``.plan`` can see the first population's row
            # mapping.  ``pins`` stays empty: pins belong to the plan-memo entry
            # that holds the population storages (``_plan_entry`` attaches this
            # call's pins), and a cross-population pin would only keep a foreign
            # buffer alive.
            _check_executable_matches(plan, cached.executable)
            return BatchExecutable(
                plan=plan,
                executable=cached.executable,
                executables=(cached.executable,),
                backend=backend_name,
                device=device,
            )
        executables, build_stage_ms = batch_chunks.build_bodies(
            [plan.merged],
            [0],
            backend_name=backend_name,
            device=device,
            options=options,
            workers=1,
        )
        executable = executables[0]
        built = BatchExecutable(
            plan=plan,
            executable=executable,
            executables=(executable,),
            backend=backend_name,
            device=device,
            build_stage_ms=build_stage_ms,
        )
        BATCH_EXEC_MEMO.store(key, built)
        return built
    bodies: List[Any] = [None] * len(plan.chunks)
    pending: List[Any] = []
    pending_keys: List[Any] = []
    pending_index: List[int] = []
    for chunk in plan.chunks:
        key = _chunk_executable_key(plan, chunk, backend_name, device, options)
        cached = BATCH_EXEC_MEMO.lookup(key)
        if cached is not None:
            _check_executable_matches(
                plan, cached, specs=chunk.specs, n_functions=len(chunk.functions)
            )
            bodies[chunk.index] = cached
            continue
        pending.append(chunk.merged)
        pending_keys.append(key)
        pending_index.append(int(chunk.index))
    if pending:
        built_bodies, build_stage_ms = batch_chunks.build_bodies(
            pending,
            pending_index,
            backend_name=backend_name,
            device=device,
            options=options,
            workers=int(plan.compile_workers),
        )
        for index, key, body in zip(pending_index, pending_keys, built_bodies):
            BATCH_EXEC_MEMO.store(key, body)
            bodies[index] = body
    else:
        build_stage_ms = {"lower": 0.0, "compile": 0.0, "load": 0.0}
    return BatchExecutable(
        plan=plan,
        executable=None,
        executables=tuple(bodies),
        backend=backend_name,
        device=device,
        build_stage_ms=build_stage_ms,
    )


# ---------------------------------------------------------------------------
# the memoized plan+build entry
# ---------------------------------------------------------------------------
def _plan_entry(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    variant: str,
    use_mse: bool,
    labels: Any = None,
    band_size: Optional[int] = None,
    bands: Optional[Any] = None,
    switch_ops: Optional[bool] = None,
    alignment: Optional[str] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    compile_workers: Optional[int] = None,
) -> BatchExecutable:
    """The memoized plan+build pipeline behind both public entry points.

    Two-level plan lookup, then one decode, then a content lookup:

    1. the IDENTITY table :data:`BATCH_PLAN_MEMO`, keyed on the three population
       identity tokens and every resolved plan parameter.  A hit skips the decode
       too -- the entry's plan WAS decoded from this very population, because the
       key holds its ``data_ptr``/shape/dtype/device/``_version`` (pinned so the
       address cannot be recycled) -- so it can never carry another population's row
       mapping.
    2. on an identity miss the population is decoded ONCE (the ``decode`` stage) and
       :func:`evogp.tree.batch_programs.plan_digest` turns the decoded programs (plus
       the ``per_row`` flag and, for ``per_row``, each row's own ``x`` bytes) into
       the content key of :data:`BATCH_CONTENT_MEMO`, together with every resolved
       plan parameter.  A content hit serves a REBUILT-but-equal population (a fresh
       ``Forest`` object, the same evolution) after that one decode -- and only
       after the stored entry's ``plan.programs`` were verified against the
       just-decoded programs: a digest collision or a corrupt entry is counted as a
       content MISS plus a ``batch_content_rejects`` increment and REBUILDS, it is
       never served -- so ``batch_content_hits`` only ever counts served entries.
    3. a rebuild calls :func:`_plan_host` exactly once, passing the population it
       already decoded (the decode is never paid twice), then :func:`build_batch`.

    The band COUNT is deliberately not in a key: it follows from the knob values
    plus the decoded population.  ``compile_workers`` IS in both keys (it changes
    the partition, hence the plan).  ``x`` joins the identity key for
    ``per_row=True``, where the dedup grouping depends on the per-row input bytes.

    What a hit skips is the PLAN work: the decode (identity hit only), the dedup,
    one trace per distinct program, the merge and the compile.  It does NOT skip the
    input conversion -- :func:`_host_rows` (``x``) and, for the loss variant,
    :func:`_host_labels` run BEFORE the lookup on every call, because the key
    carries the converted row count ``x_host.shape[0]`` and the label shape check
    needs the converted rows.

    Pinning: the identity entry carries THIS call's pins (strong references to the
    tokenized storages, keeping ``data_ptr`` ABA-proof), while the content entry is
    stored with ``pins=()`` -- it outlives the population that built it, so a pin
    there would only keep a foreign buffer alive.
    """
    if x is None:
        raise BatchUnsupported("batch_engine: an input tensor x is required")
    variant = _normalize_variant(variant, per_row)
    backend_name, device, _ = _resolve_route_backend(backend, iree_device_index)
    size = _pop_size(batch_node_value)
    if size is not None:
        _check_population(size)
    x_host = _host_rows(
        x, int(input_len), "x", per_row=bool(per_row), pop_size=size
    )
    label_host = None
    if variant == LOSS_VARIANT:
        label_host = _host_labels(labels, x_host.shape[0], int(output_len))
    resolved = {
        "per_row": bool(per_row),
        "variant": variant,
        "use_mse": bool(use_mse),
        "band_size": resolve_band_size(band_size),
        "bands": resolve_bands(bands),
        "switch_ops": resolve_switch_ops(switch_ops),
        "alignment": resolve_alignment(alignment),
        "compile_workers": resolve_compile_workers(compile_workers),
    }
    tokens = [
        dense_memo.token(batch_node_value),
        dense_memo.token(batch_node_type),
        dense_memo.token(batch_subtree_size),
    ]
    if per_row:
        tokens.append(dense_memo.token(x))
    memoized = None not in tokens
    key: Any = None
    if memoized:
        key = (
            tuple(tokens),
            int(input_len),
            int(output_len),
            int(x_host.shape[0]),
            resolved["per_row"],
            resolved["variant"],
            resolved["use_mse"],
            resolved["band_size"],
            resolved["bands"],
            resolved["switch_ops"],
            resolved["alignment"],
            backend_name,
            device,
            resolved["compile_workers"],
        )
        entry = BATCH_PLAN_MEMO.lookup(key)
        if entry is not None:
            return entry
    # Decode ONCE: the content key needs the programs, and a rebuild must not decode
    # a second time (``_plan_host`` is passed the population below).
    decode_start = time.perf_counter()
    pop = decode_population(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        int(input_len),
        int(output_len),
    )
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    batch_stages.record("decode", decode_ms)
    programs = tuple((tree.program, tree.const_out) for tree in pop.trees)
    content_key = (
        plan_digest(programs, x_host if resolved["per_row"] else None,
                    resolved["per_row"]),
        1 if resolved["per_row"] else int(x_host.shape[0]),
        int(input_len),
        int(output_len),
        resolved["per_row"],
        resolved["variant"],
        resolved["use_mse"],
        resolved["band_size"],
        resolved["bands"],
        resolved["switch_ops"],
        resolved["alignment"],
        resolved["compile_workers"],
        backend_name,
        device,
    )
    pins = ()
    if memoized:
        pinned = [batch_node_value, batch_node_type, batch_subtree_size]
        if per_row:
            pinned.append(x)
        pins = dense_memo.pin(pinned)
    content = BATCH_CONTENT_MEMO.probe(content_key)
    if content is not None:
        if content.plan.programs == programs:
            # A verified content hit: the stored plan was built from these very
            # programs (same population content, same plan parameters), so only the
            # pins are this call's.  Counted as a hit HERE, i.e. only once the entry
            # is actually served.
            BATCH_CONTENT_MEMO.record_hit(content_key)
            built = dataclasses.replace(content, pins=pins)
            if memoized:
                BATCH_PLAN_MEMO.store(key, built)
            return built
        # The key matched but the entry is not this population's: a digest
        # collision or a corrupt entry.  It is refused BEFORE any hit is counted --
        # the attempt is a miss plus a reject -- and dropped so it can never be
        # probed again; the rebuild below overwrites the slot.
        BATCH_CONTENT_MEMO.record_miss()
        BATCH_CONTENT_MEMO.discard(content_key)
        _reject_content_entry()
    else:
        BATCH_CONTENT_MEMO.record_miss()
    plan = _plan_host(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        int(input_len),
        int(output_len),
        x_host,
        label_host,
        per_row=resolved["per_row"],
        variant=resolved["variant"],
        use_mse=resolved["use_mse"],
        band_size=resolved["band_size"],
        bands=resolved["bands"],
        switch_ops=resolved["switch_ops"],
        alignment=resolved["alignment"],
        backend_name=backend_name,
        device=device,
        pop=pop,
        compile_workers=resolved["compile_workers"],
    )
    plan = dataclasses.replace(
        plan, stage_ms={"decode": decode_ms, **plan.stage_ms}
    )
    built = build_batch(plan, backend=backend_name, iree_device_index=iree_device_index)
    if memoized:
        # Strong references to the tokenized storages, held BY the stored value: a
        # live pin keeps the buffer allocated, so a ``data_ptr`` token cannot be
        # recycled by a different tensor while the entry lives (ABA).  The extra
        # field is the only reason the stored object is a ``replace`` copy.
        built = dataclasses.replace(built, pins=pins)
        BATCH_PLAN_MEMO.store(key, built)
    BATCH_CONTENT_MEMO.store(content_key, dataclasses.replace(built, pins=()))
    return built


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _inputs(entry: BatchExecutable, x_host: np.ndarray,
            label_host: Optional[np.ndarray],
            function_rows: Optional[Sequence[int]] = None) -> Tuple[Any, ...]:
    """The graph inputs of one BODY, staged for the entry's backend.

    ``per_row=True`` feeds each merged function its own ``x`` row, so the host
    block is gathered to ``(n_functions, 1, input_len)`` (axes=0 on the function
    axis); every other layout shares one ``(n_rows, input_len)`` block.
    ``function_rows`` is the row slice of the body being fed: ``None`` means the
    whole plan (the single-body path), while a :class:`BatchChunk` passes its own
    ``function_rows`` so a chunked plan gathers exactly the rows its functions were
    first seen at.  On iree each block is uploaded once per call
    (``evoxc.from_numpy`` is a zero-copy host wrap; the upload is
    ``Tensor.to(device)``); the numpy backend consumes the host array directly and
    never uploads.
    """
    plan = entry.plan
    if plan.per_row:
        rows = plan.function_rows if function_rows is None else function_rows
        block = np.ascontiguousarray(
            x_host[list(rows)][:, None, :], dtype=np.float32
        )
    else:
        block = np.ascontiguousarray(x_host, dtype=np.float32)
    staged: List[Any] = [block]
    if label_host is not None:
        staged.append(np.ascontiguousarray(label_host, dtype=np.float32))
    if entry.backend == "numpy":
        return tuple(staged)
    return tuple(evoxc.from_numpy(array).to(entry.device) for array in staged)


def _readback(result: Any, backend: str) -> np.ndarray:
    """Host float32 copy of one body's result (the only device->host crossing).

    ``numpy`` reads the interpreter's array directly; every device backend needs
    ``.to(_CPU)`` + ``.numpy()`` (evoxc has no zero-copy device export).  Both paths
    copy into a fresh ``float32`` array, so a chunk's rows can be written into the
    global block without aliasing a backend buffer.
    """
    if backend == "numpy":
        return np.asarray(result.numpy(), dtype=np.float32)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32)


def _run(entry: BatchExecutable, x_host: np.ndarray,
         label_host: Optional[np.ndarray]) -> np.ndarray:
    """One ``evoxc.run`` per compiled body; the raw host result in FUNCTION order.

    A single-chunk plan (``len(plan.chunks) == 1``) is the original statement: one
    staged input tuple, one ``evoxc.run`` of :attr:`BatchExecutable.executable`, one
    readback, returned as-is.

    A chunked plan (``len(plan.chunks) > 1``) runs its bodies in ASCENDING chunk
    order and writes each result
    into a preallocated ``(n_functions, ...)`` block at the chunk's ``offset``, so
    the rows come back in the GLOBAL function order that ``plan.row_index`` indexes
    (the readback order never decides the output order).  The runs stay SEQUENTIAL:
    there is one device and one stream, each readback must stay ordered with the run
    that produced it, and a chunk's input staging is a host gather -- a pool here
    would add contention, not throughput.

    Stages: ``upload`` wraps the staging, ``run`` the ``evoxc.run`` and ``gather``
    the readback (the caller stages ``_map_rows`` / the output tensor).
    """
    plan = entry.plan
    if len(plan.chunks) <= 1:
        with batch_stages.stage("upload"):
            staged = _inputs(entry, x_host, label_host)
        with batch_stages.stage("run"):
            with _quiet():
                result = evoxc.run(entry.executable, *staged)
        with batch_stages.stage("gather"):
            return _readback(result, entry.backend)
    tail: Tuple[int, ...] = (
        () if plan.variant == LOSS_VARIANT else (int(plan.n_rows), int(plan.output_len))
    )
    rows = np.empty((len(plan.functions),) + tail, dtype=np.float32)
    executables = tuple(entry.executables)
    if len(executables) != len(plan.chunks):
        raise BatchUnsupported(
            f"batch_engine: the chunked plan holds {len(plan.chunks)} chunk(s) but "
            f"{len(executables)} executable(s) were bound to it"
        )
    for chunk, executable in zip(plan.chunks, executables):
        size = len(chunk.functions)
        with batch_stages.stage("upload"):
            staged = _inputs(entry, x_host, label_host, chunk.function_rows)
        with batch_stages.stage("run"):
            with _quiet():
                result = evoxc.run(executable, *staged)
        with batch_stages.stage("gather"):
            block = _readback(result, entry.backend)
            if block.shape[0] != size:
                raise BatchUnsupported(
                    f"batch_engine: chunk {chunk.index} was expected to answer "
                    f"{size} function(s), got {block.shape[0]}"
                )
            rows[chunk.offset:chunk.offset + size] = block
    return rows


def _map_rows(rows: np.ndarray, plan: BatchPlan) -> np.ndarray:
    """Expand the per-function rows back to the population order (a fresh copy)."""
    return np.ascontiguousarray(rows[list(plan.row_index)])


def _to_tensor(array: np.ndarray, source: Any, out_device: Optional[Any]) -> torch.Tensor:
    """A fresh, contiguous, ``requires_grad=False`` float32 tensor on the target."""
    tensor = torch.from_numpy(array).contiguous().detach().requires_grad_(False)
    target = out_device if out_device is not None else getattr(source, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def evaluate_batch(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    band_size: Optional[int] = None,
    bands: Optional[Any] = None,
    switch_ops: Optional[bool] = None,
    alignment: Optional[str] = None,
    compile_workers: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate a whole population with ONE merged graph (drop-in for
    :func:`evogp.tree.dense_engine.evaluate_dense`).

    Contracts (mirroring :func:`evogp.tree.evoxc_engine.evaluate` exactly):

    * ``per_row=True``: ``x`` is ``(pop_size, input_len)``, tree ``i`` sees row
      ``i``; the result is ``(pop_size, output_len)``.
    * ``per_row=False``: ``x`` is ``(N, input_len)``, every tree is evaluated on all
      ``N`` rows; the result is ``(pop_size, N, output_len)``.
    * the result is fresh, contiguous, ``requires_grad=False`` float32 on ``x``'s
      device (``out_device`` overrides); ``x`` and the population are never modified.

    Duplicate programs are merged once and their rows are expanded on the host
    (``torch``-free fancy indexing) before the single result tensor is created, so
    the engine's output never makes more than one round trip.

    Raises:
        BatchUnsupported: fall back to the specialized engine (see the module
            docstring for the enumerated declines).
    """
    entry = _plan_entry(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        x,
        per_row=per_row,
        variant=VALUE_VARIANT,
        use_mse=True,
        band_size=band_size,
        bands=bands,
        switch_ops=switch_ops,
        alignment=alignment,
        backend=backend,
        iree_device_index=iree_device_index,
        compile_workers=compile_workers,
    )
    plan = entry.plan
    x_host = _input_block(entry, x, input_len, per_row)
    rows = _run(entry, x_host, None)
    with batch_stages.stage("gather"):
        if plan.per_row:
            rows = rows[:, 0, :]
        return _to_tensor(_map_rows(rows, plan), x, out_device)


def _input_block(entry: BatchExecutable, x: Any, input_len: int,
                 per_row: bool) -> np.ndarray:
    """Host copy of ``x`` for the run.

    Deliberately re-read rather than memoized: the two front tables are keyed on
    the POPULATION (a plan) and on the compiled body (an executable), and a
    per-call ``x`` conversion is one D2H -- the same single crossing the dense
    route's input table would save on a repeated dataset, without a third table to
    keep coherent (the wiring round can add one if it measures a need).
    """
    return _host_rows(x, int(input_len), "x", per_row=bool(per_row),
                      pop_size=entry.plan.pop_size)


def sr_fitness_batch(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    inputs: Any,
    labels: Any,
    use_MSE: bool = True,
    *,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
    band_size: Optional[int] = None,
    bands: Optional[Any] = None,
    switch_ops: Optional[bool] = None,
    alignment: Optional[str] = None,
    compile_workers: Optional[int] = None,
) -> torch.Tensor:
    """SR fitness with the reduction folded IN-GRAPH (drop-in for
    :func:`evogp.tree.dense_engine.sr_fitness_dense`).

    ``inputs`` is ``(N, input_len)``, ``labels`` is ``(N, output_len)``; the result
    is a ``(pop_size,)`` float32 POSITIVE loss on ``inputs``' device unless
    ``out_device``::

        err  = (pred - label) ** 2      if use_MSE
        err  = |pred - label|           otherwise
        loss = sum_N sum_out err / N

    Each merged function reduces to ONE rank-0 scalar inside the graph
    (``evoxc.reduce_sum`` over both axes, then a static ``/ N`` divide), so the
    merged output is ``(n_functions,)`` and the ONLY thing that crosses back is that
    vector -- no prediction ever round-trips.  Outputs are summed, never divided by
    ``output_len``; NaN/inf propagate unsanitized; the caller negates (the
    higher-is-better convention).  Duplicate rows are expanded host-side.

    Raises:
        BatchUnsupported: fall back to the specialized engine.
    """
    entry = _plan_entry(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        inputs,
        per_row=False,
        variant=LOSS_VARIANT,
        use_mse=bool(use_MSE),
        labels=labels,
        band_size=band_size,
        bands=bands,
        switch_ops=switch_ops,
        alignment=alignment,
        backend=backend,
        iree_device_index=iree_device_index,
        compile_workers=compile_workers,
    )
    plan = entry.plan
    x_host = _input_block(entry, inputs, input_len, False)
    label_host = _host_labels(labels, x_host.shape[0], plan.output_len)
    rows = _run(entry, x_host, label_host)
    with batch_stages.stage("gather"):
        return _to_tensor(
            np.ascontiguousarray(_map_rows(rows, plan)), inputs, out_device
        )


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------
def describe_plan(plan: BatchPlan) -> Dict[str, Any]:
    """A cheap, JSON-ish description of ``plan`` (for tests and a future runner).

    Reports the population size and the deduped function count, how many rows
    collide onto an earlier function, the merge shape (band count/strategy/sizes,
    emitted ops, class counters), the graph input shapes, the route
    (backend/device), and the plan's build shape: ``compile_workers`` (the chunk
    count actually used, ``1`` for the single-body plan), ``chunk_count`` /
    ``chunk_sizes`` (read off ``plan.chunks``, which holds one record per compiled
    body in both plan shapes) and the plan-side ``stage_ms`` split.  It reads the
    stored report only -- nothing is decoded, traced or compiled.

    A CHUNKED plan (``chunk_count > 1``) reports its whole-graph keys from the
    aggregate report: ``specs`` / ``spec_dtypes`` are empty (no single graph
    exists), ``band_sizes`` is empty and ``band_strategy`` / the class counters are
    ``None``; ``emitted_ops`` is the summed value and ``n_functions`` the global
    count.
    """
    report = plan.report or {}
    shapes = [tuple(int(dim) for dim in spec.shape) for spec in plan.specs]
    return {
        "variant": plan.variant,
        "per_row": bool(plan.per_row),
        "use_mse": bool(plan.use_mse) if plan.variant == LOSS_VARIANT else None,
        "pop_size": int(plan.pop_size),
        "functions": len(plan.functions),
        "const_functions": sum(1 for program, _ in plan.functions if program is None),
        "row_index_collisions": int(plan.pop_size) - len(plan.functions),
        "input_len": int(plan.input_len),
        "output_len": int(plan.output_len),
        "n_rows": int(plan.n_rows),
        "band_count": int(plan.band_count),
        "band_sizes": list(report.get("band_sizes") or ()),
        "band_strategy": report.get("band_strategy"),
        "switch_ops": bool(plan.switch_ops),
        "alignment": plan.alignment,
        "emitted_ops": report.get("emitted_ops"),
        "value_classes": report.get("value_classes"),
        "shared_classes": report.get("shared_classes"),
        "stacked_constants": report.get("stacked_constants"),
        "output_muxes": report.get("output_muxes"),
        "n_functions": report.get("n_functions"),
        "compile_workers": int(plan.compile_workers),
        "chunk_count": len(plan.chunks),
        "chunk_sizes": [int(len(chunk.functions)) for chunk in plan.chunks],
        "stage_ms": dict(plan.stage_ms),
        "specs": shapes,
        "spec_dtypes": [str(spec.dtype) for spec in plan.specs],
        "backend": plan.backend_name,
        "device": str(plan.device),
    }
