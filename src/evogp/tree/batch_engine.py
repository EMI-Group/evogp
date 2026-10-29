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

Opt-in
------
The route is **opt-in** and nothing in the library calls it yet: the route
switch (``EVOGP_EVAL_MODE=batch``) is added by the wiring task, which will call
:func:`evaluate_batch` / :func:`sr_fitness_batch` from the same five call sites
that ask :func:`evogp.tree.dense_engine.resolve_route` today.  Until that lands,
a caller picks this route by calling it; ``EVOGP_EVAL_MODE`` is not consulted
here at all (exactly like the two benchmark runner builders).

Declines and the fallback rule
------------------------------
Every "this route cannot (safely) handle it" answer is :class:`BatchUnsupported`
(a ``RuntimeError``), and it is always safe for the caller to fall back to the
specialized engine on it -- that engine owns the error contract for malformed
trees.  It is raised for: an empty population, a ``per_row``/``x`` shape doubt, an
unknown backend name, ``pop_size < EVOGP_BATCH_MIN_POP``, an op of the merged
graph without a batching rule, and any ``evoxc`` ``TraceError`` / ``TransformError``
raised while tracing or merging (converted, with the original message appended).
Nothing else is converted, and those two exception types never escape.

Economics (measured on this box: AMD MI300X VF, torch 2.13.0+rocm7.1, IREE 3.11,
``EVOXC_DISABLE_LOCATIONS=1``, ``input_len=8``, ``N=1024`` data rows, ``L=32``
descriptor with ``max_layer_cnt=4``, opt level ``O1``, rocm target)
--------------------------------------------------------------------------------
``plan`` = decode + dedup + trace + merge (pure host work, no compile);
``cold`` = plan + ONE merged compile + load; ``warm`` = one ``evoxc.run`` plus
one readback and the host row expansion.  ``functions`` counts the deduped
``(program, const_out)`` multiset, i.e. the merged function axis.

=========  ==========  =====  ==========  ==================  =============
pop        functions   bands  ops v/loss  plan ms             warm ms v/loss
=========  ==========  =====  ==========  ==================  =============
4          4           1      32 / 47     4                   0.61 / 0.72
8          8           1      61 / 88     4                   0.60 / 0.77
64         57          2      464 / 647   26                  0.90 / 0.87
256        194         7      1948/2625   188                 1.69 / 2.40
=========  ==========  =====  ==========  ==================  =============

* planning is ~0.5-1 ms per merged function (plus the ``decode_population`` of
  the whole population: 188 ms at pop 256 is mostly the host decode).
* compiling is linear in the merged function count and large -- 0.45 s at 4
  functions, 2.1 s at 57, 7.1 s at 194 -- which is why it happens ONCE per
  (program multiset, shape) and is memoized in :data:`BATCH_EXEC_MEMO`.
* a warm run is dispatch-light: identical graph positions across functions
  collapse into ONE batched op, so a whole 194-function population is 1.7 ms for
  1024 rows while the specialized route pays 194 invocations and 194 readbacks.
* profit against the OTHER two routes on the same workload (loss variant,
  pop 64, ``N=1024``): specialized+iree cold 13.9 s / warm 13.2 ms against
  batch+iree cold 2.4 s / warm 0.95 ms -- 5.8x cheaper cold and 14x faster warm.
  Against the specialized **numpy** warm value path (0.47 / 0.71 / 5.50 / 22.3 ms
  at pop 4 / 8 / 64 / 256) the batch+iree warm call is 0.61 / 0.60 / 0.90 /
  1.69 ms, i.e. 1.3x slower at pop 4, 1.2x faster at pop 8 and 6.1x / 13.2x
  faster at pop 64 / 256.
* the numpy backend works through the very same three calls (with
  ``Device("cpu", 0)`` and no compile options) but is **debug only**: merged
  numpy is 0.53 / 1.09 / 12.1 / 54.2 ms at pop 4 / 8 / 64 / 256 against
  0.47 / 0.71 / 5.50 / 22.3 ms for the specialized numpy route (1.1x / 1.5x /
  2.2x / 2.4x SLOWER), because the interpreter pays per-op overhead on the merged
  graph while paying almost nothing for a specialized per-tree graph.  Use it to
  check wiring and numbers, never to time anything.
* numeric parity (see :func:`describe_plan` and the validation notes in the
  report): batch+iree vs dense+iree agrees to <=4.8e-7 absolute with identical
  NaN/inf/zero-sign masks and no violation of ``rtol=atol=1e-4``; batch+numpy is
  bit-exact against the specialized numpy route; the loss variant agrees to
  <=2e-6 absolute / <=2.3e-7 relative.  The one deviation measured against the
  specialized numpy route at pop 256 (9.5e-4 absolute, one tree of 256) is the
  pre-existing iree-vs-numpy backend class -- the dense+iree route deviates by
  exactly the same amount at exactly the same index (``DENSE_EVAL.md`` section 5),
  so it is not a property of the merge.
* the price of all of it: a NEW population (a GP generation) is a new program
  multiset, hence a new compile.  This route is opt-in and pays off on repeated
  evaluation of a stable or duplicate-rich population, not on a
  generation-per-step loop as-is.

Who to blame for what (reuse, never re-implement)
-------------------------------------------------
Decoding, validation, constant folding and op emission are the frozen engine's:
:func:`evogp.tree.evoxc_engine.decode_population` gives the per-tree static
programs and :func:`evogp.tree.evoxc_engine._walk_tokens` is the emitter *and*
the multi-output protocol.  A per-function graph is traced through a Python
closure over its static program -- never with the program as a traced static arg,
which would make ``graph.static_values`` differ per tree and make
``batch_functions`` raise ``TraceError``.

Bands and switch_ops
--------------------
``bands`` splits the population into contiguous bands and merges each band
separately (one ``select`` cascade per band instead of one per population), which
trades dispatches for wasted work.  The knob here is a band SIZE
(``EVOGP_BATCH_BAND_SIZE``, default 32): ``K = ceil(n_functions / band_size)``,
clamped to ``1 <= K <= n_functions``; ``K == 1`` reduces to the flat merge.
``EVOGP_BATCH_BANDS`` overrides the model with an explicit ``K`` (or the flat
spellings ``none`` / ``flat`` / ``0``).  ``bands="auto"`` is deliberately NOT
offered: it is CPU-calibrated, resolves to ``K=1`` on this workload and only
costs a static cost model run.

``switch_ops`` (``EVOGP_BATCH_SWITCH_OPS``, default off) makes op-switching
positions share one op per variant plus a per-row ``select`` cascade.  It is a
heavier transform for GP populations (structurally divergent programs) and on the
measured workload it changed nothing at all -- ``report["switched"]`` is empty and
the emitted op count and warm time are those of the flat merge -- so it stays a
knob, off by default.  When it is on, the merge is FORCED to the flat form (``K = 1``):
``switch_ops`` and banding are not combined, and the plan records
``band_count == 1`` whenever ``switch_ops`` is set.

On-disk artifacts: deliberately NOT wired
-----------------------------------------
:mod:`evogp.tree.artifact_cache` keys an artifact by ``(name, specs, backend,
device, options)`` where ``specs`` is the ``build_cached`` spec tuple -- tensor
specs *plus* the static content that specializes the graph.  A merged batch graph
has NO static values (the programs live in Python closures), so the tensor specs
alone would collide across every population of a shape and a stale artifact would
serve wrong numbers (``signature_matches`` cannot catch it: the shapes agree).
The programs could be smuggled into the key as extra static content, but then
every generation's multiset is a distinct key, i.e. no reuse at all.  So this
route builds in memory only; a process that repeats a multiset keeps it in
:data:`BATCH_EXEC_MEMO` for the lifetime of the process.

Front tables (why a repeated call is cheap)
-------------------------------------------
:data:`BATCH_PLAN_MEMO` maps the three population identity tokens
(:func:`evogp.tree.dense_memo.token`: ``data_ptr``/shape/dtype/device/``_version``,
so an in-place mutation is a miss by construction) plus every resolved plan
parameter to the :class:`BatchExecutable` (plan + compiled executable): a repeated
evaluation of an unchanged population does no re-decode, no re-trace, no
re-merge, no re-key and (beyond the one ``x`` conversion every call needs) no
host crossing.  :data:`BATCH_EXEC_MEMO` keys the same executable on the ACTUAL
program multiset plus the shape/backend/device/options, so the same programs in a
fresh population object still hit.  Both tables are dropped by
:func:`evogp.tree.evoxc_engine.clear_cache` (the engine's single invalidation
point) and counted by :func:`batch_cache_stats` -- a separate function, because
``cache_stats()`` (8 keys) and ``dense_cache_stats()`` are pinned by the suite.

Reuse across populations (the re-binding contract)
--------------------------------------------------
:data:`BATCH_EXEC_MEMO` is keyed by the compiled body only -- the merged function
multiset plus the variant, the merge shape, the I/O shape and the
backend/device/options (see :func:`_executable_key`) -- and deliberately NOT by
the population that first built it, because one compiled merged graph really does
serve every population with that same function multiset: the function axis IS the
graph's leading axis, so which population row maps to which function is the
caller's business, not the graph's.  That makes the population size, the row
mapping and the staged inputs *per-call* data, so an executable hit is
**re-bound**: the returned :class:`BatchExecutable` pairs the memoized
``executable`` (and the ``backend`` / ``device`` it was built for) with the plan
the CURRENT call built (or hit in :data:`BATCH_PLAN_MEMO`), never with the first
population's ``pop_size``, ``row_index``, ``function_rows`` or ``programs``.

The rebinding is structural, not incidental: :func:`build_batch` is the ONLY
reader of :data:`BATCH_EXEC_MEMO` and it always constructs a fresh
:class:`BatchExecutable` around the caller's plan, so every per-population field
on the running path (:func:`_inputs` staging, :func:`_map_rows` row expansion,
the ``(pop_size, ...)`` / ``(pop_size,)`` output shapes and the ``pop_size`` row
check) is read from that plan.  :func:`_check_executable_matches` is the cheap
safety net for the one thing the key cannot re-derive: the compiled body's
recorded tensor specs (merged function count, rows, input width, outputs) must
fit the caller's plan, and a mismatch is a :class:`BatchUnsupported` decline --
never a silently wrong row mapping.

No variant is excluded from that reuse: ``per_row=True`` puts the function count
in the merged INPUT spec's leading dimension and ``per_row=False`` puts it in the
OUTPUT spec's, but both are part of the key, so a hit is reusable in both layouts
(and the invariant check above verifies exactly those dimensions).
"""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import evoxc
from evoxc.transforms import batching as _batching

from . import dense_memo
from .evoxc_engine import (
    BACKEND_ENV,
    SUPPORTED_BACKENDS,
    _CPU,
    _bits_to_float,
    _env_int,
    _freeze_options,
    _gpu_device_kind,
    _quiet,
    _register_cache_reset,
    _resolve_backend,
    _trace_const_col,
    _walk_tokens,
    decode_population,
)

__all__ = [
    "BAND_SIZE_ENV",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_BAND_SIZE",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_MIN_POP",
    "EVAL_MODE",
    "LOSS_VARIANT",
    "MIN_POP_ENV",
    "PLAN_VARIANTS",
    "VALUE_VARIANT",
    "BatchExecutable",
    "BatchPlan",
    "BatchUnsupported",
    "batch_cache_stats",
    "build_batch",
    "describe_plan",
    "evaluate_batch",
    "plan_batch",
    "resolve_alignment",
    "resolve_band_count",
    "resolve_band_size",
    "resolve_bands",
    "resolve_min_pop",
    "resolve_switch_ops",
    "sr_fitness_batch",
]

# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
#: The ``EVOGP_EVAL_MODE`` spelling of this route.  Deliberately NOT consulted by
#: this module (see the module docstring): the wiring task adds it to
#: :data:`evogp.tree.dense_engine.EVAL_MODES` and calls the two entry points here.
EVAL_MODE = "batch"
#: Band SIZE of the merge (``EVOGP_BATCH_BAND_SIZE``): ``K = ceil(n / size)``
#: bands, clamped to ``1 <= K <= n``.  32 measured as a sound default: it keeps
#: the per-band variant count low without splitting a small population into
#: bands that each re-pay their own input slice and output concatenate.
BAND_SIZE_ENV = "EVOGP_BATCH_BAND_SIZE"
DEFAULT_BAND_SIZE = 32
#: Explicit band count (``EVOGP_BATCH_BANDS``): an int ``K`` with
#: ``1 <= K <= n_functions``, or one of the flat spellings ``none`` / ``flat`` /
#: ``0``.  Unset/empty means "use the band-size model above".
BANDS_ENV = "EVOGP_BATCH_BANDS"
#: Op-switching (``EVOGP_BATCH_SWITCH_OPS``): off by default -- on the measured GP
#: workload it shares nothing (``report["switched"]`` empty) and changes no op
#: count, so it stays a knob; when on it forces the flat merge.
SWITCH_OPS_ENV = "EVOGP_BATCH_SWITCH_OPS"
#: Class alignment of a ``switch_ops`` merge (``EVOGP_BATCH_ALIGNMENT``); a pure
#: performance heuristic inside evoxc -- it can never change the numbers.
ALIGNMENT_ENV = "EVOGP_BATCH_ALIGNMENT"
DEFAULT_ALIGNMENT = "variant"
ALIGNMENTS = ("variant", "occurrence", "auto")
#: Population threshold (``EVOGP_BATCH_MIN_POP``): below it the merge (a compile
#: is linear in the function count, and the whole route's win is amortizing ONE
#: invocation) is not worth attempting, so the route declines.
MIN_POP_ENV = "EVOGP_BATCH_MIN_POP"
DEFAULT_MIN_POP = 4
#: Entries of the two front tables (``EVOGP_BATCH_CACHE_SIZE``, read at IMPORT
#: time like ``EVOGP_EVOXC_CACHE_SIZE``).  Small on purpose: a plan entry holds
#: the traced per-function graphs, the merged graph AND the compiled executable,
#: and a training loop presents a new population every generation.
CACHE_SIZE_ENV = "EVOGP_BATCH_CACHE_SIZE"
DEFAULT_CACHE_SIZE = 8
#: The two plan/graph variants: the value graph (``(pop, N, out_len)``) and the
#: SR-loss graph (one scalar per function, reduced IN-GRAPH).
VALUE_VARIANT = "value"
LOSS_VARIANT = "loss"
PLAN_VARIANTS = (VALUE_VARIANT, LOSS_VARIANT)
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_FLAT = ("none", "flat", "0")


# ---------------------------------------------------------------------------
# declines
# ---------------------------------------------------------------------------
class BatchUnsupported(RuntimeError):
    """The batch route cannot (safely) evaluate this population.

    Callers must fall back to the specialized engine
    (:func:`evogp.tree.evoxc_engine.decode_population` + ``evaluate`` /
    ``sr_fitness``), which owns the error contract for malformed trees.  The
    message carries the reason; the enumerated causes are listed in the module
    docstring ("Declines and the fallback rule"), and the two evoxc exception
    types a trace/merge can raise (``TraceError`` / ``TransformError``) are
    always converted into this one instead of escaping.
    """


# ---------------------------------------------------------------------------
# knob resolution (per call unless noted; the explicit argument always wins)
# ---------------------------------------------------------------------------
def _resolve_int(value: Any, env: str, default: int, minimum: int) -> int:
    """``explicit argument > environment > default`` for a bounded integer knob."""
    resolved = _env_int(env, default) if value is None else int(value)
    if resolved < minimum:
        raise ValueError(f"{env}={resolved} must be >= {minimum}")
    return resolved


def resolve_band_size(value: Optional[int] = None) -> int:
    """Resolve ``EVOGP_BATCH_BAND_SIZE`` (explicit argument wins).

    The band SIZE model turns a function count into ``K = ceil(n / size)`` bands.
    Read per call; anything below 1 raises ``ValueError`` naming the variable.
    """
    return _resolve_int(value, BAND_SIZE_ENV, DEFAULT_BAND_SIZE, 1)


def resolve_bands(value: Optional[Any] = None) -> Optional[int]:
    """Resolve ``EVOGP_BATCH_BANDS`` into an explicit band count or ``None``.

    ``None`` means "the band-size model decides" -- the answer for unset/empty
    AND for the explicit flat spellings (``none`` / ``flat`` / ``0``), which both
    describe a single-band (flat) merge.  An integer ``K`` is returned as given;
    :func:`resolve_band_count` validates it against the function count (which is
    only known after decoding).  Anything else raises ``ValueError`` naming the
    variable.
    """
    raw: Any = value if value is not None else os.environ.get(BANDS_ENV, "")
    if isinstance(raw, bool):  # bool is an int: reject instead of guessing
        raise ValueError(
            f"unsupported bands value {raw!r}: {BANDS_ENV} takes an integer band "
            f"count or one of {list(_FLAT)}"
        )
    if isinstance(raw, (int, np.integer)):
        count = int(raw)
        return None if count == 0 else count
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in _FLAT:
        return None
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError(
            f"unsupported {BANDS_ENV}={raw!r}: expected an integer band count "
            f"(1 <= K <= n_functions) or one of {list(_FLAT)}"
        ) from exc
    return None if count == 0 else count


def resolve_switch_ops(value: Optional[bool] = None) -> bool:
    """Resolve ``EVOGP_BATCH_SWITCH_OPS`` (explicit argument wins; default off)."""
    if value is not None:
        return bool(value)
    raw = os.environ.get(SWITCH_OPS_ENV, "").strip().lower()
    if not raw:
        return False
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(
        f"unsupported {SWITCH_OPS_ENV}={raw!r}: batch_engine accepts "
        f"{list(_TRUE + _FALSE)} or an empty value"
    )


def resolve_alignment(value: Optional[str] = None) -> str:
    """Resolve ``EVOGP_BATCH_ALIGNMENT`` (explicit argument wins).

    Only meaningful with ``switch_ops`` (non-switched positions have one variant
    per skeleton by construction); validated always, so a typo is caught even on
    the default path.  Case-insensitive; ``variant`` / ``occurrence`` / ``auto``.
    """
    raw = value if value is not None else os.environ.get(ALIGNMENT_ENV, "")
    text = str(raw).strip().lower()
    if not text:
        text = DEFAULT_ALIGNMENT
    if text not in ALIGNMENTS:
        raise ValueError(
            f"unsupported alignment {text!r}: batch_engine supports "
            f"{list(ALIGNMENTS)} (set {ALIGNMENT_ENV!r} or pass alignment=)"
        )
    return text


def resolve_min_pop(value: Optional[int] = None) -> int:
    """Resolve the population threshold ``EVOGP_BATCH_MIN_POP`` (argument wins)."""
    return _resolve_int(value, MIN_POP_ENV, DEFAULT_MIN_POP, 1)


def resolve_band_count(
    n_functions: int,
    *,
    bands: Optional[Any] = None,
    band_size: Optional[int] = None,
    switch_ops: Optional[bool] = None,
) -> int:
    """The band count ``K`` this merge should use for ``n_functions`` functions.

    Precedence: ``switch_ops`` on -> ``K = 1`` (the flat merge; banding is never
    combined with op-switching), then an explicit ``bands`` count, then the
    band-SIZE model ``ceil(n_functions / band_size)``.  The result is always
    clamped into ``1 <= K <= n_functions``; an explicit ``K`` outside that range
    raises ``ValueError`` naming ``EVOGP_BATCH_BANDS`` (it cannot be honoured, so
    silently clamping would hide a typo).
    """
    total = int(n_functions)
    if total < 1:
        raise ValueError(f"n_functions={total} must be >= 1")
    if resolve_switch_ops(switch_ops):
        return 1
    explicit = resolve_bands(bands)
    if explicit is not None:
        if not 1 <= explicit <= total:
            raise ValueError(
                f"{BANDS_ENV}={explicit} is outside 1..{total} (the number of "
                "merged functions)"
            )
        return explicit
    size = resolve_band_size(band_size)
    return max(1, min(total, -(-total // size)))


# ---------------------------------------------------------------------------
# front tables
# ---------------------------------------------------------------------------
def _default_cache_size() -> int:
    size = _env_int(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE)
    if size < 0:
        raise ValueError(f"{CACHE_SIZE_ENV}={size} must be >= 0")
    return size


#: Compiled merged executables, keyed by the actual program multiset + shape +
#: backend/device/options (see :func:`_executable_key`).  The key holds the REAL
#: programs, never a digest: a hash collision would serve wrong numbers.  The
#: value's ``plan`` belongs to the population that FIRST built it, so it must
#: never be handed to another population -- :func:`build_batch` re-binds a hit to
#: the caller's plan (see the module docstring).
BATCH_EXEC_MEMO = dense_memo.DenseMemo("batch_exec", _default_cache_size())
#: :class:`BatchExecutable` (plan + executable) of an unchanged population, keyed
#: by the three population identity tokens + every resolved plan parameter.
BATCH_PLAN_MEMO = dense_memo.DenseMemo("batch_plan", _default_cache_size())


def _clear_tables() -> None:
    """Drop both front tables (registered with the engine's invalidation point)."""
    BATCH_EXEC_MEMO.reset()
    BATCH_PLAN_MEMO.reset()


#: :func:`evogp.tree.evoxc_engine.clear_cache` drops these tables too -- the
#: engine stays the single in-memory invalidation point.  Registered as a hook
#: (not via ``dense_memo.register``) so ``dense_cache_stats()`` keeps returning
#: exactly the dense route's keys.
_register_cache_reset(_clear_tables)


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


# ---------------------------------------------------------------------------
# host helpers
# ---------------------------------------------------------------------------
def _pop_size(tensor: Any) -> Optional[int]:
    """``(pop_size, max_tree_len)`` row count, or ``None`` when unreadable.

    Read from ``shape`` only (no host copy); ``None`` means "let
    ``decode_population`` raise its documented error", so an exotic or malformed
    population tensor is never turned into a silent decline.
    """
    shape = getattr(tensor, "shape", None)
    try:
        if shape is None or len(shape) != 2:
            return None
        return int(shape[0])
    except Exception:  # pragma: no cover - exotic tensor subclasses only
        return None


def _as_host(tensor: Any, name: str) -> np.ndarray:
    """Read any tensor (torch on any device / numpy) as a float32 host array."""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(tensor)
        except Exception as exc:  # pragma: no cover - exotic inputs only
            raise BatchUnsupported(
                f"batch_engine: {name} is not convertible to a host array "
                f"({type(tensor).__name__})"
            ) from exc
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _host_rows(tensor: Any, input_len: int, name: str, *, per_row: bool,
               pop_size: Optional[int]) -> np.ndarray:
    """Validate/convert a ``(rows, input_len)`` float32 input; never mutates it.

    Every shape doubt is a :class:`BatchUnsupported` decline (never a
    ``ValueError``): the fallback engine raises the documented errors, and a route
    that declares "this is not mine" must not pre-empt that contract.
    """
    array = _as_host(tensor, name)
    if array.ndim != 2:
        raise BatchUnsupported(
            f"batch_engine: {name} must be 2-D (rows, {int(input_len)}), got shape "
            f"{array.shape}"
        )
    if array.shape[1] != int(input_len):
        raise BatchUnsupported(
            f"batch_engine: {name} has {array.shape[1]} columns but the "
            f"population expects input_len={int(input_len)}"
        )
    if per_row and pop_size is not None and array.shape[0] != pop_size:
        raise BatchUnsupported(
            f"batch_engine: per_row=True needs {name}.shape[0] == {pop_size} "
            f"(pop_size), got {array.shape[0]}"
        )
    return array


def _host_labels(labels: Any, n_rows: int, output_len: int) -> np.ndarray:
    """Validate/convert the ``(n_rows, output_len)`` SR labels."""
    array = _as_host(labels, "labels")
    if array.shape != (int(n_rows), int(output_len)):
        raise BatchUnsupported(
            f"batch_engine: labels must have shape ({int(n_rows)}, "
            f"{int(output_len)}), got {array.shape}"
        )
    return array


# ---------------------------------------------------------------------------
# per-function graph bodies (the engine's own emitters -- parity by construction)
# ---------------------------------------------------------------------------
def _constant_output(const_out: Sequence[int], n_rows: int) -> Any:
    """The ``(n_rows, 1)`` constant column of a fully constant tree.

    A constant tree still has to RETURN a tensor of the same rank/shape as every
    other function's output (a bare 0-d ``evoxc.constant`` would be a rank
    mismatch and a static output leaf, which the merge rejects).  The column is
    built the way the specialized engine assembles a constant output
    (:func:`evogp.tree.evoxc_engine._trace_const_col`), i.e. as a real
    ``(n_rows, 1)`` payload; the merge stacks differing payloads into one stacked
    constant (``report["stacked_constants"]``), so it stays exact.
    """
    if len(const_out) == 1:
        return _trace_const_col(n_rows, _bits_to_float(const_out[0]))
    cols = [_trace_const_col(n_rows, _bits_to_float(bits)) for bits in const_out]
    return evoxc.concatenate(cols, axis=1)


def _predict(program: Any, const_out: Any, x: Any, n_rows: int) -> Any:
    """The ``(n_rows, out_len)`` symbolic output of one decoded tree."""
    if program is not None:
        return _walk_tokens(program[1], int(program[0]), x)
    return _constant_output(const_out, n_rows)


def _value_fn(program: Any, const_out: Any, n_rows: int):
    """One value function: ``x -> (n_rows, out_len)`` (a closure over the program).

    The program travels in the CLOSURE, never as a ``trace`` static argument:
    static values are part of the graph's identity, so a static program would give
    every function different ``graph.static_values`` and
    ``evoxc.batch_functions`` would raise ``TraceError`` on the merge.
    """

    def fn(x):
        return _predict(program, const_out, x, n_rows)

    return fn


def _loss_fn(program: Any, const_out: Any, n_rows: int, use_mse: bool):
    """One SR-loss function: ``(x, label) -> (1,)``-free RANK-0 scalar loss.

    The whole fitness reduction is IN-GRAPH, so the merged output is one scalar
    per function (``(n_functions,)``) and only that crosses back::

        err  = (pred - label) ** 2      if use_mse
        err  = |pred - label|           otherwise
        loss = sum_N sum_out err / N

    Outputs are summed and never divided by ``output_len`` (the SR contract), and
    the ``/ N`` divide is a static float32 constant inside the graph.  Rank-0
    per-function outputs turned out to work under ``out_axes=0`` (the merged result
    is ``(n_functions,)`` directly), so no host-side squeeze is needed; the
    reduction is done with :func:`evoxc.reduce_sum` over both axes, whose batching
    rule is registered.
    """

    def fn(x, label):
        pred = _predict(program, const_out, x, n_rows)
        diff = evoxc.subtract(pred, label)
        err = evoxc.multiply(diff, diff) if use_mse else evoxc.abs(diff)
        total = evoxc.reduce_sum(err, axes=(0, 1))
        return evoxc.divide(total, _scalar(n_rows))

    return fn


def _scalar(value: float) -> Any:
    """A rank-0 float32 symbolic constant (``evoxc`` rejects numpy scalars).

    Must be called inside a trace; a rank-0 payload is shared by every merged
    function (the divide's divisor is the same constant for all of them).
    """
    return evoxc.constant(
        evoxc.core.tensor(np.asarray(np.float32(value), dtype=np.float32))
    )


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchPlan:
    """The host-side plan of one merged batch graph (no executable, no cache).

    Attributes:
        input_len: problem input width.
        output_len: outputs per tree.
        pop_size: population size (the number of rows the plan answers for).
        variant: :data:`VALUE_VARIANT` or :data:`LOSS_VARIANT`.
        use_mse: the loss variant's error metric (ignored by the value variant).
        per_row: ``True`` when every population row has its own ``x`` row.
        n_rows: rows each merged function sees (``1`` for ``per_row``, else ``N``
            for the value variant and the data-point count for the loss variant).
        programs: per population row, the ``(program, const_out)`` identity of that
            row -- ``program`` is the frozen static program (``None`` for a fully
            constant tree, whose answer is then ``const_out``).
        functions: the DEDUPED ``(program, const_out)`` multiset actually merged, in
            first-appearance order: ``functions[row_index[row]] == programs[row]``.
        row_index: per population row, its index into :attr:`functions`.
        function_rows: per merged function, the population row it was first seen
            at (the ``x`` row a ``per_row`` plan feeds it).
        band_count: number of merge bands (``1`` = the flat merge).
        switch_ops: whether op-switching was requested (forces ``band_count == 1``).
        alignment: the ``switch_ops`` class-alignment mode.
        graphs: the traced per-function :class:`evoxc.Graph` objects.
        merged: the merged :class:`evoxc.Graph` (``merged.batch_report`` describes it).
        specs: the merged graph's tensor input specs (``evoxc.TensorSpec`` objects).
        report: ``merged.batch_report.as_dict()`` -- ``n_functions``,
            ``emitted_ops``, ``bands``, ``band_sizes``, ``band_strategy``, ...
        backend_name: the backend the plan was resolved for (the default of
            :func:`build_batch`).
        device: the resolved ``evoxc.core.Device``.
    """

    input_len: int
    output_len: int
    pop_size: int
    variant: str
    use_mse: bool
    per_row: bool
    n_rows: int
    programs: Tuple[Any, ...]
    functions: Tuple[Any, ...]
    row_index: Tuple[int, ...]
    function_rows: Tuple[int, ...]
    band_count: int
    switch_ops: bool
    alignment: str
    graphs: Tuple[Any, ...]
    merged: Any
    specs: Tuple[Any, ...]
    report: Dict[str, Any]
    backend_name: str
    device: Any


@dataclass(frozen=True)
class BatchExecutable:
    """A plan plus its compiled executable (the value of both front tables).

    Attributes:
        plan: the :class:`BatchPlan` this executable was built for.
        executable: the loaded ``evoxc.Executable`` of the merged graph.
        backend: the backend name it was compiled for.
        device: the ``evoxc.core.Device`` it was loaded on.
        pins: strong references to the population/input storages the plan memo entry
            is keyed on (empty for an unmemorized build); they only keep the
            ``data_ptr`` identity tokens ABA-proof, so they are never inspected.
    """

    plan: BatchPlan
    executable: Any
    backend: str
    device: Any
    pins: Tuple[Any, ...] = ()


def _normalize_variant(variant: Any, per_row: bool) -> str:
    """The variant rules this module enforces (checked before any host work)."""
    text = str(variant).strip().lower()
    if text not in PLAN_VARIANTS:
        raise ValueError(
            f"unsupported plan variant {variant!r}: batch_engine supports "
            f"{list(PLAN_VARIANTS)}"
        )
    if text == LOSS_VARIANT and per_row:
        raise ValueError(
            "the loss variant implements sr_fitness only (per_row=False): a "
            "per_row=True loss would need one label row per tree, which is not "
            "the SR contract"
        )
    return text


def _dedup(pop: Any, x_host: np.ndarray, per_row: bool) -> Tuple[
    Tuple[Any, ...], Tuple[int, ...], Tuple[int, ...]
]:
    """Collapse identical rows: ``(functions, row_index, function_rows)``.

    Structural identity IS program equality (there is no library-level dedup
    helper, and a re-decode of an unchanged tree yields an equal program tuple), so
    duplicates and elites -- the common case -- merge into one function.

    ``per_row=True`` additionally keys on the row's own ``x`` bytes: tree ``i`` must
    see row ``i``, so two rows that share a program but not their input row are
    genuinely different functions (sharing them would serve wrong numbers).
    Constant trees are keyed by their ``const_out`` alone -- their answer does not
    depend on ``x`` at all.
    """
    functions: List[Any] = []
    positions: Dict[Any, int] = {}
    row_index: List[int] = []
    function_rows: List[int] = []
    for row, tree in enumerate(pop.trees):
        if tree.program is not None:
            key: Any = tree.program
            if per_row:
                key = (tree.program, x_host[row].tobytes())
        else:
            key = ("const", tree.const_out)
        index = positions.get(key)
        if index is None:
            index = len(functions)
            positions[key] = index
            functions.append((tree.program, tree.const_out))
            function_rows.append(row)
        row_index.append(index)
    return tuple(functions), tuple(row_index), tuple(function_rows)


def _trace(fn: Any, specs: Sequence[Any], what: str) -> Any:
    """``evoxc.trace`` with the two structural exception types converted."""
    try:
        return evoxc.trace(fn, *specs)
    except (evoxc.TraceError, evoxc.TransformError) as exc:
        raise BatchUnsupported(
            f"batch_engine: tracing the {what} graph failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _merge(graphs: Sequence[Any], axes: Any, *, switch_ops: bool,
           bands: Optional[int], alignment: str) -> Any:
    """``evoxc.batch_functions`` with the two structural exception types converted."""
    try:
        merged = evoxc.batch_functions(
            list(graphs),
            axes=axes,
            out_axes=0,
            switch_ops=True if switch_ops else None,
            bands=bands,
            alignment=alignment,
        )
    except (evoxc.TraceError, evoxc.TransformError) as exc:
        raise BatchUnsupported(
            f"batch_engine: merging {len(graphs)} per-function graphs failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _require_batching_rules(merged)
    return merged


def _require_batching_rules(merged: Any) -> None:
    """Decline when the merged graph contains an op without a batching rule.

    The merge itself already uses every rule it needs (and a missing one is
    converted by :func:`_merge`), so this is the explicit guard the route contract
    documents: walking the merged entry block keeps a future emitter honest and
    names the offending op instead of surfacing a ``TransformError`` from a later
    stage.  The ``return`` terminator is skipped (it is not an op with semantics
    and has no rule by design).
    """
    ops = getattr(getattr(getattr(merged, "module", None), "main", None),
                  "entry_block", None)
    if ops is None:  # pragma: no cover - evoxc always exposes the entry block
        raise BatchUnsupported("batch_engine: merged graph has no entry block")
    rule_of = _batching.get_batching_rule
    for op in list(ops.ops):
        name = op.name
        if name == "return":
            continue
        if name == "block_call" or name.startswith("external"):
            raise BatchUnsupported(
                f"batch_engine: the merged graph contains an op without a "
                f"batching rule: {name!r}"
            )
        if rule_of(name) is None:
            raise BatchUnsupported(
                f"batch_engine: the merged graph contains an op without a "
                f"batching rule: {name!r}"
            )


def _plan_host(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x_host: np.ndarray,
    label_host: Optional[np.ndarray],
    *,
    per_row: bool,
    variant: str,
    use_mse: bool,
    band_size: Optional[int],
    bands: Optional[Any],
    switch_ops: Optional[bool],
    alignment: Optional[str],
    backend_name: str,
    device: Any,
) -> BatchPlan:
    """The whole host planning step on already-converted inputs (see plan_batch)."""
    input_len = int(input_len)
    output_len = int(output_len)
    pop = decode_population(
        batch_node_value, batch_node_type, batch_subtree_size,
        input_len, output_len,
    )
    pop_size = int(pop.pop_size)
    per_row = bool(per_row)
    variant = _normalize_variant(variant, per_row)
    alignment = resolve_alignment(alignment)
    switch_ops = resolve_switch_ops(switch_ops)
    n_rows = 1 if per_row else int(x_host.shape[0])
    if variant == LOSS_VARIANT:
        if n_rows <= 0:
            raise BatchUnsupported(
                "batch_engine: sr_fitness needs at least one data point, got 0 rows"
            )
        if label_host is None:
            raise BatchUnsupported("batch_engine: the loss variant needs labels")
    functions, row_index, function_rows = _dedup(pop, x_host, per_row)
    band_count = resolve_band_count(
        len(functions), bands=bands, band_size=band_size, switch_ops=switch_ops
    )
    spec = evoxc.TensorSpec((n_rows, input_len), evoxc.float32)
    label_spec = evoxc.TensorSpec((n_rows, output_len), evoxc.float32)
    graphs: List[Any] = []
    for program, const_out in functions:
        if variant == LOSS_VARIANT:
            fn = _loss_fn(program, const_out, n_rows, bool(use_mse))
            graphs.append(_trace(fn, (spec, label_spec), "loss"))
        else:
            fn = _value_fn(program, const_out, n_rows)
            graphs.append(_trace(fn, (spec,), "value"))
    axes = 0 if per_row else (None,) * (2 if variant == LOSS_VARIANT else 1)
    merged = _merge(
        graphs,
        axes,
        switch_ops=switch_ops,
        bands=None if band_count <= 1 else band_count,
        alignment=alignment,
    )
    report = merged.batch_report.as_dict()
    return BatchPlan(
        input_len=input_len,
        output_len=output_len,
        pop_size=pop_size,
        variant=variant,
        use_mse=bool(use_mse),
        per_row=per_row,
        n_rows=n_rows,
        programs=tuple((tree.program, tree.const_out) for tree in pop.trees),
        functions=functions,
        row_index=row_index,
        function_rows=function_rows,
        band_count=int(report.get("bands", band_count) or 1),
        switch_ops=bool(switch_ops),
        alignment=alignment,
        graphs=tuple(graphs),
        merged=merged,
        specs=tuple(merged.tensor_specs),
        report=report,
        backend_name=backend_name,
        device=device,
    )


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
def _executable_key(
    plan: BatchPlan, backend_name: str, device: Any, options: Dict[str, Any]
) -> Tuple[Any, ...]:
    """Memo key of one merged executable.

    Everything that determines the compiled body: the ACTUAL program multiset
    (never a digest -- a collision would serve wrong numbers), the variant, the
    loss metric, the merge shape (band count, ``switch_ops``, alignment, rows), the
    I/O shape and the backend/device/options.  Two populations with the same
    programs therefore share the executable, and any of these changing rebuilds.
    """
    return (
        tuple(plan.functions),
        plan.variant,
        bool(plan.use_mse),
        int(plan.band_count),
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


def _spec_shapes(specs: Sequence[Any]) -> Tuple[Tuple[Tuple[int, ...], str], ...]:
    """``((shape, dtype), ...)`` of a sequence of ``evoxc.TensorSpec`` objects."""
    return tuple(
        (tuple(int(dim) for dim in spec.shape), str(spec.dtype)) for spec in specs
    )


def _check_executable_matches(plan: BatchPlan, executable: Any) -> None:
    """Decline when a memoized executable does not fit the CALLER's plan.

    The executable key (:func:`_executable_key`) already pins the compiled body's
    inputs -- program multiset, variant/loss metric, merge shape, rows, input
    width, outputs, backend/device -- so this check is an invariant, not the
    mechanism: it re-derives the two things the caller is about to rely on from
    the executable's own ``evoxc`` signature (:attr:`evoxc.Executable.signature`,
    recorded at ``lower`` time) and compares them with the plan:

    * the merged INPUT specs must equal the plan's ``evoxc.TensorSpec`` objects
      (shape + dtype) -- on ``per_row=True`` that leading dimension is the merged
      function count;
    * the merged OUTPUT spec must be ``(n_functions, n_rows, output_len)`` for the
      value variant and ``(n_functions,)`` for the loss variant, i.e. the merged
      function axis plus the rows/outputs the plan stages.

    A mismatch means the table served a body built for a different function
    multiset or shape; that is a :class:`BatchUnsupported` decline (the caller
    falls back to the specialized engine) rather than a silently wrong row
    mapping.  Nothing is checked when the executable records no signature at all
    (an executable we cannot introspect is still the one the key selected).
    """
    signature = getattr(executable, "signature", None)
    if signature is None:  # pragma: no cover - evoxc always records a signature
        return
    actual_in = _spec_shapes(getattr(signature, "input_specs", ()) or ())
    expected_in = _spec_shapes(plan.specs)
    if actual_in and actual_in != expected_in:
        raise BatchUnsupported(
            f"batch_engine: the memoized executable was compiled for inputs "
            f"{actual_in}, but this population's plan needs {expected_in}"
        )
    n_functions = len(plan.functions)
    if plan.variant == LOSS_VARIANT:
        expected_out: Tuple[int, ...] = (n_functions,)
    else:
        expected_out = (n_functions, int(plan.n_rows), int(plan.output_len))
    actual_out = tuple(shape for shape, _ in _spec_shapes(
        getattr(signature, "output_specs", ()) or ()))
    if len(actual_out) == 1 and actual_out[0] != expected_out:
        raise BatchUnsupported(
            f"batch_engine: the memoized executable has {len(plan.functions)} "
            f"merged function(s) and was compiled for output "
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

    Contract of :data:`BATCH_EXEC_MEMO` (the only table this function reads): an
    executable is keyed by the merged function multiset and shapes -- a HIT is
    REBOUND to the current population's plan, never returned with another
    population's row mapping.  Only the compiled ``executable`` (and the backend /
    device it was built for) crosses between populations; the returned
    :class:`BatchExecutable` always carries ``plan``, so ``pop_size``,
    ``row_index``, ``function_rows``, ``programs`` and the staged specs are this
    call's.  A repeated evaluation of an unchanged population therefore still does
    no re-lower, re-compile or re-load, and a same-multiset population of a
    different size shares the compile WITHOUT sharing its staging.

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
            failure is not a decline -- there is nothing safer to fall back to).
    """
    backend_name, device, options = _resolve_route_backend(
        backend if backend is not None else plan.backend_name,
        iree_device_index,
    )
    key = _executable_key(plan, backend_name, device, options)
    cached = BATCH_EXEC_MEMO.lookup(key)
    if cached is not None:
        # RE-BIND: the compiled body is shared, the caller's plan is not.  Only
        # ``executable`` / ``backend`` / ``device`` may travel across populations
        # (and ``backend``/``device`` are equal by key); the returned value is a
        # FRESH BatchExecutable around the CURRENT plan, so no downstream read of
        # ``.plan`` can see the first population's row mapping.  ``pins`` stays
        # empty: pins belong to the plan-memo entry that holds the population
        # storages (``_plan_entry`` attaches this call's pins), and a
        # cross-population pin would only keep a foreign buffer alive.
        _check_executable_matches(plan, cached.executable)
        return BatchExecutable(
            plan=plan,
            executable=cached.executable,
            backend=backend_name,
            device=device,
        )
    lowered = evoxc.lower(plan.merged, backend=backend_name)
    if backend_name == "iree":
        artifact = evoxc.compile(
            lowered,
            backend=backend_name,
            target_backends=[_gpu_device_kind()],
            **options,
        )
    else:
        artifact = evoxc.compile(lowered, backend=backend_name)
    executable = evoxc.load(artifact, backend=backend_name, device=device)
    built = BatchExecutable(
        plan=plan, executable=executable, backend=backend_name, device=device
    )
    BATCH_EXEC_MEMO.store(key, built)
    return built


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
) -> BatchExecutable:
    """The memoized plan+build pipeline behind both public entry points.

    The plan table is keyed on the three population identity tokens and every
    resolved plan parameter; the band COUNT is deliberately not in the key (it
    follows from the knob values plus the decoded population, and decoding only
    happens on a miss).  ``x`` joins the key for ``per_row=True``, where the dedup
    grouping depends on the per-row input bytes.

    A hit re-uses the executable by reference and needs no host work at all; a
    miss pays one ``decode_population``, the dedup, one trace per distinct program,
    the merge, and one compile (unless :data:`BATCH_EXEC_MEMO` already holds that
    program multiset for this shape -- in which case the compile is skipped and
    :func:`build_batch` re-binds the memoized executable to the plan built here).

    Both hits are safe by construction: a plan-table key holds the population's
    identity tokens (``data_ptr`` + shape + dtype + device + ``_version``, pinned
    so the address cannot be recycled), so its entry's plan WAS decoded from this
    very population -- a plan entry can never carry another population's row
    mapping, while an executable entry always can and is re-bound above.
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
        )
        entry = BATCH_PLAN_MEMO.lookup(key)
        if entry is not None:
            return entry
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
    )
    built = build_batch(plan, backend=backend_name, iree_device_index=iree_device_index)
    if memoized:
        # Strong references to the tokenized storages, held BY the stored value: a
        # live pin keeps the buffer allocated, so a ``data_ptr`` token cannot be
        # recycled by a different tensor while the entry lives (ABA).  The extra
        # field is the only reason the stored object is a ``replace`` copy.
        pinned = [batch_node_value, batch_node_type, batch_subtree_size]
        if per_row:
            pinned.append(x)
        built = dataclasses.replace(built, pins=dense_memo.pin(pinned))
        BATCH_PLAN_MEMO.store(key, built)
    return built


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _inputs(entry: BatchExecutable, x_host: np.ndarray,
            label_host: Optional[np.ndarray]) -> Tuple[Any, ...]:
    """The graph inputs of one call, staged for the entry's backend.

    ``per_row=True`` feeds each merged function its own ``x`` row, so the host
    block is gathered to ``(n_functions, 1, input_len)`` (axes=0 on the function
    axis); every other layout shares one ``(n_rows, input_len)`` block.  On iree
    each block is uploaded once per call (``evoxc.from_numpy`` is a zero-copy host
    wrap; the upload is ``Tensor.to(device)``); the numpy backend consumes the host
    array directly and never uploads.
    """
    plan = entry.plan
    if plan.per_row:
        block = np.ascontiguousarray(
            x_host[list(plan.function_rows)][:, None, :], dtype=np.float32
        )
    else:
        block = np.ascontiguousarray(x_host, dtype=np.float32)
    staged: List[Any] = [block]
    if label_host is not None:
        staged.append(np.ascontiguousarray(label_host, dtype=np.float32))
    if entry.backend == "numpy":
        return tuple(staged)
    return tuple(evoxc.from_numpy(array).to(entry.device) for array in staged)


def _run(entry: BatchExecutable, x_host: np.ndarray,
         label_host: Optional[np.ndarray]) -> np.ndarray:
    """One ``evoxc.run`` of the merged graph; returns the raw host result."""
    with _quiet():
        result = evoxc.run(entry.executable, *_inputs(entry, x_host, label_host))
    if entry.backend == "numpy":
        return np.asarray(result.numpy(), dtype=np.float32)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32)


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
    )
    plan = entry.plan
    x_host = _input_block(entry, x, input_len, per_row)
    rows = _run(entry, x_host, None)
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
    )
    plan = entry.plan
    x_host = _input_block(entry, inputs, input_len, False)
    label_host = _host_labels(labels, x_host.shape[0], plan.output_len)
    rows = _run(entry, x_host, label_host)
    return _to_tensor(np.ascontiguousarray(_map_rows(rows, plan)), inputs, out_device)


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------
def describe_plan(plan: BatchPlan) -> Dict[str, Any]:
    """A cheap, JSON-ish description of ``plan`` (for tests and a future runner).

    Reports the population size and the deduped function count, how many rows
    collide onto an earlier function, the merge shape (band count/strategy/sizes,
    emitted ops, class counters), the graph input shapes and the route
    (backend/device).  It reads the stored report only -- nothing is decoded,
    traced or compiled.
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
        "band_sizes": list(report.get("band_sizes", ())),
        "band_strategy": report.get("band_strategy"),
        "switch_ops": bool(plan.switch_ops),
        "alignment": plan.alignment,
        "emitted_ops": report.get("emitted_ops"),
        "value_classes": report.get("value_classes"),
        "shared_classes": report.get("shared_classes"),
        "stacked_constants": report.get("stacked_constants"),
        "output_muxes": report.get("output_muxes"),
        "n_functions": report.get("n_functions"),
        "specs": shapes,
        "spec_dtypes": [str(spec.dtype) for spec in plan.specs],
        "backend": plan.backend_name,
        "device": str(plan.device),
    }
