"""FUSED whole-population evaluation route: one Triton pass, no host crossings.

This module is the public facade of the ``fused`` route -- the fourth evaluation
engine beside ``specialized`` (:mod:`evogp.tree.evoxc_engine`), ``dense``
(:mod:`evogp.tree.dense_engine`) and the opt-in merged graph
(:mod:`evogp.tree.batch_engine`).  Like ``dense_engine`` it evaluates a WHOLE
population in one launch; unlike every existing route it needs no evoxc graph, no
host decode, no host-side assembly and no readback on the value path.

It is NOT wired into :func:`evogp.tree.dense_engine.resolve_route` yet: this
round ships the engine, and the call sites
(``Tree.forward`` / ``Forest.forward`` / ``Forest.batch_forward`` /
``Forest.SR_fitness``) keep routing to the three existing engines until the
wiring round lands.  ``resolve_min_pop`` already reads the threshold that round
will consume (``EVOGP_FUSED_MIN_POP``, default 64, the same break-even reasoning
as ``EVOGP_DENSE_AUTO_MIN_POP``).

Public surface
--------------
============================== ==================================================
:class:`FusedUnsupported`      per-call decline (route cannot take this call)
:func:`fused_available`        ``triton`` importable AND a torch CUDA/HIP device
:func:`evaluate_fused`         value path, ``(pop, output_len)`` / ``(pop, N, output_len)``
:func:`sr_fitness_fused`       SR loss path, ``(pop,)`` positive loss
:func:`fused_cache_stats`      this route's own counters (all prefixed ``fused_``)
:func:`clear_fused_cache`      reset them (also registered in ``clear_cache``)
============================== ==================================================

Contracts (they mirror :func:`evogp.tree.dense_engine.evaluate_dense` /
``sr_fitness_dense`` one for one, so the wiring round is a dispatch change only):

* ``per_row=True``: ``x`` is ``(pop_size, input_len)`` and tree ``i`` sees row
  ``i``; the result is ``(pop_size, output_len)``.
* ``per_row=False``: ``x`` is ``(N, input_len)`` and every tree is evaluated on
  all ``N`` rows; the result is ``(pop_size, N, output_len)``.
* Both return a FRESH, contiguous, ``requires_grad=False`` float32 tensor on the
  computation device (``out_device`` overrides); the caller's tensors are never
  modified.
* ``sr_fitness_fused`` returns ``(pop_size,)`` float32 POSITIVE loss
  ``sum_points sum_outputs err / N`` (MSE or MAE on ``label - prediction``,
  outputs summed but never divided by ``output_len``, NaN/inf unsanitized).
* A call the route cannot serve raises :class:`FusedUnsupported` with a
  human-readable ``reason`` -- never a wrong number and never a crash.  The
  reasons are: triton not importable; no CUDA/HIP device; the population or
  ``x`` not on a device tensor; an explicit ``backend="numpy"`` (a Triton kernel
  cannot run on the evoxc reference interpreter); ``output_len != 1``
  (multi-output lands later); ``pop_size < 1``; ``rows < 1``; or the scratch
  envelope (``pop * tree_len * rows * 4`` bytes vs ``EVOGP_FUSED_MAX_SCRATCH_MB``)
  being exceeded.  ``FusedUnsupported.device_fallback`` exists for symmetry with
  ``DenseUnsupported`` but the fused route never sets it: it has no numpy twin,
  so the fallback is always the specialized engine.

The one prep step
-----------------
``per_row=False`` wants the data rows CONTIGUOUS along the vector axis, i.e. a
transposed copy ``xt = x.t().contiguous()`` of shape ``(input_len, rows)``, so a
VAR node is one coalesced vector load ``xt[vi * rows + r]``.  That copy is a
device kernel, and a training loop re-evaluates the same dataset thousands of
times, so it is memoized by IDENTITY (:data:`XT_MEMO`, the ``dense_memo`` token
``(data_ptr, shape, dtype, device, _version)``, an in-place write is a miss, the
entry pins the caller's storage so the address cannot be recycled).  Hits and
misses are :func:`fused_cache_stats`'s ``fused_xt_hits`` / ``fused_xt_misses``;
``EVOGP_FUSED_HOST_MEMO`` (default 16 entries, import time) bounds the table and
``0`` disables it.  ``per_row=True`` needs no transpose: each tree reads its own
scalar row element ``x[t * input_len + vi]``.

Measured economics (six benchmark cells, warm medians, one process per cell,
MI300X; the battery reproduces every number)
------------------------------------------------------------------------------------------------------
The value path is **5.6x-57.8x faster** than the ``torch-batched`` pure-torch
arm (``test/benchmark/evoxc/arms_torch.py:BatchedEvaluator``) and the loss path
5.6x-57.8x, because that arm streams a ``(pop, L+1, N)`` buffer once per node
position while the fused kernel keeps the population's whole tree in registers
and touches each node table exactly once.  Cold first-call cost is a Triton JIT
compile per specialization (``PER_ROW`` x ``REDUCE`` x ``USE_MSE`` x block x
warps): ~0.4-1.0 s.  Numbers of record and the ``BLOCK_R``/``num_warps`` sweep
are in :mod:`evogp.tree.fused_check`'s printed table; the knobs are documented in
:mod:`evogp.tree.fused_kernels`.

Environment knobs (read per call unless noted; an unusable value raises
``ValueError`` naming the variable)
-------------------------------------------------------------------
* ``EVOGP_FUSED_BLOCK_ROWS`` (256) -- cap of the row-block width; the launch uses
  ``min(next_pow2(rows), cap)`` lanes, so a small dataset does not idle lanes.
* ``EVOGP_FUSED_NUM_WARPS`` (4) -- warps per program.
* ``EVOGP_FUSED_MAX_SCRATCH_MB`` (8192) -- memory envelope of the call,
  ``pop_size * max_tree_len * rows * 4`` bytes (the stack scratch plus the value
  buffer; the stack is ``tree_len`` slots deep even though only ``depth`` slots
  are ever touched).
* ``EVOGP_FUSED_HOST_MEMO`` (16, IMPORT TIME) -- entries of the transpose table.
* ``EVOGP_FUSED_MIN_POP`` (64) -- the pop-size threshold the wiring round will
  consume (:func:`resolve_min_pop`).  A documented reader only; nothing here
  consults it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch

from . import evoxc_engine
from . import fused_kernels
from .dense_memo import DenseMemo, pin, token

__all__ = [
    "BLOCK_ROWS_ENV",
    "DEFAULT_BLOCK_ROWS",
    "DEFAULT_HOST_MEMO",
    "DEFAULT_MAX_SCRATCH_MB",
    "DEFAULT_MIN_POP",
    "DEFAULT_NUM_WARPS",
    "FusedUnsupported",
    "clear_fused_cache",
    "evaluate_fused",
    "fused_available",
    "fused_cache_stats",
    "resolve_block_rows",
    "resolve_max_scratch_mb",
    "resolve_min_pop",
    "resolve_num_warps",
    "sr_fitness_fused",
]

#: Row-block cap of the launch (``min(next_pow2(rows), cap)`` lanes per program).
BLOCK_ROWS_ENV = fused_kernels.BLOCK_ROWS_ENV
DEFAULT_BLOCK_ROWS = fused_kernels.DEFAULT_BLOCK_ROWS
#: Warps per program (AMD wavefront = 64 lanes).
NUM_WARPS_ENV = "EVOGP_FUSED_NUM_WARPS"
DEFAULT_NUM_WARPS = fused_kernels.DEFAULT_NUM_WARPS
#: Memory envelope in MiB over ``pop_size * max_tree_len * rows * 4`` bytes.
MAX_SCRATCH_MB_ENV = "EVOGP_FUSED_MAX_SCRATCH_MB"
DEFAULT_MAX_SCRATCH_MB = 8192
#: Entries of the transposed-input table (import time; 0 disables it).
HOST_MEMO_ENV = "EVOGP_FUSED_HOST_MEMO"
DEFAULT_HOST_MEMO = 16
#: Pop-size threshold the ROUTE WIRING round will consume (unused here).
MIN_POP_ENV = "EVOGP_FUSED_MIN_POP"
DEFAULT_MIN_POP = 64

#: ``(pop*rows*L*4)`` bytes per MiB.
_MIB = 2**20


class FusedUnsupported(RuntimeError):
    """The fused (Triton) route cannot evaluate this call.

    Callers must fall back to the specialized engine
    (:func:`evoxc_engine.decode_population` + :func:`evoxc_engine.evaluate`),
    which owns the error contract for malformed trees.  ``reason`` carries the
    human-readable cause.  ``device_fallback`` mirrors
    :class:`evogp.tree.dense_engine.DenseUnsupported` so the wiring round can
    treat the two declines uniformly -- the fused route never sets it (it has no
    numpy twin, so its fallback is always the specialized engine).
    """

    def __init__(self, reason: str, *, device_fallback: bool = False):
        super().__init__(f"fused evaluation unsupported: {reason}")
        self.reason = reason
        self.device_fallback = bool(device_fallback)


# --------------------------------------------------------------------------- #
# knob readers
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    """Read an integer knob, or ``default`` when unset/empty (naming it on error)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def resolve_block_rows(value: Optional[int] = None) -> int:
    """Row-block cap (``EVOGP_FUSED_BLOCK_ROWS``, default 256)."""
    cap = _env_int(BLOCK_ROWS_ENV, DEFAULT_BLOCK_ROWS) if value is None else int(value)
    if cap < 1:
        raise ValueError(f"{BLOCK_ROWS_ENV}={cap} must be >= 1 (it is a lane count)")
    return cap


def resolve_num_warps(value: Optional[int] = None) -> int:
    """Warps per program (``EVOGP_FUSED_NUM_WARPS``, default 4)."""
    warps = _env_int(NUM_WARPS_ENV, DEFAULT_NUM_WARPS) if value is None else int(value)
    if warps < 1:
        raise ValueError(f"{NUM_WARPS_ENV}={warps} must be >= 1 (it is a warp count)")
    return warps


def resolve_max_scratch_mb(value: Optional[int] = None) -> int:
    """Memory envelope in MiB (``EVOGP_FUSED_MAX_SCRATCH_MB``, default 8192)."""
    mb = (
        _env_int(MAX_SCRATCH_MB_ENV, DEFAULT_MAX_SCRATCH_MB)
        if value is None
        else int(value)
    )
    if mb < 1:
        raise ValueError(
            f"{MAX_SCRATCH_MB_ENV}={mb} must be >= 1 (it is a memory envelope in "
            "MiB over pop_size * max_tree_len * rows * 4 bytes)"
        )
    return mb


def resolve_min_pop(value: Optional[int] = None) -> int:
    """The route's pop-size threshold (``EVOGP_FUSED_MIN_POP``, default 64).

    Documented here so the wiring round has a single reader, exactly like
    ``dense_engine.resolve_auto_min_pop``.  Nothing in this module consults it: a
    direct :func:`evaluate_fused` call is served at any ``pop >= 1``.
    """
    minimum = _env_int(MIN_POP_ENV, DEFAULT_MIN_POP) if value is None else int(value)
    if minimum < 1:
        raise ValueError(
            f"{MIN_POP_ENV}={minimum} must be >= 1 (it is a pop_size threshold)"
        )
    return minimum


def _default_host_memo() -> int:
    """``EVOGP_FUSED_HOST_MEMO`` read ONCE at import time (0 disables the table)."""
    entries = _env_int(HOST_MEMO_ENV, DEFAULT_HOST_MEMO)
    if entries < 0:
        raise ValueError(f"{HOST_MEMO_ENV}={entries} must be >= 0 (0 disables it)")
    return entries


#: Transposed-input front table: identity key -> ``(xt, pins)``.
XT_MEMO = DenseMemo("fused_xt", _default_host_memo())

#: This route's own counters (never folded into ``cache_stats()``).
_STATS: Dict[str, int] = {"fused_calls": 0, "fused_declines": 0}
_STATS_LOCK = threading.RLock()


def _bump(name: str, amount: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[name] = _STATS.get(name, 0) + int(amount)


def fused_cache_stats() -> Dict[str, int]:
    """Snapshot of the fused route's own counters (every key is ``fused_*``).

    ``fused_calls`` / ``fused_declines`` count attempts and declines of this
    facade, ``fused_kernel_compiles`` the Triton specializations this process had
    to build (one per ``PER_ROW`` x ``REDUCE`` x ``USE_MSE`` x block x warps
    combination -- the JIT cost), ``fused_xt_hits`` / ``fused_xt_misses`` the
    transposed-input table, ``fused_scratch_allocs`` / ``fused_scratch_reuses``
    the cached stack buffers, and ``fused_block_rows`` / ``fused_num_warps`` the
    knobs a call would resolve RIGHT NOW (informational, so a recording can prove
    which geometry produced a number).

    Deliberately separate from ``evoxc_engine.cache_stats()`` (pinned at exactly
    8 keys) and from ``dense_cache_stats()`` / ``batch_cache_stats()``.
    """
    with _STATS_LOCK:
        merged: Dict[str, int] = dict(_STATS)
    merged.update(XT_MEMO.stats())
    merged.update(fused_kernels.scratch_stats())
    merged["fused_block_rows"] = resolve_block_rows()
    merged["fused_num_warps"] = resolve_num_warps()
    return merged


def clear_fused_cache() -> None:
    """Reset every counter and drop the transpose/stack tables (memory only).

    Registered with :func:`evoxc_engine._register_cache_reset` at import time, so
    ``evoxc_engine.clear_cache()`` -- the single in-memory invalidation point of
    the engine -- flushes this route too.  Nothing here touches the disk.
    """
    with _STATS_LOCK:
        for name in list(_STATS):
            _STATS[name] = 0
    XT_MEMO.reset()
    fused_kernels.clear_caches()


evoxc_engine._register_cache_reset(clear_fused_cache)


# --------------------------------------------------------------------------- #
# availability and small helpers
# --------------------------------------------------------------------------- #
def fused_available() -> bool:
    """Whether this box can run the fused route at all.

    ``triton`` importable (:func:`fused_kernels.triton_available`, which does not
    import it) AND a torch-visible CUDA/HIP device.  It says nothing about a
    particular population -- that is :func:`evaluate_fused`'s decline.
    """
    if not torch.cuda.is_available():
        return False
    return fused_kernels.triton_available()


def _backend_name(backend: Optional[str]) -> Optional[str]:
    """The evoxc backend name a call would run on, or ``None`` when unspecified.

    ``None`` means "the caller did not pick an evoxc backend": the fused route
    needs no evoxc graph, so it does not consult ``EVOGP_EVOXC_BACKEND`` (whose
    default is ``numpy``) -- it runs on the device tensor's own device.  An
    explicit name is validated by the engine's own resolver (so an unknown
    spelling raises ``ValueError`` exactly like every other route).
    """
    if backend is None:
        return None
    return evoxc_engine._resolve_backend(backend, None)[0]


def _decline_if_numpy(backend: Optional[str]) -> None:
    """``backend="numpy"`` is a decline: Triton has no numpy twin."""
    name = _backend_name(backend)
    if name is not None and name != "iree":
        raise FusedUnsupported(
            f"backend {name!r} cannot run a Triton kernel (only a device backend "
            "can; pass backend='iree' or leave backend unset)"
        )


def _require_available() -> None:
    """Decline when the box cannot run Triton at all."""
    if not torch.cuda.is_available():
        raise FusedUnsupported(
            "no CUDA/HIP device is visible to torch (torch.cuda.is_available() "
            "is False), so a Triton kernel cannot be launched"
        )
    if not fused_kernels.triton_available():
        raise FusedUnsupported("triton is not importable in this interpreter")


def _population_shape(batch_subtree_size: Any) -> Tuple[int, int]:
    """``(pop_size, max_tree_len)`` from the population's row stride tensor."""
    shape = getattr(batch_subtree_size, "shape", None)
    if shape is None or len(shape) != 2:
        got = tuple(shape) if shape is not None else type(batch_subtree_size).__name__
        raise ValueError(
            f"batch_subtree_size must be a 2-D (pop_size, max_tree_len) tensor, got {got}"
        )
    return int(shape[0]), int(shape[1])


def _as_device_tensor(value: Any, dtype: torch.dtype, device: torch.device, name: str) -> torch.Tensor:
    """``value`` as a tensor of ``device``/``dtype`` without mutating the caller's."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor, got {type(value).__name__}")
    if not value.is_cuda:
        raise FusedUnsupported(
            f"{name} lives on {value.device}; the fused route needs device tensors "
            "(it never stages host memory)"
        )
    if value.device != device:
        # A mixed-device call is legal but pays one copy; the launch targets the
        # POPULATION's device, which is the one that must also hold the scratch.
        value = value.to(device)
    if value.dtype != dtype:
        value = value.to(dtype)
    return value


def _transposed(x: torch.Tensor) -> torch.Tensor:
    """``x.t().contiguous()``, memoized by the input tensor's identity.

    ``(rows, input_len)`` -> ``(input_len, rows)`` so that a VAR node of the walk
    reads one coalesced vector.  An unchanged dataset (the steady state of a
    training loop) is a table hit and costs nothing; a non-torch ``x`` is not
    memoized at all.
    """
    key = token(x)
    if key is None:
        return x.t().contiguous()
    entry = XT_MEMO.lookup(key)
    if entry is not None:
        return entry[0]
    xt = x.t().contiguous()
    # ``pin`` keeps the SOURCE storage alive, so its data_ptr (half of the key)
    # cannot be recycled by a different tensor while this entry is reachable.
    XT_MEMO.store(key, (xt, pin((x,))))
    return xt


def _check_envelope(pop: int, tree_len: int, rows: int, max_scratch_mb: Optional[int]) -> None:
    """Decline a call whose stack scratch would exceed the memory envelope.

    The kernel's scratch is ``n_prog * stack_cap * block_rows * 4`` bytes with
    ``n_prog * block_rows ~ pop * rows`` and ``stack_cap = tree_len``, i.e.
    ``pop_size * max_tree_len * rows * 4`` bytes (plus the ``(pop, rows, 1)``
    value buffer).  The slots above the tree's depth are never touched, so the
    resident cost is a fraction of that -- but the virtual allocation is what the
    envelope has to allow.
    """
    budget = resolve_max_scratch_mb(max_scratch_mb)
    needed = int(pop) * int(tree_len) * int(rows) * 4
    if needed > budget * _MIB:
        raise FusedUnsupported(
            f"scratch envelope exceeded: pop_size {pop} * max_tree_len {tree_len} * "
            f"rows {rows} * 4 bytes = {needed / _MIB:.1f} MiB > "
            f"{MAX_SCRATCH_MB_ENV} {budget} MiB (raise it, or use the dense/batch "
            "route for this shape)"
        )


# --------------------------------------------------------------------------- #
# the two entry points
# --------------------------------------------------------------------------- #
class _Prepared:
    """One validated call: population tensors + the call's geometry parameters."""

    __slots__ = (
        "node_value",
        "node_type",
        "subtree_size",
        "pop",
        "tree_len",
        "device",
        "block_rows",
        "num_warps",
    )

    def __init__(self, **kwargs: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kwargs[name])


def _prepare(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
) -> _Prepared:
    """Every check and normalisation the two entry points share.

    Deliberately pure: no ``x``/``inputs`` inspection (the value path and the
    loss path have different row semantics), no transpose, no allocation.  The
    cheap shape probes run BEFORE :func:`fused_available` so a caller that passes
    a malformed population gets the ``ValueError`` it deserves instead of an
    environment decline.
    """
    _decline_if_numpy(backend)
    pop, tree_len = _population_shape(batch_subtree_size)
    if int(output_len) != 1:
        raise FusedUnsupported(
            f"output_len={int(output_len)}: the fused route implements the "
            "single-output protocol only (multi-output lands later)"
        )
    if tree_len < 1:
        raise ValueError(f"max_tree_len must be >= 1, got {tree_len}")
    if pop < 1:
        raise FusedUnsupported("pop_size must be >= 1, got 0")
    _require_available()
    device = (
        batch_node_value.device
        if isinstance(batch_node_value, torch.Tensor)
        else torch.device("cuda")
    )
    node_value = _as_device_tensor(batch_node_value, torch.float32, device, "batch_node_value")
    node_type = _as_device_tensor(batch_node_type, torch.int16, device, "batch_node_type")
    subtree_size = _as_device_tensor(
        batch_subtree_size, torch.int16, device, "batch_subtree_size"
    )
    for name, tensor in (
        ("batch_node_value", node_value),
        ("batch_node_type", node_type),
        ("batch_subtree_size", subtree_size),
    ):
        if tuple(tensor.shape) != (pop, tree_len):
            raise ValueError(
                f"{name} must be (pop_size, max_tree_len) = {(pop, tree_len)}, got "
                f"{tuple(tensor.shape)}"
            )
    return _Prepared(
        node_value=node_value,
        node_type=node_type,
        subtree_size=subtree_size,
        pop=pop,
        tree_len=tree_len,
        device=device,
        block_rows=resolve_block_rows(block_rows),
        num_warps=resolve_num_warps(num_warps),
    )


def evaluate_fused(
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
    block_rows: Optional[int] = None,
    num_warps: Optional[int] = None,
    max_scratch_mb: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate a whole population with ONE Triton pass (see the module docstring).

    ``per_row=True`` -> ``(pop_size, output_len)``; ``per_row=False`` ->
    ``(pop_size, rows, output_len)``.  Fresh, contiguous, ``requires_grad=False``
    float32 on the computation device (the population's device; ``out_device``
    overrides).  ``x`` and the population tensors are never modified.

    Host crossings: NONE in the steady state.  ``x`` is transposed once per
    distinct dataset tensor (:func:`_transposed`, an identity-keyed table hit
    afterwards) and the value buffer is the kernel's only output -- no decode, no
    key derivation, no readback, no host assembly.

    Raises:
        FusedUnsupported: the route cannot serve the call (see the class).
    """
    _bump("fused_calls")
    try:
        return _evaluate_fused(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            input_len,
            output_len,
            x,
            per_row=per_row,
            out_device=out_device,
            backend=backend,
            block_rows=block_rows,
            num_warps=num_warps,
            max_scratch_mb=max_scratch_mb,
        )
    except FusedUnsupported:
        _bump("fused_declines")
        raise


def _evaluate_fused(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    *,
    per_row: bool,
    out_device: Optional[Any],
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
    max_scratch_mb: Optional[int],
) -> torch.Tensor:
    """Body of :func:`evaluate_fused` (the decline bookkeeping is in the wrapper)."""
    prepared = _prepare(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        backend,
        block_rows,
        num_warps,
    )
    if not isinstance(x, torch.Tensor):
        raise FusedUnsupported(
            f"x must be a device tensor, got {type(x).__name__} (the fused route "
            "never stages host memory)"
        )
    if x.dim() != 2:
        raise ValueError(
            f"x must be 2-D (rows, {int(input_len)}), got shape {tuple(x.shape)}"
        )
    if int(x.shape[1]) != int(input_len):
        raise ValueError(
            f"x has {int(x.shape[1])} columns but input_len is {int(input_len)}"
        )
    if per_row and int(x.shape[0]) != prepared.pop:
        raise ValueError(
            f"per_row=True needs one input row per tree: x has {int(x.shape[0])} rows "
            f"but pop_size is {prepared.pop}"
        )
    rows = 1 if per_row else int(x.shape[0])
    if rows < 1:
        raise FusedUnsupported("rows must be >= 1, got 0")
    x = _as_device_tensor(x, torch.float32, prepared.device, "x")
    _check_envelope(prepared.pop, prepared.tree_len, rows, max_scratch_mb)

    if per_row:
        # Tree t reads x[t]; no transpose is needed (a scalar load per VAR node).
        out = fused_kernels.launch_value(
            prepared.node_value,
            prepared.node_type,
            prepared.subtree_size,
            x,
            input_len=int(input_len),
            tree_len=prepared.tree_len,
            rows=1,
            per_row=True,
            block_rows=prepared.block_rows,
            num_warps=prepared.num_warps,
        )
        result = out.squeeze(1)  # (pop, 1, 1) -> (pop, 1) view of the fresh buffer
    else:
        result = fused_kernels.launch_value(
            prepared.node_value,
            prepared.node_type,
            prepared.subtree_size,
            _transposed(x),  # (input_len, rows), memoized by identity
            input_len=int(input_len),
            tree_len=prepared.tree_len,
            rows=rows,
            per_row=False,
            block_rows=prepared.block_rows,
            num_warps=prepared.num_warps,
        )
    if out_device is not None and torch.device(out_device) != result.device:
        result = result.to(out_device)
    return result.contiguous()


def sr_fitness_fused(
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
    block_rows: Optional[int] = None,
    num_warps: Optional[int] = None,
    max_scratch_mb: Optional[int] = None,
) -> torch.Tensor:
    """Symbolic-regression fitness via the fused evaluator (IN-KERNEL loss).

    ``inputs`` is ``(N, input_len)``, ``labels`` is ``(N, output_len)``
    (``output_len == 1`` on this route) and the result is a ``(pop_size,)``
    float32 POSITIVE loss ``sum_N sum_out err / N`` (MSE or MAE) on the
    computation device unless ``out_device``; NaN/inf propagate unsanitized --
    same formula, same float32 placement and the same ``label - prediction``
    order as :func:`evoxc_engine.sr_fitness`.

    The kernel reduces each (tree, row-block) to ONE partial, so the only device
    output of the pass is the ``(pop, n_rb)`` chart: summing it and dividing by
    ``N`` are two tiny device ops and no prediction ever crosses to the host.

    Raises:
        FusedUnsupported: the route cannot serve the call.
        ValueError: shape contract violations (mismatched rows/columns).
    """
    _bump("fused_calls")
    try:
        return _sr_fitness_fused(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            input_len,
            output_len,
            inputs,
            labels,
            use_MSE,
            out_device=out_device,
            backend=backend,
            block_rows=block_rows,
            num_warps=num_warps,
            max_scratch_mb=max_scratch_mb,
        )
    except FusedUnsupported:
        _bump("fused_declines")
        raise


def _sr_fitness_fused(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    inputs: Any,
    labels: Any,
    use_MSE: bool,
    *,
    out_device: Optional[Any],
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
    max_scratch_mb: Optional[int],
) -> torch.Tensor:
    """Body of :func:`sr_fitness_fused` (the decline bookkeeping is in the wrapper)."""
    prepared = _prepare(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        backend,
        block_rows,
        num_warps,
    )
    if not isinstance(inputs, torch.Tensor):
        raise FusedUnsupported(
            f"inputs must be a device tensor, got {type(inputs).__name__}"
        )
    if not isinstance(labels, torch.Tensor):
        raise FusedUnsupported(
            f"labels must be a device tensor, got {type(labels).__name__}"
        )
    if inputs.dim() != 2:
        raise ValueError(
            f"inputs must be 2-D (data_points, input_len), got {tuple(inputs.shape)}"
        )
    if int(inputs.shape[1]) != int(input_len):
        raise ValueError(
            f"inputs has {int(inputs.shape[1])} columns but input_len is {int(input_len)}"
        )
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be 2-D (data_points, output_len), got {tuple(labels.shape)}"
        )
    if int(labels.shape[0]) != int(inputs.shape[0]):
        raise ValueError(
            f"labels has {int(labels.shape[0])} rows but inputs has {int(inputs.shape[0])}"
        )
    if int(labels.shape[1]) != int(output_len):
        raise ValueError(
            f"labels has {int(labels.shape[1])} columns but output_len is {int(output_len)}"
        )
    rows = int(inputs.shape[0])
    if rows < 1:
        raise FusedUnsupported("sr_fitness needs at least one data point, got 0 rows")
    inputs = _as_device_tensor(inputs, torch.float32, prepared.device, "inputs")
    label_column = _as_device_tensor(labels, torch.float32, prepared.device, "labels")
    _check_envelope(prepared.pop, prepared.tree_len, rows, max_scratch_mb)
    # (N, 1) -> (N,) pointer view: ``out_len == 1`` is enforced by _prepare, so the
    # label column is contiguous and the flatten is free.
    label_ptr = label_column.reshape(-1)
    partials = fused_kernels.launch_loss(
        prepared.node_value,
        prepared.node_type,
        prepared.subtree_size,
        _transposed(inputs),
        label_ptr,
        input_len=int(input_len),
        tree_len=prepared.tree_len,
        rows=rows,
        use_mse=bool(use_MSE),
        block_rows=prepared.block_rows,
        num_warps=prepared.num_warps,
    )
    # ``sum_points sum_outputs err / N`` -- outputs are already summed (there is
    # exactly one), the divisor is the data-point count only.
    loss = partials.sum(dim=1) / float(rows)
    if out_device is not None and torch.device(out_device) != loss.device:
        loss = loss.to(out_device)
    return loss.contiguous()
