"""FUSED whole-population evaluation route: one Triton pass, no host crossings.

This module is the public facade of the ``fused`` route -- the fourth evaluation
engine beside ``specialized`` (:mod:`evogp.tree.evoxc_engine`), ``dense``
(:mod:`evogp.tree.dense_engine`) and the opt-in merged graph
(:mod:`evogp.tree.batch_engine`).  Like ``dense_engine`` it evaluates a WHOLE
population in one launch; unlike every existing route it needs no evoxc graph, no
host decode, no host-side assembly and no readback on the value path.

The route is WIRED: :func:`evogp.tree.dense_engine.resolve_route` owns the policy
branch (``EVOGP_EVAL_MODE=fused`` -> ``Route(engine="fused", ...)``, reported by
``Route.use_fused``) and all five call sites dispatch on it -- ``Forest.forward``
(``per_row=True``), ``Forest.batch_forward`` (``per_row=False``),
``Forest.SR_fitness`` (the loss path) and ``Tree.forward`` / ``Tree.SR_fitness``
(``pop_size == 1``) -- each continuing on the route's fallback after a
:class:`FusedUnsupported`.  ``EVOGP_FUSED_MIN_POP`` (default 16,
:func:`resolve_min_pop`) gates the NO-EXPLICIT-BACKEND case only: an explicitly
named backend always attempts the route, while the default policy needs a device
usable from torch and ``pop_size >= EVOGP_FUSED_MIN_POP`` and otherwise resolves
to specialized+``numpy``.  ``auto`` never selects this engine.

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
* Every tensor the kernel reads is made CONTIGUOUS first (:func:`_as_device_tensor`:
  the three population tables, ``x``, ``inputs`` and ``labels``), because the kernel
  addresses them with raw pointer arithmetic (``x_ptr + t * input_len + vi``,
  ``node_value_ptr + row_base + j``, ``labels_ptr + r * out_len + k``) and would
  read a non-contiguous view (a column slice, a ``.t()`` block, a step-2 table) with
  the wrong strides -- silently.  A non-contiguous operand is COERCED (one copy),
  never declined, and the check is a cached flag test: 0.37 us against 0.20 us for
  the cast-only body, i.e. ~0.8 % of a 0.095 ms warm value call.
* ``sr_fitness_fused`` returns ``(pop_size,)`` float32 POSITIVE loss
  ``sum_points sum_outputs err / N`` (MSE or MAE on ``label - prediction``,
  outputs summed but never divided by ``output_len``, NaN/inf unsanitized).
* A call the route cannot serve raises :class:`FusedUnsupported` with a
  human-readable ``reason`` -- never a wrong number and never a crash.  The
  reasons are: triton not importable; no CUDA/HIP device; the population or
  ``x`` not on a device tensor; an explicit ``backend="numpy"`` (a Triton kernel
  cannot run on the evoxc reference interpreter); ``output_len > MAX_OUT_LEN``
  (8 -- the multi-output kernel accumulates in the ``(pop, rows, out_len)``
  prediction buffer, so ``out_len`` costs one extra read-modify-write per tagged
  node, and 8 bounds that traffic); ``pop_size < 1``; ``rows < 1``; or the memory
  envelope (:func:`_check_envelope` -- the bytes one launch really allocates vs
  ``EVOGP_FUSED_MAX_SCRATCH_MB``) being exceeded.
  ``FusedUnsupported.device_fallback`` exists for symmetry with
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

Measured economics (six benchmark cells, warm medians of 2 warmup + 10
sync-bracketed reps, one process per cell, MI300X, the shipped ``BLOCK_R`` 256 /
1 warp; reproduce with ``python -m evogp.tree.fused_check --stage timing``)
------------------------------------------------------------------------------------------------------
The value path is **48.6x-167.6x faster** than the ``torch-batched`` pure-torch
arm (``test/benchmark/evoxc/arms_torch.py:BatchedEvaluator``), the loss path
**43.4x-156.4x**, and the per-tree (``per_row=True``) path 56x-150x::
    cell                     torch-batched    fused     speedup   n_rb
    p256-L32-i8-N1024            4.127 ms   0.084 ms    49.15x     4
    p256-L64-i32-N1024           4.578 ms   0.083 ms    55.17x     4
    p256-L64-i32-N4096           4.959 ms   0.102 ms    48.57x    16
    p1000-L64-i32-N1024          4.856 ms   0.094 ms    51.93x     4
    p4096-L64-i32-N1024         10.480 ms   0.131 ms    79.80x     4
    p2048-L128-i32-N1024-d7     26.341 ms   0.157 ms   167.62x     4
    loss (MSE): 4.173/0.096 (43.4x), 4.634/0.101 (45.7x), 4.974/0.110 (45.4x),
    4.868/0.103 (47.4x), 10.502/0.135 (77.8x), 26.368/0.169 (156.4x)
    per_row:    4.058/0.072 (56.3x), 4.535/0.073 (61.9x), 4.285/0.072 (59.3x),
    4.560/0.072 (63.2x), 4.426/0.075 (59.0x), 15.807/0.106 (149.7x)
The reason is a traffic and a launch-count argument, not a vectorization trick:
the arm materializes a ``(pop, L+1, N)`` value buffer and runs one eager-torch
kernel per (node position, op) -- ~30 launches of a ``262 144``-element tensor at
p256/``N`` 1024 plus the ``L`` buffer round trips -- while the fused route emits
ONE launch per (tree, row-block) pair (1 024 programs at p256/``N`` 1024 with
``BLOCK_R`` 256, i.e. four row blocks per tree) and keeps every operand in a
register or in the shared-memory stack, so the population's node tables are read
``rows / BLOCK_R`` times at most, i.e. 4 times instead of ``L`` times.
The ratio therefore grows with the TREE axis (167.6x on the 123-node ``-d7``
cell at pop 2 048, where the arm has ~120 positions to stream, and 79.8x at pop
4 096, where the arm's 262 144-element launches finally saturate the device while
the fused call is 4096 short programs) and is lowest on ``p256-L64-i32-N4096``
(48.6x): 16 row blocks per tree re-walk each tree 16 times, which is the one
direction where this scheme pays a real cost.
Where a warm call goes (the battery's stage split, ms): the launch itself is
**72-82 %** of it (0.060-0.128) and the host glue the rest (0.023-0.033:
validation, the transpose-memo lookup, the one ``contiguous()`` and the
``(pop, rows, 1)`` output write); the transpose a COLD dataset pays is
0.018-0.020 ms and is memoized afterwards, and the loss path adds 0.033-0.039 ms
of host glue for the ``(pop, n_rb)`` chart reduction.  Cross-checked on exactly
these populations, the value path is bitwise against the arm on 2 of the 6 cells
and differs only in the sign of a zero on the other 4 (``max_abs = 0.0``), while
the loss agrees to ``<= 3.8e-07`` relative -- the numbers above are not fast
wrong answers, they are the same numbers.
Cold (first call of a fresh process): **375-378 ms** with a warm Triton on-disk
cache (Triton import ~57 ms + the first JIT compile of the module), ~1.36-1.38 s
from a cold cache (a new ``(BLOCK_R, warps, flags)`` key has to be compiled), and
**1.85-1.89 ms** for the specialization that follows it in the same process.  That
is ~4 500x the warm call, so this route needs a loop that evaluates many
generations per process (a GP run does, and the compile is paid once per
``(PER_ROW, REDUCE, USE_MSE, block, warps)`` combination, not per generation) and
is a bad fit for a one-shot script.
Parity (``python -m evogp.tree.fused_check --stage parity``): 56/56 comparisons
against ``src/evogp/tests/reference_eval.py`` -- **bitwise** on the arithmetic
set's value paths and on every handmade adversarial row that has no NaN/zero-sign
case, ``<= 1.8e-07`` relative on the loose/rich transcendental sets' value paths
and ``<= 1e-07`` on the loss paths (float32 reduction order), plus 10/10 golden
fixtures inside 1e-4 (measured ``<= 1.9e-06``), 24/24 decline/contract checks and
31/31 wiring checks.  The ``OUT_NODE`` multi-output protocol has its own stage
(``--stage multi``: 13 case x ``out_len`` populations, 4 surfaces, 65/65
comparisons, value path BITWISE, loss ``<= 1.2e-07`` relative).
The only rows that are not bitwise are those whose result is a NaN payload or a
signed zero, where the numeric value is identical (``max_abs = 0.0``).
Tuning (``--stage tuning``: 12 configs x 6 cells, one process per config, value
path in ms; ``worst`` = the config's slowest cell relative to that cell's best)
------------------------------------------------------------------------------------------------------
    cfg        L32    L64    N4k    p1000  p4096   d7      worst
    b64  w1  0.093  0.090  0.124  0.122  0.240  0.304    1.97
    b64  w2  0.100  0.094  0.154  0.152  0.368  0.480    3.12
    b64  w4  0.099  0.093  0.154  0.149  0.371  0.471    3.06
    b128 w1  0.082  0.084  0.099  0.099  0.157  0.190    1.23
    b128 w2  0.087  0.084  0.119  0.125  0.235  0.304    1.97
    b128 w4  0.094  0.095  0.152  0.153  0.361  0.460    2.99
    b256 w1  0.084  0.082  0.097  0.096  0.134  0.161    1.04   <- shipped
    b256 w2  0.089  0.085  0.099  0.097  0.148  0.186    1.21
    b256 w4  0.083  0.093  0.119  0.117  0.225  0.287    1.86
    b512 w1  0.094  0.093  0.096  0.095  0.150  0.158    1.14
    b512 w2  0.085  0.091  0.097  0.096  0.131  0.154    1.11
    b512 w4  0.086  0.084  0.104  0.096  0.146  0.182    1.18
``BLOCK_R`` 256 / 1 warp is the minimax choice: it is the only config within
4.3 % of the per-cell best on EVERY cell (and the best on the two heaviest ones),
while 64 costs 1.5-3x on the large-row cells -- the tree walk is per PROGRAM, so a
narrow block multiplies the number of tree walks (at ``N`` = 1024, ``BLOCK_R`` 64
is 16 blocks per tree) -- and 512 is inside noise of 256 (1.11 worst) but wastes
lanes on the small-row cells.  The warps axis is the interesting one: 4 -> 1 buys
1.68x at p4096 (0.225 -> 0.134) and 1.78x at ``d7`` (0.287 -> 0.161) and costs
nothing on the small cells, because the walk is a scalar loop-carried dependency
chain -- every lane executes the same instruction sequence, so extra warps have
nothing to overlap and only add scheduling.
Environment knobs (read per call unless noted; an unusable value raises
``ValueError`` naming the variable)
-------------------------------------------------------------------
* ``EVOGP_FUSED_BLOCK_ROWS`` (256) -- CAP of the row-block width, ROUNDED DOWN to a
  power of two (``100`` -> ``64``, :func:`resolve_block_rows`), because
  ``tl.arange(0, BLOCK_R)`` accepts nothing else; the launch then uses the largest
  power of two ``<= min(rows, cap)`` lanes, so a small dataset does not idle lanes.
* ``EVOGP_FUSED_NUM_WARPS`` (1) -- warps per program.
* ``EVOGP_FUSED_MAX_SCRATCH_MB`` (8192) -- memory envelope over the bytes one
  launch ACTUALLY allocates (scratch + value buffer + loss chart, with the row axis
  rounded UP to the block; the formula is :func:`_check_envelope`'s).  It is also
  the budget handed to the scratch table for that call, so a call can never admit
  an allocation its own table refuses.  Being exact costs ~3.8 us of host glue per
  call (+4 % warm at ``p256-L64-i32-N1024``: 0.1000 -> 0.1013 ms).
* ``EVOGP_FUSED_HOST_MEMO`` (16, IMPORT TIME) -- entries of the transpose table.
* ``EVOGP_FUSED_MIN_POP`` (16) -- the pop-size threshold the POLICY consumes
  through :func:`resolve_min_pop`: with ``EVOGP_EVAL_MODE=fused`` and no explicit
  backend, the route needs a device plus ``pop_size >= EVOGP_FUSED_MIN_POP`` and
  resolves to specialized+``numpy`` otherwise (an explicitly named backend always
  attempts the route).  Nothing in this module branches on it.

A knob reader gates a CALL, so an unusable value raises ``ValueError`` naming the
variable; a RECORDING must never raise, so ``fused_cache_stats()`` reports ``-1``
for an unusable ``EVOGP_FUSED_BLOCK_ROWS`` / ``EVOGP_FUSED_NUM_WARPS``.

The threshold is a measured crossover, not a safety floor: the kernel is FLAT in
``pop_size`` (0.088-0.103 ms from pop 1 to pop 128 at ``L=64`` / ``input_len`` 32
/ ``N`` 1024, warm medians, MI300X; records
``fused_results/minpop.json``, table ``fused_results/ENVELOPE.md`` §3) while the
specialized fallback scales linearly with the tree count -- fused / specialized /
``torch-batched`` in ms: pop 1 0.093 / 0.490 / 1.396, pop 4 0.092 / 0.925 / 1.951,
pop 8 0.159 (clock-ramp outlier; the kernel is 0.088-0.103) / 1.644 / 2.662,
pop 16 0.103 / 2.725 / 3.367, pop 32 0.097 / 5.679 / 3.954, pop 64
0.097 / 12.154 / 4.233, pop 128 0.092 / 22.536 / 4.110.  The old gate of 64
therefore threw away 26.5x at pop 16 and 10.1x at pop 4.  16 rather than 1-8
because every measurement is at ``N`` 1024 only and a 1-8-tree population on a
tiny row plane is the one class where a single per-tree numpy run could undercut
the kernel's ~0.09 ms launch floor (unmeasured), and because it keeps a single
``Tree.forward`` on the specialized route instead of paying a ~1.4 s cold JIT for
a one-shot call.
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
    "MAX_OUT_LEN",
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

#: Row-block CAP (``EVOGP_FUSED_BLOCK_ROWS`` 256, rounded DOWN to a power of two):
#: the launch uses the largest power of two ``<= min(rows, cap)`` lanes per program.
BLOCK_ROWS_ENV = fused_kernels.BLOCK_ROWS_ENV
DEFAULT_BLOCK_ROWS = fused_kernels.DEFAULT_BLOCK_ROWS
#: Warps per program (AMD wavefront = 64 lanes).
NUM_WARPS_ENV = "EVOGP_FUSED_NUM_WARPS"
DEFAULT_NUM_WARPS = fused_kernels.DEFAULT_NUM_WARPS
#: Memory envelope in MiB over the bytes the launch actually allocates (scratch +
#: value buffer + loss chart); see :func:`_check_envelope`.
MAX_SCRATCH_MB_ENV = "EVOGP_FUSED_MAX_SCRATCH_MB"
DEFAULT_MAX_SCRATCH_MB = 8192
#: Entries of the transposed-input table (import time; 0 disables it).
HOST_MEMO_ENV = "EVOGP_FUSED_HOST_MEMO"
DEFAULT_HOST_MEMO = 16
#: Pop-size threshold the ROUTE POLICY consumes through ``resolve_min_pop``
#: (consulted by ``dense_engine.resolve_fused_min_pop``; not used here).  The
#: kernel is flat in ``pop_size`` (0.088-0.103 ms from pop 1 to 128) while the
#: specialized fallback scales linearly with the tree count, so the threshold is
#: the measured crossover: 16 is where ``fused`` is 26.5x ahead of the fallback
#: (pop 16: 0.103 vs 2.725 ms) while a single ``Tree.forward`` stays on the
#: specialized route instead of paying a cold Triton JIT; see the module docstring
#: for the full column and :func:`resolve_min_pop`.
MIN_POP_ENV = "EVOGP_FUSED_MIN_POP"
DEFAULT_MIN_POP = 16
#: Outputs the multi-output kernel serves.  It accumulates into the
#: ``(pop, rows, out_len)`` prediction buffer (a ``[output_len, BLOCK_R]`` register
#: tile would make triton unify two layouts across the loop phis and the LLVM
#: translation fails), so ``out_len`` costs one read-modify-write per tagged node
#: and bounds the extra traffic; the library's multi-output trees use 2-3
#: (``Classification`` at ``output_len`` classes, multi-task policies) and a call
#: above this declines into the specialized fallback with a reason naming the
#: constant.
MAX_OUT_LEN = 8

#: Bytes per MiB (the envelope's unit).
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

    The fallback's BACKEND is therefore always ``"numpy"``:
    ``dense_engine.resolve_route`` pins it on every arm (the explicit spelling
    drives ``backend`` only), because a per-tree iree compile cannot amortize.
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
    """Row-block CAP in lanes (``EVOGP_FUSED_BLOCK_ROWS``, default 256).

    A CAP, not a width: the launch uses the largest power of two ``<= min(rows, cap)``
    lanes (:func:`evogp.tree.fused_kernels.plan`).  An off-power-of-two value is
    rounded DOWN here (``100`` -> ``64``) with ``fused_kernels._pow2_floor`` -- the
    SAME rule ``plan`` applies, so the two cannot disagree -- because
    ``tl.arange(0, BLOCK_R)`` accepts nothing else.  A value below 1 is refused.
    """
    cap = _env_int(BLOCK_ROWS_ENV, DEFAULT_BLOCK_ROWS) if value is None else int(value)
    if cap < 1:
        raise ValueError(f"{BLOCK_ROWS_ENV}={cap} must be >= 1 (it is a lane count)")
    return fused_kernels._pow2_floor(cap)


def resolve_num_warps(value: Optional[int] = None) -> int:
    """Warps per program (``EVOGP_FUSED_NUM_WARPS``, default 1)."""
    warps = _env_int(NUM_WARPS_ENV, DEFAULT_NUM_WARPS) if value is None else int(value)
    if warps < 1:
        raise ValueError(f"{NUM_WARPS_ENV}={warps} must be >= 1 (it is a warp count)")
    return warps


def resolve_max_scratch_mb(value: Optional[int] = None) -> int:
    """Memory envelope in MiB (``EVOGP_FUSED_MAX_SCRATCH_MB``, default 8192).

    A budget over the bytes one launch allocates (scratch + value buffer + loss
    chart); :func:`_check_envelope` computes them and passes the resolved number
    down so the scratch TABLE is budgeted by the same one.
    """
    mb = (
        _env_int(MAX_SCRATCH_MB_ENV, DEFAULT_MAX_SCRATCH_MB)
        if value is None
        else int(value)
    )
    if mb < 1:
        raise ValueError(
            f"{MAX_SCRATCH_MB_ENV}={mb} must be >= 1 (it is a memory envelope in "
            "MiB over the bytes one fused launch allocates)"
        )
    return mb


def resolve_min_pop(value: Optional[int] = None) -> int:
    """The route's pop-size threshold (``EVOGP_FUSED_MIN_POP``, default 16).

    Documented here so the wiring round has a single reader, exactly like
    ``dense_engine.resolve_auto_min_pop``.  Nothing in this module consults it: a
    direct :func:`evaluate_fused` call is served at any ``pop >= 1``.

    The default is the MEASURED crossover, not a safety floor: the kernel is flat
    in ``pop_size`` (0.088-0.103 ms from pop 1 to pop 128 at ``L=64`` /
    ``input_len`` 32 / ``N`` 1024, warm medians, MI300X; records
    ``fused_results/minpop.json``, table ``fused_results/ENVELOPE.md`` §3) while
    the specialized fallback is linear in the tree count (pop 16: 0.103 ms against
    2.725 ms, i.e. 26.5x; pop 4: 0.092 against 0.925, i.e. 10.1x).  It stays at 16
    rather than 1-8 because every measurement is at ``N`` 1024 only -- a 1-8-tree
    population on a tiny row plane is the one class where one per-tree numpy run
    could undercut the kernel's ~0.09 ms launch floor (unmeasured) -- and because
    a single ``Tree.forward`` (``pop_size == 1``) then keeps the specialized route
    instead of paying a ~1.4 s cold Triton JIT for a one-shot call.  A value below
    1 is refused: this is a ``pop_size`` threshold.
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


def _reported_knob(resolver) -> int:
    """A geometry knob as a RECORDING sees it: the value, or ``-1`` when unusable.

    The readers raise ``ValueError`` on an unusable ambient value (they gate a CALL),
    while a stats call must never raise; ``-1`` is no knob's legal value.
    """
    try:
        return int(resolver())
    except ValueError:
        return -1


def fused_cache_stats() -> Dict[str, int]:
    """Snapshot of the fused route's own counters (every key is ``fused_*``).

    ``fused_calls`` / ``fused_declines`` count attempts and declines of this
    facade, ``fused_kernel_compiles`` the Triton specializations this process had
    to build (one per ``PER_ROW`` x ``REDUCE`` x ``USE_MSE`` x block x warps
    combination -- the JIT cost), ``fused_xt_hits`` / ``fused_xt_misses`` the
    transposed-input table, ``fused_scratch_*`` the byte-budgeted table of cached
    stack buffers (allocs / reuses / evictions / entries / bytes), and
    ``fused_block_rows`` / ``fused_num_warps`` the geometry a call would resolve
    RIGHT NOW (INFORMATIONAL: ``-1`` when the ambient knob is unusable, because a
    stats call never raises -- :func:`_reported_knob`; the readers do).

    Deliberately separate from ``evoxc_engine.cache_stats()`` (pinned at exactly
    8 keys) and from ``dense_cache_stats()`` / ``batch_cache_stats()``.
    """
    with _STATS_LOCK:
        merged: Dict[str, int] = dict(_STATS)
    merged.update(XT_MEMO.stats())
    merged.update(fused_kernels.scratch_stats())
    merged["fused_block_rows"] = _reported_knob(resolve_block_rows)
    merged["fused_num_warps"] = _reported_knob(resolve_num_warps)
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


def _launch(launcher, *args, **kwargs):
    """Call a launch helper, mapping "no triton" onto the facade's decline.

    :func:`fused_kernels.kernels` raises :class:`fused_kernels.FusedUnavailable`
    when the actual import (or the libdevice layout) fails, which
    :func:`fused_available` cannot see because it answers from
    ``importlib.util.find_spec`` without importing anything.  A caller must get
    the documented ``FusedUnsupported`` decline in that case too -- never an
    ``ImportError`` out of a route it was told is optional.
    """
    try:
        return launcher(*args, **kwargs)
    except fused_kernels.FusedUnavailable as exc:
        raise FusedUnsupported(str(exc)) from exc

def _require_available() -> None:
    """Decline when the box cannot run Triton at all, naming which half failed.

    The gate itself is the exported predicate :func:`fused_available`.
    """
    if fused_available():
        return
    if not torch.cuda.is_available():
        raise FusedUnsupported(
            "no CUDA/HIP device is visible to torch (torch.cuda.is_available() "
            "is False), so a Triton kernel cannot be launched"
        )
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
    """``value`` as a CONTIGUOUS ``device``/``dtype`` tensor, never mutating the caller's.

    The kernel derives every offset from the SHAPE and dereferences it with raw
    pointer arithmetic, so a non-contiguous view (a column slice of a padded ``x``, a
    ``.t()`` block, a step-2 population table) would be read with the wrong strides
    and yield plausible garbage -- no exception, no decline, a wrong answer.  The
    layout is normalized here, at the ONE place all five operands pass through, AFTER
    the device/dtype casts (both can keep the source's strides).  It is a cached
    flag test on a contiguous tensor and copies only when the tensor is not.
    """
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
    if not value.is_contiguous():
        value = value.contiguous()
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


def _check_envelope(
    pop: int,
    tree_len: int,
    rows: int,
    max_scratch_mb: Optional[int],
    *,
    block_rows: Optional[int] = None,
    num_warps: Optional[int] = None,
    out_len: int = 1,
    holds_values: bool = True,
) -> int:
    """Decline a call whose DEVICE allocations would exceed the memory envelope.

    The bytes one launch allocates, EXACTLY (not the round
    ``pop * tree_len * rows * 4`` this used to check)::

        scratch   pop * n_rb * tree_len * block_rows * 4   always
        values    pop * rows * out_len * 4                 value path; also the
                                                           multi-output loss path
        partials  pop * n_rb * 4                           the loss path's chart

    ``(n_rb, block_rows) = fused_kernels.plan(rows, ...)``, so ``n_rb * block_rows``
    is the row axis ROUNDED UP to the block (1 000 rows at 256 lanes is 4 blocks,
    i.e. 1 024 lanes) and ``rows`` never enters the scratch term directly.
    ``holds_values`` says which path the caller is on (the single-output loss path
    allocates no value buffer, and only the loss path allocates a chart); rows is
    the DATA-row count (``1`` for ``per_row``, hence no ``per_row`` here), and
    unresolved knobs go through the resolvers, so a caller that knows nothing still
    gets a CONSERVATIVE check.  Returns the resolved budget in MiB so the entry
    points pass it down as ``max_scratch_mb=``: the scratch TABLE is budgeted by the
    same number the envelope was checked against.  The measured peak-allocated
    tracks the scratch term within 3.5 %, value buffer on top
    (``fused_results/ENVELOPE.md``).
    """
    block_rows = resolve_block_rows(block_rows)
    num_warps = resolve_num_warps(num_warps)
    budget = resolve_max_scratch_mb(max_scratch_mb)
    pop, tree_len, rows, out_len = int(pop), int(tree_len), int(rows), int(out_len)
    # ``rows`` already encodes ``per_row`` (1 row); ``plan``'s flag does not move it.
    geom = fused_kernels.plan(rows, per_row=False, block_rows=block_rows,
                              num_warps=num_warps)
    n_rb, block = int(geom["n_rb"]), int(geom["block_rows"])
    scratch = pop * n_rb * tree_len * block * 4
    values = pop * rows * out_len * 4 if (holds_values or out_len > 1) else 0
    partials = 0 if holds_values else pop * n_rb * 4
    total = scratch + values + partials
    if total > budget * _MIB:
        raise FusedUnsupported(
            f"memory envelope exceeded: scratch {scratch / _MIB:.1f} MiB + value "
            f"buffer {values / _MIB:.1f} MiB + loss chart {partials / _MIB:.3f} MiB "
            f"= {total / _MIB:.1f} MiB > {MAX_SCRATCH_MB_ENV} {budget} MiB for "
            f"pop_size {pop} * max_tree_len {tree_len} * rows {rows} * out_len "
            f"{out_len} at {n_rb} row blocks of {block} lanes (raise it, or use the "
            "dense/batch route for this shape)"
        )
    return budget


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

    Deliberately pure: no ``x``/``inputs`` inspection (the value path and the loss
    path have different row semantics), no transpose, no allocation.  The ORDER is:
    the CHEAP SHAPE PROBES (:func:`_population_shape`, ``output_len``, ``tree_len``,
    ``pop_size``), then the BACKEND decline (:func:`_decline_if_numpy`: a Triton
    kernel has no numpy twin, so ``backend="numpy"`` is a decline, not an error),
    then the ENVIRONMENT (:func:`_require_available`).  A malformed population
    therefore gets the ``ValueError`` it deserves instead of a decline.
    """
    pop, tree_len = _population_shape(batch_subtree_size)
    if int(output_len) < 1:
        raise ValueError(f"output_len must be >= 1, got {int(output_len)}")
    if int(output_len) > MAX_OUT_LEN:
        raise FusedUnsupported(
            f"output_len={int(output_len)} > {MAX_OUT_LEN}: the multi-output kernel "
            f"accumulates in the (pop, rows, out_len) prediction buffer, so this "
            f"route serves up to {MAX_OUT_LEN} outputs (raise "
            "fused_engine.MAX_OUT_LEN if a model really needs more)"
        )
    if tree_len < 1:
        raise ValueError(f"max_tree_len must be >= 1, got {tree_len}")
    if pop < 1:
        raise FusedUnsupported("pop_size must be >= 1, got 0")
    _decline_if_numpy(backend)
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
    # Both kernels materialize the ``(pop, rows, out_len)`` buffer here, hence
    # ``holds_values=True``.
    budget_mb = _check_envelope(
        prepared.pop, prepared.tree_len, rows, max_scratch_mb,
        block_rows=prepared.block_rows, num_warps=prepared.num_warps,
        out_len=int(output_len), holds_values=True,
    )

    if per_row:
        # Tree t reads x[t]; no transpose is needed (a scalar load per VAR node).
        out = _launch(
            fused_kernels.launch_value,
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
            out_len=int(output_len),
            max_scratch_mb=budget_mb,
        )
        # (pop, 1, out_len) -> (pop, out_len) view of the fresh buffer
        result = out.squeeze(1)
    else:
        result = _launch(
            fused_kernels.launch_value,
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
            out_len=int(output_len),
            max_scratch_mb=budget_mb,
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
    (``1 <= output_len <= MAX_OUT_LEN``) and the result is a ``(pop_size,)``
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
    # The single-output kernel's only output is the ``(pop, n_rb)`` chart (no value
    # buffer); ``out_len >= 2`` still materializes the accumulator.
    budget_mb = _check_envelope(
        prepared.pop, prepared.tree_len, rows, max_scratch_mb,
        block_rows=prepared.block_rows, num_warps=prepared.num_warps,
        out_len=int(output_len), holds_values=False,
    )
    # ``out_len == 1`` wants the flat ``(N,)`` view (a free reshape of the now
    # contiguous label column); the multi-output kernel indexes ``(N, out_len)``.
    label_ptr = label_column.reshape(-1) if int(output_len) == 1 else label_column
    partials = _launch(
        fused_kernels.launch_loss,
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
        out_len=int(output_len),
        max_scratch_mb=budget_mb,
    )
    # ``sum_points sum_outputs err / N`` -- outputs are already summed (there is
    # exactly one), the divisor is the data-point count only.
    loss = partials.sum(dim=1) / float(rows)
    if out_device is not None and torch.device(out_device) != loss.device:
        loss = loss.to(out_device)
    return loss.contiguous()
