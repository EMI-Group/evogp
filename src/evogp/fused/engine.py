"""FUSED whole-population evaluation route: one Triton pass, no host crossings.

This package is the device-native evaluation route: it evaluates a WHOLE
population with ONE JIT-compiled Triton kernel launch, off the legacy population
tensors themselves -- no compiled graph, no host decode, no host-side assembly
and no readback on the value path.  This module is its host-side facade (the
kernels are :mod:`evogp.fused.kernels`, the front tables :mod:`evogp.fused.memo`).

Who calls it
------------
DIRECTLY, on the legacy triple -- ``batch_node_value`` float32, ``batch_node_type``
int16 and ``batch_subtree_size`` int16, each ``(pop_size, max_tree_len)`` in prefix
(pre-order) encoding, on a torch device:

* :class:`evogp.gp.fused_eval.FusedDeviceEvaluator` -- the torch-free functional
  stack's device-resident objective (it hands the kernel its own device buffer
  through ``out=``);
* ``evogp.fused.check`` -- the in-library battery (parity, declines, timing).

There is no route policy and no dispatch layer here: a caller that reaches this
module has ALREADY decided to evaluate on the device, so nothing below consults
an evaluation-mode environment variable or a population threshold.  A call the
route cannot serve raises :class:`FusedUnsupported` -- never a wrong number,
never a crash -- and the CALLER decides what to do with it (the gp stack falls
back to its own in-graph / host evaluator).  The ids the kernels need
(``DELTA``, ``FUNCS``, ``Func``, ``MAX_OUT_LEN``, ``OUT_NODE``) come from the
torch-free :mod:`evogp._tables`, and the ``OUT_NODE`` rule itself is
:mod:`evogp._multi_output`.

Module scope (deliberate, do not split)
---------------------------------------
This file is ONE deliberately single compilation unit and sits above the
repository's ~1000-line baseline on purpose.  The facade's pieces constrain each
other: the pre-flight (``_require_available`` / ``_decline_if_numpy`` /
``_population_shape`` / ``_as_device_tensor``), the three-term envelope
(``_check_envelope`` + ``_ENVELOPE_MEMO``), the ``XT_MEMO`` transpose preparation,
the byte-bounded scratch table and the counters are all checked TOGETHER against
the kernel geometry that :func:`evogp.fused.kernels.plan` returns, and the
compile-time constants (``MAX_OUT_LEN``, the ``stack_cap = tree_len`` scratch
geometry, the ``BLOCK_R``/warp resolution) are read at the same place the kernel
call is made.
A split would move those constants away from the call that constrains them.
Split it only with failing-test evidence.

Public surface
--------------
================================== ==================================================
:class:`FusedUnsupported`          per-call decline (the route cannot serve this call)
:func:`fused_available`            ``triton`` importable AND a torch CUDA/HIP device
:func:`envelope_decline`           the memory envelope's verdict for a shape (no raise)
:func:`evaluate_fused`             value path, ``(pop, output_len)`` / ``(pop, N, output_len)``
:func:`sr_fitness_fused`           SR loss path, ``(pop,)`` positive loss
:func:`fused_cache_stats`          this route's own counters (all prefixed ``fused_``)
:func:`clear_fused_cache`          reset them (this route's ONE reset entry)
:func:`resolve_block_rows`         ``EVOGP_FUSED_BLOCK_ROWS`` (the row-block cap)
:func:`resolve_num_warps`          ``EVOGP_FUSED_NUM_WARPS``
:func:`resolve_max_scratch_mb`     ``EVOGP_FUSED_MAX_SCRATCH_MB`` (the envelope)
:func:`resolve_dispatch`           ``EVOGP_FUSED_DISPATCH`` (``bound`` / ``jit``)
================================== ==================================================

Contracts:

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
  cannot run on the numpy reference interpreter); ``output_len > MAX_OUT_LEN``
  (8 -- the multi-output kernel accumulates in the ``(pop, rows, out_len)``
  prediction buffer, so ``out_len`` costs one extra read-modify-write per tagged
  node, and 8 bounds that traffic); ``pop_size < 1``; ``rows < 1``; or the memory
  envelope (:func:`_check_envelope` -- the bytes one launch really allocates vs
  ``EVOGP_FUSED_MAX_SCRATCH_MB``) being exceeded.
  ``FusedUnsupported.device_fallback`` is kept as an attribute so a caller can
  treat every decline uniformly; this route never sets it (it has no numpy twin).

The one prep step
-----------------
``per_row=False`` wants the data rows CONTIGUOUS along the vector axis, i.e. a
transposed copy ``xt = x.t().contiguous()`` of shape ``(input_len, rows)``, so a
VAR node is one coalesced vector load ``xt[vi * rows + r]``.  That copy is a
device kernel, and a training loop re-evaluates the same dataset thousands of
times, so it is memoized by IDENTITY (:data:`XT_MEMO`, the
:func:`evogp.fused.memo.token` ``(data_ptr, shape, dtype, device, _version)``, an
in-place write is a miss, the entry pins the caller's storage so the address
cannot be recycled).  Hits and misses are :func:`fused_cache_stats`'s
``fused_xt_hits`` / ``fused_xt_misses``; ``EVOGP_FUSED_HOST_MEMO`` (default 16
entries, import time) bounds the table and ``0`` disables it.  ``per_row=True``
needs no transpose: each tree reads its own scalar row element
``x[t * input_len + vi]``.

Measured economics -- the campaign of record (``FUSED.md`` §0/§2: six benchmark
cells, warm medians of 2 warmup + 10 sync-bracketed reps, one process per cell,
MI300X, the shipped ``BLOCK_R`` 256 / 1 warp; reproduce with ``python -m
evogp.fused.check --stage timing``)
------------------------------------------------------------------------------------------------------
Against the ``torch-batched`` pure-torch arm (the benchmark harness's batched
evaluator) that campaign reads
**41x-157x on the value (``batch_forward``) surface**, **32x-131x on the loss
(``SR_fitness``) surface** and **42x-127x on the per-tree (``per_row=True``)
surface**, i.e. **32x-157x across the three surfaces**.  Its ``batch_forward``
medians (the loss and per-row ratios follow in the same cell order)::
    cell                     torch-batched    fused     speedup   n_rb
    p256-L32-i8-N1024            4.055 ms  0.0995 ms    40.8x     4
    p256-L64-i32-N1024           4.427 ms  0.1016 ms    43.6x     4
    p256-L64-i32-N4096           4.918 ms  0.1095 ms    44.9x    16
    p1000-L64-i32-N1024          4.887 ms  0.1053 ms    46.4x     4
    p4096-L64-i32-N1024         10.640 ms  0.1460 ms    72.9x     4
    p2048-L128-i32-N1024-d7     26.251 ms  0.1673 ms   156.9x     4
    loss (sf):    32.1x / 32.6x / 38.6x / 34.2x / 59.7x / 130.9x
    per_row (fw): 46.2x / 48.7x / 50.1x / 49.9x / 42.3x / 126.6x
The reason is a traffic and a launch-count argument, not a vectorization trick:
the arm materializes a ``(pop, L+1, N)`` value buffer and runs one eager-torch
kernel per (node position, op) -- ~30 launches of a ``262 144``-element tensor at
p256/``N`` 1024 plus the ``L`` buffer round trips -- while the fused route emits
ONE launch per (tree, row-block) pair (1 024 programs at p256/``N`` 1024 with
``BLOCK_R`` 256, i.e. four row blocks per tree) and keeps every operand in a
register or in the shared-memory stack, so the population's node tables are read
``rows / BLOCK_R`` times at most, i.e. 4 times instead of ``L`` times.
The ratio therefore grows with the TREE axis (156.9x on the 123-node ``-d7``
cell at pop 2 048, where the arm has ~120 positions to stream, and 72.9x at pop
4 096, where the arm's 262 144-element launches finally saturate the device while
the fused call is 4096 short programs) and is lowest on the smallest cell (40.8x
at ``p256-L32-i8-N1024``).  The ROW axis is nearly free -- 0.1016 -> 0.1095 ms for
4x the rows, still 44.9x -- while the POP axis is what the tree walk costs:
0.1016 -> 0.1460 ms from pop 256 to 4 096 at ``L=64``, because the walk repeats
once per row block and the per-program work scales with pop.
Where a warm call goes (the campaign's stage split, ``FUSED.md`` §3, ms): the
launch itself is **51-78 %** of it (0.060-0.131) and the host glue the rest
(0.031-0.062: validation, the transpose-memo lookup, the one ``contiguous()`` and
the ``(pop, rows, 1)`` output write); the transpose a COLD dataset pays is
0.019 ms and is memoized afterwards, and the loss path adds 0.046-0.062 ms of
host glue for the ``(pop, n_rb)`` chart reduction.  Cross-checked on exactly
these populations (``FUSED.md`` §6), the value paths are **bit-identical** to
both the arm and the oracle on every finite point (the only differing bits are
NaN payloads / signed zeros, ``max_abs = 0.0``) and the loss agrees to
``<= 3.8e-07`` relative -- the numbers above are not fast wrong answers, they are
the same numbers.
Cold (first call of a fresh process, ``FUSED.md`` §5): **1.64-1.67 s** for the
value kernel from an EMPTY ``TRITON_CACHE_DIR``, plus 0.94-0.99 s for the loss
surface and 0.32 s for the per-row one (~2.9 s of one-time JIT for all three),
against **2.1 ms** for the same call on a warm Triton **disk** hit (Triton's cache
is process-independent); within one process the next surface pays ~0.47 s (loss)
or ~1.7 ms (per-row).  That is ~1.6e4x the warm value call, so this route needs a
loop that evaluates many generations per process (a GP run does, and the compile
is paid once per ``(PER_ROW, REDUCE, USE_MSE, block, warps)`` combination, not
per generation) and is a bad fit for a one-shot script.
Parity (``python -m evogp.fused.check --stage parity``): 56/56 comparisons
against the independent oracle (``test/oracle/reference_eval.py``) over five op
sets (arithmetic, ternary, loose, rich and ``compare_unary`` -- the set that draws
``>`` / ``<=`` / ``neg`` / ``abs`` / ``sinh`` / ``cosh``) -- **bitwise** on the
arithmetic and ``compare_unary`` value paths and on every handmade adversarial row
that has no NaN/zero-sign case, ``<= 1.8e-07`` relative on the loose/rich
transcendental sets' value paths and ``<= 1e-07`` on the loss paths (float32
reduction order), plus 10/10 golden fixtures inside 1e-4 (measured
``<= 1.9e-06``), 57/57 decline/contract checks and 32/32 wiring checks.  The
``OUT_NODE`` multi-output protocol has its own stage (``--stage multi``: 13
case x ``out_len`` populations, 4 surfaces, 65/65 comparisons, value path
BITWISE, loss ``<= 1.2e-07`` relative).
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
The kernel is also FLAT in ``pop_size`` (0.088-0.103 ms from pop 1 to pop 128 at
``L=64`` / ``input_len`` 32 / ``N`` 1024, warm medians, MI300X): the walk repeats
per (tree, row block), and a few hundred of those programs already fill the
device, which is why a caller never has to batch calls itself.
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
  an allocation its own table refuses.  The verdict is memoized on its own eight
  resolved inputs (:data:`_ENVELOPE_MEMO`), so being exact costs **0.5 us** of host
  glue per call instead of the 2.0 us the byte arithmetic and
  :func:`evogp.fused.kernels.plan` cost UNMEMOIZED (+2 % warm at
  ``p256-L64-i32-N1024``, host-only medians) -- and the ~1.5 us of ambient knob
  reads beside it are the same three ``os.environ`` reads the route always paid.
* ``EVOGP_FUSED_HOST_MEMO`` (16, IMPORT TIME) -- entries of the transpose table.
* ``EVOGP_FUSED_DISPATCH`` (``bound``) -- the launch path of a warm call
  (:func:`evogp.fused.kernels.resolve_dispatch`); ``jit`` restores Triton's
  ``JITFunction.run`` and is the A/B arm.

A knob reader gates a CALL, so an unusable value raises ``ValueError`` naming the
variable; a RECORDING must never raise, so ``fused_cache_stats()`` reports ``-1``
for an unusable ``EVOGP_FUSED_BLOCK_ROWS`` / ``EVOGP_FUSED_NUM_WARPS``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from . import kernels as fused_kernels
from .memo import Memo, pin, token
from .._multi_output import MAX_OUT_LEN as _MAX_OUT_LEN
from .._multi_output import MultiOutputError, out_node_message
from .._multi_output import OUT_NODE as _OUT_NODE_BIT

__all__ = [
    "BLOCK_ROWS_ENV",
    "DEFAULT_BLOCK_ROWS",
    "DEFAULT_HOST_MEMO",
    "DEFAULT_MAX_SCRATCH_MB",
    "DEFAULT_NUM_WARPS",
    "FusedUnsupported",
    "MAX_OUT_LEN",
    "clear_fused_cache",
    "envelope_decline",
    "evaluate_fused",
    "fused_available",
    "fused_cache_stats",
    "resolve_block_rows",
    "resolve_dispatch",
    "resolve_max_scratch_mb",
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
#: Outputs the multi-output kernel serves.  It accumulates into the
#: ``(pop, rows, out_len)`` prediction buffer (a ``[output_len, BLOCK_R]`` register
#: tile would make triton unify two layouts across the loop phis and the LLVM
#: translation fails), so ``out_len`` costs one read-modify-write per tagged node
#: and bounds the extra traffic; the library's multi-output trees use 2-3
#: (``Classification`` at ``output_len`` classes, multi-task policies) and a call
#: above this declines (:class:`FusedUnsupported`) with a reason naming the
#: constant, and the caller picks the fallback it wants.  Single-sourced in :mod:`evogp._tables` (the torch-free table
#: module) and re-exported here for this package's importers.
MAX_OUT_LEN = _MAX_OUT_LEN

#: Bytes per MiB (the envelope's unit).
_MIB = 2**20

#: The launch-path resolver (``EVOGP_FUSED_DISPATCH``).  Owned by
#: :mod:`evogp.fused.kernels` beside the launch it selects and re-exported here, so
#: this facade is the ONE import surface of the route.
resolve_dispatch = fused_kernels.resolve_dispatch


class FusedUnsupported(RuntimeError):
    """The fused (Triton) route cannot evaluate this call.

    ``reason`` carries the human-readable cause.  The CALLER decides what the
    decline means: :class:`evogp.gp.fused_eval.FusedDeviceEvaluator` falls back
    to its own in-graph / host route, and the check battery records the reason.
    ``device_fallback`` is kept as an attribute so a caller can treat every
    decline uniformly; this route never sets it -- it evaluates on the device and
    has no host twin of its own.
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
    lanes (:func:`evogp.fused.kernels.plan`).  An off-power-of-two value is
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


#: Memo of the three RESOLVED geometry knobs, keyed on their RAW ambient values
#: (``None`` for a knob the caller passed explicitly, the value itself otherwise).
#: Every warm call needs the triple (it is part of the prepared-call key and of the
#: envelope verdict), every call must SEE an ambient flip, but only a flip changes
#: the answer -- so the readers' cost is split: one ``os.environ.get`` per knob
#: (0.66 us) is paid per call and the parse/validation (1.9 us for all three) runs
#: on a miss.  A miss still goes through the three resolvers, which own the
#: defaults, the ``_pow2_floor`` rule and the ``ValueError`` messages, so an
#: unusable value raises exactly what it always raised.  Bounded (a knob sweep
#: cannot grow it); dropped by :func:`clear_fused_cache` as hygiene, not
#: correctness -- like :data:`_ENVELOPE_MEMO`, no entry can go stale, because its
#: key carries every input of the answer.
_GEOMETRY_MEMO: Dict[Tuple[Any, Any, Any], Tuple[int, int, int]] = {}
_GEOMETRY_MEMO_MAX = 8


def _geometry(
    block_rows: Optional[int] = None,
    num_warps: Optional[int] = None,
    max_scratch_mb: Optional[int] = None,
) -> Tuple[int, int, int]:
    """The call's resolved ``(block_rows, num_warps, max_scratch_mb)`` triple.

    The three geometry knobs as ONE resolution, memoized on their raw values in
    :data:`_GEOMETRY_MEMO` (see there).  The resolvers the memo delegates to are
    :func:`resolve_block_rows` / :func:`resolve_num_warps` /
    :func:`resolve_max_scratch_mb`, i.e. the single source of the validation and the
    error messages, so this reader can only ever be faster, never more lenient.
    """
    get = os.environ.get
    raw = (
        get(BLOCK_ROWS_ENV) if block_rows is None else block_rows,
        get(NUM_WARPS_ENV) if num_warps is None else num_warps,
        get(MAX_SCRATCH_MB_ENV) if max_scratch_mb is None else max_scratch_mb,
    )
    try:
        resolved = _GEOMETRY_MEMO.get(raw)
    except TypeError:  # an unhashable explicit value: resolve without the memo
        return (
            resolve_block_rows(block_rows),
            resolve_num_warps(num_warps),
            resolve_max_scratch_mb(max_scratch_mb),
        )
    if resolved is None:
        resolved = (
            resolve_block_rows(block_rows),
            resolve_num_warps(num_warps),
            resolve_max_scratch_mb(max_scratch_mb),
        )
        if len(_GEOMETRY_MEMO) >= _GEOMETRY_MEMO_MAX:
            _GEOMETRY_MEMO.clear()
        _GEOMETRY_MEMO[raw] = resolved
    return resolved


def _default_host_memo() -> int:
    """``EVOGP_FUSED_HOST_MEMO`` read ONCE at import time (0 disables the tables)."""
    entries = _env_int(HOST_MEMO_ENV, DEFAULT_HOST_MEMO)
    if entries < 0:
        raise ValueError(f"{HOST_MEMO_ENV}={entries} must be >= 0 (0 disables it)")
    return entries


#: Entries of every front table of this route (one ``EVOGP_FUSED_HOST_MEMO`` read).
_HOST_MEMO_ENTRIES = _default_host_memo()

#: Transposed-input front table: identity key -> ``(xt, pins)``.
XT_MEMO = Memo("fused_xt", _HOST_MEMO_ENTRIES)

#: The ``OUT_NODE`` VERDICT table: the identity tokens of the population's two
#: table tensors -> the first tagged live node (or :data:`_SERVED`, "clean").
#: The verdict costs one device reduction plus its sync, so it is memoized on
#: ``(data_ptr, shape, dtype, device, _version)`` of both tables -- the ABA-guarded
#: front-table identity key, which an in-place write invalidates -- and a
#: warm loop over one population pays it ONCE instead of per call.  Bounded like
#: the input table (``EVOGP_FUSED_HOST_MEMO``).
TAG_MEMO = Memo("fused_tag", _HOST_MEMO_ENTRIES)

#: The PREPARED-CALL table: the identity of every operand a call reads (the three
#: population tables and the data operand) plus the call's resolved geometry ->
#: one :class:`_Call`.  This is the one that makes a warm call a launch and
#: nothing else: the population's validation, the ``OUT_NODE`` verdict, the
#: envelope verdict, the resolved budget, the transposed data block and the loss
#: path's label view are all computed ONCE per (population, data, geometry) and
#: handed back on every later call.  A training loop re-evaluates one population
#: many times (elites, a fitness archive, two value calls in one generation) and
#: every benchmark loop does exactly that; a FRESH population is
#: one miss, which is the way the prepare is paid anyway.  Each entry holds strong
#: references to the tensors it keyed on (:func:`_identity`), so an entry can never
#: outlive the identity it recorded.  Bounded like the other tables.
PLAN_MEMO = Memo("fused_plan", _HOST_MEMO_ENTRIES)

#: This route's own counters (never folded into ``cache_stats()``).
_STATS: Dict[str, int] = {"fused_calls": 0, "fused_declines": 0}
_STATS_LOCK = threading.RLock()

#: Memo of the ENVELOPE VERDICT: ``(pop, tree_len, rows, out_len, holds_values,
#: block_rows, num_warps, budget)`` -> the decline message, or :data:`_SERVED`.
#: The key is the whole input of the verdict as RESOLVED (never a raw ambient
#: string), so it is a pure-function memo: a knob flip changes the key, nothing has
#: to invalidate it, and two threads may race only into recomputing one entry.
#: Bounded at :data:`_ENVELOPE_MEMO_MAX` entries by FIFO eviction (a GP run uses one
#: or two shapes; the bound protects a caller that sweeps them).
_ENVELOPE_MEMO: Dict[Tuple[int, int, int, int, bool, int, int, int], Any] = {}
_ENVELOPE_MEMO_MAX = 32
_ENVELOPE_MISS = object()
_SERVED = object()


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
    transposed-input table, ``fused_plan_hits`` / ``fused_plan_misses`` the
    prepared-call table (a hit is a warm call that skipped the whole prepare),
    ``fused_scratch_*`` the byte-budgeted table of cached stack buffers (allocs /
    reuses / evictions / entries / bytes), and ``fused_block_rows`` /
    ``fused_num_warps`` the geometry a call would resolve RIGHT NOW
    (INFORMATIONAL: ``-1`` when the ambient knob is unusable, because a stats call
    never raises -- :func:`_reported_knob`; the readers do).

    ``fused_calls`` / ``fused_declines`` are this facade's own attempt counters:
    nothing from another route reports into this dict and nothing here reports
    anywhere else.
    """
    with _STATS_LOCK:
        merged: Dict[str, int] = dict(_STATS)
    merged.update(XT_MEMO.stats())
    merged.update(TAG_MEMO.stats())
    merged.update(PLAN_MEMO.stats())
    merged.update(fused_kernels.scratch_stats())
    merged["fused_block_rows"] = _reported_knob(resolve_block_rows)
    merged["fused_num_warps"] = _reported_knob(resolve_num_warps)
    return merged


def clear_fused_cache() -> None:
    """Reset every counter and drop the transpose/verdict/plan/stack tables (memory only).

    This is the route's ONE reset entry: nothing else registers a reset hook for
    it, so a caller that wants a cold fused route calls exactly this.  Nothing here
    touches the disk.  Dropping :data:`_ENVELOPE_MEMO` / :data:`_GEOMETRY_MEMO` is
    hygiene, not correctness: their keys carry every input of the answer, so no
    entry can go stale.  Dropping :data:`PLAN_MEMO` -- and with it the single-slot
    fast path (:data:`_LAST`) -- is the route out of the one case its identity key
    does not see: a write into a population or data tensor that bypasses torch's
    version counter (a numpy view of the same memory, a foreign kernel), which no
    torch-level identity can detect.
    """
    with _STATS_LOCK:
        for name in list(_STATS):
            _STATS[name] = 0
        _ENVELOPE_MEMO.clear()
        _GEOMETRY_MEMO.clear()
    _drop_last()
    XT_MEMO.reset()
    TAG_MEMO.reset()
    PLAN_MEMO.reset()
    fused_kernels.clear_caches()


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


#: The backend spellings a Triton kernel can run on.  The route builds no graph,
#: so a backend here is only a declaration that the caller is on a device.
DEVICE_BACKENDS: Tuple[str, ...] = ("iree", "cuda", "rocm", "gpu", "torch")
#: The spellings that name a HOST reference interpreter instead.
REFERENCE_BACKENDS: Tuple[str, ...] = ("numpy",)


def _backend_name(backend: Optional[str]) -> Optional[str]:
    """The backend a call would run on, or ``None`` when the caller did not say.

    ``None`` means "no backend named": the fused route builds no graph, so it
    simply runs on the population tensors' own device.  A device spelling
    (:data:`DEVICE_BACKENDS`: ``iree`` / ``cuda`` / ``rocm`` / ``gpu`` / ``torch``,
    case- and whitespace-insensitive) is accepted and returned normalised; the
    reference interpreter (:data:`REFERENCE_BACKENDS`: ``numpy``) is returned too,
    so :func:`_decline_if_numpy` can decline it with its usual message.  Anything
    else is a caller error and raises ``ValueError`` naming the accepted ones.
    """
    if backend is None:
        return None
    name = str(backend).strip().lower()
    if name in DEVICE_BACKENDS or name in REFERENCE_BACKENDS:
        return name
    raise ValueError(
        f"unknown backend {backend!r}: the fused route accepts the device "
        f"spellings {list(DEVICE_BACKENDS)} and the reference interpreter "
        f"{list(REFERENCE_BACKENDS)} (or no backend at all)"
    )


def _decline_if_numpy(backend: Optional[str]) -> None:
    """A host backend is a decline: Triton has no numpy twin."""
    name = _backend_name(backend)
    if name is not None and name not in DEVICE_BACKENDS:
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


def _first_tagged_node(node_type: torch.Tensor, subtree_size: torch.Tensor) -> Any:
    """The first LIVE ``OUT_NODE``-tagged node, or :data:`_SERVED` when there is none.

    The device half of the single-output ``OUT_NODE`` rule: the single-output
    Triton kernel maps any type other than 0/1/2/3 onto its TERNARY branch (the
    oracle's fall-through), so a population built for ``output_len >= 2`` and
    launched as single-output would produce a plausible wrong number.  The rule
    itself (:func:`evogp._multi_output.check_multi_output`) is applied
    host-side from this answer, before any launch.

    Only the cells inside each row's own tree (``subtree_size[:, 0]``) are read:
    the padding behind a tree may hold whatever the generator left there, which is
    exactly what the kernel's ``n`` clamp ignores too.

    Returns:
        ``(row, node, raw_type)`` of the first tagged live node in prefix order, or
        :data:`_SERVED` (a negative answer is memoizable, ``None`` is not).
    """
    tree_len = int(node_type.shape[1])
    lengths = subtree_size[:, 0]
    live = torch.arange(tree_len, device=node_type.device)[None, :] < lengths[:, None]
    tagged = live & ((node_type & _OUT_NODE_BIT) != 0)
    if not bool(tagged.any()):
        return _SERVED
    row, node = (int(value) for value in tagged.nonzero()[0])
    return row, node, int(node_type[row, node])


def _refuse_out_node(
    node_type: torch.Tensor, subtree_size: torch.Tensor, output_len: int
) -> None:
    """Refuse a single-output call over a multi-output (``OUT_NODE``-tagged) population.

    Cheap in the steady state: the verdict is memoized on the identity tokens of
    the two population tables (:data:`TAG_MEMO`), so a warm loop over one
    population pays one device reduction instead of one per call, and a tagged
    population raises here -- before any allocation or launch -- instead of being
    silently evaluated as if every tagged node were a ternary ``if``.

    Raises:
        MultiOutputError: (a ``ValueError``) naming the row, the node and the raw
            type, for ``output_len == 1`` over a tagged population.
    """
    if int(output_len) != 1:
        return
    key = (token(node_type), token(subtree_size))
    if key[0] is None or key[1] is None:  # pragma: no cover - _prepare ensures tensors
        verdict = _first_tagged_node(node_type, subtree_size)
    else:
        verdict = TAG_MEMO.lookup(key)
        if verdict is None:
            verdict = _first_tagged_node(node_type, subtree_size)
            TAG_MEMO.store(key, verdict)
    if verdict is _SERVED:
        return
    raise MultiOutputError(out_node_message(*verdict))


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


def _same_device(left: torch.device, right: torch.device) -> bool:
    """Whether two torch devices name the same physical device (index-aware)."""
    if left.type != right.type:
        return False
    return (
        left.index is None
        or right.index is None
        or int(left.index) == int(right.index)
    )


def _out_destination(
    out: Any, shape: Tuple[int, ...], device: torch.device, name: str
) -> torch.Tensor:
    """Validate a caller-supplied ``out`` tensor the kernel writes into.

    The kernel addresses ``out`` with the same raw pointer arithmetic as its own
    buffer, so every property it relies on is checked here and a violation is a
    caller error (:class:`ValueError`), never a silently mis-strided write: a
    torch tensor, exactly ``shape``, float32, contiguous, ``requires_grad=False``
    and living on the computation device (an index-less spelling of the same
    device is accepted, as everywhere else in this module).
    """
    if not isinstance(out, torch.Tensor):
        raise ValueError(
            f"{name} must be a torch tensor (the fused route never stages host "
            f"memory), got {type(out).__name__}"
        )
    got = tuple(int(size) for size in out.shape)
    if got != tuple(int(size) for size in shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {got}")
    if out.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {out.dtype}")
    if not _same_device(out.device, torch.device(device)):
        raise ValueError(
            f"{name} lives on {out.device} but the population is on "
            f"{torch.device(device)}; the kernel writes in place and cannot cross "
            "devices"
        )
    if out.requires_grad:
        raise ValueError(
            f"{name} requires grad; the fused route writes into it in place"
        )
    if not out.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (the kernel writes a flat buffer), got "
            f"strides {out.stride()}"
        )
    return out


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
    allocates no value buffer, and only the loss path allocates a chart); ``rows`` is
    the DATA-row count (``1`` for ``per_row``, hence no ``per_row`` here), and
    unresolved knobs go through the resolvers, so a caller that knows nothing still
    gets a CONSERVATIVE check.  Returns the resolved budget in MiB so the entry
    points pass it down as ``max_scratch_mb=``: the scratch TABLE is budgeted by the
    same number the envelope was checked against.  The measured peak-allocated
    tracks the scratch term within 3.5 %, value buffer on top
    (``fused_results/ENVELOPE.md``).

    The verdict is a PURE function of the eight RESOLVED arguments
    (:func:`_envelope_verdict`), so it is memoized in :data:`_ENVELOPE_MEMO`: the
    key carries the resolved knobs, not the raw strings, so an in-process knob flip
    changes the key and a stale entry cannot exist (no invalidation hook).
    """
    block_rows = resolve_block_rows(block_rows)
    num_warps = resolve_num_warps(num_warps)
    budget = resolve_max_scratch_mb(max_scratch_mb)
    # No normalisation in the KEY: an ``int``/``numpy.int64``/``True`` spelling hashes
    # and compares equal, and :func:`_envelope_verdict` normalises the values it does
    # arithmetic on (a served call then pays one tuple build + one dict lookup).
    key = (pop, tree_len, rows, out_len, holds_values, block_rows, num_warps, budget)
    verdict = _ENVELOPE_MEMO.get(key, _ENVELOPE_MISS)
    if verdict is _ENVELOPE_MISS:
        verdict = _envelope_verdict(key) or _SERVED
        # Rare path, so the lock and the FIFO eviction cost nothing per served call.
        with _STATS_LOCK:
            if len(_ENVELOPE_MEMO) >= _ENVELOPE_MEMO_MAX:
                _ENVELOPE_MEMO.pop(next(iter(_ENVELOPE_MEMO)))
            _ENVELOPE_MEMO[key] = verdict
    if verdict is not _SERVED:
        raise FusedUnsupported(verdict)
    return budget


def _envelope_verdict(key: Tuple[int, int, int, int, bool, int, int, int]) -> Optional[str]:
    """The decline message of one :data:`_ENVELOPE_MEMO` key, or ``None`` when served.

    The byte formula of :func:`_check_envelope`, kept in one place: ``key`` is
    ``(pop, tree_len, rows, out_len, holds_values, block_rows, num_warps, budget)``,
    i.e. every input the verdict depends on, so this is a pure function.  The values
    are normalised HERE (the memo key is not), so a ``numpy`` integer reaching either
    public form of the check cannot change the arithmetic.
    """
    pop, tree_len, rows, out_len, holds_values, block_rows, num_warps, budget = key
    pop, tree_len, rows, out_len = int(pop), int(tree_len), int(rows), int(out_len)
    holds_values, block_rows, num_warps, budget = (
        bool(holds_values), int(block_rows), int(num_warps), int(budget),
    )
    # ``rows`` already encodes ``per_row`` (1 row); ``plan``'s flag does not move it.
    geom = fused_kernels.plan(rows, per_row=False, block_rows=block_rows,
                              num_warps=num_warps)
    n_rb, block = int(geom["n_rb"]), int(geom["block_rows"])
    scratch = pop * n_rb * tree_len * block * 4
    values = pop * rows * out_len * 4 if (holds_values or out_len > 1) else 0
    partials = 0 if holds_values else pop * n_rb * 4
    total = scratch + values + partials
    if total <= budget * _MIB:
        return None
    return (
        f"memory envelope exceeded: scratch {scratch / _MIB:.1f} MiB + value "
        f"buffer {values / _MIB:.1f} MiB + loss chart {partials / _MIB:.3f} MiB "
        f"= {total / _MIB:.1f} MiB > {MAX_SCRATCH_MB_ENV} {budget} MiB for "
        f"pop_size {pop} * max_tree_len {tree_len} * rows {rows} * out_len "
        f"{out_len} at {n_rb} row blocks of {block} lanes (raise it, or evaluate "
        "a smaller shape)"
    )


def envelope_decline(
    pop: int,
    tree_len: int,
    rows: int,
    *,
    out_len: int = 1,
    holds_values: bool = True,
    block_rows: Optional[int] = None,
    num_warps: Optional[int] = None,
    max_scratch_mb: Optional[int] = None,
) -> Optional[str]:
    """The memory envelope's verdict for a shape: the decline message, or ``None``.

    The NON-RAISING twin of :func:`_check_envelope` -- the same three-term byte
    rule through the same pure :func:`_envelope_verdict`, without the raise, the
    memo write or the resolved budget -- for callers that must DECIDE instead of
    call.  The gp stack and the check battery are those callers: they ask it
    before handing a shape to a launch that could not hold it.  Omitted knobs go
    through the same resolvers as a real call, so the verdict matches what
    :func:`evaluate_fused` would do,
    ``holds_values`` included (the single-output loss path allocates no value
    buffer and no chart).
    """
    key = (
        int(pop),
        int(tree_len),
        int(rows),
        int(out_len),
        bool(holds_values),
        resolve_block_rows(block_rows),
        resolve_num_warps(num_warps),
        resolve_max_scratch_mb(max_scratch_mb),
    )
    return _envelope_verdict(key)


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
    block_rows: int,
    num_warps: int,
) -> _Prepared:
    """Every check and normalisation the two entry points share.

    Deliberately pure: no ``x``/``inputs`` inspection (the value path and the loss
    path have different row semantics), no transpose, no allocation.  The ORDER is:
    the CHEAP SHAPE PROBES (:func:`_population_shape`, ``output_len``, ``tree_len``,
    ``pop_size``), then the BACKEND decline (:func:`_decline_if_numpy`: a Triton
    kernel has no numpy twin, so ``backend="numpy"`` is a decline, not an error),
    then the ENVIRONMENT (:func:`_require_available`), then the population's tensor
    checks and the multi-output (``OUT_NODE``) refusal (:func:`_refuse_out_node`).
    A malformed population therefore gets the ``ValueError`` it deserves instead of
    a decline.

    ``block_rows`` / ``num_warps`` are the call's ALREADY RESOLVED geometry
    (:func:`_geometry`, which both entry points resolve once and pass here, into the
    envelope and into the key build) -- they are never re-read from the environment
    in this function.
    """
    pop, tree_len = _population_shape(batch_subtree_size)
    if int(output_len) < 1:
        raise ValueError(f"output_len must be >= 1, got {int(output_len)}")
    if int(output_len) > MAX_OUT_LEN:
        raise FusedUnsupported(
            f"output_len={int(output_len)} > {MAX_OUT_LEN}: the multi-output kernel "
            f"accumulates in the (pop, rows, out_len) prediction buffer, so this "
            f"route serves up to {MAX_OUT_LEN} outputs (raise "
            "evogp.fused.engine.MAX_OUT_LEN if a model really needs more)"
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
    # The single-output protocol forbids the multi-output (``OUT_NODE``) tag, and
    # the kernel cannot say so: its type dispatch folds every type that is not
    # 0/1/2/3 into the TERNARY branch, so a population built for ``output_len >= 2``
    # would be evaluated as if each tagged node were an ``if`` -- a plausible wrong
    # number, no exception.  The shared rule (``evogp._multi_output.check_multi_output``,
    # the one every evaluator of this library applies) therefore runs HERE, on the
    # host, before any launch.
    _refuse_out_node(node_type, subtree_size, output_len)
    return _Prepared(
        node_value=node_value,
        node_type=node_type,
        subtree_size=subtree_size,
        pop=pop,
        tree_len=tree_len,
        device=device,
        block_rows=int(block_rows),
        num_warps=int(num_warps),
    )


# --------------------------------------------------------------------------- #
# the prepared-call table
# --------------------------------------------------------------------------- #
class _Call:
    """One fully prepared call, memoized in :data:`PLAN_MEMO`.

    Everything a warm call needs besides the launch itself: the population's three
    normalized tables and their geometry (from :func:`_prepare`), the data operand
    the kernel reads (the transposed ``x``, or ``x`` itself on the ``per_row``
    path), the loss path's label operand and data-row count, the resolved memory
    budget, the shape a caller-supplied ``out`` must have, the strong references
    that make the identity key of :func:`_identity` ABA-proof, and
    :attr:`run` -- the launch BOUND to all of it (``fused_kernels.bind_value`` /
    ``bind_loss``: one kernel specialization, one option dict, one scratch block),
    so a warm call is ``call.run(destination)`` and nothing else.

    It also carries the facts :func:`_last_call` compares for the single-entry fast
    path (:data:`_LAST`): the call's ``kind``, the RAW operands it was prepared from
    (``x_raw`` / ``labels_raw``), their ``_version`` vector, the resolved scalars and
    the resolved geometry.  Those are exactly the key's inputs, so a fast-path hit
    proves what a :data:`PLAN_MEMO` hit proves.
    """

    __slots__ = (
        "node_value",
        "node_type",
        "subtree_size",
        "pop",
        "tree_len",
        "device",
        "block_rows",
        "num_warps",
        "rows",
        "per_row",
        "data",
        "labels",
        "budget_mb",
        "out_shape",
        "pins",
        "run",
        "kind",
        "x_raw",
        "labels_raw",
        "versions",
        "input_len",
        "output_len",
        "use_mse",
        "backend",
        "geometry",
    )

    def __init__(
        self,
        prepared: _Prepared,
        *,
        kind: str,
        rows: int,
        per_row: bool,
        data: torch.Tensor,
        labels: Optional[torch.Tensor],
        x_raw: Any,
        labels_raw: Any,
        input_len: int,
        output_len: int,
        use_mse: bool,
        backend: Optional[str],
        geometry: Tuple[int, int, int],
        budget_mb: int,
        out_shape: Tuple[int, ...],
        pins: Tuple[Any, ...],
        run: Callable[..., Any],
        block_rows: int,
        num_warps: int,
    ) -> None:
        self.node_value = prepared.node_value
        self.node_type = prepared.node_type
        self.subtree_size = prepared.subtree_size
        self.pop = prepared.pop
        self.tree_len = prepared.tree_len
        self.device = prepared.device
        self.block_rows = int(block_rows)
        self.num_warps = int(num_warps)
        self.rows = int(rows)
        self.per_row = bool(per_row)
        self.data = data
        self.labels = labels
        self.budget_mb = int(budget_mb)
        self.out_shape = out_shape
        self.pins = pins
        self.run = run
        self.kind = kind
        self.x_raw = x_raw
        self.labels_raw = labels_raw
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.use_mse = bool(use_mse)
        self.backend = backend
        self.geometry = geometry
        # The identity vector of :func:`_last_call`, recorded where the entry is
        # built (i.e. once per prepared call, never per call).
        versions = [
            prepared.node_value._version,
            prepared.node_type._version,
            prepared.subtree_size._version,
            x_raw._version,
        ]
        if labels_raw is not None:
            versions.append(labels_raw._version)
        self.versions = tuple(versions)


#: The single-entry FAST PATH in front of :data:`PLAN_MEMO`: the prepared call
#: served last.  A training loop, an elite re-evaluation and every benchmark loop
#: call with the very same tensors again, so the whole plan key -- four
#: ``(id, _version)`` pairs, three resolved knobs, a tuple build and the table's
#: lock -- can be replaced by one short-circuit identity scan (:func:`_last_call`):
#: ~0.4 us instead of ~3.7.  It is not a second table and not a second semantics:
#: the entry is a plain :data:`PLAN_MEMO` value, the comparison proves exactly what
#: the key proves (same tensor OBJECTS -- kept alive by the entry's ``pins``, same
#: ``_version`` vector, same resolved scalars and geometry), a hit is counted as a
#: plan hit, and ``clear_fused_cache()`` drops it with the table.
_LAST: Optional[_Call] = None


def _last_call(
    kind: str,
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    x_raw: Any,
    labels_raw: Any,
    input_len: int,
    output_len: int,
    use_mse: bool,
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
    max_scratch_mb: Optional[int],
) -> Optional[_Call]:
    """The :data:`_LAST` entry when it IS the call being made, else ``None``.

    Written as one short-circuit chain, cheapest comparison first (object identity,
    then the resolved scalars, then the ``_version`` vector, then the geometry -- the
    only part that reads the environment).  Every operand is a ``torch.Tensor`` here
    by construction: a non-tensor operand can never have produced a ``_LAST`` entry
    (its key was ``None``) and the ``is`` test rejects it anyway.
    """
    call = _LAST
    if call is None or call.kind != kind:
        return None
    if (
        call.node_value is not batch_node_value
        or call.node_type is not batch_node_type
        or call.subtree_size is not batch_subtree_size
        or call.x_raw is not x_raw
        or call.labels_raw is not labels_raw
        or call.input_len != input_len
        or call.output_len != output_len
        or call.use_mse != use_mse
        or call.backend != backend
    ):
        return None
    versions = [
        batch_node_value._version,
        batch_node_type._version,
        batch_subtree_size._version,
        x_raw._version,
    ]
    if labels_raw is not None:
        versions.append(labels_raw._version)
    if call.versions != tuple(versions):
        return None
    if call.geometry != _geometry(block_rows, num_warps, max_scratch_mb):
        return None
    PLAN_MEMO.note_hit()
    return call


def _drop_last() -> None:
    """Forget the fast path's entry (``clear_fused_cache`` / ``clear_cache``)."""
    global _LAST
    _LAST = None


def _identity(tensor: Any) -> Optional[Tuple[int, int]]:
    """``(id, _version)`` of one operand, or ``None`` when it cannot be memoized.

    The identity a :data:`PLAN_MEMO` entry is keyed on.  ``id`` is the tensor
    OBJECT, not its ``data_ptr``: every entry keeps a strong reference to the
    tensors it keyed on (:attr:`_Call.pins`), so the object (and therefore its
    storage and its ``id``) cannot be recycled by a different tensor while the
    entry lives -- the ABA argument :func:`evogp.fused.memo.pin` makes for
    ``data_ptr``, one step stronger, and 0.15 us instead of the 1.1 us the full
    ``(data_ptr, shape, dtype, device, _version)`` token costs (this key is built
    on every warm call).  ``_version`` is bumped by every in-place torch write, so
    a mutated population or dataset can never be served from a stale entry.

    Not detected by design (exactly like the front tables): a write that
    bypasses torch's version counter, e.g. through a ``numpy`` view of the same
    memory or a foreign kernel.  Call :func:`clear_fused_cache` (or use torch ops)
    after such a write.  ``None`` for anything that is not a ``torch.Tensor``, so
    a caller can always fall back to doing the whole prepare.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return (id(tensor), int(tensor._version))
    except Exception:  # pragma: no cover - exotic tensor subclasses only
        return None


def _operand_key(tensors: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
    """The identity half of a plan key, or ``None`` when an operand is not memoizable."""
    parts: list = []
    for tensor in tensors:
        identity = _identity(tensor)
        if identity is None:
            return None
        parts.append(identity)
    return tuple(parts)


def _value_key(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    x: Any,
    input_len: int,
    output_len: int,
    per_row: bool,
    backend: Optional[str],
    geometry: Tuple[int, int, int],
) -> Optional[Tuple[Any, ...]]:
    """Key of the value path's :data:`PLAN_MEMO` entry, or ``None`` (never memoized).

    The operand identities plus EVERY input the prepare and the envelope verdict
    read, resolved (never a raw ambient string), so an entry cannot go stale:

    * ``input_len`` / ``output_len`` / ``per_row`` -- the shape contract;
    * the ``backend`` SPELLING as passed (an explicit ``numpy`` declines, so
      it must be part of the key rather than re-checked per call; the raw spelling
      is enough -- two spellings of one backend get two entries, which is harmless
      -- and resolving the name costs 3.4 us, more than the entry saves);
    * ``geometry`` -- the call's already resolved
      ``(block_rows, num_warps, max_scratch_mb)`` triple (:func:`_geometry`), so
      flipping a knob between two calls changes the key and can never inherit an old
      verdict or an old geometry.  It is resolved ONCE per call by the entry point
      and handed to the key, the prepare and the envelope, instead of each of them
      re-reading the environment.

    An argument the key cannot normalise (a non-string ``backend``, a non-integer
    ``input_len``) answers ``None``: the caller then takes the full path, which
    raises the error it always raised, with the message it always had.
    """
    operands = _operand_key(
        (batch_node_value, batch_node_type, batch_subtree_size, x)
    )
    if operands is None:
        return None
    if backend is not None and not isinstance(backend, str):
        return None
    try:
        scalars = (
            "value",
            bool(per_row),
            int(input_len),
            int(output_len),
            backend,
        )
    except (TypeError, ValueError):
        return None
    return operands + scalars + geometry


def _loss_key(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    inputs: Any,
    labels: Any,
    input_len: int,
    output_len: int,
    use_MSE: bool,
    backend: Optional[str],
    geometry: Tuple[int, int, int],
) -> Optional[Tuple[Any, ...]]:
    """:data:`PLAN_MEMO` key of the loss path (:func:`_value_key` plus the labels).

    ``use_MSE`` is in the key because it is a kernel constexpr (a different loss,
    not a different launch of the same one), and the label operand is keyed like
    every other input.  The backend is keyed as the raw SPELLING and the geometry as
    the call's already resolved triple, for the reasons :func:`_value_key` gives.
    """
    operands = _operand_key(
        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            inputs,
            labels,
        )
    )
    if operands is None:
        return None
    if backend is not None and not isinstance(backend, str):
        return None
    try:
        scalars = (
            "loss",
            int(input_len),
            int(output_len),
            bool(use_MSE),
            backend,
        )
    except (TypeError, ValueError):
        return None
    return operands + scalars + geometry


def _value_call(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    x: Any,
    per_row: bool,
    backend: Optional[str],
    geometry: Tuple[int, int, int],
) -> _Call:
    """Build (and validate) the value path's prepared call -- a :data:`PLAN_MEMO` miss.

    The body of the value path up to the launch, exactly as it was before the
    table existed: :func:`_prepare`, then ``x``'s contract, then the envelope, then
    the one prep step (:func:`_transposed`).  Raises exactly what it always raised.
    ``geometry`` is the call's resolved ``(block_rows, num_warps, max_scratch_mb)``
    triple (:func:`_geometry`), resolved once by the entry point.
    """
    raw_x = x
    prepared = _prepare(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        backend,
        geometry[0],
        geometry[1],
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
        prepared.pop, prepared.tree_len, rows, geometry[2],
        block_rows=prepared.block_rows, num_warps=prepared.num_warps,
        out_len=int(output_len), holds_values=True,
    )
    data = x if per_row else _transposed(x)
    return _Call(
        prepared,
        kind="value",
        rows=rows,
        per_row=per_row,
        data=data,
        labels=None,
        x_raw=raw_x,
        labels_raw=None,
        input_len=int(input_len),
        output_len=int(output_len),
        use_mse=False,
        backend=backend,
        geometry=geometry,
        budget_mb=budget_mb,
        out_shape=(prepared.pop, rows, int(output_len)),
        pins=(batch_node_value, batch_node_type, batch_subtree_size, raw_x, x),
        run=_launch(
            fused_kernels.bind_value,
            prepared.node_value,
            prepared.node_type,
            prepared.subtree_size,
            data,
            input_len=int(input_len),
            tree_len=prepared.tree_len,
            rows=rows,
            per_row=bool(per_row),
            block_rows=prepared.block_rows,
            num_warps=prepared.num_warps,
            out_len=int(output_len),
            max_scratch_mb=budget_mb,
        ),
        block_rows=prepared.block_rows,
        num_warps=prepared.num_warps,
    )


def _loss_call(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
    inputs: Any,
    labels: Any,
    use_MSE: bool,
    backend: Optional[str],
    geometry: Tuple[int, int, int],
) -> _Call:
    """Build (and validate) the loss path's prepared call -- a :data:`PLAN_MEMO` miss.

    The body of the loss path up to the launch, exactly as it was before the table
    existed: :func:`_prepare`, then ``inputs``/``labels``' contract, then the
    envelope (no value buffer on the single-output path, hence
    ``holds_values=False``), then the one prep step and the flat label view.
    ``geometry`` is the call's resolved ``(block_rows, num_warps, max_scratch_mb)``
    triple (:func:`_geometry`), resolved once by the entry point.
    """
    raw_inputs, raw_labels = inputs, labels
    prepared = _prepare(
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        input_len,
        output_len,
        backend,
        geometry[0],
        geometry[1],
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
        prepared.pop, prepared.tree_len, rows, geometry[2],
        block_rows=prepared.block_rows, num_warps=prepared.num_warps,
        out_len=int(output_len), holds_values=False,
    )
    data = _transposed(inputs)
    # ``out_len == 1`` wants the flat ``(N,)`` view (a free reshape of the now
    # contiguous label column); the multi-output kernel indexes ``(N, out_len)``.
    label_view = label_column.reshape(-1) if int(output_len) == 1 else label_column
    return _Call(
        prepared,
        kind="loss",
        rows=rows,
        per_row=False,
        data=data,
        labels=label_view,
        x_raw=raw_inputs,
        labels_raw=raw_labels,
        input_len=int(input_len),
        output_len=int(output_len),
        use_mse=bool(use_MSE),
        backend=backend,
        geometry=geometry,
        budget_mb=budget_mb,
        out_shape=(prepared.pop,),
        # The pins are the tensors the KEY was built from (the raw operands), not
        # the normalized copies ``_Call.data``/``_Call.labels`` already hold.
        pins=(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            raw_inputs,
            raw_labels,
        ),
        run=_launch(
            fused_kernels.bind_loss,
            prepared.node_value,
            prepared.node_type,
            prepared.subtree_size,
            data,
            label_view,
            input_len=int(input_len),
            tree_len=prepared.tree_len,
            rows=rows,
            use_mse=bool(use_MSE),
            block_rows=prepared.block_rows,
            num_warps=prepared.num_warps,
            out_len=int(output_len),
            max_scratch_mb=budget_mb,
        ),
        block_rows=prepared.block_rows,
        num_warps=prepared.num_warps,
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
    out: Optional[Any] = None,
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

    ``out`` (optional) is a caller-supplied float32 destination the kernel writes
    the ``(pop_size, rows, output_len)`` value buffer into instead of allocating
    one; the return value then shares its storage.  It is the destination of the
    KERNEL, so it always has the 3-D kernel shape -- ``per_row=True`` returns
    ``out.squeeze(1)`` (a view, still zero-copy).  Nothing is allocated for the
    output when it is given, which is what lets a device loop hand the kernel the
    buffer another runtime will read.

    Host crossings: NONE in the steady state.  ``x`` is transposed once per
    distinct dataset tensor (:func:`_transposed`, an identity-keyed table hit
    afterwards) and the value buffer is the kernel's only output -- no decode, no
    key derivation, no readback, no host assembly.

    Raises:
        FusedUnsupported: the route cannot serve the call (see the class).
        ValueError: shape contract violations, or an ``out`` that is not a
            contiguous float32 tensor of the kernel's shape on the computation
            device.
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
            out=out,
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
    out: Optional[Any],
    out_device: Optional[Any],
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
    max_scratch_mb: Optional[int],
) -> torch.Tensor:
    """Body of :func:`evaluate_fused` (the decline bookkeeping is in the wrapper).

    Two layers: the prepared-call TABLE (:data:`PLAN_MEMO`, keyed by
    :func:`_value_key`) and the launch.  A warm call -- the same population and the
    same dataset again, which is what a training loop, an elite re-evaluation and
    every benchmark loop do -- is one key build, one dict lookup and the launch;
    everything the prepare used to redo per call happens on a miss only.
    """
    global _LAST
    # Fast path first: the call this route served last, again (a training loop, an
    # elite re-evaluation, a benchmark loop).  One identity scan instead of the key
    # build + the table lookup; see :func:`_last_call`.
    call = _last_call(
        "value",
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        x,
        None,
        input_len,
        output_len,
        False,
        backend,
        block_rows,
        num_warps,
        max_scratch_mb,
    )
    if call is None:
        geometry = _geometry(block_rows, num_warps, max_scratch_mb)
        key = _value_key(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            x,
            input_len,
            output_len,
            per_row,
            backend,
            geometry,
        )
        call = PLAN_MEMO.lookup(key) if key is not None else None
        if call is None:
            call = _value_call(
                batch_node_value,
                batch_node_type,
                batch_subtree_size,
                input_len,
                output_len,
                x,
                per_row,
                backend,
                geometry,
            )
            if key is not None:
                PLAN_MEMO.store(key, call)
        _LAST = call
    # A caller-supplied destination is validated BEFORE the launch (fail fast,
    # never after the kernel has run) and is per CALL, so it is never memoized.
    destination = None
    if out is not None:
        if out_device is not None and torch.device(out_device) != torch.device(
            call.device
        ):
            raise ValueError(
                f"out lives on {call.device} but out_device={out_device!r} asks "
                "for the result elsewhere; pass one or the other"
            )
        destination = _out_destination(out, call.out_shape, call.device, "out")
    result = call.run(destination)
    if call.per_row:
        # (pop, 1, out_len) -> (pop, out_len) view of the value buffer
        result = result.squeeze(1)
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
    out: Optional[Any] = None,
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
    order as the oracle's ``sr_fitness_from_rows`` (``test/oracle/reference_eval.py``).

    The kernel reduces each (tree, row-block) to ONE partial, so the only device
    output of the pass is the ``(pop, n_rb)`` chart: summing it and dividing by
    ``N`` are two tiny device ops and no prediction ever crosses to the host.

    ``out`` (optional) is a caller-supplied ``(pop_size,)`` float32 device tensor
    the epilogue writes the finished loss INTO (sum into it, then divide in
    place) instead of allocating the result; the return value is that same
    tensor.  It saves one ``(pop_size,)`` allocation and, for a caller that
    already owns a buffer another runtime reads, the device copy into it.  The
    loss still crosses no boundary; the tensor must be contiguous, float32, on
    the computation device and must not require grad (a caller error otherwise).

    Raises:
        FusedUnsupported: the route cannot serve the call.
        ValueError: shape contract violations (mismatched rows/columns), or an
            ``out`` that does not satisfy the contract above.
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
            out=out,
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
    out: Optional[Any],
    out_device: Optional[Any],
    backend: Optional[str],
    block_rows: Optional[int],
    num_warps: Optional[int],
    max_scratch_mb: Optional[int],
) -> torch.Tensor:
    """Body of :func:`sr_fitness_fused` (the decline bookkeeping is in the wrapper).

    Same two layers as the value path: the prepared-call table
    (:data:`PLAN_MEMO`, keyed by :func:`_loss_key`, so the label view and the
    transposed input block are part of the entry) and the launch.
    """
    global _LAST
    # Fast path first, exactly as on the value path (a loss call and a value call
    # of the same population are different ``kind``s, so they cannot cross-hit).
    call = _last_call(
        "loss",
        batch_node_value,
        batch_node_type,
        batch_subtree_size,
        inputs,
        labels,
        input_len,
        output_len,
        bool(use_MSE),
        backend,
        block_rows,
        num_warps,
        max_scratch_mb,
    )
    if call is None:
        geometry = _geometry(block_rows, num_warps, max_scratch_mb)
        key = _loss_key(
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            inputs,
            labels,
            input_len,
            output_len,
            use_MSE,
            backend,
            geometry,
        )
        call = PLAN_MEMO.lookup(key) if key is not None else None
        if call is None:
            call = _loss_call(
                batch_node_value,
                batch_node_type,
                batch_subtree_size,
                input_len,
                output_len,
                inputs,
                labels,
                use_MSE,
                backend,
                geometry,
            )
            if key is not None:
                PLAN_MEMO.store(key, call)
        _LAST = call
    # A caller-supplied destination is validated BEFORE the launch (fail fast,
    # never after the kernel has run) and is per CALL, so it is never memoized.
    destination = None
    if out is not None:
        if out_device is not None and torch.device(out_device) != torch.device(
            call.device
        ):
            raise ValueError(
                f"out lives on {call.device} but out_device={out_device!r} asks "
                "for the result elsewhere; pass one or the other"
            )
        destination = _out_destination(out, (call.pop,), call.device, "out")
    partials = call.run()
    # ``sum_points sum_outputs err / N`` -- outputs are already summed (there is
    # exactly one), the divisor is the data-point count only.  A caller-supplied
    # destination gets both steps IN PLACE, so the finished loss is written
    # straight into the buffer another runtime reads (no allocation, no copy).
    if destination is None:
        loss = partials.sum(dim=1) / float(call.rows)
        if out_device is not None and torch.device(out_device) != loss.device:
            loss = loss.to(out_device)
        return loss.contiguous()
    torch.sum(partials, dim=1, out=destination)
    destination.div_(float(call.rows))
    return destination
