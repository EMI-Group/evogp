"""Triton kernels of the FUSED whole-population evaluator (route ``fused``).

This module is the device half of the fused evaluation route: it turns the
population's already-resident prefix-order tensors into ONE pass of Triton
programs with no host readback, no host decode and no per-node Python work.
The policy layer (declines, knobs, counters, the transposed-input memo) lives in
:mod:`evogp.tree.fused_engine`; the dev battery lives in
:mod:`evogp.tree.fused_check`.

Why a fused kernel at all
-------------------------
The dense route (``dense_engine.py``) evaluates a population with ONE evoxc graph
whose per-position op chain is unrolled over the *whole trace*: every node
position writes a ``(P, N)`` candidate buffer and every later node re-reads the
operands from that buffer, so the graph moves ``O(L^2)`` populated tensor
elements for a tree of ``L`` positions (and it is dispatch-bound at ~310 chained
device ops for the whole population -- see ``DENSE_EVAL.md`` §5).  A plain-torch
whole-population implementation (the ``torch-batched`` benchmark arm) has the
same shape of traffic: a ``(pop, L+1, N)`` buffer, one pass per position.

The prefix encoding gives a much better algorithm.  In pre-order the FIRST child
of node ``j`` is ALWAYS node ``j+1``, so walking the tree in REVERSE prefix order
(a children-before-parent walk) keeps the leftmost operand in a register: the
value that was produced by the immediately preceding loop iteration.  Only the
remaining operands (one for a binary node, two for a ternary) come off a small
explicit stack.  Consequence, and this is the traffic argument of the whole
route:

* per (tree, row-block) the walk touches the node tables ONCE per node
  (2 scalar loads: type + value) and the input rows only at VAR nodes;
* the ``(L+1)`` value buffer becomes a register (``prev``) plus a stack whose
  live depth is bounded by the tree DEPTH, not by ``L``;
* so the traffic per program is ``O(n_nodes)`` scalar loads + ``O(var_nodes)``
  vector loads + ``O(func_nodes)`` stack round trips, all of which are L1/L2
  sized (a program's scratch is ``L * BLOCK_R * 4`` bytes = 64 KiB at
  ``L = 64``, ``BLOCK_R = 256``, and only the live ``depth`` slots are touched),
  instead of ``O(L)`` full-population passes.

Measured on the six benchmark cells the fused route is 48.6x-167.6x faster than
the ``torch-batched`` arm on the value path (0.083-0.157 ms against 4.13-26.34 ms)
and 43.4x-156.4x on the loss path, and the part that is left is the launch
itself: the kernel is 72-82 % of a warm call and the host glue 0.023-0.033 ms
(numbers of record, with the ``BLOCK_R``/warp sweep: the module docstring of
:mod:`evogp.tree.fused_engine`, reproduced by ``fused_check.py``).

Geometry
--------
``grid = (pop, n_rb)`` with ``n_rb = ceil(rows / BLOCK_R)``: one program per
(tree, row-block).  The row axis is the vector axis (``BLOCK_R`` lanes, one
element per lane), the tree is walked scalar-ly inside the program.  That maps
well onto a GPU whose parallel axes are (population x rows / BLOCK_R): at
``pop 256``/``N 1024``/``BLOCK_R 256`` there are 1 024 programs of 64-iterations
each, i.e. thousands of independent dependency chains to hide the per-node
scalar-load latency behind.  ``BLOCK_R`` is ``min(next_pow2(rows), cap)`` with
``cap = EVOGP_FUSED_BLOCK_ROWS`` (default 256, rounded DOWN to a power of two --
``tl.arange`` needs one) and the launch is ``num_warps = EVOGP_FUSED_NUM_WARPS``
(default 1); the measured sweep is in ``fused_engine``'s docstring.

The kernel body
---------------
Every iteration computes ONE node value for the whole row block::

    tt = node type at j            (0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, else TFUNC)
    if   tt == 0:  val = x[vi]                 (per_row: scalar load x[t,vi];
                                                batched: coalesced load from the
                                                PRETRANSPOSED xt[vi, r])
    elif tt == 1:  val = const                 (broadcast of the node value)
    elif tt == 2:  val = unary(fid, prev)
    elif tt == 3:  val = binary(fid, prev, stack[sp-1]); sp -= 1
    else:          val = if(prev, stack[sp-1], stack[sp-2]); sp -= 2
    j -= 1
    # the lookahead decides where `val` goes: node j is the parent of this one
    # only when it is a FUNCTION node (its first child is this node), otherwise
    # this value is a pending operand and must go on the stack.
    if node_type[j] is a leaf: stack[sp] = val; sp += 1
    else:                      prev = val

The lookahead load is the same load the next iteration needs, so the walk costs
ONE ``node_type`` load and at most one ``node_value`` load per node.  The type
mapping mirrors ``reference_eval._kind_from_type`` exactly: anything that is not
0/1/2/3 -- including a type carrying the multi-output ``OUT_NODE`` bit in
single-output mode -- is a ternary ``if`` node, and a ternary node ignores its
function id.

Malformed rows (a tree that underflows the stack, an absurd ``subtree_size``, a
VAR index out of range) never read out of bounds and never read a previous
call's bytes: ``n = min(max(subtree_size[0], 0), tree_len)`` CLAMPS the row's own
length before it drives ``range(0, n)``, the first loop index and the child
lookahead, so an oversized ``subtree_size`` walks this row only instead of
spilling into the next one (and past the allocation on the last row); VAR loads
are masked with ``(vi >= 0) & (vi < input_len) & rmask`` and read ``0.0``
outside; and a stack POP ON AN EMPTY STACK reads nothing at all
(``mask=rmask & (sp >= 1)``/``(sp >= 2)``, ``other=0.0``), so the operand is a
deterministic ``0.0`` rather than whatever the shared scratch block held from an
earlier call.  ``stack_cap == tree_len`` so the deepest legal tree (a
right-leaning chain) still fits.  A malformed tree therefore produces a
defined-but-arbitrary number, which is why the route DECLINES nothing on
structure: the specialized engine (``evoxc_engine.decode_population``) owns the
malformed-tree ``ValueError`` contract, and the wiring round falls back to it
before this kernel is reached.

Numeric contract (bit-for-bit with ``src/evogp/tests/reference_eval.py``)
-----------------------------------------------------------------------
Every node value is the correctly rounded float32 result of the op on float32
operands.  Two rules make that true on the device:

1. **No FMA contraction.** The launch passes ``enable_fp_fusion=False``.  LLVM
   contracts ``a*b`` into a following ``add``/``sub`` under the default contract
   flag, which changes ``a*b + c`` on 24% of random inputs (measured: 245 257 of
   1 048 576 values differ by up to 1 ulp) while the oracle rounds the product
   to float32 before the add.  With fusion off, ``+ - * /`` are bitwise equal to
   the oracle's ``round32(f64(a) op f64(b))`` because float32 arithmetic is
   already correctly rounded and the double rounding through float64 is
   innocuous at 53 >= 2p+2 bits.
2. **The transcendentals are computed in float64 and rounded back**
   (``round32(f64op(f64(a)))``), exactly like ``reference_eval._apply`` and the
   evoxc emitters: ``sin cos tan sinh cosh tanh log exp pow sqrt``.  ``sqrt``
   needs this too -- the hardware float32 ``sqrt`` is correctly rounded but
   torch's float32 ``sqrt`` is NOT (it disagrees on 16% of inputs, measured), so
   the oracle's ``round32(sqrt(f64(a)))`` is the reference.  ``sinh``/``cosh``
   use the composed ``(exp(x) -+ exp(-x)) * 0.5`` form (evoxc has no
   ``sinh``/``cosh``, and the composed form is the emitters' semantics of
   record), with ``|a| < 1e-4`` returned as ``a`` so the sign of zero survives
   (``loose_inv(-0.0) = -1e9``).  -- measured on 65 536 random inputs: all
   twelve float64 specializations agree with numpy's float64 result bit-for-bit
   after the cast, i.e. the ladder is exact at the f32 rounding boundary and the
   only residual disagreement is the oracle's own float64 library entry point.
   Nothing weaker than the arithmetic path: ``+ - * /``, the comparisons, ``abs``
   and ``neg`` stay float32 (bit-exact ``+-1`` for the comparisons), and
   ``copysign(1e-9, y)`` reads the SIGN BIT (``bitcast<int32>() < 0``), never
   ``y >= 0``, so ``-0.0`` clamps to ``-1e-9``.

The SR loss is accumulated in-graph (``d = label[r] - val; acc += d*d`` or
``|d|``, one ``tl.sum`` per program, one ``tl.store`` per program), so the loss
path's only device output is the ``(pop, n_rb)`` partial chart -- ``1/rows`` of
the traffic of the value path, and no prediction ever leaves the device.

Multi-output (`OUT_NODE`)
-------------------------
``output_len >= 2`` selects :func:`_fused_multi_kernel`, a second kernel (the
single-output one is untouched, so its six benchmark cells keep their numbers).
It is the same walk plus the two rules of the legacy ``treeGPEvalKernel<true>``:
the base kind is ``node_type & 0x7F`` and a tagged FUNCTION node resolves its
operator from the packed value bits (``function = bits & 0xFFFF``,
``outIndex = (bits >> 16) & 0xFFFF``) and adds its own value into
``acc[outIndex]`` (an index ``>= out_len`` is dropped -- the oracle's unsigned
compare), while every node passes its RIGHT-MOST operand up to its parent
instead of its own value.  Tagged leaves accumulate nothing and untagged outputs
stay ``0.0``.  The accumulator IS the prediction buffer (``(pop, rows, out_len)``,
zeroed in-kernel and read-modify-written at each tagged node) rather than a
``[out_len, BLOCK_R]`` register tile, which would make triton unify two layouts
across the loop phis; the value epilogue therefore already holds the result, the
loss epilogue loads the ``(rows, out_len)`` label tile, sums over both axes and
still divides by ``rows`` only.

Laziness
--------
``import evogp.tree`` must keep working on a box with no triton and no GPU, and
triton's import is heavy, so **nothing here imports triton at module import
time**.  The kernel bodies below are ordinary module-level functions (so
``inspect.getsource`` sees them and their ``__globals__`` is this module's
dict); :func:`kernels` imports triton on first use, publishes ``tl`` /
``libdevice`` / ``triton`` into the module globals and REPLACES the names of the
helper bodies with their ``triton.jit`` wrapped versions, which is what the
compiled kernel resolves its callees through.  Call :func:`kernels` (or anything
in :mod:`evogp.tree.fused_engine`) before touching the functions below.

Triton writes its on-disk cache under ``~/.triton`` by default; on a box whose
home is read-only set ``TRITON_CACHE_DIR`` to a writable path (this box needs
it: compiling into ``/root/.triton`` raises ``OSError: Read-only file system``).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

from .utils import DELTA, FUNCS, Func

__all__ = [
    "BLOCK_ROWS_ENV",
    "DEFAULT_BLOCK_ROWS",
    "DEFAULT_NUM_WARPS",
    "DEFAULT_SCRATCH_BUDGET_MB",
    "FusedUnavailable",
    "clear_caches",
    "emitted_function_ids",
    "kernels",
    "launch_loss",
    "launch_value",
    "plan",
    "scratch_stats",
    "triton_available",
]

#: Cap of the row-block width, ``min(next_pow2(rows), cap)`` lanes per program.
BLOCK_ROWS_ENV = "EVOGP_FUSED_BLOCK_ROWS"
#: Lanes per program when the caller does not say otherwise.  Measured optimum:
#: in the 64/128/256/512 x warps-1/2/4 sweep (six cells, one process per config,
#: :mod:`evogp.tree.fused_check --stage tuning`) 256 is the only width that lands
#: within 4.3 % of the per-cell best on EVERY cell, and it is the per-cell best on
#: the two heaviest ones.
DEFAULT_BLOCK_ROWS = 256
#: Warps per program (AMD wavefront = 64 lanes, so 1 warp == a 256-lane block
#: being processed four lanes per thread; on NVIDIA 32 lanes/scheduler it is eight).
#: Measured optimum at ``BLOCK_R`` 256, and the sweep's surprising result: the
#: extra warps of the old default (4) only add scheduling overhead here, because
#: the tree walk is a scalar, loop-carried dependency chain -- every lane of the
#: block executes the SAME instruction sequence, so more warps cannot overlap
#: anything.  1 warp is 1.7x faster than 4 on ``p4096-L64-i32-N1024``
#: (0.134 vs 0.225 ms) and 1.8x on ``p2048-L128-i32-N1024-d7`` (0.161 vs 0.287 ms)
#: while being within 4 % of the 4-warp number on the small cells, where the call
#: is launch-bound rather than execution-bound.
DEFAULT_NUM_WARPS = 1


class FusedUnavailable(RuntimeError):
    """Triton (or a usable device) is not available at all.

    Distinct from :class:`evogp.tree.fused_engine.FusedUnsupported`, which is
    the per-call "this population cannot use the fused route" decline: this one
    is an environment fact and is raised only by :func:`kernels`.
    """


# --------------------------------------------------------------------------- #
# the operator table: data here, ``if/elif`` chains in the kernels, one check
# --------------------------------------------------------------------------- #
# ``utils.FUNCS`` is the single source of truth for the function-id vocabulary, so
# the id sets the kernels emit are declared here against ``utils.Func`` and
# cross-checked at import (see :func:`_verify_operator_table`).  A kernel body can
# only read ``tl.constexpr`` globals, so the branches themselves spell the ids out
# as literals -- which is exactly why the literals need a guard.
_EMITTED_BINARY_IDS: Tuple[int, ...] = (
    Func.ADD,
    Func.SUB,
    Func.MUL,
    Func.DIV,
    Func.LOOSE_DIV,
    Func.POW,
    Func.LOOSE_POW,
    Func.MAX,
    Func.MIN,
    Func.LT,
    Func.GT,
    Func.LE,
    Func.GE,
)
_EMITTED_UNARY_IDS: Tuple[int, ...] = (
    Func.SIN,
    Func.COS,
    Func.TAN,
    Func.SINH,
    Func.COSH,
    Func.TANH,
    Func.LOG,
    Func.LOOSE_LOG,
    Func.EXP,
    Func.INV,
    Func.LOOSE_INV,
    Func.NEG,
    Func.ABS,
    Func.SQRT,
    Func.LOOSE_SQRT,
)
#: ``utils.DELTA`` -- single-sourced, but installed as a ``tl.constexpr`` (see
#: :func:`_install_globals`) because the kernel bodies may only read constexpr
#: globals.  Before the first :func:`kernels` call this is the plain float.
_DELTA = float(DELTA)


def emitted_function_ids() -> Dict[str, Tuple[int, ...]]:
    """The function ids the kernels implement, split by arity (``binary``/``unary``).

    ``if`` (id 0) needs no branch: every node type that is not VAR/CONST/UFUNC/
    BFUNC walks the ternary path and ignores its function id, which is what the
    oracle's ``_kind_from_type`` fall-through does too.
    """
    return {"binary": _EMITTED_BINARY_IDS, "unary": _EMITTED_UNARY_IDS}


def _comparison_ids(body) -> Tuple[int, ...]:
    """The literal ids an un-jitted kernel helper compares ``func_id`` against.

    Reads the helper's own source, which is only available from a source checkout:
    that is why this is the SECOND half of the check and the data above is the
    first (see :func:`_verify_operator_table`).
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(body)))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "func_id"
        ):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(
                    comparator.value, int
                ):
                    found.append(int(comparator.value))
    return tuple(sorted(found))


def _expected_ids() -> Dict[str, Tuple[int, ...]]:
    """``utils.FUNCS`` split by the arity ranges of ``utils.Func``."""
    binary = tuple(f for f in FUNCS if Func.BF_START <= f < Func.UF_START)
    unary = tuple(f for f in FUNCS if Func.UF_START <= f < Func.END)
    return {"binary": binary, "unary": unary}


def _verify_operator_table() -> None:
    """Fail the IMPORT if the kernels' operator table is not exactly ``utils.FUNCS``.

    An uncovered function id silently evaluates to ``0.0`` in both kernels, so a new
    operator must never be able to slip in unreported.  Two independent comparisons:

    1. the declared id sets vs ``utils.FUNCS`` (no source needed), which catches an
       operator ADDED to or REMOVED from the vocabulary;
    2. the declared id sets vs the literals the ``if/elif`` chains actually compare
       (needs the source file, i.e. a checkout), which catches a branch edited
       without the declaration.
    """
    declared = emitted_function_ids()
    expected = _expected_ids()
    for arity in ("binary", "unary"):
        if tuple(sorted(declared[arity])) != tuple(sorted(expected[arity])):
            missing = sorted(set(expected[arity]) - set(declared[arity]))
            extra = sorted(set(declared[arity]) - set(expected[arity]))
            raise RuntimeError(
                f"fused_kernels: the {arity} operator table is out of step with "
                f"utils.FUNCS (missing ids {missing}, unknown ids {extra}); a "
                "function id with no branch evaluates to 0.0 in both kernels"
            )
    try:
        wired = {
            "binary": _comparison_ids(_binary),
            "unary": _comparison_ids(_unary),
        }
    except (OSError, TypeError, SyntaxError):
        # No source (a byte-code-only install): the data check above still ran.
        return
    for arity in ("binary", "unary"):
        if tuple(sorted(wired[arity])) != tuple(sorted(expected[arity])):
            missing = sorted(set(expected[arity]) - set(wired[arity]))
            extra = sorted(set(wired[arity]) - set(expected[arity]))
            raise RuntimeError(
                f"fused_kernels: the {arity} kernel body and its declared operator "
                f"table disagree (branches missing ids {missing}, branches for "
                f"unknown ids {extra})"
            )


# --------------------------------------------------------------------------- #
# numeric helpers (float32 node-value semantics; see the module docstring)
# --------------------------------------------------------------------------- #
# The names below are looked up as GLOBALS by the compiled kernels, so
# :func:`kernels` rebinds `_unary`, `_binary`, `_copysign_delta` and
# `_fused_kernel` to their ``triton.jit`` wrapped versions after the first
# import of triton.  Until then they are plain Python and must not be CALLED
# (their bodies reference the not-yet-installed `tl` global).


def _round32(value):
    """Round a float64 intermediate back to the float32 node-value storage."""
    return value.to(tl.float32)


def _wide(value):
    """Widen a float32 operand to the float64 intermediate of a CR op."""
    return value.to(tl.float64)


def _copysign_delta(y):
    """``copysign(1e-9, y)`` -- the DELTA clamp of the loose ops.

    The sign comes from the SIGN BIT (``y`` bitcast to int32), not from ``y >= 0``:
    ``-0.0`` is a "negative" clamp (``loose_inv(-0.0) = -1e9`` is pinned by
    ``ENGINE_SPEC.md`` §3 and by the ``03_edge_nan`` golden), and every
    arithmetic sign test answers "+" for it.
    """
    return tl.where(y.to(tl.int32, bitcast=True) < 0, -1e-9, 1e-9)


def _unary(func_id, a):
    """One unary node (ids 14..28); any other id yields ``0.0``.

    Mirrors ``reference_eval._apply``'s UFUNC branch one-to-one, including the
    ``round32`` of every transcendental and the two NaN/``-1e9`` placements.
    """
    out = tl.zeros_like(a)
    if func_id == 14:  # SIN
        out = _round32(libdevice.sin(_wide(a)))
    elif func_id == 15:  # COS
        out = _round32(libdevice.cos(_wide(a)))
    elif func_id == 16:  # TAN
        out = _round32(libdevice.tan(_wide(a)))
    elif func_id == 17:  # SINH -- composed, linear below 1e-4 (sign of zero)
        w = _wide(a)
        composed = _round32((libdevice.exp(w) - libdevice.exp(-w)) * 0.5)
        out = tl.where(tl.abs(a) < 1e-4, a, composed)
    elif func_id == 18:  # COSH -- composed
        w = _wide(a)
        out = _round32((libdevice.exp(w) + libdevice.exp(-w)) * 0.5)
    elif func_id == 19:  # TANH
        out = _round32(libdevice.tanh(_wide(a)))
    elif func_id == 20:  # LOG -- log(0) = -inf, log(neg) = NaN
        out = _round32(libdevice.log(_wide(a)))
    elif func_id == 21:  # LOOSE_LOG -- 0 -> -1e9, else log(|a|)
        out = tl.where(a == 0.0, -1e9, _round32(libdevice.log(tl.abs(_wide(a)))))
    elif func_id == 22:  # EXP -- overflow -> inf
        out = _round32(libdevice.exp(_wide(a)))
    elif func_id == 23:  # INV -- 0 -> NaN
        out = tl.where(a == 0.0, float("nan"), 1.0 / a)
    elif func_id == 24:  # LOOSE_INV -- |a| <= 1e-9 -> copysign(1e-9, a)
        safe = tl.where(tl.abs(a) <= 1e-9, _copysign_delta(a), a)
        out = 1.0 / safe
    elif func_id == 25:  # NEG
        out = -a
    elif func_id == 26:  # ABS
        out = tl.abs(a)
    elif func_id == 27:  # SQRT -- negative -> NaN; correctly rounded via f64
        out = _round32(libdevice.sqrt(_wide(a)))
    elif func_id == 28:  # LOOSE_SQRT -- sqrt(|a|)
        out = _round32(libdevice.sqrt(tl.abs(_wide(a))))
    return out


def _binary(func_id, a, b):
    """One binary node (ids 1..13); any other id yields ``0.0``.

    Mirrors ``reference_eval._apply``'s BFUNC branch one-to-one: the ``DIV``
    zero test answers NaN (``+0.0`` and ``-0.0`` alike), the loose ops clamp
    ``|b| <= 1e-9`` through :func:`_copysign_delta`, ``POW`` goes through float64
    (``pow(0,0)=1``, ``pow(-2,3)=-8``, ``pow(-2,0.5)=NaN``) and
    ``LOOSE_POW(0,0) = 0`` before the ``pow(|a|, b)`` form, ``MAX``/``MIN`` are
    the NaN-propagating selects and the four comparisons return bit-exact ``+-1``.
    """
    out = tl.zeros_like(a)
    if func_id == 1:  # ADD
        out = a + b
    elif func_id == 2:  # SUB
        out = a - b
    elif func_id == 3:  # MUL
        out = a * b
    elif func_id == 4:  # DIV
        out = tl.where(b == 0.0, float("nan"), a / b)
    elif func_id == 5:  # LOOSE_DIV
        safe = tl.where(tl.abs(b) <= 1e-9, _copysign_delta(b), b)
        out = a / safe
    elif func_id == 6:  # POW
        out = _round32(libdevice.pow(_wide(a), _wide(b)))
    elif func_id == 7:  # LOOSE_POW
        composed = _round32(libdevice.pow(tl.abs(_wide(a)), _wide(b)))
        out = tl.where((a == 0.0) & (b == 0.0), 0.0, composed)
    elif func_id == 8:  # MAX -- max(v, NaN) is NaN
        out = tl.where(a >= b, a, b)
    elif func_id == 9:  # MIN -- min(v, NaN) is NaN
        out = tl.where(a <= b, a, b)
    elif func_id == 10:  # LT
        out = tl.where(a < b, 1.0, -1.0)
    elif func_id == 11:  # GT
        out = tl.where(a > b, 1.0, -1.0)
    elif func_id == 12:  # LE
        out = tl.where(a <= b, 1.0, -1.0)
    elif func_id == 13:  # GE
        out = tl.where(a >= b, 1.0, -1.0)
    return out


# --------------------------------------------------------------------------- #
# the kernel
# --------------------------------------------------------------------------- #
def _fused_kernel(
    node_value_ptr,
    node_type_ptr,
    subtree_size_ptr,
    x_ptr,
    out_ptr,
    labels_ptr,
    partial_ptr,
    scratch_ptr,
    input_len,
    tree_len,
    rows,
    stack_cap,
    BLOCK_R: "tl.constexpr",
    PER_ROW: "tl.constexpr",
    REDUCE: "tl.constexpr",
    USE_MSE: "tl.constexpr",
):
    """One (tree, row-block) program: reverse-prefix walk + store/reduce.

    ``x_ptr`` is the caller's ``(pop, input_len)`` block when ``PER_ROW`` (tree
    ``t`` reads its own row, one data point) and the PRETRANSPOSED
    ``(input_len, rows)`` block otherwise (a VAR read is then
    ``x_ptr + vi * rows + r``, one coalesced vector per VAR node).
    ``REDUCE`` switches the epilogue: the value write (``out``, ``(pop, rows, 1)``)
    or the SR-loss partial (``labels``, ``(rows, 1)`` flattened; ``partial``,
    ``(pop, n_rb)``).
    """
    t = tl.program_id(0)
    rb = tl.program_id(1)
    n_rb = tl.num_programs(1)
    pid = t * n_rb + rb
    off = tl.arange(0, BLOCK_R)
    r = rb * BLOCK_R + off
    rmask = r < rows

    row_base = t * tree_len
    n = tl.load(subtree_size_ptr + row_base).to(tl.int32)
    # One scratch region per program; only slots < the walk's stack depth are
    # ever touched, so the allocation is sparse in practice (see scratch_stats).
    stack = scratch_ptr + pid.to(tl.int64) * (stack_cap * BLOCK_R)

    sp = 0
    prev = tl.zeros([BLOCK_R], tl.float32)
    val = tl.zeros([BLOCK_R], tl.float32)
    j = n - 1
    tt = tl.load(node_type_ptr + row_base + tl.maximum(j, 0))
    for _k in range(0, n):
        if tt == 0:  # VAR
            vi = tl.load(node_value_ptr + row_base + j).to(tl.int32)
            inb = (vi >= 0) & (vi < input_len)
            if PER_ROW:
                scalar = tl.load(x_ptr + t * input_len + vi, mask=inb, other=0.0)
                val = tl.broadcast_to(scalar, [BLOCK_R])
            else:
                val = tl.load(x_ptr + vi * rows + r, mask=inb & rmask, other=0.0)
        elif tt == 1:  # CONST
            val = tl.broadcast_to(tl.load(node_value_ptr + row_base + j), [BLOCK_R])
        elif tt == 2:  # UFUNC
            func_id = tl.load(node_value_ptr + row_base + j).to(tl.int32)
            val = _unary(func_id, prev)
        elif tt == 3:  # BFUNC
            func_id = tl.load(node_value_ptr + row_base + j).to(tl.int32)
            slot = tl.minimum(tl.maximum(sp - 1, 0), stack_cap - 1)
            second = tl.load(stack + slot * BLOCK_R + off, mask=rmask, other=0.0)
            sp = sp - 1
            val = _binary(func_id, prev, second)
        else:  # TFUNC -- every other type, and the function id is ignored
            slot_b = tl.minimum(tl.maximum(sp - 1, 0), stack_cap - 1)
            slot_c = tl.minimum(tl.maximum(sp - 2, 0), stack_cap - 1)
            then_value = tl.load(stack + slot_b * BLOCK_R + off, mask=rmask, other=0.0)
            else_value = tl.load(stack + slot_c * BLOCK_R + off, mask=rmask, other=0.0)
            sp = sp - 2
            val = tl.where(prev > 0.0, then_value, else_value)
        j = j - 1
        # Lookahead: this is also the next iteration's `tt`, so the walk costs
        # one node_type load per node.  Clamped at the root (j == -1) -- the last
        # iteration's answer is unused, it can only push a value nobody pops.
        tt = tl.load(node_type_ptr + row_base + tl.maximum(j, 0))
        if (tt == 0) | (tt == 1):
            slot_next = tl.minimum(tl.maximum(sp, 0), stack_cap - 1)
            tl.store(stack + slot_next * BLOCK_R + off, val, mask=rmask)
            sp = sp + 1
        else:
            prev = val

    if REDUCE:
        label = tl.load(labels_ptr + r, mask=rmask, other=0.0)
        diff = label - val
        if USE_MSE:
            err = diff * diff
        else:
            err = tl.abs(diff)
        tl.store(partial_ptr + t * n_rb + rb, tl.sum(tl.where(rmask, err, 0.0)))
    else:
        # `rows == 1` on the PER_ROW path, so this is one scalar store per tree.
        tl.store(out_ptr + t * rows + r, val, mask=rmask)


#: ``NType.OUT_NODE`` -- the multi-output tag OR-ed into ``node_type`` (the kernel
#: body spells both masks as literals: a Triton kernel may only read ``tl.constexpr``
#: globals, and ``tl`` does not exist at this module's import time).
OUT_NODE_BIT = 1 << 7
#: ``NType.TYPE_MASK`` -- the base kind under that tag.
TYPE_MASK_BITS = 0x7F


def _fused_multi_kernel(
    node_value_ptr,
    node_type_ptr,
    subtree_size_ptr,
    x_ptr,
    out_ptr,
    labels_ptr,
    partial_ptr,
    scratch_ptr,
    input_len,
    tree_len,
    rows,
    stack_cap,
    BLOCK_R: "tl.constexpr",
    PER_ROW: "tl.constexpr",
    REDUCE: "tl.constexpr",
    USE_MSE: "tl.constexpr",
    OUT_LEN: "tl.constexpr",
):
    """One (tree, row-block) program of the MULTI-OUTPUT protocol (``OUT_LEN >= 2``).

    Same reverse-prefix walk as :func:`_fused_kernel`, plus the two rules the
    multi-output protocol adds (the legacy ``treeGPEvalKernel<true>``, mirrored by
    ``reference_eval.forward_tree(multi_output=True)``):

    * the base kind is ``node_type & 0x7F`` and ``node_type & 0x80`` is the
      ``OUT_NODE`` tag; a TAGGED FUNCTION NODE resolves its operator from the packed
      value bits (``function = bits & 0xFFFF``, ``outIndex = (bits >> 16) & 0xFFFF``)
      and adds its OWN value into ``out[outIndex]`` (an index ``>= OUT_LEN`` is
      dropped, exactly like the oracle's unsigned compare);
    * every node passes its RIGHT-MOST operand up to its parent instead of its own
      value, so ``prev``/the stack carry ``pass_up``: the second operand of a binary
      node, the third of a ternary one, the operand of a unary one and its own value
      for a leaf.

    The accumulator is the program's own slice of ``out_ptr`` (``(pop, rows,
    OUT_LEN)``, zeroed here and then read-modify-written at the tagged nodes), not a
    register tile: shifting a ``[OUT_LEN, BLOCK_R]`` tile across the loop phis makes
    triton unify two layouts for the value registers and the LLVM translation then
    fails on an unrealized conversion cast.  One zero-store pass plus one load/store
    per tagged node is ~nothing next to the walk, and only 1-D tiles remain.

    Tagged LEAVES accumulate nothing (the oracle returns before it ever reads an out
    index on a VAR/CONST node) and untagged outputs stay ``0.0`` (the zeroing pass).
    The root's own value is ignored, exactly like the kernel.
    """
    t = tl.program_id(0)
    rb = tl.program_id(1)
    n_rb = tl.num_programs(1)
    pid = t * n_rb + rb
    off = tl.arange(0, BLOCK_R)
    r = rb * BLOCK_R + off
    rmask = r < rows

    row_base = t * tree_len
    n = tl.load(subtree_size_ptr + row_base).to(tl.int32)
    stack = scratch_ptr + pid.to(tl.int64) * (stack_cap * BLOCK_R)
    # Address of output 0 of every row of this program: output k is ``base + k``.
    base = (t.to(tl.int64) * rows + r.to(tl.int64)) * OUT_LEN
    # Clear this program's slice: an output no node ever tags must read 0.0.
    for _k in tl.static_range(0, OUT_LEN):
        tl.store(out_ptr + base + _k, tl.zeros([BLOCK_R], tl.float32), mask=rmask)

    sp = 0
    prev = tl.zeros([BLOCK_R], tl.float32)
    val = tl.zeros([BLOCK_R], tl.float32)
    # The value handed to the parent: a fresh tile, never an alias of ``val`` (an
    # alias makes the loop phi carry two layouts for one register).
    pass_up = tl.zeros([BLOCK_R], tl.float32)
    j = n - 1
    raw = tl.load(node_type_ptr + row_base + tl.maximum(j, 0))
    for _i in range(0, n):
        tt = raw & 127
        tagged = (raw & 128) != 0
        if tt == 0:  # VAR
            vi = tl.load(node_value_ptr + row_base + j).to(tl.int32)
            inb = (vi >= 0) & (vi < input_len)
            if PER_ROW:
                scalar = tl.load(x_ptr + t * input_len + vi, mask=inb, other=0.0)
                val = tl.broadcast_to(scalar, [BLOCK_R])
            else:
                val = tl.load(x_ptr + vi * rows + r, mask=inb & rmask, other=0.0)
            pass_up = val
        elif tt == 1:  # CONST
            val = tl.broadcast_to(tl.load(node_value_ptr + row_base + j), [BLOCK_R])
            pass_up = val
        elif tt == 2:  # UFUNC
            value = tl.load(node_value_ptr + row_base + j)
            bits = value.to(tl.int32, bitcast=True)
            func_id = tl.where(tagged, bits & 0xFFFF, value.to(tl.int32))
            val = _unary(func_id, prev)
            if tagged:
                oi = (bits >> 16) & 0xFFFF
                keep = rmask & (oi < OUT_LEN)
                addr = out_ptr + base + oi
                tl.store(addr, tl.load(addr, mask=keep, other=0.0) + val, mask=keep)
            pass_up = prev
        elif tt == 3:  # BFUNC
            value = tl.load(node_value_ptr + row_base + j)
            bits = value.to(tl.int32, bitcast=True)
            func_id = tl.where(tagged, bits & 0xFFFF, value.to(tl.int32))
            slot = tl.minimum(tl.maximum(sp - 1, 0), stack_cap - 1)
            second = tl.load(stack + slot * BLOCK_R + off, mask=rmask, other=0.0)
            sp = sp - 1
            val = _binary(func_id, prev, second)
            if tagged:
                oi = (bits >> 16) & 0xFFFF
                keep = rmask & (oi < OUT_LEN)
                addr = out_ptr + base + oi
                tl.store(addr, tl.load(addr, mask=keep, other=0.0) + val, mask=keep)
            pass_up = second
        else:  # TFUNC -- every other base kind, and the function id is ignored
            bits = tl.load(node_value_ptr + row_base + j).to(tl.int32, bitcast=True)
            slot_b = tl.minimum(tl.maximum(sp - 1, 0), stack_cap - 1)
            slot_c = tl.minimum(tl.maximum(sp - 2, 0), stack_cap - 1)
            then_value = tl.load(stack + slot_b * BLOCK_R + off, mask=rmask, other=0.0)
            else_value = tl.load(stack + slot_c * BLOCK_R + off, mask=rmask, other=0.0)
            sp = sp - 2
            val = tl.where(prev > 0.0, then_value, else_value)
            if tagged:
                oi = (bits >> 16) & 0xFFFF
                keep = rmask & (oi < OUT_LEN)
                addr = out_ptr + base + oi
                tl.store(addr, tl.load(addr, mask=keep, other=0.0) + val, mask=keep)
            pass_up = else_value
        j = j - 1
        # Lookahead: this is also the next iteration's ``raw``, so the walk costs one
        # node_type load per node.  Clamped at the root (j == -1).
        raw = tl.load(node_type_ptr + row_base + tl.maximum(j, 0))
        if ((raw & 127) == 0) | ((raw & 127) == 1):
            slot_next = tl.minimum(tl.maximum(sp, 0), stack_cap - 1)
            tl.store(stack + slot_next * BLOCK_R + off, pass_up, mask=rmask)
            sp = sp + 1
        else:
            prev = pass_up

    if REDUCE:
        # Sum over outputs first, then over the block's rows: the oracle's order.
        err_all = tl.zeros([BLOCK_R], tl.float32)
        for _k in tl.static_range(0, OUT_LEN):
            lab = tl.load(labels_ptr + r * OUT_LEN + _k, mask=rmask, other=0.0)
            diff = lab - tl.load(out_ptr + base + _k, mask=rmask, other=0.0)
            if USE_MSE:
                err_all = err_all + diff * diff
            else:
                err_all = err_all + tl.abs(diff)
        tl.store(partial_ptr + t * n_rb + rb, tl.sum(tl.where(rmask, err_all, 0.0)))


# --------------------------------------------------------------------------- #
# lazy triton installation
# --------------------------------------------------------------------------- #
_LOCK = threading.RLock()
_KERNELS: Optional[Dict[str, Any]] = None
_TRITON_IMPORT_ERROR: Optional[BaseException] = None


def triton_available() -> bool:
    """Whether ``triton`` can be imported here, WITHOUT importing it.

    Uses :func:`importlib.util.find_spec`, so asking the question costs no
    triton import (which is why the route policy may ask it per call).
    """
    import importlib.util

    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:  # pragma: no cover - a broken install path only
        return False


def _install_globals(triton_module: Any, language: Any, libd: Any) -> None:
    """Publish ``triton`` / ``tl`` / ``libdevice`` into THIS module's globals.

    Separate from :func:`kernels` on purpose: the names must not become locals of
    the builder, or the kernel bodies would reference them as closure cells while
    triton's code generator resolves free names through ``fn.__globals__``.
    """
    g = globals()
    g["triton"] = triton_module
    g["tl"] = language
    g["libdevice"] = libd


def kernels() -> Dict[str, Any]:
    """Import triton (once) and return the compiled kernel set.

    Returns a dict with ``eval`` (the JIT kernel), ``blocked`` (whether the
    triton import worked) and the module handles.  Raises
    :class:`FusedUnavailable` when triton cannot be imported at all; a GPU-less
    box is NOT an error here (the caller checks ``torch.cuda.is_available``).
    """
    global _KERNELS, _TRITON_IMPORT_ERROR
    with _LOCK:
        if _KERNELS is not None:
            return _KERNELS
        try:
            import triton as triton_module
            import triton.language as language
        except Exception as exc:  # pragma: no cover - broken install only
            _TRITON_IMPORT_ERROR = exc
            raise FusedUnavailable(f"triton is not importable: {exc}") from exc
        try:
            from triton.language.extra import libdevice as libd
        except Exception as exc:  # pragma: no cover - unusual triton layout
            raise FusedUnavailable(
                f"triton.language.extra.libdevice is not importable: {exc}"
            ) from exc
        for needed in ("sin", "cos", "tan", "exp", "log", "pow", "sqrt", "tanh"):
            if not hasattr(libd, needed):
                raise FusedUnavailable(
                    f"this triton build's libdevice has no {needed!r}; the fused "
                    "route needs the float64 transcendental ladder"
                )
        _install_globals(triton_module, language, libd)
        jit = triton_module.jit
        g = globals()
        # The compiled kernel resolves its callees through the module globals, so
        # the helper names must BE the jitted objects (see the module docstring).
        g["_round32"] = jit(_round32)
        g["_wide"] = jit(_wide)
        g["_copysign_delta"] = jit(_copysign_delta)
        g["_unary"] = jit(_unary)
        g["_binary"] = jit(_binary)
        g["_fused_kernel"] = jit(_fused_kernel)
        g["_fused_multi_kernel"] = jit(_fused_multi_kernel)
        _KERNELS = {
            "triton": triton_module,
            "tl": language,
            "libdevice": libd,
            "eval": g["_fused_kernel"],
            "eval_multi": g["_fused_multi_kernel"],
            "specializations": set(),
        }
        return _KERNELS


# --------------------------------------------------------------------------- #
# launch geometry, scratch cache and counters
# --------------------------------------------------------------------------- #
#: ``(device_index, n_prog, stack_cap, block_rows)`` -> the scratch tensor.  A
#: training loop keeps the population shape for a whole run, so this is a single
#: entry in practice; the bound only protects a caller that sweeps shapes.
_SCRATCH: "OrderedDict[Tuple[int, int, int, int], torch.Tensor]" = OrderedDict()
_SCRATCH_MAX_ENTRIES = 8
_STATS: Dict[str, int] = {
    "kernels": 0,
    "specializations": 0,
    "scratch_allocs": 0,
    "scratch_reuses": 0,
}


def plan(rows: int, *, per_row: bool, block_rows: int, num_warps: int) -> Dict[str, int]:
    """Resolve the launch geometry of one call.

    ``rows`` is the DATA-row count of the call (``1`` for ``per_row``: tree ``t``
    sees its own row).  ``block_rows`` is clamped to
    ``min(next_pow2(rows), block_rows)`` so a small dataset does not launch a
    256-lane block over 8 live lanes, and ``num_warps`` is capped so the block
    holds at least one element per warp lane (a wider block than the row count
    wastes lanes; a block narrower than 64*warps is legal but pointless).
    """
    rows = int(rows)
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    block = 1
    while block < rows and block < int(block_rows):
        block *= 2
    block = max(1, min(block, int(block_rows)))
    warps = max(1, int(num_warps))
    while warps > 1 and warps * 32 > block:
        warps //= 2
    return {
        "block_rows": block,
        "num_warps": warps,
        "n_rb": (rows + block - 1) // block,
        "rows": rows,
    }


def _scratch_for(
    device_index: int, n_prog: int, stack_cap: int, block_rows: int
) -> torch.Tensor:
    """The scratch block of one shape class, allocated once and reused."""
    key = (int(device_index), int(n_prog), int(stack_cap), int(block_rows))
    with _LOCK:
        cached = _SCRATCH.get(key)
        if cached is not None:
            _SCRATCH.move_to_end(key)
            _STATS["scratch_reuses"] += 1
            return cached
        scratch = torch.empty(
            (int(n_prog) * int(stack_cap) * int(block_rows),),
            dtype=torch.float32,
            device=torch.device("cuda", int(device_index)),
        )
        _SCRATCH[key] = scratch
        _STATS["scratch_allocs"] += 1
        while len(_SCRATCH) > _SCRATCH_MAX_ENTRIES:
            _SCRATCH.popitem(last=False)
        return scratch


def clear_caches() -> None:
    """Drop the scratch blocks and reset the launch counters (memory only)."""
    with _LOCK:
        _SCRATCH.clear()
        for name in _STATS:
            _STATS[name] = 0


def scratch_stats() -> Dict[str, int]:
    """Launch-side counters, already prefixed ``fused_`` for the facade."""
    with _LOCK:
        return {
            "fused_kernel_launches": _STATS["kernels"],
            "fused_kernel_compiles": _STATS["specializations"],
            "fused_scratch_allocs": _STATS["scratch_allocs"],
            "fused_scratch_reuses": _STATS["scratch_reuses"],
            "fused_scratch_entries": len(_SCRATCH),
        }


def _device_index(device: Any) -> int:
    """The CUDA index of a torch device (``cuda`` alone means the current one)."""
    index = getattr(device, "index", None)
    if index is None:
        return int(torch.cuda.current_device())
    return int(index)


def _launch(
    *,
    node_value: torch.Tensor,
    node_type: torch.Tensor,
    subtree_size: torch.Tensor,
    x: torch.Tensor,
    labels: Optional[torch.Tensor],
    rows: int,
    input_len: int,
    tree_len: int,
    per_row: bool,
    reduce: bool,
    use_mse: bool,
    block_rows: int,
    num_warps: int,
    out_len: int = 1,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Shared launch: geometry, scratch, dispatch on the (device, flags) key.

    Returns ``(out, geometry)`` where ``out`` is the value buffer
    (``PER_ROW``/``REDUCE`` decide what it means; the caller owns its meaning).
    The whole call is dispatched on the tensors'
    device index (a Triton launch compiles/streams against the CURRENT device, so
    the launch is wrapped in ``torch.cuda.device(...)``).

    ``out_len == 1`` runs :func:`_fused_kernel` (the single-output protocol, its
    original code path, unchanged); ``out_len >= 2`` runs
    :func:`_fused_multi_kernel`, whose accumulator is the ``(pop, rows, out_len)``
    prediction buffer.
    """
    pop = int(node_value.shape[0])
    out_len = int(out_len)
    if out_len < 1:
        raise ValueError(f"out_len must be >= 1, got {out_len}")
    multi = out_len > 1
    geom = plan(rows, per_row=per_row, block_rows=block_rows, num_warps=num_warps)
    device_index = _device_index(node_value.device)
    n_prog = pop * geom["n_rb"]
    stack_cap = max(1, int(tree_len))
    scratch = _scratch_for(device_index, n_prog, stack_cap, geom["block_rows"])

    if multi:
        # The multi-output kernel accumulates INTO the prediction buffer, so the
        # loss path needs it materialized even though only the chart is returned.
        values = torch.empty(
            (pop, rows, out_len), dtype=torch.float32, device=node_value.device
        )
        if reduce:
            out = torch.empty(
                (pop, geom["n_rb"]), dtype=torch.float32, device=node_value.device
            )
            label_ptr, partial_ptr = labels, out
        else:
            out = values
            label_ptr = partial_ptr = values
        value_ptr = values
    elif reduce:
        # One partial per (tree, row-block); the facade sums the chart.
        out = torch.empty((pop, geom["n_rb"]), dtype=torch.float32, device=node_value.device)
        label_ptr = labels
        value_ptr = partial_ptr = out
    else:
        out = torch.empty(
            (pop, rows, out_len), dtype=torch.float32, device=node_value.device
        )
        # ``labels`` and the partial chart are behind the REDUCE constexpr, i.e.
        # never traced on this path; the value buffer stands in for both so the
        # signature stays single.
        label_ptr = out
        value_ptr = partial_ptr = out
    kernel = kernels()["eval_multi" if multi else "eval"]
    key = (
        "multi" if multi else "single",
        int(out_len),
        bool(per_row),
        bool(reduce),
        bool(use_mse),
        int(geom["block_rows"]),
        int(geom["num_warps"]),
    )
    specializations = kernels()["specializations"]
    with _LOCK:
        _STATS["kernels"] += 1
        if key not in specializations:
            specializations.add(key)
            _STATS["specializations"] += 1
    extra = {"OUT_LEN": out_len} if multi else {}
    with torch.cuda.device(device_index):
        kernel[(pop, geom["n_rb"])](
            node_value,
            node_type,
            subtree_size,
            x,
            value_ptr,
            label_ptr,
            partial_ptr,
            scratch,
            int(input_len),
            int(tree_len),
            int(rows),
            int(stack_cap),
            BLOCK_R=int(geom["block_rows"]),
            PER_ROW=bool(per_row),
            REDUCE=bool(reduce),
            USE_MSE=bool(use_mse),
            # SEE THE MODULE DOCSTRING: without this, LLVM fuses `a*b` into the
            # parent `+`/`-` and the arithmetic path stops being bitwise.
            enable_fp_fusion=False,
            num_warps=int(geom["num_warps"]),
            **extra,
        )
    return out, geom


def launch_value(
    node_value: torch.Tensor,
    node_type: torch.Tensor,
    subtree_size: torch.Tensor,
    x: torch.Tensor,
    *,
    input_len: int,
    tree_len: int,
    rows: int,
    per_row: bool,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    num_warps: int = DEFAULT_NUM_WARPS,
    out_len: int = 1,
) -> torch.Tensor:
    """Evaluate the population's value: ``(pop, rows, out_len)`` fp32 (fresh).

    ``x`` is the ``(pop, input_len)`` per-tree block when ``per_row`` and the
    PRETRANSPOSED ``(input_len, rows)`` block otherwise; the facade owns that
    conversion (it is the only host-visible prep in the call path).  ``out_len >= 2``
    selects the multi-output protocol (``OUT_NODE`` tags, the accumulated chart),
    ``out_len == 1`` the original single-output walk.
    """
    out, _geom = _launch(
        node_value=node_value,
        node_type=node_type,
        subtree_size=subtree_size,
        x=x,
        labels=None,
        rows=int(rows),
        input_len=int(input_len),
        tree_len=int(tree_len),
        per_row=bool(per_row),
        reduce=False,
        use_mse=True,
        block_rows=block_rows,
        num_warps=num_warps,
        out_len=int(out_len),
    )
    return out


def launch_loss(
    node_value: torch.Tensor,
    node_type: torch.Tensor,
    subtree_size: torch.Tensor,
    xt: torch.Tensor,
    labels: torch.Tensor,
    *,
    input_len: int,
    tree_len: int,
    rows: int,
    use_mse: bool,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    num_warps: int = DEFAULT_NUM_WARPS,
    out_len: int = 1,
) -> torch.Tensor:
    """SR-loss partial chart: ``(pop, n_rb)`` fp32 (fresh).

    ``xt`` is the PRETRANSPOSED ``(input_len, rows)`` block and ``labels`` the
    label block: a flattened ``(rows,)`` view for ``out_len == 1`` and the
    row-major ``(rows, out_len)`` column block otherwise.  Dividing by ``rows``
    and summing the chart is the facade's one-line epilogue; ``out_len >= 2``
    sums the per-output errors in-kernel and still divides by ``rows`` only.
    """
    partial, _geom = _launch(
        node_value=node_value,
        node_type=node_type,
        subtree_size=subtree_size,
        x=xt,
        labels=labels,
        rows=int(rows),
        input_len=int(input_len),
        tree_len=int(tree_len),
        per_row=False,
        reduce=True,
        use_mse=bool(use_mse),
        block_rows=block_rows,
        num_warps=num_warps,
        out_len=int(out_len),
    )
    return partial
