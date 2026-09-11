"""ETL-backed evaluation & fitness engine for ``evogp.tree`` (the "true jit").

This module is the evaluation/fitness half of the replacement for the five
custom CUDA ops of the legacy compiled extension.  It reproduces the semantics
of the old ``tree_evaluate`` / ``tree_SR_fitness`` kernels (``cuda/forward.cu``,
distilled into :mod:`evogp.tree.ENGINE_SPEC`) with a *per-tree just-in-time* ETL
graph:

1. :func:`decode_population` walks the population tensors **host-side** and
   turns every individual into a flat, int-only *static program* (a prefix
   token stream, with constant subtrees already folded).  All validation and
   all constant folding happens here; no GPU work, no compilation.
2. :func:`evaluate` / :func:`sr_fitness` specialize that program into a fresh
   ETL graph through :func:`etl.build_cached` (trace -> lower -> compile ->
   load, memoized in one shared :class:`etl.CompileCache`), then run it.
3. Compiled executables are cached per *(tree structure, input shape)* pair, so
   re-evaluating the same structure (elite trees, repeated rows, the same
   population next generation) is a pure cache hit.

The step function ``_eval_step`` is the ONLY ``@etl.defn`` in this module and
lives at module level so its ``id(fn)`` -- part of the ``build_cached`` key --
is stable for the lifetime of the process.  There is no in-graph recursion:
the trace-time walker is a plain Python reverse-prefix stack walk (the same
operand discipline as ``_treeGPEvalByStack``) and emits ordinary ETL ops.

Public API (frozen; the wiring round in ``tree.py`` / ``forest.py`` depends on
exactly these names) -- see each docstring for the exact contract::

    decode_population(batch_node_value, batch_node_type, batch_subtree_size,
                      input_len, output_len) -> Population
    evaluate(population, x, *, per_row, out_device=None, ...) -> torch.Tensor
    sr_fitness(population, inputs, labels, use_MSE=True, *, out_device=None,
               ...) -> torch.Tensor

Documented additions to the frozen signatures (backwards compatible, all
keyword-only and defaulted): ``backend`` and ``iree_device_index`` on
``evaluate`` / ``sr_fitness``.  They only override the environment defaults.

Backend & device policy
-----------------------
* Default backend: ``"numpy"`` (ETL's reference interpreter, **cpu**).
  Round-2 measurements make this the only sane default: a numpy executable is
  built in ~1-3 ms and runs in ~0.02-0.2 ms per tree, while an IREE executable
  costs ~0.4-0.6 s to compile and ~2.4-3.6 ms per run (see the round-2 report
  for the numbers).  ``"numpy"`` needs no GPU at all.
* Opt-in IREE: ``backend="iree"`` (or ``EVOGP_ETL_BACKEND=iree``) plus a device
  index (``iree_device_index=`` / ``EVOGP_ETL_IREE_DEVICE``, default 0).  IREE
  runs on the CUDA HAL only: ``etl.core.Device("cuda", index)`` (ETL maps the
  index to IREE's 1-based ``device_id``, i.e. it is the *visible* CUDA device),
  ``target_backends=["cuda"]`` and ``opt_level`` ``"O0"``/``"O1"``
  (``EVOGP_ETL_IREE_OPT_LEVEL``, default ``"O1"``).  Device tensors are moved
  host->device explicitly (``Tensor.to``) and results back with ``.to(cpu)``,
  exactly as ETL's device contract requires; the host->device copy of ``x``
  happens ONCE per :func:`evaluate` call and is reused for every tree.
* Backend, device and options are always passed to ``build_cached``
  explicitly, so ambient ``ETL_BACKEND`` / ``ETL_DEVICE`` /
  ``ETL_TARGET_BACKENDS`` cannot change what this module does; they do
  participate in the cache key, together with the input shape, so numpy and
  IREE executables coexist in the cache.
* The population/input tensors may live on any torch device (they are read
  back with ``.cpu()``) and the result is returned on the input's device unless
  ``out_device`` says otherwise.

Cache sizing
------------
One module-level :data:`CACHE` (``etl.CompileCache``) is shared by every call.
Every distinct *(tree structure, input shape, backend, device, options)* pair
is one LRU entry, so a full-population fitness pass needs up to ``pop_size``
entries **per input shape**: size the cache at ``>= pop_size + headroom``
(default :data:`DEFAULT_CACHE_SIZE` = 4096, env ``EVOGP_ETL_CACHE_SIZE``).
Each entry pins a fully loaded ``Executable`` (graph + lowered program +
signature + static program) -- measured (tracemalloc, released by
  ``clear_cache``) ~270 KiB for a 30-node tree and ~478 KiB for a 60-node
  tree on the numpy backend -- so a larger cache
only costs memory, never correctness.  Eviction is plain LRU: one-off probes
with unusual shapes (e.g. every ``N`` in ``per_row=False``) can push out hot
entries.  :func:`cache_stats` / :func:`clear_cache` expose and reset the table.
The cache is **in-memory only** -- no artifact is written to disk and nothing
is shared across processes (documented ETL limitation).

Speed knob: ``ETL_DISABLE_LOCATIONS=1``
--------------------------------------
Tracing attaches a source ``ir.Location`` to every op unless the environment
variable ``ETL_DISABLE_LOCATIONS=1`` is set.  Setting it speeds tracing up by
roughly 4x on this workload (54 trees, mean 18 nodes: 1384 ms -> 346 ms; ETL's
own micro-benchmarks measure 13-30x) at the cost of losing per-op locations in
tracebacks; it is *not* part of the cache key, so changing it mid-process
affects only later compilations.  Recommended for production runs.  This
module does not set it (ETL env knobs are the caller's / operator's choice);
export it *before* the first decode/evaluate for the full win.  The round-2
measurements were taken with it set.

Multi-output status
-------------------
Fully implemented per ENGINE_SPEC section 4.3 and validated against the old
kernel: for ``output_len > 1`` an out-node's packed bits are decoded
(``function = bits & 0xFFFF``, ``outIndex = bits >> 16`` with C arithmetic
shift semantics, so a negative high half stays negative), ``outs[outIndex] +=
value`` is accumulated when ``0 <= outIndex < out_len`` and silently dropped
otherwise, EVERY function node then passes its RIGHTMOST child's value up to
its parent (unary -> its child, binary -> 2nd child, ternary -> 3rd child),
tagged leaves behave exactly like untagged leaves, and the output vector is the
accumulated per-output sums (``0.0`` for outputs nothing was tagged with).
``output_len == 1`` is the single-output protocol: the root's value is the
result, OUT_NODE-tagged nodes are rejected at decode time (documented
deviation -- the old kernel had undefined behaviour there, see below).

Documented deviations from the old kernels
------------------------------------------
* **Single-output + OUT_NODE** -> :class:`ValueError` at decode time (the
  kernel fell into the ternary ``if`` branch and corrupted the operand stack).
* **Validation instead of undefined behaviour**: out-of-range / non-finite
  variable indices, non-finite function ids and malformed subtree sizes raise
  :class:`ValueError` naming the row and node; the kernel read out of bounds
  or silently produced garbage.  Note what is *not* validated, because the
  kernel defines it: a function id matching no branch yields ``0.0`` (e.g. the
  reachable ``id == 29`` from an under-normalized roulette cdf), an unknown
  base type is evaluated as the ternary ``if``, and non-integral ids/indices
  are truncated toward zero like the C casts ``(int)`` / ``(unsigned int)``.
* **Non-float32 / non-int16 population tensors** are cast (values to float32,
  types/sizes to int16) instead of being reinterpreted as garbage storage.
* **Source math semantics, not ``-use_fast_math``**: ``pow(0, 0) = 1``,
  ``pow(-2, 3) = -8`` and full-precision ``sin``/``cos`` per ENGINE_SPEC
  section 3.  The reference build used ``-use_fast_math`` and returned NaN for
  both ``pow`` cases and last-bit-different sin/cos (asserted explicitly in
  the round-2 differential tests).
* ``sinh`` / ``cosh`` are composed as ``(exp(x) -+ exp(-x)) * 0.5`` because
  ETL has no ``sinh``/``cosh`` primitive.  Equivalent to the kernel's
  ``sinhf``/``coshf`` up to float32 rounding; for ``|x| < ~1e-4`` the
  subtraction cancels catastrophically (relative error can be large, absolute
  error stays < 1e-7).
* ``copysign`` (``loose_div`` / ``loose_inv``) is composed as
  ``select(1 / x < 0, -DELTA, DELTA)`` because ETL has no ``copysign``; the
  reciprocal sign test is the composition that matches ``copysignf`` for
  ``x == -0.0`` too (the result differs from the kernel only for an *input*
  NaN, which the ``|x| <= DELTA`` guard never sends here).
* ``x`` / ``inputs`` / ``labels`` are cast to float32 **and copied** before
  use, so a float64 caller tensor is rounded (the kernel would have read
  garbage) -- and caller tensors are never mutated.
* Trees with no ``VAR`` node in their emitted program (i.e. fully constant
  trees, including dead ternary branches) are never compiled: the result is
  produced host-side.
* ``sr_fitness`` sums the per-(data point, output) errors with torch's
  reduction order instead of the kernel's block reduction; same formula, float32
  rounding differences of order 1e-6 relative.

Limitations
-----------
* One compiled program per tree *structure*: a fresh random population is a
  cache miss per individual (that is the price of specialization).  Structurally
  identical trees -- and any tree re-evaluated with the same shape -- hit.
* Python-side per-tree work (one ``etl.run`` per tree for the numpy backend)
  dominates; the engine is meant to be correct and re-usable, not yet fast.
* The opt-in IREE backend is wired and runs, but was NOT validated numerically
  for every shape: 30-node/N=1 and 60-node/N=100 trees matched numpy
  only within > 1e-4 (the other measured shapes matched exactly) --
  treat IREE as experimental.  The default numpy backend is
  oracle-verified (see the round-2 report).
* Trees are walked with an explicit stack, so arbitrary depth (up to
  ``MAX_STACK``) is fine, but the static program of a large tree makes a large
  cache key (its content is hashed on every lookup).
* This module only *reads* the population tensors: nothing here writes back,
  and it never imports the legacy compiled extension or touches ``torch.ops``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import etl
from etl.build_cache import CompileCache, build_cached

from .utils import Func, NType

__all__ = [
    "CACHE",
    "DEFAULT_CACHE_SIZE",
    "DecodedTree",
    "Population",
    "cache_stats",
    "clear_cache",
    "decode_population",
    "evaluate",
    "sr_fitness",
]

# ---------------------------------------------------------------------------
# configuration (env read lazily at call time; see the module docstring)
# ---------------------------------------------------------------------------

CACHE_SIZE_ENV = "EVOGP_ETL_CACHE_SIZE"
BACKEND_ENV = "EVOGP_ETL_BACKEND"
IREE_DEVICE_ENV = "EVOGP_ETL_IREE_DEVICE"
IREE_OPT_LEVEL_ENV = "EVOGP_ETL_IREE_OPT_LEVEL"

#: Default ``maxsize`` of :data:`CACHE` (>= pop_size + headroom).
DEFAULT_CACHE_SIZE = 4096

#: Backends this engine is measured/documented for.  ``"numpy"`` is the default
#: (cpu, no GPU needed); ``"iree"`` is the opt-in CUDA-HAL jit backend.
SUPPORTED_BACKENDS = ("numpy", "iree")

_CPU = etl.core.Device("cpu", 0)


def _quiet() -> Any:
    """A fresh ``np.errstate(all="ignore")`` (IEEE inf/nan stays warning-free).

    A new object per use: numpy >= 2 refuses to re-enter the same ``errstate``.
    """
    return np.errstate(all="ignore")
#: ``node_type`` base-type names, for error messages only.
_BASE_NAMES = {
    int(NType.VAR): "VAR",
    int(NType.CONST): "CONST",
    int(NType.UFUNC): "UFUNC",
    int(NType.BFUNC): "BFUNC",
    int(NType.TFUNC): "TFUNC",
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:  # pragma: no cover - operator error
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def _default_cache_size() -> int:
    size = _env_int(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE)
    if size <= 0:
        raise ValueError(f"{CACHE_SIZE_ENV}={size} must be positive")
    return size


#: The one shared compile cache used by every call in this module.
CACHE = CompileCache(maxsize=_default_cache_size())


def cache_stats() -> Dict[str, int]:
    """Snapshot of the shared :data:`CACHE` (``hits``/``misses``/``entries``).

    ``entries`` counts cached executables, ``maxsize`` the LRU limit; the
    counters are reset by :func:`clear_cache`.
    """
    return {
        "hits": CACHE.hits,
        "misses": CACHE.misses,
        "entries": len(CACHE),
        "maxsize": CACHE.maxsize,
    }


def clear_cache() -> None:
    """Drop every cached executable and reset the hit/miss counters."""
    CACHE.clear()


# ---------------------------------------------------------------------------
# static program encoding
# ---------------------------------------------------------------------------
#
# A static program is ``(out_len, tokens)`` where ``tokens`` is a flat tuple of
# int-only tuples in prefix (pre-order) order:
#
#   (K_VAR,   var_index)                -- a variable leaf
#   (K_CONST, float32_bits)             -- a constant leaf (or a folded subtree)
#   (K_*,     function_id, out_index)   -- a function node, ``out_index`` is -1
#                                          when the node is not tagged
#
# Everything is an ``int``: it encodes into the ``build_cached`` key, compares
# by value at the ``etl.run`` boundary (NaN payloads would break ``==``) and is
# immune to float formatting.  ``K_UFUNC/3/4`` mirror ``NType`` so the arity is
# simply ``kind - 1``.

_K_VAR = 0
_K_CONST = 1
_K_UFUNC = 2  # arity 1
_K_BFUNC = 3  # arity 2
_K_TFUNC = 4  # arity 3


def _float_to_bits(value: Union[float, np.floating]) -> int:
    """float32 bit pattern of ``value`` as a (possibly negative) Python int."""
    return int(np.float32(value).view(np.int32))


def _bits_to_float(bits: int) -> np.float32:
    """The float32 whose bit pattern is ``bits``."""
    return np.float32(np.int32(bits).view(np.float32))


def _bits_to_float_arr(bits: Sequence[int]) -> np.ndarray:
    """``(len(bits),)`` float32 array from float32 bit patterns."""
    return np.asarray(bits, dtype=np.int32).view(np.float32)


# ---------------------------------------------------------------------------
# decoded representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedTree:
    """One decoded individual (opaque; fields are diagnostic).

    Attributes:
        program: ``(out_len, tokens)`` static program, or ``None`` when the
            tree is constant (nothing is compiled, ``const_out`` carries the
            answer).
        const_out: ``out_len`` float32 bit patterns of a fully constant tree,
            else ``None``.
        node_count: number of nodes in the prefix encoding (``subtree_size[0]``).
        collapsed_nodes: how many tree nodes the emitted program represents as
            a single folded constant leaf (``node_count - 1`` for a fully
            constant tree, ``0`` for a tree with no foldable subtree).
    """

    program: Optional[Tuple[int, Tuple[Tuple[int, ...], ...]]]
    const_out: Optional[Tuple[int, ...]]
    node_count: int
    collapsed_nodes: int


@dataclass(frozen=True)
class Population:
    """Decoded population handed to :func:`evaluate` / :func:`sr_fitness`."""

    input_len: int
    output_len: int
    pop_size: int
    trees: Tuple[DecodedTree, ...]


def _require_population(population: Any) -> Population:
    if not isinstance(population, Population):
        raise TypeError(
            "expected the Population returned by "
            "evogp.tree.etl_engine.decode_population, got "
            f"{type(population).__name__}"
        )
    return population


# ---------------------------------------------------------------------------
# host-side float32 math (constant folding + host evaluation) -- spec section 3
# ---------------------------------------------------------------------------

_DELTA = np.float32(1e-9)
_NEG_DELTA = np.float32(-1e-9)
_NEG_MAX_VAL = np.float32(-1e9)
_ZERO = np.float32(0.0)
_ONE = np.float32(1.0)
_NEG_ONE = np.float32(-1.0)
_HALF = np.float32(0.5)
#: ``sinh`` is evaluated as ``x`` below this magnitude: ``sinh`` is not available
#: in the ETL op set, and for ``|x| < 1e-4`` the correctly rounded float32 result
#: of ``sinh(x)`` is bit-exactly ``x`` (see ``_emit_unary``).
_SINH_LINEAR_BELOW = np.float32(1e-4)
_NaN = np.float32("nan")

#: Sentinel for "this value depends on a variable" during decoding/folding.
_SV = object()


def _f32(value: Any) -> np.float32:
    """Coerce to ``np.float32`` (the node-value storage of the kernels).

    Applied to the float64 intermediate of every software transcendental, this
    is the correctly rounded float32 result of the operation -- the same value
    the ETL graph emits for a non-constant operand (see the comment above
    ``_F64_DTYPE``).  Arithmetic and ``sqrt``/``1/x`` stay float32 on purpose:
    IEEE 754 defines them as correctly rounded already.
    """
    return np.float32(value)


def _f64(value: Any) -> np.float64:
    """Widen to ``np.float64`` (intermediate of a correctly rounded op)."""
    return np.float64(value)


def _fold_unary(func_id: int, a: np.float32) -> np.float32:
    """Host-side evaluation of one unary node; an unmatched id gives 0.0."""
    if func_id == Func.SIN:
        return _f32(np.sin(_f64(a)))
    if func_id == Func.COS:
        return _f32(np.cos(_f64(a)))
    if func_id == Func.TAN:
        return _f32(np.tan(_f64(a)))
    if func_id == Func.SINH:  # composed: ETL has no sinh (see _emit_unary)
        if np.abs(a) < _SINH_LINEAR_BELOW:
            return _f32(a)  # CR float32 of sinh is x; keeps sinh(-0.0) == -0.0
        return _f32(
            (np.exp(_f64(a)) - np.exp(-_f64(a))) * np.float64(0.5)
        )
    if func_id == Func.COSH:
        return _f32(
            (np.exp(_f64(a)) + np.exp(-_f64(a))) * np.float64(0.5)
        )
    if func_id == Func.TANH:
        return _f32(np.tanh(_f64(a)))
    if func_id == Func.LOG:
        return _f32(np.log(_f64(a)))  # log(0) = -inf, log(neg) = nan
    if func_id == Func.LOOSE_LOG:
        return _NEG_MAX_VAL if a == 0 else _f32(np.log(np.abs(_f64(a))))
    if func_id == Func.EXP:
        return _f32(np.exp(_f64(a)))  # overflow -> inf
    if func_id == Func.INV:
        return _NaN if a == 0 else _f32(_ONE / a)
    if func_id == Func.LOOSE_INV:
        b = _f32(np.copysign(_DELTA, a)) if np.abs(a) <= _DELTA else a
        return _f32(_ONE / b)
    if func_id == Func.NEG:
        return _f32(-a)
    if func_id == Func.ABS:
        return _f32(np.abs(a))
    if func_id == Func.SQRT:
        return np.sqrt(a)  # negative -> nan
    if func_id == Func.LOOSE_SQRT:
        return np.sqrt(np.abs(a))
    return _ZERO  # unmatched branch leaves top_val at its initializer


def _fold_binary(func_id: int, a: np.float32, b: np.float32) -> np.float32:
    """Host-side evaluation of one binary node; an unmatched id gives 0.0."""
    if func_id == Func.ADD:
        return _f32(a + b)
    if func_id == Func.SUB:
        return _f32(a - b)
    if func_id == Func.MUL:
        return _f32(a * b)
    if func_id == Func.DIV:
        return _NaN if b == 0 else _f32(a / b)
    if func_id == Func.LOOSE_DIV:
        d = _f32(np.copysign(_DELTA, b)) if np.abs(b) <= _DELTA else b
        return _f32(a / d)
    if func_id == Func.POW:
        # pow(0,0)=1, pow(-2,3)=-8, pow(-2,0.5)=nan
        return _f32(np.power(_f64(a), _f64(b)))
    if func_id == Func.LOOSE_POW:
        if a == 0 and b == 0:
            return _ZERO
        return _f32(np.power(np.abs(_f64(a)), _f64(b)))
    if func_id == Func.MAX:
        return a if a >= b else b  # ternary: max(v, NaN) = NaN
    if func_id == Func.MIN:
        return a if a <= b else b  # ternary: min(v, NaN) = NaN
    if func_id == Func.LT:
        return _ONE if a < b else _NEG_ONE
    if func_id == Func.GT:
        return _ONE if a > b else _NEG_ONE
    if func_id == Func.LE:
        return _ONE if a <= b else _NEG_ONE
    if func_id == Func.GE:
        return _ONE if a >= b else _NEG_ONE
    return _ZERO  # unmatched branch leaves top_val at its initializer


def _fold_ternary(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """Host-side ``if`` (the ternary branch always computes ``if``)."""
    return b if a > 0 else c


# ---------------------------------------------------------------------------
# graph construction: plain-Python emitters (run at trace time)
# ---------------------------------------------------------------------------


def _trace_const(value: np.float32) -> Any:
    """0-d float32 constant operand (ETL rejects numpy scalars as operands)."""
    return etl.constant(
        etl.core.tensor(np.asarray(np.float32(value), dtype=np.float32))
    )


def _trace_const_col(n_rows: int, value: np.float32) -> Any:
    """``(n_rows, 1)`` float32 constant (for output assembly)."""
    return etl.constant(
        etl.core.tensor(np.full((n_rows, 1), np.float32(value), dtype=np.float32))
    )


def _operand(value: Any) -> Any:
    """A symbolic value passes through; a folded constant becomes a 0-d const."""
    if isinstance(value, etl.SymbolicTensor):
        return value
    return _trace_const(value)


# ---------------------------------------------------------------------------
# correctly rounded float32 emission of the software transcendentals
# ---------------------------------------------------------------------------
#
# ``np.exp``/``np.log``/``np.sin``/... on float32 arrays are NOT correctly
# rounded (measured against a float64 reference: ``exp`` answers one ulp low in
# ~39% of the inputs, ``tanh`` ~33%, ``sinh`` ~26%, ``cosh`` ~22%, ``log`` ~4%,
# ``sin``/``cos`` ~15-17%, ``pow`` ~0.03%).  A one-ulp error is invisible in a
# single node value, but it is amplified by the derivative of the next node:
# ``exp`` feeding a ``tan`` next to a pole (``1 + tan^2`` up to ~1e5) turns a one
# ulp ``exp`` difference into ~1e-3 relative on the tree output -- orders of
# magnitude above the 1e-4 equivalence tolerance, and it is *not* reducible by
# the caller.
#
# The kernel contract stores every node value as float32 (ENGINE_SPEC section 2),
# so the fix is to make each software transcendental CORRECTLY ROUNDED in
# float32: evaluate it in float64 and round the result back to float32.  That is
# the best possible float32 answer -- every implementation that is not
# correctly rounded on its own (numpy's float32 kernels, torch's vectorized
# ones) is then within one ulp of the engine, and the residual disagreement
# survives only where the *other* implementation is the inaccurate one.
#
# Hardware-exact float32 operations are deliberately left alone (they are
# already correctly rounded, or exactly defined, and skipping them keeps the
# graph small): ``+ - * /``, ``sqrt``, ``abs``, ``neg``, ``copysign`` and every
# comparison/``select`` (which must stay bit-exact ``+-1``).

#: float64 dtype for the intermediate of a correctly rounded float32 op.
_F64_DTYPE = np.float64


def _copysign_delta(value: Any) -> Any:
    """``copysign(DELTA, x)`` for a symbolic ``x`` -- ETL has no ``copysign``.

    The sign of ``x`` is read as the sign of ``1 / x``, which is the only
    composition that gets ``x == -0.0`` right (all other sign tests -- ``x <
    0``, ``sign(x) < 0`` -- answer "+" for negative zero, while ``copysignf``
    answers "-").  This is reached only for ``|x| <= DELTA`` inputs.
    """
    negative = etl.less(
        etl.divide(_trace_const(_ONE), value), _trace_const(_ZERO)
    )
    return etl.select(negative, _trace_const(_NEG_DELTA), _trace_const(_DELTA))


def _to_f64(value: Any) -> Any:
    """Widen a float32 operand to float64 (intermediate of a CR op)."""
    return etl.cast(value, etl.float64)


def _round32(value: Any) -> Any:
    """Round a float64 intermediate back to the float32 node-value storage.

    The cast is ``astype``-style (round to nearest, ties to even) and preserves
    NaN / ``+-inf``, so the engine's NaN/inf placement is untouched.
    """
    return etl.cast(value, etl.float32)


def _trace_const_f64(value: Any) -> Any:
    """0-d float64 constant operand (the ``* 0.5`` of the sinh/cosh composition)."""
    return etl.constant(
        etl.core.tensor(np.asarray(np.float64(value), dtype=np.float64))
    )


def _emit_unary(func_id: int, a: Any) -> Any:
    """One unary node: fold when the operand is constant, emit ops otherwise.

    The software transcendentals (``sin``/``cos``/``tan``/``tanh``/``log``/
    ``exp``/``sinh``/``cosh``) are emitted as ``round_f32(op(f64(v)))``, i.e.
    correctly rounded in float32 -- see the comment above ``_F64_DTYPE``.  The
    hardware-exact ops (``neg``/``abs``/``sqrt``/``1/x``) stay float32.
    """
    if not isinstance(a, etl.SymbolicTensor):
        return _fold_unary(func_id, a)
    if func_id == Func.SIN:
        return _round32(etl.sin(_to_f64(a)))
    if func_id == Func.COS:
        return _round32(etl.cos(_to_f64(a)))
    if func_id == Func.TAN:
        return _round32(etl.tan(_to_f64(a)))
    if func_id == Func.SINH:  # composed: ETL has no sinh
        w = _to_f64(a)
        return _round32(
            etl.multiply(
                etl.subtract(etl.exp(w), etl.exp(etl.negate(w))),
                _trace_const_f64(_HALF),
            )
        )
    if func_id == Func.COSH:  # composed: ETL has no cosh
        w = _to_f64(a)
        return _round32(
            etl.multiply(
                etl.add(etl.exp(w), etl.exp(etl.negate(w))),
                _trace_const_f64(_HALF),
            )
        )
    if func_id == Func.TANH:
        return _round32(etl.tanh(_to_f64(a)))
    if func_id == Func.LOG:
        return _round32(etl.log(_to_f64(a)))
    if func_id == Func.LOOSE_LOG:
        return etl.select(
            etl.equal(a, _trace_const(_ZERO)),
            _trace_const(_NEG_MAX_VAL),
            _round32(etl.log(etl.abs(_to_f64(a)))),
        )
    if func_id == Func.EXP:
        return _round32(etl.exp(_to_f64(a)))
    if func_id == Func.INV:
        return etl.select(
            etl.equal(a, _trace_const(_ZERO)),
            _trace_const(_NaN),
            etl.divide(_trace_const(_ONE), a),
        )
    if func_id == Func.LOOSE_INV:
        safe = etl.select(
            etl.less_equal(etl.abs(a), _trace_const(_DELTA)),
            _copysign_delta(a),
            a,
        )
        return etl.divide(_trace_const(_ONE), safe)
    if func_id == Func.NEG:
        return etl.negate(a)
    if func_id == Func.ABS:
        return etl.abs(a)
    if func_id == Func.SQRT:
        return etl.sqrt(a)
    if func_id == Func.LOOSE_SQRT:
        return etl.sqrt(etl.abs(a))
    return _ZERO  # unmatched branch leaves top_val at its initializer


def _emit_binary(func_id: int, a: Any, b: Any) -> Any:
    """One binary node: fold when both operands are constant, else emit ops."""
    sym_a = isinstance(a, etl.SymbolicTensor)
    sym_b = isinstance(b, etl.SymbolicTensor)
    if not sym_a and not sym_b:
        return _fold_binary(func_id, a, b)
    if func_id == Func.ADD:
        return etl.add(_operand(a), _operand(b))
    if func_id == Func.SUB:
        return etl.subtract(_operand(a), _operand(b))
    if func_id == Func.MUL:
        return etl.multiply(_operand(a), _operand(b))
    if func_id == Func.DIV:
        if not sym_b:  # constant divisor: the 0 test is a plain host branch
            return _NaN if b == 0 else etl.divide(_operand(a), _operand(b))
        quotient = etl.divide(_operand(a), _operand(b))
        return etl.select(
            etl.equal(b, _trace_const(_ZERO)), _trace_const(_NaN), quotient
        )
    if func_id == Func.LOOSE_DIV:
        if not sym_b:  # constant divisor: sanitize on the host, exactly
            d = _f32(np.copysign(_DELTA, b)) if np.abs(b) <= _DELTA else b
            return etl.divide(_operand(a), _trace_const(d))
        safe = etl.select(
            etl.less_equal(etl.abs(b), _trace_const(_DELTA)),
            _copysign_delta(b),
            b,
        )
        return etl.divide(_operand(a), safe)
    if func_id == Func.POW:
        return _round32(etl.power(_to_f64(_operand(a)), _to_f64(_operand(b))))
    if func_id == Func.LOOSE_POW:
        return etl.select(
            etl.logical_and(
                etl.equal(_operand(a), _trace_const(_ZERO)),
                etl.equal(_operand(b), _trace_const(_ZERO)),
            ),
            _trace_const(_ZERO),
            _round32(
                etl.power(etl.abs(_to_f64(_operand(a))), _to_f64(_operand(b)))
            ),
        )
    if func_id == Func.MAX:
        return etl.select(
            etl.greater_equal(_operand(a), _operand(b)), _operand(a), _operand(b)
        )
    if func_id == Func.MIN:
        return etl.select(
            etl.less_equal(_operand(a), _operand(b)), _operand(a), _operand(b)
        )
    if func_id == Func.LT:
        return etl.select(
            etl.less(_operand(a), _operand(b)),
            _trace_const(_ONE),
            _trace_const(_NEG_ONE),
        )
    if func_id == Func.GT:
        return etl.select(
            etl.greater(_operand(a), _operand(b)),
            _trace_const(_ONE),
            _trace_const(_NEG_ONE),
        )
    if func_id == Func.LE:
        return etl.select(
            etl.less_equal(_operand(a), _operand(b)),
            _trace_const(_ONE),
            _trace_const(_NEG_ONE),
        )
    if func_id == Func.GE:
        return etl.select(
            etl.greater_equal(_operand(a), _operand(b)),
            _trace_const(_ONE),
            _trace_const(_NEG_ONE),
        )
    return _ZERO  # unmatched branch leaves top_val at its initializer


def _emit_ternary(a: Any, b: Any, c: Any) -> Any:
    """One ternary node (always ``if``): host pick when the condition folds."""
    if not isinstance(a, etl.SymbolicTensor):
        return b if a > 0 else c
    return etl.select(
        etl.greater(a, _trace_const(_ZERO)), _operand(b), _operand(c)
    )


def _emit_func(kind: int, func_id: int, children: Sequence[Any]) -> Any:
    """Dispatch one function token.  An unmatched id gives 0.0 (kernel spec)."""
    if kind == _K_UFUNC:
        return _emit_unary(func_id, children[0])
    if kind == _K_BFUNC:
        return _emit_binary(func_id, children[0], children[1])
    return _emit_ternary(children[0], children[1], children[2])


def _acc_add(acc: Any, value: Any) -> Any:
    """Accumulate one out-node contribution (``s_outs[o] += value``)."""
    if acc is None:
        return value
    if not isinstance(acc, etl.SymbolicTensor) and not isinstance(
        value, etl.SymbolicTensor
    ):
        return _f32(np.float32(acc) + np.float32(value))
    return etl.add(_operand(acc), _operand(value))


def _assemble_outputs(accs: Sequence[Any], n_rows: int) -> Any:
    """Stack the accumulated outputs into the ``(n_rows, out_len)`` result."""
    cols = []
    for acc in accs:
        if isinstance(acc, etl.SymbolicTensor):
            cols.append(acc)
        elif acc is None:
            cols.append(_trace_const_col(n_rows, _ZERO))
        else:
            cols.append(_trace_const_col(n_rows, np.float32(acc)))
    return etl.concatenate(cols, axis=1)


def _walk_tokens(tokens: Sequence[Sequence[int]], output_len: int, x: Any) -> Any:
    """Run one static program: emit ETL ops, or evaluate on the host.

    The walk visits the token stream in reverse prefix order and keeps a value
    stack -- the same operand discipline as ``_treeGPEvalByStack`` (the first
    pop is the leftmost child).  Values are either ``etl.SymbolicTensor``s or
    plain float32 constants that were folded at decode time; the emitters fold
    eagerly whenever every operand is constant, so this one function is also
    the host evaluator.

    Args:
        tokens: the static token stream.
        output_len: number of outputs (``> 1`` selects the multi-output
            protocol of ENGINE_SPEC section 4.3).
        x: the symbolic ``(N, input_len)`` input for tracing, or ``None`` for
            host evaluation -- the program must then contain no ``K_VAR``
            token, every operand is a float and the accumulated outputs (multi
            output) or the root value (single output) are returned as floats.
    """
    multi = output_len > 1
    n_rows = 0 if x is None else int(x.shape[0])
    stack: List[Any] = []
    accs: List[Any] = [None] * output_len if multi else []
    for tok in reversed(tokens):
        kind = tok[0]
        if kind == _K_CONST:
            stack.append(_bits_to_float(tok[1]))
            continue
        if kind == _K_VAR:
            stack.append(etl.slice(x, (0, tok[1]), (n_rows, 1)))
            continue
        arity = kind - 1
        if len(stack) < arity:
            raise RuntimeError(
                "internal error: static program operand underflow (a token "
                "stream that skipped decode-time validation?)"
            )
        children = [stack.pop() for _ in range(arity)]
        own = _emit_func(kind, tok[1], children)
        if multi:
            out_index = tok[2]
            if 0 <= out_index < output_len:
                accs[out_index] = _acc_add(accs[out_index], own)
            stack.append(children[-1])  # every function node passes up its rmost child
        else:
            stack.append(own)
    if multi:
        if x is None:
            return [_ZERO if acc is None else _f32(acc) for acc in accs]
        return _assemble_outputs(accs, n_rows)
    if len(stack) != 1:
        raise RuntimeError(
            "internal error: static program did not reduce to a single value"
        )
    root = stack[0]
    if x is None:
        return root
    if not isinstance(root, etl.SymbolicTensor):  # constant root (defensive)
        return _trace_const_col(n_rows, np.float32(root))
    return root


def _trace_program(program: Tuple[int, Sequence[Sequence[int]]], x: Any) -> Any:
    """Specialize ``(out_len, tokens)`` into ETL ops over the input ``x``."""
    out_len, tokens = program
    return _walk_tokens(tokens, int(out_len), x)


@etl.defn
def _eval_step(x, program):
    """One tree, specialized: ``x`` is ``(N, input_len)`` -> ``(N, out_len)``.

    ``program`` is a static ``(out_len, tokens)`` structure; it is decoded at
    trace time into ordinary ETL ops.  Module-level on purpose: ``build_cached``
    keys on ``id(fn)``, and a closure would be a permanent cache miss.
    """
    return _trace_program(program, x)


# ---------------------------------------------------------------------------
# backend resolution & memoized builds
# ---------------------------------------------------------------------------


def _resolve_backend(
    backend: Optional[str], iree_device_index: Optional[int]
) -> Tuple[str, Any, Dict[str, Any]]:
    """Resolve ``(backend_name, device, compile_options)``.

    Precedence: explicit argument > environment > default.  ``"numpy"`` runs on
    the host with ``Device("cpu", 0)`` and no options; ``"iree"`` is the
    CUDA-HAL device backend (see the module docstring).  Both are always
    resolved to concrete values so that ``build_cached`` never consults the
    ambient ``ETL_*`` environment.
    """
    name = backend if backend else os.environ.get(BACKEND_ENV, "").strip()
    if not name:
        name = SUPPORTED_BACKENDS[0]
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported backend {name!r}: evogp.tree.etl_engine supports "
            f"{list(SUPPORTED_BACKENDS)} (set {BACKEND_ENV!r} or pass backend=)"
        )
    if name == "numpy":
        return name, _CPU, {}
    if iree_device_index is None:
        iree_device_index = _env_int(IREE_DEVICE_ENV, 0)
    if int(iree_device_index) < 0:
        raise ValueError(f"iree_device_index={iree_device_index!r} must be >= 0")
    opt_level = os.environ.get(IREE_OPT_LEVEL_ENV, "").strip() or "O1"
    if opt_level not in ("O0", "O1"):
        raise ValueError(
            f"{IREE_OPT_LEVEL_ENV}={opt_level!r}: only 'O0' and 'O1' are supported"
        )
    device = etl.core.Device("cuda", int(iree_device_index))
    return name, device, {"target_backends": ["cuda"], "opt_level": opt_level}


def _executable(
    program: Tuple[int, Sequence[Sequence[int]]],
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Any:
    """Memoized per-(structure, shape) executable (a hit returns identity)."""
    spec = etl.TensorSpec((int(n_rows), int(input_len)), etl.float32)
    if backend_name == "numpy":
        return build_cached(
            _eval_step, spec, program, backend=backend_name, device=device, cache=CACHE
        )
    return build_cached(
        _eval_step,
        spec,
        program,
        backend=backend_name,
        device=device,
        cache=CACHE,
        **options,
    )


# ---------------------------------------------------------------------------
# tensor plumbing (read-only; never moves or mutates caller tensors)
# ---------------------------------------------------------------------------


def _as_numpy(tensor: Any, dtype: Any, name: str) -> np.ndarray:
    """Read any tensor (torch any device / numpy) as a host array of ``dtype``."""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(tensor)
        except Exception as exc:
            raise TypeError(
                f"{name} must be a torch.Tensor or a numpy array, got "
                f"{type(tensor).__name__}"
            ) from exc
    if array.dtype != dtype:
        array = array.astype(dtype)
    return np.ascontiguousarray(array)


def _input_to_numpy(tensor: Any, input_len: int, name: str) -> np.ndarray:
    """Validate/copy a ``(N, input_len)`` float32 input (never mutates it)."""
    array = _as_numpy(tensor, np.float32, name)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (rows, {input_len}), got shape {array.shape}"
        )
    if array.shape[1] != input_len:
        raise ValueError(
            f"{name} has {array.shape[1]} columns but the population expects "
            f"input_len={input_len}"
        )
    return array


# ---------------------------------------------------------------------------
# decoding (host-side; validation + constant folding)
# ---------------------------------------------------------------------------


def _decode_row(
    values: np.ndarray,
    types: np.ndarray,
    sizes: np.ndarray,
    bits: np.ndarray,
    row: int,
    input_len: int,
    output_len: int,
) -> DecodedTree:
    """Decode + validate one row; raises :class:`ValueError` on malformed input.

    One reverse-prefix pass computes, per node: its kind/function id/out index,
    its subtree size (checked against the tensor), its own folded value (or
    ``_SV`` when it depends on a variable), the value it passes up to its
    parent in multi-output mode, and whether its subtree contains an effective
    out-node.  The folded values then let the emitted token stream collapse
    whole constant subtrees into a single ``K_CONST`` token.
    """
    row_len = int(values.shape[0])
    size0 = int(sizes[0]) if row_len else 0
    if size0 <= 0:
        raise ValueError(
            f"row {row}: node 0: malformed subtree_size[0]={size0} "
            "(the tree length must be positive)"
        )
    if size0 > row_len:
        raise ValueError(
            f"row {row}: node 0: subtree_size[0]={size0} exceeds the row "
            f"length {row_len}"
        )
    multi = output_len > 1

    kinds = [0] * size0
    funcs = [0] * size0
    out_indices = [-1] * size0
    consts = [0.0] * size0
    node_sizes = [1] * size0
    repl: List[Optional[np.float32]] = [None] * size0
    # reverse-walk stack frames: (own value, passed-up value, subtree size,
    # subtree has an effective out-node); values are float32 or ``_SV``.
    stack: List[Tuple[Any, Any, int, bool]] = []

    for i in range(size0 - 1, -1, -1):
        raw = int(types[i])
        tagged = bool(raw & NType.OUT_NODE)
        if not multi:
            if tagged:
                raise ValueError(
                    f"row {row}: node {i}: node_type={raw} carries the OUT_NODE "
                    "flag but output_len == 1 -- tagged nodes are only defined "
                    "for multi-output trees (the old kernel corrupted its "
                    "operand stack here; see the module docstring)"
                )
            base = raw
        else:
            base = raw & NType.TYPE_MASK
        if base == NType.VAR or base == NType.CONST:
            # leaves: in multi-output mode a tag is masked off and ignored
            if base == NType.VAR:
                index_value = float(values[i])
                if not math.isfinite(index_value):
                    raise ValueError(
                        f"row {row}: node {i}: variable index node_value="
                        f"{values[i]!r} is not finite"
                    )
                index = int(index_value)  # C truncation toward zero
                if index < 0 or index >= input_len:
                    raise ValueError(
                        f"row {row}: node {i}: variable index {index} (from "
                        f"node_value={values[i]!r}) is outside [0, {input_len})"
                    )
                kinds[i] = _K_VAR
                funcs[i] = index
                stack.append((_SV, _SV, 1, False))
            else:
                value = np.float32(values[i])
                kinds[i] = _K_CONST
                consts[i] = value
                stack.append((value, value, 1, False))
            continue

        # -- function node ---------------------------------------------------
        if base == NType.UFUNC:
            arity = 1
        elif base == NType.BFUNC:
            arity = 2
        else:  # TFUNC and any unknown base type: the kernel's ternary branch
            arity = 3
        base_name = _BASE_NAMES.get(int(base), f"base type {int(base)}")
        if len(stack) < arity:
            raise ValueError(
                f"row {row}: node {i}: {base_name} node needs {arity} operand(s) "
                f"but only {len(stack)} node(s) follow it in the prefix encoding "
                f"-- truncated tree (subtree_size[0]={size0})"
            )
        children = [stack.pop() for _ in range(arity)]  # [leftmost .. rightmost]
        kind = (_K_UFUNC, _K_BFUNC, _K_TFUNC)[arity - 1]
        kinds[i] = kind
        if multi and tagged:
            packed = int(bits[i])  # 4-byte view of node_value (little-endian)
            funcs[i] = packed & 0xFFFF
            out_indices[i] = packed >> 16  # C arithmetic shift
        elif arity == 3:
            funcs[i] = 0  # the ternary branch ignores the function id
        else:
            func_id = float(values[i])
            if not math.isfinite(func_id):
                raise ValueError(
                    f"row {row}: node {i}: function id node_value={values[i]!r} "
                    "is not finite"
                )
            funcs[i] = int(func_id)  # C truncation toward zero
        size = 1
        for child in children:
            size += child[2]
        declared = int(sizes[i])
        if declared != size:
            raise ValueError(
                f"row {row}: node {i}: subtree_size={declared} disagrees with the "
                f"tree structure (1 + children sizes = {size})"
            )
        node_sizes[i] = size
        own: Any = _SV
        if kind == _K_UFUNC:
            if children[0][0] is not _SV:
                own = _fold_unary(funcs[i], children[0][0])
        elif kind == _K_BFUNC:
            if children[0][0] is not _SV and children[1][0] is not _SV:
                own = _fold_binary(funcs[i], children[0][0], children[1][0])
        elif (
            children[0][0] is not _SV
            and children[1][0] is not _SV
            and children[2][0] is not _SV
        ):
            own = _fold_ternary(children[0][0], children[1][0], children[2][0])
        passed_up = children[-1][1]  # rightmost child's value
        has_out = (0 <= out_indices[i] < output_len) or any(
            child[3] for child in children
        )
        if own is _SV:
            replacement = None
        elif not multi:
            replacement = own  # single output: the node's own value is the result
        elif has_out:
            replacement = None  # folding would drop an accumulation
        else:
            # multi output: the subtree only ever contributes its passed-up value
            replacement = passed_up if passed_up is not _SV else None
        repl[i] = replacement
        stack.append((own, passed_up, size, has_out))

    if len(stack) != 1:
        used = int(stack[-1][2]) if stack else 0
        raise ValueError(
            f"row {row}: node 0: subtree_size[0]={size0} but the tree rooted at "
            f"node 0 only uses {used} node(s) -- {size0 - used} trailing node(s) "
            "do not belong to the tree"
        )

    # -- emit the static token stream (reverse prefix -> prefix) -------------
    tokens: List[Tuple[int, ...]] = []
    i = 0
    while i < size0:
        replacement = repl[i]
        if replacement is not None:
            tokens.append((_K_CONST, _float_to_bits(replacement)))
            i += node_sizes[i]
            continue
        kind = kinds[i]
        if kind == _K_VAR:
            tokens.append((_K_VAR, funcs[i]))
        elif kind == _K_CONST:
            tokens.append((_K_CONST, _float_to_bits(consts[i])))
        else:
            tokens.append((kind, funcs[i], out_indices[i]))
        i += 1

    collapsed = size0 - len(tokens)
    if any(tok[0] == _K_VAR for tok in tokens):
        return DecodedTree(
            program=(output_len, tuple(tokens)),
            const_out=None,
            node_count=size0,
            collapsed_nodes=collapsed,
        )
    # A fully constant program is never compiled: evaluate it on the host with
    # the very same walk (its shape is fixed, every operand is a float).
    result = _walk_tokens(tokens, output_len, None)
    if output_len > 1:
        const_out = tuple(_float_to_bits(value) for value in result)
    else:
        const_out = (_float_to_bits(result),)
    return DecodedTree(
        program=None, const_out=const_out, node_count=size0, collapsed_nodes=collapsed
    )


def decode_population(
    batch_node_value: Any,
    batch_node_type: Any,
    batch_subtree_size: Any,
    input_len: int,
    output_len: int,
) -> Population:
    """Host-side decode of a whole population.

    Args:
        batch_node_value: ``(pop_size, max_tree_len)`` float32 node values (a
            torch tensor on any device or a numpy array).
        batch_node_type: ``(pop_size, max_tree_len)`` int16 node types.
        batch_subtree_size: ``(pop_size, max_tree_len)`` int16 subtree sizes.
        input_len: number of variables of the problem (``> 0``).
        output_len: number of outputs (``>= 1``).

    Returns:
        An opaque :class:`Population` for :func:`evaluate` / :func:`sr_fitness`.
        No GPU work happens here and the caller's tensors are never moved or
        modified (they are read back with ``.cpu()`` / ``np.asarray``).

    Raises:
        ValueError: malformed tree (row index and node index in the message):
            non-positive ``subtree_size[0]``, a tree longer than the row,
            structure beyond the declared tree, inconsistent per-node subtree
            sizes, a variable index outside ``[0, input_len)``, a non-finite
            index/function id, or an OUT_NODE tag in a single-output tree.
        TypeError: population tensors are not tensors / not 2-D.
    """
    input_len = int(input_len)
    output_len = int(output_len)
    if input_len <= 0:
        raise ValueError(f"input_len={input_len} must be positive")
    if output_len <= 0:
        raise ValueError(f"output_len={output_len} must be positive")
    values = _as_numpy(batch_node_value, np.float32, "batch_node_value")
    types = _as_numpy(batch_node_type, np.int16, "batch_node_type")
    sizes = _as_numpy(batch_subtree_size, np.int16, "batch_subtree_size")
    if values.ndim != 2:
        raise TypeError(
            "batch_node_value must be 2-D (pop_size, max_tree_len), got shape "
            f"{values.shape}"
        )
    if types.shape != values.shape or sizes.shape != values.shape:
        raise ValueError(
            "batch_node_value/batch_node_type/batch_subtree_size must have the "
            f"same shape, got {values.shape}, {types.shape}, {sizes.shape}"
        )
    bits = values.view(np.int32)
    with _quiet():  # decode-time folding does IEEE edge-case math on purpose
        trees = [
            _decode_row(
                values[row],
                types[row],
                sizes[row],
                bits[row],
                row,
                input_len,
                output_len,
            )
            for row in range(values.shape[0])
        ]
    return Population(
        input_len=input_len,
        output_len=output_len,
        pop_size=int(values.shape[0]),
        trees=tuple(trees),
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _run_tree(
    tree: DecodedTree,
    xin: np.ndarray,
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
    prepared: List[Any],
) -> np.ndarray:
    """Build-or-hit one tree and run it; returns ``(n_rows, out_len)`` float32."""
    exe = _executable(tree.program, n_rows, input_len, backend_name, device, options)
    if backend_name == "numpy":
        with _quiet():
            result = etl.run(exe, xin, tree.program)
        return np.asarray(result.numpy(), dtype=np.float32).reshape(n_rows, -1)
    if not prepared:  # one host->device copy reused for every tree in the call
        prepared.append(etl.from_numpy(xin).to(device))
    with _quiet():
        result = etl.run(exe, prepared[0], tree.program)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32).reshape(n_rows, -1)


def evaluate(
    population: Population,
    x: Any,
    *,
    per_row: bool,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate a decoded population on ``x``.

    ``per_row=True``  -- ``x`` is ``(pop_size, input_len)`` and tree ``i`` sees
    row ``i``; the result is ``(pop_size, output_len)``.

    ``per_row=False`` -- ``x`` is ``(N, input_len)`` and every tree is evaluated
    on all ``N`` rows; the result is ``(pop_size, N, output_len)``.

    The result is a FRESH, contiguous, ``requires_grad=False`` float32 tensor on
    ``x``'s device (``out_device`` overrides it, e.g. ``"cuda"`` /
    ``torch.device("cuda:1")``).  ``x`` may be a torch tensor on any device or a
    numpy array; it is cast to float32, copied, and never modified.  The
    population's tensors are not touched either.
    """
    pop = _require_population(population)
    xt = _input_to_numpy(x, pop.input_len, "x")
    if per_row:
        if xt.shape[0] != pop.pop_size:
            raise ValueError(
                f"per_row=True needs x.shape[0] == {pop.pop_size} (pop_size), "
                f"got {xt.shape[0]}"
            )
        n_rows = 1
    else:
        n_rows = int(xt.shape[0])
    backend_name, device, options = _resolve_backend(backend, iree_device_index)

    rows = np.empty((pop.pop_size, n_rows, pop.output_len), dtype=np.float32)
    prepared: List[Any] = []
    for index, tree in enumerate(pop.trees):
        if tree.const_out is not None:
            rows[index, :, :] = _bits_to_float_arr(tree.const_out)
            continue
        xin = xt[index : index + 1] if per_row else xt
        rows[index] = _run_tree(
            tree, xin, n_rows, pop.input_len, backend_name, device, options, prepared
        )
    if per_row:
        rows = np.ascontiguousarray(rows[:, 0, :])

    tensor = torch.from_numpy(rows).contiguous().detach().requires_grad_(False)
    target = out_device if out_device is not None else getattr(x, "device", None)
    if target is not None and torch.device(target) != tensor.device:
        tensor = tensor.to(target)
    return tensor


def sr_fitness(
    population: Population,
    inputs: Any,
    labels: Any,
    use_MSE: bool = True,
    *,
    out_device: Optional[Any] = None,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> torch.Tensor:
    """Symbolic-regression fitness of a population (the old ``tree_SR_fitness``).

    ``inputs`` is ``(N, input_len)``, ``labels`` is ``(N, output_len)``; the
    result is a ``(pop_size,)`` float32 POSITIVE loss on ``inputs``' device
    (``out_device`` overrides)::

        err  = (label - prediction) ** 2        if use_MSE
        err  = |label - prediction|             otherwise
        loss = sum_N sum_out err / N

    Outputs are summed but never divided by ``output_len``; NaN/inf propagate
    unsanitized; the loss is NOT negated (the caller applies the
    higher-is-better convention).  The returned tensor is fresh, contiguous and
    ``requires_grad=False``.  ``inputs``/``labels`` are cast to float32 and
    copied, never mutated.
    """
    pop = _require_population(population)
    xin = _input_to_numpy(inputs, pop.input_len, "inputs")
    if xin.shape[0] <= 0:
        raise ValueError("sr_fitness needs at least one data point, got 0 rows")
    label = _as_numpy(labels, np.float32, "labels")
    if label.ndim != 2 or label.shape != (xin.shape[0], pop.output_len):
        raise ValueError(
            f"labels must have shape ({xin.shape[0]}, {pop.output_len}), got "
            f"{label.shape}"
        )
    prediction = evaluate(
        pop,
        inputs,
        per_row=False,
        out_device=out_device,
        backend=backend,
        iree_device_index=iree_device_index,
    )
    target = torch.from_numpy(label).to(prediction.device)
    diff = target.unsqueeze(0) - prediction  # (pop_size, N, output_len)
    err = diff * diff if use_MSE else diff.abs()
    loss = err.sum(dim=(1, 2)) / float(xin.shape[0])
    return loss.contiguous().detach().requires_grad_(False)
