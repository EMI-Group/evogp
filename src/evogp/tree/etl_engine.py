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
   population next generation) is a pure cache hit; a bounded value-keyed
   in-process memo in front of it skips ``build_cached``'s own per-call key
   derivation (see "Per-tree executable memo" below).
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

Benchmark-support runner (not part of the public route)
-------------------------------------------------------
:func:`build_specialized_runner` / :class:`SpecializedRunner` expose the SAME
engine as a build-once / run-many object for benchmarking harnesses: the build
phase decodes the population and resolves every per-tree executable, ``.run()``
executes only the compiled forward pass (one ``etl.run`` per non-constant tree,
staged per-tree inputs, raw results, no readback and no cache lookup at all).
It changes nothing about ``evaluate`` / ``sr_fitness``, is called by nothing in
the library and is documented -- with the dense twin,
``dense_engine.build_dense_runner`` -- in ``RUNNER_HOOKS.md``.

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
* Fitness does not round-trip the prediction: :func:`sr_fitness` converts
  ``inputs`` and ``labels`` to host once, reduces the host rows its own
  evaluation produced -- with numpy, in place, in float32 (torch's CPU ops on
  that tensor open an intra-op thread pool that then starves the
  single-threaded ETL/numpy interpreter; see the function docstring) -- and
  moves only the ``(pop_size,)`` loss to the input device.  :func:`evaluate`
  keeps returning its full ``(pop_size, N, output_len)`` tensor on the input
  device as documented above.

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

Per-tree executable memo (in-process front table)
-------------------------------------------------
``build_cached`` is a memoized build, but its HIT path still re-derives the
cache key on every call (program re-encode + sha256 -- replayed from the
ETL-side content-fingerprint memo when the program content repeats, so a
program that is new or has fallen out of that memo pays the full walk):
measured (while the ETL fast path was still one-slot, i.e. every DIFFERENT
program paid the walk) 160-207 us per tree on the two probe fixtures
(30-63-node trees) when consecutive lookups carry DIFFERENT programs
-- which is exactly the per-tree loop's access pattern, and the population's
store of distinct programs is what keeps the derivation full -- against
2.4-2.8 us per tree for a plain value-keyed dict lookup doing the same work
(see ``warm_path_findings.md`` section 11; the battery is
``executable_memo_check.py``).  The ETL side has since cut both ends of that
per-call cost (the fingerprint memo above, plus a run-path static-validation
fast path), so what this table buys now is a cheaper lookup, not an escape
from the key walk: ~2.4-2.8 us against a ~17-25 us ``build_cached`` hit.  The
engine therefore keeps a bounded LRU memo
(:data:`_EXECUTABLE_MEMO`, one entry per executable, ``maxsize`` =
:data:`CACHE`'s) keyed by
:func:`_executable_key` -- everything ``build_cached`` keys on: ``id(_eval_step)``,
the ``program`` value, ``n_rows``, ``input_len``, the backend name, the device
and the canonicalized options.  A repeated lookup for the same
*(structure, shape, backend, device, options)* returns the very ``Executable``
:data:`CACHE` holds (identity, as on a ``build_cached`` hit) without touching
ETL at all, and only a memo miss constructs the ``TensorSpec`` and re-derives
``build_cached``'s own key.  The key is value-typed, so a re-decoded elite tree
hits.

Consequences: (a) ``CACHE.hits`` no longer counts repeated ``_executable``
lookups -- :func:`cache_stats` reports the front table as ``memo_hits`` /
``memo_misses`` / ``memo_entries`` / ``memo_maxsize`` (memo misses are the
lookups that went on to ``build_cached``, so they still land in ``CACHE``'s
counters); (b) :func:`clear_cache` resets both tables in lockstep (the ETL
documented "clear the cache after changing a process-env compile knob" advice
therefore holds unchanged); (c) the memo holds a strong reference to its keys
and values, so it can pin up to ``maxsize`` executables that ``CACHE`` has
already evicted -- bounded by design and the only cost of the front table;
(d) ambient ``ETL_TARGET_BACKENDS`` is the one env knob that still enters
``build_cached``'s key for the numpy path (``_resolve_backend_device`` adds it
when no explicit ``target_backends`` is passed) and is deliberately NOT part of
the memo key: the numpy backend documents the option as accepted-and-ignored
and the module always passes backend/device/``target_backends`` itself, so it
cannot change what this module computes (see :func:`_executable_key`).

ADOPTION POINT: the memo is a deliberate stopgap for the ETL-side ``etl.jit`` /
pre-bound-runner machinery, which has since landed (``etl.jit`` ->
``JitFunction`` / ``BoundJit``, exported from ``etl/__init__.py``; ETL branch
``evogit-agent-T1-A160``, on top of the ``main`` checkout installed here) --
a pre-bound runner keeps the bound statics out of the per-call key derivation
and the run boundary now validates them through the per-executable
content-fingerprint fast path instead of an O(#tokens) per-leaf walk.
Adopting it still means deleting :data:`_EXECUTABLE_MEMO` (and the ``memo_*``
counters in :func:`cache_stats`); nothing else in the engine depends on it, and
the swap is deliberately NOT made here.

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
* ``sr_fitness`` sums the per-(data point, output) errors on the HOST with
  numpy's pairwise float32 reduction instead of the kernel's device-side block
  reduction, and only the ``(pop_size,)`` loss reaches the input device (the
  prediction is never uploaded); same formula, float32 rounding differences of
  order 1e-6 relative.

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
import threading
from collections import OrderedDict
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
    "SpecializedRunner",
    "build_specialized_runner",
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


class _ExecutableMemo(OrderedDict):
    """Bounded LRU ``key -> Executable`` memo in front of :data:`CACHE`.

    ``etl.build_cached`` memoizes the build, but its hit path re-derives the
    cache key on every call (program re-encode + sha256, replayed from the
    ETL-side content-fingerprint memo when the program content repeats); this
    table is the plain dict lookup that skips it (see the module docstring's
    "Per-tree executable memo" section for the measurements and the full
    contract).

    * A hit returns the very ``Executable`` object :data:`CACHE` holds
      (``build_cached`` returns identity on a hit, and an executable is
      immutable and stateless -- the static program is re-passed by value at
      every ``etl.run``), so the memo can never serve a different executable
      than ``build_cached`` would for the same key.
    * ``maxsize`` bounds the table (plain LRU via :meth:`popitem`); it is set to
      :data:`CACHE`'s ``maxsize`` so both tables cover the same working set.
      Entries hold a strong reference to their key (the program tuple) and
      value, so the memo can pin up to ``maxsize`` executables that
      :data:`CACHE` has already evicted -- bounded, and the only cost of the
      front table (an entry whose executable is still CACHE-resident costs one
      key reference).
    * The counters are what :func:`cache_stats` reports; :meth:`reset` is
      :func:`clear_cache`'s half of the invalidation.
    * Like :data:`CACHE` -- whose ``CompileCache`` guards its dict and counters
      with a lock -- every operation here runs under one :class:`threading.RLock`,
      so a caller that evaluates from several threads gets the same
      first-insert-wins behaviour as ``build_cached`` (a duplicated build would
      be identical by construction: the key determines the executable).
    """

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = int(maxsize)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def lookup(self, key: Any) -> Any:
        """The memoized executable for ``key``, or ``None`` (counter updated).

        ``None`` is the "not memoized" answer: executables are never ``None``.
        """
        with self._lock:
            try:
                executable = self[key]
            except KeyError:
                self.misses += 1
                return None
            self.hits += 1
            self.move_to_end(key)  # LRU: a hit refreshes the entry
            return executable

    def store(self, key: Any, executable: Any) -> None:
        """Insert ``key`` -> ``executable`` (never for a present key) and trim."""
        with self._lock:
            self[key] = executable
            while len(self) > self.maxsize:
                self.popitem(last=False)

    def reset(self) -> None:
        """Drop every entry and zero the hit/miss counters."""
        with self._lock:
            OrderedDict.clear(self)
            self.hits = 0
            self.misses = 0


#: LRU memo of per-(program, shape, backend, device, options) executables;
#: populated by :func:`_executable`, keyed by :func:`_executable_key`.
_EXECUTABLE_MEMO = _ExecutableMemo(CACHE.maxsize)


def cache_stats() -> Dict[str, int]:
    """Snapshot of the shared :data:`CACHE` and the executable front memo.

    ``hits``/``misses``/``entries``/``maxsize`` describe the ``etl.CompileCache``
    (the compile table); ``memo_*`` describe :data:`_EXECUTABLE_MEMO`, the
    in-process value-keyed table in front of it: ``memo_hits`` are lookups
    served without touching the compile cache at all, ``memo_misses`` are the
    lookups that went on to ``build_cached`` (and are therefore also counted in
    ``CACHE``'s ``hits``/``misses``), ``memo_entries``/``memo_maxsize`` its size
    and bound.  ``CACHE.hits`` therefore no longer grows with repeated
    ``_executable`` lookups -- that is the memo working as documented, not a
    dropped hit.  Every counter is reset by :func:`clear_cache`.
    """
    return {
        "hits": CACHE.hits,
        "misses": CACHE.misses,
        "entries": len(CACHE),
        "maxsize": CACHE.maxsize,
        "memo_hits": _EXECUTABLE_MEMO.hits,
        "memo_misses": _EXECUTABLE_MEMO.misses,
        "memo_entries": len(_EXECUTABLE_MEMO),
        "memo_maxsize": _EXECUTABLE_MEMO.maxsize,
    }


def clear_cache() -> None:
    """Drop every cached executable and reset the hit/miss counters.

    Both tables are reset: the :data:`CACHE` compile cache and the
    :data:`_EXECUTABLE_MEMO` front table, in lockstep (see the module docstring:
    that is what makes ETL's "clear the cache after changing a process-env
    compile knob" advice hold for the memo too).
    """
    CACHE.clear()
    _EXECUTABLE_MEMO.reset()


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

# ``NType`` members as plain Python ints: the per-node decode loop (the hottest
# host-side loop of the whole engine) reads them several times per node, so the
# attribute lookups are hoisted here once (the values are the single source of
# truth in ``utils.py``; ``int()`` also keeps ``IntEnum`` style wrappers, should
# ``NType`` ever become one, out of the inner loop).
_NT_VAR = int(NType.VAR)
_NT_CONST = int(NType.CONST)
_NT_UFUNC = int(NType.UFUNC)
_NT_BFUNC = int(NType.BFUNC)
_NT_OUT_NODE = int(NType.OUT_NODE)
_NT_TYPE_MASK = int(NType.TYPE_MASK)


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
        # The composition ``(exp(x) - exp(-x)) / 2`` cannot keep the sign of
        # zero (``exp(-0.0) == exp(0.0) == 1`` gives ``+0.0``) and loses
        # relative accuracy for tiny ``|x|`` -- both are observable through the
        # next node (``loose_inv(-0.0) = -1e9``, ``loose_inv(+0.0) = +1e9``, and
        # ``(exp(x)-exp(-x))/2`` for |x| < 1e-4 differs from ``x`` by less than a
        # ulp of ``x``, so the correctly rounded float32 result is exactly ``x``,
        # signed zero included).  Keep ``x`` below that threshold, exactly like
        # ``_fold_unary`` does for a constant operand.
        w = _to_f64(a)
        composed = _round32(
            etl.multiply(
                etl.subtract(etl.exp(w), etl.exp(etl.negate(w))),
                _trace_const_f64(_HALF),
            )
        )
        return etl.select(
            etl.less(etl.abs(a), _trace_const(_SINH_LINEAR_BELOW)), a, composed
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


def _freeze_options(value: Any) -> Any:
    """Hashable, order-insensitive, immutable snapshot of an option value.

    Mirrors ``etl.build_cache._canonicalize_options`` exactly -- dicts become
    sorted ``(key, value)`` item tuples, lists/tuples sorted tuples of their
    frozen items, scalars pass through, and a ``TypeError`` from sorting
    (incomparable mixed types) keeps the original order -- so two option sets
    that ``build_cached`` keys identically also key identically here.  The
    result shares no mutable state with the caller's dicts/lists, so a memo key
    can never be changed from the outside.
    """
    if isinstance(value, dict):
        try:
            return tuple(
                sorted((key, _freeze_options(item)) for key, item in value.items())
            )
        except TypeError:  # incomparable mixed keys/values: keep the given order
            return tuple((key, _freeze_options(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        try:
            return tuple(sorted(_freeze_options(item) for item in value))
        except TypeError:
            return tuple(_freeze_options(item) for item in value)
    return value


def _executable_key(
    program: Tuple[int, Sequence[Sequence[int]]],
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Memo key of one executable: everything ``build_cached`` keys on.

    ``(id(_eval_step), program, n_rows, input_len, backend_name, device,
    frozen options)`` -- the same components ``build_cached`` derives its own
    key from (``id(fn)``, the traced spec tree -- here always
    ``TensorSpec((n_rows, input_len), float32)`` --, the resolved backend, the
    resolved device and the canonicalized resolved options), so the memo is as
    discriminating as :data:`CACHE` and can never serve an executable built for
    a different combination.  The key is value-typed and immutable: a re-decode
    of an unchanged tree yields an equal ``program`` tuple and therefore hits
    (the elite case), and the device is a frozen ``etl.core.Device`` (kind +
    index, the same content the ``build_cached`` codec encodes).

    Ambient ``ETL_BACKEND`` / ``ETL_DEVICE`` / ``ETL_TARGET_BACKENDS`` are not
    part of the key, deliberately: this module always passes backend, device and
    (for iree) ``target_backends`` itself, so they cannot change what it
    computes -- the numpy backend documents ``target_backends`` as
    accepted-and-ignored (``etl/backends/options.py``), which is the only one of
    the three that can still enter ``build_cached``'s resolved options.  A memo
    hit is therefore the same executable ``build_cached`` would return.
    """
    return (
        id(_eval_step),
        program,
        int(n_rows),
        int(input_len),
        backend_name,
        device,
        _freeze_options(options),
    )


def _executable(
    program: Tuple[int, Sequence[Sequence[int]]],
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Any:
    """Memoized per-(structure, shape) executable (a hit returns identity).

    One value-keyed lookup in :data:`_EXECUTABLE_MEMO` (:func:`_executable_key`)
    in front of ``etl.build_cached``; only a memo miss builds the
    ``TensorSpec`` and re-derives ``build_cached``'s own key -- still the more
    expensive path (~17-25 us for a fingerprint-replay hit against this memo's
    ~2.4-2.8 us).  ``_ExecutableMemo`` documents the contract and
    ``executable_memo_check.py`` pins it.
    """
    key = _executable_key(program, n_rows, input_len, backend_name, device, options)
    executable = _EXECUTABLE_MEMO.lookup(key)
    if executable is not None:
        return executable
    spec = etl.TensorSpec((int(n_rows), int(input_len)), etl.float32)
    if backend_name == "numpy":
        executable = build_cached(
            _eval_step, spec, program, backend=backend_name, device=device, cache=CACHE
        )
    else:
        executable = build_cached(
            _eval_step,
            spec,
            program,
            backend=backend_name,
            device=device,
            cache=CACHE,
            **options,
        )
    _EXECUTABLE_MEMO.store(key, executable)
    return executable


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
    node_sizes = [1] * size0
    repl: List[Optional[int]] = [None] * size0
    out_indices = [-1] * size0 if multi else None
    stack: List[Tuple[Any, Any, int, bool]] = []
    has_var = False

    for i in range(size0 - 1, -1, -1):
        raw = int(types[i])
        if multi:
            tagged = bool(raw & _NT_OUT_NODE)
            base = raw & _NT_TYPE_MASK
        elif raw & _NT_OUT_NODE:
            raise ValueError(
                f"row {row}: node {i}: node_type={raw} carries the OUT_NODE "
                "flag but output_len == 1 -- tagged nodes are only defined "
                "for multi-output trees (the old kernel corrupted its "
                "operand stack here; see the module docstring)"
            )
        else:
            tagged = False
            base = raw
        if base == _NT_VAR:
            index_value = values[i]
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
            has_var = True
            stack.append((_SV, _SV, 1, False))
            continue
        if base == _NT_CONST:
            value = np.float32(values[i])
            kinds[i] = _K_CONST
            funcs[i] = int(bits[i])  # float32 bit pattern of the leaf
            stack.append((value, value, 1, False))
            continue

        # -- function node ---------------------------------------------------
        if base == _NT_UFUNC:
            arity = 1
        elif base == _NT_BFUNC:
            arity = 2
        else:  # TFUNC and any unknown base type: the kernel's ternary branch
            arity = 3
        n_following = len(stack)
        if n_following < arity:
            base_name = _BASE_NAMES.get(base, f"base type {base}")
            raise ValueError(
                f"row {row}: node {i}: {base_name} node needs {arity} operand(s) "
                f"but only {n_following} node(s) follow it in the prefix encoding "
                f"-- truncated tree (subtree_size[0]={size0})"
            )
        # The decoded function id / packed out-node value is resolved *before*
        # the children are folded and before the own-tag is folded into
        # ``has_out``: the fold must use the same id the emitted token carries
        # (for a tagged node that is ``packed & 0xFFFF``, not the float32
        # reinterpretation of the packed bits, which is a denormal/NaN) and the
        # fold decision must see the node's own tag.  Both are the pre-refactor
        # decoder's semantics; the pops below cannot fail (arity was checked).
        if multi:
            if tagged:
                packed = int(bits[i])  # 4-byte view of node_value
                funcs[i] = packed & 0xFFFF
                out_indices[i] = packed >> 16  # C arithmetic shift
            elif arity == 3:
                funcs[i] = 0  # the ternary branch ignores the function id
            else:
                func_id = values[i]
                if not math.isfinite(func_id):
                    raise ValueError(
                        f"row {row}: node {i}: function id node_value={values[i]!r} "
                        "is not finite"
                    )
                funcs[i] = int(func_id)  # C truncation toward zero
        elif arity != 3:
            func_id = values[i]
            if not math.isfinite(func_id):
                raise ValueError(
                    f"row {row}: node {i}: function id node_value={values[i]!r} "
                    "is not finite"
                )
            funcs[i] = int(func_id)  # C truncation toward zero

        # pop order is leftmost first (the leftmost child is the last pushed)
        if arity == 1:
            kinds[i] = _K_UFUNC
            child0 = stack.pop()
            own = child0[0]
            if own is not _SV:
                own = _fold_unary(funcs[i], own)
            size = 1 + child0[2]
            passed_up = child0[1]
            has_out = child0[3]
        elif arity == 2:
            kinds[i] = _K_BFUNC
            child0 = stack.pop()
            child1 = stack.pop()
            a = child0[0]
            b = child1[0]
            if a is not _SV and b is not _SV:
                own = _fold_binary(funcs[i], a, b)
            else:
                own = _SV
            size = 1 + child0[2] + child1[2]
            passed_up = child1[1]
            has_out = child0[3] or child1[3]
        else:
            kinds[i] = _K_TFUNC
            child0 = stack.pop()
            child1 = stack.pop()
            child2 = stack.pop()
            a = child0[0]
            b = child1[0]
            c = child2[0]
            if a is not _SV and b is not _SV and c is not _SV:
                own = _fold_ternary(a, b, c)
            else:
                own = _SV
            size = 1 + child0[2] + child1[2] + child2[2]
            passed_up = child2[1]
            has_out = child0[3] or child1[3] or child2[3]

        declared = int(sizes[i])
        if declared != size:
            raise ValueError(
                f"row {row}: node {i}: subtree_size={declared} disagrees with the "
                f"tree structure (1 + children sizes = {size})"
            )
        node_sizes[i] = size
        if multi and 0 <= out_indices[i] < output_len:
            # The node's OWN effective tag makes its subtree unfoldable: folding
            # it would drop its ``outs[outIndex] += own`` accumulation.  An
            # out-of-range tag is not an effective out-node and still folds.
            has_out = True
        if own is _SV:
            replacement = None
        elif not multi:
            replacement = _float_to_bits(own)  # single output: own value is it
        elif has_out:
            replacement = None  # folding would drop an accumulation
        elif passed_up is _SV:
            replacement = None
        else:
            # multi output: the subtree only ever contributes its passed-up value
            replacement = _float_to_bits(passed_up)
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
    if multi:
        while i < size0:
            replacement = repl[i]
            if replacement is not None:
                tokens.append((_K_CONST, replacement))
                i += node_sizes[i]
                continue
            kind = kinds[i]
            if kind == _K_VAR:
                tokens.append((_K_VAR, funcs[i]))
            elif kind == _K_CONST:
                tokens.append((_K_CONST, funcs[i]))
            else:
                tokens.append((kind, funcs[i], out_indices[i]))
            i += 1
    else:
        while i < size0:
            replacement = repl[i]
            if replacement is not None:
                tokens.append((_K_CONST, replacement))
                i += node_sizes[i]
                continue
            kind = kinds[i]
            if kind == _K_VAR:
                tokens.append((_K_VAR, funcs[i]))
            elif kind == _K_CONST:
                tokens.append((_K_CONST, funcs[i]))
            else:
                tokens.append((kind, funcs[i], -1))
            i += 1

    collapsed = size0 - len(tokens)
    if has_var:
        return DecodedTree(
            program=(output_len, tuple(tokens)),
            const_out=None,
            node_count=size0,
            collapsed_nodes=collapsed,
        )
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
    xin_key: int,
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
    prepared: Dict[int, Any],
) -> np.ndarray:
    """Build-or-hit one tree and run it; returns ``(n_rows, out_len)`` float32.

    ``xin_key`` identifies the host input ``xin`` and is the memo key of
    ``prepared`` (host->device copies, one per distinct input): every tree of one
    ``evaluate`` call that sees the SAME rows shares a single upload, while
    ``per_row=True`` keys by tree index because tree ``i`` must see row ``i``.
    """
    exe = _executable(tree.program, n_rows, input_len, backend_name, device, options)
    if backend_name == "numpy":
        with _quiet():
            result = etl.run(exe, xin, tree.program)
        return np.asarray(result.numpy(), dtype=np.float32).reshape(n_rows, -1)
    device_input = prepared.get(xin_key)
    if device_input is None:  # one host->device copy per distinct input block
        prepared[xin_key] = device_input = etl.from_numpy(xin).to(device)
    with _quiet():
        result = etl.run(exe, device_input, tree.program)
    return np.asarray(result.to(_CPU).numpy(), dtype=np.float32).reshape(n_rows, -1)


def _evaluate_rows(
    pop: Population,
    xt: np.ndarray,
    *,
    per_row: bool,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> np.ndarray:
    """Host-rows core of :func:`evaluate` and :func:`sr_fitness`.

    ``pop`` must already be a validated :class:`Population` and ``xt`` the
    ALREADY converted ``(rows, input_len)`` float32 host array (``rows ==
    pop_size`` when ``per_row``); nothing is re-validated or re-converted here.
    The return value is the assembled **host** prediction -- ``(pop_size,
    output_len)`` for ``per_row=True``, ``(pop_size, N, output_len)`` otherwise
    -- a FRESH array that the caller may consume in place (``sr_fitness``
    reduces it without ever moving it to a device).

    On a device backend (``iree``) the host->device copy of the input block is
    memoized per distinct block: ``per_row=False`` converts ``xt`` ONCE for the
    whole call, while ``per_row=True`` transfers each tree's own
    ``(1, input_len)`` row.
    """
    n_rows = 1 if per_row else int(xt.shape[0])
    backend_name, device, options = _resolve_backend(backend, iree_device_index)

    rows = np.empty((pop.pop_size, n_rows, pop.output_len), dtype=np.float32)
    prepared: Dict[int, Any] = {}
    for index, tree in enumerate(pop.trees):
        if tree.const_out is not None:
            rows[index, :, :] = _bits_to_float_arr(tree.const_out)
            continue
        xin = xt[index : index + 1] if per_row else xt
        rows[index] = _run_tree(
            tree,
            xin,
            0 if not per_row else index,  # tree index -> per-tree x row
            n_rows,
            pop.input_len,
            backend_name,
            device,
            options,
            prepared,
        )
    if per_row:
        rows = np.ascontiguousarray(rows[:, 0, :])
    return rows


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

    On a device backend (``iree``) the host->device copy is memoized per distinct
    input block: ``per_row=False`` converts ``x`` ONCE for the whole call, while
    ``per_row=True`` transfers each tree's own ``(1, input_len)`` row.

    The result is a FRESH, contiguous, ``requires_grad=False`` float32 tensor on
    ``x``'s device (``out_device`` overrides it, e.g. ``"cuda"`` /
    ``torch.device("cuda:1")``).  ``x`` may be a torch tensor on any device or a
    numpy array; it is cast to float32, copied, and never modified.  The
    population's tensors are not touched either.

    The evaluation itself is :func:`_evaluate_rows`; this wrapper only validates
    the input, converts it once and places the assembled rows on a device.
    """
    pop = _require_population(population)
    xt = _input_to_numpy(x, pop.input_len, "x")
    if per_row and xt.shape[0] != pop.pop_size:
        raise ValueError(
            f"per_row=True needs x.shape[0] == {pop.pop_size} (pop_size), "
            f"got {xt.shape[0]}"
        )
    rows = _evaluate_rows(
        pop,
        xt,
        per_row=per_row,
        backend=backend,
        iree_device_index=iree_device_index,
    )

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

    ``inputs`` and ``labels`` are converted to host ONCE (no second conversion
    inside the evaluation), the errors are reduced on the HOST from the rows
    :func:`_evaluate_rows` just assembled -- the full ``(pop_size, N,
    output_len)`` prediction never makes a device round-trip, on any backend --
    and only the ``(pop_size,)`` loss is placed on ``inputs``' device.

    The formula is unchanged; two things about the reduction are deliberate:

    * it runs **host-side, in place on the rows** the evaluation just assembled
      (allocating no extra ``(pop_size, N, output_len)`` temporaries), and
    * it uses **numpy, not torch**: torch's CPU ops on a ``(pop_size, N)``-sized
      tensor open an intra-op thread pool whose spin-waiting then starves the
      single-threaded ETL/numpy interpreter of the *next* call (measured on the
      64-tree / N=10000 fixture: 95.3 ms median per ``SR_fitness`` with the torch
      host reduction vs 20.5 ms with the numpy one, against 22.8 ms for the old
      device-reduction path -- and the torch path's stalls land in unrelated
      stages, e.g. ``build_cached`` hits).

    Both are the same class of float32 rounding deviation as the reduction order
    noted in the module docstring (max relative difference vs the device
    reduction ~2e-7 on the fixtures, NaN/inf placement identical).
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
    rows = _evaluate_rows(
        pop,
        xin,
        per_row=False,
        backend=backend,
        iree_device_index=iree_device_index,
    )
    # err = label - prediction (then squared / |.|), summed over data points and
    # outputs in float32.  In place on ``rows``: this call owns that fresh array.
    np.subtract(label[None, :, :], rows, out=rows)  # (pop_size, N, output_len)
    if use_MSE:
        np.multiply(rows, rows, out=rows)
    else:
        np.abs(rows, out=rows)
    loss_np = rows.sum(axis=(1, 2), dtype=np.float32) / np.float32(xin.shape[0])
    loss = torch.from_numpy(loss_np).contiguous().detach().requires_grad_(False)
    dest = out_device if out_device is not None else getattr(inputs, "device", None)
    if dest is not None and torch.device(dest) != loss.device:
        loss = loss.to(dest)
    return loss


# ---------------------------------------------------------------------------
# benchmark-support runner: build once, run many (no readback, no re-key)
# ---------------------------------------------------------------------------
#
# ``build_specialized_runner`` / ``SpecializedRunner`` are the specialized half
# of the two frozen benchmark-support runner APIs; the dense half is
# ``dense_engine.build_dense_runner``.  The shared contract (exact signatures,
# what the build phase costs, what ``.run()`` does per backend and what it
# returns) is documented in ``RUNNER_HOOKS.md`` (this directory) and pinned by
# ``runner_check.py``; the short version:
#
# * the BUILD phase does everything that is not "the compiled forward pass" --
#   host decode, per-tree executable resolution (compile or cache hit), input
#   validation + conversion + staging on its final device;
# * ``.run()`` executes ONLY the forward pass: it re-uses the staged inputs and
#   the already-resolved executables, never re-decodes/re-plans/re-keys and
#   never moves a result back to the host.  N calls are N identical passes;
# * nothing in the library calls them, they do not change the public route
#   behaviour (``EVOGP_EVAL_MODE`` is NOT consulted: the caller picks the route
#   by picking the builder) and they are not part of the training path.

#: The attributes a ``Forest``-like container must expose for the runners.
#: Duck typing on purpose: ``forest.py`` imports THIS module, so a module-level
#: import back would create an import cycle.
_POPULATION_ATTRS = (
    "batch_node_value",
    "batch_node_type",
    "batch_subtree_size",
    "input_len",
    "output_len",
)


def _population_inputs(population_or_forest: Any) -> Tuple[Any, Any, Any, int, int]:
    """Duck-typed ``(values, types, sizes, input_len, output_len)`` of a population.

    Accepts any object carrying the five attributes of ``evogp.tree.Forest``
    (``batch_node_value`` / ``batch_node_type`` / ``batch_subtree_size`` /
    ``input_len`` / ``output_len``); the runners never import the container.
    Tensors may live on any device and may be numpy arrays, exactly as
    :func:`decode_population` accepts them.

    Raises:
        TypeError: ``population_or_forest`` is missing one of the attributes
            (the message names them and the offending type).
    """
    missing = [
        name for name in _POPULATION_ATTRS if not hasattr(population_or_forest, name)
    ]
    if missing:
        raise TypeError(
            "expected a Forest-like container exposing "
            f"{list(_POPULATION_ATTRS)}, got "
            f"{type(population_or_forest).__name__} (missing {missing})"
        )
    return (
        population_or_forest.batch_node_value,
        population_or_forest.batch_node_type,
        population_or_forest.batch_subtree_size,
        int(population_or_forest.input_len),
        int(population_or_forest.output_len),
    )


class SpecializedRunner:
    """A decoded population with every per-tree executable already resolved.

    The constructor IS the build phase (both :func:`build_specialized_runner`
    and a direct ``SpecializedRunner(...)`` build it the same way):

    1. duck-type the population (:func:`_population_inputs`) and validate/convert
       ``inputs`` once (``(N, input_len)`` float32, ``N == pop_size`` when
       ``per_row``) -- the same semantics as :func:`evaluate`;
    2. :func:`decode_population` the population (host decode + validation);
    3. resolve one executable per NON-constant tree through :func:`_executable`
       (value-keyed memo in front of ``build_cached``), so a structure already
       compiled for this shape is a pure lookup;
    4. stage the input block(s) for their final device: ONE shared ``(N,
       input_len)`` upload for ``per_row=False``, one uploaded ``(1, input_len)``
       row per non-constant tree for ``per_row=True`` (exactly what
       :func:`_evaluate_rows` does within a call, but once instead of per call).

    Read-only attributes (diagnostics for harnesses):
        population: the decoded :class:`Population`.
        executables: ``pop_size``-long tuple; ``None`` for constant trees.
        payloads: ``pop_size``-long tuple of the staged per-tree inputs
            (host numpy arrays on the numpy backend, device tensors otherwise);
            ``None`` for constant trees.
        constant_rows: ``pop_size``-long tuple; for a constant tree a READ-ONLY
            ``(n_rows, output_len)`` float32 broadcast view of the rows every
            data point sees, else ``None``.
        backend_name / device / options: the resolved
            :func:`_resolve_backend` triple (frozen for every ``.run()``).
        per_row, n_rows, input_len, output_len, pop_size.

    Callers that need the values on the host read the raw results back
    themselves (``raw.to(cpu).numpy()`` / ``raw.numpy()``) -- which is the
    consumer's copy, not a runner step; see :meth:`run`.
    """

    def __init__(
        self,
        population_or_forest: Any,
        inputs: Any,
        *,
        per_row: bool = False,
        backend: Optional[str] = None,
        iree_device_index: Optional[int] = None,
    ) -> None:
        values, types, sizes, input_len, output_len = _population_inputs(
            population_or_forest
        )
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.per_row = bool(per_row)
        self.population = decode_population(
            values, types, sizes, self.input_len, self.output_len
        )
        self.pop_size = self.population.pop_size
        xin = _input_to_numpy(inputs, self.input_len, "inputs")
        if self.per_row and xin.shape[0] != self.pop_size:
            raise ValueError(
                f"per_row=True needs inputs.shape[0] == {self.pop_size} "
                f"(pop_size), got {xin.shape[0]}"
            )
        self.n_rows = 1 if self.per_row else int(xin.shape[0])
        self.backend_name, self.device, self.options = _resolve_backend(
            backend, iree_device_index
        )
        to_device = self.backend_name != "numpy"
        shared = (
            etl.from_numpy(np.ascontiguousarray(xin)).to(self.device)
            if to_device and not self.per_row
            else None
        )
        executables: List[Optional[Any]] = []
        payloads: List[Optional[Any]] = []
        constant_rows: List[Optional[np.ndarray]] = []
        for index, tree in enumerate(self.population.trees):
            if tree.const_out is not None:
                # Constant tree: folded at decode time, nothing was compiled.
                executables.append(None)
                payloads.append(None)
                constant_rows.append(
                    np.broadcast_to(
                        _bits_to_float_arr(tree.const_out),
                        (self.n_rows, self.output_len),
                    )
                )
                continue
            executables.append(
                _executable(
                    tree.program,
                    self.n_rows,
                    self.input_len,
                    self.backend_name,
                    self.device,
                    self.options,
                )
            )
            if self.per_row:
                row = xin[index : index + 1]
                payloads.append(
                    etl.from_numpy(np.ascontiguousarray(row)).to(self.device)
                    if to_device
                    else row
                )
            else:
                payloads.append(shared if to_device else xin)
            constant_rows.append(None)
        self.executables: Tuple[Optional[Any], ...] = tuple(executables)
        self.payloads: Tuple[Optional[Any], ...] = tuple(payloads)
        self.constant_rows: Tuple[Optional[np.ndarray], ...] = tuple(constant_rows)

    @property
    def invocations(self) -> int:
        """How many ``etl.run`` calls one :meth:`run` issues (non-constant trees)."""
        return sum(1 for executable in self.executables if executable is not None)

    def run(self) -> Tuple[Optional[Any], ...]:
        """Execute ONE forward pass -- the only thing a harness times.

        Returns:
            A ``pop_size``-long tuple: entry ``i`` is the RAW ``etl.run`` result
            of tree ``i`` -- an ``etl`` runtime tensor of shape ``(n_rows,
            output_len)``, host-backed on the numpy backend and device-resident
            on a device backend -- or ``None`` for a constant tree (never
            compiled; its rows are staged in :attr:`constant_rows`).

        Nothing else happens: no readback, no torch conversion, no host-side
        assembly, no executable lookup, no decode.  ``.run()`` is repeatable --
        the executables and the staged inputs are re-used as-is, and the static
        program is re-passed by value exactly as the frozen engine does.
        """
        results: List[Optional[Any]] = []
        with _quiet():
            for executable, payload, tree in zip(
                self.executables, self.payloads, self.population.trees
            ):
                if executable is None:  # constant tree: nothing to run
                    results.append(None)
                    continue
                # ADOPTION POINT (ETL-side facility landed, swap NOT made
                # here): ``etl.run`` re-passes the statics by value and
                # re-validates them on every call -- now through the
                # per-executable content-fingerprint fast path (ETL bench,
                # numpy: a 60-token program 303 -> ~67-71 us per call) -- and
                # it walks the executable's structure.  ``etl.jit`` ->
                # ``JitFunction`` / ``BoundJit`` now carries the bound
                # invocation that skips the consumer-side lookup and key
                # derivation; nothing in this repository emulates it and
                # /mnt/local-ssd/bchuang/etl is not modified.  This is the one
                # line to swap.
                results.append(etl.run(executable, payload, tree.program))
        return tuple(results)


def build_specialized_runner(
    population_or_forest: Any,
    inputs: Any,
    *,
    per_row: bool = False,
    backend: Optional[str] = None,
    iree_device_index: Optional[int] = None,
) -> SpecializedRunner:
    """Build a repeatable specialized-engine runner (the frozen benchmark API).

    Args:
        population_or_forest: any ``Forest``-like container (duck-typed on
            ``batch_node_value`` / ``batch_node_type`` / ``batch_subtree_size``
            / ``input_len`` / ``output_len``; no import of the container).
        inputs: ``x`` with the semantics of :func:`evaluate` -- ``(N,
            input_len)``, converted to float32 host once; ``N == pop_size`` when
            ``per_row=True`` (tree ``i`` sees row ``i``), else the data points
            every tree is evaluated on.
        per_row: per-tree inputs (default ``False``: every tree sees all rows).
        backend / iree_device_index: exactly :func:`evaluate`'s overrides of the
            ``EVOGP_ETL_BACKEND`` / ``EVOGP_ETL_IREE_DEVICE`` environment.

    Returns:
        A :class:`SpecializedRunner`; ``.run()`` is the compiled forward pass
        only (see its docstring for the exact return contract).

    Raises:
        TypeError: the container is not Forest-like.
        ValueError: bad ``inputs`` shape, a ``per_row`` row-count mismatch, or a
            malformed tree (from :func:`decode_population`, naming row + node).
        ValueError: an unsupported backend name (from :func:`_resolve_backend`).
    """
    return SpecializedRunner(
        population_or_forest,
        inputs,
        per_row=per_row,
        backend=backend,
        iree_device_index=iree_device_index,
    )
