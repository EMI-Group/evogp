"""perf_probe.py — evoxc build/run economics probe for EvoGP's rewritten engine.

NOT a pytest module: a standalone script (run it directly) that measures what a
per-tree "true jit" evaluation loop costs with the REAL ``evoxc`` package, so the
rewritten engine's backend defaults and the ``sr_test`` scale can be chosen with
evidence instead of guesswork.

Engine shape this probe models (implemented in parallel under ``../tree/``):
the population tensors are decoded host-side into a *static* Python structure per
tree, that structure specializes a module-level ``@evoxc.defn`` step function through
``evoxc.build_cached`` (shared ``evoxc.CompileCache``: one cache entry per distinct
structure/backend/device/spec), and the resulting ``Executable`` is run per tree
over the ``(N, input_len)`` data matrix -- exactly the meta-programming pattern
pinned by ``<evoxc-checkout>/tests/test_meta_programming.py`` (the evoxc source
tree next to this project; it is the same meta-programming pattern).

What is measured (all wall-clock, per tree, ms):

1. cold build -- ``build_cached`` cache MISS = ``trace -> lower -> [compile ->] load``
   (``first_build_ms`` is the very first build of a backend/spec and includes the
   iree runtime/device init; the rest are steady-state per-tree builds).
2. warm run -- ``evoxc.run`` on the cached executable (for iree: device-only run and
   run + host readback), i.e. the inner-loop cost that every tree pays every
   generation.
3. cache-hit call overhead -- repeating ``build_cached`` on an already-built
   structure (key computation + LRU lookup, no pipeline work).
4. stage breakdown -- trace / lower / compile / load split for one tree.
5. extrapolated per-generation wall time for pop=200 and pop=1000 under an
   all-distinct (worst case) and a 50%-distinct (mixed) assumption.
6. ``EVOXC_DISABLE_LOCATIONS=1`` effect on trace/build (one subprocess per setting,
   so no import-time or interpreter state is shared between the two runs).
7. bounded parallel-build attempts (process pool + thread pool) -- build
   throughput only: ``Executable`` objects cannot cross process boundaries.

Backends: ``"numpy"`` (cpu reference backend) and ``"iree"`` on
``Device(HOST_GPU_KIND, 0)`` with ``opt_level="O1"`` -- ``HOST_GPU_KIND`` is the
device kind evoxc needs on this host (``"rocm"`` on a ROCm/HIP torch build,
else ``"cuda"``; see the comment at its definition), and ``target_backends`` is
NOT passed because ``build_cached`` infers it from that device kind (O2/O3 are
buggy on the GPU target -- never used here). Data: ``x`` of shape
``(N, input_len)`` float32 for N=100 and N=1000 (one tree over N data points, as
EvoGP evaluates it); ``--full`` adds N=10000 as an extension row.

Usage
-----
    python src/evogp/tests/perf_probe.py                 # --quick (default)
    python src/evogp/tests/perf_probe.py --full          # tighter statistics
    python src/evogp/tests/perf_probe.py --backends numpy
    python src/evogp/tests/perf_probe.py --json-out /path/to/report.json

A copy of the JSON report is always written under $TMPDIR (path is printed).
The script needs no GPU when ``--backends numpy`` is used; iree sections degrade
gracefully (recorded as an error string) when no GPU device/iree runtime is
present.

RESULTS
-------
The numbers below predate this probe's port to the renamed package: they were
taken on a CUDA box with the same library code, so they still describe the
economics; the live tables print this host's own device kind.
Measured with ``--full`` on 1x NVIDIA RTX A6000 (device 0) + 128 CPUs, Linux
6.1, python 3.11.2, evoxc 0.1.0, torch 2.6.0+cu124. 24 GP trees, 30-60 GP nodes
each (21-45 evoxc ops), 2 variables, data
``x`` of shape ``(N, 2)`` float32. Medians; the box is shared, so ``--full``
ranges are shown in brackets. ``--quick`` reproduces the same picture in ~36 s
(``--full``: ~2m20s).

    per-tree stage                 numpy (cpu)          iree (cuda:0, O1)
    ---------------------------------------------------------------------------
    process warm-up (3 builds)         0.28 s              2.0-2.8 s
    first build (init incl.)          ~100 ms              520-1760 ms
    cold build (steady state)         42 ms [24-105]       438-508 ms [372-2235]
    cache-hit build_cached             0.42 ms             0.44-0.45 ms
    warm run, N=100                    0.45 ms             1.03 ms
    warm run, N=1000                   0.48 ms             1.03 ms
    warm run, N=10000                  0.88 ms             1.03 ms
    warm run + host readback           0.45-0.88 ms        1.08-1.10 ms

    stage breakdown, N=1000 (one 43-op tree, median of 5; locations ON):
      numpy  trace 93.4  lower 3.3  compile 2.9  load  2.9 ms  -> 102.4 ms total
      iree   trace 132.5 lower 3.5  compile 302.4 load 45.8 ms -> 484.1 ms total

    extrapolated per-generation wall time (s), model =
      pop warm runs + (#distinct structures) cold builds:
                                        numpy              iree
      ------------------------------------------------------------------------
      pop=200,  N=100    8.5 / 4.3 (mixed)        87.7 / 44.0 (mixed)
      pop=1000, N=100   42.6 / 21.5 (mixed)       438.6 / 219.8 (mixed)
      pop=1000, N=1000  42.3 / 21.4 (mixed)       439.0 / 220.0 (mixed)
      pop=1000, N=10000 42.6 / 21.8 (mixed)       508.7 / 254.9 (mixed)
      all-cached floor (no builds at all): 0.45-0.88 s numpy, ~1.03 s iree
      build throughput: 23.7-23.9 (numpy) vs 1.97-2.29 (iree) distinct trees/s

    EVOXC_DISABLE_LOCATIONS=1 (subprocess per setting, N=1000):
      numpy trace       24.51 -> 1.67 ms  (14.7x,  delta -22.84 ms)
      numpy cold build  37.79 -> 9.88 ms  (3.8x,   delta -27.91 ms)
      numpy hit build    0.506 -> 0.500 ms (unchanged - no tracing on a hit)
      iree  cold build 1447.5 -> 541.9 ms (2.7x,  delta -905.6 ms)
    Parallel build attempt (iree, throughput only, 8 workers x 4 trees):
      processes 32 builds in 5.20 s = 6.16 builds/s = 2.70x serial
      threads   16 builds in 4.32 s = 3.70 builds/s = 1.62x serial
      (with 4 workers the process pool gave 1.5x and threads 0.5-1.0x)

RECOMMENDATION
--------------
Default the engine's per-tree evaluation loop to the ``"numpy"`` backend. On
every cost the GP loop pays -- cold build (42 ms vs 438-508 ms), cache-hit call
(0.42 vs 0.45 ms), and warm run of a 20-45-op graph (0.45-0.49 vs 1.03 ms at the
EvoGP data sizes) -- numpy is equal or faster, it needs no GPU/iree runtime, and
it keeps ``sr_test``, the test suite and CI runnable on a CPU-only machine. The
per-generation extrapolation is decisive: at pop=1000/N=1000 the mixed case is
21.4 s (numpy) vs 220 s (iree), and the all-distinct worst case 42 s vs 439 s;
iree only wins the all-cached floor case (1.02 s vs 0.48 s), which no realistic
generation reaches before the population converges.

Keep ``"iree"`` + ``Device(HOST_GPU_KIND, 0)`` + ``opt_level="O1"`` (the target
backend is inferred from the device kind) as an explicit opt-in path, i.e. the
engine must treat the evoxc backend and device as configuration, never as a
hard-coded constant. Two
supporting levers, both measured here: (a) set ``EVOXC_DISABLE_LOCATIONS=1`` for
the build phase -- tracing drops 14.7x (24.5 -> 1.7 ms) and the numpy cold build
3.8x, since location capture walks the Python stack once per op; (b) parallel
builds are NOT a workaround for the iree build cost (2.7x at 8 processes, and
in-process threads barely help), so the engine should never depend on them.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import torch

import evoxc

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: One cache shared by every probe section, as the engine will do. Entries are
#: keyed on (fn id, specs incl. static tree content, backend, device, options),
#: so the two backends / every N / every tree get their own entry.
SHARED_CACHE = evoxc.CompileCache()

CPU_DEVICE = evoxc.core.Device("cpu", 0)

#: The evoxc device KIND of this host's GPU. evoxc's IREE adapter loads the HAL
#: driver matching the kind, so it cannot be hard-coded: it follows the installed
#: torch build (``torch.version.hip`` is set on a ROCm/HIP build, ``None`` on a
#: CUDA build) -- the same resolution ``evogp.tree.evoxc_engine._gpu_device_kind``
#: uses. ``"cuda"`` on a CUDA box, ``"rocm"`` on a ROCm one (e.g. MI300X/gfx942).
HOST_GPU_KIND = "rocm" if getattr(torch.version, "hip", None) else "cuda"

#: iree compile knobs. ``opt_level="O1"`` is deliberate: O2/O3 are broken on the
#: GPU target. ``target_backends`` is deliberately NOT passed -- ``build_cached``
#: infers it from the resolved device kind (``"cuda"`` -> ``["cuda"]``,
#: ``"rocm"`` -> ``["rocm"]``), so the options carry only ``opt_level``.
#: ``--quick`` and ``--full`` only differ in iteration counts.
IREE_OPTIONS = {"opt_level": "O1"}

NP_SPEC_OPTS: dict = {}  # numpy backend takes no extra options

QUICK = dict(
    trees=8,
    reps=50,
    hit_reps=50,
    parallel_workers=4,
    parallel_trees_per_worker=2,
    thread_workers=4,
    thread_trees_per_worker=1,
    parallel_budget=120.0,
)
FULL = dict(
    trees=24,
    reps=200,
    hit_reps=200,
    parallel_workers=8,
    parallel_trees_per_worker=4,
    thread_workers=8,
    thread_trees_per_worker=2,
    parallel_budget=600.0,
)
BREAKDOWN_REPS = 5  # per-stage medians (the box is shared: single shots are noisy)
WARMUP_BUILDS = 3   # throwaway builds per backend before any measurement: the
                    # first iree builds of a process are 2-3x slower (compiler
                    # runtime / HAL warm-up) and would poison the first section
# --------------------------------------------------------------------------
# Tree encoding and trace-time composition
# --------------------------------------------------------------------------
# A decoded tree is a nested tuple of ints/floats ONLY (the build_cached cache
# key must be codec-encodable -- no Enums/dataclasses/sets/strings):
#
#   (0, i)        leaf: variable column i            -> 1 evoxc op
#   (1, c)        leaf: float constant c             -> 1 evoxc op
#   (2, l, r)     add                                -> 1 evoxc op
#   (3, l, r)     subtract
#   (4, l, r)     multiply
#   (5, l, r)     divide
#   (6, a)        sin                                -> 1 evoxc op
#   (7, a)        exp
#   (8, a)        tanh
#   (9, l, r)     comparison GT  -> select(l > r, 1, -1)   -> 2 evoxc ops
#   (10, l, r)    comparison LT  -> select(l < r, 1, -1)
#   (11, c, a, b) IF (ternary)   -> select(c > 0, a, b)    -> 2 evoxc ops
#   (12, l, r)    maximum
#   (13, l, r)    minimum
#
# That mirrors EvoGP's real function set (ADD/SUB/MUL/DIV/MAX/MIN/LT/GT/IF +
# SIN/EXP/TANH; see ``../tree/utils.py`` ``Func``/``FUNCS_NAMES``).
# DIV is the unprotected ``evoxc.divide`` (the LOOSE_DIV guard only changes values,
# never op counts/timing), so a few trees can produce inf/nan values -- harmless
# for timing, and excluded from the parity diff via ``nan_to_num``.
LEAF_VAR = 0
LEAF_CONST = 1
IF = 11

_BINARY = {
    2: evoxc.add,
    3: evoxc.subtract,
    4: evoxc.multiply,
    5: evoxc.divide,
    12: evoxc.maximum,
    13: evoxc.minimum,
}
_UNARY = {6: evoxc.sin, 7: evoxc.exp, 8: evoxc.tanh}
_CMP = {9: evoxc.greater, 10: evoxc.less}

# Python-side folds for all-static subtrees: evoxc ops need at least one
# SymbolicTensor operand, so a subtree that is entirely constants must fold to a
# Python scalar at trace time (exactly as the evoxc meta-programming test does).
_FOLD_BINARY = {
    2: lambda a, b: a + b,
    3: lambda a, b: a - b,
    4: lambda a, b: a * b,
    5: lambda a, b: a / b,
    12: max,
    13: min,
}
_FOLD_UNARY = {6: math.sin, 7: math.exp, 8: math.tanh}
_FOLD_CMP = {9: lambda a, b: 1.0 if a > b else -1.0,
             10: lambda a, b: 1.0 if a < b else -1.0}


def _is_static(v):
    """True when ``v`` is a Python scalar (fully folded, not a symbolic tensor)."""
    return isinstance(v, (int, float))


def _compose(tree, x):
    """Decode a static tree into an evoxc expression over ``x`` (trace time).

    Runs as plain Python inside the trace: leaf tags are tested before any
    unpacking, so leaves (2-tuples) and nodes (n-ary tuples) never collide, and
    all-static subtrees fold to Python scalars (ops require a symbolic operand).
    Python scalars stay scalar operands -- evoxc promotes them (weak scalar).
    """
    tag = tree[0]
    if tag == LEAF_VAR:
        return x[:, tree[1]]
    if tag == LEAF_CONST:
        return tree[1]
    if tag in _UNARY:
        a = _compose(tree[1], x)
        return _FOLD_UNARY[tag](a) if _is_static(a) else _UNARY[tag](a)
    if tag in _CMP:
        a, b = _compose(tree[1], x), _compose(tree[2], x)
        if _is_static(a) and _is_static(b):
            return _FOLD_CMP[tag](a, b)
        return evoxc.select(_CMP[tag](a, b), 1.0, -1.0)
    if tag == IF:
        c = _compose(tree[1], x)
        if _is_static(c):
            return _compose(tree[2] if c > 0 else tree[3], x)
        return evoxc.select(evoxc.greater(c, 0.0), _compose(tree[2], x),
                            _compose(tree[3], x))
    a, b = _compose(tree[1], x), _compose(tree[2], x)
    if _is_static(a) and _is_static(b):
        return _FOLD_BINARY[tag](a, b)
    return _BINARY[tag](a, b)


def _contains_var(tree):
    """True when the tree references at least one variable leaf.

    Enforced at generation time: a tree of pure constants folds to a Python
    scalar at trace time and would not be a valid program (the step function
    must return a tensor).
    """
    tag = tree[0]
    if tag == LEAF_VAR:
        return True
    if tag == LEAF_CONST:
        return False
    if tag in _UNARY:
        return _contains_var(tree[1])
    if tag in _CMP:
        return _contains_var(tree[1]) or _contains_var(tree[2])
    if tag == IF:
        return any(_contains_var(t) for t in tree[1:4])
    return _contains_var(tree[1]) or _contains_var(tree[2])


@evoxc.defn
def _step(x, tree):
    """The engine's step function: static tree program applied to ``x``.

    ``tree`` is a STATIC argument: it is decoded at trace time and specializes
    the graph; the returned executable is keyed on its content. Module-level and
    stable -- a closure here would make every ``build_cached`` call a miss.
    """
    return _compose(tree, x)


def _gp_nodes(tree):
    """Number of GP nodes in the tree."""
    tag = tree[0]
    if tag in (LEAF_VAR, LEAF_CONST):
        return 1
    if tag in _UNARY:
        return 1 + _gp_nodes(tree[1])
    if tag in _CMP:
        return 1 + _gp_nodes(tree[1]) + _gp_nodes(tree[2])
    if tag == IF:
        return 1 + _gp_nodes(tree[1]) + _gp_nodes(tree[2]) + _gp_nodes(tree[3])
    return 1 + _gp_nodes(tree[1]) + _gp_nodes(tree[2])


def _evoxc_ops(tree):
    """Number of evoxc ops the tree traces into (mirrors ``_compose``: all-static
    subtrees fold to Python scalars and emit no op)."""
    return _ops_and_static(tree)[0]


def _ops_and_static(tree):
    """``(evoxc_op_count, is_static)`` for a static tree (see ``_compose``)."""
    tag = tree[0]
    if tag == LEAF_VAR:
        return 1, False  # x[:, i] -- a slice op
    if tag == LEAF_CONST:
        return 0, True  # folded into the parent as a scalar operand
    if tag in _UNARY:
        child_ops, static = _ops_and_static(tree[1])
        return (0, True) if static else (child_ops + 1, False)
    if tag in _CMP:
        a_ops, a_static = _ops_and_static(tree[1])
        b_ops, b_static = _ops_and_static(tree[2])
        if a_static and b_static:
            return 0, True
        return a_ops + b_ops + 2, False  # compare + select
    if tag == IF:
        c_ops, c_static = _ops_and_static(tree[1])
        a_ops, a_static = _ops_and_static(tree[2])
        b_ops, b_static = _ops_and_static(tree[3])
        if c_static:
            return a_ops + b_ops, (a_static and b_static)
        return c_ops + a_ops + b_ops + 2, False  # greater + select
    a_ops, a_static = _ops_and_static(tree[1])
    b_ops, b_static = _ops_and_static(tree[2])
    if a_static and b_static:
        return 0, True
    return a_ops + b_ops + 1, False


# --------------------------------------------------------------------------
# Deterministic GP-shaped tree generation
# --------------------------------------------------------------------------


def _gen_random_tree(rng, depth=0):
    """Grow a random tree (depth-capped) of GP nodes.

    Node mix approximates EvoGP's default function set: ~20% unary
    (sin/exp/tanh), ~20% comparison, ~10% ternary IF, rest binary arithmetic
    (add/sub/mul/div/max/min); leaves are variables (60%) or small constants.
    """
    if depth >= 5 or rng.random() < 0.35:
        if rng.random() < 0.6:
            return (LEAF_VAR, int(rng.integers(0, 2)))
        return (LEAF_CONST, float(rng.uniform(-1.0, 1.0)))
    r = rng.random()
    if r < 0.20:
        return (int(rng.choice([6, 7, 8])), _gen_random_tree(rng, depth + 1))
    if r < 0.40:
        return (
            int(rng.choice([9, 10])),
            _gen_random_tree(rng, depth + 1),
            _gen_random_tree(rng, depth + 1),
        )
    if r < 0.50:
        return (
            IF,
            _gen_random_tree(rng, depth + 1),
            _gen_random_tree(rng, depth + 1),
            _gen_random_tree(rng, depth + 1),
        )
    return (
        int(rng.choice([2, 3, 4, 5, 12, 13])),
        _gen_random_tree(rng, depth + 1),
        _gen_random_tree(rng, depth + 1),
    )


def _gen_chain(rng, n_ops, left_deep=True):
    """A maximally skewed chain: ((((v0 op v1) op v0) op v1) ...) — the deep
    case that stresses both the interpreter and the iree codegen."""
    var = lambda: (LEAF_VAR, int(rng.integers(0, 2)))  # noqa: E731
    node = rng.choice([2, 3, 4, 5, 12, 13])
    node = int(node)
    cur = (node, var(), var())
    for _ in range(n_ops):
        node = int(rng.choice([2, 3, 4, 5, 12, 13]))
        cur = (node, cur, var()) if left_deep else (node, var(), cur)
    return cur


def build_tree_set(count, seed):
    """A deterministic set of trees: 1 left-deep chain, 1 right-deep chain and
    ``count - 2`` random trees with 30-60 GP nodes."""
    rng = np.random.default_rng(seed)
    trees = [_gen_chain(rng, 20, left_deep=True), _gen_chain(rng, 20, left_deep=False)]
    while len(trees) < count:
        # Retry until the random tree lands in the 30-60 GP-node band (the
        # generator is depth-capped, so this terminates in a couple of tries)
        # AND references at least one variable (a pure-constant tree would fold
        # to a Python scalar at trace time and is not a valid program).
        for _ in range(500):
            t = _gen_random_tree(rng)
            if 30 <= _gp_nodes(t) <= 60 and _contains_var(t):
                trees.append(t)
                break
        else:
            raise RuntimeError("could not sample a valid 30-60 GP-node tree")
    return trees[: max(count, 1)]


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------


def _stats(samples):
    """Summarize a list of ms samples into plain floats (JSON-safe)."""
    if not samples:
        return {"n": 0}
    a = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p90": float(np.percentile(a, 90)),
        "sum": float(a.sum()),
    }


def _time_call(fn, reps, warmup=2):
    """Time ``fn`` (ms per call): ``warmup`` untimed calls then ``reps`` samples."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _backend_cfg(name, device_index):
    """(kwargs for build_cached, device-or-None) for a backend name."""
    if name == "numpy":
        return dict(backend="numpy", device=CPU_DEVICE, **NP_SPEC_OPTS), None
    if name == "iree":
        device = evoxc.core.Device(HOST_GPU_KIND, device_index)
        return dict(backend="iree", device=device, **IREE_OPTIONS), device
    raise ValueError(f"unknown backend {name!r}")


def _host(t):
    """Device (or host) tensor -> numpy array."""
    if t.device.kind != "cpu":
        t = t.to(CPU_DEVICE)
    return t.numpy()


#: ``target_backends`` evoxc's iree family infers from a device kind (see
#: ``evoxc.pipeline._resolve_backend_device``). ``build_cached`` does this
#: inference itself before it lowers/compiles, so the probe passes only
#: ``opt_level`` -- except in the hand-rolled trace/lower/compile/load split of
#: :func:`measure_stage_breakdown`, which must spell the inferred target out or
#: the artifact is compiled for ``llvm-cpu`` and ``evoxc.load`` then refuses the
#: GPU device ("the iree artifact was compiled for target(s) llvm-cpu and
#: cannot run on the requested device ... never silently recompile").
_INFERRED_TARGET_BACKENDS = {
    "cpu": ["llvm-cpu"],
    "cuda": ["cuda"],
    "rocm": ["rocm"],
    "hip": ["rocm"],
    "vulkan": ["vulkan-spirv"],
}


def _stage_options(kwargs):
    """``lower``/``compile`` options for the manual stage decomposition.

    Everything ``_backend_cfg`` set except the build-only ``backend``/``device``
    keys, plus the ``target_backends`` a device-bearing backend gets from
    ``build_cached``'s own resolution (the numpy path resolves to ``cpu`` and
    passes no options at all).
    """
    opts = {k: v for k, v in kwargs.items() if k not in ("backend", "device")}
    device = kwargs.get("device")
    kind = getattr(device, "kind", None)
    if device is not None and kind in _INFERRED_TARGET_BACKENDS:
        opts.setdefault("target_backends", list(_INFERRED_TARGET_BACKENDS[kind]))
    return opts


# --------------------------------------------------------------------------
# Section 1-3: cold build / warm run / cache-hit overhead
# --------------------------------------------------------------------------


def measure_backend(backend, trees, n_values, device_index, mode_cfg, report):
    """Measure one backend over ``n_values``; returns its report sub-dict."""
    kwargs, device = _backend_cfg(backend, device_index)
    out = {"device": str(device) if device is not None else "cpu:0", "per_n": {}}

    for n in n_values:
        key = f"N{n}"
        spec = evoxc.TensorSpec((n, 2), evoxc.float32)
        x_np = np.asarray(
            np.random.default_rng(1234 + n).uniform(-1.0, 1.0, size=(n, 2)),
            dtype=np.float32,
        )
        # ``evoxc.from_numpy`` is a ZERO-COPY host wrap (it stages nothing); the
        # host->device upload is the explicit ``Tensor.to(device)`` below, and
        # ``evoxc.run`` itself never transfers (a device executable rejects host
        # inputs), which is why it is done once here, outside the timed region.
        x_run = x_np if device is None else evoxc.from_numpy(x_np).to(device)

        # A private cache per (backend, N) so miss/hit counters are unambiguous.
        cache = evoxc.CompileCache(maxsize=64)

        # -- (a) first build: includes one-time iree runtime/device init ------
        t0 = time.perf_counter()
        warm_exe = evoxc.build_cached(_step, spec, trees[0], cache=cache, **kwargs)
        first_build_ms = (time.perf_counter() - t0) * 1e3

        # -- (b) cold builds (cache misses) for the remaining distinct trees --
        cold_ms, cold_nodes, cold_ops = [], [], []
        for tree in trees[1:]:
            misses_before = cache.misses
            t0 = time.perf_counter()
            evoxc.build_cached(_step, spec, tree, cache=cache, **kwargs)
            cold_ms.append((time.perf_counter() - t0) * 1e3)
            if cache.misses != misses_before + 1:
                raise RuntimeError("expected a build_cached cache miss")
            cold_nodes.append(_gp_nodes(tree))
            cold_ops.append(_evoxc_ops(tree))
        # The tree used for the first build counts as a miss too.
        cold_nodes.append(_gp_nodes(trees[0]))
        cold_ops.append(_evoxc_ops(trees[0]))
        cold_all_ms = [first_build_ms] + cold_ms

        # -- (c) warm run: pure evoxc.run on the cached executable -----------
        run_dev_ms, run_read_ms = [], []
        for tree in trees[: min(len(trees), mode_cfg["trees"])]:
            exe = evoxc.build_cached(_step, spec, tree, cache=cache, **kwargs)
            run_dev_ms.extend(
                _time_call(lambda: evoxc.run(exe, x_run, tree), mode_cfg["reps"], 3)
            )
            # "Readback" is a no-op on the cpu backend (already host memory) --
            # measured anyway so the two columns are produced by the same code.
            run_read_ms.extend(
                _time_call(
                    lambda: _host(evoxc.run(exe, x_run, tree)),
                    mode_cfg["reps"],
                    3,
                )
            )

        # -- (d) cache-hit build_cached overhead -----------------------------
        hit_ms = _time_call(
            lambda: evoxc.build_cached(_step, spec, trees[0], cache=cache, **kwargs),
            mode_cfg["hit_reps"],
            3,
        )

        out["per_n"][key] = {
            "first_build_ms": first_build_ms,
            "cold_build_ms": _stats(cold_all_ms),
            "cold_build_steady_ms": _stats(cold_ms),
            "warm_run_device_only_ms": _stats(run_dev_ms),
            "warm_run_readback_ms": _stats(run_read_ms),
            "hit_build_ms": _stats(hit_ms),
            "cache_misses": cache.misses,
            "cache_hits": cache.hits,
            "gp_nodes": cold_nodes,
            "evoxc_ops": cold_ops,
            "spec": f"({n}, 2) float32",
        }
        report(f"  {backend:6s} {key:7s} first={first_build_ms:9.1f} ms  "
               f"cold(steady)={out['per_n'][key]['cold_build_steady_ms'].get('median', float('nan')):8.1f} ms  "
               f"run(dev)={out['per_n'][key]['warm_run_device_only_ms'].get('median', float('nan')):7.3f} ms  "
               f"hit={out['per_n'][key]['hit_build_ms']['median']:7.3f} ms")

        # Keep the executables alive so the shared-cache/GPU handles stay warm
        # (they are in ``cache`` anyway; the local names just avoid a GC race).
        del warm_exe

    return out


def warm_up_backend(backend, trees, n, device_index):
    """Throwaway builds so the measured sections start from a warm process.

    The first iree builds of a process are several times slower than the steady
    state (compiler runtime / HAL warm-up), and that penalty leaks into a whole
    section if it is not paid up front. Returns the elapsed seconds.
    """
    kwargs, _ = _backend_cfg(backend, device_index)
    spec = evoxc.TensorSpec((n, 2), evoxc.float32)
    cache = evoxc.CompileCache(maxsize=8)
    t0 = time.perf_counter()
    for tree in trees[:WARMUP_BUILDS]:
        evoxc.build_cached(_step, spec, tree, cache=cache, **kwargs)
    return time.perf_counter() - t0


def measure_stage_breakdown(backend, tree, n, device_index):
    """Split a cold build into trace / lower / compile / load (one tree).

    Three repetitions, median reported: the very first trace of a process pays
    the location-capture warm-up (linecache/inspect) and would badly overstate
    the trace stage. A throwaway build warms all four stages first.
    """
    kwargs, device = _backend_cfg(backend, device_index)
    opts = _stage_options(kwargs)
    spec = evoxc.TensorSpec((n, 2), evoxc.float32)

    def once():
        t0 = time.perf_counter()
        graph = evoxc.trace(_step, spec, tree)
        trace_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        lowered = evoxc.lower(graph, backend=backend, **opts)
        lower_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        artifact = evoxc.compile(lowered, **opts)
        compile_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        evoxc.load(artifact, backend=backend, device=device)
        load_ms = (time.perf_counter() - t0) * 1e3
        return [trace_ms, lower_ms, compile_ms, load_ms]

    once()  # warm-up (stage modules, device handles, location machinery)
    samples = np.asarray([once() for _ in range(BREAKDOWN_REPS)], dtype=np.float64)
    med = np.median(samples, axis=0)
    res = {
        name: float(med[i])
        for i, name in enumerate(("trace_ms", "lower_ms", "compile_ms", "load_ms"))
    }
    res["total_ms"] = float(med.sum())
    res["reps"] = int(samples.shape[0])
    return res


# --------------------------------------------------------------------------
# Section 4: EVOXC_DISABLE_LOCATIONS (one subprocess per setting)
# --------------------------------------------------------------------------

_CHILD_MARKER = "#PERF-PROBE-JSON# "


def child_locations(setting, trees, reps, device_index, backends):
    """Child-process body: measure with ``EVOXC_DISABLE_LOCATIONS`` as set.

    ``set(``by the parent``)``; prints one marked JSON line on stdout. A separate
    process per setting guarantees no import-time state and no shared cache leak
    between the two settings.
    """
    res = {"disable_locations": bool(setting), "trees": len(trees)}
    spec = evoxc.TensorSpec((1000, 2), evoxc.float32)

    if "numpy" in backends:
        # trace-only: the location walk happens at op construction (trace time)
        trace_ms = []
        for tree in trees:
            for _ in range(reps):
                t0 = time.perf_counter()
                evoxc.trace(_step, spec, tree)
                trace_ms.append((time.perf_counter() - t0) * 1e3)
        cache = evoxc.CompileCache(maxsize=64)
        cold_ms = []
        for tree in trees:
            t0 = time.perf_counter()
            evoxc.build_cached(_step, spec, tree, cache=cache, backend="numpy",
                               device=CPU_DEVICE)
            cold_ms.append((time.perf_counter() - t0) * 1e3)
        hit_ms = _time_call(
            lambda: evoxc.build_cached(_step, spec, trees[0], cache=cache,
                                       backend="numpy", device=CPU_DEVICE),
            reps, 3,
        )
        res["numpy"] = {
            "trace_ms": _stats(trace_ms),
            "cold_build_ms": _stats(cold_ms),
            "hit_build_ms": _stats(hit_ms),
        }

    if "iree" in backends:
        kwargs, _ = _backend_cfg("iree", device_index)
        cache = evoxc.CompileCache(maxsize=64)
        cold_ms = []
        for i, tree in enumerate(trees):
            t0 = time.perf_counter()
            evoxc.build_cached(_step, spec, tree, cache=cache, **kwargs)
            dt = (time.perf_counter() - t0) * 1e3
            if i:  # drop the first build (iree runtime init)
                cold_ms.append(dt)
        res["iree"] = {"cold_build_steady_ms": _stats(cold_ms)}

    print(_CHILD_MARKER + json.dumps(res))
    return res


def measure_locations(trees, reps, device_index, backends, probe_path, report):
    """Run the child twice (unset / "1") and diff the results."""
    out = {"per_setting": {}}
    for setting in (False, True):
        env = dict(os.environ)
        env.pop("EVOXC_DISABLE_LOCATIONS", None)
        if setting:
            env["EVOXC_DISABLE_LOCATIONS"] = "1"
        cmd = [sys.executable, probe_path, "--child-locations",
               "1" if setting else "0", "--trees", str(len(trees)),
               "--reps", str(reps), "--device-index", str(device_index),
               "--backends", ",".join(backends)]
        t0 = time.perf_counter()
        # Stdout is parsed for the marker line; iree's nanobind leak report goes
        # to stderr at exit and is irrelevant here (exit code stays 0).
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.perf_counter() - t0
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(_CHILD_MARKER):
                payload = json.loads(line[len(_CHILD_MARKER):])
        if payload is None:
            payload = {
                "error": f"child produced no result (rc={proc.returncode})",
                "stderr_tail": proc.stderr[-800:],
            }
        payload["wall_s"] = dt
        out["per_setting"]["disable_locations=%d" % (1 if setting else 0)] = payload
        report(f"  EVOXC_DISABLE_LOCATIONS={'1' if setting else '0'}: "
               f"{json.dumps({k: v for k, v in payload.items() if k != 'stderr_tail'})[:160]}")

    # Deltas between the two settings (trace + cold build, per backend).
    a = out["per_setting"]["disable_locations=0"]
    b = out["per_setting"]["disable_locations=1"]
    deltas = {}
    for backend in backends:
        if backend in a and backend in b and "error" not in a and "error" not in b:
            d = {}
            for stage in a[backend]:
                m0 = a[backend][stage].get("median")
                m1 = b[backend][stage].get("median")
                if m0 and m1:
                    d[stage] = {
                        "median_with_locations_ms": m0,
                        "median_locations_disabled_ms": m1,
                        "delta_ms": m1 - m0,
                        "speedup_x": m0 / m1,
                    }
            deltas[backend] = d
    out["deltas"] = deltas
    return out


# --------------------------------------------------------------------------
# Section 5: parallel build attempts (process pool + thread pool)
# --------------------------------------------------------------------------


def _build_worker(payload):
    """Process-/thread-pool body: build ``trees`` and time the STEADY-STATE
    cold-build path.

    ``warmup_tree`` is built first and is NOT part of the timed set (it carries
    the one-time iree runtime/device init and cannot be one of ``trees``, or the
    timed loop would be a cache hit). ``Executable`` objects cannot cross process
    boundaries (device handles / vmfb), so only the build throughput is measured;
    the return value is (index, trees_built, elapsed_seconds, error).
    """
    index, warmup_tree, trees, n, device_index, backend_name = payload
    try:
        if warmup_tree in trees or len(set(trees)) != len(trees):
            raise RuntimeError("bad payload: warmup/duplicate trees in the timed set")
        kwargs, _ = _backend_cfg(backend_name, device_index)
        spec = evoxc.TensorSpec((n, 2), evoxc.float32)
        cache = evoxc.CompileCache(maxsize=64)
        evoxc.build_cached(_step, spec, warmup_tree, cache=cache, **kwargs)
        t0 = time.perf_counter()
        for tree in trees:
            evoxc.build_cached(_step, spec, tree, cache=cache, **kwargs)
        elapsed = time.perf_counter() - t0
        return index, len(trees), elapsed, None
    except Exception as exc:  # surfaced in the report, never swallowed silently
        return index, 0, 0.0, f"{type(exc).__name__}: {exc}"


def _thread_worker(payload):
    """Thread-pool body: same measurement, same process."""
    return _build_worker(payload)


def _make_payloads(trees, workers, per_worker, n, device_index, backend_name):
    """One payload per worker: (index, warmup_tree, timed_trees, ...).

    The warm-up tree is chosen outside the worker's timed set (it absorbs the
    iree runtime/device init), and every worker's timed set holds distinct trees.
    """
    payloads = []
    for i in range(workers):
        timed = tuple(trees[(i * per_worker + j) % len(trees)]
                      for j in range(per_worker))
        warmup = next((t for t in trees if t not in timed), trees[0])
        payloads.append((i, warmup, timed, n, device_index, backend_name))
    return payloads


def measure_parallel(trees, n, device_index, backend_name, mode_cfg, report,
                     serial_ms=None):
    """Bounded process-pool and thread-pool build-throughput attempts.

    Serial baseline for the speed-up figure is the per-tree steady-state cold
    build measured in sections 1-3 (``serial_ms``).
    """
    import multiprocessing as mp

    out = {}
    budget = mode_cfg["parallel_budget"]
    base = None
    if serial_ms:
        base = {"per_tree_ms": serial_ms, "builds_per_s": 1e3 / serial_ms}
        out["serial_baseline"] = base
        report(f"  serial baseline: {serial_ms:.0f} ms/tree = "
               f"{base['builds_per_s']:.2f} builds/s")
    # ---- process pool (spawn: fork + GPU/iree state does not mix) --------
    w = mode_cfg["parallel_workers"]
    per = mode_cfg["parallel_trees_per_worker"]
    payloads = _make_payloads(trees, w, per, n, device_index, backend_name)
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    try:
        with ctx.Pool(processes=w) as pool:
            async_res = pool.map_async(_build_worker, payloads)
            results = async_res.get(timeout=budget)
        wall = time.perf_counter() - t0
        ok = [r for r in results if r[3] is None]
        built = sum(r[1] for r in ok)
        out["process_pool"] = {
            "workers": w,
            "trees_per_worker": per,
            "wall_s": wall,
            "workers_ok": len(ok),
            "trees_built": built,
            "builds_per_s": (built / wall) if wall > 0 else None,
            "per_worker_elapsed_s": [r[2] for r in results],
            "errors": [r[3] for r in results if r[3] is not None],
        }
        report(f"  process pool: {len(ok)}/{w} workers ok, {built} builds in "
               f"{wall:.2f} s = {out['process_pool']['builds_per_s']:.2f} builds/s"
               + (f" ({out['process_pool']['builds_per_s'] / (1e3 / base['per_tree_ms']):.2f}x serial)"
                  if base else ""))
    except Exception as exc:  # TimeoutError from .get, spawn failures, ...
        out["process_pool"] = {
            "workers": w, "trees_per_worker": per, "budget_s": budget,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report(f"  process pool: SKIPPED ({type(exc).__name__}: {exc})")

    # ---- thread pool (in-process; iree GPU-HAL devices are thread-affine,
    # so failures here are a finding, not a probe bug) ---------------------
    tw = mode_cfg["thread_workers"]
    tper = mode_cfg["thread_trees_per_worker"]
    tpayloads = _make_payloads(trees, tw, tper, n, device_index, backend_name)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=tw) as ex:
            futures = [ex.submit(_thread_worker, p) for p in tpayloads]
            done, not_done = wait(futures, timeout=budget)
        wall = time.perf_counter() - t0
        results = [f.result() for f in futures if f in done]
        ok = [r for r in results if r[3] is None]
        built = sum(r[1] for r in ok)
        out["thread_pool"] = {
            "threads": tw,
            "trees_per_thread": tper,
            "wall_s": wall,
            "threads_ok": len(ok),
            "threads_timed_out": len(not_done),
            "trees_built": built,
            "builds_per_s": (built / wall) if wall > 0 else None,
            "per_thread_elapsed_s": [r[2] for r in results],
            "errors": [r[3] for r in results if r[3] is not None],
        }
        report(f"  thread pool: {len(ok)}/{tw} threads ok, {built} builds in "
               f"{wall:.2f} s = {out['thread_pool']['builds_per_s']:.2f} builds/s"
               + (f" ({out['thread_pool']['builds_per_s'] / (1e3 / base['per_tree_ms']):.2f}x serial)"
                  if base else "")
               + (f" ({len(not_done)} timed out)" if not_done else ""))
    except Exception as exc:
        out["thread_pool"] = {
            "threads": tw, "trees_per_thread": tper, "budget_s": budget,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report(f"  thread pool: SKIPPED ({type(exc).__name__}: {exc})")

    return out


# --------------------------------------------------------------------------
# Section 6: extrapolation
# --------------------------------------------------------------------------


def extrapolate(per_backend, pops, distinct_fraction):
    """Per-generation wall time (s) for each backend/N/pop assumption.

    Model: every generation runs all ``pop`` trees (warm run) and each DISTINCT
    structure is built once (cold build). ``all-distinct`` is the worst case
    (D = pop); ``mixed`` assumes D = ``distinct_fraction`` * pop -- a
    deliberately pessimistic bound for a converged GP population, where most
    offspring reuse structures that already exist. ``all-cached`` (D = 0) is the
    floor.
    """
    out = {}
    for backend, data in per_backend.items():
        for nkey, d in data["per_n"].items():
            run_ms = d["warm_run_device_only_ms"].get("median")
            build_ms = d["cold_build_steady_ms"].get("median") or d["cold_build_ms"].get("median")
            if run_ms is None or build_ms is None:
                continue
            row = {}
            for pop in pops:
                row[f"pop{pop}"] = {
                    "all_distinct_s": (pop * run_ms + pop * build_ms) / 1e3,
                    "mixed_s": (pop * run_ms + distinct_fraction * pop * build_ms) / 1e3,
                    "all_cached_s": (pop * run_ms) / 1e3,
                    "builds_per_s": 1e3 / build_ms,
                    "assumed_distinct": {
                        "all_distinct": pop,
                        "mixed": int(round(distinct_fraction * pop)),
                    },
                }
            row["warm_run_ms"] = run_ms
            row["cold_build_ms"] = build_ms
            out.setdefault(backend, {})[nkey] = row
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def print_tables(out, pops, distinct_fraction, n_values):
    """Human-readable tables (returns a list of lines; also emitted as stdout
    after the JSON block)."""
    lines: list = []
    p = lines.append
    report = out.get("cold_build", {})
    p("")
    p("=" * 88)
    p("PER-TREE COST (median ms)  " + "".join(
        f"{b + ' (' + ('cpu' if b == 'numpy' else f'{HOST_GPU_KIND}:0, O1') + ')':>26s}"
        for b in report) or "no backend measured")
    p("=" * 88)
    for n in n_values:
        k = f"N{n}"
        npd = report.get("numpy", {}).get("per_n", {}).get(k, {})
        ird = report.get("iree", {}).get("per_n", {}).get(k, {})
        p(f"data {k:6s} (x shape ({n}, 2) f32)"
          + ("   [numpy readback is a no-op: already host data]"
             if n == n_values[0] else ""))
        for label, field, spec in (
            ("cold build (steady state)", "cold_build_steady_ms", "9.1f"),
            ("first build (init incl.)", "first_build_ms", "9.1f"),
            ("cache-hit build_cached", "hit_build_ms", "9.4f"),
            ("warm run (device only)", "warm_run_device_only_ms", "9.4f"),
            ("warm run + host readback", "warm_run_readback_ms", "9.4f"),
        ):
            def cell(d):
                v = d.get(field)
                if isinstance(v, dict):
                    v = v.get("median")
                return _fmt(v, spec)
            # Min-max range on the build rows: this box is shared, so a single
            # median hides a 2-4x scheduler-induced spread.
            def rng(name, d):
                v = d.get(field)
                if not isinstance(v, dict) or "min" not in v:
                    return ""
                return f"   {name} [{v['min']:.0f}-{v['max']:.0f}]"
            trailing = ("   " + "".join(rng(b, d) for b, d in
                                        (("numpy", npd), ("iree", ird)) if d)
                        if field == "cold_build_steady_ms" else "")
            p(f"    {label:28s} {cell(npd):>10s} ms          {cell(ird):>10s} ms"
              + trailing)
    p("")
    p("=" * 88)
    p(f"EXTRAPOLATED PER-GENERATION WALL TIME (s)  "
      f"[mixed = {int(distinct_fraction * 100)}% distinct structures]")
    p("=" * 88)
    p(f"{'backend / pop':22s}" + "".join(f"{str(k):>22s}" for k in n_values)
      + "   (cell = all-distinct / mixed)")
    for backend in ("numpy", "iree"):
        ex = out.get("extrapolation", {}).get(backend)
        if not ex:
            continue
        for pop in pops:
            cells = []
            for n in n_values:
                row = ex.get(f"N{n}", {}).get(f"pop{pop}")
                cells.append(
                    f"{row['all_distinct_s']:10.1f}/{row['mixed_s']:<11.1f}"
                    if row else f"{'-':>22s}"
                )
            p(f"{backend + ' pop=' + str(pop):22s}" + "".join(cells))
    p("")
    p("  build throughput / all-cached floor (per backend, last N):")
    for backend in ("numpy", "iree"):
        ex = out.get("extrapolation", {}).get(backend)
        if not ex:
            continue
        for pop in pops:
            row = ex.get(f"N{n_values[-1]}", {}).get(f"pop{pop}")
            if row is not None:
                p(f"    {backend:6s} pop={pop:<5d} N={n_values[-1]:<6d} "
                  f"{row['builds_per_s']:8.2f} distinct builds/s   "
                  f"all-cached gen = {row['all_cached_s']:7.2f} s")
    p("")
    p("=" * 88)
    p("EVOXC_DISABLE_LOCATIONS (median ms, N=1000)")
    p("=" * 88)
    deltas = out.get("locations", {}).get("deltas", {})
    for backend, stages in deltas.items():
        for stage, d in stages.items():
            p(f"  {backend:6s} {stage:26s} on={d['median_with_locations_ms']:9.3f}  "
              f"disabled={d['median_locations_disabled_ms']:9.3f}  "
              f"delta={d['delta_ms']:+8.3f} ms  ({d['speedup_x']:.2f}x)")
    if not deltas:
        p("  (no usable child results)")
    p("")
    p("=" * 88)
    p("PARALLEL BUILD ATTEMPT (throughput only; Executables cannot cross processes)")
    p("=" * 88)
    sb = out.get("parallel", {}).get("serial_baseline", {})
    if sb:
        p(f"  serial baseline: {sb['per_tree_ms']:8.1f} ms/tree = "
          f"{sb['builds_per_s']:.2f} builds/s")
    for pool, label in (("process_pool", "process pool"), ("thread_pool", "thread pool")):
        d = out.get("parallel", {}).get(pool, {})
        if not d or d.get("error"):
            p(f"  {label:13s} {d.get('error', 'skipped')}")
            continue
        p(f"  {label:13s} {d.get('workers', d.get('threads'))} workers x "
          f"{d.get('trees_per_worker', d.get('trees_per_thread'))} trees: "
          f"{d.get('trees_built', 0)} builds in {d.get('wall_s', 0):.2f} s = "
          f"{_fmt(d.get('builds_per_s'), '.2f')} builds/s"
          + (f" ({_fmt(d.get('speedup_vs_serial'), '.2f')}x serial)"
             if d.get("speedup_vs_serial") else "")
          + (f"  errors={d['errors']}" if d.get("errors") else ""))
    p("")
    p("=" * 88)
    p("RECOMMENDATION")
    p("=" * 88)
    for line in out.get("recommendation", _recommendation_lines(out)):
        p("  " + line)
    p("")
    return lines


def _recommendation_lines(out):
    """Turn the measurements into a short, evidence-stated recommendation."""
    ex = out.get("extrapolation", {})
    np_ex = ex.get("numpy", {}).get("N1000", {}).get("pop1000", {})
    ir_ex = ex.get("iree", {}).get("N1000", {}).get("pop1000", {})
    lines = []
    if np_ex and ir_ex:
        lines.append(
            "Default backend: numpy. At pop=1000/N=1000 the mixed-case generation "
            f"costs {np_ex['mixed_s']:.1f} s on numpy vs {ir_ex['mixed_s']:.1f} s on iree "
            f"(all-distinct {np_ex['all_distinct_s']:.0f} s vs {ir_ex['all_distinct_s']:.0f} s), "
            f"and numpy is also cheaper per arm ({np_ex['builds_per_s']:.0f} vs "
            f"{ir_ex['builds_per_s']:.1f} distinct builds/s)."
        )
        lines.append(
            f"Keep iree (Device('{HOST_GPU_KIND}', 0), opt_level='O1'; the target "
            "backend follows the device kind) as an opt-in: the engine must take "
            "backend/device as configuration and default to a path that runs "
            "without a GPU."
        )
        np_run = ex.get("numpy", {}).get("N1000", {}).get("warm_run_ms")
        ir_run = ex.get("iree", {}).get("N1000", {}).get("warm_run_ms")
        if np_run is not None and ir_run is not None:
            lines.append(
                f"Per-tree warm run: numpy {np_run:.3f} ms vs iree {ir_run:.3f} ms "
                "(iree is launch-bound at these graph/data sizes)."
            )
    else:
        lines.append("Insufficient data for a recommendation (see errors above).")

    # Build-time lever: location capture dominates tracing unless disabled.
    np_trace = (out.get("locations", {}).get("deltas", {})
                .get("numpy", {}).get("trace_ms"))
    if np_trace:
        lines.append(
            "Set EVOXC_DISABLE_LOCATIONS=1 for the engine's build phase: tracing "
            f"drops from {np_trace['median_with_locations_ms']:.1f} ms to "
            f"{np_trace['median_locations_disabled_ms']:.1f} ms per tree "
            f"({np_trace['speedup_x']:.1f}x) and the numpy cold build follows "
            "(locations are only useful for graph diagnostics, not for GP)."
        )

    # Parallel build: bounded finding.
    pp = out.get("parallel", {}).get("process_pool", {})
    tp = out.get("parallel", {}).get("thread_pool", {})
    if pp and pp.get("speedup_vs_serial"):
        extra = ""
        if tp and tp.get("speedup_vs_serial"):
            extra = (f"; {tp.get('threads')} in-process threads give "
                     f"{tp['speedup_vs_serial']:.2f}x")
        lines.append(
            f"Parallel builds are not a workaround: {pp.get('workers')} processes "
            f"reach only {pp['speedup_vs_serial']:.2f}x serial build throughput{extra} "
            "-- the cost is dominated by serialized iree HAL load/device work."
        )
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="evoxc build/run economics probe for EvoGP's per-tree engine."
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="fast preset (default)")
    mode.add_argument("--full", action="store_true", help="tighter statistics, slower")
    ap.add_argument("--backends", default="numpy,iree",
                    help="comma list of backends to measure (default both)")
    ap.add_argument("--n", default=None,
                    help="comma list of data sizes N (default 100,1000; "
                         "--full additionally measures 10000)")
    ap.add_argument("--trees", type=int, default=None,
                    help="number of distinct trees (default: preset)")
    ap.add_argument("--seed", type=int, default=20240, help="tree-set seed")
    ap.add_argument("--device-index", type=int, default=0,
                    help=f"iree GPU device index (device kind {HOST_GPU_KIND})")
    ap.add_argument("--skip-parallel", action="store_true",
                    help="skip the parallel-build attempt")
    ap.add_argument("--skip-locations", action="store_true",
                    help="skip the EVOXC_DISABLE_LOCATIONS subprocess comparison")
    ap.add_argument("--json-out", default=None, help="also write the JSON report here")
    # Internal: child mode for the EVOXC_DISABLE_LOCATIONS measurement.
    ap.add_argument("--child-locations", choices=("0", "1"), default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--reps", type=int, default=50, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    probe_path = os.path.abspath(__file__)
    # Unprotected DIV (the real function set's LOOSE_DIV guard only changes
    # values) intentionally produces inf/nan for some trees -- silence the
    # numpy warnings, the parity check already excludes non-finite entries.
    np.seterr(divide="ignore", invalid="ignore", over="ignore")
    mode_cfg = dict(FULL if args.full else QUICK)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    n_values = [int(v) for v in (args.n.split(",") if args.n else ("100", "1000"))
                if v.strip()]
    if args.full and args.n is None:
        n_values.append(10000)  # extension row: does iree's fixed cost amortize?
    n_trees = args.trees or mode_cfg["trees"]
    trees = build_tree_set(max(n_trees, 3), args.seed)

    if args.child_locations is not None:
        child_locations(args.child_locations == "1", trees, args.reps,
                        args.device_index, backends)
        return 0

    out = {
        "probe": {
            "mode": "full" if args.full else "quick",
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device_index": args.device_index,
            "host_gpu_kind": HOST_GPU_KIND,
            "seed": args.seed,
        },
        "config": {
            "trees": n_trees,
            "n_values": n_values,
            "reps": mode_cfg["reps"],
            "backends": backends,
            "iree_options": IREE_OPTIONS,
            "distinct_fraction": 0.5,
            "pops": [200, 1000],
        },
        "trees": {
            "gp_nodes": [_gp_nodes(t) for t in trees],
            "evoxc_ops": [_evoxc_ops(t) for t in trees],
            "shapes": ["left-deep chain", "right-deep chain"] + ["random"] * (len(trees) - 2),
        },
        "cold_build": {},
        "stage_breakdown": {},
        "parallel": {},
        "errors": {},
    }
    try:
        import importlib.metadata

        out["probe"]["evoxc_version"] = importlib.metadata.version("evoxc")
    except Exception:  # a plain checkout on PYTHONPATH is not an installed dist
        out["probe"]["evoxc_version"] = getattr(evoxc, "__version__", None)

    lines = []
    report = lines.append

    report("")
    report("EvoGP evoxc economics probe")
    report(f"  mode={out['probe']['mode']} trees={n_trees} N={n_values} "
           f"backends={backends} evoxc={out['probe'].get('evoxc_version')}")
    report(f"  tree GP nodes: {out['trees']['gp_nodes']}")
    report(f"  tree evoxc ops: {out['trees']['evoxc_ops']}")
    report("")

    if "iree" in backends:
        try:  # cheap availability probe so a missing GPU degrades gracefully
            kwargs, dev = _backend_cfg("iree", args.device_index)
            evoxc.build_cached(_step, evoxc.TensorSpec((4, 2), evoxc.float32),
                               trees[0], cache=SHARED_CACHE, **kwargs)
            report(f"  iree {HOST_GPU_KIND} available on device {args.device_index}")
        except Exception as exc:
            out["errors"]["iree"] = f"{type(exc).__name__}: {exc}"
            report(f"  iree {HOST_GPU_KIND} UNAVAILABLE ({type(exc).__name__}: "
                   f"{exc}) -> skipping")
            backends = [b for b in backends if b != "iree"]

    # ---- section 1-3 -----------------------------------------------------
    for backend in backends:
        report(f"section 1-3: cold build / warm run / cache-hit ({backend})")
        try:
            warm_s = warm_up_backend(backend, trees, min(n_values), args.device_index)
            out.setdefault("warmup_s", {})[backend] = warm_s
            report(f"  {backend:6s} warm-up: {WARMUP_BUILDS} throwaway builds in "
                   f"{warm_s:.2f} s (excluded from the stats)")
            out["cold_build"][backend] = measure_backend(
                backend, trees, n_values, args.device_index, mode_cfg, report
            )
        except Exception as exc:
            out["errors"][backend] = f"{type(exc).__name__}: {exc}"
            report(f"  {backend} FAILED: {type(exc).__name__}: {exc}")
    out["warm_run"] = {
        b: {k: v["warm_run_device_only_ms"] for k, v in d["per_n"].items()}
        for b, d in out["cold_build"].items()
    }
    out["hit_build"] = {
        b: {k: v["hit_build_ms"] for k, v in d["per_n"].items()}
        for b, d in out["cold_build"].items()
    }

    # ---- parity sanity check (the numbers must describe a working path) ---
    out["parity"] = {}
    for backend in backends:
        try:
            kwargs, dev = _backend_cfg(backend, args.device_index)
            spec = evoxc.TensorSpec((256, 2), evoxc.float32)
            tree = trees[0]
            x = np.asarray(
                np.random.default_rng(99).uniform(-1.0, 1.0, size=(256, 2)),
                dtype=np.float32,
            )
            exe = evoxc.build_cached(_step, spec, tree, cache=SHARED_CACHE, **kwargs)
            xb = x if dev is None else evoxc.from_numpy(x).to(dev)
            got = _host(evoxc.run(exe, xb, tree)).astype(np.float64)
            ref = _host(evoxc.run(
                evoxc.build_cached(_step, spec, tree, cache=SHARED_CACHE,
                                   backend="numpy", device=CPU_DEVICE),
                x, tree)).astype(np.float64)
            g = np.nan_to_num(got)
            r = np.nan_to_num(ref)
            out["parity"][backend] = {
                "max_abs_diff": float(np.max(np.abs(g - r))) if g.size else 0.0,
                "nonfinite": int((~np.isfinite(got)).sum()),
            }
            report(f"  parity {backend}: max|diff|={out['parity'][backend]['max_abs_diff']:.3e}")
        except Exception as exc:
            out["parity"][backend] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- section 4: stage breakdown --------------------------------------
    report("")
    report("section 4: cold-build stage breakdown (N=1000, one tree)")
    for backend in backends:
        try:
            out["stage_breakdown"][backend] = measure_stage_breakdown(
                backend, trees[0], 1000, args.device_index
            )
            sb = out["stage_breakdown"][backend]
            report(f"  {backend:6s} trace={sb['trace_ms']:8.2f}  lower={sb['lower_ms']:7.2f}  "
                   f"compile={sb['compile_ms']:8.2f}  load={sb['load_ms']:8.2f}  "
                   f"total={sb['total_ms']:8.2f} ms (median of {sb['reps']})")
        except Exception as exc:
            out["errors"][f"breakdown_{backend}"] = f"{type(exc).__name__}: {exc}"
            report(f"  {backend} breakdown FAILED: {type(exc).__name__}: {exc}")

    # ---- section 5: EVOXC_DISABLE_LOCATIONS --------------------------------
    report("")
    report("section 5: EVOXC_DISABLE_LOCATIONS (subprocess per setting, N=1000)")
    if args.skip_locations:
        report("  skipped (--skip-locations)")
        out["locations"] = {"skipped": True}
    else:
        try:
            out["locations"] = measure_locations(
                trees[: min(len(trees), 5)], max(5, mode_cfg["reps"] // 4),
                args.device_index, backends, probe_path, report)
        except Exception as exc:
            out["errors"]["locations"] = f"{type(exc).__name__}: {exc}"
            report(f"  FAILED: {type(exc).__name__}: {exc}")
            out["locations"] = {}

    # ---- section 6: parallel build attempts ------------------------------
    report("")
    report("section 6: parallel build attempt (iree, throughput only)")
    if args.skip_parallel or "iree" not in backends:
        report("  skipped")
        out["parallel"] = {"skipped": True}
    else:
        try:
            serial_ms = (out["cold_build"].get("iree", {}).get("per_n", {})
                         .get("N1000", {}).get("cold_build_steady_ms", {}).get("median"))
            out["parallel"] = measure_parallel(
                trees, 1000, args.device_index, "iree", mode_cfg, report,
                serial_ms=serial_ms)
            # Speed-up figures against the serial baseline.
            for pool in ("process_pool", "thread_pool"):
                d = out["parallel"].get(pool)
                if d and d.get("builds_per_s") and serial_ms:
                    d["speedup_vs_serial"] = d["builds_per_s"] / (1e3 / serial_ms)
        except Exception as exc:
            out["errors"]["parallel"] = f"{type(exc).__name__}: {exc}"
            report(f"  FAILED: {type(exc).__name__}: {exc}")
            out["parallel"] = {}

    # ---- extrapolation ---------------------------------------------------
    out["extrapolation"] = extrapolate(out["cold_build"], out["config"]["pops"], 0.5)
    out["recommendation"] = _recommendation_lines(out)

    # ---- print + persist -------------------------------------------------
    print("\n".join(lines))
    print("\n".join(print_tables(out, out["config"]["pops"], 0.5, n_values)))

    ts = time.strftime("%Y%m%d-%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"evogp_evoxc_perf_probe_{ts}.json")
    for path in ([tmp_path] + ([args.json_out] if args.json_out else [])):
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    print(f"[probe] JSON report: {tmp_path}")
    if args.json_out:
        print(f"[probe] JSON copy  : {args.json_out}")
    print("JSON-BEGIN")
    print(json.dumps(out, indent=2, default=str))
    print("JSON-END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
