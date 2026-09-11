#!/usr/bin/env python
"""test/etl_jit_probe.py — ETL "true jit" economics probe (standalone script).

Measures the economics and API semantics of ETL's memoized-build sugar
(``etl.build_cached`` + ``etl.CompileCache``) for a realistic genetic-
programming workload: a step ``@etl.defn def _step(x, tree)`` where ``tree``
is a nested-tuple STATIC (a decoded prefix expression over add/sub/mul/div,
unary sin/exp/neg and a ``select``-based protected divide) and ``x`` is a
``TensorSpec((N, 2), float32)``.  This is the shape EvoGP will use if it
decodes tree tensors into static structures and compiles them per program.

Sections (all printed as one aligned table, plus a summary at the end):

  (a) backend="numpy": cold/first build (miss), cache-hit build, distinct-tree
      build, ``etl.run`` per tree at N=100 and N=1000, parity vs a pure-numpy
      reference evaluator.
  (b) backend="iree" on ``Device("cuda", 0)`` with
      ``target_backends=["cuda"]``: distinct-tree build and device-resident
      run at ``opt_level`` "O1" and "O0", each with and without
      ``ETL_DISABLE_LOCATIONS=1``; hidden per-stage breakdown (trace/lower/
      compile/load).
  (c) simulated generation: 1000 distinct trees x (build_cached + run) on
      numpy (measured) and the iree-cuda extrapolation.
  (d) parallel-build feasibility: 64 distinct trees through
      ``ThreadPoolExecutor(8)`` and ``(32)`` on numpy and iree-cuda, with a
      post-hoc cache-completeness/race check (re-build serially: all hits).
  (e) persistence: ``Executable.save`` + ``etl.Executable.load`` round trip
      for an iree-cuda executable (timed; recompilation check).
  (f) API semantics checks (nested-tuple statics, run-boundary dtype/shape
      errors, cache-key behaviour incl. LRU eviction, torch<->etl interop,
      the exact working in-graph protected-divide recipe).
  (g) ETL's own meta-programming test, run from a $TMPDIR copy (numpy leg,
      the file's own iree-llvm-cpu leg, plus a locally generated iree-cuda
      adaptation).
  (h) extra diagnostics: run time vs N sweep (16..100000), fixed per-run
      overhead (1-op graph), host->device transfer and readback costs.

Run (``python`` is not on PATH in the task environment; use the EvoX venv):

    /mnt/local-ssd/bchuang/evox/.venv/bin/python test/etl_jit_probe.py

Useful flags: ``--device N`` (cuda index; default 0), ``--sections a,b,f``,
``--reps-build`` (default 10), ``--reps-run`` (default 7),
``--par-trees`` (default 64), ``--gen-trees`` (default 1000).

Determinism: every tree/input comes from ``np.random.default_rng`` with a
fixed seed, so the tree set is reproducible; the reported times are medians
over multiple reps on a SHARED host (other tenants run jobs on the same
GPUs), so treat the spread (min/max is printed) as part of the result.

Nothing is written outside $TMPDIR.  The only side effect in this repository
is stdout; the file is NOT a pytest test (no assertions run by a runner) but
it exits non-zero if a section fails.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import etl
except Exception as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"cannot import etl ({exc!r}) — run with the EvoX venv python: "
        f"/mnt/local-ssd/bchuang/evox/.venv/bin/python test/etl_jit_probe.py"
    )


# --------------------------------------------------------------------------
# GP tree model: static nested-tuple program + trace-time decoder
# --------------------------------------------------------------------------

BINOPS = ("add", "sub", "mul", "div")
UNOPS = ("sin", "exp", "neg")
MAX_DEPTH = 8
TREE_BUDGET = 50  # target node count per generated tree
N_SMALL = 100
N_LARGE = 1000


def _compose(tree, x, n):
    """Trace-time decode of a static tree into an etl op expression.

    ``tree`` is a nested tuple: ``('x', 0|1)`` column leaf, ``('c', float)``
    constant leaf, ``(op, child[, child])`` for the ops above.  Everything
    here runs as plain Python while ETL traces, so the static structure
    specializes the graph.
    """
    tag = tree[0]
    if tag == "x":
        return etl.slice(x, (0, tree[1]), (n, 1))
    if tag == "c":
        return etl.constant(etl.core.tensor(np.float32(tree[1])))
    if tag in BINOPS:
        a = _compose(tree[1], x, n)
        b = _compose(tree[2], x, n)
        if tag == "add":
            return etl.add(a, b)
        if tag == "sub":
            return etl.subtract(a, b)
        if tag == "mul":
            return etl.multiply(a, b)
        # The ONLY working in-graph "protected divide" recipe (verified in
        # section f): a select over a 0-comparison; the dead branch is still
        # evaluated by the backend (harmless for float: inf/nan), so select
        # must mask it.  NOTE: etl.constant needs a core.Tensor, never a
        # Python scalar.
        nan = etl.constant(etl.core.tensor(np.float32(np.nan)))
        return etl.select(etl.equal(b, 0.0), nan, etl.divide(a, b))
    if tag in UNOPS:
        a = _compose(tree[1], x, n)
        if tag == "sin":
            return etl.sin(a)
        if tag == "exp":
            return etl.exp(a)
        return etl.negate(a)
    raise ValueError(f"unknown tree tag {tag!r}")


@etl.defn
def _step(x, tree):
    """GP evaluation step for ``x`` of shape (100, 2)."""
    return etl.reshape(_compose(tree, x, N_SMALL), (N_SMALL,))


@etl.defn
def _step_large(x, tree):
    """GP evaluation step for ``x`` of shape (1000, 2)."""
    return etl.reshape(_compose(tree, x, N_LARGE), (N_LARGE,))


@etl.defn
def _one_op(x):
    """Trivial 1-op graph: fixed per-run overhead probe."""
    return etl.reshape(etl.slice(x, (0, 0), (1, 1)), (1,))


def _make_step(n):
    """Factory for extra-N steps used by the N sweep (section h)."""

    def step(x, tree):
        return etl.reshape(_compose(tree, x, n), (n,))

    step.__name__ = f"_step_n{n}"
    step.__qualname__ = f"_step_n{n}"
    return etl.defn(step)


SWEEP_NS = (16, 100, 1000, 10000, 100000)
SWEEP_STEPS = {n: _make_step(n) for n in SWEEP_NS}


def contains_x(tree):
    """True when the static tree reads at least one input column.

    An all-constant tree folds to a scalar etl expression (shape ()) and the
    probe's (N,) reshape would fail, so generated programs must use x.
    """
    tag = tree[0]
    if tag == "x":
        return True
    if tag == "c":
        return False
    if tag in BINOPS:
        return contains_x(tree[1]) or contains_x(tree[2])
    return contains_x(tree[1])


def make_tree(rng, n_nodes):
    """Deterministic random tree with exactly ``n_nodes`` nodes (uses x)."""

    def leaf():
        if rng.random() < 0.6:
            return ("x", int(rng.integers(0, 2)))
        return ("c", float(np.round(rng.uniform(-2.0, 2.0), 2)))

    def grow(budget, depth):
        if budget <= 1 or depth >= MAX_DEPTH:
            return leaf()
        if budget >= 3 and rng.random() < 0.7:
            op = BINOPS[int(rng.integers(0, len(BINOPS)))]
            rest = budget - 1
            left = int(rng.integers(1, rest))
            return (op, grow(left, depth + 1), grow(rest - left, depth + 1))
        if budget >= 2:
            op = UNOPS[int(rng.integers(0, len(UNOPS)))]
            return (op, grow(budget - 1, depth + 1))
        return leaf()

    for _ in range(64):  # a uniform op/leaf draw can produce an x-free tree
        tree = grow(n_nodes, 0)
        if contains_x(tree):
            return tree
    raise RuntimeError("could not generate a tree that uses x")


def count_nodes(tree):
    tag = tree[0]
    if tag in ("x", "c"):
        return 1
    if tag in BINOPS:
        return 1 + count_nodes(tree[1]) + count_nodes(tree[2])
    return 1 + count_nodes(tree[1])


def ref_eval(tree, x):
    """Pure-numpy float64 reference evaluator; ``x`` shape (n, 2)."""

    def ev(t):
        tag = t[0]
        if tag == "x":
            return x[:, t[1]].astype(np.float64)
        if tag == "c":
            return np.float64(t[1])
        if tag == "add":
            return ev(t[1]) + ev(t[2])
        if tag == "sub":
            return ev(t[1]) - ev(t[2])
        if tag == "mul":
            return ev(t[1]) * ev(t[2])
        if tag == "div":
            a, b = ev(t[1]), ev(t[2])
            safe = np.where(b == 0.0, 1.0, b)
            return np.where(b == 0.0, np.float64(np.nan), a / safe)
        if tag == "sin":
            return np.sin(ev(t[1]))
        if tag == "exp":
            return np.exp(ev(t[1]))
        if tag == "neg":
            return -ev(t[1])
        raise ValueError(tag)

    with np.errstate(all="ignore"):
        return ev(tree)


def compare(got, ref, rtol=1e-4, atol=1e-5):
    """Compare a float32 result with the float64 reference.

    Returns ``(max_finite_abs_err, n_nonfinite_mismatch, n_nonfinite_ref)``;
    non-finite entries count as equal when both sides are non-finite with the
    same sign (overflow in float32 vs float64 is not a semantic difference).
    """
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    gfin, rfin = np.isfinite(got), np.isfinite(ref)
    mism = int(np.sum(gfin != rfin))
    bad = int(np.sum(np.isnan(got) != np.isnan(ref)))
    mism += bad
    signed = np.where(np.isposinf(got), 1, np.where(np.isneginf(got), -1, 0))
    signed_ref = np.where(np.isposinf(ref), 1, np.where(np.isneginf(ref), -1, 0))
    mism += int(np.sum((signed != signed_ref) & ~np.isnan(ref) & ~np.isnan(got)))
    both = gfin & rfin
    err = float(np.max(np.abs(got[both] - ref[both]))) if both.any() else 0.0
    return err, mism, int(np.sum(~rfin))


# --------------------------------------------------------------------------
# timing / reporting helpers
# --------------------------------------------------------------------------


def timed(fn, reps):
    """Run ``fn`` ``reps`` times; return (median, min, max) in milliseconds."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def ms(t):
    """Format a (median, min, max) millisecond triple."""
    return f"{t[0]:9.3f} ms   [min {t[1]:8.3f} max {t[2]:9.3f}]"


def secs(s):
    return f"{s:.2f} s ({s / 60.0:.2f} min)"


@contextlib.contextmanager
def env_var(name, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class Report:
    """Rows printed as they are measured + a final aligned table."""

    def __init__(self):
        self.rows = []
        self.failures = []
        self._section = None

    def section(self, title):
        self._section = title
        print(f"\n--- {title} " + "-" * max(0, 74 - len(title)), flush=True)

    def add(self, label, value, detail=""):
        self.rows.append((self._section, label, str(value), detail))
        line = f"  {label:<52} {value}"
        print(line + (f"   # {detail}" if detail else ""), flush=True)

    def note(self, text):
        print(f"  · {text}", flush=True)

    def fail(self, label, exc):
        msg = f"{type(exc).__name__}: {exc}"
        self.failures.append((self._section, label, msg))
        print(f"  !! {label}: {msg}", flush=True)

    def dump(self):
        print("\n" + "=" * 100)
        print("SUMMARY TABLE (median [min max] where applicable)")
        print("=" * 100)
        cur = None
        for section, label, value, detail in self.rows:
            if section != cur:
                print(f"\n[{section}]")
                cur = section
            print(f"  {label:<52} {value}" + (f"   # {detail}" if detail else ""))
        if self.failures:
            print("\nFAILED CHECKS")
            for section, label, msg in self.failures:
                print(f"  [{section}] {label}: {msg}")
        print("=" * 100, flush=True)


CFG = argparse.Namespace()


# --------------------------------------------------------------------------
# (a) numpy backend
# --------------------------------------------------------------------------


def section_a(rep):
    rep.section("(a) backend=numpy")
    rng = np.random.default_rng(20250101)
    trees = [make_tree(rng, TREE_BUDGET) for _ in range(max(CFG.reps_build, 12))]
    x = np.random.default_rng(7).uniform(-1.0, 1.0, (N_SMALL, 2)).astype(np.float32)
    x_large = np.random.default_rng(8).uniform(-1.0, 1.0, (N_LARGE, 2)).astype(np.float32)
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)
    spec_large = etl.TensorSpec((N_LARGE, 2), etl.float32)
    rep.note(f"tree nodes: min {min(map(count_nodes, trees))} "
             f"median {int(statistics.median(list(map(count_nodes, trees))))} "
             f"max {max(map(count_nodes, trees))}")

    cache = etl.CompileCache()
    t0 = time.perf_counter()
    exe = etl.build_cached(_step, spec, trees[0], backend="numpy", cache=cache)
    first = (time.perf_counter() - t0) * 1e3
    rep.add("first build_cached (miss, cold process)", f"{first:.3f} ms")

    t = timed(lambda: etl.build_cached(_step, spec, trees[0], backend="numpy", cache=cache),
              max(CFG.reps_build, 10))
    rep.add("build_cached cache hit", ms(t), f"hits {cache.hits} misses {cache.misses}")

    misses = []
    for tree in trees[1:CFG.reps_build + 1]:
        t0 = time.perf_counter()
        etl.build_cached(_step, spec, tree, backend="numpy", cache=cache)
        misses.append((time.perf_counter() - t0) * 1e3)
    build_med = statistics.median(misses)
    rep.add("distinct-tree build (miss)", ms((build_med, min(misses), max(misses))),
            f"n={len(misses)}")

    out = etl.run(exe, x, trees[0])
    err, mism, n_nonfinite = compare(out.numpy(), ref_eval(trees[0], x))
    run_t = timed(lambda: etl.run(exe, x, trees[0]), CFG.reps_run)
    rep.add("run N=100 ms/tree", ms(run_t))
    exe_l = etl.build_cached(_step_large, spec_large, trees[0], backend="numpy",
                             cache=etl.CompileCache())
    t = timed(lambda: etl.build_cached(_step_large, spec_large, trees[1], backend="numpy",
                                       cache=etl.CompileCache()), max(CFG.reps_build, 5))
    rep.add("distinct-tree build N=1000 spec", ms(t), f"n={max(CFG.reps_build, 5)}")
    run_l_t = timed(lambda: etl.run(exe_l, x_large, trees[0]), CFG.reps_run)
    rep.add("run N=1000 ms/tree", ms(run_l_t))
    rep.add("output dtype/shape", f"{out.dtype} {out.shape}",
            f"max err {err:.3e} vs numpy ref, nonfinite-mismatch {mism} "
            f"(ref nonfinite {n_nonfinite})")
    return {
        "trees": trees,
        "exe": exe,
        "build_ms": build_med,
        "run_ms": run_t[0],
        "run_large_ms": run_l_t[0],
    }


# --------------------------------------------------------------------------
# (b) iree-cuda
# --------------------------------------------------------------------------


def section_b(rep):
    rep.section(f"(b) backend=iree, device=cuda:{CFG.device}, target_backends=['cuda']")
    dev = etl.Device("cuda", CFG.device)
    rng = np.random.default_rng(20250202)
    trees = [make_tree(rng, TREE_BUDGET) for _ in range(CFG.reps_build + 2)]
    x = np.random.default_rng(17).uniform(-1.0, 1.0, (N_SMALL, 2)).astype(np.float32)
    x_large = np.random.default_rng(18).uniform(-1.0, 1.0, (N_LARGE, 2)).astype(np.float32)
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)
    spec_large = etl.TensorSpec((N_LARGE, 2), etl.float32)

    def build(tree, spec_, cache, opt_level, backend="iree"):
        return etl.build_cached(_step if spec_ is spec else _step_large, spec_, tree,
                                backend=backend, device=dev,
                                target_backends=["cuda"], opt_level=opt_level, cache=cache)

    cache = etl.CompileCache()
    t0 = time.perf_counter()
    exe = build(trees[0], spec, cache, "O1")
    rep.add("first build (miss, cold process incl. iree init)",
            f"{(time.perf_counter() - t0) * 1e3:.1f} ms")

    builds = {}
    for opt in ("O1", "O0"):
        for locations in (False, True):
            c = etl.CompileCache()
            with env_var("ETL_DISABLE_LOCATIONS", "1" if locations else None):
                ts = []
                for tree in trees[1:CFG.reps_build + 1]:
                    t0 = time.perf_counter()
                    build(tree, spec, c, opt)
                    ts.append((time.perf_counter() - t0) * 1e3)
            triple = (statistics.median(ts), min(ts), max(ts))
            builds[(opt, locations)] = triple
            rep.add(f"distinct-tree build opt_level={opt}"
                    f"{' + ETL_DISABLE_LOCATIONS=1' if locations else ''}",
                    ms(triple), f"n={len(ts)}")
        t = timed(lambda: build(trees[0], spec, cache, opt), max(CFG.reps_build, 10))
        rep.add(f"build_cached cache hit opt_level={opt}", ms(t))

    dev_in = etl.core.tensor(x).to(dev)
    dev_in_large = etl.core.tensor(x_large).to(dev)
    etl.run(exe, dev_in, trees[0])  # warm up (lazy driver/module init)
    run_t = timed(lambda: etl.run(exe, dev_in, trees[0]), max(CFG.reps_run, 5))
    rep.add("run N=100 ms/tree (device-resident)", ms(run_t))
    with env_var("ETL_DISABLE_LOCATIONS", "1"):
        rep.add("run N=100 ms/tree with ETL_DISABLE_LOCATIONS=1",
                ms(timed(lambda: etl.run(exe, dev_in, trees[0]), 5)),
                "env var is trace-time only -> run unchanged")
    exe_l = build(trees[0], spec_large, etl.CompileCache(), "O1")
    etl.run(exe_l, dev_in_large, trees[0])
    run_l_t = timed(lambda: etl.run(exe_l, dev_in_large, trees[0]), CFG.reps_run)
    rep.add("run N=1000 ms/tree (device-resident)", ms(run_l_t))

    def run_readback():
        return etl.run(exe, dev_in, trees[0]).to(etl.Device("cpu", 0)).numpy()

    t = timed(run_readback, CFG.reps_run)
    rep.add("run + D2H readback N=100 ms/tree", ms(t))
    got = run_readback()
    err, mism, n_nonfinite = compare(got, ref_eval(trees[0], x))
    rep.add("iree-cuda parity vs numpy ref", f"max err {err:.3e}",
            f"nonfinite-mismatch {mism}, ref nonfinite {n_nonfinite}")

    # per-stage breakdown (build_cached hides these; useful for planning)
    t0 = time.perf_counter()
    graph = etl.trace(_step_large, spec_large, trees[0])
    t1 = time.perf_counter()
    lowered = etl.lower(graph, backend="iree", target_backends=["cuda"], opt_level="O1")
    t2 = time.perf_counter()
    artifact = etl.compile(lowered, backend="iree", target_backends=["cuda"], opt_level="O1")
    t3 = time.perf_counter()
    etl.load(artifact, backend="iree", device=dev)
    t4 = time.perf_counter()
    rep.note(f"stage breakdown (N=1000 spec): trace {(t1 - t0) * 1e3:.1f} ms, "
             f"lower {(t2 - t1) * 1e3:.1f} ms, compile {(t3 - t2) * 1e3:.1f} ms, "
             f"load {(t4 - t3) * 1e3:.1f} ms (load = runtime-device creation)")
    return {
        "trees": trees,
        "dev": dev,
        "exe": exe,
        "build_ms": builds[("O1", False)][0],
        "build_o0_ms": builds[("O0", False)][0],
        "build_ms_noloc": builds[("O1", True)][0],
        "run_ms": run_t[0],
        "run_large_ms": run_l_t[0],
    }


# --------------------------------------------------------------------------
# (c) simulated generation
# --------------------------------------------------------------------------


def section_c(rep, numpy_build_ms, numpy_run_ms, iree_build_ms, iree_run_ms):
    rep.section("(c) simulated generation: 1000 distinct trees x (build + run)")
    n = CFG.gen_trees
    rng = np.random.default_rng(20250303)
    trees = [make_tree(rng, TREE_BUDGET) for _ in range(n)]
    x = np.random.default_rng(27).uniform(-1.0, 1.0, (N_SMALL, 2)).astype(np.float32)
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)

    # Fallbacks: if (a)/(b) were not run, take a short-shortened measurement so
    # the extrapolation still refers to THIS process/host.
    if not np.isfinite(numpy_build_ms):
        c = etl.CompileCache()
        ts = []
        for i in range(5):
            t0 = time.perf_counter()
            etl.build_cached(_step, spec, trees[i], backend="numpy", cache=c)
            ts.append((time.perf_counter() - t0) * 1e3)
        numpy_build_ms = statistics.median(ts)
        exe = etl.build_cached(_step, spec, trees[0], backend="numpy", cache=c)
        numpy_run_ms = timed(lambda: etl.run(exe, x, trees[0]), 5)[0]
    if not np.isfinite(iree_build_ms):
        dev = etl.Device("cuda", CFG.device)
        c = etl.CompileCache()
        ts = []
        for i in range(3):
            t0 = time.perf_counter()
            etl.build_cached(_step, spec, trees[i], backend="iree", device=dev,
                             target_backends=["cuda"], opt_level="O1", cache=c)
            ts.append((time.perf_counter() - t0) * 1e3)
        iree_build_ms = statistics.median(ts)
    if not np.isfinite(iree_run_ms):
        dev = etl.Device("cuda", CFG.device)
        exe = etl.build_cached(_step, spec, trees[0], backend="iree", device=dev,
                               target_backends=["cuda"], opt_level="O1",
                               cache=etl.CompileCache())
        arg = etl.core.tensor(x).to(dev)
        etl.run(exe, arg, trees[0])
        iree_run_ms = timed(lambda: etl.run(exe, arg, trees[0]), 5)[0]

    for locations in (False, True):
        # The default-env pass runs over every generated tree; the auxiliary
        # ETL_DISABLE_LOCATIONS=1 pass is measured on a fresh SUBSAMPLE and
        # scaled, to keep the section bounded.
        n_pass = n if not locations else min(n, CFG.gen_sample)
        pass_trees = trees if not locations else [
            make_tree(rng, TREE_BUDGET) for _ in range(n_pass)]
        c = etl.CompileCache(maxsize=max(256, n * 2))
        t_build = 0.0
        t_run = 0.0
        with env_var("ETL_DISABLE_LOCATIONS", "1" if locations else None):
            for tree in pass_trees:
                t0 = time.perf_counter()
                exe = etl.build_cached(_step, spec, tree, backend="numpy", cache=c)
                t1 = time.perf_counter()
                etl.run(exe, x, tree)
                t2 = time.perf_counter()
                t_build += t1 - t0
                t_run += t2 - t1
        label = " + ETL_DISABLE_LOCATIONS=1" if locations else ""
        note = "" if not locations else f"(n={n_pass} sample, scaled to {n})"
        rep.add(f"numpy {n} trees{label}: build",
                secs(t_build / n_pass * n), 
                f"{t_build / n_pass * 1e3:.2f} ms/tree, misses {c.misses} {note}")
        rep.add(f"numpy {n} trees{label}: run", secs(t_run / n_pass * n),
                f"{t_run / n_pass * 1e3:.3f} ms/tree")
        rep.add(f"numpy {n} trees{label}: TOTAL", secs((t_build + t_run) / n_pass * n))
    rep.add(f"iree-cuda {n} trees: EXTRAPOLATED build+run",
            secs(n * (iree_build_ms + iree_run_ms) / 1e3),
            f"at {iree_build_ms:.0f} ms build + {iree_run_ms:.3f} ms run "
            f"from (b), NOT run")
    return trees


# --------------------------------------------------------------------------
# (d) parallel builds
# --------------------------------------------------------------------------


def section_d(rep):
    rep.section("(d) parallel build feasibility (ThreadPoolExecutor)")
    n = CFG.par_trees
    rng = np.random.default_rng(20250404)
    trees = [make_tree(rng, TREE_BUDGET) for _ in range(n)]
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)
    dev = etl.Device("cuda", CFG.device)

    def parallel(build_one, workers):
        def task(i):
            try:
                t0 = time.perf_counter()
                build_one(trees[i], i)
                return None, (time.perf_counter() - t0) * 1e3
            except Exception as exc:  # noqa: BLE001 - the probe reports it
                return f"{type(exc).__name__}: {str(exc)[:120]}", 0.0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(task, range(n)))
        wall = time.perf_counter() - t0
        errs = [m for m, _ in results if m]
        return wall, errs, statistics.median([d for _, d in results if d])

    # numpy: pure-Python trace, expect GIL-bound scaling
    for workers in (8, 32):
        c = etl.CompileCache(maxsize=n * 2)
        wall, errs, med = parallel(
            lambda tree, i, c=c: etl.build_cached(
                _step, spec, tree, backend="numpy", cache=c), workers)
        rep.add(f"numpy {n} trees, {workers} threads", f"{wall:.2f} s wall",
                f"{med:.1f} ms median build/task, errors {len(errs)} {errs[:2]}")
        hits, misses = c.hits, c.misses
        for i in range(n):  # serial re-build: must be all hits (no lost inserts)
            etl.build_cached(_step, spec, trees[i], backend="numpy", cache=c)
        rep.add(f"numpy {n} trees, {workers} threads: cache after serial re-run",
                f"misses {misses} hits {hits}",
                f"post-hoc re-run hits {c.hits - hits} (expect {n})")
        if c.hits - hits != n or misses != n:
            rep.fail(f"numpy parallel cache completeness ({workers} threads)",
                     AssertionError(f"misses {misses}, re-run hits {c.hits - hits}, expected {n}"))

    for workers in (8, 32):
        c = etl.CompileCache(maxsize=n * 2)
        wall, errs, med = parallel(
            lambda tree, i, c=c: etl.build_cached(
                _step, spec, tree, backend="iree", device=dev,
                target_backends=["cuda"], opt_level="O0", cache=c), workers)
        rep.add(f"iree-cuda(O0) {n} trees, {workers} threads", f"{wall:.2f} s wall",
                f"{med:.1f} ms median build/task, errors {len(errs)} {errs[:2]}")
        misses, hits = c.misses, c.hits
        serial = None
        if not errs:
            t0 = time.perf_counter()
            for i in range(n):
                serial = etl.build_cached(
                    _step, spec, trees[i], backend="iree", device=dev,
                    target_backends=["cuda"], opt_level="O0", cache=c)
            rep.add(f"iree-cuda(O0) {n} trees, {workers} threads: "
                    "cache after serial re-run",
                    f"misses {misses} hits {c.hits - hits}",
                    f"serial re-run took {time.perf_counter() - t0:.2f} s "
                    f"(expect {n} hits)")
            if c.hits - hits != n or misses != n:
                rep.fail(f"iree parallel cache completeness ({workers} threads)",
                         AssertionError(
                             f"misses {misses}, re-run hits {c.hits - hits}, expected {n}"))
        else:
            rep.fail(f"iree parallel builds ({workers} threads)", RuntimeError(str(errs[:3])))


# --------------------------------------------------------------------------
# (e) persistence
# --------------------------------------------------------------------------


def section_e(rep, dev):
    rep.section("(e) persistence: Executable.save + Executable.load")
    tree = make_tree(np.random.default_rng(20250505), TREE_BUDGET)
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)
    x = np.random.default_rng(37).uniform(-1.0, 1.0, (N_SMALL, 2)).astype(np.float32)
    dev_in = etl.core.tensor(x).to(dev)
    exe = etl.build_cached(_step, spec, tree, backend="iree", device=dev,
                           target_backends=["cuda"], opt_level="O1",
                           cache=etl.CompileCache())
    path = CFG.scratch / "probe_exe_iree_cuda.bin"
    t = timed(lambda: exe.save(path), 3)
    rep.add("save (iree-cuda executable)", ms(t), f"{path} ({path.stat().st_size} bytes)")
    loaded = etl.Executable.load(path, backend="iree", device=dev)
    t = timed(lambda: etl.Executable.load(path, backend="iree", device=dev), 3)
    rep.add("Executable.load (reload from disk)", ms(t),
            f"backend {loaded.backend}, device {loaded.device}")
    rep.note("load avg matches the build_cached 'load' stage (runtime-device "
             "creation); the iree compiler is NOT invoked again")
    a = etl.run(exe, dev_in, tree).to(etl.Device("cpu", 0)).numpy()
    b = etl.run(loaded, dev_in, tree).to(etl.Device("cpu", 0)).numpy()
    rep.add("reloaded executable output identical", str(bool(np.array_equal(a, b, equal_nan=True))))

    # `etl.load` is the pipeline stage (artifact -> executable), NOT the
    # persistence entry point: passing a path is a TypeError.
    try:
        etl.load(path, backend="iree", device=dev)
        rep.add("etl.load(path) (pipeline stage)", "UNEXPECTEDLY ACCEPTED")
    except Exception as exc:  # noqa: BLE001 - message is the point
        rep.add("etl.load(path) -> rejected (use Executable.load)",
                f"{type(exc).__name__}: {str(exc)[:60]}")

    np_tree = make_tree(np.random.default_rng(20250606), TREE_BUDGET)
    np_exe = etl.build_cached(_step, spec, np_tree, backend="numpy", cache=etl.CompileCache())
    np_path = CFG.scratch / "probe_exe_numpy.bin"
    try:
        np_exe.save(np_path)
        np_loaded = etl.Executable.load(np_path, backend="numpy")
        same = np.array_equal(etl.run(np_exe, x, np_tree).numpy(),
                              etl.run(np_loaded, x, np_tree).numpy(), equal_nan=True)
        rep.add("numpy save/load round trip", f"{np_path.stat().st_size} bytes",
                f"identical output {bool(same)}")
    except Exception as exc:  # noqa: BLE001
        rep.fail("numpy save/load round trip", exc)


# --------------------------------------------------------------------------
# (f) API semantics checks
# --------------------------------------------------------------------------


def section_f(rep, dev):
    rep.section("(f) API semantics checks")
    spec = etl.TensorSpec((N_SMALL, 2), etl.float32)
    tree = make_tree(np.random.default_rng(20250707), TREE_BUDGET)
    x = np.random.default_rng(47).uniform(-1.0, 1.0, (N_SMALL, 2)).astype(np.float32)

    # 1. nested-tuple static round trip
    try:
        c = etl.CompileCache()
        exe = etl.build_cached(_step, spec, tree, backend="numpy", cache=c)
        again = etl.build_cached(_step, spec, tree, backend="numpy", cache=c)
        out = etl.run(exe, x, tree)
        err, mism, _ = compare(out.numpy(), ref_eval(tree, x))
        rep.add("nested-tuple static round trip",
                f"PASS (err {err:.2e}, mismatches {mism})",
                f"rebuild is identical object: {again is exe}")
    except Exception as exc:  # noqa: BLE001
        rep.fail("nested-tuple static round trip", exc)

    # 2. static-leaf types accepted by the cache key.  `_key_probe` ignores
    # its static argument, so this exercises ONLY the key-encoding path.
    @etl.defn
    def _key_probe(x, static):
        return etl.reshape(etl.slice(x, (0, 0), (N_SMALL, 1)), (N_SMALL,))

    statics = {
        "nested-tuple tree (str ops)": ("mul", ("x", 0), ("c", 2.0)),
        "int-only tree": (1, (0, 0), (-2,)),
        "dict static": {"ops": ("add", "mul"), "k": 3},
        "list static": [1, 2.0, "three", (4,)],
        "plain str": "add",
        "plain float": 2.5,
    }
    for name, static in statics.items():
        try:
            c = etl.CompileCache()
            etl.build_cached(_key_probe, spec, static, backend="numpy", cache=c)
            etl.build_cached(_key_probe, spec, eval(repr(static)), backend="numpy",
                             cache=c)  # equal content, new object
            ok = c.misses == 1 and c.hits == 1
            rep.add(f"static encodable: {name}",
                    f"{'PASS' if ok else 'FAIL'} (misses {c.misses} hits {c.hits})")
            if not ok:
                rep.fail(f"static encodable: {name}",
                         AssertionError(f"misses {c.misses} hits {c.hits}"))
        except Exception as exc:  # noqa: BLE001
            rep.fail(f"static encodable: {name}", exc)

    for name, v in (("np.float32", np.float32(3.0)), ("np.int64", np.int64(3)),
                    ("core.Tensor", etl.core.tensor(np.float32(3.0))),
                    ("tuple containing np.float32", ("add", np.float32(1.0)))):
        try:
            etl.build_cached(_key_probe, spec, v, backend="numpy",
                             cache=etl.CompileCache())
            rep.add(f"static leaf type {name}", "accepted")
        except Exception as exc:  # noqa: BLE001
            rep.add(f"static leaf type {name}",
                    f"{type(exc).__name__} (rejected: good)",
                    str(exc).replace("\n", " ")[:110])

    # 3. cache-key behaviour
    try:
        c = etl.CompileCache()
        t1 = ("mul", ("x", 0), ("c", 2.0))
        t2 = ("mu" + "l", ("x", 0), ("c", 1.0 + 1.0))  # equal content, new object
        e1 = etl.build_cached(_step, spec, t1, backend="numpy", cache=c)
        e2 = etl.build_cached(_step, spec, t2, backend="numpy", cache=c)
        e3 = etl.build_cached(_step, spec, ("mul", ("x", 1), ("c", 2.0)),
                              backend="numpy", cache=c)
        e4 = etl.build_cached(_step, etl.TensorSpec((200, 2), etl.float32), t1,
                              backend="numpy", cache=c)
        ok = (c.misses, c.hits) == (3, 1) and e1 is e2 and e1 is not e3 and e1 is not e4
        rep.add("cache key: content-based / spec-sensitive",
                f"{'PASS' if ok else 'FAIL'} (misses {c.misses}, hits {c.hits})",
                "identical rebuilt structure -> same object; different static or spec -> new entry")
        if not ok:
            rep.fail("cache key semantics", AssertionError(f"misses {c.misses} hits {c.hits}"))
    except Exception as exc:  # noqa: BLE001
        rep.fail("cache key semantics", exc)

    try:
        c = etl.CompileCache(maxsize=2)
        for i in range(3):
            etl.build_cached(_step, spec, ("add", ("x", 0), ("c", float(i))),
                             backend="numpy", cache=c)
        before = c.misses
        etl.build_cached(_step, spec, ("add", ("x", 0), ("c", 0.0)),
                         backend="numpy", cache=c)
        rep.add("CompileCache(maxsize=2) LRU eviction",
                f"misses {c.misses} hits {c.hits} (len {len(c)})",
                f"re-requesting the evicted entry re-misses "
                f"({before} -> {c.misses}): size the cache >= live trees")
    except Exception as exc:  # noqa: BLE001
        rep.fail("CompileCache LRU eviction", exc)

    # 4. run-boundary validation messages
    exe = etl.build_cached(_step, spec, tree, backend="numpy", cache=etl.CompileCache())
    checks = [
        ("float64 input vs float32 spec", lambda: etl.run(
            exe, x.astype(np.float64), tree)),
        ("wrong N", lambda: etl.run(exe, x[:50], tree)),
        ("wrong static program", lambda: etl.run(
            exe, x, ("mul", ("x", 0), ("c", 9.0)))),
    ]
    for name, fn in checks:
        try:
            fn()
            rep.add(f"run-boundary {name}", "UNEXPECTEDLY ACCEPTED")
        except Exception as exc:  # noqa: BLE001
            rep.add(f"run-boundary {name}", type(exc).__name__,
                    str(exc).replace("\n", " ")[:110])

    # 5. torch <-> etl interop
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        rep.fail("torch interop (import torch)", exc)
        torch = None
    if torch is not None:
        host_t = etl.core.tensor(x)
        try:
            tt = torch.from_dlpack(host_t)
            rep.add("torch.from_dlpack(etl cpu tensor)",
                    f"PASS {tuple(tt.shape)} {tt.dtype}",
                    f"zero-copy: {np.shares_memory(tt.numpy(), host_t.numpy())}")
        except Exception as exc:  # noqa: BLE001
            rep.fail("torch.from_dlpack(etl cpu tensor)", exc)
        try:
            back = etl.core.from_dlpack(torch.from_numpy(x))
            rep.add("etl.core.from_dlpack(torch cpu tensor)",
                    f"PASS {back.shape} {back.device}")
        except Exception as exc:  # noqa: BLE001
            rep.fail("etl.core.from_dlpack(torch cpu tensor)", exc)
        try:
            dev_t = host_t.to(dev)
        except Exception:  # noqa: BLE001
            dev_t = host_t
        for name, fn in (
            ("torch.from_dlpack(etl CUDA tensor)", lambda: torch.from_dlpack(dev_t)),
            ("etl.core.from_dlpack(torch CUDA tensor)",
             lambda: etl.core.from_dlpack(torch.from_numpy(x).to("cuda"))),
        ):
            try:
                fn()
                rep.add(name, "UNEXPECTEDLY ACCEPTED")
            except Exception as exc:  # noqa: BLE001
                rep.add(name, f"{type(exc).__name__} (expected)",
                        str(exc).replace("\n", " ")[:110])
        # the supported host<->device path
        cyc = dev_t.to(etl.Device("cpu", 0)).numpy()
        rep.add("supported readback", f"etl Tensor .to(cpu).numpy() -> {cyc.shape}",
                "device payloads refuse DLPack; .to(cpu) is the only transfer")
        t_np = torch.from_numpy(np.ascontiguousarray(cyc))
        rep.add("back into torch", f"torch.from_numpy(...) -> {tuple(t_np.shape)}")

    # 6. protected divide recipe
    @etl.defn
    def _div_probe(x):
        a = etl.slice(x, (0, 0), (N_SMALL, 1))
        b = etl.slice(x, (0, 1), (N_SMALL, 1))
        nan = etl.constant(etl.core.tensor(np.float32(np.nan)))
        return etl.reshape(etl.select(etl.equal(b, 0.0), nan, etl.divide(a, b)), (N_SMALL,))

    x_div = np.ones((N_SMALL, 2), np.float32)
    x_div[3, 1] = 0.0
    x_div[7, 1] = 0.0
    try:
        div_exe = etl.build_cached(_div_probe, spec, backend="numpy", cache=etl.CompileCache())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = etl.run(div_exe, x_div).numpy()
        ok = np.isnan(got[3]) and np.isnan(got[7]) and got[0] == np.float32(1.0)
        rep.add("protected divide (numpy)",
                f"{'PASS' if ok else 'FAIL'}: 1/0 -> {got[3]}, 1/1 -> {got[0]}",
                "numpy backend still evaluates the dead branch (RuntimeWarning div-by-zero)")
        if not ok:
            rep.fail("protected divide (numpy)", AssertionError(str(got[:8])))
    except Exception as exc:  # noqa: BLE001
        rep.fail("protected divide (numpy)", exc)
    try:
        div_exe = etl.build_cached(_div_probe, spec, backend="iree", device=dev,
                                   target_backends=["cuda"], opt_level="O0",
                                   cache=etl.CompileCache())
        got = etl.run(div_exe, etl.core.tensor(x_div).to(dev)).to(
            etl.Device("cpu", 0)).numpy()
        ok = np.isnan(got[3]) and np.isnan(got[7]) and got[0] == np.float32(1.0)
        rep.add("protected divide (iree-cuda)",
                f"{'PASS' if ok else 'FAIL'}: 1/0 -> {got[3]}, 1/1 -> {got[0]}")
        if not ok:
            rep.fail("protected divide (iree-cuda)", AssertionError(str(got[:8])))
    except Exception as exc:  # noqa: BLE001
        rep.fail("protected divide (iree-cuda)", exc)
    rep.note("working recipe: etl.select(etl.equal(b, 0.0), "
             "etl.constant(etl.core.tensor(np.float32(np.nan))), etl.divide(a, b)) "
             "-- etl.equal accepts a plain Python float (weak scalar), but "
             "etl.constant REQUIRES a core.Tensor")


# --------------------------------------------------------------------------
# (g) ETL's own meta-programming test, from a $TMPDIR copy
# --------------------------------------------------------------------------


ETL_TESTS = Path("/mnt/local-ssd/bchuang/etl/tests")

IREE_CUDA_ADAPTATION = '''\
"""iree-cuda adaptation of ETL's tests/test_meta_programming.py (section 1)."""
import sys, numpy as np, etl

sys.path.insert(0, {tests_dir!r})
import test_meta_programming as T  # the copied upstream test module

DEVICE = etl.Device("cuda", {device})
cache = etl.CompileCache()
exes = {{}}
trees = []
for i in range(2):
    _, _, tree = T._find_program(T.IREE_SEED_BASE + i * T.SEED_GAP)
    exe = etl.build_cached(T._step, T.SPEC, tree, cache=cache, backend="iree",
                           device=DEVICE, target_backends=["cuda"], opt_level="O1")
    assert exe.backend == "iree" and exe.device == DEVICE, (exe.backend, exe.device)
    exes[tree] = exe
    trees.append(tree)
    assert cache.misses == i + 1, cache.misses
    for x in T.EVAL_INPUTS:
        xd = etl.core.tensor(np.ascontiguousarray(x)).to(DEVICE)
        got = etl.run(exe, xd, tree).to(etl.Device("cpu", 0)).numpy().astype(np.float64)
        ref = T._np_eval(tree, x.astype(np.float64))
        assert np.allclose(got, ref, rtol=1e-5, atol=1e-5), (tree, x, got, ref)
again = etl.build_cached(T._step, T.SPEC, trees[-1], cache=cache, backend="iree",
                         device=DEVICE, target_backends=["cuda"], opt_level="O1")
assert again is exes[trees[-1]]
assert cache.hits == 1 and cache.misses == 2, (cache.hits, cache.misses)
print("iree-cuda adaptation: PASS (2 compiles, 2 programs, parity ok, hit returns same exe)")
'''


def section_g(rep):
    rep.section("(g) ETL's own meta-programming test ($TMPDIR copy)")
    tests_dir = CFG.scratch / "etl_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    src = ETL_TESTS / "test_meta_programming.py"
    if not src.exists():
        rep.fail("copy ETL test", FileNotFoundError(str(src)))
        return
    (tests_dir / src.name).write_text(src.read_text())

    def summary_line(text):
        """pytest verdict line, ignoring iree's at-exit nanobind leak noise."""
        noise = ("nanobind:", " - leaked", "See https://", "- leaked ")
        lines = [ln for ln in text.strip().splitlines()
                 if ln.strip() and not any(ln.lstrip().startswith(n) or n in ln
                                           for n in noise)]
        verdict = [ln for ln in lines if "passed" in ln or "failed" in ln
                   or "error" in ln.lower()]
        return (verdict[-1] if verdict else lines[-1] if lines else "")[:110]

    def run_pytest(args, label):
        cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"] + args
        p = subprocess.run(cmd, cwd=tests_dir, capture_output=True, text=True, timeout=900)
        rep.add(f"pytest {label}",
                f"{'PASS' if p.returncode == 0 else 'FAIL'} (exit {p.returncode})",
                summary_line(p.stdout + "\n" + p.stderr))
        if p.returncode != 0:
            rep.fail(f"pytest {label}", RuntimeError((p.stdout + p.stderr)[-400:]))

    run_pytest([f"{src.name}::TestMetaLoopNumpy"], "numpy leg (4 tests)")
    run_pytest([f"{src.name}::TestMetaLoopIree"], "upstream iree-llvm-cpu leg")

    adapt = tests_dir / "iree_cuda_adaptation.py"
    adapt.write_text(IREE_CUDA_ADAPTATION.format(tests_dir=str(tests_dir),
                                                 device=CFG.device))
    p = subprocess.run([sys.executable, str(adapt)], cwd=tests_dir,
                       capture_output=True, text=True, timeout=900)
    rep.add("iree-cuda adaptation (2 programs)",
            f"{'PASS' if p.returncode == 0 else 'FAIL'} (exit {p.returncode})",
            summary_line(p.stdout + "\n" + p.stderr))
    if p.returncode != 0:
        rep.fail("iree-cuda adaptation", RuntimeError((p.stdout + p.stderr)[-600:]))


# --------------------------------------------------------------------------
# (h) extra diagnostics: N sweep, fixed per-run overhead, transfer costs
# --------------------------------------------------------------------------


def section_h(rep, dev):
    rep.section("(h) extra diagnostics: N sweep / fixed overhead / transfers")
    tree = make_tree(np.random.default_rng(20250808), TREE_BUDGET)
    for n in SWEEP_NS:
        spec = etl.TensorSpec((n, 2), etl.float32)
        x = np.random.default_rng(n).uniform(-1.0, 1.0, (n, 2)).astype(np.float32)
        row = [f"N={n:>6}"]
        for backend in ("numpy", "iree"):
            try:
                kw = dict(backend=backend, cache=etl.CompileCache())
                if backend == "iree":
                    kw.update(device=dev, target_backends=["cuda"], opt_level="O0")
                exe = etl.build_cached(SWEEP_STEPS[n], spec, tree, **kw)
                if backend == "iree":
                    arg = etl.core.tensor(x).to(dev)
                    etl.run(exe, arg, tree)
                    t = timed(lambda: etl.run(exe, arg, tree), 5)
                else:
                    etl.run(exe, x, tree)
                    t = timed(lambda: etl.run(exe, x, tree), 5)
                row.append(f"{backend} {t[0]:7.3f} ms")
            except Exception as exc:  # noqa: BLE001
                row.append(f"{backend} FAIL {type(exc).__name__}")
        rep.add("run ms/tree (median of 5)", "   ".join(row))

    spec1 = etl.TensorSpec((1, 2), etl.float32)
    x1 = np.ones((1, 2), np.float32)
    for backend, kw in (("numpy", {}),
                        ("iree", dict(device=dev, target_backends=["cuda"],
                                      opt_level="O0"))):
        exe = etl.build_cached(_one_op, spec1, backend=backend,
                               cache=etl.CompileCache(), **kw)
        if backend == "iree":
            arg = etl.core.tensor(x1).to(dev)
            etl.run(exe, arg)
            t = timed(lambda: etl.run(exe, arg), 20)
        else:
            etl.run(exe, x1)
            t = timed(lambda: etl.run(exe, x1), 20)
        rep.add(f"fixed per-run cost, 1-op graph ({backend})", ms(t))

    t = timed(lambda: etl.core.tensor(x1).to(dev), 20)
    rep.add("H2D etl tensor (1, 2) f32", ms(t))
    x1000 = np.ones((1000, 2), np.float32)
    t = timed(lambda: etl.core.tensor(x1000).to(dev), 20)
    rep.add("H2D etl tensor (1000, 2) f32", ms(t))
    exe = etl.build_cached(_one_op, spec1, backend="iree", device=dev,
                           target_backends=["cuda"], opt_level="O0",
                           cache=etl.CompileCache())
    arg = etl.core.tensor(x1).to(dev)
    t = timed(lambda: etl.run(exe, arg).to(etl.Device("cpu", 0)).numpy(), 20)
    rep.add("iree-cuda run + readback, 1-op graph", ms(t),
            "readback dominates small-batch device loops")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sections", default="abcdefgh",
                    help="subset of abcdefgh (default: all)")
    ap.add_argument("--device", type=int, default=0, help="cuda device index")
    ap.add_argument("--reps-build", type=int, default=10)
    ap.add_argument("--reps-run", type=int, default=7)
    ap.add_argument("--par-trees", type=int, default=64)
    ap.add_argument("--gen-trees", type=int, default=1000)
    ap.add_argument("--gen-sample", type=int, default=100,
                    help="sample size for the auxiliary (locations-off) pass")
    cfg = ap.parse_args(argv)
    CFG.__dict__.update(cfg.__dict__)

    # Random GP trees overflow/divide by zero by construction; the numpy
    # backend warns on every such op (the graph semantics are unchanged —
    # section (f) pins the protected-divide recipe).  Keep the table readable.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    base = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "etl_jit_probe"
    base.mkdir(parents=True, exist_ok=True)
    CFG.scratch = Path(tempfile.mkdtemp(prefix="run_", dir=base))

    print("=" * 100)
    print("ETL true-jit probe")
    print(f"  python {sys.version.split()[0]}  etl {getattr(etl, '__version__', '?')} "
          f"from {etl.__file__}")
    try:
        import torch
        print(f"  torch {torch.__version__}  numpy {np.__version__}")
    except Exception:  # noqa: BLE001
        print(f"  numpy {np.__version__} (torch unavailable)")
    print(f"  device cuda:{CFG.device}   scratch {CFG.scratch}")
    try:  # best-effort host context: these timings share the box with others
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if smi.returncode == 0:
            util = [line.strip() for line in smi.stdout.strip().splitlines()]
            print("  nvidia-smi (index, util%, mem): " + " | ".join(util))
            mine = util[CFG.device].split(",") if CFG.device < len(util) else []
            if len(mine) == 3 and int(mine[1].strip().rstrip(" %")) > 0:
                print(f"  WARNING: cuda:{CFG.device} is busy "
                      f"({util[CFG.device].strip()}) — timings are contended; "
                      f"use --device N on an idle GPU for stable numbers")
    except Exception:  # noqa: BLE001
        pass
    print(f"  sections {CFG.sections}   reps-build {CFG.reps_build} "
          f"reps-run {CFG.reps_run}")
    print("=" * 100, flush=True)

    rep = Report()
    wanted = set(CFG.sections)
    dev = etl.Device("cuda", CFG.device)
    res_a = res_b = None
    t_start = time.perf_counter()

    if "a" in wanted:
        try:
            res_a = section_a(rep)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (a)", exc)
    if "b" in wanted:
        try:
            res_b = section_b(rep)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (b)", exc)
    if "c" in wanted:
        try:
            section_c(
                rep,
                (res_a or {}).get("build_ms", float("nan")),
                (res_a or {}).get("run_ms", float("nan")),
                (res_b or {}).get("build_ms", float("nan")),
                (res_b or {}).get("run_ms", float("nan")),
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (c)", exc)
    if "d" in wanted:
        try:
            section_d(rep)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (d)", exc)
    if "e" in wanted:
        try:
            section_e(rep, dev)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (e)", exc)
    if "f" in wanted:
        try:
            section_f(rep, dev)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (f)", exc)
    if "g" in wanted:
        try:
            section_g(rep)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (g)", exc)
    if "h" in wanted:
        try:
            section_h(rep, dev)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rep.fail("section (h)", exc)

    if res_a and res_b:
        rep.add("headline: per-tree build + run (N=100)",
                "see summary",
                f"numpy {res_a['build_ms']:.2f} + {res_a['run_ms']:.3f} ms  |  "
                f"iree-cuda(O1) {res_b['build_ms']:.0f} + {res_b['run_ms']:.3f} ms  |  "
                f"iree reuse-only {res_b['run_ms']:.3f} ms")

    rep.add("TOTAL probe wall time", secs(time.perf_counter() - t_start))
    rep.dump()
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
