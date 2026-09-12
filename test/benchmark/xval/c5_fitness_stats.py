#!/usr/bin/env python
"""C5 fitness sanity for the cross-validation run (xval-only helper).

``test/benchmark/bench_gp.py`` times ``problem.evaluate(forest)`` but does not
record the returned values, so the wave-1 ``C5`` reference numbers
``fitness_mean`` / ``nan_count`` (produced by ``legacy/run_matrix.py``) cannot be
compared from the harness JSON alone.

This script re-runs the *same* C5 workload by importing the harness itself
(``build_eval_case`` / ``to_cuda`` / ``apply_env`` are reused verbatim, so the
problem/forest construction is identical) and applies the wave-1 fitness-sanity
semantics: on the last call, ``nan_count = isnan(fitness).sum()`` and
``fitness_mean = nanmean(fitness)`` of the MAXIMIZED fitness tensor cast to
float64 on the CPU.

Usage (same interpreter/PYTHONPATH as the harness run it documents)::

    PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils \
        python -u test/benchmark/xval/c5_fitness_stats.py \
        --fixture <abs>/test/goldens/forest_random.npz --eval-reps 25 \
        --json-out test/benchmark/xval/C5_xval_fitness.json

It prints the side (via the harness' ``detect_side``) and the stats; the JSON
file is a small standalone record, separate from ``C5_xval.json``.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "bench_gp.py")


def load_harness():
    spec = importlib.util.spec_from_file_location("bench_gp", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--eval-reps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    bench_gp = load_harness()
    # Same CLI namespace the harness would build for C5, so build_eval_case
    # sees exactly the flags of the timed run.
    hargs = bench_gp.parse_args(
        ["--task", "eval64", "--eval-reps", str(args.eval_reps), "--seed", str(args.seed),
         "--fixture", args.fixture, "--tag", "C5_xval_fitness"]
    )
    bench_gp.apply_env(hargs)

    mods = bench_gp.import_modules()
    side = bench_gp.detect_side()
    torch = __import__("torch")

    torch.manual_seed(hargs.seed)
    torch.cuda.manual_seed(hargs.seed)
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    problem, forest, eval_kind, description = bench_gp.build_eval_case(mods, hargs)
    torch.cuda.synchronize()
    construct_s = time.perf_counter() - wall_t0

    per_call_ms = []
    fitness = None
    for _ in range(hargs.eval_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fitness = problem.evaluate(forest)
        torch.cuda.synchronize()
        per_call_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - wall_t0

    with torch.no_grad():
        host = fitness.detach().to("cpu", dtype=torch.float64)
    nan_count = int(torch.isnan(host).sum().item())
    fitness_mean = float(torch.nanmean(host).item())

    record = {
        "tag": "C5_xval_fitness",
        "side": side,
        "task": "eval64",
        "eval_kind": eval_kind,
        "fixture": os.path.abspath(args.fixture),
        "evogp_file": getattr(mods.evogp, "__file__", None),
        "problem": description,
        "pop": int(forest.pop_size),
        "tree_len": int(forest.max_tree_len),
        "eval_reps": hargs.eval_reps,
        "seed": hargs.seed,
        "construct_s": construct_s,
        "wall_s": wall_s,
        "first_call_ms": per_call_ms[0] if per_call_ms else None,
        "median_warm_ms": statistics.median(per_call_ms[1:]) if len(per_call_ms) > 1 else None,
        "fitness_shape": list(fitness.shape),
        "nan_count": nan_count,
        "fitness_mean": fitness_mean,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "note": "wave-1 fitness-sanity semantics: last call, MAXIMIZED fitness, "
                "nan_count=isnan(fitness).sum(), fitness_mean=nanmean(fitness) in float64 on CPU",
    }
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2), flush=True)
    print(
        f"[C5_xval_fitness] side={side} first={record['first_call_ms']:.3f}ms "
        f"warm-median={record['median_warm_ms']:.3f}ms wall={wall_s:.4f}s "
        f"nan_count={nan_count} fitness_mean={fitness_mean!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
