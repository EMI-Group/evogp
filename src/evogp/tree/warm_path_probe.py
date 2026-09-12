#!/usr/bin/env python
"""``warm_path_probe.py`` -- dev CLI: per-stage warm-path timing for the specialized engine.

Small, self-contained measurement tool for the *host-side* overhead of the
specialized (per-structure ``etl.build_cached``) route in :mod:`evogp.tree.etl_engine`
(``_run_tree`` / ``_executable`` / ``decode_population`` / ``evaluate`` /
``sr_fitness``).  It is **NOT** pytest-collected, is imported by nothing, and is
read-only with respect to the library: it times the public entry points and the
private helpers as they are.

Why it exists: ``test/benchmark/stage_probe.py`` decomposes a whole
``problem.evaluate`` call but only on the GPU benchmark box through the frozen
``bench_gp.py`` harness.  This probe answers the same questions on any box (CPU
included, numpy backend) while iterating, and adds the A/B numbers for the
host-side levers (input conversion, host loss reduction, decode folding).

Usage (from the repository root, project venv)::

    PYTHONPATH=src python src/evogp/tree/warm_path_probe.py \
        --fixture test/goldens/forest_random.npz --reps 25
    ... --fixture test/benchmark/fixtures/forest200_tree64_random.npz --rows 10000
    ... --device cuda --backend iree --profile    # needs a CUDA-visible device

Stages (medians, ms): ``decode`` (``decode_population``), per-tree ``run_tree``
(one ``_run_tree`` on a decoded tree), ``build`` (the ``_executable`` lookup) and
``evaluate`` (whole-population, warm) are timed separately; ``sr_fitness`` is
timed as the public ``Forest.SR_fitness``.  ``--profile`` adds a cProfile dump so
the residual can be attributed inside ETL as well.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time

import numpy as np
import torch


def _load_fixture(path: str) -> dict:
    with np.load(path) as data:
        keys = set(data.files)
        if "forest__node_value" not in keys:
            raise SystemExit(f"{path}: not a forest fixture (keys: {sorted(keys)})")
        x_key = "x" if "x" in keys else "inputs"
        return {
            "node_value": torch.as_tensor(data["forest__node_value"]),
            "node_type": torch.as_tensor(data["forest__node_type"]),
            "subtree_size": torch.as_tensor(data["forest__subtree_size"]),
            "x": torch.as_tensor(data[x_key]),
            "labels": torch.as_tensor(data["labels"]),
            "input_len": int(torch.as_tensor(data[x_key]).shape[1]),
        }


def _median(seconds: list) -> float:
    return statistics.median(seconds) * 1e3 if seconds else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default="test/goldens/forest_random.npz")
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--device", default="cpu", help="torch device string for x/labels")
    parser.add_argument("--rows", type=int, default=0, help="tile the fixture rows up to N")
    parser.add_argument("--profile", action="store_true", help="cProfile the warm loop")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    from evogp.tree import Forest, etl_engine

    fixture = _load_fixture(args.fixture)
    input_len = fixture["input_len"]
    output_len = int(fixture["labels"].shape[1])
    # ``Forest`` relocates its tensors through ``check_tensor`` (hard-coded "cuda"),
    # so the population tensors live on the device in the real flow; the decode
    # stage is timed separately from the *fixture* arrays (as a CUDA flow would
    # hand them over for a host-side readback).
    forest = Forest(
        input_len,
        output_len,
        fixture["node_value"].to(args.device),
        fixture["node_type"].to(args.device),
        fixture["subtree_size"].to(args.device),
    )
    x = fixture["x"].to(args.device)
    labels = fixture["labels"].to(args.device)
    if args.rows and args.rows > int(x.shape[0]):
        repeat = int(np.ceil(args.rows / int(x.shape[0])))
        x = x.repeat((repeat, 1))[: args.rows].contiguous()
        labels = labels.repeat((repeat, 1))[: args.rows].contiguous()

    def timed(fn, reps) -> list:
        out = []
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            out.append(time.perf_counter() - start)
        return out

    # -- warm the compile cache once ----------------------------------------
    forest.SR_fitness(x, labels, True, "auto")

    decode_times = timed(
        lambda: etl_engine.decode_population(
            fixture["node_value"], fixture["node_type"], fixture["subtree_size"],
            input_len, output_len,
        ),
        args.reps,
    )
    pop = etl_engine.decode_population(
        fixture["node_value"], fixture["node_type"], fixture["subtree_size"],
        input_len, output_len,
    )
    evaluate_times = timed(lambda: etl_engine.evaluate(pop, x, per_row=False), args.reps)
    fitness_times = timed(lambda: forest.SR_fitness(x, labels, True, "auto"), args.reps)

    backend = etl_engine._resolve_backend(None, None)
    xt = np.ascontiguousarray(x.detach().cpu().numpy())
    n_rows = int(xt.shape[0])
    live = next(tree for tree in pop.trees if tree.const_out is None)
    run_times = timed(
        lambda: etl_engine._run_tree(
            live, xt, 0, n_rows, input_len, backend[0], backend[1], backend[2], {}
        ),
        max(20, args.reps),
    )
    build_times = timed(
        lambda: etl_engine._executable(
            live.program, n_rows, input_len, backend[0], backend[1], backend[2]
        ),
        max(20, args.reps),
    )

    invoked = sum(1 for tree in pop.trees if tree.const_out is None)
    summary = {
        "fixture": args.fixture,
        "device": args.device,
        "backend": backend[0],
        "rows": n_rows,
        "pop_size": forest.pop_size,
        "max_tree_len": forest.max_tree_len,
        "invoked_trees": invoked,
        "decode_ms": _median(decode_times),
        "evaluate_ms": _median(evaluate_times),
        "sr_fitness_ms": _median(fitness_times),
        "run_tree_ms": _median(run_times),
        "executable_ms": _median(build_times),
        "decode_us_per_tree": _median(decode_times) / max(invoked, 1) * 1e3,
        "run_tree_us": _median(run_times) * 1e3,
    }
    print(
        f"{args.fixture} [{backend[0]} / {args.device}] pop={forest.pop_size} "
        f"tree={forest.max_tree_len} rows={n_rows} invoked={invoked}"
    )
    for key in ("decode_ms", "evaluate_ms", "sr_fitness_ms", "run_tree_ms", "executable_ms"):
        print(f"  {key:16s} {summary[key]:9.3f} ms")
    print(f"  decode/tree      {summary['decode_us_per_tree']:9.2f} us")
    print(f"  run_tree         {summary['run_tree_us']:9.2f} us")

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.reps):
            forest.SR_fitness(x, labels, True, "auto")
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(args.top)
        sys.stderr.write("---- cumulative ----\n")
        stats.sort_stats("cumulative").print_stats(args.top)

    if args.json_out:
        import json

        with open(args.json_out, "w") as handle:
            json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
