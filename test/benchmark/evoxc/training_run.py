"""End-to-end training-run measurement: one complete GP run per arm, timed per generation.

What this measures
------------------
A real ``StandardPipeline`` / ``GeneticProgramming`` training run on a synthetic
symbolic-regression problem that needs NO network and NO dataset: the problem is
``src/evogp/sr_test.py``'s target, produced in-process by
``evogp.problem.SymbolicRegression(func=..., num_inputs=..., num_data=...)``.
Every generation is a sync-bracketed timing sample and the full best-fitness
trajectory is recorded next to it, because a faster arm that converges worse is
NOT a win -- the two columns are meant to be read together.

Arms (``--arm``)
----------------
``engine`` (default)
    the shipped pipeline: ``StandardPipeline.step()`` -> ``SymbolicRegression.evaluate``
    -> ``Forest.SR_fitness`` on whatever route ``EVOGP_EVAL_MODE`` /
    ``EVOGP_EVOXC_BACKEND`` resolve to (recorded in ``record["route"]``).
``torch-per-tree``
    the SAME GP loop, operators and pipeline, with ``arms_torch.PerTreeEvaluator``
    (the honest "old torch version": a Python tree walk applying torch ops per
    node) rebuilt per generation and substituted as the problem's fitness
    (``arms.SRLossProblem``).  Only the evaluator changes.
``preopt``
    identical to ``engine``; it is the label for a run against a frozen
    pre-optimization snapshot.  The snapshot is the CALLER's choice -- pass
    ``--tree-root`` (e.g. ``--tree-root /tmp/evogp_preopt``), nothing about that
    path is hard-coded here.

The library's own loop
----------------------
``StandardPipeline.run()`` is literally ``while ...: self.step()`` plus the
stopping rules, so this script drives ``pipeline.step()`` itself and puts a
``torch.cuda.synchronize()``-bracketed clock around each call.  The fitness
callable is therefore invoked exactly as the library invokes it -- no benchmark
loop, no private call path, no monkeypatching.

Seeding: ``torch.manual_seed(seed)`` + ``torch.cuda.manual_seed(seed)`` are
applied once, identically, before the problem data and the initial population are
drawn, so the three arms start from the same data and the same trees and consume
the same operator RNG stream.  The datapoint/label digests in every record prove
the data identity.

Usage (from the repository root, project venv)::

    PY=/tmp/genesis/task_967a300e3bce7f35/evogp-venv/bin/python
    export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
    export EVOXC_DISABLE_LOCATIONS=1

    # the engine arm (one process)
    $PY test/benchmark/evoxc/training_run.py --arm engine

    # the plain-torch end-to-end arm, same loop, same seed
    $PY test/benchmark/evoxc/training_run.py --arm torch-per-tree

    # the pre-optimization snapshot (the caller supplies the tree root)
    $PY test/benchmark/evoxc/training_run.py --arm preopt --tree-root /tmp/evogp_preopt

Record: one JSON per invocation at ``--out`` (default
``test/benchmark/evoxc/results/final/training/<arm>.json``), schema
``evogp-bench-evoxc-v1`` in ``env``, with ``generations[]`` (wall ms + fitness per
generation), ``best_fitness_by_generation``, ``best_fitness_at`` (generations
10/25/50 by default), ``trajectory_digest`` (cross-arm comparison), the
per-generation median/mean ms, the total wall ms, ``final_pop_size``, the
pipeline's own reported best fitness and ``notes[]``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)

ARM_ENGINE = "engine"
ARM_TORCH = "torch-per-tree"
ARM_PREOPT = "preopt"
ARMS = (ARM_ENGINE, ARM_TORCH, ARM_PREOPT)
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "final", "training")
DEFAULT_ROWS = 1000


# ---------------------------------------------------------------------------
# plumbing (mirrors worker.py so every script switches trees the same way)
# ---------------------------------------------------------------------------
def install_tree_root(tree_root: str) -> str:
    """Put ``<tree_root>/src`` (or ``tree_root``) first on ``sys.path``; return it."""
    root = os.path.abspath(tree_root)
    src = os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root
    if src not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, src)
    return src


def target_func(x):
    """``src/evogp/sr_test.py``'s target, generalized to ``input_len`` inputs.

    ``sum_i x_i**4 / (x_i**4 + 1)`` -- at ``--input-len 2`` this is exactly the
    sr_test target.  ``x`` is the per-row ``(input_len,)`` tensor
    ``torch.vmap`` hands the function inside
    ``SymbolicRegression.generate_data``, so ``x[i]`` is one scalar per data row.
    """
    val = x[0] ** 4 / (x[0] ** 4 + 1)
    for index in range(1, x.shape[0]):
        val = val + x[index] ** 4 / (x[index] ** 4 + 1)
    return val.reshape(-1)


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise SystemExit(f"--report-gens values must be >= 1 (1-indexed); got {value}")
        values.append(value)
    if not values:
        raise SystemExit("--report-gens must list at least one generation")
    return values


def median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.pop < 1:
        raise SystemExit("--pop must be >= 1")
    install_tree_root(args.tree_root)

    import torch

    common.import_evogp()
    from evogp.algorithm import (
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, dense_engine, evoxc_engine

    import arms as arms_mod  # noqa: E402  (arm namespace + SRLossProblem)
    import worker as worker_mod  # noqa: E402  (forest_stats: population context)

    # the torch arm must not be helped by an engine route (harness convention);
    # the engine/preopt arms measure whatever the caller's knobs resolve to
    if args.arm == ARM_TORCH:
        knobs = arms_mod.apply_arm_env("torch-per-tree")
    else:
        knobs = {
            "eval_mode": os.environ.get("EVOGP_EVAL_MODE"),
            "backend": os.environ.get("EVOGP_EVOXC_BACKEND"),
        }

    # one identical seed for every arm, applied before anything draws
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # -- the synthetic problem (sr_test's construction; no dataset, no network) --
    problem = SymbolicRegression(
        func=target_func,
        num_inputs=args.input_len,
        num_data=args.num_data,
        lower_bounds=args.lower_bounds,
        upper_bounds=args.upper_bounds,
    )

    # -- population + GP wiring (mirrors src/evogp/sr_test.py) -----------------
    descriptor = common.make_descriptor(args.tree_len, args.input_len, output_len=1)
    initial_forest = Forest.random_generate(args.pop, descriptor)
    population_stats = worker_mod.forest_stats(initial_forest)

    algorithm = GeneticProgramming(
        initial_forest=initial_forest,
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )

    # -- the evaluator: the library's problem, or the same loop on plain torch --
    torch_arm = None
    if args.arm == ARM_TORCH:
        # the minimal workload TorchArm wants: the SAME problem data (so the two
        # arms train on identical datapoints/labels) and the live forest
        workload = {
            "forest": initial_forest,
            "x": problem.datapoints,
            "labels": problem.labels,
            "descriptor": descriptor,
            "cell": {
                "id": f"train-p{args.pop}-L{args.tree_len}-i{args.input_len}-N{args.num_data}",
                "pop": args.pop,
                "tree_len": args.tree_len,
                "input_len": args.input_len,
                "rows": args.num_data,
            },
            "seed": args.seed,
            "using_funcs": list(common.DESCRIPTOR_FUNCS),
        }
        torch_arm = arms_mod.TorchArm("torch-per-tree", "per_tree", workload)
        train_problem = arms_mod.SRLossProblem(torch_arm.fitness)
    else:
        train_problem = problem

    pipeline = StandardPipeline(
        algorithm,
        train_problem,
        generation_limit=args.generations,
        is_show_details=False,
    )

    cache_before = evoxc_engine.cache_stats()
    artifact_stats_fn = getattr(evoxc_engine, "artifact_cache_stats", None)
    artifact_before = artifact_stats_fn() if artifact_stats_fn else None

    # -- the training run: one timed library step per generation ---------------
    generations: List[Dict[str, Any]] = []
    training_start = time.perf_counter()
    for generation in range(1, args.generations + 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fitnesses = pipeline.step()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1e3
        generations.append(
            {
                "generation": generation,
                "wall_ms": round(wall_ms, 4),
                "gen_best": float(torch.max(fitnesses)),
                "best_fitness": float(pipeline.best_fitness),
            }
        )
    total_wall_ms = (time.perf_counter() - training_start) * 1e3

    cache_after = evoxc_engine.cache_stats()
    artifact_after = artifact_stats_fn() if artifact_stats_fn else None

    walls = [entry["wall_ms"] for entry in generations]
    best_by_generation = [entry["best_fitness"] for entry in generations]
    gen_best_by_generation = [entry["gen_best"] for entry in generations]
    report_gens = parse_int_list(args.report_gens)
    best_fitness_at = {
        str(generation): (best_by_generation[generation - 1] if generation <= len(best_by_generation) else None)
        for generation in report_gens
    }

    evaluated_by = (
        "torch-per-tree: arms_torch.PerTreeEvaluator rebuilt per generation, "
        "substituted through arms.SRLossProblem"
        if args.arm == ARM_TORCH
        else "engine: SymbolicRegression.evaluate -> Forest.SR_fitness"
    )
    record: Dict[str, Any] = {
        "env": common.env_block(
            args.tree_root,
            args.arm,
            extra={
                "applied_knobs": knobs,
                "training": {
                    "arm": args.arm,
                    "pop": args.pop,
                    "generations": args.generations,
                    "tree_len": args.tree_len,
                    "input_len": args.input_len,
                    "num_data": args.num_data,
                    "seed": args.seed,
                },
            },
        ),
        "measurement": "training-run",
        "arm": args.arm,
        "family": "torch" if torch_arm is not None else "evoxc",
        "config": {
            "pop": args.pop,
            "generations": args.generations,
            "tree_len": args.tree_len,
            "input_len": args.input_len,
            "num_data": args.num_data,
            "lower_bounds": args.lower_bounds,
            "upper_bounds": args.upper_bounds,
            "seed": args.seed,
            "tree_root": os.path.abspath(args.tree_root),
            "descriptor": {
                "max_tree_len": descriptor.max_tree_len,
                "input_len": descriptor.input_len,
                "output_len": descriptor.output_len,
                "using_funcs": list(common.DESCRIPTOR_FUNCS),
                "mutation_max_layer_cnt": 3,
                "mutation_rate": 0.2,
                "survival_rate": 0.3,
                "elite_rate": 0.01,
            },
        },
        "problem": {
            "name": "synthetic symbolic regression (the src/evogp/sr_test.py target, "
                    "generalized to input_len inputs)",
            "func": "sum_i x_i**4 / (x_i**4 + 1)",
            "problem_dim": problem.problem_dim,
            "solution_dim": problem.solution_dim,
            "datapoints_digest": common.digest(problem.datapoints),
            "labels_digest": common.digest(problem.labels),
        },
        "population": population_stats,
        "protocol": {
            "one_process_per_arm": True,
            "generation_driver": "StandardPipeline.step() (run() is the same loop plus the stopping rules)",
            "fitness_call_path": evaluated_by,
            "sync_bracketed": True,
            "seed": args.seed,
            "generations": args.generations,
            "report_gens": report_gens,
            "torch_arm_setup_ms": round(torch_arm.setup_ms, 4) if torch_arm is not None else None,
        },
        "route": {
            "knobs": knobs,
            "resolved": common.route_decision(args.pop, args.num_data, knobs["eval_mode"], knobs["backend"]),
            "evaluator": evaluated_by,
        },
        "generations": generations,
        "best_fitness_by_generation": best_by_generation,
        "gen_best_by_generation": gen_best_by_generation,
        "best_fitness_at": best_fitness_at,
        "trajectory_digest": common.digest(best_by_generation),
        "gen_best_digest": common.digest(gen_best_by_generation),
        "total_wall_ms": round(total_wall_ms, 4),
        "per_generation_ms": {
            "median": round(median(walls), 4),
            "mean": round(sum(walls) / len(walls), 4),
            "min": round(min(walls), 4),
            "max": round(max(walls), 4),
            "first_generation_ms": walls[0],
        },
        "final_pop_size": int(pipeline.algorithm.forest.pop_size),
        "pipeline_reported_best_fitness": float(pipeline.best_fitness),
        "cache": {
            "before": cache_before,
            "after": cache_after,
            "compile_misses_total": cache_after["misses"] - cache_before["misses"],
            "artifact": "absent" if artifact_stats_fn is None
            else {"enabled": bool(os.environ.get("EVOGP_EVOXC_ARTIFACT_DIR")),
                  "before": artifact_before, "after": artifact_after},
        },
        "dense_policy": {
            "auto_min_pop": dense_engine.resolve_auto_min_pop(),
            "auto_max_rows": dense_engine.resolve_auto_max_rows(),
        },
        "notes": [],
    }

    record["notes"].append(
        "the fitness callable goes through the pipeline exactly as the library calls "
        "it (StandardPipeline.step() -> problem.evaluate); this script only adds the "
        "per-generation clock and the trajectory record"
    )
    record["notes"].append(
        "speed is only meaningful next to best_fitness_by_generation: a faster arm "
        "that converges worse is not a win -- compare the trajectories and the "
        "trajectory_digest before quoting per-generation milliseconds"
    )
    record["notes"].append(
        "the problem data is generated by SymbolicRegression in-process from "
        "torch.manual_seed(seed) (no dataset, no network); the datapoint/label digests "
        "are identical across arms"
    )
    if args.arm == ARM_TORCH:
        record["notes"].append(
            "torch arm: the evaluator (arms_torch.PerTreeEvaluator) is rebuilt from the "
            "live forest at every generation and that decode is inside the generation "
            "wall time, like the engine's own host-side decode; the initial-forest "
            "decode is the separate torch_arm_setup_ms"
        )
    if args.arm == ARM_PREOPT:
        record["notes"].append(
            "preopt arm: identical code path to --arm engine; the snapshot under test is "
            f"{os.path.abspath(args.tree_root)!r} (the caller supplies --tree-root, "
            "nothing is hard-coded)"
        )
        current = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
        if os.path.abspath(args.tree_root) in (current, os.path.join(current, "src")):
            record["notes"].append(
                "WARNING: --tree-root resolves to this checkout, not a snapshot; the "
                "'preopt' label is then just a label"
            )
    if args.note:
        record["notes"].append(args.note)

    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"{args.arm}.json")
    common.write_json(out, record)
    print(
        f"[training] arm={args.arm} pop={args.pop} gens={args.generations} "
        f"total={total_wall_ms:.1f}ms median/gen={record['per_generation_ms']['median']:.1f}ms "
        f"best={record['pipeline_reported_best_fitness']:.6f} "
        f"best@{'/'.join(str(g) for g in report_gens)}="
        f"{'/'.join('n/a' if v is None else f'{v:.6f}' for v in best_fitness_at.values())} "
        f"final_pop={record['final_pop_size']} -> {out}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=ARMS, default=ARM_ENGINE, help=f"one of {ARMS}")
    parser.add_argument("--pop", type=int, default=256, help="population size")
    parser.add_argument("--generations", type=int, default=50, help="training generations")
    parser.add_argument("--tree-len", type=int, default=64, help="max_tree_len of the descriptor")
    parser.add_argument(
        "--input-len", type=int, default=2,
        help="number of problem inputs (2 is exactly the sr_test problem)",
    )
    parser.add_argument("--num-data", type=int, default=DEFAULT_ROWS, help="data rows")
    parser.add_argument("--lower-bounds", type=int, default=-5, help="input lower bound")
    parser.add_argument("--upper-bounds", type=int, default=5, help="input upper bound")
    parser.add_argument("--seed", type=int, default=0, help="torch seed, identical across arms")
    parser.add_argument(
        "--report-gens", default="10,25,50",
        help="comma-separated 1-indexed generations reported in best_fitness_at",
    )
    parser.add_argument("--tree-root", default="src", help="tree root whose src/ holds evogp")
    parser.add_argument("--out", default=None, help=f"JSON record path (default under {DEFAULT_OUT_DIR})")
    parser.add_argument("--note", default=None, help="free-form note stored in the record")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
