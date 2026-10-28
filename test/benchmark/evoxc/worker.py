"""One process, one (cell, arm): the measured unit of the campaign.

Why a process per config: the engine's per-structure compile cache and its
executable memo live in process memory, so a fresh process per (cell, arm) is
the only way every record pays its own cold compile and no arm is helped by
another arm's warm cache.  The driver (``campaign.py``) spawns these one at a
time.

What one record contains: the environment block, the cell and arm, the route the
policy actually resolved, the per-workload timings (``first_call_ms`` = cold,
then warmup + reps -> median/min/max/samples), the cache deltas (compile
misses), value digests (cross-process numeric agreement) and the GP generation
step split into eval / operator parts with its per-generation cache misses.

Usage (the driver sets ``PYTHONPATH``; run it from the repository root)::

    python test/benchmark/evoxc/worker.py \
        --cell p256-L64-i32-N1024 --arm evoxc-dense-iree \
        --out test/benchmark/evoxc/results/w1/p256-L64-i32-N1024/evoxc-dense-iree.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402
import arms as arms_mod  # noqa: E402


class TimedFitness:
    """``fitness(forest)`` wrapper that records the cost of the call it wraps.

    The generation step is timed end-to-end AND its evaluation sub-call is timed
    in place (same population, same step), so the operator part is a real
    difference and not a cross-population comparison.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last_ms = 0.0
        self.last_setup_ms = 0.0

    def __call__(self, forest, report: Optional[Dict[str, Any]] = None):
        import torch

        scratch: Dict[str, Any] = {}
        start = time.perf_counter()
        loss = self.inner(forest, scratch)
        torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - start) * 1e3
        self.last_setup_ms = float(scratch.get("setup_ms", 0.0))
        if report is not None:
            report.update(scratch)
            report["eval_ms"] = self.last_ms
        return loss


def forest_stats(forest) -> Dict[str, Any]:
    """Structure context the compile counts are read against."""
    import hashlib

    import numpy as np

    sizes = forest.batch_subtree_size[:, 0].detach().to("cpu").numpy().astype(np.int64)
    values = forest.batch_node_value.detach().to("cpu").numpy()
    types = forest.batch_node_type.detach().to("cpu").numpy()
    lengths = forest.batch_subtree_size.detach().to("cpu").numpy()
    digests = set()
    for row in range(forest.pop_size):
        length = int(lengths[row, 0])
        payload = (
            values[row, :length].tobytes() + types[row, :length].tobytes() + lengths[row, :length].tobytes()
        )
        digests.add(hashlib.sha256(payload).hexdigest()[:16])
    return {
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "tree_len_mean": float(sizes.mean()),
        "tree_len_median": float(np.median(sizes)),
        "tree_len_max": int(sizes.max()),
        "distinct_structures": len(digests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="cell id, e.g. p256-L64-i32-N1024")
    parser.add_argument("--arm", required=True, help=f"one of {common.ALL_ARMS}")
    parser.add_argument("--out", required=True, help="JSON record path")
    parser.add_argument("--tree-root", default="src", help="tree root whose src/ holds evogp")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gp-warmup", type=int, default=1)
    parser.add_argument("--gp-reps", type=int, default=5)
    parser.add_argument(
        "--workloads", default="sr_fitness,batch_forward,forward,gp_step"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", default=None, help="free-form note stored in the record")
    args = parser.parse_args()

    # the --tree-root switch: import evogp from <tree_root>/src (a git-archive
    # snapshot times exactly like this checkout) instead of PYTHONPATH order
    root = os.path.abspath(args.tree_root)
    src = os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root
    if src not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, src)

    cell = common.parse_cell(args.cell)
    knobs = arms_mod.apply_arm_env(args.arm)
    common.import_evogp()

    import torch
    from evogp.tree import dense_engine, evoxc_engine

    workload = common.build_workload(cell, args.seed)
    forest = workload["forest"]
    structure = forest_stats(forest)

    record: Dict[str, Any] = {
        "env": common.env_block(
            args.tree_root,
            args.arm,
            extra={"applied_knobs": knobs, "cell": cell, "arm_spec": common.EVOXC_ARMS.get(args.arm, common.TORCH_ARMS.get(args.arm))},
        ),
        "cell": cell,
        "arm": args.arm,
        "family": "torch" if args.arm in common.TORCH_ARMS else "evoxc",
        "protocol": {
            "warmup": args.warmup,
            "reps": args.reps,
            "gp_warmup": args.gp_warmup,
            "gp_reps": args.gp_reps,
            "workloads": args.workloads.split(","),
            "seed": args.seed,
            "sync_bracketed": True,
        },
        "structures": structure,
        "routes": {},
        "workloads": {},
        "gp_step": None,
        "notes": [],
    }
    if args.note:
        record["notes"].append(args.note)

    # -- what the policy resolves for this cell (per workload row count) -----
    for name, rows in (
        ("sr_fitness", cell["rows"]),
        ("batch_forward", cell["rows"]),
        ("forward", 1),
    ):
        record["routes"][name] = common.route_decision(
            cell["pop"], rows, knobs["eval_mode"], knobs["backend"]
        )

    # -- evaluation workloads ------------------------------------------------
    arm_obj = arms_mod.make_arm(args.arm, workload)
    record["setup_ms"] = round(arm_obj.setup_ms, 4)
    if arm_obj.family == "torch":
        record["notes"].append(
            "torch arm: the tree decode / batched plan is built once, outside the "
            "timed region (setup_ms); per-call cost is torch ops only"
        )

    requested = [name for name in args.workloads.split(",") if name]
    for name in requested:
        if name == "gp_step":
            continue
        fn = getattr(arm_obj, name)
        cache_before = evoxc_engine.cache_stats()
        timing = common.time_call(
            fn, warmup=args.warmup, reps=args.reps, stats=evoxc_engine.cache_stats
        )
        cache_after = evoxc_engine.cache_stats()
        result = timing.pop("_out")
        after_first = timing.pop("cache_after_first_call")
        entry = common.public_timing(timing)
        for key in ("first_call_ms", "median_ms", "min_ms", "max_ms", "mean_ms"):
            entry[key] = round(entry[key], 4)
        entry["samples_ms"] = [round(v, 4) for v in entry["samples_ms"]]
        entry["cache"] = {
            "before": cache_before,
            "after_first_call": after_first,
            "after": cache_after,
            "compile_misses_this_workload": cache_after["misses"] - cache_before["misses"],
            "compile_misses_first_call": (
                after_first["misses"] - cache_before["misses"] if after_first else None
            ),
        }
        entry["digest"] = common.digest(result)
        record["workloads"][name] = entry
        del result

    # -- one full GP generation step ----------------------------------------
    if "gp_step" in requested:
        timed = TimedFitness(arm_obj.fitness)
        pipeline = arms_mod.make_pipeline(workload, timed)
        for _ in range(args.gp_warmup):
            pipeline.step()
        torch.cuda.synchronize()
        steps = []
        for _ in range(args.gp_reps):
            cache_before = evoxc_engine.cache_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            pipeline.step()
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1e3
            cache_after = evoxc_engine.cache_stats()
            steps.append(
                {
                    "total_ms": round(total_ms, 4),
                    "eval_ms": round(timed.last_ms, 4),
                    "ops_ms": round(total_ms - timed.last_ms, 4),
                    "torch_setup_ms": round(timed.last_setup_ms, 4),
                    "compile_misses": cache_after["misses"] - cache_before["misses"],
                    "memo_hits": cache_after["memo_hits"] - cache_before["memo_hits"],
                }
            )
        totals = sorted(step["total_ms"] for step in steps)
        evals = sorted(step["eval_ms"] for step in steps)
        mid = len(totals) // 2
        record["gp_step"] = {
            "warmup": args.gp_warmup,
            "reps": args.gp_reps,
            "steps": steps,
            "median_total_ms": totals[mid],
            "min_total_ms": totals[0],
            "max_total_ms": totals[-1],
            "median_eval_ms": evals[mid],
            "median_ops_ms": round(totals[mid] - evals[mid], 4),
            "compile_misses_total": sum(step["compile_misses"] for step in steps),
            "final_pop_size": int(pipeline.algorithm.forest.pop_size),
            "best_fitness": float(pipeline.best_fitness),
        }

    record["engine_cache_final"] = evoxc_engine.cache_stats()
    record["dense_policy"] = {
        "auto_min_pop": dense_engine.resolve_auto_min_pop(),
        "auto_max_rows": dense_engine.resolve_auto_max_rows(),
    }
    common.write_json(args.out, record)
    summary = " ".join(
        f"{name}={entry['median_ms']:.3f}ms" for name, entry in record["workloads"].items()
    )
    if record["gp_step"]:
        summary += f" gp={record['gp_step']['median_total_ms']:.3f}ms"
    print(f"[record] {args.cell} {args.arm} -> {args.out} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
