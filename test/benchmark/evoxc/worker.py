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

What a BATCH arm (``EVOGP_EVAL_MODE=batch``) adds on top of that -- and only a
batch arm: ``batch_policy`` (whether the tree under test ships the module and
declares the mode, plus the band knobs and where each value came from),
``batch_cache`` (the merged-graph cache counters before the workloads and after
the whole config, which counter names the route exposes, and the first-call and
total deltas of ``batch_exec_misses`` -- the real compile cache, a hit never
reaches ``evoxc.lower``/``compile``/``load`` -- and of ``batch_plan_misses`` --
host decode + traces + merge) and, per workload / per generation, the batch
deltas next to the evoxc ones.  That route compiles ONE merged graph per
population structure multiset, so a multi-second ``first_call_ms`` and a large
``compile_misses_first_call`` are the expected cost of the route -- recorded as
measured, never retried away as a failure.

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
from typing import Any, Dict, List, Optional, Tuple

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


def _stat_value(stats: Any, key: str) -> Optional[int]:
    """One int counter out of an OPTIONAL stats dict (``None`` when unavailable).

    The batch engine owns its counter contract, so this is deliberately tolerant:
    an absent key reads as ``0`` instead of raising, because a benchmark record must
    never be lost to a counter the route renamed or has not implemented yet.
    """
    if not isinstance(stats, dict):
        return None
    try:
        return int(stats.get(key, 0) or 0)
    except (TypeError, ValueError):
        return None


def _stats_delta(before: Any, after: Any, key: str) -> Optional[int]:
    """``after[key] - before[key]`` across two optional snapshots (``None`` if unknown)."""
    start, end = _stat_value(before, key), _stat_value(after, key)
    if start is None or end is None:
        return None
    return end - start


def _has_stat_key(before: Any, after: Any, key: str) -> bool:
    """Does at least one of the two snapshots actually carry ``key``?"""
    return any(isinstance(stats, dict) and key in stats for stats in (before, after))


#: The counter names the batch route's compile misses can live under, in preference
#: order (single-sourced in :mod:`common`).  ``batch_engine.batch_cache_stats()``
#: (eight keys) names its two front tables explicitly: ``batch_exec_*`` describes
#: ``BATCH_EXEC_MEMO`` -- the real compile cache, since a hit never reaches
#: ``evoxc.lower`` / ``compile`` / ``load`` -- and ``batch_plan_*`` describes
#: ``BATCH_PLAN_MEMO`` (host decode + traces + merge).  A bare ``misses`` is the
#: older spelling this harness used to read; it is a fallback, never the primary.
BATCH_EXEC_MISS_KEYS: Tuple[str, ...] = common.BATCH_EXEC_MISS_KEYS
#: The plan-table counter: a plan miss re-traces and re-merges, it does not recompile.
BATCH_PLAN_MISS_KEYS: Tuple[str, ...] = common.BATCH_PLAN_MISS_KEYS


def _snapshot_key(snapshot: Any, candidates: Tuple[str, ...]) -> Optional[str]:
    """The first of ``candidates`` this snapshot carries (``None`` when it carries none)."""
    if not isinstance(snapshot, dict):
        return None
    for key in candidates:
        if key in snapshot:
            return key
    return None


def batch_miss_key(snapshot: Any) -> Optional[str]:
    """Which counter holds the batch route's real compile misses in ``snapshot``.

    ``batch_exec_misses`` on the engine route we measure (``BATCH_EXEC_MEMO``: a hit
    never reaches ``evoxc.lower`` / ``compile`` / ``load``), the older bare ``misses``
    when that is all a snapshot carries, and ``None`` when the snapshot has neither --
    an unknown counter must read as null, never as "no compile".
    """
    return _snapshot_key(snapshot, BATCH_EXEC_MISS_KEYS)


def batch_plan_miss_key(snapshot: Any) -> Optional[str]:
    """Which counter holds the batch route's plan misses in ``snapshot`` (``None`` if none)."""
    return _snapshot_key(snapshot, BATCH_PLAN_MISS_KEYS)


def _stats_delta_for(before: Any, after: Any, resolver) -> Optional[int]:
    """Delta of the counter ``resolver`` names on either snapshot (``None`` if unknown).

    Strict on purpose: a snapshot that does not carry the resolved key reads as
    UNKNOWN, never as ``0``, so a null delta always means "this counter was not
    available", which is exactly what ``check.py`` and the renderer need to see.
    """
    key = resolver(before) or resolver(after)
    if key is None:
        return None
    for snapshot in (before, after):
        if not (isinstance(snapshot, dict) and key in snapshot):
            return None
    return _stats_delta(before, after, key)


def batch_miss_delta(before: Any, after: Any) -> Optional[int]:
    """``batch_exec_misses`` delta across two batch-cache snapshots (``None`` if unknown)."""
    return _stats_delta_for(before, after, batch_miss_key)


def batch_plan_miss_delta(before: Any, after: Any) -> Optional[int]:
    """``batch_plan_misses`` delta across two batch-cache snapshots (``None`` if unknown)."""
    return _stats_delta_for(before, after, batch_plan_miss_key)


def stats_keys(*snapshots: Any) -> List[str]:
    """Sorted counter names of the FIRST snapshot that has any (``[]`` when none does).

    A record must state which counters the route actually exposes: that is what makes
    a null delta checkable ("the key was absent") instead of ambiguous.
    """
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and snapshot:
            return sorted(snapshot)
    return []


class BatchStatsProbe:
    """Sample the batch cache the moment ``time_call``'s untimed first call returns.

    ``common.time_call`` only exposes one post-sequence sample point (it samples the
    ``stats`` callable right after the cold call), and the batch route's compile
    lands entirely in that first call -- so wrapping the callable is the only way to
    report ``compile_misses_first_call`` separately from the warmup+reps total.  When
    the route has no stats callable the snapshot stays ``None``; every other arm is
    handed the arm's own bound method, i.e. unchanged behaviour.
    """

    def __init__(self, fn, stats):
        self.fn = fn
        self.stats = stats
        self.first: Optional[Dict[str, Any]] = None

    def __call__(self):
        out = self.fn()
        if self.first is None and self.stats is not None:
            self.first = self.stats()
        return out


def forest_stats(forest) -> Dict[str, Any]:
    """Structure context the compile counts are read against.

    ``distinct_structures`` / ``structure_signature`` come from
    ``common.population_signature`` -- the structure-multiset fingerprint the batch
    route's merged-graph cache is keyed on -- so a record states the compile story in
    the cache's own terms instead of maintaining a second, subtly different hash.
    """
    import numpy as np

    signature = common.population_signature(forest)
    sizes = forest.batch_subtree_size[:, 0].detach().to("cpu").numpy().astype(np.int64)
    return {
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "tree_len_mean": float(sizes.mean()),
        "tree_len_median": float(np.median(sizes)),
        "tree_len_max": int(sizes.max()),
        "distinct_structures": int(signature["distinct_structures"]),
        "structure_signature": signature["signature"],
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

    # -- batch support preflight ---------------------------------------------
    # EVOGP_EVAL_MODE=batch is only meaningful when the tree under test ships the
    # module AND its policy declares the mode.  A missing route must fail here, with
    # one readable paragraph, instead of writing a record that silently measured some
    # other route (or dying deep inside the tree layer): the driver logs
    # ``status: failed`` plus this message, and no record exists for the cell.
    is_batch = common.is_batch_arm(args.arm)
    batch_policy = common.batch_policy(src) if is_batch else None
    if is_batch and not (batch_policy["module_present"] and batch_policy["mode_supported"]):
        raise SystemExit(
            f"arm {args.arm}: EVOGP_EVAL_MODE=batch needs BOTH a "
            f"<src>/evogp/tree/batch_engine.py module and 'batch' declared in "
            f"evogp.tree.dense_engine.EVAL_MODES, and this tree does not provide the "
            f"pair -- found module_present={batch_policy['module_present']} "
            f"(expected {batch_policy['module']}), "
            f"declared_modes={batch_policy['declared_modes']!r} "
            f"(mode_supported={batch_policy['mode_supported']}) for src={src}. "
            f"NO record was written for cell {args.cell} arm {args.arm}: the batch "
            f"route cannot run here, and a record of a different route would be worse "
            f"than a missing one."
        )

    common.import_evogp()

    # -- the batch route's own cache counters (optional, batch arms only) -----
    # The merged-graph cache may or may not be evoxc_engine's, so ask the batch
    # engine itself.  Resolved only for batch arms: importing that module can then
    # never break a torch/evoxc arm, and a missing ``batch_cache_stats`` is a
    # reportable state (recorded as null), not a crash.
    batch_stats = None
    if is_batch:
        try:
            from evogp.tree import batch_engine as _batch_engine_probe
        except ImportError:  # noqa: F401 - the route is optional on older trees
            _batch_engine_probe = None
        batch_stats = getattr(_batch_engine_probe, "batch_cache_stats", None)
        if not callable(batch_stats):
            batch_stats = None

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

    # -- batch route provenance / merged-graph cache counters ------------------
    # These keys exist ONLY for batch arms: "not a batch route" must stay
    # distinguishable from "batch route whose counters are unavailable", so no
    # None-noise is added to the existing record schema.
    if is_batch:
        record["batch_policy"] = batch_policy
        record["batch_cache"] = {
            "available": batch_stats is not None,
            "stats_before_workloads": batch_stats() if batch_stats is not None else None,
            "stats_after": None,
            "stats_keys": [],
            "compile_misses_first_call": None,
            "compile_misses_total": None,
            "plan_misses_first_call": None,
            "plan_misses_total": None,
        }
        record["notes"].append(
            "batch arm: one merged graph per population structure multiset, so a "
            "multi-second first_call_ms and a large compile_misses_first_call are the "
            "expected cold cost of the route and are recorded as measured, never retried"
        )
        if batch_stats is None:
            record["notes"].append(
                "batch arm: evogp.tree.batch_engine.batch_cache_stats() is unavailable on "
                "this tree, so every merged-graph cache counter is null (NOT zero) and "
                "the compile cost can only be read off first_call_ms"
            )

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
    batch_first_call_misses = None
    batch_first_call_plan_misses = None
    for name in requested:
        if name == "gp_step":
            continue
        fn = getattr(arm_obj, name)
        cache_before = evoxc_engine.cache_stats()
        batch_before = batch_stats() if batch_stats is not None else None
        # A batch arm's compile lands in time_call's untimed first call; the probe
        # snapshots the merged-graph cache the instant that call returns.  Every other
        # arm (and a batch arm with no stats callable) gets its own bound method
        # untouched, i.e. no extra call in the timed loop.
        probe = (
            BatchStatsProbe(fn, batch_stats)
            if is_batch and batch_stats is not None
            else None
        )
        timing = common.time_call(
            probe if probe is not None else fn,
            warmup=args.warmup,
            reps=args.reps,
            stats=evoxc_engine.cache_stats,
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
        if is_batch:
            # the existing keys above stay the evoxc-engine view (check.py reads them);
            # the batch counters sit next to it so the two caches are never conflated
            batch_after_first = probe.first if probe is not None else None
            batch_after = batch_stats() if batch_stats is not None else None
            entry["cache"]["batch_after_first_call"] = batch_after_first
            entry["cache"]["batch_after"] = batch_after
            entry["cache"]["batch_compile_misses_first_call"] = batch_miss_delta(
                batch_before, batch_after_first
            )
            entry["cache"]["batch_compile_misses_this_workload"] = batch_miss_delta(
                batch_before, batch_after
            )
            entry["cache"]["batch_plan_misses_first_call"] = batch_plan_miss_delta(
                batch_before, batch_after_first
            )
            entry["cache"]["batch_plan_misses_this_workload"] = batch_plan_miss_delta(
                batch_before, batch_after
            )
            if batch_first_call_misses is None:
                batch_first_call_misses = entry["cache"]["batch_compile_misses_first_call"]
            if batch_first_call_plan_misses is None:
                batch_first_call_plan_misses = entry["cache"]["batch_plan_misses_first_call"]
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
            batch_before = batch_stats() if batch_stats is not None else None
            torch.cuda.synchronize()
            start = time.perf_counter()
            pipeline.step()
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1e3
            cache_after = evoxc_engine.cache_stats()
            batch_after = batch_stats() if batch_stats is not None else None
            step = {
                "total_ms": round(total_ms, 4),
                "eval_ms": round(timed.last_ms, 4),
                "ops_ms": round(total_ms - timed.last_ms, 4),
                "torch_setup_ms": round(timed.last_setup_ms, 4),
                "compile_misses": cache_after["misses"] - cache_before["misses"],
                "memo_hits": cache_after["memo_hits"] - cache_before["memo_hits"],
            }
            if is_batch:
                # a fresh population is a new structure multiset, i.e. a new merged
                # graph: this is the counter that explains a slow batch GP step.
                # ``batch_compile_misses`` is the EXEC table (a miss recompiles),
                # ``batch_plan_misses`` the PLAN table (re-decode/re-trace/re-merge).
                step["batch_compile_misses"] = batch_miss_delta(batch_before, batch_after)
                step["batch_plan_misses"] = batch_plan_miss_delta(batch_before, batch_after)
                if _has_stat_key(batch_before, batch_after, "memo_hits"):
                    step["batch_memo_hits"] = _stats_delta(batch_before, batch_after, "memo_hits")
            steps.append(step)
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
        if is_batch:
            misses = [step["batch_compile_misses"] for step in steps]
            record["gp_step"]["batch_compile_misses_total"] = (
                None if any(value is None for value in misses) else sum(misses)
            )
            plan_misses = [step["batch_plan_misses"] for step in steps]
            record["gp_step"]["batch_plan_misses_total"] = (
                None if any(value is None for value in plan_misses) else sum(plan_misses)
            )

    if is_batch:
        # close the batch cache story for the whole config: the deltas above are per
        # workload, this is the one number a reader can price the process against
        stats_after = batch_stats() if batch_stats is not None else None
        record["batch_cache"]["stats_after"] = stats_after
        record["batch_cache"]["stats_keys"] = stats_keys(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )
        record["batch_cache"]["compile_misses_first_call"] = batch_first_call_misses
        record["batch_cache"]["compile_misses_total"] = batch_miss_delta(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )
        record["batch_cache"]["plan_misses_first_call"] = batch_first_call_plan_misses
        record["batch_cache"]["plan_misses_total"] = batch_plan_miss_delta(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )

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
