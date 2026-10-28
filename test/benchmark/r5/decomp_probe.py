#!/usr/bin/env python
"""``decomp_probe.py`` -- R5 Q1/Q5/Q7: per-call decomposition of ONE warm dense evaluate.

The question: the dense-iree ``problem.evaluate(forest)`` route costs ~5.4-6.9 ms
warm (Q4 same-window window, ``results/q4_window/``) while its pure compiled
forward is ~1.2-1.5 ms -- so where do the other ~4 ms go?

Method (analysis-only; ``src/evogp`` and the foreign ``evoxc`` are never edited):
the probe rebuilds the harness' ``--task eval64`` wiring verbatim
(:func:`probe_lib.build_eval_case` = ``test/benchmark/bench_gp.py:768-806``) and
times ``problem.evaluate(forest)`` (``bench_gp.py:1374-1381``: one
``torch.cuda.synchronize()`` pair around one call, call 0 cold, calls 1..24
warm).  Around that call it monkeypatches the dense chain's own call sites
(:func:`probe_lib.install_dense_wraps`) into >=8 non-overlapping buckets --
exclusive times, so the sum of every bucket is the outer span by construction --
plus the ``evoxc`` module proxy (``from_numpy`` / ``run``) and
``evoxc.core.Tensor.to`` bucketed by caller (``payload`` = upload, ``_to_host`` =
readback).

Recorded per fixture (one process, immutable JSON + log):

* the 25 per-call harness totals + the inner call wall + the post-call fence,
* per-call exclusive ms for every bucket + the count of every bucket,
* the reconciliation (sum of buckets vs the in-process total),
* Q7: ``evoxc.CompileCache`` hit/miss/size deltas + the ``build_dense`` wall of
  every call, and a deliberate cache-clear cold rebuild (phase ``cold``),
* two controls: a region-sync pass (a CUDA fence *inside* every span, showing
  where the GPU tail is waited for) and an uninstrumented pass (same 25-call
  shape with the wrappers inert, measuring the probe's own overhead),
* ``--cprofile`` mode: the same loop with zero instrumentation under cProfile,
  top-40 tottime + cumtime tables (ranking only; cProfile distorts absolutes).

Usage (repo root, project venv, GPU):

    PYTHONPATH=src EVOXC_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      EVOGP_EVOXC_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_EVOXC_IREE_DEVICE=0 \
      EVOGP_EVOXC_IREE_OPT_LEVEL=O1 \
      <venv>/bin/python test/benchmark/r5/decomp_probe.py \
        --fixture test/goldens/forest_random.npz --eval-reps 25 \
        --json-out test/benchmark/r5/results/q1_decomp/decomp_f64.json \
        --log-out  test/benchmark/r5/results/q1_decomp/decomp_f64.log
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import probe_lib as pl  # noqa: E402

PROBE = "r5/decomp_probe.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="R5 Q1/Q5/Q7 dense-evaluate per-call decomposition (analysis-only)"
    )
    parser.add_argument("--fixture", required=True, help="the eval64 fixture .npz")
    parser.add_argument("--eval-reps", type=int, default=25, help="measured calls (call 0 cold)")
    parser.add_argument("--sync-reps", type=int, default=5, help="region-sync control calls")
    parser.add_argument("--plain-reps", type=int, default=5, help="uninstrumented control calls")
    parser.add_argument("--cold-reps", type=int, default=1, help="cache-clear cold rebuild calls")
    parser.add_argument("--no-controls", action="store_true", help="warm phase only")
    parser.add_argument("--cprofile", action="store_true", help="profile the loop instead of timing it")
    parser.add_argument("--cprofile-out", default=None, help="cProfile text table path")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


class Tee:
    """stdout + optional log file (the log is the committed human record)."""

    def __init__(self, path=None):
        self.handle = open(path, "w") if path else None
        self.buffer = io.StringIO()

    def write(self, text):
        sys.stdout.write(text)
        self.buffer.write(text)
        if self.handle:
            self.handle.write(text)
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()

    def text(self):
        return self.buffer.getvalue()


def call_once(problem, forest, profiler, index, phase, sync_region=False):
    """One harness-shaped call: pre-sync, call, post-sync (``bench_gp``:1377-1381)."""
    import torch

    profiler.flags["sync_region"] = sync_region
    torch.cuda.synchronize()
    t0 = pl.NOW()
    if profiler.enabled:  # the plain control phase stays completely uninstrumented
        profiler.begin_call(index, phase)
    result = problem.evaluate(forest)
    call_s = pl.NOW() - t0
    if profiler.enabled:
        profiler.end_call(call_s)
    t1 = pl.NOW()
    torch.cuda.synchronize()
    post_sync_s = pl.NOW() - t1
    return {
        "index": index,
        "phase": phase,
        "call_ms": call_s * 1e3,
        "post_sync_ms": post_sync_s * 1e3,
        "total_ms": (pl.NOW() - t0) * 1e3,
        "fitness": float(result.detach().reshape(-1)[0].item()),
        "fitness_sha256": hashlib.sha256(
            result.detach().reshape(-1).cpu().numpy().tobytes()
        ).hexdigest(),
    }


def median(values):
    return statistics.median(values) if values else None


def phase_table(rows, phase):
    """(median total, median call, median post-sync) for one phase."""
    subset = [r for r in rows if r["phase"] == phase]
    warm = [r for r in subset if r["index"] != 0 or phase != "warm"]
    return {
        "calls": len(subset),
        "warm_calls": len(warm),
        "median_total_ms": median([r["total_ms"] for r in warm]),
        "median_call_ms": median([r["call_ms"] for r in warm]),
        "median_post_sync_ms": median([r["post_sync_ms"] for r in warm]),
        "first_call_ms": subset[0]["total_ms"] if subset else None,
    }


def component_rows(summary, order, total_ms):
    rows = []
    for name in order:
        entry = summary.get(name)
        if entry is None or entry["count"] == 0:
            continue
        pct = 100.0 * entry["median_self_ms"] / total_ms if total_ms else 0.0
        rows.append((name, entry["count"], entry["median_self_ms"], pct))
    return rows


def warm_reconciliation(profiler):
    """Per-call ``sum(exclusive buckets) / call wall`` ratios (the honest check)."""
    ratios = []
    for record in profiler.calls:
        if record["phase"] != "warm" or record["index"] == 0:
            continue
        booked = sum(b.get("self_s", 0.0) for b in record["buckets"].values())
        wall = record["outer_s"]
        if wall:
            ratios.append(100.0 * booked / wall)
    return {
        "per_call_pct": ratios,
        "median_pct": statistics.median(ratios) if ratios else None,
    }

def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or os.path.splitext(os.path.basename(args.fixture))[0]
    out = Tee(args.log_out)

    import numpy as np  # noqa: F401  (kept: from_numpy patch needs it loaded)
    import torch

    import evoxc
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, dense_engine

    import evogp.tree.forest as forest_module

    problem, forest, x, labels = build_case(args, out)
    rows = []
    components = {}

    if args.cprofile:
        payload = run_cprofile(args, problem, forest, out)
        stamps = pl.identity_stamp(evoxc, extra={"probe": PROBE, "tag": tag, "mode": "cprofile"})
        pl.write_json(args.json_out, {"probe": PROBE, "tag": tag, **payload, "stamp": stamps})
        out.write(f"\nwrote {args.json_out}\n")
        out.close()
        return 0

    profiler = pl.Profiler(enabled=True)
    profiler.flags = {"sync_region": False}
    pl.install_dense_wraps(
        dense_engine, forest_module, SymbolicRegression, profiler, sync_region=True
    )
    pl.install_etl_proxy(dense_engine, profiler)
    pl.patch_etl_tensor_to(evoxc, profiler)
    plan_sink = {}
    pl.install_plan_capture(dense_engine, plan_sink, profiler)
    cache = dense_engine.DENSE_CACHE
    out.write(f"=== {PROBE}: tag={tag} ===\n")
    out.write(f"fixture        : {args.fixture}\n")
    out.write(
        f"case           : Forest(pop={forest.pop_size}, max_tree_len={forest.max_tree_len}, "
        f"input_len={forest.input_len}, output_len={forest.output_len}) + "
        f"SymbolicRegression(datapoints=x{x.shape}, labels=labels{labels.shape})\n"
    )
    out.write(f"evoxc          : {evoxc.__file__}\n")
    out.write(f"wrapped        : {', '.join(profiler.install_status['wrapped'])}\n")
    out.write(f"missing        : {profiler.install_status['missing'] or 'none'}\n")
    out.write(f"compile cache  : hits={cache.hits} misses={cache.misses} size={len(cache)}\n")
    out.write("\n-- warm phase: call 0 cold, calls 1..N-1 warm (bench_gp.py:1374-1381) --\n")

    # ---- warm phase ------------------------------------------------------
    for i in range(args.eval_reps):
        rows.append(call_once(problem, forest, profiler, i, "warm"))
        record = profiler.calls[-1]
        build = record["buckets"].get("build_dense_build_cached", {})
        out.write(
            f"call {i:2d}  total {rows[-1]['total_ms']:8.3f} ms  call {rows[-1]['call_ms']:8.3f}  "
            f"post-sync {rows[-1]['post_sync_ms']:6.3f}  build {build.get('total_s', 0.0)*1e3:9.3f}  "
            f"cache hits={cache.hits} misses={cache.misses}\n"
        )

    warm = phase_table(rows, "warm")
    warm_summary = profiler.summary("warm")
    total_ms = warm["median_total_ms"]
    components["warm"] = component_rows(warm_summary, pl.BUCKET_ORDER, total_ms)
    sum_self = sum(row[2] for row in components["warm"])
    per_call_reconcile = warm_reconciliation(profiler)
    # ---- control phases --------------------------------------------------
    controls = {}
    if not args.no_controls:
        for phase, reps, sync_region in (
            ("sync_region", args.sync_reps, True),
            ("plain", args.plain_reps, False),
            ("cold", args.cold_reps, False),
        ):
            if reps <= 0:
                continue
            if phase == "cold":
                cache.clear()
                out.write(
                    f"\n-- cold rebuild control: DENSE_CACHE.clear() then {reps} call(s) --\n"
                )
            elif phase == "sync_region":
                out.write(
                    f"\n-- region-sync control: CUDA fence inside every span ({reps} calls) --\n"
                )
            else:
                out.write(f"\n-- uninstrumented control: wrappers inert ({reps} calls) --\n")
            if phase == "plain":
                profiler.enabled = False
            first = len(rows)
            for j in range(reps):
                rows.append(call_once(problem, forest, profiler, j, phase, sync_region))
            profiler.enabled = True
            for row in rows[first:]:
                out.write(
                    f"{phase:12s} call {row['index']:2d}  total {row['total_ms']:9.3f} ms  "
                    f"call {row['call_ms']:9.3f}  post-sync {row['post_sync_ms']:7.3f}\n"
                )
            phase_summary = profiler.summary(phase) if phase != "plain" else {}
            table = phase_table(rows, phase)
            controls[phase] = {
                "table": table,
                "rows": rows[first:],
                "components": component_rows(phase_summary, pl.BUCKET_ORDER, table["median_total_ms"])
                if phase != "plain"
                else [],
                "per_call_components": {
                    name: profiler.per_call_bucket_ms(name, phase)
                    for name in profiler.buckets
                    if any(
                        c["buckets"].get(name, {}).get("count", 0)
                        for c in profiler.calls
                        if c["phase"] == phase
                    )
                }
                if phase != "plain"
                else {},
                "fences": {
                    name: profiler.exclusive_ms(name) / max(1, reps)
                    for name in profiler.buckets
                    if name.startswith("fence_after:")
                }
                if phase == "sync_region"
                else {},
                "cache": {"hits": cache.hits, "misses": cache.misses, "size": len(cache)},
            }
    # ---- report ----------------------------------------------------------
    out.write("\n=== warm component table (exclusive, warm calls only) ===\n")
    out.write(
        f"{'component':28s} {'count':>5s} {'median_ms':>10s} {'%total':>7s} {'total_ms':>10s}\n"
    )
    for name, count, med, pct in components["warm"]:
        out.write(
            f"{name:28s} {count:5d} {med:10.4f} {pct:6.2f}% "
            f"{warm_summary[name]['total_self_ms']:10.3f}\n"
        )
    out.write(
        f"{'SUM(exclusive)':28s} {'':>5s} {sum_self:10.4f} {100.0 * sum_self / total_ms:6.2f}% "
        f"{sum(v['total_self_ms'] for v in warm_summary.values()):10.3f}\n"
    )
    out.write(
        f"in-process warm median total: {total_ms:.4f} ms over {warm['warm_calls']} calls "
        f"(sum-of-medians / median-total = {100.0 * sum_self / total_ms:.2f}%)\n"
    )
    out.write(
        f"per-call reconciliation (sum(exclusive)/call wall): median "
        f"{per_call_reconcile['median_pct']:.2f}%  range "
        f"{min(per_call_reconcile['per_call_pct']):.2f}-{max(per_call_reconcile['per_call_pct']):.2f}%\n"
    )
    for phase, block in controls.items():
        out.write(f"\n--- control: {phase} ---\n")
        table = block["table"]
        if table["median_total_ms"] is not None:
            out.write(
                f"median total {table['median_total_ms']:.4f} ms | median call "
                f"{table['median_call_ms']:.4f} ms | median post-sync "
                f"{table['median_post_sync_ms']:.4f} ms | first {table['first_call_ms']:.3f} ms\n"
            )
        for name, count, med, pct in block["components"]:
            out.write(f"{name:28s} {count:5d} {med:10.4f} {pct:6.2f}%\n")
        if block["fences"]:
            out.write("inline fences (mean ms booked inside their own span):\n")
            for name, ms in block["fences"].items():
                out.write(f"{name:28s} {'':>5s} {ms:10.4f}\n")
    callers = {k: v for k, v in profiler.callers.items() if k.startswith("external:")}
    if callers:
        out.write("\ncaller histograms (external buckets):\n")
        for name, hist in callers.items():
            out.write(f"{name}: {hist}\n")

    # ---- record ----------------------------------------------------------
    payload = {
        "probe": PROBE,
        "tag": tag,
        "stamp": pl.identity_stamp(
            evoxc,
            extra={
                "probe": PROBE,
                "tag": tag,
                "etl_head_probe": pl._run(["git", "rev-parse", "HEAD"], cwd=pl.ETL_REPO),
            },
        ),
        "config": {
            "fixture": os.path.abspath(args.fixture),
            "eval_reps": args.eval_reps,
            "sync_reps": args.sync_reps,
            "plain_reps": args.plain_reps,
            "cold_reps": args.cold_reps,
            "pop_size": int(forest.pop_size),
            "max_tree_len": int(forest.max_tree_len),
            "input_len": int(forest.input_len),
            "output_len": int(forest.output_len),
            "data_rows": int(x.shape[0]),
            "eval_kind": "fixture",
            "harness_call": "problem.evaluate(forest) (bench_gp.py:1374-1381)",
        },
        "route": {
            "backend": dense_engine.resolve_backend_name(),
            "eval_mode": dense_engine.resolve_eval_mode(),
            "should_use_dense": bool(dense_engine.should_use_dense(int(forest.pop_size))),
            "dense_calls": int(
                len(profiler.per_call_bucket_ms("sr_fitness_dense_self"))
            ),
            "plan": plan_sink,
        },
        "warm": {
            "table": warm,
            "rows": [r for r in rows if r["phase"] == "warm"],
            "components": [
                {
                    "component": name,
                    "count": count,
                    "median_self_ms": med,
                    "pct_of_total": pct,
                    "total_self_ms": warm_summary[name]["total_self_ms"],
                }
                for name, count, med, pct in components["warm"]
            ],
            "per_call_components": {
                name: profiler.per_call_bucket_ms(name, "warm")
                for name in profiler.buckets
            },
            "sum_exclusive_ms": sum_self,
            "reconciliation_pct": 100.0 * sum_self / total_ms,
            "reconciliation": per_call_reconcile,
            "counts": profiler.counts(),
        },        "controls": controls,
        "callers": callers,
        "cache_final": {"hits": int(cache.hits), "misses": int(cache.misses), "size": len(cache)},
        "probe_notes": profiler.notes,
        "fitness": {
            "value": rows[0]["fitness"] if rows else None,
            "sha256": rows[0]["fitness_sha256"] if rows else None,
            "all_equal": len({r["fitness_sha256"] for r in rows}) == 1,
        },
        "install_status": {
            "wrapped": profiler.install_status["wrapped"],
            "missing": profiler.install_status["missing"],
            "depth_overflow": profiler.depth_overflow,
        },
    }
    if args.json_out:
        pl.write_json(args.json_out, payload)
        out.write(f"\nrecord: {args.json_out}\nfitness: {payload['fitness']}\n")
    out.close()
    return 0


def build_case(args, out):
    """The harness wiring + a route sanity report (dense must actually be taken)."""
    import torch

    from evogp.tree import dense_engine

    torch.manual_seed(33)
    torch.cuda.manual_seed(33)
    problem, forest, x, labels = pl.build_eval_case(args.fixture)
    torch.cuda.synchronize()
    backend = dense_engine.resolve_backend_name()
    mode = dense_engine.resolve_eval_mode()
    use_dense = dense_engine.should_use_dense(int(forest.pop_size))
    out.write(
        f"route          : backend={backend} eval_mode={mode} "
        f"should_use_dense({forest.pop_size})={use_dense}\n"
    )
    if not use_dense:
        out.write(
            "  WARNING: the auto policy declines dense for this population -- the "
            "specialized route is out of Q1 scope; the component buckets will be empty.\n"
        )
    return problem, forest, x, labels


def run_cprofile(args, problem, forest, out):
    """The same loop with NO instrumentation under cProfile (ranking only)."""
    import cProfile
    import pstats

    import torch

    torch.cuda.synchronize()
    problem.evaluate(forest)  # cold build outside the profile
    torch.cuda.synchronize()
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(args.eval_reps):
        torch.cuda.synchronize()
        problem.evaluate(forest)
        torch.cuda.synchronize()
    profile.disable()
    stream = io.StringIO()
    for sort in ("tottime", "cumtime"):
        stats = pstats.Stats(profile, stream=stream).sort_stats(sort)
        stream.write(f"\n=== cProfile top-40 by {sort} (25 warm calls, instrumentation OFF) ===\n")
        stats.print_stats(40)
    text = stream.getvalue()
    out.write(text)
    if args.cprofile_out:
        with open(args.cprofile_out, "w") as handle:
            handle.write(text)
    return {"cprofile_text_written": args.cprofile_out}


if __name__ == "__main__":
    sys.exit(main())
