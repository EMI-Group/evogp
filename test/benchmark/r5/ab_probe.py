#!/usr/bin/env python
"""``ab_probe.py`` -- R5 Q3: engine-call vs frozen-runner, same process, same artifact.

The question this resolves: Q1 measured the engine's in-process ``evoxc.run`` at
2.13 / 2.37 ms (f64 / f200) while the R4/Q4 *frozen* pure forward of the same
population measured 1.21 / 1.46 ms -- two numbers that describe the same compiled
graph call.  Q3 puts the two paths in ONE process on ONE population and one
compiled artifact, and separates what actually differs:

* **payload freshness** -- does ``evoxc.run`` cost more when the payloads it consumes
  were uploaded immediately before (the engine's per-call behaviour) than when they
  are long-staged (the frozen runner's behaviour)?  (``A1-A2``, ``L1-L2``)
* **the run path itself** -- the engine's own ``_iter_plan_chunks`` seam vs the frozen
  runner's ``run()``.  (``A2-A3``)
* **upload cost in runner terms** -- the frozen runner's plan/executable fed with
  fresh payloads.  (``A4-A3``)
* **the graph variant** -- the loss artifact (the engine's SR-fitness route) vs the
  value artifact (the artifact the R4 frozen pure number measured).  (``L2-A3``)
* the explicit drain control: fresh payloads + a fence between staging and run, so a
  copy-drain tail can be told apart from a genuinely more expensive run.  (``R2-R1``)

Arms (one process per fixture; every arm is 25 harness-shaped calls, call 0 cold):

==============  ======================================================================
``A1_engine_fresh``       engine seam, VALUE artifact, fresh payloads per call
``A2_engine_staged``      engine seam, VALUE artifact, staged payloads
``A3_frozen_runner``      ``build_dense_runner(forest, x).run()`` (VALUE artifact)
``A4_frozen_fresh``       frozen runner's plan/executable/inputs, fresh payloads
``L1_engine_fresh_loss``  engine seam, LOSS artifact, fresh payloads (the real route)
``L2_engine_staged_loss`` engine seam, LOSS artifact, staged payloads
``R1_repl_fresh_loss``    probe-replicated fresh loop (replication control vs L1)
``R2_repl_fresh_drained`` the same loop with an explicit fence before ``evoxc.run``
``E1_e2e_in_process``     ``problem.evaluate(forest)`` -- the in-process e2e companion
==============  ======================================================================

Every artifact is built once (cache miss) and every later build is a content-key cache
hit; the arms hold the plan/executable/args directly, so no arm pays a per-call key
derivation -- that cost is Q6's ``M4V`` -> ``M5`` delta, and is cited, not re-measured.

Analysis-only: no ``src/evogp`` edit, no write in the foreign ``evoxc`` checkout, every
patch is a runtime monkeypatch restored before the next arm.

Usage (repo root, project venv, GPU; one locked process per fixture)::

    PYTHONPATH=src EVOXC_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      EVOGP_EVOXC_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_EVOXC_IREE_DEVICE=0 \
      EVOGP_EVOXC_IREE_OPT_LEVEL=O1 \
      <venv>/bin/python test/benchmark/r5/ab_probe.py \
        --fixture test/goldens/forest_random.npz --reps 25 --tag ab_f64 \
        --json-out test/benchmark/r5/results/q3_ab/ab_f64.json \
        --log-out  test/benchmark/r5/results/q3_ab/ab_f64.log
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import probe_lib as pl  # noqa: E402

PROBE = "r5/ab_probe.py"

#: Execution order (the two artifacts are compiled on their first arm's call 0).
ARM_ORDER = (
    "A1_engine_fresh",
    "A2_engine_staged",
    "A3_frozen_runner",
    "A3_plain_control",
    "A4_frozen_fresh",
    "L1_engine_fresh_loss",
    "L2_engine_staged_loss",
    "R1_repl_fresh_loss",
    "R2_repl_fresh_drained",
    "E1_e2e_in_process",
)

ARM_DOCS = {
    "A1_engine_fresh": "engine seam (_iter_plan_chunks), value artifact, FRESH payloads per call",
    "A2_engine_staged": "engine seam, value artifact, STAGED payloads",
    "A3_frozen_runner": "frozen build_dense_runner(forest, x).run() on staged inputs",
    "A3_plain_control": "A3 with the instrumentation inert (probe-overhead control)",
    "A4_frozen_fresh": "frozen runner's plan/executable/inputs, FRESH payloads",
    "L1_engine_fresh_loss": "engine seam, LOSS artifact, FRESH payloads (the SR route's graph)",
    "L2_engine_staged_loss": "engine seam, LOSS artifact, STAGED payloads",
    "R1_repl_fresh_loss": "probe-replicated fresh loop + evoxc.run (replication control)",
    "R2_repl_fresh_drained": "probe-replicated fresh loop + explicit fence + evoxc.run",
    "E1_e2e_in_process": "problem.evaluate(forest) -- the in-process e2e companion",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="R5 Q3 engine-call vs frozen-runner same-process A/B (analysis-only)"
    )
    parser.add_argument("--fixture", required=True, help="the eval fixture .npz")
    parser.add_argument("--reps", type=int, default=25, help="calls per arm (call 0 cold)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def engine_loop(plan, executable, xt, label, staged, dense_engine, backend_name, device):
    """One engine-seam call: iterate the chunk iterator and drop the raw result."""

    def call():
        for _chunk, _raw in dense_engine._iter_plan_chunks(
            plan,
            executable,
            xt,
            label,
            backend_name=backend_name,
            device=device,
            staged=staged,
        ):
            pass
        return None

    return call


def replicated_loss_args(evoxc, dense_engine, plan, xt, label, device):
    """The exact loss-variant argument tuple ``_iter_plan_chunks`` would build, fresh.

    Mirrors ``dense_engine._iter_plan_chunks`` for a single-chunk plan: the 7 node
    columns (``_prep_columns`` -> ``_prep_column``), the x / labels chunk rows
    (``_chunk_rows``), the padding keep-mask (``_keep_mask``) and the static ``cfg``,
    each wrapped and uploaded by hand -- the probe-side control arm, never the
    measured engine path.
    """
    n_chunk = min(plan.n_chunk, plan.n_rows)
    columns = dense_engine._prep_columns(plan.prep, 0, plan.p_chunk)
    xin = dense_engine._chunk_rows(xt, 0, n_chunk, plan.n_chunk)
    lin = dense_engine._chunk_rows(label, 0, n_chunk, plan.n_chunk)
    keep = dense_engine._keep_mask(plan.n_chunk, n_chunk)
    return tuple(evoxc.from_numpy(array).to(device) for array in columns) + (
        evoxc.from_numpy(xin).to(device),
        evoxc.from_numpy(lin).to(device),
        evoxc.from_numpy(keep).to(device),
        plan.cfg,
    )


def artifact_block(plan, executable, extra=None):
    """Identity/description of one compiled artifact (JSON-safe)."""
    block = {
        "variant": str(getattr(plan, "variant", None)),
        "trace_len": int(getattr(plan, "trace_len", -1)),
        "pop_size": int(getattr(plan, "pop_size", -1)),
        "n_rows": int(getattr(plan, "n_rows", -1)),
        "p_chunks": int(getattr(plan, "p_chunks", -1)),
        "n_chunks": int(getattr(plan, "n_chunks", -1)),
        "p_chunk": int(getattr(plan, "p_chunk", -1)),
        "n_chunk": int(getattr(plan, "n_chunk", -1)),
        "per_row": bool(getattr(plan, "per_row", False)),
        "executable_type": type(executable).__name__,
    }
    if extra:
        block.update(extra)
    return block


def main(argv=None):
    args = parse_args(argv)
    if args.reps < 2:
        raise SystemExit("--reps must be >= 2 (call 0 is the cold call and is excluded)")
    tag = args.tag or os.path.splitext(os.path.basename(args.fixture))[0]
    out = pl.Tee(args.log_out)

    import numpy as np
    import torch

    import evoxc
    from evogp.problem import SymbolicRegression
    from evogp.tree import dense_engine

    import evogp.tree.forest as forest_module

    torch.manual_seed(33)
    torch.cuda.manual_seed(33)
    problem, forest, x, labels = pl.build_eval_case(args.fixture)
    torch.cuda.synchronize()
    backend_name, device, _options = dense_engine._resolve_backend(None, None)
    use_dense = dense_engine.should_use_dense(int(forest.pop_size))

    profiler = pl.Profiler(enabled=True)
    profiler.flags = {"sync_region": False}
    wraps = pl.install_dense_wraps(
        dense_engine, forest_module, SymbolicRegression, profiler
    )
    proxy = pl.install_etl_proxy(dense_engine, profiler)
    pl.patch_etl_tensor_to(evoxc, profiler)
    host_load_start = pl.host_load()
    # The fresh arms' payload construction (fresh / replication / drain arms only).
    for bucket, name in (("chunk_rows", "_chunk_rows"), ("keep_mask", "_keep_mask")):
        pl.patch_attr(dense_engine, name, bucket, profiler, required=False)
    cache = dense_engine.DENSE_CACHE

    out.write(f"=== {PROBE}: tag={tag} ===\n")
    out.write(f"fixture      : {os.path.abspath(args.fixture)}\n")
    out.write(
        f"route        : backend={backend_name} eval_mode={dense_engine.resolve_eval_mode()} "
        f"should_use_dense({forest.pop_size})={use_dense}\n"
    )
    out.write(
        f"case         : Forest(pop={forest.pop_size}, max_tree_len={forest.max_tree_len}, "
        f"input_len={forest.input_len}, output_len={forest.output_len}) + "
        f"SymbolicRegression(datapoints=x{x.shape}, labels=labels{labels.shape})\n"
    )
    out.write(f"evoxc        : {evoxc.__file__}\n")
    out.write(f"wrapped      : {', '.join(wraps['wrapped'])}\n")
    out.write(f"missing      : {wraps['missing'] or 'none'}\n")
    out.write(
        f"cache        : hits={cache.hits} misses={cache.misses} size={len(cache)}\n\n"
    )

    # ---- the two artifacts (one compile each, on their first arm's call 0) -----
    plan_args_value = (
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        int(forest.input_len),
        int(forest.output_len),
        int(x.shape[0]),
    )
    plan_kwargs_value = {"per_row": False, "backend": None}
    plan_kwargs_loss = {
        "per_row": False,
        "variant": dense_engine.LOSS_VARIANT,
        "use_mse": True,
        "backend": None,
    }

    def build_artifact(kwargs):
        torch.cuda.synchronize()
        t0 = pl.NOW()
        hits_before, misses_before = int(cache.hits), int(cache.misses)
        plan = dense_engine.plan_dense(*plan_args_value, **kwargs)
        executable = dense_engine.build_dense(plan)
        torch.cuda.synchronize()
        return {
            "plan": plan,
            "executable": executable,
            "build_ms": (pl.NOW() - t0) * 1e3,
            "cache_hits": int(cache.hits) - hits_before,
            "cache_misses": int(cache.misses) - misses_before,
        }

    out.write("-- building the two artifacts (value first: it is also the M5/Q6 graph) --\n")
    value = build_artifact(plan_kwargs_value)
    loss = build_artifact(plan_kwargs_loss)
    value_plan, value_executable = value["plan"], value["executable"]
    loss_plan, loss_executable = loss["plan"], loss["executable"]
    out.write(f"   value artifact: {artifact_block(value_plan, value_executable)} build {value['build_ms']:.1f} ms\n")
    out.write(f"   loss  artifact: {artifact_block(loss_plan, loss_executable)} build {loss['build_ms']:.1f} ms\n")

    # host inputs (one conversion for the whole probe: a fixed x/labels)
    xt = dense_engine._input_to_numpy(x, int(forest.input_len), "inputs")
    label_host = dense_engine._as_numpy(labels, np.float32, "labels")

    # staged payload memos, filled out of band (one engine pass each)
    staged_value, staged_loss = {}, {}
    for _chunk, _raw in dense_engine._iter_plan_chunks(
        value_plan, value_executable, xt, None,
        backend_name=backend_name, device=device, staged=staged_value,
    ):
        pass
    for _chunk, _raw in dense_engine._iter_plan_chunks(
        loss_plan, loss_executable, xt, label_host,
        backend_name=backend_name, device=device, staged=staged_loss,
    ):
        pass
    torch.cuda.synchronize()
    out.write(
        f"   staged payloads: value={len(staged_value)} loss={len(staged_loss)} "
        f"(filled out of band)\n\n"
    )

    # ---- the frozen runner (its build is a content-key cache hit on the value graph)
    out.write("-- frozen runner build (build_dense_runner(forest, x)) --\n")
    torch.cuda.synchronize()
    t0 = pl.NOW()
    runner = dense_engine.build_dense_runner(forest, x)
    torch.cuda.synchronize()
    runner_build_ms = (pl.NOW() - t0) * 1e3
    runner_identity = {
        "executable_is_value_artifact": bool(runner.executable is value_executable),
        "plan_variant": str(getattr(runner.plan, "variant", None)),
        "plan_trace_len": int(getattr(runner.plan, "trace_len", -1)),
        "staged_payloads": len(runner.staged),
        "n_graph_calls_per_run": int(getattr(runner, "invocations", -1)),
        "construct_total_ms": runner_build_ms,
        "backend": runner.backend_name,
        "device_resident": bool(runner.device_resident),
    }
    out.write(
        f"   built in {runner_build_ms:.1f} ms; artifact identity "
        f"(runner.executable is the value artifact): {runner_identity['executable_is_value_artifact']}\n\n"
    )

    # ---- arms -------------------------------------------------------------
    arms = {}
    order = []
    rows_all = []

    def run_arm(name, fn, reps=args.reps, plain=False):
        first = len(rows_all)
        for index in range(reps):
            rows_all.append(
                pl.bench_call(
                    fn, profiler=None if plain else profiler, index=index, phase=name
                )
            )
        rows = rows_all[first:]
        if plain:
            summary, per_call = {}, {}
        else:
            summary = profiler.summary(name, skip_index0=True)
            per_call = {
                bucket: profiler.per_call_bucket_ms(bucket, name) for bucket in profiler.buckets
            }
        counts = {bucket: entry["count"] for bucket, entry in summary.items() if entry["count"]}
        run_entry = summary.get("etl_run")
        run_span = run_entry["median_self_ms"] if run_entry and run_entry["count"] else None
        arms[name] = {
            "doc": ARM_DOCS[name],
            "plain": bool(plain),
            "stats": pl.arm_stats(rows),
            "rows": rows,
            "etl_run_span_ms": run_span,
            "components": [
                {
                    "component": bucket,
                    "count": entry["count"],
                    "median_self_ms": entry["median_self_ms"],
                }
                for bucket, entry in sorted(summary.items())
                if entry["count"]
            ],
            "per_call_components": per_call,
            "counts": counts,
            "cache": {"hits": int(cache.hits), "misses": int(cache.misses), "size": len(cache)},
        }
        order.append(name)
        stats = arms[name]["stats"]
        out.write(
            f"{name:24s} {stats['median_warm_ms']:8.4f} ms  (min {stats['min_warm_ms']:.4f} / "
            f"max {stats['max_warm_ms']:.4f})  etl_run "
            f"{'n/a' if run_span is None else f'{run_span:.4f}'} ms  "
            f"first {stats['first_call_ms']:.1f} ms\n"
        )
        out.write(f"      counts: {', '.join(f'{k}={v}' for k, v in sorted(counts.items())[:10])}\n")
        return arms[name]

    run_arm(
        "A1_engine_fresh",
        engine_loop(value_plan, value_executable, xt, None, None, dense_engine, backend_name, device),
    )
    run_arm(
        "A2_engine_staged",
        engine_loop(value_plan, value_executable, xt, None, staged_value, dense_engine, backend_name, device),
    )
    run_arm("A3_frozen_runner", lambda: runner.run())
    profiler.enabled = False
    run_arm("A3_plain_control", lambda: runner.run(), reps=6, plain=True)
    profiler.enabled = True
    run_arm(
        "A4_frozen_fresh",
        engine_loop(runner.plan, runner.executable, runner._x, None, None, dense_engine, backend_name, device),
    )
    run_arm(
        "L1_engine_fresh_loss",
        engine_loop(loss_plan, loss_executable, xt, label_host, None, dense_engine, backend_name, device),
    )
    run_arm(
        "L2_engine_staged_loss",
        engine_loop(loss_plan, loss_executable, xt, label_host, staged_loss, dense_engine, backend_name, device),
    )

    def repl_fresh(drain):
        def call():
            args = replicated_loss_args(proxy, dense_engine, loss_plan, xt, label_host, device)
            if drain:
                with profiler.span("fence:drain"):
                    torch.cuda.synchronize()
            proxy.run(loss_executable, *args)
            return None

        return call

    run_arm("R1_repl_fresh_loss", repl_fresh(False))
    run_arm("R2_repl_fresh_drained", repl_fresh(True))
    run_arm("E1_e2e_in_process", lambda: problem.evaluate(forest))

    # ---- deltas -----------------------------------------------------------
    def delta(name, high, low, metric, meaning, label):
        def value(arm):
            if metric == "etl_run":
                return arm["etl_run_span_ms"]
            return arm["stats"]["median_warm_ms"]

        high_value = value(arms[high])
        low_value = value(arms[low])
        return {
            "name": name,
            "high": high,
            "low": low,
            "metric": metric,
            "median_high": high_value,
            "median_low": low_value,
            "delta_ms": (
                None if high_value is None or low_value is None else high_value - low_value
            ),
            "meaning": meaning,
            "evidence": label,
        }

    deltas = [
        delta("A1-A2 total", "A1_engine_fresh", "A2_engine_staged", "median_warm_ms",
              "cost of building/uploading the payloads fresh, per call (prep+upload+semaphores)",
              "measured"),
        delta("A1-A2 run span", "A1_engine_fresh", "A2_engine_staged", "etl_run",
              "payload-freshness cost INSIDE evoxc.run (fresh buffers vs staged buffers)",
              "measured"),
        delta("A2-A3 total", "A2_engine_staged", "A3_frozen_runner", "median_warm_ms",
              "engine-seam staged call vs frozen runner.run() (same staged artifact)",
              "measured"),
        delta("A2-A3 run span", "A2_engine_staged", "A3_frozen_runner", "etl_run",
              "the run path itself (engine seam vs frozen run(): same evoxc.run, same args)",
              "measured"),
        delta("A4-A3 total", "A4_frozen_fresh", "A3_frozen_runner", "median_warm_ms",
              "upload cost in runner terms (frozen plan/executable fed with fresh payloads)",
              "measured"),
        delta("A4-A3 run span", "A4_frozen_fresh", "A3_frozen_runner", "etl_run",
              "the same freshness question on the frozen artifact",
              "measured"),
        delta("L1-L2 total", "L1_engine_fresh_loss", "L2_engine_staged_loss", "median_warm_ms",
              "fresh-vs-staged payload cost on the LOSS artifact (the engine's real route)",
              "measured"),
        delta("L1-L2 run span", "L1_engine_fresh_loss", "L2_engine_staged_loss", "etl_run",
              "freshness cost inside evoxc.run on the LOSS artifact",
              "measured"),
        delta("L2-A3 run span", "L2_engine_staged_loss", "A3_frozen_runner", "etl_run",
              "LOSS artifact staged run vs VALUE artifact staged run (in-graph loss vs values)",
              "measured"),
        delta("R1-L1 total", "R1_repl_fresh_loss", "L1_engine_fresh_loss", "median_warm_ms",
              "replication fidelity of the probe-side fresh loop (must be ~0)",
              "measured"),
        delta("R2-R1 total", "R2_repl_fresh_drained", "R1_repl_fresh_loss", "median_warm_ms",
              "cost of an explicit fence between staging and the run (the drain that evoxc.run absorbs)",
              "measured"),
        delta("R2-R1 run span", "R2_repl_fresh_drained", "R1_repl_fresh_loss", "etl_run",
              "how much of evoxc.run's fresh-payload cost is the staging copy's drain",
              "measured"),
        delta("E1-A3 total", "E1_e2e_in_process", "A3_frozen_runner", "median_warm_ms",
              "in-process e2e evaluate minus frozen pure run (the e2e_companion analogue)",
              "measured"),
        delta("E1-L2 total", "E1_e2e_in_process", "L2_engine_staged_loss", "median_warm_ms",
              "in-process e2e minus the staged loss run: the per-call host glue + prologue around the run",
              "measured"),
        delta("A3-A3plain overhead", "A3_frozen_runner", "A3_plain_control", "median_warm_ms",
              "probe instrumentation overhead on the frozen run() path (plain control)",
              "measured"),
    ]

    # ---- record -----------------------------------------------------------
    host_load_end = pl.host_load()
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
        "host_load_start": host_load_start,
        "host_load_end": host_load_end,
        "config": {
            "fixture": os.path.abspath(args.fixture),
            "reps": args.reps,
            "pop_size": int(forest.pop_size),
            "max_tree_len": int(forest.max_tree_len),
            "input_len": int(forest.input_len),
            "output_len": int(forest.output_len),
            "data_rows": int(x.shape[0]),
            "eval_kind": "fixture",
            "arm_order": list(ARM_ORDER),
            "value_plan_kwargs": {k: str(v) for k, v in plan_kwargs_value.items()},
            "loss_plan_kwargs": {k: str(v) for k, v in plan_kwargs_loss.items()},
        },
        "route": {
            "backend": backend_name,
            "eval_mode": dense_engine.resolve_eval_mode(),
            "should_use_dense": bool(use_dense),
            "artifacts": {
                "value": artifact_block(value_plan, value_executable, {"build_ms": value["build_ms"], "cache": {"hits": value["cache_hits"], "misses": value["cache_misses"]}}),
                "loss": artifact_block(loss_plan, loss_executable, {"build_ms": loss["build_ms"], "cache": {"hits": loss["cache_hits"], "misses": loss["cache_misses"]}}),
            },
            "staged_payloads": {"value": len(staged_value), "loss": len(staged_loss)},
            "runner_identity": runner_identity,
        },
        "arms": arms,
        "deltas": deltas,
        "cache_final": {"hits": int(cache.hits), "misses": int(cache.misses), "size": len(cache)},
        "probe_notes": profiler.notes,
        "install_status": {
            "wrapped": profiler.install_status["wrapped"],
            "missing": profiler.install_status["missing"],
            "depth_overflow": profiler.depth_overflow,
        },
    }

    # ---- log report -------------------------------------------------------
    out.write("\n=== arm table (harness-shaped warm median, call 0 excluded) ===\n")
    out.write(f"{'arm':24s} {'total_ms':>10s} {'etl_run_ms':>11s}  doc\n")
    for name in order:
        span = arms[name]["etl_run_span_ms"]
        out.write(
            f"{name:24s} {arms[name]['stats']['median_warm_ms']:10.4f} "
            f"{'n/a' if span is None else f'{span:11.4f}'}  {ARM_DOCS[name]}\n"
        )
    out.write("\n=== deltas (median warm ms; run spans are the evoxc.run exclusive bucket) ===\n")
    out.write(f"{'delta':20s} {'metric':12s} {'delta_ms':>9s} {'label':>9s}  meaning\n")
    for entry in deltas:
        value = "n/a" if entry["delta_ms"] is None else f"{entry['delta_ms']:.4f}"
        out.write(
            f"{entry['name']:20s} {entry['metric']:12s} {value:>9s} {entry['evidence']:>9s}  "
            f"{entry['meaning']}\n"
        )

    if args.json_out:
        pl.write_json(args.json_out, payload)
        out.write(f"\nrecord: {args.json_out}\n")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
