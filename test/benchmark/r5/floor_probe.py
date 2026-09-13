#!/usr/bin/env python
"""``floor_probe.py`` -- R5 Q6: controlled-removal floor matrix of ONE warm dense evaluate.

The question: the same warm ``problem.evaluate(forest)`` that Q1 decomposes into
~7.3 ms (f64) / ~6.6 ms (f200) of bookkeeping, and whose *pure* compiled forward is
~1.2-1.5 ms (Q4), is 8-11x the legacy CUDA engine's entire evaluate (Q4 window).
Q6 removes one measured component at a time -- from the very same call -- and prices
each removal, so the decomposition stops being an attribution of *timers* and becomes
a set of *floors* that can be quoted and falsified.

The chain (one process per fixture; every arm is the SAME run/readback code with one
more component memoized, pre-staged or dropped):

============  ==========================================================================
``M0``        full ``problem.evaluate(forest)`` -- the harness call, call 0 cold
``M1``        M0 with ``plan_dense`` memoized  (decode + structural validation +
              ``np.unique`` + plan construction elided on warm calls)
``M2``        M1 with ``_prep_column`` memoized (the 7 node-column host preparations
              elided on warm calls)
``M3``        M2 with the payloads pre-staged (the loss loop runs against a staged
              payload memo -- no per-call ``etl.from_numpy``/``.to(device)``, no copy
              semaphores)
``M4``        M3 without the readback and the host assembly (the loop stops after
              ``etl.run``; the device-resident loss is dropped, as a pure forward does)
``M4V``       M4 on the VALUE variant artifact (the graph the frozen pure runner builds)
              -- same prologue, no labels payload, no in-graph loss
``M5``        the frozen runner's ``run()`` on the same staged inputs
              (``build_dense_runner(...).run()``: no per-call plan/build/backend
              prologue at all, no readback, no host assembly)
``M0R``       M0 again after every patch is restored (drift bracket, closing control)
``M3R``       M3 again (second drift-bracket point)
``M0P``       M0 with the instrumentation inert (probe-overhead control)
============  ==========================================================================

Definitions of record (what each floor removes) are printed in the log and recorded
under ``floors`` in the JSON.  The per-step deltas are compared against Q1's timer
attribution (``results/q1_decomp/SUMMARY.md``) -- that is the falsification function:
a delta that does not match its predicted bucket sum is investigated and said so.

Analysis-only: nothing here edits ``src/evogp`` or the foreign ``etl`` checkout; every
step is a runtime monkeypatch (``probe_lib.PatchStack``), and any runtime patch is
restored before the closing controls.

Usage (repo root, project venv, GPU; one locked process per fixture)::

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 \
      EVOGP_ETL_IREE_OPT_LEVEL=O1 \
      <venv>/bin/python test/benchmark/r5/floor_probe.py \
        --fixture test/goldens/forest_random.npz --reps 25 --tag floor_f64 \
        --json-out test/benchmark/r5/results/q6_floors/floor_f64.json \
        --log-out  test/benchmark/r5/results/q6_floors/floor_f64.log
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import probe_lib as pl  # noqa: E402

PROBE = "r5/floor_probe.py"

#: Execution order of the arms (the chain is cumulative; the last three are controls).
ARM_ORDER = ("M0", "M1", "M2", "M3", "M4", "M4V", "M5", "M0R", "M3R", "M0P")

#: One line per arm: what the arm is, and what it removes relative to the previous step.
ARM_DOCS = {
    "M0": (
        "full problem.evaluate(forest) -- the harness call (warm-cache: prelude)",
        "-",
    ),
    "M1": (
        "M0 + memoized plan_dense (decode/_decode_columns + np.unique + plan build)",
        "plan_dense (decode + unique + plan construction)",
    ),
    "M2": ("M1 + memoized _prep_column (7 node-column host preparations)", "_prep_column x7"),
    "M3": (
        "M2 + payloads pre-staged (memo injected into _iter_plan_chunks)",
        "etl.from_numpy + Tensor.to(device) for all 10 payloads (uploads + copy semaphores)",
    ),
    "M4": (
        "M3 without readback/host assembly (stop after etl.run)",
        "_to_host D2H + run_loss_plan host assembly + fitness H2D",
    ),
    "M4V": (
        "M4 on the VALUE artifact (the frozen pure runner's graph)",
        "the loss variant + the labels/keep payloads",
    ),
    "M5": (
        "frozen build_dense_runner(forest, x).run() -- no per-call prologue",
        "plan_dense memo lookup + build_dense cache prologue + _resolve_backend + x/labels host copies",
    ),
    "M0R": ("M0 again, all patches restored (drift bracket)", "-"),
    "M3R": ("M3 again (drift bracket)", "-"),
    "M0P": ("M0 with the instrumentation inert (probe-overhead control)", "-"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="R5 Q6 controlled-removal floor matrix (analysis-only)"
    )
    parser.add_argument("--fixture", required=True, help="the eval fixture .npz")
    parser.add_argument("--reps", type=int, default=25, help="calls per arm (call 0 cold)")
    parser.add_argument("--control-reps", type=int, default=5, help="calls for the M0P control")
    parser.add_argument("--micro-reps", type=int, default=20, help="upload micro-probe repetitions")
    parser.add_argument("--no-micro", action="store_true", help="skip the upload micro-probe")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# arm callables (all runtime patches; nothing in src/ is touched)
# ---------------------------------------------------------------------------


def install_plan_memo(dense_engine, patches, captured):
    """M1: memoize ``plan_dense`` on its own arguments (the plan is call-invariant here)."""
    real = dense_engine.plan_dense
    memo = {}

    def plan_dense_cached(*args, **kwargs):
        key = (pl.hashable_key(args), pl.hashable_key(kwargs))
        plan = memo.get(key)
        if plan is None:
            plan = memo[key] = real(*args, **kwargs)
            captured.setdefault("plan_call", (args, kwargs))
        return plan

    patches.set(dense_engine, "plan_dense", plan_dense_cached, "M1 plan_dense memo")
    return memo


def install_column_memo(dense_engine, patches):
    """M2: memoize ``_prep_column`` (host column slices depend only on the population)."""
    real = dense_engine._prep_column
    memo = {}

    def prep_column_cached(prep, index, p0, p_chunk):
        key = (id(prep), index, p0, p_chunk)
        array = memo.get(key)
        if array is None:
            array = memo[key] = real(prep, index, p0, p_chunk)
        return array

    patches.set(dense_engine, "_prep_column", prep_column_cached, "M2 _prep_column memo")
    return memo


def install_staged_seam(dense_engine, patches, staged):
    """M3/M4: inject the probe-owned payload memo into the engine's own chunk iterator."""
    real = dense_engine._iter_plan_chunks

    def iter_plan_chunks_staged(*args, **kwargs):
        if kwargs.get("staged") is None:
            kwargs["staged"] = staged
        return real(*args, **kwargs)

    patches.set(
        dense_engine, "_iter_plan_chunks", iter_plan_chunks_staged, "M3 staged payload seam"
    )
    return real


def install_build_capture(dense_engine, patches, captured):
    """Capture the plan/executable the engine builds (zero behavioural effect)."""
    real = dense_engine.build_dense

    def build_dense_captured(plan, *args, **kwargs):
        executable = real(plan, *args, **kwargs)
        captured.setdefault("plan_from_build", plan)
        captured["executable"] = executable
        return executable

    patches.set(dense_engine, "build_dense", build_dense_captured, "capture build_dense")
    return real


def make_m4(problem, forest, dense_engine):
    """M4 / M4V: the M3 prologue, the staged run, and then NOTHING (no readback).

    ``verify=True`` is the out-of-band correctness control: the same loop *with* the
    readback and the engine's own host assembly (``run_loss_plan``'s formula), so the
    fitness of the M4 code path can be compared to M0's byte for byte.  Never timed.
    """
    import numpy as np
    import torch

    def make_call(plan_call, staged_memo, with_labels):
        def call(verify=False):
            args, kwargs = plan_call
            xt = dense_engine._input_to_numpy(
                problem.datapoints, int(forest.input_len), "inputs"
            )
            label = (
                dense_engine._as_numpy(problem.labels, np.float32, "labels")
                if with_labels
                else None
            )
            plan = dense_engine.plan_dense(*args, **kwargs)  # memo hit (M1 patch)
            executable = dense_engine.build_dense(plan)  # cache hit (one artifact)
            backend_name, device, _ = dense_engine._resolve_backend(None, None)
            if not verify:
                for _chunk, _raw in dense_engine._iter_plan_chunks(
                    plan,
                    executable,
                    xt,
                    label,
                    backend_name=backend_name,
                    device=device,
                    staged=staged_memo,
                ):
                    pass  # device-resident result dropped: no readback, no assembly
                return None
            totals = np.zeros((plan.prep.pop_size,), dtype=np.float32)
            for (p0, p_chunk, _n0, _n_chunk), raw in dense_engine._iter_plan_chunks(
                plan,
                executable,
                xt,
                label,
                backend_name=backend_name,
                device=device,
                staged=staged_memo,
            ):
                host = dense_engine._to_host(raw, backend_name).reshape(plan.p_chunk)
                totals[p0 : p0 + p_chunk] += host[:p_chunk]
            loss = totals / np.float32(plan.n_rows)
            # ``SymbolicRegression.evaluate`` returns the NEGATED loss (the pipeline
            # maximizes fitness), so the control returns the same sign as M0..M3.
            return (
                -torch.from_numpy(loss)
                .contiguous()
                .detach()
                .requires_grad_(False)
                .to(problem.datapoints.device)
            )

        return call

    return make_call


def build_payload_hosts(dense_engine, plan, xt, label):
    """The 10 loss-variant host payload arrays, freshly prepared by the engine helpers."""
    import numpy as np

    assert plan.p_chunks == 1 and plan.n_chunks == 1, "single-chunk plan assumed"
    n_chunk = min(plan.n_chunk, plan.n_rows)
    hosts = [
        (f"column:{name}", dense_engine._prep_column(plan.prep, index, 0, plan.p_chunk))
        for index, name in enumerate(dense_engine._COLUMN_NAMES)
    ]
    hosts.append(("x", dense_engine._chunk_rows(xt, 0, n_chunk, plan.n_chunk)))
    hosts.append(("labels", dense_engine._chunk_rows(label, 0, n_chunk, plan.n_chunk)))
    hosts.append(("keep", dense_engine._keep_mask(plan.n_chunk, n_chunk)))
    return [(name, np.ascontiguousarray(array)) for name, array in hosts]


def upload_micro_probe(etl, device, hosts, reps, out):
    """Per-fixture upload micro-benchmark (batch + per-payload-isolated, engine payloads)."""
    import torch

    sizes = [(name, list(array.shape), int(array.nbytes)) for name, array in hosts]
    out.write("\n-- upload micro-probe (fresh etl.from_numpy(...).to(device) per payload) --\n")
    for name, shape, nbytes in sizes:
        out.write(f"   {name:12s} shape={shape} bytes={nbytes}\n")
    batch_ms = []
    per_index = {index: [] for index in range(len(hosts))}
    for _rep in range(reps):
        torch.cuda.synchronize()
        t0 = pl.NOW()
        tensors = [etl.from_numpy(array).to(device) for _name, array in hosts]
        torch.cuda.synchronize()
        batch_ms.append((pl.NOW() - t0) * 1e3)
        del tensors
        for index, (_name, array) in enumerate(hosts):
            t0 = pl.NOW()
            tensor = etl.from_numpy(array).to(device)
            torch.cuda.synchronize()
            per_index[index].append((pl.NOW() - t0) * 1e3)
            del tensor
    isolated_ms = [pl.median(per_index[index]) for index in range(len(hosts))]
    out.write(
        f"   batch-of-{len(hosts)} median {pl.median(batch_ms):.4f} ms "
        f"(min {min(batch_ms):.4f} / max {max(batch_ms):.4f}, {reps} reps)\n"
    )
    for index, (name, _shape, nbytes) in enumerate(sizes):
        out.write(
            f"   isolated {name:12s} median {isolated_ms[index]:.4f} ms  "
            f"({isolated_ms[index] * 1e3 / max(1, nbytes):.4f} us/KB)\n"
        )
    return {
        "payloads": [{"name": name, "shape": shape, "bytes": nbytes} for name, shape, nbytes in sizes],
        "batch_ms": batch_ms,
        "batch_median_ms": pl.median(batch_ms),
        "isolated_median_ms": isolated_ms,
        "reps": reps,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.reps < 2:
        raise SystemExit("--reps must be >= 2 (call 0 is the cold call and is excluded)")
    tag = args.tag or os.path.splitext(os.path.basename(args.fixture))[0]
    out = pl.Tee(args.log_out)

    import numpy as np
    import torch

    import etl
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
    pl.install_etl_proxy(dense_engine, profiler)
    pl.patch_etl_tensor_to(etl, profiler)
    plan_sink = {}
    pl.install_plan_capture(dense_engine, plan_sink, profiler)
    cache = dense_engine.DENSE_CACHE

    patches = pl.PatchStack()
    captured = {}
    install_build_capture(dense_engine, patches, captured)
    host_load_start = pl.host_load()

    staged = {}
    staged_value = {}
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
    out.write(f"etl          : {etl.__file__}\n")
    out.write(f"wrapped      : {', '.join(wraps['wrapped'])}\n")
    out.write(f"missing      : {wraps['missing'] or 'none'}\n")
    out.write(
        f"cache        : hits={cache.hits} misses={cache.misses} size={len(cache)}\n\n"
    )

    arms = {}
    order = []

    def run_arm(name, fn, reps, *, plain=False, verify=None):
        """One arm: ``reps`` harness-shaped calls, call 0 cold, last call fingerprint'd."""
        first = len(rows_all)
        use_profiler = None if plain else profiler
        for index in range(reps):
            rows_all.append(
                pl.bench_call(
                    fn,
                    profiler=use_profiler,
                    index=index,
                    phase=name,
                    keep_result=(index == reps - 1),
                )
            )
        rows = rows_all[first:]
        result = rows[-1].pop("_result", None)
        stats = pl.arm_stats(rows)
        if plain:
            summary, per_call = {}, {}
        else:
            summary = profiler.summary(name, skip_index0=True)
            per_call = {
                bucket: profiler.per_call_bucket_ms(bucket, name) for bucket in profiler.buckets
            }
        counts = {bucket: entry["count"] for bucket, entry in summary.items() if entry["count"]}
        fingerprint = pl.sha256_bytes(result) if result is not None else None
        arms[name] = {
            "doc": ARM_DOCS[name][0],
            "removes": ARM_DOCS[name][1],
            "patches_installed": list(patches.installed),
            "plain": bool(plain),
            "stats": stats,
            "rows": rows,
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
            "fitness_sha256": fingerprint,
            "verify_fitness_sha256": None,
            "loss_reference": None,
        }
        if verify is not None:
            verified = verify()
            arms[name]["verify_fitness_sha256"] = pl.sha256_bytes(verified)
            fingerprint = arms[name]["verify_fitness_sha256"]
        if fingerprint is not None and "M0" in arms:
            arms[name]["loss_reference"] = fingerprint == arms["M0"]["fitness_sha256"]
        order.append(name)
        out.write(
            f"{name:4s} {stats['median_warm_ms']:8.4f} ms warm median "
            f"(min {stats['min_warm_ms']:.4f} / max {stats['max_warm_ms']:.4f}, "
            f"{stats['warm_calls']} warm calls, first {stats['first_call_ms']:.3f})\n"
        )
        out.write(f"      removes: {ARM_DOCS[name][1]}\n")
        out.write(
            "      counts: "
            + ", ".join(f"{bucket}={count}" for bucket, count in sorted(counts.items())[:14])
            + "\n"
        )
        if fingerprint:
            out.write(
                f"      fitness sha256: {fingerprint[:16]}...  matches M0: "
                f"{arms[name]['loss_reference']}\n"
            )
        return arms[name]

    rows_all = []

    # ---- global prelude: pay the process's ONE cold compile out of band --
    # NOT a measured arm: one full evaluate + sync before M0, so M0's call 0 is a
    # warm-cache call and the ~11 s compile cannot leak into the chain's first arm.
    # Fingerprinted (it must equal M0's fitness), never timed as an arm.
    torch.cuda.synchronize()
    t0 = pl.NOW()
    prelude_result = problem.evaluate(forest)
    torch.cuda.synchronize()
    prelude_ms = (pl.NOW() - t0) * 1e3
    prelude = {
        "doc": (
            "seed/warm-up only, not a measured arm: one full problem.evaluate + sync "
            "so the process's single cold compile is paid before M0"
        ),
        "evaluate_ms": prelude_ms,
        "fitness_sha256": pl.sha256_bytes(prelude_result),
        "cache_after": {
            "hits": int(cache.hits),
            "misses": int(cache.misses),
            "size": len(cache),
        },
    }
    del prelude_result
    out.write(
        f"-- prelude (not an arm): one problem.evaluate + sync = {prelude_ms:.3f} ms, "
        f"fitness sha256 {(prelude['fitness_sha256'] or 'n/a')[:16]}... --\n\n"
    )

    # ---- M0: the full harness call ---------------------------------------
    out.write("-- M0: full problem.evaluate(forest) --\n")
    run_arm("M0", lambda: problem.evaluate(forest), args.reps)

    # ---- M1..M3: cumulative memos ---------------------------------------
    out.write("\n-- M1: + memoized plan_dense (decode/unique/plan) --\n")
    install_plan_memo(dense_engine, patches, captured)
    run_arm("M1", lambda: problem.evaluate(forest), args.reps)

    out.write("\n-- M2: + memoized _prep_column (node-column host prep) --\n")
    install_column_memo(dense_engine, patches)
    run_arm("M2", lambda: problem.evaluate(forest), args.reps)

    out.write("\n-- M3: + pre-staged payloads (no per-call uploads/semaphores) --\n")
    install_staged_seam(dense_engine, patches, staged)
    run_arm("M3", lambda: problem.evaluate(forest), args.reps)

    # ---- M4 / M4V: stop after the run ------------------------------------
    make_call = make_m4(problem, forest, dense_engine)
    m4_verify = make_call(captured["plan_call"], staged, with_labels=True)
    out.write("\n-- M4: M3 without readback + host assembly (stop after etl.run) --\n")
    run_arm("M4", lambda: m4_verify(False), args.reps, verify=lambda: m4_verify(True))

    out.write("\n-- M4V: M4 on the VALUE artifact (the frozen pure runner's graph) --\n")
    value_args = (
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        int(forest.input_len),
        int(forest.output_len),
        int(x.shape[0]),
    )
    value_kwargs = {"per_row": False, "backend": None}
    torch.cuda.synchronize()
    t0 = pl.NOW()
    value_plan = dense_engine.plan_dense(*value_args, **value_kwargs)
    value_executable = dense_engine.build_dense(value_plan)
    value_call = make_call((value_args, value_kwargs), staged_value, with_labels=False)
    value_call()  # out-of-band warm-up: fills the value staged memo
    torch.cuda.synchronize()
    out.write(
        f"      value artifact: variant={value_plan.variant} trace_len={value_plan.trace_len} "
        f"plan+build+warm-up {((pl.NOW() - t0) * 1e3):.1f} ms "
        f"payloads staged={len(staged_value)}\n"
    )
    run_arm("M4V", lambda: value_call(False), args.reps)

    # ---- M5: the frozen runner ------------------------------------------
    out.write("\n-- M5: frozen build_dense_runner(forest, x).run() --\n")
    patches.restore()
    torch.cuda.synchronize()
    t0 = pl.NOW()
    runner = dense_engine.build_dense_runner(forest, x)
    torch.cuda.synchronize()
    runner_build_ms = (pl.NOW() - t0) * 1e3
    runner_identity = {
        "executable_is_m4v_artifact": bool(runner.executable is value_executable),
        "plan_variant": str(getattr(runner.plan, "variant", None)),
        "plan_trace_len": int(getattr(runner.plan, "trace_len", -1)),
        "staged_payloads": len(runner.staged),
        "n_graph_calls_per_run": int(getattr(runner, "invocations", -1)),
        "construct_total_ms": runner_build_ms,
        "backend": runner.backend_name,
        "device_resident": bool(runner.device_resident),
    }
    out.write(
        f"      runner built in {runner_build_ms:.1f} ms; artifact identity "
        f"(runner.executable is the M4V artifact): {runner_identity['executable_is_m4v_artifact']}\n"
    )
    run_arm("M5", lambda: runner.run(), args.reps)

    # ---- closing controls ------------------------------------------------
    out.write("\n-- M0R: M0 again, all patches restored (drift bracket) --\n")
    run_arm("M0R", lambda: problem.evaluate(forest), args.reps)

    out.write("\n-- M3R: M3 again (drift bracket) --\n")
    install_plan_memo(dense_engine, patches, captured)
    install_column_memo(dense_engine, patches)
    install_staged_seam(dense_engine, patches, staged)
    run_arm("M3R", lambda: problem.evaluate(forest), args.reps)
    patches.restore()

    out.write("\n-- M0P: M0 with the instrumentation inert (probe overhead) --\n")
    profiler.enabled = False
    run_arm("M0P", lambda: problem.evaluate(forest), args.control_reps, plain=True)
    profiler.enabled = True

    # ---- Q1 attribution + falsification ----------------------------------
    # Predicted per-step costs, in ms, read off results/q1_decomp/SUMMARY.md
    # (median exclusive self time over 24 warm calls; components, not totals):
    #   M0-M1 = prepare_dense_self + plan_dense_self + 3/5 of as_numpy_d2h
    #   M1-M2 = prep_column
    #   M2-M3 = etl_tensor_to_payload + etl_from_numpy + the semaphore waits inside
    #           (cProfile: `_wait_copy_semaphore` 1.84 ms/call on f64 lives there)
    #   M3-M4 = etl_tensor_to_to_host + run_loss_plan_self + sr_fitness_dense_self
    #   M4V-M5 = build_dense cache prologue + the x/labels host copies + _resolve_backend
    q1_f64 = {"M0-M1": 0.70 + 0.077 + 0.086, "M1-M2": 0.034, "M2-M3": 3.508 + 0.037,
              "M3-M4": 0.112 + 0.512 + 0.112, "M4V-M5": 0.079 + 0.06}
    q1_f200 = {"M0-M1": 1.383 + 0.078 + 0.101, "M1-M2": 0.035, "M2-M3": 1.530 + 0.036,
               "M3-M4": 0.117 + 0.513 + 0.115, "M4V-M5": 0.077 + 0.06}
    predicted = q1_f64 if int(forest.pop_size) <= 64 else q1_f200
    steps = (
        ("M0-M1", "plan_dense: decode + np.unique + plan build", "M0", "M1"),
        ("M1-M2", "_prep_column x7 (node-column host prep)", "M1", "M2"),
        ("M2-M3", "payload uploads: etl.from_numpy + .to(device) x10", "M2", "M3"),
        ("M3-M4", "readback D2H + run_loss_plan assembly + fitness H2D", "M3", "M4"),
        ("M4-M4V", "loss variant -> value variant (graph + labels/keep payloads)", "M4", "M4V"),
        ("M4V-M5", "engine per-call prologue (plan memo hit + build_dense + x host copy)", "M4V", "M5"),
    )
    deltas = []
    for step, removes, high, low in steps:
        measured = arms[high]["stats"]["median_warm_ms"] - arms[low]["stats"]["median_warm_ms"]
        entry = {
            "step": step,
            "removes": removes,
            "median_high_ms": arms[high]["stats"]["median_warm_ms"],
            "median_low_ms": arms[low]["stats"]["median_warm_ms"],
            "delta_ms": measured,
            "q1_predicted_ms": predicted.get(step),
        }
        if entry["q1_predicted_ms"] is not None:
            entry["ratio_measured_over_q1"] = measured / entry["q1_predicted_ms"]
            entry["agrees_within_25pct"] = abs(measured - entry["q1_predicted_ms"]) <= 0.25 * entry["q1_predicted_ms"]
        deltas.append(entry)
    total_m0_m5 = arms["M0"]["stats"]["median_warm_ms"] - arms["M5"]["stats"]["median_warm_ms"]
    drift = {
        "M0_repeat_delta_pct": 100.0
        * (arms["M0R"]["stats"]["median_warm_ms"] - arms["M0"]["stats"]["median_warm_ms"])
        / arms["M0"]["stats"]["median_warm_ms"],
        "M3_repeat_delta_pct": 100.0
        * (arms["M3R"]["stats"]["median_warm_ms"] - arms["M3"]["stats"]["median_warm_ms"])
        / arms["M3"]["stats"]["median_warm_ms"],
        "M0_plain_overhead_pct": 100.0
        * (arms["M0"]["stats"]["median_warm_ms"] - arms["M0P"]["stats"]["median_warm_ms"])
        / arms["M0P"]["stats"]["median_warm_ms"],
    }
    floors = {
        "M0_full_evaluate_ms": arms["M0"]["stats"]["median_warm_ms"],
        "M3_host_glue_removed_ms": arms["M3"]["stats"]["median_warm_ms"],
        "M4_host_glue_and_readback_removed_ms": arms["M4"]["stats"]["median_warm_ms"],
        "M4V_value_variant_no_readback_ms": arms["M4V"]["stats"]["median_warm_ms"],
        "M5_pure_only_frozen_runner_ms": arms["M5"]["stats"]["median_warm_ms"],
        "M0_minus_M5_ms": total_m0_m5,
        "removes": {
            "M3_host_glue_removed": "covers M1+M2+M3: plan_dense (decode+unique+plan), _prep_column x7, and the 10 per-call payload uploads (etl.from_numpy + Tensor.to, incl. the copy-semaphore waits)",
            "M4_host_glue_and_readback_removed": "M3 plus the (pop,) loss readback D2H, the run_loss_plan host assembly and the fitness H2D",
            "M5_pure_only": "M4V plus the engine's per-call prologue (plan_dense memo hit, build_dense cache-hit prologue, _resolve_backend, the x host copy); the frozen runner holds plan/executable/args directly",
        },
    }

    # ---- record ----------------------------------------------------------
    host_load_end = pl.host_load()
    payload = {
        "probe": PROBE,
        "tag": tag,
        "stamp": pl.identity_stamp(
            etl,
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
            "control_reps": args.control_reps,
            "pop_size": int(forest.pop_size),
            "max_tree_len": int(forest.max_tree_len),
            "input_len": int(forest.input_len),
            "output_len": int(forest.output_len),
            "data_rows": int(x.shape[0]),
            "eval_kind": "fixture",
            "harness_call": "problem.evaluate(forest), one torch.cuda.synchronize() pair (bench_gp.py:1377-1381)",
            "arm_order": list(ARM_ORDER),
        },
        "route": {
            "backend": backend_name,
            "eval_mode": dense_engine.resolve_eval_mode(),
            "should_use_dense": bool(use_dense),
            "plan": plan_sink,
            "value_artifact": {
                "variant": str(getattr(value_plan, "variant", None)),
                "trace_len": int(getattr(value_plan, "trace_len", -1)),
                "pop_size": int(getattr(value_plan, "pop_size", -1)),
                "n_rows": int(getattr(value_plan, "n_rows", -1)),
                "staged_payloads": len(staged_value),
            },
            "loss_artifact": {
                "variant": str(getattr(captured.get("plan_from_build"), "variant", None)),
                "trace_len": int(getattr(captured.get("plan_from_build"), "trace_len", -1)),
                "staged_payloads": len(staged),
            },
            "m5_identity": runner_identity,
        },
        "arms": arms,
        "deltas": deltas,
        "floors": floors,
        "drift": drift,
        "prelude": prelude,
        "cache_final": {"hits": int(cache.hits), "misses": int(cache.misses), "size": len(cache)},
        "probe_notes": profiler.notes,
        "install_status": {
            "wrapped": profiler.install_status["wrapped"],
            "missing": profiler.install_status["missing"],
            "depth_overflow": profiler.depth_overflow,
        },
    }

    # ---- log report ------------------------------------------------------
    out.write("\n=== arm table (harness-shaped warm median, call 0 excluded) ===\n")
    out.write(f"{'arm':5s} {'median_ms':>10s} {'min':>9s} {'max':>9s} {'first':>10s}  removes\n")
    for name in order:
        stats = arms[name]["stats"]
        out.write(
            f"{name:5s} {stats['median_warm_ms']:10.4f} {stats['min_warm_ms']:9.4f} "
            f"{stats['max_warm_ms']:9.4f} {stats['first_call_ms']:10.3f}  {ARM_DOCS[name][1]}\n"
        )
    out.write("\n=== per-step deltas vs the Q1 timer attribution ===\n")
    out.write(f"{'step':9s} {'delta_ms':>9s} {'q1_pred':>9s} {'ratio':>7s}  removes\n")
    for entry in deltas:
        ratio = (
            f"{entry['ratio_measured_over_q1']:.2f}"
            if entry.get("ratio_measured_over_q1") is not None
            else "n/a"
        )
        predicted_text = (
            f"{entry['q1_predicted_ms']:.3f}" if entry["q1_predicted_ms"] is not None else "n/a"
        )
        out.write(
            f"{entry['step']:9s} {entry['delta_ms']:9.4f} {predicted_text:>9s} {ratio:>7s}  "
            f"{entry['removes']}\n"
        )
    out.write(
        f"\nM0 - M5 = {total_m0_m5:.4f} ms over the sum of the steps "
        f"({sum(entry['delta_ms'] for entry in deltas):.4f} ms)\n"
    )
    out.write(
        "=== floors ===\n"
        f"  M0 full evaluate                          {floors['M0_full_evaluate_ms']:8.4f} ms\n"
        f"  M3 'host glue removed'                    {floors['M3_host_glue_removed_ms']:8.4f} ms\n"
        f"  M4 'host glue + readback removed'         {floors['M4_host_glue_and_readback_removed_ms']:8.4f} ms\n"
        f"  M4V value variant, no readback            {floors['M4V_value_variant_no_readback_ms']:8.4f} ms\n"
        f"  M5 'pure only' (frozen runner.run())      {floors['M5_pure_only_frozen_runner_ms']:8.4f} ms\n"
        "\n=== drift / overhead controls ===\n"
    )
    for key, value in drift.items():
        out.write(f"  {key:28s} {value:+.2f} %\n")
    out.write("\n=== fitness equivalence (M0..M4 must agree) ===\n")
    for name in order:
        arm = arms[name]
        if arm.get("fitness_sha256") or arm.get("verify_fitness_sha256"):
            sha = arm.get("fitness_sha256") or arm.get("verify_fitness_sha256")
            out.write(
                f"  {name:4s} sha256 {sha[:24]}...  matches M0: {arm['loss_reference']}\n"
            )
        else:
            out.write(f"  {name:4s} (no fitness: forward-only floor, result dropped)\n")

    upload_micro = None
    if not args.no_micro:
        xt_host = dense_engine._input_to_numpy(x, int(forest.input_len), "inputs")
        label_host = dense_engine._as_numpy(labels, np.float32, "labels")
        captured_plan = captured["plan_from_build"]
        hosts = build_payload_hosts(dense_engine, captured_plan, xt_host, label_host)
        upload_micro = upload_micro_probe(etl, device, hosts, args.micro_reps, out)
    payload["upload_micro"] = upload_micro

    if args.json_out:
        pl.write_json(args.json_out, payload)
        out.write(f"\nrecord: {args.json_out}\n")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
