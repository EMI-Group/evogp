#!/usr/bin/env python
"""``kineto_attempt.py`` -- R5 Q2(d): ONE kineto (torch.profiler) attempt at the
dense-route GPU timeline.

nsys/ncu are blocked on this box (QDSTRM import fails against driver 535.247.01;
ncu refuses with "API call is not supported in the installed CUDA driver" -- see
``../results/tool_recon/README.md``).  ``torch.profiler`` (kineto + CUPTI) does
capture torch's own kernels here, so this is the round's single attempt to see
whether the IREE-cuda kernels of the compiled dense graph show up too.  If the
kernel table shows only torch/CUDA-runtime activity (memcpy/memset) and no IREE
kernels, the Q2 GPU mechanism stays **profiler-blocked** and this record is the
evidence; there is deliberately no second attempt.

Method: build the frozen dense runner exactly as ``../pure_forward.py`` does
(``dense_engine.build_dense_runner(forest, x)``; the compile happens OUTSIDE the
profiled window), then profile ``--invocations`` ``runner.run()`` calls with
``torch.profiler`` (CUDA + CPU activities, no schedule).  The record keeps the
compact key-averages table (name, counts, cuda/cpu totals) and, only when small
(< 5 MB), the chrome trace.

Usage (repo root, project venv, GPU):

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \\
      EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=dense EVOGP_ETL_IREE_DEVICE=0 \\
      EVOGP_ETL_IREE_OPT_LEVEL=O1 \\
      <venv>/bin/python test/benchmark/r5/gpu_probe/kineto_attempt.py \\
        --fixture test/goldens/forest_random.npz --invocations 30 \\
        --out-dir test/benchmark/r5/results/q2_gpu/kineto_f64 \\
        --json-out test/benchmark/r5/results/q2_gpu/kineto_f64.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.dirname(HERE)
if R5 not in sys.path:
    sys.path.insert(0, R5)

import probe_lib as pl  # noqa: E402

PROBE = "r5/gpu_probe/kineto_attempt.py"
TRACE_BUDGET_BYTES = 5 * 1024 * 1024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="R5 Q2(d) single kineto attempt")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=("pure", "e2e"), default="pure")
    parser.add_argument("--invocations", type=int, default=30)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = args.log_out or os.path.join(args.out_dir, "kineto.log")
    with open(log_path, "w") as out:
        out.write(f"=== {PROBE} ===\n")
        import etl
        import torch

        from evogp.tree import dense_engine

        problem, forest, x, _labels = pl.build_eval_case(args.fixture)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.mode == "pure":
            runner = dense_engine.build_dense_runner(forest, x)
            call = runner.run
        else:
            call = lambda: problem.evaluate(forest)  # noqa: E731 -- the e2e route
        if args.mode == "pure":
            call()  # the build already ran once; keep one warm call before profiling
        torch.cuda.synchronize()
        build_ms = (time.perf_counter() - t0) * 1000.0
        out.write(f"construct/first-call (outside the profile): {build_ms:.3f} ms\n")

        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.invocations):
                call()
            torch.cuda.synchronize()

        table_path = os.path.join(args.out_dir, "kineto_key_averages.txt")
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=60)
        with open(table_path, "w") as handle:
            handle.write(table)

        trace_path = os.path.join(args.out_dir, "kineto_trace.json.gz")
        trace_written = False
        try:
            prof.export_chrome_trace(trace_path)
            if os.path.getsize(trace_path) > TRACE_BUDGET_BYTES:
                os.remove(trace_path)
            else:
                trace_written = True
        except Exception as exc:  # the trace is a bonus, never a failure
            out.write(f"chrome trace export failed: {type(exc).__name__}: {exc}\n")

        events = []
        for item in prof.key_averages():
            events.append(
                {
                    "key": item.key,
                    "count": int(item.count),
                    "cpu_time_total_us": float(item.cpu_time_total),
                    "cuda_time_total_us": float(getattr(item, "device_time_total", 0.0)),
                    "self_device_time_total_us": float(
                        getattr(item, "self_device_time_total", 0.0)
                    ),
                }
            )
        cuda_events = [e for e in events if e["cuda_time_total_us"] > 0]
        iree_events = [e for e in events if re.search(r"iree|hal|dispatch_", e["key"], re.I)]
        kernels = sorted(cuda_events, key=lambda e: -e["cuda_time_total_us"])[:25]

        out.write("\n--- kineto key averages (sorted by cuda time) ---\n")
        out.write(table)
        out.write(f"\nevents: {len(events)}   with cuda time > 0: {len(cuda_events)}\n")
        out.write(f"iree-looking event names: {len(iree_events)}\n")
        for event in iree_events[:20]:
            out.write(f"  {event}\n")
        out.write("\ntop device-time events:\n")
        for event in kernels:
            out.write(
                f"  {event['cuda_time_total_us']:12.3f} us  n={event['count']:5d}  {event['key'][:110]}\n"
            )

        payload = {
            "probe": PROBE,
            "tag": args.tag,
            "config": {
                "fixture": os.path.abspath(args.fixture),
                "mode": args.mode,
                "invocations": args.invocations,
                "pop_size": int(forest.pop_size),
                "rows": int(x.shape[0]),
            },
            "construct_ms": build_ms,
            "table_path": table_path,
            "trace_path": trace_path if trace_written else None,
            "n_events": len(events),
            "n_events_with_device_time": len(cuda_events),
            "iree_events": iree_events[:20],
            "top_device_events": kernels,
            "verdict": (
                "iree kernels visible"
                if iree_events
                else (
                    "no iree-named events; device activity limited to "
                    f"{len(cuda_events)} torch/runtime events"
                )
            ),
            "stamp": pl.identity_stamp(etl, extra={"probe": PROBE, "tag": args.tag}),
        }
        if args.json_out:
            pl.write_json(args.json_out, payload)
            out.write(f"\nrecord: {args.json_out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
