#!/usr/bin/env python
"""``extra_trace.py`` -- r6 P1 extra (a)/(c): kineto (``torch.profiler``)
decomposition of one warm per-call unit, for the three sides of the round.

Why kineto: nsys/ncu are blocked on this box (r5 ``results/tool_recon/README.md``:
nsys QDSTRM import fails against driver 535.247.01, ncu refuses with "API call is
not supported in the installed CUDA driver"), while ``torch.profiler`` (kineto +
CUPTI) DOES capture the IREE-cuda kernels, the Triton kernel and the legacy
CUDA kernels (r5 ``gpu_probe/kineto_attempt.py``, ``results/q2_gpu/``).

What it produces, per target, in ONE fresh process:
  * a **clean** (unprofiled, sync-bracketed) block of ``--invocations`` calls ->
    ``clean_per_invocation_ms`` (the number to trust; the profiler inflates the
    wall slightly);
  * a **profiled** block of the same length -> the kineto key-averages table
    (top rows by device time) and the aggregate facts:
    total device time, kernel-only time, runtime-API time, kernel/launch counts,
    average kernel duration and the residual host/gap per invocation
    (``wall - device``, computed against both walls).

Targets (``--target``):
  ``triton-pure``  resident ``TritonEvaluator.run()`` (type-2 primitive, exactly
                   what ``run_triton_eval64.py --mode pure`` loops over);
  ``triton-e2e``   the unmodified public API with the dense seam rebound to the
                   Triton evaluator (type-1 envelope; ``--mode e2e`` of the driver);
  ``legacy-e2e``   ``problem.evaluate(forest)`` on the legacy CUDA engine -- run
                   this one with the legacy checkout on ``PYTHONPATH`` (the script
                   never inserts ``src/``, so the legacy ``evogp`` wins).

The triton targets import the round's own driver only for its
``build_eval_case`` (the byte-identical harness wiring); the legacy target builds
the same fixture case locally so no ETL-side module is imported at all.

Usage (repo root, project venv, CUDA-visible GPU, serialized on the benchmark
lock -- see ``CONTEXT.md``)::

    python test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
      env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
          TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache \
          CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
          <venv>/bin/python test/benchmark/r6/extra_trace.py --target triton-pure \
          --fixture test/goldens/forest_random.npz --invocations 100 \
          --out-dir test/benchmark/r6/results/<win>/extra/kernel_host/triton_pure_f64 \
          --json-out test/benchmark/r6/results/<win>/extra/kernel_host/triton_pure_f64/facts.json

Exit codes: 0 complete, 1 failure.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.getcwd()
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "test", "benchmark", "r6")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: CUDA-runtime / driver rows of a kineto table (everything else is a kernel).
RUNTIME_RE = re.compile(
    r"^(cuda[A-Z]|cu[A-Z]|memset|memcpy|malloc|free|Memset|Memcpy|Malloc|Free|"
    r"cudaLaunch|Sync)", re.I
)
TRACE_BUDGET_BYTES = 5 * 1024 * 1024


def _to_cuda(array: Any, dtype: Any) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).to("cuda")


def build_case_legacy(fixture: str):
    """(problem, forest, x, labels) -- bench_gp.build_eval_case's eval64 wiring.

    Imported lazily and WITHOUT touching ``src/`` so that, when this script runs
    with the legacy checkout on ``PYTHONPATH``, ``evogp`` is the legacy engine.
    """
    import numpy as np

    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest

    data = np.load(fixture)
    x = _to_cuda(data["x"], np.float32)
    labels = _to_cuda(data["labels"], np.float32)
    forest = Forest(
        input_len=int(x.shape[1]),
        output_len=int(labels.shape[1]),
        batch_node_value=_to_cuda(data["forest__node_value"], np.float32),
        batch_node_type=_to_cuda(data["forest__node_type"], np.int16),
        batch_subtree_size=_to_cuda(data["forest__subtree_size"], np.int16),
    )
    return SymbolicRegression(datapoints=x, labels=labels), forest, x, labels


def build_case_triton(fixture: str):
    """The driver's own ``build_eval_case`` (byte-identical harness wiring)."""
    import run_triton_eval64 as driver  # noqa: E402 -- sets EVOGP_EVAL_MODE=dense

    return driver.build_eval_case(fixture)


def clean_block(call, invocations: int) -> float:
    """One sync pair around ``invocations`` calls -> ms per invocation."""
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(invocations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / invocations


def profiled_block(call, invocations: int):
    """Same block inside a kineto profile -> the profiler object (timing comes
    from the surrounding clean blocks; the profiler's own wall is a by-product)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(invocations):
            call()
        torch.cuda.synchronize()
    return prof


def profile_facts(prof, invocations: int, clean_ms: float, tagged_ms: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for item in prof.key_averages():
        device_us = float(getattr(item, "device_time_total", 0.0) or 0.0)
        self_us = float(getattr(item, "self_device_time_total", 0.0) or 0.0)
        rows.append(
            {
                "key": item.key,
                "count": int(item.count),
                "cpu_us": float(item.cpu_time_total),
                "device_us": device_us,
                "self_device_us": self_us,
            }
        )
    device_rows = [r for r in rows if r["device_us"] > 0]
    runtime_rows = [r for r in device_rows if RUNTIME_RE.match(r["key"])]
    kernel_rows = [r for r in device_rows if not RUNTIME_RE.match(r["key"])]
    # kineto's key averages keep BOTH a C++ op row (e.g. ``aten::neg``,
    # ``evogp_cuda::tree_SR_fitness``) and the CUDA kernel rows it launched: the
    # op row's device time is its children's.  A *leaf* row is one whose device
    # time is (almost) all self time -- that is the raw-kernel view.
    leaf_rows = [r for r in kernel_rows if r["self_device_us"] >= 0.9 * r["device_us"]]
    kernel_us = sum(r["device_us"] for r in kernel_rows)
    leaf_us = sum(r["device_us"] for r in leaf_rows)
    kernel_launches = sum(r["count"] for r in kernel_rows)
    leaf_launches = sum(r["count"] for r in leaf_rows)
    runtime_us = sum(r["device_us"] for r in runtime_rows)
    device_us = kernel_us + runtime_us
    n = max(1, invocations)
    facts = {
        "invocations": invocations,
        "clean_per_invocation_ms": clean_ms,
        "profiled_per_invocation_ms": tagged_ms,
        "profiler_overhead_ms": tagged_ms - clean_ms,
        "device_time_total_us": device_us,
        "kernel_time_total_us": kernel_us,
        "runtime_api_time_total_us": runtime_us,
        "device_time_per_invocation_us": device_us / n,
        "kernel_time_per_invocation_us": kernel_us / n,
        "runtime_api_time_per_invocation_us": runtime_us / n,
        "kernel_count": kernel_launches,
        "kernels_per_invocation": kernel_launches / n,
        "avg_kernel_us": (kernel_us / kernel_launches) if kernel_launches else None,
        "kernel_time_leaf_total_us": leaf_us,
        "kernel_time_leaf_per_invocation_us": leaf_us / n,
        "leaf_kernel_count": leaf_launches,
        "leaf_kernels_per_invocation": leaf_launches / n,
        "avg_leaf_kernel_us": (leaf_us / leaf_launches) if leaf_launches else None,
        "host_residual_per_invocation_us": clean_ms * 1000.0 - device_us / n,
        "host_residual_profiled_per_invocation_us": tagged_ms * 1000.0 - device_us / n,
        "leaf_residual_per_invocation_us": clean_ms * 1000.0 - leaf_us / n,
        "double_counting_note": (
            "kineto key averages hold both a C++ op row (e.g. aten::neg, "
            "evogp_cuda::tree_SR_fitness) and the raw CUDA kernel rows it launched; "
            "the op row's device time is its children's, so kernel_time_total_us "
            "counts those children twice.  *_leaf_* fields keep only self-time-"
            "dominated rows (the raw kernels); read those as the device cost."
        ),
        "n_events": len(rows),
        "n_events_with_device_time": len(device_rows),
        "kernel_rows": sorted(kernel_rows, key=lambda r: -r["device_us"])[:15],
        "leaf_rows": sorted(leaf_rows, key=lambda r: -r["device_us"])[:15],
        "runtime_rows": sorted(runtime_rows, key=lambda r: -r["device_us"])[:15],
        "top_device_rows": sorted(device_rows, key=lambda r: -r["device_us"])[:30],
    }
    return facts


def identity_stamp(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import torch

    stamp: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        stamp["evogp_file"] = __import__("evogp").__file__
    except Exception as exc:  # pragma: no cover
        stamp["evogp_file"] = f"<import failed: {type(exc).__name__}>"
    try:
        stamp["etl_version"] = getattr(__import__("etl"), "__version__", None)
    except Exception as exc:  # pragma: no cover
        stamp["etl_version"] = f"<import failed: {type(exc).__name__}>"
    try:
        stamp["evogp_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
        ).stdout.strip() or None
    except Exception:  # pragma: no cover
        stamp["evogp_commit"] = None
    try:
        stamp["triton"] = __import__("triton").__version__
    except Exception:  # pragma: no cover
        stamp["triton"] = None
    if torch.cuda.is_available():
        stamp["gpu"] = torch.cuda.get_device_name(0)
    if extra:
        stamp.update(extra)
    return stamp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="r6 P1 kineto decomposition")
    parser.add_argument(
        "--target", choices=("triton-pure", "triton-e2e", "legacy-e2e"), required=True
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--invocations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-n", type=int, default=None, help="triton only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = args.log_out or os.path.join(args.out_dir, "extra_trace.log")
    tag = args.tag or args.target
    payload: Dict[str, Any] = {"tag": tag, "target": args.target}
    with open(log_path, "w") as out:
        out.write(f"=== extra_trace.py target={args.target} tag={tag} ===\n")
        import torch

        if args.target == "legacy-e2e":
            problem, forest, x, labels = build_case_legacy(args.fixture)
            call = lambda: problem.evaluate(forest)  # noqa: E731
            route = "legacy evogp_cuda: problem.evaluate(forest)"
            describe = {}
        else:
            import triton_eval as te

            problem, forest, x, labels = build_case_triton(args.fixture)
            if args.target == "triton-pure":
                evaluator = te.build_evaluator(
                    forest,
                    x,
                    labels,
                    use_MSE=True,
                    output="loss",
                    block_n=args.block_n,
                )
                torch.cuda.synchronize()
                call = evaluator.run
                route = "triton_eval.TritonEvaluator.run() (resident, build once)"
                describe = evaluator.describe()
            else:
                te.install_dense_seam(reset=True)
                call = lambda: problem.evaluate(forest)  # noqa: E731
                route = (
                    "public API with evogp.tree.dense_engine.sr_fitness_dense rebound to "
                    "triton_eval.sr_fitness_triton (type-1 envelope)"
                )
                describe = {}
            key_fields = (
                "staging",
                "kernel",
                "block_n",
                "num_warps",
                "launches_per_call",
                "compile_ms",
                "stage_upload_ms",
                "used_ops",
                "fallthrough_nodes",
            )
            summary = {k: describe.get(k) for k in key_fields}
            out.write(f"[{tag}] triton: {json.dumps(summary, default=str)}\n")
        payload.update(
            {
                "route": route,
                "fixture": os.path.abspath(args.fixture),
                "pop_size": int(forest.pop_size),
                "rows": int(x.shape[0]),
                "describe": describe,
            }
        )
        # --- warmup (outside every timed region) ------------------------------
        for _ in range(max(0, int(args.warmup))):
            call()
        torch.cuda.synchronize()
        # --- clean block ------------------------------------------------------
        clean_ms = clean_block(call, int(args.invocations))
        out.write(f"[{tag}] clean: {clean_ms:.4f} ms/invocation ({args.invocations} invocations)\n")
        # --- profiled block ---------------------------------------------------
        torch.cuda.synchronize()
        prof = profiled_block(call, int(args.invocations))
        tagged_ms = clean_block(call, int(args.invocations))  # a second clean block, post-trace
        table_path = os.path.join(args.out_dir, "kineto_key_averages.txt")
        table = prof.key_averages().table(sort_by="device_time_total", row_limit=40)
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
        facts = profile_facts(prof, int(args.invocations), clean_ms, tagged_ms)
        facts["median_per_invocation_ms"] = statistics.median([clean_ms, tagged_ms])
        out.write(f"[{tag}] post-trace clean: {tagged_ms:.4f} ms/invocation\n")
        out.write(f"[{tag}] kernels: {facts['kernel_count']} "
                  f"({facts['kernels_per_invocation']:.3f}/invocation), "
                  f"kernel time {facts['kernel_time_per_invocation_us']:.2f} us/invocation, "
                  f"avg kernel {facts['avg_kernel_us']} us\n")
        out.write(f"[{tag}] device total {facts['device_time_per_invocation_us']:.2f} us/invocation "
                  f"(runtime API {facts['runtime_api_time_per_invocation_us']:.2f} us), "
                  f"host residual {facts['host_residual_per_invocation_us']:.2f} us/invocation\n")
        out.write(f"[{tag}] leaf kernels: {facts['leaf_kernel_count']} "
                  f"({facts['leaf_kernels_per_invocation']:.2f}/invocation), "
                  f"{facts['kernel_time_leaf_per_invocation_us']:.2f} us/invocation, "
                  f"avg {facts['avg_leaf_kernel_us']} us, "
                  f"leaf residual {facts['leaf_residual_per_invocation_us']:.2f} us/invocation\n")
        out.write("\n--- kineto key averages (sorted by device time) ---\n")
        out.write(table)
        out.write("\ntop device rows:\n")
        for row in facts["top_device_rows"][:15]:
            out.write(f"  {row['device_us']:12.3f} us  n={row['count']:5d}  {row['key'][:110]}\n")
        out.write(f"\ntable: {table_path}\ntrace: {trace_path if trace_written else '<not kept>'}\n")
        payload.update(
            {
                "table_path": table_path,
                "trace_path": trace_path if trace_written else None,
                "wall_stamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **facts,
                "stamp": identity_stamp({"probe": "r6/extra_trace.py", "tag": tag}),
            }
        )
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            tmp = f"{args.json_out}.tmp.{os.getpid()}"
            with open(tmp, "w") as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.write("\n")
            os.replace(tmp, args.json_out)
            out.write(f"record: {args.json_out}\n")
    print(f"[{tag}] clean={payload.get('clean_per_invocation_ms'):.4f} ms  "
          f"kernel={payload.get('kernel_time_per_invocation_us'):.2f} us  "
          f"host_residual={payload.get('host_residual_per_invocation_us'):.2f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
