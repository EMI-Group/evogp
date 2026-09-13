#!/usr/bin/env python
"""``minimal_loop.py`` -- R5 Q8: the harness-artifact falsification control.

Twenty-odd lines of real work: load the fixture, build the same
``Forest`` + ``SymbolicRegression`` pair the harness builds
(``test/benchmark/bench_gp.py:768-806``), then call ``problem.evaluate(forest)``
in a plain loop, each call wrapped in ONE ``torch.cuda.synchronize()`` pair --
the same timed region the harness uses (``bench_gp.py:1377-1381``) -- with no
``bench_gp.py`` import, no JSON plumbing, no probe, no monkeypatch.

If this lands in the same band as the Q4 cells (~5.4-6.9 ms for the 64-tree
fixture), the "the harness inflates the call" hypothesis is falsified: the cost
is in ``problem.evaluate`` itself.  The only harness-side suspect a minimal loop
cannot rule out is the interpreter/venv (identical here by construction).

Usage (repo root, project venv, GPU):

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
      EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 \
      EVOGP_ETL_IREE_OPT_LEVEL=O1 \
      <venv>/bin/python test/benchmark/r5/minimal_loop.py \
        --fixture test/goldens/forest_random.npz --reps 25 \
        --json-out test/benchmark/r5/results/q8_minimal/minimal_f64.json \
        --log-out  test/benchmark/r5/results/q8_minimal/minimal_f64.log
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def stamp():
    import numpy as np
    import torch

    def run(cmd, cwd=None):
        try:
            out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
            return (out.stdout or out.stderr or "").strip() or "<empty>"
        except Exception as exc:  # pragma: no cover - stamping must never fail a run
            return f"<{type(exc).__name__}>"

    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": os.getcwd(),
        "evogp_commit": run(["git", "rev-parse", "HEAD"]),
        "etl_repo_head": run(["git", "rev-parse", "HEAD"], cwd=os.environ.get("R5_ETL_REPO", "/mnt/local-ssd/bchuang/etl")),
        "python": sys.executable,
        "torch": getattr(torch, "__version__", "<n/a>"),
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "env_knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("ETL_", "EVOGP_"))},
        "gpu": run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")]),
        "gpu_tenants": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0")]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="R5 Q8 minimal-loop control (no harness)")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    import etl
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest

    data = np.load(args.fixture)                                   # fixture -> problem, twice
    x = torch.from_numpy(np.array(data["x"], dtype=np.float32, copy=True)).to("cuda")
    labels = torch.from_numpy(np.array(data["labels"], dtype=np.float32, copy=True)).to("cuda")
    forest = Forest(
        input_len=int(x.shape[1]), output_len=int(labels.shape[1]),
        batch_node_value=torch.from_numpy(np.array(data["forest__node_value"], dtype=np.float32, copy=True)).to("cuda"),
        batch_node_type=torch.from_numpy(np.array(data["forest__node_type"], dtype=np.int16, copy=True)).to("cuda"),
        batch_subtree_size=torch.from_numpy(np.array(data["forest__subtree_size"], dtype=np.int16, copy=True)).to("cuda"),
    )
    problem = SymbolicRegression(datapoints=x, labels=labels)
    calls = []                                                     # 25 calls, call 0 cold
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        problem.evaluate(forest)
        torch.cuda.synchronize()
        calls.append((time.perf_counter() - t0) * 1000.0)
    warm = calls[1:]
    result = {
        "probe": "r5/minimal_loop.py",
        "config": {"fixture": os.path.abspath(args.fixture), "reps": args.reps,
                   "pop_size": int(forest.pop_size), "data_rows": int(x.shape[0]),
                   "call": "problem.evaluate(forest), one torch.cuda.synchronize() pair per call"},
        "per_call_ms": calls,
        "first_call_ms": calls[0],
        "median_warm_ms": statistics.median(warm),
        "min_warm_ms": min(warm),
        "max_warm_ms": max(warm),
        "etl_file": getattr(etl, "__file__", "<n/a>"),
        "stamp": stamp(),
    }
    lines = [
        "=== r5/minimal_loop.py (Q8: no harness, no probe) ===",
        f"fixture: {result['config']['fixture']}  pop={forest.pop_size} rows={x.shape[0]}  reps={args.reps}",
        f"etl    : {result['etl_file']}",
        "per-call ms: " + " ".join(f"{v:.3f}" for v in calls),
        f"first (cold): {calls[0]:.3f} ms   warm median: {result['median_warm_ms']:.3f} ms "
        f"(min {result['min_warm_ms']:.3f} / max {result['max_warm_ms']:.3f})",
    ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log_out:
        with open(args.log_out, "w") as handle:
            handle.write(text)
    if args.json_out:
        with open(args.json_out + ".tmp", "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        os.replace(args.json_out + ".tmp", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
