#!/usr/bin/env python
"""Backend numeric-agreement pre-flight for the R2 GPU round (numpy vs IREE-cuda O1).

Why this exists
---------------
The R2 round runs the same fixed fixtures on the default ``numpy`` backend and on
the opt-in IREE-cuda backend (``EVOGP_ETL_BACKEND=iree``, ``EVOGP_ETL_IREE_*
LEVEL=O1``).  ``O2``/``O3`` are known to produce *silent* CUDA miscompiles, so
before any timing number from the IREE side is quoted, both backends must be shown
to compute the same fitness on the same data.  This is a gate, not a benchmark:
every value either lands on the recorded anchors (see ``backend_agreement.md``)
or it does not.

What it does (public evogp API only -- ``bench_gp.py`` is deliberately NOT
imported: it is a run-at-import harness, not a library)
------------------------------------------------------------------------
1. load ``--fixture`` (the npz schema of ``test/benchmark/fixtures/README.md``,
   also used by ``test/goldens/forest_random.npz``);
2. rebuild the ``eval64`` case exactly like ``bench_gp.py::build_eval_case``:
   ``Forest(input_len, output_len, batch_node_value=<forest__node_value f32>,
   batch_node_type=<forest__node_type i16>,
   batch_subtree_size=<forest__subtree_size i16>)`` plus
   ``SymbolicRegression(datapoints=x, labels=labels)`` (both float32 on cuda);
3. call ``problem.evaluate(forest)`` twice, each bracketed by
   ``torch.cuda.synchronize()``, and read the ``(pop,)`` fitness tensor back as
   float64;
4. summarize (nan/inf counts, finite mean/min/max, sha256 over the float64
   little-endian bytes) and write ``--json-out`` plus ``<json-stem>_values.npy``
   beside it.

The backend is *not* a CLI flag -- it is already fixed in the environment
(``EVOGP_ETL_BACKEND``, ``EVOGP_ETL_IREE_DEVICE``, ``EVOGP_ETL_IREE_OPT_LEVEL``,
plus ``ETL_DISABLE_LOCATIONS=1``) exactly as the rest of the round sets it before
starting the interpreter; the raw env values are copied into the JSON for the
record.  Run it from the repository root with the project venv::

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
        EVOGP_ETL_BACKEND=numpy \
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \
        test/benchmark/r2/check_backend_agreement.py \
        --fixture test/goldens/forest_random.npz \
        --json-out test/benchmark/new/results/r2/backend_agreement/fixture64_numpy.json

``problem.evaluate`` returns the MAXIMIZED fitness (the negated SR loss), so the
reported mean is negative and NaN trees are counted, not hidden.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

from evogp.problem import SymbolicRegression
from evogp.tree import Forest

CALLS = 2  # cold call + warm call, both sync-bracketed


# ---------------------------------------------------------------------------
# helpers (kept byte-compatible with bench_gp.py's eval64 wiring)
# ---------------------------------------------------------------------------
def now_iso():
    """Local timestamp with UTC offset (same clock as the run logs)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_cuda(array, dtype):
    """Copy a fixture array into a writable contiguous torch tensor on cuda."""
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).to("cuda")


def build_eval_case(fixture):
    """(problem, forest, description) for the fixed-population eval task.

    Mirrors ``bench_gp.py::build_eval_case`` (the ``eval64`` branch): the forest is
    rebuilt from the fixture's ``forest__*`` arrays through the class constructor
    and the problem takes the fixture's own ``x``/``labels`` -- no RNG anywhere.
    """
    data = np.load(fixture)
    x = to_cuda(data["x"], np.float32)
    labels = to_cuda(data["labels"], np.float32)
    forest = Forest(
        input_len=int(x.shape[1]),
        output_len=int(labels.shape[1]),
        batch_node_value=to_cuda(data["forest__node_value"], np.float32),
        batch_node_type=to_cuda(data["forest__node_type"], np.int16),
        batch_subtree_size=to_cuda(data["forest__subtree_size"], np.int16),
    )
    problem = SymbolicRegression(datapoints=x, labels=labels)
    description = (
        f"Forest(input_len={forest.input_len}, output_len={forest.output_len}, "
        f"pop_size={forest.pop_size}, max_tree_len={forest.max_tree_len}) rebuilt from "
        f"{fixture}[forest__node_value/_node_type/_subtree_size] + "
        f"SymbolicRegression(datapoints=x{tuple(x.shape)}, labels=labels{tuple(labels.shape)})"
    )
    return problem, forest, description


def summarize(values):
    """NaN-aware statistics over the float64 fitness array."""
    finite = values[np.isfinite(values)]
    return {
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "inf_count": int(np.count_nonzero(np.isinf(values))),
        "count": int(values.size),
        "finite_mean": float(finite.mean()) if finite.size else None,
        "finite_min": float(finite.min()) if finite.size else None,
        "finite_max": float(finite.max()) if finite.size else None,
        "values_sha256": hashlib.sha256(
            np.ascontiguousarray(values, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def write_json(path, payload):
    """Atomic write: temp file in the target directory, then rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Numeric-agreement pre-flight: evaluate one fixed fixture twice "
                    "through the public evogp API and summarize the fitness vector "
                    "(backend chosen by EVOGP_ETL_BACKEND, not by a flag).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fixture", required=True,
                        help="eval fixture npz (forest__node_value/_node_type/_subtree_size "
                             "+ x + labels), e.g. test/goldens/forest_random.npz")
    parser.add_argument("--json-out", required=True,
                        help="JSON report path; the raw float64 values are saved beside it "
                             "as <stem>_values.npy")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started_at = now_iso()
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible (this round is GPU-only)")

    fixture_sha256 = sha256_file(args.fixture)
    device_name = torch.cuda.get_device_name()
    print(f"[agreement] fixture={args.fixture} sha256={fixture_sha256}", flush=True)
    print(f"[agreement] backend={os.environ.get('EVOGP_ETL_BACKEND') or 'unset'} "
          f"iree_opt_level={os.environ.get('EVOGP_ETL_IREE_OPT_LEVEL') or 'n/a'} "
          f"iree_device={os.environ.get('EVOGP_ETL_IREE_DEVICE') or 'n/a'} "
          f"device={device_name}", flush=True)

    problem, forest, description = build_eval_case(args.fixture)
    print(f"[agreement] {description}", flush=True)

    per_call_ms = []
    fitness = None
    for call in range(CALLS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fitness = problem.evaluate(forest)
        torch.cuda.synchronize()
        per_call_ms.append((time.perf_counter() - t0) * 1000.0)
        print(f"[agreement] call {call + 1}/{CALLS}: {per_call_ms[-1]:.1f} ms (cold if 1)",
              flush=True)

    if not isinstance(fitness, torch.Tensor):
        raise TypeError(f"problem.evaluate returned {type(fitness).__name__}, not a tensor")
    fitness = fitness.detach().cpu().to(torch.float64)
    if fitness.ndim != 1:
        raise ValueError(f"fitness tensor must be (pop,), got {tuple(fitness.shape)}")
    values = fitness.numpy()

    summary = summarize(values)
    values_path = os.path.splitext(args.json_out)[0] + "_values.npy"
    os.makedirs(os.path.dirname(os.path.abspath(values_path)), exist_ok=True)
    np.save(values_path, values)

    payload = {
        "fixture": args.fixture,
        "fixture_sha256": fixture_sha256,
        "backend": os.environ.get("EVOGP_ETL_BACKEND") or "unset",
        "iree_opt_level": os.environ.get("EVOGP_ETL_IREE_OPT_LEVEL") or None,
        "iree_device": os.environ.get("EVOGP_ETL_IREE_DEVICE") or None,
        "nan_count": summary["nan_count"],
        "inf_count": summary["inf_count"],
        "count": summary["count"],
        "finite_mean": summary["finite_mean"],
        "finite_min": summary["finite_min"],
        "finite_max": summary["finite_max"],
        "per_call_ms": per_call_ms,
        "values_sha256": summary["values_sha256"],
        "torch_version": torch.__version__,
        "cuda_device_name": device_name,
        "started_at": started_at,
        "finished_at": now_iso(),
    }
    write_json(args.json_out, payload)

    print(f"[agreement] nan_count={summary['nan_count']}/{summary['count']} "
          f"inf_count={summary['inf_count']} "
          f"finite_mean={summary['finite_mean']!r} "
          f"finite_min={summary['finite_min']!r} finite_max={summary['finite_max']!r}", flush=True)
    print(f"[agreement] values_sha256={summary['values_sha256']}", flush=True)
    print(f"[agreement] wrote {args.json_out} + {values_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
