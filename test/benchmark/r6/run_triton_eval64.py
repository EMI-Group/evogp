"""r6 cell driver — one Triton fused-evaluator cell (type-1 envelope and type-2 pure).

Round r6 (``test/benchmark/r6/``): the Triton prototype's own measurement driver,
mirroring the r4/r5 protocol (one fresh process per cell, one JSON + one log per
cell, ``torch.cuda.synchronize()``-bracketed timed regions) without touching the
shared harnesses (``bench_gp.py`` / ``pure_forward.py`` are never edited).

Two modes
---------
``--mode e2e`` (default, measurement type 1) — the **unmodified public API** is
driven, with the dense engine seam replaced by the Triton evaluator:

    ``EVOGP_EVAL_MODE=dense`` is set BEFORE ``evogp`` is imported (under the default
    ``auto`` policy on the numpy backend ``should_use_dense`` returns False and the
    call would silently use the specialized engine instead of reaching the seam);
    then ``evogp.tree.dense_engine.sr_fitness_dense`` is rebound to
    :func:`triton_eval.sr_fitness_triton` (a module-attribute rebind; ``src/`` is
    untouched, and ``Forest.SR_fitness`` resolves the seam through the module
    attribute at call time).  ``--reps`` sync-bracketed ``problem.evaluate(forest)``
    calls are then timed: call 0 is the cold call (it pays the staging upload plus
    the Triton JIT compile) and ``median_warm_ms = median(calls 1..reps-1)``.

    ``Forest.SR_fitness`` catches ``DenseUnsupported`` and silently re-runs on the
    specialized engine — the round's known fallback hazard — so the driver counts
    every seam call and every kernel launch and **exits nonzero** unless the
    counters match the protocol's expectation.

``--mode pure`` (measurement type 2) — build once, then ``--reps`` timed blocks of
``--invocations`` warm invocations of the resident evaluator (one block = one sync
pair, exactly ``pure_forward.time_block``'s semantics), followed by an optional
``--e2e-companion K``: K real ``problem.evaluate(forest)`` calls in the SAME
process with the compile already warm, so ``e2e - pure`` prices the per-call host
glue around the kernel.

JSON contract (field names are pinned; downstream gate ``check_r6.py`` reads them)
---------------------------------------------------------------------------------
``tag, side, task, fixture, mode, route, seam_patched, patch_target, binding_style,
triton_call_count, triton_version, staging, op_coverage, launches_per_call, calls_ms,
cold_call_ms, median_warm_ms, mean_warm_ms, min_warm_ms, stdev_warm_ms, wall_s,
compile, env, gpu, evogp_commit, evogp_file`` plus, for ``pure``:
``warmup, blocks_ms, median_block_ms, per_invocation_ms (*_median/*_min/*_mean)``
and ``e2e_companion``.  ``triton_call_count`` is the number of
``sr_fitness_dense`` seam invocations served by Triton (``expected_triton_call_count``
= ``--reps`` in e2e mode, ``--e2e-companion`` in pure mode) and ``triton_runs`` is the
number of kernel launches (``expected_triton_runs`` = ``--reps`` in e2e mode,
``warmup + reps*invocations + e2e-companion`` in pure mode).

Exit codes: ``0`` complete, ``1`` failure (counter mismatch, unsupported population,
exception), ``130`` interrupted.

Exact commands (repository root, project venv, CUDA-visible GPU, inside the fcntl
lock, with the Triton header fix; see ``CONTEXT.md``)::

    python test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
      env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
          TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache \
          CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
          /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r6/run_triton_eval64.py \
          --fixture test/goldens/forest_random.npz --reps 25 --tag triton_e2e_f64 \
          --json-out /tmp/r6/triton_e2e_f64.json

    python test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
      env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
          TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache \
          CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
          /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r6/run_triton_eval64.py \
          --mode pure --fixture test/goldens/forest_random.npz --warmup 10 --invocations 100 \
          --reps 5 --e2e-companion 11 --tag triton_pure_f64 --json-out /tmp/r6/triton_pure_f64.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# environment first: EVOGP_EVAL_MODE must be visible before evogp is imported
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.getcwd()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_R6_DIR = os.path.join(_REPO_ROOT, "test", "benchmark", "r6")
if _R6_DIR not in sys.path:
    sys.path.insert(0, _R6_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ["EVOGP_EVAL_MODE"] = "dense"  # the seam is only reached under `dense`
os.environ.setdefault("ETL_DISABLE_LOCATIONS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import triton  # noqa: E402

import triton_eval as te  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git is always available on the box
        return None


def _to_cuda(array: Any, dtype) -> torch.Tensor:
    """Fixture array -> writable contiguous CUDA tensor (mirrors bench_gp.to_cuda)."""
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).cuda()


def install_seam(log: _Tee, tag: str) -> None:
    """Rebind the dense seam to triton and verify the patch took effect."""
    te.install_dense_seam()  # also resets the process counters (call it ONCE, first)
    if not te.seam_is_patched():
        raise RuntimeError("seam patch did not take effect (dense_engine.sr_fitness_dense)")
    log.write(
        f"[{tag}] seam: {te.SEAM_TARGET} -> triton_eval.sr_fitness_triton "
        f"(EVOGP_EVAL_MODE={os.environ['EVOGP_EVAL_MODE']})\n"
    )


def build_eval_case(fixture: str):
    """(problem, forest, x, labels) — byte-identical to ``bench_gp.build_eval_case``.

    ``bench_gp.py::build_eval_case`` (eval64 branch) is re-implemented here rather
    than imported so the driver has no dependency on the shared harness's argparse
    namespace; the construction is the same class constructors on the same arrays.
    """
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
    problem = SymbolicRegression(datapoints=x, labels=labels)
    return problem, forest, x, labels


def _env_snapshot() -> Dict[str, Any]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "EVOGP_EVAL_MODE",
        "ETL_DISABLE_LOCATIONS",
        "TRITON_CACHE_DIR",
        "CPATH",
        "EVOGP_ETL_BACKEND",
        "EVOGP_ETL_IREE_DEVICE",
        "EVOGP_ETL_IREE_OPT_LEVEL",
        "EVOGP_DENSE_MAX_TREE_LEN",
        "TRITON_F32_DEFAULT",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
    )
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def _gpu_snapshot() -> Dict[str, Any]:
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free_mb = used_mb = None
    try:
        free, total = torch.cuda.mem_get_info(index)
        free_mb = round(free / 1048576.0, 1)
        used_mb = round((total - free) / 1048576.0, 1)
    except Exception:  # pragma: no cover - mem_get_info is available on this box
        pass
    return {
        "device_index": int(index),
        "name": torch.cuda.get_device_name(index),
        "cc": f"{props.major}.{props.minor}",
        "total_mem_mb": round(props.total_memory / 1048576.0, 1),
        "free_mem_mb_at_start": free_mb,
        "used_mem_mb_at_start": used_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _stdev(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """Atomic JSON write (tmp file + os.replace), like the shared harness."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    os.replace(tmp, path)


class _Tee:
    """stdout + a per-cell log file (the r4/r5 'one JSON + one log per cell' rule)."""

    def __init__(self, path: Optional[str]):
        self.handle = open(path, "w") if path else None

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        if self.handle:
            self.handle.write(text)
            self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="r6 Triton fused-evaluator cell (type-1 envelope / type-2 pure)."
    )
    parser.add_argument("--fixture", required=True, help="fixture npz (forest__node_* + x/labels)")
    parser.add_argument("--reps", type=int, default=25, help="e2e: timed calls; pure: timed blocks")
    parser.add_argument("--tag", required=True, help="cell tag (recorded in the JSON and the log)")
    parser.add_argument("--json-out", required=True, help="record path (written atomically)")
    parser.add_argument("--mode", choices=("e2e", "pure"), default="e2e", help="measurement mode")
    parser.add_argument("--warmup", type=int, default=10, help="pure: warmup invocations (untimed)")
    parser.add_argument("--invocations", type=int, default=100, help="pure: invocations per block")
    parser.add_argument("--e2e-companion", type=int, default=0, help="pure: same-process e2e calls")
    parser.add_argument("--block-n", type=int, default=None, help="force the kernel row block size")
    parser.add_argument("--use-mae", action="store_true", help="MAE loss instead of MSE")
    parser.add_argument("--max-tree-len", type=int, default=None, help="dense staging trace cap")
    parser.add_argument("--log-out", default=None, help="log path (default: <json-out>.log)")
    parser.add_argument("--seed", type=int, default=0, help="torch manual seed (no RNG is used)")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# e2e (type 1): the unmodified public API through the patched dense seam
# --------------------------------------------------------------------------- #
def run_e2e(args: argparse.Namespace, log: _Tee) -> Dict[str, Any]:
    problem, forest, x, labels = build_eval_case(args.fixture)
    use_MSE = not bool(args.use_mae)
    log.write(
        f"[{args.tag}] e2e: pop={forest.pop_size} max_tree_len={forest.max_tree_len} "
        f"n_rows={int(x.shape[0])} input_len={forest.input_len} "
        f"output_len={forest.output_len} fixture={args.fixture}\n"
    )
    install_seam(log, args.tag)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    calls_ms: List[float] = []
    for call in range(int(args.reps)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        problem.evaluate(forest)
        torch.cuda.synchronize()
        calls_ms.append((time.perf_counter() - t0) * 1000.0)
        if call == 0:
            log.write(f"[{args.tag}] call 0 (cold, pays staging + JIT): {calls_ms[0]:.3f} ms\n")
    wall_s = time.perf_counter() - wall_t0
    counters = te.stats()
    evaluator = te.last_built_evaluator()
    describe = evaluator.describe() if evaluator is not None else {}
    warm = calls_ms[1:]
    log.write(
        f"[{args.tag}] calls={len(calls_ms)} cold={calls_ms[0]:.3f} ms "
        f"median_warm={_median(warm):.3f} ms mean_warm={_mean(warm):.3f} ms "
        f"min_warm={min(warm):.3f} ms wall={wall_s:.3f} s\n"
    )
    log.write(
        f"[{args.tag}] seam_calls={counters['seam_calls']} triton_runs={counters['triton_runs']} "
        f"builds={counters['builds']} cache_hits={counters['cache_hits']} "
        f"data_uploads={counters['data_uploads']}\n"
    )
    record: Dict[str, Any] = {
        "task": "eval64",
        "mode": "e2e",
        "reps_requested": int(args.reps),
        "calls_ms": calls_ms,
        "cold_call_ms": calls_ms[0] if calls_ms else None,
        "median_warm_ms": _median(warm),
        "mean_warm_ms": _mean(warm),
        "min_warm_ms": min(warm) if warm else None,
        "stdev_warm_ms": _stdev(warm),
        "max_warm_ms": max(warm) if warm else None,
        "wall_s": wall_s,
        "triton_call_count": int(counters["seam_calls"]),
        "expected_triton_call_count": int(args.reps),
        "triton_runs": int(counters["triton_runs"]),
        "expected_triton_runs": int(args.reps),
        "builds": int(counters["builds"]),
        "cache_hits": int(counters["cache_hits"]),
        "data_uploads": int(counters["data_uploads"]),
        "fallback_count": int(counters["fallbacks"]),
        "compile": {
            "first_call_ms": calls_ms[0] if calls_ms else None,
            "compile_ms": describe.get("compile_ms"),
            "construct_total_ms": describe.get("construct_total_ms"),
            "stage_upload_ms": describe.get("stage_upload_ms"),
            "n_builds": int(counters["builds"]),
        },
        "kernel": describe,
    }
    return record


# --------------------------------------------------------------------------- #
# pure (type 2): build once, then tight warm blocks (+ optional e2e companion)
# --------------------------------------------------------------------------- #
def time_block(evaluator, invocations: int) -> float:
    """One sync pair around ``invocations`` resident invocations (ms per block)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(int(invocations)):
        evaluator.run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def run_pure(args: argparse.Namespace, log: _Tee) -> Dict[str, Any]:
    problem, forest, x, labels = build_eval_case(args.fixture)
    use_MSE = not bool(args.use_mae)
    log.write(
        f"[{args.tag}] pure: pop={forest.pop_size} max_tree_len={forest.max_tree_len} "
        f"n_rows={int(x.shape[0])} invocations={args.invocations} warmup={args.warmup} "
        f"reps={args.reps} e2e_companion={args.e2e_companion}\n"
    )
    # The seam is installed up front (it resets the counters once, before anything is
    # measured): the pure blocks call the evaluator API directly, so nothing routes
    # through it until the companion half -- and the counter gate stays meaningful.
    install_seam(log, args.tag)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    t0 = time.perf_counter()
    evaluator = te.build_evaluator(
        forest,
        x,
        labels,
        use_MSE=use_MSE,
        output="loss",
        block_n=args.block_n,
        max_tree_len=args.max_tree_len,
    )
    torch.cuda.synchronize()
    construct_s = time.perf_counter() - t0
    describe = evaluator.describe()
    log.write(
        f"[{args.tag}] constructed: source={te.STAGING} pop={describe['pop']} "
        f"trace_len={describe['trace_len']} block_n={describe['block_n']} "
        f"num_warps={describe['num_warps']} launches/call={describe['launches_per_call']} "
        f"compile={describe['compile_ms']:.1f} ms stage_upload={describe['stage_upload_ms']:.1f} ms "
        f"construct_total={describe['construct_total_ms']:.1f} ms\n"
    )
    log.write(f"[{args.tag}] used ops: {describe['used_ops']} (fallthrough {describe['fallthrough_nodes']})\n")
    # ---- untimed warmup ----
    warmup = int(args.warmup)
    if warmup > 0:
        time_block(evaluator, warmup)
    blocks_ms: List[float] = []
    for rep in range(int(args.reps)):
        block_ms = time_block(evaluator, int(args.invocations))
        blocks_ms.append(block_ms)
        log.write(
            f"[{args.tag}] block {rep}: {block_ms:.3f} ms  "
            f"per-invocation {block_ms / int(args.invocations):.4f} ms\n"
        )
    per_invocation = [b / int(args.invocations) for b in blocks_ms]
    # ---- optional same-process e2e companion (prices the host glue) ----
    companion = None
    if int(args.e2e_companion) > 0:
        companion_calls: List[float] = []
        for call in range(int(args.e2e_companion)):
            torch.cuda.synchronize()
            c0 = time.perf_counter()
            problem.evaluate(forest)
            torch.cuda.synchronize()
            companion_calls.append((time.perf_counter() - c0) * 1000.0)
        companion = {
            "reps": int(args.e2e_companion),
            "calls_ms": companion_calls,
            "cold_call_ms": companion_calls[0],
            "median_warm_ms": _median(companion_calls[1:]),
            "mean_warm_ms": _mean(companion_calls[1:]),
            "min_warm_ms": min(companion_calls[1:]) if len(companion_calls) > 1 else None,
            "stdev_warm_ms": _stdev(companion_calls[1:]),
            "note": (
                "same-process e2e calls with the evaluator already built and warm; "
                "e2e_median - pure_median prices the per-call host glue around the kernel"
            ),
        }
        log.write(
            f"[{args.tag}] companion e2e: cold={companion_calls[0]:.3f} ms "
            f"median_warm={companion['median_warm_ms']:.3f} ms "
            f"(+{(companion['median_warm_ms'] or 0) - (_median(per_invocation) or 0):.3f} ms "
            f"over the pure median)\n"
        )
    wall_s = time.perf_counter() - wall_t0
    counters = te.stats()
    expected_runs = (
        warmup + int(args.reps) * int(args.invocations) + int(args.e2e_companion)
    )
    log.write(
        f"[{args.tag}] median_block={_median(blocks_ms):.3f} ms "
        f"median_per_invocation={_median(per_invocation):.4f} ms "
        f"min={min(per_invocation):.4f} ms mean={_mean(per_invocation):.4f} ms "
        f"wall={wall_s:.3f} s\n"
    )
    log.write(
        f"[{args.tag}] seam_calls={counters['seam_calls']} triton_runs={counters['triton_runs']} "
        f"(expected {expected_runs}) builds={counters['builds']}\n"
    )
    record: Dict[str, Any] = {
        "task": "eval-pure",
        "mode": "pure",
        "reps_requested": int(args.reps),
        "invocations": int(args.invocations),
        "warmup": {
            "invocations": warmup,
            "timed": False,
            "note": "single discarded block of resident invocations before the timed blocks",
        },
        "construct_s": construct_s,
        "blocks_ms": blocks_ms,
        "median_block_ms": _median(blocks_ms),
        "min_block_ms": min(blocks_ms) if blocks_ms else None,
        "mean_block_ms": _mean(blocks_ms),
        "per_invocation_ms": per_invocation,
        "median_per_invocation_ms": _median(per_invocation),
        "min_per_invocation_ms": min(per_invocation) if per_invocation else None,
        "mean_per_invocation_ms": _mean(per_invocation),
        "stdev_per_invocation_ms": _stdev(per_invocation),
        # envelope-facing fields (same names as the e2e record where they overlap)
        "calls_ms": [],
        "cold_call_ms": None,
        "median_warm_ms": _median(per_invocation),
        "mean_warm_ms": _mean(per_invocation),
        "min_warm_ms": min(per_invocation) if per_invocation else None,
        "stdev_warm_ms": _stdev(per_invocation),
        "wall_s": wall_s,
        "triton_call_count": int(counters["seam_calls"]),
        "expected_triton_call_count": int(args.e2e_companion),
        "triton_runs": int(counters["triton_runs"]),
        "expected_triton_runs": expected_runs,
        "builds": int(counters["builds"]),
        "cache_hits": int(counters["cache_hits"]),
        "data_uploads": int(counters["data_uploads"]),
        "fallback_count": int(counters["fallbacks"]),
        "compile": {
            "first_call_ms": None,
            "compile_ms": describe.get("compile_ms"),
            "construct_total_ms": describe.get("construct_total_ms"),
            "stage_upload_ms": describe.get("stage_upload_ms"),
            "n_builds": int(counters["builds"]),
        },
        "e2e_companion": companion,
        "kernel": describe,
    }
    return record


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log_path = args.log_out or f"{args.json_out}.log"
    log = _Tee(log_path)
    started = time.time()
    try:
        import evogp  # noqa: F401  (import AFTER the env is set)

        record: Dict[str, Any] = {
            "tag": args.tag,
            "side": "new",
            "fixture": os.path.abspath(args.fixture),
            "route": "triton",
            "result_mode": "loss" if not args.use_mae else "mae-loss",
            "seam_patched": False,
            "patch_target": te.SEAM_TARGET,
            "binding_style": te.binding_style(),
            "staging": te.STAGING,
            "triton_version": triton.__version__,
            "launches_per_call": 1,
            "op_coverage": te.covered_ops(),
            "exact_ops": list(te.EXACT_OPS),
            "kernel_name": "sr_eval_kernel",
            "env": _env_snapshot(),
            "gpu": _gpu_snapshot(),
            "evogp_commit": _git_commit(),
            "evogp_file": os.path.abspath(evogp.__file__),
            "python": sys.version.split()[0],
            "host": platform.node(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "complete": False,
        }
        if args.mode == "e2e":
            record.update(run_e2e(args, log))
        else:
            record.update(run_pure(args, log))
        record["seam_patched"] = bool(te.seam_is_patched())
        if not record["seam_patched"]:
            record["seam_patched_note"] = (
                "pure mode without --e2e-companion: the pure blocks call the evaluator API "
                "directly, so the dense seam is never exercised in this run"
            )
        record["complete"] = True
        record["elapsed_s_total"] = time.time() - started
        # ---- counter gate (the silent-fallback hazard) --------------------------
        problems: List[str] = []
        if record["triton_call_count"] != record["expected_triton_call_count"]:
            problems.append(
                f"triton_call_count={record['triton_call_count']} != expected "
                f"{record['expected_triton_call_count']} (a call did NOT route through triton)"
            )
        if record["triton_runs"] != record["expected_triton_runs"]:
            problems.append(
                f"triton_runs={record['triton_runs']} != expected {record['expected_triton_runs']}"
            )
        if record["fallback_count"]:
            problems.append(f"fallback_count={record['fallback_count']} (must be 0)")
        record["counter_gate"] = {"passed": not problems, "problems": problems}
        _write_json(args.json_out, record)
        log.write(
            f"[{args.tag}] wrote {os.path.abspath(args.json_out)} "
            f"(+ log {os.path.abspath(log_path)})\n"
        )
        if problems:
            for problem in problems:
                log.write(f"[{args.tag}] COUNTER GATE FAILED: {problem}\n")
            log.write(f"[{args.tag}] EXIT 1 (counter gate)\n")
            return 1
        log.write(
            f"[{args.tag}] counter gate PASSED: triton_call_count="
            f"{record['triton_call_count']}/{record['expected_triton_call_count']} "
            f"triton_runs={record['triton_runs']}/{record['expected_triton_runs']} "
            f"fallbacks=0\n"
        )
        log.write(f"[{args.tag}] EXIT 0\n")
        return 0
    except KeyboardInterrupt:
        log.write(f"[{args.tag}] interrupted\n")
        return 130
    except Exception as exc:  # noqa: BLE001 - the traceback must stay diagnosable
        import traceback

        log.write(f"[{args.tag}] FAILED: {type(exc).__name__}: {exc}\n")
        log.write(traceback.format_exc())
        log.write(f"[{args.tag}] EXIT 1\n")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
