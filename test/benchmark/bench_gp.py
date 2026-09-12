#!/usr/bin/env python
"""``bench_gp.py`` -- side-agnostic benchmark harness for the legacy-CUDA vs ETL comparison.

One file, two checkouts: it drives *either* engine through the same public
``evogp`` API, selected by ``PYTHONPATH``:

* NEW = this repository at HEAD (pure-torch/ETL "true jit" engine,
  ``evogp/tree/etl_engine.py``), default ``"numpy"`` backend;
* OLD = the legacy custom-CUDA checkout at ``b613503`` (``evogp.evogp_cuda`` /
  ``src/evogp/cuda/*.cu``; no CPU fallback there).

The methodology, workloads W1-W3 and the C0-C7 config matrix are frozen in
``test/benchmark/SPEC.md``; this script implements them (see ``test/benchmark/new/NOTES.md``
for the run log, per-rep semantics and caveats).

Usage (from the repository root, i.e. the checkout root)::

    PYTHONPATH=src python test/benchmark/bench_gp.py --task sr --pop 200 --gens 100 \
        --tree-len 64 --tag C1 --reps 2 --json-out test/benchmark/new/results/C1.json

    PYTHONPATH=src python test/benchmark/bench_gp.py --task eval64 --eval-reps 25 \
        --tag C5 --json-out test/benchmark/new/results/C5.json

Tasks: ``sr`` (W1: synthetic SR training), ``xor`` (W2: parity-table training),
``eval64`` (W3: repeated ``evaluate`` on the forest rebuilt from a committed
fixture) and ``evalfresh`` (W3': repeated ``evaluate`` on a freshly generated
random population).

Environment contract (set *before* importing evogp, as the SPEC requires):
``ETL_DISABLE_LOCATIONS=1`` by default (``=0`` with ``--no-disable-locations``);
``EVOGP_ETL_BACKEND`` / ``EVOGP_ETL_IREE_DEVICE`` / ``EVOGP_ETL_IREE_OPT_LEVEL``
are read by the new engine only and are simply inert on the old one.

The fixture (``--fixture``, default ``test/goldens/forest_random.npz`` relative to
the cwd) is only read by ``eval64``/``evalfresh``.  The legacy ``b613503`` checkout
has no ``test/goldens/``, so the old side must pass ``--fixture <absolute path>``.

Old-compat audit (every evogp symbol/kwarg this harness uses, checked against the
``b613503`` blobs; ``OK`` = present with a call-compatible signature)::

    symbol / call form                              b613503 file:line        result
    Forest(input_len, output_len, batch_node_value, src/evogp/tree/forest.py OK (ctor byte-identical)
           batch_node_type, batch_subtree_size)               :13
    Forest.random_generate(pop_size=, descriptor=)  src/evogp/tree/forest.py OK (staticmethod; body
                                                                      :43    dispatches evogp_cuda)
    GenerateDescriptor(max_tree_len, input_len,     src/evogp/tree/descriptor.py OK (same kwargs;
      output_len, const_prob, out_prob, using_funcs,               :31      const_range/sample_cnt
      layer_leaf_prob, const_range, sample_cnt,                             used only when
      max_layer_cnt, const_samples)                                        const_samples is None)
    GenerateDescriptor.update(**kwargs)            descriptor.py:186         OK (returns a new one)
    GeneticProgramming(initial_forest, crossover,   algorithm/                 OK
      mutation, selection, enable_pareto_front)     genetic_programming.py:21
    DefaultCrossover()                              algorithm/crossover/      OK
                                                    default.py:13
    DefaultMutation(mutation_rate=, descriptor=)    algorithm/mutation/        OK
                                                    default.py:19
    DefaultSelection(survival_rate=, elite_rate=)   algorithm/selection/       OK
                                                    default.py:15
    SymbolicRegression(func=, num_inputs=,          problem/                   OK
      num_data=, lower_bounds=, upper_bounds=)      symbolic_regression.py:11
    SymbolicRegression(datapoints=, labels=)        (same ctor)               OK
    SymbolicRegression.evaluate(forest)             (same file)               OK (default use_MSE=True;
      .problem_dim / .solution_dim                                            old side only adds a
                                                                              CombinedForest assert)
    StandardPipeline(algorithm, problem,            pipeline/standard.py:7    OK (byte-identical
      generation_limit=, is_show_details=)                                    between revisions)
    StandardPipeline.step() / .best_fitness         (same file)               OK (best_fitness is a
                                                                              float -inf, then a 0-dim
                                                                              CPU tensor -> float())
    import evogp / evogp.tree / evogp.algorithm /   src/evogp/__init__.py     OK (old evogp/tree/
      evogp.problem / evogp.pipeline                + package __init__ files   __init__.py imports
                                                                              evogp.evogp_cuda, i.e.
                                                                              the CUDA extension is a
                                                                              hard prerequisite there)
    torch.manual_seed / torch.cuda.manual_seed /    --                        OK (SPEC-wide)
      torch.cuda.synchronize / torch.cuda.get_device_name

Known asymmetric bits (do not fail the old side): ``enable_pareto_front`` is only
passed for ``--task xor`` (as in ``example/basic.py``); ``ETL_*`` env vars are inert
on the old side (C7 therefore degenerates to a plain legacy run); the old side reads
its RNG stream from the kernels, so population structures/fitness differ from the new
side -- compare *times*, use fitness only as a convergence sanity check.
"""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# task table (defaults follow SPEC.md; ``--flag`` always wins)
# ---------------------------------------------------------------------------

TRAIN_TASKS = ("sr", "xor")
EVAL_TASKS = ("eval64", "evalfresh")
TASKS = TRAIN_TASKS + EVAL_TASKS

#: ``None`` means "not applicable / required from the workload definition".
TASK_DEFAULTS = {
    "sr": {"pop": 200, "gens": 100, "tree_len": 64, "eval_reps": None},
    "xor": {"pop": 50, "gens": 100, "tree_len": 64, "eval_reps": None},
    "eval64": {"pop": None, "gens": None, "tree_len": None, "eval_reps": 25},
    "evalfresh": {"pop": 1000, "gens": None, "tree_len": 64, "eval_reps": 10},
}

DEFAULT_FIXTURE = os.path.join("test", "goldens", "forest_random.npz")

#: top-level JSON fields that mirror rep 0
TRAIN_MIRROR = (
    "wall_s",
    "generation_loop_s",
    "cold_first_gen_ms",
    "best_fitness_final",
    "partial",
    "projection",
    "construct_s",
    "complete",
)
EVAL_MIRROR = (
    "wall_s",
    "first_call_ms",
    "median_warm_ms",
    "partial",
    "projection",
    "construct_s",
    "complete",
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unspecified workload parameters fall back to the SPEC.md defaults "
        "for the chosen task; see TASK_DEFAULTS in this file.",
    )
    parser.add_argument("--task", choices=TASKS, required=True, help="workload")
    parser.add_argument("--pop", type=int, default=None, help="population size")
    parser.add_argument("--gens", type=int, default=None, help="generations to run")
    parser.add_argument("--tree-len", type=int, default=None, help="max_tree_len")
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed")
    parser.add_argument("--tag", default=None, help="label for logs/JSON (default: task)")
    parser.add_argument(
        "--json-out", default=None, help="JSON output path (rewritten after every step)"
    )
    parser.add_argument(
        "--reps", type=int, default=1, help="repeat the whole run N times (same seed)"
    )
    parser.add_argument(
        "--eval-reps", type=int, default=None, help="repeated evaluate() calls (eval tasks)"
    )
    parser.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        help=f"npz fixture for the eval tasks (default: {DEFAULT_FIXTURE}, cwd-relative)",
    )
    parser.add_argument(
        "--no-disable-locations",
        action="store_true",
        help="force ETL_DISABLE_LOCATIONS=0 (verbose ETL call-site capture)",
    )
    parser.add_argument(
        "--max-wall-s",
        type=float,
        default=None,
        help="soft wall-clock cap per rep, checked after every generation",
    )
    parser.add_argument(
        "--min-gens",
        type=int,
        default=0,
        help="never stop early (--max-wall-s) before this many completed generations",
    )
    parser.add_argument("--notes", default="", help="free-form text stored in the JSON")
    args = parser.parse_args(argv)

    for key, value in TASK_DEFAULTS[args.task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.tag is None:
        args.tag = args.task

    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.min_gens < 0:
        parser.error("--min-gens must be >= 0")
    if args.task in TRAIN_TASKS:
        for key in ("pop", "gens", "tree_len"):
            if getattr(args, key) is None or getattr(args, key) < 1:
                parser.error(f"--{key.replace('_', '-')} is required (and > 0) for --task {args.task}")
    if args.task == "evalfresh":
        for key in ("pop", "tree_len"):
            if getattr(args, key) is None or getattr(args, key) < 1:
                parser.error(f"--{key.replace('_', '-')} is required (and > 0) for --task evalfresh")
    if args.task in EVAL_TASKS and (args.eval_reps is None or args.eval_reps < 1):
        parser.error(f"--eval-reps must be >= 1 for --task {args.task}")
    return args


def apply_env(args):
    """Fix the engine environment.  MUST run before evogp (or torch) is imported."""
    if args.no_disable_locations:
        os.environ["ETL_DISABLE_LOCATIONS"] = "0"
    else:
        os.environ.setdefault("ETL_DISABLE_LOCATIONS", "1")
    # Old side: these are inert.  New side: an opt-in backend may be requested
    # through EVOGP_ETL_BACKEND by the caller; nothing is forced here.
    return {k: v for k, v in os.environ.items() if k.startswith(("ETL_", "EVOGP_"))}


# ---------------------------------------------------------------------------
# environment / identity snapshots
# ---------------------------------------------------------------------------


def import_modules():
    """Import evogp *after* ``apply_env`` and return the symbols the harness uses."""
    import evogp
    from evogp.algorithm import (
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    return argparse.Namespace(
        evogp=evogp,
        Forest=Forest,
        GenerateDescriptor=GenerateDescriptor,
        GeneticProgramming=GeneticProgramming,
        DefaultCrossover=DefaultCrossover,
        DefaultMutation=DefaultMutation,
        DefaultSelection=DefaultSelection,
        StandardPipeline=StandardPipeline,
        SymbolicRegression=SymbolicRegression,
    )


def detect_side():
    """``"new"`` when the ETL engine module exists, ``"old"`` when it does not."""
    try:
        if importlib.util.find_spec("evogp.tree.etl_engine") is None:
            return "old"
        return "new"
    except Exception:
        return "unknown"


def _guarded(label, fn, default=None):
    try:
        return fn()
    except Exception as exc:  # never let metadata collection kill a run
        return default if default is not None else f"<{label}: {type(exc).__name__}: {exc}>"


def etl_version():
    """Version of the ``etl`` package; guarded so the old side can never crash on it."""
    try:
        from importlib.metadata import version as _md_version

        return _md_version("etl")
    except Exception:
        pass
    try:
        import etl  # noqa: F401

        return getattr(etl, "__version__", "unknown")
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def repo_snapshot(evogp_file):
    info = {
        "cwd": os.getcwd(),
        "evogp_file": evogp_file,
        "evogp_commit": None,
        "evogp_commit_error": None,
    }
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if done.returncode == 0:
            info["evogp_commit"] = done.stdout.strip()
        else:
            info["evogp_commit_error"] = (done.stderr or "").strip() or f"git exit {done.returncode}"
    except Exception as exc:
        info["evogp_commit_error"] = f"{type(exc).__name__}: {exc}"
    return info


def gpu_snapshot():
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    info["device_count"] = _guarded("device_count", torch.cuda.device_count)
    info["name"] = _guarded("name", lambda: torch.cuda.get_device_name(0))
    info["total_memory_bytes"] = _guarded(
        "total_memory", lambda: int(torch.cuda.get_device_properties(0).total_memory)
    )
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    return info


def env_snapshot():
    env = {k: v for k, v in os.environ.items() if k.startswith(("ETL_", "EVOGP_"))}
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


# ---------------------------------------------------------------------------
# JSON writing (atomic + incremental)
# ---------------------------------------------------------------------------


def write_json_atomic(path, payload):
    """Rewrite the whole JSON atomically, so an external kill still leaves usable data."""
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".bench_gp_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o644)  # mkstemp creates 0600; keep results shareable
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# workload construction (public evogp API only, identical on both sides)
# ---------------------------------------------------------------------------


def sr_target(x):
    """W1 target function (verbatim from ``src/evogp/sr_test.py``)."""
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


def w1_problem(mods):
    """W1 problem: ints for the bounds (the fixed-width scalar check quirk)."""
    return mods.SymbolicRegression(
        func=sr_target, num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5
    )


def w1_descriptor(mods, tree_len):
    return mods.GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=2,
        output_len=1,
        const_prob=0.5,
        out_prob=0.5,
        using_funcs=["+", "-", "*", "/"],
        layer_leaf_prob=0.2,
        const_range=(-5, 5),
        sample_cnt=8,
        max_layer_cnt=5,
    )


XOR_INPUTS = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]
XOR_OUTPUTS = [[0], [1], [1], [0], [1], [0], [0], [1]]


def w2_problem(mods):
    """W2 problem: the inline 8x3 parity table from ``example/basic.py`` (on cuda)."""
    datapoints = torch.tensor(XOR_INPUTS, dtype=torch.float, device="cuda")
    labels = torch.tensor(XOR_OUTPUTS, dtype=torch.float, device="cuda")
    return mods.SymbolicRegression(datapoints=datapoints, labels=labels)


def w2_descriptor(mods, problem, tree_len):
    return mods.GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=problem.problem_dim,
        output_len=problem.solution_dim,
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=5,
        const_samples=[-1, 0, 1],
    )


def build_problem_and_descriptor(mods, args):
    """(problem, descriptor, one-line description) for a training task."""
    if args.task == "sr":
        problem = w1_problem(mods)
        descriptor = w1_descriptor(mods, args.tree_len)
        desc = (
            f"W1 synthetic SR: SymbolicRegression(func=x0**4/(x0**4+1)+x1**4/(x1**4+1), "
            f"num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5); "
            f"GenerateDescriptor(max_tree_len={args.tree_len}, input_len=2, output_len=1, "
            f"const_prob=0.5, out_prob=0.5, using_funcs=['+','-','*','/'], layer_leaf_prob=0.2, "
            f"const_range=(-5,5), sample_cnt=8, max_layer_cnt=5)"
        )
    elif args.task == "xor":
        problem = w2_problem(mods)
        descriptor = w2_descriptor(mods, problem, args.tree_len)
        desc = (
            f"W2 parity: SymbolicRegression(datapoints=8x3 table, labels=8x1); "
            f"GenerateDescriptor(max_tree_len={args.tree_len}, input_len=problem.problem_dim=3, "
            f"output_len=problem.solution_dim=1, using_funcs=['+','-','*','/'], max_layer_cnt=5, "
            f"const_samples=[-1,0,1]); enable_pareto_front=True"
        )
    else:
        raise AssertionError(f"not a training task: {args.task}")
    return problem, descriptor, desc


def build_algorithm(mods, args, descriptor, initial_forest):
    kwargs = {}
    if args.task == "xor":  # mirrors example/basic.py
        kwargs["enable_pareto_front"] = True
    return mods.GeneticProgramming(
        initial_forest=initial_forest,
        crossover=mods.DefaultCrossover(),
        mutation=mods.DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=mods.DefaultSelection(survival_rate=0.3, elite_rate=0.01),
        **kwargs,
    )


def to_cuda(array, dtype):
    """Copy a fixture array into a writable contiguous torch tensor on cuda."""
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).to("cuda")


def build_eval_case(mods, args):
    """(problem, forest, eval_kind, description) for an eval-only task."""
    data = np.load(args.fixture)
    if args.task == "eval64":
        x = to_cuda(data["x"], np.float32)
        labels = to_cuda(data["labels"], np.float32)
        forest = mods.Forest(
            input_len=int(x.shape[1]),
            output_len=int(labels.shape[1]),
            batch_node_value=to_cuda(data["forest__node_value"], np.float32),
            batch_node_type=to_cuda(data["forest__node_type"], np.int16),
            batch_subtree_size=to_cuda(data["forest__subtree_size"], np.int16),
        )
        problem = mods.SymbolicRegression(datapoints=x, labels=labels)
        desc = (
            f"W3 fixed fixture: Forest(input_len={forest.input_len}, output_len={forest.output_len}, "
            f"pop_size={forest.pop_size}, max_tree_len={forest.max_tree_len}) rebuilt from "
            f"{args.fixture}[forest__node_value/_node_type/_subtree_size] via the class constructor + "
            f"SymbolicRegression(datapoints=x[{tuple(x.shape)}], labels=labels[{tuple(labels.shape)}])"
        )
        return problem, forest, "fixture", desc

    # evalfresh: fresh random population, fixture inputs/labels
    x = to_cuda(data["x"], np.float32)
    labels = to_cuda(data["labels"], np.float32)
    problem = mods.SymbolicRegression(datapoints=x, labels=labels)
    descriptor = w1_descriptor(mods, args.tree_len)
    forest = mods.Forest.random_generate(pop_size=args.pop, descriptor=descriptor)
    desc = (
        f"W3' fresh population: Forest.random_generate(pop_size={args.pop}, "
        f"descriptor=W1-style GenerateDescriptor(max_tree_len={args.tree_len})) + "
        f"SymbolicRegression(datapoints=x[{tuple(x.shape)}], labels=labels[{tuple(labels.shape)}]) "
        f"from {args.fixture}"
    )
    return problem, forest, "fresh", desc


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def fmt_ms(value, spec=".1f"):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return f"{value:{spec}}"


def fmt_fit(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def new_rep_record(rep, kind):
    record = {
        "rep": rep,
        "construct_s": None,
        "wall_s": None,
        "partial": False,
        "projection": None,
        # `complete` = "the harness finished writing this record".  It stays False
        # only when the process died hard (SIGKILL / crash), leaving a truncated
        # snapshot: the JSON itself is still valid thanks to the incremental writes.
        "complete": False,
    }
    if kind == "train":
        record.update(
            per_gen_ms=[],
            per_gen_best_fitness=[],
            generation_loop_s=None,
            cold_first_gen_ms=None,
            best_fitness_final=None,
        )
    else:
        record.update(per_call_ms=[], first_call_ms=None, median_warm_ms=None)
    return record


def train_projection(per_gen_ms, gens_done, gens_requested):
    mean_ms = mean_or_none(per_gen_ms)
    return {
        "per_gen_ms_mean": mean_ms,
        "gens_completed": gens_done,
        "gens_requested": gens_requested,
        "projected_total_s_for_requested_gens": (
            None if mean_ms is None else mean_ms * gens_requested / 1000.0
        ),
    }


def eval_projection(per_call_ms, calls_done, calls_requested):
    mean_ms = mean_or_none(per_call_ms)
    return {
        "per_call_ms_mean": mean_ms,
        "calls_completed": calls_done,
        "calls_requested": calls_requested,
        "projected_total_s_for_requested_calls": (
            None if mean_ms is None else mean_ms * calls_requested / 1000.0
        ),
    }


def mirror_rep0(payload, record, keys):
    if record["rep"] != 0:
        return
    for key in keys:
        payload[key] = record.get(key)


def summary_training(tag, side, record, reps):
    per_gen = record["per_gen_ms"]
    gens = len(per_gen)
    wall = record["wall_s"]
    flag = " (PARTIAL)" if record["partial"] else ""
    return (
        f"[{tag}] side={side} rep{record['rep']}/{reps}: {gens} gens in {fmt_ms(wall)}s, "
        f"per-gen median {fmt_ms(statistics.median(per_gen) if per_gen else None)}ms, "
        f"first-gen {fmt_ms(record['cold_first_gen_ms'])}ms, "
        f"best_fitness_final={fmt_fit(record['best_fitness_final'])}{flag}"
    )


def summary_eval(tag, side, record, reps):
    calls = record["per_call_ms"]
    flag = " (PARTIAL)" if record["partial"] else ""
    return (
        f"[{tag}] side={side} rep{record['rep']}/{reps}: first {fmt_ms(record['first_call_ms'])}ms, "
        f"warm-median {fmt_ms(record['median_warm_ms'])}ms ({len(calls)} calls){flag}"
    )


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def run_training(mods, args, payload, flush):
    """Manual ``pipeline.step()`` loop -- the same code path as ``pipeline.run()``."""
    interrupted = False
    for rep in range(args.reps):
        record = new_rep_record(rep, "train")
        payload["per_rep"].append(record)
        if rep == 0:  # top-level arrays share rep 0's list objects
            payload["per_gen_ms"] = record["per_gen_ms"]
            payload["per_gen_best_fitness"] = record["per_gen_best_fitness"]

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()  # wall_s starts just before construction
        problem, descriptor, description = build_problem_and_descriptor(mods, args)
        initial_forest = mods.Forest.random_generate(pop_size=args.pop, descriptor=descriptor)
        algorithm = build_algorithm(mods, args, descriptor, initial_forest)
        pipeline = mods.StandardPipeline(
            algorithm,
            problem,
            generation_limit=args.gens,
            is_show_details=False,
        )
        torch.cuda.synchronize()
        record["construct_s"] = time.perf_counter() - wall_t0
        payload["problem"] = description
        mirror_rep0(payload, record, TRAIN_MIRROR)
        flush()

        try:
            for gen in range(args.gens):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pipeline.step()
                torch.cuda.synchronize()
                record["per_gen_ms"].append((time.perf_counter() - t0) * 1000.0)
                record["per_gen_best_fitness"].append(float(pipeline.best_fitness))
                record["cold_first_gen_ms"] = record["per_gen_ms"][0]
                record["best_fitness_final"] = record["per_gen_best_fitness"][-1]
                mirror_rep0(payload, record, TRAIN_MIRROR)
                flush()  # a hard kill still leaves every finished generation
                if (
                    args.max_wall_s is not None
                    and time.perf_counter() - wall_t0 >= args.max_wall_s
                    and gen + 1 >= args.min_gens
                ):
                    record["partial"] = True
                    record["projection"] = train_projection(
                        record["per_gen_ms"], gen + 1, args.gens
                    )
                    print(
                        f"[{args.tag}] side={payload['side']} soft wall cap "
                        f"--max-wall-s={args.max_wall_s}s reached after {gen + 1}/{args.gens} gens",
                        flush=True,
                    )
                    break
        except KeyboardInterrupt:
            interrupted = True
            record["partial"] = True
            print(f"[{args.tag}] interrupted after {len(record['per_gen_ms'])} gens", flush=True)

        torch.cuda.synchronize()
        record["wall_s"] = time.perf_counter() - wall_t0
        record["generation_loop_s"] = sum(record["per_gen_ms"]) / 1000.0
        record["complete"] = True
        mirror_rep0(payload, record, TRAIN_MIRROR)
        flush()
        print(summary_training(args.tag, payload["side"], record, args.reps), flush=True)
        if interrupted:
            break
    return interrupted


def run_eval(mods, args, payload, flush):
    """Repeated ``problem.evaluate(forest)`` on a fixed population."""
    interrupted = False
    run_t0 = time.perf_counter()
    calls_requested = args.reps * args.eval_reps
    for rep in range(args.reps):
        record = new_rep_record(rep, "eval")
        payload["per_rep"].append(record)
        if rep == 0:
            payload["per_call_ms"] = record["per_call_ms"]

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        problem, forest, eval_kind, description = build_eval_case(mods, args)
        torch.cuda.synchronize()
        record["construct_s"] = time.perf_counter() - wall_t0
        payload["problem"] = description
        payload["eval_kind"] = eval_kind
        payload["config"]["pop"] = int(forest.pop_size)
        payload["config"]["tree_len"] = int(forest.max_tree_len)
        mirror_rep0(payload, record, EVAL_MIRROR)
        flush()

        try:
            for _ in range(args.eval_reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                problem.evaluate(forest)
                torch.cuda.synchronize()
                record["per_call_ms"].append((time.perf_counter() - t0) * 1000.0)
                mirror_rep0(payload, record, EVAL_MIRROR)
                flush()
        except KeyboardInterrupt:
            interrupted = True
            record["partial"] = True
            print(f"[{args.tag}] interrupted after {len(record['per_call_ms'])} calls", flush=True)

        torch.cuda.synchronize()
        record["wall_s"] = time.perf_counter() - wall_t0
        record["first_call_ms"] = record["per_call_ms"][0] if record["per_call_ms"] else None
        warm = record["per_call_ms"][1:]
        record["median_warm_ms"] = statistics.median(warm) if len(warm) >= 2 else None
        record["complete"] = True
        mirror_rep0(payload, record, EVAL_MIRROR)
        flush()
        print(summary_eval(args.tag, payload["side"], record, args.reps), flush=True)

        if interrupted:
            break
        calls_done = sum(len(r["per_call_ms"]) for r in payload["per_rep"])
        if (
            args.max_wall_s is not None
            and time.perf_counter() - run_t0 >= args.max_wall_s
            and calls_done >= args.min_gens
        ):
            projection = eval_projection(
                [ms for r in payload["per_rep"] for ms in r["per_call_ms"]],
                calls_done,
                calls_requested,
            )
            record["partial"] = True
            record["projection"] = projection
            payload["partial"] = True
            payload["projection"] = projection
            print(
                f"[{args.tag}] side={payload['side']} soft wall cap "
                f"--max-wall-s={args.max_wall_s}s reached after {calls_done}/{calls_requested} calls",
                flush=True,
            )
            break
    return interrupted


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_payload(args, side, mods, started_at):
    if args.task in TRAIN_TASKS:
        payload = {
            "tag": args.tag,
            "side": side,
            "task": args.task,
            "evogp_file": None,
            "evogp_commit": None,
            "etl_version": None,
            "problem": None,
            "config": {
                "pop": args.pop,
                "gens": args.gens,
                "tree_len": args.tree_len,
                "seed": args.seed,
                "reps": args.reps,
            },
            "env": {},
            "gpu": {},
            "wall_s": None,
            "generation_loop_s": None,
            "per_gen_ms": [],
            "per_gen_best_fitness": [],
            "cold_first_gen_ms": None,
            "best_fitness_final": None,
            "partial": False,
            "projection": None,
            "complete": False,
            "per_rep": [],
            "notes": args.notes,
            "started_at": started_at,
            "finished_at": None,
        }
    else:
        payload = {
            "tag": args.tag,
            "side": side,
            "task": args.task,
            "eval_kind": None,
            "fixture": os.path.abspath(args.fixture),
            "evogp_file": None,
            "evogp_commit": None,
            "etl_version": None,
            "problem": None,
            "config": {
                "pop": args.pop,
                "tree_len": args.tree_len,
                "seed": args.seed,
                "reps": args.reps,
                "eval_reps": args.eval_reps,
            },
            "env": {},
            "gpu": {},
            "per_call_ms": [],
            "first_call_ms": None,
            "median_warm_ms": None,
            "wall_s": None,
            "partial": False,
            "projection": None,
            "complete": False,
            "per_rep": [],
            "notes": args.notes,
            "started_at": started_at,
            "finished_at": None,
        }
    repo = repo_snapshot(getattr(mods.evogp, "__file__", None))
    payload["evogp_file"] = repo["evogp_file"]
    payload["evogp_commit"] = repo["evogp_commit"]
    payload["evogp_commit_error"] = repo["evogp_commit_error"]
    payload["cwd"] = repo["cwd"]
    payload["etl_version"] = etl_version()
    payload["env"] = env_snapshot()
    payload["gpu"] = gpu_snapshot()
    payload["harness"] = os.path.abspath(__file__)
    return payload


def main(argv=None):
    args = parse_args(argv)
    apply_env(args)
    started_at = now_iso()
    t0 = time.perf_counter()
    mods = import_modules()
    side = detect_side()
    payload = build_payload(args, side, mods, started_at)

    json_out = args.json_out

    def flush():
        write_json_atomic(json_out, payload)

    flush()  # metadata on disk before the first heavy step
    print(
        f"[{args.tag}] harness=bench_gp.py side={side} task={args.task} "
        f"pop={args.pop} gens={args.gens} tree_len={args.tree_len} seed={args.seed} "
        f"reps={args.reps} eval_reps={args.eval_reps} "
        f"ETL_DISABLE_LOCATIONS={payload['env'].get('ETL_DISABLE_LOCATIONS')} "
        f"EVOGP_ETL_BACKEND={payload['env'].get('EVOGP_ETL_BACKEND')} "
        f"CUDA_VISIBLE_DEVICES={payload['env'].get('CUDA_VISIBLE_DEVICES')} "
        f"gpu={payload['gpu'].get('name')} evogp={payload['evogp_file']} "
        f"commit={payload['evogp_commit']} etl={payload['etl_version']} "
        f"json_out={json_out}",
        flush=True,
    )

    interrupted = False
    try:
        if args.task in TRAIN_TASKS:
            interrupted = run_training(mods, args, payload, flush)
        else:
            interrupted = run_eval(mods, args, payload, flush)
    except KeyboardInterrupt:
        interrupted = True
        payload["partial"] = True
        print(f"[{args.tag}] interrupted outside a timed step", flush=True)

    payload["finished_at"] = now_iso()
    payload["harness_wall_s"] = time.perf_counter() - t0
    payload["complete"] = True  # this is the harness' final, deliberate write
    flush()
    print(
        f"[{args.tag}] done side={side} interrupted={interrupted} "
        f"harness_wall_s={payload['harness_wall_s']:.1f} json_out={json_out}",
        flush=True,
    )
    if json_out is None:
        print(json.dumps(payload, indent=2, allow_nan=True), flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
