#!/usr/bin/env python3
"""OLD-side benchmark harness: `test/benchmark/legacy/run_matrix.py`.

Legacy ("old") engine side of `test/benchmark/SPEC.md` — the pre-ETL revision `b613503` (custom CUDA
kernels bound as `evogp.evogp_cuda`; `evogp.tree.etl_engine` does not exist). Implements the frozen
legacy config matrix, one JSON per run plus a per-generation stdout trace (the raw timing log):
C1/C2/C2b = W1 synthetic SR, C4 = W2 XOR parity, C5/C5b = W3 eval-only microbench. C0 (ETL tracing
control) and C7 (IREE backend) are NEW-engine-only and are rejected here.

Runner invocation (CWD = legacy checkout root, so `PYTHONPATH=src` resolves `evogp`):

    cd /tmp/evogp_legacy_bench && CUDA_VISIBLE_DEVICES=<i> \\
        PYTHONPATH=src:/tmp/evogp_t1a263/siteutils \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python -u <abs>/test/benchmark/legacy/run_matrix.py \\
        --configs C1 --out <abs>/test/benchmark/legacy/results

Design notes (all deliberate):

* No top-level import of torch / numpy / evogp — everything heavy is imported lazily inside `api()`,
  so `--help`, `--list` and `py_compile` work without the compiled extension.
* `--out` and `--fixture` default relative to this file (`dirname(abspath(__file__))`), so the script
  is CWD-independent; `PYTHONPATH` (not the CWD) is what makes `evogp` importable.
* Timing follows SPEC: `torch.cuda.synchronize()` immediately before and after every timed region, a
  manual `pipeline.step()` loop (identical code path to `run()`), `is_show_details=False`,
  `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` before any construction. Each rep of a
  multi-rep config is a *fresh seeded construction*.
* Each JSON is written immediately after its run completes, so a later failure still leaves
  checkpoints for the configs that finished.
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.normpath(
    os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "goldens", "forest_random.npz")
)
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "results")

# Frozen config matrix (test/benchmark/SPEC.md -> "Config matrix"). Do NOT tune.
# `evals is None` => timed pipeline run; otherwise an eval-only microbench with that many calls.
CONFIGS = {
    # id: workload, problem label, pop, gens, tree_len, reps, evals (None => timed pipeline run)
    "C1":  {"workload": "W1", "problem": "W1_synthetic_sr", "pop": 200,  "gens": 100, "tree_len": 64,  "reps": 2, "evals": None},
    "C2":  {"workload": "W1", "problem": "W1_synthetic_sr", "pop": 1000, "gens": 100, "tree_len": 128, "reps": 1, "evals": None},
    "C2b": {"workload": "W1", "problem": "W1_synthetic_sr", "pop": 1000, "gens": 20,  "tree_len": 128, "reps": 1, "evals": None},
    "C4":  {"workload": "W2", "problem": "W2_xor_parity", "pop": 50,   "gens": 100, "tree_len": 64,  "reps": 2, "evals": None},
    "C5":  {"workload": "W3", "problem": "W3_eval_only", "eval_kind": "fixed_fixture_forest", "pop": 64, "gens": 0, "tree_len": 63, "reps": 1, "evals": 25},
    "C5b": {"workload": "W3", "problem": "W3_eval_only", "eval_kind": "fresh_random_forest", "pop": 1000, "gens": 0, "tree_len": 64, "reps": 1, "evals": 10},
}
# Configs that exist in SPEC.md but belong to the NEW engine side.
NEW_ONLY_CONFIGS = ("C0", "C7")

SEED = 0

# --- small helpers (no heavy imports) ------------------------------------------------------
def tags_for(cfg_id, meta):
    """Run tags for a config: `C1` with reps=2 -> ['C1_r1', 'C1_r2']."""
    if meta["reps"] > 1:
        return [f"{cfg_id}_r{i}" for i in range(1, meta["reps"] + 1)]
    return [cfg_id]


def env_snapshot():
    """The three SPEC env vars, exactly as read (None when unset; OLD ignores the ETL ones)."""
    return {
        key: os.environ.get(key)
        for key in ("ETL_DISABLE_LOCATIONS", "EVOGP_ETL_BACKEND", "CUDA_VISIBLE_DEVICES")
    }


def write_json(out_dir, tag, record):
    """Write one run record immediately (checkpoint) — plain floats/ints/lists only."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(f"[{tag}] wrote {path}", flush=True)
    return path


def detect_side():
    """'old' when the ETL engine module is absent (spec), else 'new'."""
    return "old" if importlib.util.find_spec("evogp.tree.etl_engine") is None else "new"


# --- lazy API access -----------------------------------------------------------------------
_API = None


def _import_stack():
    import numpy
    import torch

    import evogp
    from evogp.tree import Forest, GenerateDescriptor
    from evogp.algorithm import (
        GeneticProgramming,
        DefaultSelection,
        DefaultMutation,
        DefaultCrossover,
    )
    from evogp.problem import SymbolicRegression
    from evogp.pipeline import StandardPipeline

    return types.SimpleNamespace(
        numpy=numpy,
        torch=torch,
        evogp_file=os.path.realpath(evogp.__file__),
        side=detect_side(),
        Forest=Forest,
        GenerateDescriptor=GenerateDescriptor,
        GeneticProgramming=GeneticProgramming,
        DefaultSelection=DefaultSelection,
        DefaultMutation=DefaultMutation,
        DefaultCrossover=DefaultCrossover,
        SymbolicRegression=SymbolicRegression,
        StandardPipeline=StandardPipeline,
    )


def api():
    """Import torch/numpy/evogp once, on first use (keeps `--list` import-free)."""
    global _API
    if _API is None:
        _API = _import_stack()
    return _API


def gpu_label(torch):
    return (
        f"{torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
    )


# --- workload builders (one per workload family) -------------------------------------------
def build_w1(meta):
    """W1 (C1/C2/C2b): synthetic SR problem + descriptor + algorithm, exactly as SPEC/sr_test."""
    A = api()
    tree_len = meta["tree_len"]

    def func(x):
        val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
        return val.reshape(-1)

    problem = A.SymbolicRegression(
        func=func, num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5
    )
    descriptor = A.GenerateDescriptor(
        max_tree_len=tree_len, input_len=2, output_len=1, const_prob=0.5, out_prob=0.5,
        using_funcs=["+", "-", "*", "/"], layer_leaf_prob=0.2, const_range=(-5, 5), sample_cnt=8,
        max_layer_cnt=5,
    )
    algorithm = A.GeneticProgramming(
        initial_forest=A.Forest.random_generate(pop_size=meta["pop"], descriptor=descriptor),
        crossover=A.DefaultCrossover(),
        mutation=A.DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=A.DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    notes = (
        "W1 synthetic SR (sr_test problem): func=x0^4/(x0^4+1)+x1^4/(x1^4+1), SymbolicRegression(func=..., "
        "num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5) (ints); descriptor "
        f"max_tree_len={tree_len}, input_len=2, output_len=1, const_prob=0.5, out_prob=0.5, "
        "using_funcs=[+,-,*,/], layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8, max_layer_cnt=5; "
        "mutation DefaultMutation(mutation_rate=0.2, descriptor.update(max_layer_cnt=3)); selection "
        "DefaultSelection(survival_rate=0.3, elite_rate=0.01); StandardPipeline(generation_limit=gens, "
        "is_show_details=False); manual sync-bracketed pipeline.step() loop (torch.cuda.synchronize() before "
        "and after each step), per_gen_ms via time.perf_counter; best_fitness = pipeline running max "
        "(MAXIMIZED, NaN -> -inf); wall_s = construction + all steps; legacy custom-CUDA engine "
        "(ETL_* env vars do not apply)"
    )
    return problem, algorithm, notes


def build_w2(meta):
    """W2 (C4): 8-row 3-input XOR parity, exactly as example/basic.py."""
    A = api()
    torch = A.torch
    xor_inputs = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float,
        device="cuda",
    )
    xor_outputs = torch.tensor(
        [[0], [1], [1], [0], [1], [0], [0], [1]],
        dtype=torch.float,
        device="cuda",
    )
    problem = A.SymbolicRegression(datapoints=xor_inputs, labels=xor_outputs)
    descriptor = A.GenerateDescriptor(
        max_tree_len=64, input_len=problem.problem_dim, output_len=problem.solution_dim,
        using_funcs=["+", "-", "*", "/"], max_layer_cnt=5, const_samples=[-1, 0, 1],
    )
    algorithm = A.GeneticProgramming(
        initial_forest=A.Forest.random_generate(pop_size=meta["pop"], descriptor=descriptor),
        crossover=A.DefaultCrossover(),
        mutation=A.DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=A.DefaultSelection(survival_rate=0.3, elite_rate=0.01),
        enable_pareto_front=True,
    )
    notes = (
        "W2 XOR parity (example/basic.py verbatim): 8-row 3-input XOR table as float32 cuda tensors; "
        "SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS) (execute_mode='auto'); descriptor "
        "max_tree_len=64, input_len=problem.problem_dim(3), output_len=problem.solution_dim(1), "
        "using_funcs=[+,-,*,/], max_layer_cnt=5, const_samples=[-1,0,1], const_prob/out_prob/layer_leaf_prob "
        "at defaults (0.5/0.5/0.2); mutation DefaultMutation(mutation_rate=0.2, "
        "descriptor.update(max_layer_cnt=3)); selection DefaultSelection(survival_rate=0.3, elite_rate=0.01); "
        "enable_pareto_front=True; StandardPipeline(generation_limit=gens, is_show_details=False); manual "
        "sync-bracketed pipeline.step() loop; wall_s = construction + all steps; legacy custom-CUDA engine "
        "(ETL_* env vars do not apply)"
    )
    return problem, algorithm, notes


def _fixture_arrays(fixture_path):
    """Load the W3 fixture with numpy (raises a clear error when the file is missing)."""
    A = api()
    if not os.path.isfile(fixture_path):
        raise FileNotFoundError(
            f"W3 fixture not found: {fixture_path} "
            f"(pass --fixture PATH; default is {DEFAULT_FIXTURE})"
        )
    return A.numpy.load(fixture_path)


def _fixture_problem(fixture_path):
    """Shared W3 inputs: x (64,2) / labels (64,1) from test/goldens/forest_random.npz."""
    A = api()
    data = _fixture_arrays(fixture_path)
    x = A.torch.tensor(data["x"], device="cuda")
    labels = A.torch.tensor(data["labels"], device="cuda")
    return A.SymbolicRegression(datapoints=x, labels=labels), x.shape[0]


def build_c5(fixture_path):
    """C5: rebuild the 64-tree / 63-node fixture forest with the Forest class constructor."""
    A = api()
    data = _fixture_arrays(fixture_path)
    torch = A.torch
    forest = A.Forest(
        input_len=2,
        output_len=1,
        batch_node_value=torch.tensor(data["forest__node_value"], device="cuda"),
        batch_node_type=torch.tensor(data["forest__node_type"], device="cuda"),
        batch_subtree_size=torch.tensor(data["forest__subtree_size"], device="cuda"),
    )
    problem, num_data = _fixture_problem(fixture_path)
    notes = (
        "W3 eval-only microbench on the committed fixture forest: Forest(input_len=2, output_len=1) rebuilt "
        "from forest_random.npz keys forest__node_value (64,63) f32 / forest__node_type (64,63) i16 / "
        "forest__subtree_size (64,63) i16, tensors on cuda; problem = "
        f"SymbolicRegression(datapoints=x({num_data},2), labels=labels({num_data},1)) from the same fixture; "
        f"{CONFIGS['C5']['evals']} x problem.evaluate(forest), sync-bracketed; timings in ms; first_call_ms = "
        "cold call 1, median_warm_ms = median of calls 2..N; RNG-independent (no generation); legacy "
        "custom-CUDA engine"
    )
    return problem, forest, notes


def build_c5b(fixture_path):
    """C5b: fresh random forest, pop 1000 / tree_len 64 (fixture's documented descriptor params)."""
    A = api()
    descriptor = A.GenerateDescriptor(
        max_tree_len=64, input_len=2, output_len=1, using_funcs=["+", "-", "*", "/"], max_layer_cnt=4,
        layer_leaf_prob=0.2, const_prob=0.5, out_prob=0.5,
        const_samples=[-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2],
    )
    forest = A.Forest.random_generate(pop_size=CONFIGS["C5b"]["pop"], descriptor=descriptor)
    problem, num_data = _fixture_problem(fixture_path)
    notes = (
        "W3 eval-only microbench on a fresh random forest: Forest.random_generate(pop_size=1000, "
        "GenerateDescriptor(max_tree_len=64, input_len=2, output_len=1, using_funcs=[+,-,*,/], "
        "max_layer_cnt=4, layer_leaf_prob=0.2, const_prob=0.5, out_prob=0.5, "
        "const_samples=[-2,-1,-0.5,-0.25,0.25,0.5,1,2])) — the descriptor params documented for the committed "
        "fixture in test/goldens/README.md, with this run's tree_len/pop (64/1000); evaluated on the fixture's "
        f"x({num_data},2)/labels({num_data},1) via SymbolicRegression(datapoints=x, labels=labels); "
        f"{CONFIGS['C5b']['evals']} x problem.evaluate(forest), sync-bracketed; timings in ms; first_call_ms = "
        "cold call 1 (JIT/autotune over a fresh population), median_warm_ms = median of calls 2..N; "
        "RNG-dependent structure (seed=0); legacy custom-CUDA engine"
    )
    return problem, forest, notes


# --- run drivers ---------------------------------------------------------------------------
def _new_record(tag, A, args, extra):
    base = {
        "tag": tag,
        "side": A.side,
        "evogp_file": A.evogp_file,
        "evogp_commit": args.commit,
    }
    base.update(extra)
    return base


def run_pipeline_config(tag, cfg_id, meta, args, rep):
    """C1/C2/C2b/C4: seeded construction + timed manual generation loop (SPEC timed-run schema)."""
    A = api()
    torch = A.torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    wall_start = time.perf_counter()
    if meta["workload"] == "W1":
        problem, algorithm, notes = build_w1(meta)
    elif meta["workload"] == "W2":
        problem, algorithm, notes = build_w2(meta)
    else:
        raise ValueError(f"{cfg_id}: workload {meta['workload']} is not a pipeline config")
    pipeline = A.StandardPipeline(
        algorithm,
        problem,
        generation_limit=meta["gens"],
        is_show_details=False,
    )

    per_gen_ms = []
    per_gen_best_fitness = []
    for generation in range(meta["gens"]):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        pipeline.step()
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - tic) * 1000.0
        best = float(pipeline.best_fitness)
        per_gen_ms.append(gen_ms)
        per_gen_best_fitness.append(best)
        print(f"[{tag}] gen {generation}: {gen_ms:.3f} ms best={best:.4f}", flush=True)

    wall_s = time.perf_counter() - wall_start
    best_fitness_final = float(pipeline.best_fitness)
    cold_first_gen_ms = per_gen_ms[0] if per_gen_ms else None
    if meta["reps"] > 1:
        notes += f"; rep {rep}/{meta['reps']} (fresh seeded construction)"
    print(
        f"[{tag}] DONE gens={meta['gens']} wall={wall_s:.3f}s best={best_fitness_final:.6f} "
        f"cold_first_gen_ms={cold_first_gen_ms:.3f}",
        flush=True,
    )
    return _new_record(
        tag,
        A,
        args,
        {
            "problem": meta["problem"],
            "config": {
                "pop": meta["pop"],
                "gens": meta["gens"],
                "tree_len": meta["tree_len"],
                "seed": SEED,
            },
            "env": env_snapshot(),
            "gpu": gpu_label(torch),
            "wall_s": wall_s,
            "per_gen_ms": per_gen_ms,
            "per_gen_best_fitness": per_gen_best_fitness,
            "best_fitness_final": best_fitness_final,
            "cold_first_gen_ms": cold_first_gen_ms,
            "notes": notes,
        },
    )


def run_eval_config(tag, cfg_id, meta, args, fixture_path):
    """C5/C5b: repeated `problem.evaluate(forest)` with sync brackets (SPEC eval-only schema)."""
    torch = api().torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    wall_start = time.perf_counter()
    if cfg_id == "C5":
        problem, forest, notes = build_c5(fixture_path)
    elif cfg_id == "C5b":
        problem, forest, notes = build_c5b(fixture_path)
    else:
        raise ValueError(f"{cfg_id}: no eval-only builder")

    per_call_ms = []
    fitness = None
    for call in range(meta["evals"]):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        fitness = problem.evaluate(forest)
        torch.cuda.synchronize()
        call_ms = (time.perf_counter() - tic) * 1000.0
        per_call_ms.append(call_ms)
        print(f"[{tag}] eval {call}: {call_ms:.3f} ms", flush=True)
    wall_s = time.perf_counter() - wall_start

    # Fitness sanity (observation only): NaN-aware mean + NaN count of the returned tensor.
    fitness_host = fitness.detach().to("cpu", dtype=torch.float64)
    nan_count = int(torch.isnan(fitness_host).sum().item())
    fitness_mean = float(torch.nanmean(fitness_host).item())

    first_call_ms = per_call_ms[0]
    median_warm_ms = statistics.median(per_call_ms[1:]) if len(per_call_ms) > 1 else None
    notes += (
        f"; fitness sanity (last call, shape={tuple(fitness.shape)}): nan_count={nan_count}, "
        f"nan-aware mean={fitness_mean!r} (MAXIMIZED fitness, NaN can appear on the legacy engine); "
        f"wall_s covers construction + all {meta['evals']} calls"
    )
    warm_label = "n/a" if median_warm_ms is None else f"{median_warm_ms:.3f}"
    print(
        f"[{tag}] DONE evals={meta['evals']} wall={wall_s:.3f}s first_call_ms={first_call_ms:.3f} "
        f"median_warm_ms={warm_label} nan_count={nan_count}",
        flush=True,
    )
    return _new_record(
        tag,
        api(),
        args,
        {
            "eval_kind": meta["eval_kind"],
            "fixture": os.path.abspath(fixture_path),
            "env": env_snapshot(),
            "gpu": gpu_label(torch),
            "wall_s": wall_s,
            "per_call_ms": per_call_ms,
            "first_call_ms": first_call_ms,
            "median_warm_ms": median_warm_ms,
            "fitness_mean": fitness_mean,
            "nan_count": nan_count,
            "notes": notes,
        },
    )


# --- CLI -----------------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="run_matrix.py",
        description=(
            "OLD-side (pre-ETL custom-CUDA engine, commit b613503) benchmark harness for "
            "test/benchmark/SPEC.md: C1/C2/C2b (W1 synthetic SR), C4 (W2 XOR parity), "
            "C5/C5b (W3 eval-only microbench)."
        ),
        epilog=(
            "Example: cd <legacy checkout> && CUDA_VISIBLE_DEVICES=3 "
            "PYTHONPATH=src:/tmp/evogp_t1a263/siteutils <venv>/bin/python -u "
            f"{os.path.join(SCRIPT_DIR, 'run_matrix.py')} --configs C1 --out {DEFAULT_OUT}"
        ),
    )
    parser.add_argument(
        "--configs", default=None, metavar="LIST",
        help="comma-separated config ids from C1,C2,C2b,C4,C5,C5b (required unless --list)",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, metavar="DIR",
        help=f"directory for the per-run JSON files (default: {DEFAULT_OUT}, script-relative)",
    )
    parser.add_argument(
        "--fixture", default=DEFAULT_FIXTURE, metavar="PATH",
        help=f"W3 fixture npz (default: {DEFAULT_FIXTURE}, script-relative)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None, metavar="INT",
        help=(
            "set CUDA_VISIBLE_DEVICES=INT before the first torch/CUDA import; when omitted the "
            "caller's environment is respected and only recorded"
        ),
    )
    parser.add_argument(
        "--commit", default="b613503", metavar="STR",
        help="evogp revision recorded in the JSON `evogp_commit` field (default: b613503)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the config table and exit 0 (imports no torch/numpy/evogp)",
    )
    return parser


def resolve_configs(parser, raw):
    if raw is None:
        parser.error(
            "--configs is required (comma list from "
            + ",".join(CONFIGS)
            + "); use --list to print the matrix"
        )
    ids = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        match = token if token in CONFIGS else None
        if match is None:
            match = next((k for k in CONFIGS if k.lower() == token.lower()), None)
        if match is None:
            extra = (
                f" (note: {', '.join(NEW_ONLY_CONFIGS)} are NEW-engine-only and not runnable here)"
                if token.upper() in NEW_ONLY_CONFIGS
                else ""
            )
            parser.error(
                f"unknown config '{token}'; supported: {','.join(CONFIGS)}{extra}"
            )
        if match not in ids:
            ids.append(match)
    if not ids:
        parser.error("--configs is empty")
    return ids


def print_config_table(raw_configs=None):
    ids = list(CONFIGS) if raw_configs is None else [
        cid for cid in resolve_configs(build_parser(), raw_configs)
    ]
    header = ("config", "workload", "pop", "gens", "tree_len", "reps", "evals", "tags")
    rows = [header]
    for cfg_id in ids:
        meta = CONFIGS[cfg_id]
        rows.append(
            (
                cfg_id,
                meta["workload"],
                str(meta["pop"]),
                str(meta["gens"]),
                str(meta["tree_len"]),
                str(meta["reps"]),
                "-" if meta["evals"] is None else str(meta["evals"]),
                ", ".join(tags_for(cfg_id, meta)),
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)).rstrip())
    print()
    print(f"fixture default: {DEFAULT_FIXTURE}")
    print(f"out default:     {DEFAULT_OUT}")
    print(f"NEW-only configs in SPEC.md (not runnable here): {', '.join(NEW_ONLY_CONFIGS)}")
    print("timed pipeline configs: C1, C2, C2b, C4 ; eval-only configs: C5, C5b")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.fixture_path = os.path.abspath(args.fixture)

    # Must happen before the first torch/CUDA import (all imports below are lazy).
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.list:
        print_config_table(args.configs)
        return 0

    config_ids = resolve_configs(parser, args.configs)
    plan = []
    for cfg_id in config_ids:
        meta = CONFIGS[cfg_id]
        for rep, tag in enumerate(tags_for(cfg_id, meta), start=1):
            plan.append((cfg_id, tag, rep))

    A = api()  # lazy heavy imports: numpy, torch, evogp + evogp.evogp_cuda
    print(f"=== run_matrix.py {os.path.realpath(__file__)}", flush=True)
    print(
        f"=== side={A.side} commit={args.commit} evogp_file={A.evogp_file} "
        f"python={sys.version.split()[0]}",
        flush=True,
    )
    print(f"=== gpu={gpu_label(A.torch)}", flush=True)
    print(f"=== env={json.dumps(env_snapshot())}", flush=True)
    print(f"=== out={args.out} fixture={args.fixture_path}", flush=True)
    print(f"=== plan={[tag for _, tag, _ in plan]}", flush=True)

    for cfg_id, tag, rep in plan:
        meta = CONFIGS[cfg_id]
        print(
            f"[{tag}] START {cfg_id} workload={meta['workload']} pop={meta['pop']} "
            f"gens={meta['gens']} tree_len={meta['tree_len']} seed={SEED}",
            flush=True,
        )
        if meta["evals"] is None:
            record = run_pipeline_config(tag, cfg_id, meta, args, rep)
        else:
            record = run_eval_config(tag, cfg_id, meta, args, args.fixture_path)
        write_json(args.out, tag, record)
    print(f"=== ALL DONE ({len(plan)} run(s): {', '.join(tag for _, tag, _ in plan)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
