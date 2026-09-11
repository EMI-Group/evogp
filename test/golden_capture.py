#!/usr/bin/env python
"""Golden-value capture for the pre-ETL EvoGP engine (custom CUDA kernels).

This script drives the CURRENT implementation (hand-written CUDA kernels under
`src/evogp/cuda/`, bound through `evogp.evogp_cuda`) on a small, fully
deterministic workload and stores the resulting numbers under `test/goldens/`.
Those `.npz` files are the reference a replacement engine (e.g. an ETL/torch
implementation without custom kernels) has to reproduce.  See
`test/goldens/README.md` for provenance, the regeneration recipe and the
recommended comparison tolerances.

Usage (from the repository root; `src/` is added to `sys.path` automatically):

    python test/golden_capture.py capture    # (re)build test/goldens/*.npz
    python test/golden_capture.py verify     # re-run + compare against the files
    python test/golden_capture.py manifest   # print the inventory of each file

`verify` exits non-zero when a golden differs (exact for integer/string arrays,
`rtol=1e-4 / atol=1e-4` for floats) and reports, per array, whether the re-run
was bitwise identical - that is the determinism evidence for this engine.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from evogp.algorithm import (  # noqa: E402
    DefaultCrossover,
    DefaultMutation,
    DefaultSelection,
    GeneticProgramming,
)
from evogp.pipeline import StandardPipeline  # noqa: E402
from evogp.problem import SymbolicRegression  # noqa: E402
from evogp.tree import Forest, GenerateDescriptor  # noqa: E402
from evogp.tree.utils import FUNCS_NAMES, str_tree  # noqa: E402

GOLDENS_DIR = os.path.join(REPO_ROOT, "test", "goldens")

SEED = 0

# ---------------------------------------------------------------------------
# Fixed, small workload
# ---------------------------------------------------------------------------

#: `Forest.SR_fitness` / `Tree.SR_fitness` / `SymbolicRegression` accept these
#: four kernel execute modes ("torch" is the pure-`batch_forward` Python path
#: exposed by `SymbolicRegression` and has no kernel counterpart).
KERNEL_MODES = ("hybrid parallel", "data parallel", "tree parallel", "auto")

#: pop_size / max_tree_len of the evaluation forest (task spec: 64 x 63).
POP_SIZE = 64
MAX_TREE_LEN = 63
NUM_FIRST_TREES = 8
SMALL_BATCH = 8

#: training reference (task spec: pop_size=200, ~12 generations, seed 0)
TRAIN_POP_SIZE = 200
TRAIN_GENERATIONS = 12
TRAIN_NUM_DATA = 1000


def seed_all(seed=SEED):
    """Seed the CPU and all CUDA generators (torch.manual_seed also does CUDA)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def capture_descriptor():
    """Descriptor of the evaluation forest - no RNG involved (fixed consts)."""
    return GenerateDescriptor(
        max_tree_len=MAX_TREE_LEN,
        input_len=2,
        output_len=1,
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=4,
        layer_leaf_prob=0.2,
        const_prob=0.5,
        out_prob=0.5,
        const_samples=[-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0],
    )


def fixed_inputs(num_data=POP_SIZE):
    """Deterministic (num_data, 2) inputs + (num_data, 1) labels (no RNG)."""
    i = torch.arange(num_data, device="cuda")
    x = torch.stack(
        [
            ((i % 8).float() - 3.5) / 2.0,
            (torch.div(i, 8, rounding_mode="floor").float() - 3.5) / 2.0,
        ],
        dim=1,
    ).contiguous()
    labels = (x[:, 0:1] ** 2 - x[:, 1:2]).contiguous()
    return x, labels


def descriptor_arrays(descriptor, prefix="descriptor"):
    return {
        f"{prefix}__const_samples": to_np(descriptor.const_samples),
        f"{prefix}__depth2leaf_probs": to_np(descriptor.depth2leaf_probs),
        f"{prefix}__roulette_funcs": to_np(descriptor.roulette_funcs),
        f"{prefix}__roulette_tfuncs": to_np(descriptor.roulette_tfuncs),
        f"{prefix}__roulette_bfuncs": to_np(descriptor.roulette_bfuncs),
        f"{prefix}__roulette_ufuncs": to_np(descriptor.roulette_ufuncs),
        f"{prefix}__scalars": np.array(
            [
                descriptor.max_tree_len,
                descriptor.input_len,
                descriptor.output_len,
                descriptor.const_prob,
                descriptor.out_prob,
            ],
            dtype=np.float64,
        ),
    }


def normalize_tree_partition(node_value, node_type, subtree_size):
    """Zero-fill the dead region of every tree row (indices >= subtree_size[j]).

    The kernels only write the first `subtree_size[j]` entries of row `j`; the
    rest of each (pop_size, max_tree_len) row is uninitialised device memory and
    is NOT reproducible between processes.  The used prefix is, so the goldens
    store the prefix as-is and the dead tail as zeros (see test/goldens/README.md).
    """
    sizes = subtree_size[:, 0]
    used = np.arange(node_value.shape[1])[None, :] < sizes[:, None]
    return (
        np.where(used, node_value, np.float32(0.0)).astype(node_value.dtype),
        np.where(used, node_type, np.int16(0)).astype(node_type.dtype),
        np.where(used, subtree_size, np.int16(0)).astype(subtree_size.dtype),
    )


def forest_arrays(forest, prefix):
    node_value, node_type, subtree_size = normalize_tree_partition(
        to_np(forest.batch_node_value),
        to_np(forest.batch_node_type),
        to_np(forest.batch_subtree_size),
    )
    return {
        f"{prefix}__node_value": node_value,
        f"{prefix}__node_type": node_type,
        f"{prefix}__subtree_size": subtree_size,
    }


def fixed_case():
    """Seed, then rebuild the exact evaluation forest + inputs used by stages 1-3."""
    seed_all(SEED)
    descriptor = capture_descriptor()
    forest = Forest.random_generate(pop_size=POP_SIZE, descriptor=descriptor)
    x, labels = fixed_inputs()
    x_small = x[:SMALL_BATCH].contiguous()
    return descriptor, forest, x, labels, x_small


# ---------------------------------------------------------------------------
# Stage 1: random forest generation (tree_generate)
# ---------------------------------------------------------------------------


def capture_forest_random():
    descriptor, forest, x, labels, x_small = fixed_case()
    arrays = {
        "seed": np.array(SEED),
        "pop_size": np.array(POP_SIZE),
        "num_first_trees": np.array(NUM_FIRST_TREES),
        "x": to_np(x),
        "labels": to_np(labels),
        "x_small": to_np(x_small),
        "using_funcs": np.array(
            ["if", "+", "-", "*", "/", "loose_div", "pow", "loose_pow", "max", "min",
             "<", ">", "<=", ">=", "sin", "cos", "tan", "sinh", "cosh", "tanh",
             "log", "loose_log", "exp", "inv", "loose_inv", "neg", "abs", "sqrt",
             "loose_sqrt"],
            dtype="U16",
        ),
        "func_ids": np.arange(len(FUNCS_NAMES), dtype=np.int32),
    }
    arrays.update(descriptor_arrays(descriptor))
    arrays.update(forest_arrays(forest, "forest"))
    return arrays


# ---------------------------------------------------------------------------
# Stage 2: forward evaluation (tree_evaluate)
# ---------------------------------------------------------------------------


def capture_forward_eval():
    # `Tree.forward` / `Forest.forward` / `Forest.batch_forward` take no
    # execute_mode - they always call `tree_evaluate`. The execute modes only
    # pick the reduction kernel of `*_SR_fitness` (captured in stage 3); the
    # "torch" mode is the problem-level `batch_forward` path (also stage 3).
    _descriptor, forest, x, labels, x_small = fixed_case()

    tree_forward = np.stack(
        [to_np(forest[i].forward(x)) for i in range(NUM_FIRST_TREES)], axis=0
    )
    forest_forward = to_np(forest.forward(x))
    forest_batch_forward = to_np(forest.batch_forward(x_small))

    infix, plain, preorder = [], [], []
    for i in range(NUM_FIRST_TREES):
        tree = forest[i]
        infix.append(tree.to_infix())
        plain.append(str(tree.to_sympy_expr()))
        preorder.append(
            str_tree(
                tree.node_value.cpu().numpy(),
                tree.node_type.cpu().numpy(),
                tree.subtree_size.cpu().numpy(),
            )
        )

    arrays = {
        "seed": np.array(SEED),
        "x": to_np(x),
        "x_small": to_np(x_small),
        "labels": to_np(labels),
        "tree_forward": tree_forward,
        "forest_forward": forest_forward,
        "forest_batch_forward": forest_batch_forward,
        # string width is inferred on purpose (no silent truncation)
        "tree_infix": np.array(infix),
        "tree_sympy_str": np.array(plain),
        "tree_preorder_str": np.array(preorder),
    }
    arrays.update(forest_arrays(forest, "forest"))
    return arrays


# ---------------------------------------------------------------------------
# Stage 3: SR fitness (tree_SR_fitness) in every execute mode
# ---------------------------------------------------------------------------


def capture_sr_fitness():
    _descriptor, forest, x, labels, _x_small = fixed_case()

    arrays = {
        "seed": np.array(SEED),
        "x": to_np(x),
        "labels": to_np(labels),
    }
    for mode in KERNEL_MODES:
        key = mode.replace(" ", "_")
        for use_mse in (True, False):
            suffix = "mse" if use_mse else "nonmse"
            arrays[f"forest_sr_fitness__{key}__{suffix}"] = to_np(
                forest.SR_fitness(x, labels, use_MSE=use_mse, execute_mode=mode)
            )
            arrays[f"tree_sr_fitness__{key}__{suffix}"] = np.concatenate(
                [
                    to_np(
                        forest[i].SR_fitness(
                            x, labels, use_MSE=use_mse, execute_mode=mode
                        )
                    )
                    for i in range(NUM_FIRST_TREES)
                ]
            )
        arrays[f"problem_evaluate__{key}__mse"] = to_np(
            SymbolicRegression(
                datapoints=x.clone(), labels=labels.clone(), execute_mode=mode
            ).evaluate(forest)
        )
        arrays[f"problem_evaluate__{key}__nonmse"] = to_np(
            SymbolicRegression(
                datapoints=x.clone(), labels=labels.clone(), execute_mode=mode
            ).evaluate(forest, use_MSE=False)
        )
    # the "torch" path is batch_forward + -mean(squared error); no kernel involved
    arrays["problem_evaluate__torch__mse"] = to_np(
        SymbolicRegression(
            datapoints=x.clone(), labels=labels.clone(), execute_mode="torch"
        ).evaluate(forest)
    )
    arrays.update(forest_arrays(forest, "forest"))
    return arrays


# ---------------------------------------------------------------------------
# Stage 4: short training reference (SymbolicRegression + GP + StandardPipeline)
# ---------------------------------------------------------------------------


def sr_training_func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


def capture_training():
    seed_all(SEED)

    problem = SymbolicRegression(
        func=sr_training_func,
        num_inputs=2,
        num_data=TRAIN_NUM_DATA,
        lower_bounds=-5,
        upper_bounds=5,
    )
    descriptor = GenerateDescriptor(
        max_tree_len=128,
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
    mutation_descriptor = descriptor.update(max_layer_cnt=3)
    initial_forest = Forest.random_generate(
        pop_size=TRAIN_POP_SIZE, descriptor=descriptor
    )
    algorithm = GeneticProgramming(
        initial_forest=initial_forest,
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=mutation_descriptor),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    pipeline = StandardPipeline(
        algorithm,
        problem,
        generation_limit=TRAIN_GENERATIONS,
        is_show_details=False,
    )

    generation_fitness = []
    best_fitness = []
    for _ in range(TRAIN_GENERATIONS):
        gen_fitness = pipeline.step()
        generation_fitness.append(gen_fitness.numpy().copy())
        best_fitness.append(float(pipeline.best_fitness))

    best_tree = pipeline.best_tree
    final_forest = algorithm.forest

    arrays = {
        "seed": np.array(SEED),
        "pop_size": np.array(TRAIN_POP_SIZE),
        "generation_limit": np.array(TRAIN_GENERATIONS),
        "num_data": np.array(TRAIN_NUM_DATA),
        "datapoints": to_np(problem.datapoints),
        "labels": to_np(problem.labels),
        "generation_fitness": np.stack(generation_fitness, axis=0),
        "initial_fitness": generation_fitness[0].copy(),
        "best_fitness_per_generation": np.array(best_fitness, dtype=np.float32),
        "best_tree_infix": np.array(best_tree.to_infix()),
        "best_tree_sympy_str": np.array(str(best_tree.to_sympy_expr())),
    }
    arrays.update(descriptor_arrays(descriptor, prefix="train_descriptor"))
    arrays.update(descriptor_arrays(mutation_descriptor, prefix="mut_descriptor"))
    arrays.update(forest_arrays(initial_forest, "initial_forest"))
    arrays.update(forest_arrays(final_forest, "final_forest"))
    arrays.update(forest_arrays(best_tree.to_forest(), "best_tree"))
    return arrays


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGES = {
    "forest_random": capture_forest_random,
    "forward_eval": capture_forward_eval,
    "sr_fitness": capture_sr_fitness,
    "training": capture_training,
}

EXACT_DTYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32,
                np.uint64, np.bool_)

RTOL = 1e-4
ATOL = 1e-4


def run_capture(stages, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in stages:
        arrays = STAGES[name]()
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(path, **arrays)
        written.append((path, arrays))
        print(f"[capture] {name}: {len(arrays)} arrays -> {path}")
    return written


def array_sha1(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


def describe_arrays(arrays, sample=3):
    for key in sorted(arrays):
        arr = np.asarray(arrays[key])
        flat = arr.reshape(-1)
        head = ", ".join(
            repr(v) for v in flat[:sample]
        )
        print(
            f"    {key}: shape={arr.shape} dtype={arr.dtype} "
            f"sha1={array_sha1(arr)} samples=[{head}]"
        )


def compare_arrays(gold, new):
    """Return (status, detail) for one array pair.

    "exact" tolerates NaN in the same positions (NaN-aware bitwise equality),
    "close" is the rtol/atol fallback, anything else is a mismatch.
    """
    if gold.shape != new.shape:
        return "SHAPE", f"golden {gold.shape} vs rerun {new.shape}"
    nan_note = ""
    if gold.dtype.kind == "f":
        n_nan = int(np.isnan(gold).sum())
        if n_nan:
            nan_note = f", nan_cnt={n_nan}"
    if gold.dtype.kind in "iub" or new.dtype.kind in "iub":
        if np.array_equal(gold, new):
            return "exact", nan_note
        diff = np.abs(gold.astype(np.int64) - new.astype(np.int64))
        return "MISMATCH", f"integer diff max={diff.max()} n_diff={(diff != 0).sum()}"
    if gold.dtype.kind == "U" or new.dtype.kind == "U":
        if np.array_equal(gold, new):
            return "exact", nan_note
        return "MISMATCH", "string arrays differ"
    if np.array_equal(gold, new, equal_nan=True):
        return "exact", nan_note
    if np.allclose(gold, new, rtol=RTOL, atol=ATOL, equal_nan=True):
        with np.errstate(invalid="ignore"):
            max_abs = np.nanmax(np.abs(gold.astype(np.float64) - new.astype(np.float64)))
        return "close", f"max_abs_diff={max_abs:.3e}{nan_note}"
    with np.errstate(invalid="ignore"):
        max_abs = np.nanmax(np.abs(gold.astype(np.float64) - new.astype(np.float64)))
        denom = np.maximum(np.abs(gold.astype(np.float64)), ATOL)
        max_rel = np.nanmax(np.abs(gold - new) / denom)
    return "MISMATCH", f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}{nan_note}"


def run_verify(stages, goldens_dir):
    ok = True
    counts = {"exact": 0, "close": 0, "MISMATCH": 0, "SHAPE": 0}
    for name in stages:
        path = os.path.join(goldens_dir, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[verify] {name}: MISSING {path}")
            ok = False
            continue
        gold = np.load(path, allow_pickle=False)
        new = STAGES[name]()
        gold_keys, new_keys = set(gold.files), set(new)
        if gold_keys != new_keys:
            print(
                f"[verify] {name}: KEY MISMATCH "
                f"only_golden={sorted(gold_keys - new_keys)} "
                f"only_rerun={sorted(new_keys - gold_keys)}"
            )
            ok = False
        for key in sorted(gold_keys & new_keys):
            status, detail = compare_arrays(gold[key], np.asarray(new[key]))
            counts[status] += 1
            if status in ("MISMATCH", "SHAPE"):
                ok = False
            flag = {"exact": "bitwise", "close": "within-tol"}.get(status, status)
            extra = f" ({detail.lstrip(', ')})" if detail else ""
            print(f"[verify] {name}.{key}: {flag}{extra}")
        print(f"[verify] {name}: {counts['exact'] + counts['close']} arrays checked so far")
    print(
        f"[verify] summary: bitwise={counts['exact']} within-tolerance={counts['close']} "
        f"mismatch={counts['MISMATCH']} shape-mismatch={counts['SHAPE']}"
    )
    return ok


def run_manifest(goldens_dir):
    for name in STAGES:
        path = os.path.join(goldens_dir, f"{name}.npz")
        if not os.path.exists(path):
            print(f"{name}.npz: MISSING")
            continue
        data = np.load(path, allow_pickle=False)
        print(f"{name}.npz ({os.path.getsize(path)} bytes, {len(data.files)} arrays):")
        describe_arrays({k: data[k] for k in data.files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("capture", "verify", "manifest"), help="what to do"
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGES),
        default=None,
        help="restrict to one stage (repeatable); default: all",
    )
    parser.add_argument("--out", default=GOLDENS_DIR, help="goldens directory")
    args = parser.parse_args()

    stages = args.stage if args.stage else list(STAGES)
    if args.action == "capture":
        written = run_capture(stages, args.out)
        for path, arrays in written:
            print(f"[capture] {os.path.basename(path)}:")
            describe_arrays(arrays)
    elif args.action == "verify":
        sys.exit(0 if run_verify(stages, args.out) else 1)
    else:
        run_manifest(args.out)


if __name__ == "__main__":
    main()
