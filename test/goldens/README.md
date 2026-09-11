# `test/goldens/` — golden numeric outputs of the pre-ETL (custom CUDA kernel) engine

## Purpose

These `.npz` files are **frozen reference numbers** produced by the *legacy* EvoGP engine: the
hand-written CUDA kernels in `src/evogp/cuda/` (`generate.cu`, `mutation.cu`, `forward.cu`,
`torch_wrapper.cu`), bound through `evogp.evogp_cuda` and driven through the public Python API.
Both the kernels and the extension are gone — EvoGP now evaluates trees through the pure-torch/ETL
engine in `src/evogp/tree/etl_engine.py` — so these files cannot be re-captured from source; they are
the behaviour the rewrite had to reproduce.

They exist so an engine can be validated numerically against the behaviour it replaced: feed the
stored inputs (and, for the RNG-free checks, the stored tree tensors) to the new code and compare the
outputs against the stored arrays.

They are *not* a pass/fail test suite by themselves. `test/golden_capture.py` (sibling directory)
captures and compares arrays; the pytest suite's golden-fixture loader also scans this directory, but
expects a different key schema — see "Relation to the pytest suite" below.

## Provenance

| | |
|---|---|
| Source revision | `b613503` (`EMI-Group/evogp`, worktree `.genesis/workers/worker_T1_A263`), i.e. the last commit before the kernel-removal rewrite |
| Engine | `evogp.evogp_cuda.cpython-311-x86_64-linux-gnu.so` built from the four `.cu` files in `src/evogp/cuda/` with `python setup.py build_ext --inplace` |
| Compiler flags | the `setup.py` CUDA list: `-O3 --expt-relaxed-constexpr --ptxas-options=-v -Xptxas=-O3 -lineinfo -lcudart -use_fast_math -maxrregcount=32`, `-gencode=arch=compute_86,code=sm_86` (`TORCH_CUDA_ARCH_LIST=8.6`) |
| Toolchain | nvcc 12.6.20 (CUDA 12.6 headers), host `g++` 12.2, PyTorch 2.6.0+cu124, python 3.11.2 |
| GPU | NVIDIA RTX A6000 (sm_86), 8 visible devices, everything runs on `cuda:0` |
| Captured by | `test/golden_capture.py` (`capture`), verified by the same script (`verify`) |
| Status | frozen. The engine that produced them (kernels + `evogp.evogp_cuda` + `setup.py`'s `CUDAExtension`) was deleted in the ETL rewrite, so these exact numbers can no longer be re-captured and the files must not be overwritten |

### Environment recipe used to build and capture (historical — no longer executable)

Kept for provenance: `src/evogp/cuda/` and the `CUDAExtension` in `setup.py` no longer exist, so the
build step below cannot run against the current tree.

```bash
PYTHON=/mnt/local-ssd/bchuang/evox/.venv/bin/python     # py3.11, torch 2.6.0+cu124
CUDA_ENV=/opt/miniforge3/envs/wenet_tiktoken            # provides a full CUDA 12.6 toolkit layout
export CUDA_HOME=$CUDA_ENV/targets/x86_64-linux       # bin/nvcc, include/, lib/, nvvm/
export PATH=$CUDA_ENV/bin:$CUDA_ENV/targets/x86_64-linux/nvvm/bin:$PATH  # ptxas/nvlink from bin, cicc lives in nvvm/bin
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"       # keep the system host compiler

# python3.11 headers are not installed system-wide; extract them from the Debian package:
mkdir -p $TMPDIR/pydev
curl -sO --output-dir $TMPDIR http://deb.debian.org/debian/pool/main/p/python3.11/libpython3.11-dev_3.11.2-6+deb12u8_amd64.deb
dpkg-deb -x $TMPDIR/libpython3.11-dev_3.11.2-6+deb12u8_amd64.deb $TMPDIR/pydev
export CPATH=$TMPDIR/pydev/usr/include/python3.11:$TMPDIR/pydev/usr/include:$TMPDIR/pydev/usr/include/x86_64-linux-gnu
export CPLUS_INCLUDE_PATH=$CPATH

export TORCH_CUDA_ARCH_LIST=8.6
$PYTHON setup.py build_ext --inplace                    # -> src/evogp/evogp_cuda*.so

# `evogp.problem.__init__` imports sklearn; install it out-of-tree (cp311 wheel):
/opt/miniforge3/bin/python -m pip download --only-binary=:all: --no-deps \
    --python-version 311 --implementation cp --abi cp311 --platform manylinux2014_x86_64 \
    scikit-learn joblib threadpoolctl cloudpickle -d $TMPDIR/wheels
for w in $TMPDIR/wheels/*.whl; do /opt/miniforge3/bin/python -c "import zipfile,sys;zipfile.ZipFile('$w').extractall('$TMPDIR/siteutils')"; done
```

Note: a *system* nvcc 11.8 exists but is rejected by torch cu124 (major-version
mismatch), and `cicc`/`ptxas` must come from the CUDA 12.6 install — the
`CUDA_HOME` above is the conda "targets" layout, which is a complete toolkit.

### Regenerating / verifying (current engine)

`test/golden_capture.py` drives whatever engine is installed in the interpreter it runs under, i.e.
today the ETL engine. It therefore cannot reproduce these files — it can only *capture* the current
engine's numbers or *compare* them against the frozen ones:

```bash
cd <repo root>
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python             # project venv; `python` is not on PATH
PYTHONPATH=src $PY test/golden_capture.py manifest          # inventory: shape/dtype/sha1/samples
PYTHONPATH=src $PY test/golden_capture.py verify --out test/goldens    # diff vs the frozen captures
PYTHONPATH=src $PY test/golden_capture.py capture --out "$TMPDIR/goldens"   # NEW capture, scratch dir
PYTHONPATH=src $PY test/golden_capture.py capture --stage sr_fitness --out "$TMPDIR/goldens"  # single stage
```

The script warns when `--out` resolves to this directory, because `capture` would replace the frozen
reference numbers. On the numpy backend a stage takes seconds; the training stage (200 individuals x
12 generations, one JIT build per distinct tree) takes minutes instead of the legacy ~7 s.

`verify` exits non-zero on any mismatch (integer/string arrays exactly, floats with
`rtol=1e-4, atol=1e-4, equal_nan=True`) and reports per array whether the re-run was *bitwise*
identical. Run against these captures it validates one build against itself only: every RNG-dependent
array must differ (see caveat 2 below), while the descriptor/header/input arrays
(`descriptor__*`, `x`, `x_small`, `labels`, `*_scalars`, `seed`) are expected bitwise equal.

## Determinism (measured)

**Legacy engine** — with `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` all 96 arrays of the
committed set reproduce bitwise across processes (`verify`: `bitwise=96 within-tolerance=0
mismatch=0`), including the 12-generation training trajectory and both `-inf`/NaN patterns.

**Current engine** — re-running the same public API calls is deterministic per build the same way, but
the *values* differ from the legacy ones wherever the generator's RNG stream is involved (population
structure, mutation/crossover outcomes, training trajectory); on a `verify --out test/goldens` run of
`--stage forest_random` the 15 descriptor/header/input arrays came out bitwise identical and the three
`forest__node_*` arrays were reported as MISMATCH.

Two caveats:

1. **Dead region of the tree tensors is normalised.** As captured, the kernels only
   wrote the first `subtree_size[j]` entries of row `j` of the `(pop_size, max_tree_len)`
   tensors; the rest was *uninitialised device memory* and is not reproducible
   between processes (observed: `node_type` bytes differ, `node_value` differs by
   ~1e-39 denormals). `test/golden_capture.py` therefore zero-fills every entry
   with `index >= subtree_size[j]` before saving. The **used prefix is bitwise
   deterministic** (proved by the identical fitness/trajectory arrays). Any engine
   must be compared on the used prefix only — treat the zero tail as
   "don't care" (today's generator zeroes it itself, so this only matters for the
   stored legacy rows).
2. Everything is only reproducible **for the same RNG-consumption order**.
   `verify` compares re-runs of *this script*, which always seeds at the start of
   each stage and calls the API in a fixed order. The rewrite samples populations,
   mutation masks and crossover indices from **torch's RNG** instead of the legacy
   kernels' generator (documented distribution-equivalent, not bit-identical), so it
   does not reproduce the *generated* forests or the *trajectory* bitwise and is not
   expected to — see "How to use these goldens".

## Contents

All arrays are numpy, little-endian; `float32` outputs, `int16` tree types/sizes,
`float64` for descriptor scalar headers, `<U…` for the informational strings.

### `forest_random.npz` — forest generation (`tree_generate`), 18 arrays

`Forest.random_generate(pop_size=64, descriptor=GenerateDescriptor(max_tree_len=63, input_len=2, output_len=1, using_funcs=["+","-","*","/"], max_layer_cnt=4, layer_leaf_prob=0.2, const_prob=0.5, out_prob=0.5, const_samples=[-2,-1,-0.5,-0.25,0.25,0.5,1,2]))`

| array | shape | dtype | meaning |
|---|---|---|---|
| `forest__node_value` | (64, 63) | float32 | per-node value: constant value, variable index, or function id |
| `forest__node_type` | (64, 63) | int16 | `NType` (0=VAR, 1=CONST, 2=UFUNC, 3=BFUNC, 4=TFUNC; `+128` = output flag) |
| `forest__subtree_size` | (64, 63) | int16 | subtree size of each node (`[j,0]` = whole-tree size) |
| `x` | (64, 2) | float32 | evaluation inputs (`x[i] = ((i % 8 - 3.5)/2, (i // 8 - 3.5)/2)`) |
| `labels` | (64, 1) | float32 | `x[:,0]**2 - x[:,1]` |
| `x_small` | (8, 2) | float32 | first 8 rows of `x`, batch input for `batch_forward` |
| `descriptor__const_samples` | (8,) | float32 | the constant pool handed to the kernel |
| `descriptor__depth2leaf_probs` | (10,) | float32 | `MAX_FULL_DEPTH=10` leaf probabilities |
| `descriptor__roulette_funcs` | (29,) | float32 | cumulative function CDF (`Func.END = 29`) |
| `descriptor__roulette_tfuncs/_bfuncs/_ufuncs` | (29,) | float32 | per-arity CDFs used by the generator |
| `descriptor__scalars` | (5,) | float64 | `[max_tree_len, input_len, output_len, const_prob, out_prob]` = `[63, 2, 1, 0.5, 0.5]` |
| `using_funcs`, `func_ids` | (29,) | `<U16`, int32 | function-name/index table (`FUNCS_NAMES` / `Func.*`) |
| `seed`, `pop_size`, `num_first_trees` | () | int64 | `0`, `64`, `8` |

Sample: `forest__node_type[0] = [0, 0, 0, …]`, `forest__subtree_size[0,0] = 1`,
`x[0] = [-1.75, -1.75]`, `labels[0] = 4.8125`.

### `forward_eval.npz` — forward evaluation (`tree_evaluate`), 13 arrays

Inputs are the same forest / `x` / `labels` (bitwise identical arrays, also
stored here for self-containment). `Tree.forward`, `Forest.forward` and
`Forest.batch_forward` have no `execute_mode` argument — they always call
`tree_evaluate`; the execute modes select the *fitness reduction* kernel only and
are therefore captured in the next section (the `"torch"` mode is the
problem-level `batch_forward` path).

| array | shape | dtype | how it was produced |
|---|---|---|---|
| `tree_forward` | (8, 64, 1) | float32 | `forest[i].forward(x)` for `i = 0…7` |
| `forest_forward` | (64, 1) | float32 | `forest.forward(x)` |
| `forest_batch_forward` | (64, 8, 1) | float32 | `forest.batch_forward(x_small)` |
| `tree_infix` | (8,) | `<U68` | `Tree.to_infix()` (display form, informational) |
| `tree_sympy_str` | (8,) | `<U33` | `str(Tree.to_sympy_expr())` (informational) |
| `tree_preorder_str` | (8,) | `<U59` | `evogp.tree.utils.str_tree(...)` preorder dump (informational) |

Sample expressions (indices 0,1,7): `x0`, `((0.25 − -2.00) − (x1 − x1)) − ((-0.25 + 0.50) * (x1 + x0))`,
`((-2.00 − 1.00) − (-0.25 * x1)) − ((x0 + 2.00) * (x0 / x1))`.
`tree_forward[0, :, 0] == x[:, 0]` (tree 0 is the variable `x0`).
`tree_forward[6]` is `NaN` for the 8 rows with `x1 == 0.25`: tree 6 is
`(((-2.00 / -1.00) + (-1.00 − x1)) / ((x1 * 2.00) − (-1.00 / -2.00)))`, whose
denominator `2*x1 - 0.5` vanishes there.

### `sr_fitness.npz` — regression fitness (`tree_SR_fitness`), 31 arrays

For every kernel execute mode `M ∈ {"hybrid parallel", "data parallel",
"tree parallel", "auto"}` (keys use `_` instead of spaces) and both metrics:

| array | shape | dtype | meaning |
|---|---|---|---|
| `forest_sr_fitness__M__mse` | (64,) | float32 | `forest.SR_fitness(x, labels, use_MSE=True, execute_mode=M)` |
| `forest_sr_fitness__M__nonmse` | (64,) | float32 | same with `use_MSE=False` |
| `tree_sr_fitness__M__mse` / `__nonmse` | (8,) | float32 | `forest[i].SR_fitness(...)`, `i = 0…7`, concatenated |
| `problem_evaluate__M__mse` / `__nonmse` | (64,) | float32 | `SymbolicRegression(datapoints=x, labels=labels, execute_mode=M).evaluate(forest, use_MSE=…)` |
| `problem_evaluate__torch__mse` | (64,) | float32 | the pure-torch path (`batch_forward` + `-mean(sq err)`) |

Semantics recovered and confirmed numerically:

* `SR_fitness(use_MSE=True)` **returns the plain MSE** (positive, *not* negated),
  `use_MSE=False` returns the plain **MAE** (`mean|x−y|`); verified against numpy
  on `tree_forward` (`[5.6601562, 3.0117188, 14.257586, …]` and
  `[1.8125, 1.4609375, 2.5941963, …]`).
* `SymbolicRegression.evaluate(...)` for kernel modes is exactly
  `-forest.SR_fitness(...)` (bitwise equal negative of the arrays above), and for
  `execute_mode="torch"` it is `-mean((forest.batch_forward(datapoints) - labels)²)`.
* 3 of the 64 trees evaluate to `NaN` (unstable `/`); the goldens keep those NaNs.
* `"hybrid parallel"`, `"data parallel"` and `"auto"` agree bitwise; `"tree
  parallel"` differs by ≤ `2.29e-05` (1 ulp at magnitude ≈14, rel ≈1.6e-6) because
  it reduces in a different order.
* The pure-torch path diverges from the kernel MSE by ≤ `7.6e-06` abs
  (rel ≈1.4e-07) — float32 summation order, not a semantic difference.

### `training.npz` — 12-generation evolutionary reference, 34 arrays

`SymbolicRegression(func=x0⁴/(x0⁴+1)+x1⁴/(x1⁴+1), num_inputs=2, num_data=1000,
lower_bounds=-5, upper_bounds=5)` + `GeneticProgramming(Forest.random_generate(200,
descriptor), DefaultCrossover(), DefaultMutation(mutation_rate=0.2, descriptor.update(max_layer_cnt=3)), DefaultSelection(survival_rate=0.3, elite_rate=0.01))`
+ `StandardPipeline(..., generation_limit=12, is_show_details=False)`; the script
calls `StandardPipeline.step()` 12 times (same code path as `run()` with
`generation_limit=12`) so the trajectory can be recorded.

| array | shape | dtype | meaning |
|---|---|---|---|
| `datapoints` | (1000, 2) | float32 | generated `torch.rand` inputs (this is the RNG-sensitive part of the problem) |
| `labels` | (1000, 1) | float32 | `func(datapoints)` via `torch.vmap` |
| `generation_fitness` | (12, 200) | float32 | population fitness **before** each `GeneticProgramming.step()`, i.e. the value `problem.evaluate(forest)` returns (NaN already replaced by `-inf` in `StandardPipeline.step`) |
| `initial_fitness` | (200,) | float32 | copy of `generation_fitness[0]` |
| `initial_forest__node_value/_node_type/_subtree_size` | (200, 128) | float32/int16/int16 | generation-0 population |
| `final_forest__node_value/_node_type/_subtree_size` | (200, 128) | float32/int16/int16 | population after the 12th step |
| `best_tree__node_value/_node_type/_subtree_size` | (1, 128) | float32/int16/int16 | `StandardPipeline.best_tree` (best individual ever seen) |
| `best_fitness_per_generation` | (12,) | float32 | `pipeline.best_fitness` after each step (monotone) |
| `best_tree_infix`, `best_tree_sympy_str` | () | `<U4`, `<U9` | `1.67` / `1.6656103` (the best tree is a single constant node) |
| `train_descriptor__*`, `mut_descriptor__*` | see above | | descriptor reproductions: `const_samples` (8,), `depth2leaf_probs` (10,), `roulette_*` (29,), scalars (5,) |
| `seed`, `pop_size`, `generation_limit`, `num_data` | () | int64 | `0`, `200`, `12`, `1000` |

Reference values:

* `best_fitness_per_generation = [-0.251097, -0.23594594, -0.23594594, …]` — the
  run improves once (generation 1) and then stalls; `final best fitness =
  -0.23594594`, best tree = the constant `1.6656103` (a value from the mutation
  descriptor's `const_samples`).
* The population collapses to small trees (`subtree_size[:,0]` mean 1.7, 174/200
  single-node) — that is the genuine behaviour of the old engine at this budget,
  not a capture artefact.
* Fitness magnitudes are huge because the `DIV`-style operators clamp to
  `MAXVAL = 1e9`; 9 individuals of generation 0 evaluate to `-inf`.

## Recommended comparison tolerances

| data | tolerance |
|---|---|
| `node_type`, `subtree_size`, `func_ids`, all `*_scalars`/headers, string arrays | **exact** (integers / structural) |
| `node_value`, `*_forward*`, `*_fitness*`, `generation_fitness` | `rtol = 1e-4`, `atol = 1e-4`, `equal_nan = True` |

Why not bit-equality for the floats: the kernels were built with
`-use_fast_math` (nvcc is free to fold `1/x`, `sqrt`, … into approximate
instructions), the four execute modes reduce in different orders (max observed
spread `2.3e-05`), and any New engine will accumulate in its own order. The
observed torch-vs-kernel gap (`7.6e-06` abs) shows that `1e-4` is comfortable
while still catching real semantic errors. `rtol` alone is unsafe here: the
arrays contain exact zeros and `±inf`, so use `atol` and treat `inf`/`nan`
positions as equal only when both sides agree.

## How to use these goldens with the current (ETL) engine

0. The files themselves are ground truth as captured; an engine only needs the `.npz`
   inputs + expected outputs. `golden_capture.py verify --out test/goldens` shows
   where today's engine stands.
1. **Structure/generation** — feed `descriptor__*` + `seed` to the generator and
   compare `forest__node_*`. The current generator samples from torch's RNG, so its
   output does not match and is not expected to; use
   `*_node_type`/`*_subtree_size` as a *shape/legality* reference only (trees must
   decode, sizes must be consistent, the constants must come from
   `descriptor__const_samples`).
2. **Evaluation (RNG-independent, strongest check)** — load
   `forest__node_value/_node_type/_subtree_size` from any stage, rebuild a `Forest`
   from them and check `tree_forward`, `forest_forward`, `forest_batch_forward` and
   every `*_sr_fitness__*` array on the stored `x`/`x_small`/`labels`. This validates
   the evaluation semantics on fixed input with no RNG involved; the current engine
   reproduces them as correctly-rounded float32 rather than the legacy `-use_fast_math`
   kernels, hence the `rtol=atol=1e-4` recommendation.
3. **Problem level** — check `problem_evaluate__*` (including the `torch` path,
   which must match the engine path to ≈1e-7 rel).
4. **Trajectory** — reproduce `initial_fitness` **from the stored
   `initial_forest*` + `datapoints`/`labels`** (bitwise on the same GPU; within
   tolerance otherwise; mind the `-inf` NaN replacement). Reaching the same
   *generation-by-generation* `generation_fitness` would additionally require an
   identical RNG-consumption order, which the rewrite deliberately does not have —
   treat `best_fitness_per_generation` as a *behavioural* reference: generation 1
   must be ≥ `-0.23594594`, and the final best tree should be no worse.

## Relation to the pytest suite

`src/evogp/tests/test_equivalence.py::test_golden_fixtures` scans `$EVOGP_GOLDEN_DIR`, then
`<repo-root>/test/goldens/` (this directory), then `src/evogp/tests/goldens/` for `*.npz` fixture files.
A fixture must carry the flat keys `node_value` (pop, max_tree_len) float32, `node_type` int16,
`subtree_size` int16 and `x` (N, input_len) float32, plus either `expected` (pop, N, out_len) float32 or
`expected_fitness` (pop,) float32 (optionally `labels` and `use_mse`).

The 7 files in `src/evogp/tests/goldens/` use that schema; the raw captures here do not (they are keyed
`forest__node_value`, `initial_forest__node_value`, …). The loader currently `pytest.fail`s on the first
file that misses the required keys, so `test_golden_fixtures` fails on `forest_random.npz` — the one
failing test of the suite. Either side can fix it: relax the loader to skip non-fixture `npz` files
(the data here is legitimate but differently shaped), or derive flat-key fixtures from these captures
(`forest__node_*` → `node_value`/`node_type`/`subtree_size`, `x`/`labels` as-is, `expected_fitness` =
`forest_sr_fitness__hybrid_parallel__mse` for `sr_fitness.npz`, `expected` =
`forest_batch_forward` with `x = x_small` for `forward_eval.npz`). The conversion is deliberately not
applied here: these files are frozen reference data and `golden_capture.py verify` compares key sets
exactly.

## Limitations / known gaps

* Single output (`output_len=1`), single input pair (`input_len=2`), only the
  `+ - * /` operators and a fixed 8-value constant pool. The `if`, comparison,
  unary and `loose_*` operators, multi-output trees and `CombinedForest` are
  **not** covered.
* Mutations/crossover are exercised only through the training stage (the legacy
  `tree_mutate`/`tree_crossover` ops are never compared to a reference on
  fixed tensors), and no golden exists for `to_png`/`to_sympy_expr` beyond the
  informational strings.
* `Tree.SR_fitness`'s `"hybrid parallel"` used to map to `execute_code = 3` while
  `Forest.SR_fitness`'s mapped to `0`; both are captured, but the numbers alone do
  not pin down the legacy selector semantics (today all five mode strings route to
  the same engine path).
* The `-inf` entries in `generation_fitness` come from
  `StandardPipeline.step()`'s NaN→`-inf` replacement, so a new pipeline must keep
  that convention to compare training arrays directly.
