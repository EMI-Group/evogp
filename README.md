<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./imgs/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="./imgs/evox_brand_dark.svg">
    <img alt="EvoX Logo" height="128" width="500px" src="./imgs/evox_brand_dark.svg">
  </picture>
  </a>
  <br>
</h1>

<p align="center">
  <picture>
    <img src="imgs/SymbolicRegression.svg" alt="" height="180">
  </picture>
  <picture>
    <img src="imgs/Classification.svg" alt="" height="180">
  </picture>
  <picture>
    <img src="imgs/Halfcheetah.svg" alt="HalfCheetah" height="180">
  </picture>
</p>


<p align="center">
🌟 EvoGP: Tensor-native Tree-based Genetic Programming, JIT-compiled through ETL 🌟
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.17168">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="EvoGP Paper on arXiv">
  </a>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Backends & Performance](#evaluation-backends--performance)
- [Advanced Genetic Operations](#advanced-genetic-operations)
- [Supported Benchmarks](#supported-benchmarks)
   - [Symbolic Regression](#symbolic-regression)
   - [Classification](#classification)
   - [Transformation](#transformation)
   - [Robotics Control](#robotics-control)
- [Validation](#validation)
- [Experiments in Paper](#experiments-in-paper)
- [Citation](#citation)
- [Community & Support](#community--support)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)


## Introduction

EvoGP is a **tensor-native** Tree-based Genetic Programming (TGP) framework built on PyTorch and driven by
**ETL** (the EvoX Tensor Library). An individual is just a few ordinary tensors — a population is three parallel
`[pop_size, max_tree_len]` matrices holding prefix-encoded trees — so random tree generation, crossover and mutation
are plain vectorized torch operations, and a whole generation moves through the GPU as one batch.

Evaluation is a **"true jit"**. `evogp.tree.etl_engine` walks a population host-side into frozen, per-tree *static
programs*, specializes each program into a fresh ETL graph through `etl.build_cached`, and memoizes the compiled
executable in one shared `etl.CompileCache` keyed by *(tree structure, input shape, backend, device, options)*.
A structure is therefore compiled **once** and re-evaluated as a cache hit: elites, duplicated individuals and the
same population in the next generation never pay the compile cost twice.
No in-graph dispatch, no recursion — the tree *is* the graph.

There are **no custom CUDA/HIP kernels and no build step**: EvoGP is pure Python on top of `torch` and `etl`.
`pip install .` compiles nothing, `import evogp.tree` works straight from a checkout (`src/` on the import path), and
the only engine requirement is that `etl` is importable. Out of the box evaluation runs on ETL's CPU reference
interpreter (the `numpy` backend); compiled IREE backends (CUDA execution) are opt-in — see
[Evaluation Backends & Performance](#evaluation-backends--performance).

EvoGP supports multi-output trees and ships task layers for symbolic regression, classification, feature
transformation and robotics control (Brax / MuJoCo-Playground / Genesis), together with a library of interchangeable
crossover / mutation / selection operators.
EvoGP is a sister project of <a href="https://github.com/EMI-Group/evox">EvoX</a>.

## Key Features

- **"True jit" evaluation through ETL**:

    - Every tree structure is decoded host-side into an int-only static program, compiled once into an ETL graph and cached per structure — so the population-scale cost is dominated by *new* structures, not by generations.
    - The same program runs on any ETL backend: the CPU reference interpreter (`numpy`, default) or a compiled backend (IREE), chosen per process with environment variables or per call.

- **Tensor-native GP loop**:

    - Population, generation, mutation and crossover are batched torch tensor operations; there is no per-individual Python loop on the hot path.
    - Determinism comes from torch's global RNG: seed with `torch.random.manual_seed(...)` / `torch.cuda.manual_seed(...)` and a run is reproducible.

- **No build step, no toolchain**:

    - Installation is `pip install .` — no CUDA Toolkit, no ROCm, no `nvcc`/`hipcc`, no C++ compiler, no extension to rebuild after an edit.
    - The evaluation engine is device-agnostic and reads population tensors back host-side; the default backend needs no GPU at all.

- **Rich in extended content**:

    - Offers a range of genetic operation variants, allowing users to tailor configurations for specific tasks.
    - Supports multi-output trees, making it suitable for complex problems like classification and policy optimization.
    - Supports Symbolic Regression, Classification, Transformation and Policy Optimization (Brax) benchmarks.

- **Inspectable results**:

    - The best tree exports to a sympy expression (`Tree.to_sympy_expr`), to an infix string (`Tree.to_infix`) and to an image (`Tree.to_png`).

## Installation

**Requirements:** Python >= 3.11, plus PyTorch (`torch`, `numpy`, `scikit_learn`, `networkx`, `sympy` are installed as dependencies).
A **CUDA Toolkit / ROCm / C++ compiler is not required** — nothing is compiled, at install time or at run time.

### 1. Install ETL (EvoX Tensor Library)

ETL is EvoGP's evaluation engine. It is a sibling project and deliberately **not** a PyPI dependency of EvoGP
(see the comments in `requirements.txt`), so install it from its source checkout:

```bash
pip install /path/to/etl      # or: pip install -e /path/to/etl
```

### 2. Install EvoGP

```bash
pip install git+https://github.com/EMI-Group/evogp.git
# or, from a local checkout:
pip install .
```

There is no build step: the command installs Python code only — no compiler, no CUDA/ROCm toolchain, no extension.

### 3. Optional extras

```bash
pip install "evogp[iree]"        # compiled (IREE) backends, for GPU execution of the generated graphs
pip install "evogp[visualize]"   # pygraphviz, needed by Tree.to_png (also needs the Graphviz `dot` binary)
pip install "evogp[brax]"        # jax + brax, needed by the Brax robotics-control problem
pip install "evogp[dev]"         # pytest, needed to run the test suite
```

From a source checkout, use `pip install ".[iree]"` and so on; extras can be combined
(`pip install "evogp[iree,visualize]"`). EvoGP also works **without installation** from a checkout of this
repository by putting `src/` on the import path (`PYTHONPATH=src`), which is how the examples and tests below are run.

### 4. Validate the installation

```bash
python -m evogp.sr_test
```

See [Validation](#validation) for what this runs and how to size it.

## Quick Start

The canonical flow — identical to [`example/basic.py`](example/basic.py), a small symbolic-regression run on the
8-row XOR-3d problem. Run it from the repository root (`python example/basic.py`, or
`PYTHONPATH=src python example/basic.py` from an uninstalled checkout).

1. **Import the modules**:

```python
import torch
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression
from evogp.pipeline import StandardPipeline
```

2. **Define a problem** (here: symbolic regression on XOR-3d):

```python
XOR_INPUTS = torch.tensor(
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

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    device="cuda",
)

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS)
```

3. **Configure the algorithm**:

```python
# create descriptor for generating new trees
descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=50, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    enable_pareto_front=True,
)
```

4. **Run!**:

```python
pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()
```

5. **Check the details of the best tree**:

```python
# predictions on the training data (best is a Tree)
pred_res = best.forward(XOR_INPUTS)
print(pred_res)

# the symbolic formula of the best tree
sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# the non-dominated front, when enable_pareto_front=True
print(algorithm.pareto_front)
```

6. **Visualize** (optional, requires the `[visualize]` extra and the Graphviz `dot` binary):

```python
best.to_png("./imgs/xor_tree.png")
```

Every structure encountered during the run is compiled once and then served from the ETL compile cache, so
`best.forward(...)` after `pipeline.run()` is a cache hit for the structure/shape combination the run already built
(`EVOGP_ETL_CACHE_SIZE` defaults to 4096 entries, one per *(structure, shape, backend, device, options)*).

The complete, runnable script is [`example/basic.py`](example/basic.py) (it leaves the `to_png` call commented out so
that it writes nothing to disk); more end-to-end scripts live in [`example/`](example).

The demo is a wiring example, not a tuned search: the XOR-3d target is a hard one for the small `pop_size=50`,
100-generation budget above, so a run may plateau (for example at a constant such as `0.5`). Raise `pop_size` /
`generation_limit`, or start from [`example/regressor.py`](example/regressor.py), for a real search.

**Note on devices:** the tensor representation, the algorithm layer and the problem layer create their tensors on
`"cuda"` (the same string also selects AMD GPUs under ROCm) and tensors are relocated there on use, so an end-to-end
run expects a CUDA-visible device — even though the default evaluation backend computes on the CPU. Only the
evaluation engine itself (`evogp.tree.etl_engine`) is device-agnostic: it accepts population/input tensors from any
device and returns results on the input's device.

## Evaluation Backends & Performance

Tree evaluation is JIT-compiled and cached; the backend decides *where* the generated graph runs and how long it
takes to build.

| Backend | Enabled by | Runs on | Build cost per tree structure | Run cost |
|---|---|---|---|---|
| `numpy` (default) | nothing to do (explicitly: `EVOGP_ETL_BACKEND=numpy`) | CPU, ETL's reference interpreter | ~1–10 ms | ~0.5 ms per tree (N=100–1000, ~30–43-node trees) |
| `iree` (opt-in, experimental) | `pip install "evogp[iree]"` and `EVOGP_ETL_BACKEND=iree` | a CUDA device through IREE's CUDA HAL | ~0.4 s (up to ~0.9 s depending on opt level / tracing metadata) | ~2.4–3.6 ms per tree, device-resident |

**Caching economics — compile once per structure.** The engine pays a *build* the first time it sees a given
*(tree structure, input shape, backend, device, options)* pair and a cache-hit lookup (~0.45 ms) every time afterwards.
Measured on the reference box with the `numpy` backend and `ETL_DISABLE_LOCATIONS=1`:

- 30–43-node trees: **9.9 ms cold build**, **0.46 ms** warm run per tree at N=100 (0.49 ms at N=1000); a `build_cached` hit costs 0.45 ms.
- That is ~102 distinct builds per second, so a fresh random population of 1000 all-distinct trees costs ~10 s to build, ~5.4 s at 50% duplicates, and only ~0.5 s once everything is cached. Population decoding on top of that is ~10 ms at pop=200 and ~48 ms at pop=1000.
- Consequently the **numpy default is the right choice at population scale**, where almost every individual is a new structure and the build dominates.
- **IREE is for repeated structures and heavy per-run work**: the same trees evaluated again and again (elites, repeated rows, long inner loops over many data batches), where the ~0.4–0.9 s per-structure compile is amortized and the computation executes on the device. Building a fresh random population with IREE is impractical (one compile per individual), and at the measured small batch sizes the numpy interpreter's ~0.5 ms per run also beats IREE's ~2.4–3.6 ms, which is dominated by fixed launch/readback overhead.
- The cache is **in-memory only** (nothing is shared between processes) and holds one entry per *(structure, shape, backend, device, options)*; size it at `>= pop_size + headroom` because every distinct structure of a full-population fitness pass needs its own entry. Each entry pins a loaded executable (~270 KiB for a 30-node tree, ~478 KiB for a 60-node tree).

**Environment variables** (all read by `evogp.tree.etl_engine`; an explicit keyword argument on the engine call wins over the environment):

| Variable | Default | Meaning |
|---|---|---|
| `EVOGP_ETL_BACKEND` | `numpy` | Engine backend: `numpy` (CPU reference interpreter) or `iree` (compiled, CUDA). |
| `EVOGP_ETL_IREE_DEVICE` | `0` | *Visible* CUDA device index used by the IREE backend (only with `EVOGP_ETL_BACKEND=iree`). |
| `EVOGP_ETL_IREE_OPT_LEVEL` | `O1` | IREE optimization level; use `O0`/`O1` only (higher levels are buggy on CUDA). |
| `EVOGP_ETL_CACHE_SIZE` | `4096` | Maximum number of entries in the shared LRU compile cache. |
| `ETL_DISABLE_LOCATIONS` | unset | Set to `1` to skip per-op source locations while tracing: builds are roughly 4× faster (recommended for production runs). It is *not* part of the cache key, and it only costs debug metadata in tracebacks. |

```bash
# CPU reference interpreter (default), fastest builds
python example/basic.py

# compiled CUDA execution of the same run, via IREE
EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 ETL_DISABLE_LOCATIONS=1 python example/basic.py
```

Backend, device and options are always passed to `etl.build_cached` explicitly, so ambient `ETL_BACKEND` /
`ETL_DEVICE` / `ETL_TARGET_BACKENDS` cannot silently change what the engine does. `evogp.tree.etl_engine.cache_stats()`
and `evogp.tree.etl_engine.clear_cache()` inspect and reset the shared cache.

The economics above can be reproduced with the standalone probes `test/etl_jit_probe.py` (ETL's memoized-build
sugar on a realistic GP step; `--sections a,b,f` for a subset) and `src/evogp/tests/perf_probe.py --quick`
(end-to-end engine build/run economics).

## Advanced Genetic Operations
EvoGP includes multiple genetic operators, allowing users to freely assemble them to build customized TGP algorithms.

| Type       | Name                                  |
|------------|---------------------------------------|
| Selection  | [DefaultSelection](src/evogp/algorithm/selection/default.py) |
| Selection  | [RouletteSelection](src/evogp/algorithm/selection/roulette.py) |
| Selection  | [TruncationSelection](src/evogp/algorithm/selection/truncation.py) |
| Selection  | [RankSelection](src/evogp/algorithm/selection/rank.py) |
| Selection  | [TournamentSelection](src/evogp/algorithm/selection/tournament.py) |
| Crossover  | [DefaultCrossover](src/evogp/algorithm/crossover/default.py) |
| Crossover  | [DiversityCrossover](src/evogp/algorithm/crossover/diversity.py) |
| Crossover  | [LeafBiasedCrossover](src/evogp/algorithm/crossover/leaf_biased.py) |
| Mutation   | [DefaultMutation](src/evogp/algorithm/mutation/default.py) |
| Mutation   | [HoistMutation](src/evogp/algorithm/mutation/hoist.py) |
| Mutation   | [SinglePointMutation](src/evogp/algorithm/mutation/single_point.py) |
| Mutation   | [MultiPointMutation](src/evogp/algorithm/mutation/multi_point.py) |
| Mutation   | [InsertMutation](src/evogp/algorithm/mutation/insert.py) |
| Mutation   | [DeleteMutation](src/evogp/algorithm/mutation/delete.py) |
| Mutation   | [SingleConstMutation](src/evogp/algorithm/mutation/single_const.py) |
| Mutation   | [MultiConstMutation](src/evogp/algorithm/mutation/multi_const.py) |
| Mutation   | [CombinedMutation](src/evogp/algorithm/mutation/combined.py) |


## Supported Benchmarks

### Symbolic Regression

EvoGP supports symbolic regression tasks.
You can construct a `Problem` with your custom dataset:

```python
from evogp.problem import SymbolicRegression

problem = SymbolicRegression(datapoints=YOUR_DATA, labels=YOUR_LABELS)
```
Or use a predefined function to generate data:
```python
def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)

problem = SymbolicRegression(
    func=func, 
    num_inputs=2, 
    num_data=20000, 
    lower_bounds=-5, 
    upper_bounds=5
)
```

An evolved tree (rendered with `Tree.to_png`):

<img src="imgs/sr_tree.png" alt="Evolved symbolic-regression tree" height="300">

------

### Classification
EvoGP supports classification tasks.
You can construct a `Problem` with your custom dataset:

```python
from evogp.problem import Classification
problem = Classification(datapoints=YOUR_DATA, labels=YOUR_LABELS)
```

Or use the provided datasets:
```python
dataset_name = ["iris", "wine", "breast_cancer", "digits"]
problem = Classification(dataset="iris")
```

Class labels are handled with multi-output trees (`multi_output=True` by default), so the descriptor's
`output_len` should be `problem.solution_dim` (one output per class).

An evolved classification tree:

<img src="imgs/classifier_tree.png" alt="Evolved classification tree" height="300">

------
### Transformation

EvoGP supports feature transformation tasks, allowing the generation of new features from raw data to improve model performance. You can create a `Problem` with your custom dataset:

```python
from evogp.problem import Transformation

problem = Transformation(datapoints=YOUR_DATA, labels=YOUR_LABELS)
```

Or use a built-in dataset like "diabetes":

```python
problem = Transformation(dataset="diabetes")
```

During execution, EvoGP automatically generates features optimized for correlation with the target label. These new features can then be accessed through the `new_feature` interface.

------
### Robotics Control
EvoGP supports robotics control tasks.
You can create a Brax task with the following code:
```python
from evogp.problem import BraxProblem
problem = BraxProblem("swimmer")
```
**Note:** Using `BraxProblem` requires the optional [JAX](https://github.com/jax-ml/jax) and [Brax](https://github.com/google/brax) packages (the `[brax]` extra of EvoGP).

------
Once you create your problem, you can use the following code to solve them:
```python
problem = YOUR_HAVE_ALREADY_CREATED_IT

from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.pipeline import StandardPipeline

descriptor = GenerateDescriptor(
    max_tree_len=128,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)

algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=10,
)

pipeline.run()
```

Detailed examples for the above tasks are available in the [**examples**](https://github.com/EMI-Group/evogp/tree/main/example).

## Validation

EvoGP ships an end-to-end install validation and a pytest suite; both run from the repository root
(add `PYTHONPATH=src` if you are using a checkout without `pip install .`).

**1. End-to-end install validation**

```bash
python -m evogp.sr_test
```

This trains `SymbolicRegression` on the synthetic two-input target `x0^4/(x0^4+1) + x1^4/(x1^4+1)`
(1000 data points) with the standard wiring and torch seed 0, and prints `best fitness: ...`.
The defaults (`pop_size=200`, `generation_limit=100`, `max_tree_len=64`) are right-sized for the ETL engine and take
roughly **30 seconds** on the reference box; the best fitness improves generation over generation.
Scale the validation with the `EVOGP_SR_POP`, `EVOGP_SR_GENS` and `EVOGP_SR_MAX_TREE_LEN` environment variables
without editing the file:

```bash
EVOGP_SR_POP=400 EVOGP_SR_GENS=200 python -m evogp.sr_test
```

Bigger settings search harder and trade directly against wall time; keep `EVOGP_SR_GENS` at 100 or more, since with
seed 0 this population only starts improving around generation 43.

The script switches off ETL's per-op location capture (`ETL_DISABLE_LOCATIONS=1`) by default because tracing
dominates the cold build; export `ETL_DISABLE_LOCATIONS=0` to keep the capture (the computed fitness is identical
either way, the run is just much slower).

**2. Test suite**

```bash
# fast subset (a few seconds): skips the heavy engine-vs-oracle differential file
python -m pytest src/evogp/tests -q --ignore=src/evogp/tests/test_equivalence.py

# full suite, including test_equivalence.py (randomized differential runs, ~2 minutes)
python -m pytest src/evogp/tests -q
```

The suite checks the ETL evaluation engine against an independent plain-torch reference evaluator (randomized and
adversarial tree sets, plus any golden fixtures found), verifies generation/splice invariants and the public API
surface, and runs a tiny convergence smoke test. It needs `pytest` (`pip install "evogp[dev]"`); the engine tests run
on CPU with the default `numpy` backend, while the API/test harness places some tensors on `"cuda"` and therefore
expects a CUDA-visible device.

The differential file (`test_equivalence.py`) also validates `.npz` golden fixtures found in `$EVOGP_GOLDEN_DIR`,
`test/goldens/` and `src/evogp/tests/goldens/`: every `.npz` in those directories must use the documented fixture
format (`node_value`, `node_type`, `subtree_size`, `x`, and either `expected` or `expected_fitness`), otherwise the
test reports the offending file — see `src/evogp/tests/goldens/README.md`.

**3. Examples**

Run any example from the repository root (relative asset paths such as `imgs/` are CWD-dependent):

```bash
python example/basic.py
```

`example/basic.py` is the script shown in [Quick Start](#quick-start): it runs the full 100-generation XOR-3d demo in
about half a minute and prints the predictions of the best tree, its sympy expression and the Pareto front.

## Experiments in Paper

The following scripts in the `example/` directory are the main experimental scripts used in our paper:

- `example/uci_sr.py`: Main experimental script for Symbolic Regression tasks.
- `example/custom_sr.py`: Main experimental script for Custom Symbolic Regression tasks.
- `example/uci_classifier.py`: Main experimental script for Classification tasks.
- `example/brax_task.py`: Main experimental script for Reinforcement Learning (RL) tasks.

## Citation

If you use EvoGP in your research and want to cite it in your work, please use:

```
@misc{wang2025evogpgpuacceleratedframeworktreebased,
      title={EvoGP: A GPU-accelerated Framework for Tree-Based Genetic Programming}, 
      author={Lishuang Wang and Zhihong Wu and Kebin Sun and Zhuozhao Li and Ran Cheng},
      year={2025},
      eprint={2501.17168},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2501.17168}, 
}
```

## Community & Support

EvoGP is a new project, and we will continue to maintain it in the future. We warmly welcome suggestions for improvement!
- Engage in discussions and share your experiences on [GitHub Issues](https://github.com/EMI-Group/evogp/issues).
- Join our QQ group (ID: 297969717).

## Future Work

1. Improve EvoGP documentation and tutorials.
2. Implement more GP-related algorithms, such as LGP, CGP, GEP.
3. Add more multi-output methods for EvoGP.
4. Further optimize EvoGP to increase computation speed and reduce memory usage.

We warmly welcome community developers to contribute to EvoGP and look forward to your pull requests!

## Acknowledgements

1. Thanks to John R. Koza for the [Genetic Programming (GP) algorithm](https://www.genetic-programming.com/), which provided an excellent automatic programming technique and laid the foundation for the development of EvoGP.
2. Thanks to [PyTorch](https://pytorch.org/) for providing the tensor framework EvoGP is built on, and to the EvoX project's **ETL** (EvoX Tensor Library) for the JIT compilation and caching layer that evaluates EvoGP's trees.
3. Thanks to the following projects for their valuable contributions to GP research, which provided inspiration and guidance for EvoGP's design: [DEAP](https://github.com/DEAP/deap), [gplearn](https://github.com/trevorstephens/gplearn), [Karoo GP](https://github.com/kstaats/karoo_gp), [TensorGP](https://github.com/cdvetal/TensorGP) and [SymbolicRegressionGPU](https://github.com/RayZhhh/SymbolicRegressionGPU).
4. Thanks to [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [Brax](https://github.com/google/brax) for their benchmarking frameworks, which have provided a convenient environment for performance evaluation in EvoGP.
5. Thanks to [EvoX](https://github.com/EMI-Group/evox) for providing a flexible framework that allows EvoGP to integrate with other evolutionary algorithms, expanding its potential.
