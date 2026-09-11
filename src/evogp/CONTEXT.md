# `src/evogp/` — EvoGP: GPU-accelerated Tree-based Genetic Programming

## Intent
EvoGP is a fully GPU-accelerated Tree-based Genetic Programming (TGP) framework built on PyTorch.
Symbolic regression, classification, feature transformation and robotics-control tasks are all driven by batched GPU tensors; the per-tree interpreter and genetic operators run in custom CUDA/HIP kernels.
This directory is the install root of the `evogp` package (version 0.1.0): `setup.py` lives at the repo root and uses `find_packages(where="src")` + `package_dir={"": "src"}`.
EvoGP is the sister project of EvoX; experiment scripts live outside the package in the repo's `example/`, `tutorial/` and `test/` directories.

## Public API surface (what users import)
`src/evogp/__init__.py` is an EMPTY (0-byte) file — `import evogp` succeeds but exposes nothing, and there is no version/model re-export anywhere.
Every import in the README, `example/`, `tutorial/` and `sr_test.py` goes through a submodule:
- `from evogp.tree import Forest, CombinedForest, Tree, CombinedTree, GenerateDescriptor` (also `MAX_STACK`, `randint`, `NType`).
- `from evogp.algorithm import GeneticProgramming, DefaultSelection, DefaultMutation, DefaultCrossover` (plus all `*Crossover` / `*Mutation` / `*Selection` / `*Selector` classes, via star imports).
- `from evogp.pipeline import StandardPipeline` (and `BasePipeline`).
- `from evogp.problem import SymbolicRegression, Classification, Transformation, BraxProblem, GenesisProblem, MujocoProblem` (and `BaseProblem`; `CustomLoss` is NOT exported).
Canonical end-to-end flow (see `example/basic.py`, `sr_test.py`): `problem = <Problem>(...)` → `descriptor = GenerateDescriptor(...)` → `algorithm = GeneticProgramming(initial_forest=Forest.random_generate(pop_size=…, descriptor=…), crossover=…, mutation=…, selection=…)` → `best_tree = StandardPipeline(algorithm, problem, generation_limit=…).run()`.

## Dependency direction (the key architectural constraint)
The package is a strict four-layer DAG; preserve it when adding modules:
- `tree/` — foundation layer. Imports only torch/numpy/sympy/networkx and `evogp.evogp_cuda`. It must NEVER import from `algorithm/`, `problem/` or `pipeline/`.
- `algorithm/` — depends on `tree/` (Forest/CombinedForest + MAX_STACK/randint/NType). Note the intra-directory edge `crossover/ → selection/` (`BaseSelector`); no reverse edge exists.
- `problem/` — depends on `tree/` only.
- `pipeline/` — the top of the DAG: depends on `algorithm/` + `problem/`, and nothing may import it back (verified: zero references to `pipeline` below this layer).
All CUDA call sites (`torch.ops.evogp_cuda.*`) live ONLY in `tree/tree.py` and `tree/forest.py`; no other module touches the extension directly.

## Constraints
- Hard build requirement: an NVIDIA CUDA Toolkit + nvcc, or AMD ROCm + hipcc, and PyTorch. Install with `pip install . --no-build-isolation` (README §Installation); PyTorch's build system hipifies the same `.cu` sources for ROCm, so no source change is needed for AMD.
- `evogp.evogp_cuda` is a generated artifact (a `CUDAExtension` whose sources are exactly the four `.cu` files in `cuda/`, listed explicitly in `setup.py` — a new `.cu` must be added there by hand, there is no glob).
- `tree/__init__.py` line 2 is `import evogp.evogp_cuda`, so the shared library must exist before ANY submodule is importable. There is no CPU fallback: CPU-only machines cannot use the library at all.
- The extension registers Torch Library ops (`torch.ops.evogp_cuda.*`, CUDA dispatch key — which ROCm also uses), not pybind11 functions; the `PYBIND11_MODULE` body is empty.
- Compile-time constants are duplicated across the language boundary and must be kept in sync: `MAX_STACK = 1024` / `MAX_FULL_DEPTH = 10` in `tree/utils.py` ⇄ `cuda/defs.h`, and the `NType` / `Func` enums ⇄ `cuda/defs.h`. Changing one requires editing the other AND recompiling.
- Everything is hardcoded to the `"cuda"` device string (that is also how ROCm devices are selected) — there is no device-plumbing argument anywhere.
- Fitness convention: every problem returns a `(pop_size,)` float32 CUDA tensor that the pipeline MAXIMIZES (losses are negated), and the tensor is mutated in place (`NaN → -inf`) — so `evaluate` must return a fresh writable tensor.
- Keep `src/evogp/__init__.py` empty: adding imports there would make `import evogp` require the built extension and would not match the documented import style.
- `setup.py` deps: torch, numpy, scikit_learn, networkx, sympy; extras `visualize = [pygraphviz]` and `brax = [jax, brax]`. There is NO extra for `mujoco_playground` or `genesis` even though `problem/` needs them.

## Design Decisions
- Descriptor-driven representation: `GenerateDescriptor(max_tree_len, input_len, output_len, using_funcs, max_layer_cnt, layer_leaf_prob, const_prob, out_prob, const_samples, const_range, sample_cnt)` fully parameterizes random tree generation; it is consumed both by the CUDA `tree_generate` kernel (via materialized `depth2leaf_probs` / `roulette_funcs` cdf tensors) and by the torch-side mutation operators.
- Population = batched fixed-width tensors: three parallel `[pop_size, max_tree_len]` matrices (`batch_node_value` float32, `batch_node_type` int16, `batch_subtree_size` int16) in prefix (pre-order) encoding with node 0 = root; only the first `subtree_size[0]` entries of each row are meaningful.
- Operationally parallel by container: `Forest` (single output) and `CombinedForest` (multi-output, dict-driven via `data_info`) are interchangeable at the `GeneticProgramming`/`StandardPipeline` level, so new operators should accept the container rather than assume tensors.
- Elitism is the algorithm's job, not the operators': operators return offspring only, and `GeneticProgramming.step` prepends elite copies.
- Optional heavy dependencies are always imported lazily inside constructors (`import_jax_based_package`, `import_genesis_based_package`), never at module import time, except `go2_env.py` (which is deliberately not imported by `problem/__init__.py`).
- Multi-output trees pack an output index into the node value with the `OUT_NODE` (bit 7) flag rather than composing trees.

## Known Issues
Cross-cutting problems only; per-directory detail lives in the child CONTEXT.md files.
- The library has essentially no automated test coverage: `test/` holds ad-hoc scripts and notebooks, everything requires a GPU plus a successful kernel build, and no automated test drives the multi-output CUDA instantiations (`out_len > 1`, reachable at runtime only via `example/classifier.py`, the robotics examples, or `CombinedForest`-based code).
- The multi-output / `CombinedForest` path is partly bit-rotted: `SymbolicRegression.evaluate` still reads `forest.share_input` (removed) and is incompatible with the dict-based `CombinedForest`, and `example/combined_sr.py` / `tutorial/combined_gp.ipynb` still use the removed `formula=` / `share_input=` API. `CombinedTree.forward` ends by indexing a dict → `KeyError`, and there is no `CombinedTree.to_sympy_expr`.
- Container/operator mismatch is never validated: nothing checks that `CombinedDefaultCrossover`/`CombinedDefaultMutation` are paired with a `CombinedForest`, nor that `descriptor.input_len/output_len` match `problem.problem_dim/solution_dim` (the pipeline's dimension asserts are commented out).
- Some shipped problems are broken or stubs: `MujocoProblem.__init__` raises `NameError` (locals imported without `global`), `GenesisProblem.evaluate` returns `None` and its `go2_env` helper needs URDFs absent from the repo, `CustomLoss` is unreachable from `problem/__init__.py`.
- `algorithm/__init__.py` uses star imports with no `__all__`, so `evogp.algorithm.base` / `evogp.algorithm.default` resolve to the SELECTION submodules (last star import wins) — import the concrete subpackage (`evogp.algorithm.crossover.base`) instead.
- `GenerateDescriptor.update(**kwargs)` returns a new descriptor but also mutates the receiver's private state and re-draws random `const_samples`; `MAX_STACK` overflow in the kernels is unchecked (`max_layer_cnt` > 10 removes the leaf terminator).
- `execute_mode="auto"` is the `SymbolicRegression` default and routes to the CUDA kernels; `"torch"` is the only mode that can evaluate a `CombinedForest`.

## `sr_test.py` (top of this node)
A standalone install-validation smoke script, added by commit "add sr_test for validate installation" and documented in README §5 as `python -m evogp.sr_test`; it is NOT a unit test and is not pytest-collected.
It trains `SymbolicRegression` on a synthetic 2-input function (`func`/`num_inputs`/`num_data` constructor path) for 100 generations with pop_size 1000, seeding `torch.manual_seed(0)` / `torch.cuda.manual_seed(0)` for reproducible output.
It has module-level side effects and NO `if __name__ == "__main__"` guard, so any `import evogp.sr_test` starts a full training run — never import it from tooling.
It is the closest thing to a regression gate: if the kernels build and this script converges, the install is good.

## Notes for Agents
- Build the extension first; a failed build breaks every submodule, and the resulting error is an `ImportError` on `evogp.evogp_cuda` raised from `tree/__init__.py`.
- Keep `MAX_STACK` / `MAX_FULL_DEPTH` / `NType` / `Func` in sync between `tree/utils.py` and `cuda/defs.h`, then rebuild.
- Never edit `src/evogp/hip/` — it is the torch-hipify mirror generated at build time and is gitignored.
- Import order inside `__init__.py` files is load-bearing in `tree/` (extension import + `utils` before `descriptor`) and in `problem/` (`.base` first, since siblings do `from . import BaseProblem`).
- Determinism comes from torch's global CUDA RNG: the generate kernel seeds a taus88 per tree with `hash(index, key0, key1)`, with keys drawn from `torch.rand` at call time.
- Trees returned by `Forest.__getitem__` are views (not copies) into the population tensors — the algorithm rebinds `self.forest` each generation, so old buffers are never rewritten, but do not mutate a snapshot.
- This node owns only these CONTEXT.md files; do not modify source code here.

## Test Strategy
There is no pytest suite in this package (or in the repo): `test/` at the repo root contains `test_bind_success.py` (a raw `torch.ops.evogp_cuda.*` driver), `test_torch_env.py`, `fix_bug.py` and notebooks.
The de-facto validation sequence is: successfully build → `python -m evogp.sr_test` → run the relevant `example/*.py` script.
Known coverage gaps: the multi-output CUDA instantiations, the `*Selector`-based crossovers (`DiversityCrossover`, `LeafBiasedCrossover`), all non-default selection strategies, `Transformation.problem_dim/solution_dim`, `StandardPipeline`'s `fitness_target` path, and the robotics-control problems (jax/genesis required).

## Routing Table
- `src/evogp/tree/` → Core tree representation and GPU execution: `GenerateDescriptor`, `Tree`/`Forest`, `CombinedTree`/`CombinedForest`, tensor schema, CUDA op call sites, sympy/png export. Foundation layer — start here to understand the data model.
- `src/evogp/algorithm/` → GP operator layer: `GeneticProgramming` loop + `crossover/`, `mutation/`, `selection/` subpackages (each has its own CONTEXT.md).
- `src/evogp/problem/` → Benchmark/task layer: `BaseProblem` contract, `SymbolicRegression`, `Classification`, `Transformation`, Brax/MuJoCo/Genesis robotics problems, `CustomLoss`.
- `src/evogp/pipeline/` → Training-loop layer: `BasePipeline`, `StandardPipeline` (fitness handling, best-tree tracking, termination).
- `src/evogp/cuda/` → CUDA/HIP kernel sources (`defs.h`, `kernel.h` + four `.cu` extensions); no Python; read this to debug a build failure or kernel behavior.
- `src/evogp/sr_test.py` → install-validation smoke script (see above).
- Sibling paths (read-only from here — escalate writes to this node's parent): `../../setup.py` (build config, version, extras), `../../requirements.txt`, `../../README.md` (installation + API docs), `../../example/` (runnable end-to-end scripts), `../../tutorial/` (notebooks), `../../test/` (ad-hoc verification scripts).
