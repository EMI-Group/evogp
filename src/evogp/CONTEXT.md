# `src/evogp/` — EvoGP: Tree-based Genetic Programming on PyTorch + ETL

## Intent
EvoGP is a Tree-based Genetic Programming (TGP) framework for symbolic regression, classification, feature transformation and robotics-control benchmarks (Brax / MuJoCo-Playground / Genesis), with multi-output trees for classification and policy optimization.
Evaluation is a **"true jit"**: `tree/etl_engine.py` walks a population host-side into frozen per-tree static programs and specializes each structure into a fresh ETL graph through `etl.build_cached` (memoized in one shared `etl.CompileCache`); random generation, mutation and crossover are plain vectorized torch (`tree/generation.py`, `tree/splicing.py`).
There is **no custom CUDA/HIP kernel and no compiled extension**: nothing has to be built, `import evogp.<submodule>` works from a plain checkout, and the external **`etl`** package (EvoX Tensor Library) is the evaluation-engine dependency.
This directory is the install root of the `evogp` package (version 0.1.0): `setup.py` lives at the repo root and uses `find_packages(where="src")` + `package_dir={"": "src"}` (repo-root packaging/dependency edits are a separate task).
EvoGP is the sister project of EvoX; experiment scripts live outside the package in the repo's `example/`, `tutorial/` and `test/` directories.
`tree/ENGINE_SPEC.md` (semantics of record), `tree/CONTEXT.md` (representation, engine, API details) and `tests/CONTEXT.md` (the pytest suite) are the authoritative companions to this file.

## Public API surface (what users import)
`src/evogp/__init__.py` is an EMPTY (0-byte) file — `import evogp` succeeds, has no side effects and exposes nothing; there is no version/model re-export anywhere.
Every import in the README, `example/`, `tutorial/` and `sr_test.py` goes through a submodule:
- `from evogp.tree import Forest, CombinedForest, Tree, CombinedTree, GenerateDescriptor` (also `MAX_STACK`, `randint`, `NType`).
- `from evogp.algorithm import GeneticProgramming, DefaultSelection, DefaultMutation, DefaultCrossover` (plus all `*Crossover` / `*Mutation` / `*Selection` / `*Selector` classes, via star imports).
- `from evogp.pipeline import StandardPipeline` (and `BasePipeline`).
- `from evogp.problem import SymbolicRegression, Classification, Transformation, BraxProblem, GenesisProblem, MujocoProblem, CustomLoss` (plus `BaseProblem`).
Canonical end-to-end flow (see `example/basic.py`, `sr_test.py`): `problem = <Problem>(...)` → `descriptor = GenerateDescriptor(...)` → `algorithm = GeneticProgramming(initial_forest=Forest.random_generate(pop_size=…, descriptor=…), crossover=…, mutation=…, selection=…)` → `best_tree = StandardPipeline(algorithm, problem, generation_limit=…).run()`.

## Dependency direction (the key architectural constraint)
The package is a strict four-layer DAG; preserve it when adding modules:
- `tree/` — foundation layer: representation (`tree.py` / `forest.py` / `combined_tree.py` / `combined_forest.py`), the generation blueprint (`descriptor.py`), the pure-torch operators (`generation.py`, `splicing.py`) and the ETL evaluation engine (`etl_engine.py`). It imports only torch/numpy/sympy/networkx, `etl` and stdlib; it must NEVER import from `algorithm/`, `problem/` or `pipeline/`.
- `algorithm/` — depends on `tree/` (Forest/CombinedForest + MAX_STACK/randint/NType). Note the intra-directory edge `crossover/ → selection/` (`BaseSelector`); no reverse edge exists.
- `problem/` — depends on `tree/` only.
- `pipeline/` — the top of the DAG: depends on `algorithm/` + `problem/`, and nothing may import it back (verified: zero references to `pipeline` below this layer).
The engine is reached ONLY through the `Forest`/`Tree` wrappers (`forward` / `batch_forward` / `SR_fitness`): no module outside `tree/` imports `etl_engine` or the `etl` package.

## Constraints
- **No build step, no compiled extension**: a built shared library is no longer a prerequisite for importing the library; the only engine requirement is that `etl` is importable (top-level `import etl` in `tree/etl_engine.py`; the dev venv has `etl 0.1.0` installed editable).
- Standard invocation for every script/test: repo root as CWD, `PYTHONPATH=src`, and the project venv (`/mnt/local-ssd/bchuang/evox/.venv/bin/python`; `python` is not on PATH).
- ETL engine knobs: backend defaults to `"numpy"` (CPU reference interpreter, no GPU needed); IREE is opt-in via `EVOGP_ETL_BACKEND=iree` + `EVOGP_ETL_IREE_DEVICE` / `EVOGP_ETL_IREE_OPT_LEVEL` and is experimental. `ETL_DISABLE_LOCATIONS=1` speeds tracing up ~4× and is recommended for real runs (it is NOT part of the cache key).
- Device policy: the engine (`etl_engine.evaluate` / `sr_fitness`) is device-agnostic — it accepts tensors on any torch device and returns results on the input device unless `out_device` is given. Everything ABOVE it is still hard-coded to the `"cuda"` device string (`check_tensor` relocates to `"cuda"`, descriptor tensors, `zero_generate`, `randint`, the algorithm operators, the problem data), the same string also selects AMD GPUs under ROCm, and there is no device-plumbing argument anywhere. A full training run therefore still needs a CUDA-visible device; only the engine itself can be exercised on CPU.
- Constants are single-sourced in Python: `MAX_STACK = 1024`, `MAX_FULL_DEPTH = 10`, `NType`, `Func` live only in `tree/utils.py` (there is no C++/header mirror to keep in sync any more).
- Fitness convention: every problem returns a `(pop_size,)` float32 tensor that the pipeline MAXIMIZES (losses are negated) and mutates in place (`NaN → -inf`) — so `evaluate` must return a fresh writable tensor.
- Keep `src/evogp/__init__.py` empty: adding imports there would make `import evogp` pull submodules and would not match the documented import style (`tests/test_api_compat.py` asserts the empty-`__init__` semantics).
- `setup.py` deps (repo root): torch, numpy, scikit_learn, networkx, sympy; extras `visualize = [pygraphviz]` and `brax = [jax, brax]`. There is NO extra for `mujoco_playground` or `genesis` even though `problem/` needs them, and `etl` is not declared there yet.
- The shared ETL cache (`etl_engine.CACHE`, sized by `EVOGP_ETL_CACHE_SIZE`, default 4096 entries) is in-memory only — nothing persists across processes; `cache_stats()` / `clear_cache()` inspect/reset it.

## Design Decisions
- Descriptor-driven representation: `GenerateDescriptor(max_tree_len, input_len, output_len, using_funcs, max_layer_cnt, layer_leaf_prob, const_prob, out_prob, const_samples, const_range, sample_cnt)` fully parameterizes random tree generation; it is consumed by the pure-torch generator (`generation.py`) and by the algorithm-layer mutation operators (via its materialized `depth2leaf_probs` / roulette cdf tensors).
- Host-side decode + per-structure ETL graph: the population tensors are read back to the CPU, validated and turned into flat int-only programs (constant subtrees folded), then specialized per *tree structure* so re-evaluating elites/duplicates is a cache hit; statics are re-passed by value at `etl.run`, so the cache key fully determines the executable. Validation replaces the legacy kernels' undefined behavior (a malformed tree raises `ValueError` naming row and node).
- Pure-torch operators replace the kernels: portable, toolchain-free and fast enough at these sizes, while keeping the legacy splice semantics including the silent copy-on-invalid-position behavior.
- numpy backend default, IREE opt-in: per-structure IREE compilation (~0.4–0.6 s) dominates while numpy builds in ms and runs in ~1 ms; IREE stays first-class for callers who need device execution and accept the compile economics.
- Population = batched fixed-width tensors: three parallel `[pop_size, max_tree_len]` matrices (`batch_node_value` float32, `batch_node_type` int16, `batch_subtree_size` int16) in prefix (pre-order) encoding with node 0 = root; only the first `subtree_size[0]` entries of each row are meaningful.
- Operationally parallel by container: `Forest` (single output) and `CombinedForest` (multi-output, dict-driven via `data_info`) are interchangeable at the `GeneticProgramming`/`StandardPipeline` level, so new operators should accept the container rather than assume tensors.
- Elitism is the algorithm's job, not the operators': operators return offspring only, and `GeneticProgramming.step` prepends elite copies.
- `execute_mode` is kept for API compatibility only: the legacy `kernel_type` dispatch is gone and all five accepted strings (`"torch"`, `"hybrid parallel"`, `"data parallel"`, `"tree parallel"`, `"auto"`) map to the same engine path.
- Multi-output trees pack an output index into the node value with the `OUT_NODE` (bit 7) flag rather than composing trees; the engine implements the multi-output protocol directly and is oracle-verified for `output_len` 2/3, while single-output trees containing an OUT_NODE tag are rejected at decode time.
- Optional heavy dependencies are always imported lazily inside constructors (`import_jax_based_package`, `import_genesis_based_package`), never at module import time, except `go2_env.py` (which is deliberately not imported by `problem/__init__.py`).

## Known Issues
Cross-cutting problems only; per-directory detail lives in the child CONTEXT.md files.
- **Per-structure JIT cold-build economics**: one cache entry per distinct tree structure, so a fresh random population is a cold miss per individual — measured numpy numbers: pop 200 → 9.6 ms decode, 1.16 s cold, 105 ms warm; pop 1000 → 47.8 ms decode, 5.5 s cold, 532 ms warm. Constant trees never compile, duplicate structures hit, and changing batch size / tree shape re-compiles — keep shapes stable in hot loops. IREE compile cost makes it impractical at population scale.
- **Multi-output status**: the engine supports it (`output_len` 2/3 oracle-verified; `CombinedForest`/`CombinedTree` are the `data_info` dict-driven containers, and the single/batched `forward`/`batch_forward` input contract is covered by `tests/test_combined_forward.py`). The remaining combined-path gaps are outside the engine: `CombinedForest.__add__(CombinedTree)` reads `other.tree[i]` (AttributeError), `CombinedForest.__next__` never advances `iter_index` (endless iteration), neither combined container defines `SR_fitness` or `to_sympy_expr`, and `SymbolicRegression.evaluate` rejects a `CombinedForest` with `NotImplementedError`, so the combined route runs through `evogp.problem.CustomLoss` (the route taken by `example/combined_sr.py`, which runs end to end, and by `tutorial/combined_gp.ipynb`).
- No dimension validation exists between `GenerateDescriptor.input_len/output_len` and `problem.problem_dim/solution_dim` (the pipeline's asserts are commented out), so mismatches fail inside the tree layer / engine instead of at wiring time.
- Some shipped problems are broken or stubs: `GenesisProblem.evaluate` returns `None` (stub); the robotics problems additionally need lazily-imported `jax`/`brax`/`mujoco_playground`/`genesis` (only `brax` has a setup.py extra).
- Robotics env construction cannot be verified in the current validation venv (a third-party environment boundary, not a library defect): dm_control-suite MuJoCo envs default to `impl="warp"` and `mujoco-warp` is not installed, so `registry.load` fails with `AttributeError: type object 'int' has no attribute 'WARP'` inside `mujoco/mjx`; behind it sits the CuDNN 9.1-vs-9.8 `JaxRuntimeError ... dnn_support != nullptr` wall that also hits `example/brax_task.py`. `MujocoProblem`'s lazy-import binding itself is covered by `tests/test_mujoco_problem.py`, but no robotics rollout is verified here.
- `algorithm/__init__.py` uses star imports with no `__all__`, so `evogp.algorithm.base` / `evogp.algorithm.default` resolve to the SELECTION submodules (last star import wins) — import the concrete subpackage (`evogp.algorithm.crossover.base`) instead.
- `GenerateDescriptor.update(**kwargs)` returns a new descriptor but also mutates the receiver's captured params (kwargs are sticky) and re-draws `const_samples`; `MAX_STACK` overflow / malformed indices are now caught by the host-side decode (`ValueError`) rather than silently corrupting a kernel run.
- Sympy rendering is a display artifact with documented edge gaps (e.g. `LOOSE_POW(0,0)` renders `1` while the engine computes `0`, `LOG(0)` renders `zoo` while the engine returns `-inf`); see `tree/CONTEXT.md`.
- The repo-root `test/` scripts (`test_bind_success.py`, …) still drive the deleted extension's custom torch ops and can no longer run; they are outside this node.

## `sr_test.py` (top of this node)
Standalone install-validation smoke script documented in README §5 as `python -m evogp.sr_test`; it is NOT a unit test and is not pytest-collected.
It trains `SymbolicRegression` on a synthetic 2-input function (`func`/`num_inputs`/`num_data` constructor path) under fixed manual seeds and reports progress through `StandardPipeline`; its defaults are right-sized for the pure-Python/ETL engine (a validation run, not a benchmark) and it needs a CUDA-visible device like every other end-to-end entry point.
It has module-level side effects and NO `if __name__ == "__main__"` guard, so any `import evogp.sr_test` starts a full training run — never import it from tooling.

## Notes for Agents
- There is no build step: run everything from the repo root with `PYTHONPATH=src` and `/mnt/local-ssd/bchuang/evox/.venv/bin/python`, with `ETL_DISABLE_LOCATIONS=1` for speed. `etl` must be importable (`import etl` in `tree/etl_engine.py`).
- The pytest suite under `tests/` is the regression gate; `tests/goldens/*.npz` are the engine-vs-legacy-kernels golden fixtures and `tests/perf_probe.py` is a standalone build/run economics probe (not pytest).
- Import order inside `__init__.py` files is load-bearing in `tree/` (`utils` before `descriptor`) and in `problem/` (`.base` first, since siblings do `from . import BaseProblem`).
- `MAX_STACK` / `MAX_FULL_DEPTH` / `NType` / `Func` are defined only in `tree/utils.py`; function ids are positional across `Func`, `FUNCS_NAMES`, `FUNCS_DISPLAY`, `SYMPY_MAP`, so adding an operator means touching all four.
- Determinism comes from torch's global RNG: `generation.py` draws with `torch.rand`/`randint` and the splices use plain torch ops, so `torch.manual_seed(0)` makes generation reproducible (distribution-equivalent to the legacy taus88 stream, not bit-identical).
- Trees returned by `Forest.__getitem__` (int/slice) are views into the population tensors while `Tensor`/`ndarray` indexing returns copies — the algorithm rebinds `self.forest` each generation, so old buffers are never rewritten, but do not mutate a snapshot or write through a copy.
- Never reintroduce a compiled extension: the tree layer is deliberately extension-free and `tree/ENGINE_SPEC.md`'s Appendices are the only place where the legacy op schemas are kept, explicitly as historical reference.
- This node owns only these CONTEXT.md files; do not modify source code here (source changes belong to the specific child node's task).

## Test Strategy
The repo now HAS an automated pytest suite: `src/evogp/tests/` (see its CONTEXT.md for per-file detail) — an independent plain-torch reference evaluator (`reference_eval.py`), randomized/adversarial engine equivalence (`test_equivalence.py`), operator/splice invariants (`test_operators.py`), public-API compatibility (`test_api_compat.py`), and a tiny SR convergence smoke (`test_convergence_smoke.py`), plus the in-repo golden fixtures `tests/goldens/*.npz` and the standalone `tests/perf_probe.py`.
Run it from the repo root: `PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -q` (add `-x` to stop at the first failure, `--ignore=src/evogp/tests/test_equivalence.py` to skip the heavy differential file). Everything runs on CPU with the default numpy engine backend except the parts that build their tensors on `"cuda"` (they need a GPU).
The other de-facto validations are `python -m evogp.sr_test` and running the relevant `example/*.py` from the repository root (relative asset paths such as `imgs/` are CWD-dependent). The repo-root `test/` directory is NOT a suite any more — its scripts drive the deleted CUDA extension.
Known coverage gaps: IREE numeric parity per shape, the robotics-control problems (jax/genesis required), `CustomLoss`, `Classification(multi_output=False)`, `Transformation.new_feature`, and the multi-output `CombinedForest` end-to-end route.

## Routing Table
- `src/evogp/tree/` → Foundation layer: `GenerateDescriptor`, `Tree`/`Forest`, `CombinedTree`/`CombinedForest`, tensor schema, pure-torch generation/splicing, the ETL engine (`etl_engine.py`), sympy/png export. `ENGINE_SPEC.md` holds the semantics of record. Start here to understand the data model.
- `src/evogp/algorithm/` → GP operator layer: `GeneticProgramming` loop + `crossover/`, `mutation/`, `selection/` subpackages (each has its own CONTEXT.md).
- `src/evogp/problem/` → Benchmark/task layer: `BaseProblem` contract, `SymbolicRegression`, `Classification`, `Transformation`, Brax/MuJoCo/Genesis robotics problems, `CustomLoss`.
- `src/evogp/pipeline/` → Training-loop layer: `BasePipeline`, `StandardPipeline` (fitness handling, best-tree tracking, termination).
- `src/evogp/tests/` → The pytest suite, golden fixtures and the perf probe (see its CONTEXT.md).
- `src/evogp/sr_test.py` → install-validation smoke script (see above).
- Sibling paths (read-only from here — escalate writes to this node's parent): `../../setup.py` (build config, packaging, extras), `../../requirements.txt`, `../../README.md` (installation + API docs), `../../example/` (runnable end-to-end scripts), `../../tutorial/` (notebooks), `../../test/` (legacy ad-hoc scripts targeting the deleted extension).
