# example/ — runnable demos + paper experiment scripts

## Intent
Ten flat, standalone scripts that show how to assemble an EvoGP run end-to-end: build a `Problem`, build a `GenerateDescriptor`, build a `GeneticProgramming` algorithm, hand both to `StandardPipeline`, call `.run()`.
Four of them (`uci_sr.py`, `custom_sr.py`, `uci_classifier.py`, `brax_task.py`) are the exact scripts the EvoGP paper (arXiv 2501.17168) used for its experiments (see README → "Experiments in Paper").
This directory is documentation-by-example: it is the reference for wiring the public API, and the place to copy from when writing new experiments.
There are no subdirectories, no `__init__.py` (not a package), no shared helpers, and no test harness.

## How to run
Install first from the repo root: `pip install . --no-build-isolation` (README §4; compiles the `evogp.evogp_cuda` extension — `import evogp` fails without it).
Run from the **repo root** (`python example/basic.py`): `./imgs/...` and `data/...` paths in the scripts are CWD-relative, and `mjx_task.py`'s bare `from problem import ...` would resolve against the script's own directory, where no `problem.py` exists.
Every file runs its whole experiment at import time — there are no `if __name__ == "__main__"` guards and no `def main()`, so importing a file executes the full run.
All scripts assume a CUDA/ROCm GPU: `device="cuda"` is hard-coded in every tensor-creation call except `custom_sr.py:33` (`"cuda" if torch.cuda.is_available() else "cpu"`), and the library moves CPU tensors onto CUDA on use, so even that script cannot actually run CPU-only.

## Routing table — files grouped by purpose
Minimal / getting-started demos (cheapest to run, best starting point):
- `basic.py` → canonical README example: `SymbolicRegression` on inline 8-row XOR data, `Forest.random_generate` + default operators, `enable_pareto_front=True`, prints predictions, the sympy expression and `algorithm.pareto_front`.
- `regressor.py` → `SymbolicRegression(func=..., num_inputs=2, num_data=100, lower_bounds=-5, upper_bounds=5)` on the synthetic function `x0^4/(x0^4+1) + x1^4/(x1^4+1)`.
- `classifier.py` → `Classification` on scikit-learn `iris`, Pareto front enabled, writes a tree image.
Paper experiment reproductions (module-level knobs `seed`, `pop_size`, `problem_id` are meant to be edited before the imports):
- `custom_sr.py` → symbolic regression on externally supplied CSVs `data/{problem_id}_features.csv` / `data/{problem_id}_targets.csv` (paper id 409). CSV files are **not** shipped.
- `uci_sr.py` → same algorithm configuration as `custom_sr.py`, but data is pulled live from UCI via `ucimlrepo.fetch_ucirepo(id=problem_id)` (paper id 409).
- `uci_classifier.py` → `Classification(datapoints=..., labels=...)` on a UCI table (paper id 109) with pandas-based categorical→int preprocessing.
- `brax_task.py` → RL/policy optimization: `BraxProblem("swimmer", max_episode_length=1000)`, pop 1000, no metrics printed.
Task-specific integrations:
- `transformer.py` → `Transformation(dataset="diabetes")` feature-construction task; the only file that drives evolution manually (`algorithm.step(fitness)` / `problem.evaluate(forest)` in a loop) instead of `StandardPipeline`, then scores the augmented data with a Ridge regressor.
- `mjx_task.py` → intended MuJoCo-Playground counterpart of `brax_task.py`; currently not runnable (see Known Issues).
- `combined_sr.py` → intended demo of `CombinedForest` multi-part ("formula") trees and multi-output sympy printing; currently not runnable (see Known Issues).

## Hard-coded experimental settings a re-runner must know
RNG: seeding happens at the top of the file, before the `evogp` imports; `custom_sr.py` / `uci_sr.py` / `uci_classifier.py` use `seed = 1`, `regressor.py` / `classifier.py` / `transformer.py` / `brax_task.py` / `mjx_task.py` use `0`, and `basic.py` / `combined_sr.py` set no seed at all.
Budget: `generation_limit=100` everywhere except `regressor.py` (300) and `transformer.py` (100 manual steps); no `fitness_target` or `time_limit` is ever set, so runs always go the full generation count.
Population: 50 (`basic.py`), 100 (`custom_sr.py`, `uci_sr.py`, `uci_classifier.py`), 1000 (`classifier.py`, `regressor.py`, `transformer.py`, `brax_task.py`, `mjx_task.py`), 5000 (`combined_sr.py`).
`GenerateDescriptor`: `max_tree_len` 32–512, `max_layer_cnt` 4–9 (mutation always uses `descriptor.update(max_layer_cnt=3or4)`), `using_funcs=["+","-","*","/"]` in the demos and the RL scripts and `["+","-","*","/","sin","cos","tan"]` in the three UCI/custom scripts, `const_samples=[-1,0,1]` for demos vs `const_range=[-5,5], sample_cnt=10000` for the UCI/custom scripts (`brax_task.py` / `mjx_task.py` use `const_range=[-1,1], sample_cnt=100`), `layer_leaf_prob=0.3` (UCI/custom) / `0` (`regressor.py`) / the `0.2` default elsewhere.
Operators: `DefaultCrossover()` in all ten; `DefaultMutation(mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3))` alone in `basic.py` / `transformer.py`, wrapped in `CombinedMutation([..., DeleteMutation(mutation_rate=0.8)])` in `regressor.py` / `classifier.py` / `brax_task.py` / `mjx_task.py`, with `DefaultSelection(survival_rate=0.3, elite_rate=0.01)` as selection in all six; the UCI/custom scripts use `DefaultMutation(0.1)` + `TournamentSelection(tournament_size=20, survivor_rate=0.5, elite_rate=0.1)`; `combined_sr.py` uses `CombinedDefaultCrossover()` + `CombinedDefaultMutation(0.2, descriptors=descriptor.update(max_layer_cnt=3))` + `DefaultSelection`; `enable_pareto_front=True` only in `basic.py` and `classifier.py`.
`StandardPipeline(..., is_show_details=True)` (also the default) is passed explicitly by the experiment scripts and prints per-generation fitness stats to stdout.
`SymbolicRegression.execute_mode` is left at the `"auto"` default everywhere except `combined_sr.py`, which forces `"torch"` (required for a `CombinedForest`).
The `func` callable contract for `SymbolicRegression(func=...)`: it is `torch.vmap`-ed over the data dimension, so it receives a 1-D `Tensor[num_inputs]` and must return a Tensor reshapeable to `(1,)` (see `func` in `regressor.py`).

## Public API contract exercised (all 10 scripts)
Skeleton: `problem = <Problem>(...)` → `GenerateDescriptor(...)` → `Forest.random_generate(pop_size=, descriptor=)` → `GeneticProgramming(initial_forest=, crossover=, mutation=, selection=, enable_pareto_front=)` → `StandardPipeline(algorithm, problem, generation_limit=, is_show_details=)` → `pipeline.run()`.
Load-bearing symbols and call shapes:
- `Forest.random_generate(pop_size=, descriptor=)` in 9 scripts (all but `combined_sr.py`); `CombinedForest.random_generate(pop_size, data_info, descriptors)` in `combined_sr.py:47` (currently called with stale kwargs).
- `GenerateDescriptor(...)` params used: `max_tree_len`, `input_len=problem.problem_dim`, `output_len=problem.solution_dim`, `using_funcs`, `max_layer_cnt`, `const_samples` | `const_range`+`sample_cnt`, `layer_leaf_prob`; `descriptor.update(max_layer_cnt=...)` returns a new descriptor and also mutates the receiver's stored params.
- Operators: `DefaultCrossover()`, `DefaultMutation(mutation_rate=, descriptor=)`, `DeleteMutation(mutation_rate=)`, `CombinedMutation([...])`, `TournamentSelection(tournament_size=, survivor_rate=, elite_rate=)`, `DefaultSelection(survival_rate=, elite_rate=)`, `CombinedDefaultCrossover()`, `CombinedDefaultMutation(mutation_rate=, descriptors=)`; `DiversityCrossover` is only imported (commented-out ablation knob).
- Algorithm/pipeline: `algorithm.forest`, `algorithm.step(fitness)` (`transformer.py:64` only), `algorithm.pareto_front` (`basic.py:71`, `classifier.py:64`), `pipeline.run()` → best-ever `Tree`, `pipeline.fitness` (final-generation CPU snapshot, `custom_sr.py:85`, `uci_sr.py:90`), `pipeline.algorithm.forest.batch_subtree_size[:, 0]` (per-tree node count, `custom_sr.py:86`, `uci_sr.py:91`).
- Returned `Tree`: `best.forward(x)` (`basic.py:64`), `best.to_sympy_expr()` (`basic.py:67`, `regressor.py:62`, `classifier.py:61`), `best.to_png(path)` (`classifier.py:60`, needs the optional pygraphviz plus the graphviz `dot` binary).

## Multi-output / kernel coupling in the examples
9/10 scripts pass `output_len=problem.solution_dim` to `GenerateDescriptor`; `transformer.py:33` hardcodes `output_len=1` because `Transformation` implements neither `problem_dim` nor `solution_dim`.
`solution_dim > 1` — i.e. the OUT_NODE multi-output path — is reached by `classifier.py` (iris → 3 classes; the misplaced positional `multi_output` argument still leaves the constructor default `True`), `uci_classifier.py` (same default `True`), and `brax_task.py` / `mjx_task.py` (`env.action_size`, >1 for swimmer-type envs).
`SymbolicRegression(execute_mode="torch")` (`combined_sr.py:33`) is NOT a pure-torch evaluation path: `Forest.batch_forward` still goes through `torch.ops.evogp_cuda.tree_evaluate`; `"torch"` only moves the MSE loss reduction out of the fused `tree_SR_fitness` kernel.
No example references `torch.ops.evogp_cuda` directly; the CUDA coupling visible in this directory is limited to `device="cuda"` tensor creation, `torch.cuda.manual_seed` / `torch.cuda.synchronize` (`transformer.py:66`), and the `execute_mode` strings.

## Files written
`classifier.py` writes `./imgs/classifier_tree.png` via `best.to_png(...)` (needs the optional `pygraphviz` + a working graphviz `dot`; it overwrites a committed asset in `imgs/`).
`basic.py` has the equivalent `to_png` call commented out.
No other example writes anything to disk — no CSVs, logs, checkpoints, or result files; all metrics go to stdout via `print` (the `print`/`estimator.score` output in `transformer.py`, the `Best/MSE/During time/Mean node/Max node` block in `custom_sr.py` and `uci_sr.py`).

## Constraints for code in this directory
Keep the scripts self-contained and copy-pasteable — no imports from sibling example files, no shared helper module, one script = one complete runnable story.
Keep the module-level convention: seeds and knobs as plain top-level constants set *before* the `evogp`/`torch` imports, because seeding must precede them.
Match the existing public-API surface and keep the file runnable from the repo root; if a script needs a new runtime dependency, it must be documented here (several already are optional and undeclared, see below).

## Known Issues
`regressor.py:65` prints `algorithm.pareto_front` although the script passes `enable_pareto_front=False` (line 51): `GeneticProgramming` only creates `.pareto_front` when the flag is `True` (`src/evogp/algorithm/genetic_programming.py:43-44`), so the script raises `AttributeError` after the full 300-generation run completes.
`combined_sr.py` is stale against the current combined API: `CombinedForest.random_generate` is now `random_generate(pop_size, data_info, descriptors)` and has no `formula=` / `share_input=` parameters, so line 47 already raises `TypeError` on the unexpected `formula=` kwarg; `CombinedTree.forward` also ends in `KeyError` (`combined_tree.py:44`, `pop_res` is a dict), `CombinedTree`/`CombinedForest` expose no `to_sympy_expr`/`to_png` at all — only `Tree` does — and `SymbolicRegression.evaluate` reads `forest.share_input` (`symbolic_regression.py:73`), an attribute the current `CombinedForest` never sets, so the combined-tree path is unusable end-to-end (blocker is in `src/`, not just this file).
`mjx_task.py` is dead code: line 1 does `from problem import MujocoProblem`, but no `problem.py` module exists (in `example/` or anywhere else) → `ModuleNotFoundError`. The class lives in `src/evogp/problem/mujoco_playground.py` and is exported as `evogp.problem.MujocoProblem`; fixing the import only reaches the next failure, `NameError: name 'registry' is not defined` at `mujoco_playground.py:37`, and the class additionally needs `mujoco_playground` + jax, neither installed by any extra.
`custom_sr.py` crashes immediately with `FileNotFoundError` at lines 28-29: it reads `data/{problem_id}_features.csv` and `data/{problem_id}_targets.csv`, and no `data/` directory or CSV file exists anywhere in the repo (they are not gitignored either). The paired CSVs must be supplied by the runner.
`classifier.py:25` calls `Classification(multi_output, dataset="iris")`, passing `True` into the first positional parameter, which is `datapoints`. It only works because `labels` is `None`, so the constructor falls through to the `dataset=` branch and overwrites `datapoints`; the net effect is that `multi_output` stays at its `True` default, so the run is unaffected but the argument placement is wrong.
`uci_sr.py` has no CPU fallback (hard-codes `device="cuda"`, unlike `custom_sr.py`) and only special-cases UCI ids 1 (Abalone: `Sex` mapped `M/F/I → 0/1/2`) and 9 (Auto MPG: rows with NaN `horsepower` dropped). Any other id with string columns fails at `to_numpy(dtype=np.float32)`; `uci_classifier.py` holds the generic `convert_categoricals_to_int` helper. Both UCI scripts also require network access at run time.
`custom_sr.py` and `uci_sr.py` report `mse = -torch.max(pipeline.fitness)` and node statistics from `pipeline.algorithm.forest.batch_subtree_size[:, 0]`, i.e. from the *final* generation's population, not from the best-ever individual (`pipeline.best_fitness` / `pipeline.best_tree`). Treat those printed numbers as end-of-run population snapshots.
Near-duplicates: `custom_sr.py` ≈ `uci_sr.py` (identical descriptor/algorithm configuration, only the data-loading block differs), `brax_task.py` ≈ `mjx_task.py`, and `combined_sr.py` re-uses `basic.py`'s XOR problem, `regressor.py` re-uses `src/evogp/sr_test.py`'s synthetic function. Editing one of a pair usually means editing the other.
`uci_sr.py` and `uci_classifier.py` import `CombinedMutation`, `DeleteMutation`, `DiversityCrossover` without using them (the commented-out blocks next to the operator arguments are the paper's ablation knobs, so keep them if you touch those lines).
Undeclared optional dependencies: `pandas` (`custom_sr.py`, `uci_classifier.py`) and `ucimlrepo` (`uci_sr.py`, `uci_classifier.py`) are in neither `requirements.txt` nor `setup.py`; `pygraphviz` is only a commented line in `requirements.txt` yet `classifier.py` requires it; jax/brax are `extras_require["brax"]` but `mjx_task.py` needs the additional `mujoco_playground`.
`transformer.py` works only with `Transformation(dataset="diabetes")` (the only dataset the class supports) and hard-codes `input_len=10` and `output_len=1` because `Transformation` implements neither `problem_dim` nor `solution_dim`; `new_feature(forest, 100, 10)` means "take the top-100 trees by fitness, then greedily keep at most 10 features".

## Test Strategy
There is no CI (no `.github/`, no pytest collection) and no test imports these scripts, so nothing catches API drift here — the drift is currently visible in `combined_sr.py` and `mjx_task.py`.
Cheapest checks, in order: `python -m py_compile example/*.py` (syntax only, no GPU), then `python -m evogp.sr_test` (the README's install validator and the fastest full CUDA-extension smoke test), then `python example/basic.py` from the repo root (pop 50, 100 generations, 8 rows) as the fastest real end-to-end demo — expect a per-generation fitness line and a sympy expression.
Ground truth for "is an example still valid" is the library itself: read the signatures in `src/evogp/tree/forest.py`, `combined_forest.py`, `src/evogp/problem/*.py`, and `src/evogp/pipeline/standard.py` rather than trusting these scripts.
Slow examples (avoid in smoke tests): `regressor.py` (300 generations × pop 1000), `brax_task.py` / `mjx_task.py` (jax + 1000-step episodes), `combined_sr.py` (pop 5000), and the UCI scripts (network fetch + download).
This worktree has `python3` 3.11 but no `torch`, no `pip`, no built extension and no GPU, so the examples cannot be executed here; verification must be done on a GPU host with the package installed.

## See Also
`../README.md` — narrative walkthrough of the same API (Installation, Quick Start, Supported Benchmarks, "Experiments in Paper").
`../src/evogp/` — the library itself; routing table entry `../src/evogp/problem/` for each `Problem`, `../src/evogp/pipeline/` for `StandardPipeline`, `../src/evogp/algorithm/` and `../src/evogp/tree/` for operators and tree/forest types.
`../test/` — ad-hoc verification scripts and notebooks (`fix_bug.py`, `test_bind_success.py`, `*_test.ipynb`), sibling and read-only from here; escalate any write need to the parent.
`../tutorial/` — tutorial notebooks (`evogp_intro.ipynb`, `forest_intro.ipynb`, `combined_gp.ipynb`), sibling and read-only from here.
`../imgs/` — repo-root image assets that `classifier.py` (and the commented-out call in `basic.py`) write into.
