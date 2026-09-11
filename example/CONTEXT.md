# example/ — runnable demos + paper experiment scripts

## Intent
Ten flat, standalone scripts that show how to assemble an EvoGP run end-to-end: build a `Problem`, build a `GenerateDescriptor`, build a `GeneticProgramming` algorithm, hand both to `StandardPipeline`, call `.run()`.
Four of them (`uci_sr.py`, `custom_sr.py`, `uci_classifier.py`, `brax_task.py`) are the exact scripts the EvoGP paper (arXiv 2501.17168) used for its experiments (see README → "Experiments in Paper").
This directory is documentation-by-example: it is the reference for wiring the public API, and the place to copy from when writing new experiments.
There are no subdirectories, no `__init__.py` (not a package), no shared helpers, and no test harness.

## How to run
Install first from the repo root: `pip install -e .` (compiles the `evogp.evogp_cuda` extension; `import evogp` fails without it).
Run from the **repo root** (`python example/basic.py`): `./imgs/...` and `data/...` paths in the scripts are CWD-relative, and `mjx_task.py`'s bare `from problem import ...` resolves against the script's own directory.
Every file runs its whole experiment at import time — there are no `if __name__ == "__main__"` guards and no `def main()`, so importing a file executes the full run.
All scripts assume a CUDA/ROCm GPU (`device="cuda"` is hard-coded, and EvoGP's kernels are CUDA-only).

## Routing table — files grouped by purpose
Minimal / getting-started demos (cheapest to run, best starting point):
- `basic.py` → canonical README example: `SymbolicRegression` on inline 8-row XOR data, `Forest.random_generate` + default operators, prints predictions and the sympy expression.
- `regressor.py` → `SymbolicRegression(func=..., num_inputs=2, num_data=100, bounds=[-5,5])` on the synthetic function `x0^4/(x0^4+1) + x1^4/(x1^4+1)`.
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
RNG: `torch.random.manual_seed(...)` and `torch.cuda.manual_seed(...)` are called before the `evogp` imports; `custom_sr.py` / `uci_sr.py` / `uci_classifier.py` use `seed = 1`, all others use `0`.
Budget: `generation_limit=100` everywhere except `regressor.py` (300) and `transformer.py` (100 manual steps); no `fitness_target` or `time_limit` is ever set, so runs always go the full generation count.
Population: 50 (`basic.py`), 100 (`custom_sr.py`, `uci_sr.py`, `uci_classifier.py`), 1000 (`classifier.py`, `regressor.py`, `transformer.py`, `brax_task.py`, `mjx_task.py`), 5000 (`combined_sr.py`).
`GenerateDescriptor`: `max_tree_len` 32–512, `max_layer_cnt` 4–9 (mutation always uses `descriptor.update(max_layer_cnt=3or4)`), `using_funcs=["+","-","*","/"]` for demos and `+ sin`, `cos`, `tan` for the experiment scripts, `const_samples=[-1,0,1]` for demos vs `const_range=[-5,5], sample_cnt=10000` for the experiment scripts, `layer_leaf_prob=0.3` (experiments) / `0` (`regressor.py`).
Operators: demos and RL scripts use `CombinedMutation([DefaultMutation(0.2), DeleteMutation(0.8)])` + `DefaultSelection(survival_rate=0.3, elite_rate=0.01)`; the UCI/custom scripts use `DefaultMutation(0.1)` + `TournamentSelection(tournament_size=20, survivor_rate=0.5, elite_rate=0.1)`; `enable_pareto_front=True` only in `basic.py` and `classifier.py`.
`StandardPipeline(..., is_show_details=True)` (also the default) is passed explicitly by the experiment scripts and prints per-generation fitness stats to stdout.
`SymbolicRegression.execute_mode` is left at the `"auto"` default everywhere except `combined_sr.py`, which forces `"torch"` (required for a `CombinedForest`).
The `func` callable contract for `SymbolicRegression(func=...)`: it is `torch.vmap`-ed over the data dimension, so it receives a 1-D `Tensor[num_inputs]` and must return a `Tensor[1]` (see `func` in `regressor.py`).

## Files written
`classifier.py` writes `./imgs/classifier_tree.png` via `best.to_png(...)` (needs the optional `pygraphviz` + a working graphviz `dot`; it overwrites a committed asset in `imgs/`).
`basic.py` has the equivalent `to_png` call commented out.
No other example writes anything to disk — no CSVs, logs, checkpoints, or result files; all metrics go to stdout via `print` (the `print`/`estimator.score` output in `transformer.py`, the `Best/MSE/During time/Mean node/Max node` block in `custom_sr.py` and `uci_sr.py`).

## Constraints for code in this directory
Keep the scripts self-contained and copy-pasteable — no imports from sibling example files, no shared helper module, one script = one complete runnable story.
Keep the module-level convention: seeds and knobs as plain top-level constants set *before* the `evogp`/`torch` imports, because seeding must precede them.
Match the existing public-API surface and keep the file runnable from the repo root; if a script needs a new runtime dependency, it must be documented here (several already are optional and undeclared, see below).

## Known Issues
`mjx_task.py` is dead code: it does `from problem import MujocoProblem`, but no `problem.py` module exists (in `example/` or anywhere else). The class lives in `src/evogp/problem/mujoco_playground.py` and is exported as `evogp.problem.MujocoProblem`; it additionally needs `mujoco_playground` + jax, neither of which is installed by any extra.
`combined_sr.py` is stale against the current combined API: `CombinedForest.random_generate` is now `random_generate(pop_size, data_info, descriptors)` and has no `formula=` / `share_input=` parameters, so both its construction call and its `best.to_sympy_expr(lambda ...)` / `best.A.to_sympy_expr()` calls fail. `CombinedTree`/`CombinedForest` expose no `to_sympy_expr` at all — only `Tree` does — and `SymbolicRegression.evaluate` reads `forest.share_input`, an attribute the current `CombinedForest` never sets, so the combined-tree path is currently unusable end-to-end (blocker is in `src/`, not just this file).
`custom_sr.py` crashes immediately with `FileNotFoundError`: it reads `data/{problem_id}_features.csv` and `data/{problem_id}_targets.csv`, and no `data/` directory or CSV file exists anywhere in the repo (they are not gitignored either). The paired CSVs must be supplied by the runner.
`classifier.py` calls `Classification(multi_output, dataset="iris")`, passing `True` into the first positional parameter, which is `datapoints`. It only works because `labels` is `None`, so the constructor falls through to the `dataset=` branch and overwrites `datapoints`; the `multi_output` intent is currently lost/misordered.
`uci_sr.py` has no CPU fallback (hard-codes `device="cuda"`, unlike `custom_sr.py`) and only special-cases UCI ids 1 (Abalone: `Sex` mapped `M/F/I → 0/1/2`) and 9 (Auto MPG: rows with NaN `horsepower` dropped). Any other id with string columns fails at `to_numpy(dtype=np.float32)`; `uci_classifier.py` holds the generic `convert_categoricals_to_int` helper. Both UCI scripts also require network access at run time.
`custom_sr.py` and `uci_sr.py` report `mse = -torch.max(pipeline.fitness)` and node statistics from `pipeline.algorithm.forest.batch_subtree_size[:, 0]`, i.e. from the *final* generation's population, not from the best-ever individual (`pipeline.best_fitness` / `pipeline.best_tree`). Treat those printed numbers as end-of-run population snapshots.
Near-duplicates: `custom_sr.py` ≈ `uci_sr.py` (identical descriptor/algorithm configuration, only the data-loading block differs), `brax_task.py` ≈ `mjx_task.py`, and `combined_sr.py` re-uses `basic.py`'s XOR problem, `regressor.py` re-uses `src/evogp/sr_test.py`'s synthetic function. Editing one of a pair usually means editing the other.
`uci_sr.py` and `uci_classifier.py` import `CombinedMutation`, `DeleteMutation`, `DiversityCrossover` and `CombinedMutation`/`DeleteMutation` respectively without using them; the commented-out blocks next to the operator arguments are the paper's ablation knobs, so keep them if you touch those lines.
Undeclared optional dependencies: `pandas` (`custom_sr.py`, `uci_classifier.py`) and `ucimlrepo` (`uci_sr.py`, `uci_classifier.py`) are in neither `requirements.txt` nor `setup.py`; `pygraphviz` is only a commented line in `requirements.txt` yet `classifier.py` requires it; jax/brax are `extras_require["brax"]` but `mjx_task.py` needs the additional `mujoco_playground`.
`transformer.py` works only with `Transformation(dataset="diabetes")` (the only dataset the class supports) and hard-codes `input_len=10` and `new_feature(forest, 100, 10)`.

## Test Strategy
There is no CI (no `.github/`, no pytest collection) and no test imports these scripts, so nothing catches API drift here — the drift is currently visible in `combined_sr.py` and `mjx_task.py`.
Cheapest checks, in order: `python -m py_compile example/*.py` (syntax only, no GPU), then `python -m evogp.sr_test` (the README's install validator and the fastest full CUDA-extension smoke test), then `python example/basic.py` from the repo root (pop 50, 100 generations, 8 rows) as the fastest real end-to-end demo — expect a per-generation fitness line and a sympy expression.
Ground truth for "is an example still valid" is the library itself: read the signatures in `src/evogp/tree/forest.py`, `combined_forest.py`, `src/evogp/problem/*.py`, and `src/evogp/pipeline/standard.py` rather than trusting these scripts.
Slow examples (avoid in smoke tests): `regressor.py` (300 generations × pop 1000), `brax_task.py` / `mjx_task.py` (jax + 1000-step episodes), `combined_sr.py` (pop 5000), and the UCI scripts (network fetch + download).
This worktree has no `python`/`pip` on PATH, so the examples cannot be executed here; verification must be done on a GPU host with the package installed.

## See Also
`../README.md` — narrative walkthrough of the same API (Installation, Quick Start, Supported Benchmarks, "Experiments in Paper").
`../src/evogp/` — the library itself; routing table entry `../src/evogp/problem/` for each `Problem`, `../src/evogp/pipeline/` for `StandardPipeline`, `../src/evogp/algorithm/` and `../src/evogp/tree/` for operators and tree/forest types.
`../test/` — ad-hoc verification scripts and notebooks (`fix_bug.py`, `test_bind_success.py`, `*_test.ipynb`), sibling and read-only from here; escalate any write need to the parent.
`../tutorial/` — tutorial notebooks (`evogp_intro.ipynb`, `forest_intro.ipynb`, `combined_gp.ipynb`), sibling and read-only from here.
`../imgs/` — repo-root image assets that `classifier.py` (and the commented-out call in `basic.py`) write into.
