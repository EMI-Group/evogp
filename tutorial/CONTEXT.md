# tutorial/

## Intent
Hands-on Jupyter notebooks that teach the user-facing EvoGP API: `GenerateDescriptor`, `Forest`/`Tree`, the `GeneticProgramming` operator set, `SymbolicRegression` / `Classification` / `CustomLoss`, `StandardPipeline`, and the multi-output `CombinedForest` / `CombinedTree`.
This is a leaf directory: 3 notebooks plus one generated PNG artifact, no subdirectories and no build step.
All three notebooks are consistent with the current extension-free library (ETL engine, no custom CUDA kernels) and were **executed end-to-end cell by cell** against it; every stored output in the committed JSON comes from that run (see Test Strategy).
They are not linked from the top-level README (which points at `../example/basic.py`), so the directory is discoverable only by browsing.

## Notebooks
| Notebook | Cells (markdown/code) | Learning objective | APIs taught | Measured full-run cost |
|---|---|---|---|---|
| `evogp_intro.ipynb` | 24 (12/12) | End-to-end "getting started": configure tree generation, define problems, assemble/swap genetic operators, run, inspect | `GenerateDescriptor` (full parameter reference in markdown cell 2) → `Forest.random_generate(pop_size, descriptor)` → `descriptor.update(**kwargs)` → `SymbolicRegression(datapoints, labels)` / `Classification(datapoints, labels)` / `SymbolicRegression(func, num_inputs, num_data, lower_bounds, upper_bounds)` → `GeneticProgramming(initial_forest, crossover, mutation, selection)` → `DefaultCrossover`/`LeafBiasedCrossover`, `DefaultMutation(mutation_rate, descriptor)`, `CombinedMutation([...])`, `HoistMutation`, `MultiPointMutation(mutation_rate, descriptor)`, `DefaultSelection(survival_rate, elite_rate)`, `TournamentSelection(5)` → `StandardPipeline(algorithm, problem, generation_limit=10).run()` → `best.forward(problem.datapoints[0])`, `best.to_sympy_expr()`, `best.to_png("best_tree.png")` | ~21 s |
| `forest_intro.ipynb` | 10 (0/10) | Pure data-structure tour of `Tree`/`Forest` (no evolution, no problem) | `GenerateDescriptor(...)` + `Forest.random_generate(pop_size=10, descriptor=descriptor)`, `print(forest)`/`str(forest)`, `forest[i]` → `Tree`, `forest[:3]` → `Forest`, `forest[-1] = tree`, `forest[-3:] = sub_forest`, `forest + tree`, `forest + forest`, `forest[0:0]` → pop size 0, `forest[0:0] + forest[0:3]` | ~3 s |
| `combined_gp.ipynb` | 18 (8/10) | `CombinedTree`/`CombinedForest`: several sub-trees evolved independently and combined by a fixed user-written rule | `GenerateDescriptor`, `CombinedForest.random_generate(pop_size, data_info, descriptors)`, `CombinedDefaultCrossover()`, `CombinedDefaultMutation(mutation_rate, descriptors)`, `DefaultSelection`, `CustomLoss(existing_data, loss_func)` (the rule lives in the loss), `StandardPipeline(...).run()`, `best.to_combined_forest().batch_forward({...})`, `best.A/B/C.to_sympy_expr()`, `formula(A_expr, B_expr, C_expr)` | ~40 s |

Every code cell in all three notebooks was executed successfully; there are no skipped/unexecuted cells and no stored tracebacks.

### Exact call sites (cell index → call, current API)
`evogp_intro.ipynb` — cell 1 imports `torch`, `from evogp.tree import Forest, GenerateDescriptor`, `from evogp.problem import SymbolicRegression, Classification`, `from evogp.algorithm import *`, `from evogp.pipeline import StandardPipeline`.
- cell 0 markdown: engine/device caveat (see Environment Requirements).
- cell 2 markdown documents every `GenerateDescriptor.__init__` parameter and the 8 attributes `__str__` prints; the description matches `descriptor.py` and the printed output in cell 3.
- cell 3: `GenerateDescriptor(max_tree_len=64, input_len=2, output_len=1, using_funcs=["+", "-", "*", "/"], max_layer_cnt=5, const_samples=[-1, 0, 1])`; `print(descriptor)`; `Forest.random_generate(1, descriptor)` (positional `pop_size, descriptor`).
- cell 5: `descriptor.update(using_funcs=["sin", "cos", "tan"])` → a NEW descriptor (the original object keeps its own attributes); `Forest.random_generate(1, new_descriptor)`.
- cell 7: `SymbolicRegression(datapoints=X, labels=y)` with `torch.rand((100, 2), device="cuda")` (overwritten by cell 11, illustrative only).
- cell 9: sklearn `load_iris`; `Classification(datapoints=X, labels=y)` (default `multi_output=True` → `solution_dim=3`). Also illustrative — cell 11 replaces `problem`.
- cell 11: `SymbolicRegression(func=custom_function, num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5)` — the problem the pipeline actually runs (labels are `(1000, 1)`).
- cell 13: default operators, `Forest.random_generate(pop_size=200, descriptor=descriptor)`.
- cell 15: `LeafBiasedCrossover()`, `CombinedMutation([DefaultMutation(...), HoistMutation(mutation_rate=0.2), MultiPointMutation(mutation_rate=0.2, descriptor=descriptor)])`, `TournamentSelection(5)`, pop 200.
- cell 17: `StandardPipeline(algorithm, problem, generation_limit=10)`; `pipeline.run()`.
- cell 19: `best.forward(problem.datapoints[0])` (1-D input → `(1,)` tensor).
- cell 21: `best.to_sympy_expr()`; cell 23: `best.to_png("best_tree.png")`.
- markdown cell 14 lists the star-exported operator variants (`RouletteSelection`, `TruncationSelection`, `RankSelection`, `TournamentSelection`, `DiversityCrossover`, `LeafBiasedCrossover`, `HoistMutation`, `SinglePointMutation`, `MultiPointMutation`, `InsertMutation`, `DeleteMutation`, `SingleConstMutation`, `MultiConstMutation`, `CombinedMutation`) — all exist in `evogp.algorithm` and the signatures used here match.

`forest_intro.ipynb` — cell 0 imports `from evogp.tree import Tree, Forest, GenerateDescriptor` (`Tree` is unused, kept to name the element type) and builds the descriptor with a dict `using_funcs={"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25}` plus `const_prob`, `out_prob`, `layer_leaf_prob`, `const_range=(-1, 1)`, `sample_cnt=8`.
- cells 1–8: the current `Forest`/`Tree` surface (`str`/`print`, `__getitem__` views, `__setitem__`, `__add__`, empty slice, `forest[0:0] + forest[0:3]`). No problem, no pipeline, no fitness: generation only.
- cell 9: empty code cell (no source, no outputs).

`combined_gp.ipynb` — the `data_info`-driven multi-output route.
- markdown cells 0–3, 5, 7: the rule ("formula") is explained as *user-side*; `data_info` maps each sub-tree name to the input names it consumes; `CustomLoss` is named as the problem that carries the rule; the sympy section explains that `CombinedTree` has no `to_sympy_expr`, so the rule is applied to `best.A/B/C.to_sympy_expr()`.
- cell 4: `GenerateDescriptor(max_tree_len=16, input_len=3, output_len=1, using_funcs=["+", "-", "*", "/", "neg"], max_layer_cnt=4, const_samples=[0, 0.5, 1])`.
- cell 6: `CombinedDefaultCrossover()` / `CombinedDefaultMutation(mutation_rate=0.2, descriptors=descriptor)` / `DefaultSelection(survival_rate=0.3, elite_rate=0.01)` (bindings discarded; cell 12 rebuilds them).
- cell 9: imports `torch`, `CombinedForest`, `GenerateDescriptor`, the three combined/default operators, `CustomLoss` (via `from evogp.problem.custom_loss import CustomLoss` — it is NOT re-exported by `evogp.problem`), `StandardPipeline`.
- cell 10: XOR data on cuda; `existing_data = {"x0": XOR_INPUTS[:, 0], "x1": ..., "x2": ..., "y": XOR_OUTPUTS[:, 0]}` (1-D named tensors); `formula = lambda A, B, C: A + B * C`; `loss_func(x0, x1, x2, y, A, B, C)` returning `((formula(A, B, C) - y) ** 2).mean(dim=-1)`; `problem = CustomLoss(existing_data=existing_data, loss_func=loss_func)`. The signature order matters: `CustomLoss` vmapps over the parameters that are not `existing_data` keys, and the first `len(existing_data)` parameters must be those keys.
- cell 11: `descriptor` (as in cell 4) + `data_info = {"A": ["x0","x1","x2"], "B": [...], "C": [...]}` + `CombinedForest.random_generate(pop_size=200, data_info=data_info, descriptors=descriptor)`.
- cell 12: `GeneticProgramming(initial_forest=initial_population, crossover=CombinedDefaultCrossover(), mutation=CombinedDefaultMutation(mutation_rate=0.2, descriptors=descriptor.update(max_layer_cnt=3)), selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01))`.
- cell 13: `StandardPipeline(algorithm, problem, generation_limit=10)`; `pipeline.run()` → a `CombinedTree`.
- cell 14: `best.to_combined_forest().batch_forward({"x0": XOR_INPUTS[:, 0], "x1": ..., "x2": ...})` → `{name: (1, 8, 1)}` (one entry per sub-tree), then `formula(...)` on those entries and `.reshape(-1)`.
- cells 15–17: `best.A.to_sympy_expr()`, `print(best.B/…C.to_sympy_expr())`, and `formula(best.A.to_sympy_expr(), best.B.to_sympy_expr(), best.C.to_sympy_expr())`.
- `best.A`/`best.B`/`best.C` exist because `CombinedTree.__init__` `setattr`s every `data_info` key onto the instance.

## Environment Requirements
- Interpreter: **`/mnt/local-ssd/bchuang/evox/.venv/bin/python`** — there is no `python`/`python3` with torch on `PATH` (the system `/usr/bin/python3` has neither torch nor evogp). That venv has torch 2.6.0+cu124, sympy, networkx, matplotlib, ETL/IREE, and CUDA access (8× RTX A6000); it does **not** have jupyter/nbconvert/nbformat/nbclient/ipykernel, scikit-learn or pygraphviz.
- `evogp` is not installed in that venv (it is a src-layout checkout): every run needs `PYTHONPATH=<repo>/src` (or an editable install).
- **Run from this directory.** `evogp_intro.ipynb` cell 23 calls `best.to_png("best_tree.png")`, a path relative to the notebook's cwd, so executing the notebook anywhere else drops the PNG somewhere else.
- All three notebooks hard-code `device="cuda"` (`GenerateDescriptor`, data tensors) — a CUDA-capable GPU is required. The tree evaluation itself is not CUDA-dependent any more (ETL numpy backend by default).
- Optional/extra imports: `scikit-learn` (only `evogp_intro.ipynb` cell 9; it is a declared evogp dependency) and `pygraphviz` (only `evogp_intro.ipynb` cell 23, the `visualize` extra). In this sandbox both were installed into a scratch `--target` dir and exposed through `PYTHONPATH`; without pygraphviz the PNG cell raises `ImportError` — skip/clear it if you cannot provide it (it produces no cell output either way).
- Recommended speed env var: `ETL_DISABLE_LOCATIONS=1` (tracing is ~4× faster; the engine's own docs recommend it for production runs).
- Recorded `metadata.kernelspec` values are the authors' local environments (`evogp312`, `torch_env`) on Python 3.12; only `language: python` matters — do not read them as a requirement.

## Constraints
- The notebooks must follow `../src/evogp`, never the reverse: treat `src/evogp` signatures as the source of truth. Do not modify the library to make a notebook runnable — file the issue instead.
- No seeds are set anywhere: tree structures, expressions, fitness logs and `best_tree.png` are random samples, not golden values. Re-running a notebook rewrites its stored outputs, the execution counters and the PNG.
- Exercise notebooks only on a GPU machine; generation/mutation/`check_tensor` hard-code `"cuda"`.
- Stored outputs are part of the deliverable: a reader without a GPU should be able to study the results, so keep them consistent with the code (re-execute after editing a cell instead of hand-editing outputs).
- Notebook JSON is `nbformat 4.2` with `indent=1`, `ensure_ascii=False` and a trailing newline (what Jupyter and the run recipes below produce); keep that format if you write JSON programmatically.
- Training settings in the bodies are deliberately small (intro pop 200 × 10 generations, combined pop 200 × 10 generations; `forest_intro` generates only) so every notebook finishes in seconds-to-tens-of-seconds. They are pedagogical demos, not benchmarks — the paper-scale settings live in `../example/`.

## Known Issues
1. **`combined_gp.ipynb` cell 14 works around a broken library method.** `CombinedTree.forward(x: dict)` is unusable for a batch: with 1-D named inputs it selects the non-batched branch and dies in `Forest.forward` (`assert x.shape == (pop_size=1, input_len)`), and with 2-D named inputs it selects `batch_forward`, whose `x[name][:, None]` makes 3-D tensors that fail the same interior assert. That is why the notebook calls `best.to_combined_forest().batch_forward({... 1-D ...})` directly. Fix belongs in `../src/evogp/tree/combined_tree.py`.
2. **`SymbolicRegression` cannot score a `CombinedForest`** (it raises `NotImplementedError` by design), which is why the combined notebook uses `CustomLoss` instead. `CustomLoss` is not re-exported by `evogp/problem/__init__.py`, so the import must be `from evogp.problem.custom_loss import CustomLoss`.
3. **`CombinedTree` exposes neither `SR_fitness` nor `to_sympy_expr`** (it does not know the user's rule); the notebook applies the rule itself. `CombinedForest.__next__` also never advances `iter_index` (endless iteration), so avoid `for x in combined_forest`.
4. **Cosmetic: `Tree.__repr__`** (and therefore `print(forest)` / the `exec` display) prints `np.float32(0.25018632)` for constant-only trees, because `to_sympy_expr()` returns the raw numpy scalar in that case. Visible in `forest_intro.ipynb` cells 1–8 and `evogp_intro.ipynb` cell 5.
5. **`evogp_intro.ipynb` cells 7 and 9 are illustrative only**: the iris `Classification` problem is 4-input/3-output while the tutorial descriptor is 2-input/1-output, and both cells are immediately overwritten by the custom-function `SymbolicRegression` in cell 11; no pipeline run validates them.
6. **`evogp_intro.ipynb` relies on random generation quality**: with pop 200 × 10 generations the symbolic-regression demo shows a clear but partial fit (fitness still improving), and cell 5's unary-only descriptor occasionally generates a trivial constant tree (see issue 4).
7. `best_tree.png` (90 KB, RGBA) is a checked-in run artifact of `evogp_intro.ipynb` cell 23, not a documentation figure: nothing references it (`README.md` uses `../imgs/*.png`), and it is overwritten on every run of that notebook.

## Test Strategy
There is no automated coverage of this directory and no CI. Verification is manual on a CUDA machine and compares semantics (calls succeed, expressions/plots have the expected shape), never literal values.

**Preferred (environment with Jupyter installed):**
```
cd tutorial
PYTHONPATH=<repo>/src ETL_DISABLE_LOCATIONS=1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/jupyter nbconvert --to notebook --execute --inplace evogp_intro.ipynb
```
`jupyter`/`nbconvert` are **not** installed in the project venv, so this needs `pip install jupyter nbconvert ipykernel` first.

**Recipe used for the committed outputs (works with the venv alone):** load the notebook JSON with the stdlib `json` module, `exec` its code cells in order in a single shared namespace while capturing stdout into `stream` outputs, emit every top-level expression whose value is not `None` as an `execute_result` (IPython's displayhook), stop at the first exception recording an `error` output, and dump the JSON back with `indent=1, ensure_ascii=False`. Run it from this directory, e.g.
```
PYTHONPATH=<repo>/src ETL_DISABLE_LOCATIONS=1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python <runner>.py evogp_intro.ipynb --write
```
A cheap pre-check that the library itself imports (a frequent first failure) is `PYTHONPATH=<repo>/src /mnt/local-ssd/bchuang/evox/.venv/bin/python -c "import evogp.tree"`; the library's own gates live in `../src/evogp/tests/` (pytest) and `../src/evogp/sr_test.py`.

**Verified state of this directory (all three notebooks, single run each):** every code cell executes, all cells carry fresh outputs/counters, `evogp_intro.ipynb` reproduces the documented narrative (fitness max improves from ≈ −443 to ≈ −17 MSE on `(x0+x1)^2`, a partly correct expression, one prediction printed, PNG written), `combined_gp.ipynb` converges on XOR (max fitness ≈ −0.13, predictions near {0, 1}, `A=0.5`, `B=x0+x1+x2+1`, `C` a product of three factors) and `forest_intro.ipynb` exercises every `Forest` dunder on a 10-tree population. The PNG in this directory was regenerated by that run.

**Coverage gaps:** nothing asserts the notebook outputs; there is no CPU path; the sklearn/pygraphviz cells were validated with scratch-installed packages, not with the plain venv.

## Routing Table / See Also
This node has no child directories; all work belongs in the three notebooks here.
- `../src/evogp/tree/` → authoritative `Tree`/`Forest`/`CombinedTree`/`CombinedForest`/`GenerateDescriptor` signatures and engine semantics (`CONTEXT.md`, `ENGINE_SPEC.md`) (sibling — read-only, escalate writes to parent)
- `../src/evogp/problem/` → `SymbolicRegression`, `Classification`, `CustomLoss` (not exported), `Transformation` (sibling — read-only)
- `../src/evogp/algorithm/` → the selection/crossover/mutation operators referenced by `evogp_intro.ipynb` and `combined_gp.ipynb` (sibling — read-only)
- `../src/evogp/pipeline/` → `StandardPipeline` run loop and the `Generation: N, Cost time: ...ms` log format the notebooks capture (sibling — read-only)
- `../example/` → runnable scripts; `basic.py` is the README-linked canonical example, `combined_sr.py` still uses the removed `formula=`/`share_input=` API (sibling — read-only)
- `../test/` → ad-hoc scripts and notebooks, including `combined_test.ipynb` (same stale combined API) and `vis.ipynb` (`to_png`/`to_infix` playground) (sibling — read-only)
- `../README.md` → Basic API Usage section, the only end-to-end doc linked from the docs (sibling — read-only)
