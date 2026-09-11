# tutorial/

## Intent
Hands-on Jupyter notebooks that teach the user-facing EvoGP API: `GenerateDescriptor`, `Forest`/`Tree`, the `GeneticProgramming` operator set, `SymbolicRegression`/`Classification`, `StandardPipeline`, and `CombinedForest`.
This is a leaf directory: 3 notebooks plus one binary output asset, no subdirectories and no build step.
The notebooks are the closest thing to a narrative walkthrough, but they are **not** automated and **not** linked from the top-level README (the README's Basic API Usage section points at `../example/basic.py`), so the directory is currently undiscoverable from the docs.

## API Surface (notebook → what it teaches)
| Notebook | Learning objective | evogp APIs taught (in order) | Cost of a full run |
|---|---|---|---|
| `evogp_intro.ipynb` | End-to-end "getting started": configure tree generation, define a problem, assemble/swap genetic operators, run, inspect | `GenerateDescriptor` (full parameter reference in markdown) → `Forest.random_generate(pop_size, descriptor)` → `descriptor.update(**kwargs)` → `SymbolicRegression(datapoints, labels)` / `Classification(datapoints, labels)` / `SymbolicRegression(func, num_inputs, num_data, lower_bounds, upper_bounds)` → `GeneticProgramming(initial_forest, crossover, mutation, selection)` → `DefaultCrossover/LeafBiasedCrossover`, `DefaultMutation(mutation_rate, descriptor)`, `CombinedMutation([...])`, `HoistMutation`, `MultiPointMutation(mutation_rate, descriptor)`, `DefaultSelection(survival_rate, elite_rate)`, `TournamentSelection(5)` → `StandardPipeline(algorithm, problem, generation_limit=50).run()` → `best.forward(...)`, `best.to_sympy_expr()`, `best.to_png("best_tree.png")` | pop 1000 × 50 gens × 1000 datapoints; stored output shows 7–40 ms/gen plus ~0.3 s for generation 0 → a few seconds of GPU |
| `forest_intro.ipynb` | Pure data-structure tour of `Tree`/`Forest` (no evolution) | `Forest.random_generate(...)`, `print(forest)`/`str(forest)`, `forest[i]` → `Tree`, `forest[:3]` → `Forest`, `forest[-1] = tree`, `forest[-3:] = sub_forest`, `forest + tree`, `forest + forest`, `forest[0:0]` → pop size 0 | instant (generation only) |
| `combined_gp.ipynb` | `CombinedTree`/`CombinedForest`: a manually specified top-level formula split into independently evolved sub-trees | `CombinedForest.random_generate(...)`, `GenerateDescriptor`, `CombinedDefaultCrossover()`, `CombinedDefaultMutation(mutation_rate, descriptors)`, `DefaultSelection`, `SymbolicRegression(datapoints, labels, execute_mode="torch")`, `StandardPipeline(generation_limit=100)`, `best.forward(XOR_INPUTS)`, `best.A/B/C.to_sympy_expr()`, `best.to_sympy_expr(sympy_formula)` | pop 5000 × 100 gens × 8 datapoints (XOR); ~155 ms gen 0 then ~5–6 ms/gen → a few seconds of GPU |

`best_tree.png` is a 463×377 RGBA PNG (34.5 KB) produced by `evogp_intro.ipynb` cell 23 (`best.to_png("best_tree.png")`, a path relative to the notebook's cwd).
It is a checked-in output artifact, not a documentation figure: no notebook, the README, `MANIFEST.in` and `.gitignore` reference it, and it is byte-wise different from the README's `../imgs/{sr_tree,xor_tree,classifier_tree}.png`.
It is overwritten (and produces a noisy git diff) every time that cell is re-run, since no seed is fixed.

## Environment Requirements
- All three notebooks `import evogp`, so the compiled extension `evogp.evogp_cuda` must exist (`pip install -e .` or `python setup.py build_ext --inplace`) and a CUDA-capable GPU must be present.
- There is no CPU path: `device="cuda"` is hard-coded in the notebooks, in `GenerateDescriptor` (const samples / depth probabilities / roulette vectors are built on `cuda:0`), and in `Forest.random_generate` (`torch.randint(..., device="cuda")`). Merely constructing a `GenerateDescriptor` requires a GPU.
- `evogp_intro.ipynb` additionally needs `scikit_learn` (the iris `Classification` cell) and `sympy` (`to_sympy_expr`), plus `networkx` + `pygraphviz` for the final `to_png` cell (`pygraphviz` is the optional `visualize` extra and is commented out in `../requirements.txt`).
- `combined_gp.ipynb` needs `sympy` for the sympy-formula discussion (the current cell 17 output is sympy's rich `text/latex` repr).
- `forest_intro.ipynb` needs only `torch` + evogp.
- Recorded kernelspecs are the authors' local environments (`evogp312` for `evogp_intro`, `torch_env` for the other two) on Python 3.12; only `language: python3` matters.
- No notebook sets `torch.manual_seed`, `numpy.random.seed` or any other seed, so all stored outputs (trees, expressions, fitness traces, `best_tree.png`) are non-reproducible samples, not golden values.

## Constraints
- The notebooks must follow `../src/evogp`, never the reverse: treat `src/evogp` signatures as the source of truth when editing tutorial code.
- Do not modify source code to make a stale notebook run; fix the notebook or file a Known Issue.
- `evogp_intro.ipynb` is the only notebook currently consistent with the HEAD API; use it as the style/API reference when repairing the other two.
- Executing a notebook rewrites the committed JSON (including the multi-KB fitness-log outputs) and `best_tree.png`; prefer `--ClearOutputPreprocessor`-free targeted edits, or expect large diffs.
- Cell outputs are stored, so a reader can study results without a GPU, but only `evogp_intro.ipynb` has a self-consistent executed state (execution counters 1..12 with outputs).

## Known Issues
1. **`forest_intro.ipynb` cell 0 uses the pre-`GenerateDescriptor` API (stale).** It calls `Forest.random_generate(pop_size=10, gp_len=128, input_len=3, output_len=1, max_layer_cnt=5, **{"const_prob":..., "out_prob":..., "func_prob":..., "layer_leaf_prob":..., "const_range":..., "sample_cnt":...})`. HEAD's signature is `Forest.random_generate(pop_size: int, descriptor: GenerateDescriptor)` — `gp_len` is now `GenerateDescriptor.max_tree_len`, `func_prob` is now `using_funcs` (dict or list), `const_range`/`sample_cnt` are now `const_samples`, and `input_len`/`output_len`/`max_layer_cnt`/`layer_leaf_prob`/`const_prob`/`out_prob` moved onto the descriptor. The cell raises `TypeError: random_generate() got an unexpected keyword argument 'gp_len'` before anything else runs.
2. **`forest_intro.ipynb` is in a half-executed state.** Cell 0 has `execution_count: null` and no outputs, while cells 1–8 still carry outputs from an older run (execution counters 3..10), i.e. it was edited but never re-run after the API change. Cells 1–8 themselves (`str`, `__getitem__`, `__setitem__`, `__add__`, empty-slice behavior) still match HEAD.
3. **`combined_gp.ipynb` cells 10–17 use the removed `formula`/`share_input` CombinedForest API (stale).** Cell 11 calls `CombinedForest.random_generate(pop_size=5000, formula=lambda A, B, C: A + B * C, descriptors=descriptor, share_input=True)`. HEAD's signature is `CombinedForest.random_generate(pop_size: int, data_info: dict, descriptors: Union[List, GenerateDescriptor])`, where `data_info` maps each output name to the list of input names feeding it; it asserts `len(descriptors) == len(data_info)`, `descriptors[i].input_len == len(data_info[key])` and `descriptors[i].output_len == 1`. The top-level formula concept moved out of the tree classes entirely (see `../src/evogp/problem/custom_loss.py` for the current composition pattern).
4. **`combined_gp.ipynb` cell 17 calls a method that no longer exists.** `best.to_sympy_expr(lambda A, B, C: A + B * C)` — `CombinedTree.to_sympy_expr` was removed; `../src/evogp/tree/combined_tree.py` now exposes only `random_generate`, `forward`, `to_combined_forest` and the per-output attribute aliases. Cells 15/16 (`best.A/B/C.to_sympy_expr()`) still work because `CombinedTree.__init__` `setattr`s every `data_info` key onto the instance — but only if the key is literally named `A`/`B`/`C`. Cell 14 `best.forward(XOR_INPUTS)` is also stale: HEAD's `CombinedTree.forward` takes a `dict[str, torch.Tensor]` of named inputs and returns a dict of named outputs.
5. **A repaired `combined_gp.ipynb` still cannot run end-to-end at HEAD.** `SymbolicRegression.evaluate` reads `forest.share_input` when the forest is a `CombinedForest`, but `CombinedForest` no longer defines `share_input` (nothing in `src/` sets it) → `AttributeError`. Cell 10's `SymbolicRegression(..., execute_mode="torch")` on a `CombinedForest` therefore fails regardless of notebook edits. The fix belongs in `../src/evogp/problem/symbolic_regression.py`, outside this node.
6. **Minor consistency smell in `evogp_intro.ipynb`:** the iris `Classification` cell builds a 4-input problem while the tutorial's descriptor is `input_len=2`, and the cell's `problem` is immediately overwritten by the custom-function `SymbolicRegression` in the next section, so it is illustrative only and never validated by a pipeline run.
7. **Same stale pattern in sibling directories:** `../example/combined_sr.py` and `../test/combined_test.ipynb` contain the identical removed `formula=`/`share_input=`/`to_sympy_expr(formula)` calls, so do not use them as the reference for repairing `combined_gp.ipynb`.

## Test Strategy
There is no automated coverage of this directory: `../test/` only contains `test_bind_success.py`, `test_torch_env.py`, a few ad-hoc scripts and other notebooks, and no CI job executes the tutorials.
Verification is manual on a CUDA machine, e.g. `jupyter nbconvert --to notebook --execute --inplace <notebook>` (or running the cells by hand) and comparing semantics — API calls succeed and the expression/plots have the expected shape — never the literal values, which are random.
Cheap smoke check before touching a notebook: `python -m pytest ../test/test_bind_success.py` confirms the CUDA extension is importable, which is the usual first failure mode (`ModuleNotFoundError: No module named 'evogp.evogp_cuda'`).
When repairing a notebook, keep the existing markdown structure and correct the code cells + re-execute, so the stored outputs stay consistent with the narrative.

## Routing Table / See Also
This node has no child directories; all work belongs in the three notebooks here.
- `../src/evogp/tree/` → authoritative `Tree`/`Forest`/`CombinedTree`/`CombinedForest`/`GenerateDescriptor` signatures (sibling — read-only, escalate writes to parent)
- `../src/evogp/algorithm/` → selection/crossover/mutation operators referenced by `evogp_intro.ipynb` (sibling — read-only)
- `../src/evogp/problem/` → `SymbolicRegression`, `Classification`, `CustomLoss` (sibling — read-only)
- `../src/evogp/pipeline/` → `StandardPipeline` run loop and the generation-log format the notebooks capture (sibling — read-only)
- `../example/` → runnable scripts; `basic.py` is the README-linked canonical example, `combined_sr.py` is stale like issue 7 (sibling — read-only)
- `../test/combined_test.ipynb`, `../test/vis.ipynb` → in-repo notebooks probing CombinedForest and `to_png`; also stale for the CombinedForest API (sibling — read-only)
- `../README.md` → Basic API Usage section, the only current end-to-end doc linked from the docs (sibling — read-only)
