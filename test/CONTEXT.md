# test/ — scratch scripts & demo notebooks (NOT a test suite)

## Intent
Upstream development scratchpad: ad-hoc scripts and Jupyter notebooks used while building/debugging the CUDA kernels and the high-level API.
Files here were written to be watched by a human (print statements, stored notebook outputs), not to be executed by a test runner.
This directory is the closest thing the repo has to an install smoke test, but it is not a regression suite.

## Notes for Agents (read first)
- There is NO test framework here: no `pytest`/`unittest` imports, no `conftest.py`, no fixtures, no parametrization, no CI workflow anywhere in the repo.
- There are no Python `assert` statements anywhere under `test/`; every script only prints (the only runtime checks are the `TORCH_CHECK`/`TORCH_INTERNAL_ASSERT` macros inside the inline C++ of `test_torch_env.py`).
- A green `python test/<file>.py` (exit code 0) means "it ran", not "it is correct"; do not cite these files as regression coverage.
- All files are flat with no `__init__.py`; they are executed as standalone scripts (`python test/foo.py`), never imported as a package; `test_bind_success.py` has a `__main__` guard that calls `main()`, `fix_bug.py` runs at import.
- Only `test_torch_env.py` and `sympy_test.ipynb` do not touch evogp; every other file needs a CUDA-visible GPU plus the compiled extension `evogp.evogp_cuda` (built by `setup.py` from `src/evogp/cuda/*.cu`; importing `evogp.tree` pulls it in, and `test_bind_success.py` imports `evogp.evogp_cuda` directly on line 2).
- Notebooks were last run with a Jupyter/IPython kernel named `evogp312` (Python 3.12.8); stored outputs are stale snapshots from the author's machine, and `vis.ipynb`'s stored outputs mix several sessions.

## API Surface (what each file is)
| File | What it does | Needs GPU? | Assertions? |
|---|---|---|---|
| `test_bind_success.py` | Op-level bind smoke test: imports `evogp.evogp_cuda` (line 2), then wraps all five raw ops `torch.ops.evogp_cuda.{tree_generate,tree_mutate,tree_crossover,tree_evaluate,tree_SR_fitness}` (helpers lines 15-140) around hand-built tensors; `main()` (lines 221-230) runs generate + `crossover_test` + `evaluate_test` + `SR_fitnesses_test`, printing raw tensors and trees via `evogp.tree.utils.str_tree`. | yes | none (prints only) |
| `test_torch_env.py` | Environment preflight for the PyTorch C++ extension toolchain: compiles a tiny CPU-only `TORCH_LIBRARY(extension_cpp)` via `torch.utils.cpp_extension.load_inline` and calls `torch.ops.extension_cpp.mymuladd(a, b, 1.0)`; does not import evogp and never touches CUDA (host C++ compiler + torch headers only). | **no** | none (prints only) |
| `fix_bug.py` | Manual repro: hand-writes an 8-slot `evogp.tree.Tree(input_len=3, output_len=1)` (`node_type` `[3,3,0,0,3,0,0,0]`, `node_value` `[3.,2.,0.,2.,2.,0.,2.,0.]`, `subtree_size` `[7,3,1,1,3,1,1,0]`; preorder decode `(x[0] - x[2]) * (x[0] - x[2])`) and prints `tree.SR_fitness(XOR_INPUTS, XOR_OUTPUTS, execute_mode=...)` for `"hybrid parallel"` (line 43) then `"data parallel"` (line 46); the two values should match. | yes | none (invariant is implicit, by eye) |
| `combined_test.ipynb` | **Cannot run as shipped** — `CombinedForest` demo on XOR (5000 trees, `pow` combination, 100 generations) written against the pre-`data_info` API (`CombinedForest.random_generate(formula=..., share_input=...)`); see Known Issues for the exact failure points. | yes | none |
| `vis.ipynb` | End-to-end demo/parameter playground on the current API: full `Forest` + `StandardPipeline` run on XOR (100000 trees, 1000-generation cap, `fitness_target=-0.001`), then `best.SR_fitness`, `forward`, `to_sympy_expr`, `sympy_expression.simplify()`, `to_infix`; `best.to_png("./imgs/xor_tree.png")` is present but commented out (needs the optional `pygraphviz` extra). | yes | none |
| `sympy_test.ipynb` | 3-cell scratch check that nested `sp.Pow` constructs and renders as expected (it never calls `.simplify()`); pure sympy, no evogp import. | no | none |
| `etl_jit_probe.py` | ETL "true jit" economics probe (standalone script, `argparse`). Builds a realistic GP step — `@etl.defn _step(x, tree)` whose `tree` static is a nested tuple decoded at trace time into add/sub/mul/div + unary sin/exp/neg + a `select`-based protected divide — through `etl.build_cached` + `etl.CompileCache` on `TensorSpec((N,2), float32)`, and prints one table with per-tree build/run timings, generation-scale extrapolations, parallel-build results, persistence round trip and API-semantics checks. Does NOT import evogp. | yes (iree-cuda sections) | none (asserts are in-script only; prints `PASS`/`FAIL` and exits 1 if any section fails) |

## How to smoke-test after a build
1. `python test/test_torch_env.py` — cheapest preflight (CPU only); success = it prints `Input a:`, `Input b:` and `Result:` with a 3-element tensor (`a*b+1`). Failure here means the host C++ extension build path is broken, so `setup.py`'s extension build will also fail; success does not prove a CUDA toolchain exists.
2. `python test/test_bind_success.py` — the repo's only op-level bind check; success = no exception on `import evogp.evogp_cuda` and printed trees + tensors from `crossover_test()`, `evaluate_test()`, `SR_fitnesses_test()`; failure = `ModuleNotFoundError`/dlopen error (extension not built or wrong torch/ABI) or a `torch.ops.evogp_cuda` "no such operator" error (schemas not registered).
3. `python test/fix_bug.py` — success = the two printed fitness values are equal (hybrid-parallel vs data-parallel agreement).
4. `python -m evogp.sr_test` — the validation command actually documented in README "5. Validate Installation" (lives in `src/evogp/sr_test.py`, not here); prefer it when you need the officially supported end-to-end check.
5. Run the notebooks manually from the repo root if you want visual/API confirmation (`vis.ipynb` references `./imgs/...` relative to cwd, commented out); `combined_test.ipynb` cannot run as shipped.
6. `etl_jit_probe.py` — the only file here that exercises ETL (the foreign repo `/mnt/local-ssd/bchuang/etl`) instead of evogp, so it needs no kernel build. Exact command (the venv is required; bare `python` is not on PATH):
   `/mnt/local-ssd/bchuang/evox/.venv/bin/python test/etl_jit_probe.py` (add `--device N` for a CUDA index, `--sections a,b,f` to run a subset, `--reps-build`/`--reps-run`/`--gen-trees`/`--par-trees` to resize).
   Runtime is ~3-5 min for all sections; results print to stdout as sections (a)-(h) (each row as it is measured, plus a final `SUMMARY TABLE`), scratch files land in `$TMPDIR/etl_jit_probe/run_*/`, and the exit code is 0 unless a section or a `PASS`/`FAIL` check failed. `--device 0` is the default; if that GPU is busy (the host is shared — the header prints `nvidia-smi` utilization) the timings inflate several-fold, so prefer an idle device for quotable numbers.

## Constraints
- `test_bind_success.py` is the only file under `test/` that binds to the raw op surface; it calls the schemas positionally, mirroring `TORCH_LIBRARY(evogp_cuda, ...)` (`src/evogp/cuda/torch_wrapper.cu:291-298`), so it breaks silently if op signatures change even when the public API still works.
  - `tree_SR_fitness`'s trailing int is the kernel selector (`0` in the test — the same code `Forest.SR_fitness` uses for `"hybrid parallel"`; `Tree.SR_fitness` instead maps `"hybrid parallel"`→3 and both use 1/2/4 for data parallel/tree parallel/auto), and `tree_evaluate`'s `variables` tensor has shape `(pop_size, var_len)`.
  - `main()` never calls `mutate_test()` (defined lines 143-161), so mutation is exercised only if you invoke it yourself.
  - Module-level constants (lines 6-12) are shadowed by the helpers' defaults and are effectively unused.
- `test_bind_success.py` hard-codes `device="cuda"` and `torch.uint32` RNG keys (same dtype used by `Forest.random_generate` in `src/evogp/tree/forest.py`), so it cannot run on CPU-only hosts; device strings stay `"cuda"` under ROCm too.
- `fix_bug.py` uses only the public API (`evogp.tree.Tree` + hand-built tensors), so it is the one file here that survives removal of the kernels; but 4 of its 8 XOR rows are commented out, its `import evogp` is unused, and its tree does not fit even the active XOR rows — treat it as a repro/equivalence snippet, not a canonical test vector.

## Known Issues
- No assertion, tolerance, or expected-value check anywhere in this directory — nothing here can fail a CI job on a numeric regression (and there is no CI job).
- `combined_test.ipynb` uses the pre-`data_info` combined API and cannot run as shipped: cell 1 calls `CombinedForest.random_generate(formula=..., share_input=True)` (current signature is `random_generate(pop_size, data_info, descriptors)` → `TypeError`), and a fixed call would still hit `SymbolicRegression.evaluate` reading the removed `forest.share_input` (`src/evogp/problem/symbolic_regression.py:73`), a tensor passed to the dict-expecting `CombinedTree.forward`, and missing `CombinedTree.to_sympy_expr` for the `best.A/B.to_sympy_expr()` / `best.to_sympy_expr(lambda A, B: sp.Pow(A, B))` cells.
- Stored notebook outputs are not ground truth: `combined_test.ipynb`'s recorded (old-API) run did not solve XOR (`[1,1,1,0,0.5,0.5,0,1]` vs expected `[0,1,1,0,1,0,0,1]`), and `vis.ipynb`'s cells carry outputs from several sessions (execution counts 25/26 vs 13/14 show different expressions for `to_sympy_expr` vs `simplify`/`to_infix`).
- Notebook runs are long and GPU-heavy (`vis.ipynb`: 100000 trees, 1000-generation cap; its stored run stopped early at generation 35 via the fitness target); there is no reduced/fast variant.
- No golden/reference numeric value is recorded anywhere in this directory; the only expected-value invariant is the implicit hybrid-vs-data-parallel equality in `fix_bug.py`.
- `vis.ipynb` is the only entry point in the repo that exercises `StandardPipeline(fitness_target=...)` early stopping, and one of the few that exercises `log`/`sqrt`/`pow`/`inv` through `using_funcs`.
- The (commented) `to_png` call in `vis.ipynb` needs the optional `visualize` extra (`pygraphviz`).

## Test Strategy
- Current state: manual smoke testing only (procedure above); no automated tests, no coverage measurement, no CI.
- Coverage gaps: no numeric check of `tree_generate`/`tree_mutate`/`tree_crossover`/`tree_evaluate`/`tree_SR_fitness` against a reference; no agreement check across `execute_mode` variants ("torch" / "data parallel" / "tree parallel" / "hybrid parallel" / "auto"); no behavioral checks of `Forest`/`Tree`/`CombinedForest`, `problem`, `algorithm`, `pipeline`, or determinism/seeding.
- Lowest-friction path to a real suite: turn `fix_bug.py`'s public-API `Tree` case into an assertion-based equivalence test and add assertion-based smoke tests around `Forest.random_generate`/`forward`/`SR_fitness`; a reduced version of `vis.ipynb` (tiny pop/generation count) is the natural end-to-end test.
- If the kernels are replaced: no golden numeric values exist here, so capture expected outputs from the current implementation first (e.g. `fix_bug.py`'s Tree under each `execute_mode`) and assert the replacement matches them.

## See Also / Routing
- `../src/evogp/cuda/` → kernel + `TORCH_LIBRARY` schema registry that `test_bind_success.py` exercises (sibling — read-only, escalate writes to parent).
- `../src/evogp/tree/` → current `Tree`/`Forest`/`CombinedForest` (`data_info`) contracts behind `fix_bug.py` and the notebooks (sibling — read-only, escalate writes to parent).
- `../src/evogp/sr_test.py` → README-documented install validation entry point (`python -m evogp.sr_test`) (sibling — read-only, escalate writes to parent).
- `../example/` → maintained end-to-end usage scripts (`basic.py`, `regressor.py`, `classifier.py`, `combined_sr.py`, `custom_sr.py`, `brax_task.py`, `mjx_task.py`, `transformer.py`, `uci_sr.py`, `uci_classifier.py`) serving as the canonical runnable examples (sibling — read-only, escalate writes to parent).
- `../tutorial/` → curated presentation notebooks (`evogp_intro.ipynb`, `forest_intro.ipynb`, `combined_gp.ipynb`) — polished counterparts of the scratch notebooks here (sibling — read-only, escalate writes to parent).
