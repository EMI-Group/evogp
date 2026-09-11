# test/ — scratch scripts & demo notebooks (NOT a test suite)

## Intent
Upstream development scratchpad: ad-hoc scripts and Jupyter notebooks used while building/debugging the CUDA kernels and the high-level API.
Files here were written to be watched by a human (print statements, stored notebook outputs), not to be executed by a test runner.
This directory is the closest thing the repo has to an install smoke test, but it is not a regression suite.

## Notes for Agents (read first)
- There is NO test framework here: no `pytest`/`unittest` imports, no `conftest.py`, no fixtures, no parametrization, no CI workflow anywhere in the repo.
- **Zero assertions exist in any file under `test/`** — every script only prints. A green `python test/<file>.py` (exit code 0) means "it ran", not "it is correct".
- Do NOT assume `test/` protects against regressions; do not cite these files as coverage.
- All files are flat with no `__init__.py`; they are executed as standalone scripts (`python test/foo.py`), never imported as a package.
- Everything except `test_torch_env.py` requires a CUDA-visible GPU plus the compiled extension `evogp.evogp_cuda` (built by `setup.py` from `src/evogp/cuda/*.cu`; importing `evogp.tree` triggers `import evogp.evogp_cuda`).
- Notebooks were last run with a Jupyter/IPython kernel named `evogp312` (Python 3.12.8); the stored outputs are stale snapshots from the author's machine.

## API Surface (what each file is)
| File | What it does | Needs GPU? | Assertions? |
|---|---|---|---|
| `test_bind_success.py` | Smoke test that the compiled extension binds: `import evogp.evogp_cuda` then calls the raw `torch.ops.evogp_cuda.{tree_generate,tree_mutate,tree_crossover,tree_evaluate,tree_SR_fitness}` ops with hand-built tensors; prints generated trees via `evogp.tree.utils.str_tree`. | yes | none (prints only) |
| `test_torch_env.py` | Environment preflight for the PyTorch C++ extension toolchain: compiles a tiny CPU-only `TORCH_LIBRARY(extension_cpp)` via `torch.utils.cpp_extension.load_inline` and calls it. | **no** (CPU-only; needs a working C++ compiler) | none (prints only) |
| `fix_bug.py` | Manual repro script: hand-writes a 8-node `evogp.tree.Tree` (XOR, 3 inputs) and prints `tree.SR_fitness(...)` for `execute_mode="hybrid parallel"` vs `"data parallel"`; the two printed values should match. | yes | none (invariant is implicit, by eye) |
| `combined_test.ipynb` | Feature demo of `CombinedForest` + `CombinedDefaultMutation`/`CombinedDefaultCrossover`: full `StandardPipeline` run on XOR (5000 trees, `pow` combination, 100 generations), then `best.A/B.to_sympy_expr()` and `best.to_sympy_expr(lambda A, B: sp.Pow(A, B))`. | yes | none |
| `vis.ipynb` | End-to-end demo/parameter playground: full `Forest` + `StandardPipeline` run on XOR (100000 trees, 1000 generations, `fitness_target=-0.001`), then `best.SR_fitness`, `forward`, `to_sympy_expr`, `to_infix`; `best.to_png("./imgs/xor_tree.png")` is present but commented out (needs the optional `pygraphviz` extra). | yes | none |
| `sympy_test.ipynb` | 3-cell scratch check that nested `sp.Pow` constructs/simplifies as expected when rendering expressions. | no | none |

## How to smoke-test after a build
1. `python test/test_torch_env.py` — cheapest preflight (CPU only); success = it prints `Result:` followed by a 3-element tensor (`a*b+1`). Failure here means the C++ extension build path itself is broken, so `setup.py`'s CUDA build will also fail.
2. `python test/test_bind_success.py` — the real bind check; success = no exception on `import evogp.evogp_cuda` and printed trees + tensors from `crossover_test()`, `evaluate_test()`, `SR_fitnesses_test()`; failure = `ModuleNotFoundError`/dlopen error (extension not built or wrong torch/ABI) or a `torch.ops.evogp_cuda` "no such operator" error (schemas not registered).
3. `python test/fix_bug.py` — success = the two printed fitness values are equal (hybrid-parallel vs data-parallel agreement).
4. `python -m evogp.sr_test` — the validation command actually documented in README "5. Validate Installation" (lives in `src/evogp/sr_test.py`, not here); prefer it when you need the officially supported end-to-end check.
5. Run the notebooks manually from the repo root if you want visual/API confirmation (notebooks reference `./imgs/...` relative to cwd).

## Constraints
- `test_bind_success.py` calls the raw op schemas positionally, mirroring `TORCH_LIBRARY(evogp_cuda, ...)` in `src/evogp/cuda/torch_wrapper.cu`; it uses the internal kernel layer, not the user-facing `Forest`/`Tree` API, so it silently breaks if op signatures change even though the public API still works.
- `test_bind_success.py` hard-codes `device="cuda"` and `torch.uint32` RNG keys (same dtype used by `Forest.random_generate` in `src/evogp/tree/forest.py`), so it cannot run on CPU-only hosts; device strings stay `"cuda"` under ROCm too.
- `test_bind_success.py` defines `mutate_test()` but `main()` never calls it — mutation coverage is only exercised if you invoke it yourself.
- `fix_bug.py` has half of its XOR input/output rows commented out, and validates against a hand-copied node layout; treat it as a repro snippet, not a canonical test vector.

## Known Issues
- No assertion, tolerance, or expected-value check anywhere in this directory — nothing here can fail a CI job on a numeric regression (and there is no CI job).
- Notebooks carry stale stored outputs (e.g. `combined_test.ipynb`'s recorded run did not solve XOR: predictions `[1,1,1,0,0.5,0.5,0,1]` vs expected `[0,1,1,0,1,0,0,1]`); do not read stored outputs as ground truth.
- Notebook runs are long and GPU-heavy (`vis.ipynb`: 100000 trees × 1000 generations); there is no reduced/fast variant.
- `to_png` paths in notebooks need the optional `visualize` extra (`pygraphviz`) plus `networkx`.

## Test Strategy
- Current state: manual smoke testing only (procedure above); no automated tests, no coverage measurement, no CI.
- Coverage gaps (worth filling if tests are ever added): numeric correctness of `tree_generate`/`tree_mutate`/`tree_crossover`/`tree_evaluate`/`tree_SR_fitness` against a CPU reference; agreement across `execute_mode` variants ("data parallel" / "tree parallel" / "hybrid parallel" / "auto"); high-level `Forest`/`Tree`/`CombinedForest` API behavior; `problem`, `algorithm`, `pipeline` layers; determinism/seeding.
- Fastest place to start a real suite: `test_bind_success.py` already encodes valid op-schema call signatures, so wrapping its helpers with assertions is the lowest-friction path.

## See Also / Routing
- `../src/evogp/cuda/` → kernel + `TORCH_LIBRARY` schema registry that this directory's smoke tests exercise (sibling — read-only, escalate writes to parent).
- `../src/evogp/sr_test.py` → README-documented install validation entry point (`python -m evogp.sr_test`) (sibling — read-only, escalate writes to parent).
- `../example/` → maintained end-to-end usage scripts (`basic.py`, `regressor.py`, `classifier.py`, `combined_sr.py`, `custom_sr.py`, `brax_task.py`, `mjx_task.py`, `transformer.py`, `uci_sr.py`, `uci_classifier.py`) serving as the canonical runnable examples (sibling — read-only, escalate writes to parent).
- `../tutorial/` → curated presentation notebooks (`evogp_intro.ipynb`, `forest_intro.ipynb`, `combined_gp.ipynb`) — polished counterparts of the scratch notebooks here (sibling — read-only, escalate writes to parent).
