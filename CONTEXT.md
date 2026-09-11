# EvoGP — GPU-accelerated Tree-based Genetic Programming (repository root)
## Intent
EvoGP is a fully GPU-accelerated Tree-based Genetic Programming (TGP) framework built on PyTorch, using custom CUDA/HIP kernels for tree generation, crossover, mutation and fitness evaluation.
It targets symbolic regression, classification, feature transformation and robotics-control (Brax / MuJoCo-Playground / Genesis) benchmarks, with multi-output trees for classification and policy optimization.
It is a sister project of [EvoX](https://github.com/EMI-Group/evox) and is described in the paper at https://arxiv.org/abs/2501.17168.
This repository is the development root (source tree + examples + paper experiment scripts + notebooks), not a docs site.
The repository is ORIGINAL upstream code from EMI-Group; see `git log` for its public history.

## Repository structure & routing table
- `src/evogp/` → The entire library: `tree/` (data model, foundation layer), `algorithm/` (GP loop + crossover/mutation/selection), `problem/` (benchmarks & task layers), `pipeline/` (training loop), `cuda/` (kernel sources), plus the `sr_test.py` install-validation script. Start here.
- `src/` → Thin src-layout wrapper; contains only `evogp/`. See `src/CONTEXT.md`.
- `example/` → 10 runnable end-to-end scripts showing how to wire the public API; four of them are the paper's experiment scripts (`uci_sr.py`, `custom_sr.py`, `uci_classifier.py`, `brax_task.py`). See `example/CONTEXT.md`.
- `tutorial/` → 3 Jupyter notebooks (`evogp_intro.ipynb`, `forest_intro.ipynb`, `combined_gp.ipynb`) + `best_tree.png`; narrative walkthrough of the API, not automated. See `tutorial/CONTEXT.md`.
- `test/` → ad-hoc scripts and notebooks for manual verification; NOT a test suite (no pytest/unittest, no assertions, no CI). See `test/CONTEXT.md`.
- `imgs/` → static assets only: README figures (`SymbolicRegression.svg`, `Classification.svg`, `Halfcheetah.svg`, EvoX brand/logo files) plus the committed outputs `classifier_tree.png`, `sr_tree.png`, `xor_tree.png`. No code; do not look here for logic.

## Root files (build & packaging metadata)
- `setup.py` — the build entry point. Declares `name="evogp"`, `version="0.1.0"`, `packages=find_packages(where="src")` + `package_dir={"": "src"}` (hence the `evogp` import root), and one `CUDAExtension` named `evogp.evogp_cuda` whose sources are exactly the four `.cu` files under `src/evogp/cuda/` (listed by hand — a new `.cu` must be added here, there is no glob). Device compile flags are chosen per backend: a long nvcc flag list for CUDA, only `-O3` under `torch.version.hip is not None` (ROCm/hipcc rejects the nvcc/ptxas-only flags). `install_requires = torch, numpy, scikit_learn, networkx, sympy`; `extras_require = {"visualize": ["pygraphviz"], "brax": ["jax", "brax"]}`; `python_requires >= 3.9`. `MAX_JOBS` is set to the CPU count.
- `pyproject.toml` — build-system only (`setuptools`, `wheel`, `torch`; backend `setuptools.build_meta`). No tool config, no lint/format/test config anywhere in the repo.
- `requirements.txt` — mirrors `install_requires` (`torch`, `scikit_learn`, `sympy`, `numpy`, `networkx`); `pygraphviz` and `jax`/`brax` are listed only as comments.
- `MANIFEST.in` — ships `README.md` and `recursive-include src/evogp/cuda *.cu *.h` (the kernel sources must stay in the sdist/wheel).
- `README.md` — installation (CUDA and ROCm), validation, and the canonical API walkthrough; it is the primary user-facing doc and is used as the `long_description`.
- `LICENSE` — Apache 2.0. `.gitignore` — build artifacts, `__pycache__`, `*.so`, ninja logs, and the generated `src/evogp/hip/` hipify mirror, plus `experiment/*`, `example/new_exp.py`, `src/evogp/cuda/forward_new.cu`.

## Global invariants & constraints
- A successful extension build is a hard prerequisite: `src/evogp/tree/__init__.py` does `import evogp.evogp_cuda`, so nothing in the library is importable until that shared library exists. Build with `pip install . --no-build-isolation` (README §4); the C++/CUDA toolchain is required, and a missing/broken build surfaces as an `ImportError`/dlopen error from `evogp.tree`.
- There is no CPU fallback: `device="cuda"` is hard-coded across the library, examples, notebooks and tests. The same `"cuda"` strings also select AMD GPUs under ROCm/HIP.
- `evogp.evogp_cuda` is a GENERATED artifact — never edit it; the sources of truth are the four `.cu` files plus `cuda/defs.h` / `cuda/kernel.h`, wired through `setup.py`.
- Compile-time constants are duplicated across the Python/C++ boundary (`MAX_STACK`, `MAX_FULL_DEPTH`, `NType`, `Func` in `src/evogp/tree/utils.py` ⇄ `src/evogp/cuda/defs.h`); changing one requires editing the other AND recompiling.
- The package DAG is strict: `tree/` is the foundation (`algorithm/` and `problem/` depend on it, `pipeline/` depends on both and is imported by nothing); preserve the direction when adding modules.
- `src/evogp/__init__.py` is empty by design — all imports go through submodules (`from evogp.tree import ...`, `from evogp.algorithm import ...`, `from evogp.pipeline import ...`, `from evogp.problem import ...`).
- Fitness contract used repo-wide: `evaluate(forest)` returns a `(pop_size,)` float32 CUDA tensor that is MAXIMIZED (losses are negated) and mutated in place by the pipeline.
- Most entry points run at import time: the `example/` scripts and `src/evogp/sr_test.py` have no `if __name__ == "__main__"` guard, so importing any of them starts a full training run (and needs a GPU).
- Optional heavy dependencies (`jax`/`brax`, `mujoco_playground`, `genesis`) are imported lazily inside constructors; none of them are declared in `setup.py` extras except brax.
- Never edit `src/evogp/hip/` — it is torch-hipify's generated build-time mirror and is gitignored.

## Known Issues
Cross-cutting problems only; the owning node documents the details.
- No automated test suite and no CI in the repo: `test/` contains ad-hoc printing scripts/notebooks with zero assertions, and nothing can run without a GPU plus a successful kernel build. The de-facto validation is `python -m evogp.sr_test` (README §5) and `python test/test_bind_success.py`.
- The multi-output / `CombinedForest` path is broken end-to-end: `SymbolicRegression.evaluate` still reads the removed `forest.share_input`, `CombinedTree.forward` ends in a `KeyError`, and `CombinedTree`/`CombinedForest` expose no `to_sympy_expr`. This same stale API (`formula=`, `share_input=`) is still used by `example/combined_sr.py`, `tutorial/combined_gp.ipynb` and `test/combined_test.ipynb`, so those three files cannot run as shipped.
- Several shipped entry points are stale or unrunnable without extra setup: `example/mjx_task.py` imports a non-existent module, `example/custom_sr.py` requires CSV data not present in the repo, `tutorial/forest_intro.ipynb` uses the pre-descriptor `Forest.random_generate` API, and `MujocoProblem` raises `NameError`. Documentation is consistently ahead of the code in the multi-output area.
- No dimension validation exists between `GenerateDescriptor.input_len/output_len` and `problem.problem_dim/solution_dim` (the pipeline's asserts are commented out), so mismatches fail deep inside the kernels.

## Test Strategy
There is no test runner. The supported sequence is: build the extension → `python -m evogp.sr_test` (synthetic symbolic-regression convergence check, the documented install validation) → `python test/test_bind_success.py` (raw `torch.ops.evogp_cuda.*` bind check) → run the relevant `example/*.py` from the repository root (relative asset paths such as `imgs/` are CWD-dependent).
Known coverage gaps: the multi-output CUDA instantiations, non-default crossover/mutation/selection operators, and the robotics-control problems.

## Notes for Agents
- Read `src/evogp/CONTEXT.md` first; it is the entry point to the whole library and carries the API surface, dependency DAG and the runtime gotchas.
- The repo has no Python tooling config (no ruff/black/mypy/pytest config), so match the surrounding style manually when editing.
- Symbolic names in the codebase lag reality in places (`evogp_cuda` ops are used on AMD too, "cuda" is a device alias for all backends); do not infer portability from names.
- The most valuable single place to look for a bug in numerical results is the boundary between `src/evogp/tree/` (Python-side tensor schema, CUDA op call sites) and `src/evogp/cuda/` (kernel semantics).

## See Also
- Foreign repositories configured for this task (absolute paths): `/mnt/local-ssd/bchuang/etl` (EvoX Tensor Library) and `/mnt/local-ssd/bchuang/evox`. No file in this repository imports from either; EvoGP's relationship with EvoX is conceptual (sister project) and acknowledged in `README.md` only. Read-only delegation into them is unrestricted; write-capable delegation is root-agent-only and must be serialized.
