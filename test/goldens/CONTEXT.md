# `test/goldens/` — frozen numeric outputs of the deleted pre-ETL CUDA-kernel engine

## Intent
This directory is data-only: four `.npz` files holding frozen reference numbers produced by the legacy EvoGP engine (hand-written CUDA kernels in the now-deleted `src/evogp/cuda/`, bound via `evogp.evogp_cuda`) at revision `b613503`.
They are the behaviour the pure-torch/ETL rewrite had to reproduce and cannot be re-captured from source; they must never be overwritten.
`README.md` in this directory is the source of truth for provenance, per-array inventory, tolerances and usage recipes — read it before touching any fixture.

## Contents / API surface (data only, no code)
- `forest_random.npz` — 18 arrays: forest generation (`tree_generate`) for a 64-tree / 63-node descriptor, incl. `forest__node_value/_node_type/_subtree_size`, `x`, `labels`, `x_small`, `descriptor__*`.
- `forward_eval.npz` — 13 arrays: `tree_forward`, `forest_forward`, `forest_batch_forward` plus informational `tree_infix` / `tree_sympy_str` / `tree_preorder_str`.
- `sr_fitness.npz` — 31 arrays: `forest_sr_fitness__<mode>__{mse,nonmse}`, `tree_sr_fitness__*`, `problem_evaluate__<mode>__*` for kernel modes `hybrid_parallel`, `data_parallel`, `tree_parallel`, `auto` and the `torch` path.
- `training.npz` — 34 arrays: a 12-generation `SymbolicRegression` + `GeneticProgramming` + `StandardPipeline(step)` reference (datapoints/labels, initial/final/best forests, `generation_fitness`, `best_fitness_per_generation`).
- `README.md` — provenance + semantics + tolerances + how-to-use recipes.

## Constraints
- Read-only fixtures: `golden_capture.py capture` would replace these legacy numbers (it warns on `--out test/goldens`); `verify` only compares one build against itself.
- Floats: compare with `rtol=atol=1e-4, equal_nan=True` (legacy kernels used `-use_fast_math`, execute modes reduce in different orders, max observed spread `2.3e-05`); ints/strings exact.
- Compare on the used prefix only: dead region `index >= subtree_size[j]` was uninitialised device memory, zero-filled at capture time, "don't care".
- RNG-dependent arrays (generated structures, mutation/crossover outcomes, training trajectory) follow the legacy taus88-style stream and are NOT reproducible by today's torch-RNG generator — structural/behavioural reference only.
- RNG-independent arrays (rebuild a `Forest` from the stored tensors; `forward`/`batch_forward`/`SR_fitness`/`problem_evaluate`) are the strongest live numeric check: current ETL engine agrees to max abs `2.29e-05` / max rel `4.0e-07` with identical NaN/±inf placement.

## Known Issues
- `src/evogp/tests/test_equivalence.py::test_golden_fixtures` scans this directory but expects the flat engine-fixture schema (`node_value`, `node_type`, `subtree_size`, `x` + `expected`/`expected_fitness`); these raw captures use the `forest__*` schema, so the loader fails on `forest_random.npz` unless it is relaxed to skip non-fixture `npz` files.
- Coverage gaps: single output, `input_len=2`, only `+ - * /` with a fixed 8-value constant pool; no golden for `if`/comparison/`unary`/`loose_*` ops, multi-output `CombinedForest`, mutation/crossover on fixed tensors, or `to_png`/`to_sympy_expr` beyond informational strings.

## Routing / See Also
- Sibling tooling: `../golden_capture.py` (capture/verify/manifest driver, GPU + `PYTHONPATH=src` + project venv).
- Parent node `../CONTEXT.md` documents the whole `test/` scratch layer; `../benchmark/` is the old-vs-new performance subtree (verdict `../benchmark/REPORT.md`).
- Legacy-engine semantics of record: `../../src/evogp/tree/ENGINE_SPEC.md` (appendices) and `README.md` here.
