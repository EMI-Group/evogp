# `src/evogp/tests/` — pytest suite for the ETL-based (CUDA-free) EvoGP engine
## Intent
Automated verification of EvoGP's rewritten evaluation engine: GP trees are evaluated through ETL's "true jit" (host-side decode of the population tensors into a static structure → `etl.build_cached` specialization → shared `etl.CompileCache`), and generation/mutation/crossover are plain-torch tensor ops.
These tests are the regression gate: an independent plain-torch reference evaluator plus randomized/adversarial equivalence tests, operator-invariant tests, public-API compatibility tests, a small convergence smoke, and the numeric cross-check against the frozen outputs of the deleted CUDA kernels.
## Contents
- `reference_eval.py` — independent plain-torch oracle (slow, obviously-correct) implementing the original `_process_node` semantics; must NOT import the engine it verifies.
- `test_equivalence.py` — randomized + adversarial engine-vs-reference equivalence through the public `Forest`/`Tree` API; also owns the golden-fixture discovery, schema dispatch and every golden comparison (see "Golden fixtures" below).
- `test_operators.py` — generation validity/reproducibility + mutate/crossover splice invariants and silent-copy edge cases (int32 index tensors).
- `test_api_compat.py` — import surface (incl. the `CustomLoss` re-export), indexing/pickle/concat/sympy (incl. constant-only tree repr/sympy), execute_mode string compatibility.
- `test_combined_forward.py` — input-dict shape-contract regression for `CombinedForest.forward`/`batch_forward` and `CombinedTree.forward`: hand-checked single-point/batch values, batch==per-row equivalence, column-form equivalence and the exact rejection messages (12 tests).
- `test_mujoco_problem.py` — `MujocoProblem` lazy-import binding (module globals `jax`/`jnp`/`registry`/`MjxEnv` after `import_jax_based_package`) plus a construction smoke that skips at the third-party `mujoco-warp`/CuDNN boundary.
- `test_convergence_smoke.py` — tiny end-to-end symbolic-regression run; asserts best fitness improves.
- `perf_probe.py` — standalone ETL build/run economics probe (numpy vs iree); NOT a pytest module; run directly.
- `goldens/` — data only: 7 committed `.npz` golden fixtures (`01_random_single_out` … `07_deep_skew`) in the **engine schema** plus `README.md` documenting each case and the format. Loaded by `test_equivalence.py`; not a Python package.
## How to run
From the repository root (`ETL_DISABLE_LOCATIONS=1` speeds tracing up ~4× and is what the counts below were measured with):
- Whole suite: `ETL_DISABLE_LOCATIONS=1 PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -q` → **292 passed, 3 skipped, 6 warnings in ~1m45s** (the 3 skips are pre-existing: `test_api_compat.py`'s pygraphviz/`to_png` guard and the two "`forward` exposes no `execute_mode`" guards).
- Golden cross-checks only: append `-k golden` (6 tests, ~6 s).
- Single file: replace the path (`-x` stops at the first failure; the heavy file is `test_equivalence.py`, ~1m40s of the total).
- Perf probe: `PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python src/evogp/tests/perf_probe.py --quick`.
## Constraints
- No test may depend on a compiled extension (there is none any more) and the reference path must not require a GPU.
- `reference_eval.py` stays independent: no imports from `evogp.tree.etl_engine` (it is the oracle the engine is checked against).
- RNG semantics intentionally differ from the old kernels (torch-RNG sampling, distribution-equivalent, not bit-exact) — tests must not assert exact sampled values; the legacy fixtures are therefore used as frozen *inputs* to the engine, never as RNG transcripts.
- Semantics authority: `../tree/ENGINE_SPEC.md` (durable spec extracted from the kernels) and `../tree/CONTEXT.md`; for the legacy capture also `<repo-root>/test/goldens/README.md`.
## Golden fixtures
Two fixture sources with **two different schemas**, both optional, both handled by `test_equivalence.py`.
1. **engine schema** — `src/evogp/tests/goldens/*.npz` (7 cases, captured from this engine). One `.npz` per case: `node_value (pop, max_tree_len) f32`, `node_type i16`, `subtree_size i16`, `x (N, input_len) f32`, plus `expected (pop, N, out_len) f32` (checked against `batch_forward`) or `expected_fitness (pop,) f32` + optional `labels (N, out_len) f32` / `use_mse` (checked against `SR_fitness`). Checked by `test_golden_engine_schema_fixtures`.
2. **legacy-capture schema** — `<repo-root>/test/goldens/*.npz` (read-only for this node), the frozen outputs of the **deleted custom CUDA kernels**, produced by `test/golden_capture.py` and documented array-by-array in `test/goldens/README.md`. Files: `forest_random.npz` (generation stage), `forward_eval.npz` (tree/forest/batch forward + display strings), `sr_fitness.npz` (all four execute modes × MSE/MAE at forest, tree and `SymbolicRegression` level + the pure-torch path), `training.npz` (12-generation run: descriptors, `initial_forest__*`, `final_forest__*`, `best_tree__*`, `generation_fitness`). Recognized by any `<prefix>__node_value` + `<prefix>__node_type` + `<prefix>__subtree_size` triple (`forest`, `initial_forest`, `final_forest`, `best_tree`).
Comparison rules, taken from `test/goldens/README.md` ("Recommended comparison tolerances"): floats `rtol = atol = 1e-4` (`LEGACY_RTOL`/`LEGACY_ATOL`), integer/structural/string arrays **exact**, and NaN/±inf **placement** asserted exactly (the `assert_close_with_placement` helper; 3/64 trees in `sr_fitness.npz` are NaN by design). `Forest`/`Tree` objects are rebuilt from the stored rows, so no RNG is involved.
The six golden tests (`-k golden`) and what they check:
- `test_golden_fixture_discovery` — every discovered `*.npz` is classified; nothing here can fail the suite (reporting/robustness only).
- `test_golden_engine_schema_fixtures` — schema 1 above.
- `test_golden_legacy_capture_forward` — `tree_forward` (8 trees), `forest_forward` (64×1), `forest_batch_forward` (64×8×1) and the display strings (`to_infix`/`to_sympy_expr`/`str_tree`), all exact-placement + 1e-4.
- `test_golden_legacy_capture_sr_fitness` — `forest_sr_fitness__<mode>__<mse|nonmse>`, `tree_sr_fitness__*` and `problem_evaluate__*` (incl. the pure-torch path) for the four kernel modes.
- `test_golden_legacy_capture_generation` — `forest_random.npz`'s rows/`x`/`labels`/`x_small` are bitwise identical to the evaluation captures (documented) and reproduce `forward_eval.npz`'s `forest_batch_forward` from those rows.
- `test_golden_legacy_capture_training` — the RNG-independent parts: `initial_fitness` from the stored `initial_forest__*` (+ the pipeline's `NaN → -inf` convention), `best_tree` fitness vs the stored final `best_fitness_per_generation`, plus structural invariants.
Fixture policy (do not regress this):
- Fixtures are classified by **array names, never by directory** — the two READMEs invite copying fixtures between directories, and the loader must not feed a file to the wrong schema.
- Foreign/unreadable fixtures and arrays a fixture does not store are reported via `warnings.warn` with an explicit reason (see `_GoldenOutcome`, which also turns a test that compared nothing into a `pytest.skip`) — they never fail the suite and are never silently ignored.
- Deliberate, documented skips surfaced as warnings: regeneration of the legacy generation stage (torch-RNG stream is distribution-equivalent, not bit-identical) and the training trajectory `generation_fitness[1:]` / `final_forest__*` (needs the legacy kernels' exact RNG-consumption order).
- `$EVOGP_GOLDEN_DIR` still prepends an extra scan directory; the two in-repo directories are always scanned.
## Routing Table
No code subdirectories — `goldens/` is a data directory only (see its `README.md`), so do not look for sub-nodes to delegate to.
Sibling paths (read-only from here — escalate writes to this node's parent): `../tree/` (the engine these tests exercise), `../algorithm/`, `../problem/`, `../pipeline/`; repo-root `../../../test/` (holds `goldens/` + `golden_capture.py`), `../../../example/`, `../../../tutorial/` are sibling tasks' domains.
