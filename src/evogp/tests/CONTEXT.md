# `src/evogp/tests/` — pytest suite for the ETL-based (CUDA-free) EvoGP engine
## Intent
Automated verification of EvoGP's rewritten evaluation engine: GP trees are evaluated through ETL's "true jit" (host-side decode of the population tensors into a static structure → `etl.build_cached` specialization → shared `etl.CompileCache`), and generation/mutation/crossover are plain-torch tensor ops.
These tests are the regression gate: an independent plain-torch reference evaluator plus randomized/adversarial equivalence tests, operator-invariant tests, public-API compatibility tests, and a small convergence smoke.
## Contents
- `reference_eval.py` — independent plain-torch oracle (slow, obviously-correct) implementing the original `_process_node` semantics; must NOT import the engine it verifies.
- `test_equivalence.py` — randomized + adversarial engine-vs-reference equivalence through the public `Forest`/`Tree` API; includes the golden-fixture loader.
- `test_operators.py` — generation validity/reproducibility + mutate/crossover splice invariants and silent-copy edge cases (int32 index tensors).
- `test_api_compat.py` — import surface, indexing/pickle/concat/sympy, execute_mode string compatibility.
- `test_convergence_smoke.py` — tiny end-to-end symbolic-regression run; asserts best fitness improves.
- `perf_probe.py` — standalone ETL build/run economics probe (numpy vs iree); NOT a pytest module; run directly.
## How to run
From the repository root:
`PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q`
(Single file: replace the path; `-x` stops at the first failure. Perf probe: `PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python src/evogp/tests/perf_probe.py --quick`.)
## Constraints
- No test may import `evogp.evogp_cuda` (the custom extension no longer exists) and the reference path must not require a GPU.
- `reference_eval.py` stays independent: no imports from `evogp.tree.etl_engine` (it is the oracle the engine is checked against).
- RNG semantics intentionally differ from the old kernels (torch-RNG sampling, distribution-equivalent, not bit-exact) — tests must not assert exact sampled values.
- Semantics authority: `../tree/ENGINE_SPEC.md` (durable spec extracted from the kernels) and `../tree/CONTEXT.md`.
## Golden fixtures (old-CUDA cross-check hook)
`test_equivalence.py` scans, in order, `$EVOGP_GOLDEN_DIR`, `<repo-root>/test/goldens/`, and `src/evogp/tests/goldens/` for `*.npz` fixtures and compares engine outputs against them, skipping cleanly when none exist.
Fixture format (one `.npz` per case): `node_value (pop, max_tree_len) f32`, `node_type i16`, `subtree_size i16`, `x (N, input_len) f32`, `expected (pop, N, out_len) f32` OR `expected_fitness (pop,) f32` (+ optional `labels (N, out_len) f32` / `use_mse`); the loader is structured so an externally produced golden set can be dropped in without test changes.
## Routing Table
No child directories — do not look for sub-nodes to delegate to.
Sibling paths (read-only from here — escalate writes to this node's parent): `../tree/` (the engine these tests exercise), `../algorithm/`, `../problem/`, `../pipeline/`; repo-root `../../../test/`, `../../../example/`, `../../../tutorial/` are sibling tasks' domains.
