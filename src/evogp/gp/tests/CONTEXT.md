# `src/evogp/gp/tests/` — the pytest suite of the torch-free evoxc GP stack
## Intent
Regression gate for `src/evogp/gp/`: it pins the encoding contract that the operator layer, the evaluator and the engines will build on, and it pins the two import properties the stack exists for (no torch, no `evoxc` at import time).
It is a separate suite from `src/evogp/tests/` (the torch stack's suite) and is not collected by it; both must pass.
## API Surface
- `conftest.py` — sets `EVOGP_EVOXC_BACKEND=numpy` (hermetic, no IREE compile on a device box) and puts `src` on `sys.path`, both at import time before any `evoxc` use.
- `test_gp_stack_encoding.py` — 31 tests: static shape/dtype/`-inf`-fitness/key invariants, `row_length`/`live_mask` consistency, zero padding and zero function-node payloads, well-formedness (depth bound, arity-derived subtree sizes, opcode legality, variable-index range), key determinism (same key ⇒ identical population, new key ⇒ different), key-tensor vs seed equivalence, `pop_size` sources (`(GPTreeConfig, GPConfig)` pair), graph memoization, `output_len > 1` rejection, degenerate function sets (unary-only, single level, all leaves), the leaf/const/variable distribution against the configured probabilities, the legacy-triple round-trip (both directions, plus `OUT_NODE` rejection), and the `validate_channels` rejection matrix.
- `test_gp_stack_config.py` — 27 tests: derived tables (cdf, depth schedule, padded layout, arity/type tables), frozen/hashable/`replace`, the `using_funcs` spellings, weighted cdfs, zero-weight functions, deterministic const drawing, the validation matrix, and the `GPConfig` rules.
- `test_gp_stack_purity.py` — 9 tests: a source grep that every module of the subtree (excluding `tests/`) is free of `import torch` / `from torch` / `import evox` / `from evox`; a clean-subprocess check that `import evogp.gp` works with `PYTHONPATH=src` alone and pulls in neither torch, `evox` nor `evoxc`; a clear-error check when `evoxc` is missing at call time; and the constants parity against `evogp.tree.utils` and `evogp.tree.generation._func_tables`.
## Constraints
- numpy evoxc backend only, small sizes (pop ≤ 2 048, `max_nodes` ≤ 64), and no timing loops — device measurement belongs to the benchmarking task.
- Torch may be imported here (it is the reference oracle), never in the subtree's own modules; the constants-parity tests use `pytest.importorskip` so a torch-less box skips instead of failing.
- Tests must not assume execution order: `init_population` memoizes graphs process-wide (`evogp.gp.clear_cache()` is the reset).
## Notes for Agents
- Run: `EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python -m pytest src/evogp/gp/tests -q` → 67 passed in ~8 s.
- The `subprocess` purity tests re-invoke `sys.executable` with `PYTHONPATH` overridden; keep them independent of the ambient `PYTHONPATH` (that is the whole point).
- A new module in `src/evogp/gp/` is automatically covered by the source-grep test (it globs the directory), so a stray `import torch` fails without any test edit.
