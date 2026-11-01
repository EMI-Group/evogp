# `test/oracle/` — dev-only numeric oracle and frozen fixtures for the fused route

## Intent
This directory holds the independent, deliberately slow plain-torch reimplementation of the tree evaluator and its frozen `.npz` captures.
It is DEV-ONLY tooling: nothing here is shipped in the `evogp` wheel and nothing under `src/evogp/` imports it.
It exists because the package no longer carries a torch stack, while the Triton `fused` route still needs a numeric check written independently of the code it checks.
It was formerly shipped at `src/evogp/tests/reference_eval.py` + `src/evogp/tests/goldens/` and moved here in the 2.0.0a2 pre-release.
The consumers are `src/evogp/fused/tests/` (pytest parity cases) and `python -m evogp.fused.check --stage parity|goldens|multi` (the device battery).

## API Surface
`reference_eval.py` exposes the node/function id constants it duplicates from the CUDA-era `defs.h` (`MAX_STACK`, `DELTA`, `MAX_VAL`, the `NType` ids `VAR`/`CONST`/`UFUNC`/`BFUNC`/`TFUNC`, `TYPE_MASK`, `OUT_NODE`, every `Func` id up to `FUNC_END`), the `Node` namedtuple and `decode_row` / `flatten` / `arity_of` / `is_function_kind` helpers.
Its evaluation entry points are `forward_tree` (one decoded `Node`), `forward_from_rows` (`per_row=True`, `(pop, output_len)`), `batch_forward_from_rows` (`per_row=False`, `(pop, N, output_len)`) and `sr_fitness_from_rows` (positive SR loss, MSE or MAE).
`goldens/` holds seven frozen `.npz` cases (schemas and per-case meaning in `goldens/README.md`): `01_random_single_out`, `02_edge_finite`, `03_edge_nan`, `04_fitness_mse`, `05_fitness_mae`, `06_multi_output`, `07_deep_skew`.

## Constraints
This is the one directory where `import torch` is expected and correct; the module imports only `numpy`/`torch`.
It must stay independent of `evogp` internals — it may import nothing from `evogp`, so it can catch the library's bugs instead of inheriting them.
Every node result is the correctly rounded float32 value of the op on its float32 operands (evaluated in float64 and rounded back), because torch's own float32 `sqrt`/`exp`/`sin`/`cos`/`tan` are up to one ulp off and deep trees amplify that; comparisons and the `if` select stay float32.
The `.npz` files are frozen historical captures: never re-capture them, never edit them, and compare them at 1e-4 rtol/atol.
This directory is deliberately NOT on `PYTHONPATH`; a caller adds it (`sys.path.append(<repo>/test/oracle)`, or `PYTHONPATH=src:test/oracle`) and skips cleanly when it is absent.
Not collected by any shipped suite: `pytest src/evogp/...` never reaches it.

## Routing Table
- `./reference_eval.py` → The independent plain-torch evaluator (value paths, per-row path, SR loss) and the id constants it duplicates from the kernel era.
- `./goldens/` → Seven frozen `.npz` fixtures + `README.md` describing each schema, case and the required NaN-aware comparison.

## Notes for Agents
- Use it to answer "is the fused kernel wrong or is my expectation wrong": the oracle is the semantics of record for op placement, NaN/±inf and the multi-output accumulation rule.
- It is slow by design (per-node Python loop on the host); never put it in a hot path and never call it from library code.
- `src/evogp/_tables.py` is the package's own (torch-free) id table; the oracle duplicates the same numbers on purpose and must not be refactored to import them.
