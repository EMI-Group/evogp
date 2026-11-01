# `src/evogp/evox_ext/tests/` — the Evox 2.0 component contract suite
## Intent
Pins the four properties the `evogp.evox_ext` integration is defined by: `StdWorkflow` accepts both configs and runs, the state is tensors-only with constant shapes, best-so-far is monotone (and agrees with `evogp.gp`'s own evaluator), and the step graph is compiled once per state signature.
Not part of the release's other two gates (`src/evogp/gp/tests` and `src/evogp/fused/tests` do not collect it): run it explicitly.
## API Surface
- `test_evox_ext_contract.py` — 13 tests over a small fixed problem (128 rows, pop 32, 32-node trees, 4 generations, numpy backend, CPU):
  module resolution + ≥2 generations + the whole-loop `run()` entry point; frozen/hashable configs and the `make_*` validation; the `(shape, dtype)` signature of every state leaf across generations (algorithm and problem state); monotone best-so-far with real improvement; truth parity against `evogp.gp.graph_eval.sr_fitness_channels_numpy`; `host_channels`; `len(_step_exes) == 1` after three generations; the `opt_direction="max"` guard.
  The step-artifact cache is covered by four cases: the `accepts_cache_dir` signature probe, a `cache_dir` forwarded verbatim to a recording `StdWorkflow` stand-in (with the other keywords unchanged), an `evox` that declares no such keyword degrading with a `RuntimeWarning` instead of raising, and the same decision checked against the installed `evox` in both directions.
- No fixtures, no golden files, no timing assertions: the suite is deterministic (numpy backend, fixed seeds) and hermetic (no device, no env vars beyond `EVOGP_EVOXC_BACKEND` being irrelevant here — the workflow takes its backend explicitly).
  The cache cases never call `init`/`step`, so they compile nothing and write nothing outside `tmp_path`.
## Constraints
- `pytest.importorskip("evox")` and `pytest.importorskip("evoxc")` at module scope: without the EvoX checkout the file skips as a whole (one skipped test) instead of failing collection.
- Never assert wall-clock times here (the device/route economics belong to `../CONTEXT.md` and the harnesses); the assertions are on structure, monotonicity and the compile count.
- The one private attribute read is `StdWorkflow._step_exes` (the compile-count property), which is the object under test.
## Routing Table
- Sibling: `../CONTEXT.md` (the integration's intent, constraints and measured economics), `../../gp/` (the stack being adapted).
