# `test/benchmark/evoxc/` — evoxc engine vs plain-PyTorch controlled harness

## Intent
Answer one question with evidence: what does the evoxc-based EvoGP engine cost per evaluation compared with the plain-PyTorch implementation of the same tree evaluation, on identical populations, in the same session, warm/steady-state.
It is the "new engine vs old torch version" counterpart to `../r4/`/`../r5/`/`../r6/`, which compare the evoxc engine against the deleted legacy CUDA engine; those rounds are frozen history and are never edited from here.
Every arm is validated numerically against `src/evogp/tests/reference_eval.py` before its timings are trusted, and every record states the environment, the route that actually resolved, and the cold compile it paid.

## Contents
- `common.py` — ONE definition of every workload: the cell grid (`main_cells()`/`n64_cells()`/`parse_cell`), deterministic `build_workload` (`torch.manual_seed(0)` + `Forest.random_generate`), the `+ - * /` descriptor, `env_block` (git SHA, versions, GPU, knobs), `time_call` (cold call separated from warmup+reps medians), `digest`, `route_decision`, `src_root()`/`checkout_root()`.
- `arms_torch.py` — the two pure-torch baselines: `PerTreeEvaluator` (Python tree walk, torch ops per node) and `BatchedEvaluator` (one vectorized pass over the population, per-position masked `torch.where`, per-row child gather). Op semantics are copied from the oracle.
- `arms.py` — the arm namespace: `TorchArm`/`EvoxcArm`, `apply_arm_env` (`EVOGP_EVAL_MODE`/`EVOGP_EVOXC_BACKEND` per arm), `make_arm`, `SRLossProblem` + `make_pipeline` for the GP generation step.
- `worker.py` — ONE process, ONE `(cell, arm)`: cold call, warmup, reps, cache deltas, value digests, `gp_step` split into eval/operator, per-workload `first_call_ms`. Honors `--tree-root`.
- `campaign.py` — the sequential driver: one subprocess per config, idle-device guard (`rocm-smi --showpids`, excluding the driver's own ancestor chain), `--tree-root`/`--resume`/scope policy, per-config `.log` + `campaign.json` with the guard verdict per config.
- `validate.py` — every arm vs the oracle and vs the engine, three function sets (`arith` bitwise-mandatory, `rich` transcendental spread, `ternary` for the `if` operand order).
- `decompose.py` — per-call phase breakdown of one cell on both routes (inclusive `perf_counter_ns` timers on the engine's own functions, residual row for host glue, seam counters for dispatches/uploads/readbacks, cold compile separate).
- `summarize.py` — stdlib-only renderer: environment block, per-workload tables (median with min in parentheses), `auto` route per cell, speedup vs both torch arms, `gp_step` eval/ops split, cold-compile table, digest agreement, caveats.
- `check.py` — stdlib-only gate over a result directory: record presence/schema, `len(samples_ms) == reps >= 5`, medians re-derived from samples, `first_call_ms > 0`, evoxc route + compile-miss fields, cross-arm digest agreement, `gp_step` sanity, contaminated-guard warnings. Exits 1 on failure.
- `results/` — the evidence: `baseline/<cell>/<arm>.json` + `.log`, `campaign.json` (driver record incl. guard verdicts and scope exclusions), `validation.json`, `decompose/<cell>.json`.

## Constraints
- One process per timed config (the engine's compile cache and executable memo are per-process), and never two timed configs concurrently.
- Never run anything else that imports torch while a campaign is in flight — the box has one usable device and the numbers are the deliverable.
- Cold compile never sits in the warm numbers: `worker.py` times the first call separately (`first_call_ms`, `cache.compile_misses_first_call`) before the warmup loop.
- `specialized`-iree costs ~272-284 ms of compile per distinct tree structure on this box, so that arm is scoped to `pop ∈ {16, 64}`; the dropped cells are listed in `campaign.json` (`protocol.scope_dropped`) instead of being silently omitted.
- Records are written verbatim: a contaminated or re-run cell is annotated (`--note`, guard block), never smoothed.
- This subtree is not collected by pytest and is not part of the `src/evogp/tests/` suite.

## Routing Table
- `results/` → data only (records, logs, driver record); the summary of record is `SUMMARY.md` at this node.
- `README.md` → methodology, arms, grid, commands, validation and decomposition notes.
- `SUMMARY.md` → the headline tables (per workload × pop × route vs torch), the decomposition profile, the `auto`-route table, the validation results and the pre-optimization SHA.
- `../r4/`, `../r5/`, `../r6/` → the historical legacy-CUDA-engine comparison; read-only from here.
- `../../../src/evogp/tests/reference_eval.py` → the oracle every arm is validated against.
