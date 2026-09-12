# `test/benchmark/r4/` — R4 round: the pure forward measurement (measurement type 2)
## Intent
R4 measures the **internal graph forward pass** (measurement type 2) next to the end-to-end `problem.evaluate(forest)` numbers (measurement type 1) that R1–R3 quote: one "invocation" = one full-population forward on a fixed `x` with `per_row=False`, with the compile-key derivation, the readback, the host row assembly and the fitness reduction excluded. The same process also runs the real end-to-end path (`--e2e-companion K`), so `E2E − pure` prices the per-call host glue + readback on one box, in one window.
The task lives in the shared harness (`../bench_gp.py --task eval-pure`) with its runner/timing core in `../pure_forward.py`; this node owns the round's protocol, runner, manifest, validator and records.
The legacy CUDA side has no separable pure forward (one fused kernel call per evaluate), so `--task eval-pure` skips cleanly there (`skipped: true`, exit 3) — this node is new-side only.
## API surface / routing table
- `README.md` → the protocol: the two measurement types side by side, what one invocation times/pays/excludes, the `--task eval-pure` flag table, the JSON key glossary, frozen-vs-fallback labeling (incl. the after-round A/B check), commands, GPU/flock rules, file map.
- `run_pure.sh` → the 8-config matrix runner (`--group numpy|iree|all`, `--only`, `--list`, `--dry-run`, `--timeout`); one `bench_gp.py` process per config, `ulimit -n 65536` first, one `flock -w 7200 /tmp/gpbench_critical.lock` with `timeout` inside, per-config `EVOGP_EVAL_MODE=<mode>`, an `etl:` identity line per log, results into `results/<id>.{json,log}`, `last_run.txt` at the node root (generated, `.gitignore`d).
- `manifest.txt` → the validator's expectation list, 4 TAB-separated fields `id / mode / backend / path`, 8 canonical ids in runner order (group `numpy` then group `iree`; `pure64_*` before `pure200_*`, `*_dense_*` before `*_specialized_*`).
- `check_r4.py` → stdlib-only validator (`--manifest`, `--require-companion`, positional paths): task/side identity, modes+backend vs manifest, `pure`/`compile` structural completeness, per-rep list consistency (`per_invocation_ms[i] == block_ms[i]/invocations`, summaries == recomputed statistics), `per_population_evaluate_ms == median_per_invocation_ms`, the two normalizations, and the companion gate; exit 1 on any failure. A directory scan skips `results/smoke/`.
- `results/` → campaign records (written by the after-round agent; canonical paths are exactly the manifest's).
- `results/smoke/` → the pre-campaign plumbing records (3 pure smokes: dense-numpy / specialized-numpy / dense-iree + one `eval64` regression control) and `manifest_smoke.txt`; deliberately **not** manifest ids.
- `CONTEXT.md` → this file.
## Constraints
- Everything here is new-side and additive: `../SPEC.md`, `../REPORT.md`, `../r2/**`, `../new/check_smoke.py` and every existing task's behaviour are untouched; the harness changes are the `eval-pure` registry/flag/dispatch wiring plus `run_pure`.
- One process per config (the compile cache is per-process), warm/steady-state only: construction is timed separately (`compile`), then `--warmup 10`, then 5 timed blocks. `--reps 1 --warmup 0` is not quotable.
- The pure route (`--mode`) and the companion route (`EVOGP_EVAL_MODE`) are set together by the runner; a single-route config would compare two different routes.
- Never present a `local-fallback:*` record as a frozen-API number: `pure.runner_source` + `compile.path` say which constructor ran, and a frozen runner without a graph-call counter is recorded as `n_graph_calls_source: "assumed-1"`.
- `--stage-probe` is accepted but never installed for this task (it would tax the timed loop); `stages` stays `null` and the measurement lives in `pure`/`compile`/`e2e_companion`.
- GPU 6 by default, never GPU 7; medians carry the usual ±10 % shared-box band; IREE numbers are only quotable after that round's backend-agreement pre-flight.
- `../../../src/evogp/**` is read-only from here (the harness must never be the thing that changes the engine it measures).
## Status
- Harness + node tooling complete and smoke-validated: `bench_gp.py` has `--task eval-pure` (+ `--mode`, `--invocations`, `--warmup`, `--e2e-companion`, `--pure-runner`), `pure_forward.py` implements the frozen-first/fallback logic and the timing core, and the smoke records under `results/smoke/` pass `check_r4.py` on their scratch manifest.
- **Before-baselines pending:** the 8-config campaign (`run_pure.sh`) has NOT been run, so `results/` holds no canonical records yet; `check_r4.py --manifest manifest.txt --require-companion` must only be quoted once those records exist.
- The after-round (after the parallel code round lands the frozen `build_{dense,specialized}_runner` API): re-run `run_pure.sh` unchanged, A/B the same configs with `--pure-runner fallback` vs `--pure-runner frozen`, then fold the numbers into `../REPORT.md` in a later round.  The two `FROZEN-ADAPTER` functions in `../pure_forward.py` are the only places the frozen signature may need a one-line adjustment.
## Notes for Agents
- `results/smoke/*.json` are plumbing evidence (they were taken with reduced `--reps/--invocations`): never quote them as round numbers, and never add them to `manifest.txt`.
- Run everything from the repository root with the project venv (no `python` on `PATH`) and a CUDA-visible device; serialize heavy phases with the single shared `flock` and keep `timeout` inside it.
- The 3-place id sync is `run_pure.sh`'s group arrays + its `config_spec()` case + `manifest.txt`; a new id must be added to all three.
## Test Strategy
- No CI and no pytest coverage here: the validator (`check_r4.py`, stdlib-only) is the gate, plus `run_pure.sh --list` / `--dry-run` for the config table and exact commands.
- The end-to-end harness regression for this workstream is `bench_gp.py --task eval64` (unchanged numbers) — proving the additive wiring did not disturb an existing task.
