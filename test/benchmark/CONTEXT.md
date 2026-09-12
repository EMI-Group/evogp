# `test/benchmark/` — old-vs-new performance benchmark: legacy CUDA engine vs ETL "true jit" engine

## Intent
Quantitative comparison of EvoGP's two evaluation engines, measured on the same shared box through the
identical public API (time-only; the round-1 methodology is frozen in `SPEC.md`).
Two measurement types since R4: **type 1** = the end-to-end public-API evaluation quoted in R1–R3, **type 2** = the pure internal compiled-forward pass (`--task eval-pure` in `bench_gp.py`; precise definitions in `r4/README.md` and `r4/SUMMARY.md` §B).
- **OLD** = the legacy custom-CUDA engine at commit `b613503` (last pre-rewrite revision; hand-written
  `src/evogp/cuda/*.cu` + `evogp.evogp_cuda`), materialized and built outside the repo at
  `/tmp/evogp_legacy_bench` — the only side that needs a CUDA build.
- **NEW** = the ETL "true jit" engine (`src/evogp/tree/etl_engine.py`: host-side decode + per-structure
  `etl.build_cached` graphs) on its two backends: the default `numpy` (CPU reference interpreter) and the
  opt-in `iree` (IREE-cuda GPU), plus the dense one-graph evaluator (`src/evogp/tree/dense_engine.py`)
  that the `iree` backend takes under the default `EVOGP_EVAL_MODE=auto` for short-trace populations.
Four measurement rounds are executed and committed (R4 adds the second measurement type):
- **Round 1** — `SPEC.md` methodology, matrix C0–C7, records in `new/results/` + `legacy/results/`,
  write-up `SUMMARY.md`: legacy ~34×–7,670× faster per generation, the new side needing no build step.
- **Round 2 (R2)** — protocol `r2/README.md`, records in `new/results/r2/` + `legacy/results/r2/`,
  baseline analysis in the R2 sections of `REPORT.md`, operator log `r2/RUNLOG.md`. Warm/steady-state
  per-call medians as the headline, per-structure compilation reported separately; R2-era gap: the legacy
  engine is ≈33–148× faster than the numpy backend and ≈110–363× faster than iree on the fixture evals.
- **Round 3 (R3, current verdict)** — `r2/round3/` (protocol `r2/round3/README.md`): the R2 headline
  configs re-measured after two engine waves, in two arms — **Arm A** (merged evogp + stock ETL: records
  `r2/round3/results/{new,legacy,backend_agreement}/`, write-up `r2/round3/SUMMARY.md`) and **Arm B**
  (the same nine new-side configs under a patched-ETL checkout shadow: `r2/round3/results/armB/`,
  write-up `r2/round3/SUMMARY_armB.md`).
  Current verdict (the "Round 3" section at the head of `REPORT.md`): the dense route compressed the
  old-vs-new gap on the fixture evals to **≈16–23×** (iree, default) / **≈36–142×** (numpy) / **≈94–372×**
  (forced specialized); warm medians Arm A 10.64 / 10.50 ms vs legacy 0.6323 / 0.6357 ms per call, Arm B
  14.84 / 10.03 ms; the patched ETL measured ≈neutral on warm cost but flat iree fds (91 vs
  2 395–17 243) and R2's `iree_sr_b2` SIGSEGV did not reproduce (both routes exit 0).
- **Round 4 (R4, pure-forward before-baselines)** — `r4/` (protocol `r4/README.md`, analysis of record
  `r4/SUMMARY.md`): adds the type-2 task (`bench_gp.py --task eval-pure` — times only the compiled graph
  call(s) of a fixed population in a tight loop: device-resident inputs, no readback, no host assembly,
  no per-call key derivation; cold/compile reported separately) and records the pre-code-round baselines
  for 8 configs (fixture64/200 × dense/specialized × numpy/iree) at evogp `e272714` / etl `51a7141`, all
  labeled `local-fallback:*` (the frozen `build_*_runner` APIs were not landed yet). No old-vs-new row:
  the legacy engine has no separable pure forward. The after-round (frozen APIs + etl jit/hot-path)
  re-runs the protocol and diffs — checklist `r4/SUMMARY.md` §H.
Everything here is hand-run tooling + frozen data: nothing imports it and the repo has no CI.

## Status
- R2 is complete: **35/35 manifest ids have a record** (34 complete + the `iree_sr_b2` crash skeleton);
  `r2/check_r2.py --manifest r2/manifest.txt --require-stages` → 126 checks / 0 failures; the
  directory-scan mode reports only the 2 pre-existing prior-round artifacts (see Known Issues).
- R3 Arm A is complete: 12/12 records (`r2/round3/results/{new,legacy}/`) with both gates green
  (`check_r3.py` 45 checks / 0 failures; `check_agreement_r3.py` PASS, 32 checks).
- R3 Arm B is complete: 9/9 records (`r2/round3/results/armB/new/`, W2 records of record) with both gates
  green (`check_r3.py --manifest r2/round3/manifest_armB.txt --require-stages` → 34 checks / 0 failures;
  `check_agreement_armB.py` PASS, 38 checks, all six Arm A→B `values_sha256` pairs bit-identical).
- The `iree_sr_b2` re-attempt on the patched checkout completed both routes (exit 0).
- The current verdict, the gap statement and the landed/open attribution are the "Round 3" section of
  `REPORT.md`; the detailed arm write-ups are `r2/round3/SUMMARY.md` and `r2/round3/SUMMARY_armB.md`.
- Round-1 records stay validated: `new/check_smoke.py` → 8 files / 160 checks / 0 failures,
  `legacy/check_results.py` → PASS for all 8 records (C5 golden cross-check PASS), and the old side
  cross-validated through the unified harness in `xval/` (C1 −0.01 % wall / +5.8 % warm, C5 +1.9 %
  first call / +3.9 % warm, identical fitness).
- R4 (pure-forward before-baselines) is recorded: 8/8 canonical records complete, gate
  `check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion` → 8 complete / 369 checks /
  0 failures, IREE pre-flight `check_agreement_r4.py` → PASS (30 checks; numpy == iree-specialized
  bit-identical, iree-dense within 1e-4, all six `values_sha256` identical to the R3 pre-flight records).
  Every before-record is `local-fallback:*`; the `REPORT.md` Round-4 verdict update belongs to the
  after-round.

## Routing Table
- `README.md` → short entry point: what this directory is, the round-1 headline, the Round-3 current-verdict section, contents, re-run commands, shared-host/flock note.
- `SUMMARY.md` → round-1 headline report: verdict, per-config old-vs-new table (all numbers recomputed from the raw JSONs), trade-offs, caveats, golden anchor + validator counts, cross-validation (§5.1), reproducibility.
- `REPORT.md` → **the analysis of record**: leads with the **Round 3 verdict** (R2 → Arm A / Arm B, both engines, gates and caveats), then the R2 baseline sections (§1–§10) and the ranked levers, with Round-3 artifact rows and a re-run block in §10.
- `SPEC.md` → frozen round-1 methodology (2026-09-11): workloads W1–W3, config matrix C0–C7, fairness rules, JSON schema; do not edit.
- `bench_gp.py` → the side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; one JSON + one log per run, per-generation/per-call timings, plus the R2 flags (`--backend`, `--iree-device`/`--iree-opt-level`, `--data-rows`, `--mem-guard-mb`, `--max-cold-builds`/`--allow-cold-builds`, `--stage-probe` — reference table in `new/NOTES.md`); since R4 also the pure-forward task `--task eval-pure` (flags `--mode`, `--invocations`, `--warmup`, `--e2e-companion`, `--pure-runner`; runner core in `pure_forward.py`); `detect_side()` labels by engine family (ETL vs legacy CUDA), import-free.
- `pure_forward.py` → the `eval-pure` runner-build/invoke/timing core: frozen-API-first detection with a labeled `local-fallback:{dense|specialized}` path (module-private helpers: `plan_dense`/`build_dense` + `etl.run` for dense; `decode_population`/`_executable` + `etl.run` for specialized), the tight-loop timer, and the E2E companion timer; two marked `FROZEN-ADAPTER` seams assume `build_*(forest, x)` + no-arg `.run()`, adjust there when the frozen API lands.
- `stage_probe.py` → the `--stage-probe` module: runtime monkeypatching of the new engine's hot call sites into a cold-build / cache-hit / run / readback / upload split (never edits `src/`; self-test `python3 test/benchmark/stage_probe.py`; the dense route is outside its wrap set — see Known Issues).
- `new/` → NEW side: `run_matrix.sh` (round-1 runner), `check_smoke.py` (round-1 validator), `NOTES.md` (harness/schema, R2 flags, stage probe, R4 `eval-pure` section, calibration), `results/` (round-1 JSONs + logs) and `results/r2/` (R2 JSONs + logs, `smoke/` probe smoke, `backend_agreement/` pre-flight records).
- `legacy/` → OLD side: `setup_legacy.sh` (materialize + build `b613503`), `run_matrix.py` (round-1 driver), `check_results.py` (round-1 validator + C5 golden check), `BUILD_NOTES.md`, `NOTES.md`, `results/` + `results/r2/` (R2 legacy records).
- `xval/` → round-1 cross-validation of the old side through `bench_gp.py` (data, helper probes, `NOTES.md`, own `CONTEXT.md`).
- `r2/` → R2 protocol + the executed round 3: `README.md` (configs, rules, exit codes, wall-time expectations, results paths), `run_all.sh` (35-config runner), `manifest.txt` (validator's expectation list), `check_r2.py` (stdlib-only validator), `RUNLOG.md` (operator log, incl. the Round-3 Arm A + Arm B sections), `backend_agreement.md` + `check_backend_agreement.py` (the IREE-vs-numpy numeric pre-flight), `last_run.txt` (generated), `round3/` (round-3 sub-node).
- `r2/round3/` → the executed round 3, Arm A + Arm B: runners `run_all.sh` / `run_all_armB.sh`, manifests `manifest.txt` / `manifest_armB.txt` + `check_r3.py` gate, agreement pre-flights `run_agreement_r3.sh` / `run_agreement_armB.sh` + `check_agreement*.py`, comparison tools `compare_rounds*.py` (self-tests), records `results/{new,legacy,backend_agreement}/` + `results/armB/`, write-ups `SUMMARY.md` / `SUMMARY_armB.md`; entry `r2/round3/README.md`, node context `r2/round3/CONTEXT.md`.
- `r4/` → the executed round 4 (pure-forward, measurement type 2) + the pre-code-round baselines: protocol `README.md`, 8-config matrix runner `run_pure.sh` (fixture64/200 × dense/specialized × numpy/iree), agreement pre-flight `run_agreement_r4.sh` + `check_agreement_r4.py`, validator `check_r4.py` + `manifest.txt`, records `results/` (incl. `results/backend_agreement/`, `results/smoke/`), operator log `RUNLOG.md`, analysis of record `SUMMARY.md`; entry `r4/README.md`, node context `r4/CONTEXT.md`.
- `fixtures/` → the 200-tree cross-side fixture `forest200_tree64_random.npz` (same npz schema as `../goldens/forest_random.npz`, 64 shared data rows) + provenance `README.md` + the new-side eval sanity records; generator is `make_r2_fixtures.py`.
- `make_r2_fixtures.py` → generates/verifies the R2 fixture (new side, GPU, `--verify` self-checks).

## Known Issues
- **Stock ETL leaks iree file descriptors; the patched checkout fixes it.** Stock `etl` (`main` @ `a863545`): ~32–35 fds per compiled structure plus ~0.6–3/build call, so any iree config needs `ulimit -n 65536` and must be watched via `payload.fds.end`. The patched commit `51a7141` (in flight upstream — not on `main`) keeps iree flat at 91 fds at 170 / 517 / 1181 builds.
- **R2's `iree_sr_b2` SIGSEGV did not reproduce on the patched checkout.** R2: exit 139 after ~800–900 builds (skeleton kept in `new/results/r2/`). R3 re-attempt, both routes: exit 0 (dense 431/1500 misses in 213 s; pinned-specialized 1181/1500 in 595 s), fds flat 39 → 91 → 91, identical fitness trajectory — a "did not reproduce", not proof of the original cause; the b2 ids stay out of the frozen manifests.
- **IREE per-structure compile time is not reproducible on this shared box**: 342–2 162 ms per build (~6.3× spread for identical build counts), so read `first_call_ms`/`cold_build_ms` as "± a factor of a few", never as a hardware constant; only the fd curve and the warm numbers are stable.
- **Two-window contention (R3) — cross-window warm deltas are not resolvable.** W1 ran on GPU 6 with a foreign tenant (bursting 0→100 %); W2, the committed Arm B records of record, ran on GPU 0 with a mid-window tenant takeover. Within-window ratios, build traffic, exit codes, fd curves and fitness vectors are the robust metrics; single warm cells run both ways inside the band. The numpy rows ride the shared-host noise band in both rounds (no engine shift attributed).
- **Stage probe: dense-route degeneracy + unresolved tax.** The probe wraps the specialized call sites only, so dense-routing records (iree `auto`) report their cost in the uninstrumented `misc` bucket (`build_calls=0`), and the `stages` marker in manifests means "the config passes `--stage-probe`", not "the stages block is meaningful". The probe tax stays ≲1 ms on `eval64` and inside box noise on `sr`.
- **`--stage-probe` does not apply to `eval-pure`** (it wraps `etl.run`, i.e. exactly the loop being measured): the flag is accepted but warns and `stages` stays `null` in every `eval-pure` record; cold/warm separation comes from the `compile` / `pure` blocks instead.
- **R4 pure-forward baselines are fallback-path measurements**: `pure.runner_source = local-fallback:{dense|specialized}` (the frozen `dense_engine.build_dense_runner` / `etl_engine.build_specialized_runner` APIs were not landed at the baseline commit `e272714`); the after-round must flip the records to `frozen-api:*`, validate the two `FROZEN-ADAPTER` seams in `pure_forward.py` (assumed form `build_*(forest, x)` + no-arg `.run()`), and re-gate IREE via the r4 pre-flight once the ETL changes land.
- **ETL shadow identity is not recorded in the JSONs.** `etl_version` is `"0.1.0"` for both the stock and the patched checkout; the evidence is the per-config `etl:` header line in every `run_all_armB.sh` / `run_pure.sh` log plus the dated fresh-process probes in `r2/round3/results/armB/etl_import_probe.txt` (with the patched-file sha256 pins).
- **IREE `O2`/`O3` miscompile CUDA silently** — the rounds only ever use `O0`/`O1`, and any IREE timing is gated on the `O1` agreement pre-flight.
- **The R2 directory-scan validation always reports 2 failures**: `legacy/results/r2/C5.json` and `C5b.json` are prior-round artifacts of round 1's `legacy/run_matrix.py` driver (they predate the R2 key schema and are not manifest ids). The scan skips `new/results/r2/backend_agreement/` because those are pre-flight records, not config results.

## Notes for Agents
- `/tmp` artifacts are session-scoped: the built legacy checkout `/tmp/evogp_legacy_bench` and the offline build assets `/tmp/evogp_t1a263/siteutils` (py3.11 headers, sklearn) may be gone on a fresh session — rebuild with `legacy/setup_legacy.sh` (recipe + `.so` identity in `legacy/BUILD_NOTES.md`).
- The Arm B patched-ETL checkout is `/tmp`-scoped too: materialize it read-only with `git -C /mnt/local-ssd/bchuang/etl archive 51a7141 | tar -x -C <dir>` (never write in the foreign repo; never `worktree add` from a nested agent). The import precedence for `NEW_PYTHONPATH=src:<dir>` is verified (PYTHONPATH beats the PEP-660 editable finder), but re-prove the shadow per run window with the fresh-process probe. If the dangling commit is gc'd and `/tmp` is wiped, the patched revision is gone — the committed probe records are its evidence.
- Run everything from the repository root with the project venv (no `python` on `PATH`) and a CUDA-visible device. Serialize every heavy phase (legacy build, every timed run) with exactly one `flock -w 7200 /tmp/gpbench_critical.lock`; never nest it, and keep `timeout` inside the lock so lock waiting is never charged to a config. Never use GPU 7 (hardware ECC fault — any CUDA context on it dies); the runners default to GPU 6 and record `CUDA_VISIBLE_DEVICES` in each JSON. Treat medians as ±10 % and a contended window as 2–3× inflation.
- **Harness rules:** never edit `SPEC.md` (frozen round-1 methodology), and never rewrite the results JSONs/logs — each JSON records its own engine identity (`evogp_commit`, `evogp_file`, `etl_version`, `env`, `gpu`). In each runner, a config id exists in two places (the runner's table and its manifest) and the two must stay in sync; everything else is flag-parameterized, so a redesign round re-runs this harness by pointing `PY`/`NEW_PYTHONPATH`/`LEG_PYTHONPATH` at another checkout rather than editing configs.
- **R3 re-runs:** `bash test/benchmark/r2/round3/run_all.sh` (Arm A; groups `light|iree|iree-spec|all`) and `bash test/benchmark/r2/round3/run_all_armB.sh` (Arm B; `NEW_PYTHONPATH` defaults to `src:/tmp/evogp_etl_patched_51a7141`) — each after its `run_agreement_*.sh` pre-flight; validators are stdlib-only and GPU-free.
- **R4 pure-forward re-runs:** `bash test/benchmark/r4/run_pure.sh [--group numpy|iree|all] [--only <id>] [--list] [--dry-run]` from the repo root with the project venv + a CUDA-visible GPU (runner defaults: `CUDA_VISIBLE_DEVICES=6`, never GPU 7; one process per config; flock + timeout inside; sets `ETL_DISABLE_LOCATIONS=1`; `ulimit -n 65536` for iree). Before quoting IREE numbers run the pre-flight: `bash test/benchmark/r4/run_agreement_r4.sh` + `<venv>/bin/python test/benchmark/r4/check_agreement_r4.py`; gate the records with `python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion`. The before-records (`r4/results/**`, evogp `e272714` / etl `51a7141`, all `local-fallback:*`) are immutable evidence for the after-round comparison.
- `r2/last_run.txt` is generated output (the last invocation's id → exit-code table): disposable, not part of the manifest, not committed.
- IREE stays opt-in and explicitly non-primary (`--backend iree`, `EVOGP_ETL_IREE_OPT_LEVEL` O0/O1 only); the numpy backend is the default reference and the legacy side is inert to all `ETL_*`/`EVOGP_ETL_*` env vars.
- `--max-cold-builds N` bounds *cumulative cache misses* sampled at checkpoints (not FDs, and it can overshoot within one generation); it applies to the new side on IREE only. Budget fds separately.
- Round-1 data caveats still hold: `C2` is deliberately PARTIAL (89/100 generations, `projection` field in its JSON); `C4` canonical records are `C4_r1.json` + `C4_r2.json` (exclude `legacy/results/c4_rerun/` from headline tables); `wall_s` is not comparable across the two drivers for C5 (`bench_gp.py` warms the device before its wall clock, `run_matrix.py` includes one-time CUDA context creation) — compare per-call metrics.
- `new/results/r2/iree_sr_b2.json` must never be read as a training result: it is the R2 crash skeleton; the patched-checkout re-attempts under `r2/round3/results/armB/new/` are the current b2 evidence.

## Validation
- R3 (no GPU needed):
  `python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages` → 12 records / 45 checks / 0 failures;
  `python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages` → 9 records / 34 checks / 0 failures;
  `python3 test/benchmark/r2/round3/check_agreement_r3.py` → PASS (32 checks); `python3 test/benchmark/r2/round3/check_agreement_armB.py` → PASS (38 checks, 6/6 bit-identical A→B pairs); offline selftests `selftest_*.py` pass.
- R4 (no GPU needed): `python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion` → 8 records / 369 checks / 0 failures; the smoke manifest (`r4/results/smoke/manifest_smoke.txt`) stays green; the IREE agreement gate `<venv>/bin/python test/benchmark/r4/check_agreement_r4.py` → PASS (30 checks) — its records are regenerated by `run_agreement_r4.sh` on the GPU before re-quoting IREE numbers.
- R2 (no GPU needed for the validators):
  `python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages` → exit 0, 35 records, 126 checks, 0 failures (expected note: `iree_sr_b2` PARTIAL);
  `python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2` → exit 1 with exactly the 2 pre-existing `C5.json`/`C5b.json` failures.
- Backend pre-flight (GPU, once per backend/fixture before quoting IREE numbers): R2 — `r2/check_backend_agreement.py` per backend into `new/results/r2/backend_agreement/`, compare `values_sha256` (`r2/backend_agreement.md`); R3 — `r2/round3/run_agreement_r3.sh` / `run_agreement_armB.sh` then their checkers (the Arm B checker additionally gates Arm A→B identity); R4 — `r4/run_agreement_r4.sh` then `r4/check_agreement_r4.py` (numpy vs iree-dense vs iree-specialized per fixture; primary gate `values_sha256` equality, fallback `allclose` 1e-4).
- Round-1 records: `new/check_smoke.py <new results>` (JSON-only) and `legacy/check_results.py <legacy results> --gpu 4` (add `--golden test/goldens/sr_fitness.npz` for the C5 golden check). To re-run a round-1 matrix: `new/run_matrix.sh` from the repo root, or `legacy/run_matrix.py` from the legacy checkout root with the legacy `PYTHONPATH` (needs the `.so`).
- To re-run R2: `test/benchmark/r2/run_all.sh` (optionally `--group light` first) — configs, caps, exit codes and wall-time expectations in `r2/README.md`.

## See Also
- `../goldens/` → frozen pre-ETL numeric captures (provenance in its `README.md`).
- `../../src/evogp/tree/ENGINE_SPEC.md` → engine semantics of record, incl. the legacy fitness convention.
- `../../src/evogp/tree/warm_path_findings.md` → the wave-1 A/B stage record, residual-gap attribution and the upstream item list (A-2, B-1/B-2, batched readback) the Round-3 verdict cites.
