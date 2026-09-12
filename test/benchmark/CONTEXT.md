# `test/benchmark/` — old-vs-new performance benchmark: legacy CUDA engine vs ETL "true jit" engine

## Intent
Quantitative comparison of EvoGP's two evaluation engines, measured on the same shared box through the
identical public API (time-only; methodology frozen in `SPEC.md`).
- **OLD** = the legacy custom-CUDA engine at commit `b613503` (last pre-rewrite revision; hand-written
  `src/evogp/cuda/*.cu` + `evogp.evogp_cuda`), materialized and built outside the repo at
  `/tmp/evogp_legacy_bench` — the only side that needs a CUDA build.
- **NEW** = the ETL "true jit" engine (`src/evogp/tree/etl_engine.py`: host-side decode + per-structure
  `etl.build_cached` graphs) on its two usable backends: the default `numpy` (CPU reference interpreter)
  and the opt-in `iree` (IREE-cuda GPU).
Two measurement rounds live here, both executed and committed:
- **Round 1** — `SPEC.md` methodology, matrix C0–C7, records in `new/results/` + `legacy/results/`,
  write-up `SUMMARY.md`: legacy ~34×–7,670× faster per generation, the new side needing no build step.
- **Round 2 (R2)** — protocol in `r2/README.md`, records in `new/results/r2/` + `legacy/results/r2/`,
  write-up `REPORT.md`, operator log `r2/RUNLOG.md`. It repeats the old-vs-new comparison with
  **warm/steady-state per-call numbers as the headline and per-structure compilation reported
  separately**, and adds data-row scaling, a 200-tree fixture, the `stages` cold/warm decomposition, a
  cold-build budget for IREE, and probe-off A/B controls.
R2 headline (fixture configs, warm per-call medians): legacy **0.63–1.02 ms**, ETL/`numpy`
**9.0–105.4 ms**, ETL/`iree` **24.5–260.4 ms** — the legacy engine is ≈33–148× faster than the numpy
backend and ≈110–363× faster than IREE. IREE's warm cost is ≈2.5–3.3× the numpy backend's, and its
per-structure compile is 0.34–2.16 s versus ~3.2 ms for numpy (98.6 % of `iree_sr_a`'s first rep is
build misses), which is why compile is never mixed into the warm headline.
Everything here is hand-run tooling + frozen data: nothing imports it and the repo has no CI.

## Status
R2 is complete: all four groups ran on 2026-09-12 — legacy sanity + light group (11 legacy configs +
12 numpy mirrors), the IREE eval group (10 configs, incl. the `C7` anchor re-run), and the IREE
training pair (`iree_sr_a` measured, `iree_sr_b2` crashed). **35/35 manifest ids have a record** (34
complete + the `iree_sr_b2` crash skeleton).
Validators over the committed records: `r2/check_r2.py --manifest r2/manifest.txt --require-stages` →
35 records / 126 checks / **0 failures**; the directory-scan mode → 50 records / 2 failures, both
pre-existing prior-round artifacts (`legacy/results/r2/C5.json`, `C5b.json`, written by round 1's
driver and not R2 manifest ids).
Pre-flight gate: IREE-cuda `O1` and the `numpy` backend return **bit-identical** fitness on both
fixtures (`r2/backend_agreement.md`), so the round's IREE timings are quotable; `O2`/`O3` stay unused.
Round-1 records stay validated: `new/check_smoke.py` → 8 files / 160 checks / 0 failures,
`legacy/check_results.py` → PASS for all 8 records (C5 golden cross-check PASS), and the old side
cross-validated through the unified harness in `xval/` (C1 −0.01 % wall / +5.8 % warm, C5 +1.9 %
first call / +3.9 % warm, identical fitness).

## Routing Table
- `README.md` → short entry point: what this directory is, round-1 headline, contents, re-run commands, shared-host/flock note, R2 summary.
- `SUMMARY.md` → round-1 headline report: verdict, per-config old-vs-new table (all numbers recomputed from the raw JSONs), trade-offs, caveats, golden anchor + validator counts, cross-validation (§5.1), reproducibility.
- `REPORT.md` → the R2 report: warm/compile-separated old-vs-new comparison built from `new/results/r2/` + `legacy/results/r2/`, with the executed-round narrative in `r2/RUNLOG.md`.
- `SPEC.md` → frozen round-1 methodology (2026-09-11): workloads W1–W3, config matrix C0–C7, fairness rules, JSON schema; do not edit.
- `bench_gp.py` → the side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; one JSON + one log per run, per-generation/per-call timings, plus the R2 flags (`--backend`, `--iree-device`/`--iree-opt-level`, `--data-rows`, `--mem-guard-mb`, `--max-cold-builds`/`--allow-cold-builds`, `--stage-probe` — reference table in `new/NOTES.md`).
- `stage_probe.py` → the `--stage-probe` module: runtime monkeypatching of the new engine's hot call sites into a cold-build / cache-hit / run / readback / upload split (never edits `src/`; self-test `python3 test/benchmark/stage_probe.py`).
- `new/` → NEW side: `run_matrix.sh` (round-1 runner), `check_smoke.py` (round-1 validator), `NOTES.md` (harness/schema, R2 flags, stage probe, calibration), `results/` (round-1 JSONs + logs) and `results/r2/` (R2 JSONs + logs, `smoke/` probe smoke, `backend_agreement/` pre-flight records).
- `legacy/` → OLD side: `setup_legacy.sh` (materialize + build `b613503`), `run_matrix.py` (round-1 driver), `check_results.py` (round-1 validator + C5 golden check), `BUILD_NOTES.md`, `NOTES.md`, `results/` + `results/r2/` (R2 legacy records).
- `xval/` → round-1 cross-validation of the old side through `bench_gp.py` (data, helper probes, `NOTES.md`, own `CONTEXT.md`).
- `r2/` → the R2 protocol: `README.md` (configs, rules, exit codes, wall-time expectations, results paths), `run_all.sh` (35-config runner), `manifest.txt` (validator's expectation list), `check_r2.py` (stdlib-only validator), `RUNLOG.md` (operator log of the executed round), `backend_agreement.md` + `check_backend_agreement.py` (the IREE-vs-numpy numeric pre-flight), `last_run.txt` (generated).
- `fixtures/` → the 200-tree cross-side fixture `forest200_tree64_random.npz` (same npz schema as `../goldens/forest_random.npz`, 64 shared data rows) + provenance `README.md` + the new-side eval sanity records; generator is `make_r2_fixtures.py`.
- `make_r2_fixtures.py` → generates/verifies the R2 fixture (new side, GPU, `--verify` self-checks).

## Known Issues
- **`iree_sr_b2` (pop 1000 / gens 2 / tree 128) is a crash, not a measurement**: silent native SIGSEGV
  (runner exit 139) after ~800–900 builds and before generation 0 completed; the skeleton JSON
  (`complete: false`, no per-gen data) is kept as the round's bounded-failure datapoint. IREE is not
  runnable at that population on this box under the 8 192-entry cache / 65 536-fd limit, and the crash
  is not reproducible from the record (no core, no per-build logging).
- **IREE leaks file descriptors**: ~32–35 fds per compiled structure plus ~0.6–3 fds per build call on
  the warm path, and the nanobind teardown prints `leaked …` runtime-object warnings in every IREE log.
  Any IREE config must raise `ulimit -n 65536` first and be watched via `payload.fds.end`.
- **IREE per-structure compile time is not reproducible on this shared box**: 342–2 162 ms per build
  (~6.3× spread for identical build counts) across the round, so read `first_call_ms`/`cold_build_ms` as
  "± a factor of a few", never as a hardware constant; only the fd rate and the warm numbers are stable.
- **IREE `O2`/`O3` miscompile CUDA silently** — the round only ever uses `O0`/`O1`, and any IREE timing
  is gated on the `O1` agreement pre-flight.
- **The directory-scan validation always reports 2 failures**: `legacy/results/r2/C5.json` and `C5b.json`
  are prior-round artifacts of round 1's `legacy/run_matrix.py` driver (they predate the R2 key schema
  and are not manifest ids). The scan skips `new/results/r2/backend_agreement/` because those are
  pre-flight records, not config results.
- **Stage-probe overhead is only partly resolved**: the A/B controls put it at ≲1 ms on `eval64` and
  inside box noise on `sr`; it never changes an R2 comparison, but "probe tax" is not a per-config
  correction factor.

## Notes for Agents
- `/tmp` artifacts are session-scoped: the built legacy checkout `/tmp/evogp_legacy_bench` and the
  offline build assets `/tmp/evogp_t1a263/siteutils` (py3.11 headers, sklearn) may be gone on a fresh
  session — rebuild with `legacy/setup_legacy.sh` (recipe + `.so` identity in `legacy/BUILD_NOTES.md`).
- Run everything from the repository root with the project venv (no `python` on `PATH`) and a
  CUDA-visible device. Serialize every heavy phase (legacy build, every timed run) with exactly one
  `flock -w 7200 /tmp/gpbench_critical.lock`; never nest it, and keep `timeout` inside the lock so lock
  waiting is never charged to a config. Never use GPU 7 (hardware ECC fault — any CUDA context on it
  dies); the runner defaults to GPU 6 and records `CUDA_VISIBLE_DEVICES` in each JSON. Treat medians as
  ±10 % and a contended window as 2–3× inflation.
- **Harness rules:** never edit `SPEC.md` (frozen round-1 methodology), and never rewrite the results
  JSONs/logs — each JSON records its own engine identity (`evogp_commit`, `evogp_file`, `etl_version`,
  `env`, `gpu`). In the R2 runner, a config id exists in two places (`run_all.sh`'s table and
  `manifest.txt`) and the two must stay in sync; everything else is flag-parameterized, so the redesign
  workstream re-runs this harness by pointing `PY`/`NEW_PYTHONPATH`/`LEG_PYTHONPATH` at another checkout
  rather than editing configs.
- `r2/last_run.txt` is generated output (the last invocation's id → exit-code table): disposable, not
  part of the manifest, not committed.
- IREE stays opt-in and explicitly non-primary (`--backend iree`, `EVOGP_ETL_IREE_OPT_LEVEL` O0/O1 only);
  the numpy backend is the default reference and the legacy side is inert to all `ETL_*`/`EVOGP_ETL_*`
  env vars.
- `--max-cold-builds N` bounds *cumulative cache misses* sampled at checkpoints (not FDs, and it can
  overshoot within one generation); it applies to the new side on IREE only. Budget fds separately.
- Round-1 data caveats still hold: `C2` is deliberately PARTIAL (89/100 generations, `projection` field
  in its JSON); `C4` canonical records are `C4_r1.json` + `C4_r2.json` (exclude `legacy/results/c4_rerun/`
  from headline tables); `wall_s` is not comparable across the two drivers for C5 (`bench_gp.py` warms
  the device before its wall clock, `run_matrix.py` includes one-time CUDA context creation) — compare
  per-call metrics.
- `iree_sr_b2.json` must never be read as a training result: it is the crash skeleton (see Known Issues).

## Validation
- R2 (no GPU needed for the validators):
  `python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages`
  → exit 0, 35 records, 126 checks, 0 failures (expected note: `iree_sr_b2` PARTIAL);
  `python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2`
  → exit 1 with exactly the 2 pre-existing `C5.json`/`C5b.json` failures.
- R2 backend pre-flight (GPU, once per backend/fixture before quoting IREE numbers): run
  `r2/check_backend_agreement.py` per backend into `new/results/r2/backend_agreement/` and compare the
  `values_sha256` fields — see `r2/backend_agreement.md`.
- Round-1 records: `new/check_smoke.py <new results>` (JSON-only) and
  `legacy/check_results.py <legacy results> --gpu 4` (add `--golden test/goldens/sr_fitness.npz` for the
  C5 golden check). To re-run a round-1 matrix: `new/run_matrix.sh` from the repo root, or
  `legacy/run_matrix.py` from the legacy checkout root with the legacy `PYTHONPATH` (needs the `.so`).
- To re-run R2: `test/benchmark/r2/run_all.sh` (optionally `--group light` first) — configs, caps, exit
  codes and wall-time expectations in `r2/README.md`.

## See Also
- `../goldens/` → frozen pre-ETL numeric captures (provenance in its `README.md`).
- `../../src/evogp/tree/ENGINE_SPEC.md` → engine semantics of record, incl. the legacy fitness convention.
