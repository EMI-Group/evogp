# `test/benchmark/` — old-vs-new performance benchmark: legacy CUDA engine vs ETL "true jit" engine

## Intent
Completed, committed quantitative comparison of EvoGP's two evaluation engines, measured on the same shared box through the identical public API (time-only comparison; frozen methodology in `SPEC.md`).
- **OLD** = the legacy custom-CUDA engine at commit `b613503` (last pre-rewrite revision; hand-written `src/evogp/cuda/*.cu` + `evogp.evogp_cuda`), materialized and built outside the repo at `/tmp/evogp_legacy_bench`.
- **NEW** = the current ETL "true jit" engine (default `"numpy"` backend; the `"iree"` backend appears only as the explicitly non-primary C7 datapoint).
Headline result (details in `SUMMARY.md`): the legacy engine is ~34×–7,670× faster per warm generation depending on config, in exchange for a required CUDA build; the new engine needs no build step, no CUDA toolkit and device-agnostic code — and it is the only engine still in the repository.
Everything here is hand-run tooling + frozen data: nothing imports it and the repo has no CI.

## Status
Complete: both sides of the SPEC matrix are run, committed and validated — `C1` (2 reps), `C2` (partial 89/100, `projection` field in JSON), `C2b`, `C4` (2 reps), `C5`, `C5b` on both sides, plus new-side-only `C0` (`ETL_DISABLE_LOCATIONS=0` control) and `C7` (opt-in IREE).
Validators re-run over the committed results: `new/check_smoke.py` → 8 files / 160 checks / 0 failures; `legacy/check_results.py` → PASS for all 8 records; C5 golden cross-check PASS (`nan_count` 3/64, `fitness_mean` −18.048504493275626).
The old side is cross-validated through the unified harness (`xval/`): C1 deltas −0.01% wall / +5.8% warm with identical best fitness; C5 +1.9% first call / +3.9% warm with identical fitness stats — harness-independent within shared-box noise.

## Routing Table
- `README.md` → short entry point: what this directory is, headline result, contents, re-run commands, shared-host/flock note.
- `SUMMARY.md` → headline report: verdict, per-config old-vs-new table (all numbers recomputed from the raw JSONs), why/trade-offs, caveats, golden anchor + validator counts, cross-validation (§5.1), reproducibility.
- `SPEC.md` → frozen methodology (2026-09-11): workloads W1–W3, config matrix C0–C7, fairness rules, JSON schema; do not edit.
- `bench_gp.py` → the side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; used for all new-side runs and the old-side cross-validation.
- `new/` → NEW side: `run_matrix.sh` (config runner), `check_smoke.py` (validator), `NOTES.md`, `results/` (config JSONs + logs + smoke).
- `legacy/` → OLD side: `setup_legacy.sh` (build), `run_matrix.py` (driver), `check_results.py` (validator + C5 golden check), `BUILD_NOTES.md`, `NOTES.md`, `results/`.
- `xval/` → cross-validation results of the old side through `bench_gp.py` (data, helper probes, `NOTES.md`, own `CONTEXT.md`).

## Notes for Agents
- `/tmp` artifacts are session-scoped: the built legacy checkout `/tmp/evogp_legacy_bench` and the offline build assets `/tmp/evogp_t1a263` (py3.11 headers, sklearn) may be gone on a fresh session — rebuild with `legacy/setup_legacy.sh` (recipe + `.so` identity in `legacy/BUILD_NOTES.md`).
- Serialize every heavy phase (legacy build, every timed run) with exactly one `flock -w 7200 /tmp/gpbench_critical.lock`; never nest it; lock waits are outside all timed regions.
- Skip GPU index 7 (hardware ECC fault — any CUDA context on it dies); pick the least-busy GPU at run time; `CUDA_VISIBLE_DEVICES` is recorded in each JSON.
- `C2` is deliberately PARTIAL data (89/100 generations, SPEC soft cap) with a `projection` field in its JSON — not a completed run; `C4` canonical records are `C4_r1.json` + `C4_r2.json` (exclude `legacy/results/c4_rerun/` from headline tables).
- IREE (`C7`) is opt-in and explicitly non-primary: it needs raised FD limits, leaks runtime objects, and is not usable at GP scale.
- `ETL_*`/`EVOGP_ETL_*` env vars are inert on the old side (no `etl_engine` module there) — `C0`/`C7` are new-side-only by construction.
- `wall_s` is not comparable across the two drivers for C5 (`bench_gp.py` warms the device before its wall clock; `run_matrix.py` includes one-time CUDA context creation) — compare per-call metrics; likewise `evogp_commit` in a JSON is the harness's cwd repo, not the legacy checkout (identify the side via `side`/`evogp_file`).
- Shared box: medians are ±10%; a contended window can inflate runs 2–3× (evidence: `legacy/results/c4_rerun/`).

## Validation
- After touching any raw results, re-run the validators: `new/check_smoke.py <new results>` (JSON-only) and `legacy/check_results.py <legacy results> --gpu 4` (add `--golden test/goldens/sr_fitness.npz` for the C5 golden check).
- To re-run the matrix: `new/run_matrix.sh` (from the repo root) or `legacy/run_matrix.py` (from the legacy checkout root with the legacy `PYTHONPATH`; requires the built `.so`).

## See Also
- `../goldens/` → frozen pre-ETL numeric captures (provenance in its `README.md`).
- `../../src/evogp/tree/ENGINE_SPEC.md` → engine semantics of record, incl. the legacy fitness convention.
