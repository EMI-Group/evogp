# `test/benchmark/r2/` — round-2 records + the executed round-3 run, driven from one shared runner

## Intent
This node holds the executed round-2 measurement round (legacy CUDA engine vs the ETL "true jit" engine on numpy and IREE-cuda) as frozen, already-validated data plus the tooling that produced it.
The runner, the manifest, the validator, the agreement driver and the round log live here; the round's write-up is one level up in `../REPORT.md` (owned by the parent node), which is the stage-by-stage analysis of record and the source of the ranked optimization levers.
What this node hands to a later round is a *method* (config table, lock/`ulimit` discipline, validator contract, agreement gate), not a number to be edited.

## Files and routing table
- `run_all.sh` → the R2 matrix runner: 35 configs in two groups (`light`, `iree`), one `bench_gp.py` process per config on exactly one `flock -w 7200 /tmp/gpbench_critical.lock`, `timeout` inside the lock, exit codes 0/1/2, and `last_run.txt` for the id→exit-code table.
- `manifest.txt` → the validator's expectation list (`id <TAB> side <TAB> json path <TAB> stages?`) for those 35 ids; `stages` marks the configs run with `--stage-probe`.
- `check_r2.py` → the stdlib-only record validator (required keys per task kind, side/path/`stages` expectations, one report table); its directory scan skips a `backend_agreement/` subdirectory.
- `check_backend_agreement.py` → the read-only numeric pre-flight driver (public `evogp` API only): one JSON + one `_values.npy` per backend/fixture.
- `backend_agreement.md` → the pre-flight verdict and method of record, including the fixture npz hashes, the per-backend `values_sha256` anchors and the rule that no IREE number is quotable before this gate passes.
- `RUNLOG.md` → the executed-round operator log (per-group environment, exact commands, per-config status, FD/stage telemetry, validator output, deviations).
- `last_run.txt` → generated output of `run_all.sh` (disposable, not in the manifest).
- `round3/` → the round-3 sub-node: the R2 headline configs re-measured with the same methodology after the `etl_engine.py` host-side cuts and the `dense_engine.py` wave; entry point and gates in `round3/README.md`, node context in `round3/CONTEXT.md`; it also carries Arm B (the same nine new-side configs under a patched-ETL shadow): `round3/run_all_armB.sh`, `round3/manifest_armB.txt`, `round3/check_agreement_armB.py`, records under `round3/results/armB/`, write-up `round3/SUMMARY_armB.md`.
- `../../new/results/r2/`, `../../legacy/results/r2/` → the R2 records themselves (JSON + appended logs) with their `backend_agreement/` pre-flight evidence; each JSON carries its own engine identity (`evogp_commit`, `evogp_file`, `etl_version`, `env`, `gpu`).

## Constraints
- R2 records and this node's tooling are frozen history: never rewrite a JSON/log, never edit `../SPEC.md`, and never write round-3 output into an R2 result directory.
- All heavy phases on this shared box go through exactly one `flock -w 7200 /tmp/gpbench_critical.lock` (never nested) after `ulimit -n 65536`, on GPU 6 — never GPU 7 (hardware ECC fault).
- IREE is opt-in and non-primary: `O0`/`O1` only, always behind the agreement gate, and compile time is quoted separately from warm numbers.
- `run_all.sh` and `manifest.txt` must stay in sync by id; a config that changes in one must change in the other.
- The runner and the validator are side-agnostic and engine-agnostic: another round re-runs them by pointing `PY`/`NEW_PYTHONPATH`/`LEG_PYTHONPATH` at another checkout instead of editing the config table.

## Status
R2 is complete as records (35/35 manifest ids on disk: 34 complete plus the `iree_sr_b2` crash skeleton) and validated (`check_r2.py --manifest` → 126 checks / 0 failures).
Analysis and the headline verdict live in `../REPORT.md`; the executed-round narrative is `RUNLOG.md`.
`round3/` is executed: 12/12 config records are committed under `round3/results/{new,legacy}/` with both gates green (`check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages` → 45 checks / 0 failures; `check_agreement_r3.py` → PASS, 32 checks), its write-up is `round3/SUMMARY.md`, and its light group (`leg_*` + `np_*`) is the companion re-measurement window on GPU 0 taken after the round's first window was contended by a foreign tenant on GPU 6.
