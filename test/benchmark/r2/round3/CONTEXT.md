# `test/benchmark/r2/round3/` — R3 round after the two engine optimization waves

## Intent
Round 3 re-measures the R2 headline configs with the R2 methodology (same harness, flags, fixtures, wiring and seed) to price wave 1 (host-side cuts in `src/evogp/tree/etl_engine.py`) and wave 2 (the dense one-graph evaluator `src/evogp/tree/dense_engine.py`, now the default route for the `iree` backend under the `auto` policy).
The 12 R3 config ids intentionally match their R2 counterparts, so every record diffs directly against the untouched R2 history.
Everything R3 produces lives under this directory; nothing here writes to `../` or to any R2 result directory.
Owned artifacts: the matrix runner, the backend-agreement pre-flight runner, the manifest, the two validators, the round-comparison table generator, the protocol document and (after execution) `SUMMARY.md`.

## API surface / routing table
- `run_all.sh` → the 12-config matrix runner (`--group light|iree|iree-spec|all`, `--only`, `--list`, `--dry-run`, `--timeout`); one `bench_gp.py` process per config, results into `results/new/` and `results/legacy/`, `last_run.txt` at the node root.
- `run_agreement_r3.sh` → the six-arm backend-agreement pre-flight runner (numpy / iree / iree+`EVOGP_EVAL_MODE=specialized` × two fixtures); drives the read-only R2 driver `../check_backend_agreement.py`.
- `manifest.txt` → the validator's expectation list (`id <TAB> side <TAB> json path <TAB> stages?`) for all 12 ids; the 9 new-side ids carry the `stages` marker.
- `check_r3.py` → stdlib-only record validator (R3 copy of `../check_r2.py`); `--manifest … --require-stages` is gate (ii).
- `check_agreement_r3.py` → numpy+stdlib numeric gate over the six agreement records: per-fixture npz-identity, `values_sha256`/array integrity, NaN/±inf count equality, `np.allclose(..., rtol=atol=1e-4, equal_nan=True)` for the `numpy→iree` and `numpy→iree_spec` pairs, plus R2-anchor continuity as INFO only; exit 0 = gate PASS.
- `compare_rounds.py` → stdlib-only R2-vs-R3 table generator (eval first-call + warm medians, per-rep training medians with rep0 "churning" / rep1 "fully warm", and the `*_spec` wave-attribution table); `--markdown` for `SUMMARY.md`.
- `selftest_check_agreement.py` / `selftest_compare_rounds.py` → offline self-tests of the two checkers (fake records in a temp dir, no GPU, no repo writes).- `results/new/` and `results/legacy/` → per-config `<id>.json` + `<id>.log` (created by the runner).
- `results/backend_agreement/` → `<stem>.json` + `<stem>_values.npy` + `<stem>.log` for the six arms (skipped by the validator's directory scan by design).
- `README.md` → the protocol document (scope table, commands, methodology, gates, stage-probe limitation).
- `SUMMARY.md` → the round verdict, written after execution; not present while the round is pending.
- `.gitignore` → ignores `last_run.txt` (generated, disposable).

## Constraints
- R2 records (`../../new/results/r2/`, `../../legacy/results/r2/`) and the R2 methodology (`../SPEC.md`, `../README.md`) are read-only: never rewrite them, never add flags or change fixtures to make an R3 number nicer.
- Method is R2: one process per config, `ulimit -n 65536` first, exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per invocation with `timeout` inside it, warm median as the headline with compile/first-call reported separately, stage probe on the new side.
- GPU 6 by default, never GPU 7 (hardware ECC fault); treat medians as ±10 % on this shared box.
- The R2 stage probe patches the specialized engine's call sites only, so a dense-route run may carry an empty/degenerate `stages` block — documented limitation, not a gate failure.
- IREE stays opt-in and explicitly non-primary; `O2`/`O3` are never used, and no IREE number is quotable until the agreement gate passes.
- `last_run.txt` and the agreement runner's `run_agreement_r3.last_run.txt` are generated output: delete freely, never commit.

## Status
Scaffold in place (runner, manifest, validators, comparison tool, protocol doc) and verified without GPU execution; the round execution (agreement pre-flight + the three matrix groups) is pending.
`SUMMARY.md` is written after the run from `compare_rounds.py` output plus the two gates' results.
