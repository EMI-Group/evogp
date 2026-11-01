# `test/benchmark/r2/round3/` — R3 round: the R2 headline configs re-measured after two engine waves

Round 3 re-measures the **R2 headline configs** with the **R2 methodology** to price two waves of
engine work that landed after R2:

1. **wave 1 — host-side cuts in `src/evogp/tree/etl_engine.py`** (host cost of decode / build /
   `sr_fitness`), and
2. **wave 2 — the dense one-graph evaluator `src/evogp/tree/dense_engine.py`**, which under the
   default `EVOGP_EVAL_MODE=auto` policy is now the **default route for the `iree` backend**
   (pop ≥ 16, and ≤ 2048 data rows on `per_row=False` calls); `numpy` still runs the specialized
   per-structure engine, because dense lost there at every measured size.

**Method = R2, unchanged.** Same harness (`test/benchmark/bench_gp.py`, untouched), same flags, same
fixtures, same wiring, same implicit `--seed 0`, same lock / `ulimit` order, one process per config.
The **ids intentionally match their R2 counterparts** (`iree_fixture64_n64` here is the same workload
as `iree_fixture64_n64` there), so `compare_rounds.py` can put the two rounds side by side and each
record can be checked against its own engine identity (`evogp_commit`, `evogp_file`, `etl_version`,
`env`, `gpu`).

**All R3 artifacts live under `test/benchmark/r2/round3/`** (`results/new/`, `results/legacy/`,
`results/backend_agreement/`, `last_run.txt`). The R2 records in `test/benchmark/new/results/r2/` and
`test/benchmark/legacy/results/r2/` are **read-only history** — nothing in this directory writes to
them, and the R2 methodology of record (`../SPEC.md`, `../README.md`, `../../REPORT.md`) stays frozen.

## Scope — 12 configs in three groups

| group | configs | what |
|---|---|---|
| `light` | `leg_c5r2`, `leg_fixture200_n64`, `leg_sr_a` + `np_c5r2`, `np_fixture200_n64`, `np_sr_a` | legacy CUDA side and the same workloads on the new `numpy` backend (`--stage-probe` on) — the R2 light-group anchors, reduced to the three workloads the round quotes |
| `iree` | `iree_fixture64_n64`, `iree_fixture200_n64`, `iree_sr_a` | `--backend iree` on the **default `auto` policy** ⇒ the dense evaluator; R2 flags kept (fixture200 budget 400, `iree_sr_a` budget 1500 + `EVOGP_ETL_CACHE_SIZE=8192`) |
| `iree-spec` | `iree_fixture64_n64_spec`, `iree_fixture200_n64_spec`, `iree_sr_a_spec` | the same three IREE workloads with **`EVOGP_EVAL_MODE=specialized`** (the R2-era route), for wave attribution |

| id | side | group | args | per-config env |
|---|---|---|---|---|
| `leg_c5r2` | leg | light | `--task eval64 --eval-reps 25` | — |
| `leg_fixture200_n64` | leg | light | `--task eval64 --fixture <200-tree> --eval-reps 25` | — |
| `leg_sr_a` | leg | light | `--task sr --pop 200 --gens 10 --tree-len 64 --reps 2` | — |
| `np_c5r2` | new | light | `--task eval64 --eval-reps 25 --stage-probe` | — |
| `np_fixture200_n64` | new | light | `--task eval64 --fixture <200-tree> --eval-reps 25 --stage-probe` | — |
| `np_sr_a` | new | light | `--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --stage-probe` | — |
| `iree_fixture64_n64` | new | iree | `--task eval64 --eval-reps 25 --backend iree --stage-probe` | — |
| `iree_fixture200_n64` | new | iree | `--task eval64 --fixture <200-tree> --eval-reps 25 --backend iree --stage-probe --max-cold-builds 400` | — |
| `iree_sr_a` | new | iree | `--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --backend iree --stage-probe --max-cold-builds 1500` | `EVOGP_ETL_CACHE_SIZE=8192` |
| `iree_fixture64_n64_spec` | new | iree-spec | as `iree_fixture64_n64` | `EVOGP_EVAL_MODE=specialized` |
| `iree_fixture200_n64_spec` | new | iree-spec | as `iree_fixture200_n64` | `EVOGP_EVAL_MODE=specialized` |
| `iree_sr_a_spec` | new | iree-spec | as `iree_sr_a` | `EVOGP_EVAL_MODE=specialized EVOGP_ETL_CACHE_SIZE=8192` |

`<200-tree>` = `test/benchmark/fixtures/forest200_tree64_random.npz`; the three legacy/numpy
`*_fixture200_*` configs guard-skip (exit 3, no JSON) when that fixture is absent.
`bash test/benchmark/r2/round3/run_all.sh --list` prints the same table plus the pre-flight guard
state, and `--dry-run` prints the exact `flock … bash -c '…'` line of every config.

## Commands (all from the **repository root**)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python        # no `python` on PATH

# --- the matrix (GPU; one bench_gp.py process per config) --------------------
test/benchmark/r2/round3/run_all.sh --list             # config table, execute nothing
test/benchmark/r2/round3/run_all.sh --dry-run          # every exact command, execute nothing
test/benchmark/r2/round3/run_all.sh --group light      # 6 configs, ~minutes
test/benchmark/r2/round3/run_all.sh --group iree       # 3 configs, IREE, auto (dense) route
test/benchmark/r2/round3/run_all.sh --group iree-spec  # 3 configs, IREE, forced specialized
test/benchmark/r2/round3/run_all.sh                    # all 12 (light → iree → iree-spec)
test/benchmark/r2/round3/run_all.sh --only np_c5r2     # exactly these ids (comma-separated)
test/benchmark/r2/round3/run_all.sh --only np_c5r2 --timeout 7200

# --- backend-agreement pre-flight (GPU; BEFORE quoting any IREE number) ------
bash test/benchmark/r2/round3/run_agreement_r3.sh              # six arms, sequentially
bash test/benchmark/r2/round3/run_agreement_r3.sh --list       # arm table
bash test/benchmark/r2/round3/run_agreement_r3.sh --dry-run    # exact commands
bash test/benchmark/r2/round3/run_agreement_r3.sh --only fixture64_numpy,fixture64_iree

# --- validators / report (no GPU anywhere in this block) --------------------
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages
python3 test/benchmark/r2/round3/check_agreement_r3.py         # the numeric gate (needs numpy)
python3 test/benchmark/r2/round3/compare_rounds.py             # R2 vs R3 tables (--markdown for SUMMARY.md)

# --- offline self-tests of the two checkers (fake records in a temp dir; no GPU) ---
python3 test/benchmark/r2/round3/selftest_check_agreement.py   # PASS/FAIL/NaN/missing/malformed cases
python3 test/benchmark/r2/round3/selftest_compare_rounds.py    # table/rep/spec-attribution cases
```
`--only` overrides `--group` (unknown ids are a usage error, exit 2); selection is always in table
order, so `light` runs before `iree` before `iree-spec`.

## Arm B — the same nine new-side configs under a patched-ETL shadow

Arm B is the round's engine-delta workstream: it re-measures the **nine new-side configs** (the three
`light` `new` ids plus `iree` and `iree-spec`; every id is its Arm A id with an `armB_` prefix) with the
same method and the same flags, differing in exactly one factor — the ETL checkout is shadowed onto
`PYTHONPATH` (`NEW_PYTHONPATH=src:<patched-checkout>` puts the patched ETL ahead of the stock editable
install; the runner defaults it to `src:/tmp/evogp_etl_patched_51a7141`). The legacy side has no Arm B row
by design (its frozen records stay the reference), and the only other delta is an explicit
`EVOGP_EVAL_MODE=auto` on the three plain `iree` ids — both are recorded in each record's `env` block.

| what | Arm B |
|---|---|
| runner | `run_all_armB.sh` (`--group iree-spec\|iree\|numpy\|all`, `--only`, `--list`, `--dry-run`) → records into `results/armB/new/` |
| manifest + record gate | `manifest_armB.txt` (9 ids, each with the `stages` marker), validated by `python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages` |
| agreement gate | `run_agreement_armB.sh` + `check_agreement_armB.py`: the same six arms plus a HARD same-stem `values_sha256` identity check against Arm A's `results/backend_agreement/` |
| records + evidence | `results/armB/new/` (the W2 records of record; W1 snapshots retrievable in git at `b728466`) and `results/armB/{backend_agreement/,etl_import_probe.txt,gate_outputs.md,windows.md,armB_runlog_section.md,status_*.txt}` |
| write-up | `SUMMARY_armB.md` (the R2 / Arm A / W1 / W2 tables, the patched-ETL attribution and both gate results) |

**Materialization caveat.** The patched checkout is not a git checkout and lives outside the repository:
it is materialized read-only from the `etl` checkout with
`git -C /mnt/local-ssd/bchuang/etl archive 51a7141 | tar -x -C <dir>` (etl commit `51a7141`, **in flight
upstream** — not yet on etl `main`), and `/tmp` is session-scoped, so `<dir>` may be gone on a fresh
session. The shadow is proven per record by the `etl:` header line of every `results/armB/new/<id>.log`
and by the dated per-file sha256 probes in `results/armB/etl_import_probe.txt`.

## Methodology carried over from R2 (and why)

* **One process per config.** The engine's `etl.CompileCache` (and the dense evaluator's own
  compiled-graph cache) are per-process, so a config's `first_call_ms` / cold-build counts are never
  inherited from an earlier config, and a crash cannot poison the next one.
* **`ulimit -n 65536` first**, then **exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per
  invocation, never nested** (other agents share that lock), with `timeout -s INT -k 30 3600s`
  **inside** the lock so lock waiting is never charged to a config. The agreement runner uses the same
  lock and, deliberately, **no** `timeout` wrapper (the driver does two `evaluate()` calls).
* **GPU 6** (`CUDA_VISIBLE_DEVICES=6`, overridable); **never GPU 7** — hardware ECC fault. Medians are
  ±10 %, a contended window inflates 2–3×.
* **Warm-median headline, compile reported separately.** Per-call numbers use `median_warm_ms`
  (= median of `per_call_ms[1:]`); per-generation numbers use `median(per_gen_ms[1:])` per rep, with
  rep0 the *churning* population and rep1 the *fully warm* one (the `../../REPORT.md` convention).
  `first_call_ms` / `cold_first_gen_ms` are quoted as compile cost, never mixed into a warm median.
* **Fixtures / wiring / seed unchanged** (`test/goldens/forest_random.npz`,
  `test/benchmark/fixtures/forest200_tree64_random.npz`, `--seed 0`), so an R2/R3 delta is engine
  work, not a workload change.
* **Stage probe on the new side** (`--stage-probe`) — see the limitation below.
* **Guards never fake data**: a missing legacy checkout/siteutils or fixture → `GUARD-SKIP`
  (exit 3, OK for the batch, **no** JSON written, so the validator still reports that id as missing).

## Gates

1. **Backend agreement (`check_agreement_r3.py`, GPU pre-flight).** Six arms — numpy / iree /
   iree+`EVOGP_EVAL_MODE=specialized` on each of the two fixtures — write
   `results/backend_agreement/<stem>.{json,_values.npy,log}`. The gate is per fixture:
   * the **two `_spec` arms are expected bit-identical to numpy** — they pin the R2-era specialized
     route, so the R2 gate (`values_sha256` equality) should still hold end to end, and
   * the **`auto` (dense-default) arms must match numpy within `rtol=atol=1e-4`, `equal_nan=True`,
     with identical NaN/±inf counts** — the dense evaluator reduces in a different order, so
     bit-identity there is neither expected nor required.
   The script's hard gate is the tolerance + count comparison for the `numpy→iree` and
   `numpy→iree_spec` pairs; `values_sha256` bit-identity is *printed for every pair*
   (numpy/iree, numpy/iree_spec, iree/iree_spec) together with the max |diff| and max rel diff, and
   the `_spec` pair is the one where an exact match is the expected outcome.   Each JSON's `fixture_sha256` must equal the fixture's recorded npz hash
   (64-tree `0bfb0bbf…9ca0`, 200-tree `6789ef61…41b9`), and the recorded `values_sha256` must equal
   the hash recomputed from its `_values.npy`. **Continuity vs R2 is INFO only**: the numpy/iree hashes
   are compared with the R2 anchors (64-tree `b8192cb1…fa02`, 200-tree `07948efe…e048`) and, when the
   R2 `_values.npy` are still on disk, the R3-vs-R2 max |diff| / max rel diff are printed — a
   *difference* there is expected (the engine changed) and never fails the gate. Exit 0 = PASS,
   1 = any gate failure or missing/malformed input. Missing stems are reported per stem, so a
   mid-flight round exits 1 with a clean message instead of a traceback.
2. **Matrix record validation.** `check_r3.py --manifest test/benchmark/r2/round3/manifest.txt
   --require-stages` checks the 12 expected JSONs (required keys per task kind, side, path, `stages`
   marker for the 9 new-side configs). It is stdlib-only; a config that has not run is reported
   `MISSING` and exits 1. The scan mode (`check_r3.py` with no arguments, or with result dirs) walks
   every `*.json` and **skips a subdirectory named `backend_agreement/`** — those are pre-flight
   records, not config results.
3. **Runner exit codes.** Both runners: `0` = every selected config/arm exited 0 or was guard-skipped
   (exit 3 counts as OK), `1` = at least one failure (timeout 124/137, interrupt 130, crash 1, …),
   `2` = usage/precondition error (not run from the repo root, unknown id/arm, missing interpreter).
   A failing config never aborts the batch.

## Known limitation — the stage probe and the dense route

The R2 stage probe (`test/benchmark/stage_probe.py`) monkeypatches the **specialized** engine's call
sites. The R3 `iree` (and, on large-N calls, other) arms can take the **dense** route, which the probe
does not instrument, so those runs may record an **empty / degenerate `stages` block** (the
`cold_builds` / `stages` columns then show `?`). That is a documented instrumentation gap, **not** a
gate failure: the `stages` marker in `manifest.txt` means "this config passes `--stage-probe`", exactly
as in R2, and the workload numbers (per-call / per-generation timings, fitness) are unaffected.

## Where results land

```
test/benchmark/r2/round3/results/new/<id>.json            test/benchmark/r2/round3/results/legacy/<id>.json
test/benchmark/r2/round3/results/new/<id>.log             test/benchmark/r2/round3/results/legacy/<id>.log
test/benchmark/r2/round3/results/backend_agreement/<stem>.json | _values.npy | .log
test/benchmark/r2/round3/last_run.txt                     # generated, .gitignore'd, never committed
```

`<id>` is the manifest id; the harness `--tag` is the uppercase id. Logs are **appended** (a re-run
adds a new dated block), the JSON is the harness's incremental snapshot (rewritten after every
step, so a hard kill still leaves usable data). `results/backend_agreement/run_agreement_r3.last_run.txt`
is the agreement runner's own id → exit-code table (generated, disposable).

## Files in this directory

| file | what it is |
|---|---|
| `run_all.sh` | the 12-config matrix runner (faithful R3 adaptation of `../run_all.sh`: same harness/lock/`ulimit` mechanics, new result dirs, three groups). |
| `run_agreement_r3.sh` | the six-arm backend-agreement pre-flight runner; drives the read-only `../check_backend_agreement.py`. |
| `manifest.txt` | the validator's expectation list: `id <TAB> side <TAB> json path <TAB> stages?`. |
| `check_r3.py` | stdlib-only record validator (copy of `../check_r2.py` with R3 defaults and branding). |
| `check_agreement_r3.py` | numpy+stdlib numeric gate over the six agreement records (identity + tolerance gate + R2 continuity INFO). |
| `compare_rounds.py` | R2-vs-R3 table generator (warm medians per task kind, per-rep training medians, wave-attribution table) — the sourcing tool for `SUMMARY.md`. |
| `selftest_check_agreement.py` | offline self-test of `check_agreement_r3.py`: builds fake arm records in a temp dir and asserts the PASS / within-tolerance / out-of-tolerance / NaN-mask / missing-stem / malformed-JSON verdicts. |
| `selftest_compare_rounds.py` | offline self-test of `compare_rounds.py`: cloned R2 records with scaled timings exercise the eval, per-rep training and `_spec` attribution tables. |
| `README.md` | this protocol document. || `SUMMARY.md` | **written after the round is executed** (verdict + tables from `compare_rounds.py` + the gate results). |

## Report / next step

Execution order for the round: `run_agreement_r3.sh` (gate 1) → `run_all.sh --group light` →
`--group iree` → `--group iree-spec` → `check_r3.py --manifest … --require-stages` →
`compare_rounds.py` → `SUMMARY.md`; Arm B repeats that chain with its own four tools
(`run_agreement_armB.sh` → `run_all_armB.sh` → `check_r3.py --manifest .../manifest_armB.txt` →
`check_agreement_armB.py` → `compare_rounds_armB.py` → `SUMMARY_armB.md`). The verdict lives in
`SUMMARY.md` (Arm A) / `SUMMARY_armB.md` (Arm B); the R2 baseline they are read against is
`../../REPORT.md`.
