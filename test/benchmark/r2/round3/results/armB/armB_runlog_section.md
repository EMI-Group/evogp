---
## Arm B (9 configs: numpy + IREE auto/spec on the PATCHED ETL checkout) — <WINDOW: date + time range>
**Operator:** <fill in when the Arm B windows run; later phases APPEND to this section — never edit the
paragraphs above them, and never touch the R2/R3 sections above this one>.
**Scope:** `test/benchmark/r2/round3/**` only — the Arm B tooling (`run_all_armB.sh`,
`run_agreement_armB.sh`, `manifest_armB.txt`, `check_agreement_armB.py`,
`selftest_check_agreement_armB.py`), the nine Arm B configs
(`round3/results/armB/new/<id>.{json,log}`), the six Arm B agreement arms
(`round3/results/armB/backend_agreement/`) and the Arm B evidence files
(`round3/results/armB/{etl_import_probe.txt,windows.md,armB_runlog_section.md}`); no Arm A (`results/new/`,
`results/legacy/`, `results/backend_agreement/`), no R2 record, no file above `round3/`, no
harness/runner/validator/`src/**` file is touched by Arm B.
**Method and contract:** Arm A's method, unchanged (same harness `bench_gp.py`, same flags, same
fixtures, same wiring, same implicit `--seed 0`, same lock/`ulimit` order, one `bench_gp.py` process
per config, warm median as the headline with compile/first-call reported separately, stage probe on
the new side) — the ONLY deltas are the ETL shadow and the explicit `EVOGP_EVAL_MODE` (see
`### Deviations / anomalies`). Contract/ids: `round3/manifest_armB.txt`; runners
`round3/run_all_armB.sh` (matrix) and `round3/run_agreement_armB.sh` (pre-flight); gates
`round3/check_r3.py --manifest round3/manifest_armB.txt --require-stages` and
`round3/check_agreement_armB.py`. Arm B ids are their Arm A/R2 counterparts with an `armB_` prefix, so
every record diffs directly against `round3/results/new/` (Arm A) and `../new|legacy/results/r2/` (R2).
Raw evidence stays in the per-config JSONs + logs — a log is append-only, so a re-run adds a dated
block, while the JSON is the harness's incremental snapshot and holds the latest window.
### Environment
| item | value |
|---|---|
| repo root (cwd for every command) | repository root of each operator's worktree; the Arm B tooling window ran in the tooling worktree (every JSON carries its own `cwd`) |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |
| **ETL engine (Arm B's only engine delta)** | `PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141` — the patched checkout shadows the stock editable install (`/mnt/local-ssd/bchuang/etl`). Verified in the tooling window: shadow → `/tmp/evogp_etl_patched_51a7141/etl/__init__.py`, control → `/mnt/local-ssd/bchuang/etl/etl/__init__.py` (`results/armB/etl_import_probe.txt`), and per config by the `etl:` header line of every `results/armB/new/<id>.log`. **The JSON `etl_version` is `"0.1.0"` for BOTH stock and patched**, so the probe file + the per-log `etl:` line (not the JSON) are the shadow evidence. The patched checkout differs from stock in 3 source files (`etl/backends/adapters/iree.py` per-thread runtime-HAL-device cache, `etl/build_cache.py` content-fingerprint fast path, `etl/persist/codec.py` `has_custom_codecs()`) plus `CONTEXT.md` docs; it is NOT a git checkout, so the per-file sha256 in `etl_import_probe.txt` is the revision handle. Both checkouts are READ-ONLY for this round: nothing in `round3/**` writes to `/tmp/evogp_etl_patched_51a7141` or `/mnt/local-ssd/bchuang/etl` |
| backends | `numpy` (default, 3 configs) and `iree` (`EVOGP_ETL_IREE_DEVICE=0`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`, 6 configs) — identical to Arm A; IREE stays opt-in, non-primary, `O2`/`O3` never used, and no IREE timing is quotable before the Arm B agreement gate passes |
| eval policy | `EVOGP_EVAL_MODE=specialized` on the 3 `iree-spec` ids, `EVOGP_EVAL_MODE=auto` **explicit** on the 3 `iree` ids (Arm A left `auto` implicit), `EVOGP_ETL_CACHE_SIZE=8192` on both `*_sr_a*` ids — all passed per config via `env VAR=...`, never globally, and recorded in each JSON's `env` block |
| legacy side | **absent by design** — Arm B has no legacy config (the frozen legacy CUDA records in `results/legacy/` and `../legacy/results/r2/` stay the reference); the `LEG_PYTHONPATH` machinery in `run_all_armB.sh` is inert and kept only for env-line format parity |
| GPU | `<fill in per window: CUDA_VISIBLE_DEVICES + which card the runner defaulted to>`; runner default 6; **GPU 7 (hardware ECC fault) never touched**; per-window all-GPU snapshots + contention notes in `results/armB/windows.md` |
| foreign tenant / host load | tooling window (2026-09-12 14:49 UTC): host loadavg 84.58 105.73 103.66, GPU 6 already holding ≈30 GB (`syfang`, pid 2246336, up 2 h 54 m) — the same tenant class as Arm A's first window; `<fill in per heavy window>` |
| lock / limits | `/tmp/gpbench_critical.lock` — exactly one `flock -w 7200` per invocation (never nested), `ulimit -n 65536` first, `timeout -s INT -k 30 3600` inside the lock for each matrix config, one `bench_gp.py` process per config; the agreement runner drives one driver process per arm with no timeout wrapper |
| fixtures | `test/goldens/forest_random.npz` (64 trees, sha256 `0bfb0bbf…9ca0`) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, sha256 `6789ef61…41b9`) — the same two Arm A used; fixture guard: `<fill in: GUARD-SKIP or not>` |
### Commands (exact)
```bash
# tooling window (no GPU work; the Arm B files below were created here)
bash -n test/benchmark/r2/round3/run_all_armB.sh
bash -n test/benchmark/r2/round3/run_agreement_armB.sh
python3 -m py_compile test/benchmark/r2/round3/check_agreement_armB.py
bash test/benchmark/r2/round3/run_all_armB.sh --list
bash test/benchmark/r2/round3/run_all_armB.sh --dry-run
bash test/benchmark/r2/round3/run_agreement_armB.sh --list
bash test/benchmark/r2/round3/run_agreement_armB.sh --dry-run
python3 test/benchmark/r2/round3/selftest_check_agreement_armB.py
# heavy windows (to be filled in when run — one line per group, in execution order)
# bash test/benchmark/r2/round3/run_agreement_armB.sh              # gate (i): six arms
# bash test/benchmark/r2/round3/run_all_armB.sh --group iree-spec
# bash test/benchmark/r2/round3/run_all_armB.sh --group iree
# bash test/benchmark/r2/round3/run_all_armB.sh --group numpy
# gates (no GPU)
# python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
# python3 test/benchmark/r2/round3/check_agreement_armB.py
```
### Agreement pre-flight (gate (i): six arms on the patched ETL checkout)
<PLACEHOLDER — to be filled by the pre-flight phase. Expected shape: `run_agreement_armB.sh` →
6/6 arms exit 0, records + `run_agreement_armB.last_run.txt` under
`results/armB/backend_agreement/`; then `python3 test/benchmark/r2/round3/check_agreement_armB.py`
→ `gate: PASS` with (a) npz/array integrity, (b) numpy-vs-iree[+spec] `np.allclose(rtol=atol=1e-4,
equal_nan=True)` and identical NaN/±inf counts per fixture, and **(c) the HARD Arm A/Arm B identity
gate: all 6 same-stem pairs bit-identical (`values_sha256`) to `results/backend_agreement/`**. Until
that passes, no Arm B IREE number is quotable.>
### Matrix groups (iree-spec, iree, numpy — 9 configs)
<PLACEHOLDER — to be filled by the matrix phases. Expected shape: per group, the
`run_all_armB.sh --group <g>` invocation, the window id/GPU from `windows.md`, the per-config
status table (exit code, `complete`, `partial`, `first_ms`, `warm_ms`), the wall clock, and the
`results/armB/last_run.txt` id→exit-code table (generated + `.gitignore`d, so quote it rather than
commit it). `warm_ms` conventions are the ones at the top of this file; for `sr` ids report the
per-rep median of `per_gen_ms[1:]` as `rep0 / rep1`.>
### `iree_sr_b2` re-attempt (pop 1000 / gens 2 / tree 128 — not an Arm B matrix id)
<PLACEHOLDER — to be filled if the phase runs it. This is the one R2 config that FAILED
(silent native SIGSEGV, runner exit 139, after ≈800–900 builds and before generation 0 completed —
R2's bounded-failure datapoint `../new/results/r2/iree_sr_b2.json`, kept as a crash skeleton); the
re-attempt tests whether the patched ETL checkout (per-thread runtime-device cache / `build_cache`
fast path) survives it, with Arm A's flags unchanged:
`--task sr --pop 1000 --gens 2 --tree-len 128 --reps 2 --backend iree --stage-probe
--max-cold-builds 1500` + `EVOGP_ETL_CACHE_SIZE=8192`, `ulimit -n 65536` and FD telemetry watched
(`fds.end`). Record the exit code, whether the crash reproduces, the build count at failure and the
FD trajectory; a crash is again a bounded-failure datapoint, never a training result.>
### Deviations / anomalies
**Tooling-window decisions carried into every Arm B run (recorded here once, on purpose):**
* **`EVOGP_EVAL_MODE=auto` is passed EXPLICITLY on the three plain `iree` arms** (`armB_iree_fixture64_n64`,
  `armB_iree_fixture200_n64`, `armB_iree_sr_a`), where Arm A relied on the harness/engine default. Both
  arms then differ only in the ETL checkout shadow — that is the point of Arm B. The value is recorded
  in each JSON's `env` block, so an Arm A/Arm B diff is checkable, not assumed.
* **`--only` matching is case-insensitive** in `run_all_armB.sh` (Arm B ids carry a capital `B`, while
  Arm A's all-lowercase ids could simply be case-folded): `--only armB_np_sr_a` and
  `--only armb_np_sr_a` select the same config. Everything else about `--only`/`--group`/`--list`/
  `--dry-run`/`--timeout`/exit codes is Arm A's behavior.
* **Arm B artifacts live under new paths only** (`results/armB/new/`, `results/armB/backend_agreement/`,
  `results/armB/last_run.txt`) and the header/usage text says so; the unused `results/armB/legacy/`
  path exists for layout parity only.
* **No legacy config rows** and no legacy guard branch (the frozen legacy side is the reference, not a
  re-measurement target); the fixture guard is kept unchanged, and each config's/arm's log header gains
  the `etl:` shadow-proof line (the one addition to the agreement runner's log header, whose JSON records
  neither PYTHONPATH nor the ETL path — Arm A's `env:` line only proves the PYTHONPATH string).
* **`check_agreement_armB.py` anchors on Arm A, not on R2**: the R3 checker's INFO-only "vs the R2
  anchor" continuity is replaced by a HARD same-stem `values_sha256` identity gate against
  `results/backend_agreement/` (6 pairs, including the `_spec` arms Arm A recorded but the R2 pre-flight
  never had); a mismatch or a missing Arm A record fails the checker.
**Room-window anomalies (append below, one bullet per anomaly, with the window/GPU and the configs affected):**
* <PLACEHOLDER — e.g. contention inflation vs Arm A, lock waits, GUARD-SKIPs, timeouts, partial
  configs, FD growth, nanobind teardown noise, stage-probe degeneracy on the dense route, and anything
  that made a number non-comparable.>
**Analysis of record:** Arm B's write-up goes here once the round runs; the baseline it is read against
is `round3/SUMMARY.md` (Arm A) and, one level up, `../REPORT.md` (R2) — Arm B's headline question is
whether the patched ETL checkout changes any R3 number or fitness vector, so the tables to produce are
Arm A-vs-Arm B (warm + first-call per config) alongside the Arm B agreement gate output.
**Commits this section covers:** `<fill in per phase; the tooling phase's commit(s) create
run_all_armB.sh, run_agreement_armB.sh, manifest_armB.txt, check_agreement_armB.py,
selftest_check_agreement_armB.py and results/armB/{etl_import_probe.txt,windows.md,armB_runlog_section.md}>`.
