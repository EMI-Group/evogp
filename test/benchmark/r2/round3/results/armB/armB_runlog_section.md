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
| GPU | agreement pre-flight window (2026-09-12 15:01 UTC): `CUDA_VISIBLE_DEVICES=6` — the runner default, kept because GPU 6's snapshot showed **0 % utilization** (≈30 GB of `syfang`'s job resident, i.e. memory pressure but no compute load), one device for all six arms; **GPU 7 (hardware ECC fault) never touched**; per-window all-GPU snapshots + contention notes in `results/armB/windows.md` (window 1) |
| foreign tenant / host load | tooling window (2026-09-12 14:49 UTC): host loadavg 84.58 105.73 103.66, GPU 6 already holding ≈30 GB (`syfang`, pid 2246336, up 2 h 54 m) — the same tenant class as Arm A's first window; agreement pre-flight window (2026-09-12 15:01 UTC): loadavg 88.66 99.35 103.97 with 46 users, GPU 6 = same `syfang` tenant at ≈30 GB whose utilization fluctuated 0 → 97 % during the batch (its `gpu (before)/(after)` lines are in every arm log) — correctness gate, so contention costs wall time only |
| lock / limits | `/tmp/gpbench_critical.lock` — exactly one `flock -w 7200` per invocation (never nested), `ulimit -n 65536` first, `timeout -s INT -k 30 3600` inside the lock for each matrix config, one `bench_gp.py` process per config; the agreement runner drives one driver process per arm with no timeout wrapper |
| fixtures | `test/goldens/forest_random.npz` (64 trees, sha256 `0bfb0bbf…9ca0`) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, sha256 `6789ef61…41b9`) — the same two Arm A used; fixture guard: **no GUARD-SKIP** — both fixtures were present, so all six arms ran (the guard's exit-3 path was never taken) |
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
**EXECUTED 2026-09-12 23:01:00–23:04:05 HKT (windows 1/1b, GPU 6). Verdict: `gate: PASS` — 38 checks, 0
failures; every one of the six Arm B arms reproduces its same-stem Arm A `values_sha256` bitwise, so
the patched ETL checkout changes no fitness vector and Arm B IREE numbers are quotable under the same
rule as Arm A's.** Verbatim checker stdout + exit code: `results/armB/gate_outputs.md` (gate run 1).

**Commands (exact), one invocation for all six arms:**
```bash
bash test/benchmark/r2/round3/run_agreement_armB.sh       # repo root, all six arms sequentially
python3 test/benchmark/r2/round3/check_agreement_armB.py  # gate checker (stdlib + numpy, no GPU)
```

**Per-arm exit codes** (`results/armB/backend_agreement/run_agreement_armB.last_run.txt`, invocation
2026-09-12T23:04:05+08:00, `CUDA_VISIBLE_DEVICES=6`, `new_pythonpath=src:/tmp/evogp_etl_patched_51a7141`):

| arm | exit | status | wall (started→finished, arm log) | driver call 1 / call 2 | values_sha256 (12) |
|---|---|---|---|---|---|
| fixture64_numpy | 0 | OK | 23:01:01 → 23:01:04 (3 s) | 190.8 ms / 25.2 ms | 2e5c0fa1fdbe… |
| fixture64_iree | 0 | OK | 23:01:04 → 23:01:21 (17 s) | 13 449.9 ms / 8.1 ms | b8192cb1770b… |
| fixture64_iree_spec | 0 | OK | 23:01:21 → 23:01:46 (25 s) | 22 729.1 ms / 136.9 ms | 2e5c0fa1fdbe… |
| fixture200_numpy | 0 | OK | 23:01:46 → 23:01:51 (5 s) | 1 029.6 ms / 95.0 ms | 16d5bbe98481… |
| fixture200_iree | 0 | OK | 23:01:51 → 23:02:08 (17 s) | 13 558.5 ms / 7.5 ms | 07948efededc… |
| fixture200_iree_spec | 0 | OK | 23:02:08 → 23:04:05 (117 s) | 114 329.5 ms / 236.0 ms | 16d5bbe98481… |

`# ok=6 skipped=0 failed=0` — no GUARD-SKIP (both fixtures present), no timeout, no interrupt, and
**no arm was retried**: this six-arm pass is the batch's first successful pass. **Wall clock total:
3 min 5 s** (23:01:00 → 23:04:05 HKT) for the whole batch, of which ≈166 s is the six `call 1/2` cold
paths (= the first `problem.evaluate`, IREE-compile-dominated) — the same `call 1` paths on the three
numpy arms together total ≈1.2 s. Both
rows are wall-clock context only, not benchmark numbers — per the R2 caveat, IREE per-structure
compile on this box spreads by several ×, and the same-window readback is what the gate is built on.
(Side note for the later A/B tables: Arm A's *first* window took 12 min 28 s for the same six arms
under comparable contention, but its per-arm compile split differs in both directions — dense route
slightly cheaper here, the specialized 200-tree arm slower — so this is not a speed claim.)

**Checker output (key blocks; the full verbatim stdout is in `results/armB/gate_outputs.md`):**
```text
== fixture64  (test/goldens/forest_random.npz) ==
numpy      numpy  -                64   3 0   2e5c0fa1fdbe…  npy==json ok
iree       iree   O1 dev=0         64   3 0   b8192cb1770b…  npy==json ok
iree_spec  iree   O1 dev=0 spec    64   3 0   2e5c0fa1fdbe…  npy==json ok
gate: NaN/±inf counts identical across numpy/iree/iree_spec (nan=3, inf=0)
gate fixture64: numpy vs iree              PASS  max|diff|=7.62939e-06 max rel diff=1.08241e-07  bit-identical=False
gate fixture64: numpy vs iree_spec         PASS  max|diff|=0 max rel diff=0  bit-identical=True
gate fixture64: fixture64_numpy vs Arm A   PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture64: fixture64_iree vs Arm A    PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture64: fixture64_iree_spec vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0
== fixture200  (test/benchmark/fixtures/forest200_tree64_random.npz) ==
numpy      numpy  -               200  12 0   16d5bbe98481…  npy==json ok
iree       iree   O1 dev=0        200  12 0   07948efededc…  npy==json ok
iree_spec  iree   O1 dev=0 spec   200  12 0   16d5bbe98481…  npy==json ok
gate: NaN/±inf counts identical across numpy/iree/iree_spec (nan=12, inf=0)
gate fixture200: numpy vs iree             PASS  max|diff|=0.00390625 max rel diff=2.02762e-07  bit-identical=False
gate fixture200: numpy vs iree_spec        PASS  max|diff|=0 max rel diff=0  bit-identical=True
gate fixture200: fixture200_numpy vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture200: fixture200_iree vs Arm A  PASS  bit-identical=True  max|diff|=0 max rel diff=0
gate fixture200: fixture200_iree_spec vs Arm A PASS  bit-identical=True  max|diff|=0 max rel diff=0
== Arm A vs Arm B identity (HARD gate) ==
  INFO fixture64/numpy: Arm B 2e5c0fa1fdbe… vs Arm A 2e5c0fa1fdbe… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture64/iree: Arm B b8192cb1770b… vs Arm A b8192cb1770b… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture64/iree_spec: Arm B 2e5c0fa1fdbe… vs Arm A 2e5c0fa1fdbe… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/numpy: Arm B 16d5bbe98481… vs Arm A 16d5bbe98481… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/iree: Arm B 07948efededc… vs Arm A 07948efededc… -- bit-identical (max|diff|=0 max rel diff=0)
  INFO fixture200/iree_spec: Arm B 16d5bbe98481… vs Arm A 16d5bbe98481… -- bit-identical (max|diff|=0 max rel diff=0)
gate: PASS  (38 check(s), 0 failure(s))
  Arm B IREE timings are quotable for every fixture above (within the recorded tolerance), and every Arm B arm reproduces its Arm A fitness vector bitwise.
```
(The table rows above are the checker's own table with its `npz_sha256` column elided to fit; each
arm's npz hash was checked against the fixture it loaded — `0bfb0bbf…` / `6789ef61…` — and every
`values_sha256` was recomputed from the `<stem>_values.npy` next to its JSON: `npy==json ok` on all
six.) Reading: (b) the dense `iree` arms agree with the `numpy` reference at `max|diff|` 7.6e-06
(fixture64) and 3.9e-03 at `max rel diff` 2.0e-07 (fixture200) — inside the recorded
`rtol=atol=1e-4, equal_nan=True`, with the same 3/N and 12/N NaN placement and no ±inf; (c) the HARD
Arm A→Arm B gate, which the R3 checker does not have, shows **0 drift on all 6 pairs**.

**Window / contention notes (window 1 in `windows.md`):** GPU 6 was kept because it was idle at
snapshot time (0 % util, ≈30 GB of `syfang`'s job resident); the arms' `gpu (before)/(after)` lines
record that tenant oscillating 0 → 97 % during the batch, with the host at loadavg ≈88–126 and 46
users. Every arm took exactly one `flock -w 7200 /tmp/gpbench_critical.lock` (no lock wait visible —
the batch moved arm-to-arm with no gap), `ulimit -n 65536` was raised by the runner first, and the
IREE arms ended with the usual nanobind `leaked … runtime-object` teardown noise (≈56 records on the
64-tree arms → ≈1 033 on the 200-tree arms) without changing any exit code. Nothing in the window is a
benchmark number: this gate decides *quotability*, not speed.
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
* **Window 1 / GPU 6 — the batch's first launch (22:54) was killed by its *parent shell*, not by the
  runner or an arm.** The batch was started detached (`nohup … &` + polling) from a command whose own
  execution budget expired ~10 s in, and the platform reaped the whole process group with it: the
  console froze at 22:54:24 mid-`fixture64_iree` (header + `Forest(...)` rebuild printed, no
  `call 1/2`), no `python` process of this user remained on any GPU, and no further arm log line
  appeared. Impact: the first `fixture64_numpy` arm had already finished cleanly (exit 0, hash
  `2e5c0fa1…`), and the interrupted `fixture64_iree` arm left a *header-only* block in its log (no
  exit code). The batch was then re-launched **in the foreground of one long-budget invocation**
  (`bash test/benchmark/r2/round3/run_agreement_armB.sh`, 23:01:00 → 23:04:05) and all six arms ran to
  completion in that single pass. The runner is idempotent per arm and the workload is RNG-free (forest
  rebuilt from the stored arrays), so the re-run re-measured the same two fixtures: `fixture64_numpy`'s
  hash is identical across the two passes (`2e5c0fa1…`), and the interrupted `fixture64_iree` was
  completed by the second pass — i.e. **an infrastructure kill, not an arm failure, and no arm was
  retried after a failure** (the arm logs are append-only, so both passes are visible: `fixture64_numpy`
  shows 22:54:18–22:54:22 then 23:01:01–23:01:04, `fixture64_iree` shows a header-only 22:54 block then
  the 23:01:04–23:01:21 block with `exit code: 0`).
* **Window 1 / GPU 6 — host/GPU contention is present but did not affect the verdict.** The foreign
  tenant `syfang` (≈30 GB resident, up ≈3 h) oscillated between 0 % and 97 % utilization while the six
  arms ran at host loadavg ≈88–126 / 46 users, so the cold-call wall times in the table above carry the
  usual shared-box spread (and Arm A's own compile times are documented as non-reproducible). Because
  the gate compares *fitness vectors and NaN/±inf counts*, not times, this contention is a wall-clock
  caveat only — the window's numbers stay quotable as a correctness result.
* **No GUARD-SKIP, no timeout, no partial record, no FD-growth incident** in the six-arm pass: both
  fixtures were present, `ulimit -n 65536` was raised by the runner before any arm, every arm exited 0
  (`ok=6 skipped=0 failed=0`), and the nanobind teardown warnings on the IREE arms are the known,
  harmless noise (no `leaked …` line changed an exit code).
**Analysis of record:** Arm B's write-up goes here once the round runs; the baseline it is read against
is `round3/SUMMARY.md` (Arm A) and, one level up, `../REPORT.md` (R2) — Arm B's headline question is
whether the patched ETL checkout changes any R3 number or fitness vector, so the tables to produce are
Arm A-vs-Arm B (warm + first-call per config) alongside the Arm B agreement gate output.
**Commits this section covers:** the tooling phase's commit(s) created `run_all_armB.sh`,
`run_agreement_armB.sh`, `manifest_armB.txt`, `check_agreement_armB.py`,
`selftest_check_agreement_armB.py` and `results/armB/{etl_import_probe.txt,windows.md,armB_runlog_section.md}`
(`c926934` tooling, `759c455` evidence scaffold); the agreement pre-flight phase committed its six-arm
records + `run_agreement_armB.last_run.txt` together with the dated `etl_import_probe.txt` second block
(the agreement pre-flight probe block) and `windows.md` (window 1 + the executed-batch window 1b)
appends in `6471160`, and the follow-up commit of that phase adds
`results/armB/gate_outputs.md` plus the filled-in Environment rows, this subsection and the anomalies
above.
