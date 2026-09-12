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

#### Group `iree-spec` — EXECUTED 2026-09-12 23:09:12–23:15:32 HKT (window 2, GPU 6): 3/3 exit 0
The two other groups (`iree`, `numpy`) have **not** run yet, so the round's validator legitimately
reports their six ids as MISSING until a later phase fills them (its output is quoted at the end of this
subsection); the three `iree-spec` records below are complete and carry meaningful `stages` blocks.

**Commands (exact) — one foreground `run_all_armB.sh --only <id>` invocation per config, in run order,
each console captured to `results/armB/status_<id>.txt`; the runner owns `ulimit`/`flock`/`timeout`:**
```bash
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64_spec    # 23:09:12 → 23:09:37
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64_spec   # 23:10:01 → 23:11:31
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a_spec             # 23:11:40 → 23:15:32
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
```
No `flock` wait was observable (each config's `started` timestamp is within seconds of its invocation
and every invocation ran in the foreground of one long-budget call — the detached-`nohup` failure mode
of the agreement window was not repeated). Runner-level group wall: **6 min 20 s** (23:09:12 →
23:15:32), of which 25 s + 90 s + 232 s is inside the three configs; nothing approached the 3600 s cap.

**Per-config status table** (per-config exit code = the log's `exit code: 0 (OK)` footer *and* the
runner's own id→exit table in the same `status_<id>.txt` — the harness shell does not expose
`PIPESTATUS`, so the status files carry the runner's table rather than a separate shell line).
`first_ms` / `warm_ms` follow the metric rules at the top of this file (eval: `first_call_ms` and
`median_warm_ms` over 25 calls; `sr`: rep0's `cold_first_gen_ms` as the first figure and the per-rep
median of `per_gen_ms[1:]`):

| id | exit | complete / partial | first_ms | warm_ms (rep0 / rep1) | cold builds (json `misses_max` / stage) | fds start→cold→end | wall (started→finished) | harness_wall_s | notes |
|---|---|---|---|---|---|---|---|---|---|
| armB_iree_fixture64_n64_spec | 0 | `complete` / `partial=false` | 20 043.52 | 62.19 (25 calls) | 48 / 300 (json), stage 48 cold + 0 warm misses | 39 → 91 → 91 | 25 s (23:09:12 → 23:09:37) | 22.8 | GPU 6 foreign util 0 % before → 0 % after (≈30 GB `syfang` resident); stage block meaningful |
| armB_iree_fixture200_n64_spec | 0 | `complete` / `partial=false` | 82 065.28 | 200.09 (25 calls) | 155 / 400 (json), stage 155 cold + 0 warm misses | 39 → 91 → 91 | 90 s (23:10:01 → 23:11:31) | 87.9 | GPU 6 foreign util 64 % before → 4 % after (heaviest contention point of the group) |
| armB_iree_sr_a_spec | 0 | `complete` / `partial=false` | 64 950.88 (rep0 cold gen) | 12 889.64 / 57.32 (rep0 / rep1, median of `per_gen_ms[1:]`) | 517 / 1 500 (json), stage rep0 = 155 cold + 362 warm misses, rep1 = 0 | 39 → 91 → 91 | 232 s (23:11:40 → 23:15:32) | 228.9 | GPU 6 foreign util 0 % before → 46 % after; `best_fitness_final=-0.2510969936847687`, per-rep repair: rep0 cold 64 950.88 → rep1 cold 278.10 ms |

`stages` quality (all three are **specialized**-route runs, so the probe is meaningful by construction):
every record has `stages.partial=false`, `stages.note=""`, `installed=true`, the full
`wrapped=[decode_population,_run_tree,build_cached,run,from_numpy,evaluate,sr_fitness]` list, 25 eval
records (eval ids) / 10 + 10 train records (the `sr` id) and populated per-stage counters
(`decode / build-hit / build-miss / run / readback / upload / loss / misc`, `h2d`/`d2h`) — no
degenerate dense-route block. Stage split (console `stages[...]` lines, cold/warm totals):

| id (rep) | cold misses (ms) | warm misses (ms) | decode + build-hit (calls) + build-miss (calls) + run (calls) + readback (calls) + loss + misc | h2d / d2h |
|---|---|---|---|---|
| fixture64_spec (rep0/eval) | 48 (19 864.5) | 0 (0.0) | 38.2 + 327.5 (1 277) + 19 864.5 (48) + 1 164.7 (1 325) + 502.5 (1 325) + 21 895.0 + 55.5 | 25 / 1 325 |
| fixture200_n64_spec (rep0/eval) | 155 (81 600.2) | 0 (0.0) | 126.1 + 1 192.3 (4 120) + 81 600.2 (155) + 2 703.7 (4 275) + 1 375.1 (4 275) + 86 925.0 + 51.4 | 25 / 4 275 |
| sr_a_spec (rep0/train) | 155 (63 901.9) | 362 (159 609.0) | 36.1 + 46.8 (115) + 223 510.9 (517) + 917.1 (632) + 453.8 (632) + 225 007.7 + 816.4 | 10 / 632 |
| sr_a_spec (rep1/train) | 0 (0.0) | 0 (0.0) | 28.9 + 155.1 (632) + 0.0 (0) + 409.8 (632) + 230.7 (632) + 831.4 + 180.3 | 10 / 632 |

**Arm A twin comparison (context only — the round's verdict tables are built by the write-up phase).**
Both windows are contended and sat on GPU 6, so these are window-paired, not hardware-paired:

| id | Arm B first_ms | Arm A first_ms | Δ | Arm B warm_ms | Arm A warm_ms | Δ |
|---|---|---|---|---|---|---|
| fixture64_n64_spec | 20 043.52 | 31 485.90 | −36.3 % | 62.19 | 67.87 | −8.4 % |
| fixture200_n64_spec | 82 065.28 | 137 474.9 | −40.3 % | 200.09 | 228.50 | −12.4 % |
| sr_a_spec (rep0 / rep1 warm) | 64 950.88 (cold gen) | 142 445.2 (cold gen) | — | 12 889.64 / 57.32 | 14 741.32 / 64.66 | −12.6 % / −11.4 % |

The `sr` id's 10-generation trajectory is **identical to Arm A bitwise in best fitness**
(`per_gen_best_fitness` equal element-for-element, `best_fitness_final=-0.2510969936847687` on both
sides), i.e. the patched checkout changes no number the round cares about — consistent with the
agreement gate's six bit-identical `values_sha256` pairs.

**Validator (gate (ii), 3 of 9 ids on disk; the 6 MISSING lines are the unrun groups):**
```text
9 record(s): 3 complete, 0 partial, 0 skipped, 6 missing, 0 malformed (0 not in the manifest)
18 check(s), 6 failure(s)      # all six are `missing JSON` for armB_iree_*/armB_np_* (groups not run yet)
armB_iree_fixture64_n64_spec   new   eval64  complete   20043.52      62.19  48 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_iree_fixture200_n64_spec  new   eval64  complete   82065.28     200.09  155 (...)  0 (...)
armB_iree_sr_a_spec            new   sr      complete   64950.88   12889.64  155 (...)  362 (...)
```
Full stdout kept as `results/armB/status_check_r3_iree-spec.txt` (checker exit 1, expected while the
other two groups are unrun). The generated `results/armB/last_run.txt` holds only the **last**
invocation's table (config 3: `armB_iree_sr_a_spec new 0 OK …`, `# ok=1 skipped=0 failed=0`); it is
`.gitignore`d and not committed, which is why each invocation's table is archived in its
`status_<id>.txt` instead.
#### Group `iree` — EXECUTED 2026-09-12 23:19:02–23:21:29 HKT (window 3, GPU 6): 3/3 exit 0
This is the DENSE default route (`--backend iree` with `EVOGP_EVAL_MODE=auto` passed explicitly) — the
post-wave-2 default — and therefore the group where the R3 stage-probe caveat applies: the R2 probe
wraps only the specialized engine's call sites, so the two eval arms carry a degenerate `stages` block
(recorded verbatim below) while the `sr` arm reaches the probed call sites and reports real counters.
These three records supersede the "have not run yet" note in the `iree-spec` subsection **for the `iree`
group only**; the `numpy` group is still unrun, which is why the validator output below still lists its
three ids as MISSING.

**Commands (exact) — one foreground `run_all_armB.sh --only <id>` invocation per config, in run order,
each console captured to `results/armB/status_<id>.txt`; the runner owns `ulimit`/`flock`/`timeout`:**
```bash
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64     # 23:19:02 → 23:19:24
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64    # 23:19:35 → 23:19:51
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a              # 23:20:02 → 23:21:29
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
```
No `flock` wait was observable (a non-blocking probe held the shared lock for a moment immediately before
the group started, and each config's `started` timestamp is within seconds of its invocation). Runner-level
group wall: **2 min 27 s** (23:19:02 → 23:21:29), of which 22 s + 16 s + 87 s is inside the three configs;
nothing approached the 3600 s cap.

**Per-config status table** (per-config exit code = the log's `exit code: 0 (OK)` footer *and* the
runner's own id→exit table in the same `status_<id>.txt`; metric rules as at the top of this file —
eval: `first_call_ms` and `median_warm_ms` over 25 calls; sr: rep0's `cold_first_gen_ms` as the first
figure and the per-rep median of `per_gen_ms[1:]`):

| id | exit | complete / partial | first_ms | warm_ms (rep0 / rep1) | cold builds (json `misses_max` / stage) | fds start→cold→end | wall (started→finished) | harness_wall_s | notes |
|---|---|---|---|---|---|---|---|---|---|
| armB_iree_fixture64_n64 | 0 | `complete` / `partial=false` | 18 011.84 | 8.54 (25 calls) | 1 / 300 (json), stage 0 cold + 0 warm misses | 39 → 91 → 91 | 22 s (23:19:02 → 23:19:24) | 18.9 | GPU 6 foreign util 45 % before → 28 % after (≈30 GB `syfang` resident); dense route, degenerate stage block |
| armB_iree_fixture200_n64 | 0 | `complete` / `partial=false` | 12 227.86 | 5.20 (25 calls) | 1 / 400 (json), stage 0 cold + 0 warm misses | 39 → 91 → 91 | 16 s (23:19:35 → 23:19:51) | 13.1 | GPU 6 foreign util 44 % before → 4 % after; dense route, degenerate stage block |
| armB_iree_sr_a | 0 | `complete` / `partial=false` | 15 965.23 (rep0 cold gen) | 130.59 / 127.89 (rep0 / rep1, median of `per_gen_ms[1:]`) | 170 / 1 500 (json), stage rep0 = 0 cold + 169 warm misses, rep1 = 0 | 39 → 91 → 91 | 87 s (23:20:02 → 23:21:29) | 84.0 | GPU 6 foreign util 0 % before → 94 % after; `best_fitness_final=-0.2510969936847687`; stage block non-degenerate |

**`stages` block state, recorded verbatim (the documented dense-route gap):**
* **Two eval arms — block present, rep counters degenerate.** `stages.status` is fully populated
  (`probe=stage_probe/1`, `installed=true`, `partial=false`, `note=""`, `missing=[]`, `nested_runs=0`,
  the full `wrapped=[decode_population,_run_tree,build_cached,run,from_numpy,evaluate,sr_fitness]`
  list) and `stages.partial=false`, but the single rep-0 record carries no specialized work:
  `calls=25` with `decode_calls=0`, `build_calls=0`, `build_hits=0`, `build_misses=0`, `readback_calls=0`,
  `sr_fitness_calls=0`, `evaluate_calls=0`, `h2d_calls=0`, `d2h_calls=0`, `trees_invoked=0/400`,
  `cold_builds=0`, `warm_misses=0`; only `run_calls=25` (`run_ms=90.59` on fixture64, `79.28` on
  fixture200) is non-zero and `misc_ms=18 253.99` / `12 359.17` carries the whole probe span — i.e. the
  dense evaluator's graph execution happens outside every call site the R2 probe wraps. The JSON-level
  cold-build guard still counts the real compile (`cold_builds.misses_max=1`, budget 300/400,
  `allow_cold_builds=false`, `exceeded=false`), which is why the table's "json / stage" column reads
  `1` versus the stage block's `0`.
* **`sr` arm — block is real, not degenerate** (same shape as Arm A's and the `_spec` arm's): rep0 =
  229 `build_calls` (60 hits + 169 misses, 64 160.45 ms), 237 `run_calls` (469.13 ms), 229
  `readback_calls` (155.34 ms), 2 `decode_calls`, 2 `sr_fitness_calls`, `trees_invoked=229/400`,
  `cold_builds=0` + `warm_misses=169`; rep1 = 229 build hits / 0 misses (44.30 ms), 237 runs
  (222.13 ms), 229 readbacks (69.51 ms), `cold_builds=0` + `warm_misses=0`; `loss_ms` = `sr_fitness_ms`
  64 722.92 / 255.54 ms. Only the eval route is dense; the training route still reaches the probed
  specialized call sites.

**Arm A twin comparison (context only — the round's verdict tables are built by the write-up phase).**
Both windows sat on GPU 6 and are close in time but not simultaneous (Arm A's `iree` group ran
21:59:56–22:10:34 HKT, Arm B's 23:19:02–23:21:29 HKT), so these are window-paired:

| id | Arm B first_ms | Arm A first_ms | Δ | Arm B warm_ms | Arm A warm_ms | Δ |
|---|---|---|---|---|---|---|
| fixture64_n64 | 18 011.84 | 13 640.33 | +32.0 % | 8.54 | 10.64 | −19.8 % |
| fixture200_n64 | 12 227.86 | 13 207.38 | −7.4 % | 5.20 | 10.50 | −50.5 % |
| sr_a (rep0 cold gen / rep1 warm) | 15 965.23 (cold gen) | 16 551.84 (cold gen) | −3.5 % | 130.59 / 127.89 | 138.26 / 38.39 | −5.5 % / +233.1 % |

`src/evogp` is byte-identical between the Arm A record commit (`49c5197`) and the Arm B one
(`git diff --stat 49c5197 HEAD -- src/` is empty, as for the `iree-spec` group), so every delta above
belongs to the ETL checkout shadow and/or the window. The `sr` id's 10-generation trajectory is
**identical to Arm A bitwise in best fitness** (`per_gen_best_fitness` equal element-for-element,
`best_fitness_final=-0.2510969936847687` on both sides), and the rep *totals* move in Arm B's favour
(rep0 81 299.8 ms vs 84 780.5 ms, rep1 826.4 ms vs 953.1 ms): read the rep1 "+233 %" median together
with the bimodal-generation anomaly bullet below — it is where the 5th order statistic falls, not added
work.

**Validator (gate (ii), 6 of 9 ids on disk; the 3 MISSING lines are the unrun `numpy` group):**
```text
9 record(s): 6 complete, 0 partial, 0 skipped, 3 missing, 0 malformed (0 not in the manifest)
26 check(s), 3 failure(s)      # all three are `missing JSON` for the armB_np_* ids (group not run yet)
armB_iree_fixture64_n64        new   eval64  complete   18011.84       8.54  0 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_iree_fixture200_n64       new   eval64  complete   12227.86       5.20  0 (...)  0 (...)
armB_iree_sr_a                 new   sr      complete   15965.23     130.59  0 (...)  169 (...)
```
Full stdout kept as `results/armB/status_check_r3_iree.txt` (checker exit 1, expected while the `numpy`
group is unrun). The checker's `cold_builds` column reads the **stage block**
(`reps[0].totals.cold_builds=0` on the dense eval arms), i.e. on this group it displays the documented
dense-route gap instead of the JSON-level `misses_max` — the records themselves carry both
(`1 / 1 / 170`). The generated `results/armB/last_run.txt` holds only the **last** invocation's table
(config 3: `armB_iree_sr_a new 0 OK …`, `# ok=1 skipped=0 failed=0`); it is `.gitignore`d and not
committed, which is why each invocation's table is archived in its `status_<id>.txt` instead.

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
* **Window 2 / GPU 6 — the `iree-spec` group ran clean in one pass: 3/3 exit 0, no GUARD-SKIP, no
  timeout, no interrupt, no re-run, no kill.** Each config was one foreground `run_all_armB.sh --only`
  invocation (25 s / 90 s / 232 s wall, group total 6 min 20 s) well inside the runner's 3600 s cap and
  the platform's call budget, so the agreement window's reaped-detached-batch failure mode did not
  recur; both fixtures were present (no exit-3 guard path) and every log ends with `exit code: 0 (OK)`,
  `gpu (after)` and `finished`.
* **Window 2 / GPU 6 — host and GPU contention present, same tenant as Arm A's window.** The foreign
  `syfang` job (≈30 GB resident) was at 0 % util at window start, 64 % during `armB_iree_fixture200_n64_spec`
  and 46 % after `armB_iree_sr_a_spec`, with the host at loadavg ≈96–99 / 46 users (Arm A's iree-spec
  window: same card, tenant util 0/100/4/0, tenant ≈22 GB). Consequence to respect when reading the A/B
  table above: the first-call (−36 %/−40 %) and warm (−8 %/−13 %) deltas are **window-paired, not
  hardware-paired** — IREE per-structure compile is documented as non-reproducible ±a factor of a few on
  this box — whereas the `sr` trajectory's bitwise fitness identity is contention-independent.
* **Window 2 — the three `stages` blocks are meaningful, not degenerate.** Every record reports
  `partial=false`, `note=""`, `installed=true` and real per-stage counters for the specialized route
  (25 eval records; 10 + 10 train records), which is exactly what the `--require-stages` marker on the
  `iree-spec` manifest rows is meant to certify; the R3 caveat about a dense-route run carrying an
  empty stage block did not apply to this group.
* **Window 2 — no FD incident**: each config grew 39 → 91 fds across its cold phase and ended at 91,
  inside the known IREE range, with `ulimit -n 65536` raised by the runner first; the nanobind
  `leaked … runtime-object` teardown noise appeared in all three logs (≈4k instances on the 64-tree and
  `sr` arms, ≈12.9k on the 200-tree arm) without touching any exit code.
* **Window 3 / GPU 6 — the `iree` group ran clean in one pass: 3/3 exit 0, no GUARD-SKIP, no timeout, no
  interrupt, no re-run, no kill.** One foreground `run_all_armB.sh --only` invocation per config
  (22 s / 16 s / 87 s wall, group total 2 min 27 s), well inside the runner's 3600 s cap and the
  platform's call budget, so neither the reaped-detached-batch failure mode nor a mid-config kill
  recurred; both fixtures were present (no exit-3 guard path) and every log ends with
  `exit code: 0 (OK)`, `gpu (after)` and `finished`.
* **Window 3 / GPU 6 — GPU 6 was kept under the documented rule, decided on a series rather than on one
  snapshot.** The pre-check snapshot read GPU 6 at `0 %` (23:17:30) and `100 %` twenty seconds later, so
  an 8-sample / 4-second series was taken before the first config: `0 / 0 / 0 / 48 / 50 / 14 / 90 / 31 %`
  — the ≈30 GB `syfang` tenant oscillates and never held the sustained ≥90 % that triggers a switch,
  while GPU 0 was carrying 71–96 % of its own foreign load and was therefore not a quieter alternative;
  GPU 7 (hardware ECC fault) was never touched. Per-config contention is the `gpu (before)/(after)`
  pair in each committed log: `45 → 28 %` (fixture64), `44 → 4 %` (fixture200), `0 → 94 %` (sr) — the
  `sr` config's window overlaps the tenant's busy phase, which is the likely driver of its bimodal
  per-generation timings (next bullet). Host loadavg at window start: 109.72 / 100.65 / 102.24, 46 users.
* **Window 3 — the two dense eval arms carry a degenerate `stages` block, the `sr` arm does not**
  (verbatim contents in the `iree` subsection above). This is the R3 caveat materializing exactly as
  documented — the R2 probe wraps the specialized engine's call sites only — and it is why the manifest's
  `stages` marker means "passes `--stage-probe`", not "carries stage numbers". Gate (ii) is unaffected:
  the checker passes all six ids it can see (`9 record(s): 6 complete … 26 check(s), 3 failure(s)`, the
  three failures being the unrun `numpy` ids).
* **Window 3 — Arm B's fd telemetry stays flat on the dense route where Arm A's grew by thousands.**
  All three Arm B records report `fds 39 → 91 → 91`, while their Arm A twins reported `39 → 347 → 6 491`
  (fixture64), `39 → 347 → 6 491` (fixture200) and `39 → 347 → 9 723` (sr). Same `src/evogp` (verified:
  empty diff), same harness, same flags — the only difference is the patched checkout and the window.
  The patch most likely to explain a flat fd curve is `etl/backends/adapters/iree.py`'s per-thread
  runtime-HAL-device cache (fewer devices created ⇒ fewer runtime objects and fds); stated here as the
  window-paired observation it is, with the mechanism as inference, not as a measured hardware constant.
  The nanobind teardown noise shrinks in the same direction (`nanobind: leaked 677 / 677 / 1 820
  instances` for the fixture64 / fixture200 / sr configs, against the `_spec` twins' ≈4k / ≈12.9k / ≈4k),
  and no exit code was affected by either.
* **Window 3 — the `sr` arm's per-generation cost is bimodal, so its rep-1 median is not a stable
  comparison point.** rep-1 `per_gen_ms[1:]` is `[154.3, 127.9, 27.1, 28.2, 25.6, 27.9, 134.9, 130.8,
  134.3]` (median 127.89) on Arm B versus `[190.6, 265.6, 142.6, 140.2, 34.5, 38.4, 37.1, 34.1, 34.2]`
  (median 38.39) on Arm A: both are two-cluster distributions (≈26–38 ms vs ≈128–266 ms) and the median
  lands on opposite sides of the gap, so the "+233 %" cell in the A/B table is an artifact of where the
  5th order statistic falls. The like-for-like reads say the patched checkout added no work — identical
  stage counters (229 build calls, 60 hits + 169 misses, 237 runs per rep), `warm_misses=0` on both
  rep-1s, rep totals 81 299.8 ms → 826.4 ms (Arm B) against 84 780.5 ms → 953.1 ms (Arm A), and an
  element-for-element identical `per_gen_best_fitness` — so any write-up must not quote that median as a
  regression.
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
above. The `iree-spec` matrix phase (window 2) committed `144f1fe`
(`etl_import_probe.txt` third block + `windows.md` window 2), `fc4e9dd` + `6c0a653`
(`armB_iree_fixture64_n64_spec.{json,log}` + its `status_*.txt`; the `*.log` needed `git add -f`
because `.gitignore:32` hides `*.log` and Arm A's records were force-added the same way), `1795c0d`
(`armB_iree_fixture200_n64_spec.{json,log}` + `status_*.txt`), `72420d1`
(`armB_iree_sr_a_spec.{json,log}` + `status_*.txt`) and the follow-up commit carrying
`results/armB/status_check_r3_iree-spec.txt`, this matrix subsection and the four window-2 anomaly
bullets. The `iree` matrix phase (window 3) committed `ca413f3` (`etl_import_probe.txt` fourth block +
`windows.md` window 3), `8ee004c` (`armB_iree_fixture64_n64.{json,log}` + `status_*.txt`), `a59dac2`
(`armB_iree_fixture200_n64.{json,log}` + `status_*.txt`), `5fff188`
(`armB_iree_sr_a.{json,log}` + `status_*.txt`; the `*.log` files again needed `git add -f` because
`.gitignore:32` hides `*.log`, exactly as for Arm A's records) and the follow-up commit of that phase,
which carries `results/armB/status_check_r3_iree.txt`, the `iree`-group subsection above and the six
window-3 anomaly bullets.
