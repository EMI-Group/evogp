# SUMMARY.md — round 3: the R2 headline configs re-measured after the two engine waves

**What this is.** Round 3 re-measures the **12 R2 headline config ids** on the same shared box, through
the same harness (`test/benchmark/bench_gp.py`) and with the **R2 method unchanged** (same flags,
fixtures, wiring, implicit `--seed 0`, `ulimit`/lock order, one process per config, warm-median headline
with compile reported separately), after two waves of engine work:

* **wave 1** — host-side cuts in `src/evogp/tree/etl_engine.py` (host cost of decode / build /
  `sr_fitness`), and
* **wave 2** — the dense one-graph evaluator `src/evogp/tree/dense_engine.py`, which under the default
  `EVOGP_EVAL_MODE=auto` policy is now the **route the `iree` backend takes** (pop ≥ 16, `per_row=False`
  rows ≤ 2048, ≤ 32-node trace length; longer trees and unsupported shapes fall back to the specialized
  engine). `numpy` still runs the specialized per-structure engine — `auto` never selects dense there.

The ids deliberately match their R2 counterparts, so every record diffs against untouched R2 history.
Protocol of record: `./README.md`. R2 baseline: `../REPORT.md`. Every number below is recomputed from
the committed R3 records (`./results/{new,legacy}/<id>.json`) with `./compare_rounds.py`, or quoted from
the two gates' own output — nothing is invented. The round was measured in **two windows** (§4): read
the tables with that in mind.

## 1. Verdict

* **Wave 2 is the decisive win, and it is an `iree` win.** Warm `iree` per-call cost drops
  69.61 → **10.64 ms** (fixture64, −84.7%) and 229.49 → **10.50 ms** (fixture200, −95.4%); the fully
  warm `sr_a` generation drops 65.31 → **38.39 ms** (−41.2%) and the churning rep0 generation
  10 733.32 → **138.26 ms** (−98.7%). The compile-dominated first calls drop 82 065 → **13 640 ms**
  (−83.4%) and 105 646 → **13 207 ms** (−87.5%), and the per-process cold-build counter collapses from
  48 / 155 / 517 to **1 / 1 / 170** misses (the probe sees 0 `etl` build misses on the two eval configs).
* **Wave 1 is ~neutral on the route it touched.** The forced-specialized arms (R2-era route, wave-1
  code, same window) land on R2's numbers: fixture64 67.87 vs 69.61 ms (**−2.5%**), fixture200 228.52 vs
  229.49 (**−0.4%**), `sr_a` rep1 64.66 vs 65.31 (**−1.0%**), with identical build counts (48/155/517)
  and identical fitness. Within the R3 window the auto (dense) route is **6.38× / 21.77×** cheaper per
  warm eval call than the same code forced onto the specialized route, and **106.6×** cheaper on the
  churning `sr_a` generation — that is wave 2, measured against wave 1 in one window.
* **The legacy anchor is re-confirmed.** The companion window reproduces R2 exactly — 0.6323 vs
  0.6327 ms (fixture64), 0.6357 vs 0.6319 ms (fixture200), 1.58 vs 1.59 ms/gen (`sr_a` rep1). W1's
  ~3× inflation of the two launch-bound legacy evals was the window, not the engine; **the W2 legacy
  values are the R3 legacy headline.**
* **The old-vs-new gap on fixed-forest evaluation shrinks 6.5× / 22×**: legacy/iree 110.0× → **16.8×**
  (fixture64) and 363.2× → **16.5×** (fixture200); on the fully warm `sr_a` generation the much smaller
  gap goes 41.1× → **24.3×** (1.7× compression). The dense route is now
  *faster than the CPU interpreter* on both fixed-forest evals (2.19× / 11.2×), an ordering flip from
  R2's 3.31× / 2.45× in numpy's favour, and it is within 1.18–1.25× of numpy on `sr_a` (R2: 127.6× rep0,
  2.96× rep1). Legacy/numpy did **not** improve (36.8× / 185.0× / 19.5× vs R2's 33.3× / 148.2× / 13.9×):
  no engine-side shift is attributed there, those rows ride the shared-host noise band (§4).
* **Nothing moved numerically.** The agreement gate passes (32 checks): numpy ↔ specialized bit-identical,
  auto-dense within `rtol=atol=1e-4` (max |diff| 7.63e-06 / 3.91e-03, max rel ≤ 2.03e-07), identical
  NaN/±inf masks (3/64, 12/200), and the auto-dense values **bit-identical to the R2 anchors**. The `sr_a`
  fitness trajectories are identical to R2 *generation-for-generation*, in both windows and both reps
  (legacy −0.2257394939661026, new −0.2510969936847687): the workload did not change, only the clock.
* **No crash, no new failure mode.** 12/12 records complete (45 checks, 0 failures), no SIGSEGV, and a
  lower descriptor cost on the long `sr_a` run (9723 at exit vs 17243 for the per-structure route,
  §6). Caveat: the R3 matrix contains no pop-1000 config, so R2's `iree_sr_b2` crash datapoint is
  neither reproduced nor retired here.

## 2. Headline tables (as emitted by `compare_rounds.py --markdown`)

### 2.1 Core eval configs — first call (compile) and warm per-call median [ms]

| id | side | R2 first_ms | R2 warm_ms | R3 first_ms | R3 warm_ms | warm delta | R2 status | R3 status |
|---|---|---|---|---|---|---|---|---|
| leg_c5r2 | leg | 20.62 | 0.63 | 26.49 | 0.63 | -0.1% | ok | ok |
| leg_fixture200_n64 | leg | 20.15 | 0.63 | 23.90 | 0.64 | +0.6% | ok | ok |
| np_c5r2 | new | 216.62 | 21.06 | 207.47 | 23.29 | +10.6% | ok | ok |
| np_fixture200_n64 | new | 1009.10 | 93.67 | 1371.43 | 117.61 | +25.6% | ok | ok |
| iree_fixture64_n64 | new | 82,065 | 69.61 | 13,640 | 10.64 | -84.7% | ok | ok |
| iree_fixture200_n64 | new | 105,646 | 229.49 | 13,207 | 10.50 | -95.4% | ok | ok |

`R3` = the committed R3 snapshot (`./results/`); `R2` = `test/benchmark/new/results/r2/` +
`test/benchmark/legacy/results/r2/` (read-only history). `warm_ms` = `median_warm_ms`
(= `median(per_call_ms[1:])`), `first_ms` = `first_call_ms`. The two `iree` rows are the **auto** route
= dense (§3); the wave-attribution twins are in 2.3.

### 2.2 Core training configs (`sr`, pop 200 / 10 gens / tree 64) — warm per-generation median per rep [ms]

| id | side | rep | label | R2 warm/gen_ms | R3 warm/gen_ms | warm delta | R2 cold_1st_gen_ms | R3 cold_1st_gen_ms |
|---|---|---|---|---|---|---|---|---|
| leg_sr_a | leg | 0 | churning | 1.62 | 1.80 | +11.4% | 480.68 | 595.29 |
| leg_sr_a | leg | 1 | fully warm | 1.59 | 1.58 | -0.2% | 1.61 | 1.66 |
| np_sr_a | new | 0 | churning | 84.09 | 117.33 | +39.5% | 1356.30 | 1641.86 |
| np_sr_a | new | 1 | fully warm | 22.07 | 30.78 | +39.5% | 108.58 | 148.74 |
| iree_sr_a | new | 0 | churning | 10,733 | 138.26 | -98.7% | 82,099 | 16,552 |
| iree_sr_a | new | 1 | fully warm | 65.31 | 38.39 | -41.2% | 324.95 | 35.70 |

`rep 0` is the churning population (it repays the per-structure builds), `rep 1` the fully warm one
(same seed replayed in-process); `warm/gen_ms` = `median(per_gen_ms[1:])` per rep, `cold_1st_gen_ms` =
that rep's `cold_first_gen_ms` (compile cost, never mixed into a warm median).

### 2.3 Wave attribution — forced specialized (R2-era route) vs the R3 auto route

| spec id | metric | R2 auto (base id) | R3 auto | R3 spec (specialized) | spec/auto R3 |
|---|---|---|---|---|---|
| iree_fixture64_n64_spec | first_call_ms | 82,065 | 13,640 | 31,486 | 2.31x |
| iree_fixture64_n64_spec | warm_ms | 69.61 | 10.64 | 67.87 | 6.38x |
| iree_fixture200_n64_spec | first_call_ms | 105,646 | 13,207 | 137,475 | 10.41x |
| iree_fixture200_n64_spec | warm_ms | 229.49 | 10.50 | 228.52 | 21.77x |
| iree_sr_a_spec | rep0 warm/gen_ms | 10,733 | 138.26 | 14,741 | 106.62x |
| iree_sr_a_spec | rep1 warm/gen_ms | 65.31 | 38.39 | 64.66 | 1.68x |

The `_spec` arms run the *same code* as `auto` with `EVOGP_EVAL_MODE=specialized`, so the
`R2 auto` → `R3 spec` column is **wave 1 alone** and the `R3 spec` → `R3 auto` column is **wave 2
alone**, both inside one window. The `_spec` first-call / rep0 cells are the round's least quotable
numbers: compile time is not reproducible on this box (§6) and the window was loaded, which is why
the fixture200 and `sr_a` `_spec` first calls read **above** their R2 counterparts (137 475 vs
105 646; 142 445 vs 82 099) while every warm number they are paired with does not (and the
fixture64 `_spec` cell reads below, 31 486 vs 82 065).

### 2.4 The light group in both windows — why most of the R3 column is a W2 column

| config | metric | R2 | R3 W1 (GPU 6, contended) | R3 W2 (GPU 0, companion) | R3 headline | headline vs R2 |
|---|---|---|---|---|---|---|
| leg_c5r2 | eval warm / call | 0.6327 | 1.9300 | **0.6323** | 0.6323 | −0.1% |
| leg_fixture200_n64 | eval warm / call | 0.6319 | 1.9667 | **0.6357** | 0.6357 | +0.6% |
| np_c5r2 | eval warm / call | 21.056 | 20.584 | 23.293 | 23.293 | +10.6% |
| np_fixture200_n64 | eval warm / call | 93.674 | 95.006 | 117.612 | 117.612 | +25.6% |
| leg_sr_a | sr rep0 / rep1 warm per gen | 1.62 / 1.59 | 1.68 / 1.73 | 1.80 / 1.58 | 1.80 / 1.58 | +11.4% / −0.2% |
| np_sr_a | sr rep0 / rep1 warm per gen | 84.09 / 22.07 | 142.44 / 42.75 | 117.33 / 30.78 | 117.33 / 30.78 | +39.5% / +39.5% |
| iree_fixture64_n64 | eval warm / call | 69.61 | 10.644 | — | 10.644 | −84.7% |
| iree_fixture200_n64 | eval warm / call | 229.49 | 10.498 | — | 10.498 | −95.4% |
| iree_sr_a | sr rep0 / rep1 warm per gen | 10 733.32 / 65.31 | 138.26 / 38.39 | — | 138.26 / 38.39 | −98.7% / −41.2% |

W1 = the original window (all 12 configs, GPU 6, 2026-09-12 21:59–22:12 HKT); W2 = the companion
re-run of the six light-group configs on GPU 0 (22:20:52–22:21:34 HKT, records committed in `d38ada7`).
The **`iree` rows were measured only in W1**; the legacy and numpy rows exist in both. W1 values are
read from the pre-re-run snapshots (`git show a848fc2:…`, still in history).

### 2.5 Compile / cold first step in both windows [ms]

| config | metric | R2 | W1 | W2 |
|---|---|---|---|---|
| leg_c5r2 | `first_call_ms` | 20.62 | 29.96 | 26.49 |
| leg_fixture200_n64 | `first_call_ms` | 20.15 | 31.50 | 23.90 |
| np_c5r2 | `first_call_ms` | 216.62 | 198.17 | 207.47 |
| np_fixture200_n64 | `first_call_ms` | 1009.10 | 1094.94 | 1371.43 |
| leg_sr_a | `cold_first_gen_ms` rep0 / rep1 | 480.68 / 1.61 | 631.63 / 1.77 | 595.29 / 1.66 |
| np_sr_a | `cold_first_gen_ms` rep0 / rep1 | 1356.30 / 108.58 | 1397.64 / 186.46 | 1641.86 / 148.74 |
| iree_fixture64_n64 | `first_call_ms` | 82 065 | **13 640** | — |
| iree_fixture200_n64 | `first_call_ms` | 105 646 | **13 207** | — |
| iree_sr_a | `cold_first_gen_ms` rep0 / rep1 | 82 099 / 324.95 | **16 552 / 35.70** | — |

### 2.6 Old vs new after R3 (warm medians; ratios = legacy ÷ ETL)

| workload | legacy | ETL `iree` (dense) | ETL `numpy` | legacy/iree R2 → R3 | legacy/numpy R2 → R3 |
|---|---|---|---|---|---|
| fixture64 n64, per call | 0.6323 | 10.64 | 23.29 | 110.0× → **16.8×** | 33.3× → 36.8× |
| fixture200 n64, per call | 0.6357 | 10.50 | 117.61 | 363.2× → **16.5×** | 148.2× → 185.0× |
| sr_a, rep1 per generation | 1.58 | 38.39 | 30.78 | 41.1× → **24.3×** | 13.9× → 19.5× |

Sources: legacy and numpy = the W2 records (`./results/legacy/*.json`, `./results/new/np_*.json`);
`iree` = the W1 records (`./results/new/iree_*.json`); R2 columns from `../REPORT.md` §1 (recomputed
there from `test/benchmark/{new,legacy}/results/r2/`). The legacy/`iree` ratio therefore pairs the W2
legacy anchors with the W1 `iree` numbers; **formed entirely inside W1 it is 5.5× / 5.3× / 22.2×** (§6).
`iree` vs `numpy` inside R3: the dense route is faster on both evals (2.19× / 11.2×) but still behind
numpy on `sr_a` (1.18× rep0, 1.25× rep1) — with numpy held at its R2 level that rep1 ratio would be
1.74×, so the cross-backend lines inherit the numpy noise band.

## 3. What the dense route looks like on the wire (`iree_sr_a`, rep0)

| generation | route (evidence in the record) | per-gen ms |
|---|---|---|
| gen 0 | **dense compile** — stage probe: 1 `run` call, 0 trees invoked, 16 551.8 ms in the uninstrumented `misc` bucket | **16 551.8** |
| gen 1 | **specialized fallback** (trees exceed the 32-node dense trace cap): 127/200 trees invoked, 83 build misses + 44 hits, `sr_fitness_calls=1` | 33 184.2 |
| gen 2 | **specialized fallback**: 102/200 trees invoked, 86 build misses + 16 hits, `sr_fitness_calls=1` | 34 461.8 |
| gens 3–9 | **dense** again: 1 `run` per call, remainder in `misc` | 138.26 / 141.92 / 142.08 / 47.49 / 40.00 / 35.66 / 37.28 |
| rep 1 (all gens) | fully warm: 229 build hits, **0 misses**, no fallback build traffic | warm median **38.39** |

Build economy: rep0 records 170 cumulative cold-build misses at the guard counter while the probe sees
169 `etl` build misses — the extra one is the dense graph compile (it never goes through
`etl.build_cached`) — versus **517 per-structure builds for the same config in R2** and for the forced
`_spec` arm (`./results/new/iree_sr_a_spec.json`). For the two eval configs the auto route needs no
per-structure build at all: 25 warm calls with `build_calls=0`, `trees_invoked=0`, `run_calls=25`
(fixture64: 97.3 ms of the 14 400.7 ms total attributed to wrapped `run`; fixture200: 82.8 of
13 779.4 ms) — the warm cost sits in the same `misc` bucket as the wave-2 dense execution itself,
which the R2 probe does not instrument (§6).

## 4. Windows and provenance

Both windows ran on 2026-09-12 on the same shared box with host load average in the 80–103 range
(operator record); what differs is the GPU and what else was on it.

* **W1 — the original window (all 12 configs, GPU 6, 21:59–22:12 HKT).** A foreign tenant
  (video-attention training, pid 2246336, 22–30 GB) was resident on GPU 6: every W1 record logs
  22 247–22 567 MiB counted used at run start, and the logs' `started`/`finished` stamps span minutes
  while each config's own timed region is 1–90 s. **Effect:** the two launch-bound legacy evals (25
  back-to-back ~0.63 ms calls) inflated **3.05× / 3.11×** over R2 (1.9300 / 1.9667 ms) and `np_sr_a`
  ran 1.7–1.9× R2, while `np_c5r2`, `np_fixture200_n64` and `leg_sr_a` rep0 stayed within ~4% of R2
  (20.58 vs 21.06; 95.01 vs 93.67; 1.68 vs 1.62 — `leg_sr_a` rep1 1.73 vs 1.59 = +8.8%).
  The agreement pre-flight ran in this window.
* **W2 — the companion re-run (six light-group configs, GPU 0, 22:20:52–22:21:34 HKT, committed in
  `d38ada7`).** GPU 6 was still tenant-occupied, so the light group was re-run on GPU 0 (0–4 MiB used
  at run start, recorded in every JSON's `gpu` block). **Effect:** the legacy side recovered
  **exactly** to R2 (0.6323 / 0.6357 / 1.58 vs 0.6327 / 0.6319 / 1.59), which is the round's evidence
  that W1's legacy inflation was the window rather than the harness or the engine. The numpy side did
  **not** recover and was in fact worse than W1 for `np_c5r2` / `np_fixture200_n64` (+13.2% / +23.8%
  W2-over-W1) while better for `np_sr_a` (−17.6% / −28.0%) — run-to-run noise on the CPU-bound
  backend, which the W1/W2 spread for numpy (up to −28% / +24% between windows, +10.6% … +39.5% against
  R2) does not resolve.
* **Which number is which.** The **R3 headline for the legacy side is W2** (its recovered window); the
  **`iree` numbers exist only in W1**; the numpy rows are reported from W2 for consistency with the
  other light-group rows but carry the band above, and **no engine-side shift is attributed for numpy**
  (wave 1 was priced where it is measurable — the specialized `iree` arms, within ±2.5% of R2 — and the
  numpy route is structurally unchanged: exactly R2's per-structure builds and miss counts
  (48 / 155 / 155 cold and 0 / 0 / 362 warm misses in rep0, identical in both rounds), a
  bit-identical `sr_a` trajectory, and fixture values within 2.03e-07 relative of the R2 anchor —
  gate (i) INFO, not bit-identical there).
* **Within-window ratios are the robust ones** (same process, same minutes, same tenant): wave 1 vs
  R2 inside W1 (§2.3), wave 2 vs wave 1 inside W1 (§2.3), and the auto-vs-specialized ratios
  6.38× / 21.77× / 106.62× / 1.68×. Ratios that pair different windows (the §2.6 legacy/`iree`
  column) are quoted as structural facts with the in-window alternative given beside them.
* **The workload is provably unchanged:** `best_fitness_final` and the full `per_gen_best_fitness`
  trajectories of all three `sr_a` configs are identical across R2, W1 and W2 in both reps
  (`./results/…`, `test/benchmark/…/results/r2/…`, `git show a848fc2:…`), and every record's
  `evogp_commit` / `evogp_file` / `etl_version` / `env` / `gpu` block identifies the engine it measured
  (W1 and the `iree`/`_spec` group ran at `49c5197f`; the W2 light-group records at `a848fc2f`).

## 5. Gates

**Gate (i) — backend agreement (pre-flight; nothing IREE was quotable before it passed).** Six arms
(numpy / iree / iree+`EVOGP_EVAL_MODE=specialized` × the two fixtures) on GPU 6, 2026-09-12
22:00–22:12, all exit 0, one shot, no retries (`results/backend_agreement/run_agreement_r3.last_run.txt`
→ `ok=6 skipped=0 failed=0`). `python3 test/benchmark/r2/round3/check_agreement_r3.py`
→ **PASS (32 checks, 0 failures)**:

| fixture | numpy vs `_spec` | numpy vs auto (dense) | NaN / ±inf |
|---|---|---|---|
| fixture64 (64 trees) | bit-identical (max\|diff\| 0) | PASS — max\|diff\| 7.62939e-06, max rel 1.08241e-07 | 3 / 0 both |
| fixture200 (200 trees) | bit-identical (max\|diff\| 0) | PASS — max\|diff\| 0.00390625, max rel 2.02762e-07 | 12 / 0 both |

plus per-fixture npz identity (64-tree `0bfb0bbf…9ca0`, 200-tree `6789ef61…41b9`) and
`values_sha256` ↔ `_values.npy` integrity. Continuity vs the R2 anchors is INFO, never a gate: the
auto-dense values are **bit-identical to the R2 anchors** (`b8192cb1…fa02`, `07948efe…e048`), while
numpy / `_spec` — i.e. the wave-1 specialized route — shifted by ≤ 2.03e-07 relative. **IREE timings
from this round are quotable.**

**Gate (ii) — record validation.** `python3 test/benchmark/r2/round3/check_r3.py --manifest
test/benchmark/r2/round3/manifest.txt --require-stages` → **12 records: 12 complete, 0 partial, 0
skipped, 0 missing, 0 malformed; 45 checks, 0 failures** — the 9 new-side ids all carry the `stages`
marker. Every record is `complete: true` and every config log ends `done … interrupted=False`; no run
exited non-zero.

## 6. Method and caveats

* **Method = R2, unchanged** (`./README.md` for the full protocol): one `bench_gp.py` process per
  config, `ulimit -n 65536` first, exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per
  invocation with `timeout` inside it, GPU 6 by default, same fixtures (`test/goldens/forest_random.npz`,
  `test/benchmark/fixtures/forest200_tree64_random.npz`), same wiring and implicit `--seed 0`.
* **Warm median is the headline; compile is reported separately** (`first_call_ms` /
  `cold_first_gen_ms` / cold-build counts), never mixed into a warm median. Medians are ±10% on this
  shared box, and **`iree` compile time is not reproducible** (spread of a factor of a few for
  identical build counts) — the `_spec` first-call cells in 2.3 are indicative only.
* **Stage-probe limitation, seen live.** The R2 probe instruments the *specialized* engine, so the
  dense route's own cost is only visible in the `misc` bucket: the two auto eval configs report
  `build_calls=0`, `trees_invoked=0` and the whole 25-call total in `misc` (14 400.7 / 13 779.4 ms),
  and only `iree_sr_a`'s specialized fallback generations (gens 1–2) populate the build/run/readback
  counters. Degenerate, documented in `./README.md`, **not** a gate failure: the workload numbers and
  the fitness are unaffected. The `_spec` arms have fully populated stages (48 / 155 / 517 builds — the
  same counts as R2).
* **File descriptors / teardown.** Auto-route `fds.end` is 6491 (fixture64) / 6491 (fixture200) /
  9723 (`sr_a`) versus 2395 / 5819 / 17243 for the per-structure route (R2 and the R3 `_spec` arms,
  ~33–49 exit descriptors per compiled structure). So the dense route is **flat in tree count** on the
  two eval fixtures (the same 6491 exit count at 64 and 200 trees — no per-structure component at all,
  though still more than the 64-tree specialized arm's 2395) and ~1.8× cheaper than the per-structure
  route on the 10-generation `sr_a` run (9723 vs 17243). Every record ran under `nofile_rlimit` 65536,
  the `nanobind: leaked …` teardown warnings in the IREE logs are benign and unchanged from R2, and
  **no config crashed** (no SIGSEGV; the cold-build guard never tripped, `exceeded=false` everywhere).
* **Not re-tested:** the R2 crash config `iree_sr_b2` (pop 1000 / tree 128) is not part of the R3
  matrix, and no pop-1000 IREE config was run, so this round says nothing about that failure mode.
* **Device deviation:** W1 used the planned GPU 6; the W2 companion window used GPU 0 because GPU 6 was
  tenant-occupied. Every JSON records its own `gpu.cuda_visible_devices`, so the two windows are
  separable from the data alone. GPU 7 (uncorrectable-ECC fault) was never touched.
* **IREE stays opt-in and non-primary:** all IREE arms here are `O1` (O2/O3 miscompile CUDA silently),
  the numpy backend remains the default reference, and the legacy side is inert to all
  `ETL_*`/`EVOGP_ETL_*` variables.

## 7. Where the numbers live

| what | where |
|---|---|
| protocol, config table, commands, gates, limitations | `./README.md` |
| R3 records (12 JSONs = latest snapshots, 12 appended logs) | `./results/new/`, `./results/legacy/` |
| W1 snapshots of the six re-run configs | `git show a848fc2:test/benchmark/r2/round3/results/…` |
| backend-agreement pre-flight (JSON + `_values.npy` + log per arm) | `./results/backend_agreement/` |
| R2 baseline this verdict is read against (stage-by-stage analysis, ranked levers) | `../REPORT.md`, `../r2/README.md`, `../SPEC.md` (frozen) |
| R2 records (read-only history) | `../new/results/r2/`, `../legacy/results/r2/` |
| tools used for every table above | `./compare_rounds.py`, `./check_r3.py`, `./check_agreement_r3.py` |
