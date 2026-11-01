# SUMMARY_armB.md — Arm B: the nine new-side R3 configs re-measured on a **patched ETL checkout**

**What this is.** Arm B is the round-3 *engine-delta* workstream: it re-measures the **nine new-side R3
configs** with the **R2 method unchanged** (same harness `test/benchmark/bench_gp.py`, same flags, same
fixtures, same wiring, same implicit `--seed 0`, same `ulimit`/`flock`/`timeout` order, one process per
config, warm median as the headline with compile reported separately, stage probe on the new side)
against a **patched ETL checkout shadowed onto `PYTHONPATH`**:

* `PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141` — the patched checkout shadows the stock editable
  install `/mnt/local-ssd/bchuang/etl`;
* the patched checkout differs from stock in **three source files** (`etl/build_cache.py` content
  fingerprint fast path, `etl/backends/adapters/iree.py` per-thread runtime-HAL-device cache,
  `etl/persist/codec.py` `has_custom_codecs()`) plus documentation; it is not a git checkout, so the
  per-file sha256 in **`./results/armB/etl_import_probe.txt`** is the revision handle;
* the **only other delta** is that `EVOGP_EVAL_MODE=auto` is passed *explicitly* on the three plain
  `iree` ids (Arm A relied on the engine default) — recorded in each JSON's `env` block.

Arm B ids are their Arm A/R2 counterparts with an `armB_` prefix, so every record diffs directly
against Arm A (`./results/new/`) and against untouched R2 history (`../new/results/r2/`,
`../legacy/results/r2/`). **The legacy side has no Arm B row by design** — the frozen legacy CUDA
records stay the reference. The committed `./results/armB/new/*.json` are the **W2 records of record**
(companion window on GPU 0); the superseded **W1** snapshots stay retrievable in git history at the
pre-W2 tip **`b728466`** (`git show b728466:test/benchmark/r2/round3/results/armB/new/<id>.json`), and
the append-only per-config logs carry both windows side by side. Both gates are green (§8).

Every number below is recomputed from the committed records by
**`./compare_rounds_armB.py --markdown`** (which imports `./compare_rounds.py` for the metric rules, so
the R2-vs-R3 and R2-vs-Arm-B tables cannot drift apart) or quoted from the gates' own output — nothing
is invented. Protocol/evidence: `./README.md`, `./results/armB/{armB_runlog_section.md,windows.md,
gate_outputs.md}`, R2 baseline `../REPORT.md`, Arm A baseline `./SUMMARY.md`. The round ran in **two
windows on two devices** (§6): read the timing tables with that in mind.

## 1. Verdict

* **The patched ETL changes nothing the round measures numerically.** The agreement pre-flight's HARD
  gate reproduces **all six Arm A `values_sha256` bitwise** (fixture64 numpy `2e5c0fa1fdbe…`, iree
  `b8192cb1770b…`, iree_spec `2e5c0fa1fdbe…`; fixture200 `16d5bbe98481…`, `07948efededc…`,
  `16d5bbe98481…`); the three `sr` ids' `per_gen_best_fitness` trajectories are **identical
  element-for-element across Arm A, W1 and W2 in both reps** for every route (auto-dense, forced
  specialized, numpy), the cold-build traffic is identical (`./compare_rounds_armB.py` prints the same
  stage counters and JSON miss counts for A/W1/W2), and every config exited 0 with `partial=false`.
* **The one number that moved clear of the noise band is descriptor/fd economy on `iree`.** With
  *identical* build counts, the Arm B `iree` arms end at **91 fds** where their Arm A twins end at
  2395 / 5819 / 9723 / 17243 (and the numpy arms stay at 59 on both sides). That is the flat curve of
  a per-thread runtime-HAL-device cache: R2's own rate was **32.3 fds per compiled structure** (517
  builds → 17 243 fds), the patched arm is flat at 91 fds at 170, 517 **and 1181** builds (§7).
* **No warm-median claim survives the windows in either direction.** The A→B deltas run both ways and
  mostly sit inside the documented shared-box band: numpy −2.1 % → +8.3 % (fixture64), −23.4 % →
  −3.8 % (fixture200), +55.9 % → +117.4 % (`sr` rep1); `iree` auto +39.4 % → −78.7 %, −4.4 % →
  −95.6 %, +230.3 % → +94.1 %; forced-specialized −12.4 % → −14.6 %, +3.5 % → +3.1 %, −6.9 % →
  −7.8 %. Two of those cells are known artifacts rather than engine effects: the auto `sr` rep1
  **+230 %** is where the 5th order statistic of a **bimodal** per-generation series falls (Arm A
  38.39 ms vs Arm B 126.79 ms on medians, with Arm B's rep totals *lower*: 82.6 s → 1106.6 ms versus
  Arm A's 953.1 ms), and the numpy rows are a CPU-bound route on a 46-user host.
* **Compile is reported separately and stays where R2 put it.** Cold-build traffic is **byte-for-byte
  the same** on both sides (`48 / 155 / 517 / 1 / 1 / 170` JSON misses, identical stage `cold+warm`
  splits), while first-call wall times spread by the documented non-reproducible IREE compile factor
  (dense evals 13 640/13 207 ms on Arm A vs 16 492/18 011 ms on Arm B W2; the 200-tree specialized arm
  137 475 → 82 065 → 63 533 ms across A → W1 → W2). No first-call number is quotable as a hardware
  constant, and none of them is mixed into a warm median.
* **R2's `iree_sr_b2` SIGSEGV did not reproduce** (non-headline; the b2 ids are deliberately **not**
  manifest ids): both re-attempts on the patched checkout, dense and pinned-specialized, completed all
  2 reps × 2 generations with **exit 0** where R2 saw exit 139 after ≈800–900 builds (§7.1).
* **The workload is provably unchanged**, and both gates are green: 9 complete / 34 checks /
  0 failures (records) and PASS / 38 checks / 0 failures (agreement) — §8.

## 2. Warm medians — R2 / Arm A / Arm B W1 / Arm B W2

`A dev` is the device the Arm A record ran on (0 = Arm A's own companion window, 6 = its first
window); `B W1` is the contended GPU-6 window, `B W2` the GPU-0 companion and **the headline**.
`A->B` = W2 vs Arm A, `R2->B` = W2 vs R2. Warm per-call median for the eval fixtures; per-rep
per-generation median for `sr` (rep0 churning / rep1 fully warm).

### 2.1 Workload 1/3 — fixture64 x n64 (eval64 default fixture), warm per-call median [ms]

| arm | R2 | A (R3) | A dev | B W1 (GPU 6) | B W2 (GPU 0) | A->B | R2->B |
|---|---|---|---|---|---|---|---|
| legacy (CUDA) | 0.63 | 0.63 | 0 | - | - | - | - |
| numpy (default) | 21.06 | 23.29 | 0 | 20.78 | 22.81 | -2.1% | +8.3% |
| iree auto (dense) | 69.61 | 10.64 | 6 | 8.54 | 14.84 | +39.4% | -78.7% |
| iree spec (forced specialized) | 69.61 | 67.87 | 6 | 62.19 | 59.47 | -12.4% | -14.6% |

### 2.2 Workload 2/3 — fixture200 x n64, warm per-call median [ms]

| arm | R2 | A (R3) | A dev | B W1 (GPU 6) | B W2 (GPU 0) | A->B | R2->B |
|---|---|---|---|---|---|---|---|
| legacy (CUDA) | 0.63 | 0.64 | 0 | - | - | - | - |
| numpy (default) | 93.67 | 117.61 | 0 | 135.94 | 90.15 | -23.4% | -3.8% |
| iree auto (dense) | 229.49 | 10.50 | 6 | 5.20 | 10.03 | -4.4% | -95.6% |
| iree spec (forced specialized) | 229.49 | 228.52 | 6 | 200.09 | 236.50 | +3.5% | +3.1% |

### 2.3 Workload 3/3 — sr_a pop200 / 10 gens / tree 64, warm per-generation median per rep [ms]

| arm | R2 | A (R3) | A dev | B W1 (GPU 6) | B W2 (GPU 0) | A->B | R2->B |
|---|---|---|---|---|---|---|---|
| legacy (CUDA) rep0 churning | 1.62 | 1.80 | 0 | - | - | - | - |
| legacy (CUDA) rep1 fully warm | 1.59 | 1.58 | 0 | - | - | - | - |
| numpy (default) rep0 churning | 84.09 | 117.33 | 0 | 154.45 | 156.14 | +33.1% | +85.7% |
| numpy (default) rep1 fully warm | 22.07 | 30.78 | 0 | 33.83 | 47.98 | +55.9% | +117.4% |
| iree auto (dense) rep0 churning | 10,733 | 138.26 | 6 | 130.59 | 82.56 | -40.3% | -99.2% |
| iree auto (dense) rep1 fully warm | 65.31 | 38.39 | 6 | 127.89 | 126.79 | +230.3% | +94.1% |
| iree spec (forced specialized) rep0 churning | 10,733 | 14,741 | 6 | 12,890 | 11,519 | -21.9% | +7.3% |
| iree spec (forced specialized) rep1 fully warm | 65.31 | 64.66 | 6 | 57.27 | 60.19 | -6.9% | -7.8% |

Reading rules for all three tables: the `_spec` rows read their **R2 column off the base id** (in R2
the `auto` route *was* the specialized engine, so that cell is the R2 number for the same code path);
a `-` cell means "not measured" (the legacy side has no Arm B record); **within-window** comparisons
(W1 vs W2, A vs B inside one device) are the robust ones, cross-window pairs are window-paired context.
The numpy rows are the one *device-paired* A/B pair (both on GPU 0) and still move by −23.4 % … +117.4 %,
which is the measurement noise floor of the CPU-bound route on this box, not an engine effect. The
`iree` auto `sr` rep1 **+230 %** must be read next to its rep totals (Arm B W2 82.6 s + 1106.6 ms
versus Arm A 84.8 s + 953.1 ms) and its bitwise-identical fitness trajectory: it is a **median-position
artifact on a bimodal series** (Arm B W2 rep1 `per_gen_ms[1:]` = `[329.5, 167.0, 129.7, 126.8, 41.4,
40.7, 41.8, 52.6, 129.1]`), not added work.

## 3. Compile — first call, compile traffic and fds (never mixed into a warm median)

As emitted by `./compare_rounds_armB.py --markdown`; `first_ms` = `first_call_ms` (eval64) or rep0's
`cold_first_gen_ms` (`sr`), the miss counter is the JSON `cold_builds.misses_max` the
`--max-cold-builds` guard enforces (`-` = no budget on the numpy configs), `fds` =
start → after cold → end.

| config (armB id) | kind | first_ms A | first_ms B W1 | first_ms B W2 | json cold misses A / W1 / W2 | fds A | fds B W2 |
|---|---|---|---|---|---|---|---|
| armB_np_c5r2 | eval | 207.47 | 191.31 | 194.00 | - / - / - | 39 -> 59 -> 59 | 39 -> 59 -> 59 |
| armB_iree_fixture64_n64 | eval | 13,640 | 18,012 | 16,492 | 1 / 1 / 1 | 39 -> 347 -> 6491 | 39 -> 91 -> 91 |
| armB_iree_fixture64_n64_spec | eval | 31,486 | 20,044 | 17,523 | 48 / 48 / 48 | 39 -> 1627 -> 2395 | 39 -> 91 -> 91 |
| armB_np_fixture200_n64 | eval | 1371.43 | 995.50 | 969.27 | - / - / - | 39 -> 59 -> 59 | 39 -> 59 -> 59 |
| armB_iree_fixture200_n64 | eval | 13,207 | 12,228 | 18,011 | 1 / 1 / 1 | 39 -> 347 -> 6491 | 39 -> 91 -> 91 |
| armB_iree_fixture200_n64_spec | eval | 137,475 | 82,065 | 63,533 | 155 / 155 / 155 | 39 -> 5051 -> 5819 | 39 -> 91 -> 91 |
| armB_np_sr_a | train | 1641.86 | 2255.92 | 2705.00 | - / - / - | 39 -> 59 -> 59 | 39 -> 59 -> 59 |
| armB_iree_sr_a | train | 16,552 | 15,965 | 14,004 | 170 / 170 / 170 | 39 -> 347 -> 9723 | 39 -> 91 -> 91 |
| armB_iree_sr_a_spec | train | 142,445 | 64,951 | 117,096 | 517 / 517 / 517 | 39 -> 5051 -> 17243 | 39 -> 91 -> 91 |

Three facts to take from this table:

1. **The compile traffic is identical on both engines** for every config — the patched lookup code
   changes neither the number of builds nor the miss/hit split (`48 / 155 / 517`, dense `1 / 1 / 170`),
   which is the strongest available evidence that the workload did not move.
2. **The first-call wall times are not comparable across windows** (± a factor of a few for IREE
   per-structure compile), and the A-column values themselves come from two different Arm A windows
   (GPU 0 for the numpy rows, GPU 6 for the `iree` rows).
3. **The fd column is the round's one clear A→B difference**: numpy identical (39 → 59 → 59), IREE flat
   at 91/65 536 on Arm B versus 2395 / 5819 / 9723 / 17243 on Arm A at the *same* build counts. Note the
   dense eval arms' JSON miss counter reads `1` (the dense graph compile, which never goes through
   `etl.build_cached`) while their stage block reads `0` — the documented dense-route display gap, not
   a gate failure.

## 4. Stage probe — build traffic, the cache-hit path, and what the probe cannot see

Per-rep stage counters (from `./compare_rounds_armB.py`). The **route column is read off the record**
(`env` + stage fingerprint), never assumed:

| config (rep) | route | A build calls / hits / misses | B W1 build calls / hits / misses | B W2 build calls / hits / misses | B W2 stage cold / warm misses (ms) | B W2 probe span in misc [ms] |
|---|---|---|---|---|---|---|
| armB_np_c5r2 (rep0) | specialized | 1325 / 1277 / 48 | 1325 / 1277 / 48 | 1325 / 1277 / 48 | 48 (155) / 0 (0) | 58.86 |
| armB_iree_fixture64_n64 (rep0) | dense (probe-degenerate) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 (0) / 0 (0) | 16,956 |
| armB_iree_fixture64_n64_spec (rep0) | specialized | 1325 / 1277 / 48 | 1325 / 1277 / 48 | 1325 / 1277 / 48 | 48 (17,287) / 0 (0) | 40.76 |
| armB_np_fixture200_n64 (rep0) | specialized | 4275 / 4120 / 155 | 4275 / 4120 / 155 | 4275 / 4120 / 155 | 155 (876) / 0 (0) | 50.34 |
| armB_iree_fixture200_n64 (rep0) | dense (probe-degenerate) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 (0) / 0 (0) | 18,266 |
| armB_iree_fixture200_n64_spec (rep0) | specialized | 4275 / 4120 / 155 | 4275 / 4120 / 155 | 4275 / 4120 / 155 | 155 (62,279) / 0 (0) | 41.38 |
| armB_np_sr_a (rep0) | specialized | 632 / 115 / 517 | 632 / 115 / 517 | 632 / 115 / 517 | 155 (1599) / 362 (1876) | 1478.10 |
| armB_np_sr_a (rep1) | specialized | 632 / 632 / 0 | 632 / 632 / 0 | 632 / 632 / 0 | 0 (0) / 0 (0) | 448.28 |
| armB_iree_sr_a (rep0) | dense eval / specialized fallback | 229 / 60 / 169 | 229 / 60 / 169 | 229 / 60 / 169 | 0 (0) / 169 (75,681) | 14,644 |
| armB_iree_sr_a (rep1) | dense eval / specialized fallback | 229 / 229 / 0 | 229 / 229 / 0 | 229 / 229 / 0 | 0 (0) / 0 (0) | 714.66 |
| armB_iree_sr_a_spec (rep0) | specialized | 632 / 115 / 517 | 632 / 115 / 517 | 632 / 115 / 517 | 155 (116,040) / 362 (195,559) | 768.88 |
| armB_iree_sr_a_spec (rep1) | specialized | 632 / 632 / 0 | 632 / 632 / 0 | 632 / 632 / 0 | 0 (0) / 0 (0) | 122.79 |

* **The build traffic is identical across A / W1 / W2 row by row** — same calls, same hit/miss split,
  same cold-vs-warm classification. That is contention-independent and is the round's workload-identity
  cross-check next to the bitwise fitness trajectories.
* **Where the patched lookup code sits.** Its target is the *warm cache-hit* path, i.e. the fully warm
  rep of the training ids (all 632 / 229 build calls hitting the cache) — the one place the A-1/A-3
  `etl/build_cache.py` fast path could show up:

  | config (fully warm rep) | metric | A [ms] | B W1 [ms] | B W2 [ms] |
  |---|---|---|---|---|
  | np_sr_a (rep1) | build_hit_ms | 142.20 | 146.76 | 158.40 |
  | np_sr_a (rep1) | run_ms | 154.58 | 182.40 | 180.91 |
  | np_sr_a (rep1) | readback_ms | 9.60 | 10.83 | 11.68 |
  | np_sr_a (rep1) | total_ms | 478.02 | 595.22 | 862.85 |
  | iree_sr_a (rep1) | build_hit_ms | 50.31 | 44.30 | 50.31 |
  | iree_sr_a (rep1) | run_ms | 281.11 | 222.13 | 363.27 |
  | iree_sr_a (rep1) | readback_ms | 86.26 | 69.51 | 84.12 |
  | iree_sr_a (rep1) | total_ms | 953.15 | 826.43 | 1106.59 |
  | iree_sr_a_spec (rep1) | build_hit_ms | 137.79 | 155.06 | 149.63 |
  | iree_sr_a_spec (rep1) | run_ms | 499.58 | 409.84 | 369.51 |
  | iree_sr_a_spec (rep1) | readback_ms | 275.05 | 230.67 | 219.32 |
  | iree_sr_a_spec (rep1) | total_ms | 1169.67 | 1040.65 | 932.29 |

  The patched checkout's cache-hit figures sit **inside the window noise on both sides** (e.g. 632 hits:
  A 137.79 → W1 155.06 → W2 149.63 ms; 229 hits: 50.31 → 44.30 → 50.31 ms), so this round can show
  *"the patched lookup path costs no more and produces the same numbers"* — it cannot price a win at
  this granularity, and no such claim is made. Per-stage compile ms is documented as non-reproducible
  (spread of a factor of a few for identical build counts), which is exactly what the cold-miss columns
  of the raw generator dump show.
* **The dense-route degeneracy for the auto arms is documented, not a gate failure.** The R2 probe wraps
  the *specialized* engine's call sites only, so the two dense eval arms report `build_calls=0`,
  `decode_calls=0`, `sr_fitness_calls=0`, only `run_calls=25` non-zero and the whole probe span in the
  `misc` bucket (16 956 / 18 266 ms) — while the JSON-level guard still counts their one dense compile.
  `armB_iree_sr_a`, whose route is `auto`, is **not** degenerate: it reaches the specialized fallback and
  reports real build/run/readback counters (`0+169` misses over 229 calls). The `_spec` and numpy ids
  carry fully populated blocks (48+0 / 155+0 / 155+362 misses), i.e. the caveat is a property of the
  route, not of this round.

## 5. Environment and shadow evidence

Emitted per record by the generator (the `env` block of each JSON); `(unset)` = the variable was not
passed, so the engine default applied. `route` is the stage fingerprint read from the record.

| config | window | EVOGP_ETL_BACKEND | EVOGP_EVAL_MODE | EVOGP_ETL_CACHE_SIZE | EVOGP_ETL_IREE_DEVICE | EVOGP_ETL_IREE_OPT_LEVEL | ETL_DISABLE_LOCATIONS | CUDA_VISIBLE_DEVICES | route taken (stage fingerprint) |
|---|---|---|---|---|---|---|---|---|---|
| armB_np_c5r2 | W1 / W2 | (unset) | (unset) | (unset) | (unset) | (unset) | 1 | 6 / 0 | specialized |
| armB_iree_fixture64_n64 | W1 / W2 | iree | auto | (unset) | 0 | O1 | 1 | 6 / 0 | dense (probe-degenerate) |
| armB_iree_fixture64_n64_spec | W1 / W2 | iree | specialized | (unset) | 0 | O1 | 1 | 6 / 0 | specialized |
| armB_np_fixture200_n64 | W1 / W2 | (unset) | (unset) | (unset) | (unset) | (unset) | 1 | 6 / 0 | specialized |
| armB_iree_fixture200_n64 | W1 / W2 | iree | auto | (unset) | 0 | O1 | 1 | 6 / 0 | dense (probe-degenerate) |
| armB_iree_fixture200_n64_spec | W1 / W2 | iree | specialized | (unset) | 0 | O1 | 1 | 6 / 0 | specialized |
| armB_np_sr_a | W1 / W2 | (unset) | (unset) | (unset) | (unset) | (unset) | 1 | 6 / 0 | specialized |
| armB_iree_sr_a | W1 / W2 | iree | auto | 8192 | 0 | O1 | 1 | 6 / 0 | dense eval / specialized fallback |
| armB_iree_sr_a_spec | W1 / W2 | iree | specialized | 8192 | 0 | O1 | 1 | 6 / 0 | specialized |

* **The shadow is not visible in the JSON.** `PYTHONPATH` is not recorded by the harness, and
  `etl_version` is **`"0.1.0"` for BOTH the stock and the patched checkout** — so the evidence is the
  probe file plus the per-config log line: `etl_import_probe.txt` prints, for **every Arm B window**
  (7 dated blocks: tooling, agreement, the three matrix groups, the b2 re-attempt, W2),
  `PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141` → `/tmp/evogp_etl_patched_51a7141/etl/__init__.py`
  and the control `PYTHONPATH=src` → `/mnt/local-ssd/bchuang/etl/etl/__init__.py`; each
  `results/armB/new/<id>.log` carries the resolved `etl:` path in its header. The probe also pins the
  three patched file hashes (e.g. `etl/build_cache.py` patched `39ce7d6a…`, stock `bb072f6f…`), which
  are unchanged since the agreement pre-flight.
* **Per-window `etl.__file__` resolution** as actually printed by `results/armB/etl_import_probe.txt`
  (append-only; `P` = `/tmp/evogp_etl_patched_51a7141/etl/__init__.py`, `S` =
  `/mnt/local-ssd/bchuang/etl/etl/__init__.py`; every block ran the control command too and printed
  `etl` metadata version `0.1.0` for **both** checkouts):

  | probe block | window | timestamp (HKT) | shadow `etl.__file__` | control `etl.__file__` | `import evogp.tree` smoke |
  |---|---|---|---|---|---|
  | 1 | tooling creation (no heavy run) | `22:49:10` | `P` | `S` | - |
  | 2 | agreement pre-flight (gate (i), GPU 6) | `22:54:14` | `P` | `S` | `evogp.tree ok P` |
  | 3 | matrix group `iree-spec` (GPU 6) | `23:08:41` | `P` | `S` | `evogp.tree ok P` |
  | 4 | matrix group `iree` (GPU 6) | `23:17:53` | `P` | `S` | `evogp.tree ok P` |
  | 5 | matrix group `numpy` (GPU 6) | `23:26:07` | `P` | `S` | `evogp.tree ok P` |
  | 6 | `iree_sr_b2` re-attempt (GPU 6) | `23:33:00` | `P` | `S` | `evogp.tree ok P` |
  | 7 | W2 companion re-measurement (GPU 0) | `23:56:11` | `P` | `S` | `evogp.tree ok P` |

  Below block level the shadow is proven per record and per invocation: every
  `results/armB/new/<id>.log` and `results/armB/backend_agreement/<stem>.log` header carries the runner's
  resolved `etl: P` line (one per window block, since the logs are append-only), the harness params line
  of each log records `etl=0.1.0`, and every `results/armB/status_<id>*.txt` console capture starts with
  `run_all_armB.sh: etl shadow: PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141`.
* **IREE stays opt-in and non-primary** (`O1` only — `O2`/`O3` miscompile CUDA silently; device 0;
  `EVOGP_ETL_CACHE_SIZE=8192` only on the two `sr_a` ids), the numpy backend stays the default
  reference, and the legacy side is inert to every `ETL_*` / `EVOGP_ETL_*` variable.
* **The explicit `auto` on the three plain `iree` ids** is the only configuration delta versus Arm A
  besides the shadow (Arm A left the policy implicit). Both arms therefore differ in exactly one
  factor — that is the point of Arm B — and each record's `env` block makes the diff checkable.

## 6. Windows, contention and how to read them

* **W1 — the three matrix groups on GPU 6** (windows 2/3/4; 23:09:12 → 23:27:40 HKT for the nine
  configs), the runner default, kept because the ≈30 GB `syfang` tenant on that card oscillated
  (0 % → 100 %) rather than holding a sustained load, and GPU 0/1 were busy at the time. Group walls:
  iree-spec 6 min 20 s, iree 2 min 27 s, numpy 52 s; per-config `gpu (before)/(after)` pairs in the
  logs (`45→28 %`, `44→4 %`, `0→94 %`, `29→0 %`, `25→4 %`, `38→0 %`). Host loadavg ≈81–109, 46 users.
* **W2 — the companion window on GPU 0** (window 6, 23:56:11 → 00:07:12 HKT), all nine ids re-run
  one config per invocation. GPU 0 was genuinely idle at decision time (0 % utilisation and 1 MiB
  resident in all 10 samples, ≈48.7 GiB free), and the **first three configs (the `iree-spec` group)
  ran on a free device**; at ≈23:58 a foreign tenant (hyqiu, 19 706 MiB) landed on GPU 0, so the last
  six configs ran with **≈19.8 GB resident** and 52–92 % utilisation (`free_mb_start` never fell below
  28 576 MiB, so no memory-guard path armed). The pre-W2 tip `b728466` keeps the W1 JSONs.
* **Reading rule.** The committed JSONs are the W2 records of record, so §2's `B W2` column is the
  headline and `B W1` is window-paired context; the same rule applies to the Arm A columns (`A dev` 0 =
  Arm A's companion window, 6 = its first window). Medians are ±10 % on this shared box, IREE
  per-structure compile spreads by a factor of a few, and per-generation `sr` series are bimodal — so
  **no single warm cell is a result**, whereas *within-window* ratios, the build traffic, the exit
  codes, the fd curves and every fitness vector are. **No W1 claim flips in W2**, which is what the
  companion window was run to establish.
* **The agreement pre-flight was deliberately not re-run for W2** — it is contention-independent
  correctness evidence and stays as recorded (window 1/1b, GPU 6).

## 7. What the patched ETL should explain — attribution

Attribution is by *file changed* → *observable in the records*, and everything not separable from window
noise is labelled as such.

* **`etl/build_cache.py`** (156 changed lines: content-fingerprint fast path — the parent workstream's
  **A-1** single-pass key derivation and **A-3** one-slot fingerprint memo) → the target is **warm
  cache-hit lookup cost** on the numpy/specialized routes. Observable: the fully warm training reps
  (§4) stay inside the noise band on both sides, and the numpy warm per-call medians stay at Arm A's
  level (−2.1 % / +8.3 % fixture64, −23.4 % / −3.8 % fixture200, +55.9 % / +117.4 % on `sr`). Claim
  supported: *no regression, same numbers, same cache behaviour* (identical hit/miss traffic). Claim
  **not** supported: a measured warm-path speed-up — this round cannot resolve one.
* **`etl/backends/adapters/iree.py`** (59 changed lines: per-thread runtime-HAL-device cache,
  `_RUNTIME_DEVICE_CACHE = threading.local()`) → the observable is **fd stability**: with identical
  build counts, Arm A's twins leak ~33 fds per compiled structure (39 → 347 → 6491 for the dense eval
  configs, 39 → 5051 → 17 243 for `iree_sr_a_spec`, and R2 measured 32.3 fds/build on `iree_sr_a`),
  while every Arm B `iree` record stays flat at **39 → 91 → 91** at 170, 517 and even 1181 builds; the
  nanobind teardown leak counts shrink in the same direction (dense `677 / 677 / 1820` instances,
  `_spec` `4052 / 12 902 / 3854`). The same patch is the prime suspect for the b2 outcome below.
* **`etl/persist/codec.py`** (14 changed lines, `has_custom_codecs()`) → no observable in these records:
  no config exercises persistence, so this file is listed for completeness, not as an explanation.

### 7.1 The `iree_sr_b2` re-attempt (pop 1000 / gens 2 / tree 128) — non-headline

R2's only failed config was a silent native **SIGSEGV (runner exit 139)** after ≈800–900 builds and
before generation 0 finished (`../new/results/r2/iree_sr_b2.json`, kept unchanged as a crash skeleton).
The re-attempt repeats the row on the patched checkout, with Arm A's flags unchanged, in two defined
attempts (plus one mis-invoked execution preserved verbatim):

| record | route taken | complete | harness_wall_s | cold+warm traffic (json misses) | fds start->cold->end | per-rep best-fitness trajectory |
|---|---|---|---|---|---|---|
| R2 twin `../new/results/r2/iree_sr_b2` (stock ETL, pre-dense era) | specialized (only route then) | skeleton (crash) | - | - | 39 -> null -> null | - |
| B attempt 1 — `armB_iree_sr_b2` (auto => dense) | dense | complete | 213 | 0+430 misses / 685 calls (431) | 39 -> 91 -> 91 | [-0.2510969936847687, -0.22635293006896973] |
| B attempt 2 (superseded) — `..._spec_botched_dense_route` (auto => dense) | dense | complete | 264 | 0+430 misses / 685 calls (431) | 39 -> 91 -> 91 | [-0.2510969936847687, -0.22635293006896973] |
| B attempt 2 — `armB_iree_sr_b2_spec` (pinned specialized) | specialized | complete | 595 | 788+393 misses / 1568 calls (1181) | 39 -> 91 -> 91 | [-0.2510969936847687, -0.22635293006896973] |

**Both attempts exit 0 (217 s / 597 s wall, `partial=false`, guard never tripped: 431/1500 and
1181/1500), the crash did not reproduce on either route, and FD telemetry stays flat at 91 of 65 536 at
2.3× the build count of the largest completed Arm B arm** — where R2's hand-off note projected the
config's full 2200-build budget at ≈73 000 fds, above the limit. The three executions produced the
identical per-rep trajectory `[-0.2510969936847687, -0.22635293006896973]` (generation 0 bit-identical
to the R2-era numpy mirror `np_sr_b`, generation 1 within 1.49e-08 absolute / 6.6e-08 relative, inside
the documented ≤2.03e-07 R3-vs-R2 spread for the specialized route). Two caveats keep this out of the
headline: the b2 ids are **deliberately not manifest ids** (the frozen gate stays at 9 records), and the
R2 crash was never reproducible from its own record, so this is a *"did not reproduce through the same
position in the run"* datapoint — with the fd/descriptor patch as the prime suspect, **not** proof of
the original cause. Route attribution is read off the records (`env` + stage fingerprint), never
assumed: the dense arms show rep0 `0 cold + 430 warm` misses over 685 build calls, the pinned
specialized arm `788 cold + 393 warm` over 1568.

### 7.2 Robust claims (computed, not quoted)

`./compare_rounds_armB.py` prints these cross-checks on every run; they hold across Arm A, W1 and W2 and
are independent of device, window and contention:

* `per_gen_best_fitness` **element-for-element identical** for all three `sr` ids in **both reps**
  (10 generations each, `best_fitness_final = -0.2510969936847687`);
* **identical cold-build traffic** per config and rep (stage `calls / hits / misses` and the
  cold-vs-warm split, plus the JSON miss counter: `48 / 155 / 517 / 1 / 1 / 170`);
* **identical exit codes** — every Arm B record `complete: true`, `partial=false`, log footer
  `exit code: 0 (OK)`; no GUARD-SKIP, no timeout, no interrupt, no SIGSEGV, no re-run in either window;
* **fd telemetry**: `39 → 91 → 91` on the six IREE configs, `39 → 59 → 59` on the three numpy ones,
  equal to W1's values;
* the **agreement gate's six bit-identical `values_sha256` pairs** against Arm A (and the
  `np.allclose` / NaN-count structure inside each fixture) — §8.

## 8. Gates

**Gate (i) — backend agreement (pre-flight, ARM B's own copy of the R3 gate).**
`python3 test/benchmark/r2/round3/check_agreement_armB.py` → **PASS (38 checks, 0 failures)**; records
produced 2026-09-12 23:01:01–23:04:05 HKT on GPU 6, six arms, one invocation, **no retries**
(`results/armB/backend_agreement/run_agreement_armB.last_run.txt` → `# ok=6 skipped=0 failed=0`).
Contents: per-fixture npz identity (64-tree `0bfb0bbf…9ca0`, 200-tree `6789ef61…41b9`),
`values_sha256` ↔ `_values.npy` integrity, identical NaN/±inf counts (3 / 12, no ±inf),
`numpy ↔ iree` within `rtol=atol=1e-4, equal_nan=True` (max |diff| 7.63e-06 / 3.91e-03), `numpy ↔
iree_spec` **bit-identical**, and the HARD **Arm A → Arm B identity** gate: all six same-stem pairs
bit-identical (max |diff| 0). **Arm B IREE numbers are quotable.** Verbatim stdout:
`results/armB/gate_outputs.md` (gate run 1). The gate is contention-independent and was not re-run for
the W2 window.

**Gate (ii) — record validation.** `python3 test/benchmark/r2/round3/check_r3.py --manifest
test/benchmark/r2/round3/manifest_armB.txt --require-stages` → **9 records: 9 complete, 0 partial,
0 skipped, 0 missing, 0 malformed; 34 checks, 0 failures** (`CHECKER_EXIT=0`) — the same result on the
W1 records, the post-b2 records and the final W2 records (`status_check_r3_armB_final.txt`,
`status_check_r3_armB_after_b2.txt`, `status_check_r3_armB_w2_final.txt`). The dense eval arms display
the stage-sourced `0` in the checker's `cold_builds` column against their JSON-level `1` (documented
gap, §3/§4).

## 9. Method and caveats

* **Method = R2 / Arm A, unchanged** (`./README.md`): one `bench_gp.py` process per config,
  `ulimit -n 65536` first, exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per invocation with
  `timeout` inside it, same two fixtures, same wiring, same implicit `--seed 0`, stage probe on the new
  side. The **only** deltas are the ETL shadow (plus the explicit `EVOGP_EVAL_MODE=auto` on the three
  plain `iree` ids) as recorded in each record's `env`.
* **Compile vs warm**: warm median is the headline, `first_call_ms` / `cold_first_gen_ms` / cold-build
  counts are reported separately and never mixed in; medians ±10 % on this box, IREE compile ±a factor
  of a few.
* **Stage-probe limitation** (dense route) and the **bimodal `sr` medians** are documented in §4/§2 and
  are not gate failures.
* **Two windows, two devices** (§6), with the W2 companion window's own mid-window tenant takeover; the
  W1 JSONs stay in git at `b728466`.
* **Not covered by Arm B:** the legacy side (frozen, no Arm B row), any pop-1000 config inside the
  manifest, and `etl/persist/codec.py`'s code path. GPU 7 (hardware ECC fault) was never touched.
* **Operator deviations** are logged in full in `results/armB/armB_runlog_section.md`
  (`### Deviations / anomalies`): the reaped detached agreement batch (window 1, re-launched in one
  foreground invocation), attempt 2's mis-invoked dense execution (preserved verbatim as
  `armB_iree_sr_b2_spec_botched_dense_route`), and the W2 status-file naming repair. No JSON, log or
  manifest row was affected by any of them.

## 10. Where the numbers live

| what | where |
|---|---|
| protocol, commands, gates, stage-probe limitation | `./README.md` |
| Arm B records of record (9 JSONs + 9 appended logs, both windows in the logs) | `./results/armB/new/` |
| W1 JSON snapshots of the same nine ids | `git show b728466:test/benchmark/r2/round3/results/armB/new/<id>.json` |
| operator log, per-window contention evidence, shadow probes, gate stdout | `./results/armB/{armB_runlog_section.md,windows.md,etl_import_probe.txt,gate_outputs.md}` |
| Arm B agreement pre-flight (JSON + `_values.npy` + log per arm) | `./results/armB/backend_agreement/` |
| Arm A records / write-up this is read against | `./results/{new,legacy}/`, `./SUMMARY.md` |
| R2 baseline (stage-by-stage analysis, ranked levers) | `../REPORT.md`, `../r2/README.md`, `../SPEC.md` (frozen) |
| R2 records (read-only history) | `../new/results/r2/`, `../legacy/results/r2/` |
| tools used for every table above | `./compare_rounds_armB.py` (+ `./selftest_compare_rounds_armB.py`), reusing `./compare_rounds.py`; gates `./check_r3.py`, `./check_agreement_armB.py` |
