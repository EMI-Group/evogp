# R2 run log — the measurement round 2 as actually executed

Operator log for `run_all.sh` (the R2 matrix runner; usage/rules: `README.md`, config contract:
`manifest.txt`, frozen round-1 methodology: `../SPEC.md`). One section per group, appended in the order the
groups were run; each section states the exact commands, the box/GPU state, the per-config status table and
the consistency checks. Raw evidence stays in the harness JSONs + appended per-config logs
(`../new/results/r2/<id>.{json,log}`, `../legacy/results/r2/<id>.{json,log}`) — this file is the summary and
the interpretation, not a second source of numbers. "derived" marks values this log computed from the JSONs
(per-rep medians); everything else is a harness field.

Conventions used below: `first_ms` = `first_call_ms` (eval tasks) / `cold_first_gen_ms` (training tasks);
`warm_ms` = `median_warm_ms` (eval tasks) / median of `per_gen_ms[1:]` (training tasks). For `sr` configs the
harness stores one `per_gen_ms` list per rep, so both per-rep medians are given — `check_r2.py` reports the
**rep0** value for training records. `--stage-probe` ("probe") is on for every `np_*` config except the two
`*_noprobe` controls; the legacy side is probe-free by construction.

Shared-box policy (README): one `flock -w 7200 /tmp/gpbench_critical.lock` per config (added by the runner,
never nested by the operator), `timeout -s INT -k 30 3600` inside the lock, `ulimit -n 65536`, medians ±10%.

---

## Light group (legacy + numpy) — 2026-09-12 14:44–14:48 HKT

**Operator:** worker `worker_T3_A160` (runner operator). **Scope:** only `new/results/r2/`,
`legacy/results/r2/`, this file; no harness/runner/probe/library file was touched.

### Environment

| item | value |
|---|---|
| repo root (cwd for every command) | repo root of the worker worktree |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |
| GPU | `CUDA_VISIBLE_DEVICES=6` — `nvidia-smi` before the run: `6, NVIDIA RTX A6000, 0 %, 1/49140 MiB`; during/after: `0–4 %, 1–4 MiB`; no ECC errors. GPUs 0/1 were 61–99 % busy with tenants; GPU 7 (uncorrectable-ECC fault) never touched |
| lock | `/tmp/gpbench_critical.lock`, acquired/released per config; the runner's `flock -w 7200` never waited (no lock-wait messages in any log), zero interruptions |
| engine under test | `new` = `PYTHONPATH=src` (this worktree, `evogp_commit=6f773b4`, `etl 0.1.0`, backend default `numpy`); `leg` = `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils` (legacy CUDA checkout, `evogp_file=/tmp/evogp_legacy_bench/src/evogp/__init__.py`; `evogp_commit` in a JSON is the harness's cwd repo, not the legacy checkout) |
| fixture | `test/benchmark/fixtures/forest200_tree64_random.npz` present (new-side-generated) → the six `*_fixture200_*` configs ran instead of GUARD-SKIP |

### Commands (exact; one bash call per chunk)

```bash
bash test/benchmark/r2/run_all.sh --group light --dry-run      # 23 configs planned, nothing executed, last_run.txt NOT written
# chunk 1
bash test/benchmark/r2/run_all.sh --group light --only leg_c5r2,leg_fixture64_n1024,leg_fixture64_n10000
# chunk 2
bash test/benchmark/r2/run_all.sh --group light --only leg_fixture200_n64,leg_fixture200_n1024,leg_fixture200_n10000
# chunk 3
bash test/benchmark/r2/run_all.sh --group light --only leg_fresh16,leg_sr_a,leg_sr_b,leg_c1r2,leg_c5b_r2
# chunk 4
bash test/benchmark/r2/run_all.sh --group light --only np_c5r2,np_fixture64_n1024,np_fixture64_n10000
# chunk 5
bash test/benchmark/r2/run_all.sh --group light --only np_fixture200_n64,np_fixture200_n1024,np_fixture200_n10000
# chunk 6
bash test/benchmark/r2/run_all.sh --group light --only np_fresh16,np_sr_a,np_sr_b,np_c1r2
# chunk 7
bash test/benchmark/r2/run_all.sh --group light --only np_c5r2_noprobe,np_sr_a_noprobe
```

Each config was a separate process (`PY` + `PYTHONPATH` + `CUDA_VISIBLE_DEVICES=6` + the config's flags;
`ETL_DISABLE_LOCATIONS=1` and backend `numpy` are the runner defaults). The exact per-config command is
reproduced in the header of that config's log.

### Per-config status (23/23 exit 0, all `complete=true`)

| id | side | task | exit | first_ms | warm_ms | notes |
|---|---|---|---|---|---|---|
| leg_c5r2 | old | eval64 | 0 | 20.62 | 0.633 | 25 calls; re-run of the same-session sanity id |
| leg_fixture64_n1024 | old | eval64 | 0 | 20.64 | 0.639 | 11 calls, rows tiled 64→1024 |
| leg_fixture64_n10000 | old | eval64 | 0 | 20.29 | 0.682 | 11 calls, rows tiled 64→10000 |
| leg_fixture200_n64 | old | eval64 | 0 | 20.15 | 0.632 | 25 calls, 200-tree fixture |
| leg_fixture200_n1024 | old | eval64 | 0 | 20.54 | 0.681 | 11 calls |
| leg_fixture200_n10000 | old | eval64 | 0 | 19.14 | 1.019 | 11 calls |
| leg_fresh16 | old | evalfresh | 0 | 14.46 | 0.632 | 10 calls, 16 fresh trees |
| leg_sr_a | old | sr | 0 | 480.68 | 1.602 | 2 reps × 10 gens, pop 200; rep0 1.6 / rep1 1.6 ms (derived); best_final −0.2257 |
| leg_sr_b | old | sr | 0 | 467.09 | 1.619 | 2 reps × 20 gens, pop 1000; rep0 1.6 / rep1 1.6 ms; best_final −0.2257 |
| leg_c1r2 | old | sr | 0 | 494.27 | 1.625 | 2 reps × 100 gens, pop 200; rep0 1.650 / rep1 1.599 ms (derived); best_final −0.0868; re-run |
| leg_c5b_r2 | old | evalfresh | 0 | 15.37 | 0.642 | 10 calls, pop 1000; re-run |
| np_c5r2 | new | eval64 | 0 | 216.62 | 21.056 | probe |
| np_fixture64_n1024 | new | eval64 | 0 | 218.93 | 21.505 | probe |
| np_fixture64_n10000 | new | eval64 | 0 | 221.05 | 23.120 | probe |
| np_fixture200_n64 | new | eval64 | 0 | 1009.10 | 93.674 | probe |
| np_fixture200_n1024 | new | eval64 | 0 | 1025.21 | 96.518 | probe |
| np_fixture200_n10000 | new | eval64 | 0 | 1045.64 | 105.371 | probe |
| np_fresh16 | new | evalfresh | 0 | 102.05 | 9.003 | probe |
| np_sr_a | new | sr | 0 | 1356.30 | 84.09 / 22.07 | probe; per-rep medians rep0 / rep1; best_final −0.2511 |
| np_sr_b | new | sr | 0 | 5549.12 | 650.36 / 670.08 | probe; best_final −0.2054 |
| np_c1r2 | new | sr | 0 | 1398.39 | 89.21 / 88.52 | probe; best_final −0.1720 |
| np_c5r2_noprobe | new | eval64 | 0 | 218.62 | 20.838 | A/B control (no probe) |
| np_sr_a_noprobe | new | sr | 0 | 1467.32 | 85.78 / 24.63 | A/B control (no probe); best_final −0.2511 |

### Stage-probe highlights (`stages.reps[].totals`; warm per-call split = derived median over `records[i>0]`)

| id | rep | cold_builds | cold_build_ms | warm_misses | warm_build_miss_ms | warm per-call total_ms | warm split (median) |
|---|---|---|---|---|---|---|---|
| np_c5r2 | 0 | 48 | 151.5 | 0 | 0.0 | 21.06 | decode 1.66 + build_hit 9.66 + run 8.59 + readback 0.63 + upload 0.29 + loss 0.14 + misc 0.07 (build_hit 45.9 %, run 40.8 %, decode 7.9 %) |
| np_fixture64_n1024 | 0 | 48 | 152.3 | 0 | 0.0 | 21.51 | decode 1.65 + build_hit 9.70 + run 8.93 + readback 0.63 + upload 0.34 + loss 0.15 + misc 0.07 (45.1 / 41.5 / 7.7 %) |
| np_fixture64_n10000 | 0 | 48 | 152.7 | 0 | 0.0 | 23.12 | decode 1.66 + build_hit 9.76 + run 10.17 + readback 0.64 + upload 0.62 + loss 0.18 + misc 0.08 (42.2 / 44.0 / 7.2 %) |
| np_fixture200_n64 | 0 | 155 | 882.3 | 0 | 0.0 | 93.67 | decode 7.28 + build_hit 40.79 + run 42.50 + readback 2.03 + upload 0.79 + loss 0.19 + misc 0.16 (43.5 / 45.4 / 7.8 %) |
| np_fixture200_n1024 | 0 | 155 | 894.1 | 0 | 0.0 | 96.52 | decode 7.33 + build_hit 41.24 + run 44.49 + readback 2.03 + upload 1.00 + loss 0.21 + misc 0.17 |
| np_fixture200_n10000 | 0 | 155 | 907.1 | 0 | 0.0 | 105.37 | decode 7.34 + build_hit 41.39 + run 51.60 + readback 2.14 + upload 2.28 + loss 0.33 + misc 0.28 |
| np_fresh16 | 0 | 15 | 76.0 | 0 | 0.0 | 9.00 | decode 0.77 + build_hit 3.78 + run 3.96 + readback 0.19 + upload 0.13 + loss 0.14 + misc 0.06 |
| np_sr_a | 0 | 155 | 834.5 | 362 | 1164.2 | 84.09 | churns: 155 cold builds at gen 0, then 362 cache misses |
| np_sr_a | 1 | 0 | 0.0 | 0 | 0.0 | 22.07 | fully warm (632 build hits / 10 gens) |
| np_sr_b | 0 | 788 | 4644.7 | 3972 | 14636.5 | 650.36 | cache-thrash dominated (pop 1000, tree_len 128) |
| np_sr_b | 1 | 760 | 5539.5 | 3927 | 15514.3 | 670.08 | same steady state in the second rep |
| np_c1r2 | 0 | 155 | 831.1 | 3946 | 18177.8 | 89.21 | gen-0 cold builds + steady misses |
| np_c1r2 | 1 | 146 | 1448.3 | 3683 | 17363.7 | 88.52 | same steady state (in-process cache evicts) |

Cache-behaviour reading: on fixture workloads every warm call is fully warm (`warm_misses = 0`, all build
hits), so the warm split is ~46 % build-hit bookkeeping + ~41 % run; on the `sr` workloads the population's
structures change every generation, so warm calls still miss the per-structure cache (hundreds–thousands of
misses) and the miss cost (11.6 s / 18.2 s per rep at pop 200–1000) dominates the warm per-gen time.

### Consistency checks

1. **Legacy C5 anchor.** `leg_c5r2` warm median **0.633 ms** vs round-1 anchor **0.6042 ms** (`legacy/results/C5.json`) = **+4.7 %**, and vs the earlier same-session sanity sample committed at HEAD (`legacy/results/r2/leg_c5r2.json`: 0.6028 ms, first 22.31 ms) = **+5.0 %** — both inside the ±10 % shared-box band; first call 20.62 ms vs 21.77 / 22.31 ms.
   **nan_count / fitness_mean:** these keys do **not** exist in R2 harness JSONs (`bench_gp.py`'s eval tasks record timings only — the keys come from the round-1 `legacy/run_matrix.py` driver). Verified instead two ways: (a) the same-session driver artifact `legacy/results/r2/C5.json` holds `nan_count=3`, `fitness_mean=-18.048504493275626`; (b) an independent read-only re-derivation on the same fixture with the same problem construction as the harness's `eval64` path — **legacy engine: nan_count 3/64, fitness_mean −18.048504493275626 (exact anchor match)**; new ETL engine: nan_count 3/64, fitness_mean −18.048504309576067 (rel. 1.0e−8), NaN mask identical (`sha256(mask)[:16] = 07172d4234502fee` on both engines).
2. **Numpy anchors.** `np_c5r2` 21.056 ms vs prior 20.787 ms = **+1.3 %**; `np_c1r2` 89.21 (rep0) / 88.52 (rep1) ms vs prior 88.07 ms = **+1.3 % / +0.5 %** — both within noise (the anchor runs had no probe; the probe tax is included here and is small, see below).
3. **A/B probe overhead.**
   - `np_c5r2` (probe) 21.056 vs `np_c5r2_noprobe` 20.838 ms → **+0.218 ms, +1.05 %**.
   - `np_sr_a` (probe) vs `np_sr_a_noprobe` warm per-gen medians: rep0 84.09 vs 85.78 → **−2.0 %**; rep1 22.07 vs 24.63 → **−10.4 %** (the probe run *measures faster*, i.e. the difference is box/cache noise, not instrumentation).
   - Context for the "~1.3 ms (~5–6 %) probe tax" expectation: the h2 smoke pair (`new/results/r2/smoke/h2_np_eval64{,_noprobe}.json`, `--eval-reps 3` → only 2 warm samples) shows +1.34 ms (+6.2 %), while two *noprobe* runs of the same eval64 workload here differ by 3.4 % (20.838 vs the smoke's 21.537). So this round's A/B resolves the tax only to "≲1 ms on eval64, unresolved (< noise) on `sr`" — it does not change any R2 comparison.
4. **`leg_fixture200_*` cross-engine verdict: PASS.** All three configs exited 0 with no guard, warning or exception in their logs: the legacy CUDA engine accepts and evaluates the **new-side-generated** fixture. Numeric cross-check on the same fixture (read-only, both engines): `nan_count` 12/200 with identical NaN mask (`78da3af9dcdf813c`), `fitness_mean` −6969.418862 (legacy) vs −6969.418338 (new, rel. 7.5e−8), `finite_min` −650003.5 vs −650003.4375 (rel. 9.6e−8), `finite_max` −2.72265625 identical. Timings: warm median 0.632 ms (n64, 25 calls) / 0.681 ms (n1024) / 1.019 ms (n10000).
5. **Data-size scaling (n64 → n1024 → n10000), fixture64:** warm per-call 21.06 (np_c5r2, 25 calls) → 21.51 → 23.12 ms; the components move as expected for a *structure-bound* cold path and a *data-bound* run path: `cold_builds` constant 48 and `cold_build_ms` flat 151.5 → 152.3 → 152.7 ms (build cost ignores the data length), `build_hit` flat 9.66 → 9.70 → 9.76 ms (cache hits are keyed by structure), `decode` flat 1.66 ms, `readback` flat 0.63–0.64 ms, while `run` grows 8.59 → 8.93 → 10.17 ms and `upload` 0.29 → 0.34 → 0.62 ms. Same shape on fixture200: warm 93.67 → 96.52 → 105.37 ms, `cold_builds` 155 constant, `cold_build_ms` 882 → 894 → 907 ms, `run` 42.50 → 44.49 → 51.60 ms. (Legacy side for comparison: 0.632 → 0.681 → 1.019 ms, i.e. also data-bound but ~50× smaller absolute.)
6. **Exit codes / anomalies:** 23/23 exit 0, no TIMEOUT (124/137), no INTERRUPT (130), no FAIL, no GUARD-SKIP (→ no missing JSON), no `partial`/`interrupted` record; every JSON carries `complete=true` and the runner's per-chunk summary read `ok: N skipped: 0 failed: 0` each time.

### Validation

```bash
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2
#   -> 38 records: 35 complete, 2 partial, 1 skipped, 0 missing, 0 malformed; 93 checks, 2 failures
python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt
#   -> 35 records: 23 complete, 0 partial, 0 skipped, 12 missing, 0 malformed; 96 checks, 12 failures (all 12 = the missing iree ids)
```

Both invocations are expected in this state: the **2 failures** in directory-scan mode are the prior-round
artifacts `legacy/results/r2/C5.json` + `C5b.json` (written by the round-1 `legacy/run_matrix.py` driver during
the earlier same-session sanity pass; they are not R2 manifest ids and fail the R2 key schema by design). The
12 manifest failures are exactly the 12 `iree_*` ids, which the iree operator runs later — **zero failures are
attributable to the light group**.

### Wall clock

| phase | wall |
|---|---|
| dry run | < 1 s |
| chunk 1 / 2 / 3 / 4 / 5 / 7 | 10 s / 10 s / 22 s / 10 s / 16 s / 12 s |
| chunk 6 (`np_sr_b` + `np_c1r2`, 2 reps each) | 117 s |
| **total (dry run + all 7 chunks)** | **≈ 3 min 17 s** (first chunk 14:44:35, last chunk end 14:48:05) |
| harness-reported `harness_wall_s` sum over the 23 configs | 142.0 s (each process pays ~1.0–1.4 s import/startup; the biggest single configs are `np_c1r2` 48.9 s and `np_sr_b` 48.0 s) |

Far below the "10–20 min" estimate: the legacy side is sub-second per workload and the numpy mirrors are
1–50 s each.

### Deviations / anomalies

* **Re-run of three coincident ids.** `leg_c5r2`, `leg_c1r2`, `leg_c5b_r2` were already committed ~30 min
  earlier by the same-session legacy sanity pass (same paths/tags, different process). This run intentionally
  overwrote those JSONs and **appended** to their logs (`leg_c5r2.log` now holds the 14:0x sanity block plus the
  14:44 block of this run) — do not read an appended log as one run.
* **Requested `nan_count`/`fitness_mean` keys are not produced by the R2 harness** (see check 1). The anchor
  values were confirmed from the round-1 driver artifact and re-derived independently instead; no file outside
  this log's scope was written.
* **A/B probe overhead is below the resolver of a single pair** (check 3): +1.05 % on eval64, "negative" on the
  `sr` pair. Treat the probe tax as ≲1 ms and not a confound for the round's conclusions.
* `last_run.txt` (in this directory) is the runner's generated id → exit-code table for the *last* invocation
  only (chunk 7); it is documented as disposable by `README.md` and is not committed as evidence — the chunk
  summaries above are.
* Shared-box noise: GPU 0/1 were at 61–99 % with tenant jobs throughout, GPU 6 stayed at 0–4 % with 1–4 MiB
  used, and no config showed a warped median (max ≤ 10× median); still treat all medians as ±10 %.
* No ECC event, no lock contention, no termination by `timeout`, no IREE/FD-limit involvement (light group is
  numpy + legacy CUDA only).

---

## IREE group (eval configs) — 2026-09-12 14:56–15:13 HKT

**Operator:** worker `worker_T3_A161` (iree eval-config operator). **Scope:** only the 10 eval configs of the
iree group — `new/results/r2/iree_*.{json,log}` and this section. No harness/runner/validator/library file and no
other config's results were touched. The training pair `iree_sr_a` / `iree_sr_b2` is **not** part of this entry —
it is measured separately by the training-pair operator (see the hand-off note at the end).

### Environment

| item | value |
|---|---|
| repo root (cwd for every command) | this worker's worktree root |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |
| GPU | `CUDA_VISIBLE_DEVICES=6` — the runner default, **kept**, because GPU 6 was the least-busy non-7 device. Snapshot before (14:55:52): `6, NVIDIA RTX A6000, 0 %, 1 MiB`; the rest of the box was busy — GPU 0 / 1 at 65 % / 99 % with ~19.7 / 20.1 GiB of tenant allocation, GPUs 2–5 at 0 % util but 6.8–24.4 GiB held by tenants, GPU 7 = 2263 uncorrected ECC errors (never touched). Snapshot after (15:12:48): `6, …, 0 %, 1 MiB` (GPU 0/1 at 96 %/89 %). Inside the configs `nvidia_smi_used_mb` on GPU 6 was 4 → 501 MiB, `free_mb_start` ≈ 48 409 MiB; peak torch allocation of any config = 23.14 MiB (`iree_fixture200_n10000`) |
| lock | `/tmp/gpbench_critical.lock` — exactly one `flock -w 7200` per config (added by the runner, never nested); no lock-wait delay in any log |
| engine under test | `new` = `PYTHONPATH=src` (this worktree, no source file touched: the JSONs carry `evogp_commit=dc5c0f55425e…` for the first config and the SHA of the preceding chunk commit for the later ones — `1eb9a75`, `87b6122`, `e73080a`, `28def1f`, all the same engine tree), `etl 0.1.0`, backend `iree`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`, `EVOGP_ETL_IREE_DEVICE=0`, `ETL_DISABLE_LOCATIONS=1` (runner defaults; recorded in each JSON's `env`) |
| fixtures | `test/goldens/forest_random.npz` (64 trees) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees) present → no GUARD-SKIP |
| FD limit | `nofile_rlimit = [65536, 65536]` in every JSON (the runner's `ulimit -n 65536` applied) |

### Commands (exact; one bash call per chunk)

```bash
bash test/benchmark/r2/run_all.sh --only iree_fixture64_n64,iree_fixture64_n1024,iree_fixture64_n10000,iree_fixture200_n64,iree_fixture200_n1024,iree_fixture200_n10000,iree_fresh16,iree_fixture64_n64_noprobe,iree_fresh16_noprobe,iree_c7r --dry-run   # 14:55:52, 10 configs planned, nothing executed, last_run.txt NOT written
# chunk 1 (calibration: per-build rate + FD/build)
bash test/benchmark/r2/run_all.sh --only iree_fixture64_n64 --timeout 1800
# chunk 2
bash test/benchmark/r2/run_all.sh --only iree_fixture64_n1024,iree_fixture64_n10000 --timeout 1800
# chunk 3
bash test/benchmark/r2/run_all.sh --only iree_fixture200_n64,iree_fixture200_n1024,iree_fixture200_n10000 --timeout 1800
# chunk 4 (fresh16 + the two probe-off A/B controls)
bash test/benchmark/r2/run_all.sh --only iree_fresh16,iree_fixture64_n64_noprobe,iree_fresh16_noprobe --timeout 1800
# chunk 5 (round-1 C7 anchor re-run)
bash test/benchmark/r2/run_all.sh --only iree_c7r --timeout 1200
```

Each config is its own process (`PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=6`, the config's flags, `--tag` =
uppercase id); the exact per-config command is reproduced in that config's log header. Before chunk 3 the FD
projection from chunk 1 (`33 fds / cold build × ≤400 builds ≈ 13 k fds` + warm-hit leak) was checked against
the 65 536 limit — safe, so all three ran in one call.

### Per-config status (10/10 exit 0, `complete=true`, `partial=false`, no `skipped`, no `skip_reason`)

| id | task | trees | data rows | exit→status | first_ms | warm median ms | cold_builds | cold_build_ms | warm_misses | fds end | peak MiB | harness wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iree_fixture64_n64 | eval64 | 64 | native | 0→OK | 82065.19 | 69.61 | 48 | 81854.0 | 0 | 2395 | 0.08 | 84.6 |
| iree_fixture64_n1024 | eval64 | 64 | 1024 | 0→OK | 74575.53 | 73.39 | 48 | 74369.1 | 0 | 1947 | 0.80 | 76.7 |
| iree_fixture64_n10000 | eval64 | 64 | 10000 | 0→OK | 16636.48 | 79.37 | 48 | 16423.7 | 0 | 1947 | 7.51 | 18.1 |
| iree_fixture200_n64 | eval64 | 200 | native | 0→OK | 105645.73 | 229.49 | 155 | 105201.1 | 0 | 5819 | 0.25 | 112.0 |
| iree_fixture200_n1024 | eval64 | 200 | 1024 | 0→OK | 335615.71 | 238.46 | 155 | 335082.0 | 0 | 5371 | 2.46 | 339.3 |
| iree_fixture200_n10000 | eval64 | 200 | 10000 | 0→OK | 86662.64 | 260.36 | 155 | 86089.4 | 0 | 5371 | 23.14 | 89.9 |
| iree_fresh16 | evalfresh | 16 | native | 0→OK | 14039.21 | 24.50 | 15 | 13950.8 | 0 | 859 | 0.08 | 15.2 |
| iree_c7r | sr | 50 × 3 gens | native | 0→OK | 20737.42 | 5229.58 (derived median of `per_gen_ms[1:]`) | 39 | 20016.0 | 24 | 2171 | 0.62 | 32.8 |
| iree_fixture64_n64_noprobe | eval64 | 64 | native | 0→OK | 25532.48 | 68.02 | – (no probe) | – | – | 2395 | 0.08 | 28.2 |
| iree_fresh16_noprobe | evalfresh | 16 | native | 0→OK | 7320.03 | 19.79 | – (no probe) | – | – | 859 | 0.08 | 8.5 |

`first_ms` = harness `first_call_ms` (eval tasks) / `cold_first_gen_ms` (`iree_c7r`). `cold_builds` /
`cold_build_ms` / `warm_misses` are `stages.reps[0].totals` fields — **not** the top-level `cold_builds` (that
one is the guard object: `{enforced: true, limit: 300|400|200, misses_max: 48|155|15|63, exceeded: false}` for
every config, i.e. the `--max-cold-builds` budget was never reached and no config guard-stopped).

Top-level `per_call_ms` warm min/median/max (excludes call 0): n64 68.67 / 69.61 / 93.73; n1024 72.24 / 73.39 /
80.93; n10000 77.48 / 79.37 / 89.64; fixture200 n64 226.06 / 229.49 / 352.71; n1024 231.16 / 238.46 / 249.81;
n10000 248.58 / 260.36 / 287.96; fresh16 21.49 / 24.50 / 31.49; noprobe pair 64.86 / 68.02 / 104.96 and
19.36 / 19.79 / 44.12. `iree_c7r` `per_gen_ms = [20737.42, 4310.13, 6149.04]`,
`per_gen_best_fitness = [-0.2510969936847687] × 3`.

### Warm stage decomposition (median over `stages.reps[0].records[i>0]`, ms per call)

These are **fixture** configs, so every warm call is fully warm: `build_miss = 0.00` and `warm_misses == 0` in
`totals` on all six — that equality is the proof that the warm path never recompiled anything. `build_calls` per
call = `trees_invoked`/`calls`, the rest of the call is bookkeeping: `total_ms` = decode + build_hit + build_miss
+ run + readback + upload + loss + misc.

| id | decode | build_hit | build_miss | run | readback | upload | loss | misc | total | dominant |
|---|---|---|---|---|---|---|---|---|---|---|
| iree_fixture64_n64 | 1.68 | 11.09 | 0.00 | 35.76 | 20.22 | 0.50 | 0.17 | 0.09 | 69.61 | run 51 % + readback 29 % |
| iree_fixture64_n1024 | 1.70 | 11.16 | 0.00 | 34.73 | 21.63 | 4.07 | 0.20 | 0.10 | 73.38 | run 47 % + readback 29 % |
| iree_fixture64_n10000 | 1.72 | 11.36 | 0.00 | 26.78 | 24.62 | 14.68 | 0.25 | 0.11 | 79.37 | readback 31 % + run 34 % + upload 18 % |
| iree_fixture200_n64 | 7.36 | 46.36 | 0.00 | 108.39 | 65.31 | 1.52 | 0.23 | 0.27 | 229.49 | run 47 % + readback 28 % |
| iree_fixture200_n1024 | 7.63 | 46.35 | 0.00 | 107.98 | 68.64 | 6.17 | 0.27 | 0.31 | 238.46 | run 45 % + readback 29 % |
| iree_fixture200_n10000 | 7.35 | 46.08 | 0.00 | 90.87 | 71.08 | 45.39 | 0.32 | 0.45 | 260.36 | readback 27 % + run 35 % + upload 17 % |
| iree_fresh16 (control pair) | 1.08 | 5.93 | 0.00 | 9.48 | 7.61 | 0.27 | 0.21 | 0.10 | 24.68 / 24.50 | run 39 % + readback 31 % |

Reading: the IREE warm per-call cost is **run + readback** (the device round-trip: `run_ms` and the `d2h` of
the outputs), not decode and not the cache bookkeeping — the same shape as the numpy mirrors, but scaled up
(`np_c5r2` warm 21.06 ms with `run` 8.59 + `readback` 0.63 ms; the IREE readback is ~30× the numpy one at the
same tree count). `build_hit_ms` (11 – 46 ms) is the per-tree cache lookup, `decode_ms` is host-side tree
decoding; the *cold* path is a different animal — see the calibration below.
`iree_c7r` is the one training config here and is **not** fully warm: 39 cold builds + 24 warm misses
(63 misses total, `warm_misses` 24, `warm_build_miss_ms` 10336.3) — its warm-ish per-gen median 5229.58 ms is
dominated by `build_miss` 5168.16 ms, exactly as the `np_sr_a` behaviour predicts for a churning population.

### Build-rate calibration (chunk 1, and the same rate measured on the other configs)

| config | cold_builds | cold_build_ms | **ms / build** | fds after_cold − start | **fds / cold build** | fds end |
|---|---|---|---|---|---|---|
| iree_fixture64_n64 (calibration) | 48 | 81854.0 | **1705.3** | 1588 | **33.08** | 2395 |
| iree_fixture64_n1024 | 48 | 74369.1 | 1549.4 | 1588 | 33.08 | 1947 |
| iree_fixture64_n10000 | 48 | 16423.7 | 342.2 | 1588 | 33.08 | 1947 |
| iree_fixture200_n64 | 155 | 105201.1 | 678.7 | 5012 | 32.34 | 5819 |
| iree_fixture200_n1024 | 155 | 335082.0 | 2161.8 | 5012 | 32.34 | 5371 |
| iree_fixture200_n10000 | 155 | 86089.4 | 555.4 | 5012 | 32.34 | 5371 |
| iree_fresh16 | 15 | 13950.8 | 930.1 | 532 | 35.47 | 859 |
| iree_c7r (63 misses total) | 39 | 20016.0 | 513.2 | 1300 | 33.33 | 2171 |

* **FD rate is the reproducible number: 32–35 fds per compiled structure** — identical (33.08 / 32.34) for every
  config that shares a tree count, i.e. it is a function of the build, not of the data length. On top of it the
  warm path leaks ~0.6–3 fds per build call (`(fds_end − fds_start) / build_calls` = 1.35 – 3.27 on the fixture
  configs). Projection that was checked before chunk 3: 155 builds × 32.3 ≈ 5.0 k fds (measured `after_cold`
  5051 ✓), and even the full 400-build budget would be ≈ 13 k fds ≪ 65 536; the whole group peaked at 5819 fds.
* **ms / build is *not* reproducible on this shared box: 342 → 2162 ms per structure (6.3× spread) for identical
  build counts.** `iree_fixture64_n10000` (48 builds) compiled in 16.4 s right after `iree_fixture64_n1024`'s
  identical 48 builds took 74.4 s, and `iree_fixture200_n1024` (155 builds) took 335.1 s versus 105.2 s / 86.1 s
  for the two other 155-build configs. Take ~1.1 s/build at face value (663 builds, 733.0 s of
  `cold_build_ms` in total) and treat `first_call_ms`/`cold_build_ms` as ±(a factor of a few), not ±10 %.
* Group-level economics: **733.0 s of the group's 805.4 s total `harness_wall_s` (91 %) is per-structure IREE
  compilation** — the IREE backend is compile-bound at population scale, exactly as `README.md` predicts. For
  contrast the numpy mirrors build the same structures in ~3.2 ms each (`np_c5r2`: 48 builds / 151.5 ms), i.e.
  IREE's build is ~110–680× the numpy interpreter's, and the warm per-call penalty is ~3.3× on the 64-tree
  fixture (69.6 vs `np_c5r2` 21.1 ms) and ~2.5× on the 200-tree one (229.5 vs `np_fixture200_n64` 93.7 ms).

### A/B probe overhead

| pair | probe warm median | noprobe warm median | delta | first_call probe vs noprobe | harness wall |
|---|---|---|---|---|---|
| `iree_fixture64_n64` vs `iree_fixture64_n64_noprobe` | 69.61 ms | 68.02 ms | **+1.59 ms / +2.34 %** | 82.1 s vs 25.5 s (compile-bound, not probe) | 84.6 vs 28.2 s |
| `iree_fresh16` vs `iree_fresh16_noprobe` | 24.50 ms | 19.79 ms | **+4.71 ms / +23.8 %** | 14.04 s vs 7.32 s | 15.2 vs 8.5 s |

The eval64 pair (+2.3 %, i.e. ≈ the light group's ≲1 ms / ≈1 % probe tax on 21 ms) is inside the ±10 % band. The
`fresh16` pair is **not** resolvable either: the noprobe control also compiled its 15 structures ~2× faster
(7.32 s vs 14.04 s, same 15 builds) because it landed in a faster box window, so the +4.7 ms mixes probe tax with
box noise — the probe does add per-call bookkeeping proportional to `build_calls`, but this round cannot separate
it. Neither reading changes any R2 conclusion.

### C7 anchor vs round 1 (pop 50 / gens 3 / tree 32, `--max-cold-builds 200`)

| quantity | round 1 (`new/results/C7.json`) | this round (`iree_c7r`) | delta |
|---|---|---|---|
| `per_gen_ms` | [21830.84, 12591.85, 26017.30] | [20737.42, 4310.13, 6149.04] | gen0 −5 %, gen1 −66 %, gen2 −76 % |
| generation loop (`probe_span_ms` here / `generation_loop_s` r1) | 60.44 s | 31.20 s | **−48 %** |
| `per_gen_best_fitness` | −0.2510969936847687 (×3) | −0.2510969936847687 (×3) | **bit-identical** |
| cold builds / misses | 50 builds at gen 0 | 39 cold builds + 24 warm misses = 63 | guard not tripped (`limit 200`) |

The same seed reproduces the round-1 trajectory **exactly** on best fitness, so the speed-up is engine/box, not a
different workload. Its source is visible in the split: gen 0 costs 39 builds = 20.0 s, gens 1–2 add only 24
misses = 10.3 s, and the whole run is 31.2 s vs 60.4 s — the same order-of-magnitude as the round-1 build cost
(~0.5 s/build) with the box-dependent spread documented above. `iree_c7r` **did not** guard-stop, so it is a
complete record, not a partial/SKIP outcome.

### Validation

```bash
python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt     # exit 1
#   -> 35 record(s): 33 complete, 0 partial, 0 skipped, 2 missing, 0 malformed; 118 check(s), 2 failure(s)
#      FAIL iree_sr_a  (missing JSON) ; FAIL iree_sr_b2 (missing JSON)
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2   # exit 1
#   -> 48 record(s): 45 complete, 2 partial, 1 skipped, 0 missing, 0 malformed; 115 check(s), 2 failure(s)
#      FAIL C5  (missing required eval keys ['task','config','complete'], legacy/results/r2/C5.json)
#      FAIL C5b (same, legacy/results/r2/C5b.json)
#      notes: h1_guard SKIP (cold-builds-exceeded:4>1); leg_smoke tag 'LEG-smoke' != 'ID.upper()' (prior-round artifact)
```

Both outcomes are exactly the expected ones: the **2 manifest failures are the two training-pair ids** that this
entry deliberately does not cover (they were still MISSING when this section was written; the training-pair
operator writes them), and the **2 directory-scan failures are pre-existing prior-round driver artifacts**
(`C5.json` / `C5b.json`, written by round 1's `legacy/run_matrix.py` — same 2 failures the light-group entry
reported). **Zero failures are attributable to the 10 configs of this group**: all 8 probed ids carry a valid
`stages` block and the 2 `*_noprobe` controls correctly carry none.

### Wall clock

| phase | wall |
|---|---|
| dry run | < 1 s (14:55:52) |
| chunk 1 (calibration) | 86 s (14:56:01 → 14:57:27) |
| chunk 2 | 108 s (14:57:53 → 14:59:41) |
| chunk 3 (three fixture200 configs) | 668 s (14:59:46 → 15:10:54) |
| chunk 4 | 59 s (15:11:00 → 15:11:59) |
| chunk 5 (C7) | 36 s (15:12:05 → 15:12:41) |
| **total (dry run + 5 chunks)** | **≈ 15 min 58 s** (14:55:52 → 15:12:41, incl. commit/telemetry pauses between chunks) |
| harness-reported `harness_wall_s` sum over the 10 configs | 805.4 s (of which 733.0 s = `cold_build_ms`) |

### Deviations / anomalies

* **Compile-rate variance (main anomaly).** 6.3× spread in ms/build across configs with identical build counts
  (see the calibration table). Every value is a genuine harness measurement of that window; the group's robust
  statement is "~1.1 s/build average, ±a factor of a few", and `first_call_ms` for all `iree_*` evals should be
  read as "cold-build total", not as a stable hardware constant.
* **No guard trip, no timeout, no interrupt.** 10/10 exit 0; every `cold_builds.exceeded = false`; no 124/137/130;
  no FAIL; no GUARD-SKIP (→ no missing JSON for this group); every JSON `complete=true`, `partial=false`.
* **warm_misses == 0 held everywhere it must**: the 6 fixture configs + `iree_fresh16` (`warm_misses` 0,
  `warm_build_miss_ms` 0.0) and, as expected, the 2 probe-off controls have no `stages` block at all. The single
  non-zero case is `iree_c7r` (`warm_misses` 24), which is a **training** config whose population changes every
  generation — that is the documented cache-miss behaviour, not a regression.
* **FD pressure did not materialise**: max `fds.end` = 5819 of the 65 536 limit; `ulimit -n` was raised by the
  runner and `nofile_rlimit` confirms `[65536, 65536]` inside every process.
* **nanobind reference-leak warning in every IREE log tail** (`leaked function "…"`, "likely caused by a reference
  counting issue in the binding code"). Pre-existing IREE-python binding teardown noise: it appears after the
  harness has written its JSON and does not affect the exit code (all 0) or any measured number. Not patched here.
* **GPU contention** limited to the other tenants (GPU 0/1 65–99 % throughout, GPU 3's allocation dropped
  15.7 → 6.8 GiB mid-round); GPU 6 stayed at 0 % util / ≤501 MiB used and GPU 7 was never touched, so no config
  was run under local contention — the variance above is therefore inside the IREE compile path / box-wide noise,
  not explained by a neighbour on GPU 6.
* `test/benchmark/r2/last_run.txt` was rewritten by every chunk (it holds chunk 5 only, as documented) and is
  **not** committed; the chunk summaries above are the evidence.
* No harness/runner/validator/library file was modified; the only files written are the 10 `.json` + 10 `.log`
  pairs under `test/benchmark/new/results/r2/` and this section.
* **Hand-off (not this operator's configs):** `iree_sr_a` / `iree_sr_b2` are the remaining 2 manifest ids. The
  measured FD rate (≈33 fds per compiled structure) makes `iree_sr_a` comfortable (its `np_sr_a` mirror churned
  ~520 builds ⇒ ≈17 k fds) but implies `iree_sr_b2`'s 2200-build budget projects to ≈73 k fds, i.e. **above the
  65 536 `ulimit -n`** — the training-pair operator should expect `EMFILE` before the `--max-cold-builds 2200`
  guard trips, and may want a lower budget. Reported here rather than changed: the runner's budget is frozen.

---

## IREE group (training pair: iree_sr_a, iree_sr_b2) — 2026-09-12 15:16–15:27 HKT

**Operator:** worker `worker_T3_A165` (IREE training-pair operator). **Scope:** the last 2 manifest ids —
`new/results/r2/iree_sr_a.{json,log}`, `new/results/r2/iree_sr_b2.{json,log}`, this section, and one bounded
validator fix in `check_r2.py` (`discover()`, see *Validator* below). No `src/**`, no `bench_gp.py`, no
`run_all.sh`, no `CONTEXT.md`/`README.md`, no other config's results touched.

### Environment

| item | value |
|---|---|
| repo root (cwd for every command) | this worker's worktree root |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |
| GPU | `CUDA_VISIBLE_DEVICES=6` — the runner default, **kept**: GPU 6 was the least-busy non-7 device (0 % util, 1 MiB). Before chunk 1 (15:16:17): `6, …, 0 %, 1 MiB` while GPU 0/1 sat at 99 %/82 % with ~20 GiB each and GPUs 2/3/5 held 15.6–24.4 GiB for tenants; GPU 7 (2263 uncorrected ECC errors) never touched. Runner's own snapshots: `iree_sr_a` before `6, …, 0, 1` → after `6, …, 5, 4`; `iree_sr_b2` before `6, …, 0, 4` → after `6, …, 0, 1`. After the chunk (15:27:49): `6, …, 0 %, 4 MiB`. Inside the configs `nvidia_smi_used_mb` on GPU 6 was 4 → 401 MiB (`iree_sr_a`), `free_mb_start` 48 409 MiB, peak torch allocation 2 630 144 B = 2.51 MiB (`iree_sr_b2`'s `gpu` block never got past the start snapshot — see the crash below) |
| lock | `/tmp/gpbench_critical.lock` — exactly one `flock -w 7200` per config (added by the runner, never nested); a non-blocking probe before each chunk showed the lock free, and neither log carries a lock wait |
| engine under test | `new` = `PYTHONPATH=src` (this worktree, no `src/**` file touched: the JSONs carry `evogp_commit=cbe4237f41…` for `iree_sr_a` and `e809c1fa46…` for `iree_sr_b2` — the worker's HEAD at launch, all the same engine tree), `etl 0.1.0`, backend `iree`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`, `EVOGP_ETL_IREE_DEVICE=0`, `ETL_DISABLE_LOCATIONS=1`, `EVOGP_ETL_CACHE_SIZE=8192` (both configs' own override; all recorded in each JSON's `env`) |
| FD/OS limits | `nofile_rlimit = [65536, 65536]` in both JSONs (the runner's `ulimit -n 65536` applied); `ulimit -c = 0` (no core dumps), `vm.max_map_count = 65530`, cgroup `memory.max` = 12 GiB — see the `iree_sr_b2` crash analysis |
| fixtures | not needed for the `sr` task (`iree_sr_a` / `iree_sr_b2` generate their own populations) → no GUARD-SKIP possible |

### Commands (exact; one bash call per config)

```bash
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2   # scan mode, before/after the discover() fix
bash test/benchmark/r2/run_all.sh --only iree_sr_a --timeout 3000 --dry-run   # printed command, nothing executed
bash test/benchmark/r2/run_all.sh --only iree_sr_a  --timeout 3000   # 15:16:17
bash test/benchmark/r2/run_all.sh --only iree_sr_b2 --timeout 3600   # 15:20:36
# runner-generated per-config commands (verbatim in each log's header):
#   iree_sr_a : timeout -s INT -k 30 3000 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_CACHE_SIZE=8192 \
#               …/python test/benchmark/bench_gp.py --task sr --pop 200 --gens 10 --tree-len 64 --reps 2 \
#               --backend iree --stage-probe --max-cold-builds 3200 --tag IREE_SR_A --json-out …/iree_sr_a.json
#   iree_sr_b2: timeout -s INT -k 30 3600 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_CACHE_SIZE=8192 \
#               …/python test/benchmark/bench_gp.py --task sr --pop 1000 --gens 2 --tree-len 128 --reps 2 \
#               --backend iree --stage-probe --max-cold-builds 1500 --tag IREE_SR_B2 --json-out …/iree_sr_b2.json
```

`--dry-run` was used first for `iree_sr_a` (printed the paths, the exact `flock … bash -c '…'` line and the
per-config `timeout`, nothing executed); `run_all.sh --only` selection order is irrelevant here (one config per
call). Each config is its own process, so the 8 192-entry `etl.CompileCache` starts empty (genuine cold first
generation) and cannot leak into the next config. **Budget discrepancy vs the hand-off brief:** the frozen
runner uses `--max-cold-builds 3200` for `iree_sr_a` and `1500` for `iree_sr_b2` (`run_all.sh` lines 246 / 253,
unchanged since `d5b0085`); the brief said "cap 1500" for both. Nothing was edited — `iree_sr_a` peaked at 517
misses, and a 3 200-miss cap is in any case unreachable at 32.3 fds/build (≈103 k fds ≫ 65 536).

### `iree_sr_a` — exit 0, `complete=true`, `partial=false` (1/2 configs measured)

| rep | `cold_first_gen_ms` | `per_gen_ms` (10 gens) | warm per-gen median = median(`per_gen_ms[1:]`) | rep total ms |
|---|---|---|---|---|
| rep0 | 82 098.93 | [82 098.93, 40 632.07, 42 112.79, 12 705.31, 10 733.32, 10 485.29, 12 071.67, 8 147.41, 8 076.74, 8 104.63] | **10 733.32** | 235 985.28 |
| rep1 | 324.95 | [324.95, 197.56, 167.27, 74.44, 65.31, 59.50, 72.10, 55.47, 54.45, 52.85] | **65.31** | 1 168.36 |

Top level: `cold_first_gen_ms` = 82 098.93 (rep0's gen 0), `best_fitness_final` = −0.2510969936847687,
`harness_wall_s` = 237.9, `generation_loop_s` = 235.17 (rep0) + 1.12 (rep1) = 236.3 s, `construct_s` = 0.78.
`iree_sr_a`'s rep-1 warm median (65.31 ms) is the backend's real per-generation cost (see the split below);
rep0's is compile-dominated.

Per-rep stage totals (`stages.reps[*].totals`):

| rep | `cold_builds` | `cold_build_ms` | `total_build_misses` | `warm_misses` | `warm_build_miss_ms` | `total_ms` | `build_calls` / hits / misses | `run_ms` | `readback_ms` | `upload_ms` | `trees_invoked` / `trees_total` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rep0 | 155 | 81 142.13 | 517 | **362** | 151 824.06 | 235 168.14 | 632 / 115 / 517 | 947.43 | 445.54 | 51.67 | trees_invoked 632 / 2 000 |
| rep1 | 0 | 0.0 | 0 | **0** | 0.0 | 1 123.90 | 632 / 632 / 0 | 490.28 | 308.53 | 30.24 | trees_invoked 632 / 2 000 |

Both consistency checks required of this config **held exactly**: rep1 `totals.warm_misses == 0` (the 8 192-entry
cache holds every one of the 517 distinct structures, and rep1 repeats the seed), and the 155 / 362 split is
byte-for-byte the `np_sr_a` mirror's.

Warm per-gen decomposition, rep1 (median over `stages.reps[1].records[i>0]`, ms per call; `build_miss = 0.000`,
34 `build_calls` = 34 hits per call):

| decode | build_hit | build_miss | run | readback | upload | loss | misc | total | dominant |
|---|---|---|---|---|---|---|---|---|---|
| 2.43 | 6.09 | 0.00 | 27.02 | 17.12 | 2.39 | 0.23 | 10.14 | **65.31** | run 41 % + readback 26 % |

Same medians for rep0's warm generations (call 0 excluded): decode 3.58, build_hit 1.30, **build_miss 10 631.82**,
run 52.98, readback 26.88, upload 3.54, loss 0.44, misc 16.10, total 10 733.32 — i.e. a "warm" generation in a
churning SR population is a **compile** generation (99 % `build_miss`), which is why the numpy/IREE comparison
must be made on rep1, where nothing new is compiled.

### FD telemetry (per config) and the stop threshold

| config | `fds.start` | `fds.after_cold` | `fds.end` | `cold_builds` | **implied fds / build** | note |
|---|---|---|---|---|---|---|
| `iree_sr_a` | 39 | 5 051 | 17 243 | 155 | **32.34** | below the 38 fds/build stop threshold → cleared to run `iree_sr_b2`; the extra 12 192 fds past `after_cold` are the remaining 362 misses (~33 fds each, the same rate) plus the warm path (~0.4 fds/build call) |
| `iree_sr_b2` | 39 | `null` | `null` | — (no generation completed) | **n/a** | `after_cold`/`end` are written only at completion; the crash left no FD telemetry |

32.34 fds/build is identical to the eval group's rate for the same 155-build fixture set, i.e. the cost is per
compiled structure, not per data row or per tree size class measured so far. Peak for this pair: 17 243 fds of
65 536 (26 %).

### Fitness-trajectory bit-identity vs the numpy mirror

* `iree_sr_a` rep0 **and** rep1: `per_gen_best_fitness = [-0.2510969936847687] × 10`; all 20 values are
  **bit-identical** (exact `==`, no tolerance) to `np_sr_a` rep0/rep1, and equal to the round-1 `C7` anchor —
  so IREE-O1 and numpy still agree exactly at this scale, on both a compile-heavy and a fully warm rep. The two
  backends also produce identical structure counts (155 cold + 362 misses, 632 build calls / 632 trees invoked
  of 2 000 tree-slots — the latter is a property of the SR harness workload, not of the backend).
* `iree_sr_b2`: **not verifiable** — the run died before generation 0, so the expected first two values
  `[-0.2510969936847687, -0.22635294497013092]` (present in both `np_sr_b` reps) are unverified on IREE.

### Compile-vs-warm economics (`iree_sr_a` vs its `np_sr_a` mirror)

| quantity | `iree_sr_a` (IREE O1) | `np_sr_a` (numpy) | ratio |
|---|---|---|---|
| cold first generation | 82 098.9 ms | 1 356.3 ms | **60.5×** |
| build misses (identical 517 structures) | 232 966.2 ms (155 cold 81 142.1 + 362 warm misses 151 824.1) | 1 998.7 ms | **116.6×** |
| per cold build | 523.5 ms | 5.4 ms | 97× |
| rep0 warm per-gen median | 10 733.3 ms (99.1 % = `build_miss`) | 84.09 ms | 128× |
| rep1 warm per-gen median (fully warm) | 65.31 ms | 22.07 ms | **2.96×** |
| rep0 / rep1 wall | 235.99 s / 1.17 s | 3.49 s / 0.39 s | 67.6× / 3.0× |
| rep0 generation loop | 235 168.1 ms, of which 232 966.2 ms = 99.06 % build misses | 2 750.6 ms | — |

So the training-pair cost is **entirely per-structure IREE compilation** (~83 s before generation 0 can even be
scored; 99 % of the rep), and once the cache is warm the backend is within ~3× of the numpy interpreter on the
same 10-generation SR run. Nothing here changes the round-1 direction: the IREE path is the *compilation* cost,
not the evaluation cost.

### `iree_sr_b2` — exit 139 (SIGSEGV) before generation 0, **not** a bounded outcome

Command (runner-generated, verbatim in the log header):

```bash
timeout -s INT -k 30 3600 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_CACHE_SIZE=8192 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/bench_gp.py --task sr --pop 1000 --gens 2 \
  --tree-len 128 --reps 2 --backend iree --stage-probe --max-cold-builds 1500 --tag IREE_SR_B2 \
  --json-out test/benchmark/new/results/r2/iree_sr_b2.json
# runner: config iree_sr_b2: FAIL (exit 139)
```

Facts:

* **Exit 139 = 128 + 11 = SIGSEGV**, i.e. neither of the two outcomes this config was expected to produce
  (124 / 137 killed by `timeout`, or 3 for a guard trip) and not the `exit 1` EMFILE signature either. The
  log's whole run-output block is 3 startup lines plus the runner's `exit code: 139 (FAIL)` — **no Python
  traceback, no error text, no stderr at all**: a silent native crash.
* Process life: launched 15:20:39, runner recorded `finished 15:27:09` → ≈ **390 s**; the guard object is still
  `{limit: 1500, misses_last: 0, misses_max: 0, exceeded: false}` and the stage probe's `records` are empty, so
  **no generation finished** and no build count is recorded.
* What the JSON preserves (skeleton only; `complete=false`, `partial=false`): `config` (pop 1000 / gens 2 /
  tree_len 128 / reps 2 / seed 0 / backend iree / O1 / cache 8192 / **max_cold_builds 1500**),
  `fds = {start: 39, after_cold: null, end: null}`, `stages.reps[0].totals = null`, `per_gen_ms = []`,
  `per_gen_best_fitness = []`, `nofile_rlimit = [65536, 65536]`, `construct_s = 0.734`, `gpu` = the start
  snapshot only. `fds.after_cold`/`fds.end` are written only on completion, so the crash left **no FD
  telemetry** — an intentional consequence of the atomic-write design, not a runner defect.
* **No autopsy is possible from this box**: `ulimit -c` is `0` in the environment the runner inherits
  (`core_pattern` is `core`, but no `core` file appeared in the cwd), and `dmesg` is not readable by this user
  (`Operation not permitted`).
* **Discrimination (why this is *not* reported as EMFILE):** round 1's C7 attempt-1 hit exactly that failure
  mode when the launcher's soft limit was 1024 and it surfaced as a clean
  `OSError: [Errno 24] Too many open files` at `etl/backends/adapters/iree.py:987 _compile_mlir →
  iree/compiler/tools/binaries.py:199 → subprocess.run → os.pipe()` (`../new/NOTES.md` §9.2), i.e. the IREE
  compile path **raises** on FD exhaustion. Here the limit was a confirmed `[65536, 65536]` and the failure was
  a silent SIGSEGV with no message — consistent with a native fault in the IREE compiler/runtime or the
  nanobind bindings rather than a clean EMFILE. This record cannot say which: there is no per-build progress
  logging, no core file, and `ms/build` at tree_len 128 is unmeasured (at the measured 450–524 ms/build, 390 s
  of runtime bounds the work at ≲ 900 builds — well below the 1 500-miss guard, so the guard never had a chance
  to trip).
* **Scale context for whoever picks this up** (not a defect of this run): the numpy analog `np_sr_b` needs
  4 760 (rep0) + 4 687 (rep1) = **9 447 build misses** at pop 1000 / tree 128; at the measured 32.34 fds per
  compiled structure that is ≈ **306 k fds ≫ 65 536**, so this workload cannot complete in one process at the
  frozen settings whatever the budget, and the only bounded outcome the runner could have produced is a guard
  trip at 1 500 misses (≈ 48.5 k fds). Related box limits observed: `vm.max_map_count = 65530` (the IREE path
  backs each compiled structure with a `MappedMemory`, per the nanobind leak lists) and the agent cgroup's
  `memory.max` = 12 GiB (an OOM there would show as 137, which this run did not).
* **Not retried, not patched** (protocol): no re-run of the config, no edit to the harness/runner/library, no
  change to the frozen `--max-cold-builds 1500` or to `CONFIG_ARGS`. The crash record is committed as-is
  (`iree_sr_b2.json` + `.log`) so the matrix id resolves to what actually happened.

### Validator (`check_r2.py`) — one bounded fix, then both modes

The directory scan picked up the 4 pre-flight records in
`test/benchmark/new/results/r2/backend_agreement/*.json` (backend-agreement pre-flight, not R2 config results)
as 4 schema failures. `discover()`'s `os.walk` filter now also skips a subdirectory named
`backend_agreement` (commit `cbe4237`, 2 lines + comment); `--manifest` mode is untouched. Effect on the scan:
`6 failure(s)` → `2 failure(s)` (the 2 pre-existing prior-round artifacts below).

```bash
# verification of the fix (before -> after): the 4 `fixture*_{iree,numpy}` failures disappear
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2   # exit 1 (2 failures, both pre-existing)

# the two commands recorded for this section, verbatim:
python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt          # exit 0
#   -> 35 record(s): 34 complete, 1 partial, 0 skipped, 0 missing, 0 malformed; 126 check(s), 0 failure(s)
#      note: iree_sr_b2: PARTIAL (complete=False, partial=False)
#            iree_sr_b2: stage-counter extraction incomplete (cold=?, misses=?); stages keys: probe, side, partial, note, ...
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2   # exit 1
#   -> 50 record(s): 46 complete, 3 partial, 1 skipped, 0 missing, 0 malformed; 123 check(s), 2 failure(s)
#      FAIL C5  (missing required eval keys ['task','config','complete'], legacy/results/r2/C5.json)
#      FAIL C5b (same, legacy/results/r2/C5b.json)
#      notes: iree_sr_b2 PARTIAL; iree_sr_b2 stage-counter extraction incomplete; h1_guard SKIP
#             (cold-builds-exceeded:4>1); C5/C5b PARTIAL (complete=None); leg_smoke tag 'LEG-smoke' (prior-round artifact)
```

Both results are the expected ones: **35/35 manifest ids now have a JSON on disk** (0 missing, 0 malformed,
0 failures) so the R2 matrix is complete as *records* — with the honest caveat that `iree_sr_b2` is a
crash-skeleton (`partial` in the report because `complete=false`), not a measurement — and the scan's only
failures are the **2 pre-existing prior-round driver artifacts** (`C5.json` / `C5b.json`, written by round 1's
`legacy/run_matrix.py`), the same 2 the light-group and eval-group entries reported. No failure is
attributable to `iree_sr_a`, to the validator fix, or to the 10 eval configs.

### Wall clock

| phase | wall |
|---|---|
| validator fix + verification + commit (`cbe4237`) | ≈ 1 min (15:15:5x → 15:16:1x) |
| chunk 1 (`iree_sr_a`, exit 0) | **4 min 1 s** (15:16:17 → 15:20:18; harness-reported `harness_wall_s` 237.9 s) |
| chunk 2 (`iree_sr_b2`, exit 139) | **6 min 34 s** (15:20:36 → 15:27:10; process life ≈ 390 s, i.e. the crash is not a lock wait and not a timeout) |
| **total for this operator's pair** | **≈ 13 min** (15:15 → 15:28, incl. validation + commits) |
| harness-reported generation-loop sum (`iree_sr_a`) | 236.3 s (rep0 235.17 s + rep1 1.12 s), of which 232.97 s = build misses (98.6 %) |

### Deviations / anomalies

* **`iree_sr_b2` exit 139 (the group's headline deviation)** — silent native SIGSEGV before generation 0, no
  usable per-gen data, not reproducible from the record (no core, no dmesg, no per-build logging); not retried
  and not patched. The 2 manifest ids are therefore 1 measured + 1 crash-skeleton.
* **Budget discrepancy vs the hand-off brief (documentation only).** The frozen runner uses
  `--max-cold-builds 3200` for `iree_sr_a` and `1500` for `iree_sr_b2`
  (`run_all.sh` lines 246 / 253, unchanged since commit `d5b0085`); the task brief for this chunk said
  "cold-build cap 1500" for both. Nothing was changed: `iree_sr_a` peaked at 517 misses (16 % of its cap), and
  a 3200-miss cap would in any case be unreachable at 32.3 fds/build (3200 × 32.3 ≈ 103 k ≫ 65 536). Post-run,
  commit `a0763d3` aligned `iree_sr_a`'s cap to 1500 in `run_all.sh` for future runs (the earlier same-file
  batch edit had raced and only landed on `iree_sr_b2`); the run recorded above used the 3200 cap as stated.
* **"~500–700 cold builds" expectation → measured 517 total build misses** in `iree_sr_a` rep0 = 155 cold
  builds (gen 0, `cold_builds`) + 362 cache misses later in the same rep — exactly the `np_sr_a` mirror's
  155 / 362 split, which is what makes the trajectories bit-identical. The objective's
  `rep1 totals.warm_misses == 0` check **held exactly** (rep1 = 0 cold builds / 0 misses / 632 hits).
* **`per_gen_best_fitness` bit-identity: held.** All 10 generations of *both* reps are exactly
  `-0.2510969936847687`, identical (exact `==`, no tolerance) to `np_sr_a` rep0 and rep1 and to the round-1
  C7 anchor. `iree_sr_b2` has none (crashed before gen 0), so its expected
  `[-0.2510969936847687, -0.22635294497013092]` (present in both `np_sr_b` reps) remains unverified on IREE.
* **FD telemetry: 32.34 fds per compiled structure** in `iree_sr_a` (`(5051 − 39) / 155`,
  matching the eval group's 32.34 for the same 155-build fixture set), i.e. **below the 38 fds/build stop
  threshold** → cleared to proceed to `iree_sr_b2`. Peak `fds.end` = 17 243 of 65 536 (26 %), and the extra
  12 192 fds beyond `after_cold` are the remaining 362 misses (~33 fds each) plus the warm path — the FD rate
  is the build, not the warm call.
* **A "warm" generation in a churning SR population is a compile generation**: `iree_sr_a` rep0's warm
  per-gen median is 10 733 ms, of which 10 632 ms is `build_miss` (362 misses, ~419 ms each); only rep1
  (nothing new to compile) shows the backend's real warm cost, 65.31 ms/gen
  (run 27.0 + readback 17.1 + build-hit 6.1 + misc 10.1 ms). Same shape as `iree_c7r` (`np_sr_a` predicts it).
* **Per-call cache accounting: `build_calls` 632 = `trees_invoked` 632 of `trees_total` 2 000** (63 % of
  tree-slots evaluated, 34 build calls per generation) — identical to the numpy mirror, i.e. a property of the
  SR harness workload, not of the IREE backend.
* **nanobind leak warnings at teardown** (`leaked 4390 instances` of `HalBuffer`/`MappedMemory`/`HalDevice`/
  `HalAllocator`, `leaked 4389 keep_alive records`, in `iree_sr_a`; the eval group saw the same) — pre-existing
  binding-teardown noise after the JSON is written; exit code 0 and no measured number affected. `iree_sr_b2`
  produced **no** such block, consistent with the crash being mid-run rather than at teardown.
* **No lock contention**: both chunks acquired `/tmp/gpbench_critical.lock` immediately (a non-blocking probe
  before each chunk showed it free) and no lock wait is visible in either log; no ECC event; GPU 7 never
  touched.
* `test/benchmark/r2/last_run.txt` was rewritten by each chunk (it now holds `iree_sr_b2` only) and is **not**
  committed. Files written by this operator: 4 result files (`iree_sr_a.json/.log`,
  `iree_sr_b2.json/.log`), the `check_r2.py` 2-line `discover()` fix, and this section.
* **R2 matrix status after this entry: 35/35 ids have records** (34 complete + `iree_sr_b2` crash-skeleton),
  0 manifest failures. The report (`../REPORT.md`) should treat `iree_sr_b2` as a bounded-failure datapoint
  ("IREE is not runnable at pop 1000 / tree 128 on this box at 8 192 cache / 65 536 fds"), **not** as a
  measured training pair.

**Commits this section covers:** `cbe4237` (validator `discover()` fix), `e809c1f` (`iree_sr_a` JSON+log),
`cc12894` (`iree_sr_b2` JSON+log as the crashed record).

---
## Round 3 (12 configs: legacy + numpy + IREE auto/spec) — 2026-09-12 21:59–22:22 HKT
**Operator:** the R3 round operators (matrix groups + companion re-run) and the agreement-pre-flight operator.
**Scope:** `test/benchmark/r2/round3/**` only — the 12 R3 configs (`round3/results/new/<id>.{json,log}`,
`round3/results/legacy/<id>.{json,log}`), the six agreement arms (`round3/results/backend_agreement/`) and
this section; no R2 record, no file above `round3/`, no harness/runner/validator/`src/**` file was touched.
**Method and contract:** `round3/README.md` (scope table, flags, gates, stage-probe caveat) and
`round3/manifest.txt`; runners `round3/run_all.sh` (matrix) and `round3/run_agreement_r3.sh` (pre-flight).
Raw evidence stays in the per-config JSONs + logs — a log is append-only, so a re-run adds a dated block, while
the JSON is the harness's incremental snapshot and holds the latest window. This section is the operator record;
the verdict and the R2-vs-R3 tables are in `round3/SUMMARY.md` and are not duplicated here.
### Environment
| item | value |
|---|---|
| repo root (cwd for every command) | repo root of each operator's worktree (every JSON carries its own `cwd`) |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |
| GPU | `CUDA_VISIBLE_DEVICES=6` — the runner default, used by the agreement pre-flight and by both first-window matrix groups; the companion re-run used `CUDA_VISIBLE_DEVICES=0`. Recorded per JSON (`gpu.cuda_visible_devices`: `6` for the 6 IREE configs, `0` for the 6 light-group ones). GPU 7 (hardware ECC fault) never touched |
| foreign tenant / host load | GPU 6 carried ≈22.25–22.55 GB of a foreign tenant throughout the first window (`syfang` video-attention training, pid 2246336): every W1 log's `gpu (before/after)` line reads `6, NVIDIA RTX A6000, 0–100 %, 22 2xx MiB`. Host load average ≈80–93 during W1 and ≈85–103 during the companion window |
| lock / limits | `/tmp/gpbench_critical.lock` — exactly one `flock -w 7200` per invocation (added by the runner, never nested), `ulimit -n 65536` first, `timeout -s INT -k 30 3600` inside the lock, one `bench_gp.py` process per config; no lock-wait message and no `timeout` kill in any of the 12 configs / 6 arms |
| engine under test | `new` = `PYTHONPATH=src` (the R3 worktrees; the 12 matrix JSONs record `evogp_commit=49c5197` for the six iree/iree-spec ids and `a848fc2` for the six light-group ones — the agreement records carry no commit field, only backend/opt-level/device), `etl 0.1.0`, backends `numpy` (default) and `iree` (`EVOGP_ETL_IREE_OPT_LEVEL=O1`, `EVOGP_ETL_IREE_DEVICE=0`), plus `EVOGP_EVAL_MODE=specialized` for the three `*_spec` ids; `leg` = the legacy CUDA checkout (`PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`, `evogp_file=/tmp/evogp_legacy_bench/src/evogp/__init__.py`) |
| fixtures | `test/goldens/forest_random.npz` (64 trees, sha256 `0bfb0bbf…9ca0`) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, sha256 `6789ef61…41b9`) present → no GUARD-SKIP |
| FD limit | `nofile_rlimit = [65536, 65536]` in every matrix JSON (the runner's `ulimit -n 65536`) |
### Commands (exact)
```bash
# gate 1 / pre-flight: six arms (numpy / iree auto / iree specialized × 64-tree and 200-tree fixture)
bash test/benchmark/r2/round3/run_agreement_r3.sh
# matrix, one bench_gp.py process per config, group by group
bash test/benchmark/r2/round3/run_all.sh --group light
bash test/benchmark/r2/round3/run_all.sh --group iree
bash test/benchmark/r2/round3/run_all.sh --group iree-spec
# companion re-run of the light group on the tenant-free GPU 0 (GPU 6 still occupied)
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all.sh --only leg_c5r2
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all.sh --only leg_fixture200_n64,leg_sr_a,np_c5r2,np_fixture200_n64,np_sr_a
# gates + the comparison tool (no GPU)
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages
python3 test/benchmark/r2/round3/check_agreement_r3.py
python3 test/benchmark/r2/round3/compare_rounds.py
```
### Per-config status (12/12 exit 0, `complete=true`, `partial=false`, no `skipped`, no `skip_reason`)
`first_ms` / `warm_ms` follow the conventions at the top of this file; for `sr` ids `warm_ms` is the per-rep
median of `per_gen_ms[1:]` as `rep0 / rep1`. The committed JSONs hold the latest window: the companion one for
the six light-group ids (`0 (W2)`), the first one for the six IREE ids (`6 (W1)`).
| id | side | group | window | first_ms | warm_ms |
|---|---|---|---|---|---|
| `leg_c5r2` | leg | light | 0 (W2) | 26.49 | 0.6323 |
| `leg_fixture200_n64` | leg | light | 0 (W2) | 23.90 | 0.6357 |
| `leg_sr_a` | leg | light | 0 (W2) | 595.29 | 1.8007 / 1.5847 |
| `np_c5r2` | new | light | 0 (W2) | 207.47 | 23.2933 |
| `np_fixture200_n64` | new | light | 0 (W2) | 1371.43 | 117.6118 |
| `np_sr_a` | new | light | 0 (W2) | 1641.86 | 117.3345 / 30.7815 |
| `iree_fixture64_n64` | new | iree (auto) | 6 (W1) | 13640.33 | 10.6442 |
| `iree_fixture200_n64` | new | iree (auto) | 6 (W1) | 13207.38 | 10.4976 |
| `iree_sr_a` | new | iree (auto) | 6 (W1) | 16551.84 | 138.2627 / 38.3892 |
| `iree_fixture64_n64_spec` | new | iree-spec | 6 (W1) | 31485.90 | 67.8688 |
| `iree_fixture200_n64_spec` | new | iree-spec | 6 (W1) | 137474.89 | 228.5165 |
| `iree_sr_a_spec` | new | iree-spec | 6 (W1) | 142445.25 | 14741.3184 / 64.6597 |
### Gate outputs (verbatim key lines)
```bash
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages   # exit 0
#   -> 12 record(s): 12 complete, 0 partial, 0 skipped, 0 missing, 0 malformed (0 not in the manifest)
#      45 check(s), 0 failure(s)
python3 test/benchmark/r2/round3/check_agreement_r3.py                                                           # exit 0
#   -> gate: PASS  (32 check(s), 0 failure(s))
#      fixture64 : numpy vs iree PASS (max|diff|=7.62939e-06, max rel 1.08241e-07); numpy vs iree_spec PASS (bit-identical); nan=3 inf=0
#      fixture200: numpy vs iree PASS (max|diff|=0.00390625, max rel 2.02762e-07); numpy vs iree_spec PASS (bit-identical); nan=12 inf=0
#      continuity vs the R2 anchors is INFO only: the dense-default `iree` arm reproduces the R2 anchor bitwise, the specialized route differs by <=2.03e-07 rel
```
`results/backend_agreement/run_agreement_r3.last_run.txt` adds the pre-flight's own arm→exit-code table
(`ok=6 skipped=0 failed=0`, invocation 2026-09-12T22:12:14+08:00).
### Light group across the three windows (the round's only re-measured ids)
The R2 column is the untouched R2 record (`../legacy/results/r2/<id>.json` / `../new/results/r2/<id>.json`),
W1 is the first window (JSON snapshots in commit `a848fc2`), W2 is the companion window (the values in the
committed JSONs); `warm_ms` per the convention above.
| id | R2 | W1 (GPU 6) | W2 (GPU 0) |
|---|---|---|---|
| `leg_c5r2` | 0.6327 | 1.9300 | **0.6323** |
| `leg_fixture200_n64` | 0.6319 | 1.9667 | **0.6357** |
| `leg_sr_a` | 1.6166 / 1.5883 | 1.679 / 1.726 | 1.8007 / 1.5847 |
| `np_c5r2` | 21.0563 | 20.5844 | 23.2933 |
| `np_fixture200_n64` | 93.6742 | 95.0055 | 117.6118 |
| `np_sr_a` | 84.0937 / 22.0696 | 142.442 / 42.753 | 117.3345 / 30.7815 |
### Wall clock
| phase | wall |
|---|---|
| agreement pre-flight (six arms) | 21:59:46 → 22:12:13 |
| `--group iree` (3 configs, alternating with the spec group on the shared lock) | 21:59:56 → 22:10:34 |
| `--group iree-spec` (3 configs) | 21:59:20 → 22:08:31 |
| `--group light`, first window | 22:03:18 → 22:12:27 |
| companion re-run, `--only leg_c5r2` | 22:20:53 → 22:20:55 |
| companion re-run, the other five ids | 22:21:04 → 22:21:32 |
### Deviations / anomalies
* **W1 contention inflated the launch-bound ids ≈3×** — `leg_c5r2` 1.93 ms vs R2 0.6327, `leg_fixture200_n64`
  1.9667 vs 0.6319 — and `np_sr_a` by ≈1.7–1.9× (142.442 / 42.753 vs R2 84.0937 / 22.0696), with GPU 6 holding
  the ≈22 GB foreign tenant and the host load average at ≈80–93 for the whole window.
* **The W2 companion re-run recovered the legacy anchor exactly**: same tree, flags and seed on the tenant-free
  GPU 0 give 0.6323 / 0.6357 ms vs R2 0.6327 / 0.6319 (≤ 0.6 %). The numpy ids kept a residual shared-host noise
  band (`np_sr_a` 117.33 / 30.78 = 1.40× R2, `np_c5r2` 23.29 vs 21.06, `np_fixture200_n64` 117.61 vs 93.67), and
  the fresh `np_c5r2` / `np_fixture200_n64` values are slightly worse than their own W1 ones (20.58 / 95.01).
  Both windows stay in the append-only logs; the W1 JSON snapshots only in commit `a848fc2`.
* **The stage probe is degenerate on the dense route** (documented limitation in `round3/README.md`): the
  auto-route eval arms report `build_miss 0.0 ms` and a `misc`-dominated total (e.g. `iree_fixture64_n64` rep0:
  `run 97.3` + `misc 14400.7` ms), because `stage_probe.py` instruments the specialized call sites only. The
  numpy light-group arms stay fully instrumented (`np_c5r2` 48 cold builds / 796.8 ms total;
  `np_fixture200_n64` 155 / 4212.3; `np_sr_a` 155 cold + 362 warm misses / 3368.9). No gate reads the block.
* **`iree_sr_a` took a mixed route**: only gens 1–2 of each rep registered instrumented specialized calls (127
  and 102 build calls — 33184 / 34462 ms in rep0, 190.6 / 265.6 ms in rep1), every other generation shows
  `calls=0` (dense route, uninstrumented); `iree_sr_a_spec` is instrumented for all 10 generations of both reps
  (155 cold + 362 warm misses).
* **FD telemetry:** `fds.end` = 6491 / 6491 / 9723 on the three auto-route ids vs 2395 / 5819 / 17243 on the
  three `_spec` ids (the R2 per-structure rate of ≈32–33 fds per compiled structure, `after_cold` 347 vs
  1627 / 5051 / 5051) — all far below the 65536 limit, none near it.
* **No crash, no partial, no timeout:** 12/12 configs and 6/6 arms exit 0 with `complete=true`; the R2
  `iree_sr_b2` SIGSEGV failure mode (pop 1000 / tree 128) was out of R3 scope and did not occur.
* **nanobind teardown noise** (`nanobind: leaked …`, e.g. 877 instances / 876 keep_alive records / 5 types in
  `iree_fixture64_n64`) appears in every IREE log after the JSON is written — pre-existing binding-teardown
  noise, exit code 0, no measured number affected.
* **Workload identity with R2 holds at the counters**: `np_c5r2` 48 cold builds, `np_fixture200_n64` 155,
  `np_sr_a` 155 cold + 362 warm misses, `trees_invoked` 1325/1600, 4275/5000 and 632/2000 — all identical to the
  R2 mirrors — and the SR trajectories are unchanged (new side `-0.2510969936847687` on every generation, legacy
  `-0.2257394939661026` final, in both windows).
* `round3/last_run.txt` (the runner's id→exit-code table) is generated and `.gitignore`d; only the pre-flight's
  own `run_agreement_r3.last_run.txt` is committed as evidence.
**Analysis of record:** `round3/SUMMARY.md` — the verdict, the R2-vs-R3 tables from `compare_rounds.py`, both
gates and the round's caveats; the R2 baseline they are read against stays `../REPORT.md`.
**Commits this section covers:** `49c5197` (R3 scaffold), `988524c` (six agreement arms), `123191d` (iree-spec
records), `9e1c8af` (auto-iree records), `c1b9bbc` (light-group W1 records), `a848fc2` (merge carrying the W1
snapshots), `d38ada7` (W2 companion re-run).

---
## Round 3 — Arm B (9 configs: numpy + IREE auto/spec on the PATCHED ETL checkout) — 2026-09-12 22:49 → 2026-09-13 00:08 HKT (W1 + b2 on GPU 6; W2 on GPU 0)
**Operator:** the Arm B round operators (tooling + agreement pre-flight + four matrix windows + the
`iree_sr_b2` re-attempt + the W2 companion window).
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
# gate 1 / pre-flight: six arms (numpy / iree auto / iree specialized × 64-tree and 200-tree fixture)
bash test/benchmark/r2/round3/run_agreement_armB.sh              # 23:01:00 → 23:04:05 (window 1b)
# matrix, one foreground `run_all_armB.sh --only <id>` invocation per config, in run order (GPU 6)
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64_spec    # 23:09:12 → 23:09:37
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64_spec   # 23:10:01 → 23:11:31
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a_spec             # 23:11:40 → 23:15:32
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64         # 23:19:02 → 23:19:24
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64        # 23:19:35 → 23:19:51
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a                  # 23:20:02 → 23:21:29
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_c5r2                    # 23:26:48 → 23:26:53
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_fixture200_n64          # 23:27:12 → 23:27:20
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_sr_a                    # 23:27:28 → 23:27:40
# `iree_sr_b2` re-attempt (NOT a runner row): one direct bench_gp.py invocation per attempt, GPU 6,
#   23:32:21 → 23:51:17 — exact command lines in the `iree_sr_b2` re-attempt subsection below
# W2 companion re-run of all nine ids on the tenant-free GPU 0 (GPU 6 still occupied), 23:56:50 → 00:07:12 —
#   the same nine `--only` invocations prefixed `CUDA_VISIBLE_DEVICES=0` (exact per-config times in the W2 subsection)
# gates + the comparison tool (no GPU)
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
python3 test/benchmark/r2/round3/check_agreement_armB.py
python3 test/benchmark/r2/round3/compare_rounds_armB.py --markdown
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
#### Group `numpy` — EXECUTED 2026-09-12 23:26:48–23:27:40 HKT (window 4, GPU 6): 3/3 exit 0
This is the default-`numpy` group — the R3 reference route with **no** `EVOGP_ETL_BACKEND` /
`EVOGP_EVAL_MODE` / `EVOGP_ETL_CACHE_SIZE` overrides at all (each JSON's `env` block carries only
`ETL_DISABLE_LOCATIONS=1` and `CUDA_VISIBLE_DEVICES=6`), i.e. exactly the Arm A `np_*` wiring. On
`numpy` the engine keeps the **specialized** route, so unlike the dense `iree` eval arms in the two
subsections above all three records here carry *meaningful* `stages` blocks — the R2 probe wraps
precisely the call sites this route uses. With these three records **all nine Arm B ids are on disk**,
so the "`numpy` group is still unrun" notes in the `iree-spec`/`iree` subsections are superseded by
this one (their own validator outputs stay quoted there as the mid-flight snapshots they were).
**Commands (exact) — one foreground `run_all_armB.sh --only <id>` invocation per config, in run
order, each console captured to `results/armB/status_<id>.txt`; the runner owns
`ulimit`/`flock`/`timeout`:**
```bash
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_c5r2             # 23:26:48 → 23:26:53
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_fixture200_n64  # 23:27:12 → 23:27:20
bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_sr_a            # 23:27:28 → 23:27:40
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
```
No `flock` wait was observable (each config's `started` timestamp is within a second of its
invocation, and every invocation ran in the foreground of one long-budget call — neither the
detached-`nohup` reaping of the agreement window nor a mid-config kill occurred). Runner-level group
wall: **52 s** (23:26:48 → 23:27:40), of which 5 s + 8 s + 12 s is inside the three configs; nothing
approached the 3600 s cap.
**Per-config status table** (per-config exit code = the log's `exit code: 0 (OK)` footer *and* the
runner's own id→exit table in the same `status_<id>.txt`; metric rules as at the top of this file —
eval: `first_call_ms` and `median_warm_ms` over 25 calls; sr: rep0's `cold_first_gen_ms` as the first
figure and the per-rep median of `per_gen_ms[1:]`):
| id | exit | complete / partial | first_ms | warm_ms (rep0 / rep1) | cold builds (json `misses_max` / stage) | fds start→cold→end | wall (started→finished) | harness_wall_s | notes |
|---|---|---|---|---|---|---|---|---|---|
| armB_np_c5r2 | 0 | `complete` / `partial=false` | 191.31 | 20.78 (25 calls) | null (`enforced=false`, no budget) / stage 48 cold + 0 warm misses | 39 → 59 → 59 | 5 s (23:26:48 → 23:26:53) | 1.8 | GPU 6 foreign util 29 % before → 0 % after (≈30 GB `syfang` resident); stage block meaningful |
| armB_np_fixture200_n64 | 0 | `complete` / `partial=false` | 995.50 | 135.94 (25 calls) | null (`enforced=false`) / stage 155 cold + 0 warm misses | 39 → 59 → 59 | 8 s (23:27:12 → 23:27:20) | 4.7 | GPU 6 foreign util 25 % before → 4 % after; stage block meaningful |
| armB_np_sr_a | 0 | `complete` / `partial=false` | 2 255.92 (rep0 cold gen) | 154.45 / 33.83 (rep0 / rep1, median of `per_gen_ms[1:]`) | null (`enforced=false`) / stage rep0 = 155 cold + 362 warm misses, rep1 = 0 | 39 → 59 → 59 | 12 s (23:27:28 → 23:27:40) | 7.4 | GPU 6 foreign util 38 % before → 0 % after; `best_fitness_final=-0.2510969936847687`; per-rep repair: rep0 cold 2 255.92 → rep1 cold 229.16 ms |
`stages` quality (all three are **specialized**-route runs, so the probe is meaningful by
construction): every record has `stages.partial=false`, `stages.note=""`, `installed=true`, the full
`wrapped=[decode_population,_run_tree,build_cached,run,from_numpy,evaluate,sr_fitness]` list, 25 eval
records (eval ids) / 10 + 10 train records (the `sr` id) and populated per-stage counters
(`decode / build-hit / build-miss / run / readback / upload / loss / misc`, `h2d`/`d2h`) — no
degenerate dense-route block, and the build counts are *identical* to the Arm A twins (48 / 155 /
155 + 362), which is the same-workload cross-check the `sr` fitness trajectory gives. Stage split
(console `stages[...]` lines, cold/warm totals):
| id (rep) | cold misses (ms) | warm misses (ms) | decode + build-hit (calls) + build-miss (calls) + run (calls) + readback (calls) + loss + misc | h2d / d2h |
|---|---|---|---|---|
| c5r2 (rep0/eval) | 48 (156.3) | 0 (0.0) | 27.7 + 223.1 (1 277) + 156.3 (48) + 234.5 (1 325) + 16.3 (1 325) + 641.8 + 19.9 | 0 / 0 |
| fixture200_n64 (rep0/eval) | 155 (901.4) | 0 (0.0) | 158.1 + 1 225.3 (4 120) + 901.4 (155) + 1 666.5 (4 275) + 73.5 (4 275) + 3 917.0 + 34.6 | 0 / 0 |
| sr_a (rep0/train) | 155 (1 292.1) | 362 (2 308.7) | 40.6 + 26.7 (115) + 3 600.9 (517) + 266.6 (632) + 22.9 (632) + 3 951.0 + 956.2 | 0 / 0 |
| sr_a (rep1/train) | 0 (0.0) | 0 (0.0) | 32.0 + 146.8 (632) + 0.0 (0) + 182.4 (632) + 10.8 (632) + 361.9 + 201.3 | 0 / 0 |
(`trees_invoked/trees_total` follows the same pattern as Arm A: 1 325/1 600 and 4 275/5 000 on the eval
arms — i.e. every non-duplicate structure is built — and 632/2 000 per `sr` rep; `warm_misses=0` on
rep1 with all 632 build calls hitting the cache.)
**Arm A twin comparison (context only — the round's verdict tables are built by the write-up phase).**
The two numpy groups are **not device-paired**: Arm A's `np_*` records were the companion light-group
window on **GPU 0** (`env.CUDA_VISIBLE_DEVICES=0`, per that phase's note) while Arm B's numpy group ran
on the runner-default **GPU 6**; both are CPU-interpreter runs on a shared box, so read the deltas as
window-paired:
| id | Arm B first_ms | Arm A first_ms | Δ | Arm B warm_ms | Arm A warm_ms | Δ |
|---|---|---|---|---|---|---|
| np_c5r2 | 191.31 | 207.47 | −7.8 % | 20.78 | 23.29 | −10.8 % |
| np_fixture200_n64 | 995.50 | 1 371.43 | −27.4 % | 135.94 | 117.61 | +15.6 % |
| np_sr_a (rep0 cold gen / rep1 warm med.) | 2 255.92 (cold gen) | 1 641.86 (cold gen) | +37.4 % | 154.45 / 33.83 | 117.33 / 30.78 | +31.6 % / +9.9 % |
`src/evogp` is byte-identical between the Arm A record commit (`49c5197`) and the Arm B one
(`git diff --stat 49c5197 HEAD -- src/` is empty, as for the two IREE groups), so every delta above
belongs to the ETL checkout shadow and/or the window. The `sr` id's 10-generation trajectory is
**identical to Arm A bitwise in best fitness** (`per_gen_best_fitness` equal element-for-element,
`best_fitness_final=-0.2510969936847687` on both sides), and the patched checkout never changes the
build traffic (`48 / 155 / 155+362` misses on both sides). The `sr` rep0/rep1 gap (+31.6 % / +9.9 %) is
window noise on a CPU-bound route (host loadavg ≈81–93 at window start, rep0's `per_gen_ms` is itself
bimodal at ≈79–724 ms), not added work.
**Validator (gate (ii), all 9 of 9 ids on disk — the round's Arm B record gate, FINAL):**
```text
9 record(s): 9 complete, 0 partial, 0 skipped, 0 missing, 0 malformed (0 not in the manifest)
34 check(s), 0 failure(s)
armB_np_c5r2                   new   eval64  complete     191.31      20.78  48 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_np_fixture200_n64         new   eval64  complete     995.50     135.94  155 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_np_sr_a                   new   sr      complete    2255.92     154.45  155 (reps[0].totals.cold_builds)  362 (reps[0].totals.warm_misses)
CHECKER_EXIT=0
```
Full stdout kept as `results/armB/status_check_r3_armB_final.txt` (checker exit **0** — the first Arm B
validator run with nothing to report, same semantics as Arm A's `12 complete / 45 checks / 0
failures`: every manifest id present, at its manifest path, with the expected `side`, and a non-empty
`stages` block for each of the nine `--stage-probe` rows). The file is the raw stdout plus this phase's
append of the capture command's `CHECKER_EXIT=0` marker line (the earlier `status_check_r3_*` files hold
stdout only); the marker is what makes the exit code checkable from the file itself. On this group the checker's `cold_builds`
column is the *only* cold-build number the records have, because the numpy configs carry no
`--max-cold-builds` budget (JSON `cold_builds: {enforced: false, misses_max: null}` — same as Arm A's
`np_*` records), which is exactly why its stage-sourced `48 / 155 / 155` is meaningful here while the
dense `iree` eval arms showed `0` in the same column against a JSON-level `1`. The generated
`results/armB/last_run.txt` holds only the **last** invocation's table (config 3:
`armB_np_sr_a new 0 OK …`, `# ok=1 skipped=0 failed=0`); it is `.gitignore`d and not committed, which is
why each invocation's table is archived in its `status_<id>.txt` instead.

### `iree_sr_b2` re-attempt (pop 1000 / gens 2 / tree 128 — not an Arm B matrix id)
**EXECUTED 2026-09-12 23:32:21–23:51:17 HKT (window 5, GPU 6): BOTH attempts exit 0 — R2's SIGSEGV does
not reproduce.** This is the one R2 config that FAILED (silent native SIGSEGV, runner exit 139, after
≈800–900 builds and before generation 0 completed — R2's bounded-failure datapoint
`../new/results/r2/iree_sr_b2.json`, kept unchanged as a crash skeleton). The re-attempt repeats the R2
b2 row on the patched ETL checkout with Arm A's flags unchanged (`--task sr --pop 1000 --gens 2
--tree-len 128 --reps 2 --backend iree --stage-probe --max-cold-builds 1500 --mem-guard-mb 8192` +
`EVOGP_ETL_CACHE_SIZE=8192`, `ulimit -n 65536`, FD telemetry watched) in the two defined attempts:
* **attempt 1 = `armB_iree_sr_b2`** — the current engine defaults: `EVOGP_EVAL_MODE` unset ⇒ `auto`, and
  on the `iree` backend at pop 1000 / 1 000 data rows the `auto` policy selects the **dense** evaluator
  (`pop ≥ DEFAULT_AUTO_MIN_POP=16`, non-numpy backend, rows ≤ `DEFAULT_AUTO_MAX_ROWS=2048`).
* **attempt 2 = `armB_iree_sr_b2_spec`** — the R2-era route pinned (`EVOGP_EVAL_MODE=specialized`; in R2's
  era `auto` *was* the specialized engine, the dense evaluator is a post-R2 wave), i.e. the closest
  possible analogue of the crashed run: same route and same workload, with the patched ETL checkout as
  the only changed factor.
**Commands (exact) — one DIRECT `bench_gp.py` invocation per attempt (NOT via `run_all_armB.sh`: b2 is not
a matrix row and must not enter `manifest_armB.txt`), each in the foreground of one long-budget call with
`ulimit -n 65536` raised first and exactly one `flock -w 7200 /tmp/gpbench_critical.lock` whose `timeout`
sits INSIDE it; every command line below is the `cmd:` header line of the corresponding log:**
```bash
# attempt 1  (23:32:23 → 23:36:00, exit 0)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'timeout -s INT -k 30 3600 env PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_CACHE_SIZE=8192 /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/bench_gp.py --task sr --pop 1000 --gens 2 --tree-len 128 --reps 2 --backend iree --stage-probe --max-cold-builds 1500 --mem-guard-mb 8192 --tag ARMB_IREE_SR_B2 --json-out test/benchmark/r2/round3/results/armB/new/armB_iree_sr_b2.json'
# attempt 2, first execution  (23:36:14 → 23:40:40, exit 0 — WRONG ROUTE, see the window-5 anomaly
# bullet; the run actually taken was the dense one again and it is superseded by the re-run below)
# attempt 2, corrective re-run  (23:41:21 → 23:51:17, exit 0)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'timeout -s INT -k 30 3600 env PYTHONPATH=src:/tmp/evogp_etl_patched_51a7141 CUDA_VISIBLE_DEVICES=6 EVOGP_EVAL_MODE=specialized EVOGP_ETL_CACHE_SIZE=8192 /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/bench_gp.py --task sr --pop 1000 --gens 2 --tree-len 128 --reps 2 --backend iree --stage-probe --max-cold-builds 1500 --mem-guard-mb 8192 --tag ARMB_IREE_SR_B2_SPEC --json-out test/benchmark/r2/round3/results/armB/new/armB_iree_sr_b2_spec.json'
```
**Per-attempt outcome table** (exit code = the log's `exit code:` footer; `misses_max` = the JSON-level
cumulative ETL compile-miss counter the `--max-cold-builds` guard samples; `fds` = the JSON's
`start → after_cold → end`; wall = the log's `started`/`finished` pair):
| attempt (id) | route actually taken | exit | state | wall | `harness_wall_s` | builds (`misses_max` / limit) | rep0 stage (cold + warm misses, build calls) | rep1 stage | fds start→cold→end | JSON |
|---|---|---|---|---|---|---|---|---|---|---|
| attempt 1 — `armB_iree_sr_b2` | **dense** (`auto`) | **0** | `complete`, `partial=false` | 217 s (23:32:23 → 23:36:00) | 213.05 | 431 / 1500 | 0 + 430 (685 calls, 255 hits) | 0 + 0 (685 hits) | 39 → 91 → 91 | complete record, 2 reps × 2 gens, `best_fitness_final=-0.22635293006896973` |
| attempt 2, first execution (superseded) | **dense** (`auto` — env assignment lost) | **0** | `complete`, `partial=false` | 266 s (23:36:14 → 23:40:40) | 263.70 | 431 / 1500 | 0 + 430 (685 calls, 255 hits) | 0 + 0 (685 hits) | 39 → 91 → 91 | complete record, preserved verbatim as `new/armB_iree_sr_b2_spec_botched_dense_route.{json,log}` |
| attempt 2, corrective re-run — `armB_iree_sr_b2_spec` | **specialized** (pinned) | **0** | `complete`, `partial=false` | 597 s (23:41:21 → 23:51:17) | 594.84 | 1181 / 1500 | 788 + 393 (1568 calls, 387 hits) | 0 + 0 (1568 hits) | 39 → 91 → 91 | complete record, 2 reps × 2 gens, `best_fitness_final=-0.22635293006896973` |
| R2 twin `../new/results/r2/iree_sr_b2` (stock ETL, pre-dense era) | specialized (only route then) | **139** (SIGSEGV) | skeleton, `complete=false` | ≈390 s process life (15:20:39 → 15:27:09) | — | 0 recorded (no generation finished; ≲900 builds by the ms/build bound) | `records: []` | — | 39 → `null` → `null` | crash skeleton, unchanged (read-only reference) |
**Route attribution is read off the records, not assumed** — the `env` block plus the stage
fingerprint (the dense arms report the first-generation compile as `misc` and never as `build_miss`, so
their `rep0` shows `cold=0 + warm_M`; the same workload specialized shows `cold=788 + warm=393` with
1568 instead of 685 build calls):

| record | `env.EVOGP_EVAL_MODE` | rep0 build calls | rep0 cold + warm misses | route |
|---|---|---|---|---|
| `armB_iree_sr_b2` | *(absent ⇒ `auto`)* | 685 | 0 + 430 | dense |
| `armB_iree_sr_b2_spec_botched_dense_route` | *(absent ⇒ `auto`)* | 685 | 0 + 430 | dense |
| `armB_iree_sr_b2_spec` | `specialized` | 1568 | 788 + 393 | specialized |
| the pop-200 twins (for the same fingerprint) | `auto` / `specialized` | 229 / 632 | 0 + 169 / 155 + 362 | dense / specialized |
**FD telemetry — flat, exactly like every other Arm B `iree` run** (Arm B vs R2, same counter fields):
| record | engine | builds (`misses_max`) | fds start→cold→end | implied fds / build |
|---|---|---|---|---|
| R2 `iree_sr_a` | stock | 517 | 39 → 5 051 → 17 243 | **32.3** |
| R2 `iree_sr_b2` | stock | — (crash, no gen) | 39 → `null` → `null` | n/a |
| Arm B `armB_iree_fixture64_n64` (dense eval) | patched | 1 | 39 → 91 → 91 | ~0 |
| Arm B `armB_iree_sr_a` (dense) | patched | 170 | 39 → 91 → 91 | ~0 |
| Arm B `armB_iree_sr_a_spec` (specialized) | patched | 517 | 39 → 91 → 91 | ~0 |
| Arm B `armB_iree_sr_b2` (attempt 1, dense) | patched | 431 | 39 → 91 → 91 | ~0 |
| Arm B `armB_iree_sr_b2_spec` (attempt 2, specialized) | patched | **1 181** | 39 → 91 → 91 | **~0** |
i.e. at 2.3× the build count of the largest completed Arm B arm (517 → 1 181) the fd count stays at the
pre-flight level of 91 of 65 536; the 52 fds over `start` are one-time IREE runtime init, not per
compiled structure (the same value appears on the 1-miss eval arm). `nofile_rlimit = [65536, 65536]` in
all three b2 records; at the stock 32.3 fds/build the corrective run's 1 181 builds would have projected
to ≈38 k fds and R2's own hand-off note put the config's full 2 200-build budget at ≈73 k fds — above
the limit.
**Interpretation vs R2's SIGSEGV(139) after ≈800–900 builds.** The crash did not reproduce in either
route: both attempts completed all 2 reps × 2 generations with `exit 0`, `partial=false`, the guard
never tripped (431/1500 dense, 1 181/1500 specialized — 79 % of the cap, so the guard did not have the
last word), no timeout (217 s / 597 s against the 3 600 s cap), no interrupt, and FD telemetry flat at
91/65 536. On the three changed factors:
* **FD fix (the prime suspect) — consistent with the outcome and directly visible.** R2's twin
  `iree_sr_a` leaked 32.3 fds per compiled structure (517 builds → 17 243 fds); the patched checkout's
  per-thread HAL-device cache leaves the count flat (517 builds → 91 fds, and here 1 181 builds → 91 fds).
  A plain EMFILE was *not* the R2 crash mechanism at ≈800–900 builds (that projects to ≈26–29 k fds, below
  the 65 536 limit — R2's runlog reached the same conclusion and noted the EMFILE path raises cleanly
  anyway), but the leaked per-structure object graph (`HalBuffer` / `MappedMemory` / `HalAllocator` /
  `HalDevice`, still reported by the nanobind teardown in these logs) also drives `vm.max_map_count`
  (65 530 on this box) and native memory — and that growth is what the patch removes. This is the only
  changed factor in attempt 2 (same route, same workload, same flags).
* **Route — not a precondition for survival, but it changes the size of the run.** Attempt 1's dense
  route reaches the same two generations with 431 compile misses / 685 build calls per rep vs attempt 2's
  1 181 / 1 568 — 2.7× fewer compiles — so a route-only explanation would predict attempt 2 to be the
  risky one; it also completed (597 s, 1 181 misses). R2's own "~800–900 builds" crash point lands at the
  *end of generation 0*, i.e. exactly where the specialized route finishes its 788 cold builds (883 trees
  invoked, 349.7 s) and starts reusing the cache — the crash position is now reproducible as a position
  in the run, and the patched checkout passes through it.
* **Budget — unchanged and never reached.** `--max-cold-builds 1500` and `EVOGP_ETL_CACHE_SIZE=8192` are
  the R2 values; both attempts stayed under the cap, so no guard-stop was produced and the budget is not
  what differs. (Note for a future round: this workload is cheaper than R2 projected — the whole
  2 × 2-generation workload needed 1 181 misses on the specialized route, not ≈2 200.)
* **Fitness cross-check.** All three executions produced the identical trajectory
  `[-0.2510969936847687, -0.22635293006896973]` per rep. Generation 0 is bit-identical to the R2-era
  numpy mirror `np_sr_b`'s first value; generation 1 sits 1.49e-08 absolute / 6.6e-08 relative away from
  `np_sr_b`'s `-0.22635294497013092` — inside the ≤2.03e-07 relative R3-vs-R2 spread already documented
  for the specialized route in `../SUMMARY.md`, and *identical on both routes*, i.e. float-level, not a
  behavioural difference. No fitness regression; the config's expected two generations are now verified
  on IREE (R2's own runlog lists them as unverified).
**Gate check (the b2 records are deliberately NOT manifest ids):** with the three new records on disk,
`python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt
--require-stages` still reports `9 record(s): 9 complete, 0 partial, 0 skipped, 0 missing, 0 malformed
(0 not in the manifest)` / `34 check(s), 0 failure(s)` / `CHECKER_EXIT=0`
(`results/armB/status_check_r3_armB_after_b2.txt`); `manifest_armB.txt` was not edited, so the frozen
Arm B gate is untouched by this phase.
#### Companion window (W2, GPU 0) — EXECUTED 2026-09-12 23:56:11–00:07:12 HKT (window 6): **all 9 ids re-measured, 9/9 exit 0 — the companion records of record**
**Why a companion window.** Every W1 Arm B matrix window (windows 2/3/4) ran on the runner-default GPU 6 while the
≈30 GB `syfang` tenant was resident and bursting to 85–100 %, so the W1 warm/first-call deltas are window-paired and
noisy. R3 handled the same situation with a companion window on a free device (its light-group records of record are
that companion re-measurement; both windows stay visible in the append-only logs, the earlier JSON snapshots in git
history). W2 mirrors that pattern for Arm B: **all nine Arm B matrix ids re-run, one `run_all_armB.sh --only`
invocation per config, on the least-contended device of the moment**, so the committed `results/armB/new/` JSONs
become the W2 records of record while W1 stays retrievable (pointer sha below).
**Device decision (window 6, from the snapshot in `windows.md`).** A 10-sample / 1 s series put **GPUs 0 and 1** at
`0 %` utilization and `1 MiB` resident in *every* sample — the lowest foreign utilization on the box — with ≈48.7 GiB
free device memory (≫ the ≥12 GB requirement and the configs' `--mem-guard-mb 8192` floor, which compares free
device memory). The tie on both criteria was broken by the **lower index ⇒ GPU 0**, which is also the device of R3's
own companion (Arm A light-group) window, so the `np_*` arms are additionally device-paired with their Arm A
companions. GPU 5 was idle but carried 24 395 MiB of whhu's VLLM; GPUs 2/3/4 were busy (22–100 %); GPU 6 (the W1
device) was bursting to 100 %; GPU 7 has a hardware ECC fault (2 263 uncorrected volatile errors) and was never
touched. **Mid-window change:** hyqiu launched two new jobs while the window ran (pid 2497811 on GPU 0 at ≈23:58,
19 706 MiB; pid 2498391 on GPU 1 at ≈00:00, 20 076 MiB), so the first three configs ran with GPU 0 free and the
last six with that ≈19.8 GB tenant present (52–92 % util) — see the window-6 anomaly bullets; `free_mb_start` never
fell below 28 576 MiB, so no guard path was triggered.
**Commands (exact) — nine foreground `run_all_armB.sh --only <id>` invocations, in manifest order (iree-spec → iree
→ numpy), each console captured to a NEW `status_<id>_w2.txt` (W1's `status_<id>.txt` untouched); the runner owns
`ulimit -n 65536` → one `flock -w 7200 /tmp/gpbench_critical.lock` → `timeout -s INT -k 30 3600` (inside the lock):**
```bash
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64_spec    # 23:56:50 → 23:57:13
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64_spec   # 23:57:25 → 23:58:39
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a_spec             # 23:58:45 → 00:04:06
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture64_n64          # 00:04:14 → 00:04:35
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_fixture200_n64         # 00:04:35 → 00:04:57
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_iree_sr_a                  # 00:05:08 → 00:06:44
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_c5r2                    # 00:06:48 → 00:06:52
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_fixture200_n64          # 00:06:52 → 00:06:59
CUDA_VISIBLE_DEVICES=0 bash test/benchmark/r2/round3/run_all_armB.sh --only armB_np_sr_a                    # 00:06:59 → 00:07:12
python3 test/benchmark/r2/round3/check_r3.py --manifest test/benchmark/r2/round3/manifest_armB.txt --require-stages
```
`GPU=0` reached the runner through `CUDA_VISIBLE_DEVICES=0` (the runner reads `GPU="${CUDA_VISIBLE_DEVICES:-6}"`);
the per-config command lines are the `cmd:` header line of each `results/armB/new/<id>.log` W2 block. The Arm B
**agreement pre-flight was deliberately not re-run** — it is contention-independent and stays as recorded. No
`flock` wait was observable (each config's `started` timestamp is seconds after its launch, and every invocation ran
in the foreground of one long-budget call). Runner-level group walls: iree-spec **7 min 16 s** (23:56:50 →
00:04:06), iree **2 min 30 s** (00:04:14 → 00:06:44), numpy **24 s** (00:06:48 → 00:07:12); nothing approached the
3 600 s cap.
**Per-config status table** (per-config exit code = the log's `exit code: 0 (OK)` footer *and* the runner's
id→exit table in the same `status_<id>_w2.txt`; `first_ms`/`warm_ms` follow the metric rules at the top of this
file — eval: `first_call_ms` and `median_warm_ms` over 25 calls; sr: rep0's `cold_first_gen_ms` as the first figure
and the per-rep median of `per_gen_ms[1:]`):
| id | exit | complete / partial | first_ms | warm_ms (rep0 / rep1) | cold builds (json `misses_max` / stage) | fds start→cold→end | wall (started→finished) | harness_wall_s | notes |
|---|---|---|---|---|---|---|---|---|---|
| armB_iree_fixture64_n64_spec | 0 | `complete` / `partial=false` | 17 523.11 | 59.47 (25 calls) | 48 / 300 (json), stage 48 cold + 0 warm misses | 39 → 91 → 91 | 23 s (23:56:50 → 23:57:13) | 20.0 | GPU 0 free (0 % before → 0 % after, `used_start=4 MiB`, `free_mb_start=48 409`); stage block meaningful |
| armB_iree_fixture200_n64_spec | 0 | `complete` / `partial=false` | 63 533.30 | 236.50 (25 calls) | 155 / 400 (json), stage 155 cold + 0 warm misses | 39 → 91 → 91 | 75 s (23:57:25 → 23:58:39) | 70.3 | GPU 0 still free (`used_start=4` → `used_end=441 MiB`, `free_mb_start=48 409`) |
| armB_iree_sr_a_spec | 0 | `complete` / `partial=false` | 117 095.61 (rep0 cold gen) | 11 519.50 / 60.19 (median of `per_gen_ms[1:]`) | 517 / 1 500 (json), stage rep0 = 155 cold + 362 warm, rep1 = 0 | 39 → 91 → 91 | 322 s (23:58:45 → 00:04:06) | 317.1 | hyqiu's new GPU-0 job landed mid-config (`used` 4 → 3 542 MiB); `best_fitness_final=-0.2510969936847687`; per-rep repair rep0 cold 117 095.61 → rep1 cold 226.12 ms |
| armB_iree_fixture64_n64 | 0 | `complete` / `partial=false` | 16 491.61 | 14.84 (25 calls) | 1 / 300 (json), stage 0 cold + 0 warm (dense route) | 39 → 91 → 91 | 22 s (00:04:14 → 00:04:35) | 17.6 | ≈19.8 GB foreign tenant now resident (`used_start=5 269`, `free_mb_start=43 144`, 70 % → 31 %); dense route, degenerate stage block |
| armB_iree_fixture200_n64 | 0 | `complete` / `partial=false` | 18 010.95 | 10.03 (25 calls) | 1 / 400 (json), stage 0 cold + 0 warm (dense route) | 39 → 91 → 91 | 23 s (00:04:35 → 00:04:57) | 19.1 | tenant at ≈19.8 GB (`used_start=19 817`, `free_mb_start=28 596`, 64 % → 83 %); dense route, degenerate stage block |
| armB_iree_sr_a | 0 | `complete` / `partial=false` | 14 003.97 (rep0 cold gen) | 82.56 / 126.79 (median of `per_gen_ms[1:]`) | 170 / 1 500 (json), stage rep0 = 0 cold + 169 warm, rep1 = 0 | 39 → 91 → 91 | 97 s (00:05:08 → 00:06:44) | 93.9 | tenant 52 % → 91 %; `best_fitness_final=-0.2510969936847687`; stage block non-degenerate |
| armB_np_c5r2 | 0 | `complete` / `partial=false` | 194.00 | 22.81 (25 calls) | null (`enforced=false`, no budget) / stage 48 cold + 0 warm misses | 39 → 59 → 59 | 5 s (00:06:48 → 00:06:52) | 1.4 | numpy route, tenant 60 % → 63 %; stage block meaningful |
| armB_np_fixture200_n64 | 0 | `complete` / `partial=false` | 969.27 | 90.15 (25 calls) | null (`enforced=false`) / stage 155 cold + 0 warm misses | 39 → 59 → 59 | 8 s (00:06:52 → 00:06:59) | 3.8 | tenant 77 % → 62 %; stage block meaningful |
| armB_np_sr_a | 0 | `complete` / `partial=false` | 2 705.00 (rep0 cold gen) | 156.14 / 47.98 (median of `per_gen_ms[1:]`) | null (`enforced=false`) / stage rep0 = 155 cold + 362 warm, rep1 = 0 | 39 → 59 → 59 | 14 s (00:06:59 → 00:07:12) | 9.2 | tenant 63 % → 23 %; `best_fitness_final=-0.2510969936847687`; per-rep repair rep0 cold 2 705.00 → rep1 cold 161.62 ms |
**`stages` quality — exactly the W1 pattern, reproduced:** all nine records carry `stages.partial=false`,
`note=""`, `stages.status.installed=true`, the full `wrapped=[decode_population,_run_tree,build_cached,run,
from_numpy,evaluate,sr_fitness]` (7), `missing=[]`, `nested_runs=0`, 25 eval records (eval ids) / 10 + 10 train
records (the `sr` ids), and non-degenerate counters for every specialized-route config (`c5r2`, `fixture200_n64`,
`np_sr_a` and the three `_spec` arms). The two **dense** eval arms (`armB_iree_fixture64_n64`,
`armB_iree_fixture200_n64`) again carry the documented degenerate block (`decode_calls=build_calls=sr_fitness_calls=0`,
only `run_calls=25` non-zero, the whole probe span in `misc_ms`) while `armB_iree_sr_a` reports real counters — i.e.
the R3 stage-probe caveat is a property of the route, not of the window, and it reproduced on a different device.
**W1 → W2 comparison (same ids, same flags, same shadow — only the window and the device differ; W1 = the records
committed up to `b728466`, W2 = the JSONs committed here):**
| id | W2 first_ms | W1 first_ms | Δ | W2 warm_ms | W1 warm_ms | Δ |
|---|---|---|---|---|---|---|
| iree_fixture64_n64_spec | 17 523.11 | 20 043.52 | −12.6 % | 59.47 | 62.19 | −4.4 % |
| iree_fixture200_n64_spec | 63 533.30 | 82 065.28 | −22.6 % | 236.50 | 200.09 | +18.2 % |
| iree_sr_a_spec (rep0 / rep1) | 117 095.61 (cold gen) | 64 950.88 (cold gen) | +80.3 % | 11 519.50 / 60.19 | 12 889.64 / 57.32 | −10.6 % / +5.1 % |
| iree_fixture64_n64 | 16 491.61 | 18 011.84 | −8.4 % | 14.84 | 8.54 | +73.8 % |
| iree_fixture200_n64 | 18 010.95 | 12 227.86 | +47.3 % | 10.03 | 5.20 | +93.1 % |
| iree_sr_a (rep0 / rep1) | 14 003.97 (cold gen) | 15 965.23 (cold gen) | −12.3 % | 82.56 / 126.79 | 130.59 / 127.89 | −36.8 % / −0.9 % |
| np_c5r2 | 194.00 | 191.31 | +1.4 % | 22.81 | 20.78 | +9.8 % |
| np_fixture200_n64 | 969.27 | 995.50 | −2.6 % | 90.15 | 135.94 | −33.7 % |
| np_sr_a (rep0 / rep1) | 2 705.00 (cold gen) | 2 255.92 (cold gen) | +19.9 % | 156.14 / 47.98 | 154.45 / 33.83 | +1.1 % / +41.8 % |
**What is robust across the two windows (contention-independent) and what is window noise.**
*Robust — identical in W1 and W2, and independent of the device:* every exit code and `complete`/`partial` flag;
the cold-build traffic per config (json `misses_max` `48 / 155 / 517 / 1 / 1 / 170`, stage rep totals `48+0 /
155+0 / 155+362 and 0+0 / 0+0 / 0+0 / 0+169 and 0+0 / 48+0 / 155+0 / 155+362`) — the same numbers as the Arm A
twins; the fd telemetry (`39 → 91 → 91` on the six IREE configs, `39 → 59 → 59` on the three numpy ones); the
nanobind teardown leak counts (dense `677 / 677 / 1 820` instances, `_spec` `4 052 / 12 902 / 3 854` — the same
values W1 and window 3 reported); and the **bitwise-identical 10-generation fitness trajectory** of all three
training ids (`per_gen_best_fitness` equal element-for-element between W1 and W2, `best_fitness_final` =
`-0.2510969936847687` everywhere). No guard-skip, no timeout, no interrupt, no SIGSEGV, no re-run.
*Window noise — do not read as regressions or improvements:* every first-call / warm / per-generation figure above.
They move both ways (the `_spec` first calls −13 %/−23 %, the dense eval warms ×1.7 and ×1.9 on a 5–15 ms base, the
numpy warm median −34 % on `fixture200_n64` and +9.8 % on `c5r2`, the `sr` rep-1 medians +5 %/−0.9 %/+42 %), they
straddle the documented per-structure-compile spread (± a factor of a few) and the shared-host ±10 % band, and the
`sr` per-generation series are bimodal in both windows — W2 `iree_sr_a` rep1 `per_gen_ms[1:]` =
`[329.5, 167.0, 129.7, 126.8, 41.4, 40.7, 41.8, 52.6, 129.1]`, W1's the two-cluster shape documented in window 3 —
so the median is an unstable comparison point. **No W1 claim flips in W2**: the Arm B-vs-Arm A verdict (the patched
ETL changes no fitness, no build traffic and no number the round cares about) holds identically in both windows,
which is exactly what the companion window was run to establish.
**Validator (gate (ii), re-run over the W2 records — all 9 ids on disk, FINAL):**
```text
9 record(s): 9 complete, 0 partial, 0 skipped, 0 missing, 0 malformed (0 not in the manifest)
34 check(s), 0 failure(s)
armB_iree_fixture64_n64_spec   new   eval64  complete   17523.11      59.47  48 (reps[0].totals.cold_builds)   0 (reps[0].totals.warm_misses)
armB_iree_fixture200_n64_spec  new   eval64  complete   63533.30     236.50  155 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_iree_sr_a_spec            new   sr      complete  117095.61   11519.50  155 (reps[0].totals.cold_builds)  362 (reps[0].totals.warm_misses)
armB_iree_fixture64_n64        new   eval64  complete   16491.61      14.84  0 (reps[0].totals.cold_builds)    0 (reps[0].totals.warm_misses)
armB_iree_fixture200_n64       new   eval64  complete   18010.95      10.03  0 (reps[0].totals.cold_builds)    0 (reps[0].totals.warm_misses)
armB_iree_sr_a                 new   sr      complete   14003.97      82.56  0 (reps[0].totals.cold_builds)    169 (reps[0].totals.warm_misses)
armB_np_c5r2                   new   eval64  complete     194.00      22.81  48 (reps[0].totals.cold_builds)   0 (reps[0].totals.warm_misses)
armB_np_fixture200_n64         new   eval64  complete     969.27      90.15  155 (reps[0].totals.cold_builds)  0 (reps[0].totals.warm_misses)
armB_np_sr_a                   new   sr      complete    2705.00     156.14  155 (reps[0].totals.cold_builds)  362 (reps[0].totals.warm_misses)
CHECKER_EXIT=0
```
Full stdout kept verbatim as `results/armB/status_check_r3_armB_w2_final.txt` (raw checker stdout plus this phase's
appended `CHECKER_EXIT=0` marker line, the same convention as `status_check_r3_armB_final.txt`). `manifest_armB.txt`
was not edited: the same 9 ids, the same 9 paths, the same `stages` markers — only the JSON contents behind those
paths changed (W2 above W1). The dense eval arms still show the stage-sourced `0` in the checker's `cold_builds`
column against their JSON-level `1` — the documented dense-route display gap, unchanged.
**Records of record.** The committed `results/armB/new/*.json` files are now the **W2** records of record; the
**W1 JSONs remain retrievable in git history at the pre-window tip `b728466`** (`git show b728466:test/benchmark/
r2/round3/results/armB/new/<id>.json`; every W1 JSON's own last-change commit is listed in the footer below), and the
append-only per-config logs under `results/armB/new/*.log` carry **both** windows side by side (each W2 block starts
with its own `Arm B config <id> (side=new …) started …` header and ends with `exit code:` / `gpu (after)` /
`finished`; the W1 blocks above them are untouched).
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
* **Window 4 / GPU 6 — the `numpy` group ran clean in one pass: 3/3 exit 0, no GUARD-SKIP, no timeout, no
  interrupt, no re-run, no kill.** One foreground `run_all_armB.sh --only` invocation per config
  (5 s / 8 s / 12 s wall, group total 52 s), well inside the runner's 3600 s cap and the platform's call
  budget, so neither the reaped-detached-batch failure mode nor a mid-config kill recurred; both
  fixtures were present (no exit-3 guard path) and every log ends with `exit code: 0 (OK)`,
  `gpu (after)` and `finished`.
* **Window 4 / GPU 6 — device kept under the documented rule, and the `numpy` route makes the choice
  cheap either way.** Two GPU-6 utilization series were taken at window start (12 samples / 3 s:
  `0 0 0 0 0 0 0 0 85 42 98 97 %`; 15 samples / 3 s: `0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 %`), i.e. the
  ≈30 GB `syfang` tenant **bursts** and never holds the sustained ≥90 % that triggers a switch; GPUs 0
  and 1 had meanwhile become tenant-free (hyqiu's window-3 job exited), so GPU 6 was explicitly chosen
  for comparability with windows 1b/2/3 rather than by default inertia. GPU 7 (hardware ECC fault) was
  never touched. Per-config contention is the `gpu (before)/(after)` pair in each committed log:
  `29 → 0 %` (c5r2), `25 → 4 %` (fixture200), `38 → 0 %` (sr). Host loadavg at window start:
  80.59 / 82.54 / 92.51, 46 users — the quietest window of the round.
* **Window 4 — the three `stages` blocks are meaningful, exactly like the `iree-spec` group and unlike
  the dense `iree` eval arms** (verbatim counters in the `numpy` subsection above). Every record
  reports `partial=false`, `note=""`, `installed=true` and real per-stage counters for the specialized
  route (25 eval records; 10 + 10 train records), and the build traffic is *identical* to the Arm A
  twins (48 / 155 / 155 + 362 misses, 1 325 / 4 275 / 632 build calls), which is the second
  work-load-identity cross-check next to the bitwise `sr` fitness trajectory. The R3 caveat about a
  dense-route run carrying a degenerate stage block therefore does not apply to this group — on
  `numpy` the engine keeps the specialized route even though the backend is the CPU reference
  interpreter.
* **Window 4 — no FD or teardown incident**: each config grew 39 → 59 fds across its cold phase and
  ended at 59 (the Arm A `np_*` twins read the same 39 → 59 → 59), and the logs contain **zero**
  `nanobind: leaked …` lines — the numpy route creates no IREE runtime objects, so the teardown noise
  of the `_spec`/`iree` windows is simply absent here. `ulimit -n 65536` was raised by the runner
  first (the group needs no raise; it is the runner's uniform order).
* **Window 4 — the Arm A↔Arm B `numpy` deltas are window-paired, not device-paired.** Arm A's `np_*`
  records are the companion light-group re-measurement window taken on **GPU 0**
  (`env.CUDA_VISIBLE_DEVICES=0` in `results/new/np_*.json`) while Arm B's numpy group used the
  runner-default GPU 6, and the numpy route is CPU-bound host-side work — so the +15.6 % warm delta on
  `np_fixture200_n64` and the +31.6 % rep0 median on `np_sr_a` (whose `per_gen_ms` is bimodal at
  ≈79–724 ms) are shared-box noise, not added work: the same window shows `np_c5r2` at −10.8 % warm,
  the build counts are equal, and the `sr` fitness trajectory is bitwise identical.
* **Window 5 / GPU 6 — attempt 2's first execution silently ran the wrong route (operator error, caught
  before anything was committed).** The hand-rolled launcher dropped the `EVOGP_EVAL_MODE=specialized`
  assignment before `env` (the log's `cmd:` line has no such variable and the record's `env` block lacks
  it), so the run took the default `auto` ⇒ dense route again — same workload/flags/shadow as attempt 1
  (431 misses, fds 39 → 91 → 91, exit 0, identical fitness). It was caught immediately by the `env`
  block/build counts and fixed by re-executing attempt 2 with the verbatim command (no variable
  templating) and re-checking `env.EVOGP_EVAL_MODE=specialized` in the corrective record; the
  mis-invoked execution is preserved verbatim as
  `results/armB/new/armB_iree_sr_b2_spec_botched_dense_route.{json,log}` (with an operator annotation
  appended to its log) and is never quoted as attempt 2.
* **Window 5 / GPU 6 — no crash, no guard-stop, no timeout, no interrupt, and no further re-run.** All
  three executions exited 0 (217 s / 266 s / 597 s, each well inside the 3600 s cap, `partial=false`
  everywhere); the cold-build guard peaked at 431/1500 (dense) and 1181/1500 (specialized) and never
  tripped; `fds.end` stayed at 91 of 65 536 in all three — the same flat trajectory as the other Arm B
  `iree` arms, at 2.3× the build count of the largest of them.
* **Window 5 / GPU 6 — fitness vs the R2 numpy mirror, and the record set.** Both attempts' per-rep
  trajectories are exactly `[-0.2510969936847687, -0.22635293006896973]` (bit-identical across the
  dense, botched-dense and specialized executions); generation 0 matches the R2-era numpy mirror
  `np_sr_b`'s first value bit-for-bit and generation 1 sits 1.49e-08 absolute / 6.6e-08 relative from
  `-0.22635294497013092`, inside the documented ≤2.03e-07 relative R3-vs-R2 spread. The three b2
  records are deliberately not manifest ids: `check_r3.py --manifest manifest_armB.txt --require-stages`
  still reports 9 complete / 34 checks / 0 failures with them on disk
  (`results/armB/status_check_r3_armB_after_b2.txt`), so the frozen Arm B gate is unaffected.
* **Window 6 / GPU 0 — the companion window ran clean (9/9 exit 0, no GUARD-SKIP, no timeout, no interrupt, no kill,
  no re-run), but the chosen device was taken over mid-window.** The device decision was taken on a genuinely idle
  GPU 0 (0 % / 1 MiB in all 10 samples, ≈48.7 GiB free), and the first three configs (the `iree-spec` group) ran
  free; at ≈23:58 hyqiu launched pid 2497811 on GPU 0 (19 706 MiB) and at ≈00:00 pid 2498391 on GPU 1
  (20 076 MiB), so the last six configs ran with that ≈19.8 GB tenant on the chosen card (per-config `gpu
  (before)/(after)` pairs in the logs: `0/4` → `70/31 %` → `64/83 %` → `52/91 %` → `60/63 %` → `77/62 %` → `63/23 %`,
  with `nvidia_smi_used_mb_start` climbing 4 → 5 269 → 19 817 MiB). The configs' `--mem-guard-mb 8192` floor was
  never close (`free_mb_start` minimum **28 576 MiB**), so the guard never armed and no exit-3 path was taken. A
  companion window was **not** repeated on a third device: at window end no tenant-free GPU was left on the box
  (GPU 5 has 24 395 MiB of VLLM resident, GPU 6's ≈30 GB tenant still bursting, GPU 7 is ECC-faulted), so the W1→W2
  deltas above stay explicitly *window-paired* — the same reading rule the W1 tables carry.
* **Window 6 — device-selection record.** GPUs 0 and 1 tied on both selection criteria (0 % util, 1 MiB resident in
  every sample of the 10-sample series) and the tie was broken by the lower index ⇒ **GPU 0**, which is also the
  device of R3's own companion (Arm A light-group) window; GPU 5 was idle but non-empty (24 395 MiB), GPUs 2/3/4 were
  busy, GPU 6 was bursting to 100 %, GPU 7 was excluded (2 263 uncorrected volatile ECC errors). Full snapshot,
  compute-app inventory and rationale are in `windows.md` window 6.
* **Window 6 — no FD, guard or teardown incident**: every JSON is `complete: true` / `partial=false`, every log
  footer reads `exit code: 0 (OK)` with a `gpu (after)` line, the fd trajectories are flat (`39 → 91 → 91` on the six
  IREE configs, `39 → 59 → 59` on the three numpy ones, exactly the W1 values) and the nanobind teardown leak counts
  equal W1's (dense `677 / 677 / 1 820`; `_spec` `4 052 / 12 902 / 3 854`); the shared lock was released after every
  config and `ulimit -n 65536` was raised by the runner first.
* **Window 6 — the stage-block route caveat reproduced on a different device, unchanged**: the two dense eval arms
  carry the degenerate `stages` block, `armB_iree_sr_a` carries real counters, and the `_spec`/numpy configs carry
  fully populated ones — evidence that the R3 caveat is a property of the evaluation route, not of the window or the
  card. Gate (ii) is unaffected (all nine `stages` markers satisfied).
* **Window 6 — operator-side naming mishap in the first pair of invocations (configs 4 and 5), caught before the
  write-up.** The two dense-`iree` consoles were captured with a `${id}`-templated path that this platform shell
  expands to empty, so both landed in one mis-named `status__w2.txt` (config 4's console overwritten by config 5's)
  and the two per-config commits `c17d1c4` / `d88a03b` received the same truncated subject
  `round3: Arm B W2 config (, exit 0, GPU 0)`. Repair (commit `b7aa8e4`): config 4's console restored verbatim from
  `c17d1c4` (`git show`), config 5's from the working file, both committed under the correct
  `status_armB_iree_fixture64_n64_w2.txt` / `status_armB_iree_fixture200_n64_w2.txt` names, and the mis-named file
  removed. **No JSON, no log and no manifest row was ever affected** — the records and the append-only logs were
  correct all along; only the console captures' filenames were wrong. The same shell quirk left config 1's
  `status_*_w2.txt` `RUNNER_EXIT=` marker empty; that file now carries a corrected `RUNNER_EXIT=0` line plus an
  operator annotation (commit `c24f999`), and every W2 status file still carries the runner's own id→exit table.
**Analysis of record:** Arm B's completed write-up is **`round3/SUMMARY_armB.md`**, whose every table is
generated by `round3/compare_rounds_armB.py` (which imports `round3/compare_rounds.py` for the metric
rules, so the R2-vs-R3 and R2-vs-Arm-B tables cannot drift apart); the baseline it is read against is
`round3/SUMMARY.md` (Arm A) and, one level up, `../REPORT.md` (R2). Verdict: the patched ETL checkout
**changes nothing the round measures numerically** — all six same-stem Arm A → Arm B `values_sha256`
pairs are bit-identical, every `per_gen_best_fitness` trajectory is identical element-for-element across
Arm A / W1 / W2 (both reps, all three `sr` ids), the cold-build traffic (`48 / 155 / 517 / 1 / 1 / 170`
JSON misses) and every exit code match, and both gates are green (`check_r3.py --manifest
…/manifest_armB.txt --require-stages` → 9 complete / 34 checks / 0 failures; `check_agreement_armB.py`
→ PASS / 38 checks / 0 failures). The one number that moved clear of the noise band is `iree` fd
economy — a flat `39 → 91 → 91` at 170, 517 and 1181 builds against Arm A's 2395 / 5819 / 9723 / 17243
at *identical* build counts (`etl/backends/adapters/iree.py` per-thread device cache), with the
`iree_sr_b2` re-attempt completing exit 0 on both routes where R2 saw its SIGSEGV (non-headline: the b2
ids are deliberately not manifest ids). Every A→B warm-median delta runs in both directions and sits
inside the shared-box ±10 % band, so the write-up quotes the W2 records of record as the headline with
W1 (`b728466` snapshots) and Arm A's own companion window as window-paired context, and prices the
patched lookup path as "no regression, same numbers" rather than as a measured speed-up.
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
window-3 anomaly bullets. The `numpy` matrix phase (window 4) committed `a4cf2ad`
(`etl_import_probe.txt` fifth block + `windows.md` window 4), `f3a4cae` (`armB_np_c5r2.{json,log}` +
`status_*.txt`), `8104481` (`armB_np_fixture200_n64.{json,log}` + `status_*.txt`), `f347d47`
(`armB_np_sr_a.{json,log}` + `status_*.txt`; the `*.log` files again needed `git add -f` because
`.gitignore:32` hides `*.log`, exactly as for Arm A's records) and the follow-up commit of that phase,
which carries `results/armB/status_check_r3_armB_final.txt` (the first all-9-ids validator run, exit 0),
the `numpy`-group subsection above and the five window-4 anomaly bullets. The `iree_sr_b2` re-attempt
phase (window 5) committed `186a25e` (the three b2 records — attempt 1, the corrective attempt 2 and the
preserved mis-invoked first execution of attempt 2 — plus their three logs and
`status_check_r3_armB_after_b2.txt`; the `*.log` files again needed `git add -f` because `.gitignore:32`
hides `*.log`, exactly as for Arm A's records) and `174f33a`
(`etl_import_probe.txt` sixth block + `windows.md` window 5), and this follow-up commit carries the
filled-in `iree_sr_b2` re-attempt subsection and the three window-5 anomaly bullets.
**W1 records pointer:** the pre-W2 tip of `results/armB/new/*.json` is **`b728466`** ("round3: iree_sr_b2
re-attempt commits record (window-5 phase SHAs) in the Arm B runlog footer"), i.e. every W1 JSON snapshot is
retrievable with `git show b728466:test/benchmark/r2/round3/results/armB/new/<id>.json`. Verified in the W2 window
with `git log --oneline -1` and by listing where each W1 JSON last changed: `armB_iree_fixture64_n64_spec` `fc4e9dd`,
`armB_iree_fixture200_n64_spec` `1795c0d`, `armB_iree_sr_a_spec` `72420d1`, `armB_iree_fixture64_n64` `8ee004c`,
`armB_iree_fixture200_n64` `a59dac2`, `armB_iree_sr_a` `5fff188`, `armB_np_c5r2` `f3a4cae`,
`armB_np_fixture200_n64` `8104481`, `armB_np_sr_a` `f347d47` (the three `iree_sr_b2*` records, not matrix ids, last
changed in `186a25e` and were **not** re-run by W2).
**Commits the W2 (companion) window covers:** `5f8e6fd` (`etl_import_probe.txt` seventh block + `windows.md`
window 6, committed before the first invocation), `8f7b0a9` (`armB_iree_fixture64_n64_spec.{json,log}` +
`status_*_w2.txt`), `2fc3e65` (`armB_iree_fixture200_n64_spec.*`), `425d780` (`armB_iree_sr_a_spec.*`), `c17d1c4` and
`d88a03b` (the two dense-`iree` configs `armB_iree_fixture64_n64` / `armB_iree_fixture200_n64`; their subjects lost
the id to the shell quirk documented in the window-6 anomaly bullets), `c086d85` (`armB_iree_sr_a.*`), `8a28d6c`
(`armB_np_c5r2.*`), `0b70644` (`armB_np_fixture200_n64.*`), `b2f3af9` (`armB_np_sr_a.*`), `b7aa8e4` (status-file
re-names + the two recovered consoles, minus the mis-named file removed), `c24f999` and `4153ddd`
(`status_*_w2.txt` for config 1 repaired with its corrected `RUNNER_EXIT=0` marker, then the repair annotation's
literal `${...}` spelling restored) and this follow-up commit, which carries the W2 subsection above,
`results/armB/status_check_r3_armB_w2_final.txt` (the re-run gate, `CHECKER_EXIT=0`), the five window-6 anomaly
bullets and the post-run state appended to `windows.md` window 6. Every W2 config commit used `git add -f` for the
`*.log` files (`.gitignore:32` hides `*.log`, exactly as for Arm A's records).
