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
