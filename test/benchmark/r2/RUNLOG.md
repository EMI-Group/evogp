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

## IREE group — (pending; to be appended by the iree operator)

Not run in this entry. The 12 `iree_*` configs are the remaining manifest ids (expected wall: tens of
minutes; `iree_sr_b2` is the heaviest and may need `--timeout 7200`). Append a section in the same shape:
environment + GPU state, exact commands, per-config status table, `check_r2.py --manifest` result, wall clock,
deviations.
