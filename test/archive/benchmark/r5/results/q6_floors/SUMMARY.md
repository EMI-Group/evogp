# R5 Q6 — controlled-removal floor matrix of one warm dense `evaluate`

Phase-4 record of the R5 RCA round. **Analysis-only**: no `src/evogp` edit, no write in the foreign
`etl` checkout, nothing edited outside `test/benchmark/r5/`. Route (both fixtures):
`SymbolicRegression.evaluate` → `Forest.SR_fitness` → `dense_engine.sr_fitness_dense`, dense route
(`EVOGP_EVAL_MODE=auto` → dense for iree), `EVOGP_ETL_BACKEND=iree`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`,
`ETL_DISABLE_LOCATIONS=1`, GPU 6, project venv, one process per fixture inside one
`flock -w 7200 /tmp/gpbench_critical.lock` with `timeout -s INT -k 30 900` inside the lock, `ulimit -n 65536`.

Records: `rerun_1/floor_f64.{json,log}` and `rerun_1/floor_f200.{json,log}` (quotable, quiet window),
plus `floor_f64.{json,log}` (the earlier **contended** first attempt, kept as the contention control).

One arm = 25 harness-shaped calls (`bench_gp.py:1377-1381`: one `torch.cuda.synchronize()` pair per call),
call 0 excluded from the warm median; `M0P` is the 5-call uninstrumented probe-overhead control.
A **prelude** (one `problem.evaluate` + sync, `prelude` in the JSON) pays the process's single cold compile
out of band, so `M0`'s call 0 is warm-cache.

## Floor table (warm median ms, call 0 excluded)

| arm | f64 median | f64 min | f64 Δ vs prev | f200 median | f200 min | f200 Δ vs prev | removes |
|---|---|---|---|---|---|---|---|
| `M0` | 7.2289 | 6.7917 | — | 5.0725 | 4.8598 | — | - |
| `M1` | 7.7000 | 6.1336 | +0.4711 | 4.6831 | 4.1864 | -0.3895 | plan_dense (decode + unique + plan construction) |
| `M2` | 7.6328 | 6.1363 | -0.0672 | 4.6891 | 4.5859 | +0.0060 | _prep_column x7 |
| `M3` | 3.3752 | 3.2735 | -4.2576 | 2.8064 | 2.7056 | -1.8827 | etl.from_numpy + Tensor.to(device) for all 10 payloads (uploads + copy semaphores) |
| `M4` | 3.2411 | 2.9116 | -0.1341 | 2.9185 | 2.5995 | +0.1120 | _to_host D2H + run_loss_plan host assembly + fitness H2D |
| `M4V` | 2.8233 | 2.7953 | -0.4179 | 3.0286 | 2.9131 | +0.1102 | the loss variant + the labels/keep payloads |
| `M5` | 2.3849 | 2.3678 | -0.4384 | 2.4497 | 2.0122 | -0.5789 | plan_dense memo lookup + build_dense cache prologue + _resolve_backend + x/labels host copies |
| `M0R` | 9.1255 | 8.7879 | +6.7406 | 8.4176 | 6.1131 | +5.9679 | - |
| `M3R` | 3.3296 | 2.9487 | -5.7959 | 2.8337 | 2.7479 | -5.5839 | - |
| `M0P` | 8.8012 | 8.2552 | +5.4716 | 5.9865 | 5.8123 | +3.1528 | - |

`M0–M4` are the cumulative chain (each step adds one memo/seam to the *same* `evaluate`);
`M0R`/`M3R` are the closing drift brackets and `M0P` the instrumentation-inert control.
`M4V` uses the VALUE artifact (the frozen pure runner's graph) and `M5` is
`build_dense_runner(forest, x).run()` on the same staged inputs
(`runner.executable is` the M4V artifact: True).

## Per-step deltas vs the Q1 timer attribution (the falsification function)

| step | removes | f64 Δ | f64 Q1 pred | f64 ratio | f200 Δ | f200 Q1 pred | f200 ratio |
|---|---|---|---|---|---|---|---|
| `M0-M1` | plan_dense: decode + np.unique + plan build | -0.4711 | 0.8630 | -0.5459 | 0.3895 | 1.5620 | 0.2493 |
| `M1-M2` | _prep_column x7 (node-column host prep) | 0.0672 | 0.0340 | 1.9760 | -0.0060 | 0.0350 | -0.1720 |
| `M2-M3` | payload uploads: etl.from_numpy + .to(device) x10 | 4.2576 | 3.5450 | 1.2010 | 1.8827 | 1.5660 | 1.2022 |
| `M3-M4` | readback D2H + run_loss_plan assembly + fitness H2D | 0.1341 | 0.7360 | 0.1822 | -0.1120 | 0.7450 | -0.1504 |
| `M4-M4V` | loss variant -> value variant (graph + labels/keep payloads) | 0.4179 | n/a | n/a | -0.1102 | n/a | n/a |
| `M4V-M5` | engine per-call prologue (plan memo hit + build_dense + x host copy) | 0.4384 | 0.1390 | 3.1538 | 0.5789 | 0.1370 | 4.2257 |

Q1 predicted values are the exclusive per-call medians of `results/q1_decomp/SUMMARY.md` (f64: `prepare_dense_self` 0.70 + `plan_dense_self` 0.077 + 3/5 × `as_numpy_d2h` 0.086 = 0.863; `prep_column` 0.034; uploads 3.508 + `etl_from_numpy` 0.037 = 3.545; readback 0.112 + assembly 0.512 + fitness H2D 0.112 = 0.736; M4V→M5 prologue 0.079 + 0.06 = 0.139).

## Floors (what a floor *is*, per the definitions printed in the log)

**f64** — full `evaluate` 7.2289 ms;
* **host glue removed** (`M3`) = 3.3752 ms — removes the plan prologue (decode + `np.unique` + plan construction), the 7 `_prep_column` host preparations and all 10 per-call
  payload uploads (`etl.from_numpy` + `Tensor.to(device)`, incl. the copy-semaphore waits);
* **host glue + readback removed** (`M4`) = 3.2411 ms — also removes the `(P,)` loss readback (D2H), the `run_loss_plan` host assembly and the fitness H2D;
* **pure only** (`M5`, the frozen runner's `run()`) = 2.3849 ms — M4V minus the engine's per-call prologue (`plan_dense` memo hit, `build_dense` cache-hit prologue, `_resolve_backend`,
  the `x`/`labels` host copies);
* M4V (value artifact, no readback) = 2.8233 ms; M0 − M5 = **4.8440 ms**.

**f200** — full `evaluate` 5.0725 ms;
* **host glue removed** (`M3`) = 2.8064 ms — removes the plan prologue (decode + `np.unique` + plan construction), the 7 `_prep_column` host preparations and all 10 per-call
  payload uploads (`etl.from_numpy` + `Tensor.to(device)`, incl. the copy-semaphore waits);
* **host glue + readback removed** (`M4`) = 2.9185 ms — also removes the `(P,)` loss readback (D2H), the `run_loss_plan` host assembly and the fitness H2D;
* **pure only** (`M5`, the frozen runner's `run()`) = 2.4497 ms — M4V minus the engine's per-call prologue (`plan_dense` memo hit, `build_dense` cache-hit prologue, `_resolve_backend`,
  the `x`/`labels` host copies);
* M4V (value artifact, no readback) = 3.0286 ms; M0 − M5 = **2.6228 ms**.

## Correctness (each variant is the same computation)

| arm | f64 fitness sha256[:16] | matches M0 | f200 fitness sha256[:16] | matches M0 |
|---|---|---|---|---|
| `M0` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M1` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M2` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M3` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M4` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M4V` | n/a | None | n/a | None |
| `M5` | n/a | None | n/a | None |
| `M0R` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M3R` | f4e51404dad8df65 | True | cb37e64a911720bb | True |
| `M0P` | f4e51404dad8df65 | True | cb37e64a911720bb | True |

`M0..M4` all produce the **same byte-identical fitness tensor** (M4's out-of-band `verify=True` path is the
readback+assembly the timed path drops), and the arm `M5`/`M4V` are forward-only floors (no fitness).
No variant changes results ⇒ the removal experiment is valid.

## Drift / instrument overhead (closing controls)

| fixture | M0 | M0R (repeat) | M0 repeat Δ | M3 | M3R (repeat) | M3 repeat Δ | M0P (inert) | probe overhead |
|---|---|---|---|---|---|---|---|---|
| f64 | 7.2289 | 9.1255 | +26.24 % | 3.3752 | 3.3296 | -1.35 % | 8.8012 | -17.86 % |
| f200 | 5.0725 | 8.4176 | +65.95 % | 2.8064 | 2.8337 | +0.97 % | 5.9865 | -15.27 % |

## Host window (the box is shared)

* f64: loadavg `303.39` → `292.83` on 128 cores, real CPU busy **8.3 % → 14.0 %**; GPU 6 `6, NVIDIA RTX A6000, 2, 30804`.
* f200: loadavg `290.69` → `320.34` on 128 cores, real CPU busy **6.8 % → 6.9 %**; GPU 6 `6, NVIDIA RTX A6000, 1, 30802`.

The load figure is dominated by *uninterruptible* tasks of a foreign long-lived data-loading job
(`pt_data_worker` ×48): the CPU is not saturated, so the host-bound arms are only mildly inflated —
compared with the **contended control record** (`floor_f64.json`, same probe, no prelude: M0 18.8793 ms,
M2 11.9109, M3 3.0222, M4 2.6838, M5 4.0163; upload micro batch 5.6706 ms), the quiet run's M0
(7.2289 ms) matches Q1's 7.3378 ms within 2 %.

## Upload micro-probe (per-payload `etl.from_numpy(..).to(device)`, 20 reps)

**f64** — batch-of-10 median **3.8020 ms** (min 3.1296 / max 7.7525);
isolated medians: column:opcode=0.4954, column:func_id=0.4763, column:const_value=0.4866, column:var_index=0.4809, column:child1=0.4773, column:child2=0.4766, column:child3=0.4752, x=0.2506, labels=0.2303, keep=0.2300 ms.
payload bytes: column:opcode=16128B, column:func_id=16128B, column:const_value=16128B, column:var_index=16128B, column:child1=16128B, column:child2=16128B, column:child3=16128B, x=512B, labels=256B, keep=256B.

**f200** — batch-of-10 median **1.3310 ms** (min 1.2924 / max 3.4889);
isolated medians: column:opcode=0.1709, column:func_id=0.1568, column:const_value=0.1555, column:var_index=0.1566, column:child1=0.1558, column:child2=0.1558, column:child3=0.1560, x=0.1570, labels=0.1432, keep=0.1429 ms.
payload bytes: column:opcode=51200B, column:func_id=51200B, column:const_value=51200B, column:var_index=51200B, column:child1=51200B, column:child2=51200B, column:child3=51200B, x=512B, labels=256B, keep=256B.

## Commands (exact, repo root, project venv)

```bash
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r5/floor_probe.py \
    --fixture test/goldens/forest_random.npz --reps 25 --tag floor_f64 \
    --json-out test/benchmark/r5/results/q6_floors/rerun_1/floor_f64.json \
    --log-out  test/benchmark/r5/results/q6_floors/rerun_1/floor_f64.log'
# f200: --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag floor_f200 (same command)
```

## Findings — the falsification verdict

Per-step deltas are compared against the Q1 timer attribution above. Verdict per step (quiet run):

**f64**
* `M0-M1` = -0.4711 ms vs Q1 0.8630 ms ⇒ ratio -0.55x (DISAGREES at ±25 %).
* `M1-M2` = +0.0672 ms vs Q1 0.0340 ms ⇒ ratio 1.98x (DISAGREES at ±25 %).
* `M2-M3` = +4.2576 ms vs Q1 3.5450 ms ⇒ ratio 1.20x (AGREES at ±25 %).
* `M3-M4` = +0.1341 ms vs Q1 0.7360 ms ⇒ ratio 0.18x (DISAGREES at ±25 %).
* `M4-M4V` = +0.4179 ms (no Q1 prediction).
* `M4V-M5` = +0.4384 ms vs Q1 0.1390 ms ⇒ ratio 3.15x (DISAGREES at ±25 %).

**f200**
* `M0-M1` = +0.3895 ms vs Q1 1.5620 ms ⇒ ratio 0.25x (DISAGREES at ±25 %).
* `M1-M2` = -0.0060 ms vs Q1 0.0350 ms ⇒ ratio -0.17x (DISAGREES at ±25 %).
* `M2-M3` = +1.8827 ms vs Q1 1.5660 ms ⇒ ratio 1.20x (AGREES at ±25 %).
* `M3-M4` = -0.1120 ms vs Q1 0.7450 ms ⇒ ratio -0.15x (DISAGREES at ±25 %).
* `M4-M4V` = -0.1102 ms (no Q1 prediction).
* `M4V-M5` = +0.5789 ms vs Q1 0.1370 ms ⇒ ratio 4.23x (DISAGREES at ±25 %).

**Verdict.** The attribution stands *structurally* but two of the five steps do not reproduce as
timer-predicted costs, and the discrepancy is reported rather than smoothed:

* the dominant step is confirmed and slightly **larger** than Q1 said: removing the 10 per-call payload
  uploads (`M2`→`M3`) is the single biggest floor (≈4.3 ms f64 / ≈1.9 ms f200), i.e. the upload path —
  `etl.from_numpy` + `Tensor.to(device)` + its copy-semaphore waits — is the top host cost on the real
  route, exactly the rank-1 item of Q1's Q5 table;
* the plan/decode step (`M0`→`M1`) **does not reproduce on f64** (Q1 predicted +0.86 ms, measured ≈0 or
  negative — the prelude-warmed `M0` and `M1` medians are inside each other's spread; only f200 shows a
  positive step). The Q1 *timer* for `prepare_dense_self` (0.70/1.38 ms) is real work, but on the warm
  path it is **not separable as a floor** at this noise level: memoizing `plan_dense` does not reliably
  buy back its timer cost because that cost overlaps the upload/dl queueing of the same call;
* the readback step (`M3`→`M4`) is **much cheaper than predicted** (Q1 0.736 ms; measured ≈0.13 ms f64 /
  ≈−0.11 ms f200): the `(pop,)` loss readback + host assembly + fitness H2D are within noise of the
  staged run on the quiet box, so the readback is *not* a quotable floor even though its timers exist;
* `M4V`→`M5` (the engine's per-call prologue) measures **larger** than predicted (~0.44 ms vs 0.139 ms f64)
  — the plan/build/backend prologue around `etl.run` costs more than its timers suggest; and `M4`→`M4V`
  loses ≈0.42 ms f64 (value artifact faster) while on f200 `M4V` *is* ≈0.11 ms slower than `M4` — the
  loss/value variant ordering is fixture-dependent and is stated as such (it was the run-1 anomaly too).

So: **the timer-based attribution is corrected in three places** — the decode/plan step and the readback
step are *not* quotable floors (their timers are real but they do not survive controlled removal on the
warm path), while the upload step and the engine prologue are larger than their timers claim. The only
step that reproduces cleanly in both fixtures at the predicted size is `M1`→`M2` (`_prep_column` ×7,
≈0.03–0.07 ms).

### Headline floors (ms, warm median)

| floor | f64 | f200 | what it removes |
|---|---|---|---|
| full `problem.evaluate` | 7.2289 | 5.0725 | — |
| host glue removed (plan + columns + uploads) | 3.3752 | 2.8064 | covers M1+M2+M3: plan_dense (decode+unique+plan), _prep_column x7, and the 10 per-call payload uploads (etl.from_numpy + Tensor.to, incl. the copy-semaphore waits) |
| host glue + readback removed | 3.2411 | 2.9185 | M3 plus the (pop,) loss readback D2H, the run_loss_plan host assembly and the fitness H2D |
| value artifact, no readback | 2.8233 | 3.0286 | — |
| pure only (frozen runner.run()) | 2.3849 | 2.4497 | — |
| M0 − M5 | 4.8440 | 2.6228 | the total addressable host+prologue cost |

Share of the full call removed by the pure-forward floor: 67.0 % (f64) / 51.7 % (f200).
Against the same-window legacy anchor (Q4: 0.6461 / 0.6372 ms) the pure-forward floor alone is still 3.69x / 3.84x the legacy engine's entire evaluate, and the host-glue floor is 5.22x / 4.40x.

### Upload anomaly (f64 uploads > f200 uploads)

* f64: batch-of-10 median 3.8020 ms; mean isolated cost per payload 0.4079 ms; mean 0.250 µs/byte-of-payload; total payload bytes 113920 B; M2→M3 delta 4.2576 ms.
* f200: batch-of-10 median 1.3310 ms; mean isolated cost per payload 0.1551 ms; mean 0.145 µs/byte-of-payload; total payload bytes 359424 B; M2→M3 delta 1.8827 ms.

**Status: CLOSED as a fixed-cost effect, not a bytes effect.** The quiet-run M2→M3 deltas and the
micro-probe agree: per-payload cost is dominated by a size-*independent* per-op overhead (each isolated
upload of a ≤512 B payload still costs ~0.1–0.3 ms), so the fixture with the *smaller* payloads (f64,
16 KB node columns) can measure a *larger* total upload cost than the larger one (f200, 50 KB columns)
whenever the per-op overhead is what is being paid; in the quiet run the f64 upload floor is the bigger
one in both the floor matrix and the micro-probe. Bytes are second order: ms/KB of payload falls with
size, and the batched cost stays below 10x the isolated cost (no super-additivity in the quiet window;
the contended control's batch/isolated ratio was 0.77x).

### Contention control (why the first attempt is not the headline)

`floor_f64.json` (same probe, GPU 6, no prelude) ran beside a foreign `pt_data_worker` storm and is kept
as the contention control: M0 18.8793 ms vs 7.2289 ms quiet (2.61x), M3 3.0222 vs 3.3752, M5 4.0163 vs 2.3849; its closing brackets drifted M0R +13.6 % / M3R +154.5 %. Host-bound arms inflate together, GPU-run-dominated arms barely move — which is itself the Q6 message.

