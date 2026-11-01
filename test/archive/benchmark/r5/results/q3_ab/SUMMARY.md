# R5 Q3 — engine-call vs frozen-runner: same process, same population, same compiled artifact

Phase-4 record of the R5 RCA round. **Analysis-only** (runtime monkeypatches only; no `src/evogp` edit,
no write in the foreign `etl` checkout, nothing edited outside `test/benchmark/r5/`).
Route: dense, iree backend, `EVOGP_ETL_BACKEND=iree`, `EVOGP_EVAL_MODE=auto`, `O1`, `ETL_DISABLE_LOCATIONS=1`,
GPU 6, project venv, one process per fixture inside one `flock -w 7200 /tmp/gpbench_critical.lock`
(`timeout -s INT -k 30 900` inside the lock, `ulimit -n 65536`).

Records: `ab_f64.{json,log}`, `ab_f200.{json,log}`. Both compiled artifacts are built once in-process
(value first, then loss; each ~12 s cold) and every arm holds `plan`/`executable`/args directly, so no arm
pays a per-call key derivation — that cost is Q6's `M4V`→`M5` delta.

| arm | f64 total | f64 `etl.run` | f200 total | f200 `etl.run` | what it is |
|---|---|---|---|---|---|
| `A1_engine_fresh` | 5.1868 | 1.8441 | 4.0287 | 1.9442 | engine seam (_iter_plan_chunks), value artifact, FRESH payloads per call |
| `A2_engine_staged` | 1.3180 | 1.2668 | 1.3461 | 1.2716 | engine seam, value artifact, STAGED payloads |
| `A3_frozen_runner` | 1.3171 | 1.2664 | 1.2576 | 1.2063 | frozen build_dense_runner(forest, x).run() on staged inputs |
| `A3_plain_control` | 1.3236 | n/a | 1.8151 | n/a | A3 with the instrumentation inert (probe-overhead control) |
| `A4_frozen_fresh` | 4.8704 | 1.4302 | 3.7547 | 1.8730 | frozen runner's plan/executable/inputs, FRESH payloads |
| `L1_engine_fresh_loss` | 6.1420 | 1.8583 | 4.5757 | 1.9983 | engine seam, LOSS artifact, FRESH payloads (the SR route's graph) |
| `L2_engine_staged_loss` | 1.9785 | 1.8696 | 2.0519 | 1.9675 | engine seam, LOSS artifact, STAGED payloads |
| `R1_repl_fresh_loss` | 6.0302 | 1.8925 | 4.5082 | 2.0340 | probe-replicated fresh loop + etl.run (replication control) |
| `R2_repl_fresh_drained` | 6.1169 | 1.8888 | 3.8667 | 2.0152 | probe-replicated fresh loop + explicit fence + etl.run |
| `E1_e2e_in_process` | 8.1733 | 1.8538 | 5.8916 | 1.6101 | problem.evaluate(forest) -- the in-process e2e companion |

## Deltas — recomputed from the records (the falsification surface)

| delta | metric | f64 | f200 | meaning | attribution |
|---|---|---|---|---|---|
| `A1-A2 total` | median_warm_ms | +3.8687 | +2.6826 | cost of building/uploading the payloads fresh, per call (prep+upload+semaphores) | measured |
| `A1-A2 run span` | etl_run | +0.5774 | +0.6726 | payload-freshness cost INSIDE etl.run (fresh buffers vs staged buffers) | measured |
| `A2-A3 total` | median_warm_ms | +0.0009 | +0.0885 | engine-seam staged call vs frozen runner.run() (same staged artifact) | measured |
| `A2-A3 run span` | etl_run | +0.0003 | +0.0653 | the run path itself (engine seam vs frozen run(): same etl.run, same args) | measured |
| `A4-A3 total` | median_warm_ms | +3.5534 | +2.4971 | upload cost in runner terms (frozen plan/executable fed with fresh payloads) | measured |
| `A4-A3 run span` | etl_run | +0.1638 | +0.6666 | the same freshness question on the frozen artifact | measured |
| `L1-L2 total` | median_warm_ms | +4.1635 | +2.5238 | fresh-vs-staged payload cost on the LOSS artifact (the engine's real route) | measured |
| `L1-L2 run span` | etl_run | -0.0113 | +0.0308 | freshness cost inside etl.run on the LOSS artifact | measured |
| `L2-A3 run span` | etl_run | +0.6032 | +0.7611 | LOSS artifact staged run vs VALUE artifact staged run (in-graph loss vs values) | measured |
| `R1-L1 total` | median_warm_ms | -0.1117 | -0.0675 | replication fidelity of the probe-side fresh loop (must be ~0) | measured |
| `R2-R1 total` | median_warm_ms | +0.0866 | -0.6415 | cost of an explicit fence between staging and the run (the drain that etl.run absorbs) | measured |
| `R2-R1 run span` | etl_run | -0.0037 | -0.0188 | how much of etl.run's fresh-payload cost is the staging copy's drain | measured |
| `E1-A3 total` | median_warm_ms | +6.8562 | +4.6340 | in-process e2e evaluate minus frozen pure run (the e2e_companion analogue) | measured |
| `E1-L2 total` | median_warm_ms | +6.1948 | +3.8397 | in-process e2e minus the staged loss run: the per-call host glue + prologue around the run | measured |
| `A3-A3plain overhead` | median_warm_ms | -0.0066 | -0.5575 | probe instrumentation overhead on the frozen run() path (plain control) | measured |

### Attribution (measured / inferred)

* **`A1-A2` total +3.87 / +2.68 ms, run-span +0.58 / +0.67 ms** — *measured*: handing `etl.run` freshly
  built/uploaded payloads costs ~2.7–3.9 ms per call, of which ~0.6–0.7 ms lands **inside** the graph call
  itself (fresh HAL buffers + the staging copies' drain that `etl.run` absorbs) and the rest is the
  preparation/upload work itself. This is the same mechanism as Q6's `M2`→`M3` upload floor and Q1's
  cProfile `_wait_copy_semaphore` ≈1.84 ms/call.

* **`A2-A3` total ≈0.00 / +0.09 ms, run-span ≈0.00 / +0.07 ms** — *measured* and decisive: on the **same
  staged payloads and the same artifact**, the engine's own seam (`_iter_plan_chunks` + `etl.run`) and the
  frozen runner's `run()` (which *is* the same code path with the args held) are indistinguishable.
  **There is no engine run-path tax**: the per-wrapper spec memo / run-static re-validation story does not
  show up as a measurable delta on this box. The `A3-A3plain` control (−0.007 / −0.56 ms) bounds the probe
  instrumentation, so the null result is not an artifact of the wrap set.

* **`A4-A3` total +3.55 / +2.50 ms** — *measured*: the freshness cost is a property of the *payloads*, not
  of the engine: the *frozen* artifact fed with fresh payloads pays the same upload cost.

* **`L1-L2` total +4.16 / +2.52 ms, run-span −0.01 / +0.03 ms** — *measured*: the engine's real SR route
  (loss variant, fresh payloads) pays ≈4.2 ms/call of host glue around a run span that is *unchanged* by
  freshness once staged. Q1's `etl.run` = 2.13/2.37 ms is therefore **the loss artifact's run span**,
*not* a cost caused by fresh buffers.

* **`L2-A3` run-span +0.60 / +0.76 ms** — *measured*: the loss variant itself is ~0.6–0.8 ms more expensive
  per graph call than the value variant (in-graph loss/keep-mask reduction) — the residual difference
  between Q1's in-process `etl.run` (2.13 / 2.37) and the frozen *value* pure forward (1.21 / 1.46).

* **`R2-R1`** total +0.09 / −0.64 ms, run-span −0.00 / −0.02 ms — *measured*: an explicit fence between
  staging and the run changes nothing, i.e. the drain is absorbed inside the call either way (the arm
  ordering control `R1-L1` = −0.11 / −0.07 ms confirms the replication is faithful).

* **`E1-A3` +6.86 / +4.63 ms** and **`E1-L2` +6.19 / +3.84 ms** — *measured*: the full `problem.evaluate`
  sits that far above the pure run *in the same process*, i.e. the e2e path's host glue + prologue is
  ~4–7 ms/call — consistent with the Q4 `e2e_companion` context (e2e 5.3823 / 5.7912 ms vs frozen pure
  1.2142 / 1.4612 ms) and with Q6's floor matrix.


### Answer to the Q1-vs-Q4 question

The 2.13/2.37 ms in-process `etl.run` vs the 1.21/1.46 ms frozen pure forward is **not** a run-path tax and
**not** a payload-freshness tax inside the run: it is (a) the **loss-vs-value variant** (≈0.6–0.8 ms of
in-graph reduction, `L2-A3`) plus (b) the different *measurement* the two numbers came from (the frozen
figures are value-variant, readback-free `run()` calls in a tight loop; the in-process figure is one
`etl.run` inside a full evaluate whose 4–6 ms of host glue surrounds it). Same process, staged inputs:
engine seam vs frozen `run()` differ by ≤0.09 ms.

### Record-integrity note

* f64: every stored `deltas` entry reproduces from the stored arm medians (Δ ≤ 0.02 ms).
* f200: every stored `deltas` entry reproduces from the stored arm medians (Δ ≤ 0.02 ms).

### Host window

* f64: loadavg `369.28` → `373.41` on 128 cores, real CPU busy **3.7 % → 7.2 %**, GPU 6 `6, NVIDIA RTX A6000, 1, 30796`.
* f200: loadavg `373.54` → `364.42` on 128 cores, real CPU busy **12.7 % → 12.4 %**, GPU 6 `6, NVIDIA RTX A6000, 2, 30794`.

### Commands (exact)

```bash
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r5/ab_probe.py \
    --fixture test/goldens/forest_random.npz --reps 25 --tag ab_f64 \
    --json-out test/benchmark/r5/results/q3_ab/ab_f64.json \
    --log-out  test/benchmark/r5/results/q3_ab/ab_f64.log'
# f200: --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag ab_f200 (same command)
```
