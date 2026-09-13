# R5 Q1/Q5/Q7/Q8 — per-call decomposition of one warm dense-route `evaluate`

Phase-2 record of the R5 RCA round. **Analysis-only**: no `src/evogp` edit, no write in
`/mnt/local-ssd/bchuang/etl`, no edit outside `test/benchmark/r5/`; shared tooling
(`../bench_gp.py`, `../pure_forward.py`, `../stage_probe.py`, `../r2/**`, `../r4/**`) was read only.

## What was measured

The harness call, replicated exactly (`test/benchmark/bench_gp.py:768-806` builds
`Forest(...)` + `SymbolicRegression(datapoints=x, labels=labels)` from the fixture;
`bench_gp.py:1374-1381` times **one** `problem.evaluate(forest)` between one
`torch.cuda.synchronize()` pair, call 0 cold, calls 1..24 warm). The resolved route is
`SymbolicRegression.evaluate` → `Forest.SR_fitness` → `dense_engine.sr_fitness_dense`
(`forest.py:338-349`, `dense_engine.py:1924`), i.e. the in-graph **loss variant**, one
`(63, P)` payload set per call, `etl.run` once.

Environment (both cells): project venv, `PYTHONPATH=src`, `ETL_DISABLE_LOCATIONS=1`,
`CUDA_VISIBLE_DEVICES=6`, `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0
EVOGP_ETL_IREE_OPT_LEVEL=O1`, ETL = `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9`
(stamped in every record). One process per record, each inside exactly one
`flock -w 7200 /tmp/gpbench_critical.lock` with `timeout -s INT -k 30 900` **inside** the lock,
`ulimit -n 65536`.

## Component tables (exclusive self time, warm calls 1..24)

`decomp_f64` — warm median call **7.3378 ms** (mean 8.0228, min 6.8348, max 12.7947; first call
12017.370 ms = the 11.93 s cold build). Probe revision: evogp `1a14b77`.

| component | count | median ms | % total | what it is |
|---|---|---|---|---|
| `etl_tensor_to_payload` | 10 | **3.5084** | 47.8 % | `etl.from_numpy(host).to(device)` per payload, caller = `payload` |
| `etl_run` | 1 | **2.1267** | 29.0 % | the single compiled dense graph invocation |
| `prepare_dense_self` | 1 | 0.6958 | 9.5 % | decode + structural validation + `np.unique` of the function ids |
| `run_loss_plan_self` | 1 | 0.5120 | 7.0 % | host assembly: reshape/accumulate per chunk, divide by N |
| `as_numpy_d2h` | 5 | 0.1437 | 2.0 % | `tensor.detach().cpu().numpy()`: x, labels, node_value, node_type, subtree_size |
| `etl_tensor_to_to_host` | 1 | 0.1122 | 1.5 % | the loss readback `result.to(cpu)` inside `_to_host` |
| `sr_fitness_dense_self` | 1 | 0.1115 | 1.5 % | `resolve_backend` + `torch.from_numpy(...).to(cuda)` (the fitness H2D) |
| `build_dense_build_cached` | 1 | 0.0792 | 1.1 % | `etl.build_cached` warm-cache prologue (Q7) |
| `plan_dense_self` | 1 | 0.0771 | 1.1 % | chunk shape + specs/cfg construction |
| `etl_from_numpy` | 10 | 0.0366 | 0.5 % | the zero-copy host wrap of each payload |
| `prep_column` | 7 | 0.0344 | 0.5 % | `_prep_column` (host column transpose/pad) |
| `outer_evaluate` + `forest_sr_self` + `policy_should_use_dense` | 1 each | 0.0721 | 1.0 % | `isinstance`+negation / `check_tensor`+asserts / the route policy |
| `to_host_self` + `input_to_numpy_self` | 1 each | 0.0155 | 0.2 % | the wrappers around the two copies above |
| **SUM(exclusive)** | | **7.5252** | **102.55 %** | sum of medians vs median of sums |

`decomp_f200` — warm median call **6.6167 ms** (mean 7.1137, min 6.3238, max 9.5407; first call
12613.683 ms = the 12.55 s cold build).

| component | count | median ms | % total |
|---|---|---|---|
| `etl_run` | 1 | **2.3673** | 35.8 % |
| `etl_tensor_to_payload` | 10 | **1.5300** | 23.1 % |
| `prepare_dense_self` | 1 | **1.3830** | 20.9 % |
| `run_loss_plan_self` | 1 | 0.5134 | 7.8 % |
| `as_numpy_d2h` | 5 | 0.1683 | 2.5 % |
| `etl_tensor_to_to_host` | 1 | 0.1165 | 1.8 % |
| `sr_fitness_dense_self` | 1 | 0.1146 | 1.7 % |
| `build_dense_build_cached` | 1 | 0.0767 | 1.2 % |
| `plan_dense_self` | 1 | 0.0782 | 1.2 % |
| `etl_from_numpy` + `prep_column` | 10 / 7 | 0.0713 | 1.1 % |
| outer/forest/policy + `to_host_self`/`input_to_numpy_self` | 1 each | 0.0905 | 1.4 % |
| **SUM(exclusive)** | | **6.5106** | **98.40 %** |

### Reconciliation (the wrap set is complete)

Exclusive accounting makes the sum of every bucket the root span's wall by construction; the
per-call check (`sum(exclusive) / call wall`) is **99.94 % median for f200 (99.89–99.96 %) and
99.95 % for f64 (99.92–99.96 %)**. The sum-of-*medians* exceeds the median-of-sums on f64
(102.55 %) only because a few stalled calls (11–13 ms) inflate several components' medians
independently — the per-call reconciliation is exact. No `depth_overflow`, no wrap failures
(`missing: none`), and the fitness tensor is byte-identical across all 29 measured calls
(`all_equal: true`).

### GPU-tail placement (region-sync control, 5 calls)

A `torch.cuda.synchronize()` inserted *inside* every span after 5 warm calls fences:
`fence_after:etl_run` **0.031 ms**, `prep_column` 0.148 ms, `as_numpy_d2h` 0.064 ms, everything
else ≤0.024 ms; post-call fence 0.010 ms, and the uninstrumented control's post-sync is
0.025 ms. So the compiled forward **is drained inside `etl.run`** (it waits): nothing of the
graph's GPU tail lands after the call boundary, and the 0.15 ms `prep_column` fence is the
deferred drain of the *previous* payload's HAL copy. The uploads' own device copy is likewise
absorbed before `etl.run` returns. No fence wait is double counted: every fence is booked in
its own `fence_after:*` bucket and subtracted from the enclosing span.

### Probe overhead (uninstrumented control, 5 calls)

Wrappers inert, same 25-call shape, no `begin_call`: f64 **7.2206 ms** median vs 7.3378 ms
instrumented (+1.6 %), f200 6.95→6.62 band ordering inside the drift. The instrumentation is
therefore not the cost.

## Q5 — top-3 host costs

| rank | f64 | ms | f200 | ms |
|---|---|---|---|---|
| 1 | 10 payload uploads (`from_numpy().to(device)`) | **3.51** | `etl.run` — *not host work*: the graph call itself | 2.37 |
| 1 (host) | | | 10 payload uploads | **1.53** |
| 2 | decode (`prepare_dense`) | **0.70** | decode (`prepare_dense`) | **1.38** |
| 3 | host assembly in `run_loss_plan` | **0.51** | host assembly in `run_loss_plan` | **0.51** |
| — | readback (`_to_host` + D2H) | 0.12 | readback | 0.12 |

**Host-side top 3 (the answer to Q5): payload uploads, then the dense decode, then the
per-chunk loss assembly.** The readback is *not* in the top 3 (0.12 ms): the loss variant reads
back only `(pop_size,)` floats.

Two observations this table forces:

1. **Uploads are the single largest host cost, and their cost is not bytes-driven.** The two
   cells upload the same 10 payloads per call, f200 carrying 3.1× the bytes of f64
   (50 KB vs 16 KB per node column) yet measuring **1.53 ms vs 3.51 ms** — while their totals
   differ by only 10 % (6.62 vs 7.34 ms) and every other component scales monotonically with
   the population. The f64 upload time is also the least stable (min 3.09 / median 3.51 /
   mean 3.76 / max 5.80 ms). This is *measured*, not explained: the plausible mechanisms
   (IREE HAL allocator pool behaviour per size class, queue depth at allocation time,
   host-side staging buffer churn) are untested here and belong to the Q6 floor matrix.
2. **`etl.run` is ~2.1–2.4 ms in-process, i.e. larger than the Q4 pure forward
   (1.2142 / 1.4612 ms).** The frozen pure runner stages its payloads *once* outside the timed
   loop and re-runs the same buffers, whereas the e2e path hands the graph *freshly created*
   HAL buffers every call; the delta (≈0.7–0.9 ms) is the per-call run-path validation/re-plan
   the root CONTEXT's residual-gap note attributes to upstream ETL. Cross-process caveat: the
   pure numbers are Q4 records, the e2e `etl_run` is this cell's; the in-process ratio
   `etl_run / total` is 29.0 % (f64) / 35.8 % (f200) against Q4's `pure / e2e` 22.5 % / 24.1 %.

## Q7 — no compile churn in the warm call

Method: `dense_engine.build_dense` wrapped (it is the only `etl.build_cached` call site on this
route) plus the shared `etl.CompileCache` (`dense_engine.DENSE_CACHE is etl_engine.CACHE`)
hit/miss/size counters read per call.

* warm `build_cached` wall: **0.0792 ms median (f64) / 0.0767 ms (f200)**, range 0.054–0.264 ms
  over the 24 warm calls — a cache lookup plus `build_dense`'s own backend resolution, no
  ms-scale rebuild.
* cache counters across the loop: **hits 1..24 with `misses=1`** (the miss is call 0;
  `size=1`, one loss-variant graph for the whole cell) — the per-call `hits`/`misses` curve is
  the `cache hits=` column of every committed log line.
* cold contrast (deliberate `DENSE_CACHE.clear()` + one call, same process): **10840.713 ms
  (f64) / 11283.765 ms (f200)**, of which the build bucket is 10829.649 ms / 99.90 % (f64) —
  the same call is 7.3 ms warm. This brackets the compile tax at ≈10.8–11.3 s and confirms the
  warm call contains none of it.
* Record caveat: `cache_final` in the JSONs is read *after* the cold control (so it shows
  `hits=0, misses=1, size=1`); the warm-loop counters are in the log's per-call `cache hits/misses`
  columns and in `warm.per_call_components.build_dense_build_cached`.

## Q8 — harness-artifact falsification (partially executed)

Cited from the harness: the timed region is strictly `problem.evaluate(forest)`
(`bench_gp.py:1378-1381`, with the `torch.cuda.synchronize()` pair at 1377/1380), and the
fixture→problem construction happens **once per rep outside** that loop
(`build_eval_case` at 1363, loop at 1374) — with `--reps 1` the fixture is loaded once per
process, never per call.

`minimal_loop.py` (committed) is the control: fixture → `Forest`+`SymbolicRegression` →
`problem.evaluate(forest)` in a plain loop, no harness import, no probe, no monkeypatch, same
one-sync-pair-per-call shape, ≥25 calls, per-call ms printed and JSON-recorded.

**Executed** (`results/q8_minimal/`, one locked process per fixture, 2026-09-13 20:58 HKT): warm
median **6.117 ms (f64, min 5.651 / max 9.310)** and **6.010 ms (f200, min 5.410 / max 11.848)**,
first call 12.41 s / 12.42 s (the cold compile). Both medians sit **inside** the Q4 same-window
band (5.3823 / 5.7912 ms, control re-run +26.9 %) and **below** this decomposition's own in-process
totals (7.34 / 6.62 ms), which is the same ordering the probe's uninstrumented control showed
(7.22 / 6.95 ms). **The harness-artifact hypothesis is therefore falsified**: with no
`bench_gp.py`, no probe and no monkeypatch, the same `problem.evaluate(forest)` route costs the
same ~5.4–6.9 ms; the cost is inside `problem.evaluate` itself (see the update section below).

## Exact commands (each its own locked invocation, repo root, project venv)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
R5=test/benchmark/r5

# Q1/Q5/Q7, written by probe revision evogp 1a14b77:
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' '"$R5"'/decomp_probe.py --fixture test/goldens/forest_random.npz --eval-reps 25 --sync-reps 5 --plain-reps 5 --cold-reps 1 --tag decomp_f64 --json-out '"$R5"'/results/q1_decomp/decomp_f64.json --log-out '"$R5"'/results/q1_decomp/decomp_f64.log'
# ... same with --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag decomp_f200 (done)

# Phase-2 leftovers, all three run 2026-09-13 20:57-20:58 (same lock/env pattern as above):
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' '"$R5"'/decomp_probe.py --fixture test/goldens/forest_random.npz --cprofile --eval-reps 25 --json-out '"$R5"'/results/q1_decomp/cprofile_f64.json --cprofile-out '"$R5"'/results/q1_decomp/cprofile_f64.txt'
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' '"$R5"'/minimal_loop.py --fixture test/goldens/forest_random.npz --reps 25 --json-out '"$R5"'/results/q8_minimal/minimal_f64.json --log-out '"$R5"'/results/q8_minimal/minimal_f64.log'
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' '"$R5"'/minimal_loop.py --fixture test/benchmark/fixtures/forest200_tree64_random.npz --reps 25 --json-out '"$R5"'/results/q8_minimal/minimal_f200.json --log-out '"$R5"'/results/q8_minimal/minimal_f200.log'
```

## Records

| path | what |
|---|---|
| `results/q1_decomp/decomp_f64.json` / `.log` | 25-call warm decomposition, controls, per-call arrays + counts, plan, stamps |
| `results/q1_decomp/decomp_f200.json` / `.log` | same, 200-tree fixture |
| `probe_lib.py` | reusable helpers (exclusive `Profiler`, `patch_attr`, `EtlProxy`, `patch_etl_tensor_to`, `install_dense_wraps`, `build_eval_case`, `identity_stamp`) for the Q6/Q3 probes |
| `decomp_probe.py` | the Q1/Q5/Q7 probe (`--cprofile` mode included) |
| `minimal_loop.py` | the Q8 control (no harness, no probe) |
| `results/q1_decomp/cprofile_f64.json` / `cprofile_f64.txt` | the cProfile cross-check (25 warm calls, instrumentation OFF) |
| `results/q8_minimal/minimal_f64.json` / `.log`, `minimal_f200.json` / `.log` | the Q8 minimal-loop control, both fixtures (plus `results/q8_minimal/SUMMARY.md`) |

## Deviations / limits

* Absolute numbers live in the 6.3–12.8 ms per-call spread the round already knows (co-tenant
  drift on the shared box); every comparison above is made **inside one process** against that
  process's own total, never across processes.
* The `plain` control came after the `sync_region` control (order fixed by the probe), so its
  small offset is not a clean A/B.
* Uploads are timed by patching `etl.core.Tensor.to` and bucketing by caller function name; the
  histogram (`external:etl_tensor_to_payload: {'payload': N}`) proves every booked call came from
  `_iter_plan_chunks.payload` (10/call) and the readback from `_to_host` (1/call), so no other
  `.to` is mixed in.
* `--stage-probe`/nsys/ncu are not used; the probe is the only instrumentation and it never
  edits `src/evogp`.
* No `src/evogp` or `etl` change was needed anywhere in this phase (no NEEDS-PARENT-ACTION).
* Cleanup: all probe processes exited 0 (IREE's `nanobind: leaked ...` chatter at exit is the
  known stock-ETL teardown noise, not a failure); no GPU process was left behind by this phase.

---

# Update — Q8 + cProfile completed (2026-09-13 20:57–20:58 HKT, GPU 6)

The two Phase-2 leftovers committed but not run in the phase above are now recorded. Both ran as
their own locked invocation (one process each, `flock -w 7200 /tmp/gpbench_critical.lock` with
`timeout -s INT -k 30 900` inside, `ulimit -n 65536`, `CUDA_VISIBLE_DEVICES=6`,
`ETL_DISABLE_LOCATIONS=1`, `PYTHONPATH=src`, repo-root cwd, project venv, the same
`EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`
route) against ETL `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9` (stamped
in both records); evogp `48dd1bc` (r5 files only). Command lines are in the "Exact commands" block
above; a short record-level write-up is `results/q8_minimal/SUMMARY.md`.

## Q8 minimal loop — verdict: the harness-artifact hypothesis is **falsified**

| cell | warm median | min / max | first call (cold compile) | vs Q4 same-window band | vs this decomp's in-process total |
|---|---|---|---|---|---|
| `minimal_f64` (64 trees) | **6.117 ms** | 5.651 / 9.310 | 12411.901 ms | inside 5.3823 (window) – 6.9 (bracket) | below 7.3378 ms |
| `minimal_f200` (200 trees) | **6.010 ms** | 5.410 / 11.848 | 12422.796 ms | inside 5.7912 – 6.9 | below 6.6167 ms |

No `bench_gp.py`, no probe, no monkeypatch, the same `Forest`+`SymbolicRegression` wiring, and the
same one-`torch.cuda.synchronize()`-pair-per-call shape: the route costs the same ~6 ms, i.e. the
whole cost is inside `problem.evaluate(forest)`. The marginal 4–10 ms outliers in the tails are the
shared box's co-tenant drift (the Q4 window saw the same shape), not an artifact of harness bookkeeping.

## cProfile cross-check (`cprofile_f64.txt`) — ranking only

Zero instrumentation, 25 warm calls, one cold call outside the profile:
`48201 function calls in 0.190 s` ⇒ **7.61 ms/call**, i.e. the same figure the probe's own total
reported in that hour (7.34 ms); cProfile's ~4 % inflation is well inside the drift, but cProfile is
**ranking-only** — every absolute here is distorted by the profiler's per-call hook, and it cannot
see GPU time at all.

Top-5 by **tottime** (self time) and the matching cumulative figures:

| rank | function | ncalls | tottime | cumtime | per call |
|---|---|---|---|---|---|
| 1 | `etl/backends/adapters/iree.py:1655(_wait_copy_semaphore)` | 175 (7/call) | 0.046 s | 0.046 s | 1.84 ms |
| 2 | `iree/runtime/function.py:146(_invoke)` | 25 (1/call) | 0.045 s | 0.045 s | 1.80 ms |
| 3 | `etl/backends/adapters/iree.py:1679(_staged_input)` | 250 (10/call) | 0.020 s | 0.072 s | 2.88 ms cum |
| 4 | `dense_engine.py:932(_decode_columns)` | 25 | 0.008 s | 0.016 s | 0.64 ms cum |
| 5 | `dense_engine.py:1924(sr_fitness_dense)` | 25 | 0.006 s | 0.185 s | 7.40 ms cum (the root) |

Agreement with the timer split (same call sites, same hour):

| path | cProfile cum / 25 calls | timer bucket (medians) | verdict |
|---|---|---|---|
| payload build+upload (`dense_engine:1734(payload)` = `etl.core.tensor.py:364(to)` = `iree.py:2377(upload_tensor)`) | 0.086 s ⇒ **3.44 ms** | `etl_tensor_to_payload` **3.51 ms** | agrees (upload = #1 host cost) |
| `iree.py:1531(run)` (inside `etl.pipeline:1678(run)`) | 0.049 s ⇒ **1.96 ms** | `etl_run` **2.13 ms** | agrees (#2) |
| `_decode_columns` (inside `prepare_dense`) | 0.016 s ⇒ **0.64 ms** | `prepare_dense_self` **0.70 ms** | agrees (#3) |
| loss readback (`array_interop.py:153(_copy_to_host)`) | 0.003 s ⇒ 0.12 ms | `etl_tensor_to_to_host` **0.11 ms** | agrees (not top-3) |

The one thing cProfile adds beyond the timer split is the *shape* of the two large buckets: the
upload bucket is not a memory copy — it is `_staged_input` (10 calls/call) blocking in
`_wait_copy_semaphore` (7 waits/call, **1.84 ms/call**, the single largest tottime in the process),
and the `etl.run` bucket is dominated by the raw `iree.runtime.function.__call__/_invoke`
(**1.80 ms/call**) rather than by any ETL-side Python bookkeeping. Both facts are upstream-ETL
plumbing, i.e. they are the host-side price of staging/launching the compiled dense graph, and they
are the Q2 GPU-mechanism question's host mirror.
