# `results/q2_gpu/SUMMARY.md` — R5 Q2: the GPU-side mechanism of the dense pure forward

Phase 3 of the R5 RCA round (`test/benchmark/r5/`): Q2 cells executed 2026-09-13 **21:01–21:15 HKT**
(the Q8/cProfile leftovers were the 20:57–20:58 cells recorded in `../q1_decomp/SUMMARY.md`) on
GPU 6 of the shared `bmi-9` box. **Analysis-only**: no `src/evogp` edit, no write in
`/mnt/local-ssd/bchuang/etl` (read-only), no edit outside `r5/`; `../bench_gp.py`, `../pure_forward.py`
and `../stage_probe.py` were read, never modified.

**Question Q2 answers.** The dense-iree **pure** forward of the 64/200-tree fixture costs
1.2142 / 1.4612 ms warm (Q4 same-window) — **1.9–2.3× the legacy CUDA engine's ENTIRE
`problem.evaluate`** (0.6461 / 0.6372 ms). What is the compiled dense graph actually doing on the GPU,
and does the arithmetic explain that ratio?

**Answer in one line.** The dense graph is a **~250-dispatch serial chain** (249 dispatches + 61
device-to-device copies for the 64-tree fixture; 253 + 62 for the 200-tree one) of **1.4–1.7 µs
elementwise kernels** launched one after another; the legacy engine does the same population
evaluation in **one fused kernel + one averaging kernel (+1 memset)**. The new/legacy ratio is
therefore a **launch/serialization** ratio, not an arithmetic one: 310 device ops per call vs 3.

## 0. Session identity (what was actually measured)

| item | value (evidence) |
|---|---|
| evogp source | commit `48dd1bc` + r5 files only (`src/evogp` unmodified — `../env_probe.txt`); records were stamped at `247c1e0`/`48dd1bc`, all in this worktree |
| ETL (primary) | editable install of the foreign checkout `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9`, working tree CLEAN (`etl_repo_status: <empty>` in every record) |
| route | `EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`; dumps: `EVOGP_EVAL_MODE=auto` (resolves to the dense evaluator, `should_use_dense=True`); row series: `EVOGP_EVAL_MODE=dense` |
| fixtures | `test/goldens/forest_random.npz` (64 trees, 64 rows), `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, 64 rows) |
| env of every cell | `CUDA_VISIBLE_DEVICES=6`, `ETL_DISABLE_LOCATIONS=1`, repo-root cwd, project venv (py3.11.2, torch 2.6.0+cu124), `ulimit -n 65536`, **exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per process with `timeout -s INT -k 30 <cap>` INSIDE the lock**, one fresh process per cell |
| tooling | `iree-compile` 3.11.0 (the venv's `iree-base-compiler`) for flag validation; `torch.profiler` (kineto/CUPTI) for the timeline; nsys/ncu remain blocked (see §D) |
| co-tenants | GPU 6: 3 foreign training processes holding ~30.5 GB (`nvidia-smi` samples inside every record: `used_mb_start` 30 561–30 575 MiB, `free_mb_start` 17 838–17 852 MiB, util 0–22 %); the whole session ran in the box's normal ±10–20 % noise band except the one cell flagged in §C |
| timestamps / stamps | every record carries `stamp.when`/`started_at` (ISO, +0800), `evogp_commit`, `etl_file` + `etl_repo_head` (+ clean status), `env_knobs`, `cuda_visible_devices`; the row cells additionally carry `gpu.{used_mb_start,free_mb_start,used_mb_end}` and the compile/construct split |

## A. The compiled dense graph (dispatch dumps)

Method: `gpu_probe/dispatch_dump.py` sets `ETL_IREE_COMPILE_ARGS` (→ the iree compile stage's
`iree_compile_args`: `etl/pipeline_options.py:123`, `etl/backends/adapters/iree.py:345/953/1018`) to
`--iree-flow-dump-dispatch-graph --iree-flow-dump-dispatch-graph-output-file=… --iree-scheduling-dump-statistics-file=…-format=csv --iree-hal-dump-executable-binaries-to=…`,
then makes ONE fresh first call (which compiles) and parses the dumps. Raw dumps live under
`results/q2_gpu/dumps/<cell>/` (`dispatch_graph.dot`, `scheduling_stats.csv`, `executables/`).

| dump cell | mode | rows | dispatches | longest dispatch chain | graph nodes/edges | linked PTX kernels | DtoD copies | transient | compile | warm single-call ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `f64_pure` | pure | 64 | **249** | **125** | 578 / 1500 | 125 | 61 | 2.88 MB | 12.15 s | 2.225 / 2.230 / 2.209 |
| `f64_e2e` | e2e | 64 | **249** | **125** | 584 / 1506 | 125 | 61 | 2.86 MB | 12.03 s | 8.465 / 6.507 / 6.335 |
| `f200_pure` | pure | 64 | **253** | **127** | 587 / 1524 | 127 | 62 | 8.98 MB | 11.95 s | 2.817 / 3.180 / 3.184 |
| `f64_r8` | pure | 8 | 249 | 125 | 578 / 1500 | 125 | 61 | 0.36 MB | 12.70 s | 3.00 / 3.08 / 3.04 |
| `f64_r256` | pure | 256 | 249 | 125 | 578 / 1500 | 125 | 61 | 11.50 MB | 12.20 s | 2.70 / 2.66 / 2.65 |
| `f200_r128` | pure | 128 | 253 | 127 | 587 / 1524 | 127 | 62 | 17.97 MB | 12.50 s | 3.17 / 3.16 / 3.17 |

Facts from the dumps (measured):

* **One dispatch per elementwise step of the trace.** 249 dispatches for a 63-node trace
  (`plan.trace_len=63`, pop 64) — i.e. ~4 dispatches per tree node once the constant loads,
  broadcasts and selects are counted; 253 for the 200-tree fixture whose trace is 2 nodes longer. The
  count is a property of the *trace*, not of the population: it is identical at 8 and 256 rows
  (measured).
* **The dispatches are chained, not fanned out**: the longest path over the dispatch graph is **125**
  (127) dispatch nodes — i.e. the op dependency chain is essentially the whole trace, and each step
  waits for its predecessor.
* **Small kernels**: the linked PTX holds 125/127 kernels (`main_dispatch_*_elementwise_*`), i.e.
  one kernel per distinct op/shape; 61/62 buffer→buffer copies ride inside the same command buffer;
  `Awaits: 2, Submissions: 1, Executables: 125` in the scheduling statistics.
* **The e2e and pure routes compile the same structure** (249 dispatches / 125 kernels / chain 125);
  only the transient footprint differs (2.88 vs 2.86 MB). So the Q1/Q5 "host glue" split of the e2e
  call (pure ≈ 1/4 of the call) is not a GPU-structure difference.
* **The row count does not change the structure at all** (same 249/253 dispatches, same chain, same
  kernel count) — only buffer sizes / transient memory (0.36 → 18.0 MB).

## B. Per-dispatch math and the mechanical explanation

### B.1 The arithmetic (measured inputs, arithmetic output)

| quantity | 64-tree fixture | 200-tree fixture | source |
|---|---|---|---|
| dispatches per pure forward | 249 | 253 | dump (measured) |
| DtoD copies per forward | 61 | 62 | dump (measured) |
| total device ops per forward | **310** | **315** | = above (measured) |
| Q4 same-window pure warm median | **1.2142 ms** | **1.4612 ms** | `../q4_window/*.json` (measured) |
| ⇒ per device op | **3.92 µs** | **4.64 µs** | arithmetic |
| ⇒ per dispatch (kernels only) | **4.88 µs** | **5.78 µs** | arithmetic |
| ⇒ per serial-chain step (125 / 127 deep) | **9.7 µs** | **11.5 µs** | arithmetic |
| legacy whole evaluate (the contrast) | **0.6461 ms** | **0.6372 ms** | `../q4_window/q4_leg{64,200}.json` (measured) |
| legacy device ops for that evaluate | 3 | 3 | counted from source (below) |
| pure / legacy | **1.88×** | **2.29×** | Q4 same-window (measured) |

Every dispatch therefore costs ≈ 4.9–5.8 µs of the call, against a kernel that kineto measures at
**1.4–1.65 µs** (§D) — the rest is the launch/serialization gap.

### B.2 The legacy engine's device-op list (counted from the read-only legacy source)

`/tmp/evogp_legacy_bench/src/evogp/cuda/forward.cu` (legacy `b613503`, read-only). Both R5 fixtures
have 64 data rows and `SymbolicRegression.execute_mode` defaults to `"auto"` → legacy `Forest.SR_fitness(..., execute_mode="auto")`
(`tree/forest.py:346-347` in the legacy checkout maps `"auto"` → `execute_code = 4`) → CUDA
`SR_fitness(kernel_type=4)` (`forward.cu:849-855`): since `dataPoints (64) <= getCudaCoreCount()`
(= SM count 84 × 128 = 10 752 for this sm_86 A6000, `forward.cu:16-77`), the elected path is
**`advanced_SR_fitness`** (`forward.cu:514-551`), whose entire population evaluation is:

1. one `cudaMemsetAsync(fitnesses, …, popSize)` (device op 1),
2. **one** `treeGPRegressionFitnessKernel<false,true>` launch, grid `(popSize, 1) × 64 threads`
   (`forward.cu:536`) — each block walks one whole tree in registers/shared memory for one data
   point and accumulates the squared error (the per-tree walk happens *inside* the kernel:
   `_treeGPEvalByStack`, `forward.cu:247-304`),
3. **one** `averageFitnessValueKernel<<<ceil(popSize/64), 64>>>` launch (`forward.cu:548`) dividing by
   `dataPoints`.

⇒ **3 device ops (2 kernel launches + 1 memset) for the whole 64×64 SR-fitness evaluation**, i.e.
**~103× fewer device ops than the dense route's 310**, and the arithmetic is done by 4096 threads inside
one kernel instead of 249 separate launches over a `(63, pop×rows)` payload. The Q4 legacy wall time
(0.6461/0.6372 ms) is quoted only as the contrast anchor — nsys is unavailable here, so that number
cannot be decomposed into launch vs execution time (`unknown`, see §D).

### B.3 The mechanism, with confidence labels

* **measured** — the dense route issues **310 device ops per call**: 249 kernel launches + 61 DtoD
  copies, chain depth 125, kernels 1.4–1.65 µs each, grid (32,1,1)×(32,1,1) = 1024 threads,
  "est. achieved occupancy 1 %", 0 shared memory (kineto + dump).
* **measured** — with 30 invocations traced, the device-busy total is **513.9 µs/invocation**
  (kernels 409.7 µs + memcpy 103.4 µs) against a profiled span of 2550.8 µs (20.2 % busy) with 309
  idle gaps, median gap **5.86 µs** (`kineto_f64/trace_facts.json`). Under the profiler the wall is
  inflated (unprofiled warm 1.21–1.47 ms), so the conservative unprofiled statement is: **device-busy
  is 35–42 % of the warm call and >half of the timeline is the launch/serialize gap**.
* **measured** — the host side issues 249 launches/call; the trace's `cuda_driver` events (the
  `cuLaunchKernel` calls) measure a median **6.45 µs** each *under kineto*, 51.16 ms / 30 invocations
  = 1.71 ms/call (inflated; the unprofiled budget is 4.9–5.8 µs per dispatch for launch + execution).
* **inferred** — the mechanism is *launch-and-serialize bound, not compute bound*: the measured
  kernel time is 1.4–1.65 µs for a 4096-element op on 1024 threads at 1 % occupancy, and the
  125-deep dependency chain forbids overlap, so the call time ≈ Σ(launch latency + kernel +
  inter-dispatch gap). An arithmetic bound agrees: the whole `(63, 4096)` payload is ~1.0 M elements
  ≈ 12 MB of read+write traffic, ~25 µs at the A6000's ~500 GB/s (assumption: memory-bound
  elementwise, 3 array touches) — a few percent of the 1.21 ms call, and ~40× below the 410 µs of
  measured kernel time. The row-scaling series (§C) supports the same reading: a 32× larger payload
  changes nothing on the 64-tree fixture.
* **inferred (unmeasurable here)** — why each launch+gap is ~5 µs rather than ~2–3 µs: the ETL run
  path's per-dispatch HAL plumbing (its `etl.run` re-validates the static cfg per call; the frozen
  pure runner stages its buffers once but still walks the command buffer), the tiny grids (1024
  threads = 0.4 blocks/SM, 1 % occupancy) and the driver's launch path under a co-tenant-loaded
  context. Splitting host-side launch cost from device-side gap needs a GPU-with-CUPTI-timeline-free
  measurement (nsys/ncu) or a repaired ETL async submission path — both outside this round.
* **unknown** — the legacy side's internal split (its 0.65 ms for 3 device ops cannot be broken down
  without nsys; the 215 µs/op implied by the naive division is a property of the whole call, most of
  it host-side dispatch of the torch custom op rather than 3 launch latencies).

## C. `--data-rows` scaling series (frozen r4 pure-dense-iree protocol)

Each cell = the exact r4 `pure64_dense_iree` invocation plus `--data-rows N --output-checksum`
(`gpu_probe/run_rows.sh`, one fresh process per cell; changing rows ⇒ a new compile, reported
separately as `compile.construct_total_ms` 11.6–16.5 s). Warm numbers are the harness'
`pure.median_per_invocation_ms` (5 blocks × 100 invocations, readback/host-assembly excluded).

| fixture | rows | elements per dispatch | warm median ms | 5 block medians | per-dispatch µs | construct s | transient MB |
|---|---|---|---|---|---|---|---|
| f64 (64 trees) | 8 | 512 | **1.382** (`f64_r8b`; the first cell read 3.923, see below) | 1.71 / 1.46 / 1.38 / 1.28 / 1.25 | 5.55 | 13.3 | 0.36 (dump) |
| f64 | 16 | 1 024 | 1.429 | 1.92 / 1.51 / 1.43 / 1.29 / 1.28 | 5.74 | 15.7 | — |
| f64 | 32 | 2 048 | 1.434 | 1.77 / 1.46 / 1.43 / 1.14 / 1.18 | 5.76 | 15.5 | — |
| f64 | 64 | 4 096 | 1.360 (`f64_r64b` 1.302) | 1.80 / 1.55 / 1.34 / 1.34 / 1.36 | 5.46 | 13.1 | 2.88 (dump) |
| f64 | 128 | 8 192 | 1.469 | 1.73 / 1.48 / 1.47 / 1.38 / 1.22 | 5.90 | 16.1 | — |
| f64 | 256 | 16 384 | 1.321 | 1.43 / 1.40 / 1.32 / 1.27 / 1.27 | 5.31 | 15.5 | 11.50 (dump) |
| f200 (200 trees) | 64 | 12 800 | 1.345 (`f200_r64b` 1.423) | 1.79 / 1.57 / 1.29 / 1.30 / 1.35 | 5.32 | 13.3 | 8.98 (dump) |
| f200 | 96 | 19 200 | 1.640 | 1.92 / 1.63 / 1.65 / 1.64 / 1.60 | 6.48 | 16.1 | — |
| f200 | 128 | 25 600 | 2.406 (`f200_r128b` 2.281) | 2.46 / 2.43 / 2.41 / 2.40 / 2.38 | 9.51 (8.02 rerun) | 16.5 | 17.97 (dump) |

Least-squares fits (warm ms = a·rows + b):

| series | a (µs/row) | b (ms) | R² | reading |
|---|---|---|---|---|
| f64, rows 8→256 (r8 = rerun, r64 = rerun) | **−0.24** | **1.410** | 0.12 | **flat**: a 32× payload range costs nothing |
| f200, rows 64→128 (primary cells) | **+14.63** | **0.351** | 0.96 | +0.94 ms over 64 rows |
| f200, rows 64→128 (rerun r64) | **+13.41** | **0.494** | 0.93 | same slope, the "intercept" is one cell's noise |

* The 64-tree series is **flat in rows** and flat in transient size (0.36 → 11.5 MB): strong evidence
  for the launch-bound reading of §B.
* The 200-tree series rises **linearly with rows** (13–15 µs/row over 64 rows). It is *not* a
  dispatch-structure effect (the `f200_r128` dump has exactly the same 253 dispatches / 127 kernels /
  chain 127 as `f200_r64`), and it is not explained by payload size alone (the 64-tree series is flat
  at a larger per-dispatch element count). **Open**: the two protocols disagree in magnitude — with
  one `torch.cuda.synchronize()` pair per call (`dispatch_dump.py`) the same f200 r128 route reads
  3.17 ms vs 2.82 ms at r64 (+12 %), while the tight-block protocol reads +70 %. A GPU profiler on
  the f200 shape (or an ETL-side async-submission fix) is what would close it; label: **measured
  jump, unexplained**.
* **Cell-level caveat (contention, not protocol):** the very first series cell `f64_r8` (21:03)
  measured 3.923 ms with erratic blocks 4.97/2.53/3.92/5.12/3.82 and a 38.8 s construct (vs ~13 s
  elsewhere) — a co-tenant burst. Its re-run 6 minutes later (`f64_r8b`) reads 1.382 ms with a normal
  construct time, which is the value used above. Both records are kept.

## D. Kineto attempt (the one allowed GPU-timeline attempt) and the nsys/ncu status

`gpu_probe/kineto_attempt.py` ran the frozen pure dense path for 30 invocations under
`torch.profiler(activities=[CPU, CUDA])`, compile outside the profile; `gpu_probe/trace_analyze.py`
(stdlib) then reduced the 33 601-event chrome trace to the facts below. **The attempt succeeded** —
IREE kernels are visible to kineto/CUPTI on this box, so Q2's GPU side is *not* profiler-blocked
after all (the earlier recon only established that nsys/ncu are blocked).

| fact (30 invocations, `kineto_f64/`) | value |
|---|---|
| kernel events / memcpy events / `cuda_driver` events | **7470 / 1830 / 7470** = 249 / 61 / 249 per invocation (matches the dump exactly) |
| distinct kernel names | **125**; 63 of them run 30×/cell and 62 run 90× (3 per invocation) |
| kernel duration | median **1.408 µs**, mean 1.645, min 1.343, max 3.072 |
| memcpy duration | median **1.632 µs**, mean 1.696 (all DtoD) |
| kernel grid / block / occupancy / shared mem | **(32,1,1)×(32,1,1)** for every kernel; 1024 threads for 4096 elements; "est. achieved occupancy **1 %**"; 0 B shared |
| `cuda_driver` launch call | median **6.45 µs**, mean 6.85 (profiler-inflated) |
| device busy per invocation | **513.9 µs** (kernels 409.7 + memcpy 103.4) vs profiled span 2550.8 µs ⇒ **20.2 % busy** |
| idle gaps per invocation | 309 gaps, median **5.86 µs**, total 2037.9 µs |
| chrome trace | `kineto_f64/kineto_trace.json.gz` (368 KB, kept: under the 5 MB budget) |

Caveat: kineto inflates the wall (the profiled span 2.55 ms vs 1.21–1.47 ms unprofiled) and the
launch-time figures, so only the *counts* and the *kernel durations* are quoted as measurements; the
busy/idle split is quoted as a shape ("device idle dominates"), consistent with the unprofiled
arithmetic in §B.

nsys/ncu: still skipped. The recorded failure is environmental (`QuadDCommon::InvalidArgumentException
… Unknown runtime API function index: 440` in the QDSTRM importer against driver 535.247.01 — see
`../tool_recon/README.md`); the one variant that could plausibly dodge a *record-decoding* failure
(`nsys profile --trace=cuda --cuda-event-trace=false --sample=none --cpuctxsw=none`) does not help,
because the failing table is the generic **runtime-API** record table (`GetCudaCallbackName`,
`TraceProcessEvent.cpp:45`), which is emitted by default and decoded by the same importer. No second
attempt was made, per the round rules.

## Exact commands (each its own locked invocation, repo root, project venv)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
R5=test/benchmark/r5
ENVV='PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1'

# (a) dispatch dumps + row-extreme re-dumps (pure or e2e; --data-rows tiles like bench_gp.py)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env '"$ENVV"' '"$PY"' '"$R5"'/gpu_probe/dispatch_dump.py --fixture test/goldens/forest_random.npz --mode pure --invocations 3 --out-dir '"$R5"'/results/q2_gpu/dumps/f64_pure --json-out '"$R5"'/results/q2_gpu/dumps/f64_pure.json --tag f64_pure'
#   … same with: --mode e2e … f64_e2e | --fixture test/benchmark/fixtures/forest200_tree64_random.npz … f200_pure
#   … re-dumps: --data-rows 8 … f64_r8 | --data-rows 256 … f64_r256 | f200 + --data-rows 128 … f200_r128

# (c) the row-scaling series (frozen r4 pure-dense-iree invocation + --data-rows N --output-checksum)
bash test/benchmark/r5/gpu_probe/run_rows.sh                 # 8 cells, results/q2_gpu/rows/
#   re-run/controls (fresh dir, same command by hand): rows/rerun_1/{f64_r8b,f64_r64b,f200_r64b,f200_r96,f200_r128b}

# (d) the one kineto attempt + its offline reduction + the legacy-source op count (read-only)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=dense EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' '"$R5"'/gpu_probe/kineto_attempt.py --fixture test/goldens/forest_random.npz --mode pure --invocations 30 --out-dir '"$R5"'/results/q2_gpu/kineto_f64 --json-out '"$R5"'/results/q2_gpu/kineto_f64.json --tag kineto_f64'
python3 test/benchmark/r5/gpu_probe/trace_analyze.py --trace "$R5/results/q2_gpu/kineto_f64/kineto_trace.json.gz" --invocations 30 --json-out "$R5/results/q2_gpu/kineto_f64/trace_facts.json" --text-out "$R5/results/q2_gpu/kineto_f64/trace_facts.txt"
sed -n '514,551p' /tmp/evogp_legacy_bench/src/evogp/cuda/forward.cu   # advanced_SR_fitness: 3 device ops
```

## Records

| path | what |
|---|---|
| `dumps/{f64_pure,f64_e2e,f200_pure,f64_r8,f64_r256,f200_r128}/` + `.json` | `dispatch_graph.dot` (raw), `scheduling_stats.csv` (raw, incl. per-command rows), `executables/` (linked PTX + manifest), `dump.log`, and the parsed record (counts, positions, chain, aggregates) |
| `rows/` + `rows/rerun_1/` | the 8-cell row series (`<id>.json` harness record + `<id>.log`), the 5 re-run/control cells, `rows_meta.txt`, `rows_manifest.txt` |
| `kineto_f64.json`, `kineto_f64/` | the kineto attempt: `kineto_key_averages.txt` (kernel table), `kineto.log`, `kineto_trace.json.gz` (368 KB), `trace_facts.json`/`.txt` (the reduced timeline facts) |
| `../../gpu_probe/` | the tooling: `dispatch_dump.py`, `run_rows.sh`, `kineto_attempt.py`, `trace_analyze.py` (all analysis-only; monkeypatches nothing, edits no `src/`) |

## Deviations / limits

* The dump cells' own warm medians (2.2–8.5 ms single-sync calls) are **not** comparable to the
  frozen harness' tight-block medians (1.2–1.5 ms) — the dump probes time one `runner.run()` per
  sync pair, which exposes the full launch latency of each call and rides the same co-tenant drift.
  Only the dumps' *structure* and counts are used from those cells.
* `--iree-hal-dump-executable-binaries-to` writes one linked PTX (0.59 MB for f64, kept; it is the
  125-kernel listing, not an IR dump). No giant IR dumps were produced.
* `dispatch_dump.py` parses only a sample of the scheduling-statistics execution rows into its JSON
  (24 of 310 for f64); the CSVs are the complete raw evidence.
* The `f64_r8` primary cell is a contention outlier (kept, flagged, superseded by `f64_r8b` for the
  table/fit); every other cell's blocks are monotone-decreasing (warm-up) with the box's normal spread.
* The four re-run/control cells (`rows/rerun_1/`) exist because the primary series had one contended
  cell and one surprising f200 point; they are separate records, never overwritten.
* The f200 r64→r128 rise is left **open** (§C): the attempted explanations (dispatch structure,
  element count, transient size) are all contradicted by the measured dumps/sub-series; the honest
  statement is the two-protocol disagreement (+12 % single-sync vs +70 % tight block).
* No `src/evogp` or `etl` change was needed (no NEEDS-PARENT-ACTION). All processes exited 0; the
  IREE `nanobind: leaked …` teardown chatter is the known stock-ETL noise; no GPU process was left
  behind by this phase.
