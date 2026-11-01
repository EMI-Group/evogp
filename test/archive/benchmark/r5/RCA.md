# `RCA.md` — r5 root-cause analysis: why the ETL dense route costs ≈9.5× the legacy CUDA `evaluate`

**Round**: r5, analysis-only (no `src/evogp` change, no write in the foreign ETL checkout, nothing edited
outside `test/benchmark/r5/`). **Question**: the r4 verdict quotes a ≈9.5–10.5× old-vs-new gap on the type-1
end-to-end `problem.evaluate(forest)` (dense route, iree backend under `EVOGP_EVAL_MODE=auto`) from
*cross-window* numbers — what is one warm call made of, and which mechanisms are measured, inferred or unknown?

**Evidence convention**: every quantitative claim carries its record path, relative to `test/benchmark/r5/`.
The phase write-ups (`results/*/SUMMARY.md`) were read first and are ground truth; JSON/log records are cited
where an exact stored field is needed. Every phase's records carry their own `evogp_commit` stamp and the ETL
identity, with `src/evogp` unmodified (`results/env_probe.txt`).

---

## 1. TL;DR

1. **The ≈10× gap is two additive blocks, both measured.** A **GPU-side block** — the compiled dense graph is
   a **310-device-op serial chain** (249 dispatches + 61 device-to-device copies, longest path 125) of ~1.4 µs
   kernels at 1 % occupancy, whose *pure forward* alone costs **1.2142 / 1.4612 ms** — plus a **host-side
   block** of per-call glue the frozen pure runner never pays: 10 payload uploads (**3.51 / 1.53 ms**), dense
   decode (**0.70 / 1.38 ms**), host loss assembly (0.51 ms), readback (0.12 ms) and the engine prologue
   (`results/q4_window/SUMMARY.md` §1, `results/q1_decomp/SUMMARY.md`, `results/q2_gpu/SUMMARY.md`).
2. **Same-window 3-way anchors** (one serialized window, six processes, GPU 6, 2026-09-13 20:30–20:33):
   legacy end-to-end **0.6461 / 0.6372 ms**, dense-iree end-to-end **5.3823 / 5.7912 ms**, pure forward
   **1.2142 / 1.4612 ms** ⇒ **8.33× / 9.09×** end-to-end and **1.88× / 2.29×** for the pure forward, which is
   only **23 % / 25 %** of the e2e call (`results/q4_window/SUMMARY.md` §1, §9).
3. **What it is *not*.** Compile churn (warm `build_cached` 0.0792 / 0.0767 ms, 24 hits / 1 miss, vs
   10.84 / 11.28 s cold), readback (0.12 ms), the benchmark harness (falsified: a plain loop with no harness
   and no probe still costs **6.117 / 6.010 ms**, `results/q8_minimal/SUMMARY.md`), and an "engine run-path
   tax" (engine seam vs frozen `run()` on the same staged payloads: **+0.0009 / +0.0885 ms**,
   `results/q3_ab/SUMMARY.md`).
4. **Floors.** All host glue removed: **3.3752 / 2.8064 ms** (still **5.22× / 4.40×** legacy); pure-forward
   only: **2.3849 / 2.4497 ms** (**3.69× / 3.84×**); `M0 − M5` = **4.844 / 2.623 ms** is the total addressable
   host+prologue cost (`results/q6_floors/SUMMARY.md`).
5. **Legacy parity** (0.646 / 0.637 ms) needs both blocks gone: the GPU side from ~310 device ops to ~1 fused
   op (**≈103× fewer**, the legacy 3-op path) *and* the per-call host glue (`results/q2_gpu/SUMMARY.md` §B.2,
   `results/q6_floors/SUMMARY.md`). One mechanism stays **open**: the 200-tree row-scaling term (§5).

---

## 2. Method & evidence map

### 2.1 Probes (all committed in this node; one fresh process per cell, lock-serialized)

| phase / question | probe | what it does | records |
|---|---|---|---|
| P0 env freeze | shell probes | identity, profilers, lock/rlimit, fixtures | `results/env_probe.txt`, `results/tool_recon/README.md`, `results/tool_recon/import_log.txt` |
| Q4 same-window anchors | `window_probe.sh` | 3 smoke cells + 9 window cells (legacy ×3, e2e ×4 incl. probe + shadow, pure ×2) + post-window control | `results/smoke/`, `results/q4_window/*.json`/`.log`, `results/q4_window/window_meta.txt`, `results/q4_window/SUMMARY.md` |
| Q4 IREE gate | `../r4/run_agreement_r4.sh` + `../r4/check_agreement_r4.py` | six-arm numpy-vs-iree-vs-iree_spec pre-flight | `results/backend_agreement/` (6 records + `run_agreement_r4.stdout.txt` + `check_agreement_r4.out.txt`) |
| Q1/Q5/Q7 decomposition | `decomp_probe.py` + `probe_lib.py` | per-call-site exclusive timers over one warm `evaluate` (25 calls), region-sync/plain/cold controls, cache counters | `results/q1_decomp/decomp_f64.json`/`.log`, `results/q1_decomp/decomp_f200.json`/`.log`, `results/q1_decomp/SUMMARY.md` |
| Q1 cProfile cross-check | `decomp_probe.py --cprofile` | zero-instrumentation ranking of the same route | `results/q1_decomp/cprofile_f64.json`, `results/q1_decomp/cprofile_f64.txt` |
| Q8 harness falsification | `minimal_loop.py` | the same route in a plain loop: no harness, no probe, no monkeypatch | `results/q8_minimal/minimal_f64.json`/`.log`, `results/q8_minimal/minimal_f200.json`/`.log`, `results/q8_minimal/SUMMARY.md` |
| Q2 GPU mechanism | `gpu_probe/dispatch_dump.py`, `gpu_probe/run_rows.sh`, `gpu_probe/kineto_attempt.py`, `gpu_probe/trace_analyze.py` | IREE compile-side dispatch dumps, `--data-rows` scaling, kineto timeline + offline reduction | `results/q2_gpu/dumps/<cell>/` (`dispatch_graph.dot`, `scheduling_stats.csv`, `executables/`), `results/q2_gpu/rows/` (+ `results/q2_gpu/rows/rerun_1/`, `results/q2_gpu/rows/rows_meta.txt`), `results/q2_gpu/kineto_f64/` (`trace_facts.json`/`.txt`), `results/q2_gpu/SUMMARY.md` |
| Q6 floors | `floor_probe.py` | controlled-removal matrix M0–M5 (+M4V, brackets, inert control) + upload micro-probe | `results/q6_floors/rerun_1/floor_f64.json`, `results/q6_floors/rerun_1/floor_f200.json` (+`.log`), contended control `results/q6_floors/floor_f64.json`, `results/q6_floors/SUMMARY.md` |
| Q3 A/B | `ab_probe.py` | same-process engine-seam vs frozen-`run()` arms on the same compiled artifacts | `results/q3_ab/ab_f64.json`/`.log`, `results/q3_ab/ab_f200.json`/`.log`, `results/q3_ab/SUMMARY.md` |

### 2.2 Round identity and stamps

* **ETL (primary, every cell)**: the venv's editable install of the foreign checkout `/mnt/local-ssd/bchuang/etl`
  at `96df7da8f2ba39f52cf2f2b61888e67f70471bc9`, working tree **clean** (`etl_repo_status: <empty>` in every
  record; `results/env_probe.txt` lines 17–27). One optional cell
  (`results/q4_window/q4_eval64_iree_shadow87c7562.json`) imported the r4 shadow `/tmp/evogp_etl_patched_87c7562`
  instead — proved by its log's `etl: … file=` line and the sha256 pins (`results/q4_window/SUMMARY.md` §5).
* **evogp stamps** (unmodified source in every phase): `d23718a` (P0, `results/env_probe.txt`), `3f79356`
  (Q4 window + smoke + control), `1a14b77` (Q1 decomp f64/f200), `48dd1bc` (cProfile + Q8), `247c1e0` (all Q2
  cells), `c73006c` (Q6 contended first attempt), `682733e` (Q6 quiet `rerun_1` pair), `b4003d4` (Q3 pair).
* **Environment + protocol**: `CUDA_VISIBLE_DEVICES=6` (GPU 7 is hardware-ECC-faulty and unused,
  `results/env_probe.txt` lines 47–49), project venv (py3.11.2, torch 2.6.0+cu124), repo-root cwd,
  `PYTHONPATH=src`, `ETL_DISABLE_LOCATIONS=1`, `ulimit -n 65536`, exactly one
  `flock -w 7200 /tmp/gpbench_critical.lock` per invocation with `timeout` **inside** the lock
  (`results/q4_window/window_meta.txt` header). New-side route: `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto
  EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`. Fixtures (repo-root-relative inputs, not records):
  f64 = `test/goldens/forest_random.npz`
  sha256 `0bfb0bbf…ff9ca0` (64 trees); f200 = `test/benchmark/fixtures/forest200_tree64_random.npz`
  sha256 `6789ef61…9741b9` (200 trees); both 64 data rows (`results/env_probe.txt` lines 74–76).
* **Co-tenancy** (sampled per cell): GPU 6 carried one foreign training process (`pid 2903622`, 30 486 MiB) at
  every sample, 17 796 MiB free, utilisation 0–9 % except one 90 % burst (`results/q4_window/window_meta.txt`,
  `results/q4_window/SUMMARY.md` §4/§8); Q6/Q3 windows: loadavg 290–373 on 128 cores with low real CPU busy
  (`results/q6_floors/SUMMARY.md` §"Host window", `results/q3_ab/SUMMARY.md` §"Host window").
* **IREE numbers are gated**: the six-arm agreement pre-flight re-run in the r5 window returned
  `gate: PASS (30 check(s), 0 failure(s))`, verdict `PASS-with-note` on both fixtures — not bit-identical, but
  every pair inside `rtol=atol=1e-4` (numpy-vs-iree `max|diff|` 7.629e-06 f64 / 0.03125 abs f200;
  `numpy vs iree_spec` bit-identical, `max|diff| = 0`)
  (`results/backend_agreement/check_agreement_r4.out.txt`).

### 2.3 Tool blocks (kernel-level profilers) and the kineto success

| tool | state | evidence (verbatim) |
|---|---|---|
| `nsys` 2022.4.2.1 | collection works, **QDSTRM import/analysis fails**: `QuadDCommon::InvalidArgumentException … Unknown runtime API function index: 440` (`TraceProcessEvent.cpp(45)`, `GetCudaCallbackName`); the written `.nsys-rep` has empty kernel tables | `results/tool_recon/import_log.txt` lines 11–25, `results/tool_recon/README.md`, `results/env_probe.txt` lines 51–57 |
| `ncu` 2022.3.0 | **unusable**: every attempt dies with `CUDA error: API call is not supported in the installed CUDA driver` | `results/tool_recon/README.md`, `results/env_probe.txt` lines 51–57 |
| `torch.profiler` (kineto/CUPTI) | **SUCCEEDED** — IREE kernels and DtoD copies are visible, so the Q2 GPU side is *not* profiler-blocked; 33 601 events reduced to §5's counts/durations (only counts and kernel durations are quoted as measurements) | `results/q2_gpu/kineto_f64.json`, `results/q2_gpu/kineto_f64/trace_facts.json` (+ `kineto_trace.json.gz`, `kineto_key_averages.txt`), `results/q2_gpu/SUMMARY.md` §D |

Consequence: no kernel-level counters and no *per-launch host-vs-device* split exist this round; the mechanism
below is pinned indirectly (dumps + kineto + row series + host timers), and the one plausible nsys variant was
rejected with a named rationale instead of a second attempt (`results/q2_gpu/SUMMARY.md` §D).

---

## 3. Same-window anchor table (Q4)

### 3.1 The 3-way anchors (one serialized window, six processes, 20:30:54 – 20:33:11)

Warm = `median_warm_ms` (median of the 24 post-cold `problem.evaluate` calls; call 0 cold) for the eval64
cells; `pure.median_per_invocation_ms` (median of 5 blocks × 100 invocations after 10 warmup invocations) for
the pure cells.

| fixture | (a) legacy end-to-end | (b) dense-iree end-to-end | (c) dense-iree PURE forward | **b/a** | **c/a** | **c/b** | host glue = b−c |
|---|---|---|---|---|---|---|---|
| 64 trees | **0.6461 ms** | **5.3823 ms** | **1.2142 ms** | **8.33×** | **1.88×** | 0.226 | 4.168 ms (77.4 % of b) |
| 200 trees | **0.6372 ms** | **5.7912 ms** | **1.4612 ms** | **9.09×** | **2.29×** | 0.252 | 4.330 ms (74.8 % of b) |

Records: `results/q4_window/q4_leg64.json`, `results/q4_window/q4_leg200.json`,
`results/q4_window/q4_eval64_iree.json`, `results/q4_window/q4_eval200_iree.json`,
`results/q4_window/q4_pure64_iree.json`, `results/q4_window/q4_pure200_iree.json`; formulation
`results/q4_window/SUMMARY.md` §1, verdict §9. Cold side (never mixed into the warm medians): legacy
`first_call_ms` 22.0 / 22.8 ms; dense-iree `first_call_ms` 12497.0 / 13167.7 ms; pure construct (compile)
12216.2 / 12600.2 ms with `compile.path="frozen"`, `cache_misses=1` (`results/q4_window/SUMMARY.md` §1).

### 3.2 Repeatability: control pair, drift bracket, cross-window context

| run | cell | warm median | note (record) |
|---|---|---|---|
| window | `q4_leg64` (first) | 0.6461 ms | window anchor (`results/q4_window/q4_leg64.json`) |
| window | `q4_leg200` | 0.6372 ms | matches r3/r4 legacy 0.6323 / 0.6357 ms (`results/q4_window/q4_leg200.json`) |
| window | `q4_leg64_bracket` (last) | **1.7931 ms (+177.5 %)** | *contention event*, not drift: uniform 1.75–1.80 ms series with a 90 % co-tenant utilisation sample at that boundary (`results/q4_window/q4_leg64_bracket.json`, `results/q4_window/window_meta.txt`) |
| control | `q4_leg64` | **0.6187 ms (−4.2 %)** | the anchor repeats (`results/q4_window/rerun_1/q4_leg64.json`) |
| control | `q4_eval64_iree` | **6.8312 ms (+26.9 %)** | the new-side e2e cell does **not** repeat tightly (`results/q4_window/rerun_1/q4_eval64_iree.json`) |
| control | `q4_eval64_iree_probe` | **6.7328 ms (−0.7 %)** | the probe cell repeats tightly (`results/q4_window/rerun_1/q4_eval64_iree_probe.json`) |

* **Cross-window r4 context** (not re-measured by r5): r4 e2e 6.667 / 6.050 ms
  (`../r4/results/after/e2e/e2e_fixture64_iree_auto.json`, `../r4/results/after/e2e/e2e_fixture200_iree_auto.json`)
  vs the frozen legacy anchors 0.6323 / 0.6357 ms (`../r2/round3/results/legacy/leg_c5r2.json`,
  `../r2/round3/results/legacy/leg_fixture200_n64.json`) ⇒ 10.54× / 9.52×; r4 pure 1.4468 / 1.5930 ms
  (`../r4/results/after/pure64_dense_iree.json`, `../r4/results/after/pure200_dense_iree.json`).
* **Honest gap statement**: **8.33× / 9.09×** against the calm window anchors, **11.04×** against the control
  anchor, with the r4 quotes in between ⇒ **≈8–11×, centred on ~9.5×**; the r4 headline is not an artifact of
  its cross-window construction (`results/q4_window/SUMMARY.md` §4/§9).
* **Probe tax: nil.** The +1.3951 ms (+25.9 %) in-window delta is cell-to-cell variation: an adjacent-in-time
  pair (plain then probe, 17 s apart) sits within −1.4 %, and inside the probe cell
  `probe_span_ms − total_ms` = **+0.028 ms per call (≈+0.4 %)**
  (`results/q4_window/q4_eval64_iree_probe.json`, `results/q4_window/SUMMARY.md` §3). Dense-route caveat: the
  probe's wrap set excludes the dense path, so its `run_ms` 2.0805 ms vs 4.70 ms `misc` split is a bridge datum,
  not a Q1 result (`results/q4_window/SUMMARY.md` §3/§10.4).
* **Shadow bridge**: the `87c7562` ETL shadow reads 6.0165 ms — +11.8 % over the window's primary cell, −11.9 %
  under the control's ⇒ inside the new-side spread, so **the r4-revision-vs-current-ETL difference is not
  resolvable in this round's data** (`results/q4_window/q4_eval64_iree_shadow87c7562.json`,
  `results/q4_window/SUMMARY.md` §5).
* **Pure-route correctness**: `output_checksum.sha256` = `9132bea9…a3b0` (64) / `e77992de…f322` (200),
  **bit-equal to the r4 after-records**, same shapes ([64,64] / [200,64]) and NaN census (24 / 376)
  (`results/q4_window/q4_pure64_iree.json`, `../r4/results/after/pure64_dense_iree.json`,
  `results/q4_window/SUMMARY.md` §6). Per-tree cost 18.97 / 7.31 µs; `n_const_trees=0`;
  `n_graph_calls_per_run=1` (introspected).
* **fds**: IREE descriptors flat at **91** (after-cold = end) in every new-side cell, legacy 39/59/59 — the
  stock-ETL leak of the r2/r3 era is fixed on `96df7da` (`results/q4_window/window_meta.txt` cell table).

---

## 4. Per-call decomposition (Q1 + Q5)

Route: `SymbolicRegression.evaluate` → `Forest.SR_fitness` → `dense_engine.sr_fitness_dense` (in-graph
**loss** variant, one `(63, P)` payload set per call, one `etl.run`). Records: `results/q1_decomp/decomp_f64.json`
(warm median **7.3378 ms**), `results/q1_decomp/decomp_f200.json` (warm median **6.6167 ms**).

### 4.1 Component table (exclusive self time, warm calls 1..24)

| component | count | f64 ms | f64 % | f200 ms | f200 % | what it is |
|---|---|---|---|---|---|---|
| `etl_tensor_to_payload` | 10 | **3.5084** | 47.8 % | **1.5300** | 23.1 % | `etl.from_numpy(host).to(device)` per payload (caller = `payload`) |
| `etl_run` | 1 | **2.1267** | 29.0 % | **2.3673** | 35.8 % | the single compiled dense-graph invocation |
| `prepare_dense_self` | 1 | 0.6958 | 9.5 % | 1.3830 | 20.9 % | decode + structural validation + `np.unique` of function ids |
| `run_loss_plan_self` | 1 | 0.5120 | 7.0 % | 0.5134 | 7.8 % | host assembly: reshape/accumulate per chunk, divide by N |
| `as_numpy_d2h` | 5 | 0.1437 | 2.0 % | 0.1683 | 2.5 % | `tensor.detach().cpu().numpy()`: x, labels, node_value, node_type, subtree_size |
| `etl_tensor_to_to_host` | 1 | 0.1122 | 1.5 % | 0.1165 | 1.8 % | the loss readback `result.to(cpu)` inside `_to_host` |
| `sr_fitness_dense_self` | 1 | 0.1115 | 1.5 % | 0.1146 | 1.7 % | `resolve_backend` + the fitness H2D |
| `build_dense_build_cached` | 1 | 0.0792 | 1.1 % | 0.0767 | 1.2 % | warm `etl.build_cached` prologue (Q7, §7) |
| `plan_dense_self` | 1 | 0.0771 | 1.1 % | 0.0782 | 1.2 % | chunk shape + specs/cfg construction |
| `etl_from_numpy` | 10 | 0.0366 | 0.5 % | 0.0373 | 0.6 % | the zero-copy host wrap of each payload |
| `prep_column` | 7 | 0.0344 | 0.5 % | 0.0340 | 0.5 % | `_prep_column` (host column transpose/pad) |
| `outer_evaluate`/`forest_sr_self`/`policy`; `to_host_self`/`input_to_numpy_self` | 1 + 1 each | 0.0721 + 0.0155 | 1.2 % | 0.0905 | 1.4 % | wrappers, `isinstance`+negation, route policy |
| **SUM(exclusive)** | | **7.5252** | **102.55 %** | **6.5106** | **98.40 %** | sum of medians vs median of sums |

**Reconciliation**: the per-call check (`sum(exclusive) / call wall`) is **99.94 % median for f200
(99.89–99.96 %) and 99.95 % for f64 (99.92–99.96 %)** — complete wrap set, no `depth_overflow`, no wrap
failures (`missing: none`), and the fitness tensor is byte-identical across all 29 measured calls
(`all_equal: true`; f64 sha256 `f4e51404…13f8`). The sum-of-*medians* exceeds the median-of-sums on f64
(102.55 %) only because a few stalled 11–13 ms calls inflate several components' medians independently
(`results/q1_decomp/decomp_f64.json` `warm.reconciliation_pct`, `results/q1_decomp/SUMMARY.md`).
**Residual**: nothing is hidden in exclusive accounting — the residual is the definition residual
(sum-of-medians vs median-of-sums), reported rather than smoothed.
**GPU tail**: `fence_after:etl_run` = **0.031 ms** — the compiled forward is drained *inside* `etl.run` and the
uploads' device copy is absorbed before it returns. **Probe overhead** (uninstrumented control): f64
7.2206 ms vs 7.3378 ms instrumented (+1.6 %), i.e. the instrumentation is not the cost
(`results/q1_decomp/SUMMARY.md`).

### 4.2 Q5 — top-3 host costs, and the upload footnote

| rank | f64 | ms | f200 | ms |
|---|---|---|---|---|
| 1 | 10 payload uploads (`from_numpy().to(device)`) | **3.51** | `etl.run` — *not host work*, the graph call itself | 2.37 |
| 1 (host) | | | 10 payload uploads | **1.53** |
| 2 | decode (`prepare_dense`) | **0.70** | decode | **1.38** |
| 3 | host assembly in `run_loss_plan` | **0.51** | host assembly | **0.51** |
| — | readback (`_to_host` + D2H) | 0.12 | readback | 0.12 |

Host-side top 3: **payload uploads → decode → per-chunk loss assembly**; readback is *not* top-3 (0.12 ms — the
loss variant reads back only `(pop_size,)` floats) (`results/q1_decomp/SUMMARY.md` Q5).

**Upload footnote.** The two cells upload the same 10 payloads, f200 carrying 3.1× the bytes of f64 (50 KB vs
16 KB per node column), yet f64 measures **1.53 → 3.51 ms** *more*, and f64's upload time is the least stable
(min 3.09 / median 3.51 / mean 3.76 / max 5.80 ms). Q1 recorded this as measured-not-explained; Q6 **closed it
as a fixed-cost effect**: per-payload cost is dominated by a size-independent per-op overhead (each isolated
≤512 B upload still costs ~0.1–0.3 ms), so the fixture with smaller payloads can measure the larger total
(`results/q1_decomp/SUMMARY.md` Q5, `results/q6_floors/SUMMARY.md` §"Upload anomaly"). The *mechanism* (IREE HAL
allocator pool per size class, queue depth at allocation, staging-buffer churn) is untested and marked unknown.

### 4.3 cProfile cross-check (ranking only)

`48201 function calls in 0.190 s` ⇒ **7.61 ms/call** (~4 % profiler inflation over the probe's own 7.34 ms;
cProfile cannot see GPU time). Top-5 by `tottime`: `iree.py:1655 _wait_copy_semaphore` 175 calls (7/call),
**1.84 ms/call** — the largest self time in the process; `iree/runtime/function.py:146 _invoke` 25 calls
(1/call) **1.80 ms/call**; `iree.py:1679 _staged_input` 250 calls (10/call, 2.88 ms/call cumulative);
`dense_engine.py:932 _decode_columns` 25 calls (0.64 ms cumulative); `dense_engine.py:1924 sr_fitness_dense`
25 calls (7.40 ms cumulative, the root) (`results/q1_decomp/cprofile_f64.txt`,
`results/q1_decomp/cprofile_f64.json`). Timer-vs-cProfile ranking agrees (upload 3.44 vs 3.51; run 1.96 vs 2.13;
decode 0.64 vs 0.70; readback 0.12 vs 0.11). What cProfile adds is the *shape*: the upload bucket is not a
memcpy but `_staged_input` blocking in `_wait_copy_semaphore`, and the run bucket is dominated by the raw IREE
`_invoke` — host-side staging/launch plumbing (`results/q1_decomp/SUMMARY.md`).

---

## 5. GPU-side mechanism (Q2)

### 5.1 What the compiled dense graph is (compile-side dispatch dumps)

Method: `ETL_IREE_COMPILE_ARGS` → `--iree-flow-dump-dispatch-graph`,
`--iree-scheduling-dump-statistics-file`, `--iree-hal-dump-executable-binaries-to`, one fresh first call per cell
(`gpu_probe/dispatch_dump.py`; raw dumps under `results/q2_gpu/dumps/<cell>/`).

| dump cell | mode | rows | dispatches | longest chain | graph nodes/edges | linked PTX kernels | DtoD copies | transient | compile | warm single-call ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `f64_pure` | pure | 64 | **249** | **125** | 578 / 1500 | 125 | 61 | 2.88 MB | 12.15 s | 2.225 / 2.230 / 2.209 |
| `f64_e2e` | e2e | 64 | **249** | **125** | 584 / 1506 | 125 | 61 | 2.86 MB | 12.03 s | 8.465 / 6.507 / 6.335 |
| `f200_pure` | pure | 64 | **253** | **127** | 587 / 1524 | 127 | 62 | 8.98 MB | 11.95 s | 2.817 / 3.180 / 3.184 |
| `f64_r8` | pure | 8 | 249 | 125 | 578 / 1500 | 125 | 61 | 0.36 MB | 12.70 s | 3.00 / 3.08 / 3.04 |
| `f64_r256` | pure | 256 | 249 | 125 | 578 / 1500 | 125 | 61 | 11.50 MB | 12.20 s | 2.70 / 2.66 / 2.65 |
| `f200_r128` | pure | 128 | 253 | 127 | 587 / 1524 | 127 | 62 | 17.97 MB | 12.50 s | 3.17 / 3.16 / 3.17 |

Records: `results/q2_gpu/dumps/{f64_pure,f64_e2e,f200_pure,f64_r8,f64_r256,f200_r128}.json`; formulation
`results/q2_gpu/SUMMARY.md` §A. Measured facts: **one dispatch per elementwise step of the trace** (249 for the
63-node trace, ~4 per tree node once constant loads/broadcasts/selects are counted); the count is a property of
the **trace, not the population** (identical at 8 and 256 rows); the dispatches are **chained, not fanned out**
(longest path 125/127 ≈ the whole trace, each step waits for its predecessor); the linked PTX holds 125/127
`main_dispatch_*_elementwise_*` kernels and the aggregate scheduling row reads `Awaits: 2, Submissions: 1,
Copies: 61, Dispatches: 249, Executables: 125`. The **e2e and pure routes compile the same structure**
(249/125/125; transient 2.88 vs 2.86 MB), so §4's host-glue split is not a GPU-structure difference.

### 5.2 Per-dispatch math and the legacy contrast

| quantity | 64-tree fixture | 200-tree fixture | source |
|---|---|---|---|
| dispatches per pure forward | 249 | 253 | dump (measured) |
| DtoD copies per forward | 61 | 62 | dump (measured) |
| total device ops per forward | **310** | **315** | arithmetic from the dump |
| pure warm median (same window) | **1.2142 ms** | **1.4612 ms** | `results/q4_window/q4_pure64_iree.json`, `results/q4_window/q4_pure200_iree.json` |
| ⇒ per device op | **3.92 µs** | **4.64 µs** | arithmetic |
| ⇒ per dispatch (kernels only) | **4.88 µs** | **5.78 µs** | arithmetic |
| ⇒ per serial-chain step (125 / 127 deep) | **9.7 µs** | **11.5 µs** | arithmetic |
| legacy whole evaluate (contrast) | **0.6461 ms** | **0.6372 ms** | `results/q4_window/q4_leg64.json`, `results/q4_window/q4_leg200.json` |
| legacy device ops for that evaluate | 3 | 3 | counted from the read-only legacy source |
| pure / legacy | **1.88×** | **2.29×** | measured (Q4) |

Formulation: `results/q2_gpu/SUMMARY.md` §B.1. **Legacy contrast**: `advanced_SR_fitness`
(`/tmp/evogp_legacy_bench/src/evogp/cuda/forward.cu:514-551`, read-only legacy `b613503`) evaluates the whole
population with **3 device ops** — one `cudaMemsetAsync` + **one** `treeGPRegressionFitnessKernel<false,true>`
launch, grid `(popSize,1) × 64` (each block walks one tree in registers/shared memory) + **one**
`averageFitnessValueKernel` — **≈103× fewer device ops** than the dense route's 310, with the per-tree walk
inside the kernel (`results/q2_gpu/SUMMARY.md` §B.2). Its wall time is only a contrast anchor: with nsys
unavailable it cannot be split into launch vs execution (**unknown**).

### 5.3 Kineto facts (30 invocations, compile outside the profile)

| fact | value (record: `results/q2_gpu/kineto_f64/trace_facts.json`) |
|---|---|
| kernel / memcpy / `cuda_driver` events | **7470 / 1830 / 7470** = 249 / 61 / 249 per invocation (matches the dump exactly) |
| distinct kernel names | **125**; 63 run 30×/cell and 62 run 90× (3 per invocation) |
| kernel duration | median **1.408 µs**, mean 1.645, min 1.343, max 3.072 |
| memcpy duration | median **1.632 µs**, mean 1.696 (all DtoD) |
| grid / block / occupancy / shared mem | **(32,1,1)×(32,1,1)** for every kernel = 1024 threads for 4096 elements; est. achieved occupancy **1 %**; 0 B shared |
| `cuda_driver` launch call | median **6.45 µs**, mean 6.85 (profiler-inflated) |
| device busy per invocation | **513.9 µs** (kernels 409.7 + memcpy 103.4) vs profiled span 2550.8 µs ⇒ **20.2 % busy** |
| idle gaps per invocation | **309** gaps, median **5.86 µs**, total 2037.9 µs |

Caveat: kineto inflates the wall (profiled span 2.55 ms vs 1.21–1.47 ms unprofiled) and the launch figures, so
counts and kernel durations are measurements while the busy/idle split is a shape. The unprofiled statement is:
**device-busy is 35–42 % of the warm call and more than half of the timeline is the launch/serialize gap**
(`results/q2_gpu/SUMMARY.md` §B.3/§D).

### 5.4 `--data-rows` scaling series

Each cell = the frozen r4 `pure64_dense_iree` invocation + `--data-rows N --output-checksum`, one fresh process
per cell; warm = tight-block median (5 × 100 invocations, readback/host assembly excluded).

| fixture | rows | elements/dispatch | warm median ms | block medians ms | per-dispatch µs | construct s |
|---|---|---|---|---|---|---|
| f64 | 8 | 512 | 1.382 (`rerun_1/f64_r8b`; primary 3.923, contended) | 1.71 / 1.46 / 1.38 / 1.28 / 1.25 | 5.55 | 13.3 |
| f64 | 16 | 1 024 | 1.429 | 1.92 / 1.51 / 1.43 / 1.29 / 1.28 | 5.74 | 15.7 |
| f64 | 32 | 2 048 | 1.434 | 1.77 / 1.46 / 1.43 / 1.14 / 1.18 | 5.76 | 15.5 |
| f64 | 64 | 4 096 | 1.360 (`rerun_1/f64_r64b` 1.302) | 1.80 / 1.55 / 1.34 / 1.34 / 1.36 | 5.46 | 13.1 |
| f64 | 128 | 8 192 | 1.469 | 1.73 / 1.48 / 1.47 / 1.38 / 1.22 | 5.90 | 16.1 |
| f64 | 256 | 16 384 | 1.321 | 1.43 / 1.40 / 1.32 / 1.27 / 1.27 | 5.31 | 15.5 |
| f200 | 64 | 12 800 | 1.345 (`rerun_1/f200_r64b` 1.423) | 1.79 / 1.57 / 1.29 / 1.30 / 1.35 | 5.32 | 13.3 |
| f200 | 96 | 19 200 | 1.640 | 1.92 / 1.63 / 1.65 / 1.64 / 1.60 | 6.48 | 16.1 |
| f200 | 128 | 25 600 | 2.406 (`rerun_1/f200_r128b` 2.281) | 2.46 / 2.43 / 2.41 / 2.40 / 2.38 | 9.51 (8.02 rerun) | 16.5 |

Records: `results/q2_gpu/rows/f64_r8.json` … `results/q2_gpu/rows/f200_r128.json` and
`results/q2_gpu/rows/rerun_1/*.json`, `results/q2_gpu/rows/rows_meta.txt`; formulation
`results/q2_gpu/SUMMARY.md` §C. Least-squares fits (warm ms = a·rows + b):

| series | a (µs/row) | b (ms) | R² | reading |
|---|---|---|---|---|
| f64, rows 8→256 (r8/r64 from `rerun_1`) | **−0.24** | **1.410** | 0.12 | **flat**: a 32× payload range costs nothing |
| f200, rows 64→128 (primary cells) | **+14.63** | **0.351** | 0.96 | +0.94 ms over 64 rows |
| f200, rows 64→128 (rerun r64) | **+13.41** | **0.494** | 0.93 | same slope; the "intercept" is one cell's noise |

**Mechanism, with labels.** *measured*: 310 device ops/call, chain depth 125, 1.4–1.65 µs kernels,
(32,1,1)×(32,1,1) grids at 1 % occupancy, 309 idle gaps at 5.86 µs median, 20.2 % device-busy under the
profiler. *inferred*: **launch-and-serialize bound, not compute bound** — the 125-deep dependency chain forbids
overlap, so call time ≈ Σ(launch latency + kernel + gap); an arithmetic bound agrees (~1.0 M-element
`(63, 4096)` payload ≈ 12 MB of traffic ≈ 25 µs at ~500 GB/s, ~40× below the 410 µs of measured kernel time),
and the f64 row series is flat over a 32× payload range. *inferred (unmeasurable here)*: why each launch+gap is
~5 µs rather than ~2–3 µs (ETL per-dispatch HAL plumbing, tiny 1024-thread grids, driver launch path under a
co-tenant-loaded context) — splitting host launch from device gap needs a working GPU timeline or a repaired
ETL async-submission path. *unknown*: the legacy side's internal split, and the f200 row-scaling term (§10)
(`results/q2_gpu/SUMMARY.md` §B.3).

---

## 6. Engine-vs-frozen A/B (Q3)

Same process, same population, same two compiled artifacts (value and loss, ~12 s each in-process), every arm
holding `plan`/`executable`/args directly — so no arm pays per-call key derivation (that cost is Q6's
`M4V`→`M5`, §7).

| arm | f64 total | f64 `[etl.run span]` | f200 total | f200 `[etl.run span]` | what it is |
|---|---|---|---|---|---|
| `A1_engine_fresh` | 5.1868 | [1.8441] | 4.0287 | [1.9442] | engine seam, value artifact, **fresh** payloads per call |
| `A2_engine_staged` | 1.3180 | [1.2668] | 1.3461 | [1.2716] | engine seam, value artifact, **staged** payloads |
| `A3_frozen_runner` | 1.3171 | [1.2664] | 1.2576 | [1.2063] | frozen `build_dense_runner(forest, x).run()` on staged inputs |
| `A3_plain_control` | 1.3236 | — | 1.8151 | — | `A3` with the instrumentation inert |
| `A4_frozen_fresh` | 4.8704 | [1.4302] | 3.7547 | [1.8730] | frozen plan/executable, **fresh** payloads |
| `L1_engine_fresh_loss` | 6.1420 | [1.8583] | 4.5757 | [1.9983] | engine seam, **loss** artifact (the SR route), fresh payloads |
| `L2_engine_staged_loss` | 1.9785 | [1.8696] | 2.0519 | [1.9675] | engine seam, loss artifact, staged payloads |
| `R1_repl_fresh_loss` | 6.0302 | [1.8925] | 4.5082 | [2.0340] | probe-replicated fresh loop (replication control) |
| `R2_repl_fresh_drained` | 6.1169 | [1.8888] | 3.8667 | [2.0152] | + explicit fence between staging and run |
| `E1_e2e_in_process` | 8.1733 | [1.8538] | 5.8916 | [1.6101] | `problem.evaluate(forest)` in-process companion |

Records: `results/q3_ab/ab_f64.json`, `results/q3_ab/ab_f200.json` (medians are the stored
`stats.median_warm_ms` and `etl_run_span_ms` fields; every stored `deltas` entry reproduces from the stored arm
medians to Δ ≤ 0.02 ms — `results/q3_ab/SUMMARY.md` §"Record-integrity note").

| delta | metric | f64 | f200 | meaning | attribution |
|---|---|---|---|---|---|
| `A2−A3` total | total | **+0.0009** | **+0.0885** | engine seam vs frozen `run()` on the **same staged payloads/artifact** | measured |
| `A2−A3` run span | `etl.run` | **+0.0003** | **+0.0653** | the run path itself | measured |
| `A1−A2` total | total | +3.8687 | +2.6826 | building/uploading payloads fresh, per call | measured |
| `A1−A2` run span | `etl.run` | +0.5774 | +0.6726 | freshness *inside* the run (fresh HAL buffers + the staged copy's drain `etl.run` absorbs) | measured |
| `A4−A3` total | total | +3.5534 | +2.4971 | the **frozen** artifact fed fresh payloads pays the same upload cost | measured |
| `L1−L2` total | total | +4.1635 | +2.5238 | fresh-vs-staged on the loss artifact (the engine's real route) | measured |
| `L1−L2` run span | `etl.run` | −0.0113 | +0.0308 | freshness does **not** move the staged run span | measured |
| `L2−A3` run span | `etl.run` | +0.6032 | +0.7611 | **loss vs value artifact** (in-graph loss/keep-mask reduction) | measured |
| `E1−A3` / `E1−L2` total | total | +6.8562 / +6.1948 | +4.6340 / +3.8397 | in-process e2e vs frozen pure run / vs staged loss run | measured |
| `R1−L1` / `R2−R1` total | total | −0.1117 / +0.0866 | −0.0675 / −0.6415 | replication fidelity (must be ~0) / explicit fence | measured |
| `A3−A3plain` overhead | total | −0.0066 | −0.5575 | probe instrumentation on the frozen run path | measured |

**Conclusions** (from `results/q3_ab/SUMMARY.md` §"Attribution"/§"Answer to the Q1-vs-Q4 question"):
**no engine run-path tax** (same staged payloads/artifact: **+0.0009 / +0.0885 ms** total, run span
**+0.0003 / +0.0653 ms**, bounded by the `A3`-vs-plain control); **freshness is a payload property, not an
engine property** (the frozen artifact pays it too: `A4−A3` **+3.5534 / +2.4971 ms**; total freshness
~2.7–3.9 ms/call while the run span moves only ~0.6–0.7 ms, `A1−A2`); and **the "in-engine ≈2.0 ms vs frozen
pure 1.447 ms" puzzle = loss-vs-value artifact variant + measurement context** — Q1's `etl.run`
2.13 / 2.37 ms *is* the loss variant's run span (`L2−A3` +0.6032 / +0.7611 ms), not re-validation or freshness
(`L1−L2` run span ≈ 0), while the frozen figures are value-variant, readback-free `run()` calls in a tight loop
and the in-process figure is one `etl.run` inside a full evaluate; the remaining e2e glue is `E1−L2`.

---

## 7. Floors & falsification (Q6 + Q7/Q8)

### 7.1 Controlled-removal floor matrix (warm median ms, call 0 excluded)

Each arm = 25 harness-shaped calls (one `torch.cuda.synchronize()` pair per call); M0–M4 are the cumulative
chain on the *same* `evaluate`; `M4V` uses the value artifact and `M5` is
`build_dense_runner(forest, x).run()` on the same staged inputs (`runner.executable is` the M4V artifact: True);
`M0R`/`M3R` are closing drift brackets and `M0P` the instrumentation-inert control.

| arm | f64 median | f64 Δ vs prev | f200 median | f200 Δ vs prev | removes |
|---|---|---|---|---|---|
| `M0` | 7.2289 | — | 5.0725 | — | — |
| `M1` | 7.7000 | +0.4711 | 4.6831 | −0.3895 | `plan_dense` (decode + unique + plan construction) |
| `M2` | 7.6328 | −0.0672 | 4.6891 | +0.0060 | `_prep_column` ×7 |
| `M3` | **3.3752** | **−4.2576** | **2.8064** | **−1.8827** | `etl.from_numpy` + `Tensor.to(device)` for all 10 payloads (uploads + copy semaphores) |
| `M4` | **3.2411** | −0.1341 | **2.9185** | +0.1120 | `_to_host` D2H + `run_loss_plan` host assembly + fitness H2D |
| `M4V` | 2.8233 | −0.4179 | 3.0286 | +0.1102 | the loss variant + labels/keep payloads (value artifact) |
| `M5` | **2.3849** | −0.4384 | **2.4497** | −0.5789 | `plan_dense` memo lookup + `build_dense` cache prologue + `_resolve_backend` + x/labels host copies |
| `M0R` | 9.1255 | +6.7406 | 8.4176 | +5.9679 | (closing drift bracket) |
| `M3R` | 3.3296 | −5.7959 | 2.8337 | −5.5839 | (closing drift bracket) |
| `M0P` | 8.8012 | +5.4716 | 5.9865 | +3.1528 | (instrumentation-inert control) |

Records: `results/q6_floors/rerun_1/floor_f64.json`, `results/q6_floors/rerun_1/floor_f200.json`; formulation
`results/q6_floors/SUMMARY.md` §"Floor table"/§"Floors". Headline floors: **host glue removed (M3) = 3.3752 /
2.8064 ms**, **host glue + readback removed (M4) = 3.2411 / 2.9185 ms**, **pure only (M5) = 2.3849 / 2.4497 ms**,
**M0 − M5 = 4.8440 / 2.6228 ms** (67.0 % / 51.7 % of the call); the pure-forward floor alone is still
**3.69× / 3.84×** the same-window legacy anchor and the host-glue floor **5.22× / 4.40×**
(`results/q6_floors/SUMMARY.md` §"Headline floors").

**Correctness**: `M0..M4` (and both brackets + the control) produce the **same byte-identical fitness tensor**
(f64 `f4e51404dad8df65…`, f200 `cb37e64a911720bb…`); `M4V`/`M5` are forward-only floors with no fitness
(`results/q6_floors/rerun_1/floor_f64.json` arm hashes, `results/q6_floors/SUMMARY.md` §"Correctness"). No
removal changes a result ⇒ the experiment is valid.
**Drift / overhead (closing controls)**: f64 `M0R` +26.24 %, `M3R` −1.35 %, `M0P` −17.86 %; f200 `M0R`
+65.95 %, `M3R` +0.97 %, `M0P` −15.27 % (`results/q6_floors/SUMMARY.md` §"Drift / instrument overhead") — the
f200 opening bracket is why §10 says f200 *absolutes* carry it while within-chain deltas are robust.
**Upload micro-probe** (per-payload `etl.from_numpy(..).to(device)`, 20 reps; `upload_micro` in both floor
records): f64 batch-of-10 median **3.8020 ms** (payload bytes 113 920 B; isolated medians 0.475–0.495 ms for the
seven 16 128 B node columns, 0.23–0.25 ms for x/labels/keep); f200 batch-of-10 **1.3310 ms** (359 424 B;
isolated 0.143–0.171 ms) ⇒ **closed as a fixed-cost effect, not a bytes effect**
(`results/q6_floors/SUMMARY.md` §"Upload micro-probe"/§"Upload anomaly").
**Contended control**: the earlier first attempt (`results/q6_floors/floor_f64.json`, no prelude) ran beside a
foreign `pt_data_worker` storm: M0 **18.8793 ms** vs the quiet 7.2289 ms (2.61×), M3 3.0222, M4 2.6838,
M5 4.0163, upload micro batch 5.6706 ms — the host-bound arms inflate together while the GPU-run-dominated arms
barely move (`results/q6_floors/SUMMARY.md` §"Contention control"). The quiet `rerun_1` pair is quotable.

### 7.2 Falsification claim table

| # | claim tested | outcome | evidence |
|---|---|---|---|
| C1 | "The ~6 ms is a harness/benchmark artifact" | **FALSIFIED** | plain loop, no harness/probe/monkeypatch ⇒ **6.117 ms (f64, min 5.651 / max 9.310)**, **6.010 ms (f200, min 5.410 / max 11.848)**, first call 12.41/12.42 s (`results/q8_minimal/minimal_f64.json`, `results/q8_minimal/minimal_f200.json`, `results/q8_minimal/SUMMARY.md`) |
| C2 | "Warm calls pay compile/cache churn" | **FALSIFIED** | warm `build_cached` **0.0792 / 0.0767 ms** (range 0.054–0.264), cache **hits 1..24, misses=1, size=1**; a deliberate cold rebuild of the same call costs **10840.713 / 11283.765 ms** (build bucket 99.90 % f64) (`results/q1_decomp/decomp_f64.json`, `results/q1_decomp/decomp_f200.json`, `results/q1_decomp/SUMMARY.md` §Q7) |
| C3 | "Readback is a major cost" | **FALSIFIED as a quotable floor** | Q1 timer 0.1122 / 0.1165 ms (outside top-3); Q6 `M3`→`M4` measures **+0.1341 / −0.1120 ms** vs a 0.736 / 0.745 ms prediction (`results/q6_floors/SUMMARY.md` §"Findings") |
| C4 | "Plan/decode removal buys back its timer cost" | **FALSIFIED on f64 (ratio −0.55×), weak on f200** | `M0`→`M1` −0.4711 (f64) / +0.3895 (f200) vs 0.8630 / 1.5620 predicted — real work, not separable as a floor on the warm path (`results/q6_floors/SUMMARY.md` §"Findings") |
| C5 | "Payload uploads are the top host cost" | **CONFIRMED, larger than timed** | `M2`→`M3` = **4.2576 / 1.8827 ms** vs 3.5450 / 1.5660 ⇒ ratio **1.20 / 1.20**; the only cleanly reproducing step is `M1`→`M2` (`_prep_column` ×7, 0.0672 / −0.0060 vs 0.0340 / 0.0350) (`results/q6_floors/SUMMARY.md` §"Findings") |
| C6 | "The in-engine `etl.run` span is re-validation/freshness overhead" | **FALSIFIED / re-interpreted** | `L1−L2` run span −0.0113 / +0.0308 and `A2−A3` run span +0.0003 / +0.0653; `L2−A3` run span +0.6032 / +0.7611 = the **loss-vs-value variant** (`results/q3_ab/ab_f64.json`, `results/q3_ab/ab_f200.json`) |
| C7 | "The engine (not the payloads) costs ~3–4 ms/call" | **FALSIFIED** | `A4−A3` (frozen artifact, fresh payloads) **+3.5534 / +2.4971 ms** (`results/q3_ab/SUMMARY.md`) |
| C8 | "The pure forward spends its time on arithmetic" | **FALSIFIED (arithmetic bound)** | ~1.0 M elements ≈ 12 MB ⇒ ~25 µs at ~500 GB/s vs 410 µs measured kernel time; f64 row series flat (a = −0.24 µs/row, R² = 0.12) over 32× rows (`results/q2_gpu/SUMMARY.md` §B.3/§C) |
| C9 | "The GPU side is launch/serialization-bound" | **CONFIRMED within the evidence; mechanism labelled inferred** | 310 chained device ops, depth 125, 1.4–1.65 µs kernels, 309 gaps at 5.86 µs median, 20.2 % device-busy under kineto (35–42 % unprofiled) (`results/q2_gpu/kineto_f64/trace_facts.json`) |
| C10 | "The f200 cost has a row-scaling term" | **OPEN** | +14.63 µs/row R² 0.96 (tight block) vs +12 % (single sync) — protocols disagree (§10) (`results/q2_gpu/rows/f200_r64.json`, `results/q2_gpu/rows/f200_r128.json`, `results/q2_gpu/rows/rerun_1/f200_r64b.json`) |

---

## 8. Ranked mechanism list

| rank | mechanism | est. ms (f64 / f200) | label | evidence path | falsification status |
|---|---|---|---|---|---|
| 1 | **Dispatch-bound dense graph**: 310 device ops (249 dispatches + 61 DtoD copies) on a 125-deep serial chain, 1.4–1.65 µs kernels at 1 % occupancy | 1.21 / 1.46 (pure forward); ≥0.51 device-busy (f64 kineto only; f200 not traced) | **measured** (mechanism *inferred*) | `results/q4_window/q4_pure64_iree.json`, `results/q4_window/q4_pure200_iree.json`, `results/q2_gpu/dumps/f64_pure.json`, `results/q2_gpu/kineto_f64/trace_facts.json` | confirmed by counts + kineto; "why ~5 µs/launch" stays inferred-unmeasurable |
| 2 | **Per-call payload uploads ×10** (`etl.from_numpy` + `Tensor.to(device)`, incl. `_wait_copy_semaphore`) | 3.51 / 1.53 timed; 4.26 / 1.88 by removal | **measured** | `results/q1_decomp/decomp_f64.json` (`etl_tensor_to_payload`), `results/q6_floors/rerun_1/floor_f64.json` (`M2→M3`), `results/q1_decomp/cprofile_f64.txt` | confirmed and *larger* than its timer (ratio 1.20 / 1.20) |
| 3 | **Decode + unique (`prepare_dense`) and host loss assembly (`run_loss_plan`)** | 0.70 + 0.51 / 1.38 + 0.51 | **measured** (real work, not removable as a floor) | `results/q1_decomp/decomp_f64.json`, `results/q1_decomp/decomp_f200.json` (`prepare_dense_self`, `run_loss_plan_self`) | removal disagrees with timers (`M0→M1` −0.55× / +0.25×) |
| 4 | **Per-call freshness / re-staging** (fresh HAL buffers; drain absorbed by the run) | +3.87 / +2.68 total; +0.58 / +0.67 inside the run | **measured** | `results/q3_ab/ab_f64.json`, `results/q3_ab/ab_f200.json` (`A1−A2`, `A4−A3`, `L1−L2`) | confirmed as a payload property (frozen artifact pays it too) |
| 5 | **Upload fixed-cost / per-op semaphore overhead** (size-independent; f64 > f200 at 3.1× fewer bytes) | f64 3.80 / f200 1.33 (batch-of-10 micro) | **measured** as an effect; **unknown** as a mechanism | `results/q6_floors/rerun_1/floor_f64.json` (`upload_micro`), `results/q1_decomp/cprofile_f64.txt` (`_wait_copy_semaphore` 1.84 ms/call) | closed as fixed-cost; allocator/pool explanation untested |
| 6 | **Readback** (`_to_host` D2H + assembly + fitness H2D) | 0.12 / 0.12 timed; ≈0.13 / −0.11 by removal | **measured (small)** | `results/q1_decomp/decomp_f64.json`, `results/q6_floors/rerun_1/floor_f64.json` (`M3→M4`) | confirmed *not* major; not a quotable floor |
| 7 | **f200 row-scaling term** | +0.94 (64→128 rows); +14.63 µs/row R² 0.96 | **unknown / OPEN** | `results/q2_gpu/rows/f64_r8.json`…`results/q2_gpu/rows/f200_r128.json`, `results/q2_gpu/rows/rerun_1/*.json` | open: two protocols disagree (+12 % vs +70 %) |
| 8 | **Compile / cache churn inside a warm call** | 0.079 / 0.077 (warm `build_cached`); 10.84 / 11.28 s cold | **measured** | `results/q1_decomp/decomp_f64.json`, `results/q1_decomp/decomp_f200.json`, `results/q1_decomp/SUMMARY.md` §Q7 | **FALSIFIED** as a warm-call contributor (24 hits / 1 miss) |
| — | *(not a mechanism)* harness/probe artifacts | — | — | `results/q8_minimal/minimal_f64.json`, `results/q8_minimal/minimal_f200.json` | **FALSIFIED** (6.117 / 6.010 ms without harness or probe) |

---

## 9. What would close what (quantified floors and levers)

| lever (what is removed) | floor f64 | vs legacy 0.6461 | floor f200 | vs legacy 0.6372 | what remains |
|---|---|---|---|---|---|
| full `problem.evaluate` (M0) | 7.2289 | 11.19× | 5.0725 | 7.96× | everything |
| host glue: plan/decode + 7 `_prep_column` + 10 payload uploads (M3) | **3.3752** | **5.22×** | **2.8064** | **4.40×** | the graph run + readback/assembly + prologue |
| + readback D2H, loss assembly, fitness H2D (M4) | 3.2411 | 5.02× | 2.9185 | 4.58× | the graph run + prologue |
| value artifact, no readback (M4V) | 2.8233 | 4.37× | 3.0286 | 4.75× | the graph run + prologue |
| pure only, frozen `run()` (M5) | **2.3849** | **3.69×** | **2.4497** | **3.84×** | the ~310-op graph call itself |

Ratios recomputed from `results/q6_floors/rerun_1/floor_f64.json`, `results/q6_floors/rerun_1/floor_f200.json`
against the same-window legacy anchors `results/q4_window/q4_leg64.json`, `results/q4_window/q4_leg200.json`; the
phase's own statements (5.22×/4.40× host glue, 3.69×/3.84× pure) are in `results/q6_floors/SUMMARY.md`.

* **Host-glue removal is the biggest available lever**: f64 7.2289 → 3.3752 ms (4.2576 ms of it the upload step
  alone) and f200 5.0725 → 2.8064 ms (1.8827 ms uploads) — yet the call stays **5.22× / 4.40×** legacy, so it
  cannot close the gap alone (`results/q6_floors/SUMMARY.md`).
* **Readback on top buys ≈0.1 ms** (M4 = 3.2411 / 2.9185 ms; on f200 M4 is *above* M3 by 0.11 ms inside the
  bracket), because the loss variant reads back only `(pop,)` floats
  (`results/q6_floors/rerun_1/floor_f200.json`).
* **Staged-payload reuse** (the frozen runner's regime: `M5`, `A3`, `L1−L2`) is worth **+3.55 / +2.50 ms** (frozen
  artifact) to **+4.16 / +2.52 ms** (loss artifact) and collapses most of the host glue — the scheme the pure
  floor 2.3849 / 2.4497 ms represents, still **3.69× / 3.84×** legacy
  (`results/q3_ab/SUMMARY.md`, `results/q6_floors/SUMMARY.md`).
* **The engine prologue around the run** is worth **0.44 / 0.58 ms** (`M4V→M5`) — larger than its timers claim,
  but secondary. The **loss-vs-value artifact** is worth 0.42 ms on f64 but *negative* on f200, and the in-graph
  loss reduction costs +0.60 / +0.76 ms per graph call (`L2−A3`) — fixture-dependent, not a general lever
  (`results/q6_floors/SUMMARY.md` §"Findings", `results/q3_ab/SUMMARY.md`).
* **Legacy parity (0.646 / 0.637 ms) requires collapsing the GPU side**: 3 device ops (1 memset + 1 fused
  tree-walk kernel + 1 averaging kernel) vs the dense route's **310** (≈**103×** more). At the pure-forward floor
  the remaining ~2.4–2.5 ms sits over a 0.64 ms target, so only pushing the ~310 chained ops toward a few
  wider/fused dispatches — *plus* removing the per-call host glue — reaches the legacy range
  (`results/q2_gpu/SUMMARY.md` §B.2/§B.3, `results/q6_floors/SUMMARY.md`).
* **Not levers**: compile/cache churn (warm 0.08 ms) and the harness (falsified); the runtime-visible content of
  the pure-forward number is the graph's op count/chain depth plus per-call staging, not benchmark plumbing.

---

## 10. Open items / unanswerables

1. **f200 row scaling (+14.63 µs/row, R² 0.96) is unexplained and the two protocols disagree**: one sync pair
   per call reads 3.17 ms at 128 rows vs 2.82 ms at 64 (**+12 %**), the tight-block protocol reads **+70 %**
   (1.345 → 2.406 ms; rerun 1.423 → 2.281 ms). Dispatch structure, element count and transient size are all
   contradicted by the dumps (same 253 dispatches / 127 kernels / chain 127 at r128 as at r64). Recorded as
   **measured jump, unexplained**; closing it needs a working GPU timeline on the f200 shape or an ETL-side
   async-submission fix (`results/q2_gpu/SUMMARY.md` §C, `results/q2_gpu/dumps/f200_r128.json`,
   `results/q2_gpu/rows/f200_r128.json`). The f64 series is flat (a = −0.24 µs/row, R² 0.12).
2. **f200 absolutes carry an opening-bracket penalty**: `M0R` repeats M0 at **+65.95 %** on f200 (+26.24 % on
   f64) while the closing `M3R` reproduces to −1.35 % / +0.97 % — within-chain deltas are robust, cross-arm f200
   *absolute* comparisons are not (`results/q6_floors/SUMMARY.md` §"Drift").
3. **The new-side end-to-end cell is the least reproducible number in the round**: window vs control differ by
   +26.9 % on the same command/ETL/fixture ~8.6 min apart — hence the band (≈8–11×)
   (`results/q4_window/SUMMARY.md` §4/§9).
4. **The loss-vs-value artifact ordering is fixture-dependent**: `M4→M4V` gains 0.4179 ms on f64 but loses
   0.1102 ms on f200 (`results/q6_floors/SUMMARY.md` §"Findings"/§"Floors").
5. **Two Q1 timer attributions do not survive controlled removal** (the three corrections): decode/plan and
   readback are real work but not quotable floors on the warm path, while uploads and the engine prologue are
   *larger* than their timers; the attribution stands structurally (uploads > run > decode > assembly), not
   step-by-step (`results/q6_floors/SUMMARY.md` §"Findings").
6. **The per-launch split (host launch vs device gap) is unmeasurable here**: nsys import fails
   (`Unknown runtime API function index: 440`), ncu refuses on driver 535.247.01, and kineto inflates the wall
   (2.55 ms profiled vs 1.21–1.47 ms unprofiled) without separating the two — "why ~5 µs per launch rather than
   ~2–3 µs" is **inferred, not measured** (`results/tool_recon/README.md`,
   `results/tool_recon/import_log.txt`, `results/q2_gpu/SUMMARY.md` §D).
7. **The legacy side's internal split is unknown**: with no nsys its 0.6461 / 0.6372 ms cannot be divided into
   launch vs execution; the naive 215 µs per device op is a whole-call property
   (`results/q2_gpu/SUMMARY.md` §B.3, `results/q4_window/q4_leg64.json`).
8. **No legacy pure row exists** — the legacy engine has no separable pure forward, so the pure/legacy ratios are
   the only pure-side comparison (r4's `--task eval-pure` skips the legacy side, exit 3;
   `results/q4_window/SUMMARY.md` §1).
9. **The upload fixed cost's mechanism is unknown**: closed as a size-independent per-op effect (each isolated
   ≤512 B upload ~0.1–0.3 ms; f64 batch 3.80 ms > f200 1.33 ms at 3.1× fewer bytes), but the candidate
   explanations (IREE HAL allocator pool per size class, queue depth at allocation, staging-buffer churn) were
   not tested (`results/q6_floors/SUMMARY.md` §"Upload anomaly", `results/q1_decomp/SUMMARY.md` Q5).
10. **The r4-revision shadow bridge is not resolvable**: the `87c7562` cell reads 6.0165 ms, inside the new-side
    spread (+11.8 % over the window's primary cell, −11.9 % under the control's)
    (`results/q4_window/q4_eval64_iree_shadow87c7562.json`, `results/q4_window/SUMMARY.md` §5).
11. **`--stage-probe` on the dense route is degenerate** (cost lands in `misc`, `build_calls=0`): the probe cell
    is a probe-tax bridge only, and its `run_ms`/`misc_ms` split (2.0805 / 4.70 ms) is a bridge datum, not a Q1
    result (`results/q4_window/q4_eval64_iree_probe.json`, `results/q4_window/SUMMARY.md` §3/§10.4).
12. **f200 replication/control spread bounds single-cell f200 deltas to ~±0.6 ms**: `R2−R1` (explicit fence, must
    be ~0) reads −0.6415 ms on f200 and the `A3`-vs-plain instrumentation control −0.5575 ms there
    (`results/q3_ab/ab_f200.json`, `results/q3_ab/SUMMARY.md` §"Deltas").
