# Population scaling 64 → 8192: the four routes along one axis
This round asks one question with records: on a FIXED shape, how does each evaluation route's cost scale with the population, from pop 64 to pop 8192 — 128× the population, one doubling at a time.
Every other column of this harness moves several axes at once (`main_cells()` crosses population × tree length × input length, `large_cells()` mixes population, rows, node count and input width), so no ratio between two of their cells is the population axis alone.
Here the shape never moves: `tree_len` 32, `input_len` 8, `rows` 1024, depth cap 4 — which is also the `evogp.gp` acceptance config (`max_nodes` 32, `input_len` 8, the `+ - * /` set, layer cap 4, `N` = 1024) — and only `pop` doubles.
The answer in one line: the shipped default (`auto` → the fused Triton route on this box) is the cheapest arm at **every** one of the eight populations on all three value paths and on the whole GP generation, its cost grows **1.56×–2.42×** across the 128× population range, while the compiled `dense` route grows **40×–48×** (with a step discontinuity at pop 8192) and `torch-batched` grows 3.67×–3.84×; the win over the plain-torch batched evaluator therefore *widens* with population, from **42.0× / 59.0× / 55.6×** at pop 64 (`sr_fitness` / `batch_forward` / `forward`) to **82.1× / 93.8× / 43.8×** at pop 8192.
Box: one AMD Instinct MI300X VF (191.7 GB HBM), torch 2.13.0+rocm7.1, HIP 7.1.52802, IREE 3.11.0, numpy 2.5.4, python 3.12.3, `EVOXC_DISABLE_LOCATIONS=1`.
Provenance: tree `dbd0a52` / evoxc `474bf72` (`/root/evoxc`, the checkout that carries the native DLPack device export) on every record of the round; `results/popscale/main/campaign.json` carries the driver's own `driver_env` block, and each record carries its own `env`.
Protocol of record: `--warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --workloads sr_fitness,batch_forward,forward,gp_step`, one process per `(cell, arm)` behind the idle-device guard, sequential, a 900 s per-config cap.
All 40 configs of the headline column finished `ok` (`guard.gpu_pids.idle true`, `contaminated false`, `gpu_use_pct` 0.0 on every entry), in **461.0 s** total.
Nothing below was re-measured for this report: every number is read from a committed record under `results/popscale/`.

## 1. The column and the arms
`common.popscale_cells()` (wired into `campaign.py` as `--cells popscale`) is eight cells, population doubling, every other shape field held fixed:

| cell id | pop | tree_len | input_len | rows | depth cap | dense intermediate (`pop·tree_len·rows·4`) |
|---|---|---|---|---|---|---|
| `p64-L32-i8-N1024-d4` | 64 | 32 | 8 | 1024 | 4 | 8 MiB |
| `p128-L32-i8-N1024-d4` | 128 | 32 | 8 | 1024 | 4 | 16 MiB |
| `p256-L32-i8-N1024-d4` | 256 | 32 | 8 | 1024 | 4 | 32 MiB |
| `p512-L32-i8-N1024-d4` | 512 | 32 | 8 | 1024 | 4 | 64 MiB |
| `p1024-L32-i8-N1024-d4` | 1 024 | 32 | 8 | 1024 | 4 | 128 MiB |
| `p2048-L32-i8-N1024-d4` | 2 048 | 32 | 8 | 1024 | 4 | 256 MiB |
| `p4096-L32-i8-N1024-d4` | 4 096 | 32 | 8 | 1024 | 4 | 512 MiB |
| `p8192-L32-i8-N1024-d4` | 8 192 | 32 | 8 | 1024 | 4 | 1 GiB |

The depth cap of 4 means the generated trees carry ≤ 15 nodes (`2 ** 4 - 1`), so they sit far inside both the 32-slot row stride and dense's default trace cap of 64 — every dense record of the column proves that (`dense_route.supported true`, `max_tree_len_cap 64`, `tree_len_max 15`), i.e. **no cell declined into the per-tree `specialized` fallback**.
The memory rule holds for every cell with 8× margin (`p8192` is exactly 1 GiB against the 8 GiB `CELL_TENSOR_CEILING_BYTES`), which is why the top population is schedulable at all.
Five arms, 8 × 5 = **40 records**:

| arm | what it measures |
|---|---|
| `evoxc-auto` | the SHIPPED default; every record resolves `eval_mode auto` / `engine fused` / `backend iree` (`would_use_fused true`) |
| `evoxc-fused-iree` | the explicit `EVOGP_EVAL_MODE=fused` request — the same Triton route, asked for by name |
| `evoxc-dense-iree` | the compiled all-ops graph (one `evoxc.build_cached` graph per shape per workload) |
| `torch-batched` | the plain-PyTorch vectorized whole-population evaluator (the honest baseline) |
| `torch-per-tree` | the plain-PyTorch per-tree Python walk (the slow baseline; it costs little here and it anchors the ratio) |

The pop-64 `evoxc-auto` record was produced by a separate single-config invocation of the same protocol before the main run and reused via `--resume`; the other 39 configs are one invocation (`campaign.json` `driver_env`).
The diagnostic column `results/popscale/diag_dense_scaling/` (6 records, §3.3) was a second invocation.

## 2. Per-population warm medians
Warm/steady-state `median_ms` per workload; the cold first call is never part of these (it is §4).

### 2.1 `sr_fitness` (the training hot path, loss + readback)
| pop | torch-per-tree | torch-batched | `evoxc-auto` | `evoxc-fused-iree` | `evoxc-dense-iree` |
|---|---|---|---|---|---|
| 64 | 4.7749 | 2.1275 | 0.0507 | 0.0602 | 0.9964 |
| 128 | 9.9055 | 2.1120 | 0.0562 | 0.0611 | 1.0361 |
| 256 | 19.7574 | 2.2275 | 0.0539 | 0.0528 | 1.1454 |
| 512 | 41.4885 | 2.1953 | 0.0574 | 0.0534 | 1.3366 |
| 1 024 | 75.8904 | 2.3645 | 0.0540 | 0.0495 | 1.9294 |
| 2 048 | 166.5328 | 3.1829 | 0.0630 | 0.0592 | 3.4231 |
| 4 096 | 343.9665 | 4.6963 | 0.0705 | 0.0718 | 6.1502 |
| 8 192 | 674.9805 | 7.8074 | 0.0951 | 0.0940 | 40.2331 |

### 2.2 `batch_forward` (the whole-population value path, `(pop, N, out)`)
| pop | torch-per-tree | torch-batched | `evoxc-auto` | `evoxc-fused-iree` | `evoxc-dense-iree` |
|---|---|---|---|---|---|
| 64 | 3.0969 | 2.0252 | 0.0343 | 0.0346 | 1.2006 |
| 128 | 6.7350 | 2.0782 | 0.0327 | 0.0365 | 1.3621 |
| 256 | 12.6481 | 2.1662 | 0.0399 | 0.0355 | 1.8920 |
| 512 | 25.8703 | 2.1632 | 0.0406 | 0.0389 | 2.4949 |
| 1 024 | 49.7791 | 2.3453 | 0.0399 | 0.0441 | 3.9794 |
| 2 048 | 108.5356 | 3.1669 | 0.0442 | 0.0491 | 6.8441 |
| 4 096 | 213.9667 | 4.6802 | 0.0568 | 0.0612 | 12.7409 |
| 8 192 | 444.7336 | 7.7791 | 0.0829 | 0.0836 | 57.3357 |

### 2.3 `forward` (the per-row value path, `(pop, out)`: tree `i` sees row `i`)
| pop | torch-per-tree | torch-batched | `evoxc-auto` | `evoxc-fused-iree` | `evoxc-dense-iree` |
|---|---|---|---|---|---|
| 64 | 3.3667 | 2.0359 | 0.0366 | 0.0337 | 0.6617 |
| 128 | 6.7087 | 2.1310 | 0.0307 | 0.0356 | 0.6515 |
| 256 | 12.9030 | 2.2142 | 0.0353 | 0.0325 | 0.6399 |
| 512 | 26.9018 | 2.0712 | 0.0419 | 0.0360 | 0.6618 |
| 1 024 | 51.6868 | 2.0011 | 0.0366 | 0.0357 | 0.6657 |
| 2 048 | 112.5562 | 1.9094 | 0.0359 | 0.0406 | 0.6548 |
| 4 096 | 220.5358 | 2.0882 | 0.0399 | 0.0423 | 0.7009 |
| 8 192 | 456.8212 | 2.0500 | 0.0468 | 0.0477 | 0.7041 |

### 2.4 Speedup over the plain-torch baselines, per pop
`torch-*` median ÷ arm median, same cell, same session (>1 = the torch baseline is slower).
`sr_fitness`:

| pop | `auto` vs `torch-batched` | `fused` vs `torch-batched` | `auto` vs `torch-per-tree` | `dense` vs `torch-batched` |
|---|---|---|---|---|
| 64 | 42.0× | 35.3× | 94.2× | 2.14× |
| 128 | 37.6× | 34.6× | 176.3× | 2.04× |
| 256 | 41.3× | 42.2× | 366.6× | 1.94× |
| 512 | 38.2× | 41.1× | 722.8× | 1.64× |
| 1 024 | 43.8× | 47.8× | 1 405.4× | 1.23× |
| 2 048 | 50.5× | 53.8× | 2 643.4× | 0.93× |
| 4 096 | 66.6× | 65.4× | 4 879.0× | 0.76× |
| 8 192 | 82.1× | 83.1× | 7 097.6× | 0.19× |

`batch_forward`: 59.0× / 63.6× / 54.3× / 53.3× / 58.8× / 71.6× / 82.4× / **93.8×** (`auto` vs batched, pop 64 → 8 192) and 90.3× → **5 364.7×** vs per-tree; `dense` vs batched falls 1.69× → 0.14×.
`forward`: 55.6× / 69.4× / 62.7× / 49.4× / 54.7× / 53.2× / 52.3× / **43.8×** (`auto` vs batched) and 92.0× → **9 761.1×** vs per-tree; `dense` vs batched stays 3.08×–2.91× (this is the one workload where the compiled route is flat in pop, because the per-row path only ever touches `pop` rows).
So the evoxc route is not "fast for an engine": on the loss path it is **42×–82×** the plain-PyTorch whole-population evaluator across the entire 64–8192 range, and the multiple grows with population because the torch arm tracks `pop` while the kernel does not.

### 2.5 The whole-generation view (`gp_step`: evaluate + select + crossover + mutate)
One full generation, warm, `median_total_ms` with the evaluation slice and its share of the generation:

| pop | per-tree | batched | `auto` total | `fused` total | `dense` total | `auto` eval | `auto` operators | eval share | batched ÷ `auto` | dense ÷ `auto` |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 4.949 | 6.326 | **2.801** | 2.774 | 4.748 | 0.446 | 2.355 | 15.9 % | 2.3× | 1.7× |
| 128 | 8.217 | 11.951 | **2.929** | 2.877 | 5.182 | 0.443 | 2.486 | 15.1 % | 4.1× | 1.8× |
| 256 | 14.783 | 14.940 | **2.856** | 2.937 | 5.320 | 0.439 | 2.417 | 15.4 % | 5.2× | 1.9× |
| 512 | 25.861 | 20.458 | **2.873** | 2.902 | 6.020 | 0.419 | 2.454 | 14.6 % | 7.1× | 2.1× |
| 1 024 | 49.855 | 42.412 | **2.943** | 2.903 | 7.668 | 0.450 | 2.493 | 15.3 % | 14.4× | 2.6× |
| 2 048 | 105.135 | 92.588 | **2.931** | 2.913 | 10.133 | 0.466 | 2.464 | 15.9 % | 31.6× | 3.5× |
| 4 096 | 199.069 | 179.616 | **2.981** | 3.009 | 17.249 | 0.463 | 2.518 | 15.5 % | 60.3× | 5.8× |
| 8 192 | 413.343 | 359.246 | **2.986** | 3.149 | 59.122 | 0.473 | 2.513 | 15.9 % | 120.3× | 19.8× |

Three things to read off this table:
1. **A generation is flat on the shipped default** — 2.801 → 2.986 ms over 128× the population (1.07×), with `gp_step.compile_misses_total` 0 on every record: nothing is re-planned, re-traced or recompiled per generation.
2. **Evaluation is a bounded slice of it** — 0.419–0.473 ms, i.e. **14.6–15.9 %** of a generation, flat in pop. The operator layer (2.355–2.518 ms, 84 %+) is the loop's bottleneck at every population in this column, exactly as `results/t1a1_auto_r3/` reported at pop 256–4 096 on a different shape.
3. **The loop-level win grows with population**: 2.3× at pop 64, 120.3× at pop 8192 against `torch-batched` — because the torch arm's generation is 359.2 ms while the engine's evaluation slice barely moves.
The `dense` arm keeps its own compile-clean loop (`compile_misses_total` 0) but its generation grows 4.748 → 59.122 ms (12.5×), i.e. 19.8× the shipped default at pop 8192.
`best_fitness` after the timed steps agrees across arms within 1e-4 (`-0.092110` at pop 64, `-0.084252` from pop 1 024 up), so the scales below are not pricing a different computation.

## 3. The population scaling law of each route
`8192 / 64` ratio of the warm median (128× the population), per route and workload:

| route | `sr_fitness` | `batch_forward` | `forward` | `gp_step` |
|---|---|---|---|---|
| `evoxc-auto` (fused) | **1.88×** | **2.42×** | **1.28×** | **1.07×** |
| `evoxc-fused-iree` (fused) | **1.56×** | **2.42×** | **1.42×** | **1.14×** |
| `evoxc-dense-iree` | 40.38× | 47.76× | 1.06× | 12.45× |
| `torch-batched` | 3.67× | 3.84× | 1.01× | 56.79× |
| `torch-per-tree` | 141.36× | 143.61× | 135.69× | 83.52× |

### 3.1 The fused route: a launch floor plus a per-tree term, flat in the population
Fit of the eight points against `pop` (least squares over the column):

| workload | intercept (launch floor) | slope | implied at pop 8 192 |
|---|---|---|---|
| `sr_fitness` | 0.0531 ms | 4.71 ns/tree | 0.092 ms (measured 0.094) |
| `batch_forward` | 0.0358 ms | 5.96 ns/tree | 0.084 ms (measured 0.084) |
| `forward` | 0.0345 ms | 1.72 ns/tree | 0.049 ms (measured 0.048) |

At that slope a doubling of `batch_forward` from pop 8 192 would take another ~13 600 trees (≈ pop 21 800), and the whole 64 → 8 192 column costs at most **+0.05 ms** on any workload.
The `forward` path is the flattest (1.72 ns/tree) because it evaluates `pop` trees against `pop` rows — its work tracks the diagonal, not `pop × rows`.
**Where it stops being flat** is not the arithmetic but the memory envelope:
* the scratch table grows strictly linearly with pop — 8 396 800 B at pop 64 (131 200 B/tree), 1 074 790 400 B (1.07 GB) at pop 8 192, 2.15 GB at pop 16 384 (§3.3) — against `EVOGP_FUSED_MAX_SCRATCH_MB` 8 192, so at this shape the route would decline at ≈ pop 65 400;
* the fused probe never declined anywhere in the column: `fused_declines 0` and `fused_kernel_launches 24` (of 24 calls) on **all 16 fused records** (§5);
* the knee the measured range does show is a mild super-linearity past pop 4 096 (`batch_forward` 0.0568 → 0.0829 = 1.46× for 2× pop), i.e. per-program work starts to matter before memory does.
And the floor is why the small end is not flat *downwards*: at pop 64 the kernel is already 0.034–0.051 ms, i.e. `EVOGP_FUSED_MIN_POP` 16 and the reference box's launch floor, not a throughput crossover (`results/minpop_r3/SUMMARY.md` re-took that column and found the kernel flat from pop 1).

### 3.2 The compiled `dense` route: linear in `pop × rows` at ~0.046 ns/element, then a step
Per-element cost of a warm dense call (`pop × tree_len × rows` = the size of its largest intermediate):

| pop | elements | `sr_fitness` | ns/element | `batch_forward` | ns/element | intermediate |
|---|---|---|---|---|---|---|
| 64 | 2.1 M | 0.996 | 0.475 | 1.201 | 0.572 | 8 MiB |
| 128 | 4.2 M | 1.036 | 0.247 | 1.362 | 0.325 | 16 MiB |
| 256 | 8.4 M | 1.145 | 0.137 | 1.892 | 0.226 | 32 MiB |
| 512 | 16.8 M | 1.337 | 0.080 | 2.495 | 0.149 | 64 MiB |
| 1 024 | 33.6 M | 1.929 | 0.058 | 3.979 | 0.119 | 128 MiB |
| 2 048 | 67.1 M | 3.423 | 0.051 | 6.844 | 0.102 | 256 MiB |
| 4 096 | 134.2 M | 6.150 | 0.046 | 12.741 | 0.095 | 512 MiB |
| 8 192 | 268.4 M | 40.233 | **0.150** | 57.336 | **0.214** | **1 GiB** |

Up to pop 4 096 the route is textbook linear in `pop × rows` (0.051 → 0.046 ns/element) with a small-population overhead tail below pop 512; at pop 8 192 the per-element cost steps **3.3×** for a 2× population increase.
`forward` never steps (0.640–0.704 ms over the whole column) because it touches only `pop` rows — so the step is specific to the full-population loss/value paths, i.e. to the cells whose largest intermediate crosses 512 MiB → 1 GiB.

### 3.3 Where the dense step comes from (diagnostic column, `results/popscale/diag_dense_scaling/`)
Three extra cells move the axes separately, same protocol, same arms:

| cell | elements | largest intermediate | `dense` `sr_fitness` | ns/element | `dense` `batch_forward` | `fused` `sr_fitness` | `fused` scratch |
|---|---|---|---|---|---|---|---|
| `p8192-L32-i8-N512-d4` | 134.2 M | 512 MiB | 6.148 | 0.046 | 12.901 | 0.073 | 537 919 488 |
| `p8192-L32-i8-N1024-d4` (main) | 268.4 M | 1 GiB | 40.233 | 0.150 | 57.336 | 0.094 | 1 074 790 400 |
| `p8192-L32-i8-N2048-d4` | 536.9 M | 2 GiB | 80.160 | 0.149 | 115.251 | 0.147 | 2 148 532 224 |
| `p16384-L32-i8-N1024-d4` | 536.9 M | 2 GiB | 59.261 | 0.110 | 94.499 | 0.158 | 2 149 580 800 |

The step is a property of the **size of the largest intermediate**, not of the population: halving `rows` at pop 8 192 (512 MiB) restores exactly the pop-4 096 per-element cost (6.148 ms at 134.2 M elements against 6.150 ms there, 0.046 ns/element on both), while doubling `rows` (2 GiB) keeps the stepped cost; and at equal element count the row-heavier cell is the dearer one (`p8192-N2048` 80.160 ms against `p16384-N1024` 59.261 ms).
The knee sits between 512 MiB and 1 GiB, and past it per-element cost is 2.4×–3.3× the linear value and stays roughly flat (0.110–0.149 ns/element).
The **fused route has no such step** — 0.073 → 0.158 ms while its largest intermediate grows 4× (537 MiB → 2.15 GiB) — because it keeps one value buffer plus a scratch table instead of one `pop × tree_len × rows` plane.
Mechanism of the dense step is *not* established here (an allocation/allocator effect is the plausible reading: the same total work at 512 MiB is 3.3× cheaper per element than at 1 GiB, and the two axes are interchangeable); what the records establish is the knee's location and that it is the dense route's, not the engine's.

### 3.4 The torch baselines
`torch-per-tree` is linear in pop (141.36× / 143.61× / 135.69× over 128× pop — a Python tree walk pays per tree, as expected) and 674.98 ms on the loss path at pop 8 192.
`torch-batched` grows 3.67× / 3.84× over the column — sublinear over 128× because a vectorized GPU pass amortizes its launch, but still a real per-population cost; and its `forward` is flat (1.01×, 2.05 ms) because tree `i` sees row `i`: the work is `pop`, not `pop × rows`.
At the top of the column `torch-batched` is still 2.0500 / 7.7791 / 7.8074 ms — i.e. 24.5× / 93.1× / 83.1× the fused kernel; the fused route reverses the single-call axis that `LARGE_SCALE_ROUND2.md` found `torch-batched` winning against every compiled evoxc route.

## 4. The cold column: what a first call costs, and when it is repaid
`first_call_ms` of the `sr_fitness` workload (the first evaluation of a fresh process — the only place a compile can land), with the warm median beside it:

| pop | `evoxc-auto` first call | `evoxc-fused-iree` first call | `evoxc-dense-iree` first call | `dense` compile misses | warm `auto` | warm `dense` |
|---|---|---|---|---|---|---|
| 64 | 1 786.0 | 1 492.5 | 3 220.8 | 1 | 0.0507 | 0.9964 |
| 128 | 477.4 | 492.1 | 3 356.4 | 1 | 0.0562 | 1.0361 |
| 256 | 486.5 | 501.6 | 3 359.0 | 1 | 0.0539 | 1.1454 |
| 512 | 480.7 | 489.0 | 3 347.8 | 1 | 0.0574 | 1.3366 |
| 1 024 | 476.3 | 490.6 | 3 423.1 | 1 | 0.0540 | 1.9294 |
| 2 048 | 501.2 | 490.0 | 3 375.5 | 1 | 0.0630 | 3.4231 |
| 4 096 | 477.6 | 490.2 | 3 406.9 | 1 | 0.0705 | 6.1502 |
| 8 192 | 483.1 | 471.1 | 3 361.3 | 1 | 0.0951 | 40.2331 |

* **The fused route pays a one-time Triton JIT per kernel specialization, never a per-population compile.** Each process builds 3 specializations (`fused_kernel_compiles 3` on every fused record — the loss kernel and the two value kernels), and `cache.compile_misses_first_call` is **0** on every record of the column: no evoxc graph is built at any population. The round's first process (pop-64 `evoxc-auto`, empty `TRITON_CACHE_DIR`) paid **1 786 ms**; the pop-64 `evoxc-fused-iree` process, entering with a partially warmed disk cache, **1 492 ms**; every later fused process reads **471–502 ms**, which is what remains when the Triton build is a disk-cache hit — process setup (torch/HIP context) plus the three specialization lookups, and it does not grow with pop (477 ms at pop 128, 483 ms at pop 8 192).
* **The dense route pays a per-shape compile, per workload.** `compile_misses_first_call` is 1 on each workload and `engine_cache_final.misses` is 3 per process, at **2 694–3 423 ms** each — so the three value workloads of one cell cost ≈ 9–10 s of compile, and it is *flat in pop* (3 220.8 ms at pop 64, 3 361.3 ms at pop 8 192): the compile is a shape+function-set cost, not a workload cost.
* **Amortization.** Repaying the fused cold start against `torch-batched` on the loss path takes `cold / (batched − auto)` calls: **860** at pop 64 with an empty Triton cache, **232 / 224 / 225 / 206 / 161 / 103 / 63** at pops 128 → 8 192 with a warm one — i.e. from a few hundred generations at the small end to 63 calls at the top. The dense route's ~3.3 s compile is never repaid at all in this column: its warm loss call is already 19.7× (pop 64) to 423.1× (pop 8 192) the fused one, so the compiled route loses on both axes.
* Read the cold column as "± a factor of a few" (see §5.4); the *structure* — one JIT per specialization against one compile per shape — is the robust part.

## 5. Route honesty
### 5.1 The launch proof, per record
A fused call and a silent fallback look identical in wall time, so the proof is the kernel counter, not the timing. All 16 records whose arm may reach the fused route (`evoxc-auto`, `evoxc-fused-iree`) carry the surface (`fused_cache.available true`) and read:

| pop | `fused_calls` | `fused_kernel_launches` | `fused_declines` | `fused_kernel_compiles` | `served_every_workload` |
|---|---|---|---|---|---|
| 64 … 8 192 (all eight) | 24 | 24 | **0** | 3 | **true** |

i.e. 24 calls = the three value workloads × 8 each (1 cold + 2 warmup + 5 reps) inside the recorded window, zero declines, and `routes[*].engine` = `fused` / `backend` = `iree` on every workload of every `evoxc-auto` record — the shipped policy really did serve this column with the kernel.
The counter window closes after the three value workloads, so the `gp_step` calls of the same record are outside it (`stats_after.fused_calls` equals `totals.fused_calls` at 24).
### 5.2 The dense route served every cell
`dense_route.supported true`, `max_tree_len_cap 64`, `tree_len_max 15`, `reason null` on all eight dense records, and the worker's own direct probe (`dense_route.probe_ms`) tracks the timed median (1.1158 ms at pop 64 against 0.9964; 40.2844 ms at pop 8 192 against 40.2331) — so the measured `evoxc-dense-iree` arm is the compiled graph, not the per-structure fallback, and `auto_trace_cap` reports `cap_pinned false`, `base_cap 64`, `auto_max_tree_len 256`: the cap and its raise never engaged, because no population here carries a tree longer than 15 nodes.
### 5.3 Digests: one split class, and it is the documented one
`check.py` compares the two value paths across arms. Pops 64–512 are bit-identical across **all five** arms. Pops 1 024 / 2 048 / 4 096 / 8 192 split into exactly two groups — `{torch-per-tree, torch-batched}` against `{evoxc-auto, evoxc-dense-iree, evoxc-fused-iree}` — on both `batch_forward` and `forward`, and the two groups agree on NaN placement element-for-element (`nan` 1 024 / 6 144 / 17 408 / 41 984 in the same cell, and 0 at pops 64–512 where the single shared digest also holds).
That is the corner case this column newly puts under the digest check: the population first contains division-by-zero trees at pop 1 024, and from there the two runtimes differ only in the **NaN payload bit** (the class `VALIDATION.md` and this node's CONTEXT.md already document: dense differs from numpy only on the `sub` payload bit, and the fused route only in payload bits, with finite-space `max_abs 0.0`). It is declared with `--allow-digest-mismatch` and it is value-neutral; no finite value disagrees.
`sr_fitness` is not sha-compared across arms (the loss reduction order differs by route; 4 distinct digests among 5 records in every cell) — the gate checks NaN/±inf placement there, and `best_fitness` after the timed steps agrees to 1e-4 (§2.5).
### 5.4 Noise and the box
Warm medians on this shared box are ±10 %, and a contended window inflates them 2–3×. Three observations bracket this round's noise: the two fused arms (`evoxc-auto` and `evoxc-fused-iree`, the same route and the same kernel) differ by up to **19 %** in a cell (pop 64, 0.0507 against 0.0602) and by 1–12 % elsewhere, in both directions — so a single-cell comparison below ~20 % is not a signal; the `min_ms` sits 5–9 % under the median throughout; and the guard recorded `gpu_pids.idle true` / `contaminated false` / `gpu_use_pct 0.0` on all 39 driver entries plus the 6 diagnostic ones. Conclusions are drawn from ratios that are many times this band (38×–94× on the loss and value paths, and 1.56× against 40× on the same workload range), never from a single-cell percentage.

## 6. Verdict
**The evoxc route keeps beating the plain-PyTorch batched evaluator across the entire 64–8192 population range, and the margin grows with population.** On the loss path the shipped default reads 0.0507 ms at pop 64 and 0.0951 ms at pop 8 192 (1.88× for 128× the population) against `torch-batched`'s 2.1275 → 7.8074 ms (3.67×), i.e. **42.0× at the small end and 82.1× at the top**; on the whole-population value path 59.0× → **93.8×**; on the per-row value path 55.6× → 43.8× (the torch arm is flat there, so its multiple shrinks while staying > 43×); against the per-tree Python walk it is 94.2× → **7 097.6×** on the loss path. A whole GP generation is **2.801 → 2.986 ms** (1.07×), 2.3× → **120.3×** faster than the torch-batched loop, with evaluation a constant 14.6–15.9 % slice of it — the operator layer, not the engine, is what a 8 192-tree generation now spends its time on.
**Nothing in this column is cheaper than the fused route at any population.** The nearest non-fused arm is the compiled `dense` graph on the per-row `forward` path (0.640–0.704 ms, 14–19× dearer and flat in pop); on the loss and value paths dense is 19.7×–691.6× dearer (loss path 19.7× → 423.1×), and past pop 4 096 its own cost steps 3.3× per element where the kernel does not. So the answer to "what is the largest pop where anything is cheaper than the fused route" is: **none has been found in 64 → 8 192** — the fused route is the cheapest arm at every cell of the column on every workload, and its own limits are the ~0.035–0.053 ms launch floor at the small end and the 8 GiB scratch envelope (≈ pop 65 400 at this shape) at the large end, not a competitor.

## 7. Reproduce and gate
```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton_cache            # writable: this box's $HOME is read-only
mkdir -p "$TRITON_CACHE_DIR"
PY=/tmp/rocvenv/bin/python

# the headline column (40 configs, 8 cells x 5 arms, one process per config)
$PY test/benchmark/evoxc/campaign.py \
    --out test/benchmark/evoxc/results/popscale/main \
    --cells popscale \
    --arms evoxc-auto,evoxc-fused-iree,evoxc-dense-iree,torch-batched,torch-per-tree \
    --evoxc-root /root/evoxc \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 \
    --workloads sr_fitness,batch_forward,forward,gp_step --timeout 900

# the diagnostic column (3 cells x 2 arms: the dense knee)
$PY test/benchmark/evoxc/campaign.py \
    --out test/benchmark/evoxc/results/popscale/diag_dense_scaling \
    --cells p8192-L32-i8-N512-d4,p8192-L32-i8-N2048-d4,p16384-L32-i8-N1024-d4 \
    --arms evoxc-dense-iree,evoxc-fused-iree --evoxc-root /root/evoxc \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 \
    --workloads sr_fitness,batch_forward,forward,gp_step --timeout 900

# gate + render (GPU-free)
$PY test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/popscale/main \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/popscale/diag_dense_scaling \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/popscale/main \
    --out test/benchmark/evoxc/results/popscale/main/RENDERED.md
$PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/popscale/diag_dense_scaling \
    --out test/benchmark/evoxc/results/popscale/diag_dense_scaling/RENDERED.md
```
Gates of record:
* `results/popscale/main` — **40 records / 1 121 checks / 0 failures**, 8 declared WARNs (the `batch_forward` + `forward` digest splits of pops 1 024 / 2 048 / 4 096 / 8 192, §5.3), guard `39/39` configs on an idle device, 0 contaminated, elapsed 461.0 s.
* `results/popscale/diag_dense_scaling` — **6 records / 191 checks / 0 failures**, no WARN, guard `6/6` idle.
* Machine renderings: `results/popscale/main/RENDERED.md`, `results/popscale/diag_dense_scaling/RENDERED.md`.

## 8. Scope: what this round did not schedule, and why
* **The opt-in `batch` route is deliberately absent.** It is already characterized (`BATCH.md`, `LARGE_SCALE.md`, `LARGE_SCALE_ROUND2.md`) as a fixed-population evaluator that loses any generation loop, and its merged-graph compile is ~quadratic in emitted ops (800 s at pop 8 192, 89 % of the 900 s cap) — a per-population sweep of it would spend the round's whole budget in the compile column to re-confirm a settled verdict. `auto` never selects it.
* **`evoxc-specialized-iree` is absent.** It compiles one graph per DISTINCT tree structure at ~272–284 ms per structure, so pop 8 192 would be tens of minutes of compile per workload; the harness's own `SCOPE` rule restricts it to pops 16/64 anyway. Its scaling behaviour is the fallback story, not the shipped one, and this column never declines onto it (§5.2).
* Both are stated in `campaign.json` (`protocol.batch_arms`, `protocol.scope_rule`) rather than silently omitted; `scope_dropped` is empty because every scheduled `(cell, arm)` config ran.
* Not measured here: `EVOGP_FUSED_*` geometry tuning along the population axis (settled as a flat ±5 % surface in `src/evogp/tree/FUSED.md`), the artifact disk tier (a compile lever, and the compiled route loses on the warm axis anyway), multi-output / `CustomLoss` and the robotics problems (never timed by any round), and pops above 8 192 beyond the single `p16384` diagnostic cell.

## 9. Where the brief and the records disagree
1. **"dense should track `pop × tree_len × rows`"** — it does, at 0.046–0.051 ns/element, but only up to a 512 MiB largest intermediate; the pop-8 192 cell of this very column sits past the knee and reads 3.3× per element (§3.2/§3.3). A linear model fitted on pops 64–4 096 under-predicts the top cell by ~6×.
2. **"fused should be near-flat in pop"** — it is flat in the *arithmetic* (a launch floor plus 4.7–6.0 ns/tree) but not literally constant: the column reads 1.56×–2.42× end to end, the `forward` path being the flattest and `batch_forward` the least flat. "Near-flat" means "sub-logarithmic in pop", not "constant".
3. **"`torch-batched` tracks pop"** — it grows only 3.67×–3.84× over 128× the population on the two full-population paths and is *flat* on `forward` (1.01×), because `forward` evaluates tree `i` against row `i`. Its 2.05 ms at pop 8 192 is entirely launch/setup amortized over `pop` rows.
4. **The pop-8 192 `sr_fitness` value for `dense` (40.2331 ms) is not a typo and not contamination** — the record's `samples_ms` are 40.1351–40.4101, the worker's independent `dense_route.probe_ms` reads 40.2844, and the diagnostic column reproduces the level at two other cells of the same intermediate size. It is the knee of §3.2, and it is why the `dense` ÷ `auto` loop ratio at that pop jumps to 19.8×.
