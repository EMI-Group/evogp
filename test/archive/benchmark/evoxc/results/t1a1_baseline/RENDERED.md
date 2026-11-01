# evoxc benchmark summary: main

Source: `test/benchmark/evoxc/results/t1a1_baseline/main` — **29 records**, 6 cells x 5 arms.

The campaign ran **25 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **622.7 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-30T18:30:18Z |
| git SHA (`tree_root_sha`) | 71f8337b9cf96030fdc77e67faffdc6a9bf374eb |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A5/src` |
| python | 3.12.3 |
| torch | 2.13.0+rocm7.1 |
| numpy | 2.5.4 |
| hip | 7.1.52802 |
| iree_base_compiler | 3.11.0 |
| iree_base_runtime | 3.11.0 |
| evoxc_version | 0.0.0+unknown |
| evoxc_path | /root/evoxc/evoxc/__init__.py |
| gpu | AMD Instinct MI300X VF |
| gpu_total_memory_gb | 191.7 |
| `EVOXC_DISABLE_LOCATIONS` | `1` |

## Workload: sr_fitness

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto-iree | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 31.550 (31.377) | 4.073 (4.049) | 1.141 (1.121) | 1.125 (1.099) | 3.727 (3.595) | auto/iree → dense (evoxc-auto-iree) | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 30.511 (30.374) | 4.494 (4.465) | 2.041 (2.030) | 2.030 (2.014) | 3.729 (3.627) | auto/iree → dense (evoxc-auto-iree) | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 123.938 (123.082) | 4.854 (4.846) | 5.716 (5.696) | 5.714 (5.708) | 7.465 (7.356) | auto/iree → dense (evoxc-auto-iree) | torch-batched |
| `p2048-L128-i32-N1024-d7` | 584.902 (581.804) | 26.485 (26.474) | 105.650 (105.511) | - | 20.900 (20.577) | auto/iree → dense (evoxc-auto-iree) | evoxc-batch-iree |
| `p256-L64-i32-N4096` | 31.231 (30.812) | 4.984 (4.977) | 5.834 (5.791) | 5.821 (5.799) | 5.253 (5.185) | auto/iree → dense (evoxc-auto-iree) | torch-batched |
| `p4096-L64-i32-N1024` | 493.894 (488.786) | 10.533 (10.512) | 36.864 (36.850) | 37.247 (37.224) | 21.408 (21.302) | auto/iree → dense (evoxc-auto-iree) | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, dispatch_formation=auto, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto-iree | 27.64x | 3.57x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 28.05x | 3.62x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 8.46x | 1.09x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 14.95x | 2.20x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 15.03x | 2.21x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 8.18x | 1.21x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 21.68x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 21.69x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 16.60x | 0.65x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto-iree | 5.54x | 0.25x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 27.99x | 1.27x |
| `p256-L64-i32-N4096` | evoxc-auto-iree | 5.35x | 0.85x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 5.37x | 0.86x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 5.95x | 0.95x |
| `p4096-L64-i32-N1024` | evoxc-auto-iree | 13.40x | 0.29x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 13.26x | 0.28x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 23.07x | 0.49x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto-iree | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 24.132 (24.101) | 4.004 (3.991) | 1.777 (1.718) | 1.746 (1.693) | 3.377 (3.307) | auto/iree → dense (evoxc-auto-iree) | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 23.673 (23.317) | 4.429 (4.410) | 2.865 (2.846) | 2.810 (2.745) | 3.332 (3.298) | auto/iree → dense (evoxc-auto-iree) | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 94.515 (93.523) | 4.841 (4.830) | 7.726 (7.493) | 7.698 (7.551) | 7.222 (6.782) | auto/iree → dense (evoxc-auto-iree) | torch-batched |
| `p2048-L128-i32-N1024-d7` | 544.413 (541.954) | 26.449 (26.411) | 110.344 (109.944) | - | 19.684 (19.585) | auto/iree → dense (evoxc-auto-iree) | evoxc-batch-iree |
| `p256-L64-i32-N4096` | 23.911 (23.790) | 4.960 (4.936) | 7.993 (7.944) | 7.832 (7.678) | 5.088 (5.045) | auto/iree → dense (evoxc-auto-iree) | torch-batched |
| `p4096-L64-i32-N1024` | 373.329 (372.097) | 10.514 (10.497) | 46.954 (46.281) | 45.968 (45.872) | 21.984 (21.920) | auto/iree → dense (evoxc-auto-iree) | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, dispatch_formation=auto, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto-iree | 13.58x | 2.25x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 13.82x | 2.29x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 7.15x | 1.19x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 8.26x | 1.55x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.43x | 1.58x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 7.11x | 1.33x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 12.23x | 0.63x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 12.28x | 0.63x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 13.09x | 0.67x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto-iree | 4.93x | 0.24x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 27.66x | 1.34x |
| `p256-L64-i32-N4096` | evoxc-auto-iree | 2.99x | 0.62x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 3.05x | 0.63x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 4.70x | 0.97x |
| `p4096-L64-i32-N1024` | evoxc-auto-iree | 7.95x | 0.22x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 8.12x | 0.23x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 16.98x | 0.48x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto-iree | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |
| `p256-L64-i32-N1024` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |
| `p1000-L64-i32-N1024` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |
| `p2048-L128-i32-N1024-d7` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |
| `p256-L64-i32-N4096` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |
| `p4096-L64-i32-N1024` | - | - | - | - | - | auto/iree → dense (evoxc-auto-iree) | - |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, dispatch_formation=auto, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-batch-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | - | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto-iree | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | - | - | - | - | - |
| `p256-L64-i32-N1024` | - | - | - | - | - |
| `p1000-L64-i32-N1024` | - | - | - | - | - |
| `p2048-L128-i32-N1024-d7` | - | - | - | - | - |
| `p256-L64-i32-N4096` | - | - | - | - | - |
| `p4096-L64-i32-N1024` | - | - | - | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto-iree | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | - | - | - | - | - |
| `p256-L64-i32-N1024` | - | - | - | - | - |
| `p1000-L64-i32-N1024` | - | - | - | - | - |
| `p2048-L128-i32-N1024-d7` | - | - | - | - | - |
| `p256-L64-i32-N4096` | - | - | - | - | - |
| `p4096-L64-i32-N1024` | - | - | - | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-batch-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3440.499 | 1 | 1.141 |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3452.442 | 1 | 1.125 |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 3944.496 | 0 | 3.727 |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6608.983 | 1 | 2.041 |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6677.471 | 1 | 2.030 |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 4069.768 | 0 | 3.729 |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 6633.731 | 1 | 5.716 |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6744.411 | 1 | 5.714 |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 16684.535 | 0 | 7.465 |
| `p2048-L128-i32-N1024-d7` | evoxc-auto-iree | 14460.520 | 1 | 105.650 |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 164062.935 | 0 | 20.900 |
| `p256-L64-i32-N4096` | evoxc-auto-iree | 6608.586 | 1 | 5.834 |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 6590.938 | 1 | 5.821 |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 4341.014 | 0 | 5.253 |
| `p4096-L64-i32-N1024` | evoxc-auto-iree | 6664.529 | 1 | 36.864 |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 6812.752 | 1 | 37.247 |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 154885.081 | 0 | 21.408 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto-iree, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | batch_forward | `5a5f1f21d8bcc368f51ad33cc6897c87` | evoxc-batch-iree | 1024 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto-iree, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | batch_forward | `e8b4fe6f9366304d0efff31f203c42bb` | evoxc-batch-iree | 0 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto-iree, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `23ef6765a57472ec1f9c0f90687c0d7e` | evoxc-batch-iree | 3072 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `75caff0777c227066b76c4cb3e2e7342` | torch-per-tree, torch-batched | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `5a21db2c9c85d97de609c629c3352166` | evoxc-auto-iree | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `a060aee61a79e255f6d04db37118790f` | evoxc-batch-iree | 45056 |
| `p256-L64-i32-N4096` | batch_forward | `1298e7d5d2ecdaf398cefb2ff41cb491` | torch-per-tree, torch-batched, evoxc-auto-iree, evoxc-dense-iree | 0 |
| `p256-L64-i32-N4096` | batch_forward | `78e147ddc3a5edc9fd340187b107156d` | evoxc-batch-iree | 0 |
| `p4096-L64-i32-N1024` | batch_forward | `7fe6de16612212715d9d9aefa8029a31` | torch-per-tree, torch-batched | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `4c4d512caf5e1dfa0aa11dcd416c7b59` | evoxc-auto-iree, evoxc-dense-iree | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `2ae4b39fc2544c1494b7c227112be210` | evoxc-batch-iree | 27648 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 4/5, p256-L64-i32-N1024: 4/5, p1000-L64-i32-N1024: 4/5, p2048-L128-i32-N1024-d7: 4/4, p256-L64-i32-N4096: 4/5, p4096-L64-i32-N1024: 4/5.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
