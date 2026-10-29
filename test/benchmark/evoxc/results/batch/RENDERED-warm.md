# evoxc benchmark summary: warm

Source: `results/batch/warm` — **24 records**, 4 cells x 6 arms.

The campaign ran **24 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **979.1 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=1` untimed calls and `reps=3` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T07:49:09Z |
| git SHA (`tree_root_sha`) | 94e9e7a00382e5bbfbe2428b2556d33f29b00222 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A30/src` |
| python | 3.14.8 |
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

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 30.738 (30.733) | 4.018 (3.966) | 1.146 (1.120) | 40.052 (38.439) | 1.126 (1.109) | 2.226 (2.171) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 30.116 (29.936) | 4.395 (4.384) | 2.053 (2.039) | 39.491 (37.781) | 2.049 (2.047) | 2.167 (2.128) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L32-i8-N1024` | 120.291 (119.922) | 4.429 (4.424) | 1.894 (1.883) | 148.861 (145.524) | 1.876 (1.866) | 5.157 (5.137) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 117.118 (113.234) | 4.848 (4.846) | 5.710 (5.705) | 150.154 (145.794) | 5.704 (5.703) | 6.090 (6.085) | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 26.81x | 3.50x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.77x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 27.31x | 3.57x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 13.81x | 1.81x |
| `p256-L64-i32-N1024` | evoxc-auto | 14.67x | 2.14x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.76x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 14.70x | 2.15x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 13.89x | 2.03x |
| `p1000-L32-i8-N1024` | evoxc-auto | 63.51x | 2.34x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.81x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 64.12x | 2.36x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 23.33x | 0.86x |
| `p1000-L64-i32-N1024` | evoxc-auto | 20.51x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.78x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 20.53x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 19.23x | 0.80x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 23.387 (22.950) | 3.869 (3.819) | 1.785 (1.767) | 39.716 (38.751) | 1.923 (1.904) | 2.165 (2.078) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 22.911 (22.276) | 4.324 (4.315) | 2.866 (2.864) | 38.946 (37.654) | 2.854 (2.730) | 2.171 (2.055) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L32-i8-N1024` | 91.766 (90.647) | 4.412 (4.410) | 3.713 (3.667) | 148.857 (147.620) | 3.805 (3.785) | 5.932 (5.905) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N1024` | 92.125 (88.933) | 4.819 (4.784) | 7.469 (7.342) | 146.574 (144.297) | 7.492 (7.457) | 5.895 (5.768) | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 13.11x | 2.17x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 12.16x | 2.01x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 10.80x | 1.79x |
| `p256-L64-i32-N1024` | evoxc-auto | 8.00x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.59x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.03x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 10.55x | 1.99x |
| `p1000-L32-i8-N1024` | evoxc-auto | 24.71x | 1.19x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.62x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 24.12x | 1.16x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 15.47x | 0.74x |
| `p1000-L64-i32-N1024` | evoxc-auto | 12.33x | 0.65x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.63x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 12.30x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 15.63x | 0.82x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 23.777 (23.592) | 3.883 (3.850) | 0.669 (0.664) | 36.938 (36.368) | 0.666 (0.653) | 0.816 (0.790) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 23.422 (23.325) | 4.288 (4.257) | 0.928 (0.916) | 35.041 (34.888) | 0.900 (0.887) | 0.753 (0.753) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L32-i8-N1024` | 93.082 (91.917) | 3.624 (3.588) | 0.667 (0.663) | 140.870 (140.186) | 0.669 (0.640) | 1.570 (1.533) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N1024` | 93.802 (93.799) | 4.353 (4.314) | 0.868 (0.851) | 133.052 (132.895) | 0.824 (0.809) | 1.296 (1.271) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 35.53x | 5.80x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.64x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 35.71x | 5.83x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 29.14x | 4.76x |
| `p256-L64-i32-N1024` | evoxc-auto | 25.24x | 4.62x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 26.03x | 4.76x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 31.08x | 5.69x |
| `p1000-L32-i8-N1024` | evoxc-auto | 139.60x | 5.44x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 139.14x | 5.42x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 59.29x | 2.31x |
| `p1000-L64-i32-N1024` | evoxc-auto | 108.12x | 5.02x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.71x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 113.78x | 5.28x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 72.36x | 3.36x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 17.652 | 22.742 | 5.322 | 278.848 | 5.573 | 3639.501 |
| `p256-L64-i32-N1024` | 17.394 | 22.356 | 6.823 | 263.132 | 6.694 | 3919.813 |
| `p1000-L32-i8-N1024` | 58.667 | 55.301 | 7.479 | 1017.251 | 7.175 | 16953.686 |
| `p1000-L64-i32-N1024` | 56.880 | 68.818 | 12.710 | 913.770 | 12.539 | 17721.598 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 15.154 / 2.498 (op 14%) | 20.178 / 2.564 (op 11%) | 2.883 / 2.439 (op 46%) | 276.421 / 2.427 (op 1%) | 3.020 / 2.553 (op 46%) | 3636.634 / 2.868 (op 0%) |
| `p256-L64-i32-N1024` | 14.811 / 2.583 (op 15%) | 19.804 / 2.552 (op 11%) | 4.297 / 2.526 (op 37%) | 260.620 / 2.512 (op 1%) | 4.194 / 2.501 (op 37%) | 3916.720 / 3.094 (op 0%) |
| `p1000-L32-i8-N1024` | 56.119 / 2.549 (op 4%) | 52.760 / 2.541 (op 5%) | 4.953 / 2.526 (op 34%) | 1014.334 / 2.917 (op 0%) | 4.761 / 2.414 (op 34%) | 16950.629 / 3.057 (op 0%) |
| `p1000-L64-i32-N1024` | 54.239 / 2.641 (op 5%) | 66.317 / 2.501 (op 4%) | 10.234 / 2.476 (op 19%) | 910.791 / 2.979 (op 0%) | 10.082 / 2.457 (op 20%) | 17718.663 / 2.935 (op 0%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 3.32x | 4.27x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3.17x | 4.08x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 0.00x | 0.01x |
| `p256-L64-i32-N1024` | evoxc-auto | 2.55x | 3.28x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.07x | 0.08x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 2.60x | 3.34x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 0.00x | 0.01x |
| `p1000-L32-i8-N1024` | evoxc-auto | 7.84x | 7.39x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.05x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 8.18x | 7.71x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 0.00x | 0.00x |
| `p1000-L64-i32-N1024` | evoxc-auto | 4.48x | 5.41x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 4.54x | 5.49x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 0.00x | 0.00x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | -0.084430 | -0.084430 | -0.084430 | -0.084430 | -0.084430 | -0.084430 |
| `p256-L64-i32-N1024` | -0.084812 | -0.084812 | -0.084812 | -0.084812 | -0.084812 | -0.084812 |
| `p1000-L32-i8-N1024` | -0.084133 | -0.084133 | -0.084133 | -0.084133 | -0.084133 | -0.084133 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 3406.972 | 1 | 1.146 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 836.408 | 212 | 40.052 |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3360.522 | 1 | 1.126 |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 11156.313 | 0 | 2.226 |
| `p256-L64-i32-N1024` | evoxc-auto | 6554.981 | 1 | 2.053 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 843.533 | 219 | 39.491 |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6544.647 | 1 | 2.049 |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 12313.055 | 0 | 2.167 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3399.458 | 1 | 1.894 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3284.581 | 797 | 148.861 |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3468.752 | 1 | 1.876 |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 91039.136 | 0 | 5.157 |
| `p1000-L64-i32-N1024` | evoxc-auto | 6479.169 | 1 | 5.710 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3149.264 | 821 | 150.154 |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6428.943 | 1 | 5.704 |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 166872.866 | 0 | 6.090 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | batch_forward | `fd77ba5e8c3dc0d49c5225b2031b75f5` | evoxc-batch-iree | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L32-i8-N1024` | forward | `39f9d20e6ee43cbd19d90b7b1cff83c8` | evoxc-batch-iree | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p1000-L32-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `69f8b425f050934d448d300ecf26d656` | evoxc-batch-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-dense-iree | 7 |
| `p1000-L32-i8-N1024` | forward | `431f4fd79a790f3b93c4e9e85afeee3b` | evoxc-batch-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `8569c9182600e194030145073a49321b` | evoxc-batch-iree | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p1000-L64-i32-N1024` | forward | `3d80e2b6ce635b32257cded7638bd50c` | evoxc-batch-iree | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 5/6, p256-L64-i32-N1024: 5/6, p1000-L32-i8-N1024: 5/6, p1000-L64-i32-N1024: 5/6.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
