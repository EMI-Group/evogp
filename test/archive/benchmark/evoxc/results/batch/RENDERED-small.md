# evoxc benchmark summary: small

Source: `results/batch/small` — **36 records**, 6 cells x 6 arms.

The campaign ran **36 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **868.1 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=1` untimed calls and `reps=3` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T08:09:54Z |
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
| `p16-L32-i8-N1024` | 2.016 (1.972) | 2.809 (2.793) | 2.574 (2.498) | 2.230 (2.176) | 0.984 (0.946) | 3.858 (3.838) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 7.515 (7.472) | 4.055 (3.978) | 1.137 (1.094) | 9.637 (9.598) | 1.050 (1.043) | 1.229 (1.169) | auto/numpy → dense | evoxc-dense-iree |
| `p16-L64-i32-N64` | 1.863 (1.806) | 3.287 (3.257) | 2.391 (2.358) | 2.380 (2.329) | 1.485 (1.479) | 4.203 (3.974) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 7.367 (7.315) | 4.492 (4.457) | 1.555 (1.536) | 8.543 (8.501) | 1.461 (1.448) | 1.079 (1.030) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 30.457 (29.990) | 4.432 (4.403) | 1.372 (1.336) | 37.353 (37.265) | 1.440 (1.431) | 2.104 (2.086) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N64` | 122.843 (122.523) | 4.487 (4.453) | 1.558 (1.552) | 142.232 (139.022) | 1.554 (1.539) | 6.127 (6.084) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.78x | 1.09x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.90x | 1.26x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 2.05x | 2.86x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.52x | 0.73x |
| `p64-L32-i8-N1024` | evoxc-auto | 6.61x | 3.57x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.42x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 7.16x | 3.86x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 6.12x | 3.30x |
| `p16-L64-i32-N64` | evoxc-auto | 0.78x | 1.37x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.78x | 1.38x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.25x | 2.21x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.44x | 0.78x |
| `p64-L64-i32-N64` | evoxc-auto | 4.74x | 2.89x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.86x | 0.53x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 5.04x | 3.08x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 6.82x | 4.16x |
| `p256-L64-i32-N64` | evoxc-auto | 22.19x | 3.23x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.82x | 0.12x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 21.15x | 3.08x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 14.47x | 2.11x |
| `p1000-L64-i32-N64` | evoxc-auto | 78.83x | 2.88x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.86x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 79.04x | 2.89x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 20.05x | 0.73x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.520 (1.504) | 2.804 (2.753) | 2.428 (2.390) | 2.440 (2.244) | 1.014 (0.992) | 3.854 (3.850) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.723 (5.683) | 3.881 (3.811) | 1.173 (1.173) | 10.714 (10.634) | 1.217 (1.182) | 1.065 (1.041) | auto/numpy → dense | evoxc-batch-iree |
| `p16-L64-i32-N64` | 1.516 (1.476) | 3.207 (3.206) | 2.169 (2.135) | 2.346 (2.263) | 1.407 (1.404) | 3.922 (3.869) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.607 (5.275) | 4.104 (4.087) | 1.450 (1.429) | 8.442 (8.439) | 1.440 (1.414) | 0.871 (0.771) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 23.297 (22.529) | 4.399 (4.334) | 1.460 (1.424) | 37.475 (37.138) | 1.496 (1.478) | 1.499 (1.468) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N64` | 94.405 (94.343) | 4.319 (4.057) | 1.798 (1.707) | 141.486 (140.114) | 1.783 (1.713) | 4.011 (3.993) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.63x | 1.16x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.62x | 1.15x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.50x | 2.77x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.39x | 0.73x |
| `p64-L32-i8-N1024` | evoxc-auto | 4.88x | 3.31x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.53x | 0.36x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 4.70x | 3.19x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 5.37x | 3.64x |
| `p16-L64-i32-N64` | evoxc-auto | 0.70x | 1.48x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.65x | 1.37x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.08x | 2.28x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.39x | 0.82x |
| `p64-L64-i32-N64` | evoxc-auto | 3.87x | 2.83x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.66x | 0.49x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 3.89x | 2.85x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 6.44x | 4.71x |
| `p256-L64-i32-N64` | evoxc-auto | 15.96x | 3.01x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.62x | 0.12x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 15.58x | 2.94x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 15.54x | 2.94x |
| `p1000-L64-i32-N64` | evoxc-auto | 52.52x | 2.40x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.67x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 52.95x | 2.42x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 23.53x | 1.08x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.647 (1.623) | 2.752 (2.748) | 2.082 (2.053) | 2.351 (2.280) | 0.672 (0.652) | 5.245 (5.234) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.811 (5.788) | 3.585 (3.555) | 0.677 (0.666) | 9.325 (8.706) | 0.730 (0.716) | 0.648 (0.615) | auto/numpy → dense | evoxc-batch-iree |
| `p16-L64-i32-N64` | 1.552 (1.544) | 3.203 (3.163) | 2.099 (2.071) | 2.284 (2.222) | 0.884 (0.871) | 5.413 (5.206) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.813 (5.800) | 4.017 (3.972) | 0.889 (0.881) | 10.267 (10.037) | 0.860 (0.857) | 0.613 (0.585) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 23.684 (23.456) | 4.324 (4.324) | 0.881 (0.880) | 37.330 (37.169) | 0.857 (0.841) | 0.765 (0.725) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L64-i32-N64` | 96.626 (95.659) | 4.358 (4.051) | 0.886 (0.852) | 142.811 (134.773) | 0.878 (0.862) | 1.313 (1.312) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.79x | 1.32x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.70x | 1.17x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 2.45x | 4.09x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.31x | 0.52x |
| `p64-L32-i8-N1024` | evoxc-auto | 8.59x | 5.30x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.62x | 0.38x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 7.96x | 4.91x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 8.97x | 5.53x |
| `p16-L64-i32-N64` | evoxc-auto | 0.74x | 1.53x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.68x | 1.40x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.76x | 3.62x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.29x | 0.59x |
| `p64-L64-i32-N64` | evoxc-auto | 6.54x | 4.52x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.57x | 0.39x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6.76x | 4.67x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 9.49x | 6.56x |
| `p256-L64-i32-N64` | evoxc-auto | 26.88x | 4.91x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.63x | 0.12x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 27.64x | 5.05x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 30.97x | 5.65x |
| `p1000-L64-i32-N64` | evoxc-auto | 109.02x | 4.92x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.68x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 110.09x | 4.97x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 73.59x | 3.32x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.612 | 5.033 | 7.924 | 8.038 | 2940.313 | 760.939 |
| `p64-L32-i8-N1024` | 5.777 | 7.493 | 4.717 | 40.017 | 4.795 | 927.504 |
| `p16-L64-i32-N64` | 3.600 | 5.974 | 7.943 | 7.184 | 5671.568 | 764.624 |
| `p64-L64-i32-N64` | 6.000 | 10.172 | 5.395 | 112.315 | 5.345 | 1052.510 |
| `p256-L64-i32-N64` | 20.177 | 27.304 | 6.008 | 315.776 | 6.246 | 3587.076 |
| `p1000-L64-i32-N64` | 56.750 | 59.175 | 8.347 | 1027.014 | 8.329 | 14973.145 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.069 / 2.542 (op 70%) | 2.664 / 2.369 (op 47%) | 5.581 / 2.342 (op 30%) | 5.581 / 2.458 (op 31%) | 2937.577 / 2.736 (op 0%) | 758.236 / 2.703 (op 0%) |
| `p64-L32-i8-N1024` | 3.420 / 2.358 (op 41%) | 5.276 / 2.217 (op 30%) | 2.392 / 2.325 (op 49%) | 37.617 / 2.400 (op 6%) | 2.390 / 2.405 (op 50%) | 924.954 / 2.550 (op 0%) |
| `p16-L64-i32-N64` | 0.990 / 2.610 (op 73%) | 3.621 / 2.352 (op 39%) | 5.579 / 2.363 (op 30%) | 4.964 / 2.220 (op 31%) | 5668.803 / 2.765 (op 0%) | 761.834 / 2.790 (op 0%) |
| `p64-L64-i32-N64` | 3.599 / 2.401 (op 40%) | 7.945 / 2.227 (op 22%) | 2.921 / 2.474 (op 46%) | 109.736 / 2.580 (op 2%) | 2.949 / 2.396 (op 45%) | 1049.919 / 2.591 (op 0%) |
| `p256-L64-i32-N64` | 17.612 / 2.564 (op 13%) | 24.812 / 2.492 (op 9%) | 3.507 / 2.501 (op 42%) | 313.418 / 2.357 (op 1%) | 3.701 / 2.545 (op 41%) | 3584.278 / 2.799 (op 0%) |
| `p1000-L64-i32-N64` | 54.089 / 2.661 (op 5%) | 56.619 / 2.556 (op 4%) | 5.840 / 2.508 (op 30%) | 1024.097 / 2.917 (op 0%) | 5.874 / 2.454 (op 29%) | 14970.242 / 2.903 (op 0%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.46x | 0.64x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.45x | 0.63x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.00x | 0.00x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.00x | 0.01x |
| `p64-L32-i8-N1024` | evoxc-auto | 1.22x | 1.59x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.14x | 0.19x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 1.20x | 1.56x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 0.01x | 0.01x |
| `p16-L64-i32-N64` | evoxc-auto | 0.45x | 0.75x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.50x | 0.83x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.00x | 0.00x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.00x | 0.01x |
| `p64-L64-i32-N64` | evoxc-auto | 1.11x | 1.89x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.09x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 1.12x | 1.90x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 0.01x | 0.01x |
| `p256-L64-i32-N64` | evoxc-auto | 3.36x | 4.54x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.06x | 0.09x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 3.23x | 4.37x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 0.01x | 0.01x |
| `p1000-L64-i32-N64` | evoxc-auto | 6.80x | 7.09x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.06x | 0.06x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6.81x | 7.10x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 0.00x | 0.00x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p64-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p16-L64-i32-N64` | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 |
| `p64-L64-i32-N64` | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 |
| `p256-L64-i32-N64` | -0.090638 | -0.090638 | -0.090638 | -0.090638 | -0.090638 | -0.090638 |
| `p1000-L64-i32-N64` | -0.092798 | -0.092798 | -0.092798 | -0.092798 | -0.092798 | -0.092798 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 81.651 | 13 | 2.574 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 83.427 | 13 | 2.230 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3349.763 | 1 | 0.984 |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 3744.833 | 13 | 3.858 |
| `p64-L32-i8-N1024` | evoxc-auto | 3439.922 | 1 | 1.137 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 238.270 | 57 | 9.637 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3412.094 | 1 | 1.050 |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 2927.245 | 0 | 1.229 |
| `p16-L64-i32-N64` | evoxc-auto | 68.562 | 13 | 2.391 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 66.472 | 13 | 2.380 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6480.212 | 1 | 1.485 |
| `p16-L64-i32-N64` | evoxc-batch-iree | 3552.729 | 13 | 4.203 |
| `p64-L64-i32-N64` | evoxc-auto | 6526.263 | 1 | 1.555 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 199.239 | 57 | 8.543 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6534.685 | 1 | 1.461 |
| `p64-L64-i32-N64` | evoxc-batch-iree | 2239.548 | 0 | 1.079 |
| `p256-L64-i32-N64` | evoxc-auto | 6453.158 | 1 | 1.372 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 858.127 | 219 | 37.353 |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6594.854 | 1 | 1.440 |
| `p256-L64-i32-N64` | evoxc-batch-iree | 9887.177 | 0 | 2.104 |
| `p1000-L64-i32-N64` | evoxc-auto | 6444.238 | 1 | 1.558 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3230.183 | 821 | 142.232 |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6351.310 | 1 | 1.554 |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 157926.729 | 0 | 6.127 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | batch_forward | `ba1157e91cb09e697e821bf4caea5ee7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i8-N1024` | batch_forward | `ccd9945407c53f1df82b795467c23061` | evoxc-batch-iree | 0 |
| `p16-L32-i8-N1024` | forward | `c21e5a57d58db557165cc13d515f788f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-batch-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `960a58af4aeb7d6a855d90660a333509` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `93d4fbddbcfa25a19a1d9d6c2d1c876f` | evoxc-batch-iree | 0 |
| `p64-L32-i8-N1024` | forward | `4cfeb51431d33041bc2c71f2db3b1417` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | forward | `908aac94d387869e234090bc7ec3d1bd` | evoxc-batch-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `ac20d6b9b2fc89394ed1f5bb092e351e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `3e8dc4e8aabd0c155e65667c48b43645` | evoxc-batch-iree | 0 |
| `p16-L64-i32-N64` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-batch-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `91f8f057103e5fc3760df30ab1e15987` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `f9e4ae85f06811b98ebda8c49168e60a` | evoxc-batch-iree | 0 |
| `p64-L64-i32-N64` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | forward | `6d7d47f50dfa756b57065e746b24a4f1` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `5ddefa134a076abce9faa669870b2558` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `a2ce0ccbb45cc75ab9b0e2ee7aff3ef7` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N64` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N64` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p1000-L64-i32-N64` | batch_forward | `6891f91b5d52caf1d0ca046f766bf963` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 192 |
| `p1000-L64-i32-N64` | batch_forward | `b603b9032571ea2fc133e7985e03390e` | evoxc-auto, evoxc-dense-iree | 192 |
| `p1000-L64-i32-N64` | batch_forward | `e5e68610cfb680295ab4bd8d9ed14418` | evoxc-batch-iree | 192 |
| `p1000-L64-i32-N64` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N64` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p1000-L64-i32-N64` | forward | `3d80e2b6ce635b32257cded7638bd50c` | evoxc-batch-iree | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16-L32-i8-N1024: 5/6, p64-L32-i8-N1024: 5/6, p16-L64-i32-N64: 2/6, p64-L64-i32-N64: 3/6, p256-L64-i32-N64: 3/6, p1000-L64-i32-N64: 4/6.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
