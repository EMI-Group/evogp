# evoxc benchmark summary: small-r5

Source: `test/benchmark/evoxc/results/batch/small-r5` — **36 records**, 6 cells x 6 arms.

The campaign ran **36 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **865.4 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=1` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T08:52:26Z |
| git SHA (`tree_root_sha`) | 6f99c0ca8901b034a20c5157550cf67e2c2fa881 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A31/src` |
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
| `p16-L32-i8-N1024` | 1.867 (1.691) | 2.877 (2.857) | 2.467 (2.397) | 2.308 (2.273) | 0.988 (0.954) | 3.939 (3.924) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 7.465 (7.121) | 3.680 (3.662) | 0.974 (0.961) | 9.865 (9.078) | 0.984 (0.964) | 1.131 (1.112) | auto/numpy → dense | evoxc-auto |
| `p16-L64-i32-N64` | 1.868 (1.852) | 3.159 (3.021) | 2.170 (2.095) | 2.212 (2.169) | 1.547 (1.539) | 3.933 (3.849) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 7.418 (7.358) | 4.035 (4.025) | 1.503 (1.480) | 9.572 (9.437) | 1.410 (1.384) | 0.979 (0.965) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 28.173 (27.659) | 4.525 (4.274) | 1.434 (1.419) | 36.463 (34.752) | 1.442 (1.408) | 2.067 (2.027) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N64` | 117.159 (115.847) | 4.548 (4.537) | 1.555 (1.545) | 142.507 (139.603) | 1.613 (1.559) | 6.082 (6.061) | auto/numpy → dense | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.76x | 1.17x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.81x | 1.25x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.89x | 2.91x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.47x | 0.73x |
| `p64-L32-i8-N1024` | evoxc-auto | 7.67x | 3.78x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.76x | 0.37x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 7.58x | 3.74x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 6.60x | 3.25x |
| `p16-L64-i32-N64` | evoxc-auto | 0.86x | 1.46x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.84x | 1.43x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.21x | 2.04x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.48x | 0.80x |
| `p64-L64-i32-N64` | evoxc-auto | 4.94x | 2.68x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.77x | 0.42x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 5.26x | 2.86x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 7.58x | 4.12x |
| `p256-L64-i32-N64` | evoxc-auto | 19.65x | 3.16x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.77x | 0.12x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 19.54x | 3.14x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 13.63x | 2.19x |
| `p1000-L64-i32-N64` | evoxc-auto | 75.32x | 2.92x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.82x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 72.66x | 2.82x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 19.26x | 0.75x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.376 (1.294) | 2.812 (2.799) | 2.303 (2.160) | 2.443 (2.245) | 1.021 (0.998) | 3.796 (3.743) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.702 (5.676) | 3.625 (3.617) | 1.172 (1.148) | 9.960 (9.153) | 1.201 (1.171) | 1.028 (1.022) | auto/numpy → dense | evoxc-batch-iree |
| `p16-L64-i32-N64` | 1.444 (1.431) | 3.060 (2.870) | 2.065 (2.048) | 2.159 (2.112) | 1.431 (1.400) | 3.893 (3.796) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.656 (5.315) | 4.002 (3.990) | 1.422 (1.361) | 8.674 (8.644) | 1.422 (1.402) | 0.789 (0.776) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 21.081 (20.869) | 4.067 (4.030) | 1.482 (1.452) | 36.509 (35.441) | 1.473 (1.454) | 1.451 (1.445) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L64-i32-N64` | 89.062 (88.273) | 4.471 (4.436) | 1.762 (1.743) | 139.760 (138.756) | 1.772 (1.756) | 4.154 (4.052) | auto/numpy → dense | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.60x | 1.22x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.56x | 1.15x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.35x | 2.75x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.36x | 0.74x |
| `p64-L32-i8-N1024` | evoxc-auto | 4.86x | 3.09x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.57x | 0.36x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 4.75x | 3.02x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 5.55x | 3.53x |
| `p16-L64-i32-N64` | evoxc-auto | 0.70x | 1.48x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.67x | 1.42x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.01x | 2.14x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.37x | 0.79x |
| `p64-L64-i32-N64` | evoxc-auto | 3.98x | 2.81x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.65x | 0.46x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 3.98x | 2.81x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 7.16x | 5.07x |
| `p256-L64-i32-N64` | evoxc-auto | 14.23x | 2.74x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.58x | 0.11x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 14.32x | 2.76x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 14.53x | 2.80x |
| `p1000-L64-i32-N64` | evoxc-auto | 50.56x | 2.54x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 50.25x | 2.52x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 21.44x | 1.08x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.457 (1.432) | 2.786 (2.617) | 2.272 (2.216) | 2.114 (2.022) | 0.630 (0.606) | 5.036 (4.954) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.956 (5.796) | 3.541 (3.501) | 0.653 (0.643) | 8.669 (8.413) | 0.676 (0.647) | 0.631 (0.598) | auto/numpy → dense | evoxc-batch-iree |
| `p16-L64-i32-N64` | 1.348 (1.324) | 2.908 (2.829) | 2.028 (1.992) | 2.370 (2.119) | 0.858 (0.845) | 5.023 (4.996) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.855 (5.789) | 3.902 (3.653) | 0.839 (0.822) | 9.158 (8.330) | 0.868 (0.858) | 0.631 (0.615) | auto/numpy → dense | evoxc-batch-iree |
| `p256-L64-i32-N64` | 22.009 (21.528) | 4.252 (4.020) | 0.879 (0.862) | 36.707 (33.347) | 0.854 (0.839) | 0.783 (0.755) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L64-i32-N64` | 91.217 (90.042) | 4.347 (4.342) | 0.879 (0.845) | 142.815 (139.379) | 0.838 (0.807) | 1.375 (1.306) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.64x | 1.23x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.69x | 1.32x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 2.31x | 4.42x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.29x | 0.55x |
| `p64-L32-i8-N1024` | evoxc-auto | 9.11x | 5.42x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.69x | 0.41x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 8.81x | 5.24x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 9.44x | 5.61x |
| `p16-L64-i32-N64` | evoxc-auto | 0.66x | 1.43x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.57x | 1.23x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.57x | 3.39x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.27x | 0.58x |
| `p64-L64-i32-N64` | evoxc-auto | 6.98x | 4.65x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.43x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6.75x | 4.50x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 9.28x | 6.19x |
| `p256-L64-i32-N64` | evoxc-auto | 25.03x | 4.84x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.60x | 0.12x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 25.77x | 4.98x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 28.09x | 5.43x |
| `p1000-L64-i32-N64` | evoxc-auto | 103.80x | 4.95x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 108.92x | 5.19x |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 66.32x | 3.16x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.179 | 5.115 | 8.044 | 8.207 | 3015.831 | 782.156 |
| `p64-L32-i8-N1024` | 5.554 | 7.819 | 4.841 | 35.830 | 4.853 | 937.231 |
| `p16-L64-i32-N64` | 3.339 | 6.011 | 7.133 | 7.883 | 5791.780 | 753.797 |
| `p64-L64-i32-N64` | 5.997 | 10.136 | 5.370 | 104.641 | 5.310 | 1053.640 |
| `p256-L64-i32-N64` | 18.530 | 27.761 | 6.039 | 309.694 | 6.149 | 3628.131 |
| `p1000-L64-i32-N64` | 54.093 | 60.966 | 8.223 | 1019.972 | 8.252 | 15088.596 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 0.913 / 2.265 (op 71%) | 2.756 / 2.360 (op 46%) | 5.628 / 2.417 (op 30%) | 5.717 / 2.490 (op 30%) | 3012.940 / 2.891 (op 0%) | 779.400 / 2.756 (op 0%) |
| `p64-L32-i8-N1024` | 3.166 / 2.389 (op 43%) | 5.354 / 2.465 (op 32%) | 2.449 / 2.392 (op 49%) | 33.638 / 2.192 (op 6%) | 2.447 / 2.406 (op 50%) | 934.580 / 2.651 (op 0%) |
| `p16-L64-i32-N64` | 0.940 / 2.399 (op 72%) | 3.635 / 2.376 (op 40%) | 4.944 / 2.190 (op 31%) | 5.540 / 2.343 (op 30%) | 5788.954 / 2.826 (op 0%) | 751.047 / 2.749 (op 0%) |
| `p64-L64-i32-N64` | 3.650 / 2.348 (op 39%) | 7.830 / 2.305 (op 23%) | 2.996 / 2.374 (op 44%) | 102.341 / 2.300 (op 2%) | 2.876 / 2.434 (op 46%) | 1050.916 / 2.724 (op 0%) |
| `p256-L64-i32-N64` | 16.238 / 2.292 (op 12%) | 25.325 / 2.436 (op 9%) | 3.542 / 2.497 (op 41%) | 307.257 / 2.437 (op 1%) | 3.647 / 2.502 (op 41%) | 3625.272 / 2.859 (op 0%) |
| `p1000-L64-i32-N64` | 51.537 / 2.556 (op 5%) | 58.249 / 2.717 (op 4%) | 5.755 / 2.469 (op 30%) | 1017.149 / 2.824 (op 0%) | 5.777 / 2.475 (op 30%) | 15085.533 / 3.064 (op 0%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.40x | 0.64x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.39x | 0.62x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.00x | 0.00x |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 0.00x | 0.01x |
| `p64-L32-i8-N1024` | evoxc-auto | 1.15x | 1.62x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.16x | 0.22x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 1.14x | 1.61x |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 0.01x | 0.01x |
| `p16-L64-i32-N64` | evoxc-auto | 0.47x | 0.84x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.42x | 0.76x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.00x | 0.00x |
| `p16-L64-i32-N64` | evoxc-batch-iree | 0.00x | 0.01x |
| `p64-L64-i32-N64` | evoxc-auto | 1.12x | 1.89x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.06x | 0.10x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 1.13x | 1.91x |
| `p64-L64-i32-N64` | evoxc-batch-iree | 0.01x | 0.01x |
| `p256-L64-i32-N64` | evoxc-auto | 3.07x | 4.60x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.06x | 0.09x |
| `p256-L64-i32-N64` | evoxc-dense-iree | 3.01x | 4.51x |
| `p256-L64-i32-N64` | evoxc-batch-iree | 0.01x | 0.01x |
| `p1000-L64-i32-N64` | evoxc-auto | 6.58x | 7.41x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6.56x | 7.39x |
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
| `p16-L32-i8-N1024` | evoxc-auto | 81.053 | 13 | 2.467 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 85.536 | 13 | 2.308 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3382.648 | 1 | 0.988 |
| `p16-L32-i8-N1024` | evoxc-batch-iree | 3696.609 | 13 | 3.939 |
| `p64-L32-i8-N1024` | evoxc-auto | 3350.166 | 1 | 0.974 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 221.058 | 57 | 9.865 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3405.275 | 1 | 0.984 |
| `p64-L32-i8-N1024` | evoxc-batch-iree | 2862.761 | 0 | 1.131 |
| `p16-L64-i32-N64` | evoxc-auto | 62.626 | 13 | 2.170 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 66.248 | 13 | 2.212 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6479.169 | 1 | 1.547 |
| `p16-L64-i32-N64` | evoxc-batch-iree | 3628.095 | 13 | 3.933 |
| `p64-L64-i32-N64` | evoxc-auto | 6283.653 | 1 | 1.503 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 199.207 | 57 | 9.572 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6177.736 | 1 | 1.410 |
| `p64-L64-i32-N64` | evoxc-batch-iree | 2173.491 | 0 | 0.979 |
| `p256-L64-i32-N64` | evoxc-auto | 6267.958 | 1 | 1.434 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 801.316 | 219 | 36.463 |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6345.641 | 1 | 1.442 |
| `p256-L64-i32-N64` | evoxc-batch-iree | 9938.791 | 0 | 2.067 |
| `p1000-L64-i32-N64` | evoxc-auto | 6308.120 | 1 | 1.555 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3135.215 | 821 | 142.507 |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6397.350 | 1 | 1.613 |
| `p1000-L64-i32-N64` | evoxc-batch-iree | 158047.449 | 0 | 6.082 |

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
