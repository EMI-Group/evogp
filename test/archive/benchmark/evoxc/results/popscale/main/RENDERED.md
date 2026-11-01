# evoxc benchmark summary: main

Source: `test/benchmark/evoxc/results/popscale/main` — **40 records**, 8 cells x 5 arms.

The campaign ran **39 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **461.0 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-31T16:51:37Z |
| git SHA (`tree_root_sha`) | dbd0a526d6b7c89c5412c0591a4871340fc4081e |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A2/src` |
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

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | 75.890 (74.830) | 2.365 (2.350) | 0.054 (0.050) | 1.929 (1.908) | 0.050 (0.047) | auto/iree → fused | evoxc-fused-iree |
| `p128-L32-i8-N1024-d4` | 9.905 (9.385) | 2.112 (2.083) | 0.056 (0.050) | 1.036 (0.991) | 0.061 (0.053) | auto/iree → fused | evoxc-auto |
| `p2048-L32-i8-N1024-d4` | 166.533 (165.910) | 3.183 (3.170) | 0.063 (0.057) | 3.423 (3.380) | 0.059 (0.057) | auto/iree → fused | evoxc-fused-iree |
| `p256-L32-i8-N1024-d4` | 19.757 (19.624) | 2.228 (2.215) | 0.054 (0.051) | 1.145 (1.084) | 0.053 (0.044) | auto/iree → fused | evoxc-fused-iree |
| `p4096-L32-i8-N1024-d4` | 343.966 (334.820) | 4.696 (4.692) | 0.070 (0.067) | 6.150 (6.138) | 0.072 (0.066) | auto/iree → fused | evoxc-auto |
| `p512-L32-i8-N1024-d4` | 41.489 (40.124) | 2.195 (2.017) | 0.057 (0.053) | 1.337 (1.312) | 0.053 (0.051) | auto/iree → fused | evoxc-fused-iree |
| `p64-L32-i8-N1024-d4` | 4.775 (4.546) | 2.127 (2.072) | 0.051 (0.046) | 0.996 (0.984) | 0.060 (0.053) | auto/iree → fused | evoxc-auto |
| `p8192-L32-i8-N1024-d4` | 674.981 (655.096) | 7.807 (7.763) | 0.095 (0.090) | 40.233 (40.135) | 0.094 (0.090) | auto/iree → fused | evoxc-fused-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | evoxc-auto | 1405.38x | 43.79x |
| `p1024-L32-i8-N1024-d4` | evoxc-dense-iree | 39.33x | 1.23x |
| `p1024-L32-i8-N1024-d4` | evoxc-fused-iree | 1533.14x | 47.77x |
| `p128-L32-i8-N1024-d4` | evoxc-auto | 176.25x | 37.58x |
| `p128-L32-i8-N1024-d4` | evoxc-dense-iree | 9.56x | 2.04x |
| `p128-L32-i8-N1024-d4` | evoxc-fused-iree | 162.12x | 34.57x |
| `p2048-L32-i8-N1024-d4` | evoxc-auto | 2643.38x | 50.52x |
| `p2048-L32-i8-N1024-d4` | evoxc-dense-iree | 48.65x | 0.93x |
| `p2048-L32-i8-N1024-d4` | evoxc-fused-iree | 2813.05x | 53.77x |
| `p256-L32-i8-N1024-d4` | evoxc-auto | 366.56x | 41.33x |
| `p256-L32-i8-N1024-d4` | evoxc-dense-iree | 17.25x | 1.94x |
| `p256-L32-i8-N1024-d4` | evoxc-fused-iree | 374.19x | 42.19x |
| `p4096-L32-i8-N1024-d4` | evoxc-auto | 4878.96x | 66.61x |
| `p4096-L32-i8-N1024-d4` | evoxc-dense-iree | 55.93x | 0.76x |
| `p4096-L32-i8-N1024-d4` | evoxc-fused-iree | 4790.62x | 65.41x |
| `p512-L32-i8-N1024-d4` | evoxc-auto | 722.80x | 38.25x |
| `p512-L32-i8-N1024-d4` | evoxc-dense-iree | 31.04x | 1.64x |
| `p512-L32-i8-N1024-d4` | evoxc-fused-iree | 776.94x | 41.11x |
| `p64-L32-i8-N1024-d4` | evoxc-auto | 94.18x | 41.96x |
| `p64-L32-i8-N1024-d4` | evoxc-dense-iree | 4.79x | 2.14x |
| `p64-L32-i8-N1024-d4` | evoxc-fused-iree | 79.32x | 35.34x |
| `p8192-L32-i8-N1024-d4` | evoxc-auto | 7097.59x | 82.10x |
| `p8192-L32-i8-N1024-d4` | evoxc-dense-iree | 16.78x | 0.19x |
| `p8192-L32-i8-N1024-d4` | evoxc-fused-iree | 7180.64x | 83.06x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | 49.779 (49.025) | 2.345 (2.326) | 0.040 (0.038) | 3.979 (3.955) | 0.044 (0.042) | auto/iree → fused | evoxc-auto |
| `p128-L32-i8-N1024-d4` | 6.735 (6.486) | 2.078 (2.022) | 0.033 (0.031) | 1.362 (1.296) | 0.036 (0.034) | auto/iree → fused | evoxc-auto |
| `p2048-L32-i8-N1024-d4` | 108.536 (107.692) | 3.167 (3.156) | 0.044 (0.042) | 6.844 (6.794) | 0.049 (0.046) | auto/iree → fused | evoxc-auto |
| `p256-L32-i8-N1024-d4` | 12.648 (11.985) | 2.166 (2.149) | 0.040 (0.037) | 1.892 (1.800) | 0.035 (0.034) | auto/iree → fused | evoxc-fused-iree |
| `p4096-L32-i8-N1024-d4` | 213.967 (211.969) | 4.680 (4.664) | 0.057 (0.055) | 12.741 (12.660) | 0.061 (0.059) | auto/iree → fused | evoxc-auto |
| `p512-L32-i8-N1024-d4` | 25.870 (25.342) | 2.163 (2.116) | 0.041 (0.038) | 2.495 (2.455) | 0.039 (0.037) | auto/iree → fused | evoxc-fused-iree |
| `p64-L32-i8-N1024-d4` | 3.097 (3.018) | 2.025 (1.953) | 0.034 (0.031) | 1.201 (1.091) | 0.035 (0.032) | auto/iree → fused | evoxc-auto |
| `p8192-L32-i8-N1024-d4` | 444.734 (440.106) | 7.779 (7.759) | 0.083 (0.079) | 57.336 (56.937) | 0.084 (0.081) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | evoxc-auto | 1247.60x | 58.78x |
| `p1024-L32-i8-N1024-d4` | evoxc-dense-iree | 12.51x | 0.59x |
| `p1024-L32-i8-N1024-d4` | evoxc-fused-iree | 1128.78x | 53.18x |
| `p128-L32-i8-N1024-d4` | evoxc-auto | 205.96x | 63.55x |
| `p128-L32-i8-N1024-d4` | evoxc-dense-iree | 4.94x | 1.53x |
| `p128-L32-i8-N1024-d4` | evoxc-fused-iree | 184.52x | 56.94x |
| `p2048-L32-i8-N1024-d4` | evoxc-auto | 2455.56x | 71.65x |
| `p2048-L32-i8-N1024-d4` | evoxc-dense-iree | 15.86x | 0.46x |
| `p2048-L32-i8-N1024-d4` | evoxc-fused-iree | 2210.50x | 64.50x |
| `p256-L32-i8-N1024-d4` | evoxc-auto | 316.99x | 54.29x |
| `p256-L32-i8-N1024-d4` | evoxc-dense-iree | 6.69x | 1.14x |
| `p256-L32-i8-N1024-d4` | evoxc-fused-iree | 356.28x | 61.02x |
| `p4096-L32-i8-N1024-d4` | evoxc-auto | 3767.02x | 82.40x |
| `p4096-L32-i8-N1024-d4` | evoxc-dense-iree | 16.79x | 0.37x |
| `p4096-L32-i8-N1024-d4` | evoxc-fused-iree | 3496.19x | 76.47x |
| `p512-L32-i8-N1024-d4` | evoxc-auto | 637.20x | 53.28x |
| `p512-L32-i8-N1024-d4` | evoxc-dense-iree | 10.37x | 0.87x |
| `p512-L32-i8-N1024-d4` | evoxc-fused-iree | 665.05x | 55.61x |
| `p64-L32-i8-N1024-d4` | evoxc-auto | 90.29x | 59.04x |
| `p64-L32-i8-N1024-d4` | evoxc-dense-iree | 2.58x | 1.69x |
| `p64-L32-i8-N1024-d4` | evoxc-fused-iree | 89.51x | 58.53x |
| `p8192-L32-i8-N1024-d4` | evoxc-auto | 5364.70x | 93.84x |
| `p8192-L32-i8-N1024-d4` | evoxc-dense-iree | 7.76x | 0.14x |
| `p8192-L32-i8-N1024-d4` | evoxc-fused-iree | 5319.78x | 93.05x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | 51.687 (50.840) | 2.001 (1.984) | 0.037 (0.034) | 0.666 (0.663) | 0.036 (0.031) | auto/iree → fused | evoxc-fused-iree |
| `p128-L32-i8-N1024-d4` | 6.709 (6.664) | 2.131 (2.119) | 0.031 (0.029) | 0.651 (0.628) | 0.036 (0.033) | auto/iree → fused | evoxc-auto |
| `p2048-L32-i8-N1024-d4` | 112.556 (109.350) | 1.909 (1.847) | 0.036 (0.035) | 0.655 (0.644) | 0.041 (0.036) | auto/iree → fused | evoxc-auto |
| `p256-L32-i8-N1024-d4` | 12.903 (12.802) | 2.214 (2.146) | 0.035 (0.034) | 0.640 (0.616) | 0.033 (0.030) | auto/iree → fused | evoxc-fused-iree |
| `p4096-L32-i8-N1024-d4` | 220.536 (215.451) | 2.088 (2.038) | 0.040 (0.036) | 0.701 (0.667) | 0.042 (0.041) | auto/iree → fused | evoxc-auto |
| `p512-L32-i8-N1024-d4` | 26.902 (25.978) | 2.071 (2.053) | 0.042 (0.036) | 0.662 (0.638) | 0.036 (0.033) | auto/iree → fused | evoxc-fused-iree |
| `p64-L32-i8-N1024-d4` | 3.367 (3.336) | 2.036 (2.002) | 0.037 (0.034) | 0.662 (0.651) | 0.034 (0.031) | auto/iree → fused | evoxc-fused-iree |
| `p8192-L32-i8-N1024-d4` | 456.821 (450.716) | 2.050 (1.999) | 0.047 (0.042) | 0.704 (0.674) | 0.048 (0.045) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | evoxc-auto | 1412.21x | 54.67x |
| `p1024-L32-i8-N1024-d4` | evoxc-dense-iree | 77.64x | 3.01x |
| `p1024-L32-i8-N1024-d4` | evoxc-fused-iree | 1447.81x | 56.05x |
| `p128-L32-i8-N1024-d4` | evoxc-auto | 218.52x | 69.41x |
| `p128-L32-i8-N1024-d4` | evoxc-dense-iree | 10.30x | 3.27x |
| `p128-L32-i8-N1024-d4` | evoxc-fused-iree | 188.45x | 59.86x |
| `p2048-L32-i8-N1024-d4` | evoxc-auto | 3135.27x | 53.19x |
| `p2048-L32-i8-N1024-d4` | evoxc-dense-iree | 171.89x | 2.92x |
| `p2048-L32-i8-N1024-d4` | evoxc-fused-iree | 2772.32x | 47.03x |
| `p256-L32-i8-N1024-d4` | evoxc-auto | 365.52x | 62.73x |
| `p256-L32-i8-N1024-d4` | evoxc-dense-iree | 20.16x | 3.46x |
| `p256-L32-i8-N1024-d4` | evoxc-fused-iree | 397.02x | 68.13x |
| `p4096-L32-i8-N1024-d4` | evoxc-auto | 5527.21x | 52.34x |
| `p4096-L32-i8-N1024-d4` | evoxc-dense-iree | 314.65x | 2.98x |
| `p4096-L32-i8-N1024-d4` | evoxc-fused-iree | 5213.61x | 49.37x |
| `p512-L32-i8-N1024-d4` | evoxc-auto | 642.05x | 49.43x |
| `p512-L32-i8-N1024-d4` | evoxc-dense-iree | 40.65x | 3.13x |
| `p512-L32-i8-N1024-d4` | evoxc-fused-iree | 747.27x | 57.53x |
| `p64-L32-i8-N1024-d4` | evoxc-auto | 91.99x | 55.63x |
| `p64-L32-i8-N1024-d4` | evoxc-dense-iree | 5.09x | 3.08x |
| `p64-L32-i8-N1024-d4` | evoxc-fused-iree | 99.90x | 60.41x |
| `p8192-L32-i8-N1024-d4` | evoxc-auto | 9761.14x | 43.80x |
| `p8192-L32-i8-N1024-d4` | evoxc-dense-iree | 648.80x | 2.91x |
| `p8192-L32-i8-N1024-d4` | evoxc-fused-iree | 9576.96x | 42.98x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | 49.855 | 42.412 | 2.943 | 7.668 | 2.903 |
| `p128-L32-i8-N1024-d4` | 8.217 | 11.951 | 2.929 | 5.182 | 2.877 |
| `p2048-L32-i8-N1024-d4` | 105.135 | 92.588 | 2.931 | 10.133 | 2.913 |
| `p256-L32-i8-N1024-d4` | 14.783 | 14.940 | 2.856 | 5.320 | 2.937 |
| `p4096-L32-i8-N1024-d4` | 199.069 | 179.616 | 2.981 | 17.249 | 3.009 |
| `p512-L32-i8-N1024-d4` | 25.861 | 20.458 | 2.873 | 6.020 | 2.902 |
| `p64-L32-i8-N1024-d4` | 4.949 | 6.326 | 2.801 | 4.748 | 2.774 |
| `p8192-L32-i8-N1024-d4` | 413.343 | 359.246 | 2.986 | 59.122 | 3.149 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | 47.425 / 2.430 (op 5%) | 39.942 / 2.470 (op 6%) | 0.450 / 2.493 (op 85%) | 5.134 / 2.534 (op 33%) | 0.463 / 2.440 (op 84%) |
| `p128-L32-i8-N1024-d4` | 5.792 / 2.425 (op 30%) | 9.448 / 2.503 (op 21%) | 0.443 / 2.486 (op 85%) | 2.713 / 2.469 (op 48%) | 0.453 / 2.424 (op 84%) |
| `p2048-L32-i8-N1024-d4` | 102.082 / 3.053 (op 3%) | 89.866 / 2.722 (op 3%) | 0.466 / 2.464 (op 84%) | 7.833 / 2.300 (op 23%) | 0.445 / 2.469 (op 85%) |
| `p256-L32-i8-N1024-d4` | 12.219 / 2.564 (op 17%) | 12.316 / 2.623 (op 18%) | 0.439 / 2.417 (op 85%) | 2.847 / 2.473 (op 46%) | 0.471 / 2.465 (op 84%) |
| `p4096-L32-i8-N1024-d4` | 195.469 / 3.600 (op 2%) | 176.829 / 2.787 (op 2%) | 0.463 / 2.518 (op 84%) | 14.489 / 2.760 (op 16%) | 0.468 / 2.541 (op 84%) |
| `p512-L32-i8-N1024-d4` | 23.307 / 2.553 (op 10%) | 18.165 / 2.293 (op 11%) | 0.419 / 2.454 (op 85%) | 3.538 / 2.482 (op 41%) | 0.402 / 2.500 (op 86%) |
| `p64-L32-i8-N1024-d4` | 2.706 / 2.243 (op 45%) | 4.136 / 2.190 (op 35%) | 0.446 / 2.355 (op 84%) | 2.353 / 2.395 (op 50%) | 0.431 / 2.342 (op 84%) |
| `p8192-L32-i8-N1024-d4` | 410.334 / 3.009 (op 1%) | 356.518 / 2.728 (op 1%) | 0.473 / 2.513 (op 84%) | 56.421 / 2.701 (op 5%) | 0.500 / 2.649 (op 84%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | evoxc-auto | 16.94x | 14.41x |
| `p1024-L32-i8-N1024-d4` | evoxc-dense-iree | 6.50x | 5.53x |
| `p1024-L32-i8-N1024-d4` | evoxc-fused-iree | 17.17x | 14.61x |
| `p128-L32-i8-N1024-d4` | evoxc-auto | 2.81x | 4.08x |
| `p128-L32-i8-N1024-d4` | evoxc-dense-iree | 1.59x | 2.31x |
| `p128-L32-i8-N1024-d4` | evoxc-fused-iree | 2.86x | 4.15x |
| `p2048-L32-i8-N1024-d4` | evoxc-auto | 35.87x | 31.59x |
| `p2048-L32-i8-N1024-d4` | evoxc-dense-iree | 10.38x | 9.14x |
| `p2048-L32-i8-N1024-d4` | evoxc-fused-iree | 36.09x | 31.78x |
| `p256-L32-i8-N1024-d4` | evoxc-auto | 5.18x | 5.23x |
| `p256-L32-i8-N1024-d4` | evoxc-dense-iree | 2.78x | 2.81x |
| `p256-L32-i8-N1024-d4` | evoxc-fused-iree | 5.03x | 5.09x |
| `p4096-L32-i8-N1024-d4` | evoxc-auto | 66.79x | 60.26x |
| `p4096-L32-i8-N1024-d4` | evoxc-dense-iree | 11.54x | 10.41x |
| `p4096-L32-i8-N1024-d4` | evoxc-fused-iree | 66.16x | 59.70x |
| `p512-L32-i8-N1024-d4` | evoxc-auto | 9.00x | 7.12x |
| `p512-L32-i8-N1024-d4` | evoxc-dense-iree | 4.30x | 3.40x |
| `p512-L32-i8-N1024-d4` | evoxc-fused-iree | 8.91x | 7.05x |
| `p64-L32-i8-N1024-d4` | evoxc-auto | 1.77x | 2.26x |
| `p64-L32-i8-N1024-d4` | evoxc-dense-iree | 1.04x | 1.33x |
| `p64-L32-i8-N1024-d4` | evoxc-fused-iree | 1.78x | 2.28x |
| `p8192-L32-i8-N1024-d4` | evoxc-auto | 138.42x | 120.31x |
| `p8192-L32-i8-N1024-d4` | evoxc-dense-iree | 6.99x | 6.08x |
| `p8192-L32-i8-N1024-d4` | evoxc-fused-iree | 131.27x | 114.09x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | -0.084252 | -0.084252 | -0.084252 | -0.084252 | -0.084252 |
| `p128-L32-i8-N1024-d4` | -0.089894 | -0.089894 | -0.089894 | -0.089894 | -0.089894 |
| `p2048-L32-i8-N1024-d4` | -0.084252 | -0.084252 | -0.084252 | -0.084252 | -0.084252 |
| `p256-L32-i8-N1024-d4` | -0.090462 | -0.090462 | -0.090462 | -0.090462 | -0.090462 |
| `p4096-L32-i8-N1024-d4` | -0.084252 | -0.084304 | -0.084252 | -0.084252 | -0.084252 |
| `p512-L32-i8-N1024-d4` | -0.084265 | -0.084265 | -0.084265 | -0.084265 | -0.084265 |
| `p64-L32-i8-N1024-d4` | -0.092110 | -0.092110 | -0.092110 | -0.092110 | -0.092110 |
| `p8192-L32-i8-N1024-d4` | -0.084252 | -0.084252 | -0.084252 | -0.084252 | -0.084252 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | evoxc-auto | 476.258 | 0 | 0.054 |
| `p1024-L32-i8-N1024-d4` | evoxc-dense-iree | 3423.137 | 1 | 1.929 |
| `p1024-L32-i8-N1024-d4` | evoxc-fused-iree | 490.643 | 0 | 0.050 |
| `p128-L32-i8-N1024-d4` | evoxc-auto | 477.406 | 0 | 0.056 |
| `p128-L32-i8-N1024-d4` | evoxc-dense-iree | 3356.360 | 1 | 1.036 |
| `p128-L32-i8-N1024-d4` | evoxc-fused-iree | 492.141 | 0 | 0.061 |
| `p2048-L32-i8-N1024-d4` | evoxc-auto | 501.202 | 0 | 0.063 |
| `p2048-L32-i8-N1024-d4` | evoxc-dense-iree | 3375.530 | 1 | 3.423 |
| `p2048-L32-i8-N1024-d4` | evoxc-fused-iree | 489.993 | 0 | 0.059 |
| `p256-L32-i8-N1024-d4` | evoxc-auto | 486.543 | 0 | 0.054 |
| `p256-L32-i8-N1024-d4` | evoxc-dense-iree | 3359.048 | 1 | 1.145 |
| `p256-L32-i8-N1024-d4` | evoxc-fused-iree | 501.585 | 0 | 0.053 |
| `p4096-L32-i8-N1024-d4` | evoxc-auto | 477.569 | 0 | 0.070 |
| `p4096-L32-i8-N1024-d4` | evoxc-dense-iree | 3406.910 | 1 | 6.150 |
| `p4096-L32-i8-N1024-d4` | evoxc-fused-iree | 490.158 | 0 | 0.072 |
| `p512-L32-i8-N1024-d4` | evoxc-auto | 480.738 | 0 | 0.057 |
| `p512-L32-i8-N1024-d4` | evoxc-dense-iree | 3347.765 | 1 | 1.337 |
| `p512-L32-i8-N1024-d4` | evoxc-fused-iree | 488.970 | 0 | 0.053 |
| `p64-L32-i8-N1024-d4` | evoxc-auto | 1786.045 | 0 | 0.051 |
| `p64-L32-i8-N1024-d4` | evoxc-dense-iree | 3220.811 | 1 | 0.996 |
| `p64-L32-i8-N1024-d4` | evoxc-fused-iree | 1492.454 | 0 | 0.060 |
| `p8192-L32-i8-N1024-d4` | evoxc-auto | 483.116 | 0 | 0.095 |
| `p8192-L32-i8-N1024-d4` | evoxc-dense-iree | 3361.334 | 1 | 40.233 |
| `p8192-L32-i8-N1024-d4` | evoxc-fused-iree | 471.102 | 0 | 0.094 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p1024-L32-i8-N1024-d4` | batch_forward | `f8691c62d359ce81bd65f3fc20b12f7d` | torch-per-tree, torch-batched | 1024 |
| `p1024-L32-i8-N1024-d4` | batch_forward | `e8c9330bc43f052d1763fd53f8c30d65` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 1024 |
| `p1024-L32-i8-N1024-d4` | forward | `fd2d5d8f249e0d8e3860b7318bf080c6` | torch-per-tree, torch-batched | 1 |
| `p1024-L32-i8-N1024-d4` | forward | `cae90d43d2c576bbc2529cac43677335` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 1 |
| `p128-L32-i8-N1024-d4` | batch_forward | `2b720dbdf828b3f359db5eded0862061` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p128-L32-i8-N1024-d4` | forward | `d2b188de2e3589d87f34ee849e5397cf` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p2048-L32-i8-N1024-d4` | batch_forward | `ad9e793ae4a7443c98da4a6afa00df15` | torch-per-tree, torch-batched | 6144 |
| `p2048-L32-i8-N1024-d4` | batch_forward | `304c7d182a69291225e25d1e8deb3a88` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 6144 |
| `p2048-L32-i8-N1024-d4` | forward | `5d587f52935f454392954605b6c6f04a` | torch-per-tree, torch-batched | 6 |
| `p2048-L32-i8-N1024-d4` | forward | `ffef2603792d0b55afe397248c474313` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 6 |
| `p256-L32-i8-N1024-d4` | batch_forward | `8cebbd598d7df100e00887ecc5f6890f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p256-L32-i8-N1024-d4` | forward | `a3354e97e0d54860ecf733afd2a85490` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p4096-L32-i8-N1024-d4` | batch_forward | `97cb2b16aa2372b7f479ac2f68952f8d` | torch-per-tree, torch-batched | 17408 |
| `p4096-L32-i8-N1024-d4` | batch_forward | `04a6c9fff109c9350197fa730993094f` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 17408 |
| `p4096-L32-i8-N1024-d4` | forward | `5c25b2560e4fb4bcf3e21f0c7f63f2a1` | torch-per-tree, torch-batched | 17 |
| `p4096-L32-i8-N1024-d4` | forward | `463cbf887f58cb96cefdbd066b762f4a` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 17 |
| `p512-L32-i8-N1024-d4` | batch_forward | `d89c8e2fc1bbaf3ea0090a1d47c701b5` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p512-L32-i8-N1024-d4` | forward | `4a877ba7d334d8872a77a0c0913b1ac8` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p64-L32-i8-N1024-d4` | batch_forward | `d2d5465a3ddbfaf78983fb52907ee3d4` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p64-L32-i8-N1024-d4` | forward | `de2cef7d9d2639fbbd0fd974c9eeb8d7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 0 |
| `p8192-L32-i8-N1024-d4` | batch_forward | `e8c96d740b98ba378cefc80a5cbd4212` | torch-per-tree, torch-batched | 41984 |
| `p8192-L32-i8-N1024-d4` | batch_forward | `69f92ce2181cbbe00d04a6720091bcc8` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 41984 |
| `p8192-L32-i8-N1024-d4` | forward | `cf3b7fb567e2e1462ac847747b774da6` | torch-per-tree, torch-batched | 41 |
| `p8192-L32-i8-N1024-d4` | forward | `03d49d700d2192102735f4bcb31bc302` | evoxc-auto, evoxc-dense-iree, evoxc-fused-iree | 41 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p1024-L32-i8-N1024-d4: 4/5, p128-L32-i8-N1024-d4: 4/5, p2048-L32-i8-N1024-d4: 4/5, p256-L32-i8-N1024-d4: 4/5, p4096-L32-i8-N1024-d4: 4/5, p512-L32-i8-N1024-d4: 4/5, p64-L32-i8-N1024-d4: 4/5, p8192-L32-i8-N1024-d4: 4/5.

## Caveats

- Every scheduled config in `campaign.json` finished `ok`.
