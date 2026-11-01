# evoxc benchmark summary: main

Source: `test/benchmark/evoxc/results/largescale/main` — **57 records**, 10 cells x 7 arms.

The campaign ran **1 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **900.8 s**, failures **1**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-30T04:18:23Z |
| git SHA (`tree_root_sha`) | 009d47b28db280760c527edb06bf248c0ca74116 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A19/src` |
| python | 3.12.3 |
| torch | 2.13.0+rocm7.1 |
| numpy | 2.5.4 |
| hip | 7.1.52802 |
| iree_base_compiler | 3.11.0 |
| iree_base_runtime | 3.11.0 |
| evoxc_version | - |
| evoxc_path | - |
| gpu | AMD Instinct MI300X VF |
| gpu_total_memory_gb | 191.7 |
| `EVOXC_DISABLE_LOCATIONS` | `1` |

## Workload: sr_fitness

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | 122.687 (122.572) | 32.690 (32.582) | 446.598 (446.428) | 452.279 (448.545) | 117.187 (117.106) | 37.415 (37.304) | - | auto/numpy → specialized | torch-batched |
| `p1000-L64-i32-N4096` | 122.748 (121.527) | 10.222 (10.188) | 197.305 (195.202) | 195.514 (193.987) | 39.306 (39.183) | 13.028 (12.971) | - | auto/numpy → specialized | torch-batched |
| `p2048-L128-i32-N1024-d7` | 601.975 (600.732) | 26.428 (26.413) | 891.234 (886.441) | 889.919 (884.010) | - | 25.069 (24.740) | 105.653 (105.582) | auto/numpy → dense | evoxc-batch-iree |
| `p2048-L256-i32-N1024-d8` | 888.998 (866.288) | 48.623 (48.612) | 1355.617 (1353.451) | 1345.214 (1343.030) | - | - | 240.283 (240.236) | auto/numpy → dense | torch-batched |
| `p2048-L64-i32-N1024` | 249.436 (247.048) | 7.121 (7.111) | 10.991 (10.968) | 374.887 (371.990) | 11.061 (11.016) | 12.790 (12.700) | - | auto/numpy → dense | torch-batched |
| `p256-L64-i32-N16384` | 30.949 (30.602) | 10.434 (10.421) | 110.211 (108.997) | 107.593 (106.930) | 78.121 (77.966) | 12.609 (12.483) | - | auto/numpy → specialized | torch-batched |
| `p256-L64-i32-N4096` | 30.758 (30.488) | 5.036 (5.031) | 50.471 (49.570) | 51.903 (51.600) | 5.818 (5.773) | 5.232 (5.195) | - | auto/numpy → specialized | torch-batched |
| `p4096-L64-i32-N1024` | 493.826 (491.457) | 10.507 (10.477) | 37.181 (36.876) | 734.529 (729.233) | 37.065 (37.010) | 23.236 (23.200) | - | auto/numpy → dense | torch-batched |
| `p4096-L64-i64-N1024` | 478.036 (474.899) | 10.420 (10.384) | 37.228 (37.113) | 691.276 (675.933) | 37.018 (36.948) | 24.558 (24.373) | - | auto/numpy → dense | torch-batched |
| `p8192-L64-i32-N1024` | 1023.146 (1005.409) | 17.640 (17.627) | 76.337 (76.117) | - | 76.231 (76.057) | - | - | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L64-i32-N16384` | evoxc-auto | 0.27x | 0.07x |
| `p1000-L64-i32-N16384` | evoxc-specialized-numpy | 0.27x | 0.07x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree | 1.05x | 0.28x |
| `p1000-L64-i32-N16384` | evoxc-batch-iree | 3.28x | 0.87x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p1000-L64-i32-N4096` | evoxc-auto | 0.62x | 0.05x |
| `p1000-L64-i32-N4096` | evoxc-specialized-numpy | 0.63x | 0.05x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree | 3.12x | 0.26x |
| `p1000-L64-i32-N4096` | evoxc-batch-iree | 9.42x | 0.78x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 0.68x | 0.03x |
| `p2048-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.68x | 0.03x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 24.01x | 1.05x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 5.70x | 0.25x |
| `p2048-L256-i32-N1024-d8` | evoxc-auto | 0.66x | 0.04x |
| `p2048-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.66x | 0.04x |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 3.70x | 0.20x |
| `p2048-L64-i32-N1024` | evoxc-auto | 22.70x | 0.65x |
| `p2048-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.02x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree | 22.55x | 0.64x |
| `p2048-L64-i32-N1024` | evoxc-batch-iree | 19.50x | 0.56x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N16384` | evoxc-auto | 0.28x | 0.09x |
| `p256-L64-i32-N16384` | evoxc-specialized-numpy | 0.29x | 0.10x |
| `p256-L64-i32-N16384` | evoxc-dense-iree | 0.40x | 0.13x |
| `p256-L64-i32-N16384` | evoxc-batch-iree | 2.45x | 0.83x |
| `p256-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 0.61x | 0.10x |
| `p256-L64-i32-N4096` | evoxc-specialized-numpy | 0.59x | 0.10x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 5.29x | 0.87x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 5.88x | 0.96x |
| `p256-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto | 13.28x | 0.28x |
| `p4096-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.01x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 13.32x | 0.28x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 21.25x | 0.45x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i64-N1024` | evoxc-auto | 12.84x | 0.28x |
| `p4096-L64-i64-N1024` | evoxc-specialized-numpy | 0.69x | 0.02x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree | 12.91x | 0.28x |
| `p4096-L64-i64-N1024` | evoxc-batch-iree | 19.47x | 0.42x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree-raised | - | - |
| `p8192-L64-i32-N1024` | evoxc-auto | 13.40x | 0.23x |
| `p8192-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree | 13.42x | 0.23x |
| `p8192-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | 95.230 (94.257) | 32.667 (32.486) | 441.074 (439.750) | 439.510 (438.814) | 149.772 (149.118) | 52.258 (51.465) | - | auto/numpy → specialized | torch-batched |
| `p1000-L64-i32-N4096` | 94.792 (93.715) | 10.178 (10.149) | 197.887 (196.545) | 196.274 (194.297) | 47.953 (47.635) | 14.056 (13.535) | - | auto/numpy → specialized | torch-batched |
| `p2048-L128-i32-N1024-d7` | 533.884 (532.002) | 26.419 (26.403) | 891.593 (889.225) | 907.952 (872.948) | - | 18.640 (18.519) | 109.977 (109.848) | auto/numpy → dense | evoxc-batch-iree |
| `p2048-L256-i32-N1024-d8` | 816.808 (808.049) | 48.586 (48.540) | 1369.314 (1347.762) | 1323.573 (1304.676) | - | - | 244.878 (244.346) | auto/numpy → dense | torch-batched |
| `p2048-L64-i32-N1024` | 189.848 (189.106) | 7.108 (7.085) | 14.402 (14.166) | 370.670 (369.281) | 14.498 (14.158) | 12.285 (11.983) | - | auto/numpy → dense | torch-batched |
| `p256-L64-i32-N16384` | 23.888 (23.788) | 10.441 (10.374) | 109.081 (107.504) | 105.677 (104.278) | 79.629 (79.016) | 12.996 (12.935) | - | auto/numpy → specialized | torch-batched |
| `p256-L64-i32-N4096` | 23.641 (23.460) | 5.017 (4.977) | 50.195 (49.915) | 50.969 (50.579) | 7.495 (7.221) | 7.146 (6.970) | - | auto/numpy → specialized | torch-batched |
| `p4096-L64-i32-N1024` | 373.969 (370.432) | 10.473 (10.469) | 46.004 (45.913) | 739.423 (730.267) | 46.527 (46.296) | 22.068 (21.932) | - | auto/numpy → dense | torch-batched |
| `p4096-L64-i64-N1024` | 362.798 (356.052) | 10.375 (10.370) | 46.564 (46.287) | 722.349 (679.477) | 47.044 (46.652) | 22.527 (21.591) | - | auto/numpy → dense | torch-batched |
| `p8192-L64-i32-N1024` | 783.243 (780.109) | 17.587 (17.555) | 93.422 (93.084) | - | 94.249 (93.696) | - | - | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L64-i32-N16384` | evoxc-auto | 0.22x | 0.07x |
| `p1000-L64-i32-N16384` | evoxc-specialized-numpy | 0.22x | 0.07x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree | 0.64x | 0.22x |
| `p1000-L64-i32-N16384` | evoxc-batch-iree | 1.82x | 0.63x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p1000-L64-i32-N4096` | evoxc-auto | 0.48x | 0.05x |
| `p1000-L64-i32-N4096` | evoxc-specialized-numpy | 0.48x | 0.05x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree | 1.98x | 0.21x |
| `p1000-L64-i32-N4096` | evoxc-batch-iree | 6.74x | 0.72x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 0.60x | 0.03x |
| `p2048-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 28.64x | 1.42x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 4.85x | 0.24x |
| `p2048-L256-i32-N1024-d8` | evoxc-auto | 0.60x | 0.04x |
| `p2048-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.62x | 0.04x |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 3.34x | 0.20x |
| `p2048-L64-i32-N1024` | evoxc-auto | 13.18x | 0.49x |
| `p2048-L64-i32-N1024` | evoxc-specialized-numpy | 0.51x | 0.02x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree | 13.10x | 0.49x |
| `p2048-L64-i32-N1024` | evoxc-batch-iree | 15.45x | 0.58x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N16384` | evoxc-auto | 0.22x | 0.10x |
| `p256-L64-i32-N16384` | evoxc-specialized-numpy | 0.23x | 0.10x |
| `p256-L64-i32-N16384` | evoxc-dense-iree | 0.30x | 0.13x |
| `p256-L64-i32-N16384` | evoxc-batch-iree | 1.84x | 0.80x |
| `p256-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 0.47x | 0.10x |
| `p256-L64-i32-N4096` | evoxc-specialized-numpy | 0.46x | 0.10x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 3.15x | 0.67x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 3.31x | 0.70x |
| `p256-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto | 8.13x | 0.23x |
| `p4096-L64-i32-N1024` | evoxc-specialized-numpy | 0.51x | 0.01x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 8.04x | 0.23x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 16.95x | 0.47x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i64-N1024` | evoxc-auto | 7.79x | 0.22x |
| `p4096-L64-i64-N1024` | evoxc-specialized-numpy | 0.50x | 0.01x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree | 7.71x | 0.22x |
| `p4096-L64-i64-N1024` | evoxc-batch-iree | 16.11x | 0.46x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree-raised | - | - |
| `p8192-L64-i32-N1024` | evoxc-auto | 8.38x | 0.19x |
| `p8192-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree | 8.31x | 0.19x |
| `p8192-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | 97.102 (94.607) | 4.424 (4.398) | 0.853 (0.832) | 170.420 (169.333) | 0.868 (0.861) | 2.466 (2.413) | - | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N4096` | 95.867 (95.042) | 4.455 (4.427) | 0.860 (0.842) | 163.164 (159.649) | 0.890 (0.871) | 2.546 (2.504) | - | auto/numpy → dense | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 543.543 (539.241) | 16.312 (16.276) | 821.491 (819.602) | 783.087 (757.158) | - | 6.857 (6.777) | 1.220 (1.192) | auto/numpy → dense | evoxc-dense-iree-raised |
| `p2048-L256-i32-N1024-d8` | 841.792 (816.795) | 26.404 (24.981) | 1257.969 (1244.247) | 1251.885 (1235.059) | - | - | 2.059 (1.965) | auto/numpy → dense | evoxc-dense-iree-raised |
| `p2048-L64-i32-N1024` | 196.332 (194.572) | 4.492 (4.427) | 0.863 (0.844) | 351.244 (346.111) | 0.917 (0.889) | 3.331 (3.316) | - | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N16384` | 24.270 (24.080) | 4.446 (4.426) | 0.843 (0.824) | 41.834 (40.483) | 0.865 (0.828) | 1.867 (1.795) | - | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N4096` | 24.179 (24.148) | 4.519 (4.462) | 0.865 (0.832) | 44.328 (44.096) | 0.849 (0.836) | 1.918 (1.905) | - | auto/numpy → dense | evoxc-dense-iree |
| `p4096-L64-i32-N1024` | 383.681 (382.256) | 4.428 (4.398) | 0.982 (0.958) | 710.392 (692.572) | 0.881 (0.854) | 5.226 (5.167) | - | auto/numpy → dense | evoxc-dense-iree |
| `p4096-L64-i64-N1024` | 374.486 (367.677) | 4.407 (4.103) | 0.866 (0.836) | 636.962 (634.503) | 0.909 (0.868) | 5.115 (5.056) | - | auto/numpy → dense | evoxc-auto |
| `p8192-L64-i32-N1024` | 807.608 (801.870) | 4.573 (4.548) | 0.922 (0.908) | - | 0.911 (0.894) | - | - | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=16, bands=None, alignment=variant, switch_ops=False, min_pop=64, compile_workers=4) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L64-i32-N16384` | evoxc-auto | 113.86x | 5.19x |
| `p1000-L64-i32-N16384` | evoxc-specialized-numpy | 0.57x | 0.03x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree | 111.93x | 5.10x |
| `p1000-L64-i32-N16384` | evoxc-batch-iree | 39.38x | 1.79x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p1000-L64-i32-N4096` | evoxc-auto | 111.43x | 5.18x |
| `p1000-L64-i32-N4096` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree | 107.73x | 5.01x |
| `p1000-L64-i32-N4096` | evoxc-batch-iree | 37.65x | 1.75x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 0.66x | 0.02x |
| `p2048-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.69x | 0.02x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 79.27x | 2.38x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 445.53x | 13.37x |
| `p2048-L256-i32-N1024-d8` | evoxc-auto | 0.67x | 0.02x |
| `p2048-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.67x | 0.02x |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 408.74x | 12.82x |
| `p2048-L64-i32-N1024` | evoxc-auto | 227.58x | 5.21x |
| `p2048-L64-i32-N1024` | evoxc-specialized-numpy | 0.56x | 0.01x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree | 214.06x | 4.90x |
| `p2048-L64-i32-N1024` | evoxc-batch-iree | 58.95x | 1.35x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N16384` | evoxc-auto | 28.80x | 5.28x |
| `p256-L64-i32-N16384` | evoxc-specialized-numpy | 0.58x | 0.11x |
| `p256-L64-i32-N16384` | evoxc-dense-iree | 28.06x | 5.14x |
| `p256-L64-i32-N16384` | evoxc-batch-iree | 13.00x | 2.38x |
| `p256-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 27.96x | 5.22x |
| `p256-L64-i32-N4096` | evoxc-specialized-numpy | 0.55x | 0.10x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 28.50x | 5.33x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 12.60x | 2.36x |
| `p256-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto | 390.63x | 4.51x |
| `p4096-L64-i32-N1024` | evoxc-specialized-numpy | 0.54x | 0.01x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 435.61x | 5.03x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 73.41x | 0.85x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i64-N1024` | evoxc-auto | 432.33x | 5.09x |
| `p4096-L64-i64-N1024` | evoxc-specialized-numpy | 0.59x | 0.01x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree | 411.98x | 4.85x |
| `p4096-L64-i64-N1024` | evoxc-batch-iree | 73.22x | 0.86x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree-raised | - | - |
| `p8192-L64-i32-N1024` | evoxc-auto | 875.55x | 4.96x |
| `p8192-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree | 886.02x | 5.02x |
| `p8192-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | 51.185 | 73.859 | 913.764 | 865.338 | 127.702 | - | - |
| `p1000-L64-i32-N4096` | 53.651 | 59.771 | 881.655 | 933.518 | 47.331 | - | - |
| `p2048-L128-i32-N1024-d7` | 125.411 | 238.505 | 1857.569 | 1745.854 | - | - | 124.067 |
| `p2048-L256-i32-N1024-d8` | 92.375 | 229.332 | 1039.818 | 1066.720 | - | - | 273.064 |
| `p2048-L64-i32-N1024` | 113.985 | 147.354 | 21.719 | 2150.504 | 21.193 | 11298.315 | - |
| `p256-L64-i32-N16384` | 18.466 | 28.486 | 334.679 | 329.944 | 83.716 | 3194.202 | - |
| `p256-L64-i32-N4096` | 17.858 | 25.333 | 289.475 | 357.420 | 10.526 | - | - |
| `p4096-L64-i32-N1024` | 227.099 | 290.948 | 57.849 | 4489.597 | 57.275 | - | - |
| `p4096-L64-i64-N1024` | 232.150 | 254.661 | 56.572 | 4276.547 | 56.821 | - | - |
| `p8192-L64-i32-N1024` | 510.355 | 593.863 | 109.868 | - | 110.724 | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | 48.468 / 2.717 (op 5%) | 71.282 / 2.577 (op 3%) | 910.792 / 2.972 (op 0%) | 862.602 / 2.736 (op 0%) | 125.066 / 2.635 (op 2%) | - | - |
| `p1000-L64-i32-N4096` | 50.943 / 2.708 (op 5%) | 57.198 / 2.573 (op 4%) | 878.807 / 2.848 (op 0%) | 930.769 / 2.748 (op 0%) | 44.626 / 2.704 (op 6%) | - | - |
| `p2048-L128-i32-N1024-d7` | 122.403 / 3.008 (op 2%) | 235.682 / 2.823 (op 1%) | 1854.550 / 3.019 (op 0%) | 1742.858 / 2.997 (op 0%) | - | - | 121.313 / 2.754 (op 2%) |
| `p2048-L256-i32-N1024-d8` | 89.587 / 2.788 (op 3%) | 226.657 / 2.675 (op 1%) | 1036.575 / 3.243 (op 0%) | 1063.573 / 3.147 (op 0%) | - | - | 270.301 / 2.763 (op 1%) |
| `p2048-L64-i32-N1024` | 111.122 / 2.863 (op 3%) | 144.525 / 2.829 (op 2%) | 19.009 / 2.710 (op 12%) | 2147.336 / 3.168 (op 0%) | 18.486 / 2.707 (op 13%) | 11295.150 / 3.165 (op 0%) | - |
| `p256-L64-i32-N16384` | 15.864 / 2.602 (op 14%) | 25.761 / 2.725 (op 10%) | 332.119 / 2.560 (op 1%) | 327.403 / 2.541 (op 1%) | 81.166 / 2.550 (op 3%) | 3190.878 / 3.324 (op 0%) | - |
| `p256-L64-i32-N4096` | 15.222 / 2.636 (op 15%) | 22.699 / 2.634 (op 10%) | 286.868 / 2.607 (op 1%) | 354.759 / 2.661 (op 1%) | 8.031 / 2.495 (op 24%) | - | - |
| `p4096-L64-i32-N1024` | 223.585 / 3.514 (op 2%) | 287.938 / 3.009 (op 1%) | 55.047 / 2.802 (op 5%) | 4486.268 / 3.330 (op 0%) | 54.563 / 2.712 (op 5%) | - | - |
| `p4096-L64-i64-N1024` | 228.639 / 3.512 (op 2%) | 251.927 / 2.735 (op 1%) | 53.832 / 2.740 (op 5%) | 4273.359 / 3.188 (op 0%) | 53.927 / 2.894 (op 5%) | - | - |
| `p8192-L64-i32-N1024` | 507.376 / 2.979 (op 1%) | 591.056 / 2.807 (op 0%) | 107.011 / 2.857 (op 3%) | - | 107.978 / 2.746 (op 2%) | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L64-i32-N16384` | evoxc-auto | 0.06x | 0.08x |
| `p1000-L64-i32-N16384` | evoxc-specialized-numpy | 0.06x | 0.09x |
| `p1000-L64-i32-N16384` | evoxc-dense-iree | 0.40x | 0.58x |
| `p1000-L64-i32-N16384` | evoxc-batch-iree | - | - |
| `p1000-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p1000-L64-i32-N4096` | evoxc-auto | 0.06x | 0.07x |
| `p1000-L64-i32-N4096` | evoxc-specialized-numpy | 0.06x | 0.06x |
| `p1000-L64-i32-N4096` | evoxc-dense-iree | 1.13x | 1.26x |
| `p1000-L64-i32-N4096` | evoxc-batch-iree | - | - |
| `p1000-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 0.07x | 0.13x |
| `p2048-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.07x | 0.14x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 1.01x | 1.92x |
| `p2048-L256-i32-N1024-d8` | evoxc-auto | 0.09x | 0.22x |
| `p2048-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.09x | 0.21x |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 0.34x | 0.84x |
| `p2048-L64-i32-N1024` | evoxc-auto | 5.25x | 6.78x |
| `p2048-L64-i32-N1024` | evoxc-specialized-numpy | 0.05x | 0.07x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree | 5.38x | 6.95x |
| `p2048-L64-i32-N1024` | evoxc-batch-iree | 0.01x | 0.01x |
| `p2048-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N16384` | evoxc-auto | 0.06x | 0.09x |
| `p256-L64-i32-N16384` | evoxc-specialized-numpy | 0.06x | 0.09x |
| `p256-L64-i32-N16384` | evoxc-dense-iree | 0.22x | 0.34x |
| `p256-L64-i32-N16384` | evoxc-batch-iree | 0.01x | 0.01x |
| `p256-L64-i32-N16384` | evoxc-dense-iree-raised | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 0.06x | 0.09x |
| `p256-L64-i32-N4096` | evoxc-specialized-numpy | 0.05x | 0.07x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 1.70x | 2.41x |
| `p256-L64-i32-N4096` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto | 3.93x | 5.03x |
| `p4096-L64-i32-N1024` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 3.97x | 5.08x |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p4096-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p4096-L64-i64-N1024` | evoxc-auto | 4.10x | 4.50x |
| `p4096-L64-i64-N1024` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p4096-L64-i64-N1024` | evoxc-dense-iree | 4.09x | 4.48x |
| `p4096-L64-i64-N1024` | evoxc-batch-iree | - | - |
| `p4096-L64-i64-N1024` | evoxc-dense-iree-raised | - | - |
| `p8192-L64-i32-N1024` | evoxc-auto | 4.65x | 5.41x |
| `p8192-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree | 4.61x | 5.36x |
| `p8192-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree-raised | - | - |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|---|---|
| `p1000-L64-i32-N16384` | -0.083832 | -0.083832 | -0.083832 | -0.083832 | -0.083832 | - | - |
| `p1000-L64-i32-N4096` | -0.084133 | -0.084133 | -0.084133 | -0.084133 | -0.084133 | - | - |
| `p2048-L128-i32-N1024-d7` | -0.084304 | -0.084304 | -0.084304 | -0.084304 | - | - | -0.084304 |
| `p2048-L256-i32-N1024-d8` | -0.084252 | -0.084252 | -0.084252 | -0.084252 | - | - | -0.084252 |
| `p2048-L64-i32-N1024` | -0.084252 | -0.084252 | -0.084252 | -0.084252 | -0.084252 | -0.084252 | - |
| `p256-L64-i32-N16384` | -0.083966 | -0.083966 | -0.083966 | -0.083966 | -0.083966 | -0.084084 | - |
| `p256-L64-i32-N4096` | -0.084161 | -0.084161 | -0.084161 | -0.084161 | -0.084161 | - | - |
| `p4096-L64-i32-N1024` | -0.084252 | -0.084252 | -0.084268 | -0.084252 | -0.084268 | - | - |
| `p4096-L64-i64-N1024` | -0.083921 | -0.083921 | -0.083921 | -0.083921 | -0.083921 | - | - |
| `p8192-L64-i32-N1024` | -0.084252 | -0.084194 | -0.084252 | - | -0.084252 | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p1000-L64-i32-N16384` | evoxc-auto | 3926.392 | 821 | 446.598 |
| `p1000-L64-i32-N16384` | evoxc-specialized-numpy | 4067.597 | 821 | 452.279 |
| `p1000-L64-i32-N16384` | evoxc-dense-iree | 6803.824 | 1 | 117.187 |
| `p1000-L64-i32-N16384` | evoxc-batch-iree | 19579.835 | 0 | 37.415 |
| `p1000-L64-i32-N16384` | evoxc-dense-iree-raised | - | - | - |
| `p1000-L64-i32-N4096` | evoxc-auto | 3617.464 | 821 | 197.305 |
| `p1000-L64-i32-N4096` | evoxc-specialized-numpy | 3734.935 | 821 | 195.514 |
| `p1000-L64-i32-N4096` | evoxc-dense-iree | 6380.492 | 1 | 39.306 |
| `p1000-L64-i32-N4096` | evoxc-batch-iree | 17595.757 | 0 | 13.028 |
| `p1000-L64-i32-N4096` | evoxc-dense-iree-raised | - | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 17286.273 | 1642 | 891.234 |
| `p2048-L128-i32-N1024-d7` | evoxc-specialized-numpy | 17349.427 | 1642 | 889.919 |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - | - |
| `p2048-L128-i32-N1024-d7` | evoxc-batch-iree | 248197.772 | 0 | 25.069 |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 14151.516 | 1 | 105.653 |
| `p2048-L256-i32-N1024-d8` | evoxc-auto | 26499.988 | 1643 | 1355.617 |
| `p2048-L256-i32-N1024-d8` | evoxc-specialized-numpy | 26129.090 | 1643 | 1345.214 |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree | - | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | - | - | - |
| `p2048-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 41297.391 | 1 | 240.283 |
| `p2048-L64-i32-N1024` | evoxc-auto | 6670.126 | 1 | 10.991 |
| `p2048-L64-i32-N1024` | evoxc-specialized-numpy | 7221.175 | 1650 | 374.887 |
| `p2048-L64-i32-N1024` | evoxc-dense-iree | 6711.881 | 1 | 11.061 |
| `p2048-L64-i32-N1024` | evoxc-batch-iree | 48310.030 | 0 | 12.790 |
| `p2048-L64-i32-N1024` | evoxc-dense-iree-raised | - | - | - |
| `p256-L64-i32-N16384` | evoxc-auto | 987.287 | 219 | 110.211 |
| `p256-L64-i32-N16384` | evoxc-specialized-numpy | 1001.909 | 219 | 107.593 |
| `p256-L64-i32-N16384` | evoxc-dense-iree | 6512.631 | 1 | 78.121 |
| `p256-L64-i32-N16384` | evoxc-batch-iree | 4918.263 | 0 | 12.609 |
| `p256-L64-i32-N16384` | evoxc-dense-iree-raised | - | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 949.855 | 219 | 50.471 |
| `p256-L64-i32-N4096` | evoxc-specialized-numpy | 963.259 | 219 | 51.903 |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 6650.713 | 1 | 5.818 |
| `p256-L64-i32-N4096` | evoxc-batch-iree | 4384.567 | 0 | 5.232 |
| `p256-L64-i32-N4096` | evoxc-dense-iree-raised | - | - | - |
| `p4096-L64-i32-N1024` | evoxc-auto | 6708.382 | 1 | 37.181 |
| `p4096-L64-i32-N1024` | evoxc-specialized-numpy | 14336.693 | 3217 | 734.529 |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 6736.497 | 1 | 37.065 |
| `p4096-L64-i32-N1024` | evoxc-batch-iree | 187174.888 | 0 | 23.236 |
| `p4096-L64-i32-N1024` | evoxc-dense-iree-raised | - | - | - |
| `p4096-L64-i64-N1024` | evoxc-auto | 6557.725 | 1 | 37.228 |
| `p4096-L64-i64-N1024` | evoxc-specialized-numpy | 13452.663 | 3249 | 691.276 |
| `p4096-L64-i64-N1024` | evoxc-dense-iree | 6661.481 | 1 | 37.018 |
| `p4096-L64-i64-N1024` | evoxc-batch-iree | 200735.245 | 0 | 24.558 |
| `p4096-L64-i64-N1024` | evoxc-dense-iree-raised | - | - | - |
| `p8192-L64-i32-N1024` | evoxc-auto | 6981.901 | 1 | 76.337 |
| `p8192-L64-i32-N1024` | evoxc-specialized-numpy | - | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree | 6803.188 | 1 | 76.231 |
| `p8192-L64-i32-N1024` | evoxc-batch-iree | - | - | - |
| `p8192-L64-i32-N1024` | evoxc-dense-iree-raised | - | - | - |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p1000-L64-i32-N16384` | batch_forward | `7035449061f860e1945549bff8c98c52` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 49152 |
| `p1000-L64-i32-N16384` | batch_forward | `098f9f61688905ebe83a8ff79b532a57` | evoxc-dense-iree | 49152 |
| `p1000-L64-i32-N16384` | batch_forward | `43da4024523ce5b762b4aca4f00f20f6` | evoxc-batch-iree | 49152 |
| `p1000-L64-i32-N16384` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N16384` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p1000-L64-i32-N16384` | forward | `8f1fb3a45a4ce951057b7b5799b9b38d` | evoxc-batch-iree | 3 |
| `p1000-L64-i32-N4096` | batch_forward | `611782005ce407d5553d589ccb1984f2` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 12288 |
| `p1000-L64-i32-N4096` | batch_forward | `96382e6f65f655c9f1561f701951a4d8` | evoxc-dense-iree | 12288 |
| `p1000-L64-i32-N4096` | batch_forward | `9e511409e1ce4748dc8b2901e75d47b7` | evoxc-batch-iree | 12288 |
| `p1000-L64-i32-N4096` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N4096` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p1000-L64-i32-N4096` | forward | `8f1fb3a45a4ce951057b7b5799b9b38d` | evoxc-batch-iree | 3 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `75caff0777c227066b76c4cb3e2e7342` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `abe3c30abe7d7179c99d1b080023029d` | evoxc-batch-iree | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `5a21db2c9c85d97de609c629c3352166` | evoxc-dense-iree-raised | 45056 |
| `p2048-L128-i32-N1024-d7` | forward | `9d808e31744c3a41559deca292ca3300` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 44 |
| `p2048-L128-i32-N1024-d7` | forward | `a3b46038536b19aa02022128f85ef1cd` | evoxc-batch-iree | 44 |
| `p2048-L128-i32-N1024-d7` | forward | `5f336e24b6a23726073a25859f64ef16` | evoxc-dense-iree-raised | 44 |
| `p2048-L256-i32-N1024-d8` | batch_forward | `02d41bedea4d18b57c38f1dfcc29dfce` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 58368 |
| `p2048-L256-i32-N1024-d8` | batch_forward | `c142a18548cdb3ac3b172bd5fa4d7f57` | evoxc-dense-iree-raised | 58368 |
| `p2048-L256-i32-N1024-d8` | forward | `8563bab0fecfe8a21baba9adebb5b3d0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 57 |
| `p2048-L256-i32-N1024-d8` | forward | `34e20d2d2bb5da7208885f734371eaee` | evoxc-dense-iree-raised | 57 |
| `p2048-L64-i32-N1024` | batch_forward | `d06579209e95ff930f767b55302b1121` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 17408 |
| `p2048-L64-i32-N1024` | batch_forward | `22a0311d4541bde860f83fdf4d52f0ec` | evoxc-auto, evoxc-dense-iree | 17408 |
| `p2048-L64-i32-N1024` | batch_forward | `1604465fd03a622a4536b41caa10153e` | evoxc-batch-iree | 17408 |
| `p2048-L64-i32-N1024` | forward | `db55b855de4ff8b86e7745aabd5009a1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 17 |
| `p2048-L64-i32-N1024` | forward | `a0769faa3a5d5ce06d77b9970007d213` | evoxc-auto, evoxc-dense-iree | 17 |
| `p2048-L64-i32-N1024` | forward | `e53263842c3f3b73191c0406e0c0b293` | evoxc-batch-iree | 17 |
| `p256-L64-i32-N16384` | batch_forward | `e2076327abc38a9a31385c87d661f3a0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N16384` | batch_forward | `f0a77f6542cb6fc8582a46c0963c5f9e` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N16384` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N16384` | forward | `563fc965117916c80f1bfa9ce3119fd9` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N4096` | batch_forward | `1298e7d5d2ecdaf398cefb2ff41cb491` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N4096` | batch_forward | `78e147ddc3a5edc9fd340187b107156d` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N4096` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N4096` | forward | `563fc965117916c80f1bfa9ce3119fd9` | evoxc-batch-iree | 0 |
| `p4096-L64-i32-N1024` | batch_forward | `7fe6de16612212715d9d9aefa8029a31` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `4c4d512caf5e1dfa0aa11dcd416c7b59` | evoxc-auto, evoxc-dense-iree | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `b02ff3fda11e861fdf0480dd87c874b2` | evoxc-batch-iree | 27648 |
| `p4096-L64-i32-N1024` | forward | `207dde0651d8b57527e04c3cd2697fd1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 27 |
| `p4096-L64-i32-N1024` | forward | `bad01add294ebf4e37f5d93ff5c39060` | evoxc-auto, evoxc-dense-iree | 27 |
| `p4096-L64-i32-N1024` | forward | `385949db50d1f9c5cde43c3af7b5241e` | evoxc-batch-iree | 27 |
| `p4096-L64-i64-N1024` | batch_forward | `6803df1b79855fe4b94835b2fe915b28` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 25600 |
| `p4096-L64-i64-N1024` | batch_forward | `e9a2ab8c4e283a010e5ffdfb8e7dcd6e` | evoxc-auto, evoxc-dense-iree | 25600 |
| `p4096-L64-i64-N1024` | batch_forward | `ca034995539a100a469672a5c1588aa2` | evoxc-batch-iree | 25600 |
| `p4096-L64-i64-N1024` | forward | `e641ac8392f183b8f3cf25051f905deb` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 25 |
| `p4096-L64-i64-N1024` | forward | `2d7089d22acc2c9293820b286836c8be` | evoxc-auto, evoxc-dense-iree | 25 |
| `p4096-L64-i64-N1024` | forward | `3cdff5efcad63ff8f2e83a4883d49c48` | evoxc-batch-iree | 25 |
| `p8192-L64-i32-N1024` | batch_forward | `f050b46e6514dfb419763d531969a932` | torch-per-tree, torch-batched | 57344 |
| `p8192-L64-i32-N1024` | batch_forward | `b489b7e475bf598d082fa49464504095` | evoxc-auto, evoxc-dense-iree | 57344 |
| `p8192-L64-i32-N1024` | forward | `4b2aced566852eb67a45a59f33294feb` | torch-per-tree, torch-batched | 56 |
| `p8192-L64-i32-N1024` | forward | `8760d13bd7adb24c6400bb98011d85be` | evoxc-auto, evoxc-dense-iree | 56 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p1000-L64-i32-N16384: 5/6, p1000-L64-i32-N4096: 5/6, p2048-L128-i32-N1024-d7: 5/6, p2048-L256-i32-N1024-d8: 4/5, p2048-L64-i32-N1024: 5/6, p256-L64-i32-N16384: 5/6, p256-L64-i32-N4096: 5/6, p4096-L64-i32-N1024: 5/6, p4096-L64-i64-N1024: 5/6, p8192-L64-i32-N1024: 3/4.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- **1 non-`ok` config(s)** in `campaign.json`: the record for these is absent or partial and no table cell above is a finished measurement.

| cell | arm | status | detail | wall_s |
|---|---|---|---|---|
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | timeout | `>900s` | 900.630 |

- **1 record file(s) absent or unreadable** (rendered as `-` above):

| cell | arm | reason | campaign status |
|---|---|---|---|
| `p2048-L256-i32-N1024-d8` | evoxc-batch-iree | record file missing | timeout |
