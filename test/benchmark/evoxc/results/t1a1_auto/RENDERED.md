# evoxc benchmark summary: t1a1_auto

Source: `test/benchmark/evoxc/results/t1a1_auto` — **23 records**, 6 cells x 4 arms.

The campaign ran **23 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **323.6 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `1800.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **1 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-31T06:27:44Z |
| git SHA (`tree_root_sha`) | 576832c68fee4e992251a62961c5f71f83669835 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A109/src` |
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

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 30.733 (30.531) | 4.031 (4.019) | 0.138 (0.113) | 1.105 (1.087) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 31.270 (31.008) | 4.610 (4.299) | 0.132 (0.125) | 2.126 (2.108) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 121.208 (120.378) | 4.859 (4.847) | 0.166 (0.120) | 5.831 (5.795) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 585.147 (582.574) | 26.653 (26.623) | 0.215 (0.199) | - | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 30.457 (29.082) | 5.046 (5.040) | 0.124 (0.120) | 5.872 (5.812) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 494.558 (484.560) | 10.359 (10.330) | 0.183 (0.166) | 36.969 (36.880) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 222.38x | 29.17x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 27.82x | 3.65x |
| `p256-L64-i32-N1024` | evoxc-auto | 237.79x | 35.06x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 14.71x | 2.17x |
| `p1000-L64-i32-N1024` | evoxc-auto | 730.61x | 29.29x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 20.79x | 0.83x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 2715.30x | 123.68x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 244.83x | 40.56x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 5.19x | 0.86x |
| `p4096-L64-i32-N1024` | evoxc-auto | 2703.98x | 56.64x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 13.38x | 0.28x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 23.481 (23.465) | 3.989 (3.951) | 0.102 (0.095) | 1.808 (1.776) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 24.009 (22.696) | 4.584 (4.274) | 0.117 (0.106) | 2.949 (2.881) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 92.231 (91.326) | 4.857 (4.804) | 0.115 (0.110) | 7.701 (7.544) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 517.623 (515.974) | 26.627 (26.610) | 0.188 (0.178) | - | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 23.282 (22.657) | 5.024 (4.998) | 0.131 (0.113) | 7.678 (7.651) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 372.206 (367.158) | 10.323 (10.304) | 0.153 (0.143) | 47.018 (46.428) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 229.53x | 39.00x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 12.98x | 2.21x |
| `p256-L64-i32-N1024` | evoxc-auto | 205.38x | 39.21x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.14x | 1.55x |
| `p1000-L64-i32-N1024` | evoxc-auto | 805.51x | 42.42x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 11.98x | 0.63x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 2754.78x | 141.71x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 178.13x | 38.44x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 3.03x | 0.65x |
| `p4096-L64-i32-N1024` | evoxc-auto | 2429.54x | 67.38x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 7.92x | 0.22x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 25.689 (24.135) | 3.970 (3.961) | 0.105 (0.098) | 0.723 (0.694) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 23.586 (22.840) | 4.569 (4.206) | 0.107 (0.097) | 0.884 (0.836) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 95.329 (94.293) | 4.442 (4.431) | 0.108 (0.095) | 0.858 (0.839) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 530.318 (528.707) | 15.640 (14.621) | 0.129 (0.126) | - | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 23.912 (23.799) | 4.296 (4.270) | 0.098 (0.092) | 0.879 (0.838) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 381.605 (380.729) | 4.493 (4.463) | 0.099 (0.096) | 0.902 (0.894) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 243.96x | 37.70x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 35.54x | 5.49x |
| `p256-L64-i32-N1024` | evoxc-auto | 220.23x | 42.66x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 26.67x | 5.17x |
| `p1000-L64-i32-N1024` | evoxc-auto | 881.86x | 41.10x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 111.14x | 5.18x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 4114.18x | 121.34x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 243.25x | 43.70x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 27.21x | 4.89x |
| `p4096-L64-i32-N1024` | evoxc-auto | 3866.31x | 45.52x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 422.88x | 4.98x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 17.783 | 23.722 | 2.691 | 5.723 |
| `p256-L64-i32-N1024` | 17.863 | 21.512 | 2.817 | 6.733 |
| `p1000-L64-i32-N1024` | 57.194 | 68.047 | 2.699 | 12.589 |
| `p2048-L128-i32-N1024-d7` | 120.793 | 307.116 | 2.840 | - |
| `p256-L64-i32-N4096` | 15.967 | 22.100 | 2.583 | 10.636 |
| `p4096-L64-i32-N1024` | 208.076 | 302.590 | 2.809 | 56.370 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 15.238 / 2.545 (op 14%) | 21.194 / 2.528 (op 11%) | 0.252 / 2.438 (op 91%) | 3.092 / 2.631 (op 46%) |
| `p256-L64-i32-N1024` | 15.180 / 2.683 (op 15%) | 18.788 / 2.723 (op 13%) | 0.267 / 2.551 (op 91%) | 4.247 / 2.486 (op 37%) |
| `p1000-L64-i32-N1024` | 54.660 / 2.534 (op 4%) | 65.475 / 2.572 (op 4%) | 0.269 / 2.430 (op 90%) | 10.088 / 2.501 (op 20%) |
| `p2048-L128-i32-N1024-d7` | 117.917 / 2.875 (op 2%) | 304.352 / 2.764 (op 1%) | 0.251 / 2.590 (op 91%) | - |
| `p256-L64-i32-N4096` | 13.412 / 2.555 (op 16%) | 19.542 / 2.558 (op 12%) | 0.236 / 2.347 (op 91%) | 8.084 / 2.553 (op 24%) |
| `p4096-L64-i32-N1024` | 204.694 / 3.382 (op 2%) | 299.821 / 2.769 (op 1%) | 0.256 / 2.553 (op 91%) | 53.667 / 2.703 (op 5%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 6.61x | 8.82x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3.11x | 4.15x |
| `p256-L64-i32-N1024` | evoxc-auto | 6.34x | 7.64x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 2.65x | 3.19x |
| `p1000-L64-i32-N1024` | evoxc-auto | 21.19x | 25.21x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 4.54x | 5.41x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 42.53x | 108.14x |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 6.18x | 8.56x |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 1.50x | 2.08x |
| `p4096-L64-i32-N1024` | evoxc-auto | 74.08x | 107.73x |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 3.69x | 5.37x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | -0.084430 | -0.084430 | -0.084430 | -0.084430 |
| `p256-L64-i32-N1024` | -0.084812 | -0.084812 | -0.084812 | -0.084812 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 | -0.084145 | -0.084145 |
| `p2048-L128-i32-N1024-d7` | -0.084304 | -0.084304 | -0.084304 | - |
| `p256-L64-i32-N4096` | -0.084455 | -0.084455 | -0.084455 | -0.084455 |
| `p4096-L64-i32-N1024` | -0.084278 | -0.084278 | -0.084278 | -0.084278 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 458.242 | 0 | 0.138 |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3353.735 | 1 | 1.105 |
| `p256-L64-i32-N1024` | evoxc-auto | 445.853 | 0 | 0.132 |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6571.687 | 1 | 2.126 |
| `p1000-L64-i32-N1024` | evoxc-auto | 467.622 | 0 | 0.166 |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6635.580 | 1 | 5.831 |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 448.891 | 0 | 0.215 |
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | - | - | - |
| `p256-L64-i32-N4096` | evoxc-auto | 468.408 | 0 | 0.124 |
| `p256-L64-i32-N4096` | evoxc-dense-iree | 6520.069 | 1 | 5.872 |
| `p4096-L64-i32-N1024` | evoxc-auto | 461.424 | 0 | 0.183 |
| `p4096-L64-i32-N1024` | evoxc-dense-iree | 6687.751 | 1 | 36.969 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 0 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `75caff0777c227066b76c4cb3e2e7342` | torch-per-tree, torch-batched | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `5a21db2c9c85d97de609c629c3352166` | evoxc-auto | 45056 |
| `p2048-L128-i32-N1024-d7` | forward | `9d808e31744c3a41559deca292ca3300` | torch-per-tree, torch-batched | 44 |
| `p2048-L128-i32-N1024-d7` | forward | `5f336e24b6a23726073a25859f64ef16` | evoxc-auto | 44 |
| `p256-L64-i32-N4096` | batch_forward | `1298e7d5d2ecdaf398cefb2ff41cb491` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 0 |
| `p256-L64-i32-N4096` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-dense-iree | 0 |
| `p4096-L64-i32-N1024` | batch_forward | `7fe6de16612212715d9d9aefa8029a31` | torch-per-tree, torch-batched | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `4c4d512caf5e1dfa0aa11dcd416c7b59` | evoxc-auto, evoxc-dense-iree | 27648 |
| `p4096-L64-i32-N1024` | forward | `207dde0651d8b57527e04c3cd2697fd1` | torch-per-tree, torch-batched | 27 |
| `p4096-L64-i32-N1024` | forward | `bad01add294ebf4e37f5d93ff5c39060` | evoxc-auto, evoxc-dense-iree | 27 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 4/4, p256-L64-i32-N1024: 4/4, p1000-L64-i32-N1024: 4/4, p2048-L128-i32-N1024-d7: 3/3, p256-L64-i32-N4096: 4/4, p4096-L64-i32-N1024: 4/4.

## Caveats

- Every scheduled config in `campaign.json` finished `ok`.
- **1 config(s) never ran**: excluded by the routing scope rule for compile cost, so they have no row anywhere above.

| cell | arm | reason |
|---|---|---|
| `p2048-L128-i32-N1024-d7` | evoxc-dense-iree | dropped by the routing scope rule |
