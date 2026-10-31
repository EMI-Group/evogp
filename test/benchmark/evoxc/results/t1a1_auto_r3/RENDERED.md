# evoxc benchmark summary: t1a1_auto_r3

Source: `test/benchmark/evoxc/results/t1a1_auto_r3` — **12 records**, 6 cells x 2 arms.

The campaign ran **12 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **113.0 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `1800.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-31T11:53:50Z |
| git SHA (`tree_root_sha`) | 919f7aff18e97c1052c0db5adc31f0de7f8f78ab |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A150/src` |
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

| cell | torch-batched | evoxc-auto | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 4.035 (4.021) | 0.067 (0.064) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.460 (4.456) | 0.057 (0.050) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.903 (4.862) | 0.062 (0.057) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 26.349 (26.341) | 0.126 (0.125) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 5.041 (5.004) | 0.065 (0.062) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 10.451 (10.426) | 0.089 (0.084) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 60.23x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 77.97x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 79.21x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 208.79x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 77.56x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 117.96x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-batched | evoxc-auto | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 3.954 (3.942) | 0.049 (0.043) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.869 (4.852) | 0.045 (0.042) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.841 (4.827) | 0.050 (0.047) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 26.364 (26.318) | 0.115 (0.111) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 5.020 (4.974) | 0.052 (0.049) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 10.415 (10.409) | 0.079 (0.076) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 80.20x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 108.43x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 96.82x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 228.86x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 96.18x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 131.83x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-batched | evoxc-auto | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 3.904 (3.571) | 0.045 (0.041) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.826 (4.807) | 0.040 (0.038) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.665 (4.622) | 0.039 (0.038) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 15.889 (15.720) | 0.072 (0.070) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 4.522 (4.432) | 0.039 (0.036) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 4.450 (4.421) | 0.044 (0.041) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 85.79x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 120.95x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 119.61x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 220.07x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 115.65x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 101.84x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-batched | evoxc-auto |
|---|---|---|
| `p256-L32-i8-N1024` | 23.376 | 3.053 |
| `p256-L64-i32-N1024` | 23.674 | 2.842 |
| `p1000-L64-i32-N1024` | 69.813 | 2.841 |
| `p2048-L128-i32-N1024-d7` | 302.887 | 3.282 |
| `p256-L64-i32-N4096` | 22.248 | 2.900 |
| `p4096-L64-i32-N1024` | 306.385 | 3.072 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-batched | evoxc-auto |
|---|---|---|
| `p256-L32-i8-N1024` | 20.901 / 2.475 (op 11%) | 0.471 / 2.582 (op 85%) |
| `p256-L64-i32-N1024` | 21.026 / 2.648 (op 11%) | 0.437 / 2.404 (op 85%) |
| `p1000-L64-i32-N1024` | 67.206 / 2.608 (op 4%) | 0.435 / 2.405 (op 85%) |
| `p2048-L128-i32-N1024-d7` | 300.073 / 2.814 (op 1%) | 0.468 / 2.815 (op 86%) |
| `p256-L64-i32-N4096` | 19.620 / 2.629 (op 12%) | 0.433 / 2.467 (op 85%) |
| `p4096-L64-i32-N1024` | 303.551 / 2.834 (op 1%) | 0.466 / 2.606 (op 85%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 7.66x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 8.33x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 24.58x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 92.28x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 7.67x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 99.75x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-batched | evoxc-auto |
|---|---|---|
| `p256-L32-i8-N1024` | -0.084430 | -0.084430 |
| `p256-L64-i32-N1024` | -0.084812 | -0.084812 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 |
| `p2048-L128-i32-N1024-d7` | -0.084304 | -0.084304 |
| `p256-L64-i32-N4096` | -0.084455 | -0.084455 |
| `p4096-L64-i32-N1024` | -0.084278 | -0.084278 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 526.936 | 0 | 0.067 |
| `p256-L64-i32-N1024` | evoxc-auto | 492.406 | 0 | 0.057 |
| `p1000-L64-i32-N1024` | evoxc-auto | 487.651 | 0 | 0.062 |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 500.811 | 0 | 0.126 |
| `p256-L64-i32-N4096` | evoxc-auto | 496.298 | 0 | 0.065 |
| `p4096-L64-i32-N1024` | evoxc-auto | 494.136 | 0 | 0.089 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-batched, evoxc-auto | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-batched, evoxc-auto | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-batched, evoxc-auto | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-batched, evoxc-auto | 0 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-batched | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-batched | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto | 3 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `75caff0777c227066b76c4cb3e2e7342` | torch-batched | 45056 |
| `p2048-L128-i32-N1024-d7` | batch_forward | `5a21db2c9c85d97de609c629c3352166` | evoxc-auto | 45056 |
| `p2048-L128-i32-N1024-d7` | forward | `9d808e31744c3a41559deca292ca3300` | torch-batched | 44 |
| `p2048-L128-i32-N1024-d7` | forward | `5f336e24b6a23726073a25859f64ef16` | evoxc-auto | 44 |
| `p256-L64-i32-N4096` | batch_forward | `1298e7d5d2ecdaf398cefb2ff41cb491` | torch-batched, evoxc-auto | 0 |
| `p256-L64-i32-N4096` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-batched, evoxc-auto | 0 |
| `p4096-L64-i32-N1024` | batch_forward | `7fe6de16612212715d9d9aefa8029a31` | torch-batched | 27648 |
| `p4096-L64-i32-N1024` | batch_forward | `4c4d512caf5e1dfa0aa11dcd416c7b59` | evoxc-auto | 27648 |
| `p4096-L64-i32-N1024` | forward | `207dde0651d8b57527e04c3cd2697fd1` | torch-batched | 27 |
| `p4096-L64-i32-N1024` | forward | `bad01add294ebf4e37f5d93ff5c39060` | evoxc-auto | 27 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 2/2, p256-L64-i32-N1024: 2/2, p1000-L64-i32-N1024: 2/2, p2048-L128-i32-N1024-d7: 2/2, p256-L64-i32-N4096: 2/2, p4096-L64-i32-N1024: 2/2.

## Caveats

- Every scheduled config in `campaign.json` finished `ok`.
