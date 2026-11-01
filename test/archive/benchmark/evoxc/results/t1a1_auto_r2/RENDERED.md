# evoxc benchmark summary: t1a1_auto_r2

Source: `test/benchmark/evoxc/results/t1a1_auto_r2` — **12 records**, 6 cells x 2 arms.

The campaign ran **12 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **113.6 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `1800.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-31T08:54:46Z |
| git SHA (`tree_root_sha`) | 3bdb31f3a9590f84f6bf6645ff30f4a6d7c38c67 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A142/src` |
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
| `p256-L32-i8-N1024` | 4.039 (4.028) | 0.120 (0.108) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.564 (4.531) | 0.100 (0.090) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.875 (4.842) | 0.110 (0.103) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 26.508 (26.445) | 0.173 (0.159) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 4.974 (4.965) | 0.113 (0.096) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 10.584 (10.546) | 0.139 (0.129) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 33.60x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 45.64x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 44.40x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 153.40x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 43.90x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 76.20x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-batched | evoxc-auto | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 3.996 (3.983) | 0.086 (0.085) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.460 (4.416) | 0.078 (0.076) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.851 (4.841) | 0.095 (0.088) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 26.497 (26.411) | 0.146 (0.143) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 4.967 (4.961) | 0.085 (0.083) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 10.545 (10.497) | 0.126 (0.118) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 46.41x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 57.18x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 51.22x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 181.48x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 58.51x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 83.89x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-batched | evoxc-auto | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 4.016 (4.006) | 0.085 (0.082) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N1024` | 4.399 (4.336) | 0.073 (0.071) | auto/iree → fused | evoxc-auto |
| `p1000-L64-i32-N1024` | 4.536 (4.469) | 0.082 (0.074) | auto/iree → fused | evoxc-auto |
| `p2048-L128-i32-N1024-d7` | 16.125 (16.042) | 0.111 (0.106) | auto/iree → fused | evoxc-auto |
| `p256-L64-i32-N4096` | 4.340 (4.276) | 0.072 (0.071) | auto/iree → fused | evoxc-auto |
| `p4096-L64-i32-N1024` | 4.439 (4.425) | 0.077 (0.075) | auto/iree → fused | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 47.13x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 59.94x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 55.32x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 145.27x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 60.61x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 57.51x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-batched | evoxc-auto |
|---|---|---|
| `p256-L32-i8-N1024` | 23.121 | 2.800 |
| `p256-L64-i32-N1024` | 22.632 | 2.800 |
| `p1000-L64-i32-N1024` | 69.113 | 3.052 |
| `p2048-L128-i32-N1024-d7` | 313.877 | 3.086 |
| `p256-L64-i32-N4096` | 22.134 | 2.825 |
| `p4096-L64-i32-N1024` | 315.736 | 3.035 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-batched | evoxc-auto |
|---|---|---|
| `p256-L32-i8-N1024` | 20.620 / 2.501 (op 11%) | 0.405 / 2.395 (op 86%) |
| `p256-L64-i32-N1024` | 20.085 / 2.547 (op 11%) | 0.388 / 2.411 (op 86%) |
| `p1000-L64-i32-N1024` | 66.512 / 2.602 (op 4%) | 0.429 / 2.623 (op 86%) |
| `p2048-L128-i32-N1024-d7` | 310.972 / 2.905 (op 1%) | 0.430 / 2.656 (op 86%) |
| `p256-L64-i32-N4096` | 19.591 / 2.543 (op 11%) | 0.433 / 2.391 (op 85%) |
| `p4096-L64-i32-N1024` | 312.936 / 2.800 (op 1%) | 0.417 / 2.618 (op 86%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | - | 8.26x |
| `p256-L64-i32-N1024` | evoxc-auto | - | 8.08x |
| `p1000-L64-i32-N1024` | evoxc-auto | - | 22.65x |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | - | 101.70x |
| `p256-L64-i32-N4096` | evoxc-auto | - | 7.84x |
| `p4096-L64-i32-N1024` | evoxc-auto | - | 104.02x |

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
| `p256-L32-i8-N1024` | evoxc-auto | 495.397 | 0 | 0.120 |
| `p256-L64-i32-N1024` | evoxc-auto | 504.523 | 0 | 0.100 |
| `p1000-L64-i32-N1024` | evoxc-auto | 508.587 | 0 | 0.110 |
| `p2048-L128-i32-N1024-d7` | evoxc-auto | 499.424 | 0 | 0.173 |
| `p256-L64-i32-N4096` | evoxc-auto | 501.863 | 0 | 0.113 |
| `p4096-L64-i32-N1024` | evoxc-auto | 496.930 | 0 | 0.139 |

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
