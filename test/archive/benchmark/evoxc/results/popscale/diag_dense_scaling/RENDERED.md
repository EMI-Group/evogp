# evoxc benchmark summary: diag_dense_scaling

Source: `test/benchmark/evoxc/results/popscale/diag_dense_scaling` — **6 records**, 3 cells x 2 arms.

The campaign ran **6 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **94.3 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-31T17:01:38Z |
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

| cell | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | 59.261 (58.968) | 0.158 (0.144) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N2048-d4` | 80.160 (80.011) | 0.147 (0.144) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N512-d4` | 6.148 (6.079) | 0.073 (0.069) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | evoxc-dense-iree | - | - |
| `p16384-L32-i8-N1024-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-fused-iree | - | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | 94.499 (94.168) | 0.135 (0.131) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N2048-d4` | 115.251 (114.817) | 0.144 (0.141) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N512-d4` | 12.901 (12.893) | 0.060 (0.058) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | evoxc-dense-iree | - | - |
| `p16384-L32-i8-N1024-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-fused-iree | - | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-dense-iree | evoxc-fused-iree | auto route | fastest |
|---|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | 0.744 (0.716) | 0.059 (0.056) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N2048-d4` | 0.716 (0.693) | 0.044 (0.044) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |
| `p8192-L32-i8-N512-d4` | 0.694 (0.655) | 0.044 (0.041) | dense/iree → dense (evoxc-dense-iree) | evoxc-fused-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | evoxc-dense-iree | - | - |
| `p16384-L32-i8-N1024-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-fused-iree | - | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|
| `p16384-L32-i8-N1024-d4` | 97.456 | 3.324 |
| `p8192-L32-i8-N2048-d4` | 100.558 | 3.212 |
| `p8192-L32-i8-N512-d4` | 25.595 | 2.970 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|
| `p16384-L32-i8-N1024-d4` | 94.566 / 2.890 (op 3%) | 0.496 / 2.828 (op 85%) |
| `p8192-L32-i8-N2048-d4` | 97.890 / 2.668 (op 3%) | 0.503 / 2.709 (op 84%) |
| `p8192-L32-i8-N512-d4` | 22.863 / 2.732 (op 11%) | 0.459 / 2.511 (op 85%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | evoxc-dense-iree | - | - |
| `p16384-L32-i8-N1024-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N2048-d4` | evoxc-fused-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-dense-iree | - | - |
| `p8192-L32-i8-N512-d4` | evoxc-fused-iree | - | - |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | evoxc-dense-iree | evoxc-fused-iree |
|---|---|---|
| `p16384-L32-i8-N1024-d4` | -0.084111 | -0.084111 |
| `p8192-L32-i8-N2048-d4` | -0.083110 | -0.083110 |
| `p8192-L32-i8-N512-d4` | -0.087671 | -0.087671 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | evoxc-dense-iree | 3533.637 | 1 | 59.261 |
| `p16384-L32-i8-N1024-d4` | evoxc-fused-iree | 481.817 | 0 | 0.158 |
| `p8192-L32-i8-N2048-d4` | evoxc-dense-iree | 3359.407 | 1 | 80.160 |
| `p8192-L32-i8-N2048-d4` | evoxc-fused-iree | 1589.499 | 0 | 0.147 |
| `p8192-L32-i8-N512-d4` | evoxc-dense-iree | 3345.933 | 1 | 6.148 |
| `p8192-L32-i8-N512-d4` | evoxc-fused-iree | 470.769 | 0 | 0.073 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16384-L32-i8-N1024-d4` | batch_forward | `69d7e4ae23de70715287e2ef8da9af16` | evoxc-dense-iree, evoxc-fused-iree | 88064 |
| `p16384-L32-i8-N1024-d4` | forward | `e71858ca2ff240dead137cbb4738bc40` | evoxc-dense-iree, evoxc-fused-iree | 86 |
| `p8192-L32-i8-N2048-d4` | batch_forward | `ed4a7f0f374519a3461194a0916933dd` | evoxc-dense-iree, evoxc-fused-iree | 83968 |
| `p8192-L32-i8-N2048-d4` | forward | `03d49d700d2192102735f4bcb31bc302` | evoxc-dense-iree, evoxc-fused-iree | 41 |
| `p8192-L32-i8-N512-d4` | batch_forward | `8058818e1897c3701117fa324e47dc13` | evoxc-dense-iree, evoxc-fused-iree | 20992 |
| `p8192-L32-i8-N512-d4` | forward | `03d49d700d2192102735f4bcb31bc302` | evoxc-dense-iree, evoxc-fused-iree | 41 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16384-L32-i8-N1024-d4: 2/2, p8192-L32-i8-N2048-d4: 2/2, p8192-L32-i8-N512-d4: 2/2.

## Caveats

- Every scheduled config in `campaign.json` finished `ok`.
