# evoxc benchmark summary: gp

Source: `test/benchmark/evoxc/results/batch/long-trees/gp` — **8 records**, 2 cells x 4 arms.

The campaign ran **8 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **322.1 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T10:28:35Z |
| git SHA (`tree_root_sha`) | f8dd665f6826854859201c6d23ae8943f09fb728 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A34/src` |
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

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 30.730 (30.508) | 41.511 (40.046) | 2.210 (2.188) | 5.335 (5.260) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d7` | 74.226 (73.505) | 104.349 (102.693) | 3.777 (3.712) | 5.468 (5.455) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.74x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 13.91x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 5.76x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.71x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 19.65x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 13.57x | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 23.410 (22.945) | 41.062 (39.342) | 2.179 (2.059) | 6.319 (6.309) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d7` | 65.539 (64.962) | 103.429 (102.463) | 2.965 (2.890) | 6.210 (6.185) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.57x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 10.74x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 3.70x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.63x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 22.11x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 10.55x | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 23.932 (23.559) | 38.841 (37.891) | 0.754 (0.717) | 1.323 (1.304) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d7` | 66.890 (66.390) | 96.855 (95.599) | 1.308 (1.278) | 1.355 (1.318) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.62x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 31.72x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 18.09x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.69x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 51.14x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 49.38x | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | 22.710 | 395.474 | 5317.995 | 11.552 |
| `p256-L128-i32-N1024-d7` | 13.959 | 102.934 | 2177.223 | 11.182 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | 19.930 / 2.780 (op 12%) | 392.311 / 3.162 (op 1%) | 5314.776 / 3.219 (op 0%) | 8.830 / 2.722 (op 24%) |
| `p256-L128-i32-N1024-d7` | 11.263 / 2.696 (op 19%) | 100.078 / 2.856 (op 3%) | 2173.981 / 3.242 (op 0%) | 8.446 / 2.736 (op 24%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.06x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 0.00x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 1.97x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.14x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 0.01x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 1.25x | - |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | -0.084197 | -0.084197 | -0.084197 | -0.084197 |
| `p256-L128-i32-N1024-d7` | -0.084252 | -0.084252 | -0.084252 | -0.084252 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 894.603 | 219 | 41.511 |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 12500.341 | 0 | 2.210 |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 15781.909 | 1 | 5.335 |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 2074.574 | 218 | 104.349 |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 41738.368 | 0 | 3.777 |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 15787.415 | 1 | 5.468 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree-raised | 0 |
| `p256-L128-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L128-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree-raised | 0 |
| `p256-L128-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p256-L128-i32-N1024-d7` | batch_forward | `e40e48cbaf342851e6f386b0de8796a9` | torch-per-tree, evoxc-specialized-numpy | 7168 |
| `p256-L128-i32-N1024-d7` | batch_forward | `8885b2deaace90bd0bec657469a62965` | evoxc-batch-iree | 7168 |
| `p256-L128-i32-N1024-d7` | batch_forward | `d25aaf62a3e18911bf9e8975883fa945` | evoxc-dense-iree-raised | 7168 |
| `p256-L128-i32-N1024-d7` | forward | `e854b6fd35fdc829f6fe006bc2e68720` | torch-per-tree, evoxc-specialized-numpy | 7 |
| `p256-L128-i32-N1024-d7` | forward | `fb6ca5e78b01443ff8646277964a0b2a` | evoxc-batch-iree | 7 |
| `p256-L128-i32-N1024-d7` | forward | `8aa150adfa4f577bd16448667ca3912a` | evoxc-dense-iree-raised | 7 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L128-i32-N1024: 4/4, p256-L128-i32-N1024-d7: 4/4.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
