# evoxc benchmark summary: l128

Source: `test/benchmark/evoxc/results/batch/long-trees/l128` — **15 records**, 3 cells x 5 arms.

The campaign ran **15 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **563.2 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T10:02:13Z |
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

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 32.203 (31.782) | 37.606 (36.813) | 2.030 (2.013) | 2.167 (2.132) | 5.277 (5.242) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree |
| `p256-L128-i32-N1024-d6` | 46.444 (46.223) | 56.056 (55.441) | 2.040 (2.021) | 2.881 (2.865) | 5.379 (5.357) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree |
| `p256-L128-i32-N1024-d7` | 67.882 (66.492) | 91.122 (88.099) | 88.020 (86.893) | 3.518 (3.398) | 5.343 (5.244) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.86x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree | 15.86x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 14.86x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 6.10x | - |
| `p256-L128-i32-N1024-d6` | evoxc-specialized-numpy | 0.83x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree | 22.76x | - |
| `p256-L128-i32-N1024-d6` | evoxc-batch-iree | 16.12x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree-raised | 8.63x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.74x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree | 0.77x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 19.30x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 12.70x | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 24.285 (24.126) | 39.544 (38.534) | 2.835 (2.681) | 2.239 (2.001) | 6.037 (6.006) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d6` | 38.954 (38.639) | 56.833 (54.700) | 2.839 (2.630) | 2.313 (2.309) | 6.145 (6.000) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d7` | 61.720 (61.256) | 86.679 (85.612) | 85.674 (84.132) | 2.772 (2.655) | 6.183 (6.013) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.61x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree | 8.57x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 10.85x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 4.02x | - |
| `p256-L128-i32-N1024-d6` | evoxc-specialized-numpy | 0.69x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree | 13.72x | - |
| `p256-L128-i32-N1024-d6` | evoxc-batch-iree | 16.84x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree-raised | 6.34x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.71x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree | 0.72x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 22.27x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 9.98x | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | 25.341 (24.795) | 36.966 (36.116) | 0.942 (0.901) | 0.763 (0.735) | 1.238 (1.207) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L128-i32-N1024-d6` | 39.429 (38.776) | 54.147 (52.736) | 0.854 (0.846) | 1.101 (1.067) | 1.254 (1.237) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree |
| `p256-L128-i32-N1024-d7` | 61.842 (61.364) | 84.050 (79.947) | 144.789 (142.717) | 1.217 (1.200) | 1.207 (1.171) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree-raised |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 0.69x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree | 26.90x | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 33.22x | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 20.47x | - |
| `p256-L128-i32-N1024-d6` | evoxc-specialized-numpy | 0.73x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree | 46.18x | - |
| `p256-L128-i32-N1024-d6` | evoxc-batch-iree | 35.80x | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree-raised | 31.45x | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 0.74x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree | 0.43x | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 50.82x | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 51.22x | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | - | - | - | - | - |
| `p256-L128-i32-N1024-d6` | - | - | - | - | - |
| `p256-L128-i32-N1024-d7` | - | - | - | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|
| `p256-L128-i32-N1024` | - | - | - | - | - |
| `p256-L128-i32-N1024-d6` | - | - | - | - | - |
| `p256-L128-i32-N1024-d7` | - | - | - | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L128-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L128-i32-N1024-d6` | evoxc-specialized-numpy | - | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree | - | - |
| `p256-L128-i32-N1024-d6` | evoxc-batch-iree | - | - |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree-raised | - | - |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | - | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree | - | - |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | - | - |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | evoxc-specialized-numpy | 832.176 | 219 | 37.606 |
| `p256-L128-i32-N1024` | evoxc-dense-iree | 6607.978 | 1 | 2.030 |
| `p256-L128-i32-N1024` | evoxc-batch-iree | 13224.084 | 0 | 2.167 |
| `p256-L128-i32-N1024` | evoxc-dense-iree-raised | 14439.017 | 1 | 5.277 |
| `p256-L128-i32-N1024-d6` | evoxc-specialized-numpy | 1169.819 | 218 | 56.056 |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree | 6522.470 | 1 | 2.040 |
| `p256-L128-i32-N1024-d6` | evoxc-batch-iree | 22761.810 | 0 | 2.881 |
| `p256-L128-i32-N1024-d6` | evoxc-dense-iree-raised | 14278.435 | 1 | 5.379 |
| `p256-L128-i32-N1024-d7` | evoxc-specialized-numpy | 1840.323 | 218 | 91.122 |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree | 68987.670 | 218 | 88.020 |
| `p256-L128-i32-N1024-d7` | evoxc-batch-iree | 37918.256 | 0 | 3.518 |
| `p256-L128-i32-N1024-d7` | evoxc-dense-iree-raised | 14483.252 | 1 | 5.343 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L128-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 0 |
| `p256-L128-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L128-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 0 |
| `p256-L128-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p256-L128-i32-N1024-d6` | batch_forward | `2d6cc3d6e58ecc9b45253ecfac2a962b` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 4096 |
| `p256-L128-i32-N1024-d6` | batch_forward | `17d5787ee8303b5afa7d72ae14ffcc3b` | evoxc-batch-iree | 4096 |
| `p256-L128-i32-N1024-d6` | forward | `1b5716936e82406e1005ece8de6beb3b` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 4 |
| `p256-L128-i32-N1024-d6` | forward | `615a09be43fd7e5ba90cfd07a90d9716` | evoxc-batch-iree | 4 |
| `p256-L128-i32-N1024-d7` | batch_forward | `e40e48cbaf342851e6f386b0de8796a9` | torch-per-tree, evoxc-specialized-numpy | 7168 |
| `p256-L128-i32-N1024-d7` | batch_forward | `c7a178e37d67cfc947d6b517cc011de1` | evoxc-dense-iree | 7168 |
| `p256-L128-i32-N1024-d7` | batch_forward | `8885b2deaace90bd0bec657469a62965` | evoxc-batch-iree | 7168 |
| `p256-L128-i32-N1024-d7` | batch_forward | `d25aaf62a3e18911bf9e8975883fa945` | evoxc-dense-iree-raised | 7168 |
| `p256-L128-i32-N1024-d7` | forward | `e854b6fd35fdc829f6fe006bc2e68720` | torch-per-tree, evoxc-specialized-numpy | 7 |
| `p256-L128-i32-N1024-d7` | forward | `e0cbf5e906839a97dea59a32d8d753af` | evoxc-dense-iree | 7 |
| `p256-L128-i32-N1024-d7` | forward | `fb6ca5e78b01443ff8646277964a0b2a` | evoxc-batch-iree | 7 |
| `p256-L128-i32-N1024-d7` | forward | `8aa150adfa4f577bd16448667ca3912a` | evoxc-dense-iree-raised | 7 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L128-i32-N1024: 4/5, p256-L128-i32-N1024-d6: 4/5, p256-L128-i32-N1024-d7: 5/5.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
