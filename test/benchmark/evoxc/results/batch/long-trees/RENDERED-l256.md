# evoxc benchmark summary: l256

Source: `test/benchmark/evoxc/results/batch/long-trees/l256` — **10 records**, 2 cells x 5 arms.

The campaign ran **10 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **657.5 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T10:11:43Z |
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
| `p256-L256-i32-N1024` | 29.815 (28.412) | 41.449 (41.029) | 2.126 (2.088) | 2.237 (2.119) | 19.497 (19.465) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree |
| `p256-L256-i32-N1024-d8` | 108.797 (108.188) | 158.373 (156.928) | 110.118 (109.293) | 4.676 (4.563) | 19.662 (19.544) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L256-i32-N1024` | evoxc-specialized-numpy | 0.72x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree | 14.02x | - |
| `p256-L256-i32-N1024` | evoxc-batch-iree | 13.33x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree-raised | 1.53x | - |
| `p256-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.69x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree | 0.99x | - |
| `p256-L256-i32-N1024-d8` | evoxc-batch-iree | 23.27x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 5.53x | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L256-i32-N1024` | 22.950 (22.648) | 42.826 (42.440) | 2.884 (2.760) | 2.010 (1.989) | 20.356 (20.242) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L256-i32-N1024-d8` | 99.729 (99.530) | 153.206 (152.032) | 108.937 (106.856) | 3.955 (3.747) | 20.548 (20.409) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L256-i32-N1024` | evoxc-specialized-numpy | 0.54x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree | 7.96x | - |
| `p256-L256-i32-N1024` | evoxc-batch-iree | 11.42x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree-raised | 1.13x | - |
| `p256-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.65x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree | 0.92x | - |
| `p256-L256-i32-N1024-d8` | evoxc-batch-iree | 25.22x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 4.85x | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p256-L256-i32-N1024` | 23.544 (22.159) | 40.035 (39.408) | 0.965 (0.896) | 0.748 (0.715) | 2.178 (2.103) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |
| `p256-L256-i32-N1024-d8` | 101.124 (99.563) | 143.060 (141.304) | 151.452 (150.886) | 1.607 (1.572) | 2.108 (2.098) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L256-i32-N1024` | evoxc-specialized-numpy | 0.59x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree | 24.41x | - |
| `p256-L256-i32-N1024` | evoxc-batch-iree | 31.48x | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree-raised | 10.81x | - |
| `p256-L256-i32-N1024-d8` | evoxc-specialized-numpy | 0.71x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree | 0.67x | - |
| `p256-L256-i32-N1024-d8` | evoxc-batch-iree | 62.92x | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 47.97x | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|
| `p256-L256-i32-N1024` | - | - | - | - | - |
| `p256-L256-i32-N1024-d8` | - | - | - | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|---|
| `p256-L256-i32-N1024` | - | - | - | - | - |
| `p256-L256-i32-N1024-d8` | - | - | - | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L256-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L256-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L256-i32-N1024` | evoxc-dense-iree-raised | - | - |
| `p256-L256-i32-N1024-d8` | evoxc-specialized-numpy | - | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree | - | - |
| `p256-L256-i32-N1024-d8` | evoxc-batch-iree | - | - |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree-raised | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L256-i32-N1024` | evoxc-specialized-numpy | 911.286 | 219 | 41.449 |
| `p256-L256-i32-N1024` | evoxc-dense-iree | 7217.800 | 1 | 2.126 |
| `p256-L256-i32-N1024` | evoxc-batch-iree | 12037.503 | 0 | 2.237 |
| `p256-L256-i32-N1024` | evoxc-dense-iree-raised | 41872.099 | 1 | 19.497 |
| `p256-L256-i32-N1024-d8` | evoxc-specialized-numpy | 3297.027 | 217 | 158.373 |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree | 81070.521 | 217 | 110.118 |
| `p256-L256-i32-N1024-d8` | evoxc-batch-iree | 74726.553 | 0 | 4.676 |
| `p256-L256-i32-N1024-d8` | evoxc-dense-iree-raised | 43905.382 | 1 | 19.662 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L256-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 0 |
| `p256-L256-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L256-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, evoxc-specialized-numpy, evoxc-dense-iree, evoxc-dense-iree-raised | 0 |
| `p256-L256-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p256-L256-i32-N1024-d8` | batch_forward | `765590fb0588959156ebcf0d82cb216e` | torch-per-tree, evoxc-specialized-numpy | 5120 |
| `p256-L256-i32-N1024-d8` | batch_forward | `bf6da51ffdf75bceb241fd5bb9253e8b` | evoxc-dense-iree | 5120 |
| `p256-L256-i32-N1024-d8` | batch_forward | `e2eeb77ca2fbc57b1f00fc4e1b2ca501` | evoxc-batch-iree | 5120 |
| `p256-L256-i32-N1024-d8` | batch_forward | `34f8289423a0d5b7b9d7f15564eb1193` | evoxc-dense-iree-raised | 5120 |
| `p256-L256-i32-N1024-d8` | forward | `b81c5bfce461cee05fda4df8aa685cc7` | torch-per-tree, evoxc-specialized-numpy | 5 |
| `p256-L256-i32-N1024-d8` | forward | `a8ac163094e999773542c79fcb8d2621` | evoxc-dense-iree | 5 |
| `p256-L256-i32-N1024-d8` | forward | `e537ed01f8b3a2746fccad2f464a4c53` | evoxc-batch-iree | 5 |
| `p256-L256-i32-N1024-d8` | forward | `23b82924d25565f68847617d922c62a1` | evoxc-dense-iree-raised | 5 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L256-i32-N1024: 4/5, p256-L256-i32-N1024-d8: 5/5.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
