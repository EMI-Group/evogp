# evoxc benchmark summary: p1000

Source: `test/benchmark/evoxc/results/batch/long-trees/p1000` — **4 records**, 1 cells x 4 arms.

The campaign ran **4 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **339.1 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T10:22:48Z |
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
| `p1000-L128-i32-N1024` | 128.302 (126.869) | 153.015 (150.426) | 6.565 (6.443) | 19.478 (19.360) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L128-i32-N1024` | evoxc-specialized-numpy | 0.84x | - |
| `p1000-L128-i32-N1024` | evoxc-batch-iree | 19.54x | - |
| `p1000-L128-i32-N1024` | evoxc-dense-iree-raised | 6.59x | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|
| `p1000-L128-i32-N1024` | 96.646 (95.651) | 165.811 (164.298) | 5.861 (5.555) | 21.415 (21.215) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L128-i32-N1024` | evoxc-specialized-numpy | 0.58x | - |
| `p1000-L128-i32-N1024` | evoxc-batch-iree | 16.49x | - |
| `p1000-L128-i32-N1024` | evoxc-dense-iree-raised | 4.51x | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised | auto route | fastest |
|---|---|---|---|---|---|---|
| `p1000-L128-i32-N1024` | 98.910 (97.440) | 148.359 (141.500) | 1.305 (1.276) | 1.271 (1.246) | specialized/numpy → specialized (evoxc-specialized-numpy) | evoxc-dense-iree-raised |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L128-i32-N1024` | evoxc-specialized-numpy | 0.67x | - |
| `p1000-L128-i32-N1024` | evoxc-batch-iree | 75.81x | - |
| `p1000-L128-i32-N1024` | evoxc-dense-iree-raised | 77.81x | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|
| `p1000-L128-i32-N1024` | - | - | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | evoxc-specialized-numpy | evoxc-batch-iree | evoxc-dense-iree-raised |
|---|---|---|---|---|
| `p1000-L128-i32-N1024` | - | - | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p1000-L128-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L128-i32-N1024` | evoxc-batch-iree | - | - |
| `p1000-L128-i32-N1024` | evoxc-dense-iree-raised | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p1000-L128-i32-N1024` | evoxc-specialized-numpy | 3301.129 | 821 | 153.015 |
| `p1000-L128-i32-N1024` | evoxc-batch-iree | 178384.170 | 0 | 6.565 |
| `p1000-L128-i32-N1024` | evoxc-dense-iree-raised | 15106.255 | 1 | 19.478 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p1000-L128-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, evoxc-specialized-numpy | 3072 |
| `p1000-L128-i32-N1024` | batch_forward | `8569c9182600e194030145073a49321b` | evoxc-batch-iree | 3072 |
| `p1000-L128-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-dense-iree-raised | 3072 |
| `p1000-L128-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, evoxc-specialized-numpy | 3 |
| `p1000-L128-i32-N1024` | forward | `3d80e2b6ce635b32257cded7638bd50c` | evoxc-batch-iree | 3 |
| `p1000-L128-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-dense-iree-raised | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p1000-L128-i32-N1024: 4/4.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
