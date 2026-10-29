# evoxc benchmark summary: flat-r5

Source: `test/benchmark/evoxc/results/batch/flat-r5` — **4 records**, 2 cells x 2 arms.

The campaign ran **4 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **127.5 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=1` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T08:47:56Z |
| git SHA (`tree_root_sha`) | 6f99c0ca8901b034a20c5157550cf67e2c2fa881 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A31/src` |
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

| cell | evoxc-batch-iree | evoxc-batch-iree-flat | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 2.131 (1.879) | 2.123 (2.115) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree-flat |
| `p256-L64-i32-N1024` | 2.108 (2.084) | 2.413 (2.354) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64), `evoxc-batch-iree-flat` (bands=flat) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree-flat | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree-flat | - | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-batch-iree | evoxc-batch-iree-flat | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 2.160 (2.092) | 2.310 (2.287) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree |
| `p256-L64-i32-N1024` | 2.221 (2.120) | 2.465 (2.289) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64), `evoxc-batch-iree-flat` (bands=flat) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree-flat | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree-flat | - | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-batch-iree | evoxc-batch-iree-flat | auto route | fastest |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.831 (0.789) | 1.102 (1.088) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree |
| `p256-L64-i32-N1024` | 0.734 (0.722) | 0.981 (0.929) | batch/iree → batch (bands=None, evoxc-batch-iree) | evoxc-batch-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64), `evoxc-batch-iree-flat` (bands=flat) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree-flat | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree-flat | - | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | evoxc-batch-iree | evoxc-batch-iree-flat |
|---|---|---|
| `p256-L32-i8-N1024` | - | - |
| `p256-L64-i32-N1024` | - | - |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | evoxc-batch-iree | evoxc-batch-iree-flat |
|---|---|---|
| `p256-L32-i8-N1024` | - | - |
| `p256-L64-i32-N1024` | - | - |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-batch-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-batch-iree-flat | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-batch-iree-flat | - | - |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-batch-iree | 10638.992 | 0 | 2.131 |
| `p256-L32-i8-N1024` | evoxc-batch-iree-flat | 12104.704 | 0 | 2.123 |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 12168.143 | 0 | 2.108 |
| `p256-L64-i32-N1024` | evoxc-batch-iree-flat | 14904.477 | 0 | 2.413 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `fd77ba5e8c3dc0d49c5225b2031b75f5` | evoxc-batch-iree | 1024 |
| `p256-L32-i8-N1024` | batch_forward | `3085f730a68110be37676404f370c76c` | evoxc-batch-iree-flat | 1024 |
| `p256-L32-i8-N1024` | forward | `39f9d20e6ee43cbd19d90b7b1cff83c8` | evoxc-batch-iree | 1 |
| `p256-L32-i8-N1024` | forward | `531912511181090ff76a730264ec8c53` | evoxc-batch-iree-flat | 1 |
| `p256-L64-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N1024` | batch_forward | `0bd7b40d4e6e4e7a345cb6729f4846c4` | evoxc-batch-iree-flat | 0 |
| `p256-L64-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N1024` | forward | `d67851d562c6bb7c737fe48373520e5b` | evoxc-batch-iree-flat | 0 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 2/2, p256-L64-i32-N1024: 1/2.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
