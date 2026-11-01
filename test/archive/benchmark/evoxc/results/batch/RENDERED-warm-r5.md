# evoxc benchmark summary: warm-r5

Source: `test/benchmark/evoxc/results/batch/warm-r5` — **24 records**, 4 cells x 6 arms.

The campaign ran **24 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **970.1 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=1` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=3`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **0 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-29T08:31:34Z |
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

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 30.308 (29.375) | 3.994 (3.990) | 1.101 (1.083) | 38.596 (36.479) | 1.101 (1.094) | 1.990 (1.896) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 30.601 (29.885) | 4.466 (4.421) | 2.058 (2.043) | 38.718 (38.532) | 2.152 (2.117) | 2.163 (2.114) | auto/numpy → dense | evoxc-auto |
| `p1000-L32-i8-N1024` | 120.237 (119.472) | 4.429 (4.407) | 1.883 (1.871) | 149.221 (142.756) | 1.883 (1.853) | 5.043 (5.005) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N1024` | 118.725 (117.568) | 4.873 (4.847) | 5.726 (5.684) | 148.023 (140.453) | 5.764 (5.724) | 6.162 (6.093) | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 27.52x | 3.63x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.79x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 27.52x | 3.63x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 15.23x | 2.01x |
| `p256-L64-i32-N1024` | evoxc-auto | 14.87x | 2.17x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.79x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 14.22x | 2.07x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 14.15x | 2.07x |
| `p1000-L32-i8-N1024` | evoxc-auto | 63.87x | 2.35x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.81x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 63.85x | 2.35x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 23.84x | 0.88x |
| `p1000-L64-i32-N1024` | evoxc-auto | 20.74x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.80x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 20.60x | 0.85x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 19.27x | 0.79x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 23.349 (23.091) | 3.961 (3.944) | 1.702 (1.632) | 39.515 (38.842) | 1.866 (1.682) | 2.282 (2.135) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.461 (22.781) | 4.358 (4.055) | 2.878 (2.682) | 38.301 (36.713) | 2.854 (2.846) | 2.156 (2.020) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L32-i8-N1024` | 95.104 (95.044) | 4.408 (4.392) | 3.814 (3.709) | 146.260 (142.512) | 3.812 (3.690) | 5.796 (5.721) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 90.746 (90.455) | 4.859 (4.815) | 7.615 (7.528) | 153.235 (148.382) | 7.534 (7.489) | 5.700 (5.506) | auto/numpy → dense | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 13.72x | 2.33x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 12.51x | 2.12x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 10.23x | 1.74x |
| `p256-L64-i32-N1024` | evoxc-auto | 8.15x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.61x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.22x | 1.53x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 10.88x | 2.02x |
| `p1000-L32-i8-N1024` | evoxc-auto | 24.94x | 1.16x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.65x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 24.95x | 1.16x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 16.41x | 0.76x |
| `p1000-L64-i32-N1024` | evoxc-auto | 11.92x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 12.05x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 15.92x | 0.85x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 23.859 (23.474) | 3.872 (3.827) | 0.662 (0.634) | 35.828 (35.169) | 0.686 (0.669) | 0.797 (0.764) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.896 (23.812) | 4.375 (3.974) | 0.911 (0.885) | 35.451 (34.323) | 0.982 (0.971) | 0.739 (0.695) | auto/numpy → dense | evoxc-batch-iree |
| `p1000-L32-i8-N1024` | 96.908 (96.458) | 3.919 (3.818) | 0.693 (0.653) | 139.685 (135.689) | 0.664 (0.615) | 1.605 (1.596) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 93.848 (93.159) | 4.173 (3.998) | 0.896 (0.868) | 135.671 (134.565) | 0.924 (0.866) | 1.343 (1.302) | auto/numpy → dense | evoxc-auto |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

Batch route arms: `evoxc-batch-iree` (band_size=32, bands=None, alignment=variant, switch_ops=False, min_pop=64) — one merged graph per population, so `first_call_ms` is a compile and the warm columns are the ones to compare.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 36.06x | 5.85x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.67x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 34.79x | 5.65x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 29.94x | 4.86x |
| `p256-L64-i32-N1024` | evoxc-auto | 26.23x | 4.80x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 24.33x | 4.45x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 32.36x | 5.92x |
| `p1000-L32-i8-N1024` | evoxc-auto | 139.88x | 5.66x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.69x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 145.88x | 5.90x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 60.39x | 2.44x |
| `p1000-L64-i32-N1024` | evoxc-auto | 104.79x | 4.66x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.69x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 101.56x | 4.52x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 69.88x | 3.11x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 17.545 | 22.847 | 5.264 | 294.365 | 5.364 | 3446.200 |
| `p256-L64-i32-N1024` | 17.114 | 22.404 | 6.635 | 265.736 | 6.655 | 3884.557 |
| `p1000-L32-i8-N1024` | 60.887 | 53.435 | 7.355 | 1010.037 | 7.242 | 17052.609 |
| `p1000-L64-i32-N1024` | 56.169 | 65.538 | 12.458 | 859.450 | 12.352 | 17524.310 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 15.052 / 2.493 (op 14%) | 20.419 / 2.428 (op 11%) | 2.848 / 2.415 (op 46%) | 291.853 / 2.512 (op 1%) | 2.889 / 2.475 (op 46%) | 3443.376 / 2.824 (op 0%) |
| `p256-L64-i32-N1024` | 14.653 / 2.461 (op 14%) | 19.865 / 2.539 (op 11%) | 4.167 / 2.468 (op 37%) | 263.296 / 2.440 (op 1%) | 4.226 / 2.429 (op 36%) | 3881.506 / 3.051 (op 0%) |
| `p1000-L32-i8-N1024` | 58.115 / 2.772 (op 5%) | 50.996 / 2.440 (op 5%) | 4.892 / 2.463 (op 33%) | 1007.307 / 2.729 (op 0%) | 4.851 / 2.391 (op 33%) | 17049.646 / 2.963 (op 0%) |
| `p1000-L64-i32-N1024` | 53.458 / 2.711 (op 5%) | 62.995 / 2.543 (op 4%) | 9.985 / 2.473 (op 20%) | 856.697 / 2.753 (op 0%) | 9.946 / 2.406 (op 19%) | 17521.340 / 2.969 (op 0%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 3.33x | 4.34x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3.27x | 4.26x |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 0.01x | 0.01x |
| `p256-L64-i32-N1024` | evoxc-auto | 2.58x | 3.38x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 2.57x | 3.37x |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 0.00x | 0.01x |
| `p1000-L32-i8-N1024` | evoxc-auto | 8.28x | 7.27x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.05x |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 8.41x | 7.38x |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 0.00x | 0.00x |
| `p1000-L64-i32-N1024` | evoxc-auto | 4.51x | 5.26x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.07x | 0.08x |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 4.55x | 5.31x |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 0.00x | 0.00x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | -0.084430 | -0.084430 | -0.084430 | -0.084430 | -0.084430 | -0.084430 |
| `p256-L64-i32-N1024` | -0.084812 | -0.084812 | -0.084812 | -0.084812 | -0.084812 | -0.084812 |
| `p1000-L32-i8-N1024` | -0.084133 | -0.084133 | -0.084133 | -0.084133 | -0.084133 | -0.084133 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | evoxc-auto | 3441.941 | 1 | 1.101 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 839.833 | 212 | 38.596 |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3379.297 | 1 | 1.101 |
| `p256-L32-i8-N1024` | evoxc-batch-iree | 10879.848 | 0 | 1.990 |
| `p256-L64-i32-N1024` | evoxc-auto | 6437.416 | 1 | 2.058 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 851.286 | 219 | 38.718 |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6516.240 | 1 | 2.152 |
| `p256-L64-i32-N1024` | evoxc-batch-iree | 12208.982 | 0 | 2.163 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3401.406 | 1 | 1.883 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3158.103 | 797 | 149.221 |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3311.732 | 1 | 1.883 |
| `p1000-L32-i8-N1024` | evoxc-batch-iree | 91117.632 | 0 | 5.043 |
| `p1000-L64-i32-N1024` | evoxc-auto | 6438.283 | 1 | 5.726 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3179.320 | 821 | 148.023 |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6454.638 | 1 | 5.764 |
| `p1000-L64-i32-N1024` | evoxc-batch-iree | 161967.826 | 0 | 6.162 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | batch_forward | `fd77ba5e8c3dc0d49c5225b2031b75f5` | evoxc-batch-iree | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L32-i8-N1024` | forward | `39f9d20e6ee43cbd19d90b7b1cff83c8` | evoxc-batch-iree | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | batch_forward | `fa7420d4b56bb9109bff0cd021b883a4` | evoxc-batch-iree | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | forward | `d8ad0c0490455455379c8ecba190e5a0` | evoxc-batch-iree | 0 |
| `p1000-L32-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `69f8b425f050934d448d300ecf26d656` | evoxc-batch-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-dense-iree | 7 |
| `p1000-L32-i8-N1024` | forward | `431f4fd79a790f3b93c4e9e85afeee3b` | evoxc-batch-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `8569c9182600e194030145073a49321b` | evoxc-batch-iree | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-dense-iree | 3 |
| `p1000-L64-i32-N1024` | forward | `3d80e2b6ce635b32257cded7638bd50c` | evoxc-batch-iree | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p256-L32-i8-N1024: 5/6, p256-L64-i32-N1024: 5/6, p1000-L32-i8-N1024: 5/6, p1000-L64-i32-N1024: 5/6.

## Caveats

- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole population multiset (expect seconds to minutes) and is excluded from the warm medians by protocol — compare the warm columns only.
- Every scheduled config in `campaign.json` finished `ok`.
