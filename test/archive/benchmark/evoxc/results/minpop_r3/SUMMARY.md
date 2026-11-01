# `EVOGP_FUSED_MIN_POP` column, re-taken on the current tree

Cells `p<pop>-L64-i32-N1024`, warm medians over 5 warmups + 20 sync-bracketed reps, one process per (pop, arm), idle-device guard per config.
`fused` = `Forest.batch_forward` / `SR_fitness` with `EVOGP_EVAL_MODE=fused` (the policy's answer, so below the floor it IS the specialized fallback);
`kernel` = `fused_engine.evaluate_fused` / `sr_fitness_fused` (policy-free, served at any pop); `spec` = `EVOGP_EVAL_MODE=specialized`; `torch-1` = per-tree torch walk; `torch-N` = the harness's batched arm.
Ratios are `reference / fused-kernel` (higher = the kernel wins).

## Value surface (`batch_forward`), ms

| pop | fused | kernel | spec | torch-1 | torch-N | spec/kernel | torch-1/kernel | torch-N/kernel | kernel served | fused served |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.517 | 0.035 | 0.466 | 0.233 | 1.359 | 13.22 | 6.61 | 38.58 | yes | no |
| 2 | 0.666 | 0.035 | 0.628 | 0.254 | 1.587 | 17.93 | 7.26 | 45.32 | yes | no |
| 4 | 0.918 | 0.035 | 0.928 | 0.433 | 1.897 | 26.37 | 12.30 | 53.90 | yes | no |
| 8 | 1.599 | 0.033 | 1.600 | 0.788 | 2.503 | 47.79 | 23.54 | 74.76 | yes | no |
| 16 | 0.040 | 0.034 | 2.814 | 1.410 | 3.195 | 82.09 | 41.11 | 93.18 | yes | yes |
| 32 | 0.039 | 0.034 | 5.547 | 2.736 | 3.782 | 161.57 | 79.70 | 110.16 | yes | yes |
| 64 | 0.040 | 0.035 | 13.413 | 6.088 | 4.126 | 388.31 | 176.24 | 119.44 | yes | yes |
| 128 | 0.038 | 0.036 | 23.225 | 11.243 | 4.103 | 650.23 | 314.76 | 114.87 | yes | yes |

## Loss surface (`SR_fitness`), ms

| pop | fused | kernel | spec | torch-1 | torch-N | spec/kernel | torch-1/kernel | torch-N/kernel |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.542 | 0.040 | 0.533 | 0.248 | 1.412 | 13.21 | 6.15 | 34.98 |
| 2 | 0.680 | 0.043 | 0.672 | 0.320 | 1.627 | 15.65 | 7.45 | 37.88 |
| 4 | 0.952 | 0.040 | 0.875 | 0.540 | 1.975 | 21.76 | 13.44 | 49.14 |
| 8 | 1.571 | 0.043 | 1.656 | 1.003 | 2.544 | 38.86 | 23.52 | 59.68 |
| 16 | 0.049 | 0.041 | 2.886 | 1.848 | 3.140 | 71.20 | 45.60 | 77.49 |
| 32 | 0.049 | 0.041 | 5.588 | 3.665 | 3.820 | 137.83 | 90.41 | 94.23 |
| 64 | 0.048 | 0.042 | 11.943 | 7.577 | 4.166 | 284.02 | 180.19 | 99.07 |
| 128 | 0.048 | 0.041 | 23.431 | 14.560 | 4.139 | 568.37 | 353.17 | 100.41 |

## Route policy per population

| pop | `auto` | `EVOGP_EVAL_MODE=fused` |
|---|---|---|
| 1 | specialized | specialized |
| 2 | specialized | specialized |
| 4 | specialized | specialized |
| 8 | specialized | specialized |
| 16 | fused | fused |
| 32 | fused | fused |
| 64 | fused | fused |
| 128 | fused | fused |

## What the column says

- The kernel is flat over pop 1-128: **0.033-0.036 ms** on the value surface and **0.040-0.043 ms** on the loss surface.
- Warm value crossover against `specialized`: pop **1** and up.
- Warm value crossover against `torch_per_tree`: pop **1** and up.
- Warm value crossover against `torch_batched`: pop **1** and up.
- At pop 1 the kernel's first call is **1619 ms** (a Triton JIT: `FUSED.md` records ~1.6 s for the value kernel and ~0.94-0.99 s for the loss kernel from an empty cache, and ~2 ms once the disk cache is warm), against a 0.431 ms/call warm saving over `specialized`: a one-shot call needs **≈3,760 calls** to repay it.
- Route policy: `auto` and an explicit `EVOGP_EVAL_MODE=fused` both answer `fused` from pop 16 up and `specialized` below it, so the `fused` column below the floor IS the fallback (0 launches).
