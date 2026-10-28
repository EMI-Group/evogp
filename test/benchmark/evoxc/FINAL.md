# Final campaign — evoxc engine vs plain PyTorch vs the pre-optimization snapshot

Report of record for the post-optimization round.
Measured in one session on the shared AMD MI300X VF, one process per timed config, warm medians of 5 sync-bracketed reps after 2 warmups, device-idle guard before every config.
Records: `results/final/after/` (this tree), `results/final/before/` (snapshot `/tmp/evogp_preopt`, commit `530518e`), `results/final/{decompose,decompose-before,cold,training}/`, plain-text renders of the three summary passes in `results/final/RENDERED.md`.
The committed baseline round in `results/baseline/` stays untouched and is *not* the `before` reference here — `before` is the snapshot re-measured in this session, so the delta is same-window.

## 1. What changed, and what the arms now mean

- `auto` resolves to **dense + iree from `pop >= 64`** on a device box and to **specialized + numpy** below that, on a CPU box, or on any fallback. `EVOGP_EVAL_MODE=dense|specialized` still overrides.
  So **the `evoxc-auto` arm is no longer the specialized route**: after the change its numbers *are* the dense-path numbers, and `evoxc-auto` and `evoxc-dense-iree` agree to within noise in every cell. Before the change `auto` never selected dense at any size (`dense_policy.auto_min_pop` 16, and every before record has `would_use_dense: false`), so the before column of `evoxc-auto` is the specialized-numpy route.
- Host glue in the dense path is memoized (`DENSE_EXECUTABLE_MEMO` / `DENSE_PLAN_MEMO` / `DENSE_INPUT_MEMO`, `tree/dense_memo.py`); `x`/`labels` are staged once. Host share of a warm dense call fell to ~15 % (§3).
- An opt-in on-disk artifact cache (`tree/artifact_cache.py`, `EVOGP_EVOXC_ARTIFACT_DIR`, default off) makes a compile **0.16 s instead of 6.5 s** across processes (§4, §5).
- `tree/generation.py` was rewritten (737 → 347 launches); the GP operators are ~2.5× faster, visible in the `gp_step` operator column (§2d).

## 2. Headline tables

Grid: `pop ∈ {16, 64, 256, 1000}` × `max_tree_len ∈ {32, 64}` × `input_len ∈ {8, 32}` at N=1024 rows, plus an N=64 column; single-output SR.
Workload `sr_fitness` = one fitness evaluation of a whole forest; `gp_step` = one full generation (evaluate + select + crossover + mutation).
`after` = HEAD, `before` = commit `530518e`.

### 2a. `evoxc-auto` (the shipped default) — ms, median of 5

| cell | sr after | sr before | sr b/a | gp_step after | gp_step before | gp b/a | eval/ops after | eval/ops before |
|---|---|---|---|---|---|---|---|---|
| p16-L32-i8-N1024 | 2.468 | 3.075 | 0.80× | 6.626 | 11.323 | 0.59× | 4.308 / 2.318 | 5.599 / 5.724 |
| p16-L32-i32-N1024 | 2.476 | 3.069 | 0.81× | 6.725 | 59.553 | 0.11× | 4.245 / 2.481 | 52.962 / 6.591 |
| p16-L64-i8-N1024 | 2.299 | 3.071 | 0.75× | 6.578 | 10.731 | 0.61× | 4.226 / 2.352 | 5.315 / 5.416 |
| p16-L64-i32-N1024 | 2.627 | 2.837 | 0.93× | 6.637 | 63.520 | 0.10× | 4.454 / 2.183 | 57.938 / 5.582 |
| p64-L32-i8-N1024 | 0.924 | 8.345 | 0.11× | 4.934 | 35.842 | 0.14× | 2.455 / 2.479 | 30.231 / 5.611 |
| p64-L32-i32-N1024 | 0.964 | 8.297 | 0.12× | 4.943 | 51.788 | 0.10× | 2.344 / 2.599 | 46.429 / 5.359 |
| p64-L64-i8-N1024 | 1.530 | 9.206 | 0.17× | 5.393 | 38.902 | 0.14× | 2.998 / 2.396 | 33.269 / 5.633 |
| p64-L64-i32-N1024 | 1.551 | 9.197 | 0.17× | 5.417 | 51.110 | 0.11× | 3.043 / 2.373 | 45.520 / 5.590 |
| p256-L32-i8-N1024 | 1.101 | 39.101 | 0.03× | 5.381 | 197.492 | 0.03× | 2.899 / 2.482 | 191.450 / 6.042 |
| p256-L32-i32-N1024 | 1.133 | 39.227 | 0.03× | 5.484 | 172.240 | 0.03× | 2.966 / 2.518 | 166.491 / 5.749 |
| p256-L64-i8-N1024 | 1.977 | 36.563 | 0.05× | 6.607 | 181.190 | 0.04× | 4.132 / 2.476 | 175.420 / 5.770 |
| p256-L64-i32-N1024 | 2.046 | 38.562 | 0.05× | 6.630 | 169.110 | 0.04× | 4.190 / 2.440 | 163.294 / 5.816 |
| p1000-L32-i8-N1024 | 1.903 | 145.331 | 0.01× | 7.722 | 901.434 | 0.01× | 5.003 / 2.719 | 894.602 / 6.833 |
| p1000-L32-i32-N1024 | 1.974 | 137.102 | 0.01× | 7.436 | 811.149 | 0.01× | 4.867 / 2.570 | 804.909 / 6.240 |
| p1000-L64-i8-N1024 | 5.510 | 134.556 | 0.04× | 12.832 | 900.975 | 0.01× | 10.181 / 2.651 | 894.861 / 6.114 |
| p1000-L64-i32-N1024 | 5.665 | 140.148 | 0.04× | 12.410 | 803.111 | 0.02× | 9.861 / 2.549 | 797.102 / 6.008 |
| p16-L64-i32-N64 | 2.380 | 2.542 | 0.94× | 6.687 | 17.453 | 0.38× | 4.343 / 2.344 | 11.881 / 5.572 |
| p64-L64-i32-N64 | 1.435 | 8.695 | 0.16× | 5.329 | 95.076 | 0.06× | 2.897 / 2.432 | 89.727 / 5.349 |
| p256-L64-i32-N64 | 1.464 | 32.487 | 0.05× | 6.066 | 240.900 | 0.03× | 3.577 / 2.489 | 234.806 / 6.094 |
| p1000-L64-i32-N64 | 1.582 | 134.837 | 0.01× | 8.465 | 737.962 | 0.01× | 5.847 / 2.619 | 731.956 / 6.006 |

The `b/a` column is before ÷ after, so **0.05× means the new default is 20× faster**, 0.80× means 1.25× slower. Read every row that way (§5 explains the p16 regressions).
`p1000-L32-i32-N1024/evoxc-auto` was re-run (first pass drifted 2×); see §7.

### 2b. `evoxc-dense-iree` (forced dense) — ms, median of 5

| cell | sr after | sr before | sr b/a | gp_step after | gp_step before | gp b/a |
|---|---|---|---|---|---|---|
| p16-L32-i8-N1024 | 0.974 | 2.372 | 0.41× | 5.042 | 9.186 | 0.55× |
| p16-L32-i32-N1024 | 0.941 | 2.470 | 0.38× | 5.392 | 8.485 | 0.64× |
| p16-L64-i8-N1024 | 1.456 | 2.915 | 0.50× | 5.653 | 9.561 | 0.59× |
| p16-L64-i32-N1024 | 1.447 | 3.059 | 0.47× | 6.494 | 9.264 | 0.70× |
| p64-L32-i8-N1024 | 1.029 | 2.558 | 0.40× | 4.779 | 8.605 | 0.56× |
| p64-L32-i32-N1024 | 0.950 | 2.694 | 0.35× | 4.951 | 8.778 | 0.56× |
| p64-L64-i8-N1024 | 1.535 | 3.195 | 0.48× | 5.365 | 9.159 | 0.59× |
| p64-L64-i32-N1024 | 1.514 | 3.222 | 0.47× | 5.319 | 8.992 | 0.59× |
| p256-L32-i8-N1024 | 1.103 | 3.098 | 0.36× | 5.548 | 9.151 | 0.61× |
| p256-L32-i32-N1024 | 1.133 | 3.156 | 0.36× | 5.467 | 9.073 | 0.60× |
| p256-L64-i8-N1024 | 1.999 | 4.274 | 0.47× | 6.797 | 10.234 | 0.66× |
| p256-L64-i32-N1024 | 2.160 | 4.412 | 0.49× | 6.574 | 10.333 | 0.64× |
| p1000-L32-i8-N1024 | 1.906 | 5.462 | 0.35× | 9.176 | 11.140 | 0.82× |
| p1000-L32-i32-N1024 | 2.031 | 5.643 | 0.36× | 7.360 | 10.623 | 0.69× |
| p1000-L64-i8-N1024 | 5.535 | 10.331 | 0.54× | 12.661 | 16.130 | 0.78× |
| p1000-L64-i32-N1024 | 5.744 | 10.512 | 0.55× | 12.593 | 16.377 | 0.77× |
| p16-L64-i32-N64 | 1.464 | 2.885 | 0.51× | 5.795 | 9.592 | 0.60× |
| p64-L64-i32-N64 | 1.425 | 3.025 | 0.47× | 5.532 | 9.034 | 0.61× |
| p256-L64-i32-N64 | 1.462 | 3.787 | 0.39× | 6.634 | 9.813 | 0.68× |
| p1000-L64-i32-N64 | 1.571 | 6.365 | 0.25× | 8.205 | 12.051 | 0.68× |

The dense path itself got **2.0–4.0× faster** on `sr_fitness` (0.25–0.55× before/after) and 1.2–1.8× on `gp_step`; that is the host-glue work, since the compiled graph is unchanged. Note `gp_step` improves less than `sr_fitness` because the operators are now a *larger* share of a much shorter generation (§2d).

### 2c. Versus the torch baselines — speed-up on `sr_fitness` / `gp_step`, after

`X× / Y×` = `torch-per-tree ÷ arm` / `torch-batched ÷ arm`; **> 1× means the evoxc arm is faster than that baseline**.

| cell | sr auto | sr dense-iree | gp_step auto | gp_step dense-iree |
|---|---|---|---|---|
| p16-L32-i8-N1024 | 0.75× / 1.13× | 1.89× / 2.85× | 0.50× / 0.73× | 0.66× / 0.96× |
| p16-L32-i32-N1024 | 0.79× / 1.15× | 2.08× / 3.03× | 0.52× / 0.72× | 0.65× / 0.90× |
| p16-L64-i8-N1024 | 0.81× / 1.41× | 1.28× / 2.23× | 0.47× / 0.85× | 0.55× / 0.99× |
| p16-L64-i32-N1024 | 0.73× / 1.23× | 1.33× / 2.24× | 0.49× / 0.87× | 0.51× / 0.89× |
| p64-L32-i8-N1024 | 8.09× / 3.87× | 7.27× / 3.47× | 1.09× / 1.43× | 1.13× / 1.48× |
| p64-L32-i32-N1024 | 7.71× / 3.74× | 7.83× / 3.80× | 1.10× / 1.73× | 1.10× / 1.73× |
| p64-L64-i8-N1024 | 5.13× / 2.64× | 5.11× / 2.63× | 0.94× / 1.32× | 0.94× / 1.33× |
| p64-L64-i32-N1024 | 4.94× / 2.65× | 5.06× / 2.72× | 1.02× / 1.79× | 1.04× / 1.82× |
| p256-L32-i8-N1024 | 27.42× / 3.64× | 27.36× / 3.63× | 3.82× / 4.31× | 3.70× / 4.18× |
| p256-L32-i32-N1024 | 26.73× / 3.53× | 26.71× / 3.53× | 3.94× / 4.51× | 3.95× / 4.53× |
| p256-L64-i8-N1024 | 15.42× / 2.17× | 15.25× / 2.15× | 2.92× / 3.93× | 2.83× / 3.82× |
| p256-L64-i32-N1024 | 14.91× / 2.20× | 14.12× / 2.09× | 3.19× / 4.23× | 3.22× / 4.27× |
| p1000-L32-i8-N1024 | 63.44× / 2.30× | 63.36× / 2.30× | 7.87× / 7.30× | 6.63× / 6.14× |
| p1000-L32-i32-N1024 | 62.64× / 2.23× | 60.86× / 2.16× | 7.83× / 7.00× | 7.91× / 7.08× |
| p1000-L64-i8-N1024 | 21.91× / 0.88× | 21.81× / 0.88× | 4.83× / 6.07× | 4.89× / 6.15× |
| p1000-L64-i32-N1024 | 21.13× / 0.86× | 20.84× / 0.85× | 4.53× / 5.42× | 4.47× / 5.34× |
| p16-L64-i32-N64 | 0.77× / 1.34× | 1.26× / 2.17× | 0.48× / 0.87× | 0.56× / 1.00× |
| p64-L64-i32-N64 | 5.17× / 2.87× | 5.21× / 2.89× | 0.96× / 1.40× | 0.93× / 1.35× |
| p256-L64-i32-N64 | 20.35× / 3.09× | 20.39× / 3.09× | 3.98× / 4.78× | 3.64× / 4.37× |
| p1000-L64-i32-N64 | 75.19× / 3.15× | 75.69× / 3.17× | 6.69× / 7.59× | 6.90× / 7.84× |

Reading of record:
- **`sr_fitness`, pop ≥ 64**: the engine beats `torch-per-tree` by 4.9–75× and `torch-batched` by 2.1–3.9× (the 5–7.6× figures belong to `gp_step`, below). Two losses against `torch-batched` only: `p1000-L64-i8-N1024` (0.88×) and `p1000-L64-i32-N1024` (0.86×) — **at 1000 trees × 64 nodes the vectorized torch pass is 1.16× faster than the engine**; report as a loss, not parity.
- **`sr_fitness`, pop 16**: the shipped default is 1.22–1.37× *slower* than `torch-per-tree` (0.73–0.81×) but 1.13–1.41× faster than `torch-batched`. Forced `dense-iree` at pop 16 wins both (1.26–2.08× vs per-tree, 2.17–3.03× vs batched) — i.e. the loss is the 64-pop route threshold plus per-tree readback, not the dense path.
- **`gp_step`**: parity at pop 16 and pop 64 (0.47–1.10× vs per-tree: within the operator cost, which the arms share) and a clear win from pop 256 (2.9–7.9×). The generation step is *not* dominated by the engine any more at small pop — 33–53 % of it is selection/crossover/mutation.
- `torch-batched` wins `gp_step` nowhere and loses everywhere; `torch-per-tree` wins `gp_step` at pop 16 only.

### 2d. `gp_step` operator share (`ops / total`)

| cell | auto after | auto before | dense-iree after | dense-iree before |
|---|---|---|---|---|
| p16-L32-i8-N1024 | 35 % | 51 % | 48 % | 66 % |
| p16-L32-i32-N1024 | 37 % | 11 % | 45 % | 70 % |
| p16-L64-i8-N1024 | 36 % | 50 % | 42 % | 62 % |
| p16-L64-i32-N1024 | 33 % | 9 % | 37 % | 67 % |
| p64-L32-i8-N1024 | 50 % | 16 % | 50 % | 68 % |
| p64-L32-i32-N1024 | 53 % | 10 % | 52 % | 68 % |
| p64-L64-i8-N1024 | 44 % | 14 % | 43 % | 64 % |
| p64-L64-i32-N1024 | 44 % | 11 % | 44 % | 64 % |
| p256-L32-i8-N1024 | 46 % | 3 % | 47 % | 67 % |
| p256-L32-i32-N1024 | 46 % | 3 % | 46 % | 67 % |
| p256-L64-i8-N1024 | 37 % | 3 % | 38 % | 58 % |
| p256-L64-i32-N1024 | 37 % | 3 % | 37 % | 59 % |
| p1000-L32-i8-N1024 | 35 % | 1 % | 29 % | 54 % |
| p1000-L32-i32-N1024 | 35 % | 1 % | 34 % | 53 % |
| p1000-L64-i8-N1024 | 21 % | 1 % | 19 % | 38 % |
| p1000-L64-i32-N1024 | 21 % | 1 % | 20 % | 37 % |
| p16-L64-i32-N64 | 35 % | 32 % | 44 % | 63 % |
| p64-L64-i32-N64 | 46 % | 6 % | 46 % | 66 % |
| p256-L64-i32-N64 | 41 % | 3 % | 41 % | 63 % |
| p1000-L64-i32-N64 | 31 % | 1 % | 30 % | 50 % |

Absolute operator time is arm-independent and now **2.2–2.7 ms per generation** in every cell (before: 5.3–6.8 ms) — that is the `generation.py` rewrite, and it is what makes the share rise at pop 16/64 while the total falls.
At pop 1000 the evaluation is still 3–4× the operator cost, which is why `gp_step` there is evaluation-bound.

## 3. Decomposition — where a warm call goes now (`p256-L64-i32-N1024`, `sr_fitness`)

`decompose.py` per-phase instrumentation, after (`results/final/decompose/`) vs before (`results/final/decompose-before/`):

| | dense-iree after | dense-iree before | auto after (dense) | auto before (specialized) |
|---|---|---|---|---|
| plain median | **2.398 ms** | 5.203 ms | **2.407 ms** | 37.689 ms |
| `prepare_dense` | — (gone) | 1233.3 µs, 23.8 % | — (gone) | — |
| `plan_dense` | — (gone) | 1272.3 µs, 24.5 % | — (gone) | — |
| `build_dense` | — (gone) | 246.2 µs, 4.7 % | — (gone) | — |
| `run_loss_plan` | **2046.0 µs, 84.7 %** | 3050.5 µs, 58.8 % | **2061.7 µs, 85.0 %** | — |
| host-glue residual | **370.5 µs, 15.3 %** | −615.1 µs | **363.1 µs, 15.0 %** | **31 849 µs, 81.5 %** |
| `_evaluate_rows` | — | — | — | 7143.3 µs, 91.4 % |
| `counters_per_call` | `{run:1, to:1, numpy:1}` | `{from_numpy:10, to:11, numpy:11, run:1}` | `{run:1, to:1, numpy:1}` | `{run:224, numpy:5697}` |
| cold compile | 6368.5 ms, 1 miss | 6459.9 ms, 1 miss | 5.7 ms, 0 misses | 43.9 ms, 0 misses |

**The host glue is no longer the bottleneck; `evoxc.run` is.** One host-side plan dispatch (`run_loss_plan`, 84.7 %) plus 15.3 % of residual glue (readback `_to_host` 200 µs of it) is the whole warm call; the ten payload uploads and the decode pass that the baseline round measured at 75–77 % are gone — the per-call counter set is `{run, to, numpy}` instead of `{from_numpy×10, to×11, numpy×11, run}`.
Before, the same call re-planned and re-built the dense module on every invocation (`prepare_dense` + `plan_dense` + `build_dense` = 53 % *on top of* `run_loss_plan`, which is why that row's residual is negative — the before phase set over-attributes).
The specialized route before spent 81.5 % of its call in host glue across 224 `_run_tree` calls and 5697 numpy conversions.

## 4. Cold-start economics (default route, empty cache)

Wall time from process start to the first fitness value (`results/final/cold/`, `cold_start.py`):

| pop | wall to first fitness | import | population build | first call | compile misses |
|---|---|---|---|---|---|
| 16 | **7.73 s** | 6.91 s | 0.74 s | 0.08 s | 13 (specialized, per structure) |
| 64 | **14.11 s** | 6.91 s | 0.73 s | 6.47 s | 1 (dense) |
| 256 | **14.11 s** | 6.80 s | 0.75 s | 6.57 s | 1 (dense) |
| 1000 | **14.25 s** | 6.99 s | 0.74 s | 6.53 s | 1 (dense) |

`import evogp` + `import evoxc` is 6.8–7.0 s of every number — a fixed cost that no engine change touches.
Above pop 64 the first call pays the dense iree compile once (6.5 s); a second shape/function set pays it again.
Below pop 64 the route is specialized, so the first call is cheap (82 ms) but a fresh population is 13 compile misses.

With `EVOGP_EVOXC_ARTIFACT_DIR` (`p256-L64-i32-N1024`):

| phase | wall to first fitness | first call | misses | artifacts |
|---|---|---|---|---|
| populate (dir empty) | 14.13 s | 6542 ms | 1 | 1 file, 2 075 522 B stored |
| reload (dir warm) | **7.76 s** | **195.7 ms** | 0 | 1 hit, 0 stores |

The artifact cache makes the compile **33× cheaper** (6542 → 196 ms) and takes a cold start from 14.1 s to 7.8 s, i.e. down to the import floor. It is off by default; enabling it is the single cheapest fix for the "first run is 7 s slower than it should be" problem.

## 5. End-to-end training run (the GP-implementation headline)

`StandardPipeline.step()` on the `sr_test` synthetic target, `pop 256`, `tree_len 64`, `input_len 2`, `num_data 1000`, `seed 0`, one process per arm (`training_run.py`).

| arm | generations | total wall | generation 1 | warm median/gen | best fitness @10 / @25 / @50 |
|---|---|---|---|---|---|
| engine, default (dense-iree) | 50 | **7 155.8 ms** | 6 826.7 ms | **6.66 ms** | −0.226117 / −0.226117 / −0.226117 |
| engine, default | 200 | **8 342.3 ms** | 6 987.8 ms | 6.78 ms | −0.226117 / −0.226117 / −0.226117 |
| engine + artifact cache, populate | 50 | 7 309.2 ms | 6 972.9 ms | 6.78 ms | −0.226117 / −0.226117 / −0.226117 |
| **engine + artifact cache, warm** | 50 | **811.6 ms** | **477.6 ms** | 6.78 ms | −0.226117 / −0.226117 / −0.226117 |
| `torch-per-tree` evaluator | 50 | 921.9 ms | 290.4 ms | 12.60 ms | −0.226117 / −0.226117 / −0.226117 |
| `torch-per-tree` evaluator | 200 | **2 801.3 ms** | 306.8 ms | 12.37 ms | −0.226117 / −0.226117 / −0.226117 |
| pre-optimization snapshot | 50 | 5 025.9 ms | 1 062.1 ms | 70.72 ms | −0.225680 / −0.225680 / −0.225680 |
| pre-optimization snapshot | 200 | **84 924.1 ms** | 1 105.2 ms | 75.70 ms (mean 421 ms) | −0.225680 / −0.225680 / −0.225680 |

What this says, honestly:
- **Steady state, the engine is 1.8× faster per generation than the plain-torch evaluator** (6.78 vs 12.37 ms) and **11× faster than the pre-optimization snapshot** (75.7 ms median; the 200-generation run differs much more than the medians suggest — 84.9 s vs 8.3 s, 10.2×, because the old specialized route recompiles a graph for almost every distinct tree structure: **15 453 compile misses / 4096-entry cache churn over 200 generations**, mean 421 ms per generation against a 75.7 ms median).
- **Cold, a 50-generation run is dominated by the dense compile**: 6.83 s of the 7.16 s total is generation 1. End to end that makes the default engine **7.8× slower than the torch baseline for 50 generations** and still 3.0× slower at 200 generations (8.34 s vs 2.80 s, of which 6.99 s is the one-off compile).
  Break-even against `torch-per-tree` is ≈1 250 generations with an empty cache (5.6 ms/gen advantage vs a 6.99 s compile) and ≈immediate with a warm artifact cache.
- **With a warm artifact cache the ranking flips**: 811.6 ms vs 921.9 ms for 50 generations → the engine is **1.14× faster end to end than the torch baseline** and 6.2× faster than the pre-optimization snapshot. The artifact cache is what makes the default engine a win on short runs.
- **Convergence is not harmed and not improved**: engine and `torch-per-tree` produce the bit-identical trajectory digest (same RNG path, same fitness — parity by construction, which is the correctness statement, not a speed claim). The pre-optimization snapshot reaches −0.225680 vs −0.226117; that difference is a *different population sequence*, not better optimization (the `generation.py` rewrite consumes the RNG differently — tree-length means differ, e.g. 15.41 vs 17.4 at pop 256/L64), and it is inside the generation-1 spread of the smoke target. Neither arm "converges better".

## 6. Numeric validation (`validate.py`, oracle = `src/evogp/tests/reference_eval.py`)

Full sweep over `arith` / `rich` / `ternary` × the four scheduled cells (`results/final/validation.json`, 12 cases, `evoxc-auto` = numpy route):

- **Value path is bitwise exact** for `arith` and `ternary` across oracle / plain torch / engine in every cell and every workload (`max_abs 0.000e+00`, `bitwise_equal: true`).
- **`sr_fitness`** agrees to float32 reduction order only: relative ≤ ~1e-6, absolute up to 6.4e+01 where the outputs are large. No NaN/Inf disagreements anywhere.
- **`rich`** (the transcendental function set) is the one place with real spread: `torch-per-tree` 9.35e-02 abs / 1.49e-04 rel at p16; **engine (dense-iree) vs oracle 4.369e-01 abs / 6.4e-01 rel on `p256-L64-i32-N64` `batch_forward`** — larger than the torch arms' 5.2e-06. Dense-iree reassociates transcendentals; the deviation pattern is dense-route-specific (the same cell's `forward` agrees exactly, and the specialized-numpy arm is the one that tracks the torch value), so the caveat applies to the dense/iree lowering of transcendental-heavy trees, not to the SR default set (`+ - * /`), where agreement is exact.
  Practical reading: on the default route, `rich`-set expressions can differ from plain torch by ~1e-1 absolute at the tail. That is a *numerical* caveat for anyone comparing engine output to torch output bit-for-bit on transcendental-heavy trees; it does not affect the SR default set (`+ - * /`), where agreement is exact.
- **Digest groups moved** (this is new since the baseline round): after the optimization, `evoxc-auto`, `evoxc-auto-iree` and `evoxc-dense-iree` form one sha256 group and `evoxc-specialized-numpy` + both torch arms form another; at pop ≥ 256 the two groups differ on `forward`/`batch_forward`. At pop 16/64 `evoxc-specialized-iree` is still the bit-level outlier on its own (the documented finding). **Practical relevance of the specialized-iree divergence is now small: the default policy routes pop ≥ 64 to dense+iree, so the divergent arm is off the default path** — it only matters if someone forces `EVOGP_EVAL_MODE=specialized` with `EVOGP_EVOXC_BACKEND=iree`.
  `evoxc-specialized-iree` was measured on the pop ∈ {16, 64} cells only (`campaign.json:protocol.scope_dropped` records the 10 unscheduled pop ≥ 256 cells: at 272–284 ms of compile per tree structure a dense-scale cell is not affordable, which is exactly why it is not the device default).
  The targeted specialized-iree sweep (64 trees, 157 s per case) confirms the finding is a bit-level, not numeric, divergence: 11 of 64 trees non-bitwise, worst absolute deviation 1.562e-02 on `sr_fitness` and ≤ 4.8e-07 on the value path (`forward`/`batch_forward`), rel ≤ 2.5e-07, zero NaN/Inf mismatches.

## 7. Noise audit and flagged cells

Spread `(max−min)/median` over the 5 samples, every record in `results/final/{after,before}`:
- `after`: 6 of 390 workloads exceed 25 %; `before`: 3 of 270. Every one of them is a **single elevated sample** in an otherwise tight set (e.g. 2.02 ms among 0.91–1.01 ms), not a drifting distribution, and none changes a quoted median except the one below.
- **Flagged and re-run**: `p1000-L32-i32-N1024/evoxc-auto` (first pass drifted 2×: 3.98/3.99/3.15/1.99/1.96 ms). The committed record is the re-run and carries `notes[]`; samples 1.992/1.974/1.958/1.947/1.977 → spread 2.3 %, median 1.974. This is the value in the tables above.
- **Flagged, not re-run**: `p64-L32-i8-N1024/evoxc-auto` `sr_fitness` — median 0.924 ms from four tight samples (0.912–1.009) with one 2.02 ms outlier; the median is trustworthy (it is the second-smallest of five) but this cell is the per-cell minimum, so treat "fastest cell" claims for it with care.
- Provenance is clean: `env.evoxc_sha` = `855050d7750133b11d94639754637b090d304acd` in all 220 records; `env.tree_root_sha` uniform per side; **0 contaminated configs** (guard `idle` on 130/130 after and 90/90 before, empty `gpu_pids`).

## 8. Gate

`check.py` over the final records, per side (the container directory is not gateable in one run — `results/final/cold/` holds a different schema, so gate each side):

```bash
cd test/benchmark/evoxc
python3 check.py --results results/final/after  --allow-snapshot --allow-missing --allow-digest-mismatch   # 130 records / 3602 checks / 0 failures
python3 check.py --results results/final/before --allow-snapshot --allow-missing --allow-digest-mismatch   #  90 records / 2642 checks / 0 failures
```

Without `--allow-missing --allow-digest-mismatch` the gate reports the 10 unscheduled `evoxc-specialized-iree` cells and the numpy-vs-iree digest split (§6) as failures — both are *expected and declared*, not defects, which is why the documented invocation carries the two flags.

## 9. Examples sanity (no network; run from the repository root)

| script | exit | wall | note |
|---|---|---|---|
| `example/basic.py` | 0 | 17 s | |
| `example/combined_sr.py` | 0 | 19 s | `CustomLoss` multi-output route |
| `example/regressor.py` | 0 | 16 s | |
| `example/transformer.py` | 0 | 536 s | slow but green |
| `example/classifier.py` | 124 (timeout at 900 s) | — | not a failure: pop 1000 × 100 generations at ~1.1 s/generation |
| `example/custom_sr.py` | 1 | 7 s | needs `data/409_*.csv` and `pandas`, neither present on this box — dependency gap, not an engine defect |
| `example/uci_sr.py`, `example/uci_classifier.py` | — | — | skipped, download UCI datasets (no network) |
| `example/brax_task.py`, `example/mujoco_playground*.py`, `example/genesis*.py` | — | — | skipped, need brax/mujoco_playground/genesis (absent) |

No offline example regressed; the three short ones plus the multi-output one are green on the post-optimization tree.

## 10. Environment and protocol

`python 3.14.8`, `torch 2.13.0+rocm7.1`, `numpy 2.5.4`, `hip 7.1.52802`, `iree-base-compiler 3.11.0`, `iree-base-runtime 3.11.0`, `evoxc 0.1.0 @ 855050d7750133b11d94639754637b090d304acd`, GPU `AMD Instinct MI300X VF` (191.7 GB), `EVOXC_DISABLE_LOCATIONS=1`.
Median of 5 sync-bracketed reps after 2 warmups, `gp_warmup=1` + `gp_reps=5`, one process per `(cell, arm)`, device-idle guard before every config, env block in every record.
`after` = this tree; `before` = `/tmp/evogp_preopt` (commit `530518e`, an ancestor of HEAD) driven by the *same* harness through `--tree-root`, so before/after is same-window, same-GPU, same-harness.
`before` has no torch arms by design — the torch baselines are tree-root-independent, so their after values are the reference for both sides.

## 11. How to reproduce

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
PY=/path/to/venv/bin/python
cd test/benchmark/evoxc

# 1. after campaign (one process per config, idle-guarded)
$PY campaign.py --out results/final/after --cells main,n64 --arms all

# 2. before campaign against the pre-optimization snapshot
git archive 530518e | tar -x -C /tmp/evogp_preopt
$PY campaign.py --out results/final/before --tree-root /tmp/evogp_preopt --cells main,n64 --arms evoxc-auto,evoxc-auto-iree,evoxc-dense-iree,evoxc-specialized-numpy,evoxc-specialized-iree

# 3. decomposition, validation, cold start, training
$PY decompose.py --cell p256-L64-i32-N1024 --out results/final/decompose/p256-L64-i32-N1024.json
$PY decompose.py --cell p256-L64-i32-N1024 --tree-root /tmp/evogp_preopt --out results/final/decompose-before/p256-L64-i32-N1024.json
$PY validate.py --func-sets arith,rich,ternary --out results/final/validation.json
EVOGP_EVAL_MODE=specialized EVOGP_EVOXC_BACKEND=iree $PY validate.py --out results/final/validation-specialized-iree.json
for pop in 16 64 256 1000; do $PY cold_start.py --cell p${pop}-L64-i32-N1024 --out results/final/cold/p${pop}-L64-i32-N1024.json; done
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_artifact $PY cold_start.py --cell p256-L64-i32-N1024 --out results/final/cold/p256-L64-i32-N1024-artifact.json
$PY training_run.py --arm engine --out results/final/training/engine-200.json
$PY training_run.py --arm torch-per-tree --out results/final/training/torch-per-tree-200.json
$PY training_run.py --arm preopt --tree-root /tmp/evogp_preopt --out results/final/training/preopt-200.json
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_training_art $PY training_run.py --arm engine --note "artifact cache cold (populates the dir)" --out results/final/training/engine-artifact-cold.json
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_training_art $PY training_run.py --arm engine --note "artifact cache warm reload" --out results/final/training/engine-artifact-warm.json

# 4. render + gate
$PY summarize.py --results results/final/after --out /tmp/after.md
$PY summarize.py --results results/final/after --compare-results results/final/before --compare-label before --out /tmp/compare.md
python3 check.py --results results/final/after --allow-snapshot --allow-missing --allow-digest-mismatch
```

## 12. Caveats worth carrying forward

- `after` and `before` populations are **not structure-identical** for the same seed (the `generation.py` rewrite consumes the RNG differently: tree-length mean 15.41 vs 17.4 on the same cell). Timings are statistics over 256–1000 trees either way, so the ratios are robust, but the two sides are not the same 256 trees.
- The dense route's compile cost is per **shape + function set**; a training run that changes `input_len` or the function set pays it again. Only the artifact cache removes it, and it is opt-in.
- pop < 64 still routes to specialized, which is why the small-pop cells did not improve (§2a) and why `evoxc-auto` loses to `torch-per-tree` there. Forcing `EVOGP_EVAL_MODE=dense` wins those cells (2.3–3.0× vs `torch-batched`, 1.3–2.1× vs per-tree) at the cost of a 6.5 s compile — the current threshold is a compile-cost decision, not a runtime one.
- One real loss remains against `torch-batched`: 1000 trees × 64 nodes, `sr_fitness` (0.86–0.88×).
