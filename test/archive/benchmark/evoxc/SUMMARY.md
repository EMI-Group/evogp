# evoxc benchmark summary: baseline

> **Scope of this file.** This is the rendered report of the earlier *baseline* round, produced by `summarize.py` from `results/baseline/` (130 records); it is kept as that round's record, not as the current number set.
> The current numbers are in [`FINAL.md`](FINAL.md), which reports `results/final/after/` (HEAD) against `results/final/before/` (snapshot `530518e`); the plain-text renders of those three passes are `results/final/RENDERED.md`.
> The `evoxc-auto` columns below are the **specialized** route: the baseline `auto` policy never selected dense (every baseline record has `would_use_dense: false`), while today's `auto` resolves to dense+iree from `pop >= 64`.

Source: `test/benchmark/evoxc/results/baseline` — **130 records**, 20 cells x 7 arms.

The campaign ran **130 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **2461.2 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `30.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **10 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-28T06:01:27Z |
| git SHA (`tree_root_sha`) | fc4f9150a8b1baf93d2fa5c48bfd93c80b09d2f3 |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A38/src` |
| python | 3.14.8 |
| torch | 2.13.0+rocm7.1 |
| numpy | 2.5.4 |
| hip | 7.1.52802 |
| iree_base_compiler | 3.11.0 |
| iree_base_runtime | 3.11.0 |
| evoxc_version | 0.1.0 |
| evoxc_path | /root/evoxc/evoxc/__init__.py |
| gpu | AMD Instinct MI300X VF |
| gpu_total_memory_gb | 191.7 |
| `EVOXC_DISABLE_LOCATIONS` | `1` |

## Workload: sr_fitness

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 2.132 (2.122) | 2.730 (2.683) | 3.010 (2.964) | 2.694 (2.590) | 2.934 (2.672) | 4.367 (4.277) | 2.384 (2.351) | auto/numpy → specialized | torch-per-tree |
| `p16-L32-i32-N1024` | 2.207 (2.196) | 2.806 (2.786) | 2.760 (2.700) | 2.454 (2.438) | 2.937 (2.674) | 4.423 (4.317) | 2.465 (2.459) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i8-N1024` | 2.237 (2.213) | 3.241 (3.162) | 3.105 (3.045) | 2.901 (2.868) | 3.287 (3.006) | 4.453 (4.421) | 2.938 (2.896) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i32-N1024` | 1.977 (1.967) | 3.206 (3.188) | 3.077 (2.964) | 2.959 (2.857) | 3.067 (3.025) | 4.418 (4.343) | 3.153 (3.136) | auto/numpy → specialized | torch-per-tree |
| `p64-L32-i8-N1024` | 6.805 (6.730) | 3.717 (3.583) | 9.501 (9.377) | 2.515 (2.456) | 9.015 (8.942) | 13.571 (13.528) | 2.544 (2.496) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 6.844 (6.308) | 3.768 (3.686) | 9.142 (8.719) | 2.690 (2.626) | 8.714 (8.463) | 15.006 (14.473) | 2.501 (2.487) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 7.128 (7.067) | 4.115 (4.097) | 9.152 (9.037) | 3.180 (3.136) | 8.952 (8.334) | 14.353 (13.966) | 3.143 (3.120) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N1024` | 6.840 (6.791) | 4.132 (4.080) | 9.058 (8.566) | 3.183 (3.158) | 9.281 (9.156) | 14.166 (14.093) | 3.214 (3.167) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 30.162 (30.004) | 3.667 (3.641) | 35.601 (34.626) | 3.119 (3.082) | 38.329 (37.209) | - | 3.124 (3.067) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i32-N1024` | 28.913 (28.642) | 3.964 (3.940) | 37.325 (34.856) | 3.183 (3.133) | 37.637 (37.447) | - | 3.193 (3.152) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 28.855 (28.748) | 4.518 (4.419) | 37.529 (37.097) | 4.471 (4.434) | 37.060 (35.539) | - | 4.339 (4.307) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 28.702 (28.567) | 4.321 (4.291) | 37.881 (37.494) | 4.408 (4.329) | 37.931 (37.049) | - | 4.363 (4.311) | auto/numpy → specialized | torch-batched |
| `p1000-L32-i8-N1024` | 123.243 (121.968) | 4.458 (4.450) | 147.114 (142.275) | 5.600 (5.572) | 152.558 (152.190) | - | 5.624 (5.594) | auto/numpy → specialized | torch-batched |
| `p1000-L32-i32-N1024` | 118.341 (118.095) | 4.417 (4.408) | 148.849 (146.301) | 5.807 (5.739) | 151.190 (148.681) | - | 5.753 (5.683) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i8-N1024` | 115.073 (112.987) | 4.895 (4.881) | 148.062 (144.844) | 10.388 (10.387) | 148.925 (148.590) | - | 10.406 (10.314) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i32-N1024` | 125.329 (123.987) | 4.857 (4.841) | 146.181 (142.388) | 10.870 (10.529) | 147.159 (141.109) | - | 10.577 (10.514) | auto/numpy → specialized | torch-batched |
| `p16-L64-i32-N64` | 1.960 (1.937) | 3.126 (3.110) | 2.980 (2.977) | 2.709 (2.680) | 2.849 (2.659) | 4.249 (4.192) | 2.880 (2.830) | auto/numpy → specialized | torch-per-tree |
| `p64-L64-i32-N64` | 6.693 (6.671) | 4.226 (3.969) | 9.168 (8.552) | 3.007 (2.932) | 8.592 (7.846) | 14.026 (13.571) | 2.979 (2.961) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N64` | 27.531 (26.263) | 4.435 (4.106) | 35.886 (34.769) | 3.714 (3.679) | 34.134 (33.169) | - | 3.811 (3.731) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 112.730 (110.805) | 4.522 (4.494) | 137.932 (134.859) | 6.393 (6.336) | 134.267 (130.008) | - | 6.352 (6.349) | auto/numpy → specialized | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.71x | 0.91x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.79x | 1.01x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.73x | 0.93x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.49x | 0.63x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.89x | 1.15x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.80x | 1.02x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.90x | 1.14x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.75x | 0.96x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.50x | 0.63x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 0.90x | 1.14x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.72x | 1.04x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.77x | 1.12x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.68x | 0.99x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.50x | 0.73x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.76x | 1.10x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.64x | 1.04x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.67x | 1.08x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.64x | 1.05x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.45x | 0.73x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.63x | 1.02x |
| `p64-L32-i8-N1024` | evoxc-auto | 0.72x | 0.39x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 2.71x | 1.48x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.75x | 0.41x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.50x | 0.27x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 2.68x | 1.46x |
| `p64-L32-i32-N1024` | evoxc-auto | 0.75x | 0.41x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 2.54x | 1.40x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.79x | 0.43x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.46x | 0.25x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 2.74x | 1.51x |
| `p64-L64-i8-N1024` | evoxc-auto | 0.78x | 0.45x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 2.24x | 1.29x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.80x | 0.46x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.50x | 0.29x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 2.27x | 1.31x |
| `p64-L64-i32-N1024` | evoxc-auto | 0.76x | 0.46x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 2.15x | 1.30x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.74x | 0.45x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.48x | 0.29x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 2.13x | 1.29x |
| `p256-L32-i8-N1024` | evoxc-auto | 0.85x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 9.67x | 1.18x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.79x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 9.66x | 1.17x |
| `p256-L32-i32-N1024` | evoxc-auto | 0.77x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 9.08x | 1.25x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 9.05x | 1.24x |
| `p256-L64-i8-N1024` | evoxc-auto | 0.77x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 6.45x | 1.01x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 6.65x | 1.04x |
| `p256-L64-i32-N1024` | evoxc-auto | 0.76x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6.51x | 0.98x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.76x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6.58x | 0.99x |
| `p1000-L32-i8-N1024` | evoxc-auto | 0.84x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 22.01x | 0.80x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.81x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 21.91x | 0.79x |
| `p1000-L32-i32-N1024` | evoxc-auto | 0.80x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 20.38x | 0.76x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.78x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 20.57x | 0.77x |
| `p1000-L64-i8-N1024` | evoxc-auto | 0.78x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 11.08x | 0.47x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.77x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 11.06x | 0.47x |
| `p1000-L64-i32-N1024` | evoxc-auto | 0.86x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 11.53x | 0.45x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.85x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 11.85x | 0.46x |
| `p16-L64-i32-N64` | evoxc-auto | 0.66x | 1.05x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.72x | 1.15x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.69x | 1.10x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.46x | 0.74x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.68x | 1.09x |
| `p64-L64-i32-N64` | evoxc-auto | 0.73x | 0.46x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 2.23x | 1.41x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.78x | 0.49x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.48x | 0.30x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 2.25x | 1.42x |
| `p256-L64-i32-N64` | evoxc-auto | 0.77x | 0.12x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 7.41x | 1.19x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.81x | 0.13x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 7.22x | 1.16x |
| `p1000-L64-i32-N64` | evoxc-auto | 0.82x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 17.63x | 0.71x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.84x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 17.75x | 0.71x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.694 (1.685) | 2.666 (2.658) | 2.856 (2.747) | 2.335 (2.247) | 2.887 (2.865) | 4.244 (4.172) | 2.262 (2.254) | auto/numpy → specialized | torch-per-tree |
| `p16-L32-i32-N1024` | 1.706 (1.694) | 2.823 (2.779) | 3.079 (2.991) | 2.319 (2.245) | 2.965 (2.928) | 4.335 (4.282) | 2.256 (2.226) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i8-N1024` | 1.733 (1.714) | 3.112 (3.083) | 3.032 (2.991) | 2.813 (2.762) | 3.160 (2.894) | 4.264 (4.184) | 2.771 (2.726) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i32-N1024` | 1.716 (1.665) | 3.269 (3.195) | 3.016 (2.922) | 2.787 (2.616) | 2.926 (2.699) | 4.246 (4.180) | 2.781 (2.750) | auto/numpy → specialized | torch-per-tree |
| `p64-L32-i8-N1024` | 5.132 (5.123) | 3.452 (3.384) | 9.226 (9.179) | 2.524 (2.475) | 8.927 (8.829) | 13.420 (13.297) | 2.536 (2.316) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 4.792 (4.636) | 3.748 (3.727) | 8.868 (8.546) | 2.553 (2.467) | 8.006 (7.963) | 14.870 (14.362) | 2.648 (2.539) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i8-N1024` | 5.315 (5.283) | 4.048 (3.859) | 9.162 (9.047) | 3.187 (3.138) | 8.892 (8.880) | 13.988 (13.921) | 3.198 (3.126) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 5.101 (5.051) | 4.058 (4.043) | 9.254 (8.996) | 3.168 (3.145) | 9.244 (8.667) | 13.960 (13.460) | 3.233 (3.144) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 22.506 (22.020) | 3.858 (3.627) | 36.795 (34.795) | 3.491 (3.454) | 38.043 (37.015) | - | 3.419 (3.390) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L32-i32-N1024` | 21.683 (21.483) | 3.910 (3.888) | 37.915 (36.996) | 3.633 (3.564) | 37.541 (36.897) | - | 3.599 (3.429) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 21.634 (21.494) | 4.418 (4.261) | 37.354 (37.229) | 4.814 (4.633) | 37.431 (36.033) | - | 4.650 (4.597) | auto/numpy → specialized | torch-batched |
| `p256-L64-i32-N1024` | 21.665 (21.509) | 4.308 (4.291) | 36.021 (34.988) | 4.816 (4.669) | 37.196 (36.167) | - | 4.787 (4.658) | auto/numpy → specialized | torch-batched |
| `p1000-L32-i8-N1024` | 91.345 (90.572) | 4.426 (4.409) | 147.027 (146.243) | 7.394 (7.199) | 156.000 (153.403) | - | 7.487 (7.347) | auto/numpy → specialized | torch-batched |
| `p1000-L32-i32-N1024` | 89.906 (89.687) | 4.406 (4.391) | 147.966 (146.190) | 7.422 (7.346) | 149.986 (149.285) | - | 7.552 (7.485) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i8-N1024` | 87.638 (85.179) | 4.885 (4.856) | 148.573 (145.865) | 12.312 (12.104) | 150.585 (150.287) | - | 12.135 (12.080) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i32-N1024` | 95.082 (93.728) | 4.835 (4.823) | 147.503 (144.329) | 12.600 (12.534) | 142.928 (141.829) | - | 12.247 (12.075) | auto/numpy → specialized | torch-batched |
| `p16-L64-i32-N64` | 1.741 (1.632) | 3.018 (2.991) | 2.594 (2.538) | 2.598 (2.588) | 2.744 (2.579) | 4.195 (4.141) | 2.685 (2.654) | auto/numpy → specialized | torch-per-tree |
| `p64-L64-i32-N64` | 4.989 (4.965) | 4.123 (3.947) | 8.536 (8.355) | 2.816 (2.810) | 8.638 (8.200) | 13.472 (13.331) | 2.791 (2.745) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N64` | 20.641 (19.922) | 4.283 (4.049) | 35.092 (33.411) | 3.603 (3.565) | 33.144 (32.089) | - | 3.568 (3.555) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i32-N64` | 87.355 (86.448) | 4.219 (4.125) | 137.916 (135.118) | 6.392 (6.304) | 137.173 (134.437) | - | 6.356 (6.336) | auto/numpy → specialized | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.59x | 0.93x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.73x | 1.14x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.92x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.40x | 0.63x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.75x | 1.18x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.55x | 0.92x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.74x | 1.22x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.95x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.39x | 0.65x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 0.76x | 1.25x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.57x | 1.03x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.62x | 1.11x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.55x | 0.98x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.41x | 0.73x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.63x | 1.12x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.57x | 1.08x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.62x | 1.17x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.59x | 1.12x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.40x | 0.77x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.62x | 1.18x |
| `p64-L32-i8-N1024` | evoxc-auto | 0.56x | 0.37x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 2.03x | 1.37x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.57x | 0.39x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.26x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 2.02x | 1.36x |
| `p64-L32-i32-N1024` | evoxc-auto | 0.54x | 0.42x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 1.88x | 1.47x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.47x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.32x | 0.25x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 1.81x | 1.42x |
| `p64-L64-i8-N1024` | evoxc-auto | 0.58x | 0.44x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 1.67x | 1.27x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.60x | 0.46x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.29x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 1.66x | 1.27x |
| `p64-L64-i32-N1024` | evoxc-auto | 0.55x | 0.44x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 1.61x | 1.28x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.55x | 0.44x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.37x | 0.29x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 1.58x | 1.26x |
| `p256-L32-i8-N1024` | evoxc-auto | 0.61x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 6.45x | 1.11x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 6.58x | 1.13x |
| `p256-L32-i32-N1024` | evoxc-auto | 0.57x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 5.97x | 1.08x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 6.03x | 1.09x |
| `p256-L64-i8-N1024` | evoxc-auto | 0.58x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 4.49x | 0.92x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.58x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 4.65x | 0.95x |
| `p256-L64-i32-N1024` | evoxc-auto | 0.60x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 4.50x | 0.89x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 4.53x | 0.90x |
| `p1000-L32-i8-N1024` | evoxc-auto | 0.62x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 12.35x | 0.60x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 12.20x | 0.59x |
| `p1000-L32-i32-N1024` | evoxc-auto | 0.61x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 12.11x | 0.59x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 11.90x | 0.58x |
| `p1000-L64-i8-N1024` | evoxc-auto | 0.59x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 7.12x | 0.40x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.58x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 7.22x | 0.40x |
| `p1000-L64-i32-N1024` | evoxc-auto | 0.64x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 7.55x | 0.38x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 7.76x | 0.39x |
| `p16-L64-i32-N64` | evoxc-auto | 0.67x | 1.16x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.67x | 1.16x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.63x | 1.10x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.42x | 0.72x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.65x | 1.12x |
| `p64-L64-i32-N64` | evoxc-auto | 0.58x | 0.48x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 1.77x | 1.46x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.58x | 0.48x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.37x | 0.31x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 1.79x | 1.48x |
| `p256-L64-i32-N64` | evoxc-auto | 0.59x | 0.12x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 5.73x | 1.19x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.62x | 0.13x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 5.78x | 1.20x |
| `p1000-L64-i32-N64` | evoxc-auto | 0.63x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 13.67x | 0.66x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 13.74x | 0.66x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.710 (1.693) | 2.643 (2.624) | 2.767 (2.732) | 1.852 (1.825) | 2.736 (2.657) | 7.307 (7.206) | 1.855 (1.845) | auto/numpy → specialized | torch-per-tree |
| `p16-L32-i32-N1024` | 1.843 (1.720) | 2.719 (2.712) | 2.744 (2.531) | 1.844 (1.833) | 2.925 (2.516) | 5.681 (5.584) | 1.855 (1.832) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i8-N1024` | 1.742 (1.739) | 3.083 (3.012) | 2.874 (2.799) | 2.148 (2.140) | 2.927 (2.623) | 5.669 (5.584) | 2.110 (2.071) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i32-N1024` | 1.707 (1.691) | 3.052 (3.035) | 2.664 (2.452) | 2.113 (2.065) | 2.719 (2.454) | 5.727 (5.711) | 2.135 (2.045) | auto/numpy → specialized | torch-per-tree |
| `p64-L32-i8-N1024` | 5.216 (5.157) | 3.540 (3.496) | 9.109 (8.829) | 1.993 (1.959) | 8.170 (7.685) | 19.798 (19.514) | 2.081 (2.027) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 5.177 (4.993) | 3.611 (3.525) | 8.259 (8.142) | 1.996 (1.936) | 8.006 (7.759) | 20.128 (19.848) | 2.070 (2.026) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i8-N1024` | 5.396 (5.358) | 3.974 (3.958) | 8.320 (8.305) | 2.290 (2.256) | 8.641 (8.599) | 19.673 (19.073) | 2.242 (2.219) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N1024` | 5.202 (5.173) | 3.983 (3.925) | 8.178 (7.767) | 2.263 (2.237) | 8.377 (7.617) | 19.016 (18.878) | 2.269 (2.257) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 23.141 (22.724) | 3.825 (3.715) | 34.844 (31.821) | 2.422 (2.398) | 35.047 (33.203) | - | 2.440 (2.435) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i32-N1024` | 22.198 (22.039) | 3.665 (3.562) | 34.668 (33.786) | 2.489 (2.432) | 34.810 (34.135) | - | 2.456 (2.426) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 22.370 (22.115) | 4.429 (4.198) | 35.495 (34.746) | 2.978 (2.930) | 34.232 (33.697) | - | 2.927 (2.794) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 22.034 (21.878) | 4.277 (4.243) | 34.836 (33.593) | 2.962 (2.947) | 34.793 (33.153) | - | 3.074 (3.044) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 94.274 (93.967) | 3.932 (3.900) | 136.696 (131.065) | 4.207 (4.203) | 143.856 (142.976) | - | 4.218 (4.147) | auto/numpy → specialized | torch-batched |
| `p1000-L32-i32-N1024` | 90.851 (90.573) | 4.000 (3.947) | 138.176 (137.006) | 4.212 (4.176) | 138.759 (137.915) | - | 4.239 (4.123) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i8-N1024` | 89.425 (88.468) | 4.525 (4.444) | 137.676 (137.259) | 5.526 (5.462) | 140.442 (138.595) | - | 5.667 (5.564) | auto/numpy → specialized | torch-batched |
| `p1000-L64-i32-N1024` | 96.199 (95.228) | 4.434 (4.419) | 144.198 (142.690) | 5.548 (5.520) | 133.231 (132.661) | - | 5.516 (5.434) | auto/numpy → specialized | torch-batched |
| `p16-L64-i32-N64` | 1.682 (1.579) | 2.808 (2.758) | 2.459 (2.410) | 2.081 (2.067) | 2.747 (2.698) | 5.733 (5.605) | 2.082 (2.034) | auto/numpy → specialized | torch-per-tree |
| `p64-L64-i32-N64` | 5.158 (5.150) | 4.046 (3.913) | 8.186 (8.137) | 2.243 (2.201) | 8.596 (8.157) | 19.121 (18.744) | 2.215 (2.199) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N64` | 20.931 (20.723) | 4.173 (3.971) | 34.165 (33.365) | 2.994 (2.948) | 32.920 (31.632) | - | 2.974 (2.967) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i32-N64` | 90.392 (88.235) | 4.115 (3.998) | 137.087 (136.044) | 5.468 (5.450) | 136.534 (130.872) | - | 5.304 (4.998) | auto/numpy → specialized | torch-batched |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.62x | 0.96x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.92x | 1.43x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.62x | 0.97x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.23x | 0.36x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.92x | 1.42x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.67x | 0.99x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 1.00x | 1.47x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.63x | 0.93x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.32x | 0.48x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 0.99x | 1.47x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.61x | 1.07x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.81x | 1.44x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.60x | 1.05x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.31x | 0.54x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.83x | 1.46x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.64x | 1.15x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.81x | 1.44x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.63x | 1.12x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.30x | 0.53x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.80x | 1.43x |
| `p64-L32-i8-N1024` | evoxc-auto | 0.57x | 0.39x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 2.62x | 1.78x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.64x | 0.43x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.26x | 0.18x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 2.51x | 1.70x |
| `p64-L32-i32-N1024` | evoxc-auto | 0.63x | 0.44x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 2.59x | 1.81x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.65x | 0.45x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.26x | 0.18x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 2.50x | 1.74x |
| `p64-L64-i8-N1024` | evoxc-auto | 0.65x | 0.48x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 2.36x | 1.74x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.62x | 0.46x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.27x | 0.20x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 2.41x | 1.77x |
| `p64-L64-i32-N1024` | evoxc-auto | 0.64x | 0.49x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 2.30x | 1.76x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.62x | 0.48x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.27x | 0.21x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 2.29x | 1.76x |
| `p256-L32-i8-N1024` | evoxc-auto | 0.66x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 9.55x | 1.58x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.66x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 9.48x | 1.57x |
| `p256-L32-i32-N1024` | evoxc-auto | 0.64x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 8.92x | 1.47x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.64x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 9.04x | 1.49x |
| `p256-L64-i8-N1024` | evoxc-auto | 0.63x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 7.51x | 1.49x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.65x | 0.13x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 7.64x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-auto | 0.63x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 7.44x | 1.44x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.63x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 7.17x | 1.39x |
| `p1000-L32-i8-N1024` | evoxc-auto | 0.69x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 22.41x | 0.93x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 22.35x | 0.93x |
| `p1000-L32-i32-N1024` | evoxc-auto | 0.66x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 21.57x | 0.95x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.65x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 21.43x | 0.94x |
| `p1000-L64-i8-N1024` | evoxc-auto | 0.65x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 16.18x | 0.82x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 15.78x | 0.80x |
| `p1000-L64-i32-N1024` | evoxc-auto | 0.67x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 17.34x | 0.80x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.72x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 17.44x | 0.80x |
| `p16-L64-i32-N64` | evoxc-auto | 0.68x | 1.14x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.81x | 1.35x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.61x | 1.02x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.29x | 0.49x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.81x | 1.35x |
| `p64-L64-i32-N64` | evoxc-auto | 0.63x | 0.49x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 2.30x | 1.80x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.60x | 0.47x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.27x | 0.21x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 2.33x | 1.83x |
| `p256-L64-i32-N64` | evoxc-auto | 0.61x | 0.12x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 6.99x | 1.39x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.13x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 7.04x | 1.40x |
| `p1000-L64-i32-N64` | evoxc-auto | 0.66x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 16.53x | 0.75x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 17.04x | 0.78x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 6.797 | 8.893 | 10.492 | 9.475 | 11.421 | 625.953 | 9.262 |
| `p16-L32-i32-N1024` | 8.740 | 14.359 | 56.254 | 8.431 | 59.068 | 3662.705 | 8.553 |
| `p16-L64-i8-N1024` | 7.067 | 9.856 | 12.233 | 9.507 | 11.205 | 568.140 | 9.488 |
| `p16-L64-i32-N1024` | 8.624 | 17.463 | 65.589 | 9.506 | 64.076 | 4220.476 | 9.014 |
| `p64-L32-i8-N1024` | 9.184 | 11.623 | 40.712 | 8.593 | 38.600 | 4822.200 | 8.989 |
| `p64-L32-i32-N1024` | 9.564 | 12.636 | 53.518 | 8.577 | 51.154 | 6927.693 | 8.527 |
| `p64-L64-i8-N1024` | 9.575 | 12.646 | 36.614 | 8.982 | 39.023 | 4716.266 | 8.951 |
| `p64-L64-i32-N1024` | 9.384 | 14.138 | 52.177 | 9.065 | 51.671 | 6151.270 | 8.994 |
| `p256-L32-i8-N1024` | 19.376 | 23.653 | 194.005 | 9.203 | 197.399 | - | 9.000 |
| `p256-L32-i32-N1024` | 18.623 | 19.869 | 175.531 | 9.232 | 176.458 | - | 9.153 |
| `p256-L64-i8-N1024` | 18.805 | 25.175 | 197.160 | 10.489 | 199.960 | - | 10.229 |
| `p256-L64-i32-N1024` | 18.544 | 21.279 | 170.340 | 10.415 | 172.074 | - | 11.050 |
| `p1000-L32-i8-N1024` | 67.816 | 56.878 | 981.391 | 10.895 | 967.825 | - | 11.102 |
| `p1000-L32-i32-N1024` | 56.850 | 59.321 | 861.532 | 11.364 | 865.082 | - | 11.212 |
| `p1000-L64-i8-N1024` | 62.727 | 72.609 | 953.866 | 16.153 | 971.549 | - | 17.361 |
| `p1000-L64-i32-N1024` | 61.268 | 64.527 | 874.029 | 16.219 | 802.680 | - | 16.079 |
| `p16-L64-i32-N64` | 6.916 | 9.998 | 18.679 | 9.454 | 18.268 | 1779.283 | 9.396 |
| `p64-L64-i32-N64` | 11.496 | 19.337 | 98.348 | 9.002 | 100.336 | 9190.425 | 9.048 |
| `p256-L64-i32-N64` | 19.753 | 27.987 | 236.278 | 9.795 | 257.585 | - | 9.879 |
| `p1000-L64-i32-N64` | 55.936 | 61.496 | 759.539 | 12.212 | 731.205 | - | 12.082 |

### Eval / ops split

`median_eval_ms / median_ops_ms` (the operator part is the in-place difference on the same step, not a cross-population comparison).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.020 / 5.778 | 3.120 / 5.773 | 5.045 / 5.447 | 2.972 / 6.503 | 5.565 / 5.856 | 618.854 / 7.098 | 3.248 / 6.013 |
| `p16-L32-i32-N1024` | 2.902 / 5.838 | 8.434 / 5.925 | 50.858 / 5.396 | 2.530 / 5.901 | 52.713 / 6.355 | 3656.250 / 6.455 | 2.555 / 5.998 |
| `p16-L64-i8-N1024` | 1.052 / 6.015 | 4.043 / 5.813 | 6.354 / 5.879 | 3.640 / 5.867 | 5.462 / 5.743 | 561.545 / 6.595 | 3.398 / 6.090 |
| `p16-L64-i32-N1024` | 2.989 / 5.636 | 11.438 / 6.025 | 59.673 / 5.916 | 3.194 / 6.312 | 58.180 / 5.896 | 4213.821 / 6.655 | 3.082 / 5.932 |
| `p64-L32-i8-N1024` | 3.411 / 5.773 | 5.899 / 5.724 | 34.699 / 6.013 | 2.659 / 5.933 | 32.739 / 5.860 | 4815.277 / 6.922 | 2.726 / 6.263 |
| `p64-L32-i32-N1024` | 3.652 / 5.912 | 6.825 / 5.811 | 47.407 / 6.111 | 2.603 / 5.975 | 45.798 / 5.356 | 6920.217 / 7.476 | 2.616 / 5.911 |
| `p64-L64-i8-N1024` | 3.616 / 5.959 | 6.754 / 5.892 | 31.221 / 5.393 | 3.163 / 5.819 | 33.119 / 5.905 | 4709.314 / 6.952 | 3.151 / 5.800 |
| `p64-L64-i32-N1024` | 3.555 / 5.828 | 7.723 / 6.415 | 46.647 / 5.530 | 3.199 / 5.867 | 45.937 / 5.735 | 6144.008 / 7.261 | 3.207 / 5.786 |
| `p256-L32-i8-N1024` | 12.967 / 6.409 | 17.673 / 5.980 | 187.883 / 6.122 | 3.093 / 6.109 | 191.182 / 6.217 | - | 2.997 / 6.003 |
| `p256-L32-i32-N1024` | 12.634 / 5.989 | 13.730 / 6.139 | 169.828 / 5.703 | 3.077 / 6.155 | 170.332 / 6.127 | - | 3.054 / 6.099 |
| `p256-L64-i8-N1024` | 12.463 / 6.341 | 18.876 / 6.299 | 190.927 / 6.233 | 4.267 / 6.222 | 193.605 / 6.355 | - | 4.195 / 6.034 |
| `p256-L64-i32-N1024` | 12.552 / 5.992 | 15.466 / 5.813 | 164.479 / 5.861 | 4.324 / 6.091 | 166.379 / 5.695 | - | 4.428 / 6.621 |
| `p1000-L32-i8-N1024` | 61.755 / 6.060 | 50.956 / 5.922 | 974.594 / 6.797 | 4.937 / 5.958 | 961.155 / 6.671 | - | 5.021 / 6.081 |
| `p1000-L32-i32-N1024` | 50.781 / 6.069 | 53.343 / 5.978 | 855.077 / 6.455 | 5.098 / 6.265 | 858.696 / 6.386 | - | 5.048 / 6.164 |
| `p1000-L64-i8-N1024` | 56.390 / 6.337 | 66.336 / 6.274 | 947.214 / 6.652 | 10.070 / 6.083 | 964.857 / 6.692 | - | 10.867 / 6.494 |
| `p1000-L64-i32-N1024` | 54.571 / 6.697 | 58.440 / 6.087 | 867.158 / 6.872 | 10.189 / 6.030 | 796.515 / 6.165 | - | 10.099 / 5.980 |
| `p16-L64-i32-N64` | 1.401 / 5.516 | 4.097 / 5.901 | 12.814 / 5.864 | 3.490 / 5.965 | 12.778 / 5.490 | 1772.676 / 6.607 | 3.544 / 5.852 |
| `p64-L64-i32-N64` | 5.655 / 5.841 | 13.610 / 5.727 | 92.677 / 5.670 | 3.163 / 5.839 | 94.503 / 5.833 | 9183.328 / 7.097 | 3.131 / 5.917 |
| `p256-L64-i32-N64` | 13.801 / 5.952 | 21.829 / 6.158 | 229.959 / 6.319 | 3.621 / 6.173 | 250.557 / 7.028 | - | 3.743 / 6.136 |
| `p1000-L64-i32-N64` | 49.818 / 6.118 | 55.661 / 5.835 | 753.418 / 6.121 | 5.902 / 6.310 | 724.885 / 6.320 | - | 5.945 / 6.138 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 95.191 | 15 | 3.010 |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 3513.290 | 1 | 2.694 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 88.485 | 15 | 2.934 |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 4302.224 | 15 | 4.367 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3442.655 | 1 | 2.384 |
| `p16-L32-i32-N1024` | evoxc-auto | 94.264 | 15 | 2.760 |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 3349.539 | 1 | 2.454 |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 99.736 | 15 | 2.937 |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 4338.865 | 15 | 4.423 |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 3460.247 | 1 | 2.465 |
| `p16-L64-i8-N1024` | evoxc-auto | 98.775 | 15 | 3.105 |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 7443.881 | 1 | 2.901 |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 98.755 | 15 | 3.287 |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 4395.432 | 15 | 4.453 |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 6584.224 | 1 | 2.938 |
| `p16-L64-i32-N1024` | evoxc-auto | 94.965 | 15 | 3.077 |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 6892.124 | 1 | 2.959 |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 93.817 | 15 | 3.067 |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 4417.419 | 15 | 4.418 |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 6573.262 | 1 | 3.153 |
| `p64-L32-i8-N1024` | evoxc-auto | 215.240 | 49 | 9.501 |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 3338.383 | 1 | 2.515 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 213.204 | 49 | 9.015 |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 13903.121 | 49 | 13.571 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3424.369 | 1 | 2.544 |
| `p64-L32-i32-N1024` | evoxc-auto | 214.506 | 51 | 9.142 |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 3517.655 | 1 | 2.690 |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 205.186 | 51 | 8.714 |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 14214.174 | 51 | 15.006 |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 3414.136 | 1 | 2.501 |
| `p64-L64-i8-N1024` | evoxc-auto | 210.381 | 49 | 9.152 |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6459.572 | 1 | 3.180 |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 214.882 | 49 | 8.952 |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 13308.406 | 49 | 14.353 |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6565.181 | 1 | 3.143 |
| `p64-L64-i32-N1024` | evoxc-auto | 206.866 | 51 | 9.058 |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6574.320 | 1 | 3.183 |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 211.256 | 51 | 9.281 |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 13868.082 | 51 | 14.166 |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6473.313 | 1 | 3.214 |
| `p256-L32-i8-N1024` | evoxc-auto | 779.472 | 205 | 35.601 |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3380.400 | 1 | 3.119 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 822.959 | 205 | 38.329 |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3350.425 | 1 | 3.124 |
| `p256-L32-i32-N1024` | evoxc-auto | 818.860 | 218 | 37.325 |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3413.224 | 1 | 3.183 |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 840.690 | 218 | 37.637 |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3436.851 | 1 | 3.193 |
| `p256-L64-i8-N1024` | evoxc-auto | 824.067 | 205 | 37.529 |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 6574.111 | 1 | 4.471 |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 815.530 | 205 | 37.060 |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 6639.927 | 1 | 4.339 |
| `p256-L64-i32-N1024` | evoxc-auto | 829.229 | 218 | 37.881 |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6534.421 | 1 | 4.408 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 841.389 | 218 | 37.931 |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6497.781 | 1 | 4.363 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3289.038 | 794 | 147.114 |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 3388.581 | 1 | 5.600 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3268.208 | 794 | 152.558 |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3497.885 | 1 | 5.624 |
| `p1000-L32-i32-N1024` | evoxc-auto | 3289.374 | 818 | 148.849 |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 3437.367 | 1 | 5.807 |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 3278.228 | 818 | 151.190 |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 3460.919 | 1 | 5.753 |
| `p1000-L64-i8-N1024` | evoxc-auto | 3240.644 | 794 | 148.062 |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 6678.785 | 1 | 10.388 |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 3260.391 | 794 | 148.925 |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 6571.616 | 1 | 10.406 |
| `p1000-L64-i32-N1024` | evoxc-auto | 3332.156 | 818 | 146.181 |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 6924.580 | 1 | 10.870 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3216.599 | 818 | 147.159 |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6521.825 | 1 | 10.577 |
| `p16-L64-i32-N64` | evoxc-auto | 72.258 | 15 | 2.980 |
| `p16-L64-i32-N64` | evoxc-auto-iree | 6187.955 | 1 | 2.709 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 79.557 | 15 | 2.849 |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 4073.117 | 15 | 4.249 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6440.016 | 1 | 2.880 |
| `p64-L64-i32-N64` | evoxc-auto | 199.082 | 51 | 9.168 |
| `p64-L64-i32-N64` | evoxc-auto-iree | 6373.503 | 1 | 3.007 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 196.934 | 51 | 8.592 |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 13479.651 | 51 | 14.026 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6292.608 | 1 | 2.979 |
| `p256-L64-i32-N64` | evoxc-auto | 801.030 | 218 | 35.886 |
| `p256-L64-i32-N64` | evoxc-auto-iree | 6412.449 | 1 | 3.714 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 818.594 | 218 | 34.134 |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6373.011 | 1 | 3.811 |
| `p1000-L64-i32-N64` | evoxc-auto | 3129.727 | 818 | 137.932 |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6298.339 | 1 | 6.393 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3122.576 | 818 | 134.267 |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6358.811 | 1 | 6.352 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | batch_forward | `4ff7fdc86805329f470069a6c3b8aae7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p16-L32-i8-N1024` | batch_forward | `716e171a996460f855ad15ee526bcee0` | evoxc-specialized-iree | 1024 |
| `p16-L32-i8-N1024` | forward | `209a9a23bc848b0b78e9e120d57e1999` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 1 |
| `p16-L32-i32-N1024` | batch_forward | `5701430a0b9cf57930e15782e90c2ecd` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `529cc77cb1440dd19731e020a7e8607e` | evoxc-specialized-iree | 0 |
| `p16-L32-i32-N1024` | forward | `35463baf97c5a460b0eea39d86707e97` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `4ff7fdc86805329f470069a6c3b8aae7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p16-L64-i8-N1024` | batch_forward | `716e171a996460f855ad15ee526bcee0` | evoxc-specialized-iree | 1024 |
| `p16-L64-i8-N1024` | forward | `209a9a23bc848b0b78e9e120d57e1999` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 1 |
| `p16-L64-i32-N1024` | batch_forward | `5701430a0b9cf57930e15782e90c2ecd` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `529cc77cb1440dd19731e020a7e8607e` | evoxc-specialized-iree | 0 |
| `p16-L64-i32-N1024` | forward | `35463baf97c5a460b0eea39d86707e97` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `6e8fa35cc0377bd67505cac8c936efe4` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2048 |
| `p64-L32-i8-N1024` | batch_forward | `e12861e3d646a4318c003656339e8c55` | evoxc-specialized-iree | 2048 |
| `p64-L32-i8-N1024` | forward | `37fd68ce45259cb61c5e5c6c59a41c3b` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2 |
| `p64-L32-i8-N1024` | forward | `679d1260a48271b7f9eaaeafce90612c` | evoxc-specialized-iree | 2 |
| `p64-L32-i32-N1024` | batch_forward | `010cdcefb94d96563220c49a272319cc` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `6ca09b1eb5e161ef3eb6176ab2d48cf2` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `6e8fa35cc0377bd67505cac8c936efe4` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2048 |
| `p64-L64-i8-N1024` | batch_forward | `e12861e3d646a4318c003656339e8c55` | evoxc-specialized-iree | 2048 |
| `p64-L64-i8-N1024` | forward | `37fd68ce45259cb61c5e5c6c59a41c3b` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2 |
| `p64-L64-i8-N1024` | forward | `679d1260a48271b7f9eaaeafce90612c` | evoxc-specialized-iree | 2 |
| `p64-L64-i32-N1024` | batch_forward | `010cdcefb94d96563220c49a272319cc` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `6ca09b1eb5e161ef3eb6176ab2d48cf2` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p256-L32-i8-N1024` | batch_forward | `31b2be904b92ced0b65c329a42e6bc0f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p256-L32-i8-N1024` | batch_forward | `5a456df1db65603131c4eedcb038aecd` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p256-L32-i8-N1024` | forward | `d03896a32ebdeb367ae5e7367d0b71d8` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4 |
| `p256-L32-i8-N1024` | forward | `b69d807be015515d9b6afbc07da541df` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p256-L32-i32-N1024` | batch_forward | `5f3928f9d3a6f36239238e77eb7df793` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i32-N1024` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L64-i8-N1024` | batch_forward | `31b2be904b92ced0b65c329a42e6bc0f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p256-L64-i8-N1024` | batch_forward | `5a456df1db65603131c4eedcb038aecd` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p256-L64-i8-N1024` | forward | `d03896a32ebdeb367ae5e7367d0b71d8` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4 |
| `p256-L64-i8-N1024` | forward | `b69d807be015515d9b6afbc07da541df` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p256-L64-i32-N1024` | batch_forward | `5f3928f9d3a6f36239238e77eb7df793` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L64-i32-N1024` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p1000-L32-i8-N1024` | batch_forward | `9eb4419b453774b79662769c25081b77` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `731f6b21d50248d0c37375aa59cfd8ee` | evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `f559696c3212f4ba1df982b656adcdeb` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `f5dfcf8df1ca7dc89cc350aa57ed2d9f` | evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L32-i32-N1024` | batch_forward | `908b13d758ba2baec03b855ecc38ca4d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p1000-L32-i32-N1024` | batch_forward | `7832ebc9f0d4b673c2df28da8307164e` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p1000-L32-i32-N1024` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L32-i32-N1024` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p1000-L64-i8-N1024` | batch_forward | `9eb4419b453774b79662769c25081b77` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 7168 |
| `p1000-L64-i8-N1024` | batch_forward | `731f6b21d50248d0c37375aa59cfd8ee` | evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L64-i8-N1024` | forward | `f559696c3212f4ba1df982b656adcdeb` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 7 |
| `p1000-L64-i8-N1024` | forward | `f5dfcf8df1ca7dc89cc350aa57ed2d9f` | evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `908b13d758ba2baec03b855ecc38ca4d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p1000-L64-i32-N1024` | batch_forward | `7832ebc9f0d4b673c2df28da8307164e` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p1000-L64-i32-N1024` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L64-i32-N1024` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p16-L64-i32-N64` | batch_forward | `d86a2a4fea14cd7b737acd298647f327` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `927214dc4b0de0adcb0c6e4cc16c67b3` | evoxc-specialized-iree | 0 |
| `p16-L64-i32-N64` | forward | `35463baf97c5a460b0eea39d86707e97` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `fc6db4d28c1f58fe2d56360f2cfa48ca` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `4626235a1903869537361121a5eaf562` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `a8acb34521b847b6456ced8703a21e2d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 64 |
| `p256-L64-i32-N64` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p1000-L64-i32-N64` | batch_forward | `f5642f4213ebbe3510de1c9647b2dd33` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 256 |
| `p1000-L64-i32-N64` | batch_forward | `dec2621dfeed4fac25b576ec4c346532` | evoxc-auto-iree, evoxc-dense-iree | 256 |
| `p1000-L64-i32-N64` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L64-i32-N64` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16-L32-i8-N1024: 4/7, p16-L32-i32-N1024: 5/7, p16-L64-i8-N1024: 4/7, p16-L64-i32-N1024: 5/7, p64-L32-i8-N1024: 5/7, p64-L32-i32-N1024: 5/7, p64-L64-i8-N1024: 5/7, p64-L64-i32-N1024: 5/7, p256-L32-i8-N1024: 4/6, p256-L32-i32-N1024: 4/6, p256-L64-i8-N1024: 4/6, p256-L64-i32-N1024: 4/6, p1000-L32-i8-N1024: 4/6, p1000-L32-i32-N1024: 4/6, p1000-L64-i8-N1024: 4/6, p1000-L64-i32-N1024: 4/6, p16-L64-i32-N64: 2/7, p64-L64-i32-N64: 3/7, p256-L64-i32-N64: 3/6, p1000-L64-i32-N64: 3/6.

## Caveats

- Every scheduled config in `campaign.json` finished `ok`.
- **10 config(s) never ran**: excluded by the routing scope rule for compile cost, so they have no row anywhere above.

| cell | arm | reason |
|---|---|---|
| `p256-L32-i8-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p256-L64-i32-N64` | evoxc-specialized-iree | dropped by the routing scope rule |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | dropped by the routing scope rule |
