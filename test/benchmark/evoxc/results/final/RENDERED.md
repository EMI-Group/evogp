Rendered evoxc benchmark summaries for the final campaign: the after (post-optimization HEAD) campaign, the before (pre-optimization snapshot) campaign, and the after-vs-before compare render, produced by the node's own `summarize.py` from the committed records.
Result dirs: `results/final/after` (130 records, tree_root_sha 761ef05ce87d456c364cbb706fa234d9b184ecdd) and `results/final/before` (90 records, tree_root `/tmp/evogp_preopt`, driver note "final campaign: pre-optimization snapshot 530518e").
Commands (run from `test/benchmark/evoxc`): `python3 summarize.py --results results/final/after --out /tmp/final_after.md`; `python3 summarize.py --results results/final/before --out /tmp/final_before.md`; `python3 summarize.py --results results/final/after --compare-results results/final/before --compare-label before --out /tmp/final_compare.md`.

<!-- ===== /tmp/final_after.md ===== -->

# evoxc benchmark summary: after

Source: `results/final/after` — **130 records**, 20 cells x 7 arms.

The campaign ran **130 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **2478.5 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **10 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-28T09:09:41Z |
| git SHA (`tree_root_sha`) | 761ef05ce87d456c364cbb706fa234d9b184ecdd |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A50/src` |
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
| `p16-L32-i8-N1024` | 1.844 (1.737) | 2.779 (2.747) | 2.468 (2.254) | 3.864 (3.843) | 2.459 (2.428) | 3.978 (3.812) | 0.974 (0.956) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.962 (1.950) | 2.848 (2.757) | 2.476 (2.431) | 4.060 (3.899) | 2.567 (2.526) | 3.879 (3.827) | 0.941 (0.931) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.858 (1.829) | 3.240 (2.930) | 2.299 (2.265) | 3.911 (3.835) | 2.312 (2.244) | 3.880 (3.797) | 1.456 (1.433) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 1.922 (1.737) | 3.242 (2.974) | 2.627 (2.391) | 3.937 (3.884) | 2.607 (2.526) | 3.801 (3.792) | 1.447 (1.414) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 7.476 (7.152) | 3.572 (3.547) | 0.924 (0.912) | 0.971 (0.957) | 9.150 (9.001) | 15.451 (15.065) | 1.029 (0.991) | auto/numpy → dense | evoxc-auto |
| `p64-L32-i32-N1024` | 7.433 (7.362) | 3.607 (3.319) | 0.964 (0.932) | 0.986 (0.960) | 9.969 (9.310) | 15.522 (15.470) | 0.950 (0.933) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 7.843 (7.505) | 4.038 (4.005) | 1.530 (1.509) | 1.528 (1.511) | 10.183 (10.046) | 15.640 (15.486) | 1.535 (1.526) | auto/numpy → dense | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 7.664 (7.588) | 4.109 (4.074) | 1.551 (1.524) | 1.651 (1.574) | 10.145 (10.038) | 15.584 (15.372) | 1.514 (1.494) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L32-i8-N1024` | 30.177 (30.075) | 4.002 (3.732) | 1.101 (1.086) | 1.122 (1.068) | 40.451 (39.987) | - | 1.103 (1.090) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 30.277 (30.143) | 3.999 (3.958) | 1.133 (1.089) | 1.127 (1.100) | 39.304 (37.412) | - | 1.133 (1.117) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 30.483 (30.410) | 4.299 (4.160) | 1.977 (1.966) | 2.000 (1.974) | 38.968 (36.667) | - | 1.999 (1.971) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 30.502 (30.359) | 4.507 (4.421) | 2.046 (2.003) | 2.107 (2.094) | 39.434 (37.100) | - | 2.160 (2.122) | auto/numpy → dense | evoxc-auto |
| `p1000-L32-i8-N1024` | 120.749 (119.966) | 4.376 (4.367) | 1.903 (1.867) | 1.914 (1.910) | 152.270 (151.016) | - | 1.906 (1.864) | auto/numpy → dense | evoxc-auto |
| `p1000-L32-i32-N1024` | 123.636 (119.848) | 4.393 (4.377) | 1.974 (1.947) | 1.987 (1.968) | 155.060 (148.386) | - | 2.031 (2.006) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i8-N1024` | 120.707 (119.679) | 4.866 (4.833) | 5.510 (5.499) | 5.537 (5.529) | 154.152 (151.373) | - | 5.535 (5.508) | auto/numpy → dense | torch-batched |
| `p1000-L64-i32-N1024` | 119.694 (118.781) | 4.887 (4.872) | 5.665 (5.639) | 5.668 (5.647) | 155.934 (154.521) | - | 5.744 (5.702) | auto/numpy → dense | torch-batched |
| `p16-L64-i32-N64` | 1.842 (1.796) | 3.183 (3.153) | 2.380 (2.361) | 3.874 (3.817) | 2.397 (2.394) | 3.849 (3.838) | 1.464 (1.419) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 7.420 (7.370) | 4.119 (3.996) | 1.435 (1.424) | 1.441 (1.426) | 9.532 (9.483) | 15.502 (15.333) | 1.425 (1.391) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N64` | 29.796 (29.266) | 4.520 (4.307) | 1.464 (1.433) | 1.534 (1.486) | 41.083 (40.599) | - | 1.462 (1.429) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N64` | 118.946 (118.657) | 4.978 (4.954) | 1.582 (1.568) | 1.549 (1.539) | 139.402 (138.237) | - | 1.571 (1.533) | auto/numpy → dense | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.75x | 1.13x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.48x | 0.72x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.75x | 1.13x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.46x | 0.70x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.89x | 2.85x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.79x | 1.15x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.48x | 0.70x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.76x | 1.11x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.51x | 0.73x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 2.08x | 3.03x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.81x | 1.41x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.48x | 0.83x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.80x | 1.40x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.48x | 0.83x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 1.28x | 2.23x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.73x | 1.23x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.49x | 0.82x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.74x | 1.24x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.51x | 0.85x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 1.33x | 2.24x |
| `p64-L32-i8-N1024` | evoxc-auto | 8.09x | 3.87x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 7.70x | 3.68x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.82x | 0.39x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.48x | 0.23x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 7.27x | 3.47x |
| `p64-L32-i32-N1024` | evoxc-auto | 7.71x | 3.74x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 7.54x | 3.66x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.75x | 0.36x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.48x | 0.23x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 7.83x | 3.80x |
| `p64-L64-i8-N1024` | evoxc-auto | 5.13x | 2.64x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 5.13x | 2.64x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.77x | 0.40x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.50x | 0.26x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 5.11x | 2.63x |
| `p64-L64-i32-N1024` | evoxc-auto | 4.94x | 2.65x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 4.64x | 2.49x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.76x | 0.41x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.49x | 0.26x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 5.06x | 2.72x |
| `p256-L32-i8-N1024` | evoxc-auto | 27.42x | 3.64x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 26.89x | 3.57x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.75x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 27.36x | 3.63x |
| `p256-L32-i32-N1024` | evoxc-auto | 26.73x | 3.53x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 26.86x | 3.55x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 26.71x | 3.53x |
| `p256-L64-i8-N1024` | evoxc-auto | 15.42x | 2.17x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 15.25x | 2.15x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.11x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 15.25x | 2.15x |
| `p256-L64-i32-N1024` | evoxc-auto | 14.91x | 2.20x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 14.48x | 2.14x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 14.12x | 2.09x |
| `p1000-L32-i8-N1024` | evoxc-auto | 63.44x | 2.30x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 63.07x | 2.29x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.79x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 63.36x | 2.30x |
| `p1000-L32-i32-N1024` | evoxc-auto | 62.64x | 2.23x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 62.21x | 2.21x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.80x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 60.86x | 2.16x |
| `p1000-L64-i8-N1024` | evoxc-auto | 21.91x | 0.88x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 21.80x | 0.88x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 21.81x | 0.88x |
| `p1000-L64-i32-N1024` | evoxc-auto | 21.13x | 0.86x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 21.12x | 0.86x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 20.84x | 0.85x |
| `p16-L64-i32-N64` | evoxc-auto | 0.77x | 1.34x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.48x | 0.82x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.77x | 1.33x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.48x | 0.83x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.26x | 2.17x |
| `p64-L64-i32-N64` | evoxc-auto | 5.17x | 2.87x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 5.15x | 2.86x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.78x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.48x | 0.27x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 5.21x | 2.89x |
| `p256-L64-i32-N64` | evoxc-auto | 20.35x | 3.09x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 19.42x | 2.95x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.73x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 20.39x | 3.09x |
| `p1000-L64-i32-N64` | evoxc-auto | 75.19x | 3.15x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 76.78x | 3.21x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.85x | 0.04x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 75.69x | 3.17x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.389 (1.373) | 2.749 (2.729) | 2.255 (2.233) | 3.752 (3.694) | 2.202 (2.148) | 3.780 (3.649) | 1.020 (0.986) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.467 (1.455) | 2.790 (2.745) | 2.419 (2.322) | 4.661 (3.926) | 2.291 (2.280) | 3.791 (3.775) | 1.003 (0.987) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.389 (1.375) | 3.177 (3.135) | 2.495 (2.215) | 3.907 (3.708) | 2.225 (2.192) | 3.801 (3.701) | 1.494 (1.477) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i32-N1024` | 1.448 (1.424) | 3.057 (2.963) | 2.524 (2.447) | 3.802 (3.795) | 2.458 (2.381) | 3.819 (3.743) | 1.467 (1.444) | auto/numpy → specialized | torch-per-tree |
| `p64-L32-i8-N1024` | 5.673 (5.383) | 3.550 (3.526) | 1.207 (1.160) | 1.196 (1.159) | 9.987 (9.914) | 15.009 (14.864) | 1.196 (1.184) | auto/numpy → dense | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 5.708 (5.677) | 3.528 (3.504) | 1.153 (1.087) | 1.203 (1.182) | 9.861 (9.443) | 15.760 (15.188) | 1.151 (1.133) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 5.839 (5.785) | 3.981 (3.761) | 1.738 (1.723) | 1.762 (1.684) | 9.967 (9.922) | 15.183 (14.779) | 1.748 (1.671) | auto/numpy → dense | evoxc-auto |
| `p64-L64-i32-N1024` | 5.733 (5.709) | 4.000 (3.974) | 1.758 (1.698) | 1.755 (1.714) | 9.301 (9.089) | 15.397 (14.986) | 1.720 (1.702) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L32-i8-N1024` | 24.339 (24.121) | 3.798 (3.668) | 1.801 (1.748) | 1.920 (1.893) | 40.746 (39.823) | - | 1.929 (1.909) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 23.143 (22.757) | 3.977 (3.790) | 1.935 (1.921) | 1.916 (1.883) | 39.971 (37.989) | - | 1.940 (1.917) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 23.322 (23.223) | 4.323 (4.315) | 2.807 (2.650) | 2.846 (2.812) | 39.271 (37.780) | - | 2.858 (2.582) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.438 (23.269) | 4.365 (4.355) | 2.856 (2.676) | 2.814 (2.725) | 39.315 (37.751) | - | 2.829 (2.650) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 92.121 (91.921) | 4.346 (4.331) | 3.917 (3.835) | 3.749 (3.571) | 155.661 (154.446) | - | 3.973 (3.924) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i32-N1024` | 94.942 (94.539) | 4.383 (4.365) | 3.898 (3.840) | 4.002 (3.934) | 152.599 (152.248) | - | 3.879 (3.740) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i8-N1024` | 92.167 (91.515) | 4.816 (4.808) | 7.485 (7.418) | 7.479 (7.419) | 153.566 (151.552) | - | 7.487 (7.325) | auto/numpy → dense | torch-batched |
| `p1000-L64-i32-N1024` | 92.688 (91.733) | 4.886 (4.847) | 7.653 (7.437) | 7.566 (7.410) | 155.765 (151.898) | - | 7.469 (7.428) | auto/numpy → dense | torch-batched |
| `p16-L64-i32-N64` | 1.394 (1.371) | 3.113 (3.095) | 2.333 (2.273) | 3.946 (3.888) | 2.326 (2.319) | 3.832 (3.764) | 1.385 (1.368) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.663 (5.631) | 4.015 (3.965) | 1.432 (1.385) | 1.428 (1.403) | 9.303 (8.966) | 15.443 (15.185) | 1.421 (1.393) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N64` | 21.821 (21.069) | 4.483 (4.254) | 1.508 (1.473) | 1.579 (1.555) | 41.590 (41.459) | - | 1.514 (1.483) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N64` | 91.198 (90.986) | 4.927 (4.885) | 1.798 (1.714) | 1.765 (1.729) | 143.147 (141.764) | - | 1.798 (1.752) | auto/numpy → dense | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.62x | 1.22x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.37x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.63x | 1.25x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.37x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.36x | 2.70x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.61x | 1.15x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.31x | 0.60x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.64x | 1.22x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.39x | 0.74x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 1.46x | 2.78x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.56x | 1.27x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.36x | 0.81x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.62x | 1.43x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.37x | 0.84x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.93x | 2.13x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.57x | 1.21x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.38x | 0.80x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.59x | 1.24x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.38x | 0.80x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.99x | 2.08x |
| `p64-L32-i8-N1024` | evoxc-auto | 4.70x | 2.94x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 4.74x | 2.97x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.57x | 0.36x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.24x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 4.74x | 2.97x |
| `p64-L32-i32-N1024` | evoxc-auto | 4.95x | 3.06x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 4.75x | 2.93x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.36x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.36x | 0.22x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 4.96x | 3.06x |
| `p64-L64-i8-N1024` | evoxc-auto | 3.36x | 2.29x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 3.31x | 2.26x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.40x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.26x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 3.34x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-auto | 3.26x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 3.27x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.62x | 0.43x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.37x | 0.26x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 3.33x | 2.33x |
| `p256-L32-i8-N1024` | evoxc-auto | 13.52x | 2.11x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 12.68x | 1.98x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.60x | 0.09x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 12.62x | 1.97x |
| `p256-L32-i32-N1024` | evoxc-auto | 11.96x | 2.06x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 12.08x | 2.08x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 11.93x | 2.05x |
| `p256-L64-i8-N1024` | evoxc-auto | 8.31x | 1.54x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 8.19x | 1.52x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.11x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 8.16x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-auto | 8.21x | 1.53x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 8.33x | 1.55x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.29x | 1.54x |
| `p1000-L32-i8-N1024` | evoxc-auto | 23.52x | 1.11x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 24.57x | 1.16x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 23.19x | 1.09x |
| `p1000-L32-i32-N1024` | evoxc-auto | 24.36x | 1.12x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 23.72x | 1.10x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.62x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 24.48x | 1.13x |
| `p1000-L64-i8-N1024` | evoxc-auto | 12.31x | 0.64x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 12.32x | 0.64x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.60x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 12.31x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-auto | 12.11x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 12.25x | 0.65x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 12.41x | 0.65x |
| `p16-L64-i32-N64` | evoxc-auto | 0.60x | 1.33x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.35x | 0.79x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.60x | 1.34x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.36x | 0.81x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.01x | 2.25x |
| `p64-L64-i32-N64` | evoxc-auto | 3.96x | 2.80x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 3.97x | 2.81x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.61x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.37x | 0.26x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 3.98x | 2.82x |
| `p256-L64-i32-N64` | evoxc-auto | 14.47x | 2.97x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 13.82x | 2.84x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.52x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 14.41x | 2.96x |
| `p1000-L64-i32-N64` | evoxc-auto | 50.73x | 2.74x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 51.66x | 2.79x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 50.71x | 2.74x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.423 (1.404) | 2.701 (2.554) | 2.266 (2.197) | 5.017 (4.955) | 2.195 (1.976) | 5.232 (5.002) | 0.643 (0.640) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.511 (1.470) | 2.739 (2.690) | 2.159 (2.147) | 5.452 (5.342) | 2.229 (1.985) | 5.127 (5.047) | 0.642 (0.615) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.395 (1.335) | 3.417 (3.382) | 2.030 (1.996) | 5.115 (5.014) | 2.054 (2.022) | 5.101 (4.925) | 0.861 (0.850) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 1.476 (1.448) | 2.949 (2.931) | 2.048 (2.007) | 4.989 (4.882) | 2.216 (2.012) | 4.977 (4.883) | 0.893 (0.840) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.795 (5.773) | 3.459 (3.427) | 0.643 (0.624) | 0.696 (0.678) | 8.447 (8.280) | 24.850 (24.477) | 0.656 (0.649) | auto/numpy → dense | evoxc-auto |
| `p64-L32-i32-N1024` | 5.900 (5.787) | 3.492 (3.431) | 0.641 (0.634) | 0.649 (0.622) | 9.221 (8.882) | 22.629 (22.160) | 0.625 (0.617) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 5.917 (5.851) | 4.028 (3.928) | 0.843 (0.828) | 0.859 (0.833) | 9.346 (9.329) | 24.235 (23.559) | 0.871 (0.863) | auto/numpy → dense | evoxc-auto |
| `p64-L64-i32-N1024` | 5.819 (5.802) | 3.948 (3.875) | 0.861 (0.851) | 0.870 (0.835) | 9.215 (8.470) | 22.400 (22.184) | 0.863 (0.811) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i8-N1024` | 25.161 (24.712) | 3.951 (3.882) | 0.637 (0.624) | 0.682 (0.659) | 37.274 (36.921) | - | 0.663 (0.643) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 23.943 (23.802) | 3.922 (3.876) | 0.662 (0.632) | 0.666 (0.651) | 35.880 (34.103) | - | 0.655 (0.642) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 23.531 (23.215) | 4.312 (4.264) | 0.853 (0.849) | 0.860 (0.839) | 37.430 (36.932) | - | 0.925 (0.903) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.730 (22.678) | 4.340 (4.312) | 0.871 (0.855) | 0.852 (0.842) | 35.868 (34.208) | - | 0.868 (0.846) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 97.247 (96.309) | 4.056 (4.009) | 0.651 (0.636) | 0.646 (0.640) | 145.632 (142.609) | - | 0.694 (0.664) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i32-N1024` | 96.700 (96.512) | 3.993 (3.978) | 0.653 (0.652) | 0.658 (0.631) | 147.776 (146.109) | - | 0.654 (0.648) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i8-N1024` | 93.874 (93.169) | 4.959 (4.801) | 0.878 (0.845) | 0.934 (0.922) | 141.689 (139.923) | - | 0.877 (0.867) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 94.010 (92.354) | 4.562 (4.510) | 0.858 (0.831) | 0.932 (0.899) | 142.466 (139.444) | - | 0.872 (0.863) | auto/numpy → dense | evoxc-auto |
| `p16-L64-i32-N64` | 1.451 (1.415) | 3.122 (2.956) | 2.289 (2.250) | 4.949 (4.933) | 2.243 (2.201) | 5.817 (5.710) | 0.864 (0.822) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 6.270 (6.089) | 3.917 (3.855) | 0.866 (0.854) | 0.851 (0.823) | 9.031 (8.173) | 22.916 (22.655) | 0.895 (0.875) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i32-N64` | 23.735 (23.532) | 4.326 (4.247) | 0.924 (0.872) | 0.874 (0.857) | 40.567 (40.446) | - | 0.972 (0.937) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 97.933 (97.083) | 4.795 (4.777) | 0.863 (0.842) | 0.884 (0.835) | 141.449 (140.101) | - | 0.845 (0.824) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.63x | 1.19x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.28x | 0.54x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.65x | 1.23x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.27x | 0.52x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 2.21x | 4.20x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.70x | 1.27x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.28x | 0.50x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.68x | 1.23x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.29x | 0.53x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 2.35x | 4.26x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.69x | 1.68x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.27x | 0.67x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.68x | 1.66x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.27x | 0.67x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 1.62x | 3.97x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.72x | 1.44x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.30x | 0.59x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 1.33x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.30x | 0.59x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 1.65x | 3.30x |
| `p64-L32-i8-N1024` | evoxc-auto | 9.01x | 5.38x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 8.33x | 4.97x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.69x | 0.41x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.23x | 0.14x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 8.83x | 5.27x |
| `p64-L32-i32-N1024` | evoxc-auto | 9.21x | 5.45x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 9.09x | 5.38x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.64x | 0.38x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.26x | 0.15x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 9.44x | 5.59x |
| `p64-L64-i8-N1024` | evoxc-auto | 7.02x | 4.78x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6.89x | 4.69x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.63x | 0.43x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.24x | 0.17x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6.79x | 4.62x |
| `p64-L64-i32-N1024` | evoxc-auto | 6.76x | 4.59x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6.69x | 4.54x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.63x | 0.43x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.26x | 0.18x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6.74x | 4.57x |
| `p256-L32-i8-N1024` | evoxc-auto | 39.52x | 6.21x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 36.90x | 5.79x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.68x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 37.92x | 5.95x |
| `p256-L32-i32-N1024` | evoxc-auto | 36.14x | 5.92x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 35.93x | 5.89x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 36.54x | 5.99x |
| `p256-L64-i8-N1024` | evoxc-auto | 27.59x | 5.05x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 27.36x | 5.01x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.63x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 25.43x | 4.66x |
| `p256-L64-i32-N1024` | evoxc-auto | 27.23x | 4.98x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 27.87x | 5.10x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.66x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 27.35x | 5.00x |
| `p1000-L32-i8-N1024` | evoxc-auto | 149.43x | 6.23x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 150.56x | 6.28x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.67x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 140.06x | 5.84x |
| `p1000-L32-i32-N1024` | evoxc-auto | 147.97x | 6.11x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 146.96x | 6.07x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.65x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 147.77x | 6.10x |
| `p1000-L64-i8-N1024` | evoxc-auto | 106.87x | 5.65x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 100.50x | 5.31x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 107.08x | 5.66x |
| `p1000-L64-i32-N1024` | evoxc-auto | 109.52x | 5.31x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 100.83x | 4.89x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 107.80x | 5.23x |
| `p16-L64-i32-N64` | evoxc-auto | 0.63x | 1.36x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.29x | 0.63x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.65x | 1.39x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.25x | 0.54x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.68x | 3.61x |
| `p64-L64-i32-N64` | evoxc-auto | 7.24x | 4.52x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 7.37x | 4.61x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.69x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.27x | 0.17x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 7.01x | 4.38x |
| `p256-L64-i32-N64` | evoxc-auto | 25.68x | 4.68x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 27.17x | 4.95x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.59x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 24.43x | 4.45x |
| `p1000-L64-i32-N64` | evoxc-auto | 113.48x | 5.56x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 110.81x | 5.42x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.69x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 115.97x | 5.68x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.308 | 4.859 | 6.626 | 755.470 | 6.370 | 799.942 | 5.042 |
| `p16-L32-i32-N1024` | 3.520 | 4.830 | 6.725 | 736.358 | 7.007 | 744.433 | 5.392 |
| `p16-L64-i8-N1024` | 3.120 | 5.596 | 6.578 | 751.604 | 6.607 | 746.820 | 5.653 |
| `p16-L64-i32-N1024` | 3.280 | 5.752 | 6.637 | 736.565 | 6.926 | 754.958 | 6.494 |
| `p64-L32-i8-N1024` | 5.391 | 7.058 | 4.934 | 4.989 | 40.587 | 2760.215 | 4.779 |
| `p64-L32-i32-N1024` | 5.430 | 8.561 | 4.943 | 4.913 | 62.907 | 6326.840 | 4.951 |
| `p64-L64-i8-N1024` | 5.061 | 7.136 | 5.393 | 5.407 | 38.733 | 2739.012 | 5.365 |
| `p64-L64-i32-N1024` | 5.548 | 9.673 | 5.417 | 5.410 | 64.111 | 6477.668 | 5.319 |
| `p256-L32-i8-N1024` | 20.537 | 23.165 | 5.381 | 5.468 | 317.179 | - | 5.548 |
| `p256-L32-i32-N1024` | 21.598 | 24.759 | 5.484 | 5.468 | 374.651 | - | 5.467 |
| `p256-L64-i8-N1024` | 19.263 | 25.957 | 6.607 | 6.587 | 313.787 | - | 6.797 |
| `p256-L64-i32-N1024` | 21.141 | 28.043 | 6.630 | 6.632 | 370.964 | - | 6.574 |
| `p1000-L32-i8-N1024` | 60.794 | 56.344 | 7.722 | 7.940 | 1071.906 | - | 9.176 |
| `p1000-L32-i32-N1024` | 58.237 | 52.079 | 7.436 | 7.363 | 935.422 | - | 7.360 |
| `p1000-L64-i8-N1024` | 61.938 | 77.896 | 12.832 | 12.873 | 1044.985 | - | 12.661 |
| `p1000-L64-i32-N1024` | 56.248 | 67.241 | 12.410 | 12.803 | 887.258 | - | 12.593 |
| `p16-L64-i32-N64` | 3.232 | 5.808 | 6.687 | 754.663 | 6.726 | 755.414 | 5.795 |
| `p64-L64-i32-N64` | 5.129 | 7.478 | 5.329 | 5.319 | 24.583 | 4031.834 | 5.532 |
| `p256-L64-i32-N64` | 24.117 | 29.000 | 6.066 | 6.444 | 521.846 | - | 6.634 |
| `p1000-L64-i32-N64` | 56.602 | 64.290 | 8.465 | 8.303 | 1040.052 | - | 8.205 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 0.961 / 2.347 (op 71%) | 2.550 / 2.309 (op 48%) | 4.308 / 2.318 (op 35%) | 752.831 / 2.638 (op 0%) | 4.237 / 2.133 (op 33%) | 797.190 / 2.753 (op 0%) | 2.640 / 2.402 (op 48%) |
| `p16-L32-i32-N1024` | 1.000 / 2.520 (op 72%) | 2.577 / 2.253 (op 47%) | 4.245 / 2.481 (op 37%) | 733.684 / 2.674 (op 0%) | 4.598 / 2.409 (op 34%) | 741.732 / 2.701 (op 0%) | 2.986 / 2.406 (op 45%) |
| `p16-L64-i8-N1024` | 0.942 / 2.179 (op 70%) | 3.344 / 2.252 (op 40%) | 4.226 / 2.352 (op 36%) | 748.938 / 2.666 (op 0%) | 4.209 / 2.398 (op 36%) | 744.189 / 2.631 (op 0%) | 3.272 / 2.381 (op 42%) |
| `p16-L64-i32-N1024` | 0.934 / 2.346 (op 72%) | 3.375 / 2.376 (op 41%) | 4.454 / 2.183 (op 33%) | 733.944 / 2.621 (op 0%) | 4.558 / 2.368 (op 34%) | 752.337 / 2.620 (op 0%) | 4.072 / 2.421 (op 37%) |
| `p64-L32-i8-N1024` | 3.015 / 2.377 (op 44%) | 4.094 / 2.965 (op 42%) | 2.455 / 2.479 (op 50%) | 2.398 / 2.592 (op 52%) | 38.136 / 2.451 (op 6%) | 2757.099 / 3.117 (op 0%) | 2.385 / 2.393 (op 50%) |
| `p64-L32-i32-N1024` | 3.100 / 2.330 (op 43%) | 6.238 / 2.323 (op 27%) | 2.344 / 2.599 (op 53%) | 2.326 / 2.586 (op 53%) | 60.742 / 2.166 (op 3%) | 6323.386 / 3.454 (op 0%) | 2.394 / 2.556 (op 52%) |
| `p64-L64-i8-N1024` | 2.709 / 2.351 (op 46%) | 4.790 / 2.345 (op 33%) | 2.998 / 2.396 (op 44%) | 3.030 / 2.378 (op 44%) | 36.327 / 2.406 (op 6%) | 2735.926 / 3.086 (op 0%) | 3.037 / 2.328 (op 43%) |
| `p64-L64-i32-N1024` | 3.160 / 2.388 (op 43%) | 7.281 / 2.392 (op 25%) | 3.043 / 2.373 (op 44%) | 3.055 / 2.355 (op 44%) | 61.651 / 2.460 (op 4%) | 6474.211 / 3.457 (op 0%) | 2.995 / 2.323 (op 44%) |
| `p256-L32-i8-N1024` | 17.841 / 2.695 (op 13%) | 20.725 / 2.440 (op 11%) | 2.899 / 2.482 (op 46%) | 2.942 / 2.527 (op 46%) | 314.527 / 2.652 (op 1%) | - | 2.944 / 2.604 (op 47%) |
| `p256-L32-i32-N1024` | 18.997 / 2.600 (op 12%) | 22.209 / 2.550 (op 10%) | 2.966 / 2.518 (op 46%) | 3.007 / 2.461 (op 45%) | 372.103 / 2.549 (op 1%) | - | 2.963 / 2.504 (op 46%) |
| `p256-L64-i8-N1024` | 16.804 / 2.459 (op 13%) | 23.480 / 2.477 (op 10%) | 4.132 / 2.476 (op 37%) | 4.134 / 2.453 (op 37%) | 311.068 / 2.719 (op 1%) | - | 4.212 / 2.586 (op 38%) |
| `p256-L64-i32-N1024` | 18.559 / 2.583 (op 12%) | 25.430 / 2.613 (op 9%) | 4.190 / 2.440 (op 37%) | 4.191 / 2.442 (op 37%) | 368.163 / 2.801 (op 1%) | - | 4.169 / 2.406 (op 37%) |
| `p1000-L32-i8-N1024` | 58.091 / 2.703 (op 4%) | 53.903 / 2.442 (op 4%) | 5.003 / 2.719 (op 35%) | 5.152 / 2.788 (op 35%) | 1068.883 / 3.023 (op 0%) | - | 6.558 / 2.617 (op 29%) |
| `p1000-L32-i32-N1024` | 55.423 / 2.813 (op 5%) | 49.528 / 2.550 (op 5%) | 4.867 / 2.570 (op 35%) | 4.905 / 2.458 (op 33%) | 932.592 / 2.831 (op 0%) | - | 4.827 / 2.533 (op 34%) |
| `p1000-L64-i8-N1024` | 59.104 / 2.833 (op 5%) | 75.311 / 2.586 (op 3%) | 10.181 / 2.651 (op 21%) | 10.129 / 2.744 (op 21%) | 1041.964 / 3.021 (op 0%) | - | 10.208 / 2.453 (op 19%) |
| `p1000-L64-i32-N1024` | 53.619 / 2.629 (op 5%) | 64.601 / 2.640 (op 4%) | 9.861 / 2.549 (op 21%) | 10.198 / 2.605 (op 20%) | 884.317 / 2.941 (op 0%) | - | 10.109 / 2.484 (op 20%) |
| `p16-L64-i32-N64` | 0.928 / 2.304 (op 71%) | 3.465 / 2.343 (op 40%) | 4.343 / 2.344 (op 35%) | 751.947 / 2.716 (op 0%) | 4.359 / 2.366 (op 35%) | 752.777 / 2.637 (op 0%) | 3.229 / 2.566 (op 44%) |
| `p64-L64-i32-N64` | 2.773 / 2.356 (op 46%) | 5.143 / 2.335 (op 31%) | 2.897 / 2.432 (op 46%) | 2.893 / 2.426 (op 46%) | 22.159 / 2.424 (op 10%) | 4028.691 / 3.143 (op 0%) | 2.988 / 2.544 (op 46%) |
| `p256-L64-i32-N64` | 21.558 / 2.559 (op 11%) | 26.352 / 2.648 (op 9%) | 3.577 / 2.489 (op 41%) | 3.754 / 2.691 (op 42%) | 518.819 / 3.027 (op 1%) | - | 3.944 / 2.689 (op 41%) |
| `p1000-L64-i32-N64` | 53.955 / 2.648 (op 5%) | 61.836 / 2.454 (op 4%) | 5.847 / 2.619 (op 31%) | 5.777 / 2.526 (op 30%) | 1037.111 / 2.941 (op 0%) | - | 5.709 / 2.495 (op 30%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.50x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.52x | 0.76x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.66x | 0.96x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.52x | 0.72x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.50x | 0.69x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 0.65x | 0.90x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.47x | 0.85x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.47x | 0.85x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.55x | 0.99x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.49x | 0.87x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.47x | 0.83x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.51x | 0.89x |
| `p64-L32-i8-N1024` | evoxc-auto | 1.09x | 1.43x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 1.08x | 1.41x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.13x | 0.17x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 1.13x | 1.48x |
| `p64-L32-i32-N1024` | evoxc-auto | 1.10x | 1.73x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 1.11x | 1.74x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.09x | 0.14x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 1.10x | 1.73x |
| `p64-L64-i8-N1024` | evoxc-auto | 0.94x | 1.32x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 0.94x | 1.32x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.13x | 0.18x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 0.94x | 1.33x |
| `p64-L64-i32-N1024` | evoxc-auto | 1.02x | 1.79x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 1.03x | 1.79x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.09x | 0.15x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 1.04x | 1.82x |
| `p256-L32-i8-N1024` | evoxc-auto | 3.82x | 4.31x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3.76x | 4.24x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3.70x | 4.18x |
| `p256-L32-i32-N1024` | evoxc-auto | 3.94x | 4.51x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3.95x | 4.53x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3.95x | 4.53x |
| `p256-L64-i8-N1024` | evoxc-auto | 2.92x | 3.93x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 2.92x | 3.94x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 2.83x | 3.82x |
| `p256-L64-i32-N1024` | evoxc-auto | 3.19x | 4.23x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 3.19x | 4.23x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 3.22x | 4.27x |
| `p1000-L32-i8-N1024` | evoxc-auto | 7.87x | 7.30x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 7.66x | 7.10x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.05x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 6.63x | 6.14x |
| `p1000-L32-i32-N1024` | evoxc-auto | 7.83x | 7.00x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 7.91x | 7.07x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.06x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 7.91x | 7.08x |
| `p1000-L64-i8-N1024` | evoxc-auto | 4.83x | 6.07x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 4.81x | 6.05x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 4.89x | 6.15x |
| `p1000-L64-i32-N1024` | evoxc-auto | 4.53x | 5.42x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 4.39x | 5.25x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 4.47x | 5.34x |
| `p16-L64-i32-N64` | evoxc-auto | 0.48x | 0.87x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.48x | 0.86x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.56x | 1.00x |
| `p64-L64-i32-N64` | evoxc-auto | 0.96x | 1.40x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 0.96x | 1.41x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.21x | 0.30x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 0.93x | 1.35x |
| `p256-L64-i32-N64` | evoxc-auto | 3.98x | 4.78x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 3.74x | 4.50x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 3.64x | 4.37x |
| `p1000-L64-i32-N64` | evoxc-auto | 6.69x | 7.59x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6.82x | 7.74x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6.90x | 7.84x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p16-L32-i32-N1024` | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 |
| `p16-L64-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p16-L64-i32-N1024` | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 |
| `p64-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p64-L32-i32-N1024` | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 |
| `p64-L64-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p64-L64-i32-N1024` | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 |
| `p256-L32-i8-N1024` | -0.084110 | -0.084110 | -0.084110 | -0.084110 | -0.084110 | - | -0.084110 |
| `p256-L32-i32-N1024` | -0.084197 | -0.084197 | -0.084197 | -0.084197 | -0.084197 | - | -0.084197 |
| `p256-L64-i8-N1024` | -0.084110 | -0.084110 | -0.084110 | -0.084110 | -0.084110 | - | -0.084110 |
| `p256-L64-i32-N1024` | -0.084197 | -0.084197 | -0.084197 | -0.084197 | -0.084197 | - | -0.084197 |
| `p1000-L32-i8-N1024` | -0.083998 | -0.083998 | -0.083998 | -0.083998 | -0.083998 | - | -0.083998 |
| `p1000-L32-i32-N1024` | -0.084052 | -0.084052 | -0.084052 | -0.084052 | -0.084052 | - | -0.084052 |
| `p1000-L64-i8-N1024` | -0.084126 | -0.084126 | -0.084126 | -0.084126 | -0.084126 | - | -0.084126 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 | - | -0.084145 |
| `p16-L64-i32-N64` | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 |
| `p64-L64-i32-N64` | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 |
| `p256-L64-i32-N64` | -0.090160 | -0.090160 | -0.090160 | -0.090160 | -0.090160 | - | -0.090160 |
| `p1000-L64-i32-N64` | -0.089923 | -0.089923 | -0.089923 | -0.089923 | -0.089923 | - | -0.089923 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 86.789 | 13 | 2.468 |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 3665.534 | 13 | 3.864 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 86.399 | 13 | 2.459 |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 3728.177 | 13 | 3.978 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3368.401 | 1 | 0.974 |
| `p16-L32-i32-N1024` | evoxc-auto | 86.576 | 13 | 2.476 |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 3721.258 | 13 | 4.060 |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 86.082 | 13 | 2.567 |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 3719.149 | 13 | 3.879 |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 3400.883 | 1 | 0.941 |
| `p16-L64-i8-N1024` | evoxc-auto | 86.468 | 13 | 2.299 |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 3654.375 | 13 | 3.911 |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 89.795 | 13 | 2.312 |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 3646.269 | 13 | 3.880 |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 6513.143 | 1 | 1.456 |
| `p16-L64-i32-N1024` | evoxc-auto | 84.929 | 13 | 2.627 |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 3725.044 | 13 | 3.937 |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 87.562 | 13 | 2.607 |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 3706.924 | 13 | 3.801 |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 6504.166 | 1 | 1.447 |
| `p64-L32-i8-N1024` | evoxc-auto | 3421.932 | 1 | 0.924 |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 3431.174 | 1 | 0.971 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 233.628 | 57 | 9.150 |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 15743.470 | 57 | 15.451 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3447.375 | 1 | 1.029 |
| `p64-L32-i32-N1024` | evoxc-auto | 3538.638 | 1 | 0.964 |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 3411.362 | 1 | 0.986 |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 236.877 | 57 | 9.969 |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 15939.004 | 57 | 15.522 |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 3603.999 | 1 | 0.950 |
| `p64-L64-i8-N1024` | evoxc-auto | 6810.890 | 1 | 1.530 |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6384.062 | 1 | 1.528 |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 241.528 | 57 | 10.183 |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 15724.385 | 57 | 15.640 |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6435.226 | 1 | 1.535 |
| `p64-L64-i32-N1024` | evoxc-auto | 6543.487 | 1 | 1.551 |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6493.631 | 1 | 1.651 |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 234.982 | 57 | 10.145 |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 15925.829 | 57 | 15.584 |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6629.346 | 1 | 1.514 |
| `p256-L32-i8-N1024` | evoxc-auto | 3417.929 | 1 | 1.101 |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3548.231 | 1 | 1.122 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 886.130 | 212 | 40.451 |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3394.107 | 1 | 1.103 |
| `p256-L32-i32-N1024` | evoxc-auto | 3376.986 | 1 | 1.133 |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3412.337 | 1 | 1.127 |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 840.713 | 219 | 39.304 |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3414.225 | 1 | 1.133 |
| `p256-L64-i8-N1024` | evoxc-auto | 6389.165 | 1 | 1.977 |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 6564.834 | 1 | 2.000 |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 848.893 | 212 | 38.968 |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 6523.489 | 1 | 1.999 |
| `p256-L64-i32-N1024` | evoxc-auto | 6499.446 | 1 | 2.046 |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6592.167 | 1 | 2.107 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 842.444 | 219 | 39.434 |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6575.615 | 1 | 2.160 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3449.154 | 1 | 1.903 |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 3481.104 | 1 | 1.914 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3336.236 | 797 | 152.270 |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3496.294 | 1 | 1.906 |
| `p1000-L32-i32-N1024` | evoxc-auto | 3356.472 | 1 | 1.974 |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 3498.341 | 1 | 1.987 |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 3336.237 | 821 | 155.060 |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 3532.528 | 1 | 2.031 |
| `p1000-L64-i8-N1024` | evoxc-auto | 6851.141 | 1 | 5.510 |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 6676.701 | 1 | 5.537 |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 3362.619 | 797 | 154.152 |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 6652.726 | 1 | 5.535 |
| `p1000-L64-i32-N1024` | evoxc-auto | 6586.623 | 1 | 5.665 |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 6554.845 | 1 | 5.668 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3359.268 | 821 | 155.934 |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6663.352 | 1 | 5.744 |
| `p16-L64-i32-N64` | evoxc-auto | 70.845 | 13 | 2.380 |
| `p16-L64-i32-N64` | evoxc-auto-iree | 3582.415 | 13 | 3.874 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 68.670 | 13 | 2.397 |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 3601.541 | 13 | 3.849 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6494.950 | 1 | 1.464 |
| `p64-L64-i32-N64` | evoxc-auto | 6821.652 | 1 | 1.435 |
| `p64-L64-i32-N64` | evoxc-auto-iree | 6571.364 | 1 | 1.441 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 211.561 | 57 | 9.532 |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 14984.056 | 57 | 15.502 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6450.334 | 1 | 1.425 |
| `p256-L64-i32-N64` | evoxc-auto | 6790.567 | 1 | 1.464 |
| `p256-L64-i32-N64` | evoxc-auto-iree | 7117.313 | 1 | 1.534 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 955.806 | 219 | 41.083 |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6701.077 | 1 | 1.462 |
| `p1000-L64-i32-N64` | evoxc-auto | 6919.628 | 1 | 1.582 |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6561.672 | 1 | 1.549 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3252.223 | 821 | 139.402 |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6735.171 | 1 | 1.571 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | batch_forward | `ba1157e91cb09e697e821bf4caea5ee7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i8-N1024` | batch_forward | `ccd9945407c53f1df82b795467c23061` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L32-i8-N1024` | forward | `c21e5a57d58db557165cc13d515f788f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `ae02cdc9437e3c169a407657a6cb212f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `7c0dcb871d22ca8776d532411a044898` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L32-i32-N1024` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `ba1157e91cb09e697e821bf4caea5ee7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `ccd9945407c53f1df82b795467c23061` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i8-N1024` | forward | `c21e5a57d58db557165cc13d515f788f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `ae02cdc9437e3c169a407657a6cb212f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `7c0dcb871d22ca8776d532411a044898` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i32-N1024` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `960a58af4aeb7d6a855d90660a333509` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `d1fd2ddbb2317ca93f683e184057c988` | evoxc-specialized-iree | 0 |
| `p64-L32-i8-N1024` | forward | `4cfeb51431d33041bc2c71f2db3b1417` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | forward | `b5bf83cf773664021af5c74b9252a82e` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `f15f0b65cc977adc7eaa17ae644d9ea5` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `d44d3e2c6ac92fe124599c5d5baa529d` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `960a58af4aeb7d6a855d90660a333509` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `d1fd2ddbb2317ca93f683e184057c988` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | forward | `4cfeb51431d33041bc2c71f2db3b1417` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i8-N1024` | forward | `b5bf83cf773664021af5c74b9252a82e` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `f15f0b65cc977adc7eaa17ae644d9ea5` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `d44d3e2c6ac92fe124599c5d5baa529d` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L32-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L32-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L64-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p1000-L32-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L32-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L32-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3072 |
| `p1000-L32-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L32-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |
| `p1000-L64-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L64-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L64-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L64-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |
| `p16-L64-i32-N64` | batch_forward | `ac20d6b9b2fc89394ed1f5bb092e351e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `3e8dc4e8aabd0c155e65667c48b43645` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i32-N64` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `91f8f057103e5fc3760df30ab1e15987` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `0f0607d7f450ff24c5af100c2d775332` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `5ddefa134a076abce9faa669870b2558` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N64` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p1000-L64-i32-N64` | batch_forward | `6891f91b5d52caf1d0ca046f766bf963` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 192 |
| `p1000-L64-i32-N64` | batch_forward | `b603b9032571ea2fc133e7985e03390e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 192 |
| `p1000-L64-i32-N64` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N64` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16-L32-i8-N1024: 5/7, p16-L32-i32-N1024: 5/7, p16-L64-i8-N1024: 5/7, p16-L64-i32-N1024: 5/7, p64-L32-i8-N1024: 5/7, p64-L32-i32-N1024: 5/7, p64-L64-i8-N1024: 5/7, p64-L64-i32-N1024: 5/7, p256-L32-i8-N1024: 4/6, p256-L32-i32-N1024: 4/6, p256-L64-i8-N1024: 4/6, p256-L64-i32-N1024: 4/6, p1000-L32-i8-N1024: 4/6, p1000-L32-i32-N1024: 4/6, p1000-L64-i8-N1024: 4/6, p1000-L64-i32-N1024: 4/6, p16-L64-i32-N64: 2/7, p64-L64-i32-N64: 3/7, p256-L64-i32-N64: 2/6, p1000-L64-i32-N64: 3/6.

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

<!-- ===== /tmp/final_before.md ===== -->

# evoxc benchmark summary: before

Source: `results/final/before` — **90 records**, 20 cells x 5 arms.

The campaign ran **90 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **2035.4 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **10 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-28T09:51:06Z |
| git SHA (`tree_root_sha`) | - |
| tree_root | `/tmp/evogp_preopt` |
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

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.075 (2.816) | 2.433 (2.417) | 2.986 (2.975) | 4.443 (4.432) | 2.372 (2.369) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 3.069 (3.023) | 2.510 (2.449) | 3.009 (2.989) | 4.493 (4.360) | 2.470 (2.405) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 3.071 (3.037) | 3.157 (3.149) | 3.085 (2.991) | 4.437 (4.363) | 2.915 (2.854) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 2.837 (2.762) | 3.008 (2.992) | 3.029 (2.910) | 4.541 (4.421) | 3.059 (2.971) | auto/numpy → specialized | evoxc-auto |
| `p64-L32-i8-N1024` | 8.345 (8.161) | 2.525 (2.501) | 8.780 (8.307) | 14.097 (13.602) | 2.558 (2.500) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 8.297 (8.057) | 2.576 (2.390) | 9.271 (9.169) | 14.304 (14.077) | 2.694 (2.642) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i8-N1024` | 9.206 (9.108) | 3.084 (2.974) | 9.156 (8.412) | 14.023 (13.504) | 3.195 (3.150) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 9.197 (8.265) | 3.195 (3.143) | 8.786 (8.407) | 15.235 (15.181) | 3.222 (3.199) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 39.101 (38.721) | 3.135 (3.089) | 36.910 (35.828) | - | 3.098 (3.040) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L32-i32-N1024` | 39.227 (38.565) | 3.720 (3.495) | 35.173 (34.058) | - | 3.156 (3.093) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 36.563 (34.751) | 4.333 (4.274) | 35.542 (34.900) | - | 4.274 (4.232) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 38.562 (36.927) | 4.307 (4.284) | 34.984 (34.256) | - | 4.412 (4.404) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 145.331 (142.177) | 5.570 (5.364) | 142.200 (141.238) | - | 5.462 (5.171) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L32-i32-N1024` | 137.102 (136.725) | 5.705 (5.370) | 146.244 (139.120) | - | 5.643 (5.508) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i8-N1024` | 134.556 (133.045) | 10.315 (10.275) | 139.225 (138.632) | - | 10.331 (10.301) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L64-i32-N1024` | 140.148 (138.369) | 10.560 (10.468) | 138.560 (134.581) | - | 10.512 (10.457) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N64` | 2.542 (2.522) | 2.866 (2.806) | 2.917 (2.889) | 4.300 (4.195) | 2.885 (2.812) | auto/numpy → specialized | evoxc-auto |
| `p64-L64-i32-N64` | 8.695 (7.948) | 3.017 (2.988) | 8.175 (7.820) | 13.999 (13.944) | 3.025 (2.999) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L64-i32-N64` | 32.487 (32.209) | 3.757 (3.723) | 33.064 (32.997) | - | 3.787 (3.749) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 134.837 (132.122) | 6.370 (6.340) | 133.487 (131.363) | - | 6.365 (6.341) | auto/numpy → specialized | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | - | - |
| `p16-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-auto | - | - |
| `p16-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-auto | - | - |
| `p16-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-auto | - | - |
| `p16-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-auto | - | - |
| `p64-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-auto | - | - |
| `p64-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-auto | - | - |
| `p64-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-auto | - | - |
| `p64-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-auto | - | - |
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-auto | - | - |
| `p256-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-auto | - | - |
| `p256-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N64` | evoxc-auto | - | - |
| `p16-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N64` | evoxc-auto | - | - |
| `p64-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N64` | evoxc-auto | - | - |
| `p256-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-auto | - | - |
| `p1000-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | - | - |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.065 (2.728) | 2.256 (2.127) | 2.912 (2.875) | 4.308 (4.276) | 2.290 (2.275) | auto/numpy → specialized | evoxc-auto-iree |
| `p16-L32-i32-N1024` | 2.958 (2.929) | 2.272 (2.251) | 2.910 (2.883) | 4.372 (4.314) | 2.320 (2.286) | auto/numpy → specialized | evoxc-auto-iree |
| `p16-L64-i8-N1024` | 2.897 (2.830) | 2.761 (2.746) | 2.975 (2.618) | 4.338 (4.217) | 2.788 (2.736) | auto/numpy → specialized | evoxc-auto-iree |
| `p16-L64-i32-N1024` | 3.009 (2.735) | 2.807 (2.787) | 2.743 (2.673) | 4.643 (4.597) | 2.834 (2.691) | auto/numpy → specialized | evoxc-specialized-numpy |
| `p64-L32-i8-N1024` | 8.223 (8.109) | 2.636 (2.472) | 8.136 (8.089) | 13.506 (13.397) | 2.551 (2.468) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i32-N1024` | 8.218 (8.066) | 2.549 (2.501) | 9.451 (9.436) | 14.085 (13.741) | 2.709 (2.656) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i8-N1024` | 8.261 (8.162) | 3.107 (3.075) | 8.794 (8.340) | 13.540 (13.234) | 3.132 (3.013) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 8.323 (8.266) | 3.192 (3.077) | 9.261 (9.103) | 15.004 (14.477) | 3.254 (3.158) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 37.253 (36.254) | 3.491 (3.431) | 37.435 (36.079) | - | 3.475 (3.414) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L32-i32-N1024` | 39.282 (39.197) | 3.499 (3.387) | 37.382 (34.464) | - | 3.515 (3.397) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 34.977 (34.419) | 4.900 (4.681) | 35.396 (34.156) | - | 4.670 (4.562) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 38.278 (35.866) | 4.866 (4.797) | 34.128 (33.884) | - | 4.965 (4.941) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 143.785 (141.326) | 6.985 (6.535) | 141.837 (140.117) | - | 6.824 (6.690) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L32-i32-N1024` | 142.233 (140.017) | 7.438 (7.323) | 139.509 (134.888) | - | 6.947 (6.720) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i8-N1024` | 135.858 (133.230) | 11.985 (11.935) | 141.645 (136.951) | - | 11.915 (11.418) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 141.420 (137.746) | 12.251 (12.087) | 137.739 (133.943) | - | 12.015 (11.992) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N64` | 2.811 (2.700) | 2.608 (2.596) | 2.612 (2.530) | 4.149 (4.095) | 2.683 (2.661) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N64` | 8.572 (7.916) | 2.830 (2.816) | 8.085 (7.754) | 14.434 (13.853) | 2.768 (2.754) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N64` | 35.480 (34.767) | 3.638 (3.446) | 35.477 (33.756) | - | 3.641 (3.578) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 140.923 (135.872) | 6.268 (5.864) | 136.977 (134.929) | - | 6.303 (6.286) | auto/numpy → specialized | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | - | - |
| `p16-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-auto | - | - |
| `p16-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-auto | - | - |
| `p16-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-auto | - | - |
| `p16-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-auto | - | - |
| `p64-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-auto | - | - |
| `p64-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-auto | - | - |
| `p64-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-auto | - | - |
| `p64-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-auto | - | - |
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-auto | - | - |
| `p256-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-auto | - | - |
| `p256-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N64` | evoxc-auto | - | - |
| `p16-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N64` | evoxc-auto | - | - |
| `p64-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N64` | evoxc-auto | - | - |
| `p256-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-auto | - | - |
| `p1000-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | - | - |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 2.803 (2.786) | 1.878 (1.858) | 2.753 (2.714) | 5.768 (5.666) | 1.863 (1.746) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 2.724 (2.661) | 1.833 (1.820) | 2.703 (2.644) | 5.768 (5.715) | 1.840 (1.833) | auto/numpy → specialized | evoxc-auto-iree |
| `p16-L64-i8-N1024` | 2.817 (2.805) | 2.155 (2.113) | 2.697 (2.440) | 5.772 (5.649) | 2.070 (1.928) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 2.505 (2.503) | 2.113 (2.084) | 2.664 (2.468) | 5.951 (5.877) | 2.117 (2.099) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L32-i8-N1024` | 7.818 (7.673) | 2.044 (2.025) | 7.620 (7.603) | 19.455 (19.220) | 1.911 (1.863) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i32-N1024` | 7.548 (7.513) | 1.940 (1.911) | 8.384 (7.679) | 19.289 (19.003) | 2.091 (2.072) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i8-N1024` | 8.338 (7.612) | 2.243 (2.185) | 8.409 (7.821) | 19.817 (19.398) | 2.284 (2.245) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 8.437 (7.598) | 2.226 (2.221) | 8.685 (8.520) | 20.635 (20.105) | 2.288 (2.259) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i8-N1024` | 33.745 (31.807) | 2.443 (2.425) | 33.959 (32.588) | - | 2.448 (2.390) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L32-i32-N1024` | 33.566 (32.042) | 2.444 (2.435) | 32.587 (31.693) | - | 2.440 (2.426) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 32.354 (31.400) | 2.947 (2.772) | 32.971 (32.574) | - | 2.678 (2.648) | auto/numpy → specialized | evoxc-dense-iree |
| `p256-L64-i32-N1024` | 34.929 (33.895) | 2.982 (2.902) | 33.395 (32.762) | - | 2.927 (2.744) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L32-i8-N1024` | 133.363 (131.881) | 4.198 (4.180) | 133.603 (129.283) | - | 3.808 (3.744) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L32-i32-N1024` | 127.613 (125.183) | 4.326 (4.219) | 128.608 (124.956) | - | 4.157 (3.776) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i8-N1024` | 128.091 (126.673) | 5.526 (4.938) | 126.383 (125.832) | - | 5.509 (5.486) | auto/numpy → specialized | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 139.214 (129.416) | 5.475 (5.415) | 131.223 (128.666) | - | 5.503 (5.457) | auto/numpy → specialized | evoxc-auto-iree |
| `p16-L64-i32-N64` | 2.456 (2.430) | 2.097 (2.060) | 2.737 (2.473) | 6.211 (6.130) | 2.098 (2.047) | auto/numpy → specialized | evoxc-auto-iree |
| `p64-L64-i32-N64` | 8.317 (7.577) | 2.233 (2.209) | 8.195 (7.567) | 23.170 (22.541) | 2.233 (2.219) | auto/numpy → specialized | evoxc-auto-iree |
| `p256-L64-i32-N64` | 34.710 (33.296) | 2.967 (2.948) | 33.385 (31.477) | - | 2.991 (2.953) | auto/numpy → specialized | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 137.384 (135.275) | 5.511 (5.501) | 135.699 (133.864) | - | 5.542 (5.509) | auto/numpy → specialized | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | - | - |
| `p16-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-auto | - | - |
| `p16-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-auto | - | - |
| `p16-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-auto | - | - |
| `p16-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-auto | - | - |
| `p64-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-auto | - | - |
| `p64-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-auto | - | - |
| `p64-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-auto | - | - |
| `p64-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-auto | - | - |
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-auto | - | - |
| `p256-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-auto | - | - |
| `p256-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N64` | evoxc-auto | - | - |
| `p16-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N64` | evoxc-auto | - | - |
| `p64-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N64` | evoxc-auto | - | - |
| `p256-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-auto | - | - |
| `p1000-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | - | - |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 11.323 | 9.017 | 11.384 | 538.274 | 9.186 |
| `p16-L32-i32-N1024` | 59.553 | 8.570 | 57.202 | 3599.738 | 8.485 |
| `p16-L64-i8-N1024` | 10.731 | 9.630 | 11.691 | 553.137 | 9.561 |
| `p16-L64-i32-N1024` | 63.520 | 9.068 | 62.687 | 4038.921 | 9.264 |
| `p64-L32-i8-N1024` | 35.842 | 8.534 | 38.323 | 4650.815 | 8.605 |
| `p64-L32-i32-N1024` | 51.788 | 8.472 | 52.414 | 6656.242 | 8.778 |
| `p64-L64-i8-N1024` | 38.902 | 9.362 | 38.266 | 4650.264 | 9.159 |
| `p64-L64-i32-N1024` | 51.110 | 9.078 | 51.819 | 6116.434 | 8.992 |
| `p256-L32-i8-N1024` | 197.492 | 9.015 | 188.362 | - | 9.151 |
| `p256-L32-i32-N1024` | 172.240 | 9.184 | 163.641 | - | 9.073 |
| `p256-L64-i8-N1024` | 181.190 | 10.306 | 188.105 | - | 10.234 |
| `p256-L64-i32-N1024` | 169.110 | 10.440 | 161.254 | - | 10.333 |
| `p1000-L32-i8-N1024` | 901.434 | 11.190 | 900.803 | - | 11.140 |
| `p1000-L32-i32-N1024` | 811.149 | 11.257 | 823.100 | - | 10.623 |
| `p1000-L64-i8-N1024` | 900.975 | 15.981 | 914.087 | - | 16.130 |
| `p1000-L64-i32-N1024` | 803.111 | 15.998 | 810.350 | - | 16.377 |
| `p16-L64-i32-N64` | 17.453 | 9.330 | 19.933 | 1773.032 | 9.592 |
| `p64-L64-i32-N64` | 95.076 | 8.551 | 96.944 | 9260.709 | 9.034 |
| `p256-L64-i32-N64` | 240.900 | 9.701 | 233.051 | - | 9.813 |
| `p1000-L64-i32-N64` | 737.962 | 11.736 | 723.082 | - | 12.051 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 5.599 / 5.724 (op 51%) | 3.094 / 5.924 (op 66%) | 5.623 / 5.761 (op 51%) | 532.080 / 6.195 (op 1%) | 3.099 / 6.087 (op 66%) |
| `p16-L32-i32-N1024` | 52.962 / 6.591 (op 11%) | 2.522 / 6.048 (op 71%) | 51.462 / 5.740 (op 10%) | 3593.315 / 6.423 (op 0%) | 2.526 / 5.959 (op 70%) |
| `p16-L64-i8-N1024` | 5.315 / 5.416 (op 50%) | 3.575 / 6.055 (op 63%) | 5.718 / 5.973 (op 51%) | 546.482 / 6.655 (op 1%) | 3.594 / 5.967 (op 62%) |
| `p16-L64-i32-N1024` | 57.938 / 5.582 (op 9%) | 3.058 / 6.010 (op 66%) | 56.709 / 5.978 (op 10%) | 4032.134 / 6.787 (op 0%) | 3.062 / 6.201 (op 67%) |
| `p64-L32-i8-N1024` | 30.231 / 5.611 (op 16%) | 2.576 / 5.958 (op 70%) | 32.340 / 5.983 (op 16%) | 4643.991 / 6.824 (op 0%) | 2.753 / 5.852 (op 68%) |
| `p64-L32-i32-N1024` | 46.429 / 5.359 (op 10%) | 2.611 / 5.861 (op 69%) | 46.928 / 5.485 (op 10%) | 6649.182 / 7.060 (op 0%) | 2.831 / 5.947 (op 68%) |
| `p64-L64-i8-N1024` | 33.269 / 5.633 (op 14%) | 3.219 / 6.143 (op 66%) | 32.511 / 5.755 (op 15%) | 4643.280 / 6.984 (op 0%) | 3.254 / 5.905 (op 64%) |
| `p64-L64-i32-N1024` | 45.520 / 5.590 (op 11%) | 3.250 / 5.827 (op 64%) | 46.089 / 5.729 (op 11%) | 6109.188 / 7.247 (op 0%) | 3.200 / 5.791 (op 64%) |
| `p256-L32-i8-N1024` | 191.450 / 6.042 (op 3%) | 3.010 / 6.005 (op 67%) | 182.599 / 5.763 (op 3%) | - | 3.016 / 6.135 (op 67%) |
| `p256-L32-i32-N1024` | 166.491 / 5.749 (op 3%) | 3.023 / 6.161 (op 67%) | 157.933 / 5.708 (op 3%) | - | 3.006 / 6.067 (op 67%) |
| `p256-L64-i8-N1024` | 175.420 / 5.770 (op 3%) | 4.238 / 6.068 (op 59%) | 181.968 / 6.138 (op 3%) | - | 4.268 / 5.966 (op 58%) |
| `p256-L64-i32-N1024` | 163.294 / 5.816 (op 3%) | 4.389 / 6.051 (op 58%) | 155.529 / 5.725 (op 4%) | - | 4.250 / 6.083 (op 59%) |
| `p1000-L32-i8-N1024` | 894.602 / 6.833 (op 1%) | 5.097 / 6.093 (op 54%) | 894.896 / 5.907 (op 1%) | - | 5.132 / 6.007 (op 54%) |
| `p1000-L32-i32-N1024` | 804.909 / 6.240 (op 1%) | 5.092 / 6.165 (op 55%) | 816.731 / 6.370 (op 1%) | - | 5.027 / 5.596 (op 53%) |
| `p1000-L64-i8-N1024` | 894.861 / 6.114 (op 1%) | 9.851 / 6.130 (op 38%) | 907.608 / 6.479 (op 1%) | - | 10.021 / 6.109 (op 38%) |
| `p1000-L64-i32-N1024` | 797.102 / 6.008 (op 1%) | 10.063 / 5.935 (op 37%) | 803.959 / 6.392 (op 1%) | - | 10.320 / 6.057 (op 37%) |
| `p16-L64-i32-N64` | 11.881 / 5.572 (op 32%) | 3.400 / 5.930 (op 64%) | 14.008 / 5.925 (op 30%) | 1766.515 / 6.517 (op 0%) | 3.591 / 6.000 (op 63%) |
| `p64-L64-i32-N64` | 89.727 / 5.349 (op 6%) | 2.896 / 5.655 (op 66%) | 91.408 / 5.536 (op 6%) | 9252.776 / 7.933 (op 0%) | 3.113 / 5.921 (op 66%) |
| `p256-L64-i32-N64` | 234.806 / 6.094 (op 3%) | 3.667 / 6.034 (op 62%) | 226.944 / 6.107 (op 3%) | - | 3.619 / 6.194 (op 63%) |
| `p1000-L64-i32-N64` | 731.956 / 6.006 (op 1%) | 5.831 / 5.905 (op 50%) | 716.992 / 6.090 (op 1%) | - | 6.016 / 6.034 (op 50%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | - | - |
| `p16-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-auto | - | - |
| `p16-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-auto | - | - |
| `p16-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-auto | - | - |
| `p16-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-auto | - | - |
| `p64-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-auto | - | - |
| `p64-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-auto | - | - |
| `p64-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-auto | - | - |
| `p64-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-auto | - | - |
| `p256-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-auto | - | - |
| `p256-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-auto | - | - |
| `p256-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-auto | - | - |
| `p256-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto | - | - |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto | - | - |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto | - | - |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto | - | - |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | - | - |
| `p16-L64-i32-N64` | evoxc-auto | - | - |
| `p16-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p16-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p16-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p64-L64-i32-N64` | evoxc-auto | - | - |
| `p64-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p64-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p64-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p256-L64-i32-N64` | evoxc-auto | - | - |
| `p256-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-auto | - | - |
| `p1000-L64-i32-N64` | evoxc-auto-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | - | - |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | - | - |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | -0.158556 | -0.158556 | -0.158556 | -0.158556 | -0.158556 |
| `p16-L32-i32-N1024` | -0.084134 | -0.084134 | -0.084134 | -0.084134 | -0.084134 |
| `p16-L64-i8-N1024` | -0.158556 | -0.158556 | -0.158556 | -0.158556 | -0.158556 |
| `p16-L64-i32-N1024` | -0.085032 | -0.085032 | -0.085032 | -0.085032 | -0.085032 |
| `p64-L32-i8-N1024` | -0.107462 | -0.107462 | -0.107462 | -0.107462 | -0.107462 |
| `p64-L32-i32-N1024` | -0.095727 | -0.095727 | -0.095727 | -0.095727 | -0.095727 |
| `p64-L64-i8-N1024` | -0.107462 | -0.107462 | -0.107462 | -0.107462 | -0.107462 |
| `p64-L64-i32-N1024` | -0.120318 | -0.120318 | -0.120318 | -0.120318 | -0.120318 |
| `p256-L32-i8-N1024` | -0.081483 | -0.081483 | -0.081483 | - | -0.081483 |
| `p256-L32-i32-N1024` | -0.081483 | -0.081483 | -0.081483 | - | -0.081483 |
| `p256-L64-i8-N1024` | -0.081483 | -0.081483 | -0.081483 | - | -0.081483 |
| `p256-L64-i32-N1024` | -0.081483 | -0.081483 | -0.081483 | - | -0.081483 |
| `p1000-L32-i8-N1024` | -0.081483 | -0.081483 | -0.081483 | - | -0.081483 |
| `p1000-L32-i32-N1024` | -0.081364 | -0.081364 | -0.081364 | - | -0.081364 |
| `p1000-L64-i8-N1024` | -0.081395 | -0.081395 | -0.081395 | - | -0.081395 |
| `p1000-L64-i32-N1024` | -0.081364 | -0.081364 | -0.081364 | - | -0.081364 |
| `p16-L64-i32-N64` | -0.118324 | -0.118324 | -0.118324 | -0.118324 | -0.118324 |
| `p64-L64-i32-N64` | -0.093033 | -0.093033 | -0.093033 | -0.093033 | -0.093033 |
| `p256-L64-i32-N64` | -0.075324 | -0.075324 | -0.075324 | - | -0.075324 |
| `p1000-L64-i32-N64` | -0.069370 | -0.069370 | -0.069370 | - | -0.069370 |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 99.908 | 15 | 3.075 |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 3452.825 | 1 | 2.433 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 99.975 | 15 | 2.986 |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 4372.522 | 15 | 4.443 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3367.160 | 1 | 2.372 |
| `p16-L32-i32-N1024` | evoxc-auto | 99.120 | 15 | 3.069 |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 3451.502 | 1 | 2.510 |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 99.108 | 15 | 3.009 |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 4309.895 | 15 | 4.493 |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 3401.757 | 1 | 2.470 |
| `p16-L64-i8-N1024` | evoxc-auto | 98.364 | 15 | 3.071 |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 6418.249 | 1 | 3.157 |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 98.437 | 15 | 3.085 |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 4273.742 | 15 | 4.437 |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 6428.367 | 1 | 2.915 |
| `p16-L64-i32-N1024` | evoxc-auto | 99.328 | 15 | 2.837 |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 6378.106 | 1 | 3.008 |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 95.679 | 15 | 3.029 |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 4280.320 | 15 | 4.541 |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 6390.782 | 1 | 3.059 |
| `p64-L32-i8-N1024` | evoxc-auto | 204.367 | 49 | 8.345 |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 3409.903 | 1 | 2.525 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 207.015 | 49 | 8.780 |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 13160.652 | 49 | 14.097 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3363.236 | 1 | 2.558 |
| `p64-L32-i32-N1024` | evoxc-auto | 207.652 | 51 | 8.297 |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 3375.677 | 1 | 2.576 |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 213.911 | 51 | 9.271 |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 13949.436 | 51 | 14.304 |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 3349.777 | 1 | 2.694 |
| `p64-L64-i8-N1024` | evoxc-auto | 207.482 | 49 | 9.206 |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6428.387 | 1 | 3.084 |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 211.025 | 49 | 9.156 |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 13268.041 | 49 | 14.023 |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6485.297 | 1 | 3.195 |
| `p64-L64-i32-N1024` | evoxc-auto | 219.960 | 51 | 9.197 |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6422.879 | 1 | 3.195 |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 214.167 | 51 | 8.786 |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 13874.390 | 51 | 15.235 |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6414.506 | 1 | 3.222 |
| `p256-L32-i8-N1024` | evoxc-auto | 818.462 | 205 | 39.101 |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3284.044 | 1 | 3.135 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 798.278 | 205 | 36.910 |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3315.825 | 1 | 3.098 |
| `p256-L32-i32-N1024` | evoxc-auto | 813.512 | 218 | 39.227 |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3315.885 | 1 | 3.720 |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 802.761 | 218 | 35.173 |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3317.546 | 1 | 3.156 |
| `p256-L64-i8-N1024` | evoxc-auto | 775.041 | 205 | 36.563 |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 6414.116 | 1 | 4.333 |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 783.820 | 205 | 35.542 |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 6317.069 | 1 | 4.274 |
| `p256-L64-i32-N1024` | evoxc-auto | 809.502 | 218 | 38.562 |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6424.452 | 1 | 4.307 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 776.525 | 218 | 34.984 |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6409.836 | 1 | 4.412 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3138.202 | 794 | 145.331 |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 3328.317 | 1 | 5.570 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3051.597 | 794 | 142.200 |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3237.969 | 1 | 5.462 |
| `p1000-L32-i32-N1024` | evoxc-auto | 3088.165 | 818 | 137.102 |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 3285.077 | 1 | 5.705 |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 3071.981 | 818 | 146.244 |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 3331.462 | 1 | 5.643 |
| `p1000-L64-i8-N1024` | evoxc-auto | 3014.209 | 794 | 134.556 |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 6368.211 | 1 | 10.315 |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 3060.444 | 794 | 139.225 |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 6397.720 | 1 | 10.331 |
| `p1000-L64-i32-N1024` | evoxc-auto | 3168.754 | 818 | 140.148 |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 6355.015 | 1 | 10.560 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3087.837 | 818 | 138.560 |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6429.332 | 1 | 10.512 |
| `p16-L64-i32-N64` | evoxc-auto | 74.989 | 15 | 2.542 |
| `p16-L64-i32-N64` | evoxc-auto-iree | 6316.962 | 1 | 2.866 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 79.644 | 15 | 2.917 |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 4088.641 | 15 | 4.300 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6237.389 | 1 | 2.885 |
| `p64-L64-i32-N64` | evoxc-auto | 185.433 | 51 | 8.695 |
| `p64-L64-i32-N64` | evoxc-auto-iree | 6271.202 | 1 | 3.017 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 192.655 | 51 | 8.175 |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 13388.504 | 51 | 13.999 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6335.801 | 1 | 3.025 |
| `p256-L64-i32-N64` | evoxc-auto | 773.648 | 218 | 32.487 |
| `p256-L64-i32-N64` | evoxc-auto-iree | 6331.432 | 1 | 3.757 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 804.296 | 218 | 33.064 |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6259.309 | 1 | 3.787 |
| `p1000-L64-i32-N64` | evoxc-auto | 3044.249 | 818 | 134.837 |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6319.137 | 1 | 6.370 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3066.136 | 818 | 133.487 |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6306.006 | 1 | 6.365 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | batch_forward | `4ff7fdc86805329f470069a6c3b8aae7` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p16-L32-i8-N1024` | batch_forward | `716e171a996460f855ad15ee526bcee0` | evoxc-specialized-iree | 1024 |
| `p16-L32-i8-N1024` | forward | `209a9a23bc848b0b78e9e120d57e1999` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 1 |
| `p16-L32-i32-N1024` | batch_forward | `5701430a0b9cf57930e15782e90c2ecd` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `529cc77cb1440dd19731e020a7e8607e` | evoxc-specialized-iree | 0 |
| `p16-L32-i32-N1024` | forward | `35463baf97c5a460b0eea39d86707e97` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `4ff7fdc86805329f470069a6c3b8aae7` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p16-L64-i8-N1024` | batch_forward | `716e171a996460f855ad15ee526bcee0` | evoxc-specialized-iree | 1024 |
| `p16-L64-i8-N1024` | forward | `209a9a23bc848b0b78e9e120d57e1999` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 1 |
| `p16-L64-i32-N1024` | batch_forward | `5701430a0b9cf57930e15782e90c2ecd` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `529cc77cb1440dd19731e020a7e8607e` | evoxc-specialized-iree | 0 |
| `p16-L64-i32-N1024` | forward | `35463baf97c5a460b0eea39d86707e97` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `6e8fa35cc0377bd67505cac8c936efe4` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2048 |
| `p64-L32-i8-N1024` | batch_forward | `e12861e3d646a4318c003656339e8c55` | evoxc-specialized-iree | 2048 |
| `p64-L32-i8-N1024` | forward | `37fd68ce45259cb61c5e5c6c59a41c3b` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2 |
| `p64-L32-i8-N1024` | forward | `679d1260a48271b7f9eaaeafce90612c` | evoxc-specialized-iree | 2 |
| `p64-L32-i32-N1024` | batch_forward | `010cdcefb94d96563220c49a272319cc` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `6ca09b1eb5e161ef3eb6176ab2d48cf2` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `6e8fa35cc0377bd67505cac8c936efe4` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2048 |
| `p64-L64-i8-N1024` | batch_forward | `e12861e3d646a4318c003656339e8c55` | evoxc-specialized-iree | 2048 |
| `p64-L64-i8-N1024` | forward | `37fd68ce45259cb61c5e5c6c59a41c3b` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 2 |
| `p64-L64-i8-N1024` | forward | `679d1260a48271b7f9eaaeafce90612c` | evoxc-specialized-iree | 2 |
| `p64-L64-i32-N1024` | batch_forward | `010cdcefb94d96563220c49a272319cc` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `6ca09b1eb5e161ef3eb6176ab2d48cf2` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p256-L32-i8-N1024` | batch_forward | `31b2be904b92ced0b65c329a42e6bc0f` | evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p256-L32-i8-N1024` | batch_forward | `5a456df1db65603131c4eedcb038aecd` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p256-L32-i8-N1024` | forward | `d03896a32ebdeb367ae5e7367d0b71d8` | evoxc-auto, evoxc-specialized-numpy | 4 |
| `p256-L32-i8-N1024` | forward | `b69d807be015515d9b6afbc07da541df` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p256-L32-i32-N1024` | batch_forward | `5f3928f9d3a6f36239238e77eb7df793` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i32-N1024` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L64-i8-N1024` | batch_forward | `31b2be904b92ced0b65c329a42e6bc0f` | evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p256-L64-i8-N1024` | batch_forward | `5a456df1db65603131c4eedcb038aecd` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p256-L64-i8-N1024` | forward | `d03896a32ebdeb367ae5e7367d0b71d8` | evoxc-auto, evoxc-specialized-numpy | 4 |
| `p256-L64-i8-N1024` | forward | `b69d807be015515d9b6afbc07da541df` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p256-L64-i32-N1024` | batch_forward | `5f3928f9d3a6f36239238e77eb7df793` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L64-i32-N1024` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p1000-L32-i8-N1024` | batch_forward | `9eb4419b453774b79662769c25081b77` | evoxc-auto, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `731f6b21d50248d0c37375aa59cfd8ee` | evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `f559696c3212f4ba1df982b656adcdeb` | evoxc-auto, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `f5dfcf8df1ca7dc89cc350aa57ed2d9f` | evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L32-i32-N1024` | batch_forward | `908b13d758ba2baec03b855ecc38ca4d` | evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p1000-L32-i32-N1024` | batch_forward | `7832ebc9f0d4b673c2df28da8307164e` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p1000-L32-i32-N1024` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L32-i32-N1024` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p1000-L64-i8-N1024` | batch_forward | `9eb4419b453774b79662769c25081b77` | evoxc-auto, evoxc-specialized-numpy | 7168 |
| `p1000-L64-i8-N1024` | batch_forward | `731f6b21d50248d0c37375aa59cfd8ee` | evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L64-i8-N1024` | forward | `f559696c3212f4ba1df982b656adcdeb` | evoxc-auto, evoxc-specialized-numpy | 7 |
| `p1000-L64-i8-N1024` | forward | `f5dfcf8df1ca7dc89cc350aa57ed2d9f` | evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `908b13d758ba2baec03b855ecc38ca4d` | evoxc-auto, evoxc-specialized-numpy | 4096 |
| `p1000-L64-i32-N1024` | batch_forward | `7832ebc9f0d4b673c2df28da8307164e` | evoxc-auto-iree, evoxc-dense-iree | 4096 |
| `p1000-L64-i32-N1024` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L64-i32-N1024` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |
| `p16-L64-i32-N64` | batch_forward | `d86a2a4fea14cd7b737acd298647f327` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `927214dc4b0de0adcb0c6e4cc16c67b3` | evoxc-specialized-iree | 0 |
| `p16-L64-i32-N64` | forward | `35463baf97c5a460b0eea39d86707e97` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `fc6db4d28c1f58fe2d56360f2cfa48ca` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `4626235a1903869537361121a5eaf562` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | forward | `ed4babe724121fbb8737b4eb1e4581f2` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | forward | `f88586d051c9010bb3e8a0e3f881a15c` | evoxc-specialized-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `a8acb34521b847b6456ced8703a21e2d` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 64 |
| `p256-L64-i32-N64` | forward | `6fc0dcf69b4aab65659794279d07b4dd` | evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p1000-L64-i32-N64` | batch_forward | `f5642f4213ebbe3510de1c9647b2dd33` | evoxc-auto, evoxc-specialized-numpy | 256 |
| `p1000-L64-i32-N64` | batch_forward | `dec2621dfeed4fac25b576ec4c346532` | evoxc-auto-iree, evoxc-dense-iree | 256 |
| `p1000-L64-i32-N64` | forward | `7304a80dd5f165e3b5b5cf6e0e12bd3c` | evoxc-auto, evoxc-specialized-numpy | 4 |
| `p1000-L64-i32-N64` | forward | `fbeb0cf53f52e1c004ca2e62e578c34e` | evoxc-auto-iree, evoxc-dense-iree | 4 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16-L32-i8-N1024: 2/5, p16-L32-i32-N1024: 3/5, p16-L64-i8-N1024: 2/5, p16-L64-i32-N1024: 3/5, p64-L32-i8-N1024: 3/5, p64-L32-i32-N1024: 3/5, p64-L64-i8-N1024: 3/5, p64-L64-i32-N1024: 3/5, p256-L32-i8-N1024: 2/4, p256-L32-i32-N1024: 2/4, p256-L64-i8-N1024: 2/4, p256-L64-i32-N1024: 2/4, p1000-L32-i8-N1024: 2/4, p1000-L32-i32-N1024: 2/4, p1000-L64-i8-N1024: 2/4, p1000-L64-i32-N1024: 2/4, p16-L64-i32-N64: 2/5, p64-L64-i32-N64: 3/5, p256-L64-i32-N64: 2/4, p1000-L64-i32-N64: 2/4.

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

<!-- ===== /tmp/final_compare.md ===== -->

# evoxc benchmark summary: after

Source: `results/final/after` — **130 records**, 20 cells x 7 arms.

The campaign ran **130 configs** sequenced one at a time (`one_process_per_config: True`, `sequential: True`): every (cell, arm) pair gets its own process, so each record pays its own cold compile and inherits no other arm's compile cache.  Elapsed **2478.5 s**, failures **0**.
Per workload one cold first call is timed and reported separately (never part of the medians), followed by `warmup=2` untimed calls and `reps=5` sync-bracketed samples; the GP generation step used `gp_warmup=1` + `gp_reps=5`; seed `0`, workloads `sr_fitness, batch_forward, forward, gp_step`, guard wait `120.0 s`, scope rule `evoxc-specialized-iree` on pops [16, 64] — **10 config(s) dropped** from the schedule (`specialized` iree measured at 271.6–283.9 ms per tree structure, so a fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are listed under `## Caveats`).

## Environment

| key | value |
|---|---|
| created_utc | 2026-10-28T09:09:41Z |
| git SHA (`tree_root_sha`) | 761ef05ce87d456c364cbb706fa234d9b184ecdd |
| tree_root | `/root/evogp/.genesis/workers/worker_T1_A50/src` |
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
| `p16-L32-i8-N1024` | 1.844 (1.737) | 2.779 (2.747) | 2.468 (2.254) | 3.864 (3.843) | 2.459 (2.428) | 3.978 (3.812) | 0.974 (0.956) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.962 (1.950) | 2.848 (2.757) | 2.476 (2.431) | 4.060 (3.899) | 2.567 (2.526) | 3.879 (3.827) | 0.941 (0.931) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.858 (1.829) | 3.240 (2.930) | 2.299 (2.265) | 3.911 (3.835) | 2.312 (2.244) | 3.880 (3.797) | 1.456 (1.433) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 1.922 (1.737) | 3.242 (2.974) | 2.627 (2.391) | 3.937 (3.884) | 2.607 (2.526) | 3.801 (3.792) | 1.447 (1.414) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 7.476 (7.152) | 3.572 (3.547) | 0.924 (0.912) | 0.971 (0.957) | 9.150 (9.001) | 15.451 (15.065) | 1.029 (0.991) | auto/numpy → dense | evoxc-auto |
| `p64-L32-i32-N1024` | 7.433 (7.362) | 3.607 (3.319) | 0.964 (0.932) | 0.986 (0.960) | 9.969 (9.310) | 15.522 (15.470) | 0.950 (0.933) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 7.843 (7.505) | 4.038 (4.005) | 1.530 (1.509) | 1.528 (1.511) | 10.183 (10.046) | 15.640 (15.486) | 1.535 (1.526) | auto/numpy → dense | evoxc-auto-iree |
| `p64-L64-i32-N1024` | 7.664 (7.588) | 4.109 (4.074) | 1.551 (1.524) | 1.651 (1.574) | 10.145 (10.038) | 15.584 (15.372) | 1.514 (1.494) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L32-i8-N1024` | 30.177 (30.075) | 4.002 (3.732) | 1.101 (1.086) | 1.122 (1.068) | 40.451 (39.987) | - | 1.103 (1.090) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 30.277 (30.143) | 3.999 (3.958) | 1.133 (1.089) | 1.127 (1.100) | 39.304 (37.412) | - | 1.133 (1.117) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 30.483 (30.410) | 4.299 (4.160) | 1.977 (1.966) | 2.000 (1.974) | 38.968 (36.667) | - | 1.999 (1.971) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 30.502 (30.359) | 4.507 (4.421) | 2.046 (2.003) | 2.107 (2.094) | 39.434 (37.100) | - | 2.160 (2.122) | auto/numpy → dense | evoxc-auto |
| `p1000-L32-i8-N1024` | 120.749 (119.966) | 4.376 (4.367) | 1.903 (1.867) | 1.914 (1.910) | 152.270 (151.016) | - | 1.906 (1.864) | auto/numpy → dense | evoxc-auto |
| `p1000-L32-i32-N1024` | 123.636 (119.848) | 4.393 (4.377) | 1.974 (1.947) | 1.987 (1.968) | 155.060 (148.386) | - | 2.031 (2.006) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i8-N1024` | 120.707 (119.679) | 4.866 (4.833) | 5.510 (5.499) | 5.537 (5.529) | 154.152 (151.373) | - | 5.535 (5.508) | auto/numpy → dense | torch-batched |
| `p1000-L64-i32-N1024` | 119.694 (118.781) | 4.887 (4.872) | 5.665 (5.639) | 5.668 (5.647) | 155.934 (154.521) | - | 5.744 (5.702) | auto/numpy → dense | torch-batched |
| `p16-L64-i32-N64` | 1.842 (1.796) | 3.183 (3.153) | 2.380 (2.361) | 3.874 (3.817) | 2.397 (2.394) | 3.849 (3.838) | 1.464 (1.419) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 7.420 (7.370) | 4.119 (3.996) | 1.435 (1.424) | 1.441 (1.426) | 9.532 (9.483) | 15.502 (15.333) | 1.425 (1.391) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N64` | 29.796 (29.266) | 4.520 (4.307) | 1.464 (1.433) | 1.534 (1.486) | 41.083 (40.599) | - | 1.462 (1.429) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N64` | 118.946 (118.657) | 4.978 (4.954) | 1.582 (1.568) | 1.549 (1.539) | 139.402 (138.237) | - | 1.571 (1.533) | auto/numpy → dense | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (sr_fitness)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.75x | 1.13x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.48x | 0.72x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.75x | 1.13x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.46x | 0.70x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.89x | 2.85x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.79x | 1.15x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.48x | 0.70x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.76x | 1.11x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.51x | 0.73x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 2.08x | 3.03x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.81x | 1.41x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.48x | 0.83x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.80x | 1.40x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.48x | 0.83x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 1.28x | 2.23x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.73x | 1.23x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.49x | 0.82x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.74x | 1.24x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.51x | 0.85x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 1.33x | 2.24x |
| `p64-L32-i8-N1024` | evoxc-auto | 8.09x | 3.87x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 7.70x | 3.68x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.82x | 0.39x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.48x | 0.23x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 7.27x | 3.47x |
| `p64-L32-i32-N1024` | evoxc-auto | 7.71x | 3.74x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 7.54x | 3.66x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.75x | 0.36x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.48x | 0.23x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 7.83x | 3.80x |
| `p64-L64-i8-N1024` | evoxc-auto | 5.13x | 2.64x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 5.13x | 2.64x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.77x | 0.40x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.50x | 0.26x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 5.11x | 2.63x |
| `p64-L64-i32-N1024` | evoxc-auto | 4.94x | 2.65x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 4.64x | 2.49x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.76x | 0.41x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.49x | 0.26x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 5.06x | 2.72x |
| `p256-L32-i8-N1024` | evoxc-auto | 27.42x | 3.64x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 26.89x | 3.57x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.75x | 0.10x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 27.36x | 3.63x |
| `p256-L32-i32-N1024` | evoxc-auto | 26.73x | 3.53x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 26.86x | 3.55x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 26.71x | 3.53x |
| `p256-L64-i8-N1024` | evoxc-auto | 15.42x | 2.17x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 15.25x | 2.15x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.11x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 15.25x | 2.15x |
| `p256-L64-i32-N1024` | evoxc-auto | 14.91x | 2.20x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 14.48x | 2.14x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 14.12x | 2.09x |
| `p1000-L32-i8-N1024` | evoxc-auto | 63.44x | 2.30x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 63.07x | 2.29x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.79x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 63.36x | 2.30x |
| `p1000-L32-i32-N1024` | evoxc-auto | 62.64x | 2.23x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 62.21x | 2.21x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.80x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 60.86x | 2.16x |
| `p1000-L64-i8-N1024` | evoxc-auto | 21.91x | 0.88x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 21.80x | 0.88x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.78x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 21.81x | 0.88x |
| `p1000-L64-i32-N1024` | evoxc-auto | 21.13x | 0.86x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 21.12x | 0.86x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.77x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 20.84x | 0.85x |
| `p16-L64-i32-N64` | evoxc-auto | 0.77x | 1.34x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.48x | 0.82x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.77x | 1.33x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.48x | 0.83x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.26x | 2.17x |
| `p64-L64-i32-N64` | evoxc-auto | 5.17x | 2.87x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 5.15x | 2.86x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.78x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.48x | 0.27x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 5.21x | 2.89x |
| `p256-L64-i32-N64` | evoxc-auto | 20.35x | 3.09x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 19.42x | 2.95x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.73x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 20.39x | 3.09x |
| `p1000-L64-i32-N64` | evoxc-auto | 75.19x | 3.15x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 76.78x | 3.21x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.85x | 0.04x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 75.69x | 3.17x |

### Compare vs before: sr_fitness

primary `results/final/after` / `before` `results/final/before` (both measured in the same session; one process per config on each side).

Median ratio `primary / before` per arm, same cell: >1 means the before run was faster (the primary got slower), <1 means it was slower.  A pair missing on the primary side is quoted as `- (before <ms>)`; a pair missing on the before side stays `-`.  Only cells present in both directories are listed.

| cell | torch-per-tree / torch-per-tree(before) | torch-batched / torch-batched(before) | evoxc-auto / evoxc-auto(before) | evoxc-auto-iree / evoxc-auto-iree(before) | evoxc-specialized-numpy / evoxc-specialized-numpy(before) | evoxc-specialized-iree / evoxc-specialized-iree(before) | evoxc-dense-iree / evoxc-dense-iree(before) |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | - | - | 0.80x | 1.59x | 0.82x | 0.90x | 0.41x |
| `p16-L32-i32-N1024` | - | - | 0.81x | 1.62x | 0.85x | 0.86x | 0.38x |
| `p16-L64-i8-N1024` | - | - | 0.75x | 1.24x | 0.75x | 0.87x | 0.50x |
| `p16-L64-i32-N1024` | - | - | 0.93x | 1.31x | 0.86x | 0.84x | 0.47x |
| `p64-L32-i8-N1024` | - | - | 0.11x | 0.38x | 1.04x | 1.10x | 0.40x |
| `p64-L32-i32-N1024` | - | - | 0.12x | 0.38x | 1.08x | 1.09x | 0.35x |
| `p64-L64-i8-N1024` | - | - | 0.17x | 0.50x | 1.11x | 1.12x | 0.48x |
| `p64-L64-i32-N1024` | - | - | 0.17x | 0.52x | 1.15x | 1.02x | 0.47x |
| `p256-L32-i8-N1024` | - | - | 0.03x | 0.36x | 1.10x | - | 0.36x |
| `p256-L32-i32-N1024` | - | - | 0.03x | 0.30x | 1.12x | - | 0.36x |
| `p256-L64-i8-N1024` | - | - | 0.05x | 0.46x | 1.10x | - | 0.47x |
| `p256-L64-i32-N1024` | - | - | 0.05x | 0.49x | 1.13x | - | 0.49x |
| `p1000-L32-i8-N1024` | - | - | 0.01x | 0.34x | 1.07x | - | 0.35x |
| `p1000-L32-i32-N1024` | - | - | 0.01x | 0.35x | 1.06x | - | 0.36x |
| `p1000-L64-i8-N1024` | - | - | 0.04x | 0.54x | 1.11x | - | 0.54x |
| `p1000-L64-i32-N1024` | - | - | 0.04x | 0.54x | 1.13x | - | 0.55x |
| `p16-L64-i32-N64` | - | - | 0.94x | 1.35x | 0.82x | 0.90x | 0.51x |
| `p64-L64-i32-N64` | - | - | 0.16x | 0.48x | 1.17x | 1.11x | 0.47x |
| `p256-L64-i32-N64` | - | - | 0.05x | 0.41x | 1.24x | - | 0.39x |
| `p1000-L64-i32-N64` | - | - | 0.01x | 0.24x | 1.04x | - | 0.25x |

## Workload: batch_forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.389 (1.373) | 2.749 (2.729) | 2.255 (2.233) | 3.752 (3.694) | 2.202 (2.148) | 3.780 (3.649) | 1.020 (0.986) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.467 (1.455) | 2.790 (2.745) | 2.419 (2.322) | 4.661 (3.926) | 2.291 (2.280) | 3.791 (3.775) | 1.003 (0.987) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.389 (1.375) | 3.177 (3.135) | 2.495 (2.215) | 3.907 (3.708) | 2.225 (2.192) | 3.801 (3.701) | 1.494 (1.477) | auto/numpy → specialized | torch-per-tree |
| `p16-L64-i32-N1024` | 1.448 (1.424) | 3.057 (2.963) | 2.524 (2.447) | 3.802 (3.795) | 2.458 (2.381) | 3.819 (3.743) | 1.467 (1.444) | auto/numpy → specialized | torch-per-tree |
| `p64-L32-i8-N1024` | 5.673 (5.383) | 3.550 (3.526) | 1.207 (1.160) | 1.196 (1.159) | 9.987 (9.914) | 15.009 (14.864) | 1.196 (1.184) | auto/numpy → dense | evoxc-auto-iree |
| `p64-L32-i32-N1024` | 5.708 (5.677) | 3.528 (3.504) | 1.153 (1.087) | 1.203 (1.182) | 9.861 (9.443) | 15.760 (15.188) | 1.151 (1.133) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 5.839 (5.785) | 3.981 (3.761) | 1.738 (1.723) | 1.762 (1.684) | 9.967 (9.922) | 15.183 (14.779) | 1.748 (1.671) | auto/numpy → dense | evoxc-auto |
| `p64-L64-i32-N1024` | 5.733 (5.709) | 4.000 (3.974) | 1.758 (1.698) | 1.755 (1.714) | 9.301 (9.089) | 15.397 (14.986) | 1.720 (1.702) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L32-i8-N1024` | 24.339 (24.121) | 3.798 (3.668) | 1.801 (1.748) | 1.920 (1.893) | 40.746 (39.823) | - | 1.929 (1.909) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 23.143 (22.757) | 3.977 (3.790) | 1.935 (1.921) | 1.916 (1.883) | 39.971 (37.989) | - | 1.940 (1.917) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i8-N1024` | 23.322 (23.223) | 4.323 (4.315) | 2.807 (2.650) | 2.846 (2.812) | 39.271 (37.780) | - | 2.858 (2.582) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.438 (23.269) | 4.365 (4.355) | 2.856 (2.676) | 2.814 (2.725) | 39.315 (37.751) | - | 2.829 (2.650) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 92.121 (91.921) | 4.346 (4.331) | 3.917 (3.835) | 3.749 (3.571) | 155.661 (154.446) | - | 3.973 (3.924) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i32-N1024` | 94.942 (94.539) | 4.383 (4.365) | 3.898 (3.840) | 4.002 (3.934) | 152.599 (152.248) | - | 3.879 (3.740) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i8-N1024` | 92.167 (91.515) | 4.816 (4.808) | 7.485 (7.418) | 7.479 (7.419) | 153.566 (151.552) | - | 7.487 (7.325) | auto/numpy → dense | torch-batched |
| `p1000-L64-i32-N1024` | 92.688 (91.733) | 4.886 (4.847) | 7.653 (7.437) | 7.566 (7.410) | 155.765 (151.898) | - | 7.469 (7.428) | auto/numpy → dense | torch-batched |
| `p16-L64-i32-N64` | 1.394 (1.371) | 3.113 (3.095) | 2.333 (2.273) | 3.946 (3.888) | 2.326 (2.319) | 3.832 (3.764) | 1.385 (1.368) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 5.663 (5.631) | 4.015 (3.965) | 1.432 (1.385) | 1.428 (1.403) | 9.303 (8.966) | 15.443 (15.185) | 1.421 (1.393) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i32-N64` | 21.821 (21.069) | 4.483 (4.254) | 1.508 (1.473) | 1.579 (1.555) | 41.590 (41.459) | - | 1.514 (1.483) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i32-N64` | 91.198 (90.986) | 4.927 (4.885) | 1.798 (1.714) | 1.765 (1.729) | 143.147 (141.764) | - | 1.798 (1.752) | auto/numpy → dense | evoxc-auto-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (batch_forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.62x | 1.22x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.37x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.63x | 1.25x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.37x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 1.36x | 2.70x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.61x | 1.15x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.31x | 0.60x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.64x | 1.22x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.39x | 0.74x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 1.46x | 2.78x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.56x | 1.27x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.36x | 0.81x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.62x | 1.43x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.37x | 0.84x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.93x | 2.13x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.57x | 1.21x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.38x | 0.80x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.59x | 1.24x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.38x | 0.80x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.99x | 2.08x |
| `p64-L32-i8-N1024` | evoxc-auto | 4.70x | 2.94x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 4.74x | 2.97x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.57x | 0.36x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.24x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 4.74x | 2.97x |
| `p64-L32-i32-N1024` | evoxc-auto | 4.95x | 3.06x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 4.75x | 2.93x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.36x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.36x | 0.22x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 4.96x | 3.06x |
| `p64-L64-i8-N1024` | evoxc-auto | 3.36x | 2.29x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 3.31x | 2.26x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.40x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.38x | 0.26x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 3.34x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-auto | 3.26x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 3.27x | 2.28x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.62x | 0.43x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.37x | 0.26x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 3.33x | 2.33x |
| `p256-L32-i8-N1024` | evoxc-auto | 13.52x | 2.11x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 12.68x | 1.98x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.60x | 0.09x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 12.62x | 1.97x |
| `p256-L32-i32-N1024` | evoxc-auto | 11.96x | 2.06x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 12.08x | 2.08x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.58x | 0.10x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 11.93x | 2.05x |
| `p256-L64-i8-N1024` | evoxc-auto | 8.31x | 1.54x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 8.19x | 1.52x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.11x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 8.16x | 1.51x |
| `p256-L64-i32-N1024` | evoxc-auto | 8.21x | 1.53x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 8.33x | 1.55x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.11x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 8.29x | 1.54x |
| `p1000-L32-i8-N1024` | evoxc-auto | 23.52x | 1.11x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 24.57x | 1.16x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.59x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 23.19x | 1.09x |
| `p1000-L32-i32-N1024` | evoxc-auto | 24.36x | 1.12x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 23.72x | 1.10x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.62x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 24.48x | 1.13x |
| `p1000-L64-i8-N1024` | evoxc-auto | 12.31x | 0.64x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 12.32x | 0.64x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.60x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 12.31x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-auto | 12.11x | 0.64x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 12.25x | 0.65x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.60x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 12.41x | 0.65x |
| `p16-L64-i32-N64` | evoxc-auto | 0.60x | 1.33x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.35x | 0.79x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.60x | 1.34x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.36x | 0.81x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.01x | 2.25x |
| `p64-L64-i32-N64` | evoxc-auto | 3.96x | 2.80x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 3.97x | 2.81x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.61x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.37x | 0.26x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 3.98x | 2.82x |
| `p256-L64-i32-N64` | evoxc-auto | 14.47x | 2.97x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 13.82x | 2.84x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.52x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 14.41x | 2.96x |
| `p1000-L64-i32-N64` | evoxc-auto | 50.73x | 2.74x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 51.66x | 2.79x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.64x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 50.71x | 2.74x |

### Compare vs before: batch_forward

primary `results/final/after` / `before` `results/final/before` (both measured in the same session; one process per config on each side).

Median ratio `primary / before` per arm, same cell: >1 means the before run was faster (the primary got slower), <1 means it was slower.  A pair missing on the primary side is quoted as `- (before <ms>)`; a pair missing on the before side stays `-`.  Only cells present in both directories are listed.

| cell | torch-per-tree / torch-per-tree(before) | torch-batched / torch-batched(before) | evoxc-auto / evoxc-auto(before) | evoxc-auto-iree / evoxc-auto-iree(before) | evoxc-specialized-numpy / evoxc-specialized-numpy(before) | evoxc-specialized-iree / evoxc-specialized-iree(before) | evoxc-dense-iree / evoxc-dense-iree(before) |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | - | - | 0.74x | 1.66x | 0.76x | 0.88x | 0.45x |
| `p16-L32-i32-N1024` | - | - | 0.82x | 2.05x | 0.79x | 0.87x | 0.43x |
| `p16-L64-i8-N1024` | - | - | 0.86x | 1.41x | 0.75x | 0.88x | 0.54x |
| `p16-L64-i32-N1024` | - | - | 0.84x | 1.35x | 0.90x | 0.82x | 0.52x |
| `p64-L32-i8-N1024` | - | - | 0.15x | 0.45x | 1.23x | 1.11x | 0.47x |
| `p64-L32-i32-N1024` | - | - | 0.14x | 0.47x | 1.04x | 1.12x | 0.43x |
| `p64-L64-i8-N1024` | - | - | 0.21x | 0.57x | 1.13x | 1.12x | 0.56x |
| `p64-L64-i32-N1024` | - | - | 0.21x | 0.55x | 1.00x | 1.03x | 0.53x |
| `p256-L32-i8-N1024` | - | - | 0.05x | 0.55x | 1.09x | - | 0.56x |
| `p256-L32-i32-N1024` | - | - | 0.05x | 0.55x | 1.07x | - | 0.55x |
| `p256-L64-i8-N1024` | - | - | 0.08x | 0.58x | 1.11x | - | 0.61x |
| `p256-L64-i32-N1024` | - | - | 0.07x | 0.58x | 1.15x | - | 0.57x |
| `p1000-L32-i8-N1024` | - | - | 0.03x | 0.54x | 1.10x | - | 0.58x |
| `p1000-L32-i32-N1024` | - | - | 0.03x | 0.54x | 1.09x | - | 0.56x |
| `p1000-L64-i8-N1024` | - | - | 0.06x | 0.62x | 1.08x | - | 0.63x |
| `p1000-L64-i32-N1024` | - | - | 0.05x | 0.62x | 1.13x | - | 0.62x |
| `p16-L64-i32-N64` | - | - | 0.83x | 1.51x | 0.89x | 0.92x | 0.52x |
| `p64-L64-i32-N64` | - | - | 0.17x | 0.50x | 1.15x | 1.07x | 0.51x |
| `p256-L64-i32-N64` | - | - | 0.04x | 0.43x | 1.17x | - | 0.42x |
| `p1000-L64-i32-N64` | - | - | 0.01x | 0.28x | 1.05x | - | 0.29x |

## Workload: forward

Warm/steady-state `median_ms` with the `min_ms` in parentheses.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree | auto route | fastest |
|---|---|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 1.423 (1.404) | 2.701 (2.554) | 2.266 (2.197) | 5.017 (4.955) | 2.195 (1.976) | 5.232 (5.002) | 0.643 (0.640) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L32-i32-N1024` | 1.511 (1.470) | 2.739 (2.690) | 2.159 (2.147) | 5.452 (5.342) | 2.229 (1.985) | 5.127 (5.047) | 0.642 (0.615) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i8-N1024` | 1.395 (1.335) | 3.417 (3.382) | 2.030 (1.996) | 5.115 (5.014) | 2.054 (2.022) | 5.101 (4.925) | 0.861 (0.850) | auto/numpy → specialized | evoxc-dense-iree |
| `p16-L64-i32-N1024` | 1.476 (1.448) | 2.949 (2.931) | 2.048 (2.007) | 4.989 (4.882) | 2.216 (2.012) | 4.977 (4.883) | 0.893 (0.840) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L32-i8-N1024` | 5.795 (5.773) | 3.459 (3.427) | 0.643 (0.624) | 0.696 (0.678) | 8.447 (8.280) | 24.850 (24.477) | 0.656 (0.649) | auto/numpy → dense | evoxc-auto |
| `p64-L32-i32-N1024` | 5.900 (5.787) | 3.492 (3.431) | 0.641 (0.634) | 0.649 (0.622) | 9.221 (8.882) | 22.629 (22.160) | 0.625 (0.617) | auto/numpy → dense | evoxc-dense-iree |
| `p64-L64-i8-N1024` | 5.917 (5.851) | 4.028 (3.928) | 0.843 (0.828) | 0.859 (0.833) | 9.346 (9.329) | 24.235 (23.559) | 0.871 (0.863) | auto/numpy → dense | evoxc-auto |
| `p64-L64-i32-N1024` | 5.819 (5.802) | 3.948 (3.875) | 0.861 (0.851) | 0.870 (0.835) | 9.215 (8.470) | 22.400 (22.184) | 0.863 (0.811) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i8-N1024` | 25.161 (24.712) | 3.951 (3.882) | 0.637 (0.624) | 0.682 (0.659) | 37.274 (36.921) | - | 0.663 (0.643) | auto/numpy → dense | evoxc-auto |
| `p256-L32-i32-N1024` | 23.943 (23.802) | 3.922 (3.876) | 0.662 (0.632) | 0.666 (0.651) | 35.880 (34.103) | - | 0.655 (0.642) | auto/numpy → dense | evoxc-dense-iree |
| `p256-L64-i8-N1024` | 23.531 (23.215) | 4.312 (4.264) | 0.853 (0.849) | 0.860 (0.839) | 37.430 (36.932) | - | 0.925 (0.903) | auto/numpy → dense | evoxc-auto |
| `p256-L64-i32-N1024` | 23.730 (22.678) | 4.340 (4.312) | 0.871 (0.855) | 0.852 (0.842) | 35.868 (34.208) | - | 0.868 (0.846) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i8-N1024` | 97.247 (96.309) | 4.056 (4.009) | 0.651 (0.636) | 0.646 (0.640) | 145.632 (142.609) | - | 0.694 (0.664) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L32-i32-N1024` | 96.700 (96.512) | 3.993 (3.978) | 0.653 (0.652) | 0.658 (0.631) | 147.776 (146.109) | - | 0.654 (0.648) | auto/numpy → dense | evoxc-auto |
| `p1000-L64-i8-N1024` | 93.874 (93.169) | 4.959 (4.801) | 0.878 (0.845) | 0.934 (0.922) | 141.689 (139.923) | - | 0.877 (0.867) | auto/numpy → dense | evoxc-dense-iree |
| `p1000-L64-i32-N1024` | 94.010 (92.354) | 4.562 (4.510) | 0.858 (0.831) | 0.932 (0.899) | 142.466 (139.444) | - | 0.872 (0.863) | auto/numpy → dense | evoxc-auto |
| `p16-L64-i32-N64` | 1.451 (1.415) | 3.122 (2.956) | 2.289 (2.250) | 4.949 (4.933) | 2.243 (2.201) | 5.817 (5.710) | 0.864 (0.822) | auto/numpy → specialized | evoxc-dense-iree |
| `p64-L64-i32-N64` | 6.270 (6.089) | 3.917 (3.855) | 0.866 (0.854) | 0.851 (0.823) | 9.031 (8.173) | 22.916 (22.655) | 0.895 (0.875) | auto/numpy → dense | evoxc-auto-iree |
| `p256-L64-i32-N64` | 23.735 (23.532) | 4.326 (4.247) | 0.924 (0.872) | 0.874 (0.857) | 40.567 (40.446) | - | 0.972 (0.937) | auto/numpy → dense | evoxc-auto-iree |
| `p1000-L64-i32-N64` | 97.933 (97.083) | 4.795 (4.777) | 0.863 (0.842) | 0.884 (0.835) | 141.449 (140.101) | - | 0.845 (0.824) | auto/numpy → dense | evoxc-dense-iree |

`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: `eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that arm is absent the arm the route was read from is named in parentheses.

### Speedup vs torch (forward)

`torch-*` median divided by the arm's median, same cell, same session (>1 = the torch baseline is slower).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.63x | 1.19x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.28x | 0.54x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.65x | 1.23x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.27x | 0.52x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 2.21x | 4.20x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.70x | 1.27x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.28x | 0.50x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.68x | 1.23x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.29x | 0.53x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 2.35x | 4.26x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.69x | 1.68x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.27x | 0.67x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.68x | 1.66x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.27x | 0.67x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 1.62x | 3.97x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.72x | 1.44x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.30x | 0.59x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.67x | 1.33x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.30x | 0.59x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 1.65x | 3.30x |
| `p64-L32-i8-N1024` | evoxc-auto | 9.01x | 5.38x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 8.33x | 4.97x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.69x | 0.41x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.23x | 0.14x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 8.83x | 5.27x |
| `p64-L32-i32-N1024` | evoxc-auto | 9.21x | 5.45x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 9.09x | 5.38x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.64x | 0.38x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.26x | 0.15x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 9.44x | 5.59x |
| `p64-L64-i8-N1024` | evoxc-auto | 7.02x | 4.78x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6.89x | 4.69x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.63x | 0.43x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.24x | 0.17x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6.79x | 4.62x |
| `p64-L64-i32-N1024` | evoxc-auto | 6.76x | 4.59x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6.69x | 4.54x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.63x | 0.43x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.26x | 0.18x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6.74x | 4.57x |
| `p256-L32-i8-N1024` | evoxc-auto | 39.52x | 6.21x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 36.90x | 5.79x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.68x | 0.11x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 37.92x | 5.95x |
| `p256-L32-i32-N1024` | evoxc-auto | 36.14x | 5.92x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 35.93x | 5.89x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.67x | 0.11x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 36.54x | 5.99x |
| `p256-L64-i8-N1024` | evoxc-auto | 27.59x | 5.05x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 27.36x | 5.01x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.63x | 0.12x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 25.43x | 4.66x |
| `p256-L64-i32-N1024` | evoxc-auto | 27.23x | 4.98x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 27.87x | 5.10x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.66x | 0.12x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 27.35x | 5.00x |
| `p1000-L32-i8-N1024` | evoxc-auto | 149.43x | 6.23x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 150.56x | 6.28x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.67x | 0.03x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 140.06x | 5.84x |
| `p1000-L32-i32-N1024` | evoxc-auto | 147.97x | 6.11x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 146.96x | 6.07x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.65x | 0.03x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 147.77x | 6.10x |
| `p1000-L64-i8-N1024` | evoxc-auto | 106.87x | 5.65x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 100.50x | 5.31x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 107.08x | 5.66x |
| `p1000-L64-i32-N1024` | evoxc-auto | 109.52x | 5.31x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 100.83x | 4.89x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.66x | 0.03x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 107.80x | 5.23x |
| `p16-L64-i32-N64` | evoxc-auto | 0.63x | 1.36x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.29x | 0.63x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.65x | 1.39x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.25x | 0.54x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 1.68x | 3.61x |
| `p64-L64-i32-N64` | evoxc-auto | 7.24x | 4.52x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 7.37x | 4.61x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.69x | 0.43x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.27x | 0.17x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 7.01x | 4.38x |
| `p256-L64-i32-N64` | evoxc-auto | 25.68x | 4.68x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 27.17x | 4.95x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.59x | 0.11x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 24.43x | 4.45x |
| `p1000-L64-i32-N64` | evoxc-auto | 113.48x | 5.56x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 110.81x | 5.42x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.69x | 0.03x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 115.97x | 5.68x |

### Compare vs before: forward

primary `results/final/after` / `before` `results/final/before` (both measured in the same session; one process per config on each side).

Median ratio `primary / before` per arm, same cell: >1 means the before run was faster (the primary got slower), <1 means it was slower.  A pair missing on the primary side is quoted as `- (before <ms>)`; a pair missing on the before side stays `-`.  Only cells present in both directories are listed.

| cell | torch-per-tree / torch-per-tree(before) | torch-batched / torch-batched(before) | evoxc-auto / evoxc-auto(before) | evoxc-auto-iree / evoxc-auto-iree(before) | evoxc-specialized-numpy / evoxc-specialized-numpy(before) | evoxc-specialized-iree / evoxc-specialized-iree(before) | evoxc-dense-iree / evoxc-dense-iree(before) |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | - | - | 0.81x | 2.67x | 0.80x | 0.91x | 0.35x |
| `p16-L32-i32-N1024` | - | - | 0.79x | 2.97x | 0.82x | 0.89x | 0.35x |
| `p16-L64-i8-N1024` | - | - | 0.72x | 2.37x | 0.76x | 0.88x | 0.42x |
| `p16-L64-i32-N1024` | - | - | 0.82x | 2.36x | 0.83x | 0.84x | 0.42x |
| `p64-L32-i8-N1024` | - | - | 0.08x | 0.34x | 1.11x | 1.28x | 0.34x |
| `p64-L32-i32-N1024` | - | - | 0.08x | 0.33x | 1.10x | 1.17x | 0.30x |
| `p64-L64-i8-N1024` | - | - | 0.10x | 0.38x | 1.11x | 1.22x | 0.38x |
| `p64-L64-i32-N1024` | - | - | 0.10x | 0.39x | 1.06x | 1.09x | 0.38x |
| `p256-L32-i8-N1024` | - | - | 0.02x | 0.28x | 1.10x | - | 0.27x |
| `p256-L32-i32-N1024` | - | - | 0.02x | 0.27x | 1.10x | - | 0.27x |
| `p256-L64-i8-N1024` | - | - | 0.03x | 0.29x | 1.14x | - | 0.35x |
| `p256-L64-i32-N1024` | - | - | 0.02x | 0.29x | 1.07x | - | 0.30x |
| `p1000-L32-i8-N1024` | - | - | 0.00x | 0.15x | 1.09x | - | 0.18x |
| `p1000-L32-i32-N1024` | - | - | 0.01x | 0.15x | 1.15x | - | 0.16x |
| `p1000-L64-i8-N1024` | - | - | 0.01x | 0.17x | 1.12x | - | 0.16x |
| `p1000-L64-i32-N1024` | - | - | 0.01x | 0.17x | 1.09x | - | 0.16x |
| `p16-L64-i32-N64` | - | - | 0.93x | 2.36x | 0.82x | 0.94x | 0.41x |
| `p64-L64-i32-N64` | - | - | 0.10x | 0.38x | 1.10x | 0.99x | 0.40x |
| `p256-L64-i32-N64` | - | - | 0.03x | 0.29x | 1.22x | - | 0.32x |
| `p1000-L64-i32-N64` | - | - | 0.01x | 0.16x | 1.04x | - | 0.15x |

## gp_step

One full GP generation (evaluate + select + crossover + mutate), warm.

### Total

Every cell is the record's own `median_total_ms`.  A `*` marks a cell the aggregate key could not supply, re-derived from the upper median of that record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), so a record with raw steps is never blanked.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 3.308 | 4.859 | 6.626 | 755.470 | 6.370 | 799.942 | 5.042 |
| `p16-L32-i32-N1024` | 3.520 | 4.830 | 6.725 | 736.358 | 7.007 | 744.433 | 5.392 |
| `p16-L64-i8-N1024` | 3.120 | 5.596 | 6.578 | 751.604 | 6.607 | 746.820 | 5.653 |
| `p16-L64-i32-N1024` | 3.280 | 5.752 | 6.637 | 736.565 | 6.926 | 754.958 | 6.494 |
| `p64-L32-i8-N1024` | 5.391 | 7.058 | 4.934 | 4.989 | 40.587 | 2760.215 | 4.779 |
| `p64-L32-i32-N1024` | 5.430 | 8.561 | 4.943 | 4.913 | 62.907 | 6326.840 | 4.951 |
| `p64-L64-i8-N1024` | 5.061 | 7.136 | 5.393 | 5.407 | 38.733 | 2739.012 | 5.365 |
| `p64-L64-i32-N1024` | 5.548 | 9.673 | 5.417 | 5.410 | 64.111 | 6477.668 | 5.319 |
| `p256-L32-i8-N1024` | 20.537 | 23.165 | 5.381 | 5.468 | 317.179 | - | 5.548 |
| `p256-L32-i32-N1024` | 21.598 | 24.759 | 5.484 | 5.468 | 374.651 | - | 5.467 |
| `p256-L64-i8-N1024` | 19.263 | 25.957 | 6.607 | 6.587 | 313.787 | - | 6.797 |
| `p256-L64-i32-N1024` | 21.141 | 28.043 | 6.630 | 6.632 | 370.964 | - | 6.574 |
| `p1000-L32-i8-N1024` | 60.794 | 56.344 | 7.722 | 7.940 | 1071.906 | - | 9.176 |
| `p1000-L32-i32-N1024` | 58.237 | 52.079 | 7.436 | 7.363 | 935.422 | - | 7.360 |
| `p1000-L64-i8-N1024` | 61.938 | 77.896 | 12.832 | 12.873 | 1044.985 | - | 12.661 |
| `p1000-L64-i32-N1024` | 56.248 | 67.241 | 12.410 | 12.803 | 887.258 | - | 12.593 |
| `p16-L64-i32-N64` | 3.232 | 5.808 | 6.687 | 754.663 | 6.726 | 755.414 | 5.795 |
| `p64-L64-i32-N64` | 5.129 | 7.478 | 5.329 | 5.319 | 24.583 | 4031.834 | 5.532 |
| `p256-L64-i32-N64` | 24.117 | 29.000 | 6.066 | 6.444 | 521.846 | - | 6.634 |
| `p1000-L64-i32-N64` | 56.602 | 64.290 | 8.465 | 8.303 | 1040.052 | - | 8.205 |

### Eval / ops split

`median_eval_ms / median_ops_ms` with the operator share of the generation in parentheses (`ops / total`); the operator part is the in-place difference on the same step, not a cross-population comparison.  A `*` marks a value re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | 0.961 / 2.347 (op 71%) | 2.550 / 2.309 (op 48%) | 4.308 / 2.318 (op 35%) | 752.831 / 2.638 (op 0%) | 4.237 / 2.133 (op 33%) | 797.190 / 2.753 (op 0%) | 2.640 / 2.402 (op 48%) |
| `p16-L32-i32-N1024` | 1.000 / 2.520 (op 72%) | 2.577 / 2.253 (op 47%) | 4.245 / 2.481 (op 37%) | 733.684 / 2.674 (op 0%) | 4.598 / 2.409 (op 34%) | 741.732 / 2.701 (op 0%) | 2.986 / 2.406 (op 45%) |
| `p16-L64-i8-N1024` | 0.942 / 2.179 (op 70%) | 3.344 / 2.252 (op 40%) | 4.226 / 2.352 (op 36%) | 748.938 / 2.666 (op 0%) | 4.209 / 2.398 (op 36%) | 744.189 / 2.631 (op 0%) | 3.272 / 2.381 (op 42%) |
| `p16-L64-i32-N1024` | 0.934 / 2.346 (op 72%) | 3.375 / 2.376 (op 41%) | 4.454 / 2.183 (op 33%) | 733.944 / 2.621 (op 0%) | 4.558 / 2.368 (op 34%) | 752.337 / 2.620 (op 0%) | 4.072 / 2.421 (op 37%) |
| `p64-L32-i8-N1024` | 3.015 / 2.377 (op 44%) | 4.094 / 2.965 (op 42%) | 2.455 / 2.479 (op 50%) | 2.398 / 2.592 (op 52%) | 38.136 / 2.451 (op 6%) | 2757.099 / 3.117 (op 0%) | 2.385 / 2.393 (op 50%) |
| `p64-L32-i32-N1024` | 3.100 / 2.330 (op 43%) | 6.238 / 2.323 (op 27%) | 2.344 / 2.599 (op 53%) | 2.326 / 2.586 (op 53%) | 60.742 / 2.166 (op 3%) | 6323.386 / 3.454 (op 0%) | 2.394 / 2.556 (op 52%) |
| `p64-L64-i8-N1024` | 2.709 / 2.351 (op 46%) | 4.790 / 2.345 (op 33%) | 2.998 / 2.396 (op 44%) | 3.030 / 2.378 (op 44%) | 36.327 / 2.406 (op 6%) | 2735.926 / 3.086 (op 0%) | 3.037 / 2.328 (op 43%) |
| `p64-L64-i32-N1024` | 3.160 / 2.388 (op 43%) | 7.281 / 2.392 (op 25%) | 3.043 / 2.373 (op 44%) | 3.055 / 2.355 (op 44%) | 61.651 / 2.460 (op 4%) | 6474.211 / 3.457 (op 0%) | 2.995 / 2.323 (op 44%) |
| `p256-L32-i8-N1024` | 17.841 / 2.695 (op 13%) | 20.725 / 2.440 (op 11%) | 2.899 / 2.482 (op 46%) | 2.942 / 2.527 (op 46%) | 314.527 / 2.652 (op 1%) | - | 2.944 / 2.604 (op 47%) |
| `p256-L32-i32-N1024` | 18.997 / 2.600 (op 12%) | 22.209 / 2.550 (op 10%) | 2.966 / 2.518 (op 46%) | 3.007 / 2.461 (op 45%) | 372.103 / 2.549 (op 1%) | - | 2.963 / 2.504 (op 46%) |
| `p256-L64-i8-N1024` | 16.804 / 2.459 (op 13%) | 23.480 / 2.477 (op 10%) | 4.132 / 2.476 (op 37%) | 4.134 / 2.453 (op 37%) | 311.068 / 2.719 (op 1%) | - | 4.212 / 2.586 (op 38%) |
| `p256-L64-i32-N1024` | 18.559 / 2.583 (op 12%) | 25.430 / 2.613 (op 9%) | 4.190 / 2.440 (op 37%) | 4.191 / 2.442 (op 37%) | 368.163 / 2.801 (op 1%) | - | 4.169 / 2.406 (op 37%) |
| `p1000-L32-i8-N1024` | 58.091 / 2.703 (op 4%) | 53.903 / 2.442 (op 4%) | 5.003 / 2.719 (op 35%) | 5.152 / 2.788 (op 35%) | 1068.883 / 3.023 (op 0%) | - | 6.558 / 2.617 (op 29%) |
| `p1000-L32-i32-N1024` | 55.423 / 2.813 (op 5%) | 49.528 / 2.550 (op 5%) | 4.867 / 2.570 (op 35%) | 4.905 / 2.458 (op 33%) | 932.592 / 2.831 (op 0%) | - | 4.827 / 2.533 (op 34%) |
| `p1000-L64-i8-N1024` | 59.104 / 2.833 (op 5%) | 75.311 / 2.586 (op 3%) | 10.181 / 2.651 (op 21%) | 10.129 / 2.744 (op 21%) | 1041.964 / 3.021 (op 0%) | - | 10.208 / 2.453 (op 19%) |
| `p1000-L64-i32-N1024` | 53.619 / 2.629 (op 5%) | 64.601 / 2.640 (op 4%) | 9.861 / 2.549 (op 21%) | 10.198 / 2.605 (op 20%) | 884.317 / 2.941 (op 0%) | - | 10.109 / 2.484 (op 20%) |
| `p16-L64-i32-N64` | 0.928 / 2.304 (op 71%) | 3.465 / 2.343 (op 40%) | 4.343 / 2.344 (op 35%) | 751.947 / 2.716 (op 0%) | 4.359 / 2.366 (op 35%) | 752.777 / 2.637 (op 0%) | 3.229 / 2.566 (op 44%) |
| `p64-L64-i32-N64` | 2.773 / 2.356 (op 46%) | 5.143 / 2.335 (op 31%) | 2.897 / 2.432 (op 46%) | 2.893 / 2.426 (op 46%) | 22.159 / 2.424 (op 10%) | 4028.691 / 3.143 (op 0%) | 2.988 / 2.544 (op 46%) |
| `p256-L64-i32-N64` | 21.558 / 2.559 (op 11%) | 26.352 / 2.648 (op 9%) | 3.577 / 2.489 (op 41%) | 3.754 / 2.691 (op 42%) | 518.819 / 3.027 (op 1%) | - | 3.944 / 2.689 (op 41%) |
| `p1000-L64-i32-N64` | 53.955 / 2.648 (op 5%) | 61.836 / 2.454 (op 4%) | 5.847 / 2.619 (op 31%) | 5.777 / 2.526 (op 30%) | 1037.111 / 2.941 (op 0%) | - | 5.709 / 2.495 (op 30%) |

### Speedup vs torch (gp_step)

`torch-*` `gp_step` total divided by the arm's total, same cell, same session (>1 = the torch baseline is slower, i.e. the arm is faster).

| cell | arm | torch-per-tree / arm | torch-batched / arm |
|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 0.50x | 0.73x |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 0.52x | 0.76x |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 0.66x | 0.96x |
| `p16-L32-i32-N1024` | evoxc-auto | 0.52x | 0.72x |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 0.50x | 0.69x |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 0.65x | 0.90x |
| `p16-L64-i8-N1024` | evoxc-auto | 0.47x | 0.85x |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 0.47x | 0.85x |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 0.55x | 0.99x |
| `p16-L64-i32-N1024` | evoxc-auto | 0.49x | 0.87x |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 0.47x | 0.83x |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 0.51x | 0.89x |
| `p64-L32-i8-N1024` | evoxc-auto | 1.09x | 1.43x |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 1.08x | 1.41x |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 0.13x | 0.17x |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 1.13x | 1.48x |
| `p64-L32-i32-N1024` | evoxc-auto | 1.10x | 1.73x |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 1.11x | 1.74x |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 0.09x | 0.14x |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 1.10x | 1.73x |
| `p64-L64-i8-N1024` | evoxc-auto | 0.94x | 1.32x |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 0.94x | 1.32x |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 0.13x | 0.18x |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 0.94x | 1.33x |
| `p64-L64-i32-N1024` | evoxc-auto | 1.02x | 1.79x |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 1.03x | 1.79x |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 0.09x | 0.15x |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 1.04x | 1.82x |
| `p256-L32-i8-N1024` | evoxc-auto | 3.82x | 4.31x |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3.76x | 4.24x |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3.70x | 4.18x |
| `p256-L32-i32-N1024` | evoxc-auto | 3.94x | 4.51x |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3.95x | 4.53x |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3.95x | 4.53x |
| `p256-L64-i8-N1024` | evoxc-auto | 2.92x | 3.93x |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 2.92x | 3.94x |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 2.83x | 3.82x |
| `p256-L64-i32-N1024` | evoxc-auto | 3.19x | 4.23x |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 3.19x | 4.23x |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 3.22x | 4.27x |
| `p1000-L32-i8-N1024` | evoxc-auto | 7.87x | 7.30x |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 7.66x | 7.10x |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.05x |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 6.63x | 6.14x |
| `p1000-L32-i32-N1024` | evoxc-auto | 7.83x | 7.00x |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 7.91x | 7.07x |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.06x |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 7.91x | 7.08x |
| `p1000-L64-i8-N1024` | evoxc-auto | 4.83x | 6.07x |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 4.81x | 6.05x |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 0.06x | 0.07x |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 4.89x | 6.15x |
| `p1000-L64-i32-N1024` | evoxc-auto | 4.53x | 5.42x |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 4.39x | 5.25x |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 0.06x | 0.08x |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 4.47x | 5.34x |
| `p16-L64-i32-N64` | evoxc-auto | 0.48x | 0.87x |
| `p16-L64-i32-N64` | evoxc-auto-iree | 0.00x | 0.01x |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 0.48x | 0.86x |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 0.00x | 0.01x |
| `p16-L64-i32-N64` | evoxc-dense-iree | 0.56x | 1.00x |
| `p64-L64-i32-N64` | evoxc-auto | 0.96x | 1.40x |
| `p64-L64-i32-N64` | evoxc-auto-iree | 0.96x | 1.41x |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 0.21x | 0.30x |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 0.00x | 0.00x |
| `p64-L64-i32-N64` | evoxc-dense-iree | 0.93x | 1.35x |
| `p256-L64-i32-N64` | evoxc-auto | 3.98x | 4.78x |
| `p256-L64-i32-N64` | evoxc-auto-iree | 3.74x | 4.50x |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 3.64x | 4.37x |
| `p1000-L64-i32-N64` | evoxc-auto | 6.69x | 7.59x |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6.82x | 7.74x |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 0.05x | 0.06x |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6.90x | 7.84x |

### Best fitness after the timed steps

`gp_step.best_fitness` of the pipeline that ran the timed steps (higher is better; `-` when the record has none).

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-auto-iree | evoxc-specialized-numpy | evoxc-specialized-iree | evoxc-dense-iree |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p16-L32-i32-N1024` | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 |
| `p16-L64-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p16-L64-i32-N1024` | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 | -0.161827 |
| `p64-L32-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p64-L32-i32-N1024` | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 |
| `p64-L64-i8-N1024` | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 | -0.161465 |
| `p64-L64-i32-N1024` | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 | -0.140469 |
| `p256-L32-i8-N1024` | -0.084110 | -0.084110 | -0.084110 | -0.084110 | -0.084110 | - | -0.084110 |
| `p256-L32-i32-N1024` | -0.084197 | -0.084197 | -0.084197 | -0.084197 | -0.084197 | - | -0.084197 |
| `p256-L64-i8-N1024` | -0.084110 | -0.084110 | -0.084110 | -0.084110 | -0.084110 | - | -0.084110 |
| `p256-L64-i32-N1024` | -0.084197 | -0.084197 | -0.084197 | -0.084197 | -0.084197 | - | -0.084197 |
| `p1000-L32-i8-N1024` | -0.083998 | -0.083998 | -0.083998 | -0.083998 | -0.083998 | - | -0.083998 |
| `p1000-L32-i32-N1024` | -0.084052 | -0.084052 | -0.084052 | -0.084052 | -0.084052 | - | -0.084052 |
| `p1000-L64-i8-N1024` | -0.084126 | -0.084126 | -0.084126 | -0.084126 | -0.084126 | - | -0.084126 |
| `p1000-L64-i32-N1024` | -0.084145 | -0.084145 | -0.084145 | -0.084145 | -0.084145 | - | -0.084145 |
| `p16-L64-i32-N64` | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 | -0.165720 |
| `p64-L64-i32-N64` | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 | -0.141121 |
| `p256-L64-i32-N64` | -0.090160 | -0.090160 | -0.090160 | -0.090160 | -0.090160 | - | -0.090160 |
| `p1000-L64-i32-N64` | -0.089923 | -0.089923 | -0.089923 | -0.089923 | -0.089923 | - | -0.089923 |

### Compare vs before: gp_step total

primary `results/final/after` / `before` `results/final/before` (both measured in the same session; one process per config on each side).

`gp_step` total ratio `primary / before` per arm, same cell (>1 = the before generation was faster).  Values carry the `gp_step` fallback: a record without `median_total_ms` is read from the upper median of its `steps[].total_ms` on either side.  Only cells present in both directories are listed.

| cell | torch-per-tree / torch-per-tree(before) | torch-batched / torch-batched(before) | evoxc-auto / evoxc-auto(before) | evoxc-auto-iree / evoxc-auto-iree(before) | evoxc-specialized-numpy / evoxc-specialized-numpy(before) | evoxc-specialized-iree / evoxc-specialized-iree(before) | evoxc-dense-iree / evoxc-dense-iree(before) |
|---|---|---|---|---|---|---|---|
| `p16-L32-i8-N1024` | - | - | 0.59x | 83.78x | 0.56x | 1.49x | 0.55x |
| `p16-L32-i32-N1024` | - | - | 0.11x | 85.92x | 0.12x | 0.21x | 0.64x |
| `p16-L64-i8-N1024` | - | - | 0.61x | 78.05x | 0.57x | 1.35x | 0.59x |
| `p16-L64-i32-N1024` | - | - | 0.10x | 81.23x | 0.11x | 0.19x | 0.70x |
| `p64-L32-i8-N1024` | - | - | 0.14x | 0.58x | 1.06x | 0.59x | 0.56x |
| `p64-L32-i32-N1024` | - | - | 0.10x | 0.58x | 1.20x | 0.95x | 0.56x |
| `p64-L64-i8-N1024` | - | - | 0.14x | 0.58x | 1.01x | 0.59x | 0.59x |
| `p64-L64-i32-N1024` | - | - | 0.11x | 0.60x | 1.24x | 1.06x | 0.59x |
| `p256-L32-i8-N1024` | - | - | 0.03x | 0.61x | 1.68x | - | 0.61x |
| `p256-L32-i32-N1024` | - | - | 0.03x | 0.60x | 2.29x | - | 0.60x |
| `p256-L64-i8-N1024` | - | - | 0.04x | 0.64x | 1.67x | - | 0.66x |
| `p256-L64-i32-N1024` | - | - | 0.04x | 0.64x | 2.30x | - | 0.64x |
| `p1000-L32-i8-N1024` | - | - | 0.01x | 0.71x | 1.19x | - | 0.82x |
| `p1000-L32-i32-N1024` | - | - | 0.01x | 0.65x | 1.14x | - | 0.69x |
| `p1000-L64-i8-N1024` | - | - | 0.01x | 0.81x | 1.14x | - | 0.78x |
| `p1000-L64-i32-N1024` | - | - | 0.02x | 0.80x | 1.09x | - | 0.77x |
| `p16-L64-i32-N64` | - | - | 0.38x | 80.88x | 0.34x | 0.43x | 0.60x |
| `p64-L64-i32-N64` | - | - | 0.06x | 0.62x | 0.25x | 0.44x | 0.61x |
| `p256-L64-i32-N64` | - | - | 0.03x | 0.66x | 2.24x | - | 0.68x |
| `p1000-L64-i32-N64` | - | - | 0.01x | 0.71x | 1.44x | - | 0.68x |

## Cold compile (excluded from medians)

The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is quoted here next to the warm median it must never be compared with.

| cell | arm | first_call_ms | compile_misses_first_call | warm median_ms |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | evoxc-auto | 86.789 | 13 | 2.468 |
| `p16-L32-i8-N1024` | evoxc-auto-iree | 3665.534 | 13 | 3.864 |
| `p16-L32-i8-N1024` | evoxc-specialized-numpy | 86.399 | 13 | 2.459 |
| `p16-L32-i8-N1024` | evoxc-specialized-iree | 3728.177 | 13 | 3.978 |
| `p16-L32-i8-N1024` | evoxc-dense-iree | 3368.401 | 1 | 0.974 |
| `p16-L32-i32-N1024` | evoxc-auto | 86.576 | 13 | 2.476 |
| `p16-L32-i32-N1024` | evoxc-auto-iree | 3721.258 | 13 | 4.060 |
| `p16-L32-i32-N1024` | evoxc-specialized-numpy | 86.082 | 13 | 2.567 |
| `p16-L32-i32-N1024` | evoxc-specialized-iree | 3719.149 | 13 | 3.879 |
| `p16-L32-i32-N1024` | evoxc-dense-iree | 3400.883 | 1 | 0.941 |
| `p16-L64-i8-N1024` | evoxc-auto | 86.468 | 13 | 2.299 |
| `p16-L64-i8-N1024` | evoxc-auto-iree | 3654.375 | 13 | 3.911 |
| `p16-L64-i8-N1024` | evoxc-specialized-numpy | 89.795 | 13 | 2.312 |
| `p16-L64-i8-N1024` | evoxc-specialized-iree | 3646.269 | 13 | 3.880 |
| `p16-L64-i8-N1024` | evoxc-dense-iree | 6513.143 | 1 | 1.456 |
| `p16-L64-i32-N1024` | evoxc-auto | 84.929 | 13 | 2.627 |
| `p16-L64-i32-N1024` | evoxc-auto-iree | 3725.044 | 13 | 3.937 |
| `p16-L64-i32-N1024` | evoxc-specialized-numpy | 87.562 | 13 | 2.607 |
| `p16-L64-i32-N1024` | evoxc-specialized-iree | 3706.924 | 13 | 3.801 |
| `p16-L64-i32-N1024` | evoxc-dense-iree | 6504.166 | 1 | 1.447 |
| `p64-L32-i8-N1024` | evoxc-auto | 3421.932 | 1 | 0.924 |
| `p64-L32-i8-N1024` | evoxc-auto-iree | 3431.174 | 1 | 0.971 |
| `p64-L32-i8-N1024` | evoxc-specialized-numpy | 233.628 | 57 | 9.150 |
| `p64-L32-i8-N1024` | evoxc-specialized-iree | 15743.470 | 57 | 15.451 |
| `p64-L32-i8-N1024` | evoxc-dense-iree | 3447.375 | 1 | 1.029 |
| `p64-L32-i32-N1024` | evoxc-auto | 3538.638 | 1 | 0.964 |
| `p64-L32-i32-N1024` | evoxc-auto-iree | 3411.362 | 1 | 0.986 |
| `p64-L32-i32-N1024` | evoxc-specialized-numpy | 236.877 | 57 | 9.969 |
| `p64-L32-i32-N1024` | evoxc-specialized-iree | 15939.004 | 57 | 15.522 |
| `p64-L32-i32-N1024` | evoxc-dense-iree | 3603.999 | 1 | 0.950 |
| `p64-L64-i8-N1024` | evoxc-auto | 6810.890 | 1 | 1.530 |
| `p64-L64-i8-N1024` | evoxc-auto-iree | 6384.062 | 1 | 1.528 |
| `p64-L64-i8-N1024` | evoxc-specialized-numpy | 241.528 | 57 | 10.183 |
| `p64-L64-i8-N1024` | evoxc-specialized-iree | 15724.385 | 57 | 15.640 |
| `p64-L64-i8-N1024` | evoxc-dense-iree | 6435.226 | 1 | 1.535 |
| `p64-L64-i32-N1024` | evoxc-auto | 6543.487 | 1 | 1.551 |
| `p64-L64-i32-N1024` | evoxc-auto-iree | 6493.631 | 1 | 1.651 |
| `p64-L64-i32-N1024` | evoxc-specialized-numpy | 234.982 | 57 | 10.145 |
| `p64-L64-i32-N1024` | evoxc-specialized-iree | 15925.829 | 57 | 15.584 |
| `p64-L64-i32-N1024` | evoxc-dense-iree | 6629.346 | 1 | 1.514 |
| `p256-L32-i8-N1024` | evoxc-auto | 3417.929 | 1 | 1.101 |
| `p256-L32-i8-N1024` | evoxc-auto-iree | 3548.231 | 1 | 1.122 |
| `p256-L32-i8-N1024` | evoxc-specialized-numpy | 886.130 | 212 | 40.451 |
| `p256-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i8-N1024` | evoxc-dense-iree | 3394.107 | 1 | 1.103 |
| `p256-L32-i32-N1024` | evoxc-auto | 3376.986 | 1 | 1.133 |
| `p256-L32-i32-N1024` | evoxc-auto-iree | 3412.337 | 1 | 1.127 |
| `p256-L32-i32-N1024` | evoxc-specialized-numpy | 840.713 | 219 | 39.304 |
| `p256-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L32-i32-N1024` | evoxc-dense-iree | 3414.225 | 1 | 1.133 |
| `p256-L64-i8-N1024` | evoxc-auto | 6389.165 | 1 | 1.977 |
| `p256-L64-i8-N1024` | evoxc-auto-iree | 6564.834 | 1 | 2.000 |
| `p256-L64-i8-N1024` | evoxc-specialized-numpy | 848.893 | 212 | 38.968 |
| `p256-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i8-N1024` | evoxc-dense-iree | 6523.489 | 1 | 1.999 |
| `p256-L64-i32-N1024` | evoxc-auto | 6499.446 | 1 | 2.046 |
| `p256-L64-i32-N1024` | evoxc-auto-iree | 6592.167 | 1 | 2.107 |
| `p256-L64-i32-N1024` | evoxc-specialized-numpy | 842.444 | 219 | 39.434 |
| `p256-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N1024` | evoxc-dense-iree | 6575.615 | 1 | 2.160 |
| `p1000-L32-i8-N1024` | evoxc-auto | 3449.154 | 1 | 1.903 |
| `p1000-L32-i8-N1024` | evoxc-auto-iree | 3481.104 | 1 | 1.914 |
| `p1000-L32-i8-N1024` | evoxc-specialized-numpy | 3336.236 | 797 | 152.270 |
| `p1000-L32-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i8-N1024` | evoxc-dense-iree | 3496.294 | 1 | 1.906 |
| `p1000-L32-i32-N1024` | evoxc-auto | 3356.472 | 1 | 1.974 |
| `p1000-L32-i32-N1024` | evoxc-auto-iree | 3498.341 | 1 | 1.987 |
| `p1000-L32-i32-N1024` | evoxc-specialized-numpy | 3336.237 | 821 | 155.060 |
| `p1000-L32-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L32-i32-N1024` | evoxc-dense-iree | 3532.528 | 1 | 2.031 |
| `p1000-L64-i8-N1024` | evoxc-auto | 6851.141 | 1 | 5.510 |
| `p1000-L64-i8-N1024` | evoxc-auto-iree | 6676.701 | 1 | 5.537 |
| `p1000-L64-i8-N1024` | evoxc-specialized-numpy | 3362.619 | 797 | 154.152 |
| `p1000-L64-i8-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i8-N1024` | evoxc-dense-iree | 6652.726 | 1 | 5.535 |
| `p1000-L64-i32-N1024` | evoxc-auto | 6586.623 | 1 | 5.665 |
| `p1000-L64-i32-N1024` | evoxc-auto-iree | 6554.845 | 1 | 5.668 |
| `p1000-L64-i32-N1024` | evoxc-specialized-numpy | 3359.268 | 821 | 155.934 |
| `p1000-L64-i32-N1024` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N1024` | evoxc-dense-iree | 6663.352 | 1 | 5.744 |
| `p16-L64-i32-N64` | evoxc-auto | 70.845 | 13 | 2.380 |
| `p16-L64-i32-N64` | evoxc-auto-iree | 3582.415 | 13 | 3.874 |
| `p16-L64-i32-N64` | evoxc-specialized-numpy | 68.670 | 13 | 2.397 |
| `p16-L64-i32-N64` | evoxc-specialized-iree | 3601.541 | 13 | 3.849 |
| `p16-L64-i32-N64` | evoxc-dense-iree | 6494.950 | 1 | 1.464 |
| `p64-L64-i32-N64` | evoxc-auto | 6821.652 | 1 | 1.435 |
| `p64-L64-i32-N64` | evoxc-auto-iree | 6571.364 | 1 | 1.441 |
| `p64-L64-i32-N64` | evoxc-specialized-numpy | 211.561 | 57 | 9.532 |
| `p64-L64-i32-N64` | evoxc-specialized-iree | 14984.056 | 57 | 15.502 |
| `p64-L64-i32-N64` | evoxc-dense-iree | 6450.334 | 1 | 1.425 |
| `p256-L64-i32-N64` | evoxc-auto | 6790.567 | 1 | 1.464 |
| `p256-L64-i32-N64` | evoxc-auto-iree | 7117.313 | 1 | 1.534 |
| `p256-L64-i32-N64` | evoxc-specialized-numpy | 955.806 | 219 | 41.083 |
| `p256-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p256-L64-i32-N64` | evoxc-dense-iree | 6701.077 | 1 | 1.462 |
| `p1000-L64-i32-N64` | evoxc-auto | 6919.628 | 1 | 1.582 |
| `p1000-L64-i32-N64` | evoxc-auto-iree | 6561.672 | 1 | 1.549 |
| `p1000-L64-i32-N64` | evoxc-specialized-numpy | 3252.223 | 821 | 139.402 |
| `p1000-L64-i32-N64` | evoxc-specialized-iree | - | - | - |
| `p1000-L64-i32-N64` | evoxc-dense-iree | 6735.171 | 1 | 1.571 |

### Digest agreement

Distinct `digest.sha256` values per cell and the arms that share them, for the two value paths (`batch_forward`, `forward`); a single shared sha per cell is the expected, numerically-consistent case.

| cell | workload | sha256 | arms | nan |
|---|---|---|---|---|
| `p16-L32-i8-N1024` | batch_forward | `ba1157e91cb09e697e821bf4caea5ee7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i8-N1024` | batch_forward | `ccd9945407c53f1df82b795467c23061` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L32-i8-N1024` | forward | `c21e5a57d58db557165cc13d515f788f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `ae02cdc9437e3c169a407657a6cb212f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | batch_forward | `7c0dcb871d22ca8776d532411a044898` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L32-i32-N1024` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L32-i32-N1024` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `ba1157e91cb09e697e821bf4caea5ee7` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i8-N1024` | batch_forward | `ccd9945407c53f1df82b795467c23061` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i8-N1024` | forward | `c21e5a57d58db557165cc13d515f788f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-specialized-iree, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `ae02cdc9437e3c169a407657a6cb212f` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | batch_forward | `7c0dcb871d22ca8776d532411a044898` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i32-N1024` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N1024` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `960a58af4aeb7d6a855d90660a333509` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | batch_forward | `d1fd2ddbb2317ca93f683e184057c988` | evoxc-specialized-iree | 0 |
| `p64-L32-i8-N1024` | forward | `4cfeb51431d33041bc2c71f2db3b1417` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i8-N1024` | forward | `b5bf83cf773664021af5c74b9252a82e` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `f15f0b65cc977adc7eaa17ae644d9ea5` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | batch_forward | `d44d3e2c6ac92fe124599c5d5baa529d` | evoxc-specialized-iree | 0 |
| `p64-L32-i32-N1024` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L32-i32-N1024` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `960a58af4aeb7d6a855d90660a333509` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i8-N1024` | batch_forward | `d1fd2ddbb2317ca93f683e184057c988` | evoxc-specialized-iree | 0 |
| `p64-L64-i8-N1024` | forward | `4cfeb51431d33041bc2c71f2db3b1417` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i8-N1024` | forward | `b5bf83cf773664021af5c74b9252a82e` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `f15f0b65cc977adc7eaa17ae644d9ea5` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | batch_forward | `d44d3e2c6ac92fe124599c5d5baa529d` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N1024` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N1024` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p256-L32-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L32-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L32-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L32-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i8-N1024` | batch_forward | `d31bf4ba4573d2f7bc2820f95d8a6a74` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1024 |
| `p256-L64-i8-N1024` | forward | `e521e515e93b256615982913b0b4df3d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 1 |
| `p256-L64-i32-N1024` | batch_forward | `b84773c14f5546a004261def2c0b0680` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N1024` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p1000-L32-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L32-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L32-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L32-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L32-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L32-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3072 |
| `p1000-L32-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L32-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |
| `p1000-L64-i8-N1024` | batch_forward | `bddcff49d3e50ea78dabc501ea09fcb7` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7168 |
| `p1000-L64-i8-N1024` | batch_forward | `33710b1a3d089a4308bfa4805a58744e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7168 |
| `p1000-L64-i8-N1024` | forward | `bd4e528009f2367cca91698767e515e1` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 7 |
| `p1000-L64-i8-N1024` | forward | `dfe13d17a38ded48af5c1f1530ce7520` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 7 |
| `p1000-L64-i32-N1024` | batch_forward | `bf68ce503ba6d209acb79440cf42c0b8` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3072 |
| `p1000-L64-i32-N1024` | batch_forward | `31fa45e297dabcda8cc1442efba0518a` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3072 |
| `p1000-L64-i32-N1024` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N1024` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |
| `p16-L64-i32-N64` | batch_forward | `ac20d6b9b2fc89394ed1f5bb092e351e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | batch_forward | `3e8dc4e8aabd0c155e65667c48b43645` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p16-L64-i32-N64` | forward | `d38a87a0bafc1c570902a04698064e3e` | torch-per-tree, torch-batched, evoxc-auto, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p16-L64-i32-N64` | forward | `fd55f38bb508bfe8fa52eac1a50c6dde` | evoxc-auto-iree, evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `91f8f057103e5fc3760df30ab1e15987` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | batch_forward | `0f0607d7f450ff24c5af100c2d775332` | evoxc-specialized-iree | 0 |
| `p64-L64-i32-N64` | forward | `e15e37724af7f03c85cac50d7df1572d` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p64-L64-i32-N64` | forward | `b3493af3fe39172e6b1c34b93ba2c69c` | evoxc-specialized-iree | 0 |
| `p256-L64-i32-N64` | batch_forward | `5ddefa134a076abce9faa669870b2558` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p256-L64-i32-N64` | forward | `62f1e4def9378b547741dbc8651ac4c0` | torch-per-tree, torch-batched, evoxc-auto, evoxc-auto-iree, evoxc-specialized-numpy, evoxc-dense-iree | 0 |
| `p1000-L64-i32-N64` | batch_forward | `6891f91b5d52caf1d0ca046f766bf963` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 192 |
| `p1000-L64-i32-N64` | batch_forward | `b603b9032571ea2fc133e7985e03390e` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 192 |
| `p1000-L64-i32-N64` | forward | `c9a313ff37e5805754351d81dcfc628d` | torch-per-tree, torch-batched, evoxc-specialized-numpy | 3 |
| `p1000-L64-i32-N64` | forward | `3b46644ea4d142c4fd818ac0b37313ba` | evoxc-auto, evoxc-auto-iree, evoxc-dense-iree | 3 |

`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), so they are not compared across arms; distinct/records per cell: p16-L32-i8-N1024: 5/7, p16-L32-i32-N1024: 5/7, p16-L64-i8-N1024: 5/7, p16-L64-i32-N1024: 5/7, p64-L32-i8-N1024: 5/7, p64-L32-i32-N1024: 5/7, p64-L64-i8-N1024: 5/7, p64-L64-i32-N1024: 5/7, p256-L32-i8-N1024: 4/6, p256-L32-i32-N1024: 4/6, p256-L64-i8-N1024: 4/6, p256-L64-i32-N1024: 4/6, p1000-L32-i8-N1024: 4/6, p1000-L32-i32-N1024: 4/6, p1000-L64-i8-N1024: 4/6, p1000-L64-i32-N1024: 4/6, p16-L64-i32-N64: 2/7, p64-L64-i32-N64: 3/7, p256-L64-i32-N64: 2/6, p1000-L64-i32-N64: 3/6.

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
