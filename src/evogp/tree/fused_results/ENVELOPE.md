# Fused (Triton) route — envelope / min-pop / geometry campaign record

Records of ONE run of `src/evogp/tree/fused_envelope.py` (driver `--stage headline | envelope |
minpop | sweep | degenerate`). The JSON evidence sits beside this file:
`main/headline_20261030T212347Z.json`, `envelope.json`, `minpop.json`, `sweep/sweep.json`,
`degenerate.json`.

## Protocol and environment

- Host: AMD MI300X (ROCm), torch 2.13.0+rocm7.1; `evoxc` from the `/root/evoxc` checkout.
- Every timing config ran in its OWN process (`python -m evogp.tree.fused_envelope --stage
  <x>-one`), strictly sequential — no two GPU configs ever ran concurrently. Within one cell
  the arms (`fused` / `torch-batched` / `specialized` / `dense`) ran sequentially in that one
  process.
- Warm medians: 2 untimed warmups + 5 sync-bracketed reps (`common.time_call`); the cold first
  call is recorded separately (`first_call_ms`) and never enters the median.
- Geometry: the shipped defaults — `EVOGP_FUSED_BLOCK_ROWS` 256, `EVOGP_FUSED_NUM_WARPS` 1,
  `EVOGP_FUSED_MAX_SCRATCH_MB` 8192, `EVOGP_FUSED_MIN_POP` 64 — except in `sweep`, which sets
  them per config in the child environment. No default knob was changed by this campaign.
- Reproduce (repo root, one stage at a time):

```
TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache \
PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \
    /tmp/rocvenv/bin/python -m evogp.tree.fused_envelope --stage headline
    ... --stage envelope | minpop | sweep | degenerate
```

- All five stages exited 0 with `failures: 0` in every record file.
- `fused.served_via` is `public_api` when the route policy serves fused for the cell and
  `direct` when `pop_size < EVOGP_FUSED_MIN_POP` (the policy answers `specialized` there, so the
  kernel is asked through `evaluate_fused` / `sr_fitness_fused` and the record carries the
  policy's own answer under `fused.policy`).

## 1. Headline (six cells of FUSED.md, public API, shipped 256/1)

| cell | value | loss | per_row | arm (torch-batched) | ratio | kernel (`fused_direct`) | recorded kernel | Δ kernel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p256-L32-i8-N1024 | 0.101 | 0.116 | 0.082 | 3.937 | 38.90 | 0.080 | 0.083 | −3.6 % |
| p256-L64-i32-N1024 | 0.106 | 0.115 | 0.090 | 4.441 | 42.02 | 0.082 | 0.082 | −0.3 % |
| p256-L64-i32-N4096 | 0.121 | 0.126 | 0.092 | 5.247 | 43.35 | 0.105 | 0.097 | +8.4 % |
| p1000-L64-i32-N1024 | 0.117 | 0.122 | 0.086 | 5.375 | 45.85 | 0.101 | 0.096 | +5.0 % |
| p4096-L64-i32-N1024 | 0.143 | 0.154 | 0.102 | 10.628 | 74.21 | 0.127 | 0.134 | −5.5 % |
| p2048-L128-i32-N1024-d7 | 0.176 | 0.191 | 0.125 | 27.831 | 157.80 | 0.155 | 0.161 | −3.8 % |

ms, warm medians. `value` = `Forest.batch_forward`, `loss` = `Forest.SR_fitness`,
`per_row` = `Forest.forward` (1 input row per tree), `kernel` = the facade without the route
policy (`fused_direct`), "recorded kernel" = the per-cell numbers of the previous campaign.
No cell regresses more than 10 % (worst: +8.4 %); the public-API column sits +7…+29 % above the
kernel column, i.e. the facade + route decision + the `(pop, rows, 1)` staging cost 20–24 µs.

## 2. Envelope (12 cells; fused vs torch-batched / specialized / dense)

| cell | fused value | fused loss | arm | specialized | dense | arm/fused | spec/fused | dense/fused | scratch bytes | ok | n_rb | peak alloc / reserved MiB | value dev vs arm | loss dev vs arm | dense status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p16-L64-i32-N1024 | 0.088 | 0.099 | 3.162 | 2.845 | 1.474 | 35.81 | 32.22 | 16.70 | 4194304 | True | 4 | 4.39 / 22.0 | 0 (bitwise) | 1.59e-07 | served |
| p64-L64-i32-N1024 | 0.107 | 0.138 | 4.046 | 12.088 | 1.753 | 37.69 | 112.62 | 16.33 | 16777216 | True | 4 | 16.80 / 18.0 | 0 (bitwise) | 2.29e-07 | served |
| p256-L64-i32-N1024 | 0.102 | 0.123 | 4.809 | 45.587 | 2.889 | 47.29 | 448.29 | 28.41 | 67108864 | True | 4 | 66.42 / 68.0 | 0 (bitwise) | 2.29e-07 | served |
| p1000-L64-i32-N1024 | 0.136 | 0.122 | 5.367 | 179.966 | 7.730 | 39.46 | 1323.36 | 56.84 | 262144000 | True | 4 | 258.68 / 274.0 | 0 (bitwise) | 3.19e-07 | served |
| p256-L64-i32-N4096 | 0.126 | 0.121 | 5.203 | 49.752 | 7.855 | 41.43 | 396.16 | 62.55 | 268435456 | True | 16 | 265.18 / 278.0 | 0 (bitwise) | 2.55e-07 | served |
| p256-L64-i32-N16384 | 0.164 | 0.171 | 10.257 | 110.135 | 80.996 | 62.68 | 673.02 | 494.96 | 1073741824 | True | 64 | 1060.22 / 1062.0 | 0 (bitwise) | 2.82e-07 | served |
| p1000-L64-i32-N4096 | 0.188 | 0.193 | 10.221 | 197.589 | 49.170 | 54.46 | 1052.70 | 261.96 | 1048576000 | True | 16 | 1033.63 / 1036.0 | 0 (bitwise) | 3.89e-07 | served |
| p2048-L64-i32-N1024 | 0.121 | 0.138 | 7.122 | 369.162 | 14.561 | 59.00 | 3058.48 | 120.63 | 536870912 | True | 4 | 529.51 / 538.0 | 0 (bitwise) | 3.78e-07 | served |
| p4096-L64-i32-N1024 | 0.156 | 0.157 | 10.531 | 720.569 | 46.933 | 67.63 | 4627.04 | 301.38 | 1073741824 | True | 4 | 1058.76 / 1072.0 | 0 (bitwise) | 3.78e-07 | served |
| p8192-L64-i32-N1024 | 0.218 | 0.201 | 17.673 | 33904.431 | 95.973 | 81.24 | 155846.6 | 441.15 | 2147483648 | True | 4 | 2117.26 / 2142.0 | 0 (bitwise) | 4.05e-07 | served |
| p2048-L128-i32-N1024-d7 | 0.191 | 0.189 | 27.520 | 888.204 | n/a | 143.91 | 4644.74 | n/a | 1073741824 | True | 4 | 1042.51 / 1056.0 | 0 (bitwise) | 3.13e-07 | dense_expected_decline |
| p2048-L256-i32-N1024-d8 | 0.238 | 0.233 | 49.013 | 1398.001 | n/a | 206.01 | 5876.08 | n/a | 2147483648 | True | 4 | 2068.51 / 2096.0 | 0 (bitwise) | 3.41e-07 | dense_expected_decline |

ms, warm medians; `arm/fused` etc. are the reference's cost divided by fused's (higher = fused
wins). `dense_expected_decline` = `tree_len` above the base dense trace cap (64), where an
explicit `dense` request falls back to `specialized` and timing it would report the fallback.

- **Memory case** `p2048-L256-i32-N1024-d8`: scratch by the envelope formula 2 147 483 648 B
  (2 GiB), `envelope_ok` True against `EVOGP_FUSED_MAX_SCRATCH_MB` 8192; the ACTUAL device peak
  over the fused window was **2068.51 MiB allocated / 2096.0 MiB reserved** (delta 2064.13 /
  2048.0), i.e. the full 2 GiB block is really allocated and the call does not OOM.
- No cell declined and no cell OOMed. Peak allocated tracks `scratch_bytes` to within 3.5 %
  (4.39 / 16.80 / 66.42 / 258.68 / 265.18 / 1060.22 / 1033.63 / 529.51 / 1058.76 / 2117.26 /
  1042.51 / 2068.51 MiB against 4 / 16 / 64 / 250 / 256 / 1024 / 1000 / 512 / 1024 / 2048 /
  1024 / 2048 MiB of scratch; the surplus is the `(pop, rows, 1)` value buffer), so the envelope
  formula `pop * tree_len * rows * 4` is the memory story, not an estimate.
- Value parity vs the arm is bitwise everywhere (`max_abs` 0, `max_rel` 0); loss parity is
  1.6e-07…4.1e-07 relative (in-graph reduction order).

## 3. Min-pop column (L=64 / i32 / N=1024)

| pop | fused value (`fused_direct`) | fused loss | fused API (`Forest.*`) | specialized | torch-batched | fused vs torch | fused vs specialized | policy engine (`EVOGP_EVAL_MODE=fused`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.093 | 0.111 | 0.482 | 0.490 | 1.396 | 15.05 | 5.28 | specialized |
| 2 | 0.088 | 0.110 | 0.640 | 0.599 | 1.609 | 18.26 | 6.80 | specialized |
| 4 | 0.092 | 0.111 | 0.939 | 0.925 | 1.951 | 21.28 | 10.09 | specialized |
| 8 | 0.159 * | 0.130 | 1.655 | 1.644 | 2.662 | 16.72 | 10.32 | specialized |
| 16 | 0.103 | 0.119 | 2.941 | 2.725 | 3.367 | 32.69 | 26.46 | specialized |
| 32 | 0.097 | 0.124 | 5.928 | 5.679 | 3.954 | 40.70 | 58.46 | specialized |
| 64 | 0.097 | 0.121 | 0.118 | 12.154 | 4.233 | 43.78 | 125.69 | fused |
| 128 | 0.092 | 0.111 | 0.100 | 22.536 | 4.110 | 44.63 | 244.72 | fused |

\* the pop-8 record is a clock-ramp artifact: its samples are
`[0.1724, 0.1581, 0.1710, 0.1592, 0.0941]` (min 0.0941, in line with every other pop); the
median landed on the slow tail. Everything else in the column is 0.088–0.103 ms, i.e. the
kernel is FLAT in `pop_size` at this shape — its cost is the row-block sweep, not the trees.
`fused API` below the gate equals `specialized` by construction (the policy answers
specialized), so `EVOGP_FUSED_MIN_POP` = 64 is the only reason pop 1…32 pay 5–58× more.

## 4. Geometry sweep (`EVOGP_FUSED_BLOCK_ROWS` × `EVOGP_FUSED_NUM_WARPS`)

Warm medians, ms (value / loss). `worst` = that config's slowest cell relative to the cell's
best config (medians; a min-based `worst` in parentheses).

| cfg | p256-L64-i32-N1024 | p4096-L64-i32-N1024 | p256-L64-i32-N4096 | p2048-L128-i32-N1024-d7 | worst |
| --- | --- | --- | --- | --- | --- |
| B128-W1 | 0.088 / 0.115 | 0.161 / 0.168 | 0.109 / 0.124 | 0.204 / 0.197 | 1.27 (1.26) |
| B128-W2 | 0.098 / 0.103 | 0.240 / 0.255 | 0.125 / 0.138 | 0.309 / 0.310 | 1.94 (1.96) |
| B128-W4 | 0.101 / 0.122 | 0.370 / 0.383 | 0.168 / 0.169 | 0.469 / 0.479 | 2.93 (3.04) |
| B256-W1 (shipped) | 0.096 / 0.120 | 0.137 / 0.144 | 0.113 / 0.111 | 0.160 / 0.173 | 1.20 (1.10) |
| B256-W2 | 0.090 / 0.115 | 0.149 / 0.161 | 0.108 / 0.122 | 0.190 / 0.203 | 1.19 (1.21) |
| B256-W4 | 0.092 / 0.112 | 0.231 / 0.244 | 0.121 / 0.132 | 0.284 / 0.302 | 1.78 (1.82) |
| B512-W1 | 0.102 / 0.114 | 0.154 / 0.163 | 0.110 / 0.114 | 0.161 / 0.172 | 1.26 (1.19) |
| B512-W2 | 0.090 / 0.112 | 0.139 / 0.136 | 0.107 / 0.105 | 0.160 / 0.175 | 1.12 (1.13) |
| B512-W4 | 0.099 / 0.116 | 0.159 / 0.164 | 0.112 / 0.117 | 0.181 / 0.194 | 1.24 (1.16) |
| B1024-W1 | 0.094 / 0.110 | 0.170 / 0.154 | 0.101 / 0.108 | 0.174 / 0.179 | 1.24 (1.18) |
| B1024-W2 | 0.096 / 0.116 | 0.157 / 0.152 | 0.114 / 0.119 | 0.165 / 0.171 | 1.19 (1.16) |
| B1024-W4 | 0.081 / 0.114 | 0.140 / 0.149 | 0.108 / 0.105 | 0.164 / 0.166 | 1.07 (1.04) |

Per-cell best (medians): 0.081 / 0.137 / 0.101 / 0.160. The shipped 256/1 is the outright best
on both large-population cells (1.00), 1.12 on the row cell and 1.20 on `p256-L64-i32-N1024`
(1.10 on mins). The systematic, beyond-noise loser is a 4-warp block on 128 lanes (B128-W4:
1.67–3.04× on every large cell); everything else lies inside the protocol's noise band — the
within-record spread (`max−min)/median`) is 2–20 % at the large cells and up to 52 % at
`p256-L64-i32-N1024`, which is larger than the gaps between the top four configs.

## 5. Degenerate shapes

| case | cell | fused value | fused loss | fused per_row | arm | oracle value dev | oracle loss dev | oracle per_row dev | arm value dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rows_1 | p256-L64-i32-N1 | 0.099 | 0.136 | 0.088 | 4.433 | 0 (bitwise) | 0 | 0 (bitwise) | 0 (bitwise) |
| rows_1000 | p256-L64-i32-N1000 | 0.131 | 0.125 | 0.091 | 4.675 | 0 (bitwise) | 2.51e-07 | 0 (bitwise) | 0 (bitwise) |
| input_len_1 | p256-L64-i1-N1024 | 0.114 | 0.129 | 0.091 | 4.431 | 0 (NaN-strict flag) | 2.88e-07 | 0 | 0 |
| all_const_pop | p256-L64-i32-N1024 | 0.107 | 0.123 | 0.097 | 4.387 | 0 (bitwise) | 3.43e-07 | 0 (bitwise) | 0 (bitwise) |
| single_node_pop | p256-L64-i32-N1024 | 0.107 | 0.121 | 0.092 | 4.545 | 0 (bitwise) | 3.43e-07 | 0 (bitwise) | 0 (bitwise) |

No failure in any case: rows=1, the non-multiple-of-256 row count (n_rb 4 with a masked tail),
`input_len` 1, an all-constant population and a single-node population all serve one kernel per
call and agree with `reference_eval` to 0 absolute deviation on values (≤ 3.4e-07 relative on
the loss, which folds in-graph).

`input_len_1` reports `bitwise_equal: False` with `max_abs = max_rel = 0` and
`nan_mismatch = 0`: the cell contains 11 264 NaN points and `validate.compare`'s bitwise check
is NaN-strict (`torch.equal` is False whenever ANY position is NaN, even when both sides agree
there), so the flag is a property of the comparator, not a numeric difference.

## 6. Notes for a reader

- The kernel's floor is ~0.088 ms and its whole measured range is 0.088–0.238 ms over a
  2 048× population range and a 16 384× row range (`p16/L=64/N=1024` → `p8192/L=64/N=1024`),
  so it is launch/latency-bound, not bandwidth-bound, in this column.
- Against the strongest reference arm on the small-population cells (`dense`, whose cost is
  shape-bound rather than population-bound) the narrowest margin measured is **16.3×** at
  `p64-L64-i32-N1024` and 16.7× at `p16`. Nothing in this column loses.
- The facade costs 20–24 µs per call over the kernel (`value` 0.101–0.176 vs `kernel`
  0.080–0.155 in §1; `fused api` vs `fused direct` in §3), i.e. ~20 % of a small-population call
  and ~7–10 % at pop ≥ 4 096.
- The 2-warmup/5-rep protocol is honest but noisy at these magnitudes (±10–20 %, more at
  `p256-L64-i32-N1024`); a comparison closer than ~15 % needs more reps than this campaign ran.
