# Round 2, work item 2 — `EVOGP_BATCH_BANDS=auto` (cost model + legality clamp) measured against `b16` / `b32`

**Question.** Round 2 work item 1 established that the band *size* is a second-order knob among banded
merges (3–6 %, i.e. at this box's cross-config noise floor) and that band size 16 is the minimax setting
on that power-of-two grid. This round measures the other half of the same knob table: `EVOGP_BATCH_BANDS=auto`,
where evoxc's static cost model picks the band count `K` and the binding policy may then **clamp** it
(`K ≤ 15` because one exported dispatch binds at most 16 tensors on the ROCm/HIP HAL; `≥ 8` merged
functions per band). What does the model actually ask for on these cells, what does the clamp do to it,
what does the clamp cost, and is `auto` within noise of the per-cell argmin?

**Answer (short, numbers only — the decision belongs to the report of record).** The model **overshoots
14.2–14.7×** on the wide-row (loss / population-batched value) plans at pop 256 and 2.3× at pop 64 — it
asks for 213/220 bands at the pop-256 cells and 16 at the pop-64 cell, and is clamped to **15** (pop-256)
and **7** (pop-64, where the `≥ 8`-functions-per-band rule binds, not the binding cap) — and it
**undershoots to a bare 4 bands** (pop 256) or **1 band, i.e. the flat merge** (pop 64) on the per-row
`forward` plan, where it is not clamped at all. The clamp's second merge lands entirely in the `merge`
stage and costs **+0.18 to +1.11 s** of host merge per cold call (3.0–5.4× the single merge of the same
final shape). Warm, `auto` **never wins a column (0/9)**; it is inside the 2.6–6 % noise band in 4 of 9
columns and outside it in 5
(worst +18.9 % on the band-sensitive `forward` at `p256-L32`). `b16` and `b32` trade columns 5 : 4, and
neither dominates the other warm or cold.

## Scope and protocol

| axis | values |
|---|---|
| cells | `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p64-L64-i32-N64` (seed 0, `max_layer_cnt` 5, `+ - * /`) |
| configs | `auto` (`EVOGP_BATCH_BANDS=auto`: evoxc cost model + legality clamp), `b16` / `b32` (pin `EVOGP_BATCH_BAND_SIZE`, so `K = ceil(n_functions / size)`) |
| workloads | `sr_fitness` (loss, N = 1024 / 64 rows), `batch_forward` (whole population over N rows), `forward` (one row per tree) — in that fixed order, one plan + one compile each |
| pool width | `EVOGP_BATCH_COMPILE_WORKERS=1` for **every** config (single body, no chunking), so a difference between two configs of a cell is the band partition and the merge alone |
| engine | `EVOGP_EVAL_MODE=batch`, `EVOGP_EVOXC_BACKEND=iree`, evoxc snapshot `/tmp/evoxc_r2`, `EVOXC_DISABLE_LOCATIONS=1`, `PYTHONPATH=/tmp/evoxc_r2:src:test/benchmark/evoxc` (never `/root/evoxc`) |
| protocol | `--warmup 1 --reps 5`, warm medians, the untimed cold first call separated (it carries the merged-graph compile), one child process per `(cell, config)`, one config at a time behind the idle-device guard |
| env identity | evogp tree `db9e25d` (worktree), evoxc `2b289f0`, torch 2.13.0+rocm7.1, hip 7.1.52802, iree 3.11.0, AMD Instinct MI300X VF |

Command of record (run from the repository root):

```bash
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src:test/benchmark/evoxc
/tmp/rocvenv/bin/python test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --configs auto,b16,b32 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/auto \
  --tree-root src --warmup 1 --reps 5 --seed 0
```

9 `(cell, config)` children, **9 ok, 0 failed, 0 timeout, 0 skipped, 335.4 s wall** (30.2–35.2 s per
pop-256 config, 13.2–13.3 s per pop-64 config). Records: `<cell>/<config>.json` (schema
`evogp-bench-evoxc-v1-band-sweep`) + `<cell>/<config>.log`, index `sweep.json`.

## 1. What the cost model chose — the `band_report` block (`auto` rows)

`band_policy` is the same in every record of every config:
`{backend: iree, device_kind: rocm, max_bands: 15, min_band_size: 8, uncapped: false}`.
`n_functions` is the **deduped** function count and differs per plan variant (the per-row path dedupes by
`(tree, row)`); `requested K` is `band_requested_bands`, the model's answer **before** clamping.

| cell | workload | n_functions | requested K | final `bands` | `band_sizes` | clamped | `band_strategy` | `chunk_bands` | `cost_estimate` | `flat_estimate` | `emitted_ops` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 225 | **213** | 15 | 15×15 | **yes** | `explicit` | `[15]` | 7 350 691 | 466 453 471 | 5037 |
| p256-L32-i8-N1024 | batch_forward | 225 | **213** | 15 | 15×15 | **yes** | `explicit` | `[15]` | 6 425 728 | 372 544 768 | 4153 |
| p256-L32-i8-N1024 | forward | 245 | **4** | 4 | 1×62, 3×61 | no | `auto` | `[4]` | 1 231 549 | 1 375 992 | 3902 |
| p256-L64-i32-N1024 | sr_fitness | 232 | **220** | 15 | 7×16, 8×15 | **yes** | `explicit` | `[15]` | 7 780 152 | 524 411 684 | 5475 |
| p256-L64-i32-N1024 | batch_forward | 232 | **220** | 15 | 7×16, 8×15 | **yes** | `explicit` | `[15]` | 6 825 600 | 427 565 312 | 4568 |
| p256-L64-i32-N1024 | forward | 245 | **4** | 4 | 1×62, 3×61 | no | `auto` | `[4]` | 1 347 921 | 1 580 382 | 4481 |
| p64-L64-i32-N64 | sr_fitness | 63 | **16** | 7 | 7×9 | **yes** | `explicit` | `[7]` | 815 389 | 2 724 142 | 1463 |
| p64-L64-i32-N64 | batch_forward | 63 | **16** | 7 | 7×9 | **yes** | `explicit` | `[7]` | 654 592 | 2 182 976 | 1219 |
| p64-L64-i32-N64 | forward | 64 | **1** | 1 | 1×64 | no | `auto` | `[1]` | 321 984 | 321 984 | 1104 |

(`band_count` equals `bands` in every row; `(n×s)` is *count × band size*.)

Reading the block:

- **Two opposite failure modes of one model.** On the loss / population-batched value plans the model
  wants one band per function (213 of 225, 220 of 232 at pop 256 — 14.2–14.7× the count that can load;
  16 of 63 at pop 64, 2.3×) and the policy cuts it to the ceiling — 15 bands where the `K ≤ 15` binding
  cap binds, 7 at `p64` where the `≥ 8`-functions-per-band rule binds first (`max_bands` 15 would have
  allowed 7 = floor(63/8)).
  On the per-row `forward` plan the same model answers **4** (pop 256) and **1** — the *flat* merge —
  at pop 64, and is not clamped at all. So "auto" is not "a smarter constant band size": it is a
  different partition on the two plan families, and on the per-row family it lands *below* the
  banded region item 1 measured (K = 8–16).
- **The model's own cost estimates say the opposite of what measured.** For the clamped plans it
  prices the requested partition at 6.4–7.8 M against 3.7–5.2 × 10⁸ for the flat merge (a 58–67×
  preference for many tiny bands; 3.3× at `p64`), and at `p256-L32 forward` it prefers K = 4 over flat
  by only 12 % (1 231 549 vs 1 375 992; 1.2× at `p256-L64`, exactly 1.0× at `p64`) — the two estimates
  it produces are one to three orders of magnitude apart on plans whose measured warm medians differ by
  ≤ 19 %. At `p64 forward` the model chose K = 1 and `cost_estimate == flat_estimate` exactly (321 984),
  i.e. the model itself said "unbanded".
- **The clamp is a note as well as a number**: every clamped row carries `band_note` naming the
  requested count, the ceiling, the binding reason (HAL bind limit / minimum band size) and the
  remedy (`EVOGP_BATCH_BANDS=K` with `K ≤ cap`). The unclamped rows carry `band_note: null`.

## 2. Table 1 — cold first call + warm median per `(cell, workload, config)`, and the column's winner

Cold = `first_call_ms` (the merged-graph compile of that workload's plan, never part of a warm median).
Warm = `median_ms` over the 5 samples, `min_ms` in parentheses. `n×s` = count × band size.

| cell | workload | config | `bands` | `band_sizes` | requested K | clamped | cold ms | warm ms (min) | winner |
|---|---|---|---|---|---|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | `auto` | 15 | 15×15 | 213 | **yes** | 12 429 | 2.0621 (2.0573) | |
| p256-L32-i8-N1024 | sr_fitness | `b16` | 15 | 15×15 | — | no | 11 049 | **2.0331 (1.9931)** | **←** |
| p256-L32-i8-N1024 | sr_fitness | `b32` | 8 | 1×29, 7×28 | — | no | 12 063 | 2.1666 (2.1349) | |
| p256-L32-i8-N1024 | batch_forward | `auto` | 15 | 15×15 | 213 | **yes** | 8 543 | 2.4180 (2.3535) | |
| p256-L32-i8-N1024 | batch_forward | `b16` | 15 | 15×15 | — | no | 7 496 | **2.2305 (2.1528)** | **←** |
| p256-L32-i8-N1024 | batch_forward | `b32` | 8 | 1×29, 7×28 | — | no | 7 887 | 2.3063 (2.2512) | |
| p256-L32-i8-N1024 | forward | `auto` | 4 | 1×62, 3×61 | 4 | no | 4 161 | 1.0262 (1.0145) | |
| p256-L32-i8-N1024 | forward | `b16` | 15 | 5×17, 10×16 | — | **yes** | 3 075 | **0.8632 (0.7958)** | **←** |
| p256-L32-i8-N1024 | forward | `b32` | 8 | 5×31, 3×30 | — | no | 3 216 | 0.8830 (0.8723) | |
| p256-L64-i32-N1024 | sr_fitness | `auto` | 15 | 7×16, 8×15 | 220 | **yes** | 14 138 | 2.2703 (2.2259) | |
| p256-L64-i32-N1024 | sr_fitness | `b16` | 15 | 7×16, 8×15 | — | no | 12 591 | 2.2382 (2.2102) | |
| p256-L64-i32-N1024 | sr_fitness | `b32` | 8 | 8×29 | — | no | 12 667 | **2.2008 (2.1486)** | **←** |
| p256-L64-i32-N1024 | batch_forward | `auto` | 15 | 7×16, 8×15 | 220 | **yes** | 8 395 | 2.3059 (2.2823) | |
| p256-L64-i32-N1024 | batch_forward | `b16` | 15 | 7×16, 8×15 | — | no | 7 284 | 2.2557 (2.2062) | |
| p256-L64-i32-N1024 | batch_forward | `b32` | 8 | 8×29 | — | no | 7 555 | **2.0986 (1.9891)** | **←** |
| p256-L64-i32-N1024 | forward | `auto` | 4 | 1×62, 3×61 | 4 | no | 3 890 | 0.9189 (0.8755) | |
| p256-L64-i32-N1024 | forward | `b16` | 15 | 5×17, 10×16 | — | **yes** | 3 385 | 0.8234 (0.7723) | |
| p256-L64-i32-N1024 | forward | `b32` | 8 | 5×31, 3×30 | — | no | 3 255 | **0.8028 (0.7820)** | **←** |
| p64-L64-i32-N64 | sr_fitness | `auto` | 7 | 7×9 | 16 | **yes** | 2 516 | 1.1612 (1.1311) | |
| p64-L64-i32-N64 | sr_fitness | `b16` | 4 | 3×16, 1×15 | — | no | 2 299 | **1.1305 (1.0981)** | **←** |
| p64-L64-i32-N64 | sr_fitness | `b32` | 2 | 1×32, 1×31 | — | no | 2 273 | 1.1596 (1.1266) | |
| p64-L64-i32-N64 | batch_forward | `auto` | 7 | 7×9 | 16 | **yes** | 1 626 | 1.0151 (0.9954) | |
| p64-L64-i32-N64 | batch_forward | `b16` | 4 | 3×16, 1×15 | — | no | 1 676 | **0.8899 (0.8810)** | **←** |
| p64-L64-i32-N64 | batch_forward | `b32` | 2 | 1×32, 1×31 | — | no | 1 603 | 0.9020 (0.8889) | |
| p64-L64-i32-N64 | forward | `auto` | 1 | 1×64 | 1 | no | 1 037 | 0.6872 (0.6439) | |
| p64-L64-i32-N64 | forward | `b16` | 4 | 4×16 | — | no | 1 059 | 0.7097 (0.6703) | |
| p64-L64-i32-N64 | forward | `b32` | 2 | 2×32 | — | no | 988 | **0.6681 (0.6322)** | **←** |

Two shape facts a reader needs:

- **`b16` is itself clamped on the per-row plan** (245 functions → `ceil(245/16)` = 16 bands → capped to 15,
  `band_sizes` 5×17 + 10×16, `band_clamped: true`, `band_strategy: explicit`). So on `forward` the
  `b16`-vs-`auto` comparison is *15 bands vs 4 bands*, and the `b16`-vs-`b32` one is *15 vs 8*. On the
  wide-row plans `b16`'s K = 15 is exactly where `auto`'s clamp lands, i.e. those two configs run the
  **identical merge shape** — and, at `p256-L32`, `auto` still measures 1.4–8.4 % slower warm with that
  same shape (see §4 and the contention note in §7).
- `p64-L64-i32-N64/forward`: `auto` chose **K = 1**, the flat merge, and measured 0.6872 ms — 2.9 %
  behind `b32`'s 2-band merge and 3.2 % ahead of `b16`'s 4-band one, i.e. on the cheap cell the
  flat choice is not measurably wrong even though item 1 found flat 34 % worse on the pop-256 per-row
  plans.

## 3. Table 2 — what the clamp costs, read off the `merge` stage

The clamp is not a separate stage: `batch_bands.merge_with_spec` merges once under the model's choice,
reads `merged.batch_report`, and — when the answer is unloadable — **merges a second time** at the capped
count, both calls inside the single `timer.stage("merge")` block. The stage counter therefore records
**`batch_merge_calls = 1` for every row** (it counts block entries, not inner merge calls) and the second
merge shows up as *time in that one block*. Cold first call, `stage_first_call` (ms):

| cell | workload | `auto` K (clamped?) | merge ms | `b16` K | merge ms | `b32` K | merge ms | auto − b16 | auto / b16 |
|---|---|---|---|---|---|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 15 (**yes**) | **1 219.5** | 15 | 233.1 | 8 | 230.3 | **+986.4** | **5.23×** |
| p256-L32-i8-N1024 | batch_forward | 15 (**yes**) | **1 048.9** | 15 | 204.1 | 8 | 206.9 | **+844.8** | **5.14×** |
| p256-L32-i8-N1024 | forward | 4 (no) | 785.2 | 15 | 261.3 | 8 | 253.6 | +523.9 | 3.00× |
| p256-L64-i32-N1024 | sr_fitness | 15 (**yes**) | **1 395.2** | 15 | 280.6 | 8 | 246.6 | **+1 114.5** | **4.97×** |
| p256-L64-i32-N1024 | batch_forward | 15 (**yes**) | **1 209.5** | 15 | 247.0 | 8 | 216.9 | **+962.5** | **4.90×** |
| p256-L64-i32-N1024 | forward | 4 (no) | 670.9 | 15 | 284.5 | 8 | 269.8 | +386.4 | 2.36× |
| p64-L64-i32-N64 | sr_fitness | 7 (**yes**) | **285.5** | 4 | 94.5 | 2 | 90.3 | **+191.0** | **3.02×** |
| p64-L64-i32-N64 | batch_forward | 7 (**yes**) | **221.9** | 4 | 41.4 | 2 | 38.5 | **+180.4** | **5.35×** |
| p64-L64-i32-N64 | forward | 1 (no) | 168.5 | 4 | 99.6 | 2 | 101.6 | +68.9 | 1.69× |

**Does the extra merge land in `merge`? Yes — all of it.** The stage split closes on `first_call_ms` to
within 1.0–3.6 ms on all 27 rows (e.g. `p256-L64 auto sr_fitness`: stages sum to 14 135.0 against a
14 137.7 ms cold call), and `merge` is the **only** block that separates `auto` from its same-shape twin
`b16`: `auto`/`b16` is 4.90–5.35× on `merge` and 0.85–1.22× on every other block (`decode`/`dedup`
< 1 % of the call, `trace` 21–216 ms with host-side swings up to ±130 ms, `lower` 0.85–1.05×, `compile`
0.85–1.22× i.e. 0.76–12.1 s and dominated by the plan's own op count, `load` 0.97–1.15×). Between `b16`
and `b32` the second block that moves is `lower` (1.75–2.08× at the pop-256 wide-row plans: fewer, larger
bands cost more per lower), which is a partition effect, not a clamp one. The clamp's cost is
**+0.18 to +1.11 s per cold call**, which is why `auto`'s whole cold call is 1.12–1.35× the cheapest
cold call of its column at the six pop-256 rows and 1.01–1.11× at the three `p64` ones.

Read the two "no" rows as the model's own price tag: an **unclamped** `auto` row still pays
1.69–3.00× the `b16` merge (the model run itself plus a merge at the model's K, e.g. 4 bands of
62/61/61/61 at `p256-L32`), so the excess is *model + merge at the requested partition*, not solely
the clamp's second pass.

## 4. Table 3 — warm ratio vs the per-column argmin, and the noise band

`median_ms` of each config divided by the column's best median; `(min-based)` in parentheses.
**Comparison bound: this box's cross-config noise floor measured in round 2 item 1 is 2.6 % (the
same-shape `p64` `b64`-vs-`flat` twin) and 3–6 % across the banded configs — a difference inside that
band is marked "within noise" and must not be read as a knob effect.**

| cell | workload | `auto` | `b16` | `b32` | argmin | `auto` verdict |
|---|---|---|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 1.014 (1.032) | **1.000 (1.000)** | 1.066 (1.071) | `b16` | within noise |
| p256-L32-i8-N1024 | batch_forward | 1.084 (1.093) | **1.000 (1.000)** | 1.034 (1.046) | `b16` | **outside** (+8.4 %) |
| p256-L32-i8-N1024 | forward | 1.189 (1.275) | **1.000 (1.000)** | 1.023 (1.096) | `b16` | **outside** (+18.9 %) |
| p256-L64-i32-N1024 | sr_fitness | 1.032 (1.036) | 1.017 (1.029) | **1.000 (1.000)** | `b32` | within noise |
| p256-L64-i32-N1024 | batch_forward | 1.099 (1.147) | 1.075 (1.109) | **1.000 (1.000)** | `b32` | **outside** (+9.9 %) |
| p256-L64-i32-N1024 | forward | 1.145 (1.134) | 1.026 (1.000) | **1.000 (1.013)** | `b32` | **outside** (+14.5 %) |
| p64-L64-i32-N64 | sr_fitness | 1.027 (1.030) | **1.000 (1.000)** | 1.026 (1.026) | `b16` | within noise |
| p64-L64-i32-N64 | batch_forward | 1.141 (1.130) | **1.000 (1.000)** | 1.014 (1.009) | `b16` | **outside** (+14.1 %) |
| p64-L64-i32-N64 | forward | 1.029 (1.019) | 1.062 (1.060) | **1.000 (1.000)** | `b32` | within noise |

Column winners: **`b16` 5/9** (`p256-L32` all three, `p64` `sr_fitness` + `batch_forward`),
**`b32` 4/9** (`p256-L64` all three, `p64` `forward`), **`auto` 0/9** — and on the min-based estimator
the argmin flips to `b16` in `p256-L64 forward` (1.000 vs 1.013), making it `b16` 6 / `b32` 3.

Pooled over all nine columns (geomean of the ratio to the column argmin, median-based / min-based):

| config | median-based geomean | worst column | min-based geomean | worst column |
|---|---|---|---|---|
| `auto` | 1.0827 | 1.189 | 1.0968 | 1.275 |
| `b16` | 1.0196 | 1.075 | **1.0214** | 1.109 |
| `b32` | **1.0178** | **1.066** | 1.0284 | **1.096** |

Per-workload geomean (median-based): `sr_fitness` auto 1.0243 / b16 1.0056 / b32 1.0301;
`batch_forward` auto 1.1076 / b16 1.0244 / b32 1.0158; `forward` auto 1.1186 / b16 1.0290 / b32 1.0076.
`auto`'s cost is concentrated on the two value paths (its per-row plans run 4 or 1 bands where the
others run 8–15) and is at noise level on the loss path, where its clamp lands on `b16`'s partition.

## 5. `b16` vs `b32` — warm **and** cold on the same cells

| cell | workload | warm `b16`/`b32` | cold `b16`/`b32` | reading |
|---|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 0.938 (**−6.2 %**) | 0.916 (−8.4 %) | `b16` faster both ways |
| p256-L32-i8-N1024 | batch_forward | 0.967 (−3.3 %) | 0.950 (−5.0 %) | `b16` faster both ways (warm within noise) |
| p256-L32-i8-N1024 | forward | 0.978 (−2.2 %) | 0.956 (−4.4 %) | `b16` faster both ways (warm within noise) |
| p256-L64-i32-N1024 | sr_fitness | 1.017 (+1.7 %) | 0.994 (−0.6 %) | tie / within noise |
| p256-L64-i32-N1024 | batch_forward | 1.075 (**+7.5 %**) | 0.964 (−3.6 %) | `b32` faster warm, `b16` cheaper cold |
| p256-L64-i32-N1024 | forward | 1.026 (+2.6 %) | 1.040 (+4.0 %) | `b32` faster both ways (within noise) |
| p64-L64-i32-N64 | sr_fitness | 0.975 (−2.5 %) | 1.011 (+1.1 %) | `b16` faster warm (within noise), cold a wash |
| p64-L64-i32-N64 | batch_forward | 0.987 (−1.3 %) | 1.045 (+4.5 %) | `b16` faster warm (within noise), `b16` dearer cold |
| p64-L64-i32-N64 | forward | 1.062 (**+6.2 %**) | 1.072 (+7.2 %) | `b32` faster both ways |

`b16` is faster warm in 5 of 9 columns and cheaper cold in 5 of 9, with no cell where it wins every
column both ways: it takes all three `p256-L32` columns warm **and** cold, loses `p256-L64 batch_forward`/`forward`
warm, and on `p64` the two directions disagree per workload. Only two warm cells clear the 6 % bound at
all (`p256-L32 sr_fitness` −6.2 % for `b16`, `p256-L64 batch_forward` +7.5 % against it), and the cold
column is a compile reading on a shared box (see §7) — read as "± a factor of a few", not as a table.

## 6. Cross-config validation — the partition does not change the value

Each record diffs its own outputs against the independent plain-torch oracle
(`src/evogp/tests/reference_eval.py`, `validate.compare`, the first 16 trees of the workload).
**The oracle deviation is IDENTICAL in all three configs of every cell/workload (9/9)**: same
`max_abs`, same `max_rel`, same `bitwise_equal`.

| cell | workload | `max_abs` / `max_rel` (every config) | bitwise vs oracle |
|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | 6.0 / 4.74e-07 | no |
| p256-L32-i8-N1024 | batch_forward | 5.86e-03 / 1.13e-03 | no |
| p256-L32-i8-N1024 | forward | 0 / 0 | **yes** |
| p256-L64-i32-N1024 | sr_fitness | 32.0 / 7.76e-07 | no |
| p256-L64-i32-N1024 | batch_forward | 7.81e-02 / 1.14e-05 | no |
| p256-L64-i32-N1024 | forward | 1.19e-07 / 8.01e-08 | no |
| p64-L64-i32-N64 | sr_fitness | 0.25 / 1.62e-07 | no |
| p64-L64-i32-N64 | batch_forward | 6.10e-05 / 1.68e-07 | no |
| p64-L64-i32-N64 | forward | 1.19e-07 / 8.01e-08 | no |

These are the values round 2 item 1 measured for `b8`/`b16`/`b32`/`b64`/`flat` to the precision it
printed them (including the `forward` bitwise-exact row at `p256-L32`), so `auto` and its clamp neither
improve nor degrade the numeric path: the band partition is a scheduling decision.

Value digests (sha256 of the emitted tensor), per cell/workload with the group each config fell into:

| cell | workload | groups | spread (`abs_sum`) |
|---|---|---|---|
| p256-L32-i8-N1024 | sr_fitness | one group (`{auto,b16,b32}`, `e1021d86…`) | 0 (bitwise identical) |
| p256-L32-i8-N1024 | batch_forward | 2 groups (`{auto,b16}` / `{b32}`) | 4691774270.6655 vs .6664 ≈ 2e-12 rel |
| p256-L32-i8-N1024 | forward | 3 groups (one per config) | 671979.2556291…2556320 ≈ 5e-12 rel |
| p256-L64-i32-N1024 | sr_fitness | one group (`7cccccd5…`) | 0 (bitwise identical) |
| p256-L64-i32-N1024 | batch_forward | 2 groups (`{auto,b16}` / `{b32}`) | 316806681.8749 vs .8647 ≈ 3e-11 rel |
| p256-L64-i32-N1024 | forward | 2 groups (`{auto}` / `{b16,b32}`) | 16508.4910367 vs .0377 ≈ 6e-11 rel |
| p64-L64-i32-N64 | sr_fitness | one group (`b6584500…`) | 0 (bitwise identical) |
| p64-L64-i32-N64 | batch_forward | 3 groups (one per config) | 587503.5208594…52092 ≈ 1e-10 rel |
| p64-L64-i32-N64 | forward | one group (`6d7d47f5…`) | 0 (bitwise identical) |

`sr_fitness` is bitwise identical across the three partitions in **every** cell (including `auto`'s
clamped 15-band merge vs `b32`'s 8-band one), and the per-config digest groups on the measured-value
tensors sit at 2e-12–1e-10 relative — the same order round 2 item 1 recorded (≤ 5e-11) and the same
order as the route's documented non-bitwise `batch_forward` caveat. **The partition does not change the
value**; the digest split is low-bit arithmetic ordering, not a band-size accuracy effect.

## 7. Contention, noise, coverage

- **Guard verdicts: 9/9 `ok`, `gpu_pids.idle: true`, `contaminated: false`, `gpu_use_pct` 0.0–1.0 %.**
  The guard had to wait for two configs — 42.0 s (`p256-L32` `b16`) and 57.6 s (`p256-L64` `auto`) — and
  0.2 s for the other seven. Its `python` hint list showed a foreign `/tmp/iree311` `pytest tests/backends`
  job running on the same box (CPU-side; `gpu_pids` empty at every check). That foreign load finished
  before the last four configs; the two waits are recorded evidence, not a failure, and nothing here is
  a re-run or a smoothed number.
- **Within-config noise** (MAD of the 5 samples / median, `max:min`): 0.2–2.4 % (1.03–1.10) on 19 of 27
  rows. The other eight are 2.6–6.3 %: `p256-L64 b32 forward` 2.6 %, `p256-L32 auto batch_forward` 2.7 %,
  `p64 b32 forward` 2.8 %, `p256-L64 b16 forward` 2.9 %, `p256-L64 auto forward` 3.3 %,
  `p256-L32 b16 forward` 3.9 %, `p64 b16 forward` 5.1 % (1.11) and `p64 auto forward` 6.3 % (1.17) —
  the per-row workload is the noisy one on the cheap cell. `p256-L32 auto batch_forward` is the one row
  with a genuine **early-rep outlier** (samples 2.7629 / 2.4180 / 2.3626 / **3.2885** / 2.3535,
  `max:min` 1.40): its median is the headline and its min (2.3535) is still 9.3 % above `b16`'s min
  (2.1528), so the column verdict does not flip either way. Nothing in this report rests on a
  difference below ~5 %.
- **Cold numbers are not warm numbers**: a cold first call is 1.0–14.1 s here, of which the merged-graph
  compile alone is 0.76–12.1 s, on a shared box; the cold column's spread across configs (`auto`
  1.12–1.35× the cheapest at pop 256, 1.01–1.11× at `p64`) and the `b16`-vs-`b32` cold ratios (0.92–1.07)
  should be read as "± a factor of a few". The one exception is the **clamp's merge cost in §3**, which
  is a host-side stage reading inside a single cold call: 0.18–1.11 s against 0.04–0.28 s baselines,
  comfortably above the noise.
- **Coverage.** Measured: 3 cells × 3 configs × 3 workloads = 27 rows, all `status: ok`, none failed,
  timed out or skipped. **Not measured** (out of this spawn's charter, not attempted): `p1000-*`
  (91–162 s merged compile per workload per config on the single-executable path), the composition
  configs `auto-cw4`/`flat-cw4` (pool > 1, chunking changes the merge shape), `b8`/`b64`/`flat` (already
  measured in round 2 item 1), the `gp_step` and cold-start columns, and any `EVOGP_BATCH_BANDS=K`
  explicit pin at the clamp ceiling (`K = 15` / `K = 7`) — i.e. **"auto clamped" is measured against
  `b16`/`b32`, not against the cap itself**, and the surviving question is whether an explicit `K` at
  the cap beats both.
- **Gate status.** `check.py` is the campaign *arm-record* gate and does not apply to this schema: run on
  this directory it reads the nine records and reports `9 records / 224 checks / 28 failures` — 27 ×
  `cache.compile_misses_first_call is None` (a sweep record carries no campaign `cache` block) and 1 ×
  "`campaign.json` unreadable or absent" (a sweep writes `sweep.json`). The identical run on the round-2
  item-1 directory reports `15 records / 362 checks / 46 failures` of the same kinds. So no timing or
  value check in this report rests on that gate — the record's own `status: ok`, its stage-split closure
  (§3) and the oracle validation (§6) are the internal consistency checks, and `sweep.json` records the
  guard verdict per config.

## 8. The two questions, answered as asked

1. **Did `auto` land within noise (2.6–6 %) of the per-column argmin in every column? — No.** It is
   within the band in 4 of 9 columns (`p256-L32 sr_fitness` +1.4 %, `p256-L64 sr_fitness` +3.2 %,
   `p64 sr_fitness` +2.7 %, `p64 forward` +2.9 %) and **outside it in 5** — `p256-L32 batch_forward`
   +8.4 %, `p256-L32 forward` +18.9 %, `p256-L64 batch_forward` +9.9 %, `p256-L64 forward` +14.5 %,
   `p64 batch_forward` +14.1 % (min-based ratios are the same or worse: worst 1.275). `auto` is the
   argmin in **0 of 9** columns and the slowest of the three in **7 of 9** (in `p64 sr_fitness` only
   marginally: 1.1612 vs `b32`'s 1.1596 ms). Its shortfall is not only the
   partition: on the wide-row plans it runs `b16`'s exact merge shape (15 bands) and still measures
   1.4–8.4 % slower, which is the same order as this window's noise plus the one contaminated row.
   Cold, `auto` pays 1.12–1.35× the cheapest cold call at the pop-256 rows (1.01–1.11× at `p64`) and
   carries the clamp's extra merge (§3).
2. **Did `b16` beat `b32` on the same cells warm AND cold? — No clean sweep.** Warm, `b16` is faster in
   5 of 9 columns (−1.3 % … −6.2 %) and slower in 4 (+1.7 % … +7.5 %); only two columns clear the 6 %
   bound and they point in opposite directions (`p256-L32 sr_fitness` −6.2 % for `b16`,
   `p256-L64 batch_forward` +7.5 % against it). Cold, `b16` is cheaper in 5 of 9 (by up to 8.4 %) and
   dearer in 4 (by up to 7.2 %), which is inside the cold number's own uncertainty. `b16` takes the
   whole `p256-L32` cell both ways; `b32` takes the whole `p256-L64` warm column and the `p64 forward`
   column both ways. The pooled geomeans are a tie within noise: median-based `b32` 1.0178 / `b16`
   1.0196, min-based `b16` 1.0214 / `b32` 1.0284.

No recommendation is made here: the round-2 report of record owns the decision.
