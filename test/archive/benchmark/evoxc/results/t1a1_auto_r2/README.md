# `t1a1_auto_r2/` — the shipped default (`auto` → fused) after the host-glue round

This directory is the CURRENT campaign of record for the arm `evoxc-auto`: no `EVOGP_EVAL_MODE` and no `EVOGP_EVOXC_BACKEND` exported (each record's `env.applied_knobs` reads `{"eval_mode": null, "backend": null, "knobs": {}}`), i.e. exactly the policy a plain install gets.
It re-measures the column that `../t1a1_auto/` measured BEFORE the host-glue round (`fused_engine.PLAN_MEMO` + `fused_kernels._bind_launch`, commit `5a699ca`); that directory's JSON records are the pre-change evidence and were left untouched.
The machine-generated rendering of these records is `RENDERED.md`; this file is the human one-screen summary.
Read it against `../t1a1_fused/main/` (the EXPLICIT `EVOGP_EVAL_MODE=fused` column, an earlier session) and `../t1a1_baseline/main/` (the PRE-FUSED default, where the same arm id `evoxc-auto-iree` resolved to dense+iree).

## 1. What this campaign answers

Does the shipped default — not an opt-in flag — still read what the route's own report claims after the per-call plan memoization landed?
Six cells x two arms (`evoxc-auto`, `torch-batched`), `warmup 2`, `reps 5` sync-bracketed warm samples (medians), `gp_warmup 1` / `gp_reps 3`, cold first call reported separately, one process per `(cell, arm)`, sequential, `seed 0`, `EVOXC_DISABLE_LOCATIONS=1`: `campaign.json` records 12 configs, 12 `ok`, 0 failures, elapsed 113.6 s, 12/12 on an idle device, `contaminated` false for all 12.
The workload set is the full one of the previous round (`sr_fitness`, `batch_forward`, `forward`, `gp_step`) so this directory supersedes that one workload for workload; the `evoxc-auto`/`torch-batched` pair is the only arm pair needed for the headline ratios.

**Verdict.** Warm `Forest.batch_forward` reads **0.0780–0.1460 ms** on the five `L`-shaped cells and 0.0861 ms at `p256-L32`, i.e. **46.4×–181.5× the plain-torch batched evaluator**, with `SR_fitness` at **33.6×–153.4×** and a whole GP generation at **7.8×–104.0×**.
Against the pre-round snapshot measured in the same session and with the same harness (`--tree-root` of commit `9cc601d`, §5) the round cut `batch_forward` by **22.6–37.9 %**, `SR_fitness` by **18.9–29.6 %** and `forward` by **23.0–41.8 %** — the perf round's claim reproduces, with the largest relative gains at `p256` and the `p4096`/`d7` cells cutting ~26 % because their warm call is already >90 % kernel time (§6).
Ratios are `torch median / evoxc-auto median` (higher = the default is faster), computed from the records in this directory.

| cell | pop | tree len (trace) | input len | rows | trees (nodes, seed 0) | distinct structures |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 256 | 32 | 8 | 1 024 | depth 5: mean 15.9, max 31 | 225 |
| `p256-L64-i32-N1024` | 256 | 64 | 32 | 1 024 | depth 5: mean 15.9, max 31 | 232 |
| `p256-L64-i32-N4096` | 256 | 64 | 32 | 4 096 | depth 5: mean 15.9, max 31 | 232 |
| `p1000-L64-i32-N1024` | 1 000 | 64 | 32 | 1 024 | depth 5: mean 16.2, max 31 | 841 |
| `p4096-L64-i32-N1024` | 4 096 | 64 | 32 | 1 024 | depth 5: mean 15.7, max 31 | 3 266 |
| `p2048-L128-i32-N1024-d7` | 2 048 | 128 | 32 | 1 024 | depth 7: mean 42.9, max 123 | 1 665 |

## 2. Route proof — the table that makes the claim falsifiable

Per cell, from that cell's `evoxc-auto` record: `routes.*.eval_mode/backend/engine`, `fused_cache.served_every_workload` and the whole-config `fused_cache.totals` window.

| cell | eval_mode | backend | engine | would_use_fused | fallback backend | served_every_workload | fused_calls | declines | launches | kernel compiles | plan hits/misses | xt hits/misses | scratch MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 32.03 |
| `p256-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 64.06 |
| `p256-L64-i32-N4096` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 256.06 |
| `p1000-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 250.24 |
| `p4096-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 1 025.0 |
| `p2048-L128-i32-N1024-d7` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 | 1 025.0 |

`fused_reason` on every cell reads `auto policy: device available, triton importable and pop_size=<pop> >= 16 (EVOGP_FUSED_MIN_POP) -> fused (Triton) on the torch device (specialized fallback on numpy)`, so the resolver itself names the fused engine rather than a fallback.
`fused_calls` 24 = first call + 2 warmups + 5 reps for each of the three timed value/loss workloads; `fused_kernel_launches` 24 = one launch per call, 0 declines — a positive launch delta is the only existence proof, because the fused route compiles no evoxc graph (`cache.compile_misses_first_call` is 0 there by construction).
`fused_plan_hits`/`fused_plan_misses` 21/3 are the host-glue round's new `PLAN_MEMO` counters: one miss per timed workload (the first prepared call of each shape) and a hit on every warmup and rep after it — this is the mechanism the per-call gain comes from.
`fused_kernel_compiles` 3 is one first launch per kernel specialization (value, loss, per-row) even where Triton served it from disk; `fused_xt_hits`/`fused_xt_misses` are the transpose memo (1 entry after the first miss); `fused_scratch_bytes` is the per-cell geometry, which is why it differs per cell while the counters above do not.

## 3. Headline table — warm medians per call and per generation

`evoxc-auto` medians are `workloads.*.median_ms` and `gp_step.median_total_ms`; every ratio is the same cell's `torch-batched` median divided by the `evoxc-auto` median in the same window.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | auto `gp_step` | vs torch-batched: sf / bf / fw / gp |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.1202 ms | 0.0861 ms | 0.0852 ms | 2.7999 ms | 33.6× / 46.4× / 47.1× / 8.3× |
| `p256-L64-i32-N1024` | 0.1000 ms | 0.0780 ms | 0.0734 ms | 2.7996 ms | 45.6× / 57.2× / 59.9× / 8.1× |
| `p256-L64-i32-N4096` | 0.1133 ms | 0.0849 ms | 0.0716 ms | 2.8246 ms | 43.9× / 58.5× / 60.6× / 7.8× |
| `p1000-L64-i32-N1024` | 0.1098 ms | 0.0947 ms | 0.0820 ms | 3.0517 ms | 44.4× / 51.2× / 55.3× / 22.6× |
| `p4096-L64-i32-N1024` | 0.1389 ms | 0.1257 ms | 0.0772 ms | 3.0354 ms | 76.2× / 83.9× / 57.5× / 104.0× |
| `p2048-L128-i32-N1024-d7` | 0.1728 ms | 0.1460 ms | 0.1110 ms | 3.0864 ms | 153.4× / 181.5× / 145.3× / 101.7× |

Convention: `torch-batched` medians from the same window are 4.0389 / 4.5643 / 4.9737 / 4.8748 / 10.5843 / 26.5080 ms on `sr_fitness` and 3.9963 / 4.4598 / 4.9673 / 4.8509 / 10.5450 / 26.4966 ms on `batch_forward`.
The `dense` control column of the previous round is NOT re-run here (this campaign schedules two arms); its same-window numbers live in `../t1a1_auto/README.md` §3 and are a different window and a different `evogp` tree, so no ratio in this file crosses into them.

## 4. Cold start

`first_call_ms` per workload per cell from this run's records.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` |
|---|---|---|---|
| `p256-L32-i8-N1024` | 495.4 ms | 2.1 ms | 1.9 ms |
| `p256-L64-i32-N1024` | 504.5 ms | 1.9 ms | 1.7 ms |
| `p256-L64-i32-N4096` | 501.9 ms | 2.1 ms | 1.8 ms |
| `p1000-L64-i32-N1024` | 508.6 ms | 2.1 ms | 1.8 ms |
| `p4096-L64-i32-N1024` | 496.9 ms | 2.0 ms | 1.8 ms |
| `p2048-L128-i32-N1024-d7` | 499.4 ms | 1.9 ms | 1.8 ms |

The 495–509 ms `sr_fitness` entries are **not** a JIT: the campaign ran against a WARM Triton disk cache, so each fresh process pays a ~0.5 s disk load of the loss kernel, and the two surfaces whose keys are already resident pay 1.7–2.1 ms.
From an EMPTY Triton cache the one-time cost is the one recorded in `src/evogp/tree/FUSED.md` §5B: **`batch_forward` 1 642–1 674 ms, `sr_fitness` 940–991 ms, `forward` 317–330 ms**, and `TRITON_CACHE_DIR` must point at a writable directory (this box's `$HOME` is read-only).

## 5. The change this round measured — a same-session A/B against the pre-round tree

`/tmp/evogp_preround` = `git archive 9cc601d` (the commit before `5a699ca`, i.e. without `PLAN_MEMO` / `_bind_launch`), driven through the SAME harness in the SAME session with `--tree-root`, `--arms evoxc-auto`, the same four workloads and the same 2 warmup + 5 reps protocol.
Each pair below is one cell's median on that snapshot against this directory's median, so the delta is the round and not a window.

| cell | bf pre → post | Δ | sf pre → post | Δ | fw pre → post | Δ |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.1133 → 0.0861 | −24.0 % | 0.1483 → 0.1202 | −18.9 % | 0.1106 → 0.0852 | −23.0 % |
| `p256-L64-i32-N1024` | 0.1256 → 0.0780 | −37.9 % | 0.1406 → 0.1000 | −28.9 % | 0.1261 → 0.0734 | −41.8 % |
| `p256-L64-i32-N4096` | 0.1219 → 0.0849 | −30.4 % | 0.1566 → 0.1133 | −27.7 % | 0.1066 → 0.0716 | −32.8 % |
| `p1000-L64-i32-N1024` | 0.1223 → 0.0947 | −22.6 % | 0.1559 → 0.1098 | −29.6 % | 0.1090 → 0.0820 | −24.8 % |
| `p4096-L64-i32-N1024` | 0.1708 → 0.1257 | −26.4 % | 0.1803 → 0.1389 | −23.0 % | 0.1247 → 0.0772 | −38.1 % |
| `p2048-L128-i32-N1024-d7` | 0.1986 → 0.1460 | −26.5 % | 0.2225 → 0.1728 | −22.3 % | 0.1670 → 0.1110 | −33.5 % |

The pre-round snapshot is NOT the same as the committed `../t1a1_auto/` medians: that window measured `p256-L64-i32-N1024` `batch_forward` at 0.1169 ms where this session reads 0.1256 ms on the identical pre-round code (+7 %, inside this box's ±10 % envelope), so treat the cross-directory comparison as drift-limited and this A/B as the number for the round.
A second, cheaper A/B on the `EVOGP_FUSED_MIN_POP` column (§13 of `src/evogp/tree/FUSED.md`) reads **0.0988–0.1178 → 0.0651–0.0684 ms** for `batch_forward` at pop 16–128, `L=64`, `i32`, `N=1024` (10 sync-bracketed reps, 2 warmups, public API).

## 6. Where a warm call now goes

`fused_check.py --stage timing` in this session splits the same six shapes into `kernel` (the Triton launch alone) and `host` (everything else): 0.061/0.003, 0.061/0.003, 0.070/0.009, 0.071/0.003, 0.104/0.007, 0.129/0.011 ms, i.e. the **kernel is 88–96 % of a warm `batch_forward`** and the host glue is 0.003–0.011 ms.
Both ends matter for reading §5: the cells that cut the least in relative terms are not the ones with the most glue. `p256-L32` (−24.0 %) and `p1000` (−22.6 %) already had the smallest absolute glue, while the `p4096` and `d7` cells (−26.4 % / −26.5 %) have the largest calls and therefore the largest kernels (0.104 / 0.129 ms), so the share the round could remove there is bounded by the kernel — the ">50 % kernel time" ceiling is real, but it already applies to every cell of this grid at >88 %.
What is left is host-side Python per call (route resolution, the `PLAN_MEMO` key build, the scratch lookup) and the kernel itself; there is no per-stage timer surface on the route, so this split is reconstructed by `fused_check._stage_split` rather than exported.
The memo's gain is the REPEATED-population path, which is what the harness's workload reps measure: the key is the `(id, _version)` of the operand tensors, so a real GP generation — a new population object every step — misses it every time (measured on `p256-L64-i32-N1024`: 1 plan miss and 0 hits per `pipeline.step()`), and a same-shape interleaved hit/miss probe on that cell reads 0.0861 vs 0.2114 ms median (min 0.0758 vs 0.1264) where the pre-round tree reads 0.1122 vs 0.1154 ms for the same two shapes.
The loop is therefore NOT what the round sped up, and it is not what it slowed down either: the same `TimedFitness` bracket inside a real `StandardPipeline` on that cell reads 0.3666 ms pre-round against 0.3780 ms post-round (+3 %, inside this box's noise), which is the `−0.5 %` to `+9.0 %` recorded per cell in §7.

## 7. The GP loop — one whole generation

`gp_step` per cell: total, its eval/ops split (`median_eval_ms` / `median_ops_ms`, in-place difference on the same step), the shares, and the ratio against `torch-batched`.

| cell | auto total | auto eval | auto ops | eval share | ops share | torch-batched total | vs torch-batched |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 2.7999 ms | 0.4050 ms | 2.3949 ms | 14.46 % | 85.54 % | 23.1211 ms | 8.3× |
| `p256-L64-i32-N1024` | 2.7996 ms | 0.3883 ms | 2.4113 ms | 13.87 % | 86.13 % | 22.6321 ms | 8.1× |
| `p256-L64-i32-N4096` | 2.8246 ms | 0.4335 ms | 2.3911 ms | 15.35 % | 84.65 % | 22.1343 ms | 7.8× |
| `p1000-L64-i32-N1024` | 3.0517 ms | 0.4286 ms | 2.6231 ms | 14.04 % | 85.96 % | 69.1133 ms | 22.6× |
| `p4096-L64-i32-N1024` | 3.0354 ms | 0.4174 ms | 2.6180 ms | 13.75 % | 86.25 % | 315.7356 ms | 104.0× |
| `p2048-L128-i32-N1024-d7` | 3.0864 ms | 0.4301 ms | 2.6563 ms | 13.94 % | 86.06 % | 313.8775 ms | 101.7× |

Evaluation is **13.9–15.4 %** of a generation, so the operator layer (2.39–2.66 ms) is still the loop's bottleneck, and `gp_step.compile_misses_total` is 0 on every record: a changed population changes only the node tensors the kernel reads, and nothing is re-planned or recompiled per generation.
The eval share is NOT comparable with the previous round's committed 8.8–10.0 %: the pre-round snapshot measured 0.3716–0.4089 ms in the same bracket in this session, i.e. that split drifted between windows for both trees alike and the round did not cause it (post-round reads 0.3883–0.4335 ms, −0.5 % to +9.0 %).
`gp_step.best_fitness` agrees with the torch arm to ≤7.5e-9 absolute on every cell (−0.084430024, −0.084812447, −0.084454924, −0.084145054, −0.084277824, −0.084303960).

## 8. Numerics

Digest groups per value workload, read from the records; `sr_fitness` digests legitimately differ per arm (loss reduction order) and are not compared across arms.

| cell | `batch_forward` groups | `forward` groups |
|---|---|---|
| `p256-L32-i8-N1024` | 1 (both arms) | 1 (both arms) |
| `p256-L64-i32-N1024` | 1 (both arms) | 1 (both arms) |
| `p256-L64-i32-N4096` | 1 (both arms) | 1 (both arms) |
| `p1000-L64-i32-N1024` | 2 | 2 |
| `p4096-L64-i32-N1024` | 2 | 2 |
| `p2048-L128-i32-N1024-d7` | 2 | 2 |

The split is the documented NaN-payload / low-bit class (the gate needs `--allow-digest-mismatch`; `VALIDATION.md`, `FUSED.md` §11): `batch_forward` and `forward` are bitwise on all three function sets with `max_abs = 0.0` on the finite space, and `SR_fitness` agrees to ≤3.8e-7 relative.
The round is numerics-neutral by construction (only the host-side prepared call moved) and this campaign confirms it: the three cells whose digests were a single group are still a single group, and the three that split still split the same way.

## 9. Noise audit

The honest bound on a cross-window comparison is measurable here: the pre-round snapshot driven in THIS session (§5) reads −6.7 % to +11.5 % against the committed previous window on `batch_forward` and −6.0 % to +25.9 % on `SR_fitness` (the +25.9 % outlier is `p256-L64-i32-N4096`, 0.1244 → 0.1566 ms on the identical pre-round code).
That is why §5 is an A/B and not a directory comparison: read old-committed → this directory directly and the round looks like −15.8 / −33.3 / −35.0 / −17.3 / −17.9 / −22.3 % on `batch_forward`, numbers that mix the round with a 6–12 % window shift.
Every large relative delta sits on a sub-0.15 ms call, where a 0.02–0.045 ms absolute move reads as 20–38 %; the multi-millisecond torch column moves by single digits.
Treat a ratio below ~1.2× on a fused cell as inside this shared box's noise; the recorded window's own spread is visible in the `min_ms` column of `RENDERED.md` (e.g. `p256-L64-i32-N1024` `batch_forward` median 0.0780 ms against a 0.0761 ms minimum).

## 10. Caveats and coverage

Coverage limits: six cells at `tree_len` 32–128 and pops 256–4 096; nothing below the 16-tree floor, nothing at `rows <= 64` with 8–32-node trees, no multi-output (`output_len >= 2`) cell.
The `dense` and `torch-per-tree` controls are not scheduled in this campaign; `../t1a1_auto/` has the dense column of the previous window and `../t1a1_fused/main/` the explicit-route column of its own session.
`../t1a1_baseline/main/` ran a different tree root and only two workloads, so no ratio there crosses into this file.

## 11. Reproduce and gate

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-cache     # required: this box's $HOME is read-only
cd test/benchmark/evoxc
$PY campaign.py --out results/t1a1_auto_r2 \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-auto,torch-batched \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800 --timeout 1800
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto_r2 \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY summarize.py --results results/t1a1_auto_r2 --out results/t1a1_auto_r2/RENDERED.md

# the same-session control the deltas of §5 come from
git archive 9cc601d | tar -x -C /tmp/evogp_preround
$PY campaign.py --out /tmp/ab_preround --tree-root /tmp/evogp_preround \
    --cells <the six ids above> --arms evoxc-auto \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800
```

Gate of record: **12 records / 356 checks / 0 failures**, plus 6 declared WARN — all of them digest splits (both value workloads on `p1000`, `p4096` and `p2048-L128-...-d7`).
One process per config, never two timed configs at once, idle-device guard on every config.
