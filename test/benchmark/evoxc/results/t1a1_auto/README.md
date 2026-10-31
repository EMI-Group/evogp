# `t1a1_auto/` — the shipped default's verdict, and the PRE-CHANGE records

This directory is the campaign of record for the arm `evoxc-auto` — no `EVOGP_EVAL_MODE` and no `EVOGP_EVOXC_BACKEND` exported (each record's `env.applied_knobs` reads `{"eval_mode": null, "backend": null, "knobs": {}}`), i.e. exactly the policy a user gets from a plain install.
It has two roles now: the JSON records in its per-cell subdirectories are the PRE-CHANGE evidence (measured before the host-glue round, `fused_engine.PLAN_MEMO` + `fused_kernels._bind_launch`, commit `5a699ca`), and this file is the one-screen verdict of the shipped default, which quotes the CURRENT numbers from [`../t1a1_auto_r2/`](../t1a1_auto_r2/README.md) and the same-session A/B against that evidence.
The machine-generated rendering of the pre-change records is `RENDERED.md` in this directory; the current records have their own `RENDERED.md` under `../t1a1_auto_r2/`.
Read these numbers against `../t1a1_fused/main/` (the EXPLICIT `EVOGP_EVAL_MODE=fused` column of its own earlier session) and `../t1a1_baseline/main/` (the PRE-FUSED default, where the same arm id `evoxc-auto-iree` resolved to dense+iree and where neither `forward` nor `gp_step` was scheduled).

## 1. What these campaigns answer

The library's `auto` policy resolves to the fused (Triton) route on a device box, so the question is what the released default — not an opt-in flag — actually costs per call and per generation, and what the host-glue round changed.
Six cells x `evoxc-auto` + `torch-batched`, `warmup 2`, `reps 5` sync-bracketed warm samples (medians), `gp_warmup 1` / `gp_reps 3`, cold first call reported separately, one process per `(cell, arm)`, sequential, `seed 0`, `EVOXC_DISABLE_LOCATIONS=1`: the current campaign is 12 configs, 12 `ok`, 0 failures, elapsed 113.6 s, 12/12 on an idle device (`../t1a1_auto_r2/campaign.json`).
The pre-change campaign in this directory ran four arms over the same workloads (23 configs, 23 `ok`, elapsed 323.6 s) and additionally scheduled `torch-per-tree` and `evoxc-dense-iree`; those two control columns are quoted below where they exist and are labelled by window.

**Verdict.** On this device box the fused default is **46.4×–181.5× faster than `torch-batched` per `batch_forward` call**, **33.6×–153.4× on `SR_fitness`**, **47.1×–145.3× on `forward`** and **7.8×–104.0× per whole GP generation**, with the caveat that `auto` reaches the fused route only when triton is importable, a torch device is usable and `pop_size >= EVOGP_FUSED_MIN_POP` (16) — otherwise it stays on the dense/specialized answer it always had.
Ratios are `torch median / evoxc-auto median`, computed from the records in `../t1a1_auto_r2/`, not quoted from any report.

| cell | pop | tree len (trace) | input len | rows | trees (nodes, seed 0) | distinct structures |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 256 | 32 | 8 | 1 024 | depth 5: mean 15.9, max 31 | 225 |
| `p256-L64-i32-N1024` | 256 | 64 | 32 | 1 024 | depth 5: mean 15.9, max 31 | 232 |
| `p256-L64-i32-N4096` | 256 | 64 | 32 | 4 096 | depth 5: mean 15.9, max 31 | 232 |
| `p1000-L64-i32-N1024` | 1 000 | 64 | 32 | 1 024 | depth 5: mean 16.2, max 31 | 841 |
| `p4096-L64-i32-N1024` | 4 096 | 64 | 32 | 1 024 | depth 5: mean 15.7, max 31 | 3 266 |
| `p2048-L128-i32-N1024-d7` | 2 048 | 128 | 32 | 1 024 | depth 7: mean 42.9, max 123 | 1 665 |

## 2. Route proof — the table that makes the claim falsifiable

Per cell, from that cell's `evoxc-auto` record: `routes.sr_fitness.*`, `fused_cache.served_every_workload` and the record-level `fused_cache` window.

| cell | eval_mode | backend | engine | would_use_fused | fused_fallback_backend | served_every_workload | fused_calls | fused_declines | fused_kernel_launches | fused_kernel_compiles | fused_plan_hits/misses | fused_xt_hits/misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |
| `p256-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |
| `p256-L64-i32-N4096` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |
| `p1000-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |
| `p4096-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |
| `p2048-L128-i32-N1024-d7` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 21/3 | 1/1 |

`fused_reason` on every cell reads `auto policy: device available, triton importable and pop_size=<pop> >= 16 (EVOGP_FUSED_MIN_POP) -> fused (Triton) on the torch device (specialized fallback on numpy)`, i.e. the resolver itself names the fused engine rather than a fallback.
`fused_calls` 24 = first call + 2 warmups + 5 reps for each of the three timed value/loss workloads; `fused_kernel_launches` 24 = one launch per call with 0 declines.
`fused_plan_hits`/`fused_plan_misses` 21/3 are the host-glue round's `PLAN_MEMO` counters (one miss per timed workload, a hit on every warmup and rep after it); `fused_kernel_compiles` 3 is one first launch per kernel specialization (value, loss, per-row) even where Triton served it from disk (`src/evogp/tree/FUSED.md` §7); `fused_xt_hits`/`fused_xt_misses` are the transpose memo (`fused_xt_entries` 1, 1 miss then hits).

Why this table is the proof: the fused route compiles no evoxc graph, so `cache.compile_misses_first_call` is 0 by construction on every `evoxc-auto` record and cannot distinguish a served call from a silent `FusedUnsupported` fallback.
A positive `fused_kernel_launches` delta is the only existence proof, and `check.py` gates any record whose RESOLVED engine is `fused` as a fused record (`is_fused_record`), FAILing on a non-`fused` `routes[*].eval_mode` or a malformed `fused_cache` block and WARNing when no launch is proven — so a silent fallback cannot hide inside an `auto` record.
`fused_fallback_backend` reads `numpy` on every cell because no explicit backend was exported; with `EVOGP_EVOXC_BACKEND` set, a decline would land on specialized+iree instead.

## 3. Headline table — warm medians per call and per generation

Current medians from `../t1a1_auto_r2/`; the "round Δ" column is the same-session A/B against a `git archive 9cc601d` snapshot driven through the same harness (`../t1a1_auto_r2/README.md` §5), and the last column is the PRE-change median of this directory, i.e. the evidence kept here.
The `evoxc-dense-iree` control of the pre-change window is quoted after the table and is not part of any current ratio.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | auto `gp_step` | vs torch-batched sf / bf / fw / gp | round Δ bf | PRE-change bf (this directory) |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.1202 ms | 0.0861 ms | 0.0852 ms | 2.7999 ms | 33.6× / 46.4× / 47.1× / 8.3× | −24.0 % | 0.1023 ms |
| `p256-L64-i32-N1024` | 0.1000 ms | 0.0780 ms | 0.0734 ms | 2.7996 ms | 45.6× / 57.2× / 59.9× / 8.1× | −37.9 % | 0.1169 ms |
| `p256-L64-i32-N4096` | 0.1133 ms | 0.0849 ms | 0.0716 ms | 2.8246 ms | 43.9× / 58.5× / 60.6× / 7.8× | −30.4 % | 0.1307 ms |
| `p1000-L64-i32-N1024` | 0.1098 ms | 0.0947 ms | 0.0820 ms | 3.0517 ms | 44.4× / 51.2× / 55.3× / 22.6× | −22.6 % | 0.1145 ms |
| `p4096-L64-i32-N1024` | 0.1389 ms | 0.1257 ms | 0.0772 ms | 3.0354 ms | 76.2× / 83.9× / 57.5× / 104.0× | −26.4 % | 0.1532 ms |
| `p2048-L128-i32-N1024-d7` | 0.1728 ms | 0.1460 ms | 0.1110 ms | 3.0864 ms | 153.4× / 181.5× / 145.3× / 101.7× | −26.5 % | 0.1879 ms |

Convention: `torch-*` median ÷ `evoxc-auto` median, same cell, same session, so >1 means the default is faster.
The round's gain is larger than the window drift but smaller than a naive old-to-new comparison suggests: the pre-round snapshot re-measured in the current session reads −6.7 % to +11.5 % against the PRE-change column above on `batch_forward`, so the honest Δ is the `--tree-root` A/B (−22.6 % to −37.9 %), not the directory difference (`../t1a1_auto_r2/README.md` §9).
The pre-change window's dense control (a DIFFERENT window and a different `evogp` tree) read `sr_fitness` 1.1046 / 2.1259 / 5.8723 / 5.8309 / 36.9693 ms / `batch_forward` 1.8084 / 2.9494 / 7.6783 / 7.7009 / 47.0181 ms on the five cells where it could run, i.e. 7–9× slower than the default on its best surface (`forward`) and up to 307× on `batch_forward` at `p4096`; the `d7` cell was not scheduled for it (§9).

## 4. Cold start

`first_call_ms` per workload per cell from the current records, then this directory's PRE-change medians for the same fields.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | PRE-change sf / bf / fw |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 495.4 ms | 2.1 ms | 1.9 ms | 458.2 / 1.8 / 1.8 ms |
| `p256-L64-i32-N1024` | 504.5 ms | 1.9 ms | 1.7 ms | 445.9 / 1.9 / 1.7 ms |
| `p256-L64-i32-N4096` | 501.9 ms | 2.1 ms | 1.8 ms | 468.4 / 1.9 / 1.7 ms |
| `p1000-L64-i32-N1024` | 508.6 ms | 2.1 ms | 1.8 ms | 467.6 / 2.0 / 1.9 ms |
| `p4096-L64-i32-N1024` | 496.9 ms | 2.0 ms | 1.8 ms | 461.4 / 1.9 / 1.7 ms |
| `p2048-L128-i32-N1024-d7` | 499.4 ms | 1.9 ms | 1.8 ms | 448.9 / 2.0 / 1.7 ms |

The 495–509 ms `sr_fitness` entries are **not** a JIT: both campaigns ran against a WARM Triton disk cache (the three specializations were already resident), so each fresh process pays a ~0.45–0.5 s disk load of the loss kernel, and the two surfaces whose keys are already resident pay 1.7–2.1 ms.
From an EMPTY Triton cache the one-time cost is larger and is recorded in `src/evogp/tree/FUSED.md` §5B: **`batch_forward` 1 642–1 674 ms, `sr_fitness` 940–991 ms, `forward` 317–330 ms, i.e. ≈2.9 s of JIT for all three surfaces** (1.64 s for the value kernel alone).
That cost is paid **once per kernel specialization per machine**, never per population shape and never per generation — the opposite of the dense route's per-`(shape, used-function-set)` compile, which the pre-change window measured at 2 529–6 688 ms per cell and which the pre-fused default paid as a 3 440–14 461 ms first call.
`TRITON_CACHE_DIR` must point at a writable directory: this box's `$HOME` is read-only and the first launch dies on `$HOME/.triton` without it.

## 5. The GP loop — one whole generation (the number a user feels)

`gp_step` per cell on the current records: total, its eval/ops split (`median_eval_ms` / `median_ops_ms`, in-place difference on the same step), the shares, and the ratio against `torch-batched`.

| cell | auto total | auto eval | auto ops | eval share | ops share | torch-batched total | vs torch-batched |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 2.7999 ms | 0.4050 ms | 2.3949 ms | 14.46 % | 85.54 % | 23.1211 ms | 8.3× |
| `p256-L64-i32-N1024` | 2.7996 ms | 0.3883 ms | 2.4113 ms | 13.87 % | 86.13 % | 22.6321 ms | 8.1× |
| `p256-L64-i32-N4096` | 2.8246 ms | 0.4335 ms | 2.3911 ms | 15.35 % | 84.65 % | 22.1343 ms | 7.8× |
| `p1000-L64-i32-N1024` | 3.0517 ms | 0.4286 ms | 2.6231 ms | 14.04 % | 85.96 % | 69.1133 ms | 22.6× |
| `p4096-L64-i32-N1024` | 3.0354 ms | 0.4174 ms | 2.6180 ms | 13.75 % | 86.25 % | 315.7356 ms | 104.0× |
| `p2048-L128-i32-N1024-d7` | 3.0864 ms | 0.4301 ms | 2.6563 ms | 13.94 % | 86.06 % | 313.8775 ms | 101.7× |

Evaluation is **13.9–15.4 % of a generation**, so the operator layer — 2.39–2.66 ms of crossover/mutation/selection — is the loop's bottleneck, and a generation is 2.80–3.09 ms from 256×32 to 4 096×64 to 2 048×128.
That flatness is the change a user feels: at `p2048-L128-i32-N1024-d7` a generation is 3.0864 ms against 313.8775 ms on `torch-batched`, and the pre-change records of this directory read 2.5830–2.8400 ms with 0.2362–0.2694 ms of evaluation (8.8–10.0 %).
That eval share does NOT reproduce in the current session: the pre-round snapshot measures 0.3716–0.4089 ms in the same bracket here, so the committed 0.24–0.27 ms is window drift, and the round's own effect on the eval bracket is −0.5 % to +9.0 % (`../t1a1_auto_r2/README.md` §7).
Because nothing is re-planned, re-traced or recompiled per generation, `gp_step` pays no compile at all (`gp_step.compile_misses_total` is 0 on every `evoxc-auto` record), which is the structural reason the default wins a turnover loop where the opt-in `batch` route loses one by 17.8–71.5× over 8 generations at pop 256 (`../../batch_opt_round3/`).

## 6. What is still weak (numbers, no hedging)

The kernel is now the larger half again: `fused_check.py --stage timing` splits a warm `batch_forward` into 0.061/0.003, 0.061/0.003, 0.070/0.009, 0.071/0.003, 0.104/0.007 and 0.129/0.011 ms of kernel/host on the six shapes, i.e. **88–96 % kernel**, and the host glue left is 0.003–0.011 ms of Python-level per-call work (route resolution, the `PLAN_MEMO` key build, the scratch lookup, the `contiguous` epilogue).
A real turnover loop misses that memo every generation (the key is the `(id, _version)` of the operand tensors and each generation's population is a new object — measured 1 miss and 0 hits per `pipeline.step()`), so the loop pays the key build and the prepared-call setup at every evaluation while the typed workload reps hit it: the loop's eval bracket is 0.39–0.43 ms against the workload's 0.078–0.146 ms `batch_forward`, and a same-shape interleaved hit/miss probe on `p256-L64-i32-N1024` reads 0.0861 vs 0.2114 ms median (min 0.0758 vs 0.1264) post-round against 0.1122 vs 0.1154 ms pre-round.
The loop itself is unchanged by the round: the same `TimedFitness` bracket inside a real `StandardPipeline` reads 0.3666 ms on the pre-round snapshot against 0.3780 ms here (+3 %, inside this box's noise), which is the −0.5 % to +9.0 % the six cells record (§5).
The tree walk repeats once per row block and per-program work scales with pop: `batch_forward` reads 0.0780 / 0.0947 / 0.1257 ms at pop 256 / 1 000 / 4 096 (`L=64`, `N=1024`), i.e. +21 % then +33 %, while 4× the rows at pop 256 costs 0.0780 → 0.0849 ms (+9 %).
`output_len >= 2` materializes the `(pop, rows, out_len)` prediction buffer in device memory, where the single-output path folds the reduction in registers and moves only `1/rows` of the traffic (`FUSED.md` §7 item 5).
There is no per-stage timer surface on this route and no zero-copy device export — a `(pop, rows, 1)` value tensor stays allocated and is never read back (the route's own accounting for those buffers and the fold scratch is `fused_scratch_bytes`: 32.03 / 64.06 / 256.06 / 250.24 / 1 025.0 / 1 025.0 MB across these six cells).
The population floor is 16 (`EVOGP_FUSED_MIN_POP`), so a single `Tree.forward` / `Tree.SR_fitness` stays on specialized+numpy and never reaches the kernel.
The memory envelope (`EVOGP_FUSED_MAX_SCRATCH_MB`, default 8192) can decline a very large cell into the specialized fallback (`FUSED.md` §12), and the widest measured cell there is the 2 GiB `d8` shape.
And `EVOGP_EVOXC_BACKEND=numpy` still reports `engine="fused"` while every call declines — the engine name is not a served-call proof, which is exactly why these directories record launch counters (§2) instead of trusting it.

## 7. Numerics

Digest groups per value workload (distinct `digest.sha256` and the arms sharing each), read from the current records; `sr_fitness` digests legitimately differ per arm (loss reduction order) and are not compared across arms.

| cell | `batch_forward` groups | `forward` groups | `evoxc-auto` group |
|---|---|---|---|
| `p256-L32-i8-N1024` | 1 (both arms) | 1 (both arms) | shares the torch sha |
| `p256-L64-i32-N1024` | 1 (both arms) | 1 (both arms) | shares the torch sha |
| `p256-L64-i32-N4096` | 1 (both arms) | 1 (both arms) | shares the torch sha |
| `p1000-L64-i32-N1024` | 2 | 2 | its own group |
| `p4096-L64-i32-N1024` | 2 | 2 | its own group |
| `p2048-L128-i32-N1024-d7` | 2 | 2 | its own group |

The split is the documented NaN-payload / low-bit class, which is why the gate needs `--allow-digest-mismatch` (`VALIDATION.md`; `FUSED.md` §11): `batch_forward` and `forward` are bitwise on all three function sets with `max_abs = 0.0` on the finite space, `SR_fitness` agrees to ≤3.8e-7 relative, and the NaN/±inf placement matches everywhere.
The host-glue round is numerics-neutral by construction (only the prepared call moved) and the current campaign confirms it: the three single-group cells are still single and the three that split still split the same way.
The end-to-end check is stronger than the digests: `gp_step.best_fitness` agrees across both arms of every cell to ≤7.5e-9 absolute — `p256-L32-i8-N1024` −0.084430024, `p256-L64-i32-N1024` −0.084812447, `p256-L64-i32-N4096` −0.084454924, `p1000-L64-i32-N1024` −0.084145054, `p4096-L64-i32-N1024` −0.084277824, `p2048-L128-i32-N1024-d7` −0.084303960 — and those six values are unchanged from the pre-change window.

## 8. Noise audit

This box is shared and the warm medians of this directory's window moved by single digits on the multi-millisecond torch and dense columns but by up to 24.1 % on sub-0.15 ms fused calls (the pre-change window was itself measured twice, 46 comparable medians, median |Δ| 2.0 %).
The current window's honest bound is measured, not assumed: the pre-round snapshot driven in the same session reads −6.7 % to +11.5 % against this directory's `batch_forward` medians and −6.0 % to +25.9 % on `SR_fitness` (`../t1a1_auto_r2/README.md` §9).
Treat a ratio below ~1.2× on a fused cell as inside this shared box's noise; the recorded window's own spread is visible in the `min_ms` column of `../t1a1_auto_r2/RENDERED.md` (e.g. `p256-L64-i32-N1024` `batch_forward` median 0.0780 ms against a 0.0761 ms minimum, this directory's `p256-L32-i8-N1024` `sr_fitness` median 0.1382 ms against a 0.1128 ms minimum).

## 9. Caveats and coverage

The pre-change window's `evoxc-dense-iree` arm on `p2048-L128-i32-N1024-d7` was not scheduled; `campaign.json`'s `protocol.scope_dropped` (mirrored in `cell_exclusions.entries`) records the reason: `explicit EVOGP_EVAL_MODE=dense on a population past EVOGP_DENSE_MAX_TREE_LEN declines into per-structure specialized+iree with unbounded artifact growth and OOM-kills the job unit (pop 2048 x 123-node trees); time evoxc-dense-iree-raised (the cap pinned to the cell's tree_len) instead`.
That arm exists in `../t1a1_fused/main/p2048-L128-i32-N1024-d7/` (`evoxc-dense-iree-raised`, `EVOGP_DENSE_MAX_TREE_LEN=128`): 105.7609 ms `sr_fitness`, 110.9945 ms `batch_forward`, 1.2759 ms `forward`, 124.5304 ms `gp_step` — a different window and a different tree root, so it is a scale reference for the excluded control, not a same-session ratio.
`../t1a1_baseline/main/` ran a DIFFERENT tree root (`71f8337b…` against this directory's `576832c6…`, both with `evoxc_sha` `9f66feca…` identical) and scheduled only `sr_fitness,batch_forward`, so every cross-directory ratio mixes two `evogp` revisions and has no `forward` or `gp_step` counterpart there.
`torch-per-tree` (scheduled by the pre-change window only) had a LOWER `gp_step` than `torch-batched` on all six cells (17.7834 vs 23.7219 ms at `p256-L32-i8-N1024`): at these populations the operator layer is the generation's cost, so the per-tree Python walk's cheaper, cache-local evaluation wins the total even though its `sr_fitness` was 6.0–47.7× more expensive per call — the current campaign does not re-schedule that arm.
`../t1a1_fused/main/` ran the EXPLICIT route in its own session and predates the harness's fused counter surface, so its records carry `eval_mode = "fused"` but no `engine` and no `fused_cache` block and cannot prove a launch the way these directories' records do; it also predates the host-glue round, so its 0.0995–0.1673 ms `batch_forward` is a PRE-round figure that the current default now beats (0.0780–0.1460 ms).
Coverage limits: six cells at `tree_len` 32–128 and pops 256–4 096; nothing below the 16-tree floor, nothing at `rows <= 64` with 8–32-node trees, no multi-output (`output_len >= 2`) cell, and `evoxc-specialized-iree` is measured on pops 16/64 only (and not at all here).

## 10. Reproduce and gate

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-cache     # required: this box's $HOME is read-only
cd test/benchmark/evoxc
# current records
$PY campaign.py --out results/t1a1_auto_r2 \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-auto,torch-batched \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800 --timeout 1800
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto_r2 \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY summarize.py --results results/t1a1_auto_r2 --out results/t1a1_auto_r2/RENDERED.md
# the pre-change records kept in THIS directory (four arms), gated where they are
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto \
    --allow-snapshot --allow-missing --allow-digest-mismatch
```

Gates of record: **12 records / 356 checks / 0 failures** on `../t1a1_auto_r2/` (6 declared digest WARN) and **23 records / 640 checks / 0 failures** on this directory (7 declared WARN — one `missing arms (allowed)` for the excluded dense arm on `d7` and six digest splits).
One process per config, never two timed configs at once, idle-device guard on every config; warm medians on this shared box are ±10 %, so compare a knob only inside its own cell and window.
