# `t1a1_auto/` — the shipped default's verdict, and the PRE-CHANGE records

This directory is the campaign of record for the arm `evoxc-auto` — no `EVOGP_EVAL_MODE` and no `EVOGP_EVOXC_BACKEND` exported (each record's `env.applied_knobs` reads `{"eval_mode": null, "backend": null, "knobs": {}}`), i.e. exactly the policy a user gets from a plain install.
It has two roles now: the JSON records in its per-cell subdirectories are the PRE-CHANGE evidence (measured before the host-glue round, `fused_engine.PLAN_MEMO` + `fused_kernels._bind_launch`, commit `5a699ca`), and this file is the one-screen verdict of the shipped default, which quotes the CURRENT numbers from [`../t1a1_auto_r3/`](../t1a1_auto_r3/README.md) and the same-session A/Bs against that evidence.
The machine-generated rendering of the pre-change records is `RENDERED.md` in this directory; the previous round's records are under `../t1a1_auto_r2/` and the current ones under `../t1a1_auto_r3/`, each with its own `RENDERED.md`.
Read these numbers against `../t1a1_fused/main/` (the EXPLICIT `EVOGP_EVAL_MODE=fused` column of its own earlier session) and `../t1a1_baseline/main/` (the PRE-FUSED default, where the same arm id `evoxc-auto-iree` resolved to dense+iree and where neither `forward` nor `gp_step` was scheduled).

## 1. What these campaigns answer

The library's `auto` policy resolves to the fused (Triton) route on a device box, so the question is what the released default — not an opt-in flag — actually costs per call and per generation, and what the host-glue round and the round after it changed.
Six cells x `evoxc-auto` + `torch-batched`, `warmup 2`, `reps 5` sync-bracketed warm samples (medians), `gp_warmup 1` / `gp_reps 3`, cold first call reported separately, one process per `(cell, arm)`, sequential, `seed 0`, `EVOXC_DISABLE_LOCATIONS=1`: the current campaign is 12 configs, 12 `ok`, 0 failures, elapsed 113.0 s, 12/12 on an idle device (`../t1a1_auto_r3/campaign.json`).
The pre-change campaign in this directory ran four arms over the same workloads (23 configs, 23 `ok`, elapsed 323.6 s) and additionally scheduled `torch-per-tree` and `evoxc-dense-iree`; those two control columns are quoted below where they exist and are labelled by window.

**Verdict.** On this device box the fused default is **80.2×–228.9× faster than `torch-batched` per `batch_forward` call**, **60.2×–208.8× on `SR_fitness`**, **85.8×–220.1× on `forward`** and **7.7×–99.8× per whole GP generation**, with the caveat that `auto` reaches the fused route only when triton is importable, a torch device is usable and `pop_size >= EVOGP_FUSED_MIN_POP` (16) — otherwise it stays on the dense/specialized answer it always had.
Ratios are `torch median / evoxc-auto median`, computed from the records in `../t1a1_auto_r3/`, not quoted from any report.

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
The current records carry the same six rows digit for digit, plus the plan fast path's own counters: `fused_bound_launches` 21 of the 24 calls and `fused_tag_hits`/`fused_tag_misses` 2/1 (`../t1a1_auto_r3/README.md` §3).

Why this table is the proof: the fused route compiles no evoxc graph, so `cache.compile_misses_first_call` is 0 by construction on every `evoxc-auto` record and cannot distinguish a served call from a silent `FusedUnsupported` fallback.
A positive `fused_kernel_launches` delta is the only existence proof, and `check.py` gates any record whose RESOLVED engine is `fused` as a fused record (`is_fused_record`), FAILing on a non-`fused` `routes[*].eval_mode` or a malformed `fused_cache` block and WARNing when no launch is proven — so a silent fallback cannot hide inside an `auto` record.
`fused_fallback_backend` reads `numpy` on every cell because no explicit backend was exported; with `EVOGP_EVOXC_BACKEND` set, a decline would land on specialized+iree instead.

## 3. Headline table — warm medians per call and per generation

Current medians from `../t1a1_auto_r3/`; the "round Δ" column is the same-session `--tree-root` A/B against the `t1a1_auto_r2` tree (`git archive 5a699ca`), driven through the same harness in the same session (`../t1a1_auto_r3/README.md` §4–§5), and the last column is the PRE-change median of this directory, i.e. the evidence kept here.
The `evoxc-dense-iree` control of the pre-change window is quoted after the table and is not part of any current ratio.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | auto `gp_step` | vs torch-batched sf / bf / fw / gp | round Δ bf | PRE-change bf (this directory) |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.0670 ms | 0.0493 ms | 0.0455 ms | 3.0532 ms | 60.2× / 80.2× / 85.8× / 7.7× | −42.4 % | 0.1023 ms |
| `p256-L64-i32-N1024` | 0.0572 ms | 0.0449 ms | 0.0399 ms | 2.8416 ms | 78.0× / 108.4× / 121.0× / 8.3× | −41.8 % | 0.1169 ms |
| `p256-L64-i32-N4096` | 0.0650 ms | 0.0522 ms | 0.0391 ms | 2.9002 ms | 77.6× / 96.2× / 115.7× / 7.7× | −42.1 % | 0.1307 ms |
| `p1000-L64-i32-N1024` | 0.0619 ms | 0.0500 ms | 0.0390 ms | 2.8406 ms | 79.2× / 96.8× / 119.6× / 24.6× | −43.4 % | 0.1145 ms |
| `p4096-L64-i32-N1024` | 0.0886 ms | 0.0790 ms | 0.0437 ms | 3.0716 ms | 118.0× / 131.8× / 101.8× / 99.8× | −36.5 % | 0.1532 ms |
| `p2048-L128-i32-N1024-d7` | 0.1262 ms | 0.1152 ms | 0.0722 ms | 3.2823 ms | 208.8× / 228.9× / 220.1× / 92.3× | −23.6 % | 0.1879 ms |

Convention: `torch-*` median ÷ `evoxc-auto` median, same cell, same session, so >1 means the default is faster.
The round's gain is larger than the window drift and is measured as an A/B rather than assumed: the `t1a1_auto_r2` tree re-run in the current session reads −6.8 % to +6.2 % against its own committed `batch_forward` medians, and a code-identical control reads 0.951–1.118 against the current records, so the −42.4 % to −23.6 % above is the tree and not the box (`../t1a1_auto_r3/README.md` §4–§5, §8).
The pre-change window's dense control (a DIFFERENT window and a different `evogp` tree) read `sr_fitness` 1.1046 / 2.1259 / 5.8723 / 5.8309 / 36.9693 ms / `batch_forward` 1.8084 / 2.9494 / 7.6783 / 7.7009 / 47.0181 ms on the five cells where it could run, i.e. 7–9× slower than the default on its best surface (`forward`) and up to 307× on `batch_forward` at `p4096`; the `d7` cell was not scheduled for it (§9).

## 4. Cold start

`first_call_ms` per workload per cell from the current records, then this directory's PRE-change medians for the same fields.

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | PRE-change sf / bf / fw |
|---|---|---|---|---|
| `p256-L32-i8-N1024` | 526.9 ms | 2.2 ms | 2.0 ms | 458.2 / 1.8 / 1.8 ms |
| `p256-L64-i32-N1024` | 492.4 ms | 2.0 ms | 1.8 ms | 445.9 / 1.9 / 1.7 ms |
| `p256-L64-i32-N4096` | 496.3 ms | 2.1 ms | 1.8 ms | 468.4 / 1.9 / 1.7 ms |
| `p1000-L64-i32-N1024` | 487.7 ms | 2.0 ms | 1.8 ms | 467.6 / 2.0 / 1.9 ms |
| `p4096-L64-i32-N1024` | 494.1 ms | 2.1 ms | 1.9 ms | 461.4 / 1.9 / 1.7 ms |
| `p2048-L128-i32-N1024-d7` | 500.8 ms | 2.0 ms | 1.8 ms | 448.9 / 2.0 / 1.7 ms |

The 488–527 ms `sr_fitness` entries are **not** a JIT: all three campaigns ran against a WARM Triton disk cache (the three specializations were already resident), so each fresh process pays a ~0.45–0.5 s disk load of the loss kernel, and the two surfaces whose keys are already resident pay 1.8–2.2 ms.
From an EMPTY Triton cache the one-time cost is larger and is recorded in `src/evogp/tree/FUSED.md` §5B: **`batch_forward` 1 642–1 674 ms, `sr_fitness` 940–991 ms, `forward` 317–330 ms, i.e. ≈2.9 s of JIT for all three surfaces** (1.64 s for the value kernel alone).
That cost is paid **once per kernel specialization per machine**, never per population shape and never per generation — the opposite of the dense route's per-`(shape, used-function-set)` compile, which the pre-change window measured at 2 529–6 688 ms per cell and which the pre-fused default paid as a 3 440–14 461 ms first call.
`TRITON_CACHE_DIR` must point at a writable directory: this box's `$HOME` is read-only and the first launch dies on `$HOME/.triton` without it.

## 5. The GP loop — one whole generation (the number a user feels)

`gp_step` per cell on the current records: total, its eval/ops split (`median_eval_ms` / `median_ops_ms`, in-place difference on the same step), the shares, and the ratio against `torch-batched`.

| cell | auto total | auto eval | auto ops | eval share | ops share | torch-batched total | vs torch-batched |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 3.0532 ms | 0.4708 ms | 2.5824 ms | 15.42 % | 84.58 % | 23.376 ms | 7.7× |
| `p256-L64-i32-N1024` | 2.8416 ms | 0.4372 ms | 2.4044 ms | 15.39 % | 84.61 % | 23.674 ms | 8.3× |
| `p256-L64-i32-N4096` | 2.9002 ms | 0.4327 ms | 2.4675 ms | 14.92 % | 85.08 % | 22.248 ms | 7.7× |
| `p1000-L64-i32-N1024` | 2.8406 ms | 0.4353 ms | 2.4053 ms | 15.32 % | 84.68 % | 69.813 ms | 24.6× |
| `p4096-L64-i32-N1024` | 3.0716 ms | 0.4657 ms | 2.6059 ms | 15.16 % | 84.84 % | 306.385 ms | 99.8× |
| `p2048-L128-i32-N1024-d7` | 3.2823 ms | 0.4677 ms | 2.8146 ms | 14.25 % | 85.75 % | 302.887 ms | 92.3× |

Evaluation is **14.2–15.4 % of a generation**, so the operator layer — 2.40–2.81 ms of crossover/mutation/selection — is the loop's bottleneck, and a generation is 2.84–3.28 ms from 256×32 to 4 096×64 to 2 048×128.
That flatness is the change a user feels: at `p2048-L128-i32-N1024-d7` a generation is 3.2823 ms against 302.887 ms on `torch-batched`, and the pre-change records of this directory read 2.5830–2.8400 ms with 0.2362–0.2694 ms of evaluation (8.8–10.0 %).
The current bracket does NOT reproduce that pre-change share: the previous round's records read 2.7999–3.0864 ms with 0.3883–0.4335 ms of evaluation, and the tree of the current round reads the same (`gp_step` 0.968–1.140 of the `t1a1_auto_r2` tree in a same-session A/B), so the 8.8–10.0 % of the pre-change window is window drift on the eval slice, not a regression or a win.
Because nothing is re-planned, re-traced or recompiled per generation, `gp_step` pays no compile at all (`gp_step.compile_misses_total` is 0 on every `evoxc-auto` record), which is the structural reason the default wins a turnover loop where the opt-in `batch` route loses one by 17.8–71.5× over 8 generations at pop 256 (`../../batch_opt_round3/`).

## 6. What is still weak (numbers, no hedging)

The kernel is the whole call on the current tree: `fused_check.py --stage timing` reads a warm `batch_forward` of 0.039, 0.038, 0.047, 0.044, 0.077 and 0.108 ms on the six shapes, while the separately measured launch-only path reads 0.090, 0.089, 0.086, 0.086, 0.107 and 0.137 ms — i.e. the plan fast path now makes the WHOLE call cheaper than one independently timed launch, so the stage's kernel/host subtraction goes negative (−0.051…−0.030 ms) and no longer decomposes the call; read the total column only (`src/evogp/tree/fused_check.py`, this session's `--stage timing`).
The previous round measured that split at 0.061/0.003, 0.061/0.003, 0.070/0.009, 0.071/0.003, 0.104/0.007 and 0.129/0.011 ms of kernel/host, i.e. 88–96 % kernel, and its interleaved hit/miss probe on `p256-L64-i32-N1024` read 0.0861 vs 0.2114 ms (0.1122 vs 0.1154 ms pre-round) — those two numbers belong to that window and were not re-measured here.
A turnover loop does not profit from the call-level win: the loop's eval bracket is 0.43–0.47 ms against the typed workload's 0.045–0.115 ms `batch_forward`.
The tree walk repeats once per row block and per-program work scales with pop: `batch_forward` reads 0.0449 / 0.0500 / 0.0790 ms at pop 256 / 1 000 / 4 096 (`L=64`, `N=1024`), i.e. +11 % then +76 %, while 4× the rows at pop 256 costs 0.0449 → 0.0522 ms (+16 %).
`output_len >= 2` materializes the `(pop, rows, out_len)` prediction buffer in device memory, where the single-output path folds the reduction in registers and moves only `1/rows` of the traffic (`FUSED.md` §7 item 5).
There is no per-stage timer surface on this route and no zero-copy device export — a `(pop, rows, 1)` value tensor stays allocated and is never read back (the route's own accounting for those buffers and the fold scratch is `fused_scratch_bytes`: 33.59 / 67.17 / 268.50 / 262.40 / 1 074.79 / 1 074.79 MB across these six cells).
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
The host-glue round and the round after it are numerics-neutral by construction (only the prepared call and the decision lookup moved) and the current records confirm it: the three single-group cells are still single and the three that split still split the same way.
The end-to-end check is stronger than the digests: `gp_step.best_fitness` agrees across both arms of every cell to ≤7.5e-9 absolute — `p256-L32-i8-N1024` −0.084430024, `p256-L64-i32-N1024` −0.084812447, `p256-L64-i32-N4096` −0.084454924, `p1000-L64-i32-N1024` −0.084145054, `p4096-L64-i32-N1024` −0.084277824, `p2048-L128-i32-N1024-d7` −0.084303960 — and those six values are unchanged from the pre-change window and from the previous round.

## 8. Noise audit

This box is shared and the warm medians of this directory's window moved by single digits on the multi-millisecond torch and dense columns but by up to 24.1 % on sub-0.15 ms fused calls (the pre-change window was itself measured twice, 46 comparable medians, median |Δ| 2.0 %).
The current window's honest bound is measured, not assumed: a code-identical control reads 0.951–1.118 against it, the `t1a1_auto_r2` tree re-run in the same session reads −6.8 % to +6.2 % against its own committed `batch_forward` medians (−11.3 % to +0.1 % on `SR_fitness`), and the torch arm reproduces within −3.5 % to +9.7 % (`../t1a1_auto_r3/README.md` §8).
Treat a ratio below ~1.2× on a fused cell as inside this shared box's noise; the recorded window's own spread is visible in the `min_ms` column of `../t1a1_auto_r3/RENDERED.md` (e.g. `p256-L64-i32-N1024` `batch_forward` median 0.0449 ms against a 0.0424 ms minimum, this directory's `p256-L32-i8-N1024` `sr_fitness` median 0.1382 ms against a 0.1128 ms minimum).

## 9. Caveats and coverage

The pre-change window's `evoxc-dense-iree` arm on `p2048-L128-i32-N1024-d7` was not scheduled; `campaign.json`'s `protocol.scope_dropped` (mirrored in `cell_exclusions.entries`) records the reason: `explicit EVOGP_EVAL_MODE=dense on a population past EVOGP_DENSE_MAX_TREE_LEN declines into per-structure specialized+iree with unbounded artifact growth and OOM-kills the job unit (pop 2048 x 123-node trees); time evoxc-dense-iree-raised (the cap pinned to the cell's tree_len) instead`.
That arm exists in `../t1a1_fused/main/p2048-L128-i32-N1024-d7/` (`evoxc-dense-iree-raised`, `EVOGP_DENSE_MAX_TREE_LEN=128`): 105.7609 ms `sr_fitness`, 110.9945 ms `batch_forward`, 1.2759 ms `forward`, 124.5304 ms `gp_step` — a different window and a different tree root, so it is a scale reference for the excluded control, not a same-session ratio.
`../t1a1_baseline/main/` ran a DIFFERENT tree root (`71f8337b…` against this directory's `576832c6…`, both with `evoxc_sha` `9f66feca…` identical) and scheduled only `sr_fitness,batch_forward`, so every cross-directory ratio mixes two `evogp` revisions and has no `forward` or `gp_step` counterpart there.
`torch-per-tree` (scheduled by the pre-change window only) had a LOWER `gp_step` than `torch-batched` on all six cells (17.7834 vs 23.7219 ms at `p256-L32-i8-N1024`): at these populations the operator layer is the generation's cost, so the per-tree Python walk's cheaper, cache-local evaluation wins the total even though its `sr_fitness` was 6.0–47.7× more expensive per call — the current campaign does not re-schedule that arm.
`../t1a1_fused/main/` ran the EXPLICIT route in its own session and predates the harness's fused counter surface, so its records carry `eval_mode = "fused"` but no `engine` and no `fused_cache` block and cannot prove a launch the way these directories' records do; it also predates the host-glue round, so its 0.0995–0.1673 ms `batch_forward` is a PRE-round figure that the current default now beats (0.0449–0.1152 ms).
Coverage limits: six cells at `tree_len` 32–128 and pops 256–4 096; nothing below the 16-tree floor, nothing at `rows <= 64` with 8–32-node trees, no multi-output (`output_len >= 2`) cell, and `evoxc-specialized-iree` is measured on pops 16/64 only (and not at all here).

## 10. Reproduce and gate

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-cache     # required: this box's $HOME is read-only
cd test/benchmark/evoxc
# current records
$PY campaign.py --out results/t1a1_auto_r3 \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-auto,torch-batched \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800 --timeout 1800
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto_r3 \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY summarize.py --results results/t1a1_auto_r3 --out results/t1a1_auto_r3/RENDERED.md
# the pre-change records kept in THIS directory (four arms), gated where they are
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto \
    --allow-snapshot --allow-missing --allow-digest-mismatch
```

Gates of record: **12 records / 356 checks / 0 failures** on `../t1a1_auto_r3/` (6 declared digest WARN), the same **12 records / 356 checks / 0 failures** on the previous round `../t1a1_auto_r2/`, and **23 records / 640 checks / 0 failures** on this directory (7 declared WARN — one `missing arms (allowed)` for the excluded dense arm on `d7` and six digest splits).
One process per config, never two timed configs at once, idle-device guard on every config; warm medians on this shared box are ±10 %, so compare a knob only inside its own cell and window.
