# `t1a1_auto_r3/` — the shipped default (`auto` → fused) on the release tree
This directory is the CURRENT campaign of record for the arm `evoxc-auto`: no `EVOGP_EVAL_MODE` and no `EVOGP_EVOXC_BACKEND` exported (each record's `env.applied_knobs` reads `{"eval_mode": null, "backend": null, "knobs": {}}`), i.e. exactly the policy a plain install gets.
It re-measures the same six cells x two arms as `../t1a1_auto_r2/` on tree `919f7af`, the release tree (the route-policy memo, the fused plan fast path and this round's test/battery fixes — none of which is an opt-in flag).
`../t1a1_auto/` and `../t1a1_auto_r2/` are left untouched: their JSON are the pre-change and previous-round evidence, and they are read here only as committed numbers.
The machine-generated rendering of these records is `RENDERED.md`; this file is the human one-screen summary.
Read it against `../t1a1_fused/main/` (the explicit `EVOGP_EVAL_MODE=fused` column of an earlier session) and `../t1a1_baseline/main/` (the pre-fused default).
## 1. What this campaign answers
What does the released default cost per call and per generation on the release tree, and how much of the distance to the previous window is the tree rather than the box?
Six cells x (`evoxc-auto`, `torch-batched`) over `sr_fitness` + `batch_forward` + `forward` + `gp_step`, `warmup 2`, `reps 5` sync-bracketed warm samples (medians), `gp_warmup 1` / `gp_reps 3`, cold first call reported separately, one process per `(cell, arm)`, sequential, `seed 0`, `EVOXC_DISABLE_LOCATIONS=1`: `campaign.json` records 12 configs, 12 `ok`, 0 failures, elapsed 113.0 s, 12/12 on an idle device, `contaminated` false for all 12.
Two same-session `--tree-root` controls answer the second half (§4): the previous-round tree (`5a699ca`) and the pre-memo tree (`091a6a8`), plus a code-identical control (`cfd2f92`, whose diff against the measured tree is documentation only).
| cell | pop | tree len (trace) | input len | rows | trees (nodes, seed 0) | distinct structures |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 256 | 32 | 8 | 1 024 | depth 5: mean 15.9, max 31 | 225 |
| `p256-L64-i32-N1024` | 256 | 64 | 32 | 1 024 | depth 5: mean 15.9, max 31 | 232 |
| `p256-L64-i32-N4096` | 256 | 64 | 32 | 4 096 | depth 5: mean 15.9, max 31 | 232 |
| `p1000-L64-i32-N1024` | 1 000 | 64 | 32 | 1 024 | depth 5: mean 16.2, max 31 | 841 |
| `p4096-L64-i32-N1024` | 4 096 | 64 | 32 | 1 024 | depth 5: mean 15.7, max 31 | 3 266 |
| `p2048-L128-i32-N1024-d7` | 2 048 | 128 | 32 | 1 024 | depth 7: mean 42.9, max 123 | 1 665 |
## 2. Headline numbers — warm medians, this window
`×torch` is `torch-batched median / evoxc-auto median` on the same cell in the same record set (higher = the default is faster), computed from the JSON in this directory, not quoted from any report.
| cell | `batch_forward` ms | ×torch | `sr_fitness` ms | ×torch | `forward` ms | ×torch | `gp_step` ms (eval) | ×torch |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.0493 | 80.2 | 0.0670 | 60.2 | 0.0455 | 85.8 | 3.053 (0.471) | 7.7 |
| `p256-L64-i32-N1024` | 0.0449 | 108.4 | 0.0572 | 78.0 | 0.0399 | 121.0 | 2.842 (0.437) | 8.3 |
| `p256-L64-i32-N4096` | 0.0522 | 96.2 | 0.0650 | 77.6 | 0.0391 | 115.7 | 2.900 (0.433) | 7.7 |
| `p1000-L64-i32-N1024` | 0.0500 | 96.8 | 0.0619 | 79.2 | 0.0390 | 119.6 | 2.841 (0.435) | 24.6 |
| `p4096-L64-i32-N1024` | 0.0790 | 131.8 | 0.0886 | 118.0 | 0.0437 | 101.8 | 3.072 (0.466) | 99.8 |
| `p2048-L128-i32-N1024-d7` | 0.1152 | 228.9 | 0.1262 | 208.8 | 0.0722 | 220.1 | 3.282 (0.468) | 92.3 |
**Verdict.** Warm `Forest.batch_forward` reads **0.0449–0.1152 ms**, i.e. **80.2×–228.9× the plain-torch batched evaluator**, with `SR_fitness` at **0.0572–0.1262 ms (60.2×–208.8×)**, `forward` at **0.0390–0.0722 ms (85.8×–220.1×)** and a whole GP generation at **2.841–3.282 ms (7.7×–99.8×)**.
Evaluation is **14.2–15.4 %** of that generation (0.433–0.471 ms of 2.841–3.282 ms), so the operator layer is still the loop's cost and the per-call win does not shorten it (§7).
## 3. Route proof — the table that makes the claim falsifiable
Per cell, from that cell's `evoxc-auto` record: `routes.sr_fitness.*`, `fused_cache.served_every_workload` and the record-level `fused_cache.stats_after` window (first call + 2 warmups + 5 reps for each of the three timed value/loss workloads).
| cell | eval_mode | backend | engine | would_use_fused | fallback backend | served_every_workload | fused_calls | declines | launches | bound launches | compiles | plan hits/misses | xt hits/misses | tag hits/misses | scratch MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 33.59 |
| `p256-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 67.17 |
| `p256-L64-i32-N4096` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 268.50 |
| `p1000-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 262.40 |
| `p4096-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 1 074.79 |
| `p2048-L128-i32-N1024-d7` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 21 | 3 | 21/3 | 1/1 | 2/1 | 1 074.79 |
`fused_reason` on every cell reads `auto policy: device available, triton importable and pop_size=<pop> >= 16 (EVOGP_FUSED_MIN_POP) -> fused (Triton) on the torch device (specialized fallback on numpy)`, i.e. the resolver itself names the fused engine rather than a fallback.
`fused_calls` 24 = one launch per timed call with 0 declines; `bound launches` 21 is the prepared-launch fast path serving every warm call (3 cold calls build the three plan entries, `plan hits/misses` 21/3).
## 4. What moved since `t1a1_auto_r2` — one-session four-way A/B
Every leg below is a separate `campaign.py` run in this session, one process per config, the same six cells, the same workloads, `evoxc-auto` only; the arms are the same tree at four commits, so the comparison carries no cross-window drift beyond the process-to-process spread measured in §8.
| leg | tree | what it has |
|---|---|---|
| **A** | `5a699ca` | the previous round's tree (the `t1a1_auto_r2` commit): prepared-call memo + bound launch |
| **B** | `091a6a8` | A + the fused plan fast path and geometry memo (`940ec13`, re-landed) |
| **C** | `cfd2f92` | B + the route-policy memo (`0880fbc`); the docs-only commit on top of it — code-identical to the tree measured here |
| **D** | `919f7af` | this directory: C + this round's `conftest`/`fused_check` fixes (no evaluation code) |
| cell | `batch_forward` D/A | `batch_forward` C/B | `sr_fitness` D/A | `sr_fitness` C/B | `forward` D/A | `forward` C/B | `gp_step` D/A |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.576 | 0.897 | 0.629 | 0.927 | 0.566 | 0.895 | 1.059 |
| `p256-L64-i32-N1024` | 0.582 | 0.840 | 0.605 | 0.830 | 0.545 | 0.860 | 1.018 |
| `p256-L64-i32-N4096` | 0.579 | 0.879 | 0.606 | 0.818 | 0.507 | 0.881 | 1.035 |
| `p1000-L64-i32-N1024` | 0.566 | 0.924 | 0.609 | 0.970 | 0.513 | 0.780 | 0.968 |
| `p4096-L64-i32-N1024` | 0.635 | 0.880 | 0.637 | 0.907 | 0.542 | 0.836 | 0.982 |
| `p2048-L128-i32-N1024-d7` | 0.764 | 0.964 | 0.748 | 0.942 | 0.645 | 0.867 | 1.140 |
**D/A (the round, same session):** the release tree is **1.31×–1.77×** faster on `batch_forward`, **1.34×–1.65×** on `sr_fitness` and **1.55×–1.97×** on `forward`, and unchanged on the generation (`gp_step` 0.968–1.140, i.e. inside the spread).
**C/B (the route-policy memo alone):** **3.7 %–19.0 %** on `batch_forward`, **3.1 %–22.2 %** on `sr_fitness`, **11.7 %–28.2 %** on `forward`.
**A/B (the plan fast path alone):** 1.30×–1.77× on `batch_forward`, 1.24×–1.56× on `sr_fitness`, 1.34×–1.77× on `forward` — i.e. the fast path is the bulk of the round and the memo is the rest.
**D/C (code-identical control):** 0.951–1.118, which is this box's process-to-process spread on a sub-0.12 ms call (§8) and confirms the measured tree reads what `cfd2f92` reads.
`gp_eval` (the evaluation slice inside `gp_step`, timed in place on the same population) moves with `gp_step` rather than with the value workloads: 1.010–1.156 `D/A` and 0.919–1.062 `D/C`, i.e. the call-level win of §2 does not reach the generation (§7).
## 5. Delta against the committed `t1a1_auto_r2` numbers
Read `../t1a1_auto_r2/`'s committed medians against this directory directly and the round reads −42.7 % / −42.4 % / −38.5 % / −47.2 % / −37.2 % / −21.1 % on `batch_forward`, −44.3 % / −42.8 % / −42.6 % / −43.6 % / −36.2 % / −27.0 % on `sr_fitness` and −46.6 % / −45.6 % / −45.4 % / −52.4 % / −43.4 % / −35.0 % on `forward`.
That ratio is legitimate here for once, and the same-session controls are why: leg **A** re-runs the r2 tree on this box and reproduces the committed r2 numbers within **−6.8 %…+6.2 %** (`batch_forward`), **−11.3 %…+0.1 %** (`sr_fitness`) and **−7.3 %…+7.7 %** (`forward`), while the torch arm reproduces within **−3.5 %…+9.7 %** — so the box has not moved and the −42 %…−21 % is the tree.
The full-figure version of the same statement is §4's D/A column; this section exists because a reader will reach for the two committed directories first.
## 6. Numerics / digests
`batch_forward` and `forward` carry a single digest group across the two arms on `p256-L32`, `p256-L64` and `p256-L64-i32-N4096`, and split on `p1000`, `p4096` and `p2048-L128-...-d7`.
The split is the documented NaN-payload / low-bit class (`VALIDATION.md`, `FUSED.md` §11): `batch_forward`/`forward` are bitwise on all three function sets with `max_abs = 0.0` on the finite space, and the losses agree to ≤3.8e-7 relative.
`sr_fitness` digests legitimately differ from the torch arm on every cell (loss reduction order) and are not compared across arms; the gate needs `--allow-digest-mismatch` and declares 6 WARN, all of them these value-workload splits.
No evaluation code changed in this round (§4's D/C leg), so the digest pattern is the same as the previous round's, cell for cell.
## 7. Where a generation's cost sits
`gp_step` is 2.841–3.282 ms with evaluation at 0.433–0.471 ms (**14.2–15.4 %**) and the operator layer at 2.404–2.815 ms, so the loop is the operator layer's cost.
That generation is unchanged against the previous round's tree (`gp_step` D/A 0.968–1.140, the eval slice 1.010–1.156) even though the per-call workloads of §2 improved by 1.31×–1.97×, so the round's win does not shorten a training loop.
The loop pays no compile — `gp_step.compile_misses_total` is 0 and every step records `compile_misses` 0 — but the record carries no fused-plan window for the loop (its per-step counters are the evoxc engine's, not the fused route's), so how much of the loop's evaluation hits the plan fast path is not measurable from these records.
Its evaluation slice is **7–8× a standalone `sr_fitness` call** (0.433–0.471 ms against 0.057–0.126 ms in §2), i.e. it carries per-generation work — new trees, new data planes, the fitness wrapper's own setup — that the single-call workload does not, and that is where the remaining evaluation cost lives.
## 8. Noise audit
The same-code control of §4 (D/C) spreads **0.951–1.118**, the r2 tree re-run (A vs the committed r2 numbers) **−11.3 %…+7.7 %**, and the torch arm (unchanged code by construction) **−3.5 %…+9.7 %**.
Every relative delta of this round sits on a sub-0.12 ms call, where a 0.005 ms absolute move reads as 10 %; the multi-millisecond torch column moves by single digits.
Treat a ratio inside ~1.2× on a fused cell as inside this box's noise, and read the `min_ms` column of `RENDERED.md` for the recorded window's own spread (e.g. `p256-L64-i32-N1024` `batch_forward` median 0.0449 ms against a 0.0424 ms minimum).
## 9. Caveats and coverage
Coverage limits: six cells at `tree_len` 32–128 and pops 256–4 096; nothing below the 16-tree floor, nothing at `rows <= 64` with 8–32-node trees, no multi-output (`output_len >= 2`) cell.
The `dense` and `torch-per-tree` controls are not scheduled in this campaign; `../t1a1_auto/` has the dense column of an earlier window and `../t1a1_fused/main/` the explicit-route column of its own session.
A and B are separate processes from C and D, so §4's A/B ratio carries the process-to-process spread twice; only D-vs-C is a same-code pair.
`../t1a1_baseline/main/` ran a different tree root and only two workloads, so no ratio there crosses into this file.
## 10. Reproduce and gate
```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-cache     # required: this box's $HOME is read-only
cd test/benchmark/evoxc
$PY campaign.py --out results/t1a1_auto_r3 \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-auto,torch-batched \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800 --timeout 1800
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto_r3 \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY summarize.py --results results/t1a1_auto_r3 --out results/t1a1_auto_r3/RENDERED.md
# the three same-session controls of §4 and §5 (records under $TMPDIR, not committed)
git archive 5a699ca | tar -x -C /tmp/evogp_r2tree && git archive 091a6a8 | tar -x -C /tmp/evogp_prememo \
    && git archive cfd2f92 | tar -x -C /tmp/evogp_memo
$PY campaign.py --out /tmp/ab_r2tree   --tree-root /tmp/evogp_r2tree  --cells <the six ids above> --arms evoxc-auto --workloads sr_fitness,batch_forward,forward,gp_step --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800
$PY campaign.py --out /tmp/ab_prememo  --tree-root /tmp/evogp_prememo --cells <the six ids above> --arms evoxc-auto --workloads sr_fitness,batch_forward,forward,gp_step --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800
$PY campaign.py --out /tmp/ab_memo     --tree-root /tmp/evogp_memo    --cells <the six ids above> --arms evoxc-auto --workloads sr_fitness,batch_forward,forward,gp_step --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800
```
Gate of record: **12 records / 356 checks / 0 failures**, plus 6 declared WARN — all of them digest splits (both value workloads on `p1000`, `p4096` and `p2048-L128-...-d7`).
One process per config, never two timed configs at once, idle-device guard on every config.
