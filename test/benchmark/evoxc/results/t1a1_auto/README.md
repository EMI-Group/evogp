# `t1a1_auto/` — what the SHIPPED DEFAULT does now (`auto` → fused)

This directory is the campaign of record for the arm `evoxc-auto`: no `EVOGP_EVAL_MODE` and no `EVOGP_EVOXC_BACKEND` exported (each record's `env.applied_knobs` reads `{"eval_mode": null, "backend": null, "knobs": {}}`), i.e. exactly the policy a user gets from a plain install.
Read it against `../t1a1_fused/main/` (the EXPLICIT `EVOGP_EVAL_MODE=fused` column) and `../t1a1_baseline/main/` (the PRE-FUSED default, where the same arm id `evoxc-auto-iree` resolved to dense+iree and where neither `forward` nor `gp_step` was scheduled).
The machine-generated rendering of these records is `RENDERED.md` in this directory; this file is the human one-screen summary.

## 1. What this campaign answers

The library's `auto` policy now resolves to the fused (Triton) route on a device box, so the question is what the released default — not an opt-in flag — actually costs per call and per generation.
Six cells x four arms, `warmup 2`, `reps 5` sync-bracketed warm samples (medians), cold first call reported separately, one process per `(cell, arm)`, sequential, `seed 0`, `EVOXC_DISABLE_LOCATIONS=1`: `evoxc-auto`, `torch-batched`, `torch-per-tree`, `evoxc-dense-iree` over `sr_fitness`, `batch_forward`, `forward` and `gp_step` (`campaign.json`: 23 configs, 23 `ok`, 0 failures, elapsed 323.6 s, 23/23 on an idle device, `contaminated` false for all 23).
`forward` was scheduled ON TOP of the three headline workloads so the gate has no scoped-schedule digest gap — a campaign limited to `sr_fitness,batch_forward` reports a `FAIL … no digest anywhere in the cell` class for the value workload it never ran (see `../t1a1_baseline/`), and that is why this one gates at 0 failures.
The `evoxc-dense-iree` arm is the same-window control: it is the route the same policy resolved to before the fused route existed.

**Verdict.** On this device box the fused default is **38.4×–141.7× faster than `torch-batched` per `batch_forward` call** and **7.6×–108.1× faster per whole GP generation** across the six cells, with the caveat that `auto` reaches the fused route only when triton is importable, a torch device is usable and `pop_size >= EVOGP_FUSED_MIN_POP` (16) — otherwise it stays on the dense/specialized answer it always had.
Ratios are `torch median / evoxc-auto median` (higher = the default is faster), computed from the records below, not quoted from any report.

| cell | pop | tree len (trace) | input len | rows | trees (nodes, seed 0) | distinct structures |
|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 256 | 32 | 8 | 1 024 | depth 5: mean 15.9, max 31 | 225 |
| `p256-L64-i32-N1024` | 256 | 64 | 32 | 1 024 | depth 5: mean 15.9, max 31 | 232 |
| `p256-L64-i32-N4096` | 256 | 64 | 32 | 4 096 | depth 5: mean 15.9, max 31 | 232 |
| `p1000-L64-i32-N1024` | 1 000 | 64 | 32 | 1 024 | depth 5: mean 16.2, max 31 | 841 |
| `p4096-L64-i32-N1024` | 4 096 | 64 | 32 | 1 024 | depth 5: mean 15.7, max 31 | 3 266 |
| `p2048-L128-i32-N1024-d7` | 2 048 | 128 | 32 | 1 024 | depth 7: mean 42.9, max 123 | 1 665 |

## 2. Route proof — the table that makes the claim falsifiable

Per cell, from that cell's `evoxc-auto` record: `routes.sr_fitness.*` and the record-level `fused_cache` window (`stats_before_workloads` → `stats_after`).

| cell | eval_mode | backend | engine | would_use_fused | fused_fallback_backend | served_every_workload | fused_calls | fused_declines | fused_kernel_launches | fused_kernel_compiles | fused_xt_hits | fused_xt_misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |
| `p256-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |
| `p256-L64-i32-N4096` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |
| `p1000-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |
| `p4096-L64-i32-N1024` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |
| `p2048-L128-i32-N1024-d7` | auto | iree | fused | true | numpy | true | 24 | 0 | 24 | 3 | 15 | 1 |

`fused_reason` on every cell reads `auto policy: device available, triton importable and pop_size=<pop> >= 16 (EVOGP_FUSED_MIN_POP) -> fused (Triton) on the torch device (specialized fallback on numpy)`, i.e. the resolver itself names the fused engine rather than a fallback.
`fused_calls` 24 = first call + 2 warmups + 5 reps for each of the three timed value/loss workloads; `fused_kernel_launches` 24 = one launch per call with 0 declines.
`fused_kernel_compiles` 3 is one first launch per kernel specialization (value, loss, per-row) even where Triton served it from disk (`src/evogp/tree/FUSED.md` §7 item 7); `fused_xt_hits`/`fused_xt_misses` are the transpose memo (`fused_xt_entries` 1, 1 miss then hits), and `fused_scratch_bytes`/`fused_scratch_entries` (2) are the per-cell geometry, so they differ per cell while the counters above do not.

Why this table is the proof: the fused route compiles no evoxc graph, so `cache.compile_misses_first_call` is 0 by construction on every `evoxc-auto` record (checked on all three workloads of all six cells) and cannot distinguish a served call from a silent `FusedUnsupported` fallback.
A positive `fused_kernel_launches` delta is the only existence proof, and `check.py` gates any record whose RESOLVED engine is `fused` as a fused record (`is_fused_record`), FAILing on a non-`fused` `routes[*].eval_mode` or a malformed `fused_cache` block and WARNing when no launch is proven — so a silent fallback cannot hide inside an `auto` record.
`fused_fallback_backend` reads `numpy` on every cell because no explicit backend was exported; with `EVOGP_EVOXC_BACKEND` set, a decline would land on specialized+iree instead.

## 3. Headline table — warm medians per call and per generation

`evoxc-auto` medians are `workloads.*.median_ms` and `gp_step.median_total_ms`; the control column is the same cell's `evoxc-dense-iree` record in the same window (`not scheduled` = the arm does not exist for that cell).

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | auto `gp_step` | vs torch-per-tree / torch-batched, `sr_fitness` | same, `batch_forward` | same, `forward` | same, `gp_step` | `evoxc-dense-iree` sf / bf / fw / gp |
|---|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 0.1382 ms | 0.1023 ms | 0.1053 ms | 2.6907 ms | 222.4× / 29.2× | 229.5× / 39.0× | 244.0× / 37.7× | 6.6× / 8.8× | 1.1046 / 1.8084 / 0.7228 / 5.7229 ms |
| `p256-L64-i32-N1024` | 0.1315 ms | 0.1169 ms | 0.1071 ms | 2.8174 ms | 237.8× / 35.1× | 205.4× / 39.2× | 220.2× / 42.7× | 6.3× / 7.6× | 2.1259 / 2.9494 / 0.8844 / 6.7334 ms |
| `p256-L64-i32-N4096` | 0.1244 ms | 0.1307 ms | 0.0983 ms | 2.5830 ms | 244.8× / 40.6× | 178.1× / 38.4× | 243.3× / 43.7× | 6.2× / 8.6× | 5.8723 / 7.6783 / 0.8789 / 10.6365 ms |
| `p1000-L64-i32-N1024` | 0.1659 ms | 0.1145 ms | 0.1081 ms | 2.6992 ms | 730.6× / 29.3× | 805.5× / 42.4× | 881.9× / 41.1× | 21.2× / 25.2× | 5.8309 / 7.7009 / 0.8577 / 12.5887 ms |
| `p4096-L64-i32-N1024` | 0.1829 ms | 0.1532 ms | 0.0987 ms | 2.8087 ms | 2 704.0× / 56.6× | 2 429.5× / 67.4× | 3 866.3× / 45.5× | 74.1× / 107.7× | 36.9693 / 47.0181 / 0.9024 / 56.3699 ms |
| `p2048-L128-i32-N1024-d7` | 0.2155 ms | 0.1879 ms | 0.1289 ms | 2.8400 ms | 2 715.3× / 123.7× | 2 754.8× / 141.7× | 4 114.2× / 121.3× | 42.5× / 108.1× | not scheduled |

Convention: `torch-*` median ÷ `evoxc-auto` median, same cell, same session, so >1 means the default is faster.
The route change inside one window is visible in the control column: the `sr_fitness` call moves from 1.1046–36.9693 ms on the dense route to 0.1244–0.2155 ms on the default, and dense's smallest gap anywhere is on `forward` (0.7228–0.9024 ms there against 0.0983–0.1289 ms on the default, i.e. still 7–9× slower).

## 4. Cold start — what the new default pays vs the old one

`first_call_ms` per workload per cell from this run's records, then the same-run `evoxc-dense-iree` control, then the pre-fused default's `sr_fitness` first call from `../t1a1_baseline/main/` (`evoxc-auto-iree`, whose `routes.*.would_use_dense` is `true`).

| cell | auto `sr_fitness` | auto `batch_forward` | auto `forward` | dense `sr_fitness` | dense `batch_forward` | dense `forward` | `t1a1_baseline` auto-iree `sr_fitness` |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 458.2 ms | 1.8 ms | 1.8 ms | 3 353.7 ms | 3 199.3 ms | 2 529.3 ms | 3 440.5 ms |
| `p256-L64-i32-N1024` | 445.9 ms | 1.9 ms | 1.7 ms | 6 571.7 ms | 6 375.2 ms | 4 810.6 ms | 6 609.0 ms |
| `p256-L64-i32-N4096` | 468.4 ms | 1.9 ms | 1.7 ms | 6 520.1 ms | 6 331.3 ms | 4 853.4 ms | 6 608.6 ms |
| `p1000-L64-i32-N1024` | 467.6 ms | 2.0 ms | 1.9 ms | 6 635.6 ms | 6 379.1 ms | 4 934.6 ms | 6 633.7 ms |
| `p4096-L64-i32-N1024` | 461.4 ms | 1.9 ms | 1.7 ms | 6 687.8 ms | 6 481.4 ms | 5 315.5 ms | 6 664.5 ms |
| `p2048-L128-i32-N1024-d7` | 448.9 ms | 2.0 ms | 1.7 ms | not scheduled | not scheduled | not scheduled | 14 460.5 ms |

The 446–468 ms `sr_fitness` entries are **not** a JIT: this campaign ran against a WARM Triton disk cache (the campaign's `$TRITON_CACHE_DIR` already held the three specializations), so each fresh process pays a ~0.45 s disk load of the loss kernel, and the two surfaces whose keys are already resident pay 1.7–2.0 ms.
From an EMPTY Triton cache the one-time cost is larger and is recorded in `src/evogp/tree/FUSED.md` §5B: **`batch_forward` 1 642–1 674 ms, `sr_fitness` 940–991 ms, `forward` 317–330 ms, i.e. ≈2.9 s of JIT for all three surfaces** (1.64 s for the value kernel alone).
That cost is paid **once per kernel specialization per machine**, never per population shape and never per generation — the opposite of the dense route's per-`(shape, used-function-set)` compile, which this window measures at 2 529–6 688 ms per cell here and which the pre-fused default paid as a 3 440–14 461 ms first call.
`TRITON_CACHE_DIR` must point at a writable directory: this box's `$HOME` is read-only and the first launch dies on `$HOME/.triton` without it.

## 5. The GP loop — one whole generation (the number a user feels)

`gp_step` per cell: total, its eval/ops split (`median_eval_ms` / `median_ops_ms`, in-place difference on the same step), the shares, and the ratio against both torch arms' `gp_step.median_total_ms`.

| cell | auto total | auto eval | auto ops | eval share | ops share | torch-per-tree / torch-batched total | vs torch-per-tree | vs torch-batched |
|---|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | 2.6907 ms | 0.2523 ms | 2.4384 ms | 9.38 % | 90.62 % | 17.8 / 23.7 ms | 6.61× | 8.82× |
| `p256-L64-i32-N1024` | 2.8174 ms | 0.2667 ms | 2.5507 ms | 9.47 % | 90.53 % | 17.9 / 21.5 ms | 6.34× | 7.64× |
| `p256-L64-i32-N4096` | 2.5830 ms | 0.2362 ms | 2.3468 ms | 9.14 % | 90.86 % | 16.0 / 22.1 ms | 6.18× | 8.56× |
| `p1000-L64-i32-N1024` | 2.6992 ms | 0.2694 ms | 2.4298 ms | 9.98 % | 90.02 % | 57.2 / 68.0 ms | 21.19× | 25.21× |
| `p4096-L64-i32-N1024` | 2.8087 ms | 0.2558 ms | 2.5529 ms | 9.11 % | 90.89 % | 208.1 / 302.6 ms | 74.08× | 107.73× |
| `p2048-L128-i32-N1024-d7` | 2.8400 ms | 0.2505 ms | 2.5895 ms | 8.82 % | 91.18 % | 120.8 / 307.1 ms | 42.53× | 108.14× |

Evaluation is **8.8–10.0 % of a generation** on the shipped default, so the operator layer — 2.35–2.59 ms of crossover/mutation/selection — is now the loop's bottleneck, and a generation is flat at 2.58–2.84 ms from 256×32 to 4 096×64 to 2 048×128.
That flatness is the change a user feels: at `p2048-L128-i32-N1024-d7` a generation is 2.8400 ms against 307.1158 ms on `torch-batched` and 120.7927 ms on `torch-per-tree`, and the control route of the same window (`evoxc-dense-iree`) pays 5.7229–56.3699 ms `gp_step` on the five cells where it could run.
Because nothing is re-planned, re-traced or recompiled per generation, `gp_step` pays no compile at all (`gp_step.compile_misses_total` is 0 on every `evoxc-auto` record), which is the structural reason the default wins a turnover loop where the opt-in `batch` route loses one by 17.8–71.5× over 8 generations at pop 256 (`../../batch_opt_round3/`).

## 6. What is still weak (numbers, no hedging)

Host glue is 22–49 % of a warm call, 0.031–0.062 ms absolute and flat across three orders of population size, so it is Python-level per-call overhead (route resolution, `_prepare`, the scratch lookup, the `contiguous` epilogue) rather than data movement — further gains need host-side work, not kernel work (`src/evogp/tree/FUSED.md` §3 and §7 item 4).
The tree walk repeats once per row block and per-program work scales with pop: `batch_forward` at `L=64` / `N=1024` reads 0.1169 / 0.1145 / 0.1532 ms at pop 256 / 1 000 / 4 096, i.e. flat to pop 1 000 and +34 % from pop 1 000 to 4 096, while 4× the rows at pop 256 costs 0.1169 → 0.1307 ms (+12 %).
`output_len >= 2` materializes the `(pop, rows, out_len)` prediction buffer in device memory, where the single-output path folds the reduction in registers and moves only `1/rows` of the traffic (`FUSED.md` §7 item 5).
There is no per-stage timer surface on this route and no zero-copy device export — a `(pop, rows, 1)` value tensor stays allocated and is never read back (the route's own accounting for those buffers and the fold scratch is `fused_scratch_bytes`, 33.6 MB–1.07 GB across these six cells).
The population floor is 16 (`EVOGP_FUSED_MIN_POP`), so a single `Tree.forward` / `Tree.SR_fitness` stays on specialized+numpy and never reaches the kernel.
The memory envelope (`EVOGP_FUSED_MAX_SCRATCH_MB`, default 8192) can decline a very large cell into the specialized fallback (`FUSED.md` §12), and the widest measured cell there is the 2 GiB `d8` shape.
And `EVOGP_EVOXC_BACKEND=numpy` still reports `engine="fused"` while every call declines — the engine name is not a served-call proof, which is exactly why this directory records launch counters (§2) instead of trusting it.

## 7. Numerics

Digest groups per value workload (distinct `digest.sha256` and the arms sharing each), read from the records; `sr_fitness` digests legitimately differ per arm (loss reduction order) and are not compared across arms.

| cell | `batch_forward` groups | `forward` groups | `evoxc-auto` group |
|---|---|---|---|
| `p256-L32-i8-N1024` | 1 (all four arms) | 1 (all four arms) | shares the torch sha |
| `p256-L64-i32-N1024` | 1 (all four arms) | 1 (all four arms) | shares the torch sha |
| `p256-L64-i32-N4096` | 1 (all four arms) | 1 (all four arms) | shares the torch sha |
| `p1000-L64-i32-N1024` | 2 | 2 | its own group, with `evoxc-dense-iree` |
| `p4096-L64-i32-N1024` | 2 | 2 | its own group, with `evoxc-dense-iree` |
| `p2048-L128-i32-N1024-d7` | 2 | 2 | its own group (the only evoxc arm scheduled) |

The split is the documented NaN-payload / low-bit class, which is why the gate needs `--allow-digest-mismatch` (`VALIDATION.md`; `FUSED.md` §11): `batch_forward` and `forward` are bitwise on all three function sets with `max_abs = 0.0` on the finite space, `SR_fitness` agrees to ≤3.8e-7 relative, and the NaN/±inf placement matches everywhere (`nan` counts identical inside each cell, e.g. 45 056 on the `d7` cell and 27 648 on `p4096-L64-i32-N1024`).
The end-to-end check is stronger than the digests: `gp_step.best_fitness` agrees across every arm of every cell to ≤7.5e-9 absolute — `p256-L32-i8-N1024` −0.084430024, `p256-L64-i32-N1024` −0.084812447, `p256-L64-i32-N4096` −0.084454924, `p1000-L64-i32-N1024` −0.084145054, `p4096-L64-i32-N1024` −0.084277824, `p2048-L128-i32-N1024-d7` −0.084303960.

## 8. Noise audit

This campaign was measured twice with the same protocol (a first window, then the committed window): 46 comparable `sr_fitness`/`batch_forward` medians, median |Δ| 2.0 %, max 24.1 %.
The committed window replaced the first window's files in this directory, so those deltas are quoted from the measuring session rather than re-derivable from the committed records — the one end of the pair that is recorded here is the current value (`p256-L32-i8-N1024` `batch_forward` 0.1023 ms).
Every large delta sits on a sub-0.15 ms fused call — e.g. `p256-L32-i8-N1024` `batch_forward` 0.1347 → 0.1023 ms, a 0.03 ms absolute swing that reads as −24 % — while the multi-millisecond torch and dense columns move by single digits.
Treat a ratio below ~1.2× on a fused cell as inside this shared box's noise; the recorded window's own spread is visible in the `min_ms` column of `RENDERED.md` (e.g. `p256-L32-i8-N1024` `sr_fitness` median 0.1382 ms against a 0.1128 ms minimum).

## 9. Caveats and coverage

The `evoxc-dense-iree` arm on `p2048-L128-i32-N1024-d7` was not scheduled; `campaign.json`'s `protocol.scope_dropped` (mirrored in `cell_exclusions.entries`) records the reason: `explicit EVOGP_EVAL_MODE=dense on a population past EVOGP_DENSE_MAX_TREE_LEN declines into per-structure specialized+iree with unbounded artifact growth and OOM-kills the job unit (pop 2048 x 123-node trees); time evoxc-dense-iree-raised (the cap pinned to the cell's tree_len) instead`.
That arm exists in `../t1a1_fused/main/p2048-L128-i32-N1024-d7/` (`evoxc-dense-iree-raised`, `EVOGP_DENSE_MAX_TREE_LEN=128`): 105.7609 ms `sr_fitness`, 110.9945 ms `batch_forward`, 1.2759 ms `forward`, 124.5304 ms `gp_step` — a different window and a different tree root, so it is a scale reference for the excluded control, not a same-session ratio.
`../t1a1_baseline/main/` ran a DIFFERENT tree root (`71f8337b…` against this run's `576832c6…`, both with `evoxc_sha` `9f66feca…` identical) and scheduled only `sr_fitness,batch_forward` (`gp_reps 5`, 25 configs, elapsed 622.7 s), so every cross-directory ratio mixes two `evogp` revisions and has no `forward` or `gp_step` counterpart there.
`torch-per-tree`'s `gp_step` is LOWER than `torch-batched`'s on all six cells (17.7834 vs 23.7219 ms at `p256-L32-i8-N1024`, 15.9673 vs 22.0999 ms at `p256-L64-i32-N4096`): at these populations the operator layer is the generation's cost, so the per-tree Python walk's cheaper, cache-local evaluation wins the total even though its `sr_fitness` is 6.0–47.7× more expensive per call than `torch-batched`'s.
`../t1a1_fused/main/` ran the EXPLICIT route and predates the harness's fused counter surface, so its records carry `eval_mode = "fused"` but no `engine` and no `fused_cache` block and cannot prove a launch the way this directory's records do.
Coverage limits: six cells at `tree_len` 32–128 and pops 256–4 096; nothing below the 16-tree floor, nothing at `rows <= 64` with 8–32-node trees, no multi-output (`output_len >= 2`) cell, and `evoxc-specialized-iree` is measured on pops 16/64 only (and not at all here).

## 10. Reproduce and gate

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
export TRITON_CACHE_DIR=$TMPDIR/triton-cache     # required: this box's $HOME is read-only
cd test/benchmark/evoxc
$PY campaign.py --out results/t1a1_auto \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
    --arms evoxc-auto,torch-batched,torch-per-tree,evoxc-dense-iree \
    --workloads sr_fitness,batch_forward,forward,gp_step \
    --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 3 --seed 0 --guard-wait 1800 --timeout 1800
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_auto \
    --allow-snapshot --allow-missing --allow-digest-mismatch
$PY summarize.py --results results/t1a1_auto --out results/t1a1_auto/RENDERED.md
```

Gate of record: **23 records / 640 checks / 0 failures**, plus 7 declared WARN — 1 `missing arms (allowed)` (`evoxc-dense-iree` on the `d7` cell, `campaign status: never scheduled`) and 6 digest splits between the evoxc and torch arms (both value workloads on `p1000-L64-i32-N1024`, `p4096-L64-i32-N1024` and `p2048-L128-i32-N1024-d7`).
One process per config, never two timed configs at once, idle-device guard on every config; warm medians on this shared box are ±10 %, so compare a knob only inside its own cell and window.
`RENDERED.md` in this directory is the machine-generated rendering of the same records (per-workload tables with `min_ms`, the speedup matrix, the cold-compile and digest tables) — this README is the human one-screen summary of it.
