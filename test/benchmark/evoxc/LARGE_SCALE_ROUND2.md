# Large-scale round 2: the batch route after the dispatch-formation, `auto`-policy and artifact-tier work
This round re-asks `LARGE_SCALE.md`'s question — is `EVOGP_EVAL_MODE=batch` a better large-population evaluator than the shipped `dense+iree` route? — after three improvements landed on the library side: `EVOGP_BATCH_DISPATCH_FORMATION` (`auto` ⇒ IREE aggressive dispatch formation on rocm/cuda merged plans), a rewritten `auto` policy (the fixed rows ceiling became a memory envelope, and an over-cap population is now traced at its own longest tree rounded up to a power of two), and a persistent on-disk artifact tier for the merged graph (`EVOGP_EVOXC_ARTIFACT_DIR`).
The column is round 1's column plus one cell: nine cells over the population axis, the row axis, the node-count axis past the dense trace cap and the input-width axis (§1).
Every record of the headline column carries all three value workloads — `sr_fitness`, `forward` and `batch_forward` — so the per-row and the whole-population value paths are priced on the same populations as the loss call (§5.3), and the batch route's cold column is three plan compiles instead of one (§2.3).
**This report supersedes `LARGE_SCALE.md`'s numbers for the routes that changed** — the `batch` route (the formation knob) and `evoxc-auto` (the rows envelope and the trace-cap raise); its `dense` and `dense-raised` columns are re-measured here and agree with it within the box's noise band, while the two torch baselines read 1.2–9.3 % higher than in round 1 — a uniform, instrument-side drift (§2.5) still inside the ±10 % warm band. `LARGE_SCALE.md` keeps its `specialized`+numpy fallback column, its two declared-infeasible cells (`p8192` and `d8`) and its quadratic compile law; those are not re-measured here.
Box: one AMD Instinct MI300X VF (191.7 GB HBM), torch 2.13.0+rocm7.1, HIP 7.1.52802, IREE 3.11.0, numpy 2.5.4, python 3.12.3, `EVOXC_DISABLE_LOCATIONS=1`.
Protocol: one process per `(cell, arm)` behind the idle-device guard (`guard.gpu_pids.idle true` and `contaminated false` on every campaign entry that records the guard block; `gpu_use_pct` 0.0), sequential, `--workloads sr_fitness,forward,batch_forward` with one cold call timed separately per workload and `warmup=2` + `reps=5` sync-bracketed warm samples, `gp_warmup=1` and `gp_reps=3`, seed 0, a 900 s per-config cap (the pop-8 192 batch config used 811.6 s of it).
Provenance: evoxc `be145ee` (`/root/evoxc/.genesis/workers/worker_T1_A1`) on every record of the round; the tree under measurement is named per record — `main/` is a single re-measure at `6cdd68b` with all three value workloads, replacing a two-commit single-workload window (`c652230` on the four i32 cells of the population/row axes, `e41f704` from the pop-1000 cells on, which is where the rows envelope landed) whose last driver record is kept verbatim as `main/campaign.scoped-sr_fitness.json` and whose 49-config first invocation is transcribed in `main/campaign.invocations.md`, and `82ff822` (`auto_fix/`, `loop/`, `p8192/`), `720d621` (`band/`, three of the four pool columns), `63e6dc3` (`artifact/`, `pool/p2`), `61af76e` (`loop_auto_fix/`) and `7258b3d` (`ab_dispatch/`) carry the later fixes. §11.1 says which directory is which; every record states its own `env.tree_root_sha`.
The shipped batch route now needs an evoxc that declares the `dispatch_formation` compile option. That is not a hard dependency: the engine probes the adapter's own `KNOWN_OPTIONS["compile"]` (`evogp/tree/batch_chunks.py::supports_dispatch_formation`), and on an evoxc without it the effective spelling silently stays `default`, i.e. the pre-change graph — so an older checkout reproduces this report's BEFORE column exactly (§3), and only a malformed knob value is an error.
Every number below is read from a committed record under `results/largescale2/` or `results/largascale2/` (the second spelling holds only `ab_dispatch/`); nothing here was re-measured for the report. The `main/` table was re-measured once, as a whole, in this window; §2.5 lists what that moved.
## 1. The column
Nine cells, `common.large_cells()`'s ten minus the `d8` cell (still compile-infeasible, §1.1), scheduled by id with `--cells <ids>`; ids of the form `p<pop>-L<tree_len>-i<input_len>-N<rows>[-d<cap>]`:
| axis | cells | what it moves |
|---|---|---|
| population scale | `p2048-L64-i32-N1024`, `p4096-L64-i32-N1024`, `p8192-L64-i32-N1024` | population 2 048 → 8 192 at a fixed shape; `p8192` is the cell round 1 could not measure and this round can (§8) |
| row scale | `p256-L64-i32-N4096`, `p256-L64-i32-N16384`, `p1000-L64-i32-N4096`, `p1000-L64-i32-N16384` | `N` 4 096 / 16 384 at pop 256 and 1 000; the axis dense pays for and the axis the merged graph does not |
| node count past the cap | `p2048-L128-i32-N1024-d7` | 123-node trees (`max_layer_cnt` 7) at pop 2 048, i.e. past `EVOGP_DENSE_MAX_TREE_LEN` = 64; the cell the trace-cap raise exists for |
| input width | `p4096-L64-i64-N1024` | `input_len` 64 instead of 32 |
The memory rule is unchanged: `pop * tree_len * rows * 4` bytes for dense's largest intermediate, checked against `common.CELL_TENSOR_CEILING_BYTES` (8 GiB) by `common.cell_tensor_bytes` before a cell is scheduled.
The worst cell is `p1000-L64-i32-N16384` at 4.19 GB and `p8192-L64-i32-N1024` sits at 2.15 GB, so nothing here is memory-bound — and, since the `auto` memory envelope (`EVOGP_DENSE_AUTO_BUDGET_MB`, 8 192 MiB) is the same rule, no cell of the column reaches even half of it: `auto` never declines on memory here.
Populations come from the same `torch.manual_seed(0)` + `Forest.random_generate` recipe as every other column; the structure counts are read off each record's `structures` block (and match round 1's for the shared cells):
| cell group | distinct structures | mean / max nodes |
|---|---|---|
| pop 2 048, d5 | 1 681 | 16.00 / 31 |
| pop 4 096, d5 | 3 266 | 15.71 / 31 |
| pop 8 192, d5 | 6 544 | 15.80 / 31 |
| pop 256, d5 (both `N`) | 232 | 15.95 / 31 |
| pop 1 000, d5 (both `N`) | 841 | 16.21 / 31 |
| `p4096-L64-i64-N1024` | 3 298 | 15.71 / 31 |
| `d7` (`L128`) | 1 665 | 42.89 / 123 |
### 1.1 What was scheduled, and what was not
`main/` is eight cells × seven arms — `torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-dense-iree`, `evoxc-dense-iree-raised`, `evoxc-batch-iree` and `evoxc-batch-iree-default` — i.e. 55 records over all three value workloads (`--workloads sr_fitness,forward,batch_forward`, `--reps 5`), written by two sequential driver invocations of the same window (49 configs on the seven cells where every arm is safe, then 6 on the `d7` cell without the OOM-prone plain dense arm; `campaign.invocations.md` transcribes both, `campaign.json` keeps the second), and `p8192/` adds the same schedule minus the formation control (5 records, still `sr_fitness` only).
The `BEFORE` arm of every changed route is a real measurement, not a literature value: `evoxc-batch-iree-default` pins `EVOGP_BATCH_DISPATCH_FORMATION=default` (the pre-change compile) in the same window as the shipped `evoxc-batch-iree` (which writes no value, so it resolves `auto` ⇒ `aggressive`), and `evoxc-dense-iree-raised` pins `EVOGP_DENSE_MAX_TREE_LEN` to the cell's `tree_len` so the d7 cell has a dense arm at all.
| not measured | why |
|---|---|
| `evoxc-dense-iree` (default cap) on `p2048-L128-i32-N1024-d7` | past the cap dense declines into per-structure `specialized`+iree; the attempt OOM-killed the job unit (memcg 200 G, kernel log 11:02:24) and a bounded re-run recorded `status: timeout` at 247.6 s. The driver's note is in `main/campaign.json`'s `driver_env.note`; the `.log` is retained, there is no `.json` |
| `evoxc-batch-iree-default` on `p8192-L64-i32-N1024` | not scheduled: one 799 s compile per cell is the measurement, and repeating it for the control would have doubled the cell's cost |
| `evoxc-specialized-numpy` / `evoxc-specialized-iree` anywhere | round 1 already priced the fallback column (`LARGE_SCALE.md` §7.1) and this round's changed routes are the compiled ones |
| `p2048-L256-i32-N1024-d8` | still compile-infeasible for the batch route (~150–190 k emitted ops; round 1 recorded `timeout` at 900.63 s) |
| `EVOGP_BATCH_BANDS=auto`, `EVOGP_BATCH_ALIGNMENT`, `EVOGP_BATCH_SWITCH_OPS` at this scale | the band policy is settled (`BATCH.md` §14/§15) and the remaining knobs are not on the shipped path |
The committed campaign snapshots sum to 3 725.4 s (62.1 min) of driver wall plus the two band sweeps (1 136.5 + 165.3 s), i.e. ≈84 min of GPU as the snapshots record it; that undercounts the round twice over, because the driver overwrites `<dir>/campaign.json` and only the last invocation of each directory survives: `main/`'s second invocation is the 6-config / 794.3 s snapshot, while its first one contributed 49 configs / 2 330.6 s that no `campaign.json` holds (counting those, the round is ≈123 min), and the records are the evidence for the rest.
## 2. The tables
### 2.1 `sr_fitness` (the loss call), warm
Warm `median_ms` with `min_ms` in parentheses; `—` means no record for that cell/arm. `evoxc-batch-iree` is the AFTER route (formation `auto` ⇒ `aggressive`) and `evoxc-batch-iree-default` the BEFORE control (`default`); `evoxc-auto` is the corrected policy.
| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-batch-iree (AFTER) | evoxc-batch-iree-default (BEFORE) |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 281.36 (279.03) | **6.93** (6.91) | 11.01 (10.99) | 11.08 (11.07) | 10.94 (10.92) | 12.41 (12.27) | 14.00 (13.51) |
| `p4096-L64-i32-N1024` | 524.82 (523.04) | **10.43** (10.41) | 36.85 (36.82) | 37.15 (36.99) | 36.97 (36.82) | 21.59 (21.42) | 23.78 (23.53) |
| `p8192-L64-i32-N1024` | 998.38 (994.99) | **17.71** (17.68) | 76.03 (75.94) | 75.93 (75.65) | — | 41.93 (41.78) | — |
| `p256-L64-i32-N4096` | 31.58 (31.52) | **4.95** (4.90) | 5.82 (5.77) | 5.81 (5.79) | 5.82 (5.81) | 5.29 (5.24) | 5.18 (5.13) |
| `p256-L64-i32-N16384` | 30.85 (30.17) | **10.44** (10.41) | 78.03 (77.90) | 77.99 (77.79) | 78.51 (78.04) | 12.58 (12.34) | 12.45 (12.42) |
| `p1000-L64-i32-N4096` | 136.41 (131.94) | **10.13** (10.08) | 39.38 (39.36) | 39.31 (39.28) | 39.57 (39.45) | 13.01 (12.79) | 13.17 (13.03) |
| `p1000-L64-i32-N16384` | 123.00 (122.02) | **32.67** (32.57) | 117.74 (117.68) | 118.29 (118.24) | 118.41 (118.14) | 37.61 (37.28) | 37.25 (37.11) |
| `p2048-L128-i32-N1024-d7` | 610.55 (603.86) | 26.20 (26.18) | 105.79 (105.66) | — | 105.88 (105.77) | **20.32** (20.25) | 24.87 (24.71) |
| `p4096-L64-i64-N1024` | 511.97 (508.63) | **10.43** (10.42) | 37.25 (37.04) | 37.16 (36.91) | 37.16 (37.08) | 23.06 (22.87) | 24.05 (23.95) |
`p8192-L64-i32-N1024` is the only row not from `main/` — it is `p8192/`'s single-workload record, not re-measured (§8). The `d7` `evoxc-auto` record is now a dense-at-a-raised-cap call: its `auto_trace_cap` block reads `base_cap 64`, `auto 128`, `cap_pinned false`, so 105.79 ms is the raised dense graph against `evoxc-dense-iree-raised`'s 105.88 — the `specialized`+numpy fallback the previous window's `main/` record held (857.22 ms) has no surviving record in `main/`, because the re-measure overwrote it (§4.2).
The BEFORE control reproduces round 1's batch column (`LARGE_SCALE.md` §2.1: 12.79 / 23.24 / 5.23 / 12.61 / 13.03 / 37.41 / 25.07 / 24.56) to within −2.1 … +9.5 %, with the pop-2 048 cell the outlier (+9.5 %), so the AFTER column's change is attributable to the formation knob only to within this window's spread — §2.5 puts the per-cell delta next to the same-window control and §3.4 reads it.
### 2.2 The two changed routes, before → after
`batch` is `evoxc-batch-iree-default` → `evoxc-batch-iree` (same window, same population, §3); `auto` is `LARGE_SCALE.md` §7.1's `evoxc-auto` column → this round's, which is two separate fixes (§4).
| cell | batch BEFORE | batch AFTER | change | auto BEFORE (round 1) | auto AFTER | change | record set for `auto` |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 14.00 | 12.41 | **−11.4 %** | 10.99 | 11.01 | +0.2 % | `main/` |
| `p4096-L64-i32-N1024` | 23.78 | 21.59 | −9.2 % | 37.18 | 36.85 | −0.9 % | `main/` |
| `p8192-L64-i32-N1024` | — | 41.93 | — | 76.34 | 76.03 | −0.4 % | `p8192/` |
| `p256-L64-i32-N4096` | 5.18 | 5.29 | +2.0 % | 50.47 | 5.82 | **8.67×** | `main/` |
| `p256-L64-i32-N16384` | 12.45 | 12.58 | +1.1 % | 110.21 | 78.03 | **1.41×** | `main/` |
| `p1000-L64-i32-N4096` | 13.17 | 13.01 | −1.2 % | 197.31 | 39.38 | **5.01×** | `main/` |
| `p1000-L64-i32-N16384` | 37.25 | 37.61 | +1.0 % | 446.60 | 117.74 | **3.79×** | `main/` |
| `p2048-L128-i32-N1024-d7` | 24.87 | 20.32 | **−18.3 %** | 891.23 | 105.79 | **8.42×** | `main/` (post-raise, §4.2) |
| `p4096-L64-i64-N1024` | 24.05 | 23.06 | −4.1 % | 37.23 | 37.25 | +0.1 % | `main/` |
### 2.3 Cold `first_call_ms`
The cold call is the first synchronized evaluation in a fresh process, i.e. it contains the whole compile; it is never part of the warm medians. With three value workloads scheduled, every arm now pays one cold call per workload — the process's first workload (`sr_fitness`) is the true cold start and the two value workloads pay their own graph compile when their turn comes. The table is the loss workload's first call:
| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-batch-iree | evoxc-batch-iree-default |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 426.90 | 72.30 | 6 770.20 | 7 517.50 | 6 740.60 | 43 786.40 | 50 734.80 |
| `p4096-L64-i32-N1024` | 664.90 | 70.30 | 6 831.50 | 6 669.00 | 6 819.80 | 154 371.10 | 193 068.50 |
| `p8192-L64-i32-N1024` | 1 121.90 | 71.87 | 6 839.23 | 6 788.16 | — | 799 412.45 | — |
| `p256-L64-i32-N4096` | 158.60 | 66.90 | 6 704.80 | 6 710.40 | 6 508.60 | 4 171.50 | 4 176.50 |
| `p256-L64-i32-N16384` | 157.60 | 71.60 | 6 413.60 | 6 463.40 | 6 476.70 | 5 216.90 | 5 308.60 |
| `p1000-L64-i32-N4096` | 264.20 | 76.10 | 6 907.00 | 6 775.10 | 6 643.60 | 16 506.40 | 17 466.40 |
| `p1000-L64-i32-N16384` | 254.90 | 91.10 | 6 642.90 | 6 913.00 | 6 693.80 | 18 508.30 | 19 691.50 |
| `p2048-L128-i32-N1024-d7` | 744.90 | 86.50 | 14 315.00 | — | 14 409.10 | 157 907.90 | 259 428.50 |
| `p4096-L64-i64-N1024` | 653.40 | 69.60 | 6 990.70 | 7 037.80 | 7 054.90 | 169 785.50 | 199 064.80 |
The batch arms are the ones whose cold column became a per-workload sum, because the batch route compiles ONE plan per (population structure multiset, workload shape): three plans, three compiles. The loss plan dominates, and the two value plans are not small:
| cell | loss plan | `batch_forward` plan | `forward` plan | three-workload total | control's total |
|---|---|---|---|---|---|
| `p256-L64-i32-N4096` | 4 171.5 | 2 826.3 | 1 695.3 | **8 693.1** | 8 669.3 |
| `p256-L64-i32-N16384` | 5 216.9 | 3 202.3 | 1 822.5 | **10 241.7** | 10 309.2 |
| `p1000-L64-i32-N4096` | 16 506.4 | 10 682.6 | 5 223.2 | **32 412.2** | 33 581.6 |
| `p1000-L64-i32-N16384` | 18 508.3 | 12 187.4 | 5 375.1 | **36 070.8** | 37 466.0 |
| `p2048-L64-i32-N1024` | 43 786.4 | 26 640.2 | 10 519.5 | **80 946.1** | 88 834.8 |
| `p4096-L64-i32-N1024` | 154 371.1 | 68 184.3 | 20 736.6 | **243 292.0** | 282 980.0 |
| `p4096-L64-i64-N1024` | 169 785.5 | 66 795.0 | 21 287.3 | **257 867.8** | 287 506.2 |
| `p2048-L128-i32-N1024-d7` | 157 907.9 | 72 023.9 | 31 335.0 | **261 266.8** | 368 966.8 |
So a batch cell's cold cost is **8.7 s** at pop 256 / `N` 4 096 and **32–36 s** at pop 1 000 to pay for three plans — against dense's 17.4–19.7 s for the same three workloads, the batch route is the cheaper cold start at pop 256 (0.47–0.59×) and the dearer one from pop 1 000 up (1.8–2.0×), and 81 s at pop 2 048 / `N` 1 024 (4.1× dense), 243–258 s at the two pop-4 096 cells (13.1–13.4×) and 261 s on `d7` (6.6× the raised-cap dense's 39.7 s). The plain-formation control costs **10–41 % more cold on the four pop ≥ 2 048 cells** (`+9.7 %` at `p2048`, `+16.3 %` at `p4096`, `+11.5 %` at `p4096-i64`, `+107.7 s` = `1.41×` on `d7`) while the pop-256/1 000 cells sit within ±4 % (inside the box's cold noise).
Dense's cold is flat at 6.3–7.5 s per workload because it is one graph per (shape, function set); the batch cold is one compile per population per workload and grows with the population (4.2 s at pop 256, 44 s at pop 2 048, 154 s at pop 4 096, 799 s at pop 8 192 for the loss plan), which is the axis §8 is about.
### 2.4 Ratio blocks
Same convention as round 1: both ratio columns divide the other arm's median by `evoxc-batch-iree`'s, so **above 1 means batch is cheaper, below 1 the other arm is**; `dense / batch` uses the default-cap dense column and `raised dense / batch` the `evoxc-dense-iree-raised` one (identical on every cell but d7, where it is the only dense arm).
| cell | dense / batch | raised dense / batch | torch-batched / batch | torch-batched / dense |
|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 0.893× | 0.881× | 0.559× | 0.626× |
| `p4096-L64-i32-N1024` | 1.721× | 1.713× | 0.483× | 0.281× |
| `p8192-L64-i32-N1024` | 1.811× | — | 0.422× | 0.233× |
| `p256-L64-i32-N4096` | 1.099× | 1.100× | 0.935× | 0.851× |
| `p256-L64-i32-N16384` | 6.201× | 6.241× | 0.830× | 0.134× |
| `p1000-L64-i32-N4096` | 3.022× | 3.042× | 0.779× | 0.258× |
| `p1000-L64-i32-N16384` | 3.145× | 3.149× | 0.869× | 0.276× |
| `p2048-L128-i32-N1024-d7` | — | 5.209× | 1.289× | 0.247× |
| `p4096-L64-i64-N1024` | 1.612× | 1.612× | 0.452× | 0.281× |
Batch is 0.89–6.20× dense (it loses only the pop-2 048 / `N` 1 024 cell) and 0.42–1.29× `torch-batched` (it wins only `d7`); `torch-batched` is 0.13–0.85× dense and 0.23–0.63× dense at the three `N` = 1 024 population cells. §9 reads these together.
### 2.5 The re-measure: what the three-workload window moved
`main/` was re-run as a whole (§11.1) after this report's first draft, which had priced the column with `--workloads sr_fitness`. The loss column replicated: of the 54 `sr_fitness` values both windows measured (55 records less the `d7` `evoxc-auto` one, whose route changed under it), **45 moved by ≤ 2 %** and 9 did not — the whole `torch-per-tree` column on five cells and the plain-formation control on four:
| cell | arm | previous window | re-measure | change |
|---|---|---|---|---|
| `p2048-L64-i32-N1024` | `torch-per-tree` | 257.99 | 281.36 | +9.1 % |
| `p2048-L64-i32-N1024` | `evoxc-batch-iree-default` | 12.81 | 14.00 | +9.3 % |
| `p4096-L64-i32-N1024` | `torch-per-tree` | 492.74 | 524.82 | +6.5 % |
| `p4096-L64-i32-N1024` | `evoxc-batch-iree-default` | 22.70 | 23.78 | +4.8 % |
| `p1000-L64-i32-N4096` | `torch-per-tree` | 124.76 | 136.41 | +9.3 % |
| `p2048-L128-i32-N1024-d7` | `torch-per-tree` | 579.27 | 610.55 | +5.4 % |
| `p2048-L128-i32-N1024-d7` | `evoxc-batch-iree-default` | 24.19 | 24.87 | +2.8 % |
| `p4096-L64-i64-N1024` | `torch-per-tree` | 489.18 | 511.97 | +4.7 % |
| `p4096-L64-i64-N1024` | `evoxc-batch-iree-default` | 24.54 | 24.05 | −2.0 % |
The `torch-per-tree` drift is instrument-side and uniform in sign (it reads 1.2–9.3 % higher across the column; the torch baselines do not depend on the tree under measurement), so it does not change any ordering. The control arm's movement is not uniform: `evoxc-batch-iree-default` is the arm with the widest cold spread of the round (50.7 s against 43.8 s at pop 2 048), and its 12.81 → 14.00 ms is what makes this window's formation delta at that cell **−11.4 %** where the previous window read **−4.9 %** (§3.1, §3.4). The lesson is the one `test/benchmark/evoxc/CONTEXT.md` states: per-cell warm deltas are window-sensitive at ±10 %, so only the SIGN and the ordering of the formation knob's per-cell deltas are robust here — the magnitude of any single cell is not.
The `d7` `evoxc-auto` record is the one non-noise change: the previous window's `main/` record predated the trace-cap raise and read 857.22 ms on the `specialized`+numpy fallback (its `auto_trace_cap` block was `available: null`); the re-measure is at `6cdd68b`, where the block reads `base_cap 64` / `auto 128` / `cap_pinned false`, and the call is the raised dense graph at 105.79 ms — 1.00× the explicit `evoxc-dense-iree-raised` arm beside it. The 857.22 record no longer exists in `main/`; `auto_fix/`'s 103.34 ms (§4.2) is the surviving record of what the raise does on that cell, and `LARGE_SCALE.md` §2.1's 889.92 ms `evoxc-specialized-numpy` is the surviving record of the fallback itself.
## 3. The formation knob
`EVOGP_BATCH_DISPATCH_FORMATION` (`auto` | `aggressive` | `default` | `off`, default `auto`) resolves to `aggressive` only when the backend is iree, the compile target is a GPU HAL (`rocm`/`cuda`) and the installed evoxc adapter declares the option; it becomes part of every memo/executable/artifact key, and a record shows it as `band_report.dispatch_formation` next to `dispatch_formation_requested`.
### 3.1 Per-cell before/after (same window, same population)
| cell | warm `sr_fitness` BEFORE → AFTER | cold `first_call_ms` (loss plan) BEFORE → AFTER |
|---|---|---|
| `p2048-L64-i32-N1024` | 14.0002 → 12.4088 (**−11.4 %**) | 50 734.8 → 43 786.4 (−13.7 %) |
| `p4096-L64-i32-N1024` | 23.7820 → 21.5862 (−9.2 %) | 193 068.5 → 154 371.1 (−20.0 %) |
| `p256-L64-i32-N4096` | 5.1837 → 5.2869 (+2.0 %) | 4 176.5 → 4 171.5 (−0.1 %) |
| `p256-L64-i32-N16384` | 12.4458 → 12.5782 (+1.1 %) | 5 308.6 → 5 216.9 (−1.7 %) |
| `p1000-L64-i32-N4096` | 13.1660 → 13.0073 (−1.2 %) | 17 466.4 → 16 506.4 (−5.5 %) |
| `p1000-L64-i32-N16384` | 37.2493 → 37.6059 (+1.0 %) | 19 691.5 → 18 508.3 (−6.0 %) |
| `p2048-L128-i32-N1024-d7` | 24.8665 → 20.3248 (**−18.3 %**) | 259 428.5 → 157 907.9 (**−39.1 %**) |
| `p4096-L64-i64-N1024` | 24.0524 → 23.0584 (−4.1 %) | 199 064.8 → 169 785.5 (−14.7 %) |
On the whole three-workload cold column (§2.3) the control pays +9.7 / +16.3 / +11.5 % more at `p2048` / `p4096` / `p4096-i64` and +41.2 % (+107.7 s) on `d7`, while the pop-256/1 000 cells stay inside ±4 %.
The plan records say the two spellings produce the SAME merge — `emitted_ops` 41 376 / 21 339 / 5 497 / 78 189 / 93 751 / 80 968 identical across the pair on every cell, identical `chunk_sizes`, `dispatch_formation` `aggressive` (requested `auto`) against `default` (requested `default`) — so the whole difference is made inside the iree compile, and the harness records no dispatch count: the "fewer dispatches" half of the mechanism is unverified here and the win is measured, not explained.
### 3.2 The A/B that isolates it (`results/largascale2/ab_dispatch/`, tree `7258b3d`)
Three cells × the two spellings, `sr_fitness` + `batch_forward` + `forward`, warmup 2 / reps 5:
| cell | `sr_fitness` default → aggressive | `forward` | `batch_forward` | compile wall |
|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 12.8288 → 12.2646 (−4.4 %) | 3.3546 → 3.1324 (−6.6 %) | 12.6670 → 12.7484 (+0.6 %) | 71 343.8 → 67 769.3 |
| `p1000-L64-i32-N4096` | 13.0599 → 12.9212 (−1.1 %) | 2.5339 → 2.3776 (−6.2 %) | 14.4381 → 13.4312 (−7.0 %) | 30 327.2 → 28 836.2 |
| `p256-L64-i32-N16384` | 12.3654 → 12.5581 (+1.6 %) | 1.8677 → 1.8901 (+1.2 %) | 12.1136 → 12.6595 (+4.5 %) | 8 323.3 → 8 337.7 |
The two campaigns agree in SIGN on the loss column (−11.4/−1.2/+1.1 here against −4.4/−1.1/+1.6 in the A/B) but not in magnitude — the pop-2 048 cell moves 2.6× between the two windows — which is why §2.5/§3.4 read the knob by sign and ordering only.
### 3.3 Numerics
The formation changes the value path, by design: at `p256-L64-i32-N16384` exactly 1 of 256 output elements differs, at 4.24e-05 relative on `sr_fitness` and 8.81e-08 on `forward` (`abs_sum` 126 371 117 359 727.72 against …819.97 — that pair is the `p2048-N1024` cell), and a same-formation control is bitwise exact (`max_abs` 0.0 across default, default, aggressive, default) — so the difference is attributable to the knob and deterministic.
The re-measured `main/` records reproduce it in their digests, which is the sharpest way to read the split. `check.py` sha-gates the two value workloads (`forward`, `batch_forward` — `VALUE_WORKLOADS`) and gates `sr_fitness` on NaN/±inf placement only, and on that gate the round is clean: NaN and finite counts are identical across all seven arms on all eight cells and all three workloads.
- `forward` splits into the torch group (`torch-per-tree` + `torch-batched`), the dense group (`evoxc-auto` + `evoxc-dense-iree` + `evoxc-dense-iree-raised`) and the two batch spellings — 4 sha groups on four cells (`p4096`, both pop-1 000 cells, `p4096-i64`), 3 on the other four: at `p2048-N1024` and `d7` the two batch spellings coincide, and at `p256-N4096`/`p256-N16384` one row per tree is small enough that dense joins the torch walk bitwise.
- `batch_forward` splits into the torch group, the dense group and the batch pair — 2 groups at `p256-N4096`/`p256-N16384` (the two spellings agree there), 3 on the two pop-1 000 cells and `p4096-i64`, and 4 at `p2048-N1024`, `p4096-N1024` and `d7`, where the spellings split again.
- `sr_fitness` splits into five groups in every cell — `torch-per-tree` / `torch-batched` / dense+`auto`+raised / batch / batch-default — and the two batch spellings differ only in the loss's low bits: their `abs_sum` differs by 7.3e-13 relative at `p2048-N1024` (`126 371 117 359 819.97` against `…727.72`) with identical NaN/±inf placement. That is exactly §3.2's one-element, 4.24e-05 finding, so the fusion changes low bits on every workload, the loss included, and never the arithmetic identity.
That is why the batch arm forms its own digest group and `check.py` needs `--allow-digest-mismatch` (already the route's documented status: its `batch_forward` is non-bitwise on the arithmetic set with the knob off too, `BATCH.md` §11).
### 3.4 Verdict: partially confirmed, keep it as the default
The adapter owner's prediction was 7–9 % on the merged-plan workloads. In this window the headline workload (`sr_fitness`) comes in at **−11.4 %** and **−9.2 %** on the two `N` = 1 024 population cells of `main/`, **−4.1 %** on `p4096-i64`, **−18.3 %** on `p2048-L128-i32-N1024-d7` (the largest merge of the column) and **inside noise on the four row-axis cells** (+2.0 / +1.1 / −1.2 / +1.0 %). That is a wider spread than the previous window's −4.9 / −4.8 / −5.9 % on the same three cells, and the arm that moved is the control (§2.5): **the per-cell formation delta is window-sensitive and only the sign and the ordering are robust** — never worse than noise, one large win, and the sign of the small plans undetermined. `forward` is where the prediction holds best: −6.2/−6.6 % at pop 1 000 / 2 048 in the A/B (round 1 measured the same −6.6/−6.2 %), and the re-measured column agrees in direction on six cells (−4.7 to −16.7 %, the two pop-1 000 cells both −6.8 %) with the two `p256` cells the exceptions (+0.1 / +9.6 %, §5.3). Cold compile is never worse beyond the noise band (−13.7 % to −39.1 % on the loss plan, and the control's three-plan totals cost 0.3–41.2 % more, the top of that range being `d7`'s +107.7 s), and `emitted_ops` never change.
So: **partially confirmed** — the prediction's magnitude is right on `forward` and on the biggest merge, low on the small plans, and the loss-cell magnitudes are only reproducible in sign. It stays the shipped default because it costs nothing on the compile side, never loses beyond noise, and the one place it is large is the place the merge is largest.
## 4. The corrected `auto` policy
Two separate fixes, and the records separate them: the rows ceiling became a memory envelope (landed by tree `e41f704`, visible in `main/`'s `dense_policy` block), and an over-cap population is now traced at its own longest tree rounded up to a power of two (landed by `82ff822`, visible as the `auto_trace_cap` block plus the d7 numbers in `auto_fix/`).
### 4.1 The rows envelope
`auto` used to decline dense on every `N ≥ 4 096` cell because of a fixed `EVOGP_DENSE_AUTO_MAX_ROWS` = 2 048, and fell back to per-tree `specialized`+numpy — a route dense beats by 1.4–8.7× per call (`LARGE_SCALE.md` §7.1). The ceiling is gone: `dense_policy` now reads `{auto_min_pop 64, auto_max_rows 0, auto_budget_mb 8192, max_tree_len_cap 64}`, `auto_max_rows` is a user PIN that can only tighten, and the gate is the memory rule `pop * tree_len * rows * 4 ≤ 8 192 MiB` — the same product the harness uses as `CELL_TENSOR_CEILING_BYTES` (§1), which no cell of this column reaches.
| cell | `auto` round 1 | `auto` now | change | `evoxc-dense-iree` beside it |
|---|---|---|---|---|
| `p256-L64-i32-N4096` | 50.47 | 5.82 | **8.67×** | 5.81 |
| `p256-L64-i32-N16384` | 110.21 | 78.03 | **1.41×** | 77.99 |
| `p1000-L64-i32-N4096` | 197.31 | 39.38 | **5.01×** | 39.31 |
| `p1000-L64-i32-N16384` | 446.60 | 117.74 | **3.79×** | 118.29 |
| `p2048-L64-i32-N1024` | 10.99 | 11.01 | +0.2 % | 11.08 |
| `p4096-L64-i32-N1024` | 37.18 | 36.85 | −0.9 % | 37.15 |
| `p8192-L64-i32-N1024` | 76.34 | 76.03 | −0.4 % | 75.93 |
| `p4096-L64-i64-N1024` | 37.23 | 37.25 | +0.1 % | 37.16 |
`auto` is now indistinguishable from explicit dense on every `N ≥ 4 096` cell (0.992–1.002×, i.e. inside the box's noise band, with `auto` resolving the registered `dense` route in both cases) and unchanged at `N` = 1 024, where it already was dense. `auto_fix/` re-measures the same three row-axis cells at `82ff822` and reads 5.8613 / 118.4595 / 10.9621, i.e. the same answer.
### 4.2 The trace-cap raise
The twin defect: `auto` resolved dense for a population whose realized trees exceed the trace cap, `prepare_dense` declined (`L = min(EVOGP_DENSE_MAX_TREE_LEN 64, row_len 128) = 64 < longest 123`) and the call continued on `specialized`+numpy. Under `auto` only, and only when the cap was not user-pinned, the entry points now trace at `_next_power_of_two(max(base_cap, longest))`, kept only while `min(ceiling EVOGP_DENSE_AUTO_MAX_TREE_LEN = 256, envelope allowance)` reaches it.
| d7 record | `auto_trace_cap` | `sr_fitness` |
|---|---|---|
| `main/` previous window (`e41f704`, pre-raise) | `available: null` (no accessor yet) | 857.22 — record overwritten by the re-measure, §2.5 |
| `main/` re-measure (`6cdd68b`) | `base_cap 64`, `auto 128`, `cap_pinned false`, ceiling 256 | **105.79** (8.42× vs round 1's 891.23, 1.00× vs raised dense 105.88) |
| `auto_fix/` (`82ff822`) | `base_cap 64`, `auto 123`, `cap_pinned false`, ceiling 256 | **103.34** (8.62× vs round 1's 891.23, 1.02× vs raised dense 105.58) |
| `loop_auto_fix/` (`61af76e`) | `base_cap 64`, `auto 128`, `cap_pinned false` | 105.81 (1.02× vs raised dense) |
The rounding is the difference between the two later records and it is a loop lever, not a per-call one: `L` is part of the dense graph key, so a cap that tracked `longest` exactly compiles a fresh graph whenever the generation's longest tree changes, while a 65–128-node band shares one graph.
### 4.3 The loop consequence and its fix
| d7 `gp_step` (`evoxc-auto`) | step 1 | step 2 | step 3 | compiles | median |
|---|---|---|---|---|---|
| drifting cap (`loop/`, cap `123`) | 11 880.67 | **10 014.83** | 6 375.02 | 1 per step | 10 014.83 |
| rounded band (`loop_auto_fix/`, cap `128`) | 127.05 | 122.68 | **6 437.70** | 0, 0, 1 | **127.05** |
The drifting cap recompiled on every generation (three ~3–6 s compiles); the rounded band compiles once, when the population's longest tree crosses 128, and the other two generations run at 122.7–127.0 ms. Same cell, same arm, same `--reps`: 79× on the median.
### 4.4 The residual
The band is not free of recompiles, it only makes them rare: a generation whose longest tree crosses a power of two (`128 → 129`) still pays one compile — that is `loop_auto_fix`'s third step, 6 437.70 ms against 122.7–127.0. And a `specialized`+`iree` decline is still the answer when the ceiling or the envelope refuses the raise, with the artifact growth that implies — the raise is bounded (256 nodes) rather than unrestricted, which is why the `d8` cell (201 nodes) is unaffected by design.
## 5. The generation step
### 5.1 `gp_step`, `median_total_ms` over 3 warm generations after a 1-generation warmup
| cell | torch-per-tree | torch-batched | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-auto | evoxc-batch-iree |
|---|---|---|---|---|---|---|
| `p256-L64-i32-N4096` | 18.23 | 22.51 | **10.68** | 11.02 | 10.45 | 1 901.14 |
| `p1000-L64-i32-N16384` | **49.42** | 77.46 | 127.25 | 129.40 | 127.68 | 8 696.97 |
| `p2048-L128-i32-N1024-d7` | 125.25 | 311.37 | — | **124.46** | 127.05¹ | 11 957.57 |
¹ from `loop_auto_fix/` (rounded cap); `loop/`'s pre-rounding `evoxc-auto` on this cell is 10 014.83 ms (§4.3).
The loop ratios the round exists to report: batch is **178×** dense at `p256-L64-i32-N4096` (1 901.14 / 10.68), **68×** at `p1000-L64-i32-N16384` and **96×** at d7 against raised dense (11 957.57 / 124.46). Every one of those is compile: each generation presents a new structure multiset, so each step records `batch_plan_misses 1` and 3–4 `batch_compile_misses`, and the compile wall is 89.6 % / 94.0 % / 94.4 % of the three steps at `p256-L64-i32-N4096` (2 563.4 / 1 787.9 / 1 739.2 ms of 2 860.7 / 1 901.1 / 1 842.7 ms) — the eval is 1 898.05 ms of which 1 787.9 ms is that compile wall, and the operator layer is 3.09 ms.
Dense's generation cost is shape-sized, not node-sized: at `p256-L64-i32-N4096` a generation is 10.68 ms total (8.18 eval + 2.51 ops) because one graph covers the whole shape, and at `p1000-L64-i32-N16384` it is 127.25 ms (124.70 + 2.56) — and there `torch-per-tree` wins outright (49.42 ms), because the pipeline's post-generation trees are cheaper to walk than dense's fixed graph is to run over 16 384 rows. That is the same trade round 1 found; the corrected `auto` follows dense and therefore inherits it (`auto` 127.68).
`best_fitness` agrees across every arm of a cell (−0.08445492386817932 / −0.0838315486907959 / −0.0843039…), so the loop comparison is a timing comparison and not a different search.
### 5.2 `forward` (one row per tree), warm
| cell | torch-per-tree | torch-batched | dense | dense-raised | auto | batch |
|---|---|---|---|---|---|---|
| `p256-L64-i32-N4096` | 23.87 | 4.35 | **0.87** | 0.99 | 0.86 | 1.95 |
| `p1000-L64-i32-N16384` | 94.43 | 4.44 | **0.89** | 0.96 | 0.86 | 2.37 |
| `p2048-L128-i32-N1024-d7` | 541.70 | 15.68 | — | **1.28** | 1.29 | 6.66 |
The per-row path is the merged graph's worst case — one row per tree, so no band amortizes — and it is where the batch route is 2.25× (0.87 → 1.95), 2.67× (0.89 → 2.37) and 5.22× (1.28 → 6.66) dearer than dense; it is still 1.9–2.4× cheaper than `torch-batched` (4.35/1.95, 4.44/2.37, 15.68/6.66). The formation knob's own `forward` win (§3.2, −6.2/−6.6 %) is the reason this column is closer than round 1's.
### 5.3 The two value paths on the whole column (`main/`)
Since the re-measure, `forward` and `batch_forward` are priced on all eight headline cells in the same window as the loss call, so the batch route's position per workload is now measured rather than inferred from three loop-column cells.
`forward` (one row per tree), warm `median_ms`:
| cell | torch-per-tree | torch-batched | dense | dense-raised | auto | batch | batch-default |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 223.29 | 4.86 | 0.91 | 0.89 | **0.87** | 3.26 | 3.92 |
| `p4096-L64-i32-N1024` | 397.48 | 4.48 | **0.88** | 0.95 | 0.90 | 4.93 | 5.17 |
| `p256-L64-i32-N4096` | 24.68 | 4.36 | 0.87 | 0.90 | **0.87** | 1.99 | 1.82 |
| `p256-L64-i32-N16384` | 23.32 | 4.32 | 0.89 | **0.85** | 0.88 | 2.02 | 2.01 |
| `p1000-L64-i32-N4096` | 103.87 | 4.83 | **0.88** | 0.98 | 0.88 | 2.36 | 2.54 |
| `p1000-L64-i32-N16384` | 95.78 | 4.39 | 0.91 | 0.87 | 0.90 | 2.46 | 2.65 |
| `p2048-L128-i32-N1024-d7` | 539.78 | 15.74 | — | **1.27** | 1.38 | 6.84 | 7.20 |
| `p4096-L64-i64-N1024` | 401.88 | 4.63 | 0.93 | 0.93 | 0.91 | 4.72 | 4.99 |
Dense is flat at **0.875–0.933 ms** on every cell (0.870–0.907 on the `auto` side away from `d7`, where `auto` runs the raised 128-node cap at 1.38), because one row per tree is one row per tree regardless of the population; the batch route is **1.99–6.84 ms**, i.e. **2.27–5.60× dearer than dense**, and **0.46–1.10× of `torch-batched`**: faster than the vectorized walk on six cells (1.49–2.30×) and slower on the two pop-4 096 / `N` 1 024 cells (1.10× at `p4096-i32`, 1.02× at `p4096-i64`), where the Python walk is at its cheapest per tree.
`batch_forward` (the whole population on all rows), warm `median_ms`:
| cell | torch-per-tree | torch-batched | dense | dense-raised | auto | batch | batch-default |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 217.91 | **6.91** | 14.81 | 14.89 | 14.87 | 13.19 | 12.82 |
| `p4096-L64-i32-N1024` | 382.66 | **10.36** | 46.01 | 47.03 | 46.27 | 22.83 | 22.36 |
| `p256-L64-i32-N4096` | 24.20 | **4.92** | 7.89 | 7.80 | 7.73 | 5.27 | 5.38 |
| `p256-L64-i32-N16384` | 23.75 | **10.42** | 81.06 | 81.03 | 80.79 | 13.37 | 13.35 |
| `p1000-L64-i32-N4096` | 102.93 | **10.10** | 48.29 | 48.72 | 48.07 | 14.46 | 13.86 |
| `p1000-L64-i32-N16384` | 95.72 | **32.61** | 150.56 | 151.59 | 151.78 | 52.63 | 54.38 |
| `p2048-L128-i32-N1024-d7` | 540.10 | 26.19 | — | 110.40 | 110.34 | **19.35** | 18.91 |
| `p4096-L64-i64-N1024` | 389.05 | **10.41** | 46.50 | 46.96 | 46.74 | 22.80 | 22.16 |
Here the batch route is the cheap compiled arm on **all eight cells** — 5.27–52.63 ms against dense's 7.89–150.56 ms, i.e. **0.16–0.89× of dense** — but the merged graph does not yet beat the vectorized walk on this workload either: `torch-batched` is **1.07–2.20× faster** than the batch route on seven of the eight cells, and the batch route wins only `d7` (19.35 against 26.19 ms, **1.35×**).
**The mechanism is the plan, not the backend.** The three workloads are three plans over the same population, and the per-row one is the widest of the three while doing the least work per call: 1 862 deduped functions against the loss plan's 1 680 at `p2048-N1024` (3 670 against 3 265 at `p4096`, 245 against 232 at `p256`), a per-tree row index in every callable, and one row of work behind a dispatch count that does not shrink. A `forward` call therefore pays the merged graph's fixed per-call cost with none of the row axis to amortize it, which is exactly what the `torch-batched` column exploits (one vectorized pass over pop × 1 elements). The population-wide loss plan is the opposite extreme — `pop × rows` elements behind the same dispatch count — and there the batch route is the fastest compiled arm on every cell. The whole-population value plan (`batch_forward`, `per_row=false` over all rows) sits between them and lands where the rows dominate the fix-ups.
## 6. The artifact tier
`EVOGP_EVOXC_ARTIFACT_DIR` (the one variable the dense and batch routes share, off when unset) makes the merged graph's executables persist on disk, so a later process can skip `lower`/`compile`/`load`. Measured as a two-process pair per cell on the same directory (`/tmp/art-A`): a cold run that populates, then a fresh process that should hit.
### 6.1 Cold → hit
| cell | cold `first_call_ms` | hit `first_call_ms` | gain | plan/exec misses on the hit | lower/compile/load calls on the hit | files | digest |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 44 266.59 | **4 101.13** | **10.79×** | 1 plan / 0 exec | 0 / 0 / 0 | 0 → 8, then 16 | identical (`9429ad51…`) |
| `p1000-L64-i32-N4096` | 16 980.46 | **2 211.44** | **7.68×** | 1 plan / 0 exec | 0 / 0 / 0 | 8 → 16 | identical (`35b99d99…`) |
The hit run's compile counters are zero (`compile_misses_first_call 0`, `batch_compile_wall_ms 0.0`, `batch_compile_jobs 0`, `batch_artifact_hits 8`, `misses 0`, `stores 0`, `hit_rate 1.0`), and its warm medians are unchanged (12.3295 → 12.6023; 12.9317 → 12.9830) — the tier moves the COLD column only, exactly as designed.
The cross-process pool-1 pair makes the same point with a much bigger prize: `artifact/pool1-cross-process/` cold 140 795.44 → hit 2 244.36 ms (**62.7×**) at `p1000-L64-i32-N4096` on `EVOGP_BATCH_COMPILE_WORKERS=1`, whose cold call is a 192 s single-body compile (`batch_compile_wall_ms` 192 019.4 over 2 jobs).
A one-cell smoke (`artifact/smoke`, `p64-L64-i32-N64`, not part of the column) reads 2 234.03 → warm 1.0945 with 1 file.
### 6.2 What the hit still pays
A hit removes the device compile, not the host plan: the plan key misses (`plan_misses_first_call 1`) and the route re-decodes, re-traces and re-merges the population before it can look the executables up. On the single-workload pool hits at `p2048-L64-i32-N1024` that is 84.6 % (pool 4) and 81.8–86.6 % (pools 1/2/8) of the first call in the four instrumented stages:

| pool | trace | merge | decode | dedup | sum / first call |
|---|---|---|---|---|---|
| 1 | 1 076.3 | 2 668.9 | 28.1 | 1.1 | 3 774.4 / 4 389.2 = **86.0 %** |
| 2 | 1 146.4 | 2 332.0 | 31.3 | 1.3 | 3 511.1 / 4 053.7 = 86.6 % |
| 4 (shipped) | 1 057.3 | 2 204.2 | 29.1 | 1.1 | 3 291.8 / 3 889.5 = **84.6 %** |
| 8 | 1 132.7 | 2 350.9 | 29.4 | 1.4 | 3 514.5 / 4 298.1 = 81.8 % |

The remainder is `upload`/`run`/`gather` PLUS the artifact load itself, which is **not stage-instrumented**: on a hit `batch_load_calls` is 0 and there is no `batch_load_total_ms` for the load-from-disk path (the counter only wraps the compile-then-load step), so the ~13–18 % residual is a bound, not an attribution. Trace+merge is where a hit's time goes, and that is host-side work no artifact tier can remove.
### 6.3 The key includes the chunk partition
`artifact/pool1-cross-process/cold` ran into `/tmp/art-A`, which the pool-4 pair had already filled with 16 files; it did not hit them — it stored 2 more (files 16 → 18, `hits 0`, `stores 2`) and the next pool-1 process hit those 2 (files 18 → 18, `hits 2`, `stores 0`, 2 244.36 ms). The pool sweep makes the same point four ways: `/tmp/art-p1`, `/tmp/art-p2`, `/tmp/art-p4`, `/tmp/art-p8` are separate directories and each populates 2×4 files for its own width. **A directory populated at pool 4 does not serve a pool-1 or pool-8 run**: the chunk partition is part of the key, so the pool width cannot be changed under a warm directory, and an operator who wants to try a width must pay one cold call (or repopulate) per width.
### 6.4 Caveats
The records carry file COUNTS, not byte sizes (there is no `batch_artifact_files_bytes`-style counter), so this report cannot state the directory's on-disk size per cell — only that a pop-2 048 / `N` 1 024 population at the shipped pool of 4 writes 8 files and at pop 1 000 / `N` 4 096, 16 (two workloads). The tier is also per-path, not per-cell: the pairs above reused `/tmp/art-A` across cells, which is why the file count grows within a pair (`0 → 8` then `16`).
## 7. The knobs
### 7.1 Compile-pool width (`EVOGP_BATCH_COMPILE_WORKERS`), `p2048-L64-i32-N1024`, with the artifact tier on so that cold and hit are measured in the same shape
| pool | chunks | emitted ops | cold first | warm med (min) | hit first | hit − cold | break-even (warm calls) |
|---|---|---|---|---|---|---|---|
| 1 | 1 × 1 680 | 38 640 | **759.58 s** | **10.8055** (10.7092) | 4.389 s | −755.2 s | 539 046 |
| 2 | 2 × 840 | 39 852 | 143.70 s | 11.0202 (10.9811) | 4.054 s | −139.6 s | 91 008 |
| **4 (shipped)** | 4 × 420 | 41 376 | 42.11 s | 12.1365 (12.0500) | 3.889 s | −38.2 s | — (reference) |
| 8 | 8 × 210 | 43 254 | **23.998 s** | 14.5835 (14.3397) | 4.298 s | −19.7 s | 7 402 |
All four plans carry `dispatch_formation: aggressive` and `chunk_count == pool` with equal-sized chunks; the emitted ops rise with the width (38 640 → 43 254, +12 %) because each chunk pays its own band overhead, which is the warm penalty.
The shipped pool of 4 is **12 % off the best warm pool** (pool 1, 10.81 ms) and **18 s behind pool 8 cold** (42.11 vs 24.00 s), so no width dominates: 1 wins warm and loses 18× cold, 8 wins cold and loses 35 % warm, and the crossover to 4 is 7 402 warm calls of one population for pool 8 and 539 046 for pool 1 (i.e. 4 is the right width for a population evaluated between ~7 400 and ~540 000 times — the fixed-population regime — and no pool width rescues a turnover loop, §5.1).
The second round of this comparison reproduces round 1's shapes: round 1 (`LARGE_SCALE.md` §6.3, economics driver) read 1 085.9 / 189.3 / 47.2 / 28.8 s cold and 12.08 / 11.83 / 12.85 / 15.09 ms warm; this round's campaign records read 759.6 / 143.7 / 42.1 / 24.0 s and 10.81 / 11.02 / 12.14 / 14.58 ms. The cold column is 18–30 % cheaper across the board (a different measurement entry point — `campaign.py` with the artifact arm vs `batch_economics.py` — plus this box's cold noise), the ordering and the trade are identical.
### 7.2 Band size and band count (`band/`, `band_sweep.py` records)
`p256-L64-i32-N16384`, the per-row plan, pool 1 (so a difference is the partition alone):
| config | warm med (min) | cold first (s) | emitted ops | bands | clamped |
|---|---|---|---|---|---|
| `b8` | 9.7289 (9.6534) | 13.25 | 5 475 | 15 | true |
| `b16` (shipped) | 9.6590 (9.6237) | 13.35 | 5 475 | 15 | false |
| `b32` | 9.5497 (9.5053) | 15.01 | 5 200 | 8 | false |
| `b64` | 9.5981 (9.5885) | 17.21 | 4 978 | 4 | false |
| `k15` (count pin at the rocm cap) | 9.6497 (9.6414) | 13.31 | 5 475 | 15 | false |
| `k7` (count pin) | **9.3944** (9.3454) | 14.80 | 5 156 | 7 | false |
| `flat` (K = 1) | **12.2009** (12.1152) | 18.86 | 4 635 | 1 | false |
`p2048-L64-i32-N1024`, the loss plan, chunked at the shipped pool of 4:
| config | warm med (min) | cold first (s) | emitted ops | chunk bands | clamped |
|---|---|---|---|---|---|
| `b8-cw4` | 12.4163 (12.3359) | 42.48 | 41 376 | [15, 15, 15, 15] | true |
| `b16-cw4` (shipped) | 12.3537 (12.2984) | 41.20 | 41 376 | [15, 15, 15, 15] | true |
| `b32-cw4` | 12.3115 (12.2576) | 42.39 | 41 197 | [14, 14, 14, 14] | false |
| `b64-cw4` | 12.4437 (12.3798) | 43.56 | 39 740 | [7, 7, 7, 7] | false |
| `k15-cw4` | 12.9694 (12.9472) | 45.66 | 38 753 | [4, 4, 4, 4] | false |
| `flat-cw4` | **39.0667** (38.8860) | **865.56** | 31 735 | [1] | false |
Both sweeps ran to 0 failures (`sweep.json`), 7 and 6 configs.
The two conclusions of round 1 survive at this scale: the band SIZE does not matter beyond noise (1.9 % spread among `b8`/`b16`/`b32`/`b64` at `p256`, 1.1 % among the banded chunked configs at `p2048`, both inside this box's ±5 % cross-config band), and flat-vs-banded is the first-order decision — the single unbanded merge is **3.16× worse warm and 21× worse cold** at `p2048` (39.07 vs 12.35 ms, 865.6 vs 41.2 s) and 1.26× worse warm at `p256`. The rocm binding cap still binds: `b8`/`b16` are clamped to 15 bands (identical 5 475 / 41 376 emitted ops), the count pins land at the cap (`k15`) and *below* it (`k7`, which is the warm argmin at `p256` — 2.7 % better than `b16`, i.e. inside noise), and the chunked `k15` pin is a *global* count split per chunk (`[4,4,4,4]`), which is why it costs 5 % warm. No config change is warranted and none is made.
## 8. Pop 8 192: the batch route is feasible now, and still not the answer
Round 1 declared `p8192-L64-i32-N1024` compile-infeasible (four `iree-compile` workers had finished no chunk inside a 900 s cap). This round it finishes, in a 900 s cap, with 89 % of the wall in the compile:
| item | value |
|---|---|
| cold `first_call_ms` | **799 412.45** (campaign wall 811.8 s) |
| `batch_compile_wall_ms` | **782 463.4** over `batch_compile_jobs` 4 (= 782.5 s, **97.9 %** of the cold call, 89 % of the cap) |
| plan | `n_functions` 6 541, `emitted_ops` **152 143**, 4 chunks `[1636, 1635, 1635, 1635]`, `dispatch_formation aggressive`, `band_clamped true` |
| warm `sr_fitness` | 41.93 (41.78) ms — 1.81× cheaper than dense (75.93) |
| compile was | decode 116.1, dedup 5.4, trace 4 276.0, merge 12 309.5, lower 15 094.6, compile 3 002 103.7 (sum over 4 concurrent jobs), load 839.4 ms; upload 57.8, run 319.9, gather 11.6 ms |
Feasible ≠ useful: 782.5 s of compile against dense's 6.8 s cold call needs ≈23 000 warm calls of ONE population to repay (799 s ÷ (75.93 − 41.93 ms)), while `torch-batched` is 17.71 ms — 2.4× faster than the batch route and 4.3× faster than dense on the same cell. The quadratic compile law (`LARGE_SCALE.md` §5: 48.9 s × (ops / 41 376)²) predicts 152 k ops at ≈660 s and the record is 782.5 s, i.e. the law holds within 19 % out of sample.
## 9. Verdict
**Where the batch route wins outright (warm, one fixed population, `sr_fitness`):**
- Against the default-cap dense it wins **7 of the 8 cells where that arm has a record** — only `p2048-L64-i32-N1024` loses (**1.12×**, 12.41 against 11.08 ms) — and against the raised cap **7 of the 8 where that one does** (the same single loss, 1.13×). Adding `p8192/`'s single-workload record makes it 8 of 9 in both cases.
- Most strongly on the **row axis**: `p256-L64-i32-N16384` is 6.20× cheaper than dense (12.58 against 77.99 ms), `p1000-L64-i32-N16384` 3.15×, `p1000-L64-i32-N4096` 3.02×, `p256-L64-i32-N4096` 1.10× — dense's cost tracks the row count while the merge's does not.
- **Past the dense trace cap**: `p2048-L128-i32-N1024-d7` is **5.21×** cheaper than a raised-cap dense (20.32 against 105.88 ms), and **≈44×** cheaper than the per-tree `specialized` fallback the default cap degrades to (round 1's committed `evoxc-specialized-numpy` record, 889.92 ms, `LARGE_SCALE.md` §2.1; the fallback's own pre-raise `main/` record read 857.22 ms before the re-measure overwrote it, §2.5) — the fallback is where that route ends, and the `dense` arm there does not fit in memory at all (§1.1).
- **On the population axis**: 1.72× at pop 4 096 (21.59 against 37.15/36.97) and 1.81× at pop 8 192 — the axis where round 1 measured 1.5–1.6×.
- **On the whole-population value path** (`batch_forward`, §5.3) it is the cheaper compiled arm on all eight cells (0.16–0.89× of dense) — the only workload where that is true.
**Where dense still wins:** the small-`N`, large-population corner — `p2048-L64-i32-N1024`, where the population is big enough to fill 15 bands but the merge is not yet amortized over enough rows (**1.12×**; round 1 measured 1.16× on the same cell — dense/batch 0.86 — so only the magnitude moved, never the sign, §2.5). On the compile axis the order flips with scale: at pop 256 the batch route is the CHEAPER cold start (**8.7 / 10.2 s** for all three plans against dense's **17.4–18.3 s**), while from pop 1 000 up dense wins the cold column (32–36 s at pop 1 000, **80.9 s** at pop 2 048, **243.3 s** at pop 4 096, **257.9 s** on `p4096-i64`, **261.3 s** on `d7`, against dense's 17–20 s and 39.7 s for the raised-cap `d7` arm) — §2.3 has the per-plan split.
**Where `torch-batched` still beats both:** on **8 of 9 cells**, and by a lot — 0.42× at pop 8 192, 0.45–0.56× at pop 2 048–4 096, 0.78–0.94× on the row axis, and 1.17–7.47× against dense across the column. The one exception is `d7`, where the batch route is **1.29×** faster than `torch-batched` (20.32 against 26.20 ms) — the deep-tree cell is the only place a compiled merged graph beats the vectorized torch walk on the loss call. On the two value paths `torch-batched` is unbeaten: 1.07–2.20× faster than the batch route on `batch_forward` (losing only `d7`) and faster on six of eight cells on `forward` (§5.3).
**In a real loop the verdict does not move:** batch loses by **68–178×** (`p256-L64-i32-N4096` 1 901.14 vs auto 10.45; `p1000-L64-i32-N16384` 8 696.97 vs 127.68; d7 11 957.57 vs 127.05), because every generation is a new structure multiset and each generation re-plans and recompiles 3–4 chunks. The corrected `auto` (i.e. dense) wins at `p256-L64-i32-N4096` (10.45 against everything) and loses to `torch-per-tree` at `p1000-L64-i32-N16384` (127.68 against 49.42, 2.58×).
**What still disqualifies the batch route** is exactly the compile column — 42 s at pop 2 048, 153 s at 4 096, 800 s at 8 192 for the loss plan, and 1.6–1.9× that for the three plans of a cell — plus the per-generation replan: it is a fixed-population evaluator, and the one regime where that is real (a trained population re-evaluated against many row sets, or a fixed population over few rows) is now also the regime where the artifact tier makes the cold call 7.7–10.8× cheaper. Nothing in this round contradicts `BATCH.md`'s "dominated in a GP loop"; what changed is that the large-scale column's warm order is now decisively batch-first above ~4 000 rows, and that the p8 192 cell — mandatory-pass/fail for the route's scale envelope — passes.
## 10. Remaining gaps and who owns each
| gap | evidence | owner |
|---|---|---|
| (a) the merged graph is dispatch-bound; only a fused device-side evaluator removes it | §2.4/§5.3: batch is 1.10–6.20× cheaper than dense warm on the loss call but 2.27–5.60× dearer on the per-row `forward` path, where no band amortizes; the formation knob moved `sr_fitness` by −18.3 … +2.0 % and only the sign is reproducible (§3.4); `torch-batched` still wins `forward` and `batch_forward` | **evoxc** (dispatch fusion / a fused merged-plan lowering) |
| (b) the merged-graph compile is ~quadratic in emitted ops — 42 s / 153 s / 800 s at pop 2 048 / 4 096 / 8 192 — and the plan is rebuilt every generation | §5.1, §8 (`batch_compile_wall_ms` is 97.9 % of the p8 192 cold call; 3–4 chunk compiles per generation in a loop) | **evoxc compile** (incremental or cached compilation of a changed plan), with EvoGP able to buy back only the fixed-population case (§6) |
| (c) an artifact hit still rebuilds the plan: 82–87 % of the hit call is decode/trace/merge | §6.2 (a hit removes `lower`/`compile`/`load` — 0 calls — but the plan key misses and the population is re-traced) | **EvoGP policy** (persist the plan/host structure next to the executables; the content-addressed plan key already exists) |
| (d) the compile-pool width is a regime choice, not a constant | §7.1 (pool 1 is 12 % better warm, pool 8 is 18 s better cold, break-even 7 402 / 539 046 warm calls around the shipped 4) | **EvoGP policy** (pick the width from an expected call count; the knob exists, the policy does not) |
| (e) `torch-batched` wins the single-call axis outright, up to 7.47× over dense | §2.4 (`torch-batched / dense` = 0.13 at `p256-L64-i32-N16384`) | **nobody on the EvoGP side**: a compiled graph cannot beat one vectorized pass without a fused kernel — the r6 Triton prototype (`test/benchmark/r6/SUMMARY.md`, 4.8×/4.6× the deleted CUDA engine, test-only) is the direction, and it is not integrated |
| (f) the band size / band count stays inert at large pop | §7.2 (1.9 % among the banded configs, flat is the only separation) | **settled, no action** — `EVOGP_BATCH_BAND_SIZE` 16 / `auto` opt-in / flat never |
## 11. Evidence layout, gate and reproduce commands
### 11.1 What is where
All paths are under `test/benchmark/evoxc/`, and the round is written under TWO spellings: `results/largescale2/` holds everything except the A/B, `results/largascale2/` holds `ab_dispatch/`.
| path | contents |
|---|---|
| `results/largescale2/main/` | 55 campaign records (8 cells × 7 arms, `sr_fitness` + `forward` + `batch_forward`, tree `6cdd68b`) + `campaign.json` (the SECOND invocation's snapshot: 6 configs / 6 ok / 794.3 s, the d7 note in `driver_env.note`) + `campaign.invocations.md` (both invocations transcribed, 49 configs / 2 330.6 s then 6 / 794.3 s) + `campaign.scoped-sr_fitness.json` (the superseded single-workload window's LAST driver record: `--workloads sr_fitness`, tree `e41f704`, 7 configs / 452.3 s, i.e. the `p4096-L64-i64-N1024` cell's seven arms with the `d7` OOM note in `driver_env.note`; `campaign.invocations.md` describes that file as a 49-config record, which its own `configs` list contradicts — the 49-config count belongs to the two invocations the `.md` transcribes, so read the file, not the caption) |
| `results/largescale2/p8192/` | 5 records on `p8192-L64-i32-N1024` (`torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-dense-iree`, `evoxc-batch-iree`) + `campaign.json` (1 config, 811.8 s), tree `82ff822` |
| `results/largescale2/auto_fix/` | 4 cells × {`evoxc-auto`, `evoxc-dense-iree-raised`} at `82ff822`: the trace-cap raise (§4.2) and the rows-envelope confirmation |
| `results/largascale2/ab_dispatch/` | 6 records: 3 cells × {`evoxc-batch-iree`, `evoxc-batch-iree-default`} on `sr_fitness` + `batch_forward` + `forward`, tree `7258b3d`; gate 6 records / 188 checks / 0 failures with the three documented flags |
| `results/largescale2/auto_fix_dense/` | 3 cells × `evoxc-dense-iree` at `82ff822` — the explicit-dense column beside `auto_fix` |
| `results/largescale2/loop/` | 3 cells × 6 arms, `--workloads gp_step,forward`, `--gp-warmup 1 --gp-reps 3`, tree `82ff822` (its `campaign-g3-raised.json` is the raised-dense half) |
| `results/largescale2/loop_auto_fix/` | the rounded-cap loop re-measure: `evoxc-auto` on `d7` and `p256-N4096`, `gp_step` + `sr_fitness`, tree `61af76e` |
| `results/largescale2/artifact/{cold-miss,warm-hit}/` | the populate/reload pair at `p2048-L64-i32-N1024` and `p1000-L64-i32-N4096` on `/tmp/art-A`, tree `63e6dc3` |
| `results/largescale2/artifact/pool1-cross-process/{cold,hit}/` | the same pair on the single-body path (`...-p1`), `/tmp/art-A`, `p1000-L64-i32-N4096` |
| `results/largescale2/artifact/smoke/` | one `p64-L64-i32-N64` artifact record (the cold-path smoke) |
| `results/largescale2/pool/p{1,2,4,8}/{cold,hit}/` | the pool-width sweep with the artifact tier on, `p2048-L64-i32-N1024`, one directory per width (`/tmp/art-p{1,2,4,8}`), trees `720d621`/`63e6dc3` |
| `results/largescale2/band/p256-row/`, `band/p2048-chunked/` | the two band sweeps (§7.2) + `sweep.json` each (7 configs / 0 failures, 6 configs / 0 failures), tree `720d621` |
`main/` and `p8192/` are the `check.py`-gateable units. The artifact, pool and band directories carry other drivers' schemas (a `band_sweep.py` directory is gated by its own `sweep.json`), and `--tree-root`-relative `.log` files sit next to every record.
### 11.2 Reproduce
From the repository root, `EVOXC_DISABLE_LOCATIONS=1`, `PY=/tmp/rocvenv/bin/python`, one config at a time, and always with the evoxc checkout that declares `dispatch_formation` (`--evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1`; the plain `/root/evoxc` checkout works too and reproduces the BEFORE column):
```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
# the headline: eight cells x seven arms, all three value workloads (TWO invocations)
# invocation 1 -- the seven cells where every arm is safe (49 configs, 2 330.6 s)
$PY test/benchmark/evoxc/campaign.py --tree-root src \
  --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out test/benchmark/evoxc/results/largescale2/main --timeout 900 \
  --cells p2048-L64-i32-N1024,p4096-L64-i32-N1024,p256-L64-i32-N4096,p256-L64-i32-N16384,p1000-L64-i32-N4096,p1000-L64-i32-N16384,p4096-L64-i64-N1024 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-batch-iree,evoxc-batch-iree-default \
  --workloads sr_fitness,forward,batch_forward
# invocation 2 -- the d7 cell without the plain dense arm (6 configs, 794.3 s; this one owns campaign.json)
$PY test/benchmark/evoxc/campaign.py --tree-root src \
  --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out test/benchmark/evoxc/results/largescale2/main --timeout 900 \
  --cells p2048-L128-i32-N1024-d7 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree-raised,evoxc-batch-iree,evoxc-batch-iree-default \
  --workloads sr_fitness,forward,batch_forward
# (the d7 `evoxc-dense-iree` config OOM-kills the job unit, so it is excluded from --arms there, §1.1)
# pop 8192 (811.6 s of a 900 s cap; the formation control is deliberately absent)
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out test/benchmark/evoxc/results/largescale2/p8192 --timeout 900 --cells p8192-L64-i32-N1024 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-batch-iree --workloads sr_fitness
# the corrected auto: the raised-cap column and its confirmations
$PY test/benchmark/evoxc/campaign.py --tree-root src --out test/benchmark/evoxc/results/largescale2/auto_fix --timeout 900 \
  --cells p2048-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N16384,p2048-L128-i32-N1024-d7 \
  --arms evoxc-auto,evoxc-dense-iree-raised --workloads sr_fitness
# the formation A/B (goes to the second spelling of the round directory)
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out test/benchmark/evoxc/results/largascale2/ab_dispatch --timeout 900 \
  --cells p2048-L64-i32-N1024,p1000-L64-i32-N4096,p256-L64-i32-N16384 \
  --arms evoxc-batch-iree,evoxc-batch-iree-default --workloads sr_fitness,batch_forward,forward
# the generation step (gp_reps 3; the raised-dense half is a second invocation into the same directory)
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out test/benchmark/evoxc/results/largescale2/loop --timeout 900 --gp-reps 3 \
  --cells p256-L64-i32-N4096,p1000-L64-i32-N16384,p2048-L128-i32-N1024-d7 \
  --arms torch-per-tree,torch-batched,evoxc-dense-iree,evoxc-auto,evoxc-batch-iree --workloads gp_step,forward
# the artifact tier: ONE fixed directory, a populate run then a fresh process
$PY test/benchmark/evoxc/campaign.py --tree-root src --artifact-dir /tmp/art-A --timeout 900 \
  --out test/benchmark/evoxc/results/largescale2/artifact/cold-miss \
  --cells p2048-L64-i32-N1024,p1000-L64-i32-N4096 --arms evoxc-batch-iree-artifact \
  --workloads sr_fitness,batch_forward
$PY test/benchmark/evoxc/campaign.py --tree-root src --artifact-dir /tmp/art-A --timeout 900 \
  --out test/benchmark/evoxc/results/largescale2/artifact/warm-hit \
  --cells p2048-L64-i32-N1024,p1000-L64-i32-N4096 --arms evoxc-batch-iree-artifact \
  --workloads sr_fitness,batch_forward
# the pool sweep: one directory per width, cold then hit
$PY test/benchmark/evoxc/campaign.py --tree-root src --artifact-dir /tmp/art-p4 --timeout 900 \
  --out test/benchmark/evoxc/results/largescale2/pool/p4/cold --cells p2048-L64-i32-N1024 \
  --arms evoxc-batch-iree-artifact --workloads sr_fitness
# (repeat with --artifact-dir /tmp/art-p1|p2|p8 and the arm evoxc-batch-iree-artifact-p1|p2|p8)
# the two band sweeps
$PY test/benchmark/evoxc/band_sweep.py --cells p256-L64-i32-N16384 \
  --configs b8,b16,b32,b64,k15,k7,flat --workloads sr_fitness --warmup 2 --reps 5 \
  --tree-root src --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out-dir test/benchmark/evoxc/results/largescale2/band/p256-row
$PY test/benchmark/evoxc/band_sweep.py --cells p2048-L64-i32-N1024 \
  --configs b8-cw4,b16-cw4,b32-cw4,b64-cw4,k15-cw4,flat-cw4 --workloads sr_fitness \
  --warmup 2 --reps 5 --tree-root src --evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1 \
  --out-dir test/benchmark/evoxc/results/largescale2/band/p2048-chunked
```
### 11.3 Gate
```bash
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/largescale2/main \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# -> 55 records / 1321 checks / 0 failures (exit 0), plus 16 WARN
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/largascale2/ab_dispatch \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# -> 6 records / 188 checks / 0 failures (3 WARN: the digest splits the knob causes)
```
`main/` gates clean because the re-measure closed the 16 failures the single-workload window used to report (`FAIL <cell>/<workload>: no digest anywhere in the cell` for the two value workloads nobody had scheduled) — the records now carry `forward` and `batch_forward`, so the same class of check passes instead of being excused. The 16 WARNs are the declared sha256 splits of §3.3 (2 per cell on `forward` and `batch_forward`; `sr_fitness` is not sha-gated, `check.py`'s `VALUE_WORKLOADS` being the two value paths, and its NaN/finite placement checks pass everywhere).
Two honesty caveats about this gate unit: `campaign.json` holds only the second invocation's 6 configs, so its guard honesty is machine-checked for 6 configs, while the first invocation's 49 configs are attested by `campaign.invocations.md` (`(gpu_idle=True)` on every one) rather than by a `campaign.json`; and the missing-arm rule is weaker than it looks here: `check.py` derives its expected set from the campaign's own scheduled arms, and the surviving `campaign.json` is the `d7`-only second invocation (6 arms, no `evoxc-dense-iree`), so the arm §1.1 excludes on `d7` is not reported as missing at all — the seven other cells' plain-dense records satisfy the presence check without ever being an expectation.
Other directories, gated one at a time: `p8192/` 5 records / 62 checks / 2 failures (the old single-workload `FAIL … no digest` class, one cell — `p8192/` was not re-measured), `auto_fix/` 8 / 114 / 8, `auto_fix_dense/` 3 / 50 / 6, `loop/` 17 / 270 / 3 (+3 WARN digest splits), `loop_auto_fix/` 2 / 42 / 4. No unallowed digest mismatch and no guard failure anywhere; the band sweeps are gated by their own `sweep.json` (`failures 0`), and the artifact/pool directories by their `campaign.json` (`status ok`, `failures 0`).
### 11.4 Coverage statement
Measured: nine cells × {`torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-dense-iree`} (plus `evoxc-dense-iree-raised` on every cell of `main/`), `evoxc-batch-iree` on all nine, the formation control on eight, all three value workloads on the eight `main/` cells (§5.3), the ten-stage split and the artifact tier on two cells + the pool sweep at one, `gp_step` on three, and both band sweeps.
Not measured, with reasons: default-cap dense on `d7` (OOM / timeout, §1.1), the formation control at pop 8 192 (cost), the batch route on `d8` (compile-infeasible), `EVOGP_BATCH_BANDS=auto` and the remaining `EVOGP_BATCH_*` knobs at this scale (settled off the shipped path), `evoxc-specialized-*` (round 1's column), and any `CustomLoss`/multi-output or robotics workload — the routes have still never been timed there.
### 11.5 Where the round's brief and the records disagree
Read the records, not the brief; these are the places they differ.
1. **"batch beats dense on 7 of 8 cells"** is right for the eight cells of `main/` against the default cap — the single loss is `p2048-L64-i32-N1024` at **1.12×** (12.41 against 11.08 ms), not round 1's 0.86× or this report's first draft's 0.90× — and right again against the raised cap (1.13×). It becomes **8 of 9** once `p8192/`'s record is counted (1.81×), and **"batch loses to `torch-batched` on 7 of 8"** becomes 8 of 9 with `p8192` added (0.42×). §2.5 is why the pop-2 048 ratio moved between windows.
2. **"`forward` where cheap"** was the first draft's scope for the value column; the re-measure prices both value workloads on all eight cells, and the answer is not uniform: the batch route is 2.27–5.60× dearer than dense on `forward` (its weakest workload, §5.3) and 0.16–0.89× of dense on `batch_forward` (its strongest), and `torch-batched` wins `batch_forward` on 7 of 8 cells by 1.07–2.20× — the brief's "beats `torch-batched` by 1.07–2.20×" has the sign backwards.
3. **"on `sr_fitness` all seven arms hash identically"** is not what the records say, and it is not what `check.py` checks: the gate sha-compares only `VALUE_WORKLOADS` (`forward`, `batch_forward`) and gates `sr_fitness` on NaN/±inf placement, so the absence of an `sr_fitness` WARN means "not compared", not "agreed". The records split `sr_fitness` into five sha256 groups in every cell (torch-per-tree / torch-batched / dense+auto+raised / batch / batch-default) while agreeing on placement and to 7.3e-13 relative on `abs_sum`; §3.3 states it that way.
4. **"8 files / 80 MB per cell at pool 4"**: the 8-file count is in the records (`batch_artifact_files`), the byte size is not — no counter records it, so §6.4 states the count only. The second cell writes 16 files (two workloads × 8).
5. **"the hit path still pays 86 % trace+merge"** is right for a single-workload window (84.6 % at pool 4, 81.8–86.6 % across pools) but not for the two-workload `artifact/warm-hit` records, where the stage window spans two stubs and the trace+merge sum reads 139–157 % of the *first* call (`batch_trace_calls 2`); §6.2 uses the single-workload pool records for that reason. The artifact load itself is not stage-instrumented (0 load calls on a hit), so the residual is a bound.
6. **Formation deltas**: the first draft's cold deltas (−11.3 / −19.3 / −0.6 / −2.0 / −6.1 / −3.5 / −36.5 / −17.4 % on the loss plan) and its warm `p256-L64-i32-N4096` delta (−0.1 %, 5.2747 → 5.2718) belong to the superseded window; the re-measure reads −13.7 / −20.0 / −0.1 / −1.7 / −5.5 / −6.0 / −39.1 / −14.7 % cold and **+2.0 %** (5.1837 → 5.2869) warm there, with the three-plan totals in §2.3 as the honest cold figure.
7. **The `p256-N4096` dense-raised value** is 5.8181 ms (5.82 in the tables; the superseded window read 5.8036), and the `auto` before-values for the row axis are round 1's 50.47 / 110.21 / 197.31 / 446.60 (`LARGE_SCALE.md` §7.1), not the 51.90 / 107.59 / 195.51 / 452.28 figures that report quotes for its `evoxc-specialized-numpy` column.
8. **The `d7` `auto` record moved out of the fallback**: the first draft's note that `main/`'s `evoxc-auto` there is the pre-raise 857.22 ms no longer holds — the re-measure is the raised dense graph at 105.79 ms (`auto_trace_cap` `auto 128`) and the 857.22 ms record is gone (§2.5, §4.2). The pre-raise price survives only in `LARGE_SCALE.md` §2.1's 889.92 ms `evoxc-specialized-numpy` record.
