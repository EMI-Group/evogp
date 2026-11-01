# Large-scale round: the batch route against dense at pop ≥ 2048 and N ≥ 4096
> **Superseded where the routes changed:** `LARGE_SCALE_ROUND2.md` re-measures this column after the `EVOGP_BATCH_DISPATCH_FORMATION`, `auto`-policy and artifact-tier work and replaces this report's `batch` and `evoxc-auto` numbers (its `dense`, raised-dense and torch-baseline columns are re-measured there and agree within noise); everything else here stands.

This round asks what happens to the two compiled evoxc evaluation routes when the population, the data rows and the tree size leave the region `FINAL.md` and `BATCH.md` cover.
`FINAL.md` stops at pop 1000 / `N` = 1024 and `BATCH.md` at pop 1000 / 111-node trees; the grid below adds the population axis (2 048 → 8 192), the row axis (`N` 4 096 / 16 384), a node-count axis above the dense trace cap (123- and 201-node trees at pop 2 048) and one input-width cell (`input_len` 64).
The question is stated as a cost question, not a feature question: on the same populations, same data, same session, is `EVOGP_EVAL_MODE=batch` a better evaluator than the shipped `dense+iree` route once the population is large enough that a per-population merged graph might amortize, and what does either route cost to compile?
Two things were expected to move at this scale — host glue as a share of the warm call (dense's glue was 15.3 % at pop 256, `FINAL.md` §7) and the merged graph's compile against a population 8× larger — and both were measured rather than assumed.
The round also prices the compile envelope itself, because at pop 4 096 a batch compile is already 3 minutes and at pop 8 192 it does not finish inside a 15-minute cap.

Box: one AMD Instinct MI300X VF (191.7 GB HBM), torch 2.13.0+rocm7.1, HIP 7.1.52802, IREE 3.11.0, numpy 2.5.4, python 3.12.3, `EVOXC_DISABLE_LOCATIONS=1`.
Protocol: one process per `(cell, arm)` behind the idle-device guard (`guard.state.gpu_pids.idle true`, `contaminated false` on every record), sequential, one cold call timed separately from `warmup=2` + `reps=5` sync-bracketed warm samples, `gp_warmup=1` and `gp_reps=5` (3 for the two batch generation cells), seed 0, a 900 s per-config cap.
Provenance: the campaign ran against tree `009d47b2` and evoxc `b0544d5`; the probe against `40ed2196` / `b0544d5`; the economics column against `0195eb6` / `b0544d5` (one child of the long-tree pair imported `be145ee`, see §8.3); the band and pool sweeps against `0195eb6` / `be145ee`.
Every number below is read from a committed record under `results/largescale/` or `results/largascale/` (§10.1 states which directory holds what).

## 1. The column

Ten cells, `common.large_cells()`, scheduled with `--cells large`, ids of the form `p<pop>-L<tree_len>-i<input_len>-N<rows>[-d<cap>]`:

| axis | cells | what it moves |
|---|---|---|
| population scale | `p2048-L64-i32-N1024`, `p4096-L64-i32-N1024`, `p8192-L64-i32-N1024` | population 2 048 → 8 192 at a fixed shape; this is where the host glue either amortizes against the graph run or does not |
| row scale | `p256-L64-i32-N4096`, `p256-L64-i32-N16384`, `p1000-L64-i32-N4096`, `p1000-L64-i32-N16384` | `N` 4 096 / 16 384 at pop 256 and 1 000; the axis a per-dispatch overhead amortizes on, and the axis dense pays for |
| node count past the cap | `p2048-L128-i32-N1024-d7`, `p2048-L256-i32-N1024-d8` | 123- and 201-node trees (`max_layer_cnt` 7 / 8) at pop 2 048, i.e. past `EVOGP_DENSE_MAX_TREE_LEN` = 64 |
| input width | `p4096-L64-i64-N1024` | `input_len` 64 instead of 32 |

The memory rule of one cell is `pop * tree_len * rows * 4` bytes for the dense route's largest intermediate, checked against `common.CELL_TENSOR_CEILING_BYTES` (8 GiB) by `common.cell_tensor_bytes`.
The worst cell of the column is `p1000-L64-i32-N16384` at 4.19 GB, and `p8192-L64-i32-N1024` / `p2048-L256-i32-N1024-d8` sit at 2.15 GB — no cell reaches half the ceiling, so nothing here is memory-bound on paper; §7.2 is the one place where memory decided a measurement anyway.

The populations come from the same `torch.manual_seed(0)` + `Forest.random_generate` recipe as every other column, so only the shape is new:

| cell group | distinct structures | mean / max nodes |
|---|---|---|
| pop 2 048, d5 (`p2048-L64-*`) | 1 681 | 16.0 / 31 |
| pop 4 096, d5 | 3 266 | 15.7 / 31 |
| pop 8 192, d5 | 6 544 | 15.8 / 31 |
| pop 256 / 1 000, d5 | 232 / 841 | 15.9 / 31 and 16.2 / 31 |
| `d7` (`L128`) | 1 665 | 42.9 / 123 |
| `d8` (`L256`) | 1 665 | 68.9 / 201 |

### 1.1 What was scheduled, and what was trimmed

Arms measured: `torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-dense-iree`, `evoxc-specialized-numpy` and `evoxc-batch-iree` on all cells the route could run, plus `evoxc-dense-iree-raised` (`EVOGP_DENSE_MAX_TREE_LEN` pinned to the cell's `tree_len`) on the two long-tree cells.
The trims are all consequences of one of two numbers — the compile cost and the per-structure compile cost — and none of them is a library defect:

| trimmed | why |
|---|---|
| `evoxc-batch-iree` on `p8192-L64-i32-N1024` | not scheduled at all: the probe's 900 s cap fired while four `iree-compile` workers had finished **no** chunk of the 8×~1 000-function plan (`results/largescale/probe`, probe A, `campaign.json` status `timeout`, detail `>900s`) |
| `evoxc-batch-iree` on `p2048-L256-i32-N1024-d8` | scheduled and recorded as `timeout` at 900.63 s (`campaign-g3c-batch-d8.json`); ~150–190 k emitted ops is 2× the d7 plan |
| `evoxc-specialized-iree` anywhere in the round | 271.6–283.9 ms of compile per distinct tree structure (the campaign's `protocol.specialized_iree_compile_ms_per_structure`, `measured_range` 271.6–283.9 with source `probe p16/p64`); 232–6 544 structures is 1–31 minutes of compile per evaluation |
| `evoxc-specialized-numpy` on `p8192-L64-i32-N1024` | timed out at 900.14 s through the per-tree fallback on 6 544 structures (`campaign-g1-nonbatch.json`, config 15) — no record for that cell |
| `evoxc-dense-iree` (default cap) on both long-tree cells | no record: past the cap dense declines into per-structure `specialized`+iree, which reached the box's cgroup cap in ~8 minutes (twice at d7; not retried at d8). The evidence is the round's commit message, not a record — see §7.2 |
| batch `gp_step` beyond two cells | each generation re-pays a 20–250 s compile, and two cells already settle the loop verdict (§5.3) |

Total GPU wall, from the drivers' own `elapsed_s`: campaign 4 314.8 s (71.9 min, five committed snapshots plus the i64 group — plus the two OOM'd dense attempts, ~8 min each per the round's commit message), probe ≈31 min (`PROBE.md`), economics + pool sweeps 801.2 s (13.4 min), band sweep 5 855.1 s (97.6 min) — ≈3 h 34 min measured, ≈3 h 50 min including the OOM attempts.
That is above the round's original 2–3 h budget, and the overrun is the band sweep: its five configs cost 1 086–1 389 s each because they pin `EVOGP_BATCH_COMPILE_WORKERS=1`, so one single-body compile of the pop-2 048 plan is 18–23 minutes.
The band sweep therefore ran `--workloads sr_fitness` only; the aborted three-workload schedule is kept as `results/largascale/band/aborted-3workload-sweep.json` (`configs: []`, `timeout_s 1200`).

## 2. The headline: per-cell warm medians

### 2.1 `sr_fitness` (the loss call)

Warm `median_ms` with `min_ms` in parentheses; `—` means no record for that cell/arm.

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-batch-iree | dense / batch | torch-batched / batch |
|---|---|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 249.44 (247.05) | **7.12** (7.11) | 10.99 (10.97) | 374.89 (371.99) | 11.06 (11.02) | — | 12.79 (12.70) | 0.86× | 0.56× |
| `p4096-L64-i32-N1024` | 493.83 (491.46) | **10.51** (10.48) | 37.18 (36.88) | 734.53 (729.23) | 37.06 (37.01) | — | 23.24 (23.20) | 1.60× | 0.45× |
| `p8192-L64-i32-N1024` | 1023.15 (1005.41) | **17.64** (17.63) | 76.34 (76.12) | timeout | 76.23 (76.06) | — | not scheduled | — | — |
| `p256-L64-i32-N4096` | 30.76 (30.49) | **5.04** (5.03) | 50.47 (49.57) | 51.90 (51.60) | 5.82 (5.77) | — | 5.23 (5.19) | 1.11× | 0.96× |
| `p256-L64-i32-N16384` | 30.95 (30.60) | **10.43** (10.42) | 110.21 (109.00) | 107.59 (106.93) | 78.12 (77.97) | — | 12.61 (12.48) | 6.20× | 0.83× |
| `p1000-L64-i32-N4096` | 122.75 (121.53) | **10.22** (10.19) | 197.31 (195.20) | 195.51 (193.99) | 39.31 (39.18) | — | 13.03 (12.97) | 3.02× | 0.78× |
| `p1000-L64-i32-N16384` | 122.69 (122.57) | **32.69** (32.58) | 446.60 (446.43) | 452.28 (448.54) | 117.19 (117.11) | — | 37.41 (37.30) | 3.13× | 0.87× |
| `p2048-L128-i32-N1024-d7` | 601.97 (600.73) | 26.43 (26.41) | 891.23 (886.44) | 889.92 (884.01) | — | 105.65 (105.58) | **25.07** (24.74) | 4.21× | 1.05× |
| `p2048-L256-i32-N1024-d8` | 889.00 (866.29) | **48.62** (48.61) | 1355.62 (1353.45) | 1345.21 (1343.03) | — | 240.28 (240.24) | timeout | — | — |
| `p4096-L64-i64-N1024` | 478.04 (474.90) | **10.42** (10.38) | 37.23 (37.11) | 691.28 (675.93) | 37.02 (36.95) | — | 24.56 (24.37) | 1.51× | 0.42× |

Both ratio columns divide the other arm's median by `evoxc-batch-iree`'s — `dense / batch` uses the `evoxc-dense-iree` column (or `evoxc-dense-iree-raised` on the two long-tree cells), `torch-batched / batch` the `torch-batched` one — so the convention is the same in all three tables: **above 1 means batch is the cheaper arm, below 1 means the other arm is**.
`torch-batched` wins the loss workload on 9 of the 10 cells and `evoxc-batch-iree` wins the tenth (`d7`, 25.07 against 26.43 ms); on the other seven cells batch has a record for, `torch-batched` is 1.04–2.38× faster (0.42–0.96× from batch's side).
Against dense the ordering flips with the row axis: at `N` = 1024 batch is 1.5–1.6× cheaper at pop 4 096 but 0.86× at pop 2 048, while at `N` = 16 384 it is 6.20× cheaper at pop 256 and 3.13× at pop 1 000.

### 2.2 `batch_forward` (the whole population on all rows)

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-batch-iree | dense / batch | torch-batched / batch |
|---|---|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 189.85 (189.11) | **7.11** (7.08) | 14.40 (14.17) | 370.67 (369.28) | 14.50 (14.16) | — | 12.28 (11.98) | 1.18× | 0.58× |
| `p4096-L64-i32-N1024` | 373.97 (370.43) | **10.47** (10.47) | 46.00 (45.91) | 739.42 (730.27) | 46.53 (46.30) | — | 22.07 (21.93) | 2.11× | 0.47× |
| `p8192-L64-i32-N1024` | 783.24 (780.11) | **17.59** (17.56) | 93.42 (93.08) | timeout | 94.25 (93.70) | — | not scheduled | — | — |
| `p256-L64-i32-N4096` | 23.64 (23.46) | **5.02** (4.98) | 50.20 (49.91) | 50.97 (50.58) | 7.49 (7.22) | — | 7.15 (6.97) | 1.05× | 0.70× |
| `p256-L64-i32-N16384` | 23.89 (23.79) | **10.44** (10.37) | 109.08 (107.50) | 105.68 (104.28) | 79.63 (79.02) | — | 13.00 (12.94) | 6.13× | 0.80× |
| `p1000-L64-i32-N4096` | 94.79 (93.72) | **10.18** (10.15) | 197.89 (196.54) | 196.27 (194.30) | 47.95 (47.63) | — | 14.06 (13.53) | 3.41× | 0.72× |
| `p1000-L64-i32-N16384` | 95.23 (94.26) | **32.67** (32.49) | 441.07 (439.75) | 439.51 (438.81) | 149.77 (149.12) | — | 52.26 (51.46) | 2.87× | 0.63× |
| `p2048-L128-i32-N1024-d7` | 533.88 (532.00) | 26.42 (26.40) | 891.59 (889.22) | 907.95 (872.95) | — | 109.98 (109.85) | **18.64** (18.52) | 5.90× | 1.42× |
| `p2048-L256-i32-N1024-d8` | 816.81 (808.05) | **48.59** (48.54) | 1369.31 (1347.76) | 1323.57 (1304.68) | — | 244.88 (244.35) | timeout | — | — |
| `p4096-L64-i64-N1024` | 362.80 (356.05) | **10.37** (10.37) | 46.56 (46.29) | 722.35 (679.48) | 47.04 (46.65) | — | 22.53 (21.59) | 2.09× | 0.46× |

The whole-population forward repeats the loss column's ordering (batch beats dense everywhere except `p2048-L64`, where it wins only 1.18×, and wins by 2.1–6.1× from `N` = 4 096 on), and it is the arm's best ratio against dense among the two whole-population workloads.

### 2.3 `forward` (one row per tree)

| cell | torch-per-tree | torch-batched | evoxc-auto | evoxc-specialized-numpy | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-batch-iree | dense / batch | torch-batched / batch |
|---|---|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 196.33 (194.57) | 4.49 (4.43) | 0.86 (0.84) | 351.24 (346.11) | **0.92** (0.89) | — | 3.33 (3.32) | 0.28× | 1.35× |
| `p4096-L64-i32-N1024` | 383.68 (382.26) | 4.43 (4.40) | 0.98 (0.96) | 710.39 (692.57) | **0.88** (0.85) | — | 5.23 (5.17) | 0.17× | 0.85× |
| `p8192-L64-i32-N1024` | 807.61 (801.87) | 4.57 (4.55) | 0.92 (0.91) | timeout | **0.91** (0.89) | — | not scheduled | — | — |
| `p256-L64-i32-N4096` | 24.18 (24.15) | 4.52 (4.46) | 0.86 (0.83) | 44.33 (44.10) | **0.85** (0.84) | — | 1.92 (1.91) | 0.44× | 2.36× |
| `p256-L64-i32-N16384` | 24.27 (24.08) | 4.45 (4.43) | 0.84 (0.82) | 41.83 (40.48) | **0.87** (0.83) | — | 1.87 (1.79) | 0.46× | 2.38× |
| `p1000-L64-i32-N4096` | 95.87 (95.04) | 4.46 (4.43) | 0.86 (0.84) | 163.16 (159.65) | **0.89** (0.87) | — | 2.55 (2.50) | 0.35× | 1.75× |
| `p1000-L64-i32-N16384` | 97.10 (94.61) | 4.42 (4.40) | 0.85 (0.83) | 170.42 (169.33) | **0.87** (0.86) | — | 2.47 (2.41) | 0.35× | 1.79× |
| `p2048-L128-i32-N1024-d7` | 543.54 (539.24) | 16.31 (16.28) | 821.49 (819.60) | 783.09 (757.16) | — | **1.22** (1.19) | 6.86 (6.78) | 0.18× | 2.38× |
| `p2048-L256-i32-N1024-d8` | 841.79 (816.80) | 26.40 (24.98) | 1257.97 (1244.25) | 1251.88 (1235.06) | — | **2.06** (1.97) | timeout | — | — |
| `p4096-L64-i64-N1024` | 374.49 (367.68) | 4.41 (4.10) | 0.87 (0.84) | 636.96 (634.50) | **0.91** (0.87) | — | 5.11 (5.06) | 0.18× | 0.86× |

`forward` evaluates one row per tree, so dense's cost is population- and rows-insensitive (0.85–1.22 ms across the whole column) and it is the fastest arm on all ten cells.
Batch's merged graph pays for the per-row variant of the plan, which is the *largest* of its three plans (`n_functions` 1 862 vs 1 680 for the value plan at pop 2 048, 3 670 vs 3 265 at pop 4 096), so its per-row forward is 1.87–6.86 ms — 0.17–0.46× of dense, though still 1.35–2.38× faster than `torch-batched` on 6 of the 8 cells (it is 1.16–1.18× slower on the two pop-4 096 cells).

### 2.4 Cold: the first call pays the compile

`first_call_ms` of the `sr_fitness` workload, i.e. the cold call of each route's own graph, in ms:

| cell | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-specialized-numpy | evoxc-auto | evoxc-batch-iree | dense / batch |
|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 6 711.9 | — | 7 221.2 | 6 670.1 | 48 310.0 | 7.20× |
| `p4096-L64-i32-N1024` | 6 736.5 | — | 14 336.7 | 6 708.4 | 187 174.9 | 27.8× |
| `p8192-L64-i32-N1024` | 6 803.2 | — | timeout | 6 981.9 | not scheduled | — |
| `p256-L64-i32-N4096` | 6 650.7 | — | 963.3 | 949.9 | **4 384.6** | 0.66× |
| `p256-L64-i32-N16384` | 6 512.6 | — | 1 001.9 | 987.3 | **4 918.3** | 0.76× |
| `p1000-L64-i32-N4096` | 6 380.5 | — | 3 734.9 | 3 617.5 | 17 595.8 | 2.76× |
| `p1000-L64-i32-N16384` | 6 803.8 | — | 4 067.6 | 3 926.4 | 19 579.8 | 2.88× |
| `p2048-L128-i32-N1024-d7` | no record (OOM) | 14 151.5 | 17 349.4 | 17 286.3 | **248 197.8** | 17.5× |
| `p2048-L256-i32-N1024-d8` | no record (OOM) | 41 297.4 | 26 129.1 | 26 500.0 | timeout | — |
| `p4096-L64-i64-N1024` | 6 661.5 | — | 13 452.7 | 6 557.7 | 200 735.2 | 30.1× |

Dense's compile is 6.4–6.8 s on every d5 cell, i.e. population- and rows-insensitive, and it is the cheapest cold start in the table everywhere except `d7`/`d8`, where the cap pin makes it 14.2 s / 41.3 s.
The torch arms have no compile to pay: their cold `sr_fitness` call is 154.3–1 111.3 ms (`torch-per-tree`) and 65.8–101.3 ms (`torch-batched`), so on a cold single evaluation dense is 63–104× dearer than `torch-batched`, and batch's first call at pop 2 048 (48 310.0 ms) is 734× it.
Batch beats dense cold only where the population is small and the rows are many (`p256`: 4 384.6 vs 6 650.7, 4 918.3 vs 6 512.6 ms) — the same regime where it wins warm.

### 2.5 The generation step (`gp_step`)

Total wall per generation with `eval / ops` beside it, median over `reps` steps; the batch cells ran `--gp-reps 3`, everything else the campaign default 5.

| cell | evoxc-dense-iree | evoxc-dense-iree-raised | evoxc-auto | evoxc-specialized-numpy | torch-per-tree | torch-batched | evoxc-batch-iree |
|---|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 21.19 (18.49 / 2.71) | — | 21.72 | 2 150.50 | 113.98 | 147.35 | 11 298.32 (11 295.15 / 3.17) |
| `p256-L64-i32-N16384` | 83.72 (81.17 / 2.55) | — | 334.68 | 329.94 | 18.47 | 28.49 | 3 194.20 (3 190.88 / 3.32) |
| `p4096-L64-i32-N1024` | 57.27 (54.56 / 2.71) | — | 57.85 | 4 489.60 | 227.10 | 290.95 | — |
| `p8192-L64-i32-N1024` | 110.72 (107.98 / 2.75) | — | 109.87 | — | 510.35 | 593.86 | — |
| `p256-L64-i32-N4096` | 10.53 (8.03 / 2.50) | — | 289.47 | 357.42 | 17.86 | 25.33 | — |
| `p1000-L64-i32-N4096` | 47.33 (44.63 / 2.70) | — | 881.66 | 933.52 | 53.65 | 59.77 | — |
| `p1000-L64-i32-N16384` | 127.70 (125.07 / 2.64) | — | 913.76 | 865.34 | 51.19 | 73.86 | — |
| `p2048-L128-i32-N1024-d7` | — | 124.07 (121.31 / 2.75) | 1 857.57 | 1 745.85 | 125.41 | 238.50 | — |
| `p2048-L256-i32-N1024-d8` | — | 273.06 (270.30 / 2.76) | 1 039.82 | 1 066.72 | 92.38 | 229.33 | — |
| `p4096-L64-i64-N1024` | 56.82 (53.93 / 2.89) | — | 56.57 | 4 276.55 | 232.15 | 254.66 | — |

The operator layer is 2.4–3.5 ms per generation on every arm, so a generation is the evaluation plus ~2.5 ms and nothing else — the same operator share `FINAL.md` measured, unchanged at 8× the population.
Batch's generation is 533× dense's at pop 2 048 (11 298.32 vs 21.19 ms) and 38.2× dense's at `p256-L64-i32-N16384` (3 194.20 vs 83.72 ms); the eval-only ratio is 611× and 39.3×.
Those medians hide the mechanism (one recompile per generation), which §5.3 prices on a real loop.

## 3. The verdict

**Batch wins the warm per-call comparison against dense on 7 of the 8 cells it could be measured on, and loses the round anyway, because the compile envelope disqualifies the scale where it wins.**
Where the rows stay at 1 024 the win is thin to moderate (1.51× at `p4096-L64-i64-N1024`, 1.60× at `p4096-L64-i32-N1024`) and turns into a loss at pop 2 048 (0.86×), where dense's 21.19 ms generation is already small enough that the merged graph's own overhead dominates a 12.79 ms call; the pop-256 / `N` = 4 096 cell is the one thin win on the row axis (1.11×).
At `N` ≥ 4 096 the win is decisive (3.02–6.20× at pop 256/1 000) because dense's cost rises with the row count while the merged graph's does not — and on the node-count axis past the cap it is 4.21× against a raised-cap dense and 35.6× against what `auto` actually picks there (891.23 ms).
Against `torch-batched` batch is slower on the loss workload in 7 of its 8 measured cells (by 1.04–2.38×) and faster only on `d7` (1.05×); on the per-row `forward` it is 1.35–2.38× faster everywhere except the two pop-4 096 cells (1.16–1.18× slower there).
Cold, batch is cheaper than dense only in the two `p256` row cells (0.66× / 0.76×) and dearer everywhere else, from 2.76× at pop 1 000 to 30.1× at pop 4 096 with `input_len` 64.
And the population axis is where the round ends: at pop 8 192 the batch compile does not finish inside 900 s (§5.2), so the region where a merged graph would have the most work to amortize is the region it cannot enter.
Nothing in the knob sweeps changes this ordering: the band size is unobservable at pop 2 048 because the rocm binding cap forces every banded request to the same 15-band merge (§6.1), pool 8 halves the cold call at 17 % warm (§6.3), and the flat merge is 3.1–3.2× worse warm than any banded plan (§6.2).
What would change it is a cheaper compile, not a different partition — §9 lists the five levers in the order the records support them.

## 4. Where a warm call and a cold call go

### 4.1 The warm call is dispatch-bound and recompile-free

Ten-stage split of one `sr_fitness` warm call (`batch_stage_stats()` window, `results/largascale/economics/**/batch.json`), per call:

| cell | warm median | `run` | `upload` | `gather` | residual (host) | merge / trace / decode / dedup / lower / compile / load |
|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 12.8468 | 10.8205 (84.2 %, 4 calls) | 0.8491 (6.6 %) | 0.7895 (6.1 %) | 0.3877 (3.0 %) | 0 calls, 0 ms each |
| `p1000-L64-i32-N16384` | 37.0642 | 33.5028 (90.4 %, 4 calls) | 1.9171 (5.2 %) | 0.8430 (2.3 %) | 0.8013 (2.2 %) | 0 calls, 0 ms each |
| `p256-L64-i32-N16384` | 12.3513 | 8.9885 (72.8 %, 4 calls) | 1.8881 (15.3 %) | 0.7958 (6.4 %) | 0.6789 (5.5 %) | 0 calls, 0 ms each |
| `p2048-L128-i32-N1024-d7` | 24.9438 | 22.5527 (90.4 %, 4 calls) | 1.0103 (4.1 %) | 0.9884 (4.0 %) | 0.3924 (1.6 %) | 0 calls, 0 ms each |

Every warm call is four upload/run pairs (one per chunk of the shipped pool-4 plan) and five gathers; decode, dedup, trace, merge, lower, compile and load are all at zero calls, i.e. the plan tables hit and no host-side work is repeated.
The device run is 72.8–90.4 % of the call, so the warm path is the dispatch-bound graph `FINAL.md` §7 described, and the host glue that was 15.3 % of a dense call at pop 256 has fallen to 1.6–5.5 % here — not because the glue got cheaper but because the graph got longer.
The `p256-L64-i32-N16384` cell is the exception to the shape: its plan is only 5 497 emitted ops, so the four uploads are 15.3 % of a 12.35 ms call while the run is 72.8 %.

### 4.2 The merge and band shape is population-driven, rows-insensitive

The loss plan of each measured cell (shipped defaults: band size 16, pool 4, chunking on):

| cell | deduped functions | emitted ops | chunks × functions | band clamp |
|---|---|---|---|---|
| `p256-L64-i32-N4096` | 232 | 5 497 | 4 × 58 | no (15-band cap not reached) |
| `p256-L64-i32-N16384` | 232 | 5 497 | 4 × 58 | no |
| `p1000-L64-i32-N4096` | 841 | 21 339 | 4 × [211, 210, 210, 210] | no |
| `p1000-L64-i32-N16384` | 841 | 21 339 | 4 × [211, 210, 210, 210] | no |
| `p2048-L64-i32-N1024` | 1 680 | 41 376 | 4 × 420 | yes (model asks 27 bands/chunk, loaded 15) |
| `p4096-L64-i32-N1024` | 3 265 | 78 189 | 4 × [817, 816, 816, 816] | yes |
| `p4096-L64-i64-N1024` | 3 297 | 80 968 | 4 × [825, 824, 824, 824] | yes |
| `p2048-L128-i32-N1024-d7` | 1 664 | 93 751 | 4 × 416 | yes |
| `p8192-L64-i32-N1024` | — | — | — | never compiled |
| `p2048-L256-i32-N1024-d8` | — | — | — | compile timed out |

Two facts follow from the table and are visible in the records' `band_note`.
Rows are free: the same 232 or 841 functions give the same plan and the same `n_functions` / `emitted_ops` at `N` = 4 096 and 16 384, which is exactly why batch's warm call barely moves across the row axis while dense's grows 13.4× / 3.0×.
Population is not: `emitted_ops` grows with the deduped function count (232 → 841 → 1 680 → 3 265 → 3 297 → 1 664 at d7), and from pop 2 048 on the band-size model asks for more bands per chunk than the rocm binding cap allows (`band_requested_bands` 27 at pop 2 048), so the plan is clamped to 15 bands per chunk and every banded request at that scale lands on the same merge.
The node-count axis is what makes the d7 plan expensive at only 1 664 functions: 93 751 emitted ops against 41 376 at 1 680 functions, i.e. 2.3× the ops for the same function count, because each 123-node tree contributes more nodes per program.

### 4.3 The cold call is 90–97 % the device compile

Cold first call with its ten stages (`first_call.stage_ms`), same cells:

| cell | cold call | host pre-compile (decode + dedup + trace + merge) | of which merge | `lower` (4 chunks) | compile pool wall | load / upload / run / gather |
|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 47 194.0 | 3 477.6 (7.4 %) | 2 321.3 (4.9 %) | 2 821.9 (1 job max 754.3) | 43 643.0 (92.5 %) | 403.6 / 45.3 / 13.3 / 0.9 |
| `p1000-L64-i32-N16384` | 20 247.0 | 1 711.8 (8.5 %) | 1 136.3 (5.6 %) | 3 612.8 (max 1 077.6) | 18 444.0 (91.1 %) | 320.6 / 47.1 / 34.4 / 1.0 |
| `p256-L64-i32-N16384` | 4 843.4 | 407.4 (8.4 %) | 246.9 (5.1 %) | 1 295.8 (max 448.5) | 4 374.5 (90.3 %) | 277.5 / 47.1 / 10.0 / 1.0 |
| `p2048-L128-i32-N1024-d7` | 252 292.3 | 8 037.0 (3.2 %) | 5 513.8 (2.2 %) | 5 127.3 (max 1 381.0) | 244 157.9 (96.8 %) | 563.5 / 42.1 / 26.8 / 1.1 |

The compile pool is 90.3–96.8 % of the cold call by its own wall counter, and the per-chunk compile timers sum to 3.65× the wall at pop 2 048 (159 414.9 ms over 4 jobs), so a "compile / 4" reading understates the pool's serialization: the pool scales 159 414.9 / 43 643.0 = 3.65× and not 4×.
Host pre-compile work is 3.2–8.5 % of the call, and `merge` is the largest part of it (2 321.3 ms at pop 2 048, 5 513.8 ms at d7) — a ten-second-per-merge floor is what a 44–244 s compile sits on top of.
Attribution is complete: the identity/exec tables miss 4 (one per chunk) and the plan table misses 1, with the content table at 1 miss, i.e. no compile is unaccounted for.
One caveat on the percentages: `compile_share` in the loop records is the stage-counter ratio that can exceed 1 (2.48–3.09 here) because it sums concurrent per-chunk timers over one loop wall — it is not wall-clock attribution (§5.3 quotes the wall share instead).

## 5. Compile economics and the feasibility envelope

### 5.1 The cost law: quadratic in emitted ops, rows-insensitive

| cell | emitted ops | cold compile (campaign record) | fit `48.9 s × (ops / 41 376)²` |
|---|---|---|---|
| `p256-L64-i32-N4096` | 5 497 | 4 384.6 | 0.86 s (floor-dominated) |
| `p1000-L64-i32-N16384` | 21 339 | 19 579.8 | 13.0 s |
| `p2048-L64-i32-N1024` | 41 376 | 48 310.0 | 48.9 s (anchor) |
| `p4096-L64-i32-N1024` | 78 189 | 187 174.9 | 174.6 s (−6.7 %) |
| `p4096-L64-i64-N1024` | 80 968 | 200 735.2 | 187.3 s (−6.7 %) |
| `p2048-L128-i32-N1024-d7` | 93 751 | 248 197.8 | 251.1 s (+1.2 %) |

Above 41 000 emitted ops the quadratic fit reproduces the measured compile within 7 %; below the anchor it under-predicts, by 34 % at 21 339 ops (13.0 s predicted, 19 579.8 ms measured) and by a factor of five at 5 497 ops (0.86 s against 4 384.6 ms), because a few seconds of fixed work — pool start-up, the merge, the four chunk uploads — do not scale with ops.
The row axis is free: `p1000-L64-i32-N4096` and `p1000-L64-i32-N16384` share the same 21 339-op plan and compile in 17 595.8 / 19 579.8 ms, and the two `p256` cells compile in 4 384.6 / 4 918.3 ms with identical plans.
Population is the axis that bites, twice over: the plan is linear in the deduped function count, and the per-op compile cost is itself superlinear, so doubling the population from 2 048 to 4 096 multiplies the cold call by 3.9× (48.3 → 187.2 s).

### 5.2 Two declared boundaries

`p8192-L64-i32-N1024` dedups to 6 544 structures, ≈2× the pop-4 096 plan (78 189 ops) — the fit predicts 0.70–0.93 ks, and the probe recorded the cap firing at 900.21 s with four `iree-compile` workers alive and no chunk finished (probe A, `results/largescale/probe/campaign.json`; `PROBE.md` reads worker age 806 s, 776 s of compiling).
`p2048-L256-i32-N1024-d8` (201-node trees, ~150–190 k emitted ops) was scheduled anyway and recorded as `timeout` at 900.63 s.
Both are labeled **compile infeasible**, not defects: the route compiles one graph per population multiset, the compile is quadratic in the graph, and 6 544 structures is past what a 15-minute cap affords on this box.
The envelope this establishes, in the one currency that matters: **pop 1 000 at 841 functions is a 20 s compile; pop 2 048 at 1 680 functions is 48 s; pop 4 096 at 3 265 functions is 187 s; pop 8 192 is over 900 s.**

### 5.3 The loop: a real `StandardPipeline.step()`

Three generations of the library's own loop (`batch_economics.py`, one process per route), with dense measured in the same window:

| cell | batch total (eval) | dense total (eval) | ratio total | ratio eval-only | per generation, batch | per generation, dense |
|---|---|---|---|---|---|---|
| `p2048-L64-i32-N1024` | 36 740.9 (36 461.4) | 329.8 (52.2) | 111.4× | 698.5× | 285.9 / 24 972.5 / 11 482.5 | 283.3 / 23.9 / 22.6 |
| `p1000-L64-i32-N16384` | 22 166.9 (21 883.1) | 641.6 (370.2) | 34.5× | 59.1× | 314.4 / 13 455.4 / 8 397.1 | 384.8 / 129.0 / 127.8 |
| `p256-L64-i32-N16384` | 7 428.2 (7 150.1) | 518.7 (241.9) | 14.3× | 29.6× | 283.9 / 4 115.6 / 3 028.7 | 350.0 / 84.7 / 84.0 |
| `p2048-L128-i32-N1024-d7` | 57 954.8 (57 680.3) | 626.1 (349.6) | 92.6× | 165.0× | 292.9 / 46 195.2 / 11 466.7 | 376.9 / 126.8 / 122.4 |

Generation 1 is a hit in every loop (the timed first call already compiled the initial population), and generations 2 and 3 recompile: `generations_with_compile` 2 of 3, `distinct_population_signatures` 3, and 4 exec + 1 plan miss per recompiling generation.
That hit is why the loop's total ratio (111.4× at pop 2 048) is smaller than the per-generation ratio in steady state: dense's own generation 1 pays its 283.3–384.8 ms compile, so its steady state is generations 2–3 (22.6–23.9 / 127.8–129.0 / 84.0–84.7 / 122.4–126.8 ms) while batch's is 11 482.5 / 8 397.1 / 3 028.7 / 11 466.7 ms.
Read on the steady state, batch's generation is **508–1 045×** dense's at pop 2 048, **66–104×** at pop 1 000 / `N` = 16 384, **36–49×** at pop 256 / `N` = 16 384 and **94–364×** at d7.
The content-addressed plan key does not help a turnover loop: 3 attempts and **0 hits** in every loop (`{rows 3, with_content_hit 0, with_content_miss 2, no_attempt 1}`), so the `reuse_probe.py` win `BATCH.md` §13.10 measured for a rebuilt-but-equal population has no counterpart here.
`compile_ms_share` (the wall share of the recompiling generations over the loop) reads 0.9618–0.9949, i.e. the loop is the compile.

### 5.4 Is a full-turnover GP loop feasible at that size at all?

At pop 2 048 / `N` = 1 024, a turnover loop costs **111× dense's generation** — 19 626.71 ms of loop for 302.48 ms of dense work on the round-3 evidence, and 36 740.9 vs 329.8 ms here.
At 6 544 structures (pop 8 192) the compile does not terminate inside 900 s, so the question stops being "is it faster" and becomes "can it start".
The amortization arithmetic has no positive root in either regime: the compile is paid **per generation** (a new structure multiset every step, so 2 of 3 generations compile), and the warm call it buys is *slower* than dense's at pop 2 048 (12.79 vs 11.06 ms) and only 3.02–6.20× faster where the rows dominate — against a compile of 17.6–19.6 s at pop 1 000 and 48.3 s at pop 2 048.
The repayment arithmetic `(batch cold − dense cold) / (dense warm − batch warm)` never turns positive at pop 2 048: batch's compile is 41 598.1 ms dearer (48 310.0 against 6 711.9) and its warm call is 1.73 ms slower (12.79 against 11.06), so no number of warm calls repays it.
In the row cells both terms already favour batch — at `p256-L64-i32-N16384` it is 1 594.3 ms cheaper cold (4 918.3 against 6 512.6) and 65.5 ms cheaper per warm call (12.61 against 78.12) — so batch is ahead from the first call and there is nothing left to amortize.
So the honest answer is: **not at that size in a GP loop**, and yes in one narrow fixed-population regime (`p256`, many rows), which is the same shape of answer `BATCH.md` §15.5 gave at pop 256 and 64 — only the ordering of the two columns has changed, because at `N` ≥ 4 096 dense is the arm that scales badly.

## 6. The knob sweeps

### 6.1 Band size at pop 2 048: the clamp decides, and 16 is not contradicted

`band_sweep.py` at `p2048-L64-i32-N1024`, `--workloads sr_fitness`, every band-partition config pinned to `EVOGP_BATCH_COMPILE_WORKERS=1` so a difference is the band partition alone:

| config | cold | warm `sr_fitness` | deduped functions | emitted ops | bands | band sizes |
|---|---|---|---|---|---|---|
| `b8` | 1 090 272.9 | 11.6830 (11.6479) | 1 680 | 38 640 | 15 | `[112] × 15` |
| `b16` | 1 085 929.6 | 12.0772 (11.9440) | 1 680 | 38 640 | 15 | `[112] × 15` |
| `b32` | 1 101 281.7 | 11.9338 (11.8607) | 1 680 | 38 640 | 15 | `[112] × 15` |
| `k15` | 1 150 159.5 | 11.8283 (11.6535) | 1 680 | 38 640 | 15 | `[112] × 15` |
| `flat` | 1 379 027.5 | 37.6159 (37.5014) | 1 680 | 31 735 | 1 | `[1 680]` |

The three band SIZE configs produce the identical plan: at 1 680 functions the model asks 210 / 105 / 53 bands and the rocm cap of 15 clamps all three, so the measured spread (11.68–12.08 ms warm, 3.4 %; 1 085.9–1 101.3 s cold, 1.4 %) is this box's noise floor and says nothing about band size.
`k15` reaches the same 15 × 112 merge through an explicit count instead of a clamp, with `band_clamped: false` and a warm call of 11.83 ms — inside that same noise band, so the cap pin is not better or worse, it is the same plan reached differently.
The verdict from `BATCH.md` §14.1 therefore stands unrevised at this scale for a different reason than it stood at pop 256: there the band size was a second-order knob, here it is unobservable.
Consequence for anyone reading a warm column at pop ≥ 2 048: `EVOGP_BATCH_BAND_SIZE` cannot be tuned there, because the binding cap already decides the partition.

### 6.2 The flat merge is 3.1–3.2× worse warm

One band of 1 680 functions emits **17.9 % fewer** ops than the 15-band merge (31 735 vs 38 640) and costs 3.11–3.22× more warm (37.6159 ms against 11.6830–12.0772) and 1.20–1.27× more cold (1 379 027.5 against 1 085 929.6–1 150 159.5 ms).
That is the same direction `BATCH.md` §14.4 measured at pop 256 (34 % on the per-row `forward`, 15 % on warm `sr_fitness`) and a much larger magnitude, because at 1 680 functions one dispatch carries 1 680 programs instead of 15 dispatches of 112.
The reduction in emitted ops is real and does not pay: the run stage, not the op count, is the cost.
Oracle deviation is config-independent to the last digit: `max_rel` 7.76e-07 on all four banded configs and 8.47e-07 on flat, `max_abs` 32.0 on all five (`validation.sr_fitness` in each record).

### 6.3 Pool width: 4 stays for the steady state, and the crossover is measured

Four widths at `p2048-L64-i32-N1024`, loss plan (the pool-1 row is the band sweep's `b16`, the others are `pool2` / the three-cell economics run / `pool8`):

| pool | chunks × functions | emitted ops | cold | warm `sr_fitness` | crossover vs pool 4 (warm calls) |
|---|---|---|---|---|---|
| 1 | 1 × 1 680 | 38 640 | 1 085 929.6 | 12.0772 | 1 349 700 warm calls |
| 2 | 2 × 840 | 39 852 | 189 308.5 | 11.8305 | 139 836 |
| **4 (shipped)** | 4 × 420 | 41 376 | 47 194.0 | 12.8468 | — |
| 8 | 8 × 210 | 43 254 | **28 755.1** | 15.0892 | 8 223 |

The column is `|cold_pool − cold_4| / |warm_pool − warm_4|` in warm calls of ONE fixed population, the arithmetic `BATCH.md` §15.2 uses — and its sign decides who is ahead first, which is not the same question for all four rows.
Pools 1 and 2 are the dearer-cold case: pool 2 pays 4× the cold call (189 308.5 against 47 194.0 ms) to save 1.0163 ms per warm call and repays after 139 836 calls; pool 1 pays 23× (1 085 929.6 ms) for a warm call that sits inside the noise band (12.0772 against 12.8468) and repays after 1 349 700 calls.
Pool 8 is the *opposite* case: it is 18 438.9 ms cheaper cold (28 755.1 against 47 194.0, 0.61×) and 2.2424 ms dearer warm (15.0892 against 12.8468, 1.17×), so the 8 223 is the number of warm calls after which pool 4's saving *overtakes* pool 8's cheaper start — not a repayment pool 8 owes.
Read that way, pool 8 wins every workload that makes fewer than ≈8 200 warm calls of one population and loses a long steady-state run.
The shipped default of 4 is kept because the column it was chosen from is the warm-dominant one, and finer chunking costs ops as well (38 640 → 43 254, +11.9 %) — each chunk re-emits shared scaffolding, so the pool buys wall-clock parallelism at a real op cost.
**This round does not settle the pool width for a turnover loop, and the brief's argument for 4 does not carry over**: every loop measured here is 96–99 % compile (§5.3), which is the side of the crossover where pool 8 would win and pool 1 would lose worst. `EVOGP_BATCH_COMPILE_WORKERS=8` × a 3-generation loop is the measurement that decides it, and it was not run.

### 6.4 Already on by default, and not swept

Chunk-banding needs no opt-in here: the shipped default (band size 16, pool 4, `EVOGP_BATCH_CHUNK_BANDS` unset) chunks and bands at every cell from 232 deduped functions up, because a band SIZE never refused chunking (`BATCH.md`'s Known Issues — only an explicit `EVOGP_BATCH_BANDS` spelling does).
`EVOGP_BATCH_ALIGNMENT`, `EVOGP_BATCH_SWITCH_OPS` and `EVOGP_BATCH_BANDS=auto` were not swept this round; `auto` remains opt-in and unrecommended (`BATCH.md` §14.1), and the cost model it drives is the one thing that could in principle have picked a non-clamped partition at this scale — a candidate for the next round, not a finding here.

## 7. Policy findings

### 7.1 `auto`'s rows ceiling picks the fallback where dense is 1.4–8.7× faster

`EVOGP_DENSE_AUTO_MAX_ROWS` = 2 048 sends `auto` to the per-tree `specialized`+numpy fallback on every cell with `N` ≥ 4 096, and dense is faster than that fallback on all four of them.
The records say it directly: `evoxc-auto`'s `routes.sr_fitness` carries `would_use_dense: false` and `backend: numpy` while its timings equal `evoxc-specialized-numpy`'s.

| cell | `evoxc-auto` | `evoxc-dense-iree` | dense's advantage per call | per generation (`gp_step`) |
|---|---|---|---|---|
| `p256-L64-i32-N4096` | 50.47 | 5.82 | 8.68× | 289.47 vs 10.53 (27.5×) |
| `p256-L64-i32-N16384` | 110.21 | 78.12 | 1.41× | 334.68 vs 83.72 (4.0×) |
| `p1000-L64-i32-N4096` | 197.31 | 39.31 | 5.02× | 881.66 vs 47.33 (18.6×) |
| `p1000-L64-i32-N16384` | 446.60 | 117.19 | 3.81× | 913.76 vs 127.70 (7.2×) |

At `N` = 1 024 `auto` ≡ dense (10.99 vs 11.06, 37.18 vs 37.06, 76.34 vs 76.23, 37.23 vs 37.02 ms), so the ceiling is the only thing that separates them — and it separates them the wrong way.
This is a policy finding, not a defect in either route: `dense` is a working route on those cells and a caller who sets `EVOGP_EVAL_MODE=dense` gets it. `auto` resolves to the fallback.
On the two long-tree cells the same ceiling is not the issue — `auto` reports `would_use_dense: true` there (rows 1 024, pop ≥ 64) but the forest's `max_tree_len` is 123 / 201 against a cap of 64, so dense raises `DenseUnsupported` and the public API falls back to the per-tree path anyway: `auto` reads 891.23 / 1 355.62 ms, i.e. 8.4× / 5.6× the raised-cap dense column (105.65 / 240.28 ms) and indistinguishable from `evoxc-specialized-numpy` (889.92 / 1 345.21 ms).

### 7.2 The default-cap dense decline on large-pop long-tree cells does not fit in memory

There is no `evoxc-dense-iree` record for `p2048-L128-i32-N1024-d7` or `p2048-L256-i32-N1024-d8`; the arm was not retried after the second kill.
The decline path it takes is per-structure `specialized`+iree with 1 665 distinct structures at pop 2 048, whose compile is ~272–284 ms per structure — minutes per evaluation, and the process reached the box's cgroup cap (~236 GB) in ~8 minutes at d7.
`evoxc-dense-iree-raised` (`EVOGP_DENSE_MAX_TREE_LEN=128` / `256`) is the measured dense column on those cells, and it is a different arm on purpose: its `dense_route` block states `supported: true`, `tree_len_max` 123 ≤ cap 128, `probe_ms` 105.63.
Anyone scheduling a long-tree cell on a large population should pin the cap to the cell's `tree_len` — an unpinned run does not measure the fallback, it dies.

## 8. Numeric validation, digests and caveats

### 8.1 Validation

The band sweep validates every config against the plain-torch oracle (`../../src/evogp/tests/reference_eval.py`) on 16 trees per record: `max_abs` 32.0 and `max_rel` 7.76e-07 (banded) / 8.47e-07 (flat), `bitwise_equal: false` — the batch route's documented non-bitwise `batch_forward`/value path at ~1e-6 relative on this value scale, not a band-size effect (the deviation is identical on all five configs).
The campaign arms are validated by the harness's own `validate.py` contract (12 cases, three function sets) and the gate checks each record's `nan`/`inf`/`finite` counts against the other arms of the cell.
Every loop record's `best_fitness` agrees across routes in a cell to 8 decimal places except where noted (`p4096-L64-i32-N1024`: `evoxc-dense-iree` and `auto` −0.08426816 vs the fallback arms' −0.08425234 — the same 1-ulp-class spread the batch route's non-bitwise value path produces, visible because the arms sample different evaluation orders).

### 8.2 Digest groups and the gate's declared expectations

The gate of record is

```
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/largescale/main \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# -> 57 records / 1530 checks / 0 failures   (24 WARN: 4 structural, 20 digest splits)
```

Its `campaign:` line reads `1 configs, 0 ok, 1 not ok, failures=1, elapsed=900.8 s` because `campaign.json` is rewritten by every driver invocation and the last one was the declared-infeasible d8 batch attempt; the four other groups survive as `campaign-g1-nonbatch.json`, `campaign-g1-nonbatch-i64.json`, `campaign-g2-raised.json`, `campaign-g3a-*.json`, `campaign-g3b-batch-gpstep.json` and `campaign-g3c-batch-d8.json`.
For the same reason the gate's expected-arm set is derived from the last snapshot's arms (the batch arm), which is why the two absent `evoxc-dense-iree` long-tree records are not gate failures — they are declared in §1.1 and in this report instead.
The declared expectations, each with its reason:

| expectation | gate status | reason |
|---|---|---|
| `p8192-L64-i32-N1024` has no `evoxc-batch-iree` record | WARN `missing arms (allowed) … campaign status: never scheduled` | compile infeasible; probe A timed out at 900.21 s (§5.2) |
| `p2048-L256-i32-N1024-d8` has no `evoxc-batch-iree` record | WARN `campaign status 'timeout' (>900s) — excluded from the gate` + WARN missing arm | the d8 plan is ~2× the d7 plan; recorded as `timeout` at 900.63 s |
| `p8192-L64-i32-N1024` has no `evoxc-specialized-numpy` record | not gate-visible (no snapshot lists it) | the per-tree fallback timed out at 900.14 s through 6 544 structures |
| both long-tree cells have no default-cap `evoxc-dense-iree` record | not gate-visible (same reason) | OOM at the cgroup cap, twice at d7 (§7.2) |
| 20 per-workload digest splits | WARN `N distinct sha256 across arms … (allowed by --allow-digest-mismatch)` | `evoxc-batch-iree` forms its own group on `batch_forward`/`forward` (non-bitwise value path), `evoxc-specialized-numpy`/`torch-*` form the plain-torch group, `evoxc-dense-iree` its own on some cells, and `evoxc-dense-iree-raised` a fourth on the long-tree cells |

Without the three flags the gate reports those same declared items as failures, exactly as `CONTEXT.md` records for the earlier rounds.

### 8.3 Caveats

Noise on this box is ±10 % warm and "a factor of a few" cold; every column here is one measurement, in one window, and the band sweep's 3.4 % spread across three identical plans is the in-window estimate.
The batch cold numbers are single calls: they are the compile, and re-running them reproduces them to within a factor, not a percent.
Four provenance seams are worth carrying forward.
The band and pool rows ran on evoxc `be145ee` while the campaign and the three-cell economics column ran on `b0544d5`; the pool-4 baseline among the pool rows is the economics run, i.e. the only off-checkout row in that table.
The `d7` economics pair straddles the same seam (batch on `b0544d5`, dense on `be145ee`, 28 s apart); its dense warm call (105.7484 ms) agrees with the campaign's (105.6526 ms) to 0.1 %, so the seam does not move that comparison.
One driver invocation that wrote the `d8` non-batch records (03:34–03:39) has no committed snapshot; its records carry their own `env` block (tree `009d47b2`).
And the two OOM kills have no record at all — the only machine fact in this report whose evidence is the round's commit message.

### 8.4 Where this report and the round's brief disagree

The records win, and these are the four places:

* the `p4096-L64-i64-N1024` batch cold call is **200 735.2 ms**, not 206 997 ms — the latter is the probe's `--reps 3` reading of the same cell (`PROBE.md`);
* that cell's batch record carries all three workloads (`sr_fitness` 24.56, `batch_forward` 22.53, `forward` 5.11 ms), not the loss alone;
* the batch-vs-dense factors computed from medians are **6.20×** (`p256-L64-i32-N16384`), **3.13×** (`p1000-L64-i32-N16384`) and **4.21×** (`d7` raised), not 6.35 / 3.19 / 4.24;
* the cold-call compile attribution is the record's own `batch_compile_wall_ms` (**90.3–96.8 %** of the cold call), not the per-chunk timer sum divided by the pool width (79.6–84.4 %), because that sum counts concurrent work;
* the pool-8 trade reads the other way round from the brief: 8 223 is where **pool 4 overtakes pool 8**, not where pool 8 repays a dearer compile, so it does not by itself argue for keeping 4 in a compile-dominated loop (§6.3).

One evidence-layout fact also differs from the brief: the round wrote into **two** directories, `results/largescale/` (the campaign `main/` and the `probe/`) and `results/largascale/` (band, economics, economics-longtree, pool2, pool8 — a spelling slip, recorded in those drivers' own `out_dir`). §10.1 lists both and the reproduce block uses the paths the records carry.

## 9. What would have to change on the evoxc side

Ordered by what the records say is binding, not by what is easiest:

1. **Compile cost and scaling.** The device compile alone is 90.3–96.8 % of the cold call (the four `lower()` calls add another 2.0–26.7 %) and quadratic in emitted ops (§5.1), so at pop ≥ 2 048 the route's cost of entry is 48 s and at pop 4 096 it is 187 s; at 6 544 structures it does not fit a 15-minute cap. A cheaper lowering or a persistent on-disk artifact cache keyed by the plan content is the only thing that makes this route usable at pop ≥ 2 048 — nothing else in the round moves the cold column by more than 2×.
2. **The merge/band shape.** One exported dispatch per band caps a plan at 15 bands on ROCm, and at pop 2 048 the model asks for 27 per chunk, so the plan is clamped and the band size becomes unobservable (§6.1). The warm call is then 73–90 % device `run` with four separate upload/run/read-back trios behind it; only a wider-band or flattened dispatch with fewer trios moves warm, and this round measured that the flat merge is *worse* (3.1×), so the lever is the dispatch, not the partition.
3. **The per-chunk trios that make pool > 4 cost warm.** Pool 8 is 18.4 s cheaper cold and 2.24 ms dearer warm at pop 2 048, so its advantage expires after 8 223 warm calls of one population (§6.3) — and a compile-dominated loop sits on its side of that crossover, which is why the pool width for a turnover loop is still open. The same mechanism is what made chunking a cold-only lever at pop 256. A single-executable multi-stream path (compile the chunks concurrently, upload/run them as one binding) would remove the trade outright.
4. **The content-addressed plan key in a turnover loop.** 3 attempts and 0 hits per loop, because every generation presents a new multiset (§5.3). Reuse is only reachable if the *programs* repeat — which is the fixed-population regime, not a GP loop.
5. **Library-side, not evoxc-side, but it decides who wins at `N` ≥ 4 096:** dense's per-call cost rises with rows (5.82 → 78.12 ms at pop 256 for 4× rows) and with population (11.06 → 37.06 → 76.23 ms), and `auto`'s `EVOGP_DENSE_AUTO_MAX_ROWS` = 2 048 routes every large-row cell to a fallback dense beats by 1.4–8.7× (§7.1). A rows-aware `auto` rule and a dense plan that does not materialize `pop × tree_len × rows` would both move the headline more than any batch knob.

## 10. Evidence layout, gate and reproduce commands

### 10.1 What is where

| path | contents |
|---|---|
| `results/largescale/main/` | 57 campaign records over 10 cells (arm records + `campaign.json` + the five group snapshots `campaign-g1-nonbatch.json`, `campaign-g1-nonbatch-i64.json`, `campaign-g2-raised.json`, `campaign-g3a-batch-eval{,-d7,-p4096}.json`, `campaign-g3b-batch-gpstep.json`, `campaign-g3c-batch-d8.json`) |
| `results/largescale/main/RENDERED.md` | the `summarize.py` render of that directory (this round's tables, machine-generated) |
| `results/largescale/probe/` | the compile-feasibility probe: 15 records + five `campaign-probe*.json` snapshots + `PROBE.md` (the round's own reading of the probe) |
| `results/largascale/economics/` | three cells × {batch, dense}: cold call, warm medians, 3-generation `StandardPipeline.step()` loop, ten-stage splits, `economics.json` |
| `results/largascale/economics-longtree/` | the same pair for `p2048-L128-i32-N1024-d7` against raised-cap dense |
| `results/largascale/band/` | the band sweep: `b8`/`b16`/`b32`/`k15`/`flat` at pop 2 048 at pool 1, `sweep.json`, and `aborted-3workload-sweep.json` |
| `results/largascale/pool2/`, `results/largascale/pool8/` | the compile-pool width sweep at pop 2 048 (`EVOGP_BATCH_COMPILE_WORKERS` 2 and 8; 1 and 4 come from the band sweep's `b16` and the economics run) |

`results/largescale/main` is the only `check.py`-gateable unit; the economics and band-sweep directories carry different schemas and are read directly (their drivers' `configs[]` rows are their own gate).

### 10.2 Reproduce

From the repository root, one driver invocation per group, `EVOXC_DISABLE_LOCATIONS=1`, `PY=/tmp/rocvenv/bin/python` (the box's torch/evoxc venv), one config at a time:

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1

# G1 — non-batch arms on the seven i32 cells and the d7 long-tree cell (900 s cap; the d8 non-batch triple
#      was a separate invocation whose snapshot campaign.json overwrote)
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 \
  --note "large-scale campaign: non-batch arms" \
  --cells p2048-L64-i32-N1024,p4096-L64-i32-N1024,p8192-L64-i32-N1024,p256-L64-i32-N4096,p256-L64-i32-N16384,p1000-L64-i32-N4096,p1000-L64-i32-N16384,p2048-L128-i32-N1024-d7 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward,gp_step

# G1 i64 — the input-width cell
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 \
  --note "large-scale campaign: non-batch arms" --cells p4096-L64-i64-N1024 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward,gp_step

# G2 — raised-cap dense on the two long-tree cells
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 \
  --note "large-scale campaign: raised-cap dense on the long-tree cells" \
  --cells p2048-L128-i32-N1024-d7,p2048-L256-i32-N1024-d8 --arms evoxc-dense-iree-raised \
  --workloads sr_fitness,batch_forward,forward,gp_step

# G3a/G3b — batch on the five feasible cells, then gp_step on two of them
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 --resume \
  --note "large-scale campaign: batch eval" \
  --cells p256-L64-i32-N4096,p1000-L64-i32-N4096,p1000-L64-i32-N16384,p4096-L64-i32-N1024,p4096-L64-i64-N1024,p2048-L64-i32-N1024,p2048-L128-i32-N1024-d7 \
  --arms evoxc-batch-iree --workloads sr_fitness,batch_forward,forward
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 --resume \
  --note "large-scale campaign: batch gp loop" --gp-reps 3 \
  --cells p2048-L64-i32-N1024,p256-L64-i32-N16384 \
  --arms evoxc-batch-iree --workloads sr_fitness,batch_forward,forward,gp_step

# G3c — the declared-infeasible d8 attempt (recorded as timeout, never retried)
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/main --timeout 900 \
  --note "large-scale campaign: batch d8 (declared-infeasible attempt)" \
  --cells p2048-L256-i32-N1024-d8 --arms evoxc-batch-iree --workloads sr_fitness

# economics: three cells, batch vs dense, 3-generation loop
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p2048-L64-i32-N1024,p1000-L64-i32-N16384,p256-L64-i32-N16384 \
  --routes batch,dense --generations 3 --warmup 2 --reps 5 --seed 0 --timeout 1200 \
  --tree-root src --evoxc-root /root/evoxc \
  --out-dir test/benchmark/evoxc/results/largascale/economics \
  --note "large-scale stage split + compile economics"
# economics: the node-count axis against raised-cap dense
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p2048-L128-i32-N1024-d7 --routes batch,dense --generations 3 --timeout 1500 \
  --knob EVOGP_DENSE_MAX_TREE_LEN=128 --tree-root src --evoxc-root /root/evoxc \
  --out-dir test/benchmark/evoxc/results/largascale/economics-longtree \
  --note "large-scale d7: node-count axis, batch vs raised dense"
# pool widths 2 and 8 (1 and 4 come from the sweep / the economics run)
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p2048-L64-i32-N1024 --routes batch --generations 1 --timeout 1200 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=2 --tree-root src --evoxc-root /root/evoxc \
  --out-dir test/benchmark/evoxc/results/largascale/pool2 \
  --note "large-scale compile-pool width 2 (pool-4 default baseline: 48.3 s cold / 12.79 ms warm)"
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p2048-L64-i32-N1024 --routes batch --generations 1 --timeout 1200 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=8 --tree-root src --evoxc-root /root/evoxc \
  --out-dir test/benchmark/evoxc/results/largascale/pool8 \
  --note "large-scale compile-pool width 8 (pool-4 default baseline: 48.3 s cold / 12.79 ms warm)"

# band size / cap pin at pop 2048 — sr_fitness only, pool pinned to 1 by the band-partition configs
$PY test/benchmark/evoxc/band_sweep.py \
  --cells p2048-L64-i32-N1024 --configs b8,b16,b32,k15,flat \
  --workloads sr_fitness --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /root/evoxc \
  --out-dir test/benchmark/evoxc/results/largascale/band \
  --note "large-scale band sweep (pop 2048): sr_fitness only -- the band-partition configs pin COMPILE_WORKERS=1 and a single-body compile at pop 2048 is ~13 min per workload, so the 3-workload schedule cannot fit"
```

Then render and gate:

```bash
python3 test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/largescale/main \
  --out test/benchmark/evoxc/results/largescale/main/RENDERED.md
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/largescale/main \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# -> 57 records / 1530 checks / 0 failures
```

### 10.3 Coverage statement

Measured: ten cells × {`torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-dense-iree`, `evoxc-specialized-numpy`} on every cell the route could run, `evoxc-dense-iree-raised` on the two long-tree cells, `evoxc-batch-iree` on eight of the ten cells (warm on all eight, `gp_step` on two), cold and warm and the ten-stage split and the 3-generation loop on four cells, and the band/pool knobs at pop 2 048.
Not measured, with reasons: `evoxc-specialized-iree` anywhere (272–284 ms of compile per structure), the batch route at pop 8 192 and on the d8 cell (compile infeasible), `evoxc-specialized-numpy` at pop 8 192 (900 s cap), the default-cap dense arm on both long-tree cells (OOM), `EVOGP_BATCH_ALIGNMENT` / `EVOGP_BATCH_SWITCH_OPS` / `EVOGP_BATCH_BANDS=auto` at this scale, and any `CustomLoss`/multi-output or robotics workload — the route has still never been timed there.
