# Round 3: the refreshed cold / warm / loop columns, the compile-pool default and the band cap (`results/batch_opt_round3/`)

Round 3 re-prices the shipped `EVOGP_EVAL_MODE=batch` default (band size 16, `EVOGP_BATCH_COMPILE_WORKERS` 4, chunked) against a pool-1 single-body control and against `dense`, on the same window and the same populations as round 2, and it settles two knob questions with numbers instead of prose: the compile-pool width (`EVOGP_BATCH_COMPILE_WORKERS`) and the band size (`EVOGP_BATCH_BAND_SIZE`, pinned at the rocm binding cap).
It adds three pieces of new information: an upstream-refreshed cold column (`/tmp/evoxc_r4`), a pop-1000 cold/loop reading on the chunked composition, and a long-tree (`-d7`) pool trade.
The decision of record is that **nothing on the shipped path changes** — pool stays 4, band size stays 16, `EVOGP_BATCH_BANDS=auto` and `EVOGP_BATCH_CHUNK_BANDS` keep their round-2 state — and this document is the evidence for that.
`BATCH.md` §15 is the same round in report form; every number below is read off a committed record whose path is named next to it.

## 1. Environment and protocol

Box and stack, identical to round 2 except for the evoxc snapshot: AMD Instinct MI300X VF (ROCm 7.1, `torch 2.13.0+rocm7.1`), `/tmp/rocvenv/bin/python` 3.12.3, iree-base-compiler/runtime 3.11.0, `evoxc` read from the **read-only `/tmp/evoxc_r4` snapshot** (`evoxc_sha 2b289f034f6f8e16b6c9b55932dfed76cc54690e`) — the round-2 snapshot `/tmp/evoxc_r2` is not used anywhere in this round.
`EVOXC_DISABLE_LOCATIONS=1` on every child (≈4× faster tracing, not part of the engine's cache key), recorded in `protocol.evoxc_disable_locations` of every driver record.
One process per `(cell, route)` for `batch_economics.py` and one process per `(cell, config)` for `band_sweep.py`: the merged-graph cache, the evoxc compile cache and the executable memo are all per process, so a loop inside one process would measure warm caches only.
`band_sweep.py` calls `campaign.guard` before every config and records its verdict (`state.gpu_pids.idle`, `state.contaminated`, `state.waited_s`); `batch_economics.py` has **no** idle-device guard, so its runs were preceded by a hand check of `rocm-smi --showpids` and kept strictly sequential — the same protocol gap §14.6 records, inherited unchanged.
`--seed 0` and `common.build_workload` give the **same populations as round 2**, which is what makes the cross-round cold comparison meaningful: `structure_signature` `ad11cb66ffc21ac8` at `p256-L32` (225 deduped functions), `0a0dee3fec0cfb9c` at `p256-L64` (232), `18d209693a5a7e9e` at `p64-L64-i32-N64` (63, i.e. 64 on the per-row plan) — identical to `results/batch_opt_round2/composition/<cell>/b16-cw4.json`, and carried in each round-3 record's own `structures` block (`distinct_structures`, `tree_len_max`, `structure_signature`).

Evidence layout (33 committed JSON files = 26 measurement records + 4 `economics.json` + 2 `sweep.json` + 1 validation record):

| item | directory | records | driver |
|---|---|---:|---|
| 1 | `economics/item1/` | 9 (3 cells × {`batch`, `batch-b16-banded`, `dense`}) | `batch_economics.py` |
| 1 cross-check | `economics/crosscheck-b16-chunked/` | 2 (`batch-b16-chunked` at the two pop-256 cells) | `batch_economics.py` |
| 2 | `item2/long-tree/` | 2 (`p256-L128-i32-N1024-d7` × {`batch`, `batch-b16-banded`}) | `batch_economics.py` |
| 3 | `item3/p256/`, `item3/p64/` | 8 + 4 (`band_sweep.py` sheets) | `band_sweep.py` |
| 4 | `item4/p1000-L32-i8-N1024/` | 1 (`batch`, shipped default) | `batch_economics.py` |
| validation | `validation/validation-batch.json` | 8 cases (4 cells × {`arith`, `rich`}) × 15 checks | `validate.py` |

Every `economics.json` config carries `status: "ok"` (9 + 2 + 2 + 1 = 14/14) and both `sweep.json` files read `failures: 0` / `skipped: 0` with all 8 + 4 configs `ok`; the economics **records themselves** carry no `status` key (that is the `evogp-bench-evoxc-v1-batch-economics` schema — the verdict lives in the driver's `configs[]`), while every sweep record carries `status: "ok"`.

## 2. Item 1 — the shipped default, the pool-1 control and dense in one window

Three cells (`p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p64-L64-i32-N64`), three routes, `--generations 8 --warmup 2 --reps 5 --seed 0`, one window.
`batch` = the shipped default with **no** `EVOGP_BATCH_*` knob set at all (`route_knobs` shows `source: module` for band size 16, pool 4 and `chunk_bands off`), i.e. it times the policy, not a spelling.
`batch-b16-banded` = band size 16 with `EVOGP_BATCH_COMPILE_WORKERS=1` (`source: route`), the single-body control — the *same* band partition, one compiled body.
`dense` = `evoxc-dense-iree` through the same public `Forest.SR_fitness`.

### 2.1 Warm per-call cost (the economics records price `sr_fitness` only)

| cell | batch, shipped default (ms) | pool-1 control (ms) | dense (ms) | batch ÷ dense | pool 4 ÷ pool 1 |
|---|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 3.7250 (min 3.6478) | 1.9941 (1.9743) | 1.1015 (1.0945) | **3.382** (0.296 by §14.2's reference÷batch convention) | 1.868 |
| `p256-L64-i32-N1024` | 3.7720 (3.7386) | 2.2267 (2.1871) | 2.0560 (2.0281) | **1.835** (0.545) | 1.694 |
| `p64-L64-i32-N64` | 1.1141 (1.0866) | 1.1494 (1.1223) | 1.4378 (1.4252) | **0.775** (1.291) | 0.969 |

Provenance: `economics/item1/<cell>/{batch,batch-b16-banded,dense}.json`, `sr_fitness.median_ms` / `.min_ms`.
Two readings, one sentence each: the shipped default is **1.69–1.87×** dearer per warm `sr_fitness` call than the pool-1 control at the two pop-256 cells (four separate upload / run / read-back trios per call), and at `p64` the plan does not chunk (`chunk_count 1`, `chunk_sizes [63]`) so the two are the same call (0.969, i.e. inside noise).
The economics records price only the loss path; the three-workload reading at the same pool widths is item 3's sweep (§4 below, `b16` vs `b16-cw4`), which is the like-for-like source for `batch_forward` and the per-row `forward`.

### 2.2 Cold call (`first_call_ms` of the `sr_fitness` plan), this round against round 2 (§14.3)

| cell | batch, shipped default | pool-1 control | dense | round-2 default (§14.3) | round-2 pool-1 (§14.3) | Δ default | Δ pool 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | **4 064.48** | 10 725.86 | 3 355.83 | 4 180.63 | 11 049.41 | **−2.78 %** | **−2.93 %** |
| `p256-L64-i32-N1024` | **4 071.43** | 11 957.38 | 6 526.78 | 4 422.39 | 12 590.61 | **−7.94 %** | **−5.03 %** |
| `p64-L64-i32-N64` | **2 274.46** | 2 268.96 | 6 554.88 | 2 284.23 | 2 298.69 | **−0.43 %** | **−1.29 %** |

Provenance: this round from `economics/item1/<cell>/{batch,batch-b16-banded,dense}.json` `first_call.first_call_ms`; round 2 from `results/batch_opt_round2/composition/<cell>/b16-cw4.json` and `results/batch_opt_round2/auto/<cell>/b16.json` (verified against those records, not only against the §14.3 table).
Against dense the cold column splits: **1.211×** at `p256-L32` (batch dearer), **0.624×** at `p256-L64` and **0.347×** at `p64` (batch cheaper).
Dense's cold call tracks the trace stride, not the population — 3 355.83 ms at L32 against 6 526.78 / 6 554.88 ms at L64 (p256 and p64 alike) — which is why the pop-64 cell is the one where batch's cold call is 2.88× cheaper.

**The cross-check reproduces the default's plan shape.** Pinning the shape explicitly (`batch-b16-chunked` = `EVOGP_BATCH_BAND_SIZE=16` + `EVOGP_BATCH_CHUNK_BANDS=1` + pool 4) yields the identical plan — `compile_workers 4`, `chunk_count 4`, `chunk_sizes [57, 56, 56, 56]` / `[58, 58, 58, 58]`, `chunk_bands [4, 4, 4, 4]` — and cold 4 054.18 / 4 070.32 ms against the unpinned default's 4 064.48 / 4 071.43 ms (**−0.25 % / −0.03 %**), warm 3.6255 / 3.8977 ms against 3.7250 / 3.7720 (**−2.7 % / +3.3 %**, i.e. agreement inside this box's noise).
Provenance: `economics/crosscheck-b16-chunked/<cell>/batch-b16-chunked.json` against `economics/item1/<cell>/batch.json`.
So the shipped default's measured column **is** the banded+chunked composition §14.2 identified, and no knob spelling is doing hidden work.

### 2.3 Where the cold call goes (the ten-stage split, `first_call.stage_ms`)

| cell / route | decode | dedup | trace | merge | lower | compile (sum over jobs) | compile wall / jobs | load | upload | run | gather | cold total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32` default (pool 4) | 4.64 | 0.16 | 154.23 | 227.02 | **470.05** | 12 735.60 | 3 630.75 / 4 | 281.61 | 41.14 | 3.14 | 0.82 | 4 064.48 |
| `p256-L32` pool 1 | 4.64 | 0.16 | 155.21 | 233.12 | **98.83** | 10 053.28 | 10 264.44 / 1 | 112.26 | 63.62 | 1.97 | 0.33 | 10 725.86 |
| `p256-L64` default (pool 4) | 4.63 | 0.15 | 148.50 | 240.87 | **376.02** | 12 926.52 | 3 627.39 / 4 | 296.91 | 43.51 | 2.98 | 0.83 | 4 071.43 |
| `p256-L64` pool 1 | 4.66 | 0.16 | 154.00 | 271.06 | **97.82** | 11 240.05 | 11 454.15 / 1 | 116.22 | 68.51 | 2.05 | 0.37 | 11 957.38 |
| `p64` default (pool 4, inert) | 1.29 | 0.04 | 29.04 | 95.00 | 31.16 | 1 963.33 | 2 078.05 / 1 | 83.50 | 68.78 | 0.81 | 0.34 | 2 274.46 |
| `p64` pool 1 | 1.29 | 0.05 | 29.10 | 99.15 | 31.66 | 1 958.04 | 2 071.36 / 1 | 81.61 | 65.66 | 0.79 | 0.36 | 2 268.96 |

Provenance: `economics/item1/<cell>/<route>.json` → `first_call.stage_ms` (per-stage `_total_ms`, the compile block also `batch_compile_wall_ms` / `batch_compile_jobs`); `dense` records no stage block (it owns no merge), so its cold call is not decomposable this way.
The compile column **sums** the per-chunk timers, which run concurrently, so on a chunked row the stage sum exceeds the call (313–317 %); the wall column (`batch_compile_wall_ms`) is the comparable one and it is **89.1–89.3 %** of a chunked cold call and **95.7–95.8 %** of a pool-1 cold call at pop 256, **91.3 %** at `p64`.
`lower` — the MLIR-construction stage the upstream speedups target — is **0.8–0.9 %** of a pool-1 cold call at pop 256 and **9.2–11.6 %** of the chunked one; §5 attributes the round's cold win against exactly those two tallies.

### 2.4 The generation loop (`StandardPipeline.step()` × 8)

| cell | route | `generation_hit_rate` | `generations_with_compile` | distinct signatures | `median_warm_eval_ms` | `median_cold_eval_ms` | `Σ total` | `Σ eval` | `compile_ms_share` | `compile_share.share` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32` | batch (default) | 0.125 | 7/8 | 8 | 3.78 | **2 600.93** | 19 626.71 | 19 339.99 | 0.9863 | **3.0358** (>1) |
| `p256-L32` | pool 1 | 0.125 | 7/8 | 8 | 2.05 | **5 481.92** | 44 176.67 | 43 886.21 | 0.9938 | 0.9287 |
| `p256-L32` | dense | 1.0 | 0/8 | 8 | 2.89 | — | **302.48** | **22.28** | 0.0 | — |
| `p256-L64` | batch (default) | 0.125 | 7/8 | 8 | 3.81 | **2 894.58** | 22 346.79 | 22 056.43 | 0.9878 | **3.1381** (>1) |
| `p256-L64` | pool 1 | 0.125 | 7/8 | 8 | 2.28 | **6 793.52** | 54 356.86 | 54 061.80 | 0.9949 | 0.9297 |
| `p256-L64` | dense | 1.0 | 0/8 | 8 | 4.24 | — | **312.48** | **32.42** | 0.0 | — |
| `p64` | batch (default) | 0.125 | 7/8 | 8 | 1.15 | **588.22** | 5 587.44 | 5 293.92 | 0.9506 | 0.8456 |
| `p64` | pool 1 | 0.125 | 7/8 | 8 | 1.18 | **601.34** | 5 622.49 | 5 329.54 | 0.9511 | 0.8461 |
| `p64` | dense | 1.0 | 0/8 | 8 | 2.90 | — | **313.26** | **23.11** | 0.0 | — |

Provenance: `economics/item1/<cell>/<route>.json` → `summary` (all columns except the last, which is `summary.compile_share.share`).
Two different "compile share" numbers are recorded and must not be confused: `summary.compile_ms_share` is the **wall time of the generations that recompiled over the loop's total wall time** (an upper bound on attribution, ≤ 1 by construction, 0.951–0.995 here), while `summary.compile_share.share` divides the **sum of the per-chunk compile stage timers** by the loop's wall time and therefore reads **> 1** on a chunked run (3.04 / 3.14) — that block states it in its own note (`numerator_kind: parallel-stage-counter-sum`, `denominator_kind: loop-wall-ms`) and it is a concurrency-weighted counter ratio, never wall-clock attribution.
The hit rate is the loop's structural fact: generation 1 evaluates the population whose graph the timed first call already compiled, and every later generation presents a **new structure multiset**, so 7 of 8 generations recompile and the multiset repeats **never** (8 distinct signatures in 8 steps).
Dense, in the same window, compiles nothing, replaces the same 8 populations in 302.48 / 312.48 / 313.26 ms and spends 22.28 / 32.42 / 23.11 ms of that in evaluation.

### 2.5 What is re-measured this round, and what is carried over from round 2

Re-measured (fresh records, `/tmp/evoxc_r4`): the **shipped default** (band 16, pool 4, chunked) at the three cells; the **pool-1 control** at the same three cells; **dense** at the same three cells (cold, warm, 8-generation loop); the **batch ÷ dense ratio** on all three; the **ten-stage split** of the cold call; **pop 1000** (item 4); the **long-tree pool trade** at `-d7` (item 2); and the **band-cap pins** `k15` / `k7` in pool-1 and pool-4 form (item 3).
Carried over, **not** re-measured: the `flat` arm (§4.3, §14.1/§14.3 — the flat cold column and the "flat can no longer chunk" finding), the `b8` / `b32` / `b64` band sizes (§14.1's minimax grid), `EVOGP_BATCH_BANDS=auto` and the whole cost-model analysis (§14.4, including the 14.2–14.7× overshoot and the non-population-stable K), the round-1 real-data references (`results/batch_opt_round1/`), and §11's long-tree crossover verdict — this round prices the pool at the long-tree niche, it does not re-run §11's dense variants.

## 3. Item 2 — the compile-pool decision (break-even = Δcold ÷ Δwarm)

`EVOGP_BATCH_COMPILE_WORKERS` trades cold for warm: the pool splits one merge into concurrent chunks (cheaper cold, four upload/run/read-back trios per warm call). The break-even is how many warm calls **of one population** a run must make before the dearer cold call is repaid.

| cell | cold pool 1 (ms) | cold pool 4 (ms) | Δcold (ms) | warm pool 1 (ms) | warm pool 4 (ms) | Δwarm (ms) | break-even, median | break-even, min | cold speedup pool 1 ÷ pool 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 10 725.86 | 4 064.48 | 6 661.38 | 1.9941 | 3.7250 | 1.7309 | **3 849** | 3 981 | 2.639 |
| `p256-L64-i32-N1024` | 11 957.38 | 4 071.43 | 7 885.95 | 2.2267 | 3.7720 | 1.5453 | **5 103** | 5 083 | 2.937 |
| `p64-L64-i32-N64` | 2 268.96 | 2 274.46 | **−5.51** | 1.1494 | 1.1141 | **−0.0353** | **n.a. — there is no trade** | n.a. | 0.998 |
| `p256-L128-i32-N1024-d7` (long trees) | 38 513.95 | 8 410.96 | 30 102.99 | 3.4851 | 5.0947 | 1.6096 | **18 702** | 20 648 | 4.579 |

Provenance: `economics/item1/<cell>/{batch,batch-b16-banded}.json` and `item2/long-tree/p256-L128-i32-N1024-d7/{batch,batch-b16-banded}.json` (`first_call.first_call_ms`, `sr_fitness.median_ms` / `.min_ms`); break-even is computed here as Δcold ÷ Δwarm, not quoted.
The pool buys **2.64× / 2.94× / 4.58×** on the cold call and costs **1.87× / 1.69× / 1.46×** on the warm one.
Repaying the cold difference needs **3 849 / 5 103 / 18 702 warm calls of a single population** (min-based 3 981 / 5 083 / 20 648) — and a GP loop presents a new population every generation, so in the workload the library ships, that repayment never begins.
At `p64` the pool is **inert on the `sr_fitness` and `batch_forward` plans** (`chunk_count 1`, `chunk_sizes [63]`, Δcold −5.51 ms, Δwarm −0.04 ms), so the one regime where the batch route beats dense warm (`p64`: 1.29× on `sr_fitness` this round, and 1.54× on `batch_forward` in §14.2's shipped-default column) is exactly the regime where flipping the default to 1 would buy nothing.

**Decision: `EVOGP_BATCH_COMPILE_WORKERS` stays 4** (`src/evogp/tree/batch_knobs.py::DEFAULT_COMPILE_WORKERS`, not touched by this round). The default pays 1.69–1.87× per warm call at the two pop-256 cells and 1.46× at the long-tree niche, and buys 2.64× / 2.94× / 4.58× cold; flipping it to 1 would make the cold call **10.7 s** (p256) / **38.5 s** (long trees) instead of 4.1 s / 8.4 s while the only warm-win regime (`p64`) would not change at all.
**Documented escape hatch, not a new default:** a workload that evaluates **one fixed structure multiset** more than ≈3.8k (pop 256, L32) / ≈5.1k (pop 256, L64) / ≈18.7k (long-tree d7) times should set `EVOGP_BATCH_COMPILE_WORKERS=1` explicitly and take the cheaper warm call; the aggregate-cold-sensitive case (one short run of a big population) should keep the shipped 4.

The long-tree loop rows make the same point structurally: generation 2 is a new multiset and recompiles, and chunking wins **that single recompile** too — 4 exec misses at **3 933.60 ms** total (pool 4) against 1 miss at **8 601.53 ms** (pool 1), i.e. **2.19×** — while both remain 3.9–8.6 s against the whole generation's 3 ms of operator work.
Provenance: `item2/long-tree/p256-L128-i32-N1024-d7/{batch,batch-b16-banded}.json` → `generations[1].{total_ms,eval_ms,batch_compile_misses}`; the 2-generation loop's `generation_hit_rate` is 0.5 by construction (1 of 2 generations recompiles) and is a 2-step check, not a loop claim.

## 4. Item 3 — the band cap: `EVOGP_BATCH_BAND_SIZE` stays 16

Four configs per cell, `--warmup 1 --reps 5 --seed 0`, three workloads in fixed order: the shipped band size 16 single-body (`b16`), the same band size through the pool (`b16-cw4`), and the pin at the binding cap — `k15` at the pop-256 cells, `k7` at `p64` where 15 bands would be illegal (`pop >= 8 * K` ⇒ `k15` needs 120 functions, `k7` needs 56) — in pool-1 and pool-4 form.

### 4.1 Warm per-call medians, all three workloads

| cell | config | `sr_fitness` | `batch_forward` | `forward` |
|---|---|---:|---:|---:|
| `p256-L32-i8-N1024` | `b16` (pool 1) | 2.1447 (2.1151) | 2.2008 (2.0740) | 0.8855 (0.8489) |
| | `k15` (pool 1) | 2.0186 (1.9929) | 2.2517 (2.1229) | 0.8248 (0.7835) |
| | `b16-cw4` (pool 4) | 3.6904 (3.5923) | 3.6519 (3.5824) | 2.2757 (2.2324) |
| | `k15-cw4` (pool 4) | 3.5575 (3.4442) | 3.4360 (3.3498) | 1.9313 (1.8956) |
| `p256-L64-i32-N1024` | `b16` (pool 1) | 2.2166 (2.1885) | 2.2424 (2.2242) | 0.8432 (0.8046) |
| | `k15` (pool 1) | 2.2195 (2.1776) | 2.2703 (1.9940) | 0.8211 (0.8168) |
| | `b16-cw4` (pool 4) | 3.8369 (3.7165) | 3.4693 (3.4549) | 2.0150 (1.9750) |
| | `k15-cw4` (pool 4) | 3.7600 (3.6600) | 3.4462 (3.3883) | 1.9342 (1.8930) |
| `p64-L64-i32-N64` | `b16` (pool 1) | 1.1463 (1.1362) | 0.8966 (0.8710) | 0.6885 (0.6584) |
| | `k7` (pool 1) | 1.1468 (1.0601) | 0.9080 (0.8948) | 0.6570 (0.6259) |
| | `b16-cw4` (pool 4) | 1.1187 (1.1078) | 0.9122 (0.8925) | 1.0774 (1.0583) |
| | `k7-cw4` (pool 4) | 1.1610 (1.1157) | 0.9027 (0.8667) | 1.0317 (1.0111) |

Provenance: `item3/p256/<cell>/<config>.json` and `item3/p64/p64-L64-i32-N64/<config>.json` → `workloads.<workload>.median_ms` with `.min_ms` in parentheses; `status: "ok"` on all 12 records.

### 4.2 Is the pin a different plan? At pop 256 it is the same merge; at `p64` it is not

| cell / workload | `b16` `band_sizes` | `b16` `emitted_ops` | pin `band_sizes` | pin `emitted_ops` | verdict |
|---|---|---|---|---:|---|
| `p256-L32` `sr_fitness` | `[15]×15` | 5 037 | `[15]×15` | 5 037 | identical plan, identical ops |
| `p256-L32` `batch_forward` | `[15]×15` | 4 153 | `[15]×15` | 4 153 | identical |
| `p256-L32` `forward` | `[17]×5 + [16]×10` | 4 540 | `[17]×5 + [16]×10` | 4 540 | identical, but `b16` asks a **size** (16 bands) and is **clamped** to 15 while `k15` asks the count (`band_requested explicit`, `band_clamped false`) |
| `p256-L64` `sr_fitness` | `[16]×7 + [15]×8` | 5 475 | `[16]×7 + [15]×8` | 5 475 | identical |
| `p256-L64` `batch_forward` | `[16]×7 + [15]×8` | 4 568 | `[16]×7 + [15]×8` | 4 568 | identical |
| `p256-L64` `forward` | `[17]×5 + [16]×10` | 5 861 | `[17]×5 + [16]×10` | 5 861 | identical, `b16` clamped as above |
| `p64` `sr_fitness` | `[16,16,16,15]` (K = 4) | 1 393 | `[9]×7` (K = 7) | 1 463 | **different plan** |
| `p64` `batch_forward` | `[16,16,16,15]` | 1 155 | `[9]×7` | 1 219 | **different plan** |
| `p64` `forward` | `[16,16,16,16]` | 1 493 | `[10,9,9,9,9,9,9]` | 1 687 | **different plan** |

Provenance: `workloads.<workload>.band_report` (`bands`, `band_sizes`, `emitted_ops`, `band_requested`, `band_clamped`) of the same 12 records; `n_functions` is 225 / 225 / 245 at `p256-L32`, 232 / 232 / 245 at `p256-L64` and 63 / 63 / 64 at `p64` (for `sr_fitness` / `batch_forward` / `forward`; the per-row plan carries one function more).
At pop 256 there is therefore nothing for the pin to win or lose: the pin and the shipped size reach the **same final band shape and the same `emitted_ops`**, and the only difference is bookkeeping (the size model asks 16 and is clamped, the pin asks 15 and is not).

### 4.3 Same-window ratios (pin ÷ shipped size)

| cell | workload | pool 1 (median) | pool 1 (min) | pool 4 (median) | pool 4 (min) |
|---|---|---:|---:|---:|---:|
| `p256-L32` | `sr_fitness` | −5.9 % | −5.8 % | −3.6 % | −4.1 % |
| `p256-L32` | `batch_forward` | +2.3 % | +2.4 % | −5.9 % | −6.5 % |
| `p256-L32` | `forward` | −6.9 % | −7.7 % | −15.1 % | −15.1 % |
| `p256-L64` | `sr_fitness` | +0.1 % | −0.5 % | −2.0 % | −1.5 % |
| `p256-L64` | `batch_forward` | +1.2 % | **−10.4 %** | −0.7 % | −1.9 % |
| `p256-L64` | `forward` | −2.6 % | +1.5 % | −4.0 % | −4.2 % |
| `p64` | `sr_fitness` | **+0.04 %** | −6.7 % | +3.8 % | +0.7 % |
| `p64` | `batch_forward` | **+1.3 %** | +2.7 % | −1.0 % | −2.9 % |
| `p64` | `forward` | **−4.6 %** | −4.9 % | −4.2 % | −4.5 % |

Provenance: computed here from the medians and mins of §4.1 (no ratio is stored in the records).
The reading: on the pop-256 **same-merge** pairs (pool 1) the spread is **−6.9 % … +2.3 %** with a single **−10.4 %** min outlier (`p256-L64 batch_forward`), i.e. this box's cross-config noise band and not a band-policy effect; at `p64`, where the plan really changes, the result is **+0.04 % / +1.3 % / −4.6 %** on medians — still inside the ~5 % cross-config noise floor.
The one row outside that band is `p256-L32 forward` through the pool (−15.1 %), on an **identical** plan (`chunk_bands [4,4,4,4]`, `emitted_ops` 4 579 both configs) — the per-row `forward` workload is the band-sensitive column §14.1 already flags, and its pool-4 absolute values (2.28 / 1.93 ms) are far from the pool-1 ones (0.89 / 0.82 ms), so it is not a partition comparison.
**No pin beat the shipped size anywhere in the window**, so `EVOGP_BATCH_BAND_SIZE` stays 16 (no `src/` change) and the policy still satisfies the rocm binding limits (band ≥ 8 functions, `K ≤ 15`).

### 4.4 The oracle deviation is identical in every config — the partition is scheduling, not numerics

| cell | `sr_fitness` (abs / rel) | `batch_forward` (abs / rel) | `forward` (abs / rel) |
|---|---|---:|---:|
| `p256-L32-i8-N1024` | 6.0 / 4.74e-07 | 5.86e-03 / 1.13e-03 | 0 / 0 (**bitwise**) |
| `p256-L64-i32-N1024` | 32.0 / 7.76e-07 | 7.81e-02 / 1.14e-05 | 1.19e-07 / 8.01e-08 |
| `p64-L64-i32-N64` | 0.25 / 1.62e-07 | 6.10e-05 / 1.68e-07 | 1.19e-07 / 8.01e-08 |

Provenance: the top-level `validation.<workload>.{max_abs,max_rel,bitwise_equal,nan_mismatch,inf_mismatch}` of the 12 sweep records (oracle `src/evogp/tests/reference_eval.py`, 16 trees per workload) — **the same four numbers in all four configs of every cell/workload (9/9 rows)**, no NaN and no inf mismatch anywhere.
Value digests are the weaker witness and split by design, at **≤ 5.8e-11** relative `abs_sum`:
`sr_fitness` is **one digest group per cell in every config** (`e1021d86…` / `7cccccd5…` / `b6584500…`);
at the two pop-256 cells the split is the **pool** (`batch_forward` `a4604c19…` for `b16,k15` versus `3eec7baa…` for `b16-cw4,k15-cw4`; `forward` `562b2714…` versus `7378b6e1…` at L32, `d8ad0c04…` versus `563fc965…` at L64) — the chunked composition's documented non-bitwise value path (§12, `VALIDATION.md`);
at `p64` the split is the **band count** on `batch_forward` (`4919287125…` for `b16` and `b16-cw4` versus `85adf166…` for `k7` and `k7-cw4`), while its `sr_fitness` and `forward` digests are identical in all four configs.
Provenance: `workloads.<workload>.digest.sha256` / `.abs_sum` of the same 12 records.

## 5. Item 4 — pop 1000 on the shipped default (new information)

`p1000-L32-i8-N1024` (1 000 trees, `tree_len_max` 31, mean 16.21, **817 deduped functions**), shipped default, `--generations 2 --warmup 1 --reps 5`, `--timeout 1500`.

- Cold call **15 517.88 ms**; plan `compile_workers 4` / `chunk_count 4` / `chunk_sizes [205, 204, 204, 204]`.
- Stage split: decode 15.37, dedup 0.62, trace 557.71, merge 967.07, **lower 1 054.29**, compile **sum 48 931.70 over 4 jobs** with `batch_compile_wall_ms` **13 921.97** (3.51× over the serial sum, i.e. 88 % parallel efficiency), load 298.33, upload 40.74, run 6.58, gather 0.86 — the compile wall is **89.72 %** of the call and `lower` is 6.79 % of it.
- Warm `sr_fitness` **6.9384 ms** (samples 6.8962 / 6.9703 / 6.9334 / 7.1555 / 6.9384; min 6.8962).
- 2-generation loop: generation 1 hits the graph the timed first call compiled (270.10 ms, eval 6.92), generation 2 is a new multiset and pays **4 exec misses / 9 552.97 ms** (eval 9 549.94); `Σ total` 9 823.07 ms, `Σ eval` 9 556.85 ms, `compile_ms_share` 0.9725, `compile_share.share` 3.1662 (>1, the pool-weighted stage ratio), `generation_hit_rate` 0.5 (2-step check).

Provenance: `item4/p1000-L32-i8-N1024/p1000-L32-i8-N1024/batch.json` (`first_call.*`, `sr_fitness.*`, `summary.*`, `generations[*]`); the driver's `configs[0]` reads `status ok`, `wall_s 34.14`.
**This is new information, not a speedup measurement:** round 2 never priced pop 1000 on the chunked composition — its 91.0–162.0 s figures are the **single-executable** (`EVOGP_BATCH_COMPILE_WORKERS=1`) path of §6/§14.3, so 15.5 s here is the shipped default's own number, not an improvement over those.
The round's pre-registered 4-minute bound on a single cold compile **did not trigger** (15.5 s against that bound), so the cell is measured in full and nothing was dropped — the driver's only config for it reads `status: "ok"` / `wall_s` 34.14 against a `protocol.timeout_s` of 1 500 (`skipped` is not a field of the `evogp-bench-evoxc-v1-batch-economics` schema, so there is no skipped counter to quote).

## 6. Numeric validation of the round (`validation/validation-batch.json`)

`validate.py` with `EVOGP_EVAL_MODE=batch` (shipped default knobs), `--func-sets arith,rich --max-oracle 64 --seed 0`, 4 cells × {`arith`, `rich`} = 8 cases × 15 checks (oracle, `torch-per-tree` and `torch-batched` arms cross-diffed).
`arith`, engine versus the plain-torch oracle: `sr_fitness` 256.0 / 4.79e-07 at `p256-L32` and `p1000-L32`, 64.0 / 7.76e-07 at `p256-L64`, 64.0 / 4.94e-07 at `p64`; `batch_forward` 1.56e-02 / 1.13e-03, 7.81e-02 / 2.38e-04, 1.46e-03 / 3.35e-06; `forward` 1.91e-06 / 1.64e-07 or smaller — no NaN or inf mismatch in any case.
`rich` (the transcendental set) is the documented approximation: `batch_forward` deviates up to **1.35 / 1.61 / 0.44 absolute** with relative figures that blow up where the reference magnitude is small (7.65 / 15.61 / 0.64), while `sr_fitness` and `forward` stay at ≤ 0.25 abs / ≤ 1.9e-07 rel — the same caveat §12 and the root `CONTEXT.md` carry for the iree transcendental lowering.
The economics/sweep records carry **no** `validation` block; this file is the round's oracle evidence, and the sweep records' own per-workload `validation` blocks (§4.4) are the config-level witness.

## 7. Attribution: what the upstream evoxc speedups did to this route's cold path

The `/tmp/evoxc_r4` snapshot's faster MLIR construction (`lower()`; the round's records note it) is visible in the stage split, and its size is bounded by where the cold call's time actually is:

| cell / route | cold round 2 (ms) | cold round 3 (ms) | Δ | `lower` round 2 → round 3 (ms) | `lower` speedup | compile wall round 2 → round 3 (ms) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32` pool 1 | 11 049.41 | 10 725.86 | −2.93 % | 296.61 → 98.83 | **3.00×** (−197.8) | 10 583.57 → 10 264.44 | −3.02 % |
| `p256-L32` default (chunked) | 4 180.63 | 4 064.48 | −2.78 % | 888.52 → 470.05 | **1.89×** (−418.5) | 3 733.14 → 3 630.75 | −2.74 % |
| `p256-L64` pool 1 | 12 590.61 | 11 957.38 | −5.03 % | 286.68 → 97.82 | 2.93× | 12 077.57 → 11 454.15 | −5.16 % |
| `p256-L64` default | 4 422.39 | 4 071.43 | −7.94 % | 985.88 → 376.02 | 2.62× | 3 966.10 → 3 627.39 | −8.54 % |
| `p64` pool 1 | 2 298.69 | 2 268.96 | −1.29 % | 63.50 → 31.66 | 2.01× | 2 106.40 → 2 071.36 | −1.66 % |
| `p64` default (inert) | 2 284.23 | 2 274.46 | −0.43 % | 63.38 → 31.16 | 2.03× | 2 095.67 → 2 078.05 | −0.84 % |

Provenance: round-3 `first_call.stage_ms` of `economics/item1/<cell>/<route>.json`; round-2 `workloads.sr_fitness.stage_first_call` of `results/batch_opt_round2/composition/<cell>/b16-cw4.json` (chunked rows) and `results/batch_opt_round2/auto/<cell>/b16.json` (pool-1 rows). Round 2's `batch_compile_wall_ms` / sum pair is 10 583.57 / 10 172.56 for the pool-1 L32 row (1 job), so the wall-to-wall and sum-to-sum deltas differ: **−3.02 %** wall against **−1.17 %** sum.
The honest ceiling: `lower` is **0.8–0.9 %** of a pool-1 cold call at pop 256 and **9.2–11.6 %** of a chunked one, while the device-code compile is **89.1–95.8 %** (95.7–95.8 % pool 1, 89.1–89.3 % chunked, 91.3 % at `p64`, 97.4 % pool 1 at the long-tree cell, 89.7 % at pop 1000).
A 1.9–3.0× faster `lower()` therefore moves a 4–12 s cold call by **0.4–7.9 %**, which is what was measured; the upstream speedups cannot fix this route's cold path because the time is not in the MLIR construction.
No number in this section is a projection: every cell is a measured pair of records.

## 8. Gate and status

Records: **26** measurement records (14 economics + 12 sweep) + 6 driver records + 1 validation file = 33 JSON files, every one `status: ok` where the schema carries a `status` (12/12 sweep records) and `configs[].status == "ok"` on all 14 economics configs.
Guards: every sweep record carries `guard.available true`, `guard.state.gpu_pids.idle true`, `guard.state.contaminated false`, `guard.state.waited_s 0.2`, `gpu_pids.pids []` — **12/12**, with `policies` and the driver's own `configs[].guard` agreeing; both `sweep.json` files read `failures: 0`, `skipped: 0`, `elapsed_s` 190.4 s (`item3/p256`) and 52.1 s (`item3/p64`).
`batch_economics.py` (items 1, 1-cross-check, 2 and 4) has **no idle-device guard** and its records carry no `guard` block: idleness was verified **by hand** with `rocm-smi --showpids` before each run and the runs kept strictly sequential (9 + 2 + 2 + 1 = 14 children), a weaker protocol than the sweeps'.
Gate runs (stdlib `check.py`, run for this document on the committed records):

| directory | records | checks | failures | failure classes |
|---|---:|---:|---:|---|
| `item3/p256` | 8 | 196 | 25 | 24 × `cache.compile_misses_first_call is None` (the sweep schema records the merged-graph cache, not the evoxc engine's first-call compile misses), 1 × missing `campaign.json` |
| `item3/p64` | 4 | 99 | 13 | 12 × the same, 1 × missing `campaign.json` |
| `economics/item1` | 9 | 62 | 16 | 9 × "no workloads recorded" (economics schema), 6 × "no digest anywhere in the cell" (only `sr_fitness` is digested), 1 × missing `campaign.json` |
| `economics/crosscheck-b16-chunked` | 2 | 18 | 7 | 2 × "no workloads recorded", 4 × "no digest", 1 × missing `campaign.json` |
| `item2/long-tree` | 2 | 16 | 5 | 2 × "no workloads recorded", 2 × "no digest", 1 × missing `campaign.json` |
| `item4/p1000-L32-i8-N1024` | 1 | 10 | 4 | 1 × "no workloads recorded", 2 × "no digest", 1 × missing `campaign.json` |

**A sweep or economics directory is not a `check.py` gateable unit** — exactly as §14.6 records for round 2: `check.py` reads the campaign *arm-record* schema (`<cell>/<arm>.json` with per-workload `samples_ms`, an `env` block and a `campaign.json`), while these records are the band-sweep / economics schemas.
Their internal verdicts are the ones that stand: `status: ok`, the stage-split closure, the guard block, the per-workload oracle `validation` and the `band_report` blocks.
The two failure classes are properties of the schemas, not of a config: the sweeps never touch `evoxc_engine.cache_stats()` (the batch route compiles through the merged-graph cache, which the records do carry as `batch_cache`), and the economics loop digests only its `sr_fitness` workload.

## Commands

Copy-paste runnable from the repository root; `--note` strings are verbatim from the records so a rerun reproduces the record's own note, and a re-measure always writes a **fresh `--out-dir`** (`results/batch_opt_round2/` is now refused by both drivers' `FROZEN_RESULT_DIRS`, as are `final`, `baseline`, `batch` — and `batch_opt_round1` for the sweep).

```bash
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r4:src:test/benchmark/evoxc
PY=/tmp/rocvenv/bin/python

# item 1 — shipped default vs pool-1 control vs dense, one window, 8 generations
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --routes batch,batch-b16-banded,dense \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/item1 \
  --note "round3 item1: shipped default (band 16, pool 4, chunked) vs pool-1 control (batch-b16-banded) vs dense, same window, evoxc_r4"

# item 1 cross-check — the pinned band16+CHUNK_BANDS=1+pool4 shape against the unpinned default
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 \
  --routes batch-b16-chunked \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/crosscheck-b16-chunked \
  --note "round3 item1 cross-check: pinned band16+CHUNK_BANDS=1+pool4 against the unpinned shipped default"

# item 2 — the long-tree (d7) pool trade, warm + cold + a 2-generation check
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L128-i32-N1024-d7 \
  --routes batch,batch-b16-banded \
  --generations 2 --warmup 2 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item2/long-tree \
  --note "round3 item2: long-tree niche (d7), pool-1 control vs the chunked default, warm+cold, 2-generation loop (not a loop claim)"

# item 3 — the band cap at pop 256: k15 vs the shipped size 16, pool 1 and pool 4
$PY test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 \
  --configs b16,b16-cw4,k15,k15-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p256 \
  --note "round3 item3: K=15 pinned at the rocm binding cap vs the shipped band size 16, pool-1 partitions and pool-4 compositions, same window"

# item 3 — the same at the fixed-population niche cell, where K=15 is illegal and k7 is the cap
$PY test/benchmark/evoxc/band_sweep.py \
  --cells p64-L64-i32-N64 \
  --configs b16,b16-cw4,k7,k7-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p64 \
  --note "round3 item3: K=7 pinned vs band size 16 at the fixed-population niche cell, pool-1 and pool-4, same window"

# item 4 — pop 1000, shipped default only
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p1000-L32-i8-N1024 --routes batch \
  --generations 2 --warmup 1 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item4/p1000-L32-i8-N1024 \
  --note "round3 item4: p1000-L32-i8-N1024 shipped default only, cold + one warm reading + stage split + 2-generation loop, faster evoxc lower()"

# the round's oracle sweep (arith + rich, four cells)
EVOGP_EVAL_MODE=batch $PY test/benchmark/evoxc/validate.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64,p1000-L32-i8-N1024 \
  --func-sets arith,rich --max-oracle 64 --seed 0 \
  --out test/benchmark/evoxc/results/batch_opt_round3/validation/validation-batch.json \
  --note "round3: the batch route (shipped default knobs, evoxc_r4) vs the plain-torch oracle on the item-1/item-4 cells"
```

Gate re-run (per directory; the counts of §8 are what these print):

```bash
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch_opt_round3/item3/p256 --allow-snapshot --allow-missing --allow-digest-mismatch
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch_opt_round3/item3/p64 --allow-snapshot --allow-missing --allow-digest-mismatch
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch_opt_round3/economics/item1 --allow-snapshot --allow-missing --allow-digest-mismatch
```
