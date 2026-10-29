# Round 1 — batch route: chunked compile, the per-stage split, plan-key reuse

**Measurement record.** Every number below is re-derived from the JSON in this directory (`base/`, `after_default/`, `after_w1/`, `after_w8/`, `after_dense/`, `campaign/`); nothing was re-measured for this report.
The route under test is the opt-in `EVOGP_EVAL_MODE=batch` merged-graph route (`src/evogp/tree/batch_engine.py`), which compiles ONE graph per population structure multiset.

## 1. What the round answers

1. **Cold** — does compiling the merged graph as several per-chunk executables in parallel (the compiled-graph pool, `EVOGP_BATCH_COMPILE_WORKERS`) cut the route's multi-second first call, and by how much?
2. **Where** — where does the cold call, and the warm call, actually spend its time (the ten-stage counter split: decode/dedup/trace/merge/lower/compile/load/upload/run/gather)?
3. **Reuse** — does the content-addressed plan key buy anything inside a real `StandardPipeline.step()` generation loop?

Answers, one line each (derivations in §3–§7):

- Cold at `p256-L32-i8-N1024` goes 11.07 s → 4.44 s (**2.49×**) at the shipped default of 4 workers and → 3.78 s (**2.93×**) at 7 chunks; the pre-round tree's own cold call is 11.07 s, so chunking — not any other change — owns that reduction.
- The pool still owns **90.6 %** of the cold call at 4 workers; everything else is 368 ms of host decode+trace+merge before it and 45.0 ms of device upload+run+gather after it.
- Warm pays for the cold win: 2.011 → 3.628 ms per call at p256 (**1.80×**), because each chunk is its own upload+run+gather; at one chunk the warm call is unchanged (2.013 ms, 1.00× of pre-round).
- **No measurable plan-key reuse in a GP loop**: 6 generations produce 6 distinct structure multisets, `batch_exec_hits` stays 0, and the identity-plan counters (9 hits / 6 misses / 6 entries) are identical to the pre-round tree.
- One new capacity limit surfaced: exec entries are per **chunk**, so `EVOGP_BATCH_CACHE_SIZE=8` stores two merged graphs at 4 chunks (1.14 graphs at 7 chunks).
- Net: at these two cells the batch route stays dominated by `dense+iree` in a generation loop (45.4× at p256, 15.9× at p64) whether the pool is on or off; chunking moves the cold number, not the verdict.

## 2. What was measured, and on which box

| column | directory | tree | route knob |
|---|---|---|---|
| base | `base/` | pre-round tree `619cbf738af0bf839085879193f163d383bd9fbc`, extracted outside the repo to `$TMPDIR/evogp_preopt_619cbf7` | none — single-executable compile path, no chunking, no stage timers |
| HEAD default | `after_default/` | HEAD | `EVOGP_BATCH_COMPILE_WORKERS` unset (module default, 4 workers) |
| HEAD W=1 | `after_w1/` | HEAD | `--knob EVOGP_BATCH_COMPILE_WORKERS=1` (the old single-executable control) |
| HEAD W=8 | `after_w8/` | HEAD | `--knob EVOGP_BATCH_COMPILE_WORKERS=8` → resolves to 7 chunks |
| dense | `after_dense/` | HEAD | comparison column, `EVOGP_EVAL_MODE=dense` (owns no batch counters: its `stage_ms` is absent, not zero) |
| campaign | `campaign/` | HEAD | campaign-format (`worker.py`) gate record: batch arm + both torch arms |

- Cells: `p256-L32-i8-N1024` (pop 256, tree_len 32, input_len 8, 1024 rows, **225** distinct tree structures, `structure_signature ad11cb66ffc21ac8`) and `p64-L64-i32-N64` (pop 64, tree_len 64, input_len 32, 64 rows, **63** structures, `18d209693a5a7e9e`).
- Protocol, identical in every column: `--generations 6`, `--warmup 2`, `--reps 5`, seed 0, one child process per (cell, route), strictly sequential, driver applies `EVOXC_DISABLE_LOCATIONS=1`.
- Every batch record resolves `eval_mode=batch`, `kind=batch`, `backend=iree`, `declines_to_specialized=false`, `mode_supported=true` — no silent fallback to `specialized` in any column.
- Effective pool width is `plan.compile_workers`: **4** (default), **1** (W=1), **7** (W=8) at p256, and **1** in every p64 column, because the chunk count is `min(workers, n_functions // 32)` and the p64 merge carries 63 functions.
  `protocol.batch_compile_workers` records the KNOB, not that effective width: `4`/`module` (shipped default), `1`/`env` and `8`/`env` (the two `--knob` columns), `default`/`harness-default` (pre-round tree, no pool).
- The HEAD columns record `env.tree_root_sha` `eeaabc44` (economics columns) and the campaign record `ef914d70`; `git diff eeaabc4 ef914d7 -- src/` is empty, so all HEAD columns measure the same engine tree and differ only by committed records.

### Contention and portability of these numbers

- One **shared** `AMD Instinct MI300X VF` (191.7 GB, `torch 2.13.0+rocm7.1`, `hip 7.1.52802`, `iree-base-* 3.11.0`) for the whole round; the GPU was not exclusive to this round.
- `rocm-smi --showpids` was read idle immediately before each config, and the base column's record carries that reading verbatim ("No KFD PIDs currently running"); the campaign column's own guard read the device idle before and after all three configs (`3/3 configs on an idle device, 0 contaminated`, empty `gpu_pids`).
- Warm medians on this box are **±10 %**, and a contended window inflates them 2–3×; read the cold/first-call column as "± a factor of a few" and the sub-millisecond warm deltas (§5) as "same or one noise step apart".
  One sample set shows it directly: `after_w1/p256` carries two warm outliers (3.964 ms and 4.056 ms against a 2.013 ms median), which is why medians and not means are quoted everywhere.
- The percentages and shares below are per-call properties of one compile on this box, not portable absolutes: the compile wall tracks `iree`'s compiler on this host, and the pool's parallel efficiency (81 % of 4 workers, 78 % of 7) is a property of this box's core count and shared load.

## 3. Cold compile (`first_call.first_call_ms`)

| cell | column | `first_call_ms` | pool wall | per-chunk compile sum | chunks (jobs) | chunk sizes | efficiency (sum ÷ wall) | serial pre | serial post | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| p256 | base `619cbf7` | 11066 | absent | absent | absent | absent | absent | absent | absent | absent |
| p256 | HEAD W=1 | 11054 | 10601 | 9946 | 1 | [225] | 0.938 (94 % of 1) | 385 | 66.0 | 2.4 |
| p256 | **HEAD default (W=4)** | **4444** | **4029** | **13057** | **4** | **[57,56,56,56]** | **3.24 (81 % of 4)** | **368** | **45.0** | **2.6** |
| p256 | HEAD W=8 → 7 | 3782 | 3355 | 18218 | 7 | [33,32,32,32,32,32,32] | 5.43 (78 % of 7) | 371 | 53.3 | 2.6 |
| p256 | dense | 3383 | absent | absent | absent | absent | absent | absent | absent | absent |
| p64 | base `619cbf7` | 2248 | absent | absent | absent | absent | absent | absent | absent | absent |
| p64 | HEAD default (W=4 knob, 1 chunk) | 2241 | 2060 | 1868 | 1 | [63] | 0.907 (91 % of 1) | 117 | 63.2 | 1.1 |
| p64 | HEAD W=1 | 2257 | 2077 | 1884 | 1 | [63] | 0.907 | 116 | 63.4 | 1.1 |
| p64 | HEAD W=8 | 2279 | 2086 | 1892 | 1 | [63] | 0.907 | 123 | 68.3 | 1.2 |
| p64 | dense | 6437 | absent | absent | absent | absent | absent | absent | absent | absent |

- `serial pre` = decode+dedup+trace+merge sums, `serial post` = upload+run+gather sums, `residual` = `first_call_ms` − (pool wall + serial pre + serial post).
  The residual is 1.1–2.6 ms (≤0.06 % of the call) in every batch column, which is what places `lower` and `load` **inside** the pool window and shows the stage vocabulary accounts for the whole cold call.
- Cold ratios at p256: base → default **2.49×**, base → 7 chunks **2.93×**, W=1 → default **2.49×**, W=1 → 7 chunks **2.92×**.
  At W=1 the cold call is the pre-round cold call (11054 vs 11066 ms, 1.00×), so the pool is the only cause of the reduction.
- Parallel efficiency is honest but sub-linear: 0.938 at one worker, 3.24× at 4 workers (81 % of the width), 5.43× at 7 (78 %). Both the compile counter and the efficiency numerator are thread sums and must always be read against the pool wall, never against `first_call_ms`.
- Chunk sizes come from `min(workers, n_functions // 32)` over the merged function multiset: 225 functions → 4 chunks of 57/56/56/56 at 4 workers, 7 chunks of 33+32×6 at 8 workers asked, 1 chunk of 225 at 1 worker; 63 functions → 1 chunk of 63 at every width (the p64 columns are three repeats of the same single-executable path, and their 2060–2086 ms pool wall is the noise band of this box).
- Cold versus dense: at p256 the best batch column is still 1.12× slower than dense cold (3782 vs 3383 ms) and the shipped default 1.31× (4444 vs 3383 ms); at p64 batch is **2.87× cheaper** cold than dense (2241 vs 6437 ms).

## 4. Cold stage split (the ten stages + the pool)

Milliseconds, straight from `first_call.stage_ms` (`batch_<stage>_total_ms`); the compile, lower and load rows are sums over pool threads, which is why the compile-pool wall row sits underneath them.

| stage | p256 W=1 | p256 W=4 (default) | p256 W=8→7 | p64 default |
|---|---|---|---|---|
| decode | 4.55 | 4.66 | 4.55 | 1.18 |
| dedup | 0.164 | 0.152 | 0.139 | 0.044 |
| trace | 155 | 151 | 151 | 26.2 |
| merge | 226 | 213 | 216 | 89.2 |
| lower | 543 | 1819 | 2536 | 115 |
| compile | 9946 | 13057 | 18218 | 1868 |
| load | 111 | 278 | 462 | 76.5 |
| upload | 60.6 | 41.0 | 47.9 | 62.0 |
| run | 1.70 | 3.13 | 3.55 | 0.864 |
| gather | 3.75 | 0.841 | 1.86 | 0.348 |
| **compile-pool wall** | **10601** | **4029** | **3355** | **2060** |
| **compile jobs** | **1** | **4** | **7** | **1** |
| plan `compile_workers` / `chunk_count` / `chunk_sizes` | 1 / 1 / [225] | 4 / 4 / [57,56,56,56] | 7 / 7 / [33,32×6] | 1 / 1 / [63] |
| call counts decode→gather | 1,1,1,1,1,1,1,1,1,2 | 1,1,1,1,4,4,4,4,4,5 | 1,1,1,1,7,7,7,7,7,8 | 1,1,1,1,1,1,1,1,1,2 |

Percentages of the shipped default's 4444 ms cold call (`p256`, W=4) — the distribution answer the round asked for:

- compile-pool wall **90.6 %** (4029 ms).
- host pre-work: trace 3.39 % (151 ms) + merge 4.78 % (213 ms) = **8.17 %**, plus decode 0.10 % and dedup 0.003 %.
- device post-work: upload 0.92 % (41.0 ms) + run 0.070 % (3.13 ms) + gather 0.019 % (0.841 ms) = **1.01 %**.
- Inside the pool window: lower 40.9 % (1819 ms thread sum) and load 6.25 % (278 ms thread sum) alongside the 13057 ms compile thread sum; the wall those threads actually consume is 4029 ms, so the compile *stage* counter reads 294 % of the call and is not a share of anything.

Notes worth keeping:

- `gather` has one more call than `upload`/`run` (4 vs 5 at p256 W=4): the loss readback adds one final gather after the last chunk.
- The pre-round tree cannot produce this table at all — `batch_stage_stats()` and `batch_stage_reset()` do not exist there, so `base/` carries `stage_ms.available: false` with every counter `null` and the note "both the before and the after snapshot must be dicts carrying `batch_<stage>_*` counters".
  That is the correct reading of "not measured", and it is why every base stage cell above says *absent* rather than 0.

## 5. Warm `sr_fitness` (per call, medians of 5 samples after 2 warmups)

| cell | column | median ms | run | upload | gather | run+upload+gather | outside the stage counters | vs base | vs dense |
|---|---|---|---|---|---|---|---|---|---|
| p256 | base `619cbf7` | 2.011 | absent | absent | absent | absent | absent | 1.00 | 1.86 |
| p256 | HEAD W=1 | 2.013 | 1.198 | 0.212 | 0.318 | 1.728 | 0.286 | 1.00 | 1.86 |
| p256 | **HEAD default (W=4)** | **3.628** | **1.949** | **0.752** | **0.622** | **3.323** | **0.305** | **1.80** | **3.35** |
| p256 | HEAD W=8 → 7 | 5.033 | 2.405 | 1.247 | 0.997 | 4.649 | 0.384 | 2.50 | 4.65 |
| p256 | dense | 1.082 | absent | absent | absent | absent | absent | 0.54 | 1.00 |
| p64 | base `619cbf7` | 1.041 | absent | absent | absent | absent | absent | 1.00 | 0.752 |
| p64 | HEAD default (1 chunk) | 1.127 | 0.466 | 0.186 | 0.225 | 0.877 | 0.250 | 1.08 | 0.814 |
| p64 | HEAD W=1 | 1.107 | 0.456 | 0.193 | 0.229 | 0.877 | 0.230 | 1.06 | 0.799 |
| p64 | HEAD W=8 | 1.121 | 0.470 | 0.189 | 0.232 | 0.891 | 0.230 | 1.08 | 0.810 |
| p64 | dense | 1.385 | absent | absent | absent | absent | absent | 1.33 | 1.00 |

- Warm stage figures are per-call medians of each stage separately, so they need not add up to the median of the whole call; the "outside the stage counters" column is that difference (≤0.38 ms, host glue and synchronisation inside the timed call).
- Every other warm stage is exactly **0**, not absent: a warm call decodes, traces and merges nothing, and the counters say so (decode/dedup/trace/merge/lower/compile/load calls = 0).
- The warm penalty is pure per-chunk overhead: upload+run+gather grows 1.728 → 3.323 → 4.649 ms as 1 → 4 → 7 chunks, i.e. **≈0.52 ms per extra chunk per call** (0.538 at W=4, 0.503 at 7).
  `run` per chunk falls (1.198 → 0.487 → 0.344 ms) while `upload` per chunk stays ≈0.18–0.21 ms, so the cost is per-chunk dispatch, not per-chunk arithmetic.
- W=1 is exactly the pre-round call (2.013 vs 2.011 ms, 1.00×), and the pre-round tree is the same single-executable path — the warm regression is bought entirely by the cold win.
- Break-even at p256 (per process, steady state): the **6610 ms** cold saving of W=4 over W=1 ÷ the **1.615 ms** per-call warm penalty ≈ **4.1 × 10³ warm calls** (4092) before the pool repays itself.
  In a generation loop it repays on the first recompiling generation instead, because a recompiling generation pays the whole cold call (median cold eval 2206 ms at W=4 vs 4474 ms at W=1).
- p64 shows the other side: one chunk, chunking inert (1.107–1.127 ms across all three widths, inside the ±10 % noise band), and the batch route still beats dense warm by 1.23×.

## 6. Generation loop (6 × `StandardPipeline.step()`, same seed, same pipeline)

| cell | column | `sum_total_ms` | `compile_ms_share` | `compile_share.share` | hit rate | gens with compile | median warm eval | median cold eval | vs dense |
|---|---|---|---|---|---|---|---|---|---|
| p256 | base `619cbf7` | 24803 | 0.989 | absent (`available:false`) | 1/6 | 5/6 | 2.23 | 4475 | 88.2× |
| p256 | HEAD W=1 | 25134 | 0.990 | 0.862 | 1/6 | 5/6 | 3.74 | 4474 | 89.3× |
| p256 | **HEAD default (W=4)** | **12773** | **0.979** | **2.62** | **1/6** | **5/6** | **3.65** | **2206** | **45.4×** |
| p256 | HEAD W=8 → 7 | 12440 | 0.979 | 3.10 | 1/6 | 5/6 | 5.22 | 2243 | 44.2× |
| p256 | dense | 281 | 0.000 | absent (no batch stage counters) | 1.00 | 0/6 | 2.84 | absent | 1.00 |
| p64 | base `619cbf7` | 4545 | 0.944 | absent (`available:false`) | 1/6 | 5/6 | 1.089 | 711 | 16.1× |
| p64 | HEAD default (1 chunk) | 4476 | 0.942 | 0.845 | 1/6 | 5/6 | 1.131 | 680 | 15.9× |
| p64 | HEAD W=1 | 4485 | 0.943 | 0.847 | 1/6 | 5/6 | 1.089 | 676 | 15.9× |
| p64 | HEAD W=8 | 4481 | 0.942 | 0.845 | 1/6 | 5/6 | 1.143 | 673 | 15.9× |
| p64 | dense | 282 | 0.000 | absent (no batch stage counters) | 1.00 | 0/6 | 2.79 | absent | 1.00 |

**The two share columns are different quantities and are not interchangeable.**

- `compile_ms_share` is an ATTRIBUTION, not a timer: the wall time of the generations that recompiled ÷ the loop's total wall time (bounded 0–1, an upper bound on what the compiling generations own, since they also evaluate and mutate).
  It reads 0.942–0.990 across the batch columns (≈0.98 at p256, ≈0.94 at p64): 5 of 6 generations recompiled, and recompiling generations are essentially the whole loop.
- `compile_share.share` is the compile STAGE counter (`batch_compile_total_ms`, a SUM over pool threads) for the compiling generations ÷ the loop's wall time.
  It therefore EXCEEDS 1 whenever the pool ran in parallel — **2.62** at 4 workers, **3.10** at 7 — and is **not** a share of wall time; at one worker it stays below one (0.845–0.862), which is the compile's own share of the loop.
  It is absent (`available: false`) on the pre-round tree, which has no stage counters, and absent on the dense route, which has no batch counters.
- dense's `generation_hit_rate` reads 1.00 for a different reason than the batch route's 1/6: a dense graph is keyed on shape × function set, not on the population, so no generation recompiles at all (`generations_with_compile` 0). Read it as "nothing to miss", not as a hit rate on a merged graph.
- Chunking halves the loop at p256 (24803 → 12773 ms, 1.94×; vs W=1 25134 → 12773, 1.97×) but does not move the ordering: `dense+iree` finishes the same 6 generations in 281 ms, i.e. 45.4× faster than the shipped default and 88.2× faster than the pre-round tree.
  At p64 the loop is unmoved by the knob (4476–4485 ms across widths) and dense is 15.9× faster.
- Generation 1 is the one hit in every column and it is by construction: `StandardPipeline.step()` evaluates the INITIAL population, whose graph the timed first call already compiled (`signature_seen_before: true` for `g=1` only). Generations 2–6 all miss.

## 7. Reuse / hit rate — does the plan key buy anything?

Counters after the 6-generation loop (`summary.merged_graph_cache.stats_after_generations`; the dense route owns no such block, so its cells are absent, not zero).

| cell | column | `batch_plan_hits` | `batch_plan_misses` | `batch_plan_entries` | `batch_exec_hits` | `batch_exec_misses` | `batch_exec_entries` | `batch_exec_maxsize` |
|---|---|---|---|---|---|---|---|---|
| p256 | base `619cbf7` | 9 | 6 | 6 | 0 | 6 | 6 | 8 |
| p256 | HEAD W=1 | 9 | 6 | 6 | 0 | 6 | 6 | 8 |
| p256 | **HEAD default (W=4)** | **9** | **6** | **6** | **0** | **22** | **8** (saturated) | **8** |
| p256 | HEAD W=8 → 7 | 9 | 6 | 6 | 0 | 28 | 8 (saturated) | 8 |
| p64 | all four batch columns | 9 | 6 | 6 | 0 | 6 | 6 | 8 |

First-call exec misses (the cold compile): 1 (base), 1 (W=1), 4 (default, one per chunk), 7 (7 chunks) at p256; 1 in every p64 column.

- **The content-addressed plan key produces no measurable reuse gain in a GP loop on these two cells.**
  The identity-plan counters are byte-identical pre-round and at HEAD (9 hits / 6 misses / 6 entries), `batch_exec_hits` is 0 everywhere, and `distinct_population_signatures` is 6 in 6 generations, so no generation ever presents a population the loop has already compiled.
  Gen 1's single hit is the initial population the timed first call compiled, not reuse across generations.
- That is what a GP loop is: crossover and mutation make every generation a fresh structure multiset.
  The distinct-structure count decays with selection pressure (`p64`: 63 → 56 → 41 → 26 → 17 → 13; `p256`: 225 → 177 → 109 → 121 → 136 → 214) but the multiset is never equal to a previous one, which is precisely the case a content key would serve ("a rebuilt population with identical content").
  So its benefit is confined to a rebuilt-but-EQUAL population, which this loop does not produce.
- Only cache-side change chunking brings: exec entries are per CHUNK, so one merged graph occupies `chunk_count` entries.
  At 4 workers the exec table records 22 misses and saturates `batch_exec_entries` at its `maxsize` of 8 — i.e. **two merged graphs** — where the pre-round single-executable path recorded 6 misses / 6 entries (one per graph); at 7 chunks the 8 slots hold 1.14 graphs.
  `EVOGP_BATCH_CACHE_SIZE=8` is therefore now a capacity bound of "two populations at 4 chunks", which matters for any loop that WOULD reuse (a rebuilt population, an elite re-evaluation), even though this loop loses nothing to the saturation because it never reuses.
- Generations 3 and 4 at p256/W=4 record 3 exec misses where the others record 4. The chunk count is `min(workers, n_functions // 32)` over the merged function multiset, so a smaller multiset yields fewer chunks; the per-generation chunk count is not recorded in this column (only the first call's plan is), so read those cells as compile-probe counts, not as chunk counts.
- Not measured by this harness: the CONTENT table's own counters (`batch_content_stats()`: `batch_content_hits`/`_misses`/`_rejects`/`_entries`/`_maxsize`).
  `batch_cache_stats()` deliberately stays at its eight identity-keyed counters, and no harness file mentions `batch_content`, so the round cannot price a content hit directly — it can only show that the counters which decide whether a generation pays a compile are unchanged.

## 8. What this does not cover

- No `p1000` cells: the pop-1000 merged compile measured by the previous round is 91–169 s, and a pool sweep there asks a different (scale-envelope) question.
- No long-tree cells (`L=128`/`L=256`, or the `-d6/-d7/-d8` generation-depth siblings): the trace-cap crossover the batch route wins is a different regime.
- One seed (0), one box.
- `iree` only — the numpy fallback path is not exercised by any column.
- No `EVOGP_BATCH_BAND_SIZE` / `EVOGP_BATCH_BANDS` / `EVOGP_BATCH_ALIGNMENT` / `EVOGP_BATCH_SWITCH_OPS` sweep: the merge shape is fixed at the module defaults (band size 32, K from the merge, `variant` alignment, switch ops off).
- No warm/cold measurement of the other workloads (`batch_forward`, `forward`, `gp_step`) across the W sweep; the W columns time `sr_fitness` only, and the campaign column times all four workloads at the shipped default alone.
- No content-table counters (§7), and nothing here validates numerics: this round is timing plus counter economics, and the values were not diffed against the oracle.
- The pool's scaling is measured at 1, 4 and 7 workers on one 225-function merge; width beyond 7 (a larger population, or a raised `n_functions // 32` ceiling) is unmeasured.

## 9. Reproduce

Repository root as CWD, project venv, the device box; one process per config, sequentially, nothing else touching the GPU.

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
export EVOXC_DISABLE_LOCATIONS=1        # the driver applies this itself; exported here for clarity
PY=/tmp/rocvenv/bin/python
CELLS=p256-L32-i8-N1024,p64-L64-i32-N64
OUT=test/benchmark/evoxc/results/batch_opt_round1

# base column: the pre-round tree, extracted outside the repo; HEAD's harness drives it
$PY test/benchmark/evoxc/batch_economics.py --tree-root "$TMPDIR/evogp_preopt_619cbf7" \
  --cells $CELLS --routes batch --generations 6 --warmup 2 --reps 5 --out-dir $OUT/base

# HEAD, shipped default (EVOGP_BATCH_COMPILE_WORKERS unset -> module default, 4 workers)
$PY test/benchmark/evoxc/batch_economics.py \
  --cells $CELLS --routes batch --generations 6 --warmup 2 --reps 5 --out-dir $OUT/after_default

# HEAD, single-executable control and a wider pool (the knob is per process: one config each)
$PY test/benchmark/evoxc/batch_economics.py \
  --cells $CELLS --routes batch --generations 6 --warmup 2 --reps 5 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=1 --out-dir $OUT/after_w1
$PY test/benchmark/evoxc/batch_economics.py \
  --cells $CELLS --routes batch --generations 6 --warmup 2 --reps 5 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=8 --out-dir $OUT/after_w8

# dense comparison column on the same cells
$PY test/benchmark/evoxc/batch_economics.py \
  --cells $CELLS --routes dense --generations 6 --warmup 2 --reps 5 --out-dir $OUT/after_dense

# campaign-format gate record (worker.py path), one cell, batch arm + both torch arms
$PY test/benchmark/evoxc/campaign.py --cells p256-L32-i8-N1024 \
  --arms evoxc-batch-iree,torch-per-tree,torch-batched --reps 5 --out $OUT/campaign
```

The pre-round tree is `619cbf738af0bf839085879193f163d383bd9fbc` unpacked at `$TMPDIR/evogp_preopt_619cbf7`; it carries no `.git`, which is why every `base/` record has `env.tree_root_sha: null` and why `--allow-snapshot` belongs in any gate over records taken from it.
`--dry-run` is torch-free and prints the resolved batch policy and every templated knob before anything runs.

## 10. Gate

```bash
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch_opt_round1/campaign \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# -> 3 records / 88 checks / 0 failures  (exit 0)
```

The only messages are the two declared expectations, both allowed by `--allow-digest-mismatch`:

- `p256-L32-i8-N1024/batch_forward` splits into two sha256 groups (`evoxc-batch-iree` `f54a1dd46f3a` vs the two torch arms `d31bf4ba4573`).
- `p256-L32-i8-N1024/forward` splits the same way (`a718347235cd` vs `e521e515e93b`).

Both are the batch route's non-bitwise value path on the arithmetic set against a plain-torch oracle — the same finding recorded in `VALIDATION.md` and `BATCH.md`, not a regression.
`--allow-snapshot` and `--allow-missing` are carried because the base column is a snapshot tree and because `evoxc-specialized-iree` is unscheduled on pop ≥ 256 cells; neither produced a message on this three-record directory.

The guard read the device idle before/after all three configs (`3/3 configs on an idle device, 0 contaminated`, empty `gpu_pids`), and the base column's own economics note records `rocm-smi --showpids` reading "No KFD PIDs currently running" immediately before it ran.
The campaign record independently reproduces the warm ordering on the same cell — `sr_fitness` medians 30.87 ms (`torch-per-tree`), 4.07 ms (`torch-batched`), 3.63 ms (`evoxc-batch-iree`) — and its `gp_step` median of 2243.9 ms against 19.84 ms for `torch-per-tree` is the same recompile-dominated generation §6 prices.

## 11. Follow-up: the content-addressed plan key, measured
§7 could only show that a generation loop gets no plan-key reuse; it could not say what the *content* key table (`batch_content_stats`) buys, because a generation loop never presents the same programs twice.
`reuse_probe.py` measures it directly on one cheap cell (`p64-L64-i32-N64`, seed 0, `evoxc-batch-iree`, band 32, pool 4): four calls, one dataset object, two populations.

| call | population | HEAD (`reuse/after`) | pre-round `619cbf7` (`reuse/base`) | HEAD counters | pre-round counters |
|---|---|---|---|---|---|
| 1 | A, cold | 2340.3 ms | 2401.2 ms | exec 1 m, plan 1 m, content 1 m (`attempts` 1, `hit_rate` 0.0) | exec 1 m, plan 1 m — no content table |
| 2 | A, identity | 1.274 ms | 1.235 ms | plan 1 h, content `attempts` **0** | plan 1 h |
| 3 | **B, rebuilt-but-equal** | **2.724 ms** | **79.109 ms** | plan 1 m -> content **1 h / 0 m** (`hit_rate` 1.0), trace/merge/compile **0 calls** | exec **1 h**, plan 1 m -> re-decode + re-trace + re-merge |
| 4 | B, identity | 1.107 ms | 1.101 ms | plan 1 h, content `attempts` 0 | plan 1 h |

- `Forest` B is a different object built from the same seed: `forests.same_content` records `identical_content: true` and `no_shared_storage: true`, so the only thing that can answer call 3 cheaply is a content key.
- Measured hit-rate change: pre-round has no content surface at all (block `available: false`, every counter `null`, NOT 0) and pays a plan MISS; HEAD turns the same call into a content HIT (`1 hit / 0 misses`, `hit_rate` 1.0) at **29.0× less time** (79.109 -> 2.724 ms, i.e. 76.4 ms per rebuilt population).
- All four calls, on both trees, return the same values (sha256 `b6584500a0fb365adfbf712cc6b43d7e`, 64 finite / 0 NaN): the reuse is exact, not approximate.
- The replay column `reuse/after-replay` reproduces it (cold 2302.6 ms, rebuilt 2.649 ms) with the same `verification` block.
- **The generation-loop verdict does not move:** in the 6-generation `content-blocks/after` loop at the same cell the content table saw 6 attempts and 0 hits (`summary.content.generations` = `{rows 6, with_content_hit 0, with_content_miss 5, no_attempt 1}`; generation 1 is the population the cold call compiled, generations 2..6 present new programs).
- The pre-round column also shows what the content key is *not*: `batch_exec_hits` 1 on call 3 means the compiled body was already shared, so 76.4 ms is host-side decode + trace + merge, never a recompile.
- `worker.py`'s HEAD `batch_cache.content_totals` reads `{hits 0, misses 3, attempts 3}` — one content probe per workload plan, all misses; the warmup+reps calls behind them are answered by the identity table (`attempts` 0).
- Reproduce: `$PY test/benchmark/evoxc/reuse_probe.py --label after` / `--label base --tree-root <checkout of 619cbf7>` / `--label after-replay`, and the content-block columns with `batch_economics.py --cells p64-L64-i32-N64 --out-dir .../content-blocks/{after,base}`.
- `EVOGP_BATCH_COMPILE_WORKERS` was unset (module default 4) in every column here; at p64 the merge carries 63 functions, so the pool is inert and the numbers are the single-executable path.
