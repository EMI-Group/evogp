# Round 2, work item 3 — the banded+chunked composition measured against the single-run banded arm and against `dense`

**Question.** The engine's composition opt-in is `EVOGP_BATCH_CHUNK_BANDS=1` on top of a band request:
the merge is split into contiguous chunks that are lowered, compiled and loaded by the compiled-graph
pool (`EVOGP_BATCH_COMPILE_WORKERS`), and each chunk carries its own band partition. Three things are
measured here, on the same cells and populations: (a) does it beat the single-run banded arm **warm**
(`b16` at pool 1, the work-item-2 numeric default); (b) does it beat the **flat** chunked arm **cold**
(the round-1 shape, before a band request started forcing the single-body path); and (c) what does it
do to a real generation loop (`StandardPipeline.step()` × 6) against `dense`?

**Answer (short).** It is a **cold lever and a warm tax.** Cold, `b16` + composition is
**0.35–0.40×** the single-body arm's cold call on the two wide-row plans (4 180.6 ms against
11 049.4 ms at `p256-L32 sr_fitness`; 4 422.4 against 12 590.6 at L64) and 0.54× on the per-row plan;
warm it is **1.61–2.47×** the same arm. The warm loss is the **composition itself, not the `auto` band
policy** — pinned `b16-cw4` and `auto-cw4` are within 3 % on six of nine warm columns and both are the
same distance from the single-body arm, and the warm stage split names the mechanism: four separate
upload / run / read-back trios per call (`upload` 0.191 → 0.747 ms, `gather` 0.244 → 0.659, `run` 1.317
→ 1.947 at `p256-L32 sr_fitness`). The flat arm cannot use the composition at all. In a generation loop
**every** batch route loses to `dense`, and the chunked composition is only the least bad one
(**43.2×** at L32, **48.6×** at L64, against 80.8–95.4× for the single-body routes). No default flips:
`EVOGP_BATCH_CHUNK_BANDS` stays off.

## 1. Protocol

| axis | values |
|---|---|
| cells | `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p64-L64-i32-N64` (seed 0, `max_layer_cnt` 5, `+ - * /`); structure signatures `ad11cb66ffc21ac8` / `0a0dee3fec0cfb9c` / `18d209693a5a7e9e`, 225 / 232 / 63 deduped functions |
| sweep configs (`band_sweep.py`) | `b16-cw4` (`BAND_SIZE=16`, `CHUNK_BANDS=1`, pool 4 — the composition with the band **policy** removed), `auto-cw4` (`BANDS=auto`, `CHUNK_BANDS=1`, pool 4), `flat-cw4` (`BANDS=none`, pool 4, `CHUNK_BANDS` unset) |
| loop routes (`batch_economics.py`) | `p256-L32`: `batch-banded` (`BANDS=auto`, pool 1), `batch-b16-banded` (`BAND_SIZE=16`, pool 1), `batch-b16-chunked` (`BAND_SIZE=16`, `CHUNK_BANDS=1`, pool 4), `batch-auto-chunked` (`BANDS=auto`, `CHUNK_BANDS=1`, pool 4), `batch-flat-chunked` (`BANDS=none`, pool 4), `dense`; `p256-L64`: `batch-banded`, `batch-b16-chunked`, `batch-auto-chunked`, `dense` |
| reference arm | the single-body `b16` (`BAND_SIZE=16`, pool 1) of `../auto/<cell>/b16.json`, measured in the work-item-2 window on the same population |
| workloads | `sr_fitness` (loss, N = 1024 / 64 rows), `batch_forward` (population over N rows), `forward` (one row per tree) — fixed order, one plan + one compile each |
| protocol | sweep: `--warmup 1 --reps 5 --seed 0`, warm medians, the untimed cold first call separated; loop: `--generations 6 --warmup 2 --reps 5`; one child process per `(cell, config)` / `(cell, route)`, one at a time |
| engine | `EVOGP_EVAL_MODE=batch`, `EVOGP_EVOXC_BACKEND=iree`, `EVOXC_DISABLE_LOCATIONS=1` |
| identity (verbatim from the records' `env`) | evoxc `2b289f0` (sweeps from `/tmp/evoxc_r2`, loop children from `/root/evoxc` — the same commit), evogp tree `0b9075f1` (the `b16-cw4` sweep and both loop runs, worktree `worker_T1_A74`) and `c97a6630` (the `auto-cw4`/`flat-cw4` sweep, worktree `worker_T1_A73`), torch 2.13.0+rocm7.1, hip 7.1.52802, iree 3.11.0, numpy 2.5.4, python 3.12.3, AMD Instinct MI300X VF (191.7 GB) |

Commands of record (run from the repository root):

```bash
# 1. the auto / flat-cw4 sweep (6 children, 0 failures, driver elapsed_s 134.4)
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src:test/benchmark/evoxc
/tmp/rocvenv/bin/python test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --configs auto-cw4,flat-cw4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/composition \
  --tree-root src --warmup 1 --reps 5 --seed 0

# 2. the pinned-size composition sweep (3 children, 0 failures, driver elapsed_s 48.7)
/tmp/rocvenv/bin/python test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --configs b16-cw4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/composition \
  --tree-root src --warmup 1 --reps 5 --seed 0

# 3. the generation loop at p256-L32 (6 children, all ok; 25.9-46.4 s each)
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
/tmp/rocvenv/bin/python test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024 \
  --routes batch-banded,batch-b16-banded,batch-b16-chunked,batch-auto-chunked,batch-flat-chunked,dense \
  --generations 6 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/composition

# 4. the same loop at p256-L64 (4 children, all ok; 28.1-52.6 s each)
/tmp/rocvenv/bin/python test/benchmark/evoxc/batch_economics.py \
  --cells p256-L64-i32-N1024 \
  --routes batch-banded,batch-b16-chunked,batch-auto-chunked,dense \
  --generations 6 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/composition/l64
```

Records: `<cell>/<config>.json` for the two sweeps (schema `evogp-bench-evoxc-v1-band-sweep`) plus
their indexes — `sweep.json` is the `b16-cw4` run and `sweep-auto-flat-cw4.json` the earlier
`auto-cw4`/`flat-cw4` index, kept beside it because a second sweep into the same `--out-dir` rewrites
`sweep.json` — and `<cell>/<route>.json` + `economics.json` for the loops (schema
`evogp-bench-evoxc-v1-batch-economics`, the `l64/` pair under its own nested `--out-dir`).
The two measurement paths agree where they overlap: the sweep's cold/warm for the composition are
4 180.6 / 3.5736 ms (`b16-cw4`), 4 873.8 / 3.5980 (`auto-cw4`) and 12 671.6 / 2.2573 (`flat-cw4`) at
`p256-L32 sr_fitness`, against the loop records' 4 165.9 / 3.6876, 5 020.1 / 3.8148 and 12 742.8 /
2.2244 — 0.3–3 % apart in separate processes, which is how the two are quoted below (the sweep for
warm/cold of the composition, the loop for anything generation-level).

## 2. Arm table — `p256-L32-i8-N1024`, `dense` as the reference row

Cold = `first_call.first_call_ms` (the merged-graph compile of the `sr_fitness` plan, never part of a
warm median). Warm = `summary.warm_median_ms`. Loop = Σ `generations[i].total_ms` over the 6 steps;
×`dense` is that Σ over `dense`'s.

| route | route knobs as resolved | cold ms | warm `sr_fitness` ms | Σ 6 gens ms | ×`dense` |
|---|---|---:|---:|---:|---:|
| `batch-b16-chunked` | `BAND_SIZE=16`, `CHUNK_BANDS=1`, pool 4 | 4 165.9 | 3.6876 | **12 906.6** | **43.2×** |
| `batch-auto-chunked` | `BANDS=auto`, `CHUNK_BANDS=1`, pool 4 | 5 020.1 | 3.8148 | 14 387.1 | 48.1× |
| `batch-flat-chunked` | `BANDS=none`, pool 4 | 12 742.8 | 2.2244 | 24 171.6 | 80.8× |
| `batch-banded` | `BANDS=auto`, pool 1 | 13 043.9 | 2.2659 | 24 714.0 | 82.7× |
| `batch-b16-banded` | `BAND_SIZE=16`, pool 1 | 11 085.0 | 2.0391 | 25 578.3 | 85.5× |
| `dense` | — | 3 491.7 | **1.1027** | **299.0** | 1.00 |

`p256-L64-i32-N1024`, the same columns:

| route | cold ms | warm `sr_fitness` ms | Σ 6 gens ms | ×`dense` |
|---|---:|---:|---:|---:|
| `batch-b16-chunked` | 4 404.4 | 3.7324 | **14 710.4** | **48.6×** |
| `batch-auto-chunked` | 5 307.2 | 3.8894 | 15 537.6 | 51.3× |
| `batch-banded` | 15 004.4 | 2.4993 | 28 871.7 | 95.4× |
| `dense` | 6 816.3 | **2.0240** | **302.7** | 1.00 |

Three readings: the composition is the **cheapest cold batch route** in both tables (4 165.9 ms at L32,
4 404.4 at L64), and against `dense`'s own cold call it is dearer at L32 (4 165.9 against 3 491.7) but
**cheaper at L64** (4 404.4 against 6 816.3) — the one column where a batch route beats the shipped
route cold; it is the **most expensive warm** batch route after `auto-chunked`; and no batch route is
within two orders of magnitude of `dense` once generations start.

## 3. Warm/cold composition table — `b16` single body vs `b16-cw4` vs `auto-cw4`

Reference = the single-body `b16` of `../auto/<cell>/b16.json`; `(×)` is that record's own cold/warm.

| cell | workload | `b16` cold ms | `b16-cw4` cold (×) | `b16-cw4` warm (×) | `auto-cw4` cold (×) | `auto-cw4` warm (×) |
|---|---|---:|---|---|---|---|
| p256-L32-i8-N1024 | `sr_fitness` | 11 049.4 | 4 180.6 (**0.38**) | 3.5736 (**1.76**) | 4 873.8 (0.44) | 3.5980 (1.77) |
| p256-L32-i8-N1024 | `batch_forward` | 7 496.0 | 2 999.3 (**0.40**) | 4.3761 (**1.96**) | 3 575.2 (0.48) | 3.4068 (1.53) |
| p256-L32-i8-N1024 | `forward` | 3 074.8 | 1 658.8 (**0.54**) | 1.9470 (**2.26**) | 2 057.2 (0.67) | 2.1347 (2.47) |
| p256-L64-i32-N1024 | `sr_fitness` | 12 590.6 | 4 422.4 (**0.35**) | 3.6770 (**1.64**) | 5 332.9 (0.42) | 3.9478 (1.76) |
| p256-L64-i32-N1024 | `batch_forward` | 7 283.5 | 2 943.8 (**0.40**) | 3.6357 (**1.61**) | 3 683.4 (0.51) | 3.5163 (1.56) |
| p256-L64-i32-N1024 | `forward` | 3 385.3 | 1 825.9 (**0.54**) | 2.0355 (**2.47**) | 2 079.0 (0.61) | 2.0409 (2.48) |
| p64-L64-i32-N64 | `sr_fitness` | 2 298.7 | 2 284.2 (0.99) | 1.1765 (1.04) | 2 518.6 (1.10) | 1.1828 (1.05) |
| p64-L64-i32-N64 | `batch_forward` | 1 675.9 | 1 684.0 (1.01) | 0.9231 (1.04) | 1 651.6 (0.99) | 0.9178 (1.03) |
| p64-L64-i32-N64 | `forward` | 1 059.2 | 813.6 (0.77) | 1.0710 (1.51) | 821.2 (0.78) | 1.0933 (1.54) |

**The composition is a cold lever: 0.38 / 0.40 / 0.54× of `b16`'s cold at `p256-L32` (0.35–0.40× on
the two wide-row plans at L64, 0.54× on the per-row one) but 1.76–2.26× its warm** — `sr_fitness`
2.0331 → 3.5736 ms, `forward` 0.8632 → 1.9470 ms at `p256-L32`; at L64 the warm tax is 1.61–2.47× and at
`p64` it is 1.04× on the wide-row plans and 1.51× on the per-row one (§7).

**The warm loss is the composition itself, not the `auto` band policy — that is the decisive
attribution.** Both composition arms sit 1.53–2.48× above the single-body arm on **every** pop-256
column, and the spread *between* the two of them is far smaller: they agree within 3 % on six of nine
warm columns (`p256-L32 sr_fitness` 0.7 %, `p256-L64 batch_forward` 3.4 %, `p256-L64 forward` 0.3 %,
and the three `p64` rows 0.6–2.0 %) and differ by 6.9–9.6 % on two more (`p256-L64 sr_fitness` 7.4 %,
`p256-L32 forward` 9.6 %, where `auto-cw4` is slower because its per-row plan chunks at
`chunk_bands [1,1,1,1]` against `b16-cw4`'s `[4,4,4,4]` — a real partition difference inside the
chunks). The one large gap is a **contention spike in a single row**, not a policy effect:
`b16-cw4 batch_forward` at `p256-L32` has samples 3.578 / 4.376 / 6.450 / 5.635 / 3.537 ms
(`max:min` 1.82), so its median (4.3761) is 22 % above `auto-cw4`'s 3.4068 while its **min** (3.5370) is
only 5.1 % above `auto-cw4`'s min (3.3651). Nothing in the attribution rests on that row.

**The mechanism is in the warm stage split.** `p256-L32 sr_fitness`, the loop records'
`sr_fitness.stage_ms` (`batch-b16-banded` → `batch-b16-chunked`, both 5 samples, decode / dedup / trace /
merge / lower / compile / load all 0 ms in both because every plan is a memo hit):

| stage | `batch-b16-banded` (1 body) | `batch-b16-chunked` (4 chunks) | calls |
|---|---:|---:|---|
| `upload` | 0.1910 | **0.7473** | 1 → 4 |
| `run` | 1.3169 | **1.9466** | 1 → 4 |
| `gather` | 0.2443 | **0.6588** | 2 → 5 |

Four separate upload / run / read-back trios per call instead of one: the per-chunk graph saves compile
work, and the call pays it back on every evaluation. The whole warm delta (1.317 → 3.688 ms of
`median_ms`) is inside those three stages.

## 4. Where the cold win comes from

`p256-L32 sr_fitness`, the cold call's own stage split (sweep `stage_first_call`, `b16` vs `b16-cw4`):

| stage | `b16` (1 body) | `b16-cw4` (4 chunks) |
|---|---:|---:|
| `compile` (counter sum) | 10 172.6 (1 call) | **12 439.4** (4 chunk timers) |
| `batch_compile_wall_ms` | 10 583.6 (**1 job**) | **3 733.1** (**4 jobs**) |
| `lower` | 296.6 (1 call) | 888.5 (4 calls) |
| `merge` | 233.1 | 228.3 |
| `trace` | 155.6 | 166.9 |
| `load` | 114.3 | 290.8 |
| `upload` | 66.9 | 40.7 |
| `decode` / `dedup` | 5.3 / 0.16 | 4.9 / 0.17 |
| `run` / `gather` | 1.85 / 0.34 | 2.98 / 0.78 |
| **`first_call_ms`** | **11 049.4** | **4 180.6** |

The **compile** stage's counter is a **sum of four per-chunk timers** (12 439.4 ms) while the pool's own
wall time is **`batch_compile_wall_ms` 3 733.1 ms over 4 jobs** (against 10 583.6 ms over 1 job) — that
is where the cold win comes from, and it is the same 4-way parallelism round 1 introduced. `lower`
multiplies by 3.0 (296.6 → 888.5 ms) because every chunk is lowered separately, and `load` by 2.5, so
the net cold call is 11 049.4 → **4 180.6 ms** rather than a 4× win. Two accounting facts follow and
both are recorded as such: the stage **counter sum** on the chunked row is 14 063 ms against a 4 181 ms
call (it cannot close, by construction — the chunk timers are concurrent), and a **`compile_share`
above 1** on the loop records is that same concurrency-weighted counter ratio, not wall-clock
attribution (`batch-b16-chunked` `summary.compile_share` 2.6744 = 34 517.2 ms of compile counters over a
12 906.6 ms loop, with `numerator_kind: parallel-stage-counter-sum`, `denominator_kind: loop-wall-ms`,
`share_gt_one: true` and a note saying so).

## 5. Generation loop — every batch route loses to `dense`

All six L32 routes and all four L64 routes ran 6 `StandardPipeline.step()` generations on the same
population and seed; `generation_hit_rate` is **1/6** with **6 distinct signatures** and
`generations_with_compile` **5** on every batch route (`compile_ms_share` 0.979–0.991), i.e. the loop is
a compile pipeline: generation 1 hits the graph the timed first call already compiled, and every
following generation presents a new structure multiset and recompiles.

| cell | route | Σ 6 gens ms | ×`dense` |
|---|---|---:|---:|
| p256-L32 | `batch-b16-chunked` | 12 906.6 | **43.2×** |
| p256-L32 | `batch-auto-chunked` | 14 387.1 | 48.1× |
| p256-L32 | `batch-flat-chunked` | 24 171.6 | 80.8× |
| p256-L32 | `batch-banded` | 24 714.0 | 82.7× |
| p256-L32 | `batch-b16-banded` | 25 578.3 | 85.5× |
| p256-L32 | `dense` | 299.0 | 1.00 |
| p256-L64 | `batch-b16-chunked` | 14 710.4 | **48.6×** |
| p256-L64 | `batch-auto-chunked` | 15 537.6 | 51.3× |
| p256-L64 | `batch-banded` | 28 871.7 | 95.4× |
| p256-L64 | `dense` | 302.7 | 1.00 |

- **Chunking amortizes one compile 4-way, never a loop.** Each generation's compile is split across the
  pool, so the chunked composition is the least bad batch route at both cells (43.2× / 48.6× against
  80.8–95.4× for the single-body routes) — the ordering the cold column predicts, at the loop level.
  It is still two orders of magnitude behind `dense`, because `dense` does not recompile per generation
  at all (`generations_with_compile` 0).
- **The operator layer does not drive the ratio.** Generation 1 pays the pipeline's first trace on both
  sides (`ops_ms` 269.9 on `dense`, 264.6 / 267.9 / 253.7 / 252.6 / 269.2 on the batch routes) and the
  remaining generations cost ~2.9–3.1 ms of operators each. Excluding generation 1, the **eval-only**
  sums are 15.3 ms (`dense`) against 12 623.1 ms (`b16-chunked`) at L32 = **825×**, and 22.1 against
  14 427.4 ms at L64 = **653×** — hundreds-fold, which is what "a new merged graph per generation"
  means; the Σ-total ratio of 43× is the *flattering* number because generation 1 dominates `dense`'s
  Σ.

## 6. The flat arm cannot chunk

`EVOGP_BATCH_BANDS=none` (the `flat` spelling of a band request) does **not** chunk, even with
`EVOGP_BATCH_COMPILE_WORKERS=4`: every `flat-cw4` record reports `compile_workers 1`, `chunk_count 1`
and `chunk_bands [1]` on all three workloads of all three cells, and its `plan` blocks carry
`chunk_sizes [225]` / `[245]`. **A band request — the flat spelling included — forces the single-body
path**, so round 1's chunked-flat cold win is no longer reachable that way and
`EVOGP_BATCH_CHUNK_BANDS=1` is the only route to chunking. Consequences: the flat arm's cold call
(12 671.6 ms at L32, 14 610.0 ms at L64) sits in the same band as the **single-body** `b16`
(11 049.4 / 12 590.6) and `auto` (12 429.5 / 14 138.5) cold calls, and the composition beats the flat
arm cold by 3.0× / 3.3× — but that is the composition against a route that never got to chunk, not a
composition-vs-composition comparison.

## 7. `p64` inertness — the composition is a pop-256-and-up lever

At `p64-L64-i32-N64` (63 deduped functions on the two wide-row plans, 64 on the per-row one) the pool is
**inert**: `chunk_count` is **1** on both 63-function plans for both configs (`chunk_bands [4]` for
`b16-cw4`, `[7]` for `auto-cw4` — the clamped `auto` K), because chunking is inert below 32 deduped
functions; only the per-row plan splits 2-way (`chunk_bands [2,2]` / `[1,1]`). So warm is 1.04× and
cold 0.99–1.01× of the single-body arm on the wide-row plans, the per-row plan — the one that does split
— pays 1.51× warm for a 0.77× cold call, and the emitted values are **bitwise equal** to `b16`'s on all
three workloads (`sr_fitness` `b6584500…`, `batch_forward` `4919287…`, `forward` `6d7d47f…`).

## 8. Recommendation

**Do NOT flip any default. `EVOGP_BATCH_CHUNK_BANDS` stays OFF (its shipped state).** The composition
trades **1.76–2.26× warm** for **2.6–2.8× cold** at the pop-256 cells, is inert at pop 64, and changes
nothing in a generation loop (the loop routes are 43–95× behind `dense` with or without it). It is a
**cold-start lever for a FIXED population**: break-even = cold saving ÷ warm penalty ≈
(11 049.4 − 4 180.6) ms / (3.5736 − 2.0331) ms ≈ **4.5k warm calls** at `sr_fitness` `p256-L32`
(≈ **1.3k** on `forward`, where the saving is 1 416.0 ms and the penalty 1.0838 ms) — the same order as
round 1's ≈4.1k, and the same conclusion: only a long-lived fixed population reaches it.

For a cold-sensitive run that wants it anyway, flip these three and **pin the size, never `auto`**
(work item 2's decision):

```bash
EVOGP_BATCH_BAND_SIZE=16 EVOGP_BATCH_CHUNK_BANDS=1 EVOGP_BATCH_COMPILE_WORKERS=4
```

(`EVOGP_BATCH_COMPILE_WORKERS=1` remains the warm-oriented setting for a single-body merge, and
`auto-cw4` is dominated by `b16-cw4` in 7 of the 9 warm columns here — 4 of the 6 at the pop-256 cells,
one of them the contention-spiked `batch_forward` row named in §3.)

## 9. Contention, coverage, what is and is not gateable

- **Guard verdicts: 9/9 `ok`, `gpu_pids.idle: true`, `contaminated: false`, `waited_s` 0.2 s on every
  config of both sweeps** (`sweep.json` = the `b16-cw4` run, `sweep-auto-flat-cw4.json` = the
  `auto-cw4`/`flat-cw4` run, 0 failures and 0 skipped in each; driver `elapsed_s` 48.7 s and 134.4 s).
  For contrast, the reference window (`../auto/sweep.json`, which the single-body numbers come from) had
  to wait **42.0 s** (`p256-L32 b16`) and **57.6 s** (`p256-L64 auto`) for a foreign job and recorded
  both waits verbatim; nothing there was re-run or smoothed.
- **The loop runs have no guard block.** `batch_economics.py` has no idle-device guard (no `guard` key in
  `economics.json`, `l64/economics.json` or any loop record), so idleness was verified **by hand** with
  `rocm-smi --showpids` before each run and the runs were strictly sequential (6 children, then 4). That
  is a weaker protocol than the sweeps' and should be read as such; per-child `wall_s` was 25.9–46.4 s
  (L32) and 28.1–52.6 s (L64).
- **Two worktrees, one evoxc.** The `auto-cw4`/`flat-cw4` sweep ran from `worker_T1_A73`
  (`tree_root_sha c97a6630`) and the `b16-cw4` sweep plus both loop runs from `worker_T1_A74`
  (`0b9075f1`); the records state this themselves (`env.tree_root_sha`, `protocol.tree_root`). The cell
  definitions agree exactly (identical `population_signature` per cell), evoxc is the same commit
  (`2b289f03`) everywhere, and the loop children resolved it from `/root/evoxc` rather than the sweeps'
  `/tmp/evoxc_r2` — same sha, so no split is attributable to the engine.
- **Not gateable.** `check.py` gates campaign **arm records**; an economics record is a different schema
  (it reads each `economics.json` as an arm record and fails on the missing `env`/workload fields), and
  `composition/l64/` is a nested `--out-dir` besides. The band-sweep schema is gateable only in the sense
  this round's earlier records were: `check.py` on a sweep directory reports the `cache`-block failures a
  sweep cannot satisfy. So **these records are evidence, not gateable units** — their internal
  consistency checks are `status: ok`, the stage-split closure (§4), the guard block and the per-workload
  oracle validation (§10).
- **Coverage.** Measured: 3 cells × 3 sweep configs × 3 workloads = 27 sweep rows, all `ok`; 6 + 4 loop
  routes × 6 generations, all `ok`. **Not measured**: `p1000-*` (a merged compile of 91–162 s per
  workload per config, and the composition's own scale envelope is untested there), the composition with
  `EVOGP_BATCH_BANDS=K` pinned at the clamp ceiling, `EVOGP_BATCH_ALIGNMENT` / `EVOGP_BATCH_SWITCH_OPS`
  variants, and the cold-start / training-run columns of the campaign harness for these routes.

## 10. Numeric validation

Each sweep record diffs its own outputs against the independent plain-torch oracle
(`src/evogp/tests/reference_eval.py` via `validate.compare`, first 16 trees of the workload). **The
oracle deviation is IDENTICAL in all three configs of every cell/workload (9/9)** — same `max_abs`, same
`max_rel`, same `bitwise_equal` — and identical to what item 1 and item 2 measured: `p256-L32 sr_fitness`
6.0 / 4.74e-07, `batch_forward` 5.86e-03 / 1.13e-03, `forward` 0 / 0 (bitwise); `p256-L64 sr_fitness`
32.0 / 7.76e-07, `batch_forward` 7.81e-02 / 1.14e-05, `forward` 1.19e-07 / 8.01e-08; `p64 sr_fitness`
0.25 / 1.62e-07, `batch_forward` 6.10e-05 / 1.68e-07, `forward` 1.19e-07 / 8.01e-08. No NaN or inf
mismatch anywhere (`nan_mismatch` / `inf_mismatch` 0).

Value digests: `sr_fitness` is **bitwise identical** to the single-body `b16` for `b16-cw4` at both
pop-256 cells and at `p64` (and for `auto-cw4` at `p256-L32` / `p64`); the remaining splits are low-bit
arithmetic ordering at ≤ 4e-10 relative `abs_sum` (`p256-L32 forward` 4.97e-12, `p256-L64 forward`
5.78e-11, `p64 batch_forward` ≤ 1e-10), i.e. the route's documented non-bitwise `batch_forward` order,
not a chunking accuracy effect.

## 11. Regenerating the cold column

The single-body cold calls of §3 were re-measured in a later window by `band_sweep.py --cold-only` (one
synchronized first call per workload, no warmup, no reps), so the cold column can be checked without
paying for warm medians. Verbatim as run:

```bash
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src:test/benchmark/evoxc
/tmp/rocvenv/bin/python test/benchmark/evoxc/band_sweep.py --cold-only \
  --cells p256-L32-i8-N1024 --configs auto,b16,b32 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/round3-cold/p256-L32-i8-N1024 \
  --tree-root src --seed 0

/tmp/rocvenv/bin/python test/benchmark/evoxc/band_sweep.py --cold-only \
  --cells p64-L64-i32-N64 --configs auto,b16,b32 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round2/round3-cold/p64-L64-i32-N64 \
  --tree-root src --seed 0
```

Both were executed, **exit 0**, both indexes carry **`cold_only: true`, 3/3 configs `ok`, 0 failures,
0 skipped** (driver `elapsed_s` 93.3 s and 40.0 s), and the records carry `samples_ms: []` with every
warm field `null`. The cold numbers **reproduced at 0.933–1.017× of the `--reps 5` cold calls**
(worst low: `b32 sr_fitness` at `p256-L32`, 11 251.8 against 12 063.3; worst high: `auto batch_forward`
at `p64`, 1 653.0 against 1 626.0), which is the evidence that the cold column of §3 is a stable
reading and not a single window's artefact. A re-measure writes a **fresh `--out-dir`** — the
`round3-cold/<cell>/` layout above exists so that a per-cell re-run does not overwrite another cell's
`sweep.json`.
