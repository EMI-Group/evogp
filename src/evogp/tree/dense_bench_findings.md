# Dense vs specialized on iree-cuda — FIRM-UP measurements and corrected verdict

Scope: the numbers in `dense_bench_results.json`. All dense cells and all numpy
specialized cells were already direct; the iree specialized cells at pop
64/200/1000 were extrapolated from `_sample_model`. This round root-caused a ~3x
discrepancy in that model, took direct full-population iree measurements, corrected
the extrapolations, and re-derived the dense-vs-specialized ratios.
Box: RTX A6000 (GPU 4, `CUDA_VISIBLE_DEVICES=4`), `PYTHONPATH=src`,
`ETL_DISABLE_LOCATIONS=1`, iree backend, `L=32`, seed 0, one row count per process.

## 1. Root cause: per-call overhead amortization (decisive, order-independent)

`_sample_model` measured ONE tree per `Forest.SR_fitness` call. Every such call is a
whole call: it pays the per-call overhead (input staging, the device upload of the
input block, the result readback) once. A real population pays it once *in total*,
not once per tree. `--stage probe` (one process, P16, same forest) decomposes it:

| part (iree, P16) | N=64 | N=1024 |
|---|---|---|
| single-tree call (prefix pop=1) | 3.810 ms | — |
| single-tree call, cache hit | 3.302 ms | 4.660 ms |
| per-call overhead (`const` tree, program=None) | 0.741 ms | — |
| fit: per-call intercept / per-tree slope | 1.426 / 1.876 ms | 3.068 / 1.592 ms |
| prefix sweep pop16 / pop8 / pop4 / pop2 | 20.76 / 14.94 / 7.39 / 4.20 ms | — |
| anchor model P16 (old method) vs real per tree | 3.05x | 2.48x |
| numpy control: intercept / slope / single call | 0.343 / 1.113 / 1.456 ms | — |

A tree declared "1.876 ms warm" by the prefix fit measures 3.302 ms as a
single-tree call — the difference *is* the per-call term. That explains the whole
2.2–3.0x inflation (measured `model_over_direct_warm` 2.35–3.05 on iree, 1.29 on
numpy at P64 where the per-call term is small).

Hypotheses tested and rejected with numbers: **GPU contention** — the reversed probe
order (sample first, prefix second) reproduces the same intercept/slope, and the
in-process full-population anchor at P16 (1.34–1.71 ms/tree) matches the fit slope,
not the sample; **`clear_cache()`/`gc.collect()` perturbation** — the `identical`
part (16 copies of one tree, one executable, no clearing) shows the same
pop-proportional growth, and `single_hit` (no clearing) is the *largest* per-call
number; **harness/timing differences** — `_timed`/`_time_one` use the same median
definition everywhere, and the fully independent `--stage real` runs land within
±17% of the anchor.

## 2. Direct full-population specialized-iree measurements (all `verify=OK`)

| pop | N | cold ms (compile) | warm ms | ms/tree | builds |
|---|---|---|---|---|---|
| 16 | 64 | 4811.9 (4784.5) | 27.409 | 1.713 | 13 |
| 16 | 64 (earlier run) | 4770.7 (4749.3) | 21.485 | 1.343 | 13 |
| 16 | 1024 | 30687.3 (30665.4) | 21.859 | 1.366 | 13 |
| 16 | 10000 | 4606.3 (4579.3) | 26.974 | 1.686 | 13 |
| 24 | 64 / 1024 / 10000 | 20653.5 / 35538.4 / 7620.3 | 34.272 / 34.505 / 36.979 | 1.428 / 1.438 / 1.541 | 19 |
| 32 | 64 / 1024 | 48572.4 / 14216.4 | 43.898 / 55.326 | 1.372 / 1.729 | 26 |
| 40 | 64 | 14670.7 | 56.676 | 1.417 | — |
| 48 | 64 | 14777.6 | 74.174 | 1.545 | — |
| 64 | 64 | 22103.8 (22024.3) | 79.504 | 1.242 | 55 |
| 64 | 1024 | 20990.0 (20900.0) | 89.997 | 1.406 | 55 |
| 200 | 64 | 62439.1 | 336.897 | 1.684 | — |
| 200 | 10000 | 203819.1 | 388.479 | 1.942 | 170 |

Per-tree warm is rows-flat (1.24–1.94 ms, median 1.556 at N=64, 1.422 at N=1024,
1.686 at N=10000) and pop-flat within the 30–50% host/device contention spread, so
`warm_ms ≈ P * per_tree(N)` is confirmed by direct evidence from P16 to P200
(direct/model agreement within ±17%). Compile is ~505.8 ms per distinct program
(median of 11 direct cells) — P16–P200 need 13–170 builds, so cold runs cost
6–204 s.

**Leak finding (corrects the documented limit):** the "~25 live executables"
threshold is a **teardown-only** crash. P64 (55 builds) wrote the correct JSON and
only then hit `nanobind ... reference counting issue` at interpreter shutdown
(exit 0); P200 (170 builds) completed cleanly. Now recorded, why every pop is
staged in its own process anyway.

## 3. Corrected dense-vs-specialized ratios (warm)

Direct dense (same population, `--pool-pop 64`): P16/N64 8.86 ms (cold 13813.1,
1 build), P16/N1024 15.76, P16/N10000 31.28, P32/N64 13.89, P32/N1024 18.52.
P32/N10000 was **skipped by the device-memory guard** (8751 MB predicted vs 8000 MB
limit) and is now labeled `skipped_device_memory` instead of a fitted 1.955 ms.

Same-process same-forest A/B (`--stage probe`), the strongest evidence here:

| config | specialized | dense | ratio | verdict |
|---|---|---|---|---|
| P16 N64 | 22.64 ms | 15.24 ms | **1.49x** | dense wins |
| P16 N1024 | 22.58 ms | 17.38 ms | **1.30x** | dense wins |
| P16 N10000 | 25.12 ms | 29.51 ms | **0.85x** | dense 1.18x SLOWER |

Cross-process pairs (direct dense vs direct specialized): P16/N64 27.41/8.86 =
3.09x, P16/N1024 21.86/15.76 = 1.39x, P16/N10000 26.97/31.28 = 0.86x, P32/N64
43.90/13.89 = 3.16x, P32/N1024 55.33/18.52 = 2.99x. With the anchored-law
specialized values at higher pops: P64 ~1.556×64 = 99.6 ms / dense 15.96 (grid) =
6.2x, P200 336.9/10.60 = 31.8x, P1000 1556/2.30 = 676x — but those dense cells at
P≥64 come from the earlier grid (partly unit-cost fits), so they are **not
independently verified** this round.

**Verdict on the ">=10x dense warm speedup on iree-cuda" headline: NOT supported.**
The old 16.6x at P64/N64 was produced by the inflated 3x model; the corrected value
is ~5–6x and even that uses a fitted dense cell. Where BOTH sides are direct, dense
is 1.3–3.2x faster at N=64/1024 and 1.18x *slower* at N=10000. The honest claim is:
on iree-cuda dense wins the *cold* phase by orders of magnitude (one compile for the
whole population vs 0.5 s per distinct tree: 13.8 s vs 3.5–48.6 s at P16–P32) and
wins the *warm* phase modestly at small N; >=10x warm requires large pops
(P>=200) with many distinct trees, which is plausible from per-tree scaling but not
directly measured.

## 4. Scaling laws and crossover N

* dense: rows- and pop-linear (one graph, `(P_chunk, N_chunk)` execution) — P16
  slope ≈ 2.2e-3 ms/row (N64→1024) and 1.35e-3 ms/row (1024→10000).
* specialized: pop-linear, rows-flat (per-call ~2 ms + ~1.5 ms/tree, independent of
  N up to 10000).
* Crossover (dense warm = specialized warm): P16 ≈ 3.4–6.8k rows (from the two
  slopes), P32 ≈ 7.5–8.4k rows, P64 ≈ 11k rows — i.e. **dense never wins above
  roughly 10k rows**, while below ~3k rows it wins at every measured pop.

## 5. Policy recommendation

Implemented (direct measurement at P16/N10000 shows dense 1.18x slower, i.e. above
the 1.1x rule): `dense_engine.AUTO_MAX_ROWS_ENV = EVOGP_DENSE_AUTO_MAX_ROWS`,
`DEFAULT_AUTO_MAX_ROWS = 2048` (~2x below the tightest measured crossover),
`auto_rows_decline()` + `should_use_dense(..., rows=...)`, enforced inside
`evaluate_dense` for `per_row=False` (the wired call sites pass only `pop_size`).
`per_row=True` is never gated, an explicit `EVOGP_EVAL_MODE=dense` is never
second-guessed, and the decline surfaces as `DenseUnsupported` (normal fallback).
Validated: `dense_integration_check.py --stage iree` 7/7,
`--stage numpy,fallback` 118/118.

## 6. Uncertainties / what remains

* P1000 (793 distinct compiles) was never attempted: the JSON carries an
  anchored-law extrapolation (source `extrapolated(anchored law)`, fields
  `anchor_law` / `model_warm_ms_raw_sample`), not a measurement.
* Direct dense at P>=64 and all dense ratios above ~5x still rest on the earlier
  grid cells (some `fit(unit-cost)`); P32/N10000 is device-guard-skipped.
* Foreign tenants keep GPU 4 at 39–79% utilization: repeat spreads of 30–50% are
  normal, so single numbers are only comparable within one process.
* The direct P64/N64 cell (79.50 ms, 1.242 ms/tree) was recovered from the run log
  after a later model pass overwrote it (`provenance` in the JSON).
* numpy cells were left untouched (they were already direct measurements).
