# The opt-in `batch` evaluation route — measurement report of record

Verdict in one paragraph: the `EVOGP_EVAL_MODE=batch` route (one merged `evoxc` graph per population
structure multiset) is **not the best arm on the main grid** (pop 16–1000 × `tree_len` 32/64) except one
narrow corner, and it is **dominated by the shipped `dense` route at every pop ≥ 64 cell of that grid**.
Warm it is 1.0–2.7× slower than `dense+iree` at pop 256–1000 (0.55× / 0.995× / 0.37× / 0.94× of dense's
speed) while still being 18–30× faster than `specialized+numpy`; cold its merged-graph compile was 10.9–12.2 s
at pop 256 and 91–162 s at pop 1000 against dense's 3.4–6.5 s on the pre-round single-executable path, and is
4.4 s at pop 256 with the shipped chunked pool (§13); and in a real generation loop a crossover/mutation-
updated population is a **new structure multiset**, so the merged graph is rebuilt almost
every generation (1 hit in 8 generations, 99.4–99.7 % of generation time inside the compile), which makes
`gp_step` 193–2355× slower than `dense`. The one cell where it wins on that grid is pop 64 with 64 data rows
(`0.979 ms` vs dense's `1.410 ms`, and a *cheaper* cold call: 2.17 s vs 6.18 s) — and even there it
loses in a generation loop. `auto` never selects the route, so nothing on the default path changes.

A second regime does beat dense and is measured in **§11**: long-tree populations whose longest tree exceeds
the dense trace cap, where batch is 1.5–4.2× faster than dense with the cap raised and 23.5–25× faster than
the per-tree fallback the shipped cap forces. It is a fixed-population win only (§11.6).

This document is the batch route's report of record; `FINAL.md` remains the report of record for the
dense/specialized engine-vs-torch campaign and is untouched.

## 1. What the route is

`EVOGP_EVAL_MODE=batch` is the third evaluation route (`src/evogp/tree/batch_engine.py`, built on
`evoxc.batch_functions`). Instead of one compiled graph per tree structure (`specialized`) or one
all-ops graph per population shape (`dense`), it traces each distinct program once, merges the whole
population's programs into **one graph per population structure multiset** (banded: `ceil(n/size)`
bands of programs), compiles that merged graph with `iree` (a `numpy` fallback exists) and runs the
whole population as a single invocation.

Consequences that decide every number below:

- the compile cost is per **population**, not per tree or per shape — so it is re-paid whenever the
  multiset changes, which in a GP loop is every generation;
- the warm per-call cost does not grow with the number of trees the way `specialized` does (one
  invocation per population instead of one per tree structure), but it also collapses many programs
  into one longer graph, which is why `dense` — one graph for *all* populations of a shape, with the
  tree structure kept as runtime data — stays cheaper;
- every "this route cannot handle it" answer is a `BatchUnsupported` decline that falls back to
  `specialized`, including the population floor (`EVOGP_BATCH_MIN_POP`, default 64);
- `auto` never resolves to `batch`; the route is opt-in only.

## 2. Knobs

All single-sourced in `src/evogp/tree/batch_knobs.py`, re-read by the harness from the engine's own
resolvers (`common.BATCH_KNOBS` / `common.batch_policy()`, so a record states the *resolved* value and
its provenance, not a re-derivation).

| env var | default | meaning |
|---|---|---|
| `EVOGP_EVAL_MODE` | `auto` | `batch` selects this route; `auto` never does |
| `EVOGP_BATCH_BAND_SIZE` | `32` | programs per band: `K = ceil(n / size)` bands, clamped to `1 <= K <= n` |
| `EVOGP_BATCH_BANDS` | unset | an explicit band count `K`, or `none`/`flat`/`0` for one unbanded merge (`K = 1`, O(N²) in the population) |
| `EVOGP_BATCH_SWITCH_OPS` | `off` | whether the merge may use switch ops; when on it forces the flat merge |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | class alignment of a `switch_ops` merge (a pure evoxc heuristic, never changes numbers) |
| `EVOGP_BATCH_CACHE_SIZE` | `8` | entries of the two front tables (`BATCH_EXEC_MEMO` = compiled executables, `BATCH_PLAN_MEMO` = host decode + traces + merge); read at import time |
| `EVOGP_BATCH_MIN_POP` | `64` | population floor; below it the route declines to `specialized` |

`batch_engine.batch_cache_stats()` returns exactly eight counters, and **no bare `misses` key** — which
is what the harness originally assumed and now resolves correctly:

`batch_exec_hits` / `batch_exec_misses` / `batch_exec_entries` / `batch_exec_maxsize` (the real compile:
a hit never reaches `evoxc.lower`/`compile`/`load`) and
`batch_plan_hits` / `batch_plan_misses` / `batch_plan_entries` / `batch_plan_maxsize` (host decode, traces
and merge). Records now carry `batch_cache.compile_misses_*` (exec), `batch_cache.plan_misses_*` and
`batch_cache.stats_keys`; the `EVOGP_EVOXC_*` `cache_stats()` "misses" counter is a *different* cache
(the specialized engine's per-structure `CompileCache`) and is recorded separately.

Cache-size arithmetic worth stating once: with `pop 256` and `BAND_SIZE 32` a population spans
`K ≈ 7–8` exec entries, i.e. **one population exactly fills the default 8-entry table**. Two consecutive
populations therefore cannot both stay resident, which is the mechanism behind the measured 1/8 hit rate.

## 3. What was measured, and on which box

- Box: AMD Instinct MI300X VF (192 GB), ROCm 7.1, `torch 2.13.0+rocm7.1`, CPython 3.14.8,
  `evoxc 0.0.0+unknown` from `/root/evoxc`. Session date 2026-10-29.
  These are this box's numbers; `FINAL.md`'s dense/specialized figures come from a different session, so
  compare route *ratios* within this document rather than absolute ms across documents.
- Protocol: `EVOXC_DISABLE_LOCATIONS=1`, deterministic population (`torch.manual_seed(0)` +
  `Forest.random_generate`), one process per `(cell, arm)`, one config at a time behind the idle-device
  guard, `--warmup 1 --reps 5` per workload (the harness's `MIN_SAMPLES` floor), `--gp-warmup 1
  --gp-reps 3`, cold first call timed separately from the warm medians.
- Arms: `torch-per-tree`, `torch-batched` (plain-PyTorch baselines), `evoxc-specialized-numpy`,
  `evoxc-dense-iree`, `evoxc-auto` (the shipped policy), `evoxc-batch-iree` (default banded merge) and
  `evoxc-batch-iree-flat` (`EVOGP_BATCH_BANDS=flat`).
- Cells: `pop ∈ {16, 64, 256, 1000}` × `tree_len ∈ {32, 64}` × `input_len ∈ {8, 32}` at N=1024 rows, plus
  the N=64 column (`L64/i32`) at every pop.
- Evidence directories (all under `results/batch/`): `warm-r5/` (4 large cells × 6 arms),
  `small-r5/` (p16/p64 at N=1024 and all four N=64 cells × 6 arms), `flat-r5/` (p256 pair, banded vs
  flat), `flat1000-3w-r5/` (p1000-L32, flat merge, three value workloads), `flat1000-r5/`
  (p1000-L32, flat merge, `sr_fitness` only), the compile-economics records (`<cell>/<route>.json` +
  `economics.json`), `validation-batch.json` / `validation-dense.json`, and the `summarize.py`
  renders `RENDERED-*-r5.md`.
  The unsuffixed `warm/`, `small/`, `flat/`, `flat1000/` directories are the same grid measured as a
  **pilot at `--reps 3`**; `check.py`'s `MIN_SAMPLES = 5` makes them structurally ungateable, so they are
  retained verbatim as the pilot and are *not* the numbers quoted here.

## 4. Warm per-call cost — the headline table

`sr_fitness` / `batch_forward` / `forward` medians in ms on the same fixed population, and the cold
`first_call_ms` of `sr_fitness` (for a batch arm that is the merged-graph compile; for dense the all-ops
compile; for specialized+numpy the per-structure compile).

### 4.1 Large cells, N=1024 (`results/batch/warm-r5`)

| cell | arm | sr_fitness | first call | batch_forward | forward | gp_step total | gp eval / ops |
|---|---|---|---|---|---|---|---|
| p256-L32-i8 | torch-per-tree | 30.308 | 155.7 | 23.349 | 23.859 | 17.55 | 15.05 / 2.49 |
| p256-L32-i8 | torch-batched | 3.994 | 63.6 | 3.961 | 3.872 | 22.85 | 20.42 / 2.43 |
| p256-L32-i8 | evoxc-specialized-numpy | 38.596 | 839.8 | 39.515 | 35.828 | 294.37 | 291.85 / 2.51 |
| p256-L32-i8 | evoxc-dense-iree | **1.101** | 3379.3 | 1.866 | 0.686 | **5.36** | 2.89 / 2.48 |
| p256-L32-i8 | evoxc-auto | 1.101 | 3441.9 | 1.702 | 0.662 | 5.26 | 2.85 / 2.42 |
| p256-L32-i8 | **evoxc-batch-iree** | 1.990 | 10879.8 | 2.282 | 0.797 | 3446.20 | 3443.38 / 2.82 |
| p256-L64-i32 | torch-per-tree | 30.601 | 161.4 | 23.461 | 23.896 | 17.11 | 14.65 / 2.46 |
| p256-L64-i32 | torch-batched | 4.466 | 66.3 | 4.358 | 4.375 | 22.40 | 19.87 / 2.54 |
| p256-L64-i32 | evoxc-specialized-numpy | 38.718 | 851.3 | 38.301 | 35.451 | 265.74 | 263.30 / 2.44 |
| p256-L64-i32 | evoxc-dense-iree | **2.152** | 6516.2 | 2.854 | 0.982 | **6.66** | 4.23 / 2.43 |
| p256-L64-i32 | evoxc-auto | 2.058 | 6437.4 | 2.878 | 0.911 | 6.64 | 4.17 / 2.47 |
| p256-L64-i32 | **evoxc-batch-iree** | 2.163 | 12209.0 | 2.156 | 0.739 | 3884.56 | 3881.51 / 3.05 |
| p1000-L32-i8 | torch-per-tree | 120.237 | 246.5 | 95.104 | 96.908 | 60.89 | 58.12 / 2.77 |
| p1000-L32-i8 | torch-batched | 4.429 | 65.0 | 4.408 | 3.919 | 53.44 | 51.00 / 2.44 |
| p1000-L32-i8 | evoxc-specialized-numpy | 149.221 | 3158.1 | 146.260 | 139.685 | 1010.04 | 1007.31 / 2.73 |
| p1000-L32-i8 | evoxc-dense-iree | **1.883** | 3311.7 | 3.812 | 0.664 | **7.24** | 4.85 / 2.39 |
| p1000-L32-i8 | evoxc-auto | 1.883 | 3401.4 | 3.814 | 0.693 | 7.36 | 4.89 / 2.46 |
| p1000-L32-i8 | **evoxc-batch-iree** | 5.043 | 91117.6 | 5.796 | 1.605 | 17052.61 | 17049.65 / 2.96 |
| p1000-L64-i32 | torch-per-tree | 118.725 | 244.7 | 90.746 | 93.848 | 56.17 | 53.46 / 2.71 |
| p1000-L64-i32 | torch-batched | 4.873 | 65.0 | 4.859 | 4.173 | 65.54 | 63.00 / 2.54 |
| p1000-L64-i32 | evoxc-specialized-numpy | 148.023 | 3179.3 | 153.235 | 135.671 | 859.45 | 856.70 / 2.75 |
| p1000-L64-i32 | evoxc-dense-iree | **5.764** | 6454.6 | 7.534 | 0.924 | **12.35** | 9.95 / 2.41 |
| p1000-L64-i32 | evoxc-auto | 5.726 | 6438.3 | 7.615 | 0.896 | 12.46 | 9.99 / 2.47 |
| p1000-L64-i32 | **evoxc-batch-iree** | 6.162 | 161967.8 | 5.700 | 1.343 | 17524.31 | 17521.34 / 2.97 |

Batch speedups from this table (batch ÷ arm, >1 = batch faster):

| cell | vs torch-per-tree | vs torch-batched | vs specialized+numpy | vs dense+iree |
|---|---|---|---|---|
| p256-L32-i8 | 15.2× | 2.01× | 19.4× | **0.55×** |
| p256-L64-i32 | 14.1× | 2.06× | 17.9× | **0.995×** |
| p1000-L32-i8 | 23.8× | 0.88× | 29.6× | **0.37×** |
| p1000-L64-i32 | 19.3× | 0.79× | 24.0× | **0.94×** |

`batch_forward` and `forward` tell the same story (e.g. p1000-L32: batch 5.80 / 1.61 ms vs dense
3.81 / 0.66 ms), so the route's warm disadvantage is not an artefact of the loss variant.

### 4.2 Small populations and the N=64 column (`results/batch/small-r5`)

N=1024 cells at p16/p64 and every `L64/i32` cell at N=64 rows.

| cell | arm | sr_fitness | first call | gp_step total | gp eval / ops | batch exec misses (gp) |
|---|---|---|---|---|---|---|
| p16-L32-i8-N1024 | torch-per-tree | 1.867 | 122.0 | 3.18 | 0.91 / 2.27 | – |
| p16-L32-i8-N1024 | torch-batched | 2.877 | 158.0 | 5.12 | 2.76 / 2.36 | – |
| p16-L32-i8-N1024 | evoxc-specialized-numpy | 2.308 | 85.5 | 8.21 | 5.72 / 2.49 | – |
| p16-L32-i8-N1024 | evoxc-dense-iree | **0.988** | 3382.6 | 3015.83 | 3012.94 / 2.89 | – |
| p16-L32-i8-N1024 | evoxc-auto | 2.467 | 81.1 | 8.04 | 5.63 / 2.42 | – |
| p16-L32-i8-N1024 | evoxc-batch-iree *(declines)* | 3.939 | 3696.6 | 782.16 | 779.40 / 2.76 | 0 |
| p64-L32-i8-N1024 | torch-per-tree | 7.465 | 133.2 | 5.55 | 3.17 / 2.39 | – |
| p64-L32-i8-N1024 | torch-batched | 3.680 | 60.0 | 7.82 | 5.35 / 2.47 | – |
| p64-L32-i8-N1024 | evoxc-specialized-numpy | 9.865 | 221.1 | 35.83 | 33.64 / 2.19 | – |
| p64-L32-i8-N1024 | evoxc-dense-iree | **0.984** | 3405.3 | **4.85** | 2.45 / 2.41 | – |
| p64-L32-i8-N1024 | evoxc-auto | 0.974 | 3350.2 | 4.84 | 2.45 / 2.39 | – |
| p64-L32-i8-N1024 | **evoxc-batch-iree** | 1.131 | 2862.8 | 937.23 | 934.58 / 2.65 | 3 |
| p16-L64-i32-N64 | torch-per-tree | 1.868 | 122.5 | 3.34 | 0.94 / 2.40 | – |
| p16-L64-i32-N64 | torch-batched | 3.159 | 151.9 | 6.01 | 3.64 / 2.38 | – |
| p16-L64-i32-N64 | evoxc-specialized-numpy | 2.212 | 66.2 | 7.88 | 5.54 / 2.34 | – |
| p16-L64-i32-N64 | evoxc-dense-iree | **1.547** | 6479.2 | 5791.78 | 5788.95 / 2.83 | – |
| p16-L64-i32-N64 | evoxc-auto | 2.170 | 62.6 | 7.13 | 4.94 / 2.19 | – |
| p16-L64-i32-N64 | evoxc-batch-iree *(declines)* | 3.933 | 3628.1 | 753.80 | 751.05 / 2.75 | 0 |
| p64-L64-i32-N64 | torch-per-tree | 7.418 | 130.3 | 6.00 | 3.65 / 2.35 | – |
| p64-L64-i32-N64 | torch-batched | 4.035 | 59.0 | 10.14 | 7.83 / 2.31 | – |
| p64-L64-i32-N64 | evoxc-specialized-numpy | 9.572 | 199.2 | 104.64 | 102.34 / 2.30 | – |
| p64-L64-i32-N64 | evoxc-dense-iree | 1.410 | 6177.7 | **5.31** | 2.88 / 2.43 | – |
| p64-L64-i32-N64 | evoxc-auto | 1.503 | 6283.7 | 5.37 | 3.00 / 2.37 | – |
| p64-L64-i32-N64 | **evoxc-batch-iree** | **0.979** | **2173.5** | 1053.64 | 1050.92 / 2.72 | 3 |
| p256-L64-i32-N64 | torch-per-tree | 28.173 | 152.1 | 18.53 | 16.24 / 2.29 | – |
| p256-L64-i32-N64 | torch-batched | 4.525 | 62.8 | 27.76 | 25.33 / 2.44 | – |
| p256-L64-i32-N64 | evoxc-specialized-numpy | 36.463 | 801.3 | 309.69 | 307.26 / 2.44 | – |
| p256-L64-i32-N64 | evoxc-dense-iree | **1.442** | 6345.6 | **6.15** | 3.65 / 2.50 | – |
| p256-L64-i32-N64 | evoxc-auto | 1.434 | 6268.0 | 6.04 | 3.54 / 2.50 | – |
| p256-L64-i32-N64 | **evoxc-batch-iree** | 2.067 | 9938.8 | 3628.13 | 3625.27 / 2.86 | 3 |
| p1000-L64-i32-N64 | torch-per-tree | 117.159 | 244.9 | 54.09 | 51.54 / 2.56 | – |
| p1000-L64-i32-N64 | torch-batched | 4.548 | 65.0 | 60.97 | 58.25 / 2.72 | – |
| p1000-L64-i32-N64 | evoxc-specialized-numpy | 142.507 | 3135.2 | 1019.97 | 1017.15 / 2.82 | – |
| p1000-L64-i32-N64 | evoxc-dense-iree | **1.613** | 6397.4 | **8.25** | 5.78 / 2.48 | – |
| p1000-L64-i32-N64 | evoxc-auto | 1.555 | 6308.1 | 8.22 | 5.76 / 2.47 | – |
| p1000-L64-i32-N64 | **evoxc-batch-iree** | 6.082 | 158047.4 | 15088.60 | 15085.53 / 3.06 | 3 |

Batch speedups in this subtree: vs dense `0.87×` (p64/L32/N1024), **`1.44×` (p64/N=64 — the only win)**,
`0.70×` (p256/N=64), `0.27×` (p1000/N=64); vs specialized+numpy `8.7×` / `9.8×` / `17.6×` / `23.4×`;
vs torch-batched `3.25×` / `4.12×` / `2.19×` / `0.75×`.

Two rows deserve their own note:

- **pop 16 is the decline.** `EVOGP_BATCH_MIN_POP = 64` is the engine's measured break-even, and the
  records state it: `routes.*.would_use_batch = false`, `declines_to_specialized = true`, reason
  *"pop 16 < EVOGP_BATCH_MIN_POP 64: one merged graph cannot amortize its build"*, 0 exec misses.
  Asking for the route anyway is not free: the batch arm's fallback costs 3.94 ms vs 2.31 ms for the
  same `specialized+numpy` arm measured directly (≈1.6 ms of per-call overhead on the decline path).
- **`evoxc-dense-iree` at pop 16 is a trap** (`gp_step` 3015.8 ms / 5791.8 ms): the dense graph is
  keyed per *shape × used-function-set*, and a 16-tree population's used-function set changes as it
  evolves, so 2 of 3 generations paid a fresh ~3 s compile. `auto` avoids it by routing pop < 64 to
  `specialized`; the explicit dense arm does not. This is engine behaviour, not a batch-route finding.

### 4.3 Flat (unbanded `K = 1`) merge

| cell | arm | sr_fitness | first call | batch_forward | forward |
|---|---|---|---|---|---|
| p256-L32-i8-N1024 | evoxc-batch-iree (banded) | 2.131 | 10639.0 | 2.160 | 0.831 |
| p256-L32-i8-N1024 | evoxc-batch-iree-flat | 2.123 | 12104.7 | 2.310 | 1.102 |
| p256-L64-i32-N1024 | evoxc-batch-iree (banded) | 2.108 | 12168.1 | 2.221 | 0.734 |
| p256-L64-i32-N1024 | evoxc-batch-iree-flat | 2.413 | 14904.5 | 2.465 | 0.981 |
| p1000-L32-i8-N1024 | evoxc-batch-iree-flat | 7.619 | 102733.6 | 8.796 | 2.793 |

The flat merge costs the same order of compile as the banded one at pop 256 (12.1–14.9 s vs 10.6–12.2 s)
and is no faster warm, so banding is not the route's problem: **the per-call cost of the merged graph
itself is what loses to dense**, at every row count. At pop 1000 the flat merge compiles 102.7 s for the
loss graph, 57.8 s for the value graph and 28.5 s for the per-row graph, and its warm `sr_fitness`
(7.62 ms) is 4.0× slower than dense (1.88 ms).

## 5. Generation-level view — the workload that decides usability

`gp_step` is one full generation (evaluate + select + crossover + mutate), three timed generations per
record. The batch arms record `batch_compile_misses_total = 3`: **every generation merged and compiled a
fresh graph** (a new population is a new structure multiset), while `dense`/`auto` record 0 compile
misses at pop ≥ 64 (same shape, same used-function-set) and `specialized` reuses its per-structure cache.

| cell | dense gp_step (eval / ops) | batch gp_step (eval / ops) | batch ÷ dense |
|---|---|---|---|
| p256-L32-i8-N1024 | 5.36 ms (2.89 / 2.48) | 3446.20 ms (3443.38 / 2.82) | 642× |
| p256-L64-i32-N1024 | 6.66 ms (4.23 / 2.43) | 3884.56 ms (3881.51 / 3.05) | 584× |
| p1000-L32-i8-N1024 | 7.24 ms (4.85 / 2.39) | 17052.61 ms (17049.65 / 2.96) | 2355× |
| p1000-L64-i32-N1024 | 12.35 ms (9.95 / 2.41) | 17524.31 ms (17521.34 / 2.97) | 1419× |
| p64-L32-i8-N1024 | 4.85 ms | 937.23 ms | 193× |
| p64-L64-i32-N64 (its one warm win) | 5.31 ms | 1053.64 ms | 198× |
| p256-L64-i32-N64 | 6.15 ms | 3628.13 ms | 590× |
| p1000-L64-i32-N64 | 8.25 ms | 15088.60 ms | 1829× |

The operator layer is identical across arms (2.2–3.1 ms per generation), so the whole difference is the
evaluation half. **The route's one warm win disappears in a loop**: even at p64/N=64, where batch is the
fastest evaluator on a fixed population, a generation costs 1.05 s against dense's 5.3 ms.

## 6. Compile economics (`batch_economics.py`, 8 generations at p256, 4 at p1000)

Each child process builds the population, times ONE first call, times warm `sr_fitness` medians, then
runs a real `StandardPipeline.step()` loop recording, per generation, the population's structure-multiset
signature **before** the step, whether that signature was seen before, both caches' misses and the
wall/eval/ops split.

| cell | route | first call | warm sr_fitness | generation hit rate | generations with a compile | compile share of generation | median cold eval | median warm eval |
|---|---|---|---|---|---|---|---|---|
| p256-L32-i8 | batch | 11033.7 ms | 1.936 ms | 1/8 (0.125) | 7/8 | **99.42 %** | 5734.1 ms | 1.99 ms |
| p256-L32-i8 | dense | 3421.1 ms | 1.098 ms | – | 0/8 | 0 % | – | 2.88 ms |
| p256-L64-i32 | batch | 12393.3 ms | 2.087 ms | 1/8 (0.125) | 7/8 | **99.51 %** | 6834.8 ms | 2.14 ms |
| p256-L64-i32 | dense | 6627.5 ms | 2.038 ms | – | 0/8 | 0 % | – | 4.10 ms |
| p1000-L32-i8 | batch | 91651.1 ms | 5.013 ms | 1/4 (0.25) | 3/4 | **99.65 %** | 17015.7 ms | 5.31 ms |
| p1000-L32-i8 | dense | 3405.6 ms | 1.906 ms | – | 0/4 | 0 % | – | 4.77 ms |
| p1000-L64-i32 | batch | 169386.1 ms | 6.215 ms | 1/4 (0.25) | 3/4 | **99.65 %** | 18166.3 ms | 6.28 ms |
| p1000-L64-i32 | dense | 6774.3 ms | 5.649 ms | – | 0/4 | 0 % | – | 10.27 ms |

Generation 1 hits by construction (`StandardPipeline.step()` evaluates the population whose graph the
timed first call already compiled) — that is the only hit the route ever gets. Generations 2..N are cold
misses of 1.5–7.5 s (p256) and 12–48 s (p1000), and they are not a first-generation artefact: later
p256 generations *grow* (5.7 → 3.1 → 3.7 → 4.3 → 7.5 → 10.2 → 10.0 s) as crossover/mutation makes the
population's program set larger and less duplicate-heavy.

**Break-even.** There is none against dense: batch pays 7.6–10.5 s (p256) or 88–163 s (p1000) more
compile up front and is *also* slower per warm call, so no number of calls recovers it. Against
`specialized+numpy` the warm saving is real (36.6 ms per call at p256-L32, 143 ms at p1000) but needs
≈300 calls just to repay the p256 compile — and `auto` never chooses specialized at pop ≥ 64, where dense
is both cheaper to compile and faster per call. In the only comparison that governs the shipped policy
(batch vs dense), the route is dominated in all three dimensions: cold compile, warm call and loop cost.

## 7. The first regime where the route wins

`pop 64 × 64 data rows`, fixed population (`results/batch/small-r5/p64-L64-i32-N64`):

| arm | sr_fitness | first call | batch_forward | forward |
|---|---|---|---|---|
| evoxc-batch-iree | **0.979 ms** | **2173 ms** | **0.789 ms** | **0.631 ms** |
| evoxc-dense-iree | 1.410 ms | 6178 ms | 1.422 ms | 0.868 ms |
| evoxc-auto | 1.503 ms | 6284 ms | 1.422 ms | 0.839 ms |
| evoxc-specialized-numpy | 9.572 ms | 199 ms | 8.674 ms | 9.158 ms |
| torch-batched | 4.035 ms | 59 ms | 4.002 ms | 3.902 ms |
| torch-per-tree | 7.418 ms | 130 ms | 5.656 ms | 5.855 ms |

There batch is the fastest arm on every value workload *and* has the cheapest device compile
(2.2 s vs dense's 6.2 s), because 64 programs merge into 2 bands of a short (L=64) graph while dense
must still build its all-ops chain. It does **not** survive the generation loop (1053.6 ms per generation
vs 5.31 ms for dense) and it does not extend to larger populations: at p256/N=64 batch is 0.70× dense and
at p1000/N=64 0.27×. The rule the numbers support: *a fixed population of ~64 trees over few rows* is
the route's niche — e.g. repeated scoring of an evolved elite set, a hyper-parameter sweep re-evaluating
one population, or a study that holds the population still. A GP loop is never that.

A **second regime** is measured in §11: a fixed population whose longest tree exceeds the dense trace cap
(`EVOGP_DENSE_MAX_TREE_LEN`, default 64). There batch is 1.5–4.2× faster than dense with the cap raised and
23.5–25× faster than the per-tree `specialized` fallback the shipped cap forces. §11.6 states the limits of
that win — it is a fixed-population win only, and it pays the largest cold compile of any arm measured here.

## 8. Numeric validation

`validate.py` compares every arm against the independent CPU oracle `src/evogp/tests/reference_eval.py`
(64 trees per cell, three function sets); the batch run sets `EVOGP_EVAL_MODE=batch
EVOGP_EVOXC_BACKEND=iree`, so the records prove the route was actually taken (`eval_mode: batch`,
`kind: batch`, `would_use_batch: true`) — not silently declined.

| route | cell | func set | `sr_fitness` max rel | `batch_forward` abs / rel (bitwise) | `forward` max rel (bitwise) |
|---|---|---|---|---|---|
| batch/iree | p256-L32-i8 | `arith` | 4.79e-07 (ref 2.8e9, abs 256) | 1.56e-02 / 1.13e-03 (no) | 1.64e-07 (no) |
| batch/iree | p256-L32-i8 | `rich` | 1.90e-07 (ref 2.0e4) | 1.346 / 7.65 (no) | 9.80e-08 (no) |
| batch/iree | p256-L32-i8 | `ternary` | 3.16e-07 (ref 4.3e5) | exact (yes) | exact (yes) |
| batch/iree | p256-L64-i32 | `arith` | 7.76e-07 (ref 3.0e9, abs 64) | 7.81e-02 / 2.38e-04 (no) | 1.09e-07 (no) |
| batch/iree | p256-L64-i32 | `rich` | 1.48e-07 (ref 2.9e6) | 1.610 / 15.61 (no) | exact (no) |
| batch/iree | p256-L64-i32 | `ternary` | 2.54e-07 (ref 7.5e4) | 4.77e-07 / 1.99e-04 (no) | 1.02e-07 (no) |
| dense/iree | p256-L32-i8 | `arith` | 3.11e-07 (ref 2.8e9, abs 256) | **exact** (yes) | **exact** (yes) |
| dense/iree | p256-L32-i8 | `rich` | 1.90e-07 (ref 2.0e4) | 1.346 / 7.65 (no) | **exact** (no) |
| dense/iree | p256-L32-i8 | `ternary` | 3.16e-07 (ref 4.3e5) | **exact** (yes) | **exact** (yes) |
| dense/iree | p256-L64-i32 | `arith` | 3.43e-07 (ref 3.0e9, abs 32) | **exact** (yes) | **exact** (yes) |
| dense/iree | p256-L64-i32 | `rich` | 2.95e-07 (ref 2.9e6) | 1.610 / 15.61 (no) | **exact** (no) |
| dense/iree | p256-L64-i32 | `ternary` | 2.08e-07 (ref 7.5e4) | **exact** (yes) | **exact** (yes) |

- **Agreement is good enough to trust the timings**: the loss (`sr_fitness`) path agrees to 4.8e-07 /
  7.8e-07 relative at magnitudes up to 3e9 (dense: 3.1e-07 / 3.4e-07 — the same order), and the per-row
  `forward` path to ≤1.6e-07 relative on both routes. No NaN/Inf **placement** mismatch anywhere in the
  12 cases (the `rich` spread carries 5813 / 5804 NaN points on both routes identically).
- **But the batch `batch_forward` value path is not bitwise on `arith`, where dense is.** Deviations:
  1.5625e-02 abs (1.13e-03 rel, reference magnitude 1.35e6) on p256-L32 and 7.81e-02 abs (2.38e-04 rel,
  reference 1.70e6) on p256-L64, while dense's own `batch_forward` is bitwise exact in both cells. The
  ternary set shows the same shape of split (batch 4.77e-07 abs / 1.99e-04 rel at p256-L64 vs dense
  exact). The absolute error is ~1e-8 of the value scale, but it exceeds the 1e-4 relative tolerance the
  other routes are held to on those entries, so it is an **engine-side precision caveat, not a harness
  one** (repro:
  `EVOGP_EVAL_MODE=batch EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --func-sets arith,ternary`).
  It is also why the batch arm forms its own sha256 digest group in `check.py`.
- `rich` is identical for both routes (same abs/rel to the shown digits) and is the already-documented
  transcendental spread (`VALIDATION.md`), so it says nothing route-specific here.

## 9. Gates and evidence layout

`check.py` over each result directory with `--allow-missing --allow-digest-mismatch --allow-snapshot`:

| directory | records | checks | failures |
|---|---|---|---|
| `results/batch/warm-r5` | 24 | 682 | **0** |
| `results/batch/small-r5` | 36 | 1022 | **0** |
| `results/batch/flat-r5` | 4 | 126 | **0** |
| `results/batch/flat1000-3w-r5` | 1 | 36 | **0** |
| `results/batch/long-trees` (§11, the four sub-directories in one run) | 38 | 1000 | **0** (380 / 254 / 128 / 238 per directory) |
| `results/batch/flat1000-r5` | 1 | 22 | 2 — *"batch_forward: no digest anywhere in the cell"*, *"forward: no digest anywhere in the cell"* (that directory scheduled `--workloads sr_fitness` only; the 3-workload sibling `flat1000-3w-r5` is the gateable record) |
| pilot `warm`/`small`/`flat`/`flat1000` | 24/36/4/1 | – | one *"warm samples, need >= 5"* per workload entry (recorded at `--reps 3`) |

All arms report `guard: N/N configs on an idle device, 0 contaminated`. Every `check_batch` assertion
passes on every batch record: the mode is declared, the recorded route is `batch`, `batch_cache.available`
is true, and the merged-graph compile is attributed to `batch_exec_misses` (never a null counter).

## 10. Reproduce

From the repository root, with the project venv, one process per timed config, never two campaigns at once:

```bash
PY=/tmp/genesis/task_a996674198ffbcee/evogp-venv/bin/python
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1

# 1. compile economics (batch vs dense, real StandardPipeline.step loop)
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p1000-L32-i8-N1024,p1000-L64-i32-N1024 \
  --routes batch,dense --generations 8 --warmup 2 --reps 5 \
  --out-dir test/benchmark/evoxc/results/batch
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p1000-L32-i8-N1024,p1000-L64-i32-N1024 --routes batch,dense \
  --generations 4 --warmup 2 --reps 5 --out-dir test/benchmark/evoxc/results/batch

# 2. warm campaign, large cells (the table in §4.1)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/warm-r5 \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p1000-L32-i8-N1024,p1000-L64-i32-N1024 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-specialized-numpy,evoxc-batch-iree \
  --workloads sr_fitness,batch_forward,forward,gp_step --warmup 1 --reps 5 --gp-warmup 1 --gp-reps 3 \
  --timeout 2400

# 3. small populations + the N=64 column (§4.2)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/small-r5 \
  --cells p16-L32-i8-N1024,p64-L32-i8-N1024,p16-L64-i32-N64,p64-L64-i32-N64,p256-L64-i32-N64,p1000-L64-i32-N64 \
  --arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree,evoxc-specialized-numpy,evoxc-batch-iree \
  --workloads sr_fitness,batch_forward,forward,gp_step --warmup 1 --reps 5 --gp-warmup 1 --gp-reps 3 \
  --timeout 2400

# 4. banded vs flat merge (§4.3)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/flat-r5 \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --arms evoxc-batch-iree,evoxc-batch-iree-flat \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --timeout 2400
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/flat1000-3w-r5 \
  --cells p1000-L32-i8-N1024 --arms evoxc-batch-iree-flat \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --timeout 2400

# 5. numeric validation of the batch route and of dense, same oracle
EVOGP_EVAL_MODE=batch EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --func-sets arith,rich,ternary \
  --out test/benchmark/evoxc/results/batch/validation-batch.json
EVOGP_EVAL_MODE=dense EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --func-sets arith,rich,ternary \
  --out test/benchmark/evoxc/results/batch/validation-dense.json

# 6. gates and renders (one directory at a time; the gate is per campaign directory)
for d in warm-r5 small-r5 flat-r5 flat1000-3w-r5; do
  $PY test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch/$d \
    --allow-missing --allow-digest-mismatch --allow-snapshot
  $PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/batch/$d \
    --out test/benchmark/evoxc/results/batch/RENDERED-$d.md
done
# expected: 24/682, 36/1022, 4/126, 1/36 checks, 0 failures (see §9)
```

## 11. Long trees: the scaling crossover

Every cell in §4–§6 holds 15–17-node trees (the stock `max_layer_cnt = 5`) against a row stride of 32 or
64. The long-tree round asks the question that grid cannot answer: **does a route's cost track the trees
the population actually holds, or the trace length the route is allowed to emit?**

The two routes sit on different axes:

- `batch` traces each distinct program once and merges the population into one graph, so its emitted op
  count is a function of the population's node total and nothing else — §11.3 makes that literal
  (5200 → 17757 emitted ops as the mean tree grows 15.9 → 66.6 nodes, and 5200 for the same population at
  either stride).
- `dense` emits the all-ops chain over the whole row stride (`EVOGP_DENSE_MAX_TREE_LEN`), independent of
  what the population contains, so its per-call cost is proportional to the cap rather than to the trees.

The trap is that `tree_len` is the **row stride of the tensor schema, not the size of the trees**. With the
stock generation recipe a `tree_len = 128` cell still holds 15.9-node trees with a 31-node maximum — the
same population the `tree_len = 64` cells hold. A full tree is `2**d − 1` nodes, so tree size is
depth-driven. This round therefore pins both axes: `tree_len` 128/256 — plus the `evoxc-dense-iree-raised`
arm, which pins `EVOGP_DENSE_MAX_TREE_LEN` to the cell's stride — is the stride axis, and the `-d<k>`
cells raise the generation depth (mean/max 26.1/61 nodes at d6, 41.8/111 at d7, 66.6/187 at d8) as the
node-count axis. **A "long-tree" cell left at the stock depth cap measures nothing but stride.**

### 11.1 Warm per-call cost at long trees

`sr_fitness` median in ms, N=1024 rows, seed 0, `--reps 5`, one process per config; the parenthesised
value is that arm's cold first call in s (omitted for `torch-per-tree`, which compiles nothing — its first
call is 0.15–0.27 s). `evoxc-dense-iree` is the shipped cap of 64; `evoxc-dense-iree-raised` is the same
route with `EVOGP_DENSE_MAX_TREE_LEN` pinned to the cell's `tree_len` (128 or 256).

| cell | nodes mean / med / max | distinct programs | torch-per-tree | specialized-numpy | dense, cap 64 | dense, cap = tree_len | batch-iree |
|---|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` (d5) | 15.9 / 17 / 31 | 232/256 | 32.203 | 37.606 (0.83) | **2.030** (6.61) | 5.277 (14.44) | 2.167 (13.22) |
| `p256-L128-i32-N1024-d6` | 26.1 / 27 / 61 | 226/256 | 46.444 | 56.056 (1.17) | **2.040** (6.52) | 5.379 (14.28) | 2.881 (22.76) |
| `p256-L128-i32-N1024-d7` | 41.8 / 43 / 111 | 230/256 | 67.882 | 91.122 (1.84) | 88.020 (68.99, *declined*) | **5.343** (14.48) | 3.518 (37.92) |
| `p256-L256-i32-N1024` (d5) | 15.9 / 17 / 31 | 232/256 | 29.815 | 41.449 (0.91) | **2.126** (7.22) | 19.497 (41.87) | 2.237 (12.04) |
| `p256-L256-i32-N1024-d8` | 66.6 / 66 / 187 | 225/256 | 108.797 | 158.373 (3.30) | 110.118 (81.07, *declined*) | **19.662** (43.91) | 4.676 (74.73) |
| `p1000-L128-i32-N1024` (d5, pop 1000) | 16.2 / 17 / 31 | 841/1000 | 131.325 | 164.386 (3.66) | **5.715** (7.00) | 19.576 (15.78) | 6.503 (178.54) |

The two *declined* rows are the reading that matters: at d7 and d8 the population's longest tree (111 and
187 nodes) exceeds the shipped cap, so `evoxc-dense-iree` is not measuring dense at all. Its `dense_route`
block records `supported: false` with the reason *"longest tree is 111 nodes but the dense trace cap is 64
(EVOGP_DENSE_MAX_TREE_LEN=64, row length 128); raise the cap or fall back to the specialized engine"*, and
its 88.020 / 110.118 ms are the per-tree `specialized` fallback the public API degrades to silently.

`batch_forward` and `forward` tell the same story (at d8: batch 3.955 / 1.607 ms against 20.548 / 2.108 ms
for the raised-cap dense and 108.937 / 151.452 ms for the fallback), so the crossover is not an artefact of
the loss assembly. `evoxc-auto` was not scheduled in this round, so the renders' `auto route` column falls
back to the arm it read (`evoxc-specialized-numpy`); on the shipped policy these pop-256/1000 cells resolve
to dense+iree, and at d7/d8 that dense graph declines exactly as the cap-64 arm does.

### 11.2 Does batch win?

Ratios against three dense variants: the **need-capped** one (cap 64 where it covers the population, the
raised cap where it does not), the **same-stride** one (always the raised cap), and the **fallback** the
shipped cap degrades to.

| cell | batch ÷ need-capped dense (dense faster if > 1) | same-stride dense ÷ batch (batch faster if > 1) | cap-64 fallback ÷ batch | faster arm |
|---|---|---|---|---|
| `p256-L128-i32-N1024` (d5) | 1.07 | 2.44 | – (dense served the cell) | dense, 1.07× |
| `p256-L128-i32-N1024-d6` | 1.41 | 1.87 | – | dense, 1.41× |
| `p256-L128-i32-N1024-d7` | 0.66 | 1.52 | 25.0 | **batch, 1.52×** |
| `p256-L256-i32-N1024` (d5) | 1.05 | 8.72 | – | dense, 1.05× |
| `p256-L256-i32-N1024-d8` | 0.24 | 4.20 | 23.5 | **batch, 4.20×** |
| `p1000-L128-i32-N1024` (d5) | 1.14 | 3.01 | – | dense, 1.14× |

**While the population's longest tree fits inside the cap, need-capped dense wins every measured cell** —
1.05–1.41×, including at pop 1000 (1.14×), and there is no crossover below that boundary anywhere in these
records. **Past the cap the ordering flips and batch wins decisively**: 1.52× over the raised-cap dense at
111 nodes, 4.20× at 187 nodes, and 25.0× / 23.5× over the only other compiled option, the per-tree
`specialized` fallback the shipped route silently degrades to.

The same-stride column prices the hypothesis directly: dense pinned to the population's own stride is
1.52–8.72× slower than batch on every cell, and the gap grows with the stride (2.44× at stride 128 vs 8.72×
at stride 256 on the *same* population).

**The stride cost is measurable inside one population**, which is the mechanistic confirmation.
`p256-L128-i32-N1024` and `p256-L256-i32-N1024` hold the identical structure signature
(`0a0dee3fec0cfb9c`: 232 distinct programs, 5200 merged ops, 4082 nodes) and differ only in the row stride:

- batch moves 2.167 → 2.237 ms (1.03×, i.e. unchanged) — its plan is the same graph;
- dense at the shipped cap moves 2.030 → 2.126 ms (1.05×) — both trace 64 rows;
- dense at the raised cap moves 5.277 → 19.497 ms — **3.7× for doubling the stride.**

Raising the cap to keep dense alive on the *same 31-node trees* costs 2.6× (64 → 128) and 9.2× (64 → 256).
Dense's per-call cost is stride-driven and batch's is node-count-driven, exactly as predicted.

### 11.3 Plan facts

`batch_plan_summary` of the batch record — compile-free introspection of the merged plan, loss variant.

| cell | nodes mean / med / max | distinct programs | emitted ops | bands × size | functions | row-index collisions |
|---|---|---|---|---|---|---|
| `p256-L128-i32-N1024` (d5) | 15.9 / 17 / 31 | 232 | 5200 | 8 × 29 | 232 | 24 |
| `p256-L128-i32-N1024-d6` | 26.1 / 27 / 61 | 226 | 7827 | 8 × 29/28 | 226 | 30 |
| `p256-L128-i32-N1024-d7` | 41.8 / 43 / 111 | 230 | 11639 | 8 × 29/28 | 230 | 26 |
| `p256-L256-i32-N1024` (d5) | 15.9 / 17 / 31 | 232 | 5200 | 8 × 29 | 232 | 24 |
| `p256-L256-i32-N1024-d8` | 66.6 / 66 / 187 | 225 | 17757 | 8 × 29/28 | 225 | 31 |
| `p1000-L128-i32-N1024` (d5) | 16.2 / 17 / 31 | 841 | 20253 | 27 × 32/31 | 841 | 159 |

The ops series is the hypothesis in one line: **5200 → 7827 → 11639 → 17757 emitted ops as the mean tree
grows 15.9 → 26.1 → 41.8 → 66.6 nodes** (22.4 / 34.6 / 50.6 / 78.9 ops per distinct program), i.e. 1.27 /
1.17 / 1.09 / 1.04× the population's own node total (4082 → 6676 → 10704 → 17040 nodes; the mean is exact,
so the total is `tree_len_mean × pop`). The merge adds
roughly a constant per program on top of each program's nodes, which is why the ratio falls as trees grow.
Population, not tree size, is the other axis: the pop-1000 cell holds the same 15–17-node trees as the pop-256
d5 cell and emits 20253 ops for 841 programs (24.1 ops per program, 1.25× its 16210 nodes), i.e. the plan
grows linearly in programs and the per-program constant is unchanged. Row-index collisions are the duplicate
row indices the merge has to reconcile inside a band (24–31 at pop 256, 159 at pop 1000).

The variants differ only in what the output side has to compute: `value[per_row=0]` is the smallest in
every cell (4309 ops at d5, 16888 at d8), and the per-row value variant is the widest plan — at pop 1000
`value[per_row=1]` carries 29 bands and 920 functions at 19904 emitted ops against the loss graph's 27
bands / 841 functions / 20253 ops.

### 11.4 Generation level

`gp_step` medians in ms over `--gp-reps 5`, eval half in parentheses (`results/batch/long-trees/gp`).

| cell | torch-per-tree | specialized-numpy | dense, cap = tree_len | batch-iree |
|---|---|---|---|---|
| `p256-L128-i32-N1024` (d5) | 22.710 (19.930) | 395.474 (392.311, 671 per-structure misses) | **11.552** (8.830, 0 misses) | 5317.995 (5314.776, 5 merged-graph compiles) |
| `p256-L128-i32-N1024-d7` | 13.959 (11.263) | 102.934 (100.078, 346) | **11.182** (8.446, 0) | 2177.223 (2173.981, 5) |

Only the raised-cap dense arm was scheduled in this directory, deliberately: at d7 the shipped cap-64 route
would be the same per-tree `specialized` fallback the neighbouring column already is. The operator layer is
identical across arms (2.7–3.2 ms), so the whole difference is the evaluation half.

**The long-tree warm win does not survive a GP loop.** A fresh population re-forms the structure multiset,
so all five timed generations paid a fresh merged-graph compile (`batch_compile_misses_total` 5, 5/5 plan
misses): 5318.0 ms per generation at d5 and 2177.2 ms at d7 against 11.6 / 11.2 ms for dense — **460× and
195×**. Even `torch-per-tree` beats it by 156–234× per generation, and the per-tree `specialized+numpy` by
13–21×. The route stays a fixed-population tool, whatever the tree length.

### 11.5 Numeric agreement at long trees

- **dense, either cap, is value-identical to `specialized`-numpy and `torch-per-tree`** in every cell where
  it actually runs: at d5 and d6 those arms — dense, the raised cap, `specialized`-numpy and the tree walk —
  carry one and the same sha256 on `batch_forward` and on `forward`. Where dense forms a group of its own
  (pop 1000 at d5, and the raised-cap arm at d7/d8) the only difference is the NaN payload bit on `sub` with
  NaN as the second operand (`0xffc00000` for dense vs `0x7fc00000` for numpy). The `nan` / `finite` counts
  and their placement agree exactly across every arm of a cell (0 at d5, 4096 at d6, 7168 at d7, 5120 at d8,
  3072 at pop 1000 on `batch_forward`), so none of this is a loss-path divergence.
- **`specialized`+iree — exactly the route the cap-64 arm falls back to at d7/d8 — is the numeric outlier**:
  FMA contraction of a `mul` feeding an `add`/`sub`, per-op ≤1 ulp, ≤2.2e-3 relative at d7 and ≤6.1e-3 at
  d8 after deep-tree cancellation. A 5-node tree `add(mul(x0,x1), x0)` reproduces it bit-for-bit, and
  `add`/`sub`/`mul`/`div` alone are exact on both backends. It is the §8 finding one backend over.
- **batch forms its own digest group**, as §8 already documents for this route, which is why all four
  directories are gated with `--allow-digest-mismatch` (§9). At d5 and d6 the split is two groups (`batch`
  against the other four arms); at d7/d8 it widens to four (`torch-per-tree`+`specialized-numpy`, the cap-64
  arm's `specialized`+iree fallback, the raised-cap dense, `batch`); the pop-1000 cell splits into three
  (`dense` and `dense-raised` share one).

### 11.6 Verdict and recommendation

- **(a) The route's second win is real, and it is the long-tree corner.** At pop ≥ 64 with a fixed
  population whose longest tree exceeds the dense trace cap, batch is 23.5–25.0× faster than the only other
  compiled option (the per-tree `specialized` fallback the shipped route silently degrades to) and 1.5–4.2×
  faster than dense with the cap raised to cover the population.
- **(b) That win does not survive a generation loop.** Every generation re-forms the structure multiset and
  recompiles the merged graph, so batch is 190–460× behind dense per generation and 156–234× behind the
  plain-torch tree walk. Long trees move the crossover, not the loop economics.
- **(c) Keep `DEFAULT_MAX_TREE_LEN = 64`.** If a run needs deeper trees, raise the cap to the depth actually
  generated — pinning 256 for 31-node trees costs 9.2× — rather than switching routes, and reach for
  `EVOGP_EVAL_MODE=batch` only for one-shot / fixed-population evaluation of long-tree populations (an
  elite set re-scored, a sweep over one population), never as the evaluator of a loop.
- **(d) Cold start is the second cost.** The merged compile is 13.2 / 22.8 / 37.9 / 12.0 / 74.7 / 178.5 s
  per cell against 6.6 / 6.5 / 7.2 / 7.0 s for cap-64 dense where dense actually runs (the 69.0 / 81.1 s in
  the two declined cells is the fallback's per-structure compile, not dense) and 14.4 / 14.3 / 14.5 / 41.9 /
  43.9 / 15.8 s for the raised cap — which is itself stride-driven (14.4 s at stride 128 vs 41.9 s at 256).
  At pop 1000 that 178.5 s compile is 25× dense's 7.0 s cold compile and buys ≈27 000 of batch's own 6.5 ms
  warm calls (and ≈31 000 of dense's 5.7 ms ones) — the break-even argument in one number.

### 11.7 Reproduce

From the repository root, with the project venv, one process per timed config, never two campaigns at once:

```bash
PY=/tmp/genesis/task_a996674198ffbcee/evogp-venv/bin/python
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1

# 1. tree_len 128, generation depth d5/d6/d7 (the node-count axis at a 128-row stride)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/l128 \
  --cells p256-L128-i32-N1024,p256-L128-i32-N1024-d6,p256-L128-i32-N1024-d7 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 2. tree_len 256, generation depth d5/d8 (the stride axis at two strides)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/l256 \
  --cells p256-L256-i32-N1024,p256-L256-i32-N1024-d8 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 3. the population axis at d5
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/p1000 \
  --cells p1000-L128-i32-N1024 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 4. generation level (§11.4): stride (d5) vs node count (d7), raised-cap dense only
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/gp \
  --cells p256-L128-i32-N1024,p256-L128-i32-N1024-d7 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward,gp_step --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 5. gate the four directories as one container, then render each
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch/long-trees \
  --allow-missing --allow-digest-mismatch --allow-snapshot
# expected: 4 directories gated, 38 records / 1000 checks, 0 failures (380 / 254 / 128 / 238 per directory)
for d in l128 l256 p1000 gp; do
  $PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/batch/long-trees/$d \
    --out test/benchmark/evoxc/results/batch/long-trees/RENDERED-$d.md
done
```

The protocol recorded in each `campaign.json` is `warmup 2`, `reps 5`, `gp_warmup 1`, `gp_reps 5`, seed 0,
one process per config, `EVOXC_DISABLE_LOCATIONS=1`; the per-config cap was 20 minutes (`--timeout 1200`,
which `campaign.json` does not store — its largest recorded `wall_s` is 260.0 s, the p1000 batch arm), and the
two cells dropped for exceeding the cap are listed in §12. The four directories took 563.2 + 657.5 + 371.6 +
322.1 s = 1914 s (≈32 min) of wall time in total, plus the idle-device guard's waits.

## 12. Caveats and known issues

- **Box/session scoped.** Absolute ms belong to this MI300X session (ROCm 7.1, torch 2.13); warm medians
  move ±10 % on a shared box. Use the ratios, and re-measure before quoting a number against another
  session's record.
- **`EVOGP_BATCH_CACHE_SIZE = 8` is one p256 population.** With `BAND_SIZE 32` a pop-256 population spans
  ~7–8 exec entries, so consecutive generations evict each other by construction. Raising the cache would
  help only if populations repeated, which a GP loop does not do.
- **The decline path is not free.** At pop 16 the batch arm costs 3.94 ms against 2.31 ms for the same
  `specialized+numpy` arm asked directly, i.e. ≈1.6 ms per call for a route that declined. Reported, not
  fixed: it lives under `src/`.
- **The batch `batch_forward` value path is non-bitwise on `arith`** (≈1e-8 relative to the value scale,
  up to 1.1e-3 relative on small entries) while dense is exact — see §8. An engine-side precision
  question for the `tree/` owners; the loss path and the per-row path are clean.
- **Explicit `EVOGP_EVAL_MODE=dense` at pop 16 recompiles as the used-function-set changes** (2 compiles
  in 3 generations, ~3.0 s each; `gp_step` 3015.8 ms vs 8.0 ms on `auto`). Not a batch finding, but it
  is in this round's records and is worth the dense route's owner's attention; `auto` is unaffected.
- **`flat1000-r5` is not gateable by construction** (its schedule records `sr_fitness` only, while
  `check.py` requires every value workload to have a digest somewhere in the cell). The 3-workload
  sibling `flat1000-3w-r5` is the gateable record; the pilot directories at `--reps 3` are likewise
  ungateable by the `MIN_SAMPLES` rule and are kept verbatim as the pilot, not as evidence.
- **No `CustomLoss`/multi-output and no robotics workload was timed or validated** on this route; the
  `batch` engine's multi-output path is untested here.
- **Two long-tree cells were dropped from the schedule** (§11): `p1000` × `tree_len 256` and `p1000` × `d7`.
  Their merged-graph compile exceeded a 20-minute budget with no output, so they are recorded as dropped
  rather than measured; the p256 `-d7` cell carries the high-node-count point at pop 256, and the p1000
  cell carries the population axis at d5.
- **`evoxc-dense-iree-raised` is a measurement of dense's stride cost, not a recommended setting.** It pins
  `EVOGP_DENSE_MAX_TREE_LEN` to the cell's stride, which is 2.6–9.2× slower than the shipped cap on
  populations that already fit (§11.2), so its numbers may only be compared inside their own cell.
- **Long-tree node counts are depth-driven, so `tree_len` alone does not make trees long** (§11). A cell at
  `tree_len 128` with the stock `max_layer_cnt = 5` still holds 15.9-node / 31-max trees — raising only the
  stride buys no long trees and costs dense directly.
- **The d7/d8 cells need `--allow-digest-mismatch`**: they split into four sha256 groups per value workload
  because `specialized`+iree contracts a `mul` into the following `add`/`sub` (FMA, ≤6.1e-3 relative after
  deep-tree cancellation) while dense differs from numpy only in the NaN payload bit on `sub`. dense's
  values are identical to `specialized`-numpy and `torch-per-tree` (see §11.5, §8).
- The commit-era history of the `test/benchmark/evoxc` harness (`results/final/**`, `results/baseline/`,
  the R1–R6 rounds) is untouched by this round; a re-run of the batch grid writes a fresh directory.

## 13. Round 1: chunked parallel compile, the stage split and reuse (results/batch_opt_round1/)

This round prices the batch route's chunked compile pool, its per-stage counters and its reuse behaviour in one window: HEAD against the pre-round tree `619cbf738af0bf839085879193f163d383bd9fbc` (via `--tree-root`), 6 generations, one process per config, on the cells `p256-L32-i8-N1024` (225 distinct structures) and `p64-L64-i32-N64` (the fixed-population niche cell of §3).
The records are `results/batch_opt_round1/{base,after_w1,after_default,after_w8,after_dense}/` and the detailed report is `results/batch_opt_round1/SUMMARY.md`; this section is its digest.

### 13.1 What the route does now

`EVOGP_BATCH_COMPILE_WORKERS` (default `4`) splits the deduped function list into contiguous chunks and stages their `lower` / `compile` / `load` bodies in a thread pool, one device executable per live chunk.
The chunk count is `min(requested, n_functions // batch_knobs.MIN_CHUNK_FUNCTIONS)` with `MIN_CHUNK_FUNCTIONS = 32`, and the plan stays single-bodied for `W <= 1`, below 32 functions, for an explicit `EVOGP_BATCH_BANDS` spelling (the flat one included) and for `EVOGP_BATCH_SWITCH_OPS=1`.
`evogp.tree.batch_engine.batch_stage_stats()` / `batch_stage_reset()` expose the ten stages (`decode`, `dedup`, `trace`, `merge`, `lower`, `compile`, `load`, `upload`, `run`, `gather`) plus the pool's own `batch_compile_wall_ms` and `batch_compile_jobs`.
`evogp.tree.batch_engine.describe_plan()` reports the resolved shape as `compile_workers` / `chunk_count` / `chunk_sizes`.
The stage counters are SUMS over pool threads, so a `batch_<stage>_total_ms` is compared against `batch_compile_wall_ms` and never read as wall time.

### 13.2 Cold compile: 11065.6 ms → 4444.2 ms at the shipped default

`p256-L32-i8-N1024`, `first_call_ms` (the untimed first call, one process per config):

| record | `EVOGP_BATCH_COMPILE_WORKERS` | chunks (`chunk_sizes`) | `first_call_ms` | vs pre-round |
|---|---|---|---|---|
| `base/` (pre-round tree) | absent | 1 (whole graph) | 11065.6 | 1.00× |
| `after_w1/` | `1` | 1 `[225]` | 11054.3 | 1.00× |
| `after_default/` | `4` (shipped) | 4 `[57, 56, 56, 56]` | **4444.2** | **2.49×** |
| `after_w8/` | `8` | 7 `[33, 32, 32, 32, 32, 32, 32]` | 3782.2 | 2.93× |

`W=1` reproduces the pre-round single-executable path — one chunk, one executable, cold within 0.1 % — so the sweep isolates the pool and nothing else.
The shipped default buys a 2.49× cold reduction and the 7-chunk request a further 1.18× on top of it.

### 13.3 Parallel efficiency of the pool

Per-chunk compile sum (`batch_compile_total_ms`) against the pool's own wall time, same cell:

| record | compile sum (thread sum) | `batch_compile_wall_ms` | `batch_compile_jobs` | efficiency |
|---|---|---|---|---|
| `after_w1/` | 9946.0 | 10600.7 | 1 | 0.94× |
| `after_default/` | 13057.4 | 4028.5 | 4 | 3.24× (81 % of 4 workers) |
| `after_w8/` | 18217.5 | 3354.9 | 7 | 5.43× (78 % of 7 chunks) |

The compile stage therefore scales close to the worker count but not linearly — the per-body fixed cost (one lower + one iree compile + one load) and the pool's own overhead are what the missing 19–22 % is.

### 13.4 Where the cold call goes at W=4

Inside the 4444.2 ms call:

- compile-pool wall 4028.5 ms = **90.6 %** of the call, over 4 jobs;
- trace + merge 363.3 ms ≈ 8.2 %, decode 4.7 ms ≈ 0.1 %, upload + run + gather 45.0 ms ≈ 1.0 % — all outside the pool, so the part of the cold call outside it stays under a tenth;
- the thread-summed counters INSIDE the pool are lower 1818.8 ms, compile 13057.4 ms and load 277.6 ms (6.2 % of the call as a thread sum), and their sum spread over the 4 threads (3788.5 ms) accounts for most of the 4028.5 ms pool wall.

Lower is inside the pool at 1818.8 ms because each chunk lowers its own merged graph; the cold call is a compile problem, and the chunk that lowers is the unit of that problem.

### 13.5 The warm call pays for it, and the knee sits at 4

`sr_fitness` at `p256-L32-i8-N1024`, with the warm stage split (`run` / `upload` / `gather`, per-call medians from the stage counters):

| record | `sr_fitness` median | vs pre-round | run / upload / gather | vs `dense` |
|---|---|---|---|---|
| `base/` | 2.011 ms | 1.00× | not instrumented | 1.86× |
| `after_w1/` | 2.013 ms | 1.00× | 1.198 / 0.212 / 0.318 ms | 1.86× |
| `after_default/` | **3.628 ms** | **1.80×** | 1.949 / 0.752 / 0.622 ms | 3.35× |
| `after_w8/` | 5.033 ms | 2.50× | 2.405 / 1.247 / 0.997 ms | 4.65× |
| `after_dense/` | 1.082 ms | 0.54× | no stage counters | 1.00× |

A warm call now dispatches one invocation per chunk, which is the whole of the 1.80×: run 1.198 → 1.949 ms, upload 0.212 → 0.752 ms and gather 0.318 → 0.622 ms.
Break-even for a FIXED population in one process is ≈4100 warm calls — the 6621.4 ms of cold saved divided by the 1.617 ms of warm added — and a generation loop never gets there upward: the 6-generation loop total (`sum_total_ms`) is 12773 ms at `W=4` against 25134 ms at `W=1` and 24803 ms pre-round, i.e. the cold saving dominates (still 45× dense's 281 ms).
Going 4 → 7 chunks buys only a further 1.18× cold (4444.2 → 3782.2 ms) while costing 1.39× more warm (3.628 → 5.033 ms), so the shipped default of 4 sits on the right side of the knee at this cell and stays.

### 13.6 Chunking is inert on the fixed-population niche cell

`p64-L64-i32-N64` has 63 deduped functions, and `63 // 32 = 1` forces one chunk and one worker at every requested width:

| record | chunks | `first_call_ms` | `sr_fitness` median |
|---|---|---|---|
| `base/` | 1 | 2248.0 | 1.041 ms |
| `after_w1/` | 1 `[63]` | 2257.4 | 1.107 ms |
| `after_default/` | 1 `[63]` | 2240.8 | 1.127 ms |
| `after_w8/` | 1 `[63]` | 2279.0 | 1.121 ms |
| `after_dense/` | — | 6437.1 | 1.385 ms |

Cold (2240.8 vs 2248.0 ms pre-round) and warm (1.127 vs 1.041 ms) are unchanged within noise, and the route still beats dense there (warm 1.127 vs 1.385 ms, cold 2.2 s vs 6.4 s), so §3's niche verdict is unaffected by the pool.
The plan shape must be read from `first_call.stage_ms.plan` (`compile_workers` / `chunk_count` / `chunk_sizes`), never from `batch_policy.compile_workers`: on this cell the policy reports the requested 4 while the plan holds one chunk of 63 functions.

### 13.7 Reuse: the plan key is a no-op in a generation loop, the exec table is per chunk

The plan counters are identical pre-round and at `W=4` after one cold call plus 6 generations — `batch_plan_hits` 9, `batch_plan_misses` 6, `batch_plan_entries` 6, `batch_exec_hits` 0 — so the content-addressed plan key buys nothing in a GP loop: 6 generations present 6 distinct structure multisets (`generation_hit_rate` 1/6, the single hit being the starting population the warm workload had already evaluated).
The exec table is per chunk: a chunked plan stores one executable per chunk, so the cold call occupies 4 entries, the 6-generation loop records 22 exec misses and the table saturates `EVOGP_BATCH_CACHE_SIZE = 8` (two merged populations), where the pre-round path recorded 6 entries (one per graph).
A generation loop therefore evicts the previous population's chunks by construction at `W=4`, which is a second, independent reason the chunked pool is a cold-start lever rather than a loop lever.

### 13.8 Open items this round did not settle

- `EVOGP_BATCH_BAND_SIZE` / `EVOGP_BATCH_BANDS` against `chunk_count`: an explicit band count and `switch_ops` both force the single-body path, so no cell here separates the two knobs.
- Whether `EVOGP_BATCH_CACHE_SIZE` should scale with the chunk count, given that one population now needs `ceil(chunks / cache)` of the table.
- The long-tree and pop-1000 cells of §11 (`tree_len` 128/256, pop 1000): none measured here, because the box is shared and one merged-graph compile at pop 1000 costs minutes.

### 13.9 Reproduce and gate

From the repository root, with the project venv, one process per config (`batch_economics.py` writes `<out-dir>/<cell>/<route>.json` plus one `economics.json` per directory):

```bash
PY=/tmp/genesis/task_a996674198ffbcee/evogp-venv/bin/python
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
CELLS=p256-L32-i8-N1024,p64-L64-i32-N64
OUT=test/benchmark/evoxc/results/batch_opt_round1

# 1. HEAD, shipped default (the knob unset)
$PY test/benchmark/evoxc/batch_economics.py --cells $CELLS --routes batch --generations 6 --out-dir $OUT/after_default

# 2. the W sweep: one knob value per process, one process per config
$PY test/benchmark/evoxc/batch_economics.py --cells $CELLS --routes batch --generations 6 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=1 --out-dir $OUT/after_w1
$PY test/benchmark/evoxc/batch_economics.py --cells $CELLS --routes batch --generations 6 \
  --knob EVOGP_BATCH_COMPILE_WORKERS=8 --out-dir $OUT/after_w8

# 3. the pre-round column (the tree at 619cbf7 checked out elsewhere) and the dense column
$PY test/benchmark/evoxc/batch_economics.py --cells $CELLS --routes batch --generations 6 \
  --tree-root <checkout of 619cbf738af0bf839085879193f163d383bd9fbc> --out-dir $OUT/base
$PY test/benchmark/evoxc/batch_economics.py --cells $CELLS --routes dense --generations 6 --out-dir $OUT/after_dense

# 4. gate the round through its campaign-format record
python3 test/benchmark/evoxc/check.py --results $OUT/campaign \
  --allow-snapshot --allow-missing --allow-digest-mismatch
# expected: 3 records / 88 checks / 0 failures
```

`batch_economics.py` records carry the `evogp-bench-evoxc-v1-batch-economics` schema, so `check.py` gates a campaign-format directory only — `results/batch_opt_round1/campaign/` is that directory for this round, and the economics directories are evidence next to it, not gateable units.

### 13.10 Follow-up: the content-addressed plan key, measured (`reuse_probe.py`)

§13.7 could show that a generation loop gets **no** plan-key reuse, but not that the second key table does anything at all: a generation loop never presents the same content twice, so the surface that answers a *rebuilt-but-equal* population had no measurement behind it.
`reuse_probe.py` (new, one child process per column) supplies it with a fixed four-call sequence on one cheap cell (`p64-L64-i32-N64`, seed 0, `evoxc-batch-iree`, band 32, pool 4):
call 1 evaluates `Forest` A (cold), call 2 evaluates A again (identity fast path), call 3 evaluates `Forest` **B** — a different object built from the same seed, verified `identical_content` in `no_shared_storage` — against the *same* `x`/`labels` object, call 4 evaluates B again.
Every call records both counter surfaces before -> after (`batch_cache_stats` and `batch_content_stats`), the value digest, the sync-bracketed wall time and, where the tree ships them, the ten stage **call** counts — so "it skipped trace/merge/compile" is a measured count, not an inference from a fast call.

| call | population | HEAD `reuse/after` | pre-round `reuse/base` (`619cbf7`) | HEAD counters | pre-round counters |
|---|---|---|---|---|---|
| 1 | A, cold | 2340.3 ms | 2401.2 ms | exec 0 h / 1 m, plan 0 h / 1 m, content 0 h / 1 m (`attempts` 1, `hit_rate` 0.0) | exec 1 m, plan 1 m — no content table |
| 2 | A, identity | 1.274 ms | 1.235 ms | plan 1 h / 0 m, content `attempts` **0** | plan 1 h / 0 m |
| 3 | **B, rebuilt** | **2.724 ms** | **79.109 ms** | plan 0 h / **1 m** -> content **1 h / 0 m** (`hit_rate` 1.0); stage calls: decode 1, trace 0, merge 0, lower 0, compile 0, load 0, upload 1, run 1, gather 2 | exec **1 h** / 0 m, plan 0 h / 1 m -> re-decode, re-trace, re-merge (no recompile) |
| 4 | B, identity | 1.107 ms | 1.101 ms | plan 1 h / 0 m, content `attempts` 0 | plan 1 h / 0 m |

- The measured hit-rate change on a rebuilt population: pre-round the content surface does not exist (`available: false`, every counter `null` — NOT 0) and the plan table records a MISS; on HEAD the same call is an identity MISS the content table answers, `batch_content_hits` 1 / `batch_content_misses` 0 / `hit_rate` 1.0, i.e. **79.109 ms -> 2.724 ms (29.0×, 76.4 ms per rebuilt population)**, with the merge-stage call counts at 0.
- The answers are identical, not merely similar: all four calls on both trees return sha256 `b6584500a0fb365adfbf712cc6b43d7e` (64 finite, 0 NaN), so the content hit is a reuse of the same numbers.
- The replay column `reuse/after-replay` reproduces the sequence (cold 2302.6 ms, rebuilt 2.649 ms) and the same `verification` block; nothing in the table is a single unconfirmed sample.
- **This does not change the generation-loop verdict.** Over the 6-generation `content-blocks/after` loop at the same cell the content table saw 6 attempts, 0 hits (`hit_rate` 0.0), with `summary.content.generations` = `{rows 6, with_content_hit 0, with_content_miss 5, no_attempt 1}` — generation 1 is the population the timed first call already compiled, and generations 2..6 each present *new* programs, so there is nothing for a content key to match. The regime the content key wins is the one `reuse_probe.py` measures: a fixed population evaluated again through a fresh object (re-serialized forests, a reloaded pipeline, a caller that rebuilds the same evolution), which is exactly where the pre-round tree pays 76.4 ms per population and HEAD pays a decode.
- The pre-round column also says what the reuse was NOT: `batch_exec_hits` 1 on the rebuilt call means the compiled body was already shared by the exec table, so the recompile was never the cost there — the 76.4 ms is host-side decode + trace + merge. A content hit removes that work; it does not change what the pool compiles.
- Record fields (`test/benchmark/evoxc/reuse_probe.py`): `forests.A/B` (structure signature + storage addresses), `forests.same_content` (`fields_equal` / `fields_in_distinct_storage`), `dataset` (one shared tensor), `cache_keys` (the counter names each surface exposes, `[]` when absent), `calls[i].{call_ms, digest, cache.batch.*, content.*, stage_ms.*}` and `verification` (each expectation `true` / `false` / `null`, plus `rebuilt_call_plan_misses` / `rebuilt_call_compile_misses` / `rebuilt_call_stage_calls`).
- In the two harness records the same surface now appears as `batch_economics.py`'s `first_call.content` / `sr_fitness.content` / `generations[i].content` / `summary.content` and `worker.py`'s `batch_cache.content_before_workloads` / `content_after` / `content_keys` / `content_totals`; on the pre-round tree every one of them records `available: false` with null counters plus a note naming the absent `batch_content_stats()`.
- A first call for a *new plan shape* is an identity miss even on an unchanged object, so `worker.py`'s HEAD column reads `content_totals` `{hits 0, misses 3, attempts 3}` — one content probe per workload plan (`sr_fitness`, `batch_forward`, `forward`), each a miss; the warmup+reps calls behind them are answered by the identity table (`attempts` 0 per call, not misses).

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
OUT=test/benchmark/evoxc/results/batch_opt_round1
# HEAD, then the pre-round tree (unpacked outside the repo at commit 619cbf7)
$PY test/benchmark/evoxc/reuse_probe.py --label after
$PY test/benchmark/evoxc/reuse_probe.py --label base --tree-root <checkout of 619cbf738af0bf839085879193f163d383bd9fbc>
$PY test/benchmark/evoxc/reuse_probe.py --label after-replay          # the reproducibility column
# the content blocks inside the economics/worker records, both trees
$PY test/benchmark/evoxc/batch_economics.py --cells p64-L64-i32-N64 --routes batch,dense \
  --generations 6 --out-dir $OUT/content-blocks/after
$PY test/benchmark/evoxc/batch_economics.py --cells p64-L64-i32-N64 --routes batch,dense --generations 6 \
  --tree-root <checkout of 619cbf7> --out-dir $OUT/content-blocks/base
# the over-1 compile share, on a cell the pool really chunks (4 chunks at the shipped default)
$PY test/benchmark/evoxc/batch_economics.py --cells p256-L32-i8-N1024 --routes batch \
  --generations 3 --out-dir $OUT/content-blocks/chunked-p256
# the gate unit: a campaign-format batch record carrying batch_cache.content_*
$PY test/benchmark/evoxc/campaign.py --cells p64-L64-i32-N64 --arms evoxc-batch-iree \
  --reps 5 --gp-reps 3 --out $OUT/content-blocks/campaign-content
```
`batch_economics.py` and `reuse_probe.py` records carry their own schemas (`evogp-bench-evoxc-v1-batch-economics`, `evogp-bench-evoxc-v1-batch-reuse`), so the gate unit for this follow-up is a campaign-format directory whose batch record carries the new blocks: `content-blocks/campaign-content/` (`1 record / 40 checks / 0 failures`, `batch_cache.content_totals` = `{hits 0, misses 6, attempts 6}` — one content probe per plan shape in that config, none of them a rebuilt population) next to the round's own `campaign/` (`3 records / 88 checks / 0 failures`).
The economics/reuse directories are evidence beside them, and the frozen directories keep their counts (`warm-r5` 24/682, `small-r5` 36/1022, `flat-r5` 4/126, `flat1000-3w-r5` 1/36, `final/after` 130/3602 — all 0 failures).

## 14. Round 2: the band policy, the shipped default's chunked shape and the composition (results/batch_opt_round2/)

Round 2 answers the question §13 left open — what `EVOGP_BATCH_BAND_SIZE` / `EVOGP_BATCH_BANDS` should be when the population is not fixed, and what the banded+chunked composition costs — in three measured items on three cells (`p256-L32-i8-N1024`, `p256-L64-i32-N1024`, and `p64-L64-i32-N64`, the niche cell of §3/§7): a warm band-size sweep at pool 1 (`results/batch_opt_round2/warm/`, 15 records), `EVOGP_BATCH_BANDS=auto` against `b16`/`b32` (`results/batch_opt_round2/auto/`, 9 records), and the banded+chunked composition plus a 6-generation loop against `dense` (`results/batch_opt_round2/composition/`, 9 sweep records + 10 loop routes), with a `--cold-only` re-measure in `round3-cold/` and a one-record smoke in `smoke-cold/`.
Each directory carries its own `SUMMARY.md` with the full tables; this section is the decision of record and quotes them, and every number below is read off a committed record or one of those three summaries.

### 14.1 The decision: `auto` stays opt-in, the band size moves 32 → 16, `CHUNK_BANDS` stays off

Three decisions, recorded as the shipped state:

1. **`EVOGP_BATCH_BANDS=auto` does NOT become a default — it stays an opt-in spelling.** It is the argmin in **0 of 9** warm columns of the item-2 grid and the slowest of the three configs in **7 of 9**, so it is not a candidate for "what the engine picks when nobody says anything".
2. **`EVOGP_BATCH_BAND_SIZE`'s default is `16`, moved from `32`** (the value lives in `src/evogp/tree/batch_knobs.py::DEFAULT_BAND_SIZE`). 16 is the minimax setting of the item-1 power-of-two grid: best pooled geomean and by far the best worst column, at 3–6 % spread among the banded configs.
3. **`EVOGP_BATCH_CHUNK_BANDS` stays OFF (its shipped state).** The composition trades 1.61–2.47× warm for 2.6–2.8× cold at the pop-256 cells, is inert at pop 64, and changes nothing in a generation loop (§14.3, `composition/SUMMARY.md` §8); its break-even is ≈4.5k warm `sr_fitness` calls at `p256-L32` (≈1.3k on the per-row `forward`).

**Evidence for decision 1 — `auto` wins nothing, and it is not even a stable policy.** It is inside the 2.6–6 % cross-config noise band in **4 of 9** columns (`p256-L32 sr_fitness` 1.014, `p256-L64 sr_fitness` 1.032, `p64 sr_fitness` 1.027, `p64 forward` 1.029) and **outside it in 5**, worst **+18.9 %** on the band-sensitive per-row `forward` at `p256-L32` (1.189 median-based, 1.275 min-based), then +14.5 % (`p256-L64 forward`), +14.1 % (`p64 batch_forward`), +9.9 % (`p256-L64 batch_forward`) and +8.4 % (`p256-L32 batch_forward`).
Cold it pays 1.12–1.35× the cheapest call of its column at the two pop-256 cells (1.01–1.11× at `p64`) because of the clamp's second merge, and the model's answers are not population-stable — §14.4 carries both findings.

**Evidence for decision 2 — 16 is the minimax setting, and the first-order call is flat vs banded.** On the min-based (contamination-robust) estimator, pooled over the nine columns of item 1: `b16` geomean **1.0185** of the per-column argmin, worst column **1.061**; `b32` 1.0223 / 1.137; `b8` 1.0415 / 1.167; `b64` 1.0718 / 1.224; `flat` 1.1680 / 1.529 (`warm/SUMMARY.md` T5) — and `b16` is the only config ranked 1st or 2nd in all three cells under both estimators.
Pooled over the **18** columns of items 1+2 the mean is a tie (median-based `b16` 1.0248 against `b32` 1.0224; min-based 1.0199 against 1.0254) and **16 wins on the worst column** (`b16` 1.102 median-based / 1.109 min-based against `b32`'s 1.143 / 1.137, both on `forward` at `p256-L32`), which is the criterion a default has to satisfy on a shared box.
The honest bound: this is a **3–6 % second-order call**, the same scale as this window's cross-config noise (the 2.6 % same-shape `p64` twin, `b64` vs `flat`, both K = 1), so the default does not have to be the per-cell argmin — it has to be the setting that is never far from it; the cost of *not* moving is at most ~5 % on the worst column.
The **first-order** decision is flat versus banded: a single unbanded merge (K = 1) is **+15 %** on `sr_fitness` (geomean), **+8 %** on `batch_forward`, **+34 %** on the per-row `forward` (worst column **+53 %**) and pays the largest cold compile at both pop-256 cells (12 671.6 / 14 610.0 ms against 11 049.4 / 12 590.6 for the single-body `b16`).
**The measurement and the legality floor agree**: ROCm/HIP binds `K ≤ 15` bands with `≥ 8` merged functions per band, so at n = 225–245 functions the cap lands at band size ≈ 15–17 — the region the sweep crowns; `b16` is itself clamped to K = 15 on the per-row plan (`band_sizes` 5×17 + 10×16, `band_clamped: true`) and unclamped at K = 15 on the wide-row ones, which is why the clamp and the argmin point at the same number (`auto/SUMMARY.md` §2, `warm/SUMMARY.md` §2).

Warm medians of the shipped band size on the **control shape** (`b16` at `EVOGP_BATCH_COMPILE_WORKERS=1`, round-2 `auto/` window, ms; the shipped route's own chunked column is §14.2), with each config's ratio to that column's argmin:

| cell | workload | `b16` (ms, pool-1 control) | `b16` ÷ argmin | `b32` ÷ argmin | `auto` ÷ argmin | argmin |
|---|---|---:|---:|---:|---:|---|
| p256-L32-i8-N1024 | sr_fitness | 2.0331 | **1.000** | 1.066 | 1.014 | `b16` |
| p256-L32-i8-N1024 | batch_forward | 2.2305 | **1.000** | 1.034 | 1.084 | `b16` |
| p256-L32-i8-N1024 | forward | 0.8632 | **1.000** | 1.023 | 1.189 | `b16` |
| p256-L64-i32-N1024 | sr_fitness | 2.2382 | 1.017 | **1.000** | 1.032 | `b32` |
| p256-L64-i32-N1024 | batch_forward | 2.2557 | 1.075 | **1.000** | 1.099 | `b32` |
| p256-L64-i32-N1024 | forward | 0.8234 | 1.026 | **1.000** | 1.145 | `b32` |
| p64-L64-i32-N64 | sr_fitness | 1.1305 | **1.000** | 1.026 | 1.027 | `b16` |
| p64-L64-i32-N64 | batch_forward | 0.8899 | **1.000** | 1.014 | 1.141 | `b16` |
| p64-L64-i32-N64 | forward | 0.7097 | 1.062 | **1.000** | 1.029 | `b32` |

Two estimator caveats, as the source records them: item 1 has **two early-rep-contaminated rows** (`p256-L32 b16 sr_fitness` samples 2.3094 / 2.2812 / 2.2979 / 2.0821 / 2.0719 and `p256-L64 b8 batch_forward` 2.2083 / 2.1363 / 2.7307 / 2.3061 / 3.2639), which is why the min column exists and why the default is answered on the min-based estimator as well; and **nothing here rests on a difference below ~5 %** — the whole banded region is one tier, which is exactly why 16 is chosen as a minimax setting rather than as a win.

### 14.2 The updated per-call table — batch vs dense in the shipped default's own shape

**The shipped default's measured column at pop 256 is NOT the single-body `b16` row of §14.1 — it is the `b16-cw4` shape.** The chunk gate is `src/evogp/tree/batch_programs.py::_resolve_workers`: it refuses chunking only when an explicit `EVOGP_BATCH_BANDS` spelling is in play (`bands is not None` — an explicit K, the `none`/`flat` spellings, or `auto`) unless `EVOGP_BATCH_CHUNK_BANDS` opts in, so a band **size** does not refuse chunking.
With the shipped knobs (band size 16, `EVOGP_BATCH_COMPILE_WORKERS` default 4, `CHUNK_BANDS` off) the plan at n = 225 functions is therefore **4 chunks** — exactly the `b16-cw4` shape the composition measured (band size 16 per chunk, `chunk_bands [4, 4, 4, 4]`).
Round 1's default record already proves the mechanism empirically: `results/batch_opt_round1/after_default/` chunked into `chunk_sizes [57, 56, 56, 56]` with no band request in play.
So the `b16` rows of `../auto/` and `../warm/` are the `EVOGP_BATCH_COMPILE_WORKERS=1` single-body **control**, and the shipped default at pop 256 is priced by `b16-cw4`; at `p64-L64-i32-N64` (n = 63) chunking is inert on the two wide-row plans (`chunk_count` 1, `chunk_sizes [63]`), so there default == control.

Warm medians in ms (ratio = reference arm ÷ batch, so **> 1 means the batch route is faster** — the convention of §4.1); `dense` from the round-1 records named below, the shipped default from `composition/<cell>/b16-cw4.json`, the control from `auto/<cell>/b16.json`:

| cell | workload | dense (ms) | batch shipped default (ms) | ratio | batch single-body control (ms, `COMPILE_WORKERS=1`) | ratio |
|---|---|---:|---:|---:|---:|---:|
| p256-L32-i8-N1024 | sr_fitness | 1.1013 | 3.5736 | **0.31** | 2.0331 | 0.54 |
| p256-L32-i8-N1024 | batch_forward | 1.8657 | 4.3761 | **0.43** | 2.2305 | 0.84 |
| p256-L32-i8-N1024 | forward | 0.6858 | 1.9470 | **0.35** | 0.8632 | 0.79 |
| p256-L64-i32-N1024 | sr_fitness | 2.1524 | 3.6770 | **0.59** | 2.2382 | 0.96 |
| p256-L64-i32-N1024 | batch_forward | 2.8540 | 3.6357 | **0.79** | 2.2557 | 1.27 |
| p256-L64-i32-N1024 | forward | 0.9822 | 2.0355 | **0.48** | 0.8234 | 1.19 |
| p64-L64-i32-N64 | sr_fitness | 1.4096 | 1.1765 | **1.20** | 1.1305 | 1.25 |
| p64-L64-i32-N64 | batch_forward | 1.4216 | 0.9231 | **1.54** | 0.8899 | 1.60 |
| p64-L64-i32-N64 | forward | 0.8678 | 1.0710 | **0.81** | 0.7097 | 1.22 |

(`dense` = `results/batch/warm-r5/p256-L32-i8-N1024/evoxc-dense-iree.json` 1.1013 / 1.8657 / 0.6858, `results/batch/warm-r5/p256-L64-i32-N1024/…` 2.1524 / 2.8540 / 0.9822, `results/batch/small-r5/p64-L64-i32-N64/…` 1.4096 / 1.4216 / 0.8678.)
The one row whose median must not be read on its own: `p256-L32 batch_forward` in the default shape has samples 3.578 / 4.376 / 6.450 / 5.635 / 3.537 ms (`max:min` 1.82), i.e. **contention-spiked** — its min 3.5370 against the control's min 2.1528 keeps the column's verdict (default slower) without resting on the median.

Four notes, one sentence each:

- (a) **The dense column is the round-1 records, and it is stable across windows to ~6 %:** round 2 re-measured dense's `sr_fitness` in the composition loop window at 1.1027 (L32) / 2.0240 (L64) ms, i.e. +0.1 % / −6.0 % of the round-1 values, so the reference holds while absolute ms across documents are ± a factor (the caveat §3 and §12 already state) — the ratios are the reading.
- (b) **The shipped default is slower warm than the pool-1 control at both pop-256 cells, 1.61–2.47×** (L32 1.76 / 1.96 / 2.26, L64 1.64 / 1.61 / 2.47 — four separate upload / run / read-back trios per call, `composition/SUMMARY.md` §3), and on the two 63-function plans at `p64` it is identical to the control (`chunk_count` 1).
- (c) **Where the CONTROL beats dense on the two value paths at `p256-L64` (1.27× / 1.19×), the default does not (0.79× / 0.48×)** — and note that §4.1's prose ("`batch_forward` and `forward` tell the same story") understates those two columns, because its own rows show batch faster than dense there (dense 2.854 / 0.982 ms against batch 2.156 / 0.739 ms, i.e. 0.76× / 0.75× of dense's time); the round-2 control reproduces that direction, the shipped chunked shape loses it.
- (d) **At `p64-L64-i32-N64` (the niche cell of §3) the default beats dense on `sr_fitness` and `batch_forward` (1.20× / 1.54×) but not on the per-row `forward` (0.81×)**, because that plan is the one that splits at pop 64 (`chunk_bands [2, 2]`, `chunk_count` 1 only on the wide-row plans) and therefore pays 1.51× the control's warm call — its `forward` value 1.0710 ms is the composition's `b16-cw4` number, while 1.1765 / 0.9231 ms are the inert-shape ones.

### 14.3 The cold column: the composition is a cold lever, and flat can no longer reach it

`first_call_ms` of the `sr_fitness` plan in ms — single-run banded against flat (which can no longer chunk), banded+chunked (the shipped default's shape), round 1's `W=1 → W=4` column, and dense:

| cell | single-body banded (`b16`, pool 1) | flat, single-body (`EVOGP_BATCH_BANDS=none`) | banded+chunked (`b16-cw4` = shipped default) | round-1 default (band 32, `W=1 → W=4`) | dense |
|---|---:|---:|---:|---:|---:|
| p256-L32-i8-N1024 | 11 049.4 | 12 671.6 (loop record 12 742.8) | **4 180.6** (loop record 4 165.9) | 11 065.6 → 4 444.2 | 3 491.7 |
| p256-L64-i32-N1024 | 12 590.6 | 14 610.0 | **4 422.4** | — (round 1 swept only the L32 cell) | 6 816.3 |
| p64-L64-i32-N64 | 2 298.7 | 2 258 (the `flat-cw4` twin reads 2 237.8) | 2 284.2 (chunking inert) | 2 257.4 → 2 240.8 | 6 177.7 |

(`b16` and the flat cold values from `auto/SUMMARY.md` §2 and `../warm/` T3 / `composition/<cell>/flat-cw4.json`; the chunked column from `composition/<cell>/b16-cw4.json` with the loop records beside it; round 1 from `results/batch_opt_round1/{base,after_w1,after_default}` — the `W=1` column reads 11 065.6 in `base/` and 11 054.3 in `after_w1/`, 0.1 % apart, as §13.2 records it; dense from the composition loop records 3 491.7 / 6 816.3 ms and `results/batch/small-r5/p64-L64-i32-N64/evoxc-dense-iree.json` 6 177.7 ms.)

- **The flat spelling can no longer chunk at all:** `flat-cw4` reports `compile_workers 1` / `chunk_count 1` / `chunk_bands [1]` on all three workloads of all three cells (`plan` blocks carry `chunk_sizes [225]` / `[245]`), so round 1's chunked-flat cold win is unreachable that way and `EVOGP_BATCH_CHUNK_BANDS=1` is the only route to combining a band request with the pool (`composition/SUMMARY.md` §6).
- **The composition's cold win is 0.35–0.54× of the single-body arm across all three workloads at both pop-256 cells** (L32 0.38 / 0.40 / 0.54, L64 0.35 / 0.40 / 0.54), and it comes from four per-chunk compiles: the compile counter sums 12 439.4 ms over 4 job timers against a pool wall of `batch_compile_wall_ms` 3 733.1 ms over 4 jobs (against 10 583.6 ms over 1 job), with `lower` up 3.0× (296.6 → 888.5 ms) and `load` up 2.5× (114.3 → 290.8 ms), so the net call is **2.6–2.8×** cheaper rather than 4×.
- **The stage counter sum cannot close on a chunked row** (14 063 ms of stages against a 4 181 ms call) by construction, because those four timers run concurrently — the same reason a loop record's `compile_share` reads above 1 (`composition/SUMMARY.md` §4).
- **Against dense the default is dearer at L32 and cheaper at L64** — 4 180.6 against 3 491.7 ms at `p256-L32`, 4 422.4 against 6 816.3 ms at `p256-L64` — the one cold column where a batch route beats the shipped route, and it is the composition, not the band size, that buys it.

### 14.4 What the cost model chose vs what measured best

All nine `auto` rows of the item-2 grid: `band_policy` is `{backend iree, device_kind rocm, max_bands 15, min_band_size 8, uncapped false}` everywhere, `requested K` is the model's answer before clamping, and the last column is the measured argmin of that column with its warm median (`auto/SUMMARY.md` §1/§2/§4):

| cell | workload | n_functions | requested K | final `bands` | `band_sizes` | clamped | `cost_estimate` | `flat_estimate` | measured best (warm ms) |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| p256-L32-i8-N1024 | sr_fitness | 225 | **213** | 15 | 15×15 | **yes** | 7 350 691 | 466 453 471 | `b16` (2.0331) |
| p256-L32-i8-N1024 | batch_forward | 225 | **213** | 15 | 15×15 | **yes** | 6 425 728 | 372 544 768 | `b16` (2.2305) |
| p256-L32-i8-N1024 | forward | 245 | 4 | 4 | 1×62, 3×61 | no | 1 231 549 | 1 375 992 | `b16` (0.8632) |
| p256-L64-i32-N1024 | sr_fitness | 232 | **220** | 15 | 7×16, 8×15 | **yes** | 7 780 152 | 524 411 684 | `b32` (2.2008) |
| p256-L64-i32-N1024 | batch_forward | 232 | **220** | 15 | 7×16, 8×15 | **yes** | 6 825 600 | 427 565 312 | `b32` (2.0986) |
| p256-L64-i32-N1024 | forward | 245 | 4 | 4 | 1×62, 3×61 | no | 1 347 921 | 1 580 382 | `b32` (0.8028) |
| p64-L64-i32-N64 | sr_fitness | 63 | **16** | 7 | 7×9 | **yes** | 815 389 | 2 724 142 | `b16` (1.1305) |
| p64-L64-i32-N64 | batch_forward | 63 | **16** | 7 | 7×9 | **yes** | 654 592 | 2 182 976 | `b16` (0.8899) |
| p64-L64-i32-N64 | forward | 64 | 1 | 1 | 1×64 | no | 321 984 | 321 984 | `b32` (0.6681) |

Three findings:

- **It errs in BOTH directions.** On the wide-row (loss / population-batched value) plans it requests 213/220 bands where 15 can load — a **14.2–14.7× overshoot**, clamped to the 15-band ceiling with a `band_note` naming the request, the cap and the `EVOGP_BATCH_BANDS=K` remedy (at `p64` it requests 16 for 63 functions and the `≥ 8`-functions rule clamps it to 7). On the per-row `forward` plan the same model answers **4 bands** at pop 256 and a bare **1** — the flat merge — at pop 64, unclamped and *below* the banded region item 1 measured (K = 8–16).
- **Its own estimates are one to three orders of magnitude apart on plans whose measured medians differ by ≤ 19 %:** a 58–67× preference for many tiny bands on the clamped wide-row plans (6.4–7.8 M against 3.7–5.2 × 10⁸), 1.12× at `p256-L32 forward` (1 231 549 against 1 375 992) and 1.17× at `p256-L64 forward`, and exactly 1.0× at `p64 forward`, where `cost_estimate == flat_estimate` = 321 984 — the model itself said "unbanded" for the plan it then measured 2.9 % behind the `b32` argmin (0.6872 against 0.6681 ms).
- **It is not population-stable.** On `p256-L32-i8-N1024` (structure signature `ad11cb66ffc21ac8`, 225 deduped functions in both records) the sweep's loss plan asked **K = 213** (`emitted_ops` 5 037) while the generation-loop first call asked **K = 1** (`cost_estimate == flat_estimate` 3 663 871, `emitted_ops` 4 141) — two differently priced graphs — so the model is not keyed on the deduped function count, and across the records of that one cell the same policy answers K = 1, 4 and 213 together.

**The clamp is not free.** It is not a separate stage: the route merges once at the model's K, reads the report and, when the answer is unloadable, **merges a second time** at the capped count, both calls inside the single `timer.stage("merge")` block — so `batch_merge_calls` stays **1** and the second pass shows up as time in that block. Cold, the clamped rows pay 1 219.5 / 1 048.9 / 1 395.2 / 1 209.5 / 285.5 / 221.9 ms of merge against `b16`'s 233.1 / 204.1 / 280.6 / 247.0 / 94.5 / 41.4 ms for the *identical final band shape* — **+0.18 s to +1.11 s per cold call**, **4.90–5.35×** on the four pop-256 wide-row plans (3.02–5.35× across all six clamped rows) — and even the three unclamped per-row rows pay 1.69–3.00× (the model run plus a merge at the model's K), so `auto`'s whole cold call is **1.12–1.35×** the cheapest of its column at pop 256 (`auto/SUMMARY.md` §3).

### 14.5 The restated verdict

**`EVOGP_EVAL_MODE=batch` remains a fixed-population route: in a generation loop dense wins every covered cell by 43.2–95.4×, and at pop 256 dense wins the warm call on every column of the shipped-default shape (batch at 0.31–0.79× of dense's warm time).**

The loop numbers (`Σ` over 6 `StandardPipeline.step()` generations, `composition/SUMMARY.md` §5): `p256-L32` **12 906.6** ms chunked / 14 387.1 `auto`-chunked / 24 171.6 flat-chunked / 24 714.0 `auto`-banded / 25 578.3 `b16`-banded against dense's **299.0** ms; `p256-L64` 14 710.4 / 15 537.6 / 28 871.7 against **302.7** ms. Every batch route records `generation_hit_rate` **1/6** with **6 distinct signatures** and `generations_with_compile` **5** (`compile_ms_share` 0.979–0.991): generation 1 hits the graph the timed first call compiled and each later generation presents a new structure multiset.
The sharper, eval-only reading excludes generation 1 (which pays the pipeline's first trace on both sides): the eval sums are 15.3 ms for dense against 12 623.1 ms for `b16-chunked` at L32 = **825×**, and 22.1 against 14 427.4 ms at L64 = **653×** — the `Σ` ratio of 43× is the *flattering* number because generation 1 dominates dense's `Σ`.

What survives, unchanged in kind from §7/§11:

- **A FIXED population of ~64 trees over few data rows.** At `p64-L64-i32-N64` the shipped default is **1.20× / 1.54×** dense's speed on `sr_fitness` / `batch_forward` (0.81× on the per-row `forward`, §14.2), and its cold call is 2 284.2 against 6 177.7 ms = **2.70× cheaper** (§14.3).
- **The banded+chunked composition is the least-bad loop route and still loses:** 43.2× / 48.6× against 80.8–95.4× for the single-body routes — chunking amortizes one compile four ways, never a loop.
- **The long-tree crossover of §11 is unchanged this round:** no long-tree cell was re-measured, and the default's band change does not touch that comparison, which is about dense's `EVOGP_DENSE_MAX_TREE_LEN` trace cap rather than about the merge's band partition.

What did **not** change on the default path: `auto` never picks `batch`; `EVOGP_BATCH_BANDS` stays unset by default (so `auto` is opt-in only); `EVOGP_BATCH_CHUNK_BANDS` stays off; and the shipped default's chunked cold shape is the one round 1 already had (`4 444.2` ms at `p256-L32`, §13.2). What did change is one number: the band size, 32 → 16 — a **3–6 % second-order shift** whose case is the worst column, not the mean (§14.1).

### 14.6 Coverage statement

Measured: 3 cells (`p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p64-L64-i32-N64`) × {`b8`, `b16`, `b32`, `b64`, `flat`, `auto`} at pool 1 (15 + 9 records, `--warmup 1 --reps 5`), plus `b16-cw4` / `auto-cw4` / `flat-cw4` (9 records) and 10 generation-loop routes × 6 generations — every record `status: ok`, and the cold column reproduced by a `--cold-only` wave (§14.7).
Not measured, with the reason: `p1000-*` (a merged compile of 91–162 s per workload per config on the single-executable path, i.e. 8–14 min per config for three workloads, and the composition's scale envelope there is untested); **no explicit `EVOGP_BATCH_BANDS=K` pinned AT the clamp ceiling (K = 15 / K = 7)**, so "auto clamped" is measured against `b16`/`b32` and not against the cap itself — that is the one surviving experiment; no cold `-cw4` column at `p1000`; no `EVOGP_BATCH_ALIGNMENT` / `EVOGP_BATCH_SWITCH_OPS` variants; and no `CustomLoss`/multi-output or robotics workload.

Protocol gaps, stated as such:

- **`batch_economics.py` has NO idle-device guard.** No `guard` key exists in `composition/economics.json`, `l64/economics.json` or any loop record, so idleness was verified **by hand** with `rocm-smi --showpids` before each run and the runs kept strictly sequential (6 children, then 4) — a weaker protocol than the sweeps', which recorded `gpu_pids.idle: true` / `contaminated: false` and 0 failures on **all** of their configs (15/15 in item 1, 9/9 in item 2, 9/9 in the composition sweeps) and waited for a foreign job on five of them in item 1 (10.6 / 94.0 / 73.1 / 88.7 / 104.4 s) and two in item 2 (42.0 / 57.6 s), `waited_s` 0.2 s on every composition config — waits recorded verbatim, nothing re-run or smoothed.
- **Two worktrees, one evoxc.** The `auto-cw4` / `flat-cw4` sweep ran from `c97a6630` and the `b16-cw4` sweep plus both loops from `0b9075f1`, with identical cell `population_signature`s (`ad11cb66ffc21ac8` / `0a0dee3fec0cfb9c` / `18d209693a5a7e9e`) and one evoxc commit (`2b289f03`) everywhere, so no split is attributable to the engine or to the population.
- **Not gateable.** `check.py` reads the campaign *arm-record* schema, so on the sweep directories it reports `224` checks / `28` failures (`auto/`) and `362` / `46` (`../warm/`) of the `cache`-block kind, and it cannot gate an economics directory at all. These records are **evidence, not gateable units**; their internal checks are `status: ok`, the stage-split closure, the guard block and the per-workload oracle validation.
- **The oracle deviation is IDENTICAL in every config of a cell/workload (9/9)**, i.e. the band partition is a scheduling decision and never a numeric one: `p256-L32` `sr_fitness` 6.0 / 4.74e-07, `batch_forward` 5.86e-03 / 1.13e-03, `forward` 0 / 0 (**bitwise**); `p256-L64` 32.0 / 7.76e-07, 7.81e-02 / 1.14e-05, 1.19e-07 / 8.01e-08; `p64` 0.25 / 1.62e-07, 6.10e-05 / 1.68e-07, 1.19e-07 / 8.01e-08 — no NaN or inf mismatch anywhere. Value digests split only at **2e-12–1e-10** relative on the value tensors, and `sr_fitness` is bitwise identical across the partitions in every cell.

### 14.7 The round-3 one-command re-measure path

A single `--cold-only` wave prices an upstream evoxc `lower()`/trace speedup without paying for warm medians: one synchronized first call per workload, no warmup, no reps. Verbatim as run (`composition/SUMMARY.md` §11):

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

Both were executed, **exit 0**, **3/3 configs `ok` each** with 0 failures and 0 skipped, driver `elapsed_s` **93.3 s** and **40.0 s** (≈2.2 min for the pair); the records carry `cold_only: true`, `samples_ms: []` and every warm field `null`, and the cold numbers **reproduced at 0.933–1.017× of the `--reps 5` calls** (worst low `b32 sr_fitness` at `p256-L32`, 11 251.8 against 12 063.3; worst high `auto batch_forward` at `p64`, 1 653.0 against 1 626.0).
A re-measure always writes a **fresh `--out-dir` per cell** — that is why `round3-cold/<cell>/` exists, so a per-cell re-run does not overwrite another cell's `sweep.json` — and `results/final/`, `results/baseline/`, `results/batch/` and `results/batch_opt_round1/` are refused as output directories (`band_sweep.py`'s `FROZEN_RESULT_DIRS`; `reuse_probe.py` carries the first three).

## 15. Round 3: the refreshed cold/warm/loop columns, the compile-pool default and the band cap (results/batch_opt_round3/)

Round 3 re-prices the shipped default (band size 16, `EVOGP_BATCH_COMPILE_WORKERS` 4, chunked) on the **same three cells, populations and seeds as §14** against two references — a pool-1 single-body control at the identical band partition, and `dense` — in one window on the `/tmp/evoxc_r4` snapshot, and it settles the two knobs §14 left as "as shipped" with measured break-evens: the compile-pool width and the band size pinned at the rocm binding cap.
It adds three things round 2 did not have: an **upstream-refreshed cold column** (§15.4 attributes it), a **pop-1000 cold/loop reading on the chunked composition** (round 2 priced only the single-executable path there), and a **long-tree pool trade** at `p256-L128-i32-N1024-d7`.
The decision of record: **nothing on the shipped path changes** — pool stays 4, band size stays 16, `EVOGP_BATCH_BANDS=auto` and `EVOGP_BATCH_CHUNK_BANDS` keep their §14.1 state, and `auto` still never resolves to `batch`.
Every number below is read off a committed record or from `results/batch_opt_round3/SUMMARY.md`, which carries the full per-config tables, the stage splits, the loop rows, the evidence layout and the reproduce block.

### 15.1 The per-cell table: warm, cold and the generation loop against dense

`batch` = the shipped default with **no** `EVOGP_BATCH_*` variable set (`route_knobs` reads `source: module` for band size 16, pool 4 and `chunk_bands off`), i.e. the policy itself; `dense` = `evoxc-dense-iree` through the same public call.

| cell | warm `sr_fitness` batch (ms) | warm dense (ms) | batch ÷ dense | §14.2 convention (dense ÷ batch) | cold batch (ms) | cold dense (ms) | batch ÷ dense | loop `Σ` batch (ms) | loop `Σ` dense (ms) | `Σ` ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 3.7250 | 1.1015 | **3.382** | 0.30 | 4 064.48 | 3 355.83 | **1.211** | 19 626.71 | 302.48 | 64.9 |
| `p256-L64-i32-N1024` | 3.7720 | 2.0560 | **1.835** | 0.54 | 4 071.43 | 6 526.78 | **0.624** | 22 346.79 | 312.48 | 71.5 |
| `p64-L64-i32-N64` | 1.1141 | 1.4378 | **0.775** | 1.29 | 2 274.46 | 6 554.88 | **0.347** | 5 587.44 | 313.26 | 17.8 |

(`results/batch_opt_round3/economics/item1/<cell>/{batch,dense}.json` — `sr_fitness.median_ms`, `first_call.first_call_ms`, `summary.sum_total_ms`; the `Σ` ratio is computed here, and §14.2's convention is inverted so > 1 still means "batch faster".)
The loop ratio is the **flattering** number, exactly as in §14.5: dense's `Σ` is six warm generations of 2.9–4.2 ms each plus one pipeline warm-up, while every batch generation after the first pays a fresh merged-graph compile.
The eval-only reading: dense's whole-loop `sum_eval_ms` is **22.28 / 32.42 / 23.11** ms against batch's per-recompiling-generation `median_cold_eval_ms` of **2 600.93 / 2 894.58 / 588.22** ms (and a warm generation's 3.78 / 3.81 / 1.15 ms) — an eval `Σ` ratio of **868 / 680 / 229×**, i.e. the honest gap is 3–13× wider than the `Σ` column says.
Cold, the ordering is not uniform and the reason is dense's stride sensitivity: batch is **1.211× dearer** at `p256-L32`, **0.624×** at `p256-L64` and **0.347×** at `p64`, and dense's cold call tracks the trace stride rather than the population (3 355.83 ms at L32 against 6 526.78 / 6 554.88 ms at L64, p256 and p64 alike).
Warm, §14.2's headline holds in the new window: at pop 256 the shipped default is 1.8–3.4× dense's warm call, and at `p64` it is 1.29× faster — the fixed-population niche, unchanged in kind.

### 15.2 The compile-pool policy: `EVOGP_BATCH_COMPILE_WORKERS` stays 4, and its break-even is measured

The pool splits one merge into concurrent chunks: cheaper cold, four upload / run / read-back trios per warm call. Break-even = Δcold ÷ Δwarm, i.e. how many warm calls **of one population** a run must make before the dearer cold call is repaid.

| cell | cold pool 1 → pool 4 (ms) | Δcold | warm pool 1 → pool 4 (ms) | Δwarm | break-even (median) | break-even (min) | cold speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 10 725.86 → 4 064.48 | 6 661.38 | 1.9941 → 3.7250 | 1.7309 | **3 849** | 3 981 | 2.64× |
| `p256-L64-i32-N1024` | 11 957.38 → 4 071.43 | 7 885.95 | 2.2267 → 3.7720 | 1.5453 | **5 103** | 5 083 | 2.94× |
| `p64-L64-i32-N64` | 2 268.96 → 2 274.46 | −5.51 | 1.1494 → 1.1141 | −0.0353 | **n.a. (no trade)** | n.a. | 0.998× |
| `p256-L128-i32-N1024-d7` | 38 513.95 → 8 410.96 | 30 102.99 | 3.4851 → 5.0947 | 1.6096 | **18 702** | 20 648 | 4.58× |

(`results/batch_opt_round3/economics/item1/<cell>/{batch,batch-b16-banded}.json` and `results/batch_opt_round3/item2/long-tree/p256-L128-i32-N1024-d7/{batch,batch-b16-banded}.json`; break-even computed from those fields, not quoted.)
The default pays **1.69–1.87×** per warm `sr_fitness` call at the two pop-256 cells (1.46× at the long-tree niche) and buys **2.64× / 2.94× / 4.58×** on the cold call — repaid only after 3.8k / 5.1k / 18.7k warm calls of a **single** structure multiset, which a GP loop never makes (§15.5).
At `p64` the pool is **inert on the `sr_fitness` and `batch_forward` plans** (`chunk_count 1`, `chunk_sizes [63]`, Δcold −5.51 ms) — and `p64` is exactly the regime where batch beats dense warm, so flipping the default to 1 would buy nothing where the route wins and would make the cold call 10.7 s (pop 256) / 38.5 s (long trees) instead of 4.1 s / 8.4 s.
**Documented escape hatch, not a new default:** a workload that evaluates one fixed multiset more than ≈3.8k / 5.1k / 18.7k times should set `EVOGP_BATCH_COMPILE_WORKERS=1` explicitly; `src/evogp/tree/batch_knobs.py::DEFAULT_COMPILE_WORKERS` stays 4.
The cross-check confirms the default's plan shape is the composition §14.2 identified: pinning `EVOGP_BATCH_BAND_SIZE=16` + `EVOGP_BATCH_CHUNK_BANDS=1` + pool 4 reproduces `compile_workers 4` / `chunk_count 4` / `chunk_sizes [57,56,56,56]` and `[58,58,58,58]` / `chunk_bands [4,4,4,4]`, cold within −0.25 % / −0.03 % and warm within −2.7 % / +3.3 % (`results/batch_opt_round3/economics/crosscheck-b16-chunked/<cell>/batch-b16-chunked.json`).

### 15.3 The band-size policy after item 3: `EVOGP_BATCH_BAND_SIZE` stays 16

`k15` (and `k7` at `p64`, where 15 bands are illegal: an explicit `K` needs `pop >= 8K`) was measured against the shipped size 16 in **both** pool widths, three workloads per cell, one window.

| cell | pin vs `b16`, `sr_fitness` | `batch_forward` | `forward` | plan change? |
|---|---:|---:|---:|---|
| `p256-L32-i8-N1024` (pool 1) | −5.9 % | +2.3 % | −6.9 % | **no** — `[15]×15` and the same `emitted_ops` 5 037 / 4 153 / 4 540 |
| `p256-L64-i32-N1024` (pool 1) | +0.1 % | +1.2 % | −2.6 % | **no** — `[16]×7 + [15]×8`, identical ops 5 475 / 4 568 / 5 861 |
| `p64-L64-i32-N64` (pool 1) | +0.04 % | +1.3 % | −4.6 % | **yes** — K 4 `[16,16,16,15]` → K 7 `[9]×7`, ops 1 393 → 1 463 |

(`results/batch_opt_round3/item3/p256/<cell>/<config>.json` and `item3/p64/p64-L64-i32-N64/<config>.json` — `workloads.<workload>.median_ms` and `.band_report`; the pool-4 pairs sit at −3.6…+3.8 % with one −15.1 % `forward` row on an *identical* plan, the band-sensitive column of §14.1/§14.2, SUMMARY §4.3.)
At pop 256 the pin is a **same-merge control, not a different plan**: identical band shapes and identical `emitted_ops`, the only difference being that the size model asks 16 bands and is clamped to 15 on the per-row plan while `k15` asks the count (`band_requested explicit`, `band_clamped false`).
At `p64` the pin does change the plan and still lands inside this box's ~5 % cross-config noise floor.
**No pin beat the shipped size anywhere in the window**, so the band-size policy is unchanged (no `src/` change) and keeps satisfying band ≥ 8 functions / `K ≤ 15` on rocm.
The partition is also **numeric no-op** again this round: the oracle deviation is **exactly identical in all four configs of every cell/workload (9/9 rows)** — `p256-L32` 6.0 / 4.74e-07, 5.86e-03 / 1.13e-03, 0 / 0 (bitwise); `p256-L64` 32.0 / 7.76e-07, 7.81e-02 / 1.14e-05, 1.19e-07 / 8.01e-08; `p64` 0.25 / 1.62e-07, 6.10e-05 / 1.68e-07, 1.19e-07 / 8.01e-08 — and value digests split only at ≤ 5.8e-11 relative `abs_sum`, by **pool** at the pop-256 cells and by **band count** on `p64 batch_forward` (`4919287125…` for K = 4 against `85adf166…` for K = 7), with `sr_fitness` one digest group per cell in every config (§4.4).

### 15.4 What the three upstream evoxc speedups did to this route's cold path

Same-shape comparison against §14.3's records on the refreshed snapshot, with the stage split attributing the movement:

| cell / route | cold §14.3 → now (ms) | Δ cold | `lower` §14.3 → now (ms) | `lower` speedup | compile wall §14.3 → now (ms) | Δ wall |
|---|---:|---:|---:|---:|---:|---:|
| `p256-L32` pool 1 | 11 049.41 → 10 725.86 | −2.93 % | 296.61 → 98.83 | **3.00×** (−197.8) | 10 583.57 → 10 264.44 | −3.02 % |
| `p256-L32` default (chunked) | 4 180.63 → 4 064.48 | −2.78 % | 888.52 → 470.05 | **1.89×** (−418.5) | 3 733.14 → 3 630.75 | −2.74 % |
| `p256-L64` pool 1 | 12 590.61 → 11 957.38 | −5.03 % | 286.68 → 97.82 | 2.93× | 12 077.57 → 11 454.15 | −5.16 % |
| `p256-L64` default | 4 422.39 → 4 071.43 | −7.94 % | 985.88 → 376.02 | 2.62× | 3 966.10 → 3 627.39 | −8.54 % |
| `p64` pool 1 | 2 298.69 → 2 268.96 | −1.29 % | 63.50 → 31.66 | 2.01× | 2 106.40 → 2 071.36 | −1.66 % |
| `p64` default (inert) | 2 284.23 → 2 274.46 | −0.43 % | 63.38 → 31.16 | 2.03× | 2 095.67 → 2 078.05 | −0.84 % |

(`results/batch_opt_round3/economics/item1/<cell>/<route>.json` `first_call.stage_ms` against `results/batch_opt_round2/composition/<cell>/b16-cw4.json` / `auto/<cell>/b16.json` `workloads.sr_fitness.stage_first_call`. Round 2's L32 pool-1 pair is wall 10 583.57 over sum 10 172.56 at 1 job, so wall-to-wall is **−3.02 %** where sum-to-sum is **−1.17 %** — quote the wall column when comparing calls.)
The honest ceiling on that win is set by where the cold call's time is: `lower` is **0.8–0.9 %** of a pool-1 cold call at pop 256 (**9.2–11.6 %** of a chunked one), while the device-code compile inside iree is **89.1–95.8 %** of it — 95.7 % (`p256-L32`) / 95.8 % (`p256-L64`) pool 1, 89.1–89.3 % chunked, 91.3 % at `p64`, 97.4 % pool 1 at the long-tree cell, 89.7 % at pop 1000.
A 1.9–3.0× faster `lower()` therefore moves a 4–12 s cold call by **0.4–7.9 %**, which is what was measured: the three upstream speedups help this route's cold path but cannot fix it, because the cost is not in the MLIR construction.
Pop 1000 is **new information, not a speedup measurement** — §14.3's 91.0–162.0 s there is the single-executable (`W=1`) path, while the shipped default now reads cold **15 517.88 ms** (`compile` sum 48 931.70 over 4 jobs → wall 13 921.97, 3.51× over serial; `lower` 1 054.29, `merge` 967.07, `trace` 557.71, `load` 298.33), warm `sr_fitness` **6.9384 ms**, plan `chunk_sizes [205, 204, 204, 204]` for 817 deduped functions (`results/batch_opt_round3/item4/p1000-L32-i8-N1024/p1000-L32-i8-N1024/batch.json`).
The round's pre-registered 4-minute bound on a single cold compile **did not trigger** (15.5 s against it), so that cell is measured in full and nothing was dropped — its driver config reads `status: "ok"` / `wall_s` 34.14 against a `protocol.timeout_s` of 1 500 (`results/batch_opt_round3/item4/p1000-L32-i8-N1024/economics.json`; the economics schema has no `skipped` field).

### 15.5 What would have to change for batch to win a full-turnover GP loop

**Population-specific compilation is the blocker, and this round prices it precisely.** Every batch route compiled **7 of 8** generations (`generation_hit_rate` 0.125, `generations_with_compile` 7, **8 distinct population signatures in 8 steps**); the long-tree 2-step check reads 0.5 (1 of 2) and pop 1000 the same.
The compile attribution on the single-body routes is `summary.compile_ms_share` **0.951–0.995** (wall time of the recompiling generations over the loop's total wall time), and on the chunked routes the stage-counter `compile_share.share` reads **3.04 / 3.14** at the two pop-256 cells and **3.17** at pop 1000 (**0.85** at `p64`, where the plan does not chunk) — a pool-weighted counter ratio (four concurrent per-chunk timers over one loop wall), never wall-clock attribution, exactly as §14.3's counter-sum note explains.
Per recompiling generation the eval call alone is **2 600.93 / 2 894.58 / 588.22 ms** (`median_cold_eval_ms`) against dense's **whole generation** of 2.9 / 4.2 / 2.9 ms.
The amortization arithmetic, in the three regimes the route has:

- **At pop 256 there is no N.** Batch's warm call is 3.4× (L32) / 1.8× (L64) *slower* than dense's, so no number of repeated generations repays the compile — the steady state loses before the amortization starts.
- **At `p64` (fixed population, where batch is 1.29× faster warm) the same multiset would have to repeat for ≈7 026 generations** to repay its 2 274.46 ms cold call at 0.3237 ms saved per generation — a fixed-population workload, never a GP loop.
- **At the long-tree niche** batch beats the shipped cap's `specialized` fallback by 23.5–25.0× per call (§11, not re-measured this round); priced against a 5.09 ms batch call that is a ≈127 ms fallback, i.e. **≈69 generations** to repay the 8 410.96 ms chunked cold call — *if* the multiset repeated, which it does not (generation 2 of that cell recompiles: 4 exec misses, 3 933.60 ms, against 8 601.53 ms single-body, 2.19× for chunking that one compile).

So the gap between the niche win and a GP loop is a **structure-reuse** problem, not a band, pool or `lower()` problem: the only routes to a batch win in a loop are amortizing the compiled graph across a *repeated* population shape — which is dense's per-shape amortization, and the `p64` fixed-population regime — or a workload that is fixed by construction.
Nothing in this round changes that, and no knob setting reaches it.

### 15.6 Round-3 re-measure commands

Verbatim as run (the `--note` strings are the records' own), from the repository root, one driver invocation per item:

```bash
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r4:src:test/benchmark/evoxc
PY=/tmp/rocvenv/bin/python

$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --routes batch,batch-b16-banded,dense \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/item1 \
  --note "round3 item1: shipped default (band 16, pool 4, chunked) vs pool-1 control (batch-b16-banded) vs dense, same window, evoxc_r4"

$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --routes batch-b16-chunked \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/crosscheck-b16-chunked \
  --note "round3 item1 cross-check: pinned band16+CHUNK_BANDS=1+pool4 against the unpinned shipped default"

$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L128-i32-N1024-d7 --routes batch,batch-b16-banded \
  --generations 2 --warmup 2 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item2/long-tree \
  --note "round3 item2: long-tree niche (d7), pool-1 control vs the chunked default, warm+cold, 2-generation loop (not a loop claim)"

$PY test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 --configs b16,b16-cw4,k15,k15-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p256 \
  --note "round3 item3: K=15 pinned at the rocm binding cap vs the shipped band size 16, pool-1 partitions and pool-4 compositions, same window"

$PY test/benchmark/evoxc/band_sweep.py \
  --cells p64-L64-i32-N64 --configs b16,b16-cw4,k7,k7-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p64 \
  --note "round3 item3: K=7 pinned vs band size 16 at the fixed-population niche cell, pool-1 and pool-4, same window"

$PY test/benchmark/evoxc/batch_economics.py \
  --cells p1000-L32-i8-N1024 --routes batch \
  --generations 2 --warmup 1 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item4/p1000-L32-i8-N1024 \
  --note "round3 item4: p1000-L32-i8-N1024 shipped default only, cold + one warm reading + stage split + 2-generation loop, faster evoxc lower()"

EVOGP_EVAL_MODE=batch $PY test/benchmark/evoxc/validate.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64,p1000-L32-i8-N1024 \
  --func-sets arith,rich --max-oracle 64 --seed 0 \
  --out test/benchmark/evoxc/results/batch_opt_round3/validation/validation-batch.json \
  --note "round3: the batch route (shipped default knobs, evoxc_r4) vs the plain-torch oracle on the item-1/item-4 cells"
```

A re-measure always writes a **fresh `--out-dir`**, and `results/batch_opt_round2/` is now refused by **both** drivers (`batch_economics.py`'s `FROZEN_RESULT_DIRS` is `("final", "baseline", "batch_opt_round2")` — its own default out-dir is `results/batch`, which is why `batch` is not in that refusal list — and `band_sweep.py`'s is `("final", "baseline", "batch", "batch_opt_round1", "batch_opt_round2")`), so round-2 evidence can never be overwritten by a round-3 rerun.
`batch_economics.py` defaults `--evoxc-root` to `/root/evoxc`, which is why the live runs above pass `--evoxc-root /tmp/evoxc_r4` explicitly; `band_sweep.py` already defaults to it.

### 15.7 Coverage statement

Measured this round: the shipped default, the pool-1 control and `dense` at `p256-L32-i8-N1024` / `p256-L64-i32-N1024` / `p64-L64-i32-N64` (cold, warm `sr_fitness`, 8-generation loop, ten-stage cold split); the pinned-vs-unpinned plan-shape cross-check at the two pop-256 cells; the pool trade at `p256-L128-i32-N1024-d7` (cold, warm, 2-generation check); `k15` / `k7` against `b16` in pool-1 and pool-4 form on all three workloads of those three cells; `p1000-L32-i8-N1024` shipped default (cold, warm, 2-generation check); and a `validate.py` oracle sweep over 4 cells × {`arith`, `rich`}.
Not measured, with reasons: no `EVOGP_BATCH_ALIGNMENT` / `EVOGP_BATCH_SWITCH_OPS` variants (unchanged from §14.6); no `CustomLoss`/multi-output and no robotics workload (the route has never been timed there); no pop-1000 **single-body** control (a 4-minute-plus compile for a column §15.2's break-even already settles at pop 256 and in the long-tree niche); and no re-measure of `auto`, `flat`, `b8`, `b32` or `b64` — those are carried from §14.1/§14.3/§14.4 with their verdicts intact.
Two protocol gaps are **inherited, not fixed**: `batch_economics.py` has no idle-device guard (its records carry no `guard` block, so idleness was hand-checked with `rocm-smi --showpids` and the runs kept strictly sequential — 14 children), and neither an economics nor a band-sweep directory is a `check.py` gateable unit (`check.py` reads the campaign arm-record schema; on the round-3 directories it reports 25 failures over 196 checks at `item3/p256`, 13/99 at `item3/p64`, 16/62 on `economics/item1`, 7/18 on the cross-check, 5/16 at `item2/long-tree` and 4/10 at `item4`, all of the `cache`/`workloads`-schema kind plus a missing `campaign.json`, exactly as §14.6 records for round 2).
The sweep records themselves are clean on their own terms: **12/12 `status: ok`**, `guard.state.gpu_pids.idle true` / `contaminated false` / `waited_s 0.2` on every one, both `sweep.json` files `failures: 0` / `skipped: 0`, and all **14** economics configs `status: ok` in their driver records.
### 15.8 Scale boundary: the large-scale round
Every number above holds inside this round's window — pop ≤ 1 000, `N` ≤ 1 024, trees ≤ 111 nodes at the top of the long-tree column — and the route's picture changes at its edges.
The large-scale round (pop 2 048 / 4 096 / 8 192, `N` = 4 096 / 16 384, 123- and 201-node trees) is `LARGE_SCALE.md`, with its evidence in `results/largescale/` (campaign `main/`) and `results/largascale/` (economics, band and pool sweeps).
It **revises** the picture at pop ≥ 2 048 and at `N` ≥ 4 096 rather than confirming it: the merged graph's warm call beats `dense+iree` by 3.0–6.2× once the rows dominate, and by 4.2× against a raised-cap dense on 123-node trees, while the compile envelope — not the partition and not the warm call — disqualifies the top of the population axis (no `p8192` plan finishes inside 900 s; the `d8` plan times out), so a full-turnover loop still loses by 34.5–111× over 3 generations.
The band-size verdict of §14.1 survives for a new reason there: at pop 2 048 every banded request is clamped to the binding cap, so `b8`/`b16`/`b32`/`k15` produce one identical merge, and only the flat merge (3.1–3.2× worse warm) still separates.
The pool-width decision of §15.2 is left **open** for a compile-dominated loop: pool 8 is 0.61× the cold call and 1.17× the warm one there, and no loop arms were measured at that width.
