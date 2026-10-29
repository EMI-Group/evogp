# The opt-in `batch` evaluation route — measurement report of record

Verdict in one paragraph: the `EVOGP_EVAL_MODE=batch` route (one merged `evoxc` graph per population
structure multiset) is **not the best arm anywhere except one narrow corner**, and it is **dominated by
the shipped `dense` route at every pop ≥ 64 cell we measured**. Warm it is 1.0–2.7× slower than
`dense+iree` at pop 256–1000 (0.55× / 0.995× / 0.37× / 0.94× of dense's speed) while still being
18–30× faster than `specialized+numpy`; cold its merged-graph compile is 10.9–12.2 s at pop 256 and
91–162 s at pop 1000 against dense's 3.4–6.5 s; and in a real generation loop a crossover/mutation-
updated population is a **new structure multiset**, so the merged graph is rebuilt almost every
generation (1 hit in 8 generations, 99.4–99.7 % of generation time inside the compile), which makes
`gp_step` 193–2355× slower than `dense`. The one cell where it wins is pop 64 with 64 data rows
(`0.979 ms` vs dense's `1.410 ms`, and a *cheaper* cold call: 2.17 s vs 6.18 s) — and even there it
loses in a generation loop. `auto` never selects the route, so nothing on the default path changes.

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

## 7. The one regime where the route wins

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

## 11. Caveats and known issues

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
- The commit-era history of the `test/benchmark/evoxc` harness (`results/final/**`, `results/baseline/`,
  the R1–R6 rounds) is untouched by this round; a re-run of the batch grid writes a fresh directory.
