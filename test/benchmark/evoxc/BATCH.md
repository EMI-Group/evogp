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
