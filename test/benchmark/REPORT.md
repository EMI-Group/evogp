# REPORT.md — R2 GPU round: legacy CUDA vs ETL "true jit" (iree-cuda + numpy), warm/steady-state

**Executive summary.** On one shared box (8× RTX A6000, GPU 6) and through the same public `evogp`
API, the same fixtures/seed and one process per config, the legacy custom-CUDA engine is **110×–363×
faster than the ETL engine's `iree` (IREE-cuda, O1) backend** and **33×–148× faster than the ETL
engine's default `numpy` backend** on the fixed-forest evaluation workloads, measured on warm
(steady-state) medians (per-call median, compile/first-call excluded). On a fully warm `sr`
generation the gap narrows to **41× (iree) / 14× (numpy)**. The two new backends are numerically
interchangeable — iree-O1 fitness is bit-identical to numpy on both fixtures
(`r2/backend_agreement.md`) — so the ~2.5–3.4× that separates them is pure execution overhead:
**IREE-cuda is ~3.3× slower, warm, than the CPU reference interpreter** at these problem sizes. The
gap is structural, not arithmetic: the new engine issues **K per-tree specialized invocations plus K
device readbacks per `evaluate`** (K = 53/171 here) where the legacy engine issues one batched
launch.

## Scope & method

* **Headline policy**: the headline is warm/steady-state cost only — `median_warm_ms` for eval tasks
  (= median of `per_call_ms[1:]`) and `median(per_gen_ms[1:])` per rep for training tasks. Compilation
  and cold first-call numbers are reported **separately** in §5 and are never mixed into the headline
  (`cold first call` = `first_call_ms` / `cold_first_gen_ms`).
* **Same workload, same wiring**: one fixed forest per workload rebuilt through the public class
  constructor from the fixture arrays (`test/goldens/forest_random.npz`, 64 trees / 63 nodes;
  `test/benchmark/fixtures/forest200_tree64_random.npz`, 200 trees / 64 nodes, provenance in
  `fixtures/README.md`), `SymbolicRegression(datapoints=x, labels=labels)`, seed 0, RNG-independent at
  measurement time; both sides are driven by the side-agnostic harness `bench_gp.py`
  (`test/benchmark/r2/README.md`, method of record `SPEC.md`).
* **One process per config** (`run_all.sh`): the engine's `etl.CompileCache` and the legacy CUDA
  context are per-process, so every config's cold number is genuine and no config can inherit another's
  warm cache (`r2/README.md` §Rules).
* **Box**: all 36 R2 config JSONs record `CUDA_VISIBLE_DEVICES=6`, `NVIDIA RTX A6000`,
  `torch 2.6.0+cu124`, `etl 0.1.0`, `ETL_DISABLE_LOCATIONS=1`; the 12 iree configs add
  `EVOGP_ETL_BACKEND=iree`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`, `EVOGP_ETL_IREE_DEVICE=0`. GPU 7
  (uncorrectable-ECC fault) was never touched. One `flock -w 7200 /tmp/gpbench_critical.lock` per
  config, never nested, lock wait outside all timed regions, `timeout` inside the lock.
* **Noise**: medians are ±10% on this shared box (`r2/README.md`); compile numbers are ±a factor (§5).
  The stage probe is instrumented at runtime only (`stage_probe.py`, monkeypatching; `src/` untouched)
  and its tax is ≲1 ms per eval call (`np_c5r2` 21.056 vs `np_c5r2_noprobe` 20.838 ms = +1.05%; the
  `iree_fixture64_n64` pair +2.34%), i.e. inside the noise band.
* **Backend agreement (gate)**: `r2/backend_agreement.md` — iree-O1 vs numpy are bit-identical on both
  fixtures (`max |diff| = 0.0`, identical `values_sha256` `b8192cb1770b886d…` (64-tree) /
  `07948efededc27b2…` (200-tree), identical NaN masks), so the timing comparison is apples-to-apples.
* **Raw data**: every number below is recomputed from the committed harness JSONs
  (`new/results/r2/*.json`, `legacy/results/r2/leg_*.json`) or quoted from the committed round log
  (`r2/RUNLOG.md`) / pre-flight (`r2/backend_agreement.md`); the per-config `.log` files hold the exact
  commands and the GPU snapshots.

## 1. Headline warm table

Warm medians in ms, as committed. Legacy columns: `leg_c5r2`, `leg_fixture64_n1024`,
`leg_fixture64_n10000`, `leg_fixture200_n64`, `leg_fixture200_n1024`, `leg_fixture200_n10000`,
`leg_fresh16`; iree columns: `iree_fixture64_n64`, `iree_fixture64_n1024`, `iree_fixture64_n10000`,
`iree_fixture200_n64`, `iree_fixture200_n1024`, `iree_fixture200_n10000`, `iree_fresh16`; numpy
columns: `np_c5r2`, `np_fixture64_n1024`, `np_fixture64_n10000`, `np_fixture200_n64`,
`np_fixture200_n1024`, `np_fixture200_n10000`, `np_fresh16`.

| workload (rows = data rows) | legacy ms | new iree ms | new numpy ms | legacy/iree | legacy/numpy | iree/numpy |
|---|---|---|---|---|---|---|
| fixture64, n64 (native) | 0.633 | 69.61 | 21.056 | **110.0×** | **33.3×** | 3.31× |
| fixture64, n1024 | 0.639 | 73.39 | 21.505 | **114.9×** | **33.7×** | 3.41× |
| fixture64, n10000 | 0.682 | 79.37 | 23.120 | **116.4×** | **33.9×** | 3.43× |
| fixture200, n64 (native) | 0.632 | 229.49 | 93.674 | **363.2×** | **148.2×** | 2.45× |
| fixture200, n1024 | 0.681 | 238.46 | 96.518 | **350.0×** | **141.7×** | 2.47× |
| fixture200, n10000 | 1.019 | 260.36 | 105.371 | **255.4×** | **103.4×** | 2.47× |
| evalfresh, pop 16 / tree 32 | 0.632 | 24.50 | 9.003 | **38.8×** | **14.3×** | 2.72× |
| sr_a, pop 200 / tree 64 — warm per-gen, rep1 (fully warm) | 1.59 | 65.31 | 22.07 | **41.1×** | **13.9×** | 2.96× |
| sr_a, pop 200 / tree 64 — warm per-gen, rep0 (churning population) | 1.62 | 10 733.32 | 84.09 | **6 639×** | **52.0×** | 127.6× |
| `iree_sr_b2` (pop 1000 / tree 128 / 2 gens) | — | **CRASH** | (numpy analog `np_sr_b` 650.36) | n/a | n/a | n/a |

Sources: `legacy/results/r2/leg_c5r2.json` (0.6327), `leg_fixture64_n1024.json` (0.6386),
`leg_fixture64_n10000.json` (0.6819), `leg_fixture200_n64.json` (0.6319), `leg_fixture200_n1024.json`
(0.6814), `leg_fixture200_n10000.json` (1.0192), `leg_fresh16.json` (0.6317), `leg_sr_a.json`
(rep0 1.62 / rep1 1.59); `new/results/r2/iree_fixture*_{n64,n1024,n10000}.json`, `iree_fresh16.json`
(24.502), `iree_sr_a.json` (rep0 10 733.32 / rep1 65.31); `new/results/r2/np_c5r2.json` (21.056),
`np_fixture64_n1024.json` (21.505), `np_fixture64_n10000.json` (23.120), `np_fixture200_n64.json`
(93.674), `np_fixture200_n1024.json` (96.518), `np_fixture200_n10000.json` (105.371),
`np_fresh16.json` (9.003), `np_sr_a.json` (rep0 84.09 / rep1 22.07).

Reading:

* The headline answer to "how far apart are the two engines now?": on fixed-forest evaluation the
  legacy engine is **two to three orders of magnitude cheaper per warm call** (110–363× vs iree,
  33–148× vs numpy), and the new engine's warm cost is dominated by per-tree dispatch/readback, not by
  arithmetic (§2–§4).
* The gap **grows with tree count, not with data size**: going from 64 to 200 trees (3.23× the
  invocations) costs iree 3.30× and numpy 4.45×, while the legacy engine is flat in the population
  (0.633 → 0.632 ms at n64) and moves only 7.8% when the rows grow 156× (0.633 → 0.682 ms).
* The `sr_a` rep1 row (65.31 ms/gen iree, 22.07 ms/gen numpy, 1.59 ms/gen legacy) is the honest
  "backend warm cost" comparison: rep1 replays the same seed in the same process, so all 632
  per-generation build lookups are cache hits and `warm_misses == 0`
  (`iree_sr_a.json`/`np_sr_a.json` `stages.reps[1].totals`).
* `iree_sr_b2` produced no measurement (exit 139 SIGSEGV before generation 0) — see §7. Its numpy
  mirror `np_sr_b.json` is the only available cost reference at that scale (650.36/670.08 ms per
  generation, pop 1000 / tree 128).

## 2. Per-call floor

Warm cost normalised by the work actually executed. "per invoked tree" divides by `trees_invoked` per
call (the engine skips constant/collapsed trees: 53 of 64 slots, 171 of 200 slots); "per tree slot"
divides by the population size; the legacy engine has no per-tree loop at all, so its number is one
batched launch divided by the population.

| workload | legacy ms, whole eval | legacy per tree slot | new iree ms/invoked tree | new numpy ms/invoked tree | iree / ETL floor | numpy / ETL floor |
|---|---|---|---|---|---|---|
| fixture64 n64 | 0.633 | 0.00989 | **1.3135** | **0.3973** | 10.0× | 3.03× |
| fixture64 n1024 | 0.639 | 0.00998 | 1.3846 | 0.4058 | 10.6× | 3.10× |
| fixture64 n10000 | 0.682 | 0.01065 | 1.4976 | 0.4362 | 11.4× | 3.33× |
| fixture200 n64 | 0.632 | 0.00316 | **1.3420** | **0.5478** | 10.2× | 4.18× |
| fixture200 n1024 | 0.681 | 0.00341 | 1.3945 | 0.5644 | 10.6× | 4.31× |
| fixture200 n10000 | 1.019 | 0.00510 | 1.5226 | 0.6162 | 11.6× | 4.70× |
| evalfresh pop16 | 0.632 | 0.03948 | 1.6335 | 0.6002 | 12.5× | 4.58× |
| sr_a rep1 (per generation) | 1.59 | 0.00795 | 1.921 | 0.649 | 14.7× | 4.96× |

"ETL floor" = the per-invoke ETL work floor measured in the same round: warm `run_ms`/`run_calls` =
**0.131 ms** (`np_sr_a` rep1: 4.45 ms / 34 invocations) to **0.162 ms** (`np_c5r2`: 8.59 ms / 53
invocations) on the CPU interpreter (`new/results/r2/np_sr_a.json`, `np_c5r2.json`
`stages.reps[].records[i>0]`). The library's own committed instrument probe puts a single warm
`etl.run` of a 30–43-node graph at 0.45–0.49 ms and a `build_cached` cache hit at 0.42–0.45 ms
(`src/evogp/tests/perf_probe.py`, summary table; `src/evogp/tree/CONTEXT.md`), i.e. the same
order — the engine's warm per-tree cost is a small multiple of a bare library call, and the legacy
engine's is two orders below it. In the `sr_a` row the invocation count varies per generation
(174 → 27, median 34 — `iree_sr_a.json`/`np_sr_a.json` records), so the per-invoked-tree figure uses
the median generation's 34 invocations and the per-slot figure normalises one generation by the
population of 200.

Counts, as measured (`stages.reps[0].totals` / per-call record medians):

* Invocations per call: **53** for the 64-tree fixture (11 of 64 slots are constant/collapsed and are
  skipped by the engine), **171** for the 200-tree fixture (29 of 200 slots skipped), 15 for
  `evalfresh pop16`. The 200-tree fixture has **155 distinct structures** (`cold_builds`, the
  compile-side count) — the two numbers are different axes and are not interchangeable.
* Data-size insensitivity: 64 → 10000 rows is **156.25× more rows** and costs only **+14.0%**
  (iree 69.61 → 79.37), **+9.8%** (numpy 21.056 → 23.120) and **+7.8%** (legacy 0.633 → 0.682) on the
  64-tree fixture, and +13.4% / +12.5% / +61.3% on the 200-tree fixture
  (`iree_fixture*`, `np_fixture*`, `leg_fixture*` n64 vs n10000 JSONs). Per-call/per-tree overhead
  dominates the cost at these sizes; arithmetic is a rounding error — the `run` stage moves only
  8.59 → 10.17 ms (numpy) and 35.76 → 26.78 ms (iree) at 53 invocations, with the extra rows showing up
  in `upload` (0.29 → 0.62 ms numpy, 0.50 → 14.68 ms iree).
* Invocation count, not population size, is the cost driver: 200 trees carry 171 invocations (3.23× the
  53 of the 64-tree fixture). IREE scales almost linearly with it (69.61 → 229.49 ms = 3.30×, per-tree
  cost +2.2%), numpy does worse than linearly (21.056 → 93.674 ms = 4.45×, per-tree cost +37.9%), while
  the legacy engine stays flat at ~0.63–1.02 ms for both.

## 3. Stage decomposition (warm, per evaluate)

Per-call medians over warm records (`stages.reps[0].records[i>0]`), ms; `total` is the harness-measured
per-call time, and `total = decode + build_hit + build_miss + run + readback + upload + loss + misc`
(`stage_probe.py` chain; `new/NOTES.md` §R2). K = `run_calls` = `readback_calls` = `trees_invoked` per
call; `h2d`/`d2h` are per call.

| config | backend | decode | build_hit | build_miss | run | readback | upload | loss | misc | total | K | h2d/call | d2h/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture64 n64 | numpy | 1.66 | 9.66 | 0.00 | 8.59 | 0.63 | 0.29 | 0.14 | 0.07 | **21.06** | 53 | 0 | 0 |
| fixture64 n64 | iree | 1.68 | 11.09 | 0.00 | 35.76 | 20.22 | 0.50 | 0.17 | 0.09 | **69.61** | 53 | 1 | 53 |
| fixture64 n1024 | numpy | 1.65 | 9.70 | 0.00 | 8.93 | 0.63 | 0.34 | 0.15 | 0.07 | **21.51** | 53 | 0 | 0 |
| fixture64 n1024 | iree | 1.70 | 11.16 | 0.00 | 34.73 | 21.63 | 4.07 | 0.20 | 0.10 | **73.38** | 53 | 1 | 53 |
| fixture64 n10000 | numpy | 1.66 | 9.76 | 0.00 | 10.17 | 0.64 | 0.62 | 0.18 | 0.08 | **23.12** | 53 | 0 | 0 |
| fixture64 n10000 | iree | 1.72 | 11.36 | 0.00 | 26.78 | 24.62 | 14.68 | 0.25 | 0.11 | **79.37** | 53 | 1 | 53 |
| fixture200 n64 | numpy | 7.28 | 40.79 | 0.00 | 42.50 | 2.03 | 0.79 | 0.19 | 0.16 | **93.67** | 171 | 0 | 0 |
| fixture200 n64 | iree | 7.36 | 46.36 | 0.00 | 108.39 | 65.31 | 1.52 | 0.23 | 0.27 | **229.49** | 171 | 1 | 171 |
| fixture200 n1024 | numpy | 7.33 | 41.24 | 0.00 | 44.49 | 2.03 | 1.00 | 0.21 | 0.17 | **96.52** | 171 | 0 | 0 |
| fixture200 n1024 | iree | 7.63 | 46.35 | 0.00 | 107.98 | 68.64 | 6.17 | 0.27 | 0.31 | **238.46** | 171 | 1 | 171 |
| fixture200 n10000 | numpy | 7.34 | 41.39 | 0.00 | 51.60 | 2.14 | 2.28 | 0.33 | 0.28 | **105.37** | 171 | 0 | 0 |
| fixture200 n10000 | iree | 7.35 | 46.08 | 0.00 | 90.87 | 71.08 | 45.39 | 0.32 | 0.45 | **260.36** | 171 | 1 | 171 |

Source: `new/results/r2/{np_c5r2,np_fixture64_n1024,np_fixture64_n10000,np_fixture200_n64,np_fixture200_n1024,np_fixture200_n10000,iree_fixture64_n64,iree_fixture64_n1024,iree_fixture64_n10000,iree_fixture200_n64,iree_fixture200_n1024,iree_fixture200_n10000}.json`.

Stage meanings (one line each, `new/NOTES.md` §R2 / `stage_probe.py`):

* **decode** — host-side decode of the population tensors into per-tree static programs
  (`decode_population`), once per evaluate; present and near-identical on both backends.
* **build_hit** — memoized `build_cached` lookup for an already-compiled structure (re-encodes the
  static program and re-derives its sha256 cache key on every call).
* **build_miss** — the same call on a cache miss; **0.00 ms on every warm fixture call** and verified
  by `totals.warm_misses == 0` / `warm_build_miss_ms == 0.0` on all six fixture configs.
* **run** — the K per-tree `etl.run` executions of the specialized graphs, inside the tree loop.
* **readback** — per-tree residual `tree_total − build − run`: on iree the per-tree D2H copy of that
  tree's output plus host glue (exactly one per tree: `d2h_calls` = K per call); on numpy host glue
  only (`d2h_calls` = 0) — 0.63–2.14 ms on numpy vs 20.22–71.08 ms on iree.
* **upload** — final result H2D (`etl.from_numpy` of the assembled fitness/reference vector), 1 call
  per evaluate on iree (0.50 → 45.39 ms as rows grow), host-side glue on numpy.
* **loss** — label upload + MSE loss/reduction (`sr_fitness` minus `evaluate`).
* **misc** — the rest of `sr_fitness` (fitting, NaN replacement at the pipeline level for training).

Structural contrast: **the legacy engine has no stage block at all** — its JSONs carry `stages: null`
(e.g. `legacy/results/r2/leg_c5r2.json`, `leg_sr_a.json`) because the harness's probe is new-side-only
and no-ops on the old side; its whole `evaluate` is a single batched CUDA launch sequence with no
host-side per-tree loop, which is exactly the structural difference that produces §4.

Counts and cross-checks (from `stages.reps[0].totals`, same files):

| config | calls | cold_builds | cold_build_ms | warm_misses | h2d total | d2h total | build_calls | trees_invoked / trees_total |
|---|---|---|---|---|---|---|---|---|
| np_c5r2 | 25 | 48 | 151.5 | 0 | 0 | 0 | 1325 | 1325 / 1600 |
| iree_fixture64_n64 | 25 | 48 | 81 854.0 | 0 | 25 | 1325 | 1325 | 1325 / 1600 |
| iree_fixture64_n1024 | 11 | 48 | 74 369.1 | 0 | 11 | 583 | 583 | 583 / 704 |
| iree_fixture64_n10000 | 11 | 48 | 16 423.7 | 0 | 11 | 583 | 583 | 583 / 704 |
| np_fixture200_n64 | 25 | 155 | 882.3 | 0 | 0 | 0 | 4275 | 4275 / 5000 |
| iree_fixture200_n64 | 25 | 155 | 105 201.1 | 0 | 25 | 4275 | 4275 | 4275 / 5000 |
| np/iree_fixture200_n10000 | 11 | 155 | 907.1 / 86 089.4 | 0 | 0 / 11 | 0 / 1881 | 1881 | 1881 / 2200 |
| np_fresh16 / iree_fresh16 | 10 | 15 | 76.0 / 13 950.8 | 0 | 0 / 10 | 0 / 150 | 150 | 150 / 160 |
| iree_sr_a rep1 | 10 | 0 | 0.0 | 0 | 10 | 632 | 632 (all hits) | 632 / 2000 |

## 4. Where the gap sits (reasons, with numbers)

**(i) Per-tree specialized graphs → K invocations + K readbacks per `evaluate` vs one batched legacy
launch.** K = 53 per call on the 64-tree fixture and K = 171 on the 200-tree fixture
(`stages.reps[0].totals`: `run_calls` = `readback_calls` = `trees_invoked` per call; the engine skips
constant/collapsed trees). Each invocation carries its own ETL call overhead and, on iree, its own
D2H copy: `d2h_calls` = 53/call and 171/call respectively. The legacy engine's entire evaluation is one
batched launch: 0.633 ms for the 64-tree fixture (n64), 0.632 / 1.019 ms for the 200-tree fixture
(n64 / n10000). IREE's per-invoked-tree cost is flat in K (1.314 → 1.342 ms from 53 to 171
invocations), so its total scales linearly with K; numpy's rises 38% over the same step (0.397 →
0.548 ms) so its total scales super-linearly; the legacy engine stays constant.

**(ii) Per-invoke overhead ~10× the ETL floor (iree) / ~3–5× (numpy).** The warm engine-side
per-tree cost (§2) is 1.31–1.52 ms (iree) and 0.40–0.62 ms (numpy) against the measured in-round
per-invoke ETL floor of 0.131–0.162 ms (`np_sr_a` rep1 / `np_c5r2` warm `run_ms`/`run_calls`) — 10–15×
and 3–5×. The overhead is code-level and documented in the engine itself: `build_cached` re-encodes
the static program and re-derives its sha256 cache key on **every** `_executable` call (hit or miss),
and `etl.run` re-validates the statics by value in O(#tokens) per call, with the program a nested
tuple of ints (`src/evogp/tree/CONTEXT.md` §Engine; `src/evogp/tree/etl_engine.py` module docstring).
That is why `build_hit` alone costs 9.66 ms (numpy, 53 trees) to 46.36 ms (iree, 171 trees) per
evaluate — more than the `run` stage at 64 trees.

**(iii) Readback + sync serialization.** Warm split for `iree_sr_a` rep1 (65.31 ms per generation):
`readback` 17.12 ms = **26.2%**, `run` 27.02 ms = **41.4%**, `build_hit` 6.09 ms = 9.3%, `decode`
2.43 ms = 3.7%. The fixture evals show the same shape: iree `readback` 20.22 of 69.61 ms = **29.0%**
(fixture64 n64) and 65.31 of 229.49 ms = **28.5%** (fixture200 n64). On numpy, where no device
round-trip exists (`d2h_calls = 0`), the same stage is 0.63 of 21.06 ms = **3.0%** — i.e. on iree the
per-tree device round-trip is ~10× the numpy residual and a quarter of the whole call. Pipelines that
pay one sync per tree cannot amortise anything.

**(iv) Why iree-cuda loses to numpy-CPU here.** The arithmetic is tiny: at most 10000 rows × 171 trees
over ≤64-node trees, so GPU dispatch + sync overhead exceeds any GPU win, and the two backends share
all the host-side work (decode 1.66–7.63 ms, per-tree `build_cached` lookup, Python glue). On top of
that, iree adds round-trips the numpy backend does not have: `run` is 35.76 vs 8.59 ms (53 trees) and
108.39 vs 42.50 ms (171 trees); `readback` 20.22 vs 0.63 ms and 65.31 vs 2.03 ms; `upload` 0.50 vs
0.29 ms at n64 and 45.39 vs 2.28 ms at n10000. Result: iree/numpy = 3.31× (fixture64 n64), 3.43×
(n10000), 2.45× (fixture200), 2.72× (`evalfresh pop16`), 2.96× (`sr_a` rep1, the only fully warm
training comparison), and 127.6× on `sr_a` rep0 where iree's compile misses dominate. The pre-flight
sees the same on bare `evaluate`: iree 75.2 vs numpy 21.9 ms (warm, 64 trees) and 242.2 vs 95.1 ms
(200 trees) at bit-identical outputs (`r2/backend_agreement.md`).

**(v) Host decode + per-call ETL glue are common to both new backends.** `decode` is 1.66–1.72 ms
(64 trees) and 7.28–7.63 ms (200 trees) on **both** backends (within noise of each other), and grows
with the population, not the data size. Together with the per-tree cache lookup (`build_hit`), the
host path is already **49–54% of numpy's warm call** (11.32 ms of 21.06 at 64 trees, 48.07 of 93.67 at
200 trees) and **16–23% of iree's** (12.77 of 69.61, 53.72 of 229.49) — cost the GPU backend inherits
in full. Any redesign has to shrink this host path before a GPU win can appear, and the batching lever
(i) attacks exactly the part that multiplies it by K.

## 5. Compile economics (separate from the headline)

Cold first call (`first_call_ms` for evals, `cold_first_gen_ms` for training) and the compile side, in
`stages.reps[].totals` (`cold_builds`/`cold_build_ms` = the fresh-process compile of distinct
structures; `warm_misses`/`warm_build_miss_ms` = post-first-call misses). Guard cap = the top-level
`cold_builds.limit` (`--max-cold-builds`), never exceeded (`exceeded: false` in every JSON).

| config | distinct builds | cold_build_ms | cold first call | ms per build | warm_misses | warm_miss_ms | guard cap |
|---|---|---|---|---|---|---|---|
| iree_fixture64_n64 | 48 | 81 854.0 | **82.07 s** | 1 705.3 | 0 | 0.0 | 300 |
| iree_fixture64_n1024 | 48 | 74 369.1 | 74.58 s | 1 549.4 | 0 | 0.0 | 300 |
| iree_fixture64_n10000 | 48 | 16 423.7 | **16.64 s** | 342.2 | 0 | 0.0 | 300 |
| iree_fixture200_n64 | 155 | 105 201.1 | **105.65 s** | 678.7 | 0 | 0.0 | 300 |
| iree_fixture200_n1024 | 155 | 335 082.0 | **335.62 s** | 2 161.8 | 0 | 0.0 | 300 |
| iree_fixture200_n10000 | 155 | 86 089.4 | 86.66 s | 555.4 | 0 | 0.0 | 300 |
| iree_fresh16 | 15 | 13 950.8 | 14.04 s | 930.1 | 0 | 0.0 | 300 |
| iree_c7r (pop 50 / 3 gens) | 39 | 20 016.0 | **20.74 s** | 513.2 | 24 | 10 336.3 | 200 |
| iree_sr_a rep0 | 155 | 81 142.1 | **82.10 s** | 523.5 | 362 | 151 824.1 | 3200 |
| iree_sr_a rep1 (same seed, same process) | 0 | 0.0 | 0.32 s | — | 0 | 0.0 | 3200 |
| np_c5r2 (numpy) | 48 | **151.5** | 0.217 s | **3.2** | 0 | 0.0 | exempt |
| np_fixture200_n64 (numpy) | 155 | 882.3 | 1.009 s | 5.7 | 0 | 0.0 | exempt |
| np_sr_a rep0 (numpy) | 155 | 834.5 | 1.356 s | 5.4 | 362 | 1 164.2 | exempt |
| np_sr_a rep1 (numpy) | 0 | 0.0 | 0.109 s | — | 0 | 0.0 | exempt |

Sources: `new/results/r2/iree_*.json`, `np_*.json` (`stages.reps[*].totals`, `first_call_ms` /
`cold_first_gen_ms`); per-build ms = `cold_build_ms` / `cold_builds`.

Facts to carry forward:

* **The compile tax is the whole cost of a cold IREE run.** `iree_sr_a` rep0 spends **232.97 s of its
  235.17 s** generation loop in compile misses = **99.06%** (155 cold builds 81.14 s + 362 in-rep
  misses 151.82 s, 517 misses total, `stages.reps[0].totals`); its warm per-gen median is 10 733.32 ms,
  of which 10 631.82 ms is `build_miss`. The same workload on numpy spends 1 998.7 ms of 2 750.6 ms
  (72.7%) — trivial in absolute terms (0.8 s cold for 155 structures, 3.9 ms per miss).
* **IREE per-build cost is not a constant**: 342 → 2 162 ms per structure (a **6.3×** spread) for
  identical build counts within this round (`r2/RUNLOG.md` §Build-rate calibration; the same 48
  structures took 16.4 s in one config and 74.4 s in another). Any single cold number is ±a factor.
* **Warm ⇒ no compilation, provably**: `warm_misses == 0` and `warm_build_miss_ms == 0.0` on all six
  fixture configs and on `iree_fresh16` (`stages.reps[0].totals`), and `iree_sr_a` rep1 records 0 cold
  builds, 0 misses, 632 hits, 1.124 s of `total_ms` for the whole 10-generation rep (rep wall 1.168 s:
  the second rep of the same seed is essentially free).
* **The cache is per (structure, input shape, backend, device, options) and per process**
  (`src/evogp/tree/etl_engine.py` module docstring; `src/evogp/tree/CONTEXT.md`): each fixture config
  rebuilt its 48 / 155 structures from scratch even though the tree structures are byte-identical
  across the n64/n1024/n10000 configs, and a data-size change forces rebuilds; the training pair runs
  with `EVOGP_ETL_CACHE_SIZE=8192` to hold the 517 distinct SR structures.
* **IREE leaks file descriptors**: 32.34–35.47 fds per compiled structure
  (`(fds.after_cold − fds.start) / cold_builds`: 33.08 for fixture64, 32.34 for fixture200 and
  `iree_sr_a`, 35.47 for `iree_fresh16`), plus a warm-path leak of ~1.4–3.3 fds per build call; peaks
  are `fds.end` 2395 (fixture64 n64), 5819 (fixture200 n64) and 17 243 (`iree_sr_a`, 26% of the
  65 536 soft limit). Every iree log also ends with nanobind `leaked N instances` / keep-alive reports
  after the JSON is written (teardown noise; exit 0, no measured number affected).
* **Message**: compile is ignorable *only* when amortised — a repeated seed or a converged population
  hits the cache, a fresh population or a churning generation pays it in full. Its size is bounded by
  distinct structures × ~0.34–2.2 s (iree) vs × ~3–6 ms (numpy): the numpy backend's compile tax is
  never the story (0.15 s for 48 builds), iree's always is.

## 6. Same-session sanity vs round 1

Same box, same GPU 6, same fixtures/seed, same session as the round-1 artifacts used as anchors:

| quantity | round-1 anchor | R2 same-session re-run | delta | source |
|---|---|---|---|---|
| legacy C5 warm median (64-tree fixture) | 0.6042 ms | **0.6327 ms** | **+4.7%** | `legacy/results/C5.json` vs `legacy/results/r2/leg_c5r2.json` |
| legacy C5b warm median (1000 fresh trees) | 0.6022 ms | 0.6424 ms | +6.7% | `legacy/results/C5b.json` vs `legacy/results/r2/leg_c5b_r2.json` |
| numpy C5 warm median | 20.787 ms | **21.056 ms** | **+1.3%** | `new/results/C5.json` vs `new/results/r2/np_c5r2.json` |
| numpy C1 warm per-gen median (pop 200) | 88.07 ms (rep0) | **89.21 / 88.52 ms** (rep0/rep1) | **+1.3% / +0.5%** | `new/results/C1.json` vs `new/results/r2/np_c1r2.json` |
| legacy C5 fitness anchor | nan 3/64, mean −18.048504493275626 | identical | **exact** | `legacy/results/r2/C5.json` |
| legacy C5b fitness anchor | nan 66/1000, mean −59.737519580878846 | identical | **exact** | `legacy/results/r2/C5b.json` |
| iree C7 generation loop (pop 50 / 3 gens) | 60.44 s | **31.20 s** | −48% | `new/results/C7.json` vs `new/results/r2/iree_c7r.json` |
| iree C7 `per_gen_best_fitness` | −0.2510969936847687 ×3 | bit-identical (×3) | **exact** | same pair |
| cross-backend trajectory (pop 200 / 10 gens) | — | `iree_sr_a` and `np_sr_a`, rep0 **and** rep1, all 10 values bit-identical −0.2510969936847687 | **exact** | `new/results/r2/{iree,np}_sr_a.json` |

Reading: the same-session deltas (+0.5% to +6.7%) are inside the ±10% shared-box band, and the
fitness/trajectory anchors are bit-identical, so the R2 numbers and the round-1 numbers describe the
same engines on comparable boxes; the iree C7 speed-up is box/compile-window variance, not a different
trajectory (fitness is identical). All R2 comparisons in §1–§5 should be read at ±10% for the warm
numbers and ±a factor for the compile numbers (§5).

## 7. Bounded failures & ceilings

* **`iree_sr_b2` (pop 1000 / tree 128 / 2 gens, `EVOGP_ETL_CACHE_SIZE=8192`, cap 1500 misses): no
  measurement — exit 139 SIGSEGV.** `new/results/r2/iree_sr_b2.json` is a skeleton
  (`complete: false`, `per_gen_ms: []`, `per_gen_best_fitness: []`, `stages.reps[0].totals: null`,
  `fds.after_cold: null`, `fds.end: null`) and `iree_sr_b2.log` holds exactly three Python-side
  startup lines (cold-build guard banner, stage-probe banner, harness banner) followed by
  `exit code: 139 (FAIL)` — no Python traceback, no stderr, no per-build logging. Process life ≈ **390 s**
  (15:20:39 → 15:27:09), guard still `misses_max: 0, exceeded: false` → the crash happened **before
  generation 0 completed**. At the round's measured compile rates (450–524 ms/build, `iree_sr_a`) 390 s
  bounds the work at ~800–900 builds, i.e. inside the frozen cap and far from it (`r2/RUNLOG.md`
  §`iree_sr_b2`).
* **It is not EMFILE.** `nofile_rlimit = [65536, 65536]` in that JSON, and round 1's real EMFILE surfaced
  as a clean `OSError: [Errno 24] Too many open files` from the IREE compile path
  (`new/NOTES.md` §9.2) — absent here. No autopsy is possible from the record (no core, `dmesg`
  unreadable, no per-build logging). Not retried and not patched, deliberately.
* **The FD leak is an operational ceiling on its own.** `np_sr_b` (the numpy analog of that workload,
  pop 1000 / tree 128) needs 4 760 (rep0) + 4 687 (rep1) = **9 447 misses**
  (`np_sr_b.json` `stages.reps[*].totals.total_build_misses`); at the measured 32.34 fds per compiled
  structure that is **≈ 306 000 fds ≫ 65 536**, so that workload cannot complete in one process at the
  frozen settings whatever the build budget — the only bounded outcome is a guard trip
  (`--max-cold-builds 1500`), not a measurement. `iree_sr_a` itself peaked at 17 243 fds (26% of the
  limit).
* **No OOM anywhere.** The `--mem-guard-mb 8192` guard never triggered: every R2 JSON has
  `skipped: false`/`skip_reason: null`. Peak torch allocation on the training pair is 2.51 MiB
  (`iree_sr_a.json` 2 630 144 B), the largest peak in the whole round is 23.14 MiB
  (`iree_fixture200_n10000.json` / `np_fixture200_n10000.json`, 24 265 728 B — the 10000-row `x`/label
  tensors), and device-wide `nvidia_smi_used_mb_end` peaks at 501 MiB on the new side (legacy engine
  contexts: 1197–1715 MiB) against `free_mb_start ≈ 48 409 MiB`.
* **No timeouts, interrupts, guard trips or missing records.** All other 34 configs are
  `complete: true`; no 124/137/130 exit code; every `cold_builds.exceeded = false`; `check_r2.py`
  reports **0 missing and 0 malformed** manifest records (§10). The single SKIP note in the scan exists
  in the smoke files (`smoke/h1_guard.json`, `cold-builds-exceeded:4>1`) and is not an R2 config.

## 8. Ranked optimization levers (for the engine-redesign workstream)

Impact-ranked; each lever states its evidence and expected direction.

1. **Collapse the K per-tree invocations into a few batched launches (biggest lever).** Evidence: K = 53
   (64 trees) / 171 (200 trees) invocations + readbacks per `evaluate`, per-invoke cost 1.31–1.52 ms
   (iree) / 0.40–0.62 ms (numpy) vs the 0.131–0.162 ms ETL floor, and per-invoke overhead is ~10×
   (iree) / ~3–5× (numpy) that floor (§2, §4(ii)); the legacy engine's whole 200-tree eval is 0.632 ms
   in one launch. Direction: collapsing K per-tree invocations into ~1 batched call removes the
   per-tree `build_hit` + `readback` share — 10.29 ms of numpy's 21.06 ms (49%) and 31.31 ms of iree's
   69.61 ms (45%) at 64 trees — i.e. ≈1.8–2× warm headroom before the arithmetic itself is touched, and
   it shrinks the host lookup path that is per-tree today.
2. **Remove the serialized per-tree D2H readbacks (single batched readback).** Evidence: `readback` is
   26.2% of the warm `sr_a` rep1 generation (17.12 of 65.31 ms), 29.0% of `iree_fixture64_n64`
   (20.22 of 69.61 ms), 28.5% of `iree_fixture200_n64` (65.31 of 229.49 ms) and only 3.0% on numpy
   (0.63 of 21.06 ms, `d2h_calls = 0`) — `d2h_calls` = K per call. Direction: one copy of the assembled
   output vector instead of K per-tree copies removes ~a quarter of iree's warm call and most of the
   iree-vs-numpy gap.
3. **Reduce per-call Python/ETL glue (cache-key re-derivation + static re-validation).** Evidence: the
   `build_hit` stage alone is 9.66 ms of numpy's 21.06 ms (45.9%) and 46.36 ms of iree's 229.49 ms
   (20.2%); decode adds 1.66 ms (64 trees) / 7.28 ms (200 trees) on both backends; the engine's own
   documentation states `build_cached` re-encodes the program and re-derives its sha256 cache key on
   every call and `etl.run` re-validates statics in O(#tokens) per call
   (`src/evogp/tree/CONTEXT.md` §Engine). Direction: hoist the cache key/sha out of the hot path, avoid
   re-encoding on hits, and fold decode into the population update — the lever that helps **both**
   backends and does not depend on GPU availability.
4. **For these problem sizes, stop defaulting to GPU iree.** Evidence: iree is 2.45–3.43× slower than
   numpy-CPU warm on every fixture config (§1), 2.96× on the fully warm `sr_a` generation, 2.72× on
   `iree_fresh16`, and far slower cold — 16.6–82.1 s vs 0.15 s for 48 builds (110–542×) and
   86.7–335.6 s vs 0.9 s for 155 builds (98–380×); the engine's own instrument probe reaches the same
   conclusion (`src/evogp/tests/perf_probe.py`, recommendation block). Direction: keep iree opt-in for
   large-row cases only, and re-evaluate after levers 1–2 remove dispatch/sync overhead; nothing in
   this round shows a data size at which iree wins warm (10000 rows is still 3.43× behind at 64 trees).
5. **Compile side: keep memoisation, make it cross-process/persistent or async, and bound builds by
   FDs.** Evidence: §5 — per-build cost is 0.34–2.16 s with a 6.3× same-box spread, cold compile is
   99.06% of `iree_sr_a` rep0's generation loop, the cache is per-process (each config pays its 48/155
   builds again), and the leak is 32–35 fds per compiled structure against a 65 536 limit. Direction:
   a persistent/two-tier cache or background compilation would convert the cold path into the warm
   path; batching (lever 1) may also reduce the distinct-graph count, and the build guard must remain
   FD-aware, not only miss-count-aware.
6. **Fix or bound the pop-1000 scale before claiming GPU readiness.** Evidence: `iree_sr_b2` crashes
   (exit 139, §7) and its numpy mirror needs ~9.4 k misses ≈ 306 k fds, i.e. cannot run in one
   65 536-fd process at the frozen settings. Direction: a scale story for pop 1000 / tree 128 is a
   precondition for any IREE-based claim; today the only bounded outcome there is a guard trip.
7. **Treat every cold number as ±a factor.** The 6.3× per-build spread (§5) means a single
   `first_call_ms` is not a hardware constant; report cold numbers only alongside their build count and
   `ms/build`, and prefer back-to-back A/B windows when comparing a redesign.

## 9. What supersedes what

`SUMMARY.md` remains the round-1 record for the **numpy side** (34×–7 670×, C0–C7 matrix). This
REPORT.md supersedes it as the current GPU verdict: R2 adds the IREE-cuda backend on the same
workloads and fixtures, warm stage decomposition (decode / build-hit / run / readback / upload / loss)
with per-call invocation counts, separation of compile from steady-state numbers, data-row scaling, and
the backend-agreement gate. Where the two disagree numerically, the R2 same-session re-runs (§6) show
the round-1 numbers reproduce within +0.5% to +6.7% on this box, so the round-1 methodology is
confirmed rather than replaced; no file in `SUMMARY.md` is edited.

## 10. Artifacts & reproduction

Committed artifacts of the round (all paths relative to the repository root):

| artifact | what it holds |
|---|---|
| `REPORT.md` (this file) | the R2 verdict and the analysis above |
| `r2/RUNLOG.md` | operator log: exact commands per chunk, per-config status tables, box/GPU state, consistency checks, deviations |
| `r2/backend_agreement.md` | the pre-flight gate: iree-O1 vs numpy bit-identity on both fixtures, anchors, the 4 exact commands |
| `r2/README.md` | runner/validator/usage/exit-code contract of the round |
| `r2/run_all.sh` | the matrix runner (one process per config, `flock`, `ulimit -n 65536`, `timeout` inside the lock) |
| `r2/manifest.txt` | the 35 expected config ids with side, JSON path and stage-probe expectation |
| `r2/check_r2.py` | stdlib-only validator (manifest + directory scan modes) |
| `r2/check_backend_agreement.py` | the backend-agreement driver (public `evogp` API only) |
| `new/results/r2/*.json`, `legacy/results/r2/leg_*.json` | all 36 config records (raw numbers of §1–§7) with their `.log` files (exact command + GPU snapshot per config) |
| `new/results/r2/backend_agreement/{fixture64,fixture200}_{numpy,iree}{,_values}.json/.npy` | pre-flight records + the raw float64 fitness vectors (force-added past `.gitignore`) |
| `new/results/r2/smoke/h1_*.json`, `h2_*.json` (+ `.log`) | harness/probe smoke + FD calibration + A/B controls (`new/NOTES.md` §R2) |
| `fixtures/forest200_tree64_random.npz` + `fixtures/README.md` | the 200-tree fixture and its provenance (sha256, generator, schema) |
| `bench_gp.py`, `stage_probe.py`, `make_r2_fixtures.py` | side-agnostic harness, runtime stage probe, fixture generator |
| round-1 anchors used in §6 | `legacy/results/{C5,C5b,C1_r1}.json`, `new/results/{C5,C1,C7}.json` |

Reproduce (repository root, project venv `/mnt/local-ssd/bchuang/evox/.venv/bin/python`, GPU 6, one
`flock` per heavy phase — the runner adds it; full command list and per-chunk order in
`r2/RUNLOG.md`):

```bash
# 1. light group: 11 legacy + 12 numpy-mirror configs (minutes); --dry-run first, it prints every exact
#    command (per-config PYTHONPATH/flags/env) and executes nothing
bash test/benchmark/r2/run_all.sh --group light --dry-run
bash test/benchmark/r2/run_all.sh --group light
# 2. iree group: 12 configs, tens of minutes (compile-bound); --only for one config, --timeout to raise
#    the per-config cap (the runner flocks internally, one level, timeout inside the lock)
bash test/benchmark/r2/run_all.sh --group iree
bash test/benchmark/r2/run_all.sh --only iree_sr_a --timeout 3000
bash test/benchmark/r2/run_all.sh --only iree_sr_b2 --timeout 3600      # known SIGSEGV (exit 139)
# 3. backend-agreement pre-flight: 4 processes (fixture × backend), any order, one process each
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
  ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=numpy \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r2/check_backend_agreement.py \
  --fixture test/goldens/forest_random.npz \
  --json-out test/benchmark/new/results/r2/backend_agreement/fixture64_numpy.json
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
  ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=numpy \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r2/check_backend_agreement.py \
  --fixture test/benchmark/fixtures/forest200_tree64_random.npz \
  --json-out test/benchmark/new/results/r2/backend_agreement/fixture200_numpy.json
#   ... and the same two commands with `EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 \
#       EVOGP_ETL_IREE_OPT_LEVEL=O1` writing fixture64_iree.json / fixture200_iree.json
#   (all four verbatim in r2/backend_agreement.md §"Exact commands run")
python3 - <<'PY'
import numpy as np
a = np.load("test/benchmark/new/results/r2/backend_agreement/fixture64_numpy_values.npy")
b = np.load("test/benchmark/new/results/r2/backend_agreement/fixture64_iree_values.npy")
assert np.array_equal(np.isnan(a), np.isnan(b))
assert np.allclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
PY
```

Validator results, verbatim (stdlib only, no GPU; run read-only on the committed JSONs):

```bash
python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt      # exit 0
#   35 record(s): 34 complete, 1 partial, 0 skipped, 0 missing, 0 malformed (0 not in the manifest)
#   126 check(s), 0 failure(s)
#   notes (not failures):
#     iree_sr_b2: PARTIAL (complete=False, partial=False)
#     iree_sr_b2: stage-counter extraction incomplete (cold=?, misses=?)
python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2 test/benchmark/legacy/results/r2   # exit 1
#   50 record(s): 46 complete, 3 partial, 1 skipped, 0 missing, 0 malformed
#   123 check(s), 2 failure(s)
#   FAIL C5:  missing required eval keys ['task','config','complete'] (legacy/results/r2/C5.json)
#   FAIL C5b: same (legacy/results/r2/C5b.json)
#   notes (not failures): iree_sr_b2 PARTIAL; h1_guard SKIP (cold-builds-exceeded:4>1), a smoke artifact;
#     C5/C5b PARTIAL (complete=None) and leg_smoke tag 'LEG-smoke' -- prior-round driver artifacts
```

The two scan failures are the same pre-existing round-1 driver artifacts (`C5.json`, `C5b.json`) that
every RUNLOG entry reports; zero failures are attributable to the R2 configs, and 35/35 manifest ids
have a JSON on disk. Note `r2/last_run.txt` is generated by `run_all.sh` for the last invocation only
and is deliberately not committed (`r2/README.md`).
