# Legacy CUDA engine vs ETL "true jit" engine — headline results

Frozen methodology, workloads W1–W3 and the config matrix C0–C7: **`SPEC.md`**.
Raw data: `legacy/results/*.json` (**OLD** = pre-rewrite commit `b613503`, hand-written CUDA kernels)
and `new/results/*.json` (**NEW** = post-rewrite engine, ETL "true jit", default `"numpy"` backend).
Per-side narrative: `legacy/NOTES.md`, `new/NOTES.md`.

**Every number in this file was recomputed from those JSONs by the consolidation pass** (none copied
from the side NOTES files), and `legacy/results/c4_rerun/` is excluded from every headline table
(variance evidence only, cited in §4).

## 1. Verdict

On the same shared box (RTX A6000, py3.11 / torch 2.6.0+cu124) and driven through the identical
public API, the legacy compiled-CUDA engine is **~34× to ~7,670× faster per generation** than the new
ETL engine in its default `"numpy"` backend — warm-median generation times, both sides, same config:

| where the gap sits | per-generation cost, old → new | ratio (new/old) |
|---|---|---|
| C5 fixed-forest eval (64 trees / 63 nodes, warm `evaluate`) | 0.604 ms → 20.8 ms | **34.4×** |
| C1 — the main W1 run (pop 200 / 100 gens / tree 64) | 1.633 ms → 88.1 ms | **53.9×** |
| C4 — XOR demo (pop 50 / 100 gens / tree 64, rep 0) | 1.756 ms → 405.0 ms | **230.7×** |
| C2b — pop 1000 / 20 gens / tree 128 | 1.607 ms → 665.0 ms | **413.9×** |
| C5b — 1000 freshly generated trees, eval-only | 0.602 ms → 511.1 ms | **848.8×** |
| C2 — pop 1000 / 100 gens / tree 128 (bloat-limited, stopped at 89 gens) | 3.451 ms → 26,466 ms | **7,669×** |

What the new engine buys for that factor: **no build step at all** (plain `import evogp`, no CUDA
toolkit, no `nvcc`, no extension module — the legacy side must compile `src/evogp/cuda/*.cu` before it
can import anything), no hand-written kernels to maintain, device-agnostic evaluation code, and a
pure-Python package with nothing to load, where the legacy side pays a per-process CUDA module/context
load on its first timed step (~0.5–0.6 s) and the new side pays 0.75–5.7 s of JIT tracing/building for
gen 0. It is also the only engine still in the repository: the legacy path is a materialized `b613503`
checkout built for this comparison, not code that ships today. The gap is a deliberate
speed-for-portability trade, and the absolute numbers are small enough in most configs (tens of
milliseconds to ~0.7 s per generation) that the trade is only decisive at C2 scale.

## 2. Headline table

One row per side per config. Times in **ms** unless marked. Conventions (used for *both* sides):

* **warm median** = `median(per_gen_ms[1:])` for training / `median(per_call_ms[1:])` for evals — i.e.
  the cold first step is excluded on both sides, so the ratio column is like-for-like.
* **cold** = `per_gen_ms[0]` / `first_call_ms` (per process; includes the new side's per-structure ETL
  build and the legacy CUDA module/context load).
* **ratio** = warm median, **new/old** ("× slower" for the new engine). Where a config has reps, the
  row shows `rep0 / rep1` and the ratio uses rep 0 unless stated.
* `wall_s` is the whole-rep wall clock including construction and the cold step (rep 0 / rep 1 where
  two reps exist).

| config | workload & params | side | wall_s | warm median | cold first | best fitness (or eval stats) | × slower (new/old) | source JSON |
|---|---|---|---|---|---|---|---|---|
| **C1** | W1 SR, pop 200 / gens 100 / tree 64 | OLD | 1.516 / 0.172 | **1.633 / 1.617** | 532.1 / 1.6 | −0.086839 | — | `legacy/results/C1_r1.json`, `C1_r2.json` |
| C1 | same, same seed per rep | NEW | 24.90 / 24.59 | **88.07 / 100.23** | 1480.2 / 1780.0 | −0.171954 | **53.9×** | `new/results/C1.json` |
| **C2** | W1 SR, pop 1000 / gens 100 / tree 128 | OLD | 1.713 | **3.451** | 609.1 | −0.225680 | — | `legacy/results/C2.json` |
| C2 | same (soft cap: 89/100 gens, `partial`) | NEW | 1820.29 | **26,466.1** | 5681.1 | −0.036229 | **7,669×** | `new/results/C2.json` |
| **C2b** | W1 SR, pop 1000 / gens 20 / tree 128 | OLD | 1.355 | **1.607** | 520.3 | −0.225680 | — | `legacy/results/C2b.json` |
| C2b | same | NEW | 24.454 | **665.03** | 5703.9 | −0.205358 | **413.9×** | `new/results/C2b.json` |
| **C4** | W2 XOR, pop 50 / gens 100 / tree 64 | OLD | 0.886 / 0.184 | **1.756 / 1.769** | 519.5 / 1.8 | −0.250000 | — | `legacy/results/C4_r1.json`, `C4_r2.json` |
| C4 | same, same seed per rep | NEW | 40.62 / 8.04 | **404.95 / 75.07** | 754.7 / 48.8 | −0.250000 | **230.7×** (rep 0) · 42.7× (rep 1) | `new/results/C4.json` |
| **C5** | W3 eval-only, fixed 64-tree fixture, 25 evals | OLD | 0.168 | **0.604** | 21.77 | `nan_count` 3/64, `fitness_mean` −18.048504 | — | `legacy/results/C5.json` |
| C5 | same fixture, 25 evals | NEW | 0.750 | **20.787** | 229.2 | not recorded (new eval schema) | **34.4×** | `new/results/C5.json` |
| **C5b** | W3′ fresh pop 1000 / tree 64, 10 evals | OLD | 0.190 | **0.602** | 20.88 | `nan_count` 66/1000, `fitness_mean` −59.737520 | — | `legacy/results/C5b.json` |
| C5b | same fresh forest, 10 evals | NEW | 10.477 | **511.14** | 5622.2 | — | **848.8×** | `new/results/C5b.json` |
| **C0** | W1 SR, pop 200 / gens 100 / tree 64, `ETL_DISABLE_LOCATIONS=0` | NEW only (control) | 93.452 | **336.06** | 4650.4 | −0.171954 | **3.82× vs C1 rep 0** (same run with locations ON) | `new/results/C0.json` |
| **C7** | W1 SR, pop 50 / gens 3 / tree 32, `EVOGP_ETL_BACKEND=iree` (`O1`) | NEW only (opt-in backend) | 61.20 | 19,304.6 (per-gen 21,831 · 12,592 · 26,017) | 21,830.8 | −0.251097 | n/a — optional backend, explicitly non-primary | `new/results/C7.json` |

Per-config notes:

* **C1** — the main comparison point. The legacy warm profile is flat ~1.6 ms/gen; the new side is
  69–132 ms/gen (decade means of `per_gen_ms`) over most of the run, but its last decade explodes
  (gens 90–99: 572–1959 ms, mean 1191 ms vs a 114 ms mean over gens 10–89, i.e. ~10×), which is why
  the new mean (239.7 ms) is 2.7× its median. Both new reps replay the same seed and end at the same
  fitness; the reps differ by a median of 13.1 ms/gen (p90 232.5 ms) — shared-box noise, not cache
  reuse (§4).
* **C2** — `partial: true`, 89/100 generations in 1820.3 s (SPEC's `--max-wall-s 1800 --min-gens 20`
  soft cap, exit 0), mean 20.44 s/gen, JSON projection 2044.3 s (~34 min) for all 100. The **7,669×**
  figure is a warm-median ratio inflated by tree bloat: gens 0–19 average 1.0–1.3 s, gens 80–88 average
  33.1 s/gen, so the mean-based projection is optimistic (my own arithmetic from the tail: ≈36.4 min).
  The legacy side's C2 profile also drifts (1.6 → 3.5 ms), but by 4 orders of magnitude less.
* **C2b** — same workload as C2 truncated at 20 generations. Here the all-generation median (803.9 ms)
  differs materially from the warm median (665.0 ms) because the 5704 ms cold generation sits inside
  the 20-element median; the ratio column uses **665.0 ms**. All other ≥20-generation configs differ
  by <2% between the two conventions (C1 88.27 vs 88.07; C4 405.12 vs 404.95; C0 336.68 vs 336.06;
  C2 26,230 vs 26,466). C7 (3 generations) is the other exception.
* **C4** — tiny XOR demo; both sides plateau at exactly −0.250000 (MSE 0.25: the 50-tree / 100-gen
  budget does not fit the 8-row parity table exactly). The new side's rep 1 is 5.4× faster than rep 0
  (warm median 75.07 vs 404.95 ms; walls 8.04 vs 40.62 s) — the whole run fits the engine's 4096-entry
  compile cache — so the cross-side ratio depends on rep: 230.7× (rep 0) vs 42.7× (rep 1).
* **C5 / C5b** — RNG-independent eval microbenches, so the *only* unambiguous cross-side numbers: the
  new engine's cold call is 10.5× (C5) / 269× (C5b) the legacy one, and its warm call 34.4× / 848.8×.
  The new-side trimmed eval schema records no `nan_count`/`fitness_mean`; the old side does, and C5 on
  the old side is the correctness anchor (§5).
* **C0** — `ETL_DISABLE_LOCATIONS` control, new side only: identical workload/seed as C1, only the
  ETL call-site capture differs. Wall 3.75×, generation loop 3.86×, warm median 3.82×, cold first gen
  3.14×; the best-fitness series is bit-identical to C1 rep 0 (verified key-by-key from the JSONs), so
  this variable is purely a build-time knob — the README's "~4×" holds on a real training run.
* **C7** — optional, clearly non-primary: the opt-in IREE backend at a deliberately tiny scale
  (pop 50 / 3 gens / tree 32) costs 61.2 s wall, 12.6–26.0 s per generation, i.e. ~0.44–0.52 s of
  MLIR→vmfb compile per new structure. Generation 1 is not cheaper than generation 0 (survivors are
  rebuilt), and the IREE path leaks runtime objects (see `new/NOTES.md` §9), which is why it is not
  usable at GP scale and why it is excluded from the primary comparison.

## 3. Why the numbers look like this (brief, factual)

* **OLD**: hand-written CUDA kernels over the whole population, one dispatch sequence per generation /
  `evaluate` call. The warm cost is dominated by that fixed per-step overhead rather than by population
  size: pop 200 (C1) and pop 1000 (C2b) share the *same* ~1.6 ms warm median, and a single additional
  `evaluate` call on 64 fixed trees costs 0.604 ms (C5). Only tree size moves it — C2's 128-node trees
  give a 3.451 ms warm median, a 2× effect.
* **NEW**: the engine decodes the population host-side and specializes every distinct tree structure
  into an `etl.build_cached` graph (`src/evogp/tree/etl_engine.py`), then executes it on the default
  numpy backend. Cost is therefore dominated by **compilation of structures that are new in this
  process**; a fresh population is a cold miss per distinct structure (C5b: 1000 fresh trees → 5.6 ms
  each to build vs 0.51 ms each to merely run). Compilation is memoized in a single in-memory
  4096-entry `etl.CompileCache` (`EVOGP_ETL_CACHE_SIZE`), which only pays off for elites/survivors
  (C4 rep 1) or duplicate structures.
* **Bloat without parsimony pressure** is the third factor: W1's selection has no size penalty, so
  trees grow toward `max_tree_len` and per-structure cost grows with them — C2's per-structure build
  reaches ~33 ms at full 128-node trees, and C1's last ten generations are ~10× the mid-run steady
  state. Config matrix, workloads and fairness rules: `SPEC.md`.

## 4. Caveats

* **This is a time-only comparison.** Both sides run the same public API and the same workloads, but
  NEW draws its population from torch's RNG while OLD uses the legacy kernel RNG stream — the streams
  are *distribution-equivalent, not identical* (SPEC, `new/NOTES.md` §2). Fitness values are
  per-side convergence sanity checks, **not** a quality comparison; e.g. OLD's C1 ends at −0.086839 and
  NEW's at −0.171954 on different seeded explorations of the same task.
* **C2 is partial by construction** (89/100 generations, SPEC soft cap) and its JSON projection
  (`mean × 100`) is approximate: the most expensive generations are the missing ones, so treat
  2044 s as a lower-bound-ish estimate (tail-based arithmetic suggests ~36 min; see §2).
* **Shared box.** All runs share the A6000s with other tenants; treat medians as ±10%, not exact.
  Evidence: the SPEC-authorized C4 re-run (`legacy/results/c4_rerun/`, **excluded from the headline
  table**) measured the same config at 1.28 s / 0.78 s wall with warm medians 3.91 / 5.35 ms against
  the canonical 1.756 / 1.769 ms — i.e. a ~2–3× contention inflation in a busy window, the same window
  that also shows up in the old side's `show_details` diagnostic. Canonical C4 = `C4_r1.json` +
  `C4_r2.json`.
* **The legacy C4_r1 warm profile contains exactly one blip** (gen 66: 9.386 ms = 5.35× its 1.756 ms
  median, the only generation above 5×); `C4_r2` (max 2.051 ms) and both canonical medians agree to
  0.75%, so it is run variance, not part of the config's profile.
* **GPU 7 was excluded** from every run: hardware-faulted (uncorrectable ECC, DRAM errors pending row
  remap) — any CUDA context creation fails with `uncorrectable ECC error`. Evidence:
  `legacy/results/logs/gpu7_ecc_evidence.txt`. GPU indices actually used: OLD 4; NEW 4 (C1/C2b/C4),
  0 (C2/C0), 5 (C5), 1 (C5b/C7), recorded in each JSON's `env.CUDA_VISIBLE_DEVICES`.
* **C0 is a knob study, not a second engine**: `ETL_DISABLE_LOCATIONS=0` costs 3.75× wall for a
  bit-identical fitness trajectory, which is why `=1` is the recommended setting in every other run.
* **Cross-rep cache warmth is scale-dependent**: the identical-seed C4 rep 1 is ~5.1× faster by wall
  (5.4× by median) because that run's distinct-structure count fits the 4096-entry cache; C1's reps are
  unaffected (working set ≫ 4096, rep 1's gen 0 is even larger than rep 0's). Any second rep therefore
  reports *warm-cache* numbers.
* **OLD is a built artifact, not a checkout you can import**: `b613503` materialized and built with
  `legacy/setup_legacy.sh` under the pinned nvcc 12.6 / `TORCH_CUDA_ARCH_LIST=8.6` recipe and the
  offline assets in `/tmp/evogp_t1a263` (see `legacy/BUILD_NOTES.md`); the `.so` was byte-stable across
  the run (sha256 recorded there). The rebuild/reproduce path does not exist for the current tree.

## 5. Correctness anchor and validation

**Golden anchor (old side, C5).** C5 evaluates the frozen pre-ETL fixture forest
(`test/goldens/forest_random.npz`) on that file's own `x`/`labels`, so it is RNG-independent and
directly comparable to the frozen capture `test/goldens/sr_fitness.npz`:

| quantity | C5 run record (old side) | recomputed from the frozen golden | verdict |
|---|---|---|---|
| `nan_count` (of 64) | 3 | 3 | **PASS** (exact) |
| `fitness_mean` (NaN-aware) | −18.048504493275626 | −18.048504493275626 | **PASS** (bit-identical here; tolerance rtol=atol=1e-4) |

So the harness wires the fixture `Forest`, its datapoints and `SymbolicRegression` exactly as
`SPEC.md` W3 intends, and both engines share the same fitness convention.

**Validators, re-run on the committed JSONs during this consolidation pass** (JSON-only, no GPU, no
raw data modified):

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
PYTHONPATH=src $PY test/benchmark/new/check_smoke.py test/benchmark/new/results/*.json
#   -> 8 JSON file(s) (0 truncated by a hard kill), 160 checks, 0 failure(s)   [exit 0]
$PY test/benchmark/legacy/check_results.py test/benchmark/legacy/results/*.json --gpu 4
#   -> PASS — 8 record(s) checked, all SPEC consistency checks held           [exit 0]
$PY test/benchmark/legacy/check_results.py test/benchmark/legacy/results/C5.json --gpu 4 \
      --golden test/goldens/sr_fitness.npz
#   -> C5: GOLDEN nan_count PASS (3 vs golden 3)
#   -> C5: GOLDEN fitness_mean PASS (−18.048504493275626 vs −18.048504493275626)
#   -> PASS — 1 record(s) checked                                              [exit 0]
```

The new-side checker covers schema completeness, `side=="new"`, step counts vs `config`, rep-0
mirroring, `partial`/`projection`/`complete` consistency, and `median_warm_ms` /`generation_loop_s`
recomputation; the old-side checker covers the trimmed SPEC key set, `len(per_gen_ms)==gens`, monotone
`per_gen_best_fitness`, `cold_first_gen_ms == per_gen_ms[0]`, `first_call_ms == per_call_ms[0]`,
`median_warm_ms == median(per_call_ms[1:])`, `side=="old"` and `CUDA_VISIBLE_DEVICES == 4`.

### 5.1 Cross-validation — old side through the unified harness

The two headline OLD-side configs (**C1**, **C5**) were re-run through the *same* side-agnostic harness
`bench_gp.py` the NEW side uses, instead of the original matrix's old-side driver
`legacy/run_matrix.py`, to show the original records are harness-independent. Same flock/GPU discipline
(exactly one `flock -w 7200 /tmp/gpbench_critical.lock`, never nested; GPU 0 picked by least
utilization / least memory, GPU 7 excluded) and `--reps 1`, so C1 is compared against the cold
`C1_r1` record. Data, commands and the full delta tables: `xval/` (`xval/NOTES.md`). **C1**: `wall_s`
1.5154 s vs 1.5156 s (**−0.01%**), cold first generation 559.2 ms vs 532.1 ms (**+5.1%**), warm median
(`median(per_gen_ms[1:])`) 1.727 ms vs 1.633 ms (**+5.8%**), `best_fitness_final` identical
(−0.086839). **C5**: `first_call_ms` 22.17 ms vs 21.77 ms (**+1.9%**), warm median
(`median(per_call_ms[1:])`) 0.628 ms vs 0.604 ms (**+3.9%**), `nan_count` 3/64 and nan-aware
`fitness_mean` −18.048504493275626 identical. C5's `wall_s` is **not** comparable across the two
drivers (0.057 s vs 0.168 s): `bench_gp.py` warms the device before its wall clock starts, while
`run_matrix.py`'s window contains ~0.15 s of one-time CUDA context creation — compare the per-call
metrics there instead (C1's `wall_s`, where the 100-generation loop dominates, agrees to −0.01%).
**Verdict:** the original numbers are harness-independent within shared-box noise (a few percent and
bidirectional — an auxiliary C5 run landed *below* the original); no harness change was needed.

## 6. Reproducibility

**Environment.** One shared box with 8× RTX A6000 (49 GB each); py3.11.2 / torch 2.6.0+cu124
(`torch_cuda_version` 12.4 in the JSONs; driver 535.247.01 per the recorded `nvidia-smi`) / ETL 0.1.0,
project venv `/mnt/local-ssd/bchuang/evox/.venv/bin/python`. NEW = this repository at the
harness snapshot recorded in each JSON's `evogp_commit` (`9159e10` for the matrix runs, `7e6cd0f` for
the Phase-1 C5); OLD = `b613503` materialized at `/tmp/evogp_legacy_bench` and built there. Seeds:
`torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` before construction, per rep (SPEC); the legacy
side therefore explores different structures than the new side.

**Commands.**

```bash
# NEW side (from the repository root; runner flocks internally, one level)
./test/benchmark/new/run_matrix.sh              # all configs, SPEC priority order, or a subset
# OLD side: materialize + build once (recipe + offline assets: legacy/BUILD_NOTES.md), then run the
# matrix from the legacy checkout root with the legacy PYTHONPATH order
./test/benchmark/legacy/setup_legacy.sh
cd /tmp/evogp_legacy_bench && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/tmp/evogp_t1a263/siteutils \
  $PY <repo>/test/benchmark/legacy/run_matrix.py --configs C1,C2,C2b,C4,C5,C5b \
  --out <repo>/test/benchmark/legacy/results --commit b613503
```

Both sides drive the side-agnostic harness `bench_gp.py` through the same public API; exact per-config
invocations, env vars and timings are in `new/NOTES.md` §§4, 7 and `legacy/NOTES.md` §§5–6.

**Serialization & scheduling.** Every heavy phase (legacy build, every timed run) is wrapped in exactly
one `flock -w 7200 /tmp/gpbench_critical.lock` shared by both sides; lock waits are outside every timed
region, and the lock is never nested. GPU choice: least utilization, then least memory, at run time,
with GPU 7 excluded for its ECC fault; `CUDA_VISIBLE_DEVICES` is recorded in every JSON. New-side runs
were launched as detached `systemd-run --user` units (CPUQuota 800%, MemoryMax 12G, LimitNOFILE 65536)
so a batch survives the agent shell — see `new/NOTES.md` §9 for the two deviations this caused
(C7 needed a retry with the raised FD limit; batch C's log *headers* predate the real run by the lock
wait, while all measured regions are unaffected).
