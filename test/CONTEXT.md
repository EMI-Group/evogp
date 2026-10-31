# `test/` — hand-run validation tooling around the canonical pytest suite

## Intent
Ad-hoc, human-driven validation layer for the repo: one small public-API regression script, the frozen
**legacy-CUDA numeric reference** (`goldens/`) plus the script that captured and compares it, an `evoxc`
economics probe, the performance-benchmark harnesses under `benchmark/` (current: `benchmark/evoxc/`;
historical: the legacy-engine rounds), and three scratch notebooks.
Nothing here is imported by `evogp`, by the examples or by the pytest suite, and there is no CI (no pytest ini in the repo): every file is executed by hand from the repository root, except the harness's two torch-free pytest modules (`benchmark/evoxc/test_arm_registry.py`, `benchmark/evoxc/test_large_cells.py`), which the full gate `pytest test src/evogp/tests src/evogp/gp/tests` does collect.
The real regression gates are the library's two pytest suites, `../src/evogp/tests/` (independent plain-torch oracle,
randomized/adversarial equivalence, operator invariants, API compatibility, convergence smoke) and the torch-free
functional suite `../src/evogp/gp/tests/`: treat
this directory as the layer *around* them, i.e. legacy cross-checks, an `evoxc` tooling probe, the
benchmark harnesses and human-watched demos.
Because there is no test framework of its own here, a green run only means "it ran and its checks held"; only
`fix_bug.py` (explicit checks that raise `AssertionError`), the harness's two pytest modules and those two pytest suites verify numbers at all.

## How to run (exact commands)
Interpreter is the project venv — no `python`/`python3` on `PATH` has torch. Every evogp entry point below needs
`PYTHONPATH=src:<evoxc checkout>` (`/root/evoxc` on the validation box) and a CUDA-visible GPU (the public API relocates
every tensor to `"cuda"`); add `EVOXC_DISABLE_LOCATIONS=1` to make the evoxc tracing ~4x faster (not part of the
engine's cache key).

```bash
PY=<venv>/bin/python                                   # project venv; py3.11, torch + evoxc available
export PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1

# 1. small public-API regression check: Tree.forward / SR_fitness across all execute modes (~3 s)
$PY test/fix_bug.py                     # exit 0 + three [ok] lines

# 2. the canonical gate (~122 s; 1 262 collected, 1 255 passed / 7 skipped / 0 failed)
$PY -m pytest src/evogp/tests -q

# 2b. install-validation smoke (~9.4 s on a device box — its default pop 200 rides `auto` onto the fused route; best fitness ≈ -0.2259)
#     export TRITON_CACHE_DIR=$TMPDIR/triton-cache first, or the first Triton launch dies on a read-only $HOME
$PY -m evogp.sr_test

# 2c. torch-free functional-stack suite (~66 s on a device box; 291 passed / 36 skipped, the 36 device-gated)
$PY -m pytest src/evogp/gp/tests -q

# 2d. full gate over the harness's torch-free tests plus both library suites
#     (1 295 passed / 7 skipped / 0 failed, ~121 s); the fused route's own net alone is
#     `-m pytest src/evogp/tests/test_fused_route.py -q` → 39 passed (~14 s)
$PY -m pytest test src/evogp/tests src/evogp/gp/tests -q

# 3. legacy goldens: inventory, and compare the current engine against the frozen captures
$PY test/golden_capture.py manifest
$PY test/golden_capture.py verify --out test/goldens      # see "Legacy goldens"
$PY test/golden_capture.py capture --out "$TMPDIR/goldens"  # new capture, scratch dir

# 4. evoxc "true jit" economics probe (needs the `evoxc` package; does not import evogp)
$PY test/evoxc_jit_probe.py --sections a,b             # ~3-5 min for all sections

# 5. engine-vs-plain-PyTorch campaign, one timed config at a time (see "Benchmark harnesses")
#    PYTHONPATH must also carry the harness dir: src:/root/evoxc:test/benchmark/evoxc
#    the shipped default is now the fused (Triton) route, so export TRITON_CACHE_DIR=$TMPDIR/triton-cache
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/<round> --cells main,n64 --arms all
#    --cells also takes long,large (the large-scale column, pop 2 048-8 192); a fused arm is
#    evoxc-fused-iree / evoxc-fused, and evoxc-auto times the shipped policy
```

Notebooks are not run by anything: launch them by hand from the repository root (relative asset paths
such as `./imgs/...` are cwd-dependent).

## Contents
| file | what it is | GPU | asserts |
|---|---|---|---|
| `fix_bug.py` | Public-API regression script (name is historical, it was a mode-agreement bug repro). Builds one hand-written 8-node `Tree` that decodes to `(x0 - x2) * (x0 - x2)` with no RNG involved, then checks: `Tree.forward` equals the closed form for a 2-D batch and a 1-D row; `Tree.SR_fitness` MSE/MAE equals the loss recomputed from `forward` with the documented `(1,)` positive-loss convention (never divided by `output_len`); the five `execute_mode` strings (`"torch"`, `"hybrid parallel"`, `"data parallel"`, `"tree parallel"`, `"auto"`) return bit-identical tensors — the legacy `kernel_type` dispatch is gone and all of them map onto one engine path; `Tree.to_forest()` reproduces both; an unknown mode string is rejected. | yes | yes (`AssertionError` + exit 1) |
| `golden_capture.py` | `argparse` driver for the `.npz` reference set: `capture` / `verify` / `manifest`, `--stage`, `--out`. Drives the installed engine through the public API on a fixed 64-tree / 63-node workload (plus a 200x12-generation training reference). See "Legacy goldens" for what the committed files are and what `capture`/`verify` mean today. | yes | `verify` exits 1 on a mismatch |
| `goldens/` | Data only: the 4 frozen captures `forest_random.npz`, `forward_eval.npz`, `sr_fitness.npz`, `training.npz` (+ `README.md`). No code; provenance, per-array inventory, tolerances and the historical build recipe live in `goldens/README.md`. | — | — |
| `evoxc_jit_probe.py` | Standalone `evoxc` economics probe (unrelated to the EvoGP engine — it exercises the `evoxc` package directly and never imports `evogp`): a realistic GP step built with `@evoxc.defn` + `evoxc.build_cached` + `evoxc.CompileCache`, printing build/run timings, generation-scale extrapolations, parallel builds, persistence and API-semantics checks. Sections a–h, selectable with `--sections`. | yes (iree-cuda sections) | own `PASS`/`FAIL` checks, exits 1 on failure |
| `vis.ipynb` | Scratch playground on the current public API (XOR: `Forest.random_generate` -> `GeneticProgramming` -> `StandardPipeline.run` -> `SR_fitness`/`forward`/`to_sympy_expr`/`simplify`/`to_infix`). API usage is current (re-verified against a reduced run), but the shipped budget (100000 trees, 1000 generations) is a legacy-kernel setting that is not runnable with the evoxc engine. A banner cell at the top states this and the reduced parameters to use (its text still carries the engine's pre-rename `ETL` name); stored outputs are historical. | yes | no |
| `combined_test.ipynb` | Multi-output demo written against the removed pre-`data_info` combined API; **cannot run as shipped** (see below). Kept as history with an explanatory banner cell. | yes | no |
| `sympy_test.ipynb` | 3 pure-sympy cells that build a nested `sp.Pow` expression and render it; no evogp import, no GPU, runs as-is. | no | no |

## Benchmark harnesses (`test/benchmark/`)
`test/benchmark/` is this node's sub-node (`benchmark/CONTEXT.md` holds the full routing table).
- `benchmark/evoxc/` — the current harness: **evoxc engine vs plain-PyTorch** on identical populations, in one session, warm/steady-state, over the engine's four routes (the shipped `auto` policy over `fused`/`dense`/`specialized`, the explicit `fused` (Triton) request, and the opt-in `batch` merge that `auto` never resolves). Two torch baselines (`torch-per-tree` Python tree walk, `torch-batched` one vectorized pass) plus the registered evoxc arms (`auto`, `auto`+iree, `specialized` numpy/iree, `dense` iree, `dense` iree at a raised trace cap, `fused` iree/plain, and the `batch` spellings banded/flat/default/aggressive/artifact), timed on `sr_fitness` / `batch_forward` / `forward` / `gp_step` over `pop ∈ {16,64,256,1000}` × `tree_len ∈ {32,64}` × `input_len ∈ {8,32}` at N=1024, the N=64 and long-tree columns and the ten-cell large-scale column (`--cells large`, pop 2 048–8 192), with a cold-start column (`cold_start.py`) and whole-training-run columns (`training_run.py`); one subprocess per `(cell, arm)` behind an idle-device guard, and every arm validated against `../src/evogp/tests/reference_eval.py` before its timings are trusted.
  Methodology and commands: `benchmark/evoxc/README.md`; routing: `benchmark/evoxc/CONTEXT.md`; validation: `benchmark/evoxc/VALIDATION.md`.
  Reports of record: `benchmark/evoxc/FINAL.md` (the shipped `specialized`/`dense` routes against both torch baselines, `SUMMARY.md` renders the earlier baseline round), `benchmark/evoxc/BATCH.md` (the opt-in `batch` route, evidence in `benchmark/evoxc/results/batch/` plus the optimization rounds `benchmark/evoxc/results/batch_opt_round1/`, `benchmark/evoxc/results/batch_opt_round2/` and `benchmark/evoxc/results/batch_opt_round3/`, each with its own `SUMMARY.md`, §15 being the round-3 closing section), and the large-scale column's two rounds `benchmark/evoxc/LARGE_SCALE.md` (over `benchmark/evoxc/results/largescale/`) and `benchmark/evoxc/LARGE_SCALE_ROUND2.md` (over `benchmark/evoxc/results/largescale2/`, the current report for that column).
  The three `t1a1` evidence directories are their own reports: `benchmark/evoxc/results/t1a1_auto/README.md` is the SHIPPED-DEFAULT column (`auto` → fused), where a `batch_forward` call is 38.4×–141.7× `torch-batched` and a whole generation is 2.58–2.84 ms with evaluation only 8.8–10.0 % of it and no per-generation compile; `benchmark/evoxc/results/t1a1_fused/README.md` is the EXPLICIT `EVOGP_EVAL_MODE=fused` column, 41×–157× over `torch-batched` on `batch_forward` (32×–131× on `sr_fitness`) with bitwise value paths; `benchmark/evoxc/results/t1a1_baseline/BASELINE.md` is the PRE-FUSED default (where `auto` resolved dense+iree), the reference those two are read against.
  The fused engine's own design/report document is `../src/evogp/tree/FUSED.md`, and the torch-free functional stack's device-resident loop is `../src/evogp/gp/DEVICE_LOOP.md` (1.16–1.40 ms per generation).
  Gate of record, the same form for every side: `python3 test/benchmark/evoxc/check.py --results <dir> --allow-snapshot --allow-missing --allow-digest-mismatch` — `results/final/after`, `results/batch/{warm-r5,small-r5,flat-r5,flat1000-3w-r5}`, `results/t1a1_auto` (23 records / 640 checks / 0 failures), `results/t1a1_fused` (20 records / 580 checks / 0 failures) and `results/t1a1_baseline/main` (29 records / 538 checks / 6 declared scoped-schedule failures).
- Batch route verdict (settled — do not re-litigate without new evidence): `batch` stays opt-in (`auto` never selects it) and is dominated in any GP loop, losing to dense — the control it was priced against — by 64.9× / 71.5× / 17.8× over 8 generations at `p256-L32-i8-N1024` / `p256-L64-i32-N1024` / `p64-L64-i32-N64` (eval-only 868× / 680× / 229×) with `generation_hit_rate` 0.125, while the shipped fused default wins the same turnover loop because it re-plans and recompiles nothing per generation (`benchmark/evoxc/results/t1a1_auto/README.md`).
  Its one warm win is a fixed population over few data rows (`p64-L64-i32-N64`: 1.1141 ms against dense 1.4378 ms, i.e. 1.29× faster; cold 2 274.46 ms against 6 554.88 ms), and on the static large-scale column it leads dense on 7 of 8 cells (`benchmark/evoxc/LARGE_SCALE_ROUND2.md`).
  Round 1 added chunked parallel compile, a content-addressed plan key and per-stage timers (`benchmark/evoxc/BATCH.md` §13, evidence `benchmark/evoxc/results/batch_opt_round1/`).
  Round 3 re-priced the chunked compile and keeps `EVOGP_BATCH_COMPILE_WORKERS=4`: the pop-256 cold compile goes 10.73 / 11.96 s → 4.06 / 4.07 s for 1.69–1.87× the warm call, break-even is 3 849 / 5 103 warm calls of one population (18 702 at 111-node trees), the `p64` plan does not chunk (`chunk_count` 1) so the knob is inert there, and `=1` stays the documented fixed-multiset escape hatch (`benchmark/evoxc/BATCH.md` §15, evidence `benchmark/evoxc/results/batch_opt_round3/`).
- the rest of `benchmark/` — historical records comparing the **legacy CUDA engine at `b613503`** with the evoxc engine: R1 (`SPEC.md` + `SUMMARY.md`), R2 (`r2/`), R3 (`r2/round3/`), R4 (`r4/`, which adds measurement type 2), R5 (`r5/`, diagnosis only) and R6 (`r6/`, the Triton prototype that was superseded by the shipped `fused` route), plus the round-1 `new/` and `legacy/` sides. Their verdicts, gap analysis and gates stay valid as history: `benchmark/REPORT.md` (incl. the frozen legacy anchor 0.6323 / 0.6357 ms and the R4-era ≈9.5–10.5× gap), `benchmark/r5/RCA.md`, `benchmark/r6/SUMMARY.md`.
- Two measurement types (defined once in `benchmark/CONTEXT.md`): type 1 = the end-to-end public-API evaluation (`problem.evaluate(forest)`), type 2 = the pure compiled-graph forward only (`bench_gp.py --task eval-pure`, runner core `pure_forward.py`; the legacy engine has no separable pure forward and exits 3). `bench_gp.py`'s `detect_side()` labels the engine family by the presence of `tree/evoxc_engine.py`; `pure_forward.py` and `stage_probe.py` serve the frozen R4/R5 rounds and are history-only.
- The engine's routing knobs decide what any number here measures: `EVOGP_EVAL_MODE` (`fused`/`dense`/`specialized`/`batch`/`auto`), `EVOGP_EVOXC_BACKEND` / `_DEVICE` / `_OPT_LEVEL` / `_CACHE_SIZE`, the fused route's `EVOGP_FUSED_*` knobs, the batch route's own `EVOGP_BATCH_*` knobs, and the opt-in on-disk `EVOGP_EVOXC_ARTIFACT_DIR`. With no backend named, `auto` takes the fused (Triton) route first from pop 16 (`triton` importable, a torch device usable, the fused memory envelope satisfied), else dense+iree from pop 64, else specialized+numpy — below the thresholds, on CPU and on any fallback; an explicitly named backend never reaches the fused route under `auto`, and `batch` is never selected.
  A fused arm also needs a writable `TRITON_CACHE_DIR`, and a served fused call is proved by `cache.fused_served` (a `fused_kernel_launches` counter), never by wall time, because a declined call falls back to specialized+numpy.
- Historical, not re-runnable: the R2/R3/R4 shadow halves need session-scoped `etl` checkout exports that no longer exist, and `benchmark/r5/probe_lib.py`'s `getattr(dense_module, "etl")` proxy raises on the current engine (see Known Issues).

## Legacy goldens (`goldens/`)
Provenance: the four `.npz` files were captured by `golden_capture.py capture` from the **pre-rewrite
legacy engine** — the hand-written CUDA kernels in the (now deleted) `src/evogp/cuda/`, bound through
`evogp.evogp_cuda`, at revision `b613503` — with nvcc 12.6 / sm_86 / torch 2.6.0+cu124. They are the
numeric reference the pure-torch/evoxc rewrite had to reproduce: evaluation semantics, the SR-fitness
convention, and the shape of a short training run. `goldens/README.md` is the source of truth for the
per-array inventory, the deviations worth knowing (fast-math kernels, reduction-order spread ~2.3e-05,
`-inf` from the pipeline's NaN replacement) and the recommended tolerances (exact for integer/string
arrays, `rtol=atol=1e-4, equal_nan=True` for floats).

Consumption:
* RNG-independent and therefore the strongest check — rebuild a `Forest` from the stored
  `forest__node_value`/`_node_type`/`_subtree_size` and compare `forward` / `batch_forward` /
  `SR_fitness` against the stored `tree_forward` / `forest_forward` / `forest_batch_forward` /
  `*_sr_fitness__*` arrays (recipe in `goldens/README.md`). Measured for today's engine: agreement
  within `2.3e-05` abs / `4.0e-07` rel with identical NaN/±inf placement, i.e. inside the documented
  `rtol=atol=1e-4` — the goldens still are a live numeric reference, not just history.
* RNG-dependent — the generated structures (`forest__node_*`, `initial_forest__*`, `final_forest__*`),
  the mutation/crossover outcomes and the training trajectory follow the legacy generator's RNG stream
  and are NOT reproducible by today's torch-RNG generator: use them as structural/behavioural
  reference only (trees must be legal and decode; generation 1 no worse than
  `best_fitness_per_generation[1] = -0.23594594`).

Regeneration: `golden_capture.py` still works, but it drives *whatever engine is installed*:
`capture` would overwrite the frozen legacy numbers with current-engine values (the script prints a
warning when `--out` resolves to `test/goldens/`), and `verify` only compares one build against itself.
The original recipe (nvcc flags, `TORCH_CUDA_ARCH_LIST`, the `build_ext --inplace` step) is kept in
`goldens/README.md` as history — it can no longer be executed, because `src/evogp/cuda/` and
`setup.py`'s `CUDAExtension` are gone. Cross-engine runs are expected to report the descriptor/header/
input arrays (`descriptor__*`, `x`, `x_small`, `labels`, `*_scalars`, `seed`) as bitwise and every
RNG-dependent structure/trajectory array as MISMATCH.

Relation to the pytest suite: `../src/evogp/tests/test_equivalence.py` scans `$EVOGP_GOLDEN_DIR`, then
`<repo-root>/test/goldens/` (this directory), then its own `goldens/`, and recognizes two schemas by
array name — the flat "engine" fixtures (`node_value`, `node_type`, `subtree_size`, `x` plus `expected`
or `expected_fitness`, the 7 files in `src/evogp/tests/goldens/`) and the raw legacy captures here
(`forest__node_value`/… triples plus the `tree_forward` / `forest_forward` / `*_sr_fitness__*` / … expectations).
Files matching neither schema (e.g. foreign fixtures dropped into a golden dir) are reported through
`warnings.warn` and skipped, never failed.

## Known Issues / cross-node notes
- `goldens/**` stays frozen reference data: `golden_capture.py verify` compares key sets exactly, and the
  raw captures are deliberately not converted into the flat engine-fixture schema — the pytest loader
  consumes them in their own (legacy-capture) schema instead, see "Legacy goldens".
- The multi-output combined model is only exercised by `../src/evogp/tests/test_api_compat.py`. The
  removed `formula=` / `share_input=` API is still called by `example/combined_sr.py` and
  `tutorial/combined_gp.ipynb` (both outside this node) and by `combined_test.ipynb` here.
- No numeric assertions in the notebooks, no golden values for anything outside the four captures, no
  coverage of the robotics/classification problems, and nothing in this directory is wired into CI (no CI exists; the harness's two `test_*.py` modules are collected only by the full `pytest test src/evogp/tests src/evogp/gp/tests` gate).
- The shipped default is the fused (Triton) route, so quoting or re-gating any `benchmark/evoxc/results/t1a1_*` number needs a CUDA/ROCm-visible device, `triton`, and a writable `TRITON_CACHE_DIR`; on a CPU box or without `triton`, `auto` resolves specialized+numpy and those columns are not reproducible.
- `decompose.py`'s `ROUTES` table has no `fused` entry and `cold_start.py` hardcodes the `evoxc-auto` arm, so a per-phase or cold-start fused column needs an edit in the harness (both live under `benchmark/evoxc/`).
- `evoxc_jit_probe.py` is bound to the `evoxc` package (read from the checkout named by its `EVOXC_TESTS`
  path constant); it needs an idle GPU for quotable timings (the host is shared).
- `evoxc_jit_probe.py` emits no machine-readable output (stdout report table only; exit 1 when one of its own
  checks fails). The JSON-emitting tools are `../src/evogp/tests/perf_probe.py` (evoxc economics),
  `benchmark/bench_gp.py` (one JSON per run) and `benchmark/evoxc/` (per-cell records + `campaign.json` +
  `validation.json`).
- `benchmark/r5/probe_lib.py` still proxies `getattr(dense_module, "etl")`, so it raises on the current
  engine; the r5/r6 probe-based measurements are historical, while the stdlib gates over the committed
  records (`check_r5.py`, `check_r6.py`) still run.
- The R2/R3/R4 `etl`-shadow halves are historical: those re-runs need session-scoped `etl` checkout
  exports that no longer exist, so their committed records, probes and reports are evidence, not steps.
- The legacy-side benchmark build lives at `/tmp/evogp_legacy_bench` (session-scoped — it can vanish; the built
  `evogp_cuda` extension needs the offline assets in `/tmp/evogp_t1a263` and the CUDA 12.6 nvcc at
  `/opt/miniforge3/envs/wenet_tiktoken` to rebuild).

## Test Strategy
- Manual smoke sequence for a post-rewrite change: `test/fix_bug.py` (public-API numerics, ~3 s) ->
  `pytest src/evogp/tests -q` (the real gate; 1 255 passed / 7 skipped / 0 failed, 1 262 collected, ~122 s, with
  `PYTHONPATH=src:/root/evoxc EVOXC_DISABLE_LOCATIONS=1`; `--ignore=src/evogp/tests/test_equivalence.py`
  for the fast subset, `-k golden` for the golden cross-checks, `-k fused` for the 39-case fused-route net) ->
  `pytest src/evogp/gp/tests -q` (the torch-free functional suite, 291 passed / 36 skipped) -> `test/golden_capture.py verify --out
  test/goldens` if you touched evaluation semantics and want the legacy-CUDA diff (expect the
  RNG-dependent arrays to differ). `python -m evogp.sr_test` is the install-validation smoke
  (~9.4 s on a device box, best fitness ≈ -0.2259).
- The full gate over everything is `pytest test src/evogp/tests src/evogp/gp/tests -q` → 1 295 passed /
  7 skipped / 0 failed in ~121 s (it adds the harness's torch-free `benchmark/evoxc/` modules); the whole library
  suite is green, so there is no known red case.
- `test/evoxc_jit_probe.py` is independent of evogp's engine and is not part of that sequence.
- Engine-vs-plain-PyTorch performance questions go to `benchmark/evoxc/`: `validate.py` (numbers first) ->
  `campaign.py` (one process per config, idle device) -> `check.py` -> `summarize.py`; never run a timed
  config next to anything else that imports torch, and export a writable `TRITON_CACHE_DIR` for the fused arms.
- Re-gating a committed result directory is GPU-free: `python3 test/benchmark/evoxc/check.py --results
  <dir> --allow-snapshot --allow-missing --allow-digest-mismatch`, e.g. over
  `test/benchmark/evoxc/results/t1a1_auto`, `.../t1a1_fused` and `.../t1a1_baseline/main`.
- Coverage gaps: the raw legacy captures are not converted into the pytest fixture schema (see Known
  Issues), and there is no automated execution of the notebooks.

## Routing Table / See Also
`goldens/` is data only (documented by its own `README.md`), so do not look for sub-nodes there.
`benchmark/` is this node's sub-node with its own `CONTEXT.md` (full routing table): `benchmark/evoxc/` is the current engine-vs-plain-PyTorch harness over the engine's four routes, with six reports of record — `benchmark/evoxc/FINAL.md` (shipped `specialized`/`dense` routes), `benchmark/evoxc/BATCH.md` (opt-in `batch` route, evidence in `benchmark/evoxc/results/batch/` plus the optimization rounds `benchmark/evoxc/results/batch_opt_round1/`, `benchmark/evoxc/results/batch_opt_round2/` and `benchmark/evoxc/results/batch_opt_round3/`, each with its own `SUMMARY.md`, §15 being the round-3 closing section), `benchmark/evoxc/LARGE_SCALE.md` and `benchmark/evoxc/LARGE_SCALE_ROUND2.md` (the large-scale column, evidence in `benchmark/evoxc/results/largescale/` and `.../largescale2/`), and the two `t1a1` READMEs `benchmark/evoxc/results/t1a1_auto/README.md` (the SHIPPED default, `auto` → fused) and `benchmark/evoxc/results/t1a1_fused/README.md` (the explicit fused column), both read against `benchmark/evoxc/results/t1a1_baseline/BASELINE.md` — plus `benchmark/evoxc/CONTEXT.md`, `benchmark/evoxc/README.md` and validation `benchmark/evoxc/VALIDATION.md`, and the rest of `benchmark/` holds the historical legacy-CUDA-engine-vs-evoxc rounds (verdict: `benchmark/REPORT.md`; gap analysis: `benchmark/r5/RCA.md`; superseded Triton prototype: `benchmark/r6/SUMMARY.md`).
Sibling paths (read-only from this node — escalate any needed write to the parent `./`):
`../src/evogp/` (library; `../src/evogp/tests/` is the canonical pytest suite, the golden-fixture
loader, and the evoxc-pattern economics probe `../src/evogp/tests/perf_probe.py` — JSON-emitting, pop-200/1000
extrapolation, never imports evogp; `../src/evogp/gp/tests/` is the torch-free functional-stack suite),
`../example/` and `../tutorial/` (maintained runnable examples /
curated notebooks), `../src/evogp/tree/ENGINE_SPEC.md` (engine semantics of record, incl. the legacy op schemas).
Foreign repo `evoxc` (`/root/evoxc`, the evaluation-engine dependency) via `evoxc_jit_probe.py` (read-only).
