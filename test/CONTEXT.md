# `test/` — hand-run validation tooling around the canonical pytest suite

## Intent
Ad-hoc, human-driven validation layer for the repo: one small public-API regression script, the frozen
**pre-ETL numeric reference** (`goldens/`) plus the script that captured and compares it, an `evoxc`
economics probe, two performance-benchmark harnesses under `benchmark/`, and three scratch notebooks.
Nothing here is imported by `evogp`, by the examples or by the pytest suite, and no runner collects these
files (no CI, no pytest ini in the repo) — every file is executed by hand from the repository root.
The real regression gate is the pytest suite in `../src/evogp/tests/` (independent plain-torch oracle,
randomized/adversarial equivalence, operator invariants, API compatibility, convergence smoke): treat
this directory as the layer *around* it, i.e. legacy cross-checks, an `evoxc` tooling probe, the
benchmark harnesses and human-watched demos.
Because there is no test framework here, a green run only means "it ran and its checks held"; only
`fix_bug.py` (explicit checks that raise `AssertionError`) and the pytest suite verify numbers at all.

## How to run (exact commands)
Interpreter is the project venv — `python` is not on PATH. Every evogp entry point below needs
`PYTHONPATH=src` and a CUDA-visible GPU (the public API relocates every tensor to `"cuda"`); add
`EVOXC_DISABLE_LOCATIONS=1` to make the evoxc tracing ~4x faster (not part of the engine's cache key).

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python        # py3.11, torch 2.6.0+cu124

# 1. small public-API regression check: Tree.forward / SR_fitness across all execute modes (~3 s)
PYTHONPATH=src $PY test/fix_bug.py                     # exit 0 + three [ok] lines

# 2. the canonical gate, for reference (GPU, ~20 min)
PYTHONPATH=src $PY -m pytest src/evogp/tests -q

# 3. legacy goldens: inventory, and compare the current engine against the frozen captures
PYTHONPATH=src $PY test/golden_capture.py manifest
PYTHONPATH=src $PY test/golden_capture.py verify --out test/goldens      # see "Legacy goldens"
PYTHONPATH=src $PY test/golden_capture.py capture --out "$TMPDIR/goldens"  # new capture, scratch dir

# 4. evoxc "true jit" economics probe (needs the `evoxc` package; does not import evogp)
$PY test/evoxc_jit_probe.py --sections a,b             # ~3-5 min for all sections

# 5. engine-vs-plain-PyTorch harness, one timed config at a time (see "Benchmark harnesses")
#    PYTHONPATH=src:<evoxc checkout>:test/benchmark/evoxc
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/baseline --cells main,n64 --arms all
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
| `vis.ipynb` | Scratch playground on the current public API (XOR: `Forest.random_generate` -> `GeneticProgramming` -> `StandardPipeline.run` -> `SR_fitness`/`forward`/`to_sympy_expr`/`simplify`/`to_infix`). API usage is current (re-verified against a reduced run), but the shipped budget (100000 trees, 1000 generations) is a legacy-kernel setting that is not runnable with the evoxc engine. A banner cell at the top states this and the reduced parameters to use (the banner text still says "ETL"); stored outputs are historical. | yes | no |
| `combined_test.ipynb` | Multi-output demo written against the removed pre-`data_info` combined API; **cannot run as shipped** (see below). Kept as history with an explanatory banner cell. | yes | no |
| `sympy_test.ipynb` | 3 pure-sympy cells that build a nested `sp.Pow` expression and render it; no evogp import, no GPU, runs as-is. | no | no |

## Benchmark harnesses (`test/benchmark/`)
`test/benchmark/` is this node's sub-node (`benchmark/CONTEXT.md` holds the full routing table) and carries two different questions.
- `benchmark/evoxc/` — the current, re-runnable harness: **evoxc engine vs plain-PyTorch** on identical populations, in one session, warm/steady-state. Two torch baselines (`torch-per-tree` Python tree walk, `torch-batched` one vectorized pass) plus five evoxc routes (`auto`, `auto`+iree, `specialized` numpy/iree, `dense` iree), timed on `sr_fitness` / `batch_forward` / `forward` / `gp_step` over `pop ∈ {16,64,256,1000}` × `tree_len ∈ {32,64}` × `input_len ∈ {8,32}` at N=1024 plus an N=64 column; one subprocess per `(cell, arm)` behind an idle-device guard, and every arm validated against `../src/evogp/tests/reference_eval.py` before its timings are trusted. Methodology and commands: `benchmark/evoxc/README.md`; routing: `benchmark/evoxc/CONTEXT.md`; report: `benchmark/evoxc/SUMMARY.md` (rendered by `summarize.py` from `results/`); gate: `python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/baseline`.
- the rest of `benchmark/` — the frozen old-vs-new rounds comparing the **legacy CUDA engine at `b613503`** with the evoxc engine: R1 (`SPEC.md` + `SUMMARY.md`), R2 (`r2/`), R3 (`r2/round3/`), R4 (`r4/`, which adds measurement type 2), R5 (`r5/`, diagnosis only) and R6 (`r6/`, Triton prototype). Verdict and ranked levers: `benchmark/REPORT.md`; gap root cause: `benchmark/r5/RCA.md`; Triton verdict: `benchmark/r6/SUMMARY.md`. Those records, reports and gates are committed evidence and are not edited by the `evoxc/` harness.
Round protocol (R1–R6): warm/steady-state medians on GPU 6, one process per timed config (the compile cache and the executable memo are per process), cold compile/first call reported separately and never inside the warm numbers, two measurement types in `bench_gp.py` — type 1 = end-to-end `problem.evaluate(forest)`, type 2 = pure compiled-graph forward (`--task eval-pure`, runner core `pure_forward.py`; the legacy engine has no separable pure forward and exits 3). The `evoxc/` harness keeps the one-process-per-config and warm-median rules and guards device idleness instead of pinning GPU 6.
Harness: `bench_gp.py` drives the real public API on either engine (`--task sr|xor|eval64|evalfresh|eval-pure`, `--backend {default,numpy,iree}`, `--iree-device`, `--iree-opt-level {O0,O1}`, `--data-rows`, `--mem-guard-mb`, `--max-cold-builds`, `--stage-probe`, FD telemetry, peak-memory reporting; the pure task adds `--mode`, `--invocations`, `--warmup`, `--e2e-companion`, `--pure-runner`); every timed region is `torch.cuda.synchronize()`-wrapped and one JSON is written atomically per run.
Frozen R4 anchor (quoted from `benchmark/REPORT.md`): default iree `auto` reads **6.667 / 6.050 ms** warm on fixture64×n64 / fixture200×n64 against the cross-window legacy anchor 0.6323 / 0.6357 ms ⇒ old-vs-new gap **≈9.5–10.5×**.
Historical, not re-runnable: the R2/R3/R4 shadow halves need session-scoped `etl` checkout exports that no longer exist, and `benchmark/r5/probe_lib.py`'s `getattr(dense_module, "etl")` proxy raises on the current engine (see Known Issues).

## Legacy goldens (`goldens/`)
Provenance: the four `.npz` files were captured by `golden_capture.py capture` from the **pre-ETL
engine** — the hand-written CUDA kernels in the (now deleted) `src/evogp/cuda/`, bound through
`evogp.evogp_cuda`, at revision `b613503` — with nvcc 12.6 / sm_86 / torch 2.6.0+cu124. They are the
numeric reference the pure-torch/ETL rewrite had to reproduce: evaluation semantics, the SR-fitness
convention, and the shape of a short training run. `goldens/README.md` is the source of truth for the
per-array inventory, the deviations worth knowing (fast-math kernels, reduction-order spread ~2.3e-05,
`-inf` from the pipeline's NaN replacement) and the recommended tolerances (exact for integer/string
arrays, `rtol=atol=1e-4, equal_nan=True` for floats).

Consumption:
* RNG-independent and therefore the strongest check — rebuild a `Forest` from the stored
  `forest__node_value`/`_node_type`/`_subtree_size` and compare `forward` / `batch_forward` /
  `SR_fitness` against the stored `tree_forward` / `forest_forward` / `forest_batch_forward` /
  `*_sr_fitness__*` arrays (recipe in `goldens/README.md` §"How to use these goldens with the current
  (ETL) engine"). Measured for today's engine: agreement within `2.3e-05` abs / `4.0e-07` rel with
  identical NaN/±inf placement, i.e. inside the documented `rtol=atol=1e-4` — the goldens still are a
  live numeric reference, not just history.
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
  coverage of the robotics/classification problems, and nothing in this directory is wired into CI.
- `evoxc_jit_probe.py` is bound to the `evoxc` package (read from the checkout named by its `EVOXC_TESTS`
  path constant); it needs an idle GPU for quotable timings (the host is shared).
- `evoxc_jit_probe.py` emits no machine-readable output (stdout report table only; exit 1 when one of its own
  checks fails). The JSON-emitting tools are `../src/evogp/tests/perf_probe.py` (evoxc economics),
  `benchmark/bench_gp.py` (one JSON per run) and `benchmark/evoxc/` (per-cell records + `campaign.json` +
  `validation.json`).
- `benchmark/r5/probe_lib.py` still proxies `getattr(dense_module, "etl")`, so it raises on the current
  engine; the r5/r6 probe-based measurements are historical, while the stdlib gates over the committed
  records (`check_r5.py`, `check_r6.py`) still run.
- The `etl`-shadow half of R2/R3/R4 is historical: those re-runs need session-scoped `etl` checkout
  exports that no longer exist, so their committed records, probes and reports are evidence, not steps.
- The legacy-side benchmark build lives at `/tmp/evogp_legacy_bench` (session-scoped — it can vanish; the built
  `evogp_cuda` extension needs the offline assets in `/tmp/evogp_t1a263` and the CUDA 12.6 nvcc at
  `/opt/miniforge3/envs/wenet_tiktoken` to rebuild).

## Test Strategy
- Manual smoke sequence for a post-rewrite change: `test/fix_bug.py` (public-API numerics, ~3 s) ->
  `pytest src/evogp/tests -q` (the real gate) -> `test/golden_capture.py verify --out test/goldens` if
  you touched evaluation semantics and want the pre-ETL diff (expect the RNG-dependent arrays to differ).
- `test/evoxc_jit_probe.py` is independent of evogp's engine and is not part of that sequence.
- Engine-vs-plain-PyTorch performance questions go to `benchmark/evoxc/`: `validate.py` (numbers first) ->
  `campaign.py` (one process per config, idle device) -> `check.py` -> `summarize.py`; never run a timed
  config next to anything else that imports torch.
- Coverage gaps: the raw legacy captures are not converted into the pytest fixture schema (see Known
  Issues), and there is no automated execution of the notebooks.

## Routing Table / See Also
`goldens/` is data only (documented by its own `README.md`), so do not look for sub-nodes there.
`benchmark/` is this node's sub-node with its own `CONTEXT.md` (full routing table) and carries two harnesses:
`benchmark/evoxc/` is the current engine-vs-plain-PyTorch harness (`benchmark/evoxc/CONTEXT.md`, `benchmark/evoxc/README.md`, report `benchmark/evoxc/SUMMARY.md`), and the rest of `benchmark/` holds the frozen legacy-CUDA-engine-vs-evoxc rounds (verdict: `benchmark/REPORT.md`; gap analysis: `benchmark/r5/RCA.md`; Triton prototype: `benchmark/r6/SUMMARY.md`).
Sibling paths (read-only from this node — escalate any needed write to the parent `./`):
`../src/evogp/` (library; `../src/evogp/tests/` is the canonical pytest suite, the golden-fixture
loader, and the evoxc-pattern economics probe `../src/evogp/tests/perf_probe.py` — JSON-emitting, pop-200/1000
extrapolation, never imports evogp), `../example/` and `../tutorial/` (maintained runnable examples /
curated notebooks), `../src/evogp/tree/ENGINE_SPEC.md` (engine semantics of record, incl. the legacy op schemas).
Foreign repo `evoxc` (`/mnt/local-ssd/bchuang/evoxc`, the evaluation-engine dependency) via `evoxc_jit_probe.py` (read-only).
