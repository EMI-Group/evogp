# `test/` — hand-run validation tooling around the canonical pytest suite

## Intent
Ad-hoc, human-driven validation layer for the repo: one small public-API regression script, the frozen
**pre-ETL numeric reference** (`goldens/`) plus the script that captured and compares it, a CPU-only
torch-extension preflight, an ETL economics probe, and three scratch notebooks.
Nothing here is imported by `evogp`, by the examples or by the pytest suite, and no runner collects these
files (no CI, no pytest ini in the repo) — every file is executed by hand from the repository root.
The real regression gate is the pytest suite in `../src/evogp/tests/` (independent plain-torch oracle,
randomized/adversarial equivalence, operator invariants, API compatibility, convergence smoke): treat
this directory as the layer *around* it, i.e. legacy cross-checks, environment probes and
human-watched demos.
Because there is no test framework here, a green run only means "it ran and its checks held"; only
`fix_bug.py` (explicit checks that raise `AssertionError`) and the pytest suite verify numbers at all.

## How to run (exact commands)
Interpreter is the project venv — `python` is not on PATH. Every evogp entry point below needs
`PYTHONPATH=src` and a CUDA-visible GPU (the public API relocates every tensor to `"cuda"`); add
`ETL_DISABLE_LOCATIONS=1` to make the ETL tracing ~4x faster (not part of the engine's cache key).

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

# 4. CPU-only preflight of torch's C++-extension machinery (legacy, needs ninja + a host compiler)
$PY test/test_torch_env.py

# 5. ETL "true jit" economics probe (needs the `etl` package; does not import evogp)
$PY test/etl_jit_probe.py --sections a,b               # ~3-5 min for all sections
```

Notebooks are not run by anything: launch them by hand from the repository root (relative asset paths
such as `./imgs/...` are cwd-dependent).

## Contents
| file | what it is | GPU | asserts |
|---|---|---|---|
| `fix_bug.py` | Public-API regression script (name is historical, it was a mode-agreement bug repro). Builds one hand-written 8-node `Tree` that decodes to `(x0 - x2) * (x0 - x2)` with no RNG involved, then checks: `Tree.forward` equals the closed form for a 2-D batch and a 1-D row; `Tree.SR_fitness` MSE/MAE equals the loss recomputed from `forward` with the documented `(1,)` positive-loss convention (never divided by `output_len`); the five `execute_mode` strings (`"torch"`, `"hybrid parallel"`, `"data parallel"`, `"tree parallel"`, `"auto"`) return bit-identical tensors — the legacy `kernel_type` dispatch is gone and all of them map onto one engine path; `Tree.to_forest()` reproduces both; an unknown mode string is rejected. | yes | yes (`AssertionError` + exit 1) |
| `golden_capture.py` | `argparse` driver for the `.npz` reference set: `capture` / `verify` / `manifest`, `--stage`, `--out`. Drives the installed engine through the public API on a fixed 64-tree / 63-node workload (plus a 200x12-generation training reference). See "Legacy goldens" for what the committed files are and what `capture`/`verify` mean today. | yes | `verify` exits 1 on a mismatch |
| `goldens/` | Data only: the 4 frozen captures `forest_random.npz`, `forward_eval.npz`, `sr_fitness.npz`, `training.npz` (+ `README.md`). No code; provenance, per-array inventory, tolerances and the historical build recipe live in `goldens/README.md`. | — | — |
| `test_torch_env.py` | Legacy preflight: compiles a tiny CPU-only `TORCH_LIBRARY(extension_cpp)` through `torch.utils.cpp_extension.load_inline` and calls `torch.ops.extension_cpp.mymuladd(a, b, 1.0)`. It validated the toolchain of the deleted C++/CUDA extension build; nothing in the repo builds a C++ extension any more, it does not import evogp, needs no GPU, and in the project venv it currently raises `RuntimeError: Ninja is required to load C++ extensions`. Kept only as an environment probe. | no | no (prints) |
| `etl_jit_probe.py` | Standalone ETL economics probe (unrelated to the EvoGP engine — it exercises the `etl` package directly and never imports `evogp`): a realistic GP step built with `@etl.defn` + `etl.build_cached` + `etl.CompileCache`, printing build/run timings, generation-scale extrapolations, parallel builds, persistence and API-semantics checks. | yes (iree-cuda sections) | own `PASS`/`FAIL` checks, exits 1 on failure |
| `vis.ipynb` | Scratch playground on the current public API (XOR: `Forest.random_generate` -> `GeneticProgramming` -> `StandardPipeline.run` -> `SR_fitness`/`forward`/`to_sympy_expr`/`simplify`/`to_infix`). API usage is current (re-verified against a reduced run), but the shipped budget (100000 trees, 1000 generations) is a legacy-kernel setting that is not runnable with the ETL engine. A banner cell at the top states this and the reduced parameters to use; stored outputs are historical. | yes | no |
| `combined_test.ipynb` | Multi-output demo written against the removed pre-`data_info` combined API; **cannot run as shipped** (see below). Kept as history with an explanatory banner cell. | yes | no |
| `sympy_test.ipynb` | 3 pure-sympy cells that build a nested `sp.Pow` expression and render it; no evogp import, no GPU, runs as-is. | no | no |

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
  (ETL) engine").
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

Relation to the pytest suite: `../src/evogp/tests/test_equivalence.py::test_golden_fixtures` scans
`$EVOGP_GOLDEN_DIR`, then `<repo-root>/test/goldens/` (this directory), then its own `goldens/`, and
requires the flat fixture keys `node_value`, `node_type`, `subtree_size`, `x` plus `expected` or
`expected_fitness`. The 7 fixtures in `src/evogp/tests/goldens/` use that schema; the raw captures here
are keyed `forest__node_value`/… and are *not* in it.

## Known Issues / cross-node notes
- `src/evogp/tests/test_equivalence.py::test_golden_fixtures` is the one failure of the canonical suite
  (last full run: 1 failed / 286 passed / 3 skipped): the loader `pytest.fail`s on the first `*.npz` it
  finds in `test/goldens/` because the raw legacy captures miss the flat fixture keys. The mismatch can be
  resolved on either side — the tests node can skip/warn on files that are not in the fixture schema
  (they are still valid `npz` data), or `goldens/` can gain derived flat-key fixtures
  (`forest__node_*` -> `node_value`/`node_type`/`subtree_size`, `x`/`labels` as-is, `expected_fitness` =
  `forest_sr_fitness__hybrid_parallel__mse` for `sr_fitness.npz`). The latter is deliberately not done
  here: `goldens/**` is frozen reference data and `golden_capture.py verify` compares key sets exactly.
- `test_torch_env.py` is dead tooling against the current repo (no C++ extension exists, and it fails in
  the project venv without ninja). Removal was left to the owner of the surrounding area rather than done
  here.
- The multi-output combined model is only exercised by `../src/evogp/tests/test_api_compat.py`. The
  removed `formula=` / `share_input=` API is still called by `example/combined_sr.py` and
  `tutorial/combined_gp.ipynb` (both outside this node) and by `combined_test.ipynb` here.
- No numeric assertions in the notebooks, no golden values for anything outside the four captures, no
  coverage of the robotics/classification problems, and nothing in this directory is wired into CI.
- `etl_jit_probe.py` is bound to the `etl` package in the read-only foreign repo
  `/mnt/local-ssd/bchuang/etl`; it needs an idle GPU for quotable timings (the host is shared).

## Test Strategy
- Manual smoke sequence for a post-rewrite change: `test/fix_bug.py` (public-API numerics, ~3 s) ->
  `pytest src/evogp/tests -q` (the real gate) -> `test/golden_capture.py verify --out test/goldens` if
  you touched evaluation semantics and want the pre-ETL diff (expect the RNG-dependent arrays to differ).
- `test/test_torch_env.py` and `test/etl_jit_probe.py` are independent of evogp's engine and are not part
  of that sequence.
- Coverage gaps: the raw legacy captures are not converted into the pytest fixture schema (see Known
  Issues), and there is no automated execution of the notebooks.

## Routing Table / See Also
No code subdirectories: `goldens/` is data only (documented by its own `README.md`), so do not look for
sub-nodes to delegate to here.
Sibling paths (read-only from this node — escalate any needed write to the parent `./`):
`../src/evogp/` (library; `../src/evogp/tests/` is the canonical pytest suite and the golden-fixture
loader), `../example/` and `../tutorial/` (maintained runnable examples / curated notebooks),
`../src/evogp/tree/ENGINE_SPEC.md` (engine semantics of record, incl. the legacy op schemas).
Foreign repo `etl` (`/mnt/local-ssd/bchuang/etl`) via `etl_jit_probe.py` (read-only).
