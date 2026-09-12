# `test/benchmark/` — legacy CUDA engine vs ETL "true jit" engine

Performance comparison of the **two evaluation engines** of this project, measured on one shared box
through the identical public API:

* **OLD** — the pre-rewrite custom-CUDA engine at commit `b613503` (`src/evogp/cuda/*.cu`, built as
  `evogp.evogp_cuda`; materialized and compiled locally because the current tree no longer contains it).
* **NEW** — this repository's ETL "true jit" engine (`src/evogp/tree/etl_engine.py`, host-side decode +
  per-structure `etl.build_cached` graphs), default `"numpy"` backend, `ETL_DISABLE_LOCATIONS=1`. An
  IREE-backend run exists only as a clearly-labelled, non-primary datapoint.

**Headline result** (details and every raw number in `SUMMARY.md`): the legacy compiled-CUDA engine is
roughly **34× to 7,670× faster per generation**, depending on config — 34× for a warm 64-tree eval,
54× for the main pop-200 SR run, ~230–850× for the larger/odd configs, and ~7,700× for the pop-1000 /
tree-128 run where tree bloat makes the new side pay a fresh per-structure JIT build for almost every
offspring. The new engine trades that speed for **no build step at all** (no CUDA toolkit, no `nvcc`,
no extension module, device-agnostic code) and is the only engine still in the repository; the headroom
for a future reader is on the new side (batching/memoizing ETL builds across generations, avoiding
rebuilds of unchanged structures), while the old side's speed is not recoverable without reintroducing
compiled kernels. The old-side numbers were cross-validated through the same side-agnostic harness — see
`xval/` (summarized in `SUMMARY.md` §5.1).

## Contents

| path | what it is |
|---|---|
| `SPEC.md` | **Frozen methodology** (v1, 2026-09-11): workloads W1–W3, config matrix C0–C7, fairness rules, required JSON schema, priority order. Do not edit. |
| `SUMMARY.md` | **Headline report** — verdict, per-config old-vs-new table (all numbers recomputed from the raw JSONs), why the gap looks like this, caveats, correctness anchor + validator results, reproducibility. |
| `bench_gp.py` | Side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; one JSON per run + per-generation/per-call timings. |
| `new/` | NEW side: `run_matrix.sh` (config runner), `check_smoke.py` (JSON validator), `NOTES.md` (harness/schema, smoke suite, matrix runs, observations, deviations), `results/` (8 matrix JSONs + logs). |
| `legacy/` | OLD side: `setup_legacy.sh` (materialize + build `b613503`), `run_matrix.py` (config runner), `check_results.py` (JSON validator, incl. the C5 golden cross-check), `BUILD_NOTES.md` (build recipe/`.so` identity/GPU-7 fault), `NOTES.md`, `results/` (8 matrix JSONs + logs; `c4_rerun/` = noisier variance datapoint, not a headline record). |
| `xval/` | **Cross-validation**: the old-side C1/C5 re-run through the unified `bench_gp.py` (methodology-uniformity evidence) — deltas vs the original runs, the C5 `wall_s` driver-scope caveat and helper probes; see `xval/NOTES.md` / `xval/CONTEXT.md`, summarized in `SUMMARY.md` §5.1. |

## Re-running

Both sides need a CUDA-visible device and the project venv (no `python` on `PATH`), and both must be
run from their own checkout root with `PYTHONPATH=<checkout>/src` (OLD additionally needs
`:/tmp/evogp_t1a263/siteutils` for offline sklearn).

```bash
# NEW (from this repository root) — runner flocks internally; ~37 min of compute for all 8 configs
./test/benchmark/new/run_matrix.sh              # or: ./test/benchmark/new/run_matrix.sh C1 C5
# validate what it wrote (JSON-only, no GPU):
PYTHONPATH=src <venv>/bin/python test/benchmark/new/check_smoke.py test/benchmark/new/results/*.json

# OLD: build once (idempotent; offline assets + CUDA recipe: legacy/BUILD_NOTES.md), then run the matrix
# from the legacy checkout root with the legacy PYTHONPATH order (`--configs` is comma-separated)
./test/benchmark/legacy/setup_legacy.sh
cd /tmp/evogp_legacy_bench && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/tmp/evogp_t1a263/siteutils \
  <venv>/bin/python <repo>/test/benchmark/legacy/run_matrix.py --configs C1,C5 \
  --out <repo>/test/benchmark/legacy/results --commit b613503
<venv>/bin/python <repo>/test/benchmark/legacy/check_results.py <repo>/test/benchmark/legacy/results/*.json --gpu 4
```

**Shared host:** the box has other tenants on the A6000s, and both sides share one advisory lock —
serialize every heavy phase (builds, timed runs) with exactly one, never nested:
`flock -w 7200 /tmp/gpbench_critical.lock bash -c '<cmd>'` (the wait is outside all timed regions).
Treat medians as ±10%; pick the least-busy GPU at run time and skip GPU 7 (hardware ECC fault). Full
environment, seeds, launcher and per-config commands: `SUMMARY.md` §6, `new/NOTES.md`, `legacy/NOTES.md`.

## R2 — GPU round (legacy CUDA vs ETL on IREE-cuda)

Round 2 (`r2/`) repeats the old-vs-new comparison with this harness and adds what round 1 lacked:
data-row scaling, a 200-tree fixture, the ETL `stages` probe (cold builds vs warm cache misses), an
IREE cold-build budget, and probe-off A/B controls. Run it from the repo root:
`test/benchmark/r2/run_all.sh` (`--group light|iree|all`, `--only <id>`, `--dry-run`); validate with
`python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages`.
Configs, rules, exit codes, wall-time expectations and results paths: `r2/README.md`. The round's
written report lands in `REPORT.md` once `new/results/r2/` + `legacy/results/r2/` are filled; round-1
methodology (`SPEC.md`) is unchanged.
