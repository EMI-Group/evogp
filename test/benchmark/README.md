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

## R2 round (GPU, current)

Round 2 (`r2/`) is the current round: the same side-agnostic harness on the same box, comparing the legacy
CUDA engine against **both** ETL backends (default `numpy` CPU and opt-in IREE-cuda GPU), with
**warm/steady-state per-call numbers as the headline and per-structure compilation reported separately**
(the numbers above are round 1; the round-1 methodology in `SPEC.md` is unchanged).

* **What it measured** — the round-1 workloads again, plus data-row scaling (64 / 1 024 / 10 000 rows), a
  200-tree fixture (`fixtures/`), the `stages` cold-build vs cache-hit decomposition (`stage_probe.py`), an
  IREE cold-build budget, and probe-off A/B controls — 35 manifest configs, 34 complete.
  `iree_sr_b2` (pop 1000 / tree 128, IREE) died with SIGSEGV before generation 0 and is kept as the
  round's bounded-failure record, not a measurement.
* **Headline (warm per-call medians)** — legacy **0.63–1.02 ms** vs ETL/`numpy` **9.0–105.4 ms** vs
  ETL/`iree` **24.5–260.4 ms**; IREE's warm cost is ≈2.5–3.3× the numpy backend's, its per-structure
  compile 0.34–2.16 s versus ~3.2 ms for numpy — which is why compile stays out of the warm headline
  (legacy ≈33–148× the numpy backend and ≈110–363× the IREE backend on the fixture configs).
* **Where it lives** — `REPORT.md` (analysis), `r2/RUNLOG.md` (operator log of the executed round),
  `r2/backend_agreement.md` (pre-flight gate: IREE-cuda `O1` == `numpy` bit-identical on both fixtures),
  raw records under `new/results/r2/` + `legacy/results/r2/` (pre-flight records in
  `new/results/r2/backend_agreement/`).
* **Re-run / validate** — `test/benchmark/r2/run_all.sh` (`--group light|iree|all`, `--only <id>`,
  `--dry-run`, `--list`); `python3 test/benchmark/r2/check_r2.py --manifest
  test/benchmark/r2/manifest.txt --require-stages` (expect 35 records / 0 failures);
  `test/benchmark/r2/check_backend_agreement.py` on the GPU — once per backend, before quoting IREE
  numbers. Configs, caps, exit codes and wall-time expectations: `r2/README.md`.

## Contents

| path | what it is |
|---|---|
| `SPEC.md` | **Frozen methodology** (v1, 2026-09-11): workloads W1–W3, config matrix C0–C7, fairness rules, required JSON schema, priority order. Do not edit. |
| `SUMMARY.md` | **Round-1 headline report** — verdict, per-config old-vs-new table (all numbers recomputed from the raw JSONs), why the gap looks like this, caveats, correctness anchor + validator results, reproducibility. |
| `REPORT.md` | **R2 report** — warm/compile-separated old-vs-new comparison (legacy CUDA vs ETL `numpy`/`iree`), computed from `new/results/r2/` + `legacy/results/r2/`. |
| `bench_gp.py` | Side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; one JSON per run + per-generation/per-call timings, plus the R2 flags (`--backend`, `--data-rows`, `--max-cold-builds`, `--stage-probe`, `--mem-guard-mb`). |
| `stage_probe.py` | The `--stage-probe` module: runtime monkeypatching of the new engine's hot call sites into a cold-build / cache-hit / run / readback / upload split (self-test: `python3 test/benchmark/stage_probe.py`). |
| `r2/` | **R2 protocol**: `run_all.sh` (35-config runner), `manifest.txt` + `check_r2.py` (validator), `README.md`, `RUNLOG.md` (operator log), `backend_agreement.md` + `check_backend_agreement.py` (IREE-vs-numpy pre-flight). |
| `fixtures/` | The 200-tree cross-side fixture `forest200_tree64_random.npz` (same npz schema as `../goldens/forest_random.npz`) + provenance `README.md` + eval sanity records; generator `make_r2_fixtures.py`. |
| `new/` | NEW side: `run_matrix.sh` (round-1 config runner), `check_smoke.py` (JSON validator), `NOTES.md` (harness/schema, R2 flags, stage probe, smoke suite, matrix runs, deviations), `results/` (round-1 JSONs + logs) and `results/r2/` (R2 records). |
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
