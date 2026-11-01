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
`xval/` (summarized in `SUMMARY.md` §5.1). That is the **round-1** headline; the numbers of record are the
**Round 4** section below (fixture-eval gap ≈9.5–10.5×, down from R2's 33–363× and R3's ≈16.5–16.8×).

## Round 4 (GPU, current verdict — the pure-forward round)

Round 4 (`r4/`) adds a **second measurement type** next to the end-to-end numbers R1–R3 quote: the **pure
internal graph forward** — the compiled forward pass in a tight loop on a fixed device-resident `x`, with no
readback, no host assembly and no per-call key derivation (harness task `--task eval-pure`, runner core
`pure_forward.py`; the same process also runs the real `problem.evaluate(forest)` via `--e2e-companion K`, so
`E2E − pure` prices the host glue on one box in one window). The round is recorded in **both halves**: the
pre-code-round **before**-baselines at `evogp e272714` / etl `51a7141` (every record `local-fallback:*`) and
the **after**-round that re-ran the identical 8-config protocol at `evogp 6e8bf6f` on a **read-only shadow
export of etl `87c7562`** (every record `frozen-api:*`), plus a 14-cell type-1 (E2E) matrix. Analysis of
record: `r4/SUMMARY.md`; protocol/flag/JSON glossary: `r4/README.md`; operator log: `r4/RUNLOG.md`.

* **Pure (type 2) — before → after: every route faster** (`median_per_invocation_ms`, one full-population
  forward; fixture64 / fixture200): numpy dense 128.936 → **75.230** / 181.686 → **96.717** ms; numpy
  specialized 8.390 → **6.616** / 49.332 → **30.360** ms; iree dense 1.587 → **1.447** / 3.696 → **1.593** ms;
  iree specialized 53.214 → **5.922** / 214.655 → **18.570** ms (−8.8 % to −91.3 %). The specialized-iree
  collapse is the per-`etl.run` cost (1.0040 → 0.1117 ms; 1.2553 → 0.1086 ms), and the provenance flip is
  complete and numerically inert: all 8 records `frozen-api:*` / `compile.path="frozen"` /
  `n_graph_calls_source="introspected"` (1 / 53 / 171 calls), with the frozen-vs-fallback A/B exact
  (`max|diff| = 0`, byte-identical captures).
* **Type 1 (E2E) after — the 14-cell matrix** (warm medians, fixture64 / fixture200): iree `auto`(dense)
  **6.667 / 6.050 ms** under the shadow and 6.556 / 6.461 ms on the **same-window stock twins** ⇒ the patched
  ETL is **warm-neutral** (+1.7 % / −6.4 %); iree `dynamic` 7.398 / 6.348 ms; iree `specialized`
  59.554 / 193.086 ms; numpy `auto` 10.844 / 52.891 ms, `dynamic` 71.373 / 85.244 ms, `specialized`
  10.853 / 54.122 ms. Against the frozen legacy anchor (0.6323 / 0.6357 ms) the old-vs-new gap is
  **≈9.5–10.5×**, down from R3's ≈16.5–16.8×, and the stock twin attributes the gain to the evogp side. iree
  first call ≈10.1–10.6 s (the one-time dense compile, reported separately).
* **Gates** — `check_r4.py --manifest test/benchmark/r4/manifest_after.txt --require-companion` → 8 complete /
  369 checks / 0 failures (the before manifest gates the same 8 / 369 / 0); `check_e2e_r4.py` → 14 / 328 / 0;
  `check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after` → PASS-with-note
  (30 checks — numpy == iree-specialized bit-identical, iree-dense within 1e-4).
* **Where it lives / re-run** — records `r4/results/**` (before) and `r4/results/after/**` (the 8 pure
  records + `e2e/` + the frozen-vs-fallback `ab/` + the shadow `etl_import_probe.txt`), both immutable.
  Re-run with `test/benchmark/r4/run_pure.sh` (type 2) and `test/benchmark/r4/run_e2e.sh` (type 1), each after
  the IREE pre-flight `test/benchmark/r4/run_agreement_r4.sh` +
  `<venv>/bin/python test/benchmark/r4/check_agreement_r4.py`; the after window additionally needs
  `NEW_PYTHONPATH="src:/tmp/evogp_etl_patched_87c7562"` (`git -C /mnt/local-ssd/bchuang/etl archive 87c7562 |
  tar -x -C <dir>` — never write in the foreign repo) plus `NEW_RESULT_DIR=test/benchmark/r4/results/after`.
* **Legacy side** — has no separable pure forward (its `evaluate` is one fused kernel including the readback
  and the host assembly): `--task eval-pure` skips it by design (exit 3, `skipped: true`), so R4 has **no
  old-vs-new pure row**.

## Round 3 (GPU, previous round's verdict)

Round 3 (`r2/round3/`) re-measures the R2 headline configs with the **R2 method unchanged** after two waves
of engine work — host-side cuts in `src/evogp/tree/etl_engine.py` (wave 1) and the dense one-graph evaluator
`src/evogp/tree/dense_engine.py`, which under the default `EVOGP_EVAL_MODE=auto` policy is now the route the
**`iree`** backend takes (wave 2) — and does it in **two arms**: **Arm A** on merged `evogp` + stock ETL, and
**Arm B** on the same nine new-side configs under a **patched-ETL checkout shadow**
(`NEW_PYTHONPATH=src:<patched-etl-checkout>`, etl commit `51a7141`). The headline policy is unchanged:
**warm per-call (or per-generation) medians**, one process per config, compile/first-call reported
separately, medians ±10 % on this shared box, and both arms read in **two windows** (W1 = GPU 6 contended,
W2 = the GPU-0 companion), so within-window ratios are the robust ones.

* **Headline (warm medians; legacy ÷ ETL over fixture64 / fixture200 / `sr_a` rep1)** — R3 read the legacy
  engine as **≈16–23×** ahead of the default (`iree` dense) route, **≈36–142×** ahead of the `numpy` backend and
  **≈94–372×** ahead of the forced-specialized route (R2: 110–363× iree, 33–148× numpy). Arm A reads
  0.6323 ms vs **10.64 / 23.29 ms** (fixture64) and 0.6357 vs **10.50 / 117.61 ms** (fixture200); Arm B
  (W2 records of record) reads **14.84 / 10.03 ms** iree-dense and 22.81 / 90.15 ms numpy. The compression is
  the dense route — **6.38× / 21.77×** cheaper per warm eval call than the same code pinned to R2's
  per-structure route, which still reads R2's numbers — while wave 1 and the patched ETL measured ≈neutral
  warm (their observable is fd economy: iree flat at 91 fds versus 2 395 / 5 819 / 9 723 / 17 243).
* **Where it lives** — the verdict is the **"Round 3" section of `REPORT.md`** (now behind the Round-4
  verdict, ahead of the R2 sections); the detailed arm write-ups are `r2/round3/SUMMARY.md` (Arm A) and
  `r2/round3/SUMMARY_armB.md` (Arm B), with protocol, config table, commands and exit codes in
  `r2/round3/README.md`; records under `r2/round3/results/{new,legacy,backend_agreement,armB}/`.
* **Gates** — Arm A: `check_r3.py --manifest test/benchmark/r2/round3/manifest.txt --require-stages` → 45
  checks / 0 failures and `check_agreement_r3.py` → PASS (32 checks); Arm B: `check_r3.py --manifest
  test/benchmark/r2/round3/manifest_armB.txt --require-stages` → 34 checks / 0 failures and
  `check_agreement_armB.py` → PASS (38 checks, all six Arm A→B `values_sha256` pairs bit-identical). Runners:
  `test/benchmark/r2/round3/run_all.sh` (Arm A) and `test/benchmark/r2/round3/run_all_armB.sh` (Arm B — needs
  the patched ETL checkout materialized read-only and shadowed onto `NEW_PYTHONPATH`), each after its own
  `run_agreement_*.sh` pre-flight.
* **`iree_sr_b2`** — R2's one crash config (pop 1000 / tree 128, exit 139 SIGSEGV) was re-attempted on the
  patched checkout and **did not reproduce**: both routes exit 0 (dense 431/1500 cold-build misses in 213 s,
  pinned-specialized 1181/1500 in 595 s, fds flat at 91). The b2 ids stay outside the frozen manifests, and
  this is a "did not reproduce" datapoint, not proof of the original cause.

## R2 round (GPU, baseline)

Round 2 (`r2/`) is the baseline round: the same side-agnostic harness on the same box, comparing the legacy
CUDA engine against **both** ETL backends (default `numpy` CPU and opt-in IREE-cuda GPU), with
**warm/steady-state per-call numbers as the headline and per-structure compilation reported separately**
(the "Headline result" at the top of this file is round 1; the round-1 methodology in `SPEC.md` is
unchanged).

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
| `REPORT.md` | **Analysis of record** — leads with the **Round 4 verdict** (type-2 pure before/after, the 14-cell type-1 E2E after matrix, gates, residual blockers), then the Round-3 verdict (R2 → Arm A / Arm B, both engines) and the R2 baseline: warm/compile-separated old-vs-new comparison (legacy CUDA vs ETL `numpy`/`iree`), computed from `new/results/r2/` + `legacy/results/r2/`. |
| `bench_gp.py` | Side-agnostic harness: drives the same public API against either engine, selected by `PYTHONPATH`; one JSON per run + per-generation/per-call timings, the R2 flags (`--backend`, `--data-rows`, `--max-cold-builds`, `--stage-probe`, `--mem-guard-mb`) and the R4 pure-forward task `--task eval-pure` (`--mode`, `--invocations`, `--warmup`, `--e2e-companion`, `--pure-runner`; runner core `pure_forward.py`). |
| `stage_probe.py` | The `--stage-probe` module: runtime monkeypatching of the new engine's hot call sites into a cold-build / cache-hit / run / readback / upload split (self-test: `python3 test/benchmark/stage_probe.py`). |
| `r2/` | **R2 protocol + the executed round 3**: `run_all.sh` (35-config runner), `manifest.txt` + `check_r2.py` (validator), `README.md`, `RUNLOG.md` (operator log), `backend_agreement.md` + `check_backend_agreement.py` (IREE-vs-numpy pre-flight), and the `round3/` sub-node (round 3 Arm A + Arm B: runners, manifests, checkers, records, `SUMMARY.md` / `SUMMARY_armB.md`). |
| `r4/` | **Round 4 — the pure-forward round (before + after complete)**: `run_pure.sh` (8-config type-2 matrix) + `manifest.txt` / `manifest_after.txt` + `check_r4.py`, `run_e2e.sh` + `e2e_manifest.txt` + `check_e2e_r4.py` (the 14-cell type-1 matrix), `run_agreement_r4.sh` + `check_agreement_r4.py` (IREE pre-flight + gate), `README.md` (protocol, flag/JSON glossary, frozen-vs-fallback discipline), `SUMMARY.md` (analysis of record + verdict), `RUNLOG.md`, records in `results/` (before) + `results/after/` (pure + `e2e/` + `ab/` + the shadow probe) + `results/backend_agreement{,_after}/` + `results/smoke/`. |
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

Round 4 (the current round) re-runs from **this** repository root — new side only, both measurement types,
after the IREE agreement pre-flight:

```bash
test/benchmark/r4/run_pure.sh --list               # config table; --dry-run prints every exact command
test/benchmark/r4/run_pure.sh --group numpy        # type 2; then --group iree, or no flag for all 8
test/benchmark/r4/run_e2e.sh                       # type 1; 14 cells (--group shadow|stock|all, --only <id>)
# IREE quotability pre-flight (GPU) + gates (GPU-free):
bash test/benchmark/r4/run_agreement_r4.sh && <venv>/bin/python test/benchmark/r4/check_agreement_r4.py
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
python3 test/benchmark/r4/check_e2e_r4.py
# after-half re-run (the frozen-api side of the comparison): the read-only ETL shadow export of 87c7562
#   NEW_PYTHONPATH="src:/tmp/evogp_etl_patched_87c7562" NEW_RESULT_DIR=test/benchmark/r4/results/after
#   SHADOW_DIR=/tmp/evogp_etl_patched_87c7562   # for run_e2e.sh — never write inside the foreign repo
```

**Shared host:** the box has other tenants on the A6000s, and both sides share one advisory lock —
serialize every heavy phase (builds, timed runs) with exactly one, never nested:
`flock -w 7200 /tmp/gpbench_critical.lock bash -c '<cmd>'` (the wait is outside all timed regions).
Treat medians as ±10%; pick the least-busy GPU at run time and skip GPU 7 (hardware ECC fault). Full
environment, seeds, launcher and per-config commands: `SUMMARY.md` §6, `new/NOTES.md`, `legacy/NOTES.md`.
