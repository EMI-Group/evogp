# `test/benchmark/r6/` — Round 6: the Triton fused-kernel evaluator prototype (feasibility + measurement)

Round 6 tests the r5 RCA hypothesis that a *fused* kernel replicating the legacy CUDA engine's design
restores legacy-class per-call cost for EvoGP's single-output SR case.  It is a prototype round, and the
prototype lives in this directory only: no `src/` change, no foreign-repo write, no edit to the committed
r2/r4/r5 records or the shared harnesses.  The Triton evaluator consumes the same staged population the
dense route consumes (`dense_engine.prepare_dense`) and is validated numerically against the plain-torch
oracle + the frozen legacy goldens before it is timed.  **The round is complete**: one continuous
same-window campaign (10 cells) plus the P1 extras, the numeric validation (run 1 + a replicate), the
stdlib-only gate `check_r6.py` → **14 records / 1440 checks / 0 failures**, and the analysis of record
[`SUMMARY.md`](SUMMARY.md).

**Headline** (primary window `w1d_20260914`, warm type-1 e2e medians, ms, f64/f200): triton
**0.1314 / 0.1379** vs legacy **0.6364 / 0.6269** vs dense-iree **4.7860 / 6.3179** — 36.4× / 45.8×
faster than the dense route it replaces and 4.8× / 4.6× *faster* than the legacy engine; the verdict, the
per-call (kernel-vs-host) decomposition and the Q1–Q5 answers are in [`SUMMARY.md`](SUMMARY.md).

## Contents

| file | what it is |
|---|---|
| `triton_eval.py` | the Triton fused evaluator prototype: `build_evaluator(...)` (build-once/run-many, resident staging, one `_sr_eval_kernel` launch per call), `sr_fitness_triton` and the `install_dense_seam()` type-1 envelope |
| `run_triton_eval64.py` | the cell driver: type-1 e2e (patched dense seam + unmodified public API) and `--mode pure` type-2 loop; records the counters (`triton_call_count` / `triton_runs` / `builds` / `cache_hits` / `fallbacks`) and enforces the counter gate (nonzero exit if a timed call could have silently fallen back) |
| `validate_triton.py` | the numeric validation: fixture64/fixture200 vs `src/evogp/tests/reference_eval.py`, the 12 legacy-golden arrays, the 31-op edge-grid table and the edge-case sweep (24 strict + 6 tolerance-only); writes `results/validation/**` |
| `run_cells.sh` | the same-window campaign runner: cell table `check|legacy|dense|triton|pure|all`, one fresh process per cell, one `fcntl` lock + `timeout` inside the lock, `ulimit -n 65536`, freshness gate (one campaign write per window label) |
| `run_under_lock.py` | the `fcntl.flock` lock wrapper — the `flock(1)` **binary hangs in this sandbox**, so every GPU job is serialized through this instead |
| `extra_trace.py` | the P1 extras probe: kineto kernel-vs-host decomposition (`--target triton-pure|triton-e2e|legacy-e2e`), leaf-kernel/residual extraction; the compile-economics records are the driver re-run with a warm vs a fresh `TRITON_CACHE_DIR` |
| `check_r6.py` | stdlib-only, GPU-free gate over every record + the RUNLOG → **14 records / 1440 checks / 0 failures** (`--verbose` adds the per-cell number table) |
| `results/**` | immutable evidence: `w1d_20260914/` (primary window: 10 cells + `extra/{kernel_host,compile_econ}`), `w1c_20260914/` (superseded anomaly window), `validation/` (run 1 + `run2/`) |
| `RUNLOG.md` | the operator log: window identity, verbatim commands, per-cell numbers, anomalies/deviations, §8 = the run-2 validation revision caveat, immutability statement |
| `SUMMARY.md` | analysis of record: the verdict, the per-call decomposition and Q1–Q5 |
| `CONTEXT.md` | the node's context/routing file |

## Repro commands (verified against the real CLIs)

Repo root as CWD, project venv (no `python`/`python3` with torch on `PATH`), CUDA-visible GPU 6, one
continuous session; `--dry-run`/`--list` create nothing.  House shape for a manual GPU cell: one
`run_under_lock.py` per command with `timeout` **inside** the lock (never nested).

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python        # py3.11.2, torch 2.6.0+cu124, triton 3.2.0

# 0. gate (stdlib-only, GPU-free): recomputes every headline from the committed records
python3 test/benchmark/r6/check_r6.py                  # -> "check_r6: 14 records / 1440 checks / 0 failures", exit 0
python3 test/benchmark/r6/check_r6.py --verbose        # + the per-cell number table

# 1. the campaign (GPU 6; one fresh process per cell; the 10 cells took ~2 min in the round)
bash test/benchmark/r6/run_cells.sh --list                            # offline: the cell table + per-cell args/env
bash test/benchmark/r6/run_cells.sh --dry-run --group all             # offline: prints every command, creates nothing
bash test/benchmark/r6/run_cells.sh --group check --window <NEW_WINDOW>   # deps pre-flight -> results/<W>/check_deps.txt
bash test/benchmark/r6/run_cells.sh --group all   --window <NEW_WINDOW>   # the 10 measurement cells
#   also: --group legacy|dense|triton|pure, --only id1,id2 (overrides --group), --timeout S
#   a re-run needs a NEW --window label: the runner refuses to write into a window dir that holds records

# 2. one manual type-1 cell, verbatim as the window ran it (triton_f64; fixture200 = --fixture test/benchmark/fixtures/forest200_tree64_random.npz)
$PY test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
  env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
      TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache R6_WINDOW=<W> R6_CELL_ID=triton_f64 \
      CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
  $PY test/benchmark/r6/run_triton_eval64.py --fixture test/goldens/forest_random.npz \
      --reps 25 --tag TRITON_F64 --json-out test/benchmark/r6/results/<W>/triton_f64.json
#   type-2 pure instead of type-1: add --mode pure --warmup 10 --invocations 100 --reps 5 --e2e-companion 11

# 3. the numeric validation (GPU; ~10-11 s per run).  run 1 wrote results/validation/; a non-empty --out-dir
#    automatically gets a fresh runN/ (the replicate landed in results/validation/run2/) -- nothing is overwritten.
$PY test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
  env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
      TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache \
      CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
  $PY test/benchmark/r6/validate_triton.py --out-dir test/benchmark/r6/results/validation

# 4. one P1 extra: the kineto kernel-vs-host decomposition (triton pure, f64), as the window ran it
$PY test/benchmark/r6/run_under_lock.py timeout -s INT -k 30 900 \
  env PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 \
      TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache \
      CPATH=/tmp/triton_feas/py311hdr/usr/include/python3.11:/tmp/triton_feas/py311hdr/usr/include \
  $PY test/benchmark/r6/extra_trace.py --target triton-pure \
      --fixture test/goldens/forest_random.npz --invocations 100 \
      --out-dir test/benchmark/r6/results/<NEW_WINDOW>/extra/kernel_host/triton_pure_f64 \
      --json-out test/benchmark/r6/results/<NEW_WINDOW>/extra/kernel_host/triton_pure_f64/facts.json \
      --tag EXTRA_TRITON_PURE_F64
#   other targets: --target triton-e2e|legacy-e2e (fixture200: test/benchmark/fixtures/forest200_tree64_random.npz)
#   compile economics: the same driver as (2) with --tag EXTRA_TRITON_WARMCACHE / EXTRA_TRITON_FRESHCACHE,
#   writing into .../extra/compile_econ/{warm_cache,fresh_cache}/, cold = TRITON_CACHE_DIR=$TMPDIR/r6_fresh_cache
```

## Session-scoped environment (verify, don't assume)

* **Interpreter.** `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2, torch 2.6.0+cu124, triton
  3.2.0 with its own `ptxas`).  `python`/`python3` on `PATH` have no torch; only the stdlib gate is safe
  there.  Every evogp / triton entry point runs from the repo root with `PYTHONPATH=src`.
* **Triton `Python.h` fix.**  Runtime compilation needs `Python.h`, missing at `/usr/include/python3.11`.
  A ready header tree exists at `/tmp/triton_feas/py311hdr`; rebuild it (~5 s, no root) by downloading the
  `libpython3.11-dev` deb into `$TMPDIR`, `dpkg-deb -x`-ing it, then
  `export CPATH="$TMPDIR/py311hdr/usr/include/python3.11:$TMPDIR/py311hdr/usr/include"` — **both** entries
  are required.
* **Triton cache.**  `TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache` (reproducible compile caching, kept out
  of the repo).  A warm *disk* cache does not make the first call free: ≈0.71–0.73 s JIT per fresh process,
  ≈1.79 s from an empty cache (the extras measured both).
* **GPU policy.**  `CUDA_VISIBLE_DEVICES=6`; **never GPU 7** (hardware ECC fault — any CUDA context on it
  dies).  Check `nvidia-smi` before every timed run; a *memory-resident* co-tenant on GPU 6 is the normal
  state of this box (never kill foreign processes — measure and note the contention).
* **Lock policy.**  Serialize every GPU phase with the Python `fcntl.flock` wrapper
  (`run_under_lock.py`, lock file `/tmp/gpbench_critical.lock`, shared with r4/r5 jobs): exactly one lock
  per command, never nested, `timeout -s INT -k 30 <cap>` inside the lock.  The `flock(1)` binary hangs in
  this sandbox.
* **Legacy side (read-only).**  `/tmp/evogp_legacy_bench` (the `b613503` build, with
  `src/evogp/evogp_cuda.cpython-311-x86_64-linux-gnu.so` present) + `/tmp/evogp_t1a263/siteutils`; the
  legacy cells get the legacy `PYTHONPATH` and **no**
  `EVOGP_*`/`ETL_*` variable at all (inert there).  Rebuild recipe: `../legacy/BUILD_NOTES.md` /
  `../legacy/setup_legacy.sh`.
* **Multi-minute phases.**  Launch them with
  `systemd-run --user --unit=<name> --collect -p LimitNOFILE=65536 --setenv=CUDA_VISIBLE_DEVICES=6 bash -c '…' "$PWD"`:
  `nohup`/`setsid`/`disown` backgrounding inside a tool call dies with the call's `evogit.slice` cgroup
  (two campaign attempts were reaped mid-flight; see `RUNLOG.md` §6).  `/tmp` sits near full (≈15 GB free)
  — keep outputs small.

## House rules (immutability)

* `results/w1d_20260914/**`, `results/w1c_20260914/**` and `results/validation/**` are **immutable
  evidence**: a re-run writes into a fresh `results/<NEW_WINDOW>/` (or a fresh `runN/`), never in place;
  `run_cells.sh` enforces one campaign write per window label.  `last_run.txt` is generated/disposable
  (gitignored).
* Never edit `../SPEC.md`, the committed r2/r4/r5 records or harnesses, `../bench_gp.py`,
  `../pure_forward.py`, `../fixtures/`, `../../goldens/`, anything under `src/`, or the foreign repos; a new
  round gets its own copies.
* Validation provenance caveat: the replicate (`results/validation/run2/validation.json`) declares an
  **uncommitted** `triton_eval.py` sha256 (`e155dd13…`, in no git object); run 1 pins the committed file
  (`163b5ffd…`) and is the record of record.  `check_r6.py` asserts exactly that (`provenance:run2_*`) and
  prints it as a NOTE — see `RUNLOG.md` §8.
* Quote medians with the ±10 % shared-box band (≈2–3× inflation under heavy contention); keep files
  <~1000 lines and scratch data in `$TMPDIR`.

## See also

* `SUMMARY.md` — the round's verdict + per-call decomposition; `RUNLOG.md` — the operator log of record.
* `../r5/RCA.md` — the gap analysis this round tests; `../r4/README.md` — the type-1/type-2 protocol and
  the record layouts r6 mirrors; `../CONTEXT.md` — the benchmark node (harness rules, lock/GPU policy).
* `../../src/evogp/tree/ENGINE_SPEC.md` — engine semantics of record; `../../src/evogp/tests/reference_eval.py`
  — the plain-torch oracle; `../../goldens/` (the `forest_random.npz` fixture64 source) + `../fixtures/`
  (`forest200_tree64_random.npz`) — the two fixture sources.
