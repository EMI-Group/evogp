# `test/benchmark/r5/` — RCA round: root cause of the ~10× old-vs-new evaluate gap (analysis-only)

## Intent
Definitive root-cause analysis (RCA) of why the new ETL "true jit" engine is ≈9.5–10.5× slower than the legacy CUDA engine on the headline benchmark.
Headline of record: type-1 end-to-end `problem.evaluate(forest)` on the dense route (iree, `EVOGP_EVAL_MODE=auto`): 6.667 ms (fixture64) / 6.050 ms (fixture200) warm medians vs legacy CUDA anchors 0.6323 / 0.6357 ms (cross-window quotes from R3 W2).
This round removes the cross-window doubt (same-window 3-way anchors), decomposes one warm call into ≥8 measured components, pins the GPU-side mechanism of the pure forward (1.447 / 1.593 ms), runs falsification/floor experiments, and ranks mechanisms with measured/inferred/unknown labels.
Analysis-only: NO behavior changes to `src/evogp`; NO writes in the foreign ETL repo (`/mnt/local-ssd/bchuang/etl`); NO edits outside `r5/` (r2/r4 records+manifests are immutable and gate-checked; shared tools are read/copied, never modified; `../REPORT.md` and the parent `../CONTEXT.md` are updated by the parent).
Deliverables: `RCA.md` + evidence records + stdlib-only gate `check_r5.py` + protocol `README.md` under this node.

## Status
- Round running; parent recon complete (env facts below); unfinished questions get marked open in `RCA.md` as phases land.
- Phase plan: P0 env freeze + Q4 window (first recorded phase) → Q1/Q5/Q7/Q8 decomposition → Q2 GPU mechanism → Q6 floors → Q3 A/B → RCA write-up + check_r5.py.
- Complete: Phase 0 — `results/env_probe.txt` (evogp commit, ETL `file`/HEAD/clean status, venv+torch, driver, nsys/ncu, lock/rlimit/env), `results/tool_recon/` (nsys import failure log + README), `window_probe.sh` (the single runner: 3 smoke cells + 9 window cells + the re-run control).
- Complete: Q4 same-window window (2026-09-13 20:29–20:41, GPU 6) — `results/smoke/`, `results/q4_window/` (9/9 cells exit 0 + committed `window_meta.txt` + `SUMMARY.md` + `rerun_1/` post-window control), `results/backend_agreement/` (six-arm pre-flight re-run, checker PASS 30 checks / 0 failures, PASS-with-note).
  Headline (same window, warm medians): legacy 0.6461 / 0.6372 ms (64 / 200 trees), dense-iree e2e 5.3823 / 5.7912 ms, pure forward 1.2142 / 1.4612 ms ⇒ **new/legacy 8.33× / 9.09×**, pure/legacy 1.88× / 2.29×, pure = 23–25 % of the e2e call; r4's cross-window 10.54× / 9.52× lies between the window pair and the control pair (11.04×), and the pure `output_checksum.sha256` is bit-equal to the r4 after-records. The plain e2e cell repeats to only +26.9 % while the legacy anchor repeats to −4.2 % and the probe cell to −0.7 % ⇒ the honest gap statement is **≈8–11×**, and the `--stage-probe` tax is nil (`probe_span − total_ms` = +0.028 ms/call; the window's +25.9 % was drift). The drift bracket's +177.5 % is a co-tenant contention event (90 % utilisation sample), not protocol drift.
- Complete: Q1/Q5/Q7 decomposition (`decomp_probe.py` + `probe_lib.py`) — records `results/q1_decomp/decomp_{f64,f200}.json|log` (25 warm calls + region-sync/plain/cold controls, per-call arrays, counts, stamps) and the analysis in `results/q1_decomp/SUMMARY.md`. Headline: exclusive per-call split reconciled to 99.94–99.95 %; the compiled `etl.run` (2.13 / 2.37 ms) and the 10 payload uploads (3.51 / 1.53 ms) dominate, then decode (0.70 / 1.38 ms) and host loss assembly (0.51 ms); readback is 0.12 ms; `build_cached` warm 0.08 ms (24 hits / 1 miss) against a ≈10.8–11.3 s cold rebuild; the GPU tail is drained inside `etl.run` (inline fence 0.031 ms).
- In flight: Phase 3 — run the committed Q8 (`minimal_loop.py`) + `--cprofile` passes first (records `results/q8_minimal/`, `results/q1_decomp/cprofile_f64.txt`), then Q2 (compile-side dispatch dumps, `--data-rows` row scaling, one kineto retry; records `results/q2_gpu/`).
- Open: Q3/Q6 probes (`ab_probe.py` / `floor_probe.py`) not written; `RCA.md` / `README.md` / `check_r5.py` pending.

## Routing Table
- `README.md` → protocol: Q1–Q8, method, run order, exact re-run commands, evidence map.
- `RCA.md` → THE deliverable: same-window tables; per-call decomposition; ranked mechanisms (measured/inferred/unknown); falsification results; "what would close what" floors.
- `window_probe.sh` → the executed Q4 same-window runner (one process/cell, one flock/cell with `timeout` inside, GPU 6, per-cell log with `etl:` identity + co-tenant samples, committed `window_meta.txt`): `--group smoke|window|all` (3 pre-window smoke cells / the 9 window cells: legacy eval64 ×2 + dense-iree eval64 ×2 + dense-iree pure ×2 + `--stage-probe` bridge + legacy bracket + 87c7562 shadow bridge), `--only id1,id2`, `--result-dir DIR` (fresh `rerun_<n>/` for any re-run), `DEVIATION_NOTE=…` to stamp protocol notes. Evidence: `results/{smoke,q4_window,q4_window/rerun_1}/` + `results/q4_window/SUMMARY.md`.
- `decomp_probe.py` → Q1/Q5/Q7/Q8 decomposition of one warm dense evaluate (monkeypatch timers over the dense call sites; cProfile cross-check; compile-cache counters).
- `floor_probe.py` → Q6 floor matrix (staged-reuse / no-readback / no-upload / pure-only).
- `ab_probe.py` → Q3 same-process engine-call vs frozen-runner A/B.
- `gpu_probe/` (or `nsys/`) → Q2 compile-side dispatch dumps + `--data-rows` row-scaling + kineto retry; profiler-blocked evidence.
- `results/` → evidence records (JSON/log/txt; env-stamped; immutable once recorded; re-runs go to fresh subdirs).
- `check_r5.py` → stdlib-only validator over the records (runs GPU-free with plain `python3`).

## Key constraints
- One process per config; exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per invocation (never nested), `timeout` INSIDE the lock; GPU 6 (never GPU 7); project venv; `PYTHONPATH=src`; repo-root CWD; `ulimit -n 65536`; `ETL_DISABLE_LOCATIONS=1`.
- Monkeypatching only (extend the `../stage_probe.py` pattern) — no `src/evogp` edits; if a src/etl change seems strictly necessary, record it as NEEDS-PARENT-ACTION instead.
- ≤2 attempts per tool then fall back + record (nsys/ncu are already blocked, see below — do not re-fight beyond one kineto retry and one plausible nsys variant).
- Every record stamps: evogp commit (`git rev-parse HEAD`), `etl.__file__` + ETL repo HEAD (read-only `git -C`), env knobs, `CUDA_VISIBLE_DEVICES`, co-tenant `nvidia-smi` sample, counts, timestamps.
- `/tmp` is 97 % full (≈16 GB free) — keep capture artifacts small, clean up scratch.

## Measurement identity (round decision)
- Primary cells measure the venv-default ETL — the read-only editable install of `/mnt/local-ssd/bchuang/etl` at HEAD `96df7da8f2ba39f52cf2f2b61888e67f70471bc9` (working tree clean at round start) — stamped per record via `etl.__file__` + `git -C /mnt/local-ssd/bchuang/etl rev-parse HEAD`.
- Current main contains the r4-after shadow commit `87c7562` plus later jit changes (per-wrapper spec memo `4c0fb4f`, run-static fast path), so the r4 headline quotes (6.667 / 6.050 ms; pure 1.447 / 1.593 ms) are cross-window context only — r5's own same-window table is the anchor.
- `/tmp/evogp_etl_patched_87c7562` is kept only as an optional one-cell shadow bridge (`NEW_PYTHONPATH="src:/tmp/evogp_etl_patched_87c7562"`), never as the primary environment.
- Q4 IREE numbers are gated by re-running the full six-arm r4 agreement pre-flight into `results/backend_agreement/` (checker `check_agreement_r4.py --agree-dir …`; a missing stem is a hard failure, so all six arms are run).

## Notes for Agents (parent recon facts)
- Legacy engine intact: `/tmp/evogp_legacy_bench` (built `.so`, py3.11); import requires `import torch` FIRST (libc10); legacy PYTHONPATH = `/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`; legacy invocations `bench_gp.py --task eval64 [--fixture …] --eval-reps 25`.
- nsys 2022.4.2.1 (target-only install): collection works, but QDSTRM import/analysis FAILS against driver 535.247.01 / CUDA 12.4 — `Unknown runtime API function index: 440`, kernel tables come out empty; host toolchain `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` exists but shares the analysis code. Evidence: `/tmp/evogp_r5recon/import_log.txt` (copy into `results/tool_recon/`).
- ncu 2022.3: `CUDA error: API call is not supported in the installed CUDA driver` → no kernels profiled.
- torch.profiler (kineto) DOES capture torch kernels on this box (cutlass sgemm + elementwise visible); ONE cheap retry may test whether it also captures iree kernels (would unblock the Q2 timeline via a chrome trace).
- GPU 6 co-tenants: 3× `syfang` training procs (~30 GB each, ~135 % CPU, 0 %→burst util); ≈18 GB VRAM free; record a sample next to every window.
- fixture64 = `test/goldens/forest_random.npz`; fixture200 = `test/benchmark/fixtures/forest200_tree64_random.npz`.
- Exact cell invocations: new side `bench_gp.py --task eval64 [--fixture <f200>] --eval-reps 25 --mem-guard-mb 8192` with `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`; legacy mirror `--task eval64 [--fixture <f200>] --eval-reps 25` on LEG_PYTHONPATH; pure cells (from `../r4/run_pure.sh`): `--task eval-pure --fixture <f> --mode dense --backend iree --reps 5 --warmup 10 --invocations 100 --e2e-companion 11 --mem-guard-mb 8192 --max-cold-builds 400` with `EVOGP_EVAL_MODE=dense`.
- Dense wrap points in `../../src/evogp/tree/dense_engine.py`: `prepare_dense` L824, `_decode_columns` L932, `plan_dense` L1373, `build_dense` L1522, `_prep_column` L1600, `_prep_columns` L1609, `run_plan` L1615, `_to_host` L1654, `_iter_plan_chunks` L1661, `run_loss_plan` L1784 (payload helper L1734), `_input_to_numpy` L1813, `evaluate_dense` L1828, `sr_fitness_dense` L1924, `DenseRunner` L2029 / `.run()` L2146; plus `etl.build_cached` / `etl.run` / `etl.from_numpy` as seen from the engine path.
- iree compile-dump flags confirmed: `--iree-flow-dump-dispatch-graph[=…,-output-file=]`, `--iree-scheduling-dump-statistics-file=<path|''|stderr|->` (+`-format`), `--iree-hal-dump-executable-*`, `--dump-compilation-phases-to` (passed via `ETL_IREE_COMPILE_ARGS`).
- Known facts to build on: pure forward 1.447 / 1.593 ms (`../r4/results/after/pure{64,200}_dense_iree.json`); in-engine ≈2.0 ms inside the single `etl.run`, ≈4.6 ms host glue outside (“etl.run ≈1/3, host ≈2/3”); dense route flat in population; specialized route fully explained — do NOT re-investigate it.
- Legacy anchors (read-only): `../r2/round3/results/legacy/leg_c5r2.json`, `leg_fixture200_n64.json`; invocation template `../r2/round3/run_all.sh` (LEG_PYTHONPATH defaults there); IREE agreement gate: `../r4/run_agreement_r4.sh --agree-dir …` + `check_agreement_r4.py --agree-dir …`.
- `.gitignore` precedent: `../r4/.gitignore` (ignore `last_run.txt`; re-include `results/**/*.log`).

## See Also
- Parent `../CONTEXT.md` (round verdicts, harness rules). `../REPORT.md` (analysis of record; §8 levers — parent adds the RCA pointer).
- `../bench_gp.py` + `../pure_forward.py` (harness + frozen pure runner), `../stage_probe.py` (monkeypatch pattern), `../r4/` (protocol + after-records), `../r2/round3/` (legacy records + Arm A/B).
