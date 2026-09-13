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
- In flight: Phase 0 + Q4 — env-freeze files, the window cells (leg f64/f200, dense-iree eval64 f64/f200, pure f64/f200, +stage-probe bridge, legacy drift bracket, optional 87c7562 shadow bridge) and the six-arm agreement pre-flight into `results/backend_agreement/`.
- Open: Q1–Q3, Q5–Q8 analyses; RCA.md / README.md / check_r5.py pending.

## Routing Table
- `README.md` → protocol: Q1–Q8, method, run order, exact re-run commands, evidence map.
- `RCA.md` → THE deliverable: same-window tables; per-call decomposition; ranked mechanisms (measured/inferred/unknown); falsification results; "what would close what" floors.
- `window_probe.sh` → Q4 same-window anchors (legacy eval64 ×2 + dense-iree eval64 ×2 + dense-iree pure ×2, one serialized window).
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
