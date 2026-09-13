# `test/benchmark/legacy/` — OLD side of the engine benchmark: pre-ETL custom-CUDA checkout (`b613503`)
## Intent
Legacy ("old") half of the performance comparison defined in `../SPEC.md`: the pre-ETL revision `b613503`
with the hand-written CUDA kernels bound as `evogp.evogp_cuda`, driven through the same public API as the
NEW (ETL) side to produce the config-matrix run records.
Everything here is hand-run scratch tooling with no CI and no pytest collection; `../SPEC.md` is the frozen
methodology (workloads W1-W3, configs C1-C5b, output schema) and `NOTES.md` is the carried side of the report.
## Status
Complete: all 6 legacy configs (C1 x2 reps, C2, C2b, C4 x2 reps, C5, C5b) ran on GPU 4 on 2026-09-11, 8 JSON
records + raw logs are committed, `check_results.py` validates every record, C5's golden cross-check PASSes.
A later consolidation pass consumes `NOTES.md` + `results/*.json` (it owns `../SUMMARY.md`/`../README.md`).
## Contents
- `run_matrix.py` -> the SPEC harness: `--configs C1,C2,C2b,C4,C5,C5b`, `--out`, `--fixture`, `--gpu`, `--commit`, `--list`.
- `check_results.py` -> stdlib JSON validator (SPEC keys, gens lengths, monotone best-fitness, `side=="old"`, env/GPU, median/first-call consistency) + optional `--golden` C5 cross-check; re-runnable while the lock is held.
- `diagnostic_details_overhead.py` -> one-off, non-SPEC: measures `StandardPipeline.show_details` overhead on the C2 workload (explains the `sr_test` smoke being ~10.9 ms/gen).
- `setup_legacy.sh` + `BUILD_NOTES.md` -> materialize/build/verify `/tmp/evogp_legacy_bench`; the build record and the `.so` identity (do not duplicate, just reference).
- `NOTES.md` -> the result report: headline table per config, C5 golden check, exact commands, deviations, observations.
- `results/*.json` -> 8 SPEC records (one per run); `results/c4_rerun/` -> the SPEC-authorized C4 re-run records.
- `results/logs/` -> raw lifecycle + run logs: build/idempotency/smoke from BUILD-era, `{C1,C2,C2b,C4,C5,C5b}.log` per-gen traces, `C4.rerun.log`, `check_results.log`, `diagnostic_details_overhead.log`, `cpu_micro_show_details.log`, `gpu_snapshots.txt`, `gpu7_ecc_evidence.txt`.
- `results/r2/` -> the R2 round's legacy records from the unified harness `../bench_gp.py`: the 11 `leg_*` ids listed in `../r2/manifest.txt` (`leg_c5r2`, `leg_c1r2`, `leg_c5b_r2`, `leg_fresh16`, `leg_sr_a`, `leg_sr_b`, and the eval64 fixture family `leg_fixture{64,200}_n{64,1024,10000}` where the `_n*` suffix is the `--data-rows` tiling), each with its raw log; plus the non-manifest sanity smoke `leg_smoke` and the frozen-driver fitness anchors `C5.json`/`C5b.json` from `run_matrix.py` — the only records here carrying `nan_count`/`fitness_mean` (the R2 directory scan reports them as prior-round artifacts) — and GPU snapshots. Details in `NOTES.md` §R2.
## How to run
Interpreter `/mnt/local-ssd/bchuang/evox/.venv/bin/python`; run from `cd /tmp/evogp_legacy_bench` with `PYTHONPATH=src:/tmp/evogp_t1a263/siteutils` (`src` first so the legacy `evogp` wins); pick a GPU and never use 7.
One `flock -w 7200 /tmp/gpbench_critical.lock bash -c '...'` per invocation (shared with the NEW side; never nest); always foreground with a generous `timeout` (backgrounded processes are killed with their cgroup); `--out` must be absolute because the CWD is `/tmp`.
Exact commands for the six configs, the re-run and the checks are in `NOTES.md` §5 (do not re-derive them).
## Notes for Agents
- The built legacy checkout lives at `/tmp/evogp_legacy_bench` (NOT in git, 6.4 MB, no `.git`); reuse it — rebuilding is only needed if the `.so` disappears (see `BUILD_NOTES.md` §9, `setup_legacy.sh` is idempotent).
- Extension identity to re-verify before trusting a run: `src/evogp/evogp_cuda.cpython-311-x86_64-linux-gnu.so`, 1 112 488 B, sha256 `d7af153c4d12843dc218e3616985173db9e294c3fc3a04f9dbab457b30d715a4`.
- GPU 7 is hardware-faulted (uncorrectable ECC) and kills every CUDA context — evidence in `results/logs/gpu7_ecc_evidence.txt`; the matrix ran on index 4 (snapshots in `gpu_snapshots.txt`).
- `results/logs/*.log` were committed with `git add -f` because the repo `.gitignore` ignores `*.log`; keep that in mind when adding new logs.
- The legacy engine's per-generation cost is dominated by host-side dispatch (~1.6 ms floor, ~7.5 ms extra with `is_show_details=True`), so numbers are only comparable within a quiet window; `NOTES.md` §4-§6 carry the caveats.
- The legacy checkout is `git archive` of `b613503c3107db23222f69bfaad2a329639f28fe` ("Update CONTEXT.md for ./src/evogp/problem", 2026-09-11 — last revision before the kernel-removal rewrite).
- Round-1 records come from `run_matrix.py` (this directory); the R2 and R3 legacy records are produced by the unified side-agnostic harness `../bench_gp.py` (R3 via `../r2/round3/run_all.sh`), whose `r2/round3/results/legacy/leg_c5r2.json` and `leg_fixture200_n64.json` hold the frozen 0.6323 / 0.6357 ms per-eval anchors quoted by the parent round reports.
- The legacy engine's `evaluate` is one batched kernel launch: `SymbolicRegression.evaluate` (`execute_mode="auto"`, the default) → `Forest.SR_fitness` → a single `torch.ops.evogp_cuda.tree_SR_fitness` op call over the whole population; the kernels are built with `-use_fast_math -maxrregcount=32` (the fast-math origin of the 1e-4 golden tolerance).
- `check_results.py` requires each JSON's file stem to equal its `tag` field — keep the driver fitness-anchor records named `C5`/`C5b` in `results/r2/`.
- Timed regions are host-median-of-warm generations/calls; `wall_s` includes construction and the cold first step, and `is_show_details` is always False in the records (SPEC).
## See Also
- `../SPEC.md` (frozen spec + config matrix), `../CONTEXT.md` (benchmark-area intent/status), `../new/` (sibling NEW-engine harness + results), `../bench_gp.py` (unified dual-checkout harness).
- `../../goldens/forest_random.npz` and `../../goldens/sr_fitness.npz` (frozen pre-ETL fixtures used by C5), `../../goldens/README.md`.
- `./BUILD_NOTES.md` (build + smoke record), `./NOTES.md` (results), `src/evogp/tree/ENGINE_SPEC.md` in the legacy checkout (legacy kernel semantics of record).
