# `test/benchmark/xval/` — cross-validation of the OLD-side benchmark numbers through the unified harness

## Intent
Wave-2 evidence directory: the two headline **old-side** configs (**C1** = W1 synthetic-SR training,
**C5** = W3 fixed-fixture eval microbench) re-run through the *unified* harness
`../bench_gp.py` (instead of the wave-1 old-side driver `../legacy/run_matrix.py`), to show the wave-1
numbers are harness-independent and to document where the two drivers' metrics are comparable.
Scope: data + two small probe helpers only. It does not supersede the wave-1 records; its results are
summarized in the cross-validation subsection of `../SUMMARY.md` (§5.1).

## API Surface / Contents
- `C1_xval.json` + `C1_xval.log`, `C5_xval.json` + `C5_xval.log` — harness outputs; logs carry the exact
  command, the `nvidia-smi` snapshot and the GPU pick (`CUDA_VISIBLE_DEVICES=0`).
- `C5_xval_fitness.json` — `nan_count` / nan-aware `fitness_mean` for C5 (the harness JSON schema does not
  record `evaluate`'s return values), produced by `c5_fitness_stats.py`.
- `c5_fitness_stats.py` — xval helper: loads `../bench_gp.py` via `importlib` and reuses its
  `apply_env` / `import_modules` / `build_eval_case` verbatim, then applies wave-1's fitness-sanity
  semantics (last call, MAXIMIZED fitness, `nan_count=isnan(f).sum()`, `fitness_mean=nanmean(f)` f64 on CPU).
- `legacy_driver_scope_probe.py` — xval helper: imports `../legacy/run_matrix.py` and times its
  construction pieces and a replication of its `run_eval_config` window (`--scope-only --no-presync`
  mimics the driver's fresh-process path).
- `legacy_driver_repro.log` + `repro_legacy_driver/C5.json` — A/B re-run of wave-1's driver today.

## Constraints / How to run
- OLD side only: `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils` with
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python`; the C5 fixture must be passed as an **absolute** path
  (e.g. `/mnt/local-ssd/bchuang/evogp/test/goldens/forest_random.npz`).
- Exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per run (never nested); GPU 7 is unusable (ECC
  fault) and the GPU pick rule is "lowest utilization %, tie-break lowest used memory, then lowest index".
- Helpers import harness/driver modules read-only; no wave-1 artifact outside this directory is written.

## Known Issues / Notes for Agents
- `wall_s` is **not** comparable between `../bench_gp.py` (device warmed during `gpu_snapshot`/pre-wall
  `synchronize`, context cost excluded) and `../legacy/run_matrix.py` (first `torch.tensor(device="cuda")`
  inside its wall window, so ~0.15 s of one-time CUDA context creation lands in C5's `wall_s`). Per-call
  metrics and fitness stats are unaffected; use those for cross-driver comparisons.
- `evogp_commit` in the JSONs is the harness repo's HEAD (`git rev-parse HEAD` in the cwd), not the legacy
  checkout's `b613503` — expected, the legacy checkout is not a git repo.
- Verified result (2026-09-11, GPU 0, shared box): C1 `wall_s` Δ −0.01 %, C1 warm-median Δ +5.8 %, C1
  `best_fitness_final` identical; C5 `first_call_ms` Δ +1.9 %, warm median Δ +3.9 %, `nan_count`=3 and
  `fitness_mean`=−18.048504493275626 identical to wave-1. Verdict, deltas and limitations: `NOTES.md`.

## See Also
- `../CONTEXT.md` (benchmark node: SPEC, wave-1 side directories), `NOTES.md` (this node's full report).
- `../legacy/results/` and `../new/results/` (wave-1 raw runs, read-only from here), `../SPEC.md`.
