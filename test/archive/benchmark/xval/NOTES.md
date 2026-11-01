# `test/benchmark/xval/` — wave-2 cross-validation of the OLD-side numbers through the unified harness

Purpose: re-run the two headline old-side configs (**C1** = W1 synthetic SR, **C5** = W3 fixed-fixture
eval microbench) through the *unified* harness `test/benchmark/bench_gp.py` instead of the wave-1
old-side driver `test/benchmark/legacy/run_matrix.py`, to show the wave-1 numbers are reproducible
across harnesses / methodology-uniform, and to document where the two drivers' metrics are (and are
not) comparable. **Nothing here replaces wave-1**; `SUMMARY.md` / `README.md` are written elsewhere.

## Environment
- OLD side = built pre-ETL checkout `/tmp/evogp_legacy_bench` (git archive of `b613503` +
  `evogp_cuda.cpython-311-x86_64-linux-gnu.so`; not a git repo).
- `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`, interpreter
  `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (3.11.2, torch 2.6.0+cu124).
- Relative CWD for both runs: this worktree root (harness reads/writes its paths cwd-relative).
- **GPU pick: index 0.** Snapshot at run time (`nvidia-smi --query-gpu=index,name,utilization.gpu,
  memory.used,...`): idx0 0 %/1119 MiB, idx1 0 %/1119 MiB, idx2 0–14 %/17702 MiB, idx3 55–83 %/16488 MiB,
  idx4 57–83 %/15373 MiB, idx5 0 %/24399 MiB, idx6 92–99 %/4933 MiB, idx7 0 %/1 MiB (**excluded** —
  hardware ECC fault, `legacy/results/logs/gpu7_ecc_evidence.txt`). Lowest utilization 0 % (idx 0/1/2/5),
  tie-break lowest used memory → idx 0/1 at 1119 MiB, final tie-break lowest index → **0**.
  `CUDA_VISIBLE_DEVICES=0` exported for every run; snapshot is stored in both logs.
- Each run was wrapped in exactly one `flock -w 7200 /tmp/gpbench_critical.lock` (no nesting). Neither
  run waited: C1 started 22:11:00 and C5 22:11:09, i.e. the lock was free (lock waits are outside the
  harness's timed regions anyway).

## Commands actually used
```bash
# C1 (W1 SR training, 200 pop / 100 gens / tree_len 64 / seed 0)
flock -w 7200 /tmp/gpbench_critical.lock bash -c "CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils timeout 900 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python -u test/benchmark/bench_gp.py \
    --task sr --pop 200 --gens 100 --tree-len 64 --seed 0 --tag C1_xval \
    --json-out test/benchmark/xval/C1_xval.json" > test/benchmark/xval/C1_xval.log 2>&1

# C5 (W3 eval-only, 25 evaluate() calls on the committed 64-tree fixture forest)
flock -w 7200 /tmp/gpbench_critical.lock bash -c "CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils timeout 900 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python -u test/benchmark/bench_gp.py \
    --task eval64 --fixture /mnt/local-ssd/bchuang/evogp/test/goldens/forest_random.npz \
    --eval-reps 25 --tag C5_xval --json-out test/benchmark/xval/C5_xval.json" \
  > test/benchmark/xval/C5_xval.log 2>&1
```
The C5 flags were cross-checked against `bench_gp.py --help` and the C5 line of
`test/benchmark/new/run_matrix.sh` (`C5) CONFIG_TIMEOUT=900; CONFIG_ARGS=(--task eval64 --eval-reps 25
--tag C5)`), i.e. the same workload as wave-1 (`--reps 1`, 25 eval calls). Fixture path: the first
candidate `/mnt/local-ssd/bchuang/evox/evogp/test/goldens/forest_random.npz` does **not** exist; the
second one from the objective, `/mnt/local-ssd/bchuang/evogp/test/goldens/forest_random.npz`, exists and
was used (the harness's cwd-relative default does not resolve on the old side).

Extra runs (all inside the timebox, all under the same single-lock discipline):
- `test/benchmark/xval/c5_fitness_stats.py` — the harness does not record the *values* returned by
  `evaluate`, so this small helper re-runs the C5 workload by importing `bench_gp` itself
  (`build_eval_case`/`to_cuda`/`apply_env` reused verbatim) and applies wave-1's fitness-sanity
  semantics (last call, MAXIMIZED fitness, `nan_count=isnan(...).sum()`, `fitness_mean=nanmean(...)` in
  float64 on CPU) → `C5_xval_fitness.json`.
- `legacy/run_matrix.py --configs C5` into `xval/repro_legacy_driver/` — wave-1's own driver re-run on
  the same GPU minutes later (A/B evidence for the `wall_s` scope difference below).
- `test/benchmark/xval/legacy_driver_scope_probe.py` — attributes that scope difference (also imports
  the driver; no driver code was modified).

## Side-detection evidence (both xval runs)
Both harness prints/JSONs report `side=old`, `evogp_file=/tmp/evogp_legacy_bench/src/evogp/__init__.py`,
`etl_version=0.1.0`, `complete=true`, non-empty timing arrays (`per_gen_ms` n=100 / `per_call_ms` n=25),
`env={"ETL_DISABLE_LOCATIONS": "1", "CUDA_VISIBLE_DEVICES": "0"}` (the `ETL_*` var is inert on the old
side), `gpu={"cuda_visible_devices": "0", "name": "NVIDIA RTX A6000", "torch_version": "2.6.0+cu124"}`.
`importlib.util.find_spec("evogp.tree.etl_engine") is None` was confirmed by hand before the runs.
Note: `evogp_commit` in the JSONs is `f66fc00cff44c5c3be5d74c5f8c50a4963e68d9e`, i.e. the HEAD of the
harness's **cwd** repository (this EvoGP worktree) — **not** the legacy checkout, which is not a git repo
and is pinned at `b613503`. That is expected harness behaviour, not a mis-detection.

## Comparison vs wave-1 (`legacy/run_matrix.py`, same workloads, GPU 4, quiet window)

| metric | xval (bench_gp.py) | wave-1 | Δ |
|---|---|---|---|
| C1 `wall_s` (construct + 100 gens) | 1.515415 s | 1.515575 s | **−0.01 %** |
| C1 cold first gen (`per_gen_ms[0]`, cold engine) | 559.16 ms | 532.10 ms | +5.08 % |
| C1 warm median (median of `per_gen_ms[1:]`, n=99) | 1.7266 ms | 1.6326 ms | +5.76 % |
| C1 `best_fitness_final` | −0.08683900535106659 | −0.08683900535106659 | **0.00 %** (identical) |
| C5 `first_call_ms` | 22.170 ms | 21.766 ms | +1.86 % |
| C5 `median_warm_ms` (median of calls 2..25) | 0.6276 ms | 0.6042 ms | +3.86 % |
| C5 `nan_count` (of 64) | 3 | 3 | **0** |
| C5 `fitness_mean` (nan-aware, MAXIMIZED) | −18.048504493275626 | −18.048504493275626 | **0.00 %** (identical) |
| C5 `wall_s` (construct + 25 calls) | 0.056757 s | 0.167924 s | −66.2 % — scope artefact, see below |

Cross-checks: the auxiliary fitness run gave `first_call_ms` 23.001 ms (+5.7 % vs wave-1) and
`median_warm_ms` 0.5878 ms (**−2.7 %**), with bit-identical `nan_count`/`fitness_mean`; re-running
wave-1's own driver today gave C5 `wall` 0.1837 s, `first_call_ms` 23.708 ms, `median_warm_ms` 0.5994 ms,
`nan_count` 3, `fitness_mean` −18.048504493275626 — i.e. the old-side engine numbers are stable to a few
percent across drivers, processes and a half hour of shared-box time, and the C1/C5 *fitness outcomes*
are exactly reproducible for C1 (RNG-driven) and C5 (RNG-independent, fixture-only).

### Why C5 `wall_s` differs by 66 % (not an engine difference)
`bench_gp.py` touches the device in its metadata phase — `gpu_snapshot()` calls
`torch.cuda.get_device_properties(0)` and `run_eval`/`run_training` call `torch.cuda.synchronize()`
*before* the wall clock starts. `legacy/run_matrix.py::run_eval_config` has no such pre-wall sync, and
its first device-touching op is the `torch.tensor(..., device="cuda")` inside `build_c5`, i.e. **inside**
its wall window. Measured on GPU 0 today:
- fresh process, first `torch.cuda.synchronize()` after the seeds = **650.8 ms**; second = 0.13 ms;
- `legacy_driver_scope_probe.py --scope-only --no-presync` (fresh-process mimic of the driver's window):
  wall 189.1 ms, sum of the 25 calls 37.7 ms, residual **151.4 ms** (the first *untimed* pre-call
  synchronize is 0.1 ms, so the residual is the CUDA-context creation inside construction);
- same probe with CUDA already warm: `torch.tensor(device="cuda") ×3 = 144.29 ms` when it is the first
  allocation op, and the driver-style window is then wall 17.1 ms / residual 1.6 ms.

So the ~0.11–0.13 s gap is one-time CUDA context/allocation cost that the two drivers place on opposite
sides of their `wall_s` boundary. The per-call metrics and the fitness stats are unaffected, and the A/B
re-run of wave-1's driver today (0.1837 s) confirms the wave-1 number was not an outlier: `wall_s` is
**not** comparable between `bench_gp.py` and `run_matrix.py` for C5 (it is comparable for C1, where the
100-generation loop dominates: −0.01 %).

## Verdict
**Close — cross-validation passes.** All per-generation / per-call / fitness metrics agree within
1.9–5.8 % (C1 warm median +5.8 % and cold first gen +5.1 % are the largest gaps, consistent with the box
being shared and busy during the xval window — GPU 3/4/6 were at 55–99 % utilization — while wave-1 ran
in a quiet window; the C5 auxiliary run even landed 2.7 % *below* wave-1, so the spread is bidirectional
noise, not a systematic slowdown). The wall-clock C1 metric matches to −0.01 % and both C1 and C5
fitness outputs (best fitness / nan_count / nan-aware mean) are exactly reproducible. No harness
deficiency was found: **`bench_gp.py` needed no changes**.

## Deviations & limitations
- `--reps 1` (harness default) instead of wave-1's `--reps 2` for C1, as specified by the objective; the
  wave-1 comparison baseline is therefore `C1_r1.json` (the cold rep), and xval's cold first generation
  (559 ms) should be compared against wave-1's 532 ms, not against wave-1's warm `C1_r2` (1.6 ms).
- C5 `fitness_mean` / `nan_count` are not part of the harness JSON schema; they come from the
  xval-local helper (`c5_fitness_stats.py`, `C5_xval_fitness.json`) which reuses the harness's own
  construction code, so they are methodology-identical but are an *additional* run (23.0 ms first call vs
  22.2 ms in the timed harness run).
- `wall_s` scope difference described above — do not quote xval C5 `wall_s` against wave-1's.
- Both xval runs used GPU 0; wave-1 used GPU 4. Same model (RTX A6000) but different physical device.
- The flock was never contended during the xval window; contention from *other* (non-flock) agents'
  processes on the shared host remains the most likely source of the few-percent gaps.
- `evogp_commit` in the JSONs is the harness repo's HEAD (`f66fc00…`), not the legacy checkout's
  `b613503` (see side-detection section).
- No files outside `test/benchmark/xval/` were touched; wave-1 artifacts (`legacy/results/`,
  `new/results/`, `SUMMARY.md`, `README.md`) are untouched. The only write into `legacy/`-owned data was
  performed through `run_matrix.py --out test/benchmark/xval/repro_legacy_driver`.

## Files
| file | content |
|---|---|
| `C1_xval.json` / `C1_xval.log` | unified-harness C1 run (JSON) + exact command, GPU snapshot, stdout |
| `C5_xval.json` / `C5_xval.log` | unified-harness C5 run + the auxiliary fitness run appended to the log |
| `C5_xval_fitness.json` | wave-1-style fitness sanity for C5 (`nan_count`, nan-aware `fitness_mean`) |
| `legacy_driver_repro.log` + `repro_legacy_driver/C5.json` | wave-1 driver re-run today (A/B for `wall_s`) |
| `c5_fitness_stats.py` | xval helper: fitness sanity via the harness's own construction code |
| `legacy_driver_scope_probe.py` | xval helper: attributes the C5 `wall_s` scope difference |
