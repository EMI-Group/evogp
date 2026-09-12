# `test/benchmark/legacy/NOTES.md` — OLD side (pre-ETL custom-CUDA engine, `b613503`): benchmark matrix results

Side **OLD** of `../SPEC.md` (legacy custom-CUDA engine vs ETL "true jit" engine). This file reports the
**legacy-side matrix of the frozen SPEC configs** as actually run on the shared box on **2026-09-11
(HKT)**, and nothing else. The build of the legacy checkout is *not* repeated here — see
`BUILD_NOTES.md` (materialization, env recipe, `nvcc`, `.so` identity, GPU-7 fault, foreground/`flock`
rules). The config definitions/hyperparameters come from `../SPEC.md` and were **not changed**; the run
schema is produced by `run_matrix.py` (`--list` prints the matrix).

## 1. Status

| item | value |
|---|---|
| configs run | **C1 (2 reps), C2, C2b, C4 (2 reps), C5, C5b — all 6 complete, no partials, no timeouts** |
| run records | 8 JSONs in `results/` — one per run, exactly the SPEC schema |
| raw logs | `results/logs/{C1,C2,C2b,C4,C5,C5b}.log` (per-generation / per-call stdout trace) |
| post-run validation | `check_results.py` → **PASS for all 8 records** (key set, `len(per_gen_ms)==gens`, monotone `per_gen_best_fitness`, `best_fitness_final == per_gen_best_fitness[-1]`, `cold_first_gen_ms == per_gen_ms[0]`, `first_call_ms == per_call_ms[0]`, `median_warm_ms == median(per_call_ms[1:])`, `side == "old"`, `evogp_file` inside the legacy checkout, ETL env vars `null`, `CUDA_VISIBLE_DEVICES == 4`); raw output committed as `results/logs/check_results.log` |
| C5 golden cross-check | **PASS** (`nan_count` 3, `fitness_mean` −18.048504493275626) — §3 |
| `run_matrix.py` changes | **none** — the harness ran as committed; no bug surfaced (the blind-written driver's first real run was C5, 4 s after starting) |
| wall clock of the whole matrix | 21:04:48 → 21:06:58 (2 min 10 s) for all 6 configs; every config's *compute* is seconds (see §5.2) |

## 2. Headline results

`wall_s` = construction + all steps (SPEC schema), so it contains the cold gen 0. `gen0` = first
generation = cold (CUDA module load / first `cudaMalloc`); `warm` = generations 1..N−1. All times in ms.

| config | workload | pop / gens / tree_len | wall_s | gen0 (cold) | warm median | warm min | warm max | best fitness final |
|---|---|---|---|---|---|---|---|---|
| C1_r1 | W1 SR | 200 / 100 / 64 | 1.516 | 532.104 | **1.633** | 1.499 | 2.640 | −0.086839 |
| C1_r2 | W1 SR | 200 / 100 / 64 | 0.172 | 1.607 | **1.617** | 1.480 | 2.828 | −0.086839 |
| C2 | W1 SR | 1000 / 100 / 128 | 1.713 | 609.124 | **3.451** | 1.565 | 3.880 | −0.225680 |
| C2b | W1 SR | 1000 / 20 / 128 | 1.355 | 520.301 | **1.607** | 1.564 | 2.052 | −0.225680 |
| C4_r1 | W2 XOR | 50 / 100 / 64 | 0.886 | 519.475 | **1.756** | 1.701 | 9.386 | −0.250000 |
| C4_r2 | W2 XOR | 50 / 100 / 64 | 0.184 | 1.837 | **1.769** | 1.712 | 2.051 | −0.250000 |
| C5 | W3 eval-only, fixture forest (64 trees / 63 nodes, RNG-independent) | — / 25 evals | 0.168 | first call **21.766** | **0.604** warm median | 0.595 | 0.810 | `nan_count` 3 of 64, `fitness_mean` −18.048504493275626 |
| C5b | W3 eval-only, fresh random forest (1000 trees / 64 nodes, seed 0) | — / 10 evals | 0.190 | first call **20.878** | **0.602** warm median | 0.593 | 0.900 | `nan_count` 66 of 1000, `fitness_mean` −59.737519580878846 |

* Reproducibility: both reps agree to **0.96 %** (C1) / **0.75 %** (C4) on the warm median, and C1/C4
  produce bit-identical fitness trajectories across reps (same seeded RNG stream). C1_r2/C4_r2 show no
  cold penalty on gen 0 (1.6 / 1.8 ms) because the CUDA context of rep 1 is already warm.
* Per-generation cost is flat in the decile profile for C1 and C4; C2 (pop 1000) ramps from ~1.6 ms to
  ~3.5 ms over the 100 generations (population structures grow), which is why its median sits above
  C2b's — C2b is the same workload truncated at 20 generations and stays at ~1.6 ms.
* Fitness sanity (SPEC: times are the comparison, fitness only a convergence check): W1 improves
  (−0.251097 → −0.086839 for C1; −0.251097 → −0.225680 for C2/C2b, first reached at gen 2, exactly the
  `−0.2257` of the `sr_test` smoke); W2 plateaus at exactly −0.250000 from generation 8 in both reps —
  with the legacy loss `Σ_points Σ_outputs err / dataPoints` (`src/evogp/tree/ENGINE_SPEC.md` §5) that is
  MSE 0.25, i.e. this tiny 50-tree / 100-generation budget did not fit the 8-row parity table exactly
  (an exact fit would be 0). Convergence, not optimality, is what this config checks.
* C5b's `nan_count = 66/1000` is the expected legacy behaviour on fresh random deep trees (`/` by
  near-zero constants → NaN through the fitness; `StandardPipeline.step` replaces NaN by `−inf`), and
  C5's `nan_count = 3/64` is a property of the frozen fixture forest — not of this harness.

### 2.1 C4 re-run (SPEC-authorized contention re-check)

C4_r1's warm profile contains **one** generation above 5× the median (gen 66: 9.386 ms, 5.3× the median
of 1.756 ms); C4_r2 is clean (max 2.051 ms) and both medians agree to 0.75 %. Per the run instruction
the config was re-run **once** — into `results/c4_rerun/` so that the first-run evidence stays intact
(log: `results/logs/C4.rerun.log`):

| C4 re-run | wall_s | gen0 | warm median | warm min | warm max | outliers > 5× median |
|---|---|---|---|---|---|---|
| c4_rerun/C4_r1 | 1.283 | 530.315 | 3.910 | 1.699 | 21.201 | gens 85–92 (8 consecutive) |
| c4_rerun/C4_r2 | 0.776 | 3.309 | 5.353 | 1.697 | 20.850 | — (median itself inflated) |

The re-run is **noisier** than the original (medians 3.9 / 5.4 ms vs 1.76 / 1.77 ms, one 8-generation
contention burst). Same window shows up in the diagnostic run of §4 (details-OFF phases 3.9 / 7.8 ms
against the matrix's 1.6 ms). Conclusion: the single 9.4 ms generation in the original C4_r1 is a
transient host-side blip, **not** a warp of the config's profile — the original C4 records stay the
headline numbers and the re-run is kept as the variability datapoint. No config was re-run otherwise.

## 3. C5 golden cross-check (correctness anchor) — PASS

C5 evaluates the frozen pre-ETL fixture forest (`test/goldens/forest_random.npz`) on that file's own
`x`/`labels`, so it is RNG-independent and directly comparable to the frozen legacy capture
`test/goldens/sr_fitness.npz` (`SymbolicRegression.evaluate` returns the NEGATED `SR_fitness`, hence the
sign). Reference computed with numpy from the worktree, no GPU:

```
$PY -c "import numpy as np; a=np.load('test/goldens/sr_fitness.npz')['forest_sr_fitness__hybrid_parallel__mse'].astype(np.float64);
        f=a[np.isfinite(a)]; print(-f.mean(), int(np.isnan(a).sum()))"
        -> -18.048504493275626 3
```

| quantity | C5 record | frozen golden | verdict |
|---|---|---|---|
| `nan_count` (of 64) | 3 | 3 | **PASS** (exact) |
| `fitness_mean` (NaN-aware) | −18.048504493275626 | −nanmean(finite) = −18.048504493275626 | **PASS** (rtol=atol=1e-4; here bit-identical up to float64 rounding) |

Same check re-run — with the run's own validation — as
`$PY check_results.py results/C5.json --gpu 4 --golden test/goldens/sr_fitness.npz`; the raw output of
all three check invocations (all 8 records, the C5 golden check, the C4 re-run records) is committed as
`results/logs/check_results.log`. This confirms the harness wires the fixture Forest, the fixture
datapoints and `SymbolicRegression` exactly as `../SPEC.md` W3 intends; no wiring problem found.

## 4. C2/C2b vs the `BUILD_NOTES` smoke baseline: fully explained (no anomaly)

`BUILD_NOTES.md` §5 recorded 10.90 ms median / 9.00 ms min per generation for the shipped
`evogp.sr_test` (pop 1000 / tree_len 128), i.e. 3–7× the C2/C2b warm records above. Cause: `sr_test`
calls `StandardPipeline.run()` with the **default `is_show_details=True`**, and `show_details` is
host-side Python on the **CPU** fitness tensor — and it uses the *builtin* `max()`/`min()`, which
iterate the tensor element by element in Python.

Measured (log `results/logs/cpu_micro_show_details.log`, no GPU, 1000-element CPU f32 tensor, 10 warm
reps, min/median/max): mask filter **0.041/0.042/0.054 ms**, builtin `max()`+`min()` **7.397/7.406/7.770
ms**, `torch.mean`/`std` **0.027/0.029/0.038 ms**, whole `show_details` equivalent incl. print
**7.474/7.495/7.844 ms**. That additive ≈7.5 ms/gen closes the gap exactly: 1.6–3.5 ms (matrix, details
OFF, SPEC-mandated) + 7.5 ms ≈ 9.1–11.0 ms = the smoke's min 9.00 / median 10.90 ms. So the smoke and
the C2/C2b records agree; they are simply measured with different `is_show_details` settings — the
detail printing is **not** part of the timed region in any JSON here (`is_show_details=False`, per SPEC).

Corroborating per-generation evidence that the harness performs the *same work* as `sr_test`: C2/C2b's
best-fitness trajectory is identical to the smoke's (−0.2511 at gen 0/1, −0.2257/-0.225680 from gen 2),
which is only possible with the same seeded population and the same per-generation evaluation.

`results/logs/diagnostic_details_overhead.py` + its log (`diagnostic_details_overhead.log`) re-measured
the details-ON/OFF difference on the C2 workload itself (three 20-generation phases, reusing
`run_matrix.build_w1` so construction is byte-identical): phases details-OFF 3.931 ms / details-ON
20.534 ms / details-OFF 7.815 ms, micro-decomposition as above (builtin `max`/`min` = 7.373 ms of the
7.501 ms `show_details`). **Caveat, logged loudly:** that run happened in the noisy window of §2.1
(details-OFF medians 3.9/7.8 ms vs the matrix's 1.6 ms, phase A→C drift 3.9 ms), so its *absolute*
numbers are contention-inflated and must not be quoted as the OLD side's throughput — the
contention-insensitive CPU micro above is the number to use, and the matrix records (§2) are the
throughput records. The diagnostic script is a one-off, not part of the SPEC matrix.

## 5. Exact commands used

`$REPO` = the EvoGP worktree that owns this directory; `$PY` = `/mnt/local-ssd/bchuang/evox/.venv/bin/python`
(py3.11.2, torch 2.6.0+cu124). Every benchmark invocation ran in the **foreground**, wrapped in exactly
one `flock -w 7200 /tmp/gpbench_critical.lock bash -c '…'` (never nested), with CWD
`/tmp/evogp_legacy_bench` and `PYTHONPATH=src:/tmp/evogp_t1a263/siteutils`, stdout+stderr redirected to
`$REPO/test/benchmark/legacy/results/logs/<CFG>.log`, and `--out` given as an absolute path (the python
process runs with CWD inside `/tmp`).

### 5.1 The six config invocations

```bash
REPO=$PWD   # repo worktree root
# order actually used: C5 (harness smoke) -> C1 -> C2b -> C4 -> C2 -> C5b   [see §6]
for CFG in C5 C1 C2b C4 C2 C5b; do
  flock -w 7200 /tmp/gpbench_critical.lock bash -c "cd /tmp/evogp_legacy_bench && \
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/tmp/evogp_t1a263/siteutils timeout 1800 $PY -u \
    $REPO/test/benchmark/legacy/run_matrix.py --configs $CFG --out \
    $REPO/test/benchmark/legacy/results --commit b613503" \
    > $REPO/test/benchmark/legacy/results/logs/$CFG.log 2>&1
  echo "exit=$?"
done
```

(Each of the six calls was issued on its own, in exactly this form — shown as a loop for brevity;
`timeout 900` was used for C5, C2b, C4, C5b and `timeout 1800` for C1 and C2, and every call finished in
seconds.)

### 5.2 Timing reality check

Whole-matrix wall clock 2 min 10 s for 6 configs, of which **3.253 s is the sum of every measured
region in the 8 records** (C2 0.891 s of its 1.713 s `wall_s`, the two C1 reps 0.697 + 0.166 s, C4
0.704 + 0.179 s, C2b 0.552 s, C5/C5b 0.037/0.027 s); the rest is interpreter/import
(~1.3 s), dataset generation, cold gen 0 (~0.5–0.6 s once per process, i.e. once per rep pair) and lock
waiting (C4 waited ~22 s and C2 ~4 s behind the sibling NEW-side agent; C5/C1/C2b/C5b acquired the lock
immediately). No config came close to its `timeout`.

### 5.3 Checks and diagnostics (no lock needed — JSON/numpy only)

```bash
# per-record SPEC/schema validation + headline stats (+ C5 golden cross-check)
# raw output of all three invocations -> results/logs/check_results.log
$PY test/benchmark/legacy/check_results.py test/benchmark/legacy/results/*.json --gpu 4
$PY test/benchmark/legacy/check_results.py test/benchmark/legacy/results/C5.json --gpu 4 \
    --golden test/goldens/sr_fitness.npz

# SPEC-authorized single C4 re-run (into a separate --out so the first run stays intact)
flock -w 7200 /tmp/gpbench_critical.lock bash -c "cd /tmp/evogp_legacy_bench && \
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/tmp/evogp_t1a263/siteutils timeout 900 $PY -u \
  $REPO/test/benchmark/legacy/run_matrix.py --configs C4 --out \
  $REPO/test/benchmark/legacy/results/c4_rerun --commit b613503" \
  > $REPO/test/benchmark/legacy/results/logs/C4.rerun.log 2>&1

# §4 diagnostic (details-ON/OFF on the C2 workload), latest run recorded in a contended window
flock -w 7200 /tmp/gpbench_critical.lock bash -c "cd /tmp/evogp_legacy_bench && \
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/tmp/evogp_t1a263/siteutils timeout 900 $PY -u \
  $REPO/test/benchmark/legacy/diagnostic_details_overhead.py" \
  > $REPO/test/benchmark/legacy/results/logs/diagnostic_details_overhead.log 2>&1

# §4 contention-insensitive CPU micro (torch only, no evogp, no GPU, no lock)
$PY - <<'PY'   # full body kept in the log header of results/logs/cpu_micro_show_details.log
# 11 reps of show_details' ops on a (1000,) CPU f32 tensor: mask / builtin max+min / mean+std / print
PY
```

## 6. Environment, deviations and observations

**Environment**

* Legacy checkout: `/tmp/evogp_legacy_bench` (materialized with `git archive b613503`, **not** a git
  repository, **not** committed) — complete and importable, built extension
  `src/evogp/evogp_cuda.cpython-311-x86_64-linux-gnu.so`, 1 112 488 B, sha256
  `d7af153c4d12843dc218e3616985173db9e294c3fc3a04f9dbab457b30d715a4` (identical to `BUILD_NOTES.md`
  §9/§1 — the artifact was not rebuilt or replaced during these runs). What the extension *is* and how it
  was built: `BUILD_NOTES.md`.
* `PYTHONPATH` order puts `src` (legacy) first, and every record proves the legacy checkout won:
  `side="old"`, `evogp_file="/tmp/evogp_legacy_bench/src/evogp/__init__.py"`.
* GPU: **index 4** (RTX A6000), fixed for the whole matrix; pre/post snapshots in
  `results/logs/gpu_snapshots.txt`. Index 5 was equally idle but hosts a 24 GB vLLM engine; **index 7 was
  excluded** (hardware-faulted, uncorrectable ECC — `results/logs/gpu7_ecc_evidence.txt`). Index 4 is
  shared with other tenants (~15 GB resident, 0 % utilisation while we ran).
  `CUDA_VISIBLE_DEVICES=4` is echoed in every record's `env` and `gpu`.
* Commit recorded in every JSON: `b613503` (`b613503c3107db23222f69bfaad2a329639f28fe`).
* ETL env vars are `null` in every record (the legacy engine ignores them; SPEC: no ETL settings apply).
* Serialization: exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per benchmark invocation; the
  sibling NEW-side agent uses the same lock. Observed waiting: ~22 s (C4), ~4 s (C2), ~2 s (C5b), 0 s
  (C5, C1, C2b). All runs foreground; no backgrounded process, no `timeout` kill.
* Seeds: `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` before construction (harness), per SPEC.

**Deviations (all deliberate, all logged)**

1. **Config order:** C5 was run first (per the instruction for this session) instead of SPEC's priority
   order C1 → C5 → … — C5 is the fastest config and validates the whole harness path (fixture load,
   `Forest` ctor, `problem.evaluate`, JSON write) in 4 s. SPEC's *priority* (which configs matter most)
   is unaffected: all six configs completed. No hyperparameter, workload or seed differs from SPEC.
2. **One C4 re-run**, into `results/c4_rerun/` (§2.1), as the pre-authorized contention re-check; the
   original records were not overwritten. Result: the original was the cleaner run.
3. **Two extra non-SPEC diagnostics** were run and are committed for transparency —
   `diagnostic_details_overhead.py` (+ its log) and the CPU micro in
   `results/logs/cpu_micro_show_details.log`; `check_results.py` is the JSON validator. None of them
   touches the recorded matrix; none is part of the SPEC config set.
4. `run_matrix.py` was **not modified** (no defect found). `--out` for the C4 re-run was the only
   non-default CLI value ever used besides `--configs`/`--commit`.
5. `gpu_snapshots.txt` was created here (pre- and post-run `nvidia-smi` snapshots + the choice of index
   4 and the reason for excluding 7).

**Observations worth carrying into the consolidation pass**

* The legacy per-generation cost is dominated by **host-side dispatch**: at pop 200 (C1) and pop 1000
  (C2b) the warm median is the same ~1.6 ms; a *single* extra `evaluate` call in the eval-only
  microbench costs 0.60 ms (C5/C5b), and `show_details` alone costs ~7.5 ms/gen (§4). Any host-side
  activity therefore moves these numbers: the C4 re-run and the diagnostic ran ~2–3× slower in a busy
  window, while the six configs ran in a quiet one (21:04:48–21:06:58) with rep-to-rep agreement ≤1 %.
  Treat the §2 numbers as "captured on the shared box, in a quiet window", the same caveat
  `BUILD_NOTES.md` §5 puts on its smoke.
* Cold start is ~0.5–0.6 s **per process** (gen 0 of the first run in a process); the second rep of a
  config pays 1.6–3.3 ms, i.e. the CUDA module load is per-process, not per-forest. Eval-only C5/C5b
  show the same shape: first `evaluate` 20.9–21.8 ms vs 0.60 ms warm.
* C2's warm profile drifts upward (1.6 → 3.5 ms) within the run while C2b (20 generations of the same
  workload) stays flat at 1.6 ms — worth knowing when someone compares the two across sides.
* Cross-side fitness caveat for the consolidation pass: the legacy loss is
  `(Σ_points Σ_outputs err) / dataPoints` (mean over data points, summed over outputs,
  `src/evogp/tree/ENGINE_SPEC.md` §5) and `SymbolicRegression.evaluate` negates it; the C5 golden check
  (§3) confirms both engines agree on this convention on the frozen fixture, so fitnesses *are*
  comparable, but C1's endpoint (−0.0868) and C2's (−0.2257) are seeded different explorations, not a
  performance metric.

**See also**

* `BUILD_NOTES.md` — materialization + build of `/tmp/evogp_legacy_bench`, env recipe, `.so` identity,
  the GPU-7 fault, and the `sr_test` smoke baseline referenced in §4. Not duplicated here.
* `../SPEC.md` — the frozen methodology, workloads W1–W3, config matrix and required JSON schema.
* `run_matrix.py` (harness; `--list`), `check_results.py` (JSON validator), `results/` (8 records),
  `results/c4_rerun/` (C4 re-run records), `results/logs/` (all raw logs, incl. the pre-existing build
  and smoke evidence).
