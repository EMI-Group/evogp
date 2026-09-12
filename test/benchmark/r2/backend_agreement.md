# R2 pre-flight — IREE-cuda `O1` vs `numpy` fitness agreement (fixed fixtures)

**Verdict: PASS on both fixtures — the two backends are bit-identical.**
`EVOGP_ETL_BACKEND=iree` with `EVOGP_ETL_IREE_OPT_LEVEL=O1` produces exactly the same float64
fitness vector as the default `numpy` backend on the 64-tree and the 200-tree fixture: identical
NaN masks, identical NaN/±inf placement, `np.allclose(..., rtol=1e-4, atol=1e-4, equal_nan=True)
== True`, **max |diff| = 0.0, max rel diff = 0.0**, and the same `values_sha256` for both backends.
Both backends also land on the recorded anchors (see §"Anchor cross-check"). IREE-cuda `O1` is
therefore numerically safe to time in this round. No mismatch of any size was observed.

This is a *gate*, not a benchmark: it exists because `O2`/`O3` are known to miscompile CUDA
**silently**, so no IREE timing number from this round may be quoted before O1 has been shown to
agree with the reference backend on the same data.

## Method (one paragraph)

Both runs of a fixture load the **same npz file**, rebuild the same `Forest` through the public
class constructor (`Forest(input_len, output_len, batch_node_value=<forest__node_value f32>,
batch_node_type=<forest__node_type i16>, batch_subtree_size=<forest__subtree_size i16>)`, i.e. the
`bench_gp.py::build_eval_case` `eval64` wiring) and the same `SymbolicRegression(datapoints=x,
labels=labels)` (float32, moved to `"cuda"`), then call `problem.evaluate(forest)` twice with
`torch.cuda.synchronize()` around each call and read the `(pop,)` fitness tensor back as float64.
The population is RNG-independent (structures come from the file, not from the generator), so the
*only* variable between the two processes is `EVOGP_ETL_BACKEND` (+ the two `EVOGP_ETL_IREE_*`
settings for the IREE side); `ETL_DISABLE_LOCATIONS=1` and `CUDA_VISIBLE_DEVICES=6` are identical.
The single driver is `test/benchmark/r2/check_backend_agreement.py` (public `evogp` API only —
`from evogp.tree import Forest`, `from evogp.problem import SymbolicRegression`; it deliberately does
not import the run-at-import `bench_gp.py`). `problem.evaluate` returns the **MAXIMIZED** fitness
(the negated SR loss), so NaN trees are counted, never dropped.

## Results (4 runs, 2 fixtures × 2 backends)

| fixture | backend | nan_count | inf_count | count | finite_mean | finite_min | finite_max | `per_call_ms[0]` (cold) | `per_call_ms[1]` (warm) | `values_sha256` |
|---|---|---|---|---|---|---|---|---|---|---|
| `test/goldens/forest_random.npz` (64 trees / width 63) | `numpy` | 3 | 0 | 64 | −18.048504309576067 | −106.23929595947266 | −1.375 | 223.3 ms | 21.9 ms | `b8192cb1770b886d…` |
| same | `iree` O1 | 3 | 0 | 64 | −18.048504309576067 | −106.23929595947266 | −1.375 | 14 034.1 ms | 75.2 ms | `b8192cb1770b886d…` |
| `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees / width 64) | `numpy` | 12 | 0 | 200 | −6969.418338107302 | −650003.4375 | −2.72265625 | 994.9 ms | 95.1 ms | `07948efededc27b2…` |
| same | `iree` O1 | 12 | 0 | 200 | −6969.418338107302 | −650003.4375 | −2.72265625 | 101 016.0 ms | 242.2 ms | `07948efededc27b2…` |

Full hashes: 64-tree `values_sha256 = b8192cb1770b886d6b5decbbcb27421aadca9aa093f59cc21ff6b4b16e0afa02`
(identical for numpy and iree); 200-tree
`values_sha256 = 07948efededc27b2f30691a4eb305f5abad25f4f3fbedb39b47dd31b392be048` (identical for
numpy and iree). The hash is taken over the raw little-endian float64 bytes of the last call's
fitness vector, so *equal hashes* is the strongest possible statement: byte-for-byte identity.

Fixture identity (sha256 of the npz, recomputed at run time and stored in each JSON):
64-tree `0bfb0bbf48cd65832b2c024480bc572de5121675946d3611a2c7a5b234ff9ca0`;
200-tree `6789ef61ecf8898b9b2a00774ee655e20b0b5be7c72eb861afc8d6b8799741b9` (matches the provenance
record in `test/benchmark/fixtures/README.md` exactly, so the fixture was not regenerated).

## Verdict per fixture (iree vs numpy)

| check | 64-tree fixture | 200-tree fixture |
|---|---|---|
| `np.array_equal(np.isnan(iree), np.isnan(numpy))` | **True** | **True** |
| NaN-mask sha256[(first 16)] | `07172d4234502fee` (both) | `78da3af9dcdf813c` (both) |
| `np.allclose(iree, numpy, rtol=1e-4, atol=1e-4, equal_nan=True)` | **True** | **True** |
| bit-identical float64 bytes (`np.array_equal(…, equal_nan=True)`) | **True** | **True** |
| max abs diff | **0.0** | **0.0** |
| max rel diff | **0.0** | **0.0** |
| finite pairs compared / NaN in both / ±inf | 61 / 3 / none | 188 / 12 / none |

**No MISMATCH anywhere; nothing deviates beyond (or anywhere near) 1e-4.**

The NaN masks are the same bit-for-bit subsets *and* their digests equal the masks that were derived
independently (different driver, different process, same session) on this box earlier in the round
(`r2/RUNLOG.md` §"Consistency checks", light group: 64-tree `07172d4234502fee`, 200-tree
`78da3af9dcdf813c`; that section also records the same new-engine `finite_mean` −18.048504309576067
obtained here) — i.e. the NaN placement is reproducible, not a coincidence of two runs of the same
code path.

## Anchor cross-check

| anchor (recorded earlier in the round) | expected | numpy | iree O1 | verdict |
|---|---|---|---|---|
| 64-tree `nan_count` | 3 / 64 | 3 / 64 | 3 / 64 | **PASS** |
| 64-tree `finite_mean` (legacy/C5 record `legacy/results/r2/C5.json`) | −18.048504493275626 | −18.048504309576067 (rel 1.02e−08) | −18.048504309576067 (rel 1.02e−08) | **PASS** (≪ 1e-4) |
| 200-tree `nan_count` | 12 / 200 | 12 / 200 | 12 / 200 | **PASS** |
| 200-tree `finite_mean` (numpy anchor) | −6969.418338 | −6969.418338107302 (rel 1.54e−11) | −6969.418338107302 (rel 1.54e−11) | **PASS** |
| 200-tree `finite_mean` (legacy anchor) | −6969.418862 | rel 7.52e−08 | rel 7.52e−08 | **PASS** (< 1e-4; this is the known legacy-vs-ETL spread, `RUNLOG.md` §"Consistency checks") |
| 200-tree `finite_max` | −2.72265625 | exact | exact | **PASS** |
| 200-tree `finite_min` | −650003.4375 | exact | exact | **PASS** (the *legacy* engine's −650003.5 differs by 9.6e−08 — the same known spread) |

Both backends land on the anchors. Note the 64-tree `finite_mean` is the value the ETL engine itself
computes (−18.048504309576067, rel 1.0e−08 to the legacy/C5 record) — it is *not* drift introduced by
IREE: numpy and IREE agree with each other to 0.0.

## Anomalies / findings

1. **IREE object leaks at interpreter shutdown (known, no effect on the numbers).** Both IREE runs
   printed the `nanobind: leaked N instances` report *after* all output had been written: 375 leaked
   instances / 374 keep-alives / 5 types / 39 functions for the 64-tree run, 1190 / 1189 / 5 / 39 for
   the 200-tree run. Exit code was still **0** in both cases and the JSONs/values were complete
   (written before teardown). This is the round's already-documented IREE behaviour ("leaks runtime
   objects", `r2/README.md`), reproduced at this scale; it is a resource-teardown issue, **not** a
   numeric finding.
2. **Timing shape (recorded only because these are real IREE numbers).** IREE's cold call is
   63× (14.0 s vs 0.22 s) / 102× (101.0 s vs 0.99 s) the numpy cold call, and even the fully-warm
   call is 3.4× (64-tree: 75.2 vs 21.9 ms) / 2.5× (200-tree: 242.2 vs 95.1 ms) slower — consistent
   with the round's stance that IREE is opt-in and explicitly non-primary.
3. **Cold-build economics.** The cold call builds one ETL/IREE graph per *distinct* structure:
   14.0 s for the 64-tree fixture (≈48 distinct structures, ≈0.29 s each) and 101.0 s for the
   200-tree fixture (165 distinct rows per `fixtures/README.md`, ≈0.61 s each). Both are inside the
   0.3–1.7 s per-build band the round recorded, and the build count is only the number of distinct
   structures (≈48 / 165), far below the default `EVOGP_ETL_CACHE_SIZE=4096`, so every structure was
   built exactly once and the warm call has zero misses.
4. **FD / fd limits.** `ulimit -n 65536` was applied by each run's shell (soft limit was already
   65536); no run hit an FD-related error, and no `OSError: Too many open files` appears in any log.
5. **Shared-lock waits (outside all timings).** Each of the four processes took exactly one
   single-level `flock -w 7200 /tmp/gpbench_critical.lock`; the lock was held by the concurrent IREE
   operator's `run_all.sh` configs, so runs 3 and 4 spent ≈113 s and ≈340 s waiting for it (wall
   clock of the wrapper vs. the in-process `started_at`→`finished_at` span in each JSON). The one
   addition to the literal command lines is `timeout -s INT -k 30 3600` **inside** the lock — this
   round's own convention (`r2/run_all.sh`) so that a hung IREE process can never hold the shared
   lock indefinitely.
6. **Validator interaction (for the R2 operator).** These four JSONs are *not* `bench_gp.py` harness
   records: they carry `per_call_ms` (so `check_r2.py` infers kind `eval`) but no
   `task`/`side`/`tag`/`config`/`first_call_ms`/`complete`. A bare `python3
   test/benchmark/r2/check_r2.py` walks the default results dirs recursively and therefore reports
   these 4 files as *"missing required eval keys"* failures (measured:
   `python3 test/benchmark/r2/check_r2.py test/benchmark/new/results/r2/backend_agreement` →
   `4 record(s) … 4 failure(s)`, exit 1). The documented invocation
   `python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt
   --require-stages` is **unaffected** (manifest-only mode never scans directories), and no file
   outside this round's own artifacts was touched. The 4 rows are the backend-agreement evidence,
   not R2 configs — treat them accordingly if the bare mode is ever used.
7. **Nothing else anomalous**: `ETL_DISABLE_LOCATIONS=1` was set in all four runs, both backends
   reported `torch_version = 2.6.0+cu124` and `cuda_device_name = NVIDIA RTX A6000`, and no run
   needed the `timeout` (largest in-lock wall time: 101 s of GPU work).

## Exact commands run

Repository root = the worktree; `PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python`; each command was
preceded by `ulimit -n 65536` in the same shell and wrapped as shown (one `flock` per process, never
nested):

```bash
# 1. numpy, 64-tree fixture         launched ≈14:59:14  -> exited 0 at 14:59:20 (6 s wall, no wait)
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
    ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=numpy $PY \
    test/benchmark/r2/check_backend_agreement.py --fixture test/goldens/forest_random.npz \
    --json-out test/benchmark/new/results/r2/backend_agreement/fixture64_numpy.json

# 2. iree O1, 64-tree fixture       launched ≈14:59:25  -> exited 0 at 14:59:58 (33 s wall, no wait)
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
    ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=iree \
    EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 $PY \
    test/benchmark/r2/check_backend_agreement.py --fixture test/goldens/forest_random.npz \
    --json-out test/benchmark/new/results/r2/backend_agreement/fixture64_iree.json

# 3. numpy, 200-tree fixture        launched ≈15:00:01  -> exited 0 at 15:01:56 (115 s wall; ~113 s = lock wait)
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
    ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=numpy $PY \
    test/benchmark/r2/check_backend_agreement.py \
    --fixture test/benchmark/fixtures/forest200_tree64_random.npz \
    --json-out test/benchmark/new/results/r2/backend_agreement/fixture200_numpy.json

# 4. iree O1, 200-tree fixture      launched ≈15:02:00  -> exited 0 at 15:09:22 (443 s wall; ~340 s = lock wait)
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 env PYTHONPATH=src \
    ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 EVOGP_ETL_BACKEND=iree \
    EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 $PY \
    test/benchmark/r2/check_backend_agreement.py \
    --fixture test/benchmark/fixtures/forest200_tree64_random.npz \
    --json-out test/benchmark/new/results/r2/backend_agreement/fixture200_iree.json
```

| run | in-process window (`started_at` → `finished_at`) | `per_call_ms` (cold, warm) | exit |
|---|---|---|---|
| 1 numpy / 64 | 2026-09-12T14:59:19+08:00 → 14:59:20+08:00 | 223.3, 21.9 ms | 0 |
| 2 iree / 64 | 2026-09-12T14:59:43+08:00 → 14:59:57+08:00 | 14 034.1, 75.2 ms | 0 |
| 3 numpy / 200 | 2026-09-12T15:01:54+08:00 → 15:01:55+08:00 | 994.9, 95.1 ms | 0 |
| 4 iree / 200 | 2026-09-12T15:07:39+08:00 → 15:09:21+08:00 | 101 016.0, 242.2 ms | 0 |

## GPU state

`CUDA_VISIBLE_DEVICES=6` for all four runs (GPU 6 free/idle at selection time; **never** GPU 7 —
hardware uncorrectable-ECC fault, 2263 uncorrected volatile errors, any CUDA context on it dies).

```
# pre-round snapshot, 2026-09-12T14:58:26+08:00
index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], ecc.errors.uncorrected.volatile.total
0, NVIDIA RTX A6000, 62 %, 19713 MiB, 49140 MiB, [N/A]
1, NVIDIA RTX A6000, 94 %, 20041 MiB, 49140 MiB, [N/A]
2, NVIDIA RTX A6000,  0 %, 16587 MiB, 49140 MiB, [N/A]
3, NVIDIA RTX A6000,  0 %, 15746 MiB, 49140 MiB, [N/A]
4, NVIDIA RTX A6000,  0 %, 15373 MiB, 49140 MiB, [N/A]
5, NVIDIA RTX A6000,  0 %, 24399 MiB, 49140 MiB, [N/A]
6, NVIDIA RTX A6000,  0 %,   271 MiB, 49140 MiB, [N/A]   <- used
7, NVIDIA RTX A6000,  0 %,     1 MiB, 46068 MiB, 2263      <- ECC-faulted, never used

# post-round snapshot, 2026-09-12T15:09:43+08:00
index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], ecc.errors.uncorrected.volatile.total
0, NVIDIA RTX A6000, 100 %, 20340 MiB, 49140 MiB, [N/A]
1, NVIDIA RTX A6000,  89 %, 19879 MiB, 49140 MiB, [N/A]
2, NVIDIA RTX A6000,   0 %, 16587 MiB, 49140 MiB, [N/A]
3, NVIDIA RTX A6000,   0 %, 15391 MiB, 49140 MiB, [N/A]
4, NVIDIA RTX A6000,   0 %,  6871 MiB, 49140 MiB, [N/A]
5, NVIDIA RTX A6000,   0 %, 24399 MiB, 49140 MiB, [N/A]
6, NVIDIA RTX A6000,   1 %,   399 MiB, 49140 MiB, [N/A]
7, NVIDIA RTX A6000,   0 %,     4 MiB, 46068 MiB, 2263
```

(The box is shared and noisy — GPU 0/1 are other users' jobs at ~100 %; the fixture workloads are
tiny, so a ~5–10 % contention effect on these numbers is possible but irrelevant to an
*agreement* check, which is contention-free by construction.)

## Artifacts

```
test/benchmark/r2/check_backend_agreement.py                      # the driver (this pre-flight)
test/benchmark/new/results/r2/backend_agreement/
    fixture64_numpy.json   fixture64_numpy_values.npy   (64,)  float64
    fixture64_iree.json    fixture64_iree_values.npy    (64,)  float64
    fixture200_numpy.json  fixture200_numpy_values.npy  (200,) float64
    fixture200_iree.json   fixture200_iree_values.npy   (200,) float64
```

Each JSON holds `fixture`, `fixture_sha256`, `backend`, `iree_opt_level`, `iree_device`, the six
summary numbers, `per_call_ms[2]`, `values_sha256`, `torch_version`, `cuda_device_name`,
`started_at`, `finished_at`; the `_values.npy` beside it is the raw float64 fitness vector of the
last call (the arrays are what the verdicts above were computed from, so any reader can re-derive
them without a GPU).

The four `_values.npy` files are **force-added** to git (`git add -f`): `.gitignore` line 46 ignores
`*.npy` repo-wide, but the arrays are part of this pre-flight's evidence and every JSON's
`values_sha256` is verified against its array — the same "keep the run evidence" convention the round
already applies to `*.log` files (see `RUNLOG.md`).

## Reproduce

Re-run the four commands above (any order; one process per backend/fixture is required, because the
ETL `CompileCache` is per-process and a cold call can never be inherited), then diff the pairs:

```python
a = np.load("<...>_numpy_values.npy"); b = np.load("<...>_iree_values.npy")
assert np.array_equal(np.isnan(a), np.isnan(b))
assert np.allclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
```
