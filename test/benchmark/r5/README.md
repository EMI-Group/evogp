# `test/benchmark/r5/` — protocol: the RCA round (root cause of the ≈10× old-vs-new evaluate gap)

**Analysis-only round.** r5 answers *why* the new ETL "true jit" engine's dense route is ≈8–11×
slower than the deleted legacy CUDA engine on the type-1 end-to-end call
(`problem.evaluate(forest)`, iree + `EVOGP_EVAL_MODE=auto` ⇒ the dense evaluator), and it does so by
measurement only: **no change to `src/evogp`, no write in the foreign ETL checkout, no edit outside
this directory**. Runtime `monkeypatch`es (the `../stage_probe.py` pattern) are the whole tool set.

Deliverables of the round: `RCA.md` (the analysis of record), this `README.md` (the protocol), the
probe scripts + `window_probe.sh`, the immutable evidence under `results/`, and `check_r5.py`
(stdlib-only, GPU-free validator over those records).

Relation to the earlier rounds:

| round | role for r5 |
|---|---|
| R2 / R3 (`../r2/`, `../r2/round3/`) | R3 W2 supplied the **cross-window legacy anchor** (0.6323 / 0.6357 ms). Read-only. |
| R4 (`../r4/`) | supplied the **cross-window new-side quotes** (dense-iree e2e 6.667 / 6.050 ms, pure 1.447 / 1.593 ms) and the 87c7562 ETL-shadow protocol. Read-only; r5 re-proves the same agreement gate into `results/backend_agreement/`. |
| **R5 (this directory)** | the **same-window 3-way anchor** (legacy / dense-iree e2e / dense-iree pure) plus the per-call decomposition, the GPU-side mechanism, controlled-removal floors and the engine-vs-frozen A/B. |

**Records of record:** the r4/r2 directories are immutable and gate-checked — never rewrite them, and
never treat an r5 number as an r4 or r2 number (the ETL checkout moved: r5 measures etl `96df7da8…`,
r4's after-half measured a `87c7562` shadow).

---

## 1. The question list (Q1–Q8)

| # | question | answered by |
|---|---|---|
| **Q1** | Where does one warm dense-route `evaluate` (~5.4–6.9 ms) spend its time, bucket by bucket? | `decomp_probe.py` → `results/q1_decomp/decomp_{f64,f200}.json` |
| **Q2** | What does the compiled dense graph actually put on the GPU, and why does the pure forward cost 1.2–1.5 ms? | `gpu_probe/dispatch_dump.py`, `gpu_probe/run_rows.sh`, `gpu_probe/kineto_attempt.py` → `results/q2_gpu/` |
| **Q3** | Is there an **engine run-path tax** (engine seam vs the frozen `build_dense_runner(...).run()` on the same artifact)? | `ab_probe.py` → `results/q3_ab/ab_{f64,f200}.json` |
| **Q4** | Does the r4 cross-window ≈9.5–10.5× gap survive **one serialized same-window** comparison? | `window_probe.sh` → `results/q4_window/` (+ `results/smoke/`, `results/backend_agreement/`) |
| **Q5** | What are the **top host costs** inside the call? | Q5 section of `results/q1_decomp/SUMMARY.md` (same records as Q1) |
| **Q6** | Controlled-removal **floors**: what is left when each component is memoized/pre-staged/dropped? | `floor_probe.py` → `results/q6_floors/rerun_1/floor_{f64,f200}.json` |
| **Q7** | Is compile churn / cache-miss recompilation a warm-call contributor? | Q7 section of `results/q1_decomp/SUMMARY.md` (cache counts + the deliberate cold rebuild) |
| **Q8** | Is the cost a **harness artifact**? | `minimal_loop.py` → `results/q8_minimal/minimal_{f64,f200}.json` |

Headline (same window, warm medians, 64 / 200 trees): legacy **0.6461 / 0.6372 ms**, dense-iree e2e
**5.3823 / 5.7912 ms**, dense-iree pure **1.2142 / 1.4612 ms** ⇒ e2e/legacy **8.33× / 9.09×**,
pure/legacy **1.88× / 2.29×**, pure = 23 % / 25 % of the e2e call.

---

## 2. Method and hard rules

Every cell in this round follows the R2/R3/R4 runner rules:

* **Monkeypatching only.** Probes wrap the dense chain's own call sites at runtime
  (`probe_lib.install_dense_wraps`, `probe_lib.patch_etl_tensor_to`, module proxies for
  `etl.from_numpy` / `etl.run`). `src/evogp` is never edited and the foreign `etl` checkout is
  never written. If a `src/` / `etl` change looked necessary it would be recorded as
  `NEEDS-PARENT-ACTION`, not made here.
* **One process per cell** — the ETL `CompileCache` and the dense graph cache are per-process, so
  `first_call_ms` / `construct_total_ms` belong to that cell alone and the warm median is steady
  state by construction.
* **Exactly ONE `flock -w 7200 /tmp/gpbench_critical.lock` per invocation, never nested**, with
  `timeout -s INT -k 30 <cap>` **inside** the lock so lock-wait time is never charged to a cell.
* **GPU 6 only**, `CUDA_VISIBLE_DEVICES=6`; GPU 7 has a hardware ECC fault and is never used
  (the runners refuse it).
* **Project venv** `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2, torch 2.6.0+cu124);
  there is no `python` on `PATH`.
* **Repo-root CWD** (fixtures and records are repo-relative; the cell logs record absolute fixture
  paths).
* `ulimit -n 65536` first (IREE opens hundreds of fds); `ETL_DISABLE_LOCATIONS=1` (≈4× faster ETL
  tracing, not part of the compile-cache key); `--mem-guard-mb 8192` on every `bench_gp.py` cell.
* New-side route env (primary measurement identity):
  `PYTHONPATH=src EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`
  (the pure/row cells use `EVOGP_EVAL_MODE=dense`, which resolves to the same dense evaluator).
  Legacy-side cells: `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`.
* **Records are immutable evidence.** A re-run writes into a **fresh** directory (`rerun_1/`, a
  `--result-dir`, a new `--json-out`); never overwrite a committed JSON/log. Every record stamps
  its own identity (evogp commit, `etl.__file__` + ETL repo HEAD, env knobs, GPU sample, counts,
  timestamps).
* **`/tmp` is 97 % full** (≈16 GB free): keep capture artifacts small, prefer repo-side records,
  clean up scratch (big `.qdstrm`/`.nsys-rep`/IR dumps are deliberately not committed).
* **≤2 attempts per blocked tool, then fall back and record.** nsys/ncu are blocked (see §6); the
  one cheap kineto retry was allowed and succeeded.

---

## 3. Run order (and exactly what was run)

Phases are executed in this order; each script is an independent locked invocation.

### P0 — environment freeze (no GPU work)

Evidence: `results/env_probe.txt`, `results/tool_recon/` (see §6). No script; the facts are captured
by hand into committed text records.

### Q4 — the same-window anchor window (GPU)

```bash
# 3 pre-window smoke cells -> results/smoke/
bash test/benchmark/r5/window_probe.sh --group smoke
# the 9 window cells -> results/q4_window/   (then the post-window control into rerun_1/)
bash test/benchmark/r5/window_probe.sh --group window
bash test/benchmark/r5/window_probe.sh --group window \
  --only q4_leg64,q4_eval64_iree,q4_eval64_iree_probe \
  --result-dir test/benchmark/r5/results/q4_window/rerun_1
# IREE agreement pre-flight (mandatory before quoting any IREE number)
bash test/benchmark/r4/run_agreement_r4.sh --agree-dir test/benchmark/r5/results/backend_agreement --timeout 2700
/mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r4/check_agreement_r4.py \
  --agree-dir test/benchmark/r5/results/backend_agreement
```

`window_probe.sh` CLI: `--group smoke|window|all`, `--only id1,id2` (overrides `--group`),
`--result-dir DIR` (use `rerun_<n>/`), `--list`, `--dry-run`, `--help`; env overrides `PY`,
`CUDA_VISIBLE_DEVICES`, `LOCK`/`FLOCK_WAIT`, `NEW_PYTHONPATH`, `LEG_PYTHONPATH`, `SHADOW_DIR` /
`SHADOW_PYTHONPATH`, `ETL_REPO`, `WINDOW_RESULT_DIR`, `CAP`/`CAP_LEGACY`,
`DEVIATION_NOTE="…"` (appended to `window_meta.txt`). Exit codes: 0 all cells OK (bench_gp exit 3 =
guard-skip counts as OK), 1 a cell failed, 2 usage/precondition.

The 9 window cells: `q4_leg64`, `q4_leg200`, `q4_eval64_iree`, `q4_eval200_iree`,
`q4_pure64_iree`, `q4_pure200_iree`, `q4_eval64_iree_probe` (`--stage-probe` bridge),
`q4_leg64_bracket` (drift bracket), `q4_eval64_iree_shadow87c7562` (ETL `87c7562` bridge, opt-in).

### Q1 / Q5 / Q7 / Q8 — decomposition + controls (GPU, 4 cells)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python; R5=test/benchmark/r5
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' \
  '"$R5"'/decomp_probe.py --fixture test/goldens/forest_random.npz --eval-reps 25 --sync-reps 5 \
    --plain-reps 5 --cold-reps 1 --tag decomp_f64 \
    --json-out '"$R5"'/results/q1_decomp/decomp_f64.json \
    --log-out  '"$R5"'/results/q1_decomp/decomp_f64.log'
# without the last three flags: --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag decomp_f200
# cProfile cross-check (same loop, instrumentation OFF, text table out)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' \
  '"$R5"'/decomp_probe.py --fixture test/goldens/forest_random.npz --cprofile --eval-reps 25 \
    --json-out '"$R5"'/results/q1_decomp/cprofile_f64.json \
    --cprofile-out '"$R5"'/results/q1_decomp/cprofile_f64.txt'
# Q8 minimal loop (no bench_gp.py, no probe, no monkeypatch)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' \
  '"$R5"'/minimal_loop.py --fixture test/goldens/forest_random.npz --reps 25 \
    --json-out '"$R5"'/results/q8_minimal/minimal_f64.json \
    --log-out  '"$R5"'/results/q8_minimal/minimal_f64.log'
# f200: --fixture test/benchmark/fixtures/forest200_tree64_random.npz (+ minimal_f200 paths)
```

`decomp_probe.py` flags: `--fixture` (required), `--eval-reps` (25), `--sync-reps` (5),
`--plain-reps` (5), `--cold-reps` (1), `--no-controls`, `--cprofile`, `--cprofile-out`, `--json-out`,
`--log-out`, `--tag`.
`minimal_loop.py` flags: `--fixture` (required), `--reps` (25), `--json-out`, `--log-out`.

### Q2 — GPU-side mechanism (GPU, ~10 cells + 1 profiler attempt)

```bash
R5=test/benchmark/r5; PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
ENVV='PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1'
# (a) compile-side dispatch dumps (--iree-flow-dump-dispatch-graph etc. via ETL_IREE_COMPILE_ARGS)
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env '"$ENVV"' '"$PY"' \
  '"$R5"'/gpu_probe/dispatch_dump.py --fixture test/goldens/forest_random.npz --mode pure \
  --invocations 3 --out-dir '"$R5"'/results/q2_gpu/dumps/f64_pure \
  --json-out '"$R5"'/results/q2_gpu/dumps/f64_pure.json --tag f64_pure'
#   variants: --mode e2e … f64_e2e | f200 fixture … f200_pure
#             --data-rows 8 … f64_r8 | --data-rows 256 … f64_r256 | f200 + --data-rows 128 … f200_r128
# (b) the pure-forward row-scaling series (frozen r4 pure64_dense_iree invocation + --data-rows N)
bash test/benchmark/r5/gpu_probe/run_rows.sh [--only id1,id2] [--list] [--dry-run]
# (c) the one allowed profiler attempt (torch.profiler / kineto) + offline reduction
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=dense EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 '"$PY"' \
  '"$R5"'/gpu_probe/kineto_attempt.py --fixture test/goldens/forest_random.npz --mode pure \
    --invocations 30 --out-dir '"$R5"'/results/q2_gpu/kineto_f64 \
    --json-out '"$R5"'/results/q2_gpu/kineto_f64.json --tag kineto_f64'
python3 test/benchmark/r5/gpu_probe/trace_analyze.py \
  --trace test/benchmark/r5/results/q2_gpu/kineto_f64/kineto_trace.json.gz --invocations 30 \
  --json-out test/benchmark/r5/results/q2_gpu/kineto_f64/trace_facts.json \
  --text-out test/benchmark/r5/results/q2_gpu/kineto_f64/trace_facts.txt
```

`gpu_probe/dispatch_dump.py` flags: `--fixture` (required), `--out-dir` (required),
`--mode e2e|pure` (pure), `--invocations` (3), `--data-rows N`, `--compile-args` (override
`ETL_IREE_COMPILE_ARGS`), `--json-out`, `--log-out`, `--tag`.
`gpu_probe/run_rows.sh` flags: `--only`, `--list`, `--dry-run`; ids
`f64_{r8,r16,r32,r64,r128,r256}`, `f200_{r64,r128}`; env overrides `PY`, `CUDA_VISIBLE_DEVICES`,
`TIMEOUT_S`, `LOCK`, `FLOCK_WAIT`, `NEW_PYTHONPATH`, `ROWS_RESULT_DIR`.
`gpu_probe/kineto_attempt.py` flags: `--fixture` (required), `--out-dir` (required),
`--mode pure|e2e` (pure), `--invocations` (30), `--json-out`, `--log-out`, `--tag`.
`gpu_probe/trace_analyze.py` flags: `--trace` (required), `--invocations` (0 = infer),
`--json-out`, `--text-out` — stdlib only, runs with plain `python3`.

### Q6 — controlled-removal floor matrix (GPU, 2 cells + 1 earlier control)

```bash
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r5/floor_probe.py \
    --fixture test/goldens/forest_random.npz --reps 25 --tag floor_f64 \
    --json-out test/benchmark/r5/results/q6_floors/rerun_1/floor_f64.json \
    --log-out  test/benchmark/r5/results/q6_floors/rerun_1/floor_f64.log'
# f200: --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag floor_f200 (same command)
```

`floor_probe.py` flags: `--fixture` (required), `--reps` (25, call 0 cold), `--control-reps` (5,
M0P), `--micro-reps` (20, upload micro-probe), `--no-micro`, `--json-out`, `--log-out`, `--tag`.
The earlier, contended `results/q6_floors/floor_f64.json` is kept as the contention control; the
**record of record is `rerun_1/`**.

### Q3 — engine-call vs frozen-runner A/B (GPU, 2 cells)

```bash
flock -w 7200 /tmp/gpbench_critical.lock bash -c 'ulimit -n 65536; timeout -s INT -k 30 900 env \
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=6 ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=iree \
  EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1 \
  /mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r5/ab_probe.py \
    --fixture test/goldens/forest_random.npz --reps 25 --tag ab_f64 \
    --json-out test/benchmark/r5/results/q3_ab/ab_f64.json \
    --log-out  test/benchmark/r5/results/q3_ab/ab_f64.log'
# f200: --fixture test/benchmark/fixtures/forest200_tree64_random.npz --tag ab_f200 (same command)
```

`ab_probe.py` flags: `--fixture` (required), `--reps` (25), `--json-out`, `--log-out`, `--tag`.

### Write-up + validation

`RCA.md` (ranked mechanisms with measured/inferred/unknown labels, same-window tables, floors,
open questions) plus

```bash
python3 test/benchmark/r5/check_r5.py        # stdlib-only, GPU-free, exit 0/1 (see §7)
```

---

## 4. Script reference (all analysis-only)

| script | what it does | GPU |
|---|---|---|
| `window_probe.sh` | the Q4 runner: one `bench_gp.py` process per cell (legacy eval64 / dense-iree eval64 / dense-iree pure / probe bridge / bracket / shadow bridge), serialized on the shared lock, writes `<id>.json` + `<id>.log` + `window_meta.txt` | yes |
| `decomp_probe.py` | Q1/Q5/Q7: monkeypatch timers over the dense call sites ⇒ ≥8 exclusive buckets, reconciliation, controls (region-sync, uninstrumented plain, cache-clear cold), compile-cache counters; `--cprofile` cross-check | yes |
| `minimal_loop.py` | Q8: 25 raw `problem.evaluate` calls with no harness/probe import | yes |
| `floor_probe.py` | Q6: arm chain `M0 → M1 → M2 → M3 → M4 → M4V → M5` plus `M0R`/`M3R`/`M0P` controls, per-step deltas vs the Q1 attribution, upload micro-probe, drift/overhead controls | yes |
| `ab_probe.py` | Q3: arms `A1,A2,A3,A4,L1,L2,R1,R2,E1 (+A3_plain_control)` in one process on one population/artifact; freshness vs run-path separation | yes |
| `probe_lib.py` | shared helpers (exclusive `Profiler`, `patch_attr`, `EtlProxy`, `patch_etl_tensor_to`, `install_dense_wraps`, `build_eval_case`, `identity_stamp`, `host_load`) — imported by the Q3/Q6/Q8 probes | yes |
| `gpu_probe/dispatch_dump.py` | Q2(a): builds the dense graph under `ETL_IREE_COMPILE_ARGS` dump flags, parses `dispatch_graph.dot` / `scheduling_stats.csv` / linked PTX into a record (nodes, edges, dispatch count, longest chain, copies, transient size, compile ms, warm calls) | yes |
| `gpu_probe/run_rows.sh` | Q2(c): the row-scaling series (8 cells + `rerun_1/` controls) | yes |
| `gpu_probe/kineto_attempt.py` | Q2(d): the one `torch.profiler` (kineto/CUPTI) attempt on the pure route | yes |
| `gpu_probe/trace_analyze.py` | Q2(d): stdlib reduction of the chrome trace to `trace_facts.json`/`.txt` | no |
| `check_r5.py` | the record validator (§7) | no |

`--stage-probe` (from `../stage_probe.py`) is used only by the `q4_eval64_iree_probe` bridge cell;
on the dense route it is **degenerate** (its cost lands in the uninstrumented `misc` bucket and it
reports `build_calls=0`) — it is a probe-tax bridge, not a Q1 instrument.

---

## 5. Evidence map (`results/`)

| path | what it holds |
|---|---|
| `env_probe.txt` | P0 environment freeze: evogp commit/branch/status, ETL repo + HEAD + clean status, venv/torch, driver, GPU table (incl. the GPU-7 ECC note), nsys/ncu state, lock/rlimit/env knobs, fixtures + sha256, `/tmp` capacity. |
| `tool_recon/` | profiler reconnaissance: `import_log.txt` (nsys QDSTRM import failure, verbatim) + `README.md` (tool identities, why Q2 is measured indirectly). |
| `smoke/` | the 3 pre-window sanity cells (`smoke_leg64`, `smoke_eval64_iree`, `smoke_pure64_iree`; short reps) + their `window_meta.txt`. |
| `q4_window/` | the 9 Q4 window cells (`q4_*.json` + `.log`), the committed `window_meta.txt` (identity, per-cell start/end/exit, co-tenant samples, deviations), `SUMMARY.md` (headline tables + verdict). |
| `q4_window/rerun_1/` | the post-window control: `q4_leg64`, `q4_eval64_iree`, `q4_eval64_iree_probe` back-to-back + its `window_meta.txt` (with the DEVIATION_NOTE). |
| `backend_agreement/` | the six-arm IREE agreement pre-flight re-run for r5 (`fixture{64,200}_{numpy,iree,iree_spec}.json` + `_values.npy`), the runner stdout and the checker output (`check_agreement_r4.out.txt`, `gate: PASS`). |
| `q1_decomp/` | `decomp_{f64,f200}.json` + `.log` (per-call buckets, counts, controls, cache counters, cold rebuild), `cprofile_f64.json` + `.txt` (instrumentation-free cross-check), `SUMMARY.md` (Q1/Q5/Q7/Q8 write-up). |
| `q8_minimal/` | `minimal_{f64,f200}.json` + `.log` (25-call plain loop, no harness) + `SUMMARY.md` (the falsification verdict). |
| `q2_gpu/` | `dumps/<cell>/` (`dispatch_graph.dot`, `scheduling_stats.csv`, `executables/` linked PTX + manifest, `dump.log`) + `dumps/<cell>.json` (parsed record); `rows/` (8 primary cells + `rows_meta.txt`/`rows_manifest.txt`) + `rows/rerun_1/` (5 re-run/control cells); `kineto_f64.json` + `kineto_f64/` (key averages, log, 368 KB chrome trace, reduced `trace_facts.json`/`.txt`); `SUMMARY.md`. |
| `q6_floors/` | `rerun_1/floor_{f64,f200}.json` + `.log` (the floor matrix, record of record) + `SUMMARY.md`; the earlier contended `floor_f64.json` remains as the contention control. |
| `q3_ab/` | `ab_{f64,f200}.json` + `.log` (10 arms + 15 recomputed deltas) + `SUMMARY.md` (the no-engine-tax verdict). |

---

## 6. Identity and environment (what the r5 numbers measured)

| item | value |
|---|---|
| ETL (primary) | **editable install of the foreign checkout `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9`**, working tree clean at round start; stamped per record via `etl.__file__` + read-only `git -C … rev-parse HEAD` |
| ETL shadow (bridge only) | `/tmp/evogp_etl_patched_87c7562` — the r4 after-half's patched checkout, used by the single `q4_eval64_iree_shadow87c7562` cell, never as the primary environment |
| venv / torch | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2), torch 2.6.0+cu124, CUDA 12.4 |
| host driver | 535.247.01 (8× RTX A6000; GPU 6 used, GPU 7 hardware-ECC-faulty and refused) |
| profilers | **unavailable** — nsys 2022.4.2.1 collects but its QDSTRM import/analysis fails (`Unknown runtime API function index: 440`) and ncu 2022.3 refuses (`API call is not supported in the installed CUDA driver`); evidence in `results/tool_recon/`. Q2 therefore rests on the compile-side dumps, the row series and the kineto attempt (which **did** see IREE kernels). |
| fixtures | f64 = `test/goldens/forest_random.npz` (sha256 `0bfb0bbf…ff9ca0`, 64 trees, 64 rows); f200 = `test/benchmark/fixtures/forest200_tree64_random.npz` (sha256 `6789ef61…9741b9`, 200 trees, 64 rows) |
| legacy engine | `/tmp/evogp_legacy_bench` (built `b613503` + `evogp_cuda…so`), `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils`; import requires `import torch` first; `/tmp`-scoped, rebuild via `../legacy/setup_legacy.sh` |

Host caveat of record: the box is shared — GPU 6 carried three foreign training processes (~30 GB)
and the host loadavg sat at 290–375 on 128 cores during the sessions. Absolute per-cell numbers ride
a ±10–20 % band (and worse during contention bursts); **within-process / within-window comparisons
are the robust metrics**, and all of them are recorded in the JSONs.

---

## 7. Validation: `check_r5.py`

`check_r5.py` is a **stdlib-only, GPU-free** validator over the committed r5 evidence. It imports
nothing from `evogp`, `etl`, `torch` or `numpy` and resolves every path relative to its own location,
so it runs from any CWD with the system `python3`.

```bash
python3 test/benchmark/r5/check_r5.py                 # validate the committed records
python3 test/benchmark/r5/check_r5.py --results DIR   # validate a fresh --result-dir tree instead
```

It exits **0** when nothing failed and **1** otherwise, printing a per-failure line and a final
`N check(s), M failure(s)` summary. What it checks:

* **existence** of every record of the evidence map (window cells, decomposition, floors, A/B,
  minimal loop, dumps/rows/kineto, agreement pre-flight, env/tool-recon text);
* **structural stamps** — evogp commit, ETL identity (`etl_repo_head` / `etl_version`), env knobs,
  `CUDA_VISIBLE_DEVICES=6`, `ETL_DISABLE_LOCATIONS=1`, task/side, 25-call/24-warm shapes,
  per-cell medians recomputed from the stored per-call arrays;
* **the Q4 headline recomputed from the JSONs** — 4-dp agreement with 0.6461 / 0.6372 /
  5.3823 / 5.7912 / 1.2142 / 1.4612 ms and ratios e2e/legacy 8.33× / 9.09×, pure/legacy 1.88× /
  2.29× within ±2 %; `window_meta.txt` shows `ok=9 skipped=0 failed=0` and every cell `exit 0`;
  the bridge cell's `--stage-probe` tax (`probe_span_ms − total_ms`) stays ≤ 0.5 ms/call; the
  post-window `rerun_1/` control repeats the legacy anchor and the probe/plain pair within ±20 %;
  and the two pure cells' `output_checksum.sha256` are bit-equal to the r4 after-records;
* **the decomposition reconciles** — ≥8 buckets, Σ(median self) == `sum_exclusive_ms`,
  `sum_exclusive / median_total` inside 0.90–1.10, recorded reconciliation ≥ 99 %, the dominant
  buckets present and (with payload uploads) ≥ 50 % of the call; Q7: the whole-probe counters
  (`route.dense_calls` = 25 measured + 5 region-sync + 1 cold) show exactly one `etl.run`, one
  `build_cached` prologue, one dense decode and ten payload uploads per dense call, one cache miss,
  and a multi-second cold rebuild;
* **the Q6 floors** — all six arms M0–M5 with 25 calls/24 warm, only the safe inequalities
  (`M0 ≥ M3`, `M0 ≥ M4`, `M0 ≥ M5`, `M5 < M0`, and upload removal `M3 < M2`), the `floors` block and
  all six stored deltas recomputed from the stored arm medians, fitness digests equal across the
  equivalent arms, no compile churn (misses == 1 per artifact);
* **the Q3 A/B** — arms `A1–A4, L1, L2, E1` (+ the `A3_plain_control` and the `R1`/`R2` replication
  controls), the **no-engine-tax** claim (`|A2 − A3| ≤ 0.35 ms` on both the total and the `etl.run`
  span), freshness `L1 − L2 > 1 ms`, `A4 − A3 > 1 ms`, `E1 > A3`, all 15 stored deltas recomputed;
* **the Q2 GPU records** — dump files on disk, dispatch/executable/`Copies` counts (249/125/61 for
  f64, 253/127/62 for f200) with `Dispatches == dispatch_nodes`, the kineto facts
  (7470 kernels / 1830 memcpys / 30 invocations × 310 events), and the row series' schema;
* **the Q8 control** lands inside the Q4 band (±35 % of the same-window e2e median).

**Soft checks (notes, never failures):** `RCA.md` and this `README.md` are reported if absent, and
`RCA.md`'s required section headers (TL;DR, Method, anchor tables, decomposition, GPU, A/B, Floors,
Ranked mechanisms, What-would-close, Open) are reported if missing — the validator stays green while
the write-up is still in flight.

---

## 8. Known deviations and gotchas (read before quoting a number)

1. **The `drift bracket` did not measure drift.** `q4_leg64_bracket` reads +177.5 % over
   `q4_leg64` — a co-tenant contention event (a 90 % GPU-utilisation sample), not protocol drift.
   The post-window `rerun_1/` control supplies the drift answer instead (legacy −4.2 %, probe
   −0.7 %).
2. **The plain e2e cell repeats loosely (+26.9 %).** Same-window `q4_eval64_iree` = 5.3823 ms vs
   its post-window repeat 6.8312 ms. That is why the honest gap statement is **≈8–11×, centred on
   ~9.5×**, and why no single cell should be quoted to better than ±10 %.
3. **`q4_eval64_iree_probe` is not a Q1 instrument.** On the dense route `--stage-probe` is
   degenerate (`build_calls=0`, the cost lands in `misc`); the probe's own tax is nil (median
   `probe_span_ms − total_ms` = +0.028 / +0.029 ms) and the window's +25.9 % probe-vs-plain delta is
   window drift, not probe overhead (the calm repeat shows −1.4 %).
4. **`rows/` has two record generations.** `f64_r8` (3.923 ms, construct 38.8 s) is a contention
   outlier; its re-run `rows/rerun_1/f64_r8b` (1.382 ms) is the value used in the series. The
   `rerun_1/` cells are separate records, never overwrites, and are not in `rows_manifest.txt`.
5. **The f200 row-scaling rise is open.** The tight-block protocol reads +70 % from r64 to r128
   while the single-sync dump protocol reads +12 %; the attempted explanations (dispatch structure,
   element count, transient size) are all contradicted by the dumps. It is labelled *measured,
   unexplained* in the record, not resolved.
6. **The Q6 `floor_f64` primary record is contended** (`results/q6_floors/floor_f64.json`), kept as
   a control; the *record of record* is `results/q6_floors/rerun_1/floor_{f64,f200}.json`
   (M0 7.2289 / 5.0725 ms). `M1` on f64 sits *above* `M0` (contention + memoisation overhead) —
   do not read the M1/M2 ordering as physics, and do not order M3 vs M4 (f200 M4 > M3).
7. **`decomp_*.json`'s top-level `cache_final` is post-cold** (0 hits / 1 miss): the probe's
   deliberate cache-clear rebuild runs last, so the warm hit counts live in
   `controls.plain.cache` (34 hits / 1 miss) and in `warm.counts.build_dense_build_cached`.
8. **`cprofile_f64.json` carries no timings** — it is the pointer record (`cprofile_text_written`);
   the numbers are in `cprofile_f64.txt`.
9. **No kernel-level counters.** nsys/ncu are blocked on this box (§6); kineto inflates wall time
   (profiled span 2.55 ms vs 1.21–1.47 ms unprofiled), so only its *counts* and *kernel durations*
   are quoted as measurements.
10. **`nanobind: leaked …` chatter at every IREE process exit** is stock-ETL teardown noise; exit
    code is 0 and no timed region is affected.
11. **Mixture of record generations (recorded in each file):** evogp commit stamps differ across
    phases (`3f79356`, `1a14b77`, `247c1e0`, `48dd1bc`, `682733e`, `d23718a`, …) because the r5
    tooling was being written while the phases ran — the *source under measurement* (`src/evogp`)
    is upstream code in every one of them; only the r5 files themselves differ. `env_probe.txt`
    (`d23718a`) is the round-start freeze.
12. **`rows_meta.txt` says "(2 dirty files)"** — that is the r5 scaffolding being written at the
    time, not an engine change.

---

## See also

* `../CONTEXT.md` — the benchmark node (round verdicts, harness rules, lock/GPU policy).
* `../REPORT.md` — analysis of record for R1–R4 (§8 holds the ranked levers; the parent adds the
  RCA pointer).
* `../r4/` — the pure-forward protocol (`run_pure.sh`, `pure_forward.py`) and the after-records the
  r5 pure cells are compared with; `../r2/round3/` — the legacy anchor records.
* `../../src/evogp/tree/ENGINE_SPEC.md`, `../../src/evogp/tree/warm_path_findings.md`,
  `../../src/evogp/tree/RUNNER_HOOKS.md` — engine semantics, the wave-1 stage record, and the
  frozen runner contract.
* `RCA.md` — the round's deliverable analysis.
