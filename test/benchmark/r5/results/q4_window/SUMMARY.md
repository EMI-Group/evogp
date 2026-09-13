# `results/q4_window/SUMMARY.md` — r5 Q4: the same-window anchor window

Phase 0 + Q4 of the r5 RCA round (`test/benchmark/r5/`), executed 2026-09-13 20:29–20:41 HKT on GPU 6
of the shared `bmi-9` box. Analysis-only: no `src/evogp` change, no write in the foreign ETL repo, no
write outside `r5/`.

**Question this file answers:** the r4 verdict quotes an old-vs-new gap of ≈9.5–10.5× (dense-iree
`problem.evaluate` 6.667 / 6.050 ms vs legacy CUDA anchors 0.6323 / 0.6357 ms) built from *cross-window*
numbers. R5 measures all three quantities — legacy end-to-end, ETL dense-iree end-to-end, ETL dense-iree
pure internal forward — in **one serialized window**, adds a drift bracket and a post-window control, and
re-runs the IREE agreement pre-flight as the gate.

## 0. Session identity (what was actually measured)

| item | value (evidence) |
|---|---|
| evogp source | commit `d23718a52a8fdf0d12bf952472a637f233cca4f2` + r5 files only (worktree `worker_T1_A65`), `src/evogp` unmodified — `env_probe.txt` |
| ETL (primary) | **editable install of the foreign checkout `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9`, working tree CLEAN at round start**; every cell log carries `etl: … rev=96df7da… file=/mnt/local-ssd/bchuang/etl/etl/__init__.py` (`env_probe.txt`, `window_meta.txt`, `<cell>.log`) |
| legacy engine | built `b613503` checkout `/tmp/evogp_legacy_bench` (`evogp_cuda.cpython-311-x86_64-linux-gnu.so`), `PYTHONPATH=/tmp/evogp_legacy_bench/src:/tmp/evogp_t1a263/siteutils` |
| optional ETL shadow | `/tmp/evogp_etl_patched_87c7562`, identity re-proved by sha256 pins (see §5) |
| env of every cell | `CUDA_VISIBLE_DEVICES=6`, `ETL_DISABLE_LOCATIONS=1`, repo-root cwd, project venv `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11.2, torch 2.6.0+cu124), `ulimit -n 65536`, **exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per cell with `timeout -s INT -k 30 <cap>` INSIDE the lock**, one fresh process per cell |
| new-side route | `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1` (=`auto` ⇒ dense evaluator) |
| fixtures | `test/goldens/forest_random.npz` sha256 `0bfb0bbf…ff9ca0` (64 trees), `test/benchmark/fixtures/forest200_tree64_random.npz` sha256 `6789ef61…9741b9` (200 trees) |
| profilers | unavailable — nsys QDSTRM import fails (`Unknown runtime API function index: 440`), ncu refuses on driver 535.247.01; see `../tool_recon/README.md`. Kernels are NOT profiled in this phase |

## 1. The same-window 3-way table (headline)

Warm = `median_warm_ms` = median of the 24 post-cold `problem.evaluate()` calls for the eval64 cells
(25 calls, call 0 cold) and `pure.median_per_invocation_ms` = median of 5 blocks × 100 invocations
(10 warmup invocations) for the pure cells. All six numbers below come from six different processes in
the window `20:30:54 – 20:33:11`.

| fixture | (a) legacy end-to-end | (b) dense-iree end-to-end | (c) dense-iree PURE forward | **b/a** | **c/a** | **c/b** | host glue = b−c |
|---|---|---|---|---|---|---|---|
| 64 trees | **0.6461 ms** | **5.3823 ms** | **1.2142 ms** | **8.33×** | **1.88×** | 0.226 | 4.168 ms (77.4 % of b) |
| 200 trees | **0.6372 ms** | **5.7912 ms** | **1.4612 ms** | **9.09×** | **2.29×** | 0.252 | 4.330 ms (74.8 % of b) |

Cold / first-call side (never mixed into the warm medians):

| fixture | legacy `first_call_ms` | dense-iree `first_call_ms` | pure construct (compile) |
|---|---|---|---|
| 64 | 22.0 | 12497.0 | 12216.2 ms (`construct_total_ms`, `compile.path="frozen"`, 1 cache miss) |
| 200 | 22.8 | 13167.7 | 12600.2 ms (same shape) |

Cross-window r4 quotes for orientation (not re-measured by r5, kept separate):
r4 headline 6.667 (f64) / 6.050 (f200) ms vs legacy anchors 0.6323 / 0.6357 ms ⇒ 10.54× / 9.52×;
r4 pure (after-records) 1.4468 / 1.5930 ms. So in this window the legacy side matches r3/r4 (within
+2.2 % / +0.2 %) while the new-side end-to-end cell reads ~19 % lower than r4's quote — see §4 for the
repeatability of exactly that cell, which is the reason §9 states the verdict as a band and not a point.

## 2. All cells of the window (exit codes, timings, fds)

| cell | side | what it measures | exit | warm median | first call | fds (start/after-cold/end) |
|---|---|---|---|---|---|---|
| `q4_leg64` | old | legacy e2e, 64 trees, 25 evals | 0 | 0.6461 ms | 22.0 ms | 39/59/59 |
| `q4_leg200` | old | legacy e2e, 200 trees | 0 | 0.6372 ms | 22.8 ms | 39/59/59 |
| `q4_eval64_iree` | new | ETL dense-iree e2e, 64 trees | 0 | 5.3823 ms | 12497.0 ms | 39/91/91 |
| `q4_eval200_iree` | new | ETL dense-iree e2e, 200 trees | 0 | 5.7912 ms | 13167.7 ms | 39/91/91 |
| `q4_pure64_iree` | new | pure forward, 64 trees (5×100 inv) | 0 | **1.2142 ms** | 12216.2 ms construct | 39/91/91 |
| `q4_pure200_iree` | new | pure forward, 200 trees | 0 | **1.4612 ms** | 12600.2 ms construct | 39/91/91 |
| `q4_eval64_iree_probe` | new | `q4_eval64_iree` + `--stage-probe` | 0 | 6.7774 ms | 11819.6 ms | 39/91/91 |
| `q4_leg64_bracket` | old | `q4_leg64` repeated at window end | 0 | **1.7931 ms** | 28.7 ms | 39/59/59 |
| `q4_eval64_iree_shadow87c7562` | new | `q4_eval64_iree` on the 87c7562 ETL shadow | 0 | 6.0165 ms | 10986.6 ms | 39/91/91 |

9/9 cells exit 0, no guard skips, no timeouts (caps 1800 s new / 600 s legacy; longest cell 30 s wall,
`q4_pure64_iree`).
The IREE fd count is **flat at 91** (after-cold = end) in every new-side cell — the leak documented for
the stock ETL in the r2/r3 era is fixed on `96df7da`.

Smoke (pre-window, `../smoke/`, `--eval-reps 3` / `--reps 1 --invocations 20`): all three cells exit 0
(legacy 0.7455 ms over 3 calls; dense-iree 10.2696 ms over 3 calls; pure 1.8764 ms over 20 invocations,
`output_checksum` sha256 `9132bea9…` already equal to the r4 after-record's). No invocation or env fix
was needed for the window; the only artifact is the nanobind "leaked function/instance" dump that the
IREE Python bindings print during interpreter shutdown (cosmetic, exit 0, visible in every iree log).

## 3. Probe tax (`q4_eval64_iree_probe` vs `q4_eval64_iree`)

| comparison | delta |
|---|---|
| in-window medians (5.3823 → 6.7774) | +1.3951 ms (**+25.9 %**) |
| adjacent control pair (§4: 6.8312 → 6.7328) | **−0.0985 ms (−1.4 %)** |
| **inside the probe cell**: `probe_span_ms` − `total_ms` (warm median) | **+0.028 ms per call (≈+0.4 %)** |

Verdict: **no resolvable `--stage-probe` tax** — the probe's own timed span exceeds the harness's
`total_ms` by 0.028 ms per call, and an adjacent-in-time pair of the same two cells (plain then probe,
17 s apart, same window conditions) puts them within 1.4 % of each other. The +25.9 % in-window delta is
therefore **cell-to-cell variation of the new-side end-to-end cell** (§4), not probe overhead; it is
consistent with the r4 note "probe tax ≲1 ms on eval64". Bridge value for the later decomposition
phases: in the probe cell the wrapped `etl.run` accounts for `run_ms` median **2.0805 ms** of the
6.777 ms call and the remaining **4.70 ms lands in the uninstrumented `misc` bucket** (the dense route is
outside the probe's wrap set — the documented r2/r4 degeneracy), i.e. the same "≈1/3 in-engine, ≈2/3
outside" split that the pure/e2e subtraction gives in §1.

## 4. Drift bracket and post-window control (repeatability of the anchors)

| run | cell | warm median | UTC+8 | note |
|---|---|---|---|---|
| window | `q4_leg64` (first) | 0.6461 ms | 20:30:57–20:30:59 | window anchor |
| window | `q4_leg200` | 0.6372 ms | 20:31:01–20:31:03 | matches r3/r4 legacy (0.6323/0.6357) |
| window | `q4_leg64_bracket` (last) | **1.7931 ms (+177.5 %)** | 20:32:52–20:32:56 | a *contention event*, not drift |
| control (`rerun_1/`) | `q4_leg64` | **0.6187 ms (−4.2 %)** | 20:39:36–20:39:38 | the anchor itself repeats |
| control | `q4_eval64_iree` | **6.8312 ms (+26.9 %)** | 20:39:41–20:39:55 | the new-side cell does NOT repeat tightly |
| control | `q4_eval64_iree_probe` | **6.7328 ms (−0.7 %)** | 20:39:58–20:40:12 | the probe cell repeats tightly |

* **The bracket is a transient contention event, not protocol drift.** Its whole warm series is uniformly
  ~1.75–1.80 ms (min 0.93 / max 1.97), and the co-tenancy sample taken at that boundary — the meta's
  between-cell sample written immediately before the bracket's cell header, i.e. right after the probe
  cell released its context — recorded the foreign tenant (`pid 2903622`, 30 486 MiB) bursting at
  **90 % GPU utilization**; the bracket's own `gpu (before)/(after)` samples read 0 % / 12 %, and every
  other utilisation sample in the window read 0–9 %. The legacy anchor re-run 8.7 min later in a calm
  window returns to 0.6187 ms, i.e. −4.2 % of the window's first anchor.
* **The new-side end-to-end cell is the least reproducible number in the window** (+26.9 % between
  window and control, same command, same ETL, same fixture, ~8.6 min apart), while the pure cell's e2e
  companion (8.61 / 8.32 ms) and the dedicated pure cells (1.214 / 1.461 ms) sit between the two.
* Consequence for quoting: the r5 same-window gap is **8.33× / 9.09×** against the calm window anchors
  and **11.04×** against the control anchor; the pure/legacy ratios are 1.88× / 2.29×. All of these are
  the same order as r4's 9.52–10.54×.

## 5. Shadow bridge to the r4 headline (`q4_eval64_iree_shadow87c7562`, optional cell)

Same command/env as `q4_eval64_iree`, `PYTHONPATH="src:/tmp/evogp_etl_patched_87c7562"`: exit 0,
warm median **6.0165 ms**, first call 10986.6 ms, fds flat 91. The cell log's identity line proves the
shadow was imported (`etl: … file=/tmp/evogp_etl_patched_87c7562/etl/__init__.py`), and the export's
sha256 pins re-match the committed r4 after-round probe exactly (`etl/__init__.py`
`ba2453545f27…abf43c`, `etl/jit.py` `ece92690e7d8…f2caf`, plus `build_cache.py`/`pipeline.py`/
`persist/cache.py`/`trace/_tree.py`). The shadow read is +11.8 % over the window's primary cell but
−11.9 % under the control's primary cell, i.e. **inside the new-side cell-to-cell spread measured in
§4 — the r4-revision-vs-current-ETL difference is not resolvable in this round's data.**

## 6. Pure-route facts and the r4 checksum cross-check

Both pure cells: `mode_requested=dense == mode_resolved=dense`, `runner_source=`
**`frozen-api:build_dense_runner`**, `backend=iree`, `per_row=False`, `pop == n_trees` (64 / 200),
`n_const_trees=0`, `n_graph_calls_per_run=1` (introspected), `invocations=100`, `warmup=10`, `reps=5`,
`compile.path="frozen"` with `cache_misses=1`, `cache_hits=0` and `decode_plan_ms`/`build_ms` `null`
(the frozen construction records only `construct_total_ms`, exactly as documented for the r4
after-records). Per-block medians: 1.2142 / 1.4612 ms; per-tree cost 18.97 / 7.31 µs.

`output_checksum.sha256` after all timing (out-of-band capture):
`9132bea9eec4bd82131a27eedd7352282ff9d4a0cb8bace7c0594fec6a6aa3b0` (64) and
`e77992ded140f01dabb130fac450284346eafd3027adf082ea12100d9299f322` (200) — **both EQUAL to the r4
after-records** `../r4/results/after/pure{64,200}_dense_iree.json`, with identical shapes
([64,64] / [200,64]) and NaN census (24 / 376). The pure route therefore measures the *same outputs* as
r4; the numbers differ only in time.

## 7. Agreement pre-flight (gate) — re-run in this window, into `../backend_agreement/`

`bash test/benchmark/r4/run_agreement_r4.sh --agree-dir test/benchmark/r5/results/backend_agreement --timeout 2700`
(20:33:25–20:38:12, immediately after the last window cell, same flood-lock discipline): **6/6 records
exit 0** (`fixture64_{numpy,iree,iree_spec}`, `fixture200_{numpy,iree,iree_spec}`; runner stdout in
`../backend_agreement/run_agreement_r4.stdout.txt`, per-record logs carry their `env:` line and the
fixture sha256). Checker
`python3 test/benchmark/r4/check_agreement_r4.py --agree-dir test/benchmark/r5/results/backend_agreement`
→ **exit 0, `gate: PASS (30 check(s), 0 failure(s))`, verdict `PASS-with-note` on both fixtures**
(verbatim output in `../backend_agreement/check_agreement_r4.out.txt`; identical verdict to r4's
committed pre-flight). The note: the three roles are not bit-identical (`values_sha256` differ) but every
pair is inside `rtol=atol=1e-4` — numpy vs iree `max|diff|` 7.629e-06 (f64) / 0.03125 abs, 1.806e-07 rel
(f200); `numpy vs iree_spec` bit-identical (max|diff| = 0) on both. **R5's IREE numbers are quotable with
that deviation recorded.** `fixture200_iree_spec` completed well inside the 2700 s cap.

## 8. Window timestamps and co-tenancy

| phase | window (UTC+8) | records |
|---|---|---|
| Phase 0 (no GPU): env freeze, tool recon, runner | 20:27–20:29 | `env_probe.txt`, `../tool_recon/*`, `window_probe.sh` |
| pre-window smoke (3 cells) | 20:29:47 – 20:30:39 | `../smoke/*` |
| **Q4 window (9 cells)** | **20:30:54 – 20:33:11** | this directory's `*.json`/`*.log` + `window_meta.txt` |
| agreement pre-flight | 20:33:25 – 20:38:12 | `../backend_agreement/*` |
| agreement checker | 20:38:16 – 20:38:45 | `../backend_agreement/check_agreement_r4.out.txt` |
| post-window control | 20:39:33 – 20:40:12 | `rerun_1/*` |

Co-tenancy (recorded per cell in every log and in `window_meta.txt`): GPU 6 carried a foreign training
process `pid 2903622` (30 486 MiB held, bursty) at **every** sample; the 17 796 MiB of free VRAM
satisfied `--mem-guard-mb 8192`; device utilisation samples were 0–9 % except **90 % at the
probe→bracket boundary** (§4), which is what inflated the bracket. No other benchmark process ran (the
shared `/tmp/gpbench_critical.lock` was taken and released per cell; 9 acquisitions in the window, never
nested).

## 9. Verdict: does the ≈9.5–10.5× gap hold same-window?

**Yes — the ≈10× order and the r4 band are reproduced within this box's own cell-to-cell spread.**
Measured in one serialized window with a fresh process per cell, the dense-iree end-to-end evaluation is
**8.33× (64 trees) / 9.09× (200 trees)** slower than the legacy CUDA engine, i.e. 5.38 / 5.79 ms against
0.646 / 0.637 ms, and the cross-window r4 quote (10.54× / 9.52×, 6.667 / 6.050 ms against
0.6323 / 0.6357 ms) sits **between** the r5 window pair and the r5 control pair (**11.04×** with the
post-window control's new-side cell, 6.831 ms) — the r4 headline is not an artefact of its cross-window
construction. The legacy anchor is trustworthy to ±5 % in this window (three runs: 0.646 / 0.637 /
0.619 ms, the fourth, contended, excluded and explained); on the new side the plain end-to-end cell
repeated to only +26.9 % (the probe cell, measured twice, to −0.7 %), so the honest statement of the gap
is **≈8–11×, centred on ~9.5×**. Below
the end-to-end gap, the pure internal forward is **1.88× / 2.29× the legacy end-to-end** and only
**23 % / 25 %** of the new end-to-end call, i.e. **three quarters of the observed gap is host-side work
outside the compiled graph** (4.17 / 4.33 ms of the 5.38 / 5.79 ms), which is where the later r5 phases
(Q1/Q5/Q7/Q8) must look; the remaining ~1/3-to-1/4 is in-engine (`etl.run` median 2.08 ms in the probe
cell; pure graph call 1.21 / 1.46 ms).

## 10. Deviations, caveats, open items

1. **Fixture paths are absolute** (resolved from the repo root as the phase objective requires), so the
   records carry the r5 worktree path; they were byte-identical to the committed fixtures (sha256 above).
2. **Cell order is the objective's order**, i.e. the probe cell is *not* time-adjacent to its plain twin;
   §3 resolves the resulting ambiguity with a post-window adjacent pair rather than reordering the window.
3. **The drift bracket caught a co-tenant burst** (90 % utilisation sample) — it measures contention, not
   protocol drift; the post-window control supplies the drift answer (−4.2 % on the legacy anchor).
4. **`--stage-probe` on the dense route is degenerate** (cost lands in `misc`, `build_calls=0`): the
   probe cell is a probe-tax bridge only. Its `run_ms`/`misc_ms` split is a bridge datum for the later
   decomposition phases, not a Q1 result.
5. **The nanobind shutdown dump** (the IREE Python bindings' reference-count "leaked function / leaked
   instance" report) appears at the end of every IREE process; exit code 0 and no timed region is
   affected.
6. **`--data-rows` row scaling, compile-side dispatch dumps and the torch.profiler (kineto) retry were
   NOT run in this phase** — they belong to Q2 (GPU mechanism) and are still open; kernel-level profiling
   is unavailable (§0, `../tool_recon/`).
7. `/tmp` capacity (97 % full) was respected: all records are inside the repo; no capture artifacts were
   written and the `/tmp/evogp_r5recon` scratch of the recon phase was not extended.
8. No NEEDS-PARENT-ACTION item: nothing here required a `src/evogp` or ETL change.

## Evidence index (this directory)

`window_meta.txt` (committed window record: identity, per-cell start/end/exit, co-tenant samples,
deviations) · `q4_*.json` + `q4_*.log` (9 cells; the log carries env, `etl:` identity, gpu samples and
the exact command) · `rerun_1/` (post-window control: `window_meta.txt` with the DEVIATION_NOTE + 3
cells) · `../smoke/` (3 pre-window smoke cells) · `../backend_agreement/` (6 agreement records + runner
stdout + checker output) · `../env_probe.txt` · `../tool_recon/` · `../window_probe.sh` (the single
runner used for smoke, window and control; unchanged between them).
