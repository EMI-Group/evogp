# Specialized warm path at `f227e4c` — A/B record, stage deltas and residual-gap attribution

Scope: the **specialized** (per-structure `etl.build_cached`) evaluation path of `etl_engine.py` at the tree-docs HEAD `f227e4c` — the decode refactor `e2899c0` (plus its fixes `eb386db` / `82edbad` / `5c96d1f`) and the `sr_fitness` host-rows reduction `f1bdec5` — measured against the pre-change checkout `850db27`.
The question is what those changes buy on the warm path and what is left before the ETL layer.
Everything here is warm/steady-state: compile (cold) numbers are reported separately and never mixed into a headline, per the project benchmark policy (`test/benchmark/REPORT.md`, `test/benchmark/r2/SPEC.md`).
Raw artifacts are **ephemeral** in `/tmp/a236_warm_ab/`: `summary.json` (machine-readable form of the tables below), `SUMMARY.md`, `run.sh`, `analyze.py` and 42 per-config `*.json` / `*.log` files; nothing from that directory is committed, and the A/B numbers below are the copy of record.
Box: RTX A6000, `CUDA_VISIBLE_DEVICES=6` (GPU 7 has a hardware ECC fault), venv `/mnt/local-ssd/bchuang/evox/.venv/bin/python`, `ETL_DISABLE_LOCATIONS=1`, one `bench_gp.py` process per config, `EVOGP_EVAL_MODE=specialized` forced on **both** sides so the dense policy can never enter an A/B cell (it is routed separately, §5).

## 1. How to re-run

`bench_gp.py` is the only harness: full-population warm calls (`--eval-reps 25`; headline = median of `per_call_ms[1:]`, compile excluded), **never** per-tree sampling, and one process per config so each cell's cold number and `CompileCache` are its own.
Per side (`run.sh` in the ephemeral dir expands to exactly this):

```
flock -w 7200 /tmp/gpbench_critical.lock timeout -s INT -k 30 3600 \
  bash -c 'ulimit -n 65536; cd "$1"; shift; exec "$@"' _ <ROOT> \
  env PYTHONPATH=<PP> ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <VENV>/bin/python \
  test/benchmark/bench_gp.py <ARGS>
```

with `ROOT`/`PP` = the HEAD worktree / `src` (NEW) and `/tmp/base_850db27` / `/tmp/base_850db27/src` (OLD), and `ARGS` per config (all add `--json-out <file>.json --tag <TAG>`):

| config | args |
|---|---|
| A1 numpy | `--task eval64 --eval-reps 25 --stage-probe --fixture test/goldens/forest_random.npz` |
| A1 iree | same + `--backend iree --iree-device 0 --iree-opt-level O1` |
| A2 numpy | `--task eval64 --eval-reps 25 --stage-probe --fixture test/benchmark/fixtures/forest200_tree64_random.npz` |
| A2 iree | same + `--backend iree --iree-device 0 --iree-opt-level O1 --max-cold-builds 400` |
| A3 {numpy,iree} | `--task sr --pop 200 --gens 10 --tree-len 64 --reps 2 --stage-probe` (+ `--backend iree --iree-device 0 --iree-opt-level O1 --max-cold-builds 1500`; iree arms also `EVOGP_ETL_CACHE_SIZE=8192` on both sides, as R2's `iree_sr_a` did) |
| A4 numpy | `--task eval64 --eval-reps 11 --data-rows 10000 --stage-probe --fixture test/goldens/forest_random.npz` |
| B5 / B6a / B6b | the A1-iree / A2-iree / A1-numpy args, all **without** `EVOGP_EVAL_MODE` (default `auto`); B6b adds `--backend numpy` |
| C7 / C8 | `--task sr --pop 1000 --gens 20 --tree-len 128 --reps 1 --stage-probe` with default guards, iree (`C7`) and numpy (`C8`) |

## 2. Method and its limits

OLD and NEW were separate checkouts (worktrees) run interleaved — 3–4 rounds per numpy config, the iree arms 1–2 rounds — and the reported value is the **median of the round medians**.
One contention window is on record: the box sat at load ~9–13 until ~20:00 and at load ~73–110 from ~20:05 to the end (a foreign 22 GB tenant on GPU 6), so every cell measured in that window was re-run, and A1-numpy was re-run under load as a control (OLD 21.627 / NEW 21.516 ms, i.e. unchanged).
`bench_gp.py`'s `detect_side()` (`test/benchmark/bench_gp.py:373`) labels a run `"new"` whenever `evogp.tree.etl_engine` is importable — true for **both** checkouts — so the `side` field is useless here; every number below is taken from the explicit `--tag` labels.
The `--stage-probe` `loss_ms` column is not comparable across sides: NEW's `sr_fitness` calls the un-wrapped `_evaluate_rows`, so `evaluate_ms` stays 0 and the whole fitness wall time lands in `loss_ms`; the stage numbers in §4 are therefore read per stage (decode / build-hit / build-miss / run / readback / upload), not from `loss_ms`.

## 3. A/B (warm medians, ms; specialized mode both sides)

| config | OLD `850db27` | NEW `f227e4c` | ratio NEW/OLD | R2 committed (base-era code) |
|---|---|---|---|---|
| eval64 fixture64 n64 numpy | 21.566 | 21.154 | 0.981 | 21.056 |
| eval64 fixture64 n64 iree specialized | 71.961 | 70.485 | 0.979 | 69.61 |
| eval64 fixture200 n64 numpy | 94.446 | 91.542 | 0.969 | 93.674 |
| eval64 fixture200 n64 iree specialized | 235.214 | 234.221 | 0.996 | 229.49 |
| sr pop200 tree64 gens10 numpy (per-gen median, rep1) | 24.008 | 23.460 | 0.977 | 22.07 |
| sr pop200 tree64 gens10 iree specialized (per-gen median, rep1) | 79.657 | 64.633 | 0.811 | 65.31 (noisy: OLD r1 is a self-outlier) |
| eval64 fixture64 n10000 numpy | 24.081 | 24.051 | 0.999 | 23.120 |

Read it as: **neutral-to-slightly-better on the fixed-forest warm path** (−1 to −3%, inside the ±10% noise band), with the two real wins elsewhere — the stage split of §4 (decode and the removed prediction upload) and the iree `sr` arm, which is fast on the NEW side in both rounds (63.2 / 66.0 ms) while Old r1 (98.1 ms) is ~50% above R2's committed 65.31 ms under identical code.
OLD r2 (61.25 ms) and both NEW rounds agree with R2 within 6%, so the 0.811 ratio is the OLD outlier, not a NEW gain; the honest claim for the `sr` iree arm is "no regression, one OLD outlier".

## 4. Stage deltas (stage probe totals, warm, over the rep: 25 calls; 11 for the n10000 arm)

| config | stage | OLD | NEW |
|---|---|---|---|
| A1 numpy fixture64 n64 | decode | 42.3 / 42.8 / 45.1 | 27.6 / 27.7 / 29.0 |
| | upload (prediction round-trip) | 8.2 / 9.8 / 8.6 | **0.0** |
| | build-hit / run | 234–247 / 225–237 | 235–246 / 226–236 |
| A2 numpy fixture200 n64 | decode | 187.8 / 188.2 / 191.5 | 112.4 / 113.8 / 114.5 |
| | upload | 21.3 / 22.9 / 21.3 | **0.0** |
| | build-hit / run | 988–1020 / 1102–1156 | 983–993 / 1095–1103 |
| A4 numpy fixture64 n10000 | decode | 18.7–21.6 | 12.5–13.1 |
| | upload | 9.8–16.3 | **0.0** |
| | run | 121.2–144.2 | 119.7–123.1 (r1 load-contaminated) |
| A1 iree fixture64 n64 | decode | 46.9 | 30.5 |
| | upload | 18.3 | **0.0** |
| | build-miss (48 compiles) | 40018.5 | 36903.6 |
| A2 iree fixture200 n64 | decode | 190.1 | 120.3 |
| | upload | 47.4 | **0.0** |
| | build-miss (155 compiles) | 101480.7 | 70868.4 |

So: decode ≈ **43 → 28 ms** (fixture64) and ≈ **188 → 112 ms** (fixture200) on numpy, ≈ **19 → 12 ms** at N=10000, ≈ **47 → 30 ms** and ≈ **190 → 120 ms** on iree; the prediction upload is **0.0 on both backends** (the prediction never leaves the host); `build-hit` and `run` are unchanged (the ETL call path is untouched).
Cold/first-call numpy medians: **233.2 → 190.3 ms** (fixture64, −18%) and **1029.8 → 983.1 ms** (fixture200, −4.5%) — the decode cut also lands in the cold phase, the remaining cold cost is per-structure compilation.

## 5. Routing sanity at this HEAD (default `auto`, NEW only)

| config (default `auto`) | route | warm ms | cold | verdict |
|---|---|---|---|---|
| iree fixture64 n64 | dense | 12.083 | 13.65 s, 1 cache miss | 5.8× the specialized 70.485 |
| iree fixture200 n64 | dense | 12.529 | 15.39 s, 1 cache miss | 18.7× the specialized 234.221 |
| numpy fixture64 n64 | specialized | 22.503 | 187.9 ms, 48 misses | as designed |

No `EVOGP_EVAL_MODE` / `EVOGP_DENSE_*` semantics were changed by this work: the dense policy was re-validated by measurement only, and the numpy default route stays specialized (22.503 ms = the A/B arm within noise).
The dense arms' `fds 347 → 6491` growth is the known iree-adapter fd leak (§8), not a new one.

## 6. Pop-1000 robustness at this HEAD

* iree, `sr --pop 1000 --tree-len 128 --gens 20`, **default guards**: exit 3, `complete=True partial=True skipped=True skip_reason=cold-builds-exceeded:431>300`, 2 of 20 generations, wall 467.5 s, fds 347 → 14139 — a **clean guard stop**, not the R2-grade SIGSEGV; gen 0 ran densely (0 `build_cached` calls, 13.06 s incl. the compile) and gen 1 fell back to the specialized path (430 misses + 255 hits) because mutated trees exceed the 32-node dense trace cap, which tripped the guard.
* numpy mirror (same args, default guards): completes — 20 gens in 24.7 s, warm per-gen median 759.1 ms (R2 `np_sr_b` rep1 median 670.1 ms), cold gen 0 5832.1 ms, 4760 cumulative misses, fds flat 59 → 59.

## 7. Numerical equivalence of the levers

* **decode** (`e2899c0`): output bit-identical vs the base decoder — both fixtures × `output_len` 1/2/3, a 200-tree random population, and all 22 malformed-tree cases (exception type **and** message). The refactor was only certified after its own regressions were found and fixed by a **1143-case differential sweep** against `850db27` comparing every `DecodedTree` field bit-for-bit: 775 cases differed (NameError-only) → 86 (52 silently wrong payloads) → **0/1143 with the fixes in place** (`5c96d1f`).
* **`evaluate`** (`f1bdec5`): values bit-identical old-vs-new in all four shapes (e.g. pop200 × N=10000: 1,941,250/1,941,250 float32 values), NaN/inf placement identical, numpy and iree.
* **`sr_fitness`**: `max_rel ≤ 2.8e-7` on every finite entry (same documented float32 reduction-order class as the old device reduction), identical NaN/±inf placement, 35–70% of finite entries bit-identical; identities preserved (outputs summed, never divided by `output_len`).
* Suite gates at the measurement HEAD: `pytest src/evogp/tests -q` → 307 passed / 4 skipped, `-k golden` 6/6.

## 8. Residual gap attribution and upstream requests

* What is left on the specialized path at this HEAD is **ETL-side per-tree overhead**: `build_cached`'s hit path (program re-encode + sha256 key derivation) ≈ **85 µs/tree**, `etl.run`'s statics re-validation in O(#tokens), and the per-tree iree readback; the upstream ETL probe round measured a per-invoke floor of **0.131–0.162 ms** (numpy interpreter call path; reported, not re-derived here).
* evogp-side addressable ceiling at this HEAD is therefore ≈ **10%** of the warm specialized cost — which is why the decode/upload wins are the ones that landed rather than micro-tuning the call path (see §9).
* Legacy CUDA reference (R2 `leg_c5r2.json`): **0.633 ms** per fixture64/n64 call → the remaining warm gap is ≈ **33×** numpy specialized / ≈ **111×** iree specialized / ≈ **19×** iree auto-dense.
* Upstream `etl` requests, in priority order: (1) **batched readback / multi-program execution** — impossible today (the runtime `etl.Tensor` has no `concat`/`stack`/`__getitem__`, and `__dlpack__` raises for cuda payloads); (2) `build_cached`'s hit-path re-encode+sha256 and `etl.run`'s statics re-validation (being fixed upstream in parallel); (3) `Executable.close()/release()` plus a `CompileCache` eviction hook (eviction is a bare `popitem`; measured to free 0 native resources); (4) the iree adapter creating a fresh HAL device per load/upload (fd leak, ~97% eventfd); (5) xla `XlaExecutable.close()` unreachable; (6) tvm missing `os.close`.

## 9. Dead levers (and the measurements that killed them)

* `_quiet()` hoist out of the hot loop: **1.19 µs/tree** — below noise.
* `TensorSpec` hoist out of `_executable`: **1.35 µs/tree** — below noise.
* Decode memoization keyed on tensor identity: rejected — a silent stale-decode hazard (the population tensors are views/copies under `Forest.__getitem__`, so identity is not a correctness-safe key).
* `_resolve_backend` caching: rejected — it would change the documented per-call env-read semantics of `EVOGP_ETL_BACKEND` / `EVOGP_ETL_IREE_DEVICE` / `EVOGP_ETL_IREE_OPT_LEVEL`.
* torch host reduction for `sr_fitness`: **95.3 ms vs 20.5 ms** (numpy) at pop64/N=10000 — torch's CPU kernel opens an intra-op thread pool whose spin-waiting starves the single-threaded ETL/numpy interpreter of the *next* call (the stalls land in unrelated stages such as `build_cached` hits), which is why the host reduction is numpy.

## 10. For the post-merge R2 round

Quote the NEW column of §3 as the new baseline, then re-run the R2 protocol (`bash test/benchmark/r2/run_all.sh`) on the merged result — R2's committed numbers were measured on the pre-change base under its own conditions, so they are the "before" column and must not be compared to a merged run as if the method were identical.

## 11. Executable memo round (T1-A84) — the specialized path's per-tree lookup

Scope: the **cost of one `_executable` lookup per tree** on the specialized route (everything else in §1–10 is untouched input for this round; the dense sections were not re-measured).
Change under test: `etl_engine._executable` now consults a bounded value-keyed LRU memo (`_ExecutableMemo`, `etl_engine.py`, key = `(id(_eval_step), program, n_rows, input_len, backend_name, device, _freeze_options(options))`) before `etl.build_cached`; `clear_cache()` resets it and `cache_stats()` reports it (`memo_hits`/`memo_misses`/`memo_entries`/`memo_maxsize`).
Before = `096b2a1` (pre-memo engine + the new battery), after = `2fa8d0a`; both sides are the *same* box, `CUDA_VISIBLE_DEVICES=6`, numpy backend, `ETL_DISABLE_LOCATIONS=1`, warm medians, and the **same** probe code (the before side ran in a scratch worktree with the extended `warm_path_probe.py` copied in).

Exact commands (from the repository root; the before side from the scratch `096b2a1` worktree):

```
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <venv>/bin/python \
  src/evogp/tree/warm_path_probe.py --fixture test/goldens/forest_random.npz --reps 25
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <venv>/bin/python \
  src/evogp/tree/warm_path_probe.py --fixture test/benchmark/fixtures/forest200_tree64_random.npz --reps 25
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <venv>/bin/python \
  src/evogp/tree/warm_path_probe.py --fixture test/goldens/forest_random.npz --rows 10000 --reps 11
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <venv>/bin/python \
  src/evogp/tree/executable_memo_check.py --reps 200
```

`warm_path_probe.py` gained two stages for this round: `lookup cycle` (`_executable` over one full pass of the population's distinct non-constant programs) and `raw build_cached cycle` (the same pass straight through `etl.build_cached`) — the per-tree loop never repeats a program, so the old `executable_ms` (one tree, called 20+ times in a row) measures only the cheap end of `build_cached`'s hit.

| fixture (pop / tree / rows / invoked) | metric | before `096b2a1` | after `2fa8d0a` |
|---|---|---|---|
| fixture64 (64 / 63 / 64 / 53) | `_executable`, same program twice in a row (`executable_ms`) | 0.017 ms | **0.002 ms** |
| | lookup cycle, per tree | 175.22 us | **2.43 us** |
| | raw `build_cached` cycle, per tree (control) | 167.87 us | 160.11 us |
| | `evaluate` (`per_row=False`) | 19.561 ms | **9.313 ms** |
| | warm `Forest.SR_fitness` | 21.082 ms | **10.305 ms** |
| | one `_run_tree` | 73.39 us | 46.85 us |
| fixture200 (200 / 64 / 64 / 171) | lookup cycle, per tree | 206.04 us | **2.80 us** |
| | raw `build_cached` cycle, per tree (control) | 204.50 us | 206.95 us |
| | `evaluate` (`per_row=False`) | 86.726 ms | **55.004 ms** |
| | warm `Forest.SR_fitness` | 92.762 ms | **57.381 ms** |
| | one `_run_tree` | 227.36 us | 203.35 us |
| fixture64 @ N=10000 (64 / 63 / 10000 / 53) | lookup cycle, per tree | 160.60 us | **2.38 us** |
| | `evaluate` (`per_row=False`) | 21.245 ms | **11.231 ms** |
| | warm `Forest.SR_fitness` | 23.184 ms | **15.650 ms** |
| | one `_run_tree` | 128.85 us | 48.40 us |

Read it as: the per-tree lookup falls from **160–207 us to 2.4–2.8 us** (−98.6 to −98.9%), i.e. essentially every lookup now costs a dict hash of the program instead of `build_cached`'s per-call re-encode; the whole-population warm cost falls by **31–51%** on these fixtures (10.8 ms of 21.1 ms at pop 64, 35.4 ms of 92.8 ms at pop 200, 7.5 ms of 23.2 ms at pop 64 / N=10000 — the memo saving is per-tree and rows-flat, so its *share* shrinks as the per-tree `etl.run` grows with N).
The `raw build_cached cycle` control column is unchanged by the round (167.9→160.1 / 204.5→207.0 us), i.e. the win is entirely the memo, not an ETL or load difference; the `one _run_tree` column moves less than the lookup because that probe calls `_run_tree` on a single tree in a row (where `build_cached`'s one-slot pickle fingerprint still helps a little).

Why the earlier estimate was low: `executable_ms`/the §8 quote (~16 us, ~85 us/tree) measured a *repeated* key (one small program ⇒ cheap pickle fingerprint, and its dedup is what `CACHE`'s one-slot fast path is for). The real loop walks 53–171 DIFFERENT programs per call, which forces the full key derivation on every tree — that is the 160–207 us/tree. The memo's key is value-typed, so it helps in both patterns (and the elite/duplicate case is a hit by construction).

Battery: `src/evogp/tree/executable_memo_check.py` — **6/14 checks, exit 1** on `096b2a1` (as recorded at that commit: `m1` 26 identical calls → 25 `build_cached` calls — the battery there reset the counter after the first call, so the follow-up fix that counts all 26 reports the same pre-memo failure as 26; `m4` repeated lookups served by `CACHE`, hits delta 25; `m5` value-equal program re-enters `build_cached`; `m6` 3 programs × 2 shapes × 3 rounds → 18 calls; `k0` no `_executable_key`; `c1` memo 23.5 us vs `build_cached` 19.3 us, ratio 1.22; `c2` 23.5 us ≥ 5 us; `s1` no memo counters) → **32/32, exit 0** after, with `c1` memo 1.68 us vs `build_cached` 13.54 us (ratio 0.124; the check looks the *same* program up repeatedly on the fixture's first — small — tree, hence the smaller absolute numbers) and the differential checks (`d1`–`d3`) bit-identical before and after. Full suite at `2fa8d0a`: `pytest src/evogp/tests -q` → **308 passed / 4 skipped** (~133 s).

Consequences to keep in mind (no source outside this node was changed):
* `test/benchmark/stage_probe.py` wraps `etl_engine.build_cached`, so the `build-hit` column will now collapse toward zero (the memo absorbs those calls) while cold `build-miss` counts and the `run` column are unchanged; the probe's own self-test is unaffected (it installs a *fake* engine module with its own `build_cached`). Flagged for the `test/` node, not edited here.
* `cache_stats()["hits"]` no longer grows with repeated `_executable` lookups (that is the memo, documented in the module docstring and in `cache_stats`'s docstring); the dense batteries' `misses`-based deltas and `dense_integration_check`'s "auto+numpy stays on the per-tree path" (`misses > 1`) / "forced dense = ONE build" assertions still hold (they `clear_cache()` first).
* Conversion trims were checked and skipped: `_input_to_numpy` / `_as_numpy` are called once per public call (`evaluate` / `sr_fitness`), never per tree, so there is nothing repeated to trim; the only free trim — moving `TensorSpec` construction to the memo-miss path — is part of the change. `_quiet()` per tree and `etl.run`'s statics re-validation were left alone (previously measured below noise / an engine restructure, out of scope).
* ADOPTION POINT: the memo (and the `memo_*` counters) is a deliberate stopgap for the ETL-side `etl.jit` / pre-bound-runner work, which should absorb it — a pre-bound runner skips `build_cached`'s key derivation AND `etl.run`'s statics re-validation. The marker is in the module docstring ("Per-tree executable memo" → ADOPTION POINT) and on `_EXECUTABLE_MEMO`.
