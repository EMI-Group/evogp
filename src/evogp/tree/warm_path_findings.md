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
