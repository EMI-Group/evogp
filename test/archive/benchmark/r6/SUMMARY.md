# R6 — Triton fused-kernel prototype: feasibility + measurement round (analysis of record)

R6 tests the r5 RCA's own conclusion. `../r5/RCA.md` §8/§9 measured the residual old-vs-new gap as two
additive blocks — a **dispatch-bound dense graph** (310 device ops = 249 dispatches + 61 DtoD copies on a
125-deep chain, ~1.4 µs kernels, *pure forward alone* 1.2142 / 1.4612 ms) plus **per-call host glue**
(uploads 3.51 / 1.53 ms, decode 0.70 / 1.38, loss assembly 0.51, readback 0.12) — and stated that *"legacy
parity (0.646 / 0.637 ms) requires collapsing the GPU side: 3 device ops (1 memset + 1 fused tree-walk
kernel + 1 averaging kernel) vs the dense route's 310 (≈103× more)"*. R6 asks the obvious follow-up: if a
custom kernel replicating the legacy fused design is written for today's stack (Triton, no compiled
extension), does legacy-class performance come back for the single-output SR case?

**Round status: complete.** Prototype + numeric validation + one same-window 10-cell campaign + P1 extras +
the stdlib gate `check_r6.py` (**14 records / 1440 checks / 0 failures**, plus 10 anomaly cells, 5
kernel-host fact sheets, 2 window files and 30 cell logs) are all committed. Everything below is recomputed
from the committed records, not from prose: `results/w1d_20260914/*.json` (primary window),
`results/w1d_20260914/extra/**` (extras), `results/w1c_20260914/*.json` (anomaly window),
`results/validation/validation.json` (run 1 = record of record) and `results/validation/run2/validation.json`
(same-headline replicate). `RUNLOG.md` is the operator log with the same numbers plus the per-cell narrative.
Records are **immutable evidence**; a re-run uses a new window label.

---

## Verdict (headlines, all same-window, all gated)

| # | finding | numbers (fixture64 / fixture200) |
|---|---|---|
| 1 | **The Triton fused evaluator is semantically correct on the fixtures** | MSE vs the plain-torch oracle `allclose` (rtol=atol=1e-4, equal_nan) with max\|Δ\| 7.63e-06 / 0.0625; legacy goldens 12/12 `allclose` (max\|Δ\| 1.53e-05); values mode bit-identical-or-both-NaN 4096/4096 / 12800/12800; op coverage 31/31 exercised, `uncovered_ops=[]` |
| 2 | **It beats the legacy CUDA engine in the same window** | e2e **0.1314 / 0.1379 ms** vs legacy **0.6364 / 0.6269 ms** ⇒ triton is **4.8× / 4.6× faster** than legacy (triton/legacy = 0.207× / 0.220×) |
| 3 | **It collapses the route it replaces** | vs dense-iree e2e 4.7860 / 6.3179 ms ⇒ **36.4× / 45.8×** faster; pure route 0.05470 / 0.05472 ms vs dense pure 1.3339 / 1.4796 ms ⇒ **24.4× / 27.0×** |
| 4 | **The win is the op-count collapse, not micro-optimization** | 1 kernel launch per invocation (device 5.8 / 9.1 µs) vs the legacy engine's 3 device ops of substance (6.4 / 11.7 µs) vs the dense graph's 310 device ops |
| 5 | **What remains is host, not device** | ~50 µs/call host residual around the triton kernel (launch-bound); legacy carries the same class of cost (**617.2 / 576.5 µs** of its 631.9 / 601.8 µs call = 97.7 % / 95.8 %) |
| 6 | **One-time cost per fresh process** | first call 755.6 / 743.2 ms (compile 729.9 / 716.0), ≈0.73 s with a warm disk cache, **1 785.4 ms** with an empty cache; warm medians unaffected |
| 7 | **Verdict** | **Q5 = YES** — "same implementation ⇒ legacy-class performance" is supported, and *exceeded* (triton is faster than legacy in the same window), with the two caveats in §10/Q5: legacy's 0.63 ms is itself ≈97 % host, and the mechanism is the 3-op-vs-310-op collapse |

---

## 1. Hypothesis, prototype and acceptance criteria

| item | value |
|---|---|
| hypothesis under test | r5 `../r5/RCA.md` §9: legacy parity needs the GPU side collapsed from ~310 device ops to ~1 fused op **and** the per-call host glue removed; a custom kernel that replicates the legacy fused design (one launch, tree-walk interpreter over staged columns, no per-call uploads/readback) restores legacy-class performance for single-output SR |
| prototype | `triton_eval.py` (**877 lines**) — Triton fused evaluator, `_sr_eval_kernel` (`block_n=64`, `num_warps=2`), `launches_per_call=1`, staging **`dense_engine.prepare_dense`** (byte-identical columns to the dense route), build-once / run-many, `TritonUnsupported` (a distinct type, so `Forest.SR_fitness`'s `DenseUnsupported` fallback can never silently take over) |
| drivers | `run_triton_eval64.py` (584 lines) — type-1 cell through the **unmodified public API** with `evogp.tree.dense_engine.sr_fitness_dense` rebound at the module attribute (`binding_style` recorded; call site `src/evogp/tree/forest.py:338-351`), plus `--mode pure` (type 2) and `--e2e-companion`; `run_cells.sh` (963 lines) — the same-window campaign runner; `run_under_lock.py` (49 lines) |
| validator | `validate_triton.py` (1 887 lines) — oracle/goldens/per-op/edge-case validation; `extra_trace.py` (389 lines) — the kineto kernel-vs-host probe |
| acceptance criteria | (1) numeric correctness vs `src/evogp/tests/reference_eval.py` and `test/goldens/`; (2) same-window comparison vs legacy + dense-iree on both measurement types; (3) triton compile economics; (4) an explicit op-coverage + deviation declaration; (5) a stdlib, GPU-free gate (`check_r6.py`, 1 831 lines) recomputing every headline |
| declared op coverage | 31 declared dispatch entries (VAR, CONST, 13 BFUNC, 15 UFUNC, TFUNC:if) — **all 31 exercised, `uncovered_ops=[]`, `fallthrough_nodes=0`** over 36 built evaluators; `exact_ops = ['VAR','CONST','+','-','*','/']` |
| deviations | 4 measured, declared classes (dev-ftz-denormals, dev-fast-math-ulp, dev-zero-sign, dev-const-neglzero), each `affects_fixtures: false` (§6.4) |

---

## 2. Window, provenance and method

| item | value |
|---|---|
| primary window | **`w1d_20260914`** — 2026-09-14T00:53:54+08:00 → 00:55:53+08:00 (10 cells, 1 m 59 s wall), GPU 6 |
| superseded window | `w1c_20260914` — 00:49:19 → 00:52:33 (whole-window inflation; §9) |
| pre-flight | `check_deps.txt` → `READY (20 ok / 0 warn / 0 missing)`; `--list` / `--dry-run --group all` matched the real CLIs ⇒ `run_cells.sh` was **not** modified |
| cells | 10/10 exit 0, `check_exit=0 all_exit=0` (the runner's aggregation), and the window's own line reads `ok=10 skipped=0 failed=0`; no timeouts: `leg_{f64,f200}`, `dense_{f64,f200}`, `triton_{f64,f200}`, `densepure_{f64,f200}`, `tritonpure_{f64,f200}` |
| type 1 (e2e) protocol | 25 sync-bracketed `problem.evaluate(forest)` calls, **call 0 cold**, `median_warm_ms = median(calls 1..24)`, one fresh process per cell |
| type 2 (pure) protocol | 5 timed blocks × 100 resident invocations after a 10-invocation warmup; `median_per_invocation_ms`; same process runs an 11-call `--e2e-companion` block through the public API |
| type-1 triton envelope | `seam_patched=true`, `patch_target=evogp.tree.dense_engine.sr_fitness_dense`, `EVOGP_EVAL_MODE=dense`, `route="triton"`, `kernel_name="sr_eval_kernel"`, `launches_per_call=1`, `staging="dense_engine.prepare_dense"`, `data_uploads=1` (build-time only) |
| engine identity (new) | `evogp_commit=4c33ec86d509f657ed97855c7134ba78c8b5b415` in all 10 cells; ETL checkout `/mnt/local-ssd/bchuang/etl` @ `96df7da8f2ba39f52cf2f2b61888e67f70471bc9` (the revision the r5 window used), `etl_version=0.1.0`, `EVOGP_ETL_BACKEND=iree`, `EVOGP_EVAL_MODE=auto`, `EVOGP_ETL_IREE_DEVICE=0`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`, `ETL_DISABLE_LOCATIONS=1` |
| engine identity (legacy) | `/tmp/evogp_legacy_bench` (`b613503` build, `evogp_cuda…so` present) + `/tmp/evogp_t1a263/siteutils`; no `EVOGP_*`/`ETL_*` vars (inert) |
| GPU | `CUDA_VISIBLE_DEVICES=6` — RTX A6000, cc 8.6, 17.9 GiB free at window start; foreign memory-resident tenant PID 2903622 (30 486 MiB, 0 % util) measured, never killed; **GPU 7 never touched** (hardware ECC fault) |
| Triton runtime | 3.2.0 via the project venv (py3.11.2, torch 2.6.0+cu124); `Python.h` supplied by the session-scoped `CPATH` header tree; `TRITON_CACHE_DIR=/tmp/evogp_r6_triton_cache` |
| serialization | exactly one `fcntl.flock` per cell on `/tmp/gpbench_critical.lock` via `run_under_lock.py`, never nested, `timeout -s INT -k 30 <cap>` **inside** the lock; caps new/triton = 900 s, legacy = 600 s, check = 180 s; `ulimit -n 65536` for the iree/triton cells |
| tolerance / band | warm medians ±10 % on this shared box (a contended window reads 2–3× inflated — see §9); `results/**` is immutable evidence, re-runs go to a fresh window label |
| fixtures | fixture64 = `test/goldens/forest_random.npz` (64 trees, row_len 63, x (64,2), 64 rows); fixture200 = `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, row_len 64, same 64 rows bytewise); loader `test/benchmark/bench_gp.py::build_eval_case` |

---

## 3. Cross-window sanity — the window reproduces the r5 anchors

The r5 same-window anchors (`../r5/results/q4_window/`) were measured in one 2026-09-13 window. All six of
the w1d medians that have an anchor land inside **−11.1 % … +9.9 %**:

| cell | w1d median (ms) | r5 anchor (ms) | Δ | anchor record |
|---|---|---|---|---|
| `leg_f64` | 0.6364 | 0.6461 | −1.5 % | `q4_leg64.json` |
| `leg_f200` | 0.6269 | 0.6372 | −1.6 % | `q4_leg200.json` |
| `dense_f64` | 4.7860 | 5.3823 | −11.1 % | `q4_eval64_iree.json` |
| `dense_f200` | 6.3179 | 5.7912 | +9.1 % | `q4_eval200_iree.json` |
| `densepure_f64` | 1.3339 | 1.2142 | +9.9 % | `q4_pure64_iree.json` |
| `densepure_f200` | 1.4796 | 1.4612 | +1.3 % | `q4_pure200_iree.json` |

Consequences used throughout this file: (a) the r6 triton numbers are quotable against the r5 legacy/dense
anchors as **same-window** (7.52× / 10.08× is the dense-vs-legacy gap measured *here*, r5's was 8.33× / 9.09×
— the same ≈8–11× band); (b) the cross-round r3/r4 legacy anchor (0.6323 / 0.6357 ms) is **not** needed for any
r6 verdict.

---

## 4. Type 1 — end-to-end `problem.evaluate(forest)`, 25 warm calls

| cell | median warm (ms) | min (ms) | max (ms) | call 0 = first call (ms) | notes |
|---|---|---|---|---|---|
| `leg_f64` | **0.636393** | 0.62483 | 0.89574 | 23.408 | legacy CUDA engine, 3 device ops |
| `leg_f200` | **0.626894** | 0.62295 | 1.00030 | 22.415 | flat vs f64 (fixed host component) |
| `dense_f64` | **4.785977** | 4.58076 | 6.53011 | 11 775.7 | dense-iree `auto` (the route triton replaces) |
| `dense_f200` | **6.317906** | 5.90308 | 10.36816 | 12 317.0 | same, 200 trees |
| `triton_f64` | **0.131419** | 0.12579 | 0.56570 | 755.6 | 1st call = compile; call 1 = 0.5657, then ≈0.13 |
| `triton_f200` | **0.137907** | 0.12684 | 0.64539 | 743.2 | call 1 = 0.6454, then ≈0.13–0.14 |

Derived ratios (recomputed from the JSON medians):

| ratio | fixture64 | fixture200 |
|---|---|---|
| dense-iree e2e / legacy e2e | **7.52×** | **10.08×** |
| triton e2e / legacy e2e | **0.207×** (triton 4.8× faster) | **0.220×** (triton 4.6× faster) |
| dense e2e / triton e2e | **36.4×** | **45.8×** |
| triton e2e / triton pure | 2.40× | 2.52× |

Counter / coverage gate on every triton type-1 cell (what makes the number trustworthy —
`seam_patched=true` plus an exception type that cannot silently fall back to dense/torch):

| field | `triton_f64` | `triton_f200` |
|---|---|---|
| `triton_call_count` / `expected` | 25 / 25 | 25 / 25 |
| `triton_runs` / `expected` | 25 / 25 | 25 / 25 |
| `builds` / `cache_hits` / `fallback_count` | 1 / 24 / 0 | 1 / 24 / 0 |
| `counter_gate` | `{"passed": true, "problems": []}` | `{"passed": true, "problems": []}` |
| `op_coverage` / `used_ops` / `uncovered_ops` | 31 entries / `['*','+','-','/','CONST','VAR']` / `[]` | same / same / `[]` |
| kernel shape | pop 64, trace_len 63, rows 64, `block_n=64`, `num_warps=2`, 1 launch/call | pop 200, trace_len 64, rows 64, same |

The driver exits non-zero unless `triton_call_count`/`triton_runs` equal the expected counts and
`fallback_count == 0` (the round's **silent-fallback tripwire**); both windows' triton cells exited 0.

---

## 5. Type 2 — pure internal graph forward (no readback / host assembly / key derivation)

| cell | median per invocation (ms) | min (ms) | 5 block totals (ms) | notes |
|---|---|---|---|---|
| `densepure_f64` | **1.333942** | 1.29033 | 172.8, 155.3, 133.4, 129.0, 132.0 | frozen dense runner, 1 graph call/invocation |
| `densepure_f200` | **1.479640** | 1.40048 | 187.9, 162.1, 144.6, 148.0, 140.0 | same, 200 trees |
| `tritonpure_f64` | **0.054701** | 0.054454 | 5.470, 5.469, 5.511, 5.593, 5.445 | 1 launch/invocation; block spread < 3 % |
| `tritonpure_f200` | **0.054717** | 0.054634 | 5.463, 5.475, 5.475, 5.472, 5.471 | flat 64→200 trees (stdev 4.7e-05 ms) |

| ratio | fixture64 | fixture200 |
|---|---|---|
| dense pure / triton pure | **24.4×** | **27.0×** |
| triton pure / legacy e2e | 0.086× (legacy ≈11.6× a single pure triton call) | 0.087× (≈11.5×) |

Triton type-2 counters: `triton_call_count=11` (the companion block), `triton_runs=521` = 10 warmup + 500
timed + 11 companion, `builds=1`, `cache_hits=11`, `fallback_count=0`, `counter_gate.passed=true` — every
timed invocation went through Triton.

Provenance note (read upward in `RUNLOG.md` §6.6): the `densepure_*` records carry
`pure.runner_source="frozen-api:build_dense_runner"` and `compile.path="frozen"` (the frozen build-once /
run-many API), **not** the `local-fallback:*` label the r4 text attaches to dense pure cells, and
`compile.build_ms`/`decode_plan_ms` are `null` there (only `construct_total_ms` 11 884.6 / 12 550.2 ms).
The records are what the installed harness emits; no harness or `src/` change was made.

---

## 6. Numeric validation (run 1 = record of record, 208 checks, 0 gate failures)

Command shape: `validate_triton.py` on fixture64 + fixture200, `output='loss'` (MSE + MAE) and
`output='values'`, against `src/evogp/tests/reference_eval.py` (plain-torch oracle) and the frozen
`test/goldens/` captures; tolerances rtol = atol = 1e-4, `equal_nan=True`; bit identity =
"float32 payload bits equal **or** both NaN" (NaN payload bits / zero sign may differ across engines).
`passed=True`, `passed_strict=False` — the 10 non-strict checks are the declared deviations of §6.4.

### 6.1 Fixture losses and values vs the oracle

| fixture | metric | n | allclose | bit-identical (bit\|nan) | max \|Δ\| | max rel | max ULP | special mismatch |
|---|---|---|---|---|---|---|---|---|
| fixture64 | MSE | 64 | yes (64/64) | 50/64 (53/64) | **7.6294e-06** | 1.34e-07 | 2 | 0 |
| fixture64 | MAE | 64 | yes (64/64) | 53/64 (56/64) | **4.7684e-07** | 1.70e-07 | 2 | 0 |
| fixture200 | MSE | 200 | yes (200/200) | 120/200 (132/200) | **0.0625** | 1.81e-07 | 2 | 0 |
| fixture200 | MAE | 200 | yes (200/200) | 108/200 (120/200) | **3.0518e-05** | 1.75e-07 | 2 | 0 |
| fixture64 | values (64,64) | 4096 | yes (4096/4096) | 4088/4096 (**4096/4096**) | 0.0 | 0.0 | 0 | 0 |
| fixture200 | values (200,64) | 12800 | yes (12800/12800) | 12488/12800 (**12800/12800**) | 0.0 | 0.0 | 0 | 0 |

The loss contract is also checked structurally: positive `(pop,)` float32 on `cuda`, `negative_finite=0`,
`launches_per_call=1` (the caller negates, so this is the `(1,)`-positive-loss convention); NaN placement
matches the oracle exactly (3 / 12 NaN losses; 24 / 376 NaN predictions).

### 6.2 Exactness classes and op coverage

| op | declared bit-exact | full edge family (18 400 rows/op where applicable) | finite **core** grid (no denormal operands, no `-0.0`-literal trees) |
|---|---|---|---|
| `VAR`, `CONST` | yes | max ULP 0; CONST loses only the `-0.0` literal (§6.4) | max ULP 0 |
| `+` | yes | **max ULP 0, no mismatch class at all** | max ULP 0, bit\|nan 15 688/15 688 |
| `-` | yes | **max ULP 0** | max ULP 0 |
| `*` | yes | **max ULP 0** | max ULP 0 |
| `/` | yes | `allclose=False` on the full family (`dev-ftz-denormals`, 552 special mismatches on denormal operands/results) | `allclose=True`, bit\|nan 15 336/15 688, max\|Δ\| **5.877e-39** (one subnormal step, FTZ results) |
| `pow`, `loose_pow`, `sqrt` | no | red on both grids (`dev-fast-math-ulp` / `dev-ftz-denormals`) — not fixture ops | core `allclose=True`; `sqrt` keeps a 1-ULP class |

So: the four fixture ops are **`+ − *` bit-equal, `/` correctly rounded** (`_libdevice.div_rn`, the
ENGINE_SPEC `var2==0 ? NaN : var1/var2` select) **and bit-exact on the finite core grid modulo subnormal
results**; `VAR`/`CONST` reads are bit-exact except for the `-0.0` literal staging defect. The op table is
complete for the engine's **31 dispatch entries** (`uncovered_ops=[]`, `fallthrough_nodes=0`,
`exercised-but-undeclared=[]`, 36 evaluators built) — the stretch goal (full op table) is met, with the
transcendental last-ULP / FTZ caveats declared rather than hidden.

### 6.3 Legacy goldens (the deleted CUDA engine)

`test/goldens/sr_fitness.npz` (sha256 `250b79a3…`), provenance `b613503 -use_fast_math`:
**12/12 comparisons `allclose`**, worst max\|Δ\| **1.526e-05** on `problem_evaluate__tree_parallel__mse`
(the per-execute-mode families agree to **4.77e-07 – 1.53e-05** with **43–58 of 64** losses bit-identical; the
`tree_parallel` family is the loosest at 1.53e-05/1.43e-06 with 43–47/64 bit-identical, the
`hybrid/data/auto` MSE families sit at 7.63e-06 with 54–57/64 and their non-MSE twins at 4.77e-07 with 58/64).
A same-run cross-check also puts `engine_vs_oracle` (plain-torch ETL decode path) at
bit 55/64 max\|Δ\| 7.63e-06 (f64) and 139/200 max\|Δ\| 0.0625 (f200) — i.e. triton's error against the
*shipped* engine is the same order as the engine's error against the oracle.

### 6.4 Declared deviations (measured, reported, not fixed; all fixture-irrelevant)

| id | entries | class | what it is | fixture impact |
|---|---|---|---|---|
| `dev-ftz-denormals` | 9 552 | `denormal_operand`, `denormal_result` | subnormal operands behave as 0.0 and subnormal results come out (signed) 0.0; root cause = FTZ/JIT — **legacy-consistent** (the legacy kernel was built `-use_fast_math`) but **not oracle/ETL-consistent**; beyond-tol worst case `/` 552 entries, `sqrt` 440, `pow` 106 | none: the special-value scan of both fixtures finds **0** zeros/±0/denormals/inf/NaN in `x`, `labels` and `consts` |
| `dev-fast-math-ulp` | 10 866 | `finite_ulp` | last-ULP differences on `sin/cos/tan/cosh/exp/pow/loose_pow/sqrt/loose_sqrt` (libdevice not correctly rounded) | none (no fixture op among them) |
| `dev-zero-sign` | 440 | `zero_sign` | sign of zero differs on `neg` (both values 0.0, value-equal); run 1 only — see §11.1 | none |
| `dev-const-neglzero` | 4 000 | `const_neg_zero_literal` (diagnostic, not a gate) | a `-0.0` CONST literal reaches the op as `+0.0` because a CONST leaf is staged as `tl.zeros([BLOCK_N]) + cval`; sign-sensitive consumers (`/`, `loose_div`, `pow`, `loose_inv`) then differ — value-equal inside atol | none (no `-0.0` constant in either fixture) |

### 6.5 The silent-fallback tripwire

The one failure mode that would invalidate every timing is a timed call quietly falling through to the
dense/torch path. Three independent guards: (a) `TritonUnsupported` is deliberately **not**
`dense_engine.DenseUnsupported`, so `Forest.SR_fitness`'s `except DenseUnsupported` cannot swallow it
(`triton_eval.py` docstring); (b) the driver's counter gate compares `triton_call_count`/`triton_runs`
against the expected counts per mode and requires `fallback_count == 0`, exiting non-zero otherwise;
(c) all four triton cells record `counter_gate = {"passed": true, "problems": []}` and `fallbacks = 0`.
`check_r6.py` re-derives those identities from the records (and the type-1/type-2 call-count shapes:
25/25 with `builds=1, cache_hits=24`; 11 calls with `runs=521`).

---

## 7. Per-call decomposition (kineto, `extra/kernel_host/**/facts.json`)

Only **leaf** rows (self-time-dominated raw kernels) are read as device cost — kineto keeps both a C++ op
row (`aten::neg`, `evogp_cuda::tree_SR_fitness`) and the CUDA kernels it launched, so the aggregate
`kernel_time_total_us` double counts children (the probe emits a `double_counting_note`; a unit bug in the
probe's own residual formula was fixed on 2026-09-14, `RUNLOG.md` §6.4).

| target | clean, unprofiled (µs/inv) | leaf rows/inv | leaf device (µs/inv) | host residual (µs/inv) | read |
|---|---|---|---|---|---|
| `triton_pure_f64` | 56.204 | **1.00** | 5.825 | **50.380** | 1 launch; device = 10 % of the call |
| `triton_pure_f200` | 54.609 | **1.00** | 9.101 | **45.509** | device = 17 %; call stays flat ⇒ host-bound |
| `triton_e2e_f64` | 117.362 | 3.00 | 8.243 | 109.120 | `_sr_eval_kernel` 5.812 + the pipeline's `aten::neg` pair 1.215 ×2 (an op row + its kernel) |
| `legacy_e2e_f64` | 631.885 | 5.00 | 14.707 | **617.178** | `evogp_cuda::tree_SR_fitness` op row 6.387 + its children + the `neg` pair |
| `legacy_e2e_f200` | 601.778 | 5.00 | 25.304 | **576.474** | same shape, `tree_SR_fitness` 11.652 |

### 7.1 The legacy engine is itself ≈97 % host-side

Per invocation (f64 → f200), from the key-averages table: `treeGPRegressionFitnessKernel<false,true>`
**4.706 → 9.918 µs**, `averageFitnessValueKernel` **1.208 → 1.236**, `Memset (Device)` **0.473 → 0.498**
(= the 3 device ops of substance, 6.387 / 11.652 µs, i.e. exactly the `tree_SR_fitness` op row), plus the
torch `neg` **1.203 → 1.249**, for 5 dispatched kernels per call (`cudaLaunchKernel` ×3 + memset; the
`aten::neg` row is the C++ op). Against a 631.9 / 601.8 µs wall call that leaves **617.2 / 576.5 µs of host
residual** — and the legacy call is ≈flat from pop 64→1000 (0.604→0.603 ms in the round's feasibility
notes), i.e. do **not** read 0.63 ms as the legacy kernel's cost. This is the r5 "fixed host component"
caveat, now measured in the r6 window.

### 7.2 Triton: the kernel is right, the launch/host side is the floor

Triton's evaluator is **1 kernel launch per invocation** (`leaf_kernels_per_invocation = 1.00`, 100/100 in
both pure cells) costing 5.825 / 9.101 µs; the call costs 56.2 / 54.6 µs, so a **≈45–50 µs host residual**
dominates. Scaling the population 64→200 (3.125× trees) costs the kernel +3.28 µs while the *call* stays
inside noise — the prototype's floor is host-side, exactly as predicted before the campaign.

### 7.3 The dense route, for contrast (r5, same-window anchors)

The route triton replaces is a **310-device-op serial graph** (249 dispatches + 61 DtoD copies, longest path
125, ~1.4 µs kernels) whose pure forward alone is 1.2142 / 1.4612 ms, plus per-call glue — uploads
**3.51 / 1.53 ms** (dominant), `prepare_dense` decode **0.70 / 1.38 ms**, host loss assembly 0.51 ms, D2H
readback 0.12 ms (`../r5/RCA.md` §1/§8). Op counts per warm call: **dense 310** vs **legacy 3** vs
**triton 1**.

### 7.4 Re-staging counterfactual (`--e2e-companion` − pure, same process)

| route | pure median (ms) | companion e2e median (ms) | glue = companion − pure (ms) |
|---|---|---|---|
| triton (seam rebound) | 0.054701 / 0.054717 | 0.138499 / 0.140706 | **0.0838 / 0.0860** |
| dense-iree | 1.333942 / 1.479640 | 8.595172 / 8.524295 | **7.2612 / 7.0447** |

The triton glue (< 0.09 ms) is the launch + one `aten::neg` + the driver's own bookkeeping; the dense glue
is ≈80× larger — the r5 host-glue block restated inside one process. Cross-checked against the *separate*
type-1 cells: triton e2e − pure = **76.7 / 83.2 µs** (different processes; the companion 83.8 / 86.0 µs is the
same quantity measured in-process). The f200 dense companion is noisy (8.52 ms vs the 6.32 ms cell) — read
it as a magnitude, not a value.

---

## 8. Compile economics and cache behaviour (`extra/compile_econ/`)

One compile per shape per fresh process; `builds=1`, `cache_hits=24`, `fallbacks=0` in all three runs.

| run (fresh process each) | `first_call_ms` | `compile_ms` | `construct_total_ms` | `stage_upload_ms` | warm median (ms) | min warm (ms) | call 1 (ms) |
|---|---|---|---|---|---|---|---|
| campaign `triton_f64` (warm disk cache) | 755.6 | 729.9 | 730.3 | 0.372 | 0.1314 | 0.1258 | 0.5657 |
| extra `warm_cache/triton_f64_warmcache.json` (same `TRITON_CACHE_DIR`) | 732.7 | 707.5 | 707.8 | 0.287 | 0.1312 | 0.1269 | 0.5302 |
| extra `fresh_cache/triton_f64_freshcache.json` (empty cache) | **1 785.4** | **1 754.6** | 1 754.9 | 0.252 | 0.1832 | 0.1772 | 0.5428 |

* **A warm *disk* cache does not make the first call free.** The warm re-run is −22.9 ms (−3.0 %) vs the
  campaign cell: Triton still rebuilds its in-process artifacts, so **≈0.71–0.73 s is the realistic per-shape
  JIT cost per fresh process** on this box.
* **An empty cache adds ≈1.02–1.05 s**: first call 1 785.4 ms (2.36× the campaign first call; compile
  1 754.6 ms = +1 024.7 ms over the campaign compile, 2.40×). `RUNLOG.md` prices the penalty as
  "+1 024.7 ms (+139 % over the warm cache)" — the same delta expressed against the warm extra's first call.
* **Warm medians are unaffected** by cache state (0.1312 vs 0.1832 ms; only call 1 is inflated in the fresh
  run and even then only to 0.54 ms — the fresh window settles ~0.18 ms inside a process that paid the cold
  compile in the same call stream). *Record-vs-prose note:* `RUNLOG.md` §6.5 describes that first-launch
  effect as "call 1 is 1.39 ms"; the records' call 1 is **0.5302 ms** (warm extra), **0.5657 ms** (campaign)
  and **0.5428 ms** (fresh cache) — this file follows the records; the 1.39 ms figure has no matching field.
* **Cache growth**: the warm cache is **41 dirs / 251 files before and after** (Δ 0 — a fully warm compile
  adds nothing); the fresh cache goes **0 → 3 dirs / 9 files** after one compile.
* One compile per *process and shape* is the operating point the round measured; the feasibility session
  established that in-process relaunch is free and recompiles happen only on a new signature/constexpr
  (`CONTEXT.md` "Key facts").

---

## 9. Anomaly window `w1c_20260914` (superseded, kept as evidence)

The same 10-cell protocol ran 5 minutes earlier and the pre-registered rule *"a median > 2× its anchor ⇒
re-run into a fresh label"* fired on the **legacy** side:

| cell | `w1c` median | `w1d` median | Δ `w1c` vs anchor | Δ `w1c` vs `w1d` |
|---|---|---|---|---|
| `leg_f64` | 2.4479 | 0.6364 | **+278.9 %** | +284.6 % |
| `leg_f200` | 2.2988 | 0.6269 | **+260.8 %** | +266.7 % |
| `dense_f64` | 6.7117 | 4.7860 | +24.7 % | +40.2 % |
| `dense_f200` | 8.7835 | 6.3179 | +51.7 % | +39.0 % |
| `triton_f64` | 0.1856 | 0.1314 | — | +41.3 % |
| `triton_f200` | 0.1992 | 0.1379 | — | +44.4 % |
| `densepure_f64` | 1.8588 | 1.3339 | +53.1 % | +39.3 % |
| `densepure_f200` | 1.4281 | 1.4796 | −2.3 % | −3.5 % |
| `tritonpure_f64` | 0.05590 | 0.05470 | — | +2.2 % |
| `tritonpure_f200` | 0.05472 | 0.05472 | — | 0.0 % |

First-call/compile costs were inflated by the same factor (legacy 66.4 / 71.1 ms; dense 24 935 / 26 733 ms;
triton first call 1 624.0 / 1 289.9 ms), i.e. the *whole window* ran slow — a window property, not a cell
property. The host was CPU-oversubscribed in **both** windows (load average 356→393 / 128 cores during w1c,
347–361 at session end), so load alone does not separate them; the discriminating fact is that `w1d`
restored all six cross-round anchors while `w1c` missed them by +25 %…+279 %, with the cheapest
(most host-scheduling-sensitive) call hit hardest and the two device-resident triton-pure cells differing by
≤2 %. `w1d` is primary; `w1c` is the recorded anomaly. Nothing was edited or re-run in place.

---

## 10. VERDICT

### Q1 — Is it semantically correct?

**Yes, for the single-output SR contract on the fixtures, with four declared and quantified deviations.**
Evidence: fixture losses/values all `allclose` (max\|Δ\| 7.63e-06 / 0.0625 on MSE, 4.77e-07 / 3.05e-05 on MAE,
max ULP 2); prediction values bit-identical-or-both-NaN **4096/4096** and **12800/12800**; legacy goldens
**12/12 `allclose`** (max\|Δ\| 1.53e-05); op coverage **31/31** declared and exercised, `uncovered_ops=[]`;
the four fixture ops are `+ − *` bit-equal and `/` correctly rounded (`div_rn`, core grid max\|Δ\| 5.9e-39);
the output contract (positive `(pop,)` float32, one launch, NaN/±inf unsanitized) is asserted per call. The
deviations are exactly four, each with a root cause and a fixture-impact proof: `dev-ftz-denormals` (9 552),
`dev-fast-math-ulp` (10 866), `dev-zero-sign` (440, run 1), `dev-const-neglzero` (4 000, diagnostic) — all
`affects_fixtures: false` because the fixtures contain no subnormals, zeros/±0, infinities or NaNs (§6.4).
The silent-fallback tripwire is green in all four triton cells, so no timed call can have been the dense or
torch path.

### Q2 — How fast is it, same window?

**Faster than the legacy CUDA engine, and far faster than the dense route it replaces.**

* type 1 (`problem.evaluate`, 25-call warm median): triton **0.1314 / 0.1379 ms** vs legacy
  **0.6364 / 0.6269** vs dense-iree **4.7860 / 6.3179** ⇒ triton is **4.8× / 4.6× faster than legacy** and
  **36.4× / 45.8× faster than dense**; dense-vs-legacy reads 7.52× / 10.08× here (the r5-measured ≈8–11×
  gap, unchanged).
* type 2 (pure, resident, no readback): triton **0.05470 / 0.05472 ms** vs dense pure 1.3339 / 1.4796 ⇒
  **24.4× / 27.0×**.
* The window reproduces all six r5 anchors within −11.1 %…+9.9 % (§3), so these ratios are same-window, not
  cross-window arithmetic.

### Q3 — What remains of the gap, and what would close it?

**The original ≈8–11× dense gap is not merely closed, it is inverted.** What is left is a different, much
smaller profile:

| what remains | value | evidence |
|---|---|---|
| host residual around the triton call | **≈50 µs/call** (50.4 / 45.5 µs), i.e. 90 % / 83 % of a pure call | §7.2 |
| — of which the kernel | 5.8 / 9.1 µs | §7.2 |
| public-API envelope delta (e2e vs pure) | 76.7 / 83.2 µs across processes; 83.8 / 86.0 µs in-process | §7.4 |
| one-time per-shape compile per fresh process | 0.73 s warm disk cache, 1.79 s empty | §8 |
| legacy's own host residual | 617.2 / 576.5 µs — *the thing triton already beats* | §7.1 |

Closers, ranked by the evidence (none of them a correctness change): (1) **CUDA-graph capture** — the
feasibility session verified `torch.cuda.CUDAGraph` can capture the Triton launch, which attacks the
launch-bound 50 µs without touching the kernel; (2) **trimming the host loop** — the remaining work per call
is a Python-side launch + driver bookkeeping (kernel time is 10–17 % of the call), so anything that removes
per-call Python/dispatch work maps almost 1:1 onto the wall; (3) **one-time compile** — amortized by raising
it to process start / caching artifacts (`TRITON_CACHE_DIR` already saves ≈1.02 s of 1.79 s). Nothing in the
records suggests the kernel itself needs work: it is 1 launch, and the legacy engine's 3-op device cost
(6.4 / 11.7 µs) is *higher* than triton's single launch (5.8 / 9.1 µs).

### Q4 — Coverage, limits, integration path, effort

**Coverage / limits.**
* **Single-output SR only.** Op coverage is complete for the engine's 31-entry single-output dispatch table
  and validated on `{+,-,*,/}`, but the multi-output `CombinedForest` route, classification/policy tasks and
  the robotics problems are **not** covered (they are outside the round's scope and outside the prototype's
  contract; `Forest.SR_fitness` is the only seam used).
* **CUDA/Triton runtime required.** Triton needs an NVIDIA GPU at compile and run time; the numpy backend
  remains the default reference, so a landed runner needs an explicit fallback path (today: the dense/specialized
  ETL routes).
* **Tolerances.** FTZ/denormal behaviour and transcendental last-ULP are policy decisions, not bugs the round
  landed (§6.4/§11.2); they matter for exact-bit expectations and for tasks with subnormal or `-0.0` data.
* **Per-shape compile** (0.73–1.79 s per fresh process) is inherent to the JIT design; cache-hit and
  build-count accounting must be part of any productized runner.
* **The type-1 numbers are envelope-patched**, not a landed selection path: the driver rebinds
  `evogp.tree.dense_engine.sr_fitness_dense`, which is a measurement device, not an integration.

**Integration options (summarized; nothing here was implemented).**
* **(a) EvoGP-side runner — recommended.** Replace the body behind the dense seam
  (`src/evogp/tree/forest.py:338-351`, or the route selection in `dense_engine.should_use_dense`) with a
  Triton evaluator chosen by an eval-mode/backend switch, keeping the existing
  `decode → stage → run → (optional loss)` shape. Advantages: no ETL interop, no foreign-repo change, the
  prototype's staging is already byte-identical to `dense_engine.prepare_dense`, and the exception type is
  already designed so a fallback can never be silent. The patch point is exactly where the round measured the
  win.
* **(b) ETL-side `external_call` + `register_external_kernel(device_resident=True)`.** On iree 3.11/CUDA the
  device payloads have no public raw-pointer/DLPack export, so this needs D2D copies into a buffer the
  external kernel can address plus a private HAL binding; ETL itself has no fusion/codegen infrastructure and
  no Triton usage anywhere. It is a strictly larger and more invasive path for the same single-output result,
  and it would sit behind the very HAL that the r5 analysis blames for the upload/dispatch cost.

**Effort estimate (grounded in the artifacts, no calendar claim).** The measured work already done is
877 lines of evaluator + 584 lines of driver + 1 887 lines of validator (+ 389 lines of trace probe, 963
lines of runner, 1 831 lines of gate). A productized runner would still need: (i) an eval-mode/backend
selection + fallback policy instead of a monkeypatch; (ii) an explicit op-coverage/semantics declaration
mechanism (the 31-entry table is hard-coded today) and a decision on the FTZ/1-ULP deviations; (iii) guards
for the unsupported routes (multi-output, non-SR tasks) that fail loudly; (iv) a non-CUDA fallback;
(v) compile/cache lifecycle (per-shape, per-process, cache dir, first-call budget) wired into the pipeline's
existing `CompileCache`-style accounting; (vi) tests in the pytest suite (the r6 prototype is deliberately
`test/`-only, with `src/` untouched) plus a Triton availability check. That is integration and policy work,
not performance work: the performance question this round asked is already answered.

### Q5 — Does the evidence support "same implementation ⇒ legacy-class performance"?

**YES — confirmed, and exceeded** (triton is 4.8× / 4.6× *faster* than legacy in the same window), with the
two caveats the round's own data forces:

1. **Flat-host-cost caveat.** Legacy's 0.63 ms is itself ≈97 % host-side (617.2 / 576.5 µs residual) and
   ≈flat from 64 to 1000 trees, so "legacy-class" must be read as **envelope-class**, not kernel-class. A
   kernel-only comparison is triton 5.825 / 9.101 µs vs legacy's three device ops of substance
   6.387 / 11.652 µs (fused walk 4.706 / 9.918, averaging 1.208 / 1.236, memset 0.473 / 0.498) — triton is
   *also* ahead there, but both are sub-12 µs components of a ~130 µs (triton) / ~630 µs (legacy) envelope.
2. **3-op-vs-310-op caveat.** The win mechanism is the **op-count collapse**: dense 310 device ops
   (249 dispatches + 61 DtoD copies, 125-deep chain, ~1.4 µs kernels) → triton **1 launch**, matching the
   legacy design's 3 device ops. This is not micro-optimization of the dense graph; it is replacing the
   compilation strategy for a fixed-shape, host-staged, single-output SR evaluation. Any port that keeps the
   310-op graph and merely adds a kernel would not reproduce this result.

---

## 11. Open items

1. **Validation run 2 is a different, uncommitted kernel revision (documented provenance caveat, not a
   failure).** `results/validation/validation.json` (run 1, 2026-09-14 00:35:17, `evogp_commit=e1f5dcb2…`)
   pins `triton_eval.py` at sha256 `163b5ffdb48b…`, which **is** the committed file; run 2
   (`results/validation/run2/validation.json`, 00:39:47, `evogp_commit=599181a5…`) declares
   `e155dd13c03a…`, which exists in **no git object**. Run 2's per-op evidence corroborates a semantically
   different kernel: the `zero_sign` class (440) is gone, `dev-zero-sign` disappears, `denormal_result`
   1 792 → 832 (so `dev-ftz-denormals` 9 552 → 8 592) while `denormal_operand` (7 760) and `finite_ulp`
   (10 866) are unchanged; `dev-const-neglzero` (4 000) is present in both. The fixture/goldens headline
   numbers are byte-identical between the runs, and `check_r6.py` asserts both facts and prints this as a
   NOTE. **Run 1 is the record of record**; run 2 is a same-headline replicate on an uncommitted variant.
2. **Deviation classes remain un-landed.** `dev-ftz-denormals`, `dev-fast-math-ulp`, `dev-zero-sign` and
   `dev-const-neglzero` are reported, not fixed (the round's scope was feasibility + measurement). Candidate
   mechanisms, in the order the round's own root causes point at: bit-preserving CONST staging instead of
   `tl.zeros + cval` (kills `dev-const-neglzero`); a sign-bit-exact `neg` (sign-bit flip rather than
   arithmetic negation) for `dev-zero-sign`; an explicit FTZ/denormal policy decision — the current behaviour
   is *legacy-consistent* (the legacy kernel was built `-use_fast_math`) but not oracle/ETL-consistent; and
   correctly-rounded replacements (e.g. an inline-PTX `sqrt.rn.f32`) for the 1-ULP transcendental classes if
   bit-exactness ever becomes a requirement. None of these were probed to a landed change in this round.
3. **Compile economics are first-call-only but real**: 0.73 s (warm disk cache) to 1.79 s (empty cache) per
   fresh process and shape, plus a slightly inflated call 1 even on a fully warm run (0.53–0.57 ms vs
   ~0.13 ms steady) — a Triton-side first-launch effect, unfixed by design (fixing it would change the
   prototype's semantics). Any productized runner must budget and amortize it.
4. **`densepure_*` provenance drift (report upward, unchanged):** `runner_source="frozen-api:build_dense_runner"`,
   `compile.path="frozen"`, `compile.build_ms`/`decode_plan_ms` null — the r4 node text describes these cells
   as `local-fallback:*`. No harness or `src/` change was made; the records are what the installed harness emits.
5. **Operator infrastructure note:** `nohup`/`setsid` backgrounding does not survive a tool call in this
   sandbox (`KillMode=control-group`); two aborted campaign attempts were discarded uncommitted and every
   multi-minute phase was launched under `systemd-run --user`. Recorded so a re-run does not repeat it.
6. **Not covered by the round** (and therefore not evidenced here): multi-output `CombinedForest`, the
   classification/policy problems, robotics rollouts, CUDA-graph capture *measured* in the campaign (it was
   only probe-verified in the feasibility session), Triton `num_warps`/`block_n` tuning beyond the fixed
   `num_warps=2`/`block_n=64`, and any non-A6000 GPU.

---

## 12. Artifact index (all paths relative to `test/benchmark/r6/`; `results/**` is immutable evidence)

| artifact | what it is |
|---|---|
| `results/w1d_20260914/{leg,dense,triton,densepure,tritonpure}_{f64,f200}.json` | the 10 primary-window records (each with `.log`; triton cells also `<id>.json.log`) |
| `results/w1d_20260914/window_meta.txt` | window start/end, per-cell exit table, GPU/tenant/env provenance |
| `results/w1d_20260914/check_deps.txt` / `.log` | dependency pre-flight, verdict `READY (20 ok / 0 warn / 0 missing)` |
| `results/w1d_20260914/extra/kernel_host/{triton_pure_f64,triton_pure_f200,triton_e2e_f64,legacy_e2e_f64,legacy_e2e_f200}/` | `facts.json` + `kineto_key_averages.txt` + `kineto_trace.json.gz` + `extra_trace.log` (the §7 decomposition) |
| `results/w1d_20260914/extra/compile_econ/{warm_cache/triton_f64_warmcache.json,fresh_cache/triton_f64_freshcache.json,cache_filecounts.txt}` | the §8 compile/cache economics (each with `.json.log`) |
| `results/w1c_20260914/**` | the superseded anomaly window (10 cells + `window_meta.txt` + `check_deps.*`) |
| `results/validation/validation.json` + `.log` | run 1 numeric validation (208 checks; record of record for the committed kernel revision) |
| `results/validation/run2/validation.json` | the same-headline replicate on an uncommitted revision (§11.1) |
| `RUNLOG.md` | operator log: window story, verbatim commands, per-cell numbers, anomalies, immutability statement |
| `check_r6.py` | the stdlib-only, GPU-free gate — `python3 test/benchmark/r6/check_r6.py` → 14 records / 1 440 checks / 0 failures (+10 anomaly cells, +5 fact sheets, +2 window files, +2 deps files, +30 logs, +1 RUNLOG) |
| `triton_eval.py`, `run_triton_eval64.py`, `validate_triton.py`, `extra_trace.py`, `run_cells.sh`, `run_under_lock.py` | the round's code (evaluator, drivers/validator, trace probe, campaign runner, lock helper) |
| `README.md`, `CONTEXT.md` | protocol + node brief (environment facts, protocol conventions, routing table) |
| `../r5/RCA.md` §8/§9, `../r5/results/q4_window/*.json` | the hypothesis and the same-window anchors of §3 |
| `../r4/SUMMARY.md`, `../r4/README.md` | the measurement-type conventions the type-1/type-2 cells follow |
| `../../src/evogp/tree/dense_engine.py`, `../../src/evogp/tree/forest.py` (seam `:338-351`), `../../src/evogp/tree/ENGINE_SPEC.md`, `../../src/evogp/tests/reference_eval.py` | read-only dependencies: staging, call site, op semantics of record, oracle |
| `../../test/goldens/forest_random.npz`, `../../test/goldens/sr_fitness.npz`, `../../test/benchmark/fixtures/forest200_tree64_random.npz` | the fixtures (fixture64 / legacy goldens / fixture200) |

Re-run, briefly: pre-flight `bash test/benchmark/r6/run_cells.sh --group check --window <W>` (fresh label
required) then `--group all --window <W>`; extras as in `RUNLOG.md` §7 (each under `run_under_lock.py` with
`timeout` inside); validate with `<venv>/bin/python test/benchmark/r6/validate_triton.py`; gate with
`python3 test/benchmark/r6/check_r6.py`. Session-scoped preconditions to re-check first: the legacy build at
`/tmp/evogp_legacy_bench` (+ `/tmp/evogp_t1a263/siteutils`), the Triton `Python.h` header tree
(`/tmp/triton_feas/py311hdr`), the warm compile cache `/tmp/evogp_r6_triton_cache`, and `nvidia-smi` showing
GPU 6 free of a compute-active tenant (a memory-resident one is the normal state here).
