# R4 — pure-forward before-baselines (analysis of record)

Round 4 adds a **second kind of measurement** to the old-vs-new benchmark: the *pure internal graph
forward* (measurement type 2) next to the end-to-end `problem.evaluate(forest)` numbers (type 1) that
R1–R3 quote.  This file is the analysis of record for the **before** campaign: everything below is
recomputed from the committed records under `results/` (the harness JSONs are the source of truth;
`RUNLOG.md` is the operator log with the same numbers plus the per-rep spreads).

**Status (before-baselines RECORDED, 2026-09-13, GPU 6).**  8/8 canonical configs complete
(`run_pure.sh`, one process each), gate `check_r4.py --manifest manifest.txt --require-companion` →
**8 complete / 369 checks / 0 failures**, pre-flight `check_agreement_r4.py` → **PASS (30 checks)**.
Every record is a `local-fallback:*` measurement at `evogp e272714`, `etl 51a7141`.  The after-round
(frozen runner APIs + the etl jit/hot-path work) re-runs this protocol and diffs against this table —
see §H.

Protocol: `README.md`.  Operator log: `RUNLOG.md`.  Records: `results/<id>.{json,log}`,
`results/backend_agreement/`.  Node context: `CONTEXT.md`.

---

## A. Context

| item | value |
|---|---|
| round | R4, measurement type 2 ("pure forward") + the pre-code-round baselines |
| engine under test (new side) | `evogp` @ `e2727140b70d1fc814c03b2c75b86613185fb2e9`, `PYTHONPATH=src` |
| `etl` checkout (`etl:` line in every log) | `/mnt/local-ssd/bchuang/etl` @ `51a7141885d76a6764a9ad43f6bf1264dd12e6fe` (the **patched** checkout R3 Arm B shadowed; stock `main` is `a863545`), `etl_version=0.1.0` |
| backends / routes | `numpy` (default) and `iree` (opt-in, `O1`, device 0) × `dense` and `specialized` |
| fixtures | `test/goldens/forest_random.npz` (64 trees, 63-node width, 64 data rows) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, 64-node width, 64 data rows) |
| window | HKT 2026-09-13 03:46–03:56 (pre-flight 03:46:00–03:48:26; numpy group 03:49:01–03:52:25; iree group 03:52:29–03:55:55) |
| what the after-round does | the parallel code round (frozen `build_{dense,specialized}_runner`, `EVOGP_EVAL_MODE=dynamic` alias, possibly adjusted auto gates + `EVOGP_DENSE_MAX_TREE_LEN`, etl `jit` + hot-path fixes) re-runs the same 8-config protocol and compares `frozen-api:*` against this fallback table (§H) |

Nothing here is an old-vs-new comparison: the legacy CUDA side is skipped by design (§G).  R4 measures
the new engine against **itself** across routes/backends, and prices the host glue that type 1 adds.

## B. The two measurement types

| | TYPE 1 — END-TO-END | TYPE 2 — PURE INTERNAL GRAPH FORWARD |
|---|---|---|
| task / flag | `--task eval64` \| `evalfresh` (the numbers of record for R1–R3) | `--task eval-pure` (new in R4) |
| one unit | one `problem.evaluate(forest)` call | one full-population forward on the fixed `x` (`per_row=False`) |
| executed | `Forest.SR_fitness` → per-call host `decode_population` → per-tree cache-key derivation / `build_cached` lookup → `etl.run` (one per tree; one per chunk on dense) → per-tree (or per-chunk) readback → host row assembly → `torch.from_numpy` wrap → fitness reduction | **only** the compiled graph call(s): dense = one `etl.run(exe, *cols, x, cfg)` per chunk; specialized = one `etl.run(exe, input, program)` per non-constant tree |
| excluded (hoisted out of the timed loop) | — | population decode, cache-key derivation / build lookup, readback (results dropped, device-resident), host assembly, fitness reduction, the one-time input conversion/upload, and compilation (`compile.construct_total_ms` / `build_ms`, reported separately) |
| still paid (the honest floor) | all of the above | `etl.run`'s own wrapper validation + the synchronous backend invoke |
| timing | K calls, `torch.cuda.synchronize()` per call | warmup W=10 once, then R=5 timed blocks of N invocations (N=100 dense / 50 specialized), each block **one** `torch.cuda.synchronize()` pair; per-invocation = `block_ms/N`; headline = median over the 5 reps (min/mean also recorded) |
| R4's own type-1 sample | `e2e_companion` block (`--e2e-companion 11`): K=11 synchronized `problem.evaluate` calls **in the same process / same warm cache** as the pure build, `median_warm_ms` is the headline | — |
| derived keys | `median_warm_ms` (and `per_call_ms[]`) | `per_graph_call_ms` = per-invocation / `n_graph_calls_per_run`; `per_tree_us` = per-invocation × 1000 / `pop`; `per_population_evaluate_ms` ≡ `median_per_invocation_ms` by definition (one invocation **is** one population evaluate) |

**HOST-GLUE DELTA** = companion E2E − pure per-invocation = per-call decode + key derivation +
readback + host assembly + fitness reduction for the same population and fixture — measured in one
process, one window, off the same warm cache, so it is not a cross-run subtraction.

**Provenance label.**  All 8 before-records carry `pure.runner_source = local-fallback:{dense|specialized}`
(and `compile.path = "fallback"`): the frozen `build_{dense,specialized}_runner` APIs did not exist at
`e272714`, so the 8 configs ran the harness' fallback constructors in `../pure_forward.py`.  The
after-round records must show `frozen-api:*`.

## C. Before-baselines (8/8 complete)

Medians, recomputed from `results/<id>.json`; `construct_total_ms` is the cold construction and is
**never** part of a pure number.

| id | backend | mode | `median_per_invocation_ms` (= per-population evaluate) | `per_graph_call_ms` | `n_graph_calls_per_run` | `per_tree_us` | `compile.construct_total_ms` (cold) | companion `median_warm_ms` | Δ = companion − pure (ms) | `pure.runner_source` |
|---|---|---|---|---|---|---|---|---|---|---|
| `pure64_dense_numpy` | numpy | dense | 128.936 | 128.9358 | 1 | 2 014.62 | 2 381.5 | 102.083 | **−26.85** (see note) | `local-fallback:dense` |
| `pure64_specialized_numpy` | numpy | specialized | 8.390 | 0.1583 | 53 | 131.09 | 154.7 | 20.093 | +11.70 | `local-fallback:specialized` |
| `pure200_dense_numpy` | numpy | dense | 181.686 | 181.6856 | 1 | 908.43 | 2 173.2 | 183.618 | +1.93 | `local-fallback:dense` |
| `pure200_specialized_numpy` | numpy | specialized | 49.332 | 0.2885 | 171 | 246.66 | 990.9 | 97.133 | +47.80 | `local-fallback:specialized` |
| `pure64_dense_iree` | iree | dense | 1.587 | 1.5867 | 1 | 24.79 | 13 522.7 | 4.841 | +3.25 | `local-fallback:dense` |
| `pure64_specialized_iree` | iree | specialized | 53.214 | 1.0040 | 53 | 831.47 | 18 817.6 | 59.706 | +6.49 | `local-fallback:specialized` |
| `pure200_dense_iree` | iree | dense | 3.696 | 3.6963 | 1 | 18.48 | 14 345.2 | 5.100 | +1.40 | `local-fallback:dense` |
| `pure200_specialized_iree` | iree | specialized | 214.655 | 1.2553 | 171 | 1 073.27 | 71 335.3 | 221.875 | +7.22 | `local-fallback:specialized` |

Rows are in the runner/manifest order (group `numpy` then `iree`; `pure64_*` before `pure200_*`;
`*_dense_*` before `*_specialized_*`).  `n_graph_calls_per_run` is `measured` in all 8 records: dense
is exactly one graph call per invocation; specialized is the non-constant tree count (53 = 64 − 11 at
fixture64, 171 = 200 − 29 at fixture200), backed by 48 / 155 distinct compiled structures (5 / 16
duplicate structures hit the shared `etl.CompileCache`).  Per-rep spreads, `decode_plan_ms` /
`build_ms` splits, cache deltas and the fd curves are in `RUNLOG.md` §"Headline numbers".

**Anomalous cell — `pure64_dense_numpy` (Δ = −26.85 ms).**  This is the one config whose companion is
*faster* than its pure number, and it is a documented CPU-contention artifact, not a structural
finding (`RUNLOG.md` §"Anomalies" #1): (a) its pure value 128.936 ms is within **+3.0 %** of the same
config's smoke value (125.163 ms, `results/smoke/pure64_dense_numpy.json`); (b) the pure block trends
steadily down across the 5 reps (148.84 → 120.26 ms, first rep the highest) and its 11 companion calls
span 101.5–214.2 ms — the box's CPU load moves ~2× on a sub-second scale inside this config (numpy is a
CPU interpreter, so it sees exactly that); (c) the same two paths on the same route four minutes later
(`pure200_dense_numpy`) agree to 1.1 % (181.686 pure vs 183.618 companion).  The **field run is
retained** — no rerun, no cherry-picking; quote the 128.9 ms pure number as comparable to 120–130 ms
and treat its companion delta as noise.

## D. Derived observations (medians)

* **IREE-dense is the fastest route overall**: 128.936 → 1.587 ms at 64 trees (**81.3×**) and
  181.686 → 3.696 ms at 200 trees (**49.2×**) versus numpy-dense.
* **On IREE, dense beats specialized**: per invocation, specialized is **33.5×** (64) and **58.1×**
  (200) *slower* than dense; per graph call it is 1.0040 vs 1.5867 ms (64, dense = 1.58× the
  specialized per-call cost) and 1.2553 vs 3.6963 ms (200, dense = 2.94×).  Specialized-iree warm cost
  is therefore **~1.00–1.26 ms per `etl.run`**, paid 53 / 171 times — per-tree dispatch dominates.
* **On numpy the ordering flips**: specialized is **15.4×** (64) and **3.68×** (200) *faster* than
  dense, i.e. one numpy interpreter graph over the whole population costs more than 53 / 171 small
  per-tree calls.  Cross-backend, specialized-numpy is 6.3× (64) / 4.35× (200) faster than
  specialized-iree.
* **64 → 200 scaling** (absolute per-invocation ×, and per-tree µs from → to): dense-numpy ×1.41
  (2 014.62 → 908.43), dense-iree ×2.33 (24.79 → 18.48), specialized-numpy ×5.88 (131.09 → 246.66),
  specialized-iree ×4.03 (831.47 → 1 073.27).  Both dense routes are *sub-linear* per tree (fixed
  per-call + wider call); the specialized routes grow ~super-linearly per tree because the call count
  itself scales with the population (53 → 171) and each numpy/iree call gets dearer.
* **Host-glue delta (E2E − pure)**: small where the graph dominates — +1.40 ms (200 dense iree),
  +1.93 ms (200 dense numpy), +3.25 ms (64 dense iree), +6.49 / +7.22 ms (specialized iree) — but
  **≈1× the pure number on the numpy-specialized route** (+11.70 ms on 8.390 ms = 139.5 %; +47.80 ms
  on 49.332 ms = 96.9 %): there the per-tree readback/host assembly costs as much as the forwards.
  The 64-dense-numpy cell is the artifact of §C.
* **Cold construction is a separate budget** and **≈330×–8 500×** (2.5–3.9 orders of magnitude) larger
  than the warm number on IREE: one dense graph costs 13 522.7 ms (64) / 14 345.2 ms (200) on iree versus 2 381.5 / 2 173.2 ms
  on numpy (nearly pop-independent on iree — one whole-population graph); the specialized builds cost
  18 817.6 ms (48 builds, ≈391 ms/build) and 71 335.3 ms (155 builds, ≈460 ms/build) on iree versus
  154.7 ms / 990.9 ms on numpy.  So iree pays ~13.5–71 s of compile for a warm pure number of
  1.6–214.7 ms per invocation, and per-build iree time is **not** a hardware constant (R3 caveat).

## E. Agreement pre-flight (IREE quotability gate)

Six records under `results/backend_agreement/` — 2 fixtures × 3 roles (`numpy`; `iree` with the
default `auto` policy → the dense route; `iree` + `EVOGP_EVAL_MODE=specialized`), one process each, all
exit 0, 2 m 28 s total (`fixture200_iree_spec` is the slow one at 71 s of compile).

**Verdict: PASS-with-note on both fixtures — `check_agreement_r4.py` exit 0, 30 checks / 0 failures.**

* `numpy` == `iree_spec` **bit-identically** on both fixtures (`values_sha256` equal:
  `2e5c0fa1fdbe…` for fixture64, `16d5bbe98481…` for fixture200).
* The `iree` (auto → dense) hash differs, but every role pair is inside
  `np.allclose(rtol=1e-4, atol=1e-4, equal_nan=True)`: `max|diff|` = **7.63e-06** (fixture64) /
  **3.91e-03** (fixture200), max relative diff **1.08e-07** / **2.03e-07**.  That is the known
  dense-vs-per-structure summation-order difference (one whole-population graph vs one graph per
  tree), not a miscompile — hence **PASS-with-note: all IREE timings in this file are quotable**, with
  the deviation recorded.
* NaN placement is identical across roles (3 NaN / 64 and 12 NaN / 200), and the identity checks
  re-hashed clean (`fixture_sha256` == sha256 of the npz; every `<stem>_values.npy` hashes back to its
  JSON's `values_sha256`).
* **Continuity:** all six R4 records are bit-identical (full sha256) to the R3 pre-flight records
  (`../r2/round3/results/backend_agreement/`), and the two `iree` (dense) hashes also match the R2
  anchors — the value paths did not move between R3 and R4.

| record | exit | started → finished | `values_sha256` | per_call_ms (cold, warm) |
|---|---|---|---|---|
| `fixture64_numpy` | 0 | 03:46:00 → 03:46:01 | `2e5c0fa1fdbe…` | 186.2, 21.0 |
| `fixture64_iree` | 0 | 03:46:03 → 03:46:19 | `b8192cb1770b…` | 14 964.5, 7.6 |
| `fixture64_iree_spec` | 0 | 03:46:21 → 03:46:50 | `2e5c0fa1fdbe…` | 27 711.3, 71.2 |
| `fixture200_numpy` | 0 | 03:46:52 → 03:46:54 | `16d5bbe98481…` | 1 074.4, 97.2 |
| `fixture200_iree` | 0 | 03:46:57 → 03:47:12 | `07948efededc…` | 14 915.1, 7.6 |
| `fixture200_iree_spec` | 0 | 03:47:15 → 03:48:26 | `16d5bbe98481…` | 70 798.1, 303.3 |

## F. Provenance and exact commands

| item | value |
|---|---|
| window | HKT 2026-09-13 03:46–03:56 (pre-flight 03:46:00–03:48:26; numpy group 03:49:01–03:52:25; iree group 03:52:29–03:55:55) |
| GPU | `CUDA_VISIBLE_DEVICES=6` (runner default), NVIDIA RTX A6000, ~30.5 GiB already used by the foreign tenant (`free_mb_start` 17 628–17 935 MiB across the 8 records); **GPU 7 never touched** |
| contention | a foreign tenant held ≈**30.5 GiB resident** on GPU 6 flat across the whole window (30 477–30 785 MiB); our own utilisation samples 0–73 % |
| lock | exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per record, `timeout` inside it, never nested; no lock wait, no timeout, `ulimit -n 65536` first |
| env | `PYTHONPATH=src`, `ETL_DISABLE_LOCATIONS=1`, `CUDA_VISIBLE_DEVICES=6`, plus per config `EVOGP_EVAL_MODE=<mode>` and `EVOGP_ETL_BACKEND`/`EVOGP_ETL_IREE_{DEVICE,OPT_LEVEL}`; all other knobs are the runner defaults (`--reps 5 --warmup 10 --invocations 100|50 --e2e-companion 11 --mem-guard-mb 8192`, `--max-cold-builds 400` on iree) |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
bash test/benchmark/r4/run_agreement_r4.sh                                   # pre-flight, 6 records
$PY  test/benchmark/r4/check_agreement_r4.py                                 # gate: PASS, 30 checks
bash test/benchmark/r4/run_pure.sh --group numpy                             # 4 configs
bash test/benchmark/r4/run_pure.sh --group iree                              # 4 configs
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
#   -> 8 record(s): 8 complete, 0 partial, 0 skipped, 0 missing, 0 malformed / 369 checks, 0 failures
```

`run_pure.sh --group all` runs the same 8 in one invocation (numpy group first); `--list` / `--dry-run`
print the config table / exact commands without executing.  Every log carries the `etl:` identity line
and every JSON records `evogp_commit`, `env`, `gpu` and the fd curve.

## G. Caveats

* **Shared box.**  Medians carry the usual ±10–20 % band; single cells (`pure64_dense_numpy`) can be
  worse and are flagged, not smoothed.  Only within-window comparisons are meaningful; all deltas above
  are same-process, same-window.
* **All before numbers are fallback-path.**  `local-fallback:{dense|specialized}` / `compile.path =
  "fallback"` — never present them as frozen-runner numbers.  The fallback uses the engine's own memo
  key and the shared `etl_engine.CACHE`, so the construction and warm path are the engine's; only the
  call wrappers are the harness'.
* **The legacy side has no separable pure forward** and is skipped by design: its `evaluate` is one
  fused kernel call including the readback and the host assembly, so the type-2 number is undefined
  there (`results/smoke/oldside_evalpure_skip.json`: `skipped: true`, exit 3, "eval-pure is a
  new-side measurement only").  R4 therefore has no old-vs-new row.
* **numpy "no readback"** means no engine-level D2H conversion — the numpy backend returns into host
  memory by construction; it is not a claim that numpy avoids a copy.
* **`--stage-probe` is not applicable to type 2** (it monkeypatches `etl.run`, i.e. exactly the loop
  being measured); it is refused by design, `stages` is `null` in all 8 records, and the cold/warm split
  comes from `compile` / `pure` only.
* **IREE per-structure compile time is not a hardware constant** (342–2 162 ms per build for identical
  build counts in R3); read `construct_total_ms` as "± a factor of a few".  IREE is opt-in,
  `O0`/`O1` only, and its numbers are quotable only after this round's pre-flight (§E).
* **Smoke records are plumbing evidence**, not round numbers (reduced `--reps/--invocations`), and are
  not manifest ids.
* **The record JSONs are immutable evidence** — this file, `RUNLOG.md`, `README.md` and `CONTEXT.md`
  are derived from them, never the other way round.

## H. What the after-round must do (checklist)

1. **Wait for the parallel code round to land** the frozen `dense_engine.build_dense_runner` /
   `etl_engine.build_specialized_runner` API, the `EVOGP_EVAL_MODE=dynamic` alias, any adjusted auto
   gates + `EVOGP_DENSE_MAX_TREE_LEN`, and the etl `jit` + hot-path fixes.
2. **Re-run the agreement pre-flight FIRST** (`run_agreement_r4.sh` + `check_agreement_r4.py`) — the
   ETL changed, so the IREE numbers must be re-gated before anything else is quoted.
3. **Re-run the matrix unchanged** (`run_pure.sh --group all`, or `--group numpy`/`--group iree`) and
   then **both** checkers (`check_agreement_r4.py`, `check_r4.py --manifest manifest.txt
   --require-companion`) — the gate must stay at 8 complete / 0 failures.
4. **Verify the provenance flip**: every after-record must carry `pure.runner_source = frozen-api:*`
   (`compile.path = "frozen"`), and the two `FROZEN-ADAPTER` seams in `../pure_forward.py` must be
   validated against the landed signature — the assumed call form is `build_*(forest, x)` with a
   no-arg `.run()`.  Include an **A/B run** (`--pure-runner fallback` vs `frozen`) on ≥1 config per
   mode, and a **numeric equivalence check**: pure outputs are dropped today, so add an optional
   one-off checksum/readback *outside* the timed region (or a dedicated equivalence script) to prove
   the two constructors evaluate the same thing.  A frozen runner without a graph-call counter records
   `n_graph_calls_source: "assumed-1"` — never quote its `per_graph_call_ms` as measured.
5. **Build the after-table with the same columns as §C**, side by side with the before-table, note the
   `dynamic`-alias and gate changes, and fold the result into `../REPORT.md` (Round-4 verdict) plus
   `../CONTEXT.md` / `../README.md`.  That update belongs to that later round, **not** to R4.
6. **Keep the before-records immutable** — no reruns, no edits, no re-derivation: they are the baseline
   the whole comparison rests on.
