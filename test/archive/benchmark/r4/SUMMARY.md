# R4 — pure-forward round: before-baselines + after-round (analysis of record)

Round 4 adds a **second kind of measurement** to the old-vs-new benchmark: the *pure internal graph
forward* (measurement type 2) next to the end-to-end `problem.evaluate(forest)` numbers (type 1) that
R1–R3 quote.  This file is the analysis of record for **both halves of the round**: the pre-code-round
**before**-baselines and the **after**-round that re-ran them against the landed frozen runner APIs.
Everything below is recomputed from the committed records (the harness JSONs are the source of truth;
`RUNLOG.md` is the operator log with the same numbers plus the per-rep spreads).

**Status (before-baselines AND after-round RECORDED, 2026-09-13, GPU 6).**  *Before*, HKT 03:46–03:56:
8/8 canonical configs complete (`run_pure.sh`, one process each) at `evogp e272714` / `etl 51a7141`,
every record `local-fallback:*`; gate `check_r4.py --manifest manifest.txt --require-companion` →
**8 complete / 369 checks / 0 failures**; pre-flight `check_agreement_r4.py` → **PASS (30 checks)**.
*After*, HKT 07:34–07:52: the identical 8-config protocol re-run at `evogp 6e8bf6f` on a **read-only
shadow export of `etl 87c7562`**, every record `frozen-api:*`; gate `check_r4.py --manifest
manifest_after.txt --require-companion` → **8 complete / 369 checks / 0 failures**; the new type-1 E2E
matrix (`run_e2e.sh` + `check_e2e_r4.py`) → **14 complete / 328 checks / 0 failures**; the after
pre-flight `check_agreement_r4.py --agree-dir results/backend_agreement_after` → **PASS-with-note
(30 checks / 0 failures)**.

Protocol: `README.md`.  Operator log: `RUNLOG.md`.  Records: before `results/<id>.{json,log}` +
`results/backend_agreement/`; after `results/after/*.{json,log}` (pure), `results/after/e2e/` (type 1),
`results/after/ab/` (constructor A/B), `results/backend_agreement_after/`.  Node context: `CONTEXT.md`.

---

## Verdict (after-round, frozen-api)

* **The provenance flip is complete and numerically inert.**  All 8 after-records carry
  `pure.runner_source = frozen-api:build_{dense,specialized}_runner`, `compile.path = "frozen"` and
  `n_graph_calls_source = "introspected"` (1 / 53 / 171 graph calls per invocation).  The two
  `FROZEN-ADAPTER` seams in `../pure_forward.py` (the assumed `build_*(forest, x)` call form and the
  no-arg `.run()`) fit the landed API with **no adjustment**, and the frozen-vs-fallback A/B on
  `pure64_dense_numpy` + `pure64_specialized_numpy` is exact: equal `output_checksum` digests,
  byte-identical `.npz` captures, `max|diff| = 0` on every one of the 53 captured arrays with identical
  NaN placement (`results/after/ab/ab_equivalence_report.txt`) — the constructor swap cannot explain
  the before→after deltas.
* **The pure numbers improved on every one of the 8 configs**: numpy dense **−41.6 % / −46.8 %**, numpy
  specialized **−21.1 % / −38.5 %**, iree dense **−8.8 % / −56.9 %**, and the two specialized-iree
  cells collapse by **−88.9 %** (53.214 → 5.922 ms) and **−91.3 %** (214.655 → 18.570 ms) — their
  per-`etl.run` cost falls from 1.004 / 1.255 ms to 0.112 / 0.109 ms (§C.2).
* **Route ordering changed on the specialized route**: numpy specialized used to be 6.3× / 4.4× faster
  than iree specialized; after the round iree specialized is the faster one (1.12× / 1.63×).  On the
  dense routes iree stays far ahead of numpy (52× / 61×), and on iree dense stays ahead of specialized
  (4.1× / 11.7×) (§C.2 / §D.2).
* **Type 1 (E2E, 14 cells × 25 warm calls) is now ≈6.1–6.7 ms** on the iree `auto` route (shadow 6.667 /
  6.050 ms; same-window stock twins 6.556 / 6.461 ms ⇒ the patched ETL is **warm-neutral**, +1.7 % /
  −6.4 %).  Against the legacy anchor (0.6323 / 0.6357 ms) the old-vs-new gap is **≈9.5–10.5×**, down
  from R3's ≈16.5–16.8× — a cross-window comparison in which the same-window stock twin shows the gain
  comes from the evogp side, not from the ETL shadow (§ "Type 1 (E2E) after-round").
* **The IREE pre-flight still passes**: `--agree-dir results/backend_agreement_after` → PASS-with-note
  (30 checks / 0 failures) — numpy == iree-specialized bit-identical per fixture and iree-dense within
  `1e-4`; the iree-dense value path moved before→after while the other four roles' hashes are unchanged
  (§E).
* **Everything above is quotable only with §G's caveats**: the ±10–20 % shared-box band, cross-window
  comparisons marked as such, and `results/**` as immutable evidence.

---

## A. Context

| item | value |
|---|---|
| round | R4, measurement type 2 ("pure forward") + the pre-code-round baselines, and the after-round re-measurement (metric type 2 again, plus a metric type 1 E2E matrix) |
| engine under test (new side) — BEFORE | `evogp` @ `e2727140b70d1fc814c03b2c75b86613185fb2e9`, `PYTHONPATH=src` |
| engine under test (new side) — AFTER | `evogp` @ `6e8bf6f0ff7ee2d82e890bca464c3c833f5fae40`, `PYTHONPATH=src`, and `etl` = a **read-only shadow export** of upstream commit `87c7562713b75b235e4945c6311de054652fc919` at `/tmp/evogp_etl_patched_87c7562` (materialized with `git archive 87c7562 \| tar -x -C …`, never written inside the foreign repo; the editable `etl` install itself stays at stock `51a7141` — identity chain in §E/§G) |
| `etl` checkout (`etl:` line in every log; the **editable** install) | `/mnt/local-ssd/bchuang/etl` @ `51a7141885d76a6764a9ad43f6bf1264dd12e6fe` (the **patched** checkout R3 Arm B shadowed; stock `main` is `a863545`), `etl_version=0.1.0`; in the after-records the same line carries `file=/tmp/evogp_etl_patched_87c7562/etl/__init__.py` — the `rev=` field is the editable repo's HEAD, the `file=` path is the shadow identity (§G) |
| backends / routes | `numpy` (default) and `iree` (opt-in, `O1`, device 0) × `dense` and `specialized` (E2E after-round adds `auto` and the `dynamic` alias) |
| fixtures | `test/goldens/forest_random.npz` (64 trees, 63-node width, 64 data rows) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, 64-node width, 64 data rows) |
| window — BEFORE | HKT 2026-09-13 03:46–03:56 (pre-flight 03:46:00–03:48:26; numpy group 03:49:01–03:52:25; iree group 03:52:29–03:55:55) |
| window — AFTER | HKT 2026-09-13 07:34–07:52 in **two parallel worker sessions** interleaved through the one shared `flock`: pre-flight 07:34:54–07:42:16; pure matrix 07:44:37–07:50:52 (numpy group ≈2 m 18 s after one sandbox teardown, iree group ≈2 m 49 s); E2E matrix 07:44:34–07:51:43 in 3 chunks |
| what the after-round did | the parallel code round landed the frozen `build_{dense,specialized}_runner` hooks (§H item 1), the `EVOGP_EVAL_MODE=dynamic` → `dense` alias, the auto gates + `EVOGP_DENSE_MAX_TREE_LEN`, and the etl `jit` + hot-path work; this round re-ran the same 8-config protocol against them and adds a 14-cell metric type 1 matrix — §C.2 (pure) and the "Type 1 (E2E) after-round" section, with the diff against the before-table in the verdict above |

Nothing here is a fresh old-vs-new run: the legacy CUDA side is skipped by design (§G) and appears only
as the quoted R3 anchor in the E2E section.  R4 measures the new engine against **itself** across
routes/backends, prices the host glue that type 1 adds, and diffs the code round's before/after.

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
`e272714`, so the 8 configs ran the harness' fallback constructors in `../pure_forward.py`.  All 8
after-records carry `frozen-api:*` / `compile.path = "frozen"` (§C.2), and the two constructors were
shown to be numerically identical before anything was compared (§ "A/B" in §H item 4).

## C. Before-baselines (8/8 complete)

Medians, recomputed from `results/<id>.json`; `construct_total_ms` is the cold construction and is
**never** part of a pure number.

The before-table is reproduced unchanged (it is the baseline the whole comparison rests on):

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

## C.2 After-round (8/8 complete, `frozen-api:*`)

Same 8 configs, same protocol, re-run at `evogp 6e8bf6f` on the ETL shadow export — same columns as the
before-table plus the Δ against it.  Medians recomputed from `results/after/<id>.json`.

| id | backend | mode | `median_per_invocation_ms` (= per-population evaluate) | `per_graph_call_ms` | `n_graph_calls_per_run` | `per_tree_us` | `compile.construct_total_ms` (cold) | companion `median_warm_ms` | Δ = companion − pure (ms) | `pure.runner_source` | Δ vs before |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pure64_dense_numpy` | numpy | dense | 75.230 | 75.2302 | 1 | 1 175.47 | 1 682.1 | 64.966 | **−10.26** (see note) | `frozen-api:build_dense_runner` | −53.71 (**−41.6 %**) |
| `pure64_specialized_numpy` | numpy | specialized | 6.616 | 0.1248 | 53 | 103.38 | 154.9 | 11.206 | +4.59 | `frozen-api:build_specialized_runner` | −1.77 (**−21.1 %**) |
| `pure200_dense_numpy` | numpy | dense | 96.717 | 96.7171 | 1 | 483.59 | 1 760.2 | 84.137 | **−12.58** (see note) | `frozen-api:build_dense_runner` | −84.97 (**−46.8 %**) |
| `pure200_specialized_numpy` | numpy | specialized | 30.360 | 0.1775 | 171 | 151.80 | 1 020.9 | 50.670 | +20.31 | `frozen-api:build_specialized_runner` | −18.97 (**−38.5 %**) |
| `pure64_dense_iree` | iree | dense | 1.447 | 1.4468 | 1 | 22.61 | 11 611.1 | 8.477 | +7.03 | `frozen-api:build_dense_runner` | −0.14 (**−8.8 %**) |
| `pure64_specialized_iree` | iree | specialized | 5.922 | 0.1117 | 53 | 92.53 | 12 851.3 | 60.410 | +54.49 | `frozen-api:build_specialized_runner` | −47.29 (**−88.9 %**) |
| `pure200_dense_iree` | iree | dense | 1.593 | 1.5926 | 1 | 7.96 | 11 663.4 | 6.462 | +4.87 | `frozen-api:build_dense_runner` | −2.10 (**−56.9 %**) |
| `pure200_specialized_iree` | iree | specialized | 18.570 | 0.1086 | 171 | 92.85 | 42 603.6 | 195.368 | +176.80 | `frozen-api:build_specialized_runner` | −196.08 (**−91.3 %**) |

* **`n_graph_calls_per_run` is `introspected` in all 8 records** (the landed runners expose their own
  counter, so no record carries the `assumed-1` fallback): 1 for dense, the non-constant tree count for
  specialized (53 = 64 − 11, 171 = 200 − 29).
* **The frozen construction is opaque**: `compile.build_ms` and `compile.decode_plan_ms` are `null` in
  every after-record (the builders *are* the construction), so only `construct_total_ms` has an
  after-counterpart — the before-table's decode/build split cannot be recomputed there.  Cache counters:
  `cache_misses` 1 / 48 / 1 / 155 (the same work as before) and `cache_hits` 0 in all 8 (the before
  records counted 5 / 16 hits on the specialized numpy routes over the same 53 / 171 invocations — a raw
  counter difference between the two constructors, noted here and not interpreted).
* **Per-rep spread** (5 timed blocks, ms/invocation, *derived*): 77.63/76.56/74.98/75.23/74.50,
  6.67/6.78/6.58/6.62/6.59, 103.86/97.42/96.72/96.12/95.19, 30.84/30.58/29.78/30.36/29.97,
  1.72/1.46/1.44/1.45/1.38, 7.21/5.96/5.57/5.75/5.92, 1.78/1.56/1.59/1.59/1.60,
  18.57/18.54/17.73/19.00/22.11.
* **The companion-below-pure pair is back**: both numpy-dense cells have companion < pure (−10.26 /
  −12.58 ms) — the same class as the before-round anomaly (`RUNLOG.md` §"Anomalies" #1), reported as
  measured, with no rerun and no cherry-picking.  Every other cell has pure < companion.
* The two `median_warm_ms` values above are the `--e2e-companion 11` numbers **of the same process**
  (same warm cache), so `companion − pure` prices the per-call decode + key derivation + readback + host
  assembly + fitness reduction (§B); the type 1 numbers of record are the 25-rep cells of § "Type 1 (E2E)
  after-round".

## D. Derived observations (before-round, medians)

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

### D.2 After-round (computed from §C.2 against §C)

* **Every route got faster, and the specialized-iree route changed class**: its per-invocation cost fell
  9.0× (64) / 11.6× (200), driven by the per-`etl.run` cost (1.0040 → 0.1117 ms; 1.2553 → 0.1086 ms) —
  the per-tree dispatch that dominated the before-round is now ≈0.11 ms per call on both fixtures.
* **iree dense is no longer 33–58× ahead of iree specialized**: it is now 4.1× (5.922 vs 1.447 ms) at
  64 trees and 11.7× (18.570 vs 1.593 ms) at 200 trees.  iree dense itself moved only −8.8 % / −56.9 %
  (1.447 / 1.593 ms), so the compression comes almost entirely from the specialized side.
* **The numpy routes moved less than iree but still clearly**: dense 128.936 → 75.230 ms (×1.71) and
  181.686 → 96.717 ms (×1.88); specialized 8.390 → 6.616 ms (×1.27) and 49.332 → 30.360 ms (×1.63).
  On numpy the ordering does **not** flip: specialized stays 11.4× (64) / 3.19× (200) faster than dense.
* **The backend ordering on the specialized route inverted**: before, numpy specialized was 6.3× / 4.4×
  *faster* than iree specialized; after, iree specialized is 1.12× (64) / 1.63× (200) faster than numpy
  specialized — same fixtures, same window.
* **Cold construction got cheaper on iree too, and is still a separate budget**: dense 13 522.7 →
  11 611.1 ms (64) and 14 345.2 → 11 663.4 ms (200); specialized 18 817.6 → 12 851.3 ms (48 builds,
  ≈268 ms/build) and 71 335.3 → 42 603.6 ms (155 builds, ≈275 ms/build) — i.e. ≈2 200×–8 000× the warm
  number on iree, and still not a hardware constant (§G).
* **The host-glue delta (companion − pure) shrank where it was largest on numpy but grew on iree
  specialized** — numpy specialized +11.70 → +4.59 ms (64) and +47.80 → +20.31 ms (200); iree specialized
  +6.49 → +54.49 ms (64) and +7.22 → +176.80 ms (200), because its *companion* barely moved
  (59.706 → 60.410 ms, 221.875 → 195.368 ms) while the pure number collapsed under it.  That residual is
  exactly what the type 1 numbers of the next section price directly.

## Type 1 (E2E) after-round — the 14-cell matrix

`problem.evaluate(forest)` — the measurement R1–R3 quote: a fixed population, `--eval-reps 25`
synchronized warm calls, one `bench_gp.py` process per cell, `median_warm_ms` the headline.  The 12
**shadow** cells re-measure the R3 Arm A matrix (both backends × `auto` / `dynamic` / `specialized`);
the 2 **stock** cells are the headline iree-`auto` cells against the *editable* etl install, run in the
same window, so "patched vs stock" is a same-window delta.  Runner `run_e2e.sh`, validator
`check_e2e_r4.py`, expectation list `e2e_manifest.txt`; records `results/after/e2e/e2e_*.json`.

| id | kind | fixture | backend | mode | `median_warm_ms` | `first_call_ms` | warm min–max (calls 1–24) | fds start→cold→end | cold misses |
|---|---|---|---|---|---|---|---|---|---|
| `e2e_fixture64_iree_auto` | shadow | fixture64 | iree | auto | 6.667 | 10 060.5 | 6.50–9.34 | 39→91→91 | 1 |
| `e2e_stock_fixture64_iree_auto` | stock | fixture64 | iree | auto | 6.556 | 10 069.3 | 5.42–9.84 | 39→91→91 | 1 |
| `e2e_fixture200_iree_auto` | shadow | fixture200 | iree | auto | 6.050 | 10 569.7 | 5.82–9.03 | 39→91→91 | 1 |
| `e2e_stock_fixture200_iree_auto` | stock | fixture200 | iree | auto | 6.461 | 10 393.5 | 6.32–9.42 | 39→91→91 | 1 |
| `e2e_fixture64_numpy_auto` | shadow | fixture64 | numpy | auto | 10.844 | 191.7 | 10.70–13.00 | 39→59→59 | — (numpy) |
| `e2e_fixture64_numpy_dynamic` | shadow | fixture64 | numpy | dynamic | 71.373 | 1 537.0 | 64.72–213.64 | 39→59→59 | — (numpy) |
| `e2e_fixture64_numpy_specialized` | shadow | fixture64 | numpy | specialized | 10.853 | 192.5 | 10.76–12.26 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_auto` | shadow | fixture200 | numpy | auto | 52.891 | 1 054.2 | 51.25–66.33 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_dynamic` | shadow | fixture200 | numpy | dynamic | 85.244 | 1 639.7 | 83.57–274.74 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_specialized` | shadow | fixture200 | numpy | specialized | 54.122 | 1 121.2 | 52.97–63.77 | 39→59→59 | — (numpy) |
| `e2e_fixture64_iree_dynamic` | shadow | fixture64 | iree | dynamic | 7.398 | 10 176.3 | 7.14–10.14 | 39→91→91 | 1 |
| `e2e_fixture64_iree_specialized` | shadow | fixture64 | iree | specialized | 59.554 | 12 983.8 | 57.24–79.77 | 39→91→91 | 48 |
| `e2e_fixture200_iree_dynamic` | shadow | fixture200 | iree | dynamic | 6.348 | 10 358.9 | 5.04–12.07 | 39→91→91 | 1 |
| `e2e_fixture200_iree_specialized` | shadow | fixture200 | iree | specialized | 193.086 | 43 381.9 | 185.13–299.35 | 39→91→91 | 155 |

* **Same-window stock ↔ shadow (the headline pairs, the only two cells with a twin)**: fixture64
  6.556 → 6.667 ms (+0.111 ms, **+1.7 %**) and fixture200 6.461 → 6.050 ms (−0.411 ms, **−6.4 %**), i.e.
  the patched-ETL shadow is **warm-neutral inside the ±10–20 % shared-box band**.  Their `first_call_ms`
  are 10 069.3 / 10 393.5 (stock) vs 10 060.5 / 10 569.7 (shadow) — the ≈10.1–10.6 s first call is the
  one-time dense-graph compile, the same order as R3's Arm A cold first calls (13 640.3 / 13 207.4 ms,
  cross-window).
* **Against R3 and the legacy anchor**: R3 Arm A quoted 10.64 / 10.50 ms warm on exactly these two
  cells, so this window is **1.60× / 1.74× faster** — and because the *stock* twin measures 6.556 /
  6.461 ms in the same window, the gain is attributable to the evogp side, not to the ETL patch.  Against
  the legacy CUDA anchor (0.6323 / 0.6357 ms) the old-vs-new gap is now ≈10.5× / 9.5× (shadow) and
  ≈10.4× / 10.2× (stock) — **≈9.5–10.5×**, down from R3's ≈16.5–16.8×.  All of these are cross-window
  comparisons; only the stock↔shadow pair above is same-window.
* **numpy context (cross-window)**: R3 Arm A measured the same two workloads (numpy default,
  `auto` → specialized) at 23.293 / 117.612 ms; this window has them at 10.844 / 52.891 ms (≈2.2×
  faster).  As in R3, `auto` on numpy resolves to the **specialized** route (10.844 ≈ 10.853 ms at 64,
  52.891 ≈ 54.122 ms at 200), while `dynamic` is the dense alias and is 6.6× / 1.6× slower.
* **Route observations (shadow)**: on iree, `auto` ≈ `dynamic` (6.667 / 7.398 ms at 64; 6.050 / 6.348 ms
  at 200) — both are the dense evaluator — while `specialized` costs 8.9× / 31.9× more per call
  (59.554 / 193.086 ms), i.e. the per-tree dispatch is still the dominant iree term at type 1.
  On numpy, `auto` ≈ `specialized` (10.844 / 10.853 ms at 64; 52.891 / 54.122 ms at 200) « `dynamic`
  (71.373 / 85.244 ms).
* **Stage-probe split** (`--stage-probe` *is* installed for this task — unlike `--task eval-pure`): the
  `auto` / `dynamic` iree cells report their cost in the uninstrumented `misc` bucket
  (`build_calls = 0`, dense route), the iree `specialized` cells are dominated by the cold
  `build_miss` inside call 0 (**12 809 ms** / **42 868 ms** = 48 / 155 builds) plus a `readback` of
  **445.8 ms** / **1 354.8 ms** over the 25 calls, and the numpy cells split into
  `build_miss` 153–978 ms, `run` 232–1 237 ms, `decode` 27.6–115.9 ms, `readback` 16.0–59.7 ms.
* **fds**: 59 flat on every numpy cell, 91 flat on every iree cell (39 → 91 after the cold phase);
  no IREE descriptor growth, exit code 0 on all 14 cells (IREE processes still print a `nanobind`
  shutdown dump — §G).

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

### After-round pre-flight (`--agree-dir results/backend_agreement_after`)

The same six records re-run on the ETL shadow export at 07:34–07:42, all exit 0 (the fixture64
specialized record was resumed via `--only` after a sandbox teardown — see §G).  Gate
`check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after` → exit 0,
**PASS-with-note on both fixtures, 30 checks / 0 failures**; the default-dir gate (before records) stays
green as well.

| record | exit | started → finished | `values_sha256` | per_call_ms (cold, warm) |
|---|---|---|---|---|
| `fixture64_numpy` | 0 | 07:34:56 → 07:34:57 | `2e5c0fa1fdbe…` | 217.8, 13.3 |
| `fixture64_iree` | 0 | 07:34:59 → 07:35:11 | `fbe52b281581…` | 11 447.6, 9.9 |
| `fixture64_iree_spec` | 0 | 07:40:55 → 07:41:09 | `2e5c0fa1fdbe…` | 13 639.9, 80.2 |
| `fixture200_numpy` | 0 | 07:41:11 → 07:41:13 | `16d5bbe98481…` | 1 168.8, 64.4 |
| `fixture200_iree` | 0 | 07:41:16 → 07:41:28 | `74a46ed4bf75…` | 11 657.3, 7.8 |
| `fixture200_iree_spec` | 0 | 07:41:30 → 07:42:15 | `16d5bbe98481…` | 44 850.8, 237.3 |

**What moved before → after (per role):**

* `numpy` and `iree_spec` are **bit-identical before → after** under both fixtures (4/6 records:
  `2e5c0fa1fdbe…` / `16d5bbe98481…`) — the value paths of the reference and per-structure routes did not
  move with the code round.
* The two `iree` (auto → dense) vectors **did** move: fixture64 `b8192cb1770b…` → `fbe52b281581…`,
  fixture200 `07948efededc…` → `74a46ed4bf75…`.  Both stay inside the fallback tolerance against numpy:
  `max|diff|` **7.629e-06** / **0.03125**, max **relative** diff 1.06e-07 / 1.81e-07 (before:
  7.629e-06 / 3.906e-03 and 1.08e-07 / 2.03e-07).  Same in-tolerance class as the before round, i.e. the
  documented dense-vs-per-structure summation-order difference, so the after IREE timings stay quotable
  **with this deviation recorded**.
* NaN/±inf placement is unchanged (3 NaN / 64, 12 NaN / 200, 0 inf), and every identity check re-hashed
  clean (`fixture_sha256` == the npz, every `<stem>_values.npy` == its JSON's `values_sha256`).
* The before-window continuity statement above (all six before records bit-identical to R3) therefore
  holds for the before set only; in the after set the **dense** role is the one that moved.

## F. Provenance and exact commands

| item | value |
|---|---|
| window | HKT 2026-09-13 03:46–03:56 (pre-flight 03:46:00–03:48:26; numpy group 03:49:01–03:52:25; iree group 03:52:29–03:55:55) |
| GPU | `CUDA_VISIBLE_DEVICES=6` (runner default), NVIDIA RTX A6000, ~30.5 GiB already used by the foreign tenant (`free_mb_start` 17 628–17 935 MiB across the 8 records); **GPU 7 never touched** |
| contention | a foreign tenant held ≈**30.5 GiB resident** on GPU 6 flat across the whole window (30 477–30 785 MiB); our own utilisation samples 0–73 % |
| lock | exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per record, `timeout` inside it, never nested; no lock wait, no timeout, `ulimit -n 65536` first |
| env | `PYTHONPATH=src`, `ETL_DISABLE_LOCATIONS=1`, `CUDA_VISIBLE_DEVICES=6`, plus per config `EVOGP_EVAL_MODE=<mode>` and `EVOGP_ETL_BACKEND`/`EVOGP_ETL_IREE_{DEVICE,OPT_LEVEL}`; all other knobs are the runner defaults (`--reps 5 --warmup 10 --invocations 100` on dense and `50` on specialized, `--e2e-companion 11 --mem-guard-mb 8192`, `--max-cold-builds 400` on iree) |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124) |

**After-round provenance** (same interpreter, same lock, same GPU policy):

| item | value |
|---|---|
| window | HKT 2026-09-13 07:34–07:52 (pre-flight 07:34:54–07:42:16; pure matrix 07:44:37–07:50:52; E2E 07:44:34–07:51:43 in 3 chunks) — two parallel worker sessions interleaved through the one shared lock |
| engine / ETL | `evogp` @ `6e8bf6f0ff7ee2d82e890bca464c3c833f5fae40`; `PYTHONPATH=src:/tmp/evogp_etl_patched_87c7562` (read-only shadow export of etl `87c7562`; `results/after/etl_import_probe.txt` proves the precedence in fresh processes and pins the changed files by sha256) |
| GPU | `CUDA_VISIBLE_DEVICES=6` (NVIDIA RTX A6000), **GPU 7 never touched**; GPU 6 was idle at the start (0 %, 1 MiB) while foreign tenants held GPUs 0–2 at 90–100 %; the pure and E2E logs show 0–4 % utilisation and 1–313 MiB of peer contexts (the parallel worker's) |
| lock | exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per record, `timeout` inside it, never nested; no timeout on any of the 28 after-records; `ulimit -n 65536` first |
| fds | numpy records end at **59** fds (39 → 59), every iree record at **91** (39 → 91, flat after the cold phase) — the patched-ETL profile, not stock's per-structure growth |
| env | `PYTHONPATH` as above, `ETL_DISABLE_LOCATIONS=1`, `CUDA_VISIBLE_DEVICES=6`, `EVOGP_EVAL_MODE=<mode>` (+ `--mode` on the pure task), `EVOGP_ETL_BACKEND`/`EVOGP_ETL_IREE_{DEVICE,OPT_LEVEL}`; the pure matrix keeps the runner defaults (`--reps 5 --warmup 10 --invocations 100` on dense / `50` on specialized, `--e2e-companion 11 --mem-guard-mb 8192 --output-checksum`, `--max-cold-builds 400` on iree), the E2E cells `--task eval64 --eval-reps 25 --mem-guard-mb 8192 --stage-probe` |

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

**The after-round, verbatim** (all from the repository root; every heavy phase under the one shared
flock, `timeout` inside it):

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
SHADOW=/tmp/evogp_etl_patched_87c7562                 # git -C /mnt/local-ssd/bchuang/etl archive 87c7562 | tar -x -C …
# --- STEP 1: agreement pre-flight into the after dir -------------------------
NEW_PYTHONPATH="src:$SHADOW" bash test/benchmark/r4/run_agreement_r4.sh \
    --agree-dir test/benchmark/r4/results/backend_agreement_after
$PY test/benchmark/r4/check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after
#   -> gate PASS (30 checks, 0 failures), PASS-with-note on both fixtures
# --- STEP 2: the same 8-config pure matrix, after-artifacts dir --------------
NEW_PYTHONPATH="src:$SHADOW" NEW_RESULT_DIR=test/benchmark/r4/results/after \
    bash test/benchmark/r4/run_pure.sh --group numpy
NEW_PYTHONPATH="src:$SHADOW" NEW_RESULT_DIR=test/benchmark/r4/results/after \
    bash test/benchmark/r4/run_pure.sh --group iree
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest_after.txt --require-companion
#   -> 8 record(s): 8 complete / 369 checks, 0 failures
# --- STEP 2b: A/B equivalence (frozen vs fallback), out-of-band capture -------
NEW_RESULT_DIR=test/benchmark/r4/results/after/ab/frozen  NEW_PYTHONPATH="src:$SHADOW" \
  PURE_EXTRA_ARGS="--pure-runner frozen   --output-checksum --save-output-dir test/benchmark/r4/results/after/ab/frozen_save" \
  bash test/benchmark/r4/run_pure.sh --only pure64_dense_numpy,pure64_specialized_numpy
NEW_RESULT_DIR=test/benchmark/r4/results/after/ab/fallback NEW_PYTHONPATH="src:$SHADOW" \
  PURE_EXTRA_ARGS="--pure-runner fallback --output-checksum --save-output-dir test/benchmark/r4/results/after/ab/fallback_save" \
  bash test/benchmark/r4/run_pure.sh --only pure64_dense_numpy,pure64_specialized_numpy
# --- STEP 3: the 14-cell metric type 1 matrix --------------------------------
SHADOW_DIR="$SHADOW" bash test/benchmark/r4/run_e2e.sh          # --group shadow|stock|all, --only, --list, --dry-run
python3 test/benchmark/r4/check_e2e_r4.py
#   -> 14 record(s): 14 complete / 328 checks, 0 failures
# --- and the before gates stay green (frozen evidence) -----------------------
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
```

Runner overrides used in this window: `NEW_PYTHONPATH` / `NEW_RESULT_DIR` / `PURE_EXTRA_ARGS`
(`run_pure.sh`), `--agree-dir` (or `R4_AGREE_DIR`) in both agreement tools, `SHADOW_DIR` /
`SHADOW_PYTHONPATH` / `STOCK_PYTHONPATH` / `E2E_RESULT_DIR` (`run_e2e.sh`).

## G. Caveats

* **Shared box.**  Medians carry the usual ±10–20 % band; single cells (`pure64_dense_numpy`) can be
  worse and are flagged, not smoothed.  Only within-window comparisons are meaningful; all deltas above
  are same-process, same-window — except the explicitly labelled cross-window ones (R3, legacy, and the
  before-campaign).
* **The after window was shared with a second worker session.**  The pure matrix and the E2E matrix ran
  in parallel, interleaved through the one `flock`; GPU 6 was otherwise idle (0–4 % at the sampled
  points, a few MiB–313 MiB of peer context), and no record timed out or hit a guard.  Within a record
  the usual CPU-side noise band still applies — the numpy-dense companions below show it.
* **Sandbox teardowns cost one retry each (after-round).**  The ETL-import pre-flight's
  `fixture64_iree_spec` was killed mid-run by the tool sandbox on its first invocation and completed on
  a `--only` resume (its log therefore holds **two** blocks and `run_agreement_r4.last_run.txt` lists
  only the four resumed records); one pure-matrix batch attempt was killed the same way, leaving
  `results/after/pure64_dense_numpy.log` with a duplicated header block (35 lines vs 25).  Machine-written
  logs are left exactly as written; every record they describe is `complete` and exit 0.
* **The 14-cell E2E matrix ran in 3 chunks** (`--only` selections): the 4 headline cells (55 s), the 6
  numpy cells (2 m 33 s) and the 4 remaining iree cells (3 m 34 s).  `results/after/e2e/run_e2e_r4.last_run.txt`
  is committed but contains only the **last chunk's** 4-row table (it is regenerated per invocation).
* **The numpy-dense companion anomaly repeats (after-round)**: `pure64_dense_numpy` (companion 64.966 <
  pure 75.230, Δ = −10.26 ms) and `pure200_dense_numpy` (84.137 < 96.717, Δ = −12.58 ms) are the two
  cells whose `--e2e-companion` median lands below the pure median.  Same class as the before-round
  anomaly (`RUNLOG.md` before-round §"Anomalies" #1: a different code path, raw numpy CPU-bound, and the
  box's CPU load moves within a sub-second scale): reported as measured, no rerun, no cherry-picking.
* **IREE processes print a `nanobind: leaked … instances` dump at shutdown** (after the `done` line, in
  the specialized and dense cells alike).  All 14 E2E cells and all 8 pure iree records exited 0 and are
  `complete`; the dump is an interpreter-teardown artifact of the IREE runtime binding, recorded here so
  nobody reads it as a failed record.
* **`--stage-probe` *was* installed for the type 1 cells** (its tax is accepted there): the dense-routed
  `auto` / `dynamic` cells therefore report their cost in the uninstrumented `misc` bucket
  (`build_calls = 0`), which is a probe limitation, not a measurement of "no work".  The `specialized`
  cells and the numpy cells carry a usable `build_miss` / `run` / `readback` / `decode` split (§ the E2E
  section).  The "not applicable to type 2" caveat below applies to `--task eval-pure` only.
* **The ETL shadow is identified by path, not by version.**  Both checkouts report `etl_version=0.1.0`,
  and the `etl:` line's `rev=` field is the *editable repo's* HEAD (`51a7141`) — the shadow identity is
  the `file=/tmp/evogp_etl_patched_87c7562/etl/__init__.py` path (present in every after-log and in the
  E2E records' recorded `PYTHONPATH`).  Stronger evidence: `results/after/etl_import_probe.txt`, a
  fresh-process probe showing the PYTHONPATH precedence, the control/shadow `etl.__file__`, the sha256
  pins of the six changed files (`etl/__init__.py`, `build_cache.py`, `persist/cache.py`, `pipeline.py`,
  `trace/_tree.py`, plus the new `etl/jit.py`) and the `find_spec("etl.jit")` discriminator.  The source
  branch (`evogit-agent-T1-A160`) **no longer exists** and no ref contains `87c7562`, so the export and
  that probe are the surviving evidence — the `/tmp` shadow itself is session-scoped.
* **All after records are `frozen-api:*` / `compile.path = "frozen"`, and the frozen construction has no
  decode/build split** (`compile.build_ms` and `compile.decode_plan_ms` are `null`; only
  `construct_total_ms` is recorded).  Never reconstruct a split for them, and never present a
  `local-fallback:*` number as a frozen one (or vice versa).
* **All before numbers are fallback-path.**  `local-fallback:{dense|specialized}` / `compile.path =`
  `"fallback"` — never present them as frozen-runner numbers.  The fallback uses the engine's own memo
  key and the shared `etl_engine.CACHE`, so the construction and warm path are the engine's; only the
  call wrappers are the harness'.  The two constructors were shown numerically identical before the
  comparison was used (§H item 4).
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
* **The record JSONs and logs are immutable evidence** — `results/**`, `results/after/**` (incl. `e2e/`,
  `ab/`, `etl_import_probe.txt`) and `results/backend_agreement{,_after}/**` are read-only artifacts:
  no rerun, no edit, no re-derivation.  This file, `RUNLOG.md`, `README.md` and `CONTEXT.md` are derived
  from them, never the other way round.

## H. After-round execution record (the checklist, executed)

The six items below were the after-round's plan; each is now an executed step with its status and the
artifact that proves it.  The after campaign ran on 2026-09-13, HKT 07:34–07:52, GPU 6, in two parallel
worker sessions interleaved through the shared lock.

1. **Wait for the parallel code round to land** the frozen `dense_engine.build_dense_runner` /
   `etl_engine.build_specialized_runner` API, the `EVOGP_EVAL_MODE=dynamic` alias, the adjusted auto gates
   + `EVOGP_DENSE_MAX_TREE_LEN`, and the etl `jit` + hot-path work. — **DONE (upstream).**  All 8
   after-records carry `pure.runner_source = frozen-api:*`; the `dynamic` cells
   (`EVOGP_EVAL_MODE=dynamic`) resolve to the dense route (`dense_engine.EVAL_MODE_ALIASES`), and the
   shadow export's `etl/jit.py` is the etl-side evidence (`results/after/etl_import_probe.txt` probes
   B/F/G).
2. **Re-run the agreement pre-flight FIRST** (`run_agreement_r4.sh` + `check_agreement_r4.py`) — the ETL
   changed. — **DONE**: `results/backend_agreement_after/` (6 records, all exit 0), gate
   `check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after` → **PASS-with-note,
   30 checks / 0 failures** (§E).  The runner was driven through the additive `--agree-dir` override (the
   frozen default dir still points at the before records); one record resumed via `--only` after a sandbox
   teardown (§G).
3. **Re-run the matrix unchanged** (`run_pure.sh --group numpy` / `--group iree`) and then both checkers,
   with the gate staying at 8 complete / 0 failures. — **DONE**: `NEW_PYTHONPATH=src:<shadow>` +
   `NEW_RESULT_DIR=test/benchmark/r4/results/after`, 8/8 `complete`; `check_r4.py --manifest
   manifest_after.txt --require-companion` → **8 complete / 369 checks / 0 failures**; the before manifest
   stays green at the same 369/0.  `manifest_after.txt` is a mirror of `manifest.txt` (ids/mode/backend/
   order identical; only the paths point at `results/after/`).
4. **Verify the provenance flip** (`pure.runner_source = frozen-api:*`, `compile.path = "frozen"`), the
   two `FROZEN-ADAPTER` seams against the landed signature, an **A/B run** and a **numeric equivalence
   check**. — **DONE.**  All 8 after-records are `frozen-api:*` / `frozen`, with
   `n_graph_calls_source = "introspected"` and `n_graph_calls_per_run` 1 / 53 / 171 (the landed builders
   expose their own counter, so no record carries the `assumed-1` fallback and no `per_graph_call_ms` is
   an assumption).  The seams needed **no adjustment**: the landed signatures are
   `build_dense_runner(population_or_forest, inputs, *, …)` /
   `build_specialized_runner(population_or_forest, inputs, *, …)` with a no-arg `.run()`, exactly what
   `../pure_forward.py` assumed (`src/evogp/tree/RUNNER_HOOKS.md` §1).  The A/B ran
   `--pure-runner frozen` vs `--pure-runner fallback` on `pure64_dense_numpy` +
   `pure64_specialized_numpy` (`results/after/ab/{frozen,fallback}/`, both exit 0, the fallback arm still
   working at the new HEAD) with the engine's out-of-band capture (`--output-checksum
   --save-output-dir`, taken after all timing): equal `output_checksum` digests, **byte-identical** `.npz`
   files, `np.array_equal(…, equal_nan=True)` on all 53 arrays with `max|diff| = 0` and identical NaN
   placement — `results/after/ab/ab_equivalence_report.txt`, verdict ALL EQUAL.
5. **Build the after-table with the same columns as §C**, side by side with the before-table, note the
   `dynamic`-alias and gate changes, and fold the result into `../REPORT.md` (Round-4 verdict) plus
   `../CONTEXT.md` / `../README.md`. — **The tables are done** (§C.2, §C.3, §D.2, verdict above);
   `dynamic` is noted as the dense alias on the E2E side and the gates are recorded with fresh outputs.
   **Folded into the parent-level documents (`../REPORT.md` Round-4 section, `../CONTEXT.md`,
   `../README.md`)** — this node's four documents are current as of this round.
6. **Keep the before-records immutable** — no reruns, no edits, no re-derivation. — **HONOURED**: no file
   under `results/` was touched (both manifests gate green over the same bytes), and the after-records live
   in separate directories (`results/after/`, `results/backend_agreement_after/`) with the same
   no-rewrite rule (§G).
