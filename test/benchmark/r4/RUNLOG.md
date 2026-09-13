# R4 run log — the pure-forward before-baseline campaign as actually executed

Operator log for the R4 pre-flight + matrix (`r4/run_agreement_r4.sh`, `r4/run_pure.sh`; protocol:
`r4/README.md`, validator: `r4/check_r4.py`, config contract: `r4/manifest.txt`, harness: `../bench_gp.py` +
`../pure_forward.py`). One section per phase, in execution order. Raw evidence stays in the harness JSONs +
the appended per-config logs (`results/<id>.{json,log}`, `results/backend_agreement/<stem>.{json,log}` +
`<stem>_values.npy`) — this file is the summary and the interpretation, not a second source of numbers.
Values marked *derived* are computed here from the committed JSONs (everything else is a harness field).
Section order follows execution: the before-baseline campaign of 03:46–03:56 (this part of the file),
its post-campaign validator addendum, and then the **after-round** of 07:34–07:52 as the last section
(the frozen `build_{dense,specialized}_runner` API had landed by then).

Scope of the operator's writes: `test/benchmark/r4/` only (the two new agreement files, this log, an appended
README subsection, the campaign records under `results/`). No engine, `r2/**`, `SPEC.md`, `REPORT.md`,
`benchmark/CONTEXT.md` or pre-existing r4 script/record was touched.

---

## Window, box and engine identity

| item | value |
|---|---|
| window (HKT, 2026-09-13) | pre-flight 03:46:00–03:48:26; matrix numpy group 03:49:01–03:52:25; matrix iree group 03:52:29–03:55:55 (wall 2 m 28 s + 3 m 24 s + 3 m 26 s) |
| repo root / cwd | the worker worktree (`git rev-parse --show-toplevel` = the campaign worktree), every command run from there |
| `git rev-parse HEAD` | `e2727140b70d1fc814c03b2c75b86613185fb2e9` (the agreement-tooling commit; every campaign JSON records `evogp_commit=e2727140b70d`) |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124); no `python` on PATH |
| GPU | `CUDA_VISIBLE_DEVICES=6` (the runner default) — `6, NVIDIA RTX A6000, 0 %, 30487 MiB/49140 MiB` before the campaign ⇒ ≈18.6 GiB free. **GPU 7 never touched** (hardware ECC fault). Per-config `gpu (before)/(after)` lines in every log: resident foreign tenant holding ≈30.5 GiB throughout (30 477–30 785 MiB, flat), our own utilisation samples 0–73 % |
| contention | a foreign tenant is *resident* on GPU 6 (~30.5 GiB, flat across the whole window) but its compute utilisation stayed in the 0–73 % band at our samples; no burst-induced failure and **no config came near its guard** (see Anomalies for the one CPU-side noise artifact) |
| lock | `/tmp/gpbench_critical.lock`, one `flock -w 7200` per config/record (never nested), `timeout -s INT -k 30 1800` **inside** the lock; no lock-wait messages, zero timeouts, `ulimit -n 65536` in both runners |
| engine under test | `PYTHONPATH=src` (this worktree); backend `numpy` (default) and `iree` (opt-in, O1, device 0); `ETL_DISABLE_LOCATIONS=1`; `--tag` = UPPER(id) |
| `etl` identity (per-config `etl:` line in every log) | `repo=/mnt/local-ssd/bchuang/etl rev=51a7141885d76a6764a9ad43f6bf1264dd12e6fe file=/mnt/local-ssd/bchuang/etl/etl/__init__.py`, `etl_version=0.1.0`, checkout working tree clean. That revision is the **patched** ETL checkout R3 Arm B shadowed (stock `main` is `a863545`), so the round measured the patched library — the fd evidence agrees: every iree config ends at **91 fds** (39 → 91) with 1/48/1/155 builds, i.e. flat like the patched build, not the ~32–35 fds per compiled structure of stock |
| fixtures | `test/goldens/forest_random.npz` (64 trees, 64 data rows) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, 64 rows) |

Exact commands (one bash call each, from the repository root):

```bash
export ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6
bash test/benchmark/r4/run_agreement_r4.sh                                   # STEP 1 pre-flight  (2 m 28 s)
python3 test/benchmark/r4/check_agreement_r4.py  # -> gate PASS (numpy venv)   # STEP 1 gate
bash test/benchmark/r4/run_pure.sh --group numpy                             # STEP 2 (3 m 24 s)
bash test/benchmark/r4/run_pure.sh --group iree                              # STEP 2 (3 m 26 s)
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
```

Each config/record is one fresh `bench_gp.py`/`check_backend_agreement.py` process; the exact per-config
command line (env block + flags + `--json-out`) is reproduced in the `cmd:` header of that config's log.
The runner passes `--backend <numpy|iree>` and `EVOGP_EVAL_MODE=<mode>` per config, so the pure route
(`--mode`) and the `--e2e-companion` route agree.

---

## STEP 1 — IREE backend-agreement pre-flight (new for R4)

Six records: two fixtures × three roles (`numpy`, `iree` under the default `auto` policy → dense route,
`iree` + `EVOGP_EVAL_MODE=specialized`). All six exited 0 in 2 m 28 s (the slow one is
`fixture200_iree_spec`, 71 s), all 30 gate checks pass.

| record | exit | started → finished | values_sha256 | per_call_ms (cold, warm) |
|---|---|---|---|---|
| `fixture64_numpy` | 0 | 03:46:00 → 03:46:01 | `2e5c0fa1fdbec0743facc1beb7c72e7d9ed4c156c3b5ad5a8971f1d5f67e414d` | 186.2, 21.0 |
| `fixture64_iree` | 0 | 03:46:03 → 03:46:19 | `b8192cb1770b886d6b5decbbcb27421aadca9aa093f59cc21ff6b4b16e0afa02` | 14,964.5, 7.6 |
| `fixture64_iree_spec` | 0 | 03:46:21 → 03:46:50 | `2e5c0fa1fdbec0743facc1beb7c72e7d9ed4c156c3b5ad5a8971f1d5f67e414d` | 27,711.3, 71.2 |
| `fixture200_numpy` | 0 | 03:46:52 → 03:46:54 | `16d5bbe98481ff954733b71cdd24471235f6d840c0a50d1a8f56d16cf8a11d81` | 1,074.4, 97.2 |
| `fixture200_iree` | 0 | 03:46:57 → 03:47:12 | `07948efededc27b2f30691a4eb305f5abad25f4f3fbedb39b47dd31b392be048` | 14,915.1, 7.6 |
| `fixture200_iree_spec` | 0 | 03:47:15 → 03:48:26 | `16d5bbe98481ff954733b71cdd24471235f6d840c0a50d1a8f56d16cf8a11d81` | 70,798.1, 303.3 |

Gate (`check_agreement_r4.py`, 30 checks / 0 failures, exit 0) — **verdict: PASS-with-note on both fixtures**:

* `fixture64` and `fixture200`: `numpy` == `iree_spec` **bit-identically**; the `iree` (auto → dense) hash
  differs, but every role pair is inside `np.allclose(rtol=1e-4, atol=1e-4, equal_nan=True)` —
  `max|diff|` = 7.629e-06 (fixture64) and 3.906e-03 (fixture200), max **relative** diff 1.1e-07 / 2.0e-07.
  That is the known dense-vs-per-structure summation-order difference, not a miscompile: the dense one-graph
  evaluator sums a whole population in one graph while the specialized route runs one graph per tree.
* NaN/±inf placement is identical across the three roles (fixture64: 3 NaN / 64; fixture200: 12 NaN / 200);
  the npz and npy identities re-hashed clean (`fixture_sha256` == sha256 of the fixture; every
  `<stem>_values.npy` hashes back to its JSON's `values_sha256`).

**Continuity (strong):** all six R4 records are **bit-identical (full sha256) to the R3 pre-flight records**
(`r2/round3/results/backend_agreement/`), and the two `iree` (dense) hashes also match the R2 anchors. So
nothing in the numpy/specialized/dense value paths moved between R3 and R4 — the R4 IREE timings are quoted
on a numerically unchanged engine. (The numpy/`iree_spec` vs *R2* anchor shift is the one the R3 checker
already reports as INFO; R4 does not change it.)

| fixture | role | R4 hash == R3 | == R2 anchor |
|---|---|---|---|
| fixture64 | numpy | yes | n/a |
| fixture64 | iree | yes | yes |
| fixture64 | iree_spec | yes | n/a |
| fixture200 | numpy | yes | n/a |
| fixture200 | iree | yes | yes |
| fixture200 | iree_spec | yes | n/a |

Records: `results/backend_agreement/<stem>.{json,log}` + `<stem>_values.npy` (six of each), id → exit table in
`results/backend_agreement/run_agreement_r4.last_run.txt` (disposable, uncommitted).

---

## STEP 2 — the pure-forward matrix (8/8 canonical configs)

Both groups ran in the canonical order (numpy first, then iree), one process per config, no config was
retried and none was guard-skipped: **8/8 exit 0, `complete=true`, `partial=false`, `skipped=false`**.

| id | group | exit | status |
|---|---|---|---|
| `pure64_dense_numpy` | numpy | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure64_specialized_numpy` | numpy | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure200_dense_numpy` | numpy | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure200_specialized_numpy` | numpy | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure64_dense_iree` | iree | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure64_specialized_iree` | iree | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure200_dense_iree` | iree | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |
| `pure200_specialized_iree` | iree | 0 | OK (log header `exit code: 0 (OK)`, JSON `complete=true`) |

Note: `r4/last_run.txt` is generated per *invocation*, so it holds only the second (iree) group's table; the
numpy group's exit codes above come from the per-config log headers + the JSON `complete`/`partial` flags
(the run's stdout summary printed `ok: 4 skipped: 0 failed: 0` for both groups).

### Headline numbers (harness fields, warm/steady state)

| id | pure median per invocation | pure per graph call | graph calls/run (source) | compile construct total | companion median_warm | E2E − pure *derived* | runner source |
|---|---|---|---|---|---|---|---|
| `pure64_dense_numpy` | 128.936 ms | 128.9358 ms | 1 (measured) | 2,381.5 ms | 102.083 ms | -26.85 ms | `local-fallback:dense` |
| `pure64_specialized_numpy` | 8.390 ms | 0.1583 ms | 53 (measured) | 154.7 ms | 20.093 ms | +11.70 ms | `local-fallback:specialized` |
| `pure200_dense_numpy` | 181.686 ms | 181.6856 ms | 1 (measured) | 2,173.2 ms | 183.618 ms | +1.93 ms | `local-fallback:dense` |
| `pure200_specialized_numpy` | 49.332 ms | 0.2885 ms | 171 (measured) | 990.9 ms | 97.133 ms | +47.80 ms | `local-fallback:specialized` |
| `pure64_dense_iree` | 1.587 ms | 1.5867 ms | 1 (measured) | 13,522.7 ms | 4.841 ms | +3.25 ms | `local-fallback:dense` |
| `pure64_specialized_iree` | 53.214 ms | 1.0040 ms | 53 (measured) | 18,817.6 ms | 59.706 ms | +6.49 ms | `local-fallback:specialized` |
| `pure200_dense_iree` | 3.696 ms | 3.6963 ms | 1 (measured) | 14,345.2 ms | 5.100 ms | +1.40 ms | `local-fallback:dense` |
| `pure200_specialized_iree` | 214.655 ms | 1.2553 ms | 171 (measured) | 71,335.3 ms | 221.875 ms | +7.22 ms | `local-fallback:specialized` |

Per-rep spread (the 5 timed blocks of each config, ms/invocation, *derived*) and the cold-construction split:

| id | reps (ms/invocation) | min / median / mean | decode_plan_ms | build_ms | cache hits/misses | const trees |
|---|---|---|---|---|---|---|
| `pure64_dense_numpy` | 148.84, 128.94, 129.71, 124.57, 120.26 | 120.26 / 128.94 / 130.46 | 1.8 | 2,379.1 | 0/1 | 0/64 |
| `pure64_specialized_numpy` | 8.39, 8.50, 8.53, 8.23, 8.27 | 8.23 / 8.39 / 8.39 | 1.6 | 153.1 | 5/48 | 11/64 |
| `pure200_dense_numpy` | 178.13, 185.07, 180.60, 181.69, 182.20 | 178.13 / 181.69 / 181.54 | 1.6 | 2,170.9 | 0/1 | 0/200 |
| `pure200_specialized_numpy` | 50.22, 51.20, 47.21, 49.05, 49.33 | 47.21 / 49.33 / 49.40 | 5.3 | 985.6 | 16/155 | 29/200 |
| `pure64_dense_iree` | 1.96, 1.58, 1.58, 1.59, 1.59 | 1.58 / 1.59 / 1.66 | 1.2 | 13,514.9 | 0/1 | 0/64 |
| `pure64_specialized_iree` | 53.82, 54.01, 53.21, 53.19, 52.98 | 52.98 / 53.21 / 53.44 | 66.5 | 18,751.1 | 5/48 | 11/64 |
| `pure200_dense_iree` | 4.24, 3.11, 2.10, 3.70, 4.12 | 2.10 / 3.70 / 3.45 | 1.7 | 14,254.2 | 0/1 | 0/200 |
| `pure200_specialized_iree` | 214.65, 240.77, 218.35, 187.78, 190.58 | 187.78 / 214.65 / 210.43 | 60.5 | 71,274.7 | 16/155 | 29/200 |

### What the matrix says (derived ratios, medians)

* **iree dense is the fast pure route**: 128.9 → 1.587 ms (81×) at 64 trees and
  181.7 → 3.696 ms (49×) at 200 trees, both against numpy.
* **One graph call is not one cheap call on IREE**: dense = 1 `etl.run` per invocation (1.587 / 3.696 ms),
  while the specialized route pays 53 / 171 calls
  per invocation at 1.004 / 1.255 ms each — so on IREE the
  specialized route is 34× (64) / 58× (200) *slower* per invocation than dense, and
  6.3× / 4.4× slower than the numpy specialized route: per-tree dispatch dominates.
* **On numpy the ordering is the opposite**: specialized beats dense (8.39 vs 128.9 ms at 64 trees,
  49.3 vs 181.7 ms at 200 trees), because one numpy graph over the whole population costs more than
  53/171 small per-tree interpreter calls.
* **Cold construction is a different budget** and is reported separately in every record: one dense graph costs
  13,523–14,345 ms (iree) vs ≈2,382 ms (numpy); the 48 specialized builds cost 18,818 ms (iree) / 155 ms
  (numpy) and the 155 builds 71,335 ms (iree) / 991 ms (numpy) — i.e. iree pays ~10–72 s of compile for a
  warm pure number of 1.6–215 ms per invocation.
* `E2E − pure` (host glue + readback, one process/one window) is small where the graph dominates:
  +1.9 ms (200 dense numpy), +1.4 ms (200 dense iree),
  +7.2 ms (200 specialized iree), but ≈1× the pure number on the numpy specialized route
  (+11.7 ms of 8.4 ms, +47.8 ms of 49.3 ms) — there the per-tree readback/assembly is as
  expensive as the forwards themselves. One config is the exception, see Anomalies.

### Sanity triage before committing (the three rules)

1. **pure < companion** — holds for 7/8 configs. Violation: `pure64_dense_numpy`
   (pure 128.936 ms > companion 102.083 ms, E2E − pure = −26.85 ms). See Anomalies.
2. **graph calls** — dense = exactly 1 (`measured`) in all four dense records; specialized = the non-constant
   tree count (`measured`): 53 = 64 − 11 at fixture64, 171 = 200 − 29 at fixture200. Consistent in all 8.
3. **construction stays out of the pure numbers** — `compile.*` is a separate block, the timed blocks are
   `pure.block_ms` only, and `per_rep[0].wall_s` (= construction + block) is larger than `block_ms` in every
   record (e.g. `pure64_dense_iree`: wall_s 13.77 s = construct 13.53 s + block 0.196 s). `compile.path =
   fallback` and `pure.runner_source = local-fallback:*` in all 8 (the frozen `build_*_runner` API does not
   exist at this HEAD — expected for the *before* baseline).

Validator (`check_r4.py --manifest manifest.txt --require-companion`): **8 record(s): 8 complete, 0 partial,
0 skipped, 0 missing, 0 malformed; 361 checks, 4 failures**. All four failures are the *same* check — the
validator's ordering assumption `min <= median <= mean`, which is not an invariant for 5 samples:

| id | min / median / mean (ms) | why |
|---|---|---|
| `pure64_specialized_numpy` | 8.232 / 8.390 / 8.385 | one fast rep pulls the mean below the median (8.39, 8.50, 8.53, 8.23, 8.27) |
| `pure200_dense_numpy` | 178.133 / 181.686 / 181.540 | one fast rep pulls the mean below the median (178.13, 185.07, 180.60, 181.69, 182.20) |
| `pure200_dense_iree` | 2.098 / 3.696 / 3.452 | one fast rep pulls the mean below the median (4.24, 3.11, 2.10, 3.70, 4.12) |
| `pure200_specialized_iree` | 187.776 / 214.655 / 210.427 | one fast rep pulls the mean below the median (214.65, 240.77, 218.35, 187.78, 190.58) |

The recorded statistics equal the statistics recomputed from `per_invocation_ms` in all 8 records (those
checks pass), so this is a validator over-assertion, not a data inconsistency: with an odd number of reps a
left-skewed block legitimately has `median > mean`. The four flagged blocks are the noisiest ones on a
shared box (see the rep spreads above) — the pure numbers themselves are usable, and the deviation is a few
tenths of a percent (numpy) up to ~7 % (the two iree blocks). `check_r4.py` was **not** modified (out of the
operator's write scope); flagged here and in the completion report for the round's owner.

---

## Anomalies, deviations and caveats

1. **`pure64_dense_numpy`: pure 128.936 ms > companion 102.083 ms** — the single sanity-rule violation.
   Reading: a *CPU-contention artifact of the first config of the window*, not a structural finding.
   Evidence: (a) the pure number is 128.9 ms vs the smoke record's 125.2 ms for the same config
   (`results/smoke/pure64_dense_numpy.json`, +2.9 %) — the pure side is on its ballpark; (b) the pure block
   drifts monotonically downwards (148.84 → 128.94 → 129.71 → 124.57 → 120.26 ms) and the companion's 11
   calls are [146.3, 150.4, 102.2, 214.2, 101.9, 101.8, 102.2, 102.0, 101.7, 214.1, 101.5] — i.e. the box's
   CPU load moves by up to 2× on a sub-second scale inside a single config (the numpy backend is a CPU
   interpreter, so it sees exactly that); (c) the same pair of paths on the *same* route 4 minutes later
   (`pure200_dense_numpy`) agrees within 1.1 % (181.686 pure vs 183.618 companion), and the smoke window saw
   the companion at 113.9–231.8 ms. No rerun was made: the 128.9 ms pure number is 1.03× the smoke ballpark,
   far from the ≳3× "egregious" threshold, and a rerun would have replaced the canonical record with a
   second, equally window-dependent sample. **Quote `pure64_dense_numpy` pure = 128.9 ms (comparable to
   120–130 ms), and treat its ±27 % companion delta as noise.**
2. **The 4 validator failures** are the `min <= median <= mean` over-assertion described above (validator
   artifact, engine-independent, 0 data failures).
3. **IREE compile times are not hardware constants** (R3 caveat, reproduced here): the dense one-graph build
   cost 13,515 ms (64 trees) / 14,254 ms (200 trees) — nearly the same, as expected for a whole-population
   graph — while the specialized builds averaged 391 ms/build (48 builds) and 460 ms/build (155 builds). R3
   measured 342–2 162 ms per build for *identical* build counts on this box, so read `construct_total_ms` as
   "± a factor of a few", never as a hardware constant; only the warm numbers are stable.
4. **ETL identity**: the editable `etl` install points at the *patched* checkout revision `51a7141` (see the
   identity table) and its flat 91-fd iree behaviour is visible in all four iree records; this is recorded,
   not corrected — the objective pinned no ETL revision, and the identity line in every log is the evidence.
5. **`--stage-probe` was not used** (it is refused by design for `--task eval-pure`; `stages` is `null` in
   every record), so the cold/warm split comes from `compile`/`pure` only. `--mem-guard-mb 8192` was accepted
   in every config, and the iree cold-build guard (limit 400) was never approached: `misses_max` = 1 / 48 /
   1 / 155.
6. Nothing else unexpected: no timeouts, no crashes, no guard-skips, no missing ids, no `partial` records,
   all `nofile` rlimits at 65536, all backends/modes matching the manifest (validator checks the mode/
   backend columns).

## Artifacts of the round (committed)

| path | what |
|---|---|
| `r4/run_agreement_r4.sh` | the pre-flight runner (6 records; new this round) |
| `r4/check_agreement_r4.py` | the pre-flight gate (new this round) |
| `r4/results/backend_agreement/<stem>.{json,log,_values.npy}` | the 6 pre-flight records + logs + raw vectors |
| `r4/results/<id>.{json,log}` | the 8 campaign records + per-config logs (each log carries the `etl:` identity line) |
| `r4/RUNLOG.md` | this file |
| `r4/README.md` | + the appended "IREE agreement pre-flight" subsection |
| not committed | `r4/last_run.txt`, `results/backend_agreement/run_agreement_r4.last_run.txt` (generated/disposable) |

**After-round (executed — see the section at the end of this file).**  The frozen
`build_{dense,specialized}_runner` API landed; `run_pure.sh` was re-run unchanged into `results/after/`
(on a read-only shadow export of etl `87c7562`) and the `local-fallback:*` records above were A/B-ed
against the `frozen-api:*` ones (bit-identical outputs, `results/after/ab/ab_equivalence_report.txt`).

---

## Post-campaign addendum

**`check_r4.py` invariant corrected — commit `04fd739`** (the four failures reported in "Anomalies" #2 and in
the sanity triage were a *validator over-assertion, not a data problem*: `median > mean` is legitimate for an
odd, left-skewed sample of 5 blocks, and the recorded `min` / `median` / `mean` equal the statistics recomputed
from `per_invocation_ms` in all 8 records).  The `min <= median <= mean` check was replaced by the always-valid
bounds `min <= median <= max` and `min <= mean <= max` (`max` recomputed from the per-rep list; the sample-free
`min <= median` / `min <= mean` when that list is unusable), with the recorded-summaries == recomputed-statistics
equality checks kept as they were.  **No record, log or npy was touched.**

Final gates after the fix:

```bash
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
#   -> 8 record(s): 8 complete, 0 partial, 0 skipped, 0 missing, 0 malformed; 369 check(s), 0 failure(s)   (exit 0)
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/results/smoke/manifest_smoke.txt --require-companion
#   -> 3 record(s): 3 complete, ...; 139 check(s), 0 failure(s)   (exit 0; the three custom-`--tag` notes stay, as expected)
```

(369 = 361 + one extra bounds check per record.)  The round's write-up is `SUMMARY.md` — as is the
after-round section that follows.

---

## After-round (2026-09-13, HKT 07:34–07:52)

The after-round re-ran the same 8-config protocol at the code round's HEAD — which had landed the frozen
`build_{dense,specialized}_runner` API, the `EVOGP_EVAL_MODE=dynamic` → dense alias, the adjusted auto
gates + `EVOGP_DENSE_MAX_TREE_LEN`, and the etl `jit` + hot-path work — and added a 14-cell metric type 1
(`problem.evaluate`) matrix on the same box in the same window. Like everything above it is executed
evidence, not a plan; the analysis of record is `SUMMARY.md` (§C.2, the "Type 1 (E2E) after-round"
section, §E, §H). Two parallel worker sessions drove it — one the pre-flight + the pure matrix, one the
type 1 matrix — interleaved through the same single shared `flock`.

### Window, box and engine identity (after-round)

| item | value |
|---|---|
| window (HKT, 2026-09-13) | pre-flight 07:34:54–07:42:16 (7 m 22 s); pure matrix 07:44:37–07:50:52 (numpy group 07:45:41–07:48:00 after one sandbox teardown, iree group 07:48:02–07:50:52); type 1 matrix 07:44:34–07:51:42 in 3 chunks (07:44:34–07:45:28, 07:45:34–07:48:04, 07:48:32–07:51:42) |
| `git rev-parse HEAD` | `6e8bf6f0ff7ee2d82e890bca464c3c833f5fae40` (every after JSON records `evogp_commit=6e8bf6f0ff7e`) |
| engine under test (new side) | `PYTHONPATH=src:/tmp/evogp_etl_patched_87c7562`; backend `numpy` (default) and `iree` (opt-in, O1, device 0); `ETL_DISABLE_LOCATIONS=1`; `--tag` = upper-case id (`E2E_<ID>` on the type 1 cells) |
| `etl` identity | a **read-only shadow export** of upstream commit `87c7562713b75b235e4945c6311de054652fc919`, materialized with `git -C /mnt/local-ssd/bchuang/etl archive 87c7562 \| tar -x -C /tmp/evogp_etl_patched_87c7562` (never written inside the foreign repo, no worktree added, no ref created). The editable install is untouched, so the per-config `etl:` line reads `repo=/mnt/local-ssd/bchuang/etl rev=51a7141885d7… file=/tmp/evogp_etl_patched_87c7562/etl/__init__.py pythonpath=src:/tmp/evogp_etl_patched_87c7562` — **`rev=` is the editable repo's HEAD, the `file=` path is the shadow identity.** Both checkouts report `etl_version=0.1.0`, so that JSON field cannot distinguish them. |
| shadow evidence chain | `results/after/etl_import_probe.txt` (commit `b55b0bc`): fresh-process probes resolve `PYTHONPATH=src` → `/mnt/local-ssd/bchuang/etl/etl/__init__.py` and `PYTHONPATH=src:/tmp/evogp_etl_patched_87c7562` → the shadow (PYTHONPATH beats the PEP-660 editable finder), sha256 pins for the six changed files (`etl/__init__.py`, `build_cache.py`, `persist/cache.py`, `pipeline.py`, `trace/_tree.py` + the new `etl/jit.py`), and `find_spec("etl.jit")` as the discriminator. The source branch `evogit-agent-T1-A160` **no longer exists**, no ref contains `87c7562` and `main` (`51a7141`) is not a descendant of it, so the export + that probe are the surviving evidence (the `/tmp` shadow is session-scoped). |
| interpreter | `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124); no `python` on PATH |
| GPU | `CUDA_VISIBLE_DEVICES=6` (NVIDIA RTX A6000): **idle at the start** (`0 %, 1 MiB`), while foreign tenants held GPUs 0–2 at 90–100 %; later samples 2–4 % and 1–313 MiB (the parallel worker session's context). **GPU 7 never touched.** |
| lock | exactly one `flock -w 7200 /tmp/gpbench_critical.lock` per record, `timeout` inside it, never nested; no lock-wait message and no timeout on any of the 28 after-records; `ulimit -n 65536` first in every runner |
| fds | numpy records 39 → **59** flat, every iree record 39 → **91** flat (the patched-ETL profile, not stock's ~32–35 fds per compiled structure) |
| fixtures | unchanged: `test/goldens/forest_random.npz` (64 trees, 64 rows) and `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees, 64 rows) |

Exact commands (one bash call each, from the repository root):

```bash
export ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python
SHADOW=/tmp/evogp_etl_patched_87c7562                       # git -C /mnt/local-ssd/bchuang/etl archive 87c7562 | tar -x -C …
# STEP 1 — agreement pre-flight into the after dir (6 records, 7 m 22 s incl. one resume)
NEW_PYTHONPATH="src:$SHADOW" bash test/benchmark/r4/run_agreement_r4.sh \
    --agree-dir test/benchmark/r4/results/backend_agreement_after
$PY test/benchmark/r4/check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after
# STEP 2 — the same 8-config pure matrix, after-artifacts dir
NEW_PYTHONPATH="src:$SHADOW" NEW_RESULT_DIR=test/benchmark/r4/results/after \
    bash test/benchmark/r4/run_pure.sh --group numpy
NEW_PYTHONPATH="src:$SHADOW" NEW_RESULT_DIR=test/benchmark/r4/results/after \
    bash test/benchmark/r4/run_pure.sh --group iree
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest_after.txt --require-companion
# STEP 3 — the 14-cell metric type 1 matrix (driven in 3 chunks; run_e2e_r4.last_run.txt keeps only the last one)
SHADOW_DIR=$SHADOW bash test/benchmark/r4/run_e2e.sh
python3 test/benchmark/r4/check_e2e_r4.py
```

### STEP 1 — agreement pre-flight into the after dir (6 records, all exit 0)

Same six roles as the before pre-flight (2 fixtures × `numpy` / `iree` under the default `auto` policy →
dense / `iree` + `EVOGP_EVAL_MODE=specialized`), re-run against the shadow export. Gate
`check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after` →
**PASS-with-note on both fixtures, 30 checks / 0 failures, exit 0**.

| record | exit | started → finished | `values_sha256` | per_call_ms (cold, warm) |
|---|---|---|---|---|
| `fixture64_numpy` | 0 | 07:34:54 → 07:34:57 | `2e5c0fa1fdbe…` | 217.8, 13.3 |
| `fixture64_iree` | 0 | 07:34:57 → 07:35:11 | `fbe52b281581…` | 11 447.6, 9.9 |
| `fixture64_iree_spec` | 0 | 07:40:53 → 07:41:09 | `2e5c0fa1fdbe…` | 13 639.9, 80.2 |
| `fixture200_numpy` | 0 | 07:41:09 → 07:41:14 | `16d5bbe98481…` | 1 168.8, 64.4 |
| `fixture200_iree` | 0 | 07:41:14 → 07:41:28 | `74a46ed4bf75…` | 11 657.3, 7.8 |
| `fixture200_iree_spec` | 0 | 07:41:28 → 07:42:16 | `16d5bbe98481…` | 44 850.8, 237.3 |

* `numpy` == `iree_spec` **bit-identically** per fixture (4 of 6 records unchanged from the before set);
  the two `iree` (dense) vectors **moved** before → after (`b8192cb1770b…` → `fbe52b281581…` at
  fixture64, `07948efededc…` → `74a46ed4bf75…` at fixture200) but stay inside the fallback tolerance
  (`max|diff|` 7.629e-06 / 0.03125, max relative diff 1.06e-07 / 1.81e-07; before: 7.629e-06 / 3.906e-03
  and 1.08e-07 / 2.03e-07) — the documented dense-vs-per-structure summation-order difference, so the
  after IREE timings are quotable **with that deviation recorded**.
* NaN/±inf placement unchanged (3 NaN / 64, 12 NaN / 200, 0 inf); every identity re-hash clean
  (`fixture_sha256` = `0bfb0bbf48cd…` / `6789ef61ecf8…`; each `<stem>_values.npy` == its JSON's
  `values_sha256`).
* **Sandbox teardown (1/6):** the first invocation was killed mid `fixture64_iree_spec` (its block starts
  07:35:11 and ends without a result line). The record was completed by an `--only
  fixture64_iree_spec,fixture200_numpy,fixture200_iree,fixture200_iree_spec` resume, so
  `results/backend_agreement_after/fixture64_iree_spec.log` holds **two** header blocks (105 lines vs the
  usual 92) and the committed `run_agreement_r4.last_run.txt` lists only the four resumed records.

### STEP 2 — the pure-forward matrix, after-records (8/8 complete, `frozen-api:*`)

Both groups ran in the canonical order (numpy first, then iree), one process per config, none skipped:
**8/8 exit 0, `complete=true`**. Every record carries `pure.runner_source = frozen-api:build_{dense,
specialized}_runner`, `compile.path = "frozen"` and `n_graph_calls_source = "introspected"` — the landed
builders expose their own counter, so no record has the `assumed-1` fallback.

| id | started → finished | `median_per_invocation_ms` | per graph call | calls/run (introspected) | `construct_total_ms` | companion `median_warm_ms` | fds |
|---|---|---|---|---|---|---|---|
| `pure64_dense_numpy` | 07:45:41 → 07:46:28 | 75.230 | 75.2302 | 1 | 1 682.1 | 64.966 | 39→59→59 |
| `pure64_specialized_numpy` | 07:46:28 → 07:46:37 | 6.616 | 0.1248 | 53 | 154.9 | 11.206 | 39→59→59 |
| `pure200_dense_numpy` | 07:46:37 → 07:47:40 | 96.717 | 96.7171 | 1 | 1 760.2 | 84.137 | 39→59→59 |
| `pure200_specialized_numpy` | 07:47:40 → 07:48:00 | 30.360 | 0.1775 | 171 | 1 020.9 | 50.670 | 39→59→59 |
| `pure64_dense_iree` | 07:48:02 → 07:48:31 | 1.447 | 1.4468 | 1 | 11 611.1 | 8.477 | 39→91→91 |
| `pure64_specialized_iree` | 07:48:31 → 07:49:02 | 5.922 | 0.1117 | 53 | 12 851.3 | 60.410 | 39→91→91 |
| `pure200_dense_iree` | 07:49:02 → 07:49:45 | 1.593 | 1.5926 | 1 | 11 663.4 | 6.462 | 39→91→91 |
| `pure200_specialized_iree` | 07:49:45 → 07:50:52 | 18.570 | 0.1086 | 171 | 42 603.6 | 195.368 | 39→91→91 |

Per-rep spreads (the 5 timed blocks, ms/invocation, *derived*): 77.63/76.56/74.98/75.23/74.50,
6.67/6.78/6.58/6.62/6.59, 103.86/97.42/96.72/96.12/95.19, 30.84/30.58/29.78/30.36/29.97,
1.72/1.46/1.44/1.45/1.38, 7.21/5.96/5.57/5.75/5.92, 1.78/1.56/1.59/1.59/1.60, 18.57/18.54/17.73/19.00/22.11.

* **Before → after (the headline)**: 128.936 → 75.230 (**−41.6 %**), 8.390 → 6.616 (**−21.1 %**),
  181.686 → 96.717 (**−46.8 %**), 49.332 → 30.360 (**−38.5 %**), 1.587 → 1.447 (**−8.8 %**),
  53.214 → 5.922 (**−88.9 %**), 3.696 → 1.593 (**−56.9 %**), 214.655 → 18.570 (**−91.3 %**).  The
  specialized-iree route's per-`etl.run` cost fell 1.0040 → 0.1117 ms (64) and 1.2553 → 0.1086 ms (200),
  i.e. the per-tree dispatch that dominated the before round is now the cheap part of that route.
* **The frozen construction is opaque by design**: `compile.build_ms` and `compile.decode_plan_ms` are
  `null` in all 8 records (the builders *are* the construction), so the before-table's decode/build split
  has no after counterpart — only `construct_total_ms` (1 682.1 / 154.9 / 1 760.2 / 1 020.9 / 11 611.1 /
  12 851.3 / 11 663.4 / 42 603.6 ms). Cache counters: `cache_misses` 1 / 48 / 1 / 155 (same work as
  before) and `cache_hits` 0 everywhere (the fallback records counted 5 / 16 hits on the specialized
  numpy routes — a raw counter difference between the two constructors, recorded not interpreted).
* **`pure < companion` holds for 6/8** — the two exceptions are the numpy-dense cells
  (`pure64_dense_numpy` 75.230 pure vs 64.966 companion, `pure200_dense_numpy` 96.717 vs 84.137), the same
  class as the before-round anomaly #1: raw numpy is CPU-bound and the box's CPU load moves within a
  sub-second scale. Reported as measured — no rerun, no cherry-picking.
* **Sandbox teardown (1 config):** the first numpy-group attempt died with the sandbox at 07:44:37 right
  after `pure64_dense_numpy` started, so `results/after/pure64_dense_numpy.log` has a duplicated header
  block (35 lines vs the usual 25); the config was re-run inside the group resume (07:45:41 start). All
  logs are left exactly as the machine wrote them.
* Validator: `python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest_after.txt
  --require-companion` → **8 record(s): 8 complete, 0 partial, 0 skipped, 0 missing, 0 malformed;
  369 check(s), 0 failure(s)** (exit 0). `manifest_after.txt` mirrors `manifest.txt`: same 8 ids in the
  same order with the same mode/backend columns, only the paths point at `results/after/`.

### STEP 2b — A/B equivalence: frozen constructor vs the local fallback

Two configs (`pure64_dense_numpy`, `pure64_specialized_numpy`), both arms run with the engine's
out-of-band capture (taken after all timing, never inside a timed block):

```bash
NEW_RESULT_DIR=test/benchmark/r4/results/after/ab/frozen  NEW_PYTHONPATH="src:$SHADOW" \
  PURE_EXTRA_ARGS="--pure-runner frozen   --output-checksum --save-output-dir test/benchmark/r4/results/after/ab/frozen_save" \
  bash test/benchmark/r4/run_pure.sh --only pure64_dense_numpy,pure64_specialized_numpy
NEW_RESULT_DIR=test/benchmark/r4/results/after/ab/fallback NEW_PYTHONPATH="src:$SHADOW" \
  PURE_EXTRA_ARGS="--pure-runner fallback --output-checksum --save-output-dir test/benchmark/r4/results/after/ab/fallback_save" \
  bash test/benchmark/r4/run_pure.sh --only pure64_dense_numpy,pure64_specialized_numpy
```

Both arms exited 0 (2/2 configs each; the fallback arm still works at the new HEAD). Verdict
`results/after/ab/ab_equivalence_report.txt` — **ALL EQUAL**:

* equal `pure.output_checksum` digests per config (`0f190fee3766…` dense, `6adddedf1ccf…` specialized),
  and the frozen arm's digests equal the canonical after-records';
* the saved `<TAG>_output.npz` files are **byte-identical** across the arms (sha256 `7964012d6e5b…` /
  `edd599e44dab…`);
* all **53** captured arrays `np.array_equal(…, equal_nan=True)` with `max|diff| = 0`, identical NaN
  placement, identical shapes/dtypes.

The A/B arms' own medians (dense 77.344 frozen vs 76.543 fallback; specialized 6.498 vs 6.623) are a
separate out-of-band sample of the same route and are not the canonical numbers. The two
`FROZEN-ADAPTER` seams in `../pure_forward.py` — the assumed `build_*(forest, x)` call form and the no-arg
`.run()` — fit the landed signatures with **no adjustment** (`src/evogp/tree/RUNNER_HOOKS.md` §1).

### STEP 3 — the type 1 (`problem.evaluate`) matrix, 14 cells

`--task eval64 --eval-reps 25`, one process per cell, `--stage-probe` installed (its tax is accepted
here), 12 **shadow** cells re-measuring the R3 Arm A matrix (both backends × `auto` / `dynamic` /
`specialized`) plus 2 **stock** twins of the headline iree-`auto` cells against the editable etl install,
in the same window. Driven in 3 chunks; **14/14 exit 0, `complete=true`**, validator
`python3 test/benchmark/r4/check_e2e_r4.py` → **14 complete / 328 checks / 0 failures** (exit 0).

| id | kind | `median_warm_ms` | `first_call_ms` | fds | cold misses |
|---|---|---|---|---|---|
| `e2e_fixture64_iree_auto` | shadow | 6.667 | 10 060.5 | 39→91→91 | 1 |
| `e2e_stock_fixture64_iree_auto` | stock | 6.556 | 10 069.3 | 39→91→91 | 1 |
| `e2e_fixture200_iree_auto` | shadow | 6.050 | 10 569.7 | 39→91→91 | 1 |
| `e2e_stock_fixture200_iree_auto` | stock | 6.461 | 10 393.5 | 39→91→91 | 1 |
| `e2e_fixture64_numpy_auto` | shadow | 10.844 | 191.7 | 39→59→59 | — (numpy) |
| `e2e_fixture64_numpy_dynamic` | shadow | 71.373 | 1 537.0 | 39→59→59 | — (numpy) |
| `e2e_fixture64_numpy_specialized` | shadow | 10.853 | 192.5 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_auto` | shadow | 52.891 | 1 054.2 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_dynamic` | shadow | 85.244 | 1 639.7 | 39→59→59 | — (numpy) |
| `e2e_fixture200_numpy_specialized` | shadow | 54.122 | 1 121.2 | 39→59→59 | — (numpy) |
| `e2e_fixture64_iree_dynamic` | shadow | 7.398 | 10 176.3 | 39→91→91 | 1 |
| `e2e_fixture64_iree_specialized` | shadow | 59.554 | 12 983.8 | 39→91→91 | 48 |
| `e2e_fixture200_iree_dynamic` | shadow | 6.348 | 10 358.9 | 39→91→91 | 1 |
| `e2e_fixture200_iree_specialized` | shadow | 193.086 | 43 381.9 | 39→91→91 | 155 |

Row order is the runner's table order (stock cells follow their shadow twins).  Every cell ran 25 warm
calls; the warm min–max per cell and the full stage-probe split are in `SUMMARY.md`.

* **Same-window stock ↔ shadow** (the only two cells with a twin): 6.556 → 6.667 ms (+0.111 ms,
  **+1.7 %**) at fixture64 and 6.461 → 6.050 ms (−0.411 ms, **−6.4 %**) at fixture200 ⇒ the patched-ETL
  shadow is **warm-neutral** inside the ±10–20 % shared-box band.
* **Against R3 and the legacy anchor** (both cross-window): R3 Arm A quoted 10.64 / 10.50 ms on these two
  cells, so this window is 1.60× / 1.74× faster — and the same-window stock twin shows the gain comes
  from the evogp side, not the ETL patch.  Against the legacy CUDA anchor (0.6323 / 0.6357 ms) the
  old-vs-new gap is ≈10.5× / 9.5× (shadow) and ≈10.4× / 10.2× (stock), i.e. **≈9.5–10.5×**, down from
  R3's ≈16.5–16.8×.
* **Routes**: on iree `auto` ≈ `dynamic` (6.667 / 7.398 ms at 64, 6.050 / 6.348 ms at 200 — both are the
  dense evaluator) while `specialized` costs 8.9× / 31.9× more; on numpy `auto` == `specialized`
  (10.844 ≈ 10.853, 52.891 ≈ 54.122) and `dynamic` is the slow dense alias (71.373 / 85.244).  The
  ≈10.1–10.6 s `first_call_ms` on every iree-`auto`/`dynamic` cell is the one-time dense-graph compile
  (cross-window-compatible with R3's Arm A cold calls, 13 640.3 / 13 207.4 ms).
* **Stage probe**: the dense-routed cells (`auto`, `dynamic`) report their cost in the uninstrumented
  `misc` bucket (`build_calls = 0`, a probe limitation); the iree `specialized` cells are dominated by the
  cold `build_miss` inside call 0 (12 809 ms / 42 868 ms = 48 / 155 builds) plus readback 445.8 / 1 354.8 ms
  over the 25 calls; the numpy cells split into `build_miss` 153–978 ms, `run` 232–1 237 ms,
  `decode` 27.6–115.9 ms, `readback` 16.0–59.7 ms.
* **Chunking**: the 4 headline cells ran 07:44:34–07:45:28, the 6 numpy cells 07:45:34–07:48:04 and the
  4 remaining iree cells 07:48:32–07:51:42.  `results/after/e2e/run_e2e_r4.last_run.txt` **is committed**
  but holds only the last chunk's 4-row table (it is regenerated per invocation) — the other exit codes
  come from the per-cell log headers.

### Anomalies, deviations and caveats (after-round)

1. **Sandbox teardowns cost one retry each** (see STEP 1 and STEP 2 above): the killed
   `fixture64_iree_spec` (two blocks in its log; the committed agreement `last_run.txt` lists only the 4
   resumed records) and one pure-matrix batch attempt (`pure64_dense_numpy.log` duplicated header,
   35 vs 25 lines).  Machine-written logs were left as written; every record they describe is `complete`
   and exit 0.
2. **The type 1 matrix ran chunked**, so the committed `run_e2e_r4.last_run.txt` covers only the final
   chunk (4 rows).  All 14 JSONs pass `check_e2e_r4.py`; a reader must not read that table as "4 records
   were produced".
3. **numpy-dense companions below the pure number** (−10.26 / −12.58 ms) — the same class as the
   before-round anomaly #1 (CPU-bound backend, sub-second box noise; the same route agrees to ≈2 % one
   step later on the 200-tree fixture).  Reported as measured; no rerun and no cherry-picking.
4. **IREE shutdown dump**: every IREE process prints a `nanobind: this is likely caused by a reference
   counting issue` teardown dump after its `done` line.  All exit codes are 0 and all records are
   `complete`; it is an interpreter-teardown artifact of the IREE runtime binding, not a failed record.
5. **The ETL shadow is identified by path, not by version** (see the identity table): `etl_version` is
   `0.1.0` for both checkouts and `rev=` is the editable repo's HEAD; the shadow identity is the
   `file=/tmp/evogp_etl_patched_87c7562/etl/__init__.py` path in every log, backed by the committed
   import probe.  Because the branch is gone and `/tmp` is session-scoped, that probe + the per-log line
   are the surviving evidence.
6. **Contention**: GPU 6 was idle at the start (`0 %, 1 MiB`) with foreign tenants on GPUs 0–2 at 90–100 %
   and reached `0 %, ~269 MiB` (a context from the parallel pure worker) at the end.  No flock contention
   beyond normal interleaving, no timeout, no guard hit, `ulimit -n 65536` (rlimit recorded 65536 in all
   28 after-records), fd curves flat per backend.

### Artifacts of the after-round (committed)

| commit | paths | what |
|---|---|---|
| `a9a076b` | `check_agreement_r4.py`, `run_agreement_r4.sh`, `run_pure.sh` | tooling prep: the additive `--agree-dir` / `R4_AGREE_DIR` override chain in both agreement tools and `PURE_EXTRA_ARGS` (extra flags pass-through) in `run_pure.sh` — the frozen default paths keep pointing at the before records |
| `b55b0bc` | `results/after/etl_import_probe.txt` | the shadow materialization + fresh-process import-precedence probe (the shadow's evidence of record) |
| `1bcac8e` | `results/backend_agreement_after/**` (19 files) | STEP 1: the 6 pre-flight records + logs + `_values.npy` + the committed `run_agreement_r4.last_run.txt` |
| `8b04156` | `check_e2e_r4.py` (734 lines), `e2e_manifest.txt` (56), `run_e2e.sh` (590) | STEP 3 tooling: the type 1 runner, its 14-cell expectation list and its stdlib-only validator |
| `115a324` | `manifest_after.txt` (52), `results/after/*.{json,log}` (8 pairs), `results/after/ab/**` | STEP 2: the pure matrix records + the A/B arms, saved outputs and `ab_equivalence_report.txt` |
| `b247e15` | `results/after/e2e/**` (29 files) | STEP 3: the 14 type 1 records + logs and the committed `run_e2e_r4.last_run.txt` |
| not committed | `results/after/*/last_run.txt` (where generated), `pure*.log` scratch | generated/disposable — except the two after `last_run.txt` files that were deliberately committed with their records |

Final gates (re-run after the after-round artifacts were committed; all four exit 0):

```bash
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest_after.txt --require-companion
#   -> 8 record(s): 8 complete, ...; 369 check(s), 0 failure(s)
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
#   -> 8 record(s): 8 complete, ...; 369 check(s), 0 failure(s)   (before records untouched)
python3 test/benchmark/r4/check_e2e_r4.py
#   -> 14 record(s): 14 complete, ...; 328 check(s), 0 failure(s)
$PY test/benchmark/r4/check_agreement_r4.py --agree-dir test/benchmark/r4/results/backend_agreement_after
#   -> PASS-with-note on both fixtures; gate: PASS (30 check(s), 0 failure(s))
```

Folding these numbers into the parent-level documents (`../REPORT.md` Round-4 verdict, `../CONTEXT.md`,
`../README.md`) belongs to the parent node, not to R4.
