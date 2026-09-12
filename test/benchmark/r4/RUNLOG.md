# R4 run log — the pure-forward before-baseline campaign as actually executed

Operator log for the R4 pre-flight + matrix (`r4/run_agreement_r4.sh`, `r4/run_pure.sh`; protocol:
`r4/README.md`, validator: `r4/check_r4.py`, config contract: `r4/manifest.txt`, harness: `../bench_gp.py` +
`../pure_forward.py`). One section per phase, in execution order. Raw evidence stays in the harness JSONs +
the appended per-config logs (`results/<id>.{json,log}`, `results/backend_agreement/<stem>.{json,log}` +
`<stem>_values.npy`) — this file is the summary and the interpretation, not a second source of numbers.
Values marked *derived* are computed here from the committed JSONs (everything else is a harness field).

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

Next round (after the parallel code round lands the frozen `build_{dense,specialized}_runner` API): re-run
`r4/run_pure.sh` unchanged and A/B the `local-fallback:*` records above against the `frozen-api:*` ones.
