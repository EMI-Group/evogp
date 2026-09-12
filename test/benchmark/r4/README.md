# `test/benchmark/r4/` — R4 round: the **pure forward** measurement (measurement type 2)

Round 4 adds a second *kind* of measurement to the old-vs-new benchmark, next to the
end-to-end numbers every earlier round quoted.

| | measurement type 1 — END-TO-END | measurement type 2 — PURE FORWARD |
|---|---|---|
| what is called | `problem.evaluate(forest)` (the public API, `eval64` / `evalfresh`) | one full-population forward on a fixed `x`, `per_row=False` |
| timed region | the whole public call | the forward pass only |
| includes | host decode + compile-key derivation + `build_cached` lookup + per-tree/per-chunk `etl.run` + device→host readback + host row assembly + the fitness reduction | `etl.run` (wrapper validation + backend invoke), once per graph call |
| excludes | — | compile-key derivation, readback, host row assembly, fitness reduction, the one-time input conversion/upload |
| exists for the legacy engine? | yes (one fused kernel call per population) | **no** — the legacy evaluate is a single kernel with the readback and the assembly inside it, so the pure number is undefined there (`--task eval-pure` skips cleanly, exit 3) |

`E2E − pure` is therefore the per-call **host glue + readback** cost, which is exactly the
residual the R2/R3 reports describe qualitatively (per-tree dispatch/readback, run-path
validation, host `rows` assembly).  R4 measures both sides of that difference in the *same
process* (`--e2e-companion K`), so the subtraction is not a cross-window comparison.

**Status: harness + smoke complete; the 8-config campaign is NOT run yet** (a later agent runs
`run_pure.sh`).  The after-round will, once the parallel code round lands the frozen
`build_{dense,specialized}_runner` API: (1) re-run this same script unchanged, (2) compare the
`local-fallback:*` records against the `frozen-api:*` ones (the A/B equivalence check below),
(3) fold the numbers into `../REPORT.md` and the round write-up.

## Scope — 8 configs in two groups

The config matrix is 2 fixtures × 2 routes × 2 backends:

| id | group | mode | backend | fixture | per-config env |
|---|---|---|---|---|---|
| `pure64_dense_numpy` | numpy | dense | numpy | `test/goldens/forest_random.npz` (64 trees) | `EVOGP_EVAL_MODE=dense` |
| `pure64_specialized_numpy` | numpy | specialized | numpy | same | `EVOGP_EVAL_MODE=specialized` |
| `pure200_dense_numpy` | numpy | dense | numpy | `test/benchmark/fixtures/forest200_tree64_random.npz` (200 trees) | `EVOGP_EVAL_MODE=dense` |
| `pure200_specialized_numpy` | numpy | specialized | numpy | same | `EVOGP_EVAL_MODE=specialized` |
| `pure64_dense_iree` | iree | dense | iree | 64-tree fixture | `EVOGP_EVAL_MODE=dense` |
| `pure64_specialized_iree` | iree | specialized | iree | 64-tree fixture | `EVOGP_EVAL_MODE=specialized` |
| `pure200_dense_iree` | iree | dense | iree | 200-tree fixture | `EVOGP_EVAL_MODE=dense` |
| `pure200_specialized_iree` | iree | specialized | iree | 200-tree fixture | `EVOGP_EVAL_MODE=specialized` |

Common args (one `bench_gp.py` process per config, exactly these):

```
--task eval-pure --fixture <F> --mode <dense|specialized> --backend <numpy|iree>
--reps 5 --warmup 10 --invocations <100 dense|50 specialized> --e2e-companion 11
--mem-guard-mb 8192            # + --max-cold-builds 400 on the two iree groups' configs
```

`<F>` is passed explicitly (`test/goldens/forest_random.npz` for `pure64_*`,
`test/benchmark/fixtures/forest200_tree64_random.npz` for `pure200_*`).  `--tag` is
`UPPER(id)`; results go to `results/<id>.json` + `results/<id>.log`.

**Both knobs are set.**  The pure route comes from `--mode`; the `--e2e-companion` route comes
from the engine's own policy (`EVOGP_EVAL_MODE`), which the runner exports per config with the
same value.  That is what makes `E2E − pure` a single-route difference; a config that passed
only one of the two would silently compare two different routes.

`run_pure.sh --list` prints the table plus the pre-flight guard state, `--dry-run` prints the
exact `flock … bash -c '…'` line of every config (execute nothing).

## What one "invocation" is — and what it still pays

One invocation = **one full-population forward on the fixed `x`** (`per_row=False`, i.e. the
`eval64`/`problem.evaluate` shape).  `pure.conventions` in every JSON states this in one line.

Timed (inside the block): `etl.run(executable, …)` per graph call — the ETL wrapper's own
validation and the backend invoke.  Those cannot be removed without measuring nothing, so they
are the honest floor.

Not timed (excluded by construction):

* the cold construction — reported separately in the `compile` block;
* the compile-key derivation and the memoized `build_cached` lookup: the per-tree
  `_executable(program, n_rows, input_len, backend, device, options)` (specialized) / the
  shared dense graph build happens once, during the construction;
* the input conversion + the one-time device upload (a fixed `x`);
* the per-tree/per-chunk result readback (device→host) and the host `rows` assembly: results
  are **dropped**, never retained (holding `N` population outputs would only grow the
  allocator);
* `torch.from_numpy` wrapping and the fitness reduction (end-to-end only).

Note that whole-population device-side stacking does not exist today: the specialized pure
loop is one `etl.run` per non-constant tree (a constant/collapsed tree is evaluated as a
constant and never invoked), and the dense route is one `etl.run` per chunk.  `per_graph_call_ms`
is the honest per-`etl.run` normalization; the population-level number is
`per_population_evaluate_ms`.

## `--task eval-pure` flag reference

| flag | default | effect |
|---|---|---|
| `--task eval-pure` | — | the pure-forward task (new side only; `skipped: true` + exit 3 on the legacy side) |
| `--fixture F` | `test/goldens/forest_random.npz` | the population + `x` fixture (same schema as `eval64`) |
| `--data-rows N` | fixture rows | tile `x`/`labels` up to N rows, exactly as `eval64` does (N == native rows is a no-op) |
| `--mode {auto,dense,specialized,dynamic}` | `auto` | route to measure: explicit value wins; `auto` asks the engine (`EVOGP_EVAL_MODE`, then `dense_engine.should_use_dense(pop, backend)`); `dynamic` is a **dense alias** (recorded as requested `dynamic`) |
| `--invocations N` | `50` | full-population forwards per timed block |
| `--warmup W` | `10` | one untimed warmup block right after the construction (its measurement is discarded) |
| `--reps R` | `5` (eval-pure default) | number of timed blocks; the runner is constructed **once** for the whole task (this path consumes no RNG, so the per-rep reseed of the train/eval runners would be noise) |
| `--e2e-companion K` | `0` | after the pure reps, K synchronized `problem.evaluate(forest)` calls in the same process (the compile cache is already warm) |
| `--pure-runner {auto,frozen,fallback}` | `auto` | which constructor: `frozen` = the frozen `build_{dense,specialized}_runner` API (**hard error** when absent), `fallback` = the harness' own re-implementation (forces it even when the frozen API exists), `auto` = frozen when present else fallback |
| `--backend {default,numpy,iree}` | `default` | as everywhere else: exports `EVOGP_ETL_BACKEND` (+ the IREE settings) before `evogp` is imported |
| `--mem-guard-mb`, `--max-cold-builds`, `--allow-cold-builds`, `--tag`, `--json-out`, `--notes`, `--seed`, `--max-wall-s`, `--min-gens` | unchanged | shared harness machinery (the cold-build guard only bites on new-side iree runs) |
| `--stage-probe` | off | **accepted but never installed** for this task: the probe monkeypatches `etl.run` and would tax exactly the loop being measured.  A warning is printed and `stages` stays `null` |

Exit codes are the harness' usual ones: `0` complete, `3` skipped (guard stop / legacy side),
`1` crash, `130` interrupted.

## JSON glossary (`results/<id>.json`)

Standard keys (`tag`, `side`, `task`, `fixture`, `evogp_file`, `evogp_commit`, `etl_version`,
`env`, `gpu`, `fds`, `nofile_rlimit`, `cold_builds`, `skipped`, `skip_reason`, `partial`,
`projection`, `complete`, `per_rep`, `notes`, `started_at`, `finished_at`, `stages` = `null`)
are unchanged; `config` gains `mode`, `pure_runner`, `invocations`, `warmup`, `e2e_companion`
and (once the fixture is loaded) `n_rows`.

| key | meaning |
|---|---|
| `pure.mode_requested` | the `--mode` value as passed (`auto` means "the engine's policy decided") |
| `pure.mode_resolved` | the route actually measured: `dense` or `specialized` |
| `pure.runner_source` | `frozen-api:build_dense_runner` \| `frozen-api:build_specialized_runner` \| `local-fallback:dense` \| `local-fallback:specialized` — **which constructor produced this record** |
| `pure.n_graph_calls_per_run` / `pure.n_graph_calls_source` | `etl.run` calls per invocation (specialized: non-constant trees; dense: chunks) and how that count was obtained (`measured` \| `introspected` \| `assumed-1`) |
| `pure.n_trees` / `pure.n_const_trees` | population size / trees the specialized route never invokes (constant or collapsed).  On the **dense** routes this is `0` by construction (the graph evaluates every row, constant trees included) — a documented key convention, not a measurement |
| `pure.per_row` | always `false` (the pure task is defined on `per_row=False`) |
| `pure.invocations` / `pure.warmup` / `pure.reps` | the flags as passed |
| `pure.block_ms` | one entry per rep: the wall time of that timed block (one `torch.cuda.synchronize()` pair around it) |
| `pure.per_invocation_ms` | `block_ms[i] / invocations` per rep |
| `pure.median_per_invocation_ms` / `min_…` / `mean_…` | statistics **across the reps** of `per_invocation_ms` (the headline; `min` is a floor estimate, `median` the headline, `mean` shows spread) |
| `pure.per_population_evaluate_ms` | **equal to `median_per_invocation_ms` by definition** — one invocation IS one population evaluate, just without the readback/assembly.  It exists so a reader cannot mistake the pure number for something else |
| `pure.per_graph_call_ms` | `median_per_invocation_ms / n_graph_calls_per_run` |
| `pure.per_tree_us` | `median_per_invocation_ms * 1000 / n_trees` |
| `pure.conventions` | the one-line statement of the measurement's scope |
| `compile` | the cold construction, never mixed into a pure number: `decode_plan_ms` (host decode / dense plan + input conversion + the one-time device upload), `build_ms` (the `build_cached` calls only), `construct_total_ms` (the whole construction; the residual is the invocation-argument precompute), `cache_hits` / `cache_misses` (the shared `etl.CompileCache` delta around the construction), `path` (`frozen` \| `fallback`) |
| `e2e_companion` | `null` when `--e2e-companion 0`; else `{eval_reps, per_call_ms[], first_call_ms, median_warm_ms, note}` — the real `problem.evaluate(forest)` path in the same process, one `torch.cuda.synchronize()` pair per call.  With `K == 1` there is no warm call, so `median_warm_ms` is the median of the single call (documented, so the field is never a null) |
| `median_per_invocation_ms` (top level) | running mirror of `pure.median_per_invocation_ms` (the `median_warm_ms` convention of the eval tasks) |
| `per_rep[]` | one record per block: `{rep, construct_s (rep 0 only), wall_s, block_ms, per_invocation_ms, mode_resolved, runner_source, partial, projection, complete}`.  `rep 0`'s `wall_s` includes the construction (the eval-task convention); every later rep's `wall_s` is its own block |

`median_per_invocation_ms` is the number to quote; `compile` answers "what did it cost to get
there"; `e2e_companion` answers "how much host glue does the real call add".

## Frozen vs fallback (labeling discipline)

`pure.runner_source` and `compile.path` say which constructor ran; never present a
`local-fallback:*` number as if it were the frozen engine's.

* `fallback` (this round's only executable path — the frozen API does not exist at HEAD) is the
  local re-implementation in `test/benchmark/pure_forward.py`: dense = `plan_dense` +
  `build_dense` + the per-chunk invocation args; specialized = `decode_population` + one
  `_executable(...)` per non-constant tree.  Both use the engine's **own** memo key and the
  **shared** `etl_engine.CACHE`, so the compile block and the warm path are the engine's, not a
  re-implementation of them.
* `frozen` calls `dense_engine.build_dense_runner(forest, x)` /
  `etl_engine.build_specialized_runner(forest, x)`.  The call form and the invoke form live in
  two clearly marked `FROZEN-ADAPTER` functions in `pure_forward.py`; if the landed API wants
  the input per call (or exposes its own graph-call counter), only those need the one-line
  adjustment.  A frozen runner without a graph-call counter records
  `n_graph_calls_source: "assumed-1"`, i.e. its `per_graph_call_ms` is 1/… of the invocation —
  never quote it as a measured per-call cost.
* **A/B equivalence check (after-round):** run the same config twice, once with
  `--pure-runner fallback` and once with `--pure-runner frozen`, and diff
  `pure.median_per_invocation_ms` / `compile.construct_total_ms`.  Both are the same engine
  work, so a large gap means the frozen wrapper adds (or skips) something — investigate before
  quoting either.

## Commands (all from the **repository root**)

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python        # no `python` on PATH
# --- the matrix (GPU; one bench_gp.py process per config) --------------------
test/benchmark/r4/run_pure.sh --list                   # config table, execute nothing
test/benchmark/r4/run_pure.sh --dry-run                # every exact command, execute nothing
test/benchmark/r4/run_pure.sh --group numpy            # 4 configs, numpy backend
test/benchmark/r4/run_pure.sh --group iree             # 4 configs, IREE-cuda (ulimit -n 65536)
test/benchmark/r4/run_pure.sh                          # all 8 (numpy then iree)
test/benchmark/r4/run_pure.sh --only pure64_dense_numpy
test/benchmark/r4/run_pure.sh --only pure64_dense_iree --timeout 3600
# --- validator (no GPU, no torch) -------------------------------------------
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
python3 test/benchmark/r4/check_r4.py                  # scan results/ (smoke/ skipped)
python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/results/smoke/manifest_smoke.txt
```

`--only` overrides `--group` (unknown ids are a usage error, exit 2); selection is always in
table order (`numpy` before `iree`), so an IREE-window problem can never invalidate the numpy
rows.

### Environment overrides of the runner

`PY`, `CUDA_VISIBLE_DEVICES` (default `6`), `TIMEOUT_S` (default `1800`, or `--timeout`),
`LOCK` / `FLOCK_WAIT` (default `/tmp/gpbench_critical.lock` / `7200`), `NEW_PYTHONPATH`
(default `src`; prefix another checkout — e.g. a patched ETL — to shadow it, exactly like R3
Arm B), `NEW_RESULT_DIR` (default `test/benchmark/r4/results`), `ETL_REPO` (the checkout whose
HEAD is recorded, default `/mnt/local-ssd/bchuang/etl`).

## Method / GPU rules (unchanged from R2/R3)

* One `bench_gp.py` **process per config** — the compile cache is per-process, and each config
  logs the `etl:` line (`git -C <ETL_REPO> rev-parse HEAD` + `import etl; etl.__file__`) into
  its own `<id>.log` for shadow-evidence discipline.
* `ulimit -n 65536` first (IREE file descriptors), exactly ONE
  `flock -w 7200 /tmp/gpbench_critical.lock` per invocation, never nested, with the `timeout`
  **inside** the lock so lock wait is never charged to a config.  GPU 6 by default; **never**
  GPU 7 (hardware ECC fault).  Medians on this shared box are ±10 %.
* Warm/steady-state only: the construction is timed separately (`compile`), then `--warmup 10`,
  then the timed blocks.  A `--reps 1` `--warmup 0` run is NOT quotable.
* IREE stays opt-in and non-primary; only `O0`/`O1` optimization levels are ever used, and an
  IREE number is only quotable once the backend-agreement pre-flight of the round (R2/R3
  procedure) passes on the same box window.
* The legacy side is inert to every `ETL_*`/`EVOGP_ETL_*` variable and does not implement this
  task at all — pointing `--task eval-pure` at it must produce `skipped: true` + exit 3, never
  a crash.

## Files

| file | what it is |
|---|---|
| `README.md` | this protocol |
| `run_pure.sh` | the 8-config matrix runner (`--group numpy\|iree\|all`, `--only`, `--list`, `--dry-run`, `--timeout`); writes `results/<id>.{json,log}` and `last_run.txt` (generated, gitignored) |
| `manifest.txt` | the validator's expectation list (`id <TAB> mode <TAB> backend <TAB> path`), 8 canonical ids in runner order |
| `check_r4.py` | stdlib-only record validator (structure, modes/backends vs manifest, timing-list consistency, derived normalizations, optional companion gate); exit 1 on any failure |
| `results/smoke/` | the pre-campaign plumbing records (3 pure smokes + one `eval64` regression control) and their scratch manifest; **not** manifest ids |
| `results/<id>.json` | the campaign records (written by the next agent) |
| `CONTEXT.md` | the node's routing/context file |

`last_run.txt` is generated output and gitignored; the smoke records are committed as plumbing
evidence.

## IREE agreement pre-flight (run before quoting any IREE number)

Same convention as R2/R3, on this round's two fixtures: numpy vs IREE-cuda O1 under the default `auto`
policy (→ the dense route) vs IREE-cuda O1 with `EVOGP_EVAL_MODE=specialized`, six records (two fixtures ×
three roles), one process each, output into `results/backend_agreement/`.

```bash
# run the six records (GPU; drives the read-only R2 driver test/benchmark/r2/check_backend_agreement.py)
bash test/benchmark/r4/run_agreement_r4.sh                 # --only STEMS | --timeout S | --list | --dry-run
# then the gate (numpy only, no GPU; exit 0 = PASS / PASS-with-note, exit 1 = hard FAIL)
/mnt/local-ssd/bchuang/evox/.venv/bin/python test/benchmark/r4/check_agreement_r4.py
```

`run_agreement_r4.sh` sets the common env (`PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=6`,
`ETL_DISABLE_LOCATIONS=1`, `EVOGP_ETL_IREE_DEVICE=0`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`) plus
`EVOGP_ETL_BACKEND` (and `EVOGP_EVAL_MODE=specialized` on the `*_iree_spec` records) per record, raises
`ulimit -n 65536` first, and wraps each record in exactly one `flock -w 7200 /tmp/gpbench_critical.lock`
with `timeout -s INT -k 30 ${TIMEOUT_S:-1800}` **inside** it; stdout+stderr go to
`results/backend_agreement/<stem>.log`, the id → exit table to
`results/backend_agreement/run_agreement_r4.last_run.txt` (disposable). `fixture200_iree_spec` compiles ~170
structures and is the slow record (~1 min here) — give it a generous cap.

`check_agreement_r4.py` re-checks each record's identity (fixture sha256 vs the npz, `<stem>_values.npy` vs
the JSON's `values_sha256`, population size, NaN/±inf counts) and then, per fixture, requires the three roles
to report the **same** `values_sha256`; when a hash differs it falls back to
`np.allclose(rtol=1e-4, atol=1e-4, equal_nan=True)` on every role pair with `max|diff|` / max rel diff
reported — within tolerance is **PASS-with-note** (quotable, deviation recorded), beyond it is a **hard FAIL**
and no IREE number from the round may be quoted.

Evidence of the executed pre-flight (2026-09-13, GPU 6), its verdict and its R3/R2 hash continuity:
`RUNLOG.md` § "STEP 1"; the records, logs and raw fitness vectors are in `results/backend_agreement/`.
