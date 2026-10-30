# `test/benchmark/evoxc/` — evoxc engine vs plain-PyTorch, controlled harness

The question this subtree answers: **how much does the evoxc-based EvoGP engine cost
per evaluation compared with the plain-PyTorch implementation of the same tree
evaluation?** — on identical populations, in the same session, warm/steady-state,
with every arm validated numerically against the repo's independent oracle.

It is deliberately separate from `../r2/`, `../r4/`, `../r5/`, `../r6/`, which compare
against the *deleted legacy CUDA engine* and stay frozen as history.

The current evidence is the post-optimization campaign in `results/final/` (HEAD
vs the pre-optimization snapshot `530518e`, same window, same harness) plus its
cold-start and whole-training-run columns; [`FINAL.md`](FINAL.md) is the report of
record and [`SUMMARY.md`](SUMMARY.md) is the rendered earlier baseline round.
A second round prices the opt-in `batch` route on a fixed population and inside a
real `StandardPipeline.step()` loop — records in `results/batch/`, report of record
[`BATCH.md`](BATCH.md) — and its third round re-prices the shipped default against a
pool-1 control and `dense` and settles the compile-pool and band-size knobs
(`results/batch_opt_round3/`, [`BATCH.md`](BATCH.md) §15, summarized under
[Round 3](#round-3-the-refreshed-columns-the-compile-pool-default-and-the-band-cap) below).

```
common.py       workload construction (one definition shared by every arm): cell
                grid, env block, warm-timing helper, route inspection
arms_torch.py   the two baselines: PerTreeEvaluator (Python tree walk) and
                BatchedEvaluator (one vectorized pass per population)
arms.py         the arm namespace: TorchArm / EvoxcArm wrappers, env application,
                SR loss problem + GP pipeline construction
worker.py       ONE process, ONE (cell, arm): the measured unit
campaign.py     sequential driver: one subprocess per config, idle-device guard,
                `--tree-root` switch, per-config logs + `campaign.json`
validate.py     numeric validation of every arm against reference_eval.py
decompose.py    per-call phase breakdown of one cell on both routes
summarize.py    renders the records into one markdown report (stdlib only)
check.py        the gate over one campaign result directory (stdlib only, exit 1
                on failure); a before/after round is gated one side at a time
cold_start.py   cold-start column: import -> first fitness on the default route,
                one process per population; --artifact-dir = populate + reload
training_run.py end-to-end training run: per-generation wall ms + best-fitness
                trajectory for the engine, the plain-torch evaluator and a snapshot
batch_economics.py batch-route compile economics over a real
                StandardPipeline.step() loop (`econ_child.py` is its child half)
counter_blocks.py strict number/block helpers the optional batch counter surfaces
                share (a missing or non-numeric counter reads None, never 0)
band_report.py  the band/merge report of one merged plan (fields + policy verdict)
stages.py       the ten-stage per-call vocabulary and every derived stage block
content.py      the content-addressed plan-key table of the batch route and every
                derived block (attempts / hit_rate / per-row outcome)
reuse_probe.py  plan-key reuse workload: cold, identity, rebuilt-but-equal, identity
band_sweep.py   band-size / merge-composition sweep driver (one record per
                (cell, config): warm medians, stage split, band report, oracle)
band_measure.py its child half: the measurement loop and the record skeleton
results/        the evidence: baseline/ (frozen earlier round), final/ (after/,
                before/, decompose/, cold/, training/, validation.json, RENDERED.md)
                and batch/ (the warm-r5/small-r5/flat-r5/flat1000-r5/flat1000-3w-r5
                rounds, <cell>/<route>.json, economics.json, validation-*.json,
                RENDERED-*.md and the --reps 3 pilots), batch_opt_round1/,
                batch_opt_round2/ and batch_opt_round3/ (the newest batch evidence,
                rendered in "Round 3" below; each round has its own SUMMARY.md)
FINAL.md        report of record for the post-optimization dense/specialized campaign
BATCH.md        report of record for the batch route round (§15 = round 3)
SUMMARY.md      rendered report of the earlier baseline round
VALIDATION.md   oracle validation of the final sweep + cross-arm digest agreement
```

## Arms

| arm | what it is |
|---|---|
| `torch-per-tree` | the classic pre-jit PyTorch route: for each tree, walk it in Python and apply torch ops to the `(N, input_len)` batch on the GPU under `torch.no_grad()`. The honest "old torch version". |
| `torch-batched` | the best a pure-torch implementation can do: the whole population in one vectorized pass (per-position `(pop, N)` buffer, one masked `torch.where` per occurring `(kind, function)` case, per-row child gather). No Python loop over trees. |
| `evoxc-auto` | the shipped default: both knobs unset, i.e. dense+iree at `pop >= 64` on a device box and specialized+numpy below that |
| `evoxc-auto-iree` | `auto` with `EVOGP_EVOXC_BACKEND=iree`: the same dense+iree route from `pop >= 64`, specialized+iree below the threshold |
| `evoxc-specialized-numpy` | `EVOGP_EVAL_MODE=specialized` + numpy |
| `evoxc-specialized-iree` | `EVOGP_EVAL_MODE=specialized` + iree (one compiled graph per distinct tree structure) |
| `evoxc-dense-iree` | `EVOGP_EVAL_MODE=dense` + iree (one all-ops graph per population shape) |
| `evoxc-batch-iree` | `EVOGP_EVAL_MODE=batch` + iree (one merged graph per population structure multiset, built with `evoxc.batch_functions`; numpy is the fallback, as on the dense route) |
| `evoxc-batch-iree-flat` | the same route with `EVOGP_BATCH_BANDS=flat`: one unbanded merge, O(N²) in the population, so the flat-vs-banded merge can be compared on real GP workloads (it is expected to lose badly at pop 1000 — that comparison is what the arm records) |
| `evoxc-batch-iree-aggressive` | the same route with `EVOGP_BATCH_DISPATCH_FORMATION=aggressive`: the merged compile's evoxc dispatch-formation option pinned instead of derived, so `auto`'s own choice can be compared against it (and the spelling can be priced where `auto` does *not* take it) |
| `evoxc-batch-iree-artifact` | the same route on the **persistent on-disk executable tier** (`EVOGP_EVOXC_ARTIFACT_DIR`, pointing at `campaign.py --artifact-dir` / `$EVOGP_BENCH_ARTIFACT_DIR`, default `<out>/artifacts`): a disk hit skips `lower`/`compile`/`load`, so this is the only arm whose *cold* column can change without the route changing |
| `evoxc-batch-iree-default` | the same route with `EVOGP_BATCH_DISPATCH_FORMATION=default`: the **pre-change control** of the merged compile — no `--iree-dispatch-creation-enable-aggressive-fusion`, i.e. the spelling the route derived before the knob grew its `auto` default. `evoxc-batch-iree` (no value → `auto` → `aggressive` on a rocm/cuda target) stays the shipped default; this arm is what its records are read against |
| `evoxc-dense-iree-raised` | `EVOGP_EVAL_MODE=dense` + iree with the trace cap pinned to the **cell's** tree length (`EVOGP_DENSE_MAX_TREE_LEN={tree_len}`): identical to `evoxc-dense-iree` on a main-grid cell (L ≤ 64) and the same route with the cap raised on a long-tree cell, so dense actually serves that population instead of declining to `specialized` |

Both torch arms are *implemented in this harness* (`arms_torch.py`), with op semantics
copied from the oracle (`DIV` → NaN on a zero divisor, `loose_div`/`loose_inv` clamping
the divisor with `copysign(DELTA, y)`, `max`/`min` via `torch.where`, comparisons
returning ±1.0, `if` selecting on `cond > 0`) and the decode/plan cost built once
outside the timed region, reported as `setup_ms`.

The two `evoxc-batch-*` arms are the harness's handle on the engine's **third
evaluation route** (`EVOGP_EVAL_MODE=batch`, `src/evogp/tree/batch_engine.py` through
`evoxc.batch_functions`): instead of one graph per population *shape* (`dense`) or per
tree *structure* (`specialized`), it compiles **one merged graph per population**,
keyed on the multiset of tree structures. `auto` resolves to `dense` or `specialized`
only, so all batch arms pin the mode; none is a default-route claim, and the measured
verdict is in [`BATCH.md`](BATCH.md) and
[Batch route: compile economics](#batch-route-compile-economics-batch_economicspy).

**The improved batch route** is those same two arms plus the three below it, and they
answer the questions the shipped route could not:

* `evoxc-batch-iree-aggressive` — `EVOGP_BATCH_DISPATCH_FORMATION=aggressive`. The knob
  (single-sourced in `common.BATCH_KNOBS`, default `auto`) resolves the merged compile's
  `dispatch_formation` evoxc option: `auto` picks `aggressive` only for an iree backend
  whose compile target is a GPU (`rocm`/`cuda`) **and** whose installed evoxc declares
  the option (feature-detected off the iree adapter, `batch_chunks.supports_dispatch_formation`),
  and falls back to `default` everywhere else; `default`/`off` derive no flag at all; an
  explicit `aggressive` is honoured even on a non-GPU target (where it may be slower).
  The REQUEST is what the env var says, the EFFECTIVE spelling is on the plan
  (`BatchPlan.dispatch_formation`, reported by `describe_plan`) — a record states both,
  as `batch_plan_summary.band_report.dispatch_formation` /
  `dispatch_formation_requested` next to `batch_policy.dispatch_formation` (the request)
  and `routes.*.dispatch_formation` (the route's view of it).
* `evoxc-batch-iree-artifact` — the same route with the **persistent on-disk executable
  tier** on. `EVOGP_EVOXC_ARTIFACT_DIR` is ONE variable for the dense **and** the batch
  route (the batch route keeps its bodies in that directory's `batch/` region); a hit
  loads the compiled body and skips `lower`/`compile`/`load`, a partial hit compiles only
  the missing chunks. It is the only thing in this harness that changes the **cold**
  column without changing the route, so the arm's `first_call_ms` is the number to read
  next to its `batch_cache.artifact_totals` block (7 counters off
  `batch_engine.batch_artifact_stats()`: `enabled`/`files` are the tier's state,
  `hits`/`misses`/`load_failures`/`stores`/`save_failures` are the window's deltas). An
  operator points every process at ONE directory and a later process reloads instead of
  recompiling: `campaign.py --artifact-dir /abs/path` (exported to every worker as
  `EVOGP_BENCH_ARTIFACT_DIR`) or `export EVOGP_BENCH_ARTIFACT_DIR=/abs/path`; the default
  is `<out>/artifacts`, i.e. per run. `EVOGP_EVOXC_ARTIFACT_DIR` itself is per-arm
  provenance and is cleared for every other arm (`common.ROUTE_ENV_VARS`), so no
  pre-existing arm silently becomes a disk-cached run.
* `evoxc-batch-iree-default` — the **before/after control**, not a route of its own:
  `EVOGP_BATCH_DISPATCH_FORMATION=default` pins the merged compile to derive no
  `--iree-dispatch-creation-enable-aggressive-fusion` flag, which is what the route did
  before the knob's `auto` default started enabling it on a rocm/cuda compile target.
  `evoxc-batch-iree` sets nothing and therefore reads `auto` (`aggressive` here) and
  stays the shipped default; a record states the request (`batch_policy.dispatch_formation`:
  `auto` on one arm, `default` on this one — the `knobs_source` entry reads `arm` for
  both, since each value is one the registry really wrote) and the effective spelling
  (`batch_plan_summary.band_report.dispatch_formation`). A before/after therefore reads
  `evoxc-batch-iree-default` against `evoxc-batch-iree` on the same cell, same seed,
  same population.

Its knobs are all `EVOGP_BATCH_*`, single-sourced in `common.BATCH_KNOBS` and
resolved in a record as arm value > inherited environment > the engine's own
resolver > the documented default (`batch_policy.knobs_source` says which applied):

| knob | documented default | meaning |
|---|---|---|
| `EVOGP_BATCH_BAND_SIZE` | `16` | programs per band: the merge splits the population's deduped function count into `K = ceil(n / size)` bands, clamped by the binding policy (ROCm/HIP: `K ≤ 15`, and `≥ 8` functions per band) |
| `EVOGP_BATCH_BANDS` | engine default (unset) | an explicit band count `K`, or `none`/`flat` for one unbanded merge (O(N²) in the population — what the `-flat` arm times) |
| `EVOGP_BATCH_SWITCH_OPS` | `off` | whether the merged graph may use switch ops (read from the engine's own resolver) |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | band variants are aligned before the merge instead of being padded per call |
| `EVOGP_BATCH_CACHE_SIZE` | unset (engine default) | the merged-graph cache size; the harness records the resolved value and never sets it |
| `EVOGP_BATCH_MIN_POP` | `64` | population floor: below it one merged graph cannot amortize its build, so the route declines to `specialized` (the record's `routes.*.decline_reason` carries the sentence) |
| `EVOGP_BATCH_COMPILE_WORKERS` | `default` | the compiled-graph pool width: how many chunk compiles of the merged graph may run at once (the route default is on, 4 workers; `0`/`1` is the old single-executable path). It moves the route's **cold** cost, so it is swept with `batch_economics.py --knob EVOGP_BATCH_COMPILE_WORKERS=N` rather than with one arm per value, and the record reads the resolved width and its source |
| `EVOGP_BATCH_DISPATCH_FORMATION` | `auto` | the merged compile's evoxc `dispatch_formation` option: `auto` enables `aggressive` only for an iree backend whose compile target is a GPU (`rocm`/`cuda`) and whose evoxc declares the option, `default`/`off` derive no flag, an explicit `aggressive` is honoured everywhere (it may be slower on a non-GPU target). The record keeps the request (`batch_policy.dispatch_formation`) and the effective spelling (`batch_plan_summary.band_report.dispatch_formation`), because `auto` resolves per compile target; the `evoxc-batch-iree-aggressive` arm pins it and `evoxc-batch-iree-default` pins the pre-change `default` |

An arm spec may carry a `knobs` mapping (`{env var: value}`) beyond the two route
selectors; `-flat` is exactly `{"EVOGP_BATCH_BANDS": "flat"}`, `-aggressive` exactly
`{"EVOGP_BATCH_DISPATCH_FORMATION": "aggressive"}`, `-default` exactly
`{"EVOGP_BATCH_DISPATCH_FORMATION": "default"}`, `-artifact` exactly
`{"EVOGP_EVOXC_ARTIFACT_DIR": "{artifact_dir}"}` and `evoxc-dense-iree-raised` exactly
`{"EVOGP_DENSE_MAX_TREE_LEN": "{tree_len}"}`.
`arms.apply_arm_env(arm, cell)` clears **every** route variable
(`common.ROUTE_ENV_VARS`: the `EVOGP_BATCH_*` set, the dense route's
`EVOGP_DENSE_MAX_TREE_LEN` / `EVOGP_DENSE_AUTO_BUDGET_MB` / `EVOGP_DENSE_AUTO_MAX_ROWS`
and the disk tier's `EVOGP_EVOXC_ARTIFACT_DIR`) first
for every arm (torch arms included) and then writes the arm's own knobs, so no value
can leak from one arm's environment into another's. A knob value may be a per-cell
template (`{tree_len}`, `{pop}`, `{input_len}`, `{rows}`, resolved by
`arms.resolve_knob_value`; a template without a cell raises `ValueError` naming the arm
and the placeholder) -- plus `{artifact_dir}`, which is the harness's own hook
(`$EVOGP_BENCH_ARTIFACT_DIR`, else `<out>/artifacts`, resolved by
`common.artifact_dir`) and therefore needs no cell -- and `env.applied_knobs` carries the
**resolved** values; an
unregistered arm name raises `KeyError`. `--arms all` covers all thirteen arms and `--arms
batch` exactly the five batch arms, in registry order (registered last, so no existing
order or render changes). `campaign.py --dry-run` prints each arm's resolved knob spec
once (`arm knobs <arm> ...`), which is how an operator checks the env without a device.

The dense route's trace cap is `EVOGP_DENSE_MAX_TREE_LEN` (`dense_engine`,
default **64**): the evaluator traces `L = min(cap, max_tree_len)` positions, and a
population whose longest tree exceeds the cap raises `DenseUnsupported`, which makes
the public API fall back to the per-tree `specialized` route *silently*.
`evoxc-dense-iree` therefore measures the fallback on a long-tree cell,
`evoxc-dense-iree-raised` measures dense itself, and each explicit-dense record
states which happened under `dense_route` (`supported`, `reason`,
`max_tree_len_cap`, `tree_len_max`, `probe_ms`), with the resolved cap echoed in
`dense_policy.max_tree_len_cap`; `campaign.py` never forwards the variable from the
caller's environment, because it is per-arm provenance.

`auto`'s rows half is a **memory envelope** now, not a row ceiling:
`EVOGP_DENSE_AUTO_BUDGET_MB` (default **8192**) bounds
`pop_size * tree_len * rows * 4` bytes — dense's largest intermediate — and
`EVOGP_DENSE_AUTO_MAX_ROWS` is only an optional *tightening* pin (default unset/`0`, i.e.
unpinned). Every record states the policy it ran under in `dense_policy`
(`auto_min_pop`, `auto_max_rows`, `auto_budget_mb`, `max_tree_len_cap`) and the decision
per workload in `routes.*`, which is now computed WITH the cell's shape
(`common.route_decision` passes `pop_size=` / `rows=` / `tree_len=`), so a large-row cell
reports the envelope's own verdict instead of a pin-only reading.

Every batch arm needs a tree that ships the module *and* declares `batch` in
`dense_engine.EVAL_MODES`; where it does not, the worker's preflight fails the config
with one readable paragraph and writes **no** record (the dry runs below are the
torch-free way to learn that before paying a compile).  A tree that predates one of the
newer seams is not a failure: the missing accessor is reported as `null` with a note
(`batch_cache.artifact_totals` with `available: false` when there is no
`batch_artifact_stats()`, `band_report.dispatch_formation` as `null` when the plan
description predates the knob, `dense_policy.auto_budget_mb` as `null` when the policy
has no `resolve_auto_budget_mb`).

## Workloads

Every arm is timed on the same four things, same population, same data:

| workload | call |
|---|---|
| `sr_fitness` | `Forest.SR_fitness(x, labels)` — the training hot path, `(N, input_len)` × `(N, output_len)`, positive loss |
| `batch_forward` | `Forest.batch_forward(x)` — `(N, input_len)` → `(pop, N, output_len)` |
| `forward` | `Forest.forward(x_per_row)` — one input row per tree, `(pop, output_len)` |
| `gp_step` | one full generation through `GeneticProgramming.step()` (evaluate + select + crossover + mutation), split into eval-attributable and operator-attributable parts in the same step |

## Grid and protocol

`pop_size` ∈ {16, 64, 256, 1000} × `max_tree_len` ∈ {32, 64} × `input_len` ∈ {8, 32}
at `N` = 1024 rows, plus an `N` = 64 column at tree_len 64 / input_len 32 — 20 cells
(`--cells main,n64`, the default).

`--cells long` selects the **long-tree column** instead (`common.long_tree_cells()`,
`max_tree_len` ∈ {128, 256} = `common.LONG_TREE_LENS`), all at `N` = 1024, built by the
same `build_workload` recipe: six cells.

| cell id | `max_tree_len` (row stride) | `max_layer_cnt` (depth cap) |
|---|---|---|
| `p256-L128-i32-N1024` | 128 | 5 |
| `p256-L128-i32-N1024-d6` | 128 | 6 |
| `p256-L128-i32-N1024-d7` | 128 | 7 |
| `p1000-L128-i32-N1024-d7` | 128 | 7 |
| `p256-L256-i32-N1024` | 256 | 5 |
| `p256-L256-i32-N1024-d8` | 256 | 8 |

The column has two axes, and the pair of cells per tree length separates them.
`max_tree_len` (`tree_len`) is the row stride of the tree tensors and the length the
dense route traces: the cap `EVOGP_DENSE_MAX_TREE_LEN` defaults to 64, so at stride
128/256 dense declines the population and the API falls back to per-tree
`specialized` — which is why the column is read as `evoxc-dense-iree` (the fallback),
`evoxc-dense-iree-raised` (dense itself) and the batch arms.
`max_layer_cnt` (`-d<k>` in the id, 5 when absent) is what actually decides how many
nodes the generated trees carry (a full tree is `2 ** d − 1` nodes), so it — not
`tree_len` — governs the evaluation work: with seed 0 / pop 64 the node count grows
with the cap and is identical at stride 32, 64 and 128 for one cap (`d=5` mean 15.8 /
max 29 nodes, `d=6` 27.1 / 57, `d=7` 35.5 / 101, `d=8` 69.9 / 187).
The cap-5 cells are the stock generation with only the row stride made long, and each
`-d<k>` sibling raises the depth cap at that same stride; stride 128/256 with caps up to
8 are legal descriptor settings (`MAX_STACK` = 1024 is no constraint here, and
`2 ** d − 1 ≤ tree_len` holds for every cell above).

* Populations are generated deterministically: `torch.manual_seed(0)` +
  `Forest.random_generate(pop, GenerateDescriptor(...))` with the campaign function
  set `+ - * /` (`common.make_descriptor`, mirroring `src/evogp/sr_test.py`).
* **One process per config** (`worker.py`): the engine's compile cache and
  executable memo live in process memory, so a fresh process is the only way every
  record pays its own cold compile and no arm is helped by another arm's cache.
* **Warm/steady-state medians**: one untimed call first (that is where a cold
  compile lands — recorded as `first_call_ms` and as
  `cache.compile_misses_first_call`, never inside the warm numbers), then
  `--warmup 2` calls, then `--reps 5` sync-bracketed samples → median/min/max/mean
  and the raw samples. `gp_step` uses `--gp-warmup 1 --gp-reps 5`.
* **Sequential, one config at a time**, with an idle-device guard before each config:
  `rocm-smi --showpids` must show no process on the card other than the driver itself
  (`campaign.py` reports the verdict in every log line and keeps the whole guard block
  in its driver record — a contaminated config is reported, never silently averaged
  in).
* **`auto` is timed as the policy resolves it**, never as a forced mode: on this
  device box that is dense+iree from `pop >= 64` and specialized+numpy below the
  threshold (`EVOGP_DENSE_AUTO_MIN_POP`), so the `evoxc-auto` arm measures the dense
  path at pop ≥ 64 and the specialized path at pop 16 (`evoxc-auto-iree` differs only
  there). Each record keeps the decision under `routes.<workload>` (`would_use_dense`
  plus the pop/rows halves).
* `specialized`-iree compiles one graph per *distinct* tree structure at ~272-284 ms
  per structure on this box, so a fresh 256- or 1000-tree population costs minutes of
  compile and a GP step re-pays it every generation: that route is timed on
  `pop ∈ {16, 64}` only (the excluded cells are listed in `campaign.json` under
  `protocol.scope_dropped`) while every other arm covers the whole grid. The `pop 16`
  cell of the shipped default is the cheap specialized+**numpy** route (13
  per-structure misses, 82 ms first call) — the per-structure compile cost belongs to
  iree, not to specialized.
* **The batch arms pay a merged-graph compile instead**: one graph for the whole
  population's structure multiset, so `first_call_ms` is seconds to minutes and
  `compile_misses_first_call` is large. That is the measurement, not an anomaly: the
  same one-untimed-first-call protocol keeps it out of the warm medians, and the record
  carries the merged-graph counters in `batch_cache` (plus per-workload `cache.batch_*`
  deltas and per-generation `batch_compile_misses` / `batch_plan_misses`) next to the
  evoxc ones — `batch_exec_misses` for the exec table (a hit never reaches
  `evoxc.lower`/`compile`/`load`, so a miss is a real recompile), `batch_plan_misses`
  for the plan table, and `batch_cache.stats_keys` for what the route exposes.
* **The batch arms have no `campaign.SCOPE` rule**: `campaign.SCOPE` covers
  `evoxc-specialized-iree` only, and the batch round scheduled its cells explicitly
  instead. Its compile numbers name the path: the single-executable path
  (`EVOGP_BATCH_COMPILE_WORKERS=1`, the control arm) costs 10.9–12.2 s per workload
  at pop 256 and 91.0–162.0 s at pop 1000, plus 102.7 s for a flat, K=1 merge at
  pop 1000, while the shipped default (band size 16, pool 4, no band request) is the
  **chunked** shape at pop 256 and the single-body path at pop 64, where chunking is
  inert below 32 deduped functions. Per-path tables and the band-size history:
  [`BATCH.md`](BATCH.md) §6, §13, §14 and §15,
  [`results/batch_opt_round1/SUMMARY.md`](results/batch_opt_round1/SUMMARY.md),
  [`results/batch_opt_round2/`](results/batch_opt_round2/) and
  [`results/batch_opt_round3/`](results/batch_opt_round3/).
  `--timeout` (default 1800 s per config) bounds a runaway config, and an overrun is
  recorded as `status: "timeout"` with its worker log as evidence, never retried
  away as a defect.

## Final campaign: what was measured

The report of record is [`FINAL.md`](FINAL.md) — headline and before/after tables,
the per-call decomposition, cold-start economics, the 50- and 200-generation
training runs, the noise audit and the caveats. This section is the short version.

* **Grid**: the 20 cells of §Grid above (four populations × two tree lengths × two
  input lengths at `N = 1024`, plus the `N = 64` column), timed on
  `sr_fitness` / `batch_forward` / `forward` and on `gp_step`.
* **Arms**: `torch-per-tree`, `torch-batched`, `evoxc-auto`, `evoxc-auto-iree`,
  `evoxc-dense-iree`, `evoxc-specialized-numpy`, `evoxc-specialized-iree`
  (the last on the `pop ∈ {16, 64}` cells only) — 130 records at HEAD in
  `results/final/after/`.
* **`before`**: the same harness in the same window against the pre-optimization
  snapshot `/tmp/evogp_preopt` (commit `530518e`) through `--tree-root` — 90 records
  in `results/final/before/`, evoxc arms only, because the torch baselines are
  tree-root-independent and their after values serve both sides.
* **Rules**: one process per `(cell, arm)`, one timed config at a time, the
  idle-device guard before every config, warm medians of 5 sync-bracketed reps
  after 2 warmups, and the cold compile always reported separately.

Verdicts in one line each:

* Warm `sr_fitness` at `pop ≥ 64`: **4.9–75× faster than `torch-per-tree` and
  2.1–3.9× faster than `torch-batched`** (1.3–7.8× on `gp_step`), with one real
  loss — 1000 trees × 64 nodes at `i8`/`i32` (`0.86–0.88×` vs `torch-batched`).
* The shipped default is **slower than `torch-per-tree` at pop 16** (0.73–0.81×):
  pop < 64 still routes to specialized, a compile-cost decision rather than a
  runtime one.
* A **cold 50-generation training run is compile-dominated, 7.8× slower than the
  torch evaluator end to end**; with a warm `EVOGP_EVOXC_ARTIFACT_DIR` the same run
  is **1.14× faster end to end** (811.6 vs 921.9 ms).

### Commands of record

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
PY=/path/to/venv/bin/python
cd test/benchmark/evoxc

# 1. after campaign (one process per config, idle-guarded)
$PY campaign.py --out results/final/after --cells main,n64 --arms all

# 2. before campaign against the pre-optimization snapshot
git archive 530518e | tar -x -C /tmp/evogp_preopt
$PY campaign.py --out results/final/before --tree-root /tmp/evogp_preopt --cells main,n64 --arms evoxc-auto,evoxc-auto-iree,evoxc-dense-iree,evoxc-specialized-numpy,evoxc-specialized-iree

# 3. decomposition, validation, cold start, training
$PY decompose.py --cell p256-L64-i32-N1024 --out results/final/decompose/p256-L64-i32-N1024.json
$PY decompose.py --cell p256-L64-i32-N1024 --tree-root /tmp/evogp_preopt --out results/final/decompose-before/p256-L64-i32-N1024.json
$PY validate.py --func-sets arith,rich,ternary --out results/final/validation.json
EVOGP_EVAL_MODE=specialized EVOGP_EVOXC_BACKEND=iree $PY validate.py --out results/final/validation-specialized-iree.json
for pop in 16 64 256 1000; do $PY cold_start.py --cell p${pop}-L64-i32-N1024 --out results/final/cold/p${pop}-L64-i32-N1024.json; done
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_artifact $PY cold_start.py --cell p256-L64-i32-N1024 --out results/final/cold/p256-L64-i32-N1024-artifact.json
$PY training_run.py --arm engine --out results/final/training/engine-200.json
$PY training_run.py --arm torch-per-tree --out results/final/training/torch-per-tree-200.json
$PY training_run.py --arm preopt --tree-root /tmp/evogp_preopt --out results/final/training/preopt-200.json
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_training_art $PY training_run.py --arm engine --note "artifact cache cold (populates the dir)" --out results/final/training/engine-artifact-cold.json
EVOGP_EVOXC_ARTIFACT_DIR=$TMPDIR/evogp_training_art $PY training_run.py --arm engine --note "artifact cache warm reload" --out results/final/training/engine-artifact-warm.json

# 4. batch route round (records go to results/batch/**; nothing under results/final/** or
#    results/baseline/** is re-rendered or rewritten by them). The full command list -- economics,
#    the large-cell and small-cell campaigns, the banded-vs-flat comparison, the two validation
#    sweeps, the gates and the renders -- is BATCH.md §10, and the long-tree round (§11) is
#    BATCH.md §11 "Reproduce"; the economics half is:
python3 batch_economics.py --dry-run --routes batch,dense,batch-flat
$PY batch_economics.py --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p1000-L32-i8-N1024,p1000-L64-i32-N1024 --routes batch,dense --generations 8 --warmup 2 --reps 5 --out-dir results/batch

# 5. render + gate (one side at a time: the cold/training records use another schema)
$PY summarize.py --results results/final/after --out /tmp/after.md
$PY summarize.py --results results/final/after --compare-results results/final/before --compare-label before --out /tmp/compare.md
python3 check.py --results results/final/after  --allow-snapshot --allow-missing --allow-digest-mismatch
python3 check.py --results results/final/before --allow-snapshot --allow-missing --allow-digest-mismatch
```

`results/baseline/` is the earlier round's frozen record set (rendered by
[`SUMMARY.md`](SUMMARY.md)); a new round writes into its own directory, exactly
like `results/final/after|before/` did.

## Cold start and a whole training run

Two questions the per-call grid cannot answer, each with its own script (records under
`results/final/cold/` and `results/final/training/`, which `check.py` does not gate):

* `cold_start.py` — what a user feels: wall time from module import to the first
  synchronized fitness value on the DEFAULT route, on an empty in-memory compile cache.
  One cell and **one population per process** (every cache is per process, so `--pop`
  replaces a loop); the record keeps `wall_to_first_fitness_ms` with its breakdown
  (`import_ms`, `workload_build_ms`, `first_call_ms`), the compile misses around the
  first call, the resolved route, and `--repeat` repeats on the already-warm cache
  (`cold: false`, never averaged with the cold one). `--artifact-dir DIR` turns on the
  opt-in disk layer (`EVOGP_EVOXC_ARTIFACT_DIR`, `src/evogp/tree/artifact_cache.py`) as
  **two child processes** — `populate` (cold artifact write) then `reload` (warm
  artifact) — and a tree without the feature records `artifact_cache: "absent"`.
* `training_run.py` — one complete `StandardPipeline`/`GeneticProgramming` run on the
  in-process `src/evogp/sr_test.py` symbolic-regression problem (synthetic, no dataset,
  no network), one process per `--arm`: `engine`, `torch-per-tree` (the same GP loop and
  operators with `arms_torch.PerTreeEvaluator` as the problem's fitness) and `preopt`
  (the engine arm against a caller-supplied `--tree-root` snapshot). Every generation is
  a sync-bracketed sample, and the full `best_fitness_by_generation` trajectory,
  `best_fitness_at` (gens 10/25/50), `trajectory_digest` and `final_pop_size` are
  recorded next to the timings: a faster arm that converges worse is not a win.

Both drivers use the library's own loop — `StandardPipeline.step()` (what `run()`
executes) for training, the public `Forest.SR_fitness` for cold start — so no private
call path or monkeypatch is timed.

`--tree-root <dir>` (or a `src/` inside it) makes `worker.py` import `evogp` from
that tree and puts it first on the worker's `PYTHONPATH`; the record's `env.tree_root`
and `env.tree_root_sha` say which checkout produced the numbers (`tree_root_sha` is
`null` for a snapshot, which has no `.git` — record the archived SHA next to it).


## Batch route: compile economics (`batch_economics.py`)

The batch route compiles **one merged graph per population** (keyed on the structure
multiset) and a GP generation replaces most of that population, so the question is
whether generation `g + 1` reuses a compiled graph or pays a fresh merge; this script
**measures** the hit rate instead of asserting one. One child process per
`(cell, route)` — every cache is per process — and one record with four parts:
`first_call` (one sync-bracketed `Forest.SR_fitness` on the freshly built population,
expected to be large), `sr_fitness` (warm medians, `--warmup 2 --reps 5`), `generations`
(default 8 real `StandardPipeline.step()` iterations, not warmed up, each row carrying
the population signature snapshotted **before** the step, its timings, both caches'
compile misses and `compile_hit`) and `summary` (derived only from those rows); the
per-field schema is the `econ_child.py` docstring's, the tables [`BATCH.md`](BATCH.md)
§4–§14.

Generation 1 is **expected to hit** (`step()` evaluates `algorithm.forest` before
selecting, so it evaluates the population the timed first call already compiled); the
interesting numbers are generations 2..G, where a fresh population changes the
structure multiset. `dense` runs the identical code path and owns no merged-graph
cache, so its `batch_*` keys are **absent, not null**. Every row also prices the
route's **content-addressed** plan key (`content.py` owns the vocabulary,
`counter_blocks.py` the null-never-zero numbers: a tree without the accessor records
`available: false`, every counter `null` and a note, never a `0`), and
`summary.compile_share` states what it divides (`numerator_kind:
parallel-stage-counter-sum` over `denominator_kind: loop-wall-ms`) and whether it
exceeded 1 (`share_gt_one`), so a share above 1 is the pool's per-chunk compile timers
summed against the loop's wall time, not an accounting error; the blocks are in
`results/batch_opt_round1/content-blocks/`.

Records go to `--out-dir` (default `results/batch/`; `results/final`,
`results/baseline` and `results/batch_opt_round2` are refused) as
`<out-dir>/<cell>/<route>.json` plus one driver record `economics.json`
(`evogp-bench-evoxc-v1-batch-economics`: every config's status, wall time and which
`EVOXC_DISABLE_LOCATIONS` applied). `--dry-run` is stdlib-only and prints every planned
config, one `batch policy ...` line per batch route and a warning when the tree ships no
`src/evogp/tree/batch_engine.py`, so an unaffordable compile is never started to
discover that; an overrun of `--timeout` (default 3600 s) is recorded as `timeout` with
its log as evidence.

A route knob is swept with `--knob NAME=VALUE`, not by registering an arm per value:
it is repeatable, validated against `common.ROUTE_ENV_VARS`, and applied **after**
`arms.apply_arm_env` cleared the route variables and wrote the arm's own spec and
**before** the policy is recorded, so `batch_policy`, `env.applied_knobs` and
`knobs_override` carry the value the engine really read (exporting the variable is not
the way — the arm spec clears it first). That is how the compiled-graph pool is swept
(`EVOGP_BATCH_COMPILE_WORKERS=1`, `4` and `8` as three runs of one cell) with no new
arm.

### Named composition routes

A route token is not just an arm: `ROUTE_SPECS` gives each token its own
`{"arm", "knobs"}` spec, so several batch configurations run in one driver run. On top
of `batch` / `batch-flat` / `dense` (each exactly its registry arm) the knob-spec
routes are `batch-banded` (`EVOGP_BATCH_BANDS=auto`, pool 1), `batch-flat-chunked`
(`EVOGP_BATCH_BANDS=none`, pool 4, `EVOGP_BATCH_CHUNK_BANDS` deliberately **unset** —
the chunk gate refuses an explicit `EVOGP_BATCH_BANDS` spelling without that opt-in, so
it stays single-body despite its pool), `batch-auto-chunked` (`bands=auto` +
`CHUNK_BANDS=1` + pool 4), `batch-b16-banded` (the pool-1 control at band size 16) and
`batch-b16-chunked` (band 16 + `CHUNK_BANDS=1` + pool 4, the shape round 3 cross-checks
against the unpinned default).
`arms.apply_arm_env` **clears** every `common.ROUTE_ENV_VARS` entry
(`EVOGP_BATCH_CHUNK_BANDS` and `EVOGP_DENSE_MAX_TREE_LEN` included) and writes the
arm's spec, then the named route's spec, then an explicit `--knob` (which therefore
wins for the same variable); `protocol.route_knobs` states per variable the value, the
raw `env_value` (`null` = really unset) and the winning layer (`harness-knob` >
`route` > `arm` > `env`, else the engine's `module` resolver / `harness-default` /
`unset`). Each batch record also carries `common.band_report`'s block for the plan(s)
that served it — whether the band clamp fired, and which per-body `chunk_bands` the
merge produced, null (never `0`) when no plan could be described. The child-side loop
internals live in the torch-free `econ_child.py`; `batch_economics.py` stays the
driver, and the commands are in [Commands of record](#commands-of-record) and
[`BATCH.md`](BATCH.md) §10.

**Measured verdict (tables of record: [`BATCH.md`](BATCH.md) §4–§14 and
[`results/batch_opt_round2/`](results/batch_opt_round2/)).** The merged-graph cache
hits only generation 1, so generations 2..G recompile (a `pop 256` population already
spans ~7–8 exec entries at `EVOGP_BATCH_CACHE_SIZE = 8`, so consecutive generations
evict each other by construction), and the route is **dominated by the shipped
`dense+iree` route at every pop ≥ 64** on trees that fit the dense cap — 1.84–3.38×
dense's warm `sr_fitness` at pop 256 and a `gp_step` 193–2355× slower on the round-1
grid — though it beats `specialized+numpy` warm by 8.7–29.6×, and `auto` never selects
it. Its one win on this grid is a **fixed** population of ~64 trees over few data rows
(p64-L64-i32-N64: 0.979 ms vs dense's 1.410 ms, cold 2.17 s vs 6.18 s), which a
generation loop still loses. The loss and per-row paths match the oracle to ~1e-7 rel;
`batch_forward` on the arithmetic set is non-bitwise where dense is exact, which is why
the batch arms form their own digest group.

The band-policy decision and its evidence are [`BATCH.md`](BATCH.md) §14 — band size
16, `EVOGP_BATCH_BANDS=auto` opt-in only, `EVOGP_BATCH_CHUNK_BANDS` off — with the
round-2 band/composition records in
[`results/batch_opt_round2/`](results/batch_opt_round2/). Round 3 re-tests the band
size itself against the rocm binding cap (`k15` / `k7` pins, `BATCH.md` §15.3,
[`results/batch_opt_round3/item3/`](results/batch_opt_round3/item3/)): no pin beat the
shipped size, so the decision stands unchanged.


### Long trees: where the route actually wins

Every cell above holds 15–17-node trees, so none of them asks whether a route prices the trees a population
really contains or the trace length it is allowed to emit. A second round prices that directly in
`results/batch/long-trees/{l128,l256,p1000,gp}` — six `sr_fitness` cells (pop 256 at stride 128 at d5/d6/d7,
pop 256 at stride 256 at d5/d8, pop 1000 at d5) plus two `gp_step` cells, with `evoxc-dense-iree` (the
fallback) against `evoxc-dense-iree-raised` (the same route with `EVOGP_DENSE_MAX_TREE_LEN` pinned to the
cell's stride), `evoxc-batch-iree`, `evoxc-specialized-numpy` and `torch-per-tree`; node counts are
depth-driven, so the `-d<k>` suffix is the axis that makes the trees long, not `tree_len`.

**Verdict in one line:** while the population's longest tree fits the shipped dense cap of 64, need-capped
dense still wins by 1.05–1.41×; past that cap batch wins by 1.5–4.2× over dense with the cap raised and
23.5–25× over the per-tree `specialized` fallback the cap forces — and loses by 190–460× inside a generation
loop, so the win is a fixed-population win only. Raising the cap is itself the cost it removes: 2.030 →
5.277 ms for cap 64 → 128 and 2.126 → 19.497 ms for 64 → 256 on 31-node trees, i.e. dense's per-call cost is
stride-driven and batch's is node-count-driven. Tables, plan facts, the numeric split and the recommendation
are in [`BATCH.md`](BATCH.md) §11.

```bash
PY=/path/to/venv/bin/python
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1

# 1. tree_len 128, generation depth d5/d6/d7 (the node-count axis at a 128-row stride)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/l128 \
  --cells p256-L128-i32-N1024,p256-L128-i32-N1024-d6,p256-L128-i32-N1024-d7 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 2. tree_len 256, generation depth d5/d8 (the stride axis at two strides)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/l256 \
  --cells p256-L256-i32-N1024,p256-L256-i32-N1024-d8 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 3. the population axis at d5
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/p1000 \
  --cells p1000-L128-i32-N1024 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 4. generation level: stride (d5) vs node count (d7), raised-cap dense only
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/batch/long-trees/gp \
  --cells p256-L128-i32-N1024,p256-L128-i32-N1024-d7 \
  --arms torch-per-tree,evoxc-batch-iree,evoxc-dense-iree-raised,evoxc-specialized-numpy \
  --workloads sr_fitness,batch_forward,forward,gp_step --warmup 2 --reps 5 --gp-warmup 1 --gp-reps 5 \
  --timeout 1200

# 5. gate the four directories as one container, then render each
python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/batch/long-trees \
  --allow-missing --allow-digest-mismatch --allow-snapshot
# expected: 4 directories gated, 38 records / 1000 checks, 0 failures
for d in l128 l256 p1000 gp; do
  $PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/batch/long-trees/$d \
    --out test/benchmark/evoxc/results/batch/long-trees/RENDERED-$d.md
done
```

### Plan-key reuse: the rebuilt population (`reuse_probe.py`)

A generation loop never presents the same programs twice, so it cannot say what the
content key buys; `reuse_probe.py` asks the other regime directly — one child process per
column, one cheap cell (`p64-L64-i32-N64`, seed 0, `evoxc-batch-iree`) and a fixed
four-call sequence: `Forest` A cold, A again (the identity fast path), `Forest` **B
rebuilt-but-equal** (a different object built from the same seed, verified
`identical_content` in `no_shared_storage`) against the same `x`/`labels`, then B again —
recording per call the wall time, value digest, both cache counters, the content counters
and the ten stage **call** counts, so "it skipped trace/merge/compile" is measured.

**Result in one line:** on HEAD the rebuilt call is a content hit (1 hit / 0 misses,
`hit_rate` 1.0) in **2.724 ms against the pre-round tree's 79.109 ms (29.0×)** with
trace/merge/compile at **0 calls** and a bit-identical answer (sha256 `b6584500…`), while
in a generation loop the same key buys nothing (6 attempts / 0 hits, because a generation
presents new programs every step). The pre-round column's 76.4 ms is host decode + trace
+ merge — its exec table already shared the compiled body, so it was never a recompile.
Tables, the replay column and the per-stage counts: [`BATCH.md`](BATCH.md) §13.10.

```bash
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
OUT=test/benchmark/evoxc/results/batch_opt_round1
# HEAD, then the pre-round tree at 619cbf7 (unpacked outside the repo), then the replay column
$PY test/benchmark/evoxc/reuse_probe.py --label after
$PY test/benchmark/evoxc/reuse_probe.py --label base --tree-root <checkout of 619cbf7>
$PY test/benchmark/evoxc/reuse_probe.py --label after-replay
# the content blocks of the generation loop, both trees, plus the cell whose pool really chunks
$PY test/benchmark/evoxc/batch_economics.py --cells p64-L64-i32-N64 --routes batch,dense \
  --generations 6 --out-dir $OUT/content-blocks/after
$PY test/benchmark/evoxc/batch_economics.py --cells p64-L64-i32-N64 --routes batch,dense --generations 6 \
  --tree-root <checkout of 619cbf7> --out-dir $OUT/content-blocks/base
$PY test/benchmark/evoxc/batch_economics.py --cells p256-L32-i8-N1024 --routes batch \
  --generations 3 --out-dir $OUT/content-blocks/chunked-p256
```

## Round 3: the refreshed columns, the compile-pool default and the band cap

Round 3 ([`BATCH.md`](BATCH.md) §15, evidence [`results/batch_opt_round3/`](results/batch_opt_round3/), 33 committed JSON files = 26 measurement records + 6 driver records + 1 validation file) re-prices the shipped `EVOGP_EVAL_MODE=batch` default — band size 16, `EVOGP_BATCH_COMPILE_WORKERS` 4, chunked, with **no** `EVOGP_BATCH_*` variable set at all, i.e. the policy itself — in ONE window on the `/tmp/evoxc_r4` evoxc snapshot, against a pool-1 single-body control at the identical band partition (`batch-b16-banded`) and against `dense`, and it settles the compile-pool width and the band size with measured break-evens instead of "as shipped".
**Nothing on the shipped path changes**: the pool stays 4, the band size stays 16, `EVOGP_BATCH_BANDS=auto` and `EVOGP_BATCH_CHUNK_BANDS` keep their round-2 state, `auto` still never resolves to `batch`, and no `src/` file was touched by the round.
The numbers here are the price of the shipped default; rounds 1–2 above stay as the records of those rounds, and this section states where it supersedes them.

**What was measured**

| item | what | driver |
|---|---|---|
| 1 | shipped default, pool-1 control (`batch-b16-banded`) and `dense` at `p256-L32-i8-N1024` / `p256-L64-i32-N1024` / `p64-L64-i32-N64`: cold call, warm `sr_fitness`, an 8-generation `StandardPipeline.step()` loop, the ten-stage cold split | `batch_economics.py --generations 8` |
| 1 cross-check | `batch-b16-chunked` (`EVOGP_BATCH_BAND_SIZE=16` + `EVOGP_BATCH_CHUNK_BANDS=1` + pool 4) at the two pop-256 cells: proves the **unpinned** default resolves to exactly that composition (`chunk_sizes` `[57, 56, 56, 56]` / `[58, 58, 58, 58]`, cold within −0.25 % / −0.03 % of the default) | `batch_economics.py` |
| 2 | the long-tree niche `p256-L128-i32-N1024-d7`, pool 1 vs pool 4 (cold + warm + a 2-generation check, not a loop claim) | `batch_economics.py` |
| 3 | the band-cap pins `k15` at the two pop-256 cells and `k7` at `p64` (15 bands are illegal there: an explicit `K` needs `pop >= 8K`), against the shipped size 16, in pool-1 and pool-4 form, 3 workloads per cell | `band_sweep.py` |
| 4 | pop 1000 on the shipped default (`p1000-L32-i8-N1024`): cold call, one warm reading, 2-generation check | `batch_economics.py` |
| validation | 4 cells × {`arith`, `rich`} against the plain-torch oracle | `validate.py` |

Not re-measured (carried from round 2 with their verdicts intact): `flat`, `b8` / `b32` / `b64`, `EVOGP_BATCH_BANDS=auto` and its cost model, and the long-tree crossover of [`BATCH.md`](BATCH.md) §11 — round 3 prices the pool at the long-tree niche, it does not re-run that round's dense variants.

**The shipped default against dense, one window** (`economics/item1/<cell>/`; warm = `sr_fitness.median_ms`, cold = `first_call.first_call_ms`, loop = `summary.sum_total_ms`):

| cell | warm batch (ms) | warm dense (ms) | batch ÷ dense | cold batch (ms) | cold dense (ms) | batch ÷ dense | 8-gen loop `Σ` batch (ms) | 8-gen loop `Σ` dense (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 3.7250 | 1.1015 | 3.38 | 4 064.48 | 3 355.83 | 1.211 | 19 626.71 | 302.48 |
| `p256-L64-i32-N1024` | 3.7720 | 2.0560 | 1.84 | 4 071.43 | 6 526.78 | 0.624 | 22 346.79 | 312.48 |
| `p64-L64-i32-N64` | 1.1141 | 1.4378 | **0.775** | 2 274.46 | 6 554.88 | **0.347** | 5 587.44 | 313.26 |

Every batch column recompiled **7 of 8 generations** (hit rate 0.125, 8 distinct population signatures in 8 steps); dense compiles nothing and spends 22.28 / 32.42 / 23.11 ms of its loop in evaluation (`summary.sum_eval_ms`).
Cold, batch is 1.211× dearer at `p256-L32` and 0.624× / 0.347× cheaper at `p256-L64` / `p64`, because dense's cold call tracks the trace stride (3 355.83 ms at L32 against 6 526.78 / 6 554.88 ms at L64) rather than the population.
Warm, it is 3.38× / 1.84× dense's call at pop 256 and **1.29× faster** at the fixed-population `p64` niche — the same ordering round 2 found, re-measured in a new window.
The loop `Σ` is the flattering statistic (dense's six warm generations of 2.9–4.2 ms plus one pipeline warm-up, against a fresh merged-graph compile per batch generation): on the eval-only reading dense spends 22.28 / 32.42 / 23.11 ms where a recompiling batch generation alone spends 2 600.93 / 2 894.58 / 588.22 ms (`median_cold_eval_ms`).

**Verdict 1 — `EVOGP_BATCH_COMPILE_WORKERS` stays 4** (break-even = Δcold ÷ Δwarm, i.e. warm calls of ONE population):

| cell | cold pool 1 (ms) | cold pool 4 (ms) | Δcold (ms) | warm pool 1 (ms) | warm pool 4 (ms) | Δwarm (ms) | break-even (median / min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p256-L32-i8-N1024` | 10 725.86 | 4 064.48 | 6 661.38 | 1.9941 | 3.7250 | 1.7309 | **3 849** / 3 981 |
| `p256-L64-i32-N1024` | 11 957.38 | 4 071.43 | 7 885.95 | 2.2267 | 3.7720 | 1.5453 | **5 103** / 5 083 |
| `p64-L64-i32-N64` | 2 268.96 | 2 274.46 | −5.51 | 1.1494 | 1.1141 | −0.0353 | **n.a. — no trade** |
| `p256-L128-i32-N1024-d7` | 38 513.95 | 8 410.96 | 30 102.99 | 3.4851 | 5.0947 | 1.6096 | **18 702** / 20 648 |

The pool buys 2.64× / 2.94× / 4.58× on the cold call and costs 1.69–1.87× on the warm one (1.46× at the long-tree niche), so repaying it needs 3.8k / 5.1k / 18.7k warm calls on a **single** structure multiset — which a GP loop never makes.
At `p64` the pool is inert (`chunk_count 1`, `chunk_sizes [63]`, Δcold −5.51 ms), and `p64` is exactly where batch beats dense warm, so flipping the default to 1 would buy nothing where the route wins while making the cold call 10.7 s (pop 256) / 38.5 s (long trees) instead of 4.1 s / 8.4 s.
Escape hatch, documented and not a new default: a workload that evaluates one fixed multiset more than ≈3.8k / 5.1k / 18.7k times sets `EVOGP_BATCH_COMPILE_WORKERS=1` explicitly and takes the cheaper warm call — `src/evogp/tree/batch_knobs.py::DEFAULT_COMPILE_WORKERS` stays 4.

**Verdict 2 — `EVOGP_BATCH_BAND_SIZE` stays 16** (the pin at the rocm binding cap against the shipped size, `sr_fitness` / `batch_forward` / `forward`, pool 1):

| cell | pin vs `b16` | plan change? |
|---|---|---|
| `p256-L32-i8-N1024` | −5.9 % / +2.3 % / −6.9 % | **no** — `[15]×15` and the same `emitted_ops` 5 037 / 4 153 / 4 540 |
| `p256-L64-i32-N1024` | +0.1 % / +1.2 % / −2.6 % | **no** — `[16]×7 + [15]×8`, identical ops 5 475 / 4 568 / 5 861 |
| `p64-L64-i32-N64` | +0.04 % / +1.3 % / −4.6 % | **yes** — `K` 4 `[16,16,16,15]` → `K` 7 `[9]×7`, ops 1 393 → 1 463 |

At pop 256 the pin is a **same-merge control, not a different plan**: identical `band_sizes` and identical `emitted_ops`, the only difference being that the size model asks 16 bands and is clamped to 15 on the per-row plan while `k15` asks the count (`band_requested explicit`, `band_clamped false`).
At `p64` it does change the plan and still lands inside this box's ~5 % cross-config noise floor; the pool-4 pairs sit at −3.6…+3.8 % with one −15.1 % `forward` row on an *identical* plan (the band-sensitive column §14.1 already flags).
No pin beat the shipped size anywhere in the window, and the oracle deviation is **identical in all four configs of every cell/workload** (9/9 rows), so the band partition is scheduling, not numerics.

**Verdict 3 — pop 1000 is priced on the shipped default** (new information, not a speedup measurement): cold **15 517.88 ms** (compile wall 13 921.97 ms over 4 jobs is 89.72 % of the call, `lower` 1 054.29 ms is 6.79 %), warm `sr_fitness` **6.9384 ms**, plan `chunk_sizes [205, 204, 204, 204]` for 817 deduped functions.
The 91.0–162.0 s figure above at pop 1000 is the **single-executable** (`EVOGP_BATCH_COMPILE_WORKERS=1`) path; it is not the shipped default's price, and round 2 never priced the chunked composition there.

**What the three upstream evoxc speedups did** — same-shape pairs, the before column being round 2's records in [`results/batch_opt_round2/`](results/batch_opt_round2/):

| cell / route | cold before → now (ms) | Δ cold | `lower` before → now (ms) | `lower` speedup |
|---|---:|---:|---:|---:|
| `p256-L32` pool 1 | 11 049.41 → 10 725.86 | −2.93 % | 296.61 → 98.83 | **3.00×** |
| `p256-L32` default (chunked) | 4 180.63 → 4 064.48 | −2.78 % | 888.52 → 470.05 | **1.89×** |
| `p256-L64` pool 1 | 12 590.61 → 11 957.38 | −5.03 % | 286.68 → 97.82 | 2.93× |
| `p256-L64` default | 4 422.39 → 4 071.43 | −7.94 % | 985.88 → 376.02 | 2.62× |
| `p64` pool 1 | 2 298.69 → 2 268.96 | −1.29 % | 63.50 → 31.66 | 2.01× |
| `p64` default (inert) | 2 284.23 → 2 274.46 | −0.43 % | 63.38 → 31.16 | 2.03× |

`lower()` is **1.9–3.0× faster**, but that moves the cold call by only **0.4–7.9 %**, because the device-code compile inside iree is **89–96 %** of it — 95.7–95.8 % of a pool-1 cold call at pop 256, 89.1–89.3 % chunked, 91.3 % at `p64`, 97.4 % pool 1 at the long-tree cell, 89.7 % at pop 1000 — while `lower` itself is 0.8–0.9 % of a pool-1 call and 9.2–11.6 % of a chunked one.
The upstream speedups help this route's cold path and cannot fix it: the cost is not in the MLIR construction.

**Commands (round 3)** — verbatim as run, from the repository root, one driver invocation per item (the `--note` strings are the records' own):

```bash
export EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r4:src:test/benchmark/evoxc
PY=/tmp/rocvenv/bin/python

# item 1 — shipped default vs pool-1 control vs dense, one window, 8 generations
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64 \
  --routes batch,batch-b16-banded,dense \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/item1 \
  --note "round3 item1: shipped default (band 16, pool 4, chunked) vs pool-1 control (batch-b16-banded) vs dense, same window, evoxc_r4"

# item 1 cross-check — the pinned band16+CHUNK_BANDS=1+pool4 shape against the unpinned default
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 \
  --routes batch-b16-chunked \
  --generations 8 --warmup 2 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/economics/crosscheck-b16-chunked \
  --note "round3 item1 cross-check: pinned band16+CHUNK_BANDS=1+pool4 against the unpinned shipped default"

# item 2 — the long-tree (d7) pool trade, warm + cold + a 2-generation check
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p256-L128-i32-N1024-d7 \
  --routes batch,batch-b16-banded \
  --generations 2 --warmup 2 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item2/long-tree \
  --note "round3 item2: long-tree niche (d7), pool-1 control vs the chunked default, warm+cold, 2-generation loop (not a loop claim)"

# item 3 — the band cap at pop 256: k15 vs the shipped size 16, pool 1 and pool 4
$PY test/benchmark/evoxc/band_sweep.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024 \
  --configs b16,b16-cw4,k15,k15-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p256 \
  --note "round3 item3: K=15 pinned at the rocm binding cap vs the shipped band size 16, pool-1 partitions and pool-4 compositions, same window"

# item 3 — the same at the fixed-population niche cell, where K=15 is illegal and k7 is the cap
$PY test/benchmark/evoxc/band_sweep.py \
  --cells p64-L64-i32-N64 \
  --configs b16,b16-cw4,k7,k7-cw4 \
  --workloads sr_fitness,batch_forward,forward --warmup 1 --reps 5 --seed 0 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item3/p64 \
  --note "round3 item3: K=7 pinned vs band size 16 at the fixed-population niche cell, pool-1 and pool-4, same window"

# item 4 — pop 1000, shipped default only
$PY test/benchmark/evoxc/batch_economics.py \
  --cells p1000-L32-i8-N1024 --routes batch \
  --generations 2 --warmup 1 --reps 5 --seed 0 --timeout 1500 \
  --tree-root src --evoxc-root /tmp/evoxc_r4 \
  --out-dir test/benchmark/evoxc/results/batch_opt_round3/item4/p1000-L32-i8-N1024 \
  --note "round3 item4: p1000-L32-i8-N1024 shipped default only, cold + one warm reading + stage split + 2-generation loop, faster evoxc lower()"

# the round's oracle sweep (arith + rich, four cells)
EVOGP_EVAL_MODE=batch $PY test/benchmark/evoxc/validate.py \
  --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64,p1000-L32-i8-N1024 \
  --func-sets arith,rich --max-oracle 64 --seed 0 \
  --out test/benchmark/evoxc/results/batch_opt_round3/validation/validation-batch.json \
  --note "round3: the batch route (shipped default knobs, evoxc_r4) vs the plain-torch oracle on the item-1/item-4 cells"
```

A re-measure always writes a **fresh `--out-dir`**: both drivers refuse `results/final`, `results/baseline` and `results/batch_opt_round2` (`FROZEN_RESULT_DIRS`), and `band_sweep.py` additionally refuses `results/batch` and `results/batch_opt_round1`, so round-3 runs can never overwrite earlier evidence.
`batch_economics.py` defaults `--evoxc-root` to `/root/evoxc`, which is why the runs above pass `--evoxc-root /tmp/evoxc_r4` explicitly; `band_sweep.py` already defaults to it.

**Caveat, inherited not fixed:** `batch_economics.py` has no idle-device guard (its records carry no `guard` block — idleness was hand-checked with `rocm-smi --showpids` and the runs kept strictly sequential, 14 children, while every `band_sweep.py` record does carry `guard.state.gpu_pids.idle true` / `contaminated false`), and neither an economics nor a band-sweep directory is a `check.py`-gateable unit, because those records use the economics/band-sweep schemas rather than the campaign arm-record schema (the gate reports the missing `workloads`/`cache` fields and `campaign.json`, exactly as it does on round 2's evidence).

## Report and gate (`summarize.py` / `check.py`)

`summarize.py --results DIR --out FILE.md` renders one document from the records:
protocol statement, environment, one warm table + one speedup table per workload,
the `gp_step` section, the cold-compile exclusion, digest agreement and caveats.
Every table cell is a field of a record; a missing record or a contaminated guard
stays visible instead of being averaged away.
The final campaign's three renders live in `results/final/RENDERED.md`, interpreted by
[`FINAL.md`](FINAL.md); the batch round's are `results/batch/RENDERED-*.md` plus
`results/batch/long-trees/RENDERED-{l128,l256,p1000,gp}.md`, interpreted by
[`BATCH.md`](BATCH.md).

A batch arm renders by the same rules: its `routes[...]` carries `kind: "batch"`, so
the `auto route` column quotes `batch/iree → batch` (annotated `bands=flat` when the
merge policy differs from the default), and one `Batch route arms: ...` caption names
the resolved knobs under each workload table. Both appear **only** in a directory that
recorded a batch arm, so a batch-free render stays byte-identical, and `## Caveats`
gains one bullet: a batch arm's `first_call_ms` is a merged-graph compile for the whole
population, excluded from the warm medians by protocol.

* `gp_step` is read from `median_total_ms` / `median_eval_ms` / `median_ops_ms`, and
  the eval/ops split carries the operator share of the generation (`X.XX / Y.YY (op
  ZZ%)`). A record with `steps[]` but not those aggregate keys is read from the upper
  median of `steps[].total_ms` / `steps[].eval_ms` -- the statistic `worker.py`
  stores -- and marked `*`; a raw-steps record never blanks a cell. The section also
  renders the per-cell `gp_step` speedup against both torch arms and the timed
  pipeline's `best_fitness`.
* `--compare-results DIR --compare-label NAME` (default `before`) adds one table per
  workload plus one for the `gp_step` total: the primary directory's median over the
  compare directory's, per cell and arm. Only cells present in both are listed; an arm
  the primary side lacks is quoted as `- (before <ms>)`, and without the flag the
  render is byte-identical to the single-directory one.

`check.py --results DIR` enforces the record contract (schema, provenance, >= 5 warm
samples with `len(samples_ms) == reps`, `first_call_ms > 0`, a re-derived median,
evoxc route + compile-miss fields, cross-arm digest agreement, `gp_step` sanity
including `median_total_ms` == the upper median of `steps[].total_ms`, guard
honesty) and exits 1 on any failure. It gates ONE campaign directory, so a
before/after round is gated one side at a time (`--results results/final/after`,
then `--results results/final/before`): `results/final/` also holds the
`cold/`, `training/` and `decompose*/` records, whose different schema the gate
rejects.

* `--allow-snapshot` turns a null `env.tree_root_sha` into a warning, but only when the
  recorded `env.tree_root` really has no `.git` (a `git archive` export has no HEAD, so
  the archived SHA is quoted next to the numbers).
* `--allow-missing` and `--allow-digest-mismatch` declare the two expected
  deviations: the 10 `evoxc-specialized-iree` cells the compile-cost scope rule never
  schedules at pop ≥ 256, and the numpy-vs-iree sha256 split on
  `forward`/`batch_forward`. Without them the same records report those as failures
  (38 on the after side, 39 on `results/baseline`); with them the gates read 130
  records / 3602 checks and 90 records / 2642 checks, 0 failures.
* `check_batch()` runs on every batch-arm record and checks provenance, not
  magnitude: it **fails** when `batch_policy.mode_supported` is not true, when the
  recorded route is not `batch` (or has no `kind: "batch"`), or when `batch_cache`
  has no boolean `available`; it only **warns** when the merged-graph compile is not
  attributed to any cache counter (0 first-call misses across both caches —
  `first_call_ms` still carries the cost and a miss count is never bounded by the
  gate) and when `batch_cache.stats_keys` carries no `batch_exec_misses` (the exec
  table, the only one whose miss recompiles). Expected-arm coverage defaults to the
  campaign's own scheduled arms, so appending an arm leaves previously
  recorded campaigns gating exactly as before.
* `check_content()` is the same self-consistency check over the content blocks
  (`content.py`) and **warns** only: a block claiming `available: true` with every
  content counter null, or `available: false` with counters carrying numbers, is a
  contradiction worth printing — but a tree without `batch_content_stats()`
  legitimately records nulls, and a hit rate of 0 is a measurement. It makes no
  `checker.check` call, so the check count of a directory of frozen records (which
  carry no content keys) is exactly what it was.
* The gate still demands the `gp_step` aggregate keys: the renderer's `steps[]`
  fallback is a safety net, not a licence for a record to drop them.

## Numeric validation (`validate.py`)

The oracle is `src/evogp/tests/reference_eval.py`, the plain-torch reference the
pytest suite diffs against. For every validation cell and function set the same
population is evaluated by the oracle (CPU), by both torch arms and by the engine (the
route `EVOGP_EVAL_MODE`/`EVOGP_EVOXC_BACKEND` select), and every arm's deviation is
reported against the oracle *and* the engine: max abs, max rel, NaN/Inf placement
mismatches and a bitwise-equality flag (`pop` is sliced to `--max-oracle` trees for the
CPU oracle; the record says how many were compared).

Three function sets, because they mean different things:

| set | contents | what agreement to expect |
|---|---|---|
| `arith` | `+ - * /` (the campaign set) | float32 arithmetic is correctly rounded, so every arm must agree **bitwise** on the value path |
| `rich` | `arith` + `max min < loose_div loose_inv sin exp log abs sqrt` | transcendental float32 kernels differ legitimately — `1e-8`–`1e-5` abs on the torch path, up to `~1e-1` abs in the tail of `batch_forward` on the dense route |
| `ternary` | `if + - * /` (needs a shorter-layer descriptor: the descriptor asserts `max_tree_len >=` the fully-branching size) | value path bitwise; the `if` select is where a naive implementation gets the operand order wrong |

Measured on the final campaign (12 cases over the four scheduled cells; exact table in
`VALIDATION.md`): the value path (`batch_forward`, `forward`) is **bit-identical**
across the oracle, both torch arms and the engine on `arith` and `ternary` (max abs
`0.000e+00`), and `sr_fitness` agrees to `~3.5e-7` relative — float32 reduction order
(the engine reduces on the host in numpy, the arms in torch), not a semantic
difference. `rich` is the one set with real spread: `~9.3e-2` abs / `~1.5e-4` rel for
the torch arms and up to `4.4e-1` abs / `6.4e-1` rel for the dense route on
`batch_forward` — the dense/IREE lowering of transcendentals, not a wrong tree
evaluation.

Both torch arms were debugged this way; two classic mistakes to know before touching
them: the reversed-prefix walk pushes the **right-most** child first, so `stack.pop()`
order is left-to-right (`a`, then `b`), and child positions are **per row**
(`c1 = i+1`, `c2 = c1 + size[row, c1]`, `c3 = c2 + size[row, c2]`), so row 0's sizes
silently mis-wire every tree of a different shape.

## Decomposition (`decompose.py`)

One representative cell (`p256-L64-i32-N1024`) on both routes: module-level
inclusive `perf_counter_ns` phase timers around the engine's own functions
(`decode_population`, `_executable`, `_run_tree`, `_evaluate_rows`, `plan_dense`,
`build_dense`, `run_loss_plan`, …), the residual between the workload and its
instrumented children printed as its own row (that residual is the host glue: loss
assembly, conversions, uploads), and seam counters for `evoxc.run` dispatches,
`evoxc.Tensor.to` uploads, `Tensor.numpy` readbacks and `from_numpy` wraps. The
uninstrumented median of the same call sits next to the instrumented one so the timer
overhead is visible, and cold compile is reported separately.

## Rules for this subtree

* Never edit `../r2/`, `../r4/`, `../r5/`, `../r6/` records, reports or scripts to
  make them agree with a new measurement — they are frozen evidence from another
  question (legacy CUDA engine vs evoxc engine).
* One process per config; never run two timed configs concurrently, and never run
  anything else that imports torch while a campaign is in flight.
* A first-call compile inside the warm samples is a protocol violation, not a warm
  number: `worker.py` pays it before the warmup loop and `check.py` re-derives the
  medians from `samples_ms` to catch a record that does not.
* A batch arm's large `first_call_ms` and `compile_misses_first_call` are the
  recorded cold cost of one merged graph per population — never an anomaly to be
  smoothed, retried or scoped away; `check.py` therefore only **warns** when that
  compile is not attributed to any counter, and `batch_cache.available: false` is a
  reportable state, not a defect.
* Numerical honesty over tidiness: if a cell looks contaminated, re-run it and say
  so in the record (`--note`) instead of dropping or smoothing the number.
* `results/baseline/` and `results/final/**` are frozen evidence: a re-run writes
  into a fresh result directory and is gated there, never in place.
