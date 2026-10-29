# `test/benchmark/evoxc/` — evoxc engine vs plain-PyTorch, controlled harness

The question this subtree answers: **how much does the evoxc-based EvoGP engine cost
per evaluation compared with the plain-PyTorch implementation of the same tree
evaluation?** — on identical populations, in the same session, warm/steady-state,
with every arm validated numerically against the repo's independent oracle.

It is deliberately separate from `../r2/`, `../r4/`, `../r5/`, `../r6/`: those
rounds compare the evoxc engine against the *deleted legacy CUDA engine* and their
committed numbers stay as history. This subtree never touches them.

The current evidence is the post-optimization campaign in `results/final/` (HEAD
vs the pre-optimization snapshot `530518e`, same window, same harness) plus its
cold-start and whole-training-run columns; [`FINAL.md`](FINAL.md) is the report of
record and [`SUMMARY.md`](SUMMARY.md) is the rendered earlier baseline round.
A second round prices the opt-in `batch` route on a fixed population and inside a
real `StandardPipeline.step()` loop — records in `results/batch/`, report of record
[`BATCH.md`](BATCH.md).

```
common.py       workload construction (one definition shared by every arm), cell
                grid, env block, warm-timing helper, route inspection
arms_torch.py   the two baselines: PerTreeEvaluator (Python tree walk, torch ops
                per node) and BatchedEvaluator (one vectorized pass per population)
arms.py         the arm namespace: TorchArm / EvoxcArm wrappers, env application,
                SR loss problem + GP pipeline construction
worker.py       ONE process, ONE (cell, arm): the measured unit
campaign.py     sequential driver: one subprocess per config, idle-device guard,
                `--tree-root` switch, per-config logs + `campaign.json`
validate.py     numeric validation of every arm against reference_eval.py
decompose.py    per-call phase breakdown of one cell on both routes
summarize.py    renders the records into one markdown report (stdlib only; gp_step
                falls back to steps[], before/after via --compare-results)
check.py        the gate over one campaign result directory (stdlib only, exit 1
                on failure); a before/after round is gated one side at a time
cold_start.py   cold-start column: import -> first fitness on the default route,
                one process per population; --artifact-dir = populate + reload
training_run.py end-to-end training run: per-generation wall ms + best-fitness
                trajectory for the engine, the plain-torch evaluator and a snapshot
batch_economics.py batch-route compile economics: per-generation structure
                signature, merged-graph compile misses and hit rate for `batch`
                vs `dense` over a real StandardPipeline.step() loop
results/        the evidence: baseline/ (frozen earlier round), final/ (after/,
                before/, decompose/, cold/, training/, validation.json, RENDERED.md)
                and batch/ (warm-r5, small-r5, flat-r5, flat1000-r5, flat1000-3w-r5,
                <cell>/<route>.json, economics.json, validation-*.json,
                RENDERED-*.md; plus the --reps 3 pilot dirs warm/, small/, flat/,
                flat1000/)
FINAL.md        report of record for the post-optimization dense/specialized
                campaign (tables, gate, reproduce commands)
BATCH.md        report of record for the batch route round (warm per-call and
                gp_step tables, compile economics and break-even, the one winning
                regime, validation, gates, reproduce commands)
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

Both torch arms are *implemented in this harness* (`arms_torch.py`), op semantics
copied from the oracle: `DIV` → NaN on a zero divisor, `loose_div`/`loose_inv`
clamp the divisor with `copysign(DELTA, y)`, `max`/`min` via `torch.where`,
comparisons return ±1.0, `if` selects on `cond > 0`. Their decode/plan cost is
built once outside the timed region and reported as `setup_ms`.

The two `evoxc-batch-*` arms are the harness's handle on the engine's **third
evaluation route** (`EVOGP_EVAL_MODE=batch`, `src/evogp/tree/batch_engine.py`
through `evoxc.batch_functions`): instead of one graph per population *shape*
(`dense`) or per tree *structure* (`specialized`), it compiles **one merged graph
per population**, keyed on the multiset of tree structures. The `auto` policy
resolves to `dense` or `specialized` only, never to this route, so both arms pin the
mode; iree is the backend and numpy the fallback, exactly as on the dense route.
Neither arm is a default-route claim (`auto` never resolves to `batch`); the measured verdict is in
[`BATCH.md`](BATCH.md) and
[Batch route: compile economics](#batch-route-compile-economics-batch_economicspy).

Its knobs are all `EVOGP_BATCH_*`, single-sourced in `common.BATCH_KNOBS` and
resolved in a record as arm value > inherited environment > the engine's own
resolver > the documented default (`batch_policy.knobs_source` says which applied):

| knob | documented default | meaning |
|---|---|---|
| `EVOGP_BATCH_BAND_SIZE` | `32` | programs per band: the population is merged one band of trees at a time |
| `EVOGP_BATCH_BANDS` | engine default (unset) | an explicit band count `K`, or `none`/`flat` for one unbanded merge (O(N²) in the population — what the `-flat` arm times) |
| `EVOGP_BATCH_SWITCH_OPS` | `off` | whether the merged graph may use switch ops (read from the engine's own resolver) |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | band variants are aligned before the merge instead of being padded per call |
| `EVOGP_BATCH_CACHE_SIZE` | unset (engine default) | the merged-graph cache size; the harness records the resolved value and never sets it |
| `EVOGP_BATCH_MIN_POP` | `64` | population floor: below it one merged graph cannot amortize its build, so the route declines to `specialized` (the record's `routes.*.decline_reason` carries the sentence) |

An arm spec may carry a `knobs` mapping (`{env var: value}`) beyond the two route
selectors; `-flat` is exactly `{"EVOGP_BATCH_BANDS": "flat"}`. `arms.apply_arm_env`
clears **every** `EVOGP_BATCH_*` first for every arm (torch arms included), so no
value can leak from one arm's process environment into another's, and then writes
the arm's own knobs; an unregistered arm name still raises `KeyError`. `--arms all`
covers all nine arms and `--arms batch` selects exactly these two, in registry
order (they are registered **last**, so no existing order or render changes).

Both arms need a tree that ships the module *and* declares `batch` in
`dense_engine.EVAL_MODES`; where it does not, the worker's preflight fails the
config with one readable paragraph and writes **no** record (the dry runs below are
the torch-free way to learn that before paying a compile).

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
at `N` = 1024 rows, plus an `N` = 64 column at tree_len 64 / input_len 32 — 20 cells.

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
* **Sequential, one config at a time**, with an idle-device guard before each
  config: `rocm-smi --showpids` must show no process on the card other than the
  driver itself (`campaign.py` reports the guard verdict in every log line, and the
  driver record keeps the whole guard block — a contaminated config is reported,
  never silently averaged in).
* **`auto` is timed as the policy resolves it**, never as a forced mode: on this
  device box that is dense+iree from `pop >= 64` and specialized+numpy below the
  threshold (`EVOGP_DENSE_AUTO_MIN_POP`), so the `evoxc-auto` arm measures the dense
  path at pop ≥ 64 and the specialized path at pop 16, and `evoxc-auto-iree` differs
  from it only at pop 16 (specialized+iree). Each record keeps the decision under
  `routes.<workload>` (`would_use_dense` plus the pop/rows halves).
* `specialized`-iree compiles one graph per *distinct* tree structure and a
  measured iree compile costs ~272-284 ms per structure on this box, which makes a
  fresh 256- or 1000-tree population cost minutes of compile and a GP step re-pay it
  every generation. That route is therefore timed on `pop ∈ {16, 64}` only; the
  excluded cells are listed in `campaign.json` under `protocol.scope_dropped`.
  Every other arm covers the whole grid. (The `pop 16` cell of the shipped default
  is the cheap specialized+**numpy** route: 13 per-structure misses, 82 ms first
  call — the per-structure compile cost belongs to iree, not to specialized.)
* **The batch arms pay a merged-graph compile instead**: one graph for the whole
  population's structure multiset, so their `first_call_ms` is seconds to minutes
  and their `compile_misses_first_call` is large. That is the measurement, not an
  anomaly: the same one-untimed-first-call protocol keeps it out of the warm
  medians, and the record carries the merged-graph counters in `batch_cache` (plus
  the per-workload `cache.batch_*` deltas and the per-generation `batch_compile_misses` /
  `batch_plan_misses`) next to the evoxc ones. The compile counters are the ones
  `batch_cache_stats()` actually exposes: `batch_exec_misses` is the exec table (a hit
  never reaches `evoxc.lower`/`compile`/`load`, so a miss is a real recompile) and
  `batch_plan_misses` is the plan table (re-decode + re-trace + re-merge, no compile);
  `batch_cache.stats_keys` records which of them the route exposes.
  `forest_stats.structure_signature` names the population the graph was keyed on.
* **The batch arms have no `campaign.SCOPE` rule**: `campaign.SCOPE` covers
  `evoxc-specialized-iree` only, and the batch round scheduled its cells explicitly
  instead. The compile it measured is 2.2–2.9 s per workload at pop 64, 9.9–12.2 s
  at pop 256 and 91–169 s at pop 1000 (plus 102.7 s for a flat, K=1 merge at pop
  1000), i.e. exactly the numbers a scoping rule would be built from — see
  [`BATCH.md`](BATCH.md) §6. `--timeout` (default 1800 s per
  config) bounds a runaway config, and a batch-arm config that exceeds it is
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
  snapshot `/tmp/evogp_preopt` (commit `530518e`) through `--tree-root` — 90
  records in `results/final/before/`, evoxc arms only, because the torch baselines
  are tree-root-independent and their after values serve both sides.
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
#    sweeps, the gates and the renders -- is BATCH.md §10; the economics half is:
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

Two questions the per-call grid above cannot answer, each measured by its own
script (records under `results/final/cold/` and `results/final/training/`, which
`check.py` does not gate):

* `cold_start.py` — what a user feels: wall time from module import to the first
  synchronized fitness value on the DEFAULT route, on an empty in-memory compile
  cache. One cell and **one population per process** (the compile cache and
  executable memo are per process, so `--pop` replaces a loop); the record keeps
  `wall_to_first_fitness_ms` plus its visible breakdown (`import_ms`,
  `workload_build_ms`, `first_call_ms`), `evoxc_engine.cache_stats()` compile
  misses around the first call, the resolved route, and `--repeat` repeats that
  run on the already-warm process cache (`cold: false`, never averaged with the
  cold one). `--artifact-dir DIR` turns on the opt-in disk layer
  (`EVOGP_EVOXC_ARTIFACT_DIR`, `src/evogp/tree/artifact_cache.py`) and drives it
  as **two child processes** — `populate` (cold artifact write) then `reload`
  (warm artifact) — recording the directory's file count/bytes before and after;
  its record's `wall_to_first_fitness_ms` is a per-phase dict, and a tree without
  the feature (the pre-optimization snapshot) records `artifact_cache: "absent"`
  and skips that half.
* `training_run.py` — one complete `StandardPipeline`/`GeneticProgramming`
  training run on the in-process `src/evogp/sr_test.py` symbolic-regression
  problem (synthetic, no dataset, no network), one process per `--arm`:
  `engine` (the shipped pipeline and route), `torch-per-tree` (the same GP loop
  and operators with `arms_torch.PerTreeEvaluator` substituted as the problem's
  fitness) and `preopt` (the engine arm against a caller-supplied `--tree-root`
  snapshot). Every generation is a sync-bracketed sample and the full
  `best_fitness_by_generation` trajectory, `best_fitness_at` (gens 10/25/50),
  `trajectory_digest` and `final_pop_size` are recorded next to the timings:
  a faster arm that converges worse is not a win.

The driver in both cases is the library's own loop — `StandardPipeline.step()`
(what `run()` executes) for training, the public `Forest.SR_fitness` for cold
start — so no private call path or monkeypatch is being timed.

`--tree-root <dir>` (or a `src/` inside it) makes `worker.py` import `evogp` from
that tree and puts it first on the worker's `PYTHONPATH`; the record's `env.tree_root`
and `env.tree_root_sha` say which checkout produced the numbers (`tree_root_sha` is
`null` for a snapshot, which has no `.git` — record the archived SHA next to it).

## Batch route: compile economics (`batch_economics.py`)

The batch route raises a question no other route asks: it compiles **one merged
graph per population**, keyed on the population's structure multiset, and a GP
generation replaces most of that population. So does generation `g + 1` reuse a
graph this process already compiled, or does it pay a fresh merged-graph compile
every generation? If it recompiles every time, the compile eats the route's
warm-runtime advantage over `dense`; if elites and duplicates hold the multiset
stable, the merged-graph cache hits and the route wins. Nobody knows which happens
on a real run, so this script **measures** the hit rate instead of asserting one.

One child process per `(cell, route)` — the evoxc compile cache, the executable
memo and the merged-graph cache all live in process memory, so a route loop inside
one process would measure warm caches only. Each record has four parts:

* `first_call` — ONE sync-bracketed `Forest.SR_fitness` on the freshly built
  population: the merged-graph compile (plus the cold host decode) for `batch`, the
  dense-graph compile for `dense`, with both caches' counters around it and the
  value digest. Expected to be large; reading it as a defect is the mistake this
  script exists to prevent.
* `sr_fitness` — the warm/steady-state medians of the same call on the same
  population (`--warmup 2 --reps 5`, via `common.time_call`), with the compile
  misses of **both** caches around the untimed call. Its `median_ms` is the warm
  number the per-call grid would report.
* `generations` — `--generations` (default 8) iterations of a real
  `StandardPipeline.step()` loop, deliberately **not** warmed up, so generation 1 is
  the honest first generation. Before each step the population's structure-multiset
  signature is snapshotted (that IS the population the step is about to evaluate),
  and each row keeps `population_signature`, `distinct_structures`,
  `signature_seen_before`, total/eval/ops ms, the compile misses of both caches,
  `compile_hit` and that generation's `best_fitness`.
* `summary` — derived only from those rows, never asserted:
  `generation_hit_rate` (with `generations_unknown` next to it, for a route whose
  counters cannot judge a row), `generations_with_compile`,
  `distinct_population_signatures`, `median_warm_eval_ms` vs `median_cold_eval_ms`
  (eval ms of the hit rows vs the recompiling rows), `first_call_ms`,
  `warm_median_ms`, and `compile_ms_share` — the wall time of the recompiling
  generations over the loop's total wall, i.e. an upper bound on "how much of the
  loop the compile generations own", not a compile timer. The merged-graph cache's
  own `compile_misses_first_call` / `_generations` / `_total` ride along.

Generation 1 is **expected to hit**: `step()` evaluates `algorithm.forest` before
selecting, so generation 1 evaluates the initial population — the very population
whose graph the timed first call already compiled. That is recorded as measured
(not warmed away), and the interesting numbers are generations 2..G, where a fresh
population changes the structure multiset. `dense` goes through the identical code
path as the comparison column: its per-generation compile counter is the evoxc
engine's own, and it owns no merged-graph cache at all, so its `batch_*` keys are
**absent, not null**.

Records go to `--out-dir` (default `results/batch/`; `results/final` and
`results/baseline` are refused) as `<out-dir>/<cell>/<route>.json` — default cells
`p256-L32-i8-N1024,p1000-L32-i8-N1024,p1000-L64-i32-N1024`, default routes
`batch,dense`, plus the `batch-flat` token for the unbanded merge — with the child's
`.log` next to it and one driver record `<out-dir>/economics.json` (schema
`evogp-bench-evoxc-v1-batch-economics`) listing every config's status and wall time,
the protocol, and which `EVOXC_DISABLE_LOCATIONS` applied (the driver defaults it to
`1` when the caller left it unset, and records `caller` vs `driver-default`).
`--dry-run` is stdlib-only (no torch, no `env_block`): it prints every planned
config, one `batch policy ...` line per batch route with the resolved knobs, and a
warning when the tree under measurement does not ship
`src/evogp/tree/batch_engine.py` (or ships it without declaring the mode), so an
unaffordable compile is never started to discover that.
Like every other timed column it wants an idle GPU and one config at a time; a
config that overruns `--timeout` (default 3600 s) is recorded as `timeout` with its
log as evidence.

The batched commands are in [Commands of record](#commands-of-record) below (dry run, plus the
default `batch,dense` column) and, in full, in [`BATCH.md`](BATCH.md) §10.

**Measured verdict (see [`BATCH.md`](BATCH.md) for the tables).** On a real generation loop the
question is answered against the route: the merged-graph cache hits only generation 1 (which re-evaluates
the population the timed first call already compiled), so generations 2..G recompile — hit rate 1/8 at
pop 256 and 1/4 at pop 1000, 99.4–99.7 % of generation wall time inside the recompiling generations,
a median cold eval of 5.7–18.2 s against a warm 2.0–6.3 s. With `EVOGP_BATCH_CACHE_SIZE = 8` a pop-256
population already spans ~7–8 exec entries, so consecutive generations evict each other by construction.
Consequently the route is **dominated by the shipped `dense+iree` route at every pop ≥ 64**: 0.37–0.995×
dense's warm `sr_fitness`, a 10.9–12.2 s (pop 256) / 91–169 s (pop 1000) merged compile against dense's
3.4–6.5 s, and a `gp_step` 193–2355× slower than dense (5.36 vs 3446.20 ms at p256-L32, 12.35 vs
17524.31 ms at p1000-L64). It beats `specialized+numpy` warm by 8.7–29.6× but `auto` never selects it.
Its one win is a **fixed** population of ~64 trees over few data rows — p64-L64-i32-N64: 0.979 ms vs
dense's 1.410 ms, cold 2.17 s vs 6.18 s — a per-call win that a generation loop still loses.
The loss and per-row paths match the oracle to ~1e-7 rel; `batch_forward` on the arithmetic set is
non-bitwise where dense is exact, which is why the batch arms form their own digest group.

## Report and gate (`summarize.py` / `check.py`)

`summarize.py --results DIR --out FILE.md` renders one document from the records:
protocol statement, environment, one warm table + one speedup table per workload,
the `gp_step` section, the cold-compile exclusion, digest agreement and caveats.
Every table cell is a field of a record; a missing record or a contaminated guard
stays visible instead of being averaged away.
The final campaign's three renders (after, before, and the after-vs-before compare)
are stored as plain text in `results/final/RENDERED.md`; the report of record that
interprets them is [`FINAL.md`](FINAL.md). The batch round's renders are
`results/batch/RENDERED-warm-r5.md`, `RENDERED-small-r5.md`, `RENDERED-flat-r5.md`
and `RENDERED-flat1000-3w-r5.md`; [`BATCH.md`](BATCH.md) interprets them.

A batch arm is rendered by the same rules as any other evoxc arm, with two
additions: its `routes[...]` carries `kind: "batch"`, so the `auto route` column
quotes `batch/iree → batch` (annotated `bands=flat` when the merge policy differs
from the default — the batch arms are registered last, so they never shadow the
default policy in that column), and one `Batch route arms: ...` caption naming the
resolved knobs appears under each workload table. Both appear **only** in a
directory that actually recorded a batch arm, so a batch-free render stays
byte-identical, and `## Caveats` gains one bullet saying that a batch arm's
`first_call_ms` is a merged-graph compile for the whole population and is excluded
from the warm medians by protocol.

* `gp_step` is read from `median_total_ms` / `median_eval_ms` / `median_ops_ms`, and
  the eval/ops split carries the operator share of the generation
  (`X.XX / Y.YY (op ZZ%)`). A record that carries `steps[]` but not those aggregate
  keys is read from the upper median of `steps[].total_ms` / `steps[].eval_ms` -- the
  statistic `worker.py` stores -- and marked `*`; a raw-steps record never blanks a
  cell. The section also renders the per-cell `gp_step` speedup against both torch
  arms and the `best_fitness` the timed pipeline reached.
* `--compare-results DIR --compare-label NAME` (default `before`) adds one table per
  workload plus one for the `gp_step` total: the ratio of the primary directory's
  median to the compare directory's, per cell and arm, both from the same session.
  Only cells present in both directories are listed; an arm the primary side lacks is
  quoted from the compare side as `- (before <ms>)`. Without the flag the render is
  byte-identical to the single-directory one.

`check.py --results DIR` enforces the record contract (schema, provenance, >= 5 warm
samples with `len(samples_ms) == reps`, `first_call_ms > 0`, a re-derived median,
evoxc route + compile-miss fields, cross-arm digest agreement, `gp_step` sanity
including `median_total_ms` == the upper median of `steps[].total_ms`, guard
honesty) and exits 1 on any failure. It gates ONE campaign directory, so a
before/after round is gated one side at a time (`--results results/final/after`,
then `--results results/final/before`): `results/final/` also holds the
`cold/`, `training/` and `decompose*/` records, whose different schema the gate
rejects.

* `--allow-snapshot` turns a null `env.tree_root_sha` into a warning, but only when
  the recorded `env.tree_root` really has no `.git` (a `git archive` export has no
  HEAD, so the archived SHA has to be quoted next to the numbers).
* `--allow-missing` and `--allow-digest-mismatch` declare the two expected
  deviations: the 10 `evoxc-specialized-iree` cells that the compile-cost scope rule
  never schedules at pop ≥ 256, and the numpy-vs-iree sha256 split on
  `forward`/`batch_forward`. Without them the same records report those as failures
  (38 on the after side, 39 on `results/baseline`); with them the gates read
  130 records / 3602 checks and 90 records / 2642 checks, 0 failures.
* `check_batch()` runs on every batch-arm record and checks provenance, not
  magnitude: it **fails** when `batch_policy.mode_supported` is not true, when the
  recorded route is not `batch` (or has no `kind: "batch"`), or when `batch_cache`
  has no boolean `available`; it only **warns** when the merged-graph compile is not
  attributed to any cache counter (0 first-call misses across both caches), because
  `first_call_ms` still carries the cost and the magnitude of a miss count is never
  bounded by the gate; a `batch_cache.stats_keys` that carries no `batch_exec_misses`
  (the exec table, the only one whose miss recompiles) is likewise a warning.
  Expected-arm coverage defaults to the campaign's own
  scheduled arms, so appending these two arms leaves previously recorded campaigns
  gating exactly as before (`--allow-missing` semantics unchanged).
* The gate still demands the `gp_step` aggregate keys: the renderer's `steps[]`
  fallback is a reporting safety net, not a licence for a record to drop them.

## Numeric validation (`validate.py`)

The oracle is `src/evogp/tests/reference_eval.py`, the plain-torch reference the
pytest suite diffs against. For every validation cell and function set the same
population is evaluated by the oracle (CPU), by both torch arms and by the engine
(the route `EVOGP_EVAL_MODE`/`EVOGP_EVOXC_BACKEND` select), and the deviation of
every arm is reported against the oracle *and* against the engine: max abs, max rel,
NaN/Inf placement mismatches, bitwise-equality flag (`pop` is sliced to
`--max-oracle` trees for the CPU oracle; the record says how many were compared).

Three function sets, because they mean different things:

| set | contents | what agreement to expect |
|---|---|---|
| `arith` | `+ - * /` (the campaign set) | float32 arithmetic is correctly rounded, so every arm must agree **bitwise** on the value path |
| `rich` | `arith` + `max min < loose_div loose_inv sin exp log abs sqrt` | transcendental float32 kernels differ legitimately — `1e-8`–`1e-5` abs on the torch path, up to `~1e-1` abs in the tail of `batch_forward` on the dense route |
| `ternary` | `if + - * /` (needs a shorter-layer descriptor: the descriptor asserts `max_tree_len >=` the fully-branching size) | value path bitwise; the `if` select is where a naive implementation gets the operand order wrong |

Measured on the final campaign (12 cases over the four scheduled cells; the exact
table is in `VALIDATION.md`): the value path (`batch_forward`, `forward`) is
**bit-identical** across the oracle, both torch arms and the engine on `arith` and
`ternary` (max abs deviation `0.000e+00`), and `sr_fitness` agrees to `~3.5e-7`
relative — that residual is float32 reduction order (the engine reduces on the host
in numpy, the arms reduce in torch), not a semantic difference. `rich` is the one
set with real spread: `~9.3e-2` abs / `~1.5e-4` rel for the torch arms and up to
`4.4e-1` abs / `6.4e-1` rel for the dense route on `batch_forward`, i.e. the
dense/IREE lowering of transcendentals rather than a wrong tree evaluation.

Both torch arms were debugged this way and the two classic mistakes are worth
knowing before touching them: the reversed-prefix walk pushes the **right-most**
child first, so `stack.pop()` order is left-to-right (`a`, then `b`); and child
positions are **per row** (`c1 = i+1`, `c2 = c1 + size[row, c1]`,
`c3 = c2 + size[row, c2]`) — using row 0's sizes silently mis-wires every tree with
a different shape.

## Decomposition (`decompose.py`)

One representative cell (`p256-L64-i32-N1024`) on both routes: module-level
inclusive `perf_counter_ns` phase timers around the engine's own functions
(`decode_population`, `_input_to_numpy`, `_executable`, `_run_tree`,
`_evaluate_rows`, `prepare_dense`, `plan_dense`, `build_dense`, `run_loss_plan`, …),
the residual between the workload and its instrumented children printed as its own
row (that residual is the host glue: loss assembly, conversions, uploads), and seam
counters for `evoxc.run` dispatches, `evoxc.Tensor.to` uploads, `Tensor.numpy`
readbacks and `from_numpy` wraps. The uninstrumented median of the same call sits
next to the instrumented one so the timer overhead is visible. Cold compile is
reported separately.

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
  smoothed, retried or scoped away. `check.py` therefore only **warns** when that
  compile is not attributed to any counter, and `batch_cache.available: false` (a
  tree whose batch engine exposes no cache counters) is a reportable state, not a
  defect.
* Numerical honesty over tidiness: if a cell looks contaminated, re-run it and say
  so in the record (`--note`) instead of dropping or smoothing the number.
* `results/baseline/` and `results/final/**` are frozen evidence: a re-run writes
  into a fresh result directory and is gated there, never in place.
