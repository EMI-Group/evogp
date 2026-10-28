# `test/benchmark/evoxc/` — evoxc engine vs plain-PyTorch, controlled harness

The question this subtree answers: **how much does the evoxc-based EvoGP engine cost
per evaluation compared with the plain-PyTorch implementation of the same tree
evaluation?** — on identical populations, in the same session, warm/steady-state,
with every arm validated numerically against the repo's independent oracle.

It is deliberately separate from `../r2/`, `../r4/`, `../r5/`, `../r6/`: those
rounds compare the evoxc engine against the *deleted legacy CUDA engine* and their
committed numbers stay as history. This subtree never touches them.

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
summarize.py    renders the records into one markdown report (stdlib only)
check.py        the gate over a result directory (stdlib only, exit 1 on failure)
results/        raw per-cell JSON records + logs + campaign.json (the evidence)
```

## Arms

| arm | what it is |
|---|---|
| `torch-per-tree` | the classic pre-jit PyTorch route: for each tree, walk it in Python and apply torch ops to the `(N, input_len)` batch on the GPU under `torch.no_grad()`. The honest "old torch version". |
| `torch-batched` | the best a pure-torch implementation can do: the whole population in one vectorized pass (per-position `(pop, N)` buffer, one masked `torch.where` per occurring `(kind, function)` case, per-row child gather). No Python loop over trees. |
| `evoxc-auto` | the shipped default: `EVOGP_EVAL_MODE=auto`, backend unset (numpy on a box without IREE options set) |
| `evoxc-auto-iree` | `auto` but with `EVOGP_EVOXC_BACKEND=iree` — this is the route real training runs use |
| `evoxc-specialized-numpy` | `EVOGP_EVAL_MODE=specialized` + numpy |
| `evoxc-specialized-iree` | `EVOGP_EVAL_MODE=specialized` + iree (one compiled graph per distinct tree structure) |
| `evoxc-dense-iree` | `EVOGP_EVAL_MODE=dense` + iree (one all-ops graph per population shape) |

Both torch arms are *implemented in this harness* (`arms_torch.py`), op semantics
copied from the oracle: `DIV` → NaN on a zero divisor, `loose_div`/`loose_inv`
clamp the divisor with `copysign(DELTA, y)`, `max`/`min` via `torch.where`,
comparisons return ±1.0, `if` selects on `cond > 0`. Their decode/plan cost is
built once outside the timed region and reported as `setup_ms`.

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
* `specialized`-iree compiles one graph per *distinct* tree structure and a
  measured iree compile costs ~272-284 ms per structure on this box (p16: 4.26 s /
  15 misses; p64: 13.3 s / 49 misses), which makes a fresh 256- or 1000-tree
  population cost 1-5 minutes of compile and a GP step re-pay it every generation.
  That route is therefore timed on `pop ∈ {16, 64}` only; the excluded cells are
  listed in `campaign.json` under `protocol.scope_dropped`. Every other arm covers
  the whole grid.

## Commands

```bash
PY=/tmp/genesis/task_967a300e3bce7f35/evogp-venv/bin/python
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
export EVOXC_DISABLE_LOCATIONS=1

# 1. numeric validation of every arm against the oracle (no timing)
$PY test/benchmark/evoxc/validate.py --func-sets arith,rich,ternary \
    --out test/benchmark/evoxc/results/validation.json

# 2. the campaign (sequential; ~30-40 min for the full grid)
$PY test/benchmark/evoxc/campaign.py --out test/benchmark/evoxc/results/baseline \
    --cells main,n64 --arms all

# 3. gate + report
$PY test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/baseline
$PY test/benchmark/evoxc/summarize.py --results test/benchmark/evoxc/results/baseline \
    --out test/benchmark/evoxc/SUMMARY.md

# 4. per-call decomposition of the representative cell
$PY test/benchmark/evoxc/decompose.py --cell p256-L64-i32-N1024 \
    --out test/benchmark/evoxc/results/decompose

# 5. replay the same campaign against a frozen snapshot, apples to apples
git archive HEAD | tar -x -C /tmp/evogp_preopt
$PY test/benchmark/evoxc/campaign.py --tree-root /tmp/evogp_preopt \
    --out test/benchmark/evoxc/results/replay --cells main --arms all
```

`--tree-root <dir>` (or a `src/` inside it) makes `worker.py` import `evogp` from
that tree and puts it first on the worker's `PYTHONPATH`; the record's `env.tree_root`
and `env.tree_root_sha` say which checkout produced the numbers (`tree_root_sha` is
`null` for a snapshot, which has no `.git` — record the archived SHA next to it).

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
| `rich` | `arith` + `max min < loose_div loose_inv sin exp log abs sqrt` | transcendental float32 kernels legitimately differ in the last bits — 1e-5 abs / 1e-5 rel class |
| `ternary` | `if + - * /` (needs a shorter-layer descriptor: the descriptor asserts `max_tree_len >=` the fully-branching size) | value path bitwise; the `if` select is where a naive implementation gets the operand order wrong |

Measured on the baseline box (see `SUMMARY.md` "Numeric validation" for the exact
numbers): `batch_forward` and `forward` are **bit-identical** across the oracle, both
torch arms and the engine on `arith` (max abs deviation `0.000e+00`), and
`sr_fitness` agrees to `~2.5e-7` relative — that residual is float32 reduction
order (the engine reduces on the host in numpy, the arms reduce in torch), not a
semantic difference. On `rich` the value path deviates by `~1.5e-5` abs / `~5.8e-5`
rel, the expected transcendental-kernel spread.

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
* Numerical honesty over tidiness: if a cell looks contaminated, re-run it and say
  so in the record (`--note`) instead of dropping or smoothing the number.
