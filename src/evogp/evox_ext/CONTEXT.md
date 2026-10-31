# `src/evogp/evox_ext/` — EvoGP as a first-class Evox 2.0 citizen (`StdWorkflow` components)
## Intent
EvoX 2.0 (the functional rewrite, `/root/evox`, `evox` 2.0.0a1) has no tree/GP/symbolic-regression component, and `evogp.gp` — the torch-free functional GP stack — was a stand-alone library with its own driver.
This subtree is the bridge: it adapts `evogp.gp` to the Evox 2.0 **component contract** so a GP run is an `evox.core.StdWorkflow` run (`StdWorkflow(GPAlgorithm(...), SRProblem(...), opt_direction="max")`), with the evaluation traced inside the workflow's own graph.
What goes back is the GP layer itself: this adapter plus the `evogp.gp` stack it wraps is EvoGP's contribution to the Evox family, not a port of anything that repo ships.
It targets the `evox` revision that also carries the opt-in compiled-step artifact cache, `StdWorkflow(cache_dir=...)` (commit `b8fcc58f` in the task's evox worktree, `/root/evox/.genesis/foreign_repos/evox`; `/root/evox` HEAD `e568755d` predates it): one content-addressed `evoxc` artifact per state signature, which turns a repeated iree cold start from 1.30 s into 0.07 s.
The contract was read off the code, not assumed: the workflow resolves each component with `importlib.import_module(type(config).__module__)`, so a component is *a module holding a frozen config dataclass plus the module-level `init(config, key)` / `step(config, state, evaluate)` / `evaluate(config, state, pop)` functions* (`evox/core/algorithm.py` and `evox/core/problem.py` are documentation-only protocols; nothing inherits from them).
There is no new GP logic here: the generation sampler, the operator layer and the SR graph body are `evogp.gp`'s, reached through its own traced entry points.
Nothing outside this subtree depends on it, and it imports nothing from `evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline` (the torch stack is untouched).

## API Surface
- `gp_algorithm.py` — the **algorithm component**.
  `GPAlgorithmConfig(tree: GPTreeConfig, gp: GPConfig)` (alias `GPAlgorithm`, frozen, hashable, validated in `__post_init__`), plus the evox-style constructor `make_gp_algorithm(tree, gp=None, *, pop_size=None, **gp_options)`.
  `init(config, key) -> GPState` calls `evogp.gp.encoding._generator_fn(tree, pop_size)`'s traced generator and wraps the four outputs with an all-`-inf` fitness vector; `step(config, state, evaluate) -> GPState` calls the workflow's `evaluate(state)` once, negates the returned (workflow-scaled) vector back to MAXIMIZED, and calls `evogp.gp.variation._variation_fn(tree, gp, pop_size, width, elite_count)`'s traced operator (selection + crossover + mutation + elite splice in ONE traced call).
- `sr_problem.py` — the **problem component**.
  `SRProblemConfig(n_rows, input_len, x, y, use_mse=True, used_ids=None, sanitize=True)` (alias `SRProblem`; `x`/`y` normalized to plain float tuples so the config stays static and hashable), `SRProblemState(x, y)` (the data as `evoxc` tensors), the constructor `make_sr_problem(x, y, *, use_mse, used_ids, sanitize)`, and `init(config, key) -> SRProblemState` / `evaluate(config, state, pop) -> (fitness, state)`.
  `evaluate` is `evogp.gp.graph_eval`'s **in-graph** route: `_channel_triple` (accepts a `GPState` or the channel triple) → `_cfg` → `_device_sr_fitness`, i.e. the same traced body `graph_eval.GraphEvaluator` runs.
- `stdworkflow.py` — the wiring helpers.
  `make_stdworkflow(algorithm, problem, *, backend="numpy", device=None, num_generations=None, monitor=None, compile_options=None, cache_dir=None, opt_direction="max")` builds the `StdWorkflow` (and REJECTS any other orientation, `OPT_DIRECTION = "max"`).
  `cache_dir` is forwarded as `StdWorkflow(cache_dir=cache_dir)` when the installed `evox` declares that keyword, and `accepts_cache_dir(workflow_cls)` is the `inspect.signature` probe that decides it.
  An `evox` without the keyword drops a requested `cache_dir` with a `RuntimeWarning` instead of raising, so the adapter runs on both releases.
  Readbacks for a run: `generation_fitness(state)` (the `[pop_size]` MAXIMIZED vector the last step wrote, host numpy), `best_so_far(state)`, `best_row(state)`, `host_channels(state)` (the three channels as host numpy arrays, device-safe).
- `demo_stdworkflow.py` — the runnable demonstration (`if __name__ == "__main__"`, unlike every torch-stack entry point): builds the workflow, runs N generations of `x0**2 + x1**2` symbolic regression, prints the best-so-far trajectory and the timing.
  `--cache-dir <dir>` opts into the step-artifact cache (`--backend iree --device rocm:0` for a device run), so a second run of the same cell reloads the compiled step graph instead of paying its compile inside `init()`.
  Its evolution wiring is imported from `evogp.gp.smoke.make_configs` (the reference configuration of the driver this replaces), so the two runs are comparable and nothing is duplicated.
- `__init__.py` re-exports the whole public surface; `CONTEXT.md` is this file; `tests/` is the contract suite.

## Constraints
- **Configs hold static leaves only** (`GPTreeConfig`/`GPConfig` objects, ints, bools, float tuples): the traced graphs close over them, so a config value is baked into the compiled program and changing one is a different graph (never a mutated run).
  `SRProblemConfig.__post_init__` normalizes array-like `x`/`y` into float tuples and resolves `used_ids` once (so the config stays hashable and the dispatch chain stays static).
- **State is tensors-only with constant shapes**: the algorithm state IS `evogp.gp.state.GPState` (`opcode`/`payload`/`size` `[pop_size, max_nodes]` int32/float32/int32, `fitness` `[pop_size]` float32, `key` rank-0 int64), the problem state is `SRProblemState` (`x` `[n_rows, input_len]`, `y` `[n_rows]`, both float32).
  `StdWorkflow._state_signature` keys the compiled step graph on exactly that signature, so a shape change is a re-trace: `gp_algorithm.step` validates the state's shape against the config and raises rather than silently building the wrong graph.
- **Evaluation is in-graph, never a host callback**: the data and the targets ride in the PROBLEM STATE (runtime inputs of the step graph, uploaded once) instead of being baked as graph constants, which is what keeps a large data table out of the compiled artifact.
  `evogp.gp.graph_eval.GraphEvaluator` (or `sr_fitness_channels`) cannot be used inside a workflow: they call `evoxc.build_cached` themselves, and a nested build inside an active trace is not allowed; `evogp.gp.fused_eval.FusedDeviceEvaluator` is excluded for the same structural reason (it runs its own Triton kernel from the host).
  The private traced builders are reached directly (`encoding._generator_fn`, `variation._variation_fn`, `graph_eval._device_sr_fitness` / `_cfg` / `_channel_triple`) — `evogp.gp` ships no "traced function only" public accessor; if it grows one, swap the three call sites.
- **One compile per state signature, and `StdWorkflow.init` pays it**: `init()` builds the init graph AND the step graph (`_finish_init`), so the whole compile (`evoxc.build`, not `build_cached`) lands in `init()`; every later `step()` is one graph run.
  `StdWorkflow` uses `evoxc.build`, so the per-structure `build_cached` strategy (one graph per distinct tree structure, `evogp.tree`'s specialized route) is NOT available inside a workflow — one graph serves the whole population and every generation of a fixed shape.
  That compile is the one cost `StdWorkflow(cache_dir=...)` removes across processes: the step executable is persisted as an `evoxc` artifact container at `<cache_dir>/<key[:2]>/<key>.evoxexe`, keyed by the state signature, backend/device, compile options and the static configs, and `make_stdworkflow(cache_dir=...)` forwards it.
- **The sign convention**: `evogp.gp` fitness is MAXIMIZED, `StdWorkflow` hands the algorithm an `opt_direction`-scaled vector (`-fitness` for `"max"`), so `gp_algorithm.step` negates exactly once; `make_stdworkflow` fixes `opt_direction="max"` and refuses anything else.
  Optimization is reported in EvoGP units by `generation_fitness` / `best_so_far`, and because elitism is spliced INSIDE the variation graph those numbers are the elite values of the generation before (the offspring rows are the `-inf` "not evaluated yet" marker), which is why the trajectory is monotone non-decreasing by construction.
- **No monitor**: the candidate object a workflow hands its monitor is this stack's `GPState`, not an `(n, dim)` matrix, so `evox.workflows.EvalMonitorConfig` would record channels it cannot interpret; read the trajectory with `best_so_far` instead (`monitor=` is accepted by `make_stdworkflow` only for the "no monitor" case, and no `record_step` hook is defined).
- **No `evox_ext/` namespace shim** (deliberate): EvoX's PEP 420 plugin namespace (`src/evox_ext/<domain>/<module>.py`, merged into `evox.<domain>` by `evox_ext.autoload_ext.auto_load_extensions()` at `import evox`) would only buy the shorter spelling `evox.algorithms.<module>` — the component resolution itself is by MODULE, so a workflow takes these configs directly.
  Installing into that namespace would mean shipping a third top-level import package and running EvoGP code at `import evox` time; the module contract suffices and keeps the surface local.
- `import evogp.evox_ext` needs neither `evox` nor `evoxc`: `evox` is imported inside `make_stdworkflow`, and `evoxc` only ever through `evogp.gp`'s lazy accessor.
  A workflow run needs both (`PYTHONPATH=src:/root/evoxc:/root/evox/src`).
- Never edit `src/evogp/gp/**` from here: that stack is a separate node's file scope; a missing accessor is worked around inside this subtree (see the private traced-builder note above).

## Measured economics (validation box, MI300X, `EVOXC_DISABLE_LOCATIONS=1`)
- Demo defaults (pop 100, `max_nodes` 32, 256 data rows, numpy backend, CPU): graph build in `init()` ≈ 0.7 s, then **≈33 ms/generation** warm (32.6–33.3 ms over runs); the best fitness goes `-252.85` (generation 0) → `-56.72` (generation 24).
- The same cell on the device (`--backend iree --device rocm:0`, pop 64): **1.08 ms/generation** warm against 30.8 ms on numpy (≈28×), with ≈5.3 s of IREE compile inside `init()`.
  The trajectory is numerically identical to the numpy run (same seeds; `-252.845` → `-120.839` either way, agreeing to float reduction order), i.e. the two backends agree on this route.
- Against `evogp.gp`'s own routes at that cell: the host numpy evaluator (`SymbolicRegression.evaluate`) is **14.8 ms/generation** and the in-graph `GraphEvaluator` (two graph runs per generation) is **61.8 ms/generation**, so a workflow generation is ≈2.1× cheaper than the two-run device route but ≈2.2× dearer than the host path.
  The gap grows with the data axis (`N=1000`, pop 200, `max_nodes=64`: host 25.8 ms, workflow 415.4 ms, `GraphEvaluator` 903.6 ms), because the in-graph body materializes `[pop_size, n_rows]` intermediates per tree column while the host evaluator reduces the whole population in one vectorized numpy pass.
  This is the in-graph route's cost, not a workflow overhead: the workflow removes one graph run and every host round trip a generation.

## Routing Table
- `tests/` → the Evox 2.0 contract suite (`test_evox_ext_contract.py`): workflow acceptance + ≥2 generations, tensors-only constant-shape state, monotone best-so-far + truth parity with `evogp.gp`'s evaluator, one compiled step graph after several generations, the orientation guard. See `tests/CONTEXT.md`.
- Sibling paths (read-only from here — escalate writes to this node's parent): `../gp/` (the stack being adapted: `config`, `state`, `encoding`, `variation`, `graph_eval`, `problem`, `workflow`, `smoke`), `../tree/`, `../tests/`, and the foreign checkouts `/root/evox` (`src/evox/core/workflow.py` is the contract) and `/root/evoxc` (the compiler runtime).

## Known Issues
- No pytest in the repo's historical gates collects this suite: run it explicitly (`pytest src/evogp/evox_ext/tests`), like `src/evogp/gp/tests`.
- The in-graph evaluation is far dearer than the host numpy evaluator at large `n_rows` (16× at `N=1000`/pop 200/`max_nodes=64`); there is no route to the fused host-callback evaluator inside `StdWorkflow`, so the numpy backend is a correctness/CPU path and a device run is the fast path.
- `StdWorkflow` calls `evoxc.build` (not `build_cached`), so a per-structure graph specialization — the shipped `specialized` route's whole trick — is unavailable here, and `auto`'s `EVOGP_EVAL_MODE` policy is not consulted at all: the workflow's `backend`/`device` arguments are the only routing knobs.
- Multi-output trees are out of scope on both ends: `evogp.gp.init_population` raises `NotImplementedError` for `output_len > 1`, and `SRProblemConfig` is single-output (exactly like `evogp.gp.problem.SymbolicRegression`).
- `validation` (`evogp.gp.encoding.validate_channels`) is not run inside a workflow: it is a host pass over a numpy view of the channels, and a traced function cannot download. The operator layer's own well-formedness contract is relied on, as on every device route.
- A device run prints IREE's `nanobind ... leaked function` reference-leak report at interpreter shutdown (an `iree.runtime` binding warning, seen with `iree-base-runtime` here); the run's exit status is unaffected.
- Class names are necessarily close to their torch/`gp` namesakes (`evogp.evox_ext.SRProblemConfig` vs `evogp.gp.problem.SymbolicRegression` vs `evogp.problem.SymbolicRegression`): they are configs, not evaluator objects, and are always addressed through this package.

## Notes for Agents
- Environment: `PYTHONPATH=src:/root/evoxc:/root/evox/src`, the torch-carrying venv `/tmp/rocvenv/bin/python`, `EVOXC_DISABLE_LOCATIONS=1` for speed (the demo sets it itself).
- Run the demo: `python -m evogp.evox_ext.demo_stdworkflow` (CPU, ~1 s for 10 generations) and `--backend iree --device rocm:0` on a device box; `--quiet` prints only the summary line.
- Run the contract suite: `EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc:/root/evox/src <venv>/bin/python -m pytest src/evogp/evox_ext/tests -q` (13 tests, ~6 s; without `evox` on the path the module skips as a whole).
- Read `evox/core/workflow.py` before changing anything here: the exe cache key is `(resolved variant, state signature)` over `evoxc.tree_leaves(state)`, `evaluate` is traced with `solution_transform → problem.evaluate → opt-direction scaling → fitness_transform → monitor`, and `init()` is the call that builds the step graph.
- The data arrays live in the problem state, so a larger dataset changes only the state signature (one recompile, upload once) — do not move them back into the config.
- `evogp.gp`'s `GPState`, `GPTreeConfig`, `GPConfig` and the traced bodies are the single source of truth: change the GP behaviour THERE (in `../gp/`), never here; this subtree is an adapter and must stay thin (≤ ~300 lines per component module).
