# `src/evogp/pipeline/` — Training-Loop Layer

## Intent

Top-level run-loop layer of EvoGP: it wires a `GeneticProgramming` algorithm to a `BaseProblem`, evaluates the population each generation, tracks the single best individual seen so far, and decides when to stop.
This is the **outermost layer of the package dependency DAG** — it may import from `..algorithm`, `..problem`, and (transitively) `..tree`, but nothing in `evogp` imports `pipeline/`.
It is the layer users touch directly (`from evogp.pipeline import StandardPipeline`) and the intended extension point for new run loops, checkpointing, early stopping, and logging.
The package root `src/evogp/__init__.py` is empty — there is no `evogp.StandardPipeline`; always import from the subpackage.

## Routing Table

- `base.py` → `BasePipeline`, the abstract contract (`step()` / `run()` raise `NotImplementedError`); see "Writing a Custom Pipeline" below.
- `standard.py` → `StandardPipeline`, the only concrete implementation (~107 lines): evaluate → track best → evolve → print → check termination.
- `__init__.py` → re-exports exactly `BasePipeline` and `StandardPipeline`; add new pipelines here.
- `../algorithm/` → sibling: `GeneticProgramming`, operators (read-only; escalate writes to `src/evogp/`).
- `../problem/` → sibling: `BaseProblem` and task definitions (read-only; escalate writes to `src/evogp/`).
- `../tree/` → sibling: `Forest` / `Tree` / `GenerateDescriptor` (read-only; escalate writes to `src/evogp/`).

## API Surface

### `BasePipeline` (`base.py`)

- `__init__(self)` — no-op; subclasses keep their own signature.
- `step(self)` — one generation; raises `NotImplementedError`.
- `run(self)` — full search; raises `NotImplementedError`.
- Nothing else is part of the contract: no required attributes, no `__init__(algorithm, problem)` convention enforced by the base class.
- Because the base class has no `__init__` body, subclasses must not call `super().__init__()` expecting shared state.

### `StandardPipeline` (`standard.py`)

Constructor (all parameters positional-or-keyword; `algorithm` and `problem` are passed positionally in every example):

```
StandardPipeline(
    algorithm: GeneticProgramming,
    problem: BaseProblem,
    fitness_target: float = None,
    generation_limit: int = 100,
    time_limit: int = None,
    is_show_details: bool = True,
    valid_fitness_boundry: float = 1e8,
)
```

Instance attributes / results a caller reads:

- `algorithm`, `problem` — the wired collaborators.
- `best_tree` — the best `Tree` found across all generations so far, or `None` if no improvement has ever been recorded (see Known Issues). This is what `run()` returns.
- `best_fitness` — best scalar fitness seen so far; starts as the Python float `-inf`, and becomes a **0-dim `torch.Tensor`** after the first update (so it prints as `tensor(0.9973)`, not `0.9973`).
- `fitness` — CPU `torch.Tensor` of shape `(pop_size,)` holding the *most recent* generation's fitnesses (all trees, unsorted); `None` before the first `step()`. Public usage pattern: `mse = -torch.max(pipeline.fitness)` (`example/uci_sr.py`, `example/custom_sr.py`).
- `fitness_target`, `generation_limit`, `time_limit`, `is_show_details`, `valid_fitness_boundry` — stored verbatim; they are read every iteration, so mutating them mid-run changes termination behavior.
- `generation_timestamp` — initialized to `None` and never written anywhere; dead attribute, do not rely on it.

## Contract & Semantics

### `step()` — one generation

1. `fitnesses = self.problem.evaluate(self.algorithm.forest)` — a CUDA tensor of shape `(pop_size,)`.
2. NaN fitnesses are overwritten with `-inf` **in place** on that returned tensor.
3. `cpu_fitness = fitnesses.cpu()`; `best_idx = int(torch.argmax(cpu_fitness))`; if `torch.max(cpu_fitness) > self.best_fitness`, update `self.best_fitness` and set `self.best_tree = self.algorithm.forest[best_idx]`.
4. `self.algorithm.step(fitnesses)` — the **GPU** tensor is handed to the algorithm (selection/crossover/mutation/pareto update).
5. Returns `cpu_fitness` (CPU copy) — the GPU tensor is not returned.

### `run()` — the generation loop

- Loop body is `self.fitness = self.step()` (stored), then optional `show_details`, then three termination checks **in this order**:
  1. `fitness_target is not None and self.best_fitness >= self.fitness_target` → prints `"Fitness target reached!"`.
  2. `time_limit is not None and time.time() - tic > self.time_limit` → prints `"Time limit reached!"`.
  3. `generation_cnt += 1; generation_cnt >= self.generation_limit` → prints `"Generation limit reached!"`.
- Termination messages are plain `print` — not `logging`, and they overwrite nothing.
- `generation_cnt` is a **local** variable starting at 0 and incremented after the checks, so exactly `generation_limit` `step()` calls happen and the counter is not observable after `run()` returns.
- `tic` is taken once at the top of `run()`, so `time_limit` on a second `run()` call restarts the clock.
- Returns `self.best_tree` — a single `Tree`, never a forest, never the fitness.
- Note the interplay: `fitness_target` is compared against `best_fitness` (best-ever), while `time_limit`/`generation_limit` are wall-clock/generation counters.

### `show_details(start_time, generation_cnt, fitnesses)`

- Filters the incoming CPU fitness tensor to the band `(-valid_fitness_boundry, +valid_fitness_boundry)` and prints count/max/min/mean/std plus the step's elapsed milliseconds.
- The filter exists to hide the `±inf`/overflow "invalid" individuals from the mean/std statistics — `max`/`min` are taken over the same filtered set, so they are **not** the true population max.
- Because NaN fitnesses become `-inf` and `-inf` is outside the band, invalid trees are excluded here but still counted in `pipeline.fitness`.
- This method is the *only* overridable hook in the class (it already takes everything it needs as arguments), which makes it the natural place to bolt on custom logging.
- `show_details` receives `generation_cnt` **before** increment, so the first generation printed is `0`.

## Constraints

- **Dependency direction is one-way.** `pipeline/` must never be imported by `algorithm/`, `problem/`, or `tree/`; adding such an import creates a cycle. Verified: no reference to `pipeline` exists in those subtrees.
- **Fitness is maximized, not minimized.** `step()` uses `argmax` and the improvement test is `>`. Every built-in problem already returns a negated loss (e.g. `SymbolicRegression.evaluate` returns `-MSE`), so the maximize convention holds end-to-end — a new problem must return "higher is better" values or the pipeline will happily track the worst tree.
- **`problem.evaluate(forest)` must return a fresh, writable, dense CUDA tensor of shape `(pop_size,)`** with `dtype` compatible with `-torch.inf` assignment; `step()` mutates it in place with `fitnesses[torch.isnan(fitnesses)] = -torch.inf`.
- **`problem.evaluate` and `algorithm.step` must accept the *same* forest** — the pipeline re-reads `self.algorithm.forest` on every call, so it always evaluates the population the algorithm currently holds (`GeneticProgramming.step` rebinds `self.forest` each generation).
- `problem.problem_dim` / `problem.solution_dim` must match the forest descriptor's `input_len` / `output_len` — the pipeline does **not** check this (see Known Issues).
- `run()` assumes at least one `step()` is legal: a `generation_limit` of 0 or a negative value still runs one generation because the counter is checked after the step.
- Constructing a pipeline requires an already-built `GeneticProgramming` (hence an already-built `Forest`), which requires the compiled `evogp.evogp_cuda` extension — nothing in this directory is usable before the CUDA/HIP build.

## Design Decisions

- **`best_tree` is a live tensor-view snapshot, not a deep copy.** `Forest.__getitem__(int)` (`../tree/forest.py`) builds `Tree(self.input_len, self.output_len, self.batch_node_value[i], ...)` and `Tree.__init__` (`../tree/tree.py`) stores those tensor references untouched — basic int indexing returns **views** into the population tensors. `algorithm.step` then rebinds `self.forest = self.forest[elite_indices] + next_forest` (advanced indexing + `torch.cat`), which allocates brand-new tensors each generation, so the old population buffers stay alive only through `best_tree` and are never written again. The snapshot is therefore correct *and* free (no clone cost), but it is correct only by virtue of `GeneticProgramming` always rebinding rather than mutating — see Known Issues for the failure mode.
- Actually a `Tree`, not a `Forest` of size 1 — `.forward(x)`, `.to_sympy_expr()`, `.to_png()` work directly on the return value of `run()`.
- **CPU fitness for reporting, GPU fitness for evolution.** The algorithm needs the GPU tensor; everything the pipeline inspects (argmax, best tracking, `show_details`, `pipeline.fitness`) goes through `.cpu()` so the print/statistics path never triggers a device sync per generation and the exposed `fitness` attribute is host-side.
- **Filtered statistics** (`valid_fitness_boundry`, default `1e8`) keep overflowed/blown-up expressions from wrecking the printed mean and std.
- **Termination is checked after the step, not before it**, so a run always performs at least one generation and the "reached" conditions describe results already produced.
- **No logging framework, no callbacks, no checkpointing** — this keeps the layer a ~100-line readable reference. Extensibility is delegated to subclassing/overriding rather than an event system.
- **No assertions on problem/forest dimensionality** — the two asserts are commented out (see Known Issues).

## Known Issues

- **Missing dimension validation.** The lines `# assert algorithm.forest.input_len == problem.problem_dim` and `# assert algorithm.forest.output_len == problem.solution_dim` are commented out in `__init__`. A descriptor built with mismatched `input_len`/`output_len` is accepted silently; the failure then surfaces either as a shape assertion deep inside the CUDA kernels (`forest.forward` / `batch_forward`) or, worse, as silently wrong fitness when the shapes happen to broadcast. Re-enabling these two lines is the standard fix; the object has no other validation path.
- **`best_tree` may be `None`.** If every tree in every generation gets a NaN fitness (all become `-inf`), `-inf > -inf` is false forever, so `best_tree` stays `None` and `run()` returns `None` — callers like `example/basic.py` then crash on `best.forward(...)` with an `AttributeError` far from the real cause.
- **`show_details` crashes when nothing is valid.** If *all* fitnesses fall outside `±valid_fitness_boundry` (all `-inf` from NaN, or all blown up beyond `1e8`), `valid_fitness` is empty and `max(valid_fitness)` raises `ValueError: max() arg is an empty sequence`. With the default `is_show_details=True` this turns a "population failed to evaluate" situation into an exception inside the reporting code.
- **`best_tree` aliases population memory.** The snapshot is safe only because `GeneticProgramming.step` rebinds `self.forest` to freshly allocated tensors. `DefaultMutation.__call__` writes into its argument in place (`forest[mutate_indices] = ...`, via `Forest.__setitem__`), so any algorithm variant that mutates the *current* forest (or any future in-place crossover) would silently corrupt `best_tree` retroactively — the pipeline would return whatever that row contains at the end of the run.
- **`run()` does not reset state.** `best_tree`, `best_fitness`, and `fitness` persist across calls, and `generation_cnt` restarts at 0 each call: a second `run()` runs another `generation_limit` generations but keeps the previous best (and `best_fitness`, so the `fitness_target` check can fire immediately).
- **`fitness_target` is an untested code path.** No example, README snippet, or test passes it; only `generation_limit` (and `is_show_details` in a few experiment scripts) is exercised in practice. Its check also compares a possible 0-dim tensor (`best_fitness`) against a float.
- **`time_limit` is coarse.** The clock is only tested between generations, so a single expensive generation (RL tasks in `example/brax_task.py`, `example/mjx_task.py`) can overrun the limit substantially.
- **`best_fitness` type instability.** It is `float("-inf")` before the first improvement and a 0-dim CPU tensor afterwards; code that does `float(pipeline.best_fitness)` works either way, but `pipeline.best_fitness > 0` returns a tensor and formatting with `f"{pipeline.best_fitness:.4f}"` only works on the tensor form.
- **Dead code / unused imports.** `standard.py` still imports `numpy as np` although the statistics were switched to `torch.mean`/`torch.std`; `self.generation_timestamp` is initialized but never read or written again.
- **`is_show_details=True` is the default**, so every example prints a line per generation; the printing is synchronous `print` with no rate limiting.
- **Reporting is not resumable/persistent.** Nothing is written to disk; there is no way to recover `best_tree` after the process exits (relevant given `Forest.__getstate__`/`__setstate__` exist for pickling).

## Notes for Agents

### This is the layer to extend for new run loops

- **To add a custom pipeline**: create a new module next to `standard.py`, subclass `BasePipeline`, and re-export it from `__init__.py`. `BasePipeline` only demands `step(self)` and `run(self)`; it supplies no constructor, attributes, or helpers, so a custom pipeline is free to define its own signature.
- **The two things a custom pipeline must supply**: (1) the evaluate/evolve cycle — call `problem.evaluate(algorithm.forest)`, sanitize non-finite fitnesses, record the best if desired, then call `algorithm.step(fitnesses)` with a `(pop_size,)` tensor; and (2) the termination + return policy — a loop over that cycle that stops on some criterion and returns whatever the caller needs (the reference returns a single `Tree`).
- **Prefer copying the `step()` shape from `StandardPipeline`** rather than re-deriving it: the NaN→`-inf` sanitization and the argmax-over-CPU-copy pattern are both load-bearing.
- Candidate improvements that would live here: early-stopping callbacks, per-generation metric logging, checkpointing `best_tree` to disk, multi-objective/pareto-aware pipelines (`GeneticProgramming(enable_pareto_front=True)` already maintains `algorithm.pareto_front`, but no pipeline exposes it — examples read it directly off `algorithm`, see `example/basic.py`).
- Do **not** import `pipeline` from `algorithm/`, `problem/`, or `tree/` — that inverts the dependency DAG.

## Test Strategy

- **No tests exist in this directory**, and the repo has no automated test suite for it: `test/` contains only ad-hoc scripts and notebooks (`test/fix_bug.py`, `test/test_bind_success.py`, `test/test_torch_env.py`, `*.ipynb`), none of which import `pipeline`.
- Practical smoke tests, in increasing cost:
  - `python -m evogp.sr_test` — symbolic regression, pop_size 1000, `generation_limit=100`; the canonical end-to-end check that the pipeline + algorithm + CUDA extension all work.
  - Repo-root `example/` scripts via uv: e.g. `uv run example/basic.py` (XOR, pop 50, `generation_limit=100`, also exercises `pareto_front` and `best.to_sympy_expr()`), `example/combined_sr.py`, `example/uci_classifier.py`, `example/brax_task.py`.
  - For quick manual verification of loop bookkeeping, set `generation_limit=2` and `is_show_details=True` and check that exactly two generation lines print and that they are numbered `0` and `1`.
- All of the above require the compiled `evogp.evogp_cuda` extension (`setup.py` / `src/evogp/cuda/`), so this layer cannot be smoke-tested on CPU-only machines.
- Behavioral invariants worth asserting if tests are ever added: `len(run()-returned fitness history) == generation_limit`; `best_fitness` is monotone non-decreasing across generations; `best_tree` is never worse than any generation's max; `pipeline.fitness.shape == (pop_size,)`; `run()` returns `None`-only in the all-invalid case.
