# src/evogp/algorithm — GP operator layer

## Intent
The genetic-programming evolution engine of EvoGP: the population-level loop (`GeneticProgramming`) plus the three interchangeable operator families it composes — selection, crossover, mutation.
Everything here works on populations held in `evogp.tree` containers (`Forest`, and for multi-output problems `CombinedForest`) and is expressed as plain callable Python classes — no `nn.Module`, no autograd, no registered parameters.
Operators are duck-typed strategies: the loop only calls `__call__`, so any object honouring the documented contract can be plugged in.
This is the layer directly above `evogp.tree`: it consumes the tree/CUDA layer and is consumed by `evogp.pipeline` (and directly by `example/` scripts).

## API Surface
`algorithm/__init__.py` is 4 lines: `from .crossover import *`, `from .mutation import *`, `from .selection import *`, `from .genetic_programming import GeneticProgramming`.
None of the subpackages defines `__all__`, so every public name they bind is re-exported flat at `evogp.algorithm.<Name>`; that shallow namespace is the de-facto public API (README, `tutorial/` and every `example/*.py` import from `evogp.algorithm` only).
- Loop: `GeneticProgramming(initial_forest, crossover, mutation, selection, enable_pareto_front=False)` — `.forest` is the live population, `.step(fitness)` advances one generation.
- Pareto: `ParetoFront` lives in `genetic_programming.py` and is deliberately NOT re-exported (reachable only as `evogp.algorithm.genetic_programming.ParetoFront`); `algorithm.pareto_front` exists only when `enable_pareto_front=True`.
- Crossover (5 names): `BaseCrossover`, `DefaultCrossover`, `DiversityCrossover`, `LeafBiasedCrossover`, `CombinedDefaultCrossover`.
- Mutation (11 names): `BaseMutation`, `DefaultMutation`, `HoistMutation`, `DeleteMutation`, `InsertMutation`, `SinglePointMutation`, `SingleConstMutation`, `MultiPointMutation`, `MultiConstMutation`, `CombinedMutation`, `CombinedDefaultMutation`.
- Selection (11 names): population-level `BaseSelection`, `DefaultSelection`, `RankSelection`, `RouletteSelection`, `TournamentSelection`, `TruncationSelection` + crossover-facing `BaseSelector`, `RankSelector`, `RouletteSelector`, `TournamentSelector`, `TruncationSelector`.
- Documented/supported wiring (README quickstart, `example/basic.py`, `tutorial/evogp_intro.ipynb`): `GeneticProgramming(initial_forest=Forest.random_generate(pop_size=..., descriptor=descriptor), crossover=DefaultCrossover(), mutation=DefaultMutation(mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)), selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01), enable_pareto_front=True)`.
- Multi-output wiring (`example/combined_sr.py`): `CombinedForest.random_generate(...)` + `CombinedDefaultCrossover()` + `CombinedDefaultMutation(mutation_rate=..., descriptors=...)` + the same `DefaultSelection`.

## The generation step (data flow)
`GeneticProgramming.step(fitness)` is the entire evolution loop; fitness is produced OUTSIDE by the problem — `pipeline/standard.py` calls `problem.evaluate(algorithm.forest)`, maps NaN to `-inf`, then calls `algorithm.step(fitnesses)`.
1. Asserts `fitness.shape == (forest.pop_size,)`.
2. If enabled, folds the population into the Pareto front per tree size.
3. `elite_indices, survivor_indices = self.selection(self.forest, fitness)`.
4. `next_forest = self.crossover(forest=self.forest, survivor_indices=survivor_indices, target_cnt=self.pop_size - elite_indices.shape[0], fitness=fitness)` — crossover returns exactly the offspring and must never include the elites (all four args are passed by keyword).
5. `next_forest = self.mutation(next_forest)` — mutation only ever sees the offspring.
6. `self.forest = self.forest[elite_indices] + next_forest` — elites are COPIED (fancy indexing) and prepended, offspring appended; this new forest is both stored and returned.
Invariants this creates: each generation is `elite_cnt` verbatim elites + `(pop_size - elite_cnt)` offspring, so `pop_size` never changes; elitism is applied AFTER mutation, hence elites are never mutated; the operators never need to know elites exist.
`survival_rate` only sizes the parent pool that crossover may draw from — it does not change `pop_size`.
Selection pressure is carried by the MULTIPLICITY of entries in `survivor_indices` (crossover resamples uniformly by position), not by their order.
With `DefaultSelection` both index tensors are the top individuals sorted by descending fitness, so the new population's leading rows are the fittest individuals in ranked order.
`GeneticProgramming` keeps no best-so-far individual of its own (only the optional Pareto front); tracking the best tree is the pipeline's job.

## Pareto front (`enable_pareto_front=True`)
`ParetoFront(max_tree_len, (max_tree_len, input_len, output_len))` allocates `fitness` of shape `(max_tree_len,)` filled with `-inf` plus `solution = Forest.zero_generate(...)`, i.e. ONE best solution per possible tree size (`pop_size` of that internal forest == `max_tree_len`).
`step` calls `vmap_update_pareto_front(fitness, solution)` (the name is misleading — it is a broadcasting implementation, no `torch.vmap`): it buckets individuals by `batch_subtree_size[:, 0]`, takes the per-size max, and writes back the winners' `batch_node_value`/`batch_node_type`/`batch_subtree_size` wherever the new best beats the stored one.
Fitness is assumed non-decreasing-in-quality (higher is better) and the front is a pure by-product: it never feeds back into selection in these strategies, it is only reported (`example/basic.py`, `example/classifier.py` print it).

## Constraints
- Dependency direction is one-way: `algorithm/` may import `evogp.tree`; `tree/` and `problem/` must NEVER import `algorithm/`, and the only package consumers are `pipeline/` (`from ..algorithm import GeneticProgramming`) plus the stray runnable script `src/evogp/sr_test.py`.
- Intra-package edge to respect: `crossover/leaf_biased.py` and `crossover/diversity.py` do `from ..selection import BaseSelector`, so `selection/` must never import from `crossover/` (cycle).
- Import ORDER inside `__init__.py` is load-bearing: `genetic_programming.py` does `from . import BaseMutation, BaseCrossover, BaseSelection`, which only resolves because the three star imports run first; hoisting the `genetic_programming` import above them raises `ImportError: cannot import name ... from partially initialized module` (verified).
- The loop is container-agnostic: `initial_forest` must expose `pop_size`, `max_tree_len`, `input_len`, `output_len`, support `forest[index_tensor]`, `forest_a + forest_b` and (for Pareto) `batch_subtree_size[:, 0]` — both `Forest.random_generate(...)` and `CombinedForest.random_generate(...)` satisfy this, but the container must MATCH the operators (plain operators for `Forest`, combined operators for `CombinedForest`); nothing validates this.
- CUDA-only: tensors are created with hardcoded `device="cuda"` (or `.cuda()`) throughout crossover/mutation/selection and in `ParetoFront`; only `DefaultSelection` is device-agnostic, and importing the package transitively imports the compiled `evogp_cuda` extension, so there is no CPU path and no import without a built extension.
- Operators are stateful-by-construction strategy objects: implement `__call__` with the exact keyword names (`forest`, `survivor_indices`, `target_cnt`, `fitness` for crossover; `forest` for mutation/selection) and preserve `pop_size`; fitness index tensors are expected to be `int32` and would feed `data_ptr<int>()` CUDA ops.
- Do not add hyperparameter validation asymmetrically: `BaseSelection`/`DefaultSelection` assert, the crossover strategies do not.

## Design Decisions
- Strategy objects instead of subclasses with hooks: selection/crossover/mutation are injected into `GeneticProgramming.__init__`, so composing or replacing an operator is a constructor call, not a code change — this is why every example reads as configuration.
- The loop is deliberately dumb (select → crossover → mutate → concat): all policy lives in the operator objects, which keeps `step` 20 lines and lets the combined (multi-output) variants reuse the identical loop.
- Elites are concatenated by the loop rather than handled by the operators, so no operator has to reason about elitism and the elite block cannot be mutated or recombined away.
- The multi-output (`Combined*`) operators are separate modules because a `CombinedForest` holds one independent `Forest` per output and the plain CUDA ops take single forests; the same parent pairing is reused across sub-forests to keep an individual's outputs aligned.
- The Pareto front exists as an optional side channel (archive of best-per-size) rather than a selection strategy: it gives a size/accuracy trade-off report (`batch_subtree_size[:, 0]` is the complexity measure) without perturbing the standard loop.
- Star re-exports are used so that `from evogp.algorithm import <anything>` works for users; the cost is the flat-namespace traps listed under Known Issues.
- Plain Python classes (no `nn.Module`) because GP here has no learnable parameters — random draws and CUDA kernels do the work.

## Known Issues
- Star-import submodule shadowing: `evogp.algorithm.base` and `evogp.algorithm.default` are bound by ALL THREE subpackages, so the LAST star import (selection) wins and they resolve to `evogp.algorithm.selection.{base,default}`; `from evogp.algorithm.base import BaseCrossover` therefore fails with a confusing `ImportError`, and `import evogp.algorithm.base` raises `ModuleNotFoundError` (it was never a real module path). Always import strategies from their owning subpackage (`evogp.algorithm.crossover.*`, ...). Verified by experiment.
- No class-name COLLISION between the three subpackages (all 27 strategy names are distinct), but two near-identical families coexist at `evogp.algorithm`: `RankSelection`/`RouletteSelection`/`TournamentSelection`/`TruncationSelection` vs `RankSelector`/`RouletteSelector`/`TournamentSelector`/`TruncationSelector`, with the same math duplicated (see `selection/CONTEXT.md`).
- `pareto_front` is a conditional attribute: with `enable_pareto_front=False` (the default) `algorithm.pareto_front` raises `AttributeError` — `example/regressor.py` does exactly this after passing `False`.
- `for_update_pareto_front` in `genetic_programming.py` is dead legacy code (nothing calls it), superseded by `vmap_update_pareto_front`; it also indexes `self.forest.max_tree_len` rather than the passed solution.
- Misspelled module filename `crossover/combined_dafault.py` (class `CombinedDefaultCrossover`); the sibling `mutation/combined_default.py` is spelled correctly. The misspelled path is importable surface — see `crossover/CONTEXT.md`.
- `CombinedMutation` is a NAME TRAP: it chains a list of mutation operators over a plain `Forest` (used this way in `example/regressor.py`); only `CombinedDefaultMutation` handles `CombinedForest`.
- Crossover↔selection signature drift: `BaseCrossover.__call__(forest, survivor_indices, target_cnt, fitness)` vs `DiversityCrossover`/`LeafBiasedCrossover` declaring `(forest, fitness, survivor_indices, target_cnt)` — only keyword calling keeps it working. Details in the child contexts.
- `BaseSelection.__call__(self, forest)` in `selection/base.py` is stale (no `fitness`), while every concrete class and `GeneticProgramming.step` use `(forest, fitness)`; no runtime effect, but misleading for new operator authors.
- Mutation operators mutate their input forest IN PLACE and return the same object, and crossover strategies pass the full population rather than a pre-sliced survivor set — see the child contexts for the per-operator consequences.
- There is no device/shape/dtype validation between stages: a descriptor whose `max_tree_len`/`input_len`/`output_len` does not match the forest trips asserts deep inside `evogp.tree` (hence the `descriptor.update(max_layer_cnt=3)` idiom), and `crossover_rate`/`survival_rate` ranges are mostly unvalidated.

## Notes for Agents
- Read `genetic_programming.py` (121 lines) first — it is the whole contract; then open only the operator module you need (each is 50–140 lines and self-contained).
- To add an operator: new module in the right subpackage, subclass the `Base*` class, implement `__call__`, add the import line to that subpackage's `__init__.py`; the star import at `evogp.algorithm` makes it public automatically (no `__all__` to update).
- Never reorder `algorithm/__init__.py`; keep the three star imports above the `genetic_programming` import.
- `DefaultSelection` + `DefaultCrossover` + `DefaultMutation` is the only combination documented end-to-end in README/examples, and is the only selection that works on CPU — use it as the reference when reasoning about semantics.
- GPU + the compiled `evogp_cuda` extension are required to execute anything here; reasoning about behavior is usually done by reading the code plus `evogp/tree/` and `evogp/cuda/` (the actual kernels are not in this directory).
- Do not "fix" the flat-namespace/star-import design casually: `evogp.algorithm.<Name>` is the API that README, `tutorial/` and `example/` rely on.

## Test Strategy
- No automated tests cover this directory: `test/` holds only GPU smoke scripts and notebooks (`test_bind_success.py` exercises raw CUDA bindings, no assertions), and there is no CI workflow.
- Real coverage is end-to-end — `example/basic.py`, `example/regressor.py`, `example/classifier.py`, `example/transformer.py`, `example/combined_sr.py`, `example/custom_sr.py`, `example/uci_sr.py`, `example/uci_classifier.py`, `example/brax_task.py`, `example/mjx_task.py`, plus `tutorial/evogp_intro.ipynb` and `tutorial/combined_gp.ipynb`.
- Everything requires a CUDA/ROCm GPU and a built extension, so any new test is de-facto an integration test; cheap unit-level checks (build a small `Forest.random_generate` and assert `pop_size`/`forest.pop_size` invariants) are still possible on GPU.
- Coverage gaps: `DiversityCrossover`/`LeafBiasedCrossover` with explicit `*Selector`s, and the four non-default `*Selection` strategies, have no automated or example-level assertions (see the child contexts).

## Routing Table
- `genetic_programming.py` → `GeneticProgramming` loop (`step`, elite concat, Pareto front) — the only file in this node with cross-cutting logic.
- `crossover/` → parent-selection + subtree-swap strategies (`BaseCrossover`, `Default`, `Diversity`, `LeafBiased`, `CombinedDefault`); owns the `crossover/combined_dafault.py` typo and the crypto of `target_cnt` handling.
- `mutation/` → subtree/node perturbation strategies (`Default`, `Hoist`, `Delete`, `Insert`, `Single/MultiPoint`, `Single/MultiConst`, `Combined`, `CombinedDefault` + internal `mutation_utils.py`).
- `selection/` → both the population-level `*Selection` family and the crossover-facing `*Selector` family (`selection_utils.py`).
- `__init__.py` → star re-exports + `GeneticProgramming`; import order is load-bearing.
Sibling paths (READ-ONLY from this node — escalate writes to the parent):
- `../tree/` → `Forest`/`CombinedForest` containers, `GenerateDescriptor`, `NType`, `randint`, `MAX_STACK`; owns the CUDA kernels this layer calls.
- `../pipeline/standard.py` → the only in-package consumer (`evaluate → algorithm.step`, best-tree tracking).
- `../../example/`, `../../tutorial/`, `../../test/` → executable usage and the de-facto test suite; `../../README.md` documents the supported operator set and signatures.
- `../../src/evogp/sr_test.py` → stray runnable end-to-end script inside the package that imports `evogp.algorithm`.
