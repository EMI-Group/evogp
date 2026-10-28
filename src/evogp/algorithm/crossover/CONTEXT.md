# src/evogp/algorithm/crossover

## Intent
The crossover (recombination) operator subpackage of EvoGP, the GPU-accelerated Tree-based GP framework built on PyTorch.
Each module here implements one interchangeable parent-selection + subtree-swap strategy behind a single `BaseCrossover` interface.
A strategy takes the current `Forest` (whole population) plus the indices of the survivors chosen by the `selection/` subpackage, and returns a brand-new `Forest` of exactly `target_cnt` offspring.
All heavy lifting is delegated to `Forest.crossover`, the pure-torch prefix splice in `../../tree/splicing.py`; this subpackage only decides *which* trees and *which* subtree positions to recombine.

## API Surface
Public classes (no `__all__` anywhere; `crossover/__init__.py` imports all five, and `algorithm/__init__.py` does `from .crossover import *`, so every name below is reachable as `evogp.algorithm.<Name>` and importable from `evogp.algorithm.crossover`):
- `base.py` → `BaseCrossover`: the contract, `__call__(self, forest: Forest, survivor_indices: torch.Tensor, target_cnt: int, fitness: torch.Tensor) -> Forest`, body is `raise NotImplementedError`.
- `default.py` → `DefaultCrossover()`: no constructor arguments; uniformly random recipient/donor from the survivors, uniform subtree positions in both parents, 100% crossover.
- `diversity.py` → `DiversityCrossover(crossover_rate=0.9, recipient_selector=None, donor_selector=None)`: receptor/donor pair selection is pluggable via `BaseSelector` instances; `crossover_rate` is the fraction of offspring produced by crossover, the rest are copied unchanged.
- `leaf_biased.py` → `LeafBiasedCrossover(crossover_rate=0.9, leaf_bias=0.3, recipient_selector=None, donor_selector=None)`: same parent-selection scheme as `DiversityCrossover` plus a `leaf_bias` probability that the exchanged subtree pair is chosen among *leaf* (terminal) nodes of both parents.
- `combined_dafault.py` → `CombinedDefaultCrossover()`: no constructor; multi-output variant that operates on `CombinedForest` (one `Forest` per output) instead of `Forest`, otherwise identical to `DefaultCrossover`.
- `BaseSelector` (from the sibling `selection/` package) is *not* re-exported here; import it from `evogp.algorithm.selection`.

## Call Contract
`GeneticProgramming.step` (`algorithm/genetic_programming.py`) drives this subpackage: `elite_indices, next_indices = self.selection(self.forest, fitness)`, then `next_forest = self.crossover(forest=self.forest, survivor_indices=next_indices, target_cnt=self.pop_size - elite_indices.shape[0], fitness=fitness)`, then `next_forest = self.mutation(next_forest)`, then `self.forest = self.forest[elite_indices] + next_forest`.
Therefore a strategy MUST:
- return exactly `target_cnt` offspring (a new object, not a view of the parent),
- NOT include the elites (they are concatenated afterwards by `step`),
- build offspring from the parent `forest` (it receives the full population, not a pre-sliced survivor set) and honour `survivor_indices`,
- index parents with `int32` tensors and positions with `int32` tensors (`Forest.crossover` asserts all four args are 1-D of equal length `res_forest_size` and returns a `Forest` of that size).
`survivor_indices` are int32 indices into the whole population; with the default `DefaultSelection` they are the top-`survival_rate` ranked individuals.
## Fitness Contract (the `fitness` argument)
`fitness` is READ-ONLY in this subpackage: no strategy writes into it, stores it (or a view/slice of it), or converts its device.
- Never read: `DefaultCrossover` (`default.py:21`, docstring `default.py:30`) and `CombinedDefaultCrossover` (`combined_dafault.py:13`) declare the parameter and ignore it entirely; `base.py:11` only declares it.
- Read only when a selector is supplied: `DiversityCrossover` (`diversity.py:63`, `diversity.py:78`) and `LeafBiasedCrossover` (`leaf_biased.py:67`, `leaf_biased.py:80`) forward it as `self.<x>_selector(fitness, crossover_cnt)` with the same tensor object; with the default `recipient_selector=None`/`donor_selector=None` they never touch `fitness` either. No `fitness[...] = ...`, augmented assignment, in-place `*_` op, `out=` or `.cuda()/.to()/.cpu()` on `fitness` exists anywhere in this subpackage.
- No retention: the only instance attributes ever assigned are `crossover_rate`/`leaf_bias`/`recipient_selector`/`donor_selector` (`diversity.py:33-35`, `leaf_biased.py:37-40`); there is no `self.fitness` and no global holding it, and the selectors likewise store only hyperparameters (`../selection/selection_utils.py:22,43,60-68,119-120`). A caller-recycled fitness buffer is therefore safe for crossover — it is sampled synchronously within the single `GeneticProgramming.step` call (`../genetic_programming.py:110-116`).
- Shape/dtype: this subpackage makes no assert, `.size(0)` or sort assumption on `fitness`; the only shape guarantee is `GeneticProgramming.step`'s `fitness.shape == (pop_size,)` assert (`../genetic_programming.py:103-105`). Selector output must be full-population indices because `DiversityCrossover`/`LeafBiasedCrossover` index `forest` (not `survivor_indices`) with them (`diversity.py:91-111`; `leaf_biased.py:109-138`).
- Device: all fitness device handling happens inside the selectors (`.cuda()` at `../selection/selection_utils.py:26,47,82,97-98,126`), never here; `RankSelector`/`TruncationSelector`/`TournamentSelector` gather from fitness-device tensors indexed by CUDA index tensors (`selection_utils.py:33,87,93,130`) so they are consistent only for an already-CUDA `fitness`, while `RouletteSelector` copies the probability tensor to CUDA first (`selection_utils.py:46-47`) but requires float, non-negative, finite fitness. Selector output must be a 1-D integer index tensor feeding `size_tensor[indices, 0]` and `forest.crossover` (`diversity.py:95-111`; `leaf_biased.py:117-138`): `Forest.crossover` runs `check_tensor` on all four arguments — which only relocates them to `"cuda"` and detaches, without any dtype cast (`../../tree/utils.py:275-283`) — and then asserts each is 1-D of length `res_forest_size` (`../../tree/forest.py:217-228`); the splice itself converts any integer dtype to `long` internally, so an `int64` selector output is fine.

## Constraints
- Subclass `BaseCrossover` and implement `__call__` with the four keyword names `forest`, `survivor_indices`, `target_cnt`, `fitness` (argument *order* must not be relied on — see Known Issues).
- CUDA is assumed everywhere: every strategy calls `torch.randint(..., device="cuda")` explicitly, and the `randint` helper imported from `...tree` defaults to `device="cuda"` as well; this subpackage cannot run on CPU.
- Only touch the parent forest through its public API (`forest[indices]`, `forest.crossover(...)`, `forest + forest`); never mutate `batch_node_value` / `batch_node_type` / `batch_subtree_size` in place.
- Offspring are produced only by the `Forest.crossover` splice (`../../tree/splicing.py`); do not reimplement subtree swapping in this subpackage.
- Unlike the `selection/` classes, no strategy here validates its hyperparameter ranges (`crossover_rate`, `leaf_bias`) with asserts.

## Design Decisions
- Parent choice is separated from recombination: `DiversityCrossover` / `LeafBiasedCrossover` take `BaseSelector` objects (the `RankSelector` / `RouletteSelector` / `TournamentSelector` / `TruncationSelector` family whose API is `__call__(fitness, choosed_num) -> Tensor`) so "which parents" (exploration pressure) can be tuned independently of "how to recombine", and a different selector can be used for the receptor vs the donor.
- `crossover_rate < 1` exists so only part of the offspring is recombined and the remainder is copied verbatim from the population, limiting disruption of good individuals.
- A leaf-biased variant exists because swapping internal subtrees can change a tree's semantics violently; swapping terminals (constants/variables) gives smoother, more local numeric search — the classic "one-point crossover is disruptive" trade-off.
- `DefaultCrossover` is intentionally parameter-free so it can be used as the drop-in default in every example and in the README quickstart.
- The combined variant exists because the multi-output path stores one independent `Forest` per output inside `CombinedForest`, which `Forest.crossover` cannot handle; it applies the same pairing to every sub-forest so the recombined individuals stay aligned across outputs.

## File-by-file Mechanism
- `base.py`: contract only; no logic, no `__init__`.
- `default.py`: `survivor_forest = forest[survivor_indices]`, sample `(2, target_cnt)` random survivor indices, sample two int32 positions in `[0, int32.max)` and reduce them modulo `survivor_forest.batch_subtree_size[:, 0]` (each parent's own tree size) to get in-range crossover points, then `survivor_forest.crossover(left_indices, right_indices, left_pos, right_pos)`.
- `diversity.py`: `crossover_cnt = int(target_cnt * crossover_rate)`; receptor and donor index arrays of length `crossover_cnt` come either from `recipient_selector(fitness, crossover_cnt)` / `donor_selector(fitness, crossover_cnt)` or from a uniform random draw mapped through `survivor_indices`; both positions are drawn with `randint(low=0, high=batch_subtree_size[idx, 0])`; result is `crossovered_forest + static_forest`.
- `leaf_biased.py`: identical to `diversity.py` except positions: a nested `choose_leaf_pos` masks a random score with `subtree_size == 1` (leaf marker) and takes `argmax`, two independent position candidates (leaf and non-leaf) are computed, and a shared `leaf_bias` mask picks leaf/leaf vs non-leaf/non-leaf for each pair — so both parents are always biased the same way per offspring.
- `combined_dafault.py`: slices `survivor_forest = forest[survivor_indices]` (returns a `CombinedForest`), draws one shared `left_indices`/`right_indices` pair, then per output `i` draws positions modulo that sub-forest's tree sizes and calls `survivor_forest.forests[i].crossover(...)`, returning `CombinedForest(new_forest, forest.data_info)`.

## Known Issues
- Filename typo: `combined_dafault.py` (should be `default`); the class inside is correctly named `CombinedDefaultCrossover`. The *module path* `evogp.algorithm.crossover.combined_dafault` is part of the importable surface — the only importer is `crossover/__init__.py`, so a rename must update that line (and the sibling `mutation/combined_default.py` uses the correct spelling, so the two packages are inconsistent).
- `DiversityCrossover.__call__` and `LeafBiasedCrossover.__call__` declare their parameters as `(forest, fitness, survivor_indices, target_cnt)`, whereas `BaseCrossover` declares `(forest, survivor_indices, target_cnt, fitness)`; it only works because `GeneticProgramming.step` passes keyword arguments — a positional call following the base signature would silently misbind (fitness↔survivor_indices) and crash or produce garbage.
- In `diversity.py` and `leaf_biased.py` the non-crossover remainder is built as `static_forest = forest[random_indices]` with `random_indices` sampled in `[0, len(survivor_indices))` — those are raw population rows, not `survivor_indices[...]`, so the copied individuals are not guaranteed to be survivors (the correct pattern is used in `default.py`/`combined_dafault.py` via `forest[survivor_indices]`).
- When a `*Selector` is supplied, its output indexes the full population (the selector sees full-length `fitness`), so elites and non-survivors can act as receptor/donor; only the `None` fallback restricts parents to survivors.
- No strategy prevents a tree from being selected as its own donor, so self-crossover (a clone) occurs with ~`1/n_survivors` probability per offspring.
- `crossover_rate` is annotated `int` but used as a float ratio; it is unvalidated, and `crossover_rate > 1` makes `target_cnt - crossover_cnt` negative, which makes `torch.randint` raise.
- `choose_leaf_pos` guarantees a leaf only probabilistically: it returns `torch.argmax` of a masked random score, so if every leaf score ties at 0 the returned position 0 is a non-leaf node.
- `MAX_STACK` is imported from `...tree` in `default.py`, `diversity.py` and `leaf_biased.py` but never used (dead import).
- Importing this subpackage transitively imports `evogp.tree`, which needs only torch and the `evoxc` package — there is nothing to build, so importing always succeeds; only *executing* a strategy needs a CUDA-visible device.

## Notes for Agents
- To add a strategy: copy `default.py`, subclass `BaseCrossover`, slice `forest[survivor_indices]`, and return exactly `target_cnt` offspring (never include the elites).
- Keep the `survivor_indices`-relative indexing convention used by `default.py` (`survivor_forest = forest[survivor_indices]` then index the slice) rather than indexing `forest` directly.
- Always pass `device="cuda"` when creating new tensors here, matching the rest of the codebase.
- Do not "fix" the `combined_dafault.py` filename in isolation: `crossover/__init__.py` is the only importer, but external user code may import the misspelled module path.
- Any real run needs a CUDA device (every index/position tensor is created on `"cuda"`); no compiled extension and no build step are involved, but there is still no CPU fallback for the random draws.

## Test Strategy
- There are no test files inside this directory and the repository has no CI workflow (`.github/workflows` is absent).
- The automated regression gate is `../../tests/test_operators.py` (`Forest.crossover` splice invariants and silent-copy edge cases) plus `../../tests/test_equivalence.py` (evaluation).
- `test/combined_test.ipynb`, `test/vis.ipynb`, `tutorial/combined_gp.ipynb`, `tutorial/evogp_intro.ipynb` exercise `CombinedDefaultCrossover` / `DefaultCrossover` / `LeafBiasedCrossover` end-to-end inside a full pipeline.
- `src/evogp/sr_test.py` is a runnable end-to-end smoke script using `DefaultCrossover` (needs CUDA).
- Coverage gaps: `DiversityCrossover` with selectors and `LeafBiasedCrossover` have no automated test (`../../tests/test_operators.py` exercises only the `Forest.crossover` splice they call); a cheap check for a new strategy is to build a small `Forest.random_generate(...)` and assert the returned `forest.pop_size == target_cnt` (needs a CUDA device).

## Routing Table
This directory has no child subdirectories — route by file:
- `base.py` → abstract `BaseCrossover` contract (signature, `NotImplementedError`).
- `default.py` → `DefaultCrossover`, random pairing + uniform subtree swap (the de-facto default).
- `diversity.py` → `DiversityCrossover`, selector-driven receptor/donor + `crossover_rate` static copies.
- `leaf_biased.py` → `LeafBiasedCrossover`, adds `leaf_bias` leaf↔leaf crossover points.
- `combined_dafault.py` → `CombinedDefaultCrossover`, multi-output (`CombinedForest`) variant of `DefaultCrossover`.
- `__init__.py` → the five re-exports and nothing else.
Related paths (read-only from here — escalate writes to the parent node):
- `../selection/selection_utils.py` → `BaseSelector`/`RankSelector`/`RouletteSelector`/`TournamentSelector`/`TruncationSelector`, the `recipient_selector`/`donor_selector` implementations consumed here.
- `../genetic_programming.py` → the only in-tree caller of `BaseCrossover.__call__`.
- `../../tree/forest.py`, `../../tree/combined_forest.py`, `../../tree/utils.py`, `../../tree/splicing.py` → `Forest`/`CombinedForest`, `crossover()`, `randint`, `MAX_STACK`, and the pure-torch splice `crossover()` calls.
- `../../../../example/`, `../../../../tutorial/`, `../../../../test/` → usage examples, notebooks and the legacy CUDA-extension scripts.
