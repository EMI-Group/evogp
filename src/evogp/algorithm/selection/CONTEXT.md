# Selection Operators — `evogp.algorithm.selection`

## Intent
This subpackage serves two distinct roles that share a name but not an interface.
Role 1 — **population-level selection** for the GP loop: the `*Selection` classes decide which individuals survive (become the parent pool) and which are copied unchanged as elites.
Role 2 — **lightweight fitness-proportional index sampling**: the `*Selector` classes return `choosed_num` parent indices on demand, used only as the optional `recipient_selector` / `donor_selector` of `DiversityCrossover` and `LeafBiasedCrossover`.
The `*Selection` and `*Selector` families are independent copies of the same algorithms; neither imports the other (see Design Decisions).

## API Surface

### Family 1 — population-level `*Selection` (subclass `BaseSelection`)
All return a 2-tuple `(elite_indices, survivor_indices)`; see "Real selection contract" below.

| Class | File | Ctor (defaults) | Mechanism |
|---|---|---|---|
| `BaseSelection` | `base.py` | no `__init__` | Abstract; sole method is the (stale) `__call__` below. |
| `DefaultSelection` | `default.py` | `survival_rate=0.3, elite_cnt=None, elite_rate=None` | Pure truncation by fitness: sort descending, survivors = first `int(pop*survival_rate)`, elites = first `elite_cnt` (or `int(pop*elite_rate)`, else 0). |
| `RankSelection` | `rank.py` | `selection_pressure=0.5, survivor_rate=0.5, elite_rate=0, survivor_cnt=None, elite_cnt=None` | Linear-ranking roulette with replacement for survivors (formula in docstring); elites = top fitness. |
| `RouletteSelection` | `roulette.py` | `survivor_rate=0.5, elite_rate=0, survivor_cnt=None, elite_cnt=None` | Fitness-proportional (`fitness/sum(fitness)`) sampling with replacement for survivors; elites = top fitness. |
| `TruncationSelection` | `truncation.py` | `survivor_rate=0.5, elite_rate=0, survivor_cnt=None, elite_cnt=None` | Intended: uniform sampling from the top `int(pop*survivor_rate)`; the mask is buggy (see Known Issues) so truncation-by-fitness does not actually happen. |
| `TournamentSelection` | `tournament.py` | `tournament_size` (REQUIRED, no default), `best_probability=1, replace=True, survivor_rate=0.5, elite_rate=0, survivor_cnt=None, elite_cnt=None` | `torch.vmap`-batched tournaments: uniform contenders, winner = best of the tournament with prob. `best_probability` (geometric `log(rand)/log(1-best_p)` pick); elites = top fitness. |

### Family 2 — `*Selector` (subclass `BaseSelector`, all defined in `selection_utils.py`)
Contract: `__call__(self, fitness: Tensor, choosed_num: int) -> Tensor`, returning `choosed_num` indices into the **full population** fitness array (space `0..pop_size-1`, with replacement, duplicates allowed).
That is the same index space the crossover fallback produces via `survivor_indices[randint(...)]`, so selector output can be fed straight to `forest[...]` / `forest.crossover(...)`.

| Class | Ctor (defaults) | Mechanism | Intended consumer |
|---|---|---|---|
| `BaseSelector` | abstract | Declares `__call__(fitness, choosed_num)`. | `DiversityCrossover` / `LeafBiasedCrossover` (`recipient_selector`, `donor_selector`). |
| `RankSelector` | `selection_pressure=0.5` | Linear-rank probabilities + `multinomial`; then maps through `sorted_indices`. | Same; has the same rank bug as `RankSelection`. |
| `RouletteSelector` | `()` — no args | `multinomial(fitness/sum(fitness))`; returns the drawn positions directly (no sort/gather). | Same; the only selector that returns int32 correctly. |
| `TournamentSelector` | `tournament_size` (REQUIRED), `best_probability=1, replace=True` | Same vmap tournament code as `TournamentSelection`. | Same; `t_size > 1000` branch identical. |
| `TruncationSelector` | `survivor_rate=0.5` | Intended top-`survivor_rate` uniform sampling; same broken mask as `TruncationSelection`. | Same; has no `survivor_cnt`/`elite_*` args at all. |

### Real selection contract (verified against `../genetic_programming.py:110-118`)
- `BaseSelection.__call__(self, forest: Forest) -> Tuple[Tensor, Tensor]` in `base.py` is **stale**: it omits `fitness` and is not what any concrete class implements.
- Every concrete `*Selection.__call__(self, forest: Forest, fitness: torch.Tensor)` ignores `forest` except for `forest.pop_size`, and `GeneticProgramming.step` calls `self.selection(self.forest, fitness)`.
- The loop then does `crossover(forest, survivor_indices=next_indices, target_cnt=pop_size - elite_indices.shape[0], fitness=fitness)` and finally `self.forest = self.forest[elite_indices] + next_forest`.
- Therefore: `elite_indices.shape[0]` must equal the intended elite count (it sets the offspring budget); `survivor_indices.size(0)` must equal the intended parent-pool size (the crossover fallback samples uniformly with `high=survivor_indices.size(0)`); both must hold valid population indices and together must reproduce `pop_size`.
- Selection **pressure is carried by multiplicity**, not order: because crossover resamples survivors uniformly by position, the relative frequency of an individual inside `survivor_indices` is what biases it (this is why rank/roulette/tournament draw with `replacement=True`).
- Direction is never validated: all five `*Selection` strategies and all four `*Selector`s rank by **HIGHER fitness = better** (`torch.sort(..., descending=True)` / `torch.argmax` / `torch.argsort(..., descending=True)`), so a caller passing an error-not-negated tensor gets the wrong optimization direction.
- No operator here writes into the `fitness` tensor (`torch.sort` is out-of-place; nobody assigns into `fitness`); they read it and return index tensors — the tensor may be mutated in place only by the upstream pipeline (`../pipeline/standard.py:43`, NaN→`-inf`).

### Fitness tensor contract (verified against all eight operator files)
- Every operator only READS `fitness` (`torch.sort`/`argmax`/`argsort`/advanced indexing/`fitness / torch.sum(fitness)`); there is no `fitness[...] = ...`, no in-place method and no `out=`, and no operator stores `fitness` in `self` or a module global — the tournament closures capture it only for the duration of one `__call__`, so a caller may recycle the same fitness buffer across generations.
- `fitness` is assumed 1-D `(pop_size,)`: `torch.sort(fitness, descending=True)` uses the default `dim=-1` (a 2-D input sorts per row instead of raising), `fitness.size(0)` is the population (tournament.py:113, selection_utils.py:72/124) and `fitness[contenders]` indexes dim 0; the shape is asserted by the caller, not here (`genetic_programming.py:103-105`).
- Only `RouletteSelection`/`RouletteSelector` use fitness *magnitudes*: they pass `fitness/torch.sum(fitness)` to `torch.multinomial`, which raises on negative or non-finite weights — incompatible with the pipeline's in-place `-inf` NaN sanitization (`pipeline/standard.py:43`); all other operators only sort/argmax and tolerate `-inf`.
- Device: `DefaultSelection` is device-preserving (both outputs are `.to(dtype=torch.int32)` casts of the sort output, `default.py:68-69`); every other operator forces CUDA — see the Hardcoded CUDA Known Issue for what that means for a CPU `fitness` tensor.
- Returned dtypes are mixed (int32 for both outputs only from `DefaultSelection`; the rank/truncation gather and every non-default elite slice are int64) — see the dtype-drift Known Issue for what happens when an int64 index reaches the kernels.

## Constraints
- New population-level operators must subclass `BaseSelection`; new lightweight samplers must subclass `BaseSelector`.
- Implementations should return `torch.int32` index tensors, but this is only reliably true for `DefaultSelection` (see Known Issues).
- This subpackage must **not** import from `../crossover/` — `crossover/diversity.py` and `crossover/leaf_biased.py` do `from ..selection import BaseSelector`, so a reverse import creates a cycle.
- `evogp.tree.Forest` is the only dependency to import from outside the package, and only `DefaultSelection`/`Rank`/`Roulette`/`Tournament`/`Truncation` need it (for `pop_size`).
- Device is hardcoded to CUDA in every implementation except `DefaultSelection`; any new operator should avoid adding more `.cuda()` calls.

## Design Decisions
- Two interfaces exist because the two callers need different things: the GP loop needs both an elite set (fixed-size copy budget) and an ordered/variable-size parent pool, while crossovers only need "give me N parents by fitness" on demand.
- `BaseSelector` and its four implementations live in the single module `selection_utils.py` (not a dedicated `base.py`) because they are treated as an internal utility of the crossover operators — but they are still exported from `__init__.py` so `from ..selection import BaseSelector` (the crossover import) works and users can subclass them.
- `*Selection` deliberately reimplements rather than reuses `*Selector`: `rank.py`, `roulette.py`, `truncation.py`, and `tournament.py` duplicate the selector math inline (e.g. `t_selection_with_p` is copy-pasted verbatim), so bug fixes must be applied twice.
- `DefaultSelection` is a *destructive* truncation (it hands `survivor_indices` to the crossover, which resamples them uniformly), which is why it needs no probability math.
- `elite_cnt` is `Optional[int]` and defaults to `None` (0 elites) so that "no elitism" is the library default; README/examples always pass `elite_rate=0.01` or similar explicitly.

## Known Issues
- **Stale base signature**: `BaseSelection.__call__(self, forest)` declares neither the `fitness` argument nor the concrete `(forest, fitness)` arity; no runtime effect (Python does not check overrides) but it misleads anyone writing a custom selection or type-checking with mypy.
- **Rank vector bug (both families)**: `rank = sorted_indices.cuda()` (rank.py:81, selection_utils.py:26) uses the sort's *value* permutation where a rank vector `torch.arange(n)` is required; probabilities are therefore assigned by original population index, so `RankSelection`/`RankSelector` pressure is effectively uncorrelated with fitness.
- **Truncation mask bug (both families)**: `(sorted_indices < num_selectable)` (truncation.py:72, selection_utils.py:126) compares index *values* against a count instead of comparing rank *positions*; the "top survivors" pool is not the fittest individuals, so neither truncation class truncates as documented.
- **dtype drift**: `DefaultSelection` returns int32 for both outputs; the `sorted_indices[random_indices]` gather promotes rank/truncation survivors to int64; every elite output except `DefaultSelection`'s comes from `sorted_indices[:elite_cnt]` (int64), and the `elite_cnt == 0` early-outs build explicit int64 empty tensors (`roulette.py:72`, `tournament.py:127`). `Forest.__getitem__` accepts any integer Tensor, so the GP loop's `forest[elite_indices]` / `forest[survivor_indices]` indexing does not crash, but the int32 contract in docstrings is not upheld, and an int64 index forwarded straight into `Forest.crossover` — the `DiversityCrossover`/`LeafBiasedCrossover` `recipient_indices`/`donor_indices`, built from `RankSelector`/`TruncationSelector` output or from `survivor_indices[random_indices]` when the population selection was `RankSelection`/`TruncationSelection` — lands in the typed `data_ptr<int>()` casts of `cuda/torch_wrapper.cu:180-183`, which enforce the scalar type and therefore throw for int64 (`check_tensor` validates device/contiguity/shape only, `torch_wrapper.cu:7-17`).
- **Hardcoded CUDA**: `.cuda()` / `device="cuda"` appear in rank.py, roulette.py, tournament.py, truncation.py and throughout `selection_utils.py`; only `default.py` is device-agnostic (it uses device-preserving `.to(dtype=torch.int32)`), so CPU-only runs must use `DefaultSelection`.
- **`best_probability=1` is a degenerate default**: `log(1 - best_p) == -inf`, so the geometric pick relies on `log(rand)/-inf → 0`; it happens to select the tournament best, but `torch.rand` returning exactly 0 would yield NaN and an undefined int cast.
- **`t_size > 1000` magic branch** (tournament.py:120, selection_utils.py:106): silently ignores `best_probability` and always picks the max-fitness contender; the threshold is undocumented.
- **`t_size > pop_size` divides by zero**: `n_tournament = int(total_size/t_size)` becomes 0 and `k_times = int((cnt-1)/n_tournament)` raises `ZeroDivisionError`.
- **Naming inconsistency**: `DefaultSelection` takes `survival_rate` while every other class takes `survivor_rate`; mixing them up raises `TypeError`.
- **`survivor_cnt` is only half-honored** in the truncation classes: `num_selectable` is recomputed from `survivor_rate` (truncation.py:68), so an explicit `survivor_cnt` changes how many are drawn but not the eligible pool.
- **Roulette variants assume non-negative fitness**: they pass raw `fitness/sum(fitness)` to `torch.multinomial`, which requires non-negative, finite, positive-sum weights.
- **Tournament bookkeeping is dead weight**: `k_times`/`n_tournament` size the uniform `p` tensor, then `reshape(-1, t_size)[:survivor_cnt]` discards most drawn contenders, so the outcome is plain uniform sampling.
- **`*Selector` classes have zero in-repo instantiation sites**: they are never used in `example/`, `test/`, or `tutorial/`, and neither is `BaseSelector` subclassed by user code, so this family is effectively untested public API.
- Downstream consumers do not strictly re-derive survivors: `crossover/diversity.py:122` indexes the static-copy path with `forest[random_indices]` over the *full population* range — see `../crossover/CONTEXT.md`, do not "fix" it here.

## Notes for Agents
- To add a population-level operator: create `<name>.py`, subclass `BaseSelection`, implement `__call__(self, forest, fitness) -> (elite_indices, survivor_indices)` with int32 outputs, then add the import line to `__init__.py`.
- `__init__.py` has no `__all__` and imports all 11 names; since `evogp/algorithm/__init__.py` does `from .selection import *`, everything added here becomes public API at `evogp.algorithm.<Name>` automatically.
- Read `default.py` first — it is the only implementation that is both device-agnostic and dtype-correct, and it is the one used by README, `tutorial/`, and 9 of 10 `example/` scripts.
- Do not "clean up" the stale `BaseSelection.__call__` in isolation: `GeneticProgramming` annotates `selection: BaseSelection` and calls it with two args, so widening the base signature is safe, narrowing it is not.
- Ignore `../selection/`'s apparent helpers: there is no shared utility module for rank/roulette/tournament math, only duplicated copies.

## Test Strategy
- No automated tests cover selection: `test/` contains only `test_bind_success.py`, `test_torch_env.py`, notebooks, and `fix_bug.py`; `test/*.py` never mention selection.
- `src/evogp/sr_test.py` is a stray runnable script inside the package (not a pytest test) and uses `DefaultSelection`.
- Practical verification is manual: `example/basic.py`, `example/regressor.py`, `example/classifier.py`, `example/transformer.py`, `example/combined_sr.py`, `example/brax_task.py`, `example/mjx_task.py`, `example/uci_sr.py` (the latter three plus `example/custom_sr.py` exercise `TournamentSelection`), plus `tutorial/evogp_intro.ipynb`.
- All of those require a CUDA GPU, so any fix touching the non-default classes cannot be regression-tested with `DefaultSelection`-based examples.

## Routing Table (no child subdirectories — route by file)
- `base.py` → Abstract `BaseSelection` (its `__call__` signature is stale — see Known Issues).
- `default.py` → `DefaultSelection`, the README/example default and the only device-agnostic, int32-correct implementation.
- `rank.py` → `RankSelection` (linear ranking + selection pressure).
- `roulette.py` → `RouletteSelection` (fitness-proportional).
- `tournament.py` → `TournamentSelection` (vmap tournaments, required `tournament_size`).
- `truncation.py` → `TruncationSelection` (fitness truncation, implemented buggily).
- `selection_utils.py` → `BaseSelector`, `RankSelector`, `RouletteSelector`, `TournamentSelector`, `TruncationSelector` (the crossover-facing family).
- `../genetic_programming.py` → The only population-level caller (`GeneticProgramming.step`); sibling — read-only, escalate writes to the parent.
- `../crossover/` → Callers of `BaseSelector` (`diversity.py`, `leaf_biased.py`); sibling — read-only, escalate writes to the parent.
- `../../tree/forest.py` → `Forest`, the population container (`pop_size`, indexing, `forest.crossover`); sibling — read-only, escalate writes to the parent.
