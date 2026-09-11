# src/evogp/algorithm/mutation

## Intent
Tree-mutation operators for EvoGP: given the freshly crossed offspring `Forest`, perturb a random subset of individuals and return the mutated population.
Every operator is a plain Python class (no `nn.Module`, no registered parameters, no `device` attribute) implementing the duck-typed `BaseMutation.__call__(forest)` contract.
This subpackage is stage 3 of `GeneticProgramming.step`: `selection → crossover → mutation → elite concat`.
Structural work is a pure-torch prefix splice (`Forest.mutate` → `../../tree/splicing.py`), but every operator's randomness and index tensors are hardcoded to the `"cuda"` device string, so there is no CPU path.

## API Surface

### The contract (`base.py`)
- `BaseMutation` exposes only `__call__(self, forest: Forest)`; the base implementation raises `NotImplementedError`.
- Unenforced convention: return a population whose `pop_size` equals the input's.
- Every concrete operator mutates the incoming forest **in place** and returns that same object (no operator is pure/copying).

### Operators

| Class | File | Constructor | Mechanism |
|---|---|---|---|
| `DefaultMutation` | `default.py` | `(mutation_rate: float, descriptor: GenerateDescriptor)` | Replaces the subtree at a uniformly random position with a fresh `Forest.random_generate(...)` tree (canonical subtree replacement). |
| `HoistMutation` | `hoist.py` | `(mutation_rate: float)` | Replaces the subtree at a random position `p1` with the subtree copied from absolute position `p2 ∈ [0, size[p1])`; anti-bloat, needs no descriptor. |
| `DeleteMutation` | `delete.py` | `(mutation_rate: float, max_mutatable_size: Optional[int] = None)` | Picks a random non-leaf node and replaces **its** subtree with the subtree of one of its children, dropping the parent and its other children. |
| `InsertMutation` | `insert.py` | `(mutation_rate: float, descriptor: GenerateDescriptor)` | Generates a fresh random tree `T`, grafts the original subtree into `T` at position `p2 ∈ [1, size[T])`, then writes `T` in place of the original subtree (growth). |
| `SinglePointMutation` | `single_point.py` | `(mutation_rate: float, descriptor: GenerateDescriptor, modify_output: bool = False)` | Rewrites the payload of ONE node while preserving its type class: functions re-drawn from the matching roulette (`roulette_ufuncs`/`roulette_bfuncs`/`roulette_tfuncs`), `VAR` from `[0, input_len)`, `CONST` from `descriptor.const_samples`. |
| `SingleConstMutation` | `single_const.py` | `(mutation_rate: float, descriptor: GenerateDescriptor)` | Same node rewrite restricted to a CONST node; the position comes from a masked argmax over CONST-typed slots. |
| `MultiPointMutation` | `multi_point.py` | `(mutation_rate: float, descriptor: GenerateDescriptor, mutation_intensity: float = 0.3, modify_output: bool = False)` | Applies the `SinglePointMutation` node rewrite to every node passing a Bernoulli(`mutation_intensity`) mask within the tree. |
| `MultiConstMutation` | `multi_const.py` | `(mutation_rate: float, descriptor: GenerateDescriptor, mutation_intensity: float = 0.3)` | Same as `MultiPointMutation` but restricted to CONST nodes. |
| `CombinedMutation` | `combined.py` | `(mutation_operator: list[BaseMutation])` | Applies a list of operators sequentially to one plain `Forest`; documented usage chains `Default` + `Hoist` + `MultiPoint` + `Insert`. |
| `CombinedDefaultMutation` | `combined_default.py` | `(mutation_rate: float, descriptors: Union[List, GenerateDescriptor])` | Multi-output variant: builds one `DefaultMutation(mutation_rate / pattern_num, d_i)` per sub-forest and maps them over a `CombinedForest`. |

### Semantics of the knobs
- `single` vs `multi` = number of touched nodes per selected individual (exactly one vs a Bernoulli-masked subset); `point` vs `const` = whether any node type may be rewritten (point) or only constants (const).
- `mutation_rate` is the per-individual selection probability: `mutate_indices = torch.rand(forest.pop_size) < mutation_rate`.
- `delete.py` is the only operator that ANDs this mask with a structural precondition (`forest.batch_subtree_size[:, 0] > 1`).
- No operator guarantees at least one mutation: `if mutate_indices.sum() == 0: return forest` exits early, returning the unmodified input object.
- Selected individuals are gathered with fancy indexing `forest[mutate_indices]` (a **copy**); results are written back with `forest[mutate_indices] = ...`, i.e. in-place advanced-index assignment on the caller's tensors (`single_*.py` write straight into `forest.batch_node_value[mask, pos]`).
- `mutation_intensity` is the per-node Bernoulli probability inside each selected tree, not an exact count: the mask is `(rand < mutation_intensity) & (arange < tree_size)`, so the number of rewritten nodes varies per tree.
- `MultiPointMutation` rewrites node payloads only — it never changes the tree shape, so the flat layout (and `subtree_size`) stays valid.
- `SinglePoint`/`MultiPoint` bit-pack the multi-output index into the upper 16 bits of the float32 node value (`(func_idx | out_index << 16).view(torch.float32)`); `modify_output=False` keeps the node's existing output index, `True` re-draws it in `[0, output_len)`.

### `mutation_utils.py` (internal helpers, not public)
- `subtensor(tensor: Tensor, start: Tensor, length: Tensor) -> Tensor` — per-row window gather of `length[i]` entries starting at `start[i]`, clamped to the last column and zero-filled past the end; the output shape equals the input shape.
- `vmap_subtree(forest: Forest, pos: Tensor) -> Forest` — for each individual extracts the subtree rooted at `pos[i]` and returns a new `Forest` with the same `input_len`/`output_len`/`max_tree_len`, the subtree left-aligned and the remainder zero-padded.
- Internal only: `__init__.py` does not export it; only `hoist.py`, `delete.py` and `insert.py` import `vmap_subtree`.
- Preserving `max_tree_len` is what lets the result satisfy the shape assertions inside `Forest.mutate`.

### Exports (`__init__.py`)
- Exports `BaseMutation`, `DefaultMutation`, `HoistMutation`, `DeleteMutation`, `InsertMutation`, `SinglePointMutation`, `SingleConstMutation`, `MultiPointMutation`, `MultiConstMutation`, `CombinedMutation`, `CombinedDefaultMutation` (no `__all__`).
- `evogp/algorithm/__init__.py` does `from .mutation import *`, so all of these are public API at `evogp.algorithm.<Name>`.

## Constraints
- Subclasses implement `__call__(self, forest)` and must keep `pop_size` unchanged (`GeneticProgramming.step` concatenates the elites immediately afterwards).
- All tensors are CUDA: the package hardcodes `device="cuda"` 21 times across `default.py`, `delete.py`, `single_*.py`, `multi_*.py`, `mutation_utils.py`; the device is never derived from the input forest.
- The descriptor must match the forest: `Forest.mutate` asserts equal `input_len`, `output_len` and `max_tree_len`, so mutation descriptors are normally obtained via `descriptor.update(max_layer_cnt=3)`, which keeps `max_tree_len` while shrinking generated subtree depth.
- `replace_pos` must have shape `(pop_size,)` but NOT a specific dtype: `check_tensor` only relocates it to `"cuda"` (`../../tree/utils.py:275-283`) and the splice accepts any integer dtype (converted to `long` internally), so int64/int16 positions work too; `delete.py` and `insert.py` nonetheless still `.to(torch.int32)` before calling.
- Imports flow one way: this package imports `evogp.tree` (`Forest`, `CombinedForest`, `GenerateDescriptor`, `NType`, `MAX_STACK`, `randint`); only `evogp.pipeline` (and examples/tutorials) may import `evogp.algorithm`.
- For `CombinedForest` sub-forests, each descriptor's `output_len` must be 1.
- `MAX_STACK` is the hard ceiling on tree length (`descriptor.max_tree_len <= MAX_STACK`), which is why `default.py` uses it as a safe random position bound.

## Design Decisions
- Descriptor-based construction: strategies need the function/terminal pools to synthesize or re-draw nodes, so they hold a `GenerateDescriptor` instead of loose `input_len`/`using_funcs`/`const_samples` arguments.
- `HoistMutation` deliberately takes no descriptor: it only copies subtrees that already exist.
- The combined variants are separate modules because they operate on different containers — `combined.py` chains operators over a single `Forest`, `combined_default.py` exists for the multi-output `CombinedForest` (one sub-forest per output); they are not interchangeable.
- `CombinedDefaultMutation` divides `mutation_rate` by `pattern_num` so the number of edited sub-forests per individual stays near the requested rate instead of being multiplied by the output count.
- Mutation targets offspring only: `GeneticProgramming.step` runs crossover, then `self.mutation(next_forest)`, then appends untouched elites (`self.forest[elite_indices] + next_forest`), so operators never need to exclude elites.
- In-place mutation avoids reallocating the two `(pop_size, max_tree_len)` position tensors per individual; the pipeline hands the operator a throwaway forest and reuses the returned object.

## Known Issues
- Every operator mutates its argument in place and returns the same object, so running two operators on one `Forest` double-mutates it; the `if mutate_indices.sum() == 0: return forest` early exit returns the original object, it is not a copy.
- `HoistMutation` reads `subtree_positions` as an **absolute** position in the flat node array, not as an offset relative to `mutate_positions`; the hoisted subtree therefore usually lies outside the selected subtree (the two coincide only when `mutate_positions == 0`).
- `InsertMutation` draws the graft position in `[1, size[T])`; if the generator produced a single-leaf tree (`size == 1`, possible because the root may become a leaf when `rand < depth2leaf_probs[0]`), `randint(low=1, high=1)` returns 1, which is out of range, and the splice silently copies the target row unchanged because the position lies outside its real tree length (`../../tree/splicing.py`).
- `Forest.mutate` also silently copies the tree unchanged when the replacement would exceed `max_tree_len`, so oversized generated subtrees are dropped without warning.
- `DeleteMutation`'s `max_mutatable_size` is absent from the class docstring: it zeroes the selection weight of slots whose subtree size exceeds the threshold, and a fully masked row falls back to `argmax` = position 0 (the root).
- `SingleConstMutation` uses the same masked-argmax fallback: in a tree without CONST nodes position 0 is picked, but the final `torch.where(node_type == NType.CONST, ...)` guard makes the individual silently unchanged.
- `default.py` samples positions with `torch.randint(low=0, high=MAX_STACK)` and then takes `% tree_size`, so the pool is `MAX_STACK` wide; every other module imports `MAX_STACK` without using it (as with `Optional`, `Tuple`, `Tensor`).
- Selection masks are created on the default (CPU) device in most operators (`torch.rand(forest.pop_size)`), while only `delete.py` pins them with `device="cuda"`, so the code relies on PyTorch accepting CPU index tensors for CUDA operands.
- `single_point.py`/`multi_point.py` assume the multi-output node encoding stores the output index in the upper 16 bits of the float32 value (`output_index << 16`), matching the packing specified in `../../tree/ENGINE_SPEC.md` §2.3 (the `OUT_NODE` tag constant lives in `../../tree/utils.py`); changing that layout breaks these operators.
- `CombinedDefaultMutation` is stateful: `load_pattern_num` rewrites `self.descriptors` into a list and caches `self.pattern_num` on the first call, then asserts the sub-forest count stays the same.
- `randint` (`evogp.tree.utils`) is `low + rand*(high-low)` truncated to an integer dtype, not a true integer RNG; it is only approximately uniform and a `low == high` call silently returns `low`.
- The usage example inside `combined.py`'s docstring is stale: it constructs `MultiPointMutation(mutation_rate=0.2, mutation_intensity=0.3)` without the required `descriptor` argument.

## Notes for Agents
- Read `base.py` (6 lines) plus `GeneticProgramming.step` (`../genetic_programming.py`, line ~117) for the call contract; each operator is 50–140 lines and self-contained.
- Verifying runtime behavior requires a CUDA/ROCm GPU (the operators hardcode `device="cuda"`); there is no compiled extension and nothing to build, so the device is the only missing ingredient on a CPU-only machine.
- To add an operator: subclass `BaseMutation`, give it its own module, and add the import to `__init__.py` — `from .mutation import *` in `evogp/algorithm/__init__.py` makes it public automatically.
- Trust the code over the docstrings for `hoist`, `delete` and `insert`: the docstrings describe canonical GP semantics, while the implementations work on the flat contiguous subtree layout.

## Test Strategy
- No test inside this directory and no test that instantiates the structural/node operators: `DefaultMutation` is the only one covered automatically, indirectly, through the optimizer smoke `../../tests/test_convergence_smoke.py`; `test/` holds the legacy GPU smoke scripts (`test_bind_success.py`, which drives the deleted extension, and `fix_bug.py`, which targets tree evaluation only) plus notebooks.
- Real coverage is end-to-end: `example/basic.py`, `example/regressor.py`, `example/classifier.py`, `example/brax_task.py`, `example/mjx_task.py`, `example/combined_sr.py` plus `tutorial/evogp_intro.ipynb` and `tutorial/combined_gp.ipynb`.
- `CombinedDefaultMutation` is exercised only by `example/combined_sr.py` and `tutorial/combined_gp.ipynb`; the README operator table lists `CombinedMutation` but not `CombinedDefaultMutation`.
- Any new test needs a GPU (the operators build their tensors on `"cuda"`; nothing has to be built), so mutation tests are effectively integration tests; the pytest suite at `../../tests/` covers the underlying `Forest.mutate` splice through `test_operators.py`.

## Routing Table
- `base.py` → the mutation contract (`BaseMutation`).
- `default.py` → canonical subtree-replacement operator (the README/example default).
- `hoist.py`, `delete.py`, `insert.py` → structural subtree-shuffling operators; the only consumers of `mutation_utils.vmap_subtree`.
- `single_point.py`, `multi_point.py` → node-payload rewrite for any node type.
- `single_const.py`, `multi_const.py` → constant-only rewrite.
- `combined.py` → operator chaining over a plain `Forest`.
- `combined_default.py` → multi-output `CombinedForest` support.
- `mutation_utils.py` → internal subtree-slicing helpers.
- No child subdirectories; this node is a leaf.

## See Also
- `../CONTEXT.md` — `algorithm` package (`GeneticProgramming`, `crossover/`, `selection/`).
- `../../tree/CONTEXT.md` — `Forest` (incl. `mutate`/`random_generate`), `CombinedForest`, `GenerateDescriptor`, `NType`, `randint`, `MAX_STACK`.
- `../../tree/splicing.py` (behind `../../tree/CONTEXT.md`) — the pure-torch `mutate` splice that `Forest.mutate` calls.
- `../crossover/` and `../selection/` — siblings producing/consuming the population around mutation (read-only; escalate writes to the parent node).
