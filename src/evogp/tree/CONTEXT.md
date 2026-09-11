# src/evogp/tree — Core tree representation & GPU execution layer

## Intent
This is the **foundation layer** of EvoGP: the GP individual (`Tree`), the population of individuals (`Forest`), the multi-output variants (`CombinedTree` / `CombinedForest`), the random-generation blueprint (`GenerateDescriptor`), and the enums/helpers they share.
Everything the rest of the library manipulates (fitness, selection, crossover, mutation, visualization) is expressed in terms of the tensor representation defined here.
This package imports **nothing** from other `evogp` subpackages — it is the root of the package DAG: `algorithm/` and `problem/` are its two independent consumers and `pipeline/` sits above both (no cycles).
All heavy work is batched and lives on the GPU; there is no CPU fallback anywhere in this layer.

## Representation (the schema every other module depends on)
An individual is three parallel tensors of length `max_tree_len`, encoded as a **prefix (pre-order) expression**: node 0 is the root, children follow depth-first.
- `node_value`   — `torch.float32`, batched `[pop_size, max_tree_len]` (`Forest.batch_node_value`).
- `node_type`    — `torch.int16`,  batched `[pop_size, max_tree_len]` (`Forest.batch_node_type`).
- `subtree_size` — `torch.int16`,  batched `[pop_size, max_tree_len]` (`Forest.batch_subtree_size`); `subtree_size[i]` = number of nodes in the subtree rooted at node `i`.
Only indices `[0, subtree_size[0])` are meaningful: `subtree_size[0]` is the real tree length and the rest of the row is uninitialized garbage (the CUDA generator uses `torch::empty` and writes only the used prefix) — never read or compare beyond it.
`batch_subtree_size[:, 0]` is used throughout the algorithm layer as "the size of tree i".
`Tree` holds the same three tensors unbatched (1-D, length `max_tree_len`) as `node_value` / `node_type` / `subtree_size`.
`Forest[i]` / `Forest[a:b]` return `Tree` / `Forest` objects that **share storage** (basic indexing views) with the parent — mutations write through.

Node encoding:
- `node_type` base values: `0 VAR`, `1 CONST`, `2 UFUNC`, `3 BFUNC`, `4 TFUNC` (`NType` in `utils.py`); `NType.OUT_NODE = 1<<7` (128) is OR-ed in for multi-output nodes; `NType.TYPE_MASK = 0x7F` extracts the base type.
- `node_value` for `VAR` = variable index (float-encoded int), for `CONST` = the constant itself, for a function node = the function id (`Func.*`), i.e. its index in `FUNCS_NAMES`.
- Multi-output out-node packing: for an out-node, `node_value` is the float32 **bit-reinterpretation** of `struct OutNodeValue { int16_t function; int16_t outIndex; }` (`../cuda/kernel.h`) — hence `outIndex = int32_bits >> 16` and `function = int32_bits & 0xFF` (see `Tree._fillout_graph`, `Tree.to_sympy_expr`, `../cuda/forward.cu:_process_node`).
- Multi-output semantics (single tree, `output_len > 1`): every out-node's value is **accumulated** (`s_outs[outIdx] += value`) and its right-most child is passed up to the parent. A `Forest` with `output_len > 1` therefore computes a *sum of tagged subtrees* per output, not an independent tree per output.

Tensor contract enforced by the extension (`../cuda/torch_wrapper.cu`): values `float32`, types/sizes `int16`, contiguous, on CUDA, with exact expected shapes (`[pop_size, max_tree_len]`, `[pop_size, input_len]`, …). The C++ `check_tensor` verifies only `is_cuda() && is_contiguous()` + shape — **not dtype** — so a wrong dtype is silently misread as garbage.

## API Surface
`descriptor.py`
- `GenerateDescriptor(max_tree_len, input_len, output_len, const_prob=0.5, out_prob=0.5, depth2leaf_probs=None, roulette_funcs=None, const_samples=None, using_funcs=None, max_layer_cnt=None, layer_leaf_prob=0.2, const_range=None, sample_cnt=None)`.
- Validates inputs (see Constraints) and materializes GPU tensors: `depth2leaf_probs [MAX_FULL_DEPTH]` (probability that a node at depth `d` is a *leaf*; the kernel takes the function branch when `rand >= depth2leaf_probs[depth]`), `roulette_funcs [Func.END]` (cumulative cdf used by the CUDA generator), `roulette_ufuncs/roulette_bfuncs/roulette_tfuncs [Func.END]` (cdfs used only by the torch-side mutation operators), `const_samples [sample_cnt]`.
- `update(**kwargs)` → a **new** descriptor with kwargs applied (see Known Issues).
- `check_tree_length()` (module function) derives `depth2leaf_probs` and asserts `max_tree_len` is big enough for the worst-case full tree of `max_layer_cnt` layers over the given `using_funcs`.

`utils.py` — flat module of shared constants/enums/helpers (all imported by `algorithm/` too):
- `MAX_STACK = 1024`, `MAX_FULL_DEPTH = 10`, `DELTA`, `MAXVAL`.
- `NType` (VAR/CONST/UFUNC/BFUNC/TFUNC, `TYPE_MASK`, `OUT_NODE`, `*_OUT`), `Func` (TF 0 `IF`; BF 1..13 `ADD..GE`; UF 14..28 `SIN..LOOSE_SQRT`; `END = 29`), `FUNCS`, `FUNCS_NAMES` (index == function id), `FUNCS_DISPLAY` (unicode minus for `-`), `SYMPY_MAP`, `LooseDiv`/`LooseInv`/`LooseLog` (sympy `sp.Function`s).
- `dict2prob(prob_dict)` → normalized probability vector over the 29 function slots (the caller does the `cumsum`); `to_numpy(li)`; `check_tensor(x)`; `str_tree(value, node_type, subtree_size)`; `randint(size, low, high, dtype=torch.int32, device="cuda")`; `inspect_function(func)` (returns the parameter names of a callable, rejects defaults).

`tree.py` — `Tree(input_len, output_len, node_value, node_type, subtree_size)`:
- `Tree.random_generate(descriptor)` (delegates to `Forest.random_generate(pop_size=1)[0]`), `to_forest()`.
- `forward(x)` — `(input_len,)` → `(output_len,)`; `(batch, input_len)` → `(batch, output_len)`.
- `SR_fitness(inputs, labels, use_MSE=True, execute_mode="auto")` → `(1,)` tensor (pop_size=1).
- `to_sympy_expr(symbol_names=None)` — sympy expr for `output_len == 1`, a **list** of exprs for `output_len > 1`; moves to CPU.
- `to_infix()`, `to_png(fname)`, `__repr__` (== sympy repr), `_fillout_graph(graph)`.
- There is **no** `Tree.batch_forward` (only `Forest`/`CombinedForest` have one).

`forest.py` — `Forest(input_len, output_len, batch_node_value, batch_node_type, batch_subtree_size)`:
- `pop_size, max_tree_len = batch_node_value.shape`; all three tensors must be exactly `(pop_size, max_tree_len)`.
- `Forest.random_generate(pop_size, descriptor)` (CUDA `tree_generate`), `zero_generate(pop_size, max_tree_len, input_len, output_len)` (degenerate `CONST=0` trees).
- `forward(x)` — `(pop_size, input_len)` → `(pop_size, output_len)`; asserts `x.shape[0] == pop_size`.
- `batch_forward(x)` — `(batch, input_len)` → `(pop_size, batch, output_len)` (expands tensors with `repeat`/`repeat_interleave`; memory-heavy, uses torch ops for the reshaping).
- `mutate(replace_pos, new_sub_forest)` → **new** `Forest`; `replace_pos` is an int tensor `[pop_size]`, and row `i` of `new_sub_forest` is spliced as a whole subtree at that position (ancestor subtree sizes fixed by the kernel).
- `crossover(left_indices, right_indices, left_pos, right_pos)` → **new** `Forest` of size `left_indices.shape[0]`.
- `SR_fitness(inputs, labels, use_MSE=True, execute_mode="auto")` → `(pop_size,)` fitness (CUDA); `use_MSE=False` selects absolute error (MAE per data point), not RMSE.
- Dunders: `__getitem__` (int or 0-dim tensor → `Tree` view; slice → `Forest` view; 1-D `Tensor`/`ndarray` incl. LongTensor/bool mask → `Forest` **copy**, advanced indexing), `__setitem__` (**in-place** row/slice assignment), `__iter__`/`__next__` (yields `Tree`s), `__len__`, `__add__`/`__radd__` (concat along dim 0 with `Forest` or `Tree`), `__str__`/`__repr__` (renders every individual as a sympy expr — slow), `__getstate__`/`__setstate__` (pickle via CPU numpy; unpickling moves back to `cuda`).

`combined_forest.py` / `combined_tree.py` — the data-driven multi-output model:
- `CombinedForest(forests: List[Forest], data_info: dict[str, list[str]])` where `data_info[output_name] = [input_name, ...]` selects which columns of the input dict feed that sub-forest.
- `CombinedForest.random_generate(pop_size, data_info, descriptors)` (one `GenerateDescriptor` per output — a single descriptor is broadcast; each must have `output_len == 1` and `input_len == len(data_info[out_name])`).
- `forward(x: dict[str, 1-D len == pop_size])` → `dict[out_name, (pop_size, 1)]`; `batch_forward(x: dict[str, 1-D len == batch])` → `dict[out_name, (pop_size, batch, 1)]` (inputs are column-ized with `[:, None]` and concatenated in `data_info` order).
- Neither `CombinedForest` nor `CombinedTree` defines `SR_fitness`, and `CombinedForest` has no `share_input` attribute (only prose references remain, in the stale consumers `problem/symbolic_regression.py` and the examples); its `forward`/`batch_forward` accept the `dict[str, Tensor]` protocol only — a raw tensor raises a Python indexing error on the first string key.
- `CombinedTree(trees, data_info)` exposes each sub-tree as an attribute named after the output (`best.A.to_sympy_expr()`), `forward(x: dict)` (delegates through a temporary pop_size=1 `CombinedForest`), `to_combined_forest()`, `random_generate(descriptors, data_info)`.
- Both support indexing, iteration, `__setitem__`, `__add__` (must have identical `data_info`).

## CUDA vs. pure-torch split
Dispatched to the compiled extension (`torch.ops.evogp_cuda.*`, schemas in `../cuda/torch_wrapper.cu`):
| op | Python caller |
| --- | --- |
| `tree_generate` | `Forest.random_generate` |
| `tree_evaluate` | `Forest.forward`, `Forest.batch_forward`, `Tree.forward` |
| `tree_SR_fitness` | `Forest.SR_fitness`, `Tree.SR_fitness` |
| `tree_mutate` | `Forest.mutate` |
| `tree_crossover` | `Forest.crossover` |
Pure torch/python: construction/validation of `Forest`/`Tree`/`Combined*` wrappers, indexing/`cat`/concat, pickling, descriptor tensor construction, `randint`, and all sympy/networkx rendering.
Fitness `execute_mode` strings map to kernel ids inside the C++ dispatcher: `"data parallel"→1` (constant_tree), `"tree parallel"→2` (constant_vars), `"auto"→4` (advanced if `data_points <= #cuda cores`, else constant_tree), and `"hybrid parallel"→0` from `Forest` but `→3` from `Tree` — both 0 (normal) and 3 (advanced) are valid kernels; treat the strings as the API and do not normalize.
Tree splicing (mutation/crossover), `node_type`/`node_value` patching, and all `OUT_NODE` bit handling for the multi-output model live in the **algorithm layer** (`../../algorithm/mutation/*`), not here — this layer only exposes the raw splice/mutate primitives.

## Constraints
`MAX_STACK = 1024` and `MAX_FULL_DEPTH = 10` are defined **twice**: in `utils.py` and as `constexpr` in `../cuda/defs.h`. They must stay in sync — the kernels allocate fixed per-thread `float stack[MAX_STACK]`, `GPNode gp[MAX_STACK]`, etc. Editing one without the other silently corrupts GPU memory.
`import evogp.tree` executes `import evogp.evogp_cuda` (`__init__.py` line 2), so **everything** downstream requires the built/installed CUDA-HIP extension (`pip install -e .`); `import evogp` alone succeeds because `evogp/__init__.py` is empty.
`max_tree_len <= MAX_STACK` (asserted in `GenerateDescriptor`) and `max_tree_len >= (max_arity**max_layer_cnt - 1) / (max_arity - 1)` (asserted by `check_tree_length`). The generator itself never bounds-checks the tree length while running, so an undersized `max_tree_len` is a hard stack overflow, not an exception.
`max_layer_cnt <= MAX_FULL_DEPTH` (10): the depth-indexed leaf schedule is exactly `MAX_FULL_DEPTH` long and must contain a `1.0` terminator at depth `max_layer_cnt - 1` to stop growth. `max_layer_cnt = 11` still passes the shape assert when only unary functions are used but drops the terminator, letting the kernel index `leafProbs[10]` out of bounds and grow past its fixed arrays.
All trees in one `Forest` must share `input_len`, `output_len`, `max_tree_len` (`mutate`/`crossover` assert this); `y = forest + other` requires matching `input_len`/`output_len`.
`using_funcs` entries must be exact `FUNCS_NAMES` strings — `dict2prob` / `check_tree_length` use `FUNCS_NAMES.index(...)` and raise otherwise. Function ids are positional: inserting/renaming an operator requires touching `utils.py` (`Func`, `FUNCS_NAMES`, `FUNCS_DISPLAY`, `SYMPY_MAP`) **and** `../cuda/defs.h` together.
Import-cycle rule: module-level imports go one way only (`utils` → `descriptor` → `forest` → `combined_forest` → `combined_tree`). `tree.py` must import `forest` **inside** its methods (`Tree.random_generate`, `Tree.to_forest`) and `combined_tree.py` must import `combined_forest` inside its methods; moving those to module scope re-creates the cycle.
`tree.py` and `forest.py` use `from .utils import *`, so every public name in `utils.py` leaks into their namespaces (`forest.NType`, `tree.FUNCS_NAMES`, …) — avoid adding new public names casually.
Keep CONTEXT.md at the level of this directory; do not modify source here from a context task.

## Design Decisions
**Descriptor-driven randomness** (commit `825a90f`, "add generate descriptor to avoid disgusting generate parameters"): all generation hyper-parameters live in one immutable-ish object shared by `Forest.random_generate` and the mutation operators, and the derived GPU tensors (function cdf, constant samples, depth→leaf schedule) are precomputed once instead of being rebuilt per generation.
**Fixed-width batched tensors** so generation/evaluation/mutation/crossover are one kernel launch for the whole population; the cost is padding with garbage beyond `subtree_size[0]` and a hard `max_tree_len` cap.
**Prefix (pre-order) layout with per-node subtree sizes**: makes tree validity structural, makes subtree splice a contiguous copy, and lets evaluation be a single reverse-then-stack pass (see `_treeGPEvalByStack`).
**Fixed-size per-thread stacks on the GPU** (instead of dynamic allocation) are what makes the `MAX_STACK` duplication necessary and why `max_tree_len` is validated in Python.
**Multi-output via an `OUT_NODE` flag + additive accumulation** rather than several forests, so one population still means one set of tensors (`out_prob` controls how often a function node is tagged).
**`data_info` dict refactor** (commit `9391c4d`) replaced the earlier `formula=` / `share_input=` API: mapping each output name to the input names it consumes lets each sub-forest read a different input subset (needed by `problem/custom_loss.py`). This is the current contract; tutorials/examples were never migrated (see Known Issues).
**Mutation/crossover stay as primitive splices** in this layer while the semantic details (out-index preservation, function sampling from roulette cdfs) live in the algorithm layer, which is why `GenerateDescriptor` also exposes `roulette_ufuncs/bfuncs/tfuncs`.

## Known Issues / Gotchas
`utils.check_tensor(x)` returns **None** when given a non-tensor (the fallback branch has no `return`) — `x = check_tensor(x)` silently nulls lists/ints. Always pass torch tensors.
`check_tensor` also does `.to("cuda").detach().requires_grad_(False)`, so inputs never carry autograd and any input graph is broken.
`utils.py` defines `dict2prob` twice (a numpy version, then a torch version); the torch one wins, the numpy one is dead code.
`GenerateDescriptor.update(**kwargs)` returns a **new** descriptor but also mutates the receiver's private `__params`, so kwargs are **sticky** across successive `update()` calls on the same object. Worse, `__params` captured the *parameter* values before substitution, so updating a descriptor whose original had `const_samples=None` re-draws fresh random constants from `const_range` — the initial and mutated populations then use different constant sets.
`Forest.forward` asserts `x.shape[0] == pop_size`; for arbitrary batch sizes use `batch_forward`. `Tree.forward` accepts 1-D or 2-D input but returns the batch dimension unless the input was 1-D.
`CombinedTree.forward` ends with `return pop_res[0]` where `pop_res` is a **dict** returned by `CombinedForest.forward`/`batch_forward` — leftover from the pre-`data_info` API, so it raises `KeyError` for normal string output names. `CombinedTree` also has **no** `to_sympy_expr`/`to_png`, although the tutorial and examples call them.
`CombinedForest.__add__(CombinedTree)` reads `other.tree[i]` (attribute is `trees`) → `AttributeError`.
`CombinedForest.input_names` is built with `list(set(...))`, so its order is not deterministic across processes; it is only used for counting today, but do not rely on its ordering.
`Forest.__iter__` stores `iter_index` on the instance: iteration is not reentrant, so nested `for` loops over the same `Forest` (and `str(forest)`, which iterates) misbehave. `repr(Forest)` builds a sympy expression per individual — extremely slow for large populations.
`Forest[int]` and `Forest[slice]` return views sharing the parent's storage; `forest[i].node_value[...] = ...` writes into the forest. Indexing with a `Tensor`/`ndarray` (incl. LongTensor / bool mask, e.g. `forest[best][worthy]` in `problem/transformation.py`) uses PyTorch advanced indexing and returns a **copy**, so writes through it are lost. `__setitem__` mutates in place, while `mutate`/`crossover` return new forests.
`Forest.random_generate` seeds each individual with `hash(individual_index, keys[0], keys[1])` where the two uint32 keys come from the torch CUDA RNG — reproducibility needs `torch.cuda.manual_seed` (or torch's seeded RNG), and generation does not consume torch's per-op RNG stream.
`randint(size, low, high, ...)` truncates toward zero after `torch.rand`, so negative `low` values give a biased distribution; it defaults to `device="cuda"`, `dtype=torch.int32`.
The combined API is mid-migration: `problem/symbolic_regression.py` still reads `forest.share_input` (no longer an attribute of `CombinedForest`), and `example/combined_sr.py`, `tutorial/combined_gp.ipynb`, `test/combined_test.ipynb` still use the removed `formula=`/`share_input=` signature. `problem/custom_loss.py` is the consumer that matches the current `data_info` API.
`Tree.to_png` requires `networkx` (a hard dependency) **plus** `pygraphviz` (optional extra `visualize`) and the graphviz `dot` binary; it annotates out-nodes in red and labels edges with child order.
`Tree.__python_forward` / `Tree.__assert_valid` (name-mangled, debug-only, `ADD/SUB/MUL/DIV` subset) and `Forest.zero_generate` are unused inside the package.
`Tree.__repr__` returns the sympy repr, so printing a tree can be a CPU sync + sympy build (and prints a *list* when `output_len > 1`); `to_sympy_expr` on the multi-output path bit-reinterprets numpy scalars with `v.view(np.int32)`.

## Notes for Agents
Read `../cuda/CONTEXT.md` before touching tensor semantics: the kernels are authoritative. `../cuda/defs.h` holds the enums/constants, `../cuda/torch_wrapper.cu` holds the op schemas + shape asserts, and `forward.cu::_process_node` documents every function (including the "loose" variants) and the multi-output accumulation.
Representation questions ("what does byte/depth mean") are answered by `generate.cu` (generation + subtree-size computation) and `mutation.cu::_gpTreeReplace` (splicing) — not by the Python wrappers, which mostly validate and delegate.
`NType` and `Func` are plain `int` namespaces, not `enum.Enum`; `FUNCS_NAMES.index(name)` is the id mapping used both by Python and by the fused kernels.
The CUDA generator writes only the first `subtree_size[0]` entries of each row; the tail is uninitialized. Any code that iterates a row must bound itself by `subtree_size[:, 0]`.
`MAX_STACK` is re-exported from `evogp.tree` precisely because the algorithm layer needs it to bound random node positions.
Nothing here is device-agnostic: `random_generate`, all forwards, fitness, mutate/crossover, and even `zero_generate` construct tensors directly on `"cuda"`.

## Test Strategy
No pytest suite exists for this layer. The closest thing to a unit test is `../../../test/test_bind_success.py`, which calls `torch.ops.evogp_cuda.tree_generate/tree_mutate/tree_crossover/tree_evaluate/tree_SR_fitness` directly with hand-built descriptors and prints trees via `evogp.tree.utils.str_tree` (`python test/test_bind_success.py`).
Other ad-hoc checks live in `../../../test/fix_bug.py` (hand-builds a `Tree` from literal `node_value`/`node_type`/`subtree_size` tensors — the clearest example of the encoding — then calls `Tree.SR_fitness` in two `execute_mode`s), `../../../test/vis.ipynb` (visualization scratch notebook, `to_png` call commented out), `../../../test/sympy_test.ipynb` (bare sympy playground, does not import evogp), and `../../../test/combined_test.ipynb` (stale combined API).
`../../sr_test.py` is the end-to-end smoke test for the descriptor + `Forest` + pipeline path (`python src/evogp/sr_test.py`), and `../../../example/basic.py` is the documented minimal example.
All of these need a CUDA/HIP GPU and the built extension, so they cannot run in a CPU-only sandbox.
Cheap self-consistency check when touching the representation: compare `Tree.to_sympy_expr()` / `Tree.to_infix()` against `Tree.forward()` on random inputs, and check `subtree_size[0]` against the node-arity walk done in `Tree.__assert_valid`.

## Routing Table
This directory has **no child subdirectories** — the whole tree layer is these seven files; do not look for sub-nodes to delegate to.
Siblings (read-only from here — escalate any needed write to the parent `./src/evogp`):
- `../cuda/` → the CUDA/HIP kernels and `defs.h` constants this layer binds (`evogp_cuda`).
- `../algorithm/` → mutation/crossover/selection that write `OUT_NODE` bits and splice trees via `Forest.mutate`/`crossover`.
- `../problem/` → `custom_loss.py` (current `data_info`-based consumer of `CombinedForest`) and `symbolic_regression.py` (still references the removed `share_input`).
- `../pipeline/` → orchestrates algorithm + problem.
- `../../../test/`, `../../../example/`, `../../../tutorial/` → scripts/notebooks exercising this API (several are stale w.r.t. the combined API).
