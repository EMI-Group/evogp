# src/evogp/cuda — CUDA/HIP kernels of EvoGP

## Intent
Hand-written CUDA kernels + PyTorch operator bindings that implement every GPU-accelerated primitive of EvoGP (tree generation, mutation, crossover, expression evaluation, symbolic-regression fitness).
There is no Python code here; the four `.cu` files are the exact source list of `CUDAExtension(name="evogp.evogp_cuda", ...)` in the repo-root `setup.py` (lines 59-71).
Because `src/evogp/tree/__init__.py:2` does `import evogp.evogp_cuda`, a successful build of this directory is a hard prerequisite for the entire library (even for pure-Python tooling).

## Files
- `defs.h` — compile-time constants (`MAX_STACK = 1024`, `MAX_FULL_DEPTH = 10`, `DELTA = 1E-9f`, `MAX_VAL = 1E9f`) and the `ntype_t` / `func_t` enums. Must stay in sync with `src/evogp/tree/utils.py`.
- `kernel.h` — host prototype declarations for the 5 public entry points, POD node structs (`GPNode`, `OutNodeValue`, `LeftRightIdx`, `NchildDepth`), `RandomEngine` = `thrust::random::taus88`, and the FNV-1a `hash()` used for seeding.
- `torch_wrapper.cu` — arg validation (`check_tensor`), output tensor allocation, `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` binding registration, and `check_cuda_error()`.
- `generate.cu` — random tree-population generation (`treeGPGenerate`).
- `mutation.cu` — subtree replacement (`_gpTreeReplace`), mutation and crossover kernels.
- `forward.cu` — the expression interpreter (`_process_node`, `_treeGPEvalByStack`), 5 fitness kernels, and `getCudaCoreCount()`.

## API Surface — pybind / Torch Library ops
All ops are reachable only as `torch.ops.evogp_cuda.<name>` (schemas at `torch_wrapper.cu:291-299`, implementations at `:301-307`).
The `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` block at `torch_wrapper.cu:287` is intentionally empty (its only `m.def` is commented out) — do not look for pybind11 function registrations there.
Registration is `TORCH_LIBRARY_IMPL(evogp_cuda, CUDA, m)`: the ops exist **only for the CUDA dispatch key** (this key is also what ROCm builds dispatch on), so CPU tensors are rejected.
Schema note: `float` in a TORCH_LIBRARY schema means C++ `double` (matches the `double out_prob/const_prob` params), and schemas are positional — C++ parameter defaults are NOT part of the schema.

1. `tree_generate(pop_size, gp_len, var_len, out_len, const_samples_len, out_prob, const_prob, keys, depth2leaf_probs, roulette_funcs, const_samples) -> (value, node_type, subtree_size)`
   - Inputs: `keys{2}` uint32, `depth2leaf_probs{MAX_FULL_DEPTH=10}` f32, `roulette_funcs{Function::END=29}` f32 (ascending cumsum), `const_samples{const_samples_len}` f32.
   - Outputs: `value{pop_size,gp_len}` f32, `node_type{...}` i16, `subtree_size{...}` i16.
   - `torch_wrapper.cu:34` → `generate()` `generate.cu:210` → `generateExecuteKernel<multiOutput>` `generate.cu:180` → kernel `treeGPGenerate` `generate.cu:17`.
   - Python call site: `Forest.random_generate` (`src/evogp/tree/forest.py:64`), which draws `keys = torch.randint(..., dtype=torch.uint32, device="cuda")` on every call.
2. `tree_mutate(pop_size, gp_len, value_ori, type_ori, subtree_size_ori, mutateIndices, value_new, type_new, subtree_size_new) -> (value, node_type, subtree_size)`
   - `mutateIndices{pop_size}` int32 = per-tree index of the subtree root to replace; `*_new` are a whole replacement forest of the same pop size (row i's first `subtree_size_new[0]` nodes are used).
   - `torch_wrapper.cu:90` → `mutate()` `mutation.cu:186` → kernel `treeGPMutationKernel` `mutation.cu:118` → device helper `_gpTreeReplace` `mutation.cu:6`.
   - Python call site: `Forest.mutate` (`forest.py:214`).
3. `tree_crossover(pop_size_ori, pop_size_new, gp_len, value_ori, type_ori, subtree_size_ori, left_idx, right_idx, left_node_idx, right_node_idx) -> (value, node_type, subtree_size)`
   - Parents `{pop_size_ori, gp_len}`; the four index vectors are `{pop_size_new}` int32; output `{pop_size_new, gp_len}`.
   - `torch_wrapper.cu:140` → `crossover()` `mutation.cu:312` → kernel `treeGPCrossoverKernel` `mutation.cu:224` → same `_gpTreeReplace` helper.
   - Python call site: `Forest.crossover` (`forest.py:280`).
4. `tree_evaluate(pop_size, gp_len, var_len, out_len, value, type, subtree_size, variables) -> results`
   - `variables{pop_size,var_len}` f32 → `results{pop_size,out_len}` f32.
   - `torch_wrapper.cu:193` → `evaluate()` `forward.cu:353` (dispatches `outLen > 1` to `treeGPEvalKernel<true>` at `forward.cu:367-370`, whose per-output results are written at `:342-347`) → kernel `treeGPEvalKernel<multiOutput>` `forward.cu:305` → `_treeGPEvalByStack` `forward.cu:247` → `_process_node` `forward.cu:80`.
   - Python call sites: `Forest.forward` (`forest.py:129`), `Forest.batch_forward` (`forest.py:163`, expands to `pop_size*batch_size` rows via `repeat_interleave`), `Tree.forward` (`tree.py:61`, repeats the single tree `batch_size` times).
5. `tree_SR_fitness(pop_size, data_points, gp_len, var_len, out_len, useMSE, value, type, subtree_size, variables, labels, kernel_type) -> fitness{pop_size}`
   - `torch_wrapper.cu:235` → `SR_fitness()` `forward.cu:827` → `kernel_type` dispatch:
     - `0` `normal_SR_fitness` (`forward.cu:481`) — grid `{popSize, ceil(dataPoints/1024)}`, block 1024, shared-memory reduction + 1 `atomicAdd` per block (tree × data hybrid parallel, fixed block size).
     - `1` `constant_tree_SR_fitness` (`forward.cu:649`) — host loop over the population; each tree is copied into `__constant__` memory per tree (`cudaMemcpyToSymbolAsync`) then one kernel launch per tree (data parallel).
     - `2` `constant_vars_SR_fitness` (`forward.cu:781`) — host loop over data points; variables/labels of one data point go to `__constant__` memory, one thread per tree (tree parallel).
     - `3` `advanced_SR_fitness` (`forward.cu:514`) — same kernel as 0 but `block_size = dataPoints <= 256 ? dataPoints : 1024`.
     - `4` (C++ default, "auto") — `dataPoints <= getCudaCoreCount() ? 3 : 1`.
   - Result is the mean per-data-point error (MSE if `useMSE`, else MAE) and is **non-negative**; the problem layer negates it (`src/evogp/problem/symbolic_regression.py:83`).
   - Python call sites: `Forest.SR_fitness` (`forest.py:351`) maps `hybrid parallel->0, data parallel->1, tree parallel->2, auto->4`; `Tree.SR_fitness` (`tree.py:114`) maps `hybrid parallel->3, data parallel->1, tree parallel->2, auto->4` with `pop_size=1`. The `Tree`/`Forest` mapping of "hybrid parallel" disagrees (3 vs 0) even though 3 is just 0 with an adaptive block size.

## Python call-site → binding → kernel map (debugging entry point)
| Python | torch.ops call | wrapper | kernel file:line |
|---|---|---|---|
| `Forest.random_generate` forest.py:64 | `tree_generate` | torch_wrapper.cu:34 | generate.cu:17 |
| `Forest.mutate` forest.py:214 | `tree_mutate` | torch_wrapper.cu:90 | mutation.cu:118 |
| `Forest.crossover` forest.py:280 | `tree_crossover` | torch_wrapper.cu:140 | mutation.cu:224 |
| `Forest.forward` forest.py:129 / `batch_forward` :163 / `Tree.forward` tree.py:61 | `tree_evaluate` | torch_wrapper.cu:193 | forward.cu:305 |
| `Forest.SR_fitness` forest.py:351 / `Tree.SR_fitness` tree.py:114 | `tree_SR_fitness` | torch_wrapper.cu:235 | forward.cu:481/649/781/514 |

## Kernel design
Thread mapping: `generate`, `mutate`, `evaluate`, `constant_vars_SR_fitness` use **one thread per tree** (`n = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `n >= popSize`), so a tree is never split across threads and no inter-thread communication is needed.
`treeGPRegressionFitnessKernel` (modes 0/1/3) instead uses a 2-D grid `{population, data-point blocks}`: each thread evaluates one (tree, data point) pair, results are reduced in `__shared__ float sharedFitness[SR_BLOCK_SIZE=1024]` and added with exactly one `atomicAdd` per block into `fitnesses[nGP]`; `averageFitnessValueKernel` then divides by `dataPoints`.
Block sizes come from `cudaOccupancyMaxPotentialBlockSize` (no dynamic shared memory, no explicit cap) for every op and for fitness mode 2, except fitness modes 0/1 which use a fixed `SR_BLOCK_SIZE = 1024` and mode 3 which uses `dataPoints <= 256 ? dataPoints : 1024`.
Per-thread scratch is **fixed-size local arrays sized by `MAX_STACK`** (`generate.cu:56-57`, `mutation.cu:25-27`, `forward.cu:326-327, 440-441, 619-620, 764-765`) — never `alloca`, never dynamic shared memory.
In `forward.cu` the `float stack[MAX_STACK]` holds operands and the `int16_t infos[2*MAX_STACK]` is carved up as: `[0,MAX_STACK)` node types, variable values as `float*` at `infos+MAX_STACK`, and (multi-output only) output accumulators as `float*` at `infos+MAX_STACK+MAX_STACK/2`.
Randomness: `generate` seeds `thrust::random::taus88` per tree with `hash(n, keys[0], keys[1])` (FNV-1a), so the population is a pure function of `(pop_size, keys, distributions)` — determinism is inherited from the `keys` tensor, which `Forest.random_generate` draws from torch's global CUDA RNG (`torch.random.manual_seed(0)` + `torch.cuda.manual_seed(0)` in `src/evogp/sr_test.py` make the whole run reproducible).
Node layout: rows are stored in **prefix (pre-order) order with index 0 = the root**, and only the first `subtree_size[0]` entries are meaningful; the interpreter walks a row from the END to the START ("the inverse of prefix expression", `forward.cu:283`) — the same convention as the pure-Python reference `Tree.__python_forward` in `src/evogp/tree/tree.py:326`.
Subtree sizes are computed by that same reverse sweep (`generate.cu:131-158`, reusing `childsAndDepth` memory as `int*`), and `_gpTreeReplace` re-derives ancestor sizes with a descending walk before splicing the new subtree in.
Multi-output trees (`outLen > 1`) use the `multiOutput=true` template instantiations: unary/binary/ternary nodes may carry `OUT_NODE` (bit 7) and store `{function, outIndex}` bit-packed into the float `value` (`OutNodeValue`, `kernel.h:107`); such a node adds its value to output accumulator `outIdx` and passes its rightmost operand up to its parent (Y. Zhang multi-output tree structure, `forward.cu:237-242`).

## Constraints
`MAX_STACK` and `MAX_FULL_DEPTH` are **duplicated in Python** at `src/evogp/tree/utils.py:10-11`; changing either requires a rebuild of the extension AND an edit to that file.
`GenerateDescriptor` asserts `max_tree_len <= MAX_STACK` (`src/evogp/tree/descriptor.py:62`) and `depth2leaf_probs.shape == (MAX_FULL_DEPTH,)` (`descriptor.py:163`), and `torch_wrapper.cu` re-checks `0 < gp_len <= MAX_STACK` on all five ops.
`defs.h`'s `ntype_t` / `func_t` enums must match `NType` / `Func` in `src/evogp/tree/utils.py` (VAR=0, CONST=1, UFUNC=2, BFUNC=3, TFUNC=4, TYPE_MASK=0x7F, OUT_NODE=1<<7; IF=0 … LOOSE_SQRT=28, END=29).
`roulette_funcs` must be the ascending cumulative sum of the function probabilities (built in `descriptor.py:107`); the C++ selection loop scans from `END-1` downwards and assumes monotonicity.
All tensor arguments must be CUDA, contiguous, and exactly the right shape — enforced by `check_tensor` (`torch_wrapper.cu:7`) which checks device + contiguity + shape but **not dtype**; dtypes are only implied by `data_ptr<T>()` (float32 / int16 / int32 / uint32).
Any new kernel must compile under both nvcc and hipcc after torch's build-time hipification: no warp intrinsics or warp-size assumptions (there are currently none), use `std::` math spellings, and keep the single CUDA-only header guard (`device_launch_parameters.h` behind `#ifndef USE_ROCM`, `kernel.h:6-8`).
Adding a new `.cu` file requires adding it to the explicit `sources` list in the repo-root `setup.py:61-66` — there is no glob.

## Design Decisions
One thread per tree (rather than one thread per node) because tree contents are data-dependent and variable-length; per-thread explicit stacks make the interpreter branch-free across threads.
Fixed-size local arrays instead of `alloca` is deliberate and load-bearing: dynamic device allocations aliased across concurrently resident wavefronts on AMD GPUs; since `MAX_STACK` is a compile-time constant the plain array is the idiomatic equivalent (applied to both backends).
Three fitness execution modes exist because the optimum depends on `dataPoints` vs. population size: keeping the tree in `__constant__` memory (mode 1) or the data point in `__constant__` memory (mode 2) trades host-side serialization for cached, broadcast-friendly reads; mode 3 is mode 0 with a data-fit block size; mode 4 auto-selects by comparing `dataPoints` against the SM core count.
Relative to the SM core count, `SR_BLOCK_SIZE = 1024` and mode-2/3 thresholds (256) are empirical tuning constants.
The multi-output path is a separate template instantiation (rather than a runtime flag) so the single-output hot path keeps zero overhead and constant memory layout.

## Known Issues / Gotchas
`check_cuda_error()` (`torch_wrapper.cu:19`) exists but **every call site is commented out**, so kernel launch/execution errors are never reported at the op boundary — a failure (e.g. a device assert trap, which poisons the CUDA context) surfaces later, attributed to whatever operation runs next.
Generation never verifies that the produced tree fits: `treeGPGenerate` writes `len = gp[0].subtreeSize` entries into a row of only `gpLen`, so `len > gpLen` silently spills into the neighbouring row (and past the tensor for the last row); the only protection is the Python-side descriptor invariant (`descriptor.check_tree_length` asserting `max_tree_len >= (arity^max_layer_cnt-1)/(arity-1)`, plus `depth2leaf_probs` forcing leaves above `max_layer_cnt-1`).
`childsAndDepth[top++]` / `gp[topGP++]` in `generate.cu` have no bounds check, and `leafProbs[cd.depth]` indexes a `MAX_FULL_DEPTH`-long array with the tree depth — a perverse user-supplied `depth2leaf_probs` (e.g. all zeros) overflows both the local arrays and that index.
The `OUT_NODE` bit is masked only inside `if constexpr (multiOutput)` (`forward.cu:91-94`): feeding a tree whose types carry `OUT_NODE` to the **single-output** interpreter makes every node-type comparison fail, so nodes fall through to the final `else` (ternary `IF`) branch and evaluation silently returns garbage. The `generate`/`evaluate` op pairs are self-consistent (`outLen > 1` ⇔ `multiOutput`), so keep them matched.
Bounds checking is deliberately partial and the Python layer is trusted: `tree_mutate` checks `old_node_idx ∈ [0, old_size)` and that the result fits `maxGPLen`, and on failure **silently copies the parent tree** (no error raised); `tree_crossover` checks only `right_idx ∈ [0, pop_size_ori)` — `left_idx`, `left_node_idx` and `right_node_idx` are unchecked and index global memory directly; `_process_node` reads `s_vars[var_num]` with `var_num` taken raw from the node's float value.
Intermediate values are **not clipped**: the ±`MAX_VAL` clamping block in `_process_node` is commented out (`forward.cu:226-234`), despite `defs.h` commenting that "the absolute value of any operation will be limited to MAX_VAL"; only the `LOOSE_*` variants guard their inputs, while `DIV`/`INV` by zero intentionally produce NaN and `LOG` of 0 gives -inf. `is_nan`/`is_inf` in `kernel.h` are consequently unused.
`getCudaCoreCount()` (`forward.cu:16`) calls `cudaGetDeviceProperties(&prop, 0)` — hard-coded device 0 (wrong on multi-GPU machines) — and its SM-cores-per-SM table stops at sm_89 (everything newer falls through to 128); it is a blocking host call executed on **every** `tree_SR_fitness` with `kernel_type == 4`, i.e. once per generation in the default "auto" mode.
Output rows are allocated with `torch::empty`, so bytes beyond `subtree_size[0]` (and, in the invalid-index early-exit paths of mutate/crossover, everything past `old_size`) stay uninitialized garbage; nothing may read past `subtree_size[0]`.
The device-side `assert`s (the `assert(top == 1)` interpreter canary at `forward.cu:301/588/735`, plus the varLen/outLen capacity asserts) are only active when `NDEBUG` is undefined — never rely on them as input validation, and note that a tripped device assert prints `top: %d` and then traps, poisoning the context.
`evaluate()` (`forward.cu:363`) computes occupancy for the `<false>` instantiation even when it launches `<true>`; it and `generateExecuteKernel`/`mutate`/`crossover` only grow the grid (`if (gridSize * blockSize < popSize)`) from the occupancy-suggested value.
`-maxrregcount=32` (CUDA-only flag) plus 4-8 KB of per-thread local arrays forces local-memory spills — that is why block sizes are occupancy-derived, and why the local arrays are the dominant occupancy cost.
`kernel.h` still includes `<malloc.h>` and `<iostream>` from before the fixed-array change, and declares `LeftRightIdx` which nothing uses — all vestigial.

## Notes for Agents
Nothing in this directory is generated — but under ROCm, torch's build produces a hipified mirror at `src/evogp/hip/` (gitignored, regenerated on every build): **never edit that mirror**, edit these `.cu` files and rebuild.
Do not try to compile this directory ad hoc or with a hand-written nvcc/hipcc line: the only supported build is `pip install . --no-build-isolation` via `setup.py` (there is no CMake/Makefile), and `MAX_JOBS` is forced to the CPU count; a full rebuild takes minutes.
The compiled extension is imported as `evogp.evogp_cuda` (module name, not a package) — a build failure breaks `import evogp.tree` for every consumer, including pure-Python tests.
`.gitignore` anticipates a scratch `src/evogp/cuda/forward_new.cu`; it is not currently present and, being absent from `setup.py`'s `sources`, would not be compiled.
Build-config knowledge lives in the repo-root `setup.py`, not here: CUDA gets `["-O3","--expt-relaxed-constexpr","--ptxas-options=-v","-Xptxas=-O3","-lineinfo","-lcudart","-use_fast_math","-maxrregcount=32"]`, ROCm (`torch.version.hip is not None`) gets `["-O3"]` only because hipcc rejects the ptxas/nvcc-only flags; cxx args are `["-O3"]` on both.
No child directories — this node is a leaf; do not delegate.

## Dependencies
CUDA toolkit (`nvcc`, `thrust/random.h` for taus88, `device_launch_parameters.h`) or ROCm (`hipcc` + rocThrust), matching the installed PyTorch build (CUDA or ROCm wheel), plus a C++ compiler and a GPU.
`torch.utils.cpp_extension` (`CUDAExtension`, `BuildExtension.with_options(parallel=True)`) supplies the build driver and hipification; `python_requires >= 3.9`.

## Test Strategy
There are no unit tests in this directory.
Reference script: the repo-root `test/test_bind_success.py` — it calls every op directly through `torch.ops.evogp_cuda.*` with tiny tensors and prints trees via `str_tree`; it is the fastest op-level smoke test (read-only cross-reference, it lives outside this node).
Official gate: `python -m evogp.sr_test` (README "5. Validate Installation") — a fixed-seed 1000-population symbolic-regression run over 100 generations that must converge monotonically; this is the gate the ROCm PR validated.
Coverage gaps: the multi-output instantiations (`out_len > 1`, `OUT_NODE` machinery) are not touched by `sr_test` (which uses `output_len=1`) nor by `test_bind_success.py` (also `out_len=1`); they are reached only when a descriptor is built with `output_len > 1` — e.g. `example/brax_task.py` / `example/mjx_task.py` (action size) and `example/classifier.py` / `example/uci_classifier.py` with `multi_output=True` (class count), all of which pass `output_len=problem.solution_dim`; `CombinedForest` explicitly requires per-subtree `output_len == 1` and the torch execution path (`src/evogp/problem/symbolic_regression.py:72`).
Mode 2 (`constant_vars`) and mode 3 (`advanced`, and therefore everything reached via `execute_mode="hybrid parallel"`/`"auto"`) are only exercised through `SR_fitness` callers, never by a dedicated test.

## See Also
- `../tree/` — the Python side: `Tree`/`Forest` wrappers, the duplicated `MAX_STACK`/`MAX_FULL_DEPTH` constants, `GenerateDescriptor`, and every op call site listed above.
- Repo-root `setup.py` — the only place the source list and compile flags are defined.
- Repo-root `test/test_bind_success.py` — direct op-level smoke tests (outside this node; do not edit context there).
