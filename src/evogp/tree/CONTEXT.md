# `src/evogp/tree/` — EvoGP tree representation, ETL evaluation engine & pure-torch operators

## Intent
This is the **foundation layer** of EvoGP: the GP individual (`Tree`), the population (`Forest`), the multi-output variants (`CombinedTree` / `CombinedForest`), the random-generation blueprint (`GenerateDescriptor`), the pure-torch generation/splicing operators, and the ETL-backed evaluation engine.
Everything the rest of the library manipulates (fitness, selection, crossover, mutation, visualization) is expressed in terms of the tensor representation defined here.
This layer contains **no custom CUDA/HIP kernels and no compiled extension**: evaluation and fitness run through a per-structure ETL JIT engine (`etl_engine.py`), generation/mutation/crossover are vectorized plain torch (`generation.py`, `splicing.py`), and `import evogp.tree` works without any built extension.
This package imports **nothing** from other `evogp` subpackages — it is the root of the package DAG (`algorithm/` and `problem/` are its consumers; `pipeline/` sits above both; no cycles).
`ENGINE_SPEC.md` (this directory) is the authoritative semantics spec: prefix encoding, the full op table with edge cases, evaluation order, SR fitness, the multi-output protocol, and the generation/splicing algorithms, distilled from the legacy CUDA sources at snapshot `8a487f19`.
Its Appendices keep the legacy op schemas and the old call-site map as explicitly marked historical reference.

## Representation (the schema every other module depends on)
An individual is three parallel tensors of length `max_tree_len`, batched as `[pop_size, max_tree_len]`, encoded as a **prefix (pre-order) expression**: node 0 is the root and children follow depth-first, left to right.
- `node_value` — `torch.float32` (`Forest.batch_node_value`).
- `node_type` — `torch.int16` (`Forest.batch_node_type`).
- `subtree_size` — `torch.int16` (`Forest.batch_subtree_size`); `subtree_size[i]` = number of nodes in the subtree rooted at node `i`.
Only indices `[0, subtree_size[0])` are meaningful: `subtree_size[0]` is the real tree length, and the rest of a row must never be read (the generator zeroes unused cells, but that is an implementation detail).
`batch_subtree_size[:, 0]` is used throughout the algorithm layer as "the size of tree i".
`Tree` holds the same three tensors unbatched (1-D); `Forest[i]` / `Forest[a:b]` return `Tree` / `Forest` **views** sharing storage, while `Tensor`/`ndarray` indexing (incl. LongTensor / bool mask) goes through PyTorch advanced indexing and returns **copies**.

Node encoding:
- `node_type` base values: `0 VAR`, `1 CONST`, `2 UFUNC`, `3 BFUNC`, `4 TFUNC` (`NType`); `NType.OUT_NODE = 1<<7` is OR-ed in for multi-output nodes (`TYPE_MASK = 0x7F` extracts the base).
- `node_value`: `VAR` → variable index (float-encoded int); `CONST` → the constant; function node → function id (`Func.*`, i.e. the index in `FUNCS_NAMES`).
- Multi-output out-node packing: `node_value` is the float32 bit-reinterpretation of `struct OutNodeValue { int16_t function; int16_t outIndex; }` (`outIndex = int32_bits >> 16`, `function = int32_bits & 0xFF`).
- Function table: id 0 `IF` (ternary); ids 1–13 binary (`+ - * / loose_div pow loose_pow max min < > <= >=`); ids 14–28 unary (`sin cos tan sinh cosh tanh log loose_log exp inv loose_inv neg abs sqrt loose_sqrt`); operands are consumed leftmost-child-first.
- Exact per-operator edge cases (`DIV`/`LOG`/`SQRT`/loose variants, comparisons returning ±1, `IF` branch mapping) are in `ENGINE_SPEC.md` §3.
- Multi-output semantics (`output_len > 1`): every out-node accumulates its own value into `out[outIndex]` and passes its right-most child up to the parent; tagged leaves behave like untagged ones; outputs that were never tagged are `0.0`.
- `output_len == 1` is the single-output protocol; a tree containing an OUT_NODE-tagged node is rejected at decode time with `ValueError` (the legacy kernel had undefined behavior there).

## Engine architecture (`etl_engine.py`)
Evaluation is a "true jit": `decode_population` walks a population **host-side** into frozen per-tree static programs (int-only, constant subtrees folded), then `evaluate` / `sr_fitness` specialize each program into a fresh ETL graph via `etl.build_cached` and run it, with the compiled executables cached per *(tree structure, input shape, backend, device, options)*.
- `decode_population(batch_node_value, batch_node_type, batch_subtree_size, input_len, output_len) -> Population` — pure host work (tensors read back with `.cpu()`); validates the structure and raises `ValueError` naming the row and node (replacing the kernel's UB); `TypeError` for non-tensor / non-2-D input.
- `evaluate(population, x, *, per_row, out_device=None, backend=None, iree_device_index=None)` — `per_row=True`: `(pop, input_len) → (pop, output_len)`; `per_row=False`: `(N, input_len) → (pop, N, output_len)`; returns a fresh, contiguous, `requires_grad=False` float32 tensor on `x`'s device unless `out_device` is given.
- `sr_fitness(population, inputs, labels, use_MSE=True, *, ...)` — `(pop,)` float32 POSITIVE loss `Σ_points Σ_outputs err / N` (err = diff² for MSE else |diff|; outputs summed but never divided by `output_len`; NaN/inf propagate unsanitized); the caller applies the higher-is-better negation.
- One module-level `@etl.defn _eval_step` — it must stay a module-level function (its `id()` is part of the cache key; a closure would permanently miss); statics are codec-safe int-scalars/sequences re-passed by value at `etl.run`.
- Cache: one shared `etl.CompileCache` (`etl_engine.CACHE`), default maxsize 4096 (`EVOGP_ETL_CACHE_SIZE`), in-memory only (nothing persists across processes); `cache_stats()` / `clear_cache()` inspect/reset; measured ~270–478 KiB per entry.
- Backends: default `"numpy"` (ETL reference interpreter, cpu, no GPU needed); IREE is opt-in via `EVOGP_ETL_BACKEND=iree` + `EVOGP_ETL_IREE_DEVICE=<visible gpu index>` + `EVOGP_ETL_IREE_OPT_LEVEL` (`O0`/`O1` only on cuda) — experimental, ~0.4–0.6 s build per structure + ~2.4–3.6 ms/run, impractical at population scale.
- Node semantics are **correctly rounded float32**: the software transcendentals (`sin`/`cos`/`tan`/`sinh`/`cosh`/`tanh`/`log`/`loose_log`/`exp`/`pow`/`loose_pow`) are emitted as `cast_f32(op(cast_f64(v)))` (ETL has no `sinh`/`cosh`: they are the `exp`/`-exp` composition in f64) and every node value is stored back as f32 — the best possible f32 answer, since the f32 kernels are not correctly rounded (numpy f32 `exp`/`sin`/`cos`/`tan` are 1 ulp off on 5–39% of inputs). Hardware-exact ops stay f32 (`+ - * /`, `sqrt`, `abs`, `neg`, `copysign`, `min`/`max`, comparisons/select, which must keep their bit-exact ±1). `sinh` keeps `x` (sign of zero included) below `|x| < 1e-4`, where the correctly rounded f32 result *is* `x` — this is what makes `loose_inv(sinh(-0.0)) = -1e9` come out right on both the folded and the emitted path.
- Measured numpy economics (with `ETL_DISABLE_LOCATIONS=1`, current f64-emission code): 30–43-node trees 9.9 ms cold build / 0.46 ms warm run (N=100) and 9.8 ms / 0.49 ms (N=1000); `build_cached` cache hit 0.45 ms; ~102 distinct builds/s, so an all-distinct pop=1000 costs ~10 s cold and ~5.4 s at 50% duplicates, against a 0.5 s all-cached floor. The f64 casts add a handful of ops per software transcendental but do not change the order of magnitude (earlier revision, 30-node/60-node trees: 8.5/21.4 ms build, 0.87/1.47 ms run).
- `ETL_DISABLE_LOCATIONS=1` speeds tracing up ~4× on this workload and is recommended for production runs; it is not part of the cache key.
- Device policy: the engine accepts population/input tensors from any torch device and returns results on the input device unless `out_device`; the `Tree`/`Forest` wrappers still pass everything through `check_tensor` (`.to("cuda")`), so the public API keeps its CUDA-bound behavior.

## API Surface
`evogp.tree` exports: `GenerateDescriptor`, `Tree`, `Forest`, `CombinedForest`, `CombinedTree`, `MAX_STACK`, `randint`, `NType`.

### `tree.py` — `Tree(input_len, output_len, node_value, node_type, subtree_size)`
- `Tree.random_generate(descriptor)` delegates to `Forest.random_generate(pop_size=1)[0]`; `to_forest()` wraps a single tree as a `Forest`.
- `forward(x)` — 1-D `(input_len,)` → `(output_len,)`; 2-D `(batch, input_len)` → `(batch, output_len)`; evaluated through the ETL engine.
- `SR_fitness(inputs, labels, use_MSE=True, execute_mode="auto")` → `(1,)` positive loss.
- `to_sympy_expr(symbol_names=None)` — sympy expr for `output_len == 1`, a list of exprs for `output_len > 1`; comparisons render as the engine's ±1 (`Piecewise((1, cond), (-1, True))`), loose ops follow the engine's zero-clamp conventions, and every leaf is sympified (a constant-only tree returns a sympy `Float`, never a raw numpy scalar).
- `to_infix()` (single-output; warns and returns one string for multi-output), `to_png(fname)`, `__repr__` (== sympy repr).
- `__python_forward` — name-mangled, debug-only, full-semantics pure-Python reference evaluator (single-output) used as the in-repo cross-check oracle; `__assert_valid` is likewise debug-only.

### `forest.py` — `Forest(input_len, output_len, batch_node_value, batch_node_type, batch_subtree_size)`
- Public attributes: `input_len`, `output_len`, `pop_size`, `max_tree_len`, `batch_node_value`, `batch_node_type`, `batch_subtree_size`; no `.device` and no `.trees`; the three tensors must be exactly `(pop_size, max_tree_len)`.
- `Forest.random_generate(pop_size, descriptor)` (static) → pure-torch vectorized generation on the descriptor tensors' device (cuda in practice).
- `zero_generate(pop_size, max_tree_len, input_len, output_len)` (static) → degenerate all-`CONST`-0 trees on cuda.
- `forward(x)` — `(pop_size, input_len)` → `(pop_size, output_len)` (asserts `x.shape[0] == pop_size`); `batch_forward(x)` — `(N, input_len)` → `(pop_size, N, output_len)`.
- `mutate(replace_pos, new_sub_forest)` → new `Forest`; `crossover(left_indices, right_indices, left_pos, right_pos)` → new `Forest` of size `len(left_indices)`; both pure-torch splices with the legacy semantics (see `splicing.py`).
- `SR_fitness(inputs, labels, use_MSE=True, execute_mode="auto")` → fresh `(pop_size,)` float32 positive loss (writable; the pipeline mutates it in place).
- Dunders: `__getitem__` (int/0-dim → `Tree` view, slice → `Forest` view, `Tensor`/`ndarray` → copied `Forest`, python list → `Exception`), `__setitem__` (in place), `__iter__`/`__next__` (not reentrant: stores `iter_index` on the instance), `__len__`, `__add__`/`__radd__` (concat along dim 0; matching `input_len`/`output_len`), `__str__`/`__repr__` (renders every individual via sympy — slow), `__getstate__`/`__setstate__` (pickle via CPU numpy; restores to cuda).

### `descriptor.py` — `GenerateDescriptor(max_tree_len, input_len, output_len, const_prob=0.5, out_prob=0.5, depth2leaf_probs=None, roulette_funcs=None, const_samples=None, using_funcs=None, max_layer_cnt=None, layer_leaf_prob=0.2, const_range=None, sample_cnt=None)`
- Validates its inputs, derives `depth2leaf_probs` from `using_funcs`/`max_layer_cnt` when omitted (`check_tree_length` asserts `max_tree_len` fits the worst-case complete tree), and materializes the cuda tensors the engine and the algorithm-layer operators share: `depth2leaf_probs [MAX_FULL_DEPTH]`, `roulette_funcs [Func.END]` cdf, `roulette_ufuncs`/`roulette_bfuncs`/`roulette_tfuncs` (used only by the mutation operators), `const_samples [sample_cnt]`.
- `update(**kwargs)` returns a new descriptor (see Known Issues for its two gotchas).

### `generation.py` — `generate_population(pop_size, descriptor, device=None)`
Returns `(batch_node_value float32, batch_node_type int16, batch_subtree_size int16)`.
Vectorized level-wise torch that reproduces the descriptor-driven decision structure distribution-equivalently; draws from torch's global RNG, so `torch.manual_seed` / `torch.cuda.manual_seed` make generation reproducible (not bit-identical to the legacy taus88 stream); a length overflow raises `RuntimeError`.

### `splicing.py` — `mutate(...)` / `crossover(...)`
Pure-torch scalar-free splices: `_splice` updates ancestor `subtree_size` cells with the closed form `(j < pos) & (j + size[j] > pos)`, copies the donor subtree verbatim and shifts the tail; an invalid position or overflow copies the row UNCHANGED (silent, matching the legacy op); index tensors may be int8/16/32/64; out-of-range `left_indices` → `ValueError`; `pop_size_new == 0` → `ValueError`.

### `combined_forest.py` / `combined_tree.py` — the `data_info`-driven multi-output model
`CombinedForest(forests, data_info)` with `data_info = {out_name: [input_name, ...]}`; `CombinedForest.random_generate(pop_size, data_info, descriptors)` (one descriptor per sub-forest, each with `output_len == 1`).
The input-dict contract is single-sourced in `combined_forest.normalize_inputs` (its docstring is the contract of record, mirrored by all three forward docstrings): every required name must hold a scalar/0-D number, a 1-D `(K,)` tensor or a 2-D `(K, 1)` tensor; all names must agree on `K`; values are never broadcast; extra `x` keys are ignored completely (so e.g. `CustomLoss` can pass its `existing_data` including `"target"`); wrong shapes/keys raise `ValueError` naming the offending key, its shape and the sub-forest (no bare asserts); Python lists and `ndim >= 1` numpy arrays are rejected.
`CombinedForest.forward(dict) → {out_name: (pop_size, out_len_sub)}` is single-point only (a `K > 1` value raises and points to `batch_forward`); `CombinedForest.batch_forward(dict) → {out_name: (pop_size, K, out_len_sub)}`.
`CombinedTree(trees, data_info)` exposes each sub-tree by output name (`best.A.to_sympy_expr()`); its `forward` routes on `K`: `{out_name: (out_len_sub,)}` for `K == 1` (one point shared by the whole population) and `{out_name: (K, out_len_sub)}` for `K > 1` (the one-individual forest's pop dim is stripped).
Neither defines `SR_fitness`/`to_sympy_expr`, and neither has a `share_input` attribute; the dict input protocol is the only accepted form.

### `execute_mode` mapping
`Tree.SR_fitness`, `Forest.SR_fitness` (and the problem layer) accept the five strings `"torch"`, `"hybrid parallel"`, `"data parallel"`, `"tree parallel"`, `"auto"`.
The legacy `kernel_type` dispatch is gone and **all five map to the same ETL engine path** — the string no longer changes behavior (it is kept for API compatibility).

## Constraints
- This layer must stay free of custom kernels and of any compiled-extension import; the legacy extension name must not reappear anywhere in this directory (the old ops are kept only as historical schemas in `ENGINE_SPEC.md` Appendix A).
- `MAX_STACK = 1024`, `MAX_FULL_DEPTH = 10`, `NType`, `Func` are defined only in `utils.py` and are the single source of truth; function ids are positional across `Func`, `FUNCS_NAMES`, `FUNCS_DISPLAY`, `SYMPY_MAP`, so adding/renaming an operator means touching all four.
- `max_tree_len <= MAX_STACK` (asserted in `GenerateDescriptor`); `check_tree_length` asserts `max_tree_len` fits the worst-case tree of `using_funcs`/`max_layer_cnt`; `using_funcs` entries must be exact `FUNCS_NAMES` strings.
- Import-cycle rule: module-level imports go one way only (`utils` → `descriptor` → `forest` → `combined_forest` → `combined_tree`); `tree.py` imports `forest` inside its methods and `combined_tree.py` imports `combined_forest` inside its methods.
- `tree.py` / `forest.py` use `from .utils import *`, so every public name in `utils.py` leaks into their namespaces — avoid adding casual public names.
- Fitness contract: the engine returns fresh writable tensors; the repo-wide convention is `(pop_size,)` float32 MAXIMIZED and mutated in place by the pipeline (NaN → -inf).
- Several helpers still hard-code the `"cuda"` device string (descriptor tensors, `zero_generate`, `check_tensor`, `randint`); consistent with the rest of the repo, the same string also selects AMD GPUs under ROCm.
- `check_tensor` raises `TypeError` for non-tensors and relocates to `"cuda"` with `.detach().requires_grad_(False)` (kept deliberately: `../problem/brax_problem.py` relies on it); `randint` defaults to `device="cuda"`, `dtype=torch.int32`.
- Frozen, oracle-validated modules: `generation.py`, `splicing.py`, `etl_engine.py` — code changes need failing-test evidence and a separate commit; docstring/comment edits are free.
- `ENGINE_SPEC.md` sections 1–7 are the semantics of record; §12 lists every documented deviation (validation instead of UB, no fast-math, torch-RNG generation, composed `sinh`/`cosh` and `copysign`, constant trees never compiled, torch reduction order in `sr_fitness`, int tolerance rules).

## Design Decisions
- **Host decode + per-structure ETL graph** (instead of one generic interpreter graph): specialization makes re-evaluation of a structure a cache hit and keeps the graph free of dynamic dispatch; statics are re-passed by value so the cache key fully determines the executable.
- **numpy backend default, IREE opt-in**: the per-structure IREE compile (~0.4–0.6 s) dominates while numpy builds in ms and runs in ~1 ms; IREE stays first-class for callers who need CUDA execution and accept the compile economics.
- **Plain-torch operators** replace the kernels because they are portable and fast enough at these sizes; they keep the legacy splice semantics, including the silent no-op on invalid input.
- **Multi-output implemented in the engine** (oracle-verified for `output_len` 2/3) instead of raising; single-output trees with OUT_NODE tags raise `ValueError` instead of the legacy UB.
- Descriptors remain the shared source of randomness: `generation.py` reproduces the kernel's decision structure distribution-equivalently, while the algorithm-layer mutation operators keep consuming the descriptor's roulette cdfs.

## Known Issues / Notes for Agents
- The cache is per *(structure, shape)*: a fresh random population is a miss per individual (pop=1000 ≈ 5.5 s cold), constant trees never compile, duplicate structures hit; changing batch size or tree shape re-compiles — keep shapes stable in hot loops.
- Per-call GPU cost structure (the optimization target): `evaluate` copies the three population tensors to host once at decode (`_as_numpy:912-926`), copies `x` per call (and `labels` again in `sr_fitness`: `_input_to_numpy:929-941`, `_as_numpy:1331`), then loops over `pop_size` trees issuing one `build_cached` lookup + one `etl.run` per non-constant tree (`evaluate:1283-1290`, `_run_tree:1222-1242`, `_executable:882-904`) with NO cross-tree batching or grouping (dedup happens only inside `etl.CompileCache`: identical programs hit but are still run one by one), and finally moves the assembled `(pop, N, out_len)` result back to the caller's device once (`1294-1297`). Under the IREE backend every tree run additionally ends with a fresh host readback (`1242`), so one pop=N fitness pass costs N device invocations + N synchronizations.
- ETL per-call overheads on that path: `build_cached` re-encodes the static program and re-derives its sha256 cache key on every `_executable` call (hit or miss), and `etl.run` re-validates the statics by value in O(#tokens) per call; the program is a nested tuple of ints, so this is per-element Python work. ETL has no supported multi-program/“super graph” execution (only a hand-written interpreter graph with runtime `gather`/`select`/`while_loop`, or a device-resident `external_call` batched kernel), and no ETL GPU backend executes without per-graph AOT compilation (numpy = CPU interpreter, iree/xla compile per graph, tvm is CPU-only) — see the `etl` repo CONTEXT for details.
- `EVOGP_ETL_CACHE_SIZE` is read at import time (module-level `CACHE = CompileCache(...)` at `etl_engine.py:246`), while `EVOGP_ETL_BACKEND` / `EVOGP_ETL_IREE_DEVICE` / `EVOGP_ETL_IREE_OPT_LEVEL` are read per `evaluate` call (`_resolve_backend:848-879`); set the cache size before the first import.
- IREE is wired but experimental: not numerically verified for every shape, and O2/O3 are buggy on cuda — use O0/O1 only.
- Sympy rendering is a display artifact with documented edge gaps: `LOOSE_POW(0,0)` renders `1` (engine `0`), `DIV`/`INV` at a literal zero render sympy `zoo` while the engine computes NaN, `LOG(0)` renders `zoo` vs −inf, `MAX`/`MIN` with a NaN literal raise, and a negative-zero divisor loses its sign in loose ops (sympy has no −0.0).
- `repr(Forest)` / `str(forest)` build a sympy expr per individual (CPU sync, slow); `__iter__` is not reentrant; `Forest[int]`/`[slice]` views vs `Tensor`-indexing copies — writes through copies are lost.
- `GenerateDescriptor.update(**kwargs)` returns a new descriptor but also mutates the receiver's captured params (kwargs are sticky) and re-draws `const_samples` from `const_range` when the original had `const_samples=None`.
- `randint` truncates toward zero after `torch.rand` (biased for negative `low`); `dict2prob` is torch-only now and validates names against `FUNCS_NAMES`.
- `Tree.to_infix` warns and returns a single string when `output_len != 1`; `__python_forward` / `__assert_valid` are debug-only and not on any production path.
- The engine is correctness-first: the numpy backend does one `etl.run` per tree per call — profile before assuming throughput.
- `etl_engine.py` (~1.3k lines) is intentionally a single module: decode, op emission, the multi-output protocol and the cache policy form one compilation unit and were frozen together after oracle validation — do not split without failing-test evidence.

## Test Strategy
- No pytest suite inside this node; the validation stack is: (a) `ENGINE_SPEC.md` §12 self-consistency (compare against `Tree.__python_forward`, arity walks, splice identities); (b) the pytest suite under `../../tests/` — the primary regression gate (independent plain-torch oracle, randomized/adversarial equivalence, operator and API-compat tests, convergence smoke, plus the in-repo `goldens/` fixtures cross-validated against the legacy kernels before deletion).
- Run everything with the project venv (`/mnt/local-ssd/bchuang/evox/.venv/bin/python`), from the repo root, `PYTHONPATH=src`, and `ETL_DISABLE_LOCATIONS=1` for speed.
- Coverage gaps: IREE numeric parity per shape, non-default selection/crossover operators, robotics problems, and the multi-output training pipeline (the combined `forward`/`batch_forward` contract itself is covered by `../tests/test_combined_forward.py`).

## Routing Table
This directory has **no child subdirectories** — the layer is the files listed here; do not look for sub-nodes to delegate to.
- `utils.py` → constants/enums/helpers (`MAX_STACK`, `MAX_FULL_DEPTH`, `NType`, `Func`, `FUNCS_NAMES`, `SYMPY_MAP`, `check_tensor`, `randint`, `dict2prob`, `str_tree`); the single source of truth for the numeric constants.
- `descriptor.py` → `GenerateDescriptor` / `check_tree_length` (generation blueprint + roulette cdfs).
- `generation.py` → pure-torch population generation (`generate_population`).
- `splicing.py` → pure-torch `mutate` / `crossover` splices.
- `etl_engine.py` → decode + ETL `evaluate` + `sr_fitness` + `CACHE` / `cache_stats()` / `clear_cache()`.
- `tree.py` / `forest.py` → the public wrappers re-exported by `__init__.py`.
- `combined_forest.py` / `combined_tree.py` → `data_info`-driven multi-output model (`normalize_inputs` holds the shared input-shape contract).
- `ENGINE_SPEC.md` → authoritative semantics spec (incl. historical Appendices).
Siblings (read-only from here — escalate any needed write to the parent `./src/evogp`): `../algorithm/` (operators that splice via `Forest.mutate` / `Forest.crossover` and write OUT_NODE tags), `../problem/` (SR / classification / robotics problems), `../pipeline/` (training loop), `../tests/` (the pytest regression gate for this engine), `../../../test/`, `../../../example/`, `../../../tutorial/`.
