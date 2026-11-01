# `src/evogp/fused/` — the fused Triton evaluation route (standalone package)

## Intent
This package is the optional device route of `evogp` 2.0.0a2: ONE JIT-compiled Triton kernel launch evaluates a whole tree population off the legacy population tensors themselves (no evoxc graph, no per-node dispatch, no per-tree readback).
It is called DIRECTLY by `evogp.gp.fused_eval.FusedDeviceEvaluator` (which exports the `evoxc` channel tensors over DLPack and hands the kernel its own `out=` fitness buffer) and by `evogp.fused.check`; there is no route policy, no `Forest`/`Tree` dispatch and no `EVOGP_EVAL_MODE` here any more.
Nothing is compiled ahead of time: `triton` is imported lazily inside `kernels.kernels()` and the first call JIT-compiles (~1.6 s for the value kernel, from a writable `TRITON_CACHE_DIR`).
The `fused` extra (torch + triton) is optional: `import evogp.gp` and `import evogp` never import this package, and this package imports no `evoxc`.

## API Surface
`evogp.fused` re-exports `FusedUnsupported`, `fused_available`, `envelope_decline`, `evaluate_fused`, `sr_fitness_fused`, `fused_cache_stats`, `clear_fused_cache`, `MAX_OUT_LEN`, `resolve_block_rows`, `resolve_num_warps`, `resolve_max_scratch_mb`, `resolve_dispatch`, plus the `kernels` submodule.
`engine.evaluate_fused(batch_node_value, batch_node_type, batch_subtree_size, input_len, output_len, x, *, per_row, out=None, out_device=None, backend=None, block_rows=None, num_warps=None, max_scratch_mb=None)` is the value path and returns `(pop, output_len)` (`per_row=True`) or `(pop, N, output_len)`.
`engine.sr_fitness_fused(...)(batch_node_value, batch_node_type, batch_subtree_size, input_len, output_len, inputs, labels, use_MSE=True, *, ...)` is the SR loss path and returns `(pop_size,)` float32 positive loss; both honour `out=` in place.
`envelope_decline(pop_size, tree_len, rows, ...)` is the non-raising memory-envelope verdict; a call this route cannot serve raises `FusedUnsupported` and the CALLER decides the fallback.
`clear_fused_cache()` is the ONE reset entry (the route's own `XT_MEMO` / `TAG_MEMO` / `PLAN_MEMO` tables and the launch counters); `fused_cache_stats()` reports the `fused_*` counters.

## Constraints
Keep `import triton` inside `kernels.kernels()` — module scope must stay GPU-, triton- and evoxc-free so a plain-torch box can import the package and get a clean `FusedUnsupported`.
Never add a build step or a compiled extension: the kernels are JIT-compiled at the first call.
`engine.py` is one deliberately single large compilation unit because the pre-flight, the memory envelope, the transpose prep, the scratch table and the counters are checked together against the geometry `kernels.plan` returns; split it only with failing-test evidence.
The node ids come from the torch-free `evogp/_tables.py` (`DELTA`, `FUNCS`, `Func`, `MAX_OUT_LEN`, `OUT_NODE`, ...) — never redefine them here.
The `OUT_NODE` rule lives in `evogp/_multi_output.py`, which `evogp.gp.graph_eval` mirrors.
`EVOGP_FUSED_BLOCK_ROWS` / `_NUM_WARPS` / `_MAX_SCRATCH_MB` / `_HOST_MEMO` / `_DISPATCH` are the only knobs; `EVOGP_FUSED_MIN_POP` and the route policy were deleted with the `tree/` stack.

## Routing Table
- `./engine.py` → the call surface: pre-flight, envelope, transposed-operand prep, the prepared-call memo, the launch and the counters.
- `./kernels.py` → the Triton kernels (value, per-row, loss), the scratch geometry (`plan`) and the bound/jit launch dispatch.
- `./memo.py` → `Memo` (bounded LRU with `lookup`/`store`/`note_hit`/`reset`/`stats`) plus the identity key `token()` and its ABA guard `pin()`.
- `./check.py` → NOT present yet: the in-library battery the docstrings name (`python -m evogp.fused.check --stage all`) is still to be created from the deleted `tree/fused_check.py`.
- `./tests/` → NOT present yet: the route's pytest net (`src/evogp/tests/test_fused_route.py` before this move) is still to be relocated here.
