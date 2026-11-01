# `src/evogp/gp/` — the torch-free functional GP stack (`evoxc` + numpy)
## Intent
This subtree is the `evogp.gp` package: EvoGP's torch-free functional genetic-programming stack, built on `evoxc` (EvoX Compiler) tensors with numpy host glue.
In the 2.0.0a2 single-stack release it is the library's GP engine, beside `evogp.fused` (the device-native Triton route), `evogp.evox_ext` (the EvoX 2.0 `StdWorkflow` adapter) and the two shared torch-free modules `evogp._tables` and `evogp._multi_output`.
It solves single-output symbolic regression end to end: random generation (`init_population`) → the operator layer (`select`, `apply_crossover`, `mutate`, `apply_variation`) → a pluggable evaluation → the `init_gp` / `step` / `run` driver.
The idiom is Evox 2.0's (`/root/evox/src/evox/DESIGN.md`): frozen config dataclasses holding static leaves only, state as a dataclass whose leaves are `evoxc` tensors, plain module-level functions in the `init(config, key) -> state` / `step(config, state, evaluate) -> state` protocol, and a key-based RNG carried in the state instead of a global seed.
No module of the subtree imports torch, `evox` or `evoxc` at module import time, so `import evogp.gp` works from a plain checkout with `PYTHONPATH=src` alone.
The ONE torch boundary is `fused_eval.py`, which resolves torch, triton and the sibling `evogp.fused.engine` through `importlib.import_module` inside its functions, at call time only.
A whole loop also runs DEVICE-RESIDENT (`backend="iree"`, `evoxc.Device("rocm", 0)` on this box) with the objective on either of two device routes: `graph_eval.GraphEvaluator` — two `evoxc.run` calls per generation, the evaluation graph then the variation graph — or `fused_eval.FusedDeviceEvaluator` — ONE Triton launch per evaluation, the kernel reading the exported channel buffers and writing the fitness buffer the variation graph then consumes.
Both routes keep zero host syncs, zero readbacks and zero transfers between the two halves of a generation.
On a CPU box the same driver runs the numpy evaluator and converges in seconds: `python -m evogp.gp.smoke` measured 100 generations of pop 200 in 3.76 s on this box, `-0.27225` → `-0.10690` (best tree 57 nodes).
The economics of both device objectives are recorded in `DEVICE_LOOP.md` (the loop) and `POPSCALE.md` (the population axis); both are measurements of the pre-2.0.0a2 tree (see Known Issues).
EvoX 2.0 has no tree / GP / symbolic-regression component, so this stack plus `evogp.evox_ext` is EvoGP's contribution to the family rather than a port of anything that repository ships.

## Public API Surface
`gp/__init__.py` exports 44 names (`__all__`), grouped by concern:
- config: `GPTreeConfig`, `GPConfig`, `SELECTION_MODES` (`tournament` / `rank` / `roulette`), `MUTATION_KINDS` (`point` / `subtree` / `hoist`).
- state: `GPState`, `channels`, `replace`, `population_size`, `max_nodes`, `CHANNEL_NAMES`, `UNEVALUATED` (`-inf`).
- encoding: `init_population`, `live_mask`, `row_length`, `is_leaf`, `validate_channels`, `to_legacy_triple`, `from_legacy_triple`.
- operators: `select`, `apply_crossover`, `mutate`, `point_mutation`, `subtree_mutation`, `hoist_mutation`, `apply_variation`.
- problems and host evaluation: `BaseProblem`, `SymbolicRegression`, `CustomLoss`, `predict`, `predict_graph`, `sr_fitness_graph`.
- device evaluation: `GraphEvaluator`, `sr_fitness_channels`, `readback_fitness`.
- the optional fused backend: `FusedEvaluator`, `FusedDeviceEvaluator`, `fused_available`, `interop_available`.
- driver and results: `init_gp`, `step`, `run`, `GPResult`.
- cache surface: `cache_stats()`, `clear_cache()`.
Two names are spelled differently at package level because the bare names are the submodules: `apply_variation` is `variation.variation` and `apply_crossover` is `crossover.crossover`.
`cache_stats()` answers `{"graphs", "compile_cache_size", "compile_cache_maxsize", "layers", "device_layers", "device_graphs"}`: `layers` covers `encoding`, `selection`, `crossover`, `mutation`, `variation`, `workflow` (always 0 graphs — the driver traces nothing) and `problem` (its own `CompileCache`), `device_layers` holds `graph_eval` alone (all-zero counters on a host run) and `device_graphs` sums it.
`clear_cache()` resets every one of those layers and drops the shared `CompileCache`.
Submodule surfaces a caller imports directly (not re-exported): `evogp.gp._constants` (`MAX_STACK`, `MAX_FULL_DEPTH`, `NType`, `Func`, `FUNCS_NAMES`, `FUNC_IDS`, `FUNC_ARITY`, `FUNC_TYPE`), `evogp.gp.fused_eval.FusedDeviceEvaluator` / `export_to_torch` / `interop_available` / `ensure_triton_cache_dir`, `evogp.gp.graph_eval.GraphEvaluator` / `sr_fitness_channels` / `MultiOutputError`, `evogp.gp.smoke.main`, and the three harnesses' `main()`.
Canonical end-to-end run:
```python
from evogp.gp import GPTreeConfig, GPConfig, SymbolicRegression, run

problem = SymbolicRegression(x, y)                        # x [N, input_len], y [N]
config = (
    GPTreeConfig(max_nodes=32, input_len=8, using_funcs=("+", "-", "*", "/"),
                 max_layer_cnt=4, const_range=(-5, 5), sample_cnt=8),
    GPConfig(pop_size=200),
)
result = run(config, problem.evaluate, generations=100, key=0)
result.best_fitness, result.best_channels                   # host float, three channel rows
```
The device form keeps the same driver and the same result object: `run(config, GraphEvaluator(x, y, used_ids=..., backend="iree", device="rocm"), generations=100, key=0, backend="iree", device="rocm")`, or `FusedDeviceEvaluator(x, y)` in the same `evaluate=` slot on a box with torch and triton.

## Dependency Direction
`gp → evogp._tables` — `_constants.py` re-exports the ids from it and owns no value of its own.
`gp → evogp._multi_output` — `graph_eval` imports `MultiOutputError` from it, so the host-side `OUT_NODE` refusal is the one shared rule rather than a second copy.
Both are torch-free: `evogp._tables` imports nothing at all (no stdlib, no numpy) and `evogp._multi_output` imports numpy plus that module.
`gp` never imports `evogp.fused` at module scope; `fused_eval` resolves `evogp.fused.engine` through `importlib` at call time, which is what keeps `import evogp.gp` free of torch.
`evox_ext → gp` is the only reverse direction (`stdworkflow.py`, `gp_algorithm.py`, `sr_problem.py`, `demo_stdworkflow.py` import it), and nothing imports `gp` back.
Inside the subtree the order is `_constants` → `config` / `state` → `encoding` → `_runtime` → `selection` / `crossover` / `mutation` → `variation` → `evaluator` → `problem` → `graph_eval` / `fused_eval` → `workflow` → `smoke`, with the three `measure_*` harnesses and `_torch_baseline` on top of that.

## Constraints
- **Torch/`evox` purity is gated by source grep.** `tests/test_gp_stack_purity.py` (27 cases) asserts that no module of this subtree (`tests/` excluded, docstrings included) contains `import torch`, `from torch`, `import evox` or `from evox`, and its clean-subprocess case asserts that `import evogp.gp` with `PYTHONPATH=src` alone pulls neither torch, `evox` nor `evoxc` into `sys.modules`.
- `import evoxc` happens only at call time, through `encoding._evoxc()`, `_runtime.evoxc()` and `problem._require_evoxc`, each of which raises a `RuntimeError` naming the fix when the package is missing.
- **Channel triple**: three parallel `[pop_size, max_nodes]` channels in prefix (pre-order) order — `opcode` int32 (a `Func` id for a function node, an `NType` marker for a leaf), `payload` float32 (the constant value or the variable index), `size` int32 (subtree node count).
  Node 0 is the root, only `[0, size[:, 0])` is live in a row, a leaf is exactly a node with `size == 1`, and a function node's payload is `0.0`.
  Child addressing is the repository invariant `c1 = i + 1`, `c2 = c1 + size[c1]`, `c3 = c2 + size[c2]`.
- **Single-output only**: `init_population` raises `NotImplementedError` for `output_len > 1`, `GPTreeConfig.out_prob` is stored but unused, and every evaluator refuses an `OUT_NODE`-tagged population before any launch (`evogp._multi_output.MultiOutputError`, raised by `graph_eval` on host opcodes and by `FusedDeviceEvaluator`).
  `CustomLoss` is the escape hatch, including for a multi-output fitness.
- **Fitness contract**: `evaluate` returns a `(pop_size,)` float32 vector that is MAXIMIZED (losses negated, so the best tree has the highest value) and every non-finite entry is mapped to `-inf` — in-graph by `variation`, on the host by `problem._fitness_from_loss`, `graph_eval` and `fused_eval`.
  The driver never mutates the array a caller returned.
- **Determinism is key-based**: the RNG key (`evoxc.random`, splitmix64) lives in `GPState` and is advanced there, `run(..., key=0)` reproduces a whole run from one integer, and there is no global seed anywhere in the stack.
- **Device policy**: numpy is the default backend, the correctness backend and the only one that needs no device; `backend="iree"` with an `evoxc.Device` (or the `"rocm"` / `"rocm:0"` spelling) is the device route.
  A device executable rejects host input (`DeviceError`, no implicit transfer), and a device tensor's `.numpy()` refuses, so a download goes through `.to(evoxc.Device("cpu", 0))` (`encoding._as_numpy` is the one place that does it).
- `EVOXC_DISABLE_LOCATIONS=1` speeds tracing up and is recommended for real runs; a fused call additionally needs a writable `TRITON_CACHE_DIR` (`fused_eval.ensure_triton_cache_dir` defaults it to `$TMPDIR/evogp-triton-cache` only when the caller has not set it).
- Surviving environment knobs, all defaulted and all optional: `EVOGP_GP_COMPILE_CACHE_SIZE` (the shared `evoxc.CompileCache` of the encoding and operator layers, default 64), `EVOGP_GP_GRAPH_CACHE_SIZE` (the problem layer's own `CompileCache`, default 32) and `EVOGP_GP_EVAL_BUDGET_MB` (`predict`'s value-buffer budget, default 256 MiB, overridable per call via `budget_mb=` / `row_chunk=`).
  The `EVOGP_FUSED_*` geometry knobs (`BLOCK_ROWS` 256, `NUM_WARPS` 1, `MAX_SCRATCH_MB` 8192, `HOST_MEMO` 16, `DISPATCH` `bound`) are read by `evogp.fused`.
- **There is no route policy and no evaluation-mode environment variable**: the route is the caller's `evaluate=` argument, and the three that exist are host numpy (`SymbolicRegression` / `CustomLoss` / `predict`), the in-graph `iree` route (`GraphEvaluator`, `sr_fitness_channels`) and the fused Triton kernel (`FusedDeviceEvaluator`).
- `evoxc` has no eager mode: every op runs inside a trace, `evoxc.zeros` / `ones` / `full` build concrete tensors (wrap them in `evoxc.constant` inside a trace), `gather` is numpy `take` (hence `_runtime.take_per_row`), a Python `int` / `float` is not weakly typed (hence `_runtime.i32` / `f32`), `argsort(descending=True)` reverses tie order, and the stablehlo backend has no `cumsum` (hence `_runtime.cumsum_axis`) and tolerates only one rank-1 `scatter` per program.
- **Numerics are the recorded op semantics**: operand order (the first child is operand 1), protected `DIV`, the `LOOSE_*` family, `IF` on `first > 0`, exactly `±1` comparisons and the `MAX` / `MIN` NaN asymmetry are pinned by `evaluator.py`'s numpy emitter and cross-checked against the reference evaluator under `../../../test/oracle/`.
- Elitism lives in the variation graph (`elite_count` out of `GPConfig`): `variation` ranks the elite rows in-graph, copies them bit-identically in front of the offspring and returns the next fitness vector, which is why `step` needs no readback and `GPResult.best_fitness_history` is monotone non-decreasing.
- Validation is host-side Python (`encoding.validate_channels`, raising `ValueError` naming row and node) and is not on the hot path; `validate=False` is the documented opt-out on `init_population` / `init_gp` / `run` and on `FusedDeviceEvaluator`.

## Known Issues
- **Multi-output is not implemented**: `init_population` raises `NotImplementedError` for `output_len > 1`, the legacy bridge refuses an `OUT_NODE`-tagged row, `graph_eval` refuses one on the host, and `FusedDeviceEvaluator` refuses one before any launch; `CustomLoss` is the only way to score such a task.
- **No problem library beyond symbolic regression**: `problem.py` ships `SymbolicRegression` (MSE or MAE) and `CustomLoss` only — no classification, transformation, feature-building or robotics problem, and no benchmark suite inside this stack.
- **No export formats**: nothing here renders a tree to sympy, to an infix string or to a PNG; a solved tree leaves the stack as the three channel rows of `GPResult.best_channels`.
- **The fused device objective is dearer per call than the torch stack's own fused route in the same record** — 9.0×–19.8× on the same trees over pop 64 → 8 192 (`POPSCALE.md` §1.4/§5(a), a pre-2.0.0a2 measurement against a lane this tree no longer carries): `FusedDeviceEvaluator` exports three `evoxc` channel tensors over DLPack, converts them to the legacy triple in torch, writes the fitness buffer through `out=` and takes one `torch.cuda.synchronize()` per call, where that stack held the same layout as its own population tensors and launched pre-bound.
  What the stack wins is the GENERATION loop (1.83×–2.31× in the same record), because its operator layer is one traced graph.
- **Channel validation is a Python loop**: measured 6.9 ms for 256 trees averaging 52 nodes and 61.6 ms at 511 nodes each (≈0.5 µs per live node) on this box; no vectorized version exists and `validate=False` is the only mitigation.
- **A second `CompileCache` lives in the problem layer** (`problem._GRAPH_COMPILE_CACHE`, with its own size knob) beside the shared one in `encoding`, so the stack holds two compile caches with independent sizes and independent reset entries.
- **`FusedDeviceEvaluator` returns ONE reused device fitness buffer**, overwritten by the next device call: a caller that keeps it and re-passes it as `fitness=` reads a later generation's numbers, while the host route (`FusedEvaluator`) returns a fresh numpy array per call.
- **A device-resident `state` alone does not make `variation` / `step` device-resident**: the run's device is read off the state's channels, but the executable is built for the `backend` / `device` ARGUMENTS (both default to numpy / cpu), and every device test therefore passes them explicitly.
- **`problem`'s in-graph entry points are host-in/host-out**: `sr_fitness_graph` / `predict_graph` stage their graph inputs as host numpy arrays, so `evoxc.run` refuses a device backend at the run boundary; a device-resident in-graph run builds the executable directly through `GraphEvaluator` instead.
- **`from_legacy_triple` raises `IndexError` instead of its documented `ValueError`** for an unknown `NType` marker on the single-individual (three 1-D rows) form, because the message formats a row index out of a non-existent second axis; the library only reaches it from the tests.
- **The large-`k` elite `argsort` fallback is untested**: `variation._ARGSORT_ELITE_BREAK_EVEN` (50) is only crossed above 50 elites or at `elite_count == pop_size`, and `GPConfig` rejects the second case.
- **Four private channel parsers bypass the documented single one**: `evaluator._prepare_channels`, `fused_eval._channel_matrices`, `fused_eval._channel_objects` and `graph_eval._channel_triple` each accept their own subset of the three channel forms instead of going through `encoding._channel_arrays`, and `GPTreeConfig.func_type` has no reader outside a config test.
- **The two measurement records are historical**: `DEVICE_LOOP.md` and `POPSCALE.md` quote numbers taken on the pre-2.0.0a2 tree with torch comparison lanes that this harness set does not carry; nothing in this tree reproduces them, and they are read as records rather than as a current baseline.

## Test Strategy
Run the suite from the repo root with a writable Triton directory (the device modules JIT the kernel) and `evoxc` on the path:
`EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc:/root/evox/src TRITON_CACHE_DIR=<writable> /tmp/rocvenv/bin/python -m pytest src/evogp/gp/tests -q` → **349 passed / 0 skipped in 63.50 s** measured on this box (the validation box, where the device path is available).
The suite is this subtree's own: `src/evogp/gp/tests/` holds 17 test modules and 349 cases, and `evogp.fused` / `evogp.evox_ext` own their own suites.
Per file: encoding 33, config 26, operators 100 (selection 20, crossover 18, mutation 37, variation 25), problem 58 (contract 22, engine 6, graph 11, parity 19), workflow 36 (run 23, step 13), multi-output 12, purity 27, device-gated 57.
The count moves with the module list by design: `test_gp_stack_purity.py` parametrizes its source grep over every `*.py` of the subtree, so adding a module adds exactly one case.
The three DEVICE-GATED modules (`test_gp_stack_device_convergence.py` 2, `test_gp_stack_device_driver.py` 19, `test_gp_stack_device_fused.py` 36) skip where `evoxc`'s native DLPack export, `iree`, a visible `rocm` / `cuda` device or torch / triton is missing, so a box without a device runs fewer passes and more skips.
`conftest.py` puts `src` on `sys.path` at import time and sets no environment variable at all, so the suite is hermetic without a pin; a device test passes `backend="iree"` plus an explicit `evoxc.Device` itself and gates on `gp_operator_support.usable_device_kind()`.
Convergence smoke: `EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python -m evogp.gp.smoke` → 100 generations in 3.76 s, `-0.27225` → `-0.10690` (measured on this box; `--quiet` prints nothing and the returned `GPResult` carries the numbers).
Measurement harnesses, all `main()`-guarded and none of them part of the suite:
`python -m evogp.gp.measure_fused` (fused kernel vs in-graph vs host numpy, `--pops/--rows/--seed/--iters/--max-nodes/--no-graph/--numpy-graph/--json`, and `--check` for the parity re-verification on a fresh population, after three driver steps and over a short `gp.run` trajectory).
`python -m evogp.gp.measure_popscale` (the per-call axis, `--lanes numpy,gp,gp-fused,torch-batched,torch-per-tree` or a single `--lane`, `--pops/--rows/--reps/--warmup/--seed/--no-guard/--guard-wait/--json`).
`python -m evogp.gp.measure_generation` (the generation axis, `--sides gp,gp-fused,numpy,fused` or a single `--side`, `--cells` taking `<POP>` / `<POPxROWS>` cells or the `popscale` token, `--generations/--reps/--warmup/--json`).
Every harness runs one process per (side, cell) or per population (the parent re-executes itself and merges the children's JSON), reports warm medians behind an idle-device guard, and writes its records into `evidence/`.
The two torch lanes are self-contained: `_torch_baseline.TorchBatchedEvaluator` and `per_tree_forward` build their tensors from the population's own numpy rows over the legacy triple, so the harnesses carry no second stack with them.
A COLD number is only honest from a single-lane child (`--lanes <one lane> --reps 3 --warmup 1`), because in a multi-lane child one lane's Triton JIT and compiles land in the next lane's "cold".
The dev-only oracle lives outside the suite at `../../../test/oracle/` (`reference_eval.py`, its goldens and its CONTEXT.md) and is the independent diff target for numerics.

## Routing Table
- `__init__.py` → the 44-name public surface and the aggregate `cache_stats()` / `clear_cache()` over the seven host layers plus the one device layer.
- `config.py` → `GPTreeConfig` (tree blueprint: `max_nodes`, `input_len`, function set with weights, `max_layer_cnt`, the leaf / const / out probabilities, the constant table) and `GPConfig` (loop knobs), both frozen and hashable, with the derived tables (`roulette_funcs`, `func_arity`, `func_type`, `depth2leaf_probs`, `padded_widths()`, `padded_total()`), `SELECTION_MODES` / `MUTATION_KINDS`, and validation that raises `ValueError` instead of asserting.
- `state.py` → `GPState(opcode, payload, size, fitness, key)` (frozen) plus `channels`, `replace`, `population_size`, `max_nodes`, `CHANNEL_NAMES`, `UNEVALUATED`.
- `_constants.py` → the thin re-export of `evogp._tables` (ids, limits, `FUNC_IDS` / `FUNC_ARITY` / `FUNC_TYPE`); it owns no value.
- `encoding.py` → `init_population` (one traced generator graph per config), the host helpers `live_mask` / `row_length` / `is_leaf`, `validate_channels`, the `to_legacy_triple` / `from_legacy_triple` bridge, the single private channel parser `_channel_arrays`, and the shared `CompileCache` with `cache_stats()` / `clear_cache()`.
- `_runtime.py` → the internal plumbing the operator modules share (lazy `evoxc`, key resolution, the per-module executable memo in front of the shared `CompileCache`, `resolve_configs`, the device plumbing and the typed trace-time primitives); not part of the public surface.
- `selection.py` → `select` / `select_indices` (tournament, rank, roulette) plus `rank_probabilities`.
- `crossover.py` → the splice on channels: `crossover` (package alias `apply_crossover`), `splice`, `draw_positions`.
- `mutation.py` → `mutate` / `point_mutation` / `subtree_mutation` / `hoist_mutation`, the trace-time `apply_mutation` and the same-arity `swap_table`.
- `variation.py` → `variation` (package alias `apply_variation`): ONE traced graph per generation doing two parent draws, crossover, mutation and the elite splice.
- `evaluator.py` → `predict`, the numpy tree evaluator (the CPU path and the parity oracle) with `EVOGP_GP_EVAL_BUDGET_MB` row chunking and the op semantics of record.
- `problem.py` → `BaseProblem`, `SymbolicRegression`, `CustomLoss`, and the in-graph `predict_graph` / `sr_fitness_graph` with their own `CompileCache` and `graph_cache_stats()` / `clear_graph_cache()`.
- `graph_eval.py` → the device-resident in-graph evaluator: `sr_fitness_channels`, `sr_fitness_channels_numpy`, `readback_fitness`, `resolve_used_ids`, `GraphEvaluator`, the shared `MultiOutputError` refusal and `device_graph_stats()` / `clear_device_graph_cache()`.
- `fused_eval.py` → the optional fused (Triton) backend and the ONE torch boundary: `FusedEvaluator` (host channels), `FusedDeviceEvaluator` (device channels, DLPack export, `out=` write, one fence), `export_to_torch`, `fused_available` / `interop_available`, `ensure_triton_cache_dir`, the `InteropUnavailable` / `FusedUnavailable` declines.
- `workflow.py` → the driver: `init_gp`, `step`, `run`, `GPResult`, and its (always empty) `cache_stats()`.
- `smoke.py` → the `main()`-guarded CPU convergence run (`python -m evogp.gp.smoke`) with its `--pop/--generations/--data/--seed/--max-nodes/--tournament-size/--quiet` flags.
- `measure_fused.py` → the committed harness of the fused route against the in-graph and host-numpy routes (`--check` included).
- `measure_popscale.py` → the committed per-call harness of the population axis (`POPS` 64 → 8 192, `LANES`, `LANE_SURFACES`, `BASELINES`, `SIDE_MAX_POP`).
- `measure_generation.py` → the committed per-generation harness (`SIDES`, `POPSCALE_CELLS`, `SIDE_MAX_POP`, one process per side and cell).
- `_torch_baseline.py` → the measurement-only plain-torch baselines `TorchBatchedEvaluator` / `per_tree_forward` / `torch_ops` over the legacy triple, taking torch as an argument and importing none.
- `./tests/` → the subtree's pytest suite, its `conftest.py` and the shared `gp_operator_support.py` helpers; see its CONTEXT.md.
- `./evidence/` → the committed raw records the two reports are written from: `device_loop.json`, `device_loop_gp_fused_round2.json`, `popscale.json`, `popscale_eval.json` and `popscale_cold/{numpy,gp,gp-fused,forest-auto}.json` (the last one a record of a lane outside the harness's `LANES`).
- `DESIGN.md` → the constructions (channels, level-wise generation and argsort placement, splice arithmetic, selection / mutation sketches, "Evaluation and the driver").
- `EVOXC_API.md` → the verified `evoxc` contract; extend it rather than re-deriving the facts from the source.
- `POPSCALE.md` → the record of the population axis (per-call column, generation column, cold start, convergence), measured on the pre-2.0.0a2 tree.
- `DEVICE_LOOP.md` → the record of the device-resident loop (per-cell step / evaluation / variation tables, sync and transfer counts, convergence, the fused-vs-in-graph head-to-head), measured on the pre-2.0.0a2 tree.
- `../fused/` → the device-native Triton route this stack's `fused_eval` plugs in (`engine`, `kernels`, `memo`, `cases`, `check`, `contract`, `timing`, its own tests, and `FUSED.md` as the design/report of record).
- `../evox_ext/` → the EvoX 2.0 adapter that presents this stack as `StdWorkflow` components; the only subtree that imports `evox`.
- `../_tables.py` → the torch-free single source of truth for the ids (imports nothing) and `../_multi_output.py` → the shared `OUT_NODE` refusal rule and `MAX_OUT_LEN`.

## Notes for Agents
- Environment: repo root as CWD, `PYTHONPATH=src:/root/evoxc` (plus `/root/evox/src` for the adapter), the torch-carrying venv `/tmp/rocvenv/bin/python`, `EVOXC_DISABLE_LOCATIONS=1`, and a writable `TRITON_CACHE_DIR` for anything that touches the fused route.
  `import evogp.gp` must keep working with `PYTHONPATH=src` alone.
- Do not break the two import invariants: no module-scope `import torch` / `from torch` / `import evox` / `from evox` (grep-pinned, docstrings included) and no module-scope `import evoxc`; the lazy loaders in `fused_eval.py` and `encoding._evoxc()` exist for exactly that and must not be "tidied" into import statements.
- The ids are single-sourced in `evogp/_tables.py` and re-exported by `gp/_constants.py`: never edit a value in this subtree, and remember the tables are positional across `Func`, `FUNCS_NAMES`, `FUNC_ARITY` and `FUNC_TYPE`, so adding an operator touches all four.
- Read `EVOXC_API.md` before writing traced code: the `gather` / literal-dtype / `argsort` tie-order / `cumsum` / single-`scatter` facts there cost real debugging time and are not re-derivable from intuition.
- Keep files near or below ~1 000 lines: `fused_eval.py` (1 334) and `measure_generation.py` (1 137) are the only two over it, with `problem.py` (996), `graph_eval.py` (916), `encoding.py` (914) and `measure_popscale.py` (833) next, so extract before growing any of them.
- The numpy backend is the correctness backend and `iree` the device backend: check parity on numpy first, then on the device, and keep the same traced program serving both (no `backend == "iree"` branch in the math).
- `FusedDeviceEvaluator` returns the same device fitness buffer on every call: copy it before the next call if the numbers must be kept, and prefer `interop_available()` (a probe, never an exception) to gate the device route.
- A fully padding row (`size == 0` everywhere) is not a tree: `validate=True` rejects it, and without validation the fused kernel and the in-graph route return different arbitrary numbers, so never compare parity on one.
- Do not put timing loops in this subtree's tests: the measurements live in the three harnesses and their evidence files, and a number is quoted from a re-run of the harness, never from memory.
