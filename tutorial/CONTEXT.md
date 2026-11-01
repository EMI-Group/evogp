# tutorial/

## Intent
One hands-on Jupyter notebook that teaches the shipped `evogp.gp` stack: the torch-free functional genetic-programming layer built on `evoxc` graphs plus numpy host data.
`gp_intro.ipynb` walks the public API end to end on a synthetic symbolic-regression task -- configs, the driver, the one-line fitness contract, key-based determinism, the winning tree as infix, and the optional device path.
It teaches the current release only; the torch stack (`evogp.tree` / `algorithm` / `problem` / `pipeline`) is gone, so nothing here imports it.
This is a leaf directory: one notebook, no subdirectories, no build step, no `__init__.py`.
The notebook ships its real stored outputs (produced on the reference box, see Execution below), so a reader without a device or without a warm cache can still follow every result.

## `gp_intro.ipynb`
| Aspect | Detail |
|---|---|
| Cells | 15 (8 markdown / 7 code, code cells numbered 1-7); file is ~23 KB |
| Task | `x0**2 + x1**2` on 800 rows uniform in `[-5, 5]`; `x` is `[800, 2]` float32, `y` is `[800]` float32 |
| Public APIs taught | `GPTreeConfig`, `GPConfig`, `SymbolicRegression`, `init_gp`, `run(..., key=, callback=)`, `GPResult` (`best_fitness`, `best_fitness_history`, `best_generation`, `best_channels`), `row_length`, `problem.predict`, then `GraphEvaluator`, `FusedDeviceEvaluator`, `fused_available()`, `interop_available(kind)` |
| Train knobs | pop 256 x 30 generations, `max_nodes=64`, `using_funcs=("+","-","*","/")`, `max_layer_cnt=4`, `layer_leaf_prob=0.2`, `const_prob=0.5`, `const_range=(-5, 5)`, `sample_cnt=8`, `elite_count=2`, tournament 4, crossover 0.9, subtree mutation 0.2, `key=SEED=0` |
| Device-cell knobs | pop 128 x 5 generations, `max_nodes=32`, `backend="iree"` + the probed device kind, same operator rates |
| Measured runtime | ~15 s end to end on the reference ROCm box with a warm Triton cache: code cells 1-6 ~2.4 s (the whole CPU run is ~1 s of it), the device cell ~12 s (one-time Triton JIT plus the iree graph compile, ~7 s of which is the one-time `evoxc.devices()` probe) |
| Stored results | fitness `-127.47717` -> `-1.049584e-12` (monotone), a 23-node tree, RMSE `0.00000`, determinism all `True`, device route `fused kernel` at `-78.67564` with the numpy host evaluator agreeing to `7.63e-06` |

### Cell list
0. markdown -- what `evogp.gp` is (evoxc graphs + numpy host data, no torch/triton/device runtime), key-based determinism, the one-line fitness contract.
1. code -- imports: `numpy`, `GPTreeConfig`/`GPConfig`/`SymbolicRegression`/`init_gp`/`row_length`/`run` from `evogp.gp`, and `FUNC_ARITY`/`FUNCS_NAMES`/`Func`/`NType` from `evogp.gp._constants` for the local renderer. No output.
2. markdown -- the `x0**2 + x1**2` task, the numpy data, its own `np.random.default_rng(seed)`.
3. code -- `target_function` + `make_data`, `problem = SymbolicRegression(x, y)`, shapes and three data rows.
4. markdown -- why `GPTreeConfig` and `GPConfig` are frozen static-leaf dataclasses, field by field, and the derived tables.
5. code -- builds the two configs, prints the function set/arity/levels, the constant table, the depth schedule and the `GPConfig` repr.
6. markdown -- `init_gp` / `step` / `run`, `evaluate` as the only plug-in, elitism as the driver's job, the `callback` reporting hook.
7. code -- `init_gp(config, SEED)`, a 30-generation `run(config, problem.evaluate, generations=30, key=SEED, callback=report)` with the generation trace (0, every 5th, last), then the wall time, best fitness/generation and the monotone fitness path.
8. markdown -- determinism: one integer key seeds generation 0 and every operator draw, carried inside the state.
9. code -- a second identical `run`, printing both best values plus the equality of the best value, the full history and the winner rows (all `True`).
10. markdown -- `GPResult.best_channels` as host prefix-order `(opcode, payload, size)` rows, and the known gap: there is no public expression exporter (`MIGRATION.md` step 9).
11. code -- the local renderer copied from `example/gp_sr.py` (`render_expression` + `_apply`), the node count, the infix expression, the RMSE and two prediction/target rows.
12. markdown -- the device path: `backend="iree"` + `device=kind`, `GraphEvaluator` (torch-free in-graph, zero readbacks) vs `FusedDeviceEvaluator` (Triton kernel), the availability probes and the CPU fallback, and why the device population is its own smaller `max_nodes=32` one.
13. code -- the probe (`device_kind()` via `importlib.util.find_spec` + `evoxc.devices()`), `run_device()` choosing fused / in-graph / numpy host, a printed reason on any decline, then the route, generations, wall time, best fitness and a host re-evaluation of the winning tree.
14. markdown -- where to go next: `example/gp_sr.py`, `python -m evogp.gp.smoke`, `example/gp_custom_loss.py`, `example/gp_device.py`, `example/evox_workflow.py`, plus the two gaps (no exporter, single-output only).

## Execution
The notebook is executed with the stdlib-json runner recipe -- the project venv has no jupyter/nbconvert and no pip, so `nbconvert --execute` is not available.
The recipe: load the notebook JSON with `json`, `exec` its code cells in order in one shared `__name__ == "__main__"` namespace, capture stdout/stderr into `stream` outputs (merging adjacent same-name streams, as IPython does), emit every top-level expression whose value is not `None` as an `execute_result`, stop at the first exception recording an `error` output, and dump the JSON back with `indent=1, ensure_ascii=False` and a trailing newline (`nbformat` 4.2).
Run it from the repository root, so the notebook's relative assumptions and the `src/` layout both hold:
```
PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \
  TRITON_CACHE_DIR=/tmp/triton_cache_tutorial \
  /tmp/rocvenv/bin/python <runner>.py tutorial/gp_intro.ipynb --write
```
`/tmp/rocvenv/bin/python` is the reference venv (`python`/`python3` on `PATH` have neither torch nor iree); `/root/evox/src` is not needed by this notebook but matches the repo-wide command; `EVOXC_DISABLE_LOCATIONS=1` is evoxc's tracing-speed knob (~4x faster traces, not part of any cache key) and `TRITON_CACHE_DIR` only matters for the device cell's fused route.
Harmless noise: iree's `nanobind: leaked ...` block on stderr at interpreter shutdown, and evoxc's `RuntimeWarning: divide by zero` / `invalid value` from the numpy divide kernel on a population that contains `/0` trees.
A CPU-only box needs no Triton compilation: the device cell prints `no device (needs the iree packages and a rocm/cuda device)`, falls back to the numpy host loop and the notebook still ends green in ~4 s.

## Constraints
- Follow `../src/evogp/gp`, never the reverse: the library's signatures are the source of truth; file an issue rather than editing the library to make a cell run.
- Import only `evogp.gp` and its submodules; never the deleted `evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline`, and never `import torch` / `import triton` (the device cell uses the library's probes instead).
- The device cell must stay a probe with a graceful fallback: a box without `iree` or without a rocm/cuda device must not crash the notebook, and a declined device route must print its reason.
- No new dependency: numpy and the already-declared `evogp`/`evoxc` are the whole list.
- Keep the training settings small (pop 256 x 30 generations, device pop 128 x 5) so the full notebook stays in the tens of seconds; paper-scale runs live in `../example/`.
- Stored outputs are part of the deliverable: re-execute after editing a cell instead of hand-editing outputs (the runner above writes them).
- Notebook JSON stays `nbformat` 4.2 with `indent=1`, `ensure_ascii=False` and a trailing newline.
- No seeds global to the process: the run's reproducibility is the driver's `key`, and the data draw uses its own `np.random.default_rng(seed)`.
- `max_nodes=32` in the device cell is deliberate: `max_nodes=64` is still declined by the iree lowering of the 64-wide channel graph (`iree-compile` fails on its `stablehlo.reduce`) while 32 compiles; do not "fix" that by raising it.

## Known Issues
1. `evogp.gp` has no public expression exporter (`MIGRATION.md` step 9), so the notebook carries the same local decoder `example/gp_sr.py` uses and reads `evogp.gp._constants` (a private module) as the stand-in.
2. The stack is single-output only: `output_len > 1` raises until `MIGRATION.md` step 8 lands, so the tutorial teaches the SR surface and points at `CustomLoss` for everything else.
3. The in-graph iree evaluator declines `max_nodes=64` (channel-graph lowering), so the device section runs a separate `max_nodes=32` population rather than reusing the tutorial's own config -- the fused route does not have that limit.
4. The device and host paths agree on the winning tree only to float32 rounding (stored run: `7.63e-06` absolute on a fitness of `-78.7`, ~1e-7 relative); the notebook prints the difference instead of claiming bitwise parity.
5. Nothing asserts the stored outputs; correctness is the manual re-execution above plus the library's own gates.

## See Also
- `../example/gp_sr.py` -- the same walkthrough as a script (module constants, `main()`, report callback, the same renderer); the notebook's cells 0-11 mirror it.
- `../src/evogp/gp/smoke.py` -- the library's own converging CPU smoke (`python -m evogp.gp.smoke`, 200 trees x 100 generations on the harder `x0**4/(x0**4+1) + x1**4/(x1**4+1)` target).
- `../example/gp_device.py` -- the device cell as a script, with the timing block the notebook leaves out; `../example/gp_custom_loss.py` for `CustomLoss`; `../example/evox_workflow.py` for the `evox` 2.0 `StdWorkflow` adapter.
- `../src/evogp/gp/CONTEXT.md` and `../src/evogp/gp/DESIGN.md` -- the functional stack's API surface, dependency rules and design.
- `../src/evogp/gp/MIGRATION.md` -- what the stack still lacks against the deleted torch stack (steps 8/9 are the gaps above).
- `../README.md` -- the user-facing introduction, install steps and the route table.
