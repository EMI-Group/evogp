# example/ — runnable end-to-end demos of the shipped 2.0.0a2 stack

## Intent
Five flat, standalone scripts, one concern per file, each a complete runnable story that ends in printed numbers.
They demonstrate the stack this release actually ships: the torch-free functional stack `evogp.gp` (evoxc + numpy), its optional fused Triton objective, and the `evogp.evox_ext` adapter that presents the stack as an `evox` 2.0 `StdWorkflow`.
The torch stack (`evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline`) and the ten scripts that targeted it are gone in this release, so nothing here imports them.
This directory is documentation-by-example: it is the reference for wiring the public API, and the place to copy from when writing a new experiment.
There are no subdirectories, no `__init__.py` (not a package), no shared helpers and no test harness — each file is self-contained on purpose.
Every file carries a module docstring stating what it demonstrates, the exact run command, its requirements and its expected runtime / outcome.

## How to run
There is no build step: `import evogp.gp` works from a plain checkout, and the compiled graphs are traced at call time through the external `evoxc` package.
Run from the **repo root** with the reference venv — no `python`/`python3` on `PATH` has torch, triton or iree:
`PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 TRITON_CACHE_DIR=/tmp/triton_cache_example /tmp/rocvenv/bin/python example/<file>.py`
`/root/evox/src` is required only by `evox_workflow.py` (`import evox` needs that `src` directory; plain `/root/evox` does not work) — the other four need only `src` + `/root/evoxc`.
`EVOXC_DISABLE_LOCATIONS=1` is evoxc's tracing-speed knob (~4x faster traces, not part of any compiled program or cache key); every file sets it itself with `os.environ.setdefault`, so it is optional on the command line.
`TRITON_CACHE_DIR` only matters for `gp_device.py`'s fused route (a read-only `$HOME` makes the first launch die); a warm cache makes the second run of that file faster.
Every file is `main()`-guarded (`if __name__ == "__main__": sys.exit(main())`), so importing one does nothing — unlike the deleted torch-stack scripts, which ran their experiment at import time.
No file needs a GPU except the device branch of `gp_device.py`, which falls back to the host loop instead of failing.
`nanobind: leaked ...` lines on stderr at interpreter shutdown come from iree and are harmless; `evox_workflow.py` additionally prints two `RuntimeWarning: divide by zero / invalid value` lines from evoxc's numpy divide kernel (a GP population contains `/0` trees by construction).

## The set (one line per script)
- `gp_sr.py` → the canonical end-to-end run on `evogp.gp`: synthetic `x0**2 + x1**2` on 800 uniform `[-5, 5]` rows, inline `GPTreeConfig` + `GPConfig`, `run(config, SymbolicRegression(x, y).evaluate, generations=40, key=0, callback=...)`, then the winner's fitness, node count, **infix expression**, RMSE and three prediction/target rows.
- `gp_custom_loss.py` → the `CustomLoss` escape hatch: a classification-style accuracy objective (label `1.0` where `x0**2 > x1**2`, a single-output tree thresholded at 0) whose callable sees the WHOLE population at once, with the fitness contract printed explicitly (`[pop_size]` float32 **maximized**, `fitness == -loss`).
- `gp_uci_sr.py` → symbolic regression on a REAL table with no network and no new dependency: scikit-learn's bundled `load_diabetes` (442 rows x 10 features), z-scored inputs/target, `max_nodes=128`, and the honest verdict — the tree's RMSE / MAE / R^2 in the original target units next to an `lstsq` linear fit and the training mean.
- `gp_device.py` → the device-resident counterpart of `gp_sr.py`: the same driver with `backend="iree"` + a rocm/cuda device, `FusedDeviceEvaluator` when `fused_available()` and `interop_available()` both answer True and the torch-free `GraphEvaluator` in-graph route otherwise, timing the objective per call and the loop per generation, and falling back to the numpy host loop (reason printed) on a box with no device.
- `evox_workflow.py` → the `evogp.evox_ext` adapter: `GPAlgorithmConfig`/`GPConfig` + `SRProblemConfig`/`make_sr_problem` + `make_stdworkflow`, `workflow.init(seed=0)` (which builds BOTH graphs), ~20 `workflow.step(state)` calls with one `generation_fitness(state)` host readback each, then the same problem through the plain driver for a cost comparison.

## Per-script status (measured on the reference ROCm box)
Shipped settings, default routes, one process per run, repo root, `PYTHONPATH=src:/root/evoxc:/root/evox/src`, `EVOXC_DISABLE_LOCATIONS=1`, `TRITON_CACHE_DIR=/tmp/triton_cache_example`.

| file | shipped settings | measured result | external requirements |
|---|---|---|---|
| `gp_sr.py` | pop 256 x 40 gen, 800 rows, `max_nodes=64` | **runs**, 1.83–1.90 s wall (1.72 s in the loop, 43 ms/generation); fitness −127.47717 → −0.00000, best at generation 33; winner 11 nodes, `((x1 * ((x1 + 4.128) - 4.128)) + (x0 * x0))`, RMSE 0.00000 | numpy + evoxc |
| `gp_custom_loss.py` | pop 200 x 40 gen, 512 rows, `max_nodes=64` | **runs**, 1.36–1.40 s wall (0.96 s in the loop, 24 ms/generation); best accuracy 1.0000 at generation 13 (generation 0: 0.7305); `\|host accuracy − (best_fitness + 1)\| = 0.00e+00`, 0/512 misclassified | numpy + evoxc |
| `gp_uci_sr.py` | pop 300 x 300 gen, 442 rows x 10 features, `max_nodes=128`, `max_layer_cnt=6` | **runs**, 21.2–21.3 s wall (20.5 s in the loop, 68 ms/generation); best fitness −0.469925 (= −MSE, standardised); tree R^2 **+0.5301** vs `lstsq` **+0.5177** vs train mean 0.0000 (RMSE 52.788 / 53.476 / 77.006) | scikit-learn (already a hard requirement; dataset is bundled, no network) |
| `gp_device.py` | pop 256 x 20 gen, 512 rows, `max_nodes=64` | **runs**, 12.8 s wall with a warm Triton cache / ~14.0 s cold; route `fused`; 0.66 ms per evaluation call (median of 10, min 0.60), 1.73 ms/generation steady state, 125 ms/generation including the first-generation graph compile; fitness −118.84114 → −0.04918, winner 17 nodes | torch + triton + iree for the device route; falls back to a ~1.0 s host run without them |
| `evox_workflow.py` | pop 100 x 20 steps, 256 rows, `max_nodes=32` | **runs**, 1.79–1.81 s wall; `init()` 0.67 s (the one-time graph build), 32–35 ms/generation warm; best fitness −0.00000 at generation 12 (generation 0: −116.42137); plain driver on the same problem: 15.2–15.5 ms/generation, best −57.77399 | the `evox` checkout on `PYTHONPATH` (`/root/evox/src`); `evox` 2.0 pre-release, the only example that touches it |

Every row is deterministic in its fitness numbers (`key=0` / `seed=0`, key-based RNG, no global seed and no torch) — a second run prints identical fitness, node counts and metrics; only the timings move.

## Hard-coded settings a re-runner must know
`SEED = 0` everywhere: it is the driver's key (`run(..., key=0)`), the data seed and, in `evox_workflow.py`, the `workflow.init(seed=0)` value — there is no `torch.manual_seed`, because the stack's determinism contract is the key carried in the state.
Populations and budgets: `gp_sr.py` 256/40, `gp_custom_loss.py` 200/40, `gp_uci_sr.py` 300/300, `gp_device.py` 256/20, `evox_workflow.py` 100/20 (steps).
Tree width: `max_nodes=64` in the three synthetic demos, `128` in `gp_uci_sr.py`, `32` in `evox_workflow.py` (a complete tree of `max_layer_cnt=5` binary nodes needs 31).
Function set is `("+", "-", "*", "/")` in every script; constants are drawn from `const_range`/`sample_cnt` (`(-5, 5)` x 8 in `gp_sr.py` and `gp_device.py`, `(-1, 1)` x 8 in `gp_custom_loss.py`, `(-2, 2)` x 16 in `gp_uci_sr.py`) or from an explicit `const_samples` tuple (`evox_workflow.py`).
Operators: `elite_count=2` (3 in `gp_uci_sr.py`), `tournament_size=4`, `crossover_prob=0.9`, `mutation_prob=0.2`, `mutation_kind="subtree"` — the gp stack's closest analogues of the torch stack's `DefaultSelection(survival_rate=0.3, elite_rate=0.01)` and `DefaultMutation(0.2)`.
No file writes anything to disk: no CSVs, logs, checkpoints, plots or result files — all output goes to stdout, and nothing is read from `imgs/` or `data/` (the deleted scripts' CWD-relative asset paths are gone with them).

## Public API contract exercised
Canonical wiring (three files): `SymbolicRegression(x, y)` or `CustomLoss(x, loss, targets)` → `GPTreeConfig(...)` + `GPConfig(...)` → `run(config, problem.evaluate, generations=, key=0, callback=)` → `GPResult`.
Load-bearing symbols and call shapes:
- `run(config, evaluate, generations, *, key, pop_size, backend, device, callback, validate)` — the callback receives `(generation, state, [pop_size] float32 numpy fitness)` and is the only per-generation hook.
- `GPResult`: `best_fitness` (highest ever), `best_fitness_history` / `median_fitness_history` (one entry per evaluated generation, monotone because of elitism), `best_channels` (the winner as three 1-D host rows, a population of ONE), `best_generation`, `generations`, `state`.
- `init_gp(config, key, backend=, device=, validate=)` and `step(config, state, evaluate, key=, backend=, device=)` for a hand-driven loop (`gp_device.py` initialises a device state explicitly so the timed objective and the loop share it).
- `SymbolicRegression.evaluate` / `.loss` / `.predict` (the last one gives `[pop_size, n_rows]` and needs `[0]` for one tree), `CustomLoss.evaluate` / `.loss` with its `loss(predictions [pop, rows], targets [rows]) -> pop_size positive losses` contract.
- `row_length(channels)` — returns a per-row vector even for a single individual, so the node count is `int(row_length(result.best_channels)[0])`.
- Device surface: `GraphEvaluator(x, y, used_ids=tree_config, backend="iree", device=kind)`, `FusedDeviceEvaluator(x, y)`, `fused_available()`, `interop_available(kind)`, `readback_fitness(tensor, pop_size)`.
- `evox_ext` surface: `make_gp_algorithm(tree_config, gp_config)`, `make_sr_problem(x, y, used_ids=tree_config)`, `make_stdworkflow(algorithm, problem, backend=, device=, cache_dir=)`, `workflow.init(seed=)` / `workflow.step(state)`, `generation_fitness(state)`, `host_channels(state)`, `best_row(state)`.
- The device probe is inlined from the library's own idiom (`importlib.util.find_spec("iree.compiler")`, `find_spec("iree.runtime")`, then `evoxc.devices()` and a `device.kind` in `{"rocm", "cuda"}`) — nothing is compiled to answer "is a device usable here".

## What these examples surface about the shipped stack
`evogp.gp` has no expression exporter: `MIGRATION.md` step 9 (sympy / infix / png) is still open, so `gp_sr.py` carries a ~30-line local decoder as the stand-in and had to import the private `evogp.gp._constants` (`FUNCS_NAMES`, `FUNC_ARITY`, `Func`, `NType`) — there is no public name/arity table; the other three report a node count or the opcode rows instead of duplicating that renderer.
The gp stack is single-output (`output_len > 1` raises `NotImplementedError`), so `gp_custom_loss.py` expresses a classification objective as a threshold on ONE scalar prediction rather than one output per class.
`CustomLoss` always evaluates through the numpy host evaluator (`predict`), so it gives up the in-graph and fused device routes — those serve the mean-squared-error surfaces (`SymbolicRegression`, `sr_fitness_graph`) only.
The torch-free in-graph device route does not compile at `max_nodes=64` on this box: `GraphEvaluator(..., backend="iree", device="rocm")` raises an evoxc `BackendError` (`iree-compile` cannot distribute the channel graph's `stablehlo.reduce` on gfx942), while `max_nodes=32` compiles in ~2.5 s — which is why `gp_device.py` prefers the fused route and, on a torch/triton-less box, falls back to the host loop with that reason printed (`EVOGP_DEVICE_DEBUG=1` re-raises instead of falling back).
There is no public "is the in-graph `GraphEvaluator` usable here" probe (unlike `fused_available` / `interop_available`), so a device loop can only gate on the iree packages plus a trial call.
A `StdWorkflow` leaves the population unevaluated after `init()` and `generation_fitness(state)` returns the ELITE values only (offspring rows carry the `-inf` "not evaluated yet" mark), so step `k` reports generation `k-1`'s elites — a reporting-surface asymmetry against the driver's `run(callback=...)`, which gets the whole vector.
The workflow and the driver cannot be made to consume the same RNG stream from one seed (`StdWorkflow.init(seed=)` derives its own key), so `evox_workflow.py` compares the two on COST only and says so: at the shipped settings the workflow reaches −0.00000 and the driver −57.77399, and each route wins some seeds — never quote that pair as a convergence ranking.
`result.best_fitness` can print as `-0.00000` (an exactly-zero loss negated), which is a sign-of-zero cosmetic, not a bug.
Timing caveats worth knowing when re-measuring: `gp_device.py`'s ~13 s wall is ~7 s the one-time `evoxc.devices()` probe (it loads the iree runtime), ~1.8 s the generation-0 encoding compile and ~2.5 s the first variation-graph compile — the steady state is 1.7 ms/generation, so quote the steady-state and per-call numbers, not the wall clock, as the route's cost.

## Constraints for code in this directory
Keep every script self-contained and copy-pasteable: no imports from sibling example files, no shared helper module, one file = one complete runnable story.
Start every file with `import os` and `os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")` BEFORE the other imports, because tracing must see the switch.
Keep the `main()` guard: importing an example must not run it, and `main()` returns an int exit code (`sys.exit(main())`).
Import only from `evogp.gp` (and its submodules) and — for the workflow demo only — `evogp.evox_ext`; never from the deleted `evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline`, and never `import torch` / `import triton` (the fused route resolves both lazily inside the library; `fused_available()` / `interop_available()` are the probes).
Keep the knobs as plain module-level constants at the top (seed, pop, generations, rows, `max_nodes`, operator rates) — the directory's convention is "edit the constants and rerun", not a CLI.
Do not add a hard dependency: an example may use an already-declared package (numpy, scikit-learn) and must state any optional one in its docstring and here.
A demo must be sized to finish in well under a minute at its shipped settings (`gp_uci_sr.py` at ~21 s is deliberately the slowest; `gp_device.py`'s ~13 s is one-time compile/probe cost, not loop time).
Keep files in the 150–300 line range: the module docstring, the one-sentence-per-line reporting block and the route/fallback structure account for most of it, and `gp_device.py` at 300 lines is the widest legitimate case (three routes plus two fallbacks and the timing block).
Match the repo style: `from __future__ import annotations`, type hints, one-sentence-per-line docstrings, ~88-column lines.

## Test Strategy
No CI collects this directory and no pytest imports these scripts; the de-facto validation is running them, which is also how the numbers in the status table were produced.
Cheapest checks, in order: (1) `PYTHONPATH=src /tmp/rocvenv/bin/python -m py_compile example/*.py` (syntax only, no device); (2) `gp_sr.py` (≈1.8 s, fitness −127.47717 → −0.00000) — the canonical API smoke; (3) `gp_custom_loss.py` (≈1.4 s, best accuracy 1.0000, exit code is the contract assertion) and `evox_workflow.py` (≈1.8 s, needs `/root/evox/src`); (4) `gp_device.py` (≈13 s warm; check the printed `route` line is `fused` on a torch box, and that the fallback path prints a reason instead of a traceback — `EVOGP_DEVICE_DEBUG=1` re-raises); (5) `gp_uci_sr.py` (≈21 s, the slowest run that still fits a smoke budget).
Determinism check for any of them: run twice and diff the output with the timing fields stripped — every fitness number must be identical.
Ground truth for "is an example still valid" is the library itself: read `../src/evogp/gp/__init__.py` (the public surface), `../src/evogp/gp/problem.py` (`SymbolicRegression` / `CustomLoss` contracts), `../src/evogp/gp/workflow.py` (`run` / `GPResult`) and `../src/evogp/evox_ext/__init__.py` rather than trusting these scripts.
The library's own end-to-end runs remain the authority for the numbers behind them: `python -m evogp.gp.smoke` (the converging CPU smoke) and `python -m evogp.evox_ext.demo_stdworkflow` (the in-library workflow demo this directory's `evox_workflow.py` is the `example/` counterpart of).

## See Also
`../README.md` — the user-facing introduction, install instructions and the route table.
`../src/evogp/gp/CONTEXT.md` — the functional stack: API surface, constraints, device loop, and the reports of record (`DEVICE_LOOP.md`, `POPSCALE.md`).
`../src/evogp/gp/MIGRATION.md` — what the stack still lacks against the deleted torch stack (its steps 8/9 are the multi-output and display gaps these examples work around).
`../src/evogp/evox_ext/CONTEXT.md` — the `StdWorkflow` component contract behind `evox_workflow.py`.
`../src/evogp/gp/smoke.py` and `../src/evogp/evox_ext/demo_stdworkflow.py` — the in-library reference runs the examples are modelled on.
`../test/` — hand-run validation tooling and the benchmark harness, sibling and read-only from here; escalate any write need to the parent.
