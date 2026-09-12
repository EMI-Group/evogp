# `test/benchmark/SPEC.md` — legacy CUDA engine vs ETL "true jit" engine: performance comparison

Working spec v1 (2026-09-11), frozen by the orchestrator. Agents follow it; deviations must be logged
in the side's `NOTES.md` and flagged. The consolidation pass may fold this into `README.md`/`SUMMARY.md`.

## Question
Quantify, on simple numerical tasks and on one shared box, the performance difference between:

- **NEW** = this repo at HEAD — the ETL "true jit" evaluation engine (`src/evogp/tree/etl_engine.py`:
  host-side decode + per-structure `etl.build_cached` compiled graphs), default `"numpy"` backend,
  recommended `ETL_DISABLE_LOCATIONS=1`. The `"iree"` backend is an opt-in CUDA backend — one optional
  small datapoint only (clearly labeled), it is not the primary comparison.
- **OLD** = the legacy custom-CUDA engine at commit `b613503` ("last commit before the kernel-removal
  rewrite"): `src/evogp/cuda/*.cu` bound as `evogp.evogp_cuda`; `import evogp.evogp_cuda` is a hard
  prerequisite (no CPU fallback). Build recipe: `test/goldens/README.md` §"Environment recipe used to
  build and capture (historical)" — CUDA 12.6 toolchain `/opt/miniforge3/envs/wenet_tiktoken`,
  `TORCH_CUDA_ARCH_LIST=8.6`, `python setup.py build_ext --inplace`. There is **no internet** on this
  box; leftover build assets from the golden-capture build persist in `/tmp/evogp_t1a263`
  (`pydev/usr/include/python3.11` CPython headers, `siteutils/` sklearn 1.7.2 + joblib/threadpoolctl/
  cloudpickle — needed because `evogp.problem` imports sklearn, `wheels/`, and the successful build log
  `build6.log` showing the exact working invocation).

Both sides are driven through the **identical public API**, same workloads, same box.

## Common wiring (both sides)
- Interpreter: `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (py3.11, torch 2.6.0+cu124).
  Run from the checkout root with `PYTHONPATH=<checkout>/src` (OLD additionally needs
  `:/tmp/evogp_t1a263/siteutils` for sklearn). `python` is not on PATH.
- Device: pick the least-busy RTX A6000 at run time, export `CUDA_VISIBLE_DEVICES=<i>` (the code
  hard-codes `"cuda"`, i.e. `cuda:0` of the visible set). Record the index + a `nvidia-smi` snapshot.
- Seeds: `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)` before constructing anything.
- Timing: `torch.cuda.synchronize()` immediately before and after **every** timed region (OLD runs
  async CUDA kernels). Loop `pipeline.step()` manually (same code path as `run()`), with
  `is_show_details=False`, to get per-generation times.
- NEW: `os.environ.setdefault("ETL_DISABLE_LOCATIONS", "1")` **before** importing evogp
  (control run C0: set `"0"` instead). OLD ignores these variables.
- Shared-host coordination: serialize every heavy phase (builds, timed runs) across agents with
  `flock -w 7200 /tmp/gpbench_critical.lock bash -c '<cmd>'` — never nested; waiting on the lock is fine.
- Fairness caveat (document, don't fight it): NEW samples its population from torch's RNG, OLD from the
  legacy kernel stream — distribution-equivalent, not bit-identical. Compare **times**; fitness is only
  a convergence sanity check (both should improve over generations on W1/W2).

## Workloads
- **W1 — synthetic SR** (the `sr_test` problem):
  `func(x) = x0**4/(x0**4+1) + x1**4/(x1**4+1)`; `SymbolicRegression(func=func, num_inputs=2,
  num_data=1000, lower_bounds=-5, upper_bounds=5)` (pass ints for the bounds — the fixed-width scalar
  check quirk); `GenerateDescriptor(max_tree_len=T, input_len=2, output_len=1, const_prob=0.5,
  out_prob=0.5, using_funcs=["+","-","*","/"], layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8,
  max_layer_cnt=5)`; `GeneticProgramming(initial_forest=Forest.random_generate(pop_size=pop,
  descriptor=descriptor), crossover=DefaultCrossover(), mutation=DefaultMutation(mutation_rate=0.2,
  descriptor=descriptor.update(max_layer_cnt=3)), selection=DefaultSelection(survival_rate=0.3,
  elite_rate=0.01))`; `StandardPipeline(algorithm, problem, generation_limit=gens)`.
- **W2 — XOR parity** (the `example/basic.py` problem): 8-row 3-input parity table as in `basic.py`;
  `max_tree_len=64`, `max_layer_cnt=5`, `const_samples=[-1, 0, 1]`, pop=50, gens=100,
  `enable_pareto_front=True`.
- **W3 — eval-only microbench on a fixed forest**: rebuild the 64-tree / 63-node forest from the
  committed fixture `test/goldens/forest_random.npz` (keys `forest__node_value` (64,63) f32,
  `forest__node_type` i16, `forest__subtree_size` i16, `x` (64,2), `labels` (64,1); construct a
  `Forest` from them with the class constructor, tensors on cuda — same constructor exists at b613503
  and HEAD). Inputs `x`/`labels` from the same file. Time repeated `problem.evaluate(forest)` calls.

## Config matrix (same configs both sides unless noted)

| id | workload | config | reps | notes |
|---|---|---|---|---|
| C1 | W1 | pop 200, gens 100, tree_len 64 | 2 | main comparison point (current `sr_test` budget) |
| C2 | W1 | pop 1000, gens 100, tree_len 128 | 1 | legacy `sr_test` defaults. NEW: hard cap 30 min; if unfinished, finish ≥20 gens and mark partial + give projection |
| C2b | W1 | pop 1000, gens 20, tree_len 128 | 1 | bounded complete cu-scale datapoint |
| C4 | W2 | pop 50, gens 100 | 2 | tiny demo task |
| C5 | W3 | fixed 64-tree fixture, 25 evals | 1 | report cold (1st call) + warm (median of rest) |
| C5b | W3 | fresh `Forest.random_generate(pop=1000, tree_len=64)`, 10 evals | 1 | optional |
| C0 | W1 | pop 200, gens 100, tree_len 64 | 1 | NEW only: `ETL_DISABLE_LOCATIONS=0` control (tracing overhead) |
| C7 | W1 | pop 50, gens 3, tree_len 32, `EVOGP_ETL_BACKEND=iree` (`EVOGP_ETL_IREE_DEVICE=<i>`, `EVOGP_ETL_IREE_OPT_LEVEL=O1`) | ≤1 | optional, opt-in backend, cap 15 min, label clearly |

Priority if wall time runs short: **C1 → C5 → C2b → C4 → C2 → C0 → C5b → C7**.

## Required outputs
One JSON per run (side `results/` dir) with at least:
`{tag, side, evogp_file, evogp_commit, problem, config{pop,gens,tree_len,seed}, env{ETL_DISABLE_LOCATIONS,
EVOGP_ETL_BACKEND, CUDA_VISIBLE_DEVICES}, gpu, wall_s, per_gen_ms[], per_gen_best_fitness[],
best_fitness_final, cold_first_gen_ms, notes}` — C5/C5b instead:
`{tag, side, eval_kind, fixture, per_call_ms[], first_call_ms, median_warm_ms, notes}`.
Keep raw stdout logs next to the JSONs. `side` is auto-detected
(`importlib.util.find_spec("evogp.tree.etl_engine") is None` ⇒ `"old"`).

## Rules
- No workload/hyperparameter changes per side; anything that must change gets logged loudly in `NOTES.md`.
- Do not edit files owned by the other side, `SPEC.md`, or `CONTEXT.md`; do not touch the library
  (`src/`, `example/`, other `test/` files). If a library change seems needed, stop and report.
- Commit everything produced (scripts, notes, raw results).
