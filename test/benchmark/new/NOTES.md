# `test/benchmark/new/` — ETL "true jit" side (NEW) of the legacy-vs-ETL benchmark

Phase 1 deliverables of the new side: the side-agnostic harness `../bench_gp.py`, this side's matrix
runner `run_matrix.sh`, the smoke suite under `results/smoke/`, the first real config result
(`results/C5.json` + `C5.log`) and the validation checker `check_smoke.py`.
Methodology/config matrix: `../SPEC.md` (frozen). At Phase-1 commit time the harness was validated
against the ETL engine at `7e6cd0f` (`side="new"`).

> Sections below are the Phase-1 baseline. The matrix agent appends its own sections (results,
> comparisons, deviations) at the end — keep this header and the section titles stable.

## 1. Harness — `test/benchmark/bench_gp.py`

One file, two checkouts: it drives the **same public `evogp` API** against either engine, selected by
`PYTHONPATH` (`<checkout>/src`). No ETL-specific code outside the `ETL_*` env vars it exports; the only
repo-layout assumption is the `--fixture` default (`test/goldens/forest_random.npz`, cwd-relative), which
only the two eval tasks read.

```bash
PY=/mnt/local-ssd/bchuang/evox/.venv/bin/python            # py3.11 / torch 2.6.0+cu124
PYTHONPATH=src $PY test/benchmark/bench_gp.py --task sr --pop 200 --gens 100 --tree-len 64 \
    --tag C1 --reps 2 --json-out test/benchmark/new/results/C1.json
PYTHONPATH=src $PY test/benchmark/bench_gp.py --task eval64 --eval-reps 25 --tag C5 \
    --json-out test/benchmark/new/results/C5.json
```

Run it **from the checkout root** (it records `git rev-parse HEAD` of the cwd as `evogp_commit`).

CLI: `--task {sr,xor,eval64,evalfresh}` (required), `--pop`, `--gens`, `--tree-len`, `--seed 0`,
`--tag`, `--json-out`, `--reps 1`, `--eval-reps`, `--fixture`, `--no-disable-locations`,
`--max-wall-s`, `--min-gens 0`, `--notes`. Per-task defaults (SPEC): `sr` 200/100/64, `xor` 50/100/64,
`eval64` `--eval-reps 25`, `evalfresh` pop 1000/tree-len 64/`--eval-reps 10`.

Env handling (set **before** importing evogp): `ETL_DISABLE_LOCATIONS=1` unless
`--no-disable-locations` forces `"0"`. `EVOGP_ETL_BACKEND` / `EVOGP_ETL_IREE_DEVICE` /
`EVOGP_ETL_IREE_OPT_LEVEL` are read from the ambient environment (nothing forced); they are inert on the
old side. `side` is auto-detected: `importlib.util.find_spec("evogp.tree.etl_engine") is None` ⇒ `"old"`.
The JSON records `evogp_file`, `evogp_commit` (+`evogp_commit_error`), `etl_version`,
`env` (all `ETL_*`/`EVOGP_*` + `CUDA_VISIBLE_DEVICES`), `gpu` (name, `total_memory_bytes`,
`device_count`, torch/torch-cuda version), `cwd`, `harness`, ISO `started_at`/`finished_at`.

### Task wiring (mirrors `src/evogp/sr_test.py` / `example/basic.py` / the goldens recipe)

* `sr` (W1): `SymbolicRegression(func=x0**4/(x0**4+1)+x1**4/(x1**4+1), num_inputs=2, num_data=1000,
  lower_bounds=-5, upper_bounds=5)` (ints — the fixed-width scalar quirk) + `GenerateDescriptor`
  (`const_prob=0.5, out_prob=0.5, layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8,
  max_layer_cnt=5`) + `Forest.random_generate` + `GeneticProgramming`(Default ops,
  `mutation_rate=0.2`, `descriptor.update(max_layer_cnt=3)`, `survival_rate=0.3, elite_rate=0.01`) +
  `StandardPipeline(generation_limit=gens, is_show_details=False)`.
* `xor` (W2): inline 8×3 parity table + `SymbolicRegression(datapoints=…, labels=…)`, descriptor
  `max_tree_len=--tree-len`, `input_len=problem.problem_dim`, `output_len=problem.solution_dim`,
  `const_samples=[-1,0,1]`, `max_layer_cnt=5`, `enable_pareto_front=True`.
* `eval64` (W3): `Forest(input_len=2, output_len=1, batch_node_value/_node_type/_subtree_size)` rebuilt
  from the fixture's `forest__node_*`/`_subtree_size` (f32/i16/i16 → cuda) + `SymbolicRegression` on the
  fixture's `x`/`labels`.
* `evalfresh`: same fixture problem, fresh `Forest.random_generate(pop_size=--pop, descriptor=<W1-style
  descriptor with max_tree_len=--tree-len>)`. **Note:** no random problem data is drawn first, so the
  descriptor *is* the first RNG consumer here (W1 draws its problem data first) — the structures are
  therefore not the same draw sequence as `sr`/C1; only timing is comparable (documented in the JSON
  `notes`/`problem` fields).

`--rep`-level seeding: `torch.manual_seed(seed)` + `torch.cuda.manual_seed(seed)` immediately before
each rep's construction, i.e. **every rep repeats the same seed** — reps measure timing spread (and
`--reps 2` should reproduce identical fitness series; verified: both `SMOKE_SR` reps end at
`-0.25109699`). The ETL compile cache is per-process and is *not* flushed between reps, so rep 1
shows warm-cache numbers (this is deliberate but must be kept in mind when reading `--reps 2`: e.g.
`SMOKE_SR` rep0 1.32 s vs rep1 0.05 s for the identical 3-generation run).

### Timing conventions

* `torch.cuda.synchronize()` immediately before and after **every** timed region.
* Training: `pipeline.step()` is called manually `gens` times (the same code path as `run()`;
  `run()` itself is never called). `per_gen_ms[i]` = wall time of call *i*;
  `per_gen_best_fitness[i]` = `float(pipeline.best_fitness)` after it (a float `-inf` before the first
  update, then a 0-dim CPU tensor — the helper `float()`-casts both).
  `cold_first_gen_ms = per_gen_ms[0]` (includes the per-structure ETL build), `generation_loop_s =
  sum(per_gen_ms)/1000`, `best_fitness_final = per_gen_best_fitness[-1]`.
* Eval: `per_call_ms` over `--eval-reps` repeated `problem.evaluate(forest)` calls on the same forest;
  `first_call_ms = per_call_ms[0]` (cold builds), `median_warm_ms = median(per_call_ms[1:])` (`null`
  when there is <1 warm call).
* **`wall_s` = whole-rep wall clock, from just before construction (problem/descriptor/forest/algorithm/
  pipeline, or problem/forest for evals) to after the final sync** — it therefore contains the
  construction + cold-build cost that `generation_loop_s` excludes. `construct_s` is the extra field
  that splits it (measured up to the end of construction). Top-level `wall_s`/`harness_wall_s` are rep
  0's and the whole-process wall clock respectively.

### JSON schema

Training (`sr`/`xor`) — `{tag, side, task, evogp_file, evogp_commit, etl_version, problem,
config{pop,gens,tree_len,seed,reps}, env{}, gpu{}, wall_s, generation_loop_s, per_gen_ms[],
per_gen_best_fitness[], cold_first_gen_ms, best_fitness_final, partial, projection, complete,
per_rep[], notes, started_at, finished_at}` + extras `evogp_commit_error, cwd, harness, construct_s,
harness_wall_s`.

Eval (`eval64`/`evalfresh`) — `{tag, side, task, eval_kind, fixture, evogp_file, evogp_commit,
etl_version, config{pop,tree_len,seed,reps,eval_reps}, env{}, gpu{}, per_call_ms[], first_call_ms,
median_warm_ms, wall_s, partial, projection, complete, per_rep[], notes, started_at, finished_at}`
(+ the same extras; `problem` describes the exact forest/data wiring).

`per_rep[]` carries every per-rep value: `{rep, wall_s, construct_s, partial, projection, complete}`
plus training `{generation_loop_s, per_gen_ms[], per_gen_best_fitness[], cold_first_gen_ms,
best_fitness_final}` or eval `{per_call_ms[], first_call_ms, median_warm_ms}`. The top-level SPEC fields
**mirror rep 0** (the `per_gen_ms`/`per_call_ms` list objects are shared, so they are always in sync).

Non-finite floats (e.g. `best_fitness_final = -inf` when a rep is interrupted before generation 0, or
`-Infinity` fitness entries) are emitted as Python's `-Infinity`/`NaN` literals (default
`json.dump(..., allow_nan=True)`) — that is Python-json, not strict RFC JSON; `json.load` reads it back
fine.

### Early stop, robustness, exit codes

* `--max-wall-s`/`--min-gens`: checked after every completed generation (for evals: after every
  completed rep, with "generation" = one `evaluate` call). Stop when `elapsed >= --max-wall-s` **and**
  `ngens_done >= --min-gens`; a cap reached before `--min-gens` keeps going up to `--min-gens`
  (documented behaviour). The stop sets `partial: true` and
  `projection = {per_gen_ms_mean, gens_completed, gens_requested,
  projected_total_s_for_requested_gens}` (evals: the `calls_*` analogue). No other early stops.
* The JSON is rewritten **atomically (tmp file + `os.replace`) after every generation / eval call and
  at the end**, with `os.makedirs(dirname, exist_ok=True)` and mode 0644 — an external kill still
  leaves usable data (verified with `SIGKILL` in `SMOKE_KILL`: 37 completed generations in the file).
* `complete` (extra field) = "the harness finished writing this record": it is `false` only for a file
  abandoned by a hard kill/crash, which is how a consumer tells a truncated snapshot from a partial
  run.
* KeyboardInterrupt: the current rep is finalised (`partial: true`, `wall_s` set), the JSON is written
  and the process exits non-zero (`130`; under `timeout -s INT` the observable code is `timeout`'s
  `124`). A normal run — including a `--max-wall-s` partial one — exits `0`.
* All prints are `flush=True`, one compact summary line per rep, e.g.
  `[C1] side=new rep0/2: 100 gens in 26.5s, per-gen median 115.2ms, first-gen 1380.3ms,
  best_fitness_final=-0.1720`, evals: `[C5] side=new rep0/1: first 232.4ms, warm-median 21.4ms
  (25 calls)`. With `--json-out` omitted the full JSON is dumped to stdout at the end.

## 2. Old-compat audit (harness vs `b613503`)

Every evogp symbol/kwarg used by `bench_gp.py`, checked against the `b613503` blobs
(`git show b613503:<path>`):

| symbol / call form | b613503 file | result |
|---|---|---|
| `Forest(input_len, output_len, batch_node_value, batch_node_type, batch_subtree_size)` | `src/evogp/tree/forest.py:13` | **OK** — constructor byte-identical |
| `Forest.random_generate(pop_size=, descriptor=)` | `src/evogp/tree/forest.py:43` | **OK** — same staticmethod signature; body dispatches `torch.ops.evogp_cuda.tree_generate` |
| `forest.pop_size` / `.max_tree_len` / `.input_len` / `.output_len` | same ctor | **OK** |
| `GenerateDescriptor(max_tree_len, input_len, output_len, const_prob, out_prob, using_funcs, layer_leaf_prob, const_range, sample_cnt, max_layer_cnt, const_samples)` | `src/evogp/tree/descriptor.py:31` | **OK** — same kwargs; `const_range`/`sample_cnt` used only when `const_samples is None` |
| `GenerateDescriptor.update(**kwargs)` | `descriptor.py:186` | **OK** — returns a new descriptor |
| `GeneticProgramming(initial_forest, crossover, mutation, selection, enable_pareto_front)` | `algorithm/genetic_programming.py:21` | **OK** — file byte-identical |
| `DefaultCrossover()` | `algorithm/crossover/default.py:13` | **OK** (byte-identical) |
| `DefaultMutation(mutation_rate=, descriptor=)` | `algorithm/mutation/default.py:19` | **OK** (byte-identical) |
| `DefaultSelection(survival_rate=, elite_rate=)` | `algorithm/selection/default.py:15` | **OK** (byte-identical) |
| `from evogp.algorithm import …` exports | `algorithm/*/__init__.py` | **OK** — same as `example/basic.py@b613503` |
| `SymbolicRegression(func=, num_inputs=, num_data=, lower_bounds=, upper_bounds=)` | `problem/symbolic_regression.py:11` | **OK** |
| `SymbolicRegression(datapoints=, labels=)` | same ctor | **OK** |
| `SymbolicRegression.evaluate(forest)` (default `use_MSE=True`), `.problem_dim`, `.solution_dim` | same file | **OK** — old side only adds a `CombinedForest` assert + a `"torch"` `execute_mode` branch, neither triggered (`execute_mode` stays `"auto"`) |
| `StandardPipeline(algorithm, problem, generation_limit=, is_show_details=)`, `.step()`, `.best_fitness` | `pipeline/standard.py` | **OK** — file byte-identical; `.best_fitness` is `float("-inf")` then a 0-dim CPU tensor (the harness `float()`-casts) |
| `import evogp` / `evogp.tree` / `evogp.algorithm` / `evogp.problem` / `evogp.pipeline` | package `__init__`s | **OK** — old `evogp/tree/__init__.py` imports `evogp.evogp_cuda`, i.e. the CUDA extension is a hard prerequisite there (no CPU fallback, as expected) |
| `torch.manual_seed` / `torch.cuda.manual_seed` / `torch.cuda.synchronize` / `torch.cuda.get_device_name` / `torch.cuda.get_device_properties` / `torch.from_numpy(...).to("cuda")` | — | **OK** (SPEC-wide, old-torch-safe forms only) |

**No mismatch found with the SPEC-mandated call forms** — nothing had to be adapted, and no library
change was needed. Asymmetries the legacy side should know about:

1. **Fixture path — the legacy checkout has no `test/goldens/`.** `eval64`/`evalfresh` must be run with
   an explicit absolute fixture, e.g.
   `--fixture /mnt/local-ssd/bchuang/evogp/test/goldens/forest_random.npz` (verified from this worktree:
   `results/smoke/SMOKE_FIXABS.json`). The fixture is read-only reference data.
2. **`ETL_*`/`EVOGP_ETL_*` env vars are inert on the old side.** Consequences: the `--no-disable-locations`
   flag / `ETL_DISABLE_LOCATIONS=0` do not slow the legacy engine down (C0 is meaningless there — it is
   the same run as C1), and C7 (`EVOGP_ETL_BACKEND=iree …`) degenerates to a plain legacy run
   (pop 50 / 3 gens / tree_len 32). Label it if it is ever run on the old side.
3. `enable_pareto_front=True` is only passed for `--task xor` (as in `example/basic.py`), and the legacy
   generator/selection may hold extra state — keep the wiring untouched.
4. NEW samples populations from torch's RNG, OLD from the legacy kernel stream: compare **times**;
   fitness is only a convergence sanity check (and the old side's `--seed 0` structures will differ).

## 3. Smoke results (Phase-1 validation of the harness on the NEW engine)

All commands were run from the repository root with `PYTHONPATH=src`, the project venv, and
`CUDA_VISIBLE_DEVICES=4` (A6000; chosen at run time — see §5), each wrapped as
`flock -w 7200 /tmp/gpbench_critical.lock bash -c 'timeout -s INT -k 30 <T> env PYTHONPATH=src
CUDA_VISIBLE_DEVICES=4 $PY test/benchmark/bench_gp.py <args> > results/smoke/<TAG>.log 2>&1'`.
Logs: `results/smoke/<TAG>.log`; JSONs: `results/smoke/<TAG>.json`.

| tag | params | key numbers | JSON | flags |
|---|---|---|---|---|
| `SMOKE_HELP` (`--help`, no GPU) | — | usage printed, exit 0 | `results/smoke/SMOKE_HELP.log` | — |
| `SMOKE_SR` | sr pop=20 gens=3 tree_len=32 `--reps 2` | rep0: 3/3 gens, wall 1.32 s (construct 0.74 s), per-gen median 20.2 ms, first gen 541.5 ms, best **-0.2510970**; rep1: wall 0.05 s, median 10.9 ms (warm cache), identical fitness | `results/smoke/SMOKE_SR.json` | partial=false, complete=true |
| `SMOKE_XOR` | xor pop=20 gens=3 tree_len=64 | 3/3 gens, wall 0.81 s (construct 0.16 s), median 29.8 ms, first gen 593.9 ms, best **-0.5000** | `results/smoke/SMOKE_XOR.json` | =true |
| `SMOKE_EVAL64` | eval64 `--eval-reps 4` | first call **225.2 ms**, warm median **21.90 ms** (4 calls), wall 0.30 s | `results/smoke/SMOKE_EVAL64.json` | =true |
| `SMOKE_EVALFRESH` | evalfresh pop=100 tree_len=32 `--eval-reps 3` | first call **501.1 ms** (100 fresh structures = 100 cold builds), warm median **44.76 ms**, wall 0.73 s | `results/smoke/SMOKE_EVALFRESH.json` | =true |
| `SMOKE_FIXABS` | eval64 `--eval-reps 2 --fixture /mnt/local-ssd/bchuang/evogp/test/goldens/forest_random.npz` | first 222.8 ms, warm median `null` (only 1 warm call), wall 0.26 s | `results/smoke/SMOKE_FIXABS.json` | =true; validates the old-side fixture override |
| `SMOKE_C0` | sr pop=20 gens=2 tree_len=32 `--no-disable-locations` | 2/2 gens, wall 1.72 s, median 434.7 ms, first gen **839.9 ms** (vs 541.5 ms in `SMOKE_SR` → locations ~1.5x on a tiny run), best -0.2510970 | `results/smoke/SMOKE_C0.json` | `ETL_DISABLE_LOCATIONS=0`, =true |
| `SMOKE_C1FULL` | sr pop=200 gens=100 tree_len=64 (C1 scale) | 100/100 gens, wall **24.26 s**, per-gen median 90.5 ms, first gen 1452.0 ms, best **-0.1719539** | `results/smoke/SMOKE_C1FULL.json` | =true |
| `SMOKE_CAP` | sr pop=50 gens=200 tree_len=32 `--max-wall-s 4 --min-gens 5` | stopped after 88/200 gens (4.04 s), projection mean 36.5 ms/gen → ~7 s for 200 gens, best -0.2305344 | `results/smoke/SMOKE_CAP.json` | partial=true + projection |
| `SMOKE_INT` | sr pop=100 gens=500 tree_len=32, `timeout -s INT 6` | SIGINT after 25/500 gens, JSON written, exit non-zero (124 under `timeout`) | `results/smoke/SMOKE_INT.json` | partial=true, complete=true |
| `SMOKE_KILL` | sr pop=100 gens=500 tree_len=32, `timeout -s KILL 6` | SIGKILL: last incremental flush survived with **37 completed generations** | `results/smoke/SMOKE_KILL.json` | complete=false (truncated) |
| `C5` (real config, via the runner) | eval64 `--eval-reps 25` | first call **229.2 ms**, warm median **20.79 ms** (25 calls), wall 0.75 s, GPU 5 | `results/C5.json` (+`C5.log`) | =true |

Highlights / interpretation (new side, `"numpy"` backend, `ETL_DISABLE_LOCATIONS=1`):

* The flagship C1-scale path reproduces the `src/evogp/sr_test.py` reference **exactly**:
  generation-0 best `-0.251097`, first improvement at generation 43, final **-0.171954** at gen 100
  (docstring: “about -0.2511 … to about -0.1720, first improvement around generation 43”), i.e. the
  timing harness is measurement-neutral.
* Cost is dominated by **cold tracing**: first generation 0.5–1.5 s regardless of pop ≈ the ~65–115 ms
  steady-state per-generation cost (pop 200/100 gens: 1.45 s cold + ~9 s loop, the rest is construction
  and the first mutations); eval-only runs show the same structure (first call ≈ 10x the warm call).
* `SMOKE_EVALFRESH` normalizes that: 100 fresh trees → 501 ms first call vs 44.8 ms warm.
* Nothing in the harness changes the engine's arithmetic: the only knobs are the documented env vars.

Validation of the artifacts themselves: `check_smoke.py` re-reads every JSON in `results/smoke/`
(plus any path passed on the command line) and asserts schema completeness, `side=="new"`, step counts
vs `config`, mirroring of rep 0, the `partial`/`projection`/`complete` consistency rules, `median_warm_ms`
and `generation_loop_s` recomputation, plus the per-tag domain checks (`SMOKE_C0` must record
`ETL_DISABLE_LOCATIONS=0`, `SMOKE_C1FULL` must reproduce the `sr_test` reference fitness within 1e-3).
Last run: **10 JSON files (1 truncated by a hard kill), 184 checks, 0 failures** —
`PYTHONPATH=src $PY test/benchmark/new/check_smoke.py`.

## 4. Matrix runner — `test/benchmark/new/run_matrix.sh`

```bash
cd <checkout root>
./run_matrix.sh              # all: C1 C5 C2b C4 C2 C0 C5b C7 (SPEC priority order)
./run_matrix.sh C1 C5        # a subset
```

It refuses to run outside the checkout root (needs `test/benchmark/bench_gp.py` and `src/evogp`, else
exit 2), **does not** need an outer `flock` (it flocks internally, one level: every run is
`flock -w 7200 /tmp/gpbench_critical.lock bash -c "timeout -s INT -k 30 <T> env … $PY
test/benchmark/bench_gp.py … --json-out test/benchmark/new/results/<ID>.json"`, the timeout **inside**
the lock so only the benchmark is bounded). Per config it writes `results/<ID>.log` with a header
(`date -Is`, note, exact quoted command line + env, full `nvidia-smi`), the run output, and a footer
(`exit code: …`, `date -Is`, `nvidia-smi`). Failures never abort the batch; the final summary table
lists OK/exit/log and the script exits 1 iff any config failed.

| id | workload | params | timeout |
|---|---|---|---|
| C1 | W1 | `--task sr --pop 200 --gens 100 --tree-len 64 --reps 2` | 1500 s |
| C5 | W3 | `--task eval64 --eval-reps 25` | 900 s |
| C2b | W1 | `--task sr --pop 1000 --gens 20 --tree-len 128` | 2700 s |
| C4 | W2 | `--task xor --pop 50 --gens 100 --reps 2` | 1500 s |
| C2 | W1 | `--task sr --pop 1000 --gens 100 --tree-len 128 --max-wall-s 1800 --min-gens 20` | 3600 s |
| C0 | W1 (control) | `--task sr --pop 200 --gens 100 --tree-len 64 --no-disable-locations` (+`ETL_DISABLE_LOCATIONS=0`) | 2700 s |
| C5b | W3' | `--task evalfresh --pop 1000 --tree-len 64 --eval-reps 10` | 2700 s |
| C7 | W1 (opt-in IREE) | `--task sr --pop 50 --gens 3 --tree-len 32 --max-wall-s 900 --notes "opt-in IREE backend, cap 15min"` (+`EVOGP_ETL_BACKEND=iree EVOGP_ETL_IREE_DEVICE=0 EVOGP_ETL_IREE_OPT_LEVEL=O1`) | 960 s |

`PY` (default `/mnt/local-ssd/bchuang/evox/.venv/bin/python`) and `CUDA_VISIBLE_DEVICES` can be preset;
`EVOGP_ETL_IREE_DEVICE=0` is the *visible*-device index (README), i.e. exactly the GPU the run uses.
**`results/C5.json` + `C5.log` were produced by `./run_matrix.sh C5` during this validation run** (it is
the real C5 datapoint, not a throwaway; GPU 5, warm median 20.79 ms).

## 5. Environment notes (shared box, Phase-1 run)

* **GPU 7 is broken**: `nvidia-smi` reports `ecc.errors.uncorrected.volatile.total = 2263`, and any CUDA
  job on it dies with `CUDA error: uncorrectable ECC error encountered` (observed with the first smoke
  attempt). `run_matrix.sh` therefore *excludes* cards with a nonzero uncorrectable-ECC counter and logs
  the exclusion loudly — the only deviation from SPEC's "least utilization then least memory" pick rule
  (SPEC has no health concept). All other GPUs report `[N/A]` for that counter.
* Smoke runs used GPU 4; the `C5` matrix run picked GPU 5 (auto-pick, no `CUDA_VISIBLE_DEVICES` preset).
  Both are A6000s with other tenants (vLLM/merge jobs), so a few percent of run-to-run noise is expected
  — e.g. the same `SMOKE_CAP` settings completed 45 vs 88 generations inside the 4 s cap across two runs.
* `flock -w 7200 /tmp/gpbench_critical.lock` is shared with the legacy-side agent: waits are normal and
  are excluded from the timed region (the lock is taken before the harness starts). Never nest two flocks.

## 6. Handover to the matrix agent

* Run `./run_matrix.sh` (all eight configs, SPEC priority order) or a subset; each config refreshes
  `results/<ID>.json` + `results/<ID>.log`. C2 (`--max-wall-s 1800 --min-gens 20`) is expected to report
  `partial: true` with a projection — keep it, that is the SPEC's bounded high-pop datapoint.
* After a batch, `PYTHONPATH=src $PY test/benchmark/new/check_smoke.py results/*.json` re-validates every
  file (schema/counters/partial semantics, and `side` must stay `"new"`).
* Append results/comparison sections **below** this one; do not rewrite §1–§5 (they document the Phase-1
  harness revision the old side is mirrored against).

---

# Phase 2 — matrix results (matrix agent, `worker_T1_A8`)

## 7. Matrix runs

Phase 2 ran every SPEC config through `./test/benchmark/new/run_matrix.sh` (nothing dropped);
`results/C5.json`+`C5.log` are the Phase-1 datapoint and were deliberately **not** re-run. Exact commands,
from the repository root:

```bash
./test/benchmark/new/run_matrix.sh C1 C2b C4     # batch A  (SPEC priority: C1 -> C2b -> C4)
./test/benchmark/new/run_matrix.sh C2 C0         # batch B
./test/benchmark/new/run_matrix.sh C5b C7        # batch C  (queued at 21:25 behind B on the shared flock)
./test/benchmark/new/run_matrix.sh C7            # single retry after the first C7 attempt died (see §9)
```

Batch consoles keep the runner's own summary (`OK`/`FAIL` + exit code per config):
`results/_console_A.log`, `_console_B.log`, `_console_C.log`, `_console_C7retry.log`; each config log has the
header (date, exact quoted command + env, full `nvidia-smi`) and the `exit code:` footer.

Timeline (box-local time; "compute" excludes the shared-lock wait; GPU per the runner's auto-pick, no preset):

| batch | launched | per-config start→finish (s) | compute | result |
|---|---|---|---|---|
| A (GPU 4) | 21:15:22 | C1 21:15:22→21:16:19 (57) · C2b →21:16:51 (31) · C4 →21:17:50 (58) | 2 m 26 s | all exit 0 |
| B (GPU 0) | 21:18:26 | C2 21:18:26→21:49:19 (1853) · C0 →21:51:13 (113) | 32 m 47 s | all exit 0 |
| C (GPU 1) | 21:25:01 (lock wait ≈ 24 m) | C5b 21:49:20→21:49:34 (14) · C7 →21:52:13 (59) | 1 m 13 s | C5b 0, **C7 exit 1** |
| C7 retry (GPU 1) | 21:55:19 | C7 →21:56:25 (66) | 1 m 6 s | exit 0 |

**Total new-side compute ≈ 37.5 min** (budget ~90 min) → all eight configs have a datapoint; nothing was
dropped for time.

Headline table — all runs `side="new"`, `evogp_commit=9159e10`, `etl_version=0.1.0`, torch 2.6.0+cu124,
NVIDIA RTX A6000, `ETL_DISABLE_LOCATIONS=1` (C0: `"0"`), GPU index from the log header.
`wall_s` = rep 0; `proc` = whole harness process. "median" = median of rep-0 `per_gen_ms` / `per_call_ms`.

| id | task | params | reps | wall_s (proc) | per-gen median ms | cold first-gen ms | gen-loop s | best fitness final | GPU | partial / projection | files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | sr (W1) | pop 200 / gens 100 / tree 64 | 2 | 24.90 (50.52) | 88.3 | 1480.2 | 23.97 | -0.171953946 | 4 | no | `C1.json` + `C1.log` |
| C1 rep1 | sr (W1) | (same seed, identical run) | — | 24.59 | 100.5 | 1780.0 | 24.43 | -0.171953946 | 4 | no | — |
| C2b | sr (W1) | pop 1000 / gens 20 / tree 128 | 1 | 24.45 (25.76) | 803.9 | 5703.9 | 23.62 | -0.205357924 | 4 | no | `C2b.json` + `C2b.log` |
| C4 | xor (W2) | pop 50 / gens 100 / tree 64 | 2 | 40.62 (49.88) | 405.1 | 754.7 | 40.26 | -0.250000000 | 4 | no | `C4.json` + `C4.log` |
| C4 rep1 | xor (W2) | (same seed, identical run) | — | 8.04 | 74.8 | 48.8 | 7.87 | -0.250000000 | 4 | no | — |
| C2 | sr (W1) | pop 1000 / gens 100 / tree 128, `--max-wall-s 1800 --min-gens 20` | 1 | 1820.29 (1820.79) | 26230.3 (mean 20442.7) | 5681.1 | 1819.40 | -0.036228847 | 0 | **partial 89/100**, projection 2044.3 s | `C2.json` + `C2.log` |
| C0 | sr (W1) control, `ETL_DISABLE_LOCATIONS=0` | pop 200 / gens 100 / tree 64 | 1 | 93.45 (94.29) | 336.7 (mean 925.1) | 4650.4 | 92.51 | -0.171953946 | 0 | no | `C0.json` + `C0.log` |
| C5b | evalfresh (W3′) | pop 1000 / tree 64 / 10 evals | 1 | 10.48 (11.87) | first call **5622.2**, warm median **511.14** | — | — | — | 1 | no | `C5b.json` + `C5b.log` |
| C7 | sr (W1), `EVOGP_ETL_BACKEND=iree`, `IREE_DEVICE=0`, `O1` | pop 50 / gens 3 / tree 32 | 1 | 61.20 (62.35) | 21830.8 (per-gen `[21831, 12592, 26017]`) | 21830.8 | 60.44 | -0.251096994 | 1 | no (retry) | `C7.json` + `C7.log` |
| C5 (Phase 1) | eval64 (W3) fixture 64 trees / 63 nodes, 25 evals | `--task eval64 --eval-reps 25` | 1 | 0.75 (1.81) | first call **229.2**, warm median **20.79** | — | — | — | 5 | no | `C5.json` + `C5.log` |

Rep-level detail beyond the table: C1 rep0 wall 24.90 / loop 23.97 / median 88.3 / **mean 239.7** / first
1480.2; C4 rep0 wall 40.62 / loop 40.26 / median 405.1 / mean 402.6 / first 754.7, its `nvidia-smi` shows
other tenants on GPUs 0-6 throughout.

Validation of every artifact: `PYTHONPATH=src $PY test/benchmark/new/check_smoke.py results/*.json` →
**8 JSON files (0 truncated), 160 checks, 0 failures**; all report `side="new"`, rep-0 mirroring, `partial`
and `projection` consistency, and `config.gens == len(per_gen_ms)` (89 for C2, by design).

## 8. Observations

**Cold vs warm per-gen (the compile-tax signature).** Every config's generation 0 costs several times its
steady-state generation: C1 1480 ms vs 88 ms median (16.8x), C0 4650 vs 337 (13.8x), C2b 5704 vs 804 (7.1x),
C4 755 vs 405 (1.9x). The ETL engine builds each *distinct* `(structure, shape)` once per process, so gen 0
pays a cold build for all 200/1000 pure-random trees while later generations re-use survivors (elites +
`survival_rate=0.3`) and only build the new offspring. The eval-only configs show the same ratio without any
training dynamics: C5 first call 229.2 ms vs warm median 20.79 ms (11.0x) and C5b 5622 ms vs 511 ms (11.0x) —
for C5b that is 1000 fresh structures costing 5.6 ms each to build vs 0.51 ms each to merely run.

**C1 rep0 vs rep1 — cross-rep cache warmth is nil at this scale.** The two reps replay the *same* seed, hence
bit-identical fitness series, yet rep1 is not faster: wall 24.59 s vs 24.90 s, median 100.5 ms vs 88.3 ms, and
rep1's first generation is *larger* (1780 ms vs 1480 ms). A pop-200 × 100-generation run touches on the order
of 10-20k distinct structures (each generation's offspring are almost all new) against the engine's 4096-entry
`etl.CompileCache`, so nearly all of rep0's structures have already been evicted when rep1 starts. C4 is the counter-example: xor (pop 50, tree 64) rep1 = 8.04 s vs rep0's 40.62 s
(**5.1x**), with rep1's gen 0 at 48.8 ms — an almost entirely warm cache (`C4` timing series: rep1 max per-gen
176.8 ms, no gen > 200 ms). Rule of thumb: cross-run/cross-rep reuse only materializes when a run's distinct
structure count fits in the 4096-entry cache.

**C0 vs C1 — `ETL_DISABLE_LOCATIONS` 0 vs 1 (tracing overhead).** Identical workload/seed; the only difference
is the ETL call-site capture. C0: wall 93.45 s, loop 92.51 s, median 336.7 ms, first gen 4650.4 ms. C1 rep0:
24.90 s / 23.97 s / 88.3 ms / 1480.2 ms. Ratios: **wall 3.75x, loop 3.86x, per-gen median 3.81x, first-gen
3.14x** — i.e. the README's "~4x" holds up on a real training run. The fitness series is *bit-identical*
(`C0.per_gen_best_fitness == C1.rep0.per_gen_best_fitness`, verified), so the variable is purely a build-time
knob.

**Per-gen series: bloat dominates the tail (spikes and trend).** 10-generation block means of `per_gen_ms`:

| config | decade block means (gens 0-9, 10-19, …) |
|---|---|
| C1 rep0 | 296, 75, 100, 69, 111, 82, 132, 106, **237, 1191** |
| C1 rep1 | 286, 72, 75, 69, 90, 98, 251, 134, **212, 1157** |
| C0 | 952, 259, 309, 251, 298, 304, 382, 416, **927, 5151** |
| C4 rep0 | 215, 194, 234, 374, 506, 537, 502, 493, 531, 440 |
| C4 rep1 | 59, 53, 58, 72, 93, 77, 84, 106, 89, 97 |
| C2 (gens 0-88) | 1337, 991, **8802, 20888, 26725, 27674, 33163, 32594, 33073** (last block = gens 80-88) |

* C1's mid-run steady state is ~70-130 ms/gen, but the last ten generations (91-99) spike to 1275-1959 ms
  (median 88.3 ms vs mean 239.7 ms overall — the mean is 2.7x the median because of that ramp). C0 shows the
  same shape scaled 4x (last decade 5151 ms).
* The trend, not noise, is the story: there is no parsimony pressure in the W1 selection, so crossover/mutation
  grow the trees toward `max_tree_len` and each structure's build+run cost grows with it; once a run creates
  more distinct structures than the 4096-entry cache, the elite/survivor carry-over stops being a cache hit.
* C2 makes it explicit: after ~40 generations the cost settles at a **30-38 s/gen plateau** (~30 ms per
  structure build at full-size 128-node trees), i.e. the growth phase is over and the population is fully
  bloated. C2b (same config, first 20 generations) reproduces C2's fitness series *exactly* and its per-gen
  times within a median of 43 ms (max 190 ms) — the trajectory is deterministic and shared-box noise is small
  compared to the bloat.

**C2 vs the `sr_test.py` "needs hours" claim (explicit).** The docstring (`src/evogp/sr_test.py`, lines 9-13)
says the original CUDA-scale validation (`pop_size=1000`, `max_tree_len=128`, 100 generations) "therefore needs
hours here". Measured on this box (numpy backend, `ETL_DISABLE_LOCATIONS=1`, A6000):

* **C2 did not complete**: it stopped on its SPEC soft cap after **89/100 generations in 1820.3 s**
  (`partial: true`; the cap is a SPEC-mandated bound, not a failure).
* Measured rate: mean **20.44 s/gen** over those 89 gens, median **26.23 s/gen**, plateau **30-38 s/gen** after
  ~gen 45 (so a "per-gen rate" is only meaningful together with the generation index).
* SPEC projection (mean × requested gens, as written into the JSON): **2044.3 s ≈ 34 min** for the full 100
  generations. Continued bloat over the last 11 gens would push it to roughly ~0.6 h.
* Verdict: the docstring overestimates by roughly 3.5x if "hours" means ≥2 h — the config is a **tens-of-
  minutes** run, not an hours-long one — but the underlying reasoning is confirmed (cost scales with
  pop × tree size *and* grows with the bloat). The trap worth flagging is the opposite direction: C2b's 20
  generations average 1.18 s/gen, which would linearly extrapolate to ~118 s (2 min) for 100 gens — 17x too
  optimistic, because the growth phase lands between generation 20 and 45. Extra data point: tree_len 128 also
  solves the task far better than C1's tree 64 budget (best -0.0362 at gen 89 vs -0.17195 at gen 100), so the
  cost buys solution quality, not just more time.

**IREE economics (C7).** Opt-in backend at a deliberately tiny scale (pop 50 / 3 gens / tree 32):
per-gen `[21831, 12592, 26017] ms`, whole run 61.2 s. Per *structure* that is 0.44-0.52 s of IREE MLIR→vmfb
compile, versus the numpy backend's measured cold-build costs of 5.6 ms/structure (C5b, 1000 fresh trees) and
27 ms/structure (Phase-1 `SMOKE_SR` gen 0) — i.e. **roughly 16-90x per build, workload-dependent**. Scaled to
a whole generation at that population an estimated ~50 ms of numpy work (extrapolated from `SMOKE_SR`, no exact
same-config numpy datapoint exists) becomes 12-26 s → ~2.5 orders of magnitude. Generation 1 is not cheaper
than generation 0 despite survivors, so the cost is per new structure, not a one-off warm-up. Resource side:
the IREE path leaks (see §9) — it is the practical limit on this backend, not the raw compile time.

**Shared-box noise caveats.** The runner auto-picks a GPU per batch (C1/C2b/C4 → 4, C2/C0 → 0, C5b/C7 → 1;
GPU 7 excluded for its ECC fault), and all A6000s carry other tenants (vLLM/merge jobs holding 15-24 GB; GPU 6
was at 82-99% during the runs). Since the numpy backend does its build+interpret work host-side, the timed
regions are mostly CPU-bound under the runner's 8-core `CPUQuota`; measured noise is small relative to the
effects above — C1's two identical reps differ by 0-14% per gen (up to ~30% on the tail spikes), the
identical-trajectory C2b-vs-C2 gens by a median of 43 ms, and C5's 25 identical calls span 20.6-21.2 ms (<3%).
Treat the medians as ±10%, not exact.

## 9. Deviations / drops

1. **Nothing dropped.** All eight SPEC configs ran; C5 kept from Phase 1 per plan. Total new-side compute
   ≈ 37.5 min (cap ~90 min).
2. **C7 failed once, succeeded on the single allowed retry** (SPEC §C7 is optional and clearly labelled; both
   attempts ran the frozen runner command unchanged). Attempt 1 (21:51:14, 59 s, exit 1) died inside the IREE
   compile of the very first generation:
   `OSError: [Errno 24] Too many open files` at
   `etl/backends/adapters/iree.py:987 _compile_mlir → iree/compiler/tools/binaries.py:199 invoke_immediate →
   subprocess.run → os.pipe()`, leaving only a truncated snapshot (`C7.json`, `complete: false`, 0 gens — not a
   usable partial, so it was replaced by the retry). The trigger was the launcher's FD limit (1024 soft) — the
   standard harness environment sets `LimitNOFILE=65536`, and the retry was launched with exactly that (only
   deviation, environment parity, not a SPEC change) and completed all 3 generations, exit 0. Even then the
   process printed `nanobind: leaked 343 instances` (`HalDevice`/`HalBuffer`/`MappedMemory`/`HalAllocator`) and
   `leaked 342 keep_alive records` — the ETL IREE adapter + iree-runtime bindings leak per compiled structure,
   which is why a 150-build run needed the raised limit and why this backend is not usable at GP scale.
3. **C2 is partial by construction**: SPEC's `--max-wall-s 1800 --min-gens 20` cap, reached with 89/100 gens,
   exit 0, `partial: true` + `projection` — kept as-is (a valid SPEC result), never killed by `timeout`.
4. **Batch C was queued behind batch B** at 21:25 (the shared `flock` is held by whichever side runs), so
   `C5b.log`/`C7.log` *headers* (`config … started`) predate the real run by ~24 min: C5b's header says
   21:25:01 while its measured region ran 21:49:20→21:49:34. The lock wait is outside every timed region
   (SPEC), and `wall_s`/`generation_loop_s` are unaffected — only the header timestamps need this caveat.
5. **C5 is the Phase-1 artifact** and its JSON still records the Phase-1 worktree paths
   (`cwd=/…/worker_T1_A6`, `harness=/…/worker_T1_A6/test/benchmark/bench_gp.py`) and `evogp_commit=7e6cd0f`.
   `7e6cd0f` only carried the empty placeholder files (`bench_gp.py`/`check_smoke.py` content landed in
   9159e10), so the harness revision is the one documented in §1-§4 — checked by re-running
   `check_smoke.py` on that file (passes, including its 25-call/§3 sanity checks).
6. **Launcher precedent for anyone re-running this:** every batch was started as a detached
   `systemd-run --user --unit=gpbench_<X> -p WorkingDirectory=<root> -p CPUQuota=800% -p MemoryMax=12G
   -p LimitNOFILE=65536 …` unit, because a plain `nohup`/background job is killed when the agent's shell exits.
   The `CPUQuota=800%`/`MemoryMax=12G`/`LimitNOFILE=65536` values mirror the standard harness environment so
   results stay comparable with Phase-1 and with runs executed directly (item 2 is exactly what happens without
   them). One aborted attempt of batch A (21:10) left only a truncated `C1.json` snapshot before the process
   was killed; it was deleted and C1 re-ran clean — no result was lost and no config needed a second run.

## R2 — GPU round: harness extensions + stage probe

Additive to §1-§9 (the Phase-1 numbers above stay the baseline, nothing in them changed). This section
covers the **harness** side of round 2 — the new flags and the `--stage-probe` engine decomposition,
plus the calibration datapoints collected for the R2 report. The round-2 runner/validator/manifest
(`run_all.sh`, `check_r2.py`, `manifest.txt`, usage, exit codes) are documented in `../r2/README.md`;
the probe's own module is `../stage_probe.py`. Artifacts of this section:
`results/r2/smoke/h2_*.{json,log}` (probe smoke + probe-off control); H1's artifacts are `h1_*` there.
**R2 results:** `../../REPORT.md` (analysis), `../r2/RUNLOG.md` (operator log), raw records
`results/r2/*.json` + `*.log` (pre-flight: `results/r2/backend_agreement/`) — not repeated here.

### R2 flags — every one opt-in; with none passed the code path and JSON are the pre-R2 ones

| flag | default | effect |
|---|---|---|
| `--backend {default,numpy,iree}` | `default` | `default` sets nothing; `numpy`/`iree` export `EVOGP_ETL_BACKEND` (plus the two IREE settings) in `apply_env`, i.e. **before** `import evogp`; inert on the old side |
| `--iree-device I` | `0` | `EVOGP_ETL_IREE_DEVICE` |
| `--iree-opt-level {O0,O1}` | `O1` | `EVOGP_ETL_IREE_OPT_LEVEL` (the engine only accepts O0/O1) |
| `--data-rows N` | fixture rows | `eval64` only: tile the fixture `x`/`labels` up to N rows (N == native rows is a byte-identical no-op, so C5 stays reproducible); warns and ignores for other tasks |
| `--mem-guard-mb M` | `8192` | before any heavy step compares M with the free device bytes from `torch.cuda.mem_get_info()` |
| `--max-cold-builds N` | `300` | stop cleanly when cumulative `etl_engine.cache_stats()['misses']` exceeds N — **new side + `--backend iree` only** (a numpy run compiles no GPU program per structure; the legacy side has no `etl_engine`, so both are exempt) |
| `--allow-cold-builds` | off | disables that guard |
| `--stage-probe` | off | new-side stage decomposition of every `evaluate` call / generation (below); the old side warns and no-ops, leaving `stages: null` |

Exit codes: `0` complete · `3` skipped/stopped by a guard (JSON keeps the completed steps, sets
`partial`+`skipped`+`skip_reason`, `complete: true`) · `130` interrupted (`KeyboardInterrupt`) ·
`1` crash (traceback deliberately uncaught). The R2 batch runner maps these to `OK`/`SKIP`/`TIMEOUT`
/`INTERRUPT`/`FAIL` (`../r2/README.md`).

### `--stage-probe` — where the ETL engine's time goes

`test/benchmark/stage_probe.py` wraps the new engine's hot call sites **at runtime** — `decode_population`,
`_run_tree`, `build_cached`, `etl.run`, `etl.from_numpy`, `evaluate`, `sr_fitness` — so `src/` is never
edited and nothing is imported from the library (monkeypatching only). Wrappers are pure pass-throughs
(identical return value and exception behaviour); one single-threaded accumulator of raw
`time.perf_counter()` floats. `bench_gp.py` hooks are ~5 lines per call site: `install_stage_probe()`
before the rep loop (new side only), `begin_rep()` per rep, `begin()` before the timed region,
`snapshot(i, outer_ms)` inside it, `finish_rep()` + one summary line after the rep's `wall_s`; the
assembled block lands in the new top-level `stages` key (`null` when the flag is off, so prior/C5
JSONs and probed JSONs stay directly comparable). Missing wrapped attribute ⇒ that stage stays 0,
`stages.partial: true` and `note` names it — never a crash. Self-test (no torch/GPU/etl):
`python3 test/benchmark/stage_probe.py`.

Stage chain per record: `total_ms` (harness outer = `per_call_ms`/`per_gen_ms`) splits into
`decode_ms` + `sr_fitness_ms` + `misc_ms`; `sr_fitness_ms` splits into `evaluate_ms` + `loss_ms`
(label upload + error reduction); `evaluate_ms` into `build_hit_ms`+`build_miss_ms` + `run_ms` +
`readback_ms` (per-tree residual `tree_total − build − run`, i.e. the IREE D2H readback + host glue)
+ `upload_ms` (final result H2D). Torch-level copies outside the engine (e.g. the label tensor in
`sr_fitness`) are **not** in the H2D/D2H counters — they land in `loss_ms`/`upload_ms`. Counts:
`decode_calls`, `sr_fitness_calls`, `evaluate_calls`, `build_calls`/`build_hits`/`build_misses`,
`run_calls`, `readback_calls`, `trees_total`/`trees_invoked` (constant/collapsed trees are skipped by
the engine), `h2d_calls` (`etl.from_numpy` inside `_run_tree`, 1 per call on a device backend),
`d2h_calls` (= `readback_calls`, 0 on numpy).

```json
"stages": {                                  // null when --stage-probe is off
  "probe": "stage_probe/1", "side": "new", "partial": false, "note": "",
  "status": {"installed": true, "wrapped": ["decode_population", "..."], "missing": [],
             "engine_module": "evogp.tree.etl_engine", "etl_module": "etl", "nested_runs": 0},
  "reps": [{
    "rep": 0, "kind": "eval",                // "eval" (per call) | "train" (per generation)
    "records": [{ "i": 0,                    // call index / generation index
        "total_ms": 220.656, "probe_span_ms": 220.684, "decode_ms": 1.966, "decode_calls": 1,
        "sr_fitness_ms": 207.802, "sr_fitness_calls": 1, "evaluate_ms": 170.456,
        "evaluate_calls": 1, "loss_ms": 37.346, "misc_ms": 10.888,
        "build_ms": 155.329, "build_calls": 53, "build_hit_ms": 0.739, "build_hits": 5,
        "build_miss_ms": 154.59, "build_misses": 48, "run_ms": 13.874, "run_calls": 53,
        "readback_ms": 0.806, "readback_calls": 53, "upload_ms": 0.447,
        "trees_total": 64, "trees_invoked": 53, "h2d_calls": 0, "d2h_calls": 0 }],
    "totals": { "calls": 3, "cold_builds": 48, "cold_build_ms": 154.59,
        "total_build_misses": 48, "total_build_miss_ms": 154.59,
        "warm_misses": 0, "warm_build_miss_ms": 0.0,
        "total_ms": 266.416, "probe_span_ms": 266.496, "decode_ms": 5.554,
        "sr_fitness_ms": 249.737, "evaluate_ms": 211.937, "loss_ms": 37.8, "misc_ms": 11.125,
        "build_ms": 176.037, "build_hit_ms": 21.447, "build_miss_ms": 154.59,
        "run_ms": 32.575, "readback_ms": 2.091, "upload_ms": 1.234,
        "decode_calls": 3, "sr_fitness_calls": 3, "evaluate_calls": 3, "build_calls": 159,
        "build_hits": 111, "build_misses": 48, "run_calls": 159, "readback_calls": 159,
        "trees_total": 192, "trees_invoked": 159, "h2d_calls": 0, "d2h_calls": 0 }}]}
```

**Cold/warm separation** lives in `totals`: `cold_builds`/`cold_build_ms` = the misses (and their time)
of the **first record** of the rep (the fresh-process compile tax), `total_build_misses`/
`total_build_miss_ms` = the whole rep, and `warm_misses`/`warm_build_miss_ms` = misses **after** the
first record — `warm_misses == 0` is the machine-checkable proof that a warm pass compiled nothing
(and matches `payload.cold_builds.misses_last`, the guard's own `cache_stats()` read).

### Smoke table (all: repo root, `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=6`, `ulimit -n 65536`, one
`flock -w 7200 /tmp/gpbench_critical.lock` per process, exit 0)

| tag | command (beyond `bench_gp.py`) | key numbers | fds start→after cold→end |
|---|---|---|---|
| `h2_np_eval64` | `--task eval64 --eval-reps 3 --stage-probe` | calls 220.66/23.20/22.56 ms; cold 48 misses 154.59 ms, warm misses 0; 53 invocations/call (11 of 64 trees are constant/collapsed) | 39→59→59 |
| `h2_np_eval64_noprobe` | same, no flag (A/B control) | 217.36/22.43/20.65 ms | 39→59→59 |
| `h2_iree_fresh2` | `--task evalfresh --pop 2 --tree-len 32 --eval-reps 3 --backend iree --stage-probe --max-cold-builds 40` | 2935.43/8.05/6.87 ms; cold 2 misses 2867.08 ms (≈1.43 s/build), warm misses 0 | 39→155→219 |
| `h2_fdscan_x` | `--task evalfresh --pop 4 --tree-len 32 --eval-reps 4 --backend iree …` | 3626.95/12.25/43.69/10.49 ms; 4 builds 3550.96 ms (≈888 ms/build), 16 runs | 39→219→315 |
| `h2_fdscan_y` | `--task evalfresh --pop 16 --tree-len 32 --eval-reps 4 --backend iree …` | 26026.95/29.54/25.42/24.51 ms; 15 builds 25929.95 ms (≈1729 ms/build), 60 runs | 39→571→667 |
| `h2_fdscan_z` | same as Y with `--eval-reps 1` | 4367.60 ms; 15 builds 4281.71 ms (≈285 ms/build), 15 runs | 39→571→571 |
| `h2_sr_a_tiny` | `--task sr --pop 8 --gens 2 --tree-len 32 --reps 2 --backend iree …` | rep0 4048.4/2907.4 ms (14 misses, 6397.5 ms); rep1 28.0/20.2 ms, warm_misses 0; identical best fitness | 39→347→635 |
| `h2_np_sr_tiny` | `--task sr --pop 8 --gens 2 --tree-len 32 --reps 2 --stage-probe` | rep0 503.7/45.6 ms (14 misses, 68.8 ms); rep1 13.4/12.9 ms, warm_misses 0 | 39→59→59 |

### `h2_np_eval64` — the full stage split (default numpy backend, fixture forest 64 trees / 63 nodes)

| stage | call 0 (cold) | call 1 | call 2 | rep totals |
|---|---|---|---|---|
| `total_ms` | 220.656 | 23.203 | 22.557 | 266.416 |
| `decode_ms` | 1.966 | 1.904 | 1.684 | 5.554 |
| `build_miss_ms` (misses) | 154.590 (48) | 0 (0) | 0 (0) | 154.590 (48) |
| `build_hit_ms` (hits) | 0.739 (5) | 9.994 (53) | 10.714 (53) | 21.447 (111) |
| `run_ms` (calls) | 13.874 (53) | 9.912 (53) | 8.789 (53) | 32.575 (159) |
| `readback_ms` | 0.806 | 0.648 | 0.637 | 2.091 |
| `upload_ms` | 0.447 | 0.358 | 0.429 | 1.234 |
| `loss_ms` | 37.346 | 0.254 | 0.200 | 37.800 |
| `misc_ms` | 10.888 | 0.133 | 0.104 | 11.125 |
| `cold_builds` / `warm_misses` | — | — | — | 48 / **0** |

* On this backend the two warm `build_hit`+`run` stages are ~48% / ~40% of the 22.6 ms call
  (10.0/9.9 and 10.7/8.8) — the "warm" cost is the per-tree memoized `build_cached` lookup + trace
  cache and the numpy interpreter, not GPU work (`h2d`/`d2h` = 0).
* Compile tax: 154.59 ms / 48 misses = **3.2 ms per fresh structure** (numpy), versus 0.19 ms per
  cache hit; `loss_ms` is 37.35 ms only on call 0 (label-tensor H2D), 0.2-0.25 ms warm.

### `h2_iree_fresh2` — IREE: the compile tax is the whole story

`pop 2 / tree-len 32`, warm call ≈ 7 ms, cold call 2935.4 ms = 0.325 `decode` + **2867.084
`build_miss` (2 misses, ≈1.43 s each)** + 21.315 `run` (2) + 7.729 `readback` + 15.691 `loss` +
22.877 `misc` (`h2d` 1, `d2h` 2). Warm 8.051/6.867 ms = `decode` 0.555/0.398 + `build_hit`
1.541/1.322 (2) + `run` 2.254/1.929 + `readback` 2.900/2.562 + `loss` + `misc`, **0 misses**,
`h2d` 1 / `d2h` 2 per call. Rep totals: cold 2 = 2867.084 ms, warm_misses 0, `h2d` 3, `d2h` 6.

### Cold/warm separation in training (`h2_sr_a_tiny`, `sr` pop 8 / 2 gens, same seed both reps)

* rep0: gen0 4048.4 ms (`misc` 493.0 ms = crossover/mutation/selection + NaN replacement, 8 misses
  3523.98 ms ≈ 440 ms/build), gen1 2907.4 ms (1 hit + **6 misses** 2873.52 ms ≈ 479 ms/build —
  survivors already hit, the children are new structures); rep totals 6955.79 ms, 14 misses
  6397.50 ms, `h2d` 2 / `d2h` 15, `trees_total` 16 vs `trees_invoked` 15.
* rep1 (identical seed, same process ⇒ warm cache): gen0 28.0 ms, gen1 20.2 ms, rep total 48.26 ms,
  `cold_builds` 0, `total_build_misses` 0, **`warm_misses` 0**, 15/15 hits — i.e. **145× faster than
  rep0 with `best_fitness_final = -2.0357980728149414` on both reps** (the trajectory is unchanged).
* numpy counterpart `h2_np_sr_tiny`: rep0 549.3 ms (8 cold misses 36.97 ms ≈ 4.6 ms/build) → rep1
  26.3 ms, same fitness — the same separation at a ~200× smaller compile tax.

### FD-rate calibration (`h2_fdscan_{x,y,z}`, IREE, raw `payload.fds`)

| run | pop / eval-reps | cold builds | run calls | fds start → after cold → end |
|---|---|---|---|---|
| X | 4 / 4 | 4 | 16 | 39 → 219 → **315** |
| Y | 16 / 4 | 15 | 60 | 39 → 571 → **667** |
| Z | 16 / 1 | 15 | 15 | 39 → 571 → **571** |

* Design rates (the two pairs the scan was built to difference): **`per_build_fd ≈ (Z−X)/12 =
  256/12 = 21.3`** (X and Z differ by 12 builds at ~equal run counts) and **`per_run_fd ≈ (Y−Z)/48 =
  96/48 = 2.0`** (Y and Z have the same 15 builds, 48 run calls apart). Raw endpoints above.
* Caveats, stated because these are single noisy samples on a shared box: the probe's own counts put
  the true differences at 11 builds / 45 runs (one of the 16-tree populations has a constant tree, so
  15 invocations per call), which gives 23.3 FD/build and 2.1 FD/run; and the wider 5-point set
  (adding `h2_iree_fresh2` 2 builds/6 runs → 180 FD and `h2_sr_a_tiny` 14 builds/15 runs → 596 FD)
  does **not** fit one linear model (sr_a_tiny grows 64 FD more than Z with one build *fewer*). Read
  the rates as order-of-magnitude: **≈2 FD per run, order 10-20 FD per cold build** (round 1's C7
  failure at a 1024 soft limit implies ≈7 FD/build, so even the per-build coefficient is variable).
* Per-build **time** is even less of a constant: X/Y/Z took 888 / 1729 / 285 ms per build although
  Y and Z compile the *same* 15 structures from the same seed (25.93 s vs 4.28 s) — build rate is
  order/context-dependent (~6× spread, plausibly also box contention), so only quote `cold_build_ms`
  for a given config, never a global "s/build".

### What the matrix runners must know

* **Probe overhead (A/B, same config back-to-back):** probe-on warm 22.880 ms vs probe-off 21.537 ms
  (medians) ⇒ **≈+1.3 ms per warm call (≈5-6%)**; first call 220.66 vs 217.36 ms (+1.5%). Single
  A/B pair on a ±10% shared box — treat it as "a few ms per call". Only probed configs pay it (the
  runner's `*_noprobe` controls measure the un-instrumented side), and instrumenting never changes
  the numbers reported outside `stages`.
* **`--max-cold-builds N` counts cache misses, not FDs, and it is sampled at checkpoints** (startup,
  after the first call/gen, after every generation in `sr`) — so a limit is a bound on *cumulative
  misses*, not on peak FDs, and within one generation it can overshoot by that generation's new
  structures. Budget FDs separately: cap 300 ≈ **6.3k FDs** at 21 FD/build (plus ≈2 FD/run), so a
  legitimately-sized 3200-cap run can approach the 65536 soft limit — keep `ulimit -n 65536` first in
  any launcher, prefer the smallest cap that still finishes the config, and watch `payload.fds.end`
  (recorded in every JSON) as the real FD telemetry. IREE's per-run leak is real: every run that
  touched the backend ended with `nanobind: leaked …` instances for its compiled programs.
* Warm passes are free of compiles by construction (`warm_misses == 0`), which is what makes a
  `--reps 2` `sr` config the cheapest way to measure the warm per-generation cost of IREE.
* Do not read `cold_build_ms`/`cold_builds` from `payload.cold_builds` (guard telemetry: sampled at
  checkpoints, `limit`/`exceeded` only, `null` when the guard is exempt) — the probe's
  `stages.reps[].totals` is the authoritative cold/warm split.
* The GPU/box conventions of §9 still apply (never GPU 7, one `flock`, `timeout` inside the lock,
  medians ±10%); all eight `h2_*` runs here kept them.
