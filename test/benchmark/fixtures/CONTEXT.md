# `test/benchmark/fixtures/` — frozen cross-side evaluation fixture data

## Intent
Data node: one committed npz holding a fixed 200-tree forest in the legacy-goldens schema, so the OLD (legacy CUDA engine at `b613503`) and NEW (ETL "true jit") engines can evaluate byte-identical inputs with no generator difference.
The benchmark rounds spell it `fixture200`; `fixture64` is NOT here — it is `test/goldens/forest_random.npz` (see "fixture64 vs fixture200" below).
Nothing imports this directory: `../bench_gp.py` loads the npz through `--fixture` and the runners in `../r2/`, `../r4/`, `../r5/` point at it.

## Contents
- `forest200_tree64_random.npz` — 104 474 B, sha256 `6789ef61ecf8898b9b2a00774ee655e20b0b5be7c72eb861afc8d6b8799741b9`, array digest (sorted keys, canonical bytes) `872e793f87eb6c775458783805376a954c3f2ab0b554496fcfcc1d19235edbdb`; 5 uncompressed zip members, fixed 1980 timestamps.
- `README.md` — provenance, schema table, regeneration command, constraints.
- `sanity_eval.json` / `sanity_eval.log` — NEW-side load-and-run proof (`bench_gp.py --task eval64 --fixture <this file> --eval-reps 2`): first call 1166.6 ms (200 cold `etl.build_cached` builds, numpy backend), second call 111.0 ms warm; `median_warm_ms=null` because 2 reps leave only one warm call.

## Fixture schema (`forest200_tree64_random.npz`)
| key | shape | dtype | meaning |
|---|---|---|---|
| `forest__node_value` | (200, 64) | float32 | node values (constants / function ids), `Forest` ctor arg 1 |
| `forest__node_type` | (200, 64) | int16 | `NType` base kind + optional `OUT_NODE` bit, ctor arg 2 |
| `forest__subtree_size` | (200, 64) | int16 | column 0 = tree length, ctor arg 3 |
| `x` | (64, 2) | float32 | datapoints, bitwise from `test/goldens/forest_random.npz` |
| `labels` | (64, 1) | float32 | targets, bitwise from `test/goldens/forest_random.npz` |

Measured structure: pop 200, width 64; tree lengths 1…31 (median 17, 15 distinct lengths, all odd); 165 distinct `node_value` rows; used nodes var 764 / const 828 / bfunc 1392 (no ufunc/tfunc, no `OUT_NODE` bit); max tree depth 4 edges (layers 0…4; 45/8/2/1/144 trees at depths 0/1/2/3/4 by the pre-order subtree-size walk); dead tail zero in all three arrays; node values finite inside trees; no NaN/inf anywhere.
Targets: `x` is the 8×8 grid `x0 = ((i % 8) - 3.5)/2`, `x1 = (floor(i/8) - 3.5)/2` over i=0..63 (values {-1.75, -1.25, …, 1.75}) and `labels = x0² − x1` (min −1.6875, max 4.8125, mean 1.3125, 14 unique) — the exact `../golden_capture.py::fixed_inputs` formula, no RNG.
No json sidecar and no descriptor/scalar metadata arrays: unlike `test/goldens/forest_random.npz`, only these 5 arrays are stored (`seed`/`pop_size`/`using_funcs`/`descriptor__*` exist in the golden file but are ignored by `bench_gp.py`).
Raw arrays are CPU numpy; the eval harness converts each to a contiguous cuda tensor (`torch.from_numpy(np.array(a, dtype=…, copy=True)).to("cuda")`) before building the `Forest`; the `Forest` ctor is device-agnostic (shape asserts only) and the engine path expects cuda.

## Construction recipe (from `../make_r2_fixtures.py`)
- `torch.manual_seed(0)` + `torch.cuda.manual_seed(0)`, then `Forest.random_generate(pop_size=200, descriptor=GenerateDescriptor(max_tree_len=64, input_len=2, output_len=1, const_prob=0.5, out_prob=0.5, using_funcs=['+','-','*','/'], layer_leaf_prob=0.2, const_range=(-5,5), sample_cnt=8, max_layer_cnt=5))` — the SPEC §W1 descriptor replicated from `../bench_gp.py::w1_descriptor`.
- `x`/`labels` loaded from `test/goldens/forest_random.npz` (`--data-from` default), asserted `(rows,2)`/`(rows,1)` with equal row counts and copied verbatim as contiguous float32.
- Written with `np.savez` (uncompressed, fixed 1980 timestamps) ⇒ byte-for-byte reproducible for the same seed/pop/tree-len; regeneration and `--verify` need a CUDA device (the descriptor builds its tables on `"cuda"`).
- Exact command: `PYTHONPATH=src <venv>/bin/python test/benchmark/make_r2_fixtures.py --pop 200 --tree-len 64 --seed 0 --out test/benchmark/fixtures/forest200_tree64_random.npz --verify`.

## fixture64 vs fixture200 (how the rounds select them)
- `fixture64` = `test/goldens/forest_random.npz` (64 trees, width 63, max depth 3, legacy descriptor with `max_layer_cnt=4` and `const_samples=[-2,-1,-0.5,-0.25,0.25,0.5,1,2]`) — static data, never generated on the fly.
- `fixture200` = this directory's npz (200 trees, width 64, max depth 4, `max_layer_cnt=5`).
- `../r4/run_e2e.sh`: fixture64 cells pass NO `--fixture` (harness default `test/goldens/forest_random.npz`, cwd-relative); fixture200 cells pass `--fixture test/benchmark/fixtures/forest200_tree64_random.npz`; both use `--task eval64 --eval-reps 25 --mem-guard-mb 8192 --stage-probe`, env `ETL_DISABLE_LOCATIONS=1 EVOGP_ETL_BACKEND=… EVOGP_EVAL_MODE=…`, iree adds `--backend iree` and `--max-cold-builds 400` (fixture200 only).
- `../r4/run_pure.sh` (`pure64_*` / `pure200_*` ids), `../r2/run_all.sh`, `../r2/round3/*.sh` and `../r5/window_probe.sh` pass `--fixture` explicitly for both fixtures.
- The legacy checkout has no `test/goldens/`, so the OLD side always gets an explicit `--fixture <absolute path>` (records show a worktree-scoped `/…/worker_*/test/…` path).

## Loading identically in a standalone runner (minimal path)
```python
import numpy as np, torch
from evogp.tree import Forest
from evogp.problem import SymbolicRegression

data = np.load(fixture_path)                       # 'test/benchmark/fixtures/forest200_tree64_random.npz' (fixture200)
to_cuda = lambda a, dt: torch.from_numpy(np.array(a, dtype=dt, copy=True)).to("cuda")
x      = to_cuda(data["x"], np.float32)            # (64, 2)
labels = to_cuda(data["labels"], np.float32)       # (64, 1)
forest = Forest(input_len=int(x.shape[1]), output_len=int(labels.shape[1]),
                batch_node_value=to_cuda(data["forest__node_value"], np.float32),
                batch_node_type=to_cuda(data["forest__node_type"],  np.int16),
                batch_subtree_size=to_cuda(data["forest__subtree_size"], np.int16))
problem = SymbolicRegression(datapoints=x, labels=labels)
fitness = problem.evaluate(forest)                 # MSE (use_MSE=True default), negated/MSE fitness convention
```
- `--data-rows N` (eval64/eval-pure) tiles the fixture rows: `np.tile(a, (ceil(N/rows),)+(1,)*(ndim-1))[:N]` then `ascontiguousarray` — it repeats the fixed 64-row block, never changes it.
- Everything is RNG-free at measurement time: the npz fully determines `(forest, x, labels)`.

## Known Issues / gotchas
- Never re-run `../make_r2_fixtures.py` with its default `--out`: it overwrites this committed file; `--verify` regenerates first and then verifies the regenerated file, so it can never validate the committed bytes (compare sha256, or call `make_r2_fixtures.verify_fixture(path, 2, 1)` directly).
- Pin by sha256: the file hash above is what the `../r2/round3` and `../r4` protocols quote; regenerating after any library change would change the bytes even with `--seed 0` and invalidate the round tables.
- The GPU-free structural checks (`tree_report`) pass on the committed file (pop 200 / width 64 / lengths 1-17-31 / 165 distinct rows / var 764-const 828-bfunc 1392 / finite / dead-tail zeros) and both hashes match; the CUDA half of `--verify` (`verify_fixture`: rebuild `Forest` + one `SR_fitness`) has no post-commit re-run — its committed evidence is `README.md` (12/200 NaN/±inf, finite losses 2.72…650 003, mean 6 969) and `sanity_eval.json`.
- Paths inside committed result JSONs (`"fixture": …/.genesis/workers/worker_*/…`) are worktree-scoped and point at old worker directories; resolve fixture paths against the repository root.
- `sanity_eval.json`/`.log` are bench records, not fixture metadata, and no runner reads them.

## Routing Table / See Also
Data-only node: no sub-nodes, no code.
Siblings / consumers: `../CONTEXT.md` (benchmark rounds), `../README.md`, `../bench_gp.py` (`--fixture`, `--data-rows`, `build_eval_case`, `w1_descriptor`), `../make_r2_fixtures.py` (generator + self-checks), `../../goldens/` (the `fixture64` file and its provenance), `../r4/CONTEXT.md`, `../r2/CONTEXT.md`, `../r5/CONTEXT.md`.
