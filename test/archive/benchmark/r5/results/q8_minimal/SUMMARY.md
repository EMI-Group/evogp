# `results/q8_minimal/` — R5 Q8: the harness-artifact falsification control

Phase-2 leftover of the R5 RCA round, executed 2026-09-13 20:58 HKT on GPU 6 (shared box).
Analysis-only: no `src/evogp` edit, no write in `/mnt/local-ssd/bchuang/etl`, no edit outside `r5/`.
The fuller analysis is the "Update — Q8 + cProfile completed" section of
`../q1_decomp/SUMMARY.md`; this file is the record-level write-up of the two cells.

## What these records are

`minimal_loop.py` (committed with the Q1 phase) is the control that removes every possible
harness/probe artifact: it loads the fixture, builds the exact same `Forest` +
`SymbolicRegression(datapoints=x, labels=labels)` pair `bench_gp.py:768-806` builds, then calls
`problem.evaluate(forest)` in a plain loop, each call bracketed by one `torch.cuda.synchronize()`
pair -- the harness' own timed region shape -- with **no `bench_gp.py` import, no probe import, no
monkeypatch, no JSON plumbing**. 25 calls, call 0 cold.

Route/environment (both cells, stamped in each JSON): repo-root cwd, project venv
`/mnt/local-ssd/bchuang/evox/.venv/bin/python`, `PYTHONPATH=src`, `ETL_DISABLE_LOCATIONS=1`,
`CUDA_VISIBLE_DEVICES=6`, `EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=auto EVOGP_ETL_IREE_DEVICE=0
EVOGP_ETL_IREE_OPT_LEVEL=O1`; ETL = editable install of `/mnt/local-ssd/bchuang/etl` @
`96df7da8f2ba39f52cf2f2b61888e67f70471bc9` (clean), evogp `48dd1bc` (r5 files only). One process per
cell, each inside exactly one `flock -w 7200 /tmp/gpbench_critical.lock` with
`timeout -s INT -k 30 900` inside the lock, `ulimit -n 65536`.

## Results

| cell | fixture | pop | rows | warm median | min / max | first call |
|---|---|---|---|---|---|---|
| `minimal_f64` | `test/goldens/forest_random.npz` | 64 | 64 | **6.117 ms** | 5.651 / 9.310 | 12411.901 ms (cold compile) |
| `minimal_f200` | `test/benchmark/fixtures/forest200_tree64_random.npz` | 200 | 64 | **6.010 ms** | 5.410 / 11.848 | 12422.796 ms (cold compile) |

Reference bands for the same hour: the Q4 same-window dense-iree end-to-end cells are 5.3823 ms
(f64) / 5.7912 ms (f200) with the window's post-window control at +26.9 % (6.83 ms f64), and this
phase's own probe-process totals are 7.3378 / 6.6167 ms.

## Verdict

**The harness-artifact hypothesis is falsified.** With every harness and probe layer removed the
route still costs ~6 ms warm, inside the Q4 band and *below* the probe-process totals. The ~4 ms of
non-pure-forward cost per call therefore lives inside `problem.evaluate(forest)` itself (payload
uploads + decode + host assembly + the difference between the frozen pure runner's staged buffers
and the e2e path's per-call buffers), not in the measurement scaffolding. The tails (up to 9.3 /
11.8 ms) are the shared box's co-tenant drift, the same shape the Q4 window measured.

## Files

| file | what |
|---|---|
| `minimal_f64.json` / `.log` | 25 per-call ms, warm median/min/max, fixture shape, `etl_file`, full stamp |
| `minimal_f200.json` / `.log` | same for the 200-tree fixture |

Per-call arrays live in the JSONs (`per_call_ms`); the logs carry the same numbers in one line.
