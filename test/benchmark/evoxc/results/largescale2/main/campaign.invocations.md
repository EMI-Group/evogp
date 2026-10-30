# `results/largescale2/main` — the two driver invocations that filled this directory

The 55 records here were written by TWO sequential `campaign.py` invocations in one window
(same box, one process per config, idle-device guard on, 900 s per-config cap).
`campaign.py` rewrites `campaign.json` per config, so a directory filled by more than one
invocation keeps only the LAST one's driver record; the earlier one is transcribed below verbatim
from its stdout. `campaign.scoped-sr_fitness.json` is the previous round's driver record
(`--workloads sr_fitness` only, 49 configs), kept as the prior-window provenance of the
`sr_fitness` column.

Common invocation shape (`PYTHONPATH=src:/root/evoxc/.genesis/workers/worker_T1_A1`,
`EVOXC_DISABLE_LOCATIONS=1`, `/tmp/rocvenv/bin/python`, `--warmup 2 --reps 5 --timeout 900`,
`--workloads sr_fitness,forward,batch_forward`, `--evoxc-root /root/evoxc/.genesis/workers/worker_T1_A1`).

## Invocation 1 — the 7 cells where every arm is safe (49 configs)

`--out results/largescale2/main --cells p2048-L64-i32-N1024,p4096-L64-i32-N1024,p256-L64-i32-N4096,p256-L64-i32-N16384,p1000-L64-i32-N4096,p1000-L64-i32-N16384,p4096-L64-i64-N1024 --arms torch-per-tree,torch-batched,evoxc-dense-iree,evoxc-dense-iree-raised,evoxc-auto,evoxc-batch-iree,evoxc-batch-iree-default --note "3-workload re-measure for the headline column; fresh window replication of the sr_fitness table"`

Driver summary line: `[campaign] 49 configs, 0 failures, 2330.6s`.
Every config reported `(gpu_idle=True)`; no timeout, no failure.

| # | cell | arm | wall s |
|---|---|---|---|
| 1 | p2048-L64-i32-N1024 | torch-per-tree | 14.8 |
| 2 | p2048-L64-i32-N1024 | torch-batched | 9.4 |
| 3 | p2048-L64-i32-N1024 | evoxc-dense-iree | 29.0 |
| 4 | p2048-L64-i32-N1024 | evoxc-dense-iree-raised | 27.6 |
| 5 | p2048-L64-i32-N1024 | evoxc-auto | 28.1 |
| 6 | p2048-L64-i32-N1024 | evoxc-batch-iree | 93.2 |
| 7 | p2048-L64-i32-N1024 | evoxc-batch-iree-default | 101.5 |
| 8 | p4096-L64-i32-N1024 | torch-per-tree | 19.8 |
| 9 | p4096-L64-i32-N1024 | torch-batched | 8.9 |
| 10 | p4096-L64-i32-N1024 | evoxc-dense-iree | 27.6 |
| 11 | p4096-L64-i32-N1024 | evoxc-dense-iree-raised | 27.9 |
| 12 | p4096-L64-i32-N1024 | evoxc-auto | 27.9 |
| 13 | p4096-L64-i32-N1024 | evoxc-batch-iree | 257.3 |
| 14 | p4096-L64-i32-N1024 | evoxc-batch-iree-default | 296.8 |
| 15 | p256-L64-i32-N4096 | torch-per-tree | 8.9 |
| 16 | p256-L64-i32-N4096 | torch-batched | 8.4 |
| 17 | p256-L64-i32-N4096 | evoxc-dense-iree | 26.7 |
| 18 | p256-L64-i32-N4096 | evoxc-dense-iree-raised | 26.3 |
| 19 | p256-L64-i32-N4096 | evoxc-auto | 26.3 |
| 20 | p256-L64-i32-N4096 | evoxc-batch-iree | 17.4 |
| 21 | p256-L64-i32-N4096 | evoxc-batch-iree-default | 17.4 |
| 22 | p256-L64-i32-N16384 | torch-per-tree | 9.1 |
| 23 | p256-L64-i32-N16384 | torch-batched | 8.5 |
| 24 | p256-L64-i32-N16384 | evoxc-dense-iree | 27.1 |
| 25 | p256-L64-i32-N16384 | evoxc-dense-iree-raised | 27.4 |
| 26 | p256-L64-i32-N16384 | evoxc-auto | 27.2 |
| 27 | p256-L64-i32-N16384 | evoxc-batch-iree | 19.6 |
| 28 | p256-L64-i32-N16384 | evoxc-batch-iree-default | 20.0 |
| 29 | p1000-L64-i32-N4096 | torch-per-tree | 11.6 |
| 30 | p1000-L64-i32-N4096 | torch-batched | 9.1 |
| 31 | p1000-L64-i32-N4096 | evoxc-dense-iree | 27.9 |
| 32 | p1000-L64-i32-N4096 | evoxc-dense-iree-raised | 27.5 |
| 33 | p1000-L64-i32-N4096 | evoxc-auto | 27.4 |
| 34 | p1000-L64-i32-N4096 | evoxc-batch-iree | 42.3 |
| 35 | p1000-L64-i32-N4096 | evoxc-batch-iree-default | 43.4 |
| 36 | p1000-L64-i32-N16384 | torch-per-tree | 11.4 |
| 37 | p1000-L64-i32-N16384 | torch-batched | 9.2 |
| 38 | p1000-L64-i32-N16384 | evoxc-dense-iree | 29.1 |
| 39 | p1000-L64-i32-N16384 | evoxc-dense-iree-raised | 29.0 |
| 40 | p1000-L64-i32-N16384 | evoxc-auto | 28.8 |
| 41 | p1000-L64-i32-N16384 | evoxc-batch-iree | 46.6 |
| 42 | p1000-L64-i32-N16384 | evoxc-batch-iree-default | 48.1 |
| 43 | p4096-L64-i64-N1024 | torch-per-tree | 19.4 |
| 44 | p4096-L64-i64-N1024 | torch-batched | 9.0 |
| 45 | p4096-L64-i64-N1024 | evoxc-dense-iree | 28.6 |
| 46 | p4096-L64-i64-N1024 | evoxc-dense-iree-raised | 28.8 |
| 47 | p4096-L64-i64-N1024 | evoxc-auto | 28.3 |
| 48 | p4096-L64-i64-N1024 | evoxc-batch-iree | 272.7 |
| 49 | p4096-L64-i64-N1024 | evoxc-batch-iree-default | 301.9 |

## Invocation 2 — the long-tree cell (6 configs, plain `evoxc-dense-iree` deliberately unscheduled)

`--out results/largescale2/main --cells p2048-L128-i32-N1024-d7 --arms torch-per-tree,torch-batched,evoxc-dense-iree-raised,evoxc-auto,evoxc-batch-iree,evoxc-batch-iree-default --note "d7 cell: plain evoxc-dense-iree stays unscheduled (cap declines into 1664 per-structure compiles, cgroup OOM)"`

This is the invocation that owns the `campaign.json` in this directory (6 configs, 0 failures, 794.3 s,
`scope_dropped` empty because the OOM arm is excluded by `--arms`, not by the harness's own scope rule).

| # | cell | arm | wall s | guard |
|---|---|---|---|---|
| 1 | p2048-L128-i32-N1024-d7 | torch-per-tree | 21.95 | idle, not contaminated |
| 2 | p2048-L128-i32-N1024-d7 | torch-batched | 9.31 | idle, not contaminated |
| 3 | p2048-L128-i32-N1024-d7 | evoxc-dense-iree-raised | 49.75 | idle, not contaminated |
| 4 | p2048-L128-i32-N1024-d7 | evoxc-auto | 49.36 | idle, not contaminated |
| 5 | p2048-L128-i32-N1024-d7 | evoxc-batch-iree | 277.2 | idle, not contaminated |
| 6 | p2048-L128-i32-N1024-d7 | evoxc-batch-iree-default | 385.49 | idle, not contaminated |
