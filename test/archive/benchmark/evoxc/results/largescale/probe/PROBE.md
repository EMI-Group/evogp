# Large-scale compile-feasibility probe (records + reading)

Records only in this directory: one JSON per `(cell, arm)` plus the driver's `campaign.json` and the
worker log next to it. `campaign.json` is rewritten by every driver invocation, so each probe's copy is
kept verbatim as `campaign-probeA.json` … `campaign-probeE-candidates.json` (probe A = population axis,
B = node-count/rows axes, C = dense+torch at the extremes, D = the raised-cap dense shot, E = the
remaining candidate cells).

Commands of record (repository root, one driver invocation per probe, sequential, `EVOXC_DISABLE_LOCATIONS=1`):

```
PY=/tmp/rocvenv/bin/python
$PY test/benchmark/evoxc/campaign.py --tree-root src --evoxc-root /root/evoxc \
  --out test/benchmark/evoxc/results/largescale/probe --resume --timeout 900 \
  --note "large-scale compile-feasibility probe" \
  --cells <probe cells> --arms <probe arms> --workloads <probe workloads> \
  --warmup 1 --reps 3 --gp-reps <1|3>
```

A/B/D/E: `--arms evoxc-batch-iree --workloads sr_fitness --gp-reps 1`.
C: `--arms evoxc-dense-iree,torch-per-tree,torch-batched --workloads sr_fitness,batch_forward,gp_step --gp-reps 3`.
Total GPU wall time ≈ 31 min, one config at a time, idle-device guard `ok` on every record.

## Batch route (`EVOGP_EVAL_MODE=batch`, shipped defaults: band size 16, pool 4, chunking on)

| cell | status | cold `first_call_ms` | warm `sr_fitness` median / min (ms) | deduped fns | emitted ops | chunks | band clamp |
|---|---|---|---|---|---|---|---|
| p256-L64-i32-N4096 | ok | 4 242.98 | 5.2348 / 5.1818 | 232 | 5 497 | 4 × 58 | no |
| p256-L64-i32-N16384 | ok | 4 923.73 | 12.1887 / 12.0603 | 232 | 5 497 | 4 × 58 | no |
| p1000-L64-i32-N4096 | ok | 19 095.71 | 13.1965 / 13.0215 | 841 | 21 339 | 4 × 211 | no |
| p1000-L64-i32-N16384 | ok | 19 641.68 | 37.5065 / 37.3110 | 841 | 21 339 | 4 × 211 | no |
| p2048-L64-i32-N1024 | ok | 48 858.24 | 13.0982 / 12.9852 | 1 680 | 41 376 | 4 × 420 | yes (27 → 15/chunk) |
| p4096-L64-i32-N1024 | ok | 190 625.19 | 23.2721 / 23.1558 | 3 265 | 78 189 | 4 × 817 | yes |
| p4096-L64-i64-N1024 | ok | 206 996.84 | 24.0490 / 24.0472 | 3 297 | 80 968 | 4 × 825 | yes |
| p2048-L128-i32-N1024-d7 | ok | 249 960.54 | 25.3059 / 25.2905 | 1 664 | 93 751 | 4 × 416 | yes |
| p8192-L64-i32-N1024 | **timeout** | >900 000 (not reached) | not measured | n/a | n/a | n/a | n/a |

`p8192-L64-i32-N1024` never produced a record: the driver's 900 s cap fired while all four `iree-compile`
chunk workers were still compiling (worker age 806 s, compiles 776 s, no chunk finished). Population
8 192 dedups to 6 544 structures, i.e. ≈2× the p4096 plan ≈ 156–190k emitted ops.

Cold compile is quadratic in the plan's EMITTED OPS (equivalently in the population's total node count),
on the chunked pool-4 path: 41 376 ops → 48.9 s; 78 189 → 190.6 s; 80 968 → 207.0 s; 93 751 → 250.0 s;
5 497 → ~4.3 s. A least-squares-free fit `cold ≈ 48.9 s × (ops / 41 376)²` reproduces every point within
~10 % and predicts ≈700–900 s for the p8192 plan — which is what the timeout observed.
Rows are nearly free (`p1000-L64`: N=1024 → N=4096 → N=16384 = 19.1 / 19.6 s compile, and the p256 cells
are 4.2 / 4.9 s): the merge is rows-insensitive, population/node-count is the axis that bites.

## Dense route at the extremes (`EVOGP_EVAL_MODE=dense`, shipped cap 64)

| cell | workload | cold `first_call_ms` | median / min (ms) |
|---|---|---|---|
| p8192-L64-i32-N1024 | sr_fitness | 6 773.21 | 75.8860 / 75.7453 |
| p8192-L64-i32-N1024 | batch_forward | 6 639.24 | 94.5026 / 94.2103 |
| p8192-L64-i32-N1024 | gp_step | (in eval) | 111.6974 total (eval 108.9143, ops 2.7831) |
| p1000-L64-i32-N16384 | sr_fitness | 6 614.86 | 117.9119 / 117.7922 |
| p1000-L64-i32-N16384 | batch_forward | 6 520.47 | 149.6110 / 148.2989 |
| p1000-L64-i32-N16384 | gp_step | (in eval) | 128.2814 total (eval 125.8068, ops 2.4746) |
| p2048-L256-i32-N1024-d8 (cap pinned 256) | sr_fitness | 42 548.09 | 239.9341 / 239.7667 |

Dense compile is ~pop- and rows-insensitive (6.5–6.8 s per shape at both extremes), but its warm cost is
NOT: 2.06 ms at `p1000-L64-i32-N1024` (round-3 record) → 75.9 ms at pop 8192 (8× pop for 37× the time)
and 117.9 ms at N=16384. Both torch baselines beat dense on these cells — see the next table.

## Plain-torch baselines at the extremes

| cell | arm | sr_fitness median / min (ms) | batch_forward median (ms) | gp_step total (ms) |
|---|---|---|---|---|
| p8192-L64-i32-N1024 | torch-per-tree | 994.4037 / 994.2478 | 765.6915 | 469.8060 |
| p8192-L64-i32-N1024 | torch-batched | 17.6849 / 17.6528 | 17.6101 | 601.1526 |
| p1000-L64-i32-N16384 | torch-per-tree | 122.0989 / 121.3488 | 94.7768 | 51.4546 |
| p1000-L64-i32-N16384 | torch-batched | 32.3456 / 32.3103 | 32.4024 | 74.5493 |

`torch-per-tree` at pop 8192 is ~1.0 s per `sr_fitness` call — a rep costs 1 s, a config stays ~20 s
(setup 50 ms, 1 warmup + 3 reps), so it is affordable, just slow. `torch-batched` is the cheapest
per-call arm on all three of these large cells and beats dense by 4.3× (pop 8192) / 3.6× (N=16384).
