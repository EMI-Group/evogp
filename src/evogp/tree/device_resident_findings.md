# Device-resident dense hot path — findings

Measurement + evidence record for the **device-resident** work on
`dense_engine.py`: the SR loss is folded into the compiled dense graph and the
per-call host round trips of the dense route are counted and reduced.

The reproducible battery is `device_resident_check.py` (in this directory; a dev
CLI, NOT pytest-collected).  Every number below was produced from the repository
root with the project venv and

```bash
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \
  <venv>/bin/python src/evogp/tree/device_resident_check.py
```

## 1. Round 1 — the pre-refactor capture (FAILING)

Command (same as above, plus `--json /tmp/device_resident_before.json`):

```
== FAILED: 39/60 checks passed ==
```

The red checks are exactly the round trips this work removes:

| check | pre-refactor observation |
| --- | --- |
| `inputs converted to host exactly ONCE` | **2** conversions (`inputs` + `x`: `sr_fitness_dense` converts the inputs, then `evaluate_dense` converts them again) |
| `every readback is (P,)-scale` (unchunked) | readback `[8, 37]` — the whole prediction, 296 elements for a population of 8 |
| `no host->torch tensor larger than pop_size` | `torch.from_numpy` on `[8, 37, 1]` (296 elements) |
| `no labels round trip through a torch tensor` | `torch.from_numpy` on `[37, 1]` after a D2H, re-uploaded with `.to(device)` |
| `x prepared ONCE per n-chunk` (both-chunked) | **296** preparations for 8x37 chunks (= one per `(p0, n0)` chunk; `_prep_columns` is memoized per `p0`, the x slice is not) |
| `labels prepared ONCE per n-chunk` | **0** — `labels` never becomes a graph payload at all (torch-side reduction) |
| `resolve_eval_mode('dynamic') -> 'dense'` (+5 more policy checks) | `ValueError: unsupported eval mode 'dynamic'` |
| `the readback-free run helper exists` | `dense_engine._iter_plan_chunks` missing: every chunk result is read back inside `run_plan` |

`stage parity` was already green (13/13): the dense *value* route matches the
specialized engine within `rtol=atol=1e-4` with exact NaN/±inf masks
(`non_bit_exact_finite` 0–7, `max_rel` <= 2e-7), and that stays the bar for the
in-graph loss.

## 2. What still crosses the host (per call, after this round)

| crossing | numpy backend | device backend (iree) |
| --- | --- | --- |
| `inputs` / `labels` D2H (`_input_to_numpy` / `_as_numpy`) | 1 + 1 | 1 + 1 |
| population tensors D2H (`prepare_dense`) | 3 | 3 |
| payload H2D (`etl.from_numpy`) | **0** (host interpreter) | 7 columns per `p`-chunk (once each) + `x` once per `n`-chunk + `labels` + keep mask once per `n`-chunk — and **each exactly once per call** since this round's payload memo (it was one per `(p0, n0)` chunk before) |
| chunk result readback | 1 × `(p_chunk,)` per chunk — **`(P,)`-scale** | same |
| host→torch + H2D of the loss | 1 × `(P,)` `torch.from_numpy` + `.to(device)` | same |
| whole prediction H2D / torch-side reduction | **gone** (folded into the graph) | **gone** |

The value route (`evaluate_dense`) still assembles the `(pop_size, N, 1)`
prediction and uploads it once; a device-resident *value* path stays blocked on
ETL upstream (no zero-copy export: `core.Tensor.__dlpack__` raises for every
payload-backed tensor and `etl.from_dlpack` is host-memory v1 only).

## 3. Measurement protocol

Warm per-call medians from `device_resident_check.py --measure`: each case is
called once untimed (build/compile + cache warm), then `--repeats` timed calls
with a `torch.cuda.synchronize()` around each sample, and finally one *counted*
call whose `Traffic` summary is recorded next to the timing.  The before/after
numbers of §4 were taken from the same script at the same grid on the same box
(see §5 for the exact before run).  Run-to-run spread on the shared box is
30–50 % (other tenants), so every cell below is a median of 7 and the re-runs of
§4.3 show the spread explicitly.

## 4. Round 2 — after

### 4.1 The numpy hot path: warm time unchanged, traffic down

Exact commands (identical on both sides; the "before" side runs the SAME script
from a scratch worktree at the pre-round base commit `93a783f`):

```bash
git worktree add --detach /tmp/dr_before_wt 93a783f
cp src/evogp/tree/device_resident_check.py /tmp/dr_before_wt/src/evogp/tree/
cd /tmp/dr_before_wt   # or the repo root for "after"
PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 <venv>/bin/python \
  src/evogp/tree/device_resident_check.py --measure \
  --cases 16x64,16x1024,64x64,64x1024 --tree-rows 64 --repeats 7
```

| case (dense forced, numpy) | before ms | after ms | readback el. before → after |
| --- | --- | --- | --- |
| `SR_fitness P=16 N=64` | 64.77 | 46.89 | 1024 → 16 |
| `SR_fitness P=16 N=1024` | 103.79 | 88.57 | 16384 → 16 |
| `SR_fitness P=64 N=64` | 101.50 | 55.28 | 4096 → 64 |
| `SR_fitness P=64 N=1024` | 545.52 | 597.73 | 65536 → 64 |
| `Tree.SR_fitness N=64` (pop=1) | 52.39 | 37.17 | 64 → 1 |
| `Forest.forward P=16` (value route, unchanged code) | 45.63 | 37.75 | 16 → 16 |

### 4.2 The iree hot path: the loss fold pays off where the readback costs

Same script, `--backend iree --gpu 0` (staged, GPU 6, one process per side), on
the same fixture (`fixture(16, rows, seed=41)`); after the round the readback is
`(P,)` = 16 elements instead of the whole `(P, N)` prediction:

| P=16, iree | before ms (median) | after ms (median) | before min | after min | readback el. |
| --- | --- | --- | --- | --- | --- |
| N=64 | 4.13 | 3.90 | 4.10 | 3.25 | 1024 → 16 |
| N=1024 | 6.61 | 6.05 | 6.40 | 5.90 | 16384 → 16 |
| N=4096 | 6.56 | 7.66 | 6.07 | 7.51 | 65536 → 16 |
| N=8192 | 10.76 | 7.54 | 10.11 | 7.44 | 131072 → 16 |
| N=16384 | 21.02 | 10.64 | 17.40 | 10.44 | 262144 → 16 |

The N=4096 cell is the single inversion and sits inside the box's noise band
(the other four cells improve monotonically with N: −6 % / −8 % / −30 % / −49 %).
Fitting the two curves, the dense per-row slope drops from ≈0.81 µs/row to
≈0.45 µs/row: the removed term (the `(P, N)` D2H + its H2D back to torch + the
device-side torch reduction) is N-proportional while the added in-graph loss is a
handful of `(P, N)` ops on a ≈264·L-op graph (≈0.2 % of the interpreter work —
which is why numpy, interpreter-bound, shows no warm change either way).

### 4.3 Re-run spreads (the same two extremes, alternating sides)

`--cases 16x64,64x1024 --tree-rows 64 --repeats 9`, three rounds each:

| round | `P=16 N=64` before / after | `P=64 N=1024` before / after | `Tree N=64` before / after |
| --- | --- | --- | --- |
| 1 | 43.98 / 44.12 | 584.62 / 665.13 | 38.79 / 40.92 |
| 2 | 63.24 / 40.12 | 679.82 / 672.48 | 41.08 / 35.93 |
| 3 | 40.63 / 47.08 | 674.95 / 657.51 | 39.88 / 35.30 |

i.e. on numpy the two routes are statistically indistinguishable (the spread is
the box, not the change), which is the expected outcome: the folded loss removes
*crossings*, not interpreter work.

## 5. Task-5 decision: the `auto` gates stay at 16 / 2048

Measured with `dense_crossover_probe.py` (this directory; dev CLI, staged, one
process per side, GPU 6, `MODE=dense|specialized`, `AB_ROWS=...`, `P=16`,
`L=16`, medians of 7; before = the `/tmp/dr_before_wt` worktree at `93a783f`):

| P=16, N | dense before | dense after | specialized |
| --- | --- | --- | --- |
| 64 | 4.13 | 3.90 | 17.76 |
| 1024 | 6.61 | 6.05 | 19.00 |
| 4096 | 6.56 | 7.66 | 15.40 |
| 8192 | 10.76 | 7.54 | (rows-flat, ≈0.9–1.0 ms/tree) |
| 16384 | 21.02 | 10.64 | 26.01 |

Readings:

* the loss fold makes dense strictly better at every measured size (the one
  N=4096 inversion is inside the noise band, §4.2);
* at this shape dense is ahead of the specialized engine across the whole
  measured range, so the *measured* crossover (dense-after ≈ 10.6 ms at 16384
  against ≈15–26 ms for 16 specialized trees) is **beyond the grid** (≈26k rows
  by linear extrapolation) — the old tightest recorded crossover was 3.4–6.8k
  rows, i.e. the change moved it in the favourable direction;
* **no constant changes.**  `EVOGP_DENSE_AUTO_MIN_POP` (16) and
  `EVOGP_DENSE_AUTO_MAX_ROWS` (2048) are unchanged, because (a) the gate exists
  to keep `auto` on the safe side of a SHAPE-DEPENDENT crossover and dense's
  cost is ∝ `L` (this round measured `L=16`; the bench's reference is `L=32`,
  where dense's op count roughly doubles and the crossing moves back down), and
  (b) the numpy default route is untouched and never selects dense.  Raising the
  rows gate is a policy change that needs a direct dense-vs-specialized crossover
  measurement at the reference shape — `dense_crossover_probe.py` is that
  protocol, and §5's table is its `L=16` answer.

## 6. Battery status before → after

* pre-round base `93a783f` + the CURRENT battery:
  `== FAILED: 42/63 checks passed ==` (`/tmp/drcheck/before_stages.log`) — red on
  exactly the round's issues: 2 `inputs` conversions, the `(8, 37)` readback,
  `torch.from_numpy [8,37,1]` + `[37,1]`, 296 x preparations, 0 labels
  preparations, the missing `_iter_plan_chunks`, the missing `dynamic` alias.
* HEAD: `== OK: 70/70 checks passed ==`
  (`--stage policy,hotpath,parity,seam`, numpy).

## 7. Gates + suite on the final commit

| command | result |
| --- | --- |
| `dense_spike_check.py --backend numpy --skip-measure` | `37/37` |
| `dense_integration_check.py --stage numpy,fallback` | `119/119` |
| `dense_integration_check.py --stage iree --gpu 0` | `7/7` (teardown-only nanobind notice after `OK`, exit 0) |
| `dense_adversarial_check.py --backend numpy` | `192/192`, 0 non-bit-exact finite values |
| `pytest src/evogp/tests -q` | `308 passed, 4 skipped` (105.9 s) |
| `EVOGP_EVAL_MODE=dense pytest src/evogp/tests -q` | `308 passed, 4 skipped` (246.2 s, measured while the iree probes above were running) |

## 8. The seam a future `build_dense_runner` uses

`dense_engine._iter_plan_chunks(plan, executable, x, labels=None, *,
backend_name, device, device_resident=False)` — yields
`((p0, p_chunk, n0, n_chunk), raw_etl_result)` with **zero** host readbacks and
each distinct payload prepared AND uploaded exactly once per call (the seam the
battery's `seam` stage pins).  `device_resident=True` refuses the numpy backend
and any chunked plan: a chunked device-side assembly needs either an ETL
zero-copy export or a second compiled graph, both still upstream/structural.

## 9. Task-5 follow-up: the trace cap is now 64 (+ the one warning)

**Decision applied (item 5 of `DENSE_EVAL.md` §9).** `dense_engine.DEFAULT_MAX_TREE_LEN`
= **64** (was 32); `EVOGP_DENSE_MAX_TREE_LEN` still overrides it per process, and
`L = min(cap, row_len)` is unchanged. Evidence (measurement round over HEAD
`283e6ce`, iree, per shape signature): compile ≈13.9 s at `L=32` / ≈31.6 s at
`L=64` / ≈69.0 s at `L=128`; for a 33–64-node population at P=64 the dense route
is ONE build (≈28.9 s cold, ≈11.6 ms warm) against the silent specialized
fallback's 64 builds (≈48.8 s cold, ≈110.3 ms warm; break-even ≈45–60 distinct
structures), while at P=16 it trades 4.1× cold for 4.9× warm. `L=128` was
**rejected**: it would tax every `row_len >= 128` population (shipped examples are
128/256/512 wide) with a ≈69 s compile. numpy is unaffected under the default
`auto` (dense is never auto-selected there) and memory stayed safe in the round
(peak RSS ≤ 2.7 GB, no cell skipped). Consequence to remember: a `row_len >= 64`
population now traces `L=64` even when its REALIZED trees are short — pin
`EVOGP_DENSE_MAX_TREE_LEN=32` for such a shape; the realized-longest-tree variant
(power-of-two, clamped) is documented future work, not implemented
(`DENSE_EVAL.md` §9.4).

**The one warning.** An over-cap tree used to fall back silently on every backend.
That is the one decline whose fallback is *expensive* on a device backend, so
`prepare_dense` now emits a deduped, once-per-process `RuntimeWarning` naming the
longest tree, the cap, `EVOGP_DENSE_MAX_TREE_LEN` and the per-structure-compile
consequence (`dense_engine._warn_trace_cap` / `_TRACE_CAP_WARNED`; gated to
non-numpy backends). Every other decline (multi-output, structural doubt, the
`auto` rows ceiling) keeps the silent-fallback contract, and the dedupe flag is
only consumed by a real warning (all pinned by the checks below).

### Verified after the change (repo root, `PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6`)

| command | result |
| --- | --- |
| `pytest src/evogp/tests -q` (default) | `308 passed, 4 skipped` in 103.65 s (baseline unchanged; no new warnings) |
| `device_resident_check.py --stage policy,hotpath,parity,seam` | `79/79` (was 70/70: +9 trace-cap/warning checks) |
| `dense_integration_check.py --stage numpy,fallback` | `119/119` |
| `runner_check.py` | `48/48` |
| `dense_spike_check.py --backend numpy --skip-measure` | `37/37` |

The 9 new `policy` checks: the unresolved default is 64; `L = min(cap, row_len)`
still gives a `max_tree_len=32` population `L=32`; a 128-wide row with a short tree
now traces `L=64`; a 64-node tree fits the default cap; a 70-node tree declines
with `DenseUnsupported`; that decline is SILENT on numpy; on a device backend it
warns exactly ONCE over three calls and the message names the longest tree, the
cap, the env var and the compile consequence; and multi-output / malformed
declines stay silent without consuming the flag.

### Not re-derived here

* The dense-forced suite instrumentation (`510 attempts / 492 dense / 18
  fallbacks`) was re-attempted with a counting plugin
  (`attempts = evaluate_dense + sr_fitness_dense` calls, declines by reason) and the
  run was killed by the sandbox's memory limit at ~23 % (the numpy dense graphs are
  host-memory heavy; the earlier 246 s dense-forced run predates this factor), so
  the recorded figure stands as the baseline. Expected delta after the raise: the
  **3 over-cap** fallbacks (of 18) become dense-supported wherever those trees are
  ≤ 64 nodes, i.e. the fallback count should drop to ~15 (multi-output only) —
  to be re-measured on a box without the memory cap.
* The staged P=64 iree A/B (1 compile ≈28.9 s cold / ≈11.6 ms warm vs 64 builds)
  was NOT re-run in this round; the numbers above are the measurement round's.
  Re-run protocol: `dense_crossover_probe.py` with a 33–64-node population
  (`AB_POP=64`, one process per side, `CUDA_VISIBLE_DEVICES=6`, visible `--gpu 0`).
