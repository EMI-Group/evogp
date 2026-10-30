# `t1a1_baseline` — evoxc evaluation routes vs the plain-PyTorch references

Baseline measurement for the joint optimization effort, taken **2026-10-30** on **AMD Instinct MI300X VF**
(191.7 GB HBM), torch **2.13.0+rocm7.1**, numpy 2.5.4, HIP 7.1.52802, iree-base 3.11.0,
python 3.12.3 at **`/tmp/rocvenv/bin/python`**, `evoxc` imported from **`/root/evoxc`**
(`evoxc_path=/root/evoxc/evoxc/__init__.py`, sha `9f66feca…`), `EVOXC_DISABLE_LOCATIONS=1`, tree root at
`71f8337` (the engine code the records measured; the records commit `8b316b4` adds no engine change).

**What was measured.** evogp's evoxc evaluation routes against the two plain-PyTorch reference evaluators, on
the raw batched population forward pass (`batch_forward`) and on the training loss path (`sr_fitness`).
Six cells (`p<pop>-L<tree_len>-i<input_len>-N<rows>`, `-d7` = `max_layer_cnt 7`) × five arms
(`torch-batched`, `torch-per-tree`, `evoxc-dense-iree`, `evoxc-batch-iree`, `evoxc-auto-iree`).

**Protocol.** One process per (cell, arm) behind an idle-device guard (25/25 configs on an idle device, 0
contaminated), seed 0, `warmup 2` untimed calls, `reps 5` sync-bracketed samples, **warm medians**, the cold
first call timed separately (`first_call_ms`, never in the medians), workloads `sr_fitness,batch_forward`,
`EVOXC_DISABLE_LOCATIONS=1`; campaign elapsed 622.7 s. Two invocations share this directory:
`campaign.json` = the five safe cells × five arms (25 configs, all `status ok`), `campaign.d7.json` = the d7
cell × four arms. `evoxc-dense-iree` was deliberately **not scheduled** on the d7 cell: an explicit dense
route there gets no trace-cap raise/envelope and degrades into per-structure `specialized`+iree with
unbounded artifact growth (measured cgroup OOM kill), so `check.py` reports it as an allowed missing arm.

**Status.** 29 records (5×5 + 4), **0 failed configs**.
Gate: `python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/t1a1_baseline --allow-snapshot --allow-missing --allow-digest-mismatch`
→ **29 records / 538 checks / 6 failures**, the six being exactly the declared `no digest anywhere in the cell`
class for the `forward` workload this campaign did not schedule (an expectation of the scoped schedule, **not**
numeric mismatches: one per cell). WARNs: **1 missing-arm** (d7 `evoxc-dense-iree`, `never scheduled`) plus the
**6 digest-split** WARNs listed in §3. The gate exits 1 by design on that declared class.
(The same command over the container *as committed* also folds `decompose/` and `validation/` in as if they were
cells — 586 checks / 24 failures; a container holding `decompose*/` is documented as not gateable in one run, so
538/6 over the campaign+cells scope is the figure of record. `RENDERED.md` picks the same two directories up as
extra "cells"/arms for the same reason; §1 below is the authoritative table set.)

**Ratio convention used everywhere below.** `torch_median / arm_median`, i.e. **> 1.00 means the arm is faster
than that torch baseline** and **< 1.00 means it is slower**. Columns are read from
`<cell>/<arm>.json` → `workloads.<name>.median_ms` and `workloads.<name>.first_call_ms`.

---

## 1. Per-cell warm medians

Each cell heading carries the population served: `trace length` = `structures.max_tree_len` (the dense trace
cap the cell runs at), `actual max` = `structures.tree_len_max` (the longest real tree), the mean tree length and
the number of distinct structures.

### `p256-L32-i8-N1024` — pop 256, trace length 32, actual max 31, mean 15.95, **225 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 4.0044 | 1.00× | 6.03× | 4.0734 | 1.00× | 7.75× | 4.0 / 65.4 |
| `torch-per-tree` | 24.1321 | 0.17× | 1.00× | 31.5496 | 0.13× | 1.00× | 25.5 / 160.5 |
| `evoxc-dense-iree` | **1.7463** | **2.29×** | 13.82× | **1.1246** | **3.62×** | 28.05× | 3296.2 / 3452.4 |
| `evoxc-batch-iree` | 3.3765 | 1.19× | 7.15× | 3.7273 | 1.09× | 8.46× | 2704.4 / 3944.5 |
| `evoxc-auto-iree` | 1.7772 | 2.25× | 13.58× | 1.1415 | 3.57× | 27.64× | 3239.3 / 3440.5 |

### `p256-L64-i32-N1024` — pop 256, trace length 64, actual max 31, mean 15.95, **232 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 4.4286 | 1.00× | 5.35× | 4.4936 | 1.00× | 6.79× | 4.5 / 63.5 |
| `torch-per-tree` | 23.6729 | 0.19× | 1.00× | 30.5108 | 0.15× | 1.00× | 23.2 / 158.1 |
| `evoxc-dense-iree` | **2.8098** | **1.58×** | 8.43× | **2.0301** | **2.21×** | 15.03× | 6446.9 / 6677.5 |
| `evoxc-batch-iree` | 3.3316 | 1.33× | 7.11× | 3.7290 | 1.21× | 8.18× | 2875.0 / 4069.8 |
| `evoxc-auto-iree` | 2.8653 | 1.55× | 8.26× | 2.0408 | 2.20× | 14.95× | 6394.4 / 6609.0 |

### `p256-L64-i32-N4096` — pop 256, trace length 64, actual max 31, mean 15.95, **232 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 4.9600 | 1.00× | 4.82× | 4.9839 | 1.00× | 6.27× | 4.9 / 65.8 |
| `torch-per-tree` | 23.9108 | 0.21× | 1.00× | 31.2314 | 0.16× | 1.00× | 24.0 / 160.3 |
| `evoxc-dense-iree` | 7.8316 | **0.63×** | 3.05× | 5.8213 | **0.86×** | 5.37× | 6413.3 / 6590.9 |
| `evoxc-batch-iree` | 5.0879 | 0.97× | 4.70× | 5.2529 | 0.95× | 5.95× | 2859.1 / 4341.0 |
| `evoxc-auto-iree` | 7.9930 | 0.62× | 2.99× | 5.8341 | 0.85× | 5.35× | 6456.8 / 6608.6 |

### `p1000-L64-i32-N1024` — pop 1000, trace length 64, actual max 31, mean 16.21, **841 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 4.8415 | 1.00× | 19.52× | 4.8540 | 1.00× | 25.53× | 4.8 / 66.7 |
| `torch-per-tree` | 94.5147 | 0.05× | 1.00× | 123.9384 | 0.04× | 1.00× | 94.0 / 255.5 |
| `evoxc-dense-iree` | 7.6976 | **0.63×** | 12.28× | 5.7140 | **0.85×** | 21.69× | 6494.4 / 6744.4 |
| `evoxc-batch-iree` | 7.2219 | 0.67× | 13.09× | 7.4650 | 0.65× | 16.60× | 10981.0 / 16684.5 |
| `evoxc-auto-iree` | 7.7259 | 0.63× | 12.23× | 5.7156 | 0.85× | 21.68× | 6393.1 / 6633.7 |

### `p4096-L64-i32-N1024` — pop 4096, trace length 64, actual max 31, mean 15.71, **3266 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 10.5142 | 1.00× | 35.51× | 10.5328 | 1.00× | 46.89× | 10.5 / 68.8 |
| `torch-per-tree` | 373.3285 | 0.03× | 1.00× | 493.8939 | 0.02× | 1.00× | 373.0 / 635.8 |
| `evoxc-dense-iree` | 45.9680 | **0.23×** | 8.12× | 37.2471 | **0.28×** | 13.26× | 6581.3 / 6812.8 |
| `evoxc-batch-iree` | 21.9837 | 0.48× | 16.98× | 21.4083 | 0.49× | 23.07× | 65575.1 / 154885.1 |
| `evoxc-auto-iree` | 46.9537 | 0.22× | 7.95× | 36.8639 | 0.29× | 13.40× | 6538.9 / 6664.5 |

### `p2048-L128-i32-N1024-d7` — pop 2048, trace length 128, **actual max 123**, mean 42.89, **1665 distinct structures**

| arm | `batch_forward` ms | vs `torch-batched` | vs `torch-per-tree` | `sr_fitness` ms | vs `torch-batched` | vs `torch-per-tree` | cold `first_call_ms` (bf / sf) |
|---|---|---|---|---|---|---|---|
| `torch-batched` | 26.4492 | 1.00× | 20.58× | 26.4852 | 1.00× | 22.08× | 26.5 / 86.0 |
| `torch-per-tree` | 544.4131 | 0.05× | 1.00× | 584.9023 | 0.05× | 1.00× | 525.0 / 712.0 |
| `evoxc-dense-iree` | not scheduled | — | — | not scheduled | — | — | — |
| `evoxc-batch-iree` | **19.6839** | **1.34×** | 27.66× | **20.8999** | **1.27×** | 27.99× | 75549.1 / 164062.9 |
| `evoxc-auto-iree` | 110.3441 | **0.24×** | 4.93× | 105.6497 | **0.25×** | 5.54× | 14307.4 / 14460.5 |

The `auto` arm resolves to the dense route on every cell, so its numbers track `evoxc-dense-iree` to within
0.03–2.14 % on the five cells where both were scheduled; on the d7 cell `auto` runs dense at the **raised trace
cap 128** and the explicit dense arm was not scheduled.

---

## 2. Route resolution, merged plan and cold compile per arm

| cell | `routes.sr_fitness` (dense arm) | `dense_route.supported` | `routes.sr_fitness` (batch arm) | `routes.sr_fitness` (auto arm) | `auto_trace_cap.caps.sr_fitness` (`auto` / `dense`) |
|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | dense / iree | `true` (probe 1.4598 ms) | batch / iree | auto / iree → dense | 64 / 64 (base 64, not pinned) |
| `p256-L64-i32-N1024` | dense / iree | `true` (2.2082 ms) | batch / iree | auto / iree → dense | 64 / 64 |
| `p256-L64-i32-N4096` | dense / iree | `true` (6.0148 ms) | batch / iree | auto / iree → dense | 64 / 64 |
| `p1000-L64-i32-N1024` | dense / iree | `true` (5.8847 ms) | batch / iree | auto / iree → dense | 64 / 64 |
| `p4096-L64-i32-N1024` | dense / iree | `true` (37.6827 ms) | batch / iree | auto / iree → dense | 64 / 64 |
| `p2048-L128-i32-N1024-d7` | not scheduled | — | batch / iree | auto / iree → dense | **128** / 64 (`auto` raises the cap past the 123-node trees) |

Batch plan actually merged — `batch_plan_summary.loss` (the `sr_fitness` plan) and `.value` (the `batch_forward`
plan), shipped defaults (`EVOGP_BATCH_BAND_SIZE=16`, pool 4, `CHUNK_BANDS` off):

| cell | plan | `n_functions` | `emitted_ops` | `chunk_count` / `chunk_sizes` | `compile_workers` | `band_report.dispatch_formation` (requested) | `chunk_bands` (clamped) |
|---|---|---|---|---|---|---|---|
| `p256-L32-i8-N1024` | loss | 225 | 5 061 | 4 / [57,56,56,56] | 4 | `aggressive` (`auto`) | 4 per chunk (no) |
| | value | 225 | 4 174 | 4 / [57,56,56,56] | 4 | `aggressive` (`auto`) | 4 per chunk (no) |
| `p256-L64-i32-N1024` | loss | 232 | 5 497 | 4 / [58,58,58,58] | 4 | `aggressive` (`auto`) | 4 (no) |
| | value | 232 | 4 589 | 4 / [58,58,58,58] | 4 | `aggressive` (`auto`) | 4 (no) |
| `p256-L64-i32-N4096` | loss | 232 | 5 497 | 4 / [58,58,58,58] | 4 | `aggressive` (`auto`) | 4 (no) |
| | value | 232 | 4 589 | 4 / [58,58,58,58] | 4 | `aggressive` (`auto`) | 4 (no) |
| `p1000-L64-i32-N1024` | loss | 841 | 21 339 | 4 / [211,210,210,210] | 4 | `aggressive` (`auto`) | 14 (no) |
| | value | 841 | 17 986 | 4 / [211,210,210,210] | 4 | `aggressive` (`auto`) | 14 (no) |
| `p4096-L64-i32-N1024` | loss | 3 265 | 78 189 | 4 / [817,816,816,816] | 4 | `aggressive` (`auto`) | **15 (clamped from 52)** |
| | value | 3 265 | 65 227 | 4 / [817,816,816,816] | 4 | `aggressive` (`auto`) | **15 (clamped)** |
| `p2048-L128-i32-N1024-d7` | loss | 1 664 | 93 751 | 4 / [416,416,416,416] | 4 | `aggressive` (`auto`) | **15 (clamped from 51)** |
| | value | 1 664 | 87 121 | 4 / [416,416,416,416] | 4 | `aggressive` (`auto`) | **15 (clamped)** |

Cold compile cost (`workloads.<name>.first_call_ms` / `cache.compile_misses_first_call`):

| cell | `evoxc-dense-iree` bf / sf | `evoxc-auto-iree` bf / sf | `evoxc-batch-iree` bf / sf |
|---|---|---|---|
| `p256-L32-i8-N1024` | 3 296.2 / 3 452.4 ms, 1 / 1 miss | 3 239.3 / 3 440.5 ms, 1 / 1 | 2 704.4 / 3 944.5 ms, 0 / 0 |
| `p256-L64-i32-N1024` | 6 446.9 / 6 677.5 ms, 1 / 1 | 6 394.4 / 6 609.0 ms, 1 / 1 | 2 875.0 / 4 069.8 ms, 0 / 0 |
| `p256-L64-i32-N4096` | 6 413.3 / 6 590.9 ms, 1 / 1 | 6 456.8 / 6 608.6 ms, 1 / 1 | 2 859.1 / 4 341.0 ms, 0 / 0 |
| `p1000-L64-i32-N1024` | 6 494.4 / 6 744.4 ms, 1 / 1 | 6 393.1 / 6 633.7 ms, 1 / 1 | 10 981.0 / 16 684.5 ms, 0 / 0 |
| `p4096-L64-i32-N1024` | 6 581.3 / 6 812.8 ms, 1 / 1 | 6 538.9 / 6 664.5 ms, 1 / 1 | 65 575.1 / 154 885.1 ms, 0 / 0 |
| `p2048-L128-i32-N1024-d7` | not scheduled | 14 307.4 / 14 460.5 ms, 1 / 1 | 75 549.1 / 164 062.9 ms, 0 / 0 |

The dense/auto cold call is one `evoxc` compile per workload (engine cache miss 1), 3.24–6.81 s at trace lengths
32/64 and 14.31–14.46 s at the raised cap 128. The batch arm reports 0 engine-cache misses because it compiles
through its own merged-plan cache instead — its cold call **is** the merged compile: **2.70 s** at
`p256-L32-i8-N1024` up to **164.06 s** at the d7 cell (p4096: 65.58 s on the value plan, 154.89 s on the loss plan).
`torch-batched` pays only lazy-kernel warm-up (4.0–26.5 ms on `batch_forward`, 63.5–86.0 ms on `sr_fitness`).

---

## 3. Numeric validation (`validation/`, 3 records, 64 trees each vs the CPU oracle `src/evogp/tests/reference_eval.py`)

Records: `validation-dense-iree-p256-L64-i32-N1024`, `validation-dense-iree-p4096-L64-i32-N1024`,
`validation-batch-iree-p256-L64-i32-N1024`; each runs the three function sets `arith` (`+ - * /`), `rich`
(`+ - * / max min < sin exp log abs sqrt loose_div loose_inv`) and `ternary` (`if + - * /`), comparing the engine
against the oracle and against both torch arms.

* **`dense`+iree, arithmetic and ternary sets:** `batch_forward` and `forward` are **bitwise equal** to the oracle
  and to both torch arms (max_abs 0, 0 NaN/Inf placement mismatches) at both pop 256 and pop 4096.
  `sr_fitness` differs only by float reduction order: engine-vs-oracle 3.43e-7 relative (abs 32 on a 2.99e9 scale)
  at pop 256 and 2.29e-6 (abs 768) at pop 4096; engine-vs-torch-arm 2.22e-7…2.52e-6 relative.
* **`rich` set:** the transcendental lowering is approximate at large magnitudes — engine-vs-oracle up to
  **1.61 abs** (rel 15.6 on a 1e9 abs scale) on `batch_forward`, while the torch arms deviate from the *same*
  oracle by **1.42 abs** (rel 2.30). The spread is magnitude-driven, not route-specific; the per-row `forward`
  path agrees to ≤3e-8 abs. 0 NaN/Inf placement mismatches.
* **`batch`+iree:** `batch_forward` is **non-bitwise on the arithmetic set** — 7.81e-2 abs / 2.38e-4 relative
  (oracle abs scale 1.70e6) at pop 256, and 7.81e-2 / 2.38e-4 against both torch arms; `ternary` 4.77e-7 abs /
  1.92e-4 rel; per-row `forward` 1.91e-6 abs / 1.09e-7 rel; the loss path on the same arithmetic set agrees to
  ≤7.8e-7 relative (abs 64 vs the oracle, abs 256 / 6.35e-7 against the torch arms, on a 2.99e9 scale). `rich`
  reads the same 1.61 / 1.42 abs spread. This is why the batch records form their own digest group.
* **Cross-arm digests (`check.py`):** NaN/finite/±Inf counts agree across **all** arms on every cell for both
  workloads (the NaN counts are the population's own NaN trees: 1/3/27/44 on `sr_fitness`), as does `abs_sum` to
  ≤4e-6 relative. The `sr_fitness` sha256 nevertheless splits into 4 groups per cell (`torch-batched`,
  `torch-per-tree`, `dense`+`auto`, `batch`) — last-bit reduction-order differences; the gate does not sha-gate
  `sr_fitness`, so this is not a WARN. `batch_forward` splits into **2 sha256 groups on three cells**
  (`p256-L32-i8`, `p256-L64-i32-N1024`, `p256-L64-i32-N4096` — always the batch arm alone) and **3 on three cells**
  (`p1000`, `p4096`, `d7`, where `dense`+`auto` also split from the torch pair) — the 6 digest-split WARNs.
  `forward` was not scheduled, hence the 6 declared `no digest anywhere in the cell` failures.

---

## 4. Where the evoxc route loses

`evoxc-dense-iree` wins at **pop 256 with few rows** — 2.21× on `sr_fitness` / 1.58× on `batch_forward` at
`p256-L64-i32-N1024`, 3.62× / 2.29× at `p256-L32-i8-N1024` — and is **below `torch-batched` once rows or
population grow**: 0.86× / 0.63× at `p256-L64-i32-N4096`, 0.85× / 0.63× at `p1000`, 0.28× / 0.23× at `p4096`,
and the `auto` route at the d7 cell 0.25× / 0.24× at the raised cap 128. The loss tracks all three shape axes and
does so far harder than the torch baseline: **4× rows** (`p256-L64-i32-N1024` → `p256-L64-i32-N4096`, the same
232 structures — both cells carry `structure_signature 0a0dee3fec0cfb9c`) costs dense
2.87× its `sr_fitness` call (2.0301 → 5.8213 ms) and 2.79× its `batch_forward` call, while `torch-batched` moves
only 1.11× / 1.12× (it is launch/occupancy-bound, not work-bound); **4× population** (1000→4096) costs dense
6.52× / 5.97× against 2.17× / 2.17× for `torch-batched`; and the **node-count cell** (123-node trees, trace length
128, 93 751 emitted ops) reads the worst `sr_fitness` ratio of the column (0.25×) and is level with `p4096` on
`batch_forward` (0.24× against 0.23×).
From the committed decomposition (`decompose/<cell>.json`, route `dense-iree`, instrumented warm calls) the cost
sits **inside the compiled graph**: `run_loss_plan` / `run_plan` is **88.8–99.4 %** of the call and host glue is
only **0.6–11.2 %**, so at large shape this is device-side graph execution, not the Python layer. The one host
term that does scale is the device→host readback `_to_host`: **9.0 %** of the `sr_fitness` call at
`p256`/1 024 rows, and **26.0 %** of the `batch_forward` call at `p256`/1 024 rows, 21.5 % at `p256`/4 096 rows,
23.5 % at `p1000`, 18.8 % at `p4096` — i.e. 18.8–26.0 % of the large-shape `batch_forward` calls sit in the
readback of the returned tensor, by far the largest host term on that workload.

Only the **batch** route gets back above `torch-batched` at large shape: `batch_forward` / `sr_fitness` 1.34× /
1.27× on the d7 cell, 0.97× / 0.95× at `p256`/4 096 rows, 0.67× / 0.65× at `p1000`, 0.48× / 0.49× at `p4096`. At the two row-lean pop-256
cells it still beats `torch-batched` (1.19–1.33× `batch_forward`, 1.09–1.21× `sr_fitness`) but sits 1.19–1.93×
behind dense on `batch_forward` and 1.84–3.31× behind on `sr_fitness`. It pays for that with the merged compile:
2.70 s at `p256-L32-i8-N1024` up to 164.06 s at the d7 cell, and 65.58 / 154.89 s at `p4096`, per plan, cold.

---

## 5. Reproduce

```bash
# repo root as CWD; interpreter is the project venv used by the records
cd <repo root>
export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1
PY=/tmp/rocvenv/bin/python
OUT=test/benchmark/evoxc/results/t1a1_baseline      # a RE-RUN must write to a fresh directory

# 1. the two campaign invocations (one process per (cell, arm), idle-device guard)
$PY test/benchmark/evoxc/campaign.py --out $OUT \
    --cells p256-L32-i8-N1024,p256-L64-i32-N1024,p256-L64-i32-N4096,p1000-L64-i32-N1024,p4096-L64-i32-N1024 \
    --arms torch-batched,torch-per-tree,evoxc-dense-iree,evoxc-batch-iree,evoxc-auto-iree \
    --workloads sr_fitness,batch_forward --warmup 2 --reps 5 --seed 0 --timeout 900
$PY test/benchmark/evoxc/campaign.py --out $OUT \
    --cells p2048-L128-i32-N1024-d7 \
    --arms torch-batched,torch-per-tree,evoxc-batch-iree,evoxc-auto-iree \
    --workloads sr_fitness,batch_forward --warmup 2 --reps 5 --seed 0 --timeout 900
#    campaign.py always writes <out>/campaign.json, so the d7 invocation's driver file is committed here as
#    campaign.d7.json (4 configs) next to the 25-config campaign.json.

# 2. the decomposition column (four cells, two routes, both workloads)
for c in p256-L64-i32-N1024 p256-L64-i32-N4096 p1000-L64-i32-N1024 p4096-L64-i32-N1024; do
  $PY test/benchmark/evoxc/decompose.py --cell "$c" --routes dense-iree,auto-iree \
      --workloads sr_fitness,batch_forward --out $OUT/decompose
done

# 3. numeric validation (route selected by the env)
EVOGP_EVAL_MODE=dense EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py \
    --cells p256-L64-i32-N1024 --func-sets arith,rich,ternary \
    --out $OUT/validation/validation-dense-iree-p256-L64-i32-N1024.json
EVOGP_EVAL_MODE=dense EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py \
    --cells p4096-L64-i32-N1024 --func-sets arith,rich,ternary \
    --out $OUT/validation/validation-dense-iree-p4096-L64-i32-N1024.json
EVOGP_EVAL_MODE=batch EVOGP_EVOXC_BACKEND=iree $PY test/benchmark/evoxc/validate.py \
    --cells p256-L64-i32-N1024 --func-sets arith,rich,ternary \
    --out $OUT/validation/validation-batch-iree-p256-L64-i32-N1024.json

# 4. gate and render
python3 test/benchmark/evoxc/check.py --results $OUT --allow-snapshot --allow-missing --allow-digest-mismatch
python3 test/benchmark/evoxc/summarize.py --results $OUT --out $OUT/RENDERED.md
```

This directory is **reference evidence for the optimization effort and must not be rewritten in place**: a
re-run writes into a fresh result directory. The `.log` worker files are gitignored by design, so only the JSON
records, `RENDERED.md` and this report are committed.
