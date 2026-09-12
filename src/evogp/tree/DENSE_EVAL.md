# DENSE_EVAL.md — the dense one-graph population evaluator

Design document of record for `dense_engine.py`: the evaluator that runs a **whole population through
ONE compiled ETL executable** fed with the population's node tensors as *runtime data*, instead of one
ETL graph per tree structure.

**Status: integrated.** It is wired into the five public entry points (`Forest.forward`,
`Forest.batch_forward`, `Forest.SR_fitness`, `Tree.forward`, `Tree.SR_fitness`) and selected per call
by the `EVOGP_EVAL_MODE` master switch (`dense` / `specialized` / `auto`, default `auto`).
Semantics are unchanged: this document is about *where* the numbers are computed, never *what* they
are — `ENGINE_SPEC.md` (§3 op table, §4 evaluation order, §5 SR fitness, §12 deviations) stays the
semantics of record, and both paths share the same private op emitters by construction.

---

## 1. Why a dense evaluator exists

The specialized engine (`etl_engine.py`, the default and the reference) is per structure:

* **one compile per distinct tree structure** — `etl.build_cached` trace → lower → compile → load,
  measured ~0.5 s per distinct structure on IREE at the reference shape (L=32, realized tree lengths
  median 17; firm-up median of 11 direct full-population cells, `dense_bench_findings.md` §2, against
  0.34–1.23 s per tree in the earlier single-tree samples) and single-digit ms per structure on numpy
  (the grid's cold columns over 55 / 170 / 793 distinct programs). A fresh random population is a
  **cold miss per individual** (the bench's pop-1000 reference population held 793 distinct programs).
* **one Python invocation per tree per pass** — a `pop_size` fitness pass is `pop_size` `build_cached`
  look-ups plus `pop_size` `etl.run` calls, each re-validating its statics. Both per-lookup costs are
  small on the ETL side now (a repeated program's key is a ~17–25 µs fingerprint replay, a 60-token
  `etl.run` 67–71 µs — §7), but the *number* of Python invocations is unchanged.
* **one device readback per tree on a device backend** — results are assembled host-side, so a
  `pop_size` pass costs `pop_size` device invocations and `pop_size` synchronizations.
* **no cross-tree batching** — duplicate programs hit the cache but still run one by one, and ETL has
  no supported multi-program / "super graph" execution.

Historical floor: the deleted compiled extension ran its CUDA kernels over the **whole population** —
one dispatch sequence per `evaluate` call, a flat ~0.6–1.6 ms warm cost that barely reacted to
`pop_size` (`test/benchmark/SUMMARY.md` §3). The dense path is the closest ETL equivalent of that
shape: **one executable, whole population, one readback per call**, compile amortized over every call
of the same shape.

Measured consequence (§6): on IREE the dense path wins the **cold** phase by orders of magnitude (one
13.9 s compile at `L=32` / 31.6 s at the default cap `L=64` for the whole population against the 13–28
per-tree compiles of P=16–32, worth 4.6–48.6 s in the grid cells), while its **warm** win is modest
wherever both sides were measured directly (1.3–3.2× at N=64/1024 and 1.18× *slower* at N=10000, P=16).
On the numpy reference interpreter it loses at every size (0.006–0.91× of the specialized warm
throughput), which is why the default policy never selects it there.

---

## 2. Scheme

### 2.1 Host preprocessing (`prepare_dense`, bulk-vectorized numpy)

No per-tree host walk and no `decode_population`: the three `(pop_size, max_tree_len)` tensors are
read to host once per call (`_as_numpy`) and turned into `(L, P)` **runtime input columns**, with
`L = min(EVOGP_DENSE_MAX_TREE_LEN (64), max_tree_len)`:

| column | dtype | meaning |
| --- | --- | --- |
| `opcode` | int32 | `0` VAR, `1` CONST, `2` UFUNC, `3` BFUNC, `4` TFUNC (and any unknown base type), `5` inactive/padding |
| `func_id` | int32 | function id, truncated toward zero, clamped to `[0, 29]` |
| `const_value` | float32 | the constant of a CONST node |
| `var_index` | int32 | variable index of a VAR node (truncated, range-checked) |
| `child1/2/3` | int32 | prefix child positions (clamped to `[0, L)`) |

Validation is a vectorized **superset** of the specialized decoder's structural pass (subtree-size
consistency, first child at `i+1`, `c2 = i+1+s[c1]`, `c3 = c2+s[c2]`, children inside the node's own
subtree, subtree inside the declared tree length, leaf `subtree_size == 1`, finite/in-range variable
indices and function ids, no OUT_NODE tag). Anything doubtful raises `DenseUnsupported` rather than
reproducing the decoder's `ValueError` messages — the fallback path owns that contract.

The same pass derives the population's **used function ids** (`DensePrep.used_func_ids`): the union over
the decoded function nodes, canonical (sorted, unique), restricted to the candidate range `1…28`, empty
for a population without function nodes. It is not a runtime column — it is the static set the graph's
candidate chain is built from (§9 item 1).

Padding is inert by construction: positions `i >= tree_len[p]` carry `opcode = 5` (every mask pred
false → value `0.0`) with clamped child indices, so junk in the unused row tail can never influence a
real value (pinned by the padding-invariance families of the batteries).

### 2.2 One graph, trace-time unrolled

`_dense_step` (module-level `@etl.defn` — its `id()` is part of the cache key) traces, through
`_dense_body`, a Python `for i in range(L - 1, -1, -1)` unroll (reverse prefix order: children always
have a larger index than their parent). Per position it reshapes the seven `(L, P)` columns' row `i`
into `(P, 1)` symbolic tensors (`_col`), reads the three child values from the value buffer with the
**flatten trick** (one `gather` over a flat view of the `(rows, P, N)` buffer, index
`off * P * N + pn_off`, both clamped — `_read_buffer`), and computes the **candidate chain** for the
population's used function ids, walked in **descending** id order (the *same* private emitters the
specialized engine uses — `etl_engine._emit_unary` / `_emit_binary` / `_emit_ternary`, parity by
construction; `EVOGP_DENSE_PRUNE=0` walks `FULL_FUNC_IDS` = 28…1 and reproduces the unpruned stream
byte-for-byte — §9 item 1), selecting the node's own value with `select(pred, …)` and folding in the
ternary / leaf cases with broadcast preds.
`x` is read through a second flatten-gather (`x_flat` + `x_base`), where `x_base` selects per-tree
rows (`per_row=True`, graph `N == 1`) or all data rows (`per_row=False`).

The graph's shape is per **chunk**, not per call: `cfg = (L, P_chunk, N_chunk, input_len, per_row,
update_mode, used_func_ids)` is a *static* argument (it specializes the graph and is part of the cache
key), so runtime tensors of a given shape never trigger a recompile while a different used-id set does
(§9 item 1); the loss variant appends its static `use_mse` flag at index 7.

### 2.3 Value buffer & write modes (`EVOGP_DENSE_UPDATE_MODE`)

ETL has no in-place mutation primitive that is safe on every backend, so the `(rows, P, N)` value
buffer is emulated; the strategy is a compile-time choice:

* **`grow`** (default) — the buffer starts as the row computed at `i = L-1` and is *prepended* to with
  `concatenate([row, buf], axis=0)`; reads use the offset `child - (i+1)`, clamped. Row `L-1` is
  provably leaf-only for every validated tree (a function node there would need a child outside the
  row), so its three gathers are replaced by a constant. Costs `L(L+1)/2 · P·N` copied elements instead
  of `L²·P·N`, needs no initial-buffer constant and uses only backend-agnostic ops.
* **`concat`** — a full `(L, P, N)` zero buffer (an embedded graph constant) written with
  `concatenate([head, row, tail])` over static `slice` pieces: the IREE-safe "select-mask" update
  without a boolean mask tensor; reads are plain `child` indices.
* **`scatter`** — a full zero buffer written with `scatter(buf, [i], row, axis=0)`. **numpy only by
  default**: on iree-cuda ETL's `scatter` mutates its input tensor in place, so a retained buffer would
  be re-written by later invokes; without `EVOGP_DENSE_ALLOW_UNSAFE_SCATTER=1` the mode raises
  `DenseUnsupported` on iree (→ fallback).

### 2.4 Chunking and peak memory

One compile must serve the whole call, so the graph's `P`/`N` are fixed **chunk** sizes and the last
chunk is **padded** (`_pad_rows` repeats the chunk's first row; the padded rows are sliced off the
assembled result). The documented model (`dense_peak_bytes`) is

    peak_bytes(L, P, N, mode) = 4 * (buf_factor*L*P*N + (LIVE + 8)*P*N + 4*L*P)

with `buf_factor = 2` for `grow` (one extra copy of the concatenated buffer is live) and `3`
otherwise, and `LIVE = 40` live `(P, N)` SSA candidates (28 function candidates + 3 gathered children
+ leaf/misc). `_chunk_shape` shaves **`N` first, then `P`** until the model fits
`EVOGP_DENSE_BUDGET_MB` (default 2048 MiB), so one dense call is `p_chunks * n_chunks` executes of ONE
executable; `plan_dense` / `describe_plan` expose the realized shape, and `dense_bench.py` prints it
next to every measurement.

The model is a *design* bound, not an RSS predictor: the bench's measured host-RSS delta reaches
**38.7 kB per `pop × row` element** at small `N` (numpy, L=32, grow) — up to ~85× the model's ~456
B/element at that shape, and ≈0 whenever the allocator already held the peak
(`dense_bench_results.json` `dense/memory/host_bytes_per_pop_row`). Treat `EVOGP_DENSE_BUDGET_MB` as a
shape-control knob and watch host RSS on big grids.

### 2.5 Cache

`dense_engine.DENSE_CACHE` **is** `etl_engine.CACHE` (one shared `etl.CompileCache`, default maxsize
4096 via `EVOGP_ETL_CACHE_SIZE`, process-local, in-memory only). The key is the full spec tuple — the
seven runtime `TensorSpec`s (shape + dtype) plus the static `cfg` — combined with backend / device /
options, so one dense compile shows up as ONE miss in `etl_engine.cache_stats()` (against `pop_size`
misses on the specialized path) and a later call of the same shape **and the same used-id set** is a
pure hit (that set is a static `cfg` entry — §2.2 / §9 item 1).

---

## 3. Integration, switches and preserved contracts

Five call sites try dense first and fall back **only** on `DenseUnsupported`: `Forest.forward`
(`per_row=True`, `evaluate_dense`), `Forest.batch_forward` (`per_row=False`, `evaluate_dense`),
`Forest.SR_fitness` (`per_row=False`, `sr_fitness_dense`), `Tree.forward` (1-D `x` → `per_row=True`,
2-D → `False`, `evaluate_dense`) and `Tree.SR_fitness` (`per_row=False`, `sr_fitness_dense`).

The fallback is silent but deliberate: the specialized path re-runs the very same call and owns the
error contract for malformed trees; every other exception propagates. Result contracts are identical
either way — fresh, contiguous, `requires_grad=False` float32 on `x`'s device unless `out_device`.
**One decline warns**: an over-cap tree on a non-numpy backend, where the fallback really is the
expensive outcome (`pop_size` per-structure compiles), emits a deduped once-per-process
`RuntimeWarning` naming the longest tree, the cap, `EVOGP_DENSE_MAX_TREE_LEN` and that compile
consequence (`dense_engine._warn_trace_cap`, pinned by the `policy` stage of
`device_resident_check.py`). Every other decline reason — multi-output, structural doubt, the `auto`
rows ceiling — stays silent, and the dedupe flag is only consumed by a real warning.

**The SR hot path is LOSS-FOLDED.** `Forest.SR_fitness` / `Tree.SR_fitness` compile the *loss* variant
(`_dense_loss_step`, `DensePlan.variant="loss"`, static `use_mse` in `cfg[7]`): the graph returns the
per-tree `sum_N err` (`(p_chunk,)`), `run_loss_plan` assembles and divides by `N` host-side, and no
prediction ever crosses back. `sr_fitness_dense` therefore does **not** route through `evaluate_dense`
(one `inputs` D2H, one `labels` D2H, one `(P,)`-scale readback per chunk, one `torch.from_numpy((P,))`
+ H2D). The value path (`forward` / `batch_forward`) keeps the earlier route: one compiled graph, one
`(P, N)` chunk readback per chunk, one assembled prediction tensor. `use_MSE` is part of the loss
variant's cache key, so MSE and MAE are two graphs per shape signature; the loss value matches the
specialized engine except for float32 reduction order (~1e-7 relative on the fixtures).
Public API compatibility is complete: signatures unchanged,
`use_MSE` / `per_row` / `out_device` honoured, and the five legacy `execute_mode` strings still
accepted and still all mapping to the same engine path.

### Environment knobs (all read per call unless noted)

| variable | default | effect |
| --- | --- | --- |
| `EVOGP_EVAL_MODE` | `auto` | `dense` = always try dense (still falling back on `DenseUnsupported`); `specialized` = never; `auto` = the measured policy. `dynamic` is a first-class ALIAS of `dense` (case/whitespace-insensitive, in `EVAL_MODE_ALIASES`, never gated by the rows rule); unknown → `ValueError` |
| `EVOGP_DENSE_AUTO_MIN_POP` | `16` | `auto`'s `pop_size` threshold on a non-numpy backend (`< 1` → `ValueError`) |
| `EVOGP_DENSE_AUTO_MAX_ROWS` | `2048` | `auto`'s data-row ceiling for `per_row=False` calls: dense is rows-linear while the specialized engine is rows-flat, so more rows decline inside `evaluate_dense` (→ normal `DenseUnsupported` fallback; `< 1` → `ValueError`). `per_row=True` and an explicit `dense` are never gated |
| `EVOGP_DENSE_MAX_TREE_LEN` | `64` | trace cap `L`; a longer tree → `DenseUnsupported` (`<= 0` → `ValueError`) and — on a non-numpy backend only — a once-per-process `RuntimeWarning` (`L = min(cap, row_len)`, so a `max_tree_len=32` population still traces `L=32`) |
| `EVOGP_DENSE_PRUNE` | `1` | emit the per-position candidate chain for the population's used function ids only (`DensePrep.used_func_ids`, §9 item 1) instead of all 28; the set is a static `cfg` entry, i.e. part of the compiled-graph cache key, and a different set recompiles. `0`/`false`/`no`/`off` restores the full `FULL_FUNC_IDS` chain byte-for-byte (the A/B baseline); anything else → `ValueError` |
| `EVOGP_DENSE_BUDGET_MB` | `2048` | chunking budget in MiB (`<= 0` → `ValueError`) |
| `EVOGP_DENSE_UPDATE_MODE` | `grow` | `grow` / `concat` / `scatter` (unknown → `ValueError`; iree-`scatter` → `DenseUnsupported`) |
| `EVOGP_DENSE_ALLOW_UNSAFE_SCATTER` | unset | opt-in for `scatter` on iree |

Everything else is inherited from `etl_engine.py`: `EVOGP_ETL_BACKEND` (default `numpy`),
`EVOGP_ETL_IREE_DEVICE`, `EVOGP_ETL_IREE_OPT_LEVEL` (`O0`/`O1`, default `O1`), `EVOGP_ETL_CACHE_SIZE`
(read at import time) and the `ETL_DISABLE_LOCATIONS=1` tracing speed-up (~4×, recommended for real
runs, not part of the cache key).

### The `auto` policy, exactly

`should_use_dense(pop_size, ..., rows=None)` resolves `EVOGP_EVAL_MODE`: `specialized` → never dense;
`dense` → always attempt dense; `auto` → dense **iff** the resolved backend is not `numpy` **and**
`pop_size >= EVOGP_DENSE_AUTO_MIN_POP` (16) **and** — when the caller knows the data-row count and
passes `rows` — `rows <= EVOGP_DENSE_AUTO_MAX_ROWS` (2048) (`auto_rows_decline`, which never declines an
explicit `dense`/`specialized`). `resolve_backend_name` delegates to `etl_engine._resolve_backend`, so
explicit `backend=` / `iree_device_index=` arguments win over the environment.
The wired call sites pass `pop_size` only, so the rows half of the rule is enforced inside
`evaluate_dense`, where the row count is known — `per_row=False` calls only, which covers
`sr_fitness_dense` because it routes through `evaluate_dense` too. The decline is reported exactly like
every other "do not use dense" answer: `DenseUnsupported`, i.e. a normal silent fallback to the
specialized engine — never an error. Consequences:

* on numpy `auto` **never** uses dense — dense lost at every measured size (0.006–0.91× of the
  specialized warm throughput; the interpreter pays per-op overhead on `(L, P, N)` tensors and the
  graph carries ≈264 ops per traced position, i.e. ≈264·L ops: 2110 at L=8, 9214 at L=32);
* on iree `auto` uses dense from P=16 up *if* the population is dense-supported **and** the call is
  either `per_row=True` (one row per tree — never row-gated) or `per_row=False` with at most 2048 data
  rows; smaller populations, multi-output populations, over-cap trees and larger rows-per-tree
  workloads stay specialized;
* a single `Tree` (`pop_size == 1`) is below the threshold, so under `auto` the `Tree` entry points
  stay specialized on every backend unless `EVOGP_EVAL_MODE=dense` is set explicitly;
* `CombinedForest` sub-forests are single-output populations forwarded through `Forest.forward` /
  `Forest.batch_forward`, so each sub-forest gets the same policy decision as any other population;
* the threshold is a **flat** `pop_size` gate (it ignores `L`): one dense compile costs 13.9 s at
  `L=32` and 31.6 s at the current default cap `L=64`, against ~0.5 s per distinct tree, so the
  compile-term crossover alone is ≈25–35 trees at `L=32` and ≈45–60 at `L=64` (§9.4);
* the rows ceiling is the second half of the measured rule: dense cost grows with the data-row count
  while the specialized engine's warm cost is rows-flat (1.24–1.94 ms/tree from N=64 to N=10000), so the
  warm crossover sits at ≈3.4–6.8k rows (P=16), 7.5–8.4k (P=32) and ~11k (P=64) and dense never wins
  above ~10k rows — the default 2048 is the measured SAFE boundary, ~2× below the tightest crossover
  (`dense_bench_findings.md` §4–5, §6 below).

---

## 4. Validity envelope

* **Single output only.** `output_len != 1` raises `DenseUnsupported` before any graph work, so
  multi-output populations (OUT_NODE tags, the out-node accumulation protocol) always run on the
  specialized engine, which implements that protocol.
* **Trace cap.** The longest declared tree must be `<= L = min(EVOGP_DENSE_MAX_TREE_LEN, row_len)`;
  longer trees fall back (with the one warning above, device backends only). `L` bounds trace time and
  graph size (the candidate chain is emitted `L` times).
  **The default is 64, raised from 32 by the item-5 measurement round** (§9.4): on iree one compile
  costs ≈13.9 s (`L=32`) / ≈31.6 s (`L=64`) / ≈69.0 s (`L=128`) per shape signature, and for a 33–64-node
  population at P=64 the dense route (ONE build, ≈28.9 s cold, ≈11.6 ms warm) beats the silent fallback
  it used to take (64 builds, ≈48.8 s cold, ≈110.3 ms warm; break-even ≈45–60 distinct structures).
  At P=16 the same population trades 4.1× cold for 4.9× warm. `L=128` was rejected: it would tax every
  `row_len >= 128` population — the shipped examples are 128/256/512 — with a ≈69 s compile.
  Two consequences worth knowing before tuning this knob: (a) a population with `row_len >= 64` now
  traces `L=64` **even when its realized trees are short** (the cap is a `row_len`-derived bound, not a
  realized-length one), so a hot short-tree shape can pin `EVOGP_DENSE_MAX_TREE_LEN=32`; (b) numpy under
  the default `auto` is unaffected (dense is never auto-selected there), and the suite's shapes (16/32
  wide rows) keep tracing their old `L`.
* **Both backends, one policy.** `numpy` = correct but slow by design; `iree` = the intended
  production backend for the dense path (CUDA HAL only, `O0`/`O1`).
* **Shapes.** `per_row=True` needs `x.shape[0] == pop_size` and returns `(pop_size, output_len)`;
  `per_row=False` returns `(pop_size, N, output_len)`. Wrong `x` ndim / column count / label shape →
  `ValueError` (the same checks as the specialized engine); non-tensor inputs → `TypeError`.
* **Empty workloads** are rejected (`DenseUnsupported` for an empty population, `ValueError` for zero
  SR rows).
* **The dense path never mutates its inputs**: `x`, the three population tensors and the labels are
  only read, and the write modes operate on in-graph buffers.

---

## 5. Numerics and tolerances

Both paths call the **same private emitters**, so a dense value is bit-exact unless the specialized
engine resolved something at *decode* time that the dense graph resolves at *run* time:

* **numpy: bit-exact.** Zero non-bit-exact finite values across the whole adversarial battery
  (192/192 checks), fold-vs-emit search included; the only bit differences ever seen are NaN payload
  bits with identical NaN masks.
* **iree: fold-vs-emit ≈1 ulp.** The specialized engine folds constant subtrees host-side with numpy
  f64 transcendentals; the dense graph emits the same chain and IREE's f64 libm is not bit-identical to
  numpy's, so ~24 elements per const-heavy population differ by 1 ulp (`max_rel` 6e-8…3.3e-7).
  Element-wise dense == specialized wherever the specialized engine *emits* the op, so this is the
  engine's fold-vs-emit difference, not a dense-path defect — inside the 1e-4 tolerance, reported and
  not chased.
* **iree shared-path deviations** (both paths; iree-compile's own lowering of the verbatim StableHLO,
  not dense-specific): the emitted f64 `power` drops the C99 `pow(x, ±0) = 1` case when the exponent is
  a runtime value (`pow(0,0)` / `pow(NaN,0)` → NaN), and the emitted f32 `sqrt` (`SQRT`/`LOOSE_SQRT`)
  is 1 ulp low for some inputs (e.g. `0.47954902f`). Both vanish on numpy.
* **Comparison policy of every dense battery here**: NaN masks and ±inf signs must match exactly, the
  sign of every zero must match, finite values must agree within `rtol = atol = 1e-4`; bit-exactness is
  the *goal* (reported per element), not the pass criterion — except in
  `dense_adversarial_check.py --strict-bits`, which fails on any non-bit-exact finite value.
* **What the pytest suite asserts**: the independent plain-torch oracle
  (`src/evogp/tests/reference_eval.py`) through `test_equivalence.py`, the combined-multi-output
  forward contract and the wiring/API-compat tests — all green with dense forced
  (`EVOGP_EVAL_MODE=dense`), see §8.---

## 6. Measurements

Sources: the grid in `dense_bench_results.json` (version 1, seed 0, `L=32`, reference population with
realized tree lengths median 17 / max 31, `EVOGP_DENSE_UPDATE_MODE=grow`, GPU 4 RTX A6000,
`EVOGP_ETL_BACKEND=iree` / default numpy) and the **firm-up round** in `dense_bench_findings.md` §1–4,
which root-caused a ~3× inflation in the grid's specialized-iree model (per-call overhead charged once
per tree instead of once per population), took direct full-population iree measurements from P=16 to
P=200 and re-derived the ratios. The numbers quoted here are the firm-up numbers.

**Provenance trap:** the grid's per-cell `ratio_warm_spec_over_dense` field still carries the
pre-firm-up ratio wherever the specialized value was replaced (P=64/N=64: field 16.6× vs re-derived
5.0×; P=200/N=64: 77.9× vs 31.8×; P=1000/N=64: 163.8× vs 61.7×). Re-derive ratios from
`dense.warm_ms` / `specialized.warm_ms` / their `status` + `source` fields instead of trusting the
ratio field or the `measured` label alone.

### iree-cuda: what was measured directly

Same-process same-forest A/B (`dense_bench.py --stage probe` — the strongest evidence, since both sides
pay their per-call overhead in the same process):

| config | specialized warm | dense warm | dense vs specialized |
| --- | --- | --- | --- |
| P=16, N=64 | 22.64 ms | 15.24 ms | **1.49× faster** |
| P=16, N=1024 | 22.58 ms | 17.38 ms | **1.30× faster** |
| P=16, N=10000 | 25.12 ms | 29.51 ms | **1.18× slower** |

Cross-process pairs (direct dense vs direct specialized, one process per population):

| config | specialized warm | dense warm | dense vs specialized |
| --- | --- | --- | --- |
| P=16, N=64 | 27.41 ms | 8.86 ms | 3.09× faster |
| P=16, N=1024 | 21.86 ms | 15.76 ms | 1.39× faster |
| P=16, N=10000 | 26.97 ms | 31.28 ms | 0.86× (1.16× slower) |
| P=32, N=64 | 43.90 ms | 13.89 ms | 3.16× faster |
| P=32, N=1024 | 55.33 ms | 18.52 ms | 2.99× faster |

**Cold phase:** dense pays ONE 13–17 s compile for the whole population against the 13–28 per-tree
compiles of P=16–32 (4.6–48.6 s in the grid cells, ~0.5 s each: `dense_bench_findings.md` §2) — that
part of the win is orders of magnitude, and the dense compile is amortized over every later call of the
same shape signature.

### The "≥10× warm speedup on iree-cuda" headline is NOT supported

Where **both** sides are direct, dense is 1.3–3.2× faster at N=64/1024 and 1.18× *slower* at N=10000
(P=16); the old 16.6× at P=64/N=64 came from the inflated 3× model. The corrected P=64/N=64 value is
≈5.0× (79.50 ms recovered direct specialized / 15.96 ms dense) or ≈6.2× (anchored 1.556 ms/tree × 64 =
99.6 ms) — and even that uses an earlier-grid dense cell. ≥10× survives only as anchored extrapolation
at large populations, e.g. P=200/N=64 336.9 ms / 10.60 ms ≈ 31.8× — a pair assembled from two different
rounds/processes (dense earlier grid, specialized direct), never measured as one A/B. The honest
summary: dense wins the cold phase by orders of magnitude and the warm phase modestly at small N;
large-pop warm wins require many distinct structures (P≥200), which is plausible from per-tree scaling
but unverified.

### iree-cuda grid cells (P ≥ 64; dense cells are earlier-grid cells, this round did not re-verify them)

Warm medians; `spec` provenance per cell (`direct` = measured full-population run this round,
`anchored` = anchored-law extrapolation of it); ratio re-derived as `spec warm / dense warm`.

| P | N | dense cold (1 miss) | dense warm | spec warm | spec source | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64 | 13.03 s | 15.96 ms | 79.50 ms | direct (recovered) | 5.0× |
| 64 | 1024 | 16.94 s | 20.17 ms | 90.00 ms | direct | 4.5× |
| 64 | 10000 | 13.68 s | 70.58 ms | 107.90 ms | anchored | 1.5× |
| 200 | 64 | 16.03 s | 10.60 ms | 336.90 ms | direct | 31.8× |
| 200 | 1024 | 16.50 s | 34.59 ms | 284.40 ms | anchored | 8.2× |
| 200 | 10000 | 12.94 s | 302.74 ms | 388.48 ms | direct | 1.3× |
| 1000 | 64 | 16.61 s | 25.21 ms | 1556.0 ms | anchored (793 trees) | 61.7× |
| 1000 | 1024 | 13.81 s | 119.73 ms | 1422.0 ms | anchored | 11.9× |
| 1000 | 10000 | 14.12 s | 1334.72 ms | 1686.0 ms | anchored | 1.3× |

Two rows pair a direct specialized run with an earlier-grid dense run (P=64/N=64 → 5.0×, P=200/N=64 →
31.8×). They are measured, but in *different* processes and rounds whose repeat spread is 30–50%, and
their dense side was not re-measured with the current harness — that is the boundary of the direct
evidence, not a re-verified ≥10× claim.

P=1000's specialized side was never run (793 distinct compiles ≈ 400 s plus the etl/iree teardown
instability), so those ratios are extrapolations; P=32/N=10000 is `skipped_device_memory` (device-memory
guard, no dense cell). The dense cold column is a one-time cost per shape signature, the specialized
compile column is per population and grows with the number of distinct structures.

### numpy (default backend): dense vs specialized, both measured

Warm medians as `dense / spec` (ms), `P` rows × `N` columns:

| P \ N | 64 | 1024 | 10000 |
| --- | --- | --- | --- |
| 64 | 105.1 / 31.5 | 647.6 / 35.0 | 5365.3 / 34.4 |
| 200 | 160.8 / 102.4 | 3704.4 / 105.6 | 14384.5 / 116.9 |
| 1000 | 548.1 / 500.0 | 8629.0 / 510.0 | 77473.0 / 555.5 |

Numpy dense is element-bound (0.008–0.026 ms per `pop × row` element across the grid) and loses at
every measured size — exactly what `auto` encodes. Numpy specialized cold is 260/886/4484 ms at
P=64/200/1000; the four largest cells (P=64/200/1000 at N=10000 and P=1000 at N=1024) were timed with
a single repetition (`warmup=0`) because they are slow. These cells were not touched by the firm-up
round (they were already direct).

### How the remaining cells were obtained (methodology)

One compile per tree is ~0.5 s and a full-population cold run at `P >= 64` costs 55–793 compiles, so
the grid's specialized side used either direct full-population runs (P=16…64 and P=200 at N=64/10000) or
a **law** fitted to them:

    warm_ms(P)    ~= decode_ms(P) + P * per_tree_warm(N)     # per_tree: median of the direct cells
    compile_ms(P) ~= #distinct programs * ~505.8 ms          # median of 11 direct cells

The earlier single-tree `_sample_model` (one tree per `Forest.SR_fitness` call) is kept in the JSON as
`model_warm_ms_raw_sample` for audit: it **over-stated** per-tree warm by 2.2–3.0× because every
single-tree call re-pays the per-call overhead (input staging, upload, readback) that a population pays
once — decomposed by `--stage probe` into a 1.4–3.1 ms per-call intercept plus a 1.5–1.9 ms per-tree
slope (`dense_bench_findings.md` §1). Scaling laws from the direct cells (`dense_bench_findings.md` §4):
specialized is pop-linear and rows-flat (1.24–1.94 ms/tree, independent of N up to 10000), dense is
pop- and rows-linear (P=16 slope ≈2.2e-3 ms/row over N 64→1024, 1.35e-3 over 1024→10000), which puts the
warm crossover at ≈3.4–6.8k rows (P=16), 7.5–8.4k (P=32) and ~11k (P=64) — dense never wins above
roughly 10k rows. That crossover is what `EVOGP_DENSE_AUTO_MAX_ROWS` (2048) encodes (§3, §9).

---
## 7. ETL / IREE constraints encountered (and the work-arounds)

* **No runtime tensor slicing.** The runtime `etl.Tensor` has no `__getitem__` (only graph-time
  `SymbolicTensor` slicing, or a static-index `getitem` that would bake the index into the cache key),
  so a device-resident `(pop, input_len)` block cannot be split per tree at run time. The dense path
  sidesteps this by feeding the *whole* block and doing the per-tree row selection **in graph**
  (`x_base` + flatten-gather); the specialized `per_row=True` path must upload one row per tree instead.
* **No multi-program / super-graph execution** (`etl.jit` is function-entry caching for one program at
  a time, not multi-program execution), and **no ETL GPU backend without per-graph AOT
  compilation** (numpy = CPU interpreter, iree/xla = compile per graph, tvm = CPU only). Hence "one
  graph for the whole population" is the only lever, and the compile budget moves from per tree to per
  shape.
* **`scatter` mutates its input on iree** → `grow` (default) and `concat`, plus the explicit
  `EVOGP_DENSE_ALLOW_UNSAFE_SCATTER` opt-in for experiments.
* **Executable leak (limit corrected by the firm-up round).** The etl/iree binding leaks device buffers
  per live executable, and the recorded failure mode is a **teardown-only** crash, not a limit on how
  many executables a run may compile: a P=64 session with 55 builds wrote its correct JSON and only then
  hit the nanobind teardown traceback, and the P=200 session with 170 builds completed cleanly (a
  single-family 338-row `dense_adversarial_check` session still SIGSEGVs outright). Protocol for iree
  work: keep sessions short, split batteries with `--only` / one family per fresh process, and use
  `etl_engine.clear_cache()` / `cache_stats()` between stages.
* **Process-local cache.** Nothing persists across processes: a new process re-pays one dense compile
  per shape signature. Long runs of a stable shape amortize it to nothing; short scripts and unit tests
  do not — which is why the suite runs on numpy and `auto` keeps dense off there.
* **Per-call overheads inherited from `etl_engine`.** `build_dense` re-derives the cache key on every
  call (hit or miss) and `etl.run` re-validates the statics — the dense path pays this once per chunk
  instead of once per tree. Both sides are cheap ETL-side now: the multi-slot content-fingerprint memo
  in `CompileCache` turns a repeat visit into a ~17–25 µs key replay (~32 µs per call in a steady-state
  K=50 cycle of 60-token programs, against ~660 µs before) and the per-executable run-path
  static-validation fast path cut the per-call static overhead to ~7–15 µs at 60–170 tokens (~250 µs
  before), holding a whole 60-token `etl.run` at 67–71 µs (303 before) and 170 tokens at 87–96 µs
  (758 before). Landed ETL-side on branch `evogit-agent-T1-A160`, on top of the `main`
  checkout the venv installs (`bench_logs/hotpath/`).
* **Per-chunk upload cost.** Device backends upload one `(P_chunk, …)` column block per chunk
  (7 tensors + `x`) versus one row per tree on the specialized `per_row=True` path. So a chunk should be
  as large as the memory budget allows, and `per_row=False` workloads (the SR-fitness case) are the
  cheapest because the `x` block is shared by every tree.

---

## 8. Validation evidence (recorded in the repo; not re-run for this document)

* `PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 <venv>/bin/python -m pytest src/evogp/tests -q` →
  **308 passed, 4 skipped** (~2 min), zero dense attempts (default `auto` on numpy).
* Dense forced end to end (`EVOGP_EVAL_MODE=dense`): the cheap subset `-k "forward or fitness"` →
  **31 passed / 2 skipped**; the full dense-forced suite is no longer re-runnable at the 64-node cap on
  the 12 GB sandbox (host-memory OOM at ~23%) — the last full run (32-node cap) recorded 510 dense
  attempts, 492 evaluated densely, 18 declines, of which the 3 over-cap trees are now supported.
* `dense_spike_check.py --backend numpy --skip-measure` → **37/37** (the compile-reuse case pins the
  used-id-set granularity; the `EVOGP_DENSE_PRUNE=0` path is pinned separately); the `--full` → 57/57
  run is recorded pre-prune.
* `dense_integration_check.py --stage numpy,fallback` → **119/119**; `--stage iree` → 7/7 (recorded
  pre-prune).
* `dense_adversarial_check.py --backend numpy` → **192/192 checks, 0 non-bit-exact finite values**;
  the staged iree leg re-checked post-prune with one family (`--only binary_var --iree-max-pop 2`,
  `--gpu 0`) → **10/10 checks, rc=0** (the full 19-family iree sweep → 189/189 is recorded pre-prune).
* `device_resident_check.py --stage policy,hotpath,parity,seam,prune` → **104/104** (its prune stage's
  in-process A/B: 420→60 candidate ops; compile 1.24–1.36 s → 0.25–0.37 s; warm SR loss P=64/N=1024
  0.30–0.57 s → 0.044–0.048 s, ranges across loaded-box re-runs); `executable_memo_check.py` → 32/32;
  `runner_check.py` → 48/48 numpy / 32/32 iree.

---

## 9. Future work (in the order the measurements suggest)

1. **Function-id-set pruning — DONE (default ON, `EVOGP_DENSE_PRUNE`).** The per-position candidate chain
   emits ops for the function ids the *population* actually uses (`DensePrep.used_func_ids`) instead of
   the full 28. The set is derived host-side in `prepare_dense` and is a *static* `cfg` entry (`cfg[6]`;
   the loss variant's static `use_mse` is now `cfg[7]`), so it joins the compiled-graph cache key; the
   chain still walks the ids in descending order, so `EVOGP_DENSE_PRUNE=0` (= `FULL_FUNC_IDS`) restores
   the full chain byte-for-byte, and an empty set emits no candidate at all (every position keeps its
   defined zero fallback). Measured (the `prune` stage of `device_resident_check.py`, in-process A/B
   against `EVOGP_DENSE_PRUNE=0`, numpy, medians): candidate ops 420 → 60 (7.0×) on its 4-id fixture,
   compile 1356.7 → 256.0 ms, warm SR loss P=64/N=1024 509.9 → 48.1 ms (10.6×) and P=16/N=1024
   91.6 → 23.3 ms (3.9×); an 18-of-28-id (near-full) population still cuts the candidate chain by up to
   ~1.55×. Pruned and full-chain graphs are **bit-exact** on numpy (value and loss, `per_row` both ways,
   P+N-chunked, empty used set).
   **Granularity:** the executable is a pure function of the shape AND the used-id set, so a population
   whose used set differs from the last compiled one recompiles — it saturates quickly in a GP run
   (the key carries the population-wide union, not per-individual structure), and
   `dense_spike_check.py::case_compile_reuse` pins it (same population → 0 builds, different set → 1).
   **Not measured:** no iree legs this round (the A/B is numpy-only); the churn *below* saturation
   (recompiles per generation while the set is still growing) is unmeasured; and the dense-vs-specialized
   ratios of §3/§6 — including the "numpy never dense" rationale — are full-chain measurements that have
   not been re-run under the pruned default. Record: §10 of `device_resident_findings.md`.
2. **The still-open measurements.** Direct dense at P≥64 and every dense ratio above ~5× rest on the
   earlier grid cells (some `fit(unit-cost)`), so they need a fresh direct pass; P=1000 (793 distinct
   programs) has never been run on the specialized side, its cells are anchored-law extrapolations; and
   P=32/N=10000 is `skipped_device_memory` (the harness's 8 GB device guard), i.e. the largest rows case
   is unmeasured at that pop. Repeat spreads of 30–50% are normal (foreign tenants kept the GPU at
   39–79% utilization), so single numbers are only comparable within one process — a same-process A/B
   at P≥64 is what would settle the warm crossover.
3. **Shape-aware selection.** The rows half of the policy is now implemented
   (`EVOGP_DENSE_AUTO_MAX_ROWS`, 2048 — ~2× below the tightest measured crossover), but
   `EVOGP_DENSE_AUTO_MIN_POP` is still a flat pop gate and the crossover also depends on `L`
   (compile ∝ L); a gate on `(pop_size, L, N)` is the obvious refinement.
4. **`L > 32` economics — measured, and the cap is now 64 (item 5, done).** The trace cap bounds
   compile time and chunk memory (the graph op count is linear in `L`), so the raise was measured before
   it was made: iree compile per shape signature ≈13.9 s (`L=32`) / ≈31.6 s (`L=64`) / ≈69.0 s
   (`L=128`); for a 33–64-node population at P=64 the dense route (ONE build, ≈28.9 s cold, ≈11.6 ms
   warm) beats the specialized fallback (64 builds, ≈48.8 s cold, ≈110.3 ms warm; break-even ≈45–60
   distinct structures), and at P=16 it trades 4.1× cold for 4.9× warm (4.1×/4.9× are per-population
   ratios at the same 33–64-node shape). **Decision: default 64, not 128** — `L=128` would pay a ≈69 s
   compile for every `row_len >= 128` population, and the shipped descriptors are 128/256/512 wide.
   Memory stayed safe across the round (peak RSS ≤ 2.7 GB, no grid cell skipped) and the numpy default
   route is untouched (dense is never auto-selected on numpy). The over-cap fallback is the one decline
   that now warns (once per process, device backends only — §3). Full record:
   `device_resident_findings.md` §9.
   **Still open (documented, NOT implemented):** derive `L` from the REALIZED longest tree rounded up to
   a power of two, clamped at the cap — that would stop a short-tree population with a wide row
   (`row_len >= 64`) from paying `L=64`, at the cost of an extra compile per distinct realized length.
   Unmeasured, so it is not in the code.
5. **Cross-call caching of the prepared columns and `x`.** A generation re-evaluates a *new* population
   on the *same* `x`, so keeping the device-resident `x` block (and the columns' device copies) across
   calls would remove most of the per-chunk upload cost.
6. **Blocked writes** (several rows per buffer update) to cut the `L(L+1)/2 · P·N` concatenation
   traffic, and an in-graph fitness reduction (the loss is currently a torch reduction on the assembled
   host result).

---

## 10. Pointers
- `RUNNER_HOOKS.md` (this directory) — the build-once / run-many benchmark runners
  (`build_dense_runner` / `build_specialized_runner`), i.e. the dense route's compiled forward pass with
  no readback, no re-plan, no key derivation: `.run()` semantics per backend, the build-vs-run split and
  the `runner_check.py` evidence battery.


* `dense_engine.py` — the implementation; its module docstring is the condensed version of this file.
* `ENGINE_SPEC.md` §13 (semantics unchanged) and `CONTEXT.md` (routing + operational notes).
* `dense_spike_check.py` / `dense_integration_check.py` / `dense_adversarial_check.py` — the three
  batteries (engine-level parity + measurements; wiring/fallback; adversarial vs the independent
  oracle); `dense_bench.py` + `dense_bench_results.json` + `dense_bench_findings.md` — the measurement
  harness, its recorded grid and the firm-up round (root cause of the old model's ~3× warm inflation,
  the direct full-population iree measurements and the corrected ratios; check `status`/`source` before
  quoting a cell).
