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
  measured ~0.34–1.23 s per tree on IREE at the reference shape (L=32, realized tree lengths median
  17) and ~10–25 ms on numpy. A fresh random population is a **cold miss per individual** (the bench's
  pop-1000 reference population held 793 distinct programs).
* **one Python invocation per tree per pass** — a `pop_size` fitness pass is `pop_size` `build_cached`
  look-ups (each re-encoding its static program and re-deriving its sha256 key) plus `pop_size`
  `etl.run` calls with O(#tokens) static re-validation.
* **one device readback per tree on a device backend** — results are assembled host-side, so a
  `pop_size` pass costs `pop_size` device invocations and `pop_size` synchronizations.
* **no cross-tree batching** — duplicate programs hit the cache but still run one by one, and ETL has
  no supported multi-program / "super graph" execution.

Historical floor: the deleted compiled extension ran its CUDA kernels over the **whole population** —
one dispatch sequence per `evaluate` call, a flat ~0.6–1.6 ms warm cost that barely reacted to
`pop_size` (`test/benchmark/SUMMARY.md` §3). The dense path is the closest ETL equivalent of that
shape: **one executable, whole population, one readback per call**, compile amortized over every call
of the same shape.

Measured consequence (§6): on IREE dense wins every measured configuration (2.7×–163.8× warm
speed-up); on the numpy reference interpreter it loses at every size (1.1×–~156× slower), which is why
the default policy never selects it there.

---

## 2. Scheme

### 2.1 Host preprocessing (`prepare_dense`, bulk-vectorized numpy)

No per-tree host walk and no `decode_population`: the three `(pop_size, max_tree_len)` tensors are
read to host once per call (`_as_numpy`) and turned into `(L, P)` **runtime input columns**, with
`L = min(EVOGP_DENSE_MAX_TREE_LEN (32), max_tree_len)`:

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

Padding is inert by construction: positions `i >= tree_len[p]` carry `opcode = 5` (every mask pred
false → value `0.0`) with clamped child indices, so junk in the unused row tail can never influence a
real value (pinned by the padding-invariance families of the batteries).

### 2.2 One graph, trace-time unrolled

`_dense_step` (module-level `@etl.defn` — its `id()` is part of the cache key) traces, through
`_dense_body`, a Python `for i in range(L - 1, -1, -1)` unroll (reverse prefix order: children always
have a larger index than their parent). Per position it reshapes the seven `(L, P)` columns' row `i`
into `(P, 1)` symbolic tensors (`_col`), reads the three child values from the value buffer with the
**flatten trick** (one `gather` over a flat view of the `(rows, P, N)` buffer, index
`off * P * N + pn_off`, both clamped — `_read_buffer`), and computes the **whole candidate chain** for
function ids 28…1 with the *same* private emitters the specialized engine uses
(`etl_engine._emit_unary` / `_emit_binary` / `_emit_ternary` — parity by construction), selecting the
node's own value with `select(pred, …)` and folding in the ternary / leaf cases with broadcast preds.
`x` is read through a second flatten-gather (`x_flat` + `x_base`), where `x_base` selects per-tree
rows (`per_row=True`, graph `N == 1`) or all data rows (`per_row=False`).

The graph's shape is per **chunk**, not per call: `cfg = (L, P_chunk, N_chunk, input_len, per_row,
update_mode)` is a *static* argument (it specializes the graph and is part of the cache key), so
runtime tensors of a given shape never trigger a recompile.

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
misses on the specialized path) and a later call of the same shape is a pure hit.

---

## 3. Integration, switches and preserved contracts

Five call sites try dense first and fall back **only** on `DenseUnsupported`: `Forest.forward`
(`per_row=True`, `evaluate_dense`), `Forest.batch_forward` (`per_row=False`, `evaluate_dense`),
`Forest.SR_fitness` (`per_row=False`, `sr_fitness_dense`), `Tree.forward` (1-D `x` → `per_row=True`,
2-D → `False`, `evaluate_dense`) and `Tree.SR_fitness` (`per_row=False`, `sr_fitness_dense`).

The fallback is silent but deliberate: the specialized path re-runs the very same call and owns the
error contract for malformed trees; every other exception propagates. Result contracts are identical
either way — fresh, contiguous, `requires_grad=False` float32 on `x`'s device unless `out_device`; the
SR loss is a fresh writable `(pop_size,)` positive loss built with the same torch reduction ops as
`etl_engine.sr_fitness` (`diff = label - prediction`, `diff²` or `|diff|`, summed over rows and
outputs, divided by the row count). Public API compatibility is complete: signatures unchanged,
`use_MSE` / `per_row` / `out_device` honoured, and the five legacy `execute_mode` strings still
accepted and still all mapping to the same engine path.

### Environment knobs (all read per call unless noted)

| variable | default | effect |
| --- | --- | --- |
| `EVOGP_EVAL_MODE` | `auto` | `dense` = always try dense (still falling back on `DenseUnsupported`); `specialized` = never; `auto` = the measured policy. Unknown → `ValueError` |
| `EVOGP_DENSE_AUTO_MIN_POP` | `16` | `auto`'s `pop_size` threshold on a non-numpy backend (`< 1` → `ValueError`) |
| `EVOGP_DENSE_MAX_TREE_LEN` | `32` | trace cap `L`; a longer tree → `DenseUnsupported` (`<= 0` → `ValueError`) |
| `EVOGP_DENSE_BUDGET_MB` | `2048` | chunking budget in MiB (`<= 0` → `ValueError`) |
| `EVOGP_DENSE_UPDATE_MODE` | `grow` | `grow` / `concat` / `scatter` (unknown → `ValueError`; iree-`scatter` → `DenseUnsupported`) |
| `EVOGP_DENSE_ALLOW_UNSAFE_SCATTER` | unset | opt-in for `scatter` on iree |

Everything else is inherited from `etl_engine.py`: `EVOGP_ETL_BACKEND` (default `numpy`),
`EVOGP_ETL_IREE_DEVICE`, `EVOGP_ETL_IREE_OPT_LEVEL` (`O0`/`O1`, default `O1`), `EVOGP_ETL_CACHE_SIZE`
(read at import time) and the `ETL_DISABLE_LOCATIONS=1` tracing speed-up (~4×, recommended for real
runs, not part of the cache key).

### The `auto` policy, exactly

`should_use_dense(pop_size)` resolves `EVOGP_EVAL_MODE`: `specialized` → never dense; `dense` → always
attempt dense; `auto` → dense **iff** the resolved backend is not `numpy` **and** `pop_size >= 16`
(`resolve_backend_name` delegates to `etl_engine._resolve_backend`, so explicit `backend=` /
`iree_device_index=` arguments win over the environment). Consequences:

* on numpy `auto` **never** uses dense — dense lost at every measured size (0.006–0.91× of the
  specialized warm throughput; the interpreter pays per-op overhead on `(L, P, N)` tensors and the
  graph carries ≈264 ops per traced position, i.e. ≈264·L ops: 2110 at L=8, 9214 at L=32);
* on iree `auto` uses dense from P=16 up *if* the population is dense-supported; smaller populations,
  multi-output populations and over-cap trees stay specialized;
* a single `Tree` (`pop_size == 1`) is below the threshold, so under `auto` the `Tree` entry points
  stay specialized on every backend unless `EVOGP_EVAL_MODE=dense` is set explicitly;
* `CombinedForest` sub-forests are single-output populations forwarded through `Forest.forward` /
  `Forest.batch_forward`, so each sub-forest gets the same policy decision as any other population;
* the threshold is a **flat** `pop_size` gate (it ignores `L` and `N`): at the L=32 cap a dense compile
  costs 13–17 s against 0.34–1.23 s per distinct tree, so the compile-term crossover alone is ≈11–50
  trees, and at N=10000 the win narrows from ~164× (N=64) to ~3× (see §6, §9).

---

## 4. Validity envelope

* **Single output only.** `output_len != 1` raises `DenseUnsupported` before any graph work, so
  multi-output populations (OUT_NODE tags, the out-node accumulation protocol) always run on the
  specialized engine, which implements that protocol.
* **Trace cap.** The longest declared tree must be `<= L = min(EVOGP_DENSE_MAX_TREE_LEN, row_len)`;
  longer trees fall back. `L` bounds trace time and graph size (the candidate chain is emitted `L`
  times).
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

Source: `dense_bench_results.json` (version 1, recorded at HEAD `65c6617`; the file is unchanged in the
current tree). Seed 0, `L=32`, reference population with realized tree lengths median 17 / max 31,
`EVOGP_DENSE_UPDATE_MODE=grow`, GPU 4 RTX A6000, `EVOGP_ETL_BACKEND=iree` / default numpy. Every dense
cell and every numpy specialized cell is **directly measured** through the public API
(`Forest.SR_fitness` on CUDA tensors); the iree specialized cells are **model-extrapolated** (below).
No number in this document was measured for this round.

### iree-cuda: dense (1 compile per shape) vs specialized (1 compile per distinct tree)

Warm medians; `spec` = the specialized model; `ratio` = `spec_warm / dense_warm`.

| P | N | dense cold (1 miss) | dense warm | spec warm (model) | spec compile (model) | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 64 | 13.03 s | 15.96 ms | 264.2 ms | 67.6 s (55 trees) | 16.6× |
| 64 | 1024 | 16.94 s | 20.17 ms | 218.8 ms | 21.3 s | 10.8× |
| 64 | 10000 | 13.68 s | 70.58 ms | 238.5 ms | 18.6 s | 3.4× |
| 200 | 64 | 16.03 s | 10.60 ms | 826.0 ms | 208.8 s (170 trees) | 77.9× |
| 200 | 1024 | 16.50 s | 34.59 ms | 684.0 ms | 65.9 s | 19.8× |
| 200 | 10000 | 12.94 s | 302.74 ms | 810.0 ms | 58.2 s | 2.7× |
| 1000 | 64 | 16.61 s | 25.21 ms | 4129.5 ms | 974.1 s (793 trees) | 163.8× |
| 1000 | 1024 | 13.81 s | 119.73 ms | 3419.8 ms | 307.5 s | 28.6× |
| 1000 | 10000 | 14.12 s | 1334.72 ms | 4049.5 ms | 271.6 s | 3.0× |

The ≥10× warm win holds up to `N ≈ 1024`; at `N = 10000` dense's per-row slope catches up (warm 1.33 s
at P=1000) and the win narrows to ~3×. The cold column is a one-time cost per shape signature; the
specialized compile column is per population and grows with the number of distinct structures.

### numpy (default backend): dense vs specialized, both measured

Warm medians as `dense / spec` (ms), `P` rows × `N` columns:

| P \ N | 64 | 1024 | 10000 |
| --- | --- | --- | --- |
| 64 | 105.1 / 31.5 | 647.6 / 35.0 | 5365.3 / 34.4 |
| 200 | 160.8 / 102.4 | 3704.4 / 105.6 | 14384.5 / 116.9 |
| 1000 | 548.1 / 500.0 | 8629.0 / 510.0 | 77473.0 / 555.5 |Numpy dense is element-bound (0.008–0.026 ms per `pop × row` element across the grid) and loses at
every measured size — exactly what `auto` encodes. Numpy specialized cold is 260/886/4484 ms at
P=64/200/1000; the four largest cells (P=64/200/1000 at N=10000 and P=1000 at N=1024) were timed with
a single repetition (`warmup=0`) because they are slow.
### How the specialized-iree cells were obtained (model methodology)

One compile per tree is ~1 s and the etl/iree binding segfaults after ~25 live executables per
process, so a full-population cold run at `P >= 64` is infeasible. `dense_bench.py --stage specialized`
therefore samples single-tree populations (12 trees, cache cleared per tree) once per `N` and models
the population as

    warm_ms(P)    ~= decode_ms(P)   + P * median(per-tree warm run ms)
    compile_ms(P) ~= #distinct programs * median(per-tree compile ms)

plus one **REAL** end-to-end cross-check at `P=16` (`--stage real`): 4770.7 ms cold (13 distinct
programs of 16 trees) / 21.5 ms warm, i.e. 1.34 ms of effective per-tree warm cost against the 4.09
ms/tree sampled at pop 64 (`--spec-sample-pop`) — the model **over-states** the specialized warm cost
at small populations, so the small-P ratios above are conservative. Sampled per-tree compile medians:
1.23 s (N=64) / 0.39 s (N=1024) / 0.34 s (N=10000); per-tree warm medians 4.09 / 3.38 / 4.01 ms.

A companion `dense_bench_findings.md` (the ongoing measurement round, written next to
`dense_bench_results.json`) is expected to firm up these model cells and the harness's memory guards.
Until it lands: the dense columns and every numpy column are measured, the iree specialized columns
are the model.

---

## 7. ETL / IREE constraints encountered (and the work-arounds)

* **No runtime tensor slicing.** The runtime `etl.Tensor` has no `__getitem__` (only graph-time
  `SymbolicTensor` slicing, or a static-index `getitem` that would bake the index into the cache key),
  so a device-resident `(pop, input_len)` block cannot be split per tree at run time. The dense path
  sidesteps this by feeding the *whole* block and doing the per-tree row selection **in graph**
  (`x_base` + flatten-gather); the specialized `per_row=True` path must upload one row per tree instead.
* **No multi-program / super-graph execution**, and **no ETL GPU backend without per-graph AOT
  compilation** (numpy = CPU interpreter, iree/xla = compile per graph, tvm = CPU only). Hence "one
  graph for the whole population" is the only lever, and the compile budget moves from per tree to per
  shape.
* **`scatter` mutates its input on iree** → `grow` (default) and `concat`, plus the explicit
  `EVOGP_DENSE_ALLOW_UNSAFE_SCATTER` opt-in for experiments.
* **Executable leak.** The etl/iree binding leaks device buffers per live executable: ~25 compiled
  executables in one process segfault the interpreter (after nanobind "leaked function/HalBuffer"
  warnings). Protocol for iree work: keep sessions short, split batteries with `--only` / one family per
  fresh process, and use `etl_engine.clear_cache()` / `cache_stats()` between stages.
* **Process-local cache.** Nothing persists across processes: a new process re-pays one dense compile
  per shape signature. Long runs of a stable shape amortize it to nothing; short scripts and unit tests
  do not — which is why the suite runs on numpy and `auto` keeps dense off there.
* **Per-call overheads inherited from `etl_engine`.** `build_dense` re-derives the cache key on every
  call (hit or miss) and `etl.run` re-validates the statics by value — the dense path pays this once per
  chunk instead of once per tree.
* **Per-chunk upload cost.** Device backends upload one `(P_chunk, …)` column block per chunk
  (7 tensors + `x`) versus one row per tree on the specialized `per_row=True` path. So a chunk should be
  as large as the memory budget allows, and `per_row=False` workloads (the SR-fitness case) are the
  cheapest because the `x` block is shared by every tree.

---

## 8. Validation evidence (recorded in the repo; not re-run for this document)

* `PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 <venv>/bin/python -m pytest src/evogp/tests -q` →
  **307 passed, 4 skipped** (`-k golden` → 6/6).
* Dense forced end to end (`EVOGP_EVAL_MODE=dense`, same suite) → 307 passed / 4 skipped;
  instrumented: **510 dense attempts, 492 evaluated densely, 18 expected `DenseUnsupported` fallbacks**
  (15 multi-output `output_len` 2/3 populations, 3 trees longer than the 32-node cap).
* `dense_spike_check.py --backend numpy --skip-measure` → 37/37 (`--full --skip-measure` → 57/57).
* `dense_integration_check.py --stage numpy,fallback` → 118/118; `--stage iree` → 7/7 (auto+iree picks
  dense at P=32 with ONE compile per population; iree-vs-numpy-specialized parity clean).
* `dense_adversarial_check.py --backend numpy` → 192/192 checks, 0 non-bit-exact finite values; the iree
  leg (one family per fresh process) → 189/189 checks with a single 1-ulp `deep_chain` element.

---

## 9. Future work (in the order the measurements suggest)

1. **Function-id-set pruning (~3× op-count cut).** Pass the population's used function-id set (or the
   descriptor's `using_funcs`) as a *static* `cfg` entry and emit only those candidates instead of the
   full 28-candidate chain. Recompiles happen only when the id set changes (it saturates quickly in a GP
   run), and it cuts both the dense compile time (≈264·L ops) and the numpy per-op cost — the first
   follow-up if a round needs the iree crossover moved in.
2. **Shape-aware selection.** `EVOGP_DENSE_AUTO_MIN_POP` is a flat pop gate while the measured crossover
   also depends on `L` (compile ∝ L) and `N` (the win shrinks from ~164× at N=64 to ~3× at N=10000); a
   gate on `(pop_size, L, N)`, or at minimum a rows-aware variant, is the obvious refinement.
3. **`L > 32` economics.** The trace cap bounds compile time and chunk memory; raising it needs fresh
   measurements (the graph op count is linear in `L`).
4. **Cross-call caching of the prepared columns and `x`.** A generation re-evaluates a *new* population
   on the *same* `x`, so keeping the device-resident `x` block (and the columns' device copies) across
   calls would remove most of the per-chunk upload cost.
5. **Blocked writes** (several rows per buffer update) to cut the `L(L+1)/2 · P·N` concatenation
   traffic, and an in-graph fitness reduction (the loss is currently a torch reduction on the assembled
   host result).

---

## 10. Pointers

* `dense_engine.py` — the implementation; its module docstring is the condensed version of this file.
* `ENGINE_SPEC.md` §13 (semantics unchanged) and `CONTEXT.md` (routing + operational notes).
* `dense_spike_check.py` / `dense_integration_check.py` / `dense_adversarial_check.py` — the three
  batteries (engine-level parity + measurements; wiring/fallback; adversarial vs the independent
  oracle); `dense_bench.py` + `dense_bench_results.json` (+ the expected `dense_bench_findings.md`) —
  the measurement harness and its recorded grid.
