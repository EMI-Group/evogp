# BATCH_EVAL.md — the merged-graph ("batch") population evaluator

Design document of record for `batch_engine.py`: the evaluator that runs a **whole population through ONE
compiled evoxc executable** whose *leading axis is a function axis* — one function per **deduped** tree
program — instead of one graph per tree structure (the specialized engine, `evoxc_engine.py`) or one
all-ops graph per shape (the dense engine, `dense_engine.py`, which owns the scheme/economics document
`DENSE_EVAL.md`).
**Status: opt-in, integrated.** It is wired into the same five public entry points
(`Tree.forward`, `Tree.SR_fitness`, `Forest.forward`, `Forest.batch_forward`, `Forest.SR_fitness`) but
only ever selected by an explicit `EVOGP_EVAL_MODE=batch`; the default `auto` policy never picks it.
Semantics are unchanged: this document is about *where* the numbers are computed, never *what* they are
— `ENGINE_SPEC.md` stays the semantics of record and the batch route reuses the frozen engine's
`decode_population` / `_walk_tokens` emitters, so parity is by construction.

---

## 1. Why a merged evaluator exists

The specialized engine pays **one compile and one Python invocation per tree structure per pass**: a
`pop_size` fitness pass is `pop_size` `evoxc.run` calls plus `pop_size` readbacks, and a fresh random
population is a cold cache miss per individual (~180 ms of compile at pop 64, documented in
`warm_path_findings.md` / `CONTEXT.md`). `evoxc.batch_functions` can merge N small graphs into one
graph that carries them along a leading *function* axis, which removes both multipliers at once:

* one `evoxc.run` per call for the whole population (one invocation, one readback);
* structurally identical graph positions across functions collapse into a single batched op, so a
  duplicate-rich or shallow population is cheap;
* duplicates and elites (a GP population is 59 / 222 / 234 distinct out of 64 / 256 / 256 rows on the
  measured shapes) merge into one function instead of one graph each.

The price is that the merge — and therefore the whole compile — is **population-specific** (the programs
travel in Python closures), so it does not amortize across GP generations the way the dense graph does.

## 2. Scheme

Per call (`evaluate_batch` / `sr_fitness_batch` → `_plan_entry` → `plan_batch` → `build_batch` → `_run`):

1. **Decode** the three population tensors with the frozen engine (`evoxc_engine.decode_population`):
   per-tree static programs, constant subtrees folded, malformed trees raise `ValueError` naming
   row/node (this route never owns that error contract — the specialized engine does).
2. **Dedup** on structural identity = program equality (`_dedup`). Constant trees are keyed by their
   `const_out` alone (their answer does not depend on `x`); `per_row=True` additionally keys on the
   row's own `x` bytes, because tree `i` must see row `i`. Output: `functions`, `row_index`,
   `function_rows`.
3. **Trace one graph per distinct function** through a Python closure over its static program —
   deliberately a closure rather than the program as a traced static argument, which would make
   `graph.static_values` differ per tree and `batch_functions` raise `TraceError`.
4. **Merge** with `evoxc.batch_functions(graphs, axes=..., out_axes=0, switch_ops=..., bands=...,
   alignment=...)`; `_require_batching_rules` then declines if the merged graph contains an op without a
   batching rule (naming the op instead of surfacing a later `TransformError`).
5. **Build** with the explicit staging sequence `evoxc.lower` → `evoxc.compile` → `evoxc.load`
   (`build_cached` / `evoxc.build` reject a `Graph`). `target_backends` is the GPU HAL kind read from
   torch on iree and omitted on numpy; the compile options carry `opt_level` only.
6. **Run** once: `evoxc.run(executable, <staged x[, labels]>)`, one readback, then the host row
   expansion `_map_rows` back to population order and one fresh float32 tensor.

Two variants exist: `value` (`(pop, N, out_len)`; `per_row` selects the `(pop, input_len)` or
`(N, input_len)` input layout) and `loss` (`sr_fitness`, `per_row=False` only, the SR reduction folded
**in-graph** so only `(n_functions,)` values leave the device).

## 3. Integration, declines and fallback

* `dense_engine.resolve_route` returns `Route(use_dense, backend, fallback_backend, reason, engine)` with
  `ENGINES = ("specialized", "dense", "batch")`; `engine` is `compare=False` and `use_dense ==
  (engine == "dense")`, so assert `route.engine`, never bare route equality.
* `EVOGP_EVAL_MODE=batch` (read per call) is the **only** way to reach this route. `auto` never selects
  it — verified on the shipped policy: `auto`, `dense` and `specialized` build zero batch executables.
  A `Tree` / pop 1 always declines, because it is below the default threshold.
* Every "this route cannot handle it" answer is `BatchUnsupported` (a `RuntimeError`): an empty
  population, a `per_row`/`x` shape doubt, an unknown backend name,
  `pop_size < EVOGP_BATCH_MIN_POP`, an op without a batching rule, and any `TraceError` /
  `TransformError` from tracing or merging (converted, original message appended). Each of the five call
  sites catches exactly that type and continues on `Route.fallback_backend` — `numpy` when the batch
  route chose `iree` without an explicit backend — i.e. the specialized engine, which owns the error
  contract for malformed trees. A backend failure from `lower`/`compile`/`load` is **not** converted:
  there is nothing safer to fall back to.

## 4. Knobs

| Variable | Default | Meaning |
| --- | --- | --- |
| `EVOGP_BATCH_MIN_POP` | `64` | `pop_size` threshold; below it the route declines (measured break-even, §5) |
| `EVOGP_BATCH_BAND_SIZE` | `32` | band **size**: `K = ceil(n_functions / size)`, clamped to `1 <= K <= n` |
| `EVOGP_BATCH_BANDS` | unset | explicit band count `K`, or a flat spelling (`none` / `flat` / `0`), which means an explicit single band (`K = 1`, byte-identical to `=1`); out-of-range `K` raises `ValueError` |
| `EVOGP_BATCH_SWITCH_OPS` | off | op-switching transform; forces the flat merge (`K = 1`) and is **not** combined with banding |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | `variant` / `occurrence` / `auto`; a pure evoxc performance heuristic, can never change the numbers |
| `EVOGP_BATCH_CACHE_SIZE` | `8` | entries of both front tables; read at **import** time like `EVOGP_EVOXC_CACHE_SIZE` |
| `EVOGP_BATCH_COMPILE_WORKERS` | `4` | compile chunks per plan; `0`/`1` mean today's single-executable path verbatim (see §8.1); negative or non-integer raises `ValueError` |

Banding trades dispatches for wasted work: `K` bands means `K` `select` cascades instead of one per
population. `bands="auto"` is deliberately not offered (it is CPU-calibrated, resolves to `K=1` here,
and only costs a static cost-model run). Only unset/empty hands the decision to the band-size model; the
flat spellings are an explicit `K = 1` (§9.1).

Front tables (`batch_keys.py`; `dense_memo.DenseMemo`, identity/count-keyed): `BATCH_PLAN_MEMO` maps the three
population identity tokens (`data_ptr`/shape/dtype/device/`_version`, so an in-place write is a miss by
construction) plus every resolved plan parameter (including `compile_workers`) to a `BatchExecutable`, so
an unchanged population re-decodes, re-traces and re-merges nothing; `BATCH_EXEC_MEMO` keys the
*compiled body* on the actual function multiset + shape + backend/device/options -- plus, when chunked, the
partition `(chunk_count, chunk_sizes, chunk_index)` -- and **re-binds** a hit to the caller's plan (pop
size, row mapping and staged inputs are always the caller's — see `batch_keys.py`'s module docstring).
`BATCH_CONTENT_MEMO` is the third table: a *content*-addressed plan lookup (decoded programs + `x`, §8.3)
that serves a rebuilt-but-equal `Forest` instead of re-planning it (§8).
All three are dropped by `evoxc_engine.clear_cache()`; `batch_cache_stats()` reports only the 8 identity/
executable counters (`batch_plan_*`, `batch_exec_*`), `batch_content_stats()` the 5 content counters, and
both are separate functions because `cache_stats()` / `dense_cache_stats()` / `artifact_cache_stats()` are
pinned by the suite.

## 5. Measurements

This box: AMD MI300X VF, torch 2.13.0+rocm7.1, IREE 3.11, `EVOXC_DISABLE_LOCATIONS=1`; populations from
`Forest.random_generate` with `GenerateDescriptor(max_tree_len=L, input_len=il, using_funcs=["+","-","*",
"/","sin","exp"], max_layer_cnt=5)`, `N=1024` rows, **medians of 5 warm calls, one process per route**.
"cold" is the batch route's first call (trace + merge + lower + compile + load).

`Forest.SR_fitness` (loss variant, `per_row=False`), warm ms — batch+iree / specialized+numpy / dense+iree:

| pop | L | il | batch | specialized | dense | batch cold |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 32 | 8 | 0.76 | 0.63 | 1.13 | 659 ms |
| 4 | 64 | 8 | 0.80 | 0.57 | 1.85 | 664 ms |
| 64 | 32 | 8 | 1.10 | 8.42 | 1.14 | 3302 ms |
| 64 | 32 | 32 | 1.14 | 8.64 | 1.14 | 3612 ms |
| 64 | 64 | 8 | 1.07 | 8.45 | 1.84 | 3171 ms |
| 64 | 64 | 32 | 1.06 | 8.04 | 1.96 | 3394 ms |
| 256 | 32 | 8 | 1.86 | 32.44 | 1.29 | 10 853 ms |
| 256 | 32 | 32 | 2.07 | 30.62 | 1.28 | 12 728 ms |
| 256 | 64 | 8 | 1.86 | 33.44 | 2.34 | 10 793 ms |
| 256 | 64 | 32 | 2.03 | 33.60 | 2.39 | 12 757 ms |

Reference cold costs on the same cells: specialized+numpy 44–48 ms at pop 4, 175–184 ms at pop 64,
629–713 ms at pop 256; dense+iree 4.0–7.8 s at pop 4/64 and 4.1–8.0 s at pop 256.

Reading of the table:

* **Below pop 64 the route loses on both phases** — 659 ms cold against 48 ms specialized (13×) *and*
  0.76 ms warm against 0.63 ms (≈1.2× slower), because one merged compile cannot amortize over four
  functions. That is why `DEFAULT_MIN_POP` is **64** and why the default declines there; nothing below it
  is unsupported, so `EVOGP_BATCH_MIN_POP` (or `resolve_min_pop(value=...)`) still admits it on purpose.
* **From pop 64 up the warm win is real**: 7.6–7.9× over specialized+numpy at pop 64 and 14.8–18× at
  pop 256. Against dense+iree it is shape-dependent and close: dense is 1.2–1.6× faster warm at `L=32`
  (its per-candidate chain is `L`-long) and the batch route is 1.2–1.3× faster at `L=64`.
* **Cold**: batch is cheaper than dense at pop 4/64 (0.66 / 3.2–3.6 s against dense's 4.0–7.8 s) and
  *more* expensive at pop 256 (10.8–12.8 s against 4.1–8.0 s), because a merged compile is linear in the
  deduped function count (~0.05 s per merged function at pop 256). Derived break-even against the
  specialized warm cost: ~3.3 s / 7.3 ms ≈ 450 calls at pop 64 (`L=32`, `il=8`) and
  ~10.9 s / 30.6 ms ≈ 355 calls at pop 256 — and never at pop 4.

**One full GP generation** (`StandardPipeline.step` = evaluate + select/crossover/mutate, steady-state
last of 2 steps, ms):

| workload | batch | dense | specialized |
| --- | --- | --- | --- |
| pop 64, L=32, il=8 | 2858 | 6.1 | 120 |
| pop 256, L=32, il=8 | 11 187 | 7.0 | 532 |
| pop 256, L=64, il=32 | 12 441 | 8.9 | 514 |

The batch graph is rebuilt per **population** (the programs live in Python closures, so a merged graph is
population-specific), so a GP loop pays a full merged compile every generation, while the dense graph is
keyed by `(shape, used-function-id set)` and amortizes across generations. Stated plainly: **the batch
route is not a GP-loop route.** Its win is the lowest per-call host cost and a 7.6–18× warm win over
specialized+numpy at `pop >= 64` — it pays when the SAME population/dataset is evaluated many times
(repeated fitness passes, a frozen population being benchmarked, an analysis sweep), not per generation.

`batch_report` / `describe_plan` on the loss graph: pop 64 / `L=32` / `il=8` → 59 unique functions
(64 rows → 59 after dedup), `bands=2` with sizes `[30, 29]`, `emitted_ops=920`, 672 value classes /
70 shared / 12 stacked constants / 2 output muxes; pop 256 / `L=32` / `il=8` → 222 unique, `bands=7`
`[32,32,32,32,32,31,31]`, 3756 ops; pop 256 / `L=32` / `il=32` → 234 unique, `bands=8` (sizes 29–30),
4176 ops.

### 5.1 Scale envelope / limits

Measured on the same box (loss variant, `per_row=False`, default band-size model 32,
`using_funcs=["+","-","*","/","sin","cos","exp","log","sqrt","max","min"]`, `max_layer_cnt=5`,
`L=32`, `il=8`, `N=128`):

| pop | merged functions | `band_count` | `emitted_ops` | cold (trace+merge+compile+load) |
| --- | --- | --- | --- | --- |
| 256 | 202 | 7 | 3246 | 7.9 s |
| 512 | 387 | 13 | 6568 | 15.8 s |
| 1000 | 731 | 23 | 12 637 | 39.8 s |
| 2048 | 1443 | 46 | 25 435 | 154.2 s |

Every row **loads and runs**, so the route's practical ceiling is compile time, not a hard failure: the
per-function compile cost is ≈0.04 s at pop 256 (7.9 s / 202) and ≈0.11 s at pop 2048 (154 s / 1443) — i.e.
mildly superlinear in the merged function count, so raising `EVOGP_BATCH_MIN_POP` costs accordingly.
Explicit band counts `K = 1` (flat), `8`, `32`, `64` at pop 256 all build and run.

**Very fine banding does not load.** `bands=200` at pop 256 (band size ≈1, one band per function) fails at
`evoxc.load` with `ValueError: ... hip/native_executable.c:223: INVALID_ARGUMENT; exports[N]
binding_flags count 22 exceeds maximum of 16` — a hard IREE ROCm HAL limit on the exported bindings of one
executable, not a bug in this route. The default band SIZE of 32 keeps the band count at
`ceil(n_functions / 32)` (7 at pop 256, 46 at pop 2048), far below that wall; only an explicit
`EVOGP_BATCH_BANDS` near the function count, or a tiny `EVOGP_BATCH_BAND_SIZE` at a few hundred
functions, reaches it.

## 6. Numerics

On the whole grid above, batch vs specialized+numpy **and** batch vs dense+iree: 12/12 cells within
`1e-4` relative on finite values (worst `max_rel` 1.9e-7 relative to the scale of the vector, i.e. 0
values over tolerance), identical NaN/inf masks, and **0** `BatchUnsupported` declines — the route
provably ran (`batch_plan_misses` and `batch_exec_entries` > 0 in every cell). A direct numpy-backend
cross-check through the public API (`EVOGP_EVAL_MODE=batch EVOGP_EVOXC_BACKEND=numpy`, pop 64, `N=1024`)
is bit-exact against the specialized numpy engine: `max_abs_diff` 0.0 over the 59 finite rows with an
identical NaN mask, `batch_exec_misses=1`. The only deviation class ever seen against the specialized
numpy path is the pre-existing iree-vs-numpy backend difference that the dense route shows at the same
index (`DENSE_EVAL.md` §5), i.e. not a property of the merge.

## 7. Known gaps (honest list)

* **Every generation recompiles** — the compile is population-specific (§5). The dense route is the
  right default for a training loop; this route is for repeated evaluation of a stable population.
* **No disk artifacts.** The opt-in `artifact_cache` layer is deliberately **not** wired: a merged graph
  has no static values, so the key would be shape-only and could collide across populations, serving
  wrong numbers that `signature_matches` cannot catch (the shapes agree). Smuggling the programs into the
  key would make every generation a distinct key, i.e. no reuse at all.
* **numpy is debug-only**: merged numpy is 1.1–2.4× *slower* than the specialized numpy path (the
  interpreter pays per-op overhead on the merged graph). Use it to check wiring and numbers, never to
  time anything.
* **`switch_ops` is off**: measured **4× slower** on GP populations (structurally divergent programs
  share nothing) and it changes no op count on the value workload.
* **No pytest coverage yet** — the tests node owns it; the route is currently validated by this
  document's grid plus the smoke described in §6.
* **Large-population merges are compile-heavy**: ~0.05 s per merged function at pop 256 (10.8–12.8 s
  cold), which is the number to watch when raising `EVOGP_BATCH_MIN_POP` or lowering the band size; a very
  fine banding (~one band per function at 200+ functions) does not LOAD on the ROCm HAL (§5.1).

## 8. Chunked compile, stage timer and content-addressed plan key

### 8.1 Compile chunking (`EVOGP_BATCH_COMPILE_WORKERS`, `batch_chunks.py`)
The plan's DEDUPED function list is cut into `W` contiguous chunks and each chunk is traced, merged, lowered, compiled and loaded as its own executable; the `W = 1` case is the single-body path verbatim.
`W = resolve_compile_workers()` (`explicit argument > environment > DEFAULT_COMPILE_WORKERS = 4`); `0` and `1` both mean the single-executable path.
`W` is forced to `1` when `W <= 1`, when the deduped function count is below `batch_knobs.MIN_CHUNK_FUNCTIONS` (`32`), when `resolve_bands(bands) is not None` (any explicit band count, the flat spellings included) or when `resolve_switch_ops(switch_ops)` is true -- those knobs keep their documented per-plan meaning, so ROUND 2's band sweep runs on the `W = 1` path.
Otherwise `W = min(W, n_functions // MIN_CHUNK_FUNCTIONS)`, and the partition is `base = n // W` with the remainder handed to the first chunks (`chunk_sizes`).
Chunks carry GLOBAL function indices (`BatchChunk.offset` is the chunk's first index, `BatchChunk.function_rows` are global population rows), so row order and the assembled `(pop_size,)` / `(pop_size, N, out_len)` / `(pop_size, out_len)` results are exactly the single-body plan's.
Per-chunk band counts come from the band-SIZE model only (`resolve_band_count(chunk_len, bands=None, band_size=..., switch_ops=False)`); `BatchPlan.band_count` is `1` for a chunked plan and the truth lives in `BatchChunk.band_count`.
Trace and merge stay SEQUENTIAL in the calling thread (evoxc's transforms are only proven thread-safe for the compile triple); only `evoxc.lower` / `evoxc.compile` / `evoxc.load` run in a `ThreadPoolExecutor(max_workers=W)` -- the iree compile is a subprocess and releases the GIL.
Pool jobs each own a `batch_stages.StageTimer` and fold it once under the module lock; a chunk failure PROPAGATES (it is a bug, not a `BatchUnsupported` decline) and the first exception is re-raised.
`plan.chunks` holds exactly one `BatchChunk` per compiled body in BOTH plan shapes, so `len(plan.chunks) == describe_plan(plan)["chunk_count"] >= 1` is always the chunk count and `plan.chunks` is always the authoritative read surface.
A single-chunk plan's `chunks[0]` (`index=0`, `offset=0`) mirrors the whole-graph fields object-for-object: `plan.chunks[0].merged is plan.merged`, `plan.chunks[0].report is plan.report` and `plan.chunks[0].specs == plan.specs`.
`plan.merged` / `plan.graphs` / `plan.specs` are populated only when `chunk_count == 1`; a chunked plan sets them empty (`()` / `None`) with an aggregate `plan.report` (`chunk_count`, `chunk_sizes`, `n_functions`, summed `emitted_ops`, `bands`/`band_sizes`/`band_strategy` `None`), and its per-chunk graphs/specs/reports live in `plan.chunks`.
`BatchExecutable.executable` is `executables[0]` for a single-chunk plan and `None` when chunked; every internal read site goes through `executables`/`chunks`, and the build/run paths take the single-body branch exactly when `len(plan.chunks) == 1`.

### 8.2 Stage timer surface (`batch_stages.py`)
Ten stages: `decode`, `dedup`, `trace`, `merge`, `lower`, `compile`, `load`, `upload`, `run`, `gather`.
`batch_engine.batch_stage_stats()` reports exactly 32 keys -- `batch_<stage>_calls` / `_total_ms` / `_max_ms` for each stage plus `batch_compile_wall_ms` (wall clock of the whole parallel compile/load region) and `batch_compile_jobs` (chunk bodies actually built in the call; `0` when every chunk was a memo hit).
`batch_stage_reset()` zeroes all 32; both are thin delegations and are deliberately NOT folded into `batch_cache_stats()`.
The same numbers are recorded per record: `BatchPlan.stage_ms` (`decode` / `dedup` / `trace` / `merge` of the call that built it) and `BatchExecutable.build_stage_ms` (`lower` / `compile` / `load` totals plus `lower#<chunk>` / `compile#<chunk>` / `load#<chunk>`).
Timing is measurement only: it never reorders work, and `gather` covers the readback `.to(_CPU).numpy()` plus `_map_rows` plus the final `_to_tensor`.
Thread safety: pool workers accumulate into their own `StageTimer` and fold once under one module lock; the main thread uses the module-level `stage()`.
Runs stay SEQUENTIAL even when chunked -- one device, one stream, and the readbacks must stay ordered.

### 8.3 Content-addressed plan key (`BATCH_CONTENT_MEMO`)
`BATCH_PLAN_MEMO` is identity-keyed, so a rebuilt `Forest` with identical content used to re-plan from scratch.
The lookup is now two-level: identity tokens first (a repeat call on unchanged tensors still skips even the decode), then, on an identity miss, ONE decode followed by a `blake2b(digest_size=32)` digest over each row's decoded representation (`program` tokens + `const_out`, in population order) plus the `per_row` flag and, when `per_row`, each row's `x` bytes.
The content key also carries every resolved plan parameter (`input_len`, `output_len`, `n_rows`, `per_row`, `variant`, `use_mse`, `band_size`, `bands`, `switch_ops`, `alignment`, `compile_workers`, backend, device) and is versioned by `PROGRAM_DIGEST_FORMAT`.
A content hit is served only after `plan.programs` is verified against the just-decoded programs; a collision or verification failure rebuilds, is counted as a content MISS and increments `batch_content_rejects`, and the refused entry is dropped.
So `batch_content_hits` counts only entries actually SERVED, never a refused one.
The content entry is stored with `pins=()` (no population-specific storage pinned) while the identity entry keeps the call's pins via `dataclasses.replace`; `batch_content_stats()` reports `batch_content_hits` / `_misses` / `_entries` / `_maxsize` / `_rejects`.
The executable key stays ORDER-SENSITIVE on purpose: the function-axis order IS the output order (`row_index` indexes it and bands are contiguous slices of it), so an order-insensitive key could serve a body whose row mapping is wrong.
`_plan_host` decodes only when `pop is None` and `_plan_entry` passes its already-decoded population in, so a rebuild still calls `_plan_host` exactly once and decodes exactly once.

### 8.4 Measured (MI300X, iree, pop 256 x L=32 x i8 x N=1024, sequential single process, box shared with other agents)
Cold `Forest.SR_fitness` wall (fresh process, `clear_cache()` first, 236 deduped functions): `W=1` 10.90 s -> `W=2` 6.68 s -> `W=4` 5.27 s -> `W=8` 4.53 s (7 chunks), i.e. 1.63x / 2.07x / 2.40x faster; the compile region's own wall goes 10.47 s -> 4.75 s (4.75x on the sum of per-chunk lower+compile+load work) because more bodies give the pool more parallel work.
Warm call cost grows with the chunk count: 1.88 ms (`W=1`) / 2.45 ms (`W=2`) / 3.45 ms (`W=4`) / 5.84 ms (`W=8`) -- one extra upload + `evoxc.run` + readback per chunk, which is the price of the cold win; share the staged input across chunks (it is identical for `per_row=False`) before raising the default.
Cold stage split at `W=4` (5.15 s call): compile region 4.75 s (92%), `merge` 225 ms + `trace` 88 ms (6%), `upload`/`run`/`gather` 59 ms (1%), `decode` 3.3 ms + `dedup` 0.1 ms (<0.1%).
Inside the compile region the sum of per-chunk work is 18.2 s (`lower` 3.0 s, `compile` 14.9 s, `load` 0.33 s) compressed into a 4.75 s wall.
Warm stage split at `W=4`: `run` 1.92 ms, `upload` 0.92 ms, `gather` 0.83 ms.
Numerics: on **numpy** the chunked path is bitwise identical to the single-body path for all four variants (value `per_row`, value shared-`x`, loss MSE, loss MAE) -- identical finite bits and identical NaN masks.
On **iree** the loss path was bitwise identical on the checked fixture while the value path differed on 78 of 16128 finite elements (max 2.4e-7 absolute / 1.4e-6 relative, values up to 1.6e25); repeated runs of the SAME partition are bit-identical, so the difference is the documented iree FMA-contraction class reacting to a different band merge, not a defect.

## 9. Contracts moved out of the module docstrings

The batch route is six modules under `src/evogp/tree/` (one concern each); `batch_engine` stays the
public facade and re-exports everything, so no caller changes:

| module | owns |
| --- | --- |
| `batch_knobs.py` | the `EVOGP_BATCH_*` constants, the `resolve_*` helpers, the `value`/`loss` variant rules (`_normalize_variant`) and `BatchUnsupported` |
| `batch_programs.py` | host-side program handling: input/label conversion, per-function graph bodies, dedup, trace, merge, `_plan_host`, and the `BatchPlan` / `BatchExecutable` records |
| `batch_chunks.py` | the compile-side chunk orchestration: `chunk_sizes`, the `lower`/`compile`/`load` thread pool and its per-job stage timers |
| `batch_stages.py` | the per-stage timer surface behind `batch_stage_stats()` |
| `batch_keys.py` | the three front tables (`BATCH_PLAN_MEMO` / `BATCH_CONTENT_MEMO` / `BATCH_EXEC_MEMO`, incl. the `_ContentMemo` subclass), `_default_cache_size`, the invalidation hook `_clear_tables`, `batch_cache_stats()` / `batch_content_stats()`, the memo keys `_executable_key` / `_chunk_executable_key` (via `_body_key`) and the fit guard `_check_executable_matches` / `_spec_shapes` |
| `batch_run.py` | the run path: `_input_block`, `_inputs` staging, `_run` (one `evoxc.run` per body), `_readback`, `_map_rows` and `_to_tensor` |
| `batch_engine.py` | the plan memo (`_plan_entry`), `plan_batch` / `build_batch` / `evaluate_batch` / `sr_fitness_batch` / `describe_plan`, the backend resolution and `batch_stage_stats()` |

`batch_engine` may import the five siblings; none of them imports `batch_engine` back at module level.
`batch_keys` and `batch_run` are read by the facade only, and the facade re-exports their names, so
`evogp.tree.batch_engine.<name>` stays the one import path for callers, tests and the benchmark harness.

### 9.1 Bands and `switch_ops`, in detail

`bands` splits the population into contiguous bands and merges each band separately (one `select`
cascade per band instead of one per population), which trades dispatches for wasted work. The knob is a
band **size** (`EVOGP_BATCH_BAND_SIZE`, default 32): `K = ceil(n_functions / band_size)`, clamped to
`1 <= K <= n_functions`; `K == 1` reduces to the flat merge. `EVOGP_BATCH_BANDS` overrides the model with
an explicit `K`; the flat spellings `none` / `flat` / `0` are an explicit `K = 1` (same plan as
`EVOGP_BATCH_BANDS=1`: `band_sizes == [n_functions]`, `band_strategy == "flat"`), NOT a hand-back to the
band-size model — only unset/empty leaves the model in charge. `bands="auto"` is deliberately NOT offered:
it is CPU-calibrated, resolves to `K=1` on this workload and only costs a static cost-model run.

`switch_ops` (`EVOGP_BATCH_SWITCH_OPS`, default off) makes op-switching positions share one op per variant
plus a per-row `select` cascade. It is a heavier transform for GP populations (structurally divergent
programs) and on the measured workload it changed nothing at all — `report["switched"]` is empty and the
emitted op count and warm time are those of the flat merge — so it stays a knob, off by default. When it
is on, the merge is FORCED to the flat form (`K = 1`): `switch_ops` and banding are not combined, and the
plan records `band_count == 1` whenever `switch_ops` is set.

### 9.2 On-disk artifacts: deliberately NOT wired

`artifact_cache` keys an artifact by `(name, specs, backend, device, options)` where `specs` is the
`build_cached` spec tuple — tensor specs *plus* the static content that specializes the graph. A merged
batch graph has NO static values (the programs live in Python closures), so the tensor specs alone would
collide across every population of a shape and a stale artifact would serve wrong numbers
(`signature_matches` cannot catch it: the shapes agree). The programs could be smuggled into the key as
extra static content, but then every generation's multiset is a distinct key, i.e. no reuse at all. So
this route builds in memory only; a process that repeats a multiset keeps it in `BATCH_EXEC_MEMO` for the
lifetime of the process.

### 9.3 Front tables: why a repeated call is cheap

The tables and the memo keys they are driven by live in `batch_keys.py`.
`BATCH_PLAN_MEMO` maps the three population identity tokens (`dense_memo.token`:
`data_ptr`/shape/dtype/device/`_version`, so an in-place mutation is a miss by construction) plus every
resolved plan parameter to the `BatchExecutable`: a repeated evaluation of an unchanged population does no
re-decode, no re-trace, no re-merge, no re-key and no re-build. The `x`/`labels` conversion is **not**
memoized — `_host_rows` (and, for the loss variant, `_host_labels`) run before the lookup on every call,
because the key carries the converted row count and the label shape check needs the converted rows.
`BATCH_EXEC_MEMO` keys the same executable on the ACTUAL program multiset plus the
shape/backend/device/options, so the same programs in a fresh population object still hit. Both tables are
dropped by `evoxc_engine.clear_cache()` (the engine's single invalidation point, which calls the
`_clear_tables` hook registered there) and counted by
`batch_cache_stats()` — a separate function, because `cache_stats()` (8 keys) and `dense_cache_stats()`
are pinned by the suite.

### 9.4 Reuse across populations (the re-binding contract)

`BATCH_EXEC_MEMO` is keyed by the compiled body only — the merged function multiset plus the variant, the
merge shape, the I/O shape and the backend/device/options (see `_executable_key`) — and deliberately NOT
by the population that first built it, because one compiled merged graph really does serve every
population with that same function multiset: the function axis IS the graph's leading axis, so which
population row maps to which function is the caller's business, not the graph's. That makes the population
size, the row mapping and the staged inputs *per-call* data, so an executable hit is **re-bound**: the
returned `BatchExecutable` pairs the memoized `executable` (and the `backend` / `device` it was built for)
with the plan the CURRENT call built (or hit in `BATCH_PLAN_MEMO`), never with the first population's
`pop_size`, `row_index`, `function_rows` or `programs`.

The rebinding is structural, not incidental: `build_batch` is the ONLY reader of `BATCH_EXEC_MEMO` and it
always constructs a fresh `BatchExecutable` around the caller's plan, so every per-population field on the
running path (`_inputs` staging, `_map_rows` row expansion, the `(pop_size, ...)` / `(pop_size,)` output
shapes and the `pop_size` row check) is read from that plan. `_check_executable_matches` is the cheap
safety net for the one thing the key cannot re-derive: the compiled body's recorded tensor specs (merged
function count, rows, input width, outputs) must fit the caller's plan, and a mismatch is a
`BatchUnsupported` decline — never a silently wrong row mapping.

No variant is excluded from that reuse: `per_row=True` puts the function count in the merged INPUT spec's
leading dimension and `per_row=False` puts it in the OUTPUT spec's, but both are part of the key, so a hit
is reusable in both layouts (and the invariant check above verifies exactly those dimensions).

### 9.5 Who to blame for what (reuse, never re-implement)

Decoding, validation, constant folding and op emission are the frozen engine's:
`evoxc_engine.decode_population` gives the per-tree static programs and
`evoxc_engine._walk_tokens` is the emitter *and* the multi-output protocol. A per-function graph is traced
through a Python closure over its static program — never with the program as a traced static arg, which
would make `graph.static_values` differ per tree and make `batch_functions` raise `TraceError`.

## 10. Pointers

* `batch_engine.py` (public facade) / `batch_keys.py` (front tables + memo keys) / `batch_run.py` (run
  path) / `batch_programs.py` / `batch_chunks.py` / `batch_stages.py` / `batch_knobs.py` — the
  implementation; the facade's module docstring carries the API, the decline list, the chunked-compile
  contract and the pointer to this document, and §9's table says which module owns what.
* `DENSE_EVAL.md` — the sibling route's design document (same role: scheme, knobs, economics, fallback).
* `CONTEXT.md` (this directory) — where the three engines sit in the package and the route policy.
* `ENGINE_SPEC.md` — semantics of record (op table, evaluation order, SR fitness); the batch route
  changes none of it.
