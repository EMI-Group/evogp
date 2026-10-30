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

* one `evoxc.run` per compiled body for the whole population (one invocation and one readback per body,
  one body on the `W = 1` path);
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
6. **Run** once per compiled body -- one `evoxc.run(executable, <staged x[, labels]>)` (one body per
   population on the `W = 1` path, four at pop 256 under the shipped pool), one readback per body, then
   the host row expansion `_map_rows` back to population order and one fresh float32 tensor.

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
| `EVOGP_BATCH_BAND_SIZE` | `16` | band **size**: `K = ceil(n_functions / size)`, clamped to `1 <= K <= n`, then to the device's binding cap (§9.6); `16` is the measured minimax setting (never more than ~6 % off any of the 9 warm sweep columns, and `K = 15–16` at a pop-256 population's 225–245 functions, i.e. at the `K <= 15` binding cap) |
| `EVOGP_BATCH_BANDS` | unset | explicit band count `K`, the `auto` spelling (evoxc's cost model picks the partition, §9.6), or a flat spelling (`none` / `flat` / `0`), which means an explicit single band (`K = 1`, byte-identical to `=1`); an out-of-range `K` raises `ValueError`, and so does a `K` the device cannot LOAD |
| `EVOGP_BATCH_CHUNK_BANDS` | off | compose the band request with the chunked compile (§9.6); off keeps round 1's rule that a band request forces `W == 1`; on it wins 2.6–2.8× cold but loses 1.6–2.5× warm, so it stays a fixed-population lever |
| `EVOGP_BATCH_SWITCH_OPS` | off | op-switching transform; forces the flat merge (`K = 1`) and is **not** combined with banding |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | `variant` / `occurrence` / `auto`; a pure evoxc performance heuristic, can never change the numbers |
| `EVOGP_BATCH_CACHE_SIZE` | `8` | entries of both front tables; read at **import** time like `EVOGP_EVOXC_CACHE_SIZE` |
| `EVOGP_BATCH_COMPILE_WORKERS` | `4` | compile chunks per plan; `0`/`1` mean today's single-executable path verbatim (see §8.1); a band request forces one body unless `EVOGP_BATCH_CHUNK_BANDS` is on; negative or non-integer raises `ValueError`. Measured trade: the pool makes the cold call 2.64×/2.94×/4.58× cheaper and the warm `sr_fitness` call 1.87×/1.69×/1.46× dearer at `p256-L32` / `p256-L64` / `p256-L128-d7`, so break-even is 3 849 / 5 103 / 18 702 warm calls of ONE population; it is inert (one chunk, no trade) below 32 deduped functions and at the `p64` niche, and `=1` is the documented fixed-multiset escape hatch |

Banding trades dispatches for wasted work: `K` bands means `K` `select` cascades instead of one per
population. `bands="auto"` hands the partition to evoxc's static cost model; the shipped DEFAULT stays the
band-SIZE model at size 16, because `auto` is measured slower than the numeric default (§9.6). Only
unset/empty hands the decision to the band-size model; the flat spellings are an explicit `K = 1` (§9.1).
Every resolved band count is vetted against the device's dispatch-binding cap before the merge: the model
and `auto` are clamped with a recorded note, an explicit `K` that cannot load raises (§9.6).
Pinning the count AT the cap is a same-merge control rather than a different plan at a pop-256
population — the size model is already clamped to `K = 15` there, so `EVOGP_BATCH_BANDS=15` reaches the
identical `band_sizes` and `emitted_ops` — and at `p64`, where the pin is `K = 7` and does change the
plan, it lands inside this box's ~5 % noise floor; no pin beat the shipped size, so 16 stands
(`test/benchmark/evoxc/BATCH.md` §15.3 + `results/batch_opt_round3/SUMMARY.md` §4).

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
This grid is the SINGLE-BODY path (`W = 1`, and a band request forces that body), so its cold column is the pool-1 number, not the shipped chunked default's: at pop 256 x L=32 x il=8 the current window reads 10 725.86 ms cold / 1.9941 ms warm on the pool-1 path against 4 064.48 ms / 3.7250 ms on the shipped default, so read the per-merged-function cold cost quoted below as a pool-1 figure too.

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

`batch_report` / `describe_plan` on the loss graph (at the round-1 band-size model 32): pop 64 / `L=32` / `il=8` → 59 unique functions
(64 rows → 59 after dedup), `bands=2` with sizes `[30, 29]`, `emitted_ops=920`, 672 value classes /
70 shared / 12 stacked constants / 2 output muxes; pop 256 / `L=32` / `il=8` → 222 unique, `bands=7`
`[32,32,32,32,32,31,31]`, 3756 ops; pop 256 / `L=32` / `il=32` → 234 unique, `bands=8` (sizes 29–30),
4176 ops.

### 5.1 Scale envelope / limits

Measured on the same box (loss variant, `per_row=False`, the round-1 band-size model 32,
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
executable, not a bug in this route.
That wall is now a POLICY rather than a load failure (§9.6): the band-size model and `auto` are clamped to
the device's safe band count and an explicit `K` above it raises before the merge.
On the ROCm HAL the clamp is `K <= 15`, so the pop-2048 row above (1443 functions) resolves to 15 bands at
the shipped band size of 16 (91 modelled) and at the round-1 size of 32 (46 modelled), while numpy and
Vulkan are uncapped and keep the modelled count.
Because the model is already clamped at that cap, pinning the count there (`EVOGP_BATCH_BANDS=15`) is the
same merge at a pop-256 population — identical `band_sizes` and `emitted_ops` — while at `p64` the legal
pin is `K = 7` and the changed plan measures inside the ~5 % cross-config noise floor
(`test/benchmark/evoxc/BATCH.md` §15.3 + `results/batch_opt_round3/SUMMARY.md` §4).

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
* **No round-2 pytest coverage** — the band policy (§9.6) is pinned by the dev CLI `batch_band_check.py`
  (`OK: 29/29 passed`, numpy), not by the suite; the rest of the route's coverage belongs to the tests node.
* **A chunked numpy plan must not be the process's first build.** The numpy backend registers its kernel
  table lazily and the registration is not thread-safe, so a chunked plan whose FIRST numpy build happens
  inside the compile pool fails with `BackendError: duplicate kernel registration for op(s): ...`; a warm
  single-body build first (what the battery and the campaign recordings do) leaves it working, and iree is
  unaffected.
* **Large-population merges are compile-heavy**: ~0.05 s per merged function at pop 256 (10.8–12.8 s
  cold), which is the number to watch when raising `EVOGP_BATCH_MIN_POP` or lowering the band size; a very
  fine banding (~one band per function at 200+ functions) is refused or clamped up front (§5.1, §9.6).

## 8. Chunked compile, stage timer and content-addressed plan key

### 8.1 Compile chunking (`EVOGP_BATCH_COMPILE_WORKERS`, `batch_chunks.py`)
The plan's DEDUPED function list is cut into `W` contiguous chunks and each chunk is traced, merged, lowered, compiled and loaded as its own executable; the `W = 1` case is the single-body path verbatim.
`W = resolve_compile_workers()` (`explicit argument > environment > DEFAULT_COMPILE_WORKERS = 4`); `0` and `1` both mean the single-executable path.
The measured break-even of that default is 3 849 / 5 103 / 18 702 warm calls of a single population (`p256-L32` / `p256-L64` / `p256-L128-d7`), bought with 2.64×/2.94×/4.58× cheaper cold for 1.87×/1.69×/1.46× dearer warm, and the pool is inert below the `MIN_CHUNK_FUNCTIONS` chunk threshold — one chunk, no trade, e.g. `p64` with `chunk_count 1` / `chunk_sizes [63]` (`test/benchmark/evoxc/BATCH.md` §15.2 / `results/batch_opt_round3/SUMMARY.md` §3).
`W` is forced to `1` when `W <= 1`, when the deduped function count is below `batch_knobs.MIN_CHUNK_FUNCTIONS` (`32`), when `resolve_switch_ops(switch_ops)` is true, or when a band request is in play (an explicit count, a flat spelling or `auto`) and `EVOGP_BATCH_CHUNK_BANDS` is off -- the band sweep was measured on the `W = 1` path, so a band request implies one body unless the composition is opted into (§9.6).
Otherwise `W = min(W, n_functions // MIN_CHUNK_FUNCTIONS)`, and the partition is `base = n // W` with the remainder handed to the first chunks (`chunk_sizes`).
Chunks carry GLOBAL function indices (`BatchChunk.offset` is the chunk's first index, `BatchChunk.function_rows` are global population rows), so row order and the assembled `(pop_size,)` / `(pop_size, N, out_len)` / `(pop_size, out_len)` results are exactly the single-body plan's.
Per-chunk band counts come from each chunk's own `BandSpec` (`batch_bands.chunk_band_spec`): the plan's request sized to the chunk -- `auto` per chunk (clamped per chunk), the proportional share of an explicit global `K`, or the band-SIZE model re-applied to the chunk (§9.6); `resolve_band_count(chunk_len, bands=None, band_size=..., switch_ops=False)` is the count-only helper, and `BatchPlan.band_count` is `1` for a chunked plan while the truth lives in `BatchChunk.band_count`.
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

The batch route is the public facade plus seven sibling modules under `src/evogp/tree/` (one concern
each); `batch_engine` stays the public facade and re-exports everything, so no caller changes:

| module | owns |
| --- | --- |
| `batch_knobs.py` | the `EVOGP_BATCH_*` constants, the `resolve_*` helpers (incl. `resolve_chunk_bands` and the `auto` spelling of `resolve_bands`), the `value`/`loss` variant rules (`_normalize_variant`) and `BatchUnsupported` |
| `batch_bands.py` | the band/binding legality policy: `DISPATCH_BINDING_LIMITS` / `SAFE_BAND_COUNTS` / `MIN_BAND_SIZE`, `BandPolicy` / `BandSpec`, the precedence chain `plan_band_spec` / `resolve_band_plan`, `clamp_auto_band_count`, `chunk_band_count` / `chunk_band_spec` and the report helpers `merge_with_spec` / `aggregate_band_report`; it takes the caller's merge callable and imports `batch_knobs` + `evoxc_engine`, never `evoxc` and never `batch_engine` |
| `batch_programs.py` | host-side program handling: input/label conversion, per-function graph bodies, dedup, trace, merge, `_plan_host`, and the `BatchPlan` / `BatchExecutable` records |
| `batch_chunks.py` | the compile-side chunk orchestration: `chunk_sizes`, the `lower`/`compile`/`load` thread pool and its per-job stage timers |
| `batch_stages.py` | the per-stage timer surface behind `batch_stage_stats()` |
| `batch_keys.py` | the three front tables (`BATCH_PLAN_MEMO` / `BATCH_CONTENT_MEMO` / `BATCH_EXEC_MEMO`, incl. the `_ContentMemo` subclass), `_default_cache_size`, the invalidation hook `_clear_tables`, `batch_cache_stats()` / `batch_content_stats()`, the memo keys `_executable_key` / `_chunk_executable_key` (via `_body_key`) and the fit guard `_check_executable_matches` / `_spec_shapes` |
| `batch_run.py` | the run path: `_input_block`, `_inputs` staging, `_run` (one `evoxc.run` per body), `_readback`, `_map_rows` and `_to_tensor` |
| `batch_engine.py` | the plan memo (`_plan_entry`), `plan_batch` / `build_batch` / `evaluate_batch` / `sr_fitness_batch` / `describe_plan`, the backend resolution and `batch_stage_stats()` |

`batch_engine` may import any of the seven siblings; none of them imports `batch_engine` back at module level.
`batch_keys` and `batch_run` are read by the facade only, and the facade re-exports their names, so
`evogp.tree.batch_engine.<name>` stays the one import path for callers, tests and the benchmark harness.

### 9.1 Bands and `switch_ops`, in detail

`bands` splits the population into contiguous bands and merges each band separately (one `select`
cascade per band instead of one per population), which trades dispatches for wasted work. The knob is a
band **size** (`EVOGP_BATCH_BAND_SIZE`, default 16): `K = ceil(n_functions / band_size)`, clamped to
`1 <= K <= n_functions`; `K == 1` reduces to the flat merge. `EVOGP_BATCH_BANDS` overrides the model with
an explicit `K`; the flat spellings `none` / `flat` / `0` are an explicit `K = 1` (same plan as
`EVOGP_BATCH_BANDS=1`: `band_sizes == [n_functions]`, `band_strategy == "flat"`), NOT a hand-back to the
band-size model — only unset/empty leaves the model in charge. `bands="auto"` lets evoxc's static cost
model choose the partition instead (§9.6), and every one of those requests is then vetted against the
device's dispatch-binding cap.

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

### 9.6 The band policy: what a device can actually LOAD (`batch_bands.py`)

A banded merge emits ONE exported dispatch per band, and every IREE HAL driver caps the buffer bindings one exported dispatch may use, so a band count that is harmless on one backend is unloadable on another (`iree runtime src/iree/hal/drivers/{hip,cuda}/native_executable.c`, `.../local_task/native_executable.c`).
`batch_bands.py` states that cap as data (`DISPATCH_BINDING_LIMITS` / `SAFE_BAND_COUNTS` / `MIN_BAND_SIZE`):

| device kind | bindings per exported dispatch | safe band count | smallest band |
| --- | --- | --- | --- |
| `cuda` / `rocm` (HIP/CUDA HAL) | 16 | 15 | 8 functions (a GPU band pays a launch plus its binding setup, so a smaller one loses more than it saves) |
| `cpu` (local-task HAL) | 64 | 31 | 1 function (a band AT the binding limit overflows its per-dispatch I/O) |
| `vulkan` | none known | uncapped | 1 |
| `numpy` (the evoxc reference interpreter) | none | uncapped | 1 |

`resolve_band_policy(backend, device)` answers a `BandPolicy(backend, device_kind, max_bands, min_band_size)`: `numpy` (or an empty backend) is uncapped, `iree` takes the kind from the caller's `Device` and otherwise from torch (`_gpu_device_kind`), and an unknown kind is uncapped — "no limit known", never a guess.
`BandPolicy.cap(n_functions)` is the largest loadable band count: `n_functions` when uncapped, else `max(1, min(max_bands, n_functions, max(1, n_functions // min_band_size)))`, i.e. the binding limit, the function count and the per-band size floor, whichever bites first.
So 200 functions cap at 15 bands on a GPU HAL and at 31 on the CPU HAL, while 8 functions cap at ONE band on a GPU.

**Clamp vs refuse.** The band-SIZE model and an `auto` resolution can both exceed the cap (the cost model is CPU-calibrated and knows nothing about bindings), so they are CLAMPED and the clamp is recorded: `plan_band_spec` lowers a modelled count to `policy.cap(n)` before the merge, and `merge_with_spec` lowers an `auto` answer AFTER the merge by re-merging ONCE at the clamped count (both merges stay inside the `merge` stage), keeping the auto merge's `cost_estimate` / `flat_cost_estimate` so a record still shows what the model chose.
The report then carries `band_clamped=True` and a `band_note` naming the requested `K`, the cap, the 16-binding reason and the remedy, e.g.
`the band-size model (EVOGP_BATCH_BAND_SIZE=16) chose 16 bands but iree/rocm can only load 15: a banded merge emits one exported dispatch per band and one exported dispatch may bind at most 16 tensors on the ROCm/HIP HAL (iree runtime src/iree/hal/drivers/hip/native_executable.c), so at most 15 bands can load here; the merge was clamped to 15 bands. Raise EVOGP_BATCH_BAND_SIZE so fewer bands are chosen, or pass an explicit EVOGP_BATCH_BANDS=K with K <= 15 to pick the partition yourself.` (that is the shipped default at a pop-256 per-row plan: 245 deduped functions).
An EXPLICIT `EVOGP_BATCH_BANDS=K` is never clamped: a `K` the device provably cannot load raises `ValueError` naming the variable, the backend/device kind, the real limit, the runtime source and the remedy, instead of letting an iree `BackendError` escape at load — a plausible band count that cannot load is a configuration bug, not a preference.
The message shapes (quoted at 200 functions with `K = 48`, plus the size-floor case):

* binding-capped — `EVOGP_BATCH_BANDS=48 cannot load on the iree/rocm backend: a banded merge emits one exported dispatch per band and one exported dispatch may bind at most 16 tensors on the ROCm/HIP HAL (iree runtime src/iree/hal/drivers/hip/native_executable.c), so the band count must satisfy K <= 15 here. Lower it (EVOGP_BATCH_BANDS=15 or less) or raise EVOGP_BATCH_BAND_SIZE to 8 or more so fewer bands are chosen; the local-task CPU backend's limit is K <= 31 and numpy has no binding cap.`
* the per-band size floor binding first (a tiny population, `K = 2` over 8 functions on a GPU) — `... and a band has to carry at least 8 merged functions, so 8 functions support at most K <= 1 bands here. Lower it (EVOGP_BATCH_BANDS=1 or less), or merge more functions (a larger population, or fewer deduped programs); ...`
* the CPU variant of the same text says `... may bind at most 64 tensors on the local-task CPU HAL (iree runtime src/iree/hal/drivers/local_task/native_executable.c), so the band count must satisfy K <= 31 here ...; the CUDA/ROCm HAL's limit is K <= 15 and numpy has no binding cap.`

numpy and vulkan never clamp and never raise (no binding cap), and `resolve_band_count` stays backend-agnostic: it refuses the `auto` spelling with its own `ValueError` (`EVOGP_BATCH_BANDS='auto' has no band count to resolve here: ...`) because no count exists before the merge — `plan_band_spec` is the policy-aware resolver that carries `auto` that far.

**`EVOGP_BATCH_BANDS=auto`.** The spelling passes `bands="auto"` to `evoxc.batch_functions`, whose static cost model scores the flat partition, the two structural candidates (the fingerprint partition — programs that would share every class — and that partition with its singleton groups coalesced into one trailing band) and the geometric ladder `k` in `(2, 4, 8, 16, 32, 64)` clipped to `2..n-1` plus `k = n`, then reports the choice on the merged graph's `batch_report` (`band_strategy == "auto"`, `band_sizes`, `cost_estimate`, `flat_cost_estimate`).
Flat is candidate 0, so the chosen cost can never exceed the flat cost, and a resolved `K == 1` delegates to the flat emitter (byte-identical to `bands=None`).
The merged compile time is flat in `K` (upstream measurement), so the binding clamp costs no compile time and the model's choice is a WARM-cost choice.
The shipped DEFAULT is the band-SIZE model at size 16 (`EVOGP_BATCH_BAND_SIZE=16`, `K = ceil(n / size)`); `auto` stays an OPT-IN spelling and is measured slower than the numeric default (round-2 verdicts in `test/benchmark/evoxc/results/batch_opt_round2/`).
Pinning the count at the binding cap does not beat that size: at a pop-256 population the model is already clamped to `K = 15`, so `EVOGP_BATCH_BANDS=15` yields the identical `band_sizes` and `emitted_ops`, and at `p64` (where the legal pin is `K = 7` and the plan does change, `[16,16,16,15]` / 1 393 ops -> `[9]×7` / 1 463 ops) the measurement is +0.04 % / +1.3 % / −4.6 % on `sr_fitness` / `batch_forward` / `forward`, i.e. inside the ~5 % noise floor, while the oracle deviation is exactly identical in every config — so the partition is scheduling, never numerics (`test/benchmark/evoxc/BATCH.md` §15.3 + `results/batch_opt_round3/SUMMARY.md` §4).
On the warm sweep of the three cells × three workloads (9 columns, `test/benchmark/evoxc/results/batch_opt_round2/warm/`) band size 16 is the only setting ranked 1st or 2nd in every cell and has the best worst column (pooled geomean 1.0185 / worst 1.061, against 1.0223 / 1.137 for 32 and 1.529 for the flat merge), which is why 16 and not 32 is the numeric default.
`auto` never won a column (0/9 warm): the model is CPU-calibrated and knows nothing about a device's dispatch-binding cap, so on the wide-row (loss / population-batched value) plans it overshoots by 14.2–14.7× at pop 256 (213 / 220 bands for 225 / 232 functions) and is then clamped, while on the per-row `forward` plan it undershoots to 4 bands (pop 256) or the flat merge (pop 64) where it is not clamped at all; the clamp's re-merge costs a further 0.18–1.11 s of host merge per cold call (`test/benchmark/evoxc/results/batch_opt_round2/auto/`).
`EVOGP_BATCH_BANDS=auto` therefore remains a spelling a caller can ask for (and its report shows what the model chose), never the default.
The binding caps still bound the model (`K <= 15` with at least 8 functions per band on CUDA/ROCm, `K <= 31` on the local-task CPU HAL, uncapped on numpy/Vulkan), and the size model is CLAMPED to them with a recorded note — at 225–245 functions band size 16 lands on `K = 15–16`, i.e. exactly on that ceiling.

**Chunked compile composed with banding (`EVOGP_BATCH_CHUNK_BANDS`, default off).** Round 1 measured the band sweep on the single-body path, so by default a band request (an explicit count, a flat spelling or `auto`) forces `W == 1` and `EVOGP_BATCH_COMPILE_WORKERS` is ignored.
Round 2 measured the composition (`test/benchmark/evoxc/results/batch_opt_round2/composition/`) and it stays OFF: `b16` + composition is 0.35–0.40× the single-body arm's COLD call on the two wide-row plans and 0.54× on the per-row one (2.6–2.8× cheaper cold), but 1.61–2.47× the same arm WARM, because four separate upload / run / read-back trios pay per call where one body pays one; the warm loss is the composition itself, not the band policy.
With the knob on, each of the `W` contiguous chunks is merged with the SAME request sized to the chunk: `auto` per chunk (clamped per chunk against the chunk's own function count), an explicit global `K` split proportionally by `chunk_band_count(global_k, chunk_n, n, policy) = max(1, min(chunk_n, policy.cap(chunk_n), ceil(global_k * chunk_n / n)))`, and the band-SIZE model re-applied to the chunk.
Chunks stay contiguous slices of the global function order, so `functions` / `row_index` / `function_rows` and the assembled rows are unchanged and the numbers equal the unchunked plan's (bitwise on numpy, measured).
`switch_ops` still always forces `W == 1`, and the setting is part of BOTH plan memo keys, so flipping it in one process re-plans instead of serving the other plan.
Reproduce a benchmark record with:

```
EVOGP_BATCH_BANDS=auto EVOGP_BATCH_CHUNK_BANDS=1 EVOGP_BATCH_COMPILE_WORKERS=4 \
  PYTHONPATH=/tmp/evoxc_r2:src EVOXC_DISABLE_LOCATIONS=1 <venv>/bin/python -m <harness>
```

and run the same line with `EVOGP_BATCH_CHUNK_BANDS=0` to compare the single-body plan.

**What a record says.** `describe_plan` reports the policy's decision: `bands`, `cost_estimate`, `flat_cost_estimate`, `band_requested` (`flat` / `explicit` / `size` / `auto`), `band_requested_bands` (the `auto` `K` before clamping), `band_clamped`, `band_note`, `band_policy` (`backend` / `device_kind` / `max_bands` / `min_band_size` / `uncapped`), and the per-body `chunk_bands` / `chunk_band_strategies`.
`BatchPlan` carries `band_requested` / `band_clamped` / `band_note` / `band_policy`, every `BatchChunk.report` carries the same five `band_*` keys, and a chunked plan's plan-level keys are aggregated from its chunks (`band_requested_bands` summed before clamping, `band_clamped` true when any chunk was clamped, the clamped chunks' notes joined with `" | "`).
The gate is the dev CLI `batch_band_check.py` (29 assertions: `auto`, the default size model, the clamp, the refusal, the composition and the memo key; numpy only, `__main__`-guarded, NOT pytest-collected): `EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src /tmp/rocvenv/bin/python -m src.evogp.tree.batch_band_check` → `OK: 29/29 passed`.

## 10. Pointers

* `batch_engine.py` (public facade) / `batch_keys.py` (front tables + memo keys) / `batch_run.py` (run
  path) / `batch_programs.py` / `batch_chunks.py` / `batch_stages.py` / `batch_knobs.py` /
  `batch_bands.py` (binding policy) — the implementation; the facade's module docstring carries the API,
  the decline list, the chunked-compile contract and the pointer to this document, and §9's table says
  which module owns what.
* `batch_band_check.py` — the round-2 battery behind §9.6 (dev CLI, not pytest): `auto`, the size model,
  the clamp, the refusal, the chunked composition and the memo key.
* `DENSE_EVAL.md` — the sibling route's design document (same role: scheme, knobs, economics, fallback).
* `CONTEXT.md` (this directory) — where the three engines sit in the package and the route policy.
* `ENGINE_SPEC.md` — semantics of record (op table, evaluation order, SR fitness); the batch route
  changes none of it.
