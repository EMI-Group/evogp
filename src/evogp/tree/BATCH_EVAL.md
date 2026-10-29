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
| `EVOGP_BATCH_BANDS` | unset | explicit band count `K`, or a flat spelling (`none` / `flat` / `0`); out-of-range `K` raises `ValueError` |
| `EVOGP_BATCH_SWITCH_OPS` | off | op-switching transform; forces the flat merge (`K = 1`) and is **not** combined with banding |
| `EVOGP_BATCH_ALIGNMENT` | `variant` | `variant` / `occurrence` / `auto`; a pure evoxc performance heuristic, can never change the numbers |
| `EVOGP_BATCH_CACHE_SIZE` | `8` | entries of both front tables; read at **import** time like `EVOGP_EVOXC_CACHE_SIZE` |

Banding trades dispatches for wasted work: `K` bands means `K` `select` cascades instead of one per
population. `bands="auto"` is deliberately not offered (it is CPU-calibrated, resolves to `K=1` here,
and only costs a static cost-model run).

Front tables (`dense_memo.DenseMemo`, identity/count-keyed, **not** content-hashed): `BATCH_PLAN_MEMO`
maps the three population identity tokens (`data_ptr`/shape/dtype/device/`_version`, so an in-place write
is a miss by construction) plus every resolved plan parameter to a `BatchExecutable`, so an unchanged
population re-decodes, re-traces and re-merges nothing; `BATCH_EXEC_MEMO` keys the *compiled body* on the
actual function multiset + shape + backend/device/options and **re-binds** a hit to the caller's plan
(pop size, row mapping and staged inputs are always the caller's — see the module docstring). Both are
dropped by `evoxc_engine.clear_cache()`; `batch_cache_stats()` reports their 8 counters
(`batch_plan_*`, `batch_exec_*`) and is a separate function because `cache_stats()` /
`dense_cache_stats()` / `artifact_cache_stats()` are pinned by the suite.

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
  cold), which is the number to watch when raising `EVOGP_BATCH_MIN_POP` or lowering the band size.

## 8. Pointers

* `batch_engine.py` — the implementation; its module docstring carries the API and the reuse/re-binding
  contract.
* `DENSE_EVAL.md` — the sibling route's design document (same role: scheme, knobs, economics, fallback).
* `CONTEXT.md` (this directory) — where the three engines sit in the package and the route policy.
* `ENGINE_SPEC.md` — semantics of record (op table, evaluation order, SR fitness); the batch route
  changes none of it.
