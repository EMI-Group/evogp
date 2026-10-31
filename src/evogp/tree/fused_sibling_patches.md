# Cross-node edits for the fused route (`EVOGP_EVAL_MODE=fused`) — all applied
`FUSED_HANDOFF.md` is the one-page summary of the edits the fused route needed OUTSIDE `src/evogp/tree/`; this file is their record.
Every item below has landed in the tree, verified against the current code, so nothing here is a paste-ready instruction any more.
Re-verified on the device reference box: `pytest src/evogp/tests -q` → 1 308 passed / 7 skipped / 0 failed (both stale assertions, §1's table pin and §7's `auto` pin, have since been fixed), `pytest src/evogp/gp/tests -q` → 309 passed / 36 skipped (345 collected; the 36 are the device-gated `test_gp_stack_device_fused.py` cases, which skip while the native device hand-off is unavailable), `pytest test/benchmark/evoxc -q` → 44 passed, `fused_cache_stats()` has 25 keys (`fused_bound_launches` / `fused_dispatch_declines` included), `EVOGP_FUSED_MIN_POP` is 16 and `MAX_OUT_LEN` is 8.
## 1. The route tables (`src/evogp/tree/dense_engine.py`, pinned by `src/evogp/tests/test_dense_policy.py`)
`EVAL_MODES` is `("dense", "specialized", "batch", "fused", "auto")` and `ENGINES` is `("specialized", "dense", "batch", "fused")`, with `fused` before `auto`, never an alias.
`test_eval_mode_tables_stay_canonical` pins both plus the membership/alias rules, so the table cannot drift.
## 2. The route's pytest net (`src/evogp/tests/test_fused_route.py`)
53 cases that compile and launch the real Triton kernel, lift the session `EVOGP_EVOXC_BACKEND=numpy` pin per test and SKIP on a box without a device, `triton` or a writable `TRITON_CACHE_DIR`.
`import reference_eval` resolves as a sibling of the test file.
## 3. The harness arm registry (`test/benchmark/evoxc/common.py`)
`evoxc-fused-iree` (explicit `backend="iree"`, the unconditional arm) and `evoxc-fused` (no explicit backend, i.e. the policy's own decision) sit after the LAST batch entry, so every earlier arm id keeps its index and `ALL_ARMS` derives them.
`test_arm_registry.py` pins their position, their `eval_mode` and `is_batch_arm(...) is False`.
## 4. The route knobs are route knobs (`test/benchmark/evoxc/common.py`)
`FUSED_ENV_VARS` (`EVOGP_FUSED_MIN_POP`, `BLOCK_ROWS`, `NUM_WARPS`, `MAX_SCRATCH_MB`, `HOST_MEMO`) is folded into `ROUTE_ENV_VARS`, so an exported `EVOGP_FUSED_BLOCK_ROWS=64` can no longer leak into every arm of a round.
## 5. Route reporting (`test/benchmark/evoxc/common.py` + `summarize.py`)
`route_decision` asks the one `dense_engine.resolve_route(...)` for a fused request and stamps `backend` / `engine` / `would_use_fused` / `fused_fallback_backend` / `fused_reason`; `route_text` derives `kind` from `engine` and annotates a decline, so a served call is never rendered from a backend spelling.
## 6. The existence proof (`test/benchmark/evoxc/worker.py`)
`FUSED_STATS_KEYS` (15 of the 25 `fused_cache_stats()` counters) and `FUSED_GAUGE_KEYS` (the 6 gauges, recorded as an AFTER snapshot rather than a delta) feed `fused_served()` (`attempts - declines == calls` and `launches >= calls`), and the per-workload block lands as `record["fused_cache"]` beside `record["engine_cache_final"]`; the raw AFTER snapshot beside it carries all 25 keys, `fused_bound_launches` / `fused_dispatch_declines` included.
An engine name is not proof a call was served; `fused_kernel_launches` is (`fused_calls` counts ATTEMPTS), so a `compile_misses_first_call` of 0 proves nothing on this route — it compiles no evoxc graph.
## 7. The `auto`-policy pin
`src/evogp/tests/test_fused_route.py::test_auto_selects_fused_above_the_floor` (pop 16 / 64 / 256, 3 cases) pins `auto` → `fused` from `EVOGP_FUSED_MIN_POP` up, and `test_auto_keeps_the_pre_fused_policy_below_the_floor_or_without_triton` pins the below-the-floor / no-triton arm as `specialized`.
No pre-policy assertion that `auto` never selects the fused route is left, so the file is 53 passing cases.
## 8. The campaign's gate
`test/benchmark/evoxc/results/t1a1_fused/` is the campaign of record (the `test/` tree, beside `results/t1a1_baseline/`), and its gate is `check.py --results test/benchmark/evoxc/results/t1a1_fused --allow-snapshot --allow-missing --allow-digest-mismatch` → 20 records / 580 checks / 0 failures.
The two allowances stay: a fused `arith`-set `sr_fitness` digest differs by float32 reduction order, and this round never scheduled every arm on every cell.
