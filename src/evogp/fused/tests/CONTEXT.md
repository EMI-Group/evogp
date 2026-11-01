# `src/evogp/fused/tests/` — the pytest net of the fused (Triton) evaluation route
## Intent
This is the regression gate of `evogp.fused`: it pins the route's numerics against the independent plain-torch oracle, its multi-output (`OUT_NODE`) protocol, its result/decline contracts, its warm-call fast paths and its launch-path (`bound` / `jit`) equivalence.
Every case drives `evaluate_fused` / `sr_fitness_fused` DIRECTLY on the legacy triple (`batch_node_value` float32, `batch_node_type` int16, `batch_subtree_size` int16, all `(pop, max_tree_len)`, prefix order) — there is no `Forest`/`Tree` in the way, because the torch-native stack that owned those wrappers is gone and this package is fed by whoever holds the triple (`evogp.gp`'s `FusedDeviceEvaluator` and the battery `evogp.fused.check` are the two callers).
The portfolio is 29 test functions / 45 cases; the full old→new verdict table is at the bottom.
## How to run
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache_fused_lib && mkdir -p "$TRITON_CACHE_DIR"
PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \
  /tmp/rocvenv/bin/python -m pytest src/evogp/fused/tests -q
```
45 passed / 0 skipped / 0 failed in ~1.4 s from a warm Triton disk cache (the module-scoped `_compiled` fixture pays the four single-output specializations once).
`TRITON_CACHE_DIR` must be writable — `$HOME` is read-only in the sandbox, and a first launch dies on it; `conftest.py` points it at a `tmp_path` Triton cache when the ambient one is unusable, and skips the module with the reason when even that fails.
## Device gating
`conftest.py`'s session-autouse `fused_route_guard` skips the whole module (never fails it) when `torch.cuda.is_available()` is false, when `triton` is not importable, or when no writable Triton cache directory can be established — so a CPU-only or triton-less box collects and reports skips instead of errors.
An autouse `fused_knobs` fixture deletes `EVOGP_FUSED_BLOCK_ROWS` / `NUM_WARPS` / `MAX_SCRATCH_MB` / `HOST_MEMO` / `DISPATCH` before every case, so an ambient knob cannot change a measured number (a case that wants a knob sets it itself with `monkeypatch`).
## The oracle, by path
The numeric reference is `test/oracle/reference_eval.py` — DEV tooling, deliberately NOT on `PYTHONPATH` and imported by no library module.
`conftest.py` computes the repository root as `pathlib.Path(__file__).resolve().parents[4]` (`…/src/evogp/fused/tests/conftest.py` → repository root; guarded by an assertion that `<root>/src/evogp` exists) and appends `<root>/test/oracle` to `sys.path`.
The session-scoped `oracle` fixture `pytest.importorskip`s it, so only the cases that actually compare numbers require it; the contract/fast-path cases run without it.
## Shared helpers (`support.py`)
- `DEVICE`, `Triple`, `triple()`, `population(specs, pad_to=…, output_len=1)`, `encode(spec, …)` — build the legacy triple on the device from a node-spec DSL: `("var", i)`, `("const", v)`, `(fid, kids)` / `("fn", fid, kids)`, `("out", fid, index, kids)`, plus `("raw", value, node_type, kids)`.
- `tagged(fid, out_index, children)`, `rows_of(...)`, `data(rows, columns, seed=7)` — the `OUT_NODE` spelling, the individual rows, and a deterministic data block.
- `oracle_value(oracle, rows, x, per_row=False, out_len=1)` / `oracle_loss(oracle, rows, x, labels, use_mse=True, out_len=1)` — the oracle calls, with the device↔host boundary in one place.
- `assert_same(value, expected, context, *, rtol=None, atol=0.0)` — one comparison with one message; `rtol=None` is a BITWISE claim.
- `served(call, context)` / `declined(call, context)` / `deltas(before, after)` — the counter contract around a call.
## Kept / rewritten / dropped (32 old functions → 29 · 45 cases)
| # | old case (`src/evogp/tests/test_fused_route.py`) | verdict | why / what changed |
|---|--------------------------------------------------|---------|--------------------|
| 1 | `test_value_parity` (parametrized over 5 sets × 2 paths) | kept (rewritten body) | drives `evaluate_fused` on the triple instead of `Forest.batch_forward`/`forward`; per-set `value_tol` preserved (arithmetic/ternary 0.0 = bitwise, loose/rich 1e-6) |
| 2 | `test_single_node_and_constant_populations` | kept | single VAR / single CONST / constant-subtree populations, bitwise |
| 3 | `test_deep_chain` | kept | 40-deep left and right chains; now built with `cases.chain` through `support.encode` (which gained the bare `(fid, kids)` spelling) |
| 4 | `test_protected_div_and_nan_inf_placement` | kept | `DIV`-by-const-0 and `SQRT`/`LOG` of a negative VAR: NaN/Inf placement matches the oracle, NaN payload may differ |
| 5 | `test_sr_fitness_matches_oracle` (MSE + MAE) | kept | loss path on the triple, `1e-5` relative |
| 6 | `test_sr_fitness_result_contract` | kept | fresh / contiguous / float32 / right device / no grad, and the same for the value path |
| 7 | `test_multi_output_value_parity` (out_len 2..8) | kept | now BITWISE (measured `max_abs = 0.0`), so the old `1e-6` margin is gone rather than loosened |
| 8 | `test_multi_output_loss_parity` (out_len 2..8, MSE+MAE) | kept | `1e-6` relative kept; measured `<= 1.2e-7` |
| 9 | `test_refuse_out_node_tags_in_single_output_mode` | kept | `MultiOutputError` from `evogp._multi_output`, message names row/node |
| 10 | `test_refuse_out_node_tags_through_the_public_api` | REWRITTEN → `…_on_every_entry_point_and_name_the_row` | "the public API" is now the pair of entry points, not `Forest`: both the value and the loss path must refuse before any launch, and the message must name the row and node |
| 11 | `test_the_same_rows_are_served_once_output_len_says_two` | kept | the same triple at `out_len 1` (refused) and `out_len 2` (served) |
| 12 | `test_route_policy_selects_fused` | DROPPED | route policy deleted (`EVOGP_EVAL_MODE` / `auto` / `Forest` dispatch); `resolve_route` no longer exists |
| 13 | `test_auto_selects_fused_above_the_floor` | DROPPED | same reason (`auto` + `EVOGP_FUSED_MIN_POP` are gone with the torch stack) |
| 14 | `test_auto_keeps_the_pre_fused_policy_below_the_floor_or_without_triton` | DROPPED | same reason |
| 15 | `test_served_call_increments_launches_only` | kept | one `fused_calls` + one `fused_kernel_launches` per served call, no decline |
| 16 | `test_decline_wide_output_len` | kept | `output_len = MAX_OUT_LEN + 1` declines, message names `MAX_OUT_LEN` and the prediction buffer |
| 17 | `test_decline_tiny_scratch_budget` | kept | a 1 MiB envelope declines, message names `EVOGP_FUSED_MAX_SCRATCH_MB` and the components |
| 18 | `test_noncontiguous_inputs_are_read_correctly` | kept | column/row slices of a padded `x` are bitwise equal to their `.contiguous()` copies |
| 19 | `test_noncontiguous_population_equals_contiguous` | kept | stride-2 population tables, both value and loss paths |
| 20 | `test_subtree_size_clamp` | kept | `subtree_size[0]` clamped to the row: 30000/32767 read the honest row, `-5` floors to 0 |
| 21 | `test_clear_cache_resets_the_fused_tables` | REWRITTEN → `test_clear_fused_cache_resets_every_table_and_counter` | the old body also asserted that `evoxc_engine.clear_cache()` resets the fused tables — that engine is deleted; this pins `clear_fused_cache()` as the ONE reset entry (counters zeroed, the same shape then re-served and re-compared) |
| 22 | `test_resolve_dispatch_spellings` | kept | `bound` default, `jit` arm, `""`/unset/whitespace, a typo raises `ValueError` |
| 23 | `test_jit_mode_serves_every_call_through_the_dispatcher` | kept | `EVOGP_FUSED_DISPATCH=jit` serves every call through `JITFunction.run` (0 bound launches) |
| 24 | `test_bound_mode_serves_warm_calls_after_one_dispatch` | kept | `bound` pays the dispatcher once per prepared call, then serves pre-bound |
| 25 | `test_bound_and_jit_agree_bitwise` | kept | the two launch paths are bitwise identical |
| 26 | `test_clear_fused_cache_rearms_the_dispatch` | kept | after a reset the next call binds again instead of using a stale handle |
| 27 | `test_an_in_place_channel_mutation_misses_the_plan_fast_path` | kept | a write into a population tensor (version-counter change) misses `PLAN_MEMO` and is re-prepared |
| 28 | `test_an_in_place_mutation_through_a_view_misses_the_plan_fast_path` | kept | the same through a view of the same storage |
| 29 | `test_the_served_fast_path_is_one_launch_per_call` | kept | the single-slot fast path means exactly one launch per call, hits included |
| 30 | `test_clear_fused_cache_drops_the_fast_path_the_geometry_memo_and_the_route_memo` | REWRITTEN → `…_the_fast_path_and_the_geometry_memo` | the route-memo half is dropped (the memo belonged to the deleted route policy); the `_LAST` single-slot fast path and `_GEOMETRY_MEMO` assertions are kept, plus a before/after numbers check |
| 31 | `test_active_launch_hooks_decline` | kept | with a registered Triton launch hook the route declines the pre-bound path (profilers must keep working) |
| 32 | `test_unusable_launcher_declines` | kept | an unusable bound launcher declines onto the historical dispatch instead of raising |
## Constraints
- No case may reference `evogp.tree`, `evogp.algorithm`, `evogp.problem`, `evogp.pipeline`, `Forest`, `Tree`, `dense_engine`, `resolve_route`, `EVOGP_EVAL_MODE` or the benchmark harness: those are deleted (or dev-only), and the suite is the one place that would hide the coupling.
- Numerics are compared against the oracle only; no expectation is hard-coded from a previous engine's output except the frozen fixtures, which live in `test/oracle/goldens/` and are gated by `evogp.fused.check --stage goldens`, not here.
- A tolerance may only be WIDENED with the measured number in a comment; the objective of every case is the tightest claim the route actually keeps.
- Keep the module's `_compiled` fixture: it compiles the four single-output specializations once for the whole module, which is what makes the suite ~1.4 s instead of ~5 s.
## Known Issues
- The suite only covers shapes with `rows >= 8` and `pop <= 64`; the `rows <= 64` × 8-32-node corner (pops 1-8, where the kernel's ~0.09 ms launch floor is the whole call) is still unmeasured here — `evogp.fused.check --stage timing` starts at `rows = 1024`.
- The scalar/`out_len 8` multi-output case is one hard-coded population, not a randomized sweep over the full `MAX_OUT_LEN` width.
- `test_active_launch_hooks_decline` needs `triton.knobs`, so it is a `pytest.importorskip` on triton builds that ship the hook chain (`pytest.importorskip` inside the case, not the module).
- The oracle is dev tooling: a checkout without `test/oracle` still collects this module but skips the numeric cases, so the contract cases are the only always-on part there.
