# `EVOGP_EVAL_MODE=fused` — cross-node handoff (all edits applied)
The fused Triton route is implemented, wired, hardened and measured inside this node, and every edit it needed OUTSIDE `src/evogp/tree/` has been applied.
The route is `auto`'s FIRST choice on a device box: `dense_engine.auto_fused_decline` gates on the memoized availability probe (`triton` importable AND a torch-visible CUDA/HIP device), `pop_size >= EVOGP_FUSED_MIN_POP` (16) and the fused memory envelope; `EVOGP_EVAL_MODE=fused` forces it and an explicit backend keeps the historical dense/specialized rule.
Measurements and verdicts: `FUSED.md`; records: `test/benchmark/evoxc/results/t1a1_fused/` (read against `results/t1a1_baseline/`).
## What landed where
| edit | landed in |
|---|---|
| the `EVAL_MODES` / `ENGINES` tuple pin | `src/evogp/tests/test_dense_policy.py` (`fused` listed before `auto`, never an alias) |
| the route's pytest net | `src/evogp/tests/test_fused_route.py` (53 cases) |
| the harness arm registry | `test/benchmark/evoxc/common.py` (`evoxc-fused-iree`, `evoxc-fused`, appended after the last batch entry) |
| the five route knobs | `common.FUSED_ENV_VARS`, folded into `ROUTE_ENV_VARS` |
| route reporting | `common.route_decision` stamps `engine` / `would_use_fused`, and `summarize.route_text` annotates a decline |
| the existence proof | `worker.FUSED_STATS_KEYS` (15 keys) / `worker.FUSED_GAUGE_KEYS` (6 gauges) / `worker.fused_served()`, recorded as `record["fused_cache"]` |
| the campaign of record | `test/benchmark/evoxc/results/t1a1_fused/` (gate: 20 records / 580 checks / 0 failures) |
## 1. The pytest net
`src/evogp/tests/test_fused_route.py` is the suite's net for this route: hermetic (it SKIPS without a device, `triton` or a writable `TRITON_CACHE_DIR`) and it lifts the session `EVOGP_EVOXC_BACKEND=numpy` pin per test, because `backend="numpy"` is a documented DECLINE of the route, not a fallback.
Its `_served` helper asserts `fused_kernel_launches > 0` and `fused_declines == 0`, so a passing case proves a real Triton launch.
## 2. The suite is green
`pytest src/evogp/tests -q` reads 1 308 passed / 7 skipped / 0 failed (1 315 collected, ~121 s) on the device reference box, and `test_fused_route.py` contributes 53 of those passes.
The policy net pins the route in both directions: `test_auto_selects_fused_above_the_floor` (pop 16 / 64 / 256 — `auto` resolves to `fused` from `EVOGP_FUSED_MIN_POP` up) and `test_auto_keeps_the_pre_fused_policy_below_the_floor_or_without_triton` (`specialized` below the floor and with the fused probe off).
No test in `src/evogp/tests` is red.
## 3. Operational note
Any process that launches the fused kernels needs a writable `TRITON_CACHE_DIR`: on this box `$HOME` is read-only and the first launch dies with `OSError: [Errno 30] Read-only file system: '/root/.triton'`.
The first call of a fresh process pays the Triton JIT (~1.6 s for `batch_forward`, 0.94–0.99 s for `sr_fitness`, 0.32 s for `forward`); with a warm Triton disk cache it is ~2 ms.
## 4. Not a defect, but a deliberate limitation
`EVOGP_FUSED_MIN_POP` (16) keeps single-tree (`pop 1`) calls on the specialized engine so a one-shot `Tree.forward` never pays the JIT; `EVOGP_FUSED_MIN_POP=1` is legal and routes them through Triton (measured working), and `0` is refused.
`MAX_OUT_LEN` is 8; a wider `output_len` declines naming it.
The route declines on a non-torch population (a `ValueError`) and on a shape outside the scratch envelope, always onto specialized+`numpy`.
