# `EVOGP_EVAL_MODE=fused` — cross-node handoff (what the parent must apply outside `src/evogp/tree`)

The fused Triton route is implemented, wired, hardened and measured inside this node.
Everything that still has to change lives in a node this node cannot write.
Detail and quoted code: `fused_sibling_patches.md`; measurements and verdicts: `FUSED.md` + `fused_results/`.

## 1. Required in `src/evogp/tests/`: one stale pin, one rename

### 1a. The stale table pin (`test_dense_policy.py:339-355`)
`test_eval_mode_tables_stay_canonical` still asserts the pre-fused tuples, so the suite reports exactly one failure (1206 passed / 7 skipped / 1 failed).
Replace the two tuple asserts and add the two `fused` membership asserts:

```python
    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "batch", "fused", "auto")
    assert tuple(dense.ENGINES) == ("specialized", "dense", "batch", "fused")
    ...
    assert "fused" in dense.EVAL_MODES and "fused" in dense.ENGINES
    assert "fused" not in dense.EVAL_MODE_ALIASES
    assert "fused" not in set(dense.EVAL_MODE_ALIASES.values())
```

The test's docstring should say `fused` is the opt-in FOURTH engine, listed before `auto`, never an alias, never `auto`-selected.
Nothing else in `src/evogp/tests` fails.

### 1b. Relocate the route's pytest net (rename only)
`src/evogp/tree/fused_pytest_draft.py` is the suite's net for this route (hermetic: it SKIPS without a device, `triton` or a writable `TRITON_CACHE_DIR`).
The whole edit is one rename, no content change — after it `import reference_eval` resolves as a sibling:

```bash
git mv src/evogp/tree/fused_pytest_draft.py src/evogp/tests/test_fused_route.py
```

Paste-ready checklist and the re-verification counts: `fused_sibling_patches.md` §2.

## 2. Required for the campaign harness (`test/benchmark/evoxc/`)
Four edits, all in `test/benchmark/evoxc/`; the paste-ready code is in `fused_sibling_patches.md` §3–§6 (the subsection labels below name its sections).

### 2a. The arm registry (`common.EVOXC_ARMS`) — appendix §3
The campaign of record's registry (`fused_results/fused_campaign.py`'s `FUSED_ARMS`) registers TWO arms, appended after the last `batch` entry so every existing arm id keeps its index:

```python
    # the fused (Triton) route: one launch per (tree, row-block), no evoxc graph; the
    # explicit-backend spelling is unconditional, the bare one is the shipped policy
    "evoxc-fused-iree": {"eval_mode": "fused", "backend": "iree", "family": "evoxc"},
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
```

`evoxc-fused-iree` (explicit `backend="iree"`) makes the route resolve `fused` unconditionally, whatever the ambient backend env, with an `iree` fallback: it is the right arm when the round must not depend on an exporter's environment, and it is the one to pair with the `evoxc-dense-iree` / `evoxc-batch-iree` arms of a four-way round.
`evoxc-fused` (no explicit backend) is the POLICY's own decision — `EVOGP_FUSED_MIN_POP` and the device probe decide, exactly what a user gets without exporting `EVOGP_EVOXC_BACKEND` — so it is the right arm for "the shipped route"; on a CPU box or below the population floor its calls are the specialized fallback, i.e. its medians are not a kernel measurement (`fused_kernel_launches` tells them apart).

### 2b. `EVOGP_FUSED_*` are route knobs (`common.py`) — appendix §4
`FUSED_ENV_VARS` (the five knobs `MIN_POP`, `BLOCK_ROWS`, `NUM_WARPS`, `MAX_SCRATCH_MB`, `HOST_MEMO`) beside `DENSE_ENV_VARS`, folded into `ROUTE_ENV_VARS`; otherwise a caller's exported `EVOGP_FUSED_BLOCK_ROWS=64` silently applies to every arm of the round.

### 2c. Route reporting (`common.route_decision`, then `summarize.route_text`) — appendix §5
`route_decision` currently folds only `dense`/`batch`; under `eval_mode="fused"` it reports the dense-side `should_use_dense=False` and a `numpy` backend instead of `resolve_route(...).engine == "fused"`, which makes a record's route block wrong (not fatal) and renders a SERVED call as `fused/numpy → fused`.
The appendix's patch is an `if decision["eval_mode"] == "fused":` branch INLINE in `route_decision` (not a separate `fused_route_decision` function) asking the one `dense_engine.resolve_route(...)` and stamping `backend` / `engine` / `would_use_fused` / `fused_fallback_backend` / `fused_reason`, plus the `kind` derivation and the `declined` annotation in `route_text`.
A fused call is served on the torch device, not on an evoxc backend, so `engine` — never the backend spelling — is what the record must quote.

### 2d. The existence proof in `worker.py` — appendix §6
An engine name is not proof a call was served; `fused_kernel_launches` is (`fused_calls` counts ATTEMPTS).
The worker needs `FUSED_STATS_KEYS` (the 15 `fused_cache_stats()` keys), `FUSED_GAUGE_KEYS` (the 6 gauges among them, recorded as an AFTER snapshot, never a delta), `fused_served()` (`attempts - declines == calls` and `launches >= calls`), the `fused_window_block` mirror of `artifact.py`'s delta/gauge split, and `record["fused_cache"]` beside `record["engine_cache_final"]`.

The comment in `common.py` that still describes `EVAL_MODES` as three values needs updating.

## 3. Required: move the records and re-register the report
`src/evogp/tree/fused_results/` is the campaign of record and belongs at `test/benchmark/evoxc/results/t1a1_fused/` (that path is outside this node's write scope; `FUSED.md` §9 states this).
Its layout mirrors the baseline's: `main/` (one JSON per (cell, arm, workload) plus the `campaign*.json` indexes), `probes/`, `cold/`, `envelope.json`, `minpop.json`, `degenerate.json`, `sweep/sweep.json`, `validation-fused.json`, and the driver scripts `fused_campaign.py` / `fused_probe.py` / `fused_gate.py` / `campaign_index.py` / `fused_envelope.py` (these import `common` / `arms_torch` / `reference_eval` from the harness and the test tree, so they are harness tooling, not library code).
The harness's own gate reads the records as **20 records / 574 checks / 0 failures, 20/20 idle**, with `--allow-missing --allow-digest-mismatch` for the arms this campaign did not schedule.

## 4. Documentation drift (outside this node)
`CONTEXT.md` at `./`, `./src` and `./src/evogp` describe three evaluation routes only.
The root file additionally states that "only a fused device-side evaluator removes" the dispatch bound and that the r6 Triton prototype is "test-only, not integrated into `src/`" — both now false: the route ships as `EVOGP_EVAL_MODE=fused` at 33–157× over `torch-batched`.
`src/evogp/CONTEXT.md`'s routing table needs `fused_engine.py` / `fused_kernels.py` / `fused_check.py` / `FUSED.md` entries, and the environment-knob list needs the `EVOGP_FUSED_*` family (`MIN_POP` 16, `BLOCK_ROWS` 256, `NUM_WARPS` 1, `MAX_SCRATCH_MB` 8192, `HOST_MEMO` 16 import-time).

## 5. Operational note
Any process that launches the fused kernels needs a writable `TRITON_CACHE_DIR`: on this box `$HOME` is read-only and the first launch dies with `OSError: [Errno 30] Read-only file system: '/root/.triton'`.
The first call of a fresh process pays the Triton JIT (1.64–1.67 s cold for `batch_forward`, 0.94–0.99 s for `sr_fitness`, 0.32 s for `forward`); with a warm Triton disk cache it is ~2 ms.

## 6. Not a defect, but a deliberate limitation
`EVOGP_FUSED_MIN_POP` (16) keeps single-tree (`pop 1`) calls on the specialized engine so a one-shot `Tree.forward` never pays the JIT; `EVOGP_FUSED_MIN_POP=1` is legal and routes them through Triton (measured working), and `0` is refused.
`MAX_OUT_LEN` is 8; a wider `output_len` declines naming it.
The route declines on a non-torch population (a `ValueError`) and on a shape outside the scratch envelope, always onto specialized+`numpy`.
