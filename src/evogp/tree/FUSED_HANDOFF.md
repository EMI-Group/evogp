# `EVOGP_EVAL_MODE=fused` — cross-node handoff (what the parent must apply outside `src/evogp/tree`)

The fused Triton route is implemented, wired, hardened and measured inside this node.
Everything that still has to change lives in a node this node cannot write.
Detail and quoted code: `fused_sibling_patches.md`; measurements and verdicts: `FUSED.md` + `fused_results/`.

## 1. Required: one stale test pin (`src/evogp/tests/test_dense_policy.py:339-355`)
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

## 2. Required for the campaign harness (`test/benchmark/evoxc/`)
`common.EVOXC_ARMS` needs the arm, and the route reporting needs a `fused` branch (a fused call is served on the torch device, not on an evoxc backend):

```python
    # the fused Triton route (one launch per (tree, row-block), no evoxc graph)
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
```

`common.route_decision` (and `route_text`) currently fold only `dense`/`batch`; under `eval_mode="fused"` it reports the dense-side `should_use_dense=False` and a `numpy` backend instead of `resolve_route(...).engine == "fused"`, which makes a record's route block wrong (not fatal).
Needed: a `fused_route_decision` sibling of `batch_route_decision` reading `route.engine` / `route.use_fused`, plus the `fused_cache_stats()` existence proof in the worker (an engine name is not proof a call was served — `fused_kernel_launches` is).
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
