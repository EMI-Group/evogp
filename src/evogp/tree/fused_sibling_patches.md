# Cross-node edits for the fused route (`EVOGP_EVAL_MODE=fused`)

`FUSED_HANDOFF.md` is the authoritative one-page summary of what must be applied **outside**
`src/evogp/tree/`; this file is its code appendix — only the edits that are *still needed*, quoted
exactly as they must be pasted, verified against the current tree (never a commit pin).

Re-verified: `pytest test/benchmark/evoxc -q` → **28 passed**; `pytest src/evogp/tests -q` → **1 failed /
1206 passed / 7 skipped**, that failure being the pin in §1 (and **1207 / 7** after it);
`fused_cache_stats()` has **15** keys, `EVOGP_FUSED_MIN_POP` is **16**, `MAX_OUT_LEN` **8**.

## 1. REQUIRED — the stale table pin (`test_dense_policy.py:339-355`)

Replace the two tuple asserts (live `:339-340`), add the membership asserts beside the `batch` mirrors
(live `:353-355`) and state in the docstring that `fused` is the opt-in FOURTH engine, listed before
`auto`, never an alias, never `auto`-selected — nothing else in `src/evogp/tests/` needs an edit.

```python
    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "batch", "fused", "auto")
    assert tuple(dense.ENGINES) == ("specialized", "dense", "batch", "fused")
    ...
    assert "fused" in dense.EVAL_MODES and "fused" in dense.ENGINES
    assert "fused" not in dense.EVAL_MODE_ALIASES
    assert "fused" not in set(dense.EVAL_MODE_ALIASES.values())
```

## 2. REQUIRED — relocate the pytest net

`tree/fused_pytest_draft.py` is the SUITE's net for this route (hermetic: it SKIPS without a device /
`triton` / a writable `TRITON_CACHE_DIR`); the whole edit is
`git mv src/evogp/tree/fused_pytest_draft.py src/evogp/tests/test_fused_route.py`, after which
`import reference_eval` resolves as a sibling — a rename only, no content change. Carried by
`FUSED_HANDOFF.md` §1b.

## 3. REQUIRED — the harness arm registry (`common.py`)

Append both arms **after the LAST batch entry** (live `:185`, before the `}` at `:186`) so every
existing arm id keeps its index — verbatim from `fused_results/fused_campaign.py:92-95` (`FUSED_ARMS`):

```python
    # the fused (Triton) route: one launch per (tree, row-block), no evoxc graph; the
    # explicit-backend spelling is unconditional, the bare one is the shipped policy
    "evoxc-fused-iree": {"eval_mode": "fused", "backend": "iree", "family": "evoxc"},
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
}
```

`ALL_ARMS` is derived, so `--arms all|evoxc` pick them up and `--arms batch` does not.
`test_arm_registry.py::test_new_arm_ids_are_appended_last_and_are_batch_arms` goes red on its two
trailing pins: replace live `:154-155`, append after the `for arm in new:` loop (live `:156-158`),
keep `:159-161`:

```python
    fused = ("evoxc-fused-iree", "evoxc-fused")
    tail = new + fused
    assert tuple(common.EVOXC_ARMS)[-len(tail):] == tail
    assert tuple(common.ALL_ARMS)[-len(tail):] == tail
    ...
    for arm in fused:
        assert common.EVOXC_ARMS[arm]["eval_mode"] == "fused", arm
        assert common.is_batch_arm(arm) is False, arm
    assert common.EVOXC_ARMS["evoxc-fused-iree"]["backend"] == "iree"
    assert common.EVOXC_ARMS["evoxc-fused"]["backend"] is None
```

## 4. REQUIRED — the five `EVOGP_FUSED_*` knobs are route knobs (`common.py`, land with §3)

Otherwise a caller's exported `EVOGP_FUSED_BLOCK_ROWS=64` silently applies to every arm of the round.
Next to `DENSE_ENV_VARS` (live `:247`), folded into the sum (live `:261`):

```python
#: The FUSED route's knobs (``evogp.tree.fused_engine``): the kernel geometry, the memory
#: envelope and the transpose table's size (``HOST_MEMO`` is IMPORT-TIME).
FUSED_ENV_VARS: Tuple[str, ...] = (
    "EVOGP_FUSED_MIN_POP", "EVOGP_FUSED_BLOCK_ROWS", "EVOGP_FUSED_NUM_WARPS",
    "EVOGP_FUSED_MAX_SCRATCH_MB", "EVOGP_FUSED_HOST_MEMO",
)
ROUTE_ENV_VARS: Tuple[str, ...] = (
    BATCH_ENV_VARS + DENSE_ENV_VARS + FUSED_ENV_VARS + ARTIFACT_ENV_VARS
)
```

## 5. REQUIRED — route reporting (`common.route_decision`, then `summarize.route_text`)

Into `route_decision`, **after** the `would_use_dense` assignment (live `:1150-1154`) and before its
`return decision` (live `:1155`):

```python
    if decision["eval_mode"] == "fused":
        # ask the ONE resolver: without this a SERVED call renders as ``fused/numpy ->
        # fused`` (``resolve_backend_name(None, None)`` answers ``numpy``)
        route = dense_engine.resolve_route(
            pop, rows=rows, backend=backend, eval_mode=eval_mode, tree_len=tree_len,
        )
        decision["backend"] = route.backend
        decision["engine"] = route.engine
        decision["would_use_fused"] = bool(route.use_fused)
        decision["fused_fallback_backend"] = route.fallback_backend
        decision["fused_reason"] = route.reason
    return decision
```

Those four keys exist ONLY for a fused request, so the `set(decision)` pin in
`test_route_decision_passes_the_population_shape` (`test_arm_registry.py:331-334`) still passes (its
call leaves `eval_mode=None`, i.e. `auto`, and never enters the branch);
`engine` carries the decline — an explicit `backend="iree"` is `Route(False, "iree", "numpy",
engine="fused")` (the hardening: **the fallback is always `numpy`**), a CPU box or a population under
`EVOGP_FUSED_MIN_POP` answers `engine="specialized"` on `numpy`. In `route_text`
(`summarize.py:287-291`) replace the `kind` derivation and add the decline note:

```python
    kind = route.get("kind")
    if kind is None:
        engine = route.get("engine")
        if engine in ("batch", "fused"):
            kind = engine
        elif mode == "fused":
            kind = "fused"
        else:
            kind = "dense" if route.get("would_use_dense") else "specialized"
    text = f"{mode}/{backend} → {kind}"
    annotations: List[str] = []
    if kind == "fused" and not route.get("would_use_fused"):
        annotations.append("declined")  # the medians are the specialized fallback
```

## 6. REQUIRED — the existence proof in `worker.py`

The facade is silent on a decline and never compiles, so `cache.compile_misses_first_call` is `0` by
construction and proves nothing; `fused_kernel_launches` is the proof. Module level, next to
`stats_keys` (live `:236`), with `import counter_blocks` beside `import artifact` (live `:97`):

```python
#: The fused route's 15 counters (``fused_engine.fused_cache_stats()``); six are GAUGES.
FUSED_STATS_KEYS: Tuple[str, ...] = (
    "fused_calls", "fused_declines", "fused_xt_hits", "fused_xt_misses",
    "fused_xt_entries", "fused_xt_maxsize", "fused_kernel_launches",
    "fused_kernel_compiles", "fused_scratch_allocs", "fused_scratch_reuses",
    "fused_scratch_evictions", "fused_scratch_entries", "fused_scratch_bytes",
    "fused_block_rows", "fused_num_warps",
)
#: The ones that are GAUGES (an AFTER-state), never a delta.
FUSED_GAUGE_KEYS: Tuple[str, ...] = (
    "fused_xt_entries", "fused_xt_maxsize", "fused_scratch_entries",
    "fused_scratch_bytes", "fused_block_rows", "fused_num_warps",
)


def fused_served(block: Dict[str, Any], calls: Optional[int]) -> Optional[bool]:
    """Did ``calls`` calls reach a Triton launch?  ``fused_calls`` counts ATTEMPTS."""
    attempts = block.get("fused_calls")
    declines = block.get("fused_declines")
    launches = block.get("fused_kernel_launches")
    if None in (attempts, declines, launches) or not calls:
        return None
    return int(attempts) - int(declines) == int(calls) and int(launches) >= int(calls)
```

`fused_window_block(before, after)` is `counter_blocks.block(...)` with those keys — mirror
`artifact.py`: `delta_int` for the nine monotone counters, `integer` of the AFTER snapshot for the six
gauges. Wiring: resolve the accessor for a fused arm only, next to the batch probe (live `:546-560`);
snapshot `fused_before` per workload, pass it to `BatchStatsProbe`, and after `entry["cache"]` (live
`:802-810`) add `cache.fused`, `fused_after_first_call`, `fused_after`, `fused_first_call` and
`fused_served(...)`; record `record["fused_cache"]` = `{available, stats_before_workloads,
stats_after, stats_keys, totals, served_every_workload}` beside `record["engine_cache_final"]`,
noting an absent accessor (counters null, NOT zero) and a `served_every_workload` that is not `True`
(that median is the specialized fallback).

## Apply order and superseded items

Order: §1 alone (makes `pytest src/evogp/tests` green); §2 alone (the draft SKIPS on a CPU-only box);
§3 + §4 + §3's test edit in one commit; §5 + §6 in one commit; then move `fused_results/` to
`test/benchmark/evoxc/results/t1a1_fused/` and gate it with `check.py --allow-missing
--allow-digest-mismatch`, because a fused `arith`-set `sr_fitness` digest differs by float32 reduction
order.

Deleted as superseded: the old 13-key `FUSED_STATS_KEYS` (hardening added `fused_scratch_evictions` +
`fused_scratch_bytes`); the old §6d on `tree/CONTEXT.md`'s size/prune state (done inside this node);
the old doc-drift sentences for the three `CONTEXT.md` files and `test/benchmark/evoxc/CONTEXT.md`,
incl. their `38.9-153.5x` / `33.5-146.6x` sweep pair (now `FUSED_HANDOFF.md` §4); the
`test_arm_registry.py:284` stub `EVAL_MODES` note (inert). Dropped as optional follow-ups, not carried
by `FUSED_HANDOFF.md`: a `cold_start.py --arm` selector, a `decompose.py` `ROUTES["fused-iree"]`
entry, `common.is_fused_arm` + `campaign.py --arms fused`, and `check.py --expect-arms` advice —
`check.py` derives its arms from `campaign.json`.
