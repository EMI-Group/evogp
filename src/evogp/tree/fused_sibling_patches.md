# Sibling patches for the fused route — exact edits OUTSIDE `src/evogp/tree/`

Scope: this file lists the minimal edits the parent agent must apply **outside** `src/evogp/tree/`
so that (a) the hermetic pytest suite passes again and (b) the benchmark harness can schedule and
report an `EVOGP_EVAL_MODE=fused` arm.
Nothing here changes the library's route semantics: `tree/fused_engine.py`, `tree/dense_engine.py`
(`EVAL_MODES` / `ENGINES` / `Route.use_fused`), `tree/fused_kernels.py`, `tree/fused_check.py` and
`tree/CONTEXT.md`'s fused sections are already in place.

Verified against this worktree at commit `b96618a` (all line numbers are of that tree).
Method: every "before" block below was read from the live file; the single currently failing pytest
was reproduced with
`EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc /tmp/rocvenv/bin/python -m pytest src/evogp/tests/test_dense_policy.py -q -k eval_mode_tables_stay_canonical`
→ `1 failed, 84 deselected`, `At index 3 diff: 'fused' != 'auto'`.
The harness's own torch-free tests (`test/benchmark/evoxc/test_arm_registry.py`,
`test_large_cells.py`) are **green today** (28 passed) and go red the moment §2 appends an arm —
§2's test edit is therefore REQUIRED in the same commit as the registry edit, not optional.

| # | File | Class | One-line reason |
|---|------|-------|-----------------|
| 1 | `src/evogp/tests/test_dense_policy.py` | REQUIRED | pins the pre-`fused` `EVAL_MODES`/`ENGINES` tuples; the only red test in the suite |
| 2 | `test/benchmark/evoxc/common.py` (+ `test_arm_registry.py`) | REQUIRED | no `fused` arm exists, and appending one breaks the "trailing arms" pin |
| 3 | `test/benchmark/evoxc/common.py` `route_decision` + `summarize.py` `route_text` | REQUIRED | a fused record would render as `fused/iree → specialized` |
| 4 | `test/benchmark/evoxc/worker.py` | REQUIRED | a served fused call and a silently declined one look identical today |
| 5 | `cold_start.py`, `decompose.py`, `check.py`, `ROUTE_ENV_VARS` | OPTIONAL | pricing/tooling niceties, not needed for the arm to run |
| 6 | 4 × `CONTEXT.md` | REQUIRED (docs) | three-route wording, old `EVOGP_EVAL_MODE` spellings, and an over-size `tree/CONTEXT.md` |

---

## 1. REQUIRED — `src/evogp/tests/test_dense_policy.py::test_eval_mode_tables_stay_canonical`

**Why required:** the shipped tuples are `EVAL_MODES = ("dense", "specialized", "batch", "fused", "auto")`
(`src/evogp/tree/dense_engine.py:636`) and `ENGINES = ("specialized", "dense", "batch", "fused")`
(`:642`), while the assertion still pins the 4-/3-tuple. This is a pure table pin, not a behaviour
test — the new block only has to say what the tables now are.

Live code (`src/evogp/tests/test_dense_policy.py:331-355`):

```python
331	def test_eval_mode_tables_stay_canonical():
332	    """The alias lives in its own table; ``EVAL_MODES`` stays the canonical tuple.
333	
334	    ``batch`` is the opt-in THIRD engine: a canonical mode (never an alias, and no
335	    alias value may point at it), listed before ``auto`` so the default stays the
336	    last entry, and named in ``ENGINES`` next to the two engines the policy had
337	    before it.
338	    """
339	    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "batch", "auto")
340	    assert tuple(dense.ENGINES) == ("specialized", "dense", "batch")
341	    assert dense.DEFAULT_EVAL_MODE == "auto"
342	    assert dense.DEFAULT_EVAL_MODE == dense.EVAL_MODES[-1]
343	    assert dict(dense.EVAL_MODE_ALIASES) == {"dynamic": "dense"}
344	    assert dense.EVAL_MODE_ENV == "EVOGP_EVAL_MODE"
345	    # the alias target is canonical, i.e. `dynamic` is not a mode of its own
346	    assert set(dense.EVAL_MODE_ALIASES.values()) <= set(dense.EVAL_MODES)
347	    assert "dynamic" not in dense.EVAL_MODES
348	    # every engine names a canonical mode, and `auto` (a policy, not an engine)
349	    # names none
350	    assert set(dense.ENGINES) <= set(dense.EVAL_MODES)
351	    assert "auto" not in dense.ENGINES
352	    # `batch` is reached by its own spelling only, never through the alias table
353	    assert "batch" in dense.EVAL_MODES and "batch" in dense.ENGINES
354	    assert "batch" not in dense.EVAL_MODE_ALIASES
355	    assert "batch" not in set(dense.EVAL_MODE_ALIASES.values())
```

AFTER (copy-paste; docstring updated, `auto`-last kept, `fused` asserted as the fourth engine):

```python
def test_eval_mode_tables_stay_canonical():
    """The alias lives in its own table; ``EVAL_MODES`` stays the canonical tuple.

    ``batch`` and ``fused`` are the opt-in THIRD and FOURTH engines: canonical modes
    (never aliases, and no alias value may point at either), listed before ``auto`` so
    the default stays the last entry, and named in ``ENGINES`` next to the two engines
    the policy had before them.
    """
    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "batch", "fused", "auto")
    assert tuple(dense.ENGINES) == ("specialized", "dense", "batch", "fused")
    assert dense.DEFAULT_EVAL_MODE == "auto"
    assert dense.DEFAULT_EVAL_MODE == dense.EVAL_MODES[-1]
    assert dict(dense.EVAL_MODE_ALIASES) == {"dynamic": "dense"}
    assert dense.EVAL_MODE_ENV == "EVOGP_EVAL_MODE"
    # the alias target is canonical, i.e. `dynamic` is not a mode of its own
    assert set(dense.EVAL_MODE_ALIASES.values()) <= set(dense.EVAL_MODES)
    assert "dynamic" not in dense.EVAL_MODES
    # every engine names a canonical mode, and `auto` (a policy, not an engine)
    # names none
    assert set(dense.ENGINES) <= set(dense.EVAL_MODES)
    assert "auto" not in dense.ENGINES
    # `batch` and `fused` are reached by their own spelling only, never via aliases
    assert "batch" in dense.EVAL_MODES and "batch" in dense.ENGINES
    assert "fused" in dense.EVAL_MODES and "fused" in dense.ENGINES
    assert "batch" not in dense.EVAL_MODE_ALIASES
    assert "batch" not in set(dense.EVAL_MODE_ALIASES.values())
    assert "fused" not in dense.EVAL_MODE_ALIASES
    assert "fused" not in set(dense.EVAL_MODE_ALIASES.values())
```

### Other tests in `src/evogp/tests/` — audit result (nothing else needs an edit)

`rg -n "EVAL_MODES|ENGINES|\bfused\b" src/evogp/tests/` returns, besides the block above:

* `src/evogp/tests/test_route_policy.py:407` and `:562` — `assert route.engine in dense.ENGINES`
  (membership, not equality): **passes** with `fused` appended.
* `src/evogp/tests/test_route_policy.py:380` (`[None, "auto", "dense", "dynamic", "specialized", "batch"]`)
  and `:541` (`[None, "auto", "dense", "dynamic", "specialized"]`) are `@pytest.mark.parametrize`
  matrices (`test_resolve_route_and_should_use_dense_agree` / `test_auto_never_selects_the_batch_route`)
  that are **deliberately** explicit. They need no edit: the identity they assert still holds for the
  fused branch (`should_use_dense` delegates to `resolve_route`, so it answers `False` where the route
  says `engine="fused"`). Adding `"fused"` would pass but is out of their scope — the fused route's own
  consistency gate is `fused_check.py --stage wired` (`tree/CONTEXT.md`), which asserts the served /
  declined / `auto`-never cases directly. Leave them untouched.
* `src/evogp/tests/test_dense_policy.py:325` (`@pytest.mark.parametrize("mode", list(dense.EVAL_MODES))`)
  adapts itself — `resolve_eval_mode("fused") == "fused"` already holds.
* No test file in `src/evogp/tests/` asserts `"fused" not in ...` anywhere — the word `fused` does
  not occur in that tree at all outside this file (`rg -n "\bfused\b" src/evogp/tests/` matches
  nothing today).
* `src/evogp/tests/CONTEXT.md:147` still documents the old tuples (doc drift only, not a test;
  handed to §6).

So: **one** test edit in §1.

---

## 2. REQUIRED — the harness arm registry (`test/benchmark/evoxc/common.py`)

**Why required:** without a registry entry no campaign can schedule the route
(`worker.py` only knows arms from `common.EVOXC_ARMS` / `common.TORCH_ARMS`), and
`campaign.py --arms …` never produces the name.

Live registry tail (`test/benchmark/evoxc/common.py:177-191`) — the insert point is line 185, i.e.
immediately after the LAST batch entry, so every pre-existing arm id keeps its index:

```python
177	    "evoxc-batch-iree-artifact-p8": {
178	        "eval_mode": "batch",
179	        "backend": "iree",
180	        "family": "evoxc",
181	        "knobs": {
182	            "EVOGP_EVOXC_ARTIFACT_DIR": "{artifact_dir}",
183	            "EVOGP_BATCH_COMPILE_WORKERS": "8",
184	        },
185	    },
186	}
187	TORCH_ARMS: Dict[str, Dict[str, Any]] = {
188	    "torch-per-tree": {"impl": "per_tree", "family": "torch"},
189	    "torch-batched": {"impl": "batched", "family": "torch"},
190	}
191	ALL_ARMS: List[str] = list(TORCH_ARMS) + list(EVOXC_ARMS)
```

AFTER (matching the neighbours' shape and one-line-comment style; both arms appended LAST):

```python
    "evoxc-batch-iree-artifact-p8": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {
            "EVOGP_EVOXC_ARTIFACT_DIR": "{artifact_dir}",
            "EVOGP_BATCH_COMPILE_WORKERS": "8",
        },
    },
    # one Triton launch per population (its own routed kernel, no evoxc backend):
    # ``iree`` is only the backend SPELLING every device route uses, the facade
    # declines for any other name and its fallback is always specialized+numpy
    "evoxc-fused-iree": {"eval_mode": "fused", "backend": "iree", "family": "evoxc"},
    # the same route with BOTH route knobs unset, i.e. the shipped no-explicit-backend
    # policy: fused only on a device box at ``pop >= EVOGP_FUSED_MIN_POP`` (64), and
    # specialized+numpy below it / on a CPU box -- the arm that prices the POLICY
    # (this is what makes a pop-16 or pop-64 cell answer "what does auto's geometry
    # pick for a Triton route", not "what does a forced route cost")
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
}
```

`common.ALL_ARMS` (line 191) is **derived** (`list(TORCH_ARMS) + list(EVOXC_ARMS)`) — it needs no
edit, and `campaign.py --arms all|evoxc` picks the new arms up for free. `campaign.py --arms batch`
does **not** (it filters on `common.is_batch_arm`, `campaign.py:316-321`), which is correct: a fused
arm is not a batch arm. See §5 for the optional `--arms fused` selector.

### The consequence: `test/benchmark/evoxc/test_arm_registry.py` goes red

Live, and it is the trailing-arms pin that breaks (`test_arm_registry.py:145-161`):

```python
145	def test_new_arm_ids_are_appended_last_and_are_batch_arms():
146	    new = (
147	        "evoxc-batch-iree-aggressive",
148	        "evoxc-batch-iree-artifact",
149	        "evoxc-batch-iree-default",
150	        "evoxc-batch-iree-artifact-p1",
151	        "evoxc-batch-iree-artifact-p2",
152	        "evoxc-batch-iree-artifact-p8",
153	    )
154	    assert tuple(common.EVOXC_ARMS)[-len(new):] == new
155	    assert tuple(common.ALL_ARMS)[-len(new):] == new
156	    for arm in new:
157	        assert common.is_batch_arm(arm), arm
158	        assert common.EVOXC_ARMS[arm]["backend"] == "iree"
159	    # ``--arms batch`` picks them up from the ONE predicate (no extra rule)
160	    batch_arms = [arm for arm in common.EVOXC_ARMS if common.is_batch_arm(arm)]
161	    assert list(batch_arms) == ["evoxc-batch-iree", "evoxc-batch-iree-flat", *new]
```

After the §2 registry edit (two arms appended, 22 evoxc arms in total),
`tuple(common.EVOXC_ARMS)[-6:]` is
`("evoxc-batch-iree-default", "evoxc-batch-iree-artifact-p1", "evoxc-batch-iree-artifact-p2",
"evoxc-batch-iree-artifact-p8", "evoxc-fused-iree", "evoxc-fused")` instead of `new` — so line 154
and 155 fail and nothing else in the test does (the `for arm in new:` loop still only walks batch
arms). Ship only `"evoxc-fused-iree"` and the same two lines fail with a different tail; the fix
below is the same either way.

AFTER (copy-paste, same test, same name):

```python
def test_new_arm_ids_are_appended_last_and_are_batch_arms():
    new = (
        "evoxc-batch-iree-aggressive",
        "evoxc-batch-iree-artifact",
        "evoxc-batch-iree-default",
        "evoxc-batch-iree-artifact-p1",
        "evoxc-batch-iree-artifact-p2",
        "evoxc-batch-iree-artifact-p8",
    )
    # the fused route is the FOURTH engine and its arms are appended after the batch
    # block (registry order is render order), so they extend the trailing pin
    fused = ("evoxc-fused-iree", "evoxc-fused")
    tail = new + fused
    assert tuple(common.EVOXC_ARMS)[-len(tail):] == tail
    assert tuple(common.ALL_ARMS)[-len(tail):] == tail
    for arm in new:
        assert common.is_batch_arm(arm), arm
        assert common.EVOXC_ARMS[arm]["backend"] == "iree"
    for arm in fused:
        assert common.EVOXC_ARMS[arm]["eval_mode"] == "fused", arm
        assert common.is_batch_arm(arm) is False, arm
    assert common.EVOXC_ARMS["evoxc-fused-iree"]["backend"] == "iree"
    assert common.EVOXC_ARMS["evoxc-fused"]["backend"] is None
    # ``--arms batch`` picks up EXACTLY the batch arms from the ONE predicate
    batch_arms = [arm for arm in common.EVOXC_ARMS if common.is_batch_arm(arm)]
    assert list(batch_arms) == ["evoxc-batch-iree", "evoxc-batch-iree-flat", *new]
```

The other two order pins keep passing **because the new arms are appended** (do not insert them
between the batch entries):

* `test_arm_registry.py:137-142` — `tuple(common.EVOXC_ARMS)[: len(evoxc_ids)] == evoxc_ids` and
  `tuple(common.ALL_ARMS[: len(EXISTING_ARMS)]) == EXISTING_ARMS` with `EXISTING_ARMS` at
  `:59-70`; a prefix pin, unaffected by an append.
* `test_arm_registry.py:248-264` (`test_no_arm_leaks_a_route_knob`) iterates `common.ALL_ARMS` and
  will now also check the two fused arms: they set no `knobs`, so every `ROUTE_ENV_VARS` entry must
  be absent afterwards — which holds, and becomes a real check once §5 adds the `EVOGP_FUSED_*`
  names to `ROUTE_ENV_VARS`. **Land §5's `ROUTE_ENV_VARS` line together with §2**, otherwise an
  inherited `EVOGP_FUSED_BLOCK_ROWS` would survive into a fused arm and this test would not notice.

`ALL_ARMS` expectations: `test_arm_registry.py` has no absolute-length assertion on `ALL_ARMS`, and
`check.py` derives its expected set from `campaign.json` (§5), so no further test edit is needed.

Stale-but-harmless, no edit required: the stub class at `test_arm_registry.py:284` still says
`self.EVAL_MODES = ("dense", "specialized", "batch", "auto")`. Nothing reads it (the stubbed
`resolve_eval_mode` returns `"auto"`), so leave it — or extend it for honesty, either way green.

---

## 3. REQUIRED — route reporting for a fused record

**Why required:** `common.route_decision` has no fused branch, so `summarize.route_text` renders a
fused record as `fused/iree → specialized` (it derives the kind from `would_use_dense`, which is
`False` for every fused route).

### 3a. `common.route_decision` (`test/benchmark/evoxc/common.py:1104-1155`)

Live tail of the function:

```python
1124	    """
1125	    # The batch route is intercepted BEFORE the dense policy is imported: on the
1126	    # current tree dense_engine.EVAL_MODES == ("dense", "specialized", "auto"), so
1127	    # resolve_eval_mode("batch") raises ValueError and the batch decision must come
1128	    # from its own module (batch_route_decision), never from that call.
1129	    if _requested_eval_mode(eval_mode) == "batch":
1130	        return batch_route_decision(pop, rows)
1131	
1132	    from evogp.tree import dense_engine
1133	
1134	    shape = {"pop_size": pop, "tree_len": tree_len}
1135	    decision = {
1136	        "eval_mode": dense_engine.resolve_eval_mode(eval_mode),
1137	        "backend": dense_engine.resolve_backend_name(backend, None),
1138	    }
...
1150	    decision["would_use_dense"] = decision["should_use_dense_pop_only"] and not bool(
1151	        dense_engine.auto_rows_decline(
1152	            rows, eval_mode=decision["eval_mode"], **shape
1153	        )
1154	    )
1155	    return decision
```

AFTER — insert the fused branch **between line 1150-1154 and the `return`** (nothing else changes):

```python
    decision["would_use_dense"] = decision["should_use_dense_pop_only"] and not bool(
        dense_engine.auto_rows_decline(
            rows, eval_mode=decision["eval_mode"], **shape
        )
    )
    if decision["eval_mode"] == "fused":
        # The fused (Triton) route is not the dense half of the policy and has no
        # ``block`` of its own: it reports itself through ``Route.engine`` /
        # ``use_fused`` / ``fallback_backend`` / ``reason``, so ask the ONE resolver
        # instead of deriving a kind here.  All five keys above stay present
        # (``would_use_dense`` stays False, i.e. "this call is not dense"), and the
        # four keys below exist ONLY for a fused request so the auto path's record is
        # byte-identical (``test_route_decision_passes_the_population_shape`` pins
        # exactly those five keys).
        route = dense_engine.resolve_route(
            pop,
            rows=rows,
            backend=backend,
            eval_mode=eval_mode,
            tree_len=tree_len,
        )
        # ``backend`` is the one of the five whose VALUE the fused branch must correct:
        # ``resolve_backend_name(None, None)`` answers the engine default ``"numpy"``,
        # so a served fused call on the no-explicit-backend arm would otherwise render
        # as ``fused/numpy → fused``.  ``route.backend`` is the RESOLVED one (``iree``
        # when served, ``numpy`` when the policy declined).
        decision["backend"] = route.backend
        decision["engine"] = route.engine
        decision["would_use_fused"] = bool(route.use_fused)
        decision["fused_fallback_backend"] = route.fallback_backend
        decision["fused_reason"] = route.reason
    return decision
```

Resulting key set for a fused request (9 keys — the auto path's five names, plus four):
`eval_mode`, `backend` (the RESOLVED backend), `should_use_dense_pop_only`,
`should_use_dense_with_rows`, `would_use_dense` (always `False`: the fused route is never the dense
half), `engine`, `would_use_fused`, `fused_fallback_backend`, `fused_reason`.
`engine` is what carries the decline: `EVOGP_EVAL_MODE=fused` below `EVOGP_FUSED_MIN_POP` (or on a CPU
box) resolves `Route(False, "numpy", "numpy", …, engine="specialized")` — see
`src/evogp/tree/dense_engine.py:1608-1620`.

The sibling test that pins the 5-key shape (`test/benchmark/evoxc/test_arm_registry.py:329-333`; line 334 is blank) —
quoted because it is the reason the four new keys are gated behind the mode:

```python
329	    # the pre-existing keys keep their meaning (the pop-only half is still shape-blind)
330	    assert set(decision) == {
331	        "eval_mode", "backend", "should_use_dense_pop_only",
332	        "should_use_dense_with_rows", "would_use_dense",
333	    }
334
```

That test drives `common.route_decision(256, 4096, None, "iree", tree_len=64)` against
`_FakeDenseEngine` (`:278-302`), whose `resolve_eval_mode` returns `"auto"`, so it never reaches the
new branch and still passes unchanged — **no test edit needed for §3a**. (The stub has no
`resolve_route`; adding one is only needed if a *new* test wants to exercise the fused branch
without torch.)

One freebie in the same file: the live comment at `common.py:1126` still claims
`dense_engine.EVAL_MODES == ("dense", "specialized", "auto")` (pre-`batch`, pre-`fused`). It is a
comment, no test reads it — rewrite it to `("dense", "specialized", "batch", "fused", "auto")` while
the file is open. OPTIONAL.

### 3b. `summarize.route_text` (`test/benchmark/evoxc/summarize.py:259-303`)

Live:

```python
259	def route_text(records, cell: str, workload: str) -> str:
260	    """``eval_mode/backend -> kind`` from the cell's route record.
...
266	    column only for a directory that recorded nothing else.  ``kind`` is a field when
267	    the record states one (a batch route carries ``kind: "batch"``); otherwise the
268	    dense-vs-specialized half is derived from ``would_use_dense`` exactly as before.
269	    """
...
285	    mode = route.get("eval_mode") or "?"
286	    backend = route.get("backend") or "?"
287	    kind = route.get("kind")
288	    if kind is None:
289	        kind = "dense" if route.get("would_use_dense") else "specialized"
290	    text = f"{mode}/{backend} → {kind}"
291	    annotations: List[str] = []
292	    if kind == "batch":
...
299	    if source != "evoxc-auto":
300	        annotations.append(source)
301	    if annotations:
302	        text += f" ({', '.join(annotations)})"
303	    return text
```

AFTER — replace lines 287-289 and insert the fused annotation right after
`annotations: List[str] = []` (line 291), i.e. BEFORE the `kind == "batch"` block:

```python
    kind = route.get("kind")
    if kind is None:
        # a route that names its ENGINE reports itself (``batch`` carries ``kind``, a
        # fused record carries ``engine``); a fused REQUEST reports ``fused`` even when
        # the policy declined it -- ``Route.use_fused`` is simply ``engine == "fused"``
        # (``dense_engine.py:1488-1490``), so deriving the kind from the engine alone
        # would silently print the fallback as if it were the request; the dense-vs-
        # specialized half stays derived from ``would_use_dense``
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
        # a fused request the policy DECLINED (below ``EVOGP_FUSED_MIN_POP``, a CPU
        # box, or an explicit ``numpy`` backend): the medians are the specialized
        # fallback, so say so instead of letting ``fused/iree`` imply a Triton launch
        annotations.append("declined")
```

Also update the docstring sentence at lines 266-268 to

```
    column only for a directory that recorded nothing else.  ``kind`` is a field when
    the record states one (a batch route carries ``kind: "batch"``); a fused record
    reports its ``engine`` (``kind`` is then ``fused``, annotated ``declined`` when the
    policy answered specialized); otherwise the dense-vs-specialized half is derived
    from ``would_use_dense`` exactly as before.
```

---

## 4. REQUIRED — existence proof in `worker.py`

**Why required:** the fused facade is *silent* on a decline (`FusedUnsupported` → the call site
continues on `route.fallback_backend`), and `worker.py` records no `fused_*` counter at all. So a
fused arm that never launched a kernel (a declined route, a `output_len=2` population, a box below
`EVOGP_FUSED_MIN_POP`) produces exactly the record a served one does — with a small
`first_call_ms` and, by construction, `cache.compile_misses_first_call == 0` because the fused
route performs no evoxc compile.

Live (`test/benchmark/evoxc/worker.py:778-810`) — the `cache` block brackets the *evoxc* cache for
every arm:

```python
778	        cache_before = evoxc_engine.cache_stats()
779	        batch_before = batch_stats() if batch_stats is not None else None
...
784	        probe = (
785	            BatchStatsProbe(fn, batch_stats)
786	            if is_batch and batch_stats is not None
787	            else None
788	        )
789	        timing = common.time_call(
790	            probe if probe is not None else fn,
791	            warmup=args.warmup,
792	            reps=args.reps,
793	            stats=evoxc_engine.cache_stats,
794	        )
795	        cache_after = evoxc_engine.cache_stats()
...
802	        entry["cache"] = {
803	            "before": cache_before,
804	            "after_first_call": after_first,
805	            "after": cache_after,
806	            "compile_misses_this_workload": cache_after["misses"] - cache_before["misses"],
807	            "compile_misses_first_call": (
808	                after_first["misses"] - cache_before["misses"] if after_first else None
809	            ),
810	        }
```

### 4a. Detection + accessor (insert next to the batch probe setup, after line 570)

```python
    # -- the fused (Triton) route's own counters (optional, fused arms only) ---
    # The route performs NO evoxc compile, so ``cache.compile_misses_*`` above is 0
    # by construction and cannot prove the call was served; ``fused_cache_stats()``
    # is the only surface that can (``fused_calls - fused_declines`` vs
    # ``fused_kernel_launches``).  Resolved only for a fused arm, so importing the
    # module can never break a torch/evoxc/batch arm.
    fused_arm = common.EVOXC_ARMS.get(args.arm, {}).get("eval_mode") == "fused"
    fused_stats = None
    if fused_arm:
        try:
            from evogp.tree import fused_engine as _fused_engine_probe
        except ImportError:  # noqa: F401 - the route is optional on older trees
            _fused_engine_probe = None
        fused_stats = getattr(_fused_engine_probe, "fused_cache_stats", None)
        if not callable(fused_stats):
            fused_stats = None
```

### 4b. The block builder (module level, next to `stats_keys`, after line 245)

`fused_engine.fused_cache_stats()` returns exactly 13 counters (`src/evogp/tree/fused_engine.py:317-338`).

```python
#: The fused route's 13 counters (single-sourced in ``evogp.tree.fused_engine``):
#: eight window-delta counters and five GAUGES (three table sizes + both knob
#: readings), never all-delta and never all-gauge.
FUSED_STATS_KEYS: Tuple[str, ...] = (
    "fused_calls", "fused_declines",
    "fused_xt_hits", "fused_xt_misses",
    "fused_xt_entries", "fused_xt_maxsize",
    "fused_kernel_launches", "fused_kernel_compiles",
    "fused_scratch_allocs", "fused_scratch_reuses", "fused_scratch_entries",
    "fused_block_rows", "fused_num_warps",
)
#: The ones that are GAUGES (an AFTER-state), never a delta.
FUSED_GAUGE_KEYS: Tuple[str, ...] = (
    "fused_xt_entries", "fused_xt_maxsize", "fused_scratch_entries",
    "fused_block_rows", "fused_num_warps",
)


def fused_window_block(before: Any, after: Any) -> Dict[str, Any]:
    """One before -> after FUSED window, schema-mirroring the batch blocks.

    ``counter_blocks.block`` puts EVERY one of the 13 counter keys in the block
    (``None`` where that counter was not measured -- null, never zero), states
    ``available``/``stats_fn``/``note``, and takes the window DELTA of the eight
    monotone counters while the five gauges are the AFTER snapshot.  An absent
    accessor (both snapshots ``None``) reads ``available: false`` with every counter
    null, which is what distinguishes "the route has no counters here" from "the
    route served nothing".
    """
    available = isinstance(before, dict) and isinstance(after, dict)
    values: Dict[str, Any] = {}
    for key in FUSED_STATS_KEYS:
        if key in FUSED_GAUGE_KEYS:
            values[key] = counter_blocks.integer((after or {}).get(key))
        else:
            values[key] = counter_blocks.delta_int(before, after, key)
    return counter_blocks.block(
        available=available,
        stats_fn="evogp.tree.fused_engine.fused_cache_stats",
        note=None if available else (
            "evogp.tree.fused_engine.fused_cache_stats() is unavailable on this tree: "
            "every fused counter is null (NOT zero) and the cold cost can only be read "
            "off first_call_ms"
        ),
        keys=FUSED_STATS_KEYS,
        values=values,
    )


def fused_served(block: Dict[str, Any], calls: Optional[int]) -> Optional[bool]:
    """Did ``calls`` calls actually reach a Triton launch in this window?

    ``fused_calls`` counts ATTEMPTS (a decline moves ``fused_calls`` AND
    ``fused_declines``), so the served count is their difference and it must be
    backed by ``fused_kernel_launches``.  ``None`` when a counter is missing.
    """
    attempts = block.get("fused_calls")
    declines = block.get("fused_declines")
    launches = block.get("fused_kernel_launches")
    if None in (attempts, declines, launches) or not calls:
        return None
    return int(attempts) - int(declines) == int(calls) and int(launches) >= int(calls)
```

and extend the imports at the top of `worker.py` (after line 97, `import artifact`):

```python
import counter_blocks  # noqa: E402  (torch-free number/block rule shared by counter surfaces)
```

> Rejected alternative: `_stats_delta` (line 143) treats an absent key as `0` (`_stat_value`,
> line 128-138 — deliberate for the batch tables). `counter_blocks.delta_int` is strict (missing key
> on either side → `None`), which is the "null, never zero" rule the fused block must obey.

### 4c. Record block + per-workload deltas

Record-level (insert right after the `if is_batch:` block that ends at line 696 — i.e. before the
`# -- what the policy resolves for this cell` comment at line 698):

```python
    # -- fused route: the counters that can prove the call was served ----------
    # The keys exist ONLY for a fused arm: "not a fused route" must stay
    # distinguishable from "fused route whose counters are unavailable".
    if fused_arm:
        fused_before_workloads = fused_stats() if fused_stats is not None else None
        record["fused_cache"] = {
            "available": fused_stats is not None,
            "stats_before_workloads": fused_before_workloads,
            "stats_after": None,
            "stats_keys": [],
            "totals": None,
            "served_every_workload": None,
        }
        record["notes"].append(
            "fused arm: this route compiles NOTHING through evoxc, so "
            "cache.compile_misses_first_call / compile_misses_this_workload are 0 by "
            "construction and prove nothing; the cold cost is the Triton JIT and lives "
            "in first_call_ms, while the existence proof is fused_cache "
            "(fused_kernel_launches) and each workload's cache.fused block"
        )
        if fused_stats is None:
            record["notes"].append(
                "fused arm: evogp.tree.fused_engine.fused_cache_stats() is unavailable on "
                "this tree, so every fused counter is null (NOT zero) and a served call "
                "cannot be told from a decline -- read would_use_fused in routes"
            )
```

Per workload (inside the `for name in requested:` loop): take the two extra snapshots and change the
probe, then add the block after `entry["cache"]` is built (line 810):

```python
        fused_before = fused_stats() if fused_stats is not None else None
        batch_before = batch_stats() if batch_stats is not None else None
...
        probe = (
            BatchStatsProbe(fn, batch_stats if is_batch else fused_stats)
            if (is_batch and batch_stats is not None)
            or (fused_arm and fused_stats is not None)
            else None
        )
```

```python
        if fused_arm:
            fused_after_first = probe.first if probe is not None else None
            fused_after = fused_stats() if fused_stats is not None else None
            fused_block = fused_window_block(fused_before, fused_after)
            entry["cache"]["fused"] = fused_block
            entry["cache"]["fused_after_first_call"] = fused_after_first
            entry["cache"]["fused_after"] = fused_after
            entry["cache"]["fused_first_call"] = fused_window_block(
                fused_before, fused_after_first
            )
            entry["cache"]["fused_served"] = fused_served(
                fused_block, args.reps + args.warmup + 1
            )
            workloads_served.append(entry["cache"]["fused_served"])
```

with `workloads_served: List[Optional[bool]] = []` initialised next to
`batch_first_call_misses = None` (line 772).

Close the window next to `record["engine_cache_final"] = evoxc_engine.cache_stats()` (line 1003):

```python
    if fused_arm:
        fused_after_workloads = fused_stats() if fused_stats is not None else None
        totals = fused_window_block(
            record["fused_cache"]["stats_before_workloads"], fused_after_workloads
        )
        record["fused_cache"]["stats_after"] = fused_after_workloads
        record["fused_cache"]["stats_keys"] = stats_keys(
            record["fused_cache"]["stats_before_workloads"], fused_after_workloads
        )
        record["fused_cache"]["totals"] = totals
        record["fused_cache"]["served_every_workload"] = (
            None if not workloads_served or any(v is None for v in workloads_served)
            else all(workloads_served)
        )
        if record["fused_cache"]["served_every_workload"] is False:
            record["notes"].append(
                "fused arm: at least one workload was NOT served by the Triton route "
                "(see its cache.fused_served and routes[].would_use_fused) -- its warm "
                "median is the specialized fallback, not the fused route"
            )
        elif record["fused_cache"]["served_every_workload"] is None:
            record["notes"].append(
                "fused arm: served-ness is unknown (a counter is missing), so no warm "
                "median here may be claimed as a Triton number"
            )
```

---

## 5. OPTIONAL — tooling that would price a fused column

Each of these is independent; none is needed for the arm to run or for the suite to be green.

### 5a. OPTIONAL — `test/benchmark/evoxc/cold_start.py` hard-codes the default arm

Live (`cold_start.py:198-199`):

```python
197	    # the DEFAULT route: both route knobs unset, whatever the caller exported
198	    inherited = {key: os.environ.get(key) for key in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND")}
199	    knobs = arms_mod.apply_arm_env("evoxc-auto")
```

Reason to change: the fused route's cold cost is its **Triton JIT** (375 ms warm-disk-cache to 1.38 s
cold, per `tree/CONTEXT.md`), and `--artifact-dir` is meaningless for it — so the cold-start column
is the natural place to price it, and today no flag reaches it.
Smallest patch (2 hunks): add `parser.add_argument("--arm", default="evoxc-auto",
choices=("evoxc-auto", "evoxc-fused-iree", "evoxc-fused"), help="route to price (registry arm)")`
next to `--cell` (line 504), replace line 199 with `knobs = arms_mod.apply_arm_env(args.arm)`, and
change `"route": "default (EVOGP_EVAL_MODE / EVOGP_EVOXC_BACKEND unset)"` (line 268) to
`"route": f"arm {args.arm}"`. The record's `routes` (line 291) then carries §3a's fused keys for free.

### 5b. OPTIONAL — `test/benchmark/evoxc/decompose.py` has no fused route

Live (`decompose.py:55-62`):

```python
55	#: route name -> (EVOGP_EVAL_MODE, EVOGP_EVOXC_BACKEND)
56	ROUTES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
57	    "auto": (None, None),
58	    "auto-iree": (None, "iree"),
59	    "specialized-numpy": ("specialized", "numpy"),
60	    "specialized-iree": ("specialized", "iree"),
61	    "dense-iree": ("dense", "iree"),
62	}
```

Reason to change: without an entry there is no per-phase split for the route, and `--routes fused-iree`
exits with `unknown route` (`:313-314`).
Smallest patch: add `"fused-iree": ("fused", "iree"),` to `ROUTES` and leave `PHASES`
(`:65-90`, all evoxc/dense internals) alone — its phase rows are inclusive `perf_counter_ns` timers on
named module functions, and none of them exist in `fused_engine.py`, so a fused run reports the
`plain_ms`/`cold_ms` columns plus a residual row until someone writes
`PHASES["fused_engine"] = ["_prepare", "_transposed", "launch_value", "launch_loss", "evaluate_fused",
"sr_fitness_fused"]` (a new key is inert for the other routes because `:152` only walks the two
existing keys — that line would need a third `(fused_engine, PHASES["fused_engine"])` pair too).
Note the family line (`decompose.py:324-326`) derives `dense`/`specialized` from
`route_decision(...)["would_use_dense"]`, so a fused route prints `(specialized)` today — §3a's
`engine` key is what fixes it (`family = family_of(route_decision(...))`).

### 5c. OPTIONAL — `check.py`'s expected-arm set

Live (`check.py:648-657`):

```python
648	    if args.expect_arms:
649	        expected = [arm.strip() for arm in args.expect_arms.split(",") if arm.strip()]
650	    elif configs:
651	        # The expected set is the campaign's OWN scheduled arms, never the whole
652	        # registry: appending an arm to ``common.ALL_ARMS`` is therefore backwards
653	        # compatible, because an older campaign that never scheduled it keeps gating
654	        # exactly the arms it ran instead of failing on a record that cannot exist.
655	        expected = [arm for arm in common.ALL_ARMS
656	                    if any(key[1] == arm for key in configs)]
657	        expected += sorted({key[1] for key in configs} - set(common.ALL_ARMS))
```

Reason it is optional: this already behaves correctly for a fused arm (it is `common.ALL_ARMS ∩
scheduled`), so **no edit is required**. Two facts an operator must know:
`--expect-arms` (`:745-747`) replaces that set wholesale, so a fused column gated with
`--expect-arms torch-per-tree,torch-batched,evoxc-auto,evoxc-fused-iree` demands it on every cell; and
`check_digests` (`:555-590`) FAILS on `len(distinct) != 1` unless `--allow-digest-mismatch`, because
the fused route is **not** bitwise-identical to dense/specialized (measured ≤3.8e-7 relative on the
timed populations, `tree/CONTEXT.md`'s wired-numbers paragraph). Schedule a fused arm with
`--allow-digest-mismatch` from the start, or the gate goes red on a correct measurement. The
`nan`/`finite` placement check (`:589-598`) must still agree and does.
Second-order: `check_cache` (`:324-334`) only demands a non-`None`
`cache.compile_misses_first_call`, which a fused record satisfies with a legitimate `0` — §4's note
is what stops that `0` from being misread as "nothing compiled, nothing happened".

### 5d. OPTIONAL (but land it with §2) — `ROUTE_ENV_VARS` / `DEFAULT` knob clearing

Live (`common.py:247-261`):

```python
247	DENSE_ENV_VARS: Tuple[str, ...] = (
248	    "EVOGP_DENSE_MAX_TREE_LEN",
249	    "EVOGP_DENSE_AUTO_BUDGET_MB",
250	    "EVOGP_DENSE_AUTO_MAX_ROWS",
251	)
...
257	ARTIFACT_ENV_VARS: Tuple[str, ...] = ("EVOGP_EVOXC_ARTIFACT_DIR",)
258	#: Every env var the arms clear before applying their own knob spec, so a route knob
259	#: can never leak between arms (``BATCH_ENV_VARS`` + ``DENSE_ENV_VARS`` + the disk
260	#: tier's ``ARTIFACT_ENV_VARS``).
261	ROUTE_ENV_VARS: Tuple[str, ...] = BATCH_ENV_VARS + DENSE_ENV_VARS + ARTIFACT_ENV_VARS
```

Reason: the five `EVOGP_FUSED_*` knobs are in **neither** the set nor any arm spec, so a caller's
exported `EVOGP_FUSED_BLOCK_ROWS=64` silently applies to every arm of the round (the fused arms *and*
the batch/dense/torch arms, where it is inert but unstated) and the record's
`env.applied_knobs` cannot show it — `test_no_arm_leaks_a_route_knob` is what makes the clearing
observable.
Smallest patch — one constant plus the sum:

```python
#: The FUSED route's knobs (``evogp.tree.fused_engine``): the kernel geometry, the
#: memory envelope and the transpose table's size.  ``EVOGP_FUSED_HOST_MEMO`` is
#: IMPORT-TIME (read once at module import), so clearing it per arm is required for a
#: campaign to be able to pin it at all.
FUSED_ENV_VARS: Tuple[str, ...] = (
    "EVOGP_FUSED_MIN_POP",
    "EVOGP_FUSED_BLOCK_ROWS",
    "EVOGP_FUSED_NUM_WARPS",
    "EVOGP_FUSED_MAX_SCRATCH_MB",
    "EVOGP_FUSED_HOST_MEMO",
)
#: ... and the sum gains it:
ROUTE_ENV_VARS: Tuple[str, ...] = (
    BATCH_ENV_VARS + DENSE_ENV_VARS + FUSED_ENV_VARS + ARTIFACT_ENV_VARS
)
```

`test_arm_registry.py` needs no edit for this (`test_no_arm_leaks_a_route_knob` iterates
`common.ROUTE_ENV_VARS`), but a two-line sibling assertion mirroring `test_dense_budget_knob_is_registered`
(`:125-131`) keeps it from regressing:

```python
def test_fused_knobs_are_route_knobs():
    """The Triton route's five knobs are cleared like every other route knob."""
    for name in ("EVOGP_FUSED_MIN_POP", "EVOGP_FUSED_BLOCK_ROWS", "EVOGP_FUSED_NUM_WARPS",
                 "EVOGP_FUSED_MAX_SCRATCH_MB", "EVOGP_FUSED_HOST_MEMO"):
        assert name in common.FUSED_ENV_VARS
        assert name in common.ROUTE_ENV_VARS
    assert not set(common.FUSED_ENV_VARS) & set(common.BATCH_ENV_VARS)
```

### 5e. OPTIONAL — `campaign.py --arms fused`

`campaign.py:307-323` has selectors for `all` / `torch` / `evoxc` / `batch` only, so a fused column is
`--arms evoxc-fused-iree` (a literal id list) today. To make it a named column, promote the detection
to one predicate in `common.py` next to `is_batch_arm` (`:322-329`):

```python
def is_fused_arm(arm: str) -> bool:
    """Is this arm's route the fused (Triton) engine?  ONE definition, as ``is_batch_arm``."""
    return (EVOXC_ARMS.get(arm) or {}).get("eval_mode") == "fused"
```

and add the branch after `campaign.py:321`:

```python
        elif token == "fused":
            arms.extend(a for a in common.EVOXC_ARMS if common.is_fused_arm(a) and a not in arms)
```

`worker.py` then reads `fused_arm = common.is_fused_arm(args.arm)` instead of the inline expression
in §4a — the two spellings are equivalent; the inline one is what §4a ships so that §4 stays a
one-file patch.

One honest gap to keep in mind (no patch proposed): `worker.py`'s preflight only validates the batch
mode (`:523-537`). A fused arm pointed at a tree that does not declare `fused` in `EVAL_MODES` fails
inside `resolve_eval_mode` (a `ValueError` from the policy) and lands as `status: failed` in
`campaign.json` instead of the batch route's one-paragraph `SystemExit`.

---

## 6. REQUIRED (docs) — CONTEXT.md drift

Four files, exact sentences. **Do not** edit them from here; the parents own them.

### 6a. Repository root `CONTEXT.md`

Line 4 (Intent) — says three routes and no fused engine:

```
Evaluation is a "true jit" over evoxc (EvoX Compiler) with three selectable routes (`EVOGP_EVAL_MODE`): a dense all-ops graph per population shape (default), the original per-tree-structure specialization into a cached `evoxc.build_cached` graph, and an opt-in merged-population graph built with evoxc's `batch_functions` transform; there is NO custom CUDA/HIP kernel and no compiled extension to build.
```

Replacement:

```
Evaluation is a "true jit" over evoxc (EvoX Compiler) with four selectable routes (`EVOGP_EVAL_MODE`): a dense all-ops graph per population shape (default), the original per-tree-structure specialization into a cached `evoxc.build_cached` graph, an opt-in merged-population graph built with evoxc's `batch_functions` transform, and an opt-in Triton whole-population kernel (`fused`) that is the one device-native route and the only place a kernel lives — written in Python, JIT-compiled by Triton, not a compiled extension; there is NO hand-written CUDA/HIP kernel and nothing to build.
```

Line 29 (Engine economics) — the route list and `auto`'s sentence:

```
- Engine economics: evaluation routes are selectable via `EVOGP_EVAL_MODE` — `dense` (alias `dynamic`): one compiled all-ops graph per population shape, no per-tree dispatch/readback; `specialized`: one cached `evoxc.build_cached` graph per tree structure; `batch` (opt-in, never chosen by `auto`): ONE merged graph per population structure multiset, built with evoxc's `batch_functions` transform and cached by that multiset; `auto` (default) picks by backend and scale: on a device box with `pop ≥ 64` it takes dense+iree (a compiled graph cannot amortize below that), otherwise specialized+numpy (always specialized on a CPU box and on every fallback).
```

Replacement (only the `batch` clause and the `auto` clause change):

```
- Engine economics: evaluation routes are selectable via `EVOGP_EVAL_MODE` — `dense` (alias `dynamic`): one compiled all-ops graph per population shape, no per-tree dispatch/readback; `specialized`: one cached `evoxc.build_cached` graph per tree structure; `batch` (opt-in, never chosen by `auto`): ONE merged graph per population structure multiset, built with evoxc's `batch_functions` transform and cached by that multiset; `fused` (opt-in, never chosen by `auto`): ONE Triton launch per (tree, row-block) pair over the population tensors, no host decode and no evoxc graph, with `EVOGP_FUSED_MIN_POP` (64) as its population floor; `auto` (default) picks by backend and scale: on a device box with `pop ≥ 64` it takes dense+iree (a compiled graph cannot amortize below that), otherwise specialized+numpy (always specialized on a CPU box and on every fallback) — `auto` never selects `batch` or `fused`.
```

Also line 45 (`An explicit EVOGP_EVAL_MODE=dense never gets auto's cap raise …`) still names only
`auto`/`batch` as the answer for large populations; append `or the fused route (single-output only)`
if the owned list is being touched anyway — OPTIONAL, it is not wrong today.

### 6b. `src/evogp/CONTEXT.md`

Line 5 (Intent) — “opt-in third route”, no fused engine:

```
Evaluation is a **"true jit"**: `tree/evoxc_engine.py` walks a population host-side into frozen per-tree static programs and specializes each structure into a fresh evoxc graph through `evoxc.build_cached` (memoized in one shared `evoxc.CompileCache`); `tree/dense_engine.py` is the alternative dense all-ops route; `tree/batch_engine.py` is the opt-in third route (`EVOGP_EVAL_MODE=batch`) that merges one traced graph per DISTINCT tree program into ONE executable with `evoxc.batch_functions` (static function axis, banded merge under the device binding-cap policy of `tree/batch_bands.py`); random generation, mutation and crossover are plain vectorized torch (`tree/generation.py`, `tree/splicing.py`).
```

Replacement — keep the sentence and extend its tail:

```
… (static function axis, banded merge under the device binding-cap policy of `tree/batch_bands.py`); `tree/fused_engine.py` (+ `tree/fused_kernels.py`) is the opt-in fourth route (`EVOGP_EVAL_MODE=fused`), ONE Triton launch per (tree, row-block) pair that evaluates the population from its own node tensors — the one device-native route, written in Python and JIT-compiled by Triton, so it is still no compiled extension; random generation, mutation and crossover are plain vectorized torch (`tree/generation.py`, `tree/splicing.py`).
```

Line 34 — the mode list and “`auto` never picks it”:

```
  `EVOGP_EVAL_MODE` (`dense` / `specialized` / `batch` / `auto`) and the `EVOGP_DENSE_*` family (`AUTO_MIN_POP`, …) select/tune the dense route; the `EVOGP_BATCH_*` family (`BAND_SIZE` 16, …, `COMPILE_WORKERS` 4) tunes the batch route, which only an explicit `EVOGP_EVAL_MODE=batch` selects — `auto` never picks it.
```

Replacement:

```
  `EVOGP_EVAL_MODE` (`dense` / `specialized` / `batch` / `fused` / `auto`) and the `EVOGP_DENSE_*` family (`AUTO_MIN_POP`, `AUTO_MAX_ROWS`, `AUTO_BUDGET_MB`, `AUTO_MAX_TREE_LEN`, `MAX_TREE_LEN`, `BUDGET_MB`, `UPDATE_MODE`, `ALLOW_UNSAFE_SCATTER`, `PRUNE`, `HOST_MEMO`) select/tune the dense route; the `EVOGP_BATCH_*` family (`BAND_SIZE` 16, `BANDS` explicit/`auto`/flat, `CHUNK_BANDS` off, `SWITCH_OPS`, `CACHE_SIZE` 8, `MIN_POP` 64, `ALIGNMENT`, `COMPILE_WORKERS` 4) tunes the batch route, and the `EVOGP_FUSED_*` family (`MIN_POP` 64, `BLOCK_ROWS` 256, `NUM_WARPS` 1, `MAX_SCRATCH_MB` 8192, `HOST_MEMO` 16) tunes the fused route; both are reached only by their own explicit `EVOGP_EVAL_MODE` spelling — `auto` never picks either.
```

Line 84 — “`Route.engine` ∈ `specialized` / `dense` / `batch`”:

```
- **Default route policy**: `auto` now selects dense+iree from `pop_size >= 64` up on a device box (no explicit backend) and specialized+numpy below it / on a CPU box / on `DenseUnsupported`; the resolver is `dense_engine.resolve_route` (`Route.engine` ∈ `specialized` / `dense` / `batch`, `use_dense == (engine == "dense")`), … `EVOGP_EVAL_MODE=batch` is the only way to reach the batch engine.
```

Replacement (two edits inside that bullet):

```
… the resolver is `dense_engine.resolve_route` (`Route.engine` ∈ `specialized` / `dense` / `batch` / `fused`, `use_dense == (engine == "dense")`) …
… `EVOGP_EVAL_MODE=batch` is the only way to reach the batch engine and `EVOGP_EVAL_MODE=fused` the only way to reach the fused one.
```

Everything else in this file (the batch section, the `EVOGP_BATCH_*` / `EVOGP_DENSE_*` bullets) is
current; the fused route's own policy paragraph lives one level down in `tree/CONTEXT.md`.

### 6c. `test/benchmark/evoxc/CONTEXT.md`

Arm list, line 35:

```
  `evoxc-auto` leaves both knobs unset so it times the shipped policy; `evoxc-auto-iree` differs only below the dense threshold. The batch arms are `evoxc-batch-iree` (no knob ⇒ `dispatch_formation=auto`), `-flat` (`EVOGP_BATCH_BANDS=flat`), `-aggressive`, `-default` (the PRE-CHANGE control of the merged compile) and `-artifact` (`EVOGP_EVOXC_ARTIFACT_DIR={artifact_dir}`), plus the pool pins `-artifact-p{1,2,8}` (they set `EVOGP_BATCH_COMPILE_WORKERS`, the only way a campaign can reach the pool width).
```

Replacement (append one sentence):

```
  … plus the pool pins `-artifact-p{1,2,8}` (they set `EVOGP_BATCH_COMPILE_WORKERS`, the only way a campaign can reach the pool width). The fused arms are appended AFTER that block: `evoxc-fused-iree` (`EVOGP_EVAL_MODE=fused` + `EVOGP_EVOXC_BACKEND=iree`; the backend is only the device-route spelling, the facade declines for any other name) and `evoxc-fused` (both route knobs unset, i.e. the shipped no-explicit-backend policy: fused only on a device box at `pop >= EVOGP_FUSED_MIN_POP`, specialized+numpy otherwise). Neither is a batch arm, so `--arms batch` does not pick them up.
```

Line 99 (“`EVOGP_EVAL_MODE=batch` is the third route … and `auto` never resolves to it”) — append:

```
`EVOGP_EVAL_MODE=fused` is the fourth route (`evogp/tree/fused_engine.py` + `fused_kernels.py`): ONE Triton launch per (tree, row-block) pair, single-output only, no evoxc compile at all — so a fused record's `cache.compile_misses_*` is 0 by construction, its cold cost is the Triton JIT inside `first_call_ms`, and the existence proof is `fused_cache.fused_kernel_launches`. Knobs: `EVOGP_FUSED_MIN_POP` 64, `EVOGP_FUSED_BLOCK_ROWS` 256, `EVOGP_FUSED_NUM_WARPS` 1, `EVOGP_FUSED_MAX_SCRATCH_MB` 8192, `EVOGP_FUSED_HOST_MEMO` 16 (import-time), all in `common.FUSED_ENV_VARS` / `ROUTE_ENV_VARS`. `auto` never resolves to it either.
```

Line 114 (“A new route's counters must be pointed at their own stats callable …”) — the list is
`evoxc_engine.cache_stats()` / `batch_cache_stats` / `stages.py` / `artifact.py` / `content.py`;
append this sentence:

```
… and the fused route's 13 counters come from `evogp.tree.fused_engine.fused_cache_stats`,
recorded as the per-workload `cache.fused` block and the record-level `fused_cache` total (null,
never zero).
```

Line 118 (“`cold_start.py`, `training_run.py`, `decompose.py` and `batch_economics.py` keep their OWN
route tables …”) stays true and is the reason §5a/§5b exist — append:

```
(the fused route needs at minimum `cold_start.py --arm` and a `decompose.py` `ROUTES` entry).
```

Line 127 (“`torch-batched` wins the single-call column outright at large scale … and only a fused
device-side evaluator would change that — the r6 Triton prototype … is the measured evidence”) is now
stale in its framing: the evaluator exists. Replacement tail:

```
… and only a device-side evaluator removes it: `tree/fused_engine.py` is that route now (opt-in,
single-output, wired into all five call sites; measured 38.9–153.5× faster than `torch-batched` on the
value path, 33.5–146.6× on the loss path), with the r6 Triton prototype (`../r6/SUMMARY.md`) as its
prior evidence.
```

### 6d. `src/evogp/tree/CONTEXT.md` — over the per-file delivery limit, needs a prune pass

Facts to fix the size problem (all verified in this worktree):

* `wc -c src/evogp/tree/CONTEXT.md` → **86 544 bytes** (274 lines), against the ~64 KiB per-file
  delivery limit.
* The injected copy therefore arrives **truncated**: it stops at line 220
  (`- **No pytest covers the round-2 band policy**: …`), so lines 221-274 — i.e. the whole
  `## Test Strategy` tail and the `## Routing Table` — are invisible to an agent that is told to
  "read `tree/CONTEXT.md`".
* Nothing newer than the fused route is missing from the *content*; it is a pure length problem. The
  fused sections (lines 82-107, the `EVOGP_FUSED_*` knob bullets at 133-134, and the line-240 note)
  are already accurate.

Prune plan (parent's call, in this order of payoff):

1. Push the fused route's measured detail into a new `src/evogp/tree/FUSED.md` (mirroring
   `DENSE_EVAL.md` / `BATCH_EVAL.md`): the six-cell table, the cold-JIT numbers, the 13-counter
   surface, the decline matrix and the `fused_check.py` stage list. Leave ~6 lines in `CONTEXT.md`
   (scheme + pointer to `FUSED.md`), which is where the current 22-line fused block shrinks most.
2. Move the dense economics paragraphs into `DENSE_EVAL.md` (they are already duplicated there) and
   the batch grid paragraphs into `BATCH_EVAL.md`.
3. Keep the `## Routing Table` — it is the part that must survive truncation; move it ABOVE
   `## Environment knobs` (or to the top after `## Intent`) so a truncated read still routes.

Two content fixes that ride along (both stale today):

* Line 119 — `- \`EVOGP_EVAL_MODE\` (\`auto\`) — the master route switch: \`dense\` (alias \`dynamic\`) / \`specialized\` / \`batch\` / \`auto\`; \`auto\` picks dense+iree from \`EVOGP_DENSE_AUTO_MIN_POP\` up and never selects \`batch\`.`
  → `… / \`specialized\` / \`batch\` / \`fused\` / \`auto\`; \`auto\` picks dense+iree from \`EVOGP_DENSE_AUTO_MIN_POP\` up and never selects \`batch\` or \`fused\`.`
* Line 220's note (“No pytest covers the round-2 band policy”) is still true for the *band* policy but
  should say what DOES cover the fused route: the `fused_check.py` batteries (`--stage parity|goldens|
  declines|wired|timing|tuning`) are CLIs, not pytest — i.e. the fused route has the same coverage
  shape as the band policy, and none of it is in `src/evogp/tests/`.

Cross-check for the parents: measured on this worktree, the suite currently reads
`1 failed, 1206 passed, 7 skipped, 8 warnings in 118.90s (1214 collected)` — the single failure is
§1's. After §1 it must read **1207 passed / 7 skipped** (same 1214 collected: the edit changes no
test count). The nanobind "leaked function" block printed at interpreter shutdown by the IREE
bindings is noise, not a failure.

---

## 7. Apply order & verification

1. §1 (torch-free regression pin) — standalone, makes `pytest src/evogp/tests` green.
2. §2 + §2's test edit + §5d in ONE commit (an arm without the `ROUTE_ENV_VARS` line silently
   inherits a caller's `EVOGP_FUSED_*`).
3. §3a + §3b in one commit (a record with the new keys and an unpatched renderer would print
   `fused/iree → specialized`).
4. §4 (worker existence proof) — nothing else depends on it.
5. §5a/§5b/§5e whenever a fused column is priced; §6 doc edits independently.

Verification commands (repo root, CPU only, no GPU work):

```bash
# 1. the hermetic suite (expect 1207 passed / 7 skipped, 1214 collected)
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \
  /tmp/rocvenv/bin/python -m pytest src/evogp/tests -q -p no:cacheprovider

# 2. the harness's own torch-free gate (this is the half that the registry edit turns red)
EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/root/evoxc \
  /tmp/rocvenv/bin/python -m pytest test/benchmark/evoxc -q -p no:cacheprovider

# 3. the arm registry resolves without a device
PYTHONPATH=test/benchmark/evoxc /tmp/rocvenv/bin/python - <<'PY'
import common, arms
print(common.EVOXC_ARMS["evoxc-fused-iree"])          # eval_mode=fused, backend=iree
print(list(common.EVOXC_ARMS)[-2:])                    # the two fused arms, last
print(arms.apply_arm_env("evoxc-fused-iree", common.parse_cell("p64-L64-i32-N64")))
PY

# 4. campaign dry run (no device, no compile): the fused arm appears once per cell
PYTHONPATH=test/benchmark/evoxc:/root/evoxc /tmp/rocvenv/bin/python \
  test/benchmark/evoxc/campaign.py --cells p64-L64-i32-N64 --arms evoxc-fused-iree --dry-run
```
