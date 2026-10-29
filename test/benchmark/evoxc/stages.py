"""Per-stage cost split of one batch-route evaluation (``EVOGP_EVAL_MODE=batch``).

A batch record could say *how long* one evaluation took and *whether* it recompiled, but not *what the
time was spent on*: the merged-graph route decodes, dedupes, traces, merges, lowers, compiles, loads,
uploads, runs and gathers -- and the compile is one of those ten stages.  This module is the ONE place
that vocabulary and every derived block live, so ``batch_economics.py`` states the split without
carrying a second, subtly different key list.

The route-side contract is optional and getattr-probed: ``evogp.tree.batch_engine`` may expose
``batch_stage_stats()`` / ``batch_stage_reset()`` and ``describe_plan`` keys ``compile_workers`` /
``chunk_count`` / ``chunk_sizes``.  Nothing here imports ``evogp`` or ``torch``: it only receives
module objects and the snapshots they hand out.

**Null, never zero.**  An absent accessor, a missing key or a non-numeric value reads as ``None`` (or
``False`` for ``available``) plus a ``note`` -- never a fabricated ``0``: "this stage took no time" and
"this stage was not measured" must never look alike in a record.

**A max is not delta-able.**  ``batch_<stage>_max_ms`` is the AFTER snapshot's max, i.e. a
PROCESS-LIFETIME maximum, stated as such wherever it is recorded; only ``calls``, ``total_ms`` and the
two ``batch_compile_*`` counters are after-minus-before.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

#: The ten stages the route splits an evaluation into, in pipeline order (the compile stages follow
#: ``merge`` because the merged graph is lowered/compiled/loaded as a whole).
STAGES: Tuple[str, ...] = (
    "decode", "dedup", "trace", "merge", "lower", "compile", "load", "upload", "run", "gather",
)
MODULE_NAME = "evogp.tree.batch_engine"
STATS_FN_NAME = "batch_stage_stats"
RESET_FN_NAME = "batch_stage_reset"
STATS_FN_PATH = f"{MODULE_NAME}.{STATS_FN_NAME}"

CALL_KEYS: Tuple[str, ...] = tuple(f"batch_{stage}_calls" for stage in STAGES)
TOTAL_KEYS: Tuple[str, ...] = tuple(f"batch_{stage}_total_ms" for stage in STAGES)
MAX_KEYS: Tuple[str, ...] = tuple(f"batch_{stage}_max_ms" for stage in STAGES)
#: The compile pool's own wall time and job count (how many chunk compiles the stage ran).
COMPILE_WALL_KEY = "batch_compile_wall_ms"
COMPILE_JOBS_KEY = "batch_compile_jobs"
#: Every counter key, in block order: stage by stage (calls, total, max), then the compile pool.
COUNTER_KEYS: Tuple[str, ...] = tuple(
    key for stage in STAGES for key in (f"batch_{stage}_calls", f"batch_{stage}_total_ms",
                                        f"batch_{stage}_max_ms")
) + (COMPILE_WALL_KEY, COMPILE_JOBS_KEY)
#: The keys a block's ``None``s cover (as a set, so a consumer can diff a block against it).
UNAVAILABLE: FrozenSet[str] = frozenset(COUNTER_KEYS)
#: Integer-typed counters (everything else is a float in milliseconds).
INT_KEYS: Tuple[str, ...] = CALL_KEYS + (COMPILE_JOBS_KEY,)

ABSENT_NOTE = (
    f"the batch route exposes no {STATS_FN_PATH}(), so every stage counter is null (NOT zero): the "
    "split was not measured, which is not the same as costing nothing"
)
WINDOW_NOTE = (
    "stage window unavailable: both the before and the after snapshot must be dicts carrying "
    "batch_<stage>_* counters -- a half-read window is null (NOT a partial delta)"
)


# --- number helpers (strict: anything not a real number is None) --------------
def _number(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is not a real number (``bool`` included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> Optional[int]:
    """``value`` as an int, or ``None`` -- a missing counter never becomes ``0``."""
    number = _number(value)
    return None if number is None else int(number)


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _delta(before: Any, after: Any, key: str) -> Optional[float]:
    start, end = _number((before or {}).get(key)), _number((after or {}).get(key))
    return None if start is None or end is None else end - start


def _delta_int(before: Any, after: Any, key: str) -> Optional[int]:
    value = _delta(before, after, key)
    return None if value is None else int(value)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    """Median of the numeric entries (``None`` when there is no usable entry)."""
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return None
    mid = len(usable) // 2
    return usable[mid] if len(usable) % 2 else 0.5 * (usable[mid - 1] + usable[mid])


def _maximum(values: Sequence[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return max(usable) if usable else None


def _has_stage_keys(snapshot: Any) -> bool:
    """Does this snapshot carry at least one stage counter (i.e. is it a stage snapshot)?"""
    if not isinstance(snapshot, dict):
        return False
    return any(key in snapshot for key in COUNTER_KEYS)


# --- accessors (never raise) -------------------------------------------------
def _accessor(module: Any, name: str) -> Optional[Callable[[], Any]]:
    if module is None:
        return None
    try:
        candidate = getattr(module, name, None)
    except Exception:  # noqa: BLE001 - an accessor that raises reads as absent, never as a crash
        return None
    return candidate if callable(candidate) else None


def stage_stats_fn(module: Any) -> Optional[Callable[[], Any]]:
    """``module.batch_stage_stats`` when it is callable, else ``None``."""
    return _accessor(module, STATS_FN_NAME)


def stage_reset_fn(module: Any) -> Optional[Callable[[], Any]]:
    """``module.batch_stage_reset`` when it is callable, else ``None``."""
    return _accessor(module, RESET_FN_NAME)


# --- blocks ------------------------------------------------------------------
def _block(available: bool, note: Optional[str], values: Dict[str, Any]) -> Dict[str, Any]:
    """One block with EVERY counter key present (``None`` where the stage was not measured)."""
    block: Dict[str, Any] = {"available": bool(available), "stats_fn": STATS_FN_PATH, "note": note}
    for key in COUNTER_KEYS:
        block[key] = values.get(key)
    return block


def unavailable_block(note: Optional[str] = None) -> Dict[str, Any]:
    """The block to record when nothing could be measured: every key present, all ``None``."""
    return _block(False, note or ABSENT_NOTE, {})


def snapshot_block(after: Any) -> Dict[str, Any]:
    """One snapshot read straight off the counters (the post-``batch_stage_reset`` per-call case).

    ``max_ms`` is the process-lifetime maximum at that instant, so it describes a single call only
    when the route was reset right before it -- how :class:`StageStatsProbe` uses it.
    """
    if not _has_stage_keys(after):
        return unavailable_block(note=WINDOW_NOTE)
    values = {key: (_integer(after.get(key)) if key in INT_KEYS
                    else _rounded(_number(after.get(key)))) for key in COUNTER_KEYS}
    return _block(True, None, values)


def delta_block(before: Any, after: Any) -> Dict[str, Any]:
    """The before -> after window: ``calls``/``total_ms`` deltas, ``max_ms`` = AFTER snapshot's max.

    That max is a PROCESS-LIFETIME maximum, not a delta (a max cannot be subtracted across two
    windows).  ``calls``/``total_ms`` and the two ``batch_compile_*`` counters are after-minus-before;
    a key missing from either snapshot (or non-numeric) reads as ``None``, so a missing counter can
    never look like a free stage.
    """
    if not (_has_stage_keys(before) and _has_stage_keys(after)):
        return unavailable_block(note=WINDOW_NOTE)
    values: Dict[str, Any] = {}
    for stage in STAGES:
        calls, total, maximum = (f"batch_{stage}_calls", f"batch_{stage}_total_ms",
                                 f"batch_{stage}_max_ms")
        values[calls] = _delta_int(before, after, calls)
        values[total] = _rounded(_delta(before, after, total))
        values[maximum] = _rounded(_number(after.get(maximum)))
    values[COMPILE_WALL_KEY] = _rounded(_delta(before, after, COMPILE_WALL_KEY))
    values[COMPILE_JOBS_KEY] = _delta_int(before, after, COMPILE_JOBS_KEY)
    return _block(True, None, values)


def median_block(blocks: Sequence[Any], samples: Optional[int] = None) -> Dict[str, Any]:
    """Medians over the per-call blocks of the timed samples (``max_ms`` = max of those maxima).

    ``calls``/``total_ms``/``batch_compile_wall_ms``/``batch_compile_jobs`` are per-stage medians;
    ``max_ms`` is the largest per-call max seen in the window.  Unusable blocks (``available`` not
    true) are ignored; when none is usable every counter is ``None`` and ``available`` is false.
    """
    rows = [block for block in blocks if isinstance(block, dict) and block.get("available") is True]
    block: Dict[str, Any] = {
        "available": bool(rows), "stats_fn": STATS_FN_PATH,
        "note": None if rows else "no usable per-call stage block in the timed window",
        "samples": int(samples) if samples is not None else len(rows),
    }
    for stage in STAGES:
        calls, total, maximum = (f"batch_{stage}_calls", f"batch_{stage}_total_ms",
                                 f"batch_{stage}_max_ms")
        block[calls] = _median([_integer(row.get(calls)) for row in rows])
        block[total] = _rounded(_median([_number(row.get(total)) for row in rows]))
        block[maximum] = _rounded(_maximum([_number(row.get(maximum)) for row in rows]))
    block[COMPILE_WALL_KEY] = _rounded(_median([_number(row.get(COMPILE_WALL_KEY)) for row in rows]))
    jobs = _median([_integer(row.get(COMPILE_JOBS_KEY)) for row in rows])
    block[COMPILE_JOBS_KEY] = None if jobs is None else int(jobs)
    return block


class StageStatsProbe:
    """Wrap ``fn`` so every invocation is bracketed by a stage reset + snapshot.

    ``common.time_call`` invokes ``fn`` once untimed, then ``warmup`` times, then ``reps`` times, so
    ``blocks[0]`` is the untimed call (also kept as ``.first``) and ``blocks[1 + warmup:]`` are the
    timed samples.  The reset makes each block a PER-CALL block instead of a cumulative one; with no
    reset accessor the blocks stay cumulative and must be read as deltas.  The inner call's value is
    passed through unchanged, so ``time_call`` can use the probe as ``fn``.  A reset that raises is
    recorded in that block's ``note`` rather than aborting a measurement that may already carry a
    multi-minute compile: the failure is stated in the record, never swallowed.

    The reset + snapshot pair runs INSIDE the timed window (``time_call`` times ``fn``), so it is part
    of every number the caller records: a host-side zero/read of ~32 counters (microseconds, and only
    when the accessors exist) against a millisecond-scale call -- not free, but stated rather than
    assumed away.
    """

    def __init__(self, fn: Callable[[], Any], stats: Any = None, reset: Any = None) -> None:
        self.fn = fn
        self.stats = stats if callable(stats) else None
        self.reset = reset if callable(reset) else None
        self.blocks: List[Dict[str, Any]] = []
        self.first: Optional[Dict[str, Any]] = None

    def __call__(self) -> Any:
        note: Optional[str] = None
        if self.reset is not None:
            try:
                self.reset()
            except Exception as exc:  # noqa: BLE001 - recorded in the block, never hidden
                note = f"{RESET_FN_NAME}() failed: {exc!r}"
        out = self.fn()
        block = snapshot_block(self.stats() if self.stats is not None else None)
        if note is not None:
            block["note"] = f"{note}; {block['note']}" if block["note"] else note
        self.blocks.append(block)
        if self.first is None:
            self.first = block
        return out


def plan_fields(described: Any) -> Dict[str, Any]:
    """The merge-shape fields a ``describe_plan`` report may carry (``None`` when it does not).

    ``compile_workers`` is the compiled-graph pool width, ``chunk_count``/``chunk_sizes`` how the
    merge was split for it.  A tree whose ``describe_plan`` predates them reports ``None`` (not ``0``).
    """
    source = described if isinstance(described, dict) else {}
    return {
        "compile_workers": source.get("compile_workers"),
        "chunk_count": source.get("chunk_count"),
        "chunk_sizes": source.get("chunk_sizes"),
    }


def plan_block(facts: Any) -> Dict[str, Any]:
    """The ``stage_ms.plan`` sub-block: ``plan_fields`` off the first described plan, or unavailable.

    ``facts`` is ``worker.batch_plan_facts``'s report (compile-free introspection of
    ``BATCH_PLAN_MEMO``); ``available: false`` plus a note means the route exposes no plan
    introspection at all, and the three merge-shape fields stay ``None`` either way when the
    ``describe_plan`` report does not carry them.
    """
    facts = facts if isinstance(facts, dict) else {}
    plans = facts.get("plans") or []
    block: Dict[str, Any] = {
        "available": facts.get("available") is True,
        "note": facts.get("note"),
        "n_plans": len(plans),
    }
    block.update(plan_fields(plans[0] if plans else None))
    return block


def stage_totals(block: Any) -> Optional[Dict[str, float]]:
    """``{stage: total_ms}`` off one block (``None`` when the block measured nothing)."""
    if not isinstance(block, dict) or block.get("available") is not True:
        return None
    totals = {stage: _rounded(_number(block.get(f"batch_{stage}_total_ms"))) for stage in STAGES}
    totals = {stage: value for stage, value in totals.items() if value is not None}
    return totals or None


def _row_compiled(row: Dict[str, Any]) -> Optional[bool]:
    """Did this generation's step compile a merged graph?  (``None`` = no counter could tell.)"""
    hit = row.get("compile_hit")
    if isinstance(hit, bool):
        return not hit
    misses = _integer(row.get("batch_compile_misses"))
    if misses is not None:
        return misses > 0
    stage_ms = row.get("stage_ms")
    jobs = _integer(stage_ms.get(COMPILE_JOBS_KEY)) if isinstance(stage_ms, dict) else None
    return None if jobs is None else jobs > 0


def compile_share(generations: Sequence[Any]) -> Dict[str, Any]:
    """What share of the generation loop the COMPILE STAGE itself owns (a timer, not an upper bound).

    Numerator: the ``batch_compile_total_ms`` stage sums of the generations that carried a compile
    miss (the only generations that compile at all); denominator: the loop's total wall time, i.e.
    every row's ``total_ms``.  Unlike ``generation_summary.compile_ms_share`` -- the wall time of
    whole recompiling generations, hence an upper bound -- this is compile-attributed time only.

    Every number comes off already-recorded rows.  Rows whose compile state cannot be judged land in
    ``generations_unknown`` instead of being assumed clean, and a loop with an unreadable row reports
    ``None`` rather than a share over a partial total.
    """
    rows = [row for row in generations if isinstance(row, dict)]
    block: Dict[str, Any] = {
        "available": False, "note": None, "share": None, "compile_ms": None, "loop_ms": None,
        "generations": len(rows), "generations_with_compile": None, "generations_unknown": None,
    }
    if not rows:
        block["note"] = "no generation row was recorded, so the compile share cannot be told"
        return block
    wall = [_number(row.get("total_ms")) for row in rows]
    if any(value is None for value in wall):
        block["note"] = ("at least one generation row has no numeric total_ms, so the loop's wall "
                         "time (and therefore the share) cannot be told")
        return block
    compiled, unknown, compile_ms = 0, 0, 0.0
    for row in rows:
        state = _row_compiled(row)
        if state is None:
            unknown += 1
            continue
        if not state:
            continue
        stage_ms = row.get("stage_ms")
        value = _number(stage_ms.get("batch_compile_total_ms")) if isinstance(stage_ms, dict) else None
        if value is None:
            block["note"] = ("a generation carrying a compile miss records no batch_compile_total_ms "
                             "stage counter, so the compile share cannot be told")
            return block
        compile_ms += value
        compiled += 1
    loop_ms = float(sum(wall))
    block.update({
        "available": True, "compile_ms": _rounded(compile_ms), "loop_ms": _rounded(loop_ms),
        "generations_with_compile": compiled, "generations_unknown": unknown,
        "share": _rounded(compile_ms / loop_ms) if loop_ms > 0 else None,
    })
    if unknown and not compiled:
        # No row says a generation compiled, but none can say it did not either: a 0.0 here would
        # read as "the loop never compiled", which is exactly what we cannot claim.
        block["share"] = None
        block["note"] = (f"none of the {len(rows)} generation rows can tell whether it compiled, so "
                         "the compile share is null (NOT zero)")
    elif unknown:
        block["note"] = (f"{unknown} of {len(rows)} generation rows carry no compile counter, so the "
                         "share is computed from the rows that can tell")
    return block


def summary_block(cold_block: Any, warm_block: Any, generations: Sequence[Any]) -> Dict[str, Any]:
    """The summary's stage split: cold table, warm table and the loop's compile share.

    ``cold_stage_ms``/``warm_stage_ms`` are ``{stage: total_ms}`` read off the already-recorded
    first-call and warm-median blocks (``None`` when that block measured nothing); ``compile_share``
    comes from :func:`compile_share` over the recorded generation rows.  Nothing is asserted: an
    absent measurement is ``None`` with a ``note`` on the inner block.
    """
    return {
        "cold_stage_ms": stage_totals(cold_block),
        "warm_stage_ms": stage_totals(warm_block),
        "compile_share": compile_share(generations),
    }
