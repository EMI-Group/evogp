"""Content-addressed plan-key counters of the batch route (``batch_content_stats``).

The merged-graph route keys its compiled plans on TWO surfaces, and a benchmark record must price them
separately:

* the IDENTITY tables (``batch_cache_stats``: ``batch_exec_*`` / ``batch_plan_*``) keyed on the
  population's tensor identity tokens -- a repeated evaluation of the same ``Forest`` object hits them,
  and that is what ``stages.py`` and the existing ``batch_*`` counters already describe;
* the CONTENT table (``batch_content_stats``: ``batch_content_hits`` / ``_misses`` /
  ``_rejects`` / ``_entries`` / ``_maxsize``) keyed on the population's program digest, which answers
  what the identity tables cannot: a REBUILT ``Forest`` carrying the same content (a fresh object with
  no shared storage) hits after the one decode needed to prove it, instead of paying a full
  re-decode + re-trace + re-merge + recompile.

This module is the ONE owner of that second vocabulary and of every derived block, exactly as
``stages.py`` is for the ten-stage split.  ``batch_economics.py`` (``first_call`` / ``sr_fitness`` /
``generations[i]`` / ``summary``), ``worker.py`` (``batch_cache.content``) and ``reuse_probe.py`` record
its blocks.

The route-side contract is optional and getattr-probed (``evogp.tree.batch_engine.batch_content_stats``):
a tree that predates the content table records ``available: false`` with EVERY counter ``null`` and a
note -- never a fabricated ``0``, because "the content table served nothing" and "there is no content
table" must not look alike.  Nothing here imports ``evogp`` or ``torch``.

A gauge is not delta-able: ``batch_content_entries`` / ``batch_content_maxsize`` describe the TABLE, so a
block carries the AFTER snapshot's value for them, while ``hits`` / ``misses`` / ``rejects`` are
after-minus-before running totals.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import counter_blocks

MODULE_NAME = "evogp.tree.batch_engine"
STATS_FN_NAME = "batch_content_stats"
STATS_FN_PATH = f"{MODULE_NAME}.{STATS_FN_NAME}"

HITS_KEY = "batch_content_hits"
MISSES_KEY = "batch_content_misses"
REJECTS_KEY = "batch_content_rejects"
ENTRIES_KEY = "batch_content_entries"
MAXSIZE_KEY = "batch_content_maxsize"

#: Every counter of the surface, in block order.
COUNTER_KEYS: Sequence[str] = (HITS_KEY, MISSES_KEY, REJECTS_KEY, ENTRIES_KEY, MAXSIZE_KEY)
#: Running totals (after-minus-before) as opposed to the table gauges (AFTER value only).
DELTA_KEYS: Sequence[str] = (HITS_KEY, MISSES_KEY, REJECTS_KEY)
#: The counters a served entry moves; what a content hit is judged on.
SERVED_KEYS: Sequence[str] = (HITS_KEY, MISSES_KEY)

ABSENT_NOTE = (
    f"the batch route exposes no {STATS_FN_PATH}(), so every content counter is null (NOT zero): this "
    "tree keys merged plans on population identity only, and a rebuilt-but-equal population re-decodes, "
    "re-traces and re-merges"
)
WINDOW_NOTE = (
    "content window unavailable: both the before and the after snapshot must be dicts carrying "
    "batch_content_* counters -- a half-read window is null (NOT a partial delta)"
)


def stats_fn(module: Any) -> Optional[Callable[[], Any]]:
    """``module.batch_content_stats`` when it is callable, else ``None``."""
    return counter_blocks.accessor(module, STATS_FN_NAME)


def _block(available: bool, note: Optional[str], values: Dict[str, Any]) -> Dict[str, Any]:
    return counter_blocks.block(available, STATS_FN_PATH, note, COUNTER_KEYS, values)


def unavailable_block(note: Optional[str] = None) -> Dict[str, Any]:
    """The block to record when nothing could be measured: every key present, all ``None``."""
    return _block(False, note or ABSENT_NOTE, {})


def snapshot_block(after: Any) -> Dict[str, Any]:
    """One snapshot read straight off the counters (no window: every counter is the stored value)."""
    if not counter_blocks.has_any(after, COUNTER_KEYS):
        return unavailable_block(note=WINDOW_NOTE)
    values = {key: counter_blocks.integer(after.get(key)) for key in COUNTER_KEYS}
    return _block(True, None, values)


def delta_block(before: Any, after: Any) -> Dict[str, Any]:
    """The before -> after window: ``hits``/``misses``/``rejects`` deltas, gauges = AFTER value."""
    if not (counter_blocks.has_any(before, COUNTER_KEYS)
            and counter_blocks.has_any(after, COUNTER_KEYS)):
        return unavailable_block(note=WINDOW_NOTE)
    values: Dict[str, Any] = {key: counter_blocks.delta_int(before, after, key)
                              for key in DELTA_KEYS}
    for key in (ENTRIES_KEY, MAXSIZE_KEY):
        values[key] = counter_blocks.integer(after.get(key))
    return _block(True, None, values)


def derived(block: Any) -> Dict[str, Any]:
    """``attempts`` / ``hit_rate`` off one content block (both ``None`` when they cannot be told).

    ``attempts`` is ``hits + misses`` -- the number of times the content key was consulted at all --
    and ``hit_rate`` is ``hits / attempts``.  A block with no attempt (``attempts == 0``) has no rate to
    state, so it reads ``None`` rather than ``0.0``: the identity fast path answering a call never
    reaches the content table, and that is not a content miss.
    """
    block = block if isinstance(block, dict) else {}
    hits = counter_blocks.integer(block.get(HITS_KEY))
    misses = counter_blocks.integer(block.get(MISSES_KEY))
    if hits is None or misses is None:
        return {"attempts": None, "hit_rate": None}
    attempts = hits + misses
    return {
        "attempts": attempts,
        "hit_rate": counter_blocks.rounded(hits / attempts) if attempts else None,
    }


def window_block(before: Any, after: Any) -> Dict[str, Any]:
    """A before -> after content window plus its derived ``attempts``/``hit_rate``."""
    block = delta_block(before, after)
    block.update(derived(block))
    return block


def row_state(row: Any) -> Optional[str]:
    """What one generation/call row says the content table did.

    ``"hit"`` (an entry was served), ``"miss"`` (consulted, nothing served), ``"no_attempt"`` (the
    counters exist but did not move -- the identity fast path answered) and ``"unknown"`` when the row
    carries no usable content counters.  Read off the RECORDED row, never re-derived from the engine.
    """
    block = row.get("content") if isinstance(row, dict) else None
    if not isinstance(block, dict) or block.get("available") is not True:
        return "unknown"
    hits = counter_blocks.integer(block.get(HITS_KEY))
    misses = counter_blocks.integer(block.get(MISSES_KEY))
    if hits is None or misses is None:
        return "unknown"
    if hits > 0:
        return "hit"
    return "miss" if misses > 0 else "no_attempt"


def summary_block(before: Any, after: Any, rows: Sequence[Any] = ()) -> Dict[str, Any]:
    """The summary's content block: the whole window plus what the recorded rows say.

    ``window`` is the before -> after delta over the WHOLE measurement (cold call plus warm calls plus
    the generation loop); ``generations`` counts the recorded rows by :func:`row_state` outcome, so a
    reader sees how many generations had a content decision to make at all.  A row whose counters are
    absent lands in ``unknown`` instead of being assumed to have missed.
    """
    block = window_block(before, after)
    states = [row_state(row) for row in rows if isinstance(row, dict)]
    block["generations"] = {
        "rows": len(states),
        "with_content_hit": sum(1 for state in states if state == "hit"),
        "with_content_miss": sum(1 for state in states if state == "miss"),
        "no_attempt": sum(1 for state in states if state == "no_attempt"),
        "unknown": sum(1 for state in states if state == "unknown"),
    }
    return block
