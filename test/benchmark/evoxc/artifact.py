"""Disk-tier counters of the batch route (``batch_artifact_stats``).

The opt-in persistent on-disk executable tier is the DENSE route's idea, reused: one compiled body per
file under ``EVOGP_EVOXC_ARTIFACT_DIR`` (the batch route keeps its own in that directory's ``batch/``
region), keyed by the chunk specs plus the resolved compile options.  A disk HIT skips ``lower`` /
``compile`` / ``load`` entirely, which is why it is the only thing in the harness that can move a
record's cold column -- so a record must be able to say whether the tier was on and how many bodies it
served.

This module is the ONE owner of that vocabulary and of the before -> after block, exactly as
``content.py`` is for the content-addressed plan-key table and ``stages.py`` for the ten-stage split.
``worker.py`` records its block as ``batch_cache.artifact_totals``.

The route-side contract is optional and getattr-probed (``evogp.tree.batch_engine.batch_artifact_stats``):
a tree that predates the tier records ``available: false`` with EVERY counter ``null`` and a note --
never a fabricated ``0``, because "the tier served nothing" and "there is no tier" must not look alike.
Nothing here imports ``evogp`` or ``torch``.

Two counter families, and they are not interchangeable: ``hits`` / ``misses`` / ``load_failures`` /
``stores`` / ``save_failures`` are RUNNING TOTALS, so a block carries the after-minus-before delta,
while ``enabled`` (is the directory set?) and ``files`` (how many artifacts the region holds) describe
the tier's STATE and are read off the after snapshot.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import counter_blocks

MODULE_NAME = "evogp.tree.batch_engine"
STATS_FN_NAME = "batch_artifact_stats"
STATS_FN_PATH = f"{MODULE_NAME}.{STATS_FN_NAME}"
RESET_FN_NAME = "batch_artifact_reset"

ENABLED_KEY = "batch_artifact_enabled"
FILES_KEY = "batch_artifact_files"
HITS_KEY = "batch_artifact_hits"
MISSES_KEY = "batch_artifact_misses"
LOAD_FAILURES_KEY = "batch_artifact_load_failures"
STORES_KEY = "batch_artifact_stores"
SAVE_FAILURES_KEY = "batch_artifact_save_failures"

#: Every counter of the surface, in block order.
COUNTER_KEYS: Sequence[str] = (
    ENABLED_KEY,
    FILES_KEY,
    HITS_KEY,
    MISSES_KEY,
    LOAD_FAILURES_KEY,
    STORES_KEY,
    SAVE_FAILURES_KEY,
)
#: Running totals (after-minus-before) as opposed to the tier's state gauges (AFTER value only).
DELTA_KEYS: Sequence[str] = (
    HITS_KEY,
    MISSES_KEY,
    LOAD_FAILURES_KEY,
    STORES_KEY,
    SAVE_FAILURES_KEY,
)
#: The counters a served body moves; what a disk hit is judged on.
SERVED_KEYS: Sequence[str] = (HITS_KEY, MISSES_KEY)

ABSENT_NOTE = (
    f"the batch route exposes no {STATS_FN_PATH}(), so every disk-tier counter is null (NOT zero): "
    "this tree has no persistent artifact tier for the merged bodies, so a cold call always compiles"
)
DISABLED_NOTE = (
    f"{ENABLED_KEY} is 0: EVOGP_EVOXC_ARTIFACT_DIR was unset for this arm, so the tier could not serve "
    "anything and a cold call is a real compile -- zero hits here is the tier being OFF, not missing"
)
WINDOW_NOTE = (
    "disk-tier window unavailable: both the before and the after snapshot must be dicts carrying "
    "batch_artifact_* counters -- a half-read window is null (NOT a partial delta)"
)


def stats_fn(module: Any) -> Optional[Callable[[], Any]]:
    """``module.batch_artifact_stats`` when it is callable, else ``None``."""
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
    note = DISABLED_NOTE if values.get(ENABLED_KEY) == 0 else None
    return _block(True, note, values)


def delta_block(before: Any, after: Any) -> Dict[str, Any]:
    """The before -> after window: running totals as deltas, ``enabled``/``files`` as AFTER values."""
    if not (counter_blocks.has_any(before, COUNTER_KEYS)
            and counter_blocks.has_any(after, COUNTER_KEYS)):
        return unavailable_block(note=WINDOW_NOTE)
    values: Dict[str, Any] = {key: counter_blocks.delta_int(before, after, key)
                              for key in DELTA_KEYS}
    for key in (ENABLED_KEY, FILES_KEY):
        values[key] = counter_blocks.integer(after.get(key))
    note = DISABLED_NOTE if values.get(ENABLED_KEY) == 0 else None
    return _block(True, note, values)


def derived(block: Any) -> Dict[str, Any]:
    """``attempts`` / ``hit_rate`` off one disk-tier block (both ``None`` when they cannot be told).

    ``attempts`` is ``hits + misses`` -- how many merged bodies the build asked the tier about -- and
    ``hit_rate`` is ``hits / attempts``.  No attempt means no rate to state, so that reads ``None``
    rather than ``0.0``: a warm call that reuses the in-memory executable never asks the disk at all.
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
    """A before -> after disk-tier window plus its derived ``attempts``/``hit_rate``."""
    block = delta_block(before, after)
    block.update(derived(block))
    return block
