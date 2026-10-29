"""Strict number/block helpers shared by the batch route's optional counter surfaces.

The batch route exposes more than one optional counter contract, and every one of them is recorded the
same way: a before -> after snapshot window turned into a block that states ``available``, the accessor
it came from, a ``note`` and the counters themselves.  ``stages.py`` owns the ten-stage cost split and
``content.py`` owns the content-addressed plan-key table; this module owns the rule they both follow and
the arithmetic behind it, so neither surface keeps a second, subtly different copy.

**Null, never zero.**  A missing key, a non-numeric value or a non-boolean flag reads as ``None`` /
``False`` -- never as a fabricated ``0``: "this counter did not move" and "this counter was not measured"
must never look alike in a record.

Nothing here imports ``torch``, ``evogp`` or either vocabulary: a block is built from plain dicts, key
tuples and an accessor path handed in by the caller.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple


# --- number helpers (strict: anything not a real number is None) --------------
def number(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is not a real number (``bool`` included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def integer(value: Any) -> Optional[int]:
    """``value`` as an int, or ``None`` -- a missing counter never becomes ``0``."""
    value_as_number = number(value)
    return None if value_as_number is None else int(value_as_number)


def rounded(value: Optional[float]) -> Optional[float]:
    """``value`` rounded for a record (``None`` stays ``None``)."""
    return None if value is None else round(float(value), 4)


def delta(before: Any, after: Any, key: str) -> Optional[float]:
    """``after[key] - before[key]``; ``None`` when either side does not carry the counter."""
    start, end = number((before or {}).get(key)), number((after or {}).get(key))
    return None if start is None or end is None else end - start


def delta_int(before: Any, after: Any, key: str) -> Optional[int]:
    """:func:`delta` as an int (``None`` when either side is unusable)."""
    value = delta(before, after, key)
    return None if value is None else int(value)


def median(values: Sequence[Optional[float]]) -> Optional[float]:
    """Median of the numeric entries (``None`` when there is no usable entry)."""
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return None
    mid = len(usable) // 2
    return usable[mid] if len(usable) % 2 else 0.5 * (usable[mid - 1] + usable[mid])


def maximum(values: Sequence[Optional[float]]) -> Optional[float]:
    """Largest numeric entry (``None`` when there is none)."""
    usable = [value for value in values if value is not None]
    return max(usable) if usable else None


# --- snapshots and blocks ----------------------------------------------------
def has_any(snapshot: Any, keys: Sequence[str]) -> bool:
    """Does this snapshot carry at least one of ``keys`` (i.e. is it THIS surface's snapshot)?"""
    if not isinstance(snapshot, dict):
        return False
    return any(key in snapshot for key in keys)


def accessor(module: Any, name: str) -> Optional[Callable[[], Any]]:
    """``module.<name>`` when it is callable, else ``None`` (never raises)."""
    if module is None:
        return None
    try:
        candidate = getattr(module, name, None)
    except Exception:  # noqa: BLE001 - an accessor that raises reads as absent, never as a crash
        return None
    return candidate if callable(candidate) else None


def block(
    available: bool,
    stats_fn: str,
    note: Optional[str],
    keys: Sequence[str],
    values: Dict[str, Any],
) -> Dict[str, Any]:
    """One block with EVERY counter key present (``None`` where that counter was not measured)."""
    built: Dict[str, Any] = {"available": bool(available), "stats_fn": stats_fn, "note": note}
    for key in keys:
        built[key] = values.get(key)
    return built


def keys_of(*snapshots: Any) -> Tuple[str, ...]:
    """Sorted counter names of the FIRST snapshot that has any (``()`` when none does).

    A record must state which counters the route actually exposes: that is what makes a null delta
    checkable ("the key was absent") instead of ambiguous.
    """
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and snapshot:
            return tuple(sorted(str(key) for key in snapshot))
    return ()
