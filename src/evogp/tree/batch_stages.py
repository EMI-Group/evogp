"""Per-stage timing surface of the opt-in batch route (measurement only).

The batch route's cost is spread over stages whose ratio is not obvious from the
outside: host decode, dedup, one trace per distinct program, the merge, then the
compile staging (``lower`` / ``compile`` / ``load``), the input upload, the single
``evoxc.run`` and the readback.  This module is the ONE place those are timed, so
``batch_engine.batch_stage_stats()`` can answer "what fraction of a cold call is
compile?" without cProfile and without changing what the route computes.

Surface
-------
* :data:`STAGE_NAMES` -- the ten stage names (the order is also the report order).
* :class:`StageTimer` -- a LOCAL accumulator: ``.stage(name)`` is a context
  manager that adds one measurement to ``timer.totals``, ``.fold()`` merges this
  timer into the module-global accumulator (one lock acquisition), and
  ``.stage_ms`` is the measured-subset snapshot a record stores.
* :func:`stage` -- the calling thread's context manager; it measures and folds one
  measurement immediately (the main thread uses this, so no timer has to be
  threaded through every call site).
* :func:`fold` / :func:`record` -- fold a whole timer / one measurement.
* :func:`record_compile` -- the compile-region counters
  (``batch_compile_wall_ms`` / ``batch_compile_jobs``), which are NOT a stage: the
  wall clock of the whole (parallel) lower/compile/load region and how many chunk
  bodies it actually built (0 when every chunk was a memo hit).
* :func:`stage_stats` -- exactly 32 keys (3 per stage + the 2 compile keys).
* :func:`stage_reset` -- zero all 32.

Threading constraint
-------------------
The module-global accumulator is guarded by ONE module ``threading.Lock``.  The
intended use is:

* the calling (main) thread measures with :func:`stage` / :func:`record`, each of
  which takes the lock for the duration of one dict update;
* a pool worker builds its OWN :class:`StageTimer` (thread-local by construction:
  never share a timer between threads) and calls ``.fold()`` once when its job
  ends, so a parallel compile pays one lock acquisition per job, not one per
  stage.

Never hold the lock while doing real work, and never fold the same timer twice
(the totals would double-count).  Every measurement uses ``time.perf_counter()``;
the counters are diagnostic only -- nothing in the route's numerics, ordering or
caching depends on them.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple, Union

__all__ = [
    "STAGE_NAMES",
    "StageTimer",
    "StageTotal",
    "fold",
    "record",
    "record_compile",
    "stage",
    "stage_reset",
    "stage_stats",
]

#: The ten measured stages, in report order.
STAGE_NAMES: Tuple[str, ...] = (
    "decode",
    "dedup",
    "trace",
    "merge",
    "lower",
    "compile",
    "load",
    "upload",
    "run",
    "gather",
)
#: The three compile staging stages (built per chunk, see :func:`record_compile`).
COMPILE_STAGES: Tuple[str, ...] = ("lower", "compile", "load")

Stats = Dict[str, Union[float, int]]


class StageTotal(float):
    """One stage's accumulated total, in MILLISECONDS, carrying its counters.

    A ``float`` (so ``timer.totals["trace"] > 0`` works) that also answers
    ``totals["trace"]["total_ms"]`` / ``["calls"]`` / ``["max_ms"]`` (so a caller
    that wants the counters need not know the snapshot's shape) and exposes them
    as attributes ``calls`` / ``max_ms`` / ``total_ms``.
    """

    __slots__ = ("calls", "max_ms")

    def __new__(cls, total_ms: float = 0.0, calls: int = 0,
                max_ms: float = 0.0) -> "StageTotal":
        total = super().__new__(cls, float(total_ms))
        total.calls = int(calls)
        total.max_ms = float(max_ms)
        return total

    def __getitem__(self, key: str) -> Union[float, int]:
        if key == "total_ms":
            return float(self)
        if key == "calls":
            return self.calls
        if key == "max_ms":
            return self.max_ms
        raise KeyError(key)

    @property
    def total_ms(self) -> float:
        """The accumulated total in milliseconds (the ``float`` value itself)."""
        return float(self)

    def as_dict(self) -> Dict[str, Union[float, int]]:
        """``{"calls", "total_ms", "max_ms"}`` of this stage."""
        return {"calls": self.calls, "total_ms": float(self), "max_ms": self.max_ms}


def _check_name(name: str) -> str:
    """Reject an unknown stage name (a typo must not create a silent stage)."""
    text = str(name)
    if text not in STAGE_NAMES:
        raise ValueError(
            f"unknown batch stage {name!r}: batch_stages supports "
            f"{list(STAGE_NAMES)}"
        )
    return text


# ---------------------------------------------------------------------------
# the module-global accumulator
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
#: ``name -> [calls, total_ms, max_ms]``; only touched under :data:`_LOCK`.
_TOTALS: Dict[str, List[float]] = {name: [0, 0.0, 0.0] for name in STAGE_NAMES}
#: Wall clock of the whole compile/load region of the last calls (summed).
_COMPILE_WALL_MS: float = 0.0
#: Chunk bodies actually built (a memo hit counts 0).
_COMPILE_JOBS: int = 0


def _accumulate(rows: Dict[str, Tuple[int, float, float]]) -> None:
    """Add ``{name: (calls, total_ms, max_ms)}`` to the global accumulator."""
    with _LOCK:
        for name, (calls, total_ms, max_ms) in rows.items():
            row = _TOTALS[name]
            row[0] += int(calls)
            row[1] += float(total_ms)
            if float(max_ms) > row[2]:
                row[2] = float(max_ms)


def record(name: str, elapsed_ms: float) -> None:
    """Fold ONE measurement of ``name`` into the global accumulator."""
    stage_name = _check_name(name)
    _accumulate({stage_name: (1, float(elapsed_ms), float(elapsed_ms))})


def record_compile(wall_ms: float, jobs: int) -> None:
    """Fold one compile REGION into ``batch_compile_wall_ms`` / ``_jobs``.

    Called once per :func:`evogp.tree.batch_chunks.build_bodies` call (once per
    build, not once per chunk): ``wall_ms`` is the wall clock of the whole region
    (so it is the number parallel workers reduce) and ``jobs`` is how many chunk
    bodies that region actually built (0 for an all-memo-hit build).
    """
    global _COMPILE_WALL_MS, _COMPILE_JOBS
    with _LOCK:
        _COMPILE_WALL_MS += float(wall_ms)
        _COMPILE_JOBS += int(jobs)


class StageTimer:
    """A LOCAL stage accumulator (one per thread, folded once).

    ``.stage(name)`` times one block and adds it to this timer only, so a timer
    can be used by a pool worker without touching the module lock; ``.fold()``
    publishes everything it holds in one lock acquisition.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, List[float]] = {name: [0, 0.0, 0.0] for name in STAGE_NAMES}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the wrapped block into this timer (measured even if it raises)."""
        row = self._rows[_check_name(name)]
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            row[0] += 1
            row[1] += elapsed
            if elapsed > row[2]:
                row[2] = elapsed

    @property
    def totals(self) -> Dict[str, StageTotal]:
        """A read-only snapshot ``{stage: StageTotal}`` of every stage name."""
        return {
            name: StageTotal(row[1], int(row[0]), row[2])
            for name, row in self._rows.items()
        }

    @property
    def stage_ms(self) -> Dict[str, float]:
        """``{stage: total ms}`` of the stages this timer actually MEASURED.

        The unmeasured stages are omitted (a plan built from an already-decoded
        population has no ``decode`` time of its own), so the dict is exactly the
        set of blocks this call paid for.
        """
        return {name: row[1] for name, row in self._rows.items() if row[0]}

    def fold(self) -> None:
        """Publish this timer's totals into the module accumulator (once!)."""
        _accumulate(
            {
                name: (int(row[0]), row[1], row[2])
                for name, row in self._rows.items()
                if row[0]
            }
        )


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one block in the CALLING thread and fold it immediately."""
    stage_name = _check_name(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000.0
        _accumulate({stage_name: (1, elapsed, elapsed)})


def fold(timer: StageTimer) -> None:
    """Publish a whole :class:`StageTimer` (the pool-worker contract)."""
    timer.fold()


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def stage_stats() -> Stats:
    """Exactly 32 keys: per stage ``calls`` / ``total_ms`` / ``max_ms`` + compile.

    ``batch_<stage>_calls`` (int), ``batch_<stage>_total_ms`` (float) and
    ``batch_<stage>_max_ms`` (float) for the ten :data:`STAGE_NAMES`, plus
    ``batch_compile_wall_ms`` (float, the whole parallel lower/compile/load region)
    and ``batch_compile_jobs`` (int, chunk bodies actually built; 0 when every
    chunk was a memo hit).  Deliberately NOT folded into
    :func:`evogp.tree.batch_engine.batch_cache_stats` (8 pinned keys) nor into
    ``cache_stats()`` / ``dense_cache_stats()`` / ``artifact_cache_stats()``.
    """
    with _LOCK:
        totals = {name: (int(row[0]), row[1], row[2]) for name, row in _TOTALS.items()}
        wall = float(_COMPILE_WALL_MS)
        jobs = int(_COMPILE_JOBS)
    stats: Stats = {}
    for name in STAGE_NAMES:
        calls, total_ms, max_ms = totals[name]
        stats[f"batch_{name}_calls"] = calls
        stats[f"batch_{name}_total_ms"] = float(total_ms)
        stats[f"batch_{name}_max_ms"] = float(max_ms)
    stats["batch_compile_wall_ms"] = wall
    stats["batch_compile_jobs"] = jobs
    return stats


def stage_reset() -> None:
    """Zero all 32 counters (the caches are untouched -- this is timing only)."""
    global _COMPILE_WALL_MS, _COMPILE_JOBS
    with _LOCK:
        for row in _TOTALS.values():
            row[0] = 0
            row[1] = 0.0
            row[2] = 0.0
        _COMPILE_WALL_MS = 0.0
        _COMPILE_JOBS = 0
