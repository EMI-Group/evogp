"""In-process front tables for the DENSE evaluator's host-side work.

:mod:`evogp.tree.dense_engine` spends most of a warm call OUTSIDE the compiled
graph: it re-derives the ``build_cached`` key, re-decodes the population
host-side, re-converts ``x``/``labels`` and re-uploads every graph input, on
every single call -- even when the caller evaluates the very same ``Forest``
against the very same dataset (which is what a training loop and every
benchmark do thousands of times).  This module holds the three front tables
that remove that work, mirroring the specialized route's
``evoxc_engine._EXECUTABLE_MEMO``:

========================== ============================================= ==========
table                      keyed on                                      owner
========================== ============================================= ==========
``EXECUTABLES``            everything ``evoxc.build_cached`` keys on     :func:`evogp.tree.dense_engine.build_dense`
``PLANS``                  the population tensors + the plan parameters  :func:`evogp.tree.dense_engine._plan_entry`
``INPUTS``                 the ``x`` / ``labels`` tensors                :func:`evogp.tree.dense_engine._resident_input`
========================== ============================================= ==========

The population/input keys are **identity keys** (:func:`token`): the tensor's
``data_ptr``, shape, dtype, device and ``_version``.  ``_version`` is bumped by
every in-place torch write, so an in-place-mutated population can never be
served from a stale entry (a half-evolved population is a miss, by
construction).  Two tensors with the same ``data_ptr`` necessarily share the
same bytes, and :func:`pin` keeps their storage alive for as long as the entry
lives, so the address can never be recycled by a *different* tensor while an
entry still points at it -- that is the ABA guard, and it costs one storage
reference per tensor instead of a full copy.

Not detected by design: a write that bypasses torch's version counter (e.g.
through a ``numpy`` view of the same memory, ``t.numpy()[...] = ...``).  Use
torch ops (or :func:`evogp.tree.evoxc_engine.clear_cache`) after such a write.

Every table is bounded (LRU) and every table is dropped by
:func:`evogp.tree.evoxc_engine.clear_cache` -- the single invalidation point of
the engine -- through :func:`clear`, which is registered there at import time.
The plan/input tables are deliberately SMALL (they hold decoded numpy blocks and
device payloads, not just keys): a training loop sees a new population every
generation, so a large table would only pin memory.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "DenseMemo",
    "clear",
    "pin",
    "register",
    "stats",
    "token",
]


def token(tensor: Any) -> Optional[Tuple[Any, ...]]:
    """Identity token of a torch tensor, or ``None`` (never memoized).

    ``(data_ptr, shape, dtype, device, _version)`` -- everything a repeat call
    needs to prove that the tensor it is about to read is the very tensor (same
    buffer, same bytes) a previous call read.  ``None`` for anything that is not
    a torch tensor (a numpy ``x`` is simply not memoized) and for a tensor whose
    identity cannot be read, so a caller can always fall back to doing the work.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return (
            int(tensor.data_ptr()),
            tuple(int(dim) for dim in tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            int(tensor._version),
        )
    except Exception:  # pragma: no cover - exotic tensor subclasses only
        return None


def pin(tensors: Sequence[Any]) -> Tuple[Any, ...]:
    """Strong references that make the tokens of ``tensors`` ABA-proof.

    Holding a tensor's storage keeps its buffer allocated, so its ``data_ptr``
    cannot be handed to a different tensor while the memo entry lives; the entry
    holds the result for exactly that long.  Returns ``()`` when a storage
    cannot be reached (the caller then keeps no pin -- the tokens are still
    correct, only their ABA guard is weaker).
    """
    pins: List[Any] = []
    for tensor in tensors:
        try:
            pins.append(tensor.untyped_storage())
        except Exception:  # pragma: no cover - exotic tensor subclasses only
            pass
    return tuple(pins)


class DenseMemo(OrderedDict):
    """Bounded LRU ``key -> value`` table with hit/miss counters.

    Deliberately the same shape as ``evoxc_engine._ExecutableMemo`` (an
    ``OrderedDict`` under one ``RLock``, ``maxsize`` enforced by
    :meth:`popitem`), because the dense route's front tables exist for the same
    reason and are invalidated by the same :func:`evogp.tree.evoxc_engine.clear_cache`.
    ``None`` is never stored: it is the "not memoized" answer of :meth:`lookup`.
    """

    def __init__(self, name: str, maxsize: int) -> None:
        super().__init__()
        self.name = str(name)
        self.maxsize = int(maxsize)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def lookup(self, key: Any) -> Any:
        """The memoized value for ``key``, or ``None`` (counter updated)."""
        with self._lock:
            if self.maxsize <= 0:  # disabled table: every lookup is a miss
                self.misses += 1
                return None
            try:
                value = self[key]
            except KeyError:
                self.misses += 1
                return None
            self.hits += 1
            self.move_to_end(key)
            return value

    def store(self, key: Any, value: Any) -> None:
        """Insert ``key`` -> ``value`` and trim the table to ``maxsize``."""
        with self._lock:
            if self.maxsize <= 0:  # disabled table: nothing is ever retained
                return
            self[key] = value
            while len(self) > self.maxsize:
                self.popitem(last=False)

    def reset(self) -> None:
        """Drop every entry and zero the counters."""
        with self._lock:
            OrderedDict.clear(self)
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, int]:
        """``<name>_hits`` / ``_misses`` / ``_entries`` / ``_maxsize``."""
        with self._lock:
            return {
                f"{self.name}_hits": self.hits,
                f"{self.name}_misses": self.misses,
                f"{self.name}_entries": len(self),
                f"{self.name}_maxsize": self.maxsize,
            }


#: every live table, in registration order (see :func:`register`)
_TABLES: List[DenseMemo] = []


def register(table: DenseMemo) -> DenseMemo:
    """Add ``table`` to the set :func:`clear` / :func:`stats` cover; returns it."""
    _TABLES.append(table)
    return table


def clear() -> None:
    """Reset every registered table (called by ``evoxc_engine.clear_cache``)."""
    for table in _TABLES:
        table.reset()


def stats() -> Dict[str, int]:
    """Merged counters of every registered table (``dense_cache_stats``)."""
    merged: Dict[str, int] = {}
    for table in _TABLES:
        merged.update(table.stats())
    return merged
