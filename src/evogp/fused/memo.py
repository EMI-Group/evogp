"""In-process front tables for the FUSED evaluator's host-side work.

:mod:`evogp.fused.engine` keeps the expensive parts of a call OUT of the launch
and behind three identity-keyed tables -- the transposed data operand, the
``OUT_NODE`` verdict and the whole prepared call -- so a warm call (the same
population against the same dataset again, which is what a training loop, an
elite re-evaluation and every benchmark loop do thousands of times) is one
launch and nothing else:

``XT_MEMO``
    the ``x`` / ``inputs`` tensor, keyed on its identity (:func:`evogp.fused.engine._transposed`);
``TAG_MEMO``
    the population's ``node_type`` + ``subtree_size``, holding the ``OUT_NODE`` verdict
    (:func:`evogp.fused.engine._refuse_out_node`);
``PLAN_MEMO``
    every operand identity plus the call's scalars and resolved geometry, holding the whole
    prepared call (:func:`evogp.fused.engine._value_call` / ``_loss_call``).

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
torch ops (or :func:`evogp.fused.engine.clear_fused_cache`) after such a write.

Every table is bounded (LRU) and every table is dropped by
:func:`evogp.fused.engine.clear_fused_cache`, which is this route's ONE
invalidation entry and resets its own tables explicitly.  The plan/input tables
are deliberately SMALL (they hold decoded device payloads, not just keys): a
training loop sees a new population every generation, so a large table would
only pin memory.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Memo",
    "pin",
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


class Memo(OrderedDict):
    """Bounded LRU ``key -> value`` table with hit/miss counters.

    An ``OrderedDict`` under one ``RLock`` with ``maxsize`` enforced by
    :meth:`popitem`; the fused route keeps three of them (:data:`XT_MEMO`,
    :data:`TAG_MEMO`, :data:`PLAN_MEMO` in :mod:`evogp.fused.engine`) and resets
    them all from ``clear_fused_cache()``.  ``None`` is never stored: it is the
    "not memoized" answer of :meth:`lookup`.
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

    def note_hit(self) -> None:
        """Count a hit on an entry this table holds, served WITHOUT a lookup.

        The fused route keeps the entry it served last in a single-slot fast path
        in front of its :data:`PLAN_MEMO` (``engine._LAST``) and reports that
        hit here, so ``*_hits`` keeps meaning "a warm call that did no host work"
        instead of splitting into two counters.  Nothing is read or moved: the
        entry is already the most recently used one.
        """
        self.hits += 1

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
