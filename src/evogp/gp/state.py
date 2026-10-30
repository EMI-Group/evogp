"""The functional GP state: one dataclass whose leaves are evoxc tensors.

Follows the Evox 2.0 idiom (see ``CONTEXT.md``): *state* is a dataclass of
concrete ``evoxc`` tensors, *config* is a frozen dataclass of static Python
values (:mod:`evogp.gp.config`), and every transition is a plain module-level
function ``f(config, state, ...) -> state``.  There is no class hierarchy and no
hidden state.

**Every leaf shape is constant for a whole run.**
``opcode``/``payload``/``size`` are ``[pop_size, max_nodes]`` and
``fitness``/``key`` are ``[pop_size]``/rank-``key_rank`` for every generation;
only *values* change.  That is what lets one traced graph serve a whole run --
a shape change forces a re-trace and a re-compile, so a run whose shape varies
(e.g. by re-sizing the population) is out of contract by construction.

Channels (prefix / pre-order encoding, node 0 = root, padded past the row's
live node count), replacing the tree layer's
``(value float32, type int16, size int16)`` triple:

* ``opcode`` -- int32: the :class:`~evogp.gp._constants.Func` id for a function
  node, or the :class:`~evogp.gp._constants.NType` leaf marker (``VAR`` 0 /
  ``CONST`` 1) for a leaf.  A leaf is recognised by ``size == 1`` -- a function
  node always has at least one child, so its subtree size is >= 2 -- which is
  what disambiguates ``opcode == 0`` (``VAR``) from ``Func.IF`` and
  ``opcode == 1`` (``CONST``) from ``Func.ADD``.
* ``payload`` -- float32: the constant value (``CONST`` leaf) or the variable
  index (``VAR`` leaf); always 0.0 on function nodes (and in the padding).
* ``size`` -- int32: the subtree node count of every node; ``size[:, 0]`` is the
  row's real length and the padding is 0.

The tree layer packs ``(index << 16) | function_id`` into a float bit pattern;
evoxc has no ``bitcast``, which is exactly why the payload is a separate float
channel and the opcode an int channel here.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import evoxc

__all__ = [
    "GPState",
    "replace",
    "channels",
    "population_size",
    "max_nodes",
    "UNEVALUATED",
    "CHANNEL_NAMES",
]

#: The three encoding channels, in the order ``channels()`` returns them.
CHANNEL_NAMES = ("opcode", "payload", "size")


@dataclasses.dataclass(frozen=True)
class GPState:
    """One generation of a GP population (leaves are ``evoxc`` tensors).

    Args:
        opcode: int32 ``[pop_size, max_nodes]`` -- function id or leaf marker.
        payload: float32 ``[pop_size, max_nodes]`` -- const value / var index.
        size: int32 ``[pop_size, max_nodes]`` -- subtree sizes, 0 in the padding.
        fitness: float32 ``[pop_size]`` -- MAXIMIZED fitness of the current
            generation (losses negated); ``-inf`` marks an unevaluated
            individual.
        key: the RNG key of the *next* generation (rank-0 for the default
            ``splitmix64`` algorithm); ``split``/``split_n`` it inside a trace
            to draw.
    """

    opcode: "evoxc.Tensor"
    payload: "evoxc.Tensor"
    size: "evoxc.Tensor"
    fitness: "evoxc.Tensor"
    key: "evoxc.Tensor"

    def channels(self) -> Tuple["evoxc.Tensor", "evoxc.Tensor", "evoxc.Tensor"]:
        """``(opcode, payload, size)`` -- the encoding-only view."""
        return (self.opcode, self.payload, self.size)


def channels(state) -> Tuple["evoxc.Tensor", "evoxc.Tensor", "evoxc.Tensor"]:
    """``(opcode, payload, size)`` of a state (a bare triple is passed through)."""
    if isinstance(state, GPState):
        return state.channels()
    opcode, payload, size = state
    return opcode, payload, size


def replace(state: GPState, **changes: Any) -> GPState:
    """A new :class:`GPState` with ``changes`` applied (``dataclasses.replace``).

    Kept as a named function so callers do not have to import ``dataclasses``
    (and so the dataclass can grow a field without touching every call site).
    """
    return dataclasses.replace(state, **changes)


def population_size(state: GPState) -> int:
    """``pop_size`` from the channel shape (a shape read, never a host sync)."""
    return int(state.opcode.shape[0])


def max_nodes(state: GPState) -> int:
    """``max_nodes`` from the channel shape (a shape read, never a host sync)."""
    return int(state.opcode.shape[1])


#: Fitness of an individual that has not been evaluated yet.
UNEVALUATED: float = -math.inf
