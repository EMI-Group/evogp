"""Subtree crossover of the evoxc GP stack: the splice, by index arithmetic.

The semantics are the deleted tree layer's ``splicing.py::_splice`` /
``ENGINE_SPEC.md`` §7, reproduced verbatim (both are *reference*
implementations, not something to re-invent):

Replacing the subtree rooted at ``pos`` of a target row (real length
``old_len``, subtree size ``old_sub``) by the donor subtree of size ``new_sub``
rooted at ``donor_pos``:

1. ``size_diff = new_sub - old_sub`` and ``old_offset = pos + old_sub``;
2. nodes ``[0, pos)`` of the target are kept (every **strict ancestor** of
   ``pos`` grows by ``size_diff``: ``j < pos`` and ``j + size[j] > pos``);
3. the donor columns ``[donor_pos, donor_pos + new_sub)`` are copied verbatim to
   ``[pos, pos + new_sub)`` -- a pre-order subtree is contiguous, so no remap
   happens inside the copied block;
4. the target tail ``[old_offset, old_len)`` moves to
   ``[pos + new_sub, old_len + size_diff)`` (``size[j]`` unchanged, read at the
   old column);
5. a row whose result would exceed ``max_nodes`` is left untouched, verbatim.

Everything is expressed in static shapes over ``j = arange(max_nodes)`` -- three
index maps and three ``select``s, no scan, no dynamic shape:

``old_col   = where(j < pos + new_sub, j, j - size_diff)``  (target side)
``donor_col = donor_pos + j - pos``                        (donor side, clamped)
``in_block  = (pos <= j) & (j < pos + new_sub)``

The legacy kernel read ``left_indices`` out of bounds (undefined behaviour) and
left an out-of-range position silently on the "bad pos" path; here an index
outside ``[0, pop_size)`` raises ``ValueError`` (:func:`evogp.gp._runtime.index_array`)
and positions are drawn inside ``[0, size[row, 0])`` by construction, so the
undefined paths are unreachable.

Public surface
--------------
``crossover(parents, key, ...)``
    One traced graph per ``(pop_size, max_nodes, rows, prob, backend, device)``:
    draw the donor rows and both splice positions, splice, return the offspring
    population (``fitness`` all ``-inf``: an operator never evaluates).
``splice(config, ...)``
    The trace-time primitive ``_splice_traced`` re-exported for the sibling
    modules (``mutation``'s subtree replacement reuses it with ``donor_pos = 0``)
    and for the semantics tests that need explicit positions.
``draw_positions(size, donor_size, key, ...)``
    The position draw on its own (a memoized graph), so the uniform-over-nodes
    property can be measured directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _runtime, encoding
from .state import GPState

__all__ = [
    "crossover",
    "splice",
    "draw_positions",
    "cache_stats",
    "clear_cache",
]

#: ``(static configuration) -> (graph closure, executable)``.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the crossover layer (graph memo + shared compile cache)."""
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop every memoized crossover graph."""
    _runtime.reset(_EXECUTABLE_MEMO)


# --------------------------------------------------------------------------- #
# the splice (shared with mutation's subtree replacement)
# --------------------------------------------------------------------------- #
def _splice_traced(evoxc_module, target, donor, pos, donor_pos, gate=None):
    """Splice one donor subtree into every target row (trace-time).

    Args:
        evoxc_module: the imported ``evoxc`` module.
        target: ``(opcode, payload, size)`` of the target population, each
            ``[pop_size, max_nodes]``.
        donor: the same triple for the donor population (row ``i`` of the donor
            belongs to row ``i`` of the target).
        pos: ``int32 [pop_size]`` subtree root inside the target row.
        donor_pos: ``int32 [pop_size]`` donor subtree root.
        gate: ``bool [pop_size]`` or ``None`` (apply to every row).

    Returns:
        Three symbolic channels, cast back to ``(int32, float32, int32)``.

    Notes:
        Every index is clamped into the row before it is used, and a row that is
        not ``gate``-selected, does not fit ``max_nodes``, or whose position is
        out of range is restored verbatim from ``target`` -- so a malformed or
        overflowing splice can never produce a corrupt row.
    """
    opcode, payload, size = target
    donor_opcode, donor_payload, donor_size = donor
    pop_size, width = int(opcode.shape[0]), int(opcode.shape[1])
    last = width - 1

    cols = _runtime.columns(evoxc_module, width)  # [1, width] int32
    pos_c = evoxc_module.reshape(pos, (pop_size, 1))
    donor_c = evoxc_module.reshape(donor_pos, (pop_size, 1))

    # sizes: the replaced block's old and new extent, and the tail shift.
    old_sub = _runtime.row_values(evoxc_module, size, pos)  # [pop, 1]
    new_sub = _runtime.row_values(evoxc_module, donor_size, donor_pos)
    size_diff = evoxc_module.subtract(new_sub, old_sub)
    old_len = size[:, 0:1]
    new_len = evoxc_module.add(old_len, size_diff)
    block_end = evoxc_module.add(pos_c, new_sub)

    in_block = evoxc_module.logical_and(
        evoxc_module.greater_equal(cols, pos_c), evoxc_module.less(cols, block_end)
    )
    # Where the target side of every output column comes from: the prefix at
    # its own column, the (shifted) tail at `j - size_diff`.
    old_col = evoxc_module.select(
        evoxc_module.less(cols, block_end), cols,
        evoxc_module.subtract(cols, size_diff),
    )
    old_col = evoxc_module.clamp(old_col, 0, last)
    donor_col = evoxc_module.clamp(
        evoxc_module.add(evoxc_module.subtract(cols, pos_c), donor_c), 0, last
    )

    spliced_opcode = evoxc_module.select(
        in_block,
        _runtime.take_per_row(evoxc_module, donor_opcode, donor_col),
        _runtime.take_per_row(evoxc_module, opcode, old_col),
    )
    spliced_payload = evoxc_module.select(
        in_block,
        _runtime.take_per_row(evoxc_module, donor_payload, donor_col),
        _runtime.take_per_row(evoxc_module, payload, old_col),
    )

    # Every strict ancestor of `pos` grows by size_diff; the block takes the
    # donor's own sizes; the tail keeps its own (read at its old column).
    ancestors = evoxc_module.logical_and(
        evoxc_module.less(cols, pos_c),
        evoxc_module.greater(evoxc_module.add(cols, size), pos_c),
    )
    spliced_size = evoxc_module.select(
        ancestors,
        evoxc_module.add(size, size_diff),
        _runtime.take_per_row(evoxc_module, size, old_col),
    )
    spliced_size = evoxc_module.select(
        in_block,
        _runtime.take_per_row(evoxc_module, donor_size, donor_col),
        spliced_size,
    )

    # Rows that overflow (or are gated off) are the target row, verbatim.
    keep = evoxc_module.logical_and(
        evoxc_module.less(cols, new_len),
        evoxc_module.less_equal(new_len, _runtime.i32(evoxc_module, width)),
    )
    if gate is not None:
        keep = evoxc_module.logical_and(
            keep, evoxc_module.reshape(gate, (pop_size, 1))
        )

    return (
        evoxc_module.cast(
            evoxc_module.select(keep, spliced_opcode, opcode), evoxc_module.int32
        ),
        evoxc_module.cast(
            evoxc_module.select(keep, spliced_payload, payload), evoxc_module.float32
        ),
        evoxc_module.cast(
            evoxc_module.select(keep, spliced_size, size), evoxc_module.int32
        ),
    )


#: The splice primitive, under its documented public name for sibling modules.
splice = _splice_traced


def _draw_positions(evoxc_module, key, size, donor_size):
    """``(pos, donor_pos)`` -- one uniform draw per row over its live nodes."""
    k_pos, k_donor = evoxc_module.random.split_n(key, 2)
    pop_size = int(size.shape[0])
    pos = evoxc_module.random.randint(
        k_pos, (pop_size,), 0, size[:, 0], dtype=evoxc_module.int32
    )
    donor_pos = evoxc_module.random.randint(
        k_donor, (pop_size,), 0, donor_size[:, 0], dtype=evoxc_module.int32
    )
    return pos, donor_pos


def _position_fn(pop_size: int, width: int):
    """The traced ``(size, donor_size, key) -> (pos, donor_pos)`` graph."""

    def positions(size, donor_size, key):
        return _draw_positions(_runtime.evoxc(), key, size, donor_size)

    return positions


def draw_positions(
    size,
    donor_size=None,
    key=0,
    *,
    backend: str = "numpy",
    device: str = "cpu",
):
    """Draw one ``(pos, donor_pos)`` pair per row (a memoized graph).

    Positions are uniform over a row's **live nodes**: ``randint`` draws in
    ``[0, size[row, 0])``, so a node is picked proportionally to how often it
    occurs in pre-order -- which is exactly "uniform over nodes", and the reason
    the induced subtree-size distribution is the node multiset's, not uniform
    over sizes.

    Args:
        size: ``int32 [pop_size, max_nodes]`` subtree-size channel (numpy array
            or ``evoxc.Tensor``).
        donor_size: the donor channel; ``size`` itself when omitted.
        key: int seed or a concrete key tensor.
        backend: evoxc backend name.
        device: evoxc device string.

    Returns:
        Two ``evoxc`` tensors of ``int32`` indices, shape ``[pop_size]``.

    Raises:
        ValueError: on a non-2-D ``size`` channel or disagreeing donor shape.
    """
    values = np.asarray(encoding._as_numpy(size, "size"))
    if values.ndim != 2:
        raise ValueError(f"size must be 2-D [pop_size, max_nodes], got {values.shape}")
    donor_values = (
        values
        if donor_size is None
        else np.asarray(encoding._as_numpy(donor_size, "donor_size"))
    )
    if donor_values.shape != values.shape:
        raise ValueError(
            f"donor_size must have the shape of size {values.shape}, got "
            f"{donor_values.shape}"
        )
    pop_size, width = (int(dim) for dim in values.shape)
    evoxc_module = _runtime.evoxc()
    key_tensor = _runtime.resolve_key(key)
    memo_key = (
        pop_size,
        width,
        backend,
        device,
        tuple(int(size_) for size_ in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _position_fn(pop_size, width),
        (
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            _runtime.key_spec(evoxc_module, key_tensor),
        ),
        backend,
        device,
    )
    pos, donor_pos = evoxc_module.run(
        built,
        values.astype(np.int32),
        donor_values.astype(np.int32),
        key_tensor,
    )
    return pos, donor_pos


# --------------------------------------------------------------------------- #
# the operator graph
# --------------------------------------------------------------------------- #
def _crossover_fn(pop_size: int, width: int, rows: int, prob):
    """The traced crossover graph of one static configuration."""

    def operator(opcode, payload, size, left_indices, key):
        evoxc_module = _runtime.evoxc()
        k_donor, k_pos, k_gate, k_next = evoxc_module.random.split_n(key, 4)
        donor_rows = evoxc_module.random.randint(
            k_donor, (rows,), 0, pop_size, dtype=evoxc_module.int32
        )
        left = (
            _runtime.take_rows(evoxc_module, opcode, left_indices),
            _runtime.take_rows(evoxc_module, payload, left_indices),
            _runtime.take_rows(evoxc_module, size, left_indices),
        )
        donor = (
            _runtime.take_rows(evoxc_module, opcode, donor_rows),
            _runtime.take_rows(evoxc_module, payload, donor_rows),
            _runtime.take_rows(evoxc_module, size, donor_rows),
        )
        pos, donor_pos = _draw_positions(evoxc_module, k_pos, left[2], donor[2])
        gate = _runtime.gate_tensor(evoxc_module, k_gate, prob, rows)
        opcode_out, payload_out, size_out = _splice_traced(
            evoxc_module, left, donor, pos, donor_pos, gate
        )
        return opcode_out, payload_out, size_out, k_next

    return operator


def crossover(
    parents,
    key=0,
    *,
    left_indices=None,
    prob: Optional[float] = None,
    backend: str = "numpy",
    device: str = "cpu",
) -> GPState:
    """Subtree crossover of a population with itself (offspring only).

    Per row: a donor row and both splice positions are drawn in-graph
    (``donor`` uniformly over the population, positions uniformly over the live
    nodes of the target and of the donor -- the deleted ``splicing.py``'s
    semantics), then the donor's subtree replaces the target's subtree at
    ``pos``.  Rows that would exceed ``max_nodes``, and rows filtered out by
    ``prob``, are verbatim copies of their parent.

    Args:
        parents: the parent population (a :class:`~evogp.gp.state.GPState` or an
            ``(opcode, payload, size)`` triple).
        key: int seed or a concrete key tensor.
        left_indices: optional ``int32 [rows]`` vector of target rows.  Entries
            outside ``[0, pop_size)`` raise ``ValueError`` (the legacy kernel
            read them out of bounds).  ``None`` (default) targets every row, so
            ``rows == pop_size``.
        prob: per-row crossover probability in ``[0, 1]``; ``None`` (default)
            applies to every row.
        backend: evoxc backend name.
        device: evoxc device string.

    Returns:
        A :class:`~evogp.gp.state.GPState` of ``rows`` offspring with the
        channel dtypes of the encoding, an all-``-inf`` (unevaluated) fitness
        vector and the next RNG key.

    Raises:
        ValueError: on a non-2-D channel, disagreeing channel shapes, an empty
            row selection, ``left_indices`` outside range, or a ``prob``
            outside ``[0, 1]``.
    """
    evoxc_module = _runtime.evoxc()
    opcode_array, payload_array, size_array = _runtime.channels_numpy(parents)
    if size_array.ndim != 2:
        raise ValueError(
            f"channels must be 2-D [pop_size, max_nodes], got {size_array.shape}"
        )
    if not (
        opcode_array.shape == payload_array.shape == size_array.shape
    ):
        raise ValueError(
            "the three channels must share one shape, got "
            f"{opcode_array.shape}, {payload_array.shape}, {size_array.shape}"
        )
    pop_size, width = (int(dim) for dim in size_array.shape)
    if pop_size == 0 or width == 0:
        raise ValueError("the population must be non-empty")

    if left_indices is None:
        left_array = np.arange(pop_size, dtype=np.int32)
    else:
        left_array = _runtime.index_array(left_indices, "left_indices", pop_size)
    rows = int(left_array.shape[0])
    if prob is not None and not 0.0 <= float(prob) <= 1.0:
        raise ValueError(f"prob must be in [0.0, 1.0], got {prob}")

    key_tensor = _runtime.resolve_key(key)
    memo_key = (
        pop_size,
        width,
        rows,
        None if prob is None else float(prob),
        backend,
        device,
        tuple(int(size_) for size_ in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _crossover_fn(pop_size, width, rows, prob),
        (
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.float32),
            evoxc_module.TensorSpec((pop_size, width), evoxc_module.int32),
            evoxc_module.TensorSpec((rows,), evoxc_module.int32),
            _runtime.key_spec(evoxc_module, key_tensor),
        ),
        backend,
        device,
    )
    opcode, payload, size, next_key = evoxc_module.run(
        built,
        opcode_array.astype(np.int32),
        payload_array.astype(np.float32),
        size_array.astype(np.int32),
        left_array,
        key_tensor,
    )
    return GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=_runtime.unevaluated(rows),
        key=next_key,
    )
