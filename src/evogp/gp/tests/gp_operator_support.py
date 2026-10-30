"""Shared helpers of the operator-layer test suite (not a test module).

The four ``test_gp_stack_operators_*.py`` modules all need the same three
things, and they are kept here -- once -- so the tests themselves stay about
*behaviour*:

``assert_well_formed(state)``
    The public encoding invariant (:func:`evogp.gp.encoding.validate_channels`)
    plus the operator layer's own, stronger one: the ``size`` channel must equal
    the subtree sizes **recomputed** bottom-up from the opcodes' arities
    (:func:`recompute_sizes`).  That recomputation is the single most important
    check in this suite -- a wrong ancestor mask or tail shift in the splice
    silently corrupts a run while leaving every other channel plausible.  It is
    only decidable from the opcodes alone when the function set has no
    ``Func.IF``/``Func.ADD`` (ids 0/1 collide with the ``NType.VAR``/``CONST``
    leaf markers), which is why :data:`SAFE_CONFIG` exists.

``recompute_sizes(opcode, is_function, length)``
    The bottom-up recomputation itself (a test-local pre-order walk; it also
    asserts that the row tiles exactly).

``reference_splice_row`` / ``splice_with_positions``
    An independent, list-based implementation of ``ENGINE_SPEC.md`` §7 (plain
    list slicing + a from-scratch size recomputation -- not the operator's index
    arithmetic) and the harness that pushes explicit positions through
    :func:`evogp.gp.crossover.splice`.  Together they pin the splice semantics
    on the *rich* function set (where the recomputation cannot disambiguate
    ``Func.IF``/``ADD`` from leaves, but a bit-exact oracle comparison can).
"""

from __future__ import annotations

import dataclasses

import numpy as np

import evoxc

from evogp.gp._constants import FUNC_ARITY
from evogp.gp.config import GPTreeConfig
from evogp.gp.crossover import splice
from evogp.gp.encoding import validate_channels

__all__ = [
    "SAFE_FUNCS",
    "BINARY_FUNCS",
    "RICH_FUNCS",
    "SAFE_CONFIG",
    "RICH_CONFIG",
    "Row",
    "channel_arrays",
    "row_count",
    "recompute_sizes",
    "assert_well_formed",
    "rows",
    "reference_splice_row",
    "splice_with_positions",
    "assert_row_equal",
]

#: A function set with no ``Func.IF`` (0) and no ``Func.ADD`` (1), so a node is
#: a function iff its opcode is ``>= 2`` and the size channel can be recomputed
#: from the opcodes alone.  Mixed arity (``*``/``-`` binary, ``sin``/``cos``
#: unary), so the generated trees have varied shapes.
SAFE_FUNCS = ("*", "-", "sin", "cos")

#: An all-binary subset of :data:`SAFE_FUNCS` (both ids ``>= 2``): with
#: ``layer_leaf_prob = 0`` every generated row is a *complete* binary tree, which
#: is what makes the ``max_nodes`` boundary reachable on demand.
BINARY_FUNCS = ("*", "-")

#: The full mixed set (ternary + binary + unary), including the two ids that
#: collide with the leaf markers.
RICH_FUNCS = ("+", "*", "-", "sin", "if")


def SAFE_CONFIG(**overrides) -> GPTreeConfig:
    """The 2-ary ``SAFE_FUNCS`` config (32 slots, 4 levels, mixed leaves)."""
    fields = dict(
        max_nodes=32,
        input_len=2,
        using_funcs=SAFE_FUNCS,
        max_layer_cnt=4,
        layer_leaf_prob=0.25,
        const_prob=0.5,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
    )
    fields.update(overrides)
    return GPTreeConfig(**fields)


def RICH_CONFIG(**overrides) -> GPTreeConfig:
    """The mixed-arity ``RICH_FUNCS`` config (64 slots, 4 levels)."""
    fields = dict(
        max_nodes=64,
        input_len=3,
        using_funcs=RICH_FUNCS,
        max_layer_cnt=4,
        layer_leaf_prob=0.2,
        const_prob=0.4,
        const_range=(-1.0, 1.0),
        sample_cnt=5,
    )
    fields.update(overrides)
    return GPTreeConfig(**fields)


@dataclasses.dataclass(frozen=True)
class Row:
    """One tree as numpy arrays (``opcode``/``payload``/``size``/``is_function``)."""

    opcode: np.ndarray
    payload: np.ndarray
    size: np.ndarray
    is_function: np.ndarray


def channel_arrays(state):
    """``(opcode, payload, size)`` as ``int32``/``float32``/``int32`` numpy."""
    if hasattr(state, "channels"):
        opcode, payload, size = state.channels()
    else:
        opcode, payload, size = state
    return (
        np.asarray(opcode.numpy() if hasattr(opcode, "numpy") else opcode, np.int32),
        np.asarray(payload.numpy() if hasattr(payload, "numpy") else payload, np.float32),
        np.asarray(size.numpy() if hasattr(size, "numpy") else size, np.int32),
    )


def row_count(state) -> int:
    """Number of rows of a population state."""
    return int(channel_arrays(state)[2].shape[0])


def recompute_sizes(opcode, is_function, length: int) -> np.ndarray:
    """Subtree sizes recomputed bottom-up from the opcodes' arities.

    Args:
        opcode: ``[width]`` opcode row (numpy).
        is_function: ``[width]`` bool -- ``True`` where the node is a function
            (a leaf's opcode is a leaf marker, a function's is a ``Func`` id).
        length: the row's real node count (``size[0]`` as written by the test
            caller's own knowledge, never taken from the channel under test).

    Returns:
        ``[width]`` int64 sizes, zero past ``length``.

    Raises:
        AssertionError: when the row does not tile exactly (more or fewer nodes
            than ``length``), which is itself a corruption signal.
    """
    width = int(opcode.shape[0])
    sizes = np.zeros(width, dtype=np.int64)
    end = _walk(opcode, is_function, 0, int(length), sizes)
    assert end == int(length), (
        f"the row does not tile: the walk consumed {end} of {length} nodes"
    )
    return sizes


def _walk(opcode, is_function, index: int, length: int, sizes: np.ndarray) -> int:
    assert index < length, (
        f"row does not tile: node {index} is past the row length {length}"
    )
    if not bool(is_function[index]):
        sizes[index] = 1
        return index + 1
    arity = FUNC_ARITY[int(opcode[index])]
    assert arity >= 1, f"node {index} is marked a function but has arity {arity}"
    cursor = index + 1
    total = 1
    for _ in range(arity):
        end = _walk(opcode, is_function, cursor, length, sizes)
        total += end - cursor
        cursor = end
    sizes[index] = total
    return cursor


def assert_well_formed(state, *, context: str = "", zero_padding: bool = False) -> None:
    """Every channel round-trips through validation and the size recomputation.

    ``is_function`` is derived from the opcodes (``opcode >= 2``), so this is
    only sound for a config built from :data:`SAFE_FUNCS` / :data:`SAFE_CONFIG`.

    Args:
        state: a :class:`~evogp.gp.state.GPState` or an
            ``(opcode, payload, size)`` triple.
        context: label used in the assertion messages.
        zero_padding: also require zeroed padding.  Only the encoder guarantees
            that; an operator keeps the parent's stale columns past the row's new
            length (exactly like ``../tree/splicing.py``, which documents that
            "only ``[0, subtree_size[row, 0])`` is semantically meaningful"), so
            this defaults to ``False``.
    """
    validate_channels(state)
    opcode, _payload, size = channel_arrays(state)
    prefix = f"{context}: " if context else ""
    for row in range(opcode.shape[0]):
        length = int(size[row, 0])
        is_function = opcode[row] >= 2
        # A leaf marker is 0/1 and a function id is >= 2 in a SAFE_FUNCS config,
        # so the classification must also agree with the encoded leaf sizes.
        assert np.array_equal(
            is_function[:length], size[row, :length] > 1
        ), f"{prefix}row {row}: an opcode < 2 is a leaf and a subtree of size 1 is a leaf"
        expected = recompute_sizes(opcode[row], is_function, length)
        assert np.array_equal(expected[:length], size[row, :length]), (
            f"{prefix}row {row}: size channel {size[row, :length].tolist()} "
            f"!= recomputed {expected[:length].tolist()}"
        )
        if zero_padding:
            assert np.all(size[row, length:] == 0), (
                f"{prefix}row {row}: the padding past {length} must be zero"
            )


def rows(state):
    """One :class:`Row` per population row (input-side classification)."""
    opcode, payload, size = channel_arrays(state)
    out = []
    for index in range(opcode.shape[0]):
        length = int(size[index, 0])
        out.append(
            Row(
                opcode=opcode[index].copy(),
                payload=payload[index].copy(),
                size=size[index].copy(),
                is_function=size[index] > 1,
            )
        )
    return out


def reference_splice_row(target: Row, donor: Row, pos: int, donor_pos: int,
                         max_nodes: int):
    """``ENGINE_SPEC.md`` §7 by plain list slicing (the independent oracle).

    Args:
        target: the row being edited.
        donor: the row the subtree comes from.
        pos: subtree root inside ``target``.
        donor_pos: subtree root inside ``donor``.
        max_nodes: the row width; a result past it is left untouched.

    Returns:
        The resulting :class:`Row` (sizes recomputed from scratch), or the
        ``target`` itself when the splice would overflow ``max_nodes``.

    Raises:
        AssertionError: when the splice produces a row that does not tile.
    """
    length = int(target.size[0])
    old_sub = int(target.size[pos])
    new_sub = int(donor.size[donor_pos])
    new_length = length - old_sub + new_sub
    if new_length > max_nodes:
        return target

    width = int(target.size.shape[0])
    opcode = np.zeros(width, dtype=np.int32)
    payload = np.zeros(width, dtype=np.float32)
    is_function = np.zeros(width, dtype=bool)

    head = pos
    opcode[:head] = target.opcode[:head]
    payload[:head] = target.payload[:head]
    is_function[:head] = target.is_function[:head]

    block = slice(pos, pos + new_sub)
    opcode[block] = donor.opcode[donor_pos:donor_pos + new_sub]
    payload[block] = donor.payload[donor_pos:donor_pos + new_sub]
    is_function[block] = donor.is_function[donor_pos:donor_pos + new_sub]

    tail_start = pos + old_sub
    tail = length - tail_start
    if tail:
        opcode[pos + new_sub:new_length] = target.opcode[tail_start:length]
        payload[pos + new_sub:new_length] = target.payload[tail_start:length]
        is_function[pos + new_sub:new_length] = target.is_function[tail_start:length]

    sizes = recompute_sizes(opcode, is_function, new_length)
    return Row(opcode, payload, sizes, is_function)


def splice_with_positions(state, donor_rows, pos, donor_pos, backend: str = "numpy"):
    """Run :func:`evogp.gp.crossover.splice` with explicit positions/rows.

    Args:
        state: the population (a state or an ``(opcode, payload, size)`` triple).
        donor_rows: ``[pop_size]`` donor row indices (inside the population).
        pos: ``[pop_size]`` target positions.
        donor_pos: ``[pop_size]`` donor positions.
        backend: evoxc backend name.

    Returns:
        ``(opcode, payload, size)`` numpy arrays of the spliced population.
    """
    opcode, payload, size = channel_arrays(state)
    pop_size, width = (int(dim) for dim in size.shape)

    @evoxc.defn
    def graph(op, pl, sz, donor, lpos, rpos):
        gathered = (
            evoxc.gather(op, donor, axis=0),
            evoxc.gather(pl, donor, axis=0),
            evoxc.gather(sz, donor, axis=0),
        )
        return splice(evoxc, (op, pl, sz), gathered, lpos, rpos, None)

    built = evoxc.build(
        graph,
        evoxc.TensorSpec((pop_size, width), evoxc.int32),
        evoxc.TensorSpec((pop_size, width), evoxc.float32),
        evoxc.TensorSpec((pop_size, width), evoxc.int32),
        evoxc.TensorSpec((pop_size,), evoxc.int32),
        evoxc.TensorSpec((pop_size,), evoxc.int32),
        evoxc.TensorSpec((pop_size,), evoxc.int32),
        backend=backend,
    )
    out = evoxc.run(
        built,
        opcode,
        payload,
        size,
        np.asarray(donor_rows, dtype=np.int32),
        np.asarray(pos, dtype=np.int32),
        np.asarray(donor_pos, dtype=np.int32),
    )
    return tuple(array.numpy() for array in out)


def assert_row_equal(actual, expected, *, context: str = "") -> None:
    """``opcode``/``payload``/``size`` equal over the row's live nodes.

    The declared lengths must agree first; past them the columns are padding and
    semantically irrelevant (the operator keeps the parent's stale values there,
    as ``../tree/splicing.py`` does).
    """
    prefix = f"{context}: " if context else ""
    lhs_len = int(np.asarray(actual.size)[0])
    rhs_len = int(np.asarray(expected.size)[0])
    assert lhs_len == rhs_len, (
        f"{prefix}live length mismatch: actual {lhs_len}, expected {rhs_len}"
    )
    for name in ("opcode", "payload", "size"):
        lhs = np.asarray(getattr(actual, name))[:lhs_len]
        rhs = np.asarray(getattr(expected, name))[:rhs_len]
        assert np.array_equal(lhs, rhs), (
            f"{prefix}{name} mismatch over the {lhs_len} live nodes:\n"
            f"  actual   {lhs.tolist()}\n  expected {rhs.tolist()}"
        )
