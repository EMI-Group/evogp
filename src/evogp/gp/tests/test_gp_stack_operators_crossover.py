"""Crossover tests: the splice semantics, its distributions and its guards.

Everything runs on the numpy evoxc backend at small sizes (no GPU, no timing).

What is pinned here:

* **the splice is exactly ``ENGINE_SPEC.md`` §7** -- an independent, list-based
  reference implementation (``gp_operator_support.reference_splice_row``, which
  rebuilds the row by slicing and recomputes every subtree size from scratch)
  agrees bit for bit with the operator's index arithmetic, on the mixed-arity
  function set (``Func.IF``/``Func.ADD`` included, plus constant payloads) and
  with explicit positions, so the ancestor mask and the tail shift are checked
  directly rather than through a distribution;
* **overflow is a no-op per row** -- a splice whose result would exceed
  ``max_nodes`` leaves the row verbatim, while the other rows of the same call
  are spliced;
* **both draws are uniform over live nodes** -- positions and the induced
  subtree sizes match the node multiset's frequencies, so no depth bias sneaks
  in;
* **the guards** -- ``left_indices`` outside ``[0, pop_size)`` raises
  ``ValueError`` (the legacy kernel's undefined behaviour) and ``prob`` filters
  rows;
* **determinism** -- one key, one output; two keys, two outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from evogp.gp import GPConfig, GPTreeConfig, init_population, validate_channels
from evogp.gp import crossover as X

from gp_operator_support import (
    BINARY_FUNCS,
    RICH_CONFIG,
    SAFE_CONFIG,
    assert_row_equal,
    assert_well_formed,
    channel_arrays,
    reference_splice_row,
    rows,
    splice_with_positions,
)


# --------------------------------------------------------------------------- #
# well-formedness of the offspring
# --------------------------------------------------------------------------- #
def test_offspring_are_well_formed_and_unevaluated():
    config = SAFE_CONFIG()
    parents = init_population(config, 11, pop_size=12)
    offspring = X.crossover(parents, 3, prob=1.0)

    validate_channels(offspring)
    assert_well_formed(offspring, context="crossover")
    assert offspring.opcode.shape == parents.opcode.shape
    assert offspring.opcode.dtype == parents.opcode.dtype
    assert offspring.payload.dtype == parents.payload.dtype
    assert offspring.size.dtype == parents.size.dtype
    assert np.all(np.isneginf(offspring.fitness.numpy())), (
        "an operator returns offspring only -- fitness must be re-evaluated"
    )
    assert offspring.key is not None


@pytest.mark.parametrize("key", [0, 1, 2, 17, 12345])
def test_offspring_are_well_formed_for_many_keys(key):
    config = RICH_CONFIG()
    parents = init_population(config, key + 100, pop_size=10)
    offspring = X.crossover(parents, key, prob=1.0)
    validate_channels(offspring)
    assert int(offspring.size.shape[0]) == 10


# --------------------------------------------------------------------------- #
# exact semantics against an independent reference
# --------------------------------------------------------------------------- #
def test_splice_matches_the_list_reference_on_explicit_positions():
    config = RICH_CONFIG()
    parents = init_population(config, 5, pop_size=16)
    target_rows = rows(parents)
    donor_rows = np.roll(np.arange(16, dtype=np.int32), 5)
    # The donor positions must be drawn against the rows that actually donate,
    # which is what the operator does (it draws on the gathered donor channel).
    donor_size = channel_arrays(parents)[2][donor_rows]

    for key in (0, 1, 2, 3, 4):
        pos, donor_pos = X.draw_positions(parents.size, donor_size, key=key)
        opcode, payload, size = splice_with_positions(
            parents, donor_rows, pos.numpy(), donor_pos.numpy()
        )
        actual = rows((opcode, payload, size))
        for index in range(16):
            expected = reference_splice_row(
                target_rows[index],
                target_rows[int(donor_rows[index])],
                int(pos.numpy()[index]),
                int(donor_pos.numpy()[index]),
                config.max_nodes,
            )
            assert_row_equal(
                actual[index], expected,
                context=f"key {key} row {index}",
            )


def test_splice_handles_the_root_and_the_last_leaf():
    config = SAFE_CONFIG()
    parents = init_population(config, 21, pop_size=6)
    target_rows = rows(parents)
    _opcode, _payload, size = channel_arrays(parents)
    donor_rows = np.arange(6, dtype=np.int32)[::-1].copy()

    # Root-in / root-out, last-leaf-in / last-leaf-out and both crossed, with
    # every position inside its own row (the operator draws them that way).
    for index in range(6):
        length = int(size[index, 0])
        donor_len = int(size[int(donor_rows[index]), 0])
        for pos, donor_pos in (
            (0, 0),
            (length - 1, donor_len - 1),
            (0, donor_len - 1),
            (length - 1, 0),
        ):
            out = splice_with_positions(
                parents,
                donor_rows,
                _uniform(6, pos, index),
                _uniform(6, donor_pos, index),
            )
            actual = rows(out)[index]
            expected = reference_splice_row(
                target_rows[index],
                target_rows[int(donor_rows[index])],
                pos,
                donor_pos,
                config.max_nodes,
            )
            assert_row_equal(
                actual, expected,
                context=f"row {index} pos {pos} donor_pos {donor_pos}",
            )


def _uniform(count: int, value: int, row: int) -> np.ndarray:
    """A per-row index vector that is ``value`` at ``row`` (the rest is 0)."""
    vector = np.zeros(count, dtype=np.int32)
    vector[row] = value
    return vector


# --------------------------------------------------------------------------- #
# overflow rows are untouched
# --------------------------------------------------------------------------- #
def _complete_population(pop_size: int = 8, seed: int = 7):
    """A population of complete full-width trees (``layer_leaf_prob = 0``).

    With ``max_nodes == 15`` and a 2-ary set, "no leaf before the last level"
    means every row is exactly 15 nodes, which is the only way to make the
    overflow boundary reachable on demand.
    """
    config = SAFE_CONFIG(
        max_nodes=15, max_layer_cnt=4, layer_leaf_prob=0.0, using_funcs=BINARY_FUNCS
    )
    parents = init_population(config, seed, pop_size=pop_size)
    size = channel_arrays(parents)[2]
    assert np.all(size[:, 0] == config.max_nodes), "expected full-width trees"
    return config, parents


def test_overflowing_rows_are_left_verbatim():
    config, parents = _complete_population()
    opcode, _payload, size = channel_arrays(parents)
    width = config.max_nodes
    donor_rows = np.zeros(8, dtype=np.int32)

    # pos on the last leaf of a full row (subtree size 1), donor = a whole
    # 15-node tree: 15 - 1 + 15 == 29 > 15, so the row must not be touched.
    pos = np.full(8, width - 1, dtype=np.int32)
    donor_pos = np.zeros(8, dtype=np.int32)
    out_opcode, out_payload, out_size = splice_with_positions(
        parents, donor_rows, pos, donor_pos
    )
    for index in range(8):
        for name, out in (
            ("opcode", out_opcode),
            ("payload", out_payload),
            ("size", out_size),
        ):
            assert np.array_equal(out[index], getattr(parents, name).numpy()[index]), (
                f"row {index} would overflow max_nodes={width} and must stay "
                f"verbatim, but its {name} channel changed"
            )


def test_overflow_is_decided_per_row():
    config, parents = _complete_population()
    target_rows = rows(parents)
    opcode, _payload, size = channel_arrays(parents)
    width = config.max_nodes
    donor_rows = np.zeros(8, dtype=np.int32)

    # Rows 0-3 request an overflowing splice, rows 4-7 a fitting one: the same
    # call must splice the latter and leave the former alone.
    pos = np.full(8, width - 1, dtype=np.int32)
    pos[4:] = 0
    donor_pos = np.zeros(8, dtype=np.int32)
    out_opcode, out_payload, out_size = splice_with_positions(
        parents, donor_rows, pos, donor_pos
    )
    actual = rows((out_opcode, out_payload, out_size))
    for index in range(8):
        expected = reference_splice_row(
            target_rows[index], target_rows[0],
            int(pos[index]), int(donor_pos[index]), width,
        )
        assert_row_equal(actual[index], expected, context=f"row {index}")
        if index < 4:
            assert np.array_equal(out_opcode[index], opcode[index]), (
                f"row {index} overflows and must stay verbatim"
            )
    # ... and the fitting half really did change.
    assert not np.array_equal(out_opcode[4:], opcode[4:])


def test_rows_that_exactly_fill_max_nodes_are_spliced():
    config, parents = _complete_population()
    target_rows = rows(parents)
    opcode, _payload, size = channel_arrays(parents)
    width = config.max_nodes
    donor_rows = np.zeros(8, dtype=np.int32)

    # Replace a leaf by a leaf: the row keeps its width, i.e. the result is
    # exactly max_nodes long and must NOT be treated as overflow.
    pos = np.zeros(8, dtype=np.int32)  # node 0 of a complete tree is a function
    for index in range(8):
        pos[index] = int(np.nonzero(size[index, :width] == 1)[0][0])  # first leaf
    donor_pos = np.zeros(8, dtype=np.int32)
    for index in range(8):
        donor_pos[index] = int(np.nonzero(size[index, :width] == 1)[0][-1])
    out_opcode, out_payload, out_size = splice_with_positions(
        parents, donor_rows, pos, donor_pos
    )
    actual = rows((out_opcode, out_payload, out_size))
    for index in range(8):
        expected = reference_splice_row(
            target_rows[index], target_rows[int(donor_rows[index])],
            int(pos[index]), int(donor_pos[index]), width,
        )
        assert_row_equal(actual[index], expected, context=f"row {index}")
        assert int(out_size[index, 0]) == width, (
            f"row {index} must still be exactly max_nodes long"
        )


# --------------------------------------------------------------------------- #
# draw distributions
# --------------------------------------------------------------------------- #
def _longest_tree_size(config, *, minimum: int, pop_size: int = 8):
    """A ``[8, width]`` size channel tiling the first tree of ``length >= minimum``."""
    for seed in range(200):
        parents = init_population(config, seed, pop_size=pop_size)
        size = channel_arrays(parents)[2]
        for row in range(pop_size):
            if int(size[row, 0]) >= minimum:
                return np.tile(size[row : row + 1], (8, 1))
    raise AssertionError(f"no tree of >= {minimum} nodes in 200 populations")


def test_positions_are_uniform_over_live_nodes():
    config = SAFE_CONFIG()
    # One tree tiled over the rows, so every row has the same live columns and
    # the histogram over columns is a direct uniformity test over *nodes*.
    size = _longest_tree_size(config, minimum=12)
    live_columns = int(size[0, 0])
    counts = np.zeros(config.max_nodes, dtype=np.int64)
    draws = 240
    for key in range(draws):
        pos, donor_pos = X.draw_positions(size, size, key=key)
        values = pos.numpy()
        assert np.all(values < size[:, 0]), "a position must be inside its row"
        assert np.all(donor_pos.numpy() < size[:, 0])
        counts += np.bincount(values, minlength=config.max_nodes)

    total = draws * size.shape[0]
    assert counts.sum() == total
    expected = total / live_columns
    for column in range(live_columns):
        assert abs(counts[column] - expected) <= 5.0 * np.sqrt(expected), (
            f"column {column} was drawn {counts[column]} times, expected "
            f"{expected:.1f} -- the draw must be uniform over live nodes"
        )
    assert np.all(counts[live_columns:] == 0), "padding columns are unreachable"


def test_subtree_size_distribution_matches_the_node_multiset():
    config = SAFE_CONFIG(max_layer_cnt=4)
    size = _longest_tree_size(config, minimum=12)
    length = int(size[0, 0])
    sizes_of_nodes = size[0, :length]

    draws = 4000
    observed = {}
    for key in range(draws):
        pos, _donor_pos = X.draw_positions(size, size, key=key)
        value = int(sizes_of_nodes[int(pos.numpy()[0])])
        observed[value] = observed.get(value, 0) + 1

    unique, counts = np.unique(sizes_of_nodes, return_counts=True)
    for value, occurrences in zip(unique, counts):
        expected = draws * occurrences / length
        got = observed.get(int(value), 0)
        assert abs(got - expected) <= 5.0 * np.sqrt(expected), (
            f"subtree size {value} drawn {got} times, expected {expected:.1f} "
            f"({occurrences} nodes of that size out of {length})"
        )


# --------------------------------------------------------------------------- #
# guards, filtering and determinism
# --------------------------------------------------------------------------- #
def test_out_of_range_left_indices_raise():
    config = SAFE_CONFIG()
    parents = init_population(config, 3, pop_size=4)
    with pytest.raises(ValueError, match="left_indices"):
        X.crossover(parents, 0, left_indices=[0, 1, 4, 2], prob=1.0)
    with pytest.raises(ValueError, match="left_indices"):
        X.crossover(parents, 0, left_indices=[-1, 0], prob=1.0)
    with pytest.raises(ValueError, match="left_indices"):
        X.crossover(parents, 0, left_indices=[], prob=1.0)
    with pytest.raises(ValueError, match="left_indices"):
        X.crossover(parents, 0, left_indices=np.zeros((2, 2), np.int32), prob=1.0)


def test_prob_zero_copies_the_selected_parents():
    config = SAFE_CONFIG()
    parents = init_population(config, 31, pop_size=6)
    opcode, payload, size = channel_arrays(parents)
    left = np.asarray([5, 3, 3, 0, 1, 2], dtype=np.int32)
    offspring = X.crossover(parents, 4, left_indices=left, prob=0.0)
    for index in range(6):
        source = int(left[index])
        assert np.array_equal(offspring.opcode.numpy()[index], opcode[source])
        assert np.array_equal(offspring.payload.numpy()[index], payload[source])
        assert np.array_equal(offspring.size.numpy()[index], size[source])


def test_prob_one_changes_most_rows():
    config = SAFE_CONFIG()
    parents = init_population(config, 41, pop_size=16)
    offspring = X.crossover(parents, 5, prob=1.0)
    same = np.all(
        (offspring.opcode.numpy() == parents.opcode.numpy())
        & (offspring.payload.numpy() == parents.payload.numpy())
        & (offspring.size.numpy() == parents.size.numpy()),
        axis=1,
    )
    assert same.sum() <= 2, (
        "with prob=1 nearly every offspring must differ from its parent, got "
        f"{int(same.sum())} identical rows out of 16"
    )


def test_crossover_is_deterministic():
    config = SAFE_CONFIG()
    parents = init_population(config, 51, pop_size=8)
    first = X.crossover(parents, 7, prob=0.5)
    second = X.crossover(parents, 7, prob=0.5)
    for name in ("opcode", "payload", "size"):
        assert np.array_equal(
            getattr(first, name).numpy(), getattr(second, name).numpy()
        ), f"{name} must be reproducible from the key"
    assert first.key.numpy() == second.key.numpy()

    other = X.crossover(parents, 8, prob=0.5)
    assert not np.array_equal(first.opcode.numpy(), other.opcode.numpy())


def test_bad_channel_shapes_raise():
    config = SAFE_CONFIG()
    parents = init_population(config, 61, pop_size=4)
    opcode = parents.opcode.numpy()
    payload = parents.payload.numpy()
    size = parents.size.numpy()
    with pytest.raises(ValueError, match="2-D"):
        X.crossover((opcode, payload, size[:, 0]), 0)
    with pytest.raises(ValueError, match="share one shape"):
        X.crossover((opcode, payload, size[:, : size.shape[1] - 1]), 0)
    with pytest.raises(ValueError, match="prob"):
        X.crossover(parents, 0, prob=1.5)
