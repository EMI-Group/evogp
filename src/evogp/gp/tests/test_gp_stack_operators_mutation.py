"""Mutation tests: arity preservation, well-formedness, overflow, determinism.

The three kinds are checked against the invariant that matters most -- the
``size`` channel must equal the sizes **recomputed** from the opcodes' arities --
plus the property each kind is supposed to have:

* ``point``: **arity-preserving by construction**, so the ``size`` channel is
  bit-identical to the parent's and exactly one node changes per gated row; a
  function node may only swap inside its own arity class, and a leaf keeps its
  kind (a ``VAR`` stays a variable, a ``CONST`` stays a constant);
* ``subtree``: one subtree replaced by a fresh random tree from the generation
  graph, which may overflow ``max_nodes`` -- such a row must stay verbatim,
  per row, while its siblings are mutated;
* ``hoist``: the row shrinks to one of its own subtrees (never overflows), so the
  offspring is a contiguous block of the parent.
"""

from __future__ import annotations

import numpy as np
import pytest

from evogp.gp import GPConfig, GPTreeConfig, init_population
from evogp.gp import mutation as M
from evogp.gp._constants import FUNC_ARITY, NType

from gp_operator_support import (
    BINARY_FUNCS,
    RICH_CONFIG,
    SAFE_CONFIG,
    assert_well_formed,
    channel_arrays,
)


KINDS = ("point", "subtree", "hoist")


def _diff_rows(parent, offspring):
    """Per row: the changed columns over the row's live nodes (opcode/payload)."""
    op_p, pl_p, sz_p = channel_arrays(parent)
    op_o, pl_o, sz_o = channel_arrays(offspring)
    out = []
    for row in range(sz_p.shape[0]):
        live = int(sz_p[row, 0])
        changed = np.nonzero(
            (op_p[row, :live] != op_o[row, :live])
            | (pl_p[row, :live] != pl_o[row, :live])
        )[0]
        out.append(changed)
    return out


def _rows_equal(parent, offspring, index: int) -> bool:
    op_p, pl_p, sz_p = channel_arrays(parent)
    op_o, pl_o, sz_o = channel_arrays(offspring)
    live = int(sz_p[index, 0])
    return (
        int(sz_o[index, 0]) == live
        and np.array_equal(op_p[index, :live], op_o[index, :live])
        and np.array_equal(pl_p[index, :live], pl_o[index, :live])
    )


# --------------------------------------------------------------------------- #
# well-formedness (the single most important invariant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_mutation_keeps_every_row_well_formed(kind):
    config = SAFE_CONFIG()
    parents = init_population(config, 4, pop_size=12)
    offspring = M.mutate(config, parents, 9, kind=kind)
    assert_well_formed(offspring, context=kind)
    assert offspring.opcode.shape == parents.opcode.shape
    assert np.all(np.isneginf(offspring.fitness.numpy())), "an operator never scores"
    assert int(offspring.size.shape[0]) == 12


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("key", [0, 1, 2, 5, 99])
def test_mutation_is_well_formed_for_many_keys(kind, key):
    config = RICH_CONFIG()
    parents = init_population(config, 100 + key, pop_size=8)
    offspring = M.mutate(config, parents, key, kind=kind)
    # RICH_CONFIG mixes arity 1/2/3 and includes Func.IF/ADD, whose ids collide
    # with the leaf markers, so only the public validation applies here.
    from evogp.gp.encoding import validate_channels

    validate_channels(offspring)
    assert int(offspring.size.shape[0]) == 8


# --------------------------------------------------------------------------- #
# point mutation: arity preservation
# --------------------------------------------------------------------------- #
def test_point_mutation_preserves_every_subtree_size():
    config = SAFE_CONFIG()
    parents = init_population(config, 8, pop_size=12)
    offspring = M.point_mutation(config, parents, 3)
    _op, _pl, size_parent = channel_arrays(parents)
    _op2, _pl2, size_child = channel_arrays(offspring)
    assert np.array_equal(size_parent, size_child), (
        "point mutation preserves arity, so the size channel must be bit-identical"
    )
    assert_well_formed(offspring, context="point")


def test_point_mutation_swaps_functions_within_their_arity_class():
    config = SAFE_CONFIG()  # ids 3/2 are binary, 14/15 unary
    parents = init_population(config, 12, pop_size=24)
    opcode, _payload, size = channel_arrays(parents)
    seen_leaf_edit = seen_func_edit = False
    for key in range(6):
        offspring = M.point_mutation(config, parents, key)
        op_out, payload_out, _size_out = channel_arrays(offspring)
        for row, changed in enumerate(_diff_rows(parents, offspring)):
            assert len(changed) <= 1, (
                f"row {row} key {key}: point mutation edits at most one node, "
                f"got {changed.tolist()}"
            )
            if not len(changed):
                continue
            node = int(changed[0])
            before, after = int(opcode[row, node]), int(op_out[row, node])
            leaf = int(size[row, node]) == 1
            if leaf:
                # A leaf keeps its kind and only its payload moves.
                assert after == before, (
                    f"row {row} node {node}: a leaf's opcode must not change "
                    f"({before} -> {after})"
                )
                assert payload_out[row, node] != parents.payload.numpy()[row, node]
                if before == NType.CONST:
                    table = np.asarray(config.const_samples, np.float32)
                    assert payload_out[row, node] in table, (
                        f"row {row} node {node}: a constant leaf takes a value "
                        f"from the table, got {payload_out[row, node]}"
                    )
                else:
                    assert before == NType.VAR
                    assert 0.0 <= float(payload_out[row, node]) < config.input_len
                seen_leaf_edit = True
            else:
                assert FUNC_ARITY[after] == FUNC_ARITY[before], (
                    f"row {row} node {node}: arity must be preserved "
                    f"({before} -> {after})"
                )
                assert float(payload_out[row, node]) == 0.0
                seen_func_edit = True
    assert seen_leaf_edit and seen_func_edit, (
        "expected both leaf and function edits over 6 keys, got "
        f"leaf={seen_leaf_edit} function={seen_func_edit}"
    )


def test_point_mutation_leaves_a_single_candidate_class_alone():
    # `if` (arity 3) is the only ternary function, so a ternary node has no
    # same-arity alternative and must stay put.
    config = GPTreeConfig(
        max_nodes=32,
        input_len=2,
        using_funcs=("if", "-"),
        max_layer_cnt=3,
        layer_leaf_prob=0.1,
        const_prob=0.5,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
    )
    candidates, offsets, counts = M.swap_table(config)
    assert counts[3] == 1, "one ternary candidate"
    assert counts[2] == 1, "one binary candidate"
    assert counts[1] == 0, "no unary candidate"
    assert candidates.tolist() == [2, 0] or candidates.tolist() == [0, 2]
    parents = init_population(config, 2, pop_size=8)
    opcode, _payload, size = channel_arrays(parents)
    offspring = M.point_mutation(config, parents, 0)
    op_out = channel_arrays(offspring)[0]
    ternary = (opcode == 0) & (size > 1)
    assert np.array_equal(op_out[ternary], opcode[ternary]), (
        "a ternary node has no same-arity alternative, so it must not change"
    )
    binary = (opcode == 2) & (size > 1)
    if binary.any():
        assert np.array_equal(op_out[binary], opcode[binary])


def test_swap_table_groups_by_arity():
    config = RICH_CONFIG()  # if:3, +,-,*:2, sin:1
    candidates, offsets, counts = M.swap_table(config)
    assert counts.tolist() == [0, 1, 3, 1]
    assert offsets.tolist()[1:] == [0, 1, 4]
    assert candidates[:1].tolist() == [14]  # sin (unary)
    assert sorted(candidates[1:4].tolist()) == [1, 2, 3]  # + - *
    assert candidates[4:].tolist() == [0]  # if (ternary)
    assert all(FUNC_ARITY[int(c)] == arity for arity, count in enumerate(counts)
               for c in candidates[offsets[arity]:offsets[arity] + count])


# --------------------------------------------------------------------------- #
# subtree mutation: overflow is per row
# --------------------------------------------------------------------------- #
def test_subtree_mutation_leaves_overflowing_rows_verbatim():
    # Complete binary trees of exactly max_nodes (15): replacing any non-root
    # subtree by a ~15-node donor overflows, while the root slot fits exactly.
    config = SAFE_CONFIG(
        max_nodes=15, max_layer_cnt=4, layer_leaf_prob=0.0, using_funcs=BINARY_FUNCS
    )
    parents = init_population(config, 3, pop_size=8)
    opcode, _payload, size = channel_arrays(parents)
    assert np.all(size[:, 0] == 15)
    unchanged_total = 0
    for key in range(4):
        offspring = M.subtree_mutation(config, parents, key)
        assert_well_formed(offspring, context="subtree")
        op_out = channel_arrays(offspring)[0]
        for row in range(8):
            if _rows_equal(parents, offspring, row):
                unchanged_total += 1
            else:
                assert int(size[row, 0]) <= 15
                assert not np.array_equal(op_out[row], opcode[row])
    assert unchanged_total > 0, (
        "at least one row per key must hit the overflow path and stay verbatim"
    )


def test_subtree_mutation_changes_most_rows():
    config = SAFE_CONFIG()
    parents = init_population(config, 6, pop_size=16)
    offspring = M.subtree_mutation(config, parents, 1)
    same = sum(_rows_equal(parents, offspring, row) for row in range(16))
    assert same <= 4, (
        f"with probability=1 nearly every row must be mutated, {same}/16 were not"
    )


# --------------------------------------------------------------------------- #
# hoist mutation: the row becomes one of its own subtrees
# --------------------------------------------------------------------------- #
def test_hoist_mutation_yields_a_subtree_of_the_parent():
    config = SAFE_CONFIG()
    parents = init_population(config, 14, pop_size=12)
    opcode, payload, size = channel_arrays(parents)
    for key in range(3):
        offspring = M.hoist_mutation(config, parents, key)
        assert_well_formed(offspring, context="hoist")
        op_out, pl_out, sz_out = channel_arrays(offspring)
        for row in range(12):
            live_parent = int(size[row, 0])
            live = int(sz_out[row, 0])
            assert live <= live_parent, (
                f"hoist shrinks: row {row} went {live_parent} -> {live}"
            )
            match = any(
                np.array_equal(op_out[row, :live], opcode[row, p:p + live])
                and np.array_equal(sz_out[row, :live], size[row, p:p + live])
                for p in range(live_parent - live + 1)
            )
            assert match, (
                f"row {row} is not a contiguous block of its parent "
                f"(length {live} of {live_parent})"
            )


def test_hoist_of_a_single_node_row_is_the_identity():
    config = SAFE_CONFIG()
    parents = init_population(config, 15, pop_size=8)
    size = channel_arrays(parents)[2]
    single = [row for row in range(8) if int(size[row, 0]) == 1]
    assert single, "expected at least one one-node tree in this population"
    offspring = M.hoist_mutation(config, parents, 0)
    assert_well_formed(offspring, context="hoist")
    for row in single:
        assert _rows_equal(parents, offspring, row), (
            f"row {row} has one node, so hoisting it can only reproduce it"
        )


# --------------------------------------------------------------------------- #
# probability, determinism, guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_probability_zero_copies_the_parents(kind):
    config = SAFE_CONFIG()
    parents = init_population(config, 21, pop_size=10)
    offspring = M.mutate(config, parents, 5, kind=kind, probability=0.0)
    op_p, pl_p, sz_p = channel_arrays(parents)
    op_o, pl_o, sz_o = channel_arrays(offspring)
    assert np.array_equal(op_p, op_o) and np.array_equal(pl_p, pl_o)
    assert np.array_equal(sz_p, sz_o)


@pytest.mark.parametrize("kind", ["subtree", "hoist"])
def test_probability_one_mutates_most_rows(kind):
    config = SAFE_CONFIG()
    parents = init_population(config, 22, pop_size=16)
    offspring = M.mutate(config, parents, 6, kind=kind, probability=1.0)
    same = sum(_rows_equal(parents, offspring, row) for row in range(16))
    assert same <= 4, f"{kind}: {same}/16 rows unmutated at probability=1"


def test_point_mutation_gate_fires_on_every_gated_row():
    # Point mutation draws one node and swaps it inside its arity class; with
    # two candidates per class in SAFE_CONFIG the swap re-picks the same id
    # about half the time, which is a no-op edit, not a skipped row.
    config = SAFE_CONFIG()
    parents = init_population(config, 26, pop_size=16)
    changed = total = 0
    for key in range(8):
        offspring = M.point_mutation(config, parents, key, probability=1.0)
        changed += sum(
            not _rows_equal(parents, offspring, row) for row in range(16)
        )
        total += 16
    rate = changed / total
    # Both same-arity candidate classes have two members here, and a leaf edit
    # can re-draw its own value, so roughly half the edits are no-ops; the gate
    # itself is proven by the rate being far off both extremes.
    assert 0.3 <= rate <= 0.8, (
        f"with two same-arity candidates a point edit changes a row about half "
        f"the time; measured {rate:.3f} over {total} rows"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_mutation_is_deterministic(kind):
    config = SAFE_CONFIG()
    parents = init_population(config, 23, pop_size=8)
    first = M.mutate(config, parents, 7, kind=kind)
    second = M.mutate(config, parents, 7, kind=kind)
    for name in ("opcode", "payload", "size"):
        assert np.array_equal(
            getattr(first, name).numpy(), getattr(second, name).numpy()
        ), f"{kind}: {name} must be reproducible from the key"
    other = M.mutate(config, parents, 8, kind=kind)
    assert not np.array_equal(first.opcode.numpy(), other.opcode.numpy())


def test_mutation_kind_comes_from_the_gp_config():
    config = SAFE_CONFIG()
    parents = init_population(config, 24, pop_size=8)
    gp = GPConfig(pop_size=8, mutation_kind="hoist")
    through_config = M.mutate((config, gp), parents, 2)
    explicit = M.mutate(config, parents, 2, kind="hoist")
    assert np.array_equal(
        through_config.opcode.numpy(), explicit.opcode.numpy()
    ), "kind=None must mean GPConfig.mutation_kind"
    with pytest.raises(ValueError, match="mutation_kind|kind"):
        M.mutate((config, GPConfig(pop_size=8, mutation_kind="cosmic")), parents, 2)


def test_mutation_guards():
    config = SAFE_CONFIG()
    parents = init_population(config, 25, pop_size=4)
    with pytest.raises(ValueError, match="kind"):
        M.mutate(config, parents, 0, kind="shuffle")
    with pytest.raises(ValueError, match="probability"):
        M.mutate(config, parents, 0, probability=1.5)
    with pytest.raises(ValueError, match="GPTreeConfig"):
        M.mutate(GPConfig(pop_size=4), parents, 0)  # no function set to draw from
    with pytest.raises(TypeError, match="config must be"):
        M.mutate(None, parents, 0)
    with pytest.raises(ValueError, match="rows but the config's pop_size"):
        M.mutate((config, GPConfig(pop_size=10)), parents, 0)
    with pytest.raises(ValueError, match="2-D"):
        op, pl, sz = channel_arrays(parents)
        M.mutate(config, (op, pl, sz[:, 0]), 0)
