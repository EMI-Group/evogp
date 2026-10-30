"""Encoding tests of the evoxc GP stack: generation, validation and the bridge.

Everything runs on the numpy evoxc backend at small sizes (this suite never
touches a GPU and never times anything -- the GPU benchmarking task owns that).

The invariants pinned here are the contract the operator layer and the engines
will rely on: static shape/dtype, a well-formed prefix encoding, a decision
structure that matches the torch reference's distribution, key determinism, an
exact round-trip through the tree layer's legacy triple, and a validation that
rejects every malformed tree with a ``ValueError``.
"""

import numpy as np
import pytest

import evoxc

from evogp.gp import (
    GPConfig,
    GPState,
    GPTreeConfig,
    cache_stats,
    clear_cache,
    from_legacy_triple,
    init_population,
    is_leaf,
    live_mask,
    replace,
    row_length,
    to_legacy_triple,
    validate_channels,
)
from evogp.gp._constants import FUNC_ARITY, NType, Func

#: A mixed function set (ternary + binary + unary) on the padded 3-ary layout.
FUNCS = ("+", "*", "-", "sin", "if")

#: The reference config of the suite: 64 slots, 4 levels, mixed leaves.
CONFIG = GPTreeConfig(
    max_nodes=64,
    input_len=3,
    using_funcs=FUNCS,
    max_layer_cnt=4,
    layer_leaf_prob=0.2,
    const_prob=0.4,
    const_range=(-1.0, 1.0),
    sample_cnt=5,
)


def depths_of(opcode: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Depth of every live node of one row (a test-local pre-order walk)."""
    length = int(size[0])
    depths = np.zeros(length, dtype=np.int64)
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        depths[node] = depth
        if size[node] == 1:
            continue
        child = node + 1
        for _ in range(FUNC_ARITY[int(opcode[node])]):
            stack.append((child, depth + 1))
            child += int(size[child])
    return depths


def sampled(pop_size: int = 16, key=11, config: GPTreeConfig = CONFIG) -> GPState:
    return init_population(config, key, pop_size=pop_size)


# --------------------------------------------------------------------------- #
# shape / dtype / mask invariants
# --------------------------------------------------------------------------- #
def test_channels_are_static_shaped_and_typed():
    pop_size = 12
    state = sampled(pop_size)

    for channel in (state.opcode, state.payload, state.size):
        assert isinstance(channel, evoxc.Tensor)
        assert tuple(channel.shape) == (pop_size, CONFIG.max_nodes)
    assert state.opcode.dtype == evoxc.int32
    assert state.payload.dtype == evoxc.float32
    assert state.size.dtype == evoxc.int32

    assert tuple(state.fitness.shape) == (pop_size,)
    assert state.fitness.dtype == evoxc.float32
    assert np.all(np.isneginf(state.fitness.numpy())), "an unevaluated state is -inf"

    assert tuple(state.key.shape) == ()  # splitmix64 keys are rank-0 int64
    assert state.key.dtype == evoxc.int64


def test_row_length_and_live_mask_match_the_size_channel():
    state = sampled(16)
    _, _, size = state.channels()
    size = size.numpy()

    length = row_length(state)
    assert length.shape == (16,)
    np.testing.assert_array_equal(length, size[:, 0])
    assert np.all(length >= 1)
    assert np.all(length <= CONFIG.max_nodes)

    mask = live_mask(state)
    assert mask.shape == size.shape and mask.dtype == bool
    np.testing.assert_array_equal(mask.sum(axis=1), length)
    # the live cells of a row are exactly its first `length` columns
    assert np.all(mask == (np.arange(size.shape[1])[None, :] < length[:, None]))


def test_padding_is_zero_and_the_payload_of_a_function_node_is_zero():
    state = sampled(16)
    opcode, payload, size = (c.numpy() for c in state.channels())
    mask = live_mask(state)
    assert not np.any(size[~mask])
    assert not np.any(opcode[~mask])
    assert not np.any(payload[~mask])
    function = size > 1
    assert not np.any(payload[function])


def test_is_leaf_matches_the_subtree_size():
    state = sampled(8)
    size = state.size.numpy()
    np.testing.assert_array_equal(is_leaf(state.size), size == 1)
    assert live_mask(state)[size == 1].all()


# --------------------------------------------------------------------------- #
# well-formedness
# --------------------------------------------------------------------------- #
def test_generated_trees_are_well_formed():
    state = sampled(64)
    validate_channels(state, input_len=CONFIG.input_len)  # must not raise

    opcode, payload, size = (c.numpy() for c in state.channels())
    for row in range(size.shape[0]):
        length = int(size[row, 0])

        # depth never exceeds the schedule's depth
        depths = depths_of(opcode[row], size[row])
        assert depths.max() <= CONFIG.max_layer_cnt - 1

        # the arity-derived subtree sizes agree with the size channel
        for node in range(length - 1, -1, -1):
            if size[row, node] == 1:
                assert opcode[row, node] in (NType.VAR, NType.CONST)
                continue
            arity = FUNC_ARITY[int(opcode[row, node])]
            child = node + 1
            expected = 1
            for _ in range(arity):
                expected += int(size[row, child])
                child += int(size[row, child])
            assert size[row, node] == expected

        # every live node is either a known function or a leaf marker
        for node in range(length):
            if size[row, node] == 1:
                continue
            assert 0 <= opcode[row, node] < Func.END
            # a variable leaf carries an index inside input_len
        variables = (size[row, :length] == 1) & (
            opcode[row, :length] == NType.VAR
        )
        assert np.all(payload[row, :length][variables] >= 0)
        assert np.all(payload[row, :length][variables] < CONFIG.input_len)


def test_the_same_key_reproduces_the_population_and_a_new_key_changes_it():
    first = sampled(16, key=1234)
    again = sampled(16, key=1234)
    other = sampled(16, key=1235)

    for left, right in zip(first.channels(), again.channels()):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    np.testing.assert_array_equal(first.key.numpy(), again.key.numpy())

    opcode_a = first.opcode.numpy()
    opcode_b = other.opcode.numpy()
    assert not np.array_equal(opcode_a, opcode_b)
    assert not np.array_equal(first.size.numpy(), other.size.numpy())


def test_next_generation_key_is_deterministic_and_differs_from_the_input_key():
    state = sampled(8, key=99)
    assert int(state.key.numpy()) != 99
    again = sampled(8, key=99)
    assert int(again.key.numpy()) == int(state.key.numpy())


def test_key_accepts_an_explicit_evoxc_key_tensor():
    from_seed = init_population(CONFIG, 4242, pop_size=8)
    from_tensor = init_population(CONFIG, evoxc.random.key(4242), pop_size=8)
    for left, right in zip(from_seed.channels(), from_tensor.channels()):
        np.testing.assert_array_equal(left.numpy(), right.numpy())


def test_pop_size_comes_from_the_gp_config_in_a_pair():
    state = init_population((CONFIG, GPConfig(pop_size=23)), 5)
    assert tuple(state.opcode.shape) == (23, CONFIG.max_nodes)
    assert tuple(state.fitness.shape) == (23,)
    assert tuple(live_mask(state).shape) == (23, CONFIG.max_nodes)

    with pytest.raises(ValueError):
        init_population(CONFIG, 5)  # a bare tree config needs pop_size
    with pytest.raises(ValueError):
        init_population((CONFIG, GPConfig(pop_size=8)), 5, pop_size=9)


def test_the_generation_graph_is_memoized():
    clear_cache()
    sampled(8, key=1)
    after_first = cache_stats()
    assert after_first["graphs"] == 1
    sampled(8, key=2)
    sampled(8, key=3)
    assert cache_stats()["graphs"] == 1, "one graph per (config, pop_size, device, key)"
    sampled(8, key=4, config=CONFIG.replace(const_prob=0.25))
    assert cache_stats()["graphs"] == 2, "a different config is a different graph"


def test_multi_output_is_rejected():
    multi = CONFIG.replace(output_len=2)
    with pytest.raises(NotImplementedError):
        init_population(multi, 1, pop_size=4)


def test_generation_works_for_degenerate_function_sets():
    unary = GPTreeConfig(
        max_nodes=16,
        input_len=2,
        using_funcs=("sin", "neg"),
        max_layer_cnt=3,
        const_range=(-1.0, 1.0),
        sample_cnt=2,
    )
    state = init_population(unary, 7, pop_size=16)
    validate_channels(state, input_len=2)
    assert row_length(state).max() <= 3  # one slot per level

    single_level = GPTreeConfig(
        max_nodes=8,
        input_len=2,
        using_funcs=("+", "sin"),
        max_layer_cnt=1,
        const_range=(-1.0, 1.0),
        sample_cnt=2,
    )
    state = init_population(single_level, 7, pop_size=8)
    validate_channels(state, input_len=2)
    np.testing.assert_array_equal(row_length(state), np.ones(8, dtype=np.int64))


# --------------------------------------------------------------------------- #
# distribution (the tree layer's decision structure)
# --------------------------------------------------------------------------- #
def test_leaf_const_and_variable_proportions_match_the_config():
    config = CONFIG.replace(const_prob=0.35, layer_leaf_prob=0.3)
    state = init_population(config, 2024, pop_size=2048)
    opcode, _, size = (c.numpy() for c in state.channels())

    leaf = size == 1
    function = size > 1
    live = live_mask(state)

    const_fraction = float((opcode[leaf] == NType.CONST).mean())
    var_fraction = float((opcode[leaf] == NType.VAR).mean())
    assert abs(const_fraction - 0.35) < 0.05, const_fraction
    assert abs(var_fraction - 0.65) < 0.05, var_fraction

    # depth 0..max_layer_cnt-2 draw functions with probability 1 - layer_leaf_prob
    per_depth = {depth: [] for depth in range(config.max_layer_cnt)}
    for row in range(size.shape[0]):
        depths = depths_of(opcode[row], size[row])
        for depth in range(config.max_layer_cnt):
            per_depth[depth].append(function[row, : size[row, 0]][depths == depth])
    for depth in range(config.max_layer_cnt - 1):
        observed = np.concatenate(per_depth[depth])
        assert observed.size > 1000
        assert abs(float(observed.mean()) - 0.7) < 0.05, (depth, observed.mean())
    deepest = np.concatenate(per_depth[config.max_layer_cnt - 1])
    assert deepest.size > 0
    assert not deepest.any(), "the deepest level is all leaves by the schedule"

    # population-level function fraction of a 3-ary-ish tree is well inside
    # the reference campaign's band (0.44-0.45 for this config)
    assert 0.30 < function[live].mean() < 0.60


def test_tree_length_and_depth_are_bounded():
    state = init_population(CONFIG, 77, pop_size=512)
    opcode, _, size = (c.numpy() for c in state.channels())
    assert 4.0 < size[:, 0].mean() < 20.0, "a mixed 4-level population"
    for row in range(size.shape[0]):
        assert int(size[row, 0]) <= CONFIG.max_nodes
        assert depths_of(opcode[row], size[row]).max() <= CONFIG.max_layer_cnt - 1


# --------------------------------------------------------------------------- #
# the legacy-triple bridge
# --------------------------------------------------------------------------- #
def test_legacy_triple_round_trips_exactly():
    state = sampled(32)
    value, legacy_type, legacy_size = to_legacy_triple(state)
    assert value.dtype == np.float32
    assert legacy_type.dtype == np.int16
    assert legacy_size.dtype == np.int16
    assert value.shape == legacy_type.shape == legacy_size.shape == (
        32,
        CONFIG.max_nodes,
    )

    opcode, payload, size = (c.numpy() for c in state.channels())
    np.testing.assert_array_equal(legacy_size.astype(np.int32), size)
    # a function node carries its id as the value, a leaf its payload
    function = size > 1
    np.testing.assert_array_equal(
        value[function].astype(np.int32), opcode[function]
    )
    np.testing.assert_array_equal(value[size == 1], payload[size == 1])

    back = from_legacy_triple(value, legacy_type, legacy_size)
    np.testing.assert_array_equal(back[0], opcode)
    np.testing.assert_array_equal(back[1], payload)
    np.testing.assert_array_equal(back[2], size)
    validate_channels(back, input_len=CONFIG.input_len)

    again = to_legacy_triple(back)
    for left, right in zip(again, (value, legacy_type, legacy_size)):
        np.testing.assert_array_equal(left, right)


def test_legacy_triple_accepts_evoxc_tensors():
    state = sampled(8)
    numpy_triple = to_legacy_triple(state)
    tensor_triple = to_legacy_triple(state.channels())
    for left, right in zip(numpy_triple, tensor_triple):
        np.testing.assert_array_equal(left, right)


def test_legacy_bridge_rejects_multi_output_tags():
    value = np.zeros((2, 4), np.float32)
    legacy_type = np.zeros((2, 4), np.int16)
    legacy_type[0, 0] = NType.BFUNC | NType.OUT_NODE
    legacy_size = np.zeros((2, 4), np.int16)
    legacy_size[0, 0] = 2
    with pytest.raises(NotImplementedError):
        from_legacy_triple(value, legacy_type, legacy_size)

    opcode = np.zeros((2, 4), np.int32)
    opcode[0, 0] = NType.TFUNC | NType.OUT_NODE
    with pytest.raises(NotImplementedError):
        to_legacy_triple((opcode, value, legacy_size.astype(np.int32)))


def test_from_legacy_triple_rejects_unknown_type_markers():
    triple = (
        np.zeros((1, 4), np.float32),
        np.full((1, 4), 9, np.int16),
        np.zeros((1, 4), np.int16),
    )
    with pytest.raises(ValueError):
        from_legacy_triple(*triple)


def test_to_legacy_triple_rejects_an_out_of_range_function_id():
    channels = _channels([99, NType.VAR], [0.0, 0.0], [2, 1])
    with pytest.raises(ValueError, match="outside"):
        to_legacy_triple(channels)


# --------------------------------------------------------------------------- #
# validation rejects every malformed tree
# --------------------------------------------------------------------------- #
def _channels(opcode, payload, size, width=8):
    opcode = np.asarray(opcode, np.int32).reshape(1, -1)
    payload = np.asarray(payload, np.float64).reshape(1, -1).astype(np.float32)
    size = np.asarray(size, np.int32).reshape(1, -1)
    pad = width - opcode.shape[1]
    assert pad >= 0
    return (
        np.pad(opcode, ((0, 0), (0, pad))),
        np.pad(payload, ((0, 0), (0, pad))),
        np.pad(size, ((0, 0), (0, pad))),
    )


def test_validate_accepts_a_hand_built_tree():
    # if(x[0] > 0, sin(x[1]), x[2])  -- pre-order, 5 nodes
    opcode = [Func.IF, NType.VAR, Func.SIN, NType.VAR, NType.VAR]
    payload = [0.0, 0.0, 0.0, 1.0, 2.0]
    size = [5, 1, 2, 1, 1]
    validate_channels(_channels(opcode, payload, size), input_len=3)


@pytest.mark.parametrize(
    "case, opcode, payload, size, match",
    [
        ("empty-row", [0], [0.0], [0], "is not positive"),
        ("past-max-nodes", [Func.SIN, NType.VAR], [0.0, 0.0], [9, 1], "exceeds max_nodes"),
        ("zero-child", [Func.ADD, NType.VAR, NType.VAR], [0, 0, 0], [3, -1, 1], "not positive"),
        (
            "child-outside-subtree",
            [Func.ADD, NType.VAR, NType.VAR],
            [0, 0, 0],
            [3, 5, 1],
            "falls outside the subtree",
        ),
        (
            "size-disagrees-with-children",
            [Func.ADD, NType.VAR, NType.VAR],
            [0, 0, 0],
            [4, 1, 1],
            "disagrees with its children",
        ),
        ("leaf-with-function-opcode", [Func.MUL], [0.0], [1], "must carry NType.VAR"),
        ("function-too-small", [Func.ADD, NType.VAR], [0, 0], [2, 1], "too small for"),
        ("unknown-opcode", [99, NType.VAR, NType.VAR, NType.VAR], [0, 0, 0, 0], [4, 1, 1, 1], "neither a leaf marker"),
        ("subtree-past-row", [Func.IF, NType.VAR, NType.VAR, NType.VAR], [0, 0, 0, 0], [7, 1, 1, 1], "disagrees with its children"),
    ],
)
def test_validate_rejects_malformed_trees(case, opcode, payload, size, match):
    channels = _channels(opcode, payload, size)
    with pytest.raises(ValueError) as excinfo:
        validate_channels(channels, input_len=3)
    message = str(excinfo.value)
    assert match in message, (case, message)
    assert "row 0" in message, (case, message)


def test_validate_rejects_an_out_of_range_variable_index():
    channels = _channels([NType.VAR], [7.0], [1])
    with pytest.raises(ValueError, match="variable index"):
        validate_channels(channels, input_len=3)
    validate_channels(channels)  # without input_len the check is skipped


def test_validate_rejects_shape_mismatches():
    opcode, payload, size = _channels([NType.VAR], [0.0], [1])
    with pytest.raises(ValueError, match="share one shape"):
        validate_channels((opcode, payload[:, :-1], size))
    with pytest.raises(ValueError, match="2-D"):
        validate_channels((opcode[None], payload[None], size[None]))
    with pytest.raises(ValueError, match="one individual must share one length"):
        validate_channels((opcode[0], payload[0, :-1], size[0]))


def test_validate_accepts_the_single_individual_form():
    # The three 1-D channel rows of one tree are a population of one (this is
    # what `GPResult.best_channels` hands back).
    opcode, payload, size = _channels([Func.SIN, NType.VAR], [0.0, 0.0], [2, 1])
    single = (opcode[0], payload[0], size[0])
    validate_channels(single, input_len=2)  # must not raise
    assert row_length(single).tolist() == [2]  # the host helpers take it too
    np.testing.assert_array_equal(
        live_mask(single), live_mask((opcode, payload, size))
    )


# --------------------------------------------------------------------------- #
# state helpers
# --------------------------------------------------------------------------- #
def test_replace_builds_a_new_state():
    state = sampled(4)
    fitness = evoxc.full((4,), 1.5, dtype=evoxc.float32)
    updated = replace(state, fitness=fitness)
    assert updated is not state
    assert updated.opcode is state.opcode
    np.testing.assert_allclose(updated.fitness.numpy(), np.full(4, 1.5, np.float32))
    assert np.all(np.isneginf(state.fitness.numpy()))
