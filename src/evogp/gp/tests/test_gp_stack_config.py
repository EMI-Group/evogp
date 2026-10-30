"""Tests of the static configuration objects (``GPTreeConfig`` / ``GPConfig``).

The config is the contract between the untraced world and the traced graphs: it
must hold static leaves only, derive every table a graph needs, and reject a
descriptor the generator could not honour (the same rules
``../tree/descriptor.py`` enforces).
"""

import dataclasses

import numpy as np
import pytest

from evogp.gp import GPConfig, GPTreeConfig
from evogp.gp._constants import FUNC_ARITY, FUNC_TYPE, MAX_FULL_DEPTH, MAX_STACK, Func

BASE = dict(
    max_nodes=64,
    input_len=3,
    using_funcs=("+", "*", "-", "sin", "if"),
    max_layer_cnt=4,
    layer_leaf_prob=0.2,
    const_prob=0.4,
    const_range=(-1.0, 1.0),
    sample_cnt=5,
)


def make(**changes):
    kwargs = dict(BASE)
    kwargs.update(changes)
    return GPTreeConfig(**kwargs)


def test_defaults_and_derived_tables():
    config = make()
    assert config.max_tree_len == config.max_nodes == 64
    assert config.input_len == 3
    assert config.output_len == 1

    # cdf: cumsum of the normalized weights, one slot per function id
    assert len(config.roulette_funcs) == Func.END
    assert config.roulette_funcs[-1] == pytest.approx(1.0)
    weights = np.diff((0.0,) + config.roulette_funcs)
    assert weights[Func.IF] == pytest.approx(1 / 5)
    assert weights[Func.SIN] == pytest.approx(1 / 5)
    assert weights[Func.ADD] == pytest.approx(1 / 5)
    assert weights[Func.COS] == 0.0

    # the depth schedule is GenerateDescriptor's
    assert config.depth2leaf_probs == (0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert len(config.depth2leaf_probs) == MAX_FULL_DEPTH
    assert config.levels == 4
    assert config.max_arity == 3
    assert config.padded_widths() == (1, 3, 9, 27)
    assert config.padded_total() == 40
    assert config.func_arity == FUNC_ARITY
    assert config.func_type == FUNC_TYPE


def test_config_is_frozen_and_hashable():
    config = make()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_nodes = 128  # type: ignore[misc]
    assert hash(config) == hash(make())
    assert make().replace(const_prob=0.2) != config
    # replace() leaves the receiver alone
    assert config.const_prob == 0.4


def test_function_set_spellings_are_equivalent():
    names = GPTreeConfig(**{**BASE, "using_funcs": ["+", "sin"]})
    weighted = GPTreeConfig(**{**BASE, "using_funcs": (("+", 1.0), ("sin", 1.0))})
    mapping = GPTreeConfig(**{**BASE, "using_funcs": {"+": 1.0, "sin": 1.0}})
    assert names.using_funcs == (("+", 1.0), ("sin", 1.0))
    assert weighted.roulette_funcs == names.roulette_funcs
    assert mapping.roulette_funcs == names.roulette_funcs
    assert names.max_arity == 2


def test_weighted_function_set_scales_the_cdf():
    config = make(using_funcs=(("+", 3.0), ("sin", 1.0)))
    weights = np.diff((0.0,) + config.roulette_funcs)
    assert weights[Func.ADD] == pytest.approx(0.75)
    assert weights[Func.SIN] == pytest.approx(0.25)


def test_a_zero_weight_function_is_still_part_of_the_layout():
    config = make(using_funcs=(("+", 1.0), ("if", 0.0)))
    assert config.max_arity == 3, "arity follows the listed set, not the weights"
    weights = np.diff((0.0,) + config.roulette_funcs)
    assert weights[Func.IF] == 0.0


def test_const_samples_are_drawn_deterministically_from_the_range():
    first = make()
    second = make()
    assert first.const_samples == second.const_samples
    assert len(first.const_samples) == 5
    assert all(-1.0 <= value <= 1.0 for value in first.const_samples)
    assert make(const_seed=1).const_samples != first.const_samples
    explicit = make(const_samples=(1.5, -2.5))
    assert explicit.const_samples == (1.5, -2.5)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"max_nodes": 0}, "max_nodes"),
        ({"max_nodes": MAX_STACK + 1}, "max_nodes"),
        ({"max_nodes": "64"}, "max_nodes"),
        ({"input_len": 0}, "input_len"),
        ({"output_len": 0}, "output_len"),
        ({"const_prob": 1.5}, "const_prob"),
        ({"out_prob": -0.1}, "out_prob"),
        ({"layer_leaf_prob": 2.0}, "layer_leaf_prob"),
        ({"max_layer_cnt": 0}, "max_layer_cnt"),
        ({"max_layer_cnt": MAX_FULL_DEPTH + 2}, "max_layer_cnt"),
        ({"using_funcs": ()}, "using_funcs"),
        ({"using_funcs": ("nope",)}, "unknown function name"),
        ({"using_funcs": ("+", "+")}, "duplicate function"),
        ({"using_funcs": (("+", -1.0),)}, r"finite and >= 0.0"),
        ({"using_funcs": (("+", 0.0),)}, "sum to 0.0"),
        ({"using_funcs": (("+", 1.0, 2.0),)}, "entries must be"),
        # a complete 3-ary tree of 6 levels needs 364 > 128 nodes
        ({"max_nodes": 128, "max_layer_cnt": 6}, "too small"),
    ],
)
def test_invalid_configs_raise_value_error(changes, match):
    with pytest.raises(ValueError, match=match):
        make(**changes)


def test_constants_must_be_given_or_drawable():
    with pytest.raises(ValueError, match="const_samples is None"):
        GPTreeConfig(
            max_nodes=16, input_len=2, using_funcs=("+",), max_layer_cnt=2
        )
    with pytest.raises(ValueError, match="sample_cnt must be positive"):
        make(sample_cnt=0)
    with pytest.raises(ValueError, match="const_samples must not be empty"):
        make(const_samples=())


def test_single_level_config_is_consistent():
    config = make(max_nodes=8, max_layer_cnt=1)
    assert config.levels == 1
    assert config.padded_widths() == (1,)
    assert config.depth2leaf_prob_at(0) == 1.0
    assert config.depth2leaf_prob_at(5) == 1.0


def test_gp_config_validation():
    config = GPConfig(pop_size=64)
    assert (config.elite_count, config.tournament_size) == (1, 2)
    assert 0.0 <= config.crossover_prob <= 1.0
    assert config.replace(pop_size=8).pop_size == 8
    assert config.pop_size == 64

    with pytest.raises(ValueError, match="pop_size"):
        GPConfig(pop_size=0)
    with pytest.raises(ValueError, match="elite_count"):
        GPConfig(pop_size=4, elite_count=4)
    with pytest.raises(ValueError, match="tournament_size"):
        GPConfig(pop_size=4, tournament_size=0)
    with pytest.raises(ValueError, match="crossover_prob"):
        GPConfig(pop_size=4, crossover_prob=2.0)
    with pytest.raises(ValueError, match="mutation_prob"):
        GPConfig(pop_size=4, mutation_prob=-0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.pop_size = 8  # type: ignore[misc]
