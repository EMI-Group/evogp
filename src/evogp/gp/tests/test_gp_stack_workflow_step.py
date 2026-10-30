"""Driver tests: one generation (``workflow.step``).

``step`` is ``GeneticProgramming.step`` for this stack: evaluate the current
population once, keep the best ``GPConfig.elite_count`` individuals verbatim and
let :func:`evogp.gp.variation.variation` fill the rest with offspring.  This
module pins exactly that:

* the elite rows come out of the *parent* state bit-identically (all three
  channels, padding included) and carry their known fitness, while every
  offspring row is ``-inf`` (unevaluated until the next call);
* ``evaluate`` is called exactly once per generation and receives a concrete
  :class:`~evogp.gp.state.GPState` -- never a symbolic tensor, i.e. the driver
  never traces an objective;
* the result is a well-formed population of the same static shape/dtypes, with
  the RNG key advanced;
* the contract failures (wrong fitness length, a non-callable objective, a
  population that disagrees with the config, a bare ``GPConfig``) raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import evoxc

from evogp.gp import GPConfig, GPState, init_population, step
from evogp.gp.config import GPConfig as _GPConfig  # noqa: F401 - import-path pin

from gp_operator_support import (
    SAFE_CONFIG,
    assert_well_formed,
    channel_arrays,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _length_fitness(state) -> np.ndarray:
    """A deterministic objective: shorter trees win (MAXIMIZED)."""
    _opcode, _payload, size = channel_arrays(state)
    return (-size[:, 0]).astype(np.float32)


def _top(count: int, fitness: np.ndarray) -> np.ndarray:
    """The expected elite rows: descending fitness, stable on ties."""
    return np.argsort(-fitness, kind="stable")[:count]


def _population(config, gp):
    return init_population((config, gp), 0)


# --------------------------------------------------------------------------- #
# the shape of a generation
# --------------------------------------------------------------------------- #
def test_step_returns_a_well_formed_population_of_the_same_shape():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=12, elite_count=2)
    parents = _population(config, gp)
    children = step((config, gp), parents, _length_fitness)

    assert isinstance(children, GPState)
    assert_well_formed(children, context="step")
    for name in ("opcode", "payload", "size"):
        assert getattr(children, name).shape == getattr(parents, name).shape
        assert getattr(children, name).dtype == getattr(parents, name).dtype
    assert children.fitness.shape == (12,)


def test_the_elite_rows_are_bit_identical_copies_of_the_parents():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=12, elite_count=3)
    parents = _population(config, gp)
    fitness = _length_fitness(parents)
    children = step((config, gp), parents, lambda _state: fitness)

    elite = _top(3, fitness)
    parent_opcode, parent_payload, parent_size = channel_arrays(parents)
    child_opcode, child_payload, child_size = channel_arrays(children)
    for slot, source in enumerate(elite):
        # The WHOLE row, padding included: an elite is a row copy, not a
        # re-generated tree, so a later generation can never mutate it.
        np.testing.assert_array_equal(child_opcode[slot], parent_opcode[source])
        np.testing.assert_array_equal(child_payload[slot], parent_payload[source])
        np.testing.assert_array_equal(child_size[slot], parent_size[source])


def test_the_next_fitness_carries_the_elite_and_marks_the_offspring_unevaluated():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=12, elite_count=3)
    parents = _population(config, gp)
    fitness = _length_fitness(parents)
    children = step((config, gp), parents, lambda _state: fitness)

    recorded = np.asarray(children.fitness.numpy())
    elite = _top(3, fitness)
    np.testing.assert_array_equal(recorded[:3], fitness[elite])
    assert np.all(np.isneginf(recorded[3:]))


def test_zero_elites_is_pure_offspring():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=10, elite_count=0)
    parents = _population(config, gp)
    children = step((config, gp), parents, _length_fitness)

    assert_well_formed(children, context="elite_count=0")
    assert np.all(np.isneginf(np.asarray(children.fitness.numpy())))


def test_an_all_unevaluated_population_still_produces_a_generation():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=10, elite_count=2)
    parents = _population(config, gp)
    children = step(
        (config, gp), parents, lambda _state: np.full(10, -np.inf, np.float32)
    )

    assert_well_formed(children, context="all -inf")
    recorded = np.asarray(children.fitness.numpy())
    assert np.all(np.isneginf(recorded[:2]))  # rows 0/1 win the tie, still -inf
    parent_opcode, _parent_payload, _parent_size = channel_arrays(parents)
    child_opcode, _child_payload, _child_size = channel_arrays(children)
    np.testing.assert_array_equal(child_opcode[0], parent_opcode[0])


def test_step_advances_the_key():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    children = step((config, gp), parents, _length_fitness)

    assert children.key is not None
    assert not np.array_equal(
        children.key.numpy(), parents.key.numpy()
    ), "the operator graph must advance the key it was given"


# --------------------------------------------------------------------------- #
# the objective contract: once per generation, on concrete data, never traced
# --------------------------------------------------------------------------- #
def test_evaluate_is_called_exactly_once_per_generation():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    state = _population(config, gp)
    calls = []

    def evaluate(population):
        calls.append(1)
        return _length_fitness(population)

    for _ in range(4):
        state = step((config, gp), state, evaluate)
    assert len(calls) == 4


def test_evaluate_receives_concrete_channels_and_never_a_symbolic_tensor():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    seen = []

    def evaluate(population):
        seen.append(population)
        for channel in (population.opcode, population.payload, population.size):
            # A SymbolicTensor has neither `numpy()` nor a concrete dtype, so
            # this is the "the driver never traces the objective" assertion.
            assert isinstance(channel, evoxc.Tensor)
            assert isinstance(channel.numpy(), np.ndarray)
        return _length_fitness(population)

    step((config, gp), parents, evaluate)
    assert len(seen) == 1 and isinstance(seen[0], GPState)


def test_a_nan_fitness_can_never_become_an_elite():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=2)
    parents = _population(config, gp)
    fitness = _length_fitness(parents)
    fitness[0] = np.nan  # a problem that leaks a NaN despite the contract
    children = step((config, gp), parents, lambda _state: fitness)

    recorded = np.asarray(children.fitness.numpy())
    assert np.all(np.isfinite(recorded[:2]))
    assert not np.any(np.isnan(recorded))
    # The NaN row is sanitized to -inf before the elite search, so it loses.
    sanitized = np.array(fitness, dtype=np.float32)
    sanitized[np.isnan(sanitized)] = -np.inf
    elite = _top(2, sanitized)
    assert 0 not in elite.tolist()
    np.testing.assert_array_equal(recorded[:2], sanitized[elite])
    parent_opcode, _parent_payload, _parent_size = channel_arrays(parents)
    child_opcode, _child_payload, _child_size = channel_arrays(children)
    for slot, source in enumerate(elite):
        np.testing.assert_array_equal(child_opcode[slot], parent_opcode[source])


# --------------------------------------------------------------------------- #
# error contract
# --------------------------------------------------------------------------- #
def test_a_fitness_of_the_wrong_length_is_rejected():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    with pytest.raises(ValueError, match="pop_size=8"):
        step((config, gp), parents, lambda _state: np.zeros(7, np.float32))
    with pytest.raises(ValueError, match="pop_size=8"):
        step((config, gp), parents, lambda _state: np.zeros((8, 1), np.float32))


def test_a_non_callable_objective_is_rejected():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    with pytest.raises(TypeError, match="callable"):
        step((config, gp), parents, np.zeros(8, np.float32))


def test_a_population_that_disagrees_with_the_config_is_rejected():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    smaller = GPConfig(pop_size=6, elite_count=1)
    with pytest.raises(ValueError, match="contradicts"):
        step((config, smaller), parents, _length_fitness)


def test_a_bare_gp_config_is_not_enough_to_vary():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, elite_count=1)
    parents = _population(config, gp)
    with pytest.raises(ValueError, match="GPTreeConfig"):
        step(gp, parents, _length_fitness)
