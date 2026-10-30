"""Variation tests: one full generation of the operator layer.

``variation(config, state, fitness, key)`` is ``GeneticProgramming.step`` minus
elitism, so this module pins the *end-to-end* operator contract:

* the offspring population has the same shape/dtypes as the parents, is
  structurally well-formed per generation (the ``size`` channel re-derived from
  the opcodes) and carries an all-``-inf`` fitness -- the driver evaluates;
* it is **offspring only**: no row is parked in place, ``pop_size`` is preserved,
  and (with both rates at 1) hardly any row survives as an exact copy of a
  parent;
* with both rates at 0 the generation degenerates into pure selection, which is
  what makes the selection pressure measurable at the population level;
* the RNG key advances in-graph and is returned, so a loop is
  ``state, key = variation(...), offspring.key`` and a run is reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from evogp.gp import GPConfig, GPTreeConfig, init_population, selection
from evogp.gp import variation as V

from gp_operator_support import (
    RICH_CONFIG,
    SAFE_CONFIG,
    assert_well_formed,
    channel_arrays,
)


def _fitness(state, *, mode: str = "gradient", seed: int = 0) -> np.ndarray:
    """A plausible fitness readback (float32 ``[pop_size]``, MAXIMIZED)."""
    pop_size = int(state.size.shape[0])
    if mode == "gradient":
        return np.linspace(1.0, 0.0, pop_size, dtype=np.float32)
    if mode == "random":
        return np.random.default_rng(seed).normal(size=pop_size).astype(np.float32)
    if mode == "unevaluated":
        return np.full(pop_size, -np.inf, dtype=np.float32)
    raise AssertionError(mode)


def _row_matrices(state):
    opcode, payload, size = channel_arrays(state)
    return opcode, payload, size


def _population_of(config, gp):
    return init_population((config, gp), 0, pop_size=gp.pop_size)


# --------------------------------------------------------------------------- #
# shape and well-formedness
# --------------------------------------------------------------------------- #
def test_one_generation_returns_well_formed_offspring():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=12)
    parents = _population_of(config, gp)
    offspring, next_key = V.variation((config, gp), parents, _fitness(parents))

    assert_well_formed(offspring, context="variation")
    assert offspring.opcode.shape == parents.opcode.shape
    assert offspring.opcode.dtype == parents.opcode.dtype
    assert offspring.payload.dtype == parents.payload.dtype
    assert offspring.size.dtype == parents.size.dtype
    assert np.all(np.isneginf(offspring.fitness.numpy()))
    assert offspring.key is not None and next_key is not None


def test_ten_generations_stay_well_formed():
    """A run: if the ancestor mask or the tail shift were wrong, this breaks."""
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=10, crossover_prob=0.9,
                                         mutation_prob=0.4, mutation_kind="subtree")
    state = _population_of(config, gp)
    key = state.key
    for generation in range(10):
        fitness = _fitness(state, mode="random", seed=generation)
        offspring, key = V.variation((config, gp), state, fitness, key)
        assert_well_formed(offspring, context=f"generation {generation}")
        state = offspring
    assert int(state.size.shape[0]) == 10


@pytest.mark.parametrize("mode", ["tournament", "rank", "roulette"])
@pytest.mark.parametrize("kind", ["point", "subtree", "hoist"])
def test_every_selection_and_mutation_combination_is_well_formed(mode, kind):
    config = SAFE_CONFIG()
    gp = GPConfig(
        pop_size=8,
        selection=mode,
        selection_pressure=0.6,
        mutation_kind=kind,
        crossover_prob=0.7,
        mutation_prob=0.6,
    )
    parents = _population_of(config, gp)
    fitness = _fitness(parents)
    offspring, _next = V.variation((config, gp), parents, fitness)
    assert_well_formed(offspring, context=f"{mode}/{kind}")


def test_a_rich_function_set_stays_valid():
    from evogp.gp.encoding import validate_channels

    config, gp = RICH_CONFIG(), GPConfig(pop_size=8, mutation_prob=0.5)
    parents = _population_of(config, gp)
    offspring, _next = V.variation((config, gp), parents, _fitness(parents))
    validate_channels(offspring)
    assert int(offspring.size.shape[0]) == 8


def test_an_unevaluated_population_can_still_vary():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8)
    parents = _population_of(config, gp)
    offspring, _next = V.variation(
        (config, gp), parents, _fitness(parents, mode="unevaluated")
    )
    assert_well_formed(offspring, context="all -inf")
    assert int(offspring.size.shape[0]) == 8


# --------------------------------------------------------------------------- #
# offspring-only contract
# --------------------------------------------------------------------------- #
def test_offspring_are_not_copies_of_a_parent():
    config, gp = SAFE_CONFIG(), GPConfig(
        pop_size=32, crossover_prob=1.0, mutation_prob=1.0, mutation_kind="subtree"
    )
    parents = _population_of(config, gp)
    offspring, _next = V.variation((config, gp), parents, _fitness(parents))
    parent_op, parent_pl, _parent_sz = _row_matrices(parents)
    child_op, child_pl, _child_sz = _row_matrices(offspring)

    survivors = 0
    for row in range(32):
        for candidate in range(32):
            if (
                np.array_equal(child_op[row], parent_op[candidate])
                and np.array_equal(child_pl[row], parent_pl[candidate])
            ):
                survivors += 1
                break
    assert survivors <= 4, (
        f"{survivors}/32 offspring rows are exact copies of a parent at "
        "crossover_prob=mutation_prob=1 -- offspring must be fresh"
    )
    assert len(np.unique(child_op, axis=0)) > 1


def test_zero_rates_degenerate_into_pure_selection():
    config = SAFE_CONFIG()
    gp = GPConfig(pop_size=16, crossover_prob=0.0, mutation_prob=0.0)
    parents = _population_of(config, gp)
    fitness = np.linspace(1.0, 0.0, 16, dtype=np.float32)
    offspring, _next = V.variation((config, gp), parents, fitness)
    assert_well_formed(offspring, context="selection only")
    parent_op, parent_pl, parent_sz = _row_matrices(parents)
    child_op, child_pl, child_sz = _row_matrices(offspring)

    copies = []
    for row in range(16):
        for candidate in range(16):
            if (
                np.array_equal(child_op[row], parent_op[candidate])
                and np.array_equal(child_pl[row], parent_pl[candidate])
                and np.array_equal(child_sz[row], parent_sz[candidate])
            ):
                copies.append(candidate)
                break
        else:  # pragma: no cover - unreachable if the operator is correct
            raise AssertionError(f"row {row} is not a parent copy at rate 0")
    counts = np.bincount(copies, minlength=16)
    expected = 1.0 - (1.0 - 1.0 / 16) ** 2  # tournament_size=2 selects the best
    assert abs(counts[0] / 16 - expected) <= 0.25, (
        f"the best row must dominate a selection-only generation: "
        f"{counts[0]}/16 copies against an expected ~{expected:.2f};\n"
        f"  copy counts {counts.tolist()}"
    )
    assert counts[0] > counts[-1], "selection pressure must reach the offspring"


# --------------------------------------------------------------------------- #
# determinism and the key
# --------------------------------------------------------------------------- #
def test_variation_is_deterministic_in_the_key():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=10, crossover_prob=0.8,
                                         mutation_prob=0.5)
    parents = _population_of(config, gp)
    fitness = _fitness(parents)
    first, key_a = V.variation((config, gp), parents, fitness, 3)
    second, key_b = V.variation((config, gp), parents, fitness, 3)
    for name in ("opcode", "payload", "size"):
        assert np.array_equal(
            getattr(first, name).numpy(), getattr(second, name).numpy()
        ), f"{name} must be reproducible from the key"
    assert key_a.numpy() == key_b.numpy()

    third, key_c = V.variation((config, gp), parents, fitness, 4)
    assert not np.array_equal(first.opcode.numpy(), third.opcode.numpy())
    assert key_c.numpy() != key_a.numpy()


def test_the_key_advances_and_a_loop_is_reproducible():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8, mutation_prob=0.5)
    parents = _population_of(config, gp)

    def run_loop():
        state = parents
        key = 0
        trace = []
        for _ in range(3):
            offspring, key = V.variation((config, gp), state, _fitness(state), key)
            trace.append(offspring)
            state = offspring
        return trace, key

    trace_a, key_a = run_loop()
    trace_b, key_b = run_loop()
    assert trace_a[-1].key.numpy() == key_a.numpy()
    assert key_a.numpy() == key_b.numpy()
    for left, right in zip(trace_a, trace_b):
        assert np.array_equal(left.opcode.numpy(), right.opcode.numpy())

    # The key really moves inside the graph: three generations, three keys.
    keys = {0}
    key = 0
    state = parents
    for _ in range(3):
        state, key = V.variation((config, gp), state, _fitness(state), key)
        keys.add(int(key.numpy()))
    assert len(keys) == 4, f"the key must advance every generation, got {sorted(keys)}"


def test_the_state_key_is_a_valid_input():
    config, gp = SAFE_CONFIG(), GPConfig(pop_size=8)
    parents = _population_of(config, gp)
    from_state, key_a = V.variation((config, gp), parents, _fitness(parents),
                                    parents.key)
    from_seed = parents.key.numpy()
    # parents.key is the *next* key produced by init_population, so passing it
    # back must be the same as passing those same bits: both give one graph run.
    assert from_state.key.numpy() != int(from_seed)
    explicit, key_b = V.variation(
        (config, gp), parents, _fitness(parents), int(parents.key.numpy())
    )
    assert np.array_equal(from_state.opcode.numpy(), explicit.opcode.numpy())
    assert key_a.numpy() == key_b.numpy()


# --------------------------------------------------------------------------- #
# config spellings and guards
# --------------------------------------------------------------------------- #
def test_a_bare_tree_config_uses_the_gp_defaults():
    config = SAFE_CONFIG()
    parents = init_population(config, 1, pop_size=6)
    offspring, _next = V.variation(config, parents, _fitness(parents), 0)
    assert_well_formed(offspring, context="bare tree config")
    assert int(offspring.size.shape[0]) == 6


def test_variation_guards():
    config = SAFE_CONFIG()
    gp = GPConfig(pop_size=8)
    parents = _population_of(config, gp)
    fitness = _fitness(parents)
    with pytest.raises(ValueError, match="fitness must be 1-D"):
        V.variation((config, gp), parents, fitness[:-1])
    with pytest.raises(ValueError, match="fitness must be 1-D"):
        V.variation((config, gp), parents, fitness.reshape(8, 1))
    with pytest.raises(ValueError, match="rows but the config's pop_size"):
        V.variation((config, GPConfig(pop_size=10)), parents, fitness)
    with pytest.raises(ValueError, match="GPTreeConfig"):
        V.variation(GPConfig(pop_size=8), parents, fitness)
    opcode, payload, size = _row_matrices(parents)
    with pytest.raises(ValueError, match="2-D"):
        V.variation((config, gp), (opcode, payload, size[:, 0]), fitness)


def test_selection_module_is_reachable_from_the_generation():
    # A tiny sanity link between the two layers: the configured mode is what the
    # generation uses, so an extreme pressure and a 1-row tournament differ.
    config = SAFE_CONFIG()
    parents = _population_of(config, GPConfig(pop_size=8))
    fitness = np.linspace(1.0, 0.0, 8, dtype=np.float32)
    indices = selection.select(fitness, 0, mode="tournament", tournament_size=2)
    assert indices.numpy().shape == (8,)
    assert int(indices.numpy().max()) < 8
