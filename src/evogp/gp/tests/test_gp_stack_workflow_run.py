"""Driver tests: the loop (``workflow.run``) and the reported result.

``run`` is the plain Python loop of ``DESIGN.md`` section "Evaluation and the
driver": one traced operator graph plus one ``evaluate`` call per generation,
with elitism applied by the driver.  This module pins the properties that makes
it a usable driver:

* one ``evaluate`` call and one history entry per generation, with
  ``generations=0`` and ``generations=1`` behaving as documented;
* the per-generation best fitness never decreases (elitism) and
  ``GPResult.best_fitness`` is its maximum;
* a run is reproducible from its seed alone -- the same key replays the same
  trajectory, a different key does not -- and the reported best individual is
  the one that produced ``best_fitness`` (re-evaluating its channels reproduces
  the number);
* the stack actually converges on the smoke problem, and the package surface
  exposes the driver without evoxc or torch at import time.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from evogp.gp import (
    CustomLoss,
    GPConfig,
    GPResult,
    GPTreeConfig,
    SymbolicRegression,
    cache_stats,
    clear_cache,
    init_gp,
    row_length,
    run,
    smoke,
    step,
    to_legacy_triple,
    validate_channels,
)

from gp_operator_support import SAFE_CONFIG, channel_arrays


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _length_fitness(state) -> np.ndarray:
    """A deterministic objective: shorter trees win (MAXIMIZED).

    ``row_length`` is the shared channel parser's accessor, so this objective
    also takes a single individual (``GPResult.best_channels``) as a population
    of one -- the same contract the shipped problems have.
    """
    return (-row_length(state)).astype(np.float32)


def _config(pop_size: int, elite_count: int = 1):
    return SAFE_CONFIG(), GPConfig(pop_size=pop_size, elite_count=elite_count)


def _tiny_problem(num_data: int = 120, seed: int = 0):
    """The smoke target at test scale (same function, fewer rows)."""
    x, y = smoke.make_data(num_data, seed=seed)
    return SymbolicRegression(x, y)


# --------------------------------------------------------------------------- #
# generations = 0 / 1
# --------------------------------------------------------------------------- #
def test_zero_generations_returns_an_unevaluated_result():
    config, gp = _config(8)
    calls = []
    result = run(
        (config, gp), lambda state: calls.append(1) or _length_fitness(state),
        generations=0,
    )

    assert isinstance(result, GPResult)
    assert result.generations == 0
    assert result.best_fitness_history == ()
    assert result.median_fitness_history == ()
    assert result.best_fitness == -np.inf
    assert result.best_channels is None
    assert result.best_generation == -1
    assert calls == [], "nothing is evaluated when no generation runs"
    assert result.state is not None and result.state.opcode.shape[0] == 8
    assert np.all(np.isneginf(np.asarray(result.state.fitness.numpy())))


def test_one_generation_evaluates_once_and_reports_that_generation():
    config, gp = _config(8)
    calls = []
    result = run(
        (config, gp), lambda state: calls.append(1) or _length_fitness(state),
        generations=1,
    )

    assert len(calls) == 1
    assert result.generations == 1
    assert len(result.best_fitness_history) == 1
    assert result.best_generation == 0
    assert result.best_fitness == result.best_fitness_history[0]
    assert result.best_channels is not None
    # the reported state is the generation AFTER the evaluated one
    assert result.state is not None and result.state.opcode.shape[0] == 8


# --------------------------------------------------------------------------- #
# elitism: the trajectory is monotone
# --------------------------------------------------------------------------- #
def test_the_per_generation_best_never_decreases():
    config, gp = _config(16, elite_count=2)
    result = run((config, gp), _length_fitness, generations=12)

    history = np.asarray(result.best_fitness_history)
    assert history.shape == (12,)
    assert np.all(np.diff(history) >= 0.0), (
        f"elitism must make the best fitness monotone, got {history.tolist()}"
    )
    assert result.best_fitness == float(history.max()) == float(history[-1])


def test_the_elite_survives_a_generation_of_the_loop():
    config, gp = _config(10, elite_count=2)
    state = init_gp((config, gp), 0)
    fitness = _length_fitness(state)
    elite = np.argsort(-fitness, kind="stable")[:2]
    parent_opcode, parent_payload, parent_size = channel_arrays(state)

    children = step((config, gp), state, lambda _state: fitness)

    child_opcode, child_payload, child_size = channel_arrays(children)
    for slot, source in enumerate(elite):
        np.testing.assert_array_equal(child_opcode[slot], parent_opcode[source])
        np.testing.assert_array_equal(child_payload[slot], parent_payload[source])
        np.testing.assert_array_equal(child_size[slot], parent_size[source])


# --------------------------------------------------------------------------- #
# one evaluate call per generation, and the callback
# --------------------------------------------------------------------------- #
def test_evaluate_is_called_exactly_once_per_generation():
    config, gp = _config(8)
    calls = []

    def evaluate(state):
        calls.append(state)  # hold the reference: `id()` alone can be reused
        return _length_fitness(state)

    result = run((config, gp), evaluate, generations=5)
    assert len(calls) == 5 == result.generations
    assert len({id(state) for state in calls}) == 5, (
        "every generation hands over its own population object"
    )


def test_the_callback_sees_every_generation_in_order():
    config, gp = _config(8)
    seen = []

    def callback(generation, state, fitness):
        seen.append((generation, int(fitness.shape[0]), float(np.max(fitness))))

    result = run((config, gp), _length_fitness, generations=4, callback=callback)
    assert [entry[0] for entry in seen] == [0, 1, 2, 3]
    assert [entry[1] for entry in seen] == [8, 8, 8, 8]
    assert [entry[2] for entry in seen] == list(result.best_fitness_history)


# --------------------------------------------------------------------------- #
# determinism and the reported best individual
# --------------------------------------------------------------------------- #
def test_the_same_key_replays_the_same_trajectory():
    config, gp = _config(12, elite_count=2)
    first = run((config, gp), _length_fitness, generations=6, key=3)
    second = run((config, gp), _length_fitness, generations=6, key=3)

    assert first.best_fitness_history == second.best_fitness_history
    for lhs, rhs in zip(first.best_channels, second.best_channels):
        np.testing.assert_array_equal(lhs, rhs)


def test_a_different_key_searches_a_different_population():
    config, gp = _config(12, elite_count=2)
    first = run((config, gp), _length_fitness, generations=6, key=0)
    other = run((config, gp), _length_fitness, generations=6, key=1)

    first_opcode, _payload, _size = channel_arrays(first.state)
    other_opcode, _payload, _size = channel_arrays(other.state)
    assert not np.array_equal(first_opcode, other_opcode), (
        "the seed must change the RNG stream the population is drawn from"
    )


def test_the_reported_best_channels_reproduce_the_reported_fitness():
    config, gp = _config(12, elite_count=2)
    result = run((config, gp), _length_fitness, generations=6)

    opcode, payload, size = result.best_channels
    assert opcode.shape == payload.shape == size.shape == (config.max_nodes,)
    assert 1 <= int(size[0]) <= config.max_nodes
    # `best_channels` is a single individual and is fed back AS IS: the one
    # shared parser turns it into a population of one.
    again = _length_fitness(result.best_channels)
    assert float(again[0]) == float(result.best_fitness)
    assert again.shape == (1,)


def test_a_real_problem_reproduces_the_reported_best_fitness():
    problem = _tiny_problem()
    config = smoke.make_configs(pop_size=24, max_nodes=32)
    # The tree width must match the descriptor the problem sees; 32 slots with
    # the smoke's function set/levels is a valid configuration.
    result = run(config, problem.evaluate, generations=4)

    again = problem.evaluate(result.best_channels)
    assert float(again[0]) == float(result.best_fitness)
    assert np.isfinite(result.best_fitness)


def test_the_reported_best_individual_is_a_usable_single_individual():
    """``result.best_channels`` is the natural user input, everywhere."""
    problem = _tiny_problem()
    config = smoke.make_configs(pop_size=24, max_nodes=32)
    result = run(config, problem.evaluate, generations=4)
    opcode, payload, size = result.best_channels
    width = config[0].max_nodes

    # the reported shape is the three 1-D channel rows of ONE tree
    assert opcode.shape == payload.shape == size.shape == (width,)
    assert opcode.dtype == np.int32 and payload.dtype == np.float32
    assert size.dtype == np.int32

    validate_channels(result.best_channels, input_len=config[0].input_len)

    fitness = problem.evaluate(result.best_channels)
    assert fitness.shape == (1,)
    assert abs(float(fitness[0]) - float(result.best_fitness)) <= 1e-6

    predictions = problem.predict(result.best_channels)
    assert predictions.shape == (1, problem.x.shape[0])

    # the legacy bridge sees the same one-row triple as for the 2-D form
    single = to_legacy_triple(result.best_channels)
    paired = to_legacy_triple((opcode[None, :], payload[None, :], size[None, :]))
    for lhs, rhs in zip(single, paired):
        assert lhs.shape == rhs.shape == (1, width)
        np.testing.assert_array_equal(lhs, rhs)
    assert single[0].dtype == np.float32
    assert single[1].dtype == np.int16 and single[2].dtype == np.int16


# --------------------------------------------------------------------------- #
# the objective surface: a CustomLoss instance and a bare tree config
# --------------------------------------------------------------------------- #
def test_a_custom_loss_problem_drives_the_run_through_its_call_operator():
    x, y = smoke.make_data(80, seed=0)
    problem = CustomLoss(x, lambda predictions, targets: np.mean(
        (predictions - targets[None, :]) ** 2, axis=1
    ), targets=y)
    config = smoke.make_configs(pop_size=16, max_nodes=32)
    result = run(config, problem, generations=3)  # `problem` is callable

    assert len(result.best_fitness_history) == 3
    assert np.isfinite(result.best_fitness)
    # the same run through the bound method is bit-identical
    again = run(config, problem.evaluate, generations=3)
    assert again.best_fitness_history == result.best_fitness_history


def test_a_bare_tree_config_needs_and_accepts_pop_size():
    config = SAFE_CONFIG()
    with pytest.raises(ValueError, match="pop_size"):
        run(config, _length_fitness, generations=1)
    result = run(config, _length_fitness, generations=1, pop_size=8)
    assert result.best_fitness_history != ()
    assert result.state.opcode.shape == (8, config.max_nodes)


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("generations", [-1, 1.5, "3", None, True])
def test_generations_must_be_a_non_negative_int(generations):
    config, gp = _config(8)
    with pytest.raises(ValueError, match="generations"):
        run((config, gp), _length_fitness, generations=generations)


def test_a_fitness_contract_break_fails_the_run():
    config, gp = _config(8)
    with pytest.raises(ValueError, match="pop_size=8"):
        run((config, gp), lambda _state: np.zeros(3, np.float32), generations=2)


# --------------------------------------------------------------------------- #
# the convergence smoke
# --------------------------------------------------------------------------- #
def test_a_short_run_improves_the_fitness_materially():
    result = smoke.run_smoke(
        pop_size=64, generations=30, num_data=300, tournament_size=4, verbose=False
    )

    history = np.asarray(result.best_fitness_history)
    assert history.shape == (30,)
    assert np.all(np.diff(history) >= 0.0)
    assert history[-1] > history[0] + 0.02, (
        "the driver must actually improve on the random generation, got "
        f"{history[0]:.5f} -> {history[-1]:.5f}"
    )
    assert result.best_fitness > -0.5  # far better than predicting the mean (~ -1.0)
    assert result.best_channels is not None


# --------------------------------------------------------------------------- #
# cache surface
# --------------------------------------------------------------------------- #
def test_the_stack_cache_surface_aggregates_and_clears_every_layer():
    config, gp = _config(8)
    run((config, gp), _length_fitness, generations=2)
    stats = cache_stats()
    assert set(stats) >= {"graphs", "compile_cache_size", "layers"}
    assert set(stats["layers"]) == {
        "encoding",
        "selection",
        "crossover",
        "mutation",
        "variation",
        "workflow",
        "problem",
    }
    assert stats["graphs"] == sum(
        layer["graphs"] for layer in stats["layers"].values()
    )
    assert stats["graphs"] > 0, "a run memoizes its graphs"

    clear_cache()
    assert cache_stats()["graphs"] == 0


# --------------------------------------------------------------------------- #
# the public surface of the package
# --------------------------------------------------------------------------- #
SUBTREE = pathlib.Path(__file__).resolve().parents[1]
SRC = SUBTREE.parents[1]

#: Every name the driver adds to ``evogp.gp``.
WORKFLOW_NAMES = ("init_gp", "step", "run", "GPResult", "cache_stats", "clear_cache")


def test_the_package_exposes_the_driver_without_evoxc_or_torch():
    code = (
        "import sys\n"
        "import evogp.gp as gp\n"
        "missing = [name for name in "
        f"{WORKFLOW_NAMES!r} + ('apply_variation', 'apply_crossover', 'select',\n"
        "                      'predict', 'sr_fitness_graph', 'SymbolicRegression',\n"
        "                      'CustomLoss', 'GPState', 'init_population')\n"
        "           if name not in gp.__all__ or not hasattr(gp, name)]\n"
        "assert not missing, missing\n"
        "assert 'evogp.gp.workflow' in sys.modules and len(gp.__all__) >= 37\n"
        "loaded = sorted(n for n in ('torch', 'evox', 'evoxc') if n in sys.modules)\n"
        "assert not loaded, f'importing evogp.gp pulled in {loaded}'\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)  # deliberately NOT the evoxc checkout
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok", result.stdout


def test_apply_variation_and_apply_crossover_keep_the_modules_importable():
    import evogp.gp as gp
    import evogp.gp.crossover
    import evogp.gp.variation

    assert gp.variation.variation is gp.apply_variation
    assert gp.crossover.crossover is gp.apply_crossover
