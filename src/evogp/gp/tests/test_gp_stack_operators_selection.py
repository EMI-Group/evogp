"""Selection tests: pressure ordering, the three modes, guards, determinism.

Selection is a pure function of a fitness vector and a key -- it needs no engine
knowledge -- so everything here is a distributional check with an *analytic*
target rather than a golden value:

* a **strict fitness gradient** makes tournament selection pick rank ``r`` with
  probability ``((n-r)**k - (n-r-1)**k) / n**k`` (the chance that a uniform
  ``k``-subset of the ranks is topped by ``r``); the measured frequencies are
  compared against that closed form, and against the usual headline number
  ``1 - (1 - 1/n)**k`` for the best individual;
* an **all-equal fitness** population must be sampled uniformly in every mode
  (the pressure ordering has nothing to order);
* ``rank`` mode's mass is ``rank_probabilities`` (linear in the rank, ratio
  ``(1+sp)`` between best and worst) and ``roulette`` mode's is the *shifted*
  fitness ``f - min(f)`` (the legacy normalisation, so the worst row of an
  all-positive population is never selected), with ``-inf`` rows getting zero.

Determinism, the range of the returned indices and every guard are pinned too.
"""

from __future__ import annotations

import numpy as np
import pytest

from evogp.gp import selection as S


def _draws(fitness, *, draws: int, mode: str, **kwargs) -> np.ndarray:
    """``[draws, pop_size]`` matrix of selected indices (one call per key)."""
    rows = []
    for key in range(draws):
        indices = S.select(fitness, key, mode=mode, **kwargs)
        values = indices.numpy()
        assert values.shape == (len(fitness),)
        assert values.dtype == np.int32
        assert np.all(values >= 0) and np.all(values < len(fitness))
        rows.append(values)
    return np.asarray(rows)


def _frequencies(fitness, *, draws: int, mode: str, **kwargs) -> np.ndarray:
    """Selection frequency per row of ``fitness``, averaged over ``draws``."""
    matrix = _draws(fitness, draws=draws, mode=mode, **kwargs)
    counts = np.bincount(matrix.reshape(-1), minlength=len(fitness))
    return counts / counts.sum()


def _assert_close(measured, expected, total: int, *, context: str, z: float = 5.0):
    """Compare frequencies against a probability vector (normal-approx bands)."""
    measured = np.asarray(measured, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert measured.shape == expected.shape
    bands = z * np.sqrt(np.maximum(expected * (1.0 - expected), 1e-12) / total)
    worst = np.argmax(np.abs(measured - expected))
    assert np.all(np.abs(measured - expected) <= bands + 1.0 / total), (
        f"{context}: frequency mismatch at index {worst}: measured "
        f"{measured[worst]:.4f}, expected {expected[worst]:.4f} "
        f"(band {bands[worst]:.4f} over {total} draws)\n"
        f"  measured {np.round(measured, 4).tolist()}\n"
        f"  expected {np.round(expected, 4).tolist()}"
    )


# --------------------------------------------------------------------------- #
# tournament pressure
# --------------------------------------------------------------------------- #
def test_tournament_picks_a_strict_gradient_by_the_closed_form():
    pop_size, tournament_size, draws = 8, 3, 400
    fitness = np.arange(pop_size, 0, -1, dtype=np.float32)  # 8 .. 1, no ties
    measured = _frequencies(
        fitness, draws=draws, mode="tournament", tournament_size=tournament_size
    )
    # P(rank r wins a k-subset) = ((n-r)**k - (n-r-1)**k) / n**k
    ranks = np.arange(pop_size)
    better = pop_size - ranks
    expected = (better**tournament_size - (better - 1) ** tournament_size) / (
        pop_size**tournament_size
    )
    expected = expected / expected.sum()
    _assert_close(
        measured, expected, draws * pop_size,
        context=f"tournament k={tournament_size}, pop={pop_size}",
    )
    # The headline number for the best individual.
    best = 1.0 - (1.0 - 1.0 / pop_size) ** tournament_size
    assert abs(measured[0] - best) <= 5.0 * np.sqrt(best * (1 - best) / (draws * pop_size))
    assert measured[0] > measured[-1] * 5.0, "the best row must dominate the worst"


def test_tournament_pressure_grows_with_the_tournament_size():
    pop_size, draws = 16, 200
    fitness = np.linspace(1.0, 0.0, pop_size, dtype=np.float32)
    values = []
    for tournament_size in (1, 2, 4, 8):
        measured = _frequencies(
            fitness, draws=draws, mode="tournament", tournament_size=tournament_size
        )
        best = 1.0 - (1.0 - 1.0 / pop_size) ** tournament_size
        tolerance = 5.0 * np.sqrt(best * (1 - best) / (draws * pop_size))
        assert abs(measured[0] - best) <= tolerance, (
            f"k={tournament_size}: best-row frequency {measured[0]:.4f}, "
            f"expected {best:.4f} (band {tolerance:.4f})"
        )
        values.append(measured[0])
    assert values == sorted(values), (
        f"selection pressure must grow with k, got {np.round(values, 4).tolist()}"
    )


def test_tournament_of_one_is_a_uniform_draw():
    fitness = np.arange(16, 0, -1, dtype=np.float32)
    measured = _frequencies(fitness, draws=300, mode="tournament", tournament_size=1)
    _assert_close(measured, np.full(16, 1.0 / 16), 300 * 16, context="k=1")


# --------------------------------------------------------------------------- #
# all-equal fitness: uniform in every mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["tournament", "roulette"])
def test_equal_fitness_is_sampled_uniformly(mode):
    pop_size, draws = 16, 300
    fitness = np.full(pop_size, 2.5, dtype=np.float32)
    measured = _frequencies(fitness, draws=draws, mode=mode)
    _assert_close(
        measured, np.full(pop_size, 1.0 / pop_size), draws * pop_size, context=mode
    )


# --------------------------------------------------------------------------- #
# rank and roulette modes
# --------------------------------------------------------------------------- #
def test_rank_probabilities_match_the_linear_archive():
    pop_size, pressure = 10, 0.5
    probs = S.rank_probabilities(pop_size, pressure)
    ranks = np.arange(pop_size)
    expected = (1.0 + pressure * (1.0 - 2.0 * ranks / (pop_size - 1))) / pop_size
    expected = expected / expected.sum()
    assert probs.shape == (pop_size,)
    assert np.allclose(probs, expected, atol=1e-6)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    # best / worst == (1 + sp) / (1 - sp)
    assert np.isclose(
        probs[0] / probs[-1], (1.0 + pressure) / (1.0 - pressure), rtol=1e-5
    )


def test_rank_probabilities_edge_pressures():
    uniform = S.rank_probabilities(8, 0.0)
    assert np.allclose(uniform, np.full(8, 1.0 / 8), atol=1e-7)
    full = S.rank_probabilities(8, 1.0)
    ranks = np.arange(8)
    assert np.allclose(full, 2.0 * (7.0 - ranks) / (8.0 * 7.0), atol=1e-7)
    assert np.isclose(full[-1], 0.0, atol=1e-7), "sp=1 leaves the worst row out"
    assert S.rank_probabilities(1, 0.5).tolist() == [1.0]


def test_rank_mode_applies_pressure_even_when_fitness_ties():
    pop_size, draws, pressure = 8, 300, 0.75
    # An all-equal population has no *ranking* to discover, but the archive's
    # rank mode still applies its probability vector -- the sort breaks the ties
    # deterministically (evoxc's argsort has no randomized tie-break), so the
    # frequency *set* is the archive's vector under some permutation.
    measured = _frequencies(
        np.full(pop_size, 2.5, np.float32),
        draws=draws,
        mode="rank",
        selection_pressure=pressure,
    )
    expected = S.rank_probabilities(pop_size, pressure)
    assert np.allclose(np.sort(measured), np.sort(expected), atol=0.02), (
        f"measured {np.round(np.sort(measured), 4).tolist()} vs expected "
        f"{np.round(np.sort(expected), 4).tolist()}"
    )


def test_rank_mode_follows_the_probability_mass():
    pop_size, draws = 8, 400
    fitness = np.arange(pop_size, 0, -1, dtype=np.float32)
    measured = _frequencies(fitness, draws=draws, mode="rank", selection_pressure=0.75)
    expected = S.rank_probabilities(pop_size, 0.75)
    _assert_close(
        measured, expected, draws * pop_size, context="rank sp=0.75"
    )


def test_roulette_is_proportional_to_the_shifted_fitness():
    fitness = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    draws = 400
    measured = _frequencies(fitness, draws=draws, mode="roulette")
    # The archive's roulette normalises by f - min(f), so the worst row (weight
    # 0) is never selected and the rest are proportional to their excess.
    expected = np.asarray([0.0, 1.0, 2.0, 3.0])
    expected = expected / expected.sum()
    _assert_close(measured, expected, draws * len(fitness), context="roulette")
    assert measured[0] == 0.0, "a zero-weight row can never be selected"


def test_roulette_skips_unevaluated_rows():
    draws = 300
    fitness = np.asarray([5.0, -np.inf, 3.0, 1.0, -np.inf], dtype=np.float32)
    measured = _frequencies(fitness, draws=draws, mode="roulette")
    assert measured[1] == 0.0 and measured[4] == 0.0, (
        f"-inf (unevaluated) rows must never be selected, got "
        f"{np.round(measured, 4).tolist()}"
    )
    # ... and the finite rows keep their shifted-fitness proportions:
    # [5, 3, 1] - 1 == [4, 2, 0] -> [2/3, 1/3, 0]
    _assert_close(
        measured,
        np.asarray([4.0, 0.0, 2.0, 0.0, 0.0]) / 6.0,
        draws * len(fitness),
        context="roulette with -inf rows",
    )


def test_roulette_handles_an_all_unevaluated_population():
    fitness = np.full(5, -np.inf, dtype=np.float32)
    values = S.select(fitness, 0, mode="roulette").numpy()
    assert values.shape == (5,)
    assert np.all((values >= 0) & (values < 5))


# --------------------------------------------------------------------------- #
# determinism, purity, guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["tournament", "rank", "roulette"])
def test_selection_is_deterministic_and_key_dependent(mode):
    rng = np.random.default_rng(3)
    fitness = rng.normal(size=16).astype(np.float32)
    first = S.select(fitness, 11, mode=mode).numpy()
    second = S.select(fitness, 11, mode=mode).numpy()
    assert np.array_equal(first, second), "the same key must give the same indices"
    others = [S.select(fitness, 12, mode=mode).numpy() for _ in range(3)]
    assert any(not np.array_equal(first, other) for other in others), (
        "a different key must give a different draw"
    )


def test_selection_does_not_touch_the_fitness_vector():
    fitness = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    before = fitness.copy()
    for mode in ("tournament", "rank", "roulette"):
        S.select(fitness, 0, mode=mode)
    assert np.array_equal(fitness, before), "selection must not mutate its input"


def test_a_single_row_population_selects_it():
    for mode in ("tournament", "rank", "roulette"):
        assert S.select(np.asarray([1.0], np.float32), 0, mode=mode).numpy().tolist() == [0]


def test_pop_size_of_one_row_is_honoured():
    fitness = np.asarray([1.0, 2.0, 3.0], np.float32)
    assert S.select(fitness, 0, pop_size=3).numpy().shape == (3,)
    with pytest.raises(ValueError, match="contradicts the fitness length"):
        S.select(fitness, 0, pop_size=4)


def test_bad_arguments_raise():
    fitness = np.asarray([1.0, 2.0, 3.0], np.float32)
    with pytest.raises(ValueError, match="1-D"):
        S.select(fitness.reshape(3, 1), 0)
    with pytest.raises(ValueError, match="must not be empty"):
        S.select(np.zeros(0, np.float32), 0)
    with pytest.raises(ValueError, match="mode"):
        S.select(fitness, 0, mode="truncation")
    with pytest.raises(ValueError, match="tournament_size"):
        S.select(fitness, 0, tournament_size=0)
    with pytest.raises(ValueError, match="tournament_size"):
        S.select(fitness, 0, tournament_size=2.5)
    with pytest.raises(ValueError, match="selection_pressure"):
        S.select(fitness, 0, mode="rank", selection_pressure=1.5)
    with pytest.raises(ValueError, match="rank_probabilities|pop_size"):
        S.rank_probabilities(0, 0.5)

def test_rank_mode_breaks_ties_towards_the_lower_row_index():
    """One tie rule across the stack: the first maximal row index wins.

    ``rank`` mode ranks with ``np.argsort(-fitness, kind="stable")`` -- the same
    rule ``variation._argmax_order`` implements for the elite slots and the same
    one the torch stack's ``argmax`` gives.  ``argsort(..., descending=True)``
    would flip a tie group's order (the backend implements ``descending`` as a
    reversal of the ascending indices), so a flat fitness vector would hand the
    TOP rank to the LAST row instead of the first.  The measured frequency
    vector therefore has to be the archive's probability vector in ROW order,
    not merely some permutation of it (the existing flat-fitness test above
    compares the sorted set, which cannot see the flip).
    """
    pop_size, draws = 8, 400
    fitness = np.full(pop_size, 1.5, np.float32)
    measured = _frequencies(
        fitness, draws=draws, mode="rank", selection_pressure=1.0
    )
    expected = S.rank_probabilities(pop_size, 1.0)
    assert expected[0] > 3 * expected[-1], "fixture invariant: rank 0 dominates"
    _assert_close(measured, expected, draws * pop_size, context="rank tie order")
    assert measured[0] == max(measured), (
        "a tie must rank row 0 first, got the frequency vector "
        f"{np.round(measured, 4).tolist()} (a descending sort would rank row "
        f"{pop_size - 1} first)"
    )
