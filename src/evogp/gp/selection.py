"""Selection operators of the evoxc GP stack: one traced graph, indices out.

Selection consumes a **fitness vector** -- an input tensor that comes straight
from the host readback of an evaluation -- and returns the row indices of the
selected individuals.  It needs no engine knowledge and no tree structure, which
is what keeps it independent of evaluation: the tree layer's
``../algorithm/selection/`` family (``TournamentSelection``, ``RankSelection``,
``RouletteSelection``) is mirrored here as three modes of one stateless graph.

Three modes (``SELECTION_MODES``):

``tournament``
    ``tournament_size`` contenders drawn uniformly per row with
    ``evoxc.random.randint``, the winner is the ``argmax`` of their fitness
    (``TournamentSelection`` with ``best_probability = 1``).  Ties keep the
    first contender, so an all-equal-fitness population selects uniformly over
    the row axis; the best individual of a strict gradient is selected with
    probability ``1 - (1 - 1/pop) ** tournament_size``.
``rank``
    ``RankSelection``'s linear rank distribution
    ``p(r) = (1 + sp (1 - 2r/(n-1))) / n`` over the *descending* fitness order
    (``argsort``), sampled by inverse-CDF; the sampled rank is mapped back
    through the sorted indices.
``roulette``
    ``RouletteSelection``: probability proportional to the (shifted,
    non-negative) fitness, ``p_i = (f_i - min f) / sum_j (f_j - min f)``.
    An unevaluated row (``-inf``) gets weight 0 and a degenerate (all-equal)
    fitness vector falls back to the uniform distribution instead of dividing
    by zero.

Why inverse-CDF and not ``multinomial``: ``evoxc.random.multinomial`` is the one
random op without a StableHLO emitter (numpy backend only), so it is banned in
this subtree; and ``evoxc`` has no ``searchsorted``.  The sampled index is
therefore the number of cdf entries ``<= u``, accumulated as one ``sum`` over a
``[n, pop]`` comparison (the same construction
:func:`evogp.gp.encoding._generator_fn` uses for its function-id lookup).  That
temporary is quadratic in the population size -- acceptable for the opt-in
``rank``/``roulette`` modes, while the default ``tournament`` mode is linear and
draws only ``randint``/``gather``/``argmax``.

Public surface
--------------
``select(fitness, key, ...)``
    Host-facing: build/get the memoized graph of one static configuration, run
    it on the fitness vector, return the ``int32`` index tensor.
``select_indices(config, fitness, key)``
    The trace-time counterpart (``config`` is the resolved knob tuple), used by
    :mod:`evogp.gp.variation` inside its own graph.
``cache_stats()`` / ``clear_cache()``
    The module's graph memo, like ``encoding.cache_stats``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _runtime, encoding
from .config import SELECTION_MODES

__all__ = [
    "SELECTION_MODES",
    "select",
    "select_indices",
    "rank_probabilities",
    "cache_stats",
    "clear_cache",
]

#: ``(static configuration) -> (graph closure, executable)``.
_EXECUTABLE_MEMO: dict = {}


def cache_stats() -> dict:
    """Counters of the selection layer (graph memo + shared compile cache)."""
    return _runtime.stats(_EXECUTABLE_MEMO)


def clear_cache() -> None:
    """Drop every memoized selection graph."""
    _runtime.reset(_EXECUTABLE_MEMO)


# --------------------------------------------------------------------------- #
# static tables and validation
# --------------------------------------------------------------------------- #
def _check_mode(mode: str) -> str:
    if mode not in SELECTION_MODES:
        raise ValueError(
            f"unknown selection mode {mode!r}; the modes are {SELECTION_MODES}"
        )
    return mode


def _check_pressure(selection_pressure: float) -> float:
    pressure = float(selection_pressure)
    if not 0.0 <= pressure <= 1.0:
        raise ValueError(
            f"selection_pressure must be in [0.0, 1.0], got {selection_pressure}"
        )
    return pressure


def rank_probabilities(pop_size: int, selection_pressure: float) -> np.ndarray:
    """``RankSelection``'s selection probability per rank (rank 0 = best).

    ``p(r) = (1 + sp (1 - 2r / (n - 1))) / n`` -- linear in the rank, so the
    best individual is ``(1 + sp)`` times as likely as the worst.  Returned
    normalized to sum to 1 (the formula already does for ``n > 1``).
    """
    n = int(pop_size)
    if n <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    pressure = _check_pressure(selection_pressure)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    ranks = np.arange(n, dtype=np.float64)
    weights = (1.0 + pressure * (1.0 - 2.0 * ranks / (n - 1))) / n
    weights = weights / weights.sum()
    return weights.astype(np.float32)


# --------------------------------------------------------------------------- #
# trace-time primitives
# --------------------------------------------------------------------------- #
def _sample_cdf(evoxc_module, key, probs, pop_size: int, n_samples: int):
    """Inverse-CDF sampling of ``n_samples`` indices from ``probs`` (no searchsorted)."""
    total = evoxc_module.sum(probs, axes=None)
    cdf = evoxc_module.cumsum(evoxc_module.divide(probs, total), axis=0)
    uniform = evoxc_module.random.uniform(
        key, (int(n_samples),), 0.0, 1.0
    )
    grid = evoxc_module.broadcast(cdf, (int(n_samples), int(pop_size)))
    targets = evoxc_module.broadcast(
        evoxc_module.reshape(uniform, (int(n_samples), 1)),
        (int(n_samples), int(pop_size)),
    )
    hits = evoxc_module.cast(
        evoxc_module.less_equal(grid, targets), evoxc_module.int32
    )
    index = evoxc_module.sum(hits, axes=1)
    return evoxc_module.cast(
        evoxc_module.minimum(index, _runtime.i32(evoxc_module, int(pop_size) - 1)),
        evoxc_module.int32,
    )


def _tournament(evoxc_module, fitness, key, pop_size: int, tournament_size: int):
    """Tournament winners (``argmax`` of ``tournament_size`` uniform contenders)."""
    k = int(tournament_size)
    draw = evoxc_module.random.randint(
        key, (int(pop_size), k), 0, int(pop_size), dtype=evoxc_module.int32
    )
    contenders = evoxc_module.gather(fitness, draw, axis=0)
    best = evoxc_module.argmax(contenders, axis=1)
    offsets = _runtime.row_offsets(evoxc_module, int(pop_size), k)
    flat_index = evoxc_module.reshape(
        evoxc_module.add(
            offsets,
            evoxc_module.reshape(evoxc_module.cast(best, evoxc_module.int64), (int(pop_size), 1)),
        ),
        (int(pop_size),),
    )
    winners = evoxc_module.gather(
        evoxc_module.reshape(draw, (int(pop_size) * k,)), flat_index, axis=0
    )
    return evoxc_module.cast(winners, evoxc_module.int32)


def _rank(evoxc_module, fitness, key, pop_size: int, selection_pressure: float):
    """Rank-proportional selection (sampled rank mapped back through the order)."""
    probs = evoxc_module.constant(
        evoxc_module.from_numpy(rank_probabilities(pop_size, selection_pressure))
    )
    order = evoxc_module.argsort(fitness, descending=True)
    picked = _sample_cdf(evoxc_module, key, probs, pop_size, pop_size)
    return evoxc_module.cast(
        evoxc_module.gather(order, picked, axis=0), evoxc_module.int32
    )


def _roulette(evoxc_module, fitness, key, pop_size: int):
    """Fitness-proportional selection with the ``-inf``/degenerate guards."""
    sentinel = _runtime.f32(evoxc_module, 3.0e38)
    missing = evoxc_module.less(fitness, _runtime.f32(evoxc_module, -3.0e38))
    # `-inf - (-inf)` is NaN, so a non-finite row is replaced before the shift:
    # its weight becomes 0 after subtracting the lowest FINITE fitness.
    finite = evoxc_module.select(missing, sentinel, fitness)
    lowest = evoxc_module.min(finite, axes=None)
    base = evoxc_module.select(missing, lowest, fitness)
    low = evoxc_module.min(base, axes=None)
    high = evoxc_module.max(base, axes=None)
    flat = evoxc_module.less_equal(
        evoxc_module.subtract(high, low), _runtime.f32(evoxc_module, 0.0)
    )
    shifted = evoxc_module.subtract(base, low)
    uniform = evoxc_module.broadcast(
        _runtime.f32(evoxc_module, 1.0), (int(pop_size),)
    )
    probs = evoxc_module.select(flat, uniform, shifted)
    return _sample_cdf(evoxc_module, key, probs, pop_size, pop_size)


def select_indices(evoxc_module, fitness, key, pop_size: int, mode: str,
                   tournament_size: int = 2, selection_pressure: float = 0.5):
    """``int32 [pop_size]`` row indices (the trace-time selection primitive).

    Args:
        evoxc_module: the imported ``evoxc`` module.
        fitness: ``float32 [pop_size]`` symbolic fitness (MAXIMIZED; ``-inf``
            marks an unevaluated row).
        key: a symbolic RNG key.
        pop_size: number of rows (static).
        mode: one of :data:`SELECTION_MODES`.
        tournament_size: contenders per row (``tournament`` mode).
        selection_pressure: ``sp`` of the ``rank`` mode.

    Returns:
        int32 indices in ``[0, pop_size)``, shape ``[pop_size]``.
    """
    _check_mode(mode)
    _check_pressure(selection_pressure)
    if int(pop_size) == 1:
        # A one-row population: the only index, without a draw (a zero-width
        # `randint` range would be rejected by the backend).
        return evoxc_module.cast(
            evoxc_module.broadcast(_runtime.i32(evoxc_module, 0), (1,)),
            evoxc_module.int32,
        )
    if mode == "tournament":
        return _tournament(evoxc_module, fitness, key, pop_size, tournament_size)
    if mode == "rank":
        return _rank(evoxc_module, fitness, key, pop_size, selection_pressure)
    return _roulette(evoxc_module, fitness, key, pop_size)


# --------------------------------------------------------------------------- #
# host-facing entry point
# --------------------------------------------------------------------------- #
def _selection_fn(pop_size: int, mode: str, tournament_size: int,
                  selection_pressure: float):
    """The traced ``(fitness, key) -> indices`` graph of one static config."""

    def selection(fitness, key):
        return (select_indices(_runtime.evoxc(), fitness, key, pop_size, mode,
                               tournament_size, selection_pressure),)

    return selection


def select(
    fitness,
    key,
    *,
    pop_size: Optional[int] = None,
    mode: str = "tournament",
    tournament_size: int = 2,
    selection_pressure: float = 0.5,
    backend: str = "numpy",
    device: str = "cpu",
):
    """Select ``pop_size`` individuals from ``fitness`` (``int32`` indices).

    One traced graph per ``(pop_size, mode, tournament_size,
    selection_pressure, backend, device)``, memoized in this module and compiled
    through the shared :class:`evoxc.CompileCache`; only the fitness values and
    the key vary per call.

    Args:
        fitness: ``float32 [pop_size]`` fitness (MAXIMIZED, ``-inf`` =
            unevaluated).  A numpy array or an ``evoxc.Tensor`` from a host
            readback.
        key: an int seed (``evoxc.random.key(seed)``) or a concrete key tensor.
        pop_size: cross-check of the fitness length (taken from it when
            omitted).
        mode: ``"tournament"`` (default), ``"rank"`` or ``"roulette"``.
        tournament_size: contenders per tournament (``>= 1``).
        selection_pressure: rank-mode pressure ``sp`` in ``[0, 1]``.
        backend: evoxc backend name (``"numpy"`` default).
        device: evoxc device string.

    Returns:
        An ``evoxc.Tensor`` of ``int32`` indices, shape ``[pop_size]``, each in
        ``[0, pop_size)`` (read back with ``.numpy()``).

    Raises:
        ValueError: on a non-1-D fitness vector, a length mismatch with
            ``pop_size``, an unknown mode, a tournament size ``< 1`` or a
            selection pressure outside ``[0, 1]``.
        TypeError: on a fitness dtype that is not float-convertible.
    """
    _check_mode(mode)
    if not isinstance(tournament_size, int) or isinstance(tournament_size, bool):
        raise ValueError(
            f"tournament_size must be an int, got {tournament_size!r}"
        )
    if tournament_size < 1:
        raise ValueError(
            f"tournament_size must be >= 1, got {tournament_size}"
        )
    _check_pressure(selection_pressure)

    values = np.asarray(encoding._as_numpy(fitness, "fitness"))
    if values.ndim != 1:
        raise ValueError(
            f"fitness must be 1-D [pop_size], got shape {values.shape}"
        )
    if values.size == 0:
        raise ValueError("fitness must not be empty")
    values = values.astype(np.float32)
    length = int(values.shape[0])
    if pop_size is None:
        pop_size = length
    elif int(pop_size) != length:
        raise ValueError(
            f"pop_size={pop_size} contradicts the fitness length {length}"
        )

    evoxc_module = _runtime.evoxc()
    key_tensor = _runtime.resolve_key(key)
    memo_key = (
        int(pop_size),
        mode,
        int(tournament_size),
        float(selection_pressure),
        backend,
        device,
        tuple(int(size) for size in key_tensor.shape),
        str(key_tensor.dtype),
    )
    built = _runtime.executable(
        _EXECUTABLE_MEMO,
        memo_key,
        lambda: _selection_fn(pop_size, mode, tournament_size, selection_pressure),
        (
            evoxc_module.TensorSpec((int(pop_size),), evoxc_module.float32),
            _runtime.key_spec(evoxc_module, key_tensor),
        ),
        backend,
        device,
    )
    (indices,) = evoxc_module.run(built, values, key_tensor)
    return indices
