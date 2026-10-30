"""The problem layer of the evoxc GP stack: data, predictions and fitness.

This is the counterpart of ``../problem/`` for the functional stack
(:mod:`evogp.gp`), and it follows the same two rules as the rest of the subtree:
**no torch** and **numpy host data**.  A problem owns the task data and exposes
the one plug-in surface ``evaluate(channels) -> [pop_size] float32`` that the
future driver loop calls once per generation (``DESIGN.md`` section
"Evaluation and the driver").

Fitness contract, identical to the repo-wide one:

* ``evaluate`` returns a **maximized** ``[pop_size]`` float32 result -- a loss is
  negated, so the lowest-error tree has the highest fitness and ``argmax`` picks
  it;
* the returned array is **fresh and writable** (the pipeline mutates it in place:
  ``NaN -> -inf``), and it never contains a NaN anyway: a non-finite loss is
  mapped to ``-inf`` here;
* the divisor is the number of DATA ROWS only, never the number of outputs
  (``../tree/ENGINE_SPEC.md`` section 5).

Two problems are provided:

* :class:`SymbolicRegression` -- ``X [n_rows, input_len]`` + ``y [n_rows]``,
  fitness ``-mean((y - pred)^2)`` (or ``-mean(|y - pred|)`` with
  ``use_mse=False``), the same positive loss ``Forest.SR_fitness`` computes.
* :class:`CustomLoss` -- a user-supplied ``loss(predictions, targets)`` callable
  over the whole population at once, negated the same way.  This is the escape
  hatch for tasks whose fitness is not a mean squared error (multi-output /
  combined trees, classification-style accuracy, penalties, ...).

Both evaluate through :func:`evogp.gp.evaluator.predict`, the numpy tree
evaluator, i.e. the host-side path.  ``problem.py`` additionally carries the
**in-graph** variant of the SR fitness (:func:`sr_fitness_graph` /
:func:`predict_graph`): one traced ``evoxc`` graph that evaluates a whole
population and reduces the loss inside the graph, built once per shape and
memoized.  See the "in-graph variant" section at the bottom of this file for its
contract and for the one place where its numerics can differ from the host
evaluator.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ._constants import FUNC_ARITY, FUNC_TYPE, Func, NType
from .evaluator import predict as _host_predict
from .state import GPState, population_size

__all__ = [
    "BaseProblem",
    "SymbolicRegression",
    "CustomLoss",
    "sr_fitness_graph",
    "predict_graph",
    "graph_cache_stats",
    "clear_graph_cache",
    "GRAPH_CACHE_SIZE_ENV",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_data(x, name: str, ndim: int) -> np.ndarray:
    """Validated float32 task data (C-contiguous, ``ndim``-dimensional)."""
    array = np.asarray(x, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _channel_pop_size(channels) -> int:
    """``pop_size`` from a state or an ``(opcode, payload, size)`` triple."""
    if isinstance(channels, GPState):
        return population_size(channels)
    try:
        opcode = channels[0]
    except (TypeError, IndexError) as exc:
        raise TypeError(
            "channels must be a GPState or an (opcode, payload, size) triple"
        ) from exc
    return int(opcode.shape[0])


def _fitness_from_loss(loss: np.ndarray, pop_size: int) -> np.ndarray:
    """``-loss`` as a fresh ``[pop_size]`` float32 array, non-finite -> ``-inf``.

    The sign flip is the repo-wide "maximize" convention; the ``-inf`` mapping is
    what the pipeline would do to a NaN anyway, done here so every problem
    returns the same clean contract (and so ``argmax`` can never pick a NaN).
    """
    values = np.array(loss, dtype=np.float32, copy=True).reshape(-1)
    if values.shape[0] != pop_size:
        raise ValueError(
            f"the loss has {values.shape[0]} entries but the population has "
            f"{pop_size} individuals"
        )
    fitness = -values
    fitness[~np.isfinite(fitness)] = -np.inf
    return fitness


# --------------------------------------------------------------------------- #
# the problem surface
# --------------------------------------------------------------------------- #
class BaseProblem:
    """The duck-typed contract of a problem (mirrors ``../problem/base.py``).

    A problem is anything with ``evaluate(channels) -> [pop_size] float32``
    (maximized); :attr:`problem_dim` / :attr:`solution_dim` describe the data
    for wiring checks (``input_len`` / ``output_len`` of the descriptor).
    """

    def evaluate(self, channels) -> np.ndarray:
        raise NotImplementedError

    def predict(self, channels) -> np.ndarray:
        raise NotImplementedError

    @property
    def problem_dim(self) -> int:
        raise NotImplementedError

    @property
    def solution_dim(self) -> int:
        raise NotImplementedError


class SymbolicRegression(BaseProblem):
    """Symbolic regression on a fixed numpy data set.

    Args:
        x: ``[n_rows, input_len]`` float32 inputs (an ``[n_rows]`` 1-column
            matrix must be passed 2-D).
        y: ``[n_rows]`` float32 targets.
        use_mse: ``True`` (default) for the squared error, ``False`` for the
            absolute error -- the ``use_MSE`` flag of ``Forest.SR_fitness``.

    The stored data is a float32 copy: mutating the caller's arrays afterwards
    does not change the problem, and the problem never mutates its own data.
    """

    def __init__(self, x, y, use_mse: bool = True, *, validate: bool = True):
        self.x = _as_data(x, "x", 2)
        self.y = _as_data(y, "y", 1)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"x has {self.x.shape[0]} rows but y has {self.y.shape[0]} entries"
            )
        self.use_mse = bool(use_mse)
        self._validate = bool(validate)

    # ------------------------------------------------------------- interface
    def predict(self, channels) -> np.ndarray:
        """``[pop_size, n_rows]`` float32 predictions of a population."""
        return _host_predict(
            channels, self.x, input_len=self.x.shape[1], validate=self._validate
        )

    def loss(self, channels) -> np.ndarray:
        """``[pop_size]`` float32 POSITIVE loss (``mean`` over data rows)."""
        predictions = self.predict(channels)
        # A non-finite prediction makes the loss non-finite too (the semantics of
        # record never sanitize: NaN/inf propagate).  Suffix overflow (a huge
        # error squared) is the same story, so no warning is raised for it --
        # `_fitness_from_loss` turns the result into -inf.
        with np.errstate(all="ignore"):
            diff = self.y[None, :] - predictions
            err = diff * diff if self.use_mse else np.abs(diff)
            return err.sum(axis=1, dtype=np.float32) / np.float32(self.x.shape[0])

    def evaluate(self, channels) -> np.ndarray:
        """``[pop_size]`` float32 MAXIMIZED fitness (the negated loss)."""
        return _fitness_from_loss(self.loss(channels), _channel_pop_size(channels))

    def __call__(self, channels) -> np.ndarray:
        """``evaluate`` -- so a problem is directly usable as ``evaluate=``."""
        return self.evaluate(channels)

    # ------------------------------------------------------------ properties
    @property
    def problem_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def solution_dim(self) -> int:
        return 1


class CustomLoss(BaseProblem):
    """A problem defined by a user-supplied numpy loss over the population.

    Args:
        x: ``[n_rows, input_len]`` float32 inputs.
        loss: ``loss(predictions, targets) -> per-individual loss``, called with
            ``predictions`` of shape ``[pop_size, n_rows]`` (and ``targets`` when
            it was given) and returning one positive loss value per individual
            (any shape with ``pop_size`` entries, e.g. ``[pop_size]`` or
            ``[pop_size, 1]``).  Negate nothing here: the problem negates it.
        targets: optional targets handed to ``loss`` as the second argument.  A
            one-argument callable is supported too (``targets=None``).

    The callable receives numpy arrays and runs once per ``evaluate`` over the
    WHOLE population, which is what makes this the multi-output escape hatch:
    the loss can combine several predicted columns (a ``CombinedForest`` in the
    torch stack) without the per-tree loop the engine would otherwise impose.
    """

    def __init__(
        self,
        x,
        loss: Callable,
        targets=None,
        *,
        validate: bool = True,
    ):
        if not callable(loss):
            raise TypeError(f"loss must be callable, got {type(loss).__name__}")
        self.x = _as_data(x, "x", 2)
        self.loss_func = loss
        if targets is not None:
            array = np.asarray(targets, dtype=np.float32)
            if array.ndim not in (1, 2):
                raise ValueError(
                    f"targets must be 1-D or 2-D, got shape {array.shape}"
                )
            if array.shape[0] != self.x.shape[0]:
                raise ValueError(
                    f"x has {self.x.shape[0]} rows but targets has "
                    f"{array.shape[0]}"
                )
            targets = np.ascontiguousarray(array)
        self.targets = targets
        self._validate = bool(validate)

    # ------------------------------------------------------------- interface
    def predict(self, channels) -> np.ndarray:
        """``[pop_size, n_rows]`` float32 predictions of a population."""
        return _host_predict(
            channels, self.x, input_len=self.x.shape[1], validate=self._validate
        )

    def loss(self, channels) -> np.ndarray:
        """``[pop_size]`` float32 POSITIVE loss, as returned by the callable."""
        predictions = self.predict(channels)
        pop_size = predictions.shape[0]
        raw = (
            self.loss_func(predictions)
            if self.targets is None
            else self.loss_func(predictions, self.targets)
        )
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        if values.shape[0] != pop_size:
            raise ValueError(
                f"the loss callable returned {values.shape[0]} values but the "
                f"population has {pop_size} individuals"
            )
        return values

    def evaluate(self, channels) -> np.ndarray:
        """``[pop_size]`` float32 MAXIMIZED fitness (the negated loss)."""
        return _fitness_from_loss(self.loss(channels), _channel_pop_size(channels))

    def __call__(self, channels) -> np.ndarray:
        """``evaluate`` -- so a problem is directly usable as ``evaluate=``."""
        return self.evaluate(channels)

    # ------------------------------------------------------------ properties
    @property
    def problem_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def solution_dim(self) -> int:
        if self.targets is None:
            return 1
        return int(self.targets.shape[1]) if self.targets.ndim > 1 else 1
