"""The problem layer of the evoxc GP stack: data, predictions and fitness.

This is the counterpart of the deleted torch stack's ``problem/`` for the
functional stack (:mod:`evogp.gp`), and it follows the same two rules as the
rest of the subtree:
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
  (the deleted ``ENGINE_SPEC.md`` section 5).

Two problems are provided:

* :class:`SymbolicRegression` -- ``X [n_rows, input_len]`` + ``y [n_rows]``,
  fitness ``-mean((y - pred)^2)`` (or ``-mean(|y - pred|)`` with
  ``use_mse=False``), the same positive loss the deleted ``Forest.SR_fitness``
  computed.
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
from .encoding import _channel_arrays, _evoxc as _require_evoxc
from .evaluator import _prepare_channels, predict as _host_predict
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
    """``pop_size`` from a state or an ``(opcode, payload, size)`` triple.

    A single individual (the three 1-D channel rows of a
    ``GPResult.best_channels``) counts as a population of one, exactly like the
    shared parser (:func:`evogp.gp.encoding._channel_arrays`) normalizes it.
    """
    if isinstance(channels, GPState):
        return population_size(channels)
    try:
        channels[0]
    except (TypeError, IndexError) as exc:
        raise TypeError(
            "channels must be a GPState or an (opcode, payload, size) triple"
        ) from exc
    return int(_channel_arrays(channels)[0].shape[0])


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
    """The duck-typed contract of a problem (mirrors the deleted
    ``BaseProblem``).

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
            absolute error -- the ``use_MSE`` flag of the deleted
            ``Forest.SR_fitness``.

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
        """``[pop_size, n_rows]`` float32 predictions of a population.

        One individual (``GPResult.best_channels``) is a population of one.
        """
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

    def predict_graph(self, channels, **options) -> np.ndarray:
        """``predict`` through one traced ``evoxc`` graph (:func:`predict_graph`)."""
        return predict_graph(
            channels,
            self.x,
            input_len=self.x.shape[1],
            validate=self._validate,
            **options,
        )

    def evaluate_graph(self, channels, **options) -> np.ndarray:
        """``evaluate`` through one traced ``evoxc`` graph (:func:`sr_fitness_graph`)."""
        return sr_fitness_graph(
            channels,
            self.x,
            self.y,
            use_mse=self.use_mse,
            input_len=self.x.shape[1],
            validate=self._validate,
            **options,
        )

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
    deleted torch stack) without the per-tree loop an engine would otherwise
    impose.
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
        """``[pop_size, n_rows]`` float32 predictions of a population.

        One individual (``GPResult.best_channels``) is a population of one.
        """
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


# --------------------------------------------------------------------------- #
# the in-graph variant: one traced evoxc graph per population shape
# --------------------------------------------------------------------------- #
#
# The host evaluator above walks the population in numpy.  This section does the
# same walk INSIDE an ``evoxc`` trace, so the "in-graph math" story of
# ``DESIGN.md`` is real: one graph per ``(pop_size, max_nodes, n_rows,
# input_len, used function ids)`` evaluates a whole population and -- for the
# fitness variant -- reduces the SR loss in-graph, so the host reads back a
# ``[pop_size]`` vector instead of ``[pop_size, n_rows]`` predictions.
#
# Two things change because a trace has no dynamic control flow:
#
# * the population's STRUCTURE (which cell is a leaf, which function id it
#   carries, where its children are) becomes runtime *inputs* -- the host
#   pre-pass :func:`_graph_inputs` computes them with the same vectorized numpy
#   arithmetic the host evaluator uses (``c1 = i+1``, ``c2 = c1 + size[c1]``,
#   ``c3 = c2 + size[c2]``, clipped into the row);
# * dispatch is a select chain over the population's USED function ids instead
#   of a per-column mask, because an id that is not emitted would need a
#   data-dependent branch.  The used-id set is part of the graph key, so a
#   population with a different function set recompiles once and then hits.
#
# The value buffer is grown column by column (``node = L-1`` down to ``0``) as a
# ``[rows, pop_size, n_rows]`` tensor.  Only the second (and, for a ternary
# function set, the third) child needs a gather -- ONE flat ``gather`` each, the
# construction of :func:`_read_buffer` below (the idiom the pre-2.0 dense engine
# established): pre-order makes a node's first child the next
# column, so ``v1`` is the previous column's value tensor itself and costs no
# read at all.
#
# Numerics: every emitter below is the op-for-op counterpart of the numpy
# emitter in :mod:`evogp.gp.evaluator` (and of the tree stack's
# ``evoxc_engine``, deleted in this release), so
# the graph and the host evaluator agree to the last bit on the numpy backend --
# pinned by ``tests/test_gp_stack_problem_graph.py``.  ``sinh``/``cosh`` are
# composed from ``exp`` exactly as the tree layer does.

#: Environment variable holding the size of this layer's graph cache.
GRAPH_CACHE_SIZE_ENV = "EVOGP_GP_GRAPH_CACHE_SIZE"
_DEFAULT_GRAPH_CACHE_SIZE = 32

#: ``(variant, L, pop, rows, input_len, used_ids, backend, device, use_mse)``
#: -> ``(graph_fn, executable)``.
_GRAPH_MEMO: dict = {}
_GRAPH_COMPILE_CACHE = None

_DELTA = np.float32(1e-9)
_MAX_VAL = np.float32(1e9)
_NEG_MAX_VAL = np.float32(-1e9)
_NAN = np.float32("nan")
_SINH_LINEAR_BELOW = np.float32(1e-4)

#: Node kinds of the traced dispatch (``NType`` base markers, as the torch
#: stack's dense route uses them).
_K_VAR = NType.VAR
_K_CONST = NType.CONST
_K_UFUNC = NType.UFUNC
_K_BFUNC = NType.BFUNC
_K_TFUNC = NType.TFUNC


def _graph_cache_size() -> int:
    import os

    raw = os.environ.get(GRAPH_CACHE_SIZE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_GRAPH_CACHE_SIZE
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{GRAPH_CACHE_SIZE_ENV} must be an integer, got {raw!r}"
        ) from exc
    if size <= 0:
        raise ValueError(f"{GRAPH_CACHE_SIZE_ENV} must be positive, got {size}")
    return size


def _graph_compile_cache():
    global _GRAPH_COMPILE_CACHE
    if _GRAPH_COMPILE_CACHE is None:
        _GRAPH_COMPILE_CACHE = _require_evoxc().CompileCache(maxsize=_graph_cache_size())
    return _GRAPH_COMPILE_CACHE


def graph_cache_stats() -> dict:
    """Counters of this layer (memoized graphs + the evoxc compile cache)."""
    cache = _GRAPH_COMPILE_CACHE
    return {
        "graphs": len(_GRAPH_MEMO),
        "compile_cache_size": 0 if cache is None else len(cache),
        "compile_cache_maxsize": _graph_cache_size(),
    }


def clear_graph_cache() -> None:
    """Drop every memoized graph and the compile cache of this layer."""
    global _GRAPH_COMPILE_CACHE
    _GRAPH_MEMO.clear()
    if _GRAPH_COMPILE_CACHE is not None:
        _GRAPH_COMPILE_CACHE.clear()
    _GRAPH_COMPILE_CACHE = None


# --------------------------------------------------------------------------- #
# trace-time constants and emitters (op-for-op with evaluator.py)
# --------------------------------------------------------------------------- #
def _c32(evoxc, value):
    """0-d float32 graph constant."""
    return evoxc.constant(evoxc.core.tensor(np.asarray(np.float32(value), np.float32)))


def _i32(evoxc, value):
    """0-d int32 graph constant."""
    return evoxc.constant(evoxc.core.tensor(np.asarray(np.int32(value), np.int32)))


def _c32_array(evoxc, array):
    """A full float32 array as a graph constant (only for small arrays)."""
    return evoxc.constant(evoxc.core.tensor(np.ascontiguousarray(array, np.float32)))


def _c64(evoxc, value):
    """0-d float64 graph constant."""
    return evoxc.constant(evoxc.core.tensor(np.asarray(np.float64(value), np.float64)))


def _operand(evoxc, value):
    """A symbolic value passes through; a folded constant becomes a 0-d const."""
    return value if isinstance(value, evoxc.SymbolicTensor) else _c32(evoxc, value)


def _t64(evoxc, value):
    """Widen to float64 (the intermediate of a correctly rounded op)."""
    return evoxc.cast(value, evoxc.float64)


def _t32(evoxc, value):
    """Round a float64 intermediate back to the float32 node storage."""
    return evoxc.cast(value, evoxc.float32)


def _copysign_delta(evoxc, value):
    """``copysign(DELTA, value)`` -- evoxc has no copysign.

    The sign is read as ``sign(1 / value)``, the only composition that gets
    ``-0.0`` right (the emitters of the deleted ``evoxc_engine.py`` used the same
    trick).  It is only reached for ``|value| <= DELTA``.
    """
    negative = evoxc.less(
        evoxc.divide(_c32(evoxc, 1.0), value), _c32(evoxc, 0.0)
    )
    return evoxc.select(negative, _c32(evoxc, -_DELTA), _c32(evoxc, _DELTA))


def _emit_unary(evoxc, function_id: int, a):
    """One unary node of the semantics of record (see :mod:`evogp.gp.evaluator`)."""
    zero = _c32(evoxc, 0.0)
    if function_id == Func.SIN:
        return _t32(evoxc, evoxc.sin(_t64(evoxc, a)))
    if function_id == Func.COS:
        return _t32(evoxc, evoxc.cos(_t64(evoxc, a)))
    if function_id == Func.TAN:
        return _t32(evoxc, evoxc.tan(_t64(evoxc, a)))
    if function_id == Func.SINH:  # composed: evoxc has no sinh
        w = _t64(evoxc, a)
        composed = _t32(
            evoxc,
            evoxc.multiply(
                evoxc.subtract(evoxc.exp(w), evoxc.exp(evoxc.negate(w))),
                _c64(evoxc, 0.5),
            ),
        )
        return evoxc.select(
            evoxc.less(evoxc.abs(a), _c32(evoxc, _SINH_LINEAR_BELOW)), a, composed
        )
    if function_id == Func.COSH:  # composed: evoxc has no cosh
        w = _t64(evoxc, a)
        return _t32(
            evoxc,
            evoxc.multiply(
                evoxc.add(evoxc.exp(w), evoxc.exp(evoxc.negate(w))),
                _c64(evoxc, 0.5),
            ),
        )
    if function_id == Func.TANH:
        return _t32(evoxc, evoxc.tanh(_t64(evoxc, a)))
    if function_id == Func.LOG:
        return _t32(evoxc, evoxc.log(_t64(evoxc, a)))
    if function_id == Func.LOOSE_LOG:
        return evoxc.select(
            evoxc.equal(a, zero),
            _c32(evoxc, _NEG_MAX_VAL),
            _t32(evoxc, evoxc.log(evoxc.abs(_t64(evoxc, a)))),
        )
    if function_id == Func.EXP:
        return _t32(evoxc, evoxc.exp(_t64(evoxc, a)))
    if function_id == Func.INV:
        return evoxc.select(
            evoxc.equal(a, zero),
            _c32(evoxc, _NAN),
            evoxc.divide(_c32(evoxc, 1.0), a),
        )
    if function_id == Func.LOOSE_INV:
        safe = evoxc.select(
            evoxc.less_equal(evoxc.abs(a), _c32(evoxc, _DELTA)),
            _copysign_delta(evoxc, a),
            a,
        )
        return evoxc.divide(_c32(evoxc, 1.0), safe)
    if function_id == Func.NEG:
        return evoxc.negate(a)
    if function_id == Func.ABS:
        return evoxc.abs(a)
    if function_id == Func.SQRT:
        return evoxc.sqrt(a)
    if function_id == Func.LOOSE_SQRT:
        return evoxc.sqrt(evoxc.abs(a))
    return zero  # an id that matches no branch leaves 0.0


def _emit_binary(evoxc, function_id: int, a, b):
    """One binary node of the semantics of record."""
    zero = _c32(evoxc, 0.0)
    one = _c32(evoxc, 1.0)
    minus_one = _c32(evoxc, -1.0)
    if function_id == Func.ADD:
        return evoxc.add(a, b)
    if function_id == Func.SUB:
        return evoxc.subtract(a, b)
    if function_id == Func.MUL:
        return evoxc.multiply(a, b)
    if function_id == Func.DIV:
        return evoxc.select(
            evoxc.equal(b, zero), _c32(evoxc, _NAN), evoxc.divide(a, b)
        )
    if function_id == Func.LOOSE_DIV:
        safe = evoxc.select(
            evoxc.less_equal(evoxc.abs(b), _c32(evoxc, _DELTA)),
            _copysign_delta(evoxc, b),
            b,
        )
        return evoxc.divide(a, safe)
    if function_id == Func.POW:
        return _t32(evoxc, evoxc.power(_t64(evoxc, a), _t64(evoxc, b)))
    if function_id == Func.LOOSE_POW:
        return evoxc.select(
            evoxc.logical_and(evoxc.equal(a, zero), evoxc.equal(b, zero)),
            zero,
            _t32(evoxc, evoxc.power(evoxc.abs(_t64(evoxc, a)), _t64(evoxc, b))),
        )
    if function_id == Func.MAX:
        return evoxc.select(evoxc.greater_equal(a, b), a, b)
    if function_id == Func.MIN:
        return evoxc.select(evoxc.less_equal(a, b), a, b)
    if function_id == Func.LT:
        return evoxc.select(evoxc.less(a, b), one, minus_one)
    if function_id == Func.GT:
        return evoxc.select(evoxc.greater(a, b), one, minus_one)
    if function_id == Func.LE:
        return evoxc.select(evoxc.less_equal(a, b), one, minus_one)
    if function_id == Func.GE:
        return evoxc.select(evoxc.greater_equal(a, b), one, minus_one)
    return zero


def _emit_ternary(evoxc, a, b, c):
    """One ternary node: the kernels always evaluate ``if`` for a TFUNC node."""
    return evoxc.select(evoxc.greater(a, _c32(evoxc, 0.0)), b, c)


# --------------------------------------------------------------------------- #
# the traced body
# --------------------------------------------------------------------------- #
def _column(tensor, index: int, pop_size: int):
    """Column ``index`` of an ``(L, pop)`` input as a ``(pop, 1)`` tensor."""
    import evoxc  # noqa: PLC0415 - the trace-time module (never at import time)

    return evoxc.reshape(evoxc.slice(tensor, (index, 0), (1, pop_size)), (pop_size, 1))


def _row_offsets(evoxc, pop: int, n_rows: int):
    """The ``(pop, n_rows)`` flat-offset grid ``p * n_rows + r`` of one block.

    Built ONCE per graph: it is a pure function of the static shape, while the
    walk below reads two child buffers per column.
    """
    return evoxc.add(
        evoxc.reshape(evoxc.cast(evoxc.arange(pop), evoxc.int32) * n_rows, (pop, 1)),
        evoxc.reshape(evoxc.cast(evoxc.arange(n_rows), evoxc.int32), (1, n_rows)),
    )


def _row_column(tensor, index: int, pop_size: int):
    """Column ``index`` of a ``(pop, L)`` tensor as a ``(pop, 1)`` tensor."""
    import evoxc  # noqa: PLC0415

    return evoxc.slice(tensor, (0, index), (pop_size, 1))


def _child_bases(evoxc, child, width: int, pop: int, n_rows: int, pn: int):
    """``(pop, width)`` gather BASES of one child channel.

    Entry ``[p, i]`` is ``clip(child[p, i] - (i+1), 0, rows_i - 1) * pn``, i.e.
    everything a child read at column ``i`` needs from the index arithmetic but
    the ``+ p * n_rows + r`` of :func:`_row_offsets`.  The clamp is a pure
    function of the static shape (``rows_i = width - 1 - i``) and the whole
    channel is available at trace time, so it is computed ONCE per graph instead
    of once per child per column -- same numbers, a third of the dispatches.
    """
    columns = evoxc.reshape(evoxc.cast(evoxc.arange(width), evoxc.int32), (1, width))
    upper = evoxc.subtract(_i32(evoxc, width - 2), columns)  # rows_i - 1
    offset = evoxc.subtract(
        evoxc.transpose(child, (1, 0)), evoxc.add(columns, _i32(evoxc, 1))
    )  # (pop, width)
    offset = evoxc.maximum(evoxc.minimum(offset, upper), _i32(evoxc, 0))
    return evoxc.multiply(offset, pn)


def _read_buffer(flat, base, index: int, pop: int, n_rows: int, row_offsets):
    """Child value of column ``index`` for every tree -> ``(pop, n_rows)``.

    ``flat`` is the ``rows * pop * n_rows`` flattening of the value buffer (row 0
    = the newest column), ``base`` the :func:`_child_bases` grid of one child
    channel and ``row_offsets`` the :func:`_row_offsets` grid.  ONE flat
    ``gather`` per child, exactly as the pre-2.0 dense engine's ``_read_buffer``
    did it.

    The index is flattened to ``(pn,)`` and the block reshaped back on purpose:
    IREE lowers a 1-D index measurably faster than the equivalent ``(pop,
    n_rows)`` one (measured at pop 256: 0.71 vs 0.89 ms at 1 024 rows, 1.56 vs
    2.46 ms at 4 096), so the two reshapes are cheaper than the index they save.
    """
    import evoxc  # noqa: PLC0415

    pn = pop * n_rows
    index = evoxc.add(_row_column(base, index, pop), row_offsets)  # (pop, n_rows)
    flat_index = evoxc.reshape(index, (pn,))
    return evoxc.reshape(evoxc.gather(flat, flat_index, axis=0), (pop, n_rows))


def _graph_body(kind, func_id, const_value, var_index, child1, child2, child3, x, cfg):
    """Trace-time body: node 0 of every tree as a ``(pop, n_rows)`` tensor.

    ``cfg`` is a static tuple ``(L, pop, n_rows, input_len, used_ids)``; it
    specializes the graph, so it is part of the build and run signature.

    Two static facts of the pre-order walk keep the reads cheap:

    * ``child1`` is no longer read: in prefix (pre-order) encoding the FIRST
      child of a node is always the next column, so ``v1`` is the previous
      column's value itself -- no read at all instead of a data-dependent
      gather.  The argument stays in the signature because it is part of the
      graph's input contract (:func:`_graph_inputs`,
      ``graph_eval._structure_inputs``);
    * a function set without :class:`~evogp.gp._constants.Func` ``IF`` cannot
      produce a ternary node, so the ``phi`` and its THIRD-child gather are dead
      code and are not emitted (``graph_eval._uncovered_rows`` masks a
      ``TFUNC`` row for exactly that case).
    """
    import evoxc  # noqa: PLC0415

    width = int(cfg[0])
    pop = int(cfg[1])
    n_rows = int(cfg[2])
    input_len = int(cfg[3])
    used_ids = tuple(int(v) for v in cfg[4])
    pn = pop * n_rows
    ternary = Func.IF in used_ids

    x_flat = evoxc.reshape(x, (n_rows * input_len,))
    x_base = evoxc.reshape(
        evoxc.cast(evoxc.arange(n_rows), evoxc.int32) * input_len, (1, n_rows)
    )
    row_offsets = _row_offsets(evoxc, pop, n_rows)
    base2 = _child_bases(evoxc, child2, width, pop, n_rows, pn)
    base3 = _child_bases(evoxc, child3, width, pop, n_rows, pn) if ternary else None
    zero = _c32(evoxc, 0.0)
    values = [None] * width  # column index -> its (pop, n_rows) value
    buffer = None
    rows = 0
    root = None

    for i in range(width - 1, -1, -1):
        op_i = _column(kind, i, pop)
        f_i = _column(func_id, i, pop)
        cv_i = _column(const_value, i, pop)
        vi_i = _column(var_index, i, pop)

        if buffer is None:
            # A function node at the last column would need a child outside the
            # row, which the host decode validation rejects; the gathers are dead
            # code there and are replaced by a constant.
            v1 = v2 = v3 = _c32(evoxc, 0.0)
        else:
            v1 = values[i + 1]  # pre-order: the first child of node i is node i+1
            flat = evoxc.reshape(buffer, (rows * pn,))
            v2 = _read_buffer(flat, base2, i, pop, n_rows, row_offsets)
            v3 = (
                _read_buffer(flat, base3, i, pop, n_rows, row_offsets)
                if ternary
                else v1
            )

        # -- leaf value -------------------------------------------------------
        x_index = evoxc.add(vi_i, x_base)  # (pop, n_rows)
        xv = evoxc.gather(x_flat, x_index, axis=0)
        leaf = evoxc.select(evoxc.equal(op_i, _K_VAR), xv, cv_i)

        if buffer is None:
            value = leaf
        else:
            # -- function value: the select chain over the used ids -----------
            product = zero
            for fid in reversed(used_ids):
                if fid == Func.IF:  # ternary: dispatched below, never by id
                    continue
                candidate = (
                    _emit_binary(evoxc, fid, v1, v2)
                    if fid > Func.IF
                    and FUNC_ARITY[fid] == 2
                    else _emit_unary(evoxc, fid, v1)
                )
                product = evoxc.select(evoxc.equal(f_i, fid), candidate, product)
            if ternary:
                product = evoxc.select(
                    evoxc.equal(op_i, _K_TFUNC),
                    _emit_ternary(evoxc, v1, v2, v3),
                    product,
                )
            value = evoxc.select(evoxc.less_equal(op_i, _K_CONST), leaf, product)

        if i == 0:
            root = value
            break
        values[i] = value
        row = evoxc.reshape(value, (1, pop, n_rows))
        buffer = row if buffer is None else evoxc.concatenate([row, buffer], axis=0)
        rows += 1
    return root


def _graph_predict(kind, func_id, const_value, var_index, child1, child2, child3, x, cfg):
    """Traced entry point of :func:`predict_graph` (module-level: stable identity)."""
    return _graph_body(
        kind, func_id, const_value, var_index, child1, child2, child3, x, cfg
    )


def _graph_sr_fitness(
    kind, func_id, const_value, var_index, child1, child2, child3, x, y, cfg
):
    """Traced entry point of :func:`sr_fitness_graph` (module-level identity)."""
    import evoxc  # noqa: PLC0415

    n_rows = int(cfg[2])
    use_mse = bool(cfg[5])
    root = _graph_body(
        kind, func_id, const_value, var_index, child1, child2, child3, x, cfg[:5]
    )
    diff = evoxc.subtract(evoxc.reshape(y, (1, n_rows)), root)
    err = evoxc.multiply(diff, diff) if use_mse else evoxc.abs(diff)
    total = evoxc.sum(err, axes=1)
    loss = evoxc.divide(total, _c32(evoxc, np.float32(n_rows)))
    return evoxc.negate(loss)  # the MAXIMIZED fitness the repo-wide contract wants


# --------------------------------------------------------------------------- #
# host pre-pass: the population's structure as runtime inputs
# --------------------------------------------------------------------------- #
def _graph_inputs(channels, x) -> Tuple[tuple, tuple]:
    """``((kind, func_id, const, var_index, c1, c2, c3), used_ids)`` of a population.

    Every array is ``[max_nodes, pop_size]`` (columns as rows, the layout the
    traced body slices), computed with the same vectorized arithmetic as
    :func:`evogp.gp.evaluator._evaluate_block`.
    """
    opcode, payload, size = _prepare_channels(channels)
    pop_size, width = opcode.shape
    input_len = x.shape[1]

    function = size > 1
    ids = np.clip(opcode, 0, Func.END - 1)
    kind = np.where(
        function, np.asarray(FUNC_TYPE, np.int32)[ids], opcode
    ).astype(np.int32)
    func_id = np.where(function, opcode, 0).astype(np.int32)
    const_value = np.where(
        function,
        np.float32(0.0),
        np.where(opcode == NType.CONST, payload, np.float32(0.0)),
    ).astype(np.float32)
    var_index = np.where(
        function | (opcode == NType.CONST),
        0,
        np.clip(np.trunc(payload).astype(np.int64), 0, input_len - 1),
    ).astype(np.int32)

    # c1 = i+1, c2 = c1 + size[c1], c3 = c2 + size[c2] -- clipped into the row
    # exactly like the host evaluator's addressing.
    columns = np.arange(width)
    c1 = np.minimum(columns + 1, width - 1)
    s1 = size[:, c1]
    c2 = np.minimum(columns[None, :] + 1 + s1, width - 1)
    s2 = np.take_along_axis(size, c2, axis=1)
    c3 = np.minimum(c2 + s2, width - 1)
    c1 = np.broadcast_to(c1, size.shape)

    used = tuple(sorted({int(v) for v in func_id[function]}))
    arrays = (
        kind,
        func_id,
        const_value,
        var_index,
        np.asarray(c1, np.int32),
        np.asarray(c2, np.int32),
        np.asarray(c3, np.int32),
    )
    return (
        tuple(np.ascontiguousarray(array.T) for array in arrays),  # (pop, L)->(L, pop)
        used,
    )


def _build_graph(variant: str, width, pop_size, n_rows, input_len, used_ids, use_mse,
                 backend, device):
    """The memoized executable of one ``(shape, used ids, backend, device)``."""
    evoxc = _require_evoxc()
    key = (variant, width, pop_size, n_rows, input_len, tuple(used_ids), bool(use_mse),
           backend, device)
    entry = _GRAPH_MEMO.get(key)
    if entry is not None:
        return entry[1]
    column = evoxc.TensorSpec((width, pop_size), evoxc.int32)
    floats = evoxc.TensorSpec((width, pop_size), evoxc.float32)
    rows = evoxc.TensorSpec((n_rows, input_len), evoxc.float32)
    targets = evoxc.TensorSpec((n_rows,), evoxc.float32)
    cfg = (width, pop_size, n_rows, input_len, tuple(used_ids), bool(use_mse))
    if variant == "predict":
        graph = _graph_predict
        specs = (column, column, floats, column, column, column, column, rows, cfg[:5])
    elif variant == "fitness":
        graph = _graph_sr_fitness
        specs = (column, column, floats, column, column, column, column, rows, targets, cfg)
    else:  # pragma: no cover - guarded by the callers
        raise ValueError(f"unknown graph variant {variant!r}")
    executable = evoxc.build_cached(
        graph, *specs, backend=backend, device=device, cache=_graph_compile_cache()
    )
    _GRAPH_MEMO[key] = (graph, executable)
    return executable


def predict_graph(
    channels,
    x,
    *,
    input_len: Optional[int] = None,
    validate: bool = True,
    backend: str = "numpy",
    device: str = "cpu",
) -> np.ndarray:
    """``[pop_size, n_rows]`` predictions from one traced ``evoxc`` graph.

    The in-graph counterpart of :func:`evogp.gp.evaluator.predict`: same
    semantics, same numbers (bit-identical on the numpy backend), but the whole
    population is evaluated by ONE compiled graph whose structure inputs are the
    population's channels.  The graph is memoized per
    ``(pop_size, max_nodes, n_rows, input_len, used function ids, backend)``.

    Args:
        channels: a :class:`~evogp.gp.state.GPState` or the ``(opcode, payload,
            size)`` triple (a single individual's three 1-D channel rows are a
            population of one).
        x: the data matrix ``[n_rows, input_len]``.
        input_len: when given it must match ``x.shape[1]`` (the ``VAR`` bound the
            validation uses).
        validate: run :func:`~evogp.gp.encoding.validate_channels` first.
        backend / device: evoxc backend and device (``"numpy"``/``"cpu"`` are the
            reference pair; ``"iree"`` compiles a device graph).

    Returns:
        ``[pop_size, n_rows]`` float32.
    """
    opcode, payload, size = _prepare_channels(channels)
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {data.shape}")
    if input_len is not None and int(input_len) != data.shape[1]:
        raise ValueError(
            f"input_len={input_len} contradicts X's {data.shape[1]} columns"
        )
    if validate:
        from .encoding import validate_channels

        validate_channels((opcode, payload, size), input_len=data.shape[1])

    arrays, used_ids = _graph_inputs((opcode, payload, size), data)
    pop_size, width = opcode.shape
    executable = _build_graph(
        "predict", width, pop_size, data.shape[0], data.shape[1], used_ids, False,
        backend, device,
    )
    evoxc = _require_evoxc()
    cfg = (width, pop_size, data.shape[0], data.shape[1], used_ids)
    with np.errstate(all="ignore"):  # the numpy backend's kernels warn like ours
        out = evoxc.run(executable, *arrays, data, cfg)
    return np.asarray(out.numpy(), dtype=np.float32)


def sr_fitness_graph(
    channels,
    x,
    y,
    *,
    use_mse: bool = True,
    input_len: Optional[int] = None,
    validate: bool = True,
    backend: str = "numpy",
    device: str = "cpu",
) -> np.ndarray:
    """``[pop_size]`` float32 MAXIMIZED SR fitness from one traced graph.

    Same contract as :meth:`SymbolicRegression.evaluate` (negated loss over the
    data rows, non-finite -> ``-inf``, fresh and writable), but the prediction
    AND the loss reduction run inside one compiled ``evoxc`` graph, so the host
    reads back a ``[pop_size]`` vector.  The loss formula is the semantics of
    record: ``sum_rows (y - pred)^2 / n_rows`` (or ``|y - pred|`` with
    ``use_mse=False``).

    The non-finite mapping is applied on the host after the readback, exactly
    like the numpy problem does it.
    """
    opcode, payload, size = _prepare_channels(channels)
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if data.ndim == 1:
        data = data[None, :]
    targets = np.ascontiguousarray(np.asarray(y, dtype=np.float32)).reshape(-1)
    if data.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {data.shape}")
    if targets.shape[0] != data.shape[0]:
        raise ValueError(
            f"y has {targets.shape[0]} entries but X has {data.shape[0]} rows"
        )
    if input_len is not None and int(input_len) != data.shape[1]:
        raise ValueError(
            f"input_len={input_len} contradicts X's {data.shape[1]} columns"
        )
    if validate:
        from .encoding import validate_channels

        validate_channels((opcode, payload, size), input_len=data.shape[1])

    arrays, used_ids = _graph_inputs((opcode, payload, size), data)
    pop_size, width = opcode.shape
    executable = _build_graph(
        "fitness", width, pop_size, data.shape[0], data.shape[1], used_ids,
        bool(use_mse), backend, device,
    )
    evoxc = _require_evoxc()
    cfg = (width, pop_size, data.shape[0], data.shape[1], used_ids, bool(use_mse))
    with np.errstate(all="ignore"):  # the numpy backend's kernels warn like ours
        out = evoxc.run(executable, *arrays, data, targets, cfg)
    fitness = np.array(out.numpy(), dtype=np.float32, copy=True).reshape(-1)
    if fitness.shape[0] != pop_size:
        raise RuntimeError(
            f"the graph returned {fitness.shape[0]} fitness values for a "
            f"population of {pop_size}"
        )
    fitness[~np.isfinite(fitness)] = -np.inf
    return fitness
