"""The numpy tree evaluator of the evoxc GP stack (torch-free, CPU).

The plug-in evaluation surface of the functional stack is one function::

    predict(channels, X) -> [pop_size, n_rows] float32

``channels`` is a :class:`~evogp.gp.state.GPState` or the ``(opcode, payload,
size)`` triple of the encoding (:mod:`evogp.gp.encoding`); ``X`` is the data
matrix ``[n_rows, input_len]``.  Row ``p`` of the result is the value of tree
``p`` on every data row.

**Why a column walk works.**  The encoding is prefix (pre-order), so node ``i``
*always* lives at column ``i`` and its children are at ``i+1``,
``i+1+size[i+1]`` and ``i+1+size[i+1]+size[i+1+size[i+1]]`` (``DESIGN.md``).
Walking the columns from ``max_nodes-1`` down to ``0`` therefore visits children
before parents, and one vectorized pass per column over the whole population and
all data rows computes every node value in place.  There is no Python loop over
individuals and none over data points -- only over columns (``max_nodes``, the
same bound as the tree layer's row width) and over the distinct function ids
present in a column.

Semantics are the tree layer's semantics of record
(``../tree/ENGINE_SPEC.md`` section 3 / section 4.1, mirrored by the op emitters
of ``../tree/evoxc_engine.py``):

* the FIRST (leftmost) child is operand 1, so ``-`` is ``first - second``,
  ``/`` is ``first / second`` and ``if`` tests the first child;
* ``DIV`` is protected (``b == 0`` -> NaN), ``LOOSE_DIV`` / ``LOOSE_INV`` clamp a
  divisor with ``|b| <= DELTA`` to ``copysign(DELTA, b)``, ``LOOSE_LOG`` answers
  ``-MAX_VAL`` at exactly zero, ``LOOSE_SQRT`` takes ``sqrt(|x|)``, ``LOOSE_POW``
  answers ``0`` for ``pow(0, 0)`` and ``pow(|a|, b)`` otherwise;
* ``IF`` is ``first child > 0 ? second : third`` -- both branches are computed by
  numpy and only *selected*, so the untaken branch can never leak a value; every
  warning (divide-by-zero / invalid / overflow) is suppressed inside one
  ``np.errstate`` block, which is the numpy counterpart of the kernels' lack of
  exception handling (NaN / +-inf are *not* sanitized: they propagate, exactly as
  ``ENGINE_SPEC`` section 3 records);
* comparisons answer exactly ``+-1`` and a false comparison (including any NaN
  involvement) answers ``-1``;
* ``MAX``/``MIN`` are ternaries, not ``fmax``/``fmin``: ``max(v, NaN)`` is NaN
  while ``max(NaN, v)`` is ``v``;
* every node value is *stored as float32* (``ENGINE_SPEC`` section 2).  The
  software transcendentals (``sin``/``cos``/``tan``/``tanh``/``log``/``exp``/
  ``pow`` and the composed ``sinh``/``cosh``) are evaluated in float64 and
  rounded back to float32, i.e. **correctly rounded** in float32; numpy's own
  float32 kernels are not (``../tree/evoxc_engine.py`` measures ``exp`` one ulp
  low on ~39% of inputs, ``tanh`` ~33%), and a one-ulp node value is amplified by
  the next node's derivative.  ``+ - * /``, ``sqrt``, ``1/x``, ``abs``, ``neg``
  and every comparison/select stay float32 -- IEEE 754 makes them correctly
  rounded (or exact) there already.

Constants, arithmetic and variables come from the three channels: a leaf with
``size == 1`` and ``opcode == NType.CONST`` is the ``payload`` value, a leaf with
``opcode == NType.VAR`` reads ``X[:, int(payload)]`` (C truncation toward zero,
like ``(int)node_value`` in the kernels); a cell with ``size > 1`` is a function
node whose ``opcode`` is its ``Func`` id and whose arity is
:data:`~evogp.gp._constants.FUNC_ARITY` of that id.

**Memory.**  The value buffer is ``pop_size * max_nodes * n_rows * 4`` bytes (one
float32 per node per data row), and the per-op temporaries are one or two
``pop_size * n_rows * 4``-byte copies at a time.  When the buffer would exceed
the budget (``EVOGP_GP_EVAL_BUDGET_MB``, default 256 MiB, or an explicit
``budget_mb`` / ``row_chunk``) the data rows are split into contiguous chunks and
the evaluation is repeated per chunk, so the peak stays inside the budget at the
price of re-walking the columns once per chunk.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from ._constants import FUNC_ARITY, Func, NType
from .encoding import _channel_arrays, validate_channels

__all__ = ["predict", "ROW_BUDGET_ENV"]

#: Environment variable holding the value-buffer budget of one ``predict`` call.
ROW_BUDGET_ENV = "EVOGP_GP_EVAL_BUDGET_MB"
_DEFAULT_BUDGET_MB = 256

# --------------------------------------------------------------------------- #
# constants of the semantics of record (ENGINE_SPEC section 3)
# --------------------------------------------------------------------------- #
_DELTA = np.float32(1e-9)
_MAX_VAL = np.float32(1e9)
_NEG_MAX_VAL = np.float32(-1e9)
_ZERO = np.float32(0.0)
_ONE = np.float32(1.0)
_NEG_ONE = np.float32(-1.0)
_NAN = np.float32("nan")
#: ``sinh`` is composed from ``exp`` (evoxc has no ``sinh``, and neither has the
#: tree layer's emitter); below this magnitude the correctly rounded float32
#: result of ``sinh(x)`` is bit-exactly ``x``, signed zero included.
_SINH_LINEAR_BELOW = np.float32(1e-4)
_HALF = np.float64(0.5)

#: ``FUNC_ARITY`` as an ``np.int8`` lookup table (indexed by a function id).
_ARITY = np.asarray(FUNC_ARITY, dtype=np.int8)


def _round32(value: np.ndarray) -> np.ndarray:
    """Round a float64 intermediate back to the float32 node-value storage.

    ``astype`` is round-to-nearest-even and preserves NaN / ``+-inf``, so the
    NaN/inf placement of the semantics is untouched by the widening.
    """
    return value.astype(np.float32)


def _f64(value: np.ndarray) -> np.ndarray:
    """Widen a float32 operand to the float64 intermediate of a CR op."""
    return value.astype(np.float64)


# --------------------------------------------------------------------------- #
# the op table (one vectorized numpy function per Func id)
# --------------------------------------------------------------------------- #
def _u_sin(a):
    return _round32(np.sin(_f64(a)))


def _u_cos(a):
    return _round32(np.cos(_f64(a)))


def _u_tan(a):
    return _round32(np.tan(_f64(a)))


def _u_sinh(a):
    # Composed exactly like ../tree/evoxc_engine.py::_emit_unary: evoxc has no
    # sinh, and the composition cannot keep the sign of zero nor the accuracy of
    # tiny |x|, so the linear branch is selected back in.
    w = _f64(a)
    composed = _round32((np.exp(w) - np.exp(-w)) * _HALF)
    return np.where(np.abs(a) < _SINH_LINEAR_BELOW, a, composed)


def _u_cosh(a):
    w = _f64(a)
    return _round32((np.exp(w) + np.exp(-w)) * _HALF)


def _u_tanh(a):
    return _round32(np.tanh(_f64(a)))


def _u_log(a):
    # log(0) = -inf, log(x < 0) = NaN -- neither is special-cased.
    return _round32(np.log(_f64(a)))


def _u_loose_log(a):
    return np.where(a == _ZERO, _NEG_MAX_VAL, _round32(np.log(np.abs(_f64(a)))))


def _u_exp(a):
    return _round32(np.exp(_f64(a)))


def _u_inv(a):
    return np.where(a == _ZERO, _NAN, _ONE / a)


def _u_loose_inv(a):
    safe = np.where(np.abs(a) <= _DELTA, np.copysign(_DELTA, a), a)
    return _ONE / safe


def _u_neg(a):
    return -a


def _u_abs(a):
    return np.abs(a)


def _u_sqrt(a):
    return np.sqrt(a)


def _u_loose_sqrt(a):
    return np.sqrt(np.abs(a))


def _b_add(a, b):
    return a + b


def _b_sub(a, b):
    return a - b


def _b_mul(a, b):
    return a * b


def _b_div(a, b):
    return np.where(b == _ZERO, _NAN, a / b)


def _b_loose_div(a, b):
    safe = np.where(np.abs(b) <= _DELTA, np.copysign(_DELTA, b), b)
    return a / safe


def _b_pow(a, b):
    # pow(0,0)=1, pow(-2,3)=-8, pow(-2,0.5)=NaN (float64 intermediate, rounded)
    return _round32(np.power(_f64(a), _f64(b)))


def _b_loose_pow(a, b):
    return np.where(
        (a == _ZERO) & (b == _ZERO), _ZERO, _round32(np.power(np.abs(_f64(a)), _f64(b)))
    )


def _b_max(a, b):
    return np.where(a >= b, a, b)


def _b_min(a, b):
    return np.where(a <= b, a, b)


def _b_lt(a, b):
    return np.where(a < b, _ONE, _NEG_ONE)


def _b_gt(a, b):
    return np.where(a > b, _ONE, _NEG_ONE)


def _b_le(a, b):
    return np.where(a <= b, _ONE, _NEG_ONE)


def _b_ge(a, b):
    return np.where(a >= b, _ONE, _NEG_ONE)


#: Unary emitters, keyed by ``Func`` id (ids 14..28).
_UNARY = {
    Func.SIN: _u_sin,
    Func.COS: _u_cos,
    Func.TAN: _u_tan,
    Func.SINH: _u_sinh,
    Func.COSH: _u_cosh,
    Func.TANH: _u_tanh,
    Func.LOG: _u_log,
    Func.LOOSE_LOG: _u_loose_log,
    Func.EXP: _u_exp,
    Func.INV: _u_inv,
    Func.LOOSE_INV: _u_loose_inv,
    Func.NEG: _u_neg,
    Func.ABS: _u_abs,
    Func.SQRT: _u_sqrt,
    Func.LOOSE_SQRT: _u_loose_sqrt,
}

#: Binary emitters, keyed by ``Func`` id (ids 1..13).
_BINARY = {
    Func.ADD: _b_add,
    Func.SUB: _b_sub,
    Func.MUL: _b_mul,
    Func.DIV: _b_div,
    Func.LOOSE_DIV: _b_loose_div,
    Func.POW: _b_pow,
    Func.LOOSE_POW: _b_loose_pow,
    Func.MAX: _b_max,
    Func.MIN: _b_min,
    Func.LT: _b_lt,
    Func.GT: _b_gt,
    Func.LE: _b_le,
    Func.GE: _b_ge,
}


# --------------------------------------------------------------------------- #
# channel / data normalisation
# --------------------------------------------------------------------------- #
def _prepare_channels(channels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(opcode int32, payload float32, size int64)`` 2-D views of the encoding.

    Accepts the same forms as :func:`evogp.gp.encoding._channel_arrays`, the one
    shared parser: a state, a population triple, or the three 1-D channel rows
    of one individual, which is normalized to a one-row population.
    """
    opcode, payload, size = _channel_arrays(channels)
    opcode = np.ascontiguousarray(np.asarray(opcode).astype(np.int32, copy=False))
    payload = np.ascontiguousarray(np.asarray(payload).astype(np.float32, copy=False))
    size = np.ascontiguousarray(np.asarray(size).astype(np.int64, copy=False))
    if opcode.ndim != 2:
        raise ValueError(
            "channels must be 2-D [pop_size, max_nodes] (or the three 1-D "
            f"channel rows of one individual), got shape {opcode.shape}"
        )
    if opcode.shape != payload.shape or opcode.shape != size.shape:
        raise ValueError(
            "the three channels must share one shape, got opcode "
            f"{opcode.shape}, payload {payload.shape}, size {size.shape}"
        )
    if opcode.shape[0] == 0 or opcode.shape[1] == 0:
        raise ValueError(f"channels must be non-empty, got shape {opcode.shape}")
    if np.any((opcode & NType.OUT_NODE) != 0):
        raise NotImplementedError(
            "evogp.gp.evaluator: multi-output (OUT_NODE-tagged) trees are not "
            "implemented in this stack yet; encode the population with "
            "output_len=1"
        )
    return opcode, payload, size


def _prepare_data(x) -> np.ndarray:
    """``X`` as a float32 ``[n_rows, input_len]`` array (a 1-D row is accepted)."""
    data = np.asarray(x, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {data.shape}")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(f"X must be non-empty, got shape {data.shape}")
    return data


def _resolve_budget_mb(budget_mb: Optional[float]) -> float:
    """The value-buffer budget in MiB (``EVOGP_GP_EVAL_BUDGET_MB``, default 256)."""
    if budget_mb is None:
        raw = os.environ.get(ROW_BUDGET_ENV)
        if raw is None or raw == "":
            return float(_DEFAULT_BUDGET_MB)
        try:
            budget_mb = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{ROW_BUDGET_ENV} must be a number of MiB, got {raw!r}"
            ) from exc
    budget_mb = float(budget_mb)
    if not np.isfinite(budget_mb) or budget_mb <= 0:
        raise ValueError(f"budget_mb must be positive, got {budget_mb!r}")
    return budget_mb


def _row_chunk(
    pop_size: int, width: int, n_rows: int, row_chunk: Optional[int], budget_mb
) -> int:
    """Rows per evaluation block (``n_rows`` = one block, the common case).

    The buffer is ``pop_size * width * rows * 4`` bytes, so ``rows`` is capped at
    ``budget / (pop_size * width * 4)`` rounded down and at least 1.
    """
    if row_chunk is not None:
        rows = int(row_chunk)
        if rows <= 0:
            raise ValueError(f"row_chunk must be positive, got {row_chunk!r}")
        return min(rows, n_rows)
    per_row = pop_size * width * 4
    budget = int(_resolve_budget_mb(budget_mb) * (1 << 20))
    return max(1, min(n_rows, budget // max(per_row, 1)))


# --------------------------------------------------------------------------- #
# the evaluation itself
# --------------------------------------------------------------------------- #
def _evaluate_block(
    opcode: np.ndarray, payload: np.ndarray, size: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """``[pop_size, n_rows]`` node 0 values of one contiguous block of data rows."""
    pop_size, width = opcode.shape
    n_rows = x.shape[0]
    values = np.zeros((pop_size, width, n_rows), dtype=np.float32)

    # Child addressing, per row: c1 = i+1, c2 = c1 + size[c1], c3 = c2 + size[c2].
    # The clamped index columns are shared by every data-row block; a malformed
    # population (only reachable with validate=False) reads a neighbour instead
    # of running off the row.
    columns = np.arange(width)
    c1 = np.minimum(columns + 1, width - 1)
    s1 = size[:, c1]
    c2 = np.minimum(columns[None, :] + 1 + s1, width - 1)
    s2 = np.take_along_axis(size, c2, axis=1)
    c3 = np.minimum(c2 + s2, width - 1)

    rows = np.arange(pop_size)
    with np.errstate(all="ignore"):
        for i in range(width - 1, -1, -1):
            size_i = size[:, i]
            active = size_i > 0
            if not active.any():
                continue
            leaf = active & (size_i == 1)
            function = active & (size_i > 1)
            column = values[:, i, :]

            if leaf.any():
                leaf_opcode = opcode[leaf, i]
                const = payload[leaf, i]
                index = np.clip(
                    np.trunc(const).astype(np.int64), 0, x.shape[1] - 1
                )
                column[leaf] = np.where(
                    (leaf_opcode == NType.CONST)[:, None], const[:, None], x[:, index].T
                )

            if not function.any():
                continue

            node_rows = np.nonzero(function)[0]
            node_ids = opcode[node_rows, i]
            arities = _ARITY[np.clip(node_ids, 0, Func.END - 1)]
            first = values[:, min(i + 1, width - 1), :]
            second = (
                values[rows, c2[:, i], :]
                if bool((arities >= 2).any())
                else None
            )
            third = (
                values[rows, c3[:, i], :] if bool((arities >= 3).any()) else None
            )

            for node_id in np.unique(node_ids):
                selected = node_rows[node_ids == node_id]
                op = int(node_id)
                if op == Func.IF:
                    # The ternary branch always computes `if` (the kernels ignore
                    # the function id of a ternary node).
                    column[selected] = np.where(
                        first[selected] > _ZERO, second[selected], third[selected]
                    )
                    continue
                if op <= Func.GE:
                    binary = _BINARY.get(op)
                    # An id that matches no branch leaves the value at 0.0, the
                    # initializer of the kernels; a well-formed population never
                    # gets here (validate_channels checks the arity).
                    column[selected] = (
                        binary(first[selected], second[selected])
                        if binary is not None
                        else _ZERO
                    )
                else:
                    unary = _UNARY.get(op)
                    column[selected] = (
                        unary(first[selected]) if unary is not None else _ZERO
                    )
    return values[:, 0, :]


def predict(
    channels,
    x,
    *,
    input_len: Optional[int] = None,
    validate: bool = True,
    row_chunk: Optional[int] = None,
    budget_mb: Optional[float] = None,
) -> np.ndarray:
    """Evaluate a prefix-encoded population on ``x``.

    Args:
        channels: a :class:`~evogp.gp.state.GPState` or the ``(opcode, payload,
            size)`` triple of a population (numpy arrays or ``evoxc`` tensors; a
            tensor is read back with ``.numpy()``).  The three 1-D channel rows
            of ONE individual (``GPResult.best_channels``) are accepted too and
            evaluate as a one-row population.
        x: the data matrix ``[n_rows, input_len]`` float32 (a 1-D array is one
            row).
        input_len: when given, ``VAR`` payloads must index into ``[0,
            input_len)``; it also has to match ``x.shape[1]``.  Defaults to
            ``x.shape[1]``.
        validate: run :func:`~evogp.gp.encoding.validate_channels` on the
            population first (default on).  It is a Python pass over the
            population, not over the data rows; turn it off in a hot loop whose
            population is structurally trusted.
        row_chunk: number of data rows evaluated per block (default: as many as
            the budget allows).
        budget_mb: value-buffer budget in MiB (``EVOGP_GP_EVAL_BUDGET_MB``).

    Returns:
        ``[pop_size, n_rows]`` float32 -- one prediction per tree and data row.

    Raises:
        ValueError: for a malformed population (naming the row and the node) or
            a ``VAR`` index outside ``[0, input_len)``.
        NotImplementedError: for ``OUT_NODE``-tagged (multi-output) opcodes.
    """
    opcode, payload, size = _prepare_channels(channels)
    data = _prepare_data(x)
    if input_len is not None and int(input_len) != data.shape[1]:
        raise ValueError(
            f"input_len={input_len} contradicts X's {data.shape[1]} columns"
        )
    columns = data.shape[1]
    if validate:
        validate_channels((opcode, payload, size), input_len=columns)
    pop_size, width = opcode.shape
    rows = _row_chunk(pop_size, width, data.shape[0], row_chunk, budget_mb)
    if rows >= data.shape[0]:
        return _evaluate_block(opcode, payload, size, data)
    return np.concatenate(
        [
            _evaluate_block(opcode, payload, size, data[start : start + rows])
            for start in range(0, data.shape[0], rows)
        ],
        axis=1,
    )
