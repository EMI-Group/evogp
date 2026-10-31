"""The symbolic-regression problem component of EvoGP's Evox 2.0 integration.

An *Evox 2.0 problem component* is a plain module holding a frozen config
dataclass plus a module-level ``evaluate(config, problem_state, pop)`` (and,
for a stateful problem, ``init(config, key)``); ``evox.core.StdWorkflow``
resolves them with ``importlib.import_module(type(config).__module__)`` and
traces ``evaluate`` into the step graph. The stateful form is what this module
uses: the data rows and the targets are *tensors of the problem state*, so they
enter the step graph as runtime inputs -- uploaded once by ``init`` -- instead
of being baked into the compiled program as graph constants (which is what
``evox``'s numerical problems do with their ``shift``/``affine`` arrays; that
works for a `(dim,)` vector, not for a data table).

The evaluation itself is :mod:`evogp.gp.graph_eval`'s **in-graph** route: the
population's three channels are decomposed with
``evogp.gp.graph_eval._channel_triple``, the static problem description is built
with ``evogp.gp.graph_eval._cfg`` and the traced body is
``evogp.gp.graph_eval._device_sr_fitness``, i.e. exactly the code
:class:`evogp.gp.graph_eval.GraphEvaluator` runs. It is traced HERE rather than
called through ``GraphEvaluator.evaluate`` because that method builds and runs
its own executable -- ``StdWorkflow`` composes the problem's ``evaluate`` into
ITS graph, and a nested build inside an active trace is not allowed. The
host-callback :class:`evogp.gp.fused_eval.FusedDeviceEvaluator` cannot be used
inside a workflow for the same reason (it runs its own Triton kernel from the
host); the in-graph route is the one that fits the component contract.

Sign convention
---------------
``evaluate`` returns the **MAXIMIZED** fitness (the repo-wide EvoGP contract:
the SR loss negated, non-finite mapped to ``-inf``). ``StdWorkflow`` scales it
into its own orientation, so pair this problem with
:class:`evogp.evox_ext.gp_algorithm.GPAlgorithmConfig` and
``opt_direction="max"`` -- :func:`evogp.evox_ext.stdworkflow.make_stdworkflow`
does that by default.

Notes:
    * ``pop`` is the candidate object the algorithm handed to ``evaluate``:
      the :class:`evogp.gp.state.GPState` (or its ``(opcode, payload, size)``
      triple), i.e. the same input ``evogp.gp``'s own ``evaluate=`` callables
      take.
    * ``used_ids`` narrows the traced dispatch chain to the function ids the
      run's tree config can produce (``evogp.gp.graph_eval.resolve_used_ids``);
      ``None`` traces all of :class:`~evogp.gp._constants.Func`, which is always
      correct and simply a wider chain. A population carrying an id outside the
      set is marked ``-inf`` rather than silently mis-evaluated.
    * Nothing here imports ``evox``; ``evoxc`` is resolved lazily by
      ``evogp.gp``, so importing :mod:`evogp.evox_ext` works on a box without
      the EvoX checkout.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import numpy as np

from evogp.gp import _runtime
from evogp.gp import graph_eval as _graph_eval

__all__ = [
    "SRProblemConfig",
    "SRProblem",
    "SRProblemState",
    "make_sr_problem",
    "init",
    "evaluate",
]


@dataclasses.dataclass(frozen=True)
class SRProblemConfig:
    """A single-output symbolic-regression problem (the ``evox`` problem config).

    Args:
        n_rows: number of data rows (``y``'s length, ``x``'s first dimension).
        input_len: number of input variables (``x``'s second dimension) -- the
            ``GPTreeConfig.input_len`` of the paired algorithm.
        x: the data, row-major flat (``n_rows * input_len`` values) or a 2-D
            array-like of that shape; normalized to a float32 tuple here.
        y: the targets, ``n_rows`` values.
        use_mse: squared (default) vs absolute error, ``SymbolicRegression``'s
            knob.
        used_ids: the function ids the traced dispatch chain covers -- a
            sequence of ids / function names or a
            :class:`~evogp.gp.config.GPTreeConfig` (its ``using_funcs``;
            narrows and speeds up the chain). ``None`` covers every id of
            :class:`~evogp.gp._constants.Func`, which is always correct.
        sanitize: map a non-finite fitness to ``-inf`` in-graph (the repo-wide
            fitness contract; leave it on).

    The data lives in the config because that is what ``init(config, key)``
    receives; it is stored as plain float tuples so the config stays hashable
    and immutable, and uploaded to the problem state as tensors by :func:`init`.
    """

    n_rows: int
    input_len: int
    x: Tuple[float, ...]
    y: Tuple[float, ...]
    use_mse: bool = True
    used_ids: Optional[Tuple[int, ...]] = None
    sanitize: bool = True

    def __post_init__(self) -> None:
        rows = _positive_int(self.n_rows, "n_rows")
        columns = _positive_int(self.input_len, "input_len")
        object.__setattr__(self, "n_rows", rows)
        object.__setattr__(self, "input_len", columns)
        object.__setattr__(
            self, "x", _float_tuple(self.x, "x", rows * columns)
        )
        object.__setattr__(self, "y", _float_tuple(self.y, "y", rows))
        # Normalized once, on the host: the traced dispatch chain closes over
        # this tuple, so it must be static and hashable.
        object.__setattr__(
            self, "used_ids", _graph_eval.resolve_used_ids(self.used_ids)
        )
        object.__setattr__(self, "use_mse", bool(self.use_mse))
        object.__setattr__(self, "sanitize", bool(self.sanitize))


#: Evox's naming convention spells the config ``<Problem>Config`` and the bare
#: problem name as an alias of it (``Sphere``/``SphereConfig``).
SRProblem = SRProblemConfig


@dataclasses.dataclass(frozen=True)
class SRProblemState:
    """The problem state: the data and the targets as ``evoxc`` tensors.

    Held in the workflow state, so they are runtime inputs of the step graph
    (uploaded once, moved to the workflow's device by ``StdWorkflow``) and
    never re-traced.
    """

    x: Any  # float32 [n_rows, input_len]
    y: Any  # float32 [n_rows]


def make_sr_problem(
    x: Any,
    y: Any,
    *,
    use_mse: bool = True,
    used_ids: Any = None,
    sanitize: bool = True,
) -> SRProblemConfig:
    """Build an :class:`SRProblemConfig` from host arrays.

    Args:
        x: ``[n_rows, input_len]`` float array-like (a 1-D array is one row).
        y: ``[n_rows]`` targets, float array-like.
        use_mse: squared (default) vs absolute error.
        used_ids: see :attr:`SRProblemConfig.used_ids` (a
            :class:`~evogp.gp.config.GPTreeConfig` is the convenient spelling).
        sanitize: in-graph non-finite -> ``-inf`` (leave it on).

    Returns:
        The validated config, with ``x``/``y`` as plain float tuples.

    Raises:
        ValueError: for a non-2-D ``x``, mismatched lengths or non-finite data.
    """
    data = np.asarray(x, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError(
            f"x must be 1-D or 2-D [n_rows, input_len], got shape {data.shape}"
        )
    rows, columns = (int(value) for value in data.shape)
    targets = np.asarray(y, dtype=np.float32).reshape(-1)
    if int(targets.size) != rows:
        raise ValueError(
            f"y has {int(targets.size)} entries but x has {rows} rows"
        )
    return SRProblemConfig(
        n_rows=rows,
        input_len=columns,
        x=tuple(float(value) for value in data.reshape(-1)),
        y=tuple(float(value) for value in targets),
        use_mse=use_mse,
        used_ids=used_ids,
        sanitize=sanitize,
    )


# --------------------------------------------------------------------------- #
# the problem protocol
# --------------------------------------------------------------------------- #
def init(config: SRProblemConfig, key: Any) -> SRProblemState:
    """Upload the data and the targets into ``evoxc`` tensors.

    Runs inside ``StdWorkflow``'s init graph, which is built for the numpy
    backend on the CPU; every returned leaf is a graph constant, so the state
    carries concrete tensors once the init graph has run. ``StdWorkflow`` moves
    them to the workflow's device afterwards, and the step graph then takes them
    as inputs.

    Args:
        config: the :class:`SRProblemConfig`.
        key: unused (this problem is deterministic); kept for the protocol.

    Returns:
        The :class:`SRProblemState`.
    """
    evoxc = _runtime.evoxc()
    data = np.asarray(config.x, dtype=np.float32).reshape(
        int(config.n_rows), int(config.input_len)
    )
    targets = np.asarray(config.y, dtype=np.float32).reshape(int(config.n_rows))
    return SRProblemState(
        x=_constant(evoxc, np.ascontiguousarray(data)),
        y=_constant(evoxc, np.ascontiguousarray(targets)),
    )


def evaluate(
    config: SRProblemConfig, state: SRProblemState, pop: Any
) -> Tuple[Any, SRProblemState]:
    """Evaluate a whole GP population in-graph.

    Args:
        config: the :class:`SRProblemConfig`.
        state: the :class:`SRProblemState` holding the data tensors.
        pop: the population the algorithm handed to ``evaluate`` -- a
            :class:`evogp.gp.state.GPState` or its ``(opcode, payload, size)``
            triple of ``[pop_size, max_nodes]`` tensors.

    Returns:
        ``(fitness, state)`` with ``fitness`` a ``[pop_size]`` float32
        MAXIMIZED vector (the SR loss negated, non-finite mapped to ``-inf``)
        and the problem state threaded through unchanged.

    Raises:
        ValueError: for a channel shape the in-graph route cannot take (the
            device path needs the 2-D population form).
    """
    opcode, payload, size, pop_size, width = _graph_eval._channel_triple(pop)
    if int(config.input_len) != int(state.x.shape[1]):
        raise ValueError(
            f"config.input_len={int(config.input_len)} contradicts the problem "
            f"state's {int(state.x.shape[1])} data columns"
        )
    if int(config.n_rows) != int(state.x.shape[0]):
        raise ValueError(
            f"config.n_rows={int(config.n_rows)} contradicts the problem "
            f"state's {int(state.x.shape[0])} rows"
        )
    cfg = _graph_eval._cfg(
        width,
        pop_size,
        int(config.n_rows),
        int(config.input_len),
        config.used_ids,
        bool(config.use_mse),
        bool(config.sanitize),
    )
    fitness = _graph_eval._device_sr_fitness(
        opcode, payload, size, state.x, state.y, cfg
    )
    return fitness, state


# --------------------------------------------------------------------------- #
# host-side helpers
# --------------------------------------------------------------------------- #
def _constant(evoxc_module, value: np.ndarray):
    """``value`` as a traced graph constant (the ``evox`` config-array idiom)."""
    return evoxc_module.constant(evoxc_module.tensor(value))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def _float_tuple(value: Any, name: str, expected: int) -> Tuple[float, ...]:
    """``value`` as a finite float32 tuple of exactly ``expected`` entries."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(array.size) != int(expected):
        raise ValueError(
            f"{name} has {int(array.size)} values but the declared shape needs "
            f"{int(expected)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite, got non-finite entries")
    return tuple(float(item) for item in array)
