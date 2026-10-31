"""``StdWorkflow`` wiring helpers for the EvoGP components.

The component contract (an algorithm module + a problem module, resolved by
``type(config).__module__``) is all ``evox.core.StdWorkflow`` needs, but the
PAIRING still has an orientation: ``evogp.gp`` fitness is MAXIMIZED while
``StdWorkflow`` hands its algorithm an ``opt_direction``-scaled vector and
defaults to ``"min"``. :func:`make_stdworkflow` builds the pair the way EvoGP
means it -- ``opt_direction="max"``, so the components and the workflow agree
on the sign -- and the two readback helpers below report the run in EvoGP's
own units.

Typical use::

    workflow = make_stdworkflow(
        make_gp_algorithm(tree_config, gp_config),
        make_sr_problem(x, y, used_ids=tree_config),
        cache_dir="~/.cache/evogp/evox_steps",   # opt-in: reuse the step artifact
    )
    state = workflow.init(seed=0)
    for _ in range(generations):
        state = workflow.step(state)
        print(best_so_far(state))

Nothing here is traced: ``StdWorkflow`` owns the graph composition, and these
helpers are host-side glue (``evox`` is imported inside the functions so that
``import evogp.evox_ext`` stays cheap on a box without the EvoX checkout).
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional

import numpy as np

from evogp.gp import encoding as _encoding
from evogp.gp import graph_eval as _graph_eval
from evogp.gp.state import GPState

__all__ = [
    "make_stdworkflow",
    "accepts_cache_dir",
    "host_channels",
    "generation_fitness",
    "best_so_far",
    "best_row",
    "OPT_DIRECTION",
]

#: The orientation the EvoGP components are built for (fitness is MAXIMIZED).
OPT_DIRECTION = "max"

#: The opt-in step-artifact cache keyword of ``evox.core.StdWorkflow``.
CACHE_DIR_KWARG = "cache_dir"


def accepts_cache_dir(workflow_cls: Any) -> bool:
    """Whether ``workflow_cls.__init__`` declares a ``cache_dir`` keyword.

    The step-artifact cache (``StdWorkflow(cache_dir=...)``) is an opt-in
    ``evox`` feature, so the forward in :func:`make_stdworkflow` is
    FEATURE-DETECTED rather than pinned: an ``evox`` whose workflow builds the
    step graph every time takes no such keyword. A signature that cannot be
    introspected (a builtin type) answers ``False``, i.e. "do not pass it".
    """
    try:
        parameters = inspect.signature(workflow_cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtin/C-extension class
        return False
    return CACHE_DIR_KWARG in parameters


def make_stdworkflow(
    algorithm: Any,
    problem: Any,
    *,
    backend: str = "numpy",
    device: Any = None,
    num_generations: Optional[int] = None,
    monitor: Any = None,
    compile_options: Optional[dict] = None,
    cache_dir: Any = None,
    opt_direction: str = OPT_DIRECTION,
) -> Any:
    """Build an :class:`evox.core.StdWorkflow` for the EvoGP components.

    Args:
        algorithm: a :class:`evogp.evox_ext.gp_algorithm.GPAlgorithmConfig`.
        problem: a :class:`evogp.evox_ext.sr_problem.SRProblemConfig`.
        backend: ``evoxc`` backend name (``"numpy"`` default, ``"iree"`` for a
            device run).
        device: an ``evoxc`` device (spelling or ``evoxc.Device``); ``None`` is
            the CPU.
        num_generations: the default generation count of ``run``/``fit``.
        monitor: an ``evox`` monitor config, or ``None``. ``evogp.evox_ext``
            ships no monitor: the candidate object is a GP population (a
            ``GPState``), not an ``(n, dim)`` matrix, so an ``EvalMonitor``
            would record channels it cannot interpret -- read the trajectory
            with :func:`best_so_far` instead.
        compile_options: extra keyword options forwarded to ``evoxc.build``.
        cache_dir: opt-in step-artifact cache directory (``str`` or
            ``os.PathLike``), forwarded to ``StdWorkflow(cache_dir=...)``: one
            content-addressed ``evoxc`` artifact per state signature, so a
            later process reloads the compiled step graph instead of
            recompiling it. ``None`` (the default) keeps the historical
            build-every-time behaviour. An ``evox`` that declares no
            ``cache_dir`` keyword cannot honour it: the workflow is built
            without the argument and a ``RuntimeWarning`` names the drop.
        opt_direction: must stay ``"max"`` (the components' contract).

    Returns:
        The configured :class:`evox.core.StdWorkflow`.

    Raises:
        ValueError: for an ``opt_direction`` other than ``"max"``.
        ImportError: when ``evox`` is not importable.

    Warns:
        RuntimeWarning: when ``cache_dir`` is given but the installed ``evox``
        declares no such keyword (the request is dropped, not fatal).
    """
    if opt_direction != OPT_DIRECTION:
        raise ValueError(
            f"the EvoGP components maximize fitness, so opt_direction must be "
            f"{OPT_DIRECTION!r}, got {opt_direction!r}; a minimization "
            "orientation would invert the selection pressure"
        )
    from evox.core import StdWorkflow  # noqa: PLC0415 - optional dependency

    kwargs = {
        "monitor": monitor,
        "opt_direction": opt_direction,
        "num_generations": num_generations,
        "backend": backend,
        "device": device,
        "compile_options": compile_options,
    }
    if accepts_cache_dir(StdWorkflow):
        kwargs[CACHE_DIR_KWARG] = cache_dir
    elif cache_dir is not None:
        warnings.warn(
            "the installed evox's StdWorkflow declares no cache_dir keyword "
            "(it predates the opt-in step-artifact cache), so "
            f"cache_dir={cache_dir!r} is ignored and every process recompiles "
            "the step graph; an evox carrying StdWorkflow(cache_dir=...) "
            "honours it",
            RuntimeWarning,
            stacklevel=2,
        )
    return StdWorkflow(algorithm, problem, **kwargs)


def _algorithm_state(state: Any) -> Any:
    """The GP algorithm state of a workflow state (``ValueError`` if absent)."""
    algorithm_state = getattr(state, "algorithm_state", None)
    if not isinstance(algorithm_state, GPState):
        raise ValueError(
            "the workflow state carries no GP algorithm state (expected "
            f"state.algorithm_state to be a GPState); got "
            f"{type(algorithm_state).__name__}"
        )
    return algorithm_state


def host_channels(state: Any) -> tuple:
    """``(opcode, payload, size)`` of a workflow state as host numpy arrays.

    The one explicit download of a device run (``evoxc`` never transfers
    implicitly; the move goes through ``evogp.gp.encoding._as_numpy``, the same
    helper ``evogp.gp.workflow`` uses for its final report), returned as fresh
    writable arrays so a caller can inspect the winning trees without touching
    the run's device buffers.

    Args:
        state: a ``StdWorkflow`` state (``WorkflowState``).

    Returns:
        Three ``(pop_size, max_nodes)`` numpy arrays -- int32, float32, int32.

    Raises:
        ValueError: when the state carries no GP algorithm state.
    """
    algorithm_state = _algorithm_state(state)
    return tuple(
        np.array(_encoding._as_numpy(channel, name), copy=True)
        for name, channel in zip(("opcode", "payload", "size"), algorithm_state.channels())
    )


def generation_fitness(state: Any) -> np.ndarray:
    """The fitness vector the last step wrote, as host numpy (MAXIMIZED).

    ``evogp.gp``'s operator layer writes the next generation's fitness in the
    same graph that assembles it: the elite rows' (known) values followed by
    ``-inf`` for the offspring. So after ``n`` steps this is the elite fitness
    of generation ``n - 1`` -- the best-so-far reading a loop wants.

    Args:
        state: a ``StdWorkflow`` state (``WorkflowState``).

    Returns:
        A fresh ``(pop_size,)`` float32 numpy array, non-finite entries set to
        ``-inf``.

    Raises:
        ValueError: when the state carries no GP algorithm state.
    """
    fitness = _algorithm_state(state).fitness
    return _graph_eval.readback_fitness(fitness, int(fitness.shape[0]))


def best_so_far(state: Any) -> float:
    """The best MAXIMIZED fitness in the workflow state (``-inf`` if unevaluated)."""
    fitness = generation_fitness(state)
    finite = fitness[np.isfinite(fitness)]
    return float(finite.max()) if finite.size else float("-inf")


def best_row(state: Any) -> int:
    """The row index of :func:`best_so_far` in the workflow state.

    Raises:
        ValueError: when nothing has been evaluated yet.
    """
    fitness = generation_fitness(state)
    if not np.isfinite(fitness).any():
        raise ValueError("nothing has been evaluated yet")
    return int(np.argmax(fitness))
