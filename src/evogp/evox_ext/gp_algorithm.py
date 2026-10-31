"""The GP algorithm component of EvoGP's Evox 2.0 integration surface.

An *Evox 2.0 algorithm component* is a plain module holding a frozen config
dataclass **plus** the module-level functions ``init(config, key)`` and
``step(config, state, evaluate)``; ``evox.core.StdWorkflow`` resolves them
with ``importlib.import_module(type(config).__module__)`` and composes ``step``
into one compiled graph per step variant and state signature (see
``evox.core.algorithm`` -- the protocol is documentation only, nothing
inherits from it).

This module is the thin EvoGP adapter: it owns no GP logic of its own, it
hands the work to the torch-free stack in :mod:`evogp.gp`.

* ``init`` -- :func:`evogp.gp.encoding._generator_fn`'s traced generator draws
  the generation-0 population (the level-wise sampler + one ``argsort``
  placement of ``DESIGN.md`` section "Generation"), and the fitness vector is
  the all-``-inf`` "unevaluated" marker the design uses.
* ``step`` -- the workflow's ``evaluate`` closure is called once, the returned
  fitness is turned back into the MAXIMIZED orientation the operator layer
  ranks on, and :func:`evogp.gp.variation._variation_fn`'s traced operator does
  selection, crossover and mutation and splices the ``GPConfig.elite_count``
  elite rows in front of the offspring -- one traced call, no second
  executable for the splice.

Both builders return the **traced** function, never a built executable:
``evogp.gp``'s public entry points (``init_population`` / ``variation``) call
``evoxc.build_cached`` themselves and a build inside an active trace is not
allowed, while ``StdWorkflow`` needs the plain body to compose into ITS graph.

The sign convention
-------------------
``evogp.gp`` fitness is MAXIMIZED (the repo-wide contract: losses negated),
and ``StdWorkflow`` hands the algorithm an ``opt_direction``-scaled vector --
``-fitness`` when the workflow runs in ``"max"`` orientation. :func:`step`
therefore negates ONCE. Pair this algorithm with a problem that returns
MAXIMIZED fitness (such as :class:`evogp.evox_ext.sr_problem.SRProblemConfig`)
and with ``opt_direction="max"``;
:func:`evogp.evox_ext.stdworkflow.make_stdworkflow` builds exactly that pair.

Notes:
    * Nothing here imports ``evox``: an algorithm component is resolved by
      module, so ``evogp.evox_ext`` stays importable on a box without the EvoX
      checkout. ``evoxc`` is resolved at trace time through ``evogp.gp``'s own
      lazy accessor.
    * The state is :class:`evogp.gp.state.GPState` unchanged -- five ``evoxc``
      tensors whose shapes are constant for a whole run (``opcode``/``payload``/
      ``size`` are ``[pop_size, max_nodes]``, ``fitness`` is ``[pop_size]`` and
      ``key`` is rank-0), which is exactly the ``(shape, dtype)`` stability
      ``StdWorkflow._state_signature`` needs to keep ONE compiled step graph.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from evogp.gp import _runtime
from evogp.gp import encoding as _encoding
from evogp.gp import variation as _variation
from evogp.gp.config import GPConfig, GPTreeConfig
from evogp.gp.state import UNEVALUATED, GPState

__all__ = [
    "GPAlgorithmConfig",
    "GPAlgorithm",
    "make_gp_algorithm",
    "init",
    "step",
]


@dataclasses.dataclass(frozen=True)
class GPAlgorithmConfig:
    """The GP loop's static configuration (the ``evox`` algorithm config).

    Args:
        tree: the tree blueprint -- :class:`evogp.gp.config.GPTreeConfig`
            (buffer width, function set, depth schedule, constant table).
        gp: the loop's parameters -- :class:`evogp.gp.config.GPConfig`
            (``pop_size``, ``elite_count``, tournament pressure, operator
            rates, mutation kind).

    Both are already frozen, hashable, static dataclasses, so this config is
    static too: it is closed over when ``StdWorkflow`` traces the step graph,
    and changing a field produces a different graph (never a mutated run).
    """

    tree: GPTreeConfig
    gp: GPConfig

    def __post_init__(self) -> None:
        for name, expected in (("tree", GPTreeConfig), ("gp", GPConfig)):
            value = getattr(self, name)
            if not isinstance(value, expected):
                raise TypeError(
                    f"{type(self).__name__}.{name} must be an "
                    f"{expected.__name__}, got {type(value).__name__}"
                )
        if int(self.gp.pop_size) <= 0:
            raise ValueError(
                f"gp.pop_size must be positive, got {self.gp.pop_size}"
            )
        if int(self.tree.max_nodes) < int(self.tree.padded_total()):
            raise ValueError(
                f"tree.max_nodes={self.tree.max_nodes} cannot hold the padded "
                f"generation layout of {self.tree.padded_total()} slots"
            )


#: Evox's naming convention spells the config ``<Algorithm>Config`` and the
#: bare algorithm name as an alias of it (``OpenES``/``OpenESConfig``).
GPAlgorithm = GPAlgorithmConfig


def make_gp_algorithm(
    tree: GPTreeConfig,
    gp: Optional[GPConfig] = None,
    *,
    pop_size: Optional[int] = None,
    **gp_options: Any,
) -> GPAlgorithmConfig:
    """Build a :class:`GPAlgorithmConfig`, validating the pair.

    Args:
        tree: the :class:`~evogp.gp.config.GPTreeConfig`.
        gp: an explicit :class:`~evogp.gp.config.GPConfig`. Either this or
            ``pop_size`` must be given.
        pop_size: shortcut for ``GPConfig(pop_size=..., **gp_options)``.
        **gp_options: further ``GPConfig`` fields (only with ``pop_size``);
            the other fields keep their ``GPConfig`` defaults.

    Returns:
        The validated :class:`GPAlgorithmConfig`.

    Raises:
        TypeError: for a non-config argument or a contradictory pair.
        ValueError: for a bad ``pop_size`` or an unknown ``gp_options`` key.
    """
    if gp is not None:
        if pop_size is not None or gp_options:
            raise TypeError(
                "make_gp_algorithm: pass either gp= or pop_size=/gp_options=, "
                "not both"
            )
        return GPAlgorithmConfig(tree=tree, gp=gp)
    if pop_size is None:
        raise TypeError(
            "make_gp_algorithm needs a GPConfig (gp=) or a pop_size= together "
            "with the other GPConfig fields"
        )
    return GPAlgorithmConfig(tree=tree, gp=GPConfig(pop_size=pop_size, **gp_options))


def _configs(config: GPAlgorithmConfig) -> tuple:
    """:func:`make_gp_algorithm`'s validation for a hand-built config."""
    if not isinstance(config, GPAlgorithmConfig):
        raise TypeError(
            "the algorithm component needs a GPAlgorithmConfig (or its alias "
            f"GPAlgorithm), got {type(config).__name__}"
        )
    return config.tree, config.gp


# --------------------------------------------------------------------------- #
# the step protocol
# --------------------------------------------------------------------------- #
def init(config: GPAlgorithmConfig, key: Any) -> GPState:
    """The generation-0 population (:func:`evogp.gp.init_population`, traced).

    Called by ``StdWorkflow`` inside its init graph, so everything here is a
    traced expression: the population is drawn by the same traced generator
    ``evogp.gp``'s ``init_population`` wraps, and the fitness vector is the
    ``-inf`` "not evaluated yet" marker.

    Args:
        config: the :class:`GPAlgorithmConfig`.
        key: the rank-0 ``int64`` RNG key the workflow splits off its seed.

    Returns:
        A :class:`~evogp.gp.state.GPState` holding the three channels and the
        *next* generation's key.
    """
    tree, gp = _configs(config)
    generate = _encoding._generator_fn(tree, int(gp.pop_size))
    opcode, payload, size, next_key = generate(key)
    evoxc = _runtime.evoxc()
    fitness = evoxc.broadcast(
        _runtime.f32(evoxc, UNEVALUATED), (int(gp.pop_size),)
    )
    return GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=fitness,
        key=next_key,
    )


def step(
    config: GPAlgorithmConfig,
    state: GPState,
    evaluate: Callable[[Any], Any],
) -> GPState:
    """Run ONE generation: evaluate the population, then select/cross/mutate.

    Args:
        config: the :class:`GPAlgorithmConfig`.
        state: the current :class:`~evogp.gp.state.GPState`.
        evaluate: the workflow's traced closure. It applies the solution
            transform, the problem's ``evaluate`` and the ``opt_direction``
            scaling, and returns a ``[pop_size]`` ``float32`` vector in the
            workflow's (minimization) orientation.

    Returns:
        The next generation: the elite rows (bit-identical copies of the
        parents) followed by the offspring, the fitness vector carrying the
        elite values and ``-inf`` for the unevaluated offspring, and the
        advanced key.

    Raises:
        ValueError: when the state's channel shape contradicts the config or
            when ``evaluate`` returns a vector of the wrong length.
    """
    tree, gp = _configs(config)
    pop_size, width = int(state.opcode.shape[0]), int(state.opcode.shape[1])
    if pop_size != int(gp.pop_size):
        raise ValueError(
            f"the state holds {pop_size} rows but the config says "
            f"gp.pop_size={int(gp.pop_size)}; a population size change is a "
            "different run (the step graph is traced per state shape)"
        )
    if width != int(tree.max_nodes):
        raise ValueError(
            f"the state is {width} nodes wide but the config says "
            f"tree.max_nodes={int(tree.max_nodes)}"
        )

    operator = _variation._variation_fn(
        tree, gp, pop_size, width, int(gp.elite_count)
    )
    scaled = evaluate(state)
    shape = tuple(int(dim) for dim in scaled.shape)
    if shape != (pop_size,):
        raise ValueError(
            f"evaluate returned shape {shape}; the contract is a "
            f"[pop_size={pop_size}] float32 fitness vector"
        )
    # `StdWorkflow` scaled the fitness into its own orientation (opt_direction
    # "max" means it arrives negated); the operator layer ranks MAXIMIZED.
    fitness = _runtime.evoxc().negate(scaled)
    opcode, payload, size, next_fitness, next_key = operator(
        state.opcode, state.payload, state.size, fitness, state.key
    )
    return GPState(
        opcode=opcode,
        payload=payload,
        size=size,
        fitness=next_fitness,
        key=next_key,
    )
