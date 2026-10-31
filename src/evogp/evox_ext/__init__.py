"""EvoGP's Evox 2.0 integration surface: run the torch-free GP stack in ``StdWorkflow``.

EvoX 2.0 is a functional rewrite: an *algorithm* and a *problem* are plain
modules exposing a frozen config dataclass plus module-level ``init`` / ``step``
/ ``evaluate`` functions, and ``evox.core.StdWorkflow`` resolves them with
``importlib.import_module(type(config).__module__)`` and composes one generation
into a single compiled ``evoxc`` graph (``evox.core.algorithm`` /
``evox.core.problem`` are the documentation-only protocols).

This package adapts :mod:`evogp.gp` -- EvoGP's torch-free functional GP stack --
to that contract, so EvoGP is a first-class Evox 2.0 citizen instead of a
stand-alone library:

* :class:`~evogp.evox_ext.gp_algorithm.GPAlgorithmConfig` -- the GP algorithm
  component (population generation and one generation of
  select/crossover/mutate with elitism, both traced into the workflow's graph);
* :class:`~evogp.evox_ext.sr_problem.SRProblemConfig` -- the single-output
  symbolic-regression problem component (data held in the problem state, the
  fitness traced in-graph) ;
* :func:`~evogp.evox_ext.stdworkflow.make_stdworkflow` -- builds the
  ``StdWorkflow`` with the orientation the pair requires
  (``opt_direction="max"``), plus the fitness readbacks a run wants.

Example::

    from evogp.evox_ext import make_gp_algorithm, make_sr_problem, make_stdworkflow
    from evogp.gp.config import GPConfig, GPTreeConfig

    tree = GPTreeConfig(max_nodes=64, input_len=2, using_funcs=("+", "-", "*", "/"))
    algorithm = make_gp_algorithm(tree, GPConfig(pop_size=200))
    problem = make_sr_problem(x, y, used_ids=tree)
    workflow = make_stdworkflow(algorithm, problem)
    state = workflow.run(generations=50)

There is deliberately NO ``evox_ext/`` shim (EvoX's PEP 420 autoload namespace,
``src/evox_ext/<domain>/<module>.py`` merged into ``evox.<domain>`` at
``import evox``): the component-resolution contract is by MODULE, so a workflow
takes these configs directly and the plugin namespace would only buy the
shorter spelling ``evox.algorithms.…`` at the price of installing a third
top-level import package and running EvoGP code at ``import evox`` time. See
``CONTEXT.md`` for the full rationale.
"""

from __future__ import annotations

from evogp.evox_ext.gp_algorithm import (
    GPAlgorithm,
    GPAlgorithmConfig,
    make_gp_algorithm,
)
from evogp.evox_ext.sr_problem import (
    SRProblem,
    SRProblemConfig,
    SRProblemState,
    make_sr_problem,
)
from evogp.evox_ext.stdworkflow import (
    OPT_DIRECTION,
    accepts_cache_dir,
    best_row,
    best_so_far,
    generation_fitness,
    host_channels,
    make_stdworkflow,
)

__all__ = [
    "GPAlgorithmConfig",
    "GPAlgorithm",
    "make_gp_algorithm",
    "SRProblemConfig",
    "SRProblem",
    "SRProblemState",
    "make_sr_problem",
    "make_stdworkflow",
    "accepts_cache_dir",
    "generation_fitness",
    "host_channels",
    "best_so_far",
    "best_row",
    "OPT_DIRECTION",
]
