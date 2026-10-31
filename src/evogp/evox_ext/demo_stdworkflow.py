"""Run the torch-free GP stack inside ``evox`` 2.0's ``StdWorkflow`` (a demo).

The script EvoGP's Evox 2.0 integration is validated with: it builds an
``evox.core.StdWorkflow`` out of the two components in
:mod:`evogp.evox_ext` -- the GP algorithm
(:class:`~evogp.evox_ext.gp_algorithm.GPAlgorithmConfig`) and a single-output
symbolic-regression problem
(:class:`~evogp.evox_ext.sr_problem.SRProblemConfig`) -- and runs a short
generation loop on a synthetic target, printing the best-so-far fitness
trajectory and the wall-clock cost of a generation.

Usage (CPU, the numpy backend -- the defaults)::

    PYTHONPATH=src:/root/evoxc:/root/evox/src python -m evogp.evox_ext.demo_stdworkflow

Usage (a device box, the ``iree`` backend compiling for the local GPU)::

    PYTHONPATH=src:/root/evoxc:/root/evox/src \\
        python -m evogp.evox_ext.demo_stdworkflow --backend iree --device rocm:0

Everything is small on purpose: the numpy backend interprets the composed step
graph, so the shipped defaults (100 trees, 32-node trees, 256 data rows,
25 generations) run in a couple of seconds and the fitness improves visibly all
the way through. The evolution wiring (function set, constant table, depth
schedule, tournament size, survivor/elite counts, operator rates) is
**imported from** :func:`evogp.gp.smoke.make_configs`, the reference
configuration of the driver the workflow replaces, so the two are comparable
and nothing is duplicated here.

Unlike the torch-stack entry points (``evogp.sr_test``, ``example/*.py``, which
run at import time), this module is guarded: importing it does nothing.
"""

from __future__ import annotations

import os

# Tracing dominates a compile of this size, and the switch is a pure speed knob
# (it is not part of any cache key). Set before anything traces.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from evogp.evox_ext import (  # noqa: E402
    generation_fitness,
    host_channels,
    make_gp_algorithm,
    make_sr_problem,
    make_stdworkflow,
)
from evogp.gp.smoke import make_configs  # noqa: E402

__all__ = [
    "DEFAULT_POP_SIZE",
    "DEFAULT_GENERATIONS",
    "DEFAULT_NUM_DATA",
    "DEFAULT_MAX_NODES",
    "DEFAULT_SEED",
    "target_function",
    "make_data",
    "run_demo",
    "main",
]

#: Population size of the shipped demo run.
DEFAULT_POP_SIZE = 100
#: Number of evaluated generations of the shipped demo run.
DEFAULT_GENERATIONS = 25
#: Data rows of the synthetic problem.
DEFAULT_NUM_DATA = 256
#: Tree-buffer width (``max_nodes``).
DEFAULT_MAX_NODES = 32
#: RNG seed of the workflow (the driver's key and the data seed).
DEFAULT_SEED = 0
#: Bounds of the uniform data draw.
DATA_BOUNDS = (-5.0, 5.0)


def target_function(x: np.ndarray) -> np.ndarray:
    """``x0**2 + x1**2`` for ``x`` of shape ``[2, n_rows]`` (or ``[n_rows, 2]``)."""
    values = np.asarray(x, dtype=np.float64)
    if values.shape[0] == 2:
        first, second = values[0], values[1]
    else:
        first, second = values[:, 0], values[:, 1]
    return (first**2 + second**2).reshape(-1)


def make_data(
    num_data: int = DEFAULT_NUM_DATA, seed: int = DEFAULT_SEED
) -> tuple:
    """The synthetic SR problem: ``(x [num_data, 2], y [num_data])`` float32.

    A quadratic target on purpose: the reference workload of
    ``evogp.gp.smoke`` (``x0**4/(x0**4+1) + x1**4/(x1**4+1)``) needs hundreds of
    generations to move, while this one is reachable with ``+``/``*`` in a few
    dozen, which is what a short demo wants to show.
    """
    if not isinstance(num_data, int) or num_data <= 0:
        raise ValueError(f"num_data must be a positive int, got {num_data!r}")
    generator = np.random.default_rng(seed)
    x = generator.uniform(
        DATA_BOUNDS[0], DATA_BOUNDS[1], size=(int(num_data), 2)
    ).astype(np.float32)
    return x, target_function(x.T).astype(np.float32)


def run_demo(
    pop_size: int = DEFAULT_POP_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    num_data: int = DEFAULT_NUM_DATA,
    seed: int = DEFAULT_SEED,
    max_nodes: int = DEFAULT_MAX_NODES,
    backend: str = "numpy",
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Build the workflow, run ``generations`` steps and report the trajectory.

    Args:
        pop_size: number of trees.
        generations: number of evaluated generations.
        num_data: data rows of the synthetic problem.
        seed: the workflow's RNG seed (also the data seed).
        max_nodes: tree-buffer width.
        backend: ``evoxc`` backend (``"numpy"`` default, ``"iree"`` for a device).
        device: ``evoxc`` device spelling (``None`` = the CPU); ``"rocm:0"`` /
            ``"cuda:0"`` on a device box.
        verbose: print the per-generation trajectory.

    Returns:
        A dict with ``trajectory`` (the per-generation best MAXIMIZED fitness),
        ``best_fitness``, ``best_generation``, ``row`` (the winning row),
        ``step_seconds`` (mean warm generation), ``init_seconds`` (building
        BOTH graphs plus running generation 0 -- ``StdWorkflow.init`` builds the
        step graph, so the compile of a whole generation is paid there, once)
        and ``state`` (the final workflow state -- read the winning trees out of
        it with :func:`evogp.evox_ext.host_channels`).
    """
    x, y = make_data(num_data, seed=seed)
    tree_config, gp_config = make_configs(pop_size=pop_size, max_nodes=max_nodes)
    algorithm = make_gp_algorithm(tree_config, gp_config)
    problem = make_sr_problem(x, y, used_ids=tree_config)

    started = time.time()
    workflow = make_stdworkflow(algorithm, problem, backend=backend, device=device)
    state = workflow.init(seed=seed)
    # `StdWorkflow.init` also builds the step graph, so the compile of the whole
    # generation (evaluation + operators) is paid HERE, once.
    init_seconds = time.time() - started

    trajectory: list = []
    step_total = 0.0
    evaluated = generation_fitness(state)
    for generation in range(int(generations)):
        step_started = time.time()
        state = workflow.step(state)
        step_total += time.time() - step_started
        # ONE host readback per generation: the elite values the step graph
        # wrote (the offspring rows are the ``-inf`` "not evaluated yet" mark,
        # so their turn comes next generation).
        evaluated = generation_fitness(state)
        finite = evaluated[np.isfinite(evaluated)]
        trajectory.append(float(finite.max()) if finite.size else float("-inf"))
        if verbose:
            print(
                f"generation {generation:3d}  best {trajectory[-1]:+12.5f}  "
                f"elite {finite.size:3d}  {time.time() - started:6.2f}s",
                flush=True,
            )

    best_fitness = max(trajectory) if trajectory else float("-inf")
    best_generation = int(np.argmax(trajectory)) if trajectory else -1
    row = int(np.argmax(evaluated)) if np.isfinite(evaluated).any() else -1
    result = {
        "trajectory": trajectory,
        "best_fitness": best_fitness,
        "best_generation": best_generation,
        "init_seconds": init_seconds,
        "step_seconds": step_total / max(int(generations), 1),
        "state": state,
        "row": row,
    }
    if verbose:
        length = int(host_channels(state)[2][row, 0]) if row >= 0 else 0
        print(
            f"\n{int(generations)} generations in "
            f"{init_seconds + step_total:.2f}s "
            f"({1000.0 * result['step_seconds']:.2f} ms/generation warm, "
            f"{init_seconds:.2f}s of it the one-time graph build in init())"
        )
        print(
            f"best fitness {best_fitness:+.5f} at generation {best_generation} "
            f"(generation 0: {trajectory[0]:+.5f}); "
            f"best tree length {length} nodes"
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evogp.evox_ext.demo_stdworkflow",
        description=(
            "Run the torch-free EvoGP GP stack inside an evox 2.0 StdWorkflow "
            "(symbolic regression on x0^2 + x1^2)."
        ),
    )
    parser.add_argument(
        "--pop", type=int, default=DEFAULT_POP_SIZE,
        help="population size (default %(default)s)",
    )
    parser.add_argument(
        "--generations", type=int, default=DEFAULT_GENERATIONS,
        help="evaluated generations (default %(default)s)",
    )
    parser.add_argument(
        "--data", type=int, default=DEFAULT_NUM_DATA,
        help="data rows (default %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed of the workflow and of the data (default %(default)s)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=DEFAULT_MAX_NODES,
        help="tree-buffer width (default %(default)s)",
    )
    parser.add_argument(
        "--backend", default="numpy",
        help="evoxc backend: numpy (default) or iree",
    )
    parser.add_argument(
        "--device", default=None,
        help="evoxc device, e.g. rocm:0 or cuda:0 (default: the CPU)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="print only the summary line",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m evogp.evox_ext.demo_stdworkflow`` -- parse the flags and run."""
    args = _build_parser().parse_args(argv)
    result = run_demo(
        pop_size=args.pop,
        generations=args.generations,
        num_data=args.data,
        seed=args.seed,
        max_nodes=args.max_nodes,
        backend=args.backend,
        device=args.device,
        verbose=not args.quiet,
    )
    if args.quiet:
        print(
            f"best fitness {result['best_fitness']:+.5f} at generation "
            f"{result['best_generation']}; "
            f"{1000.0 * result['step_seconds']:.2f} ms/generation"
        )
    return 0 if result["trajectory"] else 1


if __name__ == "__main__":
    sys.exit(main())
