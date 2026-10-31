"""CPU smoke run of the evoxc GP stack: ``python -m evogp.gp.smoke``.

A real, converging end-to-end run of the driver on a CPU-only box -- the
counterpart of ``../../evogp/sr_test.py`` for the torch-free stack, and
deliberately its opposite in one respect: this module does **nothing** at import
time.  The reference workload is the same synthetic 2-input target
``x0**4 / (x0**4 + 1) + x1**4 / (x1**4 + 1)`` on 1 000 points drawn uniformly
from ``[-5, 5]`` (the target ``sr_test.py`` fits), so the two fitness numbers are
comparable.

Everything is small on purpose: the numpy evoxc backend compiles in
milliseconds and the numpy tree evaluator walks 200 trees over 1 000 rows per
generation, so the shipped defaults (200 trees, 100 generations, 64-slot trees)
run in a few seconds and the fitness improves all the way through.  Use the
flags (or :func:`run_smoke`) to scale it.

Wiring, for the record -- the torch reference (``sr_test.py``) uses
``DefaultCrossover`` + ``DefaultMutation(mutation_rate=0.2)`` +
``DefaultSelection(survival_rate=0.3, elite_rate=0.01)``.  This stack has no
truncation selector, so the closest pressure analogue is a
``tournament_size=4`` tournament, with the same 0.9 / 0.2 operator rates and
``elite_count = 2`` (1 % of 200) elites.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import GPConfig, GPTreeConfig
from .problem import SymbolicRegression
from .workflow import GPResult, run

__all__ = [
    "target_function",
    "make_data",
    "make_configs",
    "run_smoke",
    "main",
    "DEFAULT_POP_SIZE",
    "DEFAULT_GENERATIONS",
    "DEFAULT_NUM_DATA",
]

#: Population size of the shipped smoke run.
DEFAULT_POP_SIZE = 200
#: Number of evaluated generations of the shipped smoke run.
DEFAULT_GENERATIONS = 100
#: Data rows of the synthetic problem (the reference script's ``num_data``).
DEFAULT_NUM_DATA = 1000
#: RNG seed -- the driver's key, and the data seed unless overridden.
DEFAULT_SEED = 0
#: Tree-buffer width (``max_tree_len`` in the torch stack).
DEFAULT_MAX_NODES = 64
#: Tournament size (the torch stack's ``survival_rate=0.3`` analogue).
DEFAULT_TOURNAMENT_SIZE = 4
#: Per-individual crossover / mutation rates (the torch stack's defaults).
DEFAULT_CROSSOVER_PROB = 0.9
DEFAULT_MUTATION_PROB = 0.2
#: Number of elites carried into every generation (1 % of 200, as in ``sr_test``).
DEFAULT_ELITE_COUNT = 2
#: The function set and constant table of the reference descriptor.
USING_FUNCS = ("+", "-", "*", "/")
CONST_RANGE = (-5.0, 5.0)
SAMPLE_CNT = 8
MAX_LAYER_CNT = 5
LAYER_LEAF_PROB = 0.2
CONST_PROB = 0.5
#: Bounds of the uniform data draw.
DATA_BOUNDS = (-5.0, 5.0)


def target_function(x) -> np.ndarray:
    """``x0**4/(x0**4+1) + x1**4/(x1**4+1)`` for ``x`` of shape ``[2, n_rows]``."""
    values = np.asarray(x, dtype=np.float64)
    return (
        values[0] ** 4 / (values[0] ** 4 + 1) + values[1] ** 4 / (values[1] ** 4 + 1)
    ).reshape(-1)


def make_data(
    num_data: int = DEFAULT_NUM_DATA, seed: int = DEFAULT_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """The synthetic SR problem: ``(x [num_data, 2], y [num_data])`` float32."""
    if not isinstance(num_data, int) or num_data <= 0:
        raise ValueError(f"num_data must be a positive int, got {num_data!r}")
    generator = np.random.default_rng(seed)
    x = generator.uniform(
        DATA_BOUNDS[0], DATA_BOUNDS[1], size=(int(num_data), 2)
    ).astype(np.float32)
    y = target_function(x.T).astype(np.float32)
    return x, y


def make_configs(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    elite_count: int = DEFAULT_ELITE_COUNT,
    crossover_prob: float = DEFAULT_CROSSOVER_PROB,
    mutation_prob: float = DEFAULT_MUTATION_PROB,
) -> Tuple[GPTreeConfig, GPConfig]:
    """The ``(GPTreeConfig, GPConfig)`` pair of the smoke run."""
    tree_config = GPTreeConfig(
        max_nodes=int(max_nodes),
        input_len=2,
        using_funcs=USING_FUNCS,
        max_layer_cnt=MAX_LAYER_CNT,
        layer_leaf_prob=LAYER_LEAF_PROB,
        const_prob=CONST_PROB,
        const_range=CONST_RANGE,
        sample_cnt=SAMPLE_CNT,
    )
    gp_config = GPConfig(
        pop_size=int(pop_size),
        elite_count=int(elite_count),
        tournament_size=int(tournament_size),
        crossover_prob=float(crossover_prob),
        mutation_prob=float(mutation_prob),
        mutation_kind="subtree",
    )
    return tree_config, gp_config


def run_smoke(
    pop_size: int = DEFAULT_POP_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    num_data: int = DEFAULT_NUM_DATA,
    seed: int = DEFAULT_SEED,
    max_nodes: int = DEFAULT_MAX_NODES,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    verbose: bool = True,
) -> GPResult:
    """Run the smoke problem and return the driver's :class:`GPResult`.

    Args:
        pop_size: number of trees.
        generations: number of evaluated generations.
        num_data: data rows of the synthetic problem.
        seed: the driver's RNG seed (also the data seed).
        max_nodes: tree-buffer width.
        tournament_size: tournament pressure.
        verbose: print the best fitness of every generation.

    Returns:
        The :class:`~evogp.gp.workflow.GPResult` of the run.
    """
    x, y = make_data(num_data, seed=seed)
    problem = SymbolicRegression(x, y)
    config = make_configs(
        pop_size=pop_size, max_nodes=max_nodes, tournament_size=tournament_size
    )
    started = time.time()

    def report(generation, _state, fitness) -> None:
        if not verbose:
            return
        finite = fitness[np.isfinite(fitness)]
        median = float(np.median(finite)) if finite.size else float("-inf")
        print(
            f"generation {generation:3d}  best {float(np.max(fitness)):+12.5f}  "
            f"median {median:+12.5f}  {time.time() - started:6.1f}s",
            flush=True,
        )

    result = run(
        config,
        problem.evaluate,
        generations=generations,
        key=seed,
        callback=report,
    )
    if verbose:
        wall = time.time() - started
        print(
            f"\n{generations} generations in {wall:.2f}s "
            f"({1000.0 * wall / max(generations, 1):.1f} ms/generation)"
        )
        if result.best_fitness_history:
            print(
                f"best fitness {result.best_fitness:.5f} at generation "
                f"{result.best_generation} (generation 0: "
                f"{result.best_fitness_history[0]:.5f})"
            )
        else:
            print(f"best fitness {result.best_fitness:.5f} (nothing evaluated)")
        if result.best_channels is not None:
            print(f"best tree length: {int(result.best_channels[2][0])} nodes")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evogp.gp.smoke",
        description=(
            "Converging CPU smoke run of the torch-free evoxc GP stack "
            "(symbolic regression on x0^4/(x0^4+1) + x1^4/(x1^4+1))."
        ),
    )
    parser.add_argument("--pop", type=int, default=DEFAULT_POP_SIZE,
                        help="population size (default %(default)s)")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="evaluated generations (default %(default)s)")
    parser.add_argument("--data", type=int, default=DEFAULT_NUM_DATA,
                        help="data rows (default %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed of the driver and of the data "
                             "(default %(default)s)")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES,
                        help="tree-buffer width (default %(default)s)")
    parser.add_argument("--tournament-size", type=int,
                        default=DEFAULT_TOURNAMENT_SIZE,
                        help="tournament pressure (default %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing (read the returned GPResult instead)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m evogp.gp.smoke`` -- parse the flags and run the smoke."""
    args = _build_parser().parse_args(argv)
    result = run_smoke(
        pop_size=args.pop,
        generations=args.generations,
        num_data=args.data,
        seed=args.seed,
        max_nodes=args.max_nodes,
        tournament_size=args.tournament_size,
        verbose=not args.quiet,
    )
    return 0 if result.best_channels is not None else 1


if __name__ == "__main__":
    sys.exit(main())
