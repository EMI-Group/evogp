"""The ``CustomLoss`` escape hatch of the torch-free ``evogp.gp`` stack.

Shows the one plug-in surface of the functional stack that is not a mean squared
error: :class:`evogp.gp.CustomLoss` takes a plain numpy callable
``loss(predictions, targets)``, calls it ONCE per generation over the WHOLE
population (``predictions`` is ``[pop_size, n_rows]`` -- every tree's every row,
not one tree's) and negates its per-individual return, so ``evaluate`` hands
back a MAXIMIZED ``[pop_size]`` float32 array.

The task is a classification-style table on ``[-1, 1]^2``: label ``1.0`` where
``x0^2 > x1^2`` (the sign XOR of the square's two diagonals), scored as the
accuracy of a single-output tree thresholded at ``0`` -- an objective no
``mean((pred - y)^2)`` can express.

Run from the repository root, on the numpy host backend (no torch, no device)::

    PYTHONPATH=src:/root/evoxc python example/gp_custom_loss.py

Expected: ~1 s (pop 200, 40 generations, 512 rows), the fitness contract, the
per-generation best and median accuracy, and the winner's accuracy recomputed on
the host -- matching ``result.best_fitness`` with a delta of 0.0.  The run is
key-based deterministic, so a second run prints identical numbers (timings aside).
"""

from __future__ import annotations

import os
# Tracing dominates a compile of this size, and the switch is a pure speed knob
# (it is not part of any cache key). Set before anything traces.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from typing import Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from evogp.gp import (  # noqa: E402
    CustomLoss,
    GPConfig,
    GPTreeConfig,
    init_gp,
    row_length,
    run,
)

#: RNG seed (the driver's key and the data seed), run size, and tree width.
SEED, POP_SIZE, GENERATIONS, NUM_ROWS, MAX_NODES = 0, 200, 40, 512, 64
#: Bounds of the uniform data draw, and the class threshold on a prediction.
DATA_BOUNDS, THRESHOLD = (-1.0, 1.0), 0.0


def make_data(rows: int = NUM_ROWS, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """The table: ``(x [rows, 2], y [rows])`` float32, labels in ``{0, 1}``."""
    generator = np.random.default_rng(seed)
    x = generator.uniform(*DATA_BOUNDS, size=(int(rows), 2)).astype(np.float32)
    return x, np.where(x[:, 0] ** 2 > x[:, 1] ** 2, 1.0, 0.0).astype(np.float32)


def accuracy_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-individual loss ``-accuracy``, so the negated fitness IS the accuracy.

    ``CustomLoss`` calls this ONCE per generation with ``predictions`` of shape
    ``[pop_size, n_rows]`` -- ``pop_size * n_rows`` values at once -- and
    ``targets`` of shape ``[n_rows]``, and it must return ``pop_size`` entries
    (here ``[pop_size]``).  The problem negates them, so this returns the
    quantity to MINIMIZE: ``-accuracy`` for a fitness equal to the accuracy, or
    the positive rate ``1 - accuracy`` (:func:`misclassification_rate`) for a
    fitness of ``accuracy - 1``; the two rank the population identically.  A NaN
    prediction compares ``False`` and counts as a miss, so a broken tree scores 0
    instead of poisoning the rest.
    """
    correct = (predictions > THRESHOLD) == (targets > 0.5)[None, :]
    return (-np.mean(correct, axis=1)).astype(np.float32)


def misclassification_rate(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The same score as a strictly positive loss: ``1 - accuracy``."""
    return (1.0 + accuracy_loss(predictions, targets)).astype(np.float32)


def make_configs() -> Tuple[GPTreeConfig, GPConfig]:
    """The ``(GPTreeConfig, GPConfig)`` pair: ``+ - * /``, 2 inputs, 1 output."""
    tree = GPTreeConfig(
        max_nodes=MAX_NODES, input_len=2, using_funcs=("+", "-", "*", "/"),
        output_len=1,  # the only width evogp.gp can evaluate today
        max_layer_cnt=4, layer_leaf_prob=0.2, const_prob=0.5,
        const_range=(-1.0, 1.0), sample_cnt=8,
    )
    gp = GPConfig(
        pop_size=POP_SIZE, elite_count=2, tournament_size=4,
        crossover_prob=0.9, mutation_prob=0.2, mutation_kind="subtree",
    )
    return tree, gp


def host_accuracy(problem: CustomLoss, channels, y: np.ndarray) -> Tuple[float, int]:
    """Accuracy and misclassified count of ONE tree (a population of one)."""
    predictions = problem.predict(channels)[0]
    correct = int(np.count_nonzero((predictions > THRESHOLD) == (y > 0.5)))
    return correct / float(y.shape[0]), int(y.shape[0]) - correct


def head(values: np.ndarray, count: int = 4) -> str:
    """The first ``count`` entries of a vector, as a short repr."""
    return np.array2string(values[:count], precision=4)


def report_contract(
    problem: CustomLoss, config: Tuple[GPTreeConfig, GPConfig], x: np.ndarray
) -> np.ndarray:
    """Print the fitness contract on the generation-0 population."""
    initial = init_gp(config, SEED)
    fitness, loss = problem.evaluate(initial), problem.loss(initial)
    print("CustomLoss.evaluate(initial_state) -- the fitness contract:")
    print(f"  type {type(fitness).__name__}, shape {fitness.shape}, "
          f"dtype {fitness.dtype}")
    print(f"  MAXIMIZED fitness (= accuracy)     {head(fitness)} ...")
    print(f"  what the callable returned (loss)  {head(loss)} ...")
    print(f"  fitness == -loss everywhere: {np.array_equal(fitness, -loss)}; "
          f"range [{fitness.min():.4f}, {fitness.max():.4f}]")
    print(f"  ONE call saw {fitness.shape[0]} x {x.shape[0]} = "
          f"{fitness.shape[0] * x.shape[0]} prediction values")
    positive = CustomLoss(x, misclassification_rate, problem.targets)
    print(f"  a POSITIVE loss (1 - accuracy) there: "
          f"{head(positive.loss(initial))} ... -> fitness "
          f"{head(positive.evaluate(initial))} ...")
    print("  (same ranking: its fitness is ours minus 1.0, the constant that "
          "1 - accuracy adds)\n")
    return fitness


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evolution and print the contract, the history and the winner."""
    del argv  # no flags: the knobs are the module constants above
    x, y = make_data()
    problem = CustomLoss(x, accuracy_loss, y)
    config = make_configs()
    initial_fitness = report_contract(problem, config, x)
    started = time.time()

    def report(generation: int, _state, fitness: np.ndarray) -> None:
        finite = fitness[np.isfinite(fitness)]
        median = float(np.median(finite)) if finite.size else float("-inf")
        print(f"generation {generation:3d}  best accuracy {np.max(fitness):.4f}  "
              f"median {median:.4f}  {time.time() - started:5.1f}s", flush=True)

    result = run(config, problem.evaluate, generations=GENERATIONS, key=SEED,
                 callback=report)
    wall, history = time.time() - started, result.best_fitness_history
    print(f"\n{GENERATIONS} generations in {wall:.2f}s "
          f"({1000.0 * wall / GENERATIONS:.1f} ms/generation)")
    print(f"result.best_fitness      {result.best_fitness:.5f}  (= best accuracy)")
    print(f"result.best_generation   {result.best_generation}")
    print(f"best_fitness_history     [0] {history[0]:.5f} -> [-1] {history[-1]:.5f}")
    print(f"generation-0 cross-check history[0] == max(initial fitness): "
          f"{history[0] == float(np.max(initial_fitness))}")

    accuracy, wrong = host_accuracy(problem, result.best_channels, y)
    opcode, payload, size = result.best_channels
    nodes, shown = int(row_length(result.best_channels)[0]), 12
    print(f"\nwinning tree: {nodes}/{size.shape[0]} nodes; opcodes "
          f"{opcode[:shown].tolist()}{' ...' if nodes > shown else ''}")
    print(f"              payloads {np.round(payload[:shown], 3).tolist()}"
          f"{' ...' if nodes > shown else ''}")
    # evogp.gp has no expression exporter yet (MIGRATION step 9), so the channel
    # rows above stand in for a sympy/infix rendering.
    print(f"host accuracy (pred > 0) {accuracy:.5f}  "
          f"({wrong}/{y.shape[0]} misclassified)")
    delta = abs(accuracy - result.best_fitness)
    print(f"|host accuracy - best_fitness| = {delta:.2e}")
    return 0 if delta <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- #
# When CustomLoss is the right tool
# --------------------------------------------------------------------------- #
# Reach for it whenever the fitness is NOT `mean((pred - y)**2)`: classification
# accuracy (this file, where a squared error ranks trees differently from a
# thresholded metric); a robust loss (Huber / log-cosh) that must not blow up on
# the exploding trees every GP population contains; correlation or ranking
# objectives (Pearson / Spearman, AUC) over the rows; a combination of several
# predicted columns -- the reason the callable gets the whole `[pop_size, n_rows]`
# block at once; or any objective carrying a penalty term (node count, sparsity,
# a physics residual) that is easier to write in numpy against the predictions.
# What it costs: CustomLoss always evaluates through the numpy host evaluator
# (`predict`), so it gives up the in-graph / fused device routes, which serve the
# mean-squared-error surfaces only (`SymbolicRegression`, `sr_fitness_graph`), and
# it expects a single-output tree (`output_len > 1` raises NotImplementedError),
# so a multi-column objective needs several single-output evaluations or a
# thresholded/derived scalar, as done here.
