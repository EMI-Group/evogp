"""Symbolic regression on a REAL dataset through the torch-free ``evogp.gp``.

The real-data sibling of ``gp_sr.py``: same wiring (``GPTreeConfig`` +
``GPConfig`` -> :func:`evogp.gp.run` with the problem's ``evaluate``), same
key-based determinism, but the data is scikit-learn's BUNDLED ``load_diabetes``
table (442 rows x 10 numeric features, continuous target) instead of a synthetic
function.  This is the offline stand-in for the old ``uci_sr.py`` paper script,
which needed ``ucimlrepo`` + network access + ``pandas``; this loader ships
inside scikit-learn, a hard requirement of ``setup.py``, so there is **no
network, no new dependency** and nothing to download.

Run it from the repository root (``evoxc`` is imported at call time only)::

    PYTHONPATH=src:/root/evoxc python example/gp_uci_sr.py

Expected: ~20 s on a CPU box (pop 300 x 300 generations over 442 rows, numpy
host backend): the per-generation best fitness every 5th generation, then the
winner's node count, its RMSE / MAE / R^2 in the ORIGINAL target units
(``disease progression``) and two baselines on the same features -- the training
mean (R^2 = 0 by construction) and a ``numpy.linalg.lstsq`` linear fit -- so the
reader can judge whether the tree is worth its nodes.  Two runs print identical
fitness numbers (``key=0`` carries the whole RNG stream in the state).

``evogp.gp`` still has no expression exporter (MIGRATION step 9 is open), so this
file prints a compact opcode summary instead of an infix string; ``gp_sr.py``
carries the local stand-in renderer.
"""

from __future__ import annotations

import os
# A pure tracing-speed knob (~4x faster traces), never part of a compiled
# program or a cache key.  Set before anything could trace.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from typing import Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.datasets import load_diabetes  # noqa: E402

from evogp.gp import (  # noqa: E402
    GPConfig,
    GPTreeConfig,
    SymbolicRegression,
    row_length,
    run,
)

#: The driver's key (whole RNG stream, no global seed) and the run size.
SEED, POP_SIZE, GENERATIONS = 0, 300, 300
#: Tree-buffer width (``max_tree_len`` of the old torch stack) and depth schedule.
MAX_NODES, MAX_LAYER_CNT, LAYER_LEAF_PROB = 128, 6, 0.2
#: Half the leaves are constants, drawn over ``SAMPLE_CNT`` values of the range.
CONST_PROB, CONST_RANGE, SAMPLE_CNT = 0.5, (-2.0, 2.0), 16
#: Operator rates (the torch stack's defaults) and the report cadence.
ELITE_COUNT, TOURNAMENT_SIZE, CROSSOVER_PROB, MUTATION_PROB = 3, 4, 0.9, 0.2
REPORT_EVERY, USING_FUNCS = 5, ("+", "-", "*", "/")


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """The bundled table as ``(x [442, 10], y [442])`` float32, original units."""
    bunch = load_diabetes()
    return (
        np.asarray(bunch.data, dtype=np.float32),
        np.asarray(bunch.target, dtype=np.float32),
    )


def standardize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score a matrix column-wise (or a vector): ``(z, mean, std)``."""
    mean = np.asarray(values.mean(axis=0, dtype=np.float64))
    std = np.asarray(values.std(axis=0, dtype=np.float64))
    std = np.where(std == 0.0, 1.0, std)  # a constant column is otherwise 0/0
    return ((values - mean) / std).astype(np.float32), mean, std


def scores(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """``(rmse, mae, r2)`` of ``predictions`` against ``targets``, in float64."""
    residual = np.asarray(predictions, dtype=np.float64) - targets
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    return (
        float(np.sqrt(np.mean(residual * residual))),
        float(np.mean(np.abs(residual))),
        1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("-inf"),
    )


def linear_baseline(x: np.ndarray, y: np.ndarray) -> Tuple[str, float, float, float]:
    """``numpy.linalg.lstsq`` fit of ``y ~ [1, x]``, scored in ORIGINAL units."""
    design = np.column_stack([np.ones(x.shape[0]), x.astype(np.float64)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return ("lstsq linear", *scores(design @ coefficients, y))


def opcode_summary(channels, shown: int = 10) -> str:
    """A compact stand-in for an infix print: the first ``shown`` opcodes."""
    opcode, payload, _size = (np.asarray(part) for part in channels)
    return (
        f"opcodes {opcode[:shown].tolist()}{' ...' if opcode.shape[0] > shown else ''}"
        f"  constants {np.round(payload[:shown], 3).tolist()}"
    )


def make_configs(input_len: int) -> Tuple[GPTreeConfig, GPConfig]:
    """The ``(GPTreeConfig, GPConfig)`` pair: ``+ - * /``, ``input_len`` inputs."""
    tree = GPTreeConfig(
        max_nodes=MAX_NODES, input_len=int(input_len), using_funcs=USING_FUNCS,
        output_len=1,  # the only width evogp.gp can evaluate today
        max_layer_cnt=MAX_LAYER_CNT, layer_leaf_prob=LAYER_LEAF_PROB,
        const_prob=CONST_PROB, const_range=CONST_RANGE, sample_cnt=SAMPLE_CNT,
    )
    gp = GPConfig(
        pop_size=POP_SIZE, elite_count=ELITE_COUNT,
        tournament_size=TOURNAMENT_SIZE, crossover_prob=CROSSOVER_PROB,
        mutation_prob=MUTATION_PROB, mutation_kind="subtree",
    )
    return tree, gp


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evolution, then print the winner and the baseline comparison."""
    del argv  # no flags: the knobs are the module constants above
    x_raw, y_raw = load_data()
    x, _x_mean, _x_std = standardize(x_raw)
    y, y_mean, y_std = standardize(y_raw)
    print(f"load_diabetes: {x_raw.shape[0]} rows, {x_raw.shape[1]} features "
          f"(scikit-learn's bundled copy -- no network, no extra dependency)")
    print(f"z-scored inputs and target on the host (mean 0 / std 1); the target's "
          f"original units are mean {float(y_mean):.1f} / std {float(y_std):.1f}, "
          f"inverted for every metric below")

    problem = SymbolicRegression(x, y)  # fitness = -MSE in STANDARDISED units
    config = make_configs(x.shape[1])
    started = time.perf_counter()

    def report(generation: int, _state, fitness: np.ndarray) -> None:
        """Print this generation's best fitness on the report cadence."""
        if generation % REPORT_EVERY and generation != GENERATIONS - 1:
            return
        print(f"generation {generation:3d}  best {float(np.max(fitness)):+12.5f}  "
              f"{time.perf_counter() - started:6.2f}s", flush=True)

    result = run(config, problem.evaluate, generations=GENERATIONS, key=SEED,
                 callback=report)
    wall, history = time.perf_counter() - started, result.best_fitness_history
    print(f"\npop {POP_SIZE} x {GENERATIONS} generations over {x.shape[0]} rows in "
          f"{wall:.2f}s ({1000.0 * wall / GENERATIONS:.1f} ms/generation)")
    print(f"result.best_fitness   {result.best_fitness:+.6f}  (= -MSE, standardised "
          f"units) at generation {result.best_generation} of {len(history)}")
    print(f"best_fitness_history  [0] {history[0]:+.6f} -> [-1] {history[-1]:+.6f}")
    if result.best_channels is None:
        print("no finite individual was ever evaluated")
        return 1

    nodes = int(row_length(result.best_channels)[0])
    print(f"winning tree          {nodes} of {MAX_NODES} nodes")
    print(f"                      {opcode_summary(result.best_channels)}")
    print("  (evogp.gp has no expression exporter yet -- MIGRATION step 9 -- so "
          "the opcode rows stand in for infix)")

    prediction = problem.predict(result.best_channels)[0]  # [442], standardised
    tree = ("gp tree", *scores(prediction * y_std + y_mean, y_raw))
    linear = linear_baseline(x_raw, y_raw)
    mean_only = ("train mean", *scores(np.full_like(y_raw, y_raw.mean()), y_raw))
    print(f"\noriginal target units ({y_raw.shape[0]} rows, predict-then-invert):")
    for label, rmse, mae, r2 in (tree, linear, mean_only):
        print(f"  {label:<12} rmse {rmse:8.3f}   mae {mae:8.3f}   r2 {r2:+.4f}")
    verdict = "below" if tree[3] < linear[3] else "above"
    print(f"-> the evolved tree lands {verdict} the linear baseline on R^2 "
          f"({tree[3] - linear[3]:+.4f}); raising pop/generations is the knob")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- #
# Takeaway
# --------------------------------------------------------------------------- #
# The fitness is `-MSE` in standardised units, so `result.best_fitness` is
# `-(rmse_standardised ** 2)` and every other number here is a host-side readback.
# On 442 real rows the evolved tree lands within a couple of hundredths of R^2 of
# the 11-coefficient linear fit -- just above it at the shipped settings -- so on
# a table this small GP buys a closed-form model (once the exporter lands), not
# an accuracy win.  Raise POP_SIZE / GENERATIONS (or widen CONST_RANGE) to trade
# wall time for fitness; the per-generation trace above shows where it plateaus.
