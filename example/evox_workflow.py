"""Run the torch-free ``evogp.gp`` GP stack inside an ``evox`` 2.0 ``StdWorkflow``.

The ``example/`` counterpart of ``src/evogp/evox_ext/demo_stdworkflow.py``: the
GP algorithm and the symbolic-regression problem are handed to EvoX 2.0 as
*components* (a frozen config dataclass plus module-level ``init`` / ``step`` /
``evaluate``), and :class:`evox.core.StdWorkflow` composes a whole generation --
evaluation included -- into ONE compiled ``evoxc`` graph.  Nothing here touches
torch, triton or a device: the default run is ``evoxc``'s numpy backend on a CPU
box, on the target ``x0**2 + x1**2`` over ``[-5, 5]`` (reachable with the
shipped ``+ - * /`` set, so a short run converges instead of only drifting).

Run it from the repository root.  This is the ONLY example that needs the
``evox`` checkout (``evox`` 2.0.0a1 is a PyPI pre-release, and no ``evogp``
module imports it outside ``evogp.evox_ext``); note ``/root/evox/src``, since
the checkout root itself does not put ``evox`` on the path::

    PYTHONPATH=src:/root/evoxc:/root/evox/src python example/evox_workflow.py

``--backend iree --device rocm:0`` selects the compiled device route (the IREE
compile then moves into ``init()``).  Expected: about 2 s on a CPU box for pop
100 x 20 generations over 256 rows -- ≈0.7 s of it the one-time graph build in
``init()``, then ≈35 ms per warm generation, best fitness climbing from about
-116 to an exact 0.00 fit around generation 12.  The closing block re-runs the
same problem through the plain driver ``evogp.gp.run`` at the same pop and
generations.  ``evogp.gp`` has no expression exporter yet (``MIGRATION.md`` step
9), so the winner is reported by node count only.  ``RuntimeWarning: divide by
zero`` stderr lines are the trees' own ``/`` nodes on the numpy backend: a
non-finite evaluation becomes the ``-inf`` fitness marker, so they are expected.
"""

from __future__ import annotations

import os

# evoxc records a call-site location per traced op; dropping that capture is a
# pure tracing-speed knob (~4x faster traces) and is not part of any compiled
# program or cache key.  Set it before anything could trace.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from evogp.evox_ext import (  # noqa: E402
    best_row,
    generation_fitness,
    host_channels,
    make_gp_algorithm,
    make_sr_problem,
    make_stdworkflow,
)
from evogp.gp import (  # noqa: E402
    GPConfig,
    GPTreeConfig,
    SymbolicRegression,
    row_length,
    run,
)

SEED = 0  # the workflow's seed; also the driver's key and the data seed
POP_SIZE = 100  # trees per generation
GENERATIONS = 20  # evaluated generations (steps), identical for both runs
NUM_DATA = 256  # data rows of the synthetic problem
INPUT_LEN = 2  # always ``x0``, ``x1``
MAX_NODES = 32  # tree-buffer width (the torch stack's ``max_tree_len``)
USING_FUNCS = ("+", "-", "*", "/")
DATA_BOUNDS = (-5.0, 5.0)


def make_data(
    num_data: int = NUM_DATA, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """The synthetic problem: ``(x [num_data, 2], y [num_data])`` float32."""
    generator = np.random.default_rng(seed)
    low, high = DATA_BOUNDS
    x = generator.uniform(low, high, size=(int(num_data), INPUT_LEN))
    y = x[:, 0] ** 2 + x[:, 1] ** 2
    return x.astype(np.float32), y.astype(np.float32)


def make_configs() -> tuple[GPTreeConfig, GPConfig]:
    """The ``(GPTreeConfig, GPConfig)`` pair both runs share (static leaves)."""
    tree_config = GPTreeConfig(
        max_nodes=MAX_NODES,
        input_len=INPUT_LEN,
        using_funcs=USING_FUNCS,
        max_layer_cnt=5,  # a complete binary tree then needs 31 <= MAX_NODES
        layer_leaf_prob=0.2,
        const_prob=0.5,  # half the leaves are constants ...
        const_samples=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),  # ... from here
    )
    gp_config = GPConfig(
        pop_size=POP_SIZE,
        elite_count=2,  # ~1 %, the torch stack's elite_rate=0.01
        tournament_size=4,  # the pressure analogue of survival_rate=0.3
        crossover_prob=0.9,
        mutation_prob=0.2,
        mutation_kind="subtree",
    )
    return tree_config, gp_config


def run_workflow(
    x: np.ndarray,
    y: np.ndarray,
    configs: tuple[GPTreeConfig, GPConfig],
    backend: str = "numpy",
    device: Optional[str] = None,
) -> dict:
    """Wrap the GP stack in a ``StdWorkflow``, step it, and report what it wrote."""
    tree_config, gp_config = configs
    started = time.perf_counter()
    # ``opt_direction="max"`` is fixed by the adapter: evogp.gp fitness is
    # MAXIMIZED, and StdWorkflow would hand its algorithm a scaled vector.
    workflow = make_stdworkflow(
        make_gp_algorithm(tree_config, gp_config),
        make_sr_problem(x, y, used_ids=tree_config),
        backend=backend,
        device=device,
    )
    # ``init()`` builds the generation-0 population graph AND the step graph, so
    # the compile of a whole generation (evaluation + operators) is paid HERE,
    # once; every ``step`` below is one graph run.
    state = workflow.init(seed=SEED)
    init_seconds = time.perf_counter() - started
    print(
        f"init() built both graphs in {init_seconds:.2f}s "
        f"({backend}{'/' + device if device else ''}); every step is one graph run"
    )

    # ONE host readback per generation: the elite values the step graph wrote.
    # The offspring rows carry the ``-inf`` "not evaluated yet" mark, so step
    # ``k`` evaluates generation ``k - 1`` and splices its elites into
    # generation ``k`` -- what the readback below reports, elite values only.
    # That is the intended reporting contract, not the workflow's default loop.
    trajectory: list[float] = []
    step_total = 0.0
    for step in range(1, GENERATIONS + 1):
        step_started = time.perf_counter()
        state = workflow.step(state)
        step_total += time.perf_counter() - step_started
        evaluated = generation_fitness(state)
        trajectory.append(float(np.max(evaluated)))  # the elite rows are finite
        print(
            f"generation {step - 1:3d}  best {trajectory[-1]:+12.5f}  "
            f"elite {int(np.isfinite(evaluated).sum()):3d}  "
            f"{time.perf_counter() - started:6.2f}s",
            flush=True,
        )

    winner = best_row(state)
    lengths = row_length(host_channels(state))  # one node count per row
    return {
        "trajectory": trajectory,
        "best_fitness": max(trajectory),
        "best_generation": int(np.argmax(trajectory)),
        "winner": winner,
        "nodes": int(lengths[winner]),
        "init_seconds": init_seconds,
        "ms_per_generation": 1000.0 * step_total / GENERATIONS,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run both loops and print the comparison (``__main__`` does nothing else)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="numpy", help="numpy (default) or iree")
    parser.add_argument("--device", default=None, help="iree device, e.g. rocm:0")
    args = parser.parse_args(argv)

    x, y = make_data()
    configs = make_configs()
    print(
        f"pop {POP_SIZE} x {GENERATIONS} generations over {NUM_DATA} rows, "
        f"max_nodes {MAX_NODES}, evogp.gp inside an evox 2.0 StdWorkflow"
    )
    result = run_workflow(x, y, configs, args.backend, args.device)
    print(
        f"\n{GENERATIONS} steps: {result['ms_per_generation']:.2f} ms/generation "
        f"warm, {result['init_seconds']:.2f}s of the run the one-time graph build "
        f"in init(); best fitness {result['best_fitness']:+.5f} at generation "
        f"{result['best_generation']} (generation 0: {result['trajectory'][0]:+.5f})"
    )
    print(
        f"winning row {result['winner']}: {result['nodes']} nodes "
        "(no expression exporter in evogp.gp yet)"
    )

    started = time.perf_counter()  # the same problem through the plain driver
    driver = run(
        configs, SymbolicRegression(x, y).evaluate, generations=GENERATIONS, key=SEED
    )
    driver_ms = 1000.0 * (time.perf_counter() - started) / GENERATIONS
    print(
        f"\nplain driver (evogp.gp.run, same pop/generations/seed): "
        f"{driver_ms:.2f} ms/generation, best fitness {driver.best_fitness:+.5f} "
        f"at generation {driver.best_generation}"
    )
    if abs(driver.best_fitness - result["best_fitness"]) > 1e-6:
        print(
            f"the two trajectories differ (workflow {result['best_fitness']:+.5f} "
            f"vs driver {driver.best_fitness:+.5f}): different RNG plumbing -- the "
            "workflow draws from its own key, the driver from key=SEED, while the "
            "problem, pop, generations and evaluator are the same"
        )
    else:
        print("the two trajectories agree to within 1e-6 on the best fitness")
    return 0 if np.isfinite(result["best_fitness"]) else 1


if __name__ == "__main__":
    sys.exit(main())
