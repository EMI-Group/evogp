"""Symbolic regression end-to-end on the torch-free ``evogp.gp`` stack.

The canonical "how a run is wired" example of the functional stack: build a
synthetic 2-input regression problem, describe the tree population and the
operators with :class:`~evogp.gp.GPTreeConfig` / :class:`~evogp.gp.GPConfig`,
hand the problem's ``evaluate`` to :func:`evogp.gp.run`, and read the winner off
the returned :class:`~evogp.gp.GPResult`.  Nothing here touches torch, triton or
a device: the stack is ``evoxc`` + numpy and runs on a CPU box.

The target is ``x0**2 + x1**2`` on ``[-5, 5]``: expressible with the shipped
``+ - * /`` set and a constant table, so the run converges onto it instead of
merely improving, which keeps the reported expression short and checkable.  (The
harder reference target ``x0**4/(x0**4+1) + x1**4/(x1**4+1)`` is what
``python -m evogp.gp.smoke`` fits.)

Run it from the repository root (``evoxc`` must be on the path -- it is imported
at call time, never at module scope)::

    PYTHONPATH=src:/root/evoxc python example/gp_sr.py

Expected: about 2 s on a CPU box; the printed per-generation best fitness climbs
from about -127 to 0.00 (the fitness is the negated mean squared error, so 0.00
is an exact fit) and the run settles on it around generation 33, then the
winner's node count, its infix expression, its RMSE against the data and the
first three prediction/target pairs.  ``key=0`` is the whole determinism
contract -- the RNG stream lives in the state (:func:`evogp.gp.run`, no global
seed and no torch), so two runs print identical fitness numbers.

``evogp.gp`` has no public ``to_sympy_expr`` yet (step 9 of ``MIGRATION.md`` is
still open: the torch stack's exporter went away with its stack), so the local
decoder in this file -- :func:`render_expression` -- is the stand-in.
"""

from __future__ import annotations

import os

# evoxc records a call-site location per traced op; dropping that capture is a
# pure tracing-speed knob (~4x faster traces) and is not part of any compiled
# program or cache key.  Set it before anything could trace.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import sys  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402

from evogp.gp import (  # noqa: E402
    GPConfig,
    GPTreeConfig,
    SymbolicRegression,
    row_length,
    run,
)
# The tree layer's own id/name tables.  This module is private -- which is
# exactly the missing exporter above: an example may read it as a stand-in, the
# library must not grow an infix printer as a side effect of it.
from evogp.gp._constants import FUNC_ARITY, FUNCS_NAMES, Func, NType  # noqa: E402

# --------------------------------------------------------------------------- #
# the knobs of the shipped run (repo style: plain module-level constants)
# --------------------------------------------------------------------------- #
POP_SIZE = 256  # trees per generation
GENERATIONS = 40  # evaluated generations
NUM_DATA = 800  # data rows of the synthetic problem
SEED = 0  # the driver's key and the data seed (one integer, no global RNG)
MAX_NODES = 64  # tree-buffer width (the torch stack's ``max_tree_len``)
ELITE_COUNT = 2  # 1 % elites (the torch stack's ``elite_rate=0.01``)
TOURNAMENT_SIZE = 4  # the pressure analogue of ``survival_rate=0.3``
CROSSOVER_PROB = 0.9  # per-individual crossover rate
MUTATION_PROB = 0.2  # per-individual subtree-mutation rate
REPORT_EVERY = 5  # print generation 0, every 5th one, and the last
DATA_BOUNDS = (-5.0, 5.0)  # the uniform data draw
INPUT_LEN = 2  # always ``x0``, ``x1``
USING_FUNCS = ("+", "-", "*", "/")  # the tree layer's function names
MAX_LAYER_CNT = 4  # depth schedule of the generation
LAYER_LEAF_PROB = 0.2
CONST_PROB = 0.5  # half the leaves are constants ...
CONST_RANGE = (-5.0, 5.0)  # ... drawn from this range,
SAMPLE_CNT = 8  # 8 values of it (the constant table)


def target_function(x: np.ndarray) -> np.ndarray:
    """``x0**2 + x1**2`` for ``x`` of shape ``[n_rows, 2]``."""
    values = np.asarray(x, dtype=np.float64)
    return values[:, 0] ** 2 + values[:, 1] ** 2


def make_data(
    num_data: int = NUM_DATA, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """The synthetic problem: ``(x [num_data, 2], y [num_data])`` float32."""
    generator = np.random.default_rng(seed)
    low, high = DATA_BOUNDS
    x = generator.uniform(low, high, size=(int(num_data), INPUT_LEN))
    x = x.astype(np.float32)
    return x, target_function(x).astype(np.float32)


def render_expression(channels, input_len: int = INPUT_LEN) -> str:
    """Readable infix string of one tree (the stand-in for ``to_sympy_expr``).

    ``channels`` is the ``(opcode, payload, size)`` triple of one individual
    (``GPResult.best_channels``), prefix order: node ``i``'s children start at
    ``i+1`` and each next child follows the previous child's own ``size``.  A
    ``size[i] == 1`` cell is a leaf -- ``opcode == NType.CONST`` renders
    ``payload`` as a number, ``NType.VAR`` renders ``x{int(payload)}`` -- and a
    ``size[i] > 1`` cell is a function node whose ``opcode`` is its ``Func`` id.
    """
    opcode, payload, size = (np.asarray(part) for part in channels)
    width = int(size.shape[0])

    def walk(node: int, depth: int) -> str:
        """Render the subtree rooted at ``node`` (``depth`` bounds the walk)."""
        if node >= width or size[node] <= 0 or depth > width:
            return "?"
        if size[node] == 1:
            if int(opcode[node]) == NType.CONST:
                return f"{float(payload[node]):.4g}"
            return f"x{int(float(payload[node])) % input_len}"
        function_id = int(opcode[node])
        children, child = [], node + 1
        for _ in range(FUNC_ARITY[function_id]):
            if child >= width:
                children.append("?")
                break
            children.append(walk(child, depth + 1))
            child += int(size[child])
        return _apply(function_id, children)

    return walk(0, 0)


def _apply(function_id: int, children: list[str]) -> str:
    """Render one function node from its already-rendered children."""
    if len(children) == 3:
        return f"if({children[0]}, {children[1]}, {children[2]})"
    if len(children) == 2:
        symbol = {Func.ADD: "+", Func.SUB: "-", Func.MUL: "*", Func.DIV: "/"}
        if function_id in symbol:
            return f"({children[0]} {symbol[function_id]} {children[1]})"
        return f"{FUNCS_NAMES[function_id]}({children[0]}, {children[1]})"
    if len(children) == 1:
        if function_id == Func.NEG:
            return f"(-{children[0]})"
        return f"{FUNCS_NAMES[function_id]}({children[0]})"
    return FUNCS_NAMES[function_id]


def main(argv: list[str] | None = None) -> int:
    """Run the regression and print the winner (``__main__`` does nothing else)."""
    del argv  # no flags: the constants above are the shipped run
    x, y = make_data()
    problem = SymbolicRegression(x, y)
    config = (
        GPTreeConfig(
            max_nodes=MAX_NODES,
            input_len=INPUT_LEN,
            using_funcs=USING_FUNCS,
            max_layer_cnt=MAX_LAYER_CNT,
            layer_leaf_prob=LAYER_LEAF_PROB,
            const_prob=CONST_PROB,
            const_range=CONST_RANGE,
            sample_cnt=SAMPLE_CNT,
        ),
        GPConfig(
            pop_size=POP_SIZE,
            elite_count=ELITE_COUNT,
            tournament_size=TOURNAMENT_SIZE,
            crossover_prob=CROSSOVER_PROB,
            mutation_prob=MUTATION_PROB,
            mutation_kind="subtree",
        ),
    )
    started = time.perf_counter()

    def report(generation: int, _state, fitness) -> None:
        """Print this generation's best fitness on the report cadence."""
        if generation % REPORT_EVERY and generation != GENERATIONS - 1:
            return
        best = float(np.max(fitness))
        print(
            f"generation {generation:3d}  best {best:+12.5f}  "
            f"{time.perf_counter() - started:6.2f}s",
            flush=True,
        )

    result = run(
        config, problem.evaluate, generations=GENERATIONS, key=SEED, callback=report
    )
    wall = time.perf_counter() - started

    history = result.best_fitness_history
    print(
        f"\npop {POP_SIZE} x {GENERATIONS} generations over {NUM_DATA} rows in "
        f"{wall:.2f}s ({1000.0 * wall / GENERATIONS:.1f} ms/generation)"
    )
    print(
        f"best fitness   {result.best_fitness:+.5f} at generation "
        f"{result.best_generation} of {len(history)} evaluated"
    )
    print(f"fitness path   {history[0]:+.5f} -> {history[-1]:+.5f}")
    if result.best_channels is None:
        print("no finite individual was ever evaluated")
        return 1
    print(f"winning tree   {int(row_length(result.best_channels)[0])} nodes")
    print(f"expression     {render_expression(result.best_channels)}")

    predictions = problem.predict(result.best_channels)[0]
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2, dtype=np.float64)))
    print(f"rmse           {rmse:.5f}")
    for row in range(3):
        print(
            f"  row {row}          prediction {predictions[row]:+12.6f}  "
            f"target {y[row]:+12.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
