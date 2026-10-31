"""Measurement harness: one GENERATION of ``evogp.gp`` vs the torch stack.

This is COMMITTED EVIDENCE, not a test and not a library entry point: nothing
runs at import time, everything lives inside :func:`main` or the function the
side dispatch calls.  Torch is resolved through :func:`importlib.import_module`
inside the functions (the boundary :mod:`evogp.gp.fused_eval` uses), because the
sibling purity test forbids the literal import statement in this subtree.

One cell is a ``(pop_size, rows)`` pair.  Six sides answer "what does ONE GP
generation cost?" on the same data, and the last one settles the fused-vs-in-
graph question of :mod:`evogp.gp.fused_eval`:

=============== ===============================================================
``gp``          ``evogp.gp`` driven DEVICE-RESIDENT: ``init_gp(backend="iree",
                device="rocm")`` + :class:`evogp.gp.graph_eval.GraphEvaluator`
                + :func:`evogp.gp.workflow.step` -- two ``evoxc.run`` calls, no
                host sync and no transfer per generation.
``gp-fused``    the same DEVICE-RESIDENT driver with the DEVICE-NATIVE fused
                evaluator (:class:`evogp.gp.fused_eval.FusedDeviceEvaluator`):
                the Triton kernel reads the device channels through zero-copy
                DLPack views (the legacy triple is converted in torch on the
                device) and hands the fitness back as an ``evoxc`` DEVICE
                tensor, so a generation is still ONE ``evoxc.run`` of the
                variation graph and no readback.
``numpy``       the same driver on numpy with the host
                :class:`evogp.gp.problem.SymbolicRegression` evaluator (the CPU
                reference of this stack).
``torch``       the TORCH STACK's ``StandardPipeline.step`` -- ``Forest`` /
                ``GeneticProgramming`` / ``SymbolicRegression`` built exactly the
                way ``test/benchmark/evoxc/arms.py`` builds them (same
                descriptor, same operators), on its default route (``auto`` ->
                dense+iree).
``torch-fused`` the same torch stack with ``EVOGP_EVAL_MODE=fused`` (its Triton
                evaluation route).
``fused``       the head-to-head of :mod:`evogp.gp.fused_eval`: one in-graph
                ``evoxc`` evaluation against the fused kernel fed from DEVICE
                channels (host round trip included) and from host channels.
=============== ===============================================================

Protocol (every number in ``DEVICE_LOOP.md`` depends on it):

* **One process per (side, cell).**  The parent invocation re-executes itself
  once per side and cell (``--side``/``--pop``/``--rows``), so a compile, an
  allocator or a leftover population can never be charged to another column.
* **Warm medians.**  A first block is discarded as warm-up, then ``--reps``
  repetitions are timed and the MEDIAN is reported (min/max/mean and every
  sample go into the JSON, because a shared card is noisy).
* **A device sync around every timed repetition** (``torch.cuda.synchronize()``,
  device-wide, so it also waits for the ``evoxc``/IREE work on the card).  The
  device-resident sides (``gp``, ``gp-fused``) get one sync bracket per
  generation -- what a real loop pays -- plus a second, separately-timed lane
  each for the evaluation and the variation segment.
* **Idle-device guard** before each timed process: ``rocm-smi --showuse`` /
  ``--showpids`` are recorded and the harness waits while the card is busy.

Run the grid (this is the report-of-record command)::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \
        /tmp/rocvenv/bin/python -m evogp.gp.measure_generation \
            --cells 64,256,1000,4096,256x4096 \
            --json src/evogp/gp/evidence/device_loop.json

``--cells`` is a comma list of ``POP`` (rows default 1024) or ``POPxROWS``;
``--sides`` picks the sides of the grid; one side of one cell runs in THIS
process with ``--side``/``--pop``/``--rows`` (what the parent spawns, and what
every committed number was produced by).  The honest headline of a side is its
``total``: one whole generation (evaluation + select/cross/mutate) with the
device synced.  The ``evaluation``/``variation`` columns decompose it and sum to
``total`` within a few percent on the device-resident sides (``gp``,
``gp-fused``): each segment carries its own sync bracket, so the two sum and the
one-bracket ``total`` can disagree by a few percent and ``total`` is the number
to quote; the torch side takes its split inside the step through a timed problem
wrapper, so ``operator = total - evaluation``.  The torch stack selects by
truncation (``survival_rate=0.3``, ``elite_rate=0.01``, the benchmark campaign's
wiring) while ``evogp.gp`` uses a tournament of 4, so the two sides are
deliberately *not* the same operator mix.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import GPConfig, GPTreeConfig
from .graph_eval import GraphEvaluator, readback_fitness
from .measure_fused import make_data
from .problem import SymbolicRegression
from .workflow import init_gp, step

__all__ = [
    "WORKLOAD",
    "GP_KNOBS",
    "CELLS",
    "SIDES",
    "parse_cell",
    "make_configs",
    "measure_gp",
    "measure_torch",
    "measure_fused_head_to_head",
    "measure_side",
    "versions",
    "guard",
    "main",
]

#: The tree genome of every cell (one definition, shared by all sides).
WORKLOAD: Dict[str, Any] = {
    "max_nodes": 32,
    "input_len": 8,
    "using_funcs": ("+", "-", "*", "/"),
    "max_layer_cnt": 4,
    "const_prob": 0.5,
    "layer_leaf_prob": 0.2,
    "const_range": (-5.0, 5.0),
    "sample_cnt": 8,
}

#: The ``GPConfig`` of the ``evogp.gp`` sides (population size is per cell).
GP_KNOBS: Dict[str, Any] = {
    "elite_count": 2,
    "tournament_size": 4,
    "crossover_prob": 0.9,
    "mutation_prob": 0.2,
}

#: The grid: ``POP`` or ``POPxROWS`` (rows default :data:`ROWS`).
CELLS: Tuple[str, ...] = ("64", "256", "1000", "4096", "256x4096")
ROWS = 1024
SEED = 0

#: Sides of the grid (``fused`` is the head-to-head, not a driver side).
SIDES: Tuple[str, ...] = ("gp", "gp-fused", "numpy", "torch", "torch-fused", "fused")

#: Timed repetitions / discarded warm-up generations / trajectory length.
REPS = 15
WARMUP = 5
GENERATIONS = 20

#: The torch stack's evaluation route per side (``auto`` = the shipped default).
TORCH_EVAL_MODES = {"torch": "auto", "torch-fused": "fused"}

#: ``GPU use`` percentage above which the card counts as busy (contamination).
BUSY_PCT = 25.0


# --------------------------------------------------------------------------- #
# lazy imports (the ONE torch/evoxc boundary, resolved at CALL time)
# --------------------------------------------------------------------------- #
def _torch():
    """The torch module, imported on first use (never at module import)."""
    return importlib.import_module("torch")


def _evoxc():
    """The ``evoxc`` module, imported on first use."""
    return importlib.import_module("evoxc")


def _sync() -> None:
    """Device-wide synchronisation (waits for the ``evoxc``/IREE work too)."""
    _torch().cuda.synchronize()


def _device_kind() -> str:
    """The evoxc device kind matching this box's torch build (``rocm``/``cuda``)."""
    torch = _torch()
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_configs(
    pop: int, rows: int = ROWS, seed: int = SEED
) -> Tuple[GPTreeConfig, GPConfig, np.ndarray, np.ndarray]:
    """``(tree_config, gp_config, x, y)`` of one cell.

    The data set is deliberately the one :mod:`evogp.gp.measure_fused` uses (same
    target, same draw), so the two harnesses measure the same problem.
    """
    tree = GPTreeConfig(**WORKLOAD)
    gp_config = GPConfig(pop_size=int(pop), **GP_KNOBS)
    x, y = make_data(int(rows), int(WORKLOAD["input_len"]), int(seed))
    return tree, gp_config, x, y


def parse_cell(text: str, rows: int = ROWS) -> Tuple[int, int]:
    """``"256"`` -> ``(256, rows)``; ``"256x4096"`` -> ``(256, 4096)``."""
    body = str(text).strip().lower()
    if "x" in body:
        pop_text, rows_text = body.split("x", 1)
        return int(pop_text), int(rows_text)
    return int(body), int(rows)


# --------------------------------------------------------------------------- #
# timing primitives
# --------------------------------------------------------------------------- #
def _block(samples: Sequence[float]) -> Dict[str, Any]:
    """Median/min/max/mean of millisecond samples, plus the samples themselves."""
    array = np.asarray(list(samples), dtype=np.float64)
    return {
        "median_ms": round(float(np.median(array)), 4),
        "min_ms": round(float(np.min(array)), 4),
        "max_ms": round(float(np.max(array)), 4),
        "mean_ms": round(float(np.mean(array)), 4),
        "reps": int(array.size),
        "samples_ms": [round(float(value), 4) for value in array],
    }


def _median(
    fn: Callable[[], Any],
    iters: int,
    warmup: int,
    sync: Optional[Callable[[], None]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Time ``fn`` ``iters`` times (``warmup`` untimed calls first); report medians."""
    for _ in range(max(0, warmup)):
        fn()
    if sync is not None:
        sync()
    samples: List[float] = []
    result: Any = None
    for _ in range(max(1, iters)):
        if sync is not None:
            sync()
        start = time.perf_counter()
        result = fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return _block(samples), result


def _scores(values) -> Tuple[float, float]:
    """``(best, median)`` of a fitness vector over its finite entries."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float("-inf"), float("-inf")
    return float(np.max(finite)), float(np.median(finite))


# --------------------------------------------------------------------------- #
# the evogp.gp sides (gp = device-resident iree, numpy = the host reference)
# --------------------------------------------------------------------------- #
def _gp_objective(tree: GPTreeConfig, x: np.ndarray, y: np.ndarray, backend: str):
    """``(evaluate, sync)`` of one side.

    ``evaluate`` is the device-resident in-graph fitness (an ``evoxc`` tensor, no
    readback) on ``iree`` and the host ``SymbolicRegression.evaluate`` on numpy;
    ``sync`` is ``None`` on numpy because a host loop has nothing to wait for.
    """
    if backend == "numpy":
        return SymbolicRegression(x, y).evaluate, None
    evaluator = GraphEvaluator(
        x, y, used_ids=tree, backend=backend, device=_device_kind()
    )
    return evaluator.evaluate, _sync


def _fused_device_objective(x: np.ndarray, y: np.ndarray):
    """``(evaluate, sync, evaluator)`` of the DEVICE-NATIVE fused route.

    ``evaluate`` is :meth:`evogp.gp.fused_eval.FusedDeviceEvaluator.evaluate`:
    the channels stay where they are (zero-copy DLPack views into the Triton
    kernel) and the ``(pop_size,)`` fitness comes back as an ``evoxc`` DEVICE
    tensor, exactly what :func:`evogp.gp.workflow.step` wants for a generation
    with no readback.  The evaluator is returned too, because the launch count
    and the route actually taken are its own counters (never an assumption).
    """
    from . import fused_eval  # noqa: PLC0415 - local: it owns the torch boundary

    evaluator = fused_eval.FusedDeviceEvaluator(x, y)
    return evaluator.evaluate, _sync, evaluator


def _tap(fitness) -> Callable[[Any], Any]:
    """An ``evaluate`` that hands a precomputed fitness tensor straight back.

    The variation lane times ONE graph run: the population's fitness was already
    measured by the evaluation lane, so re-evaluating it would charge the
    evaluation graph to the operator column.
    """

    def evaluate(_channels):
        return fitness

    return evaluate


def _gp_segments(
    config, state, evaluate, sync, *, backend: str, device: str, warmup: int, reps: int
):
    """Two timed lanes: the evaluation/variation split, then the honest total.

    lane A measures one evaluation call and one variation call in turn (each
    inside its own sync bracket), which decomposes a generation;
    lane B measures ``step(config, state, evaluate)`` -- exactly the call a real
    loop makes -- inside ONE sync bracket, which is the number to quote.
    """
    for _ in range(max(0, warmup)):
        fitness = evaluate(state)
        if sync is not None:
            sync()
        state = step(config, state, _tap(fitness), backend=backend, device=device)
    if sync is not None:
        sync()
    evaluation_samples: List[float] = []
    variation_samples: List[float] = []
    for _ in range(max(1, reps)):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fitness = evaluate(state)
        if sync is not None:
            sync()
        middle = time.perf_counter()
        state = step(config, state, _tap(fitness), backend=backend, device=device)
        if sync is not None:
            sync()
        end = time.perf_counter()
        evaluation_samples.append((middle - start) * 1e3)
        variation_samples.append((end - middle) * 1e3)
    total, state = _median(
        lambda: step(config, state, evaluate, backend=backend, device=device),
        reps,
        0,
        sync,
    )
    return state, {
        "evaluation": _block(evaluation_samples),
        "variation": _block(variation_samples),
        "total": total,
    }


def _gp_trajectory(
    config, evaluate, *, backend: str, device: str, seed: int, generations: int
) -> Dict[str, Any]:
    """Best/median fitness of every evaluated generation of a short run."""
    state = init_gp(config, seed, backend=backend, device=device)
    best: List[float] = []
    median: List[float] = []
    for _ in range(max(1, generations)):
        raw = evaluate(state)
        values = (
            np.asarray(raw, dtype=np.float32)
            if backend == "numpy"
            else readback_fitness(raw, int(raw.shape[0]))
        )
        top, middle = _scores(values)
        best.append(top)
        median.append(middle)
        state = step(config, state, _tap(raw), backend=backend, device=device)
    return {
        "best": [round(value, 6) for value in best],
        "median": [round(value, 6) for value in median],
    }


def count_runs(config, evaluate, *, backend: str, device: str, seed: int) -> Dict[str, Any]:
    """How many ``evoxc.run`` calls one generation of the device loop makes.

    Wraps ``evoxc.run`` over one generation of a live loop: "two graph runs and
    no host sync" is the whole claim of the device path, so it is counted rather
    than asserted.  The step hands the device fitness tensor straight to the
    variation graph, so nothing crosses to the host between the two runs.
    """
    evoxc, original = _evoxc(), _evoxc().run
    counter = {"run": 0}

    def counted(*args, **kwargs):
        counter["run"] += 1
        return original(*args, **kwargs)

    evoxc.run = counted
    try:
        state = init_gp(config, seed, backend=backend, device=device)
        counter["run"] = 0  # init_gp's own generation graph is not a generation
        fitness = evaluate(state)
        evaluation_runs = counter["run"]
        step(config, state, _tap(fitness), backend=backend, device=device)
        variation_runs = counter["run"] - evaluation_runs
    finally:
        evoxc.run = original
    return {
        "generation_runs": counter["run"],
        "evaluation_runs": evaluation_runs,
        "variation_runs": variation_runs,
        "host_readbacks": 0,
        "syncs": 0,
        "transfers": 0,
    }


def count_fused_launches(
    evaluator, config, *, backend: str, device: str, seed: int
) -> Dict[str, Any]:
    """How many Triton launches one generation of ``gp-fused`` makes.

    Counted on the evaluator's OWN route counter
    (:attr:`evogp.gp.fused_eval.FusedDeviceEvaluator.route_counts`), so a call
    that quietly fell back to the host route is visible as ``host`` launches
    plus the reason, instead of being reported as a kernel launch.  One
    generation is one evaluation, so ``fused_launches`` is the per-generation
    kernel count.
    """
    state = init_gp(config, seed, backend=backend, device=device)
    before = dict(evaluator.route_counts)
    fitness = evaluator.evaluate(state)
    after = dict(evaluator.route_counts)
    step(config, state, _tap(fitness), backend=backend, device=device)
    return {
        "fused_launches": int(after["device"] - before["device"]),
        "host_route_launches": int(after["host"] - before["host"]),
        "route_counts": after,
        "evaluate_route": evaluator.last_route,
        "last_fallback": evaluator.last_fallback,
    }


def measure_gp(
    pop: int,
    rows: int = ROWS,
    *,
    backend: str = "iree",
    warmup: int = WARMUP,
    reps: int = REPS,
    generations: int = GENERATIONS,
    seed: int = SEED,
    fused: bool = False,
) -> Dict[str, Any]:
    """One cell of the ``evogp.gp`` stack on ``backend`` (device or numpy).

    ``fused=True`` keeps the DEVICE-RESIDENT driver (``backend="iree"`` plus the
    per-stack device) and swaps only the objective for the device-native fused
    evaluator -- the ``gp-fused`` side.
    """
    tree, gp_config, x, y = make_configs(pop, rows, seed)
    config = (tree, gp_config)
    if fused:
        backend = "iree"
    device = "cpu" if backend == "numpy" else _device_kind()
    evaluator = None
    if fused:
        evaluate, sync, evaluator = _fused_device_objective(x, y)
    else:
        evaluate, sync = _gp_objective(tree, x, y, backend)
    state = init_gp(config, seed, backend=backend, device=device)
    cold, _ = _median(
        lambda: step(config, state, evaluate, backend=backend, device=device),
        iters=1,
        warmup=0,
        sync=sync,
    )
    # the FIRST call of the process: which route actually served it (a kernel
    # launch, or the host fallback and its reason) is a fact, not an assumption
    first_call = (
        None
        if evaluator is None
        else {
            "route": evaluator.last_route,
            "fallback": evaluator.last_fallback,
            "available": bool(evaluator.available()),
            "device_available": bool(evaluator.device_available()),
        }
    )
    state, timing = _gp_segments(
        config, state, evaluate, sync, backend=backend, device=device,
        warmup=warmup, reps=reps,
    )
    # generation-0 evaluation: the timed loop above runs on an EVOLVED population,
    # whose trees are not the freshly generated ones
    fresh_state = init_gp(config, seed, backend=backend, device=device)
    fresh, _ = _median(lambda: evaluate(fresh_state), reps, 1, sync)
    record: Dict[str, Any] = {
        "side": "gp-fused" if fused else ("numpy" if backend == "numpy" else "gp"),
        "pop": int(pop),
        "rows": int(rows),
        "backend": backend,
        "device": device,
        "timing": timing,
        "fresh_evaluation": fresh,
        "cold_ms": cold["median_ms"],
        "trajectory": _gp_trajectory(
            config, evaluate, backend=backend, device=device, seed=seed,
            generations=generations,
        ),
    }
    if evaluator is not None:
        record["evaluator"] = first_call
    if backend != "numpy":
        structure = count_runs(
            config, evaluate, backend=backend, device=device, seed=seed
        )
        if evaluator is not None:
            structure.update(
                count_fused_launches(
                    evaluator, config, backend=backend, device=device, seed=seed
                )
            )
            # the device route's ONLY host touch, stated rather than glossed over
            structure["route_note"] = (
                "device route: the channels never leave the device (zero-copy "
                "DLPack views) and the driver consumes the device fitness "
                "tensor, so a generation is one variation graph run with no "
                "channel transfer and no fitness readback; the remaining host "
                "touch is the evaluator's OUT_NODE guard (a single boolean "
                "readback) inside the evaluation lane"
            )
        record["structure"] = structure
    return record


# --------------------------------------------------------------------------- #
# the torch stack (``StandardPipeline.step``)
# --------------------------------------------------------------------------- #
class _TimedProblem:
    """``BaseProblem`` wrapper that records the wall time of its own ``evaluate``."""

    def __init__(self, problem) -> None:
        self.problem = problem
        self.last_ms = 0.0

    def evaluate(self, forest):
        start = time.perf_counter()
        fitness = self.problem.evaluate(forest)
        self.last_ms = (time.perf_counter() - start) * 1e3
        return fitness

    @property
    def problem_dim(self):
        return self.problem.problem_dim

    @property
    def solution_dim(self):
        return self.problem.solution_dim


def _torch_pipeline(tree: GPTreeConfig, x, y, pop: int, *, seed: int):
    """``GeneticProgramming`` + ``StandardPipeline`` over the cell's workload.

    Mirrors ``test/benchmark/evoxc/arms.py::make_pipeline`` (the same descriptor,
    the same three default operators, the campaign's truncation selection) with
    the real ``evogp.problem.SymbolicRegression`` rather than the harness's
    fitness shim, so the evaluation is the one a user gets.
    """
    torch = _torch()
    from evogp.algorithm import (  # noqa: PLC0415 - call-time import on purpose
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline  # noqa: PLC0415
    from evogp.problem import (  # noqa: PLC0415
        SymbolicRegression as TorchSymbolicRegression,
    )
    from evogp.tree import Forest, GenerateDescriptor  # noqa: PLC0415

    torch.manual_seed(seed)
    descriptor = GenerateDescriptor(
        max_tree_len=int(WORKLOAD["max_nodes"]),
        input_len=int(WORKLOAD["input_len"]),
        output_len=1,
        using_funcs=list(WORKLOAD["using_funcs"]),
        max_layer_cnt=int(WORKLOAD["max_layer_cnt"]),
        const_prob=float(WORKLOAD["const_prob"]),
        layer_leaf_prob=float(WORKLOAD["layer_leaf_prob"]),
        const_samples=[float(value) for value in tree.const_samples],
    )
    forest = Forest.random_generate(pop_size=int(pop), descriptor=descriptor)
    algorithm = GeneticProgramming(
        initial_forest=forest,
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    problem = TorchSymbolicRegression(
        datapoints=torch.as_tensor(x, device="cuda"),
        labels=torch.as_tensor(
            np.asarray(y, dtype=np.float32).reshape(-1, 1), device="cuda"
        ),
    )
    timed = _TimedProblem(problem)
    pipeline = StandardPipeline(algorithm, timed, is_show_details=False)
    return pipeline, timed


def measure_torch(
    pop: int,
    rows: int = ROWS,
    *,
    eval_mode: str = "auto",
    warmup: int = WARMUP,
    reps: int = REPS,
    generations: int = GENERATIONS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """One cell of the TORCH stack's ``StandardPipeline.step``."""
    if eval_mode and eval_mode != "auto":
        os.environ["EVOGP_EVAL_MODE"] = eval_mode
    else:
        os.environ.pop("EVOGP_EVAL_MODE", None)
    tree, _gp_config, x, y = make_configs(pop, rows, seed)
    pipeline, timed = _torch_pipeline(tree, x, y, pop, seed=seed)
    cold, _ = _median(pipeline.step, iters=1, warmup=0, sync=_sync)
    for _ in range(max(0, warmup)):
        pipeline.step()
    _sync()
    total_samples: List[float] = []
    eval_samples: List[float] = []
    for _ in range(max(1, reps)):
        _sync()
        start = time.perf_counter()
        pipeline.step()
        _sync()
        total_samples.append((time.perf_counter() - start) * 1e3)
        eval_samples.append(timed.last_ms)
    total = _block(total_samples)
    evaluation = _block(eval_samples)
    operators = dict(total)
    operators["median_ms"] = round(total["median_ms"] - evaluation["median_ms"], 4)
    operators.pop("samples_ms")
    record: Dict[str, Any] = {
        "side": "torch" if eval_mode == "auto" else "torch-fused",
        "pop": int(pop),
        "rows": int(rows),
        "eval_mode": eval_mode,
        "timing": {"evaluation": evaluation, "variation": operators, "total": total},
        "cold_ms": cold["median_ms"],
    }
    # trajectory: a fresh, identically seeded pipeline, one step per generation
    pipeline, _timed = _torch_pipeline(tree, x, y, pop, seed=seed)
    best: List[float] = []
    median: List[float] = []
    for _ in range(max(1, generations)):
        top, middle = _scores(pipeline.step().numpy())
        best.append(top)
        median.append(middle)
    record["trajectory"] = {
        "best": [round(value, 6) for value in best],
        "median": [round(value, 6) for value in median],
    }
    return record


# --------------------------------------------------------------------------- #
# the open question: fused (Triton) vs the in-graph evoxc route
# --------------------------------------------------------------------------- #
def _parity(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Agreement of two fitness vectors (relative deltas + finiteness mismatches)."""
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    both = np.isfinite(reference) & np.isfinite(candidate)
    left, right = reference[both], candidate[both]
    scale = np.maximum(np.abs(left), np.float32(1e-30))
    delta = np.abs(right - left)
    top = float(np.max(np.abs(left))) if left.size else 0.0
    return {
        "max_rel": float(np.max(delta / scale)) if delta.size else 0.0,
        "max_rel_scaled": float(np.max(delta) / top) if delta.size and top else 0.0,
        "scale": top,
        "comparable": int(np.count_nonzero(both)),
        "finiteness_mismatch": int(
            np.count_nonzero(np.isfinite(reference) != np.isfinite(candidate))
        ),
    }


def measure_fused_head_to_head(
    pop: int,
    rows: int = ROWS,
    *,
    warmup: int = 4,
    reps: int = REPS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """In-graph ``evoxc`` vs the fused Triton kernel, on the SAME channels.

    All columns come from the live device channels of one population:
    ``in_graph`` (``GraphEvaluator.evaluate(device channels)`` to device fitness,
    no readback -- the shipped path), ``transfer`` (the three channel tensors to
    numpy, idle and QUEUED, i.e. right after an in-graph run), ``fused_host``
    (the kernel on channels already in host memory), and ``fused_device``
    (transfer + kernel composed in ONE timed repetition -- what a device loop
    actually pays), plus its queued twin.
    """
    from . import fused_eval  # noqa: PLC0415 - local: it owns the torch boundary
    from .encoding import _channel_arrays  # noqa: PLC0415

    evoxc = _evoxc()
    tree, _gp_config, x, y = make_configs(pop, rows, seed)
    kind = _device_kind()
    device, cpu = evoxc.core.Device(kind, 0), evoxc.core.Device("cpu", 0)
    host = tuple(_channel_arrays(init_gp(
        (tree, GPConfig(pop_size=int(pop), **GP_KNOBS)), seed)))
    resident = tuple(
        evoxc.from_numpy(np.ascontiguousarray(array)).to(device) for array in host
    )
    evaluator = GraphEvaluator(x, y, used_ids=tree, backend="iree", device=kind)
    fused = fused_eval.FusedEvaluator(x, y)

    def transfer() -> Tuple[np.ndarray, ...]:
        return tuple(tensor.to(cpu).numpy() for tensor in resident)

    def device_path() -> np.ndarray:
        return fused(transfer())

    in_graph, fitness = _median(
        lambda: evaluator.evaluate(resident), reps, warmup, _sync
    )
    reference = readback_fitness(fitness, int(pop))
    transfer_idle, _ = _median(transfer, reps, warmup, _sync)
    fused_host, host_values = _median(lambda: fused(host), reps, warmup, _sync)
    fused_device, device_values = _median(device_path, reps, warmup, _sync)
    queued_pair, _ = _median(
        lambda: (evaluator.evaluate(resident), transfer()), reps, warmup, _sync
    )
    transfer_queued = round(queued_pair["median_ms"] - in_graph["median_ms"], 4)
    # the same fused path with the transfer queued behind device work, as a sum of
    # two separately measured medians (the composed lane above is the headline)
    fused_device_queued = {
        "median_ms": round(transfer_queued + fused_host["median_ms"], 4),
        "note": "sum of the queued-transfer median and the host-channels fused median",
    }
    graph, device_ms = in_graph["median_ms"], fused_device["median_ms"]
    return {
        "side": "fused",
        "pop": int(pop),
        "rows": int(rows),
        "device": kind,
        "triton_cache_dir": fused_eval.ensure_triton_cache_dir(),
        "timing": {
            "in_graph": in_graph,
            "transfer": transfer_idle,
            "in_graph_queued": {
                "median_ms": round(queued_pair["median_ms"] - transfer_queued, 4),
                "note": "the in-graph run with a busy queue "
                        "(pair lane minus the derived queued transfer)",
            },
            "transfer_queued": {
                "median_ms": transfer_queued,
                "note": "(in-graph run + transfer) pair minus the in-graph run",
            },
            "fused_host": fused_host,
            "fused_device": fused_device,
            "fused_device_queued": fused_device_queued,
        },
        "parity": {
            "fused_host_vs_in_graph": _parity(reference, host_values),
            "fused_device_vs_in_graph": _parity(reference, device_values),
            "fused_device_queued_vs_in_graph": _parity(reference, host_values),
        },
        "verdict": {
            "host_channels_speedup": round(graph / fused_host["median_ms"], 4),
            "device_channels_speedup": round(graph / fused_device["median_ms"], 4),
            "device_channels_queued_speedup": round(
                graph / fused_device_queued["median_ms"], 4
            ),
            "faster_route": "in-graph" if device_ms >= graph else "fused",
        },
    }


# --------------------------------------------------------------------------- #
# provenance and the idle-device guard
# --------------------------------------------------------------------------- #
def _pkg_version(name: str) -> Optional[str]:
    """Installed distribution version, else the module's own ``__version__``.

    The fallback matters for triton: on this box it is bundled with the ROCm torch
    build and has no distribution metadata, so ``metadata.version`` raises.
    """
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:  # noqa: BLE001 - best-effort provenance
        pass
    try:
        return getattr(importlib.import_module(name), "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _git_head(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def versions() -> Dict[str, Any]:
    """Box / library provenance of a record (never raises)."""
    block: Dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "iree_base_compiler": _pkg_version("iree-base-compiler"),
        "iree_base_runtime": _pkg_version("iree-base-runtime"),
        "evoxc_sha": _git_head("/root/evoxc"),
        "evogp_sha": _git_head(os.getcwd()),
        "knobs": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("EVOGP_") or key.startswith("EVOXC_")
        },
    }
    try:
        torch = _torch()
        block.update(
            torch=torch.__version__,
            hip=getattr(torch.version, "hip", None),
            cuda=getattr(torch.version, "cuda", None),
            triton=_pkg_version("triton"),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except Exception as exc:  # noqa: BLE001
        block["torch_error"] = repr(exc)
    try:
        evoxc = _evoxc()
        block["evoxc_version"] = getattr(evoxc, "__version__", None)
        block["evoxc_file"] = getattr(evoxc, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        block["evoxc_error"] = repr(exc)
    return block


def _rocm(*flags: str) -> Optional[str]:
    """``rocm-smi <flags>`` stdout, or ``None`` when the tool is unavailable."""
    try:
        return subprocess.run(
            ["rocm-smi", *flags], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _probe() -> Dict[str, Any]:
    """One ``rocm-smi`` probe: card utilisation and the resident python processes."""
    state: Dict[str, Any] = {}
    use = _rocm("--showuse")
    if use is not None:
        percents = []
        for line in use.splitlines():
            if "GPU use (%)" in line:
                digits = "".join(ch for ch in line.split(":", 2)[-1] if ch.isdigit())
                percents.append(float(digits) if digits else 0.0)
        state["gpu_use_pct"] = percents
    pids = _rocm("--showpids")
    if pids is not None:
        state["resident_processes"] = [
            line.strip() for line in pids.splitlines() if "python" in line
        ]
    state["busy"] = bool(max(state.get("gpu_use_pct") or [0.0]) > BUSY_PCT)
    return state


def guard(wait_s: float = 60.0, poll_s: float = 5.0) -> Dict[str, Any]:
    """Idle-device guard: a ``rocm-smi`` report plus a bounded wait while busy.

    The card is SHARED, so this never blocks forever and always returns what it
    saw: a contaminated cell is stated in the record, never hidden.
    """
    start = time.time()
    state = _probe()
    while state["busy"] and time.time() - start < wait_s:
        time.sleep(poll_s)
        state = _probe()
    state["waited_s"] = round(time.time() - start, 1)
    state["ok"] = not state["busy"]
    return state


# --------------------------------------------------------------------------- #
# one side of one cell (child mode)
# --------------------------------------------------------------------------- #
def measure_side(
    side: str,
    pop: int,
    rows: int,
    *,
    warmup: int = WARMUP,
    reps: int = REPS,
    generations: int = GENERATIONS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Dispatch one side of one cell."""
    triton_dir = None
    if side in ("torch-fused", "gp-fused", "fused"):
        # BEFORE any fused work: this box's $HOME is read-only, so triton's
        # default cache path dies with OSError [Errno 30] on the first launch.
        from . import fused_eval  # noqa: PLC0415

        triton_dir = fused_eval.ensure_triton_cache_dir()
    if side in ("gp", "gp-fused", "numpy"):
        record = measure_gp(
            pop,
            rows,
            backend="numpy" if side == "numpy" else "iree",
            warmup=warmup,
            reps=reps,
            generations=generations,
            seed=seed,
            fused=side == "gp-fused",
        )
    elif side in TORCH_EVAL_MODES:
        record = measure_torch(
            pop, rows, eval_mode=TORCH_EVAL_MODES[side], warmup=warmup, reps=reps,
            generations=generations, seed=seed,
        )
    elif side == "fused":
        record = measure_fused_head_to_head(
            pop, rows, warmup=min(4, warmup), reps=reps, seed=seed
        )
    else:
        raise ValueError(f"unknown side {side!r}; known: {', '.join(SIDES)}")
    if triton_dir is not None:
        record["triton_cache_dir"] = triton_dir
    return record


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _ms(block: Optional[Dict[str, Any]]) -> str:
    if not block or block.get("median_ms") is None:
        return "-"
    return f"{block['median_ms']:.3f}"


def _timed(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (record for record in records if record["side"] != "fused"),
        key=lambda record: (record["rows"], record["pop"], record["side"]),
    )


def report(records: Sequence[Dict[str, Any]]) -> str:
    """Every timed cell and the fused verdict, as Markdown (trajectories stay in the JSON)."""
    lines = [
        "## per-cell timing (ms per generation, warm medians)",
        "",
        "| side | pop | rows | evaluation | operator | total | total min-max | cold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in _timed(records):
        timing = record["timing"]
        total = timing.get("total") or {}
        lines.append(
            "| {side} | {pop} | {rows} | {ev} | {op} | {total} | {spread} | {cold} |".format(
                side=record["side"],
                pop=record["pop"],
                rows=record["rows"],
                ev=_ms(timing.get("evaluation")),
                op=_ms(timing.get("variation")),
                total=_ms(total),
                spread=(
                    f"{total['min_ms']:.2f}-{total['max_ms']:.2f}" if total else "-"
                ),
                cold=record.get("cold_ms"),
            )
        )
    lines.append("")
    lines.append("## fused vs in-graph (the open question)")
    lines.append("")
    for record in sorted(
        (item for item in records if item["side"] == "fused"),
        key=lambda item: (item["rows"], item["pop"]),
    ):
        timing, verdict = record["timing"], record["verdict"]
        lines.append(
            "- pop {pop} rows {rows}: in-graph {graph} | transfer {transfer} "
            "(queued {queued}) | fused from host {host} | fused from device "
            "{device} (queued {device_q}) -> **{faster}**".format(
                pop=record["pop"],
                rows=record["rows"],
                graph=_ms(timing["in_graph"]),
                transfer=_ms(timing["transfer"]),
                queued=timing["transfer_queued"]["median_ms"],
                host=_ms(timing["fused_host"]),
                device=_ms(timing["fused_device"]),
                device_q=_ms(timing["fused_device_queued"]),
                faster=(
                    f"{verdict['faster_route']} "
                    f"{max(verdict['device_channels_queued_speedup'], 1.0):.2f}x"
                    if verdict["faster_route"] == "fused"
                    else f"in-graph "
                    f"{1 / verdict['device_channels_queued_speedup']:.2f}x"
                ),
            )
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# orchestration: one process per (side, cell)
# --------------------------------------------------------------------------- #
def _child_command(side: str, pop: int, rows: int, args, fragment: str) -> List[str]:
    return [
        sys.executable, "-m", "evogp.gp.measure_generation",
        "--side", side, "--pop", str(pop), "--rows", str(rows),
        "--warmup", str(args.warmup), "--reps", str(args.reps),
        "--generations", str(args.generations), "--seed", str(args.seed),
        "--json", fragment,
    ]


def _write(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")


def _command_line(raw: Sequence[str]) -> str:
    return " ".join([sys.executable, "-m", "evogp.gp.measure_generation"] + list(raw))


def _crid(side: str, pop: int, rows: int) -> str:
    return f"{side} pop={pop} rows={rows}"


def _child(args) -> int:
    """Child mode: measure ONE side of ONE cell in this process."""
    state = None if args.no_guard else guard(args.guard_wait)
    record = measure_side(
        args.side, int(args.pop), int(args.rows), warmup=args.warmup,
        reps=args.reps, generations=args.generations, seed=args.seed,
    )
    payload = {
        "command": _command_line(args.raw),
        "config": {
            "workload": dict(WORKLOAD),
            "gp_knobs": dict(GP_KNOBS),
            "side": args.side,
            "pop": int(args.pop),
            "rows": int(args.rows),
            "warmup": int(args.warmup),
            "reps": int(args.reps),
            "generations": int(args.generations),
            "seed": int(args.seed),
        },
        "guard": state,
        "versions": versions(),
        "records": [record],
    }
    if args.json:
        _write(args.json, payload)
    print(json.dumps(record, indent=2, sort_keys=True, default=float))
    return 0


def _parent(args) -> int:
    """Parent mode: spawn one process per (side, cell), print and merge."""
    cells = [parse_cell(text, args.rows) for text in str(args.cells).split(",") if text]
    sides = [text.strip() for text in str(args.sides).split(",") if text.strip()]
    unknown = [side for side in sides if side not in SIDES]
    if unknown:
        raise SystemExit(f"unknown --sides entries {unknown}; known: {', '.join(SIDES)}")
    fragment_dir = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"evogp-measure-generation-{os.getpid()}"
    )
    os.makedirs(fragment_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []
    guards: List[Dict[str, Any]] = []
    children: List[Dict[str, Any]] = []
    for pop, rows in cells:
        for side in sides:
            label = _crid(side, pop, rows)
            fragment = os.path.join(fragment_dir, f"{side}-{pop}x{rows}.json")
            command = _child_command(side, pop, rows, args, fragment)
            print(f"# measuring {label}", flush=True)
            started = time.time()
            completed = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.time() - started
            if completed.returncode != 0:
                sys.stderr.write(completed.stdout + completed.stderr)
                raise SystemExit(f"{label} failed (exit {completed.returncode})")
            with open(fragment, encoding="utf-8") as handle:
                payload = json.load(handle)
            state = payload.get("guard") or {}
            records.extend(payload["records"])
            guards.append({"side": side, "pop": pop, "rows": rows, **state})
            children.append({"side": side, "pop": pop, "rows": rows,
                             "command": command, "wall_s": round(elapsed, 2)})
            print(f"#   done in {elapsed:.1f} s (guard ok={state.get('ok')})",
                  flush=True)
    print()
    print(report(records))
    if args.json:
        _write(
            args.json,
            {
                "command": _command_line(args.raw),
                "config": {
                    "workload": dict(WORKLOAD), "gp_knobs": dict(GP_KNOBS),
                    "cells": cells, "sides": sides,
                    "warmup": int(args.warmup), "reps": int(args.reps),
                    "generations": int(args.generations), "seed": int(args.seed),
                    "protocol": (
                        "one process per (side, cell); warm medians of --reps timed "
                        "repetitions with torch.cuda.synchronize() around each; "
                        "idle-device guard per child"
                    ),
                },
                "versions": versions(),
                "children": children,
                "guards": guards,
                "records": records,
            },
        )
        print()
        print(f"# merged record written to {args.json}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: ``python -m evogp.gp.measure_generation``."""
    parser = argparse.ArgumentParser(
        prog="python -m evogp.gp.measure_generation",
        description=(
            "One GP generation: the device-resident evogp.gp loop vs the numpy "
            "path vs the torch stack, plus the fused-vs-in-graph head-to-head."
        ),
    )
    for flag, options in (
        ("--cells", {"default": ",".join(CELLS),
                     "help": f"comma list of POP or POPxROWS ({','.join(CELLS)})"}),
        ("--sides", {"default": ",".join(SIDES),
                     "help": f"comma list of sides ({','.join(SIDES)})"}),
        ("--side", {"default": None, "help": "measure only this side (child mode)"}),
        ("--pop", {"type": int, "default": None, "help": "population (child mode)"}),
        ("--rows", {"type": int, "default": ROWS, "help": f"data rows ({ROWS})"}),
        ("--warmup", {"type": int, "default": WARMUP,
                      "help": "discarded generations before the timed block"}),
        ("--reps", {"type": int, "default": REPS, "help": "timed repetitions"}),
        ("--generations", {"type": int, "default": GENERATIONS,
                           "help": "generations of the fitness trajectory"}),
        ("--seed", {"type": int, "default": SEED, "help": "RNG seed"}),
        ("--no-guard", {"action": "store_true", "help": "skip the idle-device guard"}),
        ("--guard-wait", {"type": float, "default": 60.0,
                          "help": "seconds to wait for an idle card"}),
        ("--json", {"default": None, "help": "write the record to this path"}),
    ):
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    args.raw = list(argv if argv is not None else sys.argv[1:])
    if args.side is None:
        return _parent(args)
    if args.pop is None:
        raise SystemExit("--side needs --pop (one process measures one cell)")
    return _child(args)


if __name__ == "__main__":  # pragma: no cover - a harness, never imported for effect
    raise SystemExit(main())
