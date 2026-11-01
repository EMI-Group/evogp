"""Device-resident run of the torch-free ``evogp.gp`` stack.

``gp_sr.py`` is the CPU walkthrough of this stack; this is its device
counterpart.  A generation can live on the GPU even though the stack is
``evoxc`` + numpy: the channel triple never leaves the device, ``evaluate`` is
the in-graph ``evoxc`` route (:class:`~evogp.gp.GraphEvaluator`) or -- when
torch and triton are installed -- the sibling Triton kernel behind
:class:`~evogp.gp.FusedDeviceEvaluator`, and :func:`evogp.gp.run` is handed
``backend="iree"`` + the device kind so the operator graphs run there too.
Nothing here imports torch or triton: both are resolved lazily inside
``evogp.gp.fused_eval``, and :func:`~evogp.gp.fused_available` /
:func:`~evogp.gp.interop_available` are the probes that decide the route.  The
problem is the small synthetic target ``x0**2 + x1**2`` on ``[-5, 5]``, the
population 256 trees of at most 64 nodes, and 20 generations.

Run it from the repository root (``evoxc`` on the path; a writable
``TRITON_CACHE_DIR`` is what makes the SECOND run fast)::

    PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \
        TRITON_CACHE_DIR=/tmp/triton_cache_example \
        /tmp/rocvenv/bin/python example/gp_device.py

On a box with no usable device (no ``iree`` packages, or no rocm/cuda device)
the probe answers ``None`` and the script prints "no device: falling back to
the numpy host loop" and runs the same problem through
:class:`~evogp.gp.SymbolicRegression` on the CPU; a ``FusedUnavailable`` /
``InteropUnavailable`` / iree compile failure on a device box takes the same
fallback, reason first.  It therefore always exits 0, and a CPU-only box needs
no Triton compilation at all (~1 s end to end).  On the reference box that
printed reason is a real one -- ``evoxc``'s iree lowering refuses the 64-wide
channel graph (``iree-compile`` fails to distribute its ``stablehlo.reduce``)
while ``max_nodes=32`` compiles -- but torch + triton are installed there, so
the fused route serves the run.  ``EVOGP_DEVICE_DEBUG=1`` re-raises instead of
falling back, which is how a genuine bug in this file stays visible.

Expected on the reference ROCm box: ~14 s with an empty Triton cache and ~13 s
with it warm (~7 s of either is the one-time ``evoxc.devices()`` probe, which
loads the iree runtime); past the graph compiles a generation is ~1.7 ms and
one evaluation ~0.66 ms at pop 256 over 512 rows (the host loop reads ~9.6 ms
per call).  Fitness numbers are identical between runs (``key=SEED`` carries
the RNG stream in the state); timings are not.  ``evogp.gp`` has no expression
exporter and no multi-output path yet (``MIGRATION.md`` steps 8/9), so the
winner is reported as a node count.
"""

from __future__ import annotations

import os

# evoxc records a call-site location per traced op; dropping that capture is a
# pure tracing-speed knob (~4x faster traces) and is not part of any compiled
# program or cache key.  Set it before anything could trace.
os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from evogp.gp import (  # noqa: E402
    FusedDeviceEvaluator,
    GPConfig,
    GPTreeConfig,
    GraphEvaluator,
    SymbolicRegression,
    fused_available,
    init_gp,
    interop_available,
    readback_fitness,
    row_length,
    run,
)


# --------------------------------------------------------------------------- #
# the knobs of the shipped run (repo style: plain module-level constants)
# --------------------------------------------------------------------------- #
POP_SIZE = 256  # trees per generation
GENERATIONS = 20  # evaluated generations
ROWS = 512  # data rows of the synthetic problem
SEED = 0  # the driver's key and the data seed (one integer, no global RNG)
MAX_NODES = 64  # tree-buffer width (the torch stack's ``max_tree_len``)
TIMED_CALLS = 10  # timed objective calls after the (untimed) warm-up call
REPORT_EVERY = 5  # print generation 0, every 5th one, and the last
BACKEND = "iree"  # the only device-capable evoxc backend
DEBUG_ENV = "EVOGP_DEVICE_DEBUG"  # set it to re-raise instead of falling back
DATA_BOUNDS = (-5.0, 5.0)  # the uniform data draw
INPUT_LEN = 2  # always ``x0``, ``x1``
USING_FUNCS = ("+", "-", "*", "/")  # the tree layer's function names
MAX_LAYER_CNT = 4  # depth schedule of the generation
LAYER_LEAF_PROB = 0.2  # leaf probability of every non-terminal depth
CONST_PROB = 0.5  # half the leaves are constants ...
CONST_RANGE = (-5.0, 5.0)  # ... drawn from this range,
SAMPLE_CNT = 8  # ... 8 values of it (the constant table)
ELITE_COUNT = 2  # 1 % elites (the torch stack's ``elite_rate=0.01``)
TOURNAMENT_SIZE = 4  # the pressure analogue of ``survival_rate=0.3``
CROSSOVER_PROB = 0.9  # per-individual crossover rate
MUTATION_PROB = 0.2  # per-individual subtree-mutation rate


def _device_kind() -> str | None:
    """``"rocm"`` / ``"cuda"`` when an iree device is usable here, else ``None``.

    The committed probe idiom of ``evogp.gp.tests``: the two ``iree`` packages
    (``iree`` is the only device-capable evoxc backend) and a device ``evoxc``
    enumerates, nothing compiled.  Any probing error means "not available
    here", so the caller falls back instead of failing.
    """
    if importlib.util.find_spec("iree.compiler") is None:
        return None
    if importlib.util.find_spec("iree.runtime") is None:
        return None
    try:
        import evoxc

        devices = evoxc.devices()
    except Exception:  # noqa: BLE001 - a probe answers "no device", never raises
        return None
    for kind in ("rocm", "cuda"):
        if any(device.kind == kind for device in devices):
            return kind
    return None


def _reason(exc: BaseException, width: int = 96) -> str:
    """The first line of an exception's message, clipped (iree dumps pages)."""
    lines = str(exc).strip().splitlines()
    line = lines[0].rstrip(":") if lines else ""
    return f"{line[:width]}..." if len(line) > width else line


def make_data(rows: int = ROWS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """``(x [rows, 2], y [rows])`` of ``x0**2 + x1**2``, float32."""
    generator = np.random.default_rng(seed)
    x = generator.uniform(*DATA_BOUNDS, size=(int(rows), INPUT_LEN))
    x = x.astype(np.float32)
    return x, (x[:, 0] ** 2 + x[:, 1] ** 2).astype(np.float32)


def make_configs() -> tuple[GPTreeConfig, GPConfig]:
    """The ``(GPTreeConfig, GPConfig)`` pair of this run."""
    tree_config = GPTreeConfig(
        max_nodes=MAX_NODES, input_len=INPUT_LEN, using_funcs=USING_FUNCS,
        max_layer_cnt=MAX_LAYER_CNT, layer_leaf_prob=LAYER_LEAF_PROB,
        const_prob=CONST_PROB, const_range=CONST_RANGE, sample_cnt=SAMPLE_CNT,
    )
    gp_config = GPConfig(
        pop_size=POP_SIZE, elite_count=ELITE_COUNT,
        tournament_size=TOURNAMENT_SIZE, crossover_prob=CROSSOVER_PROB,
        mutation_prob=MUTATION_PROB, mutation_kind="subtree",
    )
    return tree_config, gp_config


def _device_evaluator(x, y, tree_config, kind) -> tuple[str, object]:
    """``(route name, device objective)`` -- the fused kernel, else the graph.

    Both are contract-compatible ``evaluate=`` plug-ins returning the fitness on
    the device; the fused one additionally needs torch + triton, so it is
    offered only when its two probes agree, and the in-graph ``evoxc`` evaluator
    is the torch-free route for everything else.
    """
    if fused_available() and interop_available(kind):
        return "fused", FusedDeviceEvaluator(x, y)
    return "in-graph iree", GraphEvaluator(
        x, y, used_ids=tree_config, backend=BACKEND, device=kind
    )


def _time_calls(evaluate, state, calls: int = TIMED_CALLS):
    """``(median ms, min ms, first fitness)`` of ``calls`` evaluate calls.

    The first call is the warm-up and is not timed: on the fused route it pays
    the one-time Triton JIT (~1-2 s from an empty cache, ~2 ms from a warm disk
    cache), which belongs to the cold start rather than to the objective.  A
    device call is already synchronous from the host's point of view, so no
    cross-runtime fence is needed around it.
    """
    first = evaluate(state)
    samples = []
    for _ in range(max(1, int(calls))):
        started = time.perf_counter()
        evaluate(state)
        samples.append(1000.0 * (time.perf_counter() - started))
    samples.sort()
    return samples[len(samples) // 2], samples[0], first


def _reporter(generations: int, started: float, marks: list):
    """``run``'s callback: the best fitness on the report cadence, plus a mark."""

    def report(generation: int, _state, fitness) -> None:
        marks.append(time.perf_counter() - started)
        if generation % REPORT_EVERY and generation != generations - 1:
            return
        print(
            f"generation {generation:3d}  best {float(np.max(fitness)):+12.5f}  "
            f"{marks[-1]:6.2f}s",
            flush=True,
        )

    return report


def _report(result, generations: int, wall: float, first, marks) -> int:
    """Print the run's numbers and its winner; returns the exit code."""
    shape = tuple(int(value) for value in np.shape(first))
    values = readback_fitness(first, int(shape[0]))
    print(
        f"fitness vector {type(first).__name__} {shape} -> host {values.dtype} "
        f"{values.shape}, max {float(values.max()):+.6f}"
    )
    print(
        f"run           {generations} generations in {wall:.2f}s "
        f"({1000.0 * wall / max(generations, 1):.1f} ms/generation, "
        "first-generation compile included)"
    )
    deltas = np.diff(np.asarray(marks, dtype=np.float64)) * 1000.0
    if deltas.size > 1:
        print(
            f"steady state  {float(np.median(deltas[1:])):.2f} ms/generation "
            f"(median of {deltas.size - 1}, past the one-off graph compile)"
        )
    history = result.best_fitness_history
    if not history or result.best_channels is None:
        print("no finite individual was ever evaluated")
        return 1
    print(
        f"best fitness  {result.best_fitness:+.5f} at generation "
        f"{result.best_generation} of {len(history)} evaluated"
    )
    print(f"fitness path  {history[0]:+.5f} -> {history[-1]:+.5f}")
    print(f"winning tree  {int(row_length(result.best_channels)[0])} nodes")
    print("expression    none -- evogp.gp has no exporter (MIGRATION.md step 9)")
    return 0


def _run(route, evaluate, state, config, *, backend: str, device: str) -> int:
    """Time the objective, then run and report the loop of ``route``."""
    print(f"route         {route} (backend {backend}, device {device})")
    if route == "fused":
        print(
            "note          the first fused call pays a one-time Triton JIT (~1-2 s\n"
            "              cold, ~2 ms warm), so the warm-up call is excluded"
        )
    per_call = _time_calls(evaluate, state)
    print(
        f"evaluation    {per_call[0]:.4f} ms/call (median of {TIMED_CALLS} "
        f"calls, min {per_call[1]:.4f})"
    )
    started = time.perf_counter()
    marks: list = []
    result = run(
        config, evaluate, generations=GENERATIONS, key=SEED, backend=backend,
        device=device, callback=_reporter(GENERATIONS, started, marks),
    )
    return _report(result, GENERATIONS, time.perf_counter() - started,
                   per_call[2], marks)


def main(argv: list[str] | None = None) -> int:
    """Run the device example, or the host loop when no device is usable."""
    del argv  # no flags: the constants above are the shipped run
    x, y = make_data()
    config = make_configs()
    print(
        f"evogp.gp device-resident run: pop {POP_SIZE} x {GENERATIONS} "
        f"generations over {ROWS} rows (key {SEED})"
    )
    kind = _device_kind()
    if kind is not None:
        try:
            route, evaluator = _device_evaluator(x, y, config[0], kind)
            state = init_gp(config, SEED, backend=BACKEND, device=kind)
            return _run(route, evaluator, state, config, backend=BACKEND,
                        device=kind)
        except Exception as exc:  # noqa: BLE001 - the device route is an
            # optimisation and the host loop is the floor.  The reason is
            # printed below, so a declined route never looks like a success;
            # EVOGP_DEVICE_DEBUG=1 re-raises it instead (a real bug in this
            # file must not hide behind the fallback).
            if os.environ.get(DEBUG_ENV):
                raise
            print(f"device route failed ({type(exc).__name__}: {_reason(exc)})")
            print("no device: falling back to the numpy host loop")
    else:
        print(
            "no device: falling back to the numpy host loop (needs the iree "
            "packages and a rocm/cuda device)"
        )
    problem = SymbolicRegression(x, y)
    state = init_gp(config, SEED)
    return _run("numpy host", problem.evaluate, state, config, backend="numpy",
                device="cpu")


if __name__ == "__main__":
    sys.exit(main())
