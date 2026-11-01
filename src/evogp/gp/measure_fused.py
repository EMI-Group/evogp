"""Measurement harness: the optional FUSED route vs the in-graph ``evoxc`` route.

This is COMMITTED EVIDENCE, not a test and not a library entry point: it does
nothing at import time and everything inside :func:`main`, so the purity grep and
``pytest src/evogp/gp/tests`` see a plain module.

Run it as a module (relative imports, and ``evoxc`` must be on the path for the
in-graph side)::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \\
        /tmp/rocvenv/bin/python -m evogp.gp.measure_fused

It reports, for the workload the fused route is interesting at (``max_nodes=32``,
``input_len=8``, ``+ - * /``, ``max_layer_cnt=4``, ``N=1024``) and pop 64 / 256 /
1000:

=========================================== ====================================================
``fused cold``                              first call of the process (Triton JIT included)
``fused warm``                              kernel + ``(pop,)`` loss readback, channels resident
``upload``                                  numpy -> torch device of the three channel tables
``host channels``                           ``fused_eval.sr_fitness_fused_channels`` end to end
``evoxc -> host``                           the three evoxc DEVICE channel tensors -> numpy
``device channels``                         ``evoxc -> host`` + ``host channels`` (a device loop)
``in-graph evoxc``                          one device-resident ``evoxc.run`` + fitness readback
``host numpy``                              ``SymbolicRegression.evaluate`` (the pure-numpy path)
=========================================== ====================================================

and then the two verdicts that matter: the HOST-CHANNELS case (the population
tables are already in host memory, so the fused route only pays the upload) and
the DEVICE-CHANNELS case (the tables live on an evoxc device -- the only shape a
device-resident loop would have.  It is measured here the hard way, through a
host round trip, which is what the shipped ``FusedDeviceEvaluator`` no longer
pays: it hands the kernel its ``evoxc`` buffers over that engine's native DLPack
export (``Tensor.__dlpack__`` -> ``torch.from_dlpack``, the fitness buffer
written through ``out=``, one ``torch.cuda.synchronize()``), and it takes this
host hand-off only on an ``evoxc`` without the export).

Why the in-graph side is driven through ``problem``'s internals: the public
:func:`evogp.gp.problem.sr_fitness_graph` stages its graph inputs from host numpy
arrays, and ``evoxc.run`` refuses host tensors for a device executable
(``DeviceError: no implicit device-to-host or device-to-host transfer happens at
the run boundary``).  A device-resident comparison therefore has to build the
same executable (:func:`evogp.gp.problem._build_graph`) and keep the inputs on
the device itself; that is what :func:`_in_graph_runner` does, and the fact that
it is necessary is itself a finding.

What it showed (one MI300X/ROCm box, 2026-10-31; the full table is in
``CONTEXT.md`` under Known Issues): the kernel is flat in population at
~0.11 ms per call, the upload is 0.05-0.09 ms and the host decode 0.05-0.36 ms,
so the fused route is 1.3-1.6x the device-resident in-graph route when the
channel tables have to be brought home first, and 2.3-2.7x when they are already
in host memory -- which is why it ships as an opt-in evaluator rather than as a
driver path.
"""

from __future__ import annotations

import argparse
import importlib
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fused_eval
from .config import GPTreeConfig
from .encoding import _channel_arrays, init_population, to_legacy_triple
from .problem import SymbolicRegression, sr_fitness_graph

__all__ = [
    "WORKLOAD",
    "POPS",
    "make_data",
    "make_config",
    "measure_pop",
    "parity",
    "main",
]

#: The measurement workload (every knob of the tree generator, fixed on purpose).
WORKLOAD: Dict[str, Any] = {
    "max_nodes": 32,
    "input_len": 8,
    "using_funcs": ("+", "-", "*", "/"),
    "max_layer_cnt": 4,
    "const_prob": 0.5,
    "layer_leaf_prob": 0.2,
    "const_range": (-1.0, 1.0),
    "sample_cnt": 8,
}
#: Population sizes reported.
POPS: Tuple[int, ...] = (64, 256, 1000)
#: Data rows of the synthetic problem.
ROWS = 1024
#: RNG seed for the population and the data.
SEED = 0
#: Timed repetitions (medians) / untimed warm-up calls per column.
ITERS = 20
WARMUP = 5


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_data(rows: int = ROWS, input_len: int = 8, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """A smooth, finite 8-input target: a linear part plus one interaction."""
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(rows, input_len)).astype(np.float32)
    weights = np.zeros(input_len, np.float32)
    weights[: min(4, input_len)] = (2.0, -1.0, 0.5, 3.0)[: min(4, input_len)]
    y = x @ weights
    if input_len > 4:
        y = y + np.float32(0.5) * x[:, 0] * x[:, 4]
    return np.ascontiguousarray(x), np.ascontiguousarray(y, np.float32)


def make_config(**overrides: Any) -> GPTreeConfig:
    """The tree-genome config of the measurement workload (population size is not part of it)."""
    knobs = dict(WORKLOAD)
    knobs.update(overrides)
    return GPTreeConfig(**knobs)


def _iree_device_kind() -> str:
    """The evoxc/iree device kind matching this box's torch build."""
    torch = importlib.import_module("torch")
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def _median_ms(fn: Callable[[], Any], iters: int = ITERS, warmup: int = WARMUP) -> float:
    """Median wall time of ``fn`` in milliseconds (``warmup`` untimed calls)."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #
def parity(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Agreement of two fitness vectors on the comparable (finite) entries.

    ``max_rel`` is the pointwise relative delta (with a tiny floor, so a
    near-zero reference shows up honestly); ``max_rel_scaled`` is
    ``max|delta| / max|reference|``, the value-scale relative delta that a
    float32 kernel can actually promise; ``scale`` is that denominator (a
    population with one exploding tree makes the value-scale figure meaningless,
    so both numbers are reported).  ``finiteness_mismatch`` counts the entries
    where one route says "unevaluable" (``-inf``) and the other does not.
    """
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} vs {candidate.shape}"
        )
    both = np.isfinite(reference) & np.isfinite(candidate)
    reference_finite, candidate_finite = reference[both], candidate[both]
    scale = np.maximum(np.abs(reference_finite), np.float32(1e-30))
    delta = np.abs(candidate_finite - reference_finite)
    top = float(np.max(np.abs(reference_finite))) if reference_finite.size else 1.0
    return {
        "max_rel": float(np.max(delta / scale)) if delta.size else 0.0,
        "max_rel_scaled": float(np.max(delta) / top) if delta.size and top else 0.0,
        "scale": float(top),
        "comparable": int(np.count_nonzero(both)),
        "finiteness_mismatch": int(
            np.count_nonzero(np.isfinite(reference) != np.isfinite(candidate))
        ),
    }


# --------------------------------------------------------------------------- #
# the in-graph evoxc comparison (device-resident)
# --------------------------------------------------------------------------- #
def _in_graph_runner(
    channels,
    x: np.ndarray,
    y: np.ndarray,
    use_mse: bool = True,
) -> Callable[[], np.ndarray]:
    """A DEVICE-RESIDENT in-graph fitness callable over the same population.

    Builds the identical executable :func:`evogp.gp.problem.sr_fitness_graph`
    builds (same graph builder, same memo, same run tuple) but uploads the seven
    staged program tables plus ``x``/``y`` ONCE, so each call is exactly one
    ``evoxc.run`` plus the ``(pop,)`` readback -- the steady state of a
    device-resident loop, and the number the fused route must beat.
    """
    from . import problem as _problem

    evoxc = importlib.import_module("evoxc")
    opcode, payload, size = _channel_arrays(channels)
    arrays, used_ids = _problem._graph_inputs((opcode, payload, size), x)
    pop_size, width = opcode.shape
    n_rows, input_len = x.shape
    executable = _problem._build_graph(
        "fitness",
        width,
        pop_size,
        n_rows,
        input_len,
        used_ids,
        bool(use_mse),
        "iree",
        _iree_device_kind(),
    )
    config = (width, pop_size, n_rows, input_len, used_ids, bool(use_mse))
    device = evoxc.core.Device(_iree_device_kind(), 0)
    cpu = evoxc.core.Device("cpu", 0)
    resident = tuple(evoxc.from_numpy(array).to(device) for array in arrays)
    x_device = evoxc.from_numpy(x).to(device)
    y_device = evoxc.from_numpy(y).to(device)

    def run() -> np.ndarray:
        out = evoxc.run(executable, *resident, x_device, y_device, config)
        return np.array(out.to(cpu).numpy(), dtype=np.float32, copy=True)

    return run


def _device_channels(channels) -> Tuple[Any, ...]:
    """The three channel matrices as evoxc DEVICE tensors (a device loop's form)."""
    evoxc = importlib.import_module("evoxc")
    opcode, payload, size = _channel_arrays(channels)
    device = evoxc.core.Device(_iree_device_kind(), 0)
    return tuple(
        evoxc.from_numpy(np.ascontiguousarray(array)).to(device)
        for array in (opcode, payload, size)
    )


def _transfer_runner(device_tensors: Sequence[Any]) -> Callable[[], Tuple[np.ndarray, ...]]:
    """``Tensor.to(cpu).numpy()`` of the resident channel tensors."""
    evoxc = importlib.import_module("evoxc")
    cpu = evoxc.core.Device("cpu", 0)

    def transfer() -> Tuple[np.ndarray, ...]:
        return tuple(tensor.to(cpu).numpy() for tensor in device_tensors)

    return transfer


# --------------------------------------------------------------------------- #
# one measured column
# --------------------------------------------------------------------------- #
def measure_pop(
    pop_size: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    iters: int = ITERS,
    warmup: int = WARMUP,
    measure_cold: bool = True,
    graph: bool = True,
    numpy_graph: bool = False,
    max_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Every timed column of one population size."""
    torch = importlib.import_module("torch")
    engine = importlib.import_module("evogp.fused.engine")
    config = make_config(**({} if max_nodes is None else {"max_nodes": max_nodes}))
    state = init_population(config, SEED, pop_size=pop_size, backend="numpy")
    reference = SymbolicRegression(x, y)
    expected = reference.evaluate(state)

    record: Dict[str, Any] = {"pop": int(pop_size), "input_len": int(x.shape[1])}
    record["cold"] = (
        _median_ms(
            lambda: fused_eval.sr_fitness_fused_channels(state, x, y),
            iters=1,
            warmup=0,
        )
        if measure_cold
        else None
    )
    record["fused_parity"] = parity(
        expected, fused_eval.sr_fitness_fused_channels(state, x, y)
    )
    # columns 1-3: what a host-channels caller pays
    value, node_type, subtree_size = to_legacy_triple(state)
    device = torch.device("cuda")

    def upload():
        return (
            torch.as_tensor(value, device=device),
            torch.as_tensor(node_type, device=device),
            torch.as_tensor(subtree_size, device=device),
        )

    record["upload"] = _median_ms(upload, iters, warmup)
    population = upload()
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y[:, None], device=device)

    def fused():
        loss = engine.sr_fitness_fused(
            population[0], population[1], population[2], x.shape[1], 1,
            x_tensor, y_tensor, use_MSE=True,
        )
        return (-loss).detach().to(device="cpu", dtype=torch.float32).numpy()

    record["fused"] = _median_ms(fused, iters, warmup)
    record["host_channels"] = _median_ms(
        lambda: fused_eval.sr_fitness_fused_channels(state, x, y), iters, warmup
    )
    evaluator = fused_eval.FusedEvaluator(x, y)
    record["evaluator"] = _median_ms(lambda: evaluator(state), iters, warmup)
    record["host_numpy"] = _median_ms(lambda: reference.evaluate(state), iters, warmup)
    # columns 4-6: what a DEVICE-channels caller pays (a device-resident loop).
    # ``evaluator`` is the steady state of a loop -- the data tensors are already
    # resident -- so the device totals are built from it, not from ``host_channels``
    # (which re-uploads x/y on every call, the stateless-function figure).
    tensors = _device_channels(state)
    transfer = _transfer_runner(tensors)
    record["transfer"] = _median_ms(transfer, iters, warmup)
    record["device_channels"] = record["transfer"] + record["evaluator"]
    if graph:
        in_graph = _in_graph_runner(state, x, y)
        record["in_graph_parity"] = parity(expected, in_graph())
        record["in_graph"] = _median_ms(in_graph, iters, warmup)
        # The same transfer measured with the device queue BUSY (a real loop
        # reads the tables back right after the operator graph that wrote them),
        # which is what the parent's 0.57/0.70/0.97 ms transfer figures reflect.
        record["transfer_queued"] = _median_ms(
            lambda: (in_graph(), transfer()), iters, warmup
        ) - record["in_graph"]
        record["device_channels_queued"] = (
            record["transfer_queued"] + record["evaluator"]
        )
    else:
        record["in_graph"] = None
        record["in_graph_parity"] = None
        record["transfer_queued"] = None
        record["device_channels_queued"] = None
    if numpy_graph:
        record["numpy_graph"] = _median_ms(
            lambda: sr_fitness_graph(state, x, y, backend="numpy"),
            iters=3,
            warmup=1,
        )
    else:
        record["numpy_graph"] = None
    return record


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


_HEAD = (
    "| pop | fused cold | fused warm | upload | host channels | evaluator "
    "| evoxc->host | device channels | in-graph evoxc | host numpy |"
)
_RULE = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _table(records: Sequence[Dict[str, Any]]) -> str:
    lines = [_HEAD, _RULE]
    for record in records:
        lines.append(
            "| {pop} | {cold} | {fused} | {upload} | {host_channels} | {evaluator} "
            "| {transfer} | {device_channels} | {in_graph} | {host_numpy} |".format(
                pop=record["pop"],
                cold=_ms(record["cold"]),
                fused=_ms(record["fused"]),
                upload=_ms(record["upload"]),
                host_channels=_ms(record["host_channels"]),
                evaluator=_ms(record["evaluator"]),
                transfer=_ms(record["transfer"]),
                device_channels=_ms(record["device_channels"]),
                in_graph=_ms(record["in_graph"]),
                host_numpy=_ms(record["host_numpy"]),
            )
        )
    lines.append("")
    lines.append(
        "all times in ms per call (medians of the timed repetitions); "
        "``host channels`` re-uploads x/y every call, ``evaluator`` keeps them resident"
    )
    lines.append("")
    lines.append(
        "| pop | parity max rel | parity rel-to-scale | value scale "
        "| finiteness mismatch | in-graph parity rel-to-scale |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for record in records:
        fused_parity = record["fused_parity"]
        graph_parity = record["in_graph_parity"]
        lines.append(
            "| {pop} | {mr:.2e} | {mrs:.2e} | {sc:.2e} | {mm} | {grs} |".format(
                pop=record["pop"],
                mr=fused_parity["max_rel"],
                mrs=fused_parity["max_rel_scaled"],
                sc=fused_parity["scale"],
                mm=fused_parity["finiteness_mismatch"],
                grs=(
                    f"{graph_parity['max_rel_scaled']:.2e}"
                    if graph_parity
                    else "-"
                ),
            )
        )
    return "\n".join(lines)


def _verdicts(records: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for record in records:
        graph = record["in_graph"]
        if graph is None:
            continue
        host = record["evaluator"]
        device = record["device_channels"]
        queued = record["device_channels_queued"]
        lines.append(
            "pop {pop:4d}: HOST channels  fused {host:6.3f} vs in-graph {graph:6.3f} ms "
            "-> fused {hs:5.2f}x | DEVICE channels fused {device:6.3f} "
            "(queued {queued:6.3f}) vs {graph:6.3f} ms -> {ds:5.2f}x (queued {qs:5.2f}x)".format(
                pop=record["pop"],
                host=host,
                graph=graph,
                hs=graph / host,
                device=device,
                queued=queued,
                ds=graph / device,
                qs=graph / queued if queued else float("nan"),
            )
        )
    if not lines:
        lines.append("no in-graph column measured (--no-graph)")
    lines.append("")
    lines.append(
        "HOST channels = the population tables are already in host memory (what the "
        "numpy-backed operator layer hands an evaluator);"
    )
    lines.append(
        "DEVICE channels = the tables live on an evoxc device, so the fused route has "
        "to bring them home first (evoxc cannot export a device buffer to torch)."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the acceptance check (parity against the numpy oracle)
# --------------------------------------------------------------------------- #
#: Relative tolerance the adapter's parity is accepted at (the fused route is a
#: float32 kernel against a float32 numpy oracle; measured ~1e-7).
PARITY_TOLERANCE = 1e-4


def check(
    pop: int = 32,
    rows: int = 256,
    generations: int = 3,
    seed: int = 0,
    *,
    max_nodes: Optional[int] = None,
) -> int:
    """Parity of the fused adapter against ``SymbolicRegression`` (0 = pass, 1 = fail).

    Three comparisons, all cheap (CPU-side apart from the kernel):

    1. a freshly generated population;
    2. the SAME population after ``generations`` driver steps, so the operator
       layer's stale padding and the elite copies are in play;
    3. a short ``run`` driven by each evaluator in turn -- the two
       ``best_fitness_history`` trajectories, which is what "the adapter is a
       drop-in ``evaluate`` callable" means in practice.

    (1) and (2) are the acceptance criterion: every relative delta must stay
    inside :data:`PARITY_TOLERANCE` and no entry may change finiteness.  (3) is
    reported, not enforced -- a 1e-7 fitness difference could in principle
    reorder a tournament, so a diverging trajectory would be a finding, not a
    contract violation.
    """
    from .config import GPConfig
    from .workflow import init_gp, run as run_gp, step

    tree = make_config(**({} if max_nodes is None else {"max_nodes": max_nodes}))
    config = (
        tree,
        GPConfig(pop_size=pop, elite_count=2, tournament_size=4,
                 crossover_prob=0.9, mutation_prob=0.2),
    )
    x, y = make_data(rows, tree.input_len, seed)
    host = SymbolicRegression(x, y)
    fused = fused_eval.FusedEvaluator(x, y)
    status = 0

    def compare(label: str, state) -> None:
        nonlocal status
        expected = host.evaluate(state)
        observed = fused(state)
        report = parity(expected, observed)
        ok = (
            report["max_rel"] <= PARITY_TOLERANCE
            and report["finiteness_mismatch"] == 0
        )
        status = status or (0 if ok else 1)
        print(
            f"{label:<28} comparable {report['comparable']:>4}/{expected.size:<4} "
            f"max rel {report['max_rel']:.3e} rel-to-scale "
            f"{report['max_rel_scaled']:.3e} mismatch "
            f"{report['finiteness_mismatch']} -> {'OK' if ok else 'FAIL'}"
        )

    compare("fresh population", init_gp(config, seed))
    state = init_gp(config, seed)
    for _ in range(max(0, int(generations))):
        state = step(config, state, host)
    compare(f"after {generations} driver steps", state)

    fused_hist = run_gp(config, fused, generations=max(1, int(generations)),
                        key=seed).best_fitness_history
    host_hist = run_gp(config, host, generations=max(1, int(generations)),
                       key=seed).best_fitness_history
    agree = bool(np.allclose(fused_hist, host_hist, rtol=PARITY_TOLERANCE, atol=0.0))
    print(
        f"{'driver trajectory':<28} fused {[round(float(v), 4) for v in fused_hist]}\n"
        f"{'':<28} host  {[round(float(v), 4) for v in host_hist]} -> "
        f"{'agree' if agree else 'DIVERGED'}"
    )
    return status


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Measure and print; returns a shell exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m evogp.gp.measure_fused",
        description="FUSED (Triton) vs in-graph evoxc evaluation of evogp.gp populations.",
    )
    parser.add_argument(
        "--pops", default=",".join(str(p) for p in POPS),
        help="comma-separated population sizes (default: 64,256,1000)",
    )
    parser.add_argument("--rows", type=int, default=ROWS, help="data rows (default 1024)")
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 0)")
    parser.add_argument("--iters", type=int, default=ITERS, help="timed repetitions")
    parser.add_argument(
        "--max-nodes", type=int, default=int(WORKLOAD["max_nodes"]),
        help="tree-buffer width (default 32)",
    )
    parser.add_argument(
        "--no-graph", action="store_true",
        help="skip the in-graph evoxc comparison (the evoxc device-transfer column "
        "is still measured, so evoxc stays required)",
    )
    parser.add_argument(
        "--numpy-graph", action="store_true",
        help="also time the numpy-backend in-graph route (slow: ~2 s/call at pop 1000)",
    )
    parser.add_argument(
        "--json", default=None,
        help="write the raw records (every measured column) to this path",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="run the parity acceptance check instead of the timing columns",
    )
    args = parser.parse_args(argv)

    pops = [int(part) for part in str(args.pops).split(",") if part.strip()]
    max_nodes = int(args.max_nodes)

    if args.check:
        if not fused_eval.fused_available():
            print(
                "the fused route is NOT available on this box (needs torch with a "
                "CUDA/HIP device, triton, and a writable TRITON_CACHE_DIR)"
            )
            return 1
        print("# fused adapter parity vs SymbolicRegression (numpy oracle)")
        print(f"triton cache dir: {fused_eval.ensure_triton_cache_dir()}")
        return check(
            pop=pops[0] if pops else 32,
            rows=int(args.rows),
            generations=3,
            seed=int(args.seed),
            max_nodes=max_nodes,
        )

    x, y = make_data(args.rows, int(WORKLOAD["input_len"]), args.seed)

    print("# FUSED route vs the in-graph evoxc route (evogp.gp)")
    print()
    print(
        f"workload: max_nodes={max_nodes} input_len={WORKLOAD['input_len']} "
        f"funcs={' '.join(WORKLOAD['using_funcs'])} max_layer_cnt={WORKLOAD['max_layer_cnt']} "
        f"const_prob={WORKLOAD['const_prob']} layer_leaf_prob={WORKLOAD['layer_leaf_prob']} "
        f"rows={args.rows} pops={','.join(str(p) for p in pops)}"
    )
    if not fused_eval.fused_available():
        print()
        print(
            "the fused route is NOT available on this box (needs torch with a "
            "CUDA/HIP device, triton, and a writable TRITON_CACHE_DIR) -- nothing "
            "to measure"
        )
        return 1
    print()
    print(f"triton cache dir: {fused_eval.ensure_triton_cache_dir()}")

    records = []
    for index, pop in enumerate(pops):
        record = measure_pop(
            pop,
            x,
            y,
            iters=max(1, int(args.iters)),
            measure_cold=index == 0,
            graph=not args.no_graph,
            numpy_graph=bool(args.numpy_graph),
            max_nodes=max_nodes,
        )
        records.append(record)
        print(f"measured pop {pop}", flush=True)

    print()
    print(_table(records))
    print()
    print("## verdict")
    print()
    print(_verdicts(records))
    if args.json:
        import json

        payload = {
            "workload": dict(WORKLOAD, max_nodes=max_nodes),
            "rows": int(args.rows),
            "seed": int(args.seed),
            "iters": int(args.iters),
            "records": records,
            "triton_cache_dir": fused_eval.ensure_triton_cache_dir(),
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=float)
            handle.write("\n")
        print()
        print(f"records written to {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a harness, never imported for effect
    raise SystemExit(main())
