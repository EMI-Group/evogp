"""Driver tests: the DEVICE-RESIDENT run (``init_gp`` / ``step`` / ``run``).

``workflow`` drives both routes with the same functions: a population of host
arrays with a numpy objective is the loop the rest of the suite pins, and a
population of ``evoxc`` tensors ON A DEVICE with a device objective
(:class:`evogp.gp.graph_eval.GraphEvaluator`) is the device loop added here.
What this module pins, in order:

* ``init_gp(..., backend="iree", device=<device>)`` returns a state whose every
  leaf (channels, fitness, key) lives on that device, and whose generation-0
  population is bit-identical to the host run's for the same seed -- the
  encoding graph is the same program on both routes;
* one device ``step`` (``GraphEvaluator`` as ``evaluate``) produces the SAME
  population as the numpy ``step``: integer channels bitwise, float payloads
  inside 1e-6, the elite rows the parent's own top-``elite_count`` rows and the
  offspring marked ``-inf`` -- i.e. the driver's device path is a transport
  change, never a semantic one;
* the budget: exactly ONE host readback per generation (the fitness vector the
  elite search and the history share) and no channel download at all inside the
  loop -- the elite splice runs as a traced graph on the device;
* ``run`` on the device reproduces the numpy trajectory's per-generation best
  fitness inside 1e-5, reports ``best_channels`` as HOST rows and leaves
  ``result.state`` device-resident;
* the package surface exposes the device pieces (``GraphEvaluator``,
  ``sr_fitness_channels``, ``readback_fitness``, ``FusedEvaluator``,
  ``fused_available``) and the cache surface covers the device layer.

Everything is gated on a genuinely usable device: the module skips (never
fails) on a box without ``iree`` or without a visible ``rocm``/``cuda`` device,
and the tests pass ``backend="iree"`` + an explicit ``evoxc.Device`` to every
call because the suite's ``conftest.py`` pins the numpy backend for hermeticity.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

evoxc = pytest.importorskip(
    "evoxc", reason="the device driver is built on evoxc"
)

from evogp.gp import (
    GPConfig,
    GPTreeConfig,
    GraphEvaluator,
    SymbolicRegression,
    cache_stats,
    clear_cache,
    init_gp,
    run,
    smoke,
    step,
)
from evogp.gp import encoding, graph_eval

# --------------------------------------------------------------------------- #
# availability gate: a real device behind the iree backend, or skip
# --------------------------------------------------------------------------- #
def _usable_device_kind():
    """``"rocm"`` / ``"cuda"`` when an iree-driven device is usable, else ``None``.

    Three conditions, all probed without compiling anything: the ``iree``
    compiler and runtime Python packages (the ``iree`` backend is the only one
    that runs on a device), and a device ``evoxc`` itself enumerates.  The probe
    is deliberately lenient -- any probing error means "not available here", so
    the module skips instead of failing on a host where one half is missing.
    """
    if importlib.util.find_spec("iree.compiler") is None:
        return None
    if importlib.util.find_spec("iree.runtime") is None:
        return None
    try:
        devices = evoxc.devices()
    except Exception:  # noqa: BLE001 - a probe must never fail the collection
        return None
    for kind in ("rocm", "cuda"):
        if any(device.kind == kind for device in devices):
            return kind
    return None


DEVICE_KIND = _usable_device_kind()

pytestmark = pytest.mark.skipif(
    DEVICE_KIND is None,
    reason=(
        "no evoxc device to run on: the driver's device path needs the iree "
        "compiler + runtime and a visible rocm/cuda device"
    ),
)

#: The run this module pins.  Small on purpose (one compile per graph) and
#: fixed, so the cross-route comparisons below are deterministic.
SEED = 0
NUM_DATA = 64
POP_SIZE = 16
ELITE_COUNT = 2
TOL = 1e-6


def _host(value) -> np.ndarray:
    """One channel/fitness tensor as a host numpy array (an explicit download)."""
    return encoding._as_numpy(value, "channel")


def _is_device_tensor(value) -> bool:
    """``True`` for a concrete ``evoxc`` tensor that lives on a device."""
    return isinstance(value, evoxc.Tensor) and value.device.kind != "cpu"


# --------------------------------------------------------------------------- #
# fixtures (module scoped: one compile per graph for the whole module)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    """The device under test (an ``evoxc.Device``, never a bare string)."""
    return evoxc.Device(DEVICE_KIND, 0)


@pytest.fixture(scope="module")
def tree_config():
    return GPTreeConfig(
        max_nodes=16,
        input_len=2,
        using_funcs=("+", "-", "*", "/"),
        max_layer_cnt=4,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
    )


@pytest.fixture(scope="module")
def run_config(tree_config):
    return (tree_config, GPConfig(pop_size=POP_SIZE, elite_count=ELITE_COUNT))


@pytest.fixture(scope="module")
def data():
    """``(x, y)`` of the smoke SR problem, the same data both routes evaluate."""
    return smoke.make_data(NUM_DATA, seed=SEED)


@pytest.fixture(scope="module")
def host_state(run_config):
    """Generation 0 of the host (numpy) run."""
    return init_gp(run_config, SEED)


@pytest.fixture(scope="module")
def device_state(run_config, device):
    """Generation 0 of the device run -- the same seed, hence the same trees."""
    return init_gp(run_config, SEED, backend="iree", device=device)


@pytest.fixture(scope="module")
def evaluator(data, tree_config, device):
    """The device objective: in-graph SR fitness, device tensor out, no sync."""
    x, y = data
    return GraphEvaluator(x, y, used_ids=tree_config, backend="iree", device=device)


# --------------------------------------------------------------------------- #
# init: the population starts on the device
# --------------------------------------------------------------------------- #
def test_init_gp_puts_every_leaf_of_the_state_on_the_device(
    device_state, host_state, device
):
    for name in ("opcode", "payload", "size", "fitness", "key"):
        leaf = getattr(device_state, name)
        assert _is_device_tensor(leaf), f"{name} must live on the device"
        assert leaf.device == device
    # Same seed, same program: the encoding graph is shared by both routes, so
    # generation 0 is bit-identical (only its location differs).
    for name in ("opcode", "payload", "size"):
        np.testing.assert_array_equal(
            _host(getattr(device_state, name)), _host(getattr(host_state, name))
        )
    np.testing.assert_array_equal(
        _host(device_state.key), _host(host_state.key)
    )
    assert np.all(np.isneginf(_host(device_state.fitness)))


# --------------------------------------------------------------------------- #
# step: the same population as the numpy step, elites included
# --------------------------------------------------------------------------- #
def test_a_device_step_reproduces_the_numpy_step(
    run_config, host_state, device_state, evaluator, data, device
):
    x, y = data
    problem = SymbolicRegression(x, y)
    numpy_child = step(run_config, host_state, problem.evaluate)
    device_child = step(
        run_config, device_state, evaluator, backend="iree", device=device
    )

    # Integer channels are bitwise: the splice/selection/mutation graph is the
    # same program, and the two evaluators agree far more closely than any
    # selection margin in this fixture.
    for name in ("opcode", "size"):
        np.testing.assert_array_equal(
            _host(getattr(device_child, name)), _host(getattr(numpy_child, name))
        )
    np.testing.assert_allclose(
        _host(device_child.payload),
        _host(numpy_child.payload),
        rtol=0.0,
        atol=TOL,
        err_msg="float payloads must agree inside 1e-6 across the two routes",
    )
    # And the device population is still a device population.
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert _is_device_tensor(getattr(device_child, name))


def test_the_elite_rows_of_a_device_step_are_the_parents_own_top_rows(
    run_config, device_state, evaluator, device
):
    fitness = evaluator.host_fitness(evaluator.evaluate(device_state))
    device_child = step(
        run_config, device_state, evaluator, backend="iree", device=device
    )

    # `-fitness` + a stable argsort: the driver's own rule for the elite search.
    elite = np.argsort(-fitness, kind="stable")[:ELITE_COUNT]
    names = ("opcode", "payload", "size")
    parent = {name: _host(getattr(device_state, name)) for name in names}
    child = {name: _host(getattr(device_child, name)) for name in names}
    for slot, source in enumerate(elite):
        for name in names:
            np.testing.assert_array_equal(
                child[name][slot], parent[name][source],
                err_msg=f"elite slot {slot} must be a bit-identical parent row",
            )

    recorded = _host(device_child.fitness)
    np.testing.assert_allclose(recorded[:ELITE_COUNT], fitness[elite], rtol=0, atol=TOL)
    assert np.all(np.isneginf(recorded[ELITE_COUNT:])), (
        "offspring are unevaluated until the next evaluate call"
    )


def test_a_host_objective_can_drive_a_device_population(
    run_config, device_state, data, device
):
    """The fitness KIND decides the transport, not the caller.

    A numpy-returning objective (the problem layer's host evaluator) on a
    device-resident population still works: the channels stay on the device, only
    the ``pop_size``-element fitness vector is uploaded, and the result is a
    device state again.
    """
    x, y = data
    device_child = step(
        run_config,
        device_state,
        SymbolicRegression(x, y).evaluate,
        backend="iree",
        device=device,
    )
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert _is_device_tensor(getattr(device_child, name))
    recorded = _host(device_child.fitness)
    assert np.all(np.isfinite(recorded[:ELITE_COUNT]))
    assert np.all(np.isneginf(recorded[ELITE_COUNT:]))


# --------------------------------------------------------------------------- #
# the budget: one readback per generation, no channel download in the loop
# --------------------------------------------------------------------------- #
def test_a_device_step_reads_back_exactly_one_fitness_vector(
    run_config, device_state, evaluator, device, monkeypatch
):
    readbacks = []
    downloads = []
    real_readback = graph_eval.readback_fitness
    real_as_numpy = encoding._as_numpy

    def counting_readback(fitness, pop_size=None):
        readbacks.append(1)
        return real_readback(fitness, pop_size)

    def counting_as_numpy(value, name):
        if _is_device_tensor(value):
            downloads.append(name)
        return real_as_numpy(value, name)

    monkeypatch.setattr(graph_eval, "readback_fitness", counting_readback)
    monkeypatch.setattr(encoding, "_as_numpy", counting_as_numpy)

    step(run_config, device_state, evaluator, backend="iree", device=device)

    assert len(readbacks) == 1, (
        "a generation has exactly one host readback -- the fitness vector the "
        f"elite search and the history share (got {len(readbacks)})"
    )
    assert downloads == [], (
        "no channel may leave the device inside a generation (the elite splice "
        f"runs on the device); got downloads of {downloads}"
    )


def test_a_device_run_reads_back_one_vector_per_generation(
    run_config, evaluator, device, monkeypatch
):
    generations = 2
    readbacks = []
    downloads = []
    real_readback = graph_eval.readback_fitness
    real_as_numpy = encoding._as_numpy

    def counting_readback(fitness, pop_size=None):
        readbacks.append(1)
        return real_readback(fitness, pop_size)

    def counting_as_numpy(value, name):
        if _is_device_tensor(value):
            downloads.append(name)
        return real_as_numpy(value, name)

    monkeypatch.setattr(graph_eval, "readback_fitness", counting_readback)
    monkeypatch.setattr(encoding, "_as_numpy", counting_as_numpy)

    run(
        run_config,
        evaluator,
        generations=generations,
        key=SEED,
        backend="iree",
        device=device,
    )

    assert len(readbacks) == generations
    # Two channel round trips, neither of them inside the loop: `init_gp`'s
    # host-side validation of the generation-0 population, and the final
    # `GPResult.best_channels` report (one row download).
    assert downloads == ["opcode", "payload", "size"] * 2, downloads


# --------------------------------------------------------------------------- #
# run: the same trajectory, host rows in the result, state still on the device
# --------------------------------------------------------------------------- #
def test_a_device_run_reproduces_the_numpy_trajectory(
    run_config, evaluator, data, device
):
    generations = 3
    x, y = data
    device_result = run(
        run_config,
        evaluator,
        generations=generations,
        key=SEED,
        backend="iree",
        device=device,
    )
    host_result = run(
        run_config, SymbolicRegression(x, y).evaluate, generations=generations, key=SEED
    )

    assert device_result.generations == generations
    assert len(device_result.best_fitness_history) == generations
    np.testing.assert_allclose(
        np.asarray(device_result.best_fitness_history),
        np.asarray(host_result.best_fitness_history),
        rtol=0.0,
        atol=1e-5,
        err_msg="the two routes must follow the same trajectory",
    )
    assert np.all(np.diff(device_result.best_fitness_history) >= 0.0), (
        "elitism keeps the per-generation best monotone on the device too"
    )

    # The report is HOST data, whatever the population is made of...
    opcode, payload, size = device_result.best_channels
    for row, dtype in ((opcode, np.int32), (payload, np.float32), (size, np.int32)):
        assert isinstance(row, np.ndarray)
        assert row.dtype == dtype
        assert row.shape == (run_config[0].max_nodes,)
    # ...and re-evaluating it reproduces the reported fitness.
    replayed = SymbolicRegression(x, y).evaluate(
        (opcode[None, :], payload[None, :], size[None, :])
    )
    np.testing.assert_allclose(
        np.asarray(replayed, dtype=np.float32),
        np.float32(device_result.best_fitness),
        rtol=0.0,
        atol=1e-4,
    )
    # The state stays device-resident for whoever continues the run.
    for name in ("opcode", "payload", "size", "fitness", "key"):
        assert _is_device_tensor(getattr(device_result.state, name))


# --------------------------------------------------------------------------- #
# the package + cache surface
# --------------------------------------------------------------------------- #
DEVICE_SURFACE = (
    "GraphEvaluator",
    "sr_fitness_channels",
    "readback_fitness",
    "FusedEvaluator",
    "fused_available",
)


def test_the_package_exposes_the_device_surface():
    import evogp.gp as gp

    missing = [name for name in DEVICE_SURFACE if name not in gp.__all__]
    assert not missing, f"evogp.gp does not export {missing}"
    for name in DEVICE_SURFACE:
        assert hasattr(gp, name)
    # The documented spellings still hold.
    assert gp.apply_variation is gp.variation.variation
    assert gp.apply_crossover is gp.crossover.crossover


def test_the_cache_surface_covers_the_device_layer(run_config, evaluator, device):
    clear_cache()
    run(
        run_config,
        evaluator,
        generations=1,
        key=SEED,
        backend="iree",
        device=device,
    )
    stats = cache_stats()
    assert "graph_eval" in stats["device_layers"]
    assert stats["device_layers"]["graph_eval"]["graphs"] > 0, (
        "a device run memoizes its evaluation graph"
    )
    assert stats["device_graphs"] == sum(
        layer["graphs"] for layer in stats["device_layers"].values()
    )
    assert set(stats["layers"]) == {
        "encoding",
        "selection",
        "crossover",
        "mutation",
        "variation",
        "workflow",
        "problem",
    }, "the host layer table keeps its seven entries"

    clear_cache()
    assert cache_stats()["device_graphs"] == 0
