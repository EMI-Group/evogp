"""Convergence test: a SHORT real SR run on the DEVICE path.

The rest of the device suite pins single calls (``init_population``,
``variation``, ``GraphEvaluator``, one ``step``).  This module pins the thing
those calls exist for: a device-driven ``run`` that actually CONVERGES, and does
so along the same trajectory as the numpy route.

The workload is the ``evogp.gp.smoke`` problem (``x0**4/(x0**4+1) +
x1**4/(x1**4+1)``, :func:`evogp.gp.smoke.make_data` /
:func:`evogp.gp.smoke.make_configs`), shrunk to a short run -- population 150,
18 evaluated generations, 256 data rows, 32-node trees -- so the test costs
~9 s here (three iree compiles dominate; the generations themselves are
milliseconds) while still showing real progress:

    generation-0 best -0.27158 -> -0.24199 (best constant predictor -0.27158)

What is asserted, and why it can be asserted at all:

* the best fitness IMPROVES over the run and lands in the same ballpark as the
  numpy path (the last generation is <= -0.20 on this fixture);
* the device trajectory agrees with the numpy one generation by generation
  (``rtol=0``, ``atol=1e-5``) -- measured 1.49e-08 on this box.  The two routes
  run the SAME traced programs and the same ``splitmix64`` key stream, so the
  only source of difference is the device's float lowering of the SR loss
  inside the evaluation graph (iree contracts multiply-add): the truncation is
  far below any selection margin, so the populations and their rankings stay in
  step;
* the run reports a replayable best tree and leaves ``result.state`` on the
  device, i.e. the loop ends where the next generation would start.

The module is DEVICE-GATED: it skips (never fails) without ``iree`` or without a
visible ``rocm``/``cuda`` device, and passes ``backend="iree"`` + an explicit
``evoxc.Device`` because the suite's ``conftest.py`` pins the numpy backend.
Nothing here touches torch or the fused route, and the module builds exactly
three graphs (encoding, variation, in-graph evaluation) for one shape.
"""

from __future__ import annotations

import numpy as np
import pytest

evoxc = pytest.importorskip(
    "evoxc", reason="the device path is built on evoxc"
)

from evogp.gp import GraphEvaluator, SymbolicRegression, run, smoke

from gp_operator_support import DEVICE_SKIP_REASON, is_device_tensor, usable_device_kind

DEVICE_KIND = usable_device_kind()

pytestmark = pytest.mark.skipif(DEVICE_KIND is None, reason=DEVICE_SKIP_REASON)

#: The run this module pins -- short, fixed and small enough for one compile per
#: graph, but long enough that the elite search actually finds something.
SEED = 0
POP_SIZE = 150
GENERATIONS = 18
NUM_DATA = 256
MAX_NODES = 32
#: ``(pop 150 x 18 generations)`` on this fixture: the best constant predictor
#: scores -0.27158 and the run reaches -0.24199, so this bound is a ballpark with
#: real slack that still fails if the device loop never improves anything.
BEST_FITNESS_CEILING = -0.20
#: Cross-route agreement of the per-generation best (measured 1.49e-08).
TRAJECTORY_TOL = 1e-5


# --------------------------------------------------------------------------- #
# fixtures (module scoped: the run happens once, the tests read its report)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    """The device under test (an ``evoxc.Device``, never a bare string)."""
    return evoxc.Device(DEVICE_KIND, 0)


@pytest.fixture(scope="module")
def config():
    """The ``smoke`` problem's config pair, shrunk to this module's budget."""
    return smoke.make_configs(pop_size=POP_SIZE, max_nodes=MAX_NODES)


@pytest.fixture(scope="module")
def data():
    """``(x, y)`` of the smoke SR problem (the same target, fewer rows)."""
    return smoke.make_data(NUM_DATA, seed=SEED)


@pytest.fixture(scope="module")
def device_result(config, data, device):
    """The device run: in-graph evaluation, device-resident operator layer."""
    x, y = data
    tree_config = config[0]
    evaluator = GraphEvaluator(
        x, y, used_ids=tree_config, backend="iree", device=device
    )
    return run(
        config,
        evaluator,
        generations=GENERATIONS,
        key=SEED,
        backend="iree",
        device=device,
    )


@pytest.fixture(scope="module")
def host_result(config, data):
    """The numpy run of the same seed -- the reference trajectory."""
    x, y = data
    return run(
        config, SymbolicRegression(x, y).evaluate, generations=GENERATIONS, key=SEED
    )


# --------------------------------------------------------------------------- #
# convergence, and the trajectory it follows
# --------------------------------------------------------------------------- #
def test_a_short_device_run_converges_like_the_numpy_run(device_result, host_result):
    assert device_result.generations == GENERATIONS
    device_history = np.asarray(device_result.best_fitness_history)
    host_history = np.asarray(host_result.best_fitness_history)
    assert device_history.shape == (GENERATIONS,)
    assert host_history.shape == (GENERATIONS,)

    # (a) It converges: elitism makes the per-generation best monotone, and the
    # run must actually improve on generation 0 and end in the numpy ballpark.
    assert np.all(np.diff(device_history) >= 0.0), (
        f"the per-generation best must never regress, got {device_history}"
    )
    assert device_history[-1] > device_history[0], (
        "the device run must improve over its own generation 0: "
        f"{device_history[0]} -> {device_history[-1]}"
    )
    assert device_history[-1] <= BEST_FITNESS_CEILING, (
        f"the device run must land in the expected ballpark (<="
        f"{BEST_FITNESS_CEILING}), got {device_history[-1]}"
    )

    # (b) Generation by generation, it is the numpy run: same programs, same key
    # stream, same ranking -- only the device's float lowering differs.
    np.testing.assert_allclose(
        device_history,
        host_history,
        rtol=0.0,
        atol=TRAJECTORY_TOL,
        err_msg="the device trajectory must follow the numpy trajectory",
    )
    assert abs(device_result.best_fitness - host_result.best_fitness) <= TRAJECTORY_TOL
    assert device_result.best_generation == host_result.best_generation, (
        "both routes must find their best tree in the same generation: "
        f"{device_result.best_generation} vs {host_result.best_generation}"
    )


def test_the_device_run_reports_a_replayable_best_tree(device_result, data, device):
    """The loop's report is usable host data, and its state stays on the device."""
    x, y = data
    opcode, payload, size = device_result.best_channels
    for row, dtype, wanted in (
        (opcode, np.int32, MAX_NODES),
        (payload, np.float32, MAX_NODES),
        (size, np.int32, MAX_NODES),
    ):
        assert isinstance(row, np.ndarray)
        assert row.dtype == np.dtype(dtype)
        assert row.shape == (wanted,)

    # Replaying the reported tree through the HOST evaluator reproduces the
    # reported fitness: the device loss and the host loss are the same loss.
    replayed = SymbolicRegression(x, y).evaluate(
        (opcode[None, :], payload[None, :], size[None, :])
    )
    np.testing.assert_allclose(
        np.asarray(replayed, dtype=np.float32),
        np.float32(device_result.best_fitness),
        rtol=0.0,
        atol=1e-4,
        err_msg="the reported best tree must reproduce the reported fitness",
    )

    # The loop ends where the next generation would start: a device population.
    for name in ("opcode", "payload", "size", "fitness", "key"):
        leaf = getattr(device_result.state, name)
        assert is_device_tensor(leaf), (
            f"{name} of the final state must be on the device"
        )
        assert leaf.device == device
