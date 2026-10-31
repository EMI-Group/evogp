"""DEVICE-GATED tests: the native DLPack hand-off and the device-native fused evaluator.

Two surfaces are under test and this is their net:

``evogp.gp.fused_eval``'s native interop
    the standard DLPack protocol between an ``evoxc`` DEVICE tensor and torch --
    the evoxc tensor exports (``Tensor.__dlpack__``), ``torch.from_dlpack``
    consumes, and ``evoxp.gp.fused_eval.export_to_torch`` validates what came
    back (shape / dtype / device kind / DLPack device id).  It is the only place
    in this subtree that shares a pointer with torch, and its failure mode has to
    be a loud :class:`InteropUnavailable` naming the host path -- never a
    silently reinterpreted buffer.  The RETURN direction needs no native import:
    the evaluator owns an ``evoxc`` device fitness buffer, exports it once, and
    writes the kernel's result into it with a device-local ``copy_``.

``evogp.gp.fused_eval.FusedDeviceEvaluator``
    the route built on it: device channels -> zero-copy torch views -> the
    legacy triple computed in torch on the device -> the sibling Triton kernel
    -> the ``(pop_size,)`` fitness written into that device buffer (which the
    variation graph consumes without a readback), with a host fallback that is
    RECORDED (``last_route`` / ``route_counts`` / ``last_fallback``).

What this module pins, in order:

* the skip gate -- a box without a visible device, without ``iree``, without
  torch/triton or without the DLPack protocol SKIPS this module, and no test
  here imports torch at module scope, so collection cannot fail on such a box;
* ``interop_available`` semantics: true here, false (never raising) for a
  ``cpu`` kind, an unknown kind and the other build's kind;
* ``export_to_torch`` declines a host tensor, a numpy array and a bare channel
  triple with a message naming the fallback, and otherwise returns a torch view
  that SHARES the buffer: mutating it is visible through the evoxc tensor;
* the device route's fitness buffer IS an evoxc device tensor whose torch view
  shares its memory, so a write through the view is consumed by a ``variation``
  graph without any import step, and the buffer is reused across calls (its
  torch view and the evoxc handle are the same storage every time);
* ``_legacy_triple_device`` is BITWISE-equal to the host bridge
  (``encoding.to_legacy_triple``) for a fresh population, for a population after
  five chained ``variation`` calls (stale cells past the live length included)
  and for a hand-built padded population -- and an ``OUT_NODE``-tagged opcode
  raises ``NotImplementedError`` instead of producing a number;
* ``FusedDeviceEvaluator`` parity: device channels in, an ``evoxc`` DEVICE
  fitness tensor out (never numpy), equal to the in-graph ``GraphEvaluator``
  route inside 1e-5 relative with identical finiteness;
* the fallback: host channels (and the three 1-D rows of one individual) are
  served by the host route and recorded as such, and a device attempt that fails
  mid-call is recorded too -- a device state is never silently downloaded;
* driver integration: five chained ``workflow.step`` calls driven by
  ``FusedDeviceEvaluator`` produce the SAME channels as the same chain driven by
  ``GraphEvaluator`` (fitness may drift inside 1e-6), keep the state
  device-resident and advance the key every generation;
* import discipline: ``evogp.gp.fused_eval`` imports with only ``src`` on
  ``PYTHONPATH`` (no ``evoxc``, no torch) and its probes answer ``False`` there
  instead of raising, and this test module itself resolves torch lazily.  The
  subtree's own source grep lives in ``test_gp_stack_purity.py``.

Everything is DEVICE-GATED on the same probe the other device modules use
(``gp_operator_support.usable_device_kind``) plus the interop/fused probes, and
every test skips -- never fails -- where a piece is missing.  Sizes are small on
purpose (pop 16, 32 slots, 64 data rows) and nothing here is a timing loop.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

evoxc = pytest.importorskip(
    "evoxc", reason="the device interop and its tests are built on evoxc"
)

from evogp.gp import (
    GPConfig,
    GPTreeConfig,
    GraphEvaluator,
    SymbolicRegression,
    init_gp,
    init_population,
    smoke,
    step,
)
from evogp.gp import encoding, fused_eval, graph_eval
from evogp.gp import variation as V
from evogp.gp._constants import FUNC_TYPE, NType

from gp_operator_support import (
    DEVICE_SKIP_REASON,
    is_device_tensor,
    to_host,
    usable_device_kind,
)

# --------------------------------------------------------------------------- #
# the skip gate: device + iree + torch + triton + the DLPack protocol, or skip
# --------------------------------------------------------------------------- #
#: ``"rocm"``/``"cuda"`` when the iree backend has a real device here, else None.
DEVICE_KIND = usable_device_kind()

#: This module's path (``src/evogp/gp/tests``) and the ``src`` root a clean
#: subprocess gets on ``PYTHONPATH``.
TESTS_DIR = pathlib.Path(__file__).resolve().parent
GP_DIR = TESTS_DIR.parent
SRC = GP_DIR.parents[1]


def _gate_reason():
    """Why this module must skip, or ``None`` when every piece is present.

    The device probe comes first (it needs no torch), so a CPU box never imports
    torch just to be told to skip.
    """
    if DEVICE_KIND is None:
        return DEVICE_SKIP_REASON
    if importlib.util.find_spec("torch") is None:
        return (
            "the device-native fused route needs torch (the Triton kernel is a "
            "torch artefact); resolve it lazily with pytest.importorskip"
        )
    if importlib.util.find_spec("triton") is None:
        return (
            "the device-native fused route needs triton (the fused kernel is "
            "JIT-compiled at the first call)"
        )
    if not fused_eval.interop_available(DEVICE_KIND):
        return (
            "the native device hand-off is not available on this box: no torch "
            "CUDA/HIP device, no triton, or an evoxc whose device payloads do "
            "not export DLPack (verify with fused_eval.interop_available)"
        )
    if not fused_eval.fused_available():
        return (
            "the fused route is not available on this box (the sibling engine "
            "declines, or no writable TRITON_CACHE_DIR)"
        )
    return None


SKIP_REASON = _gate_reason()

pytestmark = pytest.mark.skipif(
    SKIP_REASON is not None, reason=SKIP_REASON or DEVICE_SKIP_REASON
)

#: The run this module pins: one compile per graph, deterministic across routes.
SEED = 0
NUM_DATA = 64
POP_SIZE = 16
ELITE_COUNT = 2
#: Relative agreement of two evaluation routes (measured at 1e-07-3e-07 here).
REL_TOL = 1e-5
#: Absolute drift allowed between the two routes' fitness vectors per generation.
FITNESS_TOL = 1e-6

CHANNELS = ("opcode", "payload", "size")
CHANNEL_DTYPES = {"opcode": "int32", "payload": "float32", "size": "int32"}


# --------------------------------------------------------------------------- #
# fixtures (module scoped: the device, the population and one compile per graph)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def torch_module():
    """torch, resolved lazily so a torch-less box skips instead of failing."""
    return pytest.importorskip(
        "torch", reason="the device-native fused route is a torch artefact"
    )


@pytest.fixture(scope="module")
def device():
    """The device under test (an ``evoxc.Device``, never a bare kind string)."""
    return evoxc.Device(DEVICE_KIND, 0)


@pytest.fixture(scope="module")
def tree_config():
    """32 slots, 2 inputs, a mixed-arity arithmetic set (``+ - * /``)."""
    return GPTreeConfig(
        max_nodes=32,
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
    """``(x, y)`` of the smoke SR problem -- the data every route evaluates."""
    return smoke.make_data(NUM_DATA, seed=SEED)


@pytest.fixture(scope="module")
def host_state(run_config):
    """Generation 0 of the host (numpy) run."""
    return init_gp(run_config, SEED)


@pytest.fixture(scope="module")
def device_state(run_config, device):
    """Generation 0 of the device run (same seed, hence the same trees)."""
    return init_gp(run_config, SEED, backend="iree", device=device)


@pytest.fixture(scope="module")
def graph_evaluator(data, tree_config, device):
    """The in-graph device route: the reference the fused route is diffed against."""
    x, y = data
    return GraphEvaluator(x, y, used_ids=tree_config, backend="iree", device=device)


@pytest.fixture
def fused_device(data):
    """A FRESH fused evaluator per test (its route counters are per instance)."""
    x, y = data
    return fused_eval.FusedDeviceEvaluator(x, y)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _host_channels(population):
    """``(opcode, payload, size)`` numpy arrays of a population."""
    return tuple(to_host(getattr(population, name)) for name in CHANNELS)


def _device_channels(population):
    """``(opcode, payload, size)`` as given -- device tensors stay on the device."""
    return tuple(getattr(population, name) for name in CHANNELS)


def _to_device(device, arrays):
    """Upload numpy channel arrays (an explicit transfer: evoxc never implies one)."""
    return tuple(
        evoxc.from_numpy(np.ascontiguousarray(array)).to(device) for array in arrays
    )


def _legacy_triple_on_device(torch_module, channels):
    """The device-side legacy conversion of one population, downloaded as numpy.

    ``_legacy_triple_device`` is private on purpose (it is the device twin of
    ``encoding.to_legacy_triple``, not a second public conversion), so it is
    exercised here the way its only caller uses it: exported channels, the
    ``FUNC_TYPE`` table on the launch device, three torch tensors out.
    """
    opcode, payload, size = (
        fused_eval.export_to_torch(tensor) for tensor in channels
    )
    table = torch_module.as_tensor(np.asarray(FUNC_TYPE, np.int32), device=opcode.device)
    value, node_type, subtree_size = fused_eval._legacy_triple_device(  # noqa: SLF001
        torch_module, table, opcode, payload, size
    )
    return (
        value.detach().cpu().numpy(),
        node_type.detach().cpu().numpy(),
        subtree_size.detach().cpu().numpy(),
    )


def assert_relative_close(actual, expected, *, rtol: float = REL_TOL, context: str = ""):
    """``actual`` reproduces ``expected`` inside ``rtol`` relative, finiteness included.

    The fitness vectors of two routes are losses of wildly different magnitude,
    so the comparison is relative value by value (``|a-b| <= rtol*max(|a|,|b|)``,
    which is also right for a pair of exact zeros); ``-inf`` is a real
    "unevaluable" value here, so the two finiteness patterns must MATCH rather
    than be compared numerically.
    """
    prefix = f"{context}: " if context else ""
    lhs = np.asarray(actual, dtype=np.float64).reshape(-1)
    rhs = np.asarray(expected, dtype=np.float64).reshape(-1)
    assert lhs.shape == rhs.shape, f"{prefix}shape {lhs.shape} != {rhs.shape}"
    lhs_finite = np.isfinite(lhs)
    rhs_finite = np.isfinite(rhs)
    assert np.array_equal(lhs_finite, rhs_finite), (
        f"{prefix}the finiteness patterns differ: {int(lhs_finite.sum())} finite "
        f"values here against {int(rhs_finite.sum())} there"
    )
    if not lhs_finite.any():
        return
    left = lhs[lhs_finite]
    right = rhs[lhs_finite]
    delta = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    offending = delta > rtol * scale
    if offending.any():
        worst = int(np.argmax(delta - rtol * scale))
        relative = delta / np.where(scale > 0.0, scale, 1.0)
        raise AssertionError(
            f"{prefix}{int(offending.sum())} of {left.size} values differ by more "
            f"than {rtol:g} relative; worst: {left[worst]!r} against "
            f"{right[worst]!r} (delta {delta[worst]!r}, scale {scale[worst]!r}); "
            f"largest relative delta {float(relative.max()):.3e}"
        )


# --------------------------------------------------------------------------- #
# the gate itself
# --------------------------------------------------------------------------- #
def test_the_module_only_runs_where_the_device_route_is_usable():
    """The gate above is the contract: skip (never fail) when a piece is missing.

    A skipped module runs no test, so this case documents and re-checks the
    conditions that let it run at all: a visible device, the interop protocol and
    the fused engine.
    """
    assert SKIP_REASON is None
    assert _gate_reason() is None
    assert DEVICE_KIND in ("rocm", "cuda")
    assert fused_eval.interop_available() is True
    assert fused_eval.interop_available(DEVICE_KIND) is True
    assert fused_eval.fused_available() is True


# --------------------------------------------------------------------------- #
# interop_available(): a probe, never a raise
# --------------------------------------------------------------------------- #
def test_interop_available_declines_the_kinds_the_native_path_cannot_serve():
    """``cpu``, an unknown kind and the other build's kind are all ``False``.

    The hand-off only serves kinds whose DLPack code maps onto a torch device of
    the same physical kind: the host path already covers ``cpu``, and a ``cuda``
    capsule on a ROCm build (or ``rocm`` on an NVIDIA build) would be a silent
    misinterpretation of the pointer.
    """
    assert fused_eval.interop_available("cpu") is False
    assert fused_eval.interop_available("vulkan") is False
    other_build = "cuda" if DEVICE_KIND == "rocm" else "rocm"
    assert fused_eval.interop_available(other_build) is False
    # A probe answers, it never raises -- whatever it is handed.
    assert fused_eval.interop_available(object()) is False


def test_the_dispatch_probes_see_a_device_and_only_a_device(
    device_state, host_state, device
):
    """``is_device_tensor`` / ``device_of`` on both kinds of tensor.

    These two are what :meth:`FusedDeviceEvaluator.evaluate` dispatches on before
    any transfer, so they must be free of host round trips, free of torch and free
    of exceptions: ``False`` for everything that is not a device tensor, and an
    :class:`InteropUnavailable` (not a wrong device) when a device is asked of a
    host buffer.  Both read the tensor's own public metadata, never a private
    payload handle.
    """
    for name in CHANNELS:
        tensor = getattr(device_state, name)
        assert fused_eval.is_device_tensor(tensor) is True, name
        assert fused_eval.device_of(tensor) == device
        assert fused_eval.is_device_tensor(getattr(host_state, name)) is False, name
        with pytest.raises(fused_eval.InteropUnavailable):
            fused_eval.device_of(getattr(host_state, name))
    assert fused_eval.is_device_tensor(np.zeros((2, 2), np.float32)) is False
    assert fused_eval.is_device_tensor(None) is False
    with pytest.raises(fused_eval.InteropUnavailable):
        fused_eval.device_of(object())


# --------------------------------------------------------------------------- #
# export: zero-copy through the public DLPack protocol, or a loud refusal
# --------------------------------------------------------------------------- #
EXPORT_REJECTIONS = (
    pytest.param(lambda host, device: host.opcode, id="host-evoxc-tensor"),
    pytest.param(
        lambda host, device: np.asarray(to_host(host.opcode)), id="numpy-array"
    ),
    pytest.param(
        lambda host, device: tuple(to_host(getattr(host, name)) for name in CHANNELS),
        id="bare-channel-triple",
    ),
    pytest.param(lambda host, device: object(), id="not-a-tensor"),
)


@pytest.mark.parametrize("make_argument", EXPORT_REJECTIONS)
def test_export_to_torch_declines_a_non_device_buffer(
    make_argument, host_state, device
):
    """Only a device payload can be shared; everything else names the host path."""
    argument = make_argument(host_state, device)
    with pytest.raises(fused_eval.InteropUnavailable) as info:
        fused_eval.export_to_torch(argument)
    message = str(info.value).lower()
    assert "host path" in message, (
        "a refusal must name the fallback the caller can use; got: " + str(info.value)
    )


def test_export_to_torch_rejects_a_contradicting_device_kind(device_state):
    """``device_kind`` is an assertion: a contradiction is a caller error.

    Agreeing with the tensor's own device is accepted; naming a different kind is
    a ``ValueError`` (a capsule minted for the wrong device type would be a
    silently wrong mapping, so it never gets that far).
    """
    exported = fused_eval.export_to_torch(
        device_state.opcode, device_kind=DEVICE_KIND
    )
    assert tuple(exported.shape) == tuple(
        int(dim) for dim in device_state.opcode.shape
    )
    with pytest.raises(ValueError):
        fused_eval.export_to_torch(device_state.opcode, device_kind="cpu")


@pytest.mark.parametrize("name", CHANNELS)
def test_export_returns_a_sharing_view_of_every_channel(
    name, device_state, torch_module
):
    """Shape, dtype, device and VALUES of the exported view match the evoxc tensor."""
    tensor = getattr(device_state, name)
    exported = fused_eval.export_to_torch(tensor)
    assert isinstance(exported, torch_module.Tensor)
    assert tuple(exported.shape) == tuple(int(dim) for dim in tensor.shape)
    assert str(exported.dtype) == f"torch.{CHANNEL_DTYPES[name]}"
    # torch's spelling of a ROCm buffer is still "cuda" (the repository policy).
    assert exported.device.type == "cuda"
    assert exported.is_contiguous()
    np.testing.assert_array_equal(
        exported.detach().cpu().numpy(),
        to_host(tensor),
        err_msg=f"the exported {name} view must carry the evoxc values",
    )


def test_mutating_the_torch_view_is_visible_through_the_evoxc_tensor(
    run_config, device, torch_module
):
    """No copy anywhere: a write through one handle is a write through the other.

    This is the whole point of the hand-off (the fused route reads the population
    through these views), so it is measured on a THROWAWAY population rather than
    on the module fixture -- the mutation is real.
    """
    population = init_population(
        run_config, 11, pop_size=POP_SIZE, backend="iree", device=device
    )
    before = to_host(population.opcode)
    exported = fused_eval.export_to_torch(population.opcode)
    exported.add_(1)
    np.testing.assert_array_equal(
        to_host(population.opcode),
        before + 1,
        err_msg="a torch write must land in the evoxc buffer (no copy)",
    )
    # ...and a SECOND export sees the same bytes, i.e. both views are live on one
    # buffer rather than two snapshots.
    np.testing.assert_array_equal(
        fused_eval.export_to_torch(population.opcode).detach().cpu().numpy(),
        before + 1,
    )
    assert exported.data_ptr() == fused_eval.export_to_torch(
        population.opcode
    ).data_ptr(), "two exports of one evoxc tensor must share the pointer"


def test_export_to_torch_declines_a_buffer_whose_metadata_disagrees(
    device_state, monkeypatch, torch_module
):
    """A shape/dtype/device disagreement is a loud refusal, never a reinterpretation.

    The guard exists because a DLPack capsule is a raw pointer: if what torch
    hands back does not describe the same buffer the evoxc tensor describes, the
    caller would be reading the wrong bytes -- so the call declines (and the
    evaluator falls back) instead of returning the tensor.
    """
    real_from_dlpack = torch_module.from_dlpack

    def falsifying_from_dlpack(tensor, *args, **kwargs):
        return real_from_dlpack(tensor, *args, **kwargs).reshape(-1)

    monkeypatch.setattr(torch_module, "from_dlpack", falsifying_from_dlpack)
    with pytest.raises(fused_eval.InteropUnavailable) as info:
        fused_eval.export_to_torch(device_state.opcode)
    assert "shape" in str(info.value).lower(), str(info.value)


# --------------------------------------------------------------------------- #
# the return direction: an evoxc device buffer the variation graph consumes
# --------------------------------------------------------------------------- #
def test_a_torch_write_into_an_exported_buffer_feeds_a_variation_graph(
    run_config, device_state, device, torch_module
):
    """A written-through buffer IS an evoxc device tensor: the graph consumes it.

    This is the return direction the evaluator uses (it owns an ``evoxc`` device
    buffer, exports it ONCE, and ``copy_``s every kernel result into it), so the
    elite rows of the state ``variation`` returns must carry exactly the written
    values -- with no import step and no host round trip.
    """
    fitness = np.linspace(0.5, 3.0, POP_SIZE, dtype=np.float32)
    expected_elites = np.argsort(-fitness, kind="stable")[:ELITE_COUNT]

    buffer = evoxc.zeros((POP_SIZE,), evoxc.float32).to(device)
    view = fused_eval.export_to_torch(buffer)
    view.copy_(torch_module.as_tensor(fitness, device=view.device))
    torch_module.cuda.synchronize(view.device)

    assert fused_eval.is_device_tensor(buffer)
    assert buffer.device == device
    assert tuple(int(dim) for dim in buffer.shape) == (POP_SIZE,)
    np.testing.assert_array_equal(to_host(buffer), fitness)

    offspring, next_key = V.variation(
        run_config,
        device_state,
        buffer,
        SEED,
        elite_count=ELITE_COUNT,
        backend="iree",
        device=device,
    )
    recorded = to_host(offspring.fitness)
    np.testing.assert_array_equal(
        recorded[:ELITE_COUNT],
        fitness[expected_elites],
        err_msg="the elite rows must carry the written fitness values",
    )
    assert np.all(np.isneginf(recorded[ELITE_COUNT:])), (
        "offspring are unevaluated until the next evaluate call"
    )
    parent = {name: to_host(getattr(device_state, name)) for name in CHANNELS}
    for slot, source in enumerate(expected_elites):
        for name in CHANNELS:
            np.testing.assert_array_equal(
                to_host(getattr(offspring, name))[slot],
                parent[name][source],
                err_msg=f"elite slot {slot} must be parent row {source}",
            )
    assert is_device_tensor(next_key)


def test_the_device_route_writes_into_one_reused_fitness_buffer(
    device_state, fused_device
):
    """The returned tensor is a persistent device buffer, not a fresh allocation.

    The route exists to keep a generation free of host round trips, so the fitness
    comes back as the SAME evoxc buffer every call (the kernel result is copied
    into it on the device) and its torch view is the same storage -- which is what
    lets ``variation`` consume it zero-copy.
    """
    first = fused_device.evaluate(device_state)
    second = fused_device.evaluate(device_state)
    assert first is second, "the fitness buffer must be reused, not re-allocated"
    assert fused_eval.is_device_tensor(first)
    view = fused_eval.export_to_torch(first)
    np.testing.assert_array_equal(
        view.detach().cpu().numpy(),
        to_host(first),
        err_msg="the buffer's torch view and its evoxc handle are one storage",
    )
    np.testing.assert_array_equal(to_host(second), to_host(first))


def test_the_fitness_buffer_follows_the_population_size(tree_config, device, fused_device):
    """A different population size gets its own buffer (the shape is static).

    Every leaf of a ``GPState`` has a run-constant shape, so the fitness buffer is
    keyed by the shape too: a smaller/larger population must not read back a
    buffer of the wrong length, and the smaller shape has to keep working after
    the larger one was served.
    """
    small_pop = POP_SIZE // 2
    small = init_population(
        tree_config, 3, pop_size=small_pop, backend="iree", device=device
    )
    first = fused_device.evaluate(small)
    assert tuple(int(dim) for dim in first.shape) == (small_pop,)
    larger = init_population(
        tree_config, 3, pop_size=POP_SIZE * 2, backend="iree", device=device
    )
    second = fused_device.evaluate(larger)
    assert tuple(int(dim) for dim in second.shape) == (POP_SIZE * 2,)
    assert first is not second
    third = fused_device.evaluate(small)
    assert tuple(int(dim) for dim in third.shape) == (small_pop,)


# --------------------------------------------------------------------------- #
# the conversion: bitwise the host bridge, on every cell
# --------------------------------------------------------------------------- #
#: A hand-built population that exercises the padding rule on purpose:
#: row 0 is a 3-node tree followed by STALE non-zero cells whose ``size`` is 0,
#: row 1 is padding end to end, row 2 is a 5-node tree with three stale cells.
_PADDING_POPULATION = (
    np.array(
        [
            [3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 1, 0, 1, 7, 7, 7],
        ],
        dtype=np.int32,
    ),
    np.array(
        [
            [0.0, 1.0, 0.25, 7.5, -2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0, -0.25, 9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [3, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 3, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.int32,
    ),
)


def _five_variations(run_config, device, seed: int = 5):
    """A device population after five chained generations (stale cells included)."""
    population = init_population(
        run_config, seed, pop_size=POP_SIZE, backend="iree", device=device
    )
    fitness = np.linspace(0.5, 3.0, POP_SIZE, dtype=np.float32)
    for generation in range(5):
        population, _key = V.variation(
            run_config,
            population,
            fitness + float(generation),
            seed + generation,
            elite_count=ELITE_COUNT,
            backend="iree",
            device=device,
        )
    return population


LEGACY_TRIPLE_CASES = ("fresh", "after-five-variations", "hand-built-padding")


@pytest.mark.parametrize("case", LEGACY_TRIPLE_CASES)
def test_the_device_legacy_conversion_is_bitwise_the_host_bridge(
    case, run_config, device_state, device, torch_module
):
    """``_legacy_triple_device`` == ``encoding.to_legacy_triple``, cell for cell.

    Both are ``where`` chains over the same rule (a function node carries the id
    and its ``FUNC_TYPE`` marker, a leaf carries its payload and opcode, a
    ``size == 0`` cell is zero in all three), so a difference anywhere -- the
    stale cells past a row's live length included -- would be a real drift
    between what the kernel reads and what the host bridge computes.
    """
    if case == "fresh":
        channels = _device_channels(device_state)
    elif case == "after-five-variations":
        channels = _device_channels(_five_variations(run_config, device))
    else:
        channels = _to_device(device, _PADDING_POPULATION)

    value, node_type, subtree_size = _legacy_triple_on_device(torch_module, channels)
    expected_value, expected_type, expected_size = encoding.to_legacy_triple(
        tuple(to_host(tensor) for tensor in channels)
    )
    np.testing.assert_array_equal(value, expected_value, err_msg=f"{case}: value")
    np.testing.assert_array_equal(
        node_type, expected_type, err_msg=f"{case}: node type"
    )
    np.testing.assert_array_equal(
        subtree_size, expected_size, err_msg=f"{case}: subtree size"
    )
    assert value.dtype == np.float32
    assert node_type.dtype == np.int16
    assert subtree_size.dtype == np.int16

    if case == "hand-built-padding":
        host_opcode, host_payload, host_size = (to_host(tensor) for tensor in channels)
        # The padding rule really is exercised: the stale cells carry data, the
        # fully padded row carries nothing, and the conversion drops both.
        assert np.any(host_payload[0, 3:] != 0), "row 0 keeps stale payload cells"
        assert np.any(host_opcode[2, 5:] != 0), "row 2 keeps stale opcode cells"
        assert np.all(host_size[0, 3:] == 0) and np.all(host_size[2, 5:] == 0)
        for output in (value, node_type, subtree_size):
            assert np.all(output[0, 3:] == 0), "a size == 0 cell must convert to 0"
            assert np.all(output[2, 5:] == 0), "a size == 0 cell must convert to 0"
            assert np.all(output[1] == 0), "a fully padded row must convert to 0"
        # ...while the live nodes are untouched, so the zeros above are the rule
        # and not an all-zero fixture.
        assert np.all(subtree_size[0, :3] > 0)
        np.testing.assert_array_equal(subtree_size[2, :5], host_size[2, :5])
    if case == "after-five-variations":
        # Stale cells past the live length are what makes this case different from
        # the fresh one: the operator layer keeps them, the conversion must zero
        # the ones the parent marked with size 0.
        host_channels = tuple(to_host(tensor) for tensor in channels)
        stale = [
            row
            for row in range(POP_SIZE)
            if np.any(host_channels[0][row, int(host_channels[2][row, 0]):] != 0)
        ]
        assert stale, "the chained population must keep stale cells to be meaningful"


def test_an_out_node_tagged_opcode_raises_not_implemented(
    fused_device, device_state, device
):
    """Multi-output trees are rejected, and the rejection is not a silent fallback."""
    opcode, payload, size = _host_channels(device_state)
    tagged = opcode.copy()
    tagged[0, 0] = int(tagged[0, 0]) | int(NType.OUT_NODE)
    assert np.any((tagged & NType.OUT_NODE) != 0)
    channels = _to_device(device, (tagged, payload, size))

    before = dict(fused_device.route_counts)
    with pytest.raises(NotImplementedError) as info:
        fused_device.evaluate(channels)
    assert "multi-output" in str(info.value), str(info.value)
    assert fused_device.route_counts == before, (
        "an OUT_NODE rejection must not be recorded as a served route"
    )
    assert fused_device.last_route is None
    assert fused_device.last_fallback is None


# --------------------------------------------------------------------------- #
# the device-native evaluator: parity with the in-graph route
# --------------------------------------------------------------------------- #
def test_the_device_evaluator_serves_device_channels_with_device_fitness(
    device_state, graph_evaluator, fused_device, data, device
):
    """Device channels in, an ``evoxc`` DEVICE fitness tensor out -- the numbers too.

    The reference is the in-graph route on the same device; the numbers must
    agree inside 1e-5 relative (two different kernels over the same trees, not
    the same program) and the finiteness must agree exactly, because ``-inf`` is
    a real "unevaluable" verdict and not a small number.
    """
    x, y = data
    problem = SymbolicRegression(x, y)

    raw = fused_device.evaluate(device_state)
    assert not isinstance(raw, np.ndarray), "the device route must not hand back numpy"
    assert isinstance(raw, evoxc.Tensor)
    assert is_device_tensor(raw)
    assert raw.device == device
    assert tuple(int(dim) for dim in raw.shape) == (POP_SIZE,)
    assert np.dtype(raw.dtype) == np.float32

    fused_values = graph_eval.readback_fitness(raw, POP_SIZE)
    assert fused_values.dtype == np.float32
    assert_relative_close(
        fused_values,
        graph_evaluator.evaluate_numpy(device_state),
        context="fused vs in-graph",
    )
    # ...and the host problem agrees with both, so neither is a shared mistake.
    assert_relative_close(
        fused_values,
        problem.evaluate(_host_channels(device_state)),
        context="fused vs host problem",
    )

    assert fused_device.last_route == "device"
    assert fused_device.route_counts == {"device": 1, "host": 0}
    assert fused_device.last_fallback is None

    assert fused_device.problem_dim == problem.problem_dim == 2
    assert fused_device.solution_dim == problem.solution_dim == 1
    assert fused_device.available() is True
    assert fused_device.device_available() is True
    assert fused_device.device_kind == DEVICE_KIND

    # The callable IS the evaluator: the driver wires it in as `evaluate=`.
    np.testing.assert_array_equal(
        graph_eval.readback_fitness(fused_device(device_state), POP_SIZE), fused_values
    )


def test_a_host_population_takes_the_host_route(host_state, fused_device, data):
    """Host channels are served by the host route -- and the route is recorded.

    The fused kernel is reachable from numpy channels too (that is
    ``FusedEvaluator``'s path); what must never happen is a device population
    being downloaded without the fallback showing up in ``last_route``.
    """
    x, y = data
    problem = SymbolicRegression(x, y)

    result = fused_device.evaluate(host_state)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (POP_SIZE,)
    assert_relative_close(result, problem.evaluate(host_state), context="host route")
    assert fused_device.last_route == "host"
    assert fused_device.route_counts == {"device": 0, "host": 1}

    # The three 1-D rows of ONE individual are served the same way (a population
    # of one): the host path's promotion, not a device dispatch.
    rows = tuple(to_host(getattr(host_state, name))[0] for name in CHANNELS)
    single = fused_device.evaluate(rows)
    assert isinstance(single, np.ndarray)
    assert single.shape == (1,)
    assert_relative_close(single, problem.evaluate(rows), context="one individual")
    assert fused_device.route_counts == {"device": 0, "host": 2}


def test_a_failed_device_attempt_falls_back_and_is_recorded(
    device_state, fused_device, data, monkeypatch
):
    """A mid-call refusal retries on the host and says so -- never silently.

    ``InteropUnavailable`` / ``FusedUnavailable`` are the two fallback signals;
    the instrumentation proves the device route WAS attempted (so this is the
    recorded-fallback path and not the host-only path), and the numbers are the
    host problem's.
    """
    x, y = data
    attempts = []
    real_export = fused_eval.export_to_torch

    def refusing_export(tensor, **kwargs):
        attempts.append(tensor)
        raise fused_eval.InteropUnavailable(
            "simulated: this box cannot share the pointer; the host path still "
            "serves this call"
        )

    monkeypatch.setattr(fused_eval, "export_to_torch", refusing_export)
    result = fused_device.evaluate(device_state)
    assert attempts, "the device route must be attempted before it can fall back"

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (POP_SIZE,)
    assert fused_device.last_route == "host"
    assert fused_device.route_counts == {"device": 0, "host": 1}
    assert fused_device.last_fallback is not None
    assert "InteropUnavailable" in fused_device.last_fallback
    assert "simulated" in fused_device.last_fallback
    assert_relative_close(
        result,
        SymbolicRegression(x, y).evaluate(device_state),
        context="fallback",
    )

    # Positive control: with the export restored, the same evaluator serves the
    # device route again (so the assertion above is about the injected failure).
    monkeypatch.setattr(fused_eval, "export_to_torch", real_export)
    raw = fused_device.evaluate(device_state)
    assert is_device_tensor(raw)
    assert fused_device.last_route == "device"
    assert fused_device.last_fallback is None
    assert fused_device.route_counts == {"device": 1, "host": 1}


# --------------------------------------------------------------------------- #
# driver integration: five generations, the same population as the in-graph chain
# --------------------------------------------------------------------------- #
def test_a_five_generation_chain_matches_the_in_graph_chain(
    run_config, graph_evaluator, fused_device, device
):
    """The fused route is an EVALUATOR, so the whole generation must not move.

    Both chains start from the same seed on the same device and differ only in
    ``evaluate=``: the channels stay bitwise identical for five generations
    (selection/splice/mutation are the same traced graph), the fitness vectors may
    drift inside 1e-6, and every state stays device-resident with an advanced key.
    """
    generations = 5
    in_graph = init_gp(run_config, SEED, backend="iree", device=device)
    fused = init_gp(run_config, SEED, backend="iree", device=device)
    previous_key = to_host(fused.key)

    for generation in range(generations):
        in_graph = step(
            run_config, in_graph, graph_evaluator, backend="iree", device=device
        )
        fused = step(run_config, fused, fused_device, backend="iree", device=device)

        for name in CHANNELS:
            np.testing.assert_array_equal(
                to_host(getattr(fused, name)),
                to_host(getattr(in_graph, name)),
                err_msg=(
                    f"generation {generation}: {name} must be bit-identical across "
                    "the two evaluation routes"
                ),
            )
        theirs = to_host(in_graph.fitness)
        ours = to_host(fused.fitness)
        assert np.array_equal(np.isfinite(ours), np.isfinite(theirs)), (
            f"generation {generation}: the two routes disagree on fitness finiteness"
        )
        finite = np.isfinite(ours)
        if finite.any():
            drift = float(np.max(np.abs(ours[finite] - theirs[finite])))
            assert drift <= FITNESS_TOL, (
                f"generation {generation}: fitness drift {drift:.3e} exceeds "
                f"{FITNESS_TOL:g} (two kernels, so it is not zero by construction)"
            )

        for name in CHANNELS + ("fitness", "key"):
            assert is_device_tensor(getattr(fused, name)), (
                f"generation {generation}: {name} must stay on the device"
            )
        key = to_host(fused.key)
        assert not np.array_equal(key, previous_key), (
            f"generation {generation}: the key must advance every generation"
        )
        previous_key = key

    assert fused_device.route_counts == {"device": generations, "host": 0}


# --------------------------------------------------------------------------- #
# import discipline
# --------------------------------------------------------------------------- #
def test_the_fused_module_imports_and_declines_without_evoxc():
    """``src`` alone on ``PYTHONPATH``: no evoxc, no torch, and a probe that answers.

    ``fused_eval`` is the torch side of this subtree, so its import must still be
    free of torch/evox/evoxc (everything is resolved through ``importlib`` at call
    time); with no evoxc there is simply nothing to share, and every probe must
    say ``False`` instead of raising.
    """
    assert SRC.name == "src", SRC
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from evogp.gp import fused_eval as interop\n"
        "eager = sorted(n for n in ('torch', 'evox', 'evoxc') if n in sys.modules)\n"
        "assert not eager, f'importing fused_eval pulled in {eager}'\n"
        "try:\n"
        "    interop.export_to_torch(np.zeros((2, 2), np.float32))\n"
        "except interop.InteropUnavailable as exc:\n"
        "    assert 'host path' in str(exc).lower(), str(exc)\n"
        "else:\n"
        "    raise AssertionError('export_to_torch must decline without evoxc')\n"
        "assert interop.interop_available() is False\n"
        "assert interop.interop_available('rocm') is False\n"
        "assert interop.is_device_tensor(object()) is False\n"
        "assert interop.is_device_tensor(np.zeros((2, 2), np.float32)) is False\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)  # deliberately NOT the evoxc checkout
    env.pop("EVOXC_BACKEND", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok", result.stdout


def test_this_module_resolves_torch_lazily():
    """A torch-less box must SKIP this module, not fail to import it.

    The subtree's own source grep (``test_gp_stack_purity.py``) covers the
    modules under test; this pins the test module itself to the same rule, so a
    stray module-scope ``import torch`` here would turn a skip into a collection
    error on exactly the box this gate exists for.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    eager = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            eager.append(node.module or "")
    offenders = [
        name for name in eager if name.split(".")[0] in ("torch", "evox", "evoxc")
    ]
    assert offenders == [], (
        "torch / evox / evoxc must be resolved through pytest.importorskip or "
        f"through the package under test; found module-scope imports of {offenders}"
    )
