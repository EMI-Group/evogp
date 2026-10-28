"""Route-policy regression tests: dense vs specialized AND the backend it picks.

``evogp.tree.dense_engine.resolve_route`` is the ONE route policy the five wired
call sites ask (``Forest.forward`` / ``Forest.batch_forward`` /
``Forest.SR_fitness``, ``Tree.forward`` / ``Tree.SR_fitness``):
:func:`evogp.tree.dense_engine.should_use_dense` is its boolean view, and the
route -- not an ambient environment variable -- hands the concrete backend to the
engine.  This module pins that policy:

* the DEFAULT (``auto``, no explicit backend): on a device box dense on ``iree``
  from ``EVOGP_DENSE_AUTO_MIN_POP`` (64) up and specialized on ``numpy`` below it
  -- including ``pop_size == 1`` (``Tree``) -- with a ``numpy`` FALLBACK for the
  dense attempt (a per-tree iree compile can never amortize);
* no device at all (pure CPU box): specialized + ``numpy`` at every size;
* an explicitly NAMED backend (``backend=`` argument or ``EVOGP_EVOXC_BACKEND``)
  keeps the historical rule exactly, and an explicit ``numpy`` never probes the
  device;
* ``EVOGP_EVAL_MODE=specialized`` never uses dense; ``EVOGP_EVAL_MODE=dense``
  uses the device backend on a device box and ``numpy`` elsewhere;
* the rows rule (``EVOGP_DENSE_AUTO_MAX_ROWS``, ``auto_rows_decline``) is
  unchanged and still only gates ``auto``;
* ``resolve_route`` and ``should_use_dense`` agree everywhere (one rule).

Like ``test_dense_policy.py`` this module never builds or runs a graph:
``backend="iree"`` is only a *name* here, and the device probe is a pinned
process-global (``dense._DEVICE_AVAILABLE``), so nothing initializes CUDA.
"""

from __future__ import annotations

import pytest

from evogp.tree import dense_engine as dense

#: The engine's own name for the backend variable (never a stale copy).
EVOXC_BACKEND_ENV = dense.BACKEND_ENV
assert EVOXC_BACKEND_ENV == "EVOGP_EVOXC_BACKEND"

#: A population at / above the ``auto`` threshold, and one below it.
ABOVE = dense.DEFAULT_AUTO_MIN_POP
BELOW = dense.DEFAULT_AUTO_MIN_POP - 1
#: Rows inside / above the ``auto`` ceiling.
ROWS_OK = dense.DEFAULT_AUTO_MAX_ROWS
ROWS_OVER = dense.DEFAULT_AUTO_MAX_ROWS + 1


@pytest.fixture(autouse=True)
def hermetic_knobs(monkeypatch):
    """Clear the knobs, and pin the device probe to "no device" by default.

    Every test then states the box it simulates explicitly, so the module's
    expectations never depend on the machine running the suite.
    """
    for name in (
        dense.EVAL_MODE_ENV,
        dense.MAX_TREE_LEN_ENV,
        dense.AUTO_MIN_POP_ENV,
        dense.AUTO_MAX_ROWS_ENV,
        EVOXC_BACKEND_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", False)


def on_device_box(monkeypatch, available=True):
    """Pin the route policy's device probe (monkeypatched global, no CUDA init)."""
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", available)


# --------------------------------------------------------------------------- #
# 1. the cached, monkeypatchable device probe
# --------------------------------------------------------------------------- #
def test_device_probe_is_memoized_and_monkeypatchable(monkeypatch):
    """``device_available()`` probes once per process, never once per call."""
    calls = []

    def fake_is_available():
        calls.append(True)
        return True

    monkeypatch.setattr(dense.torch.cuda, "is_available", fake_is_available)
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", None)
    assert dense.device_available() is True
    assert dense.device_available() is True
    assert len(calls) == 1, calls
    # ... and the memo is what a test (or a probe script) can pin
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", False)
    assert dense.device_available() is False
    assert len(calls) == 1


def test_explicit_numpy_backend_never_probes_the_device(monkeypatch):
    """``EVOGP_EVOXC_BACKEND=numpy`` forces the pure-CPU route, no probe at all."""

    def explode():
        raise AssertionError("the route policy probed the device for an explicit numpy")

    monkeypatch.setattr(dense.torch.cuda, "is_available", explode)
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", None)
    monkeypatch.setenv(EVOXC_BACKEND_ENV, "numpy")
    for pop in (1, BELOW, ABOVE, 10 ** 4):
        for mode in (None, "auto", "specialized"):
            route = dense.resolve_route(pop, eval_mode=mode)
            assert route.use_dense is False
            assert route.backend == "numpy" and route.fallback_backend == "numpy"
    assert dense._DEVICE_AVAILABLE is None  # not even memoized


# --------------------------------------------------------------------------- #
# 2. the default `auto` policy on a device box
# --------------------------------------------------------------------------- #
def test_auto_on_a_device_box_is_dense_iree_at_the_threshold(monkeypatch):
    on_device_box(monkeypatch)
    assert ABOVE == 64, "the documented default threshold"
    route = dense.resolve_route(ABOVE, rows=ROWS_OK)
    assert route.use_dense is True
    assert route.backend == "iree"
    # a per-tree iree compile (271-284 ms/tree) can never amortize: the fallback
    # after DenseUnsupported is the specialized engine on numpy
    assert route.fallback_backend == "numpy"
    assert dense.AUTO_MIN_POP_ENV in route.reason
    assert dense.resolve_route(10 ** 3).use_dense is True
    assert dense.resolve_route(10 ** 3).backend == "iree"


def test_auto_on_a_device_box_is_specialized_numpy_below_the_threshold(monkeypatch):
    on_device_box(monkeypatch)
    for pop in (1, 8, 16, BELOW):
        route = dense.resolve_route(pop)
        assert route.use_dense is False, pop
        assert route.backend == "numpy" and route.fallback_backend == "numpy", pop
    # ... and with the threshold lowered (a very long run), dense starts there
    monkeypatch.setenv(dense.AUTO_MIN_POP_ENV, "16")
    route = dense.resolve_route(16)
    assert route.use_dense is True and route.backend == "iree"
    assert dense.resolve_route(15).use_dense is False


def test_a_single_tree_never_starts_a_per_tree_compile(monkeypatch):
    """``Tree.forward`` / ``Tree.SR_fitness`` evaluate a population of one."""
    on_device_box(monkeypatch)
    route = dense.resolve_route(1)
    assert route.use_dense is False
    assert route.backend == "numpy"


# --------------------------------------------------------------------------- #
# 3. no device: unchanged specialized + numpy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pop", [1, 16, 64, 1000, 10 ** 4])
def test_no_device_box_stays_specialized_numpy(pop, monkeypatch):
    on_device_box(monkeypatch, available=False)
    route = dense.resolve_route(pop)
    assert route.use_dense is False
    assert route.backend == "numpy" and route.fallback_backend == "numpy"
    assert "device" in route.reason


# --------------------------------------------------------------------------- #
# 4. an explicitly named backend keeps the historical rule
# --------------------------------------------------------------------------- #
def test_explicit_numpy_backend_wins_over_the_policy(monkeypatch):
    on_device_box(monkeypatch)
    for pop in (1, BELOW, ABOVE, 10 ** 4):
        route = dense.resolve_route(pop, backend="numpy")
        assert route.use_dense is False
        assert route.backend == "numpy"
    monkeypatch.setenv(EVOXC_BACKEND_ENV, "numpy")
    for pop in (1, BELOW, ABOVE, 10 ** 4):
        assert dense.should_use_dense(pop) is False


def test_explicit_iree_backend_keeps_the_historical_threshold(monkeypatch):
    """An explicit backend is the pre-policy rule: dense iff pop >= MIN_POP."""
    on_device_box(monkeypatch)
    assert dense.resolve_route(ABOVE, backend="iree").use_dense is True
    assert dense.resolve_route(BELOW, backend="iree").use_dense is False
    below = dense.resolve_route(BELOW, backend="iree")
    assert below.backend == "iree" and below.fallback_backend == "iree"
    # the same through the environment, and it needs no probe either
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", None)
    monkeypatch.setenv(EVOXC_BACKEND_ENV, "iree")
    assert dense.resolve_route(ABOVE).use_dense is True
    assert dense.resolve_route(ABOVE).backend == "iree"
    assert dense.resolve_route(BELOW).use_dense is False
    assert dense._DEVICE_AVAILABLE is None


def test_explicit_specialized_never_uses_dense(monkeypatch):
    on_device_box(monkeypatch)
    for pop in (1, BELOW, ABOVE, 10 ** 4):
        assert dense.resolve_route(pop, eval_mode="specialized").use_dense is False
        assert (
            dense.resolve_route(pop, eval_mode="specialized", backend="iree").backend
            == "iree"
        )
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_route(ABOVE).use_dense is False
    assert dense.resolve_route(ABOVE).backend == "numpy"


def test_explicit_dense_uses_the_device_backend_when_one_is_available(monkeypatch):
    on_device_box(monkeypatch)
    for pop in (1, BELOW, ABOVE):
        route = dense.resolve_route(pop, eval_mode="dense")
        assert route.use_dense is True
        assert route.backend == "iree"
    # the `dynamic` alias is the same first-class spelling
    assert dense.resolve_route(1, eval_mode="dynamic").backend == "iree"
    # ... and numpy on a box with no device, or with an explicit backend
    on_device_box(monkeypatch, available=False)
    assert dense.resolve_route(ABOVE, eval_mode="dense").backend == "numpy"
    on_device_box(monkeypatch)
    assert dense.resolve_route(ABOVE, eval_mode="dense", backend="numpy").backend == "numpy"


# --------------------------------------------------------------------------- #
# 5. the rows rule is unchanged and still only gates `auto`
# --------------------------------------------------------------------------- #
def test_rows_rule_declines_auto_to_specialized_numpy(monkeypatch):
    on_device_box(monkeypatch)
    assert dense.resolve_route(ABOVE, rows=ROWS_OK).use_dense is True
    declined = dense.resolve_route(ABOVE, rows=ROWS_OVER)
    assert declined.use_dense is False
    assert declined.backend == "numpy"
    assert dense.AUTO_MAX_ROWS_ENV in declined.reason
    # per_row=True (unknown row count) and an explicit dense are never gated
    assert dense.resolve_route(ABOVE, rows=None).use_dense is True
    assert dense.resolve_route(ABOVE, rows=ROWS_OVER, eval_mode="dense").use_dense is True
    # an explicit backend keeps today's behaviour: specialized on THAT backend
    explicit = dense.resolve_route(ABOVE, rows=ROWS_OVER, backend="iree")
    assert explicit.use_dense is False and explicit.backend == "iree"
    # the public rows helper still reports the same decline
    assert dense.auto_rows_decline(ROWS_OVER, eval_mode="auto") is not None
    assert dense.auto_rows_decline(ROWS_OVER, eval_mode="dense") is None


# --------------------------------------------------------------------------- #
# 6. one rule: `should_use_dense` is the boolean view of `resolve_route`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("eval_mode", [None, "auto", "dense", "dynamic", "specialized"])
@pytest.mark.parametrize("backend", [None, "numpy", "iree"])
@pytest.mark.parametrize("pop", [1, 16, 64, 256])
def test_resolve_route_and_should_use_dense_agree(
    pop, backend, eval_mode, available, monkeypatch
):
    on_device_box(monkeypatch, available=available)
    for rows in (None, ROWS_OK, ROWS_OVER):
        route = dense.resolve_route(
            pop, rows=rows, backend=backend, eval_mode=eval_mode
        )
        assert (
            dense.should_use_dense(
                pop, rows=rows, backend=backend, eval_mode=eval_mode
            )
            is route.use_dense
        ), (pop, backend, eval_mode, available, rows)
        # the route always names a concrete backend for both halves
        assert route.backend in dense_engine_backends()
        assert route.fallback_backend in dense_engine_backends()


def dense_engine_backends():
    """The backend names the engine resolves (imported lazily: it is a tuple)."""
    from evogp.tree.evoxc_engine import SUPPORTED_BACKENDS

    return tuple(SUPPORTED_BACKENDS)
