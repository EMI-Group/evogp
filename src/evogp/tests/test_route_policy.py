"""Route-policy regression tests: dense vs specialized vs the opt-in batch engine.

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
* the OPT-IN batch route (``EVOGP_EVAL_MODE=batch``, the merged-graph engine in
  ``evogp.tree.batch_engine``): always attempted, with NO pop/rows gate -- the
  explicitly named backend, else ``iree`` on a device box and ``numpy``
  elsewhere, with a specialized+``numpy`` fallback, and named by
  ``Route.engine`` / ``Route.use_batch`` rather than by the dense flag;
* ``auto`` NEVER selects batch, and neither does ``dense`` / ``dynamic`` /
  ``specialized``: the opt-in switch is the only way into the batch engine;
* the rows rule (``auto_rows_decline``) still only gates ``auto``, but it is
  now a per-shape MEMORY ENVELOPE (``EVOGP_DENSE_AUTO_BUDGET_MB`` over
  ``pop_size * tree_len * rows * 4`` bytes) with ``EVOGP_DENSE_AUTO_MAX_ROWS`` as
  an optional tightening pin -- so the large-row / large-population cells stay on
  dense instead of falling back to the rows-flat specialized engine;
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
#: The row stride the rows envelope is evaluated at (the population's
#: ``max_tree_len``, the harness's ``tree_len``).
TREE_LEN = 64
#: Rows at / above the ``auto`` memory envelope's allowance for that shape.
ROWS_OK = dense.auto_row_allowance(ABOVE, TREE_LEN)
ROWS_OVER = ROWS_OK + 1
assert ROWS_OVER > 2048, "the reference shape must be well above the old ceiling"


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
        dense.AUTO_BUDGET_MB_ENV,
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
# 5. the rows rule is a memory envelope and still only gates `auto`
# --------------------------------------------------------------------------- #
def test_rows_rule_declines_auto_to_specialized_numpy(monkeypatch):
    on_device_box(monkeypatch)
    ok = dense.resolve_route(ABOVE, rows=ROWS_OK, tree_len=TREE_LEN)
    assert ok.use_dense is True, ok.reason
    declined = dense.resolve_route(ABOVE, rows=ROWS_OVER, tree_len=TREE_LEN)
    assert declined.use_dense is False
    assert declined.backend == "numpy"
    assert dense.AUTO_BUDGET_MB_ENV in declined.reason
    # per_row=True (unknown row count) and an explicit dense are never gated
    assert dense.resolve_route(ABOVE, rows=None, tree_len=TREE_LEN).use_dense is True
    assert (
        dense.resolve_route(
            ABOVE, rows=ROWS_OVER, tree_len=TREE_LEN, eval_mode="dense"
        ).use_dense
        is True
    )
    # an explicit backend keeps today's behaviour: specialized on THAT backend
    explicit = dense.resolve_route(
        ABOVE, rows=ROWS_OVER, tree_len=TREE_LEN, backend="iree"
    )
    assert explicit.use_dense is False and explicit.backend == "iree"
    # the public rows helper still reports the same decline
    assert (
        dense.auto_rows_decline(
            ROWS_OVER, eval_mode="auto", pop_size=ABOVE, tree_len=TREE_LEN
        )
        is not None
    )
    assert (
        dense.auto_rows_decline(
            ROWS_OVER, eval_mode="dense", pop_size=ABOVE, tree_len=TREE_LEN
        )
        is None
    )


@pytest.mark.parametrize("pop", [ABOVE, 256, 1000])
@pytest.mark.parametrize("rows", [4096, 16384])
def test_rows_envelope_keeps_dense_on_the_large_row_cells(pop, rows, monkeypatch):
    """The large-row cells stay on dense: the blunt 2048-row ceiling is gone.

    ``p256-L64-i32-N4096`` / ``p256-L64-i32-N16384`` / ``p1000-L64-i32-N4096`` /
    ``p1000-L64-i32-N16384`` are the cells where ``auto`` used to decline into
    ``specialized`` + numpy, which is 1.4-8.7x slower there
    (``test/benchmark/evoxc/LARGE_SCALE.md``).
    """
    on_device_box(monkeypatch)
    route = dense.resolve_route(pop, rows=rows, tree_len=TREE_LEN)
    assert route.use_dense is True, route.reason
    assert route.backend == "iree"
    assert dense.auto_rows_decline(rows, pop_size=pop, tree_len=TREE_LEN) is None
    assert rows > 2048  # the old fixed ceiling would have declined every one


def test_rows_envelope_scales_with_population_and_tree_length(monkeypatch):
    """The allowance is a byte budget: it shrinks as pop_size / tree_len grow."""
    on_device_box(monkeypatch)
    base = dense.auto_row_allowance(ABOVE, TREE_LEN)
    assert dense.auto_row_allowance(ABOVE * 4, TREE_LEN) == base // 4
    assert dense.auto_row_allowance(ABOVE, TREE_LEN * 2) == base // 2
    assert dense.auto_row_allowance(ABOVE, TREE_LEN, auto_budget_mb=16384) == base * 2
    # an unknown shape cannot apply the envelope (only an explicit pin)
    assert dense.auto_row_allowance(None, TREE_LEN) == 0
    assert dense.auto_row_allowance(ABOVE, None) == 0
    # the envelope itself declines above the byte budget, naming the knob
    over = dense.resolve_route(ABOVE * 4, rows=base // 4 + 1, tree_len=TREE_LEN)
    assert over.use_dense is False
    assert dense.AUTO_BUDGET_MB_ENV in over.reason and "GiB" in over.reason


def test_auto_max_rows_pin_still_wins(monkeypatch):
    """``EVOGP_DENSE_AUTO_MAX_ROWS`` is an optional, tightening pin over the envelope."""
    on_device_box(monkeypatch)
    pin = 1024
    assert dense.auto_row_allowance(ABOVE, TREE_LEN, auto_max_rows=pin) == pin
    # a pin ABOVE the envelope cannot loosen it
    assert dense.auto_row_allowance(ABOVE, TREE_LEN, auto_max_rows=ROWS_OK * 4) == ROWS_OK
    declined = dense.resolve_route(
        ABOVE, rows=pin + 1, tree_len=TREE_LEN, auto_max_rows=pin
    )
    assert declined.use_dense is False
    assert f"{dense.AUTO_MAX_ROWS_ENV}={pin}" in declined.reason
    assert "memory envelope" not in declined.reason
    # ... and through the environment, the documented user-facing knob
    monkeypatch.setenv(dense.AUTO_MAX_ROWS_ENV, str(pin))
    assert dense.resolve_auto_max_rows() == pin
    assert dense.resolve_route(ABOVE, rows=pin, tree_len=TREE_LEN).use_dense is True
    assert dense.resolve_route(ABOVE, rows=pin + 1, tree_len=TREE_LEN).use_dense is False
    # 0 is the "unpinned" spelling
    assert dense.resolve_auto_max_rows("0") == 0


def test_rows_rule_needs_a_shape_or_a_pin():
    """Without ``(pop_size, tree_len)`` only an explicit pin can decline.

    The wired call sites pass ``pop_size`` only, so this is what keeps them from
    re-creating the old false ceiling: the envelope is enforced inside
    ``evaluate_dense`` / ``sr_fitness_dense``, which know the real shape.
    """
    assert dense.auto_rows_decline(10 ** 7, eval_mode="auto") is None
    assert (
        dense.auto_rows_decline(10 ** 7, eval_mode="auto", auto_max_rows=1024)
        is not None
    )
    assert dense.auto_rows_decline(1024, eval_mode="auto", auto_max_rows=1024) is None


def test_cpu_box_rows_never_reach_dense(monkeypatch):
    """A non-device box stays specialized+numpy at every row count (unchanged)."""
    on_device_box(monkeypatch, available=False)
    for rows in (None, 4096, 16384, ROWS_OVER):
        for pop in (BELOW, ABOVE, 10 ** 4):
            route = dense.resolve_route(pop, rows=rows, tree_len=TREE_LEN)
            assert route.use_dense is False
            assert route.backend == "numpy"


def test_auto_never_picks_batch_at_large_rows(monkeypatch):
    """The rows half of ``auto`` may only flip dense <-> specialized."""
    on_device_box(monkeypatch)
    for rows in (ROWS_OK, ROWS_OVER, 16384):
        route = dense.resolve_route(
            ABOVE, rows=rows, tree_len=TREE_LEN, eval_mode="auto"
        )
        assert route.engine != "batch"
        assert route.use_batch is False
        assert route.use_dense is (route.engine == "dense")


# --------------------------------------------------------------------------- #
# 6. one rule: `should_use_dense` is the boolean view of `resolve_route`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize(
    "eval_mode", [None, "auto", "dense", "dynamic", "specialized", "batch"]
)
@pytest.mark.parametrize("backend", [None, "numpy", "iree"])
@pytest.mark.parametrize("pop", [1, 16, 64, 256])
def test_resolve_route_and_should_use_dense_agree(
    pop, backend, eval_mode, available, monkeypatch
):
    """The boolean view and the full route never disagree, batch included.

    ``batch`` is part of the matrix on purpose: it resolves to
    ``use_dense=False`` everywhere, so the dense flag stays the DENSE half of the
    policy only -- and the backend assertions below therefore also cover the
    route the batch engine is handed (both fields are always engine names).
    """
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
        # the route always names a concrete engine and a concrete backend for
        # both halves
        assert route.engine in dense.ENGINES
        assert route.backend in dense_engine_backends()
        assert route.fallback_backend in dense_engine_backends()


# --------------------------------------------------------------------------- #
# 7. the opt-in batch route (`EVOGP_EVAL_MODE=batch`)
# --------------------------------------------------------------------------- #
def test_batch_on_a_device_box_runs_iree_with_a_numpy_fallback(monkeypatch):
    """Batch is unconditional on a device box: iree, falling back to numpy.

    The merged graph needs ONE backend for the whole population, so the device
    branch mirrors the dense one (``iree``); the fallback is the specialized
    engine on ``numpy``, because a per-tree iree compile can never amortize after
    a ``BatchUnsupported``.  ``engine``/``use_batch`` -- not ``use_dense`` -- are
    what report the batch engine.
    """
    on_device_box(monkeypatch)
    for pop in (1, BELOW, ABOVE, 10 ** 3):
        route = dense.resolve_route(pop, eval_mode="batch")
        assert route.engine == "batch", pop
        assert route.use_batch is True, pop
        assert route.use_dense is False, pop
        assert route.backend == "iree", pop
        assert route.fallback_backend == "numpy", pop
        assert "batch" in route.reason and "iree" in route.reason, pop


@pytest.mark.parametrize("pop", [1, BELOW, ABOVE, 10 ** 3])
def test_batch_on_a_cpu_box_stays_numpy_on_numpy(pop, monkeypatch):
    """With no usable device the batch route is numpy on both halves, and says so."""
    on_device_box(monkeypatch, available=False)
    route = dense.resolve_route(pop, eval_mode="batch")
    assert route.engine == "batch" and route.use_batch is True
    assert route.use_dense is False
    assert route.backend == "numpy" and route.fallback_backend == "numpy"
    assert "batch" in route.reason and "device" in route.reason


@pytest.mark.parametrize("name", ["numpy", "iree"])
def test_batch_honours_an_explicitly_named_backend(name, monkeypatch):
    """An explicit backend names BOTH halves of a batch route, on any box.

    Precedence is the policy-wide one (``backend=`` argument, then
    ``EVOGP_EVOXC_BACKEND``, then the device probe), so the name wins even when a
    device is available and no probe is needed at all.
    """
    for available in (True, False):
        on_device_box(monkeypatch, available=available)
        for pop in (1, ABOVE, 10 ** 3):
            explicit = dense.resolve_route(pop, eval_mode="batch", backend=name)
            assert explicit.engine == "batch" and explicit.use_batch is True
            assert explicit.backend == name and explicit.fallback_backend == name
    # ... and the same name through the environment, with the probe never taken

    def explode():
        raise AssertionError("the route policy probed the device for an explicit backend")

    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", None)
    monkeypatch.setattr(dense.torch.cuda, "is_available", explode)
    monkeypatch.setenv(EVOXC_BACKEND_ENV, name)
    routed = dense.resolve_route(BELOW, eval_mode="batch")
    assert routed.engine == "batch"
    assert routed.backend == name and routed.fallback_backend == name
    assert dense._DEVICE_AVAILABLE is None  # not even memoized


@pytest.mark.parametrize("pop", [1, BELOW, ABOVE, 10 ** 3])
def test_batch_route_has_no_pop_or_rows_gate(pop, monkeypatch):
    """No threshold and no rows ceiling: the batch engine declines for itself.

    ``EVOGP_DENSE_AUTO_MIN_POP`` and ``EVOGP_DENSE_AUTO_MAX_ROWS`` belong to the
    dense half of the policy, which batch never enters -- a huge ``rows`` value
    must not turn a batch route into a specialized one, and moving the thresholds
    anywhere leaves the batch decision untouched.
    """
    on_device_box(monkeypatch)
    for rows in (None, ROWS_OK, ROWS_OVER):
        route = dense.resolve_route(pop, rows=rows, eval_mode="batch")
        assert route.engine == "batch" and route.use_batch is True, (pop, rows)
        assert route.use_dense is False and route.backend == "iree", (pop, rows)
        assert dense.should_use_dense(pop, rows=rows, eval_mode="batch") is False
    monkeypatch.setenv(dense.AUTO_MIN_POP_ENV, str(10 ** 6))
    monkeypatch.setenv(dense.AUTO_MAX_ROWS_ENV, "1")
    for rows in (None, ROWS_OK, ROWS_OVER):
        route = dense.resolve_route(pop, rows=rows, eval_mode="batch")
        assert route.engine == "batch" and route.use_batch is True, (pop, rows)
        assert dense.AUTO_MIN_POP_ENV not in route.reason, "batch is not the dense half"


def test_route_engine_and_use_dense_stay_consistent():
    """``Route`` owns the invariant ``use_dense == (engine == "dense")``.

    A call site can therefore never end up trying two engines at once: a batch
    route must say ``use_dense=False``, a dense engine must say
    ``use_dense=True``, an old-style ``Route(True, ...)`` (no ``engine=``) is
    promoted to ``engine="dense"``, and an unknown engine name is rejected.
    """
    assert dense.Route(False, "numpy", "numpy").engine == "specialized"
    assert dense.Route(False, "numpy", "numpy").use_batch is False
    promoted = dense.Route(True, "iree", "numpy")
    assert promoted.engine == "dense" and promoted.use_batch is False
    batch = dense.Route(False, "iree", "numpy", "why", engine="batch")
    assert batch.engine == "batch" and batch.use_batch is True
    with pytest.raises(ValueError):
        dense.Route(True, "iree", "numpy", engine="batch")
    with pytest.raises(ValueError):
        dense.Route(False, "iree", "numpy", engine="dense")
    with pytest.raises(ValueError):
        dense.Route(False, "iree", "numpy", engine="merged")


def test_batch_route_equality_ignores_the_engine_field():
    """``Route.engine`` is ``compare=False``: pin it, never compare whole routes.

    A batch route compares EQUAL to an old three-field record with the same
    backends and reason, so "the policy picked batch" can only be asserted
    through ``route.engine`` / ``route.use_batch`` (which is what the tests above
    do).  The exclusion is deliberate: it keeps every equality check written
    against the previous three-field ``Route`` meaning the same thing.
    """
    old_style = dense.Route(False, "iree", "numpy", "why")
    batch = dense.Route(False, "iree", "numpy", "why", engine="batch")
    assert old_style == batch  # the derived field is excluded on purpose
    assert old_style.engine != batch.engine
    assert old_style.use_batch is False and batch.use_batch is True


# --------------------------------------------------------------------------- #
# 8. `auto` NEVER selects the batch engine
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("backend", [None, "numpy", "iree"])
@pytest.mark.parametrize(
    "eval_mode", [None, "auto", "dense", "dynamic", "specialized"]
)
@pytest.mark.parametrize("pop", [1, 16, 64, 256, 10 ** 4])
def test_auto_never_selects_the_batch_route(
    pop, backend, eval_mode, available, monkeypatch
):
    """Batch is opt-in ONLY: no policy combination may route a call there.

    The merged graph is a different trade-off (one graph for the whole
    population, its own ``EVOGP_BATCH_MIN_POP``), so it is never a silent default
    -- not at the large-pop device case that picks dense, and not at any
    ``dense`` / ``dynamic`` / ``specialized`` opt-in either.  Only
    ``EVOGP_EVAL_MODE=batch`` (section 7) reaches it.
    """
    on_device_box(monkeypatch, available=available)
    for rows in (None, ROWS_OK, ROWS_OVER):
        route = dense.resolve_route(
            pop, rows=rows, backend=backend, eval_mode=eval_mode
        )
        assert route.engine != "batch", (pop, backend, eval_mode, available, rows)
        assert route.use_batch is False, (pop, backend, eval_mode, available, rows)
        assert route.engine in dense.ENGINES
        assert route.backend in dense_engine_backends()
        assert route.fallback_backend in dense_engine_backends()


def dense_engine_backends():
    """The backend names the engine resolves (imported lazily: it is a tuple)."""
    from evogp.tree.evoxc_engine import SUPPORTED_BACKENDS

    return tuple(SUPPORTED_BACKENDS)
