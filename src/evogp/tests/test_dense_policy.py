"""Policy-surface regression tests for the dense evaluator (``evogp.tree.dense_engine``).

The dense route ("one compiled graph for the whole population", the value and
SR-loss variants of :func:`evogp.tree.dense_engine.plan_dense`) is selected by a
small policy surface that is read *per call* and is easy to break silently: every
knob is an explicit argument falling back to an ``EVOGP_*`` environment variable.
This module pins that surface -- the ``dynamic`` alias of ``dense``, the
``_iter_plan_chunks`` device-resident refusals, and the trace cap with its
once-per-process warning.

Coverage
--------
* ``resolve_eval_mode``: the ``dynamic`` -> ``dense`` alias (case/whitespace
  folded, before validation), the canonical ``EVAL_MODES`` table, unknown values
  raising a ``ValueError`` naming ``EVOGP_EVAL_MODE``, empty/``None`` meaning
  ``auto``, the explicit-argument-wins rule and the per-call environment read.
* The alias as a FIRST-CLASS spelling: ``should_use_dense`` /
  ``auto_rows_decline`` behave exactly as for ``dense`` (an explicit opt-in is
  never second-guessed by the ``auto`` rows rule), while ``auto`` keeps its
  measured thresholds.
* ``_iter_plan_chunks(device_resident=True)`` refuses the numpy backend and any
  chunked plan (needs one single-chunk plan), and its payload contract (the loss
  variant needs ``labels``, the value variant takes none).
* The trace cap: ``EVOGP_DENSE_MAX_TREE_LEN`` / ``max_tree_len`` resolution, the
  ``L = min(cap, row_len)`` rule, the over-cap decline's message, and the deduped
  ``RuntimeWarning`` -- emitted only on a non-numpy backend, once per process,
  never consuming the flag for any other decline reason.

Provenance and constraints
--------------------------
These checks are ported from the dev battery ``src/evogp/tree/device_resident_check.py``
(stages ``policy`` and ``seam``, which report PASS/FAIL tables, not pytest
outcomes).  Like the battery, this module

* never builds or runs an iree graph: ``backend="iree"`` is only a *name* passed
  to ``prepare_dense`` / ``_iter_plan_chunks``, which resolve it without
  compiling anything;
* never asserts wall-clock time (engine timing belongs in the standalone
  ``perf_probe.py``), and keeps its one compiled graph small
  (pop 8 x 8 rows x L 16, numpy backend, milliseconds to build);
* is hermetic: the knobs it depends on are cleared per test by the autouse
  fixture below, the route policy's device probe is pinned there too, every
  scenario resets the process-global once-per-process flag through
  ``monkeypatch``, and warnings are captured inside
  ``warnings.catch_warnings`` so nothing leaks into the suite summary;
* leaves the ROUTE policy (dense vs specialized and the backend it picks:
  ``resolve_route`` / ``device_available``, the device-vs-CPU branches and the
  explicit-backend precedence) to ``test_route_policy.py``, and asks the policy
  here only through explicit ``backend=`` / ``eval_mode=`` arguments.
* works under ``EVOGP_EVAL_MODE=dense`` (and any other forced mode), because every
  policy entry point here is called with an explicit argument.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pytest

from evogp.tree import NType, dense_engine as dense, evoxc_engine
from evogp.tree.utils import Func

#: Env var naming the backend the engine would resolve, taken from the ENGINE's own
#: constant (``evoxc_engine.BACKEND_ENV``) so the autouse clearing fixture below can
#: never target a name the engine stopped reading.
EVOXC_BACKEND_ENV = evoxc_engine.BACKEND_ENV
assert EVOXC_BACKEND_ENV == "EVOGP_EVOXC_BACKEND", (
    f"evoxc_engine.BACKEND_ENV is now {EVOXC_BACKEND_ENV!r}: update this module's "
    "docstring/expectations instead of clearing a stale name"
)

VAR = int(NType.VAR)
UFUNC = int(NType.UFUNC)
SIN = int(Func.SIN)

#: Smallest fixture the dense policy can be exercised on (see the module docstring):
#: L = 16, P = 8, N = 8, one 8-node chain plus padding rows.
POP_SIZE = 8
ROWS = 8
INPUT_LEN = 2
ROW_LEN = 16
CHAIN_LEN = 8

#: A row count far above ``DEFAULT_AUTO_MAX_ROWS``: what the ``auto`` rows rule
#: declines and an explicit ``dense``/``dynamic`` never does.
HUGE_ROWS = 10 ** 6

#: Budgets (MiB) walked to force a plan with BOTH chunk counts > 1 -- ported from
#: the battery's ``_CANDIDATE_BUDGETS``.  ``_chunk_shape`` shaves ``n_chunk`` only
#: from ``pop_size`` individuals up, and shaves ``P`` below one individual's
#: element budget, so both-chunked is the regime where the per-chunk element
#: budget falls under ``pop_size``.
DOUBLE_CHUNK_BUDGETS_MB = (0.0015, 0.0012, 0.002, 0.0008, 0.0004)


@pytest.fixture(autouse=True)
def hermetic_knobs(monkeypatch):
    """Start every test from the documented defaults, whatever the ambient env says.

    The knobs read here (the eval-mode master switch, the trace cap, the ``auto``
    policy thresholds and the engine backend) are cleared, so "the default is
    ``auto``" / "the default cap is 64" / "no backend was named explicitly" mean the
    same thing on a box that exports them.  The route policy's device probe is
    pinned to "a device is available" (the interesting branch of the
    ``auto`` policy), so no assertion here depends on the box the suite runs on --
    a CPU-box scenario is a test of its own in ``test_route_policy.py``.  Each test
    then sets what it needs on the same ``monkeypatch`` and pytest restores the
    ambient values at teardown.
    """
    for name in (
        dense.EVAL_MODE_ENV,
        dense.MAX_TREE_LEN_ENV,
        dense.AUTO_MIN_POP_ENV,
        dense.AUTO_MAX_ROWS_ENV,
        EVOXC_BACKEND_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", True)


# --------------------------------------------------------------------------- #
# fixtures: chains, plans, payloads
# --------------------------------------------------------------------------- #
def chain_rows(
    longest: int, row_len: int, pop_size: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(node_value, node_type, subtree_size)`` of a chain of EXACTLY ``longest`` nodes.

    Ported from the battery's ``device_resident_check.cap_chain``: a right-leaning
    chain of ``longest - 1`` ``SIN`` nodes over one ``VAR``, so the REALIZED tree
    length is deterministic -- which is what the trace-cap checks need (a random
    population's realized length is not).  Row 0 carries the chain; extra rows are
    single-``VAR`` trees, so the same helper builds a ``(pop_size, row_len)``
    population.  The declared lengths stay inside ``row_len`` by construction.
    """
    assert 1 <= longest <= row_len, f"chain of {longest} nodes does not fit row_len={row_len}"
    values = np.zeros((pop_size, row_len), dtype=np.float32)
    types = np.zeros((pop_size, row_len), dtype=np.int16)
    sizes = np.ones((pop_size, row_len), dtype=np.int16)
    for index in range(longest - 1):
        types[0, index] = UFUNC
        values[0, index] = float(SIN)
        sizes[0, index] = longest - index
    types[0, longest - 1] = VAR
    values[0, longest - 1] = 0.0
    sizes[0, longest - 1] = 1
    return values, types, sizes


def host_rows(rows: int, columns: int, seed: int = 0) -> np.ndarray:
    """Deterministic ``(rows, columns)`` float32 payload (no global RNG involved)."""
    return np.ascontiguousarray(
        np.random.RandomState(seed).standard_normal((rows, columns)), dtype=np.float32
    )


#: The data payloads every plan in this module is run against.
X = host_rows(ROWS, INPUT_LEN, seed=11)
LABELS = host_rows(ROWS, 1, seed=12)


def tiny_plan(
    *,
    per_row: bool = False,
    budget_mb: Optional[float] = None,
    variant: str = dense.VALUE_VARIANT,
    rows: int = ROWS,
    max_tree_len: Optional[int] = None,
):
    """A small numpy ``DensePlan`` built from the module's chain fixture.

    ``backend``/``update_mode`` are explicit so the plan never depends on the
    ambient ``EVOXC_*`` environment; the population shape (P=8, N=8, L=16) keeps
    this module's compiled graphs small (one per variant/chunk shape, numpy).
    """
    values, types, sizes = chain_rows(CHAIN_LEN, ROW_LEN, POP_SIZE)
    return dense.plan_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        rows,
        per_row=per_row,
        max_tree_len=max_tree_len,
        budget_mb=budget_mb,
        update_mode="grow",
        backend="numpy",
        variant=variant,
    )


def find_double_chunked_budget():
    """``(budget_mb, plan)`` of the first budget producing a both-chunked plan."""
    for budget in DOUBLE_CHUNK_BUDGETS_MB:
        plan = tiny_plan(budget_mb=budget)
        if plan.p_chunks > 1 and plan.n_chunks > 1:
            return budget, plan
    raise AssertionError(
        f"no budget in {DOUBLE_CHUNK_BUDGETS_MB} (MiB) chunked both P and N for a "
        f"{POP_SIZE}x{ROWS} plan with L={ROW_LEN}: the single-chunk requirement of "
        "the device-resident seam could not be exercised"
    )


@pytest.fixture(scope="module")
def value_single_chunk():
    """The value variant, small enough to be one chunk: ``(plan, executable)``."""
    plan = tiny_plan()
    assert (plan.p_chunks, plan.n_chunks) == (1, 1), (plan.p_chunks, plan.n_chunks)
    return plan, dense.build_dense(plan, backend="numpy")


@pytest.fixture(scope="module")
def value_chunked():
    """The value variant forced into several P- and N-chunks: ``(plan, executable)``."""
    _, plan = find_double_chunked_budget()
    assert plan.p_chunks > 1 and plan.n_chunks > 1, (plan.p_chunks, plan.n_chunks)
    return plan, dense.build_dense(plan, backend="numpy")


@pytest.fixture(scope="module")
def loss_single_chunk():
    """The SR-loss variant (one chunk): ``(plan, executable)``; needs ``labels``."""
    plan = tiny_plan(variant=dense.LOSS_VARIANT)
    assert (plan.p_chunks, plan.n_chunks) == (1, 1), (plan.p_chunks, plan.n_chunks)
    return plan, dense.build_dense(plan, backend="numpy")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def iterate_chunks(plan, executable, labels=None, **kwargs):
    """``_iter_plan_chunks`` on the module's payloads (numpy backend unless overridden)."""
    settings = {"backend_name": "numpy", "device": None}
    settings.update(kwargs)
    return dense._iter_plan_chunks(plan, executable, X, labels, **settings)


def first_chunk(plan, executable, labels=None, **kwargs):
    """The first yielded ``((p0, p_chunk, n0, n_chunk), raw)`` -- raising eagerly.

    ``_iter_plan_chunks`` is a generator, so its pre-flight contract checks only run
    on the first ``next()``; this helper makes that step explicit in the tests.
    """
    return next(iterate_chunks(plan, executable, labels, **kwargs))


def prep_attempt(values, types, sizes, *, backend=None, output_len=1):
    """``prepare_dense`` on raw rows -> ``(DensePrep or None, DenseUnsupported reason or None)``.

    The once-per-process warning flag is deliberately NOT touched here: the caller
    owns it through ``monkeypatch``, which is what makes "warns exactly once across
    consecutive calls" observable.
    """
    try:
        return dense.prepare_dense(
            values, types, sizes, INPUT_LEN, output_len, backend=backend
        ), None
    except dense.DenseUnsupported as exc:
        return None, exc.reason


def cap_attempt(longest, row_len, *, backend=None, output_len=1):
    """``prep_attempt`` on a ``chain_rows(longest, row_len)`` population."""
    return prep_attempt(*chain_rows(longest, row_len), backend=backend, output_len=output_len)


def runtime_warnings(recorded):
    """The ``RuntimeWarning`` messages captured by ``recorded`` (the trace-cap one)."""
    return [str(item.message) for item in recorded if issubclass(item.category, RuntimeWarning)]


# --------------------------------------------------------------------------- #
# 1. `dynamic` as an alias of `dense` in `resolve_eval_mode`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spelling", ["dynamic", "DYNAMIC", "  DyNaMic ", "\tDynamic\n"])
def test_dynamic_alias_resolves_to_dense(spelling):
    """``EVAL_MODE_ALIASES`` is folded in after case folding/trimming."""
    assert dense.resolve_eval_mode(spelling) == "dense"


@pytest.mark.parametrize("mode", list(dense.EVAL_MODES))
def test_canonical_eval_modes_round_trip_case_insensitively(mode):
    assert dense.resolve_eval_mode(mode) == mode
    assert dense.resolve_eval_mode(f"  {mode.upper()}  ") == mode


def test_eval_mode_tables_stay_canonical():
    """The alias lives in its own table; ``EVAL_MODES`` stays the canonical tuple."""
    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "auto")
    assert dense.DEFAULT_EVAL_MODE == "auto"
    assert dict(dense.EVAL_MODE_ALIASES) == {"dynamic": "dense"}
    assert dense.EVAL_MODE_ENV == "EVOGP_EVAL_MODE"
    # the alias target is canonical, i.e. `dynamic` is not a mode of its own
    assert set(dense.EVAL_MODE_ALIASES.values()) <= set(dense.EVAL_MODES)
    assert "dynamic" not in dense.EVAL_MODES


@pytest.mark.parametrize("unknown", ["dynamic2", "dens", "DINAMIC", "0"])
def test_unknown_eval_mode_raises_naming_the_env_var(unknown, monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        dense.resolve_eval_mode(unknown)
    message = str(excinfo.value)
    assert dense.EVAL_MODE_ENV in message
    assert "dynamic" in message  # the alias table is part of the message
    monkeypatch.setenv(dense.EVAL_MODE_ENV, unknown)
    with pytest.raises(ValueError, match=dense.EVAL_MODE_ENV):
        dense.resolve_eval_mode()


def test_empty_or_unset_eval_mode_is_auto(monkeypatch):
    """An empty/whitespace value means the default, in the argument AND in the env."""
    assert dense.resolve_eval_mode("") == "auto"
    assert dense.resolve_eval_mode("   ") == "auto"
    assert dense.resolve_eval_mode(None) == "auto"  # env cleared by hermetic_knobs
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "")
    assert dense.resolve_eval_mode() == "auto"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_eval_mode(" ") == "auto"  # an explicit empty value wins too


def test_explicit_eval_mode_argument_wins_over_the_env(monkeypatch):
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_eval_mode("dense") == "dense"
    assert dense.resolve_eval_mode("dynamic") == "dense"
    assert dense.resolve_eval_mode() == "specialized"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "dense")
    assert dense.resolve_eval_mode("specialized") == "specialized"
    assert dense.resolve_eval_mode("auto") == "auto"
    # the same precedence at the policy entry point
    assert dense.should_use_dense(4, eval_mode="dynamic", backend="numpy") is True
    assert dense.should_use_dense(4, eval_mode="specialized", backend="numpy") is False


def test_eval_mode_env_is_read_per_call(monkeypatch):
    """A process can flip strategy between calls (no memo of the master switch)."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "dense")
    assert dense.resolve_eval_mode() == "dense"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_eval_mode() == "specialized"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "dynamic")
    assert dense.resolve_eval_mode() == "dense"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, " AUTO ")
    assert dense.resolve_eval_mode() == "auto"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "not-a-mode")
    with pytest.raises(ValueError, match=dense.EVAL_MODE_ENV):
        dense.resolve_eval_mode()


# --------------------------------------------------------------------------- #
# 2. the alias is a first-class spelling of `dense`
# --------------------------------------------------------------------------- #
def test_dynamic_alias_targets_dense_in_the_policy():
    """`dynamic` is an explicit opt-in: dense is attempted, never rows-declined."""
    assert dense.should_use_dense(4, eval_mode="dynamic", backend="numpy") is True
    assert dense.should_use_dense(4, eval_mode="dense", backend="numpy") is True
    assert dense.auto_rows_decline(HUGE_ROWS, eval_mode="dynamic") is None
    assert dense.auto_rows_decline(HUGE_ROWS, eval_mode="dense") is None
    assert dense.auto_rows_decline(HUGE_ROWS, eval_mode="auto") is not None
    # ... and the rows rule does not sneak back in through should_use_dense
    assert (
        dense.should_use_dense(4, eval_mode="dynamic", backend="numpy", rows=HUGE_ROWS)
        is True
    )
    assert (
        dense.should_use_dense(4, eval_mode="dense", backend="numpy", rows=HUGE_ROWS)
        is True
    )
    assert (
        dense.should_use_dense(4, eval_mode="auto", backend="numpy", rows=HUGE_ROWS)
        is False
    )
    assert (
        dense.should_use_dense(4, eval_mode="specialized", backend="iree", rows=HUGE_ROWS)
        is False
    )


def test_auto_rows_rule_keeps_its_measured_thresholds():
    """Only ``auto`` is gated, and only above the inclusive ``AUTO_MAX_ROWS`` ceiling."""
    assert dense.DEFAULT_AUTO_MIN_POP == 64
    assert dense.DEFAULT_AUTO_MAX_ROWS == 2048
    assert dense.resolve_auto_min_pop() == dense.DEFAULT_AUTO_MIN_POP
    assert dense.resolve_auto_max_rows() == dense.DEFAULT_AUTO_MAX_ROWS
    assert dense.auto_rows_decline(None, eval_mode="auto") is None  # unknown row count
    assert dense.auto_rows_decline(dense.DEFAULT_AUTO_MAX_ROWS, eval_mode="auto") is None
    decline = dense.auto_rows_decline(dense.DEFAULT_AUTO_MAX_ROWS + 1, eval_mode="auto")
    assert decline is not None and dense.AUTO_MAX_ROWS_ENV in decline
    # `resolve_backend_name('iree')` only NAMES the backend here: nothing compiles,
    # nothing runs, no device is touched.
    assert dense.resolve_backend_name("iree") == "iree"
    assert (
        dense.should_use_dense(
            64,
            eval_mode="auto",
            backend="iree",
            rows=dense.DEFAULT_AUTO_MAX_ROWS,
            auto_min_pop=dense.DEFAULT_AUTO_MIN_POP,
            auto_max_rows=dense.DEFAULT_AUTO_MAX_ROWS,
        )
        is True
    )
    assert (
        dense.should_use_dense(
            64,
            eval_mode="auto",
            backend="iree",
            rows=dense.DEFAULT_AUTO_MAX_ROWS + 1,
            auto_min_pop=dense.DEFAULT_AUTO_MIN_POP,
            auto_max_rows=dense.DEFAULT_AUTO_MAX_ROWS,
        )
        is False
    )



# --------------------------------------------------------------------------- #
# 3. `_iter_plan_chunks` refusals and chunk census
# --------------------------------------------------------------------------- #
def test_device_resident_refuses_the_numpy_backend(value_single_chunk):
    """A "device-resident" result would be a host array on numpy."""
    plan, executable = value_single_chunk
    assert (plan.p_chunks, plan.n_chunks) == (1, 1)  # the plan is otherwise eligible
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        first_chunk(plan, executable, backend_name="numpy", device=None, device_resident=True)
    reason = excinfo.value.reason
    assert "device backend" in reason and "numpy" in reason


def test_device_resident_refuses_a_chunked_plan(value_chunked):
    """Device-resident results need ONE chunk: they cannot be assembled on-device.

    The refusal fires before any payload is staged or uploaded, so the backend is
    only a NAME here (``device=None``): no iree graph is built and no device is
    used -- the single-chunk requirement is checked first, on purpose.
    """
    plan, executable = value_chunked
    assert plan.p_chunks > 1 and plan.n_chunks > 1, (plan.p_chunks, plan.n_chunks)
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        first_chunk(plan, executable, backend_name="iree", device=None, device_resident=True)
    reason = excinfo.value.reason
    assert "single-chunk" in reason
    assert dense.BUDGET_MB_ENV in reason  # names the escape hatch (stop chunking)
    # the same plan IS accepted on the numpy (non-resident) path
    chunks = list(iterate_chunks(plan, executable, backend_name="numpy", device=None))
    assert len(chunks) == plan.p_chunks * plan.n_chunks


def test_iter_plan_chunks_yields_every_chunk(value_single_chunk, value_chunked, loss_single_chunk):
    """The positive path: one raw, un-read result per chunk, keys tiling (P, N)."""
    cases = [
        ("value/unchunked", value_single_chunk, None),
        ("value/chunked", value_chunked, None),
        ("loss/unchunked", loss_single_chunk, LABELS),
    ]
    for label, (plan, executable), labels in cases:
        chunks = list(iterate_chunks(plan, executable, labels))
        keys = [key for key, _ in chunks]
        expected = [
            (p0, min(plan.p_chunk, plan.pop_size - p0), n0, min(plan.n_chunk, plan.n_rows - n0))
            for p0 in range(0, plan.pop_size, plan.p_chunk)
            for n0 in ([0] if plan.per_row else range(0, plan.n_rows, plan.n_chunk))
        ]
        assert len(chunks) == len(expected) == plan.p_chunks * plan.n_chunks, label
        assert keys == expected, label
        for (_, p_chunk, _, n_chunk), raw in chunks:
            assert p_chunk >= 1 and n_chunk >= 1, label
            # the seam hands out the evoxc result itself, never a host array
            assert raw is not None and not isinstance(raw, np.ndarray), label
            assert hasattr(raw, "numpy"), label


def test_iter_plan_chunks_payload_contract(value_single_chunk, loss_single_chunk):
    """``labels`` is required by (and only by) the loss variant -- checked pre-flight."""
    plan, executable = value_single_chunk
    with pytest.raises(ValueError, match="no labels payload"):
        first_chunk(plan, executable, LABELS, backend_name="numpy", device=None)
    loss_plan, loss_executable = loss_single_chunk
    assert loss_plan.variant == dense.LOSS_VARIANT
    with pytest.raises(ValueError, match="needs the labels payload"):
        first_chunk(loss_plan, loss_executable, backend_name="numpy", device=None)


# --------------------------------------------------------------------------- #
# 4. trace cap resolution
# --------------------------------------------------------------------------- #
def test_trace_cap_default_explicit_and_env(monkeypatch):
    """``EVOGP_DENSE_MAX_TREE_LEN`` unset -> 64; explicit argument wins over the env."""
    assert dense.DEFAULT_MAX_TREE_LEN == 64
    assert dense._resolve_max_tree_len(None) == 64 == dense.DEFAULT_MAX_TREE_LEN
    assert dense._resolve_max_tree_len(32) == 32
    monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, "12")
    assert dense._resolve_max_tree_len(None) == 12
    assert dense._resolve_max_tree_len(16) == 16
    monkeypatch.delenv(dense.MAX_TREE_LEN_ENV)
    assert dense._resolve_max_tree_len(None) == 64


@pytest.mark.parametrize("bad", [0, -1, -64])
def test_trace_cap_rejects_non_positive_values(bad, monkeypatch):
    with pytest.raises(ValueError, match="must be positive"):
        dense._resolve_max_tree_len(bad)
    if bad == 0:
        monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, "0")
        with pytest.raises(ValueError, match="must be positive"):
            dense._resolve_max_tree_len(None)


def test_trace_cap_env_reaches_prepare_dense_and_explicit_wins(monkeypatch):
    """The knob the decline message names is the one ``prepare_dense`` applies."""
    monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, "12")
    prep = dense.prepare_dense(*chain_rows(10, 128), INPUT_LEN, 1, backend="numpy")
    assert (prep.trace_len, prep.max_tree_length) == (12, 10)
    over_prep, reason = cap_attempt(20, 128, backend="numpy")
    assert over_prep is None and "cap is 12" in reason
    prep = dense.prepare_dense(
        *chain_rows(20, 128), INPUT_LEN, 1, max_tree_len=32, backend="numpy"
    )
    assert prep.trace_len == 32


# --------------------------------------------------------------------------- #
# 5. L = min(cap, row_len)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "longest,row_len,expected_trace_len",
    [
        (10, 16, 16),  # a short row is unaffected by the default cap
        (10, 32, 32),
        (10, 64, 64),
        (10, 128, 64),  # a long row no longer traces its whole width
        (64, 128, 64),  # a realized 64-node tree fits the default cap exactly
    ],
)
def test_trace_len_is_min_of_cap_and_row_len(longest, row_len, expected_trace_len):
    prep = dense.prepare_dense(*chain_rows(longest, row_len), INPUT_LEN, 1, backend="numpy")
    assert prep.trace_len == expected_trace_len
    assert prep.max_tree_length == longest


def test_over_cap_tree_declines_naming_the_length_and_the_cap():
    """The over-cap decline is the ONE expensive fallback, and its reason says why."""
    prep, reason = cap_attempt(70, 128, backend="numpy")
    assert prep is None
    assert "70 nodes" in reason and "cap is 64" in reason
    assert dense.MAX_TREE_LEN_ENV in reason
    # an explicit, smaller cap is what the reason reports (L = min(cap, row_len))
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        dense.prepare_dense(
            *chain_rows(40, 128), INPUT_LEN, 1, max_tree_len=32, backend="numpy"
        )
    assert "40 nodes" in excinfo.value.reason and "cap is 32" in excinfo.value.reason


# --------------------------------------------------------------------------- #
# 6. the trace-cap warning: once per process, non-numpy only
# --------------------------------------------------------------------------- #
def test_over_cap_decline_is_silent_on_numpy(monkeypatch):
    """numpy is cheap at the cap, so the decline is not worth reporting."""
    monkeypatch.setattr(dense, "_TRACE_CAP_WARNED", False)
    assert dense.resolve_backend_name(None) == "numpy"  # the default backend
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        explicit = cap_attempt(70, 128, backend="numpy")
        default = cap_attempt(70, 128)
    for prep, reason in (explicit, default):
        assert prep is None and "70 nodes" in reason
    assert runtime_warnings(recorded) == []
    assert dense._TRACE_CAP_WARNED is False


def test_over_cap_decline_warns_once_per_process_on_a_device_backend(monkeypatch):
    """The first over-cap population on a device backend warns; later ones do not.

    ``backend="iree"`` is a NAME here (see the module docstring): only the warning
    decision is under test, and every call still declines.
    """
    monkeypatch.setattr(dense, "_TRACE_CAP_WARNED", False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        attempts = [cap_attempt(70, 128, backend="iree") for _ in range(3)]
    assert [prep for prep, _ in attempts] == [None, None, None]
    assert all("70 nodes" in reason for _, reason in attempts)
    messages = runtime_warnings(recorded)
    assert len(messages) == 1
    message = messages[0]
    assert "70 nodes" in message
    assert "L=64" in message
    assert dense.MAX_TREE_LEN_ENV in message
    assert "PER DISTINCT TREE STRUCTURE" in message
    assert dense._TRACE_CAP_WARNED is True


def test_other_declines_stay_silent_and_do_not_consume_the_once_flag(monkeypatch):
    """Every non-cap decline is silent AND leaves the once-per-process flag free."""
    monkeypatch.setattr(dense, "_TRACE_CAP_WARNED", False)
    values, types, sizes = chain_rows(4, 8)
    malformed_types = types.copy()
    malformed_types[0, 0] = VAR  # a VAR still declaring a 4-node subtree
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        multi_prep, multi_reason = cap_attempt(4, 8, backend="iree", output_len=2)
        bad_prep, bad_reason = prep_attempt(values, malformed_types, sizes, backend="iree")
        assert multi_prep is None and bad_prep is None
        assert "output_len" in multi_reason
        assert "subtree_size" in bad_reason
        assert runtime_warnings(recorded) == []
        assert dense._TRACE_CAP_WARNED is False
        # ... so the next over-cap population still gets its one report
        over_prep, over_reason = cap_attempt(70, 128, backend="iree")
    assert over_prep is None and "70 nodes" in over_reason
    assert len(runtime_warnings(recorded)) == 1
    assert dense._TRACE_CAP_WARNED is True
