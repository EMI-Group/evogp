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
* ``batch``, the opt-in THIRD engine (:mod:`evogp.tree.batch_engine`): a
  canonical mode of its own (no alias, and no alias value points at it), forced
  through the same master switch from the argument or ``EVOGP_EVAL_MODE``, and
  never reported by ``should_use_dense`` / the ``auto`` rows rule -- the batch
  engine is named by ``Route.engine`` / ``Route.use_batch`` instead.
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
* The ``auto`` trace-cap RAISE (``EVOGP_DENSE_AUTO_MAX_TREE_LEN``, bounded by the
  cap-axis envelope of ``EVOGP_DENSE_AUTO_BUDGET_MB``): the exact
  ``min(ceiling, next_power_of_two(max(base, longest)))`` arithmetic (65 / 123 /
  127 -> 128, 129 / 201 -> 256), the POWER-OF-TWO BAND that makes one compiled
  dense graph span a whole range of realized lengths, the allowance re-checked at
  the ROUNDED value (a 110-node ceiling refuses a 100-node population), the
  identity outside ``auto`` / under a pinned cap / for a population that fits,
  the refusal's message naming the binding term and the rounded request, and both
  entry points handing the raised cap to the plan seam -- plus the
  ``DenseUnsupported(device_fallback=True)`` refusal, decided before the plan so
  no numpy fallback can be reached.

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
* leaves the ROUTE policy (dense vs specialized vs the opt-in batch engine, and
  the backend it picks: ``resolve_route`` / ``device_available``, the
  device-vs-CPU branches, the batch route's iree/numpy decision and the
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
import torch

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

#: The reference shape the ``auto`` rows envelope is evaluated at.
ENVELOPE_POP = dense.DEFAULT_AUTO_MIN_POP
ENVELOPE_TREE_LEN = 64
#: One row above that shape's allowance: what the ``auto`` rows rule declines and
#: an explicit ``dense``/``dynamic`` never does.
HUGE_ROWS = dense.auto_row_allowance(ENVELOPE_POP, ENVELOPE_TREE_LEN) + 1

#: The cell the ``auto`` trace-cap RAISE was designed for (``LARGE_SCALE.md`` §7):
#: 123-node trees at pop 2 048, a 128-wide row stride and N = 1 024 data rows.
#: The default cap (64) is exceeded there while the raise's allowance (256) covers
#: the population, so ``auto`` serves it instead of falling back to per-tree numpy.
CAP_POP = 2048
CAP_ROW_LEN = 128
CAP_LONGEST = 123
CAP_ROWS = 1024
#: The cap ``auto`` actually raises the reference cell to: ``CAP_LONGEST`` rounded
#: UP to the next power of two.  ``L`` is part of the dense graph key, so the raise
#: tracks a BAND of realized lengths (65..128) rather than the exact length -- see
#: :func:`_next_power_of_two` and section 7.
CAP_RAISED = 128
#: A budget (MiB) whose ``auto`` envelope allows exactly ONE trace node at that
#: shape: ``8 * 2**20 // (2048 * 1024 * 4) == 1`` -- the envelope refusing the raise.
CAP_TIGHT_BUDGET_MB = 8

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
    a CPU-box scenario is a test of its own in ``test_route_policy.py``.  The fused
    probe is pinned to "no triton" as well, because a triton-equipped device box
    would otherwise make ``auto`` resolve to the fused route and every dense
    assertion below would be measuring a policy this file does not own.  Each test
    then sets what it needs on the same ``monkeypatch`` and pytest restores the
    ambient values at teardown.
    """
    for name in (
        dense.EVAL_MODE_ENV,
        dense.MAX_TREE_LEN_ENV,
        dense.AUTO_MIN_POP_ENV,
        dense.AUTO_MAX_ROWS_ENV,
        dense.AUTO_BUDGET_MB_ENV,
        dense.AUTO_MAX_TREE_LEN_ENV,
        EVOXC_BACKEND_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", True)
    monkeypatch.setattr(dense, "_FUSED_AVAILABLE", False)


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
    """The alias lives in its own table; ``EVAL_MODES`` stays the canonical tuple.

    ``batch`` is the opt-in THIRD engine: a canonical mode (never an alias, and no
    alias value may point at it), listed before ``auto`` so the default stays the
    last entry, and named in ``ENGINES`` next to the two engines the policy had
    before it.
    """
    assert tuple(dense.EVAL_MODES) == ("dense", "specialized", "batch", "auto")
    assert tuple(dense.ENGINES) == ("specialized", "dense", "batch")
    assert dense.DEFAULT_EVAL_MODE == "auto"
    assert dense.DEFAULT_EVAL_MODE == dense.EVAL_MODES[-1]
    assert dict(dense.EVAL_MODE_ALIASES) == {"dynamic": "dense"}
    assert dense.EVAL_MODE_ENV == "EVOGP_EVAL_MODE"
    # the alias target is canonical, i.e. `dynamic` is not a mode of its own
    assert set(dense.EVAL_MODE_ALIASES.values()) <= set(dense.EVAL_MODES)
    assert "dynamic" not in dense.EVAL_MODES
    # every engine names a canonical mode, and `auto` (a policy, not an engine)
    # names none
    assert set(dense.ENGINES) <= set(dense.EVAL_MODES)
    assert "auto" not in dense.ENGINES
    # `batch` is reached by its own spelling only, never through the alias table
    assert "batch" in dense.EVAL_MODES and "batch" in dense.ENGINES
    assert "batch" not in dense.EVAL_MODE_ALIASES
    assert "batch" not in set(dense.EVAL_MODE_ALIASES.values())


@pytest.mark.parametrize("unknown", ["dynamic2", "dens", "DINAMIC", "0"])
def test_unknown_eval_mode_raises_naming_the_env_var(unknown, monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        dense.resolve_eval_mode(unknown)
    message = str(excinfo.value)
    assert dense.EVAL_MODE_ENV in message
    assert "dynamic" in message  # the alias table is part of the message
    assert "batch" in message  # ... and so is every canonical mode, batch included
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
# 1b. `batch`: the opt-in third engine is a mode of the same master switch
# --------------------------------------------------------------------------- #
def test_batch_mode_is_forced_by_the_argument_and_by_the_env(monkeypatch):
    """``batch`` reaches ``resolve_eval_mode`` exactly like ``dense`` does.

    The mode is resolved per call from the explicit argument or from
    ``EVOGP_EVAL_MODE`` (no memo), so flipping either between calls flips the
    resolved mode back -- which is what lets a process compare routes without a
    restart, and what makes a forced-batch run independent of the caller.
    """
    assert dense.resolve_eval_mode("batch") == "batch"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "batch")
    assert dense.resolve_eval_mode() == "batch"
    assert dense.resolve_eval_mode(None) == "batch"
    # per-call read: the environment is consulted fresh on every call
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_eval_mode() == "specialized"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "batch")
    assert dense.resolve_eval_mode() == "batch"


def test_batch_mode_obeys_the_explicit_argument_wins_rule(monkeypatch):
    """An explicit ``eval_mode=`` always beats ``EVOGP_EVAL_MODE``, batch included."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "batch")
    assert dense.resolve_eval_mode("specialized") == "specialized"
    assert dense.resolve_eval_mode("dense") == "dense"
    assert dense.resolve_eval_mode("auto") == "auto"
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "specialized")
    assert dense.resolve_eval_mode("batch") == "batch"
    # an explicit empty value still means the default, not the ambient batch
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "batch")
    assert dense.resolve_eval_mode(" ") == "auto"


@pytest.mark.parametrize("pop", [1, 4, POP_SIZE, 10 ** 4])
@pytest.mark.parametrize("rows", [None, ROWS, HUGE_ROWS])
def test_batch_mode_is_never_the_dense_half_of_the_policy(pop, rows, monkeypatch):
    """A ``batch`` opt-in is NOT a dense opt-in for the wired call sites.

    The batch engine owns the whole population (reported by ``Route.engine`` /
    ``Route.use_batch``), so ``should_use_dense`` must stay ``False`` at every
    size and row count -- and the ``auto`` rows rule, which belongs to the dense
    half, must not report a decline for it either.
    """
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", True)
    for backend in ("numpy", "iree"):
        assert (
            dense.should_use_dense(
                pop,
                eval_mode="batch",
                backend=backend,
                rows=rows,
                tree_len=ENVELOPE_TREE_LEN,
            )
            is False
        ), (pop, backend, rows)
    assert (
        dense.auto_rows_decline(
            rows,
            eval_mode="batch",
            pop_size=ENVELOPE_POP,
            tree_len=ENVELOPE_TREE_LEN,
        )
        is None
    )


# --------------------------------------------------------------------------- #
# 2. the alias is a first-class spelling of `dense`
# --------------------------------------------------------------------------- #
def test_dynamic_alias_targets_dense_in_the_policy(monkeypatch):
    """`dynamic` is an explicit opt-in: dense is attempted, never rows-declined."""
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", True)
    assert dense.should_use_dense(4, eval_mode="dynamic", backend="numpy") is True
    assert dense.should_use_dense(4, eval_mode="dense", backend="numpy") is True
    assert dense.auto_rows_decline(HUGE_ROWS, eval_mode="dynamic") is None
    assert dense.auto_rows_decline(HUGE_ROWS, eval_mode="dense") is None
    assert (
        dense.auto_rows_decline(
            HUGE_ROWS,
            eval_mode="auto",
            pop_size=ENVELOPE_POP,
            tree_len=ENVELOPE_TREE_LEN,
        )
        is not None
    )
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


def test_auto_rows_rule_is_a_memory_envelope():
    """Only ``auto`` is gated, and it is gated on a BYTE budget, not a row count."""
    assert dense.DEFAULT_AUTO_MIN_POP == 64
    # unpinned by default: the memory envelope alone decides (``0`` = no pin)
    assert dense.DEFAULT_AUTO_MAX_ROWS == 0
    assert dense.resolve_auto_min_pop() == dense.DEFAULT_AUTO_MIN_POP
    assert dense.resolve_auto_max_rows() == 0
    assert dense.resolve_auto_budget_mb() == dense.DEFAULT_AUTO_BUDGET_MB == 8192
    # unknown row count, or an unknown shape: nothing the rule can decline
    assert dense.auto_rows_decline(None, eval_mode="auto") is None
    assert dense.auto_rows_decline(10 ** 9, eval_mode="auto") is None
    allowance = dense.auto_row_allowance(64, 64)
    assert allowance == 8 * 2 ** 30 // (64 * 64 * 4)
    assert (
        dense.auto_rows_decline(allowance, eval_mode="auto", pop_size=64, tree_len=64)
        is None
    )
    decline = dense.auto_rows_decline(
        allowance + 1, eval_mode="auto", pop_size=64, tree_len=64
    )
    assert decline is not None and dense.AUTO_BUDGET_MB_ENV in decline
    # `resolve_backend_name('iree')` only NAMES the backend here: nothing compiles,
    # nothing runs, no device is touched.
    assert dense.resolve_backend_name("iree") == "iree"
    kwargs = dict(
        eval_mode="auto",
        backend="iree",
        auto_min_pop=dense.DEFAULT_AUTO_MIN_POP,
        auto_max_rows=0,
        tree_len=64,
    )
    assert dense.should_use_dense(64, rows=allowance, **kwargs) is True
    assert dense.should_use_dense(64, rows=allowance + 1, **kwargs) is False



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


# --------------------------------------------------------------------------- #
# 7. the `auto` trace-cap raise (EVOGP_DENSE_AUTO_MAX_TREE_LEN)
# --------------------------------------------------------------------------- #
# Under `auto` ONLY, and only when the trace cap was not pinned, the entry points
# (`evaluate_dense` / `sr_fitness_dense`) raise the cap for an over-cap population
# to `min(ceiling, next_power_of_two(max(base, longest)))` -- the cap-axis twin of
# the rows rule -- and, when that allowance cannot reach the ROUNDED request,
# decline with `DenseUnsupported(..., device_fallback=True)` so the call site
# continues on the DEVICE backend instead of per-tree numpy.
#
# The rounding is the point of the rule, not a detail: `L` is part of the dense
# graph key, so a cap that tracked the realized longest tree exactly would compile
# a fresh graph every generation of a GP loop (measured `gp_step` 10 014.83 ms at
# the drifting cap against 124.46 ms at a pinned L=128), while a power-of-two band
# (65..128, 129..256) shares ONE graph for a few percent more trace work.
#
# Everything below is policy algebra plus a `_plan_entry` seam: the raise is
# inspected at the scale it was designed for (pop 2 048, 123-node trees, N = 1 024)
# WITHOUT building a graph, which is why those caps are large numbers while the
# only compiled plans in this module stay at L = 16 (sections 1-6).  The call-site
# half (`decline_backend`) lives in `test_route_policy.py`.

#: The entry points' data payloads at the reference cell's shape.
CAP_X = host_rows(CAP_ROWS, INPUT_LEN, seed=21)
CAP_LABELS = host_rows(CAP_ROWS, 1, seed=22)
CAP_PER_ROW_X = host_rows(CAP_POP, INPUT_LEN, seed=23)


class PlanCaptured(Exception):
    """Sentinel raised by the ``_plan_entry`` seam: carries the cap it was handed."""


def cap_population(pop_size: int = CAP_POP):
    """``(node_value, node_type, subtree_size)`` of the raise's reference population.
    One ``CAP_LONGEST``-node chain in row 0 (so the realized longest tree is what
    ``subtree_size[:, 0].max()`` answers) and single-``VAR`` padding rows, at the
    ``CAP_ROW_LEN`` stride -- i.e. every tree past the default cap of 64, which is
    the only state in which the raise (or its refusal) is decided.
    """
    return chain_rows(CAP_LONGEST, CAP_ROW_LEN, pop_size)


def capture_plan(monkeypatch) -> dict:
    """Replace ``_plan_entry`` by a recorder; the dense plan is never built.

    The fake records the arguments the entry point resolved (that is where the
    raised cap becomes visible, and ``n_rows`` is the sixth POSITIONAL argument)
    and raises :class:`PlanCaptured`, which is NOT a ``DenseUnsupported`` -- so no
    fallback can hide a call that reached the plan.
    """
    recorded: dict = {}

    def fake(*args, **kwargs):
        recorded.update(kwargs)
        recorded["args"] = args
        recorded["n_rows"] = args[5]
        raise PlanCaptured(kwargs.get("max_tree_len"))

    monkeypatch.setattr(dense, "_plan_entry", fake)
    return recorded


def forbid_plan(monkeypatch) -> None:
    """Make the plan seam AND the population's host decode explode when reached."""

    def explode(*args, **kwargs):
        raise AssertionError(
            "a population the auto raise refused must not reach the dense plan"
        )

    monkeypatch.setattr(dense, "_plan_entry", explode)
    monkeypatch.setattr(dense, "prepare_dense", explode)


def call_cap_entry(entry: str, **kwargs):
    """Drive one of the two ``auto`` entry points on the reference population."""
    values, types, sizes = cap_population()
    if entry == "value":
        return dense.evaluate_dense(
            values,
            types,
            sizes,
            INPUT_LEN,
            1,
            CAP_PER_ROW_X,
            per_row=True,
            **kwargs,
        )
    return dense.sr_fitness_dense(
        values, types, sizes, INPUT_LEN, 1, CAP_X, CAP_LABELS, **kwargs
    )


def test_auto_cap_rule_arithmetic_is_exact():
    """``min(ceiling, next_power_of_two(max(base, longest)))`` on the documented cells.

    The cap ``auto`` raises to is the population's realized longest tree ROUNDED UP
    to a power of two -- never the realized length itself -- because ``L`` is part
    of the dense graph key: an exact raise would compile a fresh graph every time
    the longest tree drifts, i.e. once per generation of a GP loop.
    """
    assert dense.AUTO_MAX_TREE_LEN_ENV == "EVOGP_DENSE_AUTO_MAX_TREE_LEN"
    assert dense.DEFAULT_AUTO_MAX_TREE_LEN == 256
    assert dense.resolve_auto_max_tree_len() == dense.DEFAULT_AUTO_MAX_TREE_LEN
    # p2048-L128-i32-N1024-d7: envelope 1024, ceiling 256 -> allowance 256
    envelope = dense.DEFAULT_AUTO_BUDGET_MB * 2 ** 20 // (CAP_POP * CAP_ROWS * 4)
    assert envelope == 1024
    assert dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS) == 256
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST,
            CAP_ROW_LEN,
            eval_mode="auto",
            pop_size=CAP_POP,
            rows=CAP_ROWS,
        )
        == CAP_RAISED
    )
    assert (
        dense.auto_cap_decline(
            CAP_LONGEST, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        is None
    )
    # the hypothetical 201-node cell at a 256-wide stride / N 1024 is served too --
    # 201 rounds UP to 256, which is exactly the raise's default ceiling
    assert (
        dense.auto_trace_cap(
            201, 256, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == 256
    )
    # per_row=True is ONE data row per tree, so the envelope cannot bind there
    assert dense.auto_tree_len_allowance(CAP_POP, None) == 256
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST, CAP_ROW_LEN, eval_mode="auto", pop_size=CAP_POP
        )
        == CAP_RAISED
    )
    # ... and the raise never invents a cap: an allowance below the ROUNDED request
    # refuses (300 nodes ask for 512, far above the 256-node allowance)
    assert (
        dense.auto_trace_cap(
            300, 512, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == dense.DEFAULT_MAX_TREE_LEN
    )
    assert (
        dense.auto_cap_decline(300, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
        is not None
    )


@pytest.mark.parametrize(
    "longest, row_len, expected",
    [
        (65, CAP_ROW_LEN, 128),  # just past the base cap 64 -> the 65..128 band
        (CAP_LONGEST, CAP_ROW_LEN, 128),  # the reference cell (123 nodes)
        (127, CAP_ROW_LEN, 128),  # the top of the same band
        (201, 256, 256),  # the 129..256 band, served right at the ceiling
    ],
)
def test_auto_cap_rounds_a_realized_length_up_to_a_power_of_two(longest, row_len, expected):
    """65 / 123 / 127 nodes -> 128 and 201 -> 256: the raise rounds UP, never down.

    The base cap is still 64 and the reference cell's allowance still 256, so the
    rounding alone decides these answers: a population whose longest tree drifted
    from 65 to 127 keeps hitting ONE ``L = 128`` compiled graph instead of paying a
    fresh compile at every intermediate length.
    """
    assert dense.DEFAULT_MAX_TREE_LEN == 64
    assert dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS) == 256
    assert (
        dense.auto_trace_cap(
            longest, row_len, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == expected
    )
    # every one of those cells is SERVED (the refusal is the other half of the rule)
    assert (
        dense.auto_cap_decline(
            longest, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        is None
    )


def test_one_power_of_two_band_shares_the_effective_cap():
    """The point of the change: a drifting longest tree keeps ONE compiled graph.

    Two populations inside one band resolve to the SAME effective cap (123 / 125 /
    127 -> 128), and -- since ``128`` is itself a power of two above the base cap 64
    -- so does a population whose longest tree reaches the top of the band.  Only
    the generation that CROSSES the boundary (128 -> 129) re-splits the graph, which
    is the change's residual limit: a GP loop recompiles once per power-of-two
    boundary, never once per realized length.
    """
    cell = dict(eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
    assert [dense.auto_trace_cap(n, CAP_ROW_LEN, **cell) for n in (123, 125, 127)] == [
        128,
        128,
        128,
    ]
    # ... and the boundary itself belongs to the LOWER band: 128 stays 128 (the
    # raise's answer, not the base cap, because 128 > 64)
    assert dense.auto_trace_cap(128, CAP_ROW_LEN, **cell) == 128
    assert dense.auto_trace_cap(129, 256, **cell) == 256
    assert dense.auto_trace_cap(201, 256, **cell) == 256
    # both bands sit under the 256-node allowance, so neither declines
    assert dense.auto_cap_decline(123, **cell) is None
    assert dense.auto_cap_decline(129, **cell) is None


def test_the_allowance_is_checked_at_the_rounded_cap():
    """A ceiling above the realized length but below the ROUNDED request REFUSES.

    ``longest=100`` would be raised to the 128-node band, so a
    ``EVOGP_DENSE_AUTO_MAX_TREE_LEN=110`` ceiling cannot serve it even though 110 is
    above the 100 real nodes: the raise never traces a non-power-of-two cap and
    never clamps the rounded request down to the allowance (no partial raise).  Once
    the ceiling reaches the rounded value the very same population is served.
    """
    cell = dict(eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
    assert dense.DEFAULT_MAX_TREE_LEN == 64
    assert (
        dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS, auto_max_tree_len=110) == 110
    )
    # 100 real nodes -> a 128-node request, 110 allowed -> the base cap (a DECLINE)
    assert (
        dense.auto_trace_cap(100, CAP_ROW_LEN, auto_max_tree_len=110, **cell)
        == dense.DEFAULT_MAX_TREE_LEN
    )
    reason = dense.auto_cap_decline(100, auto_max_tree_len=110, **cell)
    assert reason is not None
    assert f"{dense.AUTO_MAX_TREE_LEN_ENV}=110 is the tighter term" in reason
    assert "128" in reason  # the rounded request the allowance does not reach
    # the ceiling reaching the ROUNDED value serves the same population
    assert (
        dense.auto_trace_cap(100, CAP_ROW_LEN, auto_max_tree_len=128, **cell) == 128
    )
    assert dense.auto_cap_decline(100, auto_max_tree_len=128, **cell) is None


def test_auto_tree_len_allowance_is_the_envelope_solved_for_the_trace_length(
    monkeypatch,
):
    """``min(ceiling, floor(budget_bytes / (pop_size * rows * 4)))``, term by term."""
    # no population shape: the ceiling alone (the envelope cannot be applied)
    assert dense.auto_tree_len_allowance(None, CAP_ROWS) == 256
    # rows=None is the per_row=True shape, i.e. ONE data row per tree
    assert dense.auto_tree_len_allowance(CAP_POP, None) == dense.auto_tree_len_allowance(
        CAP_POP, 1
    )
    # the ceiling is a KNOB (it may exceed the default) ...
    assert dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS, auto_max_tree_len=512) == 512
    # ... while the envelope can be the tighter term
    assert dense.DEFAULT_AUTO_BUDGET_MB * 2 ** 20 // (8192 * 2048 * 4) == 128
    assert dense.auto_tree_len_allowance(8192, 2048) == 128
    # a tight budget is the tighter term, and the allowance never reaches 0
    assert (
        dense.auto_tree_len_allowance(
            CAP_POP, CAP_ROWS, auto_budget_mb=CAP_TIGHT_BUDGET_MB
        )
        == 1
    )
    assert dense.auto_tree_len_allowance(10 ** 6, 10 ** 6, auto_budget_mb=1) == 1
    # both knobs are also read from the environment, per call
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "64")
    assert dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS) == 64
    monkeypatch.setenv(dense.AUTO_BUDGET_MB_ENV, str(CAP_TIGHT_BUDGET_MB))
    assert dense.auto_tree_len_allowance(CAP_POP, CAP_ROWS) == 1


def test_resolve_auto_max_tree_len_default_env_and_explicit(monkeypatch):
    """The ceiling resolves like every other knob: argument, env, documented default."""
    assert dense.resolve_auto_max_tree_len() == 256
    assert dense.resolve_auto_max_tree_len(64) == 64
    assert dense.resolve_auto_max_tree_len(512) == 512
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "128")
    assert dense.resolve_auto_max_tree_len() == 128
    assert dense.resolve_auto_max_tree_len(64) == 64  # explicit argument wins
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, " 128 ")
    assert dense.resolve_auto_max_tree_len() == 128
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "")
    assert dense.resolve_auto_max_tree_len() == 256
    monkeypatch.delenv(dense.AUTO_MAX_TREE_LEN_ENV)
    assert dense.resolve_auto_max_tree_len() == 256


@pytest.mark.parametrize("bad", [0, -1, -64])
def test_resolve_auto_max_tree_len_rejects_non_positive_values(bad, monkeypatch):
    with pytest.raises(ValueError, match=dense.AUTO_MAX_TREE_LEN_ENV):
        dense.resolve_auto_max_tree_len(bad)
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, str(bad))
    with pytest.raises(ValueError, match=dense.AUTO_MAX_TREE_LEN_ENV):
        dense.resolve_auto_max_tree_len()


@pytest.mark.parametrize("raw", ["junk", "64.5", "auto"])
def test_resolve_auto_max_tree_len_rejects_non_integers(raw, monkeypatch):
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, raw)
    with pytest.raises(ValueError, match=dense.AUTO_MAX_TREE_LEN_ENV):
        dense.resolve_auto_max_tree_len()


def test_trace_cap_pinned_reports_the_argument_and_the_env(monkeypatch):
    """A cap the user asked for is never second-guessed by the raise."""
    assert dense.trace_cap_pinned() is False
    assert dense.trace_cap_pinned(64) is True
    for blank in ("", "   "):
        monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, blank)
        assert dense.trace_cap_pinned() is False
    monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, "128")
    assert dense.trace_cap_pinned() is True
    # a pinned cap IS the raise's answer -- and can never be refused
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST, 256, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == 128
    )
    assert (
        dense.auto_cap_decline(
            CAP_LONGEST, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        is None
    )
    # a pinned cap is never ROUNDED either: the rounding exists to keep an
    # UNPINNED raise loop-friendly, while 96 here is the user's own number and is
    # returned exactly -- from the environment and from the argument
    monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, "96")
    assert dense.trace_cap_pinned() is True
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST, 256, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == 96
    )
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST,
            256,
            eval_mode="auto",
            pop_size=CAP_POP,
            rows=CAP_ROWS,
            max_tree_len=96,
        )
        == 96
    )
    monkeypatch.delenv(dense.MAX_TREE_LEN_ENV)
    assert dense.trace_cap_pinned() is False


@pytest.mark.parametrize("mode", ["dense", "dynamic", "specialized", "batch"])
def test_the_raise_is_auto_only(mode):
    """Every explicit mode (and the alias) keeps the base cap and never declines."""
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST, CAP_ROW_LEN, eval_mode=mode, pop_size=CAP_POP, rows=CAP_ROWS
        )
        == dense.DEFAULT_MAX_TREE_LEN
    )
    assert (
        dense.auto_cap_decline(
            CAP_LONGEST, eval_mode=mode, pop_size=CAP_POP, rows=CAP_ROWS
        )
        is None
    )
    # ... even when the allowance could not cover the population
    assert (
        dense.auto_trace_cap(
            300, 512, eval_mode=mode, pop_size=CAP_POP, rows=CAP_ROWS
        )
        == dense.DEFAULT_MAX_TREE_LEN
    )
    assert (
        dense.auto_cap_decline(300, eval_mode=mode, pop_size=CAP_POP, rows=CAP_ROWS)
        is None
    )


def test_a_population_that_fits_the_cap_takes_no_raise():
    """``longest <= base`` -- and a row stride that already fits -- is the identity.

    Rounding never applies here: the raise (and therefore its power-of-two
    rounding) only ever starts from an over-cap population, so a fitting one keeps
    the base cap verbatim and is never declined.
    """
    assert (
        dense.auto_trace_cap(64, 128, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
        == 64
    )
    assert (
        dense.auto_cap_decline(64, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
        is None
    )
    assert (
        dense.auto_trace_cap(8, 128, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
        == 64
    )
    assert (
        dense.auto_cap_decline(8, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS)
        is None
    )
    # a row stride inside the cap cannot contain an over-cap tree: defensive, since
    # the realized longest tree of a valid population never exceeds its stride
    assert (
        dense.auto_trace_cap(
            CAP_LONGEST, 64, eval_mode="auto", pop_size=CAP_POP, rows=CAP_ROWS
        )
        == 64
    )


def test_auto_cap_decline_names_the_binding_term():
    """The reason names BOTH terms of the allowance and says which one bound it."""
    ceiling_reason = dense.auto_cap_decline(
        CAP_LONGEST,
        eval_mode="auto",
        pop_size=CAP_POP,
        rows=CAP_ROWS,
        auto_max_tree_len=64,
    )
    assert ceiling_reason is not None
    assert f"the longest tree is {CAP_LONGEST} nodes" in ceiling_reason
    assert f"pop_size={CAP_POP} x {CAP_ROWS} row(s)" in ceiling_reason
    assert "above the 64 nodes the automatic trace-cap raise allows" in ceiling_reason
    assert f"{dense.AUTO_MAX_TREE_LEN_ENV}=64 is the tighter term" in ceiling_reason
    assert "would allow 1024 nodes" in ceiling_reason  # the envelope alone
    assert ceiling_reason.count("is the tighter term") == 1
    budget_reason = dense.auto_cap_decline(
        CAP_LONGEST,
        eval_mode="auto",
        pop_size=CAP_POP,
        rows=CAP_ROWS,
        auto_budget_mb=CAP_TIGHT_BUDGET_MB,
    )
    assert "above the 1 nodes the automatic trace-cap raise allows" in budget_reason
    assert (
        f"{dense.AUTO_BUDGET_MB_ENV}={CAP_TIGHT_BUDGET_MB} MiB is the tighter term"
        in budget_reason
    )
    assert "would allow 256 nodes" in budget_reason  # the ceiling alone
    assert budget_reason.count("is the tighter term") == 1
    # an unknown population shape can only apply the ceiling, and says so
    shape_reason = dense.auto_cap_decline(300, eval_mode="auto", rows=CAP_ROWS)
    assert "a population of unknown shape" in shape_reason
    assert "the population shape is unknown, so the envelope cannot be applied" in (
        shape_reason
    )
    # every refusal carries the same remedies
    for reason in (ceiling_reason, budget_reason, shape_reason):
        assert f"Raise {dense.AUTO_MAX_TREE_LEN_ENV} or {dense.AUTO_BUDGET_MB_ENV}" in reason
        assert f"set an explicit {dense.MAX_TREE_LEN_ENV}" in reason
        assert "EVOGP_EVAL_MODE=dense" in reason


@pytest.mark.parametrize("entry", ["value", "loss"])
def test_the_raise_reaches_the_plan_of_both_entry_points(entry, monkeypatch):
    """``auto`` SERVES the population: the plan is built at the ROUNDED raised cap."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    recorded = capture_plan(monkeypatch)
    with pytest.raises(PlanCaptured) as excinfo:
        call_cap_entry(entry, backend="numpy")
    assert excinfo.value.args[0] == CAP_RAISED  # the cap the plan was handed
    assert recorded["max_tree_len"] == CAP_RAISED
    assert recorded["max_tree_len"] == dense.auto_trace_cap(
        CAP_LONGEST,
        CAP_ROW_LEN,
        eval_mode="auto",
        pop_size=CAP_POP,
        rows=None if entry == "value" else CAP_ROWS,
    )
    if entry == "value":
        assert recorded["per_row"] is True and recorded["variant"] == dense.VALUE_VARIANT
        assert recorded["n_rows"] == CAP_POP  # one data row per tree
    else:
        assert recorded["per_row"] is False and recorded["variant"] == dense.LOSS_VARIANT
        assert recorded["n_rows"] == CAP_ROWS


def test_a_population_that_fits_the_cap_pays_no_host_read(monkeypatch):
    """The raise's population read is behind the row-stride check (hot path)."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    reads = []

    def explode(sizes):  # pragma: no cover - it must not run
        reads.append(1)
        raise AssertionError(
            "the auto raise read the population although the row stride fits the cap"
        )

    monkeypatch.setattr(dense, "_population_longest", explode)
    recorded = capture_plan(monkeypatch)
    values, types, sizes = chain_rows(CHAIN_LEN, ROW_LEN, POP_SIZE)  # stride 16 <= cap 64
    with pytest.raises(PlanCaptured):
        dense.evaluate_dense(
            values, types, sizes, INPUT_LEN, 1, X, per_row=True, backend="numpy"
        )
    assert reads == []
    assert recorded["max_tree_len"] is None  # the caller's value, untouched


def test_a_long_row_stride_reads_the_longest_but_raises_nothing(monkeypatch):
    """A stride above the cap whose every tree fits: one read, no raise."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    reads = []
    original = dense._population_longest

    def spy(sizes):
        reads.append(1)
        return original(sizes)

    monkeypatch.setattr(dense, "_population_longest", spy)
    recorded = capture_plan(monkeypatch)
    values, types, sizes = chain_rows(CHAIN_LEN, 128, POP_SIZE)  # stride 128, longest 8
    with pytest.raises(PlanCaptured):
        dense.evaluate_dense(
            values, types, sizes, INPUT_LEN, 1, X, per_row=True, backend="numpy"
        )
    assert reads == [1]
    assert recorded["max_tree_len"] is None


@pytest.mark.parametrize("entry", ["value", "loss"])
def test_a_refused_raise_declines_flagged_before_the_plan(entry, monkeypatch):
    """The ceiling refusing the raise is a DELIBERATE, device-fallback decline."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "64")
    monkeypatch.setattr(dense, "_TRACE_CAP_WARNED", False)
    forbid_plan(monkeypatch)
    # ``backend="iree"`` is only a NAME: the refusal is decided before the backend
    # is resolved, so nothing compiles and no device is touched.
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        call_cap_entry(entry, backend="iree")
    exc = excinfo.value
    assert exc.device_fallback is True
    assert f"{dense.AUTO_MAX_TREE_LEN_ENV}=64 is the tighter term" in exc.reason
    assert "automatic trace-cap raise" in exc.reason
    # ... and it is not the once-per-process over-cap surprise (that warning belongs
    # to the pinned paths, which reach ``prepare_dense``)
    assert dense._TRACE_CAP_WARNED is False


def test_a_rounded_refusal_declines_flagged_before_the_plan(monkeypatch):
    """The ROUNDED request refusing at a wired entry point: the same flagged decline.

    A 100-node population at the reference shape asks for a 128-node trace cap (the
    next power of two), so a ``EVOGP_DENSE_AUTO_MAX_TREE_LEN=110`` ceiling refuses it
    even though 110 is above the realized length -- the allowance is re-checked at
    the ROUNDED value, not at ``longest``.  The refusal is decided from the shape
    BEFORE the plan (``forbid_plan``: neither the plan seam nor the population's host
    decode may run) and leaves the once-per-process over-cap warning free, so a wired
    call site continues on the route's DEVICE backend instead of per-tree numpy.
    ``backend="iree"`` is only a NAME here: nothing compiles.
    """
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "110")
    monkeypatch.setattr(dense, "_TRACE_CAP_WARNED", False)
    forbid_plan(monkeypatch)
    values, types, sizes = chain_rows(100, CAP_ROW_LEN, CAP_POP)
    assert int(sizes[:, 0].max()) == 100  # the realized longest tree
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        dense.evaluate_dense(
            values,
            types,
            sizes,
            INPUT_LEN,
            1,
            CAP_PER_ROW_X,
            per_row=True,
            backend="iree",
        )
    exc = excinfo.value
    assert exc.device_fallback is True
    assert f"{dense.AUTO_MAX_TREE_LEN_ENV}=110 is the tighter term" in exc.reason
    assert "128" in exc.reason  # the rounded request, not the 100 realized nodes
    # the refusal happens before ``prepare_dense``, so the pinned-path warning that
    # reports an over-cap population is still unspent
    assert dense._TRACE_CAP_WARNED is False


def test_a_budget_refused_raise_declines_flagged(monkeypatch):
    """The envelope refusing the raise declines the same way (``per_row=True``).
    The loss entry point cannot see this refusal at the reference shape: both rules
    are solved from the same byte budget, and the ROWS rule (which is checked
    first) sizes its envelope with the full 128-wide row stride, so it fires
    before the cap axis gets a chance.  A per-row call has no data-row axis, which
    is where the cap envelope is the only bound.
    """
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    monkeypatch.setenv(dense.AUTO_BUDGET_MB_ENV, "1")
    assert dense.auto_tree_len_allowance(CAP_POP * 2, None, auto_budget_mb=1) == 64
    forbid_plan(monkeypatch)
    values, types, sizes = chain_rows(CAP_LONGEST, CAP_ROW_LEN, CAP_POP * 2)
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        dense.evaluate_dense(
            values,
            types,
            sizes,
            INPUT_LEN,
            1,
            np.zeros((CAP_POP * 2, INPUT_LEN), dtype=np.float32),
            per_row=True,
            backend="numpy",
        )
    assert excinfo.value.device_fallback is True
    assert (
        f"{dense.AUTO_BUDGET_MB_ENV}=1 MiB is the tighter term" in excinfo.value.reason
    )


@pytest.mark.parametrize("pin", ["argument", "env", "mode"])
def test_a_pinned_cap_keeps_the_unflagged_over_cap_decline(pin, monkeypatch):
    """A pinned cap (or an explicit mode) is never raised: the OLD decline stands."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    kwargs = {}
    if pin == "argument":
        kwargs["max_tree_len"] = dense.DEFAULT_MAX_TREE_LEN
        assert dense.trace_cap_pinned(kwargs["max_tree_len"]) is True
    elif pin == "env":
        monkeypatch.setenv(dense.MAX_TREE_LEN_ENV, str(dense.DEFAULT_MAX_TREE_LEN))
        assert dense.trace_cap_pinned() is True
    else:
        monkeypatch.setenv(dense.EVAL_MODE_ENV, "dense")
    values, types, sizes = chain_rows(CAP_LONGEST, CAP_ROW_LEN, 2)
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        dense.sr_fitness_dense(
            values, types, sizes, INPUT_LEN, 1, X, LABELS, backend="numpy", **kwargs
        )
    exc = excinfo.value
    assert exc.device_fallback is False  # the numpy fallback, as before the rule
    assert f"longest tree is {CAP_LONGEST} nodes" in exc.reason
    assert f"cap is {dense.DEFAULT_MAX_TREE_LEN}" in exc.reason
    assert dense.MAX_TREE_LEN_ENV in exc.reason


def test_the_longest_read_is_one_memoized_host_read(monkeypatch):
    """The raise costs ONE population read, and a repeat decline repays nothing."""
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    monkeypatch.setenv(dense.AUTO_MAX_TREE_LEN_ENV, "64")
    evoxc_engine.clear_cache()  # start from empty front tables
    values, types, sizes = (torch.as_tensor(row) for row in cap_population())
    inputs = torch.as_tensor(CAP_X)
    labels = torch.as_tensor(CAP_LABELS)
    with pytest.raises(dense.DenseUnsupported) as excinfo:
        dense.sr_fitness_dense(
            values, types, sizes, INPUT_LEN, 1, inputs, labels, backend="numpy"
        )
    assert excinfo.value.device_fallback is True
    token = dense.dense_memo.token(sizes)
    assert token is not None
    assert dense.DENSE_INPUT_MEMO.lookup(("longest", token)) == CAP_LONGEST
    # ... and a repeat call is served by the front tables: the real host conversion
    # of ``inputs`` is never repeated (the reduction's memo is the entry above).
    def explode(*args, **kwargs):
        raise AssertionError("the repeat call re-read the host")

    monkeypatch.setattr(dense, "_as_numpy", explode)
    with pytest.raises(dense.DenseUnsupported) as repeat:
        dense.sr_fitness_dense(
            values, types, sizes, INPUT_LEN, 1, inputs, labels, backend="numpy"
        )
    assert repeat.value.device_fallback is True
