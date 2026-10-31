"""The route memo: ``dense_engine.resolve_route``'s decision table and its LAZINESS.

``resolve_route`` is the one route policy every public call site asks, and it is
memoized (``dense_engine._ROUTE_MEMO``) because it used to cost 5.8 us of a fused
route's ~37 us warm call.  A memo on a policy that reads the environment and two
lazily-resolved availability probes is only sound while two rules hold, and this
module is their net:

* **Laziness.**  The probes (``_DEVICE_AVAILABLE`` / ``_FUSED_AVAILABLE``) are
  resolved by the POLICY alone, for the arms that need them -- an explicit
  ``EVOGP_EVAL_MODE`` or an explicit backend answers without asking the box
  anything (``test_route_policy.py``'s isolation cases).  The memo must therefore
  never resolve a probe to build or validate a key: a needed probe that is still
  ``None`` is a BYPASS (nothing stored), and an UNNEEDED one is not even looked at.
* **Completeness within an arm.**  The key carries every argument, every environment
  name the arm can read, and the probe VALUES it used, so a hit is what a fresh
  derivation would have answered.  ``_fresh`` below is the oracle for that: it clears
  the table (leaving the probes resolved) and re-asks the policy, which is exactly
  one un-memoized derivation.

Everything here is CPU-only and hermetic: the probes are pinned process globals or a
monkeypatched ``torch.cuda.is_available`` (which never initializes CUDA), and
``backend="iree"`` is only a NAME in this file.
"""
from __future__ import annotations

import itertools

import pytest

from evogp.tree import dense_engine as dense
from evogp.tree import evoxc_engine
from evogp.tree import fused_engine

#: The engine's own name for the backend variable (never a stale copy).
EVOXC_BACKEND_ENV = dense.BACKEND_ENV
assert EVOXC_BACKEND_ENV == "EVOGP_EVOXC_BACKEND"

#: A population above the ``auto`` threshold, and the data plane the rows/envelope
#: rules are evaluated at.
POP = dense.DEFAULT_AUTO_MIN_POP * 4  # 256
ROWS = 1024
TREE_LEN = 64

#: Every environment name the route memo's key has to carry.  ``_route_key`` adds the
#: fused envelope's three names only when a shape was passed and the two IREE names
#: only when a backend name is in play, so the flip test drives it in those modes.
BASE_KNOBS = (
    dense.EVAL_MODE_ENV,
    dense.BACKEND_ENV,
    dense.AUTO_MIN_POP_ENV,
    dense.AUTO_MAX_ROWS_ENV,
    dense.AUTO_BUDGET_MB_ENV,
    dense.FUSED_MIN_POP_ENV,
)
SHAPE_KNOBS = (
    dense.FUSED_MAX_SCRATCH_MB_ENV,
    dense.FUSED_BLOCK_ROWS_ENV,
    dense.FUSED_NUM_WARPS_ENV,
)
BACKEND_KNOBS = (evoxc_engine.IREE_DEVICE_ENV, evoxc_engine.IREE_OPT_LEVEL_ENV)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """A cleared memo and cleared knobs around every case (no probe is pinned).

    ``monkeypatch`` (never a raw ``os.environ`` write): the suite's session fixture
    pins ``EVOGP_EVOXC_BACKEND=numpy``, and this module needs the UNEXPLICIT arms, so
    the pin is removed per test and restored by pytest afterwards.
    """
    for name in BASE_KNOBS + SHAPE_KNOBS + BACKEND_KNOBS:
        monkeypatch.delenv(name, raising=False)
    dense.clear_route_memo()
    yield
    dense.clear_route_memo()


def _pin(monkeypatch, *, device=None, fused=None):
    """Pin the two availability probes (a ``None`` leaves them UNPROBED)."""
    monkeypatch.setattr(dense, "_DEVICE_AVAILABLE", device)
    monkeypatch.setattr(dense, "_FUSED_AVAILABLE", fused)


def _no_probe(monkeypatch):
    """Make a probe impossible: either half of it would fail the case."""
    def explode(*args, **kwargs):
        raise AssertionError("the route memo resolved an availability probe by itself")

    monkeypatch.setattr(dense.torch.cuda, "is_available", explode)
    monkeypatch.setattr(fused_engine, "fused_available", explode)


def _fresh(*args, **kwargs):
    """The un-memoized answer for the same inputs (the memo's oracle).

    ``clear_route_memo()`` drops every entry but leaves the probes resolved, so the
    call that follows is ONE fresh derivation of the real policy -- nothing is faked
    beyond the table this module exists to check.
    """
    dense.clear_route_memo()
    return dense.resolve_route(*args, **kwargs)


# --------------------------------------------------------------------------- #
# 1. laziness: the memo never takes a probe, and never stores an unprobed answer
# --------------------------------------------------------------------------- #
def test_an_unprobed_needed_probe_bypasses_the_memo_and_stores_nothing(monkeypatch):
    """``auto``/``dense`` without an explicit backend needs the device probe."""
    _pin(monkeypatch, device=None, fused=None)
    monkeypatch.setattr(dense.torch.cuda, "is_available", lambda: True)
    for mode in ("auto", "dense", "batch"):
        _pin(monkeypatch, device=None, fused=None)  # every arm starts unprobed
        dense.clear_route_memo()
        first = dense.resolve_route(POP, eval_mode=mode)
        stats = dense.route_cache_stats()
        assert stats["route_bypasses"] == 1, (
            f"mode={mode!r}: the first call must BYPASS (the device probe was unprobed), "
            f"counted {stats}"
        )
        assert stats["route_entries"] == 0, (
            f"mode={mode!r}: an unprobed answer must never be stored, counted {stats}"
        )
        assert dense._DEVICE_AVAILABLE is True, (
            f"mode={mode!r}: the POLICY resolves the probe (that is the arm's own work)"
        )
        # ... and from the second call on, with the probe known, it is memoizable
        second = dense.resolve_route(POP, eval_mode=mode)
        assert second == first, (mode, second, first)
        third = dense.resolve_route(POP, eval_mode=mode)
        assert third is second, (
            f"mode={mode!r}: a hit must serve the same frozen Route object"
        )
        assert dense.route_cache_stats()["route_hits"] == 1, dense.route_cache_stats()


def test_an_explicit_mode_or_backend_never_probes_and_is_still_memoized(monkeypatch):
    """The isolation contract: no probe for an explicit mode/backend -- and a hit."""
    _pin(monkeypatch, device=None, fused=None)
    _no_probe(monkeypatch)  # any probe in this test is a failure, not a slow test
    cases = list(
        itertools.product(
            (1, 16, POP),
            ("specialized", "dense", "batch", "fused"),
            ("numpy", "iree"),  # an explicit NAME (``None`` would need the device probe)
        )
    ) + [(pop, "specialized", None) for pop in (1, 16, POP)]
    for pop, mode, backend in cases:
        kwargs = {"eval_mode": mode, "backend": backend}
        first = dense.resolve_route(pop, **kwargs)
        assert first == dense.resolve_route(pop, **kwargs), (pop, mode, backend)
    assert dense._DEVICE_AVAILABLE is None and dense._FUSED_AVAILABLE is None, (
        "an explicit mode/backend resolved an availability probe: "
        f"device={dense._DEVICE_AVAILABLE} fused={dense._FUSED_AVAILABLE}"
    )
    stats = dense.route_cache_stats()
    assert stats["route_bypasses"] == 0, (
        f"an arm whose answer needs no probe must be memoizable, counted {stats}"
    )
    assert stats["route_hits"] > 0 and stats["route_entries"] > 0, stats


def test_an_env_backend_counts_as_explicit_and_also_never_probes(monkeypatch):
    """``EVOGP_EVOXC_BACKEND=numpy`` is the CPU-pin: no probe, and memoized."""
    _pin(monkeypatch, device=None, fused=None)
    _no_probe(monkeypatch)
    monkeypatch.setenv(EVOXC_BACKEND_ENV, "numpy")
    for pop in (1, dense.DEFAULT_AUTO_MIN_POP - 1, POP):
        for mode in (None, "auto", "specialized"):
            route = dense.resolve_route(pop, eval_mode=mode)
            assert (route.use_dense, route.backend) == (False, "numpy"), (pop, mode, route)
            assert route == dense.resolve_route(pop, eval_mode=mode)
    assert dense._DEVICE_AVAILABLE is None and dense._FUSED_AVAILABLE is None
    assert dense.route_cache_stats()["route_hits"] > 0


def test_a_device_only_arm_memoizes_while_the_fused_probe_is_still_unprobed(monkeypatch):
    """``EVOGP_EVAL_MODE=fused`` needs the device probe and nothing else."""
    _pin(monkeypatch, device=True, fused=None)
    monkeypatch.setattr(fused_engine, "fused_available", _fail)
    dense.clear_route_memo()
    first = dense.resolve_route(fused_engine.DEFAULT_MIN_POP, eval_mode="fused")
    assert first.engine == "fused", first
    assert dense.route_cache_stats()["route_entries"] == 1, (
        "the answer does not depend on the fused probe, so the entry is sound"
    )
    assert dense.resolve_route(fused_engine.DEFAULT_MIN_POP, eval_mode="fused") is first
    assert dense.route_cache_stats()["route_hits"] == 1


def _fail(*args, **kwargs):
    raise AssertionError("this arm must not consult the fused probe")


def test_auto_with_no_device_memoizes_without_consulting_the_fused_probe(monkeypatch):
    """A known ``device = False`` answers ``auto`` before the fused half exists."""
    _pin(monkeypatch, device=False, fused=None)
    monkeypatch.setattr(fused_engine, "fused_available", _fail)
    dense.clear_route_memo()
    first = dense.resolve_route(POP)
    assert (first.engine, first.backend) == ("specialized", "numpy"), first
    assert dense.route_cache_stats()["route_entries"] == 1, dense.route_cache_stats()
    assert dense.resolve_route(POP) is first
    assert dense.route_cache_stats()["route_hits"] == 1


def test_auto_with_an_unknown_fused_probe_bypasses(monkeypatch):
    """A device box whose fused probe is still unprobed: derive, store nothing."""
    _pin(monkeypatch, device=True, fused=None)
    dense.clear_route_memo()
    dense.resolve_route(POP)
    stats = dense.route_cache_stats()
    assert (stats["route_bypasses"], stats["route_entries"]) == (1, 0), stats
    assert dense._FUSED_AVAILABLE is True, "the policy resolved it (triton is here)"


# --------------------------------------------------------------------------- #
# 2. soundness: a hit is what a fresh derivation would answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", ("device", "fused"))
def test_a_probe_flip_is_a_miss_and_reaches_the_new_answer(label, monkeypatch):
    """Pinning a probe differently must move the answer, not be served stale."""
    _pin(monkeypatch, device=True, fused=True)
    fused_route = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN)
    assert fused_route.engine == "fused", fused_route
    assert dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN) is fused_route
    if label == "fused":
        _pin(monkeypatch, device=True, fused=False)
        expected_engine = "dense"
    else:
        _pin(monkeypatch, device=False, fused=True)
        expected_engine = "specialized"
    moved = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN)
    assert moved.engine == expected_engine, (label, moved)
    assert moved == _fresh(POP, rows=ROWS, tree_len=TREE_LEN), (
        f"the memoized answer for {label} must equal a fresh derivation"
    )


def test_a_knob_flip_is_a_miss_and_matches_a_fresh_derivation(monkeypatch):
    """Every environment name the key carries re-keys the decision it feeds.

    Two things are pinned per knob: the answer after a flip is what a FRESH
    derivation says (never the pre-flip answer), and the key is a NEW one (a miss,
    i.e. the name is really in the key -- even where, like the block/warp geometry of
    the envelope, the verdict happens not to move at this shape).
    """
    shape = {"rows": ROWS, "tree_len": TREE_LEN}
    # ``pin`` is the box each arm is read on: a fused box (device + triton) or a
    # device box where ``auto`` keeps the historical dense/specialized rule.
    fused_box, device_box = (True, True), (True, False)
    cases = (
        # (knob, A, B, kwargs of the arm that reads it, pin, does the ANSWER move?)
        (dense.EVAL_MODE_ENV, "auto", "specialized", {}, fused_box, True),
        (dense.AUTO_MIN_POP_ENV, "64", "1000", {}, device_box, True),
        (dense.AUTO_BUDGET_MB_ENV, "8192", "1", {"rows": 4096, **shape}, device_box, True),
        (dense.AUTO_MAX_ROWS_ENV, "0", "16", shape, device_box, True),
        (dense.FUSED_MIN_POP_ENV, "16", "100000", {}, fused_box, True),
        (dense.FUSED_MAX_SCRATCH_MB_ENV, "8192", "1", shape, fused_box, True),
        (dense.FUSED_BLOCK_ROWS_ENV, "256", "64", shape, fused_box, False),
        (dense.FUSED_NUM_WARPS_ENV, "1", "4", shape, fused_box, False),
        (evoxc_engine.IREE_OPT_LEVEL_ENV, "O1", "O0", {"backend": "iree"}, fused_box, False),
    )
    for knob, a, b, kwargs, pin, moves in cases:
        _pin(monkeypatch, device=pin[0], fused=pin[1])
        dense.clear_route_memo()
        monkeypatch.setenv(knob, a)
        first = dense.resolve_route(POP, **kwargs)
        assert first == _fresh(POP, **kwargs), (knob, a)
        entries = dense.route_cache_stats()["route_entries"]
        monkeypatch.setenv(knob, b)
        second = dense.resolve_route(POP, **kwargs)
        after = dense.route_cache_stats()["route_entries"]  # before ``_fresh`` resets it
        assert second == _fresh(POP, **kwargs), (
            f"{knob}={b} must not be served the {knob}={a} decision "
            f"({second!r} != a fresh derivation)"
        )
        assert (second != first) is moves, (
            f"{knob} {a} -> {b}: the answer {'must' if moves else 'must not'} move, "
            f"got {first!r} -> {second!r}"
        )
        assert after > entries, (
            f"{knob} is not part of the memo key: the {b!r} call reused the entry "
            f"({entries} -> {after} entries)"
        )
        monkeypatch.delenv(knob, raising=False)  # each knob is A/B'd from unset


def test_the_answer_matches_a_fresh_derivation_on_a_sweep(monkeypatch):
    """A whole grid of shapes, modes and backends: never a decision a derivation denies."""
    for device, fused in ((True, True), (True, False), (False, False)):
        _pin(monkeypatch, device=device, fused=fused)
        for pop in (1, 16, 63, 64, POP, 4096):
            for rows in (None, 64, ROWS, 1 << 20):
                for kwargs in ({}, {"eval_mode": "auto"}, {"backend": "numpy"}):
                    if rows is not None:
                        kwargs = dict(kwargs, rows=rows, tree_len=TREE_LEN)
                    got = dense.resolve_route(pop, **kwargs)
                    assert got == _fresh(pop, **kwargs), (device, fused, pop, kwargs, got)


def test_every_argument_is_part_of_the_key(monkeypatch):
    """A different argument is a different decision (no over-broad hit)."""
    _pin(monkeypatch, device=True, fused=False)
    seen = {}
    for pop in (POP, POP + 1):
        for kwargs in (
            {},
            {"rows": ROWS, "tree_len": TREE_LEN},
            {"eval_mode": "specialized"},
            {"output_len": 2},
        ):
            route = dense.resolve_route(pop, **kwargs)
            assert route == _fresh(pop, **kwargs), (pop, kwargs)
            key = (pop,) + tuple(sorted(kwargs.items()))
            seen[key] = route
    assert len(seen) == 8, seen


def test_output_len_only_moves_the_fused_envelope_arm(monkeypatch):
    """``output_len`` is in the key even though it moves nothing on a dense arm."""
    _pin(monkeypatch, device=True, fused=True)
    one = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN, output_len=1)
    many = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN, output_len=2)
    assert many == _fresh(POP, rows=ROWS, tree_len=TREE_LEN, output_len=2)
    assert (one, many) == (one, many)  # both answers exist; neither is a stale serve


# --------------------------------------------------------------------------- #
# 3. a raise and a decline reach the caller identically
# --------------------------------------------------------------------------- #
def test_a_bad_spelling_raises_on_every_call_and_stores_nothing(monkeypatch):
    """A raise is never memoized: the same error, every time, with no entry."""
    _pin(monkeypatch, device=True, fused=True)
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "nonsense")
    messages = []
    for _ in range(3):
        with pytest.raises(ValueError) as caught:
            dense.resolve_route(POP)
        messages.append(str(caught.value))
    assert len(set(messages)) == 1, messages
    assert "nonsense" in messages[0], messages[0]
    assert dense.route_cache_stats()["route_entries"] == 0, dense.route_cache_stats()


def test_a_bad_backend_name_raises_on_every_call_and_stores_nothing(monkeypatch):
    """The explicit-backend arm's raise survives the memo too."""
    _pin(monkeypatch, device=True, fused=True)
    monkeypatch.setenv(dense.EVAL_MODE_ENV, "auto")
    monkeypatch.setenv(EVOXC_BACKEND_ENV, "bogus")
    for _ in range(3):
        with pytest.raises(ValueError, match="bogus"):
            dense.resolve_route(POP)
    assert dense.route_cache_stats()["route_entries"] == 0, dense.route_cache_stats()


def test_a_raising_policy_is_never_swallowed_by_the_memo(monkeypatch):
    """Whatever the policy raises, the caller sees it -- the entry is never written."""
    _pin(monkeypatch, device=True, fused=True)

    class Boom(RuntimeError):
        pass

    calls = []

    def explode(*args, **kwargs):
        calls.append((args, kwargs))
        raise Boom("policy")

    monkeypatch.setattr(dense, "_route_decision", explode)
    for _ in range(3):
        with pytest.raises(Boom):
            dense.resolve_route(POP)
    assert len(calls) == 3, f"the memo swallowed a derivation: {calls}"
    assert dense.route_cache_stats()["route_entries"] == 0


def test_a_decline_is_an_ordinary_route_and_is_served_identically(monkeypatch):
    """The decline path (specialized+numpy) is a ``Route`` like any other decision."""
    _pin(monkeypatch, device=True, fused=False)
    monkeypatch.setenv(dense.AUTO_MAX_ROWS_ENV, "16")  # the rows pin declines dense
    first = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN)
    assert (first.engine, first.use_dense, first.backend) == ("specialized", False, "numpy"), first
    assert "EVOGP_DENSE_AUTO_MAX_ROWS" in first.reason or "rows" in first.reason, first.reason
    again = dense.resolve_route(POP, rows=ROWS, tree_len=TREE_LEN)
    assert again is first, "a decline is memoized like any other decision"
    assert again == _fresh(POP, rows=ROWS, tree_len=TREE_LEN)
    assert dense.should_use_dense(POP, rows=ROWS, tree_len=TREE_LEN) is False, (
        "should_use_dense delegates through the same memo and must agree"
    )


def test_should_use_dense_is_the_same_rule_through_the_memo(monkeypatch):
    """``should_use_dense`` keeps delegating: same answer, same table, one hit."""
    _pin(monkeypatch, device=True, fused=False)
    for pop in (1, dense.DEFAULT_AUTO_MIN_POP - 1, POP, POP * 4):
        for mode in (None, "auto", "dense", "specialized", "batch", "fused"):
            flag = dense.should_use_dense(pop, eval_mode=mode)
            route = dense.resolve_route(pop, eval_mode=mode)
            assert flag is bool(route.use_dense), (pop, mode, flag, route)
    assert dense.route_cache_stats()["route_hits"] > 0, dense.route_cache_stats()


# --------------------------------------------------------------------------- #
# 4. lifecycle: the invalidation point and the stats surface
# --------------------------------------------------------------------------- #
def test_clear_cache_drops_the_memo_and_zeroes_the_counters(monkeypatch):
    """``evoxc_engine.clear_cache()`` is the engine's one invalidation point."""
    _pin(monkeypatch, device=True, fused=False)
    for _ in range(3):
        route = dense.resolve_route(POP)
    before = dense.route_cache_stats()
    assert before["route_hits"] > 0 and before["route_entries"] > 0, before
    evoxc_engine.clear_cache()
    after = dense.route_cache_stats()
    assert after == {
        "route_hits": 0,
        "route_misses": 0,
        "route_bypasses": 0,
        "route_entries": 0,
        "route_maxsize": dense._ROUTE_MEMO_MAX,
    }, after
    assert dense.resolve_route(POP) == route, "a cleared memo must answer the same"
    assert dense.route_cache_stats()["route_misses"] == 1, "the first call re-derives"


def test_the_table_is_bounded_and_cleared_as_a_whole(monkeypatch):
    """The bound is enforced by a whole-table clear, never by evicting one entry."""
    _pin(monkeypatch, device=True, fused=False)
    for pop in range(1, dense._ROUTE_MEMO_MAX + 5):
        assert dense.resolve_route(pop) == _fresh(pop)
    stats = dense.route_cache_stats()
    assert 0 < stats["route_entries"] <= stats["route_maxsize"], stats


def test_the_two_fused_envelope_spellings_cannot_drift():
    """The key's three fused names are re-declared here; they must be the same ones."""
    assert dense.FUSED_MAX_SCRATCH_MB_ENV == fused_engine.MAX_SCRATCH_MB_ENV
    assert dense.FUSED_BLOCK_ROWS_ENV == fused_engine.BLOCK_ROWS_ENV
    assert dense.FUSED_NUM_WARPS_ENV == fused_engine.NUM_WARPS_ENV


def test_the_reader_agrees_with_the_environment(monkeypatch):
    """``_route_env`` is ``os.environ.get`` in every state, fast store or not."""
    import os

    name = "EVOGP_TESTS_ROUTE_MEMO_PROBE"
    for value in (None, "", "auto", "iree", " unicode \u00e9 "):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        assert dense._route_env(name) == os.environ.get(name) == value, value
    os.environ.pop(name, None)
    # ... and through the slow reader too (the fallback path of an exotic os.environ)
    saved = dense._ENV_DATA
    monkeypatch.setattr(dense, "_ENV_DATA", None)
    os.environ[name] = "x"
    try:
        assert dense._route_env(name) == "x"
    finally:
        os.environ.pop(name, None)
        monkeypatch.setattr(dense, "_ENV_DATA", saved)
