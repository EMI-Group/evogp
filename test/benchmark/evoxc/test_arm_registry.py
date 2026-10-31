"""Torch-free checks of the arm registry, the batch knob table and the route report.

This directory is normally not collected by pytest, but the project gate runs
``pytest test src/evogp/tests -q``, which walks into it -- so this module is the ONE
automated check that the harness can NAME what the engine grew, and it stays
import-light: ``common`` / ``arms`` / ``campaign`` / ``check`` / ``band_report`` /
``artifact`` / ``worker`` only (every one of them torch-free at import).  The last group
(``test_dry_run_*``) runs ``campaign.main --dry-run``, which resolves the batch policy
from the tree on ``PYTHONPATH``; that is the harness's own dry run, i.e. what a later
agent inspects before paying a compile.

What is pinned here:

* the new ``EVOGP_BATCH_DISPATCH_FORMATION`` knob is in the single-sourced
  ``common.BATCH_KNOBS`` table exactly once, and it therefore joins
  ``BATCH_ENV_VARS`` / ``ROUTE_ENV_VARS`` (so it can never leak between arms);
* the disk tier's ``EVOGP_EVOXC_ARTIFACT_DIR`` is a route knob too
  (``ARTIFACT_ENV_VARS``), and the dense route's new ``EVOGP_DENSE_AUTO_BUDGET_MB``
  is part of ``DENSE_ENV_VARS``;
* every pre-existing arm id is unchanged (byte-identical, in registry order) and the
  three new arms -- ``evoxc-batch-iree-aggressive`` / ``evoxc-batch-iree-artifact`` /
  ``evoxc-batch-iree-default`` -- resolve to the env they claim, the disk tier through
  the harness's own hook;
* ``evoxc-batch-iree-default`` is the PRE-CHANGE control of the merged compile: the
  dispatch-formation request pinned to ``default`` (no aggressive fusion) while
  ``evoxc-batch-iree`` stays the shipped default (``auto``, i.e. no knob at all);
* no arm leaks a route knob: every ``common.ROUTE_ENV_VARS`` entry is cleared before an
  arm's own spec is written, for every arm including the torch baselines;
* the fused (Triton) route's five ``EVOGP_FUSED_*`` knobs are route knobs too
  (``FUSED_ENV_VARS``), and its two arms -- ``evoxc-fused-iree`` / ``evoxc-fused`` --
  set the route selector and nothing else, appended after the last batch arm;
* ``worker.fused_served`` is the fused route's existence proof: a launch counter, never
  ``compile_misses_first_call`` (the route compiles no evoxc graph, so that is 0 by
  construction), read null-never-zero out of ``fused_window_block``;
* ``common.route_decision`` passes the population SHAPE (``pop_size`` / ``rows`` /
  ``tree_len``) into the policy, so a record states the memory-envelope decision and
  not a shape-blind one -- and it asks the ONE resolver (``resolve_route``) wherever the
  POLICY owns the engine, so a served ``fused`` call and a shipped-default (``auto``) call
  both render as the engine that served them instead of as a fallback;
* every engine handle is resolved by ``common.installed_module`` (the ``sys.modules`` entry
  first, an import of the full dotted name second, NEVER the parent package's cached
  attribute), and ``fake_dense_engine`` clears the route environment -- because the project
  gate runs ``src/evogp/tests`` first, whose ``conftest.py`` leaves
  ``EVOGP_EVOXC_BACKEND=numpy`` in the process and whose import of ``evogp.tree`` caches the
  real engine on that package;
* ``worker.may_use_fused`` is the harness's "may reach the fused route" predicate (an
  explicit ``fused`` arm OR ``auto``, either spelling), so the launch proof is opened for
  the shipped default too and a silent ``specialized`` fall-back cannot hide in it;
* ``check.is_fused_record`` gates a record by the engine its routes RESOLVED, so an
  ``evoxc-auto`` record that resolved fused is checked like a fused arm (and one that did
  not resolve there is not);
* ``campaign.CELL_EXCLUSIONS`` is a stated, unconditional drop -- ``evoxc-dense-iree`` on
  the ``d7`` cell, whose explicit dense mode OOM-kills the job unit -- recorded in
  ``campaign.json``'s ``scope_dropped[*].reason`` / ``cell_exclusions`` and printed by
  ``--dry-run``;
* ``band_report`` surfaces the merged compile's ``dispatch_formation`` next to the
  request, and reads ``None`` (never a fabricated ``"default"``) when a description
  predates the knob;
* ``artifact.window_block`` is the strict null-never-zero block of the disk tier;
* ``campaign --dry-run`` states each new knob exactly once.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # pytest's prepend mode normally does this; be explicit
    sys.path.insert(0, _HERE)

import artifact  # noqa: E402  (sibling harness modules, all torch-free at import)
import arms  # noqa: E402
import band_report  # noqa: E402
import campaign  # noqa: E402
import check  # noqa: E402  (the gate's predicates; stdlib-only, reads the same registry)
import common  # noqa: E402
import worker  # noqa: E402  (its fused block + launch proof; torch-free at import)

#: The arm ids that existed BEFORE the dispatch-formation / disk-tier arms were added,
#: in registry order.  Pinned so a rename (which would invalidate every committed
#: record's ``arm`` key and every report's arm column) fails here instead of silently.
EXISTING_ARMS = (
    "torch-per-tree",
    "torch-batched",
    "evoxc-auto",
    "evoxc-auto-iree",
    "evoxc-specialized-numpy",
    "evoxc-specialized-iree",
    "evoxc-dense-iree",
    "evoxc-batch-iree",
    "evoxc-batch-iree-flat",
    "evoxc-dense-iree-raised",
)

DISPATCH_ENV = "EVOGP_BATCH_DISPATCH_FORMATION"
ARTIFACT_DIR_VAR = "EVOGP_EVOXC_ARTIFACT_DIR"

#: Every variable these tests may write into ``os.environ`` (restored after each test so
#: ``apply_arm_env`` can be called freely -- it is a writer, not a context manager).
MANAGED_VARS = tuple(common.ROUTE_ENV_VARS) + (
    "EVOGP_EVAL_MODE",
    "EVOGP_EVOXC_BACKEND",
    common.ARTIFACT_DIR_ENV,
)

#: The fake policy's fused crossover (the real one is ``EVOGP_FUSED_MIN_POP``, 16):
#: ``auto`` resolves to the fused route from this population up and to specialized below it.
FUSED_MIN_POP = 16


@pytest.fixture(autouse=True)
def restore_environment():
    """Snapshot / restore every variable ``apply_arm_env`` and ``common`` may touch."""
    saved = {name: os.environ.get(name) for name in MANAGED_VARS}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# the knob tables (single-sourced in common.py)
# ---------------------------------------------------------------------------
def test_dispatch_formation_knob_is_registered_exactly_once():
    """One entry, with the documented env name, default and resolver."""
    entries = [entry for entry in common.BATCH_KNOBS if entry[0] == "dispatch_formation"]
    assert len(entries) == 1
    spec_key, env_name, default, candidates = entries[0]
    assert spec_key == "dispatch_formation"
    assert env_name == DISPATCH_ENV
    assert default == "auto"
    assert candidates == ("resolve_dispatch_formation",)
    # ... and the env-var lists are derived from it, so the knob joins the cleared set
    assert DISPATCH_ENV in common.BATCH_ENV_VARS
    assert DISPATCH_ENV in common.ROUTE_ENV_VARS
    assert [entry[1] for entry in common.BATCH_KNOBS].count(DISPATCH_ENV) == 1


def test_artifact_dir_is_a_route_knob():
    """The disk tier's env var is named once and cleared like every other route knob."""
    assert common.ARTIFACT_ENV_VARS == (ARTIFACT_DIR_VAR,)
    assert ARTIFACT_DIR_VAR in common.ROUTE_ENV_VARS
    assert common.ROUTE_ENV_VARS.count(ARTIFACT_DIR_VAR) == 1
    # it is not a batch knob: the dense route reads the same variable
    assert ARTIFACT_DIR_VAR not in common.BATCH_ENV_VARS


def test_dense_budget_knob_is_registered():
    """``auto``'s rows rule is a memory envelope now, and its knob is a dense knob."""
    assert "EVOGP_DENSE_AUTO_BUDGET_MB" in common.DENSE_ENV_VARS
    assert "EVOGP_DENSE_AUTO_MAX_ROWS" in common.DENSE_ENV_VARS
    for name in ("EVOGP_DENSE_AUTO_BUDGET_MB", "EVOGP_DENSE_AUTO_MAX_ROWS"):
        assert name in common.ROUTE_ENV_VARS
    assert not set(common.DENSE_ENV_VARS) & set(common.BATCH_ENV_VARS)


def test_fused_knobs_are_route_knobs():
    """The five ``EVOGP_FUSED_*`` knobs are cleared per arm like every other route knob."""
    assert common.FUSED_ENV_VARS == (
        "EVOGP_FUSED_MIN_POP", "EVOGP_FUSED_BLOCK_ROWS", "EVOGP_FUSED_NUM_WARPS",
        "EVOGP_FUSED_MAX_SCRATCH_MB", "EVOGP_FUSED_HOST_MEMO",
    )
    for name in common.FUSED_ENV_VARS:
        assert name in common.ROUTE_ENV_VARS, name
        assert common.ROUTE_ENV_VARS.count(name) == 1, name
        assert name not in common.BATCH_ENV_VARS, name
        assert name not in common.DENSE_ENV_VARS, name
    # the cleared set is the sum of the four tables, in that order
    assert common.ROUTE_ENV_VARS == (
        common.BATCH_ENV_VARS + common.DENSE_ENV_VARS
        + common.FUSED_ENV_VARS + common.ARTIFACT_ENV_VARS
    )


# ---------------------------------------------------------------------------
# the two new arms
# ---------------------------------------------------------------------------
def test_existing_arm_ids_are_unchanged():
    torch_count = len(common.TORCH_ARMS)
    evoxc_ids = EXISTING_ARMS[torch_count:]
    assert tuple(common.TORCH_ARMS) == EXISTING_ARMS[:torch_count]
    assert tuple(common.EVOXC_ARMS)[: len(evoxc_ids)] == evoxc_ids
    assert tuple(common.ALL_ARMS[: len(EXISTING_ARMS)]) == EXISTING_ARMS


def test_new_arm_ids_are_appended_last_and_are_batch_arms():
    new = (
        "evoxc-batch-iree-aggressive",
        "evoxc-batch-iree-artifact",
        "evoxc-batch-iree-default",
        "evoxc-batch-iree-artifact-p1",
        "evoxc-batch-iree-artifact-p2",
        "evoxc-batch-iree-artifact-p8",
    )
    # the fused (Triton) route is the FOURTH engine and is appended AFTER the batch arms,
    # so every arm id above keeps its registry index (a committed record's ``arm`` key)
    fused = ("evoxc-fused-iree", "evoxc-fused")
    tail = new + fused
    assert tuple(common.EVOXC_ARMS)[-len(tail):] == tail
    assert tuple(common.ALL_ARMS)[-len(tail):] == tail
    for arm in new:
        assert common.is_batch_arm(arm), arm
        assert common.EVOXC_ARMS[arm]["backend"] == "iree"
    # ``--arms batch`` picks them up from the ONE predicate (no extra rule)
    batch_arms = [arm for arm in common.EVOXC_ARMS if common.is_batch_arm(arm)]
    assert list(batch_arms) == ["evoxc-batch-iree", "evoxc-batch-iree-flat", *new]
    for arm in fused:
        assert common.EVOXC_ARMS[arm]["eval_mode"] == "fused", arm
        assert common.is_batch_arm(arm) is False, arm
    assert common.EVOXC_ARMS["evoxc-fused-iree"]["backend"] == "iree"
    assert common.EVOXC_ARMS["evoxc-fused"]["backend"] is None


def test_default_formation_arm_is_the_pre_change_control():
    """``evoxc-batch-iree-default`` pins the PRE-``auto`` spelling, nothing else."""
    arm = "evoxc-batch-iree-default"
    assert common.EVOXC_ARMS[arm] == {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {DISPATCH_ENV: "default"},
    }
    applied = arms.apply_arm_env(arm)
    # exactly the two route selectors plus the one knob: no band, no disk tier
    assert applied["knobs"] == {DISPATCH_ENV: "default"}
    assert os.environ["EVOGP_EVAL_MODE"] == "batch"
    assert os.environ["EVOGP_EVOXC_BACKEND"] == "iree"
    assert os.environ[DISPATCH_ENV] == "default"
    applied_vars = {
        name for name in MANAGED_VARS if name in os.environ
    }
    assert applied_vars == {
        "EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", DISPATCH_ENV,
    }, applied_vars
    assert common.arm_knob_env(arm) == {DISPATCH_ENV: "default"}
    # the shipped default arm keeps its own spec: ``auto``, i.e. NO knob at all
    assert common.EVOXC_ARMS["evoxc-batch-iree"] == {
        "eval_mode": "batch", "backend": "iree", "family": "evoxc",
    }
    arms.apply_arm_env("evoxc-batch-iree")
    assert DISPATCH_ENV not in os.environ


def test_aggressive_arm_pins_the_request_formed_spelling():
    applied = arms.apply_arm_env("evoxc-batch-iree-aggressive")
    assert applied["knobs"] == {DISPATCH_ENV: "aggressive"}
    assert os.environ[DISPATCH_ENV] == "aggressive"
    assert os.environ["EVOGP_EVAL_MODE"] == "batch"
    assert os.environ["EVOGP_EVOXC_BACKEND"] == "iree"
    # the arm does not touch the disk tier
    assert ARTIFACT_DIR_VAR not in os.environ
    assert common.arm_knob_env("evoxc-batch-iree-aggressive") == {DISPATCH_ENV: "aggressive"}


def test_artifact_arm_points_at_the_harness_hook(tmp_path, monkeypatch):
    hook = tmp_path / "shared-cache"
    monkeypatch.setenv(common.ARTIFACT_DIR_ENV, str(hook))
    applied = arms.apply_arm_env("evoxc-batch-iree-artifact")
    expected = os.path.abspath(str(hook))
    assert applied["knobs"] == {ARTIFACT_DIR_VAR: expected}
    assert os.environ[ARTIFACT_DIR_VAR] == expected
    # the placeholder is the harness's, not a cell property: no cell was passed
    assert common.arm_knob_env("evoxc-batch-iree-artifact") == {ARTIFACT_DIR_VAR: expected}


def test_pool_pinned_artifact_arms_set_the_pool_and_the_disk_tier(tmp_path, monkeypatch):
    """A pool width is only reachable through an ARM: every route knob is cleared first."""
    hook = tmp_path / "shared-cache"
    expected = os.path.abspath(str(hook))
    for pool in ("1", "2", "8"):
        arm = f"evoxc-batch-iree-artifact-p{pool}"
        assert common.is_batch_arm(arm), arm
        assert common.EVOXC_ARMS[arm]["knobs"] == {
            "EVOGP_EVOXC_ARTIFACT_DIR": "{artifact_dir}",
            "EVOGP_BATCH_COMPILE_WORKERS": pool,
        }
        monkeypatch.setenv(common.ARTIFACT_DIR_ENV, str(hook))
        applied = arms.apply_arm_env(arm)
        assert applied["knobs"] == {
            ARTIFACT_DIR_VAR: expected, "EVOGP_BATCH_COMPILE_WORKERS": pool,
        }, arm
        assert os.environ["EVOGP_BATCH_COMPILE_WORKERS"] == pool
        assert os.environ[ARTIFACT_DIR_VAR] == expected
    # the shipped artifact arm keeps the engine's own pool default: no pin at all
    assert "EVOGP_BATCH_COMPILE_WORKERS" not in (
        common.EVOXC_ARMS["evoxc-batch-iree-artifact"]["knobs"]
    )


def test_fused_arms_set_the_route_and_no_knobs():
    """The two fused arms are route selectors only: one explicit backend, one policy."""
    iree = arms.apply_arm_env("evoxc-fused-iree")
    assert iree == {"eval_mode": "fused", "backend": "iree", "knobs": {}}
    assert os.environ["EVOGP_EVAL_MODE"] == "fused"
    assert os.environ["EVOGP_EVOXC_BACKEND"] == "iree"
    assert common.arm_knob_env("evoxc-fused-iree") == {}
    # the bare spelling leaves the backend to the route's own policy
    bare = arms.apply_arm_env("evoxc-fused")
    assert bare == {"eval_mode": "fused", "backend": None, "knobs": {}}
    assert os.environ["EVOGP_EVAL_MODE"] == "fused"
    assert "EVOGP_EVOXC_BACKEND" not in os.environ
    for name in common.FUSED_ENV_VARS:
        assert name not in os.environ, name
    for arm in ("evoxc-fused-iree", "evoxc-fused"):
        assert common.EVOXC_ARMS[arm]["family"] == "evoxc", arm
        assert common.is_batch_arm(arm) is False, arm


def test_fused_launch_proof_is_the_only_existence_proof():
    """``fused_kernel_launches`` proves a served call; ``compile_misses`` cannot."""
    assert len(worker.FUSED_STATS_KEYS) == 15
    assert len(worker.FUSED_GAUGE_KEYS) == 6
    assert set(worker.FUSED_GAUGE_KEYS) < set(worker.FUSED_STATS_KEYS)
    assert worker.FUSED_STATS_FN_PATH == "evogp.tree.fused_engine.fused_cache_stats"
    served = {"fused_calls": 8, "fused_declines": 0, "fused_kernel_launches": 8}
    assert worker.fused_served(served, 8) is True
    # a call count the counters cannot account for is NOT a served call
    assert worker.fused_served(served, 9) is False
    # the silent fallback: one of the eight attempts declined, so nothing launched
    declined = {"fused_calls": 8, "fused_declines": 1, "fused_kernel_launches": 7}
    assert worker.fused_served(declined, 8) is False
    assert worker.fused_served(declined, 7) is True
    # unknown counters / no call count read ``None``, never ``False``
    assert worker.fused_served({"fused_calls": None, "fused_declines": 0,
                               "fused_kernel_launches": 0}, 8) is None
    assert worker.fused_served(served, None) is None


def test_fused_window_block_is_a_delta_with_after_state_gauges():
    before = {key: 0 for key in worker.FUSED_STATS_KEYS}
    before["fused_scratch_bytes"] = 1024
    after = dict(before)
    after.update({
        "fused_calls": 8, "fused_kernel_launches": 8, "fused_xt_hits": 3,
        "fused_xt_misses": 5, "fused_kernel_compiles": 2,
        "fused_scratch_bytes": 4096,
        "fused_block_rows": 32, "fused_num_warps": 4,
    })
    block = worker.fused_window_block(before, after)
    assert block["available"] is True
    assert block["stats_fn"] == worker.FUSED_STATS_FN_PATH
    assert block["fused_calls"] == 8          # running total -> delta
    assert block["fused_xt_misses"] == 5
    assert block["fused_kernel_compiles"] == 2
    assert block["fused_scratch_bytes"] == 4096   # gauge -> AFTER value
    assert block["fused_block_rows"] == 32
    assert block["fused_num_warps"] == 4
    # an absent half of the window is null, never zero
    absent = worker.fused_window_block(None, None)
    assert absent["available"] is False
    for key in worker.FUSED_STATS_KEYS:
        assert absent[key] is None, key
    assert absent["note"]


def test_artifact_dir_falls_back_to_a_fixed_directory(monkeypatch):
    monkeypatch.delenv(common.ARTIFACT_DIR_ENV, raising=False)
    assert common.artifact_dir() == common.DEFAULT_ARTIFACT_DIR
    # an EMPTY export reads as unset: the engine treats "" as "tier off"
    monkeypatch.setenv(common.ARTIFACT_DIR_ENV, "  ")
    assert common.artifact_dir() == os.path.abspath(common.DEFAULT_ARTIFACT_DIR)


def test_no_arm_leaks_a_route_knob(monkeypatch):
    """Every arm starts from a cleared table, whatever the caller exported."""
    sentinels = {name: "leaked-" + name for name in common.ROUTE_ENV_VARS}
    for arm in common.ALL_ARMS:
        for name, value in sentinels.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv(common.ARTIFACT_DIR_ENV, os.path.join(os.sep, "tmp", "hook"))
        applied = arms.apply_arm_env(arm, common.parse_cell("p256-L64-i32-N1024"))
        own = applied["knobs"]
        for name, value in sentinels.items():
            if name in own:
                assert os.environ[name] == own[name] != value, (arm, name)
            else:
                assert name not in os.environ, (arm, name)
        # the harness hook is never turned into an engine variable by another arm
        if ARTIFACT_DIR_VAR not in own:
            assert ARTIFACT_DIR_VAR not in os.environ, arm


def test_cell_template_still_needs_its_cell():
    """The new placeholder does not soften the old contract for per-cell templates."""
    with pytest.raises(ValueError, match="unknown placeholder"):
        arms.apply_arm_env("evoxc-dense-iree-raised")
    applied = arms.apply_arm_env("evoxc-dense-iree-raised", common.parse_cell("p256-L128-i32-N1024"))
    assert applied["knobs"] == {"EVOGP_DENSE_MAX_TREE_LEN": "128"}


# ---------------------------------------------------------------------------
# the route report: the shape the policy is asked about
# ---------------------------------------------------------------------------
class _FakeDenseEngine(types.ModuleType):
    """A stand-in ``evogp.tree.dense_engine`` that RECORDS how it was called."""

    def __init__(self):
        super().__init__("evogp.tree.dense_engine")
        self.calls = {}
        self.EVAL_MODES = ("dense", "specialized", "batch", "auto")

    def resolve_eval_mode(self, value=None):
        # echo an EXPLICIT request (the fused branch keys on it), the default otherwise
        return "auto" if value is None else value

    def resolve_route(self, pop_size, *, rows=None, backend=None, eval_mode=None,
                      tree_len=None, **_kwargs):
        """The ONE route resolver: a fused request, and the shipped ``auto`` policy.

        The fused stub is unconditional (an explicit ``EVOGP_EVAL_MODE=fused`` request); the
        ``auto`` case models the real policy's crossover -- fused on this fake device box from
        :data:`FUSED_MIN_POP` up, specialized+numpy below it -- because an ``evoxc-auto``
        record has to be able to state BOTH answers.
        """
        self.calls["resolve_route"] = {
            "pop_size": pop_size, "rows": rows, "backend": backend,
            "eval_mode": eval_mode, "tree_len": tree_len,
        }
        if eval_mode != "fused" and (backend is not None or pop_size < FUSED_MIN_POP):
            return types.SimpleNamespace(
                use_dense=False, use_fused=False, backend="numpy",
                fallback_backend="numpy",
                reason="stub: below the fused crossover -> specialized on numpy",
                engine="specialized",
            )
        return types.SimpleNamespace(
            use_dense=False, use_fused=True, backend="iree", fallback_backend="numpy",
            reason="stub: fused served", engine="fused",
        )

    def resolve_backend_name(self, value, index=None):
        return "iree" if value != "numpy" else "numpy"

    def should_use_dense(self, pop_size, *, rows=None, tree_len=None, **_kwargs):
        self.calls["should_use_dense"] = {
            "pop_size": pop_size, "rows": rows, "tree_len": tree_len,
        }
        return True

    def auto_rows_decline(self, rows, *, pop_size=None, tree_len=None, **_kwargs):
        self.calls["auto_rows_decline"] = {
            "rows": rows, "pop_size": pop_size, "tree_len": tree_len,
        }
        return None


class _DecoyDenseEngine(types.ModuleType):
    """The REAL policy's slot on the parent package -- any call here is the bug.

    ``from evogp.tree import dense_engine`` answers with this attribute, so a stub that
    replaces only ``sys.modules["evogp.tree.dense_engine"]`` is invisible to it.  Every
    attribute access therefore raises, naming the resolution rule the harness must keep.
    """

    def __init__(self):
        super().__init__("evogp.tree.dense_engine")

    def __getattr__(self, name):
        raise AssertionError(
            f"the parent package's dense_engine attribute was consulted for {name!r}: "
            "the engine must be resolved from sys.modules (common.installed_module)"
        )


def _install_fake_dense_engine(monkeypatch, parent_dense_engine=None):
    """Put ``_FakeDenseEngine`` at ``sys.modules["evogp.tree.dense_engine"]`` for one test.

    ``parent_dense_engine`` is what the fake ``evogp.tree`` package exposes as its
    ``dense_engine`` ATTRIBUTE: the stub itself by default (both spellings agree), or
    :class:`_DecoyDenseEngine` to prove the resolution ignores that cached attribute.

    Hermetic in the ENVIRONMENT as well as in ``sys.modules``: ``route_decision`` reads
    ``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` as the fallback for a caller that
    pinned neither, and the project gate runs ``src/evogp/tests`` first, whose
    ``conftest.py`` leaves ``EVOGP_EVOXC_BACKEND=numpy`` in the ambient environment --
    an ambient pin would make the ``auto`` branch look like an explicit-backend request
    (and skip the resolver) in the gate but not in isolation.  So no route variable may
    survive into the decision: these tests are about the POLICY, not who ran before.
    """
    stub = _FakeDenseEngine()
    evogp = types.ModuleType("evogp")
    tree = types.ModuleType("evogp.tree")
    evogp.tree = tree
    tree.dense_engine = stub if parent_dense_engine is None else parent_dense_engine
    monkeypatch.setitem(sys.modules, "evogp", evogp)
    monkeypatch.setitem(sys.modules, "evogp.tree", tree)
    monkeypatch.setitem(sys.modules, "evogp.tree.dense_engine", stub)
    for name in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND") + tuple(common.ROUTE_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    return stub


@pytest.fixture
def fake_dense_engine(monkeypatch):
    """Install the stub as ``evogp.tree.dense_engine`` (no torch, no evogp import)."""
    return _install_fake_dense_engine(monkeypatch)


def test_route_decision_passes_the_population_shape(fake_dense_engine):
    decision = common.route_decision(256, 4096, None, "iree", tree_len=64)
    assert fake_dense_engine.calls["should_use_dense"] == {
        "pop_size": 256, "rows": 4096, "tree_len": 64,
    }
    assert fake_dense_engine.calls["auto_rows_decline"] == {
        "rows": 4096, "pop_size": 256, "tree_len": 64,
    }
    assert decision["would_use_dense"] is True
    # the pre-existing keys keep their meaning (the pop-only half is still shape-blind)
    assert set(decision) == {
        "eval_mode", "backend", "should_use_dense_pop_only",
        "should_use_dense_with_rows", "would_use_dense",
    }


def test_route_decision_tree_len_is_optional(fake_dense_engine):
    """A caller that does not know the stride still gets a decision, with ``None``."""
    common.route_decision(256, 1024, None, "iree")
    assert fake_dense_engine.calls["auto_rows_decline"]["tree_len"] is None


def test_route_decision_fused_asks_the_one_resolver(fake_dense_engine):
    """A ``fused`` request reports the resolver's engine, not ``resolve_backend_name``'s."""
    decision = common.route_decision(256, 4096, "fused", "iree", tree_len=64)
    assert fake_dense_engine.calls["resolve_route"] == {
        "pop_size": 256, "rows": 4096, "backend": "iree",
        "eval_mode": "fused", "tree_len": 64,
    }
    assert decision["eval_mode"] == "fused"
    assert decision["engine"] == "fused"
    assert decision["would_use_fused"] is True
    assert decision["fused_fallback_backend"] == "numpy"
    assert decision["fused_reason"] == "stub: fused served"
    # the backend is the RESOLVED one: without the resolver a served call would render as
    # ``fused/numpy -> fused`` and look like a decline
    assert decision["backend"] == "iree"
    # the four fused keys are ADDED, never renamed, and appear for a fused request only
    assert set(decision) - {
        "eval_mode", "backend", "should_use_dense_pop_only",
        "should_use_dense_with_rows", "would_use_dense",
    } == {"engine", "would_use_fused", "fused_fallback_backend", "fused_reason"}
    auto = common.route_decision(256, 4096, None, "iree", tree_len=64)
    assert "engine" not in auto and "would_use_fused" not in auto


def test_route_decision_auto_asks_the_one_resolver(fake_dense_engine):
    """The shipped default (no mode, no backend) must say WHICH engine served the cell.

    ``evoxc-auto`` sets neither variable, so the policy -- not a request -- owns the engine;
    without this branch a silent fall-back into ``specialized`` would hide inside the record.
    """
    decision = common.route_decision(256, 1024, None, None, tree_len=32)
    assert fake_dense_engine.calls["resolve_route"] == {
        "pop_size": 256, "rows": 1024, "backend": None, "eval_mode": None, "tree_len": 32,
    }
    assert decision["eval_mode"] == "auto"
    assert decision["engine"] == "fused"
    assert decision["backend"] == "iree"
    assert decision["would_use_fused"] is True
    assert decision["fused_fallback_backend"] == "numpy"
    assert decision["fused_reason"]
    # additive: the five pre-existing keys keep their meaning, the four resolver keys join them
    assert decision["would_use_dense"] is True
    assert set(decision) - {
        "eval_mode", "backend", "should_use_dense_pop_only",
        "should_use_dense_with_rows", "would_use_dense",
    } == {"engine", "would_use_fused", "fused_fallback_backend", "fused_reason"}
    # below the fused crossover the same call answers the specialized engine, so a record can
    # state a DECLINE as well as a served call
    small = common.route_decision(4, 64, None, None, tree_len=64)
    assert small["eval_mode"] == "auto"
    assert small["engine"] == "specialized"
    assert small["backend"] == "numpy"
    assert small["would_use_fused"] is False


def test_route_decision_resolves_the_engine_from_sys_modules(monkeypatch):
    """The engine handle is resolved FRESH from ``sys.modules``, never off the package.

    The parent package caches ``dense_engine`` as an attribute, so a ``from evogp.tree
    import dense_engine`` would return the REAL policy for a stub installed only at
    ``sys.modules["evogp.tree.dense_engine"]`` -- the order-dependent failure the gate hit
    once ``src/evogp/tests`` had imported ``evogp.tree`` for real.  The decoy parent
    attribute raises on any access, so a regression fails here instead of in the gate.
    """
    stub = _install_fake_dense_engine(monkeypatch, parent_dense_engine=_DecoyDenseEngine())
    decision = common.route_decision(256, 1024, None, None, tree_len=32)
    assert stub.calls["resolve_route"] == {
        "pop_size": 256, "rows": 1024, "backend": None, "eval_mode": None, "tree_len": 32,
    }
    assert decision["engine"] == "fused"
    assert decision["backend"] == "iree"


def test_fake_dense_engine_neutralises_a_leaked_backend_pin(monkeypatch):
    """A route variable leaked by an earlier suite cannot change the ``auto`` answer.

    ``route_decision`` honours ``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` when the
    caller pinned neither -- correct for a record, which must state the policy the
    process really follows -- so the FIXTURE is what makes these unit tests hermetic.
    The gate runs ``src/evogp/tests`` first and its ``conftest.py`` leaves
    ``EVOGP_EVOXC_BACKEND=numpy`` behind; without this neutralisation the shipped-default
    case looks like an explicit-backend request in the gate but not in isolation.
    """
    monkeypatch.setenv("EVOGP_EVOXC_BACKEND", "numpy")
    monkeypatch.setenv("EVOGP_EVAL_MODE", "dense")
    stub = _install_fake_dense_engine(monkeypatch)
    assert "EVOGP_EVOXC_BACKEND" not in os.environ
    assert "EVOGP_EVAL_MODE" not in os.environ
    decision = common.route_decision(256, 1024, None, None, tree_len=32)
    assert stub.calls["resolve_route"] == {
        "pop_size": 256, "rows": 1024, "backend": None, "eval_mode": None, "tree_len": 32,
    }
    assert decision["engine"] == "fused"


def test_installed_module_prefers_the_sys_modules_entry(fake_dense_engine):
    """``common.installed_module`` is the ONE resolution rule, and the stub wins.

    Torch-free by construction: the "installed" case is the fixture's stub.
    """
    assert common.installed_module("evogp.tree.dense_engine") is fake_dense_engine


def test_installed_module_imports_a_full_name_when_nothing_is_installed(
    tmp_path, monkeypatch
):
    """Absent from ``sys.modules``, the module is IMPORTED by its full dotted name."""
    package = tmp_path / "harness_resolution_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "engine.py").write_text("NAME = 'engine'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "harness_resolution_probe", raising=False)
    monkeypatch.delitem(sys.modules, "harness_resolution_probe.engine", raising=False)
    module = common.installed_module("harness_resolution_probe.engine")
    assert module.NAME == "engine"
    assert sys.modules["harness_resolution_probe.engine"] is module
    # a name no tree ships raises the import error the guarded callers already handle
    with pytest.raises(ImportError):
        common.installed_module("harness_resolution_probe.no_such_module")


def test_route_decision_reports_no_engine_for_a_route_it_does_not_run(fake_dense_engine):
    """A torch arm reports the policy but measures no engine: it must claim none.

    ``resolve_engine=False`` is that caller's switch -- an ``engine`` on a torch record would
    make ``check.is_fused_record`` gate it as a fused measurement it never made.
    """
    decision = common.route_decision(256, 4096, None, None, tree_len=64, resolve_engine=False)
    assert "engine" not in decision and "would_use_fused" not in decision
    assert "resolve_route" not in fake_dense_engine.calls
    # an explicit backend never reaches the fused route under auto, so the resolver is not
    # asked either (that arm keeps exactly its pre-existing five keys)
    explicit = common.route_decision(256, 4096, None, "numpy", tree_len=64)
    assert "engine" not in explicit


# ---------------------------------------------------------------------------
# which engine served the record: the gate's fused-record predicate
# ---------------------------------------------------------------------------
def test_fused_record_is_decided_by_the_resolved_engine():
    """An explicit fused arm OR a route that resolved fused -- the second half is the new one.

    ``evoxc-auto`` must not be able to hide a silent ``specialized`` fall-back behind its
    timings: a record whose ``routes[*].engine`` says fused is gated exactly like a fused arm.
    """
    # an explicit fused arm whose own routes do not carry an engine (a frozen record)
    assert check.is_fused_record(
        "evoxc-fused-iree", {"routes": {"sr_fitness": {"eval_mode": "fused"}}}
    ) is True
    # the shipped default that resolved fused
    auto_fused = {"routes": {"sr_fitness": {"eval_mode": "auto", "engine": "fused"}}}
    assert check.resolved_engine(auto_fused) == "fused"
    assert check.is_fused_record("evoxc-auto", auto_fused) is True
    # the shipped default that resolved anywhere else is NOT a fused record ...
    auto_specialized = {"routes": {"sr_fitness": {"eval_mode": "auto", "engine": "specialized"}}}
    assert check.resolved_engine(auto_specialized) == "specialized"
    assert check.is_fused_record("evoxc-auto", auto_specialized) is False
    # ... a record that predates the engine keys answers None (not recorded, not declined),
    # which is what keeps a frozen directory gating at exactly the checks it did before,
    frozen = {"routes": {"sr_fitness": {"eval_mode": "auto", "backend": "numpy"}}}
    assert check.resolved_engine(frozen) is None
    assert check.is_fused_record("evoxc-auto", frozen) is False
    # and an arm that never runs the policy can never be one
    assert check.resolved_engine({}) is None
    assert check.is_fused_record("torch-batched", {"routes": {}}) is False


def test_check_fused_accepts_an_auto_record_only_when_it_resolved_fused():
    """The routing FAIL widened to the auto shape; everything else is still a FAIL/WARN."""
    def record(eval_mode, engine):
        return {
            "routes": {"sr_fitness": {"eval_mode": eval_mode, "engine": engine}},
            "fused_cache": {"available": True, "served_every_workload": True},
            "workloads": {"sr_fitness": {"median_ms": 0.1, "cache": {"fused_served": True}}},
        }

    for eval_mode, engine in (("fused", "fused"), ("auto", "fused")):
        checker = check.Checker()
        check.check_fused(checker, "cell/arm", record(eval_mode, engine))
        assert checker.failures == [], (eval_mode, engine)
        assert checker.warnings == [], (eval_mode, engine)
    # an auto record that resolved a fall-back cannot pass as a fused measurement
    checker = check.Checker()
    check.check_fused(checker, "cell/arm", record("auto", "specialized"))
    assert checker.failures, "an auto record whose engine is specialized must FAIL"
    assert any("specialized" in message for message in checker.warnings)
    # ... and neither can a route with no engine at all
    checker = check.Checker()
    check.check_fused(checker, "cell/arm", record("auto", None))
    assert checker.failures
    # the launch proof stays a WARN, never a FAIL: the frozen campaign must keep gating
    checker = check.Checker()
    check.check_fused(checker, "cell/arm", {
        "routes": {"sr_fitness": {"eval_mode": "fused"}},
        "workloads": {"sr_fitness": {"median_ms": 0.1, "cache": {}}},
    })
    assert checker.failures == []
    assert checker.warnings


# ---------------------------------------------------------------------------
# the band / merge report: the compile-option seam
# ---------------------------------------------------------------------------
def test_band_report_carries_the_dispatch_formation():
    described = {
        "band_count": 15,
        "chunk_count": 1,
        "dispatch_formation": "aggressive",
        "dispatch_formation_requested": "auto",
    }
    block = band_report.band_report(described)
    assert block["dispatch_formation"] == "aggressive"
    assert block["dispatch_formation_requested"] == "auto"
    assert block["available"] is True
    assert "dispatch_formation" in band_report.BAND_REPORT_KEYS
    assert "dispatch_formation_requested" in band_report.BAND_REPORT_KEYS


def test_band_report_reads_none_for_a_plan_that_predates_the_knob():
    """Null, never a fabricated ``default``: the two keys stay ``None`` when absent."""
    block = band_report.band_report({"band_count": 4})
    assert block["dispatch_formation"] is None
    assert block["dispatch_formation_requested"] is None
    absent = band_report.band_report(None)
    assert absent["available"] is False
    for key in band_report.BAND_REPORT_KEYS:
        assert absent[key] is None, key


# ---------------------------------------------------------------------------
# the disk tier's counter block
# ---------------------------------------------------------------------------
def test_artifact_block_is_a_delta_with_after_state_gauges():
    before = {
        artifact.ENABLED_KEY: 1, artifact.FILES_KEY: 0, artifact.HITS_KEY: 0,
        artifact.MISSES_KEY: 3, artifact.LOAD_FAILURES_KEY: 0, artifact.STORES_KEY: 2,
        artifact.SAVE_FAILURES_KEY: 0,
    }
    after = {
        artifact.ENABLED_KEY: 1, artifact.FILES_KEY: 4, artifact.HITS_KEY: 4,
        artifact.MISSES_KEY: 5, artifact.LOAD_FAILURES_KEY: 1, artifact.STORES_KEY: 3,
        artifact.SAVE_FAILURES_KEY: 0,
    }
    block = artifact.window_block(before, after)
    assert block["available"] is True
    assert block["stats_fn"] == artifact.STATS_FN_PATH
    assert block[artifact.HITS_KEY] == 4
    assert block[artifact.MISSES_KEY] == 2
    assert block[artifact.LOAD_FAILURES_KEY] == 1
    assert block[artifact.STORES_KEY] == 1
    assert block[artifact.SAVE_FAILURES_KEY] == 0
    # the gauges are the AFTER snapshot, never a delta
    assert block[artifact.FILES_KEY] == 4
    assert block[artifact.ENABLED_KEY] == 1
    assert block["attempts"] == 6
    assert block["hit_rate"] == pytest.approx(0.6667)


def test_artifact_block_is_null_never_zero():
    absent = artifact.window_block(None, None)
    assert absent["available"] is False
    for key in artifact.COUNTER_KEYS:
        assert absent[key] is None, key
    assert absent["attempts"] is None and absent["hit_rate"] is None
    assert absent["note"]
    # a served but DISABLED tier says so instead of looking like "nothing happened"
    disabled = artifact.window_block(
        {artifact.ENABLED_KEY: 0, artifact.HITS_KEY: 0, artifact.MISSES_KEY: 0},
        {artifact.ENABLED_KEY: 0, artifact.HITS_KEY: 0, artifact.MISSES_KEY: 0},
    )
    assert disabled["available"] is True
    assert disabled[artifact.HITS_KEY] == 0
    assert disabled["note"] == artifact.DISABLED_NOTE
    assert artifact.stats_fn(types.ModuleType("m")) is None


# ---------------------------------------------------------------------------
# the schedule's cell exclusions: dropped WITH a reason, never silently
# ---------------------------------------------------------------------------
def test_cell_exclusion_keeps_the_d7_dense_arm_out_of_the_schedule():
    """``evoxc-dense-iree`` on the d7 cell is a stated exclusion, not a coverage rule."""
    d7 = common.parse_cell("p2048-L128-i32-N1024-d7")
    other = common.parse_cell("p256-L32-i8-N1024")
    reason = campaign.CELL_EXCLUSIONS[(d7["id"], "evoxc-dense-iree")]
    assert isinstance(reason, str) and "OOM" in reason
    assert campaign.exclusion_reason("evoxc-dense-iree", d7) == reason
    assert campaign.in_scope("evoxc-dense-iree", d7) is False
    # --full-specialized-iree widens the POPULATION scope, never this table
    assert campaign.in_scope("evoxc-dense-iree", d7, True) is False
    # the same arm stays in scope on every other cell, its raised-cap sibling included
    assert campaign.exclusion_reason("evoxc-dense-iree", other) is None
    assert campaign.in_scope("evoxc-dense-iree", other) is True
    assert campaign.in_scope("evoxc-dense-iree-raised", d7) is True
    # the registry itself is untouched: the arm exists, in its original position
    assert "evoxc-dense-iree" in common.EVOXC_ARMS
    assert common.ALL_ARMS.index("evoxc-dense-iree") < common.ALL_ARMS.index("evoxc-dense-iree-raised")
    # a drop states WHY, and the population-scope rule states its own reason
    assert campaign.drop_reason("evoxc-dense-iree", d7) == reason
    assert "coverage rule" in campaign.drop_reason("evoxc-specialized-iree", other)
    # the marked cell is the one the campaign would otherwise schedule
    scheduled = campaign.schedule([d7], ["evoxc-dense-iree"])
    assert scheduled == [{"cell": d7, "arm": "evoxc-dense-iree"}]


def test_scope_protocol_records_every_drop_with_its_reason():
    """``campaign.json``'s scope evidence: the drop, its reason, and the exclusion rule."""
    d7 = common.parse_cell("p2048-L128-i32-N1024-d7")
    small = common.parse_cell("p256-L32-i8-N1024")
    dropped = [
        {"cell": d7, "arm": "evoxc-dense-iree"},
        {"cell": small, "arm": "evoxc-specialized-iree"},
    ]
    block = campaign.scope_protocol(dropped, full=True)
    reasons = {(entry["cell"], entry["arm"]): entry["reason"]
               for entry in block["scope_dropped"]}
    assert reasons[("p2048-L128-i32-N1024-d7", "evoxc-dense-iree")] == \
        campaign.CELL_EXCLUSIONS[("p2048-L128-i32-N1024-d7", "evoxc-dense-iree")]
    # the population rule states its own reason, and `full=True` does not change the text
    assert "coverage rule" in reasons[("p256-L32-i8-N1024", "evoxc-specialized-iree")]
    assert all(isinstance(entry["reason"], str) and entry["reason"]
               for entry in block["scope_dropped"])
    assert block["scope_rule"] == {
        arm: list(rule["pops"]) for arm, rule in campaign.SCOPE.items()
    }
    entries = {entry["cell"]: entry for entry in block["cell_exclusions"]["entries"]}
    assert entries["p2048-L128-i32-N1024-d7"]["arm"] == "evoxc-dense-iree"
    assert "full-specialized-iree" in block["cell_exclusions"]["rule"]


def test_dry_run_states_a_cell_exclusion_and_its_reason(tmp_path, capsys):
    """The dry run names the dropped config plus the reason before anything is compiled."""
    code = campaign.main([
        "--out", str(tmp_path / "dry"),
        "--cells", "p256-L32-i8-N1024,p2048-L128-i32-N1024-d7",
        "--arms", "evoxc-dense-iree", "--dry-run",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "would run p256-L32-i8-N1024 evoxc-dense-iree" in printed
    reason = campaign.CELL_EXCLUSIONS[("p2048-L128-i32-N1024-d7", "evoxc-dense-iree")]
    assert ("would drop p2048-L128-i32-N1024-d7 evoxc-dense-iree: " + reason) in printed
    assert "would run p2048-L128-i32-N1024-d7 evoxc-dense-iree" not in printed


# ---------------------------------------------------------------------------
# the dry run: each new knob stated once
# ---------------------------------------------------------------------------
def test_dry_run_states_the_new_knobs_once(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(common.ARTIFACT_DIR_ENV, raising=False)
    out = tmp_path / "dry"
    code = campaign.main([
        "--out", str(out), "--cells", "p64-L64-i32-N64", "--arms", "batch", "--dry-run",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    # the new arms are scheduled, and each new variable is stated once per arm that
    # pins it (the dispatch formation has two such arms: aggressive and default)
    for arm in (
        "evoxc-batch-iree-aggressive",
        "evoxc-batch-iree-artifact",
        "evoxc-batch-iree-default",
    ):
        assert f"would run p64-L64-i32-N64 {arm}" in printed, arm
    assert printed.count(DISPATCH_ENV) == 2
    # the disk tier now has four arms: the artifact arm plus its three pool pins
    assert printed.count(ARTIFACT_DIR_VAR) == 4
    assert f"{DISPATCH_ENV}=aggressive" in printed
    assert f"{DISPATCH_ENV}=default" in printed
    # ... and the dry run resolves the route per arm, so the control reads ``default``
    # where the shipped default arm reads ``auto``
    policy_lines = {
        line.split()[2]: line for line in printed.splitlines()
        if line.startswith("batch policy ")
    }
    assert "dispatch_formation=default" in policy_lines["evoxc-batch-iree-default"]
    assert "dispatch_formation=auto" in policy_lines["evoxc-batch-iree"]
    # the disk directory is the campaign's own per-run default
    assert f"{ARTIFACT_DIR_VAR}={os.path.abspath(os.path.join(str(out), 'artifacts'))}" in printed


def test_dry_run_honours_an_explicit_artifact_dir(tmp_path):
    shared = tmp_path / "shared-cache"
    code = campaign.main([
        "--out", str(tmp_path / "dry"), "--cells", "p64-L64-i32-N64",
        "--arms", "evoxc-batch-iree-artifact", "--artifact-dir", str(shared),
        "--dry-run",
    ])
    assert code == 0
    assert campaign.resolve_artifact_dir(types.SimpleNamespace(
        artifact_dir=str(shared), out=str(tmp_path / "dry")
    )) == os.path.abspath(str(shared))


def test_resolve_artifact_dir_precedence(tmp_path, monkeypatch):
    """CLI path > exported hook > ``<out>/artifacts`` (per run)."""
    args = types.SimpleNamespace(artifact_dir=None, out=str(tmp_path / "run"))
    monkeypatch.delenv(common.ARTIFACT_DIR_ENV, raising=False)
    assert campaign.resolve_artifact_dir(args) == os.path.abspath(str(tmp_path / "run" / "artifacts"))
    monkeypatch.setenv(common.ARTIFACT_DIR_ENV, str(tmp_path / "exported"))
    assert campaign.resolve_artifact_dir(args) == os.path.abspath(str(tmp_path / "exported"))
    args.artifact_dir = str(tmp_path / "cli")
    assert campaign.resolve_artifact_dir(args) == os.path.abspath(str(tmp_path / "cli"))
