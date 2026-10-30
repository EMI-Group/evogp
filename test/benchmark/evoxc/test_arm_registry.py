"""Torch-free checks of the arm registry, the batch knob table and the route report.

This directory is normally not collected by pytest, but the project gate runs
``pytest test src/evogp/tests -q``, which walks into it -- so this module is the ONE
automated check that the harness can NAME what the engine grew, and it stays
import-light: ``common`` / ``arms`` / ``campaign`` / ``band_report`` / ``artifact``
only.  The last group (``test_dry_run_*``) runs ``campaign.main --dry-run``, which
resolves the batch policy from the tree on ``PYTHONPATH``; that is the harness's own
dry run, i.e. what a later agent inspects before paying a compile.

What is pinned here:

* the new ``EVOGP_BATCH_DISPATCH_FORMATION`` knob is in the single-sourced
  ``common.BATCH_KNOBS`` table exactly once, and it therefore joins
  ``BATCH_ENV_VARS`` / ``ROUTE_ENV_VARS`` (so it can never leak between arms);
* the disk tier's ``EVOGP_EVOXC_ARTIFACT_DIR`` is a route knob too
  (``ARTIFACT_ENV_VARS``), and the dense route's new ``EVOGP_DENSE_AUTO_BUDGET_MB``
  is part of ``DENSE_ENV_VARS``;
* every pre-existing arm id is unchanged (byte-identical, in registry order) and the
  two new arms -- ``evoxc-batch-iree-aggressive`` / ``evoxc-batch-iree-artifact`` --
  resolve to the env they claim, the disk tier through the harness's own hook;
* no arm leaks a route knob: every ``common.ROUTE_ENV_VARS`` entry is cleared before an
  arm's own spec is written, for every arm including the torch baselines;
* ``common.route_decision`` passes the population SHAPE (``pop_size`` / ``rows`` /
  ``tree_len``) into the policy, so a record states the memory-envelope decision and
  not a shape-blind one;
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
import common  # noqa: E402

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
    new = ("evoxc-batch-iree-aggressive", "evoxc-batch-iree-artifact")
    assert tuple(common.EVOXC_ARMS)[-len(new):] == new
    assert tuple(common.ALL_ARMS)[-len(new):] == new
    for arm in new:
        assert common.is_batch_arm(arm), arm
        assert common.EVOXC_ARMS[arm]["backend"] == "iree"
    # ``--arms batch`` picks them up from the ONE predicate (no extra rule)
    batch_arms = [arm for arm in common.EVOXC_ARMS if common.is_batch_arm(arm)]
    assert list(batch_arms) == ["evoxc-batch-iree", "evoxc-batch-iree-flat", *new]


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
        return "auto"

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


@pytest.fixture
def fake_dense_engine(monkeypatch):
    """Install the stub as ``evogp.tree.dense_engine`` (no torch, no evogp import)."""
    stub = _FakeDenseEngine()
    evogp = types.ModuleType("evogp")
    tree = types.ModuleType("evogp.tree")
    evogp.tree = tree
    tree.dense_engine = stub
    monkeypatch.setitem(sys.modules, "evogp", evogp)
    monkeypatch.setitem(sys.modules, "evogp.tree", tree)
    monkeypatch.setitem(sys.modules, "evogp.tree.dense_engine", stub)
    monkeypatch.delenv("EVOGP_EVAL_MODE", raising=False)
    return stub


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
    # the two new arms are scheduled, and each new variable is stated exactly once
    assert "evoxc-batch-iree-aggressive" in printed
    assert "evoxc-batch-iree-artifact" in printed
    assert printed.count(DISPATCH_ENV) == 1
    assert printed.count(ARTIFACT_DIR_VAR) == 1
    assert f"{DISPATCH_ENV}=aggressive" in printed
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
