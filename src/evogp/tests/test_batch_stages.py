"""Stage accounting and chunk geometry of the opt-in "batch" route.

``src/evogp/tree/batch_engine.py`` traces every DISTINCT program of a population
and merges those graphs into ONE executable per ``evoxc.batch_functions`` call.
Once the merged compile is split into several CHUNKS (``EVOGP_BATCH_COMPILE_WORKERS``)
a caller can no longer tell from the numbers alone what the route actually did:
the same loss comes back whether the population was merged in one graph, in two
bodies or in four.  This file pins the OBSERVABILITY contract that closes that
gap -- the measurement surface, never the economics:

* :mod:`evogp.tree.batch_stages` (``STAGE_NAMES`` / ``StageTimer`` /
  ``stage_stats`` / ``stage_reset``) and its facade
  ``batch_engine.batch_stage_stats()`` / ``batch_engine.batch_stage_reset()``
  report EXACTLY 32 counters: for each of the ten stages ``decode``, ``dedup``,
  ``trace``, ``merge``, ``lower``, ``compile``, ``load``, ``upload``, ``run``,
  ``gather`` the keys ``batch_<stage>_calls`` (int), ``batch_<stage>_total_ms``
  (float) and ``batch_<stage>_max_ms`` (float), plus ``batch_compile_wall_ms``
  (float) and ``batch_compile_jobs`` (int).  Nothing else -- an added key breaks
  the callers that read a delta, and a stage key leaking into ``batch_cache_stats``
  (exactly its 8 keys), ``cache_stats()``, ``dense_cache_stats()``,
  ``artifact_cache_stats()`` or ``batch_content_stats()`` breaks the files that
  pin those surfaces;
* the counters MOVE for a real ``plan_batch`` -> ``build_batch`` ->
  ``sr_fitness_batch`` sequence, in the right order (the plan-side stages on the
  plan, ``lower``/``compile``/``load`` on the build, ``run``/``gather`` on the
  evaluation) and with sane arithmetic (``0 <= max_ms <= total_ms``, a cold build
  has ``compile_jobs >= 1`` and ``compile_wall_ms > 0``);
* a reused executable adds NO compile job (the economics claim "the compile is
  shared" is only observable through this counter);
* ``batch_stage_reset()`` zeroes all 32 -- through both the facade and
  ``batch_stages`` itself -- while :func:`evogp.tree.evoxc_engine.clear_cache`
  does NOT: the stages are MEASUREMENT, the front tables are cache STATE;
* ``describe_plan`` / ``BatchPlan`` / ``BatchExecutable`` carry the same
  information structurally: ``compile_workers``, ``chunk_count``,
  ``chunk_sizes``, a plan-side ``stage_ms`` dict and a ``build_stage_ms`` dict
  with the ``lower``/``compile``/``load`` totals plus the per-chunk
  ``lower#0`` / ``compile#0`` / ``load#0`` entries.

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_chunking.py`` owns the split itself (bitwise equality of a chunked
  run against the single executable, the partition rule, the exec-table keying
  and the ``EVOGP_BATCH_COMPILE_WORKERS`` resolution);
* ``test_batch_content_key.py`` owns the second half of the plan key (the content
  digest) and its own 5-key stats surface;
* ``test_batch_engine.py`` / ``test_batch_knobs.py`` / ``test_batch_route.py`` own
  the numeric contract, the knobs and the public wiring.

Mechanics
---------
* numpy ONLY (``backend="numpy"`` passed explicitly), CPU tensors, tiny shapes
  (4 distinct programs, ``N = 5`` rows, ``L = 16``), so every numpy compile stays
  milliseconds.  The plan memo is bracketed with ``evoxc_engine.clear_cache()``.
* The stage counters are read as DELTAS / after an explicit
  ``batch_stage_reset()``: they are process-global measurement, not per-call state.
* No wall-clock threshold anywhere -- the assertions are key sets, ordering,
  monotonicity and strict positivity of a measured compile, never "fast enough".
* ``upload`` is the backend-dependent stage (the numpy backend consumes the host
  array directly and never uploads), so it is asserted as non-negative and
  arithmetically consistent rather than ``> 0``; the nine other stages must have
  run.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_stages.py -q -p no:cacheprovider
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from typing import Any, Dict, Iterator, Sequence, Tuple

import pytest
import torch

# --------------------------------------------------------------------------- #
# shared helpers: the tree-spec DSL lives in test_equivalence.py (imported by
# path, exactly like the sibling engine-round files do)
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import test_equivalence as te  # noqa: E402  (the shared spec DSL)

from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_knobs  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: this file is about counters, not economics)
# --------------------------------------------------------------------------- #
#: The host backend: nothing compiles to a device, nothing is uploaded.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: FOUR distinct programs -- three computed ones and one fully constant tree, i.e.
#: the population shape a GP loop presents.  Fewer than ``MIN_CHUNK_FUNCTIONS``
#: (32), so the DEFAULT plan here is the single-chunk one.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
POP_SIZE = len(_PROGRAMS)

#: The ten instrumented stages, exactly as the round's contract enumerates them.
STAGE_NAMES_TEN: Tuple[str, ...] = (
    "decode",
    "dedup",
    "trace",
    "merge",
    "lower",
    "compile",
    "load",
    "upload",
    "run",
    "gather",
)
#: The stages a ``plan_batch`` call (host decode + dedup + trace + merge) accounts for.
PLAN_STAGES: Tuple[str, ...] = ("decode", "dedup", "trace", "merge")
#: The stages a ``build_batch`` call (lower + compile + load) accounts for.
BUILD_STAGES: Tuple[str, ...] = ("lower", "compile", "load")
#: The stages one ``evoxc.run`` + row expansion accounts for (``upload`` optional).
EVAL_STAGES: Tuple[str, ...] = ("run", "gather")

#: The exact key set of ``batch_cache_stats()`` (two ``dense_memo.DenseMemo``
#: tables) -- pinned here only to prove the stage counters stayed out of it.
_CACHE_STAT_KEYS = frozenset(
    {
        "batch_plan_hits",
        "batch_plan_misses",
        "batch_plan_entries",
        "batch_plan_maxsize",
        "batch_exec_hits",
        "batch_exec_misses",
        "batch_exec_entries",
        "batch_exec_maxsize",
    }
)
#: ``cache_stats()``'s documented 8 keys (the engine's own contract).
_ENGINE_STAT_KEYS = frozenset(
    {
        "hits",
        "misses",
        "entries",
        "maxsize",
        "memo_hits",
        "memo_misses",
        "memo_entries",
        "memo_maxsize",
    }
)
#: ``batch_content_stats()``'s 5 keys (owned by ``test_batch_content_key.py``;
#: repeated here because the stage counters must stay out of THAT surface too).
_CONTENT_STAT_KEYS = frozenset(
    {
        "batch_content_hits",
        "batch_content_misses",
        "batch_content_entries",
        "batch_content_maxsize",
        "batch_content_rejects",
    }
)

#: The knobs an ambient value could change under the assertions below.  The
#: compile-worker knob joins them (read through ``getattr`` so this module still
#: COLLECTS while the tree layer that defines it is landing in parallel; the
#: fixture then deletes it like every other knob).
_ENV_KNOBS: Tuple[str, ...] = (
    B.BAND_SIZE_ENV,
    B.BANDS_ENV,
    B.SWITCH_OPS_ENV,
    B.ALIGNMENT_ENV,
    getattr(batch_knobs, "COMPILE_WORKERS_ENV", "EVOGP_BATCH_COMPILE_WORKERS"),
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _expected_stage_keys() -> frozenset:
    """The 32 documented counter names: ``10 * (calls, total_ms, max_ms)`` + 2."""
    keys = {
        f"batch_{stage}_{suffix}"
        for stage in STAGE_NAMES_TEN
        for suffix in ("calls", "total_ms", "max_ms")
    }
    keys.update({"batch_compile_wall_ms", "batch_compile_jobs"})
    return frozenset(keys)


def _stages_module() -> Any:
    """``evogp.tree.batch_stages`` imported at CALL time (it lands in parallel).

    A module-level ``from evogp.tree import batch_stages`` would fail the WHOLE
    file at collection time instead of failing each test on the missing API, and it
    would hide every other assertion behind one ``ImportError``.
    """
    return importlib.import_module("evogp.tree.batch_stages")


def _stage_stats() -> Dict[str, Any]:
    """The public stage counters (``batch_engine`` is the import path for callers)."""
    return B.batch_stage_stats()


def _reset() -> None:
    """Zero the process-global stage counters (measurement, not cache state)."""
    B.batch_stage_reset()


def _population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors of ``_PROGRAMS``."""
    return te.rows_from_specs(_PROGRAMS, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int = N_ROWS, seed: int = 101) -> torch.Tensor:
    """Deterministic ``(rows, 1)`` SR labels from a LOCAL generator."""
    return te.x_random(rows, 1, seed=seed)


def _plan() -> Any:
    """``plan_batch`` over a FRESH population (so no plan-memo hit can hide the work).

    The loss variant is used throughout: it is what the training loop spends its
    compiles on and its merged output is one scalar per function.  ``plan_batch``
    never compiles, so this is cheap.
    """
    values, types, sizes = _population()
    return B.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        _x(),
        per_row=False,
        variant=B.LOSS_VARIANT,
        labels=_labels(),
        backend=BACKEND,
    )


def _loss() -> torch.Tensor:
    """One ``sr_fitness_batch`` over a fresh population (numpy backend)."""
    values, types, sizes = _population()
    return B.sr_fitness_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        _x(),
        _labels(),
        True,
        backend=BACKEND,
    )


# --------------------------------------------------------------------------- #
# hermeticity
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the knobs, admit small populations, and drop the front tables.

    The knobs are DELETED (not pinned to a default) so "4 workers / 32 functions
    per chunk" means the same thing on a box that exports them; ``MIN_POP`` is SET
    to ``1`` because the library default (64) is a real decline for every fixture
    here.  ``evoxc_engine.clear_cache()`` brackets the caches; the STAGE counters
    are deliberately NOT reset here -- the tests that read them call
    :func:`_reset` themselves, and one test pins that ``clear_cache()`` does not
    touch them.
    """
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    E.clear_cache()
    try:
        yield
    finally:
        E.clear_cache()


# =========================================================================== #
# 1. the measurement surface: ten stages, exactly 32 counters
# =========================================================================== #
def test_the_stage_surface_is_exactly_the_documented_thirty_two_counters() -> None:
    """``batch_stage_stats()`` is EXACTLY 32 keys, with the documented types.

    An EXTRA key (a per-chunk breakdown, a second wall-clock total) would make the
    counter contract unbounded, and a MISSING one would silently report zero work;
    ``set(stats) == expected`` is therefore asserted, never ``>=``.  The types are
    part of it: the call counters are ints (delta arithmetic) and the timings are
    floats (they come from a clock, never from a count).
    """
    stats = _stage_stats()
    expected = _expected_stage_keys()
    assert len(expected) == 32, len(expected)
    assert set(stats) == expected, sorted(set(stats) ^ expected)
    for stage in STAGE_NAMES_TEN:
        assert isinstance(stats[f"batch_{stage}_calls"], int), stage
        assert isinstance(stats[f"batch_{stage}_total_ms"], float), stage
        assert isinstance(stats[f"batch_{stage}_max_ms"], float), stage
    assert isinstance(stats["batch_compile_jobs"], int)
    assert isinstance(stats["batch_compile_wall_ms"], float)
    assert set(_stages_module().stage_stats()) == expected, (
        "the module surface and the batch_engine facade must report the same table"
    )


def test_the_stage_names_are_the_ten_documented_stages() -> None:
    """``STAGE_NAMES`` is the ten stages the counters are built from.

    The order is an implementation detail (the ``StageTimer`` may register them in
    any grouping), the MEMBERSHIP is not: a renamed stage silently orphans its
    three counters, so the set is asserted and every name must appear in the stats
    keys.
    """
    module = _stages_module()
    names = tuple(module.STAGE_NAMES)
    assert len(names) == 10, names
    assert set(names) == set(STAGE_NAMES_TEN), sorted(set(names) ^ set(STAGE_NAMES_TEN))
    stats = _stage_stats()
    for stage in names:
        assert f"batch_{stage}_calls" in stats, stage


def test_the_stage_counters_are_disjoint_from_every_other_stats_surface() -> None:
    """Six stats surfaces, six disjoint key sets.

    ``batch_cache_stats()`` (exactly 8 keys), ``cache_stats()``, ``dense_cache_stats()``,
    ``artifact_cache_stats()`` and ``batch_content_stats()`` are all pinned by other
    files in this suite.  The stage counters are MEASUREMENT and must not be folded
    into any of them -- a caller reading a cache delta would otherwise see it move
    on every evaluation.
    """
    stage_keys = set(_stage_stats())
    assert stage_keys == set(_expected_stage_keys())
    surfaces = {
        "batch_cache_stats": set(B.batch_cache_stats()),
        "cache_stats": set(E.cache_stats()),
        "dense_cache_stats": set(D.dense_cache_stats()),
        "artifact_cache_stats": set(E.artifact_cache_stats()),
        "batch_content_stats": set(B.batch_content_stats()),
    }
    assert surfaces["batch_cache_stats"] == set(_CACHE_STAT_KEYS)
    assert surfaces["cache_stats"] == set(_ENGINE_STAT_KEYS)
    assert surfaces["batch_content_stats"] == set(_CONTENT_STAT_KEYS)
    assert surfaces["dense_cache_stats"], "the dense route registers its own tables"
    for name, keys in surfaces.items():
        assert stage_keys.isdisjoint(keys), (name, sorted(stage_keys & keys))
        leaked = sorted(
            key
            for key in keys
            if key.endswith(("_calls", "_total_ms", "_max_ms"))
            or key in ("batch_compile_wall_ms", "batch_compile_jobs")
        )
        assert not leaked, f"a stage counter leaked into {name}: {leaked}"


# =========================================================================== #
# 2. the counters move on real work (plan -> build -> run)
# =========================================================================== #
def test_a_real_plan_build_and_run_moves_the_stage_counters() -> None:
    """Plan, build and evaluation each account for THEIR stages, in order.

    This is the whole point of the surface: the plan-side stages (decode, dedup,
    trace, merge) must be attributed to ``plan_batch``, the build stages
    (lower/compile/load) to ``build_batch`` and run/gather to the evaluation --
    a single counter for "the call" would pass a smoke test and tell a caller
    nothing.  A cold build additionally has ``compile_jobs >= 1`` and a POSITIVE
    ``compile_wall_ms``: the merged compile is real work, not a rounding artifact.
    """
    _reset()
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()

    plan = B.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        per_row=False,
        variant=B.LOSS_VARIANT,
        labels=labels,
        backend=BACKEND,
    )
    after_plan = _stage_stats()
    for stage in PLAN_STAGES:
        assert after_plan[f"batch_{stage}_calls"] >= 1, stage
    for stage in BUILD_STAGES + EVAL_STAGES + ("upload",):
        assert after_plan[f"batch_{stage}_calls"] == 0, (
            f"{stage} must not be accounted for by plan_batch"
        )
    assert after_plan["batch_compile_jobs"] == 0, "plan_batch never compiles"
    assert after_plan["batch_compile_wall_ms"] == 0.0

    entry = B.build_batch(plan)
    assert entry.executable is not None, "the fixture must be the single-chunk path"
    after_build = _stage_stats()
    for stage in BUILD_STAGES:
        assert after_build[f"batch_{stage}_calls"] >= 1, stage
    assert after_build["batch_run_calls"] == 0, "build_batch never runs the graph"
    assert after_build["batch_compile_jobs"] >= 1
    assert after_build["batch_compile_wall_ms"] > 0.0

    loss = B.sr_fitness_batch(
        values, types, sizes, INPUT_LEN, 1, inputs, labels, True, backend=BACKEND
    )
    after_run = _stage_stats()
    for stage in EVAL_STAGES:
        assert after_run[f"batch_{stage}_calls"] >= 1, stage
    assert tuple(loss.shape) == (POP_SIZE,), tuple(loss.shape)

    for stage in STAGE_NAMES_TEN:
        calls = after_run[f"batch_{stage}_calls"]
        total = after_run[f"batch_{stage}_total_ms"]
        peak = after_run[f"batch_{stage}_max_ms"]
        assert isinstance(calls, int) and calls >= 0, stage
        assert 0.0 <= peak <= total, (stage, peak, total)
    ran = {stage for stage in STAGE_NAMES_TEN if after_run[f"batch_{stage}_calls"] > 0}
    assert set(PLAN_STAGES + BUILD_STAGES + EVAL_STAGES) <= ran, sorted(ran)
    assert isinstance(after_run["batch_compile_jobs"], int)
    assert after_run["batch_compile_jobs"] >= 1
    assert after_run["batch_compile_wall_ms"] > 0.0


def test_a_reused_executable_adds_no_compile_job() -> None:
    """The compile is shared across populations with the same program multiset.

    ``BATCH_EXEC_MEMO`` is keyed on the ACTUAL program multiset, so a second
    population with the same content re-binds the compiled body instead of
    compiling again -- and ``batch_compile_jobs`` is the counter that proves it.
    Without this test the counter could simply count ``build_batch`` calls and the
    reuse claim would be unobservable.
    """
    _reset()
    B.build_batch(_plan())
    first = _stage_stats()
    assert first["batch_compile_jobs"] == 1
    assert first["batch_compile_calls"] == 1

    B.build_batch(_plan())  # a FRESH population object, the same content
    second = _stage_stats()
    assert second["batch_compile_jobs"] == 1, "a reused body is not a new compile job"
    assert second["batch_compile_calls"] == 1
    assert second["batch_lower_calls"] == 1
    assert B.batch_cache_stats()["batch_exec_hits"] == 1, (
        "the second build must be an executable-memo hit"
    )


# =========================================================================== #
# 3. reset vs invalidation (measurement is not cache state)
# =========================================================================== #
def test_batch_stage_reset_zeroes_all_thirty_two_counters() -> None:
    """``batch_stage_reset()`` zeroes every counter, through both surfaces.

    The reset is the caller's "start a new measurement window" hook, so it has to
    be complete (a leftover ``max_ms`` would make a fresh window look like it
    already saw a slow call) and it has to be visible on the module table the
    facade re-exports, not only in a stale copy.
    """
    _reset()
    B.build_batch(_plan())
    _loss()
    measured = _stage_stats()
    assert any(value for value in measured.values()), (
        "the fixture must have measured something before the reset"
    )

    _reset()
    zeroed = _stage_stats()
    assert set(zeroed) == _expected_stage_keys()
    assert all(value == 0 for value in zeroed.values()), {
        key: value for key, value in zeroed.items() if value
    }
    module_zeroed = _stages_module().stage_stats()
    assert set(module_zeroed) == _expected_stage_keys()
    assert all(value == 0 for value in module_zeroed.values()), (
        "the batch_stages surface must report the same reset table"
    )


def test_clear_cache_does_not_zero_the_stage_counters() -> None:
    """``evoxc_engine.clear_cache()`` drops cache STATE and keeps MEASUREMENT.

    The engine's invalidation point exists so a caller can change a compile knob
    and re-pay the work; the stage counters answer "what did this process spend",
    which a cache reset would falsify (a benchmark brackets its cache, not its
    clock).  So ``clear_cache()`` empties the front tables and leaves the 32
    counters exactly where they were.
    """
    _reset()
    B.build_batch(_plan())
    before = _stage_stats()
    assert before["batch_compile_jobs"] >= 1, "the fixture must have compiled"
    assert B.batch_cache_stats()["batch_exec_entries"] == 1

    E.clear_cache()

    after = _stage_stats()
    assert B.batch_cache_stats()["batch_exec_entries"] == 0, (
        "clear_cache() must still drop the front tables"
    )
    assert after == before, (
        "clear_cache() must not touch the stage measurement: "
        f"{sorted(key for key in after if after[key] != before[key])}"
    )


# =========================================================================== #
# 4. the structural report: describe_plan / BatchPlan / BatchExecutable
# =========================================================================== #
def test_describe_plan_reports_the_chunk_geometry_and_the_plan_stage_timings() -> None:
    """``describe_plan`` gained exactly the four chunk/stage fields it documents.

    A caller (and the runner) has to be able to tell from the REPORT -- without
    touching a private record -- how many workers were resolved, whether the
    population was split, and how the split was made.  ``chunk_sizes`` must
    account for every merged function, and ``stage_ms`` must carry the plan-side
    stages (it is the plan's own share of the accounting).
    """
    plan = _plan()
    info = B.describe_plan(plan)

    assert isinstance(info["compile_workers"], int)
    assert info["compile_workers"] >= 1
    assert isinstance(info["chunk_count"], int)
    assert info["chunk_count"] == len(plan.chunks) >= 1
    assert isinstance(info["chunk_sizes"], list)
    assert all(isinstance(size, int) for size in info["chunk_sizes"])
    assert info["chunk_sizes"] == list(plan.chunk_sizes)
    assert sum(info["chunk_sizes"]) == info["functions"] == len(plan.functions)
    assert plan.compile_workers == info["compile_workers"]

    assert isinstance(info["stage_ms"], dict)
    assert set(PLAN_STAGES) <= set(info["stage_ms"]), sorted(info["stage_ms"])
    assert set(info["stage_ms"]) <= set(_stages_module().STAGE_NAMES), sorted(
        info["stage_ms"]
    )
    assert all(value >= 0.0 for value in info["stage_ms"].values()), info["stage_ms"]


def test_the_single_chunk_build_reports_the_build_stages_and_the_one_body() -> None:
    """A single-chunk plan keeps the merged graph, the graphs/specs and one body.

    The chunked path stores per-chunk graphs instead, so the single-chunk contract
    is exactly what must NOT degrade while chunking lands: ``merged``/``graphs``/
    ``specs``/``report`` populated, ``executables`` of length one and ``executable``
    that body, plus the ``lower``/``compile``/``load`` totals and their
    ``#0`` per-chunk entries in ``build_stage_ms``.
    """
    plan = _plan()
    assert len(plan.chunks) == 1, "4 functions are below MIN_CHUNK_FUNCTIONS (32)"
    assert plan.merged is not None
    assert len(plan.graphs) == len(plan.functions)
    assert len(plan.specs) == 2, "the data block and the label block (loss variant)"
    assert plan.chunk_sizes == (len(plan.functions),)
    assert plan.report, "the merged graph's report must survive the single-chunk path"
    assert set(PLAN_STAGES) <= set(plan.stage_ms), sorted(plan.stage_ms)
    assert all(value >= 0.0 for value in plan.stage_ms.values()), plan.stage_ms

    entry = B.build_batch(plan)
    assert len(entry.executables) == 1
    assert entry.executable is entry.executables[0], (
        "the single-chunk plan exposes its one body through both fields"
    )
    stages = entry.build_stage_ms
    for stage in BUILD_STAGES:
        assert stage in stages, sorted(stages)
        assert stages[stage] >= 0.0, stage
        assert f"{stage}#0" in stages, sorted(stages)
        assert stages[f"{stage}#0"] >= 0.0, stage
    assert stages["compile"] > 0.0, "a cold numpy compile takes measurable time"
