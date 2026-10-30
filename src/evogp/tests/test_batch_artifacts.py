"""The batch route's persistent on-disk artifact tier (``EVOGP_EVOXC_ARTIFACT_DIR``).

``evogp.tree.batch_artifacts`` reuses the engine's disk layer
(``evogp.tree.artifact_cache``) for the opt-in batch route: one compiled body is
one file, looked up BEFORE ``evoxc.lower`` / ``compile`` / ``load`` and saved
after a real build.  ``evogp.tree.artifact_cache`` itself is pinned by
``test_artifact_cache.py`` (the dense / specialized round trips, the key codec,
the atomic write); this file pins what the BATCH route adds on top of it:

* **off by default** -- with ``EVOGP_EVOXC_ARTIFACT_DIR`` unset the tier costs one
  environment read per body: no key is derived, no directory is created, no file
  is opened and the route's seven counters stay at zero;
* **the route's own region** -- a body is written to ``<dir>/batch/``, never the
  root, so one directory serves the dense / specialized routes (which keep the
  root) without a name collision;
* **a hit skips the compile** -- after ``evoxc_engine.clear_cache()`` (the
  IN-MEMORY invalidation, which never deletes a file) the next call of the same
  population reloads the body: ``batch_artifact_hits`` grows,
  ``batch_chunks.build_bodies`` is not called at all, and the numbers are bitwise
  identical to the fresh build;
* **a miss is per CONTENT** -- the body's name carries a digest of its merged
  function list (the same bytes the executable memo keys on), so a different
  population is a different file, and the same population at the same plan is the
  same file; the resolved ``dispatch_formation`` is part of the artifact key, so a
  re-spelt compile option is a different file too;
* **a corrupt file can only cost one rebuild** -- ``garbage`` in the artifact is a
  ``load_failures`` increment, a real compile and a rewritten file, never a wrong
  number;
* **the counters are their own surface** -- ``batch_artifact_stats()`` is exactly
  seven ``batch_artifact_*`` keys, disjoint from the eight-key
  ``batch_cache_stats()``, the five-key ``batch_content_stats()`` and the
  whole-disk ``artifact_cache_stats()``;
* **the region and the reset semantics** -- ``batch_artifact_reset()`` zeroes the
  counters and touches no file (a dense-style artifact in the root survives it),
  and ``clear_cache()`` stays memory-only.

Mechanics
---------
* numpy ONLY (``backend="numpy"`` passed explicitly), CPU tensors, tiny fixtures
  (four distinct programs, ``N = 5`` rows, ``L = 16``), so every compile is a few
  milliseconds of the reference interpreter.
* The switch is a per-test ``monkeypatch.setenv`` on pytest's ``tmp_path``: no
  test can leave the disk layer on for another one, and
  ``evoxc_engine.clear_cache()`` brackets the in-memory tables (whose lifetime is
  deliberately separate -- that separation is itself asserted here).
* ``EVOGP_BATCH_MIN_POP=1`` admits the tiny populations (the library default of 64
  is the measured break-even, i.e. a real decline).
* No wall-clock assertion: the checks are values, file paths/counts and counters.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_artifacts.py -q -p no:cacheprovider
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Any, Dict, Iterator, List, Sequence, Tuple

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

from evogp.tree import artifact_cache  # noqa: E402
from evogp.tree import batch_artifacts  # noqa: E402
from evogp.tree import batch_build  # noqa: E402
from evogp.tree import batch_chunks  # noqa: E402
from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: every case is a few millisecond numpy compile)
# --------------------------------------------------------------------------- #
#: The host backend: the body being cached is a numpy-interpreter executable.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: FOUR distinct programs -- three computed ones and one fully constant tree.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
#: A SECOND population of the SAME size, program for program different: the
#: content digest (and therefore the file name) must separate the two.
_PROGRAMS_B: Sequence[Any] = (
    te.fn(te.SUB, te.var(0), te.const(1.5)),
    te.fn(te.ADD, te.var(1), te.var(1)),
    te.var(0),
    te.const(0.25),
)
POP_SIZE = len(_PROGRAMS)
#: Every knob an ambient value could move out from under the assertions (the
#: compile-worker count included: it decides whether a plan has one body or many).
_ENV_KNOBS: Tuple[str, ...] = (
    B.BAND_SIZE_ENV,
    B.BANDS_ENV,
    B.SWITCH_OPS_ENV,
    B.ALIGNMENT_ENV,
    B.MIN_POP_ENV,
    B.CACHE_SIZE_ENV,
    batch_artifacts.DIR_ENV,
)
#: The exact key set of ``batch_artifact_stats()``: two describing the tier, five
#: counting it.  Pinned as a set so an added/missing/renamed key fails here.
_ARTIFACT_STAT_KEYS = frozenset(
    {
        "batch_artifact_enabled",
        "batch_artifact_files",
        "batch_artifact_hits",
        "batch_artifact_misses",
        "batch_artifact_load_failures",
        "batch_artifact_stores",
        "batch_artifact_save_failures",
    }
)


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the knobs (the artifact switch included) and bracket the counters.

    The knobs are DELETED so "the tier is off by default" means the same thing on a
    box that exports the directory; ``EVOGP_BATCH_MIN_POP`` is SET to ``1`` because
    the library default (64) is a real decline for these fixtures.  Both counter
    surfaces are reset per test (the route's own and the whole disk layer's) and
    ``clear_cache()`` brackets the in-memory tables -- which never touches a file,
    the point of the two lifetimes.
    """
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(B.MIN_POP_ENV, "1")
    B.batch_artifact_reset()
    artifact_cache.reset_stats()
    E.clear_cache()
    try:
        yield
    finally:
        E.clear_cache()
        B.batch_artifact_reset()
        artifact_cache.reset_stats()


@pytest.fixture
def store(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Enable the disk layer on a fresh ``tmp_path`` directory and zero the counters."""
    directory = tmp_path / "artifacts"
    monkeypatch.setenv(batch_artifacts.DIR_ENV, str(directory))
    B.batch_artifact_reset()
    artifact_cache.reset_stats()
    return directory


def _population(
    programs: Sequence[Any] = _PROGRAMS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors of ``programs``."""
    return te.rows_from_specs(programs, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int = N_ROWS, seed: int = 101) -> torch.Tensor:
    """Deterministic ``(rows, 1)`` SR labels from a LOCAL generator."""
    return te.x_random(rows, 1, seed=seed)


def _loss(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """``sr_fitness_batch`` with the shape grid and the numpy backend pinned."""
    return B.sr_fitness_batch(
        values, types, sizes, INPUT_LEN, 1, inputs, labels, True, backend=BACKEND
    )


def _plan(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
) -> Any:
    """The plan of the same population (host work only -- nothing is compiled)."""
    return B.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=False,
        variant=B.LOSS_VARIANT,
        labels=_labels(x.shape[0]),
        backend=BACKEND,
    )


def _files(directory: pathlib.Path) -> List[pathlib.Path]:
    """Every artifact file below ``directory`` (root and the ``batch/`` region)."""
    if not directory.is_dir():
        return []
    return sorted(directory.rglob(f"*{artifact_cache.SUFFIX}"))


def _region(directory: pathlib.Path) -> pathlib.Path:
    """The batch route's own region below the enabled directory."""
    return directory / batch_artifacts.SUBDIR


class _BuildCounter:
    """Count ``batch_chunks.build_bodies`` calls (a hit must reach none)."""

    def __init__(self) -> None:
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = batch_chunks.build_bodies

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", "build_bodies")
        monkeypatch.setattr(batch_chunks, "build_bodies", wrapper)


# =========================================================================== #
# 1. off by default: nothing on disk, nothing derived
# =========================================================================== #
def test_the_tier_is_off_by_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset means one environment read per body: zeros, no directory, no file."""
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_enabled"] == 0
    assert stats["batch_artifact_files"] == 0
    assert stats["batch_artifact_hits"] == 0
    assert stats["batch_artifact_misses"] == 0
    assert B.batch_artifact_stats() == stats, "the off state must be stable"
    monkeypatch.chdir(tmp_path)  # any relative write would land here
    values, types, sizes = _population()
    loss = _loss(values, types, sizes, _x(), _labels())
    assert tuple(loss.shape) == (POP_SIZE,)
    assert B.batch_artifact_stats() == stats, "the off route moved a counter"
    assert _files(tmp_path) == [], "a file was written with the switch off"


def test_the_off_route_never_touches_the_disk_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_body`` returns before the engine's loader/saver: no stat, no key."""
    monkeypatch.setattr(
        artifact_cache, "load", lambda *a, **k: pytest.fail("load while disabled")
    )
    monkeypatch.setattr(
        artifact_cache, "save", lambda *a, **k: pytest.fail("save while disabled")
    )
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    chunk = plan.chunks[0]
    device = plan.device
    assert batch_artifacts.body_path(plan, chunk, BACKEND, device, {}) is None
    assert batch_artifacts.load_body(plan, chunk, BACKEND, device, {}) == (
        None,
        None,
        0.0,
    )
    assert batch_artifacts.save_body(object(), None) is False
    assert tuple(_loss(values, types, sizes, _x(), _labels()).shape) == (POP_SIZE,)
    assert B.batch_artifact_stats()["batch_artifact_stores"] == 0


# =========================================================================== #
# 2. the cold build: one body, one file, in the route's own region
# =========================================================================== #
def test_a_cold_build_stores_one_body_under_the_batch_region(
    store: pathlib.Path,
) -> None:
    """The first call compiles and saves; the file lands in ``<dir>/batch/``.

    The root stays empty: the dense and specialized routes keep the root, so the
    batch route nesting its files is what lets one directory serve every route.
    """
    values, types, sizes = _population()
    _loss(values, types, sizes, _x(), _labels())
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_enabled"] == 1
    assert stats["batch_artifact_files"] == 1
    assert stats["batch_artifact_misses"] == 1, "a cold build loads before it compiles"
    assert stats["batch_artifact_stores"] == 1
    assert stats["batch_artifact_hits"] == 0
    assert stats["batch_artifact_load_failures"] == 0
    assert stats["batch_artifact_save_failures"] == 0
    # the file is in the route's region, not the root
    assert _region(store).is_dir()
    (artifact,) = _files(store)
    assert artifact.parent == _region(store), artifact
    assert artifact.suffix == artifact_cache.SUFFIX
    assert artifact.stat().st_size > 0
    assert list(store.glob(f"*{artifact_cache.SUFFIX}")) == []
    # ... and the name carries the plan's variant, the chunk index and a digest
    plan = _plan(values, types, sizes, _x())
    assert artifact.name.startswith(f"batch-{plan.variant}-c0-"), artifact.name


# =========================================================================== #
# 3. the hit path: a memory clear costs a reload, not a compile
# =========================================================================== #
def test_a_memory_clear_serves_the_next_call_from_disk(
    store: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clear_cache()`` (memory only) -> the next call reloads the body.

    The two lifetimes are the whole point: the in-memory tables are dropped, the
    file stays, and the rebuild is a load.  ``build_bodies`` (the only site that
    lowers / compiles / loads) must not be reached at all -- a build that happened
    to produce the same numbers would still be the bug this tier exists to avoid.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    cold = B.batch_artifact_stats()
    assert cold["batch_artifact_stores"] == 1

    E.clear_cache()  # forget every in-memory table, keep the file
    assert len(_files(store)) == 1, "clear_cache must not delete a file"
    builds = _BuildCounter()
    builds.install(monkeypatch)
    again = _loss(values, types, sizes, inputs, labels)

    assert builds.calls == 0, "the disk hit still compiled a body"
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_hits"] == 1, stats
    assert stats["batch_artifact_misses"] == cold["batch_artifact_misses"]
    assert stats["batch_artifact_stores"] == 1, "a hit must not rewrite the file"
    assert stats["batch_artifact_load_failures"] == 0
    assert torch.equal(first, again), "the reloaded body changed the loss"
    # A disk hit is an executable HIT in the in-memory record, never a compile miss
    # (the tables were zeroed by the clear, so these are this call's counters).
    in_memory = B.batch_cache_stats()
    assert in_memory["batch_exec_hits"] == 1, in_memory
    assert in_memory["batch_exec_misses"] == 0, in_memory
    assert in_memory["batch_exec_entries"] == 1, in_memory


def test_the_same_plan_twice_reuses_the_same_file(
    store: pathlib.Path,
) -> None:
    """The path is content- and plan-derived: nothing about the call is in it."""
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    device = plan.device
    options = batch_build._build_options(plan, {})
    chunk = plan.chunks[0]
    first = batch_artifacts.body_path(plan, chunk, BACKEND, device, options)
    second_plan = _plan(*_population(), _x())  # equal plan, fresh tensors
    second = batch_artifacts.body_path(
        second_plan, second_plan.chunks[0], BACKEND, device, options
    )
    assert first is not None and second is not None
    assert first == second, "two equal plans disagreed on their artifact file"
    assert batch_artifacts.body_name(plan, chunk) == batch_artifacts.body_name(
        second_plan, second_plan.chunks[0]
    )
    assert batch_artifacts.body_name(plan, chunk).startswith("batch:loss:c0:")
    assert first.startswith(str(_region(store))), first


# =========================================================================== #
# 4. the miss path: a different population is a different file
# =========================================================================== #
def test_a_different_population_is_a_new_body_and_a_new_file(
    store: pathlib.Path,
) -> None:
    """The name carries a digest of the merged functions, so content decides.

    Same population size, same shapes, same plan parameters -- only the programs
    differ. A name derived from the plan SHAPE alone would serve one population's
    compiled body to the other here, which is the failure mode the digest prevents.
    """
    values, types, sizes = _population()
    loss_a = _loss(values, types, sizes, _x(), _labels())
    assert len(_files(store)) == 1
    cold = B.batch_artifact_stats()

    E.clear_cache()
    other = _population(_PROGRAMS_B)
    loss_b = _loss(*other, _x(), _labels())
    files = _files(store)
    assert len(files) == 2, [path.name for path in files]
    assert not torch.equal(loss_a, loss_b), "the two populations must differ"
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_misses"] == cold["batch_artifact_misses"] + 1
    assert stats["batch_artifact_stores"] == cold["batch_artifact_stores"] + 1
    assert stats["batch_artifact_hits"] == cold["batch_artifact_hits"]
    assert stats["batch_artifact_files"] == 2
    # The two bodies are distinct plans with distinct digests ...
    plan_a = _plan(*_population(), _x())
    plan_b = _plan(*_population(_PROGRAMS_B), _x())
    name_a = batch_artifacts.body_name(plan_a, plan_a.chunks[0])
    name_b = batch_artifacts.body_name(plan_b, plan_b.chunks[0])
    assert name_a != name_b
    assert batch_artifacts.body_digest(plan_a.functions) != batch_artifacts.body_digest(
        plan_b.functions
    )
    # ... and each population reloads its OWN file after a memory clear (a
    # mixed-up mapping would be a wrong-number bug, so this is asserted on values).
    for programs, expected in ((_PROGRAMS, loss_a), (_PROGRAMS_B, loss_b)):
        E.clear_cache()
        assert torch.equal(
            _loss(*_population(programs), _x(), _labels()), expected
        ), "a reload served another body's numbers"
        assert B.batch_artifact_stats()["batch_artifact_load_failures"] == 0
    final = B.batch_artifact_stats()
    assert final["batch_artifact_hits"] == 2, final
    assert final["batch_artifact_stores"] == 2, "a reload must not rewrite a file"
    assert final["batch_artifact_files"] == 2


# =========================================================================== #
# 5. corruption: one rebuild, never a wrong number
# =========================================================================== #
def test_a_corrupt_file_is_discarded_and_rebuilt(
    store: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``garbage`` in the artifact -> ``load_failures``, a real compile, a rewrite."""
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    expected = _loss(values, types, sizes, inputs, labels)
    (artifact,) = _files(store)
    good_size = artifact.stat().st_size
    artifact.write_text("garbage")
    assert artifact.stat().st_size < good_size

    E.clear_cache()
    builds = _BuildCounter()
    builds.install(monkeypatch)
    rebuilt = _loss(values, types, sizes, inputs, labels)

    assert builds.calls == 1, "a corrupt file must fall back to a real compile"
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_load_failures"] == 1, stats
    assert stats["batch_artifact_stores"] == 2, "the rebuild must rewrite the file"
    assert stats["batch_artifact_hits"] == 0
    assert torch.equal(rebuilt, expected), "the rebuild changed the loss"
    (artifact,) = _files(store)
    assert artifact.stat().st_size == good_size, "the rewritten file is not whole"
    # ... and the rewritten file is usable again.
    E.clear_cache()
    builds.calls = 0
    assert torch.equal(_loss(values, types, sizes, inputs, labels), expected)
    assert builds.calls == 0
    assert B.batch_artifact_stats()["batch_artifact_hits"] == 1


# =========================================================================== #
# 6. key sensitivity: the two things the route adds to the engine's key
# =========================================================================== #
def test_the_path_separates_programs_and_the_dispatch_formation(
    store: pathlib.Path,
) -> None:
    """Different programs -> different file; a re-spelt compile option too.

    The artifact key is the engine's (name, specs, backend, device, frozen
    options): the batch route contributes the CONTENT in the name and the resolved
    ``dispatch_formation`` through the options, so both must reach the path --
    otherwise a body compiled with one codegen setting would be served to a plan
    that asked for the other.
    """
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x())
    device = plan.device
    options = batch_build._build_options(plan, {})
    mine = batch_artifacts.body_path(plan, plan.chunks[0], BACKEND, device, options)
    assert mine is not None

    other = _plan(*_population(_PROGRAMS_B), _x())
    assert (
        batch_artifacts.body_path(other, other.chunks[0], BACKEND, device, options)
        != mine
    ), "a different population shares the compiled body's file"

    aggressive = dataclasses.replace(
        plan, dispatch_formation="aggressive"
    )
    aggressive_options = batch_build._build_options(aggressive, {})
    assert aggressive_options != options
    assert (
        batch_artifacts.body_path(
            aggressive, aggressive.chunks[0], BACKEND, device, aggressive_options
        )
        != mine
    ), "a re-spelt dispatch formation shares the compiled body's file"
    # ... the same plan with the SAME options still answers its own path.
    assert (
        batch_artifacts.body_path(plan, plan.chunks[0], BACKEND, device, options)
        == mine
    )


# =========================================================================== #
# 7. the counters: their own surface, disjoint from every other one
# =========================================================================== #
def test_the_artifact_stats_are_their_own_seven_keys(store: pathlib.Path) -> None:
    """``batch_artifact_stats()`` is exactly seven int keys -- and leaks nowhere.

    The eight-key ``batch_cache_stats()``, the five-key ``batch_content_stats()``
    and the whole-disk ``artifact_cache_stats()`` are pinned surfaces; a shared
    counter would make one of them change meaning.  The comparison is against those
    three OTHER surfaces only (comparing a set with itself proves nothing).
    """
    stats = B.batch_artifact_stats()
    assert set(stats) == _ARTIFACT_STAT_KEYS, sorted(set(stats) ^ _ARTIFACT_STAT_KEYS)
    assert all(isinstance(value, int) for value in stats.values()), stats
    others: Dict[str, Dict[str, Any]] = {
        "batch_cache_stats": B.batch_cache_stats(),
        "batch_content_stats": B.batch_content_stats(),
        "artifact_cache_stats": E.artifact_cache_stats(),
    }
    union = set()
    for name, other in others.items():
        union |= set(other)
        assert not (set(stats) & set(other)), (
            f"a batch_artifact_ key leaked into {name}: "
            f"{sorted(set(stats) & set(other))}"
        )
    leaked = {key for key in union if key.startswith("batch_artifact_")}
    assert not leaked, f"another surface reports this route's keys: {sorted(leaked)}"
    # The values are read off the live objects, so a build really moves them (and
    # moves nothing else: the two keys that describe the tier do not count calls).
    values, types, sizes = _population()
    _loss(values, types, sizes, _x(), _labels())
    moved = B.batch_artifact_stats()
    assert set(moved) == _ARTIFACT_STAT_KEYS
    assert moved["batch_artifact_stores"] == 1
    assert moved["batch_artifact_files"] == 1
    assert moved["batch_artifact_enabled"] == stats["batch_artifact_enabled"] == 1
    assert moved["batch_artifact_save_failures"] == 0


def test_the_files_counter_reads_the_route_region_not_the_directory(
    store: pathlib.Path,
) -> None:
    """``batch_artifact_files`` counts this route's artifacts, not the shared directory."""
    values, types, sizes = _population()
    _loss(values, types, sizes, _x(), _labels())
    foreign = store / f"dense-loss-{'f' * 64}{artifact_cache.SUFFIX}"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("another route's artifact")
    assert B.batch_artifact_stats()["batch_artifact_files"] == 1
    assert E.artifact_cache_stats()["artifact_files"] == 2
    region_files = _files(_region(store))
    assert len(region_files) == 1, [path.name for path in region_files]
    assert B.batch_artifact_stats()["batch_artifact_files"] == len(region_files)


# =========================================================================== #
# 8. the region and the reset semantics
# =========================================================================== #
def test_the_batch_region_is_separate_and_a_reset_keeps_every_file(
    store: pathlib.Path,
) -> None:
    """``batch/`` nests the batch bodies; a reset zeroes counters, deletes nothing.

    A dense-style artifact in the ROOT (where the dense and specialized routes
    write) must survive ``batch_artifact_reset()`` untouched: the reset is the
    route's counter lifetime, not its file lifetime (``clear_artifacts()`` is the
    other one, and it is the engine's).
    """
    values, types, sizes = _population()
    _loss(values, types, sizes, _x(), _labels())
    (body,) = _files(store)
    assert body.parent == _region(store)
    dense_style = store / f"dense-loss-{'0' * 64}{artifact_cache.SUFFIX}"
    dense_style.write_text("a dense route artifact")
    assert sorted(_files(store)) == sorted([body, dense_style])

    B.batch_artifact_reset()
    stats = B.batch_artifact_stats()
    assert stats["batch_artifact_hits"] == 0
    assert stats["batch_artifact_misses"] == 0
    assert stats["batch_artifact_load_failures"] == 0
    assert stats["batch_artifact_stores"] == 0
    assert stats["batch_artifact_save_failures"] == 0
    assert stats["batch_artifact_enabled"] == 1, "the reset must not disable the tier"
    # Counter reset only: both files are still there, and the file counts still
    # distinguish the route's region from the whole directory.
    assert body.is_file() and dense_style.is_file()
    assert stats["batch_artifact_files"] == 1, "only the batch region counts here"
    assert E.artifact_cache_stats()["artifact_files"] == 2, "the whole layer sees both"
    # A memory clear is equally harmless to the files.
    E.clear_cache()
    assert body.is_file() and dense_style.is_file()
    # ... and the tier still works afterwards: the file is a hit, not a rebuild.
    E.clear_cache()
    _loss(values, types, sizes, _x(), _labels())
    assert B.batch_artifact_stats()["batch_artifact_hits"] == 1
