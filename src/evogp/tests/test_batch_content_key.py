"""The content level of the "batch" route's plan key (a rebuilt-equal population hits).

``BATCH_PLAN_MEMO`` used to be keyed on the population's three IDENTITY tokens only
(``data_ptr``/shape/dtype/device/``_version``), so any caller that re-allocated an
equal population -- a runner rebuilding a ``Forest``, a checkpoint replay, a
``torch.cat`` of the same rows -- paid the whole plan again (decode + one trace per
distinct program + the merge) and could not reuse the compiled body.  The round
makes the key two-level:

1. **identity** first, unchanged -- a repeat call on the very same tensors still
   hits ``batch_plan_hits`` and never re-decodes (and never computes a digest);
2. then a **content digest** over the decoded programs (plus the per-row ``x``
   bytes when ``per_row``), with the stored ``plan.programs`` VERIFIED on a content
   hit: a digest collision must never serve another population's row mapping.

This file pins exactly that, because every one of the four failure modes is silent:

* **a rebuilt-equal population HITS the content level** -- ``batch_content_hits``
  grows, ``_plan_host`` is not called again and the front table of executables is
  untouched (the compile is not re-paid), and the numbers are identical;
* **an in-place mutation and a genuinely different population MISS** -- torch bumps
  ``_version`` on an in-place write, so the identity level cannot serve a
  half-evolved population, and the digest differs, so the content level cannot
  either;
* **a forced digest collision is REJECTED** -- the stored programs are compared
  with the newly decoded ones, so population B gets B's numbers (never A's), and
  ``batch_content_rejects`` counts the refusal;
* **the table is empty after ``evoxc_engine.clear_cache()``** -- the engine stays
  the single in-memory invalidation point of the whole route.

Surface pinned here: ``batch_engine.batch_content_stats()`` is EXACTLY five int
counters (``batch_content_{hits,misses,entries,maxsize,rejects}``),
``batch_engine.BATCH_CONTENT_MEMO`` is the table (with ``.maxsize``), and none of
that appears in ``batch_cache_stats()`` (still its 8 keys), ``cache_stats()``,
``dense_cache_stats()``, ``artifact_cache_stats()`` or ``batch_stage_stats()``.

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_knobs.py`` owns the identity-level counters and the mode-independent
  knob surface; ``test_batch_stages.py`` owns the stage counters and the chunk
  geometry; ``test_batch_chunking.py`` owns the split itself:
* ``test_batch_engine.py`` owns the merged graph's numeric contract.

Mechanics
---------
* numpy ONLY (``backend="numpy"``), CPU tensors, the usual tiny fixture (FOUR
  distinct programs, ``N = 5`` rows, ``L = 16``), admitted through
  ``EVOGP_BATCH_MIN_POP=1``; the front tables are bracketed with
  ``evoxc_engine.clear_cache()`` and the knobs are deleted per test.
* No wall-clock assertion: the checks are counters, ``torch.equal`` and the
  independent oracle (``test_equivalence.reference_sr_fitness`` over a ``Forest``
  built from the rows under test -- ``reference_eval`` has no rows-level
  re-export there).
* The digest helper is located through a small candidate list at CALL time (the
  tree layer that defines it is landing in parallel): a module-level reference
  would break this whole file's collection instead of failing one test.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_content_key.py -q -p no:cacheprovider
"""

from __future__ import annotations

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

from evogp.tree import Forest  # noqa: E402
from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_knobs  # noqa: E402
from evogp.tree import batch_programs  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: a handful of numpy compiles, no device)
# --------------------------------------------------------------------------- #
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
N_ROWS = 5
#: FOUR distinct programs: three computed ones and one fully constant tree.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
#: A DIFFERENT population of the SAME size and shape: nothing about it may be
#: served from the first one's entry (that is what a collided digest would do).
_OTHER_PROGRAMS: Sequence[Any] = (
    te.fn(te.SUB, te.var(0), te.const(1.25)),
    te.fn(te.DIV, te.var(1), te.const(2.0)),
    te.var(0),
    te.const(4.5),
)
POP_SIZE = len(_PROGRAMS)

#: The exact key set of ``batch_content_stats()`` -- the round's fifth stats
#: surface, and the only one that describes the plan key's content level.
_CONTENT_STAT_KEYS = frozenset(
    {
        "batch_content_hits",
        "batch_content_misses",
        "batch_content_entries",
        "batch_content_maxsize",
        "batch_content_rejects",
    }
)
#: ``batch_cache_stats()``'s documented 8 keys (proved disjoint from the content
#: table here as well as in the knob file).
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
#: ``cache_stats()``'s 8 engine keys.
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

#: The likely spellings of the content-digest helper, most specific first.  The
#: collision test needs to replace it with a CONSTANT, and the tree layer that
#: defines it lands in parallel, so the name is discovered at call time instead of
#: being imported here (which would break this file's collection).
_DIGEST_NAME_CANDIDATES: Tuple[str, ...] = (
    "_content_digest",
    "_program_digest",
    "_plan_digest",
    "_content_key_digest",
    "_digest_programs",
    "_digest",
)

#: The knobs an ambient value could change under the assertions (the compile
#: worker knob included; read through ``getattr`` so collection stays possible).
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
class _Counter:
    """Wrap a module-level callable and count its invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> None:
        original = getattr(module, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", name)
        monkeypatch.setattr(module, name, wrapper)


def _plan_host_counter(monkeypatch: pytest.MonkeyPatch) -> _Counter:
    """Count ``_plan_host`` (the whole host planning step) wherever it is reachable.

    ``_plan_host`` is ``decode_population`` + dedup + one trace per program + the
    merge, i.e. everything a plan hit must SKIP.  It is patched on every module that
    exposes it, so the counter is correct whether the memoized entry point lives in
    ``batch_engine`` or in ``batch_programs``.
    """
    counter = _Counter()
    for module in (B, batch_programs):
        if callable(getattr(module, "_plan_host", None)):
            counter.install(monkeypatch, module, "_plan_host")
    return counter


def _digest_seam() -> Tuple[str, List[Any]]:
    """``(name, modules)`` of the content-digest helper, for the collision test.

    Fails with an explicit diagnostic (never a skip) when no helper can be found:
    the collision-safety contract is untestable without a seam, and silently
    dropping the test would leave the most dangerous failure mode -- serving
    population A's numbers to population B -- unguarded.
    """
    for name in _DIGEST_NAME_CANDIDATES:
        modules = [
            module for module in (B, batch_programs)
            if callable(getattr(module, name, None))
        ]
        if modules:
            return name, modules
    discovered = sorted(
        {
            name
            for module in (B, batch_programs)
            for name, value in vars(module).items()
            if callable(value) and "digest" in name.lower()
        }
    )
    if len(discovered) == 1:
        name = discovered[0]
        return name, [
            module for module in (B, batch_programs)
            if callable(getattr(module, name, None))
        ]
    pytest.fail(
        "no content-digest helper found on evogp.tree.batch_engine / "
        "evogp.tree.batch_programs (candidates checked: "
        f"{list(_DIGEST_NAME_CANDIDATES)}, 'digest' matches: {discovered}); the "
        "collision-safety test needs a seam on the digest function"
    )


def _population(
    specs: Sequence[Any] = _PROGRAMS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors for ``specs``."""
    return te.rows_from_specs(specs, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


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
    use_mse: bool = True,
) -> torch.Tensor:
    """``sr_fitness_batch`` with the shape grid and the numpy backend pinned."""
    return B.sr_fitness_batch(
        values, types, sizes, INPUT_LEN, 1, inputs, labels, use_mse, backend=BACKEND
    )


def _content_stats() -> Dict[str, int]:
    """The content-level counters of the plan key."""
    return B.batch_content_stats()


def _oracle_forest(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
) -> Forest:
    """A CPU ``Forest`` of exactly the rows under test (the oracle needs the container).

    Built from the SAME tensors the engine is handed (never regenerated from the
    specs), so a mismatch can only come from the engine.
    """
    return Forest(INPUT_LEN, 1, values, types, sizes)


def _oracle(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    use_mse: bool = True,
) -> torch.Tensor:
    """The independent plain-torch loss (``reference_eval``) of the same rows.

    The repo's oracle recipe is the container-based one (``test_batch_engine.py``'s
    ``_forest`` + ``te.reference_sr_fitness``); ``test_equivalence`` has no
    rows-level ``sr_fitness_from_rows`` re-export.
    """
    return te.reference_sr_fitness(
        _oracle_forest(values, types, sizes), inputs, labels, use_mse
    )


# --------------------------------------------------------------------------- #
# hermeticity
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the knobs, admit small populations and empty the front tables.

    ``evoxc_engine.clear_cache()`` is the route's single invalidation point and
    drops the content table through the registered hook, so every test starts from
    an empty table with zeroed counters (one test pins exactly that).
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
# 1. the surface: five counters, one table
# =========================================================================== #
def test_the_content_table_reports_exactly_the_documented_keys() -> None:
    """``batch_content_stats()`` is EXACTLY five int counters of one table.

    The key set is a contract (a caller reads deltas), and ``maxsize`` has to come
    from the table itself: ``EVOGP_BATCH_CACHE_SIZE`` is read at IMPORT time, so the
    reported bound must be the one the live table was built with.
    """
    stats = _content_stats()
    assert set(stats) == set(_CONTENT_STAT_KEYS), sorted(
        set(stats) ^ set(_CONTENT_STAT_KEYS)
    )
    assert all(isinstance(value, int) for value in stats.values()), stats
    assert stats["batch_content_entries"] == 0, "the fixture starts empty"
    assert stats["batch_content_maxsize"] == B.BATCH_CONTENT_MEMO.maxsize
    assert B.BATCH_CONTENT_MEMO.maxsize >= 0
    assert stats["batch_content_hits"] == 0 and stats["batch_content_misses"] == 0
    assert stats["batch_content_rejects"] == 0


def test_the_content_table_is_disjoint_from_every_other_stats_surface() -> None:
    """Five surfaces, five disjoint key sets -- and the content table is the newest.

    ``batch_cache_stats()`` stays exactly its 8 keys (the content level is a
    separate measurement), and no ``batch_content_*`` key may appear in
    ``cache_stats()``, ``dense_cache_stats()``, ``artifact_cache_stats()`` or
    ``batch_stage_stats()``.
    """
    content_keys = set(_content_stats())
    assert content_keys == set(_CONTENT_STAT_KEYS)
    surfaces = {
        "batch_cache_stats": set(B.batch_cache_stats()),
        "cache_stats": set(E.cache_stats()),
        "dense_cache_stats": set(D.dense_cache_stats()),
        "artifact_cache_stats": set(E.artifact_cache_stats()),
        "batch_stage_stats": set(B.batch_stage_stats()),
    }
    assert surfaces["batch_cache_stats"] == set(_CACHE_STAT_KEYS)
    assert surfaces["cache_stats"] == set(_ENGINE_STAT_KEYS)
    assert surfaces["dense_cache_stats"], "the dense route registers its own tables"
    for name, keys in surfaces.items():
        assert content_keys.isdisjoint(keys), (name, sorted(content_keys & keys))
        leaked = sorted(key for key in keys if key.startswith("batch_content_"))
        assert not leaked, f"the content table leaked into {name}: {leaked}"


# =========================================================================== #
# 2. level 1: identical tensors never even compute a digest
# =========================================================================== #
def test_identical_tensors_short_circuit_at_the_identity_level() -> None:
    """A repeat call on the SAME tensors is still a plan hit, not a content hit.

    The identity level is checked first and returns the stored entry, so the
    digest is never computed for an unchanged population: ``batch_content_hits``
    stays ``0`` while ``batch_plan_hits`` grows, and the compile is not touched.
    Making the content level the primary key would turn every warm call into a
    hash of the whole population -- a silent regression this pins.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()

    first = _loss(values, types, sizes, inputs, labels)
    after_first = _content_stats()
    assert after_first["batch_content_misses"] == 1
    assert after_first["batch_content_entries"] == 1
    assert after_first["batch_content_hits"] == 0
    cache_after_first = B.batch_cache_stats()
    assert cache_after_first["batch_plan_misses"] == 1
    assert cache_after_first["batch_exec_misses"] == 1

    for _ in range(2):
        again = _loss(values, types, sizes, inputs, labels)
        assert torch.equal(again, first)

    after = _content_stats()
    assert after["batch_content_hits"] == 0, (
        "an identity hit must return before the digest is computed"
    )
    assert after["batch_content_misses"] == 1
    assert after["batch_content_entries"] == 1
    cache = B.batch_cache_stats()
    assert cache["batch_plan_hits"] == 2
    assert cache["batch_plan_misses"] == 1
    assert cache["batch_exec_hits"] == cache_after_first["batch_exec_hits"]
    assert cache["batch_exec_misses"] == cache_after_first["batch_exec_misses"]


# =========================================================================== #
# 3. level 2: a rebuilt-equal population hits
# =========================================================================== #
def test_a_rebuilt_equal_population_hits_the_content_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same content in new buffers: a CONTENT hit, no re-plan, no re-compile.

    This is the round's headline behaviour: ``_population()`` allocates fresh
    tensors, so the identity level is a miss -- but the decoded programs are equal,
    so the content level serves the stored plan, ``_plan_host`` (decode + dedup +
    trace + merge) is never called again, the executable table is untouched and the
    re-bound plan answers with the same numbers.
    """
    plans = _plan_host_counter(monkeypatch)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1
    exec_after_first = B.batch_cache_stats()
    assert exec_after_first["batch_exec_misses"] == 1

    copied = _population()
    assert torch.equal(values, copied[0]), "the fixture copy must have equal content"
    second = _loss(*copied, inputs, labels)

    assert plans.calls == 1, (
        "a content hit must not decode / dedup / trace / merge again"
    )
    stats = _content_stats()
    assert stats["batch_content_hits"] == 1, stats
    assert stats["batch_content_misses"] == 1
    assert stats["batch_content_entries"] == 1
    cache = B.batch_cache_stats()
    assert cache["batch_plan_misses"] == 2, "the identity level is still a miss"
    assert cache["batch_plan_hits"] == 0
    assert cache["batch_exec_misses"] == exec_after_first["batch_exec_misses"], (
        "a content hit reuses the plan AND its executable: the compile is not re-paid"
    )
    assert cache["batch_exec_hits"] == exec_after_first["batch_exec_hits"]
    assert torch.equal(first, second), "the content hit serves the same numbers"


# =========================================================================== #
# 4. level 2: mutations and different populations miss
# =========================================================================== #
def test_an_in_place_mutation_is_a_content_miss_and_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-place write bumps ``_version``: neither level may serve the old entry.

    A half-evolved population is the normal state of a GP loop between a genome
    edit and the next evaluation, so this is the case that must never hit.  The
    loss also has to be the MUTATED population's: the fixture edits the constant
    root of the last row, which changes exactly that row.
    """
    plans = _plan_host_counter(monkeypatch)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1

    values[3, 0] = -2.5  # an in-place torch write: the version counter moves
    mutated = _loss(values, types, sizes, inputs, labels)

    assert plans.calls == 2, "an in-place mutation must re-plan"
    stats = _content_stats()
    assert stats["batch_content_misses"] == 2, stats
    assert stats["batch_content_hits"] == 0, stats
    assert not torch.equal(mutated, first), (
        "the mutated constant must change the loss (the fixture must be sensitive)"
    )
    te.assert_close_with_placement(
        mutated,
        _oracle(values, types, sizes, inputs, labels),
        context="the mutated population's loss",
    )


def test_a_different_population_is_a_content_miss_and_answers_for_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different population of the same size shares nothing with the first.

    Same ``pop_size``, same ``N``, same ``input_len`` -- so only the CONTENT can
    tell them apart, and the answer must be the second population's own (the
    oracle's) numbers, never the first one's.
    """
    plans = _plan_host_counter(monkeypatch)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1

    other = _population(_OTHER_PROGRAMS)
    second = _loss(*other, inputs, labels)

    assert plans.calls == 2, "a different population is a miss at both levels"
    stats = _content_stats()
    assert stats["batch_content_misses"] == 2, stats
    assert stats["batch_content_hits"] == 0, stats
    assert not torch.equal(first, second), "the fixtures must not be numerically equal"
    te.assert_close_with_placement(
        second,
        _oracle(*other, inputs, labels),
        context="the second population's loss",
    )


# =========================================================================== #
# 5. collision safety: a digest hit is VERIFIED
# =========================================================================== #
def test_a_digest_collision_never_serves_another_populations_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced digest collision must rebuild, and must answer for the CURRENT rows.

    The digest is replaced by a constant, so population B's content key is A's.
    The route therefore has to compare the stored ``plan.programs`` with the freshly
    decoded ones before serving the entry: a collision that is served would return
    A's row mapping -- wrong numbers, no exception, the worst possible failure.  It
    must either count the refusal (``batch_content_rejects``) or rebuild the plan.
    """
    name, modules = _digest_seam()

    def constant_digest(*args: Any, **kwargs: Any) -> str:
        return "collision"

    for module in modules:
        monkeypatch.setattr(module, name, constant_digest)

    plans = _plan_host_counter(monkeypatch)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    loss_a = _loss(values, types, sizes, inputs, labels)
    assert plans.calls == 1

    other = _population(_OTHER_PROGRAMS)
    loss_b = _loss(*other, inputs, labels)

    assert plans.calls == 2, (
        f"the collided content ({name} -> constant) was served instead of re-planned"
    )
    assert not torch.equal(loss_b, loss_a), (
        "population B received population A's numbers -- the digest collision was "
        "served"
    )
    te.assert_close_with_placement(
        loss_b,
        _oracle(*other, inputs, labels),
        context="a collided content key must still answer for the current rows",
    )
    stats = _content_stats()
    assert stats["batch_content_hits"] == 0, (
        "a collided content key that was then REBUILT must not be counted as a hit "
        f"-- that would report a served entry that was never served: {stats}"
    )
    assert stats["batch_content_rejects"] >= 1 or stats["batch_content_misses"] >= 2, (
        "a hard digest mismatch (the table found the key, the stored programs did "
        "not match) must be visible on the content surface -- either as a reject or "
        f"as a second miss, never as silence: {stats}"
    )


# =========================================================================== #
# 6. invalidation
# =========================================================================== #
def test_clear_cache_drops_the_content_table_and_repays_the_plan() -> None:
    """``evoxc_engine.clear_cache()`` empties the content table too.

    The engine is the route's single in-memory invalidation point; leaving the
    content table populated would let a "clear the cache after changing a compile
    knob" call serve a plan built with the OLD knobs.
    """
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()
    first = _loss(values, types, sizes, inputs, labels)
    assert _content_stats()["batch_content_entries"] == 1

    E.clear_cache()

    cleared = _content_stats()
    assert cleared["batch_content_entries"] == 0, cleared
    assert cleared["batch_content_hits"] == 0, cleared
    assert cleared["batch_content_misses"] == 0, cleared
    assert cleared["batch_content_rejects"] == 0, cleared
    assert cleared["batch_content_maxsize"] == B.BATCH_CONTENT_MEMO.maxsize

    again = _loss(values, types, sizes, inputs, labels)
    after = _content_stats()
    assert after["batch_content_misses"] == 1, "the plan must be re-paid after a clear"
    assert after["batch_content_hits"] == 0
    assert after["batch_content_entries"] == 1
    assert B.batch_cache_stats()["batch_plan_misses"] == 1
    assert B.batch_cache_stats()["batch_exec_misses"] == 1
    assert torch.equal(first, again)
