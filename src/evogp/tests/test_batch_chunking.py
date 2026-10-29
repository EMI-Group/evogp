"""The chunked merged compile of the opt-in "batch" route: same numbers, split build.

``src/evogp/tree/batch_engine.py`` merged a whole population into ONE graph, and a
merged compile is linear in the deduped function count -- which turned a fresh
generation into one long compile.  The round splits that compile into CHUNKS
(``EVOGP_BATCH_COMPILE_WORKERS``), one body per contiguous slice of the merged
function axis.  Chunking is a COMPILATION strategy, never a semantics, so this
file pins exactly that:

* **numbers do not move** -- a chunked run is BITWISE identical to the
  single-executable run on the numpy backend for all four shapes the route
  supports: the value variant ``per_row=True``, the value variant on a shared
  ``x``, the loss variant with MSE and the loss variant with MAE;
* **the partition is coherent** -- ``plan.chunks`` is contiguous, near-equal,
  every chunk has at least ``MIN_CHUNK_FUNCTIONS`` functions, the sizes sum to the
  merged function count, ``chunk.offset`` is the global function offset and
  ``plan.row_index`` keeps GLOBAL function indices (a per-chunk local index would
  serve the wrong function's numbers);
* **a different partition never reuses a body** -- the executable table is keyed
  per chunk on that chunk's own function tuple, so building the same population at
  ``W = 2`` and then at ``W = 4`` pays the misses instead of hitting, and a repeat
  at the same ``W`` hits;
* **the split only happens when it is worth it** -- ``W <= 1``, an explicit
  ``bands=`` (the flat spellings included), ``switch_ops=True`` and a population
  below ``MIN_CHUNK_FUNCTIONS`` all keep the single-executable path (``merged``
  populated, one body, the same plan report);
* **a chunk compile failure is NOT a decline** -- ``evoxc.compile`` blowing up on
  one chunk propagates as the original exception (there is nothing safer to fall
  back to), while the ``BatchUnsupported`` declines (threshold, unknown backend, a
  ``per_row``/``x`` shape doubt) are unchanged under chunking;
* **the knob parses** -- ``EVOGP_BATCH_COMPILE_WORKERS`` / the keyword-only
  ``compile_workers=`` argument on ``plan_batch`` / ``evaluate_batch`` /
  ``sr_fitness_batch``, with an explicit argument winning and a negative or
  non-integer value raising ``ValueError`` naming the variable.

Related files (deliberately not duplicated here)
------------------------------------------------
* ``test_batch_stages.py`` owns the stage counters and the ``describe_plan`` /
  ``BatchPlan`` / ``BatchExecutable`` reporting of the chunk geometry;
* ``test_batch_content_key.py`` owns the content level of the plan key;
* ``test_batch_engine.py`` owns the numeric contract of the merged graph itself.

Mechanics
---------
* numpy ONLY (``backend="numpy"`` passed explicitly), CPU tensors, tiny fixtures:
  FOUR distinct programs, ``N = 5`` rows, ``L = 16``.  Chunking on such a fixture
  needs ``MIN_CHUNK_FUNCTIONS`` lowered through the MODULE OBJECT
  (``monkeypatch.setattr(batch_knobs, "MIN_CHUNK_FUNCTIONS", 2)``): the library
  default of 32 is a real "too small to split" answer, not a test inconvenience.
* No wall-clock assertion; the checks are bitwise values, record fields and cache
  counters.
* Every plan/build pair is bracketed with ``evoxc_engine.clear_cache()`` and the
  worker knob is deleted per test, so an ambient value cannot change the split
  under the assertions.
* ``evoxc`` is imported for the one test that has to make ``evoxc.compile`` raise:
  the seam is the real module attribute, exactly as the engine calls it.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_chunking.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Iterator, Sequence, Tuple

import pytest
import torch

import evoxc

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
from evogp.tree import evoxc_engine as E  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: every case is a few millisecond compiles)
# --------------------------------------------------------------------------- #
#: The host backend: the bitwise-equality contract is stated for numpy.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 16
#: Data rows of every shared-``x`` (``per_row=False``) fixture.
N_ROWS = 5
#: FOUR distinct programs: the merged function count every case splits.
_PROGRAMS: Sequence[Any] = (
    te.fn(te.ADD, te.var(0), te.const(0.5)),
    te.fn(te.MUL, te.var(1), te.var(0)),
    te.var(1),
    te.const(-2.25),
)
POP_SIZE = len(_PROGRAMS)

#: The knobs an ambient value could change under the assertions.  The compile
#: worker knob is read through ``getattr`` so this module still COLLECTS while the
#: tree layer that defines it is landing in parallel.
_ENV_KNOBS: Tuple[str, ...] = (
    B.BAND_SIZE_ENV,
    B.BANDS_ENV,
    B.SWITCH_OPS_ENV,
    B.ALIGNMENT_ENV,
    getattr(batch_knobs, "COMPILE_WORKERS_ENV", "EVOGP_BATCH_COMPILE_WORKERS"),
)


class _ChunkCompileBoom(RuntimeError):
    """A backend compile failure -- deliberately NOT a :class:`B.BatchUnsupported`."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh ``(values, types, sizes)`` CPU tensors of ``_PROGRAMS``.

    A fresh call allocates FRESH buffers: the plan key's identity level is keyed
    on the tensors themselves, so a test that wants the PLAN work to run passes a
    new object (the content level is ``test_batch_content_key.py``'s business).
    """
    return te.rows_from_specs(_PROGRAMS, output_len=1, pad_to=ROW_LEN, tail_value=0.0)


def _x(rows: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(rows, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int = N_ROWS, seed: int = 101) -> torch.Tensor:
    """Deterministic ``(rows, 1)`` SR labels from a LOCAL generator."""
    return te.x_random(rows, 1, seed=seed)


def _chunk_emitted_ops(chunk: Any) -> int:
    """``emitted_ops`` of one chunk's merge report.

    A ``BatchChunk`` carries its own merge report; it is read as a mapping when the
    engine stores ``batch_report.as_dict()`` (what the single-chunk plan exposes as
    ``report``) and as an attribute otherwise, so the aggregate-vs-per-chunk
    assertion does not depend on which of the two shapes the record keeps.
    """
    report = chunk.report
    if isinstance(report, dict):
        return int(report["emitted_ops"])
    return int(report.emitted_ops)


def _set_min_chunk_functions(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """Force chunking on the tiny fixtures.

    ``MIN_CHUNK_FUNCTIONS`` is read by ``batch_programs`` THROUGH the ``batch_knobs``
    module object, so patching the module attribute is the documented seam (and the
    only one: the default 32 correctly declines to split a 4-function population).
    """
    monkeypatch.setattr(batch_knobs, "MIN_CHUNK_FUNCTIONS", value)


def _plan(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    *,
    per_row: bool,
    loss: bool = True,
    compile_workers: Any = None,
    **knobs: Any,
) -> Any:
    """``plan_batch`` with the shape grid, the numpy backend and the split knobs.

    ``plan_batch`` is pure host work (decode, dedup, trace, merge per chunk), so the
    geometry checks cost no compile; the same keyword is threaded through
    ``evaluate_batch`` / ``sr_fitness_batch`` below.
    """
    kwargs: Dict[str, Any] = dict(
        per_row=per_row, backend=BACKEND, compile_workers=compile_workers, **knobs
    )
    if loss:
        kwargs.update(variant=B.LOSS_VARIANT, labels=_labels(x.shape[0]))
    return B.plan_batch(values, types, sizes, INPUT_LEN, 1, x, **kwargs)


def _loss(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    use_mse: bool = True,
    *,
    compile_workers: Any = None,
    backend: str = BACKEND,
    **knobs: Any,
) -> torch.Tensor:
    """``sr_fitness_batch`` with the shape grid and the numpy backend pinned."""
    return B.sr_fitness_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        labels,
        use_mse,
        backend=backend,
        compile_workers=compile_workers,
        **knobs,
    )


def _value(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    *,
    per_row: bool,
    compile_workers: Any = None,
    **knobs: Any,
) -> torch.Tensor:
    """``evaluate_batch`` with the shape grid and the numpy backend pinned."""
    return B.evaluate_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=per_row,
        backend=BACKEND,
        compile_workers=compile_workers,
        **knobs,
    )


# --------------------------------------------------------------------------- #
# hermeticity
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the knobs, admit small populations and bracket the front tables.

    The knobs are DELETED (not pinned) so "the default is 4 workers / 32 functions
    per chunk" is what the tests observe; ``MIN_POP`` is SET to ``1`` because the
    library default (64) is a real decline for every fixture here.  Each test sets
    ``MIN_CHUNK_FUNCTIONS`` itself through :func:`_set_min_chunk_functions`.
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
# 1. numbers: chunked == single executable, bit for bit
# =========================================================================== #
@pytest.mark.parametrize("minimum", [2, 1], ids=["two_chunks", "four_chunks"])
@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_chunked_value_matches_the_single_executable_bitwise(
    monkeypatch: pytest.MonkeyPatch, per_row: bool, minimum: int
) -> None:
    """Both value layouts: a chunked run is the single-executable run, bit for bit.

    Chunking must not reassociate anything: each chunk keeps its own graph, its own
    input staging and its own output rows, so the numpy backend has to reproduce
    the one-body result EXACTLY (``torch.equal``, not a tolerance).  The plan is
    re-planned per call (the cache is cleared), so the second call really builds
    the chunked bodies instead of hitting the first plan.
    """
    _set_min_chunk_functions(monkeypatch, minimum)
    values, types, sizes = _population()
    x = _x(POP_SIZE, seed=7) if per_row else _x(N_ROWS, seed=7)

    E.clear_cache()
    single = _value(values, types, sizes, x, per_row=per_row, compile_workers=1)
    E.clear_cache()
    chunked = _value(values, types, sizes, x, per_row=per_row, compile_workers=4)

    info = B.describe_plan(
        _plan(values, types, sizes, x, per_row=per_row, loss=False, compile_workers=4)
    )
    assert info["chunk_count"] > 1, (
        f"the fixture did not chunk (chunk_count={info['chunk_count']}, "
        f"min_functions={minimum})"
    )
    shape = (POP_SIZE, 1) if per_row else (POP_SIZE, N_ROWS, 1)
    assert tuple(chunked.shape) == shape, tuple(chunked.shape)
    assert torch.equal(chunked, single), (
        "a chunked run must reproduce the single executable exactly: max|diff| = "
        f"{float((chunked - single).abs().max()):.3g}"
    )


@pytest.mark.parametrize("minimum", [2, 1], ids=["two_chunks", "four_chunks"])
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_chunked_loss_matches_the_single_executable_bitwise(
    monkeypatch: pytest.MonkeyPatch, use_mse: bool, minimum: int
) -> None:
    """Both loss metrics: the in-graph reduction is per function, so the split is exact.

    The reduction happens INSIDE each function graph, which is what makes a chunked
    loss bitwise identical rather than merely close: nothing is summed across the
    function axis on the host.  ``torch.equal`` pins it for MSE and MAE.
    """
    _set_min_chunk_functions(monkeypatch, minimum)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()

    E.clear_cache()
    single = _loss(
        values, types, sizes, inputs, labels, use_mse, compile_workers=1
    )
    E.clear_cache()
    chunked = _loss(
        values, types, sizes, inputs, labels, use_mse, compile_workers=4
    )

    info = B.describe_plan(
        _plan(values, types, sizes, inputs, per_row=False, compile_workers=4)
    )
    assert info["chunk_count"] > 1, info["chunk_count"]
    assert tuple(chunked.shape) == (POP_SIZE,), tuple(chunked.shape)
    assert torch.equal(chunked, single), (
        "the chunked loss must be bitwise identical: max|diff| = "
        f"{float((chunked - single).abs().max()):.3g}"
    )


# =========================================================================== #
# 2. the partition itself
# =========================================================================== #
@pytest.mark.parametrize(
    "workers, minimum, expected_chunks",
    [(4, 2, 2), (4, 1, 4), (2, 1, 2)],
    ids=["w4_min2", "w4_min1", "w2_min1"],
)
def test_the_chunk_partition_is_contiguous_balanced_and_complete(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    minimum: int,
    expected_chunks: int,
) -> None:
    """``plan.chunks`` tiles the merged function axis: contiguous, balanced, complete.

    The four properties a chunked plan has to satisfy, because each of them
    silently serves wrong numbers when broken:

    * the sizes SUM to the merged function count (a dropped function is a row with
      no answer);
    * the chunks are CONTIGUOUS and ``chunk.offset`` is the global function offset
      (a gap or a shifted offset reads a foreign program's body);
    * ``row_index`` keeps GLOBAL function indices -- with four distinct programs
      the last row's index exceeds the first chunk's size, which a per-chunk local
      index could not;
    * every chunk has at least ``MIN_CHUNK_FUNCTIONS`` functions and the sizes are
      near-equal (a single oversized chunk defeats the point of splitting).

    A chunked plan also drops the single merged graph (``merged is None``,
    ``graphs``/``specs`` empty) and keeps per-chunk merge data instead, with an
    AGGREGATE report whose ``emitted_ops`` is the sum of the chunks'.
    """
    _set_min_chunk_functions(monkeypatch, minimum)
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), per_row=False, compile_workers=workers)
    info = B.describe_plan(plan)

    assert info["chunk_count"] == expected_chunks == len(plan.chunks)
    sizes_list = list(plan.chunk_sizes)
    assert info["chunk_sizes"] == sizes_list
    assert sum(sizes_list) == len(plan.functions) == POP_SIZE
    assert all(size >= minimum for size in sizes_list), (sizes_list, minimum)
    assert max(sizes_list) - min(sizes_list) <= 1, sizes_list

    for index, chunk in enumerate(plan.chunks):
        assert chunk.offset == sum(sizes_list[:index]), (index, chunk.offset)
        assert chunk.report is not None
        assert chunk.band_count >= 1

    # global function indices, never per-chunk ones
    assert min(plan.row_index) == 0
    assert max(plan.row_index) == POP_SIZE - 1
    assert max(plan.row_index) > sizes_list[0], (
        "the row mapping must address functions beyond the first chunk"
    )
    assert sorted(set(plan.row_index)) == list(range(POP_SIZE))

    # the single merged graph is gone; the aggregate report replaces it
    assert plan.merged is None
    assert plan.graphs == ()
    assert plan.specs == ()
    assert plan.band_count == 1
    assert info["specs"] == []
    assert info["band_count"] == 1
    assert info["band_strategy"] is None
    assert not info["band_sizes"], info["band_sizes"]
    assert info["n_functions"] == POP_SIZE
    assert info["emitted_ops"] == sum(
        _chunk_emitted_ops(chunk) for chunk in plan.chunks
    ), "the aggregate report must sum the chunks' emitted ops"


# =========================================================================== #
# 3. the executable table is keyed per chunk
# =========================================================================== #
def test_a_different_worker_count_never_reuses_an_executable_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-partition is a MISS per chunk; only the same partition hits.

    ``BATCH_EXEC_MEMO`` keys a compiled body on that chunk's own function tuple, so
    ``W = 2`` (chunks ``[f0,f1]``/``[f2,f3]``) and ``W = 4`` (``[f0]``/``[f1]``/...)
    share nothing: the second build pays four misses instead of two hits.  A repeat
    at the same ``W`` on an equal population then hits every chunk -- the compile
    really is shared, and the counters say which side of the line a call is on.
    """
    _set_min_chunk_functions(monkeypatch, 1)
    inputs, labels = _x(), _labels()

    E.clear_cache()
    values, types, sizes = _population()
    B.build_batch(
        _plan(values, types, sizes, inputs, per_row=False, compile_workers=2)
    )
    after_two = B.batch_cache_stats()
    assert after_two["batch_exec_misses"] == 2, after_two
    assert after_two["batch_exec_hits"] == 0, after_two

    values, types, sizes = _population()  # equal content, fresh identity tokens
    plan_four = _plan(values, types, sizes, inputs, per_row=False, compile_workers=4)
    assert len(plan_four.chunks) == 4, (
        "the caller's worker count must reach the partition: a served plan would "
        f"report {len(plan_four.chunks)} chunks (the previous call asked for 2)"
    )
    B.build_batch(plan_four)
    after_four = B.batch_cache_stats()
    assert after_four["batch_exec_misses"] == 2 + 4, (
        "a different partition must pay one miss per chunk, not hit a body"
    )
    assert after_four["batch_exec_hits"] == 0, after_four

    values, types, sizes = _population()
    B.build_batch(
        _plan(values, types, sizes, inputs, per_row=False, compile_workers=4)
    )
    after_repeat = B.batch_cache_stats()
    assert after_repeat["batch_exec_misses"] == 6, after_repeat
    assert after_repeat["batch_exec_hits"] == 4, (
        "a repeat at the same worker count must hit every chunk's body"
    )
    assert after_repeat["batch_exec_entries"] == 6

    # the numbers never depend on the partition (bitwise, numpy backend)
    E.clear_cache()
    two = _loss(values, types, sizes, inputs, labels, True, compile_workers=2)
    E.clear_cache()
    four = _loss(values, types, sizes, inputs, labels, True, compile_workers=4)
    assert torch.equal(two, four)


# =========================================================================== #
# 4. the knob: EVOGP_BATCH_COMPILE_WORKERS
# =========================================================================== #
def test_the_compile_worker_constants_are_the_documented_ones() -> None:
    """The knob's name, its default and the chunk floor are public constants.

    They are the seam a wiring task and an operator read, so each is spelled once
    here; the autouse fixture clears the variable through the same constant.
    """
    assert batch_knobs.COMPILE_WORKERS_ENV == "EVOGP_BATCH_COMPILE_WORKERS"
    assert batch_knobs.DEFAULT_COMPILE_WORKERS == 4
    assert batch_knobs.MIN_CHUNK_FUNCTIONS == 32


@pytest.mark.parametrize("value", [0, 1, 4, 8])
def test_the_worker_count_parses_from_the_argument_and_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: int
) -> None:
    """``0``/``1``/``4``/``8`` resolve to themselves, from either source.

    ``0`` and ``1`` both mean "no chunking", but neither is normalized away in the
    RESOLVER: the plan records what the caller asked for, and the chunking decision
    compares ``<= 1``.  An explicit argument beats the environment, which is what
    lets a runner pin the split for one call.
    """
    assert batch_knobs.resolve_compile_workers(value) == value
    monkeypatch.setenv(batch_knobs.COMPILE_WORKERS_ENV, str(value))
    assert batch_knobs.resolve_compile_workers() == value
    assert batch_knobs.resolve_compile_workers(None) == value
    other = 0 if value != 0 else 8
    assert batch_knobs.resolve_compile_workers(other) == other, (
        "the explicit argument must win over the environment"
    )


def test_the_worker_count_defaults_to_four_when_nothing_is_set() -> None:
    """Unset/empty means ``DEFAULT_COMPILE_WORKERS`` (4), not "chunking on".

    The default is a WORKER count: whether a population is actually split still
    depends on ``MIN_CHUNK_FUNCTIONS``, so the resolver never encodes the floor.
    """
    assert batch_knobs.resolve_compile_workers() == batch_knobs.DEFAULT_COMPILE_WORKERS
    assert batch_knobs.resolve_compile_workers(None) == 4


@pytest.mark.parametrize("raw", ["-1", "four", "2.5"])
def test_a_malformed_worker_count_is_a_value_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A negative or non-integer worker count is the CALLER's mistake, not a decline.

    A typo must fail loudly (``ValueError`` naming the variable) instead of being
    silently clamped to one worker or converted into a ``BatchUnsupported`` a call
    site would swallow by falling back.
    """
    monkeypatch.setenv(batch_knobs.COMPILE_WORKERS_ENV, raw)
    with pytest.raises(ValueError) as excinfo:
        batch_knobs.resolve_compile_workers()
    assert batch_knobs.COMPILE_WORKERS_ENV in str(excinfo.value), str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        batch_knobs.resolve_compile_workers(raw)
    assert batch_knobs.COMPILE_WORKERS_ENV in str(excinfo.value), str(excinfo.value)
    assert not isinstance(excinfo.value, B.BatchUnsupported)


# =========================================================================== #
# 5. when the split is forced OFF
# =========================================================================== #
@pytest.mark.parametrize(
    "knobs, minimum",
    [
        ({"compile_workers": 1}, 1),
        ({"compile_workers": 0}, 1),
        ({"compile_workers": 4, "bands": 2}, 1),
        ({"compile_workers": 4, "bands": "flat"}, 1),
        ({"compile_workers": 4, "switch_ops": True}, 1),
        ({"compile_workers": 4}, 32),
    ],
    ids=[
        "one_worker",
        "zero_workers",
        "explicit_bands",
        "flat_bands",
        "switch_ops",
        "below_the_chunk_floor",
    ],
)
def test_the_single_executable_path_survives_every_chunking_killer(
    monkeypatch: pytest.MonkeyPatch, knobs: Dict[str, Any], minimum: int
) -> None:
    """Four killers, one contract: the plan is the single-executable plan.

    ``W <= 1`` (one graph, one body), an explicit ``bands=`` (the flat spellings
    included, because a banded or op-switched merge already splits the body), and a
    population below ``MIN_CHUNK_FUNCTIONS`` all keep the pre-round shape: ``merged``
    populated, ``graphs``/``specs`` present, ``chunk_count == 1`` and exactly one
    compiled body.  A regression here would silently change every small
    population's compile economics.
    """
    _set_min_chunk_functions(monkeypatch, minimum)
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), per_row=False, **knobs)
    info = B.describe_plan(plan)

    assert info["chunk_count"] == 1, (knobs, info["chunk_count"])
    assert len(plan.chunks) == 1
    assert plan.merged is not None
    assert len(plan.graphs) == len(plan.functions)
    assert plan.chunk_sizes == (len(plan.functions),)
    assert info["chunk_sizes"] == [len(plan.functions)]

    entry = B.build_batch(plan)
    assert entry.executable is not None
    assert len(entry.executables) == 1
    assert entry.executable is entry.executables[0]
    assert "compile#0" in entry.build_stage_ms, sorted(entry.build_stage_ms)


# =========================================================================== #
# 6. failures and declines
# =========================================================================== #
def test_a_chunk_compile_failure_propagates_as_the_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One chunk's backend failure is NOT converted into a decline.

    ``BatchUnsupported`` means "fall back to the specialized engine", and a
    failing backend is not that: falling back would hide a broken compiler behind a
    slower route.  The second chunk's ``evoxc.compile`` raises, and the caller sees
    that very exception.
    """
    _set_min_chunk_functions(monkeypatch, 1)
    boom = _ChunkCompileBoom("chunk 1 exploded")
    calls = {"n": 0}
    original = evoxc.compile

    def failing_compile(lowered: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise boom
        return original(lowered, *args, **kwargs)

    monkeypatch.setattr(evoxc, "compile", failing_compile)
    values, types, sizes = _population()
    plan = _plan(values, types, sizes, _x(), per_row=False, compile_workers=4)
    assert B.describe_plan(plan)["chunk_count"] == 4, "the fixture must really chunk"

    with pytest.raises(_ChunkCompileBoom) as excinfo:
        B.build_batch(plan)
    assert excinfo.value is boom, "the chunk compile failure must not be wrapped"
    assert not isinstance(excinfo.value, B.BatchUnsupported)
    assert calls["n"] >= 2, calls


def test_the_declines_are_unchanged_under_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declines keep their meaning -- and a malformed knob stays a ``ValueError``.

    Chunking changes WHERE the compile happens, not WHAT the route refuses: an
    unknown backend name, a ``per_row``/``x`` shape doubt and a population below
    ``EVOGP_BATCH_MIN_POP`` are still ``BatchUnsupported`` (the caller falls back),
    while a bad worker count is still the caller's ``ValueError``.  The order
    matters too: the worker knob is validated by the resolver, not turned into a
    decline.
    """
    _set_min_chunk_functions(monkeypatch, 1)
    values, types, sizes = _population()
    inputs, labels = _x(), _labels()

    # a malformed worker count: the caller's mistake, raised (MIN_POP=1 admits the
    # population, so nothing else can decline first)
    with pytest.raises(ValueError) as excinfo:
        _loss(values, types, sizes, inputs, labels, True, compile_workers=-1)
    assert batch_knobs.COMPILE_WORKERS_ENV in str(excinfo.value), str(excinfo.value)
    assert not isinstance(excinfo.value, B.BatchUnsupported)

    # the threshold still declines with the library default (64)
    monkeypatch.delenv(B.MIN_POP_ENV, raising=False)
    with pytest.raises(B.BatchUnsupported) as excinfo:
        _loss(values, types, sizes, inputs, labels, True, compile_workers=4)
    assert B.MIN_POP_ENV in str(excinfo.value), str(excinfo.value)
    monkeypatch.setenv(B.MIN_POP_ENV, "1")

    # an unknown backend name is still "this route does not build that"
    with pytest.raises(B.BatchUnsupported):
        _loss(
            values, types, sizes, inputs, labels, True, compile_workers=4, backend="tpu"
        )

    # a per_row/x shape doubt is still a decline, never the specialized engine's
    # malformed-tree error
    with pytest.raises(B.BatchUnsupported) as excinfo:
        _value(
            values, types, sizes, _x(N_ROWS), per_row=True, compile_workers=4
        )
    assert "per_row" in str(excinfo.value), str(excinfo.value)
