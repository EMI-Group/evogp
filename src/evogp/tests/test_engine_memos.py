"""Regression tests for the engine's two in-process memos (count-observable only).

The engine round that landed under ``evogp/tree/`` added two memos. Both are
observable through call counters, which is what this file pins -- there is **no
wall-clock assertion anywhere** (the hit-cost/throughput half lives in the dev
batteries, not in pytest).

* **Payload memo** -- ``dense_engine._iter_plan_chunks`` prepares and uploads
  each distinct chunk payload exactly ONCE per call, through the
  ``(host_array, device_tensor_or_None)`` entry behind its nested ``payload``
  closure (or the caller-injected ``staged`` dict).  The ``p0``-outer/``n0``-inner
  loop would otherwise re-prepare and re-upload the ``p0``-keyed node columns
  once per ``(p0, n0)`` chunk -- measured as 296 host preparations instead of the
  distinct payload count, for an 8x37 population in 1x1 chunks.
* **Executable memo** -- ``etl_engine._EXECUTABLE_MEMO`` (an ``_ExecutableMemo``
  LRU in front of ``etl_engine.CACHE``) absorbs repeated / duplicate / elite
  ``_executable`` lookups, so ``etl.build_cached`` (which re-derives its cache key
  on *every* call, hits included) runs once per distinct
  ``(program, shape, backend, device, options)``.

The dev batteries are the semantics of record and these tests port their checks
rather than re-inventing them: stage ``seam`` of ``evogp/tree/device_resident_check.py``
(payload memo) and ``evogp/tree/executable_memo_check.py`` (executable memo,
``CountingSection`` / ``_no_memo_executable`` / ``_distinct_programs``).

How the counts are observed:

* payload PREPARATIONS -- counting pass-through wrappers on the module globals the
  seam resolves at call time: ``dense_engine._prep_column`` (the 7 node columns,
  keyed per ``p0``), ``dense_engine._chunk_rows`` (``x``, plus ``labels`` in the
  loss variant, keyed per ``n0``) and ``dense_engine._keep_mask`` (per ``n0``);
* payload UPLOADS -- the numpy backend never uploads (host interpreter), so the
  "preparations == uploads" equality is pinned with a *synthetic* device backend:
  ``etl.from_numpy`` is monkeypatched to a tiny fake whose ``.to(device)`` returns
  the host array.  The executable was built on numpy and ``etl.run`` therefore
  still receives host arrays, so nothing is compiled or executed on iree -- only
  the seam's ``to_device`` branch is exercised;
* ``etl.run`` invocations -- exactly one per chunk (the chunk product);
* executable BUILDS -- a counting wrapper on the module global
  ``etl_engine.build_cached`` (``_executable`` calls it as a module global).

Hermeticity: the two memos are process-global, so every memo test defines its own
baseline with ``etl_engine.clear_cache()`` and only asserts counter *deltas* it
established itself.  Shapes are deliberately tiny (P=8, L=16, N=12) and the engine
backend is pinned to ``"numpy"``, so nothing leaves the host;
``Forest.SR_fitness`` is the only call that needs CUDA (``check_tensor`` relocates
its inputs) and that one test skips without a GPU.  The direct ``etl_engine`` calls
are always the specialized route; the file passes with ``EVOGP_EVAL_MODE`` unset
and with ``EVOGP_EVAL_MODE=dense`` (the ``Forest``-level differential compares
three calls made in the same mode).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch

import etl

from evogp.tree import Forest
from evogp.tree import dense_engine as D
from evogp.tree import etl_engine as E

# --------------------------------------------------------------------------- #
# The suite's tree-spec DSL (var/const/fn -> prefix rows) is reused, not copied
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import test_equivalence as _eq
except ImportError:  # pragma: no cover - only under pytest's importlib import mode
    _spec = importlib.util.spec_from_file_location("evogp_memo_helpers", _HERE / "test_equivalence.py")
    _eq = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eq)

# --------------------------------------------------------------------------- #
# Fixture shapes (small on purpose: the counts, not the size, are the signal)
# --------------------------------------------------------------------------- #
INPUT_LEN = 3
POP = 8
ROW_LEN = 16  # the population rows' width == the dense trace length L here
#: Data points per chunked call.  Small on purpose: one ``etl.run`` costs the same
#: ~28 ms on the numpy interpreter whether its chunk block is 1x1 or 8x37, so the
#: run COUNT is what the runtime scales with.  The memo regression was measured on
#: an 8x37 population (296 preparations instead of 37); the counts asserted here are
#: shape-derived, so a smaller N keeps the same evidence at a fraction of the time.
N_ROWS = 12

#: A chunk budget small enough that ``plan_dense`` splits BOTH axes into 1x1
#: chunks at this shape -- ``_chunk_shape`` only produces ``p_chunks > 1 and
#: n_chunks > 1`` when the per-chunk element budget drops below ``pop_size``, and
#: 1x1 is exactly the worst case the payload memo exists for (it would re-prepare
#: every column once per ``(p0, n0)`` chunk).  The fixture asserts the resulting
#: chunk shape, so a change in the budget model fails loudly here.
_CHUNK_BUDGET_MB = 0.0004


def _tree_specs() -> List[Any]:
    """4 hand-built trees, each repeated twice -> P=8 with D=4 distinct programs.

    Hand-built (no RNG) so "the decoded programs contain duplicates" is exact; a
    ``Forest.random_generate`` population would not guarantee it.
    """
    return [
        _eq.fn(_eq.ADD, _eq.var(0), _eq.const(1.5)),
        _eq.fn(_eq.MUL, _eq.fn(_eq.SUB, _eq.var(1), _eq.const(0.5)), _eq.var(0)),
        _eq.fn(_eq.SIN, _eq.var(2)),
        _eq.fn(_eq.DIV, _eq.var(0), _eq.const(2.0)),
    ] * 2


def _x_host() -> np.ndarray:
    """Deterministic ``(N_ROWS, INPUT_LEN)`` float32 host input (no global RNG)."""
    rng = np.random.RandomState(0)
    return np.ascontiguousarray(rng.randn(N_ROWS, INPUT_LEN), dtype=np.float32)


def _labels_host() -> np.ndarray:
    """Deterministic ``(N_ROWS, 1)`` float32 host labels (no global RNG)."""
    rng = np.random.RandomState(1)
    return np.ascontiguousarray(rng.randn(N_ROWS, 1), dtype=np.float32)


def _bits_equal(left: Any, right: Any) -> bool:
    """Bit-exact float32 equality of two arrays/tensors (NaN payloads included)."""
    return _int32_view(left).shape == _int32_view(right).shape and np.array_equal(
        _int32_view(left), _int32_view(right)
    )


def _int32_view(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32).view(np.int32)


# --------------------------------------------------------------------------- #
# Counting seams
# --------------------------------------------------------------------------- #
class _FakeDeviceTensor:
    """Stand-in for an uploaded payload: ``.to(device)`` IS the host array.

    The seam only ever hands the *device* half of a payload to ``etl.run``.  With
    this fake, the numpy-built executable receives the very host arrays the numpy
    path would pass, so every ``etl.run`` stays a real, fully validated run while
    the wrapper still observes one "upload" per distinct payload.
    """

    __slots__ = ("host",)

    def __init__(self, host: np.ndarray) -> None:
        self.host = host

    def to(self, device: Any) -> np.ndarray:
        return self.host


class _SeamCounters:
    """Count the seam's payload preparations, uploads and ``etl.run`` calls.

    A context manager that patches the module globals ``dense_engine`` resolves at
    call time (``_prep_column`` / ``_chunk_rows`` / ``_keep_mask`` / ``etl.run``,
    plus ``etl.from_numpy`` in synthetic-device mode) with counting pass-throughs
    and restores every one of them on exit, exception included -- the same
    set/restore discipline as the batteries' ``Traffic``.

    ``run_arg_ids`` records the ``id()`` of every host array handed to ``etl.run``
    (per call): with a payload memo the second iteration over one ``staged`` dict
    must pass the IDENTICAL array objects, which is what "reused by reference"
    means.  The ids are stable because ``staged`` keeps every payload alive.
    """

    def __init__(self, synthetic_device: bool = False) -> None:
        self.synthetic_device = bool(synthetic_device)
        self.prepares: Dict[str, int] = {"column": 0, "rows": 0, "keep": 0}
        self.uploads: List[Tuple[int, ...]] = []
        self.runs = 0
        self.run_arg_ids: List[Tuple[int, ...]] = []

    # -- measurement -------------------------------------------------------
    def reset(self) -> None:
        """Zero every counter (the run-arg transcript included)."""
        for key in self.prepares:
            self.prepares[key] = 0
        self.uploads.clear()
        self.runs = 0
        self.run_arg_ids.clear()

    @property
    def upload_count(self) -> int:
        return len(self.uploads)

    def upload_shapes(self) -> Counter:
        """Upload multiplicity by payload shape (the "once per distinct payload" evidence)."""
        return Counter(self.uploads)

    # -- patching ----------------------------------------------------------
    def __enter__(self) -> "_SeamCounters":
        self._originals = (
            D._prep_column,
            D._chunk_rows,
            D._keep_mask,
            D.etl.run,
            D.etl.from_numpy,
        )
        prep_column, chunk_rows, keep_mask, run, from_numpy = self._originals

        def _prep_column(*args: Any, **kwargs: Any) -> np.ndarray:
            self.prepares["column"] += 1
            return prep_column(*args, **kwargs)

        def _chunk_rows(*args: Any, **kwargs: Any) -> np.ndarray:
            self.prepares["rows"] += 1
            return chunk_rows(*args, **kwargs)

        def _keep_mask(*args: Any, **kwargs: Any) -> np.ndarray:
            self.prepares["keep"] += 1
            return keep_mask(*args, **kwargs)

        def _run(executable: Any, *args: Any, **kwargs: Any) -> Any:
            self.runs += 1
            self.run_arg_ids.append(tuple(id(arg) for arg in args if isinstance(arg, np.ndarray)))
            return run(executable, *args, **kwargs)

        D._prep_column = _prep_column
        D._chunk_rows = _chunk_rows
        D._keep_mask = _keep_mask
        D.etl.run = _run
        if self.synthetic_device:

            def _from_numpy(array: np.ndarray) -> _FakeDeviceTensor:
                self.uploads.append(tuple(np.shape(array)))
                return _FakeDeviceTensor(array)

            D.etl.from_numpy = _from_numpy
        return self

    def __exit__(self, *exc: Any) -> bool:
        (
            D._prep_column,
            D._chunk_rows,
            D._keep_mask,
            D.etl.run,
            D.etl.from_numpy,
        ) = self._originals
        return False


class _BuildCounter:
    """Pass-through wrapper around ``etl_engine.build_cached`` counting calls."""

    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.original(*args, **kwargs)


#: The un-wrapped ``build_cached`` (the twin below must not go through a counter).
_ORIGINAL_BUILD_CACHED = E.build_cached


def _memo_free_executable(
    program: Any,
    n_rows: int,
    input_len: int,
    backend_name: str,
    device: Any,
    options: Dict[str, Any],
) -> Any:
    """``E._executable``'s pre-memo body (adapted from the battery's twin).

    ``_run_tree`` resolves ``E._executable`` at call time, so swapping this in
    sends every per-tree lookup straight to ``build_cached`` -- the reference the
    memo must be bit-identical to.
    """
    spec = etl.TensorSpec((int(n_rows), int(input_len)), etl.float32)
    if backend_name == "numpy":
        return _ORIGINAL_BUILD_CACHED(
            E._eval_step, spec, program, backend=backend_name, device=device, cache=E.CACHE
        )
    return _ORIGINAL_BUILD_CACHED(
        E._eval_step,
        spec,
        program,
        backend=backend_name,
        device=device,
        cache=E.CACHE,
        **options,
    )


# --------------------------------------------------------------------------- #
# Expected payload counts (the multiplicity the memo must produce)
# --------------------------------------------------------------------------- #
def _expected_payload_keys(plan: Any) -> set:
    """Every payload key the seam must memoize for ``plan`` (see the seam's layout).

    ``("column", index, p0)`` for the 7 node columns, then the row-keyed payloads:
    ``("x", p0)`` for ``per_row`` plans (tree ``i`` sees row ``i``), else
    ``("x", n0)`` plus the loss variant's ``("labels", n0)`` / ``("keep", n0)``.
    """
    keys = {
        ("column", index, p0)
        for index in range(len(D._COLUMN_NAMES))
        for p0 in range(0, plan.pop_size, plan.p_chunk)
    }
    if plan.per_row:
        keys |= {("x", p0) for p0 in range(0, plan.pop_size, plan.p_chunk)}
        return keys
    row_offsets = range(0, plan.n_rows, plan.n_chunk)
    keys |= {("x", n0) for n0 in row_offsets}
    if plan.variant == D.LOSS_VARIANT:
        keys |= {("labels", n0) for n0 in row_offsets}
        keys |= {("keep", n0) for n0 in row_offsets}
    return keys


def _expected_column_preparations(plan: Any) -> int:
    return len(D._COLUMN_NAMES) * plan.p_chunks


def _expected_row_preparations(plan: Any) -> int:
    """``_chunk_rows`` calls: ``x`` per chunk, plus ``labels`` in the loss variant."""
    chunks = plan.p_chunks if plan.per_row else plan.n_chunks
    per_chunk = 2 if (plan.variant == D.LOSS_VARIANT and not plan.per_row) else 1
    return per_chunk * chunks


def _expected_keep_preparations(plan: Any) -> int:
    if plan.variant == D.LOSS_VARIANT and not plan.per_row:
        return plan.n_chunks
    return 0


def _expected_runs(plan: Any) -> int:
    return plan.p_chunks * (1 if plan.per_row else plan.n_chunks)


def _assert_memoized_payload_counts(plan: Any, counters: _SeamCounters) -> None:
    """One iteration prepares each distinct payload once -- never the chunk product."""
    assert counters.prepares["column"] == _expected_column_preparations(plan), (
        f"column preparations={counters.prepares['column']} for {plan.p_chunks} "
        f"p-chunks (expected {_expected_column_preparations(plan)}; the naive "
        f"per-(p0, n0) loop would prepare "
        f"{_expected_column_preparations(plan) * (1 if plan.per_row else plan.n_chunks)})"
    )
    assert counters.prepares["rows"] == _expected_row_preparations(plan)
    assert counters.prepares["keep"] == _expected_keep_preparations(plan)
    assert counters.runs == _expected_runs(plan), (
        f"etl.run calls={counters.runs} for {plan.p_chunks}x"
        f"{1 if plan.per_row else plan.n_chunks} chunks (expected {_expected_runs(plan)})"
    )


def _assemble(plan: Any, chunks: List[Any]) -> np.ndarray:
    """The seam's raw results assembled exactly like ``run_plan`` (host)."""
    out = np.empty((plan.pop_size, plan.n_rows), dtype=np.float32)
    for (p0, p_chunk, n0, n_chunk), raw in chunks:
        host = np.asarray(raw.numpy(), dtype=np.float32).reshape(plan.p_chunk, plan.n_chunk)
        if plan.per_row:
            out[p0 : p0 + p_chunk, 0] = host[:p_chunk, 0]
        else:
            out[p0 : p0 + p_chunk, n0 : n0 + n_chunk] = host[:p_chunk, :n_chunk]
    return out


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rows() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The P=8 population rows (4 structures x 2) as CPU ``(values, types, sizes)``."""
    return _eq.rows_from_specs(_tree_specs(), pad_to=ROW_LEN)


@pytest.fixture(scope="module", params=["both-chunked", "unchunked"])
def value_plan(request: Any, rows: Any) -> Tuple[Any, Any]:
    """``(plan, executable)`` of the value variant, both-chunked and unchunked."""
    values, types, sizes = rows
    budget = _CHUNK_BUDGET_MB if request.param == "both-chunked" else None
    plan = D.plan_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        N_ROWS,
        per_row=False,
        budget_mb=budget,
        variant=D.VALUE_VARIANT,
        backend="numpy",
    )
    if request.param == "both-chunked":
        assert (plan.p_chunks, plan.n_chunks, plan.p_chunk, plan.n_chunk) == (
            POP,
            N_ROWS,
            1,
            1,
        ), (
            f"the chunked fixture must split both axes into 1x1 chunks, got "
            f"{plan.p_chunks}x{plan.n_chunks} chunks of "
            f"({plan.p_chunk}, {plan.n_chunk})"
        )
    else:
        assert (plan.p_chunks, plan.n_chunks) == (1, 1), (
            f"the unchunked fixture must stay in one chunk, got "
            f"{plan.p_chunks}x{plan.n_chunks}"
        )
    return plan, D.build_dense(plan, backend="numpy")


@pytest.fixture(scope="module")
def loss_plan(rows: Any) -> Tuple[Any, Any]:
    """``(plan, executable)`` of the both-chunked SR-loss variant."""
    values, types, sizes = rows
    plan = D.plan_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        N_ROWS,
        per_row=False,
        budget_mb=_CHUNK_BUDGET_MB,
        variant=D.LOSS_VARIANT,
        use_mse=True,
        backend="numpy",
    )
    assert plan.p_chunks > 1 and plan.n_chunks > 1, "the loss fixture must be chunked on both axes"
    return plan, D.build_dense(plan, backend="numpy")


@pytest.fixture(scope="module")
def memo_fixture(rows: Any) -> Tuple[Any, ...]:
    """``(pop, x, labels, forest, distinct_programs)`` for the executable-memo tests."""
    values, types, sizes = rows
    pop = E.decode_population(values, types, sizes, INPUT_LEN, 1)
    programs = [tree.program for tree in pop.trees if tree.program is not None]
    assert len(programs) == pop.pop_size, (
        "every fixture tree must be compilable: a constant-only tree carries "
        "``const_out`` instead of a program and never reaches ``_executable``"
    )
    distinct = len(set(programs))
    assert 0 < distinct < pop.pop_size, (
        f"the fixture needs duplicates: {pop.pop_size} trees decoded to {distinct} "
        "distinct static programs"
    )
    forest = Forest(INPUT_LEN, 1, values, types, sizes)
    return pop, torch.from_numpy(_x_host()), torch.from_numpy(_labels_host()), forest, distinct


# --------------------------------------------------------------------------- #
# 1-2. payload memo: one preparation and one upload per distinct payload
# --------------------------------------------------------------------------- #
def test_seam_prepares_each_payload_once(value_plan: Any) -> None:
    """One full iteration prepares each distinct payload exactly once."""
    plan, executable = value_plan
    x = _x_host()
    with _SeamCounters() as counters:
        chunks = list(
            D._iter_plan_chunks(plan, executable, x, backend_name="numpy", device=D._CPU)
        )
    _assert_memoized_payload_counts(plan, counters)
    assert counters.prepares["keep"] == 0, "the value variant has no keep mask payload"
    assert counters.upload_count == 0, "the numpy backend never uploads (host interpreter)"
    assert len(chunks) == _expected_runs(plan)
    if plan.p_chunks > 1 and plan.n_chunks > 1:
        naive = _expected_column_preparations(plan) * plan.n_chunks
        assert counters.prepares["column"] != naive, (
            "the column payloads were re-prepared per (p0, n0) chunk -- the payload "
            f"memo is not in effect ({naive} preparations)"
        )


def test_seam_uploads_each_payload_once_on_a_device_backend(value_plan: Any) -> None:
    """preparations == uploads, each payload uploaded once per call.

    Synthetic device backend: ``etl.from_numpy`` returns a fake whose ``.to()`` is
    the host array, so the numpy-built executable still runs on host arrays (it is
    a real, validated ``etl.run``).  No iree graph is compiled or executed.
    """
    plan, executable = value_plan
    x = _x_host()
    with _SeamCounters(synthetic_device=True) as counters:
        chunks = list(
            D._iter_plan_chunks(
                plan,
                executable,
                x,
                backend_name="iree",
                device=etl.core.Device("cuda", 0),
            )
        )
    _assert_memoized_payload_counts(plan, counters)
    assert counters.upload_count == sum(counters.prepares.values()), (
        "preparations and uploads must be one-to-one per payload: "
        f"{counters.prepares} vs {counters.upload_count} uploads"
    )
    assert counters.upload_shapes() == Counter(
        {
            # The fixtures are per_row=False: 7 node columns of (L, p_chunk) per
            # p-chunk, then one x payload of (n_chunk, input_len) per n-chunk.
            (plan.trace_len, plan.p_chunk): _expected_column_preparations(plan),
            (plan.n_chunk, plan.input_len): _expected_row_preparations(plan),
        }
    ), f"uploaded shapes {dict(counters.upload_shapes())}"
    assert len(chunks) == _expected_runs(plan)
    # The synthetic path must compute exactly what the numpy path computes.
    reference = D.run_plan(plan, executable, x, backend_name="numpy", device=D._CPU)
    assert _bits_equal(_assemble(plan, chunks), reference)


def test_seam_memoizes_the_loss_variant_payloads(loss_plan: Any) -> None:
    """The loss variant's extra ``labels`` and ``keep`` payloads are memoized too."""
    plan, executable = loss_plan
    x, labels = _x_host(), _labels_host()
    with _SeamCounters() as counters:
        chunks = list(
            D._iter_plan_chunks(
                plan, executable, x, labels, backend_name="numpy", device=D._CPU
            )
        )
    _assert_memoized_payload_counts(plan, counters)
    assert counters.prepares["rows"] == 2 * plan.n_chunks, (
        f"_chunk_rows={counters.prepares['rows']} for x + labels over "
        f"{plan.n_chunks} n-chunks"
    )
    assert counters.prepares["keep"] == plan.n_chunks
    assert counters.upload_count == 0
    assert len(chunks) == _expected_runs(plan)
    reference = D.run_loss_plan(
        plan, executable, x, labels, backend_name="numpy", device=D._CPU
    )
    totals = np.zeros((plan.pop_size,), dtype=np.float32)
    for (p0, p_chunk, _n0, _n_chunk), raw in chunks:
        totals[p0 : p0 + p_chunk] += np.asarray(raw.numpy(), dtype=np.float32).reshape(
            plan.p_chunk
        )[:p_chunk]
    assert _bits_equal(totals / np.float32(plan.n_rows), reference)


# --------------------------------------------------------------------------- #
# 3. payload memo: the injected ``staged`` dict
# --------------------------------------------------------------------------- #
def test_seam_staged_memo_is_filled_once_and_reused(value_plan: Any) -> None:
    """First iteration fills the injected dict; later ones only re-``etl.run``."""
    plan, executable = value_plan
    x = _x_host()
    staged: Dict[Any, Tuple[np.ndarray, Any]] = {}
    with _SeamCounters() as counters:
        first = list(
            D._iter_plan_chunks(
                plan, executable, x, backend_name="numpy", device=D._CPU, staged=staged
            )
        )
        first_prepares, first_runs = dict(counters.prepares), counters.runs
        first_arg_ids = list(counters.run_arg_ids)
        entry_ids = {key: id(entry) for key, entry in staged.items()}
        counters.reset()
        second = list(
            D._iter_plan_chunks(
                plan, executable, x, backend_name="numpy", device=D._CPU, staged=staged
            )
        )
        second_prepares, second_runs = dict(counters.prepares), counters.runs
        second_arg_ids = list(counters.run_arg_ids)

    assert set(staged) == _expected_payload_keys(plan), (
        f"staged keys differ: {sorted(set(staged) ^ _expected_payload_keys(plan))}"
    )
    assert len(staged) == len(_expected_payload_keys(plan))
    assert first_prepares == {
        "column": _expected_column_preparations(plan),
        "rows": _expected_row_preparations(plan),
        "keep": _expected_keep_preparations(plan),
    }, f"staging iteration preparations={first_prepares}"
    assert first_runs == _expected_runs(plan)
    assert second_prepares == {"column": 0, "rows": 0, "keep": 0}, (
        f"the second iteration over the same staged dict re-prepared payloads: "
        f"{second_prepares}"
    )
    assert second_runs == first_runs == _expected_runs(plan), (
        "the staged iteration must issue exactly the same etl.run calls "
        f"({second_runs} vs {first_runs})"
    )
    assert {key: id(entry) for key, entry in staged.items()} == entry_ids, (
        "the staged entries were replaced instead of reused"
    )
    assert second_arg_ids == first_arg_ids, (
        "the second iteration handed DIFFERENT host arrays to etl.run: the staged "
        "payloads are not reused by reference"
    )
    assert all(
        _bits_equal(np.asarray(a[1].numpy()), np.asarray(b[1].numpy()))
        for a, b in zip(first, second)
    ), "re-running the staged payloads must reproduce the same bits"


# --------------------------------------------------------------------------- #
# 4. executable memo: cache_stats()/clear_cache() contract
# --------------------------------------------------------------------------- #
def test_cache_stats_exposes_both_tables() -> None:
    """8 int keys: the compile table plus the executable front table."""
    stats = E.cache_stats()
    assert set(stats) == {
        "hits",
        "misses",
        "entries",
        "maxsize",
        "memo_hits",
        "memo_misses",
        "memo_entries",
        "memo_maxsize",
    }, f"cache_stats keys={sorted(stats)}"
    assert all(isinstance(value, int) for value in stats.values()), (
        f"non-int counter(s): {[k for k, v in stats.items() if not isinstance(v, int)]}"
    )
    assert stats["maxsize"] == E.CACHE.maxsize
    assert stats["memo_maxsize"] == E.CACHE.maxsize, (
        "the front table must be bounded by the compile cache's maxsize"
    )
    assert stats["memo_maxsize"] == E._EXECUTABLE_MEMO.maxsize


# --------------------------------------------------------------------------- #
# 5. executable memo: one build per distinct program
# --------------------------------------------------------------------------- #
def test_executable_memo_builds_once_per_distinct_program(
    memo_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D distinct programs -> D builds; a second evaluation builds nothing."""
    pop, x, _labels, _forest, distinct = memo_fixture
    counter = _BuildCounter(E.build_cached)
    monkeypatch.setattr(E, "build_cached", counter)
    E.clear_cache()

    first = E.evaluate(pop, x, per_row=False, backend="numpy")
    after_first = E.cache_stats()
    assert counter.count == distinct, (
        f"one evaluation of {pop.pop_size} trees ({distinct} distinct programs) "
        f"reached build_cached {counter.count} times"
    )
    assert after_first["memo_misses"] == distinct
    assert after_first["memo_hits"] == pop.pop_size - distinct, (
        "the duplicate structures must be served from the memo within the same call"
    )
    assert after_first["memo_entries"] == distinct
    assert after_first["misses"] == distinct and after_first["hits"] == 0, (
        "every memo miss goes on to build_cached, which counts exactly one lookup"
    )
    assert after_first["entries"] == distinct

    counter.reset()
    second = E.evaluate(pop, x, per_row=False, backend="numpy")
    after_second = E.cache_stats()
    assert counter.count == 0, "the second evaluation rebuilt an executable"
    assert after_second["memo_hits"] - after_first["memo_hits"] == pop.pop_size, (
        "every tree performs one memo lookup per evaluation"
    )
    assert after_second["memo_misses"] == after_first["memo_misses"]
    assert after_second["hits"] - after_first["hits"] == 0, (
        "a memo hit must not touch the compile cache at all"
    )
    assert after_second["entries"] == after_first["entries"]
    assert _bits_equal(first, second)


def test_clear_cache_resets_both_tables_and_invalidates(
    memo_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clear_cache()`` zeroes both tables and really invalidates the memo."""
    pop, x, _labels, _forest, distinct = memo_fixture
    counter = _BuildCounter(E.build_cached)
    monkeypatch.setattr(E, "build_cached", counter)
    E.clear_cache()
    empty = E.cache_stats()
    assert all(
        empty[key] == 0
        for key in ("hits", "misses", "entries", "memo_hits", "memo_misses", "memo_entries")
    ), f"clear_cache() left {empty}"

    E.evaluate(pop, x, per_row=False, backend="numpy")
    warm = E.cache_stats()
    assert counter.count == distinct
    assert warm["memo_misses"] == distinct and warm["memo_hits"] == pop.pop_size - distinct
    assert warm["entries"] == distinct and warm["misses"] == distinct

    E.clear_cache()
    cleared = E.cache_stats()
    assert all(
        cleared[key] == 0
        for key in ("hits", "misses", "entries", "memo_hits", "memo_misses", "memo_entries")
    ), f"the second clear_cache() left {cleared}"

    counter.reset()
    E.evaluate(pop, x, per_row=False, backend="numpy")
    rebuilt = E.cache_stats()
    assert counter.count == distinct, "clear_cache() did not invalidate the memo"
    assert rebuilt["memo_misses"] == distinct, (
        "a cleared memo must miss (and rebuild) once per distinct program"
    )
    assert rebuilt["memo_entries"] == distinct and rebuilt["entries"] == distinct


# --------------------------------------------------------------------------- #
# 6. executable memo: bit-identical to the memo-free engine
# --------------------------------------------------------------------------- #
def test_executable_memo_is_bit_identical_to_the_memo_free_twin(
    memo_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm memo, cold rebuild and the pre-memo body agree bit for bit."""
    pop, x, labels, _forest, _distinct = memo_fixture
    E.clear_cache()
    memo_eval = E.evaluate(pop, x, per_row=False, backend="numpy")
    memo_fitness = E.sr_fitness(pop, x, labels, use_MSE=True, backend="numpy")

    E.clear_cache()  # a full cold rebuild must reproduce the same bits
    cold_eval = E.evaluate(pop, x, per_row=False, backend="numpy")
    cold_fitness = E.sr_fitness(pop, x, labels, use_MSE=True, backend="numpy")

    monkeypatch.setattr(E, "_executable", _memo_free_executable)
    E.clear_cache()
    plain_eval = E.evaluate(pop, x, per_row=False, backend="numpy")
    plain_fitness = E.sr_fitness(pop, x, labels, use_MSE=True, backend="numpy")

    assert _bits_equal(memo_eval, cold_eval), "the cold rebuild changed the evaluate result"
    assert _bits_equal(memo_eval, plain_eval), "the memo path differs from the memo-free one"
    assert _bits_equal(memo_fitness, cold_fitness), "the cold rebuild changed SR_fitness"
    assert _bits_equal(memo_fitness, plain_fitness), (
        "the memo path differs from the memo-free one for SR_fitness"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Forest.SR_fitness relocates its inputs to cuda (check_tensor)",
)
def test_executable_memo_is_bit_identical_through_forest_sr_fitness(
    memo_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public ``Forest.SR_fitness`` route agrees with itself and the twin.

    On the default (numpy/auto) route this is the specialized engine, i.e. the
    memo; under ``EVOGP_EVAL_MODE=dense`` all three calls take the dense route, so
    the differential stays a self-consistency check there.
    """
    _pop, _x, labels, forest, _distinct = memo_fixture
    inputs = torch.from_numpy(_x_host()).to("cuda")
    targets = labels.to("cuda")

    warm = forest.SR_fitness(inputs, targets, True, "auto")
    E.clear_cache()
    cold = forest.SR_fitness(inputs, targets, True, "auto")
    monkeypatch.setattr(E, "_executable", _memo_free_executable)
    E.clear_cache()
    plain = forest.SR_fitness(inputs, targets, True, "auto")

    assert _bits_equal(warm, cold), "the cold rebuild changed Forest.SR_fitness"
    assert _bits_equal(warm, plain), (
        "Forest.SR_fitness differs between the memo path and the memo-free one"
    )
