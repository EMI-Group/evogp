"""Regression tests for the two build-once / run-many benchmark runner hooks.

``RUNNER_HOOKS.md`` (``evogp/tree/``) is the design document of record for the
two benchmark-support APIs this file pins -- they are NOT part of the public
route, nothing in the library calls them, and they deliberately ignore
``EVOGP_EVAL_MODE`` (the caller picks the route by picking the builder):

* ``dense_engine.build_dense_runner(population, x, *, per_row=False, ...)`` --
  the build phase is duck-type + input conversion once + ``plan_dense`` +
  ONE compile (one cache entry for the WHOLE population) + input staging (the
  7 ``(L, P)`` node columns and ``x``, prepared/uploaded once through the
  readback-free seam) + one warm-up ``etl.run``; ``.run()`` is ONE ``etl.run``
  over the staged payloads returning the RAW backend result of shape
  ``(pop_size, n_rows)`` (host-backed on numpy, device-resident on a device
  backend), with no re-plan / re-key / re-decode / readback / torch conversion.
* ``etl_engine.build_specialized_runner(population, x, *, per_row=False, ...)``
  -- the build phase is one ``decode_population`` + one ``_executable`` per
  non-constant tree + staging the input block(s); ``.run()`` is ``n_nonconst``
  ``etl.run`` calls returning a ``pop_size``-long tuple whose entries are the RAW
  per-tree results (``(n_rows, output_len)``) or ``None`` for a constant tree
  (staged read-only in ``.constant_rows``), and it never assembles host rows.

What is covered here (the MODEST pytest port of ``evogp/tree/runner_check.py``'s
numpy leg -- P=8, N=37, L=16, one constant tree injected, deterministic
hand-built trees, numpy backend only):

1. dense contract -- build succeeds on a single-chunk plan with ONE compile for
   the whole population, ``.invocations == 1``, ``.device_resident is False``,
   host-backed result of shape ``(pop_size, n_rows)``, ``.staged`` = the 7 node
   columns + ``x`` re-used by reference;
2. dense run -- repeated calls are bit-identical and every ``etl_engine``
   ``cache_stats()`` delta of a repeated ``.run()`` is EMPTY (the
   "no re-plan / no re-decode / no re-key" invariant);
3. dense numerics -- ``per_row=False`` / ``per_row=True`` vs
   ``etl_engine.evaluate`` (rtol=atol=1e-4 on finite values, exact NaN/±inf
   masks);
4. specialized contract + numerics -- tuple length ``pop_size``, ``None``
   exactly on the constant-tree slot, non-``None`` entries matching per-tree
   ``etl_engine`` evaluation (1e-4 + exact masks), ``.invocations ==
   n_nonconst``, repeats bit-identical with EMPTY cache deltas, and (via
   monkeypatch counters) one decode + one ``_executable`` per non-constant tree
   with one ``build_cached`` per DISTINCT program at build and none at run;
5. build-time contracts -- multi-output and a chunked plan raise
   ``DenseUnsupported`` (refused BEFORE any compile), wrong ``x`` columns and a
   ``per_row`` row-count mismatch raise ``ValueError``, a non-Forest-like
   container raises ``TypeError`` naming the missing attributes, and both
   builders ignore ``EVOGP_EVAL_MODE`` without writing to the environment.

Not covered here, deliberately: the ``iree`` leg.  Its device-resident numbers
(and the per-call traffic counters of both routes) are pinned by the dev battery
``evogp/tree/runner_check.py`` (``--backend iree``, one config per process)
because in-suite iree execution is far too heavy for a pytest file and the
recorded teardown-only nanobind leak closes the process after the run -- a
failure mode that must not be able to take the suite with it.  Wall-clock
measurements live in the dev battery / ``perf_probe.py``; this file contains no
timing assertion by design.

Run from the repository root::

    ETL_DISABLE_LOCATIONS=1 PYTHONPATH=src <venv>/bin/python -m pytest \\
        src/evogp/tests/test_runner_hooks.py -q
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch

# --------------------------------------------------------------------------- #
# Reused fixtures / comparator (see src/evogp/tests/CONTEXT.md's reusable
# helpers): the tiny spec DSL, the deterministic row materializer and the
# 1e-4-with-exact-NaN/±inf-placement comparator.  This file adds its own
# device/size policy on top so it stays a small, hermetic pytest module.
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

#: The two imports below are guarded by ``ImportError`` ONLY, deliberately: a
#: genuinely missing dependency (``etl`` not installed, a pre-merge worktree without
#: ``test_equivalence.py``) skips the whole file, while an import REGRESSION inside
#: the engine or the shared fixtures -- a ``SyntaxError``, an ``AttributeError``, a
#: half-merged module -- must fail collection here instead of silently reporting 20
#: green "skipped" tests.
try:
    import test_equivalence as eq
except ImportError as exc:  # pragma: no cover - only in a pre-merge worktree
    eq = None
    _FIXTURE_IMPORT_ERROR: Any = exc
else:
    _FIXTURE_IMPORT_ERROR = None

try:
    from evogp.tree import Forest

    from evogp.tree import dense_engine as D
    from evogp.tree import etl_engine as E

    _ENGINE_IMPORT_ERROR: Any = None
except ImportError as exc:  # pragma: no cover - only when a dependency is missing
    Forest = None
    D = E = None
    _ENGINE_IMPORT_ERROR = exc


pytestmark = pytest.mark.skipif(
    _ENGINE_IMPORT_ERROR is not None or _FIXTURE_IMPORT_ERROR is not None,
    reason=(
        f"the runner hooks need evogp.tree + the tests' shared fixtures "
        f"(engine: {_ENGINE_IMPORT_ERROR!r}, fixtures: {_FIXTURE_IMPORT_ERROR!r})"
    ),
)

#: The engine is device-agnostic (``_as_numpy`` accepts torch tensors on any
#: device); the runners resolve the numpy backend to ``Device("cpu", 0)``
#: themselves.  Test tensors follow the suite's convention (cuda when present,
#: so the host<->device conversion of the population is exercised too).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: The dev battery's small fixture shape (``runner_check.py``'s defaults).
POP = 8
ROWS = 37
INPUT_LEN = 3
MAX_TREE_LEN = 16
#: The single constant (never-compiled) tree's value.
CONST_VALUE = 2.5
#: Warm ``.run()`` repeats for the bit-identity / zero-traffic checks.
REPEATS = 3
#: The dense VALUE variant stages exactly these ``(L, P)`` node columns plus the
#: ``x`` block (RUNNER_HOOKS.md section 2).
DENSE_COLUMNS = 7
#: One canonical quiet NaN, for bit comparisons of NaN-carrying fixtures.
_NAN_BITS = np.uint32(0x7FC00000)
_RTOL_ATOL_NOTE = "rtol=atol=1e-4 on finite values, exact NaN/inf placement"


# --------------------------------------------------------------------------- #
# fixture data (deterministic: no RNG is involved in the population)
# --------------------------------------------------------------------------- #
def population_specs() -> List[Any]:
    """7 non-constant trees + ONE constant tree (last row), hand-built on purpose.

    The adversarial cases are the NaN/±inf producers that make the exact-mask
    comparisons meaningful, and they are chosen to have NO rounding-level
    boundary: ``log(x2)`` is ``-inf`` exactly on the single all-zero data row and
    NaN exactly on the negative inputs, so ``abs(log(x2))`` is ``+inf`` there.
    A mask mismatch can therefore only be a real engine disagreement.  Row 6
    repeats row 0 exactly: the specialized build compiles ONE graph per DISTINCT
    program, not one per tree.
    """
    shared = eq.fn(eq.ADD, eq.fn(eq.MUL, eq.var(0), eq.var(1)), eq.fn(eq.SIN, eq.var(2)))
    return [
        shared,  # x0 * x1 + sin(x2)
        eq.fn(eq.LOG, eq.var(2)),  # -inf on the zero row, NaN on negatives
        eq.fn(eq.ABS, eq.fn(eq.LOG, eq.var(2))),  # +inf / NaN placements
        eq.fn(eq.IF, eq.var(0), eq.var(1), eq.var(2)),  # ternary
        eq.fn(eq.ADD, eq.fn(eq.MUL, eq.const(1.0), eq.const(2.0)), eq.var(1)),  # folded subtree
        eq.fn(eq.MAX, eq.fn(eq.SQRT, eq.fn(eq.ABS, eq.var(2))), eq.const(-1.0)),
        shared,  # duplicate PROGRAM: one compile serves both trees
        eq.const(CONST_VALUE),  # constant tree: folded at decode, never compiled
    ]


def multi_output_specs() -> List[Any]:
    """A small ``output_len=2`` (OUT_NODE) population: the dense path's refusal."""
    return [
        eq.fn(
            eq.ADD,
            eq.out_fn(eq.ADD, 0, eq.var(0), eq.var(1)),
            eq.out_fn(eq.MUL, 1, eq.var(1), eq.var(2)),
        ),
        eq.fn(
            eq.IF,
            eq.var(0),
            eq.out_fn(eq.TANH, 1, eq.var(1)),
            eq.out_fn(eq.ABS, 0, eq.var(2)),
        ),
    ]


def build_forest(
    specs: List[Any],
    *,
    input_len: int = INPUT_LEN,
    output_len: int = 1,
    max_tree_len: int = MAX_TREE_LEN,
    device: torch.device = DEVICE,
) -> "Forest":
    """Materialize hand-built specs as a ``Forest`` on ``device``."""
    values, types, sizes = eq.rows_from_specs(
        specs, output_len=output_len, pad_to=max_tree_len
    )
    return Forest(
        input_len,
        output_len,
        values.to(device),
        types.to(device),
        sizes.to(device),
    )


def make_inputs(rows: int, *, seed: int = 5) -> torch.Tensor:
    """``(rows, input_len)`` float32 data with ONE all-zero row.

    The zero row is what makes the ``log(x2)`` tree produce an exact ``-inf``
    (its negative inputs give NaN), so the mask comparisons below are not
    vacuous.  ``eq.x_random`` uses its own ``torch.Generator``: no global RNG
    state is involved.
    """
    data = torch.cat(
        [eq.x_random(rows - 1, INPUT_LEN, seed=seed), torch.zeros(1, INPUT_LEN)]
    )
    return data.to(DEVICE)


def reference_rows(forest: "Forest", x: torch.Tensor, *, per_row: bool) -> torch.Tensor:
    """``etl_engine.evaluate`` rows for ``forest`` (the engine's own assembler)."""
    population = E.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    return E.evaluate(population, x, per_row=per_row, backend="numpy")


def constant_slots(runner: Any) -> List[int]:
    """Row indices the decode folded into a constant (never compiled)."""
    return [
        index
        for index, tree in enumerate(runner.population.trees)
        if tree.const_out is not None
    ]


def assemble(runner: Any, results: Tuple[Any, ...]) -> np.ndarray:
    """Host ``(pop, n_rows, output_len)`` rows from a specialized ``.run()``.

    The runner itself never assembles anything: the constant trees come from the
    broadcast read-only ``.constant_rows`` and every other slot is the raw ETL
    result, read back here by the TEST (the consumer's step).
    """
    rows = np.empty(
        (runner.pop_size, runner.n_rows, runner.output_len), dtype=np.float32
    )
    for index, raw in enumerate(results):
        if raw is None:
            rows[index] = runner.constant_rows[index]
        else:
            rows[index] = np.asarray(raw.numpy(), dtype=np.float32).reshape(
                runner.n_rows, runner.output_len
            )
    return rows


def canonical_bits(array: Any) -> np.ndarray:
    """``uint32`` view of a float32 array with every NaN collapsed to one pattern.

    Bit-exact comparison without the NaN trap: ``-0.0``/``+0.0`` stay distinct
    (a real difference), NaN payloads do not.
    """
    data = np.ascontiguousarray(array, dtype=np.float32)
    bits = data.view(np.uint32).copy()
    bits[np.isnan(data)] = _NAN_BITS
    return bits


def assert_same_bits(got: Any, want: Any, *, context: str = "") -> None:
    """Bit-identical float32 comparison (NaN-aware, ±0.0-sensitive)."""
    a, b = np.asarray(got, dtype=np.float32), np.asarray(want, dtype=np.float32)
    assert a.shape == b.shape, f"{context}: shape {a.shape} != {b.shape}"
    assert np.array_equal(canonical_bits(a), canonical_bits(b)), (
        f"{context}: not bit-identical "
        f"({int((canonical_bits(a) != canonical_bits(b)).sum())} differing entries)"
    )


def cache_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    """Changed ``etl_engine.cache_stats()`` counters ({} = no cache activity)."""
    return {key: after[key] - before[key] for key in after if after[key] != before[key]}


def staged_ids(runner: Any) -> Dict[Any, int]:
    """Identity of every staged host payload (the memo must be re-used as-is)."""
    return {key: id(value[0]) for key, value in runner.staged.items()}


@dataclass
class RunnerCase:
    """A built runner plus its fixture and the BUILD-phase cache delta."""

    runner: Any
    forest: "Forest"
    x: torch.Tensor
    build_cache_delta: Dict[str, int]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fixture_data() -> Tuple["Forest", torch.Tensor]:
    """The shared ``(forest, x)`` pair: P=8, N=37, L=16, one constant tree."""
    return build_forest(population_specs()), make_inputs(ROWS)


def build_runner_under_cache_bracket(
    build: Callable[..., Any], forest: "Forest", x: torch.Tensor, **kwargs: Any
) -> Tuple[Any, Dict[str, int]]:
    """Build one runner with a clean cache and return its build-phase delta.

    The two runners share the process-wide ``etl_engine.CACHE`` (``dense_engine.
    DENSE_CACHE`` IS that same object), so a build-phase delta is only
    meaningful from a cleared cache -- and bracketing it does not disturb the
    already-built runners, which hold their executables directly.
    """
    E.clear_cache()
    before = E.cache_stats()
    runner = build(forest, x, **kwargs)
    return runner, cache_delta(before, E.cache_stats())


@pytest.fixture(scope="module")
def dense_case(fixture_data: Tuple["Forest", torch.Tensor]) -> RunnerCase:
    """``build_dense_runner(..., per_row=False)`` on the shared fixture."""
    forest, x = fixture_data
    runner, delta = build_runner_under_cache_bracket(
        D.build_dense_runner, forest, x, backend="numpy"
    )
    return RunnerCase(runner, forest, x, delta)


@pytest.fixture(scope="module")
def dense_per_row_case(fixture_data: Tuple["Forest", torch.Tensor]) -> RunnerCase:
    """``build_dense_runner(..., per_row=True)``: one input row per tree."""
    forest, _ = fixture_data
    x = make_inputs(POP, seed=11)  # N == pop_size
    runner, delta = build_runner_under_cache_bracket(
        D.build_dense_runner, forest, x, per_row=True, backend="numpy"
    )
    return RunnerCase(runner, forest, x, delta)


@pytest.fixture(scope="module")
def specialized_case(fixture_data: Tuple["Forest", torch.Tensor]) -> RunnerCase:
    """``build_specialized_runner(..., per_row=False)`` on the shared fixture."""
    forest, x = fixture_data
    runner, delta = build_runner_under_cache_bracket(
        E.build_specialized_runner, forest, x, backend="numpy"
    )
    return RunnerCase(runner, forest, x, delta)


# --------------------------------------------------------------------------- #
# 1. dense runner: build contract
# --------------------------------------------------------------------------- #
def test_dense_build_compiles_once_for_the_whole_population(dense_case: RunnerCase) -> None:
    """ONE graph per shape signature: one cache entry, no per-tree memo traffic."""
    delta = dense_case.build_cache_delta
    assert delta.get("misses") == 1, f"build-phase cache delta {delta}"
    assert delta.get("entries") == 1, f"build-phase cache delta {delta}"
    # The dense route never resolves a per-tree executable (nothing imports the
    # decode/plan of the specialized route).
    assert delta.get("memo_misses", 0) == 0, f"build-phase cache delta {delta}"
    assert delta.get("memo_hits", 0) == 0, f"build-phase cache delta {delta}"


def test_dense_build_stages_one_single_chunk_plan(dense_case: RunnerCase) -> None:
    """A single-chunk plan + the 7 node columns and ``x`` staged exactly once."""
    runner = dense_case.runner
    assert runner.per_row is False
    # The population/data shape diagnostics live on the plan (the runner keeps
    # the plan itself as its public diagnostic attribute).
    assert (runner.plan.pop_size, runner.plan.n_rows) == (POP, ROWS)
    assert runner.plan.p_chunks == 1 and runner.plan.n_chunks == 1
    assert runner.plan.trace_len == MAX_TREE_LEN
    # The documented payload set of a single-chunk VALUE plan; each entry is
    # ``(host_array, device_tensor_or_None)`` and numpy keeps the device half None.
    assert set(runner.staged) == {
        ("column", index, 0) for index in range(DENSE_COLUMNS)
    } | {("x", 0)}
    assert all(entry[1] is None for entry in runner.staged.values())


def test_dense_run_returns_the_raw_host_backed_block(dense_case: RunnerCase) -> None:
    """``.run()`` = ONE ``etl.run`` returning the raw ``(pop_size, n_rows)`` block."""
    runner = dense_case.runner
    assert runner.invocations == 1
    assert runner.backend_name == "numpy"
    assert runner.device_resident is False  # numpy is host-only by construction

    raw = runner.run()
    host = raw.numpy()
    assert isinstance(host, np.ndarray)
    assert host.dtype == np.float32
    assert host.shape == (POP, ROWS)
    # Host-backed means "nothing is copied": repeated reads of the same raw
    # result share one buffer (a per-call copy would have a different address).
    assert host.__array_interface__["data"][0] == raw.numpy().__array_interface__["data"][0]


# --------------------------------------------------------------------------- #
# 2. dense runner: repeatability and a frozen cache
# --------------------------------------------------------------------------- #
def test_dense_run_repeats_are_bit_identical_and_pay_no_cache_traffic(
    dense_case: RunnerCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private invariant of the hook: N calls are N identical passes, nothing else."""
    runner = dense_case.runner
    first = runner.run().numpy()
    before, ids_before = E.cache_stats(), staged_ids(runner)

    # The cache-stats delta below is the black-box half of the invariant; these
    # counters are the white-box half, so a repeated ``.run()`` that re-prepared a
    # payload, re-decoded the population or re-resolved an executable anywhere would
    # fail here even if it happened to leave the cache counters alone.
    counts = {"pad_rows": 0, "from_numpy": 0, "decode": 0, "executable": 0}

    def counting(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for module, name, attribute in (
        (D, "pad_rows", "_pad_rows"),
        (D.etl, "from_numpy", "from_numpy"),
        (E, "decode", "decode_population"),
        (E, "executable", "_executable"),
    ):
        monkeypatch.setattr(module, attribute, counting(name, getattr(module, attribute)))

    for call in range(REPEATS):
        host = runner.run().numpy()
        assert_same_bits(host, first, context=f"dense .run() call {call + 1}")

    assert counts == {"pad_rows": 0, "from_numpy": 0, "decode": 0, "executable": 0}, (
        "a repeated .run() must re-use the staged payloads and the held executable, "
        f"got {counts}"
    )
    assert cache_delta(before, E.cache_stats()) == {}, (
        "a repeated .run() must not touch the compile cache or the executable memo"
    )
    assert staged_ids(runner) == ids_before, (
        "the staged payload memo must be re-used by reference, never re-prepared"
    )


# --------------------------------------------------------------------------- #
# 3. dense runner: numerics
# --------------------------------------------------------------------------- #
def test_dense_rows_match_specialized_evaluation(dense_case: RunnerCase) -> None:
    """``per_row=False``: ``(pop, N)`` rows vs ``etl_engine.evaluate``."""
    runner, forest, x = dense_case.runner, dense_case.forest, dense_case.x
    got = runner.run().numpy().reshape(runner.plan.pop_size, runner.plan.n_rows)
    assert got.shape == (POP, ROWS)
    want = reference_rows(forest, x, per_row=False)
    eq.assert_close_with_placement(
        got,
        want[:, :, 0],
        context=f"dense .run() vs evaluate(per_row=False) ({_RTOL_ATOL_NOTE})",
    )
    # The fixture is only meaningful if it really carries NaN and ±inf.
    assert np.isnan(got).any() and np.isposinf(got).any() and np.isneginf(got).any()


def test_dense_per_row_rows_match_specialized_evaluation(dense_per_row_case: RunnerCase) -> None:
    """``per_row=True``: ``(pop, 1)`` rows vs ``etl_engine.evaluate(per_row=True)``."""
    runner, forest, x = (
        dense_per_row_case.runner,
        dense_per_row_case.forest,
        dense_per_row_case.x,
    )
    assert runner.per_row is True
    assert (runner.plan.pop_size, runner.plan.n_rows) == (POP, 1)
    assert runner.invocations == 1
    got = runner.run().numpy().reshape(runner.plan.pop_size, runner.plan.n_rows)
    assert got.shape == (POP, 1)
    eq.assert_close_with_placement(
        got,
        reference_rows(forest, x, per_row=True),
        context=f"dense .run() vs evaluate(per_row=True) ({_RTOL_ATOL_NOTE})",
    )


# --------------------------------------------------------------------------- #
# 4. specialized runner: contract, numerics, repeatability
# --------------------------------------------------------------------------- #
def test_specialized_run_contract(specialized_case: RunnerCase) -> None:
    """``pop_size`` raw per-tree results with ``None`` on the constant slots."""
    runner = specialized_case.runner
    results = runner.run()
    slots = constant_slots(runner)
    assert slots == [POP - 1], "the fixture injects exactly ONE constant tree"
    assert isinstance(results, tuple)
    assert len(results) == POP
    assert [index for index, raw in enumerate(results) if raw is None] == slots
    assert runner.invocations == POP - len(slots)
    assert runner.per_row is False and runner.n_rows == ROWS
    for index, raw in enumerate(results):
        if raw is None:
            assert runner.constant_rows[index].shape == (ROWS, runner.output_len)
            assert np.all(runner.constant_rows[index] == np.float32(CONST_VALUE))
        else:
            assert raw.numpy().shape == (ROWS, runner.output_len)


def test_specialized_rows_match_specialized_evaluation(specialized_case: RunnerCase) -> None:
    """Assembled rows vs ``etl_engine.evaluate`` (the constant slot included)."""
    runner, forest, x = (
        specialized_case.runner,
        specialized_case.forest,
        specialized_case.x,
    )
    rows = assemble(runner, runner.run())
    want = reference_rows(forest, x, per_row=False)
    eq.assert_close_with_placement(
        rows,
        want,
        context=f"specialized .run() vs evaluate(per_row=False) ({_RTOL_ATOL_NOTE})",
    )
    assert np.isnan(rows).any() and np.isposinf(rows).any() and np.isneginf(rows).any()


def test_specialized_per_row_rows_match_specialized_evaluation(
    fixture_data: Tuple["Forest", torch.Tensor],
) -> None:
    """``per_row=True`` stages one input row per non-constant tree."""
    forest, _ = fixture_data
    x = make_inputs(POP, seed=17)
    runner = E.build_specialized_runner(forest, x, per_row=True, backend="numpy")
    assert runner.per_row is True and runner.n_rows == 1
    results = runner.run()
    assert [index for index, raw in enumerate(results) if raw is None] == [POP - 1]
    # ``evaluate(per_row=True)`` returns ``(pop, output_len)``: the assembled
    # ``(pop, 1, output_len)`` rows are the same block one axis wider.
    eq.assert_close_with_placement(
        assemble(runner, results)[:, 0, :],
        reference_rows(forest, x, per_row=True),
        context=f"specialized .run() vs evaluate(per_row=True) ({_RTOL_ATOL_NOTE})",
    )


def test_specialized_run_repeats_are_bit_identical_and_pay_no_cache_traffic(
    specialized_case: RunnerCase,
) -> None:
    """N calls issue the same ``n_nonconst`` runs and never move a cache counter."""
    runner = specialized_case.runner
    first = assemble(runner, runner.run())
    before = E.cache_stats()

    for call in range(REPEATS):
        results = runner.run()
        assert len(results) == POP
        assert_same_bits(
            assemble(runner, results), first, context=f"specialized .run() call {call + 1}"
        )

    assert cache_delta(before, E.cache_stats()) == {}, (
        "a repeated .run() must not re-decode, re-key or re-compile anything"
    )


def test_specialized_build_decodes_once_and_compiles_once_per_distinct_program(
    fixture_data: Tuple["Forest", torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build-phase counts: 1 decode, 1 executable per tree, 1 compile per PROGRAM."""
    forest, x = fixture_data
    counts = {"builds": 0, "lookups": 0, "decodes": 0}

    def counting(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name, attribute in (
        ("builds", "build_cached"),
        ("lookups", "_executable"),
        ("decodes", "decode_population"),
    ):
        monkeypatch.setattr(E, attribute, counting(name, getattr(E, attribute)))

    E.clear_cache()
    before = E.cache_stats()
    runner = E.build_specialized_runner(forest, x, backend="numpy")
    after = E.cache_stats()

    programs = [
        tree.program for tree in runner.population.trees if tree.program is not None
    ]
    distinct = len(set(programs))
    assert counts["decodes"] == 1, "the population must be decoded exactly once"
    assert counts["lookups"] == len(programs) == runner.invocations
    assert counts["builds"] == distinct < len(programs), (
        "one build_cached per DISTINCT program (the duplicate row is a memo hit), "
        f"got {counts['builds']} for {distinct} distinct of {len(programs)} trees"
    )
    assert after["misses"] - before["misses"] == distinct
    assert after["memo_hits"] - before["memo_hits"] == len(programs) - distinct

    # The same three counters must stay frozen across a ``.run()``.
    frozen = dict(counts)
    runner.run()
    assert counts == frozen, "a .run() must not decode, look up or compile anything"


# --------------------------------------------------------------------------- #
# 5. build-time contracts
# --------------------------------------------------------------------------- #
def _build_dense(forest: Any, x: Any, **kwargs: Any) -> Any:
    return D.build_dense_runner(forest, x, backend="numpy", **kwargs)


def _build_specialized(forest: Any, x: Any, **kwargs: Any) -> Any:
    return E.build_specialized_runner(forest, x, backend="numpy", **kwargs)


def test_dense_builder_refuses_a_multi_output_population(
    fixture_data: Tuple["Forest", torch.Tensor],
) -> None:
    """Multi-output is the specialized route's job; the dense builder refuses."""
    _, x = fixture_data
    forest = build_forest(multi_output_specs(), output_len=2)
    assert forest.output_len == 2
    before = E.cache_stats()
    with pytest.raises(D.DenseUnsupported, match="single-output"):
        _build_dense(forest, x)
    assert cache_delta(before, E.cache_stats()) == {}, "refuse before compiling"


def test_dense_builder_refuses_a_chunked_plan(dense_case: RunnerCase) -> None:
    """A chunked result cannot be assembled without a host step: refuse at build."""
    forest, x = dense_case.forest, dense_case.x
    before = E.cache_stats()
    with pytest.raises(D.DenseUnsupported, match="single-chunk"):
        D.build_dense_runner(forest, x, backend="numpy", budget_mb=1e-6)
    assert cache_delta(before, E.cache_stats()) == {}, "refuse before compiling"


@pytest.mark.parametrize(
    "build",
    [pytest.param(_build_dense, id="dense"), pytest.param(_build_specialized, id="specialized")],
)
def test_builders_reject_wrong_input_columns(
    fixture_data: Tuple["Forest", torch.Tensor], build: Callable[..., Any]
) -> None:
    """``x`` must have exactly ``input_len`` columns (engine semantics)."""
    forest, x = fixture_data
    bad = torch.zeros(x.shape[0], INPUT_LEN + 1, device=DEVICE)
    with pytest.raises(ValueError, match="columns"):
        build(forest, bad)


@pytest.mark.parametrize(
    "build",
    [pytest.param(_build_dense, id="dense"), pytest.param(_build_specialized, id="specialized")],
)
def test_builders_reject_per_row_row_count_mismatch(
    fixture_data: Tuple["Forest", torch.Tensor], build: Callable[..., Any]
) -> None:
    """``per_row=True`` needs one input row per tree."""
    forest, x = fixture_data
    assert x.shape[0] != forest.pop_size
    with pytest.raises(ValueError, match=r"per_row=True needs .*pop_size"):
        build(forest, x, per_row=True)


@pytest.mark.parametrize(
    "build",
    [pytest.param(_build_dense, id="dense"), pytest.param(_build_specialized, id="specialized")],
)
def test_builders_reject_non_forest_like_containers(
    fixture_data: Tuple["Forest", torch.Tensor], build: Callable[..., Any]
) -> None:
    """Duck typing: the error must name the missing ``Forest`` attributes."""
    _, x = fixture_data
    with pytest.raises(TypeError, match="Forest-like") as caught:
        build(object(), x)
    message = str(caught.value)
    for name in (
        "batch_node_value",
        "batch_node_type",
        "batch_subtree_size",
        "input_len",
        "output_len",
    ):
        assert name in message, f"the TypeError must name the missing {name}: {message}"


def test_builders_ignore_eval_mode_and_do_not_write_the_environment(
    fixture_data: Tuple["Forest", torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route is chosen by choosing the builder; ``EVOGP_EVAL_MODE`` is not read."""
    forest, x = fixture_data

    monkeypatch.setenv(D.EVAL_MODE_ENV, "specialized")
    environ_before = dict(os.environ)
    dense = D.build_dense_runner(forest, x, backend="numpy")
    assert dict(os.environ) == environ_before, "the builder must not write the environment"
    assert isinstance(dense.plan, D.DensePlan), "EVOGP_EVAL_MODE=specialized must not reroute"

    monkeypatch.setenv(D.EVAL_MODE_ENV, "dense")
    environ_before = dict(os.environ)
    specialized = E.build_specialized_runner(forest, x, backend="numpy")
    assert dict(os.environ) == environ_before, "the builder must not write the environment"
    assert isinstance(specialized.population, E.Population)
    # The EXACT shape (not just "something compiled"): one slot per tree, ``None``
    # exactly on the fixture's single constant tree -- the same check the
    # specialized-contract test makes.
    assert len(specialized.executables) == POP, (
        f"{len(specialized.executables)} executable slots for {POP} trees"
    )
    assert [
        index for index, executable in enumerate(specialized.executables) if executable is None
    ] == [POP - 1], (
        "the fixture injects exactly ONE constant tree (last row): it is folded at "
        "decode time and must be the only slot without an executable"
    )
    assert specialized.invocations == POP - 1

    # The numbers are the route's own, not the mode's: both builders agree.
    eq.assert_close_with_placement(
        dense.run().numpy(),
        assemble(specialized, specialized.run())[:, :, 0],
        context=f"dense vs specialized under flipped EVOGP_EVAL_MODE ({_RTOL_ATOL_NOTE})",
    )
