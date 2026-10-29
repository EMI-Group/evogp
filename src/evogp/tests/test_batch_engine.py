"""Contract of the opt-in "batch" route: ONE merged evoxc graph per population.

What this file pins
-------------------
``src/evogp/tree/batch_engine.py`` is the third evaluation route (after the
specialized per-structure engine and the dense all-ops engine): it traces every
DISTINCT program of a population into its own small graph and merges those graphs
into ONE executable with ``evoxc.batch_functions``, so a single ``evoxc.run``
evaluates the whole population and only ``(n_functions,)`` values (loss variant)
or a ``(n_functions, N, out_len)`` block (value variant) crosses back.
Reaching it needs an explicit ``EVOGP_EVAL_MODE=batch`` (the default ``auto``
policy never selects it), so this file calls the two entry points directly,
exactly like the runner builders are called by their tests.

The three things the route can silently get WRONG are the three this file hammers:

* **numbers** -- the merged graph must be numerically the SAME evaluation as the
  specialized per-structure engine (``evoxc_engine.decode_population`` +
  ``evaluate`` / ``sr_fitness``) and as the independent plain-torch oracle
  (``reference_eval.py``, reached through ``test_equivalence``'s plumbing):
  value variant for both input layouts, loss variant for MSE and MAE, and
  multi-output (``output_len`` 2 and 3);
* **row expansion** -- identical rows are merged into one function and expanded
  back host-side, so every population row must still be ITS program's answer (a
  wrong ``row_index`` or a foreign row mapping is served silently, not with an
  error), including a constant tree mixed in with computed ones;
* **placement** -- a NaN/±inf from one program must land on exactly that program's
  rows (``te.assert_close_with_placement`` asserts the mask first, then the
  finite entries).

Plus the pieces a caller observes: the fresh/writable/float32/dtype/device/
contiguity contract of both entry points, the ``describe_plan`` report of a
known duplicate-rich fixture, ``build_batch``'s memoized-but-re-bound executable,
and the ``BatchUnsupported`` declines (an unknown backend name, an empty
population, a shape doubt, ``pop_size < EVOGP_BATCH_MIN_POP``).

The regression the engine author flagged is pinned by
:func:`test_all_constant_population_survives_a_pop_size_escalation`: an
all-constant population evaluated at pop 4 -> 64 -> 256 in ONE process, where a
memoized executable used to be reused with the FIRST population's size (a stale
``(4,)`` loss) or the call was declined for the wrong pop size.

Shape grid
----------
``input_len = 3``, ``max_tree_len = 16``, ``N <= 9`` data rows, populations built
from FOUR distinct programs repeated (``pop_size`` 4..16 for the parity work, and
the constant-only 4 / 64 / 256 escalation). That keeps the merged function count
-- and therefore the numpy compile -- small: the whole file stays a few seconds.
There is **no wall-clock assertion anywhere** in this file (the economics live in
``src/evogp/tree/BATCH_EVAL.md``); the checks are values, shapes, flags and
counters/report fields.

Hermeticity
-----------
* ``backend="numpy"`` is passed EXPLICITLY on every call, so no test can compile
  an iree graph; the single iree leg is ``skipif``-guarded on the IREE packages
  (absent in this venv) plus a CUDA-visible device and therefore SKIPS here.
* an autouse fixture clears the four merge knobs per test
  (``EVOGP_BATCH_BAND_SIZE`` / ``BANDS`` / ``SWITCH_OPS`` / ``ALIGNMENT``) and
  sets ``EVOGP_BATCH_MIN_POP=1``, because every fixture here is smaller than the
  library's measured break-even (64); the ambient values are restored at teardown.
* nothing outside ``src/evogp/tests/`` is touched, no file is written, and
  ``reference_eval.py`` (the oracle) is used as-is -- never re-implemented here.
* neither entry point consults ``EVOGP_EVAL_MODE``, so the file behaves identically
  under the default run and under a forced ``EVOGP_EVAL_MODE=dense`` run.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python \\
        -m pytest src/evogp/tests/test_batch_engine.py -q -p no:cacheprovider
    EVOGP_EVAL_MODE=dense EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \\
        <venv>/bin/python -m pytest src/evogp/tests/test_batch_engine.py -q \\
        -p no:cacheprovider
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest
import torch

# --------------------------------------------------------------------------- #
# shared helpers: the oracle plumbing and the tree-spec DSL live in
# test_equivalence.py (imported by path, exactly like pytest's prepend import mode
# makes it importable from a sibling test module)
# --------------------------------------------------------------------------- #
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import test_equivalence as te
except ImportError:  # pragma: no cover - only for the importlib import mode
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "evogp_test_equivalence_helpers", _HERE / "test_equivalence.py"
    )
    te = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(te)

from evogp.tree import Forest  # noqa: E402
from evogp.tree import batch_engine  # noqa: E402
from evogp.tree import dense_engine  # noqa: E402
from evogp.tree import evoxc_engine  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: the whole file must stay a few seconds)
# --------------------------------------------------------------------------- #
BACKEND = "numpy"
INPUT_LEN = 3
ROW_LEN = 16
#: Data rows of the loss fixtures (a single chunk, no rounding anywhere).
N_ROWS = 9
#: The value the all-constant escalation evaluates (exactly representable).
CONST_VALUE = 3.5

#: FOUR distinct finite programs: three computed ones and one fully constant tree,
#: i.e. the population shape a GP loop actually presents (duplicates + a constant).
_PROGRAMS: List[Any] = [
    te.fn(te.ADD, te.var(0), te.var(1)),
    te.fn(te.MUL, te.var(2), te.const(1.5)),
    te.var(0),
    te.const(-2.25),
]
#: 16 rows out of those 4 programs -> 12 duplicate rows (the dedup fixture).
_DUPLICATE_POP: List[Any] = _PROGRAMS * 4

#: Hostile programs: NaN, +inf and a division by an exact 0 side by side, so the
#: NaN/±inf PLACEMENT of the merged graph is testable row by row.
_MIXED_SPECS: List[Any] = [
    te.fn(te.LOG, te.var(0)),  # NaN on the negative row
    te.fn(te.EXP, te.var(0)),  # +inf on the overflow row
    te.fn(te.DIV, te.var(1), te.const(0.0)),  # NaN (0 divisor)
    te.const(1.0),  # finite (constant-folded subtree)
]

#: Hostile data: ``log`` of a negative (NaN), ``exp(100)`` (overflow to +inf) and
#: a zero divisor (NaN) -- hand-built, so the mixed fixture is deterministic.
_MIXED_X = torch.tensor(
    [
        [-1.0, 0.5, 2.0],
        [100.0, 1.0, -0.5],
        [0.5, -2.0, 3.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

#: Multi-output specifications: two and three outputs per tree, folded from the
#: same three variables (``te.out_fn`` is the packed OUT_NODE DSL entry).
_TWO_OUTPUT_SPECS: List[Any] = [
    te.out_fn(te.ADD, 0, te.var(0), te.var(1)),
    te.out_fn(te.MUL, 1, te.var(2), te.const(2.0)),
]
_THREE_OUTPUT_SPECS: List[Any] = _TWO_OUTPUT_SPECS + [
    te.out_fn(te.SUB, 2, te.var(1), te.const(0.5)),
]

#: The merge knobs an ambient value could change under the assertions (all four are
#: cleared per test; ``MIN_POP`` is SET instead -- see the fixture).
_MERGE_KNOBS: Tuple[str, ...] = (
    batch_engine.BAND_SIZE_ENV,
    batch_engine.BANDS_ENV,
    batch_engine.SWITCH_OPS_ENV,
    batch_engine.ALIGNMENT_ENV,
)

#: Every ``describe_plan`` key the route documents (the report is a public seam).
_DESCRIBE_KEYS = frozenset(
    {
        "variant",
        "per_row",
        "use_mse",
        "pop_size",
        "functions",
        "const_functions",
        "row_index_collisions",
        "input_len",
        "output_len",
        "n_rows",
        "band_count",
        "band_sizes",
        "band_strategy",
        "switch_ops",
        "alignment",
        "emitted_ops",
        "n_functions",
        "specs",
        "spec_dtypes",
        "backend",
        "device",
    }
)


@pytest.fixture(autouse=True)
def _batch_knobs() -> Iterator[None]:
    """Clear the four merge knobs and admit small populations, per test.

    ``EVOGP_BATCH_BAND_SIZE`` / ``BANDS`` / ``SWITCH_OPS`` / ``ALIGNMENT`` are
    DELETED (not pinned to a default) so each test states the merge shape it wants
    through an explicit argument and an ambient value can never change it.

    ``EVOGP_BATCH_MIN_POP`` is SET to ``1``: the library default (64) is the
    measured call-count break-even, i.e. a real decline for every fixture in this
    file, and the entry points have no ``min_pop=`` argument -- the environment is
    the only way to admit a small population (the decline itself is exercised by
    :func:`test_population_below_the_threshold_declines`).

    The previous values of all five names are restored at teardown, so the rest of
    the suite is unaffected.
    """
    names = _MERGE_KNOBS + (batch_engine.MIN_POP_ENV,)
    saved = {name: os.environ.get(name) for name in names}
    for name in _MERGE_KNOBS:
        os.environ.pop(name, None)
    os.environ[batch_engine.MIN_POP_ENV] = "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --------------------------------------------------------------------------- #
# fixtures: deterministic hand-built populations (no global RNG involved)
# --------------------------------------------------------------------------- #
def _rows(specs: Sequence[Any], output_len: int = 1) -> Tuple[torch.Tensor, ...]:
    """``specs`` -> the three ``(pop_size, ROW_LEN)`` tensors with a CLEAN tail.

    ``te.rows_from_specs`` writes CONST/0.0 with subtree size 0 into the unused
    tail, so the fixture never depends on uninitialized memory.
    """
    values, types, sizes = te.rows_from_specs(
        specs, output_len, pad_to=ROW_LEN, tail_value=0.0
    )
    return values, types, sizes


def _forest(specs: Sequence[Any], output_len: int = 1) -> Forest:
    """A CPU ``Forest`` of the same rows (the oracle plumbing needs the container)."""
    values, types, sizes = _rows(specs, output_len)
    return Forest(INPUT_LEN, output_len, values, types, sizes)


def _x(rows: int, seed: int = 5) -> torch.Tensor:
    return te.x_random(rows, INPUT_LEN, seed=seed)


def _labels(rows: int, out_len: int = 1, seed: int = 101) -> torch.Tensor:
    """Deterministic labels from a LOCAL generator (no global-RNG coupling)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((rows, out_len), generator=generator, dtype=torch.float32)


def _constant_rows(pop_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An all-constant population: one node, ``CONST_VALUE``, every row identical."""
    values = torch.zeros((pop_size, 4), dtype=torch.float32)
    values[:, 0] = CONST_VALUE
    types = torch.zeros((pop_size, 4), dtype=torch.int16)
    types[:, 0] = te.CONST
    sizes = torch.zeros((pop_size, 4), dtype=torch.int16)
    sizes[:, 0] = 1
    return values, types, sizes


# --------------------------------------------------------------------------- #
# the call sites under test / the two references
# --------------------------------------------------------------------------- #
def _value(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    *,
    per_row: bool,
    output_len: int = 1,
    input_len: int = INPUT_LEN,
    **kwargs: Any,
) -> torch.Tensor:
    """``evaluate_batch`` with the shape grid and the numpy backend pinned."""
    return batch_engine.evaluate_batch(
        values,
        types,
        sizes,
        input_len,
        output_len,
        x,
        per_row=per_row,
        backend=BACKEND,
        **kwargs,
    )


def _loss(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    use_mse: bool = True,
    *,
    output_len: int = 1,
    input_len: int = INPUT_LEN,
    **kwargs: Any,
) -> torch.Tensor:
    """``sr_fitness_batch`` with the shape grid and the numpy backend pinned."""
    return batch_engine.sr_fitness_batch(
        values,
        types,
        sizes,
        input_len,
        output_len,
        inputs,
        labels,
        use_mse,
        backend=BACKEND,
        **kwargs,
    )


def _specialized_value(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    x: torch.Tensor,
    *,
    per_row: bool,
    output_len: int = 1,
) -> torch.Tensor:
    """The reference engine's value output for the SAME rows (the fallback route)."""
    population = evoxc_engine.decode_population(
        values, types, sizes, INPUT_LEN, output_len
    )
    return evoxc_engine.evaluate(population, x, per_row=per_row, backend=BACKEND)


def _specialized_loss(
    values: torch.Tensor,
    types: torch.Tensor,
    sizes: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    use_mse: bool = True,
    *,
    output_len: int = 1,
) -> torch.Tensor:
    """The reference engine's loss for the SAME rows (the fallback route)."""
    population = evoxc_engine.decode_population(
        values, types, sizes, INPUT_LEN, output_len
    )
    return evoxc_engine.sr_fitness(
        population, inputs, labels, use_mse, backend=BACKEND
    )


# --------------------------------------------------------------------------- #
# the skip guard of the single iree leg
# --------------------------------------------------------------------------- #
def _find_spec(name: str) -> Any:
    """``importlib.util.find_spec`` that answers ``None`` for a missing PARENT.

    ``find_spec("iree.compiler")`` raises ``ModuleNotFoundError`` when ``iree``
    itself is absent, which is exactly the case this guard must turn into a skip.
    """
    from importlib.util import find_spec

    try:
        return find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def _iree_packages_available() -> bool:
    return _find_spec("iree.compiler") is not None and _find_spec("iree.runtime") is not None


def _iree_leg_skip_reason() -> str:
    if not _iree_packages_available():
        return (
            "the iree backend requires the IREE Python packages and neither "
            "'iree.compiler' nor 'iree.runtime' is importable in this venv, so evoxc "
            "raises BackendError('the iree backend requires the IREE Python "
            "packages'); the numpy legs above cover the numeric contract"
        )
    if not torch.cuda.is_available():
        return (
            "the iree route compiles for the GPU HAL kind torch reports and this "
            "box has no CUDA-visible device (torch.cuda.is_available() is False)"
        )
    return "no iree leg on this box"


requires_device_iree = pytest.mark.skipif(
    not (_iree_packages_available() and torch.cuda.is_available()),
    reason=_iree_leg_skip_reason(),
)

#: The band spellings the merge accepts, with the ``(band_count, band_strategy)``
#: pair ``describe_plan`` must report for the 4-function duplicate fixture.
_BAND_CASES = (
    pytest.param({"bands": 1}, 1, "flat", id="bands-1"),
    pytest.param({"bands": 2}, 2, "explicit", id="bands-2-mid"),
    pytest.param({"bands": 4}, 4, "explicit", id="bands-4-all"),
    pytest.param({"bands": "none"}, 1, "flat", id="bands-none"),
    pytest.param({"bands": "flat"}, 1, "flat", id="bands-flat"),
    pytest.param({"band_size": 1}, 4, "explicit", id="band-size-1"),
    pytest.param({"switch_ops": True}, 1, "flat", id="switch-ops"),
)


def _assert_fresh_tensor(
    got: torch.Tensor, sources: Sequence[Tuple[str, Any]], *, shape: Tuple[int, ...]
) -> None:
    """The fresh/writable/contiguous/float32 contract every output must satisfy."""
    assert tuple(got.shape) == shape, f"shape {tuple(got.shape)} != {shape}"
    assert got.dtype is torch.float32, f"dtype {got.dtype}"
    assert got.is_contiguous(), "the result must be contiguous"
    assert got.requires_grad is False, "the result must not carry a grad flag"
    for name, source in sources:
        if isinstance(source, torch.Tensor):
            assert (
                got.untyped_storage().data_ptr()
                != source.untyped_storage().data_ptr()
            ), f"the result shares its storage with {name}"
    before = got.clone()
    got.copy_(before)  # a read-only / frozen buffer would raise here
    assert torch.equal(got, before), "the result must be writable in place"


# =========================================================================== #
# 1. the value variant: numbers
# =========================================================================== #
@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_value_route_matches_the_specialized_engine(per_row: bool) -> None:
    """Both input layouts are the SAME numbers as the specialized per-tree route.

    ``per_row=True`` feeds tree ``i`` row ``i`` of ``x`` (``(pop, input_len)`` ->
    ``(pop, output_len)``); ``per_row=False`` evaluates every tree on all rows
    (``(N, input_len)`` -> ``(pop, N, output_len)``). The comparison is
    mask-first (``rtol = atol = 1e-4``); the bit-exactness on top of it is the
    MEASURED numpy-backend behaviour of this route, pinned as current behaviour,
    not as a promise a legitimate reassociation would have to keep.
    """
    values, types, sizes = _rows(_PROGRAMS)
    x = _x(len(_PROGRAMS)) if per_row else _x(N_ROWS, seed=7)

    got = _value(values, types, sizes, x, per_row=per_row)
    want = _specialized_value(values, types, sizes, x, per_row=per_row)

    shape = (len(_PROGRAMS), 1) if per_row else (len(_PROGRAMS), N_ROWS, 1)
    assert tuple(got.shape) == shape, f"{tuple(got.shape)} != {shape}"
    te.assert_close_with_placement(
        got, want, context=f"value variant, per_row={per_row}, vs specialized"
    )
    assert torch.equal(got, want), (
        "measured bit-exact on the numpy backend (same emitters, same graph); a "
        f"differing result means the merge changed the arithmetic: "
        f"max|diff| = {float((got - want).abs().max()):.3g}"
    )


def test_value_route_matches_the_independent_oracle() -> None:
    """The merged graph is the oracle's ``(pop, N, out_len)`` evaluation, not a lookalike.

    ``te.reference_batch_forward`` is the obviously-correct plain-torch path
    (float64 intermediates, one Python op per node), so it is the only reference
    that is independent of every engine here.
    """
    specs = _PROGRAMS
    values, types, sizes = _rows(specs)
    x = _x(N_ROWS, seed=11)
    forest = _forest(specs)

    got = _value(values, types, sizes, x, per_row=False)
    te.assert_close_with_placement(
        got,
        te.reference_batch_forward(forest, x),
        context="value variant vs the independent oracle",
    )


@pytest.mark.parametrize("output_len", [2, 3], ids=["two_outputs", "three_outputs"])
@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_multi_output_value_matches_the_specialized_engine(
    output_len: int, per_row: bool
) -> None:
    """Multi-output populations are SUPPORTED by the merge (never a decline).

    The packed OUT_NODE rows of ``output_len`` 2 and 3 must give the same
    ``(pop, output_len)`` (``per_row``) / ``(pop, N, output_len)`` block as the
    specialized engine: this route must not be a single-output-only path.
    """
    specs = (_THREE_OUTPUT_SPECS if output_len == 3 else _TWO_OUTPUT_SPECS) * 2
    values, types, sizes = _rows(specs, output_len)
    rows = len(specs)
    x = _x(rows, seed=13) if per_row else _x(N_ROWS, seed=14)

    got = _value(
        values, types, sizes, x, per_row=per_row, output_len=output_len
    )
    want = _specialized_value(
        values, types, sizes, x, per_row=per_row, output_len=output_len
    )

    shape = (rows, output_len) if per_row else (rows, N_ROWS, output_len)
    assert tuple(got.shape) == shape, f"{tuple(got.shape)} != {shape}"
    assert got.shape[-1] == output_len
    te.assert_close_with_placement(
        got,
        want,
        context=f"multi-output (out_len={output_len}), per_row={per_row}",
    )


# =========================================================================== #
# 2. the loss variant: numbers
# =========================================================================== #
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_loss_route_matches_the_specialized_engine(use_mse: bool) -> None:
    """The in-graph SR reduction is the same loss as the specialized route's.

    The loss is folded into the merged graph (one rank-0 scalar per function), so
    this is the check that the folded reduction -- ``sum_N sum_out err / N``, the
    outputs SUMMED and never divided by ``output_len`` -- did not change.
    """
    values, types, sizes = _rows(_PROGRAMS)
    inputs = _x(N_ROWS, seed=17)
    labels = _labels(N_ROWS, seed=19)

    got = _loss(values, types, sizes, inputs, labels, use_mse)
    want = _specialized_loss(values, types, sizes, inputs, labels, use_mse)

    assert tuple(got.shape) == (len(_PROGRAMS),), f"{tuple(got.shape)}"
    te.assert_close_with_placement(
        got, want, context=f"loss variant (use_mse={use_mse}) vs specialized"
    )


@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_loss_route_matches_the_independent_oracle(use_mse: bool) -> None:
    """The loss equals the independent oracle's, and MSE/MAE really differ.

    The oracle tolerance is ``1e-4`` (it reduces in float64, and the graph reduces
    in float32, so the two are not bit-identical); the ``!=`` assertion keeps the
    metric parametrization from being vacuous.
    """
    specs = _PROGRAMS
    values, types, sizes = _rows(specs)
    inputs = _x(N_ROWS, seed=23)
    labels = _labels(N_ROWS, seed=29)
    forest = _forest(specs)

    got = _loss(values, types, sizes, inputs, labels, use_mse)
    te.assert_close_with_placement(
        got,
        te.reference_sr_fitness(forest, inputs, labels, use_mse),
        context=f"loss variant (use_mse={use_mse}) vs the independent oracle",
    )
    assert not torch.equal(got, _loss(values, types, sizes, inputs, labels, not use_mse)), (
        "MSE and MAE returned the same tensor: the metric is not reaching the graph"
    )


@pytest.mark.parametrize("output_len", [2, 3], ids=["two_outputs", "three_outputs"])
def test_multi_output_loss_matches_the_specialized_engine(output_len: int) -> None:
    """Multi-output loss: every output contributes, and no ``/ output_len`` sneaks in.

    A per-output loss would be 3x smaller at ``output_len=3``; the specialized
    engine's value is the contract, and the oracle is checked against it too.
    """
    specs = (_THREE_OUTPUT_SPECS if output_len == 3 else _TWO_OUTPUT_SPECS) * 2
    values, types, sizes = _rows(specs, output_len)
    inputs = _x(N_ROWS, seed=31)
    labels = _labels(N_ROWS, out_len=output_len, seed=37)
    forest = _forest(specs, output_len)

    got = _loss(values, types, sizes, inputs, labels, True, output_len=output_len)
    want = _specialized_loss(
        values, types, sizes, inputs, labels, True, output_len=output_len
    )
    assert tuple(got.shape) == (len(specs),), f"{tuple(got.shape)}"
    te.assert_close_with_placement(
        got,
        want,
        context=f"multi-output loss (out_len={output_len}) vs specialized",
    )
    te.assert_close_with_placement(
        got,
        te.reference_sr_fitness(forest, inputs, labels, True),
        context=f"multi-output loss (out_len={output_len}) vs the oracle",
    )


# =========================================================================== #
# 3. NaN / +-inf: the merged graph must keep the placement, not sanitize it
# =========================================================================== #
def test_nan_and_inf_placement_is_preserved() -> None:
    """Row ``i`` of the output keeps program ``i``'s NaNs/±infs, in both variants.

    A merge that reordered the function axis, reused one function's slots for
    another, or reduced the loss over the wrong rows would move a NaN (which no
    finite tolerance can detect) -- the mask-first comparison is what catches it.
    """
    values, types, sizes = _rows(_MIXED_SPECS)
    labels = _labels(_MIXED_X.shape[0], seed=41)

    got_value = _value(values, types, sizes, _MIXED_X, per_row=False)
    want_value = _specialized_value(values, types, sizes, _MIXED_X, per_row=False)
    assert torch.isnan(got_value).any() and torch.isinf(got_value).any(), (
        "the hostile fixture stopped producing NaN and inf"
    )
    te.assert_close_with_placement(
        got_value, want_value, context="hostile value variant vs specialized"
    )

    got_loss = _loss(values, types, sizes, _MIXED_X, labels, True)
    want_loss = _specialized_loss(values, types, sizes, _MIXED_X, labels, True)
    assert torch.isnan(got_loss).any() and torch.isinf(got_loss).any(), (
        "the loss route sanitized (or lost) the NaN/inf rows"
    )
    te.assert_close_with_placement(
        got_loss, want_loss, context="hostile loss variant vs specialized"
    )
    # the loss is NOT sanitized: the caller negates and the pipeline maps NaN
    te.assert_close_with_placement(
        got_loss,
        te.reference_sr_fitness(_forest(_MIXED_SPECS), _MIXED_X, labels, True),
        context="hostile loss variant vs the independent oracle",
    )


# =========================================================================== #
# 4. the output contract (fresh, writable, float32, device, no mutation)
# =========================================================================== #
@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_value_output_contract(per_row: bool) -> None:
    """``(pop, out_len)`` / ``(pop, N, out_len)``, fresh and float32, on ``x``'s device."""
    values, types, sizes = _rows(_PROGRAMS)
    x = _x(len(_PROGRAMS), seed=43) if per_row else _x(N_ROWS, seed=43)

    got = _value(values, types, sizes, x, per_row=per_row)
    shape = (len(_PROGRAMS), 1) if per_row else (len(_PROGRAMS), N_ROWS, 1)
    _assert_fresh_tensor(got, [("x", x), ("node_value", values)], shape=shape)
    assert got.device == x.device, f"{got.device} != {x.device}"

    again = _value(values, types, sizes, x, per_row=per_row)
    assert got.untyped_storage().data_ptr() != again.untyped_storage().data_ptr(), (
        "two calls handed back the same storage: the result is not fresh"
    )


def test_loss_output_contract() -> None:
    """``(pop,)`` float32 on ``inputs``' device, fresh, writable, grad-free."""
    values, types, sizes = _rows(_PROGRAMS)
    inputs = _x(N_ROWS, seed=47)
    labels = _labels(N_ROWS, seed=53)

    got = _loss(values, types, sizes, inputs, labels, True)
    _assert_fresh_tensor(
        got, [("inputs", inputs), ("labels", labels)], shape=(len(_PROGRAMS),)
    )
    assert got.device == inputs.device, f"{got.device} != {inputs.device}"

    again = _loss(values, types, sizes, inputs, labels, True)
    assert got.untyped_storage().data_ptr() != again.untyped_storage().data_ptr(), (
        "two calls handed back the same storage: the result is not fresh"
    )


def test_out_device_overrides_the_output_placement() -> None:
    """``out_device`` is honoured on both entry points (CPU->CPU is enough here).

    A real cross-device check needs a second torch device, so this test only pins
    that the argument is applied instead of being ignored; the cross-device
    behaviour lives in the route's dev batteries.
    """
    values, types, sizes = _rows(_PROGRAMS)
    inputs = _x(N_ROWS, seed=59)
    labels = _labels(N_ROWS, seed=61)

    got = _value(values, types, sizes, inputs, per_row=False, out_device="cpu")
    assert got.device == torch.device("cpu")
    te.assert_close_with_placement(
        got,
        _value(values, types, sizes, inputs, per_row=False),
        context="value variant with out_device='cpu'",
    )

    got_loss = _loss(values, types, sizes, inputs, labels, True, out_device="cpu")
    assert got_loss.device == torch.device("cpu")
    te.assert_close_with_placement(
        got_loss,
        _loss(values, types, sizes, inputs, labels, True),
        context="loss variant with out_device='cpu'",
    )


@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_inputs_and_population_are_never_mutated(per_row: bool) -> None:
    """Neither entry point may write through the caller's tensors.

    The population is the caller's live buffer in a GP loop, so an in-place touch
    (a cast, a ``zero_()``, a scratch write) would corrupt the next generation
    silently.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    x = _x(len(_DUPLICATE_POP), seed=67) if per_row else _x(N_ROWS, seed=67)
    inputs = _x(N_ROWS, seed=71)
    labels = _labels(N_ROWS, seed=73)
    sources = (values, types, sizes, x, inputs, labels)
    clones = [tensor.clone() for tensor in sources]

    _value(values, types, sizes, x, per_row=per_row)
    _loss(values, types, sizes, inputs, labels, True)
    assert _loss(values, types, sizes, inputs, labels, False) is not None

    names = ("node_value", "node_type", "subtree_size", "x", "inputs", "labels")
    for tensor, clone, name in zip(sources, clones, names):
        assert torch.equal(tensor, clone), f"{name} was mutated by the route"


# =========================================================================== #
# 5. duplicate expansion: every row is still ITS program's answer
# =========================================================================== #
@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_duplicate_rows_are_merged_once_and_expanded_exactly(per_row: bool) -> None:
    """16 rows of 4 programs: 4 merged functions, 12 collisions, correct expansion.

    Each row is compared against the ORACLE's answer for its own program (a
    one-program ``Forest``), so a permuted ``row_index`` or a row served from the
    wrong function fails here even though the value would still be "some
    program's output". The constant tree is part of the fixture.
    """
    specs = _DUPLICATE_POP
    values, types, sizes = _rows(specs)
    # ``per_row=True`` keys the dedup on the row's own ``x`` bytes, so the x rows
    # are grouped the same way the programs are (one row per program, repeated).
    x = (
        _x(len(_PROGRAMS), seed=73).repeat(len(specs) // len(_PROGRAMS), 1)
        if per_row
        else _x(N_ROWS, seed=73)
    )

    plan = batch_engine.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=per_row,
        backend=BACKEND,
    )
    info = batch_engine.describe_plan(plan)
    assert info["pop_size"] == len(specs)
    assert info["functions"] == len(_PROGRAMS), info["functions"]
    assert info["const_functions"] == 1, "the constant program is not counted as one"
    assert info["row_index_collisions"] == len(specs) - len(_PROGRAMS)
    assert info["row_index_collisions"] == info["pop_size"] - info["functions"]

    got = _value(values, types, sizes, x, per_row=per_row)
    # The oracle's answer for EVERY program (a one-program Forest), then the row's
    # own program picked out: per_row picks the data row, the shared-x layout keeps
    # the whole ``(N, out_len)`` block.
    per_program = [
        te.reference_batch_forward(_forest([spec]), x)[0] for spec in _PROGRAMS
    ]
    for row, index in enumerate(plan.row_index):
        want = per_program[index][row] if per_row else per_program[index]
        te.assert_close_with_placement(
            got[row],
            want,
            context=f"row {row} of per_row={per_row} vs its program {index}",
        )


def test_per_row_dedup_keys_on_the_input_row() -> None:
    """``per_row=True`` must NOT share a program across two different ``x`` rows.

    Tree ``i`` sees row ``i``; sharing the two evaluations would serve one row's
    numbers to the other. 16 rows whose programs repeat 4x, but with DISTINCT x
    rows, therefore merge only where a constant tree makes ``x`` irrelevant.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    x = _x(len(_DUPLICATE_POP), seed=79)

    plan = batch_engine.plan_batch(
        values, types, sizes, INPUT_LEN, 1, x, per_row=True, backend=BACKEND
    )
    info = batch_engine.describe_plan(plan)
    constant_rows = 1  # the ``te.const(-2.25)`` program appears 4x
    assert info["const_functions"] == constant_rows
    assert info["functions"] == len(_DUPLICATE_POP) - len(_DUPLICATE_POP) // 4 + 1
    assert info["row_index_collisions"] == len(_DUPLICATE_POP) // 4 - 1

    got = _value(values, types, sizes, x, per_row=True)
    per_program = [_forest([spec]) for spec in _PROGRAMS]
    for row in range(len(_DUPLICATE_POP)):
        want = te.reference_batch_forward(
            per_program[row % len(_PROGRAMS)], x[row : row + 1]
        )[0, 0]
        te.assert_close_with_placement(
            got[row], want, context=f"per_row row {row} on its own x row"
        )


# =========================================================================== #
# 6. the merge knobs: bands / switch_ops change the shape, never the numbers
# =========================================================================== #
@pytest.mark.parametrize("knobs,expected_bands,expected_strategy", _BAND_CASES)
def test_bands_do_not_change_the_loss(
    knobs: Dict[str, Any], expected_bands: int, expected_strategy: str
) -> None:
    """Every band spelling is numerically the flat merge (``bands`` is a cost knob).

    Bands trade dispatches for wasted work inside the merged graph; a band that
    mixed two functions' slots would change the value. ``describe_plan`` reports
    the band count/strategy and per-band sizes that sum to the function count.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    inputs = _x(N_ROWS, seed=83)
    labels = _labels(N_ROWS, seed=89)

    baseline = _loss(values, types, sizes, inputs, labels, True)
    got = _loss(values, types, sizes, inputs, labels, True, **knobs)
    assert torch.equal(got, baseline), (
        f"{knobs} changed the loss: max|diff| = "
        f"{float((got - baseline).abs().max()):.3g}"
    )

    plan = batch_engine.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        per_row=False,
        variant=batch_engine.LOSS_VARIANT,
        use_mse=True,
        labels=labels,
        backend=BACKEND,
        **knobs,
    )
    info = batch_engine.describe_plan(plan)
    assert info["band_count"] == expected_bands, f"{knobs}: {info['band_count']}"
    assert info["band_strategy"] == expected_strategy, f"{knobs}: {info['band_strategy']}"
    assert sum(info["band_sizes"]) == info["n_functions"] == info["functions"]
    assert len(info["band_sizes"]) == info["band_count"]


@pytest.mark.parametrize("knobs,expected_bands,expected_strategy", _BAND_CASES)
def test_bands_do_not_change_the_per_row_values(
    knobs: Dict[str, Any], expected_bands: int, expected_strategy: str
) -> None:
    """The value variant's per-row layout is band-invariant too, and so is its plan.

    ``switch_ops`` forces the flat merge (``band_count == 1``) by construction, so
    the case table's expectation is checked rather than assumed.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    x = _x(len(_PROGRAMS), seed=91).repeat(len(_DUPLICATE_POP) // len(_PROGRAMS), 1)

    baseline = _value(values, types, sizes, x, per_row=True)
    got = _value(values, types, sizes, x, per_row=True, **knobs)
    assert torch.equal(got, baseline), (
        f"{knobs} changed the value output: max|diff| = "
        f"{float((got - baseline).abs().max()):.3g}"
    )

    plan = batch_engine.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=True,
        backend=BACKEND,
        **knobs,
    )
    info = batch_engine.describe_plan(plan)
    assert info["band_count"] == expected_bands, f"{knobs}: {info['band_count']}"
    assert info["band_strategy"] == expected_strategy, f"{knobs}: {info['band_strategy']}"
    assert info["switch_ops"] is bool(knobs.get("switch_ops", False))
    assert sum(info["band_sizes"]) == info["n_functions"] == len(_PROGRAMS)


def test_switch_ops_forces_the_flat_merge() -> None:
    """``switch_ops=True`` is never combined with banding (``K = 1`` always).

    Precedence is ``switch_ops`` -> explicit ``bands`` -> the band-size model, so
    asking for both must still produce one band, and the numbers must not move.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    inputs = _x(N_ROWS, seed=97)
    labels = _labels(N_ROWS, seed=101)

    plan = batch_engine.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        per_row=False,
        variant=batch_engine.LOSS_VARIANT,
        labels=labels,
        switch_ops=True,
        bands=4,
        backend=BACKEND,
    )
    info = batch_engine.describe_plan(plan)
    assert info["band_count"] == 1, f"switch_ops did not force the flat merge: {info}"
    assert info["switch_ops"] is True
    assert info["band_sizes"] == [info["n_functions"]]


# =========================================================================== #
# 7. describe_plan: the report of a known fixture
# =========================================================================== #
def test_describe_plan_reports_the_known_duplicate_fixture() -> None:
    """Every documented field of the report, on a population whose answer is known.

    16 rows / 4 distinct programs / 1 fully constant program, ``use_mse=False``
    (the value must travel into the report, not be normalized away).
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    inputs = _x(N_ROWS, seed=103)
    labels = _labels(N_ROWS, seed=107)

    plan = batch_engine.plan_batch(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        inputs,
        per_row=False,
        variant=batch_engine.LOSS_VARIANT,
        use_mse=False,
        labels=labels,
        backend=BACKEND,
    )
    info = batch_engine.describe_plan(plan)

    assert set(info) >= _DESCRIBE_KEYS, sorted(_DESCRIBE_KEYS - set(info))
    assert info["variant"] == batch_engine.LOSS_VARIANT
    assert info["per_row"] is False
    assert info["use_mse"] is False
    assert info["pop_size"] == len(_DUPLICATE_POP)
    assert info["functions"] == len(_PROGRAMS)
    assert info["const_functions"] == 1
    assert info["row_index_collisions"] == len(_DUPLICATE_POP) - len(_PROGRAMS)
    assert info["input_len"] == INPUT_LEN
    assert info["output_len"] == 1
    assert info["n_rows"] == N_ROWS
    assert info["n_functions"] == len(_PROGRAMS)
    assert info["band_count"] == 1 and info["band_sizes"] == [len(_PROGRAMS)]
    assert info["switch_ops"] is False and info["alignment"] == batch_engine.DEFAULT_ALIGNMENT
    assert isinstance(info["emitted_ops"], int) and info["emitted_ops"] > 0
    assert info["backend"] == BACKEND
    assert "cpu" in str(info["device"]), info["device"]
    # the graph inputs: the data block, then the label block (loss variant only)
    assert info["specs"] == [(N_ROWS, INPUT_LEN), (N_ROWS, 1)], info["specs"]
    assert info["spec_dtypes"] == ["float32", "float32"]


@pytest.mark.parametrize("per_row", [True, False], ids=["per_row", "shared_x"])
def test_describe_plan_reports_the_value_layout(per_row: bool) -> None:
    """The value variant has no metric and its input spec follows ``per_row``.

    ``per_row=True`` stages ONE input row per merged function
    (``(n_functions, 1, input_len)``), every other layout one shared
    ``(n_rows, input_len)`` block -- exactly the staging the runner feeds.
    """
    values, types, sizes = _rows(_PROGRAMS)
    x = _x(len(_PROGRAMS), seed=109) if per_row else _x(N_ROWS, seed=109)

    plan = batch_engine.plan_batch(
        values, types, sizes, INPUT_LEN, 1, x, per_row=per_row, backend=BACKEND
    )
    info = batch_engine.describe_plan(plan)

    assert info["variant"] == batch_engine.VALUE_VARIANT
    assert info["per_row"] is per_row
    assert info["use_mse"] is None, "the value variant has no error metric"
    assert info["n_rows"] == (1 if per_row else N_ROWS)
    expected = [(len(_PROGRAMS), 1, INPUT_LEN)] if per_row else [(N_ROWS, INPUT_LEN)]
    assert info["specs"] == expected, info["specs"]
    assert info["spec_dtypes"] == ["float32"]


# =========================================================================== #
# 8. build_batch: memoized compile, re-bound plan
# =========================================================================== #
def test_build_batch_rebinds_a_memoized_executable() -> None:
    """A second build reuses the compiled body but carries the CALLER's plan.

    The executable is keyed on the merged program multiset, while pop size, row
    mapping and staged inputs always belong to the current plan -- a returned
    executable whose ``.plan`` were the first population's is exactly the stale
    ``(4,)`` bug the next test pins from the public side.
    """
    values, types, sizes = _rows(_DUPLICATE_POP)
    inputs = _x(N_ROWS, seed=113)

    plan = batch_engine.plan_batch(
        values, types, sizes, INPUT_LEN, 1, inputs, per_row=False, backend=BACKEND
    )
    first = batch_engine.build_batch(plan)
    second = batch_engine.build_batch(plan)

    assert first.plan is plan and second.plan is plan
    assert first.backend == BACKEND and second.backend == BACKEND
    assert first.executable is second.executable, (
        "a repeated build re-lowered/re-compiled instead of hitting the memo"
    )


# =========================================================================== #
# 9. THE REGRESSION: an all-constant population across pop sizes, one process
# =========================================================================== #
def test_all_constant_population_survives_a_pop_size_escalation() -> None:
    """REGRESSION: pop 4 -> 64 -> 256 of the SAME constant in ONE process.

    Every row of an all-constant population merges into ONE function, so the
    compiled body is byte-identical across the three populations while the row
    expansion is not. A memoized executable that was handed back with the FIRST
    population's plan produced a stale ``(4,)`` loss for the 64/256-row calls (and
    an earlier variant declined with a wrong-pop-size message) -- both are silent,
    so each call is asserted on its own SHAPE and on its own VALUES here.
    """
    inputs = _x(5, seed=127)
    labels = _labels(5, seed=131)
    expected_loss = float(((CONST_VALUE - labels[:, 0]) ** 2).sum() / labels.shape[0])

    for pop_size in (4, 64, 256):
        values, types, sizes = _constant_rows(pop_size)

        loss = _loss(values, types, sizes, inputs, labels, True)
        assert tuple(loss.shape) == (pop_size,), (
            f"pop {pop_size}: the loss came back as {tuple(loss.shape)} (a stale "
            f"plan from an earlier smaller population)"
        )
        assert torch.allclose(loss, torch.full_like(loss, expected_loss), rtol=1e-6), (
            f"pop {pop_size}: loss {loss[0].item()} != {expected_loss}"
        )

        out = _value(values, types, sizes, inputs, per_row=False)
        assert tuple(out.shape) == (pop_size, inputs.shape[0], 1), (
            f"pop {pop_size}: value output {tuple(out.shape)}"
        )
        assert torch.equal(out, torch.full_like(out, CONST_VALUE)), (
            f"pop {pop_size}: a constant tree did not return its constant"
        )


# =========================================================================== #
# 10. declines: the fallback contract
# =========================================================================== #
def test_batch_unsupported_is_a_declining_runtime_error() -> None:
    """A decline is a ``RuntimeError`` the five call sites can catch and fall back on."""
    assert issubclass(batch_engine.BatchUnsupported, RuntimeError)


def test_population_below_the_threshold_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pop_size < EVOGP_BATCH_MIN_POP`` declines with the variable named.

    The default threshold is the measured break-even, not a safety floor: a
    merged compile cannot amortize over a handful of functions, so the caller must
    be told to fall back rather than pay 0.45 s for four trees.
    """
    monkeypatch.delenv(batch_engine.MIN_POP_ENV, raising=False)
    values, types, sizes = _rows(_PROGRAMS)
    inputs = _x(N_ROWS, seed=137)
    labels = _labels(N_ROWS, seed=139)

    with pytest.raises(batch_engine.BatchUnsupported) as excinfo:
        _loss(values, types, sizes, inputs, labels, True)
    message = str(excinfo.value)
    assert batch_engine.MIN_POP_ENV in message, message
    assert str(len(_PROGRAMS)) in message, message

    with pytest.raises(batch_engine.BatchUnsupported):
        _value(values, types, sizes, _x(len(_PROGRAMS)), per_row=True)


def test_empty_population_and_a_shape_doubt_decline() -> None:
    """An empty population and a ``per_row``/``x`` shape doubt are declines, never errors.

    Both are "this route cannot handle it" answers: the caller falls back to the
    specialized engine, which owns the ``ValueError`` contract. An empty
    population must not be reported as merely "too small".
    """
    empty_values = torch.zeros((0, ROW_LEN), dtype=torch.float32)
    empty_types = torch.zeros((0, ROW_LEN), dtype=torch.int16)
    empty_sizes = torch.zeros((0, ROW_LEN), dtype=torch.int16)
    inputs = _x(N_ROWS, seed=149)
    labels = _labels(N_ROWS, seed=151)

    with pytest.raises(batch_engine.BatchUnsupported) as excinfo:
        _loss(empty_values, empty_types, empty_sizes, inputs, labels, True)
    assert "empty population" in str(excinfo.value), str(excinfo.value)

    values, types, sizes = _rows(_PROGRAMS)
    # per_row=True needs exactly pop_size input rows ...
    with pytest.raises(batch_engine.BatchUnsupported) as excinfo:
        _value(values, types, sizes, _x(len(_PROGRAMS) + 1), per_row=True)
    assert "per_row" in str(excinfo.value), str(excinfo.value)
    # ... and input_len columns.
    with pytest.raises(batch_engine.BatchUnsupported):
        _value(values, types, sizes, _x(N_ROWS)[:, : INPUT_LEN - 1], per_row=False)
    # an unknown backend name is a decline too (this route does not build it)
    with pytest.raises(batch_engine.BatchUnsupported):
        batch_engine.sr_fitness_batch(
            values,
            types,
            sizes,
            INPUT_LEN,
            1,
            inputs,
            labels,
            True,
            backend="tpu",
        )


# =========================================================================== #
# 11. the one device leg (SKIPS here: no IREE packages in this venv)
# =========================================================================== #
@requires_device_iree
def test_iree_batch_matches_the_iree_dense_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a device box the batch loss equals the dense route's, both on iree.

    Both sides run the SAME backend class, so this compares the two compiled
    implementations (merged per-function graph vs one all-ops graph) rather than
    the iree-vs-numpy lowering difference, with the usual mask-first tolerance.
    The fixture stays at 4 distinct merged functions so the two compiles remain a
    few seconds on a box that has iree.
    """
    monkeypatch.delenv("EVOGP_EVOXC_BACKEND", raising=False)
    values, types, sizes = _rows(_DUPLICATE_POP)
    inputs = _x(N_ROWS, seed=157)
    labels = _labels(N_ROWS, seed=163)

    got = batch_engine.sr_fitness_batch(
        values, types, sizes, INPUT_LEN, 1, inputs, labels, True, backend="iree"
    )
    want = dense_engine.sr_fitness_dense(
        values, types, sizes, INPUT_LEN, 1, inputs, labels, True, backend="iree"
    )
    assert tuple(got.shape) == (len(_DUPLICATE_POP),), f"{tuple(got.shape)}"
    te.assert_close_with_placement(
        got, want, context="batch+iree loss vs dense+iree loss"
    )
