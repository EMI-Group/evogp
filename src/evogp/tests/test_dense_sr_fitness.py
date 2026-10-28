"""Dense SR-loss contract + ``(pop_size,)``-scale host-traffic regression tests.

What this file pins
-------------------
The SR-loss entry point of the dense evaluator -- ``dense_engine.sr_fitness_dense``
(``src/evogp/tree/dense_engine.py``), reached by ``Forest.SR_fitness`` when the dense
route is selected. Its contract is *not* "the dense prediction reduced somewhere":
the loss is folded into the compiled graph, so

* the returned tensor is a ``(pop_size,)`` float32 POSITIVE loss
  ``sum_N sum_out err / N`` (MSE or MAE) on ``inputs``' device unless ``out_device``,
  with NaN/±inf propagated unsanitized and NaN/±inf *placement* identical to the
  independent plain-torch oracle (``reference_eval.py``) and to the specialized
  (per-structure) engine;
* ``inputs`` and ``labels`` are each converted to host ONCE per call, ``x``/``labels``
  are re-prepared once per data chunk (never once per ``(p0, n0)`` chunk), and the
  ONLY data that crosses back is the per-chunk ``(p_chunk,)`` loss sum -- never the
  ``(pop_size, N)`` prediction, and never a labels-shaped tensor through a torch
  tensor;
* chunking (a tiny ``budget_mb``) changes the reduction order only: the chunked loss
  equals the single-chunk loss, including a padded last chunk whose row count is not
  a multiple of the chunk size.

This is a pytest PORT of the ``hotpath`` / ``parity`` stages of
``src/evogp/tree/device_resident_check.py`` (the dev battery documented in
``src/evogp/tree/device_resident_findings.md`` §1/§2/§6/§8): the same counters on the
same patched sites, the same caller-frame filters, the same comparison policy
(``rtol = atol = 1e-4`` with exact NaN/±inf masks; bit-exact for a perfect fit and for
the unread row tail; ``1e-6`` for chunked-vs-single-chunk). The battery stays the
measurement record -- this file is the regression gate, so it contains
**no wall-clock assertion anywhere** (counts only) and never executes the ``iree``
backend (``backend="numpy"`` everywhere; a device backend would compile per shape).

Hermeticity / scope
-------------------
* small shapes only: ``pop_size <= 8``, ``N <= 13``, ``L <= 16``, numpy backend;
* every knob that would change the plan (``backend`` / ``budget_mb`` /
  ``max_tree_len``) is passed EXPLICITLY, and a MODULE-scoped autouse fixture pins
  the ambient ones for the whole file -- ``EVOGP_EVAL_MODE=dense`` +
  ``EVOGP_EVOXC_BACKEND=numpy`` plus the two plan-level knobs
  ``EVOGP_DENSE_UPDATE_MODE`` / ``EVOGP_DENSE_PRUNE`` (at their library defaults),
  the module scope being what makes the module-scoped plan fixtures and the call
  sites agree -- so no ambient ``EVOGP_DENSE_*`` / ``EVOGP_EVOXC_*`` value can change
  what is measured;
* no file outside ``src/evogp/tests/`` is touched, and ``reference_eval.py`` (the
  oracle) is used as-is -- never re-implemented here.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src <venv>/bin/python -m pytest \
        src/evogp/tests/test_dense_sr_fitness.py -q -p no:cacheprovider
    EVOGP_EVAL_MODE=dense EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src <venv>/bin/python \
        -m pytest src/evogp/tests/test_dense_sr_fitness.py -q -p no:cacheprovider
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

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

import evoxc  # noqa: E402  (the engine dependency; only its patching sites are named)
from evogp.tree import Forest  # noqa: E402
from evogp.tree import dense_engine  # noqa: E402
from evogp.tree import evoxc_engine  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: the whole file must stay a few seconds)
# --------------------------------------------------------------------------- #
BACKEND = "numpy"
INPUT_LEN = 3
ROW_LEN = 16  # hand-built population rows -> the dense trace is L = min(cap, 16)
#: Explicit trace cap: wins over ``EVOGP_DENSE_MAX_TREE_LEN``, so
#: ``L = min(32, row_len)`` is deterministic and every cache key is stable.
MAX_TREE_LEN = 32
#: Data points. 13 is deliberately not a multiple of the tuned chunk sizes, so the
#: padded-last-chunk case is reachable.
N_ROWS = 13
#: The library's own default budget -> ONE chunk (the realistic regime).
UNCHUNKED_BUDGET_MB = 2048.0
#: Budgets the plan search walks (the battery's ``_CANDIDATE_BUDGETS``), from the
#: SMALLEST budget up; :func:`_tuned_plan` walks them downwards (see there).
_BUDGET_CANDIDATES = (
    0.0004,
    0.0006,
    0.0008,
    0.0012,
    0.002,
    0.004,
    0.008,
    0.016,
    0.05,
    0.2,
    1.0,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="this check needs a CUDA device: Forest.SR_fitness relocates its inputs "
    "to 'cuda' (check_tensor) and the device-placement check is only meaningful "
    "with a second torch device",
)


def _pinned_environment() -> Dict[str, str]:
    """The environment every test of this file runs under (see the fixture below)."""
    return {
        dense_engine.EVAL_MODE_ENV: "dense",
        "EVOGP_EVOXC_BACKEND": BACKEND,
        dense_engine.UPDATE_MODE_ENV: dense_engine.DEFAULT_UPDATE_MODE,
        dense_engine.PRUNE_ENV: "1" if dense_engine.DEFAULT_PRUNE else "0",
    }


@pytest.fixture(autouse=True, scope="module")
def _pin_route_and_backend() -> Iterator[None]:
    """Pin every ambient knob that could change what this file measures, for the module.

    ``EVOGP_EVAL_MODE=dense``: ``sr_fitness_dense`` consults the mode exactly once
    (``auto_rows_decline``), so an ambient ``EVOGP_DENSE_AUTO_MAX_ROWS`` below ``N``
    would otherwise turn a direct dense call into a decline. ``dense`` is never
    declined, so the file behaves identically under the default run and under the
    forced-dense run of the suite.

    ``EVOGP_EVOXC_BACKEND=numpy``: the wired entry points (``Forest.SR_fitness`` in the
    public-route test) resolve the backend from the environment; no test in this file
    may compile an iree graph.

    ``EVOGP_DENSE_UPDATE_MODE`` / ``EVOGP_DENSE_PRUNE``: the buffer write strategy and
    the candidate-chain pruning are plan-level knobs neither the helpers here nor the
    public call sites pass explicitly, so an ambient value would silently change the
    compiled graph (and its cfg cache key) under every assertion in the file. They are
    pinned to the library defaults (``dense_engine.DEFAULT_UPDATE_MODE`` /
    ``DEFAULT_PRUNE``), not merely cleared.

    The scope is MODULE, not function, on purpose: the plan-building fixtures below
    (``unchunked_plan`` / ``both_chunked_plan``) are module-scoped, and pytest
    instantiates them BEFORE any function-scoped autouse fixture -- so a function-scoped
    pin would leave the fixture plans planned under the ambient environment while the
    call sites run under the pinned one, and ``EVOGP_DENSE_UPDATE_MODE`` changes the
    peak-bytes model (``buf_factor``), i.e. the chunk shape, so that is a disagreement
    rather than a pin. The four values are saved and restored around the module, so the
    rest of the suite is unaffected.
    """
    pins = _pinned_environment()
    saved = {name: os.environ.get(name) for name in pins}
    os.environ.update(pins)
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


def _labels(n: int, out_len: int = 1, seed: int = 101) -> torch.Tensor:
    """Deterministic labels from a LOCAL generator (no global-RNG coupling)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, out_len), generator=generator, dtype=torch.float32)


def _forest(
    specs: Sequence[Any], *, input_len: int = INPUT_LEN, row_len: int = ROW_LEN
) -> Forest:
    """A single-output ``Forest`` of hand-built trees with a CLEAN junk tail.

    ``te.rows_from_specs`` writes CONST/0.0 with subtree size 0 into the unused tail,
    which is the "clean" reference of the row-tail test.
    """
    values, types, sizes = te.rows_from_specs(specs, 1, pad_to=row_len, tail_value=0.0)
    return Forest(input_len, 1, values, types, sizes)


#: Hostile data: ``log`` of a negative (NaN), ``exp(100)`` (overflow to +inf) and a
#: zero divisor (NaN) -- hand-built, so the mixed population is deterministic.
_MIXED_X = torch.tensor(
    [
        [-1.0, 0.5, 2.0],
        [100.0, 1.0, -0.5],
        [0.5, -2.0, 3.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)

#: A MIXED population: finite, NaN and ±inf losses side by side (row by row).
_MIXED_SPECS = [
    te.fn(te.ADD, te.var(0), te.var(1)),  # finite
    te.fn(te.LOG, te.var(0)),  # NaN loss (log of a negative row)
    te.fn(te.EXP, te.var(0)),  # +inf loss (exp overflow)
    te.const(1.0),  # finite (constant-folded subtree)
    te.fn(te.DIV, te.var(1), te.const(0.0)),  # NaN loss (division by an exact 0)
    te.fn(te.SUB, te.const(0.0), te.fn(te.EXP, te.var(0))),  # -inf prediction
    te.var(2),  # finite
]

#: A fully FINITE population (used where a compare must not be vacuous).
_FINITE_SPECS = [
    te.fn(te.ADD, te.var(0), te.var(1)),
    te.fn(te.MUL, te.var(2), te.const(1.5)),
    te.var(0),
    te.const(-2.25),
    te.fn(te.SIN, te.var(1)),
    te.fn(te.MUL, te.var(1), te.const(-0.75)),
    te.fn(te.SUB, te.fn(te.MUL, te.var(0), te.var(1)), te.const(0.5)),
    te.fn(te.TANH, te.var(2)),
]

#: The one-tree row-tail fixture: 3 nodes in an 8-wide row, i.e. 5 junk slots.
_TAIL_SPECS = [te.fn(te.ADD, te.var(0), te.var(1))]
_TAIL_ROW_LEN = 8


def _mixed_case() -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    return _forest(_MIXED_SPECS), _MIXED_X.clone(), _labels(_MIXED_X.shape[0], seed=21)


def _finite_case() -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    return (
        _forest(_FINITE_SPECS),
        te.x_random(N_ROWS, INPUT_LEN, seed=5),
        _labels(N_ROWS, seed=6),
    )


def _random_case() -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    """A genuinely randomly generated population (real GP structures).

    ``Forest.random_generate`` draws through ``GenerateDescriptor``, whose parameter
    tensors are hard-coded to ``"cuda"`` in this library, so the case skips on a
    CPU-only box (the same boundary every other random-population test has).
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "Forest.random_generate builds GenerateDescriptor, whose parameter "
            "tensors are hard-coded to the 'cuda' device"
        )
    _, forest = te.make_random_population(
        pop_size=6, seed=7, max_tree_len=16, input_len=INPUT_LEN, const_prob=0.3
    )
    return forest, te.x_random(9, INPUT_LEN, seed=8), _labels(9, seed=9)


_CASES: Dict[str, Callable[[], Tuple[Forest, torch.Tensor, torch.Tensor]]] = {
    "mixed": _mixed_case,
    "finite": _finite_case,
    "random": _random_case,
}


# --------------------------------------------------------------------------- #
# the call sites under test / plan search
# --------------------------------------------------------------------------- #


def _dense_loss(
    forest: Forest,
    x: torch.Tensor,
    labels: torch.Tensor,
    use_mse: bool = True,
    *,
    budget_mb: float = UNCHUNKED_BUDGET_MB,
    out_device: Optional[Any] = None,
) -> torch.Tensor:
    """The loss-route call site under test, with the shape grid pinned.

    ``backend`` / ``budget_mb`` / ``max_tree_len`` are explicit arguments (they win
    over the environment), so an ambient ``EVOGP_DENSE_*`` knob can never change the
    plan the assertions were written for.
    """
    return dense_engine.sr_fitness_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        x,
        labels,
        use_MSE=use_mse,
        out_device=out_device,
        backend=BACKEND,
        max_tree_len=MAX_TREE_LEN,
        budget_mb=budget_mb,
    )


def _plan(
    forest: Forest, rows: int, *, budget_mb: float, use_mse: bool = True
) -> Any:
    """``plan_dense`` for the LOSS variant (host-side only: no graph is built)."""
    return dense_engine.plan_dense(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
        rows,
        per_row=False,
        variant=dense_engine.LOSS_VARIANT,
        use_mse=use_mse,
        max_tree_len=MAX_TREE_LEN,
        budget_mb=budget_mb,
        backend=BACKEND,
    )


def _plan_line(plan: Any) -> str:
    return (
        f"p_chunk={plan.p_chunk} n_chunk={plan.n_chunk} "
        f"p_chunks={plan.p_chunks} n_chunks={plan.n_chunks} L={plan.trace_len}"
    )


def _tuned_plan(
    forest: Forest, rows: int, what: str, predicate: Callable[[Any], bool]
) -> Tuple[float, Any]:
    """The first candidate budget whose plan satisfies ``predicate``.

    Ported from ``device_resident_check._tuned_budget``: the chunk shape is a
    function of the budget, so the tests ASK for the plan they need instead of
    hard-coding a budget a future chunk-shape change would invalidate.

    The candidates are walked from the LARGEST budget down, i.e. the cheapest plan
    that still exhibits the regime is picked. The battery walks them upwards because
    it also reports the extreme (one element per chunk) regime; a regression test
    only needs a witness of the regime, and chunk count dominates the runtime here.
    """
    for budget in reversed(_BUDGET_CANDIDATES):
        plan = _plan(forest, rows, budget_mb=budget)
        if predicate(plan):
            return budget, plan
    raise AssertionError(
        f"no candidate budget produces a {what} loss plan; tried {_BUDGET_CANDIDATES}"
    )


@pytest.fixture(scope="module")
def unchunked_plan() -> Tuple[float, Any]:
    """The default-budget (single-chunk) plan of the finite fixture."""
    forest, x, _ = _finite_case()
    plan = _plan(forest, x.shape[0], budget_mb=UNCHUNKED_BUDGET_MB)
    assert plan.p_chunks == 1 and plan.n_chunks == 1, _plan_line(plan)
    return UNCHUNKED_BUDGET_MB, plan


@pytest.fixture(scope="module")
def both_chunked_plan() -> Tuple[float, Any]:
    """A plan chunked in BOTH dimensions (the regime the traffic invariants need)."""
    forest, x, _ = _finite_case()
    return _tuned_plan(
        forest,
        x.shape[0],
        "P- AND N-chunked",
        lambda plan: plan.p_chunks > 1 and plan.n_chunks > 1,
    )


@pytest.fixture(scope="module")
def padded_plan() -> Tuple[float, Any]:
    """A row-chunked plan whose LAST chunk is partial (``N % n_chunk != 0``)."""
    forest, x, _ = _finite_case()
    return _tuned_plan(
        forest,
        x.shape[0],
        "padded-last-chunk",
        lambda plan: plan.n_chunk > 1
        and plan.n_chunks > 1
        and x.shape[0] % plan.n_chunk != 0,
    )


# --------------------------------------------------------------------------- #
# counted host crossings (pytest port of ``device_resident_check.Traffic``)
# --------------------------------------------------------------------------- #
#: callers that make up the dense run machinery's READBACK path
_READBACK_FRAMES = ("run_plan", "run_loss_plan", "_to_host", "_iter_plan_chunks")
#: callers that invoke the compiled graph (one call per chunk is the invariant)
_RUN_FRAMES = ("run_plan", "run_loss_plan", "_iter_plan_chunks")
#: callers that make up its host -> torch assembly path
_DENSE_FRAMES = (
    "evaluate_dense",
    "sr_fitness_dense",
    "run_plan",
    "run_loss_plan",
    "_to_host",
    "_iter_plan_chunks",
)


def _caller(depth: int = 2) -> str:
    """Name of the function that called the PATCHED site (one frame above the wrapper)."""
    return sys._getframe(depth).f_code.co_name


class _HostCrossings:
    """Every host<->torch crossing of ONE dense loss call, by site and caller.

    A pytest port of ``Traffic`` in ``src/evogp/tree/device_resident_check.py``: the
    same six patched sites, one of them SWAPPED -- ``dense_engine._as_numpy`` /
    ``_pad_rows`` / ``_to_host``, ``evoxc.core.Tensor.numpy``, ``evoxc.run``,
    ``torch.from_numpy`` here against the battery's ``dense_engine._as_numpy`` /
    ``_pad_rows``, ``evoxc.core.Tensor.numpy``, ``evoxc.run``, ``torch.from_numpy``
    and the upload counter (``evoxc.from_numpy`` + the ``.to(device)`` that follows
    it in ``_iter_plan_chunks``'s ``payload()`` -- evoxc's ``from_numpy`` is a
    zero-copy host wrap, so ``Tensor.to(device)`` is the real host->device UPLOAD)
    -- because the numpy backend never uploads anything:
    ``_to_host`` counts the readback/assembly crossing of that backend instead,
    which is what the loss route's host traffic consists of. The caller-frame names
    and the "the caller is always the first field" convention are the battery's.
    ``evoxc.core.Tensor.numpy`` is also called thousands of times inside the numpy
    interpreter's own arithmetic, which is why the readback view filters by caller
    frame.

    The wrappers are installed through ``monkeypatch`` and the originals are put back
    on exit, so one test may take several independent counts. Every instance starts
    from an empty baseline, so the counts are per-call deltas by construction and no
    compile-cache reset is needed anywhere in this file. COUNTS ONLY: nothing here
    measures time.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        #: ``(caller, payload name, elements)`` per torch -> host conversion
        self.as_numpy: List[Tuple[str, str, int]] = []
        #: ``(caller, shape)`` per evoxc -> host readback
        self.tensor_numpy: List[Tuple[str, Tuple[int, ...]]] = []
        #: ``(caller, shape)`` per host -> torch tensor creation
        self.torch_from_numpy: List[Tuple[str, Tuple[int, ...]]] = []
        #: ``(caller, pre-pad shape)`` per payload PREPARATION
        self.pad_rows: List[Tuple[str, Tuple[int, ...]]] = []
        #: ``(caller, executable type)`` per compiled-graph invocation
        self.run_calls: List[Tuple[str, str]] = []
        #: ``(caller, host shape)`` per ``_to_host`` result (what the loss route reads)
        self.to_host: List[Tuple[str, Tuple[int, ...]]] = []

    def __enter__(self) -> "_HostCrossings":
        origin_as_numpy = dense_engine._as_numpy
        origin_pad_rows = dense_engine._pad_rows
        origin_to_host = dense_engine._to_host
        origin_tensor_numpy = evoxc.core.Tensor.numpy
        origin_run = evoxc.run
        origin_torch_from_numpy = torch.from_numpy
        self._originals = (
            (dense_engine, "_as_numpy", origin_as_numpy),
            (dense_engine, "_pad_rows", origin_pad_rows),
            (dense_engine, "_to_host", origin_to_host),
            (evoxc.core.Tensor, "numpy", origin_tensor_numpy),
            (evoxc, "run", origin_run),
            (torch, "from_numpy", origin_torch_from_numpy),
        )

        def as_numpy(tensor: Any, dtype: Any, name: Any) -> Any:
            array = origin_as_numpy(tensor, dtype, name)
            self.as_numpy.append((_caller(), str(name), int(array.size)))
            return array

        def pad_rows(array: Any, rows: int) -> Any:
            self.pad_rows.append((_caller(), tuple(array.shape)))
            return origin_pad_rows(array, rows)

        def to_host(result: Any, backend_name: str) -> Any:
            host = origin_to_host(result, backend_name)
            self.to_host.append((_caller(), tuple(host.shape)))
            return host

        def tensor_numpy(self_: Any, *args: Any, **kwargs: Any) -> Any:
            result = origin_tensor_numpy(self_, *args, **kwargs)
            self.tensor_numpy.append((_caller(), tuple(self_.shape)))
            return result

        def run(executable: Any, *args: Any, **kwargs: Any) -> Any:
            self.run_calls.append((_caller(), type(executable).__name__))
            return origin_run(executable, *args, **kwargs)

        def torch_from_numpy(array: Any, *args: Any, **kwargs: Any) -> Any:
            self.torch_from_numpy.append((_caller(), tuple(array.shape)))
            return origin_torch_from_numpy(array, *args, **kwargs)

        self._monkeypatch.setattr(dense_engine, "_as_numpy", as_numpy)
        self._monkeypatch.setattr(dense_engine, "_pad_rows", pad_rows)
        self._monkeypatch.setattr(dense_engine, "_to_host", to_host)
        self._monkeypatch.setattr(evoxc.core.Tensor, "numpy", tensor_numpy)
        self._monkeypatch.setattr(evoxc, "run", run)
        self._monkeypatch.setattr(torch, "from_numpy", torch_from_numpy)
        return self

    def __exit__(self, *exc: Any) -> None:
        for holder, name, original in self._originals:
            setattr(holder, name, original)
        return None

    # -- views -------------------------------------------------------------
    @staticmethod
    def _filtered(records: Sequence[Any], frames: Sequence[str]) -> List[Any]:
        return [record for record in records if record[0] in frames]

    @property
    def readbacks(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """evoxc -> host readbacks issued by the dense run machinery."""
        return self._filtered(self.tensor_numpy, _READBACK_FRAMES)

    @property
    def runs(self) -> List[Tuple[str, str]]:
        """Compiled-graph invocations issued by the dense run machinery."""
        return [record for record in self.run_calls if record[0] in _RUN_FRAMES]

    @property
    def torch_creations(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Host -> torch tensor creations issued by the dense path."""
        return self._filtered(self.torch_from_numpy, _DENSE_FRAMES)

    def conversions(self, *names: str) -> List[Tuple[str, str, int]]:
        """``_as_numpy`` (torch -> host) calls whose payload name is in ``names``."""
        wanted = set(names)
        return [record for record in self.as_numpy if record[1] in wanted]

    def preparations(self, last_dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """``_pad_rows`` payload PREPARATIONS whose trailing axis is ``last_dim``.

        ``_pad_rows`` is private to ``dense_engine``, so the battery deliberately does
        not filter it by caller frame (a frame filter would stop counting the very
        calls the invariant is about). The trailing-axis filter is unambiguous for
        this file's shapes: ``L = 16`` for the node columns, ``input_len = 3`` for
        ``x`` and ``output_len = 1`` for ``labels``.
        """
        return [
            record
            for record in self.pad_rows
            if record[1] and record[1][-1] == last_dim
        ]


def _counted_loss(
    monkeypatch: pytest.MonkeyPatch,
    forest: Forest,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    budget_mb: float,
) -> Tuple[torch.Tensor, _HostCrossings]:
    """One dense SR-loss call with every host crossing counted."""
    with _HostCrossings(monkeypatch) as traffic:
        loss = _dense_loss(forest, x, labels, True, budget_mb=budget_mb)
    return loss, traffic


# =========================================================================== #
# 1. the loss contract vs the independent oracle (MSE + MAE)
# =========================================================================== #


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_loss_contract_matches_the_independent_oracle(case: str, use_mse: bool) -> None:
    """``(pop_size,)`` float32 POSITIVE ``sum_N err / N``, masks included."""
    forest, x, labels = _CASES[case]()
    context = f"dense SR loss ({case}, use_MSE={use_mse})"
    got = _dense_loss(forest, x, labels, use_mse)
    want = te.reference_sr_fitness(forest, x, labels, use_mse)

    assert tuple(got.shape) == (forest.pop_size,), f"{context}: shape {tuple(got.shape)}"
    assert got.dtype == torch.float32, f"{context}: dtype {got.dtype}"
    te.assert_close_with_placement(got, want, context=context)

    if case == "mixed":
        # The mask assertions above are vacuous unless the case stays MIXED.
        assert int(torch.isnan(got).sum()) >= 1, f"{context}: no NaN tree survived"
        assert int(torch.isinf(got).sum()) >= 1, f"{context}: no inf tree survived"
        assert int(torch.isfinite(got).sum()) >= 1, f"{context}: no finite tree present"
    if case == "finite":
        assert bool(torch.isfinite(got).all()), f"{context}: fixture is no longer finite"
        # the loss is MAXIMIZED by the caller, so it must stay a POSITIVE loss here
        assert bool((got >= 0).all()), f"{context}: negative loss {got}"


def test_use_mse_is_a_static_cfg_entry() -> None:
    """MSE and MAE are two graphs of the SAME shape signature (``cfg`` differs)."""
    forest, x, _ = _finite_case()
    rows = int(x.shape[0])
    mse = _plan(forest, rows, budget_mb=UNCHUNKED_BUDGET_MB, use_mse=True)
    mae = _plan(forest, rows, budget_mb=UNCHUNKED_BUDGET_MB, use_mse=False)

    assert mse.variant == mae.variant == dense_engine.LOSS_VARIANT
    assert mse.cfg[:-1] == mae.cfg[:-1], "only the trailing use_mse entry may differ"
    assert mse.cfg[-1] is True and mae.cfg[-1] is False
    assert mse.used_func_ids == mae.used_func_ids
    assert mse.specs != mae.specs, "distinct specs -> distinct build_cached keys"


# =========================================================================== #
# 2. cross-check against the specialized (per-structure) engine
# =========================================================================== #


@pytest.mark.parametrize("case", ["mixed", "finite"])
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_dense_loss_matches_the_specialized_engine(case: str, use_mse: bool) -> None:
    """Forced-dense vs forced-specialized on the same population and data.

    The battery's ``parity`` bar: ``rtol = atol = 1e-4`` with exact NaN/±inf masks.
    The in-graph reduction order differs from the specialized host reduction, so this
    comparison is deliberately NOT bit-exact.
    """
    forest, x, labels = _CASES[case]()
    context = f"dense vs specialized ({case}, use_MSE={use_mse})"
    got = _dense_loss(forest, x, labels, use_mse)
    population = evoxc_engine.decode_population(
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    want = evoxc_engine.sr_fitness(
        population, x, labels, use_MSE=use_mse, backend=BACKEND
    )
    assert tuple(want.shape) == (forest.pop_size,), f"{context}: shape {want.shape}"
    te.assert_close_with_placement(got, want, context=context)


# =========================================================================== #
# 3. result contract: shape / dtype / device / freshness
# =========================================================================== #


def test_result_contract_is_a_fresh_population_sized_float32_tensor() -> None:
    forest, x, labels = _finite_case()
    got = _dense_loss(forest, x, labels, True)
    assert tuple(got.shape) == (forest.pop_size,), f"shape {tuple(got.shape)}"
    assert got.dtype == torch.float32, f"dtype {got.dtype}"
    assert got.is_contiguous(), "the loss must be contiguous"
    assert got.requires_grad is False, "the loss must not require grad"
    assert getattr(got, "_base", None) is None, "the loss must be a fresh tensor"
    assert got.device == x.device, f"{got.device} != {x.device} (inputs' device)"


def test_result_is_fresh_and_writable() -> None:
    """Two calls are distinct tensors and writing into one cannot corrupt the next."""
    forest, x, labels = _finite_case()
    first = _dense_loss(forest, x, labels, True)
    second = _dense_loss(forest, x, labels, True)
    assert first is not second
    assert first.data_ptr() != second.data_ptr(), "two calls must not share a buffer"

    baseline = second.clone()
    first.fill_(float("nan"))  # what StandardPipeline does with a NaN fitness
    third = _dense_loss(forest, x, labels, True)
    te.assert_close_with_placement(
        third, baseline, exact=True, context="in-place write leaked into the next call"
    )


def test_result_device_is_the_inputs_device() -> None:
    """Without ``out_device`` the loss lands on ``inputs``' device (CPU here)."""
    forest, x, labels = _finite_case()
    got = _dense_loss(forest, x, labels, True)
    assert got.device == x.device == torch.device("cpu")
    overridden = _dense_loss(forest, x, labels, True, out_device="cpu")
    assert overridden.device == torch.device("cpu")
    te.assert_close_with_placement(overridden, got, context="out_device='cpu'")


@requires_cuda
def test_result_device_follows_cuda_inputs_and_out_device_override() -> None:
    forest, x, labels = _finite_case()
    x_cuda = x.to("cuda")
    labels_cuda = labels.to("cuda")

    on_device = _dense_loss(forest, x_cuda, labels_cuda, True)
    assert on_device.device.type == "cuda", f"{on_device.device}"
    assert on_device.device == x_cuda.device, "the loss follows inputs' device"

    back_on_host = _dense_loss(forest, x_cuda, labels_cuda, True, out_device="cpu")
    assert back_on_host.device.type == "cpu", f"{back_on_host.device}"

    host_reference = _dense_loss(forest, x, labels, True)
    te.assert_close_with_placement(
        on_device, host_reference, context="cuda inputs vs cpu inputs"
    )
    te.assert_close_with_placement(
        back_on_host, host_reference, context="cuda inputs + out_device='cpu'"
    )


# =========================================================================== #
# 4. safety: unread row tail, perfect fit, rejected inputs
# =========================================================================== #


def test_unread_row_tail_junk_does_not_change_the_loss() -> None:
    """Junk past ``subtree_size[0]`` is inert (CONST / BFUNC / TFUNC flavors)."""
    clean_values, clean_types, clean_sizes = te.rows_from_specs(
        _TAIL_SPECS, 1, pad_to=_TAIL_ROW_LEN, tail_value=0.0
    )
    clean = Forest(INPUT_LEN, 1, clean_values, clean_types, clean_sizes)
    _, x, labels = _finite_case()

    baseline = _dense_loss(clean, x, labels, True)
    assert bool(torch.isfinite(baseline).all()), "the tail fixture must stay finite"

    tree_len = int(clean_sizes[0, 0])
    assert tree_len < _TAIL_ROW_LEN, "the fixture needs an unused tail to junk up"
    junk_flavors = {
        "const": (int(te.CONST), 123.5, 1),
        # BFUNC/TFUNC declare 2/3 operands, so a read tail would run past the row
        "bfunc": (int(te.BFUNC), 3.0, 4),
        "tfunc": (int(te.TFUNC), 7.0, 7),
    }
    for flavor, (node_type, value, size) in junk_flavors.items():
        values, types, sizes = te.rows_from_specs(
            _TAIL_SPECS, 1, pad_to=_TAIL_ROW_LEN, tail_value=0.0
        )
        tail = slice(tree_len, _TAIL_ROW_LEN)
        types[:, tail] = node_type
        values[:, tail] = value
        sizes[:, tail] = size
        junk = Forest(INPUT_LEN, 1, values, types, sizes)
        got = _dense_loss(junk, x, labels, True)
        assert torch.equal(got, baseline), (
            f"{flavor} junk changed the loss: {got} != {baseline}"
        )


def test_perfect_fit_is_exactly_zero() -> None:
    """A tree reproducing the labels scores an exact 0.0 (``exact=True`` bar)."""
    forest = _forest([te.var(0)])
    x = te.x_random(4, INPUT_LEN, seed=13)
    labels = x[:, :1].contiguous()
    got = _dense_loss(forest, x, labels, True)
    te.assert_close_with_placement(
        got, torch.zeros(1), exact=True, context="perfect fit"
    )


def test_bad_labels_shape_and_empty_data_raise_value_error() -> None:
    forest, x, labels = _finite_case()
    with pytest.raises(
        ValueError, match=rf"labels must have shape \({N_ROWS}, 1\), got \(5, 1\)"
    ):
        _dense_loss(forest, x, labels[:5], True)
    with pytest.raises(ValueError, match=rf"labels must have shape \({N_ROWS}, 1\)"):
        _dense_loss(forest, x, torch.cat([labels, labels], dim=1), True)
    with pytest.raises(ValueError, match="needs at least one data point, got 0 rows"):
        _dense_loss(forest, x[:0], labels[:0], True)
    with pytest.raises(ValueError, match=rf"inputs must be 2-D \(rows, {INPUT_LEN}\)"):
        _dense_loss(forest, x[:, 0], labels, True)


# =========================================================================== #
# 5. counted host crossings: the (pop_size,)-scale contract of the loss route
# =========================================================================== #


def test_counted_call_converts_inputs_and_labels_to_host_exactly_once(
    monkeypatch: pytest.MonkeyPatch, both_chunked_plan: Tuple[float, Any]
) -> None:
    """``inputs`` and ``labels`` cross to the host ONCE per call (pre-refactor: 2)."""
    budget, plan = both_chunked_plan
    assert plan.p_chunks > 1 and plan.n_chunks > 1, _plan_line(plan)
    forest, x, labels = _finite_case()
    _, traffic = _counted_loss(monkeypatch, forest, x, labels, budget_mb=budget)

    names = [name for _, name, _ in traffic.as_numpy]
    for payload in (
        "inputs",
        "labels",
        "batch_node_value",
        "batch_node_type",
        "batch_subtree_size",
    ):
        count = len(traffic.conversions(payload))
        assert count == 1, (
            f"{payload} converted to host {count}x: {traffic.as_numpy}"
        )
    # ... and nothing ELSE converts a torch tensor to the host on this route
    assert len(names) == 5, f"unexpected extra torch->host conversions: {traffic.as_numpy}"


@pytest.mark.parametrize("regime", ["unchunked", "p-and-n-chunked"])
def test_counted_call_readbacks_are_population_scale(
    monkeypatch: pytest.MonkeyPatch,
    regime: str,
    unchunked_plan: Tuple[float, Any],
    both_chunked_plan: Tuple[float, Any],
) -> None:
    """Only ``(p_chunk,)`` chunk losses cross back -- never the ``(P, N)`` prediction."""
    plan_by_regime = {
        "unchunked": unchunked_plan,
        "p-and-n-chunked": both_chunked_plan,
    }
    budget, plan = plan_by_regime[regime]
    forest, x, labels = _finite_case()
    _, traffic = _counted_loss(monkeypatch, forest, x, labels, budget_mb=budget)

    chunks = plan.p_chunks * plan.n_chunks
    if regime == "unchunked":
        assert chunks == 1, _plan_line(plan)

    readbacks = traffic.readbacks
    assert len(readbacks) == chunks, (
        f"{len(readbacks)} readbacks for {chunks} chunks: {readbacks}"
    )
    # Positive control for the caller-frame filter: the numpy interpreter calls
    # ``Tensor.numpy`` thousands of times internally, so a filter that matched
    # nothing (0) or everything (thousands) fails right here -- the ONLY frame that
    # may appear is the dense route's ``_to_host``.
    assert [frame for frame, _ in readbacks] == ["_to_host"] * chunks, (
        f"unexpected readback sites: {readbacks}"
    )

    host_shapes = [shape for _, shape in traffic.to_host]
    assert host_shapes == [(plan.p_chunk,)] * chunks, (
        f"readback shapes {host_shapes} != {chunks} x ({plan.p_chunk},) "
        f"(a full prediction would be ({plan.p_chunk}, {plan.n_chunk}))"
    )
    assert all(len(shape) == 1 for shape in host_shapes), (
        f"a 2-D readback is the (P, N) prediction: {host_shapes}"
    )
    elements = sum(math.prod(shape) for shape in host_shapes)
    assert elements == plan.p_chunk * chunks, (
        f"{elements} read-back elements != p_chunk x chunks = {plan.p_chunk * chunks}"
    )
    assert max(math.prod(shape) for shape in host_shapes) <= forest.pop_size
    if regime == "unchunked":
        # the whole population's loss in ONE (pop_size,)-scale readback
        assert host_shapes == [(forest.pop_size,)]
        assert elements == forest.pop_size

    # one compiled-graph invocation per chunk (never more, never fewer)
    assert len(traffic.runs) == chunks, (
        f"{len(traffic.runs)} evoxc.run calls for {chunks} chunks: {traffic.runs}"
    )


def test_counted_call_creates_no_torch_tensor_larger_than_pop_size(
    monkeypatch: pytest.MonkeyPatch, both_chunked_plan: Tuple[float, Any]
) -> None:
    """The only host -> torch tensor is the ``(pop_size,)`` loss; labels never bounce."""
    budget, plan = both_chunked_plan
    forest, x, labels = _finite_case()
    rows = int(x.shape[0])
    _, traffic = _counted_loss(monkeypatch, forest, x, labels, budget_mb=budget)

    creations = traffic.torch_creations
    assert creations, "the call materializes its loss through torch.from_numpy"
    elements = [math.prod(shape) for _, shape in creations]
    assert max(elements) <= forest.pop_size, (
        f"a host->torch tensor larger than pop_size={forest.pop_size} was created: "
        f"{creations}"
    )
    assert all(count != rows for count in elements), (
        f"a labels-shaped creation means D2H -> torch -> H2D: {creations} ({rows} rows)"
    )


def test_counted_call_prepares_x_and_labels_once_per_n_chunk(
    monkeypatch: pytest.MonkeyPatch, both_chunked_plan: Tuple[float, Any]
) -> None:
    """A payload is prepared once per ``n``-chunk, not once per ``(p0, n0)`` chunk."""
    budget, plan = both_chunked_plan
    assert plan.p_chunks > 1 and plan.n_chunks > 1, _plan_line(plan)
    forest, x, labels = _finite_case()
    _, traffic = _counted_loss(monkeypatch, forest, x, labels, budget_mb=budget)

    x_preparations = traffic.preparations(INPUT_LEN)
    label_preparations = traffic.preparations(forest.output_len)
    assert len(x_preparations) == plan.n_chunks, (
        f"{len(x_preparations)} x preparations for n_chunks={plan.n_chunks} "
        f"(p_chunks={plan.p_chunks})"
    )
    assert len(label_preparations) == plan.n_chunks, (
        f"{len(label_preparations)} labels preparations for n_chunks={plan.n_chunks}"
    )


# =========================================================================== #
# 6. chunking: the padded last chunk must not change the loss
# =========================================================================== #


def test_chunked_loss_equals_single_chunk_loss(
    padded_plan: Tuple[float, Any], unchunked_plan: Tuple[float, Any]
) -> None:
    """Chunking is a reduction-order change only (the battery's ``1e-6`` bar).

    ``N`` is not a multiple of the forced chunk size, so the last chunk is PADDED and
    its padding slots must reduce to an exact ``+0.0`` (the graph's ``keep`` mask)
    instead of double counting the repeated row.
    """
    budget, plan = padded_plan
    forest, x, labels = _finite_case()
    rows = int(x.shape[0])
    assert plan.n_chunk > 1 and plan.n_chunks > 1, _plan_line(plan)
    assert rows % plan.n_chunk != 0, f"{rows} rows are a multiple of {plan.n_chunk}"

    for use_mse in (True, False):
        single = _dense_loss(forest, x, labels, use_mse, budget_mb=UNCHUNKED_BUDGET_MB)
        chunked = _dense_loss(forest, x, labels, use_mse, budget_mb=budget)
        assert torch.allclose(
            single, chunked, rtol=1e-6, atol=1e-6, equal_nan=True
        ), (
            f"use_MSE={use_mse}: chunked != single chunked, max|diff| = "
            f"{float((single - chunked).abs().max()):.3g}"
        )
        # ... and the chunked value is still the oracle's value, not just self-consistent
        te.assert_close_with_placement(
            chunked,
            te.reference_sr_fitness(forest, x, labels, use_mse),
            context=f"chunked (padded last chunk, use_MSE={use_mse}) vs oracle",
        )


# =========================================================================== #
# 7. the wired public route under an explicitly forced dense mode
# =========================================================================== #


@requires_cuda
def test_public_route_forced_dense_matches_the_dense_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Forest.SR_fitness`` with ``EVOGP_EVAL_MODE=dense`` returns the dense loss.

    ``check_tensor`` relocates the public call's inputs to ``'cuda'``, hence the CUDA
    guard. The dense call is recorded, so the test cannot pass through the silent
    fallback instead of the route it claims to check.
    """
    forest, x, labels = _finite_case()
    calls: List[int] = []
    original = dense_engine.sr_fitness_dense

    def recording(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dense_engine, "sr_fitness_dense", recording)
    monkeypatch.setenv(dense_engine.EVAL_MODE_ENV, "dense")
    monkeypatch.setenv("EVOGP_EVOXC_BACKEND", BACKEND)

    got = forest.SR_fitness(x.to("cuda"), labels.to("cuda"), True, "auto")
    assert calls, "the wired route did not take the dense branch"
    assert tuple(got.shape) == (forest.pop_size,), f"shape {tuple(got.shape)}"
    assert got.device.type == "cuda", f"{got.device}"

    direct = _dense_loss(forest, x, labels, True)
    te.assert_close_with_placement(
        got, direct, context="Forest.SR_fitness (forced dense) vs sr_fitness_dense"
    )
