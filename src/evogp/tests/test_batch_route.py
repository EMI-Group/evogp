"""Public-route coverage of the OPT-IN batch (merged-graph) evaluation route.

What is wired, and what this file pins
--------------------------------------
``EVOGP_EVAL_MODE=batch`` is the only way to reach
:mod:`evogp.tree.batch_engine` (the third route: every DISTINCT program of the
population is traced into its own small graph and those graphs are merged into ONE
executable, so one ``evoxc.run`` evaluates the whole population). The default
``auto`` policy never selects it.

The route is reached through ``dense_engine.resolve_route(pop_size)``, and FIVE
public call sites act on ``route.engine``:

* ``Forest.forward(x)`` -- population of ``pop_size``, ``per_row=True``;
* ``Forest.batch_forward(x)`` -- same population, ``per_row=False``;
* ``Forest.SR_fitness(inputs, labels, use_MSE, execute_mode)`` -- the loss variant;
* ``Tree.forward(x)`` -- a population of ONE (the row is ``unsqueeze``d), so with
  the default ``EVOGP_BATCH_MIN_POP`` it always declines; a 1-D ``x`` means
  ``per_row=True`` and a 2-D ``x`` means ``per_row=False``;
* ``Tree.SR_fitness(inputs, labels, use_MSE, execute_mode)`` -- a population of one.

Each site calls ``batch_engine.evaluate_batch`` / ``batch_engine.sr_fitness_batch``
with ``backend=route.backend`` and catches EXACTLY ``batch_engine.BatchUnsupported``
(a ``RuntimeError`` subclass), on which it re-runs through
``route.fallback_backend`` and the specialized engine
(``evoxc_engine.decode_population`` + ``evaluate`` / ``sr_fitness``). Any other
exception propagates, so a bug in the merge cannot hide behind the fallback.

This file is the PUBLIC-route gate: it proves the wiring takes the batch branch when
the mode forces it, that the numbers never change on the way (against the
specialized reference and against the entry point called by hand), that a decline
really falls back instead of escaping, and that ``Tree``'s pop-1 decline is the
threshold and not a broken route. The engine's own numeric contract (value/loss
variants, row expansion, ``describe_plan``, declines, the all-constant escalation)
is covered by ``test_batch_engine.py``; the merge knobs and the plan/executable memo
belong to the sibling ``test_batch_knobs.py`` -- this file therefore asserts
NOTHING about ``batch_cache_stats()`` and passes no merge knob.

Hermeticity / shape grid
-----------------------
* the whole module is ``skipif(not torch.cuda.is_available())``: every public entry
  point relocates its inputs with ``check_tensor`` to the hard-coded ``"cuda"``
  device string, so a CPU-only box must SKIP this file, not error in it;
* ``EVOGP_EVOXC_BACKEND=numpy`` on every test (the suite's ``conftest.py`` pins it
  session-wide anyway) and every engine call is made on ``backend="numpy"``: no test
  may compile an iree graph, and the iree packages are absent in this venv, so a
  ``batch``+``iree`` leg is not attempted anywhere in the file;
* ``EVOGP_EVAL_MODE`` is set per test with ``monkeypatch`` (never an unrestored
  ``os.environ`` write), so the file behaves identically under the default run and
  under a forced ``EVOGP_EVAL_MODE=dense`` / ``=batch`` run of the suite;
* small shapes only (``pop_size <= 7``, ``N <= 7``, ``L = 16``, ``<= 7`` merged
  functions), so the numpy compiles stay in the millisecond range;
* ``reference_eval.py`` (the oracle) is used as-is through ``test_equivalence``'s
  helpers -- never re-implemented here -- and nothing outside
  ``src/evogp/tests/`` is touched.

There is **no wall-clock assertion anywhere** in this file: the economics of the
route live in ``src/evogp/tree/BATCH_EVAL.md``.

Run it from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc <venv>/bin/python -m pytest \\
        src/evogp/tests/test_batch_route.py -q -p no:cacheprovider
    EVOGP_EVAL_MODE=dense EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src:/root/evoxc \\
        <venv>/bin/python -m pytest src/evogp/tests/test_batch_route.py -q \\
        -p no:cacheprovider
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

from evogp.tree import Forest, Tree  # noqa: E402
from evogp.tree import batch_engine, batch_knobs, dense_engine  # noqa: E402

# --------------------------------------------------------------------------- #
# the shape grid (small by design: the whole file must stay a few seconds)
# --------------------------------------------------------------------------- #
#: The only backend this file ever compiles on (see the module docstring).
BACKEND = "numpy"
#: The device ``check_tensor`` relocates every public call's inputs to.
DEVICE = "cuda"
INPUT_LEN = 3
#: Row width of every hand-built population (the unused tail is a CLEAN CONST/0.0).
ROW_LEN = 16
#: Data rows of the single-output fixtures; equals the population size, so
#: ``Forest.forward``'s ``(pop_size, input_len)`` contract is satisfiable.
N_ROWS = 5
#: Data rows of the multi-output fixture (its own population, no duplication).
N_MULTI = 4

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the public call sites under test (Forest/Tree forward and SR_fitness) "
    "relocate their inputs to 'cuda' via check_tensor, and the batch engine's "
    "numpy path is entered through them; there is nothing to route without a "
    "CUDA-visible device",
)
#: Every test needs a CUDA device: the file SKIPS cleanly on a CPU-only box.
pytestmark = requires_cuda


@pytest.fixture(autouse=True)
def _pin_engine_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin everything ambient that could change what a test observes.

    ``EVOGP_EVOXC_BACKEND=numpy`` -- the wired call sites resolve the backend from the
    environment, and no test in this file may compile an iree graph.

    The merge knobs (``EVOGP_BATCH_BAND_SIZE`` / ``BANDS`` / ``SWITCH_OPS`` /
    ``ALIGNMENT`` / ``COMPILE_WORKERS``) are REMOVED, so the library defaults apply
    and an ambient value cannot change the merge shape (or split the compile into
    chunks) under the assertions. ``EVOGP_BATCH_MIN_POP`` is removed for the same
    reason: the tests that care about the threshold set it explicitly, and the ones
    that must see the LIBRARY default (``64``) rely on it being absent.

    ``EVOGP_EVAL_MODE`` is deliberately NOT pinned here -- each test sets the mode it
    needs through ``monkeypatch`` (see :func:`_in_mode` / :func:`_force_batch`), which
    is what makes the file independent of the mode the suite is run with.
    ``EVOGP_DENSE_AUTO_MIN_POP`` / ``EVOGP_DENSE_AUTO_MAX_ROWS`` are removed too,
    because the trip-wire test also reads the ``auto`` route.
    """
    monkeypatch.setenv("EVOGP_EVOXC_BACKEND", BACKEND)
    for name in (
        batch_engine.BAND_SIZE_ENV,
        batch_engine.BANDS_ENV,
        batch_engine.SWITCH_OPS_ENV,
        batch_engine.ALIGNMENT_ENV,
        getattr(batch_knobs, "COMPILE_WORKERS_ENV", "EVOGP_BATCH_COMPILE_WORKERS"),
        batch_engine.MIN_POP_ENV,
        dense_engine.AUTO_MIN_POP_ENV,
        dense_engine.AUTO_MAX_ROWS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# hand-built populations (no global RNG involved, so every case is deterministic)
# --------------------------------------------------------------------------- #
#: FOUR distinct finite programs plus one duplicate row: the population shape a GP
#: loop actually presents, with a constant tree among computed ones.
_FINITE_SPECS: List[Any] = [
    te.fn(te.ADD, te.var(0), te.var(1)),
    te.fn(te.MUL, te.var(2), te.const(1.5)),
    te.var(0),
    te.const(-2.25),
    te.fn(te.ADD, te.var(0), te.var(1)),  # duplicate of row 0 -> merged once
]

#: A DIFFERENT population of the same size: different programs, different numbers,
#: so a stale row mapping / a memoized answer from another fixture cannot pass.
_OTHER_SPECS: List[Any] = [
    te.fn(te.SIN, te.var(1)),
    te.fn(te.TANH, te.var(2)),
    te.fn(te.SUB, te.fn(te.MUL, te.var(0), te.var(1)), te.const(0.5)),
    te.fn(te.MAX, te.var(0), te.const(-1.0)),
    te.fn(te.MUL, te.var(2), te.const(-0.75)),
]

#: Hostile data (hand-built, so the mixed population is deterministic): one row per
#: tree, arranged so the ``per_row`` layout alone reaches a NaN (the zero divisor),
#: ``+inf`` (``exp(100)`` on row 2) and ``-inf`` (its negation on row 5), while the
#: other rows stay finite.
_MIXED_X = torch.tensor(
    [
        [-1.0, 0.5, 2.0],
        [3.0, 1.0, -0.5],
        [100.0, -2.0, 3.0],
        [1.0, 0.0, 0.0],
        [2.0, 3.0, -1.0],
        [100.0, 0.5, 0.75],
        [-3.0, 4.0, 0.5],
    ],
    dtype=torch.float32,
)

#: A MIXED population: finite, NaN and ±inf values side by side (row by row).
_MIXED_SPECS: List[Any] = [
    te.fn(te.ADD, te.var(0), te.var(1)),  # finite
    te.fn(te.LOG, te.var(0)),  # finite on this row; NaN for any x0 < 0
    te.fn(te.EXP, te.var(0)),  # +inf on the overflowing row
    te.const(1.0),  # finite (constant-folded subtree)
    te.fn(te.DIV, te.var(1), te.const(0.0)),  # NaN (division by an exact zero)
    te.fn(te.SUB, te.const(0.0), te.fn(te.EXP, te.var(0))),  # -inf on that row
    te.var(2),  # finite
]

#: A multi-output population (``output_len = 2``): the out-node flag packs the output
#: index into the node value, and the engine merges it like any other program.
_MULTI_SPECS: List[Any] = [
    te.out_fn(te.ADD, 0, te.var(0), te.var(1)),
    te.out_fn(te.MUL, 1, te.var(2), te.const(0.5)),
    te.out_fn(te.SUB, 1, te.var(1), te.const(2.0)),
    te.const(1.25),
]


def _forest(
    specs: Sequence[Any],
    output_len: int = 1,
    *,
    input_len: int = INPUT_LEN,
    row_len: int = ROW_LEN,
) -> Forest:
    """A ``Forest`` of hand-built trees with a CLEAN junk tail, on the engine device.

    ``te.rows_from_specs`` writes CONST/0.0 with subtree size 0 into the unused tail,
    i.e. the rows are the "clean" reference encoding (nothing is read past
    ``subtree_size[0]`` by either engine).
    """
    values, types, sizes = te.rows_from_specs(specs, output_len, pad_to=row_len, tail_value=0.0)
    return Forest(
        input_len,
        output_len,
        values.to(DEVICE),
        types.to(DEVICE),
        sizes.to(DEVICE),
    )


def _x(n: int = N_ROWS, seed: int = 5) -> torch.Tensor:
    """Deterministic ``(n, input_len)`` inputs from a LOCAL generator."""
    return te.x_random(n, INPUT_LEN, seed=seed).to(DEVICE)


def _labels(n: int = N_ROWS, out_len: int = 1, seed: int = 6) -> torch.Tensor:
    """Deterministic labels from a LOCAL generator (no global-RNG coupling)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((n, out_len), generator=generator, dtype=torch.float32).to(DEVICE)


def _mixed_case() -> Tuple[Forest, torch.Tensor, torch.Tensor]:
    """The NaN/±inf fixture: forest, its hostile inputs and its labels."""
    return _forest(_MIXED_SPECS), _MIXED_X.to(DEVICE), _labels(_MIXED_X.shape[0], seed=21)


# --------------------------------------------------------------------------- #
# the public call sites under test
# --------------------------------------------------------------------------- #
#: ``name -> (the wired call, the batch entry point it must reach, the ``per_row`` it
#: must pass or ``None`` when the entry point takes no ``per_row``)`` for ``Forest``.
_CALL_SITES: Dict[str, Tuple[Callable[..., torch.Tensor], str, Optional[bool]]] = {
    "Forest.forward": (
        lambda forest, x, labels: forest.forward(x),
        "evaluate_batch",
        True,
    ),
    "Forest.batch_forward": (
        lambda forest, x, labels: forest.batch_forward(x),
        "evaluate_batch",
        False,
    ),
    "Forest.SR_fitness": (
        lambda forest, x, labels: forest.SR_fitness(x, labels, True, "auto"),
        "sr_fitness_batch",
        None,
    ),
}

#: The same table for ``Tree``: the population is always ONE row, and a 1-D ``x`` is
#: the ``per_row`` layout while a 2-D ``x`` evaluates the single tree on every row.
_TREE_SITES: Dict[str, Tuple[Callable[..., torch.Tensor], str, Optional[bool]]] = {
    "Tree.forward(1-D x)": (
        lambda tree, x, labels: tree.forward(x[0]),
        "evaluate_batch",
        True,
    ),
    "Tree.forward(2-D x)": (
        lambda tree, x, labels: tree.forward(x),
        "evaluate_batch",
        False,
    ),
    "Tree.SR_fitness": (
        lambda tree, x, labels: tree.SR_fitness(x, labels, True, "auto"),
        "sr_fitness_batch",
        None,
    ),
}


class _BatchSpy:
    """Record -- and optionally force a decline of -- ONE batch entry point.

    Installed on the ``batch_engine`` module attribute, i.e. exactly the object the
    wired call site dereferences, so ``calls`` answers "did the public call really
    take the batch branch?" (the silent fallback would leave it empty). Each entry is
    the call's KEYWORD arguments, which is what lets the tests pin ``backend`` and
    ``per_row`` without reaching into the engine's internals.

    ``decline = True`` turns every attempt into a real
    :class:`batch_engine.BatchUnsupported` -- the only exception type the call sites
    catch -- so the fallback itself can be tested against a route that never helps.
    ``monkeypatch`` puts the original back at teardown.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
        self.entry = entry
        self.calls: List[Dict[str, Any]] = []
        self.decline = False
        self._original = getattr(batch_engine, entry)
        monkeypatch.setattr(batch_engine, entry, self._call)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.decline:
            raise batch_engine.BatchUnsupported(
                "test-forced decline: pretend this route cannot help"
            )
        return self._original(*args, **kwargs)

    @property
    def attempts(self) -> int:
        """How many times the wired call site tried this entry point."""
        return len(self.calls)

    @property
    def last(self) -> Dict[str, Any]:
        """The keyword arguments of the most recent attempt."""
        assert self.calls, "the wired route never attempted this entry point"
        return self.calls[-1]


def _in_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str, call: Callable[..., Any], *args: Any
) -> Any:
    """Run ``call`` with ``EVOGP_EVAL_MODE=mode`` (``monkeypatch`` restores it later).

    ``specialized`` is how every reference number in this file is produced: it is the
    per-structure engine the batch route falls back to, reached through the SAME
    public call site, so a reference can never be "the batch route compared against
    itself".
    """
    monkeypatch.setenv(dense_engine.EVAL_MODE_ENV, mode)
    return call(*args)


def _force_batch(monkeypatch: pytest.MonkeyPatch, min_pop: int = 1) -> None:
    """Force the batch route and admit the small populations these fixtures use.

    The library's own threshold (``64``) is the merged compile's measured break-even,
    so every fixture here is below it and needs an explicit ``EVOGP_BATCH_MIN_POP``.
    A test that must see the threshold DECLINE passes a ``min_pop`` above the
    population instead (or leaves the ambient value removed, i.e. the default).
    """
    monkeypatch.setenv(dense_engine.EVAL_MODE_ENV, "batch")
    monkeypatch.setenv(batch_engine.MIN_POP_ENV, str(min_pop))


def _assert_forces_the_batch_engine(pop_size: int) -> None:
    """Precondition: the ambient knobs resolve to the batch route on ``numpy``.

    A diagnostic, not a policy test (``test_route_policy.py`` owns the policy): if the
    wiring ever stopped routing ``EVOGP_EVAL_MODE=batch`` here, this names the reason
    instead of letting the number assertions fail for an unrelated cause.
    """
    route = dense_engine.resolve_route(pop_size)
    assert route.engine == "batch", f"resolve_route({pop_size}) -> {route}"
    assert route.backend == BACKEND, f"resolve_route({pop_size}).backend -> {route.backend}"


def _direct_engine(
    site: str, forest: Forest, x: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """The batch entry point called BY HAND with the same rows/x/labels.

    This is the "the route's own maths" half of the parity claim: the wired call must
    return what a direct call to the merged-graph evaluator returns. It is always made
    with ``backend="numpy"`` and BEFORE a spy is installed, so it cannot inflate any
    attempt count.
    """
    _, entry, per_row = _CALL_SITES[site]
    arguments = (
        forest.batch_node_value,
        forest.batch_node_type,
        forest.batch_subtree_size,
        forest.input_len,
        forest.output_len,
    )
    if entry == "evaluate_batch":
        return batch_engine.evaluate_batch(*arguments, x, per_row=per_row, backend=BACKEND)
    return batch_engine.sr_fitness_batch(
        *arguments, x, labels, use_MSE=True, backend=BACKEND
    )


# --------------------------------------------------------------------------- #
# 1. the batch branch really ran -- and returns the route's numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_forced_batch_route_takes_the_batch_branch(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """A forced-batch ``Forest`` call reaches the batch entry point, with the route's semantics.

    Pins three things at once, which is the whole point of the public-route gate:
    (a) the entry point is called EXACTLY ONCE (a silent fallback would leave the spy
    empty), (b) it is called with the route's backend and with the ``per_row`` layout
    that call site owns, and (c) the numbers equal both the directly-called engine
    output and the specialized-engine reference (NaN/±inf placement asserted first,
    then the finite entries at ``rtol = atol = 1e-4``) on a float32 tensor that stays
    on the inputs' device.
    """
    call, entry, expected_per_row = _CALL_SITES[site]
    forest = _forest(_FINITE_SPECS)
    x, labels = _x(), _labels()

    # The reference: the SAME public call site with the batch route disabled.
    reference = _in_mode(monkeypatch, "specialized", call, forest, x, labels)

    _force_batch(monkeypatch)
    _assert_forces_the_batch_engine(forest.pop_size)
    direct = _direct_engine(site, forest, x, labels)

    spy = _BatchSpy(monkeypatch, entry)
    got = call(forest, x, labels)

    assert spy.attempts == 1, (
        f"{site}: the wired route did not take the batch branch "
        f"({spy.entry} attempts: {spy.attempts})"
    )
    assert spy.last.get("backend") == BACKEND, f"{site}: backend {spy.last}"
    if expected_per_row is not None:
        assert spy.last.get("per_row") is expected_per_row, f"{site}: {spy.last}"
    assert got.device.type == DEVICE, f"{site}: result device {got.device}"
    assert got.dtype == torch.float32, f"{site}: result dtype {got.dtype}"

    te.assert_close_with_placement(
        got, direct, context=f"{site} (forced batch) vs batch_engine.{entry}"
    )
    te.assert_close_with_placement(
        got, reference, context=f"{site} (forced batch) vs the specialized engine"
    )


# --------------------------------------------------------------------------- #
# 2. a silent fallback cannot pass: different rows, and a forced decline
# --------------------------------------------------------------------------- #
def test_forced_batch_route_matches_the_specialized_reference_on_a_different_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parity above is not an artefact of one fixture: a DIFFERENT population matches too.

    The population here has different programs AND different numbers than the core
    test's fixture, and the test additionally asserts that its loss differs from the
    finite fixture's loss on the same rows -- so a stale result (a memo entry reused
    across fixtures, or a wrong row mapping) would be caught instead of agreeing with
    a reference computed for the same wrong rows.
    """
    forest = _forest(_OTHER_SPECS)
    x, labels = _x(), _labels(seed=13)

    reference_value = _in_mode(monkeypatch, "specialized", forest.forward, x)
    reference_loss = _in_mode(
        monkeypatch, "specialized", forest.SR_fitness, x, labels, True, "auto"
    )
    other_loss = _in_mode(
        monkeypatch,
        "specialized",
        _forest(_FINITE_SPECS).SR_fitness,
        x,
        labels,
        True,
        "auto",
    )

    _force_batch(monkeypatch)
    _assert_forces_the_batch_engine(forest.pop_size)
    value_spy = _BatchSpy(monkeypatch, "evaluate_batch")
    loss_spy = _BatchSpy(monkeypatch, "sr_fitness_batch")

    got_value = forest.forward(x)
    got_loss = forest.SR_fitness(x, labels, True, "auto")

    assert value_spy.attempts == 1, "Forest.forward did not take the batch branch"
    assert loss_spy.attempts == 1, "Forest.SR_fitness did not take the batch branch"
    te.assert_close_with_placement(
        got_value, reference_value, context="other population, Forest.forward"
    )
    te.assert_close_with_placement(
        got_loss, reference_loss, context="other population, Forest.SR_fitness"
    )
    assert not torch.allclose(
        got_loss.cpu(), other_loss.cpu(), rtol=1e-3, atol=1e-3
    ), "the two fixtures produced the same loss -- a stale result would hide here"


@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_a_forced_batch_decline_falls_back_to_the_specialized_numbers(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """A real ``BatchUnsupported`` is caught by the wiring: the fallback is not an escape.

    The entry point is monkeypatched to raise unconditionally (the decline the engine
    raises for a shape doubt / a missing batching rule / a compile that cannot merge),
    the attempt is recorded, and the call still returns the specialized numbers. They
    are compared BIT-FOR-BIT on purpose: both sides then run the identical
    ``decode_population`` + ``evaluate``/``sr_fitness`` code on identical inputs, so
    this is an equality of code paths, not a numeric tolerance claim.
    """
    call, entry, _ = _CALL_SITES[site]
    forest = _forest(_FINITE_SPECS)
    x, labels = _x(), _labels()
    reference = _in_mode(monkeypatch, "specialized", call, forest, x, labels)

    _force_batch(monkeypatch)
    spy = _BatchSpy(monkeypatch, entry)
    spy.decline = True

    got = call(forest, x, labels)

    assert spy.attempts == 1, (
        f"{site}: the wired route did not attempt the batch branch "
        f"({spy.entry} attempts: {spy.attempts})"
    )
    assert tuple(got.shape) == tuple(reference.shape), f"{site}: shape {tuple(got.shape)}"
    assert torch.equal(
        got.detach().cpu(), reference.detach().cpu()
    ), f"{site}: the decline did not fall back to the specialized numbers"


# --------------------------------------------------------------------------- #
# 3. the population threshold: a decline raised by the ENGINE, not skipped by policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_threshold_decline_falls_back_with_the_specialized_numbers(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """``EVOGP_BATCH_MIN_POP`` above the population: one attempt, then the known numbers.

    The route itself does NOT gate the population (the batch engine owns that
    decision), so the batch branch must be ENTERED and the decline raised from inside
    the engine -- that is what the recorded attempt proves. The result must equal the
    specialized reference bit-for-bit (same code path on both sides).
    """
    call, entry, _ = _CALL_SITES[site]
    forest = _forest(_FINITE_SPECS)
    x, labels = _x(), _labels()
    reference = _in_mode(monkeypatch, "specialized", call, forest, x, labels)

    # The library default threshold is 64; make the "above the population" explicit.
    _force_batch(monkeypatch, min_pop=64)
    assert forest.pop_size < 64, f"fixture population {forest.pop_size} is not below the threshold"
    spy = _BatchSpy(monkeypatch, entry)

    got = call(forest, x, labels)

    assert spy.attempts == 1, (
        f"{site}: the batch branch was never entered, so the decline was skipped "
        f"rather than raised by the engine ({spy.entry} attempts: {spy.attempts})"
    )
    assert torch.equal(
        got.detach().cpu(), reference.detach().cpu()
    ), f"{site}: the threshold decline changed the numbers"


# --------------------------------------------------------------------------- #
# 4. a Tree is a population of ONE: it declines at the default threshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("site", list(_TREE_SITES))
def test_tree_declines_at_the_default_population_threshold(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """``Tree`` calls always attempt the merge and always decline at the default threshold.

    ``Tree.forward`` / ``Tree.SR_fitness`` route a population of ONE
    (``resolve_route(1)``), which is far below ``EVOGP_BATCH_MIN_POP`` (64 in the
    library; this test's autouse fixture removes any ambient value), so the batch
    branch is entered, the engine raises ``BatchUnsupported``, and the call falls back
    to the specialized engine -- with the ``per_row`` layout the input shape implies
    (1-D ``x`` -> ``per_row=True``, 2-D ``x`` -> ``per_row=False``).
    """
    call, entry, expected_per_row = _TREE_SITES[site]
    forest = _forest(_FINITE_SPECS)
    tree: Tree = forest[0]
    x, labels = _x(), _labels()
    reference = _in_mode(monkeypatch, "specialized", call, tree, x, labels)

    monkeypatch.setenv(dense_engine.EVAL_MODE_ENV, "batch")
    assert batch_engine.resolve_min_pop() == batch_engine.DEFAULT_MIN_POP, (
        "an ambient EVOGP_BATCH_MIN_POP was visible -- the default-threshold claim "
        "would be vacuous"
    )
    spy = _BatchSpy(monkeypatch, entry)

    got = call(tree, x, labels)

    assert spy.attempts == 1, (
        f"{site}: the wired route did not attempt the batch branch "
        f"({spy.entry} attempts: {spy.attempts})"
    )
    if expected_per_row is not None:
        assert spy.last.get("per_row") is expected_per_row, f"{site}: {spy.last}"
    assert torch.equal(
        got.detach().cpu(), reference.detach().cpu()
    ), f"{site}: the pop-1 decline changed the numbers"


@pytest.mark.parametrize("site", list(_TREE_SITES))
def test_tree_takes_the_batch_branch_below_the_threshold(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """The same ``Tree`` calls DO run through the merge once the threshold admits pop 1.

    This is the control for the decline above: if ``Tree`` returned the right numbers
    because the batch branch was broken (rather than because the threshold declined),
    this test would return something else. Layouts and numbers must be unchanged.
    """
    call, entry, expected_per_row = _TREE_SITES[site]
    forest = _forest(_FINITE_SPECS)
    tree: Tree = forest[0]
    x, labels = _x(), _labels()
    reference = _in_mode(monkeypatch, "specialized", call, tree, x, labels)

    _force_batch(monkeypatch)
    _assert_forces_the_batch_engine(1)
    spy = _BatchSpy(monkeypatch, entry)

    got = call(tree, x, labels)

    assert spy.attempts == 1, (
        f"{site}: the wired route did not take the batch branch "
        f"({spy.entry} attempts: {spy.attempts})"
    )
    if expected_per_row is not None:
        assert spy.last.get("per_row") is expected_per_row, f"{site}: {spy.last}"
    te.assert_close_with_placement(
        got, reference, context=f"{site} (forced batch, pop 1) vs the specialized engine"
    )


# --------------------------------------------------------------------------- #
# 5. the remaining public combinations: MAE, NaN/±inf placement, multi-output
# --------------------------------------------------------------------------- #
def test_mae_loss_through_the_batch_route_matches_the_specialized_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_MSE=False`` (MAE) is routed like MSE and keeps the NaN/±inf placement.

    The mixed fixture makes the choice observable in the numbers as well as in the
    recorded ``use_MSE``: the batch loss must equal the specialized MAE reference
    (masks first) and must differ from the MSE one, so a route that silently reduced
    with the wrong metric could not pass.
    """
    forest, x, labels = _mixed_case()
    reference_mae = _in_mode(
        monkeypatch, "specialized", forest.SR_fitness, x, labels, False, "auto"
    )
    reference_mse = _in_mode(
        monkeypatch, "specialized", forest.SR_fitness, x, labels, True, "auto"
    )

    _force_batch(monkeypatch)
    spy = _BatchSpy(monkeypatch, "sr_fitness_batch")
    got = forest.SR_fitness(x, labels, False, "auto")

    assert spy.attempts == 1, "Forest.SR_fitness did not take the batch branch"
    assert spy.last.get("use_MSE") is False, f"recorded {spy.last}"
    te.assert_close_with_placement(got, reference_mae, context="MAE through the batch route")
    assert tuple(got.shape) == (forest.pop_size,), f"shape {tuple(got.shape)}"
    finite = torch.isfinite(got.cpu()) & torch.isfinite(reference_mse.cpu())
    assert finite.any(), "the MAE/MSE trip-wire has no finite entry to compare"
    assert not torch.allclose(
        got.cpu()[finite], reference_mse.cpu()[finite], rtol=1e-3, atol=1e-3
    ), "MAE equalled MSE -- the metric selection did not survive the route"


def test_mixed_population_keeps_the_nan_and_inf_placement_through_the_batch_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NaN/±inf rows land on exactly the same trees as on the specialized engine.

    ``assert_close_with_placement`` asserts the NaN mask, the ±inf mask and the inf
    signs FIRST and only then compares the finite entries, which is the only way a
    merged-graph route can be trusted with hostile programs (``log`` of a negative,
    ``exp`` overflow, an exact division by zero).
    """
    forest, x, labels = _mixed_case()
    reference_value = _in_mode(monkeypatch, "specialized", forest.forward, x)
    reference_loss = _in_mode(
        monkeypatch, "specialized", forest.SR_fitness, x, labels, True, "auto"
    )
    assert torch.isnan(reference_value.cpu()).any(), "fixture lost its NaN row"
    assert torch.isposinf(reference_value.cpu()).any(), "fixture lost its +inf row"
    assert torch.isneginf(reference_value.cpu()).any(), "fixture lost its -inf row"

    _force_batch(monkeypatch)
    value_spy = _BatchSpy(monkeypatch, "evaluate_batch")
    loss_spy = _BatchSpy(monkeypatch, "sr_fitness_batch")
    got_value = forest.forward(x)
    got_loss = forest.SR_fitness(x, labels, True, "auto")

    assert value_spy.attempts == 1, "Forest.forward did not take the batch branch"
    assert loss_spy.attempts == 1, "Forest.SR_fitness did not take the batch branch"
    te.assert_close_with_placement(
        got_value, reference_value, context="mixed population, Forest.forward"
    )
    te.assert_close_with_placement(
        got_loss, reference_loss, context="mixed population, Forest.SR_fitness"
    )


@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_multi_output_forest_through_the_batch_route_matches_the_specialized_reference(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """``output_len = 2`` is SUPPORTED by the merge: it must match, never decline.

    Out-node rows are ordinary programs to the merged graph (the output index is
    packed into the node value), so the three ``Forest`` call sites must all take the
    batch branch -- ``attempts == 1`` rules out a ``BatchUnsupported`` that the
    fallback would otherwise have hidden -- and return the specialized numbers on the
    ``(pop_size, output_len)`` / ``(pop_size, N, output_len)`` / ``(pop_size,)``
    shapes.
    """
    call, entry, expected_per_row = _CALL_SITES[site]
    forest = _forest(_MULTI_SPECS, output_len=2)
    x, labels = _x(N_MULTI, seed=8), _labels(N_MULTI, out_len=2, seed=9)
    reference = _in_mode(monkeypatch, "specialized", call, forest, x, labels)

    _force_batch(monkeypatch)
    _assert_forces_the_batch_engine(forest.pop_size)
    spy = _BatchSpy(monkeypatch, entry)
    got = call(forest, x, labels)

    assert spy.attempts == 1, (
        f"{site}: multi-output did not go through the batch branch "
        f"({spy.entry} attempts: {spy.attempts})"
    )
    if expected_per_row is not None:
        assert spy.last.get("per_row") is expected_per_row, f"{site}: {spy.last}"
    assert forest.output_len == 2, f"fixture output_len drifted to {forest.output_len}"
    te.assert_close_with_placement(
        got, reference, context=f"{site} (multi-output, forced batch) vs specialized"
    )


# --------------------------------------------------------------------------- #
# 6. the trip-wire: forcing the mode never changes a number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_forced_batch_never_changes_the_numbers_against_specialized_and_auto(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """``batch`` == ``specialized`` == ``auto`` for one fixture -- the one-line contract.

    ``EVOGP_EVAL_MODE`` picks WHICH engine evaluates a population, never WHAT it
    computes; a future route (or a future merge) that computed something else would
    break this trip-wire. ``auto`` is the box's real default: with the numpy backend
    pinned it resolves to the specialized engine for this (sub-64) population, so the
    three results must agree.
    """
    call, _, _ = _CALL_SITES[site]
    forest = _forest(_FINITE_SPECS)
    x, labels = _x(), _labels()

    reference_specialized = _in_mode(monkeypatch, "specialized", call, forest, x, labels)
    reference_auto = _in_mode(monkeypatch, "auto", call, forest, x, labels)
    te.assert_close_with_placement(
        reference_specialized,
        reference_auto,
        context=f"{site}: specialized vs auto (the box's default route)",
    )

    _force_batch(monkeypatch)
    got = call(forest, x, labels)
    te.assert_close_with_placement(
        got, reference_specialized, context=f"{site}: forced batch vs specialized"
    )


@pytest.mark.parametrize("site", list(_CALL_SITES))
def test_numpy_backend_batch_and_specialized_are_bit_identical(
    monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    """MEASURED numpy behaviour, not a contract: the two routes agree to the last bit.

    On ``backend="numpy"`` both routes evaluate the same float32 arithmetic in the
    same interpreter, so the merged graph reproduces the specialized engine exactly --
    measured, ``max|diff| == 0``. It is asserted with ``torch.equal`` here because an
    exact equality is the strongest available trip-wire against a merge that reorders
    or re-associates an op; do NOT read it as a portability guarantee (an
    ``iree``/device backend is entitled to the documented ``1e-4`` tolerance only, and
    this file never compiles one).
    """
    call, _, _ = _CALL_SITES[site]
    forest = _forest(_FINITE_SPECS)
    x, labels = _x(), _labels()
    reference = _in_mode(monkeypatch, "specialized", call, forest, x, labels)

    _force_batch(monkeypatch)
    got = call(forest, x, labels)

    got_cpu, reference_cpu = got.detach().cpu(), reference.detach().cpu()
    assert got_cpu.dtype == torch.float32, f"{site}: dtype {got_cpu.dtype}"
    assert torch.equal(got_cpu, reference_cpu), (
        f"{site}: numpy batch and specialized differ by "
        f"{(got_cpu - reference_cpu).abs().max()} (measured bit-exact behaviour)"
    )
