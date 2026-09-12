"""Candidate-chain pruning (``EVOGP_DENSE_PRUNE``) regression coverage.

The dense graph's dominant static cost was the per-position *candidate chain*: a
``select`` chain over all 28 function ids at every emitted position.  The lever
emits candidates only for the ids the POPULATION uses (``DensePrep.used_func_ids``,
the canonical -- sorted, unique, ``1..28`` -- union over the decoded function
nodes, derived host-side in :func:`dense_engine.prepare_dense`), is ON by default
and is reverted byte-for-byte by ``EVOGP_DENSE_PRUNE=0``.

This file ports a focused subset of the dev battery
``src/evogp/tree/device_resident_check.py`` (stage ``prune``, 25 checks) into the
pytest suite.  Three promises are pinned:

1. **the knob resolves as documented** -- ``resolve_prune``: explicit argument
   wins; unset/empty env means ``DEFAULT_PRUNE`` (pruned); ``1/true/yes/on`` and
   ``0/false/no/off`` are the accepted spellings (case-insensitive, trimmed);
   anything else raises a ``ValueError`` naming ``EVOGP_DENSE_PRUNE``.
2. **the used-id set is exact** -- a canonical tuple, strictly inside
   ``FULL_FUNC_IDS`` for a "narrow" population, empty for a population without
   (in-range) function nodes, unaffected by row-tail junk; the plan carries it as
   the static ``cfg[6]`` entry, so it is part of the compiled-graph cache key
   (same set -> same executable, a different set -> exactly one new build).
3. **pruning changes emitted ops, never values** -- for every route the pruned
   and full-chain graphs are BIT-EXACT (float32 bitwise, NaN payloads / ``-0.0``
   included): the value route (``evaluate_dense`` / ``run_plan``) with
   ``per_row`` both ways, the SR loss (``sr_fitness_dense`` / ``run_loss_plan``)
   for MSE and MAE, a P+N-chunked plan of each variant, a near-full id set, a
   NaN/inf fixture and the degenerate EMPTY used set; and the emitted candidate
   chain really is shorter (``(emitted positions) x |used|`` vs ``x 28``,
   counted with the battery's emitter patch).

Out of scope here (the lever does not touch them): the per-op semantics of the
candidate chain (``test_equivalence.py`` + ``reference_eval.py`` own those, and
the oracle is used here only for the degenerate no-candidate fixtures), and every
device backend -- the lever cuts *emitted ops*, so this file is numpy-only.

Hermeticity: explicit ``backend``/``max_tree_len``/``budget_mb``/``update_mode``
arguments and explicit ``prune=``/env values, small shapes (``P <= 8``,
``N <= 37``, ``L = 16``), seeded host inputs, no timing assertion anywhere.  The
used-id set is a cache-key entry, so a pruned and a full-chain graph COEXIST in
the shared compile cache: the cache checks read hit/miss *deltas* or clear the
cache explicitly, and never assume an eviction.

Run (repository root)::

    ETL_DISABLE_LOCATIONS=1 PYTHONPATH=src <venv>/bin/python -m pytest \\
        src/evogp/tests/test_dense_prune.py -q -p no:cacheprovider
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

#: Shared spec DSL (``fn``/``var``/``const``/``raw``), row builders and the
#: NaN/inf-aware oracle comparison.  Importing ``test_equivalence`` here follows
#: the same directory-local import pattern it uses for ``reference_eval``.
import test_equivalence as eq  # noqa: E402

from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402
from evogp.tree.utils import Func, NType  # noqa: E402

FN, VAR, CONST, RAW = eq.fn, eq.var, eq.const, eq.raw

#: The host backend: the lever is about emitted ops and the bit-exactness claim
#: was measured on numpy (no GPU, no iree, no compile economics in the picture).
BACKEND = "numpy"
INPUT_LEN = 3
#: Row length == trace cap, so ``L = 16`` for every fixture here.
ROW_LEN = 16
#: Population / data-row counts of the two shapes this file uses.  All fixtures
#: of one shape share the full-chain graph (its ``cfg`` does not depend on the
#: population at all), which keeps the compile budget small.
POP_SIZE = 8
N_ROWS = 37
EDGE_POP_SIZE = 6
EDGE_ROWS = 5
BUDGET_MB = D.DEFAULT_BUDGET_MB
UPDATE_MODE = "grow"

# --------------------------------------------------------------------------- #
# fixtures: hand-built rows with a KNOWN function-id set
# --------------------------------------------------------------------------- #
#: The "narrow" fixture: a 4-id population, ``(1, 2, 3, 22)`` = ADD/SUB/MUL/EXP.
#: Each tree uses a subset of the four, so the union (not a per-tree set) is what
#: the tests assert.  Four distinct rows, doubled to ``POP_SIZE``.
_MAIN_SPECS = [
    FN(Func.ADD, VAR(0), FN(Func.MUL, VAR(1), CONST(2.0))),  # ADD, MUL
    FN(Func.SUB, VAR(0), FN(Func.EXP, VAR(2))),  # SUB, EXP
    FN(Func.MUL, VAR(1), CONST(-1.5)),  # MUL
    FN(Func.ADD, VAR(2), CONST(0.25)),  # ADD
] * 2
_MAIN_USED = (Func.ADD, Func.SUB, Func.MUL, Func.EXP)

#: Same shape signature as ``_MAIN_SPECS`` (``POP_SIZE`` rows of ``ROW_LEN``, so
#: the trace length and the chunk shape match) but a DIFFERENT used-id set -- the
#: cache-key fixture.
_OTHER_SPECS = [
    FN(Func.SIN, VAR(0)),
    FN(Func.SIN, FN(Func.COS, VAR(1))),
] * 4
_OTHER_USED = (Func.SIN, Func.COS)

#: A "wide" (18-of-28 ids) population: the pruned chain is nearly the full one,
#: which is the strongest bit-exactness case of the lever.
_WIDE_BFUNCS = (Func.ADD, Func.SUB, Func.MUL, Func.DIV, Func.POW, Func.MAX)
_WIDE_UFUNCS = (
    Func.SIN,
    Func.COS,
    Func.TAN,
    Func.SINH,
    Func.COSH,
    Func.TANH,
    Func.LOG,
    Func.LOOSE_LOG,
    Func.EXP,
    Func.INV,
    Func.LOOSE_INV,
    Func.NEG,
)
_WIDE_SPECS = [
    FN(_WIDE_BFUNCS[i], FN(_WIDE_UFUNCS[2 * i], VAR(0)), FN(_WIDE_UFUNCS[2 * i + 1], VAR(1)))
    for i in range(len(_WIDE_BFUNCS))
]
_WIDE_USED = tuple(sorted(_WIDE_BFUNCS + _WIDE_UFUNCS))

#: NaN/inf sources: ``DIV`` by zero and ``LOG``/``SQRT`` of a non-positive value
#: yield NaN in this engine (see ``etl_engine._emit_binary`` / ``_emit_unary``),
#: ``LOG(0)`` is ``-inf`` and ``EXP(1e30)`` overflows to ``+inf``.
_NAN_INF_SPECS = [
    FN(Func.LOG, CONST(0.0)),  # -inf
    FN(Func.EXP, CONST(1e30)),  # +inf
    FN(Func.DIV, CONST(0.0), CONST(0.0)),  # NaN
    FN(Func.LOG, CONST(-1.0)),  # NaN
    FN(Func.ADD, FN(Func.EXP, CONST(1e30)), FN(Func.MUL, VAR(1), CONST(2.0))),  # +inf
    FN(Func.MUL, FN(Func.LOG, CONST(0.0)), VAR(2)),  # ±inf (sign of x2)
]
_NAN_INF_USED = (Func.ADD, Func.MUL, Func.DIV, Func.LOG, Func.EXP)

#: Leaf-only population: no function node at all.  The CONST/VAR VALUES look
#: like function ids (3.0, 28.0, 1.0), which is exactly what must not leak into
#: the used set -- the id comes from FUNCTION nodes only.
_LEAF_ONLY_SPECS = [VAR(0), CONST(1.5), VAR(2), CONST(-0.25), VAR(1), CONST(28.0)]

#: IF-only population: every node is the ternary (id 0), which the graph emits
#: separately, so the used set is EMPTY too.
_IF_ONLY_SPECS = [FN(Func.IF, VAR(0), VAR(1), VAR(2)), FN(Func.IF, CONST(1.0), CONST(2.0), CONST(3.0))] * 3

#: Ids outside the candidate range: 29 is unmatched everywhere (value 0.0), a
#: bigger float clamps to 29, a negative one clamps to 0, and 0 on a BFUNC /
#: TFUNC node is the ternary, which is not a candidate id either.  ``27.9``
#: truncates INTO range (27 = SQRT): the one used id.
_UNMATCHED_ID_SPECS = [
    RAW(NType.UFUNC, 29.0, VAR(0)),
    RAW(NType.UFUNC, 33.5, VAR(1)),
    RAW(NType.BFUNC, -3.0, VAR(0), VAR(1)),
    RAW(NType.TFUNC, 0.0, VAR(0), VAR(1), VAR(2)),
    RAW(NType.UFUNC, 27.9, VAR(2)),
    FN(Func.SQRT, VAR(0)),
]
_UNMATCHED_ID_USED = (Func.SQRT,)

#: ``(name, specs, expected used ids)`` of the fixtures that share the
#: ``EDGE_POP_SIZE``/``EDGE_ROWS`` shape -- one full-chain compile covers them all.
_EDGE_FIXTURES: Tuple[Tuple[str, Sequence[Any], Tuple[int, ...]], ...] = (
    ("leaf_only", _LEAF_ONLY_SPECS, ()),
    ("if_only", _IF_ONLY_SPECS, ()),
    ("unmatched_ids", _UNMATCHED_ID_SPECS, _UNMATCHED_ID_USED),
    ("nan_inf", _NAN_INF_SPECS, _NAN_INF_USED),
    ("wide", _WIDE_SPECS, _WIDE_USED),
)
#: The subsets whose used set is EMPTY: no candidate op is emitted at all, and
#: the oracle cross-check applies (their values must still be the real ones).
_DEGENERATE = ("leaf_only", "if_only", "unmatched_ids")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rows(specs: Sequence[Any], pad_to: int = ROW_LEN, output_len: int = 1):
    """``(node_value, node_type, subtree_size)`` CPU rows from specs."""
    return eq.rows_from_specs(specs, output_len=output_len, pad_to=pad_to)


def _x(rows: int, seed: int, input_len: int = INPUT_LEN) -> np.ndarray:
    """Deterministic ``(rows, input_len)`` float32 host input."""
    return np.ascontiguousarray(eq.x_random(rows, input_len, seed=seed).numpy(), dtype=np.float32)


def _labels(rows: int, seed: int) -> np.ndarray:
    """Deterministic ``(rows, 1)`` float32 host labels."""
    return np.ascontiguousarray(eq.x_random(rows, 1, seed=seed).numpy(), dtype=np.float32)


def _plan(
    rows3: Sequence[Any],
    n_rows: int,
    *,
    prune: Any = None,
    variant: str = "value",
    per_row: bool = False,
    budget_mb: float = BUDGET_MB,
    use_mse: bool = True,
    update_mode: str = UPDATE_MODE,
) -> Any:
    """Plan one fixture; every knob that could leak from the environment is explicit."""
    values, types, sizes = rows3
    return D.plan_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        n_rows,
        per_row=per_row,
        max_tree_len=ROW_LEN,
        budget_mb=budget_mb,
        update_mode=update_mode,
        backend=BACKEND,
        variant=variant,
        use_mse=use_mse,
        prune=prune,
    )


def _pruned_and_full_plans(rows3: Sequence[Any], n_rows: int, **kwargs: Any) -> Tuple[Any, Any]:
    """``(pruned, full)`` plan pair of one population, with the non-vacuity guard."""
    pruned = _plan(rows3, n_rows, prune=True, **kwargs)
    full = _plan(rows3, n_rows, prune=False, **kwargs)
    used = tuple(int(v) for v in pruned.used_func_ids)
    assert 0 < len(used) < len(D.FULL_FUNC_IDS), (
        f"the fixture uses {len(used)} of {len(D.FULL_FUNC_IDS)} candidate ids "
        f"({used}): the pruned/full comparison would be vacuous"
    )
    assert tuple(full.used_func_ids) == D.FULL_FUNC_IDS
    assert used != tuple(full.used_func_ids)
    return pruned, full


def _full_func_ids() -> Tuple[int, ...]:
    """The chain ``EVOGP_DENSE_PRUNE=0`` restores: every id ``1..28``."""
    assert D.FULL_FUNC_IDS == tuple(range(1, 29))
    return D.FULL_FUNC_IDS


def _update_mode() -> str:
    """``grow`` (the default): the emission count below depends on it."""
    assert D.DEFAULT_UPDATE_MODE == UPDATE_MODE
    return D.DEFAULT_UPDATE_MODE


def _use_dense_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt into dense explicitly: the ``auto`` rows/population rules are not this file's subject."""
    monkeypatch.setenv(D.EVAL_MODE_ENV, "dense")


def _build(plan: Any) -> Any:
    return D.build_dense(plan, backend=BACKEND)


def _run_value(plan: Any, x: np.ndarray) -> np.ndarray:
    """Build-or-hit + run the VALUE variant through the plan API (host result)."""
    executable = _build(plan)
    backend_name, device, _ = E._resolve_backend(BACKEND, None)
    return D.run_plan(plan, executable, x, backend_name=backend_name, device=device)


def _run_loss(plan: Any, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build-or-hit + run the LOSS variant through the plan API (host result)."""
    executable = _build(plan)
    backend_name, device, _ = E._resolve_backend(BACKEND, None)
    return D.run_loss_plan(plan, executable, x, labels, backend_name=backend_name, device=device)


def _evaluate_dense(rows3: Sequence[Any], x: np.ndarray, *, per_row: bool) -> torch.Tensor:
    """The public value API of the dense route (plan + build + run + torch assembly)."""
    values, types, sizes = rows3
    return D.evaluate_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        per_row=per_row,
        backend=BACKEND,
        max_tree_len=ROW_LEN,
        budget_mb=BUDGET_MB,
        update_mode=UPDATE_MODE,
    )


def _sr_fitness_dense(
    rows3: Sequence[Any], x: np.ndarray, labels: np.ndarray, use_mse: bool
) -> torch.Tensor:
    """The public SR-loss API of the dense route (in-graph loss, ``(pop,)`` result)."""
    values, types, sizes = rows3
    return D.sr_fitness_dense(
        values,
        types,
        sizes,
        INPUT_LEN,
        1,
        x,
        labels,
        use_mse,
        backend=BACKEND,
        max_tree_len=ROW_LEN,
        budget_mb=BUDGET_MB,
        update_mode=UPDATE_MODE,
    )


def _as_f32(value: Any) -> np.ndarray:
    """Host float32 array of a torch tensor / numpy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def assert_bit_exact(pruned: Any, full: Any, label: str) -> None:
    """Bit-exact (int32 view) equality of the pruned and full-chain results.

    ``float32`` bit equality rather than a tolerance: the lever changes *which*
    ops are emitted, so the two graphs must agree to the last bit -- NaN
    payloads, ``±inf`` and ``-0.0`` included (``np.array_equal`` on the int32
    views is NaN-safe and sign-of-zero-sensitive by construction).
    """
    a, b = _as_f32(pruned), _as_f32(full)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    flat_a, flat_b = a.reshape(-1).view(np.int32), b.reshape(-1).view(np.int32)
    diff = flat_a != flat_b
    if diff.any():
        index = int(np.flatnonzero(diff)[0])
        pytest.fail(
            f"{label}: the pruned and full-chain graphs differ at {int(diff.sum())} of "
            f"{a.size} elements; first mismatch at {np.unravel_index(index, a.shape)}: "
            f"{a.reshape(-1)[index]!r} (pruned) vs {b.reshape(-1)[index]!r} (full) -- "
            "pruning must change the emitted ops, never the values"
        )


def _reference_value(rows3: Sequence[Any], x: np.ndarray) -> torch.Tensor:
    """The independent plain-torch oracle's ``(pop, N, 1)`` prediction."""
    values, types, sizes = rows3
    return eq.reference.batch_forward_from_rows(
        values, types, sizes, torch.from_numpy(np.ascontiguousarray(x)), out_len=1
    )


def _emitted_positions(plan: Any) -> int:
    """Positions the candidate chain is emitted for (battery helper).

    In ``grow`` mode the LAST position starts from an empty buffer, where a
    function node could not have children (decode-time validation rejects it), so
    the body emits no candidate/ternary there; the other modes start from a full
    buffer and emit at every position.
    """
    if plan.update_mode == "grow":
        return max(0, int(plan.trace_len) - 1)
    return int(plan.trace_len)


def _counted_build(plan: Any) -> Tuple[Any, Dict[str, int]]:
    """Cold-build ``plan`` counting the emitted candidate ops (battery helper).

    The trace runs INSIDE ``build_cached``, so the cache is cleared first and the
    build is asserted to be a MISS: the counts are the graph's *emitted* ops, not
    a hit's zeros.  ``_dense_body`` resolves ``_emit_unary``/``_emit_binary``/
    ``_emit_ternary`` as module globals while tracing, so patching them on this
    module counts exactly the candidate chain.
    """
    counts = {"unary": 0, "binary": 0, "ternary": 0}
    originals = (D._emit_unary, D._emit_binary, D._emit_ternary)

    def unary(function_id: int, operand: Any) -> Any:
        counts["unary"] += 1
        return originals[0](function_id, operand)

    def binary(function_id: int, left: Any, right: Any) -> Any:
        counts["binary"] += 1
        return originals[1](function_id, left, right)

    def ternary(first: Any, second: Any, third: Any) -> Any:
        counts["ternary"] += 1
        return originals[2](first, second, third)

    D._emit_unary, D._emit_binary, D._emit_ternary = unary, binary, ternary
    try:
        E.clear_cache()
        misses = E.cache_stats()["misses"]
        executable = _build(plan)
        assert E.cache_stats()["misses"] == misses + 1, (
            "the counted build was a cache HIT, so no op was traced: clear the cache "
            "(or use a fresh cfg) before counting"
        )
    finally:
        D._emit_unary, D._emit_binary, D._emit_ternary = originals
    counts["candidates"] = counts["unary"] + counts["binary"]
    return executable, counts


# --------------------------------------------------------------------------- #
# 1. resolve_prune: the knob's semantics
# --------------------------------------------------------------------------- #
def test_prune_env_name_and_default_are_documented(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EVOGP_DENSE_PRUNE``; unset/empty/whitespace means the default (pruned)."""
    assert D.PRUNE_ENV == "EVOGP_DENSE_PRUNE"
    assert D.DEFAULT_PRUNE is True
    assert _full_func_ids() == tuple(range(1, 29))

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    assert D.resolve_prune() is True
    assert D.resolve_prune() is D.DEFAULT_PRUNE

    for raw in ("", "   ", "\t\n"):
        monkeypatch.setenv(D.PRUNE_ENV, raw)
        assert D.resolve_prune() is True, f"{D.PRUNE_ENV}={raw!r} must mean the default"
        assert D.resolve_prune() is D.DEFAULT_PRUNE


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", " 0 ", "FALSE", "No", "OFF", "\tOff\n"])
def test_resolve_prune_false_spellings(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Every false spelling (case-insensitive, trimmed) restores the full chain."""
    monkeypatch.setenv(D.PRUNE_ENV, raw)
    assert D.resolve_prune() is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", " 1 ", "TRUE", "Yes", "ON", "\tOn\n"])
def test_resolve_prune_true_spellings(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Every true spelling (case-insensitive, trimmed) keeps pruning on."""
    monkeypatch.setenv(D.PRUNE_ENV, raw)
    assert D.resolve_prune() is True


@pytest.mark.parametrize("raw", ["maybe", "2", "on off", "tru", "0.0", "prune"])
def test_resolve_prune_rejects_unknown_spellings(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Anything else is a ValueError naming the environment variable."""
    monkeypatch.setenv(D.PRUNE_ENV, raw)
    with pytest.raises(ValueError) as excinfo:
        D.resolve_prune()
    assert D.PRUNE_ENV in str(excinfo.value), (
        f"the error must name {D.PRUNE_ENV}, got: {excinfo.value}"
    )
    assert raw.strip().lower() in str(excinfo.value)


def test_resolve_prune_explicit_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``prune=`` argument beats the environment, both directions."""
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    assert D.resolve_prune(True) is True
    monkeypatch.setenv(D.PRUNE_ENV, "1")
    assert D.resolve_prune(False) is False
    # ... even when the environment value is one that would raise on its own.
    monkeypatch.setenv(D.PRUNE_ENV, "maybe")
    assert D.resolve_prune(True) is True
    assert D.resolve_prune(False) is False
    # booleans are coerced, not parsed as strings.
    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    assert D.resolve_prune(0) is False
    assert D.resolve_prune(1) is True


# --------------------------------------------------------------------------- #
# 2. DensePrep.used_func_ids / the plan's cfg entry
# --------------------------------------------------------------------------- #
def test_used_func_ids_is_the_exact_canonical_union() -> None:
    """The set is the sorted-unique union of the DECODED function nodes, ``1..28``."""
    cases = (
        ("narrow", _MAIN_SPECS, _MAIN_USED),
        ("other", _OTHER_SPECS, _OTHER_USED),
        ("wide", _WIDE_SPECS, _WIDE_USED),
        ("nan_inf", _NAN_INF_SPECS, _NAN_INF_USED),
    )
    for name, specs, expected in cases:
        prep = D.prepare_dense(*_rows(specs)[:3], INPUT_LEN, 1, backend=BACKEND)
        used = prep.used_func_ids
        assert isinstance(used, tuple), f"{name}: used_func_ids must be a tuple, got {type(used)}"
        assert used == tuple(expected), f"{name}: {used} != {tuple(expected)}"
        assert used == tuple(sorted(set(used))), f"{name}: {used} is not canonical"
        assert all(1 <= int(v) <= 28 for v in used), f"{name}: ids outside 1..28: {used}"
        assert set(used) <= set(_full_func_ids()), f"{name}: {used} escapes FULL_FUNC_IDS"
        # the non-vacuity guard: a real GP population uses a few ops of the 28.
        assert 0 < len(used) < len(_full_func_ids()), (
            f"{name}: {len(used)} of {len(_full_func_ids())} ids would make the A/B vacuous"
        )
    # the wide fixture really is the near-full case, the narrow one really is not.
    assert len(_WIDE_USED) > len(_MAIN_USED)
    assert len(_WIDE_USED) == 18


def test_used_func_ids_ignores_the_row_tail() -> None:
    """Only nodes INSIDE the declared tree length contribute (row-tail junk does not)."""
    values, types, sizes = _rows(_MAIN_SPECS)
    clean = D.prepare_dense(values, types, sizes, INPUT_LEN, 1, backend=BACKEND)
    assert clean.used_func_ids == tuple(_MAIN_USED)

    def polluted_tail(row_values, row_types, row_sizes, function_id: float):
        """Same rows, with function-type junk past every row's tree length."""
        junk_values, junk_types = row_values.clone(), row_types.clone()
        for row, length in enumerate(row_sizes[:, 0].tolist()):
            junk_types[row, int(length) :] = NType.UFUNC
            junk_values[row, int(length) :] = function_id
        return junk_values, junk_types

    # SQRT (27) appears in the tail only: the used set must not notice it.
    junk_values, junk_types = polluted_tail(values, types, sizes, 27.0)
    assert (
        D.prepare_dense(junk_values, junk_types, sizes, INPUT_LEN, 1, backend=BACKEND).used_func_ids
        == tuple(_MAIN_USED)
    ), "row-tail junk leaked into the used-id set"

    # ... and a leaf-only population with the same polluted tail stays empty.
    leaf_values, leaf_types, leaf_sizes = _rows(_LEAF_ONLY_SPECS)
    leaf_values, leaf_types = polluted_tail(leaf_values, leaf_types, leaf_sizes, 12.0)
    assert (
        D.prepare_dense(leaf_values, leaf_types, leaf_sizes, INPUT_LEN, 1, backend=BACKEND).used_func_ids
        == ()
    ), "row-tail function nodes must not create a used id"


def test_used_func_ids_empty_for_leaves_and_unmatched_ids() -> None:
    """No in-range function node -> ``()``; a value that only LOOKS like an id is ignored."""
    specs_by_name = {name: specs for name, specs, _ in _EDGE_FIXTURES}
    for name, want in (
        ("leaf_only", ()),  # no function node at all
        ("if_only", ()),  # IF (id 0) is emitted separately, never as a candidate
        ("unmatched_ids", (Func.SQRT,)),  # ids 0 / 29 / clamped ones match no candidate
    ):
        prep = D.prepare_dense(*_rows(specs_by_name[name])[:3], INPUT_LEN, 1, backend=BACKEND)
        assert prep.used_func_ids == want, f"{name}: {prep.used_func_ids} != {want}"
        assert set(prep.used_func_ids) <= set(_full_func_ids())

    # The leaf-only rows are VAR/CONST nodes whose VALUES look like function ids
    # (1.0 and 28.0 among them): only function NODES contribute, so the set stays
    # empty.
    values, types, sizes = _rows(specs_by_name["leaf_only"])
    assert set(types[:, 0].tolist()) == {NType.VAR, NType.CONST}
    assert float(values[:, 0].max()) == 28.0, "the fixture must carry an id-looking value"
    assert D.prepare_dense(values, types, sizes, INPUT_LEN, 1, backend=BACKEND).used_func_ids == ()

    # The unmatched-id fixture DOES contain function nodes (so "no function node"
    # is not the reason its set is small): 27.9 truncates into range, everything
    # else (0 / negative / >= 29) is no candidate id at all.
    unmatched = _rows(specs_by_name["unmatched_ids"])
    root_types = unmatched[1][:, 0].tolist()
    assert set(root_types) <= {NType.UFUNC, NType.BFUNC, NType.TFUNC} and NType.UFUNC in root_types
    assert D.prepare_dense(*unmatched[:3], INPUT_LEN, 1, backend=BACKEND).used_func_ids == (Func.SQRT,)


def test_plan_used_ids_and_cfg_follow_the_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """``plan.used_func_ids`` == the static ``cfg[6]`` entry the graph specialises on."""
    rows3 = _rows(_MAIN_SPECS)
    expected = tuple(_MAIN_USED)

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _plan(rows3, N_ROWS)
    assert pruned.prune is True and pruned.used_func_ids == expected
    assert tuple(pruned.cfg[6]) == expected, "the used-id set must be the cfg entry (cache key)"
    assert pruned.specs[-1] == pruned.cfg, "the trailing static build_cached argument is cfg"

    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _plan(rows3, N_ROWS)
    assert full.prune is False and tuple(full.used_func_ids) == _full_func_ids()
    assert tuple(full.cfg[6]) == _full_func_ids()
    assert full.specs[-1] == full.cfg

    # the explicit argument wins over the environment, both directions.
    assert _plan(rows3, N_ROWS, prune=True).used_func_ids == expected
    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    assert tuple(_plan(rows3, N_ROWS, prune=False).used_func_ids) == _full_func_ids()
    # only the candidate set changed: every other static cfg entry is identical.
    assert pruned.cfg[:6] == full.cfg[:6] and pruned.cfg[7:] == full.cfg[7:]


# --------------------------------------------------------------------------- #
# 3. pruned vs full chain: BIT-EXACT results
# --------------------------------------------------------------------------- #
def test_prune_value_route_is_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """``evaluate_dense``/``run_plan``, ``per_row=False``: default pruned vs ``=0``."""
    rows3 = _rows(_MAIN_SPECS)
    x = _x(N_ROWS, seed=11)
    _use_dense_mode(monkeypatch)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, N_ROWS)
    assert pruned_plan.p_chunks == 1 and pruned_plan.n_chunks == 1, (
        "the unchunked route is what this test compares (see the chunked test for the rest)"
    )

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _evaluate_dense(rows3, x, per_row=False)
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _evaluate_dense(rows3, x, per_row=False)

    assert tuple(pruned.shape) == (POP_SIZE, N_ROWS, 1), tuple(pruned.shape)
    assert_bit_exact(pruned, full, "value route, per_row=False")
    # ... and the plan API's raw rows agree with the assembled tensor.
    assert_bit_exact(_run_value(full_plan, x), _as_f32(full)[:, :, 0], "value route (run_plan)")


def test_default_env_route_builds_the_pruned_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env unset, the API route compiles the PRUNED cfg -- observed in the cache.

    Values cannot show this (pruning is bit-exact by design), so the evidence is
    the compile cache: after a cold clear, ``evaluate_dense`` with
    ``EVOGP_DENSE_PRUNE`` unset must leave the explicit pruned plan of the same
    shape as a HIT -- a full-chain default would have left a second miss here.
    """
    rows3 = _rows(_MAIN_SPECS)
    x = _x(N_ROWS, seed=19)
    _use_dense_mode(monkeypatch)
    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, N_ROWS)
    assert tuple(full_plan.used_func_ids) == _full_func_ids()

    E.clear_cache()  # deterministic: whatever ran before, the next build is a cold miss
    result = _evaluate_dense(rows3, x, per_row=False)  # env unset -> the pruned route
    after_api = E.cache_stats()
    assert after_api["misses"] == 1, (
        f"the env-unset API call must compile exactly one graph, got {after_api}"
    )
    _build(pruned_plan)
    after_plan = E.cache_stats()
    assert after_plan["misses"] == 1 and after_plan["hits"] == 1, (
        "the env-unset evaluate_dense did not use the PRUNED cfg: the explicit pruned "
        f"plan rebuilt instead of hitting ({after_plan}) -- is DEFAULT_PRUNE still on?"
    )
    assert_bit_exact(result, _run_value(pruned_plan, x)[:, :, None], "env-unset API route")


def test_prune_value_route_per_row_is_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """``per_row=True`` (one row per tree, ``x`` has ``pop_size`` rows) is bit-exact too."""
    rows3 = _rows(_MAIN_SPECS)
    x = _x(POP_SIZE, seed=12)
    _use_dense_mode(monkeypatch)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, POP_SIZE, per_row=True)
    # ``per_row=True`` makes the graph's own N equal 1 (one x row per tree).
    assert pruned_plan.per_row is True and pruned_plan.n_rows == 1
    assert pruned_plan.p_chunks == 1 and pruned_plan.n_chunks == 1

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _evaluate_dense(rows3, x, per_row=True)
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _evaluate_dense(rows3, x, per_row=True)

    assert tuple(pruned.shape) == (POP_SIZE, 1), tuple(pruned.shape)
    assert_bit_exact(pruned, full, "value route, per_row=True")
    assert_bit_exact(_run_value(pruned_plan, x), _as_f32(pruned), "per_row run_plan")


@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
def test_prune_loss_route_is_bit_exact(monkeypatch: pytest.MonkeyPatch, use_mse: bool) -> None:
    """``sr_fitness_dense`` (the in-graph SR loss) is bit-exact for MSE and MAE."""
    rows3 = _rows(_MAIN_SPECS)
    x = _x(N_ROWS, seed=13)
    labels = _labels(N_ROWS, seed=14)
    _use_dense_mode(monkeypatch)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, N_ROWS, variant="loss", use_mse=use_mse)
    assert pruned_plan.cfg[7] is use_mse, "use_mse is the loss variant's static cfg slot"

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _sr_fitness_dense(rows3, x, labels, use_mse)
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _sr_fitness_dense(rows3, x, labels, use_mse)

    assert tuple(pruned.shape) == (POP_SIZE,), tuple(pruned.shape)
    assert_bits = _as_f32(pruned)
    assert bool(np.isfinite(assert_bits).all()), f"loss must be finite here, got {assert_bits}"
    assert bool((assert_bits >= 0.0).all()), f"the SR loss is a POSITIVE loss, got {assert_bits}"
    assert_bit_exact(pruned, full, f"loss route, {'MSE' if use_mse else 'MAE'}")
    # the plan API (``run_loss_plan``) agrees with the public one.
    assert_bit_exact(_run_loss(full_plan, x, labels), assert_bits, "loss route (run_loss_plan)")


@pytest.mark.parametrize("name", [case[0] for case in _EDGE_FIXTURES])
def test_prune_value_route_bit_exact_per_fixture(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """The edge fixtures (empty used set, unmatched ids, NaN/inf, near-full) are bit-exact.

    All of them share one shape, so the FULL-chain graph is a single compile; the
    pruned side is one compile per distinct used-id set.  The degenerate subsets
    (no in-range function node -> no candidate emitted at all) are additionally
    cross-checked against the independent plain-torch oracle: their values must
    stay the real ones.
    """
    specs, expected = next((s, e) for n, s, e in _EDGE_FIXTURES if n == name)
    rows3 = _rows(specs)
    x = _x(EDGE_ROWS, seed=15)
    _use_dense_mode(monkeypatch)
    prep = D.prepare_dense(*rows3[:3], INPUT_LEN, 1, backend=BACKEND)
    assert prep.used_func_ids == tuple(expected), (
        f"{name}: {prep.used_func_ids} != {tuple(expected)}"
    )

    pruned_plan = _plan(rows3, EDGE_ROWS, prune=True)
    full_plan = _plan(rows3, EDGE_ROWS, prune=False)
    assert tuple(pruned_plan.used_func_ids) == tuple(expected)
    assert tuple(full_plan.used_func_ids) == _full_func_ids()
    if expected:
        assert 0 < len(expected) < len(_full_func_ids())

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _evaluate_dense(rows3, x, per_row=False)
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _evaluate_dense(rows3, x, per_row=False)
    assert_bit_exact(pruned, full, f"value route, {name}")

    # the plan API's rows are the same numbers, and the oracle is the ground truth
    # for the degenerate ("no candidate emitted") fixtures.
    rows = _run_value(pruned_plan, x)
    assert_bit_exact(rows, _as_f32(pruned)[:, :, 0], f"value route (run_plan), {name}")
    if name in _DEGENERATE:
        eq.assert_close_with_placement(
            torch.from_numpy(np.ascontiguousarray(rows))[:, :, None],
            _reference_value(rows3, x),
            context=f"dense value route vs oracle ({name})",
        )


def test_prune_is_bit_exact_with_nan_and_inf(monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN/±inf placement and payloads are identical: the two graphs agree bit for bit."""
    rows3 = _rows(_NAN_INF_SPECS)
    x = _x(EDGE_ROWS, seed=16)
    _use_dense_mode(monkeypatch)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, EDGE_ROWS)
    assert tuple(pruned_plan.used_func_ids) == tuple(_NAN_INF_USED)

    monkeypatch.delenv(D.PRUNE_ENV, raising=False)
    pruned = _evaluate_dense(rows3, x, per_row=False)
    monkeypatch.setenv(D.PRUNE_ENV, "0")
    full = _evaluate_dense(rows3, x, per_row=False)

    as_f32 = _as_f32(pruned)
    # non-vacuity: the fixture really exercises the NaN / +inf / -inf placement.
    assert bool(np.isnan(as_f32).any()), "fixture produces no NaN"
    assert bool(np.isposinf(as_f32).any()), "fixture produces no +inf"
    assert bool(np.isneginf(as_f32).any()), "fixture produces no -inf"
    assert_bit_exact(pruned, full, "NaN/inf value route")


def test_prune_chunked_plans_are_bit_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A P- AND N-chunked plan of each variant is bit-exact (chunk assembly included)."""
    rows3 = _rows(_MAIN_SPECS)
    x = _x(EDGE_ROWS, seed=17)
    labels = _labels(EDGE_ROWS, seed=18)
    _use_dense_mode(monkeypatch)

    def chunked_budget(**kwargs: Any) -> float:
        """The largest budget of the ladder that splits BOTH axes (fewest chunks)."""
        for budget in (0.0024, 0.0023, 0.002, 0.0012, 0.0008, 0.0004):
            plan = _plan(rows3, EDGE_ROWS, prune=True, budget_mb=budget, **kwargs)
            if plan.p_chunks > 1 and plan.n_chunks > 1:
                return budget
        pytest.fail("no budget of the ladder produced a P+N-chunked plan")

    for kwargs, runner in (
        ({"variant": "value"}, lambda plan: _run_value(plan, x)),
        ({"variant": "loss"}, lambda plan: _run_loss(plan, x, labels)),
    ):
        budget = chunked_budget(**kwargs)
        pruned_plan = _plan(rows3, EDGE_ROWS, prune=True, budget_mb=budget, **kwargs)
        full_plan = _plan(rows3, EDGE_ROWS, prune=False, budget_mb=budget, **kwargs)
        assert pruned_plan.p_chunks > 1 and pruned_plan.n_chunks > 1, (
            f"{kwargs}: the plan must be chunked on BOTH axes"
        )
        assert pruned_plan.p_chunks * pruned_plan.n_chunks <= 40, (
            "keep the number of chunks (and so etl.run calls) small"
        )
        assert pruned_plan.p_chunk < pruned_plan.pop_size
        used = tuple(int(v) for v in pruned_plan.used_func_ids)
        assert 0 < len(used) < len(_full_func_ids())
        assert tuple(full_plan.used_func_ids) == _full_func_ids()

        pruned = runner(pruned_plan)
        full = runner(full_plan)
        assert_bit_exact(pruned, full, f"chunked {kwargs['variant']} route")
        if kwargs["variant"] == "value":
            assert _as_f32(pruned).shape == (POP_SIZE, EDGE_ROWS)


# --------------------------------------------------------------------------- #
# 4. the emitted candidate chain / the compile cache key
# --------------------------------------------------------------------------- #
def test_pruned_graph_emits_only_the_used_candidate_chain() -> None:
    """Counted ops: ``positions x |used|`` candidates (``x 28`` for the full chain)."""
    assert _update_mode() == "grow"
    rows3 = _rows(_MAIN_SPECS)
    pruned_plan, full_plan = _pruned_and_full_plans(rows3, N_ROWS)
    used = tuple(int(v) for v in pruned_plan.used_func_ids)
    positions = _emitted_positions(pruned_plan)
    assert positions == ROW_LEN - 1, "grow mode emits at every position but the last"

    _, pruned_counts = _counted_build(pruned_plan)
    _, full_counts = _counted_build(full_plan)

    assert pruned_counts["candidates"] == positions * len(used), (
        f"pruned={pruned_counts['candidates']} candidates, expected "
        f"{positions} positions x {len(used)} ids"
    )
    assert full_counts["candidates"] == positions * len(_full_func_ids()), (
        f"the full chain must emit every candidate: {full_counts}"
    )
    assert pruned_counts["candidates"] < full_counts["candidates"]
    # the arity split is the emitted op's: binary ids are the <= 13 ones.
    assert pruned_counts["binary"] == positions * len([v for v in used if v <= 13])
    assert pruned_counts["unary"] == positions * len([v for v in used if v > 13])
    # id 0 (IF) is the separately emitted ternary, once per emitted position.
    assert pruned_counts["ternary"] == positions
    assert full_counts["ternary"] == positions


def test_used_id_set_is_part_of_the_compile_cache_key() -> None:
    """Same shape + same used set = a hit; a different set = exactly one new build.

    The set is a static ``cfg`` entry (see ``test_plan_used_ids_and_cfg_follow_the_knob``),
    which is why a fresh population of the same shape recompiles whenever its
    id set differs -- and why the pruned and the full-chain graphs of ONE
    population coexist in the shared cache (neither is assumed to be evicted).
    """
    main_rows = _rows(_MAIN_SPECS)
    other_rows = _rows(_OTHER_SPECS)
    plan_a1 = _plan(main_rows, N_ROWS, prune=True)
    plan_a2 = _plan(main_rows, N_ROWS, prune=True)
    plan_b = _plan(other_rows, N_ROWS, prune=True)
    plan_b_full = _plan(other_rows, N_ROWS, prune=False)

    def shape(plan: Any) -> Tuple[Any, ...]:
        return (
            plan.trace_len,
            plan.pop_size,
            plan.p_chunk,
            plan.n_chunk,
            plan.input_len,
            plan.update_mode,
            plan.variant,
        )

    assert shape(plan_a1) == shape(plan_b), "the fixtures must share the shape signature"
    assert plan_a1.used_func_ids != plan_b.used_func_ids
    assert plan_a1.used_func_ids == tuple(_MAIN_USED)
    assert plan_b.used_func_ids == tuple(_OTHER_USED)
    assert 0 < len(plan_b.used_func_ids) < len(_full_func_ids())

    # -- the same population (a fresh plan object) never rebuilds -----------------
    executable_a = _build(plan_a1)  # hit or miss, it does not matter here
    misses = E.cache_stats()["misses"]
    executable_a2 = _build(plan_a2)
    assert E.cache_stats()["misses"] == misses, (
        "the same shape AND the same used-id set rebuilt the graph"
    )
    assert executable_a2 is executable_a, "a cache hit must hand back the same executable"

    # -- a different used-id set is exactly ONE new build ------------------------
    E.clear_cache()  # deterministic: the next build is a cold miss, whatever ran before
    executable_b = _build(plan_b)
    after_b = E.cache_stats()
    assert (after_b["misses"], after_b["hits"]) == (1, 0), after_b
    assert executable_b is not executable_a

    # ... and the full chain of the same population is ONE more entry that
    # coexists with the pruned one (both live in the cache afterwards).
    executable_b_full = _build(plan_b_full)
    after_full = E.cache_stats()
    assert after_full["misses"] == 2 and executable_b_full is not executable_b
    assert tuple(plan_b_full.used_func_ids) == _full_func_ids()

    executable_b2 = _build(plan_b)
    after_rehit = E.cache_stats()
    assert after_rehit["misses"] == 2 and after_rehit["hits"] == 1, after_rehit
    assert executable_b2 is executable_b
