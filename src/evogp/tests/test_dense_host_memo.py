"""Dense-route host memos: the executable, plan and input front tables.

The dense route pays its non-graph work host-side: the population decode
(``plan_dense`` -> 3 D2H + the vectorized decode), the ``build_cached`` key
derivation (``build_dense``), and the ``x`` / ``labels`` conversion plus the
payload uploads (``_iter_plan_chunks`` staging).  A training loop -- and every
benchmark -- repeats the very same call on the very same tensors thousands of
times, so :mod:`evogp.tree.dense_memo` puts three bounded front tables in front
of that work (``EXECUTABLES``, ``PLANS``, ``INPUTS``), all reset by the single
:func:`evogp.tree.evoxc_engine.clear_cache`.

This file pins the contract of those tables with counters (never with timing):
1. **a repeat call hits** -- the second call of the same population/dataset
   invokes neither ``plan_dense`` nor ``build_dense`` and converts nothing;
2. **a changed population misses** -- an in-place torch write bumps
   ``_version``, which is part of the identity token, and a different tensor is
   of course a different token too;
3. **``clear_cache()`` resets them** -- counters zeroed and the work re-paid;
4. **the output stays fresh and writable** -- the pipeline mutates the returned
   ``(pop_size,)`` tensor in place (``NaN -> -inf``), so a memo must never hand
   out a buffer it keeps: mutating one result never affects the next;
5. **a declined attempt pays nothing** -- the ``auto`` rows gate fires before
   the host conversion, so a ``DenseUnsupported`` call converts nothing.

Hermeticity: numpy backend only (the CPU reference route the suite pins), tiny
shapes (``P = 4``, ``N = 6``, ``L = 8``), seeded host data, no timing assertion
anywhere, and the cache checks read deltas or clear the cache explicitly.
Run (repository root)::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src <venv>/bin/python -m pytest \\
        src/evogp/tests/test_dense_host_memo.py -q
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import test_equivalence as eq  # noqa: E402  (the shared spec DSL + oracle)

from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import evoxc_engine as E  # noqa: E402
from evogp.tree.utils import Func, NType  # noqa: E402

FN, VAR, CONST = eq.fn, eq.var, eq.const

#: The host backend: no device, no compile, no iree in the picture.
BACKEND = "numpy"
INPUT_LEN = 2
ROW_LEN = 8
POP_SIZE = 4
N_ROWS = 6

_SPECS: Sequence[Any] = [
    FN(Func.ADD, VAR(0), CONST(0.5)),
    FN(Func.MUL, VAR(1), FN(Func.SIN, VAR(0))),
    FN(Func.SUB, CONST(0.25), VAR(1)),
    CONST(1.5),
]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _population() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``(values, types, sizes)`` CPU int/float population tensors."""
    values, types, sizes = eq.rows_from_specs(_SPECS, output_len=1, pad_to=ROW_LEN)
    return (
        torch.from_numpy(np.ascontiguousarray(values)),
        torch.from_numpy(np.ascontiguousarray(types)),
        torch.from_numpy(np.ascontiguousarray(sizes)),
    )


def _data(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic ``(N, input_len)`` inputs and ``(N, 1)`` labels."""
    x = torch.from_numpy(
        np.ascontiguousarray(eq.x_random(N_ROWS, INPUT_LEN, seed=seed).numpy(), dtype=np.float32)
    )
    y = torch.from_numpy(
        np.ascontiguousarray(eq.x_random(N_ROWS, 1, seed=seed + 1).numpy(), dtype=np.float32)
    )
    return x, y


class _Counter:
    """Wrap a module-level callable and count its invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> None:
        original: Callable[..., Any] = getattr(module, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)

        wrapper.__name__ = getattr(original, "__name__", name)
        monkeypatch.setattr(module, name, wrapper)


def _sr(values: Any, types: Any, sizes: Any, x: Any, y: Any) -> torch.Tensor:
    return D.sr_fitness_dense(
        values, types, sizes, INPUT_LEN, 1, x, y, True, backend=BACKEND
    )


@pytest.fixture(autouse=True)
def _clean_tables() -> Any:
    """Start and end every test from empty front tables."""
    E.clear_cache()
    yield
    E.clear_cache()


# --------------------------------------------------------------------------- #
# 1. a repeat call hits every front table
# --------------------------------------------------------------------------- #
def test_repeat_call_skips_decode_build_and_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, types, sizes = _population()
    x, y = _data()
    plans, builds, conversions = _Counter(), _Counter(), _Counter()
    plans.install(monkeypatch, D, "plan_dense")
    builds.install(monkeypatch, D, "build_dense")
    conversions.install(monkeypatch, D, "_as_numpy")
    first = _sr(values, types, sizes, x, y)
    assert plans.calls == 1 and builds.calls == 1
    # 3 population tensors (the decode) + inputs + labels
    assert conversions.calls == 5
    second = _sr(values, types, sizes, x, y)
    assert plans.calls == 1, "the population decode must be memoized"
    assert builds.calls == 1, "the build_cached key derivation must be memoized"
    assert conversions.calls == 5, "nothing may be converted again"
    assert torch.equal(first, second)


def test_a_new_population_object_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    values, types, sizes = _population()
    x, y = _data()
    plans = _Counter()
    plans.install(monkeypatch, D, "plan_dense")
    _sr(values, types, sizes, x, y)
    other = _population()  # same values, different buffers -> different tokens
    _sr(*other, x, y)
    assert plans.calls == 2


# --------------------------------------------------------------------------- #
# 2. an in-place write invalidates the population AND the dataset entries
# --------------------------------------------------------------------------- #
def test_in_place_population_write_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    values, types, sizes = _population()
    x, y = _data()
    plans = _Counter()
    plans.install(monkeypatch, D, "plan_dense")
    before = _sr(values, types, sizes, x, y)
    assert plans.calls == 1
    # torch bumps ``_version`` on an in-place write: the token changes, so a
    # half-evolved population can never be served from the old decode.
    values[0, 2] = 0.75  # the CONST leaf of row 0 (a valid edit to the tree)
    after = _sr(values, types, sizes, x, y)
    assert plans.calls == 2
    assert not torch.equal(before, after)
    _sr(values, types, sizes, x, y)
    assert plans.calls == 2, "the mutated population is memoized again"


def test_in_place_dataset_write_is_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    values, types, sizes = _population()
    x, y = _data()
    conversions = _Counter()
    conversions.install(monkeypatch, D, "_as_numpy")
    _sr(values, types, sizes, x, y)
    assert conversions.calls == 5
    x.add_(1.0)  # in-place: bumps ``_version``
    _sr(values, types, sizes, x, y)
    assert conversions.calls == 6, "only the mutated tensor is converted again"


# --------------------------------------------------------------------------- #
# 3. clear_cache() resets the front tables
# --------------------------------------------------------------------------- #
def test_clear_cache_resets_the_front_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    values, types, sizes = _population()
    x, y = _data()
    _sr(values, types, sizes, x, y)
    _sr(values, types, sizes, x, y)
    stats = D.dense_cache_stats()
    assert stats["dense_plan_hits"] >= 1
    assert stats["dense_input_hits"] >= 1
    assert stats["dense_exec_entries"] >= 1
    E.clear_cache()
    cleared = D.dense_cache_stats()
    for key in ("dense_plan", "dense_input", "dense_exec"):
        assert cleared[f"{key}_hits"] == 0
        assert cleared[f"{key}_misses"] == 0
        assert cleared[f"{key}_entries"] == 0
    plans = _Counter()
    plans.install(monkeypatch, D, "plan_dense")
    _sr(values, types, sizes, x, y)
    assert plans.calls == 1, "clear_cache() must re-pay the decode"


def test_the_specialized_memo_is_reset_too() -> None:
    # ``clear_cache()`` owns both caches: the dense hooks ride on its reset.
    E.cache_stats()  # the documented 8-key contract stays untouched
    E.clear_cache()
    assert D.DENSE_PLAN_MEMO.maxsize == D.DEFAULT_HOST_MEMO
    assert D.DENSE_EXECUTABLE_MEMO.maxsize == D.DENSE_CACHE.maxsize


# --------------------------------------------------------------------------- #
# 4. the output is fresh, writable and never shared with a memo
# --------------------------------------------------------------------------- #
def test_loss_output_is_fresh_and_writable() -> None:
    values, types, sizes = _population()
    x, y = _data()
    first = _sr(values, types, sizes, x, y)
    reference = first.clone()
    assert first.dtype == torch.float32
    assert first.requires_grad is False
    assert first.shape == (POP_SIZE,)
    assert first.is_contiguous()
    first.fill_(-1.0)  # what the pipeline does (NaN -> -inf)
    second = _sr(values, types, sizes, x, y)
    assert torch.equal(second, reference), "a memoized buffer leaked into the output"
    assert second.data_ptr() != first.data_ptr()


def test_value_output_is_fresh_and_writable() -> None:
    values, types, sizes = _population()
    x, _ = _data()
    call = lambda: D.evaluate_dense(  # noqa: E731 - one-line helper
        values, types, sizes, INPUT_LEN, 1, x, per_row=False, backend=BACKEND
    )
    first = call()
    reference = first.clone()
    assert first.shape == (POP_SIZE, N_ROWS, 1)
    assert first.dtype == torch.float32 and first.requires_grad is False
    first.fill_(-1.0)
    assert torch.equal(call(), reference)


# --------------------------------------------------------------------------- #
# 5. a declined attempt converts nothing (the gate runs before the D2H)
# --------------------------------------------------------------------------- #
def test_declined_call_converts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, types, sizes = _population()
    x, y = _data()
    monkeypatch.setenv(D.AUTO_MAX_ROWS_ENV, str(N_ROWS - 1))  # one row too many
    conversions = _Counter()
    conversions.install(monkeypatch, D, "_as_numpy")
    with pytest.raises(D.DenseUnsupported):
        _sr(values, types, sizes, x, y)
    with pytest.raises(D.DenseUnsupported):
        D.evaluate_dense(
            values, types, sizes, INPUT_LEN, 1, x, per_row=False, backend=BACKEND
        )
    assert conversions.calls == 0, "a declined attempt must not touch the data"


def test_declined_call_leaves_the_tables_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values, types, sizes = _population()
    x, y = _data()
    _sr(values, types, sizes, x, y)  # warm: one plan, one executable
    before: Dict[str, int] = D.dense_cache_stats()
    monkeypatch.setenv(D.AUTO_MAX_ROWS_ENV, str(N_ROWS - 1))
    with pytest.raises(D.DenseUnsupported):
        _sr(values, types, sizes, x, y)
    after = D.dense_cache_stats()
    assert after["dense_plan_entries"] == before["dense_plan_entries"]
    assert after["dense_exec_entries"] == before["dense_exec_entries"]


# --------------------------------------------------------------------------- #
# the front tables are bounded and switchable
# --------------------------------------------------------------------------- #
def test_front_tables_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    assert D.DENSE_EXECUTABLE_MEMO.maxsize == D.DENSE_CACHE.maxsize
    assert D.DENSE_PLAN_MEMO.maxsize == D.DEFAULT_HOST_MEMO
    assert D.DENSE_INPUT_MEMO.maxsize == D.DEFAULT_HOST_MEMO
    assert D.DEFAULT_HOST_MEMO == D.DEFAULT_HOST_MEMO


def test_a_zero_sized_front_table_disables_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    values, types, sizes = _population()
    x, y = _data()
    monkeypatch.setenv(D.HOST_MEMO_ENV, "0")
    assert D._resolve_host_memo_entries(None) == 0
    plans = _Counter()
    plans.install(monkeypatch, D, "plan_dense")
    monkeypatch.setattr(D.DENSE_PLAN_MEMO, "maxsize", 0)
    monkeypatch.setattr(D.DENSE_INPUT_MEMO, "maxsize", 0)
    _sr(values, types, sizes, x, y)
    _sr(values, types, sizes, x, y)
    assert plans.calls == 2, "a disabled table must fall back to per-call work"
