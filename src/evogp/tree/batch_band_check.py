"""Round-2 gate for the batch route's band policy (dev CLI, NOT pytest).

Runnable script (no ``test_`` prefix, no pytest import): the acceptance battery for
:mod:`evogp.tree.batch_bands` as surfaced on the PUBLIC facade
:mod:`evogp.tree.batch_engine` -- ``EVOGP_BATCH_BANDS=auto``, the dispatch-binding
clamp, the ``EVOGP_BATCH_CHUNK_BANDS`` composition and the band keys of
``describe_plan``.  Numpy backend, tiny fixtures: no GPU traffic, no device compile.

``auto``    ``bands="auto"`` reports strategy/request ``auto``, both cost estimates,
            no clamp, and every documented band key.
``default`` no knobs: the band-SIZE model still decides (``requested == "size"``,
            ``bands == ceil(n / EVOGP_BATCH_BAND_SIZE)``, ``explicit``), so round 2
            changed nothing on the default path.
``clamp``   an ``iree``/``rocm`` policy lowers an auto ``K`` to ``<= 15`` with a note
            naming the 16-binding cap, and that note reaches the report through
            ``merge_with_spec`` (stub merge: numpy never asks for more than 1 band).
``reject``  an explicit ``K`` a device cannot LOAD raises instead of clamping
            (``K <= 15`` + the remedy on rocm, ``K <= 31`` on cpu, accepted on
            numpy/vulkan), and ``resolve_band_count`` refuses ``auto`` cleanly.
``compose`` ``EVOGP_BATCH_CHUNK_BANDS``: off + ``auto`` plans one body, on plans
            three (``band_strategy == "auto"`` per chunk), and both evaluate to
            bitwise identical values.
``memo``    flipping the setting on the SAME population never serves the other
            setting's plan (the entry is keyed on the resolved ``chunk_bands``).

Usage (repo root, project venv, numpy backend on CPU)::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=/tmp/evoxc_r2:src \
      /tmp/rocvenv/bin/python -m src.evogp.tree.batch_band_check

Exit code 0 = every check passed, 1 = at least one failed, 2 = hard error.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evogp.tree import batch_bands as P  # noqa: E402
from evogp.tree import batch_engine as B  # noqa: E402
from evogp.tree import batch_knobs as K  # noqa: E402
from evogp.tree.utils import Func, NType  # noqa: E402

BACKEND, INPUT_LEN, ROW_LEN, N_ROWS = "numpy", 2, 32, 5
#: Every knob this battery states for itself (cleared first, so no ambient value).
_KNOBS = (K.BANDS_ENV, K.BAND_SIZE_ENV, K.SWITCH_OPS_ENV, K.ALIGNMENT_ENV,
          K.COMPILE_WORKERS_ENV, K.CHUNK_BANDS_ENV)
#: The round-2 band keys ``describe_plan`` must report.
_BAND_KEYS = ("bands", "cost_estimate", "flat_cost_estimate", "band_requested",
              "band_requested_bands", "band_clamped", "band_note", "band_policy",
              "chunk_bands", "chunk_band_strategies")
_ROCM = P.BandPolicy("iree", "rocm", 15, 8)  # tables: 16 bindings, 15 safe
_ARITY = {1: int(NType.UFUNC), 2: int(NType.BFUNC), 3: int(NType.TFUNC)}
_FUNCS = (Func.ADD, Func.MUL, Func.SUB, Func.LOOSE_DIV, Func.MAX, Func.MIN)
_UNARY = (Func.SIN, Func.COS, Func.TANH, Func.EXP)
#: The mixed-arity program shapes ``_programs`` cycles through.
_KINDS = (
    lambda i: (_UNARY[i % len(_UNARY)], [("var", 0)]),
    lambda i: (_FUNCS[i % len(_FUNCS)], [("var", 1), ("var", 0)]),
    lambda i: (Func.MUL, [(Func.SUB, [("var", 0), ("var", 1)]),
                          (Func.MAX, [("var", 1), ("const", 1.5 + i)])]),
    lambda i: (Func.IF, [("var", 0), ("const", 0.25 + i), ("var", 1)]),
)


class Report:
    """Collects pass/fail entries and prints one line per check."""

    def __init__(self) -> None:
        self.entries: List[bool] = []

    def add(self, name: str, ok: Any, detail: str = "") -> bool:
        self.entries.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(self.entries)

    def summary(self) -> str:
        return f"{sum(self.entries)}/{len(self.entries)} passed"


# ---------------------------------------------------------------------------
# fixtures (prefix encoder: no descriptor, hence no cuda tensors)
# ---------------------------------------------------------------------------
def _encode(program: Any, max_tree_len: int) -> Tuple[torch.Tensor, ...]:
    """One nested program (``(func, children)`` / ``("var"|"const", x)``) as a row."""
    values: List[float] = []
    types: List[int] = []
    sizes: List[int] = []

    def walk(node: Any) -> int:
        if node[0] in ("var", "const"):
            values.append(float(node[1]))
            types.append(int(NType.VAR if node[0] == "var" else NType.CONST))
            sizes.append(1)
            return 1
        index = len(values)
        values.append(float(node[0]))
        types.append(_ARITY[len(node[1])])
        sizes.append(0)
        sizes[index] = 1 + sum(walk(child) for child in node[1])
        return sizes[index]

    walk(program)
    pad = max_tree_len - len(values)
    if pad < 0:
        raise ValueError(f"program needs {len(values)} nodes, row holds {max_tree_len}")
    return tuple(  # type: ignore[return-value]
        torch.tensor(padded, dtype=dtype)
        for padded, dtype in ((values + [0.0] * pad, torch.float32),
                              (types + [0] * pad, torch.int16),
                              (sizes + [0] * pad, torch.int16)))


def _population(count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``count`` pairwise-distinct programs as the three population matrices."""
    # The per-index constant is what keeps every row a DISTINCT merged function.
    programs = [(Func.ADD, [_KINDS[i % len(_KINDS)](i), ("const", 0.5 + i)])
                for i in range(count)]
    rows = [_encode(program, ROW_LEN) for program in programs]
    return tuple(torch.stack([row[axis] for row in rows]) for axis in range(3))  # type: ignore[return-value]


def _x() -> torch.Tensor:
    """The shared ``(N_ROWS, INPUT_LEN)`` input block (fixed seed)."""
    return torch.randn(N_ROWS, INPUT_LEN, generator=torch.Generator().manual_seed(0))


def _plan(population: Tuple[torch.Tensor, ...], **knobs: Any) -> Any:
    """``plan_batch`` (value variant) on the fixture grid."""
    values, types, sizes = population
    return B.plan_batch(values, types, sizes, INPUT_LEN, 1, _x(), per_row=False,
                        backend=BACKEND, **knobs)


def _values(population: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """``evaluate_batch`` (value variant) on the fixture grid: auto, 4 workers."""
    values, types, sizes = population
    return B.evaluate_batch(values, types, sizes, INPUT_LEN, 1, _x(), per_row=False,
                            backend=BACKEND, bands=K.BANDS_AUTO, compile_workers=4)


def _set_env(name: str, value: Optional[str]) -> Optional[str]:
    """Set (``value``) or delete (``None``) ``name``; return the previous value."""
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    return previous


class _Chunked:
    """Lower ``MIN_CHUNK_FUNCTIONS`` for a tiny fixture, restoring it afterwards."""

    def __enter__(self) -> None:
        self.previous = K.MIN_CHUNK_FUNCTIONS
        K.MIN_CHUNK_FUNCTIONS = 4

    def __exit__(self, *exc: Any) -> None:
        K.MIN_CHUNK_FUNCTIONS = self.previous


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def case_auto(population: Tuple[torch.Tensor, ...], report: Report) -> None:
    """``EVOGP_BATCH_BANDS=auto``: the cost model decides, and the plan says so."""
    info = B.describe_plan(_plan(population, bands=K.BANDS_AUTO))
    report.add("auto: 6-12 distinct functions merged by the cost model",
               info["band_strategy"] == "auto" and 6 <= info["functions"] <= 12,
               f"strategy={info['band_strategy']!r} functions={info['functions']}")
    report.add("auto: requested 'auto' and both cost estimates reported",
               info["band_requested"] == "auto" and info["cost_estimate"] is not None
               and info["flat_cost_estimate"] is not None,
               f"cost={info['cost_estimate']} flat={info['flat_cost_estimate']}")
    report.add("auto: numpy is uncapped, so no clamp, no note, policy recorded",
               info["band_clamped"] is False and info["band_note"] is None
               and info["band_policy"].get("uncapped") is True,
               f"clamped={info['band_clamped']} policy={info['band_policy']}")
    report.add("auto: bands == band_count == 1 and chunk_bands mirrors it",
               info["bands"] == info["band_count"] == 1 and info["chunk_bands"] == [1],
               f"bands={info['bands']} chunk_bands={info['chunk_bands']}")
    missing = [key for key in _BAND_KEYS if key not in info]
    report.add("auto: describe_plan carries every round-2 band key", not missing,
               f"missing={missing}")


def case_default(report: Report) -> None:
    """No knobs: the band-SIZE model still decides, exactly as in round 1."""
    info = B.describe_plan(_plan(_population(40)))
    expected = -(-info["n_functions"] // K.DEFAULT_BAND_SIZE)
    report.add("default: 40 distinct functions, so the size model really bands",
               info["n_functions"] == 40 and expected == 2,
               f"n_functions={info['n_functions']} expected_K={expected}")
    report.add("default: requested 'size' and bands == ceil(n / band size)",
               info["band_requested"] == "size" and info["bands"] == expected
               and info["band_count"] == expected, f"bands={info['bands']}")
    report.add("default: the size model's 'explicit' strategy and its band sizes",
               info["band_strategy"] == "explicit"
               and len(info["band_sizes"]) == expected,
               f"strategy={info['band_strategy']!r} sizes={info['band_sizes']}")
    report.add("default: nothing is clamped and no model estimate is claimed",
               info["band_clamped"] is False and info["band_note"] is None
               and info["cost_estimate"] is None, f"cost={info['cost_estimate']}")


class _StubMerge:
    """A stand-in for ``evoxc.batch_functions``: reports the bands it was asked for."""

    def __init__(self, chosen: int) -> None:
        self.chosen, self.calls = int(chosen), []

    def __call__(self, graphs: Any, axes: Any, *, switch_ops: bool, bands: Any,
                 alignment: str) -> Any:
        self.calls.append(bands)
        answer = {"bands": self.chosen if bands == "auto" else int(bands or 1),
                  "band_strategy": "auto"}
        merged = type("Merged", (), {})()
        merged.batch_report = type("Report", (), {"as_dict": lambda _s=None: dict(answer)})
        return merged


def case_clamp(report: Report) -> None:
    """The dispatch-binding clamp: it lowers a modelled/auto count and says so."""
    report.add("clamp: the rocm policy caps 200 functions at 15 bands (16 bindings)",
               _ROCM.cap(200) == 15 and P.DISPATCH_BINDING_LIMITS["rocm"] == 16,
               f"cap={_ROCM.cap(200)} bindings={P.DISPATCH_BINDING_LIMITS['rocm']}")
    capped, note = P.clamp_auto_band_count(40, 200, _ROCM)
    report.add("clamp: an auto K of 40 is lowered to <= 15", capped <= 15, f"{capped}")
    report.add("clamp: the note names the auto choice, the 16-binding cap + remedy",
               note is not None and "chose 40 bands" in note and "16 tensors" in note
               and "K <= 15" in note, f"note={note!r}")
    below, quiet = P.clamp_auto_band_count(3, 200, _ROCM)
    report.add("clamp: a compliant auto K is left alone with no note",
               below == 3 and quiet is None, f"capped={below} note={quiet!r}")
    # The clamp + note reach the report a plan stores.  The numpy cost model never
    # asks for >1 band, so the merge callable is stubbed with a device-shaped answer:
    # same code path ``_plan_host`` runs.
    merge = _StubMerge(chosen=40)
    _merged, merged = P.merge_with_spec(
        merge, [], 0, P.BandSpec("auto", None, False, None, _ROCM),
        switch_ops=False, alignment="variant", auto_total=200)
    report.add("clamp: the report carries band_clamped + the note, re-merged at 15",
               merged["band_clamped"] is True and merged["band_note"] == note
               and merged["bands"] == 15 and merge.calls == ["auto", 15],
               f"clamped={merged['band_clamped']} bands={merged['bands']} "
               f"calls={merge.calls}")
    report.add("clamp: the AUTO spelling and its pre-clamp K survive onto the report",
               merged["band_requested"] == "auto"
               and merged["band_requested_bands"] == 40
               and merged["band_policy"]["device_kind"] == "rocm",
               f"requested={merged['band_requested']!r} "
               f"requested_bands={merged['band_requested_bands']}")


def case_reject(report: Report) -> None:
    """An explicit ``K`` a device cannot LOAD raises; the cap-less policies accept."""
    for policy, cap_text, hal in ((_ROCM, "K <= 15", "ROCm/HIP"),
                                  (P.BandPolicy("iree", "cpu", 31, 1), "K <= 31",
                                   "local-task CPU")):
        try:
            P.plan_band_spec(200, bands=48, policy=policy)
        except ValueError as exc:
            text = str(exc)
            report.add(f"reject: K=48 on {policy.label} raises naming '{cap_text}'",
                       cap_text in text and f"{K.BANDS_ENV}=48" in text
                       and "Lower it" in text and hal in text, text)
        else:
            report.add(f"reject: K=48 on {policy.label} raises", False, "no raise")
    accepted = [P.plan_band_spec(200, bands=48, policy=P.BandPolicy(name, kind, None, 1))
                .num_bands for name, kind in (("numpy", "cpu"), ("iree", "vulkan"))]
    report.add("reject: numpy and vulkan are uncapped, so K=48 is accepted",
               accepted == [48, 48], f"accepted={accepted}")
    try:
        K.resolve_band_count(10, bands=K.BANDS_AUTO)
    except (ValueError, TypeError) as exc:  # a TypeError is the regression this pins
        text = str(exc)
        report.add("reject: resolve_band_count refuses 'auto' with a ValueError",
                   isinstance(exc, ValueError) and K.BANDS_ENV in text
                   and "'auto'" in text and "cost model at merge time" in text
                   and "no band count to resolve here" in text, text)
    else:
        report.add("reject: resolve_band_count(10, bands='auto') raises", False, "no raise")
    report.add("reject: the documented band-count semantics are untouched",
               K.resolve_band_count(1000, band_size=1) == 1000
               and K.resolve_band_count(10, bands=3, band_size=1) == 3
               and K.resolve_band_count(10, band_size=1) == 10, "model + explicit K")
    try:
        K.resolve_band_count(4, bands=5)
    except ValueError as exc:
        report.add("reject: an out-of-range explicit K still names the variable",
                   K.BANDS_ENV in str(exc), str(exc))
    else:
        report.add("reject: an out-of-range explicit K still raises", False, "no raise")


def case_compose(report: Report) -> None:
    """``EVOGP_BATCH_CHUNK_BANDS``: does a band request compose with chunking?"""
    B._clear_tables()  # fresh front tables, so nothing is served from an earlier case
    population = _population(12)
    with _Chunked():
        off = B.describe_plan(_plan(population, bands=K.BANDS_AUTO, chunk_bands=False,
                                    compile_workers=4))
        on = B.describe_plan(_plan(population, bands=K.BANDS_AUTO, chunk_bands=True,
                                   compile_workers=4))
    report.add("compose: off + auto forces the single-body path",
               off["compile_workers"] == 1 and off["chunk_count"] == 1,
               f"workers={off['compile_workers']} chunks={off['chunk_count']}")
    report.add("compose: on composes the band request with three chunks",
               on["compile_workers"] == 3 and on["chunk_count"] == 3
               and on["chunk_band_strategies"] == ["auto"] * 3
               and len(on["chunk_bands"]) == 3,
               f"workers={on['compile_workers']} sizes={on['chunk_sizes']} "
               f"strategies={on['chunk_band_strategies']}")
    report.add("compose: the chunked plan keeps the global function count/order",
               on["n_functions"] == off["n_functions"] == 12
               and sum(on["chunk_sizes"]) == 12 and on["band_requested"] == "auto",
               f"n_functions={on['n_functions']} sizes={on['chunk_sizes']}")
    previous = _set_env(K.CHUNK_BANDS_ENV, None)
    with _Chunked():
        single = _values(population)
        _set_env(K.CHUNK_BANDS_ENV, "1")
        chunked = _values(population)
    _set_env(K.CHUNK_BANDS_ENV, previous)
    report.add("compose: chunked + banded values are bitwise identical to one body",
               torch.equal(single, chunked),
               f"shape={tuple(chunked.shape)} max_abs_diff="
               f"{float((single - chunked).abs().max()):.3g}")


def case_memo(report: Report) -> None:
    """The plan entry is keyed on ``chunk_bands``: no cross-serving."""
    B._clear_tables()  # empty tables, so every flip below must reach ``_plan_host``
    population = _population(12)
    plans: List[Any] = []
    original_host = B._plan_host

    def recording_plan_host(*args: Any, **kwargs: Any) -> Any:
        plan = original_host(*args, **kwargs)
        plans.append(plan)
        return plan

    def seen(index: int) -> Tuple[int, List[int]]:
        info = B.describe_plan(plans[index])
        return info["compile_workers"], info["chunk_bands"]

    previous = _set_env(K.CHUNK_BANDS_ENV, None)
    B._plan_host = recording_plan_host
    try:
        with _Chunked():
            _values(population)                     # off: one body
            built_off, workers_off, bands_off = len(plans), *seen(0)
            _set_env(K.CHUNK_BANDS_ENV, "1")
            _values(population)                     # on: must NOT reuse the off plan
            built_on, workers_on, bands_on = len(plans), *seen(1)
            _set_env(K.CHUNK_BANDS_ENV, None)
            _values(population)                     # off again: the off plan is back
            built_back = len(plans)
    finally:
        B._plan_host = original_host
        _set_env(K.CHUNK_BANDS_ENV, previous)
    report.add("memo: the off setting records one body and one chunk band count",
               built_off == 1 and workers_off == 1 and bands_off == [1],
               f"plans={built_off} workers={workers_off} chunk_bands={bands_off}")
    report.add("memo: flipping the setting REPLANS instead of serving the other plan",
               built_on == 2 and workers_on == 3 and bands_on == [1, 1, 1],
               f"plans={built_on} workers={workers_on} chunk_bands={bands_on}")
    report.add("memo: flipping back is served the original off plan (no third build)",
               built_back == built_on, f"plans={built_back} (expected {built_on})")
    with _Chunked():
        argument_off = B.describe_plan(_plan(population, bands=K.BANDS_AUTO,
                                             chunk_bands=False, compile_workers=4))
        argument_on = B.describe_plan(_plan(population, bands=K.BANDS_AUTO,
                                            chunk_bands=True, compile_workers=4))
    report.add("memo: the explicit chunk_bands argument decides the partition too",
               argument_off["compile_workers"] == 1
               and argument_on["compile_workers"] == 3,
               f"off={argument_off['compile_workers']} on={argument_on['compile_workers']}")


def main() -> int:
    """Run every case; 0 = all passed, 1 = at least one failed."""
    for name in _KNOBS:
        os.environ.pop(name, None)
    # The fixtures sit far below the route's measured break-even (64): this battery is
    # about the band policy, not about economics.
    os.environ[K.MIN_POP_ENV] = "1"
    print(f"batch_band_check: backend={BACKEND} input_len={INPUT_LEN} rows={N_ROWS} "
          f"row_len={ROW_LEN} (numpy interpreter only: no GPU traffic)")
    report = Report()
    case_auto(_population(11), report)
    case_default(report)
    case_clamp(report)
    case_reject(report)
    case_compose(report)
    case_memo(report)
    print(f"== {'OK' if report.ok else 'FAILED'}: {report.summary()} ==")
    return 0 if report.ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # a fixture error is a hard error, not a failed check
        import traceback

        traceback.print_exc()
        sys.exit(2)
