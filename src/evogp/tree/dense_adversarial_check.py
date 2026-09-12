"""Adversarial dense-vs-oracle equivalence battery (PUBLIC API + read-only oracle).

Runnable script, deliberately NOT a pytest module (no ``test_`` prefix, no pytest
import).  It complements the two existing batteries:

* ``dense_spike_check.py`` compares ``dense_engine`` internals against the
  *specialized engine* (same tensors, same backend) -- parity by construction;
* ``dense_integration_check.py`` checks the *wiring* (``EVOGP_EVAL_MODE``,
  fallback, contracts) on a few random/hand-built populations.

This script is the missing leg: it compares the **public dense path**
(``EVOGP_EVAL_MODE=dense`` forced through ``Forest.forward`` /
``Forest.batch_forward`` / ``Forest.SR_fitness`` and the ``Tree`` variants)
against the **independent plain-torch oracle** ``src/evogp/tests/reference_eval.py``
(loaded read-only by path, exactly like ``tests/test_equivalence.py`` does), on
deliberately hostile populations, and it also records the *cause class* of every
non-bit-exact finite value (folded-constant-chain vs emitted, dense-only vs
specialized-only vs both).

    # numpy battery (CPU math, but the public API's ``check_tensor`` needs a GPU)
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=3 \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        src/evogp/tree/dense_adversarial_check.py

    # one family at a time on iree (keep each process under the etl/iree
    # executable-leak budget, see dense_spike_check's robustness note)
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=3 \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        src/evogp/tree/dense_adversarial_check.py --backend iree --only zero_denominator

Flags: ``--backend numpy|iree``, ``--only FAMILY`` (repeatable, substring match on
the family name), ``--list`` (print the families and exit), ``--gpu N`` (IREE
device index), ``--max-tree-len L`` (dense trace cap, default 32),
``--iree-max-pop K`` (rows kept per family on iree; stride-sampled, default 4),
``--n-batch N``, ``--skip-sr``, ``--skip-batch``, ``--json PATH``, ``--seed S``.

What each family stresses
-------------------------
``per_row_contract``   identity trees (``VAR 0``) on adversarial per-row ``x``:
                       every tree must see its OWN row (``+-0.0``, ``+-inf``, NaN,
                       denormals) -- the contract the ``etl_engine per_row=True``
                       fix implements; dense, specialized and oracle must all
                       return exactly those rows, bit for bit.
``unary_var`` /
``unary_const``        every unary function id 14-28 (incl. the loose variants)
                       against ``+-0.0, +-1e-45, +-1e-9, +-0.5, +-1, +-2, +-1e9,
                       +-inf, NaN`` -- once with a VAR leaf (both paths emit) and
                       once with a CONST leaf (the specialized engine folds the
                       constant tree host-side, the dense graph emits it).
``binary_var``         every binary id 1-13 against zero / denormal / huge /
                       ``+-inf`` / NaN operand pairs (DIV by ``+-0.0``,
                       LOOSE_DIV through the DELTA clamp, POW/LOOSE_POW edges).
``ternary_if``         IF branch mapping (``cond > 0`` -> then, else -> else) with
                       NaN/``+-inf`` in the UNSELECTED branch, plus const-leaf IFs.
``compare``            ``< > <= >=`` at the boundaries (equal, ``+-0.0``, NaN,
                       ``+-inf``): the result must be bit-exact ``+-1``.
``maxmin``             MAX/MIN NaN rules (``a >= b ? a : b`` / ``a <= b ? a : b``).
``zero_denominator``   DIV/LOOSE_DIV/INV/LOOSE_INV/LOOSE_LOG/LOOSE_SQRT/LOOSE_POW
                       with ``+-0.0``, ``+-1e-45``, ``+-1e-9`` denominators.
``const_tree``         fully constant trees (fold-vs-emit search): LOOSE_POW(0,0),
                       DIV(1,0), LOG(0), SQRT(-1), EXP overflow, nested chains,
                       comparisons, IF, MAX(nan, x) ...
``deep_chain``         unary chains of 16-31 nodes (deep rewrites of the value
                       buffer), mixed functions, VAR and CONST bottoms.
``junk_tail``          the same tree padded with four different hostile tails
                       (zeros / NaN sentinel / hostile random junk / valid-looking
                       extension): all variants must agree bit for bit, so the
                       padding can never leak into the result.
``unknown_id``         function ids 29, 30, 40, 100, 1e6, -1, -0.5, 13.9 on unary
                       and binary nodes -> every path must yield ``0.0``.
``unknown_type``       node types 5, 6, 29, 100, 32767 (the kernel's ternary
                       branch; the function id is ignored).
``random_pop``         ``Forest.random_generate`` populations (several seeds /
                       const probabilities) through the public API.
``sr_labels``          ``SR_fitness`` MSE and MAE with ``+-0.0`` / ``+-inf`` / NaN
                       labels (masks must match exactly).
``long_tree``          trees at/over the 32-node dense trace cap: the dense path
                       must raise ``DenseUnsupported`` and the public call must
                       fall back SILENTLY to the specialized engine and match the
                       oracle.
``fallback_error``     populations the specialized decoder rejects (OUT_NODE tag,
                       malformed subtree sizes, out-of-range / non-finite variable
                       index, non-finite function id, tree longer than the row):
                       the dense path must raise ``DenseUnsupported`` and the public
                       call must raise the *same* ``ValueError`` message.
``tree_api`` / ``wiring``  the ``Tree`` entry points (1-D and 2-D ``x``, MSE/MAE
                       fitness) and the ``cache_stats`` proof that ONE dense
                       compile serves a whole population against one per tree for
                       the specialized path.

Comparison policy (mirrors ``tests/`` and the other batteries): NaN masks and the
``+-inf`` signs must match EXACTLY, the sign of every zero must match, finite
values must be within rtol/atol 1e-4, and non-bit-exact finite values are
reported individually with their cause class.  Bit-exactness is the *goal* (both
paths share the private emitters, so the only legitimate difference is a folded
constant chain resolved by the host libm vs emitted on the device); it is not the
pass criterion -- except in ``--strict-bits`` mode, where any non-bit-exact
finite value fails the run.

Exit code 0 = every check inside the policy, 1 = at least one mismatch, 2 = a
hard error (no oracle / no GPU / unexpected exception while building a case).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import pathlib
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evogp.tree import Forest, GenerateDescriptor  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402

# The tree builders are shared with the spike battery (same directory): TreeSpec
# (prefix spec with automatic subtree sizes), pack_trees (padding/junk) and the
# hostile-junk generator.  Reusing them keeps the population-construction rules
# identical across batteries instead of forking a second copy.
from evogp.tree.dense_spike_check import (  # noqa: E402
    TreeSpec,
    pack_trees,
    random_junk,
)
from evogp.tree.utils import FUNCS_NAMES, Func, NType  # noqa: E402

# --------------------------------------------------------------------------- #
# the independent oracle (READ ONLY -- never imported from evogp)
# --------------------------------------------------------------------------- #
_ORACLE_PATH = pathlib.Path(_SRC) / "evogp" / "tests" / "reference_eval.py"


def _load_oracle():
    spec = importlib.util.spec_from_file_location("evogp_reference_eval_ro", _ORACLE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise RuntimeError(f"cannot load the oracle from {_ORACLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load_oracle()

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

_RTOL = 1e-4
_ATOL = 1e-4
_DEVICE = "cuda"

#: Adversarial scalar set: signed zeros, denormals, tiny/huge values, infinities
#: and NaN.  1e-45 rounds to the smallest subnormal, which is exactly the point.
EDGE_VALUES: Tuple[float, ...] = (
    0.0,
    -0.0,
    1e-45,
    -1e-45,
    1e-9,
    -1e-9,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    1e9,
    -1e9,
    float("inf"),
    float("-inf"),
    float("nan"),
)

UNARY_FUNCS: Tuple[int, ...] = (
    int(Func.SIN),
    int(Func.COS),
    int(Func.TAN),
    int(Func.SINH),
    int(Func.COSH),
    int(Func.TANH),
    int(Func.LOG),
    int(Func.LOOSE_LOG),
    int(Func.EXP),
    int(Func.INV),
    int(Func.LOOSE_INV),
    int(Func.NEG),
    int(Func.ABS),
    int(Func.SQRT),
    int(Func.LOOSE_SQRT),
)

BINARY_FUNCS: Tuple[int, ...] = (
    int(Func.ADD),
    int(Func.SUB),
    int(Func.MUL),
    int(Func.DIV),
    int(Func.LOOSE_DIV),
    int(Func.POW),
    int(Func.LOOSE_POW),
    int(Func.MAX),
    int(Func.MIN),
    int(Func.LT),
    int(Func.GT),
    int(Func.LE),
    int(Func.GE),
)

#: Operand pairs: signed zeros, DELTA-scale denominators, denormals, huge values,
#: infinities and NaN in every position (see ENGINE_SPEC section 3).
BINARY_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, -0.0),
    (-0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.0, 1e-45),
    (1.0, 1e-45),
    (1.0, 1e-9),
    (-1.0, 1e-9),
    (1.0, -1e-9),
    (0.0, 1e9),
    (1e9, 1e9),
    (1e20, -1e20),
    (-2.0, 3.0),
    (-2.0, 0.5),
    (2.0, 100.0),
    (float("inf"), 0.0),
    (float("inf"), float("inf")),
    (float("inf"), float("-inf")),
    (float("-inf"), 1.0),
    (float("nan"), 0.0),
    (0.0, float("nan")),
    (float("nan"), float("nan")),
    (float("nan"), float("inf")),
)

_FAMILIES: Tuple[str, ...] = (
    "per_row_contract",
    "unary_var",
    "unary_const",
    "binary_var",
    "ternary_if",
    "compare",
    "maxmin",
    "zero_denominator",
    "const_tree",
    "deep_chain",
    "junk_tail",
    "unknown_id",
    "unknown_type",
    "random_pop",
    "sr_labels",
    "long_tree",
    "fallback_error",
    "tree_api",
    "wiring",
)


# --------------------------------------------------------------------------- #
# tiny utilities
# --------------------------------------------------------------------------- #


def _host(tensor: Any) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _f32(value: float) -> float:
    return float(np.float32(value))


def _fmt(value: float) -> str:
    if isinstance(value, float) and value != value:
        return "nan"
    return f"{value:.6g}"


@contextlib.contextmanager
def eval_mode(mode: str):
    """Force ``EVOGP_EVAL_MODE`` for the duration of the block (read per call)."""
    key = D.EVAL_MODE_ENV
    previous = os.environ.get(key)
    os.environ[key] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


class Report:
    """Collects pass/fail entries and prints one line per check."""

    def __init__(self, strict_bits: bool = False) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.bit_diffs: List[Dict[str, Any]] = []
        self.strict_bits = strict_bits

    def add(self, name: str, ok: bool, detail: str = "", **extra: Any) -> bool:
        entry = {"check": name, "ok": bool(ok), "detail": detail}
        entry.update(extra)
        self.entries.append(entry)
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        return ok

    def add_bit_diffs(self, entries: List[Dict[str, Any]]) -> None:
        self.bit_diffs.extend(entries)

    @property
    def ok(self) -> bool:
        if not all(entry["ok"] for entry in self.entries):
            return False
        if self.strict_bits and self.bit_diffs:
            return False
        return True


# --------------------------------------------------------------------------- #
# spec helpers (fold sensitivity)
# --------------------------------------------------------------------------- #

_ARITY_BY_KIND = {"var": 0, "const": 0, "u": 1, "b": 2, "t": 3}


def _arity_of(kind: Any) -> int:
    """Node arity the way BOTH decoders read it (unknown base type -> ternary)."""
    if isinstance(kind, str):
        return _ARITY_BY_KIND[kind]
    base = int(kind) & int(NType.TYPE_MASK)
    return {0: 0, 1: 0, 2: 1, 3: 2}.get(base, 3)


def _is_const_leaf(kind: Any) -> bool:
    if isinstance(kind, str):
        return kind == "const"
    return (int(kind) & int(NType.TYPE_MASK)) == int(NType.CONST)


def fold_sensitive(nodes: Sequence[Tuple[Any, float]]) -> Tuple[bool, bool]:
    """``(const_subtree_present, root_is_const)`` for a prefix node list.

    A function node whose whole subtree is constant is FOLDED host-side by the
    specialized engine (``etl_engine._decode_row``) while the dense graph emits
    it -- the only place where the two paths can legitimately differ (host libm
    vs device libm at f64, rounded to f32).  ``root_is_const`` additionally means
    the specialized path never compiles a graph at all.
    """
    stack: List[bool] = []  # all-const flag per completed subtree
    fold = False
    for kind, _ in reversed(nodes):
        arity = _arity_of(kind)
        if len(stack) < arity:
            raise ValueError(f"bad spec: node needs {arity} operands")
        children = [stack.pop() for _ in range(arity)]
        if arity == 0:
            const = _is_const_leaf(kind)
        else:
            const = bool(children) and all(children)
            if const:
                fold = True
        stack.append(const)
    if len(stack) != 1:
        raise ValueError("bad spec: the prefix stream does not reduce to one tree")
    return fold, bool(stack[0])


class Row:
    """One population row: a tree spec plus the ``x`` row it must be fed."""

    __slots__ = ("name", "family", "spec", "x", "junk", "fold", "const_root", "note")

    def __init__(
        self,
        name: str,
        family: str,
        spec: Any,
        x: Sequence[float],
        *,
        junk: Optional[Callable[[int, int, int], Tuple[float, int, int]]] = None,
        note: str = "",
    ) -> None:
        self.name = name
        self.family = family
        self.spec = spec
        self.x = tuple(float(v) for v in x)
        self.junk = junk
        self.note = note
        self.fold, self.const_root = fold_sensitive(spec.nodes)


class Block:
    """A population to evaluate: rows sharing ``input_len`` / row length."""

    def __init__(
        self,
        name: str,
        rows: Sequence[Row],
        input_len: int,
        row_len: int,
        *,
        note: str = "",
        group_key: Optional[Callable[[Row], str]] = None,
    ) -> None:
        self.name = name
        self.rows = list(rows)
        self.input_len = input_len
        self.row_len = row_len
        self.note = note
        #: Optional row -> group key: rows of one group are the SAME tree with
        #: different padding, so their results must be identical bit for bit.
        self.group_key = group_key
        for row in self.rows:  # one x row of exactly input_len columns per tree
            if len(row.x) != int(input_len):
                raise ValueError(
                    f"{name}: row {row.name} carries {len(row.x)} x values but the "
                    f"block declares input_len={input_len}"
                )
        self.batch_x = _batch_inputs(input_len, 6)
        self.labels = (0.0, -0.0, 1.0, -1.0, float("inf"), float("nan"))


def _batch_inputs(input_len: int, rows: int) -> Tuple[Tuple[float, ...], ...]:
    """``(rows, input_len)`` adversarial shared batch (every column hits edges)."""
    out = []
    for i in range(rows):
        out.append(tuple(EDGE_VALUES[(i + 3 * j) % len(EDGE_VALUES)] for j in range(input_len)))
    return tuple(out)


def raw_spec(nodes: Sequence[Tuple[int, float]]):
    """A prefix spec with EXPLICIT int16 node types (unknown types -> ternary).

    Same ``.tensors()`` contract as ``dense_spike_check.TreeSpec``, so it plugs
    straight into the shared ``pack_trees``; it exists because ``TreeSpec`` only
    knows the five named kinds and the adversarial families need raw type ids.
    """

    class _Raw:
        def __init__(self, items):
            self.nodes = [(int(t), float(v)) for t, v in items]

        def tensors(self):
            items = self.nodes
            n = len(items)
            sizes = np.zeros(n, dtype=np.int32)
            stack: List[int] = []
            for i in range(n - 1, -1, -1):
                arity = _arity_of(items[i][0])
                if len(stack) < arity:
                    raise ValueError(f"bad raw spec at node {i}: needs {arity} operands")
                size = 1 + sum(stack.pop() for _ in range(arity))
                sizes[i] = size
                stack.append(size)
            if len(stack) != 1:
                raise ValueError("bad raw spec: the prefix stream does not reduce")
            values = np.asarray([v for _, v in items], dtype=np.float32)
            types = np.asarray([t for t, _ in items], dtype=np.int16)
            return values, types, sizes

    return _Raw(nodes)


def build_tensors(rows: Sequence[Row], row_len: int):
    """``(values, types, sizes)`` ``(P, row_len)`` numpy arrays from rows."""
    width = len(rows[0].x) if rows else 0
    for row in rows:  # defensive: exactly one x row of ``input_len`` per tree
        if len(row.x) != width:
            raise ValueError(
                f"row {row.name} carries {len(row.x)} x values while the "
                f"population uses {width} columns"
            )
    specs = [row.spec for row in rows]

    def junk(p: int, i: int, length: int):
        hook = rows[p].junk
        if hook is None:
            return (0.0, 0, 0)
        return hook(p, i, length)

    return pack_trees(specs, row_len=row_len, junk=junk)


# --------------------------------------------------------------------------- #
# comparison policy
# --------------------------------------------------------------------------- #


def _row_index(index: Tuple[int, ...], rows: Optional[Sequence[Row]]):
    if rows is None:
        return None
    return rows[int(index[0])]


def classify(index: Tuple[int, ...], rows, dense_ok: bool, spec_ok: bool) -> Dict[str, Any]:
    """Cause class of one non-bit-exact finite element."""
    row = _row_index(index, rows)
    if row is None:
        return {"path": "unknown", "who": "unknown", "case": "?"}
    who = "both"
    if dense_ok and not spec_ok:
        who = "specialized_only"
    elif spec_ok and not dense_ok:
        who = "dense_only"
    return {
        "path": "const_folded" if row.fold else "emitted",
        "root_const": bool(row.const_root),
        "who": who,
        "case": row.name,
        "family": row.family,
    }


def compare(
    report: Report,
    name: str,
    ref: Any,
    got: Any,
    *,
    rows: Optional[Sequence[Row]] = None,
    counterpart: Optional[Any] = None,
    counterpart_name: str = "specialized",
    counterpart_ok: Optional[np.ndarray] = None,
    max_repros: int = 8,
    detail_extra: str = "",
) -> bool:
    """Compare float32 results under the battery policy.

    ``ref`` is the oracle, ``got`` the dense result, ``counterpart`` optionally
    the specialized result (bit-equality per element then classifies the cause of
    a dense-vs-oracle difference).
    """
    a = np.asarray(_host(ref), dtype=np.float32)
    b = np.asarray(_host(got), dtype=np.float32)
    if a.shape != b.shape:
        report.add(name, False, f"shape {b.shape} != {a.shape}")
        return False
    mask_nan = np.array_equal(np.isnan(a), np.isnan(b))
    mask_inf = np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(
        np.isneginf(a), np.isneginf(b)
    )
    finite = np.isfinite(a) & np.isfinite(b)
    zero_exact = True
    if (a == 0.0).any():
        zero_exact = bool(
            np.array_equal(np.signbit(a[a == 0.0]), np.signbit(b[a == 0.0]))
        )
    bits_a = a.view(np.int32)
    bits_b = b.view(np.int32)
    bit = bits_a == bits_b
    non_bit_finite = finite & ~bit
    close = (
        bool(np.allclose(a[finite], b[finite], rtol=_RTOL, atol=_ATOL))
        if finite.any()
        else True
    )
    if finite.any():
        diff = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        max_abs = float(diff.max())
        denom = np.abs(a[finite].astype(np.float64))
        rel = np.where(denom > 0, diff / np.maximum(denom, 1e-45), diff)
        max_rel = float(rel.max())
    else:
        max_abs = max_rel = 0.0

    ok = bool(mask_nan and mask_inf and zero_exact and close)
    n_non_bit = int(non_bit_finite.sum())
    detail = (
        f"shape={a.shape} nan_ok={mask_nan} inf_ok={mask_inf} "
        f"zero_sign_ok={zero_exact} within_tol={close}"
    )
    if detail_extra:
        detail = f"{detail} {detail_extra}"
    if n_non_bit:
        detail += f" non_bit_exact_finite={n_non_bit} max_rel={max_rel:.3g}"
    report.add(
        name,
        ok,
        detail,
        bit_exact=_bit_equal(a, b),
        nan_mask_ok=bool(mask_nan),
        inf_mask_ok=bool(mask_inf),
        zero_sign_ok=bool(zero_exact),
        within_tol=close,
        n_non_bit_finite=n_non_bit,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
    )

    diffs: List[Dict[str, Any]] = []
    if n_non_bit:
        if counterpart is not None:
            c = np.asarray(_host(counterpart), dtype=np.float32)
            c_bits = c.view(np.int32)
            spec_ok = c_bits == bits_a  # the counterpart matched the oracle here
        else:
            spec_ok = np.ones_like(bit)
        if counterpart_ok is not None:
            spec_ok = counterpart_ok
        for index_tuple in np.argwhere(non_bit_finite)[:max_repros]:
            index = tuple(int(v) for v in index_tuple)
            entry = {
                "comparison": name,
                "index": list(index),
                "oracle": float(a[index]),
                "got": float(b[index]),
                "oracle_bits": int(bits_a[index]),
                "got_bits": int(bits_b[index]),
                counterpart_name: float(np.asarray(_host(counterpart))[index])
                if counterpart is not None
                else None,
                counterpart_name + "_matches_oracle": bool(spec_ok[index]),
            }
            entry.update(classify(index, rows, bool(not spec_ok[index]), bool(spec_ok[index])))
            diffs.append(entry)
        report.add_bit_diffs(diffs)
        for entry in diffs:
            row = _row_index(tuple(entry["index"]), rows)
            print(
                f"        bit-diff {entry['comparison']} idx={entry['index']} "
                f"path={entry['path']} who={entry['who']} case={entry['case']} "
                f"(row {row.note if row is not None else ''}) oracle={_fmt(entry['oracle'])} "
                f"dense={_fmt(entry['got'])}"
            )
    elif not _bit_equal(a, b):
        print(
            f"        {name}: all differing elements are non-finite (payload bits only)"
        )
    return ok


def _bits_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.view(np.int32) == b.view(np.int32)


def _bit_equal(a: Any, b: Any) -> bool:
    """Bit-pattern equality (``np.array_equal`` is False for NaN == NaN)."""
    x = np.asarray(_host(a), dtype=np.float32)
    y = np.asarray(_host(b), dtype=np.float32)
    if x.shape != y.shape:
        return False
    return bool(_bits_equal(x, y).all())


# --------------------------------------------------------------------------- #
# family builders
# --------------------------------------------------------------------------- #


def _rows_per_value(
    family: str,
    funcs: Sequence[int],
    values: Sequence[float],
    leaf: str,
) -> List[Row]:
    rows: List[Row] = []
    for func in funcs:
        for value in values:
            if leaf == "var":
                spec = TreeSpec([("u", float(func)), ("var", 0.0)])
                x = (value,)
            else:
                spec = TreeSpec([("u", float(func)), ("const", value)])
                x = (0.0,)
            rows.append(
                Row(
                    f"{family} f={func} v={_fmt(value)}",
                    family,
                    spec,
                    x,
                    note=f"leaf={leaf} func={func} value={_fmt(value)}",
                )
            )
    return rows


def build_per_row_contract() -> Block:
    values = (0.0, -0.0, 1e-45, float("inf"), float("-inf"), float("nan"), 1.0)
    rows = [
        Row(
            f"per_row_contract v={_fmt(value)}",
            "per_row_contract",
            TreeSpec([("var", 0.0)]),
            (value,),
            note=f"identity tree, x={_fmt(value)}",
        )
        for value in values
    ]
    return Block("per_row_contract", rows, 1, 1)


def build_unary_var() -> Block:
    return Block(
        "unary_var", _rows_per_value("unary_var", UNARY_FUNCS, EDGE_VALUES, "var"), 1, 2
    )


def build_unary_const() -> Block:
    return Block(
        "unary_const",
        _rows_per_value("unary_const", UNARY_FUNCS, EDGE_VALUES, "const"),
        1,
        2,
    )


def build_binary_var() -> Block:
    rows: List[Row] = []
    for func in BINARY_FUNCS:
        for left, right in BINARY_PAIRS:
            rows.append(
                Row(
                    f"binary_var f={func} a={_fmt(left)} b={_fmt(right)}",
                    "binary_var",
                    TreeSpec([("b", float(func)), ("var", 0.0), ("var", 1.0)]),
                    (left, right),
                    note=f"func={func} a={_fmt(left)} b={_fmt(right)}",
                )
            )
    return Block("binary_var", rows, 2, 3)


IF_TRIPLES: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 1.0, float("nan")),
    (1.0, float("nan"), 1.0),
    (-1.0, float("nan"), 2.0),
    (-1.0, 2.0, float("nan")),
    (0.0, 1.0, 2.0),
    (-0.0, 1.0, 2.0),
    (1e-45, 1.0, 2.0),
    (-1e-45, 1.0, 2.0),
    (float("inf"), float("inf"), float("-inf")),
    (float("-inf"), float("inf"), float("-inf")),
    (float("nan"), 1.0, 2.0),
    (float("nan"), float("nan"), float("nan")),
    (2.0, 0.0, -0.0),
    (-2.0, 0.0, -0.0),
    (1.0, 1e9, -1e9),
    (-1.0, 1e9, -1e9),
    (0.5, 1e-45, -1e-45),
    (-0.5, 1e-45, -1e-45),
    (1.0, float("inf"), 1.0),
    (-1.0, float("inf"), 1.0),
)


def build_ternary_if() -> Block:
    rows = [
        Row(
            f"ternary_if c={_fmt(c)} t={_fmt(t)} e={_fmt(e)}",
            "ternary_if",
            TreeSpec([("t", 0.0), ("var", 0.0), ("var", 1.0), ("var", 2.0)]),
            (c, t, e),
            note=f"cond={_fmt(c)} then={_fmt(t)} else={_fmt(e)}",
        )
        for c, t, e in IF_TRIPLES
    ]
    # const-leaf IFs (the specialized engine folds the whole tree)
    for c, t, e in IF_TRIPLES[:8]:
        rows.append(
            Row(
                f"ternary_if const c={_fmt(c)} t={_fmt(t)} e={_fmt(e)}",
                "ternary_if",
                TreeSpec([("t", 0.0), ("const", c), ("const", t), ("const", e)]),
                (0.0, 0.0, 0.0),
                note=f"const cond={_fmt(c)} then={_fmt(t)} else={_fmt(e)}",
            )
        )
    return Block("ternary_if", rows, 3, 4)


COMPARE_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (-0.0, 0.0),
    (0.0, -0.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (2.0, 1.0),
    (-1.0, -1.0),
    (float("nan"), 1.0),
    (1.0, float("nan")),
    (float("inf"), float("inf")),
    (float("inf"), float("-inf")),
    (float("-inf"), float("-inf")),
    (1e9, 1e9),
    (1e-45, 0.0),
)


def build_compare() -> Block:
    rows: List[Row] = []
    for func in (int(Func.LT), int(Func.GT), int(Func.LE), int(Func.GE)):
        for left, right in COMPARE_PAIRS:
            rows.append(
                Row(
                    f"compare f={func} a={_fmt(left)} b={_fmt(right)}",
                    "compare",
                    TreeSpec([("b", float(func)), ("var", 0.0), ("var", 1.0)]),
                    (left, right),
                    note=f"func={func} a={_fmt(left)} b={_fmt(right)}",
                )
            )
    return Block("compare", rows, 2, 3)


MAXMIN_PAIRS: Tuple[Tuple[float, float], ...] = (
    (float("nan"), 1.0),
    (1.0, float("nan")),
    (float("nan"), float("nan")),
    (float("nan"), float("inf")),
    (float("inf"), float("nan")),
    (0.0, -0.0),
    (-0.0, 0.0),
    (float("inf"), 1.0),
    (1.0, float("-inf")),
    (-0.0, -0.0),
    (1e9, -1e9),
    (1.0, 1.0),
)


def build_maxmin() -> Block:
    rows: List[Row] = []
    for func in (int(Func.MAX), int(Func.MIN)):
        for left, right in MAXMIN_PAIRS:
            rows.append(
                Row(
                    f"maxmin f={func} a={_fmt(left)} b={_fmt(right)}",
                    "maxmin",
                    TreeSpec([("b", float(func)), ("var", 0.0), ("var", 1.0)]),
                    (left, right),
                    note=f"func={func} a={_fmt(left)} b={_fmt(right)}",
                )
            )
    return Block("maxmin", rows, 2, 3)


ZERO_DENOMINATORS: Tuple[float, ...] = (
    0.0,
    -0.0,
    1e-45,
    -1e-45,
    1e-9,
    -1e-9,
    1e-10,
    -1e-10,
)
ZERO_NUMERATORS: Tuple[float, ...] = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    1e9,
)


def build_zero_denominator() -> Block:
    rows: List[Row] = []
    # DIV / LOOSE_DIV on the full numerator x denominator grid
    for func in (int(Func.DIV), int(Func.LOOSE_DIV)):
        for num in ZERO_NUMERATORS:
            for den in ZERO_DENOMINATORS:
                rows.append(
                    Row(
                        f"zero_denominator f={func} a={_fmt(num)} b={_fmt(den)}",
                        "zero_denominator",
                        TreeSpec([("b", float(func)), ("var", 0.0), ("var", 1.0)]),
                        (num, den),
                        note=f"func={func} a={_fmt(num)} b={_fmt(den)}",
                    )
                )
    # INV / LOOSE_INV / LOG / LOOSE_LOG / SQRT / LOOSE_SQRT at zero-ish inputs
    unary_cases = (
        (int(Func.INV), ZERO_DENOMINATORS),
        (int(Func.LOOSE_INV), ZERO_DENOMINATORS),
        (int(Func.LOG), (0.0, -0.0, -1e-45, 1e-45, -1.0, float("-inf"), float("inf"), float("nan"))),
        (int(Func.LOOSE_LOG), (0.0, -0.0, -1e-45, 1e-45, -1.0, float("-inf"), float("inf"), float("nan"))),
        (int(Func.SQRT), (0.0, -0.0, -1e-45, 1e-45, -1.0, float("-inf"), float("inf"), float("nan"))),
        (int(Func.LOOSE_SQRT), (0.0, -0.0, -1e-45, 1e-45, -1.0, float("-inf"), float("inf"), float("nan"))),
    )
    for func, values in unary_cases:
        for value in values:
            rows.append(
                Row(
                    f"zero_denominator u f={func} a={_fmt(value)}",
                    "zero_denominator",
                    TreeSpec([("u", float(func)), ("var", 0.0)]),
                    (value, 1.0),
                    note=f"unary func={func} a={_fmt(value)}",
                )
            )
    # LOOSE_POW edges
    for left, right in ((0.0, 0.0), (-0.0, 0.0), (0.0, -1.0), (-2.0, 0.5), (0.0, float("nan")), (float("nan"), 0.0), (1.0, 1e9)):
        rows.append(
            Row(
                f"zero_denominator loose_pow a={_fmt(left)} b={_fmt(right)}",
                "zero_denominator",
                TreeSpec([("b", float(Func.LOOSE_POW)), ("var", 0.0), ("var", 1.0)]),
                (left, right),
                note=f"loose_pow a={_fmt(left)} b={_fmt(right)}",
            )
        )
    return Block("zero_denominator", rows, 2, 3)


def _const_tree(name: str, nodes: Sequence[Tuple[str, float]], x_len: int = 1) -> Row:
    return Row(name, "const_tree", TreeSpec(list(nodes)), (0.0,) * x_len)


def build_const_tree() -> Block:
    c = lambda v: ("const", v)  # noqa: E731 - compact const node helper
    u = lambda f: ("u", float(f))  # noqa: E731
    b = lambda f: ("b", float(f))  # noqa: E731
    t = ("t", 0.0)
    rows: List[Row] = [
        _const_tree("const_tree LOOSE_POW(0,0)", [b(Func.LOOSE_POW), c(0.0), c(0.0)]),
        _const_tree("const_tree POW(0,0)", [b(Func.POW), c(0.0), c(0.0)]),
        _const_tree("const_tree POW(-2,0.5)", [b(Func.POW), c(-2.0), c(0.5)]),
        _const_tree("const_tree DIV(1,0)", [b(Func.DIV), c(1.0), c(0.0)]),
        _const_tree("const_tree DIV(0,0)", [b(Func.DIV), c(0.0), c(0.0)]),
        _const_tree("const_tree LOOSE_DIV(1,0)", [b(Func.LOOSE_DIV), c(1.0), c(0.0)]),
        _const_tree("const_tree LOOSE_DIV(1,-0)", [b(Func.LOOSE_DIV), c(1.0), c(-0.0)]),
        _const_tree("const_tree LOOSE_DIV(0,0)", [b(Func.LOOSE_DIV), c(0.0), c(0.0)]),
        _const_tree("const_tree LOG(0)", [u(Func.LOG), c(0.0)]),
        _const_tree("const_tree LOG(-1)", [u(Func.LOG), c(-1.0)]),
        _const_tree("const_tree LOOSE_LOG(0)", [u(Func.LOOSE_LOG), c(0.0)]),
        _const_tree("const_tree LOOSE_LOG(-1)", [u(Func.LOOSE_LOG), c(-1.0)]),
        _const_tree("const_tree SQRT(-1)", [u(Func.SQRT), c(-1.0)]),
        _const_tree("const_tree LOOSE_SQRT(-1)", [u(Func.LOOSE_SQRT), c(-1.0)]),
        _const_tree("const_tree EXP(1000)", [u(Func.EXP), c(1000.0)]),
        _const_tree("const_tree EXP(-1000)", [u(Func.EXP), c(-1000.0)]),
        _const_tree("const_tree INV(0)", [u(Func.INV), c(0.0)]),
        _const_tree("const_tree TAN(1e9)", [u(Func.TAN), c(1e9)]),
        _const_tree("const_tree SIN(1e9)", [u(Func.SIN), c(1e9)]),
        _const_tree("const_tree MAX(nan,1)", [b(Func.MAX), c(float("nan")), c(1.0)]),
        _const_tree("const_tree MIN(1,nan)", [b(Func.MIN), c(1.0), c(float("nan"))]),
        _const_tree("const_tree LT(1,1)", [b(Func.LT), c(1.0), c(1.0)]),
        _const_tree("const_tree GE(nan,1)", [b(Func.GE), c(float("nan")), c(1.0)]),
        _const_tree("const_tree IF(1,2,3)", [t, c(1.0), c(2.0), c(3.0)]),
        _const_tree("const_tree IF(nan,2,3)", [t, c(float("nan")), c(2.0), c(3.0)]),
        _const_tree("const_tree IF(inf,nan,3)", [t, c(float("inf")), c(float("nan")), c(3.0)]),
        # folded operand under an emitted node (mixed fold/emit inside one tree)
        Row(
            "const_tree mixed fold/emit",
            "const_tree",
            TreeSpec([("b", float(Func.MUL)), ("b", float(Func.ADD)), c(1.0), c(2.0), ("var", 0.0)]),
            (1.5,),
            note="const subtree under a MUL with a VAR",
        ),
        Row(
            "const_tree mixed fold/emit div",
            "const_tree",
            TreeSpec([("b", float(Func.DIV)), ("u", float(Func.SIN)), ("var", 0.0), ("b", float(Func.MUL)), c(3.0), c(4.0)]),
            (0.5,),
            note="SIN(var) / (3*4)",
        ),
    ]
    # nested const chains of increasing depth (fold-vs-emit at every depth)
    for depth in (2, 3, 5, 8, 12, 20, 31):
        nodes: List[Tuple[str, float]] = [("const", 0.75)]
        for i in range(depth - 1):
            func = (Func.SIN, Func.COS, Func.EXP, Func.LOOSE_LOG, Func.TANH)[i % 5]
            nodes.insert(0, ("u", float(func)))
        rows.append(_const_tree(f"const_tree chain depth={depth}", nodes))
    return Block("const_tree", rows, 1, 32)


def build_deep_chain() -> Block:
    func_cycle = (Func.SIN, Func.EXP, Func.INV, Func.LOOSE_SQRT, Func.TANH, Func.COS)
    rows: List[Row] = []
    for depth in (16, 20, 24, 28, 30, 31):
        for leaf in ("var", "const"):
            nodes: List[Tuple[str, float]] = [
                (leaf, 0.5 if leaf == "const" else 0.0)
            ]
            for i in range(depth - 1):
                nodes.insert(0, ("u", float(func_cycle[i % len(func_cycle)])))
            rows.append(
                Row(
                    f"deep_chain depth={depth} leaf={leaf}",
                    "deep_chain",
                    TreeSpec(nodes),
                    (0.25,),
                    note=f"depth={depth} leaf={leaf}",
                )
            )
    # one deep chain per single function (long rewrite runs)
    for func in (Func.SIN, Func.COSH, Func.LOOSE_INV, Func.SQRT, Func.POW):
        if func == Func.POW:
            nodes = [("const", 2.0)]
            for _ in range(14):
                nodes.insert(0, ("b", float(Func.POW)))
                nodes.insert(1, ("const", 1.0000001))
            rows.append(
                Row(
                    "deep_chain repeated POW",
                    "deep_chain",
                    TreeSpec(nodes),
                    (0.5,),
                    note="repeated POW(const 1.0000001)",
                )
            )
        else:
            nodes = [("var", 0.0)]
            for _ in range(24):
                nodes.insert(0, ("u", float(func)))
            rows.append(
                Row(
                    f"deep_chain repeated f={int(func)}",
                    "deep_chain",
                    TreeSpec(nodes),
                    (0.75,),
                    note=f"24 x func {int(func)}",
                )
            )
    return Block("deep_chain", rows, 1, 32)


def _junk_zero(p: int, i: int, L: int):
    return (0.0, 0, 0)


def _junk_nan(p: int, i: int, L: int):
    return (float("nan"), int(NType.CONST), 0)


def _junk_random(p: int, i: int, L: int):
    return random_junk(p, i, L)


def _junk_valid_extension(p: int, i: int, L: int):
    """A *valid-looking* node stream past the tree length: a leak would change the value."""
    return (3.0, int(NType.CONST), 1)


def _junk_deep_function(p: int, i: int, L: int):
    """Junk that looks like the head of a deep function chain with junk sizes."""
    return (1e6, int(NType.BFUNC), max(1, L - i))


def build_junk_tail() -> Block:
    bases: Sequence[Tuple[str, Sequence[Tuple[str, float]], Sequence[float]]] = (
        ("var_identity", [("var", 0.0)], (0.5,)),
        ("sin_var", [("u", float(Func.SIN)), ("var", 0.0)], (1.25,)),
        (
            "mix",
            [("b", float(Func.DIV)), ("u", float(Func.EXP)), ("var", 0.0), ("b", float(Func.MUL)), ("const", 3.0), ("var", 0.0)],
            (0.5,),
        ),
        (
            "if_var",
            [("t", 0.0), ("var", 0.0), ("u", float(Func.COS)), ("var", 0.0), ("u", float(Func.SIN)), ("var", 0.0)],
            (0.75,),
        ),
    )
    variants = (
        ("zeros", _junk_zero),
        ("nan", _junk_nan),
        ("random", _junk_random),
        ("valid_ext", _junk_valid_extension),
        ("deep_fn", _junk_deep_function),
    )
    rows: List[Row] = []
    for base_name, nodes, x in bases:
        for variant_name, junk in variants:
            rows.append(
                Row(
                    f"junk_tail {base_name}/{variant_name}",
                    "junk_tail",
                    TreeSpec(list(nodes)),
                    x,
                    junk=junk,
                    note=f"base={base_name} junk={variant_name}",
                )
            )
    return Block(
        "junk_tail",
        rows,
        1,
        24,
        group_key=lambda row: row.note.split(" junk=")[0].replace("base=", ""),
        note="same tree, five hostile padding tails",
    )


def build_unknown_id() -> Block:
    rows: List[Row] = []
    unary_ids = (0, 1, 13, 29, 30, 40, 100, 1_000_000, -1, -3.7, -0.5, 13.9, 28.9)
    for func in unary_ids:
        rows.append(
            Row(
                f"unknown_id unary id={_fmt(float(func))}",
                "unknown_id",
                TreeSpec([("u", float(func)), ("var", 0.0)]),
                (1.5, -0.25),
                note=f"unary node, function id {func}",
            )
        )
    binary_ids = (0, 14, 28, 29, 40, 1_000_000, -1, -0.5, 12.9)
    for func in binary_ids:
        rows.append(
            Row(
                f"unknown_id binary id={_fmt(float(func))}",
                "unknown_id",
                TreeSpec([("b", float(func)), ("var", 0.0), ("var", 1.0)]),
                (1.5, -0.25),
                note=f"binary node, function id {func}",
            )
        )
    ternary_ids = (0, 29, 1_000_000, -1, 5.5)
    for func in ternary_ids:
        rows.append(
            Row(
                f"unknown_id ternary id={_fmt(float(func))}",
                "unknown_id",
                TreeSpec([("t", float(func)), ("const", 1.0), ("const", 2.0), ("const", 3.0)]),
                (0.0, 0.0),
                note=f"ternary node, function id {func} (id ignored)",
            )
        )
    return Block("unknown_id", rows, 2, 4)


#: Unknown base types WITHOUT bit 7 (bit 7 is OUT_NODE and is rejected in a
#: single-output tree -- that pair lives in the ``fallback_error`` family).
UNKNOWN_TYPES: Tuple[int, ...] = (5, 6, 29, 100, 127)


def build_unknown_type() -> Block:
    rows: List[Row] = []
    for node_type in UNKNOWN_TYPES:
        # type with three const children: the ternary branch -> IF semantics
        nodes = [
            (node_type, 0.0),
            (int(NType.CONST), 1.0),
            (int(NType.CONST), 2.0),
            (int(NType.CONST), 3.0),
        ]
        rows.append(
            Row(
                f"unknown_type t={node_type} const",
                "unknown_type",
                raw_spec(nodes),
                (0.0,),
                note=f"node_type={node_type} (const operands)",
            )
        )
        # type with a VAR condition and an unknown-type child
        nodes = [
            (node_type, 0.0),
            (int(NType.VAR), 0.0),
            (node_type, 0.0),
            (int(NType.CONST), -1.0),
            (int(NType.CONST), 7.5),
            (int(NType.CONST), -7.5),
            (int(NType.VAR), 0.0),
        ]
        rows.append(
            Row(
                f"unknown_type t={node_type} nested",
                "unknown_type",
                raw_spec(nodes),
                (1.0,),
                note=f"nested node_type={node_type} with VAR operands",
            )
        )
    return Block("unknown_type", rows, 1, 7)


def build_random_pop(seed: int) -> Block:
    descriptor = GenerateDescriptor(
        max_tree_len=32,
        input_len=3,
        output_len=1,
        using_funcs=list(FUNCS_NAMES),
        max_layer_cnt=3,  # IF in using_funcs -> the worst case is (3**3-1)/2 = 13
        const_prob=0.35,
        const_range=(-2.0, 2.0),
        sample_cnt=8,
        layer_leaf_prob=0.25,
    )
    torch.manual_seed(seed)
    forest = Forest.random_generate(pop_size=24, descriptor=descriptor)
    rows: List[Row] = []
    rng = np.random.default_rng(seed)
    for i in range(forest.pop_size):
        tree = forest[i]
        nodes = _nodes_from_view(tree)
        rows.append(
            Row(
                f"random_pop seed={seed} row={i}",
                "random_pop",
                TreeSpec(nodes),
                tuple(rng.choice([0.0, -0.0, 1.0, -1.0, 0.5, 1e9, float("inf")], size=3)),
                note=f"seed={seed} row={i}",
            )
        )
    return Block("random_pop", rows, 3, 32)


def _nodes_from_view(tree: Any) -> List[Tuple[str, float]]:
    """Rebuild a prefix node list from a ``Tree`` view (for the oracle path)."""
    length = int(tree.subtree_size[0])
    return list(zip(_decode_kinds_from_view(tree, length), _values_from_view(tree, length)))


def _decode_kinds_from_view(tree: Any, length: int) -> List[str]:
    out: List[str] = []
    for i in range(length):
        base = int(tree.node_type[i]) & int(NType.TYPE_MASK)
        out.append({0: "var", 1: "const", 2: "u", 3: "b"}.get(base, "t"))
    return out


def _values_from_view(tree: Any, length: int) -> List[float]:
    return [float(tree.node_value[i]) for i in range(length)]


def build_sr_labels() -> Block:
    descriptor = GenerateDescriptor(
        max_tree_len=16,
        input_len=3,
        output_len=1,
        using_funcs=list(FUNCS_NAMES),
        max_layer_cnt=3,
        const_prob=0.5,
        const_range=(-1.0, 1.0),
        sample_cnt=8,
        layer_leaf_prob=0.3,
    )
    torch.manual_seed(11)
    forest = Forest.random_generate(pop_size=8, descriptor=descriptor)
    rows = [
        Row(
            f"sr_labels row={i}",
            "sr_labels",
            TreeSpec(_nodes_from_view(forest[i])),
            (0.5, -0.5, 1.0),
            note=f"row={i}",
        )
        for i in range(forest.pop_size)
    ]
    return Block("sr_labels", rows, 3, 16)


def build_tree_api() -> Block:
    """Single-tree specs reused for the ``Tree`` entry-point checks."""
    c = lambda v: ("const", v)  # noqa: E731
    rows = [
        Row("tree_api sin", "tree_api", TreeSpec([("u", float(Func.SIN)), ("var", 0.0)]), (0.75,)),
        Row("tree_api const", "tree_api", TreeSpec([("const", 2.5)]), (0.0,)),
        Row(
            "tree_api div0",
            "tree_api",
            TreeSpec([("b", float(Func.DIV)), ("var", 0.0), c(0.0)]),
            (1.0,),
        ),
        Row(
            "tree_api deep",
            "tree_api",
            TreeSpec(
                [("u", float(Func.SIN))] * 24
                + [("u", float(Func.EXP))]
                + [("var", 0.0)]
            ),
            (0.5,),
        ),
    ]
    return Block("tree_api", rows, 1, 32)


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #


def _tensors(values: np.ndarray, types: np.ndarray, sizes: np.ndarray):
    vt = torch.from_numpy(np.ascontiguousarray(values)).to(_DEVICE)
    tt = torch.from_numpy(np.ascontiguousarray(types)).to(_DEVICE)
    st = torch.from_numpy(np.ascontiguousarray(sizes)).to(_DEVICE)
    return vt, tt, st


def _sample(rows: Sequence[Row], limit: Optional[int]) -> List[Row]:
    """Stride-sample ``rows`` down to ``limit`` (deterministic spread)."""
    rows = list(rows)
    if limit is None or len(rows) <= limit:
        return rows
    stride = max(1, len(rows) // limit)
    return rows[::stride][:limit]


def run_block(
    block: Block,
    args: argparse.Namespace,
    report: Report,
    results: Dict[str, Any],
    *,
    per_row_limit: Optional[int] = None,
) -> bool:
    rows = _sample(block.rows, per_row_limit)
    values, types, sizes = build_tensors(rows, block.row_len)
    vt, tt, st = _tensors(values, types, sizes)
    forest = Forest(block.input_len, 1, vt, tt, st)
    x = torch.tensor([list(row.x) for row in rows], dtype=torch.float32, device=_DEVICE)
    xb = torch.tensor(list(block.batch_x), dtype=torch.float32, device=_DEVICE)
    labels = torch.tensor(list(block.labels), dtype=torch.float32, device=_DEVICE).reshape(-1, 1)

    label = f"{block.name} (P={len(rows)}, L={block.row_len}, input_len={block.input_len})"
    print(f"\n-- {label} {block.note}")

    # (0) the population must be dense-supported: the public dense call is only
    #     meaningful if prepare_dense accepts it (otherwise the call falls back
    #     silently and the comparison would prove nothing about the dense path).
    try:
        D.prepare_dense(vt, tt, st, block.input_len, 1)
        dense_supported = True
        reason = ""
    except D.DenseUnsupported as exc:
        dense_supported = False
        reason = exc.reason
    report.add(
        f"{label}: dense-supported",
        dense_supported,
        reason or "prepare_dense accepted the population",
    )
    if not dense_supported:
        return False

    ref_fwd = reference.forward_from_rows(values, types, sizes, x.cpu(), out_len=1)
    with eval_mode("dense"):
        dense_fwd = forest.forward(x)
    with eval_mode("specialized"):
        spec_fwd = forest.forward(x)
    dense_fwd_h = np.asarray(_host(dense_fwd), dtype=np.float32)
    spec_fwd_h = np.asarray(_host(spec_fwd), dtype=np.float32)
    ref_fwd_h = np.asarray(_host(ref_fwd), dtype=np.float32)
    spec_matches_oracle = _bits_equal(spec_fwd_h, ref_fwd_h)

    ok = compare(
        report,
        f"{label}: forward dense vs oracle",
        ref_fwd,
        dense_fwd,
        rows=rows,
        counterpart=spec_fwd,
        counterpart_ok=spec_matches_oracle,
    )
    compare(
        report,
        f"{label}: forward specialized vs oracle",
        ref_fwd,
        spec_fwd,
        rows=rows,
    )
    # the forced-dense public call must be the direct dense result (proves the
    # wiring actually took the dense branch instead of falling back)
    direct = D.evaluate_dense(vt, tt, st, block.input_len, 1, x, per_row=True)
    report.add(
        f"{label}: public dense == direct dense_engine",
        _bit_equal(dense_fwd_h, direct),
    )

    if block.group_key is not None:
        # Same tree, different padding: the tail must not leak, so every variant
        # of a group has to produce the identical bit pattern.
        groups: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(block.group_key(row), []).append(index)
        for key, members in groups.items():
            reference_row = dense_fwd_h[members[0]]
            same = all(
                _bit_equal(dense_fwd_h[index], reference_row) for index in members[1:]
            )
            report.add(
                f"{label}: padding invariance [{key}]",
                same,
                f"{len(members)} variants bit-identical",
            )
    if not args.skip_batch:
        ref_batch = reference.batch_forward_from_rows(values, types, sizes, xb.cpu(), out_len=1)
        with eval_mode("dense"):
            dense_batch = forest.batch_forward(xb)
        with eval_mode("specialized"):
            spec_batch = forest.batch_forward(xb)
        spec_batch_matches = _bits_equal(
            np.asarray(_host(spec_batch), dtype=np.float32),
            np.asarray(_host(ref_batch), dtype=np.float32),
        )
        ok &= compare(
            report,
            f"{label}: batch_forward dense vs oracle",
            ref_batch,
            dense_batch,
            rows=rows,
            counterpart=spec_batch,
            counterpart_ok=spec_batch_matches,
        )
        compare(
            report,
            f"{label}: batch_forward specialized vs oracle",
            ref_batch,
            spec_batch,
            rows=rows,
        )
        if not args.skip_sr:
            for use_mse in (True, False):
                metric = "MSE" if use_mse else "MAE"
                ref_sr = reference.sr_fitness_from_rows(
                    values, types, sizes, xb.cpu(), labels.cpu(), use_mse=use_mse, out_len=1
                )
                with eval_mode("dense"):
                    dense_sr = forest.SR_fitness(xb, labels, use_MSE=use_mse)
                with eval_mode("specialized"):
                    spec_sr = forest.SR_fitness(xb, labels, use_MSE=use_mse)
                spec_sr_matches = _bits_equal(
                    np.asarray(_host(spec_sr), dtype=np.float32),
                    np.asarray(_host(ref_sr), dtype=np.float32),
                )
                ok &= compare(
                    report,
                    f"{label}: SR_fitness({metric}) dense vs oracle",
                    ref_sr,
                    dense_sr,
                    rows=rows,
                    counterpart=spec_sr,
                    counterpart_ok=spec_sr_matches,
                )
                compare(
                    report,
                    f"{label}: SR_fitness({metric}) specialized vs oracle",
                    ref_sr,
                    spec_sr,
                    rows=rows,
                )

    results[block.name] = {
        "rows": len(rows),
        "row_len": block.row_len,
        "input_len": block.input_len,
        "bit_diffs": sum(1 for entry in report.bit_diffs if block.name in entry["comparison"]),
    }
    return ok


def case_long_tree(args, report: Report, results: Dict[str, Any]) -> bool:
    """Trees at/over the dense trace cap must fall back and still be exact."""
    cap = args.max_tree_len
    rows: List[Row] = []
    for length in (cap, cap + 1, cap + 8):
        nodes: List[Tuple[str, float]] = [("var", 0.0)]
        for i in range(length - 1):
            func = (Func.SIN, Func.EXP, Func.TANH, Func.LOOSE_INV)[i % 4]
            nodes.insert(0, ("u", float(func)))
        rows.append(
            Row(
                f"long_tree len={length}",
                "long_tree",
                TreeSpec(nodes),
                (0.5,),
                note=f"tree length {length} (cap {cap})",
            )
        )
    values, types, sizes = build_tensors(rows, cap + 8)
    vt, tt, st = _tensors(values, types, sizes)
    forest = Forest(1, 1, vt, tt, st)
    x = torch.tensor([[row.x[0]] for row in rows], dtype=torch.float32, device=_DEVICE)

    label = f"long_tree (P={len(rows)}, L={cap + 8})"
    print(f"\n-- {label} trees over the dense trace cap")
    try:
        D.prepare_dense(vt, tt, st, 1, 1, max_tree_len=cap)
        report.add(f"{label}: dense refuses (DenseUnsupported)", False, "prepare_dense accepted")
    except D.DenseUnsupported as exc:
        report.add(
            f"{label}: dense refuses (DenseUnsupported)", True, exc.reason
        )
    ref = reference.forward_from_rows(values, types, sizes, x.cpu(), out_len=1)
    with eval_mode("dense"):
        got = forest.forward(x)
    report.add(
        f"{label}: long trees equal tree length",
        _bit_equal(got, ref),
        "forced-dense public call fell back and matched the oracle bit for bit",
    )
    results["long_tree"] = {"rows": len(rows), "row_len": cap + 8}
    return True


def case_error_parity(args, report: Report, results: Dict[str, Any]) -> bool:
    """Invalid populations: same ValueError from both forced public calls."""
    cases: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, str]] = []
    values = np.zeros((1, 4), dtype=np.float32)
    types = np.zeros((1, 4), dtype=np.int16)
    sizes = np.zeros((1, 4), dtype=np.int16)

    def make(tag, v, t, s, input_len=1, note=""):
        cases.append(
            (
                tag,
                np.asarray([v], dtype=np.float32),
                np.asarray([t], dtype=np.int16),
                np.asarray([s], dtype=np.int16),
                input_len,
                note,
            )
        )

    # OUT_NODE tag in a single-output tree
    make(
        "out_node_tag",
        [0.0] * 4,
        [int(NType.CONST) | int(NType.OUT_NODE), 0, 0, 0],
        [1, 0, 0, 0],
        note="type carries OUT_NODE with output_len == 1",
    )
    # negative node type (sign bit == OUT_NODE bit)
    make("negative_type", [0.0] * 4, [-1, 0, 0, 0], [1, 0, 0, 0], note="node_type=-1")
    # unknown base type WITH bit 7 set (a valid unknown type otherwise)
    make(
        "type_bit7",
        [0.0] * 4,
        [32767, 0, 0, 0],
        [1, 0, 0, 0],
        note="node_type=32767 carries bit 7 (OUT_NODE)",
    )
    # inconsistent subtree sizes
    make(
        "bad_subtree_size",
        [0.0] * 4,
        [int(NType.UFUNC), int(NType.VAR), 0, 0],
        [3, 1, 0, 0],
        note="SIN var with subtree_size[0]=3",
    )
    # variable index out of range
    make(
        "var_index_oor",
        [5.0, 0.0, 0.0, 0.0],
        [int(NType.VAR), 0, 0, 0],
        [1, 0, 0, 0],
        note="VAR index 5 with input_len=1",
    )
    # non-finite variable index
    make(
        "var_index_nan",
        [float("nan"), 0.0, 0.0, 0.0],
        [int(NType.VAR), 0, 0, 0],
        [1, 0, 0, 0],
        note="VAR with node_value=NaN",
    )
    # non-finite function id
    make(
        "func_id_nan",
        [float("nan"), 0.0, 0.0, 0.0],
        [int(NType.UFUNC), int(NType.VAR), 0, 0],
        [2, 1, 0, 0],
        note="SIN node with node_value=NaN (function id)",
    )
    # tree longer than the row
    make(
        "tree_longer_than_row",
        [0.0] * 4,
        [int(NType.VAR), 0, 0, 0],
        [9, 0, 0, 0],
        note="subtree_size[0]=9 in a 4-wide row",
    )
    # non-positive tree length
    make("zero_tree_len", [0.0] * 4, [0, 0, 0, 0], [0, 0, 0, 0], note="subtree_size[0]=0")

    print(f"\n-- fallback_error ({len(cases)} invalid populations)")
    ok = True
    for tag, vals, typs, szs, input_len, note in cases:
        vt, tt, st = _tensors(vals, typs, szs)
        forest = Forest(input_len, 1, vt, tt, st)

        def run(mode: str):
            with eval_mode(mode):
                forest.forward(torch.zeros((1, input_len), dtype=torch.float32, device=_DEVICE))

        messages: Dict[str, str] = {}
        for mode in ("dense", "specialized"):
            try:
                run(mode)
                messages[mode] = "<no exception>"
            except ValueError as exc:
                messages[mode] = str(exc)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                messages[mode] = f"{type(exc).__name__}: {exc}"
        try:
            D.evaluate_dense(
                vt,
                tt,
                st,
                input_len,
                1,
                torch.zeros((1, input_len), dtype=torch.float32, device=_DEVICE),
                per_row=True,
            )
            refused, reason = False, "evaluate_dense accepted the population"
        except D.DenseUnsupported as exc:
            refused, reason = True, exc.reason
        parity = messages["dense"] == messages["specialized"]
        raised = messages["specialized"].startswith("<no exception>") is False
        ok &= report.add(
            f"fallback_error {tag}: dense refuses",
            refused,
            reason if refused else note,
        )
        ok &= report.add(
            f"fallback_error {tag}: error parity with the specialized path",
            parity and raised and len(messages["specialized"]) > 0,
            f"dense={messages['dense'][:120]!r} specialized={messages['specialized'][:120]!r}",
        )
    results["fallback_error"] = {"cases": len(cases)}
    return bool(ok)


def case_tree_api(args, block: Block, report: Report, results: Dict[str, Any]) -> bool:
    """The ``Tree`` entry points (1-D and 2-D ``x``, MSE/MAE fitness)."""
    ok = True
    xb = torch.tensor(list(block.batch_x), dtype=torch.float32, device=_DEVICE)
    labels = torch.tensor(list(block.labels), dtype=torch.float32, device=_DEVICE).reshape(-1, 1)
    for row in block.rows:
        values, types, sizes = build_tensors([row], block.row_len)
        vt, tt, st = _tensors(values, types, sizes)
        forest = Forest(block.input_len, 1, vt, tt, st)
        tree = forest[0]
        x1 = torch.tensor(list(row.x), dtype=torch.float32, device=_DEVICE)
        ref1 = reference.forward_from_rows(values, types, sizes, x1.reshape(1, -1).cpu(), out_len=1)[0]
        ref2 = reference.batch_forward_from_rows(values, types, sizes, xb.cpu(), out_len=1)[0]
        with eval_mode("dense"):
            got1 = tree.forward(x1)
            got2 = tree.forward(xb)
        label = f"tree_api {row.name} (L={block.row_len})"
        ok &= compare(report, f"{label}: forward(1-D) dense vs oracle", ref1, got1, rows=[row])
        ok &= compare(report, f"{label}: forward(2-D) dense vs oracle", ref2, got2, rows=[row])
        if not args.skip_sr:
            for use_mse in (True, False):
                metric = "MSE" if use_mse else "MAE"
                ref_sr = reference.sr_fitness_from_rows(
                    values, types, sizes, xb.cpu(), labels.cpu(), use_mse=use_mse, out_len=1
                )
                with eval_mode("dense"):
                    got_sr = tree.SR_fitness(xb, labels, use_MSE=use_mse)
                ok &= compare(
                    report,
                    f"{label}: SR_fitness({metric}) dense vs oracle",
                    ref_sr,
                    got_sr,
                    rows=[row],
                )
    results["tree_api"] = {"trees": len(block.rows), "row_len": block.row_len}
    return ok


def case_wiring(args, report: Report, results: Dict[str, Any]) -> bool:
    """One dense compile per population vs one per tree for the specialized path."""
    if D.resolve_backend_name() != "numpy":
        report.add("wiring: dense builds == 1 (numpy only)", True, "skipped on iree")
        return True
    descriptor = GenerateDescriptor(
        max_tree_len=8,
        input_len=2,
        output_len=1,
        using_funcs=list(FUNCS_NAMES),
        max_layer_cnt=2,
        const_prob=0.3,
        const_range=(-1.0, 1.0),
        sample_cnt=4,
        layer_leaf_prob=0.3,
    )
    torch.manual_seed(5)
    forest = Forest.random_generate(pop_size=5, descriptor=descriptor)
    x = torch.rand(5, 2, dtype=torch.float32, device=_DEVICE)

    E.clear_cache()
    before = E.cache_stats()
    with eval_mode("dense"):
        forest.forward(x)
    dense_stats = E.cache_stats()
    dense_builds = dense_stats["misses"] - before["misses"]

    E.clear_cache()
    before = E.cache_stats()
    with eval_mode("specialized"):
        forest.forward(x)
    spec_stats = E.cache_stats()
    spec_builds = spec_stats["misses"] - before["misses"]

    print(f"\n-- wiring: dense builds={dense_builds} specialized builds={spec_builds} (P=5)")
    report.add(
        "wiring: ONE dense compile for the whole population",
        dense_builds == 1,
        f"dense builds={dense_builds} (expected 1)",
    )
    report.add(
        "wiring: specialized compiles per distinct structure",
        spec_builds >= 2,
        f"specialized builds={spec_builds} (expected ~P for distinct structures)",
    )
    results["wiring"] = {"dense_builds": dense_builds, "specialized_builds": spec_builds}
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("numpy", "iree"), default="numpy")
    parser.add_argument("--gpu", type=int, default=0, help="IREE device index (visible)")
    parser.add_argument("--only", action="append", default=[], metavar="FAMILY")
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--skip-batch", action="store_true", default=False)
    parser.add_argument("--skip-sr", action="store_true", default=False)
    parser.add_argument("--strict-bits", action="store_true", default=False)
    parser.add_argument("--max-tree-len", type=int, default=32)
    parser.add_argument(
        "--iree-max-pop",
        type=int,
        default=4,
        help="rows kept per family on iree (stride-sampled)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    return parser


def _wanted(args, name: str) -> bool:
    if not args.only:
        return True
    return any(token in name for token in args.only)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print("\n".join(_FAMILIES))
        return 0
    if args.only and not any(_wanted(args, name) for name in _FAMILIES):
        print(f"[warn] --only {args.only} matched no family: {list(_FAMILIES)}")
        return 2
    if not _ORACLE_PATH.is_file():
        print(f"[error] the oracle {_ORACLE_PATH} is missing")
        return 2
    if os.environ.get("EVOGP_EVAL_MODE"):
        print(
            f"[warn] EVOGP_EVAL_MODE={os.environ['EVOGP_EVAL_MODE']!r} is set; "
            "this script forces the mode per call and restores it afterwards"
        )
    if os.environ.get("ETL_DISABLE_LOCATIONS", "").strip() not in ("1", "true"):
        print("[warn] ETL_DISABLE_LOCATIONS=1 is not set: tracing will be ~4x slower")
    if not torch.cuda.is_available():
        print("[error] the public API relocates to 'cuda'; no GPU is visible")
        return 2
    if args.backend == "iree":
        # resolve the backend through the engine so the environment matches
        os.environ["EVOGP_ETL_BACKEND"] = "iree"
        os.environ["EVOGP_ETL_IREE_DEVICE"] = str(args.gpu)
    else:
        os.environ.pop("EVOGP_ETL_BACKEND", None)

    args.max_tree_len = int(D.resolve_auto_min_pop(args.max_tree_len)) if False else args.max_tree_len
    report = Report(strict_bits=args.strict_bits)
    results: Dict[str, Any] = {}
    backend = D.resolve_backend_name()
    per_row_limit = args.iree_max_pop if backend != "numpy" else None
    print(
        f"== dense adversarial check: backend={backend} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cap={args.max_tree_len} per_family_limit={per_row_limit} strict_bits={args.strict_bits} =="
    )

    builders: Dict[str, Callable[[], Block]] = {
        "per_row_contract": build_per_row_contract,
        "unary_var": build_unary_var,
        "unary_const": build_unary_const,
        "binary_var": build_binary_var,
        "ternary_if": build_ternary_if,
        "compare": build_compare,
        "maxmin": build_maxmin,
        "zero_denominator": build_zero_denominator,
        "const_tree": build_const_tree,
        "deep_chain": build_deep_chain,
        "junk_tail": build_junk_tail,
        "unknown_id": build_unknown_id,
        "unknown_type": build_unknown_type,
        "random_pop": lambda: build_random_pop(args.seed),
        "sr_labels": build_sr_labels,
    }

    def guarded(name: str, fn: Callable[[], bool]) -> bool:
        if not _wanted(args, name):
            return True
        try:
            return bool(fn())
        except Exception as exc:  # noqa: BLE001 - recorded as a hard failure
            report.add(f"{name}: unexpected exception", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return False

    def run_family(name: str) -> bool:
        block = builders[name]()
        if name == "tree_api":
            return case_tree_api(args, block, report, results)
        return run_block(block, args, report, results, per_row_limit=per_row_limit)

    for name in _FAMILIES:
        if name in ("long_tree", "fallback_error", "tree_api", "wiring"):
            continue
        guarded(name, lambda name=name: run_family(name))
    if _wanted(args, "tree_api"):
        guarded("tree_api", lambda: case_tree_api(args, build_tree_api(), report, results))
    guarded("long_tree", lambda: case_long_tree(args, report, results))
    guarded("fallback_error", lambda: case_error_parity(args, report, results))
    guarded("wiring", lambda: case_wiring(args, report, results))

    report_dict: Dict[str, Any] = {
        "backend": backend,
        "gpu": args.gpu,
        "cap": args.max_tree_len,
        "strict_bits": args.strict_bits,
        "only": list(args.only),
        "results": results,
        "entries": report.entries,
        "bit_diffs": report.bit_diffs,
        "ok": report.ok,
    }
    path = args.json or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"dense_adversarial_{backend}.json",
    )
    try:
        with open(path, "w") as handle:
            json.dump(report_dict, handle, indent=2, default=str)
        print(f"\n[json] {path}")
    except Exception as exc:  # pragma: no cover - scratch output only
        print(f"[warn] could not write {path}: {exc}")

    passed = sum(1 for entry in report.entries if entry["ok"])
    by_class: Dict[str, int] = {}
    for entry in report.bit_diffs:
        key = f"{entry['path']}/{entry['who']}"
        by_class[key] = by_class.get(key, 0) + 1
    print(
        f"\n== bit-exactness: {len(report.bit_diffs)} non-bit-exact finite value(s) "
        f"reported{(' ' + str(by_class)) if by_class else ''} =="
    )
    print(
        f"== {'OK' if report.ok else 'FAILED'}: {passed}/{len(report.entries)} checks passed "
        f"(backend={backend}) =="
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())