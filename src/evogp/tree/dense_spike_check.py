"""Parity + measurement harness for the dense evaluator prototype (SPIKE).

Runnable script, deliberately NOT a pytest module (no ``test_`` prefix, no
pytest import): it re-derives the numbers quoted in the spike report and is meant
to be run by hand.

    # numpy (CPU) parity battery + measurements
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 \
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \
        src/evogp/tree/dense_spike_check.py --backend numpy --quick

    # iree-cuda (GPU 3) parity + compile/run numbers (keep it short)
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=3 \
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \
        src/evogp/tree/dense_spike_check.py --backend iree --quick

Flags: ``--backend numpy|iree``, ``--mode grow|concat|scatter``,
``--quick``/``--full``/``--tiny``, ``--gpu N`` (visible index), ``--max-tree-len L``,
``--seed S``, ``--skip-measure``, ``--skip-parity``, ``--json PATH``,
``--only NAME`` (repeatable; run only the cases whose name contains ``NAME``,
names in ``CASE_NAMES`` -- keeps an iree session inside its GPU/leak budget),
``--repeats K`` (``--tiny`` shrinks the measurement grid to P in {4,16} and
L in {8,16} so an iree run stays inside a short GPU budget).  Exit code 0 =
every parity case passed, 1 = at least one mismatch, 2 = a hard error while
building/running a case.

An iree session must stay under the etl/iree executable-leak budget (~25
compiled executables per process segfault the interpreter, see
``dense_engine``'s iree-robustness note): the heavy per_row cases
(``function_sweep`` = 1200 trees, ``random_populations`` = 84 trees, ``shapes``
= 24 trees through the chunked dense path) cannot run whole on iree -- split
them with ``--only`` / keep the affected cases in their own short processes.

What it does
------------
1. **Parity battery** -- dense vs the specialized engine (same tensors, same
   backend), on ``numpy`` first and then on the opt-in ``iree`` backend:
   * random populations from the public API (``Forest.random_generate``), several
     seeds/descriptors, ``per_row`` True/False, ``sr_fitness`` MSE + MAE;
   * ``case_engine_per_row_iree_bug`` -- the engine's own ``per_row=True``
     contract (every tree sees its own ``x`` row, on every backend) plus the
     host->device upload counts that keep ``per_row=False`` at one upload;
   * an adversarial sweep: every function id against edge operand values
     (``+-0.0``, ``+-1e-9``, tiny/large, ``+-inf``, NaN), ``DIV``/``INV`` by
     0, ``POW``/``LOOSE_POW`` edges, ``LOG`` of <= 0, ``SQRT`` of negatives,
     ``sinh``/``cosh``/``tanh`` at tiny/large inputs, ``IF`` with NaN/inf in the
     UNSELECTED branch, comparisons at boundaries, unknown function ids and
     unknown base types, deep chains;
   * padding/junk-tail invariance (same tree, different junk tails -> identical
     bits) and fully constant trees (the specialized engine folds them
     host-side; the dense graph emits them);
   * const-heavy random populations -- the fold-vs-emit search: every
     non-bit-exact FINITE value is reported with a minimal repro (decoded token
     stream + the input row that produced it).
   Comparison rule: NaN/``+-inf`` masks must match exactly, finite values use
   rtol/atol 1e-4, and non-bit-exact finite values are reported explicitly
   (bit-exactness is the goal, not the requirement).
   Also checks the public shape/dtype/contiguity/requires_grad/writability
   contracts and that a second call with a different population of the same
   shape does NOT rebuild (``etl_engine.cache_stats()`` deltas).
2. **Measurements** -- dense vs specialized, as a function of ``N`` and ``P``
   (numpy: decode ms, cold build ms, warm run ms, speedup) plus the ``L_trace``
   scaling curve (trace/lower/compile/load ms, graph op count, warm run ms) and
   the IREE compile/run numbers.  All timings are medians of ``--repeats``
   warm invocations after one warm-up; the dense cold build uses a private
   ``etl.CompileCache`` so the shared cache is not polluted.
3. **IREE experiments** -- with ``--backend iree``: (a) repeated-invoke stability
   of the ``scatter`` write mode on an in-graph-constructed buffer (~100
   invokes, needs ``EVOGP_DENSE_ALLOW_UNSAFE_SCATTER=1``) and (b) whether the
   scatter mode mutates its runtime INPUT tensors.

Everything is printed as a human-readable log plus one JSON blob (``--json``,
default under ``$TMPDIR``/``/tmp``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import etl  # noqa: E402  (after the sys.path bootstrap)

from evogp.tree import Forest, GenerateDescriptor  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402
from evogp.tree.utils import FUNCS_NAMES, Func, NType  # noqa: E402

# ---------------------------------------------------------------------------
# tiny utilities
# ---------------------------------------------------------------------------

_RTOL = 1e-4
_ATOL = 1e-4


def _fmt(value: float) -> str:
    if isinstance(value, float) and value != value:
        return "nan"
    return f"{value:.6g}"


def _median_ms(fn: Callable[[], Any], repeats: int) -> Dict[str, float]:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "min_ms": min(times),
        "median_ms": median(times),
        "samples": len(times),
    }


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - platform dependent
        return float("nan")


def _as_host(tensor: Any) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
# hand-built populations
# ---------------------------------------------------------------------------

_ARITY = {
    "var": 0,
    "const": 0,
    "u": 1,
    "b": 2,
    "t": 3,
}
_TYPE = {
    "var": int(NType.VAR),
    "const": int(NType.CONST),
    "u": int(NType.UFUNC),
    "b": int(NType.BFUNC),
    "t": int(NType.TFUNC),
}


class TreeSpec:
    """Prefix-order node list with automatic subtree sizes (validation helper)."""

    def __init__(self, nodes: Sequence[Tuple[str, float]]):
        self.nodes = list(nodes)

    def tensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = self.nodes
        n = len(nodes)
        sizes = np.zeros(n, dtype=np.int32)
        stack: List[int] = []
        for i in range(n - 1, -1, -1):
            kind, _ = nodes[i]
            arity = _ARITY[kind]
            if len(stack) < arity:
                raise ValueError(f"bad spec at node {i}: needs {arity} operands")
            size = 1 + sum(stack.pop() for _ in range(arity))
            sizes[i] = size
            stack.append(size)
        if len(stack) != 1:
            raise ValueError("bad spec: the prefix stream does not reduce to one tree")
        values = np.asarray([float(v) for _, v in nodes], dtype=np.float32)
        types = np.asarray([_TYPE[k] for k, _ in nodes], dtype=np.int16)
        return values, types, sizes


def pack_trees(
    specs: Sequence[TreeSpec],
    row_len: Optional[int] = None,
    junk: Optional[Callable[[int, int, int], Tuple[float, int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(P, L)`` population tensors from tree specs (padded with junk)."""
    built = [spec.tensors() for spec in specs]
    longest = max(int(sizes[0]) for _, _, sizes in built)
    L = int(row_len) if row_len is not None else longest
    P = len(built)
    values = np.zeros((P, L), dtype=np.float32)
    types = np.zeros((P, L), dtype=np.int16)
    sizes = np.zeros((P, L), dtype=np.int16)
    rng = np.random.default_rng(1234)
    for p, (v, t, s) in enumerate(built):
        n = int(s[0])
        if n > L:
            raise ValueError(f"tree {p} has {n} nodes but the row length is {L}")
        values[p, :n] = v[:n]
        types[p, :n] = t[:n]
        sizes[p, :n] = s[:n]
        for i in range(n, L):
            if junk is None:
                values[p, i] = 0.0
                types[p, i] = 0
                sizes[p, i] = 0
            else:
                values[p, i], types[p, i], sizes[p, i] = junk(p, i, L)
    return values, types, sizes


def random_junk(p: int, i: int, L: int) -> Tuple[float, int, int]:
    """Deliberately hostile padding: type/value/size that mean nothing."""
    rng = np.random.default_rng(1000 * p + i)
    kind = int(rng.integers(0, 5))
    value = float(rng.choice([0.0, -0.0, 3.0, -2.5, 1e9, 17.0, 0.5, 1e-8]))
    size = int(rng.integers(-2, 6))
    if kind in (2, 3, 4):
        value = float(rng.integers(-3, 30))
    return value, kind, size


def constant_chain(length: int, value: float = 1.0) -> TreeSpec:
    """A unary chain of ``length`` nodes (deep reads; constant operand)."""
    nodes: List[Tuple[str, float]] = [("const", value)]
    for i in range(length - 1):
        nodes.insert(0, ("u", float(Func.SIN if i % 2 else Func.COS)))
    return TreeSpec(nodes)


def constant_tree(length: int) -> TreeSpec:
    """A fully constant tree of ``length`` nodes (engine folds it host-side)."""
    nodes: List[Tuple[str, float]] = [("const", 1.5)]
    while len(nodes) < length - 2:
        nodes.insert(0, ("b", float(Func.ADD if len(nodes) % 2 else Func.MUL)))
        nodes.insert(1, ("const", float(len(nodes))))
    while len(nodes) < length - 1:
        nodes.insert(0, ("u", float(Func.SIN)))
    # close the remaining arity with variables so the tree stays valid but is
    # still mostly constant
    return TreeSpec(nodes)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def compare(name: str, ref: Any, got: Any, report: Dict[str, Any], max_repros: int = 5) -> bool:
    """Compare two float32 results (masks exact, finite rtol/atol, bit-exact count)."""
    a = np.asarray(_as_host(ref), dtype=np.float32)
    b = np.asarray(_as_host(got), dtype=np.float32)
    entry: Dict[str, Any] = {"case": name, "shape": list(a.shape)}
    if a.shape != b.shape:
        entry.update(ok=False, error=f"shape {b.shape} != {a.shape}")
        report["cases"].append(entry)
        print(f"[FAIL] {name}: shape {b.shape} != {a.shape}")
        return False
    nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
    inf_ok = np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(
        np.isneginf(a), np.isneginf(b)
    )
    finite = np.isfinite(a) & np.isfinite(b)
    bits_a = a.view(np.int32)
    bits_b = b.view(np.int32)
    bit = bits_a == bits_b
    non_bit_finite = finite & ~bit
    if finite.any():
        diff = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        denom = np.abs(a[finite].astype(np.float64))
        rel = np.where(denom > 0, diff / np.maximum(denom, 1e-45), diff)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float(rel.max()) if rel.size else 0.0
    else:
        max_abs = max_rel = 0.0
    close = bool(np.allclose(a[finite], b[finite], rtol=_RTOL, atol=_ATOL)) if finite.any() else True
    ok = bool(nan_ok and inf_ok and close)
    entry.update(
        ok=ok,
        bit_exact=bool(np.array_equal(a, b)),
        nan_mask_ok=bool(nan_ok),
        inf_mask_ok=bool(inf_ok),
        n_non_bit_finite=int(non_bit_finite.sum()),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        within_tol=close,
    )
    if non_bit_finite.any():
        offenders = np.argwhere(non_bit_finite)[:max_repros]
        entry["repros"] = [
            {
                "index": [int(x) for x in idx],
                "engine": float(a[tuple(idx)]),
                "dense": float(b[tuple(idx)]),
                "engine_bits": int(bits_a[tuple(idx)]),
                "dense_bits": int(bits_b[tuple(idx)]),
            }
            for idx in offenders
        ]
    report["cases"].append(entry)
    status = "ok " if ok else "FAIL"
    extra = ""
    if entry["n_non_bit_finite"]:
        extra = f" non_bit_exact_finite={entry['n_non_bit_finite']} max_rel={max_rel:.3g}"
    elif not entry["bit_exact"]:
        # the only differing bits are inside NaN payloads (masks already match)
        extra = " (bit_exact=False: non-finite payload bits only)"
    entry["finite_bit_exact"] = bool(non_bit_finite.sum() == 0)
    print(
        f"[{status}] {name}: shape={a.shape} bit_exact={entry['bit_exact']} "
        f"nan_ok={nan_ok} inf_ok={inf_ok}{extra}"
    )
    for repro in entry.get("repros", []):
        print(
            f"        repro idx={repro['index']} engine={_fmt(repro['engine'])} "
            f"(0x{repro['engine_bits'] & 0xFFFFFFFF:08x}) "
            f"dense={_fmt(repro['dense'])} (0x{repro['dense_bits'] & 0xFFFFFFFF:08x})"
        )
    return ok


# ---------------------------------------------------------------------------
# parity battery
# ---------------------------------------------------------------------------


def _descriptor(max_tree_len: int, input_len: int, const_prob: float, funcs: Sequence[str], layers: int) -> GenerateDescriptor:
    return GenerateDescriptor(
        max_tree_len=max_tree_len,
        input_len=input_len,
        output_len=1,
        using_funcs=list(funcs),
        max_layer_cnt=layers,
        const_prob=const_prob,
        const_range=(-2.0, 2.0),
        sample_cnt=8,
        layer_leaf_prob=0.25,
    )


def _random_population(seed: int, pop: int, max_tree_len: int, input_len: int, const_prob: float, funcs: Sequence[str], layers: int):
    torch.manual_seed(seed)
    descriptor = _descriptor(max_tree_len, input_len, const_prob, funcs, layers)
    forest = Forest.random_generate(pop_size=pop, descriptor=descriptor)
    return (
        forest.batch_node_value.cpu().numpy(),
        forest.batch_node_type.cpu().numpy(),
        forest.batch_subtree_size.cpu().numpy(),
    )


def _engine_reference(
    values: np.ndarray,
    types: np.ndarray,
    sizes: np.ndarray,
    input_len: int,
    x: torch.Tensor,
    per_row: bool,
    backend: str,
    out_device: Any = None,
) -> torch.Tensor:
    """Specialized-engine reference result on ``backend`` (dense vs engine are
    always compared on ONE backend, so ``backend`` is passed through unchanged;
    the engine's ``per_row=True`` contract is pinned by
    ``case_engine_per_row_iree_bug``).
    """
    population = E.decode_population(values, types, sizes, input_len, 1)
    return E.evaluate(
        population, x, per_row=per_row, backend=backend, out_device=out_device
    )


def case_random_populations(args, report, results: Dict[str, Any]) -> bool:
    ok = True
    configs = [
        dict(seed=1, pop=24, max_tree_len=40, input_len=2, const_prob=0.3, funcs=FUNCS_NAMES, layers=3),
        dict(seed=2, pop=16, max_tree_len=40, input_len=3, const_prob=0.0, funcs=FUNCS_NAMES, layers=3),
        dict(seed=3, pop=12, max_tree_len=64, input_len=4, const_prob=0.6, funcs=FUNCS_NAMES, layers=4),
        dict(seed=4, pop=32, max_tree_len=40, input_len=2, const_prob=0.9, funcs=FUNCS_NAMES, layers=3),
    ]
    if args.full:
        configs += [
            dict(seed=5, pop=64, max_tree_len=64, input_len=5, const_prob=0.5, funcs=FUNCS_NAMES, layers=4),
            dict(seed=6, pop=48, max_tree_len=40, input_len=1, const_prob=0.2, funcs=FUNCS_NAMES, layers=3),
            dict(seed=7, pop=40, max_tree_len=40, input_len=3, const_prob=0.75, funcs=FUNCS_NAMES, layers=3),
        ]
    for cfg in configs:
        values, types, sizes = _random_population(
            cfg["seed"], cfg["pop"], cfg["max_tree_len"], cfg["input_len"],
            cfg["const_prob"], cfg["funcs"], cfg["layers"],
        )
        P = values.shape[0]
        x = torch.randn(16, cfg["input_len"]) * 2.0
        tag = f"random seed={cfg['seed']} pop={P} const_prob={cfg['const_prob']}"
        got = D.evaluate_dense(
            values, types, sizes, cfg["input_len"], 1, x,
            per_row=False, backend=args.backend, iree_device_index=args.gpu,
            max_tree_len=args.max_tree_len, update_mode=args.mode,
        )
        ref = _engine_reference(values, types, sizes, cfg["input_len"], x, False, args.backend)
        ok &= compare(f"{tag} per_row=False", ref, got, report)
        xp = torch.randn(P, cfg["input_len"]) * 2.0
        got_p = D.evaluate_dense(
            values, types, sizes, cfg["input_len"], 1, xp,
            per_row=True, backend=args.backend, iree_device_index=args.gpu,
            max_tree_len=args.max_tree_len, update_mode=args.mode,
        )
        ref_p = _engine_reference(values, types, sizes, cfg["input_len"], xp, True, args.backend)
        ok &= compare(f"{tag} per_row=True", ref_p, got_p, report)
        labels = torch.randn(16, 1)
        for use_mse in (True, False):
            got_f = D.sr_fitness_dense(
                values, types, sizes, cfg["input_len"], 1, x, labels, use_MSE=use_mse,
                backend=args.backend, iree_device_index=args.gpu,
                max_tree_len=args.max_tree_len, update_mode=args.mode,
            )
            ref_f = E.sr_fitness(
                E.decode_population(values, types, sizes, cfg["input_len"], 1),
                x, labels, use_MSE=use_mse, backend=args.backend,
            )
            ok &= compare(
                f"{tag} sr_fitness(use_MSE={use_mse})", ref_f, got_f, report
            )
        results.setdefault("contracts", {}).setdefault(tag, _contract_checks(got, ref, got))
    return ok


class _count_uploads:
    """Count ``etl.from_numpy`` calls (the engine's host->device uploads)."""

    def __init__(self) -> None:
        self.count = 0
        self._original: Any = None

    def __enter__(self) -> "_count_uploads":
        self._original = E.etl.from_numpy
        original = self._original

        def counting(array: Any) -> Any:
            self.count += 1
            return original(array)

        E.etl.from_numpy = counting
        return self

    def __exit__(self, *exc: Any) -> bool:
        E.etl.from_numpy = self._original
        return False


def case_engine_per_row_iree_bug(args, report, results: Dict[str, Any]) -> bool:
    """REGRESSION: ``evaluate(..., per_row=True)`` gives every tree its OWN row.

    Minimal repro: two ``SUB(var0, var1)`` trees over ``x=[[10, 1], [20, 2]]``
    must evaluate to ``[9, 18]`` (tree ``i`` on row ``i``).  ``_run_tree`` used to
    memoize its host->device upload as ``prepared[0]`` and reuse it for every
    tree of one ``evaluate`` call: correct for ``per_row=False`` (all trees see
    the same rows), but on the ``iree`` backend ``per_row=True`` silently ran
    every tree on ``x``'s ROW 0 (``[9, 9]``); the ``numpy`` backend consumes the
    per-tree ``xin`` directly and was always correct.

    The upload counts pin the cost side of the contract: one upload per tree for
    ``per_row=True``, exactly ONE upload for the shared ``per_row=False`` block.
    """
    values, types, sizes = pack_trees(
        [TreeSpec([("b", float(Func.SUB)), ("var", 0.0), ("var", 1.0)])] * 2,
        row_len=4,
    )
    x = torch.tensor([[10.0, 1.0], [20.0, 2.0]], dtype=torch.float32)
    population = E.decode_population(values, types, sizes, 2, 1)
    expected = np.asarray([[9.0], [18.0]], dtype=np.float32)
    expected_all = np.asarray([[9.0, 18.0], [9.0, 18.0]], dtype=np.float32)

    got = E.evaluate(
        population, x, per_row=True, backend=args.backend,
        iree_device_index=args.gpu,
    )
    ok = compare(
        f"engine per_row=True ({args.backend}) vs [9, 18]", expected, got, report
    )
    with _count_uploads() as per_row_uploads:
        E.evaluate(
            population, x, per_row=True, backend=args.backend,
            iree_device_index=args.gpu,
        )
    with _count_uploads() as shared_uploads:
        got_all = E.evaluate(
            population, x, per_row=False, backend=args.backend,
            iree_device_index=args.gpu,
        )
    ok &= compare(
        f"engine per_row=False ({args.backend}) vs [[9, 18], [9, 18]]",
        expected_all, got_all.squeeze(-1), report,
    )
    uploads = {
        "per_row_true": per_row_uploads.count,
        "per_row_false": shared_uploads.count,
    }
    # iree uploads per tree when every tree sees its own row, and shares the one
    # (N, input_len) block otherwise; the numpy backend never uploads.
    expected_uploads = (
        {"per_row_true": int(values.shape[0]), "per_row_false": 1}
        if args.backend == "iree"
        else {"per_row_true": 0, "per_row_false": 0}
    )
    ok_uploads = uploads == expected_uploads
    ok &= ok_uploads
    results.setdefault("engine_per_row", {}).update(uploads)
    report["cases"].append(
        {
            "case": f"engine per_row upload counts ({args.backend})",
            "ok": ok_uploads,
            "uploads": uploads,
            "expected": expected_uploads,
        }
    )
    print(
        f"[{'ok ' if ok_uploads else 'FAIL'}] engine per_row upload counts "
        f"({args.backend}): {uploads} expected {expected_uploads}"
    )
    return ok


def _contract_checks(got: torch.Tensor, ref: torch.Tensor, got_f: torch.Tensor) -> Dict[str, Any]:
    """Shape/dtype/contiguity/requires_grad/writability contract checks."""
    checks = {
        "dtype_is_float32": got.dtype == torch.float32,
        "same_shape_as_engine": tuple(got.shape) == tuple(ref.shape),
        "contiguous": bool(got.is_contiguous()),
        "no_grad": not got.requires_grad,
        "writable": False,
        "fitness_positive": bool((_as_host(got_f) >= 0).all() or not np.isfinite(_as_host(got_f)).all() is False),
    }
    try:
        probe = got.clone()
        probe.add_(0.0)
        checks["writable"] = True
    except Exception:
        checks["writable"] = False
    return checks


def _edge_values() -> List[float]:
    return [
        0.0,
        -0.0,
        1e-9,
        -1e-9,
        1e-8,
        -1e-8,
        1e-5,
        -1e-5,
        1e-4,
        -1e-4,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        3.0,
        -3.0,
        100.0,
        -100.0,
        1e30,
        -1e30,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]


def case_function_sweep(args, report) -> bool:
    """Every function id against every edge operand pair (VAR leaves)."""
    values_h = _edge_values()
    ok = True
    specs: List[TreeSpec] = []
    rows: List[Tuple[float, float, float]] = []
    max_id = int(Func.END)
    funcs = range(0, max_id) if args.full else range(0, max_id, 2)
    for fid in funcs:
        arity = 1 if fid >= int(Func.SIN) else (3 if fid == int(Func.IF) else 2)
        for a in values_h:
            for b in values_h if arity >= 2 else [0.0]:
                for c in values_h if arity == 3 else [0.0]:
                    if arity == 1:
                        nodes = [("u", float(fid)), ("var", 0.0)]
                    elif arity == 2:
                        nodes = [("b", float(fid)), ("var", 0.0), ("var", 1.0)]
                    else:
                        nodes = [
                            ("t", float(fid)),
                            ("var", 0.0),
                            ("var", 1.0),
                            ("var", 2.0),
                        ]
                    specs.append(TreeSpec(nodes))
                    rows.append((a, b, c))
                    if not args.full and len(specs) >= 1200:
                        break
                if not args.full and len(specs) >= 1200:
                    break
            if not args.full and len(specs) >= 1200:
                break
        if not args.full and len(specs) >= 1200:
            break
    values, types, sizes = pack_trees(specs, row_len=8)
    x = torch.tensor(rows, dtype=torch.float32)
    got = D.evaluate_dense(
        values, types, sizes, 3, 1, x, per_row=True, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len, update_mode=args.mode,
    )
    ref = _engine_reference(values, types, sizes, 3, x, True, args.backend)
    ok &= compare(f"function sweep ({len(specs)} trees x edge values)", ref, got, report)
    return ok


def case_unknown_ids(args, report) -> bool:
    """Out-of-table function ids, wrong-arity ids and unknown base types.

    Every tree here is *structurally valid* (the arity comes from ``node_type``
    alone, never from the function id), so both engines evaluate it: the
    specialized decoder keeps the id and the emission table falls through to
    ``0.0`` (binary/unary) or ignores it (ternary/unknown base), and the dense
    graph must reach the same value.
    """
    specs = [
        TreeSpec([("b", 29.0), ("var", 0.0), ("const", 1.0)]),  # id past the table
        TreeSpec([("b", 1000.0), ("var", 0.0), ("const", -1.0)]),
        TreeSpec([("b", -3.0), ("var", 0.0), ("const", 2.0)]),  # truncates to 0 (IF id)
        TreeSpec([("u", 29.0), ("var", 0.0)]),
        TreeSpec([("u", -7.0), ("var", 0.0)]),
        TreeSpec([("b", 14.0), ("var", 0.0), ("var", 1.0)]),  # unary id on a binary node
        TreeSpec([("u", 3.0), ("var", 0.0)]),  # binary id on a unary node
        TreeSpec([("t", 5.0), ("var", 0.0), ("const", 3.0), ("const", 4.0)]),
        TreeSpec([("t", Func.IF), ("var", 0.0), ("const", 3.0), ("const", 4.0)]),
        TreeSpec([("b", 2.9), ("var", 0.0), ("var", 1.0)]),  # truncation
        TreeSpec([("b", -0.5), ("var", 0.0), ("var", 1.0)]),  # truncation to 0
        TreeSpec([("t", 0.0), ("var", 0.0), ("var", 1.0), ("const", 1.0)]),
    ]
    values, types, sizes = pack_trees(specs, row_len=8)
    # unknown base types: every base outside {0..4} takes the ternary branch and
    # ignores the function id (the kernel's `else` case)
    types[7, 0] = 5
    types[8, 0] = int(NType.TFUNC) | 0
    types[11, 0] = 127
    values[11, 0] = 1e9  # must be ignored: the ternary branch drops the id
    x = torch.tensor(
        [[0.5, -1.5], [2.0, 0.25], [0.0, -0.0], [-3.0, 5.0]] * 3,
        dtype=torch.float32,
    )[: len(specs)]
    got = D.evaluate_dense(
        values, types, sizes, 2, 1, x, per_row=True, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len, update_mode=args.mode,
    )
    ref = _engine_reference(values, types, sizes, 2, x, True, args.backend)
    ok = compare("unknown ids / base types / truncation", ref, got, report)
    ok &= case_unsupported_fallback(args, report)
    return ok
def case_unsupported_fallback(args, report) -> bool:
    """A malformed structure must signal ``DenseUnsupported`` (so the caller can
    fall back); the specialized path still owns the ValueError contract."""
    specs = [
        TreeSpec([("b", float(Func.ADD)), ("var", 0.0), ("var", 1.0)]),
    ]
    values, types, sizes = pack_trees(specs, row_len=8)
    sizes[0, 0] = 2  # declared subtree size disagrees with the structure
    x = torch.tensor([[0.5, -1.5]], dtype=torch.float32)
    signaled = False
    reason = ""
    try:
        D.evaluate_dense(
            values, types, sizes, 2, 1, x, per_row=True, backend=args.backend,
            iree_device_index=args.gpu, max_tree_len=args.max_tree_len,
            update_mode=args.mode,
        )
    except D.DenseUnsupported as exc:
        signaled = True
        reason = str(getattr(exc, "reason", exc))
    engine_raised = False
    try:
        E.decode_population(values, types, sizes, 2, 1)
    except ValueError:
        engine_raised = True
    ok = bool(signaled and engine_raised)
    report["cases"].append(
        {
            "case": "malformed structure -> DenseUnsupported",
            "ok": ok,
            "dense_signaled": signaled,
            "reason": reason,
            "engine_raised_valueerror": engine_raised,
        }
    )
    print(
        f"[{'ok ' if ok else 'FAIL'}] malformed structure -> DenseUnsupported "
        f"(dense={signaled} {reason!r}, engine ValueError={engine_raised})"
    )
    return ok


def case_junk_tails(args, report) -> bool:
    """Same trees, different junk tails -> identical bits (and engine parity)."""
    specs = [
        TreeSpec([("b", float(Func.ADD)), ("var", 0.0), ("const", 1.0)]),
        TreeSpec([("u", float(Func.SQRT)), ("var", 1.0)]),
        constant_chain(6),
        TreeSpec(
            [
                ("t", float(Func.IF)),
                ("b", float(Func.GT)),
                ("var", 0.0),
                ("const", 0.0),
                ("const", 2.0),
                ("u", float(Func.SIN)),
                ("var", 1.0),
            ]
        ),
    ]
    x = torch.tensor(
        [[1.0, 0.0], [-2.0, 0.5], [0.0, -1.0], [3.0, -0.25]], dtype=torch.float32
    )[: len(specs)]
    clean = pack_trees(specs, row_len=16)
    dirty = pack_trees(specs, row_len=16, junk=random_junk)
    ok = True
    outs = []
    for name, (values, types, sizes) in (("clean", clean), ("dirty", dirty)):
        got = D.evaluate_dense(
            values, types, sizes, 2, 1, x, per_row=True, backend=args.backend,
            iree_device_index=args.gpu, max_tree_len=args.max_tree_len, update_mode=args.mode,
        )
        ref = _engine_reference(values, types, sizes, 2, x, True, args.backend)
        ok &= compare(f"junk tails ({name}) vs engine", ref, got, report)
        outs.append(_as_host(got))
    ok &= compare("junk tails (dirty vs clean)", outs[0], outs[1], report)
    return ok


def case_fold_search(args, report, results: Dict[str, Any]) -> bool:
    """Const-heavy populations: search for fold-vs-emit differences."""
    ok = True
    seeds = [11, 12, 13, 14] if args.full else [11, 12]
    total_non_bit = 0
    for seed in seeds:
        for const_prob in (1.0, 0.999, 0.99, 0.95):
            values, types, sizes = _random_population(
                seed, 64, 64, 3, const_prob, FUNCS_NAMES, 4
            )
            x = torch.randn(24, 3) * 3.0
            got = D.evaluate_dense(
                values, types, sizes, 3, 1, x, per_row=False, backend=args.backend,
                iree_device_index=args.gpu, max_tree_len=args.max_tree_len,
                update_mode=args.mode,
            )
            ref = _engine_reference(values, types, sizes, 3, x, False, args.backend)
            before = len(report["cases"])
            ok &= compare(
                f"fold search seed={seed} const_prob={const_prob}", ref, got, report
            )
            total_non_bit += report["cases"][before]["n_non_bit_finite"]
    results["fold_search_non_bit_exact_finite"] = total_non_bit
    constant_values, constant_types, constant_sizes = pack_trees(
        [constant_chain(5), constant_chain(9, 2.5), constant_tree(9)], row_len=16
    )
    x = torch.randn(11, 2)
    got = D.evaluate_dense(
        constant_values, constant_types, constant_sizes, 2, 1, x, per_row=False,
        backend=args.backend, iree_device_index=args.gpu,
        max_tree_len=args.max_tree_len, update_mode=args.mode,
    )
    ref = _engine_reference(constant_values, constant_types, constant_sizes, 2, x, False, args.backend)
    ok &= compare("fully constant trees (fold vs emit)", ref, got, report)
    return ok


def case_shapes(args, report) -> bool:
    """Chunked execution: a small budget forces many P/N chunks."""
    values, types, sizes = _random_population(21, 24, 40, 2, 0.3, FUNCS_NAMES, 3)
    x = torch.randn(37, 2)
    got = D.evaluate_dense(
        values, types, sizes, 2, 1, x, per_row=False, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len,
        update_mode=args.mode, budget_mb=0.0008,
    )
    ref = _engine_reference(values, types, sizes, 2, x, False, args.backend)
    plan = D.plan_dense(
        values, types, sizes, 2, 1, 37, per_row=False, max_tree_len=args.max_tree_len,
        budget_mb=0.0008, update_mode=args.mode, backend=args.backend,
    )
    report["chunking"] = D.describe_plan(plan)
    ok = compare(
        f"chunked (p_chunk={plan.p_chunk}, n_chunk={plan.n_chunk})", ref, got, report
    )
    xp = torch.randn(24, 2)
    got_p = D.evaluate_dense(
        values, types, sizes, 2, 1, xp, per_row=True, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len,
        update_mode=args.mode, budget_mb=0.0008,
    )
    ref_p = _engine_reference(values, types, sizes, 2, xp, True, args.backend)
    ok &= compare("chunked per_row=True", ref_p, got_p, report)
    return ok


def case_dense_unsupported(args, report, results: Dict[str, Any]) -> bool:
    """DenseUnsupported must fire for everything outside the supported subset."""
    values, types, sizes = _random_population(31, 4, 40, 2, 0.3, FUNCS_NAMES, 3)
    checks: Dict[str, bool] = {}

    def expect(name: str, fn: Callable[[], Any]) -> bool:
        try:
            fn()
        except D.DenseUnsupported:
            checks[name] = True
            return True
        checks[name] = False
        return False

    ok = True
    ok &= expect("output_len>1", lambda: D.prepare_dense(values, types, sizes, 2, 2))
    long_values, long_types, long_sizes = _random_population(
        32, 4, 40, 2, 0.3, FUNCS_NAMES, 4
    )
    ok &= expect(
        "tree longer than cap",
        lambda: D.prepare_dense(long_values, long_types, long_sizes, 2, 1, max_tree_len=4),
    )
    bad_types = types.copy()
    bad_types[0, 0] = int(bad_types[0, 0]) | int(NType.OUT_NODE)
    ok &= expect(
        "OUT_NODE tag",
        lambda: D.prepare_dense(values, bad_types, sizes, 2, 1),
    )
    bad_sizes = sizes.copy()
    bad_sizes[0, 0] = 999
    ok &= expect(
        "tree length beyond the row",
        lambda: D.prepare_dense(values, bad_sizes, sizes, 2, 1, max_tree_len=40),
    )
    var_values, var_types, var_sizes = pack_trees(
        [TreeSpec([("b", float(Func.ADD)), ("var", 0.0), ("var", 1.0)])], row_len=8
    )
    var_values[0, 1] = 99.0  # node 1 is a VAR -> index 99 is out of range for input_len=2
    ok &= expect(
        "variable index out of range",
        lambda: D.prepare_dense(var_values, var_types, var_sizes, 2, 1),
    )
    bad_struct = sizes.copy()
    bad_struct[0, 0] = max(1, int(bad_struct[0, 0]) - 1)
    ok &= expect(
        "inconsistent subtree size",
        lambda: D.prepare_dense(values, types, bad_struct, 2, 1, max_tree_len=40),
    )
    results["unsupported_signals"] = checks
    print(f"[{'ok ' if ok else 'FAIL'}] DenseUnsupported signals: {checks}")
    return ok


def case_compile_reuse(args, report, results: Dict[str, Any]) -> bool:
    """Two calls, different populations, same shape -> exactly one build."""
    values_a, types_a, sizes_a = _random_population(41, 16, 40, 2, 0.3, FUNCS_NAMES, 3)
    values_b, types_b, sizes_b = _random_population(42, 16, 40, 2, 0.3, FUNCS_NAMES, 3)
    x = torch.randn(9, 2)
    before = E.cache_stats()
    D.evaluate_dense(
        values_a, types_a, sizes_a, 2, 1, x, per_row=False, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len, update_mode=args.mode,
    )
    after_first = E.cache_stats()
    D.evaluate_dense(
        values_b, types_b, sizes_b, 2, 1, x, per_row=False, backend=args.backend,
        iree_device_index=args.gpu, max_tree_len=args.max_tree_len, update_mode=args.mode,
    )
    after_second = E.cache_stats()
    built_first = after_first["misses"] - before["misses"]
    built_second = after_second["misses"] - after_first["misses"]
    results["compile_reuse"] = {
        "built_on_first_call": built_first,
        "built_on_second_call": built_second,
        "entries_after": after_second["entries"],
    }
    ok = built_second == 0 and built_first >= 1
    print(
        f"[{'ok ' if ok else 'FAIL'}] compile reuse: builds first={built_first} "
        f"second={built_second} (dense chunk shapes={D.plan_dense(values_a, types_a, sizes_a, 2, 1, 9, per_row=False, max_tree_len=args.max_tree_len, update_mode=args.mode).p_chunks * D.plan_dense(values_a, types_a, sizes_a, 2, 1, 9, per_row=False, max_tree_len=args.max_tree_len, update_mode=args.mode).n_chunks})"
    )
    report["cases"].append({"case": "compile reuse", "ok": ok, **results["compile_reuse"]})
    return ok


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def _chain_population(L: int, pop: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pack_trees([constant_chain(L)] * pop, row_len=L)


def measure_l_scaling(args, results: Dict[str, Any]) -> None:
    """Trace / lower / compile / load / run vs the trace length L."""
    print("\n== L_trace scaling (dense graph cost; P=8, N=8) ==")
    print(
        f"{'L':>4} {'ops':>7} {'trace_ms':>9} {'lower_ms':>9} {'compile_ms':>11} "
        f"{'load_ms':>8} {'build_ms':>9} {'run_ms':>8} {'peak_MB':>8}"
    )
    backend_name, device, options = E._resolve_backend(args.backend, args.gpu)
    table = []
    lengths = [8, 16] if args.tiny else ([8, 16, 32] if args.quick else [8, 16, 32, 64])
    for L in lengths:
        values, types, sizes = _chain_population(L)
        x = torch.randn(8, 2, dtype=torch.float32).numpy()
        plan = D.plan_dense(
            values, types, sizes, 2, 1, 8, per_row=False, max_tree_len=L,
            update_mode=args.mode, backend=args.backend, budget_mb=args.budget_mb,
        )
        stage: Dict[str, float] = {}
        t0 = time.perf_counter()
        graph = etl.trace(D._dense_step, *plan.specs)
        stage["trace_ms"] = (time.perf_counter() - t0) * 1e3
        ops = len(list(graph.module.main.entry_block.ops))
        t0 = time.perf_counter()
        lowered = etl.lower(graph, backend=backend_name, **options)
        stage["lower_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        artifact = etl.compile(lowered, backend=backend_name, **options)
        stage["compile_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        executable = etl.load(artifact, backend=backend_name, device=device)
        stage["load_ms"] = (time.perf_counter() - t0) * 1e3
        cache = etl.CompileCache(maxsize=4)
        t0 = time.perf_counter()
        etl.build_cached(
            D._dense_step, *plan.specs, backend=backend_name, device=device,
            cache=cache, **options,
        )
        stage["build_ms"] = (time.perf_counter() - t0) * 1e3
        timing = _median_ms(
            lambda: D.run_plan(
                plan, executable, x, backend_name=backend_name, device=device
            ),
            args.repeats,
        )
        row = {
            "L": L,
            "ops": ops,
            "peak_bytes_model": plan.peak_bytes,
            "run_median_ms": timing["median_ms"],
            **stage,
        }
        table.append(row)
        print(
            f"{L:>4} {ops:>7} {stage['trace_ms']:>9.2f} {stage['lower_ms']:>9.2f} "
            f"{stage['compile_ms']:>11.2f} {stage['load_ms']:>8.2f} "
            f"{stage['build_ms']:>9.2f} {timing['median_ms']:>8.3f} "
            f"{plan.peak_bytes / 2**20:>8.3f}"
        )
    results["l_scaling"] = table


def measure_crossover(args, results: Dict[str, Any]) -> None:
    """Dense vs specialized warm run as a function of (P, N)."""
    print("\n== dense vs specialized (warm run, numpy-vs-backend) ==")
    print(
        f"{'P':>5} {'N':>6} {'decode_ms':>9} {'eng_cold_ms':>11} {'eng_warm_ms':>11} "
        f"{'dense_prep_ms':>13} {'dense_cold_ms':>13} {'dense_warm_ms':>13} "
        f"{'speedup':>8} {'dense_peak_MB':>13}"
    )
    rows = []
    p_values = [4, 16] if args.tiny else ([64, 200] if args.quick else [64, 200, 1000])
    n_values = [64, 1024] if args.quick or args.tiny else [64, 1024, 10000]
    backend_name, device, options = E._resolve_backend(args.backend, args.gpu)
    for P in p_values:
        for N in n_values:
            torch.manual_seed(100 + P)
            descriptor = _descriptor(64, 2, 0.3, FUNCS_NAMES, 4)
            forest = Forest.random_generate(pop_size=P, descriptor=descriptor)
            values = forest.batch_node_value.cpu().numpy()
            types = forest.batch_node_type.cpu().numpy()
            sizes = forest.batch_subtree_size.cpu().numpy()
            x = torch.randn(N, 2)
            x_host = x.numpy()
            # --- specialized ---------------------------------------------------
            t0 = time.perf_counter()
            population = E.decode_population(values, types, sizes, 2, 1)
            decode_ms = (time.perf_counter() - t0) * 1e3
            E.clear_cache()
            t0 = time.perf_counter()
            E.evaluate(population, x, per_row=False, backend=args.backend)
            engine_cold_ms = (time.perf_counter() - t0) * 1e3
            eng = _median_ms(
                lambda: E.evaluate(population, x, per_row=False, backend=args.backend),
                args.repeats,
            )
            E.clear_cache()
            # --- dense ---------------------------------------------------------
            t0 = time.perf_counter()
            plan = D.plan_dense(
                values, types, sizes, 2, 1, N, per_row=False, backend=args.backend,
                max_tree_len=args.max_tree_len, update_mode=args.mode,
                budget_mb=args.budget_mb,
            )
            dense_prep_ms = (time.perf_counter() - t0) * 1e3
            cache = etl.CompileCache(maxsize=4)
            t0 = time.perf_counter()
            executable = etl.build_cached(
                D._dense_step, *plan.specs, backend=backend_name, device=device,
                cache=cache, **options,
            )
            dense_cold_ms = (time.perf_counter() - t0) * 1e3
            dense = _median_ms(
                lambda: D.run_plan(
                    plan, executable, x_host, backend_name=backend_name, device=device
                ),
                args.repeats,
            )
            row = {
                "backend": args.backend,
                "update_mode": plan.update_mode,
                "P": P,
                "N": N,
                "L": plan.trace_len,
                "p_chunk": plan.p_chunk,
                "n_chunk": plan.n_chunk,
                "chunks": plan.p_chunks * plan.n_chunks,
                "max_tree_length": plan.prep.max_tree_length,
                "engine_decode_ms": decode_ms,
                "engine_cold_ms": engine_cold_ms,
                "engine_warm_ms": eng["median_ms"],
                "dense_prep_ms": dense_prep_ms,
                "dense_cold_ms": dense_cold_ms,
                "dense_warm_ms": dense["median_ms"],
                "speedup": eng["median_ms"] / dense["median_ms"],
                "dense_peak_bytes_model": plan.peak_bytes,
            }
            rows.append(row)
            print(
                f"{P:>5} {N:>6} {decode_ms:>9.2f} {engine_cold_ms:>11.1f} "
                f"{eng['median_ms']:>11.3f} {dense_prep_ms:>13.2f} "
                f"{dense_cold_ms:>13.1f} {dense['median_ms']:>13.3f} "
                f"{row['speedup']:>8.2f} {plan.peak_bytes / 2**20:>13.3f}"
            )
            E.clear_cache()
    results["crossover"] = rows


def measure_iree(args, results: Dict[str, Any]) -> None:
    """IREE compile/run economics plus repeated-invoke scatter stability."""
    print("\n== iree-cuda economics ==")
    backend_name, device, options = E._resolve_backend("iree", args.gpu)
    for L in (8, 16) if args.quick else (8, 16, 32):
        values, types, sizes = _chain_population(L, pop=4)
        x = torch.randn(64, 2, dtype=torch.float32).numpy()
        for mode in ("grow",):
            plan = D.plan_dense(
                values, types, sizes, 2, 1, 64, per_row=False, backend="iree",
                max_tree_len=L, update_mode=mode, budget_mb=args.budget_mb,
            )
            cache = etl.CompileCache(maxsize=4)
            t0 = time.perf_counter()
            executable = etl.build_cached(
                D._dense_step, *plan.specs, backend=backend_name, device=device,
                cache=cache, **options,
            )
            build_ms = (time.perf_counter() - t0) * 1e3
            timing = _median_ms(
                lambda: D.run_plan(
                    plan, executable, x, backend_name=backend_name, device=device
                ),
                args.repeats,
            )
            row = {
                "L": L,
                "mode": mode,
                "compile_build_ms": build_ms,
                "run_median_ms": timing["median_ms"],
            }
            results.setdefault("iree", []).append(row)
            print(
                f"L={L:>3} mode={mode} build_ms={build_ms:.1f} "
                f"run_median_ms={timing['median_ms']:.3f}"
            )

    # -- OPTIONAL EXPERIMENT: is scatter on an in-graph buffer stable? ---------
    if os.environ.get(D.ALLOW_UNSAFE_SCATTER_ENV, "").strip() in ("", "0"):
        print(
            f"[skip] in-graph scatter experiment needs "
            f"{D.ALLOW_UNSAFE_SCATTER_ENV}=1"
        )
        return
    values, types, sizes = _chain_population(16, pop=4)
    x = torch.randn(32, 2, dtype=torch.float32).numpy()
    plan = D.plan_dense(
        values, types, sizes, 2, 1, 32, per_row=False, backend="iree",
        max_tree_len=16, update_mode="scatter",
    )
    cache = etl.CompileCache(maxsize=4)
    executable = etl.build_cached(
        D._dense_step, *plan.specs, backend=backend_name, device=device,
        cache=cache, **options,
    )
    def one():
        columns = D._prep_columns(plan.prep, 0, plan.p_chunk)
        args_in = tuple(etl.from_numpy(a).to(device) for a in columns) + (
            etl.from_numpy(x).to(device),
            plan.cfg,
        )
        with E._quiet():
            out = etl.run(executable, *args_in)
        return np.asarray(out.to(E._CPU).numpy(), dtype=np.float32)

    first = one()
    same = True
    for _ in range(100):
        same &= bool(np.array_equal(first, one()))
    grow_plan = D.plan_dense(
        values, types, sizes, 2, 1, 32, per_row=False, backend="iree",
        max_tree_len=16, update_mode="grow",
    )
    growth = etl.build_cached(
        D._dense_step, *grow_plan.specs, backend=backend_name, device=device,
        cache=cache, **options,
    )
    growth_out = D.run_plan(grow_plan, growth, x, backend_name=backend_name, device=device)
    results.setdefault("iree_experiments", {})["scatter_stable_across_100_invokes"] = same
    results["iree_experiments"]["scatter_matches_grow"] = bool(
        np.array_equal(first, growth_out.reshape(first.shape))
    )
    print(
        f"[exp] in-graph-buffer scatter stable over 100 invokes: {same}; "
        f"bit-identical to grow: {results['iree_experiments']['scatter_matches_grow']}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("numpy", "iree"), default="numpy")
    parser.add_argument("--mode", choices=D.UPDATE_MODES, default=None)
    parser.add_argument("--quick", action="store_true", default=False)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument(
        "--tiny",
        action="store_true",
        default=False,
        help="lowest-GPU-budget measurement profile (P in {4,16}, L in {8,16})",
    )
    parser.add_argument("--gpu", type=int, default=0, help="IREE device index (visible)")
    parser.add_argument("--max-tree-len", type=int, default=64)
    parser.add_argument("--budget-mb", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-parity", action="store_true", default=False)
    parser.add_argument("--skip-measure", action="store_true", default=False)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help=f"run only the cases whose name contains NAME (repeatable): {list(CASE_NAMES)}",
    )
    parser.add_argument("--json", default=None)
    return parser


CASE_NAMES = (
    "random_populations",
    "function_sweep",
    "unknown_ids",
    "engine_per_row",
    "junk_tails",
    "fold_search",
    "shapes",
    "dense_unsupported",
    "compile_reuse",
)


def _wanted(args, name: str) -> bool:
    """``--only`` filter: run a case when one of its substrings matches."""
    if not args.only:
        return True
    return any(token in name for token in args.only)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.only and not any(_wanted(args, name) for name in CASE_NAMES):
        print(f"[warn] --only {args.only} matched no case: {list(CASE_NAMES)}")
        return 2
    if os.environ.get("ETL_DISABLE_LOCATIONS", "").strip() not in ("1", "true"):
        print(
            "[warn] ETL_DISABLE_LOCATIONS=1 is not set: tracing will be ~4x slower"
        )
    report: Dict[str, Any] = {
        "backend": args.backend,
        "mode": args.mode,
        "quick": args.quick,
        "full": args.full,
        "etl_disable_locations": os.environ.get("ETL_DISABLE_LOCATIONS", ""),
        "cases": [],
    }
    results: Dict[str, Any] = {}
    if not args.quick and not args.full:
        args.quick = True
    ok = True
    print(
        f"== dense spike check: backend={args.backend} mode={args.mode} "
        f"quick={args.quick} full={args.full} =="
    )
    if not args.skip_parity:
        if _wanted(args, "random_populations"):
            ok &= case_random_populations(args, report, results)
        if _wanted(args, "function_sweep"):
            ok &= case_function_sweep(args, report)
        if _wanted(args, "unknown_ids"):
            ok &= case_unknown_ids(args, report)
        if _wanted(args, "engine_per_row"):
            ok &= case_engine_per_row_iree_bug(args, report, results)
        if _wanted(args, "junk_tails"):
            ok &= case_junk_tails(args, report)
        if _wanted(args, "fold_search"):
            ok &= case_fold_search(args, report, results)
        if _wanted(args, "shapes"):
            ok &= case_shapes(args, report)
        if _wanted(args, "dense_unsupported"):
            ok &= case_dense_unsupported(args, report, results)
        if _wanted(args, "compile_reuse"):
            ok &= case_compile_reuse(args, report, results)
    if not args.skip_measure:
        if args.backend == "numpy":
            measure_l_scaling(args, results)
            measure_crossover(args, results)
        else:
            measure_iree(args, results)
    report["results"] = results
    report["parity_ok"] = ok
    path = args.json or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"dense_spike_{args.backend}_{'quick' if args.quick else 'full'}.json",
    )
    try:
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\n[json] {path}")
    except Exception as exc:  # pragma: no cover - scratch output only
        print(f"[warn] could not write {path}: {exc}")
    print(
        f"\n== parity {'OK' if ok else 'FAILED'} "
        f"({sum(1 for c in report['cases'] if c.get('ok'))}/{len(report['cases'])} "
        "cases passed) =="
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
