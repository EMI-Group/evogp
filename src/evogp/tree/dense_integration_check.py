"""Parity + fallback checks for the WIRED dense evaluator (``EVOGP_EVAL_MODE``).

Runnable script, deliberately NOT a pytest module (no ``test_`` prefix, no pytest
import): it exercises the *public API* with the dense path forced on and compares
against the same call with the specialized path forced, so it covers the wiring
in ``forest.py`` / ``tree.py`` rather than ``dense_engine``'s internals (the
engine-level battery is ``dense_spike_check.py``).

    # 1. numpy public-API parity + auto-policy check (no GPU needed for the math,
    #    but the tensors go through check_tensor -> "cuda", so a GPU must be visible)
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=3 \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        src/evogp/tree/dense_integration_check.py --stage numpy,fallback

    # 2. iree smoke: P=32/N=32, EVOGP_EVAL_MODE=auto + EVOGP_ETL_BACKEND=iree,
    #    one dense compile per call (cache_stats deltas) -- keep the process short
    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=3 \\
        /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        src/evogp/tree/dense_integration_check.py --stage iree

Stages
------
``numpy``     (default) three populations -- a random one, a deep/const-heavy one
              and a hand-built one containing a constant-only tree -- each
              evaluated both ways: ``Forest.SR_fitness`` (MSE and MAE),
              ``Forest.forward``, ``Forest.batch_forward``, ``Tree.forward`` (1-D
              and 2-D input) and ``Tree.SR_fitness``.  Comparison rule: NaN/±inf
              masks exact, finite values within rtol/atol 1e-4, with the
              non-bit-exact finite count reported.  The returned tensors are
              contract-checked (fresh, contiguous, no grad, float32; the SR loss is
              writable) and the ``auto`` policy on numpy is verified to stay on the
              specialized path (one compile per tree instead of one for the
              population).
``fallback``  an ``output_len=2`` population with OUT_NODE-tagged function nodes:
              ``evaluate_dense`` must signal ``DenseUnsupported``, the public calls
              must fall back silently and produce exactly the specialized result
              (and the analytic out-node values).
``iree``      P=32, N=32 on the CUDA HAL: a numpy+specialized reference, then
              ``EVOGP_EVAL_MODE=auto`` + ``EVOGP_ETL_BACKEND=iree`` through
              ``Forest.SR_fitness``; prints the compile/run/warm times and the
              ``etl_engine.cache_stats()`` deltas that prove ONE dense compile for
              the whole population.

Exit code 0 = every check passed, 1 = at least one mismatch, 2 = a hard error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from evogp.tree import CombinedForest, Forest, GenerateDescriptor  # noqa: E402
from evogp.tree import dense_engine as D  # noqa: E402
from evogp.tree import etl_engine as E  # noqa: E402
from evogp.tree.utils import FUNCS_NAMES, Func, NType  # noqa: E402

# Comparison conventions mirror dense_spike_check.py: masks exact, finite values
# within rtol/atol; bit-exactness is reported, not required (folded constant
# chains vs emitted ones can differ by ulps of the host transcendental).
_RTOL = 1e-4
_ATOL = 1e-4

_STAGES = ("numpy", "fallback", "iree")
_DEVICE = "cuda"


# ---------------------------------------------------------------------------
# tiny utilities
# ---------------------------------------------------------------------------


def _host(tensor: Any) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


class Report:
    """Collects pass/fail entries and prints one line per check."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.entries.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        return ok

    @property
    def ok(self) -> bool:
        return all(entry["ok"] for entry in self.entries)


def compare(name: str, ref: Any, got: Any, report: Report) -> bool:
    """Float32 comparison: NaN/±inf masks exact, finite values within 1e-4."""
    a = np.asarray(_host(ref), dtype=np.float32)
    b = np.asarray(_host(got), dtype=np.float32)
    if a.shape != b.shape:
        return report.add(name, False, f"shape {b.shape} != {a.shape}")
    nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
    inf_ok = np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(
        np.isneginf(a), np.isneginf(b)
    )
    finite = np.isfinite(a) & np.isfinite(b)
    close = (
        bool(np.allclose(a[finite], b[finite], rtol=_RTOL, atol=_ATOL))
        if finite.any()
        else True
    )
    non_bit = int((finite & (a.view(np.int32) != b.view(np.int32))).sum())
    max_rel = 0.0
    if finite.any():
        diff = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        denom = np.abs(a[finite].astype(np.float64))
        rel = np.where(denom > 0, diff / np.maximum(denom, 1e-45), diff)
        max_rel = float(rel.max()) if rel.size else 0.0
    detail = (
        f"shape={a.shape} nan_ok={nan_ok} inf_ok={inf_ok} "
        f"non_bit_exact_finite={non_bit} max_rel={max_rel:.3g}"
    )
    return report.add(name, bool(nan_ok and inf_ok and close), detail)


def check_contract(
    name: str, tensor: torch.Tensor, report: Report, *, require_leaf: bool = True
) -> bool:
    """Fresh / contiguous / non-grad / float32 contract of an engine result.

    ``require_leaf=False`` for the ``Tree.forward`` results: those are ``res[0]``,
    an int-indexed *view* of the engine's fresh ``(1, output_len)`` result (that
    indexing is pre-existing public behaviour, identical on both engines).
    """
    problems = []
    if tensor.dtype != torch.float32:
        problems.append(f"dtype={tensor.dtype}")
    if tensor.requires_grad:
        problems.append("requires_grad=True")
    if not tensor.is_contiguous():
        problems.append("not contiguous")
    if require_leaf and getattr(tensor, "_base", None) is not None:
        problems.append("is a view of another tensor")
    return report.add(name, not problems, ", ".join(problems) or "fresh float32 leaf")


def check_writable(name: str, tensor: torch.Tensor, recompute, report: Report) -> bool:
    """The SR loss must be a FRESH writable tensor (the pipeline mutates it)."""
    before = tensor.clone()
    tensor.fill_(float("nan"))  # what StandardPipeline does with NaN fitness
    after = recompute()
    ok = bool(
        torch.allclose(
            after.detach().cpu().float(),
            before.detach().cpu().float(),
            rtol=_RTOL,
            atol=_ATOL,
            equal_nan=True,
        )
    )
    return report.add(
        name, ok, "in-place write to the returned loss does not affect the next call"
    )


def set_mode(mode: str, backend: Optional[str] = None) -> None:
    """Flip the master switch (and optionally the engine backend) in-process."""
    os.environ["EVOGP_EVAL_MODE"] = mode
    if backend is None:
        os.environ.pop("EVOGP_ETL_BACKEND", None)
    else:
        os.environ["EVOGP_ETL_BACKEND"] = backend


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------

_ARITY = {"var": 0, "const": 0, "u": 1, "b": 2, "t": 3}
_TYPE = {
    "var": int(NType.VAR),
    "const": int(NType.CONST),
    "u": int(NType.UFUNC),
    "b": int(NType.BFUNC),
    "t": int(NType.TFUNC),
}


def tree_arrays(nodes: Sequence[Tuple[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefix-order node list -> (values, types, sizes) with computed subtree sizes."""
    sizes = np.zeros(len(nodes), dtype=np.int32)
    stack: List[int] = []
    for i in range(len(nodes) - 1, -1, -1):
        arity = _ARITY[nodes[i][0]]
        if len(stack) < arity:
            raise ValueError(f"bad tree spec at node {i}: needs {arity} operands")
        sizes[i] = 1 + sum(stack.pop() for _ in range(arity))
        stack.append(int(sizes[i]))
    if len(stack) != 1:
        raise ValueError("bad tree spec: the prefix stream does not reduce to one tree")
    values = np.asarray([float(v) for _, v in nodes], dtype=np.float32)
    types = np.asarray([_TYPE[k] for k, _ in nodes], dtype=np.int16)
    return values, types, sizes


def pack(
    specs: Sequence[Sequence[Tuple[str, float]]],
    row_len: int,
    input_len: int,
    device: str = _DEVICE,
) -> Forest:
    """A ``Forest`` of hand-built trees, zero-padded to ``row_len`` (no junk tail)."""
    pop = len(specs)
    values = np.zeros((pop, row_len), dtype=np.float32)
    types = np.zeros((pop, row_len), dtype=np.int16)
    sizes = np.zeros((pop, row_len), dtype=np.int16)
    for p, spec in enumerate(specs):
        v, t, s = tree_arrays(spec)
        n = int(s[0])
        if n > row_len:
            raise ValueError(f"tree {p} has {n} nodes but the row length is {row_len}")
        values[p, :n], types[p, :n], sizes[p, :n] = v[:n], t[:n], s[:n]
    return Forest(
        input_len,
        1,
        torch.from_numpy(values).to(device),
        torch.from_numpy(types).to(device),
        torch.from_numpy(sizes).to(device),
    )


def random_population(
    seed: int,
    pop: int,
    max_tree_len: int,
    input_len: int,
    const_prob: float,
    funcs: Sequence[str],
    layers: int,
) -> Forest:
    torch.manual_seed(seed)
    descriptor = GenerateDescriptor(
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
    return Forest.random_generate(pop_size=pop, descriptor=descriptor)


def hand_built_population(device: str = _DEVICE) -> Forest:
    """Three small trees, one of them constant-only (the specialized engine folds it)."""
    fn_if, fn_gt, fn_mul, fn_div = (
        float(Func.IF),
        float(Func.GT),
        float(Func.MUL),
        float(Func.DIV),
    )
    return pack(
        [
            [("b", float(Func.ADD)), ("var", 0.0), ("const", 2.5)],
            [("const", 1.25)],  # constant-only tree (never compiled by the engine)
            # if (x1 > 0) then x0 * 3.0 else 1.0 / x0
            [
                ("t", fn_if),
                ("b", fn_gt),
                ("var", 1.0),
                ("const", 0.0),
                ("b", fn_mul),
                ("var", 0.0),
                ("const", 3.0),
                ("b", float(Func.DIV)),
                ("const", 1.0),
                ("var", 0.0),
            ],
            [
                ("b", fn_div),
                ("u", float(Func.SIN)),
                ("var", 2.0),
                ("b", float(Func.ADD)),
                ("const", 0.5),
                ("var", 0.0),
            ],
        ],
        row_len=16,
        input_len=3,
        device=device,
    )


def deep_population(device: str = _DEVICE) -> Forest:
    """A 21-node unary chain and a 13-node binary chain (long dense traces)."""
    unary = [("const", 1.5)] + [
        ("u", float(Func.SIN if i % 2 else Func.COS)) for i in range(20)
    ]
    binary = [("var", 0.0)]
    for i in range(6):
        binary = [("b", float(Func.ADD)), ("const", float(i + 1))] + binary
    return pack(
        [list(reversed(unary)), binary], row_len=24, input_len=3, device=device
    )


# ---------------------------------------------------------------------------
# stage: numpy public-API parity
# ---------------------------------------------------------------------------


def _parity_case(
    forest: Forest,
    label: str,
    report: Report,
    *,
    tree_indices: Sequence[int] = (0, 1),
) -> None:
    pop, input_len = forest.pop_size, forest.input_len
    torch.manual_seed(1234)
    x_forward = torch.randn(pop, input_len, device=_DEVICE)
    x_batch = torch.randn(5, input_len, device=_DEVICE) * 1.5
    inputs = torch.randn(7, input_len, device=_DEVICE)
    labels = torch.randn(7, 1, device=_DEVICE)

    set_mode("specialized")
    ref_loss_mse = forest.SR_fitness(inputs, labels, True, "auto")
    ref_loss_mae = forest.SR_fitness(inputs, labels, False, "auto")
    ref_fwd = forest.forward(x_forward)
    ref_batch = forest.batch_forward(x_batch)

    set_mode("dense")
    got_loss_mse = forest.SR_fitness(inputs, labels, True, "auto")
    got_loss_mae = forest.SR_fitness(inputs, labels, False, "auto")
    got_fwd = forest.forward(x_forward)
    got_batch = forest.batch_forward(x_batch)

    compare(f"{label}: Forest.SR_fitness(MSE)", ref_loss_mse, got_loss_mse, report)
    compare(f"{label}: Forest.SR_fitness(MAE)", ref_loss_mae, got_loss_mae, report)
    compare(f"{label}: Forest.forward", ref_fwd, got_fwd, report)
    compare(f"{label}: Forest.batch_forward", ref_batch, got_batch, report)
    report.add(
        f"{label}: result shapes",
        tuple(got_loss_mse.shape) == (pop,)
        and tuple(got_fwd.shape) == (pop, 1)
        and tuple(got_batch.shape) == (pop, 5, 1),
        f"loss={tuple(got_loss_mse.shape)} forward={tuple(got_fwd.shape)} "
        f"batch={tuple(got_batch.shape)}",
    )
    check_contract(f"{label}: Forest.SR_fitness contract", got_loss_mse, report)
    check_contract(f"{label}: Forest.forward contract", got_fwd, report)

    for index in tree_indices:
        if index >= pop:
            continue
        tree = forest[index]
        point = torch.randn(input_len, device=_DEVICE)
        rows = torch.randn(4, input_len, device=_DEVICE)

        set_mode("specialized")
        ref_1d = tree.forward(point)
        ref_2d = tree.forward(rows)
        ref_fit = tree.SR_fitness(inputs, labels, True, "auto")

        set_mode("dense")
        got_1d = tree.forward(point)
        got_2d = tree.forward(rows)
        got_fit = tree.SR_fitness(inputs, labels, True, "auto")

        compare(f"{label}: Tree[{index}].forward(1-D)", ref_1d, got_1d, report)
        compare(f"{label}: Tree[{index}].forward(2-D)", ref_2d, got_2d, report)
        compare(f"{label}: Tree[{index}].SR_fitness", ref_fit, got_fit, report)
        report.add(
            f"{label}: Tree[{index}] shapes",
            tuple(got_1d.shape) == (1,)
            and tuple(got_2d.shape) == (4, 1)
            and tuple(got_fit.shape) == (1,),
            f"1-D={tuple(got_1d.shape)} 2-D={tuple(got_2d.shape)} "
            f"fit={tuple(got_fit.shape)}",
        )
        check_contract(
            f"{label}: Tree[{index}].forward(1-D) contract",
            got_1d,
            report,
            require_leaf=False,  # Tree.forward returns res[0], a view by design
        )
        check_contract(f"{label}: Tree[{index}].SR_fitness contract", got_fit, report)
        check_writable(
            f"{label}: Tree[{index}] SR loss is fresh and writable",
            got_fit,
            lambda tree=tree: tree.SR_fitness(inputs, labels, True, "auto"),
            report,
        )

    # Forest.forward on a 1-row (per_row) input, i.e. the single-row dense shape
    forest_view = forest[tree_indices[0] : tree_indices[0] + 1]
    one_row = torch.randn(1, input_len, device=_DEVICE)
    set_mode("specialized")
    ref_single = forest_view.forward(one_row)
    set_mode("dense")
    got_single = forest_view.forward(one_row)
    compare(f"{label}: Forest.forward(pop_size=1)", ref_single, got_single, report)

    check_writable(
        f"{label}: Forest SR loss is fresh and writable",
        got_loss_mse,
        lambda: forest.SR_fitness(inputs, labels, True, "auto"),
        report,
    )


def stage_numpy(args, report: Report) -> None:
    print("== stage numpy: public-API parity (dense forced vs specialized forced) ==")
    populations = [
        (
            "random seed=0",
            random_population(0, 8, 16, 3, 0.3, FUNCS_NAMES, 3),
            (0, 3),
        ),
        (
            "const-heavy seed=7",
            random_population(
                7,
                6,
                32,
                2,
                0.8,
                ["+", "-", "*", "/", "sin", "exp", "if", ">"],
                3,
            ),
            (0, 2),
        ),
        ("deep chains", deep_population(), (0, 1)),
        ("hand-built (const-only tree)", hand_built_population(), (0, 1, 2)),
    ]
    for label, forest, tree_indices in populations:
        _parity_case(forest, label, report, tree_indices=tree_indices)

    # -- the `auto` policy on numpy must stay on the specialized path ----------
    print("-- auto policy check (numpy) --")
    forest = random_population(21, 12, 16, 3, 0.4, FUNCS_NAMES, 3)
    inputs = torch.randn(6, forest.input_len, device=_DEVICE)
    labels = torch.randn(6, 1, device=_DEVICE)

    set_mode("auto", backend="numpy")
    E.clear_cache()
    before = E.cache_stats()
    auto_loss = forest.SR_fitness(inputs, labels, True, "auto")
    auto_stats = E.cache_stats()
    auto_builds = auto_stats["misses"] - before["misses"]

    set_mode("dense", backend="numpy")
    E.clear_cache()
    before = E.cache_stats()
    dense_loss = forest.SR_fitness(inputs, labels, True, "auto")
    dense_stats = E.cache_stats()
    dense_builds = dense_stats["misses"] - before["misses"]

    set_mode("specialized", backend="numpy")
    ref_loss = forest.SR_fitness(inputs, labels, True, "auto")

    report.add(
        "auto+numpy stays specialized (per-tree builds)",
        auto_builds > 1,
        f"{auto_builds} cache misses for pop_size={forest.pop_size}",
    )
    report.add(
        "forced dense+numpy does ONE build for the population",
        dense_builds == 1,
        f"{dense_builds} cache misses",
    )
    compare("auto+numpy loss == specialized loss", ref_loss, auto_loss, report)
    compare("forced dense loss == specialized loss", ref_loss, dense_loss, report)
    set_mode("auto", backend="numpy")  # the ambient state this check asserts on
    report.add(
        "should_use_dense() policy signals",
        D.should_use_dense(forest.pop_size) is False
        and D.should_use_dense(forest.pop_size, eval_mode="dense") is True
        and D.should_use_dense(forest.pop_size, eval_mode="specialized") is False
        and D.should_use_dense(16, eval_mode="auto", backend="iree") is True
        and D.should_use_dense(15, eval_mode="auto", backend="iree") is False
        and D.should_use_dense(10**4, eval_mode="auto", backend="numpy") is False,
        "auto+numpy=False (even at P=1e4), dense=True, specialized=False, "
        "auto+iree: P=16 -> True, P=15 -> False",
    )

    # -- the combined multi-output container forwards through Forest ------------
    print("-- combined multi-output container (CombinedForest -> Forest) --")
    combined = CombinedForest(
        [
            random_population(31, 4, 16, 2, 0.4, FUNCS_NAMES, 3),
            random_population(32, 4, 16, 1, 0.4, FUNCS_NAMES, 3),
        ],
        {"A": ["x0", "x1"], "B": ["x1"]},
    )
    torch.manual_seed(5150)
    batch = {
        "x0": torch.randn(3, device=_DEVICE),
        "x1": torch.randn(3, device=_DEVICE),
    }
    point = {"x0": 0.5, "x1": -0.25}
    set_mode("specialized")
    ref_batch = combined.batch_forward(batch)
    ref_point = combined.forward(point)
    set_mode("dense")
    got_batch = combined.batch_forward(batch)
    got_point = combined.forward(point)
    for name in ("A", "B"):
        compare(f"combined: {name} batch_forward", ref_batch[name], got_batch[name], report)
        compare(f"combined: {name} forward", ref_point[name], got_point[name], report)

    set_mode("specialized")


# ---------------------------------------------------------------------------
# stage: forced-dense fallback on a multi-output population
# ---------------------------------------------------------------------------


def _pack_out_node(function: int, out_index: int) -> float:
    """The engine's out-node packing: float32 bits of (out_index << 16) | function."""
    bits = np.int32((int(out_index) << 16) | (int(function) & 0xFF))
    return float(bits.view(np.float32))


def multi_output_forest(device: str = _DEVICE) -> Forest:
    """Two trees, ``output_len=2``, one OUT_NODE-tagged function node each.

    tree 0: ``out[0] += x0 + 2.0``  (node 0 tagged with outIndex 0)
    tree 1: ``out[1] += x0 * 3.0``  (node 0 tagged with outIndex 1)
    """
    row_len, input_len = 4, 1
    values = np.zeros((2, row_len), dtype=np.float32)
    types = np.zeros((2, row_len), dtype=np.int16)
    sizes = np.zeros((2, row_len), dtype=np.int16)
    for p, (function, out_index, const) in enumerate(
        ((Func.ADD, 0, 2.0), (Func.MUL, 1, 3.0))
    ):
        # node 0: the tagged function node (its own value is accumulated into the
        # output its outIndex names, so tree p produces out[out_index] = x0 <op> const)
        values[p, 0] = _pack_out_node(function, out_index)
        types[p, 0] = np.int16(int(NType.BFUNC) | int(NType.OUT_NODE))
        sizes[p, 0] = 3
        values[p, 1], types[p, 1], sizes[p, 1] = 0.0, int(NType.VAR), 1
        values[p, 2], types[p, 2], sizes[p, 2] = const, int(NType.CONST), 1
    return Forest(
        input_len,
        2,
        torch.from_numpy(values).to(device),
        torch.from_numpy(types).to(device),
        torch.from_numpy(sizes).to(device),
    )


def stage_fallback(args, report: Report) -> None:
    print("== stage fallback: output_len=2 (OUT_NODE) must fall back cleanly ==")
    forest = multi_output_forest()
    x = torch.tensor([[2.0], [5.0]], dtype=torch.float32, device=_DEVICE)
    inputs = torch.tensor([[2.0], [5.0]], dtype=torch.float32, device=_DEVICE)
    labels = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32, device=_DEVICE)

    # the dense evaluator must refuse this population (that IS the fallback signal)
    refused = False
    try:
        D.evaluate_dense(
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
            forest.input_len,
            forest.output_len,
            inputs,
            per_row=False,
        )
    except D.DenseUnsupported as exc:
        refused = True
        reason = str(exc.reason)
    report.add("dense refuses the multi-output population", refused, reason)

    set_mode("specialized")
    ref_fwd = forest.forward(x)
    ref_batch = forest.batch_forward(inputs)
    ref_fit_mse = forest.SR_fitness(inputs, labels, True, "auto")
    ref_fit_mae = forest.SR_fitness(inputs, labels, False, "auto")

    set_mode("dense")  # forced dense -> attempts, refuses, falls back
    got_fwd = forest.forward(x)
    got_batch = forest.batch_forward(inputs)
    got_fit_mse = forest.SR_fitness(inputs, labels, True, "auto")
    got_fit_mae = forest.SR_fitness(inputs, labels, False, "auto")

    compare("fallback: Forest.forward", ref_fwd, got_fwd, report)
    compare("fallback: Forest.batch_forward", ref_batch, got_batch, report)
    compare("fallback: Forest.SR_fitness(MSE)", ref_fit_mse, got_fit_mse, report)
    compare("fallback: Forest.SR_fitness(MAE)", ref_fit_mae, got_fit_mae, report)

    # analytic out-node values: tree 0 sums x0+2.0 into out[0], tree 1 x0*3 into out[1]
    want_forward = torch.tensor([[4.0, 0.0], [0.0, 15.0]], dtype=torch.float32)
    want_batch = torch.tensor(
        [
            [[4.0, 0.0], [7.0, 0.0]],
            [[0.0, 6.0], [0.0, 15.0]],
        ],
        dtype=torch.float32,
    )
    compare("fallback: out-node values (forward)", want_forward, got_fwd.cpu(), report)
    compare(
        "fallback: out-node values (batch_forward)", want_batch, got_batch.cpu(), report
    )

    # the loss is the multi-output MSE/MAE of those predictions vs the labels
    pred = got_batch[:, :, :]  # (2 trees, 2 points, 2 outputs)
    diff = labels.cpu()[None, :, :] - pred.cpu()
    want_mse = (diff * diff).sum(dim=(1, 2)) / 2.0
    want_mae = diff.abs().sum(dim=(1, 2)) / 2.0
    compare("fallback: SR MSE formula", want_mse, got_fit_mse.cpu(), report)
    compare("fallback: SR MAE formula", want_mae, got_fit_mae.cpu(), report)
    check_contract("fallback: SR_fitness contract", got_fit_mse, report)
    set_mode("specialized")


# ---------------------------------------------------------------------------
# stage: iree smoke (one dense compile for the whole population)
# ---------------------------------------------------------------------------


def stage_iree(args, report: Report) -> None:
    print("== stage iree: auto policy on the CUDA HAL (P=32, N=32) ==")
    pop, input_len, num_data = 32, 3, 32
    forest = random_population(99, pop, 16, input_len, 0.5, FUNCS_NAMES, 3)
    torch.manual_seed(4321)
    inputs = torch.randn(num_data, input_len, device=_DEVICE)
    labels = torch.randn(num_data, 1, device=_DEVICE)

    # reference: numpy backend, specialized path
    set_mode("specialized", backend="numpy")
    ref = forest.SR_fitness(inputs, labels, True, "auto")

    # the call under test: auto + iree through the standard public API
    set_mode("auto", backend="iree")
    if args.gpu is not None:
        os.environ["EVOGP_ETL_IREE_DEVICE"] = str(args.gpu)
    policy = D.should_use_dense(pop)
    report.add(
        "auto+iree selects dense for P=32",
        policy is True,
        f"backend={D.resolve_backend_name()} should_use_dense={policy}",
    )

    E.clear_cache()
    before = E.cache_stats()
    t0 = time.perf_counter()
    got = forest.SR_fitness(inputs, labels, True, "auto")
    cold_ms = (time.perf_counter() - t0) * 1e3
    after_cold = E.cache_stats()
    t0 = time.perf_counter()
    got_warm = forest.SR_fitness(inputs, labels, True, "auto")
    warm_ms = (time.perf_counter() - t0) * 1e3
    after_warm = E.cache_stats()

    misses = after_cold["misses"] - before["misses"]
    hits = after_cold["hits"] - before["hits"]
    report.add(
        "dense performed ONE compile for the whole population",
        misses == 1,
        f"cold: misses={misses} hits={hits} entries={after_cold['entries']} "
        f"(pop_size={pop})",
    )
    report.add(
        "second call is a pure cache hit",
        after_warm["hits"] - after_cold["hits"] == 1
        and after_warm["misses"] == after_cold["misses"],
        f"warm: hits={after_warm['hits'] - after_cold['hits']} "
        f"misses={after_warm['misses'] - after_cold['misses']}",
    )
    print(
        f"[time] iree cold (compile+run) {cold_ms:.0f} ms, warm run {warm_ms:.0f} ms "
        f"(P={pop}, N={num_data}, policy={policy})"
    )
    compare("iree dense vs numpy specialized (cold)", ref, got, report)
    compare("iree dense vs numpy specialized (warm)", ref, got_warm, report)
    check_contract("iree: SR_fitness contract", got, report)
    report.add(
        "iree values are the same on both calls",
        bool(torch.allclose(got.cpu().float(), got_warm.cpu().float(), rtol=1e-6, atol=1e-6, equal_nan=True)),
        "cold vs warm",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        default="numpy,fallback",
        help=f"comma-separated subset of {list(_STAGES)} (default numpy,fallback)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="visible IREE device index (default 0)"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stages = [s.strip().lower() for s in args.stage.split(",") if s.strip()]
    unknown = [s for s in stages if s not in _STAGES]
    if unknown:
        print(f"unknown stage(s) {unknown}: pick from {list(_STAGES)}")
        return 2
    if os.environ.get("ETL_DISABLE_LOCATIONS", "").strip() not in ("1", "true"):
        print("[warn] ETL_DISABLE_LOCATIONS=1 is not set: tracing will be ~4x slower")
    print(
        f"== dense integration check: stages={stages} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} =="
    )
    report = Report()
    try:
        for stage in stages:
            if stage == "numpy":
                stage_numpy(args, report)
            elif stage == "fallback":
                stage_fallback(args, report)
            elif stage == "iree":
                stage_iree(args, report)
    except Exception as exc:  # hard error (not a parity mismatch)
        import traceback

        traceback.print_exc()
        print(f"\n== HARD ERROR: {exc!r} ==")
        return 2
    passed = sum(1 for entry in report.entries if entry["ok"])
    print(
        f"\n== {'OK' if report.ok else 'FAILED'}: {passed}/{len(report.entries)} "
        "checks passed =="
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
