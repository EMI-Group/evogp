"""Plain-torch baselines for the measurement harnesses: a self-contained evaluator.

This is measurement scaffolding, not library surface: the two harnesses that
report a plain-torch column (:mod:`evogp.gp.measure_popscale` for the per-call
axis) import it, and nothing else does.  It holds the two idioms a pure-torch
implementation of the engine amounts to -- the WHOLE-POPULATION pass
(:class:`TorchBatchedEvaluator`) and the per-tree python walk
(:func:`per_tree_forward`) -- over the legacy triple
(:func:`evogp.gp.encoding.to_legacy_triple`) alone.

There is deliberately NO torch import here (the module takes the ``torch``
module as an argument, as the rest of this subtree resolves the boundary at call
time), no ``evogp.tree`` and no ``Forest``: the tensors are built from the
population's own numpy rows, so the baselines survive the removal of the
torch-native stack in 2.0.0a2.  The torch comparison code that was built on that
stack was removed in 2.0.0a2; recover it at commit 6720723.

Semantics are the ones the deleted engine documented (its ``ENGINE_SPEC.md``,
kept in git history; recover it at commit 6720723): ``DIV`` masks a zero
divisor with NaN,
``MAX``/``MIN`` compare-select, comparisons yield +-1, ``IF`` selects on
``cond > 0``, the ``LOOSE_*`` family clamps as the spec says.  A single-output
population is what the harnesses generate, so an ``output_len != 1`` triple is
refused rather than mis-measured.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TorchBatchedEvaluator",
    "per_tree_forward",
    "torch_ops",
]

#: Arity of a node KIND (``VAR``/``CONST`` 0, ``UFUNC`` 1, ``BFUNC`` 2, ternary 3).
_KIND_ARITY = {2: 1, 3: 2, 4: 3}

#: ``LOOSE_LOG(0)`` returns -MAX_VAL and ``LOOSE_DIV``/``LOOSE_INV`` clamp to DELTA.
_MAX_VAL = 1e9
_DELTA = 1e-9


def _safe_int(value: float) -> int:
    """A node payload as an int (NaN/``inf`` read as 0, like the engine's cast)."""
    return 0 if not np.isfinite(value) else int(np.trunc(float(value)))


def _kind_from_type(node_type: int) -> int:
    """The engine's if/else chain: 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, else ternary."""
    if node_type in (0, 1, 2, 3):
        return int(node_type)
    return 4


def _arity_of(function_id: int) -> int:
    if function_id == 0:  # IF
        return 3
    if function_id <= 12:  # LT .. GE
        return 2
    return 1


def _kind_arity(kind: int) -> int:
    return _KIND_ARITY.get(int(kind), 0)


def torch_ops(torch) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    """``(unary, binary, compare)`` op tables of the plain-torch baseline (by id)."""
    t = torch
    unary = {14: t.sin, 15: t.cos, 16: t.tan, 17: t.sinh, 18: t.cosh, 19: t.tanh,
             20: t.log, 22: t.exp, 25: t.neg, 26: t.abs, 27: t.sqrt}
    binary = {1: t.add, 2: t.sub, 3: t.mul, 6: t.pow}
    compare = {10: t.lt, 11: t.gt, 12: t.le, 13: t.ge}
    return unary, binary, compare


class _Ops:
    """The engine's op semantics in plain float32 torch, shared by both baselines."""

    def __init__(self, torch, device) -> None:
        self._t = torch
        self.device = device
        self._constants: Dict[float, Any] = {}
        self._nan = self._scalar(float("nan"))
        self._unary_ops, self._binary_ops, self._compare_ops = torch_ops(torch)

    # -- setup helpers -------------------------------------------------------
    def _scalar(self, value: float):
        return self._t.tensor(float(value), dtype=self._t.float32, device=self.device)

    def constant(self, value: float):
        key = float(value)
        tensor = self._constants.get(key)
        if tensor is None:
            tensor = self._constants[key] = self._scalar(key)
        return tensor

    # -- the ops -------------------------------------------------------------
    def unary(self, fid: int, a):
        t = self._t
        if fid == 21:  # LOOSE_LOG: log|a| with -MAX_VAL at zero
            return t.where(a == 0.0, self._scalar(-_MAX_VAL), t.log(t.abs(a)))
        if fid == 23:  # INV masks a zero input with NaN
            return t.where(a == 0.0, self._nan, t.reciprocal(a))
        if fid == 24:  # LOOSE_INV clamps |a| to DELTA (copysign)
            clamped = t.where(t.abs(a) <= _DELTA, t.copysign(self._scalar(_DELTA), a), a)
            return t.reciprocal(clamped)
        if fid == 28:  # LOOSE_SQRT takes |a| when a <= 0
            return t.sqrt(t.where(a <= 0.0, t.abs(a), a))
        operator = self._unary_ops.get(int(fid))
        return operator(a) if operator is not None else t.zeros_like(a)

    def binary(self, fid: int, a, b):
        t = self._t
        if fid == 4:  # DIV masks a zero divisor with NaN
            return t.where(b == 0.0, self._nan, a / b)
        if fid == 5:  # LOOSE_DIV clamps |b| to DELTA (copysign)
            clamped = t.where(t.abs(b) <= _DELTA, t.copysign(self._scalar(_DELTA), b), b)
            return a / clamped
        if fid == 7:  # LOOSE_POW of (0, 0) is 0 and the base is |a|
            return t.where((a == 0.0) & (b == 0.0), self._scalar(0.0), t.pow(t.abs(a), b))
        if fid == 8:  # MAX selects (NaN goes right, like the engine)
            return t.where(a >= b, a, b)
        if fid == 9:  # MIN selects
            return t.where(a <= b, a, b)
        compare = self._compare_ops.get(int(fid))
        if compare is not None:  # LT/GT/LE/GE return +-1, never a boolean
            return t.where(compare(a, b), self._scalar(1.0), self._scalar(-1.0))
        operator = self._binary_ops.get(int(fid))
        return operator(a, b) if operator is not None else t.zeros_like(a)

    def ternary(self, a, b, c):
        """``IF(a, b, c)``: ``b`` when ``a > 0``, else ``c``."""
        return self._t.where(a > 0.0, b, c)


class TorchBatchedEvaluator:
    """The plain-torch batched baseline: the whole population in ONE pass.

    Copied idiom (not imported) from the benchmark campaign's ``BatchedEvaluator``,
    which is the best a pure-torch implementation does: every node position of the
    prefix-encoded population becomes a ``(pop, rows)`` value in a
    position-indexed buffer, children are fetched per row by index (no python loop
    over trees), and the node kind / function id is resolved with
    per-``(kind, function)`` masks.  Only the candidates that occur at a position
    are computed, so a call costs ``max_tree_len`` positions x (few ops) whatever
    the population.

    The population is the legacy triple
    (:func:`evogp.gp.encoding.to_legacy_triple`), never a ``Forest``; the torch
    module is passed in (never imported at this module's scope) and the whole
    per-position plan is built in ``__init__``, i.e. setup is never timed.
    """

    def __init__(
        self,
        torch,
        legacy: Sequence[np.ndarray],
        input_len: int,
        output_len: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self._t = torch
        self.device = torch.device(device or "cuda")
        values, types, sizes = legacy
        self.pop_size, self.max_tree_len = values.shape
        self.output_len = int(output_len)
        self.input_len = int(input_len)
        if self.output_len != 1:
            raise NotImplementedError("the plain-torch baselines cover single-output populations")
        self._ops = _Ops(torch, self.device)
        self._nan = self._ops._nan

        pop, length = self.pop_size, self.max_tree_len
        self.kind = np.vectorize(_kind_from_type, otypes=[np.int32])(types)
        self.active = np.arange(length)[None, :] < sizes[:, 0][:, None]
        sizes_p = np.concatenate([sizes, np.zeros((pop, 1), dtype=np.int64)], axis=1)
        rows = np.arange(pop)
        self.child1 = np.minimum(np.arange(1, length + 1), length)
        self.child2: List[Any] = []
        self.child3: List[Any] = []
        for index in range(length):
            first = int(self.child1[index])
            first_size = sizes_p[:, first] if first < length else np.zeros(pop, np.int64)
            second = np.minimum(first + first_size, length)
            second_size = sizes_p[rows, np.minimum(second, length)]
            third = np.minimum(second + second_size, length)
            self.child2.append(self._index(second))
            self.child3.append(self._index(third))
        self.plan: List[Dict[str, Any]] = []
        for index in range(length):
            kinds = self.kind[:, index]
            live = self.active[:, index]
            step: Dict[str, Any] = {"index": index, "cases": []}
            if np.any(live & (kinds == 0)):
                step["var_mask"] = self._mask(live & (kinds == 0))
                step["var_index"] = self._t.as_tensor(
                    np.array([_safe_int(value) for value in values[:, index]]),
                    device=self.device,
                ).clamp_(0, self.input_len - 1)
            if np.any(live & (kinds == 1)):
                step["const_mask"] = self._mask(live & (kinds == 1))
                step["const_value"] = self._t.as_tensor(
                    np.asarray(values[:, index], dtype=np.float32), device=self.device
                ).view(pop, 1)
            for kind in (2, 3, 4):
                base = live & (kinds == kind)
                if not np.any(base):
                    continue
                ids = np.array([_safe_int(value) for value in values[:, index]])
                for fid in sorted({int(value) for value in ids[base]}):
                    valid = base & (ids == fid) & (_arity_of(fid) == _kind_arity(kind))
                    if not np.any(valid):
                        continue
                    step["cases"].append({"kind": kind, "fid": fid,
                                          "mask": self._mask(valid)})
            self.plan.append(step)

    # -- setup helpers -------------------------------------------------------
    def _mask(self, boolean):
        return self._t.as_tensor(boolean, device=self.device).view(self.pop_size, 1)

    def _index(self, values):
        array = np.asarray(values, dtype=np.int64)
        if int(array.min()) == int(array.max()):
            return int(array[0])
        return self._t.as_tensor(array, device=self.device, dtype=self._t.int64)

    def _child(self, buffer, index, rows):
        if isinstance(index, int):
            return buffer[:, index]
        return buffer[rows, index]

    # -- the one pass --------------------------------------------------------
    def _evaluate(self, get_var, n: int):
        pop, length = self.pop_size, self.max_tree_len
        rows = self._t.arange(pop, device=self.device)
        buffer = self._t.zeros(
            (pop, length + 1, n), dtype=self._t.float32, device=self.device
        )
        for step in reversed(self.plan):
            index = step["index"]
            value = self._t.zeros((pop, n), dtype=self._t.float32, device=self.device)
            if "const_mask" in step:
                value = self._t.where(step["const_mask"], step["const_value"], value)
            if "var_mask" in step:
                value = self._t.where(step["var_mask"], get_var(index), value)
            first = int(self.child1[index])
            for case in step["cases"]:
                kind, fid = case["kind"], case["fid"]
                if kind == 2:
                    candidate = self._ops.unary(fid, buffer[:, first])
                elif kind == 3:
                    candidate = self._ops.binary(
                        fid, buffer[:, first],
                        self._child(buffer, self.child2[index], rows),
                    )
                else:
                    candidate = self._ops.ternary(
                        buffer[:, first],
                        self._child(buffer, self.child2[index], rows),
                        self._child(buffer, self.child3[index], rows),
                    )
                value = self._t.where(case["mask"], candidate, value)
            buffer[:, index] = value
        return buffer[:, 0]

    def batch_forward(self, x):
        """``(rows, input_len)`` -> ``(pop, rows)`` (single-output populations)."""

        def get_var(index: int):
            return self._t.index_select(x, 1, self.plan[index]["var_index"]).transpose(0, 1)

        with self._t.no_grad():
            return self._evaluate(get_var, int(x.shape[0]))


def per_tree_forward(
    torch,
    legacy: Sequence[np.ndarray],
    x,
    *,
    input_len: Optional[int] = None,
    device: Optional[str] = None,
):
    """The plain-torch per-tree walk: one recursive evaluation per tree, stacked.

    The honest "evaluate each tree with torch" route: one cluster of ops per node
    per tree, i.e. the shape of the old ``Tree.forward`` call, but driven straight
    off the legacy triple (no ``Forest``).  It costs O(pop) python work per call,
    which is why the harness scopes it to the smaller populations.
    """
    values, types, sizes = legacy
    ops = _Ops(torch, torch.device(device or "cuda"))
    x = x if x.device == ops.device else torch.as_tensor(x, device=ops.device)
    rows = int(x.shape[0])
    width = int(input_len if input_len is not None else x.shape[1])
    predictions = []
    for tree in range(int(values.shape[0])):
        length = int(sizes[tree, 0])
        signal = _tree_signals(
            torch, ops, values[tree], types[tree], sizes[tree], length, x, width
        )
        # a tree whose root is a constant (or a fully constant subtree) is a
        # 0-dim tensor; every prediction is one row vector, so broadcast it
        predictions.append(signal.expand(rows) if signal.dim() == 0
                           else signal.reshape(rows))
    return torch.stack(predictions, dim=0)


def _tree_signals(torch, ops, values, types, sizes, length: int, x, input_len: int):
    """``(rows,)`` value of the root of ONE tree, walked node by node."""

    def node(index: int):
        if index >= length:
            raise ValueError(f"node index {index} is outside the tree ({length} nodes)")
        kind = _kind_from_type(int(types[index]))
        if kind == 0:  # VAR
            column = min(max(_safe_int(float(values[index])), 0), input_len - 1)
            return x[:, column]
        if kind == 1:  # CONST
            return ops.constant(float(values[index]))
        first = index + 1
        if kind == 2:  # UFUNC
            return ops.unary(_safe_int(float(values[index])), node(first))
        second = first + int(sizes[first])
        if kind == 3:  # BFUNC
            return ops.binary(
                _safe_int(float(values[index])), node(first), node(second)
            )
        third = second + int(sizes[second])  # ternary (IF)
        return ops.ternary(node(first), node(second), node(third))

    return node(0)
