"""Plain-PyTorch baselines: "the old torch version" of the same tree evaluation.

Two arms, both on GPU tensors, both with ``torch.no_grad()``, neither using
evoxc and none jitted:

``PerTreeEvaluator`` -- the classic pre-jit route.  Every tree is decoded once
into a flat prefix program and then *walked in Python*: one cluster of torch ops
per node, over the ``(N,)`` batch of data points, one tree at a time.  The per
tree output is a ``(N,)`` tensor; the population result is a ``torch.stack`` of
them.  This is what "evaluate each tree with torch" honestly costs: O(pop x
nodes) kernel launches of O(N) work.

``BatchedEvaluator`` -- the best a pure-torch implementation can do.  The whole
population is one vectorized pass: every node position of the prefix-encoded
forest becomes a ``(P, N)`` value in a position-indexed buffer, children are
fetched by index (no per-tree Python loop), and the node kind / function id is
resolved with per-(kind, function) boolean masks and ``torch.where``.  Only the
function candidates that actually occur in the population are computed, so the
per-call cost is ``L`` positions x (few ops) regardless of ``pop_size``.

Both implement the documented engine semantics node-for-node:

* prefix (pre-order) rows, only ``[0, subtree_size[0])`` is meaningful;
* ``node_type`` base ``0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC``, anything else is a
  ternary ``IF`` (the kernel's if/else chain, single-output mode);
* node values are stored back as float32 per node, operand order is left-most
  first, a function id outside the node's arity yields ``0.0``;
* ``DIV`` masks a zero divisor with NaN, ``LOOSE_DIV``/``LOOSE_INV`` clamp
  ``|b|`` to ``DELTA`` (copysign), ``MAX/MIN`` select (NaN goes right),
  comparisons yield +-1.0, ``IF`` selects on ``cond > 0``.

The source of truth for these rules is ``src/evogp/tests/reference_eval.py``
(the repo's independent oracle); ``test/benchmark/evoxc/validate.py`` diffs both
arms against it and against each other.  Both arms run in plain float32 (as a
pre-jit torch implementation would): with the campaign's arithmetic-only
function set the float32 ops are correctly rounded, so they agree with the
oracle bit-for-bit on ``+ - * /``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ids (mirrors of evogp.tree.utils.NType / Func; kept local so the baseline
# depends on no engine internals)
# ---------------------------------------------------------------------------
VAR, CONST, UFUNC, BFUNC, TFUNC = 0, 1, 2, 3, 4
(
    IF,
    ADD,
    SUB,
    MUL,
    DIV,
    LOOSE_DIV,
    POW,
    LOOSE_POW,
    MAX,
    MIN,
    LT,
    GT,
    LE,
    GE,
    SIN,
    COS,
    TAN,
    SINH,
    COSH,
    TANH,
    LOG,
    LOOSE_LOG,
    EXP,
    INV,
    LOOSE_INV,
    NEG,
    ABS,
    SQRT,
    LOOSE_SQRT,
) = range(29)
DELTA = 1e-9
MAX_VAL = 1e9

_TFUNC_END = GE  # ids <= GE take two operands, the rest take one


def arity_of(function_id: int) -> int:
    if function_id == IF:
        return 3
    if function_id <= _TFUNC_END:
        return 2
    return 1


def kind_arity(kind: int) -> int:
    if kind == TFUNC:
        return 3
    if kind == BFUNC:
        return 2
    if kind == UFUNC:
        return 1
    return 0


def kind_from_type(node_type: int) -> int:
    """The kernel's if/else chain: 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, else TFUNC."""
    if node_type == VAR:
        return VAR
    if node_type == CONST:
        return CONST
    if node_type == UFUNC:
        return UFUNC
    if node_type == BFUNC:
        return BFUNC
    return TFUNC


def _scalar(value: float, device) -> torch.Tensor:
    """A 0-dim float32 tensor -- broadcasts into ``(N,)``/``(P, N)`` results."""
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _cached(constants: Dict[float, torch.Tensor], value: float, device) -> torch.Tensor:
    key = float(value)
    tensor = constants.get(key)
    if tensor is None:
        tensor = constants[key] = _scalar(key, device)
    return tensor


# ---------------------------------------------------------------------------
# per-op semantics (float32, plain torch)
# ---------------------------------------------------------------------------
def _unary(fid: int, a: torch.Tensor, nan: torch.Tensor, device) -> torch.Tensor:
    if fid == SIN:
        return torch.sin(a)
    if fid == COS:
        return torch.cos(a)
    if fid == TAN:
        return torch.tan(a)
    if fid == SINH:
        return torch.sinh(a)
    if fid == COSH:
        return torch.cosh(a)
    if fid == TANH:
        return torch.tanh(a)
    if fid == LOG:
        return torch.log(a)
    if fid == LOOSE_LOG:
        return torch.where(a == 0.0, _scalar(-MAX_VAL, device), torch.log(torch.abs(a)))
    if fid == EXP:
        return torch.exp(a)
    if fid == INV:
        return torch.where(a == 0.0, nan, torch.reciprocal(a))
    if fid == LOOSE_INV:
        b = torch.where(torch.abs(a) <= DELTA, torch.copysign(_scalar(DELTA, device), a), a)
        return torch.reciprocal(b)
    if fid == NEG:
        return -a
    if fid == ABS:
        return torch.abs(a)
    if fid == SQRT:
        return torch.sqrt(a)
    if fid == LOOSE_SQRT:
        return torch.sqrt(torch.where(a <= 0.0, torch.abs(a), a))
    return torch.zeros_like(a)


def _binary(fid: int, a: torch.Tensor, b: torch.Tensor, nan: torch.Tensor, device) -> torch.Tensor:
    if fid == ADD:
        return a + b
    if fid == SUB:
        return a - b
    if fid == MUL:
        return a * b
    if fid == DIV:
        return torch.where(b == 0.0, nan, a / b)
    if fid == LOOSE_DIV:
        b2 = torch.where(torch.abs(b) <= DELTA, torch.copysign(_scalar(DELTA, device), b), b)
        return a / b2
    if fid == POW:
        return torch.pow(a, b)
    if fid == LOOSE_POW:
        return torch.where(
            (a == 0.0) & (b == 0.0), _scalar(0.0, device), torch.pow(torch.abs(a), b)
        )
    if fid == MAX:
        return torch.where(a >= b, a, b)
    if fid == MIN:
        return torch.where(a <= b, a, b)
    if fid == LT:
        return torch.where(a < b, _scalar(1.0, device), _scalar(-1.0, device))
    if fid == GT:
        return torch.where(a > b, _scalar(1.0, device), _scalar(-1.0, device))
    if fid == LE:
        return torch.where(a <= b, _scalar(1.0, device), _scalar(-1.0, device))
    if fid == GE:
        return torch.where(a >= b, _scalar(1.0, device), _scalar(-1.0, device))
    return torch.zeros_like(a)


# ---------------------------------------------------------------------------
# host-side decoding (setup only; never inside a timed region)
# ---------------------------------------------------------------------------
def _host_rows(forest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = forest.batch_node_value.detach().to("cpu", torch.float32).numpy()
    types = forest.batch_node_type.detach().to("cpu", torch.int16).numpy().astype(np.int32)
    sizes = forest.batch_subtree_size.detach().to("cpu", torch.int16).numpy().astype(np.int32)
    return np.ascontiguousarray(values), np.ascontiguousarray(types), np.ascontiguousarray(sizes)


def _child(buffer: torch.Tensor, index: Any, rows: torch.Tensor) -> torch.Tensor:
    """``(pop, n)`` child values: a column slice when uniform, else a per-row gather."""
    if isinstance(index, int):
        return buffer[:, index]
    return buffer[rows, index]


def _index(values, device) -> Any:
    """Per-row child index: a plain int when the whole population agrees, else int64 tensor."""
    arr = np.asarray(values, dtype=np.int64)
    if int(arr.min()) == int(arr.max()):
        return int(arr[0])
    return torch.as_tensor(arr, device=device, dtype=torch.int64)
def _safe_int(value: float) -> int:
    if not np.isfinite(value):
        return 0
    return int(np.trunc(float(value)))


# ---------------------------------------------------------------------------
# arm 1: one torch op cluster per node, one tree at a time
# ---------------------------------------------------------------------------
class PerTreeEvaluator:
    """The classic pre-jit route: Python tree walk + torch ops per node."""

    def __init__(self, forest, device: str = "cuda"):
        self.device = torch.device(device)
        values, types, sizes = _host_rows(forest)
        self.pop_size, self.max_tree_len = values.shape
        self.output_len = forest.output_len
        self.input_len = forest.input_len
        self._constants: Dict[float, torch.Tensor] = {}
        self._nan = _scalar(float("nan"), self.device)
        self.programs: List[List[Tuple[int, float]]] = []
        for row in range(self.pop_size):
            length = int(sizes[row, 0])
            program: List[Tuple[int, float]] = []
            for i in range(length):
                kind = kind_from_type(int(types[row, i]))
                program.append((kind, float(values[row, i])))
            self.programs.append(program)
        self.setup_ms: Optional[float] = None

    # -- one tree -----------------------------------------------------------
    def _eval_tree(self, program: Sequence[Tuple[int, float]], x: torch.Tensor) -> torch.Tensor:
        stack: List[torch.Tensor] = []
        for kind, value in reversed(program):
            if kind == CONST:
                stack.append(_cached(self._constants, value, self.device))
                continue
            if kind == VAR:
                index = _safe_int(value)
                if index < 0 or index >= x.shape[1]:
                    raise IndexError(f"VAR index {index} out of range for input_len={x.shape[1]}")
                stack.append(x[:, index])
                continue
            if kind == UFUNC:
                a = stack.pop()
                stack.append(_unary(_safe_int(value), a, self._nan, self.device))
                continue
            if kind == BFUNC:
                # the reversed prefix walk pushes the RIGHT-most child first, so
                # the top of the stack is the LEFT-most operand
                a = stack.pop()
                b = stack.pop()
                stack.append(_binary(_safe_int(value), a, b, self._nan, self.device))
                continue
            cond = stack.pop()
            then_value = stack.pop()
            else_value = stack.pop()
            stack.append(torch.where(cond > 0.0, then_value, else_value))
        if len(stack) != 1:
            raise ValueError(f"malformed tree: {len(stack)} values left on the stack")
        value = stack[0]
        # a leaf-only tree (or a constant folded subtree) yields a 0-dim scalar;
        # the kernel still produces one value per data point
        if value.dim() == 0:
            value = value.expand(x.shape[0])
        return value

    # -- workloads ----------------------------------------------------------
    def batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, input_len)`` -> ``(pop, N, out_len)`` (every tree sees all rows)."""
        with torch.no_grad():
            per_tree = [self._eval_tree(program, x) for program in self.programs]
            if self.output_len == 1:
                return torch.stack(per_tree, dim=0).unsqueeze(-1)
            raise NotImplementedError("the baseline arms cover single-output forests only")

    def forward(self, x_per_row: torch.Tensor) -> torch.Tensor:
        """``(pop, input_len)`` -> ``(pop, out_len)`` (tree ``i`` sees row ``i``)."""
        with torch.no_grad():
            per_tree = [
                self._eval_tree(program, x_per_row[i : i + 1])
                for i, program in enumerate(self.programs)
            ]
            return torch.stack(per_tree, dim=0)

    def sr_fitness(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Positive MSE loss per tree, ``(pop,)`` -- the engine's convention."""
        with torch.no_grad():
            target = labels[:, 0]
            losses = [
                ((self._eval_tree(program, x) - target) ** 2).sum() / x.shape[0]
                for program in self.programs
            ]
            return torch.stack(losses, dim=0)


# ---------------------------------------------------------------------------
# arm 2: the whole population in one vectorized torch pass
# ---------------------------------------------------------------------------
class BatchedEvaluator:
    """Best-effort pure-torch batched evaluator (no Python loop over trees)."""

    def __init__(self, forest, device: str = "cuda"):
        self.device = torch.device(device)
        values, types, sizes = _host_rows(forest)
        self.pop_size, self.max_tree_len = values.shape
        self.output_len = forest.output_len
        self.input_len = forest.input_len
        self._constants: Dict[float, torch.Tensor] = {}
        self._nan = _scalar(float("nan"), self.device)

        pop, length = self.pop_size, self.max_tree_len
        self.kind = np.vectorize(kind_from_type, otypes=[np.int32])(types)
        self.active = np.arange(length)[None, :] < sizes[:, 0][:, None]
        # prefix-order child positions, PER ROW: c1 = i+1, c2 = c1 + size[c1],
        # c3 = c2 + size[c2].  ``length`` is a dummy slot (zeros) so the padded
        # children of a leaf on the last position never index a live node.
        sizes_p = np.concatenate([sizes, np.zeros((pop, 1), dtype=np.int64)], axis=1)
        rows = np.arange(pop)
        self.child1 = np.minimum(np.arange(1, length + 1), length)
        self.child2: List[Any] = []
        self.child3: List[Any] = []
        for i in range(length):
            v = int(self.child1[i])
            s1 = sizes_p[:, v] if v < length else np.zeros(pop, dtype=np.int64)
            c2 = np.minimum(v + s1, length)
            s2 = sizes_p[rows, np.minimum(c2, length)]
            c3 = np.minimum(c2 + s2, length)
            self.child2.append(_index(c2, self.device))
            self.child3.append(_index(c3, self.device))
        # per-position plan (setup only): leaf columns + the (kind, function) cases
        # that actually occur at that position
        self.plan: List[Dict[str, Any]] = []
        for i in range(length):
            kinds = self.kind[:, i]
            live = self.active[:, i]
            step: Dict[str, Any] = {"index": i, "cases": []}
            if np.any(live & (kinds == VAR)):
                step["var_mask"] = torch.as_tensor(live & (kinds == VAR), device=self.device).view(pop, 1)
                step["var_index"] = torch.as_tensor(
                    np.array([_safe_int(v) for v in values[:, i]]), device=self.device
                ).clamp_(0, self.input_len - 1)
            if np.any(live & (kinds == CONST)):
                step["const_mask"] = torch.as_tensor(
                    live & (kinds == CONST), device=self.device
                ).view(pop, 1)
                step["const_value"] = torch.as_tensor(
                    np.asarray(values[:, i], dtype=np.float32), device=self.device
                ).view(pop, 1)
            for kind in (UFUNC, BFUNC, TFUNC):
                base = live & (kinds == kind)
                if not np.any(base):
                    continue
                ids = np.array([_safe_int(v) for v in values[:, i]])
                for fid in sorted({int(v) for v in ids[base]}):
                    valid = base & (ids == fid) & (arity_of(fid) == kind_arity(kind))
                    if not np.any(valid):
                        continue
                    step["cases"].append(
                        {
                            "kind": kind,
                            "fid": fid,
                            "mask": torch.as_tensor(valid, device=self.device).view(pop, 1),
                        }
                    )
            self.plan.append(step)
        self.setup_ms: Optional[float] = None

    # -- the one pass -------------------------------------------------------
    def _evaluate(self, get_var, n: int) -> torch.Tensor:
        """Run the population; ``get_var(i)`` returns the ``(pop, n)`` var column."""
        pop, length = self.pop_size, self.max_tree_len
        rows = torch.arange(pop, device=self.device)
        # one stacked buffer (pop, length+1, n): buffer[:, i] is node i, dropping a
        # per-row index tensor into dim 1 gathers each tree's own child value
        buffer = torch.zeros((pop, length + 1, n), dtype=torch.float32, device=self.device)
        for step in reversed(self.plan):
            index = step["index"]
            value = torch.zeros((pop, n), dtype=torch.float32, device=self.device)
            if "const_mask" in step:
                value = torch.where(step["const_mask"], step["const_value"], value)
            if "var_mask" in step:
                value = torch.where(step["var_mask"], get_var(index), value)
            c1 = int(self.child1[index])
            for case in step["cases"]:
                kind, fid = case["kind"], case["fid"]
                if kind == UFUNC:
                    candidate = _unary(fid, buffer[:, c1], self._nan, self.device)
                elif kind == BFUNC:
                    candidate = _binary(
                        fid,
                        buffer[:, c1],
                        _child(buffer, self.child2[index], rows),
                        self._nan,
                        self.device,
                    )
                else:
                    candidate = torch.where(
                        buffer[:, c1] > 0.0,
                        _child(buffer, self.child2[index], rows),
                        _child(buffer, self.child3[index], rows),
                    )
                value = torch.where(case["mask"], candidate, value)
            buffer[:, index] = value
        return buffer[:, 0]

    # -- workloads ----------------------------------------------------------
    def batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, input_len)`` -> ``(pop, N, out_len)``."""
        rows = x.shape[0]

        def get_var(i: int) -> torch.Tensor:
            return torch.index_select(x, 1, self.plan[i]["var_index"]).transpose(0, 1)

        with torch.no_grad():
            return self._evaluate(get_var, rows).unsqueeze(-1)

    def forward(self, x_per_row: torch.Tensor) -> torch.Tensor:
        """``(pop, input_len)`` -> ``(pop, out_len)``."""

        def get_var(i: int) -> torch.Tensor:
            return x_per_row.gather(1, self.plan[i]["var_index"].view(-1, 1))

        with torch.no_grad():
            return self._evaluate(get_var, 1)

    def sr_fitness(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.batch_forward(x)
        with torch.no_grad():
            diff = labels.unsqueeze(0) - pred
            return (diff * diff).sum(dim=(1, 2)) / x.shape[0]


def make_torch_arm(impl: str, forest) -> Any:
    if impl == "per_tree":
        return PerTreeEvaluator(forest)
    if impl == "batched":
        return BatchedEvaluator(forest)
    raise ValueError(f"unknown torch arm implementation {impl!r}")
