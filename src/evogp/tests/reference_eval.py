"""Independent, slow, obviously-correct reference evaluator for EvoGP trees.

This module is the ORACLE used by ``test_equivalence.py`` to verify the ETL-based
("true jit") evaluation engine that replaced the custom
``torch.ops.evogp_cuda.tree_evaluate`` / ``tree_SR_fitness`` CUDA kernels.

Design rules (do not break them):

* **Standalone.** It imports only ``numpy``/``torch`` and nothing from ``evogp``
  (in particular *not* ``evogp.tree.etl_engine`` — the module under test).
* **CPU/plain-torch.** Every op is a vectorized torch op over the ``(N,)`` batch
  of data points; plain torch on any device works, but the default/intended use
  is CPU float32 tensors.
* **Source of truth** is the ORIGINAL CUDA implementation, not its prose:
  ``src/evogp/cuda/defs.h`` (node/function ids, ``DELTA``/``MAX_VAL``),
  ``src/evogp/cuda/kernel.h`` (``OutNodeValue`` packing),
  ``src/evogp/cuda/forward.cu::_process_node`` (per-operator semantics, the
  multi-output "pass the right-most operand up" rule and the out-value
  accumulation), ``forward.cu::_treeGPEvalByStack`` / ``treeGPEvalKernel``
  (prefix decoding, evaluation order, what ends up in the result tensor) and
  ``forward.cu::calculate_fit`` + ``averageFitnessValueKernel`` (SR fitness).

Semantics implemented here (see the docstrings of the individual helpers for
the exact placement of NaN/±inf):

* Encoding: rows are prefix (pre-order), node 0 is the root, only indices
  ``[0, subtree_size[0])`` are meaningful, ``subtree_size[i]`` is the node count
  of the subtree rooted at ``i``.
* ``node_type`` base values ``0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, 4 TFUNC``;
  ``OUT_NODE = 1 << 7`` is OR-ed in for multi-output nodes. The mode is selected
  by ``out_len > 1`` exactly like ``forward.cu::evaluate`` does — with
  ``out_len == 1`` the raw type is compared (so a node carrying the OUT_NODE bit
  falls into the TFUNC branch), with ``out_len > 1`` the type is masked with
  ``0x7F`` and the value is bit-reinterpreted as ``{int16 function, int16
  outIndex}``.
* Functions ids: ``0 IF`` (ternary), ``1..13`` binary, ``14..28`` unary; the
  left-most operand is operand 1. A TFUNC node is *always* an ``IF`` (the kernel
  ignores its function id) and a function id that does not belong to the node's
  arity yields ``0.0`` (the kernel leaves its ``float top_val{}`` zero-initialized).
* SR fitness is the POSITIVE loss ``sum_over_points sum_over_outputs err /
  data_points`` (it is negated one layer up, in ``problem/``), where
  ``err = diff**2`` (MSE) or ``|diff|`` (MAE) and ``diff = label - pred``.
  Outputs are summed but the divisor is the number of data points.

Run the suite that uses this module from the repository root::

    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

__all__ = [
    "MAX_STACK",
    "DELTA",
    "MAX_VAL",
    "VAR",
    "CONST",
    "UFUNC",
    "BFUNC",
    "TFUNC",
    "TYPE_MASK",
    "OUT_NODE",
    "IF",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "LOOSE_DIV",
    "POW",
    "LOOSE_POW",
    "MAX",
    "MIN",
    "LT",
    "GT",
    "LE",
    "GE",
    "SIN",
    "COS",
    "TAN",
    "SINH",
    "COSH",
    "TANH",
    "LOG",
    "LOOSE_LOG",
    "EXP",
    "INV",
    "LOOSE_INV",
    "NEG",
    "ABS",
    "SQRT",
    "LOOSE_SQRT",
    "FUNC_END",
    "Node",
    "arity_of",
    "is_function_kind",
    "decode_row",
    "flatten",
    "forward_tree",
    "forward_from_rows",
    "batch_forward_from_rows",
    "sr_fitness_from_rows",
]

# --------------------------------------------------------------------------- #
# Constants — duplicated verbatim from src/evogp/cuda/defs.h / tree/utils.py
# --------------------------------------------------------------------------- #
MAX_STACK = 1024          # defs.h: MAX_STACK
DELTA = np.float32(1e-9)  # defs.h: DELTA = 1E-9f (float32!)
MAX_VAL = np.float32(1e9)  # defs.h: MAX_VAL = 1E9f

VAR, CONST, UFUNC, BFUNC, TFUNC = 0, 1, 2, 3, 4
TYPE_MASK = 0x7F  # NodeType::TYPE_MASK
OUT_NODE = 1 << 7  # NodeType::OUT_NODE

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
FUNC_END = 29  # Function::END

_TFUNC_END = GE  # ids <= GE are binary, the rest (up to FUNC_END) are unary


def arity_of(function_id: int) -> int:
    """Number of children of a function node (defs.h arities)."""
    if function_id == IF:
        return 3
    if function_id <= _TFUNC_END:
        return 2
    return 1


def kind_arity(kind: int) -> int:
    """Number of children implied by a node *type* (not by the function id)."""
    if kind == TFUNC:
        return 3
    if kind == BFUNC:
        return 2
    if kind == UFUNC:
        return 1
    return 0


def _kind_from_type(node_type: int) -> int:
    """Mirror the kernel's if/else chain: 0 VAR, 1 CONST, 2 UFUNC, 3 BFUNC, else TFUNC.

    ``_process_node`` has no default branch other than the ternary one, so ANY
    unmapped type (including a type with the OUT_NODE bit left in place in
    single-output mode) is evaluated as a 3-operand ``IF`` node.
    """
    if node_type == VAR:
        return VAR
    if node_type == CONST:
        return CONST
    if node_type == UFUNC:
        return UFUNC
    if node_type == BFUNC:
        return BFUNC
    return TFUNC


def is_function_kind(kind: int) -> bool:
    return kind in (UFUNC, BFUNC, TFUNC)


# --------------------------------------------------------------------------- #
# Decoded node
# --------------------------------------------------------------------------- #
# ``value`` keeps the raw float32 ``node_value`` (for out-nodes it is the
# function id extracted from the packed bits, see decode_row). ``out_idx`` is the
# uint32 of the packed out index (only meaningful when ``is_out``), mirroring the
# ``unsigned int outIdx`` local of ``_process_node``.
Node = namedtuple("Node", "kind value children is_out out_idx size")


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _as_float32_np(row) -> np.ndarray:
    if isinstance(row, torch.Tensor):
        return row.detach().to("cpu", torch.float32).numpy().reshape(-1)
    return np.asarray(row, dtype=np.float32).reshape(-1)


def _as_int16_np(row) -> np.ndarray:
    if isinstance(row, torch.Tensor):
        return row.detach().to("cpu", torch.int16).numpy().reshape(-1)
    return np.asarray(row).astype(np.int16).reshape(-1)


def _to_uint32(x: int) -> int:
    """C++ ``(unsigned int)`` conversion of an int16_t value."""
    return x & 0xFFFFFFFF


def _packed_out_node(value_f32: float):
    """Unpack ``struct OutNodeValue { int16_t function, outIndex; }``.

    The float is bit-reinterpreted as two int16s (little endian): the low half is
    the function id, the high half the output index (``kernel.h``).
    """
    bits = int(np.float32(value_f32).view(np.int32))
    function = _to_uint32(bits & 0xFFFF)  # high bit of the low half -> huge uint32
    out_index = _to_uint32((bits >> 16) & 0xFFFF)
    return function, out_index


def decode_row(node_value_row, node_type_row, subtree_size_row, out_len: int = 1, check: bool = True) -> Node:
    """Decode one row of the ``(pop, max_tree_len)`` tensors into a nested tree.

    Returns the root :class:`Node`. Mirrors the kernel's own decoding:

    * the tree length is ``subtree_size[0]`` and everything past it is ignored;
    * ``out_len > 1`` selects the multi-output decoding (mask the type with
      ``0x7F``, unpack the function/out-index from the value bits), exactly like
      ``treeGPEvalKernel<true>``; with ``out_len == 1`` (the default) the raw
      node type is used and the OUT_NODE bit, if present, makes the node fall
      through to the TFUNC branch (``_process_node<false>``);
    * the node type/base mapping follows the kernel's if/else chain:
      ``== 0`` VAR, ``== 1`` CONST, ``== 2`` UFUNC, ``== 3`` BFUNC, else TFUNC.

    With ``check=True`` (default) the encoding is validated: the prefix walk must
    consume exactly ``subtree_size[0]`` nodes and every declared
    ``subtree_size[i]`` must equal ``1 + sum(child sizes)``. Raises ``ValueError``
    otherwise (a malformed row is a bug in generation/splicing, not in evaluation).
    """
    value = _as_float32_np(node_value_row)
    raw_type = _as_int16_np(node_type_row)
    size = _as_int16_np(subtree_size_row)

    if value.shape[0] == 0:
        raise ValueError("empty node_value row")
    if not (value.shape[0] == raw_type.shape[0] == size.shape[0]):
        raise ValueError(
            f"row length mismatch: node_value={value.shape[0]}, node_type={raw_type.shape[0]}, "
            f"subtree_size={size.shape[0]}"
        )
    if size[0] < 1:
        raise ValueError(f"subtree_size[0]={int(size[0])} must be >= 1")
    tree_size = int(size[0])
    if tree_size > value.shape[0]:
        raise ValueError(f"subtree_size[0]={tree_size} exceeds the row length {value.shape[0]}")

    multi_output = out_len > 1
    # Prefix-order (children-less) nodes, then assembled bottom-up into a nest.
    flat: list[Node] = []
    for i in range(tree_size):
        t = int(raw_type[i])
        node_value = float(value[i])
        out_idx_u = None
        if multi_output:
            is_out = bool(t & OUT_NODE)
            kind = _kind_from_type(t & TYPE_MASK)
            # The kernels unpack the value bits for out-flagged *function* nodes
            # only; CONST/VAR nodes return before their (uninitialized) outIdx is
            # ever read, so they contribute nothing to an output.
            if is_out and is_function_kind(kind):
                function_u, out_idx_u = _packed_out_node(node_value)
                node_value = float(function_u)
        else:
            is_out = False
            kind = _kind_from_type(t)
        flat.append(Node(kind, node_value, (), is_out, out_idx_u, int(size[i])))

    stack: list[Node] = []
    for node in reversed(flat):
        k = kind_arity(node.kind)
        if len(stack) < k:
            raise ValueError(
                f"malformed prefix expression: node at prefix position needs {k} children "
                f"but only {len(stack)} sub-trees remain on the stack"
            )
        children = tuple(reversed([stack.pop() for _ in range(k)]))
        computed_size = 1 + sum(child.size for child in children)
        if check and computed_size != node.size:
            raise ValueError(
                f"inconsistent subtree_size: declared {node.size} but the subtree has "
                f"{computed_size} nodes (kind={node.kind}, value={node.value})"
            )
        stack.append(node._replace(children=children, size=computed_size))
    if len(stack) != 1:
        raise ValueError(f"malformed prefix expression: {len(stack)} sub-trees left over")
    return stack[0]


def flatten(node: Node) -> tuple:
    """Return the decoded tree as a tuple of nodes in prefix (pre-order) order."""
    out: list[Node] = []
    work = [node]
    while work:
        current = work.pop()
        out.append(current)
        work.extend(reversed(current.children))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _as_x(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.detach().to(torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {tuple(x.shape)}")
    return x


def _full(n: int, v) -> torch.Tensor:
    return torch.full((n,), float(v), dtype=torch.float32)


def _apply(kind: int, function_id: int, args, n: int) -> torch.Tensor:
    """One node of ``forward.cu::_process_node`` (``args`` = left-most first)."""
    if kind == UFUNC:
        a = args[0]
        if function_id == SIN:
            return torch.sin(a)
        if function_id == COS:
            return torch.cos(a)
        if function_id == TAN:
            return torch.tan(a)
        if function_id == SINH:
            return torch.sinh(a)
        if function_id == COSH:
            return torch.cosh(a)
        if function_id == TANH:
            return torch.tanh(a)
        if function_id == LOG:
            return torch.log(a)  # LOG(0) -> -inf, LOG(x<0) -> NaN
        if function_id == LOOSE_LOG:
            return torch.where(a == 0.0, _full(n, -MAX_VAL), torch.log(torch.abs(a)))
        if function_id == EXP:
            return torch.exp(a)
        if function_id == INV:
            return torch.where(a == 0.0, _full(n, float("nan")), 1.0 / a)
        if function_id == LOOSE_INV:
            b = torch.where(torch.abs(a) <= DELTA, torch.copysign(_full(n, DELTA), a), a)
            return 1.0 / b
        if function_id == NEG:
            return -a
        if function_id == ABS:
            return torch.abs(a)
        if function_id == SQRT:
            return torch.sqrt(a)  # SQRT(neg) -> NaN
        if function_id == LOOSE_SQRT:
            b = torch.where(a <= 0.0, torch.abs(a), a)
            return torch.sqrt(b)
        return torch.zeros(n, dtype=torch.float32)

    if kind == BFUNC:
        a, b = args[0], args[1]
        if function_id == ADD:
            return a + b
        if function_id == SUB:
            return a - b
        if function_id == MUL:
            return a * b
        if function_id == DIV:
            return torch.where(b == 0.0, _full(n, float("nan")), a / b)
        if function_id == LOOSE_DIV:
            b2 = torch.where(torch.abs(b) <= DELTA, torch.copysign(_full(n, DELTA), b), b)
            return a / b2
        if function_id == POW:
            return torch.pow(a, b)
        if function_id == LOOSE_POW:
            return torch.where((a == 0.0) & (b == 0.0), _full(n, 0.0), torch.pow(torch.abs(a), b))
        if function_id == MAX:
            return torch.where(a >= b, a, b)
        if function_id == MIN:
            return torch.where(a <= b, a, b)
        if function_id == LT:
            return torch.where(a < b, _full(n, 1.0), _full(n, -1.0))
        if function_id == GT:
            return torch.where(a > b, _full(n, 1.0), _full(n, -1.0))
        if function_id == LE:
            return torch.where(a <= b, _full(n, 1.0), _full(n, -1.0))
        if function_id == GE:
            return torch.where(a >= b, _full(n, 1.0), _full(n, -1.0))
        return torch.zeros(n, dtype=torch.float32)

    # TFUNC: the kernel ignores the function id and always evaluates IF.
    cond, then_value, else_value = args[0], args[1], args[2]
    return torch.where(cond > 0.0, then_value, else_value)


def forward_tree(tree: Node, x, out_len: int = 1, multi_output: bool = None):
    """Evaluate a decoded tree on ``x`` (``(N, input_len)``) -> ``(N, out_len)``.

    Nodes are processed in REVERSE prefix order, i.e. children before their
    parent, exactly like ``_treeGPEvalByStack``. With ``multi_output`` (default:
    ``out_len > 1``, the kernel's rule) every node passes its RIGHT-MOST operand
    up to its parent and each out-node accumulates its own value into
    ``out[outIdx]`` (out indices ``>= out_len`` are dropped); otherwise the root's
    value is the single result.

    The output tensor is the accumulated one for multi-output trees: the value of
    the root is ignored by ``treeGPEvalKernel`` in that mode.
    """
    if multi_output is None:
        multi_output = out_len > 1
    if multi_output and out_len < 1:
        raise ValueError("out_len must be >= 1")

    x = _as_x(x)
    n = x.shape[0]
    nodes = flatten(tree)

    outs = torch.zeros((n, out_len), dtype=torch.float32) if multi_output else None
    stack: list[torch.Tensor] = []
    for node in reversed(nodes):
        if node.kind == CONST:
            stack.append(_full(n, node.value))
            continue
        if node.kind == VAR:
            index = int(node.value)
            if index < 0 or index >= x.shape[1]:
                raise IndexError(
                    f"VAR node index {index} out of range for input_len={x.shape[1]} "
                    "(the kernel would read out of bounds)"
                )
            stack.append(x[:, index].clone())
            continue

        k = kind_arity(node.kind)
        if len(stack) < k:
            raise ValueError(f"stack underflow while evaluating a node of kind {node.kind}")
        args = [stack.pop() for _ in range(k)][::-1]  # left-most operand first
        top_val = _apply(node.kind, int(node.value), args, n)
        if multi_output and node.out_idx is not None and node.out_idx < out_len:
            outs[:, node.out_idx] = outs[:, node.out_idx] + top_val
        stack.append(args[-1] if multi_output else top_val)

    if multi_output:
        return outs
    if len(stack) != 1:
        raise ValueError(f"malformed tree: {len(stack)} values left on the stack (expected 1)")
    return stack[0].unsqueeze(1)


def _forward_pop(node_value, node_type, subtree_size, x_per_row, out_len: int):
    """``x_per_row``: ``(pop, N, input_len)`` -> ``(pop, N, out_len)``."""
    pop = int(x_per_row.shape[0])
    if pop == 0:
        return torch.zeros((0, 0, out_len), dtype=torch.float32)
    results = []
    for i in range(pop):
        tree = decode_row(
            _row(node_value, i),
            _row(node_type, i),
            _row(subtree_size, i),
            out_len=out_len,
        )
        results.append(forward_tree(tree, x_per_row[i], out_len=out_len))
    return torch.stack(results, dim=0)


def _row(tensor, index: int):
    """Row ``index`` of a ``(pop, max_tree_len)`` tensor/numpy array, or a 1-D row."""
    if tensor is None:
        raise ValueError("row tensor is None")
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().to("cpu")
        return t if t.dim() == 1 else t[index]
    arr = np.asarray(tensor)
    return arr if arr.ndim == 1 else arr[index]


def forward_from_rows(node_value, node_type, subtree_size, inputs, out_len: int = 1):
    """Reference ``Forest.forward``: rows ``(pop, ...)`` + ``(pop, input_len)`` -> ``(pop, out_len)``.

    Each row ``i`` is evaluated with its own input row ``inputs[i]`` (this is what
    ``treeGPEvalKernel`` does with ``variables + n * varLen``).
    """
    x = _as_x(inputs)
    pop = _pop_size(node_value)
    if x.shape[0] != pop:
        raise ValueError(f"inputs has {x.shape[0]} rows but the population has {pop}")
    results = forward_tree(
        decode_row(_row(node_value, 0), _row(node_type, 0), _row(subtree_size, 0), out_len=out_len),
        x[0],
        out_len=out_len,
    )
    rows = [results]
    for i in range(1, pop):
        tree = decode_row(_row(node_value, i), _row(node_type, i), _row(subtree_size, i), out_len=out_len)
        rows.append(forward_tree(tree, x[i], out_len=out_len))
    return torch.cat(rows, dim=0)


def batch_forward_from_rows(node_value, node_type, subtree_size, x, out_len: int = 1):
    """Reference ``Forest.batch_forward``: rows ``(pop, ...)`` + ``(N, input_len)`` -> ``(pop, N, out_len)``.

    Every tree is evaluated on the *same* ``x`` (the kernel call tiles ``x`` over
    the population with ``x.repeat(pop_size, 1)``).
    """
    x = _as_x(x)
    pop = _pop_size(node_value)
    per_row = x.unsqueeze(0).expand(pop, x.shape[0], x.shape[1])
    return _forward_pop(node_value, node_type, subtree_size, per_row, out_len)


def _pop_size(node_value) -> int:
    if isinstance(node_value, torch.Tensor):
        return int(node_value.shape[0]) if node_value.dim() > 1 else 1
    arr = np.asarray(node_value)
    return int(arr.shape[0]) if arr.ndim > 1 else 1


def sr_fitness_from_rows(
    node_value,
    node_type,
    subtree_size,
    inputs,
    labels,
    use_mse: bool = True,
    out_len: int = None,
):
    """Reference ``Forest.SR_fitness`` -> ``(pop,)`` POSITIVE loss (float32).

    ``loss = sum_points sum_outputs err / data_points`` with
    ``err = (label - pred)**2`` (``use_mse=True``) or ``|label - pred|``
    (``use_mse=False``). The divisor is the number of data points only -- a
    multi-output tree is NOT normalized by ``out_len`` (``averageFitnessValueKernel``
    divides by ``dataPoints``). Nothing is negated here: the sign flip lives in
    ``problem/symbolic_regression.py``.
    """
    x = _as_x(inputs)
    lab = labels
    if not isinstance(lab, torch.Tensor):
        lab = torch.as_tensor(lab, dtype=torch.float32)
    lab = lab.detach().to(torch.float32)
    if lab.dim() != 2:
        raise ValueError(f"labels must be 2-D (data_points, out_len), got {tuple(lab.shape)}")
    if lab.shape[0] != x.shape[0]:
        raise ValueError(f"labels has {lab.shape[0]} rows but inputs has {x.shape[0]}")
    if out_len is None:
        out_len = int(lab.shape[1])

    pred = batch_forward_from_rows(node_value, node_type, subtree_size, x, out_len=out_len)
    diff = lab[None, :, :] - pred
    err = diff * diff if use_mse else torch.abs(diff)
    # per data point: sum over outputs, then sum over data points, then / data_points
    return err.sum(dim=2).sum(dim=1) / float(x.shape[0])
