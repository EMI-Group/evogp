"""Operator tests for EvoGP's CUDA-free GP engine.

Covers the plain-torch replacements of the old ``tree_generate`` /
``tree_mutate`` / ``tree_crossover`` kernels:

* ``Forest.random_generate(pop_size, descriptor)``: structural validity of the
  generated population (arity, types, var/const ranges, function subset, depth
  schedule) and reproducibility under a reset torch RNG.
* ``Forest.mutate`` / ``Forest.crossover``: the prefix splices of
  ``cuda/mutation.cu::_gpTreeReplace`` (ancestor subtree size += sizeDiff, tail
  shifted by sizeDiff, donor subtree copied verbatim) and the silent-copy edge
  cases (invalid index / overflow), checked against an independent reference
  splice implemented in this file (no import of the code under test).

The rewritten RNG is torch-based and only *distribution equivalent* to the old
taus88 kernels, so no test asserts sampled values - only invariants and
reproducibility.

Run from the repository root::

    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q
"""

from __future__ import annotations

import importlib
import inspect
import random

import pytest
import torch

try:  # the tree layer is importable only once the ETL-based engine is in place
    from evogp.tree import Forest, GenerateDescriptor, NType
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only in pre-merge worktrees
    Forest = GenerateDescriptor = NType = None
    _IMPORT_ERROR = exc


@pytest.fixture(autouse=True)
def _require_engine():
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "evogp.tree is not importable (the ETL-based engine is not present in this "
            f"worktree yet): {_IMPORT_ERROR!r}"
        )


# --- Node encoding constants (fixed by `cuda/defs.h` <-> `tree/utils.py`) ----- #
VAR, CONST, UFUNC, BFUNC, TFUNC = 0, 1, 2, 3, 4
TYPE_MASK = 0x7F
OUT_NODE = 1 << 7
# arity indexed by base node type (VAR, CONST, UFUNC, BFUNC, TFUNC)
ARITY = (0, 0, 1, 2, 3)
# function id -> name, same order as `evogp.tree.utils.FUNCS_NAMES`
FUNCS_NAMES = [
    "if", "+", "-", "*", "/", "loose_div", "pow", "loose_pow", "max", "min",
    "<", ">", "<=", ">=", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "log", "loose_log", "exp", "inv", "loose_inv", "neg", "abs", "sqrt",
    "loose_sqrt",
]
# function id -> arity: IF=0 is ternary, 1..13 binary, 14..28 unary
FUNC_ARITY = [3] + [2] * 13 + [1] * 15


def expected_arity(func_id: int) -> int:
    """Arity of the function with id ``func_id`` (``-1`` when out of range)."""
    if 0 <= func_id < len(FUNC_ARITY):
        return FUNC_ARITY[func_id]
    return -1


def _base_type(node_type: int) -> int:
    return int(node_type) & TYPE_MASK


# --- Device probe: follow the device the public API itself picks ------------- #
_DEVICE = None


def api_device() -> torch.device:
    """Device used by the engine's own tensor factories (never hard-coded twice)."""
    global _DEVICE
    if _DEVICE is None:
        zero_generate = getattr(Forest, "zero_generate", None)
        assert callable(zero_generate), "Forest.zero_generate is part of the public API"
        _DEVICE = zero_generate(1, 8, 1, 1).batch_node_value.device
    return _DEVICE


def index_tensor(values, dtype=torch.int32) -> torch.Tensor:
    """1-D index tensor as the operator contract requires (int32, engine device)."""
    return torch.tensor(list(values), dtype=dtype, device=api_device())


# --- Row model: build/compare prefix (pre-order) encoded trees ---------------- #
class Row:
    """One individual: prefix-ordered node values/types/sizes (meaningful part only)."""

    __slots__ = ("value", "type", "size", "total")

    def __init__(self, value, node_type, size, total):
        self.value = list(value)
        self.type = list(node_type)
        self.size = list(size)
        self.total = int(total)
        assert self.total == len(self.value) == len(self.type) == len(self.size)

    def __repr__(self):  # pragma: no cover - debugging aid
        nodes = [(self.type[i], round(self.value[i], 4), self.size[i]) for i in range(self.total)]
        return f"Row(total={self.total}, nodes={nodes})"


def encode_node(node):
    """Nested node tuple -> ``(values, node_types)`` in prefix order.

    Grammar: ``("var", i)``, ``("const", v)``, ``("u", fid, child)``,
    ``("b", fid, left, right)``, ``("t", fid, a, b, c)`` and
    ``("out", out_index, function_node)`` for multi-output out-nodes.
    """
    tag = node[0]
    if tag in ("var", "const"):
        return [float(node[1])], [VAR if tag == "var" else CONST]
    if tag == "out":
        out_index, sub = node[1], node[2]
        values, types = encode_node(sub)
        assert types[0] in (UFUNC, BFUNC, TFUNC), "only function nodes are out-nodes"
        bits = (int(out_index) << 16) | (int(values[0]) & 0xFF)
        values[0] = torch.tensor([bits], dtype=torch.int32).view(torch.float32)[0].item()
        types[0] |= OUT_NODE
        return values, types
    base = {"u": UFUNC, "b": BFUNC, "t": TFUNC}[tag]
    assert len(node[2:]) == ARITY[base], f"{tag} takes {ARITY[base]} children"
    assert expected_arity(int(node[1])) == ARITY[base], f"bad function id {node[1]}"
    values, types = [float(node[1])], [base]
    for child in node[2:]:
        child_values, child_types = encode_node(child)
        values += child_values
        types += child_types
    return values, types


def sizes_from_types(node_types) -> tuple[list, int]:
    """Bottom-up subtree size of every node (reverse of the prefix walk)."""
    sizes = [0] * len(node_types)
    stack = []
    for i in reversed(range(len(node_types))):
        arity = ARITY[_base_type(node_types[i])]
        size = 1
        for _ in range(arity):
            size += stack.pop()
        stack.append(size)
        sizes[i] = size
    assert len(stack) == 1, "node types do not describe a single tree"
    return sizes, stack[0]


def row_from_node(node) -> Row:
    values, types = encode_node(node)
    sizes, total = sizes_from_types(types)
    return Row(values, types, sizes, total)


def build_forest(rows, max_tree_len, input_len=2, output_len=1, device=None) -> Forest:
    """Materialize hand-built rows (the unused row tail is filled with sentinels)."""
    device = device if device is not None else api_device()
    shape = (len(rows), max_tree_len)
    value = torch.full(shape, -999.0, dtype=torch.float32, device=device)
    ntype = torch.full(shape, 7, dtype=torch.int16, device=device)
    size = torch.zeros(shape, dtype=torch.int16, device=device)
    for i, row in enumerate(rows):
        assert row.total <= max_tree_len, "hand-built tree does not fit max_tree_len"
        value[i, : row.total] = torch.tensor(row.value, dtype=torch.float32, device=device)
        ntype[i, : row.total] = torch.tensor(row.type, dtype=torch.int16, device=device)
        size[i, : row.total] = torch.tensor(row.size, dtype=torch.int16, device=device)
    return Forest(input_len, output_len, value, ntype, size)


def row_of(forest: Forest, index: int) -> Row:
    """Extract the meaningful prefix of population row ``index``."""
    total = int(forest.batch_subtree_size[index, 0].item())
    assert 0 < total <= forest.max_tree_len
    return Row(
        forest.batch_node_value[index, :total].cpu().tolist(),
        forest.batch_node_type[index, :total].cpu().tolist(),
        forest.batch_subtree_size[index, :total].cpu().tolist(),
        total,
    )


def forest_rows(forest: Forest) -> list:
    return [row_of(forest, i) for i in range(forest.pop_size)]


def assert_rows_equal(actual: Row, expected: Row, label: str = ""):
    prefix = f"{label}: " if label else ""
    assert actual.total == expected.total, f"{prefix}tree length {actual.total} != {expected.total}"
    assert actual.size == expected.size, f"{prefix}subtree sizes differ"
    assert actual.type == expected.type, f"{prefix}node types differ"
    assert actual.value == pytest.approx(expected.value, rel=1e-6, abs=1e-6), (
        f"{prefix}node values differ"
    )


def snapshot(forest: Forest):
    """Full-tensor snapshot used to prove that an operator left its input alone."""
    return tuple(
        tensor.detach().cpu().clone()
        for tensor in (
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
        )
    )


def assert_snapshot_unchanged(forest: Forest, before, label: str):
    names = ("batch_node_value", "batch_node_type", "batch_subtree_size")
    for got, want, name in zip(snapshot(forest), before, names):
        assert torch.equal(got, want), f"{label}: input forest was mutated in place ({name})"


# --- Structural validator (the per-row invariants of the representation) ----- #


def validate_row(
    row: Row, *, input_len=None, output_len=1, using_funcs=None, const_bounds=None,
    max_layer_cnt=None, label: str = "",
) -> int:
    """Check every structural invariant of one prefix-encoded row; return its depth.

    ``const_bounds`` is the inclusive ``(lo, hi)`` interval a CONST value may fall
    in - either the descriptor's ``const_range`` or the span of an explicit
    ``const_samples`` set (a torch-RNG sampler may legitimately pick any value of
    the range, so membership in the sampled set is not required).
    """
    prefix = f"{label}: " if label else ""
    total = row.total
    assert 1 <= total <= 1 << 20, f"{prefix}invalid tree length {total}"
    func_ids = None if using_funcs is None else {FUNCS_NAMES.index(n) for n in using_funcs}
    const_lo, const_hi = (const_bounds if const_bounds is not None else (None, None))
    depths = []
    open_children = []  # remaining children of every open ancestor (a stack)
    remaining, idx = 1, 0
    while remaining > 0:
        assert idx < total, f"{prefix}prefix walk ran past subtree_size[0]={total}"
        while open_children and open_children[-1] == 0:
            open_children.pop()
        depths.append(len(open_children))
        if open_children:
            open_children[-1] -= 1
        remaining -= 1
        node_type = int(row.type[idx])
        base = _base_type(node_type)
        value = float(row.value[idx])
        assert base in (VAR, CONST, UFUNC, BFUNC, TFUNC), f"{prefix}bad node type {node_type}"
        if output_len == 1:
            assert not node_type & OUT_NODE, f"{prefix}OUT_NODE set with output_len == 1"
        if base in (UFUNC, BFUNC, TFUNC):
            if node_type & OUT_NODE:
                assert output_len > 1, f"{prefix}OUT_NODE set with output_len == 1"
                bits = int(torch.tensor([value], dtype=torch.float32).view(torch.int32)[0])
                func_id, out_index = bits & 0xFF, bits >> 16
                assert 0 <= out_index < output_len, f"{prefix}out index {out_index} out of range"
            else:
                func_id = int(value)
                assert float(func_id) == value, f"{prefix}function id {value} is not an integer"
            assert expected_arity(func_id) == ARITY[base], (
                f"{prefix}function id {func_id} is not a {base}-arity function node"
            )
            if func_ids is not None:
                assert func_id in func_ids, (
                    f"{prefix}function {FUNCS_NAMES[func_id]} outside using_funcs"
                )
        elif base == VAR:
            assert float(int(value)) == value, f"{prefix}variable index {value} is not an integer"
            if input_len is not None:
                assert 0 <= int(value) < input_len, f"{prefix}variable index {value} out of range"
        else:  # CONST: exact membership would over-constrain the sampler, so check the range
            tol = 1e-5 * max(1.0, abs(const_lo or 0.0), abs(const_hi or 0.0))
            assert value == value, f"{prefix}constant is NaN"
            if const_lo is not None:
                assert const_lo - tol <= value <= const_hi + tol, (
                    f"{prefix}constant {value} outside the descriptor's const range "
                    f"[{const_lo}, {const_hi}]"
                )
        arity = ARITY[base]
        remaining += arity
        open_children.append(arity)
        idx += 1
    assert idx == total, f"{prefix}prefix walk consumed {idx} nodes, subtree_size[0]={total}"

    # subtree_size[i] == number of nodes in the subtree rooted at i
    sizes, root_size = sizes_from_types(row.type)
    assert root_size == total, f"{prefix}arity walk gives {root_size} nodes, declared {total}"
    assert row.size == sizes, f"{prefix}subtree sizes are not self-consistent"

    max_depth = max(depths)
    if max_layer_cnt is not None:
        # The descriptor's leaf schedule is [layer_leaf_prob] * (max_layer_cnt - 1) + 1.0,
        # so a node may be generated at most max_layer_cnt levels deep (depth is 0-based).
        assert max_depth <= max_layer_cnt, (
            f"{prefix}node at depth {max_depth} exceeds max_layer_cnt={max_layer_cnt}"
        )
    return max_depth


def validate_forest(forest: Forest, **kwargs) -> int:
    """Validate every row; return the observed maximum depth."""
    depths = []
    for i in range(forest.pop_size):
        depths.append(validate_row(row_of(forest, i), label=f"row {i}", **kwargs))
    return max(depths) if depths else 0


# --- Reference splice (independent oracle, from mutation.cu::_gpTreeReplace) --- #
def reference_splice(old: Row, old_idx: int, new: Row, new_idx: int, max_tree_len: int) -> Row:
    """Expected result of replacing subtree ``old_idx`` of ``old`` with the subtree at
    ``new_idx`` of ``new``.

    Mirrors ``_gpTreeReplace``: the ancestors of ``old_idx`` grow/shrink by
    ``sizeDiff``, the donor subtree is copied verbatim and the old tail is shifted
    by ``sizeDiff``.  Invalid indices and overflow (> ``max_tree_len``) return
    ``old`` unchanged (the documented silent-copy behaviour).
    """
    old_total = old.total
    if old_idx < 0 or old_idx >= old_total:
        return old  # silent copy: position is not a node of the tree
    if new_idx < 0 or new_idx >= new.total:
        return old  # silent copy: donor position is not a node
    new_total = new.size[new_idx]
    if new_idx + new_total > new.total:
        return old  # silent copy: donor subtree does not fit in its own row
    old_sub = old.size[old_idx]
    diff = new_total - old_sub
    if old_total + diff > max_tree_len:
        return old  # silent copy: the spliced tree would not fit max_tree_len

    values, types, sizes = list(old.value), list(old.type), list(old.size)
    # pad so that the grown tail can be shifted into place
    values += [0.0] * (max_tree_len - len(values))
    types += [0] * (max_tree_len - len(types))
    sizes += [0] * (max_tree_len - len(sizes))

    # 1. walk root -> old_idx and grow/shrink every ancestor's subtree size
    cur = 0
    while cur < old_idx:
        sizes[cur] += diff
        base = _base_type(old.type[cur])
        cur += 1
        if cur >= old_idx:
            break
        if base == BFUNC:
            right = cur + old.size[cur]
            if old_idx >= right:
                cur = right
        elif base == TFUNC:
            mid = cur + old.size[cur]
            if old_idx >= mid:
                right = mid + old.size[mid]
                cur = mid if old_idx < right else right

    # 2. copy the donor subtree verbatim
    for i in range(new_total):
        values[old_idx + i] = new.value[new_idx + i]
        types[old_idx + i] = new.type[new_idx + i]
        sizes[old_idx + i] = new.size[new_idx + i]

    # 3. shift the untouched tail of the old tree
    for i in range(old_idx + old_sub, old_total):
        values[i + diff] = old.value[i]
        types[i + diff] = old.type[i]
        sizes[i + diff] = old.size[i]

    result_total = old_total + diff
    assert result_total == sizes[0], "reference splice produced an inconsistent root size"
    return Row(values[:result_total], types[:result_total], sizes[:result_total], result_total)


# --- Reusable hand-built trees ------------------------------------------------ #
T_SMALL = row_from_node(("b", 3, ("var", 0), ("var", 1)))  # x0 * x1
T_MED = row_from_node(
    (
        "b", 1,                                             # +
        ("b", 3, ("u", 14, ("var", 0)), ("const", 0.25)),   # * sin(x0)  0.25
        (
            "t", 0,                                         # if(x1, (inv(x0)), 0.5 <=> x1)
            ("var", 1),
            ("u", 25, ("var", 0)),
            ("b", 2, ("const", -0.5), ("var", 1)),
        ),
    )
)  # fmt: skip
T_UFUNC = row_from_node(("u", 20, ("u", 14, ("var", 1))))  # log(sin(x1))
D_BIG = row_from_node(
    (
        "b", 4,                                             # /
        ("b", 3, ("var", 0), ("const", 2.0)),               # x0 * 2
        ("b", 1, ("u", 25, ("var", 1)), ("var", 0)),        # -x1 + x0
    )
)  # fmt: skip
D_TWO = row_from_node(("u", 26, ("var", 1)))  # abs(x1)
D_LEAF = row_from_node(("var", 1))


# (case id, old tree, replaced node index, donor tree, max_tree_len)
MUTATE_CASES = [
    ("root_shrink", T_MED, 0, D_BIG, 32),
    ("root_grow_to_ufunc", T_UFUNC, 0, D_BIG, 32),
    ("grow_leaf_deep_right", T_MED, 10, D_BIG, 32),
    ("grow_mid_child_of_tfunc", T_MED, 8, D_BIG, 32),
    ("grow_right_child_of_tfunc", T_MED, 11, D_BIG, 32),
    ("shrink_tfunc_subtree", T_MED, 5, D_LEAF, 32),
    ("shrink_left_child_of_root", T_MED, 1, D_LEAF, 32),
    ("equal_size_replace", T_MED, 7, D_TWO, 32),
    ("replace_last_node", T_MED, 11, D_LEAF, 32),
    ("leaf_to_big", T_SMALL, 1, D_BIG, 32),
    ("big_to_leaf", T_SMALL, 0, D_LEAF, 32),
    # donor subtree does not fit the declared tree length -> silent copy
    ("overflow", T_MED, 10, D_BIG, T_MED.total),
]


@pytest.mark.parametrize(
    "case_id,old,index,donor,max_tree_len",
    MUTATE_CASES,
    ids=[case[0] for case in MUTATE_CASES],
)
def test_mutate_matches_reference_splice(case_id, old, index, donor, max_tree_len):
    forest = build_forest([old], max_tree_len, input_len=2, output_len=1)
    donors = build_forest([donor], max_tree_len, input_len=2, output_len=1)
    donor_before = snapshot(donors)

    result = forest.mutate(index_tensor([index]), donors)

    assert result is not forest
    assert result.pop_size == forest.pop_size
    assert result.max_tree_len == forest.max_tree_len
    assert (result.input_len, result.output_len) == (forest.input_len, forest.output_len)
    expected = reference_splice(old, index, donor, 0, max_tree_len)
    assert_rows_equal(row_of(result, 0), expected, case_id)
    validate_row(row_of(result, 0), output_len=1, label=case_id)
    assert_snapshot_unchanged(donors, donor_before, case_id)


def test_reference_splice_replaces_root_with_donor_subtree():
    """Self-check of the oracle used by the mutation/crossover tests."""
    assert D_LEAF.total < D_BIG.total
    assert_rows_equal(reference_splice(D_LEAF, 0, D_BIG, 0, 32), D_BIG, "root replacement")
    assert_rows_equal(reference_splice(D_BIG, 0, D_LEAF, 0, 32), D_LEAF, "root replacement")
    # donor subtree taken from inside a donor row (not its root)
    inner = row_from_node(("u", 25, ("var", 0)))
    assert_rows_equal(reference_splice(D_LEAF, 0, T_MED, 7, 32), inner, "inner donor")
    # ancestors grow by exactly sizeDiff
    grown = reference_splice(T_MED, 10, D_BIG, 0, 32)
    assert grown.total == T_MED.total + D_BIG.total - 1
    assert grown.size[0] == grown.total
    assert grown.size[5] == T_MED.size[5] + (D_BIG.total - 1)


def test_mutate_vectorized_rows_are_spliced_independently():
    olds = [T_MED, T_MED, T_SMALL, T_UFUNC, D_BIG]
    donors_ = [D_BIG, D_LEAF, D_LEAF, D_LEAF, T_MED]
    indices = [0, 10, 1, 0, 0]
    forest = build_forest(olds, 32)
    donors = build_forest(donors_, 32)

    result = forest.mutate(index_tensor(indices), donors)

    assert result.pop_size == len(olds)
    for i, (old, donor, index) in enumerate(zip(olds, donors_, indices)):
        expected = reference_splice(old, index, donor, 0, 32)
        assert_rows_equal(row_of(result, i), expected, f"row {i}")
        validate_row(row_of(result, i), output_len=1, label=f"row {i}")


def test_mutate_does_not_touch_the_original_forest():
    forest = build_forest([T_MED, T_SMALL], 32)
    donors = build_forest([D_BIG, D_LEAF], 32)
    before, donors_before = snapshot(forest), snapshot(donors)

    result = forest.mutate(index_tensor([0, 1]), donors)

    assert result is not forest
    assert_snapshot_unchanged(forest, before, "original")
    assert_snapshot_unchanged(donors, donors_before, "donor")
    # same args twice -> same output (pure function of its inputs)
    again = forest.mutate(index_tensor([0, 1]), donors)
    for i in range(result.pop_size):
        assert_rows_equal(row_of(again, i), row_of(result, i), f"row {i}")


@pytest.mark.parametrize("index", [-1, T_MED.total, T_MED.total + 5, 10_000])
def test_mutate_invalid_position_copies_row_unchanged(index):
    forest = build_forest([T_SMALL, T_MED], 32)
    donors = build_forest([D_LEAF, D_LEAF], 32)

    result = forest.mutate(index_tensor([index, index]), donors)

    for i, original in enumerate(forest_rows(forest)):
        assert_rows_equal(row_of(result, i), original, f"row {i} (index {index})")


def test_mutate_overflow_copies_row_unchanged():
    max_tree_len = T_MED.total  # no room to grow
    forest = build_forest([T_MED], max_tree_len)
    donors = build_forest([D_BIG], max_tree_len)
    index = 10  # a leaf inside the ternary node's right subtree

    result = forest.mutate(index_tensor([index]), donors)

    assert_rows_equal(row_of(result, 0), T_MED, "overflow must be a silent copy")


def test_mutate_multi_output_rows_stay_valid():
    """OUT_NODE-tagged function nodes survive a splice unchanged."""
    tree = row_from_node(("out", 1, ("b", 3, ("var", 0), ("out", 0, ("u", 14, ("var", 1))))))
    donor = row_from_node(("out", 2, ("u", 26, ("var", 0))))
    forest = build_forest([tree], 32, input_len=2, output_len=3)
    donors = build_forest([donor], 32, input_len=2, output_len=3)

    result = forest.mutate(index_tensor([2]), donors)

    expected = reference_splice(tree, 2, donor, 0, 32)
    assert_rows_equal(row_of(result, 0), expected, "multi output mutate")
    validate_row(row_of(result, 0), input_len=2, output_len=3, label="multi output mutate")


# --- Crossover ----------------------------------------------------------------- #
# population used by the crossover cases: row 0 T_MED, row 1 D_BIG, row 2 D_LEAF, row 3 T_UFUNC
XOVER_POP = [T_MED, D_BIG, D_LEAF, T_UFUNC]
XOVER_POP_SIZE = len(XOVER_POP)

# (case id, left index, right index, left pos, right pos, max_tree_len)
XOVER_CASES = [
    ("root_swap_with_donor_root", 0, 1, 0, 0, 32),
    ("shrink_root_to_leaf", 0, 2, 0, 0, 32),
    ("same_tree_inner_subtrees", 0, 0, 1, 9, 32),
    ("equal_size_subtree_swap", 0, 0, 2, 7, 32),
    ("grow_leaf_with_donor_subtree", 0, 3, 8, 0, 32),
    ("grow_with_inner_donor_subtree", 0, 1, 11, 1, 32),
    ("head_for_unary_tree", 3, 0, 0, 3, 32),
    ("overflow_is_silent_copy", 0, 1, 10, 0, T_MED.total),
]


@pytest.mark.parametrize(
    "case_id,left,right,left_pos,right_pos,max_tree_len",
    XOVER_CASES,
    ids=[case[0] for case in XOVER_CASES],
)
def test_crossover_matches_reference_splice(case_id, left, right, left_pos, right_pos, max_tree_len):
    forest = build_forest(XOVER_POP, max_tree_len, input_len=2, output_len=1)

    result = forest.crossover(
        index_tensor([left]), index_tensor([right]), index_tensor([left_pos]), index_tensor([right_pos])
    )

    assert result.pop_size == 1
    assert result.max_tree_len == max_tree_len
    assert (result.input_len, result.output_len) == (forest.input_len, forest.output_len)
    expected = reference_splice(
        XOVER_POP[left], left_pos, XOVER_POP[right], right_pos, max_tree_len
    )
    assert_rows_equal(row_of(result, 0), expected, case_id)
    validate_row(row_of(result, 0), output_len=1, label=case_id)


def test_crossover_output_size_is_len_of_left_indices():
    forest = build_forest(XOVER_POP, 32)
    left = [0, 1, 2, 3, 0]
    right = [1, 0, 3, 0, 2]
    left_pos = [0, 1, 0, 2, 3]
    right_pos = [0, 0, 0, 1, 1]

    result = forest.crossover(
        index_tensor(left), index_tensor(right), index_tensor(left_pos), index_tensor(right_pos)
    )

    assert result.pop_size == len(left)
    for i in range(len(left)):
        expected = reference_splice(
            XOVER_POP[left[i]], left_pos[i], XOVER_POP[right[i]], right_pos[i], 32
        )
        assert_rows_equal(row_of(result, i), expected, f"row {i}")


@pytest.mark.parametrize("bad_right", [-1, XOVER_POP_SIZE, XOVER_POP_SIZE + 100])
def test_crossover_bad_right_index_copies_left_parent(bad_right):
    forest = build_forest(XOVER_POP, 32)

    result = forest.crossover(
        index_tensor([0]), index_tensor([bad_right]), index_tensor([0]), index_tensor([0])
    )

    assert_rows_equal(row_of(result, 0), XOVER_POP[0], f"right index {bad_right}")


@pytest.mark.parametrize("bad_left_pos", [T_MED.total, T_MED.total + 3, -1])
def test_crossover_invalid_left_position_copies_left_parent(bad_left_pos):
    forest = build_forest(XOVER_POP, 32)

    result = forest.crossover(
        index_tensor([0]), index_tensor([1]), index_tensor([bad_left_pos]), index_tensor([0])
    )

    assert_rows_equal(row_of(result, 0), XOVER_POP[0], f"left pos {bad_left_pos}")


def test_crossover_invalid_right_position_copies_left_parent():
    forest = build_forest(XOVER_POP, 32)
    bad_right_pos = D_BIG.total + 2  # past the right parent's last node

    result = forest.crossover(
        index_tensor([0]), index_tensor([1]), index_tensor([1]), index_tensor([bad_right_pos])
    )

    assert_rows_equal(row_of(result, 0), XOVER_POP[0], "invalid right position")


def test_crossover_overflow_copies_left_parent():
    max_tree_len = T_MED.total
    forest = build_forest(XOVER_POP, max_tree_len)
    # replace the left root (12 nodes) with D_BIG (8 nodes): fits -> sanity
    fits = forest.crossover(
        index_tensor([0]), index_tensor([1]), index_tensor([0]), index_tensor([0])
    )
    assert row_of(fits, 0).total == D_BIG.total
    # replace a leaf with a bigger subtree: would exceed max_tree_len -> silent copy
    overflow = forest.crossover(
        index_tensor([0]), index_tensor([1]), index_tensor([10]), index_tensor([0])
    )
    assert_rows_equal(row_of(overflow, 0), T_MED, "overflow must be a silent copy")


def test_crossover_pop_size_zero_raises():
    forest = build_forest(XOVER_POP, 32)
    empty = index_tensor([])

    with pytest.raises(Exception):
        forest.crossover(empty, empty, empty, empty)


def test_crossover_accepts_int32_index_tensors_and_leaves_inputs_untouched():
    forest = build_forest(XOVER_POP, 32)
    before = snapshot(forest)
    left, right = index_tensor([0, 3]), index_tensor([1, 2])
    left_pos, right_pos = index_tensor([1, 0]), index_tensor([0, 0])
    assert left.dtype == torch.int32 and right_pos.dtype == torch.int32

    result = forest.crossover(left, right, left_pos, right_pos)

    assert result is not forest
    assert_snapshot_unchanged(forest, before, "crossover parents")
    for i, (l, r, lp, rp) in enumerate([(0, 1, 1, 0), (3, 2, 0, 0)]):
        expected = reference_splice(XOVER_POP[l], lp, XOVER_POP[r], rp, 32)
        assert_rows_equal(row_of(result, i), expected, f"row {i}")


def test_crossover_index_tensors_are_read_as_integers():
    """An int32 index tensor must be used verbatim (not reinterpreted as float bits)."""
    forest = build_forest([T_MED], 32)
    donor = build_forest([D_LEAF], 32)

    at_root = forest.mutate(index_tensor([0]), donor)
    at_leaf = forest.mutate(index_tensor([10]), donor)

    assert_rows_equal(row_of(at_root, 0), D_LEAF, "mutate at 0")
    assert_rows_equal(row_of(at_leaf, 0), reference_splice(T_MED, 10, D_LEAF, 0, 32), "mutate at 10")


# --- Optional: compare an exposed splice helper against the oracle ------------ #
SPLICE_HOOK_MODULES = [
    "evogp.tree.splicing",  # the vectorized mutate/crossover splice primitive
    "evogp.tree.forest", "evogp.tree.tree", "evogp.tree.gp_ops", "evogp.tree.ops",
    "evogp.tree.torch_ops", "evogp.tree.evoxc_engine",
]
SPLICE_HOOK_NAMES = [
    "splice", "tree_splice", "splice_subtree", "replace_subtree", "tree_replace",
    "gp_tree_replace", "_splice", "_tree_replace", "_gpTreeReplace",
]


def find_splice_hook():
    """Locate a module-level splice primitive, if the rewritten engine exposes one."""
    for module_name in SPLICE_HOOK_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in SPLICE_HOOK_NAMES:
            hook = getattr(module, name, None)
            if callable(hook):
                return f"{module_name}.{name}", hook
    return None, None


def normalize_splice_result(result):
    """Best-effort conversion of a hook result into a Row (None when unsupported)."""
    if isinstance(result, Forest):
        return row_of(result, 0)
    if isinstance(result, tuple) and len(result) == 3:
        parts = []
        for part in result:
            # a batch-wise hook (one row per call) returns ``[1, max_tree_len]``
            # tensors -- squash the leading dimension before listing the row
            if callable(getattr(part, "dim", None)) and part.dim() > 1:
                part = part.reshape(-1)
            if hasattr(part, "tolist"):
                part = part.tolist()
            parts.append(part)
        values, types, sizes = parts
        if not (values and types and sizes):
            return None
        total = int(sizes[0])
        if min(len(values), len(types), len(sizes)) >= total:
            return Row(values[:total], types[:total], sizes[:total], total)
    return None


def test_internal_splice_hook_matches_reference_when_exposed():
    """Guard-railed check of an internal splice helper (skipped when not exposed)."""
    path, hook = find_splice_hook()
    if hook is None:
        pytest.skip("no module-level splice helper exposed under the probed names")

    old, donor, index, max_tree_len = T_MED, D_BIG, 10, 32
    expected = reference_splice(old, index, donor, 0, max_tree_len)
    device = api_device()

    def row_tensor(row, dtype):
        return torch.tensor(row, dtype=dtype, device=device)

    # (old value/type/size, index, new value/type/size, new index, max_tree_len)
    row_wise = lambda: hook(
        row_tensor(old.value, torch.float32), row_tensor(old.type, torch.int16),
        row_tensor(old.size, torch.int16), index,
        row_tensor(donor.value, torch.float32), row_tensor(donor.type, torch.int16),
        row_tensor(donor.size, torch.int16), 0, max_tree_len,
    )
    # (forest, indices, donor forest) - the container-wise calling convention
    container_wise = lambda: hook(
        build_forest([old], max_tree_len), index_tensor([index]),
        build_forest([donor], max_tree_len),
    )
    # (value/type/size, pos, donor value/type/size, donor pos, new subtree size)
    # - the vectorized ``_splice`` convention of ``splicing.py``: full-width
    # ``[pop_size, max_tree_len]`` tree tensors plus 1-D *long* index tensors
    def batch_wise():
        target = build_forest([old], max_tree_len)
        source = build_forest([donor], max_tree_len)
        return hook(
            target.batch_node_value, target.batch_node_type, target.batch_subtree_size,
            index_tensor([index]).long(),
            source.batch_node_value, source.batch_node_type, source.batch_subtree_size,
            index_tensor([0]).long(),
            source.batch_subtree_size[:, 0].long(),
        )

    for attempt in (batch_wise, row_wise, container_wise):
        try:
            result = attempt()
        except TypeError:
            continue  # signature does not match this calling convention
        except AttributeError:
            continue  # scalar where the hook expects a tensor (e.g. ``.clamp``)
        row = normalize_splice_result(result)
        assert row is not None, f"{path} returned an unsupported result type {type(result)!r}"
        assert_rows_equal(row, expected, path)
        return
    pytest.skip(f"{path} exists but has an unsupported signature: {inspect.signature(hook)}")


# --- Generation ---------------------------------------------------------------- #
GEN_CASES = {
    "binary_only": dict(using_funcs=["+", "-", "*"], max_layer_cnt=4, const_prob=0.3),
    "with_ternary": dict(using_funcs=["+", "sin", "if"], max_layer_cnt=3, const_prob=0.5),
    "unary_only": dict(using_funcs=["sin", "neg"], max_layer_cnt=4, const_prob=0.5),
    "vars_only": dict(using_funcs=["*"], max_layer_cnt=3, const_prob=0.0),
    "consts_only": dict(using_funcs=["*"], max_layer_cnt=3, const_prob=1.0),
    "leaf_only": dict(using_funcs=["+"], max_layer_cnt=1, const_prob=0.5),
    "deep_wide": dict(
        max_tree_len=256, using_funcs=["+", "-", "*"], max_layer_cnt=5, layer_leaf_prob=0.05
    ),
    "multi_output": dict(
        using_funcs=["+", "*", "sin"], max_layer_cnt=3, output_len=3, out_prob=1.0, const_prob=0.4
    ),
    "single_output_high_out_prob": dict(
        using_funcs=["+", "*"], max_layer_cnt=3, output_len=1, out_prob=1.0
    ),
}

GEN_POP_SIZE = 24
GEN_SEEDS = (0, 1, 7)
DESCRIPTOR_DEFAULTS = dict(
    max_tree_len=64, input_len=3, output_len=1, const_prob=0.5, out_prob=0.5,
    using_funcs=["+", "-", "*"], max_layer_cnt=3, layer_leaf_prob=0.25,
    const_range=(-1.0, 1.0), sample_cnt=8,
)


def make_descriptor(**overrides) -> GenerateDescriptor:
    params = dict(DESCRIPTOR_DEFAULTS)
    params.update(overrides)
    return GenerateDescriptor(**params)


def const_bounds_for(case: dict) -> tuple:
    """Inclusive ``(lo, hi)`` interval a CONST of this descriptor case may fall in.

    Constants are drawn from the ``const_samples`` set (drawn from ``const_range``
    when only a range is configured), so the allowed span is the range itself when
    one was given and the sample span otherwise.
    """
    params = dict(DESCRIPTOR_DEFAULTS)
    params.update(case)
    samples = params.get("const_samples")
    if samples is not None:
        samples = samples.cpu().tolist() if hasattr(samples, "tolist") else list(samples)
        return min(samples), max(samples)
    lo, hi = params["const_range"]
    return lo, hi

def generate(case: dict, seed: int, pop_size: int = GEN_POP_SIZE) -> Forest:
    """Deterministically (re)seed every RNG the generator may use, then generate."""
    descriptor = make_descriptor(**case)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    return Forest.random_generate(pop_size=pop_size, descriptor=descriptor)


def iter_nodes(forest: Forest):
    """Yield ``(node_type, value)`` for every meaningful node of every row."""
    types = forest.batch_node_type.cpu().tolist()
    values = forest.batch_node_value.cpu().tolist()
    for row in range(forest.pop_size):
        for idx in range(int(forest.batch_subtree_size[row, 0])):
            yield types[row][idx], values[row][idx]


@pytest.mark.parametrize("case_id", sorted(GEN_CASES), ids=sorted(GEN_CASES))
@pytest.mark.parametrize("seed", GEN_SEEDS, ids=[f"seed{s}" for s in GEN_SEEDS])
def test_random_generate_rows_are_valid(case_id, seed):
    case = GEN_CASES[case_id]
    forest = generate(case, seed)

    assert forest.pop_size == GEN_POP_SIZE
    assert forest.max_tree_len == case.get("max_tree_len", 64)
    assert (forest.input_len, forest.output_len) == (case.get("input_len", 3), case.get("output_len", 1))
    assert forest.batch_node_value.dtype == torch.float32
    assert forest.batch_node_type.dtype == torch.int16
    assert forest.batch_subtree_size.dtype == torch.int16
    assert forest.batch_node_value.shape == (GEN_POP_SIZE, forest.max_tree_len)

    observed_depth = validate_forest(
        forest,
        input_len=forest.input_len,
        output_len=forest.output_len,
        using_funcs=case["using_funcs"],
        const_bounds=const_bounds_for(case),
        max_layer_cnt=case["max_layer_cnt"],    )
    assert 0 <= observed_depth <= case["max_layer_cnt"]
    for i in range(forest.pop_size):
        assert 1 <= int(forest.batch_subtree_size[i, 0]) <= forest.max_tree_len


def test_random_generate_leaf_only_schedule_produces_single_nodes():
    """``max_layer_cnt == 1`` => the depth-0 leaf schedule is 1.0 (always a leaf)."""
    for seed in GEN_SEEDS:
        forest = generate(GEN_CASES["leaf_only"], seed)
        assert torch.equal(
            forest.batch_subtree_size[:, 0], torch.ones_like(forest.batch_subtree_size[:, 0])
        )


def test_random_generate_binary_schedule_bounds_tree_size():
    """Binary functions + ``max_layer_cnt == 2`` => at most 1 + 2 nodes per tree."""
    forest = generate(dict(using_funcs=["+"], max_layer_cnt=2, const_prob=0.5), seed=3)
    assert int(forest.batch_subtree_size[:, 0].max().item()) <= 3


def test_random_generate_honours_the_function_subset():
    """A single-function descriptor must only produce that function id."""
    func_id = FUNCS_NAMES.index("*")
    forest = generate(dict(using_funcs=["*"], max_layer_cnt=3, const_prob=0.5), seed=5)
    for node_type, value in iter_nodes(forest):
        if _base_type(node_type) in (UFUNC, BFUNC, TFUNC):
            assert int(value) == func_id


def test_random_generate_vars_only_and_consts_only():
    vars_forest = generate(GEN_CASES["vars_only"], seed=11)
    assert {_base_type(t) for t, _ in iter_nodes(vars_forest)} <= {VAR, BFUNC}

    const_forest = generate(GEN_CASES["consts_only"], seed=11)
    constants = [value for node_type, value in iter_nodes(const_forest) if _base_type(node_type) == CONST]
    assert constants, "const_prob == 1.0 must produce constant leaves"
    lo, hi = const_bounds_for(GEN_CASES["consts_only"])
    assert all(lo - 1e-6 <= value <= hi + 1e-6 for value in constants)

def test_random_generate_multi_output_sets_out_nodes():
    """``out_prob == 1.0`` tags every function node; ``output_len == 1`` never tags."""
    case = GEN_CASES["multi_output"]
    forest = generate(case, seed=2)
    output_len = case["output_len"]
    func_ids = {FUNCS_NAMES.index(f) for f in case["using_funcs"]}

    tagged = 0
    for node_type, value in iter_nodes(forest):
        if _base_type(node_type) in (UFUNC, BFUNC, TFUNC):
            assert node_type & OUT_NODE, "out_prob == 1.0 must tag every function node"
            tagged += 1
            bits = int(torch.tensor([value], dtype=torch.float32).view(torch.int32)[0])
            assert 0 <= bits >> 16 < output_len, "out index out of range"
            assert bits & 0xFF in func_ids, "out node function outside using_funcs"
    assert tagged > 0, "multi-output generation produced no function nodes"

    single = generate(GEN_CASES["single_output_high_out_prob"], seed=2)
    assert not (single.batch_node_type.cpu().to(torch.int32) & OUT_NODE).any()


def test_random_generate_is_reproducible_after_reseeding():
    descriptor = make_descriptor(using_funcs=["+", "sin", "if"], max_layer_cnt=3)

    def run():
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        random.seed(0)
        return Forest.random_generate(pop_size=16, descriptor=descriptor)

    first, second = run(), run()

    assert torch.equal(first.batch_subtree_size[:, 0].cpu(), second.batch_subtree_size[:, 0].cpu())
    for i in range(first.pop_size):
        assert_rows_equal(row_of(second, i), row_of(first, i), f"row {i}")


@pytest.mark.parametrize("pop_size", [0, -1])
def test_random_generate_rejects_non_positive_pop_size(pop_size):
    with pytest.raises(Exception):
        Forest.random_generate(pop_size=pop_size, descriptor=make_descriptor())
