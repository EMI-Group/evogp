"""Input-dict contract regression tests for the combined multi-output containers.

``CombinedForest.forward`` / ``CombinedForest.batch_forward`` and
``CombinedTree.forward`` share ONE shape contract, implemented in
``evogp.tree.combined_forest.normalize_inputs`` (whose docstring is the contract of
record, mirrored by the two ``forward`` docstrings):

* a Python/NumPy scalar number or a 0-D tensor (one value, ``K == 1``),
* a 1-D tensor of shape ``(K,)``,
* a 2-D tensor of shape ``(K, 1)`` (the column form of the same ``(K,)`` values),
* every other ``x`` entry is ignored completely, and values are NEVER broadcast.

``CombinedTree.forward`` picks the method from ``K`` (``K == 1`` is one shared data
point, ``K > 1`` is a batch), while ``CombinedForest.forward`` is single-point only
and ``CombinedForest.batch_forward`` is batch only.  This file pins that behaviour,
including the exact ``ValueError`` messages of each rejection, plus the
population-level shapes ``(pop_size, output_len)`` / ``(pop_size, K, output_len)``
that the loss code reads.

Run from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest \\
        src/evogp/tests/test_combined_forward.py -q
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch

try:  # the tree layer is importable only once the evoxc-based engine is in place
    from evogp.tree import CombinedForest, CombinedTree, NType, Tree

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only in pre-merge worktrees
    CombinedForest = CombinedTree = NType = Tree = None
    _IMPORT_ERROR = exc


@pytest.fixture(autouse=True)
def _require_engine():
    require_engine()


def require_engine():
    """Fail fast (with one clear message) while the tree layer is not importable."""
    if _IMPORT_ERROR is not None:
        pytest.fail(
            "evogp.tree is not importable (the evoxc-based engine is not present in this "
            f"worktree yet): {_IMPORT_ERROR!r}"
        )


def needs(*parts: str) -> str:
    """Regex matching the literal ``parts`` in order, for ``pytest.raises(match=...)``.

    ``re.escape`` keeps regex specials of the messages (``[``, ``(``, ``)``, ``.``)
    literal; ``.*`` bridges the text between two asserted fragments.
    """
    return ".*".join(re.escape(part) for part in parts)


DEVICE = "cuda"
# function ids: FUNCS_NAMES == ["if", "+", "-", "*", ...]
FUNC_ADD, FUNC_MUL = 1, 3
# one value per output of the single-point fixture; one row per output of a batch
SINGLE_A, SINGLE_B = [3.0], [6.0]
BATCH_A, BATCH_B = [[3.0], [4.0], [5.0]], [[6.0], [9.0], [12.0]]


def mk(nodes, input_len, output_len=1, max_len=16, device=DEVICE):
    """Build a one-tree ``Tree`` from ``(node_type, node_value, subtree_size)`` rows."""
    nv = torch.zeros(max_len, dtype=torch.float32, device=device)
    nt = torch.zeros(max_len, dtype=torch.int16, device=device)
    st = torch.zeros(max_len, dtype=torch.int16, device=device)
    for i, (t, v, s) in enumerate(nodes):
        nv[i], nt[i], st[i] = v, t, s
    return Tree(input_len, output_len, nv, nt, st)


def tensor(values, device=DEVICE):
    return torch.tensor(values, dtype=torch.float32, device=device)


def values(res, out_name):
    """Detached nested lists of one output tensor (keeps the exact float values)."""
    return res[out_name].detach().cpu().tolist()


# --------------------------------------------------------------------------- #
# Fixtures: two trees with DIFFERENT input arities ("A" reads x0,x1; "B" reads x1)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data_info():
    return {"A": ["x0", "x1"], "B": ["x1"]}


@pytest.fixture(scope="module")
def trees():
    a = mk([(NType.BFUNC, FUNC_ADD, 3), (NType.VAR, 0, 1), (NType.CONST, 2.0, 1)], 2)
    b = mk([(NType.BFUNC, FUNC_MUL, 3), (NType.VAR, 0, 1), (NType.CONST, 3.0, 1)], 1)
    return a, b


@pytest.fixture(scope="module")
def combined_tree(trees, data_info):
    """A = x0 + 2.0, B = x1 * 3.0."""
    return CombinedTree(list(trees), data_info)


@pytest.fixture(scope="module")
def combined_forest(combined_tree):
    """The single-individual forest view of ``combined_tree`` (pop_size == 1)."""
    return combined_tree.to_combined_forest()


@pytest.fixture(scope="module")
def batch():
    """Three data points: (x0, x1) == (1, 2), (2, 3), (3, 4)."""
    return {"x0": tensor([1.0, 2.0, 3.0]), "x1": tensor([2.0, 3.0, 4.0])}


@pytest.fixture(scope="module")
def pop2_forest(trees, data_info):
    """A population of two: A = (x0 + 2.0, x0 + 3.0), B = (x1 * 3.0, x1 * 5.0)."""
    a2 = mk([(NType.BFUNC, FUNC_ADD, 3), (NType.VAR, 0, 1), (NType.CONST, 3.0, 1)], 2)
    b2 = mk([(NType.BFUNC, FUNC_MUL, 3), (NType.VAR, 0, 1), (NType.CONST, 5.0, 1)], 1)
    return CombinedForest(
        [trees[0].to_forest() + a2.to_forest(), trees[1].to_forest() + b2.to_forest()],
        data_info,
    )


# --------------------------------------------------------------------------- #
# Single point: every accepted spelling, plus ignored extra keys
# --------------------------------------------------------------------------- #
def test_single_point_spellings_agree(combined_tree, combined_forest):
    """python floats, 0-D, (1,) and (1, 1) tensors all describe the same one point."""
    assert (combined_tree.input_len, combined_tree.output_len) == (2, 2)

    spellings = {
        "python floats": {"x0": 1.0, "x1": 2.0},
        "0-D tensors": {"x0": tensor(1.0), "x1": tensor(2.0)},
        "(1,) tensors": {"x0": tensor([1.0]), "x1": tensor([2.0])},
        "(1, 1) tensors": {"x0": tensor([[1.0]]), "x1": tensor([[2.0]])},
    }
    for label, x in spellings.items():
        res = combined_tree.forward(x)
        assert set(res) == {"A", "B"}, label
        for out_name, expected in (("A", SINGLE_A), ("B", SINGLE_B)):
            assert tuple(res[out_name].shape) == (1,), label
            assert values(res, out_name) == expected, label

        # CombinedForest.forward keeps the leading pop dimension (pop_size == 1)
        pop_res = combined_forest.forward(x)
        for out_name, expected in (("A", SINGLE_A), ("B", SINGLE_B)):
            assert tuple(pop_res[out_name].shape) == (1, 1), label
            assert values(pop_res, out_name) == [expected], label


def test_extra_dict_entries_are_ignored(combined_tree, combined_forest, batch):
    """Extra ``x`` keys are never read: ``CustomLoss`` passes its data dict through."""
    point = {"x0": 1.0, "x1": 2.0}
    # a 3-D tensor would be rejected if it were validated as an input
    with_target = {**point, "target": torch.zeros(3, 3, 3, device=DEVICE)}
    base, extra = combined_tree.forward(point), combined_tree.forward(with_target)
    for out_name, expected in (("A", SINGLE_A), ("B", SINGLE_B)):
        assert values(extra, out_name) == expected
        assert torch.equal(extra[out_name], base[out_name])

    # batch path: the ignored key may even disagree on K with the real inputs
    base_batch = combined_forest.batch_forward(batch)
    extra_batch = combined_forest.batch_forward({**batch, "target": torch.zeros(4, 4, device=DEVICE)})
    for out_name, expected in (("A", BATCH_A), ("B", BATCH_B)):
        assert tuple(extra_batch[out_name].shape) == (1, 3, 1)
        assert values(extra_batch, out_name) == [expected]
        assert torch.equal(extra_batch[out_name], base_batch[out_name])


# --------------------------------------------------------------------------- #
# Batched data: shapes, values, per-row equivalence and the column form
# --------------------------------------------------------------------------- #
def test_batch_forward_shapes_and_values(combined_tree, combined_forest, batch):
    """K == 3: the individual gets (K, output_len), the forest (1, K, output_len)."""
    res = combined_tree.forward(batch)
    for out_name, expected in (("A", BATCH_A), ("B", BATCH_B)):
        assert tuple(res[out_name].shape) == (3, 1)
        assert values(res, out_name) == expected

    pop_res = combined_forest.batch_forward(batch)
    for out_name, expected in (("A", BATCH_A), ("B", BATCH_B)):
        assert tuple(pop_res[out_name].shape) == (1, 3, 1)
        assert values(pop_res, out_name) == [expected]
        # one individual: the wrapped result is exactly the pop-free one
        assert torch.equal(pop_res[out_name][0], res[out_name])

    single = combined_forest.forward({"x0": 1.0, "x1": 2.0})
    for out_name, expected in (("A", SINGLE_A), ("B", SINGLE_B)):
        assert tuple(single[out_name].shape) == (1, 1)
        assert values(single, out_name) == [expected]


def test_batch_forward_row_slice_is_readable(combined_forest, batch):
    """The loss code reads ``res[name][0, :, 0]`` out of the (1, K, 1) batch result."""
    res = combined_forest.batch_forward(batch)
    assert values(res, "A") == [BATCH_A]
    assert res["A"][0, :, 0].detach().cpu().tolist() == [3.0, 4.0, 5.0]
    assert res["B"][0, :, 0].detach().cpu().tolist() == [6.0, 9.0, 12.0]
    assert tuple(res["A"][0].shape) == (3, 1)


def test_batch_matches_per_row_single_point_evaluation(combined_tree, batch):
    """Row r of a batch result is the same individual evaluated at row r alone."""
    batched = combined_tree.forward(batch)
    for row in range(batch["x0"].shape[0]):
        point = {name: batch[name][row] for name in ("x0", "x1")}
        single = combined_tree.forward(point)
        for out_name in ("A", "B"):
            assert tuple(single[out_name].shape) == (1,)
            assert torch.equal(single[out_name], batched[out_name][row])


def test_column_form_equals_flat_batch(combined_tree, combined_forest, batch):
    """(K, 1) is the same column of values as (K,), for both containers."""
    column = {"x0": tensor([[1.0], [2.0], [3.0]]), "x1": tensor([[2.0], [3.0], [4.0]])}
    tree_flat, tree_column = combined_tree.forward(batch), combined_tree.forward(column)
    forest_flat = combined_forest.batch_forward(batch)
    forest_column = combined_forest.batch_forward(column)
    for out_name, expected in (("A", BATCH_A), ("B", BATCH_B)):
        assert values(tree_column, out_name) == expected
        assert torch.equal(tree_column[out_name], tree_flat[out_name])
        assert tuple(forest_column[out_name].shape) == (1, 3, 1)
        assert torch.equal(forest_column[out_name], forest_flat[out_name])


# --------------------------------------------------------------------------- #
# A population of several individuals
# --------------------------------------------------------------------------- #
def test_population_forward_and_batch_forward(pop2_forest):
    """pop_size == 2: forward broadcasts one point, batch_forward evaluates K."""
    assert pop2_forest.pop_size == 2

    res = pop2_forest.forward({"x0": 3.0, "x1": 4.0})
    # individual 0 = (x0 + 2.0, x1 * 3.0), individual 1 = (x0 + 3.0, x1 * 5.0)
    for out_name, expected in (("A", [[5.0], [6.0]]), ("B", [[12.0], [20.0]])):
        assert tuple(res[out_name].shape) == (2, 1)
        assert values(res, out_name) == expected

    batched = pop2_forest.batch_forward({"x0": tensor([1.0, 3.0]), "x1": tensor([2.0, 4.0])})
    for out_name, expected in (
        ("A", [[[3.0], [5.0]], [[4.0], [6.0]]]),
        ("B", [[[6.0], [12.0]], [[10.0], [20.0]]]),
    ):
        assert tuple(batched[out_name].shape) == (2, 2, 1)
        assert values(batched, out_name) == expected
        # row i of the population scored at both data points: [i][0] is the first one
        assert batched[out_name][0, 0].detach().cpu().tolist() == expected[0][0]

    # the (K, 1) column form must agree with the (K,) form for a population too
    column = pop2_forest.batch_forward({"x0": tensor([[1.0], [3.0]]), "x1": tensor([[2.0], [4.0]])})
    for out_name in ("A", "B"):
        assert torch.equal(column[out_name], batched[out_name])


# --------------------------------------------------------------------------- #
# Rejected inputs: every error names the offending input and its sub-forest
# --------------------------------------------------------------------------- #
def test_rejects_unsupported_value_shapes(combined_tree):
    for bad in (torch.zeros(2, 1, 1, device=DEVICE), torch.zeros(3, 2, device=DEVICE)):
        with pytest.raises(
            ValueError,
            match=needs(
                "x['x0']",
                "'A'",
                "must be a scalar (0-D) or a 1-D (K,) / 2-D (K, 1) tensor",
            ),
        ):
            combined_tree.forward({"x0": bad, "x1": 2.0})


def test_rejects_missing_required_input(combined_tree, combined_forest):
    with pytest.raises(ValueError, match=needs("x is missing the input 'x1' required by 'A'")):
        combined_tree.forward({"x0": 1.0})
    with pytest.raises(ValueError, match=needs("x is missing the input 'x0' required by 'A'")):
        combined_forest.batch_forward({"x1": 2.0})


def test_rejects_disagreeing_numbers_of_data_points(combined_tree):
    """Values are never broadcast: a lone scalar next to K > 1 names is an error."""
    with pytest.raises(
        ValueError,
        match=needs(
            "x['x1']",
            "all input names must agree on the number of data points (values are never broadcast)",
        ),
    ):
        combined_tree.forward({"x0": tensor([1.0, 2.0, 3.0]), "x1": 2.0})


def test_forest_forward_rejects_a_batch(combined_forest):
    """``forward`` evaluates ONE point for the whole population."""
    with pytest.raises(
        ValueError,
        match=needs("x['x0']", "use batch_forward() for a batch of data points"),
    ):
        combined_forest.forward({"x0": tensor([1.0, 2.0]), "x1": tensor([1.0, 2.0])})


def test_rejects_non_tensor_values(combined_tree):
    """Only scalars and tensors are inputs: 1-D numpy arrays and plain lists are not."""
    with pytest.raises(
        ValueError,
        match=needs("x['x0']", "a numpy array of shape"),
    ):
        combined_tree.forward({"x0": np.array([1.0, 2.0]), "x1": 2.0})

    # a 0-D numpy scalar IS an accepted spelling (a Python number)
    res = combined_tree.forward({"x0": np.float64(1.0), "x1": np.float64(2.0)})
    assert values(res, "A") == SINGLE_A and values(res, "B") == SINGLE_B

    with pytest.raises(
        ValueError,
        match=needs(
            "x['x0']",
            "must be a scalar (0-D) or a 1-D (K,) / 2-D (K, 1) tensor",
        ),
    ):
        combined_tree.forward({"x0": [1.0], "x1": 2.0})
