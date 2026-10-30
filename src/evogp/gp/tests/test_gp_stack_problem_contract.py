"""Contract tests of the numpy evaluator and the problem layer.

Everything here is hand-built or hand-checked -- no oracle -- so the file pins
the *semantics* the parity suite only measures: the operand order, the ``if``
select, the protected / loose edges, the ``+-1`` comparison clamps, the
``max``/``min`` NaN asymmetry, the maximize-and-negate fitness convention, the
fresh-writable output, the ``VAR`` bounds error, the non-finite -> ``-inf``
mapping and the memory-bounded row chunking.
"""

import numpy as np
import pytest

from evogp.gp._constants import Func, NType
from evogp.gp.evaluator import predict
from evogp.gp.problem import CustomLoss, SymbolicRegression

VAR = NType.VAR
CONST = NType.CONST


# --------------------------------------------------------------------------- #
# hand-built populations
# --------------------------------------------------------------------------- #
def _emit(spec, opcodes, payloads, sizes) -> int:
    """Flatten one nested spec into prefix order; returns its node count."""
    if spec[0] == "var":
        opcodes.append(VAR)
        payloads.append(float(spec[1]))
        sizes.append(1)
        return 1
    if spec[0] == "const":
        opcodes.append(CONST)
        payloads.append(float(spec[1]))
        sizes.append(1)
        return 1
    index = len(opcodes)
    opcodes.append(int(spec[0]))
    payloads.append(0.0)
    sizes.append(0)
    total = 1
    for child in spec[1:]:
        total += _emit(child, opcodes, payloads, sizes)
    sizes[index] = total
    return total


def population(*trees, width=None):
    """``(opcode, payload, size)`` channels of hand-written trees.

    A leaf is ``("var", index)`` or ``("const", value)``; a function node is
    ``(Func.ID, child, ...)``.
    """
    rows = []
    for tree in trees:
        opcodes, payloads, sizes = [], [], []
        _emit(tree, opcodes, payloads, sizes)
        rows.append((opcodes, payloads, sizes))
    if width is None:
        width = max(len(row[0]) for row in rows)
    pop = len(rows)
    opcode = np.zeros((pop, width), np.int32)
    payload = np.zeros((pop, width), np.float32)
    size = np.zeros((pop, width), np.int32)
    for index, (opcodes, payloads, sizes) in enumerate(rows):
        count = len(opcodes)
        opcode[index, :count] = opcodes
        payload[index, :count] = payloads
        size[index, :count] = sizes
    return opcode, payload, size


def var(index):
    return ("var", index)


def const(value):
    return ("const", value)


# --------------------------------------------------------------------------- #
# evaluator semantics
# --------------------------------------------------------------------------- #
def test_closed_form_tree_is_evaluated_by_hand():
    """``2 * x0 + x1`` built node by node."""
    channels = population(
        (Func.ADD, (Func.MUL, var(0), const(2.0)), var(1)),
        width=6,
    )
    x = np.asarray([[3.0, 5.0, 0.0], [-1.5, 0.25, 7.0]], dtype=np.float32)
    got = predict(channels, x)
    expected = np.float32(2.0) * x[:, 0] + x[:, 1]
    np.testing.assert_array_equal(got, expected[None, :])


def test_the_first_child_is_the_left_operand():
    """``-``, ``/`` and ``if`` all take the FIRST child as operand 1."""
    x = np.asarray([[0.5, 2.0, 0.0], [-0.5, 2.0, 0.0], [0.25, 4.0, 0.0]], dtype=np.float32)
    sub = population((Func.SUB, var(0), var(1)), width=4)
    np.testing.assert_array_equal(predict(sub, x)[0], x[:, 0] - x[:, 1])
    div = population((Func.DIV, var(1), var(0)), width=4)
    np.testing.assert_array_equal(predict(div, x)[0], x[:, 1] / x[:, 0])
    # if(cond, then, else): the condition is the first child and is tested > 0
    ternary = population((Func.IF, var(0), var(1), const(9.0)), width=8)
    np.testing.assert_array_equal(
        predict(ternary, x)[0], np.where(x[:, 0] > 0, x[:, 1], np.float32(9.0))
    )


def test_protected_and_loose_operators():
    x = np.asarray([[0.0, -4.0]], dtype=np.float32)
    cases = {
        # (tree, expected value(s), "exact"/"nan"/"allclose")
        "div0": ((Func.DIV, const(1.0), const(0.0)), np.nan),
        "inv0": ((Func.INV, const(0.0)), np.nan),
        "log0": ((Func.LOG, const(0.0)), -np.inf),
        "sqrt_neg": ((Func.SQRT, const(-4.0)), np.nan),
        "loose_log0": ((Func.LOOSE_LOG, const(0.0)), np.float32(-1e9)),
        "loose_inv0": ((Func.LOOSE_INV, const(0.0)), np.float32(1e9)),
        "loose_inv_neg0": ((Func.LOOSE_INV, (Func.MUL, const(-1.0), const(0.0))), np.float32(-1e9)),
        "loose_sqrt_neg": ((Func.LOOSE_SQRT, const(-4.0)), np.float32(2.0)),
        "pow00": ((Func.POW, const(0.0), const(0.0)), np.float32(1.0)),
        "loose_pow00": ((Func.LOOSE_POW, const(0.0), const(0.0)), np.float32(0.0)),
        "loose_div0": ((Func.LOOSE_DIV, const(2.0), const(0.0)), np.float32(2.0) / np.float32(1e-9)),
        "loose_div_neg0": (
            (Func.LOOSE_DIV, const(2.0), (Func.MUL, const(-1.0), const(0.0))),
            np.float32(2.0) / np.float32(-1e-9),
        ),
    }
    trees = [spec for spec, _ in cases.values()]
    expected = [value for _, value in cases.values()]
    channels = population(*trees)
    got = predict(channels, x)
    for index, (name, (_, value)) in enumerate(cases.items()):
        if isinstance(value, float) and np.isnan(value):
            assert np.isnan(got[index, 0]), name
        elif isinstance(value, float) and np.isinf(value):
            assert got[index, 0] == value, name
        else:
            assert got[index, 0] == value, (name, got[index, 0], value)
    assert got.shape == (len(trees), 1)


def test_comparisons_return_the_confidence_clamp():
    """``< > <= >=`` answer exactly +-1; a false test (NaN included) answers -1."""
    x = np.asarray([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    nan_tree = (Func.DIV, const(0.0), const(0.0))  # 0/0 -> NaN
    channels = population(
        (Func.LT, var(0), var(1)),
        (Func.GT, var(0), var(1)),
        (Func.LE, var(0), var(1)),
        (Func.GE, var(0), var(1)),
        (Func.LT, nan_tree, const(1.0)),
        (Func.GE, nan_tree, const(1.0)),
    )
    got = predict(channels, x)
    np.testing.assert_array_equal(got[0], np.where(x[:, 0] < x[:, 1], 1.0, -1.0))
    np.testing.assert_array_equal(got[1], np.where(x[:, 0] > x[:, 1], 1.0, -1.0))
    np.testing.assert_array_equal(got[2], np.where(x[:, 0] <= x[:, 1], 1.0, -1.0))
    np.testing.assert_array_equal(got[3], np.where(x[:, 0] >= x[:, 1], 1.0, -1.0))
    np.testing.assert_array_equal(got[4], np.full(3, -1.0))  # NaN < 1 is false
    np.testing.assert_array_equal(got[5], np.full(3, -1.0))  # NaN >= 1 is false


def test_max_and_min_are_ternaries_not_fmax():
    """``MAX``/``MIN`` are ternaries: the NaN asymmetry of ENGINE_SPEC section 3."""
    nan_tree = (Func.DIV, const(0.0), const(0.0))
    x = np.zeros((2, 1), dtype=np.float32)
    channels = population(
        (Func.MAX, nan_tree, const(1.0)),  # max(NaN, 1) = 1  (the test fails -> b)
        (Func.MAX, const(1.0), nan_tree),  # max(1, NaN) = NaN
        (Func.MIN, nan_tree, const(1.0)),  # min(NaN, 1) = 1
        (Func.MIN, const(1.0), nan_tree),  # min(1, NaN) = NaN
        (Func.MAX, const(2.0), const(3.0)),
        (Func.MIN, const(2.0), const(3.0)),
    )
    got = predict(channels, x)
    assert got[0, 0] == 1.0
    assert np.isnan(got[1, 0])
    assert got[2, 0] == 1.0
    assert np.isnan(got[3, 0])
    assert got[4, 0] == 3.0
    assert got[5, 0] == 2.0


def test_unary_and_binary_dispatch_by_id():
    """The transcendental emitter agrees with a float64-then-round by hand."""
    x = np.asarray([[0.3, -0.7]], dtype=np.float32)
    expected_sin = np.sin(x[:, 0].astype(np.float64)).astype(np.float32)
    expected_tanh = np.tanh(x[:, 1].astype(np.float64)).astype(np.float32)
    expected_pow = np.power(
        x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    ).astype(np.float32)
    channels = population(
        (Func.SIN, var(0)),
        (Func.TANH, var(1)),
        (Func.POW, var(0), var(1)),
        (Func.LOOSE_POW, var(1), var(0)),  # pow(|a|, b)
        (Func.EXP, var(0)),
        (Func.ABS, var(1)),
        (Func.NEG, var(0)),
    )
    got = predict(channels, x)
    np.testing.assert_array_equal(got[0], expected_sin)
    np.testing.assert_array_equal(got[1], expected_tanh)
    np.testing.assert_array_equal(got[2], expected_pow)
    np.testing.assert_array_equal(
        got[3], np.power(np.abs(x[:, 1].astype(np.float64)), x[:, 0].astype(np.float64)).astype(np.float32)
    )
    np.testing.assert_array_equal(got[4], np.exp(x[:, 0].astype(np.float64)).astype(np.float32))
    np.testing.assert_array_equal(got[5], np.abs(x[:, 1]))
    np.testing.assert_array_equal(got[6], -x[:, 0])


# --------------------------------------------------------------------------- #
# predict's contracts
# --------------------------------------------------------------------------- #
def test_predict_shape_dtype_and_state_input():
    channels = population((Func.ADD, var(0), const(1.0)), width=4)
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    got = predict(channels, x)
    assert got.shape == (1, 3) and got.dtype == np.float32
    # a 1-D data array is a single row
    assert predict(channels, x[0]).shape == (1, 1)


def test_variable_index_out_of_range_raises_value_error():
    channels = population((Func.ADD, var(2), const(1.0)), width=4)
    x = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="row 0 node 1"):
        predict(channels, x)
    with pytest.raises(ValueError, match="input_len"):
        predict(channels, np.zeros((2, 3), dtype=np.float32), input_len=2)


def test_a_non_integral_variable_index_raises():
    opcode = np.asarray([[VAR]], np.int32)
    payload = np.asarray([[0.5]], np.float32)
    size = np.asarray([[1]], np.int32)
    with pytest.raises(ValueError, match="not an integer"):
        predict((opcode, payload, size), np.zeros((2, 2), dtype=np.float32))


def test_malformed_channels_are_rejected():
    x = np.zeros((1, 2), dtype=np.float32)
    # a leaf (subtree size 1) carrying a function opcode: SUB is neither VAR nor
    # the CONST marker, so it cannot be read as a leaf
    with pytest.raises(ValueError, match="leaf"):
        predict(
            (
                np.asarray([[Func.SUB]], np.int32),
                np.asarray([[0.0]], np.float32),
                np.asarray([[1]], np.int32),
            ),
            x,
        )
    # a root whose declared subtree size disagrees with its children
    with pytest.raises(ValueError, match="too small for"):
        predict(
            (
                np.asarray([[Func.ADD, VAR, VAR]], np.int32),
                np.asarray([[0.0, 0.0, 1.0]], np.float32),
                np.asarray([[2, 1, 1]], np.int32),
            ),
            x,
        )
    # a row whose declared length exceeds the row width
    with pytest.raises(ValueError, match="exceeds max_nodes"):
        predict(
            (
                np.asarray([[VAR, 0, 0]], np.int32),
                np.asarray([[0.0, 0.0, 0.0]], np.float32),
                np.asarray([[4, 1, 1]], np.int32),
            ),
            x,
        )
    # shape mismatches
    with pytest.raises(ValueError, match="share one shape"):
        predict(
            (
                np.zeros((1, 2), np.int32),
                np.zeros((1, 3), np.float32),
                np.zeros((1, 2), np.int32),
            ),
            x,
        )


def test_out_node_tagged_opcodes_are_not_implemented():
    opcode = np.asarray([[NType.UFUNC_OUT]], np.int32)
    payload = np.asarray([[0.0]], np.float32)
    size = np.asarray([[1]], np.int32)
    with pytest.raises(NotImplementedError, match="OUT_NODE"):
        predict((opcode, payload, size), np.zeros((1, 2), dtype=np.float32))


def test_row_chunking_is_value_identical_and_budget_driven():
    rng = np.random.default_rng(7)
    from evogp.gp import GPTreeConfig, init_population

    config = GPTreeConfig(
        max_nodes=64,
        input_len=3,
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=4,
        layer_leaf_prob=0.25,
        const_samples=(0.5, -1.0, 2.0),
    )
    state = init_population(config, 3, pop_size=8)
    x = rng.uniform(-1, 1, size=(11, 3)).astype(np.float32)
    whole = predict(state, x)
    for chunk in (1, 2, 5, 10):
        np.testing.assert_array_equal(predict(state, x, row_chunk=chunk), whole)
    # a tiny budget forces the same chunked path without an explicit row_chunk
    np.testing.assert_array_equal(predict(state, x, budget_mb=1e-6), whole)
    with pytest.raises(ValueError, match="row_chunk"):
        predict(state, x, row_chunk=0)
    with pytest.raises(ValueError, match="budget_mb"):
        predict(state, x, budget_mb=0)


def test_validate_false_is_the_escape_hatch():
    channels = population((Func.ADD, var(0), const(1.5)), width=4)
    x = np.asarray([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        predict(channels, x, validate=False), predict(channels, x)
    )


# --------------------------------------------------------------------------- #
# the problem layer
# --------------------------------------------------------------------------- #
def test_fitness_is_the_negated_mean_squared_error():
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float32)  # y = 2x + 1
    good = (Func.ADD, (Func.MUL, var(0), const(2.0)), const(1.0))
    bad = const(0.0)
    problem = SymbolicRegression(x, y)
    fitness = problem.evaluate(population(good, bad, width=8))
    expected_good = 0.0
    expected_bad = -float(np.mean(y.astype(np.float64) ** 2))
    assert fitness.shape == (2,) and fitness.dtype == np.float32
    assert fitness[0] == pytest.approx(expected_good, abs=1e-6)
    assert fitness[1] == pytest.approx(expected_bad, rel=1e-6)
    # the lowest-MSE tree has the HIGHEST fitness -> argmax picks it
    assert int(np.argmax(fitness)) == 0


def test_mae_fitness_matches_the_absolute_error():
    x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = np.asarray([1.0, 1.0], dtype=np.float32)
    channels = population(const(0.0), width=2)
    mse = SymbolicRegression(x, y).evaluate(channels)[0]
    mae = SymbolicRegression(x, y, use_mse=False).evaluate(channels)[0]
    assert mse == pytest.approx(-1.0)
    assert mae == pytest.approx(-1.0)  # |0-1| = 1 for both rows
    y2 = np.asarray([0.0, 2.0], dtype=np.float32)
    assert SymbolicRegression(x, y2).evaluate(channels)[0] == pytest.approx(-2.0)
    assert SymbolicRegression(x, y2, use_mse=False).evaluate(channels)[0] == pytest.approx(-1.0)


def test_fitness_is_fresh_and_writable():
    x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = np.asarray([1.0, 2.0], dtype=np.float32)
    problem = SymbolicRegression(x, y)
    channels = population((Func.ADD, var(0), const(1.0)), width=4)
    first = problem.evaluate(channels)
    assert first.flags.writeable
    before = first.copy()
    first[:] = -1e9  # the pipeline mutates the fitness in place
    second = problem.evaluate(channels)
    assert second is not first
    np.testing.assert_array_equal(second, before)
    # the predictions are a fresh buffer too (never a view into the channels)
    predictions = problem.predict(channels)
    predictions[:] = 123.0
    np.testing.assert_array_equal(problem.predict(channels), (x[:, 0] + 1.0)[None, :])


def test_a_non_finite_loss_becomes_minus_infinity():
    x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = np.asarray([1.0, 2.0], dtype=np.float32)
    problem = SymbolicRegression(x, y)
    broken = population(
        (Func.DIV, const(1.0), const(0.0)),  # 1/0 -> NaN
        (Func.MUL, const(1e30), const(1e30)),  # squared error overflows
        width=4,
    )
    fitness = problem.evaluate(broken)
    assert np.all(np.isneginf(fitness)), fitness
    assert not np.isnan(fitness).any()


def test_problem_dimensions_and_data_validation():
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)
    problem = SymbolicRegression(x, y)
    assert problem.problem_dim == 2 and problem.solution_dim == 1
    with pytest.raises(ValueError, match="y has 1 entries"):
        SymbolicRegression(x, np.zeros(1, dtype=np.float32))
    with pytest.raises(ValueError, match="x must be 2-D"):
        SymbolicRegression(np.zeros(3, dtype=np.float32), y)
    with pytest.raises(ValueError, match="must not be empty"):
        SymbolicRegression(np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32))


def test_the_problem_is_callable_as_an_evaluate_hook():
    x = np.asarray([[0.0]], dtype=np.float32)
    y = np.asarray([1.0], dtype=np.float32)
    problem = SymbolicRegression(x, y)
    channels = population((Func.ADD, var(0), const(1.0)), width=4)
    np.testing.assert_array_equal(problem(channels), problem.evaluate(channels))


def test_custom_loss_with_and_without_targets():
    x = np.asarray([[1.0], [2.0]], dtype=np.float32)
    y = np.asarray([1.0, 0.0], dtype=np.float32)
    channels = population((Func.ADD, var(0), const(1.0)), width=4)

    def two_arg(predictions, targets):
        return np.mean((targets[None, :] - predictions) ** 2, axis=1)

    def one_arg(predictions):
        return np.abs(predictions).sum(axis=1)

    with_targets = CustomLoss(x, two_arg, y)
    assert with_targets.problem_dim == 1 and with_targets.solution_dim == 1
    np.testing.assert_array_equal(
        with_targets.evaluate(channels), SymbolicRegression(x, y).evaluate(channels)
    )
    without = CustomLoss(x, one_arg)
    # |x + 1| summed over the two rows
    assert without.evaluate(channels)[0] == pytest.approx(-(2.0 + 3.0))


def test_custom_loss_contract_errors():
    x = np.zeros((2, 1), dtype=np.float32)
    channels = population((Func.ADD, var(0), const(0.0)), width=4)
    with pytest.raises(TypeError, match="callable"):
        CustomLoss(x, 3)
    with pytest.raises(ValueError, match="rows but targets has"):
        CustomLoss(x, lambda p, t: p, np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="returned 2 values"):
        CustomLoss(x, lambda p: np.asarray([1.0, 2.0], np.float32)).evaluate(channels)
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        CustomLoss(x, lambda p: p, np.zeros((2, 1, 1), dtype=np.float32))


def test_custom_loss_is_the_multi_output_escape_hatch():
    """One loss over both predicted columns (the combined-tree shape)."""
    x = np.asarray([[1.0, 2.0]], dtype=np.float32)
    targets = np.asarray([[1.0, 3.0]], dtype=np.float32)
    # two "outputs" read out of two different sub-expressions of one tree
    channels = population(
        (Func.ADD, var(0), var(1)),
        (Func.SUB, var(1), var(0)),
        width=4,
    )
    predictions = predict(channels, x)
    assert predictions.shape == (2, 1)

    def combined(pred, target):
        # individual p is scored against target column p % 2 -- arbitrary
        # multi-output math, which is exactly what this escape hatch is for
        return np.asarray(
            [
                np.mean((pred[p] - target[:, p % target.shape[1]]) ** 2)
                for p in range(pred.shape[0])
            ],
            dtype=np.float32,
        )

    problem = CustomLoss(x, combined, targets)
    fitness = problem.evaluate(channels)
    assert fitness.shape == (2,) and fitness.dtype == np.float32
    assert np.isfinite(fitness).all()
    assert problem.solution_dim == 2
