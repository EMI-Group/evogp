"""Parity of ``evogp.gp.evaluator.predict`` against the plain-torch oracle.

The oracle is ``evogp.tests.reference_eval`` -- the independent, slow,
obviously-correct evaluator the torch stack's own equivalence suite uses
(``src/evogp/tests/test_equivalence.py``).  Torch appears here ONLY as that
oracle: the subtree itself stays torch-free (``test_gp_stack_purity.py``).

Each case draws a real population with ``init_population`` (so the encoding,
its padded growth and its pre-order scatter are under test too), converts it to
the tree layer's ``(value, type, size)`` triple with
``encoding.to_legacy_triple`` and compares the two evaluators on the same data.

Cases cover: arithmetic only (bit-exact by construction), the ternary ``if``
set, the comparison set, and a transcendental-heavy set (``sin``/``cos``/``tan``/
``exp``/``log``/``sqrt``/``pow`` and the ``loose_*`` variants), at several tree
depths, row widths and ``input_len`` values.

The measurements are also collected into one table (:func:`test_parity_table`)
so the numbers can be reported without re-running the comparison by hand:
``pytest src/evogp/gp/tests/test_gp_stack_problem_parity.py -q -s`` prints it.
"""

import numpy as np
import pytest

from evogp.gp import GPTreeConfig, init_population, to_legacy_triple
from evogp.gp.evaluator import predict
from evogp.gp.problem import SymbolicRegression

from evogp.tests import reference_eval

pytest.importorskip("torch")


#: ``name, funcs, max_layer_cnt, layer_leaf_prob, const_prob, input_len, pop,
#: max_nodes, n_rows, seed``
CASES = [
    ("arith-l3-n2", ["+", "-", "*", "/"], 3, 0.2, 0.5, 2, 16, 32, 5, 11),
    ("arith-l4-n3", ["+", "-", "*", "/"], 4, 0.25, 0.5, 3, 24, 64, 7, 12),
    ("arith-l5-n5", ["+", "-", "*", "/"], 5, 0.15, 0.3, 5, 16, 128, 6, 13),
    ("if-l4-n3", ["if", "+", "-", "*", "/"], 4, 0.25, 0.4, 3, 24, 64, 8, 14),
    ("if-l5-n8", ["if", "+", "-", "max", "min"], 5, 0.2, 0.4, 8, 16, 128, 4, 15),
    ("cmp-l4-n4", ["<", ">", "<=", ">=", "+", "*"], 4, 0.2, 0.4, 4, 20, 64, 6, 16),
    (
        "rich-l4-n4",
        [
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pow",
            "loose_div",
            "loose_pow",
            "loose_log",
            "loose_inv",
            "loose_sqrt",
            "inv",
            "sinh",
            "cosh",
            "+",
            "-",
            "*",
        ],
        4,
        0.3,
        0.4,
        4,
        24,
        128,
        6,
        17,
    ),
    ("rich-l5-n2", ["sin", "cos", "exp", "log", "abs", "neg", "loose_log", "*"], 5, 0.35, 0.5, 2, 16, 128, 5, 18),
    ("all-funcs-l3-n3", None, 3, 0.3, 0.5, 3, 20, 64, 5, 19),
]


def _parse_case(case):
    """``(name, config, pop, n_rows, seed)`` of a case tuple (``None`` = all)."""
    name, funcs, depth, leaf_prob, const_prob, input_len, pop, max_nodes, n_rows, seed = case
    config = GPTreeConfig(
        max_nodes=max_nodes,
        input_len=input_len,
        using_funcs=funcs if funcs is not None else ALL_FUNCS,
        max_layer_cnt=depth,
        layer_leaf_prob=leaf_prob,
        const_prob=const_prob,
        const_samples=(0.5, -1.0, 2.0, 0.0, 1.5),
    )
    return name, config, pop, n_rows, seed


#: Every function id, by name (the "all-funcs" case).
ALL_FUNCS = [
    "if",
    "+",
    "-",
    "*",
    "/",
    "loose_div",
    "pow",
    "loose_pow",
    "max",
    "min",
    "<",
    ">",
    "<=",
    ">=",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "loose_log",
    "exp",
    "inv",
    "loose_inv",
    "neg",
    "abs",
    "sqrt",
    "loose_sqrt",
]


def _data(config, n_rows, seed):
    """Deterministic float32 data with a mix of small, zero and large values."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_rows, config.input_len)).astype(np.float32)
    x[0, 0] = 0.0  # exercise the zero-divisor / log(0) / pow(0, 0) edges
    if n_rows > 1:
        x[1] = np.float32(0.5)
    return x


def _deviations(predicted, oracle):
    """``(max abs dev, max rel dev, non-finite mismatches)`` of two arrays.

    The relative deviation is taken over the entries where the oracle is not
    negligible (``|oracle| > 1e-12``); the absolute deviation is over the entries
    where BOTH are finite, and the NaN/+-inf pattern is compared separately
    because a semantic difference shows up there first.
    """
    both_finite = np.isfinite(predicted) & np.isfinite(oracle)
    same_nonfinite = np.array_equal(np.isnan(predicted), np.isnan(oracle)) and np.array_equal(
        np.isposinf(predicted), np.isposinf(oracle)
    ) and np.array_equal(np.isneginf(predicted), np.isneginf(oracle))
    if not both_finite.any():
        return 0.0, 0.0, 0 if same_nonfinite else 1
    delta = np.abs(predicted[both_finite].astype(np.float64) - oracle[both_finite].astype(np.float64))
    scale = np.abs(oracle[both_finite].astype(np.float64))
    relative = np.where(scale > 1e-12, delta / np.maximum(scale, 1e-12), 0.0)
    return float(delta.max()), float(relative.max()), 0 if same_nonfinite else 1


#: ``case name -> (max abs dev, max rel dev)`` for the report table.
MEASURED = {}


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_predict_matches_the_torch_oracle(case):
    name, config, pop, n_rows, seed = _parse_case(case)
    state = init_population(config, seed, pop_size=pop)
    x = _data(config, n_rows, seed)

    predicted = predict(state, x)
    value, kind, size = to_legacy_triple(state)
    oracle = reference_eval.batch_forward_from_rows(value, kind, size, x).numpy()[:, :, 0]

    assert predicted.shape == oracle.shape == (pop, n_rows)
    assert predicted.dtype == np.float32
    max_abs, max_rel, nonfinite_mismatches = _deviations(predicted, oracle)
    MEASURED[name] = (max_abs, max_rel)
    assert nonfinite_mismatches == 0, f"{name}: the NaN/inf pattern differs from the oracle"
    # The evaluator mirrors the tree layer's op emitters op for op, so it is
    # bit-exact in practice; the tolerances leave room for a libm difference in
    # the transcendentals between numpy and torch on another platform.
    assert max_abs <= 1e-5, f"{name}: max abs deviation {max_abs}"
    assert max_rel <= 1e-5, f"{name}: max rel deviation {max_rel}"


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_fitness_matches_the_torch_oracle(case):
    """``SymbolicRegression.evaluate`` == ``-reference_eval.sr_fitness_from_rows``."""
    name, config, pop, n_rows, seed = _parse_case(case)
    state = init_population(config, seed, pop_size=pop)
    x = _data(config, n_rows, seed)
    rng = np.random.default_rng(seed + 1000)
    y = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)

    problem = SymbolicRegression(x, y)
    fitness = problem.evaluate(state)
    value, kind, size = to_legacy_triple(state)
    oracle = -reference_eval.sr_fitness_from_rows(value, kind, size, x, y[:, None]).numpy()

    assert fitness.shape == (pop,) and fitness.dtype == np.float32
    # The problem's contract is "never NaN, -inf for a non-finite loss", so the
    # oracle's NaN entries are compared as -inf (`_fitness_from_loss`).
    assert not np.isnan(fitness).any(), f"{name}: a fitness is NaN"
    assert not np.isposinf(fitness).any(), f"{name}: a fitness is +inf"
    expected = np.where(np.isfinite(oracle), oracle, -np.inf)
    assert np.array_equal(np.isneginf(fitness), np.isneginf(expected)), (
        f"{name}: the non-finite fitness pattern differs from the oracle"
    )
    finite = np.isfinite(expected)
    assert np.allclose(fitness[finite], expected[finite], rtol=1e-5, atol=1e-5), name
    # The ranking is what the GP loop consumes: argmax must agree.
    assert int(np.argmax(fitness)) == int(np.argmax(expected))


def test_parity_table(capsys):
    """Print the measured deviation table (``pytest -q -s`` shows it)."""
    if len(MEASURED) < len(CASES):  # pragma: no cover - only in a partial run
        pytest.skip("the per-case tests have not all run in this session")
    lines = [f"{'case':<20}{'max abs dev':>14}{'max rel dev':>14}"]
    for name, (max_abs, max_rel) in MEASURED.items():
        lines.append(f"{name:<20}{max_abs:>14.3e}{max_rel:>14.3e}")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
