"""The in-graph variant: one traced ``evoxc`` graph per population shape.

``problem.predict_graph`` / ``problem.sr_fitness_graph`` evaluate a whole
population inside a single compiled graph (the structure of the population is
runtime input, the dispatch is a select chain over the used function ids).  The
tests here pin the two things that make that variant usable: it agrees with the
host evaluator BIT FOR BIT on the numpy backend -- including the NaN/inf pattern
and the composed ``sinh``/``cosh`` -- and it is memoized per shape.

Row widths and populations are deliberately tiny: the numpy backend builds a
graph in well under a second at these sizes, and this suite is not a benchmark
(GPU/device measurement is another task's job).
"""

import numpy as np
import pytest

from evogp.gp import GPTreeConfig, init_population
from evogp.gp.evaluator import predict
from evogp.gp.problem import (
    SymbolicRegression,
    clear_graph_cache,
    graph_cache_stats,
    predict_graph,
    sr_fitness_graph,
)

pytest.importorskip("torch")  # only for the shared oracle-comparison fixture


CASES = [
    # name, functions, max_layer_cnt, max_nodes, pop_size, input_len, n_rows, seed
    ("arith", ["+", "-", "*", "/"], 3, 16, 6, 3, 5, 21),
    ("if-cmp", ["if", "<", ">", "+", "-", "max", "min"], 3, 16, 5, 2, 4, 22),
    (
        "rich",
        ["sin", "cos", "exp", "log", "sqrt", "pow", "loose_div", "loose_log", "tanh", "+", "*"],
        3,
        16,
        5,
        4,
        4,
        23,
    ),
]


def _case(case):
    name, funcs, depth, max_nodes, pop, input_len, n_rows, seed = case
    config = GPTreeConfig(
        max_nodes=max_nodes,
        input_len=input_len,
        using_funcs=funcs,
        max_layer_cnt=depth,
        layer_leaf_prob=0.25,
        const_samples=(0.5, -1.0, 2.0, 0.0),
    )
    state = init_population(config, seed, pop_size=pop)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_rows, input_len)).astype(np.float32)
    x[0, 0] = 0.0  # the zero edges: div / log / pow(0, 0)
    y = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    return name, state, x, y


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_graph_predict_is_bit_identical_to_the_host_evaluator(case):
    name, state, x, _ = _case(case)
    host = predict(state, x)
    graph = predict_graph(state, x)
    assert graph.shape == host.shape and graph.dtype == np.float32
    np.testing.assert_array_equal(
        np.isnan(graph), np.isnan(host), err_msg=f"{name}: NaN pattern"
    )
    np.testing.assert_array_equal(np.isposinf(graph), np.isposinf(host))
    np.testing.assert_array_equal(np.isneginf(graph), np.isneginf(host))
    finite = np.isfinite(host)
    np.testing.assert_array_equal(graph[finite], host[finite])


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_graph_fitness_is_bit_identical_to_the_problem(case):
    name, state, x, y = _case(case)
    problem = SymbolicRegression(x, y)
    host = problem.evaluate(state)
    graph = problem.evaluate_graph(state)
    assert graph.shape == host.shape and graph.dtype == np.float32
    # the host maps a non-finite loss to -inf; the graph does the same readback
    finite = np.isfinite(host) & np.isfinite(graph)
    np.testing.assert_array_equal(
        np.isneginf(graph), np.isneginf(host), err_msg=f"{name}: -inf pattern"
    )
    np.testing.assert_array_equal(graph[finite], host[finite])
    mae_graph = sr_fitness_graph(state, x, y, use_mse=False)
    mae_host = SymbolicRegression(x, y, use_mse=False).evaluate(state)
    finite = np.isfinite(mae_host) & np.isfinite(mae_graph)
    np.testing.assert_array_equal(mae_graph[finite], mae_host[finite])


def test_graph_predictions_match_the_problem_predict():
    _, state, x, y = _case(CASES[0])
    problem = SymbolicRegression(x, y)
    np.testing.assert_array_equal(problem.predict_graph(state), problem.predict(state))


def test_the_graph_is_memoized_per_shape_and_function_set():
    clear_graph_cache()
    try:
        _, state, x, y = _case(CASES[0])
        empty = graph_cache_stats()
        assert empty["graphs"] == 0
        first = predict_graph(state, x)
        after_first = graph_cache_stats()
        assert after_first["graphs"] == 1
        second = predict_graph(state, x)
        assert graph_cache_stats()["graphs"] == 1  # a hit, not a rebuild
        np.testing.assert_array_equal(first, second)
        # the fitness variant is its own graph, and the same shape/function set
        # shares nothing with the prediction graph
        sr_fitness_graph(state, x, y)
        assert graph_cache_stats()["graphs"] == 2
        assert graph_cache_stats()["compile_cache_maxsize"] > 0
    finally:
        clear_graph_cache()
    assert graph_cache_stats()["graphs"] == 0


def test_a_new_function_set_rebuilds_the_graph():
    clear_graph_cache()
    try:
        base = GPTreeConfig(
            max_nodes=8,
            input_len=2,
            using_funcs=["+", "-"],
            max_layer_cnt=2,
            layer_leaf_prob=0.3,
            const_samples=(1.0,),
        )
        other = base.replace(using_funcs=["*", "/"])
        x = np.asarray([[1.0, 2.0]], dtype=np.float32)
        predict_graph(init_population(base, 1, pop_size=4), x)
        predict_graph(init_population(other, 1, pop_size=4), x)
        # the two populations use disjoint function ids -> two graphs
        assert graph_cache_stats()["graphs"] == 2
    finally:
        clear_graph_cache()


def test_graph_input_validation():
    _, state, x, y = _case(CASES[0])
    with pytest.raises(ValueError, match="input_len"):
        predict_graph(state, x, input_len=x.shape[1] + 1)
    with pytest.raises(ValueError, match="y has"):
        sr_fitness_graph(state, x, y[:-1])
    with pytest.raises(ValueError, match="X must be 1-D or 2-D"):
        predict_graph(state, np.zeros((2, 2, 2), dtype=np.float32))
    # a VAR index outside the data width is caught by the shared validation
    opcode = np.asarray([[4, 0]], np.int32)  # ADD(VAR 3)
    payload = np.asarray([[0.0, 3.0]], np.float32)
    size = np.asarray([[2, 1]], np.int32)
    channels = (opcode, payload, size)
    with pytest.raises(ValueError, match="input_len"):
        predict_graph(channels, np.zeros((2, 2), dtype=np.float32))
    # validate=False skips the check (the index is then clipped like the host
    # evaluator does) instead of raising
    clipped = predict_graph(channels, np.zeros((2, 2), dtype=np.float32), validate=False)
    np.testing.assert_array_equal(clipped, predict(channels, np.zeros((2, 2), np.float32), validate=False))


def test_graph_cache_knob_is_validated(monkeypatch):
    monkeypatch.setenv("EVOGP_GP_GRAPH_CACHE_SIZE", "not-a-number")
    with pytest.raises(ValueError, match="EVOGP_GP_GRAPH_CACHE_SIZE"):
        graph_cache_stats()
    monkeypatch.setenv("EVOGP_GP_GRAPH_CACHE_SIZE", "0")
    with pytest.raises(ValueError, match="must be positive"):
        graph_cache_stats()
