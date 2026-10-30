"""Optional cross-check against the torch stack's real engine (``evogp.tree.Forest``).

This is the third oracle the evaluation layer is validated against, next to the
plain-torch reference evaluator (``test_gp_stack_problem_parity.py``) and the
hand-built trees (``test_gp_stack_problem_contract.py``): it drives the SHIPPED
engine (``Forest.batch_forward`` / ``Forest.SR_fitness``) with the numpy
``evoxc`` backend on the CPU and compares it to ``evogp.gp``'s host evaluator.

It is optional by construction: torch is a test-only dependency of this subtree
(the stack itself is torch-free), and the tree layer still hard-codes ``"cuda"``
in most of its entry points, so the module skips cleanly whenever torch, evoxc
or a usable engine path is missing.  Nothing here is required for the layer
under test to be correct -- a skip is a non-result, never a failure.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the tree layer oracle needs torch")

from evogp.gp import GPTreeConfig, init_population, to_legacy_triple  # noqa: E402
from evogp.gp.evaluator import predict  # noqa: E402
from evogp.gp.problem import SymbolicRegression  # noqa: E402

pytest.importorskip("evoxc", reason="the tree layer engine is built on evoxc")


CASES = (
    # name, using_funcs, max_nodes, max_layer_cnt, input_len
    ("arith", ("+", "-", "*", "/"), 24, 3, 3),
    ("if-cmp", ("+", "-", "*", "/", "if", "<", ">"), 32, 3, 4),
    ("rich", ("+", "*", "sin", "exp", "log", "sqrt", "cos"), 32, 3, 2),
)


def _forest(config, state):
    """Build the torch stack's ``Forest`` from a ``GPState`` (CPU, numpy backend)."""
    from evogp.tree import Forest

    value, node_type, size = to_legacy_triple(state)
    return Forest(
        config.input_len,
        1,
        torch.tensor(value),
        torch.tensor(node_type),
        torch.tensor(size),
    )


def _case(case):
    name, funcs, max_nodes, max_layer_cnt, input_len = case
    config = GPTreeConfig(
        max_nodes=max_nodes,
        input_len=input_len,
        using_funcs=funcs,
        max_layer_cnt=max_layer_cnt,
        layer_leaf_prob=0.25,
        const_samples=(0.5, -1.0, 2.0, 0.25),
    )
    state = init_population(config, 1234, pop_size=12)
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.5, 1.5, size=(9, input_len)).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, size=9).astype(np.float32)
    return name, config, state, x, y


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_engine_forward_matches_the_host_evaluator(case):
    """``Forest.batch_forward`` (numpy evoxc) reproduces ``evaluator.predict``."""
    name, config, state, x, _ = _case(case)
    forest = _forest(config, state)
    try:
        engine = forest.batch_forward(torch.tensor(x)).detach().cpu().numpy()[:, :, 0]
    except Exception as exc:  # pragma: no cover - engine path unavailable
        pytest.skip(f"the tree layer engine is not drivable here: {type(exc).__name__}: {exc}")

    host = predict(state, x)
    np.testing.assert_allclose(host, engine, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_engine_fitness_matches_the_host_problem(case):
    """``Forest.SR_fitness`` agrees with ``SymbolicRegression.evaluate`` (negated)."""
    name, config, state, x, y = _case(case)
    forest = _forest(config, state)
    try:
        loss = (
            forest.SR_fitness(
                torch.tensor(x), torch.tensor(y[:, None]), True, "auto"
            )
            .detach()
            .cpu()
            .numpy()
        )
    except Exception as exc:  # pragma: no cover - engine path unavailable
        pytest.skip(f"the tree layer engine is not drivable here: {type(exc).__name__}: {exc}")

    problem = SymbolicRegression(x, y)
    fitness = problem.evaluate(state)
    # The tree layer reports the LOSS and leaves a non-finite one as NaN; the
    # gp layer reports the MAXIMIZED fitness and maps non-finite to -inf by
    # contract, so the NaN/±inf pattern is compared through that normalisation.
    expected = np.where(np.isfinite(loss), -loss, -np.inf).astype(np.float32)
    assert not np.any(np.isnan(fitness)), "fitness must never be NaN"
    assert not np.any(np.isposinf(fitness)), "fitness must never be +inf"
    np.testing.assert_allclose(fitness, expected, rtol=0, atol=0)
