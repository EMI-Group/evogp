"""Oracle parity for ``evogp.gp``'s evaluation layer (dev-only reference evaluator).

This file used to cross-check the stack against the shipped torch engine
(``Forest.batch_forward`` / ``Forest.SR_fitness``, driven on the numpy
``evoxc`` backend).  That torch stack is deleted in this release, so the
cross-check now runs against the evaluator the stack was ultimately diffed
against anyway: the independent, slow, correctly-rounded reference evaluator
``test/oracle/reference_eval.py`` (``batch_forward_from_rows`` /
``sr_fitness_from_rows``).

What stayed: the exact same populations, data and claims -- one test for the
value path (``evaluator.predict`` vs the oracle's per-tree forward) and one for
the loss/fitness path (``problem.SymbolicRegression.loss`` / ``.evaluate`` vs
``sr_fitness_from_rows``), so the layer is still pinned against an
implementation that is written from the SEMANTICS OF RECORD (the deleted CUDA
kernels), not from the code under test.

What ended: nothing in the layer's coverage -- both cross-checks survived; only
their reference implementation changed.  The torch engine had nothing the
oracle does not cover here, and the oracle covers it without a device, without
``evoxc`` at call time and without the torch stack's ``"cuda"`` hard-coding
(which is why the old file had to skip on a CPU box).

The oracle is dev-only tooling: it lives under the repository's ``test/oracle/``
(NOT on ``PYTHONPATH``, never imported by ``src/evogp/``), imports only
``numpy``/``torch`` and is written against float64-widened, correctly-rounded
float32 semantics -- so this is a parity check, NOT a bitwise one: the gp path
is plain float32 numpy on the CPU.

Measured deviations on the three populations below (masked to the elements that
are finite on both sides, relative to ``max(|observed|, |expected|)``):

* value path: ``0.0`` on all three cases (numpy's float32 transcendentals are
  round-trip float64, i.e. correctly rounded for these operands);
* loss path: at most ``1.34e-07`` relative (``abs`` 6.10e-05 on a ~4.6e+02
  loss), i.e. a couple of ulps of float32 accumulation.

The bounds are pinned with ~75x margin over those measurements so the tests
fail on a real regression and not on library-version noise in the last ulp.
"""

import pathlib
import sys

import numpy as np
import pytest

from evogp.gp import GPTreeConfig, init_population, to_legacy_triple
from evogp.gp.evaluator import predict
from evogp.gp.problem import SymbolicRegression

pytest.importorskip("torch", reason="the reference evaluator is written in plain torch")

#: ``<repo>/test/oracle``: dev-only tooling, absent from a plain checkout.
ORACLE_DIR = pathlib.Path(__file__).resolve().parents[4] / "test" / "oracle"
if ORACLE_DIR.is_dir() and str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))
reference_eval = pytest.importorskip(
    "reference_eval",
    reason=f"the dev-only oracle is not present at {ORACLE_DIR}",
)

pytest.importorskip("evoxc", reason="init_population draws the population via evoxc")


CASES = (
    # name, using_funcs, max_nodes, max_layer_cnt, input_len
    ("arith", ("+", "-", "*", "/"), 24, 3, 3),
    ("if-cmp", ("+", "-", "*", "/", "if", "<", ">"), 32, 3, 4),
    ("rich", ("+", "*", "sin", "exp", "log", "sqrt", "cos"), 32, 3, 2),
)

#: Pinned tolerances: measured 0.0 (values) and <=1.34e-07 (loss) -- see above.
MAX_VALUE_REL_DEVIATION = 1e-5
MAX_LOSS_REL_DEVIATION = 1e-5


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


def _numpy(tensor):
    """The oracle returns torch tensors; compare as float64 numpy."""
    return tensor.detach().cpu().numpy().astype(np.float64)


def _relative_deviation(observed, expected):
    """``(max relative, max absolute)`` over the elements finite on both sides.

    The elements are almost all finite; where an operator produced NaN/±inf the
    exact placement is compared separately (``np.isfinite`` equality), so the
    magnitude check only has to cover the finite ones.
    """
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert observed.shape == expected.shape, (observed.shape, expected.shape)
    both = np.isfinite(observed) & np.isfinite(expected)
    assert both.any(), "no finite element to compare"
    difference = np.abs(observed[both] - expected[both])
    scale = np.maximum(np.abs(observed[both]), np.abs(expected[both]))
    scale = np.where(scale == 0.0, 1.0, scale)  # 0 vs 0 -> absolute, never NaN
    return float(np.max(difference / scale)), float(np.max(difference))


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_oracle_forward_matches_the_host_evaluator(case):
    """``evaluator.predict`` reproduces ``batch_forward_from_rows``."""
    name, config, state, x, _ = _case(case)
    value, node_type, size = to_legacy_triple(state)

    host = predict(state, x)
    oracle = _numpy(
        reference_eval.batch_forward_from_rows(
            value, node_type, size, x, out_len=1
        )[:, :, 0]
    )
    assert host.shape == oracle.shape == (12, 9), (host.shape, oracle.shape)

    # NaN/±inf placement is part of the contract (the semantics of record never
    # sanitize: ``/`` by 0 and ``log`` of a negative are left to propagate).
    np.testing.assert_array_equal(
        np.isfinite(host),
        np.isfinite(oracle),
        err_msg=f"{name}: the non-finite pattern moved",
    )

    relative, absolute = _relative_deviation(host, oracle)
    assert relative <= MAX_VALUE_REL_DEVIATION, (
        f"{name}: max relative deviation {relative:.3e} exceeds "
        f"{MAX_VALUE_REL_DEVIATION:.0e} (max absolute {absolute:.3e})"
    )


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_oracle_loss_and_fitness_match_the_host_problem(case):
    """``SymbolicRegression`` agrees with ``sr_fitness_from_rows`` (negated)."""
    name, config, state, x, y = _case(case)
    value, node_type, size = to_legacy_triple(state)
    problem = SymbolicRegression(x, y)

    loss = problem.loss(state)
    oracle_loss = _numpy(
        reference_eval.sr_fitness_from_rows(
            value, node_type, size, x, y[:, None], use_mse=True
        )
    )
    assert loss.shape == oracle_loss.shape == (12,), (loss.shape, oracle_loss.shape)

    np.testing.assert_array_equal(
        np.isfinite(loss),
        np.isfinite(oracle_loss),
        err_msg=f"{name}: the non-finite pattern moved",
    )
    relative, absolute = _relative_deviation(loss, oracle_loss)
    assert relative <= MAX_LOSS_REL_DEVIATION, (
        f"{name}: max relative loss deviation {relative:.3e} exceeds "
        f"{MAX_LOSS_REL_DEVIATION:.0e} (max absolute {absolute:.3e})"
    )

    # The maximized-fitness contract: ``-loss`` with non-finite mapped to -inf,
    # on a fresh writable float32 array that is never NaN and never +inf.
    fitness = problem.evaluate(state)
    assert fitness.dtype == np.float32
    assert fitness.flags.writeable
    np.testing.assert_array_equal(
        fitness, np.where(np.isfinite(loss), -loss, -np.inf).astype(np.float32)
    )
    assert not np.any(np.isnan(fitness)), "fitness must never be NaN"
    assert not np.any(np.isposinf(fitness)), "fitness must never be +inf"

    # ... and it is still the oracle's loss, negated, within the same bound.
    oracle_fitness = np.where(np.isfinite(oracle_loss), -oracle_loss, -np.inf)
    relative, absolute = _relative_deviation(fitness, oracle_fitness)
    assert relative <= MAX_LOSS_REL_DEVIATION, (
        f"{name}: max relative fitness deviation {relative:.3e} exceeds "
        f"{MAX_LOSS_REL_DEVIATION:.0e} (max absolute {absolute:.3e})"
    )
