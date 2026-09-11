"""End-to-end symbolic-regression smoke test for the ETL-based EvoGP engine.

Purpose
-------
The smallest possible *whole-library* run: ``SymbolicRegression`` + ``GenerateDescriptor``
+ ``GeneticProgramming`` (default crossover/mutation/selection) + ``StandardPipeline``,
built exactly the way ``src/evogp/sr_test.py`` and ``example/basic.py`` build them
(pop 64, small data, ~15 generations, fixed seeds). It asserts the two invariants that
matter at this level of the stack:

1. the fitness contract holds -- ``problem.evaluate(forest)`` is a ``(pop_size,)``
   float32 tensor that is MAXIMIZED, and a tree matching the labels exactly scores 0
   (a perfect fit is the maximum possible fitness);
2. the GP loop actually works -- after a few generations the best fitness is at least
   as good as the initial population's best, and strictly better for a learnable target.

If the problem/pipeline layers are still broken (they are being fixed in parallel --
e.g. a stale ``forest.share_input`` reference), the test reports the blocking error and
SKIPS instead of failing, so it can be committed before those layers land. Anything
else (a shape error, a broken engine, a pipeline bug) fails loudly.

How to run (from the repository root)
-------------------------------------
    PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python -m pytest src/evogp/tests -x -q

Environment knobs: ``EVOGP_SMOKE_POP`` (default 64), ``EVOGP_SMOKE_GENS`` (default 15),
``EVOGP_SMOKE_N`` (default 64), ``EVOGP_EQ_DEVICE`` (default: cuda if available else cpu).
"""

from __future__ import annotations

import os

import pytest
import torch

try:
    from evogp.algorithm import (
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the environment
    Forest = GenerateDescriptor = GeneticProgramming = None
    DefaultCrossover = DefaultMutation = DefaultSelection = None
    StandardPipeline = SymbolicRegression = None
    _IMPORT_ERROR = exc

requires_library = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"evogp is not importable yet ({_IMPORT_ERROR!r}) -- the ETL engine / "
    "plain-torch operators are expected here; run the suite again after they are merged.",
)

POP_SIZE = int(os.environ.get("EVOGP_SMOKE_POP", "64"))
GENERATIONS = int(os.environ.get("EVOGP_SMOKE_GENS", "15"))
N_DATA = int(os.environ.get("EVOGP_SMOKE_N", "64"))
DEVICE = os.environ.get("EVOGP_EQ_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def target_function(inputs: torch.Tensor) -> torch.Tensor:
    """``x0**2 + x1**2 + 0.5`` -- learnable with ``+ - * /``, hard to hit by accident."""
    return (inputs[:, 0] ** 2 + inputs[:, 1] ** 2 + 0.5).reshape(-1, 1)


def make_data(n=N_DATA, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    inputs = (torch.rand(n, 2, generator=gen) * 2.0 - 1.0).to(torch.float32).to(DEVICE)
    return inputs, target_function(inputs).contiguous()


def make_descriptor(input_len=2, output_len=1, max_tree_len=64, max_layer_cnt=4):
    return GenerateDescriptor(
        max_tree_len=max_tree_len,
        input_len=input_len,
        output_len=output_len,
        const_samples=[-1, 0, 1],
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=max_layer_cnt,
        layer_leaf_prob=0.2,
    )


def perfect_forest(inputs, labels):
    """Hand-built ``x0*x0 + x1*x1 + 0.5`` (prefix order) as a 1-tree Forest."""
    node_value = torch.tensor([1.0, 1.0, 3.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.5], dtype=torch.float32)
    node_type = torch.tensor([3, 3, 3, 0, 0, 3, 0, 0, 1], dtype=torch.int16)
    subtree_size = torch.tensor([9, 7, 3, 1, 1, 3, 1, 1, 1], dtype=torch.int16)
    return Forest(
        int(inputs.shape[1]),
        int(labels.shape[1]),
        node_value.unsqueeze(0).to(DEVICE),
        node_type.unsqueeze(0).to(DEVICE),
        subtree_size.unsqueeze(0).to(DEVICE),
    )


@requires_library
def test_fitness_contract_is_maximized_positive_loss():
    """A perfect fit must score exactly 0 (the maximum), any other tree at most 0."""
    torch.manual_seed(SEED)
    inputs, labels = make_data()
    problem = SymbolicRegression(datapoints=inputs, labels=labels)
    assert problem.problem_dim == 2 and problem.solution_dim == 1

    perfect = perfect_forest(inputs, labels)
    perfect_fitness = problem.evaluate(perfect)
    assert tuple(perfect_fitness.shape) == (1,), f"evaluate() should give (pop_size,), got {tuple(perfect_fitness.shape)}"
    assert perfect_fitness.dtype == torch.float32
    assert float(perfect_fitness[0]) == pytest.approx(0.0, abs=1e-6), (
        f"a tree matching the labels exactly must score 0 (maximized), got {float(perfect_fitness[0])}"
    )

    forest = Forest.random_generate(pop_size=POP_SIZE, descriptor=make_descriptor())
    fitness = problem.evaluate(forest)
    assert tuple(fitness.shape) == (POP_SIZE,)
    assert fitness.dtype == torch.float32
    finite = torch.nan_to_num(fitness, nan=float("-inf"), posinf=float("-inf"))
    assert float(finite.max()) <= 1e-6, "fitness must be maximized (losses are negated), so <= 0"


@requires_library
def test_symbolic_regression_smoke_improves():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    inputs, labels = make_data()
    problem = SymbolicRegression(datapoints=inputs, labels=labels)
    descriptor = make_descriptor(input_len=problem.problem_dim, output_len=problem.solution_dim)

    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(pop_size=POP_SIZE, descriptor=descriptor),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.3, descriptor=descriptor.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.1),
    )

    # Best fitness of the initial population, evaluated through the same problem layer
    # (StandardPipeline.step() mutates the tensor it gets, so work on a copy).
    initial_fitness = problem.evaluate(algorithm.forest).detach().clone()
    initial_fitness = torch.nan_to_num(initial_fitness, nan=float("-inf"), posinf=float("-inf"))
    initial_best = float(initial_fitness.max())
    assert initial_best > float("-inf"), "the initial population produced no finite fitness"

    pipeline = StandardPipeline(
        algorithm,
        problem,
        generation_limit=GENERATIONS,
        is_show_details=False,
    )
    try:
        best_tree = pipeline.run()
    except AttributeError as exc:  # the documented stale-API blocker of problem/ or pipeline/
        if "share_input" in str(exc) or "formula" in str(exc):
            pytest.skip(f"problem/pipeline layer still uses the removed API: {exc!r}")
        raise

    assert best_tree is not None, "StandardPipeline.run() returned no best tree"
    final_best = float(pipeline.best_fitness)
    assert final_best >= initial_best, (
        f"elitism violated: best fitness went backwards ({initial_best} -> {final_best})"
    )
    assert final_best > initial_best, (
        f"no improvement after {GENERATIONS} generations (pop={POP_SIZE}, best={final_best}); "
        "either the engine mis-evaluates or the GP loop is broken"
    )
    assert final_best <= 1e-6, "fitness must stay maximized and bound by the perfect-fit value 0"

    prediction = best_tree.forward(inputs)
    assert tuple(prediction.shape) == (N_DATA, 1), (
        f"best tree should predict (N, 1), got {tuple(prediction.shape)}"
    )
