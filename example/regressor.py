"""Symbolic-regression demo: recover a synthetic 2-input target from data.

Fits ``x0**4 / (x0**4 + 1) + x1**4 / (x1**4 + 1)`` on 100 random points in [-5, 5] through
the ``func=``/``num_inputs=``/``num_data=`` constructor path of ``SymbolicRegression``,
using only ``+ - * /`` and no leaf shortcut (``layer_leaf_prob=0``), then prints the
best-ever tree as a sympy expression.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/regressor.py

The GP settings below are right-sized for the ETL evaluation engine, which builds and runs
one program per tree structure instead of a single CUDA batch op: the same run at the
CUDA-scale settings (``pop_size=1000``, ``max_tree_len=128``, ``max_layer_cnt=7``, 300
generations) costs tens of seconds per generation, i.e. hours. As shipped (pop 200 x 100
generations, ``max_tree_len=64``) it takes a few minutes with ``ETL_DISABLE_LOCATIONS=1``
and several times that without it -- that variable is ETL's tracing-only speed knob and does
not change the fitness. Raise ``pop_size`` / ``generation_limit`` for a harder search.
"""

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
)
from evogp.problem import SymbolicRegression


def func(x):
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


problem = SymbolicRegression(
    func=func, num_inputs=2, num_data=100, lower_bounds=-5, upper_bounds=5
)

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=6,
    const_samples=[-1, 0, 1],
    layer_leaf_prob=0,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=200, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=CombinedMutation(
        [
            DefaultMutation(
                mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
            ),
            DeleteMutation(mutation_rate=0.8),
        ]
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    enable_pareto_front=False,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

sympy_expression = best.to_sympy_expr()
print(sympy_expression)
