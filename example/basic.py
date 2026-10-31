"""Minimal end-to-end demo: symbolic regression on a 3-input XOR/parity table.

The smallest complete EvoGP run and the program behind the README's Quick Start: a data set
built inline, a ``GenerateDescriptor``, ``Forest.random_generate``, the three default
operators, ``StandardPipeline`` with the Pareto front enabled, and finally the best tree's
predictions, its sympy expression and the Pareto front.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/basic.py

As shipped (pop 50 x 100 generations) the run takes about 9.5 s on the reference box at the
shipped default route (``auto`` -> the fused Triton route) with ``EVOXC_DISABLE_LOCATIONS=1``
-- evoxc's tracing-only speed knob, it does not change the fitness, and it is the recommended
setting for real runs; without it the evoxc call-site capture makes tracing several times
slower. The first fused launch per process also pays a one-time Triton JIT of ~1.6 s unless
``TRITON_CACHE_DIR`` points at a warm, writable cache. The run is unseeded, and it ends
around a fitness of -0.25, occasionally -0.375 (MSE on the 8 table rows): with ``+ - * /``
and the constants -1/0/1 the parity function is only approximated, not solved exactly.
"""

import torch

from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
)
from evogp.problem import SymbolicRegression
from evogp.pipeline import StandardPipeline

XOR_INPUTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.float,
    device="cuda",
)

XOR_OUTPUTS = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float,
    device="cuda",
)

problem = SymbolicRegression(datapoints=XOR_INPUTS, labels=XOR_OUTPUTS)

# create decriptor for generating new trees
descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=5,
    const_samples=[-1, 0, 1],
)

# create the algorithm
algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=50, descriptor=descriptor),
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(
        mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    enable_pareto_front=True,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

pred_res = best.forward(XOR_INPUTS)
print(pred_res)

sympy_expression = best.to_sympy_expr()
print(sympy_expression)

# best.to_png("./imgs/xor_tree.png")
print(algorithm.pareto_front)
