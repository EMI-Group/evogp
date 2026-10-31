"""Classification demo: evolve a small decision tree per class on a built-in data set.

``Classification`` scores a forest by accuracy: the tree has one output per class
(``solution_dim = num_classes``), and ``argmax`` of the softmaxed outputs is compared with
the labels. The data comes from scikit-learn through ``dataset="iris"``; set
``dataset_name`` below to ``"wine"``, ``"breast_cancer"`` or ``"digits"`` for the other
built-in data sets.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/classifier.py

As shipped (pop 1000 x 100 generations, the original settings) the run exceeds a 15-minute
smoke budget on the reference box, so shrink ``pop_size`` / ``generation_limit`` while
experimenting. Evaluation is one fused Triton kernel launch per generation on the shipped
default route (``auto`` -> fused), so ``pop_size`` trades against wall time through the
operator layer and the per-generation print/sync rather than through per-tree compilation.
The run writes ``./imgs/classifier_tree.png`` through the optional ``pygraphviz`` package
plus the graphviz ``dot`` binary -- comment that call out if either is missing, everything
else only needs torch + scikit-learn. ``EVOXC_DISABLE_LOCATIONS=1`` is evoxc's tracing-only
speed knob; it does not change the fitness.
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
from evogp.problem import Classification



dataset_name = ["iris", "wine", "breast_cancer", "digits"]


multi_output = True

problem = Classification(multi_output=multi_output, dataset="iris")

descriptor = GenerateDescriptor(
    max_tree_len=64,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=4,
    const_samples=[-1, 0, 1],
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=1000, descriptor=descriptor),
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
    enable_pareto_front=True,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

best = pipeline.run()

best.to_png("./imgs/classifier_tree.png")
sympy_expression = best.to_sympy_expr()
print(sympy_expression)

print(algorithm.pareto_front)
