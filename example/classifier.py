"""Classification demo: evolve a small decision tree per class on a built-in data set.

``Classification`` scores a forest by accuracy: the tree has one output per class
(``solution_dim = num_classes``), and ``argmax`` of the softmaxed outputs is compared with
the labels. The data comes from scikit-learn through ``dataset="iris"``; set
``dataset_name`` below to ``"wine"``, ``"breast_cancer"`` or ``"digits"`` for the other
built-in data sets.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/classifier.py

As shipped (pop 200 x 100 generations) the run takes roughly a minute with
``ETL_DISABLE_LOCATIONS=1`` (ETL's tracing-only speed knob; it does not change the fitness)
and it writes ``./imgs/classifier_tree.png`` through the optional ``pygraphviz`` package
plus the graphviz ``dot`` binary -- comment that call out if either is missing, everything
else only needs torch + scikit-learn.

``pop_size`` is right-sized for the ETL evaluation engine (one program per tree structure):
the CUDA-scale ``pop_size=1000`` of the original settings costs about an hour here. Raise it
for a harder search.
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
