"""Brax policy optimization (the paper's robotics-control task).

Evolves a population of trees as the control policy of a Brax environment:
``BraxProblem.evaluate`` rolls every individual out and returns its accumulated episode
return, which the pipeline maximizes.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/brax_task.py

External requirements: ``jax`` and ``brax`` (installed together by the ``brax`` extra,
``pip install ".[brax]"``). Both are imported lazily by the problem constructor, so a
missing one fails at start-up with the message below. This is a long run (pop 1000 x 100
generations x 1000-step episodes): shrink ``pop_size`` / ``generation_limit`` while
experimenting.
"""

import torch

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    DefaultMutation,
    DefaultCrossover,
    DeleteMutation,
    CombinedMutation,
)
from evogp.pipeline import StandardPipeline
from evogp.problem import BraxProblem

try:
    problem = BraxProblem(
        "swimmer",
        max_episode_length=1000,
    )
except ImportError as exc:
    raise SystemExit(
        "example/brax_task.py needs the optional dependencies 'jax' and 'brax' "
        '(pip install ".[brax]"); loading the environment failed with: ' + str(exc)
    ) from exc


descriptor = GenerateDescriptor(
    max_tree_len=256,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/"],
    max_layer_cnt=6,
    const_range=[-1, 1],
    sample_cnt=100,
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
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
)

pipeline.run()
