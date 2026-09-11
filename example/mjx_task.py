"""MuJoCo-Playground policy optimization (the MuJoCo counterpart of ``brax_task.py``).

Evolves a population of trees as the control policy of a MuJoCo-Playground environment:
``MujocoProblem.evaluate`` rolls every individual out and returns its accumulated episode
return, which the pipeline maximizes.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/mjx_task.py

External requirements: ``jax`` and ``mujoco_playground``. Both are imported lazily by the
problem constructor, so a missing one fails at start-up with the message below -- note that
``mujoco_playground`` has NO ``setup.py`` extra (unlike brax), so it must be installed by
hand. Like ``brax_task.py`` this is a long run (pop 1000 x 100 generations x 1000-step
episodes): shrink ``pop_size`` / ``generation_limit`` while experimenting.
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
from evogp.problem import MujocoProblem

try:
    problem = MujocoProblem(
        "SwimmerSwimmer6",
        max_episode_length=1000,
    )
except ImportError as exc:
    raise SystemExit(
        "example/mjx_task.py needs the optional dependencies 'jax' and "
        "'mujoco_playground' (pip install jax mujoco_playground; evogp declares no extra "
        f"for them); loading the environment failed with: {exc}"
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
