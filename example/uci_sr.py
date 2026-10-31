"""Paper reproduction: symbolic regression on a UCI data set (paper id 409).

Identical descriptor / algorithm / selection configuration to ``example/custom_sr.py``, but
the features and targets are downloaded live from the UCI repository with
``fetch_ucirepo(id=problem_id)``.

Run from the repository root (needs a CUDA-visible device)::

    PYTHONPATH=src python example/uci_sr.py

External requirements: ``ucimlrepo`` (``pip install ucimlrepo``) plus network access at run
time. On the reference box the paper's pop 100 x 100 generations with ``max_tree_len=512``
take about 10.7 s at the shipped default route (``auto`` -> the fused Triton route) with
``EVOXC_DISABLE_LOCATIONS=1`` -- evoxc's tracing-only speed knob, it does not change the
fitness; without it tracing is several times slower. The measured best fitness there is
-73.29.

Only the paper's own ids are handled: id 1 (Abalone, ``Sex`` mapped to 0/1/2) and id 9
(Auto MPG, rows with a NaN ``horsepower`` dropped); other ids that contain string columns
fail in ``to_numpy(dtype=np.float32)``. The printed ``MSE`` and node statistics come from
the FINAL generation's population (``pipeline.fitness`` / ``algorithm.forest``), not from
the best-ever individual (``pipeline.best_tree`` / ``pipeline.best_fitness``).
"""

import time

seed = 1
pop_size = 100
problem_id = 409

import numpy as np
import torch

torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

from evogp.pipeline import StandardPipeline
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    TournamentSelection,
    DefaultMutation,
    DefaultCrossover,
    CombinedMutation,
    DeleteMutation,
    DiversityCrossover,
)
from evogp.problem import SymbolicRegression

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as exc:
    raise SystemExit(
        "example/uci_sr.py downloads its data with ucimlrepo (pip install ucimlrepo) and "
        "needs network access at run time."
    ) from exc
problem = fetch_ucirepo(id=problem_id)
if problem_id == 1:
    mapping = {"M": 0, "F": 1, "I": 2}
    problem.data.features.loc[:, "Sex"] = problem.data.features["Sex"].map(mapping)
elif problem_id == 9:
    problem.data.features.dropna(subset=["horsepower"], inplace=True)
    problem.data.targets = problem.data.targets.loc[problem.data.features.index]

X = problem.data.features.to_numpy(dtype=np.float32)
y = problem.data.targets.to_numpy(dtype=np.float32)
X_torch = torch.tensor(X, dtype=torch.float32, device="cuda").contiguous()
y_torch = torch.tensor(y, dtype=torch.float32, device="cuda").contiguous()

datapoint = X.shape[0]

problem = SymbolicRegression(datapoints=X_torch, labels=y_torch)

descriptor = GenerateDescriptor(
    max_tree_len=512,
    input_len=problem.problem_dim,
    output_len=problem.solution_dim,
    using_funcs=["+", "-", "*", "/", "sin", "cos", "tan"],
    max_layer_cnt=9,
    const_range=[-5, 5],
    sample_cnt=10000,
    layer_leaf_prob=0.3,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=pop_size, descriptor=descriptor),
    crossover=DefaultCrossover(),
    # crossover=DiversityCrossover(crossover_rate=0.9),
    # mutation=CombinedMutation(
    #     [
    #         DefaultMutation(
    #             mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=3)
    #         ),
    #         DeleteMutation(mutation_rate=0.8),
    #     ]
    # ),
    mutation=DefaultMutation(
        mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=4)
    ),
    selection=TournamentSelection(
        tournament_size=20, survivor_rate=0.5, elite_rate=0.1
    ),
    enable_pareto_front=False,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
    is_show_details=True,
)

start_time = time.time()
best = pipeline.run()
end_time = time.time()
during_time = end_time - start_time
mse = -torch.max(pipeline.fitness)
length = pipeline.algorithm.forest.batch_subtree_size[:, 0]
mean_node = torch.mean(length.float())
max_node = torch.max(length)

print( "Best: ", best)
print( "MSE: ", mse)
print( "During time: ", during_time)
print( "Mean node: ", mean_node)
print( "Max node: ", max_node)