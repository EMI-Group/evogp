"""Paper reproduction: symbolic regression on an externally supplied UCI-style data set.

Same descriptor / algorithm / selection configuration as ``example/uci_sr.py`` (the paper's
experiment script), but the data is read from the two CSVs of the paper's problem id instead
of being fetched from the UCI repository::

    data/{problem_id}_features.csv     (Datapoint, *features)
    data/{problem_id}_targets.csv      (Datapoint, *targets)

Those files are NOT part of this repository -- supply them, keep ``problem_id`` in sync, and
run from the repository root (both paths are CWD-relative)::

    PYTHONPATH=src python example/custom_sr.py

External requirements: ``pandas`` (``pip install pandas``), a CUDA-visible device and, in
practice, several minutes: pop 100 x 100 generations with ``max_tree_len=512`` and the
three extra transcendental functions is the paper's setting and is several times slower
without ``ETL_DISABLE_LOCATIONS=1`` (ETL's tracing-only speed knob; it does not change the
fitness).

The printed ``MSE`` and node statistics come from the FINAL generation's population
(``pipeline.fitness`` / ``algorithm.forest``), not from the best-ever individual
(``pipeline.best_tree`` / ``pipeline.best_fitness``).
"""

import time

seed = 1
pop_size = 100
problem_id = 409

import os

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
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "example/custom_sr.py reads its inputs with pandas (pip install pandas)."
    ) from exc

feature_csv = f"data/{problem_id}_features.csv"
target_csv = f"data/{problem_id}_targets.csv"
for path in (feature_csv, target_csv):
    if not os.path.isfile(path):
        raise SystemExit(
            f"example/custom_sr.py needs {feature_csv} and {target_csv} next to the "
            "repository root, but they are not shipped with the repository. Supply the "
            f"paired feature/target CSVs for problem_id={problem_id} (or edit problem_id) "
            "and run again from the repository root."
        )

X_df = pd.read_csv(feature_csv)
y_df = pd.read_csv(target_csv)
X = X_df.to_numpy(dtype=np.float32)
y = y_df.to_numpy(dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
X_torch = torch.tensor(X, dtype=torch.float32, device=device).contiguous()
y_torch = torch.tensor(y, dtype=torch.float32, device=device).contiguous()

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
