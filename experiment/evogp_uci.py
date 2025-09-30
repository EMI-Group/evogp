import sys
import time
from exp_utils import update_csv

epoch = int(sys.argv[1])
pop_size = int(sys.argv[2])
problem_id = int(sys.argv[3])

import numpy as np
import torch

torch.random.manual_seed(epoch)
torch.cuda.manual_seed(epoch)

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
)
from evogp.problem import SymbolicRegression

from ucimlrepo import fetch_ucirepo

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
    layer_leaf_prob=0.1,
)


algorithm = GeneticProgramming(
    initial_forest=Forest.random_generate(pop_size=pop_size, descriptor=descriptor),
    crossover=DefaultCrossover(),
    # mutation=CombinedMutation(
    #     [
    #         DefaultMutation(
    #             mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=3)
    #         ),
    #         DeleteMutation(mutation_rate=0.8),
    #     ]
    # ),
    mutation=DefaultMutation(
        mutation_rate=0.1, descriptor=descriptor.update(max_layer_cnt=3)
    ),
    selection=TournamentSelection(
        tournament_size=20, survivor_rate=0.3, elite_rate=0.03
    ),
    enable_pareto_front=False,
)

pipeline = StandardPipeline(
    algorithm,
    problem,
    generation_limit=100,
    is_show_details=False,
)

start_time = time.time()
best = pipeline.run()
end_time = time.time()
during_time = end_time - start_time
mse = -torch.max(pipeline.fitness)
length = pipeline.algorithm.forest.batch_subtree_size[:, 0]
mean_node = torch.mean(length.float())
max_node = torch.max(length)

update_csv(
    f"data/evogp_{epoch}",
    pop_size,
    datapoint,
    mse,
    during_time,
    mean_node,
    max_node,
)
