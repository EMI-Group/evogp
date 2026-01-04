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

import pandas as pd
X_df = pd.read_csv(f"data/{problem_id}_features.csv")
y_df = pd.read_csv(f"data/{problem_id}_targets.csv")
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