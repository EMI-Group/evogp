import time
import pandas as pd

seed = 1
pop_size = 100
problem_id = 109

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
from evogp.problem import Classification

from ucimlrepo import fetch_ucirepo

problem = fetch_ucirepo(id=problem_id)


def convert_categoricals_to_int(
    features_df: pd.DataFrame, targets_df: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
    """
    Converts categorical columns (dtype 'object') in feature and target
    DataFrames to integer codes.
    """
    X_processed = features_df.copy()
    y_processed = targets_df.copy()

    for col in X_processed.columns:
        # Check for columns with string data
        if X_processed[col].dtype == "object":
            print(f"Converting feature column '{col}' to integer codes.")
            # Use pandas to factorize the column into integer codes
            X_processed[col] = X_processed[col].astype("category").cat.codes

    for col in y_processed.columns:
        if y_processed[col].dtype == "object":
            print(f"Converting target column '{col}' to integer codes.")
            y_processed[col] = y_processed[col].astype("category").cat.codes

    return X_processed, y_processed

X_original_df = problem.data.features
y_original_df = problem.data.targets

# 2. Apply the new preprocessing function
X_processed_df, y_processed_df = convert_categoricals_to_int(
    X_original_df, y_original_df
)

X = X_processed_df.to_numpy(dtype=np.float32)
y = y_processed_df.to_numpy(dtype=np.float32)
X_torch = torch.tensor(X, dtype=torch.float32, device="cuda").contiguous()
y_torch = torch.tensor(y, dtype=torch.float32, device="cuda").contiguous()

datapoint = X.shape[0]
labels = y_torch[:, 0]
problem = Classification(datapoints=X_torch, labels=labels)

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

best = pipeline.run()