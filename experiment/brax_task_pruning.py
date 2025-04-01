import os
import csv
import time
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(description="evogp brax experiment")
parser.add_argument("--env_name", type=str, default="swimmer")
parser.add_argument("--device_name", type=str, default="4060Ti")
parser.add_argument("--pop_size", type=int, default=1000)
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--generation", type=int, default=200)
parser.add_argument("--save_path", type=str, default="./experiment/data")
args = parser.parse_args()

FILE_NAME = os.path.join(
    args.save_path,
    f"{args.env_name}_{args.device_name}_{args.pop_size}_{args.seed}_very_tiny_delete",
)
LOG_HEADERS = [
    "generation",
    "best_fitness",
    "mean_fitness",
    "max_tree_size",
    "mean_tree_size",
    "min_tree_size",
    "current_evaluation_time",
    "current_algorithm_time",
    "total_time",
]


import torch

torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from evogp.tree import Forest, GenerateDescriptor
from evogp.problem import BraxProblem
from evogp.algorithm import (
    GeneticProgramming,
    DefaultSelection,
    TournamentSelection,
    DefaultMutation,
    DefaultCrossover,
    DeleteMutation,
    CombinedMutation,
    LeafBiasedCrossover,
    MultiConstMutation,
    MultiPointMutation,
    SinglePointMutation,
)


def create_log_file():
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(FILE_NAME + ".csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(LOG_HEADERS)


def append_row_to_csv(row):
    print(row)
    with open(FILE_NAME + ".csv", mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)


def run():
    create_log_file()

    problem = BraxProblem(
        args.env_name,
        max_episode_length=100,
    )

    descriptor = GenerateDescriptor(
        max_tree_len=128,
        input_len=problem.problem_dim,
        output_len=problem.solution_dim,
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=7,
        const_range=[-5, 5],
        sample_cnt=100,
        layer_leaf_prob=0,
    )

    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(
            pop_size=args.pop_size, descriptor=descriptor
        ),
        crossover=LeafBiasedCrossover(leaf_bias=0.95),
        mutation=CombinedMutation(
            [
                # DefaultMutation(
                #     mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
                # ),
                MultiConstMutation(
                    mutation_rate=0.8, mutation_intensity=0.3, descriptor=descriptor
                ),
                MultiPointMutation(
                    mutation_rate=0.8,
                    descriptor=descriptor,
                    mutation_intensity=0.3,
                    modify_output=True,
                ),
                DeleteMutation(mutation_rate=0.8, max_mutatable_size=3),
            ]
        ),
        # selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
        selection=TournamentSelection(
            tournament_size=20,
            best_probability=0.9,
            replace=False,
            survivor_rate=0.5,
            elite_rate=0.05,
        ),
        enable_pareto_front=True,
    )

    total_time = 0
    for i in range(args.generation):
        if i == 100:
            algorithm.mutation = CombinedMutation(
                [
                    MultiConstMutation(
                        mutation_rate=0.8,
                        mutation_intensity=0.05,
                        descriptor=descriptor,
                    ),
                    MultiPointMutation(
                        mutation_rate=0.8,
                        descriptor=descriptor,
                        mutation_intensity=0.05,
                        modify_output=False,
                    ),
                    DeleteMutation(mutation_rate=0.8, max_mutatable_size=3),
                ]
            )

        # evaluate
        evaluate_tic = time.time()
        fitness = problem.evaluate(algorithm.forest)
        torch.cuda.synchronize()
        evaluate_time = time.time() - evaluate_tic

        # update
        algorithm_tic = time.time()
        algorithm.step(fitness)
        torch.cuda.synchronize()
        algorithm_time = time.time() - algorithm_tic

        # evaluate + update
        total_time += time.time() - evaluate_tic

        # record
        scores = fitness.cpu().numpy()
        valid_fitness = scores[~np.isinf(scores)]
        max_f, mean_f = (
            max(valid_fitness),
            np.mean(valid_fitness),
        )

        size = algorithm.forest.batch_subtree_size[:, 0].cpu().numpy()
        max_size, mean_size, min_size = (max(size), np.mean(size), min(size))
        append_row_to_csv(
            [
                i,
                max_f,
                mean_f,
                max_size,
                mean_size,
                min_size,
                evaluate_time,
                algorithm_time,
                total_time,
            ]
        )

    with open(FILE_NAME + ".pkl", "wb") as f:
        pickle.dump(algorithm.pareto_front, f)


if __name__ == "__main__":
    run()
