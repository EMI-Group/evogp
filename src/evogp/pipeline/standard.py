import time
import numpy as np
import torch

from ..algorithm import GeneticProgramming
from ..problem import BaseProblem
from . import BasePipeline


class StandardPipeline(BasePipeline):
    def __init__(
        self,
        algorithm: GeneticProgramming,
        problem: BaseProblem,
        fitness_target: float = None,
        generation_limit: int = 100,
        time_limit: int = None,
        is_show_details: bool = True,
        valid_fitness_boundry: float = 1e8,
    ):

        # assert algorithm.forest.input_len == problem.problem_dim
        # assert algorithm.forest.output_len == problem.solution_dim

        self.algorithm = algorithm
        self.problem = problem
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.time_limit = time_limit
        self.is_show_details = is_show_details
        self.valid_fitness_boundry = valid_fitness_boundry

        self.best_tree = None
        self.best_fitness = float("-inf")
        self.fitness = None
        self.generation_timestamp = None

    def step(self):
        # evaluate fitness
        fitnesses = self.problem.evaluate(self.algorithm.forest)
        
        # transfer nan value in fitnesses to -inf
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # update best tree info
        cpu_fitness = fitnesses.cpu()
        best_idx, best_fitness = int(torch.argmax(cpu_fitness)), torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

        self.algorithm.step(fitnesses)

        return cpu_fitness

    def run(self):
        tic = time.time()

        generation_cnt = 0
        while True:

            if self.is_show_details:
                start_time = time.time()

            self.fitness = self.step()

            if self.is_show_details:
                self.show_details(start_time, generation_cnt, self.fitness)

            if (
                self.fitness_target is not None
                and self.best_fitness >= self.fitness_target
            ):
                print("Fitness target reached!")
                break

            if self.time_limit is not None and time.time() - tic > self.time_limit:
                print("Time limit reached!")
                break

            generation_cnt += 1
            if generation_cnt >= self.generation_limit:
                print("Generation limit reached!")
                break

        return self.best_tree

    def show_details(self, start_time, generation_cnt, fitnesses):

        valid_fitness = fitnesses[
            (fitnesses < self.valid_fitness_boundry)
            & (fitnesses > -self.valid_fitness_boundry)
        ]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitness),
            min(valid_fitness),
            np.mean(valid_fitness),
            np.std(valid_fitness),
        )
        cost_time = time.time() - start_time

        print(
            f"Generation: {generation_cnt}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tfitness: valid cnt: {len(valid_fitness)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )
