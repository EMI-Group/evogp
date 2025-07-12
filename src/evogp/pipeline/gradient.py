import time
import numpy as np
import torch

from ..algorithm import GeneticProgramming
from ..problem import BaseProblem
from . import BasePipeline


class GradientPipeline(BasePipeline):
    def __init__(
        self,
        algorithm: GeneticProgramming,
        problem: BaseProblem,
        fitness_target: float = None,
        generation_limit: int = 100,
        time_limit: int = None,
        is_show_details: bool = True,
        valid_fitness_boundry: float = 1e8,
        training_steps: int = 10,
    ):

        self.algorithm = algorithm
        self.problem = problem
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.time_limit = time_limit
        self.is_show_details = is_show_details
        self.valid_fitness_boundry = valid_fitness_boundry

        self.best_tree = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.training_steps = training_steps

    def evaluate_with_training(self, forest, datapoints, labels):
        # This function is used to evaluate the forest with training
        # It is used to avoid the problem that the forest is not trained when using torch
        # The forest should be trained before using this function
        # The forest should be in training mode
        # The datapoints and labels should be in the same device as the forest
        # The datapoints and labels should be in the same dtype as the forest
        # The datapoints and labels should be in the same shape as the forest
        # The datapoints and labels should be in the same range as the forest

        with torch.no_grad():
            forest.batch_forward(datapoints)
            forest.batch_backward(datapoints, labels)
        forest.batch_step()
        # The forest should be in eval mode
        forest.eval()
        # The forest should be in eval mode
        # The datapoints and labels should be in the same device as the forest
        # The datapoints and labels should be in the same dtype as the forest
        # The datapoints and labels should be in the same shape as the forest
        # The datapoints and labels should be in the same range as the forest
        with torch.no_grad():
            forest.batch_forward(datapoints)
            forest.batch_backward(datapoints, labels)
        forest.batch_step()

    def step(self):
        for _ in range(self.training_steps):
            self.evaluate_with_training(
                self.algorithm.forest, self.problem.datapoints, self.problem.labels
            )

        # evaluate fitness
        fitnesses = self.problem.evaluate(self.algorithm.forest)

        # transfer nan value in fitnesses to -inf
        fitnesses[torch.isnan(fitnesses)] = -torch.inf

        # update best tree info
        cpu_fitness = fitnesses.cpu().numpy()
        best_idx, best_fitness = int(np.argmax(cpu_fitness)), np.max(cpu_fitness)
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

            cpu_fitness = self.step()

            if self.is_show_details:
                self.show_details(start_time, generation_cnt, cpu_fitness)

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
