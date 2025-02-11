import torch

from . import BaseMutation, BaseCrossover, BaseSelection
from evogp.tree import Forest


class GeneticProgramming:

    def __init__(
        self,
        initial_forest: Forest,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
    ):
        self.forest = initial_forest
        self.pop_size = initial_forest.pop_size
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def step(self, fitness: torch.Tensor):
        assert self.forest is not None, "forest is not initialized"
        assert fitness.shape == (
            self.forest.pop_size,
        ), f"fitness shape should be ({self.forest.pop_size}, ), but got {fitness.shape}"

        elite_indices, next_indices = self.selection(self.forest, fitness)
        next_forest = self.crossover(
            forest=self.forest,
            survivor_indices=next_indices,
            target_cnt=self.pop_size - elite_indices.shape[0],
            fitness=fitness,
        )
        next_forest = self.mutation(next_forest)
        self.forest = self.forest[elite_indices] + next_forest

        return self.forest
