import torch
from torch import Tensor

from . import BaseMutation, BaseCrossover, BaseSelection
from evogp.tree import Forest


class ParetoFront:
    def __init__(self, size, forest_descriptor):
        self.fitness = torch.full(
            (size,), float("-inf"), dtype=torch.float32, device="cuda"
        )
        self.solution = Forest.zero_generate(size, *forest_descriptor)

    def __str__(self):
        result = []
        for idx in range(len(self.fitness)):
            result.append(
                f"size: {idx}, fitness: {self.fitness[idx]:.2e}, solution: {self.solution[idx]}"
            )
        return "\n".join(result)

    def __repr__(self):
        return self.fitness.__repr__() + self.solution.__repr__()


class GeneticProgramming:

    def __init__(
        self,
        initial_forest: Forest,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection,
        enable_pareto_front: bool = False,
    ):
        self.forest = initial_forest
        self.pop_size = initial_forest.pop_size
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.enable_pareto_front = enable_pareto_front
        if enable_pareto_front:
            self.pareto_front = ParetoFront(
                self.forest.max_tree_len,
                (
                    self.forest.max_tree_len,
                    self.forest.input_len,
                    self.forest.output_len,
                ),
            )

    def for_update_pareto_front(self, fitness, solution: Forest):
        for i in range(self.forest.max_tree_len):
            masked_fitness = torch.where(
                solution.batch_subtree_size[:, 0] == i,
                fitness,
                float("-inf"),
            )
            best_idx = torch.argmax(masked_fitness)
            if masked_fitness[best_idx] > self.pareto_front.fitness[i]:
                self.pareto_front.fitness[i] = fitness[best_idx]
                self.pareto_front.solution[i] = solution[best_idx]

    def vmap_update_pareto_front(self, fitness: Tensor, solution: Forest):
        max_tree_len = solution.max_tree_len
        size = solution.batch_subtree_size[:, 0]
        fitness_expanded = fitness.broadcast_to(max_tree_len, -1)
        size_expanded = size.broadcast_to(max_tree_len, -1)

        # get the best fitness and indices for each subtree size
        masked_fitness = torch.where(
            size_expanded == torch.arange(max_tree_len, device="cuda").unsqueeze(1),
            fitness_expanded,
            float("-inf"),
        )
        best_fitness, best_indices = torch.max(masked_fitness, dim=1)

        # update the Pareto front
        better_mask = best_fitness > self.pareto_front.fitness
        self.pareto_front.fitness = torch.where(
            better_mask,
            best_fitness,
            self.pareto_front.fitness,
        )
        for tensor_name in [
            "batch_node_value",
            "batch_node_type",
            "batch_subtree_size",
        ]:
            setattr(
                self.pareto_front.solution,
                tensor_name,
                torch.where(
                    better_mask.unsqueeze(1),
                    getattr(solution, tensor_name)[best_indices],
                    getattr(self.pareto_front.solution, tensor_name),
                ),
            )

    def step(self, fitness: torch.Tensor):
        assert self.forest is not None, "forest is not initialized"
        assert fitness.shape == (
            self.forest.pop_size,
        ), f"fitness shape should be ({self.forest.pop_size}, ), but got {fitness.shape}"

        if self.enable_pareto_front:
            self.vmap_update_pareto_front(fitness, self.forest)

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
