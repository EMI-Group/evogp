import torch

from . import BaseMutation, BaseCrossover, BaseSelection, DefaultSelection
from evogp.core import Forest
from evogp.core.utils import *


class GeneticProgramming:

    def __init__(
        self,
        crossover: BaseCrossover,
        mutation: BaseMutation,
        selection: BaseSelection = DefaultSelection,
        elite_selection: BaseSelection = None,
        recipient_selection: BaseSelection = None,
        donor_selection: BaseSelection = None,
        elite_rate: float = 0.01,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ):
        self.forest = None
        self.pop_size = -1
        self.crossover = crossover
        self.mutation = mutation
        self.default_selection = selection
        self.elite_selection = (
            elite_selection if elite_selection is not None else selection
        )
        self.recipient_selection = (
            recipient_selection if recipient_selection is not None else selection
        )
        self.donor_selection = (
            donor_selection if donor_selection is not None else selection
        )
        self.elite_rate = elite_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize(self, pop_size, **generate_configs):
        self.pop_size = pop_size
        (
            depth2leaf_probs,
            roulette_funcs,
            const_samples,
        ) = parse_generate_configs(**generate_configs)
        self.forest = Forest.random_generate(
            pop_size,
            generate_configs["gp_len"],
            generate_configs["input_len"],
            generate_configs["output_len"],
            generate_configs["out_prob"],
            generate_configs["const_prob"],
            depth2leaf_probs,
            roulette_funcs,
            const_samples,
        )
        return self.forest

    def step(self, fitness: torch.Tensor):
        assert self.forest is not None, "forest is not initialized"
        assert fitness.shape == (
            self.forest.pop_size,
        ), "fitness shape should be ({self.forest.pop_size}, ), but got {fitness.shape}"

        # select elite
        elite_num = int(self.pop_size * self.elite_rate)
        if elite_num > 0:
            elite_indices = self.elite_selection(fitness, elite_num)
            elite = self.forest[elite_indices].clone()

        # crossover
        crossover_num = int(self.pop_size * self.crossover_rate)
        if crossover_num > 0:
            recipient_indices = self.recipient_selection(fitness, crossover_num)
            donor_indices = self.donor_selection(fitness, crossover_num)
            self.forest[:crossover_num] = self.crossover(
                self.forest, recipient_indices, donor_indices
            )

        # mutation
        mutate_indices = torch.arange(0, self.pop_size)[
            torch.rand(self.pop_size) < self.mutation_rate
        ]
        if len(mutate_indices) > 0:
            self.forest[mutate_indices] = self.mutation(self.forest, mutate_indices)

        if elite_num > 0:
            self.forest[-elite_num:] = elite
        return self.forest
