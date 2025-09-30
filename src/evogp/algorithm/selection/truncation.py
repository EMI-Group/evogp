from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class TruncationSelection(BaseSelection):
    """
    TruncationSelection implements a selection strategy where individuals are selected randomly from the population.

    All individuals are sorted by their fitness, and a certain proportion of low-fitness individuals are excluded. 
    The next generation is then created by randomly sampling from the remaining individuals with replacement and equal probability.
    """

    def __init__(
        self,
        survivor_rate: float = 0.5,
        elite_rate: float = 0,
        survivor_cnt: Optional[int] = None,
        elite_cnt: Optional[int] = None,
    ):
        """
        Args:
            survivor_rate (float): The proportion of individuals to retain in the next generation, based on their fitness. Should be between 0 and 1.
            elite_rate (float): The proportion of elite individuals to retain based on fitness. Should be between 0 and 1.
            survivor_cnt (Optional[int]): The exact number of survivors to retain (if not None).
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if not None).
        """
        super().__init__()

        # Ensure survivor_rate and elite_rate are within the valid range [0, 1].
        assert 0 <= survivor_rate <= 1, "survival_rate should be in [0, 1]"
        assert 0 <= elite_rate <= 1, "elite_rate should be in [0, 1]"

        # Initialize the parameters.
        self.survivor_rate = survivor_rate
        self.survivor_cnt = survivor_cnt
        self.elite_rate = elite_rate
        self.elite_cnt = elite_cnt

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        """
        Perform truncation selection and return the indices of selected elite and survivor individuals.

        Args:
            forest (Forest): The population of individuals represented as a Forest object.
            fitness (torch.Tensor): A tensor containing the fitness values of individuals in the population.

        Returns:
            elite_indices (torch.Tensor): Indices of the individuals selected as elites.
            survivor_indices (torch.Tensor): Indices of the individuals selected as survivors.
        """

        # Preprocess survivor and elite counts based on provided counts or rates.
        if self.survivor_cnt is not None:
            survivor_cnt = self.survivor_cnt
        else:
            survivor_cnt = int(forest.pop_size * self.survivor_rate)

        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        else:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # Survivor selection: sort individuals by fitness and select a subset to retain.
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        num_selectable = int(forest.pop_size * self.survivor_rate)

        # Randomly select survivors from the top `num_selectable` individuals.
        random_indices = torch.multinomial(
            (sorted_indices < num_selectable).to("cuda", torch.float),
            survivor_cnt,
            replacement=True,
        ).to(torch.int32)
        survivor_indices = sorted_indices[random_indices]

        # Elite selection: select the top individuals based on fitness.
        elite_indices = sorted_indices[:elite_cnt]

        return elite_indices, survivor_indices
