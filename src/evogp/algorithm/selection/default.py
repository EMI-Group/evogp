from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class DefaultSelection(BaseSelection):
    """
    DefaultSelection is the default selection strategy for the TGP algorithm.

    This strategy first preserves a certain proportion of elite individuals (those ranked at the top of the population by fitness). Then, individuals ranked at the bottom of the population are eliminated based on the survival rate. The strategy allows flexibility in defining the number or proportion of elite individuals to retain.
    """

    def __init__(
        self,
        survival_rate: float = 0.3,
        elite_cnt: Optional[int] = None,
        elite_rate: Optional[float] = None,
    ):
        """
        Args:
            survival_rate (float): The survival rate, determining the proportion of individuals that survive (between 0 and 1).
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if not None).
            elite_rate (Optional[float]): The proportion of elite individuals to retain (if not None).
        """
        super().__init__()

        # Ensure that survival_rate is within the valid range [0, 1].
        assert 0 <= survival_rate <= 1, "survival_rate should be in [0, 1]"

        # Ensure that elite_cnt and elite_rate are not both set at the same time.
        assert (
            elite_cnt is None or elite_rate is None
        ), "elite_cnt and elite_rate should not be set at the same time"

        # Initialize the parameters.
        self.survival_rate = survival_rate
        self.elite_cnt = elite_cnt
        self.elite_rate = elite_rate

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        """
        Perform the selection operation and return the indices of selected elite and survivor individuals.

        Args:
            forest (Forest): The population of individuals represented as a Forest object.
            fitness (torch.Tensor): A tensor containing the fitness values of individuals in the population.

        Returns:
            elite_indices (torch.Tensor): Indices of the individuals selected as elites.
            survivor_indices (torch.Tensor): Indices of the individuals selected to survive based on the survival rate.
        """
        # Calculate the number of individuals that survive based on survival_rate.
        survive_cnt = int(forest.pop_size * self.survival_rate)

        # Determine the number of elite individuals to retain.
        elite_cnt = 0
        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        elif self.elite_rate is not None:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # Sort the population by fitness in descending order.
        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)

        # Select the indices of the surviving individuals and elite individuals.
        survivor_indices = sorted_indices[:survive_cnt].to(dtype=torch.int32)
        elite_indices = sorted_indices[:elite_cnt].to(dtype=torch.int32)

        return elite_indices, survivor_indices
