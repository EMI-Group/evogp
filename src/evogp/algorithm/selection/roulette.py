from typing import Optional
import torch

from ...tree import Forest
from .base import BaseSelection


class RouletteSelection(BaseSelection):
    """
    Each individual is selected with a probability proportional to its fitness, ensuring that individuals with higher fitness are more likely to be chosen for the next generation.
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
            survivor_rate (float): The proportion of individuals to retain in the next generation. Should be in the range [0, 1].
            elite_rate (float): The proportion of elite individuals to retain based on fitness. Should be in the range [0, 1].
            survivor_cnt (Optional[int]): The exact number of individuals to retain as survivors (if not None).
            elite_cnt (Optional[int]): The exact number of elite individuals to retain (if not None).
        """
        super().__init__()
        assert 0 <= survivor_rate <= 1, "survival_rate should be in [0, 1]"
        assert 0 <= elite_rate <= 1, "elite_rate should be in [0, 1]"
        self.survivor_rate = survivor_rate
        self.survivor_cnt = survivor_cnt
        self.elite_rate = elite_rate
        self.elite_cnt = elite_cnt

    def __call__(self, forest: Forest, fitness: torch.Tensor):
        # preprocess
        if self.survivor_cnt is not None:
            survivor_cnt = self.survivor_cnt
        else:
            survivor_cnt = int(forest.pop_size * self.survivor_rate)

        if self.elite_cnt is not None:
            elite_cnt = self.elite_cnt
        else:
            elite_cnt = int(forest.pop_size * self.elite_rate)

        # survivor selection
        survivor_indices = torch.multinomial(
            (fitness / torch.sum(fitness)).cuda(),
            survivor_cnt,
            replacement=True,
        ).to(torch.int32)

        # elite selection
        if elite_cnt == 0:
            elite_indices = torch.tensor([], device="cuda", dtype=torch.int64)
        else:
            sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
            elite_indices = sorted_indices[:elite_cnt]

        return elite_indices, survivor_indices
