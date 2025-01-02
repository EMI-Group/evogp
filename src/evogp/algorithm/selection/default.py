import torch
from torch import Tensor

from .base import BaseSelection


class DefaultSelection(BaseSelection):

    def __init__(self, survival_rate: float = 0.3):
        assert 0 <= survival_rate <= 1, "survival_rate should be in [0, 1]"
        self.survival_rate = survival_rate

    def __call__(self, fitness: Tensor, choose_num: int) -> Tensor:
        survival_cnt = int(fitness.size(0) * self.survival_rate)
        obsolete_cnt = fitness.size(0) - survival_cnt

        sorted_fitness, sorted_indices = torch.sort(fitness, descending=True)
        probabilities = torch.cat([torch.ones(survival_cnt), torch.zeros(obsolete_cnt)])
        choosed_indices = sorted_indices.to(torch.int32)[
            torch.multinomial(probabilities, choose_num, replacement=True)
        ]

        return choosed_indices
