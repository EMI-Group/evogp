from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK


class CombinedMutation(BaseMutation):

    def __init__(
        self,
        mutation_operator: list[BaseMutation],
    ):
        self.mutation_operator = mutation_operator

    def __call__(self, forest: Forest):
        for mutation in self.mutation_operator:
            forest = mutation(forest)

        return forest
