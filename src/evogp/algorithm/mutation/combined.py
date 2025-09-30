from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK


class CombinedMutation(BaseMutation):
    """
    CombinedMutation combines multiple mutation strategies into a single comprehensive mutation operation.
    It applies each mutation operator in sequence to the input forest (population),
    allowing for a diverse set of mutation behaviors to be applied to the population.

    Example:
        CombinedMutation(
            [
                DefaultMutation(mutation_rate=0.2, descriptor=descriptor),
                HoistMutation(mutation_rate=0.2),
                MultiPointMutation(mutation_rate=0.2, mutation_intensity=0.3),
                InsertMutation(mutation_rate=0.2, descriptor=descriptor),
            ]
        )
    """

    def __init__(
        self,
        mutation_operator: list[BaseMutation],
    ):
        """
        Args:
            mutation_operator (list[BaseMutation]): A list of mutation strategies that will be combined
                                                      into one. Each strategy will be applied sequentially
                                                      to the population.
        """
        self.mutation_operator = mutation_operator

    def __call__(self, forest: Forest):
        """
        Applies all mutation operators in the sequence to the forest (population).

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after applying all mutation strategies.
        """
        # Sequentially apply each mutation operator to the forest
        for mutation in self.mutation_operator:
            forest = mutation(forest)

        return forest
