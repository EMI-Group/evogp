from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, GenerateDescriptor


class DefaultMutation(BaseMutation):
    """
    DefaultMutation implements a mutation strategy where a randomly selected subtree in each individual 
    of the forest is replaced with a newly generated random subtree.

    The mutation occurs with a probability determined by the `mutation_rate` parameter. For each individual 
    selected for mutation, a new subtree is randomly generated and inserted at a random position within the 
    individual's tree structure.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor

    def __call__(self, forest: Forest):
        """
        Perform mutation on the forest by applying mutation to a subset of individuals based on the mutation rate.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        # Generate random subtrees to replace parts of the trees
        sub_forest = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            descriptor=self.descriptor
        )

        # Generate random mutation positions in each tree
        mutate_positions_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(forest_to_mutate.pop_size,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        # Ensure the mutation positions are valid within each tree's size
        mutate_positions = (
            mutate_positions_unlimited % forest_to_mutate.batch_subtree_size[:, 0]
        )

        # Perform the mutation by replacing the selected positions with the generated subtrees
        forest[mutate_indices] = forest_to_mutate.mutate(mutate_positions, sub_forest)

        return forest
