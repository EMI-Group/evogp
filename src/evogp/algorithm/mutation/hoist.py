from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint
from .mutation_utils import vmap_subtree


class HoistMutation(BaseMutation):
    """
    HoistMutation implements a mutation strategy where a subtree is randomly selected from a GP individual, 
    and then a subtree within it is selected and moved to replace the original subtree's root.
    This operation is designed to help mitigate excessive growth (bloating) in GP individuals by 
    ensuring that larger subtrees are potentially shrunk or replaced with more compact structures.
    """

    def __init__(
        self,
        mutation_rate: float,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
        """
        self.mutation_rate = mutation_rate

    def __call__(self, forest: Forest):
        """
        Perform the hoist mutation by selecting a subtree from a GP individual and moving a subtree within it 
        to the root position of the original subtree.

        The mutation helps reduce bloating by potentially replacing large subtrees with more compact structures 
        that are part of the individual.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone the hoist operation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest
        else:
            # Extract the subset of trees that need to mutate
            forest_to_mutate = forest[mutate_indices]

        # Generate random mutation positions within the selected trees
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
            dtype=torch.int32,
        )

        # Generate positions for subtrees within the selected mutation positions
        subtree_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[
                torch.arange(num_mutate), mutate_positions
            ],
            dtype=torch.int64,
        )

        # Select the subtrees to be "hoisted" (moved to the root position)
        subtrees = vmap_subtree(forest_to_mutate, subtree_positions)
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions, subtrees
        )

        return forest
