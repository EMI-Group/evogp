from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint, NType
from .mutation_utils import vmap_subtree


class DeleteMutation(BaseMutation):
    """
    DeleteMutation implements a mutation strategy where a randomly selected non-leaf node in the tree
    has one of its child subtrees removed. This deletion effectively removes part of the individual’s
    genetic material, modifying the structure of the tree.
    """

    def __init__(
        self,
        mutation_rate: float,
        max_mutatable_size: Optional[int] = None,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
        """
        self.mutation_rate = mutation_rate
        self.max_mutatable_size = max_mutatable_size

    def __call__(self, forest: Forest):
        """
        Perform mutation by removing a randomly selected basic operator from the individual’s tree.

        The mutation process involves selecting a non-leaf node in the tree and removing one of its child subtrees
        to modify the individual's genetic material.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone deletion of a subtree.
        """
        # Determine which trees need to mutate based on the mutation rate,
        # ensuring the tree has more than one node to allow for deletion.
        mutate_indices = (
            torch.rand(forest.pop_size, device="cuda") < self.mutation_rate
        ) & (forest.batch_subtree_size[:, 0] > 1)

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        def choose_nonleaf_pos(size_tensor: Tensor):
            """
            Choose a non-leaf node position for mutation.

            Args:
                size_tensor (Tensor): The size of each individual's subtree.

            Returns:
                Tensor: The position of a non-leaf node selected for mutation.
            """
            random = torch.rand(size_tensor.shape, device="cuda")
            # Mask out the exceeding parts of the individual
            arange_tensor = torch.arange(size_tensor.shape[1], device="cuda")
            mask = arange_tensor < size_tensor[:, 0].unsqueeze(1)
            random = random * mask
            # Mask out the leaf nodes
            random = torch.where(size_tensor == 1, 0, random)
            if self.max_mutatable_size:
                random = torch.where(size_tensor > self.max_mutatable_size, 0, random)
            return torch.argmax(random, 1)

        # Choose non-leaf positions to perform mutation
        size_tensor = forest_to_mutate.batch_subtree_size
        mutate_positions = choose_nonleaf_pos(size_tensor)

        # Determine the number of children for the selected mutation positions
        num_mutate = forest_to_mutate.pop_size
        mutate_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        child_nums = (mutate_type & NType.TYPE_MASK) - NType.UFUNC + 1
        # Select a random child to delete
        nth_childs = randint(
            size=(num_mutate,),
            low=1,
            high=child_nums,
            dtype=torch.int64,
        )

        # Determine the positions of the subtrees to be deleted
        pos1 = mutate_positions + 1
        pos2 = pos1 + size_tensor[torch.arange(num_mutate), pos1]
        pos3 = pos2 + size_tensor[torch.arange(num_mutate), pos2]
        next_positions = torch.where(nth_childs == 2, pos2, pos1)
        next_positions = torch.where(nth_childs == 3, pos3, next_positions)

        # Delete the middle part (subtree)
        subtrees = vmap_subtree(forest_to_mutate, next_positions)
        forest[mutate_indices] = forest_to_mutate.mutate(
            mutate_positions.to(torch.int32), subtrees
        )

        return forest
