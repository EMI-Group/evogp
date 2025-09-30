from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType, GenerateDescriptor


class SingleConstMutation(BaseMutation):
    """
    SingleConstMutation implements a mutation strategy where a single constant node within 
    an individual (tree) is randomly selected and replaced with a new constant value. 
    This mutation helps introduce variation by modifying constant nodes in the trees of the population.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor

    def __call__(self, forest: Forest):
        """
        Perform the single-constant mutation where a single constant node in the tree 
        is randomly selected and mutated to a new constant value.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone 
                    single-constant mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        def choose_constant_pos(size_tensor: Tensor, type_tensor: Tensor):
            """
            Randomly choose a constant node to mutate based on the node type and tree size.

            Args:
                size_tensor (Tensor): The size of each tree in the population.
                type_tensor (Tensor): The type of each node in the tree.

            Returns:
                Tensor: A tensor indicating the position of the constant node selected for mutation.
            """
            random = torch.rand(size_tensor.shape, device="cuda")
            # Mask out the exceeding parts of the tree (those beyond the tree size)
            arange_tensor = torch.arange(size_tensor.shape[1], device="cuda")
            mask = arange_tensor < size_tensor[:, 0].unsqueeze(1)
            random = random * mask
            # Mask out the non-constant nodes
            random = torch.where(type_tensor == NType.CONST, random, 0)
            return torch.argmax(random, 1)

        # Choose the constant position to mutate
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = choose_constant_pos(
            forest_to_mutate.batch_subtree_size,
            forest_to_mutate.batch_node_type,
        )

        # Randomly generate new constant values for mutation
        random_idx = torch.randint(
            low=0,
            high=self.descriptor.const_samples.shape[0],
            size=(num_mutate,),
            device="cuda",
        )
        random_const = self.descriptor.const_samples[random_idx]

        # Mutate the selected constant nodes in the trees
        mutated_node_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        original_node_value = forest_to_mutate.batch_node_value[
            torch.arange(num_mutate), mutate_positions
        ]
        forest.batch_node_value[mutate_indices, mutate_positions] = torch.where(
            mutated_node_type == NType.CONST, random_const, original_node_value
        )

        return forest
