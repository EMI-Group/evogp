from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType, GenerateDescriptor


class MultiConstMutation(BaseMutation):
    """
    MultiConstMutation implements a mutation strategy where a specific number of constant nodes 
    within an individual (tree) are selected based on the `mutation_intensity` parameter 
    and each of these selected nodes is modified to a new constant value. 
    This helps introduce variation in constant values across the individuals in the population.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
        mutation_intensity: float = 0.3,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
            mutation_intensity (float): The proportion of constant nodes in the tree that will be mutated. 
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor
        self.mutation_intensity = mutation_intensity

    def __call__(self, forest: Forest):
        """
        Perform the multi-constant mutation where a specific number of constant nodes in the tree 
        are selected and mutated to new constant values.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone 
                    multi-constant mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:  
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        def choose_mutation_targets(size_tensor, type_tensor):
            """
            Randomly choose mutation targets (constant nodes to mutate) based on mutation intensity.
            A specific number of constant nodes are selected as mutation targets in each tree.

            Args:
                size_tensor (Tensor): The size of each tree.
                type_tensor (Tensor): The type of each node in the tree.

            Returns:
                Tensor: A tensor indicating which constant nodes should be mutated (True/False).
            """
            tree_size = size_tensor[:, 0].reshape(-1, 1)
            random = torch.rand(tree_size.shape, device="cuda")
            return (
                (random < self.mutation_intensity)  # Select mutation targets based on intensity
                & (type_tensor == NType.CONST)  # Only target constant nodes
                & (torch.arange(size_tensor.shape[1], device="cuda") < tree_size)  # Stay within the tree size
            )

        # Generate mutation indices and positions for constant nodes
        mutation_targets = choose_mutation_targets(
            forest_to_mutate.batch_subtree_size, forest_to_mutate.batch_node_type
        )
        num_targets = mutation_targets.sum()

        # Randomly generate new constant values for the mutation targets
        random_idx = torch.randint(
            low=0,
            high=self.descriptor.const_samples.shape[0],
            size=(num_targets,),
            device="cuda",
        )
        random_const = self.descriptor.const_samples[random_idx]

        # Mutate the constant nodes by replacing them with new constant values
        forest_to_mutate.batch_node_value[mutation_targets] = random_const
        
        # Update the forest with the mutated trees
        forest[mutate_indices] = forest_to_mutate
        return forest
