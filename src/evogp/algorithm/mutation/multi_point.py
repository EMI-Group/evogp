from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, randint, NType, GenerateDescriptor


class MultiPointMutation(BaseMutation):
    """
    MultiPointMutation implements a mutation strategy where a specific number of nodes
    within an individual (tree) are selected based on the `mutation_intensity` parameter
    and each of these selected nodes undergoes a SinglePointMutation. This helps introduce
    more diversity into the individual by making multiple small changes.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
        mutation_intensity: float = 0.3,
        modify_output: bool = False,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
            mutation_intensity (float): Determines the proportion of nodes in the tree that will be mutated.
            modify_output (bool): Whether to modify the index of output node. Default is False.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor
        self.mutation_intensity = mutation_intensity
        self.modify_output = modify_output

    def __call__(self, forest: Forest):
        """
        Perform the multi-point mutation where a specific proportion of nodes in the tree are selected
        and each undergoes SinglePointMutation.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone multi-point mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        def choose_mutation_targets(size_tensor):
            """
            Randomly choose mutation targets (nodes to mutate) based on mutation intensity.
            A specific number of nodes are selected as mutation targets in each tree.

            Args:
                size_tensor (Tensor): The size of each tree.

            Returns:
                Tensor: A tensor indicating which nodes should be mutated (True/False).
            """
            tree_size = size_tensor[:, 0].reshape(-1, 1)
            random = torch.rand(tree_size.shape, device="cuda")
            return (random < self.mutation_intensity) & (
                torch.arange(size_tensor.shape[1], device="cuda") < tree_size
            )

        # Generate mutation indices and positions based on mutation intensity
        mutation_targets = choose_mutation_targets(forest_to_mutate.batch_subtree_size)
        num_targets = mutation_targets.sum()

        # random function
        mutated_node_type = forest_to_mutate.batch_node_type[mutation_targets]
        output_flag = (mutated_node_type & NType.OUT_NODE).to(torch.bool)
        mutated_node_type = mutated_node_type & NType.TYPE_MASK

        random_uf = torch.searchsorted(
            self.descriptor.roulette_ufuncs,
            torch.rand(num_targets, device="cuda"),
            out_int32=True,
        )
        random_bf = torch.searchsorted(
            self.descriptor.roulette_bfuncs,
            torch.rand(num_targets, device="cuda"),
            out_int32=True,
        )
        random_tf = torch.searchsorted(
            self.descriptor.roulette_tfuncs,
            torch.rand(num_targets, device="cuda"),
            out_int32=True,
        )
        random_func = torch.stack([random_uf, random_bf, random_tf])
        random_func = random_func[
            (mutated_node_type.to(int) - NType.UFUNC).clamp(0, 2),
            torch.arange(random_func.shape[1]),
        ]

        if self.modify_output is False:
            mutated_node_value = forest_to_mutate.batch_node_value[mutation_targets]
            output_index = torch.where(
                output_flag, mutated_node_value.view(torch.int32) >> 16, 0
            )
        else:
            output_index = torch.randint(
                0, forest.output_len, (num_targets,), dtype=torch.int32, device="cuda"
            )

        random_func = torch.where(
            output_flag,
            (random_func + (output_index << 16)).view(torch.float32),
            random_func,
        )

        # random varible
        random_var = randint(
            size=(num_targets,),
            low=0,
            high=forest.input_len,
        )
        random_var_func = torch.where(
            mutated_node_type == NType.VAR, random_var, random_func
        )

        # random constant
        random_idx = randint(
            size=(num_targets,),
            low=0,
            high=self.descriptor.const_samples.shape[0],
        )
        random_const = self.descriptor.const_samples[random_idx]
        forest_to_mutate.batch_node_value[mutation_targets] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_var_func
        )

        # Update the forest with the mutated trees
        forest[mutate_indices] = forest_to_mutate
        return forest
