from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...tree import Forest, MAX_STACK, NType, randint, GenerateDescriptor


class SinglePointMutation(BaseMutation):
    """
    SinglePointMutation implements a mutation strategy where a random node in the tree is selected
    and replaced with a new node of the same type, chosen randomly from a node pool.
    This operation helps maintain the structure of the tree while introducing variation by changing individual nodes.
    """

    def __init__(
        self,
        mutation_rate: float,
        descriptor: GenerateDescriptor,
        modify_output: bool = False,
    ):
        """
        Args:
            mutation_rate (float): The probability of each individual undergoing mutation. Should be between 0 and 1.
            descriptor (GenerateDescriptor): The descriptor used to generate random subtrees for mutation.
            modify_output (bool): Whether to modify the index of output node. Default is False.
        """
        self.mutation_rate = mutation_rate
        self.descriptor = descriptor
        self.modify_output = modify_output

    def __call__(self, forest: Forest):
        """
        Perform the single-point mutation by randomly selecting a node in the tree and replacing it
        with a new node of the same type from the node pool.

        Args:
            forest (Forest): The current population of trees (Forest object).

        Returns:
            Forest: The updated population after mutation, where some individuals have undergone the single-point mutation.
        """
        # Determine which trees need to mutate based on the mutation rate
        mutate_indices = torch.rand(forest.pop_size) < self.mutation_rate

        # If no trees are selected for mutation, return the original forest
        if mutate_indices.sum() == 0:
            return forest

        # Extract the subset of trees that need to mutate
        forest_to_mutate = forest[mutate_indices]

        # Generate random mutation positions within the trees
        num_mutate = forest_to_mutate.pop_size
        mutate_positions = randint(
            size=(num_mutate,),
            low=0,
            high=forest_to_mutate.batch_subtree_size[:, 0],
        )

        # random function
        mutated_node_type = forest_to_mutate.batch_node_type[
            torch.arange(num_mutate), mutate_positions
        ]
        output_flag = (mutated_node_type & NType.OUT_NODE).to(torch.bool)
        mutated_node_type = mutated_node_type & NType.TYPE_MASK

        random_uf = torch.searchsorted(
            self.descriptor.roulette_ufuncs,
            torch.rand(num_mutate, device="cuda"),
            out_int32=True,
        )
        random_bf = torch.searchsorted(
            self.descriptor.roulette_bfuncs,
            torch.rand(num_mutate, device="cuda"),
            out_int32=True,
        )
        random_tf = torch.searchsorted(
            self.descriptor.roulette_tfuncs,
            torch.rand(num_mutate, device="cuda"),
            out_int32=True,
        )
        random_func = torch.stack([random_uf, random_bf, random_tf])
        random_func = random_func[
            (mutated_node_type.to(int) - NType.UFUNC).clamp(0, 2),
            torch.arange(random_func.shape[1]),
        ]

        if self.modify_output is False:
            mutated_node_value = forest_to_mutate.batch_node_value[
                torch.arange(num_mutate), mutate_positions
            ]
            output_index = torch.where(
                output_flag, mutated_node_value.view(torch.int32) >> 16, 0
            )
        else:
            output_index = torch.randint(
                0, forest.output_len, (num_mutate,), dtype=torch.int32, device="cuda"
            )

        random_func = torch.where(
            output_flag,
            (random_func + (output_index << 16)).view(torch.float32),
            random_func,
        )

        # random varible
        random_var = randint(
            size=(num_mutate,),
            low=0,
            high=forest.input_len,
        )
        random_var_func = torch.where(
            mutated_node_type == NType.VAR, random_var, random_func
        )

        # random constant
        random_idx = randint(
            size=(num_mutate,),
            low=0,
            high=self.descriptor.const_samples.shape[0],
        )
        random_const = self.descriptor.const_samples[random_idx]
        forest.batch_node_value[mutate_indices, mutate_positions] = torch.where(
            mutated_node_type == NType.CONST, random_const, random_var_func
        )

        return forest
