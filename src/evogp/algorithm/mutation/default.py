from typing import Optional, Tuple
import torch
from torch import Tensor

from .base import BaseMutation
from ...core import Forest, MAX_STACK
from ...core.utils import *


class DefaultMutation(BaseMutation):

    def __init__(self, **generate_configs):
        self.gp_len = generate_configs["gp_len"]
        self.input_len = generate_configs["input_len"]
        self.output_len = generate_configs["output_len"]
        self.out_prob = generate_configs["out_prob"]
        self.const_prob = generate_configs["const_prob"]
        (
            self.depth2leaf_probs,
            self.roulette_funcs,
            self.const_samples,
        ) = parse_generate_configs(**generate_configs)

    def __call__(self, forest: Forest, mutate_indices: Tensor) -> Forest:

        forest_to_mutate = forest[mutate_indices]

        # generate sub trees
        sub_forest = Forest.random_generate(
            pop_size=forest_to_mutate.pop_size,
            gp_len=self.gp_len,
            input_len=self.input_len,
            output_len=self.output_len,
            out_prob=self.out_prob,
            const_prob=self.const_prob,
            depth2leaf_probs=self.depth2leaf_probs,
            roulette_funcs=self.roulette_funcs,
            const_samples=self.const_samples,
        )

        # generate mutate positions
        mutate_positions_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(forest_to_mutate.pop_size,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        mutate_positions = (
            mutate_positions_unlimited % forest_to_mutate.batch_subtree_size[:, 0]
        )

        return forest_to_mutate.mutate(mutate_positions, sub_forest)
