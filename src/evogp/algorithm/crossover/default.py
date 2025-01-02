import torch
from torch import Tensor

from ...core import Forest, MAX_STACK
from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):

    def __init__(self):
        pass

    def __call__(
        self,
        forest: Forest,
        recipient_indices: Tensor,
        donor_indices: Tensor,
    ) -> Forest:
        # choose recipient and donor positions
        tree_sizes = forest.batch_subtree_size[:, 0]
        recipient_pos_unlimited, donor_pos_unlimited = torch.randint(
            low=0,
            high=MAX_STACK,
            size=(2, recipient_indices.size(0)),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        recipient_pos = recipient_pos_unlimited % tree_sizes[recipient_indices]
        donor_pos = donor_pos_unlimited % tree_sizes[donor_indices]

        return forest.crossover(
            recipient_indices, donor_indices, recipient_pos, donor_pos
        )
