from typing import Optional
import torch
from torch import Tensor

from ...tree import Forest, MAX_STACK, randint
from ..selection import BaseSelector
from .base import BaseCrossover


class LeafBiasedCrossover(BaseCrossover):
    """
    LeafBiasedCrossover implements a crossover strategy similar to `DiversityCrossover`, but with a bias towards 
    exchanging leaf nodes between individuals.

    This strategy introduces a `leaf_bias` parameter, which determines the probability of performing crossover 
    on leaf nodes of the trees. By biasing the crossover towards leaf nodes, it ensures more stable and controlled 
    genetic exchange, especially useful in tree-based genetic representations where leaf nodes typically represent 
    terminal values.
    """

    def __init__(
        self,
        crossover_rate: int = 0.9,
        leaf_bias: float = 0.3,
        recipient_selector: Optional[BaseSelector] = None,
        donor_selector: Optional[BaseSelector] = None,
    ):
        """
        Args:
            crossover_rate (float): The proportion of individuals that will undergo crossover. Should be between 0 and 1.
            leaf_bias (float): The probability of selecting leaf nodes for crossover. Should be between 0 and 1.
            recipient_selector (Optional[BaseSelector]): A selection operator used to choose recipient individuals for crossover. 
                If None, random selection will be used.
            donor_selector (Optional[BaseSelector]): A selection operator used to choose donor individuals for crossover.
                If None, random selection will be used.
        """
        self.crossover_rate = crossover_rate
        self.leaf_bias = leaf_bias
        self.recipient_selector = recipient_selector
        self.donor_selector = donor_selector

    def __call__(
        self,
        forest: Forest,
        fitness: torch.Tensor,
        survivor_indices: torch.Tensor,
        target_cnt: torch.Tensor,
    ):
        """
        Perform crossover on the survivors with a bias towards leaf nodes, based on the leaf_bias parameter.

        Args:
            forest (Forest): The population of individuals represented as a Forest object.
            fitness (torch.Tensor): A tensor containing the fitness values of individuals.
            survivor_indices (torch.Tensor): Indices of the individuals selected as survivors for crossover.
            target_cnt (torch.Tensor): The total number of individuals to produce.

        Returns:
            torch.Tensor: A tensor of new individuals formed by crossover and direct copying.
        """

        # Calculate the number of crossovers to perform based on the crossover rate.
        crossover_cnt = int(target_cnt * self.crossover_rate)

        # Choose recipient and donor indices for crossover based on selection strategies.
        if self.recipient_selector is not None:
            recipient_indices = self.recipient_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            recipient_indices = survivor_indices[random_indices]

        if self.donor_selector is not None:
            donor_indices = self.donor_selector(fitness, crossover_cnt)
        else:
            random_indices = torch.randint(
                low=0,
                high=survivor_indices.size(0),
                size=(crossover_cnt,),
                dtype=torch.int32,
                device="cuda",
                requires_grad=False,
            )
            donor_indices = survivor_indices[random_indices]

        # Choose recipient and donor positions within the trees.
        def choose_leaf_pos(size_tensor: Tensor):
            """
            Choose leaf node positions within the tree based on the subtree sizes. 
            Leaf nodes are the terminal nodes in the tree structure, and we bias the 
            crossover towards these nodes.
            """
            random = torch.rand(size_tensor.shape, device="cuda")
            # Mask out the exceeding parts of the individual
            arange_tensor = torch.arange(size_tensor.shape[1], device="cuda")
            mask = arange_tensor < size_tensor[:, 0].unsqueeze(1)
            random = random * mask
            # Mask out the non-leaf nodes (only leaf nodes can participate in crossover)
            random = torch.where(size_tensor == 1, random, 0)
            return torch.argmax(random, 1).to(torch.int32)

        # Select leaf node positions for both recipient and donor
        size_tensor = forest.batch_subtree_size
        recipient_leaf_pos = choose_leaf_pos(size_tensor[recipient_indices])
        donor_leaf_pos = choose_leaf_pos(size_tensor[donor_indices])

        # Select normal positions (non-leaf nodes) for both recipient and donor
        recipient_normal_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[recipient_indices, 0],
            dtype=torch.int32,
        )
        donor_normal_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[donor_indices, 0],
            dtype=torch.int32,
        )

        # Bias crossover towards leaf nodes based on the leaf_bias probability
        leaf_pair = torch.rand(crossover_cnt, device="cuda") < self.leaf_bias
        recipient_pos = torch.where(leaf_pair, recipient_leaf_pos, recipient_normal_pos)
        donor_pos = torch.where(leaf_pair, donor_leaf_pos, donor_normal_pos)

        # Perform the crossover operation to generate new trees from the recipient and donor.
        crossovered_forest = forest.crossover(
            recipient_indices,
            donor_indices,
            recipient_pos,
            donor_pos,
        )

        # Select remaining individuals that will directly copy to the new generation without crossover.
        random_indices = torch.randint(
            low=0,
            high=survivor_indices.size(0),
            size=(target_cnt - crossover_cnt,),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        static_forest = forest[random_indices]

        # Combine the crossovered trees and the statically copied trees to form the new population.
        return crossovered_forest + static_forest
