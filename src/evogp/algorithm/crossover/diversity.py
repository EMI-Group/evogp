from typing import Optional
import torch

from ...tree import Forest, MAX_STACK, randint
from ..selection import BaseSelector
from .base import BaseCrossover


class DiversityCrossover(BaseCrossover):
    """
    DiversityCrossover implements a crossover strategy where individuals undergo crossover based on their diversity.

    Unlike `DefaultCrossover`, this strategy allows for flexible selection of the recipient and donor individuals 
    using specific selection operators (`recipient_selector` and `donor_selector`). Additionally, a `crossover_rate` 
    parameter determines the proportion of the population that will undergo crossover. The remaining individuals bypass 
    the crossover process and are copied directly to the next generation.
    """

    def __init__(
        self,
        crossover_rate: int = 0.9,
        recipient_selector: Optional[BaseSelector] = None,
        donor_selector: Optional[BaseSelector] = None,
    ):
        """
        Args:
            crossover_rate (float): The proportion of individuals that will undergo crossover. Should be between 0 and 1.
            recipient_selector (Optional[BaseSelector]): A selection operator used to choose recipient individuals for crossover. 
                If None, random selection will be used.
            donor_selector (Optional[BaseSelector]): A selection operator used to choose donor individuals for crossover.
                If None, random selection will be used.
        """
        self.crossover_rate = crossover_rate
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
        Perform crossover on the survivors based on selected recipient and donor individuals.

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

        # Choose recipient indices for crossover based on the recipient_selector (if provided), 
        # otherwise choose randomly from survivor_indices.
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

        # Choose donor indices for crossover based on the donor_selector (if provided), 
        # otherwise choose randomly from survivor_indices.
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

        # Select positions within the recipient and donor trees for crossover.
        size_tensor = forest.batch_subtree_size
        recipient_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[recipient_indices, 0],
            dtype=torch.int32,
        )
        donor_pos = randint(
            size=(crossover_cnt,),
            low=0,
            high=size_tensor[donor_indices, 0],
            dtype=torch.int32,
        )

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
