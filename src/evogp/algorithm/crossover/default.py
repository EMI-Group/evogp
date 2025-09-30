import torch

from ...tree import Forest, MAX_STACK
from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):
    """
    DefaultCrossover implements a crossover strategy where a random subtree of the recipient is replaced with a random subtree from the donor.
    Both the recipient and donor are chosen randomly from the survivors.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        forest: Forest,
        survivor_indices: torch.Tensor,
        target_cnt: int,
        fitness: torch.Tensor,
    ):
        """
        Perform crossover between randomly selected individuals in the survivor population.

        Args:
            forest (Forest): The population of individuals represented as a Forest object.
            survivor_indices (torch.Tensor): Indices of the individuals selected as survivors for crossover.
            target_cnt (int): The number of crossover operations to perform.
            fitness (torch.Tensor): A tensor containing the fitness values of individuals in the population.

        Returns:
            torch.Tensor: A tensor containing the resulting individuals after performing the crossover.
        """

        # Get the forest of survivors based on the given indices.
        survivor_forest = forest[survivor_indices]

        # Randomly select pairs of individuals for crossover (left and right).
        left_indices, right_indices = torch.randint(
            low=0,
            high=len(survivor_forest),
            size=(2, target_cnt),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )

        # Randomly select positions for crossover within the individuals' trees.
        tree_sizes = survivor_forest.batch_subtree_size[:, 0]
        left_pos_unlimited, right_pos_unlimited = torch.randint(
            low=0,
            high=torch.iinfo(torch.int32).max,
            size=(2, target_cnt),
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        # Ensure positions are within the subtree sizes.
        left_pos = left_pos_unlimited % tree_sizes[left_indices]
        right_pos = right_pos_unlimited % tree_sizes[right_indices]

        # Perform crossover on the selected individuals and return the results.
        return survivor_forest.crossover(
            left_indices, right_indices, left_pos, right_pos
        )
