import torch

from ...tree import CombinedForest
from .base import BaseCrossover


class CombinedDefaultCrossover(BaseCrossover):
    def __call__(
        self,
        forest: CombinedForest,
        survivor_indices: torch.Tensor,
        target_cnt: int,
        fitness: torch.Tensor,
    ):
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

        # crossover each forest in the combined forest
        new_forest = []
        for i in range(len(forest.forests)):
            tree_sizes = survivor_forest.forests[i].batch_subtree_size[:, 0]

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

            new_forest.append(
                survivor_forest.forests[i].crossover(
                    left_indices, right_indices, left_pos, right_pos
                )
            )

        return CombinedForest(
            new_forest, forest.data_info
        )
