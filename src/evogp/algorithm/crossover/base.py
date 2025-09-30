import torch
from ...tree import Forest


class BaseCrossover:
    def __call__(
        self,
        forest: Forest,
        survivor_indices: torch.Tensor,
        target_cnt: int,
        fitness: torch.Tensor,
    ) -> Forest:
        raise NotImplementedError
