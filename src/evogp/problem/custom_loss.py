import torch
from typing import Callable
from . import BaseProblem
from ..tree import CombinedForest
from ..tree.utils import inspect_function


class CustomLoss(BaseProblem):
    def __init__(self, existing_data: dict, loss_func: Callable):
        self.existing_data = existing_data
        self.loss_func = loss_func
        self.loss_parameters = inspect_function(loss_func)
        self.vmap_loss_func = torch.vmap(
            self.loss_func,
            in_dims=tuple(
                [None] * len(existing_data)
                + [0] * (len(self.loss_parameters) - len(existing_data))
            ),
            out_dims=0,
        )

    def evaluate(self, forest: CombinedForest):
        batch_res = forest.batch_forward(self.existing_data)
        input_data = []
        for n in self.loss_parameters:
            if n in self.existing_data:
                input_data.append(self.existing_data[n])
            else:
                # remove last dimension
                input_data.append(batch_res[n].squeeze(-1))

        # return negetive loss as fitness
        return -self.vmap_loss_func(*input_data)
