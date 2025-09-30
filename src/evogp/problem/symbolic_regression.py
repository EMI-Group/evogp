from typing import Optional, Callable

import torch
from torch import Tensor
from . import BaseProblem
from evogp.tree import Forest, CombinedForest


class SymbolicRegression(BaseProblem):
    def __init__(
        self,
        datapoints: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        func: Optional[Callable] = None,
        num_inputs: Optional[int] = None,
        num_data: Optional[int] = 100,
        lower_bounds: Optional[Tensor] = -1,
        upper_bounds: Optional[Tensor] = 1,
        execute_mode: str = "hybrid parallel",
    ):

        assert execute_mode in [
            "torch",
            "hybrid parallel",
            "data parallel",
            "tree parallel",
            "auto",
        ], f"execute_mode should be one of ['torch', 'hybrid parallel', 'data parallel', 'tree parallel', 'auto'], but got {execute_mode}"
        self.execute_mode = execute_mode

        if datapoints is not None and labels is not None:
            self.datapoints = datapoints
            self.labels = labels
            return
        assert (
            func is not None and num_inputs is not None
        ), "func and num_inputs, must be provided when datapoints and labels are not provided"

        self.datapoints, self.labels = self.generate_data(
            func, num_inputs, num_data, lower_bounds, upper_bounds
        )

    def generate_data(self, func, num_inputs, num_data, lower_bounds, upper_bounds):
        if isinstance(upper_bounds, int) or isinstance(lower_bounds, float):
            upper_bounds = torch.full(
                (num_inputs,), upper_bounds, device="cuda", requires_grad=False
            )
        if isinstance(lower_bounds, int) or isinstance(lower_bounds, float):
            lower_bounds = torch.full(
                (num_inputs,), lower_bounds, device="cuda", requires_grad=False
            )
        upper_bounds = upper_bounds[None, :]
        lower_bounds = lower_bounds[None, :]

        inputs = (
            torch.rand(num_data, num_inputs, device="cuda", requires_grad=False)
            * (upper_bounds - lower_bounds)
            + lower_bounds
        )

        outputs = torch.vmap(func)(inputs)

        return inputs, outputs

    def evaluate(
        self,
        forest: Forest,
        use_MSE: bool = True,

    ):
        if isinstance(forest, CombinedForest):
            assert self.execute_mode == "torch", "execute_mode should be 'torch' when using CombinedForest"
            if not forest.share_input:
                assert False, "Currently, combinedForest with share_input=False is not supported"

        if self.execute_mode == "torch":
            # shape (pop_size, datapoints, output_len)
            pred = forest.batch_forward(self.datapoints)
            if use_MSE:
                return -torch.mean((pred - self.labels[None, :, :]) ** 2, dim=(1, 2))

        else:
            return -forest.SR_fitness(
                self.datapoints,
                self.labels,
                use_MSE,
                self.execute_mode,
            )

    @property
    def problem_dim(self):
        return self.datapoints.shape[1]

    @property
    def solution_dim(self):
        return self.labels.shape[1]
