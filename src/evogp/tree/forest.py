from typing import Optional, Tuple

import torch
from torch import Tensor
import numpy as np
from .utils import *
from . import Tree


class Forest:
    def __init__(
        self,
        input_len,
        output_len,
        batch_node_value: Tensor,
        batch_node_type: Tensor,
        batch_subtree_size: Tensor,
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.pop_size, self.gp_len = batch_node_value.shape

        assert batch_node_value.shape == (
            self.pop_size,
            self.gp_len,
        ), f"node_value shape should be ({self.pop_size}, {self.gp_len}), but got {batch_node_value.shape}"
        assert batch_node_type.shape == (
            self.pop_size,
            self.gp_len,
        ), f"node_type shape should be ({self.pop_size}, {self.gp_len}), but got {batch_node_type.shape}"
        assert batch_subtree_size.shape == (
            self.pop_size,
            self.gp_len,
        ), f"subtree_size shape should be ({self.pop_size}, {self.gp_len}), but got {batch_subtree_size.shape}"

        self.batch_node_value = batch_node_value
        self.batch_node_type = batch_node_type
        self.batch_subtree_size = batch_subtree_size

    @staticmethod
    def random_generate(
        pop_size: int,
        gp_len: int,
        input_len: int,
        output_len: int,
        const_prob: float,
        out_prob: Optional[float] = None,
        depth2leaf_probs: Optional[Tensor] = None,
        roulette_funcs: Optional[Tensor] = None,
        const_samples: Optional[Tensor] = None,
        func_prob: Optional[dict] = None,
        max_layer_cnt: Optional[int] = None,
        layer_leaf_prob: Optional[float] = None,
        const_range: Optional[Tuple[float, float]] = None,
        sample_cnt: Optional[int] = None,
    ):
        assert (
            gp_len <= MAX_STACK
        ), f"gp_len={gp_len} is too large, MAX_STACK={MAX_STACK}"

        if depth2leaf_probs is None:
            assert (
                max_layer_cnt is not None
            ), "max_layer_cnt should not be None when depth2leaf_probs is None"
            assert (
                layer_leaf_prob is not None
            ), "layer_leaf_prob should not be None when depth2leaf_probs is None"
            assert (
                2**max_layer_cnt <= gp_len
            ), f"max_layer_cnt is too large for gp_len={gp_len}"

            depth2leaf_probs = torch.tensor(
                [layer_leaf_prob] * max_layer_cnt
                + [1.0] * (MAX_FULL_DEPTH - max_layer_cnt),
                device="cuda",
                requires_grad=False,
            )
        if roulette_funcs is None:
            assert (
                func_prob is not None
            ), "func_prob should not be None when roulette_funcs is None"
            roulette_funcs = torch.tensor(
                dict2cdf(func_prob),
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            )
        if const_samples is None:
            assert (
                const_range is not None
            ), "const_range should not be None when const_samples is None"
            assert (
                sample_cnt is not None
            ), "sample_cnt should not be None when const_samples is None"
            const_samples = (
                torch.rand(sample_cnt, device="cuda", requires_grad=False)
                * (const_range[1] - const_range[0])
                + const_range[0]
            )

        if output_len > 1:
            assert (
                out_prob is not None
            ), "out_prob should not be None when output_len > 1"

        assert depth2leaf_probs.shape == (
            MAX_FULL_DEPTH,
        ), f"depth2leaf_probs shape should be ({MAX_FULL_DEPTH}), but got {depth2leaf_probs.shape}"
        assert roulette_funcs.shape == (
            Func.END,
        ), f"roulette_funcs shape should be ({Func.END}), but got {roulette_funcs.shape}"
        assert (
            const_samples.dim() == 1
        ), f"const_samples dim should be 1, but got {const_samples.dim()}"

        keys = torch.randint(
            low=0,
            high=1000000,
            size=(2,),
            dtype=torch.uint32,
            device="cuda",
            requires_grad=False,
        )

        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        ) = torch.ops.evogp_cuda.tree_generate(
            pop_size,
            gp_len,
            input_len,
            output_len,
            const_samples.shape[0],
            out_prob,
            const_prob,
            keys,
            depth2leaf_probs,
            roulette_funcs,
            const_samples,
        )

        return Forest(
            input_len,
            output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        )

    def forward(self, x: Tensor):
        x = check_tensor(x)

        assert x.shape == (
            self.pop_size,
            self.input_len,
        ), f"x shape should be ({self.pop_size}, {self.input_len}), but got {x.shape}"

        res = torch.ops.evogp_cuda.tree_evaluate(
            self.pop_size,  # popsize
            self.gp_len,  # gp_len
            self.input_len,  # var_len
            self.output_len,  # out_len
            self.batch_node_value,  # value
            self.batch_node_type,  # node_type
            self.batch_subtree_size,  # subtree_size
            x,  # variables
        )

        return res

    def mutate(self, replace_pos: Tensor, new_sub_forest: "Forest"):
        replace_pos = check_tensor(replace_pos)
        assert replace_pos.shape == (
            self.pop_size,
        ), f"replace_indices shape should be ({self.pop_size}, ), but got {replace_pos.shape}"
        assert (
            self.pop_size == new_sub_forest.pop_size
        ), f"pop_size should be {self.pop_size}, but got {new_sub_forest.pop_size}"
        assert (
            self.input_len == new_sub_forest.input_len
        ), f"input_len should be {self.input_len}, but got {new_sub_forest.input_len}"
        assert (
            self.output_len == new_sub_forest.output_len
        ), f"output_len should be {self.output_len}, but got {new_sub_forest.output_len}"
        assert (
            self.gp_len == new_sub_forest.gp_len
        ), f"gp_len should be {self.gp_len}, but got {new_sub_forest.gp_len}"

        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        ) = torch.ops.evogp_cuda.tree_mutate(
            self.pop_size,
            self.gp_len,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            replace_pos,
            new_sub_forest.batch_node_value,
            new_sub_forest.batch_node_type,
            new_sub_forest.batch_subtree_size,
        )

        return Forest(
            self.input_len,
            self.output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        )

    def crossover(
        self,
        left_indices: Tensor,
        right_indices: Tensor,
        left_pos: Tensor,
        right_pos: Tensor,
    ):
        left_indices = check_tensor(left_indices)
        right_indices = check_tensor(right_indices)
        left_pos = check_tensor(left_pos)
        right_pos = check_tensor(right_pos)

        res_forest_size = left_indices.shape[0]

        assert left_indices.shape == (
            res_forest_size,
        ), f"left_indices shape should be ({res_forest_size}, ), but got {left_indices.shape}"
        assert right_indices.shape == (
            res_forest_size,
        ), f"right_indices shape should be ({res_forest_size}, ), but got {right_indices.shape}"
        assert left_pos.shape == (
            res_forest_size,
        ), f"left_pos shape should be ({res_forest_size}, ), but got {left_pos.shape}"
        assert right_pos.shape == (
            res_forest_size,
        ), f"right_pos shape should be ({res_forest_size}, ), but got {right_pos.shape}"

        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        ) = torch.ops.evogp_cuda.tree_crossover(
            self.pop_size,
            res_forest_size,
            self.gp_len,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            left_indices,
            right_indices,
            left_pos,
            right_pos,
        )

        return Forest(
            self.input_len,
            self.output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        )

    def SR_fitness(self, inputs: Tensor, labels: Tensor, use_MSE: bool = True):
        inputs = check_tensor(inputs)
        labels = check_tensor(labels)

        batch_size = inputs.shape[0]
        assert inputs.shape == (
            batch_size,
            self.input_len,
        ), f"inputs shape should be ({batch_size}, {self.input_len}), but got {inputs.shape}"

        assert labels.shape == (
            batch_size,
            self.output_len,
        ), f"outputs shape should be ({batch_size}, {self.output_len}), but got {labels.shape}"

        res = torch.ops.evogp_cuda.tree_SR_fitness(
            self.pop_size,
            batch_size,
            self.gp_len,
            self.input_len,
            self.output_len,
            use_MSE,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            inputs,
            labels,
        )

        return res

    def __getitem__(self, index):
        if isinstance(index, int):
            return Tree(
                self.input_len,
                self.output_len,
                self.batch_node_value[index],
                self.batch_node_type[index],
                self.batch_subtree_size[index],
            )
        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            return Forest(
                self.input_len,
                self.output_len,
                self.batch_node_value[index],
                self.batch_node_type[index],
                self.batch_subtree_size[index],
            )
        else:
            raise NotImplementedError

    def __setitem__(self, index, value):
        if isinstance(index, int):
            assert isinstance(
                value, Tree
            ), f"value should be Tree when index is int, but got {type(value)}"
            self.batch_node_value[index] = value.node_value
            self.batch_node_type[index] = value.node_type
            self.batch_subtree_size[index] = value.subtree_size
        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            assert isinstance(
                value, Forest
            ), f"value should be Forest when index is slice, but got {type(value)}"
            self.batch_node_value[index] = value.batch_node_value
            self.batch_node_type[index] = value.batch_node_type
            self.batch_subtree_size[index] = value.batch_subtree_size
        else:
            raise NotImplementedError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < self.pop_size:
            res = Tree(
                self.input_len,
                self.output_len,
                self.batch_node_value[self.iter_index],
                self.batch_node_type[self.iter_index],
                self.batch_subtree_size[self.iter_index],
            )
            self.iter_index += 1
            return res
        else:
            raise StopIteration

    def __str__(self):
        res = "[\n"
        for tree in self:
            res += f"  {str(tree)}, \n"
        res += "]"
        return res

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.pop_size

    def __add__(self, other):
        assert isinstance(
            other, Forest
        ), f"other should be Forest, but got {type(other)}"
        return Forest(
            self.input_len,
            self.output_len,
            torch.cat([self.batch_node_value, other.batch_node_value], dim=0),
            torch.cat([self.batch_node_type, other.batch_node_type], dim=0),
            torch.cat([self.batch_subtree_size, other.batch_subtree_size], dim=0),
        )
