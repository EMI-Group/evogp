from typing import Optional, Tuple
import time
import torch
from torch import Tensor
import numpy as np
from .utils import *
from .tree import Tree
from .descriptor import GenerateDescriptor


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
        self.pop_size, self.max_tree_len = batch_node_value.shape

        assert batch_node_value.shape == (
            self.pop_size,
            self.max_tree_len,
        ), f"node_value shape should be ({self.pop_size}, {self.max_tree_len}), but got {batch_node_value.shape}"
        assert batch_node_type.shape == (
            self.pop_size,
            self.max_tree_len,
        ), f"node_type shape should be ({self.pop_size}, {self.max_tree_len}), but got {batch_node_type.shape}"
        assert batch_subtree_size.shape == (
            self.pop_size,
            self.max_tree_len,
        ), f"subtree_size shape should be ({self.pop_size}, {self.max_tree_len}), but got {batch_subtree_size.shape}"

        self.batch_node_value = batch_node_value
        self.batch_node_type = batch_node_type
        self.batch_subtree_size = batch_subtree_size

    @staticmethod
    def random_generate(
        pop_size: int,
        descriptor: GenerateDescriptor,
    ) -> "Forest":
        assert (
            isinstance(pop_size, int) and pop_size > 0
        ), "pop_size should be a positive integer"

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
            descriptor.max_tree_len,
            descriptor.input_len,
            descriptor.output_len,
            descriptor.const_samples.shape[0],
            descriptor.out_prob,
            descriptor.const_prob,
            keys,
            descriptor.depth2leaf_probs,
            descriptor.roulette_funcs,
            descriptor.const_samples,
        )

        return Forest(
            descriptor.input_len,
            descriptor.output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        )

    @staticmethod
    def zero_generate(
        pop_size: int,
        max_tree_len: int,
        input_len: int,
        output_len: int,
    ):
        batch_node_value = torch.zeros(
            (pop_size, max_tree_len), dtype=torch.float32, device="cuda"
        )
        batch_node_type = torch.zeros(
            (pop_size, max_tree_len), dtype=torch.int16, device="cuda"
        )
        batch_subtree_size = torch.zeros(
            (pop_size, max_tree_len), dtype=torch.int16, device="cuda"
        )
        batch_node_type[:, 0] = NType.CONST
        batch_subtree_size[:, 0] = 1
        return Forest(
            input_len,
            output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluate the expression forest.

        Args:
            x: The input values. Shape should be (pop_size, input_len).

        Returns:
            The output values. Shape is (pop_size, output_len).
        """
        x = check_tensor(x)

        assert x.shape == (
            self.pop_size,
            self.input_len,
        ), f"x shape should be ({self.pop_size}, {self.input_len}), but got {x.shape}"

        res = torch.ops.evogp_cuda.tree_evaluate(
            self.pop_size,
            self.max_tree_len,
            self.input_len,
            self.output_len,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            x,
        )

        return res

    # (batch_size, input_len) -> (pop_size, batch_size, output_len)
    def batch_forward(self, x: Tensor) -> Tensor:
        x = check_tensor(x)

        assert (x.dim() == 2) and (
            x.shape[1] == self.input_len
        ), f"x shape[1] should be {self.input_len}, but got {x.shape[1]}"

        batch_size = x.shape[0]
        assist_batch_node_value = self.batch_node_value.repeat_interleave(
            batch_size, dim=0
        )
        assist_batch_node_type = self.batch_node_type.repeat_interleave(
            batch_size, dim=0
        )
        assist_batch_subtree_size = self.batch_subtree_size.repeat_interleave(
            batch_size, dim=0
        )

        assist_x = x.repeat(self.pop_size, 1)

        assist_res = torch.ops.evogp_cuda.tree_evaluate(
            self.pop_size * batch_size,
            self.max_tree_len,
            self.input_len,
            self.output_len,
            assist_batch_node_value,
            assist_batch_node_type,
            assist_batch_subtree_size,
            assist_x,
        )

        res = assist_res.reshape(self.pop_size, batch_size, self.output_len)

        return res

    def mutate(self, replace_pos: Tensor, new_sub_forest: "Forest") -> "Forest":
        """
        Mutate the current forest by replacing subtrees at specified positions
        with new subtrees from a new_sub_forest.

        Args:
            replace_pos: A tensor indicating the positions to replace.
            new_sub_forest: A Forest containing new subtrees for replacement.

        Returns:
            A new mutated Forest object.
        """
        replace_pos = check_tensor(replace_pos)

        # Validate shapes and dimensions
        assert replace_pos.shape == (
            self.pop_size,
        ), f"replace_pos shape should be ({self.pop_size}, ), but got {replace_pos.shape}"
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
            self.max_tree_len == new_sub_forest.max_tree_len
        ), f"max_tree_len should be {self.max_tree_len}, but got {new_sub_forest.max_tree_len}"

        # Perform mutation operation using CUDA
        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        ) = torch.ops.evogp_cuda.tree_mutate(
            self.pop_size,
            self.max_tree_len,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            replace_pos,
            new_sub_forest.batch_node_value,
            new_sub_forest.batch_node_type,
            new_sub_forest.batch_subtree_size,
        )

        # Return a new Forest object with the mutated trees
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
    ) -> "Forest":
        """
        Perform crossover operation.

        Args:
            left_indices (Tensor): indices of trees to be used as the left parent
            right_indices (Tensor): indices of trees to be used as the right parent
            left_pos (Tensor): subtree position in the left parent where the crossover happens
            right_pos (Tensor): subtree position in the right parent where the crossover happens

        Returns:
            Forest: a new Forest object with the crossovered trees
        """
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

        res_forest_size = left_indices.shape[0]

        (
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
        ) = torch.ops.evogp_cuda.tree_crossover(
            self.pop_size,
            res_forest_size,
            self.max_tree_len,
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

    def SR_fitness(
        self,
        inputs: Tensor,
        labels: Tensor,
        use_MSE: bool = True,
        execute_mode: str = "auto",
    ) -> Tensor:
        """
        Calculate the fitness of the current population using the SR metric.

        Args:
            inputs (Tensor): inputs to the GP trees
            labels (Tensor): labels to compute the fitness
            use_MSE (bool, optional): whether to use the Mean Squared Error (MSE) as the fitness metric. Defaults to True.

        Returns:
            Tensor: a tensor of shape (pop_size,) containing the fitness values
        """
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

        assert execute_mode in [
            "hybrid parallel",
            "data parallel",
            "tree parallel",
            "auto",
        ], f"execute_mode should be one of ['hybrid parallel', 'data parallel', 'tree parallel', 'auto'], but got {execute_mode}"

        if execute_mode == "hybrid parallel":
            execute_code = 0
        elif execute_mode == "data parallel":
            execute_code = 1
        elif execute_mode == "tree parallel":
            execute_code = 2
        elif execute_mode == "auto":
            execute_code = 4
        batch_size = inputs.shape[0]

        # Perform SR fitness computation using CUDA
        res = torch.ops.evogp_cuda.tree_SR_fitness(
            self.pop_size,
            batch_size,
            self.max_tree_len,
            self.input_len,
            self.output_len,
            use_MSE,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            inputs,
            labels,
            execute_code,
        )

        return res

    def __getitem__(self, index):
        if isinstance(index, int) or (hasattr(index, "shape") and index.shape == ()):
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
            raise Exception("Do not support index type {}".format(type(index)))

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
        res = f"Forest(pop size: {self.pop_size})\n"
        res += "[\n"
        for tree in self:
            res += f"  {str(tree)}, \n"
        res += "]"
        return res

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.pop_size

    def __add__(self, other):
        assert other.input_len == self.input_len
        assert other.output_len == self.output_len

        if isinstance(other, Forest):
            return Forest(
                self.input_len,
                self.output_len,
                torch.cat([self.batch_node_value, other.batch_node_value], dim=0),
                torch.cat([self.batch_node_type, other.batch_node_type], dim=0),
                torch.cat([self.batch_subtree_size, other.batch_subtree_size], dim=0),
            )
        if isinstance(other, Tree):
            return Forest(
                self.input_len,
                self.output_len,
                torch.cat(
                    [self.batch_node_value, other.node_value.unsqueeze(0)], dim=0
                ),
                torch.cat([self.batch_node_type, other.node_type.unsqueeze(0)], dim=0),
                torch.cat(
                    [self.batch_subtree_size, other.subtree_size.unsqueeze(0)], dim=0
                ),
            )
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __getstate__(self):
        return {
            "input_len": self.input_len,
            "output_len": self.output_len,
            "batch_node_value": self.batch_node_value.cpu().numpy(),
            "batch_node_type": self.batch_node_type.cpu().numpy(),
            "batch_subtree_size": self.batch_subtree_size.cpu().numpy(),
        }

    def __setstate__(self, state):
        self.input_len = state["input_len"]
        self.output_len = state["output_len"]
        self.pop_size, self.max_tree_len = state["batch_node_value"].shape
        self.batch_node_value = (
            torch.from_numpy(state["batch_node_value"]).to("cuda").requires_grad_(False)
        )
        self.batch_node_type = (
            torch.from_numpy(state["batch_node_type"]).to("cuda").requires_grad_(False)
        )
        self.batch_subtree_size = (
            torch.from_numpy(state["batch_subtree_size"])
            .to("cuda")
            .requires_grad_(False)
        )
