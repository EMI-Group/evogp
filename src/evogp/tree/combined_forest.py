from typing import Optional, Tuple, Callable, List, Tuple, Union
import time
import torch
from torch import Tensor
import numpy as np

from evogp.tree.utils import check_tensor

from .forest import Forest
from .descriptor import GenerateDescriptor
from .combined_tree import CombinedTree


class CombinedForest:
    def __init__(
        self,
        forests: List[Forest],
        data_info: dict,
    ):
        self.data_info = data_info
        self.forests = forests
        self.output_names = list(data_info.keys())
        self.input_names = []
        for vals in data_info.values():
            self.input_names.extend(vals)
        self.input_names = list(set(self.input_names))

        self.input_len = len(self.input_names)
        self.output_len = len(self.output_names)

        self.pop_size = forests[0].pop_size

    @staticmethod
    def random_generate(
        pop_size: int,
        data_info: dict,
        descriptors: Union[List, GenerateDescriptor],
    ):
        if isinstance(descriptors, GenerateDescriptor):
            descriptors = [descriptors] * len(data_info)

        assert isinstance(descriptors, list) and len(descriptors) == len(
            data_info
        ), f"there are {len(data_info)} sub_forests, but got {len(descriptors)} descriptors"

        for i, (key, vals) in enumerate(data_info.items()):
            # check input_len
            assert descriptors[i].input_len == len(vals), "input size not match"
            # check output_len
            assert descriptors[i].output_len == 1, "output size mush be 1"

        forests = [
            Forest.random_generate(pop_size=pop_size, descriptor=d) for d in descriptors
        ]

        return CombinedForest(
            forests=forests,
            data_info=data_info,
        )

    # (pop_size, input_len) -> (pop_size, output_len)
    def forward(
        self,
        x: dict[str, torch.Tensor],
    ):
        outputs = {}
        for i, f in enumerate(self.forests):
            out_name = self.output_names[i]
            inputs = [x[name][:, None] for name in self.data_info[out_name]]
            inputs_torch = torch.cat(inputs, dim=1)
            outputs[out_name] = f.forward(inputs_torch)

        return outputs

    # (batch_size, input_len) -> (pop_size, batch_size, output_len)
    def batch_forward(
        self,
        x: dict[str, torch.Tensor],
    ):
        outputs = {}
        for i, f in enumerate(self.forests):
            out_name = self.output_names[i]
            inputs = [x[name][:, None] for name in self.data_info[out_name]]
            inputs_torch = torch.cat(inputs, dim=1)
            outputs[out_name] = f.batch_forward(inputs_torch)

        return outputs

    def __getitem__(self, index):
        if isinstance(index, int):
            return CombinedTree(
                data_info=self.data_info,
                trees=[f[index] for f in self.forests],
            )
        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            return CombinedForest(
                data_info=self.data_info,
                forests=[f[index] for f in self.forests],
            )
        else:
            raise Exception("Do not support index type {}".format(type(index)))

    def __setitem__(self, index, value):
        if isinstance(index, int):
            assert isinstance(
                value, CombinedTree
            ), f"value should be Tree when index is int, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.trees[i]

        elif (
            isinstance(index, slice)
            or isinstance(index, Tensor)
            or isinstance(index, np.ndarray)
        ):
            assert isinstance(
                value, CombinedForest
            ), f"value should be Forest when index is slice, but got {type(value)}"
            for i in range(len(self.forests)):
                self.forests[i][index] = value.forests[i]
        else:
            raise NotImplementedError

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < self.pop_size:
            return self[self.iter_index]
        else:
            raise StopIteration

    def __len__(self):
        return self.pop_size

    def __add__(self, other):
        assert self.data_info == other.data_info
        if isinstance(other, CombinedForest):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.forests[i])
            return CombinedForest(new_forests, self.data_info)
        if isinstance(other, CombinedTree):
            new_forests = []
            for i in range(len(self.forests)):
                new_forests.append(self.forests[i] + other.tree[i])
            return CombinedForest(new_forests, self.data_info)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)
