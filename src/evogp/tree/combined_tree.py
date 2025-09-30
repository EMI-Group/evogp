from typing import List, Union
import torch

from .descriptor import GenerateDescriptor


class CombinedTree:
    def __init__(self, trees, data_info):
        self.trees = trees
        self.data_info = data_info
        self.output_names = list(data_info.keys())
        self.input_names = []
        for vals in data_info.values():
            self.input_names.extend(vals)
        self.input_names = list(set(self.input_names))

        self.input_len = len(self.input_names)
        self.output_len = len(self.output_names)

        for i, names in enumerate(self.output_names):
            setattr(self, names, self.trees[i])

    @staticmethod
    def random_generate(
        descriptors: Union[List, GenerateDescriptor],
        data_info: dict,
    ):
        from .combined_forest import CombinedForest

        return CombinedForest.random_generate(
            pop_size=1,
            descriptors=descriptors,
            data_info=data_info,
        )[0]

    def forward(self, x: dict[str, torch.Tensor]):

        is_batch = list(x.values())[0].dim() == 2

        if not is_batch:
            pop_res = self.to_combined_forest().forward(x)
        else:
            pop_res = self.to_combined_forest().batch_forward(x)

        return pop_res[0]  # remove the pop dimension

    def to_combined_forest(self):
        from .combined_forest import CombinedForest

        return CombinedForest(
            forests=[tree.to_forest() for tree in self.trees],
            data_info=self.data_info,
        )