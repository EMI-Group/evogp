from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor

from . import BaseMutation, DefaultMutation
from ...tree import CombinedForest, MAX_STACK, GenerateDescriptor


class CombinedDefaultMutation(BaseMutation):
    def __init__(
        self,
        mutation_rate: float,
        descriptors: Union[List, GenerateDescriptor],
    ):
        # lazy load pattern
        self.pattern_num = None
        self.mutation_rate = mutation_rate

        self.descriptors = descriptors

    def __call__(self, combined_forest: CombinedForest):
        # lazy load pattern
        current_pattern_num = len(combined_forest.forests)
        self.load_pattern_num(current_pattern_num)

        new_forests = []
        for i in range(len(combined_forest.forests)):
            new_forests.append(self.default_mutations[i](combined_forest.forests[i]))

        return CombinedForest(
            new_forests, combined_forest.data_info
        )

    def load_pattern_num(self, current_pattern_num):
        if self.pattern_num is None:
            self.pattern_num = current_pattern_num
            if isinstance(self.descriptors, GenerateDescriptor):
                self.descriptors = [self.descriptors] * self.pattern_num
            elif isinstance(self.descriptors, list):
                assert (
                    len(self.descriptors) == self.pattern_num
                ), f"the length of descriptors should be {self.pattern_num}, but got {len(self.descriptors)}"

            self.default_mutations = [
                DefaultMutation(self.mutation_rate / self.pattern_num, d)
                for d in self.descriptors
            ]
        else:
            assert (
                self.pattern_num == current_pattern_num
            ), f"the pattern_num should be {self.pattern_num}, but got {current_pattern_num}"
