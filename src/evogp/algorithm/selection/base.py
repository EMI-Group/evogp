from typing import Tuple

from torch import Tensor

from evogp.tree import Forest


class BaseSelection:
    def __call__(self, forest: Forest) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
