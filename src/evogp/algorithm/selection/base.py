from typing import Tuple
from evogp.core import Forest


class BaseSelection:
    def __call__(self, forest: Forest) -> Tuple[Forest, Forest]:
        raise NotImplementedError
