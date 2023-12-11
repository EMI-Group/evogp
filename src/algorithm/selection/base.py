from typing import Tuple
import jax


class Selection:

    def __init__(self):
        pass

    def __call__(self, key, fitnesses: jax.Array, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (left indices, right indices)
        raise NotImplementedError
