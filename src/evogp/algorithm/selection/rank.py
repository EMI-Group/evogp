from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection


class RankSelection(Selection):
    """
    Args:
      selection_pressure: the range is [0, 1].
        0 means no selection pressure.
        1 means high selection pressure.
    """
    def __init__(
            self,
            selection_pressure: float,
    ):
        super().__init__()
        assert 0 <= selection_pressure <= 1
        self.sp = selection_pressure

    def __call__(self, key, trees, fitnesses, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (trees, fitness)

        idx = jnp.arange(0, config["pop_size"])
        idx_rank = jnp.argsort(fitnesses)[::-1] # the index of individual from high to low
        rank = jnp.argsort(idx_rank) # the rank of every individual

        n = config["pop_size"]
        selected_idx = jax.random.choice(
            key,
            idx,
            shape=(config["pop_size"],),
            replace=True,
            p = (1 / n) * (1 + self.sp * (1 - 2 * rank / (n - 1))),
        )
        return trees[selected_idx], fitnesses[selected_idx]