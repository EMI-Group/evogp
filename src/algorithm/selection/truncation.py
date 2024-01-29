from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection

class TruncationSelection(Selection):

    def __init__(
            self,
            survivor_rate: float,
    ):
        super().__init__()
        self.survivor_rate = survivor_rate

    def __call__(self, key, trees, fitnesses, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (trees, fitness)

        idx = jnp.arange(0, config["pop_size"])
        idx_rank = jnp.argsort(fitnesses)[::-1] # the index of individual from high to low
        rank = jnp.argsort(idx_rank) # the rank of every individual

        # select survivors
        survivor_num = int(config["pop_size"] * self.survivor_rate)
        survivor_mask = rank < survivor_num
        survivor_idx = jax.random.choice(
            key,
            idx,
            shape=(config["pop_size"],),
            replace=True,
            p = survivor_mask / jnp.sum(survivor_mask),
        )
        return trees[survivor_idx], fitnesses[survivor_idx]