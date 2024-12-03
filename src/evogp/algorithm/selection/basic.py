from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection

class BasicSelection(Selection):
    """
    save the elites and choose the survivors by random
    """
    def __init__(
            self,
            elite_rate: float,
            survivor_rate: float,
    ):
        super().__init__()
        self.elite_rate = elite_rate
        self.survivor_rate = survivor_rate

    def __call__(self, key, trees, fitnesses, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (trees, fitness)
        
        idx = jnp.arange(0, config["pop_size"])
        idx_rank = jnp.argsort(fitnesses)[::-1] # the index of individual from high to low
        rank = jnp.argsort(idx_rank) # the rank of every individual

        # select elites
        elite_num = int(config["pop_size"] * self.elite_rate)
        zeros = jnp.zeros(config["pop_size"], dtype=jnp.int32)
        elite_idx = jnp.where(idx < elite_num, idx_rank, zeros)

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

        selected_idx = jnp.where(idx < elite_num, elite_idx, survivor_idx)
        return trees[selected_idx], fitnesses[selected_idx]