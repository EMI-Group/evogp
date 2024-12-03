from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection

class RouletteSelection(Selection):

    def __init__(
            self,
    ):
        super().__init__()

    def __call__(self, key, trees, fitnesses, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (trees, fitness)

        selected_idx = jax.random.choice(
            key,
            jnp.arange(0, config["pop_size"]),
            shape=(config["pop_size"],),
            replace=True,
            p = fitnesses / jnp.sum(fitnesses),
        )
        return trees[selected_idx], fitnesses[selected_idx]