from typing import Tuple
import jax
import jax.numpy as jnp
from .base import Selection


class BasicSelection(Selection):

    def __init__(
            self,
            elite_rate: float,
            survive_rate: float,
    ):
        super().__init__()
        self.elite_rate = elite_rate
        self.survive_rate = survive_rate

    def __call__(self, key, fitnesses: jax.Array, config) -> Tuple[jax.Array, jax.Array]:
        # returns: (left indices, right indices)

        elite_num = jnp.array(config["pop_size"] * self.elite_rate, dtype=jnp.int32)
        survive_num = jnp.array(config["pop_size"] * self.survive_rate, dtype=jnp.int32)

        rank_idx = jax.numpy.argsort(fitnesses)[::-1]  # rank from high to low
        rank_score = jax.numpy.argsort(rank_idx)

        survive_mask = rank_score < survive_num

        left_elite = rank_idx[:elite_num]
        right_elite = rank_idx[:elite_num]

        left_survive, right_survive = jax.random.choice(
            key,
            jnp.arange(0, config["pop_size"]),
            p=survive_mask / jnp.sum(survive_mask),
            shape=(2, config["pop_size"] - elite_num),
            replace=True,
        )

        return (
            jnp.concatenate([left_elite, left_survive]),
            jnp.concatenate([right_elite, right_survive]),
        )

