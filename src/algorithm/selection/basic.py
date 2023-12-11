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

        idx = jnp.arange(0, config["pop_size"])

        elite_num = jnp.array(config["pop_size"] * self.elite_rate, dtype=jnp.int32)
        survive_num = jnp.array(config["pop_size"] * self.survive_rate, dtype=jnp.int32)

        rank_idx = jax.numpy.argsort(fitnesses)[::-1]  # rank from high to low
        rank_score = jax.numpy.argsort(rank_idx)

        survive_mask = rank_score < survive_num

        left, right = jnp.zeros((2, config["pop_size"]), dtype=jnp.int32)

        # elite
        left = jnp.where(idx < elite_num, rank_idx, left)
        right = jnp.where(idx < elite_num, rank_idx, right)

        # left_elite = rank_idx[:elite_num]  these cause JAX IndexError as DynamicSliceOp
        # right_elite = rank_idx[:elite_num]

        # select father and mother from survivors
        left_survive, right_survive = jax.random.choice(
            key,
            idx,
            p=survive_mask / jnp.sum(survive_mask),
            shape=(2, config["pop_size"]),
            replace=True,
        )

        left = jnp.where(idx >= elite_num, left_survive, left)
        right = jnp.where(idx >= elite_num, right_survive, right)

        return left, right

