import jax
import jax.numpy as jnp

from .base import Crossover
from src.cuda.utils import tree_size
from src.cuda.operations import crossover


class BasicCrossover(Crossover):

    def __init__(self):
        super().__init__()

    def __call__(self, key, recipient_trees, donor_trees, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3 = jax.random.split(key, 3)
        pop_size = config["pop_size"]

        trees = jnp.concatenate([recipient_trees, donor_trees], axis=0)

        recipient_idx = jnp.arange(pop_size * 2, dtype=jnp.int32)
        donor_idx = jax.random.randint(k1, (pop_size,), pop_size, pop_size * 2, dtype=jnp.int32)
        donor_idx = jnp.pad(donor_idx, (0, pop_size), constant_values=(-1))

        trees_size = jax.vmap(tree_size)(trees)
        recipient_pos = jax.random.randint(k2, (pop_size * 2,), 0, trees_size[recipient_idx], dtype=jnp.int16)
        donor_pos = jax.random.randint(k3, (pop_size * 2,), 0, trees_size[donor_idx], dtype=jnp.int16)
        nodes = jnp.stack([recipient_pos, donor_pos], axis=1)

        trees = crossover(trees, recipient_idx, donor_idx, nodes)
        return trees[:pop_size]