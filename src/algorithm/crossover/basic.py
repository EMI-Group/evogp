import jax
import jax.numpy as jnp

from .base import Crossover
from src.cuda.utils import tree_size
from src.cuda.operations import crossover


class BasicCrossover(Crossover):

    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, left, right, config) -> jax.Array:  # prefix_trees (in cuda)
        tree_sizes = jax.vmap(tree_size)(trees)

        k1, k2 = jax.random.split(key)

        l_pos = jax.random.randint(k1, (trees.shape[0],), 0, tree_sizes[left])
        r_pos = jax.random.randint(k2, (trees.shape[0],), 0, tree_sizes[right])

        nodes = jnp.stack([l_pos, r_pos], axis=1).astype(jnp.int16)

        return crossover(
            trees,
            left,
            right,
            nodes,
        )
