import jax
import jax.numpy as jnp

from .base import Mutation
from evogp.cuda.utils import tree_size
from evogp.cuda.operations import mutation, generate


class SubtreeMutation(Mutation):
    """
    Mutate with a randomly generated subtree
    """

    def __init__(self, max_sub_tree_len, leaf_prob):
        super().__init__()
        self.max_sub_tree_len = max_sub_tree_len
        self.leaf_prob = jnp.array(leaf_prob, dtype=jnp.float32)

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        tree_sizes = jax.vmap(tree_size)(trees)

        k1, k2, k3 = jax.random.split(key, 3)
        consts = const(k1)
        sub_trees = generate(
            key=k2,
            leaf_prob=self.leaf_prob,
            funcs_prob_acc=config["func_prob_cdf"],
            const_samples=consts,
            pop_size=config["pop_size"],
            max_len=self.max_sub_tree_len,
            num_inputs=config["num_inputs"],
            num_outputs=config["num_outputs"],
            output_prob=config["output_prob"],
            const_prob=config["const_prob"],
        )

        indices = jax.random.randint(k2, (trees.shape[0],), 0, tree_sizes)

        sub_trees = mutation(trees, indices, sub_trees)

        return sub_trees
