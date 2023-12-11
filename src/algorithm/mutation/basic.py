import jax

from .base import Mutation
from src.cuda.utils import tree_size
from src.cuda.operations import mutation, generate


class BasicMutation(Mutation):

    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        tree_sizes = jax.vmap(tree_size)(trees)

        k1, k2, k3 = jax.random.split(key, 3)
        consts = const(k1)
        sub_trees = generate(
            key=k2,
            leaf_prob=config["leaf_prob"],
            funcs_prob_acc=config["func_prob_cdf"],
            const_samples=consts,
            pop_size=config["pop_size"],
            max_len=config["max_sub_tree_len"],
            num_inputs=config["num_inputs"],
            num_outputs=config["num_outputs"],
            output_prob=config["output_prob"],
            const_prob=config["const_prob"]
        )

        indices = jax.random.randint(k2, (trees.shape[0],), 0, tree_sizes)

        return mutation(
            trees,
            indices,
            sub_trees,
        )
