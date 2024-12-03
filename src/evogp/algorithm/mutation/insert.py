import jax
import jax.numpy as jnp

from .base import Mutation
from evogp.cuda.utils import from_cuda_node, to_cuda_node_multi_output
from evogp.cuda.utils import tree_size
from evogp.cuda.operations import mutation, generate
from evogp.utils import sub_arr

class InsertMutation(Mutation):
    """
    Insert a Function Node before the node of specific index
    """
    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3 = jax.random.split(key, 3)
        pop_size = config["pop_size"]
        max_len = config["max_len"]
        consts = const(k1)
        trees_sizes = jax.vmap(tree_size)(trees)
        indices = jax.random.randint(k2, (pop_size,), 0, trees_sizes)

        new_trees = generate(
            key=k2,
            leaf_prob=jnp.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32),
            funcs_prob_acc=config["func_prob_cdf"],
            const_samples=consts,
            pop_size=config["pop_size"],
            max_len=config["max_sub_tree_len"],
            num_inputs=config["num_inputs"],
            num_outputs=config["num_outputs"],
            output_prob=config["output_prob"],
            const_prob=config["const_prob"]
        )
        new_trees_sizes = jax.vmap(tree_size)(new_trees)
        new_trees_indices = jax.random.randint(k3, (pop_size,), 1, new_trees_sizes)

        """get the sub_trees"""
        # to JAX
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)

        # extract the subtrees
        new_arr_sizes = subtree_sizes[jnp.arange(pop_size), indices]
        node_values = sub_arr(node_values, indices, new_arr_sizes, max_len)
        node_types = sub_arr(node_types, indices, new_arr_sizes, max_len)
        subtree_sizes = sub_arr(subtree_sizes, indices, new_arr_sizes, max_len)
        output_indices = sub_arr(output_indices, indices, new_arr_sizes, max_len)

        # to CUDA
        vmap_to_cuda_node = jax.vmap(to_cuda_node_multi_output, in_axes=(0, 0, 0, 0, None))
        sub_trees = vmap_to_cuda_node(
            node_values, 
            node_types, 
            subtree_sizes, 
            output_indices, 
            config["num_outputs"])
        """"""

        insert_trees = mutation(
            new_trees,
            new_trees_indices,
            sub_trees
        )

        return mutation(
            trees,
            indices,
            insert_trees
        )
