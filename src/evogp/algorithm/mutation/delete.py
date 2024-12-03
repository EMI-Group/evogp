import jax
import jax.numpy as jnp

from .base import Mutation
from evogp.cuda.utils import from_cuda_node, to_cuda_node_multi_output
from evogp.cuda.utils import tree_size
from evogp.cuda.operations import mutation, generate
from evogp.utils import sub_arr

class DeleteMutation(Mutation):
    """
    Delete the node of specific index and replace it with its random child
    It's too slow
    """
    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3 = jax.random.split(key, 3)
        pop_size = config["pop_size"]
        max_len = config["max_len"]
        consts = const(k1)

        def choose_nonzero_index(row, key):
            random_values = jax.random.uniform(key, row.shape)
            random_values_adjusted = jnp.where(row == 0, -jnp.inf, random_values)
            chosen_index = jnp.argmax(random_values_adjusted)
            return chosen_index
        
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)
        keys = jax.random.split(k2, subtree_sizes.shape[0])
        indices = jax.vmap(choose_nonzero_index)(subtree_sizes, keys)

        rows = jnp.arange(pop_size)
        child_nums = node_types[rows, indices]
        nth_childs = jax.random.randint(k3, (pop_size,), 0, child_nums - 1)

        def jump_fn(i, row_and_idx):
            row, idx = row_and_idx
            jump_distance = row[idx] + 1
            new_idx = idx + jump_distance
            return row, new_idx

        def process_row(row, init_idx, jumps):
            _, final_idx = jax.lax.fori_loop(0, jumps, lambda n: n + I.s[n], (row, init_idx))
            return final_idx

        child_indices = jax.vmap(process_row)(subtree_sizes, indices + 1, nth_childs)

        """get the sub_trees"""
        # to JAX
        # node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)

        # extract the subtrees
        new_arr_sizes = subtree_sizes[jnp.arange(pop_size), child_indices]
        node_values = sub_arr(node_values, child_indices, new_arr_sizes, max_len)
        node_types = sub_arr(node_types, child_indices, new_arr_sizes, max_len)
        subtree_sizes = sub_arr(subtree_sizes, child_indices, new_arr_sizes, max_len)
        output_indices = sub_arr(output_indices, child_indices, new_arr_sizes, max_len)

        # to CUDA
        vmap_to_cuda_node = jax.vmap(to_cuda_node_multi_output, in_axes=(0, 0, 0, 0, None))
        sub_trees = vmap_to_cuda_node(
            node_values, 
            node_types, 
            subtree_sizes, 
            output_indices, 
            config["num_outputs"])
        """"""
        
        return mutation(
            trees,
            indices,
            sub_trees
        )