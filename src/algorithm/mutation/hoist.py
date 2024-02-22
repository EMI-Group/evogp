import jax
import jax.numpy as jnp

from .base import Mutation
from src.cuda.utils import from_cuda_node, to_cuda_node_multi_output
from src.cuda.operations import mutation
from src.utils import sub_arr

class HoistMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        pop_size = config["pop_size"]
        max_len = config["max_len"]
        consts = const(k1)
        
        # to JAX
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)

        # the first time to extract the subtrees
        mutated_node_index = jax.random.randint(k2, shape=(pop_size,), minval=0, maxval=subtree_sizes[:,0])
        new_arr_sizes = subtree_sizes[jnp.arange(pop_size), mutated_node_index]
        node_values = sub_arr(node_values, mutated_node_index, new_arr_sizes, max_len)
        node_types = sub_arr(node_types, mutated_node_index, new_arr_sizes, max_len)
        subtree_sizes = sub_arr(subtree_sizes, mutated_node_index, new_arr_sizes, max_len)
        output_indices = sub_arr(output_indices, mutated_node_index, new_arr_sizes, max_len)
        
        # the second time to extract the subtrees
        hoisted_node_index = jax.random.randint(k3, shape=(pop_size,), minval=0, maxval=subtree_sizes[:,0])
        new_arr_sizes = subtree_sizes[jnp.arange(pop_size), hoisted_node_index]
        node_values = sub_arr(node_values, hoisted_node_index, new_arr_sizes, max_len)
        node_types = sub_arr(node_types, hoisted_node_index, new_arr_sizes, max_len)
        subtree_sizes = sub_arr(subtree_sizes, hoisted_node_index, new_arr_sizes, max_len)
        output_indices = sub_arr(output_indices, hoisted_node_index, new_arr_sizes, max_len)

        # to CUDA
        vmap_to_cuda_node = jax.vmap(to_cuda_node_multi_output, in_axes=(0, 0, 0, 0, None))
        subtrees = vmap_to_cuda_node(
            node_values, 
            node_types, 
            subtree_sizes, 
            output_indices, 
            config["num_outputs"])
        return mutation(trees, mutated_node_index, subtrees)