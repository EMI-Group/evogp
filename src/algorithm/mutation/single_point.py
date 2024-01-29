import jax
import jax.numpy as jnp

from .base import Mutation
from src.cuda.utils import from_cuda_node, to_cuda_node_multi_output

class SinglePointMutation(Mutation):

    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        pop_size = config["pop_size"]
        num_node_type = jnp.array([config["num_inputs"], -1, 10, 11, 1]) # [VAR, CONST, UFUNC, BFUNC, TFUNC]
        rows = jnp.arange(pop_size)
        consts = const(k1)
        
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)

        # one node for every individual
        mutated_node_index = jax.random.randint(k2, shape=(pop_size,), minval=0, maxval=subtree_sizes[:,0])
        mutated_node_type = node_types[rows, mutated_node_index]
        mutated_node_value = jax.random.randint(k3, shape=(pop_size,), minval=0, maxval=num_node_type[mutated_node_type])
        random_const = jax.random.choice(k4, consts, shape=(pop_size,))
        mutated_node_value = jnp.where(mutated_node_type == 1, random_const, mutated_node_value)
        node_values = node_values.at[rows, mutated_node_index].set(mutated_node_value)

        vmap_to_cuda_node = jax.vmap(to_cuda_node_multi_output, in_axes=(0, 0, 0, 0, None))
        return vmap_to_cuda_node(node_values, node_types, subtree_sizes, output_indices, config["num_outputs"])