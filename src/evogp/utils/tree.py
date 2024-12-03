"""GP Tree representation in JAX"""
import jax
import jax.numpy as jnp
from jax import lax

class Tree:
    def __init__(self, node_values, node_types, subtree_sizes, output_indices):
        self.node_values = node_values
        self.node_types = node_types
        self.subtree_sizes = subtree_sizes
        self.output_indices = output_indices

    def tree_flatten(self):
        all_data = self.node_values, self.node_types, self.subtree_sizes, self.output_indices
        return all_data
    
    def subtree(self, node_index, max_len = 1024):
        # Try not to change the **max_len** parameter or it will cause recompilation.
        subtree_size = self.subtree_sizes[node_index]
    
        def body_fun(i, val):
            sub_arr, arr = val
            return (sub_arr.at[i].set(arr[node_index + i]), arr)

        node_values = jnp.zeros(max_len, dtype=jnp.float32)
        node_types, subtree_sizes, output_indices = jnp.zeros((3, max_len), dtype=jnp.int16)

        node_values, _ = lax.fori_loop(0, subtree_size, body_fun, (node_values, self.node_values))
        node_types, _ = lax.fori_loop(0, subtree_size, body_fun, (node_types, self.node_types))
        subtree_sizes, _ = lax.fori_loop(0, subtree_size, body_fun, (subtree_sizes, self.subtree_sizes))
        output_indices, _ = lax.fori_loop(0, subtree_size, body_fun, (output_indices, self.output_indices))
        return Tree(node_values, node_types, subtree_sizes, output_indices)