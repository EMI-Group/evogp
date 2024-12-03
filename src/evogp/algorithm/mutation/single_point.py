import jax
import jax.numpy as jnp

from .base import Mutation
from evogp.cuda.utils import from_cuda_node, to_cuda_node_multi_output
from evogp.utils.enum import FUNCS
from functools import partial

class SinglePointMutation(Mutation):

    def __init__(self):
        super().__init__()

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        pop_size = config["pop_size"]
        rows = jnp.arange(pop_size)
        vars = jnp.arange(config["num_inputs"])
        consts = const(k1)
        funcs = jnp.arange(len(FUNCS))

        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(trees)

        # one node for every individual
        mutated_node_index = jax.random.randint(k2, shape=(pop_size,), minval=0, maxval=subtree_sizes[:, 0])
        mutated_node_type = node_types[rows, mutated_node_index]

        @partial(jax.vmap, in_axes=(0, 0), out_axes=0)
        def random_choice_funcs(key, prob):
            return jax.random.choice(key, funcs, p=prob)

        keys = jax.random.split(k3, pop_size)
        probs = jnp.tile(config["func_prob"], (pop_size, 1))
        type_mask = jnp.array([[0, 1], [0, 1], [12, 22], [1, 12], [0, 1]])  # [VAR, CONST, UFUNC, BFUNC, TFUNC]
        mask_ranges = type_mask[mutated_node_type]
        masks = (funcs >= mask_ranges[:, 0, None]) & (funcs < mask_ranges[:, 1, None])
        probs = probs * masks
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        mutated_node_value = random_choice_funcs(keys, probs)
        random_var = jax.random.choice(k4, vars, shape=(pop_size,))
        mutated_node_value = jnp.where(mutated_node_type == 0, random_var, mutated_node_value)
        random_const = jax.random.choice(k5, consts, shape=(pop_size,))
        mutated_node_value = jnp.where(mutated_node_type == 1, random_const, mutated_node_value)

        node_values = node_values.at[rows, mutated_node_index].set(mutated_node_value)

        vmap_to_cuda_node = jax.vmap(to_cuda_node_multi_output, in_axes=(0, 0, 0, 0, None))
        return vmap_to_cuda_node(node_values, node_types, subtree_sizes, output_indices, config["num_outputs"])
