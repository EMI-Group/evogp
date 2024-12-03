import jax
import jax.numpy as jnp

from evogp.cuda import generate, from_cuda_node
from evogp.utils.tools import dict2cdf


key = jax.random.PRNGKey(0)
leaf_prob = jnp.array([0, 0, 0, 1.])
funcs_prob_acc = dict2cdf({
    "+": 0.25,
    "-": 0.25,
    "*": 0.25,
    "/": 0.25,
})
const_samples = jnp.array([-1.0, 0.0, 1.0])
pop_size = 1
max_len = 8
num_inputs = 2

trees = generate(
    key,
    leaf_prob,
    funcs_prob_acc,
    const_samples,
    pop_size=pop_size,
    max_len=max_len,
    num_inputs=num_inputs,
)

node_val, node_type, tree_size, output_index = from_cuda_node(trees)

print(node_val, node_type, tree_size, output_index, sep='\n')

