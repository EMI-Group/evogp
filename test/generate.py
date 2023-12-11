import jax
import jax.numpy as jnp

from src.cuda import generate, from_cuda_node
from src.utils.tools import dict2cdf


key = jax.random.PRNGKey(0)
leaf_prob = jnp.array([0, 0, 0, 0, 1.0])
funcs_prob_acc = dict2cdf({
    "+": 0.25,
    "-": 0.5,
    "*": 0.75,
    "/": 1.0,
})
const_samples = jnp.array([-1.0, 0.0, 1.0])
pop_size = 2
max_len = 16
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

print(from_cuda_node(trees))

