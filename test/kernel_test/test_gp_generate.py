import sys

import jax.numpy as jnp

import time
from functools import partial
import jax

from src.core.kernel.gp_kernel_bind import gp_generate
from src.core.kernel.utils import *


@partial(jax.jit, static_argnums=list(range(3, 10)))
def gp_generate_(
    depth_to_leaf_prob,
    functions_prob_accumulate,
    const_samples,
    pop_size,
    max_prefix_len,
    variable_len,
    output_len,
    output_prob,
    const_prob,
    seed
):
    return gp_generate(
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        pop_size=pop_size,
        max_len=max_prefix_len,
        variable_len=variable_len,
        output_len=output_len,
        output_prob=output_prob,
        const_prob=const_prob,
        seed=seed,
    )


########
# Test #
########


################
# Further Test #
################

depth_to_leaf_prob = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32
)
functions_prob_accumulate = jnp.array(
    [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.82,
        0.84,
        0.86,
        0.88,
        0.9,
        0.92,
        0.94,
        0.96,
        0.98,
        0.99,
        1.0,
    ],
    dtype=jnp.float32,
)
const_samples = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.float32)
N = 200_000
max_prefix_len = 1024
variable_len = 2
output_len = 4
output_prob = 0.5
const_prob = 0.5
seed = 42

gps = gp_generate_(
    depth_to_leaf_prob,
    functions_prob_accumulate,
    const_samples,
    N,
    max_prefix_len,
    variable_len,
    output_len,
    output_prob,
    const_prob,
    seed
)
gps.block_until_ready()

t = time.time()
gps = gp_generate_(
    depth_to_leaf_prob,
    functions_prob_accumulate,
    const_samples,
    N,
    max_prefix_len,
    variable_len,
    output_len,
    output_prob,
    const_prob,
    seed
)
print(time.time() - t)
_, a, b, _ = from_cuda_node(gps)
print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])
