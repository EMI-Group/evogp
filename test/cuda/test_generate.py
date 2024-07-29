import time
import sys

sys.path.append("/wuzhihong/TensorGP")
from src.cuda.operations import generate
from src.cuda.utils import *


def test():
    depth_to_leaf_prob = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
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
    key = jax.random.PRNGKey(42)

    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        variable_len,
        output_len,
        output_prob,
        const_prob,
    )
    gps.block_until_ready()

    t = time.time()
    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        variable_len,
        output_len,
        output_prob,
        const_prob,
    )
    print(time.time() - t)
    _, a, b, _ = from_cuda_node(gps)
    print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
    print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def test2():
    k1 = jax.random.PRNGKey(42)
    for i in range(10000):
        k1, k2 = jax.random.split(k1, 2)
        sub_trees = generate(
            key=k2,
            leaf_prob=jnp.array([0. , 0. , 0. , 0. , 0.1, 0.2, 1. , 1. , 1. , 1. ]),
            funcs_prob_acc=jnp.array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
        1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ]),
            const_samples=jnp.array([-1.,  0.,  1.]),
            pop_size=10,
            max_len=128,
            num_inputs=27,
            num_outputs=10,
            output_prob=0.8,
            const_prob=0.5,
        )
        print(i)
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(sub_trees)
        
        non_negative_one_indices = jnp.where(output_indices != -1)

        print(non_negative_one_indices)


if __name__ == "__main__":
    test2()