from timer import timer

import numpy as np
from evogp.cuda.operations import sr_fitness, generate, forward, constant_sr_fitness
from evogp.cuda.utils import *
from evogp.utils import dict2cdf

POP_SIZE = int(1000)
DEPTH_TO_LEAF_PROB = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 1], dtype=jnp.float32
)
FUNCS_CUM_PROB = dict2cdf({
    "+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25
})
MAX_LEN = 32
INPUT_LEN = 2
OUTPUT_LEN = 1
OUTPUT_PROB = 0.5
CONST_PROB = 0.5
SEED = 42

DATA_POINT_NUM = int(4096*4096)

def generate_dataset():
    inputs = jnp.array([0., 1.])
    targets = jnp.array([0.])
    inputs = jnp.tile(inputs, (DATA_POINT_NUM, 1))
    outputs = jnp.tile(targets, (DATA_POINT_NUM, 1))
    return inputs, outputs


if __name__ == '__main__':
    node_types = [3, 0, 3, 3, 0, 0, 1, 0]
    node_values = [1, 0, 1, 3, 1, 1, 1, 0]
    subtree_sizes = [7, 1, 5, 3, 1, 1, 1, 0]

    jax_node_types = jnp.array(node_types + [0] * (MAX_LEN - len(node_types)), dtype=jnp.float32)
    jax_node_values = jnp.array(node_values + [0] * (MAX_LEN - len(node_values)), dtype=jnp.float32)
    jax_subtree_sizes = jnp.array(subtree_sizes + [0] * (MAX_LEN - len(subtree_sizes)), dtype=jnp.float32)

    gp_tree = to_cuda_node(jax_node_values, jax_node_types, jax_subtree_sizes)
    trees = jnp.tile(gp_tree, (POP_SIZE, 1))
    print(trees.shape)

    # trees, cost_time = timer(generate_tree)
    # print(f"Generate {POP_SIZE} trees, cost time {cost_time}s")

    (inputs, targets), cost_time = timer(generate_dataset)
    print(f"Generate {DATA_POINT_NUM} data, cost time {cost_time}s")

    forward_res = forward(trees, inputs[0:POP_SIZE])

    jit_sr_fitness = jax.jit(sr_fitness)

    def block_sr_fitness(trees, inputs, targets):
        return jit_sr_fitness(trees, inputs, targets).block_until_ready()

    res, cost_time = timer(block_sr_fitness, trees, inputs, targets)
    print(f"Calculate sr_fitness, cost time {cost_time}s")

    res, cost_time = timer(block_sr_fitness, trees, inputs, targets)
    print(f"Calculate sr_fitness second time, cost time {cost_time}s")

    jit_constant_sr_fitness = jax.jit(constant_sr_fitness)

    def block_sr_fitness(trees, inputs, targets):
        return jit_constant_sr_fitness(trees, inputs, targets).block_until_ready()

    res, cost_time = timer(block_sr_fitness, trees, inputs, targets)
    print(f"Calculate constant sr_fitness, cost time {cost_time}s")

    res, cost_time = timer(block_sr_fitness, trees, inputs, targets)
    print(f"Calculate constant sr_fitness second time, cost time {cost_time}s")