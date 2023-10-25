import sys

import jax.numpy as jnp

sys.path.append("./src/")
import time

import jax

from src.core.kernel.gp_kernel_bind import gp_sr_fitness
from src.core.kernel.utils import *


@jax.jit
def gp_sr_fitness_(prefixGPs, data_points, targets):
    return gp_sr_fitness(prefixGPs, data_points, targets, use_MSE=True)


########
# Test #
########

gp_maxlen = 16
node_type = [
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.UFUNC,
    NodeType.CONST,
]
subtree_size = [16, 5, 3, 1, 1, 1, 11, 3, 1, 1, 7, 3, 1, 1, 2, 1]
subtree_size = [16, 5, 3, 1, 1, 1, 11, 3, 1, 1, 7, 3, 1, 1, 2, 1]
output_index = [1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1]
prefixGP = [
    Function.ADD,
    Function.MUL,
    Function.MUL,
    0,
    0,
    0,
    Function.ADD,
    Function.MUL,
    1,
    1,
    Function.ADD,
    Function.MUL,
    0,
    1,
    Function.SIN,
    4,
]
gp_len = len(prefixGP)
node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]
prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
variable = [1, 2]
target = [1, 6.2431974, -1]

N = 512
gps = jnp.tile(
    to_cuda_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
data_points = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])
targets = jnp.tile(jnp.array(target, dtype=jnp.float32), [N, 1])

a = gp_sr_fitness_(gps, data_points, targets[:, 1])
a.block_until_ready()

t = time.time()
a = gp_sr_fitness_(gps, data_points, targets[:, 1])
print(time.time() - t)
print(a[0], a[N // 2], a[N - 1])

# Multiple output test
gps = jnp.tile(
    to_cuda_node_multi_output(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
        jnp.array(output_index),
        targets.shape[1],
    ),
    [N, 1],
)

a = gp_sr_fitness_(gps, data_points, targets)
a.block_until_ready()

t = time.time()
a = gp_sr_fitness_(gps, data_points, targets)
print(time.time() - t)
print(a[0], a[N // 2], a[N - 1])


################
# Further Test #
################

gp_maxlen = 1024
node_type = [
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
]
subtree_size = [3, 1, 1]
output_index = [0, -1, -1]
prefixGP = [Function.ADD, 0, 0]
for i in range(8):
    node_type = [NodeType.BFUNC] + node_type + node_type
    subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
    output_index = [i + 1] + output_index + output_index
    prefixGP = [Function.ADD] + prefixGP + prefixGP
gp_len = len(prefixGP)
node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]
variable = [1]
target = [512, 511, 511]

N = 128
M = 1024 * 4
gps = jnp.tile(
    to_cuda_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
data_points = jnp.tile(jnp.array(variable, dtype=jnp.float32), [M, 1])
targets = jnp.tile(jnp.array(target, dtype=jnp.float32), [M, 1])

a = gp_sr_fitness_(gps, data_points, targets[:, 0])
a.block_until_ready()

t = time.time()
a = gp_sr_fitness_(gps, data_points, targets[:, 0])
print(time.time() - t)
print(a[0])

# Multiple output test
gps = jnp.tile(
    to_cuda_node_multi_output(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
        jnp.array(output_index),
        targets.shape[1],
    ),
    [N, 1],
)

a = gp_sr_fitness_(gps, data_points, targets)
a.block_until_ready()

t = time.time()
a = gp_sr_fitness_(gps, data_points, targets)
print(time.time() - t)
print(a[0], a[N // 2], a[N - 1])
