import sys

import jax.numpy as jnp

sys.path.append("./src/")
import time

import jax

from gp_kernel_bind import gp_eval
from kernel.gpdefs import *


@jax.jit
def gp_eval_(prefixGPs, variables):
    return gp_eval(prefixGPs, variables)


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
prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
variable = [1, 2]

N = 200000
gps = jnp.tile(
    to_jax_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
variables = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])

a = gp_eval_(gps, variables)
a.block_until_ready()

t = time.time()
a = gp_eval_(gps, variables)
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
prefixGP = [Function.ADD, 0, 0]
for _ in range(8):
    node_type = [NodeType.BFUNC] + node_type + node_type
    subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
    prefixGP = [Function.ADD] + prefixGP + prefixGP
gp_len = len(prefixGP)
node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
variable = [1]

N = 200000
gps = jnp.tile(
    to_jax_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
variables = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])

a = gp_eval_(gps, variables)

t = time.time()
a = gp_eval_(gps, variables)
print(time.time() - t)
print(a[0])
