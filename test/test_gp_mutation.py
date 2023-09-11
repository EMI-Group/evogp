import sys

import jax.numpy as jnp

sys.path.append("./src/")
import time

import jax

from gp_kernel_bind import gp_mutation
from kernel.gpdefs import *


@jax.jit
def gp_mutation_(prefixGPs, nodes, newGPs):
    return gp_mutation(prefixGPs, nodes, newGPs)


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
node = [1]

sub_maxlen = 8
sub_node_type = [NodeType.BFUNC, NodeType.VAR, NodeType.VAR]
sub_tree_size = [3, 1, 1]
sub_gp = [Function.DIV, 0, 0]
gp_len = len(sub_node_type)
sub_node_type = [sub_node_type[i] if i < gp_len else 0 for i in range(sub_maxlen)]
sub_tree_size = [sub_tree_size[i] if i < gp_len else 0 for i in range(sub_maxlen)]
sub_gp = [sub_gp[i] if i < gp_len else 0 for i in range(sub_maxlen)]

N = 200000
gps = jnp.tile(
    to_jax_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
sub_gps = jnp.tile(
    to_jax_node(
        jnp.array(sub_gp, dtype=jnp.float32),
        jnp.array(sub_node_type),
        jnp.array(sub_tree_size),
    ),
    [N, 1],
)
nodes = jnp.tile(
    jnp.array(node, dtype=jnp.uint32), N
)

a = gp_mutation_(gps, nodes, sub_gps)
a.block_until_ready()

t = time.time()
a = gp_mutation_(gps, nodes, sub_gps)
print(time.time() - t)
_, a, b = from_jax_node(a)
print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])


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
node = [7]

N = 200000
gps = jnp.tile(
    to_jax_node(
        jnp.array(prefixGP, dtype=jnp.float32),
        jnp.array(node_type),
        jnp.array(subtree_size),
    ),
    [N, 1],
)
sub_gps = jnp.tile(
    to_jax_node(
        jnp.array(sub_gp, dtype=jnp.float32),
        jnp.array(sub_node_type),
        jnp.array(sub_tree_size),
    ),
    [N, 1],
)
nodes = jnp.tile(
    jnp.array(node, dtype=jnp.uint32), N
)

a = gp_mutation_(gps, nodes, sub_gps)
a.block_until_ready()

t = time.time()
a = gp_mutation_(gps, nodes, sub_gps)
print(time.time() - t)
_, a, b = from_jax_node(a)
print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])