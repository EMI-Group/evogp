import jax.numpy as jnp
import sys

sys.path.append("./src/")
from gp_kernel_bind import gp_crossover


########
# Test #
########

import jax
from kernel.gpdefs import *

prefixGP_maxlen = 1024
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
    NodeType.BFUNC,
    NodeType.CONST,
    NodeType.VAR,
]
subtree_size = [17, 5, 3, 1, 1, 1, 11, 3, 1, 1, 7, 3, 1, 1, 3, 1, 1]
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
    Function.DIV,
    4,
    0,
]
prefixGP_len = len(node_type)
node_info = [[node_type[i], subtree_size[i]] if i < prefixGP_len else [0, 0] for i in range(prefixGP_maxlen)]
prefixGP = [prefixGP[i] if i < prefixGP_len else 0 for i in range(prefixGP_maxlen)]
left_right_nodes = [11, 1]

N = 100000
node_infos = jnp.tile(jnp.asarray(node_info, dtype=jnp.uint16), [N, 1, 1])
prefixGPs = jnp.tile(jnp.asarray(prefixGP, dtype=jnp.float32), [N, 1])
left_right_node_indices = jnp.tile(jnp.asarray(left_right_nodes, dtype=jnp.uint16), [N, 1])
key = jax.random.PRNGKey(1)
key1, key2 = jax.random.split(key, 2)
lefts = jax.random.uniform(key1, shape=(N,))
rights = jax.random.uniform(key2, shape=(N,))
left_perms = jnp.argsort(lefts)
right_perms = jnp.argsort(rights)

@jax.jit
def gp_crossover_(node_infos, prefixGPs, left_perms, right_perms, left_right_node_indices):
    return gp_crossover(node_infos, prefixGPs, left_perms, right_perms, left_right_node_indices)

a, b = gp_crossover_(node_infos, prefixGPs, left_perms, right_perms, left_right_node_indices)
a.block_until_ready()

import time
t = time.time()
a, b = gp_crossover_(node_infos, prefixGPs, left_perms, right_perms, left_right_node_indices)
print(time.time() - t)
print(a[0, :19])