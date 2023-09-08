import jax.numpy as jnp
import sys

sys.path.append("./src/")
from gp_kernel_bind import gp_mutation


########
# Test #
########

import jax
from kernel.gpdefs import *

prefixGP_maxlen = 512
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

N = 1_000_00
node_infos = jnp.tile(jnp.asarray(node_info, dtype=jnp.uint16), [N, 1, 1])
prefixGPs = jnp.tile(jnp.asarray(prefixGP, dtype=jnp.float32), [N, 1])
node_indices = jnp.full(shape=(N,), fill_value=11, dtype=jnp.int32)

prefixGP_maxlen = 16
node_type = [
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.VAR,
]
subtree_size = [5, 3, 1, 1, 1]
prefixGP = [
    Function.MUL,
    Function.MUL,
    0,
    0,
    0,
]
prefixGP_len = len(node_type)
node_info = [[node_type[i], subtree_size[i]] if i < prefixGP_len else [0, 0] for i in range(prefixGP_maxlen)]
prefixGP = [prefixGP[i] if i < prefixGP_len else 0 for i in range(prefixGP_maxlen)]

new_subtree_info = jnp.tile(jnp.asarray(node_info, dtype=jnp.uint16), [N, 1, 1])
new_subtree_node = jnp.tile(jnp.asarray(prefixGP, dtype=jnp.float32), [N, 1])

@jax.jit
def gp_mutation_(node_infos, prefixGPs, node_indices, new_subtree_info, new_subtree_node):
    return gp_mutation(node_infos, prefixGPs, node_indices, new_subtree_info, new_subtree_node)

a, b = gp_mutation_(node_infos, prefixGPs, node_indices, new_subtree_info, new_subtree_node)
a.block_until_ready()

import time
t = time.time()
a, b = gp_mutation_(node_infos, prefixGPs, node_indices, new_subtree_info, new_subtree_node)
print(time.time() - t)
print(a[0, :19])