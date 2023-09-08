import jax.numpy as jnp
import sys

sys.path.append("./src/")
from gp_kernel_bind import gp_eval


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
variable = [1, 2]

N = 100000
node_infos = jnp.tile(jnp.asarray(node_info, dtype=jnp.uint16), [N, 1, 1])
prefixGPs = jnp.tile(jnp.asarray(prefixGP, dtype=jnp.float32), [N, 1])
variables = jnp.tile(jnp.asarray(variable, dtype=jnp.float32), [N, 1])

@jax.jit
def gp_eval_(node_infos, prefixGPs, variables):
    return gp_eval(node_infos, prefixGPs, variables)

a = gp_eval_(node_infos, prefixGPs, variables)
a.block_until_ready()

import time
t = time.time()
a = gp_eval_(node_infos, prefixGPs, variables)
print(time.time() - t)
print(a[0], a[N // 2], a[N - 1])


################
# Further Test #
################

prefixGP_maxlen = 1024
node_type = [
    NodeType.BFUNC,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
    NodeType.BFUNC,
    NodeType.VAR,
    NodeType.VAR,
]
subtree_size = [7, 3, 1, 1, 3, 1, 1]
prefixGP = [Function.ADD, Function.ADD, 0, 0, Function.ADD, 0, 0]
for _ in range(7):
    node_type = [NodeType.BFUNC] + node_type + node_type
    subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
    prefixGP = [Function.ADD] + prefixGP + prefixGP
prefixGP_len = len(prefixGP)
print(prefixGP_len)
node_info = [[node_type[i], subtree_size[i]] if i < prefixGP_len else [0, 0] for i in range(prefixGP_maxlen)]
prefixGP = [prefixGP[i] if i < prefixGP_len else 0 for i in range(prefixGP_maxlen)]
variable = [1]

N = 100000
node_infos = jnp.tile(jnp.asarray(node_info, dtype=jnp.uint16), [N, 1, 1])
prefixGPs = jnp.tile(jnp.array(prefixGP, dtype=jnp.float32), [N, 1])
variables = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])

a = gp_eval_(node_infos, prefixGPs, variables)

t = time.time()
a = gp_eval_(node_infos, prefixGPs, variables)
print(time.time() - t)
print(a[0])
