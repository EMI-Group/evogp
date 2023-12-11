import time

from src.utils.enum import Func
from src.cuda.operations import crossover
from src.cuda.utils import *


def test():
    gp_maxlen = 16
    node_type = [
        NType.BFUNC,
        NType.BFUNC,
        NType.BFUNC,
        NType.VAR,
        NType.VAR,
        NType.VAR,
        NType.BFUNC,
        NType.BFUNC,
        NType.VAR,
        NType.VAR,
        NType.BFUNC,
        NType.BFUNC,
        NType.VAR,
        NType.VAR,
        NType.UFUNC,
        NType.CONST,
    ]
    subtree_size = [16, 5, 3, 1, 1, 1, 11, 3, 1, 1, 7, 3, 1, 1, 2, 1]
    prefixGP = [
        Func.ADD,
        Func.MUL,
        Func.MUL,
        0,
        0,
        0,
        Func.ADD,
        Func.MUL,
        1,
        1,
        Func.ADD,
        Func.MUL,
        0,
        1,
        Func.SIN,
        4,
    ]
    gp_len = len(prefixGP)
    node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    left_right_nodes = [1, 11]  # [11, 1]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    left_right_node_indices = jnp.tile(
        jnp.array(left_right_nodes, dtype=jnp.int16), [N, 1]
    )
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, 2)
    lefts = jax.random.uniform(key1, shape=(N,))
    rights = jax.random.uniform(key2, shape=(N,))
    left_perms = jnp.argsort(lefts)
    right_perms = jnp.argsort(rights)

    a = crossover(gps, left_perms, right_perms, left_right_node_indices)
    a.block_until_ready()

    t = time.time()
    a = crossover(gps, left_perms, right_perms, left_right_node_indices)
    print(time.time() - t)
    _, a, b, _ = from_cuda_node(a)
    print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
    print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])

    ################
    # Further Test #
    ################

    gp_maxlen = 1024
    node_type = [
        NType.BFUNC,
        NType.VAR,
        NType.VAR,
    ]
    subtree_size = [3, 1, 1]
    prefixGP = [Func.ADD, 0, 0]
    for _ in range(8):
        node_type = [NType.BFUNC] + node_type + node_type
        subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
        prefixGP = [Func.ADD] + prefixGP + prefixGP
    gp_len = len(prefixGP)
    node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    left_right_nodes = [5, 7]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    left_right_node_indices = jnp.tile(
        jnp.array(left_right_nodes, dtype=jnp.int16), [N, 1]
    )
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, 2)
    lefts = jax.random.uniform(key1, shape=(N,))
    rights = jax.random.uniform(key2, shape=(N,))
    left_perms = jnp.argsort(lefts)
    right_perms = jnp.argsort(rights)

    a = crossover(gps, left_perms, right_perms, left_right_node_indices)
    a.block_until_ready()

    t = time.time()
    a = crossover(gps, left_perms, right_perms, left_right_node_indices)
    print(time.time() - t)
    _, a, b, _ = from_cuda_node(a)
    print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
    print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])
