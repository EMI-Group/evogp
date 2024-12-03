import time

from evogp.cuda.operations import mutation
from evogp.cuda.utils import *
from evogp.utils.enum import Func


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
    node = [1]

    sub_maxlen = 8
    sub_node_type = [NType.BFUNC, NType.VAR, NType.VAR]
    sub_tree_size = [3, 1, 1]
    sub_gp = [Func.DIV, 0, 0]
    gp_len = len(sub_node_type)
    sub_node_type = [sub_node_type[i] if i < gp_len else 0 for i in range(sub_maxlen)]
    sub_tree_size = [sub_tree_size[i] if i < gp_len else 0 for i in range(sub_maxlen)]
    sub_gp = [sub_gp[i] if i < gp_len else 0 for i in range(sub_maxlen)]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    sub_gps = jnp.tile(
        to_cuda_node(
            jnp.array(sub_gp, dtype=jnp.float32),
            jnp.array(sub_node_type),
            jnp.array(sub_tree_size),
        ),
        [N, 1],
    )

    nodes = jnp.tile(jnp.array(node, dtype=jnp.int32), N)

    a = mutation(gps, nodes, sub_gps)
    a.block_until_ready()

    t = time.time()
    a = mutation(gps, nodes, sub_gps)
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
    node = [7]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    sub_gps = jnp.tile(
        to_cuda_node(
            jnp.array(sub_gp, dtype=jnp.float32),
            jnp.array(sub_node_type),
            jnp.array(sub_tree_size),
        ),
        [N, 1],
    )
    nodes = jnp.tile(jnp.array(node, dtype=jnp.int32), N)

    a = mutation(gps, nodes, sub_gps)
    a.block_until_ready()

    t = time.time()
    a = mutation(gps, nodes, sub_gps)
    print(time.time() - t)
    _, a, b, _ = from_cuda_node(a)
    print(a[0, : b[0, 0]], a[N // 2, : b[N // 2, 0]], a[N - 1, : b[N - 1, 0]])
    print(b[0, : b[0, 0]], b[N // 2, : b[N // 2, 0]], b[N - 1, : b[N - 1, 0]])

if __name__ == "__main__":
    test()