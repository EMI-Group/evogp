import time

from src.cuda.operations import forward
from src.cuda.utils import *
from src.utils.enum import Func


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
    output_index = [1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1]
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
    output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    variable = [1, 2]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    variables = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])

    a = forward(gps, variables)
    a.block_until_ready()

    t = time.time()
    a = forward(gps, variables)
    print(time.time() - t)
    print(a[0], a[N // 2], a[N - 1])

    #####################
    # Multi-output Test #
    #####################

    N_outs = 3
    gps = jnp.tile(
        to_cuda_node_multi_output(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
            jnp.array(output_index),
            N_outs,
        ),
        [N, 1],
    )

    a = forward(gps, variables, result_length=N_outs)
    a.block_until_ready()

    t = time.time()
    a = forward(gps, variables, result_length=N_outs)
    print(time.time() - t)
    print(a[0], a[N // 2], a[N - 1])

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
    output_index = [0, -1, -1]
    prefixGP = [Func.ADD, 0, 0]
    for i in range(8):
        node_type = [NType.BFUNC] + node_type + node_type
        subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
        output_index = [i + 1] + output_index + output_index
        prefixGP = [Func.ADD] + prefixGP + prefixGP
    gp_len = len(prefixGP)
    node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    variable = [1]

    N = 200000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    variables = jnp.tile(jnp.array(variable, dtype=jnp.float32), [N, 1])

    a = (gps, variables)

    t = time.time()
    a = forward(gps, variables)
    print(time.time() - t)
    print(a[0])

    #####################
    # Multi-output Test #
    #####################

    N_outs = max(output_index)
    gps = jnp.tile(
        to_cuda_node_multi_output(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
            jnp.array(output_index),
            N_outs,
        ),
        [N, 1],
    )

    a = forward(gps, variables, result_length=N_outs)
    a.block_until_ready()

    t = time.time()
    a = forward(gps, variables, result_length=N_outs)
    print(time.time() - t)
    print(a[0], a[N // 2], a[N - 1])
