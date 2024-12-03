import sys
sys.path.append("/home/skb/TensorGP/")
import time

from evogp.cuda.operations import sr_fitness
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
    target = [1, 6.2431974, -1]

    N = 1000
    M = 4096 * 4096
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

    a = sr_fitness(gps, data_points, targets[:, 1])
    a.block_until_ready()

    t = time.time()
    a = sr_fitness(gps, data_points, targets[:, 1])
    a.block_until_ready()
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

    a = sr_fitness(gps, data_points, targets)
    a.block_until_ready()

    t = time.time()
    a = sr_fitness(gps, data_points, targets)
    a.block_until_ready()
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
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]
    variable = [1]
    target = [512, 511, 511]

    N = 1000
    M = 1024 * 1024
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
    
    gps.block_until_ready()
    data_points.block_until_ready()
    targets.block_until_ready()
    a = sr_fitness(gps, data_points, targets[:, 0])
    a.block_until_ready()

    t = time.time()
    a = sr_fitness(gps, data_points, targets[:, 0])
    a.block_until_ready()
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

    a = sr_fitness(gps, data_points, targets)
    a.block_until_ready()

    t = time.time()
    a = sr_fitness(gps, data_points, targets)
    a.block_until_ready()
    print(time.time() - t)
    print(a[0], a[N // 2], a[N - 1])

if __name__ == "__main__":
    test()