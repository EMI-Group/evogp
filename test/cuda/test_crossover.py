import time

from evogp.utils.enum import Func
from evogp.cuda.operations import crossover
from evogp.cuda.utils import *


def test1():
    gp_maxlen = 32
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
    subtree_size = [16, 5, 3, 1, 1, 1, 11, 3, 1, 1, 7, 3, 1, 1, 2, 1]
    output_index = [1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1]
    gp_len = len(prefixGP)
    node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]

    N = 100_0000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    nodes_pos = jnp.tile(
        jnp.array([1, 11], dtype=jnp.int16),
        [N, 1]
    )
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, 2)
    recipient_indices = jnp.argsort(jax.random.uniform(key1, shape=(N,)))
    donor_indices = jnp.argsort(jax.random.uniform(key2, shape=(N,)))

    a = crossover(gps, recipient_indices, donor_indices, nodes_pos)
    a.block_until_ready()
    print("ready")

    t = time.time()
    a = crossover(gps, recipient_indices, donor_indices, nodes_pos)
    a.block_until_ready()
    print(time.time() - t)
    
    # graph_start = to_graph(a[0])
    # graph_middle = to_graph(a[N // 2])
    # graph_end = to_graph(a[N - 1])

    # to_png(graph_start, "output/graph_start.png")
    # to_png(graph_middle, "output/graph_middle.png")
    # to_png(graph_end, "output/graph_end.png")


def test2():
    gp_maxlen = 512
    node_type = [
        NType.BFUNC,
        NType.VAR,
        NType.VAR,
    ]
    prefixGP = [Func.ADD, 0, 0]
    subtree_size = [3, 1, 1]
    output_index = [0, -1, -1]
    for i in range(7):
        node_type = [NType.BFUNC] + node_type + node_type
        prefixGP = [Func.ADD] + prefixGP + prefixGP
        subtree_size = [2 * len(subtree_size) + 1] + subtree_size + subtree_size
        output_index = [i + 1] + output_index + output_index
    gp_len = len(prefixGP)
    node_type = [node_type[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    prefixGP = [prefixGP[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    subtree_size = [subtree_size[i] if i < gp_len else 0 for i in range(gp_maxlen)]
    output_index = [output_index[i] if i < gp_len else -1 for i in range(gp_maxlen)]

    N = 10_0000
    gps = jnp.tile(
        to_cuda_node(
            jnp.array(prefixGP, dtype=jnp.float32),
            jnp.array(node_type),
            jnp.array(subtree_size),
        ),
        [N, 1],
    )
    nodes_pos = jnp.tile(
        jnp.array([5, 7], dtype=jnp.int16),
        [N, 1]
    )
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, 2)
    recipient_indices = jnp.argsort(jax.random.uniform(key1, shape=(N,)))
    donor_indices = jnp.argsort(jax.random.uniform(key2, shape=(N,)))

    a = crossover(gps, recipient_indices, donor_indices, nodes_pos)
    a.block_until_ready()
    print("ready")

    t = time.time()
    a = crossover(gps, recipient_indices, donor_indices, nodes_pos)
    a.block_until_ready()
    print(time.time() - t)
    
    # graph_start = to_graph(a[0])
    # graph_middle = to_graph(a[N // 2])
    # graph_end = to_graph(a[N - 1])

    # to_png(graph_start, "output/graph_start.png")
    # to_png(graph_middle, "output/graph_middle.png")
    # to_png(graph_end, "output/graph_end.png")


if __name__ == "__main__":
    test1()
    test2()