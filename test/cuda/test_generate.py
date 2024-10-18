import time

from src.cuda.operations import generate
from src.cuda.utils import *


def test1():
    depth_to_leaf_prob = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    functions_prob_accumulate = jnp.array(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ],
        dtype=jnp.float32,
    )
    const_samples = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.float32)
    N = 100_0000
    max_prefix_len = 32
    num_inputs = 2
    key = jax.random.PRNGKey(42)

    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        num_inputs,
    )
    gps.block_until_ready()

    t = time.time()
    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        num_inputs,
    )
    gps.block_until_ready()
    print(time.time() - t)

    # graph_start = to_graph(gps[0])
    # graph_middle = to_graph(gps[N // 2])
    # graph_end = to_graph(gps[N - 1])

    # to_png(graph_start, "output/graph_start.png")
    # to_png(graph_middle, "output/graph_middle.png")
    # to_png(graph_end, "output/graph_end.png")

def test2():
    depth_to_leaf_prob = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    functions_prob_accumulate = jnp.array(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ],
        dtype=jnp.float32,
    )
    const_samples = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.float32)
    N = 10_0000
    max_prefix_len = 512
    num_inputs = 2
    key = jax.random.PRNGKey(42)

    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        num_inputs,
    )
    gps.block_until_ready()

    t = time.time()
    gps = generate(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        N,
        max_prefix_len,
        num_inputs,
    )
    gps.block_until_ready()
    print(time.time() - t)

def test3():
    k1 = jax.random.PRNGKey(42)
    for i in range(10000):
        k1, k2 = jax.random.split(k1, 2)
        sub_trees = generate(
            key=k2,
            leaf_prob=jnp.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0]),
            functions_prob_accumulate = jnp.array(
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.82,
                    0.84,
                    0.86,
                    0.88,
                    0.9,
                    0.92,
                    0.94,
                    0.96,
                    0.98,
                    0.99,
                    1.0,
                    1.0,
                    1.0
                ],
                dtype=jnp.float32,
            ),
            const_samples=jnp.array([-1.0, 0.0, 1.0]),
            pop_size=10,
            max_len=128,
            num_inputs=27,
            num_outputs=10,
            output_prob=0.8,
            const_prob=0.5,
        )
        print(i)
        node_values, node_types, subtree_sizes, output_indices = from_cuda_node(sub_trees)

        non_negative_one_indices = jnp.where(output_indices != -1)

        print(non_negative_one_indices)


if __name__ == "__main__":
    test1()
    test2()
