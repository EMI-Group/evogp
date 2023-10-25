from src.gp import NormalGP
from src.config import GPConfig
from src.gp.enum import tree2str, to_string


def test_new_tree():
    config = GPConfig(
        new_tree_depth=5,
        max_subtree_size=32,
    )
    key = jax.random.PRNGKey(43)
    state = NormalGP(config).setup(key)
    print(state)
    node_type, node_val = new_tree(key, state)
    print(node_type, node_val)
    print(to_string(node_type, node_val))
    sub_size = get_sub_size(node_type, node_val)
    print(sub_size)
    tree = Tree(node_type, node_val, sub_size)
    print(cal(tree, jnp.array([1, 100])))


def test_batch_operation():
    config = GPConfig(
        new_tree_depth=4,
        max_subtree_size=32,
    )
    key = jax.random.PRNGKey(43)
    state = NormalGP(config).setup(key)

    batch_size = 10
    batch_new_tree = jax.jit(jax.vmap(new_tree, in_axes=(0, None)))
    trees = batch_new_tree(jax.random.split(key, batch_size), state)
    print(tree2str(trees[0]))
    print(tree2str(trees[2]))

    new = jax.jit(crossover)(trees[0], trees[2], key)
    print(tree2str(new))

    mutated = jax.jit(mutation)(new, key, state)
    print(tree2str(mutated))


if __name__ == '__main__':
    # test_new_tree()
    test_batch_operation()
