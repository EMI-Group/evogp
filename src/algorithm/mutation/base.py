import jax


class Mutation:

    def __init__(self):
        pass

    def __call__(self, key, trees, const, config) -> jax.Array:  # prefix_trees (in cuda)
        raise NotImplementedError
