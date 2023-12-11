import jax


class Crossover:

    def __init__(self):
        pass

    def __call__(self, key, trees, left, right, config) -> jax.Array:  # prefix_trees (in cuda)
        raise NotImplementedError
