import jax


class Const:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, key, *args, **kwargs) -> jax.Array:
        raise NotImplementedError
