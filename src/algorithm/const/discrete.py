import jax
from .base import Const


class DiscreteConst(Const):

    def __init__(self, points: jax.Array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = jax.numpy.array(points, dtype=jax.numpy.float32)

    def __call__(self, key, *args, **kwargs) -> jax.Array:
        return self.points
