import jax
from .base import Const


class NormalConst(Const):

    def __init__(self, mean, std, cnt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std
        self.cnt = cnt

    def __call__(self, key, *args, **kwargs) -> jax.Array:
        return jax.random.normal(key, (self.cnt,)) * self.std + self.mean
