import jax.numpy as jnp
import jax
from .func_fit import FuncFit, FuncFitConfig


class Polynomial_3d(FuncFit):

    def __init__(self, config: FuncFitConfig = FuncFitConfig()):
        self.config = config
        super().__init__(config)

    @property
    def inputs(self):
        x1 = jnp.arange(-1, 1, 0.01).reshape(200, 1)
        x2 = jnp.arange(-1, 1, 0.01).reshape(200, 1)
        return jnp.concatenate([x1, x2], axis=1)

    @property
    def targets(self):
        x1 = jnp.arange(-1, 1, 0.01).reshape(200, 1)
        x2 = jnp.arange(-1, 1, 0.01).reshape(200, 1)
        x = jnp.concatenate([x1, x2], axis=1)
        return target_func(x)
    
    @property
    def input_shape(self):
        return 200, 2

    @property
    def output_shape(self):
        return 200, 1

@jax.vmap
def target_func(x):
    res = 1 / (1 + jnp.power(x[0], -4)) + 1 / (1 + jnp.power(x[1], -4))
    return res