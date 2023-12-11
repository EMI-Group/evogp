import jax.numpy as jnp
import jax
from .func_fit import FuncFit, FuncFitConfig


class Polynomial(FuncFit):

    def __init__(self, config: FuncFitConfig = FuncFitConfig()):
        self.config = config
        super().__init__(config)

    @property
    def inputs(self):
        return jnp.arange(-1, 1, 0.01).reshape(200, 1)

    @property
    def targets(self):
        return target_func(jnp.arange(-1, 1, 0.01).reshape(200, 1))
    
    @property
    def input_shape(self):
        return 200, 1

    @property
    def output_shape(self):
        return 200, 1

@jax.vmap
def target_func(x):
    sum = 0
    for i in range(4):
        sum += jnp.power(x, i)
    return sum