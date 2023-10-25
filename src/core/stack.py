from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp


@register_pytree_node_class
class Stack:

    @staticmethod
    def new(max_shape, dtype=jnp.float32):
        return Stack(jnp.full(max_shape, jnp.nan, dtype=dtype), 0)

    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def push(self, value):
        new_data = self.data.at[self.idx].set(value)
        new_idx = self.idx + 1
        return Stack(new_data, new_idx)

    def not_push(self, value):  # for brunching
        return self

    def pop(self):
        val = self.data[self.idx - 1]
        new_data = self.data.at[self.idx - 1].set(jnp.nan)
        new_idx = self.idx - 1
        return val, Stack(new_data, new_idx)

    def peek(self):
        return self.data[self.idx - 1]

    def size(self):
        return self.idx

    def tree_flatten(self):
        children = self.data, self.idx
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
