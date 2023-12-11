"""GP Tree representation in JAX"""

from __future__ import annotations

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Tree:
    def __init__(self, node_types, node_vals, subtree_size, output_indices):
        self.node_types = node_types
        self.node_vals = node_vals
        self.subtree_size = subtree_size
        self.output_indices = output_indices

    def tree_flatten(self):
        children = self.node_types, self.node_vals, self.subtree_size, self.output_indices
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __getitem__(self, idx):
        return self.__class__(
            self.node_types[idx],
            self.node_vals[idx],
            self.subtree_size[idx],
            self.output_indices[idx],
        )

    def set(self, idx, val: Tree):
        return self.__class__(
            self.node_types.at[idx].set(val.node_types),
            self.node_vals.at[idx].set(val.node_vals),
            self.subtree_size.at[idx].set(val.subtree_size),
            self.output_indices.at[idx].set(val.output_indices),
        )