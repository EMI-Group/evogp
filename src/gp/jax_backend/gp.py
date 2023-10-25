from src.core import Algorithm, State
from src.core.utils import expand

from .operations import *


class NormalGP(Algorithm):
    def __init__(self, config):
        self.config = config.gp

    def setup(self, randkey, state: State = State()):
        state = state.update(
            randkey=randkey,
            size=self.config.max_size,
            subtree_size=self.config.max_subtree_size,
            num_inputs=self.config.num_inputs,
            const_pool=jnp.array(self.config.const_pool),
            func_pool=jnp.array(self.config.func_pool),
            func_prob=self.config.func_prob,
            var_prob=self.config.var_prob,
            const_prob=self.config.const_prob,
            new_tree_depth=self.config.new_tree_depth,
            generation=0,
        )

        state = state.update(
            empty_tree=jnp.zeros((state.size,)),
            empty_subtree=jnp.zeros((state.subtree_size,)),
        )
        return self.initialize(state)

    def forward(self, state: State, inputs, tree: Tree):
        return cal(tree, inputs)

    def ask_algorithm(self, state: State):
        """ask the specific algorithm for a new population"""

        return state.trees

    def tell_algorithm(self, state: State, fitness):
        """tell the specific algorithm the fitness of the population"""
        fitness = jnp.where(jnp.isnan(fitness), -float('inf'), fitness)
        k1, k2, k3, k4, new_key = jax.random.split(state.randkey, 5)

        children = state.trees

        pop_size, pop_idx = fitness.shape[0], jnp.arange(0, fitness.shape[0])
        mutation_keys = jax.random.split(k3, pop_size)

        mutated_children = jax.vmap(mutation, in_axes=(0, 0, None))(
            children,
            mutation_keys,
            state,
        )

        mutate_p = jax.random.uniform(k4, (pop_size,))

        selected_mask = (mutate_p < self.config.mutate_prob)[:, None]

        children = Tree(
            jnp.where(selected_mask, mutated_children.node_types, children.node_types),
            jnp.where(selected_mask, mutated_children.node_vals, children.node_vals),
            jnp.where(selected_mask, mutated_children.subtree_size, children.subtree_size),
        )

        return state.update(
            trees=children,
            randkey=new_key,
            generation=state.generation + 1,
        )

    def initialize(self, state: State):
        key, new_key = jax.random.split(state.randkey)
        trees = jax.vmap(new_tree, in_axes=(0, None, None))(
            jax.random.split(key, self.config.pop_size),
            state,
            True
        )

        def expand_tree(t: Tree):
            new_types = expand(t.node_types, state.size, NType.NAN)
            new_vals = expand(t.node_vals, state.size, 0)
            new_subsize = expand(t.subtree_size, state.size, 0)
            return Tree(new_types, new_vals, new_subsize)

        trees = jax.vmap(expand_tree)(trees)

        return state.update(
            trees=trees,
            randkey=new_key,
        )
