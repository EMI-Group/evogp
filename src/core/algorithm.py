from functools import partial

import jax

from .state import State
from .tree import Tree


class Algorithm:

    def setup(self, randkey, state: State = State()):
        """initialize the state of the algorithm"""

        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, state: State):
        """require the population to be evaluated"""

        return self.ask_algorithm(state)

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, state: State, fitness):
        """update the state of the algorithm"""

        return self.tell_algorithm(state, fitness)

    @partial(jax.jit, static_argnums=(0,))
    def act(self, state: State, inputs, tree: Tree):
        return self.forward(state, inputs, tree)

    def forward(self, state: State, inputs, tree: Tree):
        raise NotImplementedError

    def ask_algorithm(self, state: State):
        """ask the specific algorithm for a new population"""
        raise NotImplementedError

    def tell_algorithm(self, state: State, fitness):
        """tell the specific algorithm the fitness of the population"""

        raise NotImplementedError
