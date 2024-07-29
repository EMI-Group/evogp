import jax
from src.utils import State
from jax import pmap
from rich.console import Console
import jax.numpy as jnp

class General:

    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem

    def setup(self, randkey, state=State()):
        k1, k2 = jax.random.split(randkey, 2)
        alg_state = self.algorithm.setup(k1, state)
        return state.update(
            alg_state=alg_state,
            generation=0,
            randkey=k2,
        )

    # def step(self, state):
    #     console = Console()
    #     trees = self.algorithm.ask(state.alg_state)
    #     fitness = self.problem.evaluate(state.randkey, trees)
    #     alg_state = self.algorithm.tell(state.alg_state, fitness)
    #     return state.update(
    #         alg_state=alg_state,
    #         generation=state.generation + 1,
    #     ), fitness
    
    def step(self, state):
        # console = Console()
        trees = self.algorithm.ask(state.alg_state)
        fitness = self.problem.evaluate(state.randkey, trees)
        alg_state = self.algorithm.tell(state.alg_state, fitness)
        return state.update(
            alg_state=alg_state,
            generation=state.generation + 1,
        ), fitness