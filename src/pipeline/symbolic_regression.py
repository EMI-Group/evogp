import jax
import jax.numpy as jnp
from src.cuda.operations import sr_fitness
from src.utils import State

inputs = jnp.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=jnp.float32)

targets = jnp.array([
    [0],
    [1],
    [1],
    [0]
], dtype=jnp.float32)


class SymbolicRegression:

    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem

    def setup(self, randkey, state=State()):
        k1, k2 = jax.random.split(randkey, 2)
        alg_state = self.algorithm.setup(k1, state)
        # prob_state = self.problem.setup(k2, state)
        return state.update(
            alg_state=alg_state,
            # pro_state=prob_state,
            generation=0,
        )

    def step(self, state):
        fitness = -sr_fitness(
            self.algorithm.ask(state.alg_state),
            inputs,
            targets
        )
        print(max(fitness))
        alg_state = self.algorithm.tell(state.alg_state, fitness)
        return state.update(
            alg_state=alg_state,
            generation=state.generation + 1,
        )
