"""
pipeline for jitable env like func_fit, gymnax
"""

from functools import partial
from typing import Type

import jax
import time
import numpy as np

from src.config import Config
from src.core import State, Algorithm, Problem


class Pipeline:

    def __init__(self, config: Config, algorithm: Algorithm, problem_type: Type[Problem]):

        assert problem_type.jitable, "problem must be jitable"

        self.config = config
        self.algorithm = algorithm
        self.problem = problem_type(config.problem)

        self.act_func = self.algorithm.act

        for _ in range(len(self.problem.input_shape) - 1):
            self.act_func = jax.vmap(self.act_func, in_axes=(None, 0, None))

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.generation_timestamp = None

    def setup(self):
        key = jax.random.PRNGKey(self.config.basic.seed)
        algorithm_key, evaluate_key = jax.random.split(key, 2)
        state = State()
        state = self.algorithm.setup(algorithm_key, state)
        return state.update(
            evaluate_key=evaluate_key
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state):

        key, sub_key = jax.random.split(state.evaluate_key)
        keys = jax.random.split(key, self.config.gp.pop_size)

        pop = self.algorithm.ask(state)
        fitnesses = jax.vmap(self.problem.evaluate, in_axes=(0, None, None, 0))(keys, state, self.act_func,
                                                                                pop)

        state = self.algorithm.tell(state, fitnesses)

        return state.update(evaluate_key=sub_key), fitnesses

    def auto_run(self, ini_state):
        state = ini_state
        for _ in range(self.config.basic.generation_limit):

            self.generation_timestamp = time.time()

            previous_pop = self.algorithm.ask(state)

            state, fitnesses = self.step(state)

            fitnesses = jax.device_get(fitnesses)

            self.analysis(state, previous_pop, fitnesses)

            if max(fitnesses) >= self.config.basic.fitness_target:
                print("Fitness limit reached!")
                return state, self.best_genome

        print("Generation limit reached!")
        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):

        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = pop[max_idx]

        print(f"Generation: {state.generation}",
              f"fitness: {max_f:.6f}, {min_f:.6f}, {mean_f:.6f}, {std_f:.6f}, Cost time: {cost_time * 1000:.6f}ms")

    def show(self, state, tree):
        self.problem.show(state.evaluate_key, state, self.act_func, tree)

    def pre_compile(self, state):
        tic = time.time()
        print("start compile")
        self.step.lower(self, state).compile()
        print(f"compile finished, cost time: {time.time() - tic}s")
