"""
pipeline for jitable env like func_fit, gymnax
"""

from typing import Type

import jax
from jax import numpy as jnp, vmap, jit

from src.config import *
from src.core import State
from src.core.problem import Problem

from src.gp.enum import FUNCS
from .operations import *

from src.core.kernel.utils import *


class Pipeline:
    def __init__(self, conf: Config, problem_type: Type[Problem]):
        assert problem_type.jitable, "problem must be jitable"

        self.config = conf
        self.problem = problem_type(conf.problem)

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.func_cum = func_cum_prob(self.config.gp.func)

    def setup(self):
        key = jax.random.PRNGKey(self.config.basic.seed)

        return State(
            randkey=key,
            trees=self.new_trees(key, self.config.gp.subtree.subtree_size),
            generation=0,
        )

    @partial(jit, static_argnums=(0,))
    def step(self, state: State):
        fitness = self.evaluate(state)

        jax.debug.print("{}", jnp.max(fitness))

        k1, k2, k3, k4, new_key = jax.random.split(state.randkey, 5)

        pop_size, pop_idx = fitness.shape[0], jnp.arange(0, fitness.shape[0])

        sorted_idx = jnp.argsort(fitness)[::-1]
        ranks = jnp.argsort(sorted_idx)

        selected_num = int(self.config.gp.parent_rate * pop_size)
        selected_p = jnp.where(ranks < selected_num, 1, 0)
        selected_p = selected_p / jnp.sum(selected_p)

        left, right = jax.random.choice(k1, pop_idx, p=selected_p, shape=(2, pop_size), replace=True)

        children = self.crossover_trees(
            k1,
            state.trees,
            left,
            right,
        )

        # children = state.trees

        mutated_children = self.mutate_trees(
            k2,
            children,
        )

        mutate_p = jax.random.uniform(k4, (pop_size,))
        selected_mask = (mutate_p < self.config.gp.mutate_prob)[:, None]

        children = jnp.where(
            selected_mask,
            mutated_children,
            children,
        )

        return state.update(
            trees=children,
            randkey=new_key,
            generation=state.generation + 1,
        )

    def evaluate(self, state: State):
        res = sr_fitness(
            state.trees,
            data_points=self.problem.inputs.astype(jnp.float32),
            targets=self.problem.targets.astype(jnp.float32),
        )

        return res

    def mutate_trees(self, key, trees):
        tree_sizes = vmap(tree_size)(trees)

        k1, k2 = jax.random.split(key)

        sub_trees = self.new_trees(k1, self.config.gp.subtree.subtree_size)

        indices = vmap(random_idx)(jax.random.split(k2, num=trees.shape[0]), tree_sizes)

        return mutation(
            trees,
            indices,
            sub_trees,
        )

    def crossover_trees(self, key, trees, l_indices, r_indices):
        tree_sizes = vmap(tree_size)(trees)

        k1, k2 = jax.random.split(key)

        l_pos = vmap(random_idx)(jax.random.split(k1, num=trees.shape[0]), tree_sizes[l_indices])
        r_pos = vmap(random_idx)(jax.random.split(k2, num=trees.shape[0]), tree_sizes[r_indices])

        nodes = jnp.stack([l_pos, r_pos], axis=1).astype(jnp.int16)

        return crossover(
            trees,
            l_indices,
            r_indices,
            nodes,
        )

    def new_trees(self, key, size):

        k1, k2 = jax.random.split(key)

        if self.config.gp.const.type == "discrete":
            consts = jnp.array(self.config.gp.const.pool)
        else:
            consts = (
                    jax.random.normal(k1, (self.config.gp.const.points_cnt,))
                    * self.config.gp.const.std
                    + self.config.gp.const.mean
            )

        # seed = int(jax.random.randint(key, (), 0, int(2 ** 10 - 1)))
        seed = 0

        trees = generate(
            seed=seed,
            pop_size=self.config.gp.pop_size,
            max_len=size,
            num_inputs=self.config.gp.num_inputs,
            num_outputs=self.config.gp.num_outputs,
            leaf_prob=jnp.array(self.config.gp.subtree.leaf_prob),
            functions_prob_accumulate=self.func_cum,
            const_samples=consts,
            output_prob=self.config.gp.output_prob,
            const_prob=self.config.gp.const.const_prob,
        )

        return trees

    def pop_forward(self, trees, inputs):
        return forward(
            trees,
            inputs,
            result_length=self.config.gp.num_outputs,
        )


def func_cum_prob(conf: FuncConfig):
    probs = np.zeros(len(FUNCS))
    for func, prob in zip(conf.pool, conf.prob):
        probs[func] = prob

    return probs.cumsum()


def random_idx(key, size):
    return jax.random.randint(key, (), 0, size - 1)
