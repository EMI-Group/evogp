from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ..problem import Problem
from evogp.cuda.operations import forward
from evogp.cuda.utils import from_cuda_node


class RLEnv(Problem):

    def __init__(self, output_transform: Callable, output_length: int, max_step: int=1000):
        super().__init__()
        self.output_transform = output_transform
        self.output_length = output_length
        self.max_step = max_step

    def evaluate(self, randkey, trees):
        pop_size = trees.shape[0]
        reset_keys = jax.random.split(randkey, pop_size)
        observations, env_states = jax.vmap(self.reset)(reset_keys)

        done = jnp.zeros(pop_size, dtype=jnp.bool_)
        fitnesses = jnp.zeros(pop_size)

        def cond(carry):
            _, _, _, d, _, sc = carry
            return (~jnp.all(d)) & (sc < self.max_step) 

        def body(carry):
            obs, e_s, rng, d, f, sc = carry  # sc -> step_cnt
            rng, k1 = jax.random.split(rng)
            vmap_keys = jax.random.split(rng, pop_size)
            actions = forward(trees, obs, result_length=self.output_length)
            actions = jax.vmap(self.output_transform)(actions)
            # actions = jax.random.uniform(k1, (pop_size, self.output_length), obs.dtype, minval=-1.0, maxval=1.0)
            obs, e_s, reward, current_done, info = jax.vmap(self.step)(vmap_keys, e_s, actions)

            f += reward * jnp.logical_not(d) 
            d = jnp.logical_or(d, current_done)
            sc += 1
            return obs, e_s, rng, d, f, sc

        _, _, _, _, fitnesses, _ = jax.lax.while_loop(
            cond,
            body,
            (observations, env_states, randkey, done, fitnesses, 0)
        )

        return fitnesses

    @partial(jax.jit, static_argnums=(0,))
    def step(self, randkey, env_state, action):
        return self.env_step(randkey, env_state, action)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, randkey):
        return self.env_reset(randkey)

    def env_step(self, randkey, env_state, action):
        raise NotImplementedError

    def env_reset(self, randkey):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, randkey, prefix_trees):
        raise NotImplementedError
