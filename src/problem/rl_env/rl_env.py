from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from ..problem import Problem
from src.cuda.operations import forward


class RLEnv(Problem):

    def __init__(self, output_transform: Callable, output_length: int):
        super().__init__()
        self.output_transform = output_transform
        self.output_length = output_length

    def evaluate(self, randkey, trees):
        pop_size = trees.shape[0]

        reset_keys = jax.random.split(randkey, pop_size)
        observations, env_states = jax.vmap(self.reset)(reset_keys)

        done = jnp.zeros(pop_size, dtype=jnp.bool_)
        fitnesses = jnp.zeros(pop_size)

        # carry: observations, env_states, rng, done, fitnesses
        def cond(carry):
            _, _, _, d, _ = carry
            return ~jnp.all(d)

        def body(carry):
            obs, e_s, rng, d, f = carry
            rng, _ = jax.random.split(rng)
            vmap_keys = jax.random.split(rng, pop_size)
            actions = forward(trees, obs, result_length=self.output_length)
            actions = jax.vmap(self.output_transform)(actions)
            obs, e_s, reward, current_done, info = jax.vmap(self.step)(vmap_keys, e_s, actions)

            f += reward * jnp.logical_not(d)
            d = jnp.logical_or(d, current_done)

            return obs, e_s, rng, d, f

        _, _, _, _, fitnesses = jax.lax.while_loop(
            cond,
            body,
            (observations, env_states, randkey, done, fitnesses)
        )

        return fitnesses
        # while not jnp.all(done):
        #     randkey, _ = jax.random.split(randkey)
        #     vmap_keys = jax.random.split(randkey, pop_size)
        #
        #     actions = forward(trees, observations, result_length=self.output_length)
        #     observations, env_states, reward, current_done, _ = jax.vmap(self.step)(vmap_keys, env_states, actions)
        #
        #     fitnesses += reward * jnp.logical_not(done)
        #     done = jnp.logical_or(done, current_done)

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
