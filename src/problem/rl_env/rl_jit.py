from dataclasses import dataclass
from typing import Callable

from jax import vmap

from src.config import ProblemConfig

from src.core import Problem, State
from src.gp.operations import *


@dataclass(frozen=True)
class RLEnvConfig(ProblemConfig):
    output_transform: Callable = lambda x: x


class RLEnv(Problem):
    jitable = True

    def __init__(self, config: RLEnvConfig = RLEnvConfig()):
        super().__init__(config)
        self.config = config

    def evaluate(self, randkey, trees):
        pop_size = trees.shape[0]

        reset_keys = jax.random.split(randkey, pop_size)
        observations, env_states = vmap(self.reset)(reset_keys)

        done = jnp.zeros(pop_size, dtype=jnp.bool_)
        fitnesses = jnp.zeros(pop_size)

        # def cond(carry):
        #     _, _, _, d, _ = carry
        #     return ~jnp.all(d)
        #
        # def body(carry):
        #     obs, env_state, rng, d, fitnesses = carry
        #     rng, _ = jax.random.split(rng)
        #     vmap_keys = jax.random.split(rng, pop_size)
        #     actions = forward(trees, obs, result_length=self.output_length)
        #     obs, env_state, re, current_done, _ = vmap(self.step)(vmap_keys, env_state, actions)
        #
        #     fitnesses += reward * jnp.logical_not(done)

        while not jnp.all(done):
            randkey, _ = jax.random.split(randkey)
            vmap_keys = jax.random.split(randkey, pop_size)

            actions = forward(trees, observations, result_length=self.output_length)
            observations, env_states, reward, current_done, _ = vmap(self.step)(vmap_keys, env_states, actions)

            fitnesses += reward * jnp.logical_not(done)
            done = jnp.logical_or(done, current_done)

        return -fitnesses

        # rng_reset, rng_episode = jax.random.split(randkey)
        # init_obs, init_env_state = self.reset(rng_reset)

        # def cond_func(carry):
        #     _, _, _, done, _ = carry
        #     return ~done
        #
        # def body_func(carry):
        #     obs, env_state, rng, _, tr = carry  # total reward
        #     net_out = act_func(state, obs, params)
        #     action = self.config.output_transform(net_out)
        #     next_obs, next_env_state, reward, done, _ = self.step(rng, env_state, action)
        #     next_rng, _ = jax.random.split(rng)
        #     return next_obs, next_env_state, next_rng, done, tr + reward
        #
        # _, _, _, _, total_reward  = jax.lax.while_loop(
        #     cond_func,
        #     body_func,
        #     (init_obs, init_env_state, rng_episode, False, 0.0)
        # )

        # return total_reward

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

    @property
    def output_length(self):
        if self.output_shape == ():
            return 1
        elif len(self.output_shape) > 0:
            return self.output_shape[-1]

    def show(self, randkey, state: State, act_func: Callable, params, *args, **kwargs):
        raise NotImplementedError
