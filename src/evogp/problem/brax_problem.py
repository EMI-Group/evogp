import torch
from typing import Callable
from evogp.tree import Forest

from . import BaseProblem


def to_jax_array(x: torch.Tensor):
    return jax.dlpack.from_dlpack(x.detach())


def from_jax_array(x) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(x)


def import_jax_based_package(jax_pre_allocate_memory):
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{jax_pre_allocate_memory}"

    global jax, brax, jnp
    import jax, jax.numpy as jnp
    import brax.envs


class BraxProblem(BaseProblem):
    def __init__(
        self,
        env_name: str,
        max_episode_length: int,
        seed: int = 42,
        pop_size: int = None,
        backend: str = None,
        output_transform: Callable = torch.tanh,
        jax_pre_allocate_memory=0.5,
    ):
        import_jax_based_package(jax_pre_allocate_memory)

        self.env: brax.envs.Env = (
            brax.envs.get_environment(env_name=env_name)
            if backend is None
            else brax.envs.get_environment(env_name=env_name, backend=backend)
        )

        self.batch_reset = jax.jit(jax.vmap(self.env.reset))
        self.batch_setp = jax.jit(jax.vmap(self.env.step))

        self.max_episode_length = max_episode_length

        self.pop_size = pop_size
        self.output_transform = output_transform
        self.randkey = jax.random.PRNGKey(seed)

    def evaluate(self, forest: Forest):
        if self.pop_size is None:
            self.pop_size = len(forest)

        self.randkey, subkey = jax.random.split(self.randkey)
        reset_keys = jax.random.split(subkey, self.pop_size)
        brax_states = self.batch_reset(reset_keys)

        done = jnp.zeros(self.pop_size, dtype=bool)
        total_reward = jnp.zeros(self.pop_size)

        for _ in range(self.max_episode_length):
            # jax observations to pytorch
            observations = from_jax_array(brax_states.obs)

            # torch action to jax
            torch_output = forest.forward(observations)
            torch_action = self.output_transform(torch_output)
            actions = to_jax_array(torch_action)

            # jax step
            brax_states = self.batch_setp(brax_states, actions)

            observations, reward, currunt_done = (
                brax_states.obs,
                brax_states.reward,
                brax_states.done,
            )

            # update reward
            total_reward += reward * ~done

            # update done
            done = jnp.logical_or(done, currunt_done)

            # break if done
            if jnp.all(done):
                break

        return from_jax_array(total_reward)

    @property
    def problem_dim(self):
        return self.env.observation_size

    @property
    def solution_dim(self):
        return self.env.action_size
