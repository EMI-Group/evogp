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

    import jax
    import jax.numpy as jnp
    from mujoco_playground import MjxEnv, registry


class MujocoProblem(BaseProblem):
    def __init__(
        self,
        env_name: str,
        max_episode_length: int,
        seed: int = 42,
        pop_size: int = None,
        output_transform: Callable = torch.tanh,
        jax_pre_allocate_memory=0.5,
    ):
        import_jax_based_package(jax_pre_allocate_memory)

        self.env: MjxEnv = registry.load(env_name=env_name)

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
        mjx_states = self.batch_reset(reset_keys)

        done = jnp.zeros(self.pop_size, dtype=bool)
        total_reward = jnp.zeros(self.pop_size)

        for _ in range(self.max_episode_length):
            # jax observations to pytorch
            obs = mjx_states.obs
            if not isinstance(obs, jax.Array):
                if "state" in obs:
                    obs = obs["state"]
                else:
                    raise ImportError(
                        f"This Pytree observation space is not supported yet: {obs}"
                    )
            observations = from_jax_array(obs)

            # torch action to jax
            torch_output = forest.forward(observations)
            torch_action = self.output_transform(torch_output)
            actions = to_jax_array(torch_action)

            # jax step
            mjx_states = self.batch_setp(mjx_states, actions)

            reward, currunt_done = (
                mjx_states.reward,
                mjx_states.done,
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
