from typing import Callable

import gymnax

from .rl_env import RLEnv


class GymNaxEnv(RLEnv):

    def __init__(
            self,
            output_transform: Callable,
            output_length: int = 1,
            env_name: str = "CartPole-v1",
    ):
        super().__init__(output_transform, output_length)
        assert env_name in gymnax.registered_envs, f"Env {env_name} not registered"
        self.env, self.env_params = gymnax.make(env_name)

    def env_step(self, randkey, env_state, action):
        return self.env.step(randkey, env_state, action, self.env_params)

    def env_reset(self, randkey):
        return self.env.reset(randkey, self.env_params)

    @property
    def input_shape(self):
        return self.env.observation_space(self.env_params).shape

    @property
    def output_shape(self):
        return self.env.action_space(self.env_params).shape

    def show(self, randkey, prefix_trees):
        raise NotImplementedError

    @classmethod
    def all_envs(cls):
        return gymnax.registered_envs
