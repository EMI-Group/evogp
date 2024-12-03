from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from brax import envs
from .rl_env import RLEnv


class BraxEnv(RLEnv):
    def __init__(
            self,
            output_transform: Callable,
            output_length: int = 1,
            env_name: str = "ant",
            back_end: str = "generalized"
    ):
        super().__init__(output_transform, output_length)
        self.env = envs.create(env_name=env_name, backend=back_end)

    def env_step(self, randkey, env_state, action):
        state = self.env.step(env_state, action)
        return state.obs, state, state.reward, state.done.astype(jnp.bool_), state.info

    def env_reset(self, randkey):
        init_state = self.env.reset(randkey)
        return init_state.obs, init_state

    @property
    def input_shape(self):
        return (self.env.observation_size,)

    @property
    def output_shape(self):
        return (self.env.action_size,)

    # def show(self, randkey, state: State, act_func: Callable, params, save_path=None, height=512, width=512,
    #          duration=0.1, *args,
    #          **kwargs):
    #
    #     import jax
    #     import imageio
    #     import numpy as np
    #     from brax.io import image
    #     from tqdm import tqdm
    #
    #     obs, env_state = self.reset(randkey)
    #     reward, done = 0.0, False
    #     state_histories = []
    #
    #     def step(key, env_state, obs):
    #         key, _ = jax.random.split(key)
    #         net_out = act_func(state, obs, params)
    #         action = self.config.output_transform(net_out)
    #         obs, env_state, r, done, _ = self.step(randkey, env_state, action)
    #         return key, env_state, obs, r, done
    #
    #     while not done:
    #         state_histories.append(env_state.pipeline_state)
    #         key, env_state, obs, r, done = jax.jit(step)(randkey, env_state, obs)
    #         reward += r
    #
    #     imgs = [image.render_array(sys=self.env.sys, state=s, width=width, height=height) for s in
    #             tqdm(state_histories, desc="Rendering")]
    #
    #     def create_gif(image_list, gif_name, duration):
    #         with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
    #             for image in image_list:
    #                 # 确保图像的数据类型正确
    #                 formatted_image = np.array(image, dtype=np.uint8)
    #                 writer.append_data(formatted_image)
    #
    #     create_gif(imgs, save_path, duration=0.1)
    #     print("Gif saved to: ", save_path)
    #     print("Total reward: ", reward)
