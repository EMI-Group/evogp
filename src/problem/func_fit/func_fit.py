from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from src.config import ProblemConfig
from src.core import Problem, State
from src.gp.operations import sr_fitness

@dataclass(frozen=True)
class FuncFitConfig(ProblemConfig):
    error_method: str = 'mse'

    def __post_init__(self):
        assert self.error_method in {'mse', 'rmse', 'mae', 'mape'}


class FuncFit(Problem):
    jitable = True

    def __init__(self, config: FuncFitConfig = FuncFitConfig()):
        self.config = config
        super().__init__(config)

    def evaluate(self, randkey, trees):
        res = sr_fitness(
            trees,
            data_points=self.inputs.astype(jnp.float32),
            targets=self.targets.astype(jnp.float32),
        )

        return res

    def show(self, randkey, state: State, act_func: Callable, params):
        predict = act_func(state, self.inputs, params)
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        loss = -self.evaluate(randkey, state, act_func, params)
        msg = ""
        for i in range(inputs.shape[0]):
            msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
