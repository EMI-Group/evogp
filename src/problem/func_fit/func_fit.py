import jax.numpy as jnp

from ..problem import Problem

from src.cuda.operations import sr_fitness, constant_sr_fitness

class FuncFit(Problem):
    def __init__(self, error_method="mse"):
        super().__init__()
        assert error_method in {"mse", "rmse", "mae", "mape"}
        self.error_method = error_method

    def evaluate(self, randkey, trees):
        res = constant_sr_fitness(
            trees,
            data_points=self.inputs.astype(jnp.float32),
            targets=self.targets.astype(jnp.float32),
        )

        # import jax
        # pop_size = trees.shape[0]
        # res = jax.random.uniform(randkey, (pop_size,))

        return -res

    def show(self, randkey, prefix_trees):
        pass
        # predict = act_func(state, self.inputs, params)
        # inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        # loss = -self.evaluate(randkey, state, act_func, params)
        # msg = ""
        # for i in range(inputs.shape[0]):
        #     msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        # msg += f"loss: {loss}\n"
        # print(msg)

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
