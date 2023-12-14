from typing import Callable

import jax
from jax import Array
import numpy as np

from .func_fit import FuncFit


class General(FuncFit):

    def __init__(
            self,
            func: Callable,
            low_bounds: Array,
            upper_bounds: Array,
            # step_size: Array,
            error_method: str = 'mse',
    ):
        super().__init__(error_method)
        try:
            out = func(low_bounds)
        except Exception as e:
            raise ValueError(f"func(low_bounds) raise an exception: {e}")
        assert low_bounds.shape == upper_bounds.shape

        self.func = func
        self.low_bounds = low_bounds
        self.upper_bounds = upper_bounds

        self.data_inputs = None
        self.data_outputs = None

    def generate(
            self,
            method: str = 'sample',
            num_samples: int = 100,
            step_size: Array = None,
    ):
        assert method in {'sample', 'grid'}

        if method == 'sample':
            assert num_samples > 0, f"num_samples must be positive, got {num_samples}"

            inputs = np.zeros((num_samples, self.low_bounds.shape[0]), dtype=np.float32)
            for i in range(self.low_bounds.shape[0]):
                inputs[:, i] = np.random.uniform(
                    low=self.low_bounds[i],
                    high=self.upper_bounds[i],
                    size=(num_samples,)
                )
        elif method == 'grid':
            assert step_size is not None, "step_size must be provided when method is 'grid'"
            assert step_size.shape == self.low_bounds.shape, "step_size must have the same shape as low_bounds"
            assert np.all(step_size > 0), "step_size must be positive"

            inputs = np.zeros((1, 1))
            for i in range(self.low_bounds.shape[0]):
                new_col = np.arange(self.low_bounds[i], self.upper_bounds[i], step_size[i])
                inputs = cartesian_product(inputs, new_col[:, None])
            inputs = inputs[:, 1:]
        else:
            raise ValueError(f"Unknown method: {method}")

        outputs = jax.vmap(self.func)(inputs)

        self.data_inputs = inputs
        self.data_outputs = outputs

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return self.data_inputs.shape

    @property
    def output_shape(self):
        return self.data_outputs.shape


def cartesian_product(arr1, arr2):
    assert arr1.ndim == arr2.ndim, "arr1 and arr2 must have the same number of dimensions"
    assert arr1.ndim <= 2, "arr1 and arr2 must have at most 2 dimensions"

    len1 = arr1.shape[0]
    len2 = arr2.shape[0]

    repeated_arr1 = np.repeat(arr1, len2, axis=0)
    tiled_arr2 = np.tile(arr2, (len1, 1))

    new_arr = np.concatenate((repeated_arr1, tiled_arr2), axis=1)
    return new_arr
