import numpy as np
from numpy.typing import NDArray
from .func_fit import FuncFit


class Normalized(FuncFit):
    def __init__(self,
                 origin_problem: FuncFit,
                 method: str = "interval",
                 bound: NDArray = np.array([0, 1]),
                 mean: float = 0,
                 std: float = 1
                 ):

        super().__init__(origin_problem.error_method)
        assert method in {"interval", "gaussian"}, "method must be one of {'interval', 'gaussian'}"
        if method == "gaussian":
            assert mean is not None, "mean must be specified when method is 'gaussian'"
            assert std is not None, "std must be specified when method is 'gaussian'"
        if method == "interval":
            assert bound is not None, "bound must be specified when method is 'interval'"
            assert bound.shape == (2,), "bound must be a tuple of length 2"
            assert bound[0] < bound[1], "bound[0] must be smaller than bound[1]"

        self.origin_problem = origin_problem
        self.method = method
        self.bound = bound
        self.mean = mean
        self.std = std

    @property
    def inputs(self):
        if self.method == "interval":
            min_ = np.min(self.origin_problem.inputs)
            max_ = np.max(self.origin_problem.inputs)
            return (self.origin_problem.inputs - min_) / (max_ - min_) * (self.bound[1] - self.bound[0]) + self.bound[0]
        if self.method == "gaussian":
            origin_mean = np.mean(self.origin_problem.inputs)
            origin_std = np.std(self.origin_problem.inputs)
            return (self.origin_problem.inputs - origin_mean) / origin_std * self.std + self.mean

    @property
    def targets(self):
        # I don't know whether this should be normalized or not
        return self.origin_problem.targets

    def input_shape(self):
        return self.origin_problem.input_shape

    def output_shape(self):
        return self.origin_problem.output_shape
