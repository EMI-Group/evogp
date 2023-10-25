from typing import Callable

from src.config import ProblemConfig
from .state import State


class Problem:
    jitable = None

    def __init__(self, problem_config: ProblemConfig = ProblemConfig()):
        self.config = problem_config

    def evaluate(self, randkey, state: State, act_func: Callable, params):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, randkey, state: State, act_func: Callable, params):
        """
        show how a genome perform in this problem
        """
        raise NotImplementedError
