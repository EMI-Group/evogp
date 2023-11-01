from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.gp.enum import Func


@dataclass(frozen=True)
class BasicConfig:
    seed: int = 42
    fitness_target: float = 1.0
    generation_limit: int = 1000


@dataclass(frozen=True)
class ConstConfig:
    type: str = 'discrete'
    pool: Tuple = (-1, 1, 2, -2, 0.5, -0.5)
    mean: float = 0.0
    std: float = 1.0
    points_cnt: int = 100

    const_prob: float = 0.25  # probability of const in leaves

    def __post_init__(self):
        assert self.type in ('discrete', 'continuous'), "type must in {discrete, continuous}"
        assert 0 < self.const_prob < 1, "const_prob must in (0, 1)"


@dataclass(frozen=True)
class FuncConfig:
    pool: Tuple = (
        Func.IF,
        Func.ADD,
        Func.SUB,
        Func.MUL,
        Func.DIV,
        Func.MAX,
        Func.MIN,
        Func.LT,
        Func.GT,
        Func.LE,
        Func.GE,
        Func.SIN,
        Func.COS,
        Func.SINH,
        Func.COSH,
        Func.LOG,
        Func.EXP,
        Func.INV,
        Func.NEG,
        Func.POW2,
        Func.POW3,
        Func.SQRT)

    # no prob for IF
    ##prob: Tuple = tuple([0] + ([1 / (len(pool) - 1)] * (len(pool) - 1)))
    # all equal prob
    prob: Tuple = tuple([1 / len(pool)] * len(pool))

    def __post_init__(self):
        assert len(self.pool) > 0, "pool must not be empty"
        assert len(self.pool) == len(self.prob), "pool and prob must have same length"
        assert np.allclose(np.sum(self.prob), 1), "prob must sum to 1"


@dataclass(frozen=True)
class SubtreeConfig:
    subtree_size: int = 32
    leaf_prob: Tuple = (0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 1)  # the probability of a node is leaf in each depth


@dataclass(frozen=True)
class GPConfig(BasicConfig):
    pop_size: int = 100
    max_size: int = 1024
    num_inputs: int = 2
    num_outputs: int = 1

    const: ConstConfig = ConstConfig()
    func: FuncConfig = FuncConfig()
    subtree: SubtreeConfig = SubtreeConfig()

    mutate_prob: float = 0.8

    output_prob: float = 0.5  # probability of output node in leaves
    parent_rate: float = 0.6  # the rate of parents in crossover


@dataclass(frozen=True)
class ProblemConfig:
    pass


@dataclass(frozen=True)
class Config:
    basic: BasicConfig = BasicConfig()
    gp: GPConfig = GPConfig()
    problem: ProblemConfig = ProblemConfig()
