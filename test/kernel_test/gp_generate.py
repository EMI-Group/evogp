import sys
import os
sys.path.append(os.getcwd())

from src.config import *
from src.core.kernel.utils import *

from src.gp.cuda_backend.pipeline import Pipeline
from src.problems import XOR, XOR3d, FuncFitConfig


def main():
    conf = Config(
        gp=GPConfig(
            pop_size=10000
        ),
        problem=FuncFitConfig(),
    )
    pipeline = Pipeline(conf, XOR3d)
    state = pipeline.setup()
    for _ in range(1000):
        state = pipeline.step(state)
    fitness = pipeline.evaluate(state)
    print(cuda_tree_to_string(state.trees[jnp.argmin(fitness)]))


if __name__ == '__main__':
    main()
