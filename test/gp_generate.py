import os
# sys.path.append(os.getcwd())

from src.config import *
from src.cuda.utils import *

from src.gp.pipeline import Pipeline
from src.problem import Polynomial_3d, FuncFitConfig
import shutil

def main():
    conf = Config(
        gp=GPConfig(
            pop_size=10000
        ),
        problem=FuncFitConfig(),
    )
    pipeline = Pipeline(conf, Polynomial_3d)
    state = pipeline.setup()
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.mkdir("output")
    for _ in range(1000):
        state = pipeline.step(state)
    fitness = pipeline.evaluate(state)
    print(cuda_tree_to_string(state.trees[jnp.argmin(fitness)]))


if __name__ == '__main__':
    main()
