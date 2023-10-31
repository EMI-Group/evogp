import sys
import os
sys.path.append(os.getcwd())

from src.config import *

from src.gp.cuda_backend.pipeline import Pipeline
from src.problems import XOR, FuncFitConfig


def main():
    conf = Config(
        gp=GPConfig(
            pop_size=10000
        ),
        problem=FuncFitConfig(),
    )
    pipeline = Pipeline(conf, XOR)
    state = pipeline.setup()
    for _ in range(100):
        state = pipeline.step(state)


if __name__ == '__main__':
    main()
