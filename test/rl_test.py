from src.config import *

from src.gp.pipeline import Pipeline
from src.problem import GymNaxEnv, GymNaxConfig


def main():
    conf = Config(
        gp=GPConfig(
            pop_size=1000
        ),
        problem=GymNaxConfig(),
    )
    pipeline = Pipeline(conf, GymNaxEnv)
    state = pipeline.setup()
    # if os.path.exists("output"):
    #     shutil.rmtree("output")
    # os.mkdir("output")
    for _ in range(1000):
        state = pipeline.step(state)
    fitness = pipeline.evaluate(state)
    # print(cuda_tree_to_string(state.trees[jnp.argmin(fitness)]))


if __name__ == '__main__':
    main()
