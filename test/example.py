from src.gp.jax_backend.gp import NormalGP
from src.config import *
from src.gp.enum import tree2str
from src.problems import XOR, FuncFitConfig

from src.gp.jax_backend.pipeline import Pipeline


def main():
    conf = Config(
        basic=BasicConfig(
            seed=43,
            fitness_target=-1e-6,
            generation_limit=50,
        ),
        gp=GPConfig(
            pop_size=1000,
            max_subtree_size=8,
            new_tree_depth=2,
            var_prob=0.5,
            const_prob=0,
            const_pool=(1., )
        ),
        problem=FuncFitConfig(
            error_method='rmse'
        )
    )
    algorithm = NormalGP(conf)
    # full pipeline
    pipeline = Pipeline(conf, algorithm, XOR)
    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    print(tree2str(best))
    pipeline.show(state, best)
    #
    # import numpy as np
    #
    # best = jax.device_get(best)
    #
    # # 设置numpy打印选项
    # np.set_printoptions(threshold=99999)
    # print(jax.device_get(best.subtree_size))


if __name__ == '__main__':
    main()
