from src.gp import NormalGP
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
            max_size=100,
            max_subtree_size=8,
            new_tree_depth=2,
            var_prob=0.5,
            const_prob=0,
            const_pool=(1.,)
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

    tree = state.trees[11]
    print(tree2str(tree))
    assert check_correct(tree)

    new = mutation(tree, jax.random.PRNGKey(14), state)
    print(tree2str(new))
    assert check_correct(new)

    # for i in range(40):
    #     key = jax.random.PRNGKey(i)
    #     new = mutation(tree, key, state)
    #     assert check_correct(new)
    #     print(tree2str(new))


if __name__ == '__main__':
    main()
