import sys

sys.path.append("/home/kelvin/test/TensorGP")
import jax
import jax.numpy as jnp
import numpy as np
from src.algorithm import DiscreteConst, GeneticProgramming as GP
from src.algorithm import BasicCrossover
from src.algorithm import (
    BasicMutation,
    HoistMutation,
    SinglePointMutation,
    LambdaPointMutation,
    InsertMutation,
    DeleteMutation,
)
from src.algorithm import (
    BasicSelection,
    TruncationSelection,
    RouletteSelection,
    RankSelection,
    TournamentSelection,
)
from src.pipeline import General
from src.problem.func_fit import GeneralFuncFit
import time

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main():
    alg = GP(
        pop_size=80,
        num_inputs=2,
        num_outputs=1,
        max_len=1024,
        max_sub_tree_len=128,
        crossover=BasicCrossover(),
        crossover_rate=0.9,
        mutation=(
            BasicMutation(),
            HoistMutation(),
            SinglePointMutation(),
            LambdaPointMutation(0.9),
            InsertMutation(),
            DeleteMutation(),
        ),
        mutation_rate=(0.5, 0.8, 0.9, 0.05, 0.1, 0.1),
        selection=TournamentSelection(50, 0.9, False),
        const=DiscreteConst(
            jax.numpy.array(
                [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            )
        ),
        leaf_prob=[0, 0, 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1],
        # size   =[2, 4, 8, 16,   32,   64,  128, 256, 512, 1024]
    )

    # create general function fitting problem and then initialize it
    # in this example, all data points are generated by sampling from a grid of 2D points
    prob = GeneralFuncFit(
        func=lambda x: (x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)),
        low_bounds=jax.numpy.array([-5, -5]),
        upper_bounds=jax.numpy.array([5, 5]),
    )
    prob.generate(
        method="sample",
        num_samples=64*64,
    )

    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(42)
    state = pipeline.setup(key)

    jit_step = jax.jit(pipeline.step)
    # jit_step = pipeline.step

    print("--------initialization finished--------")
    start_time = time.time()
    for i in range(10): 
        state, fitnesses = jit_step(state)

        fitnesses = jax.device_get(fitnesses)
        print(f"gen:{i}, max: {np.max(fitnesses)},", end=" ")
        from src.cuda.utils import tree_size, from_cuda_node

        trees = pipeline.algorithm.ask(state.alg_state)
        print(f"mean_size: {np.mean(jax.vmap(tree_size)(trees))}")
        # print(from_cuda_node(trees))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
