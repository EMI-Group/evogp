import sys

sys.path.append("/wuzhihong/TensorGP")
import jax
import jax.numpy as jnp
import numpy as np
from evogp.algorithm import DiscreteConst, GeneticProgramming as GP
from evogp.algorithm import BasicCrossover
from evogp.algorithm import (
    BasicMutation,
    HoistMutation,
    SinglePointMutation,
    LambdaPointMutation,
    InsertMutation,
    DeleteMutation,
)
from evogp.algorithm import (
    BasicSelection,
    TruncationSelection,
    RouletteSelection,
    RankSelection,
    TournamentSelection,
)
from evogp.pipeline import General
from evogp.problem.func_fit import GeneralFuncFit
import time
import pandas as pd

import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

def update_csv(file_path, pop_size, datapoint, mse, time):
    df = pd.read_csv(file_path, index_col='pop')
    
    mse_column = f'{datapoint}^2 MSE'
    time_column = f'{datapoint}^2 Time'
    
    if pop_size in df.index:
        df.at[pop_size, mse_column] = mse
        df.at[pop_size, time_column] = time
    else:
        new_row = pd.DataFrame({mse_column: [mse], time_column: [time]}, index=[pop_size])
        df = pd.concat([df, new_row])

    df.to_csv(file_path, index_label='pop')

def main():
    try:
        with open(f"len1024_{sys.argv[3]}.csv", 'x') as file:
            file.write('pop')
    except:
        pass

    datapoint = int(sys.argv[2])
    alg = GP(
        pop_size=int(sys.argv[1]),
        num_inputs=2,
        num_outputs=1,
        max_len=1024,
        max_sub_tree_len=128,
        crossover=BasicCrossover(),
        crossover_rate=0.9,
        mutation=(
            BasicMutation(
                max_sub_tree_len=128,
                leaf_prob=[0, 0, 0,  0, 0.1, 0.2,   1,   1,   1,    1],
                # size   =[1, 3, 7, 15,  31,  63, 127, 255, 511, 1023]
                ),
        ),
        mutation_rate=(0.1,),
        selection=TournamentSelection(20, 0.9, False),
        const=DiscreteConst(
            jax.numpy.array(
                [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            )
        ),
        leaf_prob=[0, 0, 0,  0, 0.1, 0.2, 0.4, 0.8,   1,    1],
        # size   =[1, 3, 7, 15,  31,  63, 127, 255, 511, 1023]
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
        num_samples=datapoint * datapoint,
    )
    # prob.generate(
    #     method="grid",
    #     step_size=jax.numpy.array([2.5, 2.5]),
    # )

    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(int(sys.argv[3]))
    state = pipeline.setup(key)

    jit_step = jax.jit(pipeline.step)
    # jit_step = pipeline.step

    # hlo_computation = jax.xla_computation(jit_step)(state)
    # hlo_text = hlo_computation.as_hlo_text()  # 将 HLO 转换为文本格式
    # with open('function.hlo', 'w') as file:
    #     file.write(hlo_text)  # 写入文件

    state, fitnesses = jit_step(state)
    print("--------initialization finished--------")

    start_time = time.time()
    for i in range(50):
        state, fitnesses = jit_step(state)
        print(i)

        # fitnesses = jax.device_get(fitnesses)
        # print(f"gen:{i}, max: {np.max(fitnesses)},", end=" ")
        # from evogp.cuda.utils import tree_size, from_cuda_node

        # trees = pipeline.algorithm.ask(state.alg_state)
        # print(f"mean_size: {np.mean(jax.vmap(tree_size)(trees))}")

        # print(from_cuda_node(trees))

    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.3f} seconds")

    fitnesses = jax.device_get(fitnesses)
    print(f"pop:{alg.config['pop_size']}, max: {-np.max(fitnesses)},", end=" ")

    update_csv(f"len1024_{sys.argv[3]}.csv", alg.config['pop_size'], datapoint, -np.max(fitnesses), end_time - start_time)
    # start_time = time.time()
    # for i in range(20):
    #     state, fitnesses = jit_step(state)
    #     print(i)

    #     # fitnesses = jax.device_get(fitnesses)
    #     # print(f"gen:{i}, max: {np.max(fitnesses)},", end=" ")
    #     # from evogp.cuda.utils import tree_size, from_cuda_node

    #     # trees = pipeline.algorithm.ask(state.alg_state)
    #     # print(f"mean_size: {np.mean(jax.vmap(tree_size)(trees))}")

    #     # print(from_cuda_node(trees))

    # end_time = time.time()
    # print(f"\nExecution time: {end_time - start_time} seconds")

    # fitnesses = jax.device_get(fitnesses)
    # print(f"gen:{i}, max: {np.max(fitnesses)},", end=" ")

if __name__ == "__main__":
    main()
