import jax
import numpy as np
from evogp.algorithm import GeneticProgramming as GP
from evogp.algorithm import DiscreteConst, BasicCrossover, BasicMutation, TournamentSelection
from evogp.pipeline import General
from evogp.problem.func_fit import New_Prob
import sys
import pandas as pd

def update_csv(file_path, gen, mse):
    df = pd.read_csv(file_path, index_col='gen')
    
    mse_column = 'MSE'
    
    if gen in df.index:
        df.at[gen, mse_column] = mse
    else:
        new_row = pd.DataFrame({mse_column: [mse]}, index=[gen])
        df = pd.concat([df, new_row])

    df.to_csv(file_path, index_label='gen')

def main():
    target_name = sys.argv[1]
    try:
        with open(f"output/new_prob_{target_name}.csv", 'x') as file:
            file.write('gen')
    except:
        pass

    alg = GP(
        pop_size=20000,
        num_inputs=15,
        num_outputs=1,
        max_len=1024,
        max_sub_tree_len=128,
        crossover=BasicCrossover(),
        crossover_rate=0.5,
        mutation=(
            BasicMutation(
                max_sub_tree_len=128,
                leaf_prob=[0, 0, 0,  0, 0.1, 0.2,   1,   1,   1,    1],
                # size   =[1, 3, 7, 15,  31,  63, 127, 255, 511, 1023]
                ),
        ),
        mutation_rate=(0.5,),
        selection=TournamentSelection(20, 0.9, False),
        const=DiscreteConst(
            jax.numpy.array([-1., 0., 1.])
        ),
        leaf_prob=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    prob = New_Prob(target_name)
    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(42)
    state = pipeline.setup(key)

    jit_step = jax.jit(pipeline.step)

    for i in range(500):
        state, fitnesses = jit_step(state)
        
        fitnesses = jax.device_get(fitnesses)
        print(f'{i}, max: {-np.max(fitnesses)}, min: {-np.min(fitnesses)}, mean: {-np.mean(fitnesses)}')
        update_csv(f"output/new_prob_{target_name}.csv", i, -np.max(fitnesses))

    np.save(f"output/new_prob_{target_name}.npy", state.alg_state.trees)


if __name__ == '__main__':
    main()
