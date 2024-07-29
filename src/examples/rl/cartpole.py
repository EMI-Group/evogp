import sys

sys.path.append("/wuzhihong/TensorGP")

import jax
import numpy as np
from src.algorithm import GeneticProgramming as GP
from src.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation, TournamentSelection
from src.pipeline import General
from src.problem.rl_env import GymNaxEnv

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    alg = GP(
        pop_size=1000,
        num_inputs=2,
        num_outputs=1,
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
            jax.numpy.array([-1., 0., 1.])
        ),
        leaf_prob=[0, 0, 0,  0, 0.1, 0.2,   0.4,   0.8,   1,    1],
    )

    prob = GymNaxEnv(
        output_transform=lambda x: jax.numpy.tanh(x),
        output_length=1,
        env_name='CartPole-v1',
    )

    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(42)
    state = pipeline.setup(key)

    jit_step = jax.jit(pipeline.step)

    for i in range(1000):
        state, fitnesses = jit_step(state)

        fitnesses = jax.device_get(fitnesses)
        print(f'max: {np.max(fitnesses)}, min: {np.min(fitnesses)}, mean: {np.mean(fitnesses)}')


if __name__ == '__main__':
    main()
