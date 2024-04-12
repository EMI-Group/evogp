import sys

sys.path.append("/home/kelvin/test/TensorGP")
import jax
import numpy as np
from src.algorithm import GeneticProgramming as GP
from src.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation, TournamentSelection
from src.pipeline import General
from src.problem.rl_env import BraxEnv


def main():
    alg = GP(
        pop_size=10000,
        num_inputs=27,
        num_outputs=8,
        crossover=BasicCrossover(),
        crossover_rate=0.9,
        mutation=(
            BasicMutation(),

        ),
        mutation_rate=(0.1,),
        selection=TournamentSelection(20, 0.9, False),
        const=DiscreteConst(
            jax.numpy.array([-1., 0., 1.])
        ),
        leaf_prob=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    prob = BraxEnv(
        output_transform=lambda x: jax.numpy.tanh(x),
        output_length=8,
        env_name='ant',
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
