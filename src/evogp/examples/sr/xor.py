import jax
import numpy as np
from evogp.algorithm import GeneticProgramming as GP
from evogp.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation
from evogp.pipeline import General
from evogp.problem.func_fit import XOR


def main():
    alg = GP(
        pop_size=1000,
        num_inputs=2,
        num_outputs=1,
        crossover=BasicCrossover(),
        mutation=BasicMutation(),
        selection=BasicSelection(
            elite_rate=0.1,
            survivor_rate=0.4,
        ),
        const=DiscreteConst(
            jax.numpy.array([-1., 0., 1.])
        ),
        leaf_prob=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    prob = XOR()
    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(42)
    state = pipeline.setup(key)

    jit_step = jax.jit(pipeline.step)

    for i in range(100):
        state, fitnesses = jit_step(state)

        fitnesses = jax.device_get(fitnesses)
        print(f'max: {np.max(fitnesses)}, min: {np.min(fitnesses)}, mean: {np.mean(fitnesses)}')


if __name__ == '__main__':
    main()
