import jax
import jax.numpy as jnp

import numpy as np
from src.algorithm import GeneticProgramming as GP
from src.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation
from src.pipeline import General
from src.problem.rl_env import GymNaxEnv


# TODO: Seems like bugs exist in Multi-Output GP Tree.

def main():
    alg = GP(
        pop_size=1000,
        num_inputs=6,
        num_outputs=3,
        crossover=BasicCrossover(),
        mutation=BasicMutation(),
        selection=BasicSelection(
            elite_rate=0.1,
            survivor_rate=0.4,
        ),
        const=DiscreteConst(
            jax.numpy.array([-1., 0., 1.])
        ),
        leaf_prob=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    prob = GymNaxEnv(
        output_transform=lambda x: jnp.array(jnp.argmax(x), dtype=jnp.int32),  # action space is {0, 1, 2}
        output_length=3,
        env_name='Acrobot-v1',
    )

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
