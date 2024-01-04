import jax
import jax.numpy as jnp

from src.problem.func_fit import GeneralFuncFit
from src.cuda.operations import generate, sr_fitness
from src.utils import dict2cdf

STEP_CNT = 32
LOW_BOUNDS = jax.numpy.array([-5.0, -5.0])
UPPER_BOUNDS = jax.numpy.array([5.0, 5.0])


def main():
    prob = GeneralFuncFit(
        func=lambda x: (
                1 / (1 + jnp.power(x[0], -4)) +
                1 / (1 + jnp.power(x[1], -4))
        ),
        low_bounds=LOW_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
    )

    prob.generate(
        method='grid',
        step_size=jax.numpy.array((UPPER_BOUNDS - LOW_BOUNDS) / STEP_CNT)
    )
    print("generate finish")

    print(prob.data_inputs.shape)
    print(prob.data_inputs)

    trees = generate(
        key=jax.random.PRNGKey(42),
        leaf_prob=jnp.array(
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32
        ),
        funcs_prob_acc=dict2cdf({"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25}),
        const_samples=jax.numpy.array([-1., 0., 1.]),
        pop_size=10000000,
        max_len=8,
        num_inputs=2,
        num_outputs=1,
    )

    print(trees)
    import time

    tic = time.time()
    jax_sr_fitness = jax.jit(sr_fitness)
    for _ in range(50):
        fitnesses = jax_sr_fitness(
            trees,
            prob.data_inputs,
            prob.data_outputs,
        )
    fitnesses.block_until_ready()
    print(time.time() - tic)


if __name__ == '__main__':
    main()
