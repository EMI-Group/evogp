import jax
from src.algorithm import GeneticProgramming as GP
from src.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation
from src.pipeline.symbolic_regression import SymbolicRegression


def main():
    alg = GP(
        pop_size=100,
        num_inputs=2,
        num_outputs=1,
        crossover=BasicCrossover(),
        mutation=BasicMutation(),
        selection=BasicSelection(
            elite_rate=0.1,
            survive_rate=0.4,
        ),
        const=DiscreteConst(
            jax.numpy.array([-1, 0, 1])
        ),
    )

    # prob = XOR()
    pipeline = SymbolicRegression(alg, None)

    key = jax.random.PRNGKey(42)
    state = pipeline.setup(key)

    for i in range(100):
        state = pipeline.step(state)




if __name__ == '__main__':
    main()
