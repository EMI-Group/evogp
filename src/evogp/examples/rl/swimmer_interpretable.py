import sys
import time

sys.path.append("/wuzhihong/EvoGP")
import jax
import jax.numpy as jnp
from evogp.cuda.operations import forward
import numpy as np
from evogp.algorithm import GeneticProgramming as GP
from evogp.algorithm import DiscreteConst, BasicSelection, BasicCrossover, BasicMutation, TournamentSelection
from evogp.pipeline import General
from evogp.problem.rl_env import BraxEnv
import pandas as pd
from evogp.cuda.utils import *

import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


def update_csv(file_path, gen, max, min, mean, time, env_time):
    df = pd.read_csv(file_path, index_col="gen")

    if gen in df.index:
        df.at[gen, "max"] = max
        df.at[gen, "min"] = min
        df.at[gen, "mean"] = mean
        df.at[gen, "time"] = time
        df.at[gen, "env_time"] = env_time
    else:
        new_row = pd.DataFrame(
            {"max": [max], "min": [min], "mean": [mean], "time": [time], "env_time": [env_time]}, index=[gen]
        )
        df = pd.concat([df, new_row])

    df.to_csv(file_path, index_label="gen")


def main():
    sys.argv.append("0")

    try:
        with open(f"output/swimmer_{sys.argv[1]}.csv", "x") as file:
            file.write("gen")
    except Exception:
        pass

    alg = GP(
        max_len=32,
        pop_size=2000,
        num_inputs=8,
        num_outputs=2,
        crossover=BasicCrossover(),
        crossover_rate=0.7,
        mutation=(
            BasicMutation(
                max_sub_tree_len=8,
                leaf_prob=[0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1],
                # size   =[1,   3,  7, 15, 31, 63, 127, 255, 511, 1023]
            ),
        ),
        mutation_rate=(0.7,),
        selection=TournamentSelection(20, 0.9, False),
        const=DiscreteConst(jnp.arange(-5, 5.5, 0.5, dtype=jnp.float32)),
        leaf_prob=[0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1, 1],
        func_prob_dict={
            "if": 1,
            "+": 1,
            "-": 1,
            "*": 1,
            "/": 1,
            "pow": 1,
            "max": 1,
            "min": 1,
            "<": 1,
            ">": 1,
            "<=": 1,
            ">=": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "sinh": 1,
            "cosh": 1,
            "tanh": 1,
            "log": 1,
            "exp": 1,
            "inv": 1,
            "neg": 1,
            "abs": 1,
            "sqrt": 1,
        },
    )
    prob = BraxEnv(
        output_transform=lambda x: jax.numpy.tanh(x),
        output_length=2,
        env_name="swimmer",
    )

    pipeline = General(alg, prob)

    key = jax.random.PRNGKey(int(sys.argv[1]))
    state = pipeline.setup(key)

    def new_step(self, state):
        trees = jax.jit(self.algorithm.ask)(state.alg_state)
        fitness, env_time = self.problem.evaluate(state.randkey, trees)
        alg_state = jax.jit(self.algorithm.tell)(state.alg_state, fitness)
        return (
            jax.jit(state.update)(
                alg_state=alg_state,
                generation=state.generation + 1,
            ),
            fitness,
            env_time,
        )

    def new_evaluate(self, randkey, trees):

        @jax.jit
        def env_step(randkey, env_states, fitnesses, done, actions):
            rng, randkey = jax.random.split(randkey)
            vmap_keys = jax.random.split(rng, pop_size)
            obs, env_states, reward, current_done, _ = jax.vmap(self.step)(vmap_keys, env_states, actions)
            fitnesses += reward * jnp.logical_not(done)
            done = jnp.logical_or(done, current_done)
            return randkey, env_states, fitnesses, done, obs

        @jax.jit
        def env_init(randkey):
            reset_keys = jax.random.split(randkey, pop_size)
            obs, env_states = jax.vmap(self.reset)(reset_keys)  # observations
            fitnesses, done = jnp.zeros(pop_size), jnp.zeros(pop_size, dtype=jnp.bool_)
            return env_states, fitnesses, done, obs

        pop_size = trees.shape[0]
        start_time = time.time()
        env_states, fitnesses, done, obs = env_init(randkey)
        end_time = time.time()
        env_time = end_time - start_time
        print(env_time)

        for _ in range(self.max_step):
            actions = forward(trees, obs, result_length=self.output_length)
            actions = jax.jit(jax.vmap(self.output_transform))(actions)

            start_time = time.time()
            randkey, env_states, fitnesses, done, obs = env_step(randkey, env_states, fitnesses, done, actions)
            end_time = time.time()
            env_time += end_time - start_time
            if jnp.all(done):
                break

        print(env_time)
        return fitnesses, env_time

    import types

    pipeline.step = types.MethodType(new_step, pipeline)
    prob.evaluate = types.MethodType(new_evaluate, prob)

    for i in range(200):
        start_time = time.time()
        state, fitnesses, env_time = pipeline.step(state)
        time_per_gen = time.time() - start_time

        fitnesses = jax.device_get(fitnesses)
        print(
            f"gen: {i}, max: {np.max(fitnesses)}, min: {np.min(fitnesses)}, mean: {np.mean(fitnesses)}, time_per_gen: {time_per_gen}, env_time: {env_time}"
        )
        update_csv(
            f"output/swimmer_{sys.argv[1]}.csv",
            i,
            np.max(fitnesses),
            np.min(fitnesses),
            np.mean(fitnesses),
            time_per_gen,
            env_time,
        )

    def save_the_top10(trees, fitnesses):
        idx = jnp.argsort(fitnesses)[::-1]
        top10_t = trees[idx[:10]]
        np.save(f"output/swimmer_{sys.argv[1]}_top10.npy", top10_t)
        top10_f = fitnesses[idx[:10]]
        for i in range(10):
            graph_i = to_graph(top10_t[i])
            to_png(graph_i, f"output/swimmer{i}_fitness{top10_f[i]}.png")

    save_the_top10(state.alg_state.trees, fitnesses)


if __name__ == "__main__":
    main()
