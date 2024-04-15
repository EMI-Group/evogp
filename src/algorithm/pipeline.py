"""
pipeline for jitable env like func_fit, gymnax
"""

from typing import Type

from jax import vmap

from src.config import *
from src.utils.state import State
from src.problem.problem import Problem

from src.gp.enum import FUNCS
from .operations import *

from src.cuda.utils import *

class Pipeline:
    def __init__(self, conf: Config, problem_type: Type[Problem]):
        assert problem_type.jitable, "problem must be jitable"

        self.config = conf
        self.problem = problem_type(conf.problem)

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.func_cum = func_cum_prob(self.config.gp.func)

    def setup(self):
        key = jax.random.PRNGKey(self.config.basic.seed)
        k1, key = jax.random.split(key)

        # fig = plt.figure()
        # if self.problem.input_shape[1] == 1:
        #     x = self.problem.inputs
        #     y = self.problem.targets
        #     ax = fig.add_subplot()
        #     ax.plot(x.ravel(), y.ravel(), label='base')
        # else:
        #     x = self.problem.inputs
        #     y = self.problem.targets
        #     x_split = jnp.split(x, self.problem.input_shape[1], axis=1)
        #     ax = fig.add_subplot(projection='3d')
        #     ax.plot(x_split[0].ravel(), x_split[1].ravel(), y.ravel(), label='base')

        state = State(
            randkey=key,
            trees=self.new_trees(k1, self.config.gp.max_size),
            generation=0,
            # ax=ax,
        )
        return state

    # @partial(jit, static_argnums=(0,))
    def step(self, state: State):
        fitness = self.evaluate(state)

        jax.debug.print("Gen {}: {}", state.generation, jnp.min(fitness))
        best = state.trees[jnp.argmin(fitness)]
        jax.debug.print("Gen {}: {}\n\t{}", state.generation, jnp.min(fitness), cuda_tree_to_string(best))
        graph = to_graph(best)
        # to_png(graph, f"output/{state.generation}.png")
        print("-------------------", to_sympy(graph))
        # out = forward(jnp.tile(best, [self.problem.input_shape[0], 1]), self.problem.inputs.astype(jnp.float32))

        # for line in state.ax.lines:
        #     from matplotlib import lines
        #     line: lines.Line2D
        #     if line.get_label() == 'regress':
        #         line.remove()
        #
        # if self.problem.input_shape[1] == 1:
        #     x = self.problem.inputs
        #     state.ax.plot(x.ravel(), out.ravel(), color='red', label='regress')
        # else:
        #     x = self.problem.inputs
        #     x_split = jnp.split(x, self.problem.input_shape[1], axis=1)
        #     state.ax.plot(x_split[0].ravel(), x_split[1].ravel(), out.ravel(), color='red', label='regress')
        # plt.pause(0.01)

        k1, k2, k3, k4, new_key = jax.random.split(state.randkey, 5)

        pop_size, pop_idx = fitness.shape[0], jnp.arange(0, fitness.shape[0])

        # # size penalty
        # fitness += (jnp.max(fitness) - jnp.min(fitness)) / 100 / vmap(tree_size)(state.trees)

        # TODO: is this a rank?
        sorted_idx = jnp.argsort(fitness)
        ranks = jnp.argsort(sorted_idx)

        selected_num = int(self.config.gp.parent_rate * pop_size)
        selected_p = jnp.where(ranks < selected_num, 1, 0)
        selected_p = selected_p / jnp.sum(selected_p)

        left, right = jax.random.choice(k1, pop_idx, p=selected_p, shape=(2, pop_size), replace=True)

        children = self.crossover_trees(
            k2,
            state.trees,
            left,
            right,
        )

        # children = state.trees

        mutated_children = self.mutate_trees(
            k3,
            children,
        )

        mutate_p = jax.random.uniform(k4, (pop_size,))
        selected_mask = (mutate_p < self.config.gp.mutate_prob)[:, None]

        children = jnp.where(
            selected_mask,
            mutated_children,
            children,
        )

        return state.update(
            trees=children,
            randkey=new_key,
            generation=state.generation + 1,
            # ax=state.ax,
        )

    def evaluate(self, state: State):
        return self.problem.evaluate(state.randkey, state.trees)

    def mutate_trees(self, key, trees):
        tree_sizes = vmap(tree_size)(trees)

        k1, k2 = jax.random.split(key)

        sub_trees = self.new_trees(k1, self.config.gp.subtree.subtree_size)

        ## indices = vmap(random_idx)(jax.random.split(k2, num=trees.shape[0]), tree_sizes)
        indices = jax.random.randint(k2, (trees.shape[0],), 0, tree_sizes)

        # l_size = vmap(lambda x, i: tree_size(jnp.array([x[i]])))(trees, indices)
        # r_size = vmap(tree_size)(sub_trees)
        # new_size = r_size - l_size + tree_sizes
        # jax.debug.print("mutate: {}", jnp.count_nonzero(new_size > 1024))

        return mutation(
            trees,
            indices,
            sub_trees,
        )

    def crossover_trees(self, key, trees, l_indices, r_indices):
        tree_sizes = vmap(tree_size)(trees)

        k1, k2 = jax.random.split(key)

        ## l_pos = vmap(random_idx)(jax.random.split(k1, num=trees.shape[0]), tree_sizes[l_indices])
        ## r_pos = vmap(random_idx)(jax.random.split(k2, num=trees.shape[0]), tree_sizes[r_indices])
        l_pos = jax.random.randint(k1, (trees.shape[0],), 0, tree_sizes[l_indices])
        r_pos = jax.random.randint(k2, (trees.shape[0],), 0, tree_sizes[r_indices])

        # l_size = vmap(lambda x, i: tree_size(jnp.array([x[i]])))(trees[l_indices], l_pos)
        # r_size = vmap(lambda x, i: tree_size(jnp.array([x[i]])))(trees[r_indices], r_pos)
        # new_size = r_size - l_size + tree_sizes[l_indices]
        # jax.debug.print("crossover: {}", jnp.count_nonzero(new_size > 1024))

        nodes = jnp.stack([l_pos, r_pos], axis=1).astype(jnp.int16)

        return crossover(
            trees,
            l_indices,
            r_indices,
            nodes,
        )

    def new_trees(self, key, size):

        k1, k2 = jax.random.split(key)

        if self.config.gp.const.type == "discrete":
            consts = jnp.array(self.config.gp.const.pool)
        else:
            consts = (
                    jax.random.normal(k1, (self.config.gp.const.points_cnt,))
                    * self.config.gp.const.std
                    + self.config.gp.const.mean
            )

        trees = generate(
            seed=k2,
            pop_size=self.config.gp.pop_size,
            max_len=size,
            num_inputs=self.problem.input_shape[-1],
            num_outputs=self.config.gp.num_outputs,
            leaf_prob=jnp.array(self.config.gp.subtree.leaf_prob),
            functions_prob_accumulate=jnp.array(self.func_cum),
            const_samples=consts,
            output_prob=self.config.gp.output_prob,
            const_prob=self.config.gp.const.const_prob,
        )

        return trees

    def pop_forward(self, trees, inputs):
        return forward(
            trees,
            inputs,
            result_length=self.config.gp.num_outputs,
        )


def func_cum_prob(conf: FuncConfig):
    probs = np.zeros(len(FUNCS))
    for func, prob in zip(conf.pool, conf.prob):
        probs[func] = prob

    return probs.cumsum()

# def random_idx(key, size):
#     return jax.random.randint(key, (), 0, size - 1)
