import jax
import jax.numpy as jnp

from src.utils import State, dict2cdf
from src.cuda.operations import generate
from . import Mutation, Crossover, Selection, Const


class GeneticProgramming:

    def __init__(
            self,
            pop_size: int,
            num_inputs: int,
            num_outputs: int,
            crossover: Crossover,
            mutation: Mutation,
            selection: Selection,
            const: Const,
            max_len: int = 1024,
            max_sub_tree_len: int = 32,
            leaf_prob: list = None,
            output_prob: float = 0.5,
            const_prob: float = 0.5,
            func_prob_dict=None,
            crossover_rate = 0.5,
            mutation_rate = 0.5,
    ):
        if func_prob_dict is None:
            func_prob_dict = {"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25}

        if leaf_prob is None:
            leaf_prob = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.const = const
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.config = {
            "pop_size": pop_size,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "max_len": max_len,
            "max_sub_tree_len": max_sub_tree_len,
            "leaf_prob": jnp.array(leaf_prob, dtype=jnp.float32),
            "output_prob": output_prob,
            "const_prob": const_prob,
            "func_prob_cdf": dict2cdf(func_prob_dict),
        }

    def setup(self, randkey, state: State = State()):
        k1, k2, k3 = jax.random.split(randkey, 3)
        consts = self.const(k1)
        trees = generate(
            key=k2,
            leaf_prob=self.config["leaf_prob"],
            funcs_prob_acc=self.config["func_prob_cdf"],
            const_samples=consts,
            pop_size=self.config["pop_size"],
            max_len=self.config["max_len"],
            num_inputs=self.config["num_inputs"],
            num_outputs=self.config["num_outputs"],
            output_prob=self.config["output_prob"],
            const_prob=self.config["const_prob"]
        )
        return state.update(
            alg_key=k3,
            trees=trees,
        )

    def ask(self, state: State):
        return state.trees

    def tell(self, state: State, fitness):
        trees = self.ask(state)
        k1, k2, k3, k4, k5, k6 = jax.random.split(state.alg_key, 6)
        pop_size = self.config["pop_size"]

        # selection
        recipient_trees, fitness = self.selection(k1, trees, fitness, self.config)
        donor_trees, fitness = self.selection(k2, trees, fitness, self.config)

        # crossover
        crossover_trees = self.crossover(k3, recipient_trees, donor_trees, self.config)
        crossover_num = int(pop_size * self.crossover_rate)
        crossover_mask = (jnp.arange(0, pop_size) < crossover_num)[:, None]
        new_trees = jnp.where(crossover_mask, crossover_trees, trees)
        new_trees = jax.random.permutation(k4, new_trees)

        # mutation
        mutation_trees = self.mutation(k5, new_trees, self.const, self.config)
        mutation_num = int(pop_size * self.mutation_rate)
        mutation_mask = (jnp.arange(0, pop_size) < mutation_num)[:, None]
        new_trees = jnp.where(mutation_mask, mutation_trees, trees)

        return state.update(
            alg_key=k6,
            trees=new_trees,
        )