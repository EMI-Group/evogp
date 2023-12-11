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
    ):
        if func_prob_dict is None:
            func_prob_dict = {"+": 0.25, "-": 0.25, "*": 0.25, "/": 0.25}

        if leaf_prob is None:
            leaf_prob = [0, 0, 0, 0, 0, 1.0]

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.const = const

        self.config = {
            "pop_size": pop_size,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "max_len": max_len,
            "max_sub_tree_len": max_sub_tree_len,
            "leaf_prob": jnp.array(leaf_prob),
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
            max_len=self.config["max_sub_tree_len"],
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
        k1, k2, k3, k4 = jax.random.split(state.alg_key, 4)

        left, right = self.selection(k1, fitness, self.config)
        trees = self.crossover(k2, trees, left, right, self.config)
        trees = self.mutation(k3, trees, self.const, self.config)
        return state.update(
            alg_key=k4,
            trees=trees,
        )
