from functools import partial

from src.core.kernel.utils import *
from src.core.kernel.gp_kernel_bind import gp_generate_, gp_crossover_, gp_mutation_, gp_eval_, gp_sr_fitness_


@partial(jax.jit, static_argnames=[
    "pop_size",
    "max_len",
    "num_inputs",
    "num_outputs",
    "output_prob",
    "const_prob",
])
def generate(
        seed,
        pop_size,
        max_len,
        num_inputs,
        num_outputs,
        leaf_prob,
        functions_prob_accumulate,
        const_samples,
        output_prob,
        const_prob,
):
    return gp_generate_(
        seed,
        leaf_prob,
        functions_prob_accumulate,
        const_samples,
        pop_size=pop_size,
        max_len=max_len,
        variable_len=num_inputs,
        output_len=num_outputs,
        output_prob=output_prob,
        const_prob=const_prob
    )


@jax.jit
def crossover(prefixGPs, left, right, nodes):
    # left, right: indices of trees
    # nodes: pos of tree

    return gp_crossover_(prefixGPs, left, right, nodes)


@jax.jit
def mutation(prefixGPs, index, newGPs):
    return gp_mutation_(prefixGPs, index, newGPs)


@partial(jax.jit, static_argnames=["result_length"])
def forward(prefixGPs, variables, result_length=1):
    return gp_eval_(prefixGPs, variables, result_length=result_length)


@jax.jit
def sr_fitness(prefixGPs, data_points, targets):
    return gp_sr_fitness_(prefixGPs, data_points, targets, use_MSE=True)
