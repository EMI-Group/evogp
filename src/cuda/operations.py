from functools import partial
import jax
from .gp_kernel_bind import gp_generate_, gp_crossover_, gp_mutation_, gp_eval_, gp_sr_fitness_


@partial(jax.jit, static_argnames=[
    "pop_size",
    "max_len",
    "num_inputs",
    "num_outputs",
    "output_prob",
    "const_prob",
])
def generate(
        key,
        leaf_prob,
        funcs_prob_acc,
        const_samples,
        pop_size,
        max_len,
        num_inputs,
        num_outputs=1,
        output_prob=0.5,
        const_prob=0.5,
):
    """
    The function generates a population of GPs via given sizes and probabilities.

    args:
        key: JAX random key

        leaf_prob: probabilities for generating leaf nodes.
            jax.Array(shape=(log2(MAX_LENGTH),), dtype=jnp.float32)
            The nodes of depth `i` in tree have probability `leaf_prob[i]` to be a leaf node.
            Larger length for "leaf_prob" may cause unexpected results.

        funcs_prob_acc: accumulated probabilities for generating function nodes.
            jax.Array(shape=(sizeof(Function),), dtype=jnp.float32)
            The cumulative distribution function (CDF) of `utils.enum.Funcs` indicating the probabilities of all
            corresponding function types. Note that the last value must be 1.

        const_samples: samples for generating constant nodes. dtype=jnp.float32
            The array as a sample set from desired constant value distribution.
            For example:
                jnp.array([-1.0, 0.0, 1.0]); for discrete constants
                jax.random.normal(key, (100,)); for normal distribution constants

        pop_size: int, the size of population

        max_len: int, the maximum length of trees

        num_inputs: int, the number of input variables for a gp tree

        num_outputs: int, the number of output variables for a gp tree

        output_prob: float, The chance that a function node in a generated GP tree is selected to be an output node.
            Not used when `output_len <= 1`.

        const_prob: float, The chance that a leaf node in a generated GP tree is selected to be a constant node
            rather than a variable node.

    output:
        prefixTrees, that is, GP Trees in CUDA.
        Can use `src.cuda.utils.from_cuda_node` to convert it to JAX Array.
    """
    return gp_generate_(
        key,
        leaf_prob,
        funcs_prob_acc,
        const_samples,
        pop_size=pop_size,
        max_len=max_len,
        variable_len=num_inputs,
        output_len=num_outputs,
        output_prob=output_prob,
        const_prob=const_prob
    )


@jax.jit
def crossover(prefixTrees, left, right, nodes):
    """
    Crossover a population of Trees with given permutations and node indices.
    The new Trees of index `i` is the input GP of index **`left[i]`** while its subtree identified by root node
    `[left[i], left_right_node_indices[i, 0]]` is fully replaced by the one as in
    `[right[i], left_right_node_indices[i, 1]]`.

    args:
        prefixTrees: GP Trees in CUDA.
        left: `jax.Array(shape=(pop_size, ), dtype=jnp.int32)`
            Indices of left parents.
        right: `jax.Array(shape=(pop_size, ), dtype=jnp.int32)`
            Indices of right parents.
        nodes: `jax.Array(shape=(pop_size, 2), dtype=jnp.int16)`
            The left and right node indices indicating the subtrees' locations

    output:
        prefixTrees, that is, GP Trees in CUDA.
    """
    return gp_crossover_(prefixTrees, left, right, nodes)


@jax.jit
def mutation(prefixTrees, index, new_sub_trees):
    """
    Mutates a population of Trees with given node indices.
    The new tree of index `i` is the input tree of index `i` while its subtree identified by root node
    `[i, node_indices[i]]` is fully replaced by the one as in `new_sub_trees[i]`.

    args:
        prefixTrees: GP Trees in CUDA.
        index: jax.Array(shape=(pop_size, ), dtype= jnp.int16)
            The node indices indicating the subtrees' locations
        new_trees: PrefixTrees.

    output:
        prefixGPs, that is, GP Trees in CUDA.
    """

    return gp_mutation_(prefixTrees, index, new_sub_trees)


@partial(jax.jit, static_argnames=["result_length"])
def forward(prefixTrees, variables, result_length=1):
    """
    Inference a population of GPs with corresponding population of input variables.

    args:
        prefixTrees: GP Trees in CUDA.
        variables: jax.Array(shape=(pop_size, num_inputs), dtype=(jnp.float32 | jnp.float64))
            The input variables for each GP.
        result_length: int,
            The length of each result of GP. Value larger than 1 implies that a multi-node output GP will be used instead of root-node output GP.

    output:
        result: jax.Array(shape=(pop_size, result_length))
            The result of each GP in the population.

    """
    return gp_eval_(prefixTrees, variables, result_length=result_length)


@jax.jit
def sr_fitness(prefixTrees, data_points, targets):
    """
    Evaluating the fitness values in Symbolic Regression (SR) for a population of (possibly different) GPs with
    given data points.

    args:
        prefixTrees: GP Trees in CUDA.
    data_points : jax.Array(shape=(data_size, var_len), dtype=(jnp.float32 | jnp.float64))
        The corresponding data point arrays, must be of same dtype as `prefixGPs`.
    targets : jax.Array(shape=(data_size, ), dtype=(jnp.float32 | jnp.float64))
        The corresponding target values, must be of same dtype as `prefixGPs`.

    output:
        fitness: jax.Array(shape=(pop_size, ))
            The fitness values of each GP in the population.

    """
    return gp_sr_fitness_(prefixTrees, data_points, targets, use_MSE=True)
