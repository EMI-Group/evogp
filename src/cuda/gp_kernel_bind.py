from functools import partial

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

from .build import gpu_ops


# Create _gp_eval_fwd_p for forward operation.
_gp_eval_fwd_p = core.Primitive("gp_eval_fwd")
_gp_eval_fwd_p.multiple_results = False
_gp_eval_fwd_p.def_impl(partial(xla.apply_primitive, _gp_eval_fwd_p))

_gp_crossover_fwd_p = core.Primitive("gp_crossover_fwd")
_gp_crossover_fwd_p.multiple_results = False
_gp_crossover_fwd_p.def_impl(partial(xla.apply_primitive, _gp_crossover_fwd_p))

_gp_mutation_fwd_p = core.Primitive("gp_mutation_fwd")
_gp_mutation_fwd_p.multiple_results = False
_gp_mutation_fwd_p.def_impl(partial(xla.apply_primitive, _gp_mutation_fwd_p))

_gp_sr_fitness_fwd_p = core.Primitive("gp_sr_fitness_fwd")
_gp_sr_fitness_fwd_p.multiple_results = False
_gp_sr_fitness_fwd_p.def_impl(partial(xla.apply_primitive, _gp_sr_fitness_fwd_p))

_constant_gp_sr_fitness_fwd_p = core.Primitive("constant_gp_sr_fitness_fwd")
_constant_gp_sr_fitness_fwd_p.multiple_results = True
_constant_gp_sr_fitness_fwd_p.def_impl(partial(xla.apply_primitive, _constant_gp_sr_fitness_fwd_p))

_gp_generate_fwd_p = core.Primitive("gp_generate_fwd")
_gp_generate_fwd_p.multiple_results = False
_gp_generate_fwd_p.def_impl(partial(xla.apply_primitive, _gp_generate_fwd_p))


def gp_eval_(prefixGPs: jax.Array, variables: jax.Array, result_length: int = 1) -> jax.Array:
    """
    The (forward) function for evaluating (inference) a population of (possibly different) GPs with corresponding population of input variables in parallel using CUDA.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len))`
        The prefix arrays of all GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `variables` : `jax.Array(shape=(pop_size, var_len), dtype=(jnp.float32 | jnp.float64))`
        The corresponding variable arrays of all GPs in the population, must be of same dtype as `prefixGPs`

    `result_length` : `int`
        The length of each result of GP. Value larger than 1 implies that a multi-node output GP will be used instead of root-node output GP.

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A `jax.Array(shape=pop_size, dtype=variables.dtype)` as the evaluation results of all the GPs in the population.

    Note that none of the array contents (except the multi-output indices) are checked for performance issues. Hence, make sure that the max stack size do not exceed [MAX_STACK](src/cuda/gpdefs.h).
    """
    results = _gp_eval_fwd_p.bind(prefixGPs, variables, result_length=result_length)
    return results


def gp_crossover_(
    prefixGPs: jax.Array,
    left_perms: jax.Array,
    right_perms: jax.Array,
    left_right_node_indices: jax.Array,
) -> jax.Array:
    """
    The (forward) function for crossover a population of GPs with given permutations and node indices in parallel using CUDA.
    The new GP of index `i` is the input GP of index `left_perms[i]` while its subtree identified by root node `algorithm[left_perms[i], left_right_node_indices[i, 0]]` is fully replaced by the one as in `algorithm[right_perms[i], left_right_node_indices[i, 1]]`.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len))`
        The prefix arrays of all GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `left_perms` : `jax.Array(shape=(pop_size, ), dtype=jnp.int32)`
        The left permutation to identify the output order of GP to be crossover

    `right_perms` : `jax.Array(shape=(pop_size, ), dtype=jnp.int32)`
        The right permutation to identify the output order of GP to crossover

    `left_right_node_indices` : `jax.Array(shape=(pop_size, 2), dtype=jnp.int16)`
        The left and right node indices indicating the subtrees' locations

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A new array with same sizes and types representing the new population of GPs.

    Note that the `left_perms` are not checked and assumed to be within the range of [0, `pop_size`), while output GP(s) with invalid `right_perms` is/are direct copies of `algorithm[left_perms[i]]`. Besides, any crossover(s) that may cause size overflow is/are ignored as well.
    """
    results = _gp_crossover_fwd_p.bind(prefixGPs, left_perms, right_perms, left_right_node_indices)
    return results


def gp_mutation_(
    prefixGPs: jax.Array,
    node_indices: jax.Array,
    new_subtree_prefixGPs: jax.Array,
) -> jax.Array:
    """
    The (forward) function for mutating a population of GPs with given node indices in parallel using CUDA.
    The new GP of index `i` is the input GP of index `i` while its subtree identified by root node `algorithm[i, node_indices[i]]` is fully replaced by the one as in `new_subtree[i]`.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len))`
        The prefix arrays of all GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `node_indices` : `jax.Array(shape=(pop_size, 2), dtype= jnp.int16)`
        The node indices indicating the subtrees' locations

    `new_subtree_prefixGPs` : `jax.Array(shape=(pop_size, max_new_len))`
        The prefix arrays of all new subtree GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A new array with same sizes and types representing the new population of GPs.

    Note that the output GP(s) with invalid `node_indices` is/are direct copies of `algorithm[i]`. Besides, any mutation(s) that may cause size overflow is/are ignored as well.
    """
    results = _gp_mutation_fwd_p.bind(
        prefixGPs,
        node_indices,
        new_subtree_prefixGPs,
    )
    return results


def gp_sr_fitness_(
    prefixGPs: jax.Array,
    data_points: jax.Array,
    targets: jax.Array,
    use_MSE: bool = True,
) -> jax.Array:
    """
    The (forward) function for evaluating the fitness values in Symbolic Regression (SR) for a population of (possibly different) GPs with given data points in parallel using CUDA.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len))`
        The prefix arrays of all GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `data_points` : `jax.Array(shape=(data_size, var_len), dtype=(jnp.float32 | jnp.float64))`
        The corresponding data point arrays, must be of same dtype as `prefixGPs`

    `targets` : `jax.Array(shape=(data_size, var_len), dtype=(jnp.float32 | jnp.float64))`
        The target (label) array corresponding to the input `data_points`. Must be of same type as `data_points`.

    `use_MSE` : `bool`
        Whether the output fitness values are Mean Squared Errors (MSE) or Mean Absolute Errors (MAE).

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A `jax.Array(shape=data_size, dtype=targets.dtype)` as the fitness values of all the GPs in the population.

    Note that none of the array contents (except the multi-output indices) are checked for performance issues. Hence, make sure that the max stack size do not exceed [MAX_STACK](src/cuda/gpdefs.h).
    """
    results = _gp_sr_fitness_fwd_p.bind(prefixGPs, data_points, targets, use_MSE=use_MSE)
    return results


def constant_gp_sr_fitness_(
    prefixGPs: jax.Array,
    data_points: jax.Array,
    targets: jax.Array,
    use_MSE: bool = True,
) -> jax.Array:
    """
    Similar to `gp_sr_fitness_`, but the fitness values are calculated with the assumption that the GPs are stored in constant memory.
    The (forward) function for evaluating the fitness values in Symbolic Regression (SR) for a population of (possibly different) GPs with given data points in parallel using CUDA.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len))`
        The prefix arrays of all GPs in the population, storing all functions (see [cuda.gpdefs.Function](src/cuda/utils.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `data_points` : `jax.Array(shape=(data_size, var_len), dtype=(jnp.float32 | jnp.float64))`
        The corresponding data point arrays, must be of same dtype as `prefixGPs`

    `targets` : `jax.Array(shape=(data_size, var_len), dtype=(jnp.float32 | jnp.float64))`
        The target (label) array corresponding to the input `data_points`. Must be of same type as `data_points`.

    `use_MSE` : `bool`
        Whether the output fitness values are Mean Squared Errors (MSE) or Mean Absolute Errors (MAE).

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A `jax.Array(shape=data_size, dtype=targets.dtype)` as the fitness values of all the GPs in the population.

    Note that none of the array contents (except the multi-output indices) are checked for performance issues. Hence, make sure that the max stack size do not exceed [MAX_STACK](src/cuda/gpdefs.h).
    """
    results = _constant_gp_sr_fitness_fwd_p.bind(prefixGPs, data_points, targets, use_MSE=use_MSE)
    return results


def gp_generate_(
    key: jax.Array,
    depth_to_leaf_prob: jax.Array,
    functions_prob_accumulate: jax.Array,
    const_samples: jax.Array,
    pop_size: int,
    max_len: int,
    variable_len: int,
    output_len: int = 1,
    output_prob: float = 0.5,
    const_prob: float = 0.5,
    random_generator: gpu_ops.RandomEngine = gpu_ops.RandomEngine.Default,
) -> jax.Array:
    """
    The (forward) function generates a population of GPs via given sizes and probabilities.

    Parameters
    ----------
    `depth_to_leaf_prob` : `jax.Array(shape=(log2(MAX_LENGTH),), dtype=jnp.float32)`
        The array indicating the probabilities of generating leaf nodes. The GP nodes of depth `i` have probability `depth_to_leaf_prob[i]` to be a leaf node. Note that no checks will be performed regarding this array. Hence, if these probabilities allow GP tree size larger than `MAX_LENGTH`, there may be unexpected outcomes.

    `functions_prob_accumulate` : `jax.Array(shape=(sizeof(Function),), dtype=jnp.float32)`
        The cumulative distribution function (CDF) of `gpdefs.Function` indicating the probabilities of all corresponding function types. Note that the last value must be 1.

    `const_samples` : `jax.Array`
        The array as a sample set from desired constant value distribution. For example, if the desired constant values are limited to `[-1, 0, 1]`, `const_samples` shall be `[-1, 0, 1]`; if the desired constant value is a standard normal distribution, `const_samples` shall be `jax.random.normal(key, desired_shape)` where `desired_shape` can be arbitrarily large.

    `pop_size` : `int`
        The desired output population size.

    `max_prefix_len` : `int`
        The maximum possible length of an generated GP tree.

    `variable_len` : `int`
        The number of input variables to be given to the generated GP trees. Currently, these variables are equiprobably chosen.

    `output_len` : `int`
        The number of output variables that the generated GP trees shall give. Currently, these outputs are equiprobably assigned.

    `output_prob` : `float`
        The chance that a function node in a generated GP tree is selected to be an output node. Not used when `output_len <= 1`.

    `const_prob` : `float`
        The chance that a leaf node in a generated GP tree is selected to be a constant node rather than a variable node.

    `key` : `jax.random.PRNGKey`
        The key controlling the random outcomes.

    `random_generator` : `gpu_ops.RandomEngine`
        The random number generator type. The possible values are `Default`, `RANLUX24`, `RANLUX48` and `TAUS88`.

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A new array with same sizes and types representing the new population of GPs.

    Note that the output GP(s) with invalid `node_indices` is/are direct copies of `algorithm[i]`. Besides, any mutation(s) that may cause size overflow is/are ignored as well.
    """
    results = _gp_generate_fwd_p.bind(
        key,
        depth_to_leaf_prob,
        functions_prob_accumulate,
        const_samples,
        pop_size=pop_size,
        max_prefix_len=max_len,
        variable_len=variable_len,
        output_len=output_len,
        output_prob=output_prob,
        const_prob=const_prob,
        random_generator=random_generator,
    )
    return results


####################
# Lowering to MLIR #
####################

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.get_gp_eval_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
for _name, _value in gpu_ops.get_gp_crossover_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
for _name, _value in gpu_ops.get_gp_mutation_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
for _name, _value in gpu_ops.get_gp_sr_fitness_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
for _name, _value in gpu_ops.get_constant_gp_sr_fitness_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
for _name, _value in gpu_ops.get_gp_generate_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type(element_type):
    _mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
        ir.ComplexType.get(ir.F32Type.get()): gpu_ops.ElementType.F32,
        ir.ComplexType.get(ir.F64Type.get()): gpu_ops.ElementType.F64,
    }
    return _mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _gp_eval_fwd_cuda_lowering(ctx, prefixGPs, variables, result_length):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    var_info = ir.RankedTensorType(variables.type)

    opaque = gpu_ops.create_gp_descriptor(
        gp_info.shape[0],
        gp_info.shape[1],
        var_info.shape[1],
        result_length,
        element_type_to_descriptor_type(var_info.element_type),
    )
    out_shape = (var_info.shape[0],) if result_length == 1 else (var_info.shape[0], result_length)
    out = custom_call(
        b"gp_eval_forward",
        out_types=[
            ir.RankedTensorType.get(out_shape, var_info.element_type),
        ],
        operands=[prefixGPs, variables],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, var_info.shape),
        result_layouts=default_layouts(out_shape),
    )
    return out


def _gp_crossover_fwd_cuda_lowering(ctx, prefixGPs, left_perms, right_perms, left_right_node_idx):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    lp_info = ir.RankedTensorType(left_perms.type)
    rp_info = ir.RankedTensorType(right_perms.type)
    lr_info = ir.RankedTensorType(left_right_node_idx.type)

    opaque = gpu_ops.create_gp_descriptor(
        gp_info.shape[0],
        gp_info.shape[1],
        0,
        0,  # not used in this function
        element_type_to_descriptor_type(gp_info.element_type),
    )
    out = custom_call(
        b"gp_crossover_forward",
        out_types=[
            ir.RankedTensorType.get(gp_info.shape, gp_info.element_type),
        ],
        operands=[prefixGPs, left_perms, right_perms, left_right_node_idx],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, lp_info.shape, rp_info.shape, lr_info.shape),
        result_layouts=default_layouts(gp_info.shape),
    )
    return out


def _gp_mutation_fwd_cuda_lowering(
    ctx,
    prefixGPs,
    node_indices,
    new_subtree_prefixGPs,
):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    ii_info = ir.RankedTensorType(node_indices.type)
    sp_info = ir.RankedTensorType(new_subtree_prefixGPs.type)

    opaque = gpu_ops.create_gp_descriptor(
        gp_info.shape[0],
        gp_info.shape[1],
        sp_info.shape[1],
        0,  # not used in this function
        element_type_to_descriptor_type(gp_info.element_type),
    )
    out = custom_call(
        b"gp_mutation_forward",
        out_types=[
            ir.RankedTensorType.get(gp_info.shape, gp_info.element_type),
        ],
        operands=[prefixGPs, node_indices, new_subtree_prefixGPs],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, ii_info.shape, sp_info.shape),
        result_layouts=default_layouts(gp_info.shape),
    )
    return out


def _gp_sr_fitness_fwd_cuda_lowering(ctx, prefixGPs, data_points, targets, use_MSE):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    dp_info = ir.RankedTensorType(data_points.type)
    t_info = ir.RankedTensorType(targets.type)

    opaque = gpu_ops.create_gp_sr_descriptor(
        gp_info.shape[0],
        dp_info.shape[0],
        gp_info.shape[1],
        dp_info.shape[1],
        0 if len(t_info.shape) == 1 or t_info.shape[1] == 1 else t_info.shape[1],
        element_type_to_descriptor_type(t_info.element_type),
        use_MSE,
    )
    out_shape = (gp_info.shape[0],)
    out = custom_call(
        b"gp_sr_fitness_forward",
        out_types=[
            ir.RankedTensorType.get(out_shape, t_info.element_type),
        ],
        operands=[prefixGPs, data_points, targets],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, dp_info.shape, t_info.shape),
        result_layouts=default_layouts(out_shape),
    )
    return out


# This seems to be important.
def _constant_gp_sr_fitness_fwd_cuda_lowering(ctx, prefixGPs, data_points, targets, use_MSE):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    dp_info = ir.RankedTensorType(data_points.type)
    t_info = ir.RankedTensorType(targets.type)

    opaque = gpu_ops.create_gp_sr_descriptor(
        gp_info.shape[0],  # pop_size
        dp_info.shape[0],  # dataPoints
        gp_info.shape[1],  # gpLen
        dp_info.shape[1],  # varLen
        0 if len(t_info.shape) == 1 or t_info.shape[1] == 1 else t_info.shape[1],  # outLen
        element_type_to_descriptor_type(t_info.element_type),  # type
        use_MSE,  # useMSE
    )

    SR_BLOCK_SIZE = 1024
    block_fitness_space_size = (dp_info.shape[0] - 1) // SR_BLOCK_SIZE + 1
    block_fitness_space_shape = (block_fitness_space_size,)

    out_shape = (gp_info.shape[0],)
    out = custom_call(
        b"constant_gp_sr_fitness_forward",
        out_types=[
            ir.RankedTensorType.get(out_shape, t_info.element_type),
            ir.RankedTensorType.get(block_fitness_space_shape, t_info.element_type),
        ],
        operands=[prefixGPs, data_points, targets],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, dp_info.shape, t_info.shape),
        result_layouts=default_layouts(out_shape, block_fitness_space_shape),
    )
    return out


def _gp_generate_fwd_cuda_lowering(
    ctx,
    key,
    depth_to_leaf_prob,
    functions_prob_accumulate,
    const_samples,
    pop_size,
    max_prefix_len,
    variable_len,
    output_len,
    output_prob,
    const_prob,
    random_generator,
):
    key_info = ir.RankedTensorType(key.type)
    d2l_info = ir.RankedTensorType(depth_to_leaf_prob.type)
    fp_info = ir.RankedTensorType(functions_prob_accumulate.type)
    cs_info = ir.RankedTensorType(const_samples.type)

    opaque = gpu_ops.create_gp_generate_descriptor(
        pop_size,
        max_prefix_len,
        variable_len,
        output_len,
        cs_info.shape[0],
        output_prob,
        const_prob,
        random_generator,
        element_type_to_descriptor_type(cs_info.element_type),
    )
    out_shape = (pop_size, max_prefix_len)
    out_type = (
        ir.ComplexType.get(ir.F32Type.get())
        if cs_info.element_type == ir.F32Type.get()
        else ir.ComplexType.get(ir.F64Type.get())
    )
    out = custom_call(
        b"gp_generate_forward",
        out_types=[
            ir.RankedTensorType.get(out_shape, out_type),
        ],
        operands=[key, depth_to_leaf_prob, functions_prob_accumulate, const_samples],
        backend_config=opaque,
        operand_layouts=default_layouts(key_info.shape, d2l_info.shape, fp_info.shape, cs_info.shape),
        result_layouts=default_layouts(out_shape),
    )
    return out


mlir.register_lowering(
    _gp_eval_fwd_p,
    _gp_eval_fwd_cuda_lowering,
    platform="gpu",
)

mlir.register_lowering(
    _gp_crossover_fwd_p,
    _gp_crossover_fwd_cuda_lowering,
    platform="gpu",
)

mlir.register_lowering(
    _gp_mutation_fwd_p,
    _gp_mutation_fwd_cuda_lowering,
    platform="gpu",
)

mlir.register_lowering(
    _gp_sr_fitness_fwd_p,
    _gp_sr_fitness_fwd_cuda_lowering,
    platform="gpu",
)

mlir.register_lowering(
    _constant_gp_sr_fitness_fwd_p,
    _constant_gp_sr_fitness_fwd_cuda_lowering,
    platform="gpu",
)

mlir.register_lowering(
    _gp_generate_fwd_p,
    _gp_generate_fwd_cuda_lowering,
    platform="gpu",
)


#######################
# Abstract evaluation #
#######################


def _gp_eval_fwd_abstract(prefixGPs, variables, result_length):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    var_type = dtypes.canonicalize_dtype(variables.dtype)
    var_shape = variables.shape

    assert result_length >= 1
    assert len(gp_shape) == 2
    assert len(var_shape) == 2
    assert gp_shape[0] == var_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]

    return ShapedArray(
        (gp_shape[0],) if result_length == 1 else (gp_shape[0], result_length),
        var_type,
        named_shape=variables.named_shape,
    )


def _gp_crossover_fwd_abstract(prefixGPs, left_perms, right_perms, left_right_node_idx):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    lp_type = dtypes.canonicalize_dtype(left_perms.dtype)
    lp_shape = left_perms.shape
    rp_type = dtypes.canonicalize_dtype(right_perms.dtype)
    rp_shape = right_perms.shape
    lr_type = dtypes.canonicalize_dtype(left_right_node_idx.dtype)
    lr_shape = left_right_node_idx.shape

    assert len(gp_shape) == 2
    assert len(lp_shape) == 1 and len(rp_shape) == 1
    assert len(lr_shape) == 2 and lr_shape[1] == 2
    pop_size = gp_shape[0]
    assert pop_size == lp_shape[0] and pop_size == rp_shape[0] and pop_size == lr_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]
    assert lp_type in [jnp.int32, jnp.int32] and rp_type in [jnp.int32, jnp.int32]
    assert lr_type in [jnp.int16, jnp.uint16]

    return ShapedArray(gp_shape, gp_type, named_shape=prefixGPs.named_shape)


def _gp_mutation_fwd_abstract(prefixGPs, node_indices, new_subtree_prefixGPs):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    ii_type = dtypes.canonicalize_dtype(node_indices.dtype)
    ii_shape = node_indices.shape
    sp_type = dtypes.canonicalize_dtype(new_subtree_prefixGPs.dtype)
    sp_shape = new_subtree_prefixGPs.shape

    assert len(gp_shape) == 2
    assert len(sp_shape) == 2
    assert len(ii_shape) == 1
    pop_size = gp_shape[0]
    assert pop_size == ii_shape[0] and pop_size == sp_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]
    assert gp_type == sp_type

    assert ii_type in [jnp.int32, jnp.int32]

    return ShapedArray(gp_shape, gp_type, named_shape=prefixGPs.named_shape)


def _gp_sr_fitness_fwd_abstract(prefixGPs, data_points, targets, use_MSE=False):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    dp_type = dtypes.canonicalize_dtype(data_points.dtype)
    dp_shape = data_points.shape
    t_type = dtypes.canonicalize_dtype(targets.dtype)
    t_shape = targets.shape

    assert len(gp_shape) == 2
    assert len(dp_shape) == 2
    assert len(t_shape) == 1 or len(t_shape) == 2
    assert dp_shape[0] == t_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]
    assert dp_type in [jnp.float32, jnp.float64] and dp_type == t_type
    assert dp_type == jnp.float32 if gp_type == jnp.complex64 else dp_type == jnp.float64

    return ShapedArray(
        (gp_shape[0],),
        t_type,
        named_shape=targets.named_shape,
    )


# important
def _constant_gp_sr_fitness_fwd_abstract(prefixGPs, data_points, targets, use_MSE=False):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    dp_type = dtypes.canonicalize_dtype(data_points.dtype)
    dp_shape = data_points.shape
    t_type = dtypes.canonicalize_dtype(targets.dtype)
    t_shape = targets.shape

    assert len(gp_shape) == 2
    assert len(dp_shape) == 2
    assert len(t_shape) == 1 or len(t_shape) == 2
    assert dp_shape[0] == t_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]
    assert dp_type in [jnp.float32, jnp.float64] and dp_type == t_type
    assert dp_type == jnp.float32 if gp_type == jnp.complex64 else dp_type == jnp.float64

    SR_BLOCK_SIZE = 1024
    block_fitness_space_size = (dp_shape[0] - 1) // SR_BLOCK_SIZE + 1
    block_fitness_space_shape = (block_fitness_space_size,)

    return (
        ShapedArray(
            (gp_shape[0],),  # popsize
            t_type,  # datatype?
            named_shape=targets.named_shape,
        ),
        ShapedArray(
            block_fitness_space_shape,
            t_type,  # datatype?
            named_shape=targets.named_shape,
        ),
    )


def _gp_generate_fwd_abstract(
    key,
    depth_to_leaf_prob,
    functions_prob_accumulate,
    const_samples,
    pop_size: int,
    max_prefix_len: int,
    variable_len: int,
    output_len: int = 1,
    output_prob: float = 0.5,
    const_prob: float = 0.5,
    random_generator: str = "Default",
):
    key_type = dtypes.canonicalize_dtype(key.dtype)
    key_shape = key.shape
    dl_type = dtypes.canonicalize_dtype(depth_to_leaf_prob.dtype)
    dl_shape = depth_to_leaf_prob.shape
    fp_type = dtypes.canonicalize_dtype(functions_prob_accumulate.dtype)
    fp_shape = functions_prob_accumulate.shape
    cs_type = dtypes.canonicalize_dtype(const_samples.dtype)
    cs_shape = const_samples.shape

    assert key_type == jnp.uint32 and len(key_shape) == 1 and key_shape[0] == 2

    assert len(dl_shape) == 1 and len(fp_shape) == 1 and len(cs_shape) == 1
    assert dl_type in [jnp.float32, jnp.float64]
    assert dl_type == fp_type and dl_type == cs_type

    assert pop_size > 0 and max_prefix_len > 0 and variable_len > 0 and output_len >= 0
    if output_len > 1:
        assert output_prob > 0 and output_prob < 1
    assert const_prob >= 0 and const_prob <= 1
    assert random_generator in [
        gpu_ops.RandomEngine.Default,
        gpu_ops.RandomEngine.RANLUX24,
        gpu_ops.RandomEngine.RANLUX48,
        gpu_ops.RandomEngine.TAUS88,
    ]

    return ShapedArray(
        (pop_size, max_prefix_len),
        jnp.complex64 if dl_type == jnp.float32 else jnp.complex128,
        named_shape=const_samples.named_shape,
    )


_gp_eval_fwd_p.def_abstract_eval(_gp_eval_fwd_abstract)
_gp_crossover_fwd_p.def_abstract_eval(_gp_crossover_fwd_abstract)
_gp_mutation_fwd_p.def_abstract_eval(_gp_mutation_fwd_abstract)
_gp_sr_fitness_fwd_p.def_abstract_eval(_gp_sr_fitness_fwd_abstract)
_constant_gp_sr_fitness_fwd_p.def_abstract_eval(_constant_gp_sr_fitness_fwd_abstract)
_gp_generate_fwd_p.def_abstract_eval(_gp_generate_fwd_abstract)
