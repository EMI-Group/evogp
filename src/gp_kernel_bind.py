from functools import partial

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

from kernel.build import gpu_ops


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


def gp_eval(
    prefixGPs: jax.Array,
    variables: jax.Array,
) -> jax.Array:
    """
    The (forward) function for evaluating (inference) a population of (possibly different) GPs with corresponding population of input variables in parallel using CUDA.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len), dtype=gpdefs.get_GPNode_type(jnp.float32 | jnp.float64))`
        The prefix arrays of all GPs in the population, storing all functions (see [kernel.gpdefs.Function](src/kernel/gpdefs.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `variables` : `jax.Array(shape=(pop_size, var_len), dtype=(jnp.bfloat16 | jnp.float16 | jnp.float32 | jnp.float64))`
        The corresponding variable arrays of all GPs in the population, must be of same dtype as `prefixGPs`

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A `jax.Array(shape=pop_size, dtype=variables.dtype)` as the evaluation results of all the GPs in the population.
    """
    results = _gp_eval_fwd_p.bind(prefixGPs, variables)
    return results


def gp_crossover(
    prefixGPs: jax.Array,
    left_perms: jax.Array,
    right_perms: jax.Array,
    left_right_node_indices: jax.Array,
) -> jax.Array:
    """
    The (forward) function for crossover a population of GPs with given permutations and node indices in parallel using CUDA.
    The new GP of index `i` is the input GP of index `left_perms[i]` while its subtree identified by root node `gp[left_perms[i], left_right_node_indices[i, 0]]` is fully replaced by the one as in `gp[right_perms[i], left_right_node_indices[i, 1]]`.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len), dtype=gpdefs.get_GPNode_type(jnp.float32 | jnp.float64))`
        The prefix arrays of all GPs in the population, storing all functions (see [kernel.gpdefs.Function](src/kernel/gpdefs.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `left_perms` : `jax.Array(shape=(pop_size, ), dtype=(jnp.uint32 | jnp.int32))`
        The left permutation to identify the output order of GP to be crossover

    `right_perms` : `jax.Array(shape=(pop_size, ), dtype=(jnp.uint32 | jnp.int32))`
        The right permutation to identify the output order of GP to crossover

    `left_right_node_indices` : `jax.Array(shape=(pop_size, 2), dtype=(jnp.uint16 | jnp.int16))`
        The left and right node indices indicating the subtrees' locations

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A new array with same sizes and types representing the new population of GPs.
    """
    results = _gp_crossover_fwd_p.bind(
        prefixGPs, left_perms, right_perms, left_right_node_indices
    )
    return results


def gp_mutation(
    prefixGPs: jax.Array,
    node_indices: jax.Array,
    new_subtree_prefixGPs: jax.Array,
) -> jax.Array:
    """
    The (forward) function for mutating a population of GPs with given node indices in parallel using CUDA.
    The new GP of index `i` is the input GP of index `i` while its subtree identified by root node `gp[i, node_indices[i]]` is fully replaced by the one as in `new_subtree[i]`.

    Parameters
    ----------
    `prefixGPs` : `jax.Array(shape=(pop_size, max_len), dtype=gpdefs.get_GPNode_type(jnp.float32 | jnp.float64))`
        The prefix arrays of all GPs in the population, storing all functions (see [kernel.gpdefs.Function](src/kernel/gpdefs.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    `node_indices` : `jax.Array(shape=(pop_size, 2), dtype=(jnp.uint16 | jnp.int16))`
        The node indices indicating the subtrees' locations

    `new_subtree_prefixGPs` : `jax.Array(shape=(pop_size, max_new_len), dtype=gpdefs.get_GPNode_type(jnp.float32 | jnp.float64))`
        The prefix arrays of all new subtree GPs in the population, storing all functions (see [kernel.gpdefs.Function](src/kernel/gpdefs.py)), constants and variable indices as floating point type and corresponding node types and subtree sizes

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A new array with same sizes and types representing the new population of GPs.
    """
    results = _gp_mutation_fwd_p.bind(
        prefixGPs,
        node_indices,
        new_subtree_prefixGPs,
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


def _gp_eval_fwd_cuda_lowering(ctx, prefixGPs, variables):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    var_info = ir.RankedTensorType(variables.type)

    opaque = gpu_ops.create_gp_descriptor(
        gp_info.shape[0],
        gp_info.shape[1],
        var_info.shape[1],
        element_type_to_descriptor_type(var_info.element_type),
    )
    out = custom_call(
        b"gp_eval_forward",
        out_types=[
            ir.RankedTensorType.get((var_info.shape[0],), var_info.element_type),
        ],
        operands=[prefixGPs, variables],
        backend_config=opaque,
        operand_layouts=default_layouts(gp_info.shape, var_info.shape),
        result_layouts=default_layouts((gp_info.shape[0],)),
    )
    return out


def _gp_crossover_fwd_cuda_lowering(
    ctx, prefixGPs, left_perms, right_perms, left_right_node_idx
):
    gp_info = ir.RankedTensorType(prefixGPs.type)
    lp_info = ir.RankedTensorType(left_perms.type)
    rp_info = ir.RankedTensorType(right_perms.type)
    lr_info = ir.RankedTensorType(left_right_node_idx.type)

    opaque = gpu_ops.create_gp_descriptor(
        gp_info.shape[0],
        gp_info.shape[1],
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
        operand_layouts=default_layouts(
            gp_info.shape, lp_info.shape, rp_info.shape, lr_info.shape
        ),
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
        element_type_to_descriptor_type(gp_info.element_type),
    )
    out = custom_call(
        b"gp_mutation_forward",
        out_types=[
            ir.RankedTensorType.get(gp_info.shape, gp_info.element_type),
        ],
        operands=[
            prefixGPs,
            node_indices,
            new_subtree_prefixGPs
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            gp_info.shape, ii_info.shape, sp_info.shape
        ),
        result_layouts=default_layouts(gp_info.shape),
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


#######################
# Abstract evaluation #
#######################


def _gp_eval_fwd_abstract(prefixGPs, variables):
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    var_type = dtypes.canonicalize_dtype(variables.dtype)
    var_shape = variables.shape

    assert len(gp_shape) == 2
    assert len(var_shape) == 2
    assert gp_shape[0] == var_shape[0]
    assert gp_type in [jnp.complex64, jnp.complex128]

    return ShapedArray((gp_shape[0],), var_type, named_shape=variables.named_shape)


def _gp_crossover_fwd_abstract(
    prefixGPs, left_perms, right_perms, left_right_node_idx
):
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
    assert (
        pop_size == lp_shape[0]
        and pop_size == rp_shape[0]
        and pop_size == lr_shape[0]
    )

    assert gp_type in [jnp.complex64, jnp.complex128]
    assert lp_type in [jnp.uint32, jnp.int32] and rp_type in [jnp.uint32, jnp.int32]
    assert lr_type in [jnp.uint16, jnp.int16]

    return ShapedArray(gp_shape, gp_type, named_shape=prefixGPs.named_shape)


def _gp_mutation_fwd_abstract(
    prefixGPs, node_indices, new_subtree_prefixGPs
):
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
    assert (
        pop_size == ii_shape[0]
        and pop_size == sp_shape[0]
    )

    assert gp_type in [jnp.complex64, jnp.complex128]
    assert gp_type == sp_type

    assert ii_type in [jnp.uint32, jnp.int32]

    return ShapedArray(gp_shape, gp_type, named_shape=prefixGPs.named_shape)


_gp_eval_fwd_p.def_abstract_eval(_gp_eval_fwd_abstract)
_gp_crossover_fwd_p.def_abstract_eval(_gp_crossover_fwd_abstract)
_gp_mutation_fwd_p.def_abstract_eval(_gp_mutation_fwd_abstract)
