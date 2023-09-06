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


def gp_eval(
    prefixGP_lengths: jax.Array,
    node_types: jax.Array,
    prefixGPs: jax.Array,
    variables: jax.Array,
):
    """
    The (forward) function for evaluating (inference) a population of (possibly different) GPs with corresponding population of input variables in parallel using CUDA.

    Parameters
    ----------
    `prefixGP_lengths` : `jax.Array(shape=pop_size, dtype=(jnp.int32 | jnp.uint32))`
        The lengths (number of nodes) of all GPs in the population

    `node_types` : `jax.Array(shape=(pop_size, max_len), dtype=(jnp.int8 | jnp.uint8))`
        The prefix node type arrays of all GPs in the population, see [kernel.gpdefs.NodeType](src/kernel/gpdefs.py)

    `prefixGPs` : `jax.Array(shape=(pop_size, max_len), dtype=(jnp.bfloat16 | jnp.float16 | jnp.float32 | jnp.float64))`
        The prefix arrays of all GPs in the population, storing all functions (see [kernel.gpdefs.Function](src/kernel/gpdefs.py)), constants and variable indices as floating point type

    `variables` : `jax.Array(shape=(pop_size, var_len), dtype=(jnp.bfloat16 | jnp.float16 | jnp.float32 | jnp.float64))`
        The corresponding variable arrays of all GPs in the population, must be of same dtype as `prefixGPs`

    Raises
    ------
    AssertionError
        If the dtypes or shapes are inconsistent

    Return
    ------
    A `jax.Array(shape=pop_size, dtype=variables.dtype)` as the evaluation results of all the GPs in the population.

    Usage
    -----
    ```
    import jax
    import jax.numpy as jnp
    from kernel.gpdefs import *
    from gp_eval_bind import gp_eval

    # predefined GP with M nodes
    prefixGP_maxlen = 1024
    node_type = [
        # Your GP node types here
    ]
    prefixGP = [
        # Your GP in prefix here
    ]
    node_type = [node_type[i] if i < prefixGP_len else 0 for i in range(prefixGP_maxlen)]
    prefixGP = [prefixGP[i] if i < prefixGP_len else 0 for i in range(prefixGP_maxlen)]
    variable = [1, 2]

    # replicate to test
    N = 1_000_000
    prefixGP_lengths = M * jnp.ones((N,), dtype=jnp.uint32)
    node_types = jnp.tile(jnp.asarray(node_type, dtype=jnp.uint8), [N, 1])
    prefixGPs = jnp.tile(jnp.asarray(prefixGP, dtype=jnp.float32), [N, 1])
    variables = jnp.tile(jnp.asarray(variable, dtype=jnp.float32), [N, 1])

    # execute
    results = jax.jit(gp_eval)(prefixGP_lengths, node_types, prefixGPs, variables)
    ```
    """
    results = _gp_eval_fwd_p.bind(prefixGP_lengths, node_types, prefixGPs, variables)
    return results


####################
# Lowering to MLIR #
####################

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.get_gp_eval_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _gp_eval_fwd_cuda_lowering(ctx, prefixGP_lengths, node_types, prefixGPs, variables):
    len_info = ir.RankedTensorType(prefixGP_lengths.type)
    nt_info = ir.RankedTensorType(node_types.type)
    gp_info = ir.RankedTensorType(prefixGPs.type)
    var_info = ir.RankedTensorType(variables.type)

    opaque = gpu_ops.create_gp_eval_descriptor(
        len_info.shape[0],
        nt_info.shape[1],
        var_info.shape[1],
        element_type_to_descriptor_type_mapping(var_info.element_type),
    )
    out = custom_call(
        b"gp_eval_forward",
        out_types=[
            ir.RankedTensorType.get(len_info.shape, var_info.element_type),
        ],
        operands=[prefixGP_lengths, node_types, prefixGPs, variables],
        backend_config=opaque,
        operand_layouts=default_layouts(
            len_info.shape, nt_info.shape, gp_info.shape, var_info.shape
        ),
        result_layouts=default_layouts(len_info.shape),
    )
    return out


mlir.register_lowering(
    _gp_eval_fwd_p,
    _gp_eval_fwd_cuda_lowering,
    platform="gpu",
)


#######################
# Abstract evaluation #
#######################


def _gp_eval_fwd_abstract(prefixGP_lengths, node_types, prefixGPs, variables):
    len_type = dtypes.canonicalize_dtype(prefixGP_lengths.dtype)
    len_shape = prefixGP_lengths.shape
    nt_type = dtypes.canonicalize_dtype(node_types.dtype)
    nt_shape = node_types.shape
    gp_type = dtypes.canonicalize_dtype(prefixGPs.dtype)
    gp_shape = prefixGPs.shape
    var_type = dtypes.canonicalize_dtype(variables.dtype)
    var_shape = variables.shape

    assert len(len_shape) == 1
    assert len(nt_shape) == 2
    assert len(gp_shape) == 2
    assert len(var_shape) == 2
    popSize = len_shape[0]
    assert popSize == nt_shape[0] and popSize == gp_shape[0] and popSize == var_shape[0]
    gp_max_len = nt_shape[1]
    assert gp_max_len == gp_shape[1]

    assert len_type in [jnp.uint32, jnp.int32]
    assert nt_type in [jnp.uint8, jnp.int8]
    assert gp_type == var_type

    return ShapedArray(len_shape, var_type, named_shape=prefixGP_lengths.named_shape)


_gp_eval_fwd_p.def_abstract_eval(_gp_eval_fwd_abstract)
