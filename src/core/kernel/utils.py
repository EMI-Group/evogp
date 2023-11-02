import jax
import jax.numpy as jnp

from src.gp.enum import NType, FUNCS_NAMES


def tree2str(tree):
    return to_string(tree.node_types, tree.node_vals)

def cuda_tree_to_string(tree):
    node_val, node_type, node_size, _ = from_cuda_node(tree)
    return to_string(node_type[:node_size[0]], node_val[:node_size[0]])

def to_string(node_type, node_val):
    node_type, node_val = list(node_type), list(node_val)
    res = ""
    for t, v in zip(node_type, node_val):
        if t == NType.VAR:
            res += f"in[{int(v)}]"
        elif t == NType.CONST:
            res += f"{v}"
        elif t == NType.UFUNC or t == NType.BFUNC or t == NType.TFUNC:
            res += f"{FUNCS_NAMES[int(v)]}"
        res += " "
    return res


def to_cuda_node(
        node_values: jax.Array, node_types: jax.Array, subtree_sizes: jax.Array
) -> jax.Array:
    """
    Convert the given GP node values, types and subtree sizes to a compact form used in CUDA.

    Can be compiled by `jax.jit`. Inputs shall be a batch of nodes (a GP tree).

    Parameters
    ----------
    `node_values` : The node values composed of functions (see `Function`) as floats, constant values and variable indices as floats.

    `node_types` : The node type codes (see `NodeType`) of any data type (`jnp.int16` or `jnp.int32` is recommended).

    `subtree_sizes` : The subtree sizes of the nodes (include current node) of any data type (`jnp.int16` or `jnp.int32` is recommended).
    """
    assert node_values.dtype in [jnp.float32, jnp.float64]

    int_type = jnp.int16 if node_values.dtype == jnp.float32 else jnp.int32
    nt_and_sz = jnp.column_stack(
        [node_types.astype(int_type), subtree_sizes.astype(int_type)]
    )
    nt_and_sz = jax.lax.bitcast_convert_type(nt_and_sz, node_values.dtype)
    return jax.lax.complex(node_values, nt_and_sz)


def to_cuda_node_multi_output(
        node_values: jax.Array,
        node_types: jax.Array,
        subtree_sizes: jax.Array,
        output_indices: jax.Array,
        result_length: int,
) -> jax.Array:
    """
    Convert the given GP node values, types, subtree sizes and output indices to a compact form used in CUDA.

    Can be compiled by `jax.jit`. Inputs shall be a batch of nodes.

    Parameters
    ----------
    `node_values` : The node values composed of functions (see `Function`) as floats, constant values and variable indices as floats.

    `node_types` : The node type codes (see `NodeType`) of any data type (`jnp.int16` or `jnp.int32` is recommended).

    `subtree_sizes` : The subtree sizes of the nodes (include current node) of any data type (`jnp.int16` or `jnp.int32` is recommended).

    `output_indices` : The output index array of the nodes. Nodes with `output_indices` values between `[0, result_length)` are considered output nodes. Note that all resulting values of different nodes with duplicate output indices will be summed to give the final result when evaluating a GP tree.

    `result_length` : An `int` representing the maximum number of outputs. When using `jax.jit`, this input shall be considered as a static argument. If this number is smaller than 2, then `to_jax_node` will be used instead.
    """
    if result_length <= 1:
        return to_cuda_node(node_values, node_types, subtree_sizes)

    assert node_values.dtype in [jnp.float32, jnp.float64]

    int_type = jnp.int16 if node_values.dtype == jnp.float32 else jnp.int32
    output_indices = output_indices.astype(int_type)
    func_and_outIdx = jnp.column_stack([node_values.astype(int_type), output_indices])
    func_and_outIdx = jax.lax.bitcast_convert_type(func_and_outIdx, node_values.dtype)
    is_out_node = jnp.logical_and(output_indices >= 0, output_indices < result_length)
    node_types = node_types.astype(int_type)
    node_types = jnp.where(
        is_out_node,
        jnp.bitwise_or(node_types, jnp.array(NType.OUT_NODE, dtype=int_type)),
        node_types,
    )
    nt_and_sz = jnp.column_stack([node_types, subtree_sizes.astype(int_type)])
    nt_and_sz = jax.lax.bitcast_convert_type(nt_and_sz, jnp.float32)
    return jax.lax.complex(
        jnp.where(
            is_out_node,
            func_and_outIdx,
            node_values,
        ),
        nt_and_sz,
    )


def from_cuda_node(
        cuda_nodes: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Convert the compact form of GP used in CUDA to distinct GP node values, types, subtree sizes and (possible) output indices.

    Can be compiled by `jax.jit`. Inputs can either be a batch of nodes (a GP) or a batch of GPs.

    Parameters
    ----------
    `cuda_nodes` : The single-array GP tree(s) representation similar to the output array of `to_jax_node_multi_output` or `to_jax_node`.

    Returns
    -------
    `node_values` : The node values composed of functions (see `Function`) as floats, constant values and variable indices as floats.

    `node_types` : The node type codes (see `NodeType`) of any data type (`jnp.int16` or `jnp.int32` is recommended).

    `subtree_sizes` : The subtree sizes of the nodes (include current node) of any data type (`jnp.int16` or `jnp.int32` is recommended).

    `output_indices` : The output index array of the nodes. Note that there is no output for output length since it shall be a static argument for `jax.jit`. May contain invalid values if the GP tree(s) do(es) not output multiple values.
    """
    assert cuda_nodes.dtype in [jnp.complex64, jnp.complex128]

    int_type = jnp.int16 if cuda_nodes.dtype == jnp.complex64 else jnp.int32
    float_type = jnp.float32 if cuda_nodes.dtype == jnp.complex64 else jnp.float64
    nt_and_sz = jax.lax.bitcast_convert_type(jnp.imag(cuda_nodes), int_type)
    node_values = jnp.real(cuda_nodes)
    if len(nt_and_sz.shape) > 2:
        node_types = jax.vmap(lambda x: x[:, 0], in_axes=-2)(nt_and_sz).T
        subtree_sizes = jax.vmap(lambda x: x[:, 1], in_axes=-2)(nt_and_sz).T
    else:
        node_types = nt_and_sz[:, 0]
        subtree_sizes = nt_and_sz[:, 1]
    func_and_outIdx = jax.lax.bitcast_convert_type(node_values, int_type)
    if len(nt_and_sz.shape) > 2:
        func_values = jax.vmap(lambda x: x[:, 0], in_axes=-2)(func_and_outIdx).T
        output_indices = jax.vmap(lambda x: x[:, 1], in_axes=-2)(func_and_outIdx).T
    else:
        func_values = func_and_outIdx[:, 0]
        output_indices = func_and_outIdx[:, 1]
    node_values = jnp.where(
        (node_types & NType.OUT_NODE) != 0,
        func_values.astype(float_type),
        node_values,
    )
    output_indices = jnp.where(
        (node_types & NType.OUT_NODE) != 0,
        output_indices.astype(int_type),
        -1,
    )
    node_types = jnp.where(
        (node_types & NType.OUT_NODE) != 0,
        node_types - NType.OUT_NODE,
        node_types,
    )

    return node_values, node_types, subtree_sizes, output_indices


@jax.jit
def tree_size(cuda_tree: jax.Array):
    """
    Get the size of the given GP tree in CUDA representation.

    Parameters
    ----------
    cuda_tree : The single-array GP tree representation similar to the output array of `to_cuda_node_multi_output` or `to_cuda_node`.

    Returns
    -------
    `tree_size` : The size of the given GP tree.
    """

    assert len(cuda_tree.shape) == 1
    assert cuda_tree.dtype in [jnp.complex64, jnp.complex128]

    int_type = jnp.int16 if cuda_tree.dtype == jnp.complex64 else jnp.int32
    nt_and_sz = jax.lax.bitcast_convert_type(jnp.imag(cuda_tree[0]), int_type)
    return nt_and_sz[1]
