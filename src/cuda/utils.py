import jax
import jax.numpy as jnp

from src.utils.enum import NType, FUNCS_NAMES
import networkx as nx
import operator, sympy


class Function:
    """
    A general function
    """

    def __init__(self, func, arity, symbol=None):
        self.func = func
        self.arity = arity
        self.symbol = symbol
        self.name = func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


function_set = [
    Function(operator.eq, 3),  # 0
    Function(operator.add, 2, "+"),  # 1
    Function(operator.sub, 2, "-"),  # 2
    Function(operator.mul, 2, "×"),  # 3
    Function(operator.truediv, 2, "/"),  # 4
    Function(operator.pow, 2),  # 5
    Function(sympy.Max, 2),  # 6
    Function(sympy.Min, 2),  # 7
    Function(operator.lt, 2, "<"),  # 8
    Function(operator.gt, 2, ">"),  # 9
    Function(operator.le, 2, "≤"),  # 10
    Function(operator.ge, 2, "≥"),  # 11
    Function(sympy.sin, 1),  # 12
    Function(sympy.cos, 1),  # 13
    Function(sympy.tan, 1),  # 14
    Function(sympy.sinh, 1),  # 15
    Function(sympy.cosh, 1),  # 16
    Function(sympy.tanh, 1),  # 17
    Function(sympy.log, 1),  # 18
    Function(sympy.exp, 1),  # 19
    Function(operator.inv, 1, "1/"),  # 20
    Function(operator.neg, 1, "-"),  # 21
    Function(operator.abs, 1),  # 22
    Function(sympy.sqrt, 1, "√"),  # 23
]


def tree2str(tree):
    return to_string(tree.node_types, tree.node_vals)


def cuda_tree_to_string(tree):
    node_val, node_type, node_size, _ = from_cuda_node(tree)
    return to_string(node_type[: node_size[0]], node_val[: node_size[0]])


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


""" Recursive Traversal """


def fillout_graph(graph: nx.DiGraph, type_list, val_list, output_list):
    """Recursive Traversal"""
    node_id = graph.node_count
    node_type, node_val, output_index = (
        type_list[node_id],
        val_list[node_id],
        output_list[node_id],
    )
    if node_type == NType.CONST:
        node_label = str(node_val)
        child_remain = 0
    elif node_type == NType.VAR:
        node_label = chr(ord("A") + int(node_val))
        child_remain = 0
    elif node_type == NType.UFUNC:
        node_label = FUNCS_NAMES[int(node_val)]
        child_remain = 1
    elif node_type == NType.BFUNC:
        node_label = FUNCS_NAMES[int(node_val)]
        child_remain = 2
    elif node_type == NType.TFUNC:
        node_label = FUNCS_NAMES[int(node_val)]
        child_remain = 3

    if output_index == -1:
        graph.add_node(node_id, label=node_label)
    else:
        graph.add_node(
            node_id, label=node_label, xlabel=f"out[{output_index}]", color="red"
        )

    for i in range(child_remain):
        graph.node_count += 1
        graph.add_edge(graph.node_count, node_id, order=i)
        fillout_graph(graph, type_list, val_list, output_list)


def to_graph(tree):
    node_val, node_type, subtree_size, output_index = from_cuda_node(tree)
    node_type, node_val, output_index = (
        list(node_type[: subtree_size[0]]),
        list(node_val[: subtree_size[0]]),
        list(output_index[: subtree_size[0]]),
    )
    graph = nx.DiGraph()
    graph.node_count = 0
    fillout_graph(graph, node_type, node_val, output_index)
    graph.node_count += 1
    return graph


def to_png(graph, fname):
    from networkx.drawing.nx_agraph import to_agraph
    import pygraphviz

    agraph: pygraphviz.agraph.AGraph = to_agraph(graph)
    agraph.graph_attr.update(rankdir="BT")
    agraph.graph_attr["label"] = f"size: {graph.node_count}"
    agraph.draw(fname, format="png", prog="dot")


def concat(func: Function, args):
    if func.arity == 1:
        if func.symbol:
            expr = f"{func.symbol}{args[0]}"
        else:
            expr = f"{func.name}({args[0]})"
    elif func.arity == 2:
        if func.symbol:
            expr = f"{args[0]}{func.symbol}{args[1]}"
        else:
            expr = f"{func.name}({args[0]},{args[1]})"
    elif func.arity == 3:
        expr = f"{func.name}({args[0]},{args[1]},{args[2]})"
    return expr


def to_sympy(graph: nx.DiGraph):
    tp_sort = list(nx.topological_sort(graph))
    # if len(tp_sort) <= 0: return
    dict_expr = dict()
    for node_id in tp_sort:
        inputs = []
        for input_node_id in graph.predecessors(node_id):
            for order in graph.get_edge_data(input_node_id, node_id).values():
                inputs.append((input_node_id, order))
        inputs.sort(key=operator.itemgetter(1))
        args = []
        for input in inputs:
            input_node_id = input[0]
            args.append(dict_expr[input_node_id])
        sym_func = graph.nodes[node_id]["func"]
        if sym_func in function_set:
            try:
                expr = sym_func(*args)
                expr = sympy.simplify(expr)
            except:
                # print("sympy error: cannot parse")
                try:
                    expr = concat(sym_func, args)
                except:
                    print("error: not a GP tree")
                    expr = sym_func.name
            dict_expr[node_id] = expr
        else:
            dict_expr[node_id] = sym_func
    return dict_expr[tp_sort[-1]]


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

    Can be compiled by `jax.jit`. Inputs shall be a batch of nodes 

    ***Attention: this function is only used for a single GP tree.

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
