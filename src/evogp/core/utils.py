import numpy as np
import torch
import networkx as nx
from typing import Optional, Tuple
from torch import Tensor


MAX_STACK = 1024
MAX_FULL_DEPTH = 10


class NType:
    """
    The enumeration class for GP node types.
    """

    VAR = 0  # variable
    CONST = 1  # constant
    UFUNC = 2  # unary function
    BFUNC = 3  # binary function
    TFUNC = 4  # ternary function
    TYPE_MASK = 0x7F  # node type mask
    OUT_NODE = 1 << 7  # out node flag
    UFUNC_OUT = UFUNC + OUT_NODE  # unary function, output node
    BFUNC_OUT = BFUNC + OUT_NODE  # binary function, output node
    TFUNC_OUT = TFUNC + OUT_NODE  # ternary function, output node


class Func:
    """
    The enumeration class for GP function types.
    """

    IF = 0

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 5
    MAX = 6
    MIN = 7
    LT = 8
    GT = 9
    LE = 10
    GE = 11

    SIN = 12
    COS = 13
    TAN = 14
    SINH = 15
    COSH = 16
    TANH = 17
    LOG = 18
    EXP = 19
    INV = 20
    NEG = 21
    ABS = 22
    SQRT = 23
    END = 24


FUNCS = [
    Func.IF,
    Func.ADD,
    Func.SUB,
    Func.MUL,
    Func.DIV,
    Func.POW,
    Func.MAX,
    Func.MIN,
    Func.LT,
    Func.GT,
    Func.LE,
    Func.GE,
    Func.SIN,
    Func.COS,
    Func.TAN,
    Func.SINH,
    Func.COSH,
    Func.TANH,
    Func.LOG,
    Func.EXP,
    Func.INV,
    Func.NEG,
    Func.ABS,
    Func.SQRT,
]

FUNCS_NAMES = [
    "if",  # 0
    "+",  # 1
    "-",  # 2
    "*",  # 3
    "/",  # 4
    "pow",  # 5
    "max",  # 6
    "min",  # 7
    "<",  # 8
    ">",  # 9
    "<=",  # 10
    ">=",  # 11
    "sin",  # 12
    "cos",  # 13
    "tan",  # 14
    "sinh",  # 15
    "cosh",  # 16
    "tanh",  # 17
    "log",  # 18
    "exp",  # 19
    "inv",  # 20
    "neg",  # 21
    "abs",  # 22
    "sqrt",  # 23
]


def to_numpy(li):
    for idx, e in enumerate(li):
        if type(e) == torch.Tensor:
            li[idx] = e.cpu().numpy()
    return li


def dict2prob(prob_dict):
    # Probability Dictionary to Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    prob = np.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert (
            key in FUNCS_NAMES
        ), f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        prob[idx] = val

    # normalize
    prob = prob / prob.sum()

    return prob


def dict2cdf(prob_dict):
    # Probability Dictionary to Cumulative Distribution Function
    prob = dict2prob(prob_dict)

    return np.cumsum(prob)


def check_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device="cuda", requires_grad=False)
    else:
        x = x.requires_grad_(False).cuda()
        return x


def validate_forest(forest):
    batch_node_type = forest.batch_node_type.cpu()
    batch_subtree_size = forest.batch_subtree_size.cpu()
    for i in range(forest.pop_size):
        node_type = batch_node_type[i]
        subtree_size = batch_subtree_size[i]
        assert subtree_size[0] > 0, "Empty tree"
        assert subtree_size[0] <= MAX_STACK, "Tree size exceeds MAX_STACK"
        for j in range(subtree_size[0]):
            match node_type[j]:
                case NType.CONST | NType.VAR:
                    assert (
                        subtree_size[j] == 1
                    ), "Invalid subtree size of constant or variable"
                case NType.UFUNC:
                    assert (
                        subtree_size[j] == 1 + subtree_size[j + 1]
                    ), "Invalid subtree size of unary function"
                case NType.BFUNC:
                    child1_size = subtree_size[j + 1]
                    child2_size = subtree_size[j + 1 + child1_size]
                    assert (
                        subtree_size[j] == 1 + child1_size + child2_size
                    ), "Invalid subtree size of binary function"
                case NType.TFUNC:
                    child1_size = subtree_size[j + 1]
                    child2_size = subtree_size[j + 1 + child1_size]
                    child3_size = subtree_size[j + 1 + child1_size + child2_size]
                    assert (
                        subtree_size[j] == 1 + child1_size + child2_size + child3_size
                    ), "Invalid subtree size of ternary function"
                case _:
                    raise ValueError("Invalid node type")


def str_tree(value, node_type, subtree_size):
    res = ""
    for i in range(0, subtree_size[0]):
        if (
            (node_type[i] == NType.UFUNC)
            or (node_type[i] == NType.BFUNC)
            or (node_type[i] == NType.TFUNC)
        ):
            res = res + FUNCS_NAMES[int(value[i])]
        elif node_type[i] == NType.VAR:
            res = res + f"x[{int(value[i])}]"
        elif node_type[i] == NType.CONST:
            res = res + f"{value[i]:.2f}"
        res += " "

    return res


def to_infix(num, node_type, node_val):
    node_type, node_val = list(node_type[:num][::-1]), list(node_val[:num][::-1])
    stack = []
    for t, v in zip(node_type, node_val):
        if t == NType.VAR:
            stack.append(f"in[{int(v)}]")
        elif t == NType.CONST:
            stack.append(f"{v}")
        elif t == NType.UFUNC:
            stack.append(f"{FUNCS_NAMES[int(v)]}({stack.pop()})")
        elif t == NType.BFUNC:
            if int(v) in [5, 6, 7]:
                stack.append(f"{FUNCS_NAMES[int(v)]}({stack.pop()},{stack.pop()})")
            else:
                stack.append(f"({stack.pop()}{FUNCS_NAMES[int(v)]}{stack.pop()})")
        elif t == NType.TFUNC:
            stack.append(
                f"{FUNCS_NAMES[int(v)]}({stack.pop()},{stack.pop()},{stack.pop()})"
            )
    return stack.pop()


def fillout_graph(graph: nx.DiGraph, type_list, val_list, output_list):
    """Recursive Traversal"""
    node_id = graph.node_count
    node_type, node_val, output_index = (
        type_list[node_id],
        val_list[node_id],
        output_list[0],
        # output_list[node_id],
    )
    if node_type == NType.CONST:
        node_label = int(node_val)
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
    node_type, node_val, output_index = (
        list(tree.node_type[: tree.subtree_size[0]]),
        list(tree.node_value[: tree.subtree_size[0]]),
        list([-1]),
        # list(tree.output_index[: tree.subtree_size[0]]),
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
    agraph.close()


def parse_generate_configs(
    gp_len: int,
    # depth2leaf_probs
    depth2leaf_probs: Optional[Tensor] = None,
    max_layer_cnt: Optional[int] = None,
    layer_leaf_prob: Optional[float] = None,
    # roulette_funcs
    roulette_funcs: Optional[Tensor] = None,
    func_prob: Optional[dict] = None,
    # const_samples
    const_samples: Optional[Tensor] = None,
    const_range: Optional[Tuple[float, float]] = None,
    sample_cnt: Optional[int] = None,
    **kwargs,
):
    """
    Args:
        gp_len: The length of each GP.
        depth2leaf_probs (optional): The probability of generating a leaf node at each depth.
        max_layer_cnt (optional): The maximum number of layers of the GP.
        layer_leaf_prob (optional): The probability of generating a leaf node at each layer.
        roulette_funcs (optional): The probability of generating each function.
        func_prob (optional): The probability of generating each function.
        const_samples (optional): The samples of constant values.
        const_range (optional): The range of constant values.
        sample_cnt (optional): The number of samples of constant values.
    """

    assert gp_len <= MAX_STACK, f"gp_len={gp_len} is too large, MAX_STACK={MAX_STACK}"

    if depth2leaf_probs is None:
        assert (
            max_layer_cnt is not None
        ), "max_layer_cnt should not be None when depth2leaf_probs is None"
        assert (
            layer_leaf_prob is not None
        ), "layer_leaf_prob should not be None when depth2leaf_probs is None"
        assert (
            2**max_layer_cnt <= gp_len
        ), f"max_layer_cnt is too large for gp_len={gp_len}"

        depth2leaf_probs = torch.tensor(
            [layer_leaf_prob] * max_layer_cnt
            + [1.0] * (MAX_FULL_DEPTH - max_layer_cnt),
            device="cuda",
            requires_grad=False,
        )
        depth2leaf_probs[max_layer_cnt - 1] = 1.0

    if roulette_funcs is None:
        assert (
            func_prob is not None
        ), "func_prob should not be None when roulette_funcs is None"
        roulette_funcs = torch.tensor(
            dict2cdf(func_prob),
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )

    if const_samples is None:
        assert (
            const_range is not None
        ), "const_range should not be None when const_samples is None"
        assert (
            sample_cnt is not None
        ), "sample_cnt should not be None when const_samples is None"
        const_samples = (
            torch.rand(sample_cnt, device="cuda", requires_grad=False)
            * (const_range[1] - const_range[0])
            + const_range[0]
        )

    assert depth2leaf_probs.shape == (
        MAX_FULL_DEPTH,
    ), f"depth2leaf_probs shape should be ({MAX_FULL_DEPTH}), but got {depth2leaf_probs.shape}"
    assert roulette_funcs.shape == (
        Func.END,
    ), f"roulette_funcs shape should be ({Func.END}), but got {roulette_funcs.shape}"
    assert (
        const_samples.dim() == 1
    ), f"const_samples dim should be 1, but got {const_samples.dim()}"

    return depth2leaf_probs, roulette_funcs, const_samples
