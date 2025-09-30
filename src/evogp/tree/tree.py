import torch
from torch import Tensor
import sympy as sp

from .utils import *
from .descriptor import GenerateDescriptor


class Tree:
    def __init__(
        self,
        input_len,
        output_len,
        node_value: Tensor,
        node_type: Tensor,
        subtree_size: Tensor,
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.max_tree_len = node_value.shape[0]

        assert node_value.shape == (
            self.max_tree_len,
        ), f"node_value shape should be {self.max_tree_len}, but got {node_value.shape}"
        assert node_type.shape == (
            self.max_tree_len,
        ), f"node_type shape should be {self.max_tree_len}, but got {node_type.shape}"
        assert subtree_size.shape == (
            self.max_tree_len,
        ), f"subtree_size shape should be {self.max_tree_len}, but got {subtree_size.shape}"

        self.node_value = node_value
        self.node_type = node_type
        self.subtree_size = subtree_size

    @staticmethod
    def random_generate(descriptor: GenerateDescriptor):
        # Delayed import to avoid circular dependency with the Forest class
        from .forest import Forest

        return Forest.random_generate(pop_size=1, descriptor=descriptor)[0]

    def forward(self, x: Tensor):
        x = check_tensor(x)

        assert x.dim() <= 2, f"x dim should be <= 2, but got {x.dim()}"

        is_expand_input = False
        if x.dim() == 1:
            is_expand_input = True
            x = x.unsqueeze(0)
        assert (
            x.shape[1] == self.input_len
        ), f"x shape should be {self.input_len}, but got {x.shape[1]}"

        batch_size = x.shape[0]
        batch_node_value = self.node_value.repeat(batch_size, 1)
        batch_node_type = self.node_type.repeat(batch_size, 1)
        batch_subtree_size = self.subtree_size.repeat(batch_size, 1)

        res = torch.ops.evogp_cuda.tree_evaluate(
            batch_size,
            self.max_tree_len,
            self.input_len,
            self.output_len,
            batch_node_value,
            batch_node_type,
            batch_subtree_size,
            x,
        )

        if is_expand_input:
            return res[0]
        else:
            return res

    def SR_fitness(
        self,
        inputs: Tensor,
        labels: Tensor,
        use_MSE: bool = True,
        execute_mode: str = "auto",
    ):
        inputs = check_tensor(inputs)
        labels = check_tensor(labels)

        assert execute_mode in [
            "hybrid parallel",
            "data parallel",
            "tree parallel",
            "auto",
        ], f"execute_mode should be one of ['hybrid parallel', 'data parallel', 'tree parallel', 'auto'], but got {execute_mode}"

        if execute_mode == "hybrid parallel":
            execute_code = 3
        elif execute_mode == "data parallel":
            execute_code = 1
        elif execute_mode == "tree parallel":
            execute_code = 2
        elif execute_mode == "auto":
            execute_code = 4
        batch_size = inputs.shape[0]

        assert inputs.shape == (
            batch_size,
            self.input_len,
        ), f"inputs shape should be ({batch_size}, {self.input_len}), but got {inputs.shape}"

        assert labels.shape == (
            batch_size,
            self.output_len,
        ), f"outputs shape should be ({batch_size}, {self.output_len}), but got {labels.shape}"

        res = torch.ops.evogp_cuda.tree_SR_fitness(
            1,
            batch_size,
            self.max_tree_len,
            self.input_len,
            self.output_len,
            use_MSE,
            self.node_value[None, :],
            self.node_type[None, :],
            self.subtree_size[None, :],
            inputs,
            labels,
            execute_code,
        )

        return res

    def to_forest(self):
        from .forest import Forest

        return Forest(
            self.input_len,
            self.output_len,
            self.node_value[None, :],
            self.node_type[None, :],
            self.subtree_size[None, :],
        )

    def __repr__(self):
        return self.to_sympy_expr().__repr__()
        # value, node_type, subtree_size = to_numpy(
        #     [self.node_value, self.node_type, self.subtree_size]
        # )
        # res = ""
        # for i in range(0, subtree_size[0]):
        #     if (
        #         (node_type[i] == NType.UFUNC)
        #         or (node_type[i] == NType.BFUNC)
        #         or (node_type[i] == NType.TFUNC)
        #     ):
        #         res = res + FUNCS_NAMES[int(value[i])]
        #     elif node_type[i] == NType.VAR:
        #         res = res + f"x{int(value[i])}"
        #     elif node_type[i] == NType.CONST:
        #         res = res + f"{value[i]:.2f}"
        #     res += " "

        # return res

    def to_infix(self):
        if self.output_len != 1:
            print(
                "Warning: output_len != 1, 'to_infix' function only support single output"
            )
        num = self.subtree_size[0]
        node_type = list(torch.flip(self.node_type[:num], [0]))
        node_val = list(torch.flip(self.node_value[:num], [0]))
        stack = []
        for t, v in zip(node_type, node_val):
            if t == NType.VAR:
                stack.append(f"x{int(v)}")
            elif t == NType.CONST:
                stack.append(f"{v:.2f}")
            elif t == NType.UFUNC:
                stack.append(f"{FUNCS_DISPLAY[int(v)]}({stack.pop()})")
            elif t == NType.BFUNC:
                if int(v) in [5, 6, 7]:
                    stack.append(
                        f"{FUNCS_DISPLAY[int(v)]}({stack.pop()},{stack.pop()})"
                    )
                else:
                    stack.append(
                        f"({stack.pop()} {FUNCS_DISPLAY[int(v)]} {stack.pop()})"
                    )
            elif t == NType.TFUNC:
                stack.append(
                    f"{FUNCS_DISPLAY[int(v)]}({stack.pop()},{stack.pop()},{stack.pop()})"
                )
        return stack.pop()

    def _fillout_graph(self, graph):
        """Recursive Traversal"""
        node_id = graph.node_idx
        node_type, node_val, output_index = (
            self.node_type[node_id] & NType.TYPE_MASK,
            self.node_value[node_id],
            (
                self.node_value[node_id].view(torch.int32) >> 16
                if (self.node_type[node_id] & NType.OUT_NODE)
                else -1
            ),
        )
        if node_type == NType.CONST:
            node_label = f"{node_val:.2f}"
            child_remain = 0
        elif node_type == NType.VAR:
            node_label = f"x{int(node_val)}"
            child_remain = 0
        elif node_type == NType.UFUNC:
            if output_index == -1:
                node_label = FUNCS_DISPLAY[int(node_val)]
            else:
                node_label = FUNCS_DISPLAY[node_val.view(torch.int32) & 0xFF]
            child_remain = 1
        elif node_type == NType.BFUNC:
            if output_index == -1:
                node_label = FUNCS_DISPLAY[int(node_val)]
            else:
                node_label = FUNCS_DISPLAY[node_val.view(torch.int32) & 0xFF]
            child_remain = 2
        elif node_type == NType.TFUNC:
            if output_index == -1:
                node_label = FUNCS_DISPLAY[int(node_val)]
            else:
                node_label = FUNCS_DISPLAY[node_val.view(torch.int32) & 0xFF]
            child_remain = 3

        if output_index == -1:
            graph.add_node(node_id, label=node_label)
        else:
            graph.add_node(
                node_id, label=node_label, xlabel=f"out[{output_index}]", color="red"
            )

        for i in range(child_remain):
            graph.node_idx += 1
            graph.add_edge(graph.node_idx, node_id, order=i)
            self._fillout_graph(graph)

    def to_png(self, fname):
        import networkx as nx
        from networkx.drawing.nx_agraph import to_agraph
        import pygraphviz

        graph = nx.DiGraph()
        graph.node_idx = 0
        self._fillout_graph(graph)
        agraph: pygraphviz.agraph.AGraph = to_agraph(graph)
        agraph.graph_attr.update(rankdir="BT")
        for edge in agraph.edges():
            edge.attr["dir"] = "back"
        agraph.graph_attr["label"] = f"size: {graph.node_idx + 1}"
        agraph.draw(fname, format="png", prog="dot")
        agraph.close()

    def to_sympy_expr(self, symbol_names=None):
        node_value, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
        tree_size = subtree_size[0]
        if symbol_names is not None:
            x = sp.symbols(symbol_names, real=True)
        else:
            x = sp.symbols(f"x:{self.input_len}", real=True)
        expr_stack = []
        if self.output_len == 1:
            for i in reversed(range(tree_size)):
                t, v = node_type[i], node_value[i]
                if t == NType.VAR:
                    expr_stack.append(x[int(v)])
                elif t == NType.CONST:
                    expr_stack.append(v)
                else:  # Function
                    if t == NType.UFUNC:
                        mid = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](mid)
                    elif t == NType.BFUNC:
                        left = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, right)
                    elif t == NType.TFUNC:
                        left = expr_stack.pop(-1)
                        mid = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, mid, right)
                    expr_stack.append(res)
            return expr_stack[0]
        if self.output_len > 1:
            expr = [0] * self.output_len
            for i in reversed(range(tree_size)):
                t, v = node_type[i], node_value[i]
                if t & NType.OUT_NODE:
                    out_idx = v.view(np.int32) >> 16
                    v = v.view(np.int32) & 0xFF
                else:
                    out_idx = -1
                t = t & NType.TYPE_MASK

                if t == NType.VAR:
                    expr_stack.append(x[int(v)])
                elif t == NType.CONST:
                    expr_stack.append(v)
                else:  # Function
                    if t == NType.UFUNC:
                        mid = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](mid)
                    elif t == NType.BFUNC:
                        left = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, right)
                    elif t == NType.TFUNC:
                        left = expr_stack.pop(-1)
                        mid = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, mid, right)
                    if out_idx != -1:
                        expr[out_idx] += res
                        expr_stack.append(right)
                    else:
                        expr_stack.append(res)
            return expr

    def __python_forward(self, inputs: np.ndarray) -> np.ndarray:
        assert inputs.shape == (self.input_len,)
        assert self.output_len == 1
        value, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
        tree_size = subtree_size[0]
        operents = []
        for i in range(tree_size - 1, -1, -1):
            if node_type[i] == NType.VAR:
                operents.append(inputs[int(value[i])])
            elif node_type[i] == NType.CONST:
                operents.append(value[i])
            else:
                op1 = operents.pop(-1)  # right child
                op2 = operents.pop(-1)  # left child
                if value[i] == Func.ADD:
                    res = op2 + op1
                elif value[i] == Func.SUB:
                    res = op2 - op1
                elif value[i] == Func.MUL:
                    res = op2 * op1
                elif value[i] == Func.DIV:
                    if np.allclose(op1, 0.0):
                        res = op2
                    else:
                        res = op2 / op1
                else:
                    raise NotImplementedError
                operents.append(res)

        # check sucess
        assert len(operents) == 1
        return operents[0]

    def __assert_valid(self):
        _, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
        # check forward success
        dummy_input = np.array([0.0] * self.input_len)
        self.__python_forward(dummy_input)

        # check subtree size valid
        needed_length, idx = 1, 0
        while True:
            needed_length -= 1
            if node_type[idx] == NType.UFUNC:
                needed_length += 1
            elif node_type[idx] == NType.BFUNC:
                needed_length += 2
            elif node_type[idx] == NType.TFUNC:
                needed_length += 3

            idx += 1
            if needed_length == 0:
                break

        root_real_size = idx

        assert subtree_size[0] == root_real_size
        # check subsize valid
        operents = []
        for i in range(root_real_size - 1, -1, -1):
            if node_type[i] == NType.VAR:
                size = 1
            elif node_type[i] == NType.CONST:
                size = 1
            elif node_type[i] == NType.UFUNC:
                op1 = operents.pop(-1)
                size = op1 + 1
            elif node_type[i] == NType.BFUNC:
                op1 = operents.pop(-1)
                op2 = operents.pop(-1)
                size = op1 + op2 + 1
            elif node_type[i] == NType.TFUNC:
                op1 = operents.pop(-1)
                op2 = operents.pop(-1)
                op3 = operents.pop(-1)
                size = op1 + op2 + op3 + 1
            else:
                raise NotImplementedError
            operents.append(size)
            assert subtree_size[i] == size

        assert len(operents) == 1

        assert True
