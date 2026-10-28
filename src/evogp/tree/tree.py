import torch
from torch import Tensor
import numpy as np
import sympy as sp

from .utils import *
from .descriptor import GenerateDescriptor
from . import dense_engine, evoxc_engine


def _comparison_as_value(res):
    """Render a comparison the way the engine evaluates it.

    ``SYMPY_MAP`` renders ``< > <= >=`` (``Func`` ids 10-13) as sympy relations, but
    the engine returns exactly ``1.0`` when the comparison holds and ``-1.0`` when it
    does not - including for any NaN operand (ENGINE_SPEC section 3) - so a bare
    relation (or the boolean it folds to) would evaluate to ``True``/``False``, i.e.
    1/0, in numeric use. Sympy's ``Boolean`` is also the base class of plain
    ``Symbol``, hence the mapping keys on ``Relational`` and the boolean atoms only.
    Anything else is returned untouched.
    """
    if getattr(res, "is_Relational", False):
        return sp.Piecewise((1, res), (-1, True))
    if res is sp.true:
        return sp.S(1)
    if res is sp.false:
        return sp.S(-1)
    return res


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

        # EVOGP_EVAL_MODE decides between the dense (one graph for the whole
        # population) and the specialized (one graph per tree) engine; a single
        # tree is a population of one, and DenseUnsupported falls back below.
        if dense_engine.should_use_dense(1):
            try:
                res = dense_engine.evaluate_dense(
                    self.node_value.unsqueeze(0),
                    self.node_type.unsqueeze(0),
                    self.subtree_size.unsqueeze(0),
                    self.input_len,
                    self.output_len,
                    x,
                    per_row=is_expand_input,
                )
                return res[0]
            except dense_engine.DenseUnsupported:
                pass

        pop = evoxc_engine.decode_population(
            self.node_value.unsqueeze(0),
            self.node_type.unsqueeze(0),
            self.subtree_size.unsqueeze(0),
            self.input_len,
            self.output_len,
        )

        # A 1-D input is the single tree evaluated on one row (per_row); an
        # explicit 2-D input evaluates the tree on every row of the batch.  The
        # population axis (always 1 here) is dropped from the result either way.
        res = evoxc_engine.evaluate(pop, x, per_row=is_expand_input)

        return res[0]

    def SR_fitness(
        self,
        inputs: Tensor,
        labels: Tensor,
        use_MSE: bool = True,
        execute_mode: str = "auto",
    ):
        inputs = check_tensor(inputs)
        labels = check_tensor(labels)

        # The former kernel_type ids are gone; every mode string (including "torch")
        # evaluates through the same evoxc engine path.
        assert execute_mode in [
            "torch",
            "hybrid parallel",
            "data parallel",
            "tree parallel",
            "auto",
        ], f"execute_mode should be one of ['torch', 'hybrid parallel', 'data parallel', 'tree parallel', 'auto'], but got {execute_mode}"

        batch_size = inputs.shape[0]

        assert inputs.shape == (
            batch_size,
            self.input_len,
        ), f"inputs shape should be ({batch_size}, {self.input_len}), but got {inputs.shape}"

        assert labels.shape == (
            batch_size,
            self.output_len,
        ), f"outputs shape should be ({batch_size}, {self.output_len}), but got {labels.shape}"

        # Dense attempt / fallback as in ``forward`` (the population is this one
        # tree, i.e. pop_size == 1).
        if dense_engine.should_use_dense(1):
            try:
                return dense_engine.sr_fitness_dense(
                    self.node_value[None, :],
                    self.node_type[None, :],
                    self.subtree_size[None, :],
                    self.input_len,
                    self.output_len,
                    inputs,
                    labels,
                    use_MSE=use_MSE,
                )
            except dense_engine.DenseUnsupported:
                pass

        pop = evoxc_engine.decode_population(
            self.node_value[None, :],
            self.node_type[None, :],
            self.subtree_size[None, :],
            self.input_len,
            self.output_len,
        )

        return evoxc_engine.sr_fitness(pop, inputs, labels, use_MSE=use_MSE)

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
        for raw_t, v in zip(node_type, node_val):
            t = int(raw_t) & NType.TYPE_MASK
            if int(raw_t) & NType.OUT_NODE and t >= NType.UFUNC:
                # a tagged function node packs {function, outIndex} into its value; only
                # the function id is renderable here (the per-output sums are not).
                v = int(v.view(torch.int32)) & 0xFF
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
                    # sympify the raw numpy scalar: a constant-only tree would otherwise
                    # render (and return) ``np.float32(...)`` instead of a sympy number
                    expr_stack.append(sp.sympify(v))
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
                    expr_stack.append(_comparison_as_value(res))
            return expr_stack[0]
        if self.output_len > 1:
            # Multi-output protocol (ENGINE_SPEC sections 2.3 / 4.3): each output is the
            # sum of the values computed by the *tagged* function nodes carrying that
            # outIndex - an out-node's own value, not its right-most child's. Every
            # function node, tagged or not, then passes only its right-most child's value
            # up to its parent, and a tag on a leaf is ignored (a leaf is evaluated from
            # its raw node_value like any other leaf).
            expr = [0] * self.output_len
            for i in reversed(range(tree_size)):
                t, v = node_type[i], node_value[i]
                t = t & NType.TYPE_MASK

                if t == NType.VAR:
                    expr_stack.append(x[int(v)])
                elif t == NType.CONST:
                    # sympify as in the single-output branch, so a constant reached from a
                    # tagged node keeps the output a sympy expression
                    expr_stack.append(sp.sympify(v))
                else:  # Function
                    if node_type[i] & NType.OUT_NODE:
                        # tagged value: bit-reinterpretation of {function, outIndex}
                        out_idx = int(v.view(np.int32) >> 16)
                        v = v.view(np.int32) & 0xFF
                    else:
                        out_idx = -1

                    if t == NType.UFUNC:
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](right)
                    elif t == NType.BFUNC:
                        left = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, right)
                    elif t == NType.TFUNC:
                        left = expr_stack.pop(-1)
                        mid = expr_stack.pop(-1)
                        right = expr_stack.pop(-1)
                        res = SYMPY_MAP[v](left, mid, right)
                    else:
                        # the engine would evaluate any other base type as the ternary
                        # `if`; refuse instead of rendering something misleading
                        raise NotImplementedError(f"unknown node type {t} at node {i}")

                    # out-of-range output indices are dropped
                    if 0 <= out_idx < self.output_len:
                        expr[out_idx] += _comparison_as_value(res)
                    expr_stack.append(right)
            return expr

    def __python_forward(self, inputs: np.ndarray) -> np.ndarray:
        """CPU reference evaluator for a single-output tree (ENGINE_SPEC sections 3 / 4.1).

        Nodes are walked in reverse prefix order, so the first popped operand is the
        *first* (leftmost) child, exactly like the stack walk of the engine: ``-`` is
        ``first_child - second_child``, ``/`` is ``first_child / second_child`` and the
        ternary ``if`` tests the first child. A node whose base type is neither
        VAR/CONST/UFUNC/BFUNC is evaluated as that ternary ``if`` (the kernel does not
        even read its function id), and a function id matching no branch yields 0.0.
        """
        assert inputs.shape == (self.input_len,)
        assert self.output_len == 1
        value, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
        tree_size = subtree_size[0]
        delta, maxval = np.float32(DELTA), np.float32(MAXVAL)
        operents = []
        # NaN/±inf are part of the semantics (they propagate), so numpy's warnings are
        # silenced instead of masking the values - the engine sanitizes nothing either.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for i in range(tree_size - 1, -1, -1):
                if node_type[i] == NType.VAR:
                    operents.append(inputs[int(value[i])])
                elif node_type[i] == NType.CONST:
                    operents.append(value[i])
                elif node_type[i] == NType.UFUNC:
                    var1 = operents.pop(-1)  # first (only) child
                    func = value[i]
                    if func == Func.SIN:
                        res = np.sin(var1)
                    elif func == Func.COS:
                        res = np.cos(var1)
                    elif func == Func.TAN:
                        res = np.tan(var1)
                    elif func == Func.SINH:
                        res = np.sinh(var1)
                    elif func == Func.COSH:
                        res = np.cosh(var1)
                    elif func == Func.TANH:
                        res = np.tanh(var1)
                    elif func == Func.LOG:
                        res = np.log(var1)  # log(0) = -inf, log(x < 0) = nan
                    elif func == Func.LOOSE_LOG:
                        res = -maxval if var1 == 0.0 else np.log(np.abs(var1))
                    elif func == Func.EXP:
                        res = np.exp(var1)
                    elif func == Func.INV:
                        res = float("nan") if var1 == 0.0 else 1.0 / var1
                    elif func == Func.LOOSE_INV:
                        if np.abs(var1) <= delta:
                            var1 = np.copysign(delta, var1)
                        res = 1.0 / var1
                    elif func == Func.NEG:
                        res = -var1
                    elif func == Func.ABS:
                        res = np.abs(var1)
                    elif func == Func.SQRT:
                        res = np.sqrt(var1)  # negative input -> nan
                    elif func == Func.LOOSE_SQRT:
                        if var1 <= 0.0:
                            var1 = np.abs(var1)
                        res = np.sqrt(var1)
                    else:
                        res = 0.0  # unmatched id: the kernel leaves top_val at 0.0
                    operents.append(res)
                elif node_type[i] == NType.BFUNC:
                    var1 = operents.pop(-1)  # first child
                    var2 = operents.pop(-1)  # second child
                    func = value[i]
                    if func == Func.ADD:
                        res = var1 + var2
                    elif func == Func.SUB:
                        res = var1 - var2
                    elif func == Func.MUL:
                        res = var1 * var2
                    elif func == Func.DIV:
                        res = float("nan") if var2 == 0.0 else var1 / var2
                    elif func == Func.LOOSE_DIV:
                        if np.abs(var2) <= delta:
                            var2 = np.copysign(delta, var2)
                        res = var1 / var2
                    elif func == Func.POW:
                        res = np.power(var1, var2)
                    elif func == Func.LOOSE_POW:
                        if var1 == 0.0 and var2 == 0.0:
                            res = 0.0
                        else:
                            res = np.power(np.abs(var1), var2)
                    # comparisons are ternaries, not fmax/fmin: a NaN operand takes the
                    # second branch (max(v1, nan) = nan but max(nan, v2) = v2).
                    elif func == Func.MAX:
                        res = var1 if var1 >= var2 else var2
                    elif func == Func.MIN:
                        res = var1 if var1 <= var2 else var2
                    elif func == Func.LT:
                        res = 1.0 if var1 < var2 else -1.0
                    elif func == Func.GT:
                        res = 1.0 if var1 > var2 else -1.0
                    elif func == Func.LE:
                        res = 1.0 if var1 <= var2 else -1.0
                    elif func == Func.GE:
                        res = 1.0 if var1 >= var2 else -1.0
                    else:
                        res = 0.0  # unmatched id: the kernel leaves top_val at 0.0
                    operents.append(res)
                else:
                    # TFUNC - and, like the engine, any other base type (e.g. a node
                    # still carrying the OUT_NODE tag): evaluated as `if`, whose
                    # function id is never looked at.
                    var1 = operents.pop(-1)  # condition (first child)
                    var2 = operents.pop(-1)
                    var3 = operents.pop(-1)
                    operents.append(var2 if var1 > 0.0 else var3)

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
