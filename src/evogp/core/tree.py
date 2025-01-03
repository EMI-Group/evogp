import torch
from torch import Tensor
from .utils import *


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
        self.gp_len = node_value.shape[0]

        assert node_value.shape == (
            self.gp_len,
        ), f"node_value shape should be {self.gp_len}, but got {node_value.shape}"
        assert node_type.shape == (
            self.gp_len,
        ), f"node_type shape should be {self.gp_len}, but got {node_type.shape}"
        assert subtree_size.shape == (
            self.gp_len,
        ), f"subtree_size shape should be {self.gp_len}, but got {subtree_size.shape}"

        self.node_value = node_value
        self.node_type = node_type
        self.subtree_size = subtree_size

    @staticmethod
    def random_generate(*args, **kwargs):
        # Delayed import to avoid circular dependency with the Forest class
        from .forest import Forest

        return Forest.random_generate(pop_size=1, *args, **kwargs)[0]

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

        res = torch.ops.evogp.tree_evaluate(
            batch_size,  # popsize
            self.gp_len,  # gp_len
            self.input_len,  # var_len
            self.output_len,  # out_len
            batch_node_value,  # value
            batch_node_type,  # node_type
            batch_subtree_size,  # subtree_size
            x,  # variables
        )

        if is_expand_input:
            return res[0]
        else:
            return res

    def SR_fitness(self, inputs: Tensor, labels: Tensor, use_MSE: bool = True):
        inputs = check_tensor(inputs)
        labels = check_tensor(labels)

        batch_size = inputs.shape[0]
        assert inputs.shape == (
            batch_size,
            self.input_len,
        ), f"inputs shape should be ({batch_size}, {self.input_len}), but got {inputs.shape}"

        assert labels.shape == (
            batch_size,
            self.output_len,
        ), f"outputs shape should be ({batch_size}, {self.output_len}), but got {labels.shape}"

        res = torch.ops.evogp.tree_SR_fitness(
            1,
            batch_size,
            self.gp_len,
            self.input_len,
            self.output_len,
            use_MSE,
            self.batch_node_value,
            self.batch_node_type,
            self.batch_subtree_size,
            inputs,
            labels,
        )

        return res

    def python_forward(self, inputs: np.ndarray) -> np.ndarray:
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

    def assert_valid(self):
        value, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
        # check forward success
        dummy_input = np.array([0.0] * self.input_len)
        self.python_forward(dummy_input)

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

    def __str__(self):
        value, node_type, subtree_size = to_numpy(
            [self.node_value, self.node_type, self.subtree_size]
        )
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
