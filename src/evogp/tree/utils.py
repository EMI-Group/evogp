import inspect
from typing import Callable
import numpy as np
import torch
import sympy as sp

DELTA = 1e-9
MAXVAL = 1e9

# Tree engine limits: ``max_tree_len`` (the row stride of a population) must not
# exceed ``MAX_STACK``, and the depth -> leaf-probability schedule has exactly
# ``MAX_FULL_DEPTH`` entries.
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
    OUT_NODE = 1 << 7  # out node fl ag
    UFUNC_OUT = UFUNC + OUT_NODE  # unary function, output node
    BFUNC_OUT = BFUNC + OUT_NODE  # binary function, output node
    TFUNC_OUT = TFUNC + OUT_NODE  # ternary function, output node


class Func:
    """
    The enumeration class for GP function types.
    """

    TF_START = 0
    IF = 0

    BF_START = 1
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    LOOSE_DIV = 5
    POW = 6
    LOOSE_POW = 7
    MAX = 8
    MIN = 9
    LT = 10
    GT = 11
    LE = 12
    GE = 13

    UF_START = 14
    SIN = 14
    COS = 15
    TAN = 16
    SINH = 17
    COSH = 18
    TANH = 19
    LOG = 20
    LOOSE_LOG = 21
    EXP = 22
    INV = 23
    LOOSE_INV = 24
    NEG = 25
    ABS = 26
    SQRT = 27
    LOOSE_SQRT = 28

    END = 29


FUNCS = [
    Func.IF,
    Func.ADD,
    Func.SUB,
    Func.MUL,
    Func.DIV,
    Func.LOOSE_DIV,
    Func.POW,
    Func.LOOSE_POW,
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
    Func.LOOSE_LOG,
    Func.EXP,
    Func.INV,
    Func.LOOSE_INV,
    Func.NEG,
    Func.ABS,
    Func.SQRT,
    Func.LOOSE_SQRT,
]

FUNCS_NAMES = [
    "if",
    "+",
    "-",
    "*",
    "/",
    "loose_div",
    "pow",
    "loose_pow",
    "max",
    "min",
    "<",
    ">",
    "<=",
    ">=",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "loose_log",
    "exp",
    "inv",
    "loose_inv",
    "neg",
    "abs",
    "sqrt",
    "loose_sqrt",
]

FUNCS_DISPLAY = [
    "if",
    "+",
    "−",
    "*",
    "/",
    "loose_div",
    "pow",
    "loose_pow",
    "max",
    "min",
    "<",
    ">",
    "<=",
    ">=",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "loose_log",
    "exp",
    "inv",
    "loose_inv",
    "neg",
    "abs",
    "sqrt",
    "loose_sqrt",
]


def _loose_quotient(x, y):
    """``x / y`` with the divisor clamp of the "loose" operators applied.

    ``loose_div`` and ``loose_inv`` replace a divisor whose magnitude is at most
    ``DELTA`` by ``copy_sign(DELTA, y)`` *before* dividing (ENGINE_SPEC section 3,
    ids 5 and 24), so ``x / 0`` is a finite ``1e9 * x`` and never an unbounded
    quotient. In float32 - which is what the engine evaluates in - ``1 / DELTA``
    rounds to exactly ``MAX_VAL = 1e9``, so the clamped case is rendered as
    ``x * copy_sign(MAXVAL, y)`` and keeps the clean ``1e9`` placeholder (``x / 1e-9``
    would render ``1e9 * x`` as 999999999.9999999 in float64). Symbolic and larger
    divisors are left as a plain ``x / y``.
    """
    if y.is_number and abs(y) <= DELTA:
        # The engine keeps the sign of the clamped divisor, but sympy has no negative
        # zero (``sympify(-0.0)`` is ``Zero``), so an exact zero divisor renders
        # positive here.
        return x * (sp.Float(MAXVAL) if y >= 0 else sp.Float(-MAXVAL))
    return x / y


class LooseDiv(sp.Function):
    @classmethod
    def eval(cls, x, y):
        return _loose_quotient(x, y)


class LooseInv(sp.Function):
    @classmethod
    def eval(cls, x):
        return _loose_quotient(1, x)


class LooseLog(sp.Function):
    @classmethod
    def eval(cls, x):
        if x == 0:
            # ENGINE_SPEC section 3 id 21: the zero case is ``-MAX_VAL``, i.e. -1e9
            return sp.S(-MAXVAL)
        return sp.log(sp.Abs(x))


SYMPY_MAP = {
    Func.IF: lambda x, y, z: sp.Piecewise((y, x > 0), (z, True)),
    Func.ADD: sp.Add,
    Func.SUB: lambda x, y: x - y,
    Func.MUL: sp.Mul,
    Func.DIV: lambda x, y: x / y,
    Func.LOOSE_DIV: LooseDiv,
    Func.POW: sp.Pow,
    Func.LOOSE_POW: lambda x, y: sp.Pow(sp.Abs(x), y),
    Func.MAX: sp.Max,
    Func.MIN: sp.Min,
    Func.LT: sp.Lt,
    Func.GT: sp.Gt,
    Func.LE: sp.Le,
    Func.GE: sp.Ge,
    Func.SIN: sp.sin,
    Func.COS: sp.cos,
    Func.TAN: sp.tan,
    Func.SINH: sp.sinh,
    Func.COSH: sp.cosh,
    Func.TANH: sp.tanh,
    Func.LOG: sp.log,
    Func.LOOSE_LOG: LooseLog,
    Func.EXP: sp.exp,
    Func.INV: lambda x: 1 / x,
    Func.LOOSE_INV: LooseInv,
    Func.NEG: lambda x: -x,
    Func.ABS: sp.Abs,
    Func.SQRT: sp.sqrt,
    Func.LOOSE_SQRT: lambda x: sp.sqrt(sp.Abs(x)),
    Func.END: None,
}


def to_numpy(li):
    for idx, e in enumerate(li):
        if type(e) == torch.Tensor:
            li[idx] = e.cpu().numpy()
    return li


def dict2prob(prob_dict):
    # Probability Dictionary to Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    prob = torch.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert (
            key in FUNCS_NAMES
        ), f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        prob[idx] = val

    # normalize
    prob = prob / prob.sum()

    return prob


def check_tensor(x):
    if not isinstance(x, torch.Tensor):
        # The old fallback built a tensor and then returned None, silently turning every
        # ``x = check_tensor(x)`` call into ``x = None``; fail loudly instead.
        raise TypeError(
            f"check_tensor expects a torch.Tensor, but got {type(x).__name__}; "
            "convert lists / numpy arrays with torch.tensor(...) first"
        )
    return x.to("cuda").detach().requires_grad_(False)


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


def randint(size, low, high, dtype=torch.int32, device="cuda", requires_grad=False):
    random = low + torch.rand(size, device=device, requires_grad=requires_grad) * (
        high - low
    )
    return random.to(dtype=dtype)


def inspect_function(func):
    assert isinstance(func, Callable), "formula should be Callable"
    sig = inspect.signature(func)
    parameters = sig.parameters
    assert len(parameters) > 0, "formula should have at least one parameter"
    for name, param in parameters.items():
        assert (
            param.default is inspect.Parameter.empty
        ), f"formula should not have default parameters, but got {name}={param.default}"

    return list(parameters.keys())
