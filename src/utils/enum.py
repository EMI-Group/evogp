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
    "if", # 0
    "+", # 1
    "-", # 2
    "*", # 3
    "/", # 4
    "pow", # 5
    "max", # 6
    "min", # 7
    "<", # 8
    ">", # 9
    "<=", # 10
    ">=", # 11
    "sin", # 12
    "cos", # 13
    "tan", # 14
    "sinh", # 15
    "cosh", # 16
    "tanh", # 17
    "log", # 18
    "exp", # 19
    "inv", # 20
    "neg", # 21
    "abs", # 22
    "sqrt", # 23
]
