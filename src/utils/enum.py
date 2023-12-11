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

    # ternary
    IF = 0  # if (a > 0) { return b } return c

    # binary
    ADD = 1  # return a + b
    SUB = 2  # return a - b
    MUL = 3  # return a * b
    DIV = 4  # if (|b| < DELTA) { return a / DELTA * sign(b) } return a / b
    POW = 5  # |a|^b
    MAX = 6  # if (a > b) { return a } return b
    MIN = 7  # if (a < b) { return a } return b
    LT = 8  # if (a < b) { return 1 } return -1
    GT = 9  # if (a > b) { return 1 } return -1
    LE = 10  # if (a <= b) { return 1 } return -1
    GE = 11  # if (a >= b) { return 1 } return -1

    # unary
    SIN = 12  # return sin(a)
    COS = 13  # return cos(a)
    SINH = 14  # return sinh(a)
    COSH = 15  # return cosh(a)
    LOG = 16  # return if (a == 0) { return -MAX_VAL } return log(|a|)
    EXP = 17  # return min(exp(a), MAX_VAL)
    INV = 18  # if (|a| < DELTA) { return 1 / DELTA * sign(a) } return 1 / a
    NEG = 19  # return -a
    ABS = 20  # return |a|
    SQRT = 21  # return sqrt(|a|)


# Total Function Pool
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
    Func.SINH,
    Func.COSH,
    Func.LOG,
    Func.EXP,
    Func.INV,
    Func.NEG,
    Func.ABS,
    Func.SQRT,
]


# String Representation of Functions
FUNCS_NAMES = [
    "if",
    "+",
    "-",
    "*",
    "/",
    "pow",
    "max",
    "min",
    "<",
    ">",
    "<=",
    ">=",
    "sin",
    "cos",
    "sinh",
    "cosh",
    "log",
    "exp",
    "inv",
    "neg",
    "abs",
    "sqrt",
]
