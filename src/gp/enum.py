class NType:
    """
    The enumeration class for GP node types.

    ```C++
    VAR = 0,   // variable
    CONST = 1, // constant
    UFUNC = 2, // unary function
    BFUNC = 3,  // binary function
    TFUNC = 4, // ternary function
    TYPE_MASK = 0x7F, // node type mask
    OUT_NODE = 1 << 7, // out node flag
    UFUNC_OUT = UFUNC + OUT_NODE, // unary function, output node
    BFUNC_OUT = BFUNC + OUT_NODE,  // binary function, output node
    TFUNC_OUT = TFUNC + OUT_NODE,  // ternary function, output node
    ```
    """

    VAR = 0
    CONST = 1
    UFUNC = 2
    BFUNC = 3
    TFUNC = 4
    TYPE_MASK = 0x7F
    OUT_NODE = 1 << 7
    UFUNC_OUT = UFUNC + OUT_NODE
    BFUNC_OUT = BFUNC + OUT_NODE
    TFUNC_OUT = TFUNC + OUT_NODE


class Func:
    """
    The enumeration class for GP function types.

    ```C++
    IF,  // arity: 3, return if (a > 0) { return b } return c
    ADD, // arity: 2, return a + b
    SUB, // arity: 2, return a - b
    MUL, // arity: 2, return a * b
    DIV, // arity: 2, if (b == 0) { b = DELTA } return a / b
    MAX, // arity: 2, if (a > b) { return a } return b
    MIN, // arity: 2, if (a < b) { return a } return b
    LT,  // arity: 2, if (a < b) { return 1 } return 0
    GT,  // arity: 2, if (a > b) { return 1 } return 0
    LE,  // arity: 2, if (a <= b) { return 1 } return 0
    GE,  // arity: 2, if (a >= b) { return 1 } return 0
    SIN, // arity: 1, return sin a
    COS, // arity: 1, return cos a
    SINH,// arity: 1, return sinh a
    COSH,// arity: 1, return cosh a
    LOG, // arity: 1, return log a
    EXP, // arity: 1, return exp a
    INV, // arity: 1, if (a == 0) { a = DELTA } return 1 / a
    NEG, // arity: 1, return -a
    POW2,// arity: 1, return a * a
    POW3,// arity: 1, return a * a * a
    SQRT,// arity: 1, return sqrt(a)
    ```
    """

    IF = 0

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    MAX = 5
    MIN = 6
    LT = 7
    GT = 8
    LE = 9
    GE = 10

    SIN = 11
    COS = 12
    SINH = 13
    COSH = 14
    LOG = 15
    EXP = 16
    INV = 17
    NEG = 18
    POW2 = 19
    POW3 = 20
    SQRT = 21


FUNCS = [
    Func.IF,
    Func.ADD,
    Func.SUB,
    Func.MUL,
    Func.DIV,
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
    Func.POW2,
    Func.POW3,
    Func.SQRT
]

FUNCS_NAMES = [
    "if",
    "+",
    "-",
    "*",
    "/",
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
    "^2",
    "^3",
    "√"
]


def tree2str(tree):
    return to_string(tree.node_types, tree.node_vals)


def to_string(node_type, node_val):
    node_type, node_val = list(node_type), list(node_val)
    res = ""
    for t, v in zip(node_type, node_val):
        if t == NType.VAR:
            res += f"in[{v}]"
        elif t == NType.CONST:
            res += f"{v}"
        elif t == NType.UFUNC or t == NType.BFUNC:
            res += f"{FUNCS_NAMES[int(v)]}"

        res += " "
    return res

# FUNC_TYPES = jnp.array([
#     Type.BFUNC,
#     Type.BFUNC,
#     Type.BFUNC,
#     Type.BFUNC,
#     Type.UFUNC,
#     Type.UFUNC,
#     Type.UFUNC,
#     Type.UFUNC,
#     Type.UFUNC,
#     Type.BFUNC,
#     Type.BFUNC,
#     Type.UFUNC,
#     Type.UFUNC,
#     Type.UFUNC
# ])
#
# CHILDREN_NUM = jnp.array([
#     0, 0, 1, 2
# ])
#
# MAX_VAL = 1000
# MIN_VAL = -1000
#
#
# def ADD(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     return a + b, s
#
#
# def SUB(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     return a - b, s
#
#
# def MUL(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     return a * b, s
#
#
# def DIV(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     res = a / b
#
#     res = jnp.where(jnp.isnan(res), 0, res)
#     res = jnp.clip(res, MIN_VAL, MAX_VAL)
#
#     return res, s
#
#
# def TAN(s: Stack):
#     a, s = s.pop()
#
#     res = jnp.tan(a)
#     res = jnp.where(jnp.isnan(res), 0, res)
#     res = jnp.clip(res, MIN_VAL, MAX_VAL)
#
#     return res, s
#
#
# def SIN(s: Stack):
#     a, s = s.pop()
#     return jnp.sin(a), s
#
#
# def COS(s: Stack):
#     a, s = s.pop()
#     return jnp.cos(a), s
#
#
# def LOG(s: Stack):
#     a, s = s.pop()
#
#     res = jnp.log(a)
#     res = jnp.where(jnp.isnan(res), 0, res)
#     res = jnp.clip(res, MIN_VAL, MAX_VAL)
#
#     return res, s
#
#
# def EXP(s: Stack):
#     a, s = s.pop()
#
#     res = jnp.exp(a)
#     res = jnp.clip(res, MIN_VAL, MAX_VAL)
#
#     return res, s
#
#
# def MAX(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     return jnp.maximum(a, b), s
#
#
# def MIN(s: Stack):
#     a, s = s.pop()
#     b, s = s.pop()
#     return jnp.minimum(a, b), s
#
#
# def INV(s: Stack):
#     a, s = s.pop()
#
#     res = 1 / a
#     res = jnp.where(jnp.isnan(res), 0, res)
#     res = jnp.clip(res, MIN_VAL, MAX_VAL)
#
#     return res, s
#
#
# def NEG(s: Stack):
#     a, s = s.pop()
#     return -a, s
#
#
# def ABS(s: Stack):
#     a, s = s.pop()
#     return jnp.abs(a), s


# FUNCS = [
#     ADD,
#     SUB,
#     MUL,
#     DIV,
#     TAN,
#     SIN,
#     COS,
#     LOG,
#     EXP,
#     MAX,
#     MIN,
#     INV,
#     NEG,
#     ABS
# ]
