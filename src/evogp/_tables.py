"""Torch-free single source of truth for the node-type / function ids.

These are the ids shared by :mod:`evogp.gp` (the functional GP stack) and
:mod:`evogp.fused` (the device-native evaluation route): the node-type markers
of a population's ``node_type`` channel, the function ids of its function
channel, and the tables indexed by those ids.

Nothing below this module may import torch, numpy, ``evox`` or ``evoxc``, and
this module itself imports NOTHING (stdlib included) -- it is plain literals
and integer arithmetic over earlier names, so it can be read by a torch-free
stack, a device kernel's host glue, or a standalone table dump with the same
result.  ``evogp.gp._constants`` is a thin re-export of this module, kept only
so the ``evogp.gp`` surface (and its documentation) keeps its old import path.

Every id is POSITIONAL across :class:`Func`, :data:`FUNCS`,
:data:`FUNCS_NAMES`, :data:`FUNC_ARITY` and :data:`FUNC_TYPE`, so adding an
operator touches all of them (and the module-level length checks below fail
loudly if one is forgotten).
"""

DELTA = 1e-9

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
    OUT_NODE = 1 << 7  # out node flag
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


#: Every function id, positional with :class:`Func`.
FUNCS = (
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
)

#: Function names, positional with the ids of :class:`Func`.
FUNCS_NAMES = (
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
)

#: ``name -> id`` (the inverse of :data:`FUNCS_NAMES`).
FUNC_IDS = {name: index for index, name in enumerate(FUNCS_NAMES)}


def _arity(function_id):
    """Number of children of a function node with that id (1, 2 or 3)."""
    if Func.TF_START <= function_id < Func.BF_START:
        return 3
    if Func.BF_START <= function_id < Func.UF_START:
        return 2
    return 1


def _node_type(function_id):
    """The :class:`NType` marker of a function node with that id."""
    if Func.TF_START <= function_id < Func.BF_START:
        return NType.TFUNC
    if Func.BF_START <= function_id < Func.UF_START:
        return NType.BFUNC
    return NType.UFUNC


#: ``FUNC_ARITY[id]`` = number of children of a function node with that id.
FUNC_ARITY = tuple(_arity(i) for i in range(Func.END))
#: ``FUNC_TYPE[id]`` = the ``NType`` marker of a function node with that id.
FUNC_TYPE = tuple(_node_type(i) for i in range(Func.END))

#: The widest multi-output tree the fused Triton route serves.  The multi-output
#: kernel accumulates into a ``(pop, rows, out_len)`` prediction buffer, so the
#: width is its tile/register budget rather than a protocol limit; the fused
#: route is the only consumer, and a wider call declines naming this value.
MAX_OUT_LEN = 8

#: ``NType.OUT_NODE`` as a plain int (one spelling for every engine).
OUT_NODE = int(NType.OUT_NODE)

if len(FUNCS) != Func.END or len(FUNCS_NAMES) != Func.END:
    raise RuntimeError(
        f"evogp._tables: FUNCS has {len(FUNCS)} entries and FUNCS_NAMES has "
        f"{len(FUNCS_NAMES)} but Func.END is {Func.END}; the function ids and "
        "the id/name tables must stay positional."
    )
if len(FUNC_ARITY) != Func.END or len(FUNC_TYPE) != Func.END:
    raise RuntimeError(
        f"evogp._tables: FUNC_ARITY has {len(FUNC_ARITY)} entries and FUNC_TYPE "
        f"has {len(FUNC_TYPE)} but Func.END is {Func.END}; the derived tables "
        "must stay positional with Func."
    )

__all__ = [
    "DELTA",
    "MAX_STACK",
    "MAX_FULL_DEPTH",
    "NType",
    "Func",
    "FUNCS",
    "FUNCS_NAMES",
    "FUNC_IDS",
    "FUNC_ARITY",
    "FUNC_TYPE",
    "MAX_OUT_LEN",
    "OUT_NODE",
]
