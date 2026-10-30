"""Torch-free access to the tree layer's ids: ``MAX_STACK``, ``MAX_FULL_DEPTH``,
``NType``, ``Func`` and ``FUNCS_NAMES``.

``../tree/utils.py`` is the repository's single source of truth for these
values and this subtree does not redefine them.  It cannot simply be imported
either: ``tree/utils.py`` starts with ``import torch`` (and ``import sympy``),
and importing anything below ``evogp.tree`` executes that package's
``__init__.py``, which pulls in the whole torch stack, the evoxc engines and
``networkx``.  That would make this deliberately torch-free subtree torch-bound
at *import* time and would break ``import evogp.gp`` on a plain checkout that
has numpy but neither torch nor evoxc.

So the values are READ OUT OF THAT SAME FILE instead of being re-typed: the
source is parsed with :mod:`ast` and the constants are evaluated from the
parsed expressions (they are plain integer/string literals and integer
arithmetic over earlier names of the same class).  A change upstream therefore
propagates automatically; ``tests/test_gp_stack_constants.py`` additionally
pins every value against the real ``evogp.tree.utils`` module, so a drift (or a
refactor that makes an expression non-constant) fails loudly instead of
silently changing the encoding.

Public surface: ``MAX_STACK``, ``MAX_FULL_DEPTH``, ``NType``, ``Func``,
``FUNCS_NAMES`` (the tree layer's, verbatim) plus the two derived tables this
stack indexes by function id -- ``FUNC_ARITY`` and ``FUNC_TYPE`` -- and the
``FUNC_IDS`` name -> id mapping.
"""

from __future__ import annotations

import ast
import pathlib
import types

__all__ = [
    "MAX_STACK",
    "MAX_FULL_DEPTH",
    "NType",
    "Func",
    "FUNCS_NAMES",
    "FUNC_IDS",
    "FUNC_ARITY",
    "FUNC_TYPE",
    "SOURCE_PATH",
]

#: The file every value below is read from (never written here).
SOURCE_PATH = pathlib.Path(__file__).resolve().parent.parent / "tree" / "utils.py"

#: Module-level integer assignments we mirror.
_LIMITS = ("MAX_STACK", "MAX_FULL_DEPTH")
#: Module-level tuple/list assignments we mirror.
_TABLES = ("FUNCS_NAMES",)
#: Constant-only classes we mirror.
_CLASSES = ("NType", "Func")


def _evaluate(node: ast.expr, namespace: dict, what: str):
    """Evaluate one constant expression of ``tree/utils.py``.

    The expression is compiled straight from the parsed source, with an empty
    builtin namespace, so only literals and the names already defined earlier
    in the same class body can be referenced (``UFUNC_OUT = UFUNC + OUT_NODE``
    is the only shape that occurs).  Anything else -- a call, an import, a
    non-constant -- raises with the offending name and file.
    """
    try:
        return eval(  # noqa: S307 - constant expression of our own source file
            compile(ast.Expression(body=node), str(SOURCE_PATH), "eval"),
            {"__builtins__": {}},
            dict(namespace),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        raise RuntimeError(
            f"evogp.gp cannot read {what} from {SOURCE_PATH}: the expression "
            f"{ast.unparse(node)!r} is not a plain constant. This stack reads "
            "the tree layer's ids out of that file instead of importing it "
            "(importing it requires torch); restore the constant definition "
            "or teach src/evogp/gp/_constants.py the new shape."
        ) from exc


def _parse_source():
    """``(limits, classes, tables)`` of ``tree/utils.py``."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    limits: dict[str, int] = {}
    tables: dict[str, tuple] = {}
    classes: dict[str, dict] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in _LIMITS:
                    limits[target.id] = int(_evaluate(node.value, {}, target.id))
                elif target.id in _TABLES:
                    tables[target.id] = tuple(_evaluate(node.value, {}, target.id))
        elif isinstance(node, ast.ClassDef) and node.name in _CLASSES:
            namespace: dict = {}
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                value = _evaluate(stmt.value, namespace, f"{node.name}.{ast.unparse(stmt)}")
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = value
            classes[node.name] = namespace

    missing = [n for n in _LIMITS if n not in limits]
    missing += [n for n in _TABLES if n not in tables]
    missing += [n for n in _CLASSES if n not in classes]
    if missing:
        raise RuntimeError(
            f"evogp.gp: {SOURCE_PATH} no longer defines {', '.join(missing)}; "
            "src/evogp/gp/_constants.py reads the tree layer's ids from that "
            "file and must be updated together with it."
        )
    return limits, classes, tables


def _materialize(name: str, attributes: dict) -> type:
    """Turn a parsed constant class body into a real (constant) class."""
    namespace = dict(attributes)
    namespace["__module__"] = __name__
    namespace["__slots__"] = ()
    namespace["__doc__"] = (
        f"Mirror of ``evogp.tree.utils.{name}`` (read from {SOURCE_PATH.name}); "
        "see src/evogp/gp/_constants.py."
    )
    return types.new_class(name, (), {}, lambda ns: ns.update(namespace))


_limits, _classes, _tables = _parse_source()

#: ``max_tree_len`` upper bound of the tree layer (1024).
MAX_STACK: int = _limits["MAX_STACK"]
#: Number of entries of the depth -> leaf-probability schedule (10).
MAX_FULL_DEPTH: int = _limits["MAX_FULL_DEPTH"]

#: Node-type markers (``VAR``/``CONST``/``UFUNC``/``BFUNC``/``TFUNC`` + ``OUT_NODE``).
NType = _materialize("NType", _classes["NType"])
#: Function ids (``IF``/``ADD``/... /``LOOSE_SQRT``).
Func = _materialize("Func", _classes["Func"])

#: Function names, positional with the ids of :class:`Func`.
FUNCS_NAMES: tuple = _tables["FUNCS_NAMES"]
#: ``name -> id`` (the inverse of :data:`FUNCS_NAMES`).
FUNC_IDS: dict = {name: index for index, name in enumerate(FUNCS_NAMES)}


def _arity(function_id: int) -> int:
    if Func.TF_START <= function_id < Func.BF_START:
        return 3
    if Func.BF_START <= function_id < Func.UF_START:
        return 2
    return 1


def _node_type(function_id: int) -> int:
    if Func.TF_START <= function_id < Func.BF_START:
        return NType.TFUNC
    if Func.BF_START <= function_id < Func.UF_START:
        return NType.BFUNC
    return NType.UFUNC


#: ``FUNC_ARITY[id]`` = number of children of a function node with that id.
FUNC_ARITY: tuple = tuple(_arity(i) for i in range(Func.END))
#: ``FUNC_TYPE[id]`` = the ``NType`` marker of a function node with that id.
FUNC_TYPE: tuple = tuple(_node_type(i) for i in range(Func.END))

if len(FUNCS_NAMES) != Func.END or len(FUNC_ARITY) != Func.END:
    raise RuntimeError(
        f"evogp.gp: FUNCS_NAMES has {len(FUNCS_NAMES)} entries but Func.END is "
        f"{Func.END}; the function ids and the name table must stay positional."
    )
