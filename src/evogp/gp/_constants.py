"""The ``evogp.gp`` view of the ids: ``MAX_STACK``, ``MAX_FULL_DEPTH``,
``NType``, ``Func``, ``FUNCS_NAMES`` and the two id-indexed tables.

The values themselves live in :mod:`evogp._tables` -- the torch-free single
source of truth for the node-type/function ids shared by
:mod:`evogp.gp` and :mod:`evogp.fused`.  This module only re-exports them, so
the ``evogp.gp`` surface keeps its historical import path
(``from evogp.gp._constants import Func``) without owning a copy: there is
nothing to keep in sync and no source file is parsed.

Importing :mod:`evogp._tables` pulls in NOTHING -- not torch, not numpy, not
``evox``/``evoxc``, not even a stdlib module -- which is what keeps this
subtree importable from a plain checkout (``tests/test_gp_stack_purity.py``).

Public surface: ``MAX_STACK``, ``MAX_FULL_DEPTH``, ``NType``, ``Func``,
``FUNCS_NAMES`` plus the two derived tables this stack indexes by function id
-- ``FUNC_ARITY`` and ``FUNC_TYPE`` -- and the ``FUNC_IDS`` name -> id mapping.
"""

from __future__ import annotations

from .._tables import (
    FUNC_ARITY,
    FUNC_IDS,
    FUNC_TYPE,
    FUNCS_NAMES,
    MAX_FULL_DEPTH,
    MAX_STACK,
    Func,
    NType,
)

__all__ = [
    "MAX_STACK",
    "MAX_FULL_DEPTH",
    "NType",
    "Func",
    "FUNCS_NAMES",
    "FUNC_IDS",
    "FUNC_ARITY",
    "FUNC_TYPE",
]
