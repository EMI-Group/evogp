"""Purity tests: the ``evogp.gp`` subtree stays torch-free and import-light.

Three independent guards, because they can fail independently:

1. **Source grep** -- no module of this subtree (``tests/`` excluded) may contain
   an ``import torch`` / ``from torch`` / ``import evox`` / ``from evox``.
   Torch may appear in this subtree only inside tests, as an oracle.
2. **Import in a clean subprocess** -- ``import evogp.gp`` and its modules must
   work with only ``src`` on ``PYTHONPATH`` and must not pull torch, ``evox`` or
   even ``evoxc`` into ``sys.modules``: the graphs need evoxc at *call* time, not
   at import time, which is what makes the package importable from a plain
   checkout.
3. **Table invariants** -- the ids the stack indexes by live in the torch-free
   single source of truth :mod:`evogp._tables` (re-exported, not copied, by
   ``_constants.py``), so they are pinned here directly: the published values,
   the positional agreement of ``Func`` / ``FUNCS_NAMES`` / ``FUNC_ARITY`` /
   ``FUNC_TYPE``, the limits and the ``NType`` markers -- plus the two
   structural facts that make the module usable from anywhere: the re-export
   hands out the very same objects, and the table module itself imports
   NOTHING (no torch, no numpy, not even stdlib).
"""

import os
import pathlib
import subprocess
import sys

import pytest

from evogp.gp import _constants

SUBTREE = pathlib.Path(__file__).resolve().parents[1]
SRC = SUBTREE.parents[1]

#: Imports this subtree must never contain (``tests/`` is excluded on purpose).
FORBIDDEN_IMPORTS = ("torch", "evox")


@pytest.mark.parametrize(
    "path", sorted(SUBTREE.glob("*.py")), ids=lambda path: path.name
)
def test_subtree_sources_do_not_import_torch_or_evox(path):
    source = path.read_text(encoding="utf-8")
    for module in FORBIDDEN_IMPORTS:
        for token in (f"import {module}\n", f"import {module} ", f"from {module} "):
            assert token not in source, (
                f"{path.name} must stay {module}-free (found {token.strip()!r}); "
                f"{module} is only allowed in tests/ as an oracle"
            )


def test_the_src_layout_root_is_what_the_subprocess_gets():
    assert SRC.name == "src", SRC
    assert (SRC / "evogp" / "gp" / "__init__.py").is_file()


def test_import_works_without_torch_evox_or_evoxc():
    code = (
        "import sys\n"
        "import evogp.gp, evogp.gp.encoding, evogp.gp.state, evogp.gp.config\n"
        "loaded = sorted(\n"
        "    name for name in ('torch', 'evox', 'evoxc') if name in sys.modules\n"
        ")\n"
        "assert not loaded, f'importing evogp.gp pulled in {loaded}'\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)  # deliberately NOT the evoxc checkout
    env.pop("EVOXC_BACKEND", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok", result.stdout


def test_a_public_entry_point_without_evoxc_raises_a_clear_error():
    code = (
        "from evogp.gp import GPTreeConfig, init_population\n"
        "config = GPTreeConfig(max_nodes=8, input_len=2, using_funcs=('+',),\n"
        "                      max_layer_cnt=1, const_range=(-1.0, 1.0), sample_cnt=2)\n"
        "try:\n"
        "    init_population(config, 0, pop_size=2)\n"
        "except RuntimeError as exc:\n"
        "    assert 'evoxc' in str(exc), str(exc)\n"
        "    print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok", result.stdout


def test_function_ids_names_arity_and_type_are_positional():
    """At least one of each arity is enough: they all come from ``Func``."""
    assert _constants.Func.END == 29
    assert (
        len(_constants.FUNCS_NAMES)
        == len(_constants.FUNC_ARITY)
        == len(_constants.FUNC_TYPE)
        == _constants.Func.END
    )

    # ids -> names (a wrong name at an id is a silent semantic bug: the id is
    # what the opcode channel carries, the name is what users write).
    assert _constants.FUNCS_NAMES[_constants.Func.IF] == "if"
    assert _constants.FUNCS_NAMES[_constants.Func.ADD] == "+"
    assert _constants.FUNCS_NAMES[_constants.Func.LOOSE_SQRT] == "loose_sqrt"

    # ids -> arity / node type, one function of each arity class.
    assert _constants.FUNC_ARITY[_constants.Func.IF] == 3
    assert _constants.FUNC_ARITY[_constants.Func.ADD] == 2
    assert _constants.FUNC_ARITY[_constants.Func.SIN] == 1
    assert _constants.FUNC_TYPE[_constants.Func.IF] == _constants.NType.TFUNC
    assert _constants.FUNC_TYPE[_constants.Func.ADD] == _constants.NType.BFUNC
    assert _constants.FUNC_TYPE[_constants.Func.SIN] == _constants.NType.UFUNC


def test_limits_node_type_markers_and_multi_output_wide():
    from evogp import _tables

    assert _constants.MAX_STACK == 1024
    assert _constants.MAX_FULL_DEPTH == 10
    assert _tables.MAX_OUT_LEN == 8
    assert _tables.OUT_NODE == 1 << 7
    assert _tables.OUT_NODE == _constants.NType.OUT_NODE

    ntype = _constants.NType
    assert (
        ntype.VAR,
        ntype.CONST,
        ntype.UFUNC,
        ntype.BFUNC,
        ntype.TFUNC,
        ntype.TYPE_MASK,
    ) == (0, 1, 2, 3, 4, 0x7F)


def test_constants_reexports_the_table_module_objects():
    """``_constants`` must not own a copy -- the same objects, not equal ones."""
    from evogp import _tables

    assert _constants.FUNCS_NAMES is _tables.FUNCS_NAMES
    assert _constants.FUNC_ARITY is _tables.FUNC_ARITY
    assert _constants.FUNC_TYPE is _tables.FUNC_TYPE
    assert _constants.Func is _tables.Func
    assert _constants.NType is _tables.NType


def test_the_table_module_imports_nothing():
    """The single source of truth is plain literals: no torch, numpy or stdlib."""
    import evogp._tables as tables

    source = "\n" + pathlib.Path(tables.__file__).read_text(encoding="utf-8")
    for token in ("\nimport ", "\nfrom "):
        assert token not in source, (
            f"evogp/_tables.py must stay import-free (found {token.strip()!r} "
            "at the start of a line); it is read by the torch-free stack and by "
            "a device kernel's host glue, neither of which may pull a dependency"
        )
