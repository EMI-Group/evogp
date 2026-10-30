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
3. **Constants parity** -- the ids the stack reads out of ``../tree/utils.py``
   (see ``_constants.py``) must equal the real ones, and the derived arity/type
   tables must equal the tree layer's own ``_func_tables``.
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


def test_constants_equal_the_tree_layer_values():
    utils = pytest.importorskip(
        "evogp.tree.utils", reason="the reference values need torch"
    )
    assert _constants.MAX_STACK == utils.MAX_STACK
    assert _constants.MAX_FULL_DEPTH == utils.MAX_FULL_DEPTH
    assert _constants.FUNCS_NAMES == tuple(utils.FUNCS_NAMES)

    for name in ("NType", "Func"):
        mirror = getattr(_constants, name)
        reference = getattr(utils, name)
        expected = {
            key: value
            for key, value in vars(reference).items()
            if not key.startswith("_")
        }
        observed = {
            key: value for key, value in vars(mirror).items() if not key.startswith("_")
        }
        assert observed == expected, name


def test_arity_and_type_tables_equal_the_tree_layers_own_tables():
    torch = pytest.importorskip("torch", reason="the reference tables need torch")
    generation = pytest.importorskip("evogp.tree.generation")
    arity, node_type = generation._func_tables(torch.device("cpu"))  # noqa: SLF001
    assert tuple(int(v) for v in arity.tolist()) == _constants.FUNC_ARITY
    assert tuple(int(v) for v in node_type.tolist()) == _constants.FUNC_TYPE
