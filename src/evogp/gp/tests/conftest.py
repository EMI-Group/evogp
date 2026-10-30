"""Test-level setup for the ``evogp.gp`` suite.

Two jobs, both done at import time (before any test or ``evoxc`` use):

* pin the tree layer's engine backend to the hermetic numpy reference, the same
  way ``src/evogp/tests/conftest.py`` does for the torch stack, so a device box
  never compiles an IREE graph while running this suite;
* put ``src`` on ``sys.path`` so the suite runs from a plain checkout without
  ``PYTHONPATH`` (the ``evoxc`` checkout still has to be importable -- it is the
  engine this stack is built on).
"""

import os
import pathlib
import sys

os.environ.setdefault("EVOGP_EVOXC_BACKEND", "numpy")


def _ensure_src_on_path() -> None:
    src = pathlib.Path(__file__).resolve().parents[3]
    if src.name != "src":  # pragma: no cover - layout guard
        raise RuntimeError(f"expected a src-layout parent, computed {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()
