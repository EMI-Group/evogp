"""Test-level setup for the ``evogp.gp`` suite.

One job, done at import time (before any test or ``evoxc`` use): put ``src`` on
``sys.path`` so the suite runs from a plain checkout without ``PYTHONPATH``
(the ``evoxc`` checkout still has to be importable -- it is the engine this
stack is built on).

The suite is hermetic without an environment pin: every entry point of this
stack takes its backend explicitly (default ``"numpy"``), so a device box never
compiles an IREE graph while running this suite.
"""

import pathlib
import sys


def _ensure_src_on_path() -> None:
    src = pathlib.Path(__file__).resolve().parents[3]
    if src.name != "src":  # pragma: no cover - layout guard
        raise RuntimeError(f"expected a src-layout parent, computed {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()
