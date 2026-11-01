"""Test wiring of the ``evogp.fused`` net: the oracle by PATH, the device gate, knobs.

Three jobs, and no test logic of its own:

1. **The oracle.**  The independent plain-torch reference
   (``test/oracle/reference_eval.py``) is DEV tooling: it is shipped nowhere,
   imported by no library module, and deliberately NOT on ``PYTHONPATH``.  This
   conftest appends ``<repo>/test/oracle`` to ``sys.path`` (parent index 4, a
   src-layout guard fails loudly if that is not the repository root) and exposes
   it as the ``oracle`` fixture, which does the ``importorskip`` -- so a case that
   needs the oracle skips (never fails, never silently compares against nothing)
   when the directory is missing.  The three ``dispatch``/``cache`` cases need no
   oracle at all and keep running there.

2. **The device gate.**  The route needs a CUDA/HIP device visible to torch, an
   importable ``triton`` and a WRITABLE Triton JIT cache.  The session-scoped
   autouse ``fused_route_guard`` fixture answers all three once, per session:
   it skips the whole suite (``pytest.skip`` in an autouse session fixture) when
   the device or ``triton`` is missing, and otherwise makes sure
   ``TRITON_CACHE_DIR`` points at a writable directory -- the ambient value when
   that is writable, else a session ``tmp_path`` (a read-only ``$HOME`` would
   otherwise kill the first launch).  It runs BEFORE the module-scoped ``_warm_jit``
   fixture, because a session-scoped fixture is instantiated first.

3. **Knobs.**  The ``EVOGP_FUSED_*`` knobs are read per call, so an ambient value
   in the developer's shell could silently change what a case measures; the autouse
   ``fused_knobs`` fixture removes them (and ``EVOGP_EVALUATION_BACKEND`` is gone
   with the ``tree/`` stack -- nothing here reads it any more).  A case that wants
   a knob sets it itself, with ``monkeypatch``.

The shared builders (the legacy triple from nested specs, the data block, the
comparison assertion and the decline/launch proofs) live in
:mod:`evogp.fused.tests.support`; they are re-exported here so a case can also
reach them as ``conftest.population`` etc.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import sys

import pytest

# --- 1. the oracle by path ------------------------------------------------------ #
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if not (_REPO_ROOT / "src" / "evogp").is_dir():  # pragma: no cover - layout guard
    raise RuntimeError(
        "expected src/evogp/fused/tests/conftest.py, i.e. a repository root at "
        f"parents[4] holding src/evogp, computed {_REPO_ROOT}"
    )
_ORACLE_DIR = _REPO_ROOT / "test" / "oracle"
_ORACLE_FILE = _ORACLE_DIR / "reference_eval.py"
if _ORACLE_DIR.is_dir() and str(_ORACLE_DIR) not in sys.path:
    sys.path.append(str(_ORACLE_DIR))

_ORACLE_MODULE_NAME = "_evogp_fused_oracle"


def load_oracle():
    """The oracle module, or ``None`` when ``test/oracle/`` is not there.

    Imported by name when ``test/oracle`` is on ``sys.path`` (the normal path) and
    loaded from the file otherwise, so a copy of this suite moved elsewhere still
    finds the repository's oracle.
    """
    if _ORACLE_DIR.is_dir():
        try:
            return importlib.import_module("reference_eval")
        except ImportError:
            pass
    if not _ORACLE_FILE.is_file():
        return None
    spec = importlib.util.spec_from_file_location(_ORACLE_MODULE_NAME, _ORACLE_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ORACLE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def oracle():
    """The independent plain-torch oracle (``test/oracle/reference_eval.py``)."""
    module = load_oracle()
    if module is None:
        pytest.skip(
            "the plain-torch oracle is not at <repo>/test/oracle/reference_eval.py "
            "(dev-only tooling, see test/oracle/CONTEXT.md)"
        )
    return module


# --- 2. the device / triton / cache gate ---------------------------------------- #
def _writable_cache_dir() -> pathlib.Path:
    """A writable Triton cache directory (the ambient one, or a fresh temp dir)."""
    ambient = os.environ.get("TRITON_CACHE_DIR", "").strip()
    if ambient:
        candidate = pathlib.Path(ambient)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except OSError:
            pass
    import tempfile

    return pathlib.Path(tempfile.mkdtemp(prefix="evogp-fused-triton-"))


def skip_reasons():
    """Why this box cannot run the route at all (an empty list = run the suite)."""
    reasons = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"torch is not importable ({exc!r})"]
    if not torch.cuda.is_available():
        reasons.append("no CUDA/HIP device is visible to torch")
    if importlib.util.find_spec("triton") is None:
        reasons.append("triton is not importable")
    return reasons


@pytest.fixture(scope="session", autouse=True)
def fused_route_guard():
    """Skip the suite where the route cannot exist, and make the JIT cache writable."""
    reasons = skip_reasons()
    if reasons:
        pytest.skip("; ".join(reasons))
    cache = _writable_cache_dir()
    previous = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    try:
        yield cache
    finally:
        if previous is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous


# --- 3. the knobs --------------------------------------------------------------- #
#: Every knob the route reads per call.  A case that wants one sets it explicitly.
_FUSED_KNOBS = (
    "EVOGP_FUSED_BLOCK_ROWS",
    "EVOGP_FUSED_NUM_WARPS",
    "EVOGP_FUSED_MAX_SCRATCH_MB",
    "EVOGP_FUSED_HOST_MEMO",
    "EVOGP_FUSED_DISPATCH",
)


@pytest.fixture(autouse=True)
def fused_knobs(monkeypatch):
    """Drop the ambient ``EVOGP_FUSED_*`` knobs for the duration of a case."""
    for name in _FUSED_KNOBS:
        monkeypatch.delenv(name, raising=False)
    yield


# --- the shared builders (re-exported; the definitions live in support.py) -------- #
from support import (  # noqa: E402  (after the path wiring above, by design)
    DEVICE,
    assert_same,
    chain,
    data,
    declined,
    deltas,
    encode,
    full_ternary,
    function_id,
    oracle_loss,
    oracle_value,
    population,
    rows_of,
    served,
    tagged,
    triple,
)

__all__ = [
    "DEVICE",
    "assert_same",
    "chain",
    "data",
    "declined",
    "deltas",
    "encode",
    "full_ternary",
    "function_id",
    "load_oracle",
    "oracle_loss",
    "oracle_value",
    "population",
    "rows_of",
    "served",
    "skip_reasons",
    "tagged",
    "triple",
]
