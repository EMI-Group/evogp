"""Suite-wide hermeticity: the pytest suite runs the CPU reference route.

The default route policy (``dense_engine.resolve_route``) picks dense+iree from
``EVOGP_DENSE_AUTO_MIN_POP`` (64) up when torch sees a device, so an unset
``EVOGP_EVOXC_BACKEND`` on a GPU box would compile iree graphs inside the suite
(slow, and it needs a working HAL driver).  Pinning ``numpy`` here keeps every
test on the deterministic CPU reference route with no device probe side effects;
the device route is exercised by the standalone ``tree/`` CLIs instead.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def hermetic_backend():
    """Pin ``EVOGP_EVOXC_BACKEND=numpy`` unless ``EVOGP_TESTS_ALLOW_DEVICE=1``.

    The pin is a SESSION pin, not a process-wide one: the caller's value (or its
    absence) is restored at teardown, so a later module collected in the same
    process — e.g. the harness suite under ``test/benchmark/evoxc``, which drives
    its own arms and resolves routes per record — never inherits a phantom
    backend it did not ask for.
    """
    if os.environ.get("EVOGP_TESTS_ALLOW_DEVICE", "").strip() == "1":
        yield
        return
    before = os.environ.get("EVOGP_EVOXC_BACKEND")
    os.environ.setdefault("EVOGP_EVOXC_BACKEND", "numpy")
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("EVOGP_EVOXC_BACKEND", None)
        else:
            os.environ["EVOGP_EVOXC_BACKEND"] = before
