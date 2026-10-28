"""Regression tests for the lazy jax/mujoco_playground imports of ``MujocoProblem``.

Purpose
-------
``evogp.problem.mujoco_playground.import_jax_based_package`` is called from
``MujocoProblem.__init__`` and is expected to bind ``jax``, ``jnp``, ``MjxEnv`` and
``registry`` as *module globals* (the module's ``global`` statement). When those
``import`` statements were plain function locals, ``registry.load(...)`` inside
``__init__`` raised ``NameError: name 'registry' is not defined``.

1. ``test_import_jax_based_package_binds_module_globals`` locks the binding itself:
   after the call the four names must be attributes of the module.
2. ``test_mujoco_problem_construction_does_not_raise_nameerror`` is the end-to-end
   smoke: a ``NameError`` from the constructor must FAIL, while an environment
   boundary (the third-party mujoco/mujoco-warp or jax/CuDNN stack not being usable
   here) SKIPS with a reason. Both tests skip cleanly when jax or mujoco_playground
   are absent.

Run from the repository root::

    EVOXC_DISABLE_LOCATIONS=1 PYTHONPATH=src /mnt/local-ssd/bchuang/evox/.venv/bin/python \\
        -m pytest src/evogp/tests/test_mujoco_problem.py -q
"""
from __future__ import annotations

import os

import pytest

MJA_ENV_VAR = "XLA_PYTHON_CLIENT_MEM_FRACTION"


def test_import_jax_based_package_binds_module_globals():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    import evogp.problem.mujoco_playground as mp

    saved = os.environ.get(MJA_ENV_VAR)
    try:
        mp.import_jax_based_package(0.5)
        # optional: the function sets the env var before importing jax
        # assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
    finally:
        if saved is None:
            os.environ.pop(MJA_ENV_VAR, None)
        else:
            os.environ[MJA_ENV_VAR] = saved

    for name in ("jax", "jnp", "registry", "MjxEnv"):
        assert hasattr(mp, name), (
            f"evogp.problem.mujoco_playground is missing module attribute {name!r} — "
            "the lazy imports are not bound as globals"
        )


def test_mujoco_problem_construction_does_not_raise_nameerror():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    from evogp.problem import MujocoProblem

    saved = os.environ.get(MJA_ENV_VAR)
    try:
        problem = MujocoProblem("SwimmerSwimmer6", 1000)
    except NameError as exc:
        raise AssertionError(
            "MujocoProblem.__init__ raised NameError — the lazy jax/mujoco_playground "
            f"imports are not bound as module globals: {exc!r}"
        ) from exc
    except Exception as exc:
        message = str(exc)
        if "WARP" in message or "dnn_support" in message or "CuDNN" in message:
            pytest.skip(
                "environment boundary — the third-party mujoco/jax stack is not fully "
                f"usable here: {exc!r}"
            )
        raise
    finally:
        if saved is None:
            os.environ.pop(MJA_ENV_VAR, None)
        else:
            os.environ[MJA_ENV_VAR] = saved

    assert problem.env is not None
