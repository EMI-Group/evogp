"""Thin driver: the harness GATE (``check.py``) with the two FUSED arms registered.

``check.py`` decides which arms a result directory must contain from
``campaign.json`` and from ``common.EVOXC_ARMS`` (``is_evoxc_arm``), and its
``--expect-arms`` default is the arm list the driver record carries.  A directory
written by :mod:`evogp.tree.fused_results.fused_campaign` therefore names arms this
process must know about before the gate runs -- registering them here is what keeps
the gate's own view of the registry in step with the round being gated::

    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc \\
        /tmp/rocvenv/bin/python src/evogp/tree/fused_results/fused_gate.py \\
        --results src/evogp/tree/fused_results/main \\
        --allow-missing --allow-digest-mismatch

The registration preamble is the same one :mod:`.fused_campaign` uses (paths +
``EVOXC_ARMS`` / ``ALL_ARMS``); it is duplicated on purpose rather than factored into a
fifth module, so each of these three scripts stays runnable on its own (a bare
``python <script> --help`` with no environment at all).  A worker bootstrap is NOT
installed here: the gate only reads records, it never spawns one.

Everything else -- the flags, the per-record checks, the warn/fail policy and the
summary block -- is ``check.main``'s own.
"""

from __future__ import annotations

import os
import sys

#: ``<repo>`` -- four directory hops up from this file (five counted from the file
#: itself); :func:`_find_repo_root` adds a bounded ascending guard.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start: str) -> str:
    """Resolve ``<repo>`` from this file's location, without naming a worktree.

    The canonical depth is ``<repo>/src/evogp/tree/fused_results/``; the bounded ascending
    scan is only a guard against a copy placed deeper (it looks for the harness directory),
    so it either finds the checkout that contains this file or gives up and returns the
    canonical answer.
    """
    root = os.path.abspath(os.path.join(start, "..", "..", "..", ".."))
    probe = root
    for _ in range(6):
        if os.path.isdir(os.path.join(probe, "test", "benchmark", "evoxc")):
            return probe
        probe = os.path.dirname(probe)
    return root


_REPO_ROOT = _find_repo_root(_HERE)

#: The library under test, the harness and the evaluation-engine checkout.
SRC_DIR = os.path.join(_REPO_ROOT, "src")
HARNESS_DIR = os.path.join(_REPO_ROOT, "test", "benchmark", "evoxc")
EVOXC_ROOT = "/root/evoxc"

#: The two arms a fused round is gated on (kept byte-identical to
#: ``fused_campaign.FUSED_ARMS``; the gate and the driver must never disagree about
#: what a fused arm is called).
FUSED_ARMS = {
    "evoxc-fused-iree": {"eval_mode": "fused", "backend": "iree", "family": "evoxc"},
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
}


def bootstrap_paths() -> list:
    """Put the library, the harness and the evoxc checkout first on ``sys.path``."""
    added = []
    for path in (HARNESS_DIR, SRC_DIR, EVOXC_ROOT):
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added


def register_fused_arms(common_module) -> dict:
    """Register :data:`FUSED_ARMS` in an already imported harness ``common``.

    ``ALL_ARMS`` is an import-time snapshot of the two sub-registries, so membership is
    ensured separately from the registry entry; an arm the harness already names keeps
    the harness's own spec.
    """
    registered, already, all_arms_added = [], [], []
    for name, spec in FUSED_ARMS.items():
        if name in common_module.EVOXC_ARMS:
            already.append(name)
        else:
            common_module.EVOXC_ARMS[name] = dict(spec)
            registered.append(name)
        if name not in common_module.ALL_ARMS:
            common_module.ALL_ARMS.append(name)
            all_arms_added.append(name)
    return {
        "registered": registered,
        "already": already,
        "all_arms_added": all_arms_added,
    }


def main(argv=None) -> int:
    """Register the fused arms, then hand the whole CLI to ``check.main``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap_paths()
    import common  # noqa: PLC0415  (the harness; imported after the path is set)

    outcome = register_fused_arms(common)
    print(
        "fused arms registered: "
        + ", ".join(sorted(set(outcome["registered"]) | set(outcome["already"])))
        + f" (new={outcome['registered']}, already={outcome['already']}, "
        f"ALL_ARMS+={outcome['all_arms_added']})",
        file=sys.stderr,
        flush=True,
    )

    import check  # noqa: PLC0415

    return int(check.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
