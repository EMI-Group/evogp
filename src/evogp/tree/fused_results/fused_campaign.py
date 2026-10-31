"""Thin driver: the ``test/benchmark/evoxc`` campaign with the two FUSED arms registered.

The harness (``test/benchmark/evoxc``) is the campaign of record for every evaluation
route, and its arm registry is an import-time snapshot of ``common.EVOXC_ARMS`` that
names the shipped routes only.  The fused (Triton) route therefore cannot be measured
by ``campaign.py`` until its two arms are registered, and the harness is OUTSIDE this
node's write scope -- so the registration lives here, in the node that owns the route,
and this script is the driver a fused round is run with::

    TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache \\
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \\
        /tmp/rocvenv/bin/python src/evogp/tree/fused_results/fused_campaign.py \\
        --out src/evogp/tree/fused_results/main \\
        --cells p256-L32-i8-N1024 --arms evoxc-fused-iree

**Where records land.**  ``src/evogp/tree/fused_results/main/`` -- the harness's own
``test/benchmark/evoxc/results/`` directory is outside this node's write scope, so a
fused round writes into this node's result directory instead (one subdirectory per
cell, one JSON per (cell, arm), exactly the layout ``check.py`` gates).  The default is
the same directory ``campaign_index.py`` reads.

What this script does, and nothing else:

1. puts ``<repo>/src``, ``<repo>/test/benchmark/evoxc`` and the evoxc checkout on
   ``sys.path`` (the repo root is five directory hops up from this file, so nothing
   here is tied to the worktree the module happens to live in);
2. registers the two fused arms in ``common.EVOXC_ARMS`` and appends their names to
   ``common.ALL_ARMS`` (that list is an import-time snapshot, i.e. appending to
   ``EVOXC_ARMS`` alone would leave ``--arms all`` blind to them);
3. because ``campaign.py`` runs every config in its OWN worker process (whose
   ``common`` has the stock registry, so ``arms.apply_arm_env`` would raise ``KeyError``
   there), it also installs a ``subprocess`` proxy on the ``campaign`` module that
   rewrites each ``python <harness>/worker.py ...`` command into
   ``python -c <bootstrap> <harness>/worker.py ...``, the bootstrap registering the
   same two arms before the worker runs.  Nothing on disk is modified: the proxy is
   scoped to the ``campaign`` module and restored when the campaign returns;
4. delegates to ``campaign.main()`` on the remaining argv (``sys.argv[1:]``), so every
   flag, the guard, the artifact hook and the driver record are the harness's own.

The registered arm names are printed once to stderr, so a run's log states what the
process added to the registry.
"""

from __future__ import annotations

import os
import sys

#: ``<repo>`` -- four directory hops up from this file
#: (``<repo>/src/evogp/tree/fused_results/fused_campaign.py``; five hops counted from
#: the file itself).  :func:`_find_repo_root` adds a bounded ascending guard so a copy
#: placed one level deeper still finds the harness instead of silently mis-resolving.
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

#: The library under test (``evogp`` lives directly in here).
SRC_DIR = os.path.join(_REPO_ROOT, "src")
#: The harness package (``common`` / ``campaign`` / ``check`` / ``worker`` / ...).
HARNESS_DIR = os.path.join(_REPO_ROOT, "test", "benchmark", "evoxc")
#: The evaluation-engine dependency (a sibling checkout, read-only).
EVOXC_ROOT = "/root/evoxc"

#: This node's result directory: ``test/benchmark/evoxc/results/`` is outside this
#: node's write scope, so a fused round writes here.  One subdirectory per cell, one
#: JSON per (cell, arm) -- the layout ``check.py`` gates.
RESULTS_DIR = os.path.join(_HERE, "main")

#: The two arms this driver adds.  ``evoxc-fused-iree`` is the shipped spelling (the
#: explicit backend makes the route resolve ``fused`` unconditionally, with an ``iree``
#: fallback); ``evoxc-fused`` is the same route with NO explicit backend, i.e. the
#: policy's own decision (``EVOGP_FUSED_MIN_POP`` and the device probe decide, exactly
#: what a user gets without exporting ``EVOGP_EVOXC_BACKEND``).
FUSED_ARMS = {
    "evoxc-fused-iree": {"eval_mode": "fused", "backend": "iree", "family": "evoxc"},
    "evoxc-fused": {"eval_mode": "fused", "backend": None, "family": "evoxc"},
}

#: The bootstrap a worker process runs instead of the worker script directly: it
#: registers the two arms in the worker's OWN ``common`` and then executes the script
#: with the argv the driver built (``python -c CODE <script> <args...>`` leaves the
#: script path at ``sys.argv[1]``).
_WORKER_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0, {here!r});"
    "import fused_campaign as _fused;"
    "_fused.bootstrap_paths();"
    "import common;"
    "_fused.register_fused_arms(common);"
    "_argv = list(sys.argv[1:]);"
    "sys.argv = _argv;"
    "import runpy;"
    "runpy.run_path(_argv[0], run_name='__main__')"
)


def bootstrap_paths() -> list:
    """Put the library, the harness and the evoxc checkout first on ``sys.path``.

    Idempotent and side-effect free otherwise (no import happens here), so it is also
    what the worker bootstrap calls before importing ``common``.  Returns the paths it
    added, in the order they were inserted.
    """
    added = []
    for path in (HARNESS_DIR, SRC_DIR, EVOXC_ROOT):
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added


def register_fused_arms(common_module) -> dict:
    """Register :data:`FUSED_ARMS` in an already imported harness ``common``.

    Both the registry entry and the ``ALL_ARMS`` membership are ensured (the list is an
    import-time snapshot of the two sub-registries, so a name added to ``EVOXC_ARMS``
    later is invisible to ``--arms all`` until it is appended).  An arm that is already
    registered is left untouched -- the harness's own entry, if it ever grows one, wins.
    Returns ``{"registered": [...], "already": [...], "all_arms_added": [...]}``.
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


class _WorkerBootstrapProxy:
    """A ``subprocess`` stand-in for one imported module: rewrite worker commands only.

    ``campaign.campaign.subprocess`` is the real module, so patching ``subprocess.run``
    itself would be process-global (``common._git_head`` calls it too, and so does
    anything else in the driver).  This proxy exposes every attribute of the real module
    and overrides exactly one method, which keeps the rewrite scoped to the harness
    module that is being driven.  Only commands of the shape
    ``[<python>, .../worker.py, <args>...]`` are rewritten -- anything else (the guard's
    ``pgrep``, ``git``, a caller's own child) passes through untouched.
    """

    def __init__(self, module) -> None:
        self._module = module
        self.rewritten = 0

    def __getattr__(self, name):
        return getattr(self._module, name)

    def run(self, cmd, *args, **kwargs):
        if (
            isinstance(cmd, (list, tuple))
            and len(cmd) >= 2
            and os.path.basename(str(cmd[1])) == "worker.py"
        ):
            self.rewritten += 1
            cmd = [cmd[0], "-c", _WORKER_BOOTSTRAP.format(here=_HERE), *cmd[1:]]
        return self._module.run(cmd, *args, **kwargs)


def install_worker_bootstrap(campaign_module) -> _WorkerBootstrapProxy:
    """Make the fused arms resolvable inside every worker ``campaign.py`` spawns.

    ``worker.py`` resolves ``--arm`` through its own ``common.EVOXC_ARMS`` (the process
    is spawned fresh, precisely so no arm inherits another's warm cache), so a
    registration made in THIS process is invisible there and ``arms.apply_arm_env``
    raises ``KeyError: 'evoxc-fused-iree'``.  The proxy above closes that gap without
    touching the harness (which is outside this node's write scope).  Idempotent: a
    proxy already installed by an earlier call is returned as is.
    """
    existing = getattr(campaign_module, "subprocess", None)
    if isinstance(existing, _WorkerBootstrapProxy):
        return existing
    proxy = _WorkerBootstrapProxy(existing)
    campaign_module.subprocess = proxy
    return proxy


def main(argv=None) -> int:
    """Register the fused arms, then hand the whole CLI to ``campaign.main``."""
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

    import campaign  # noqa: PLC0415

    proxy = install_worker_bootstrap(campaign)
    try:
        status = campaign.main(argv)
    finally:
        if proxy.rewritten:
            print(
                f"worker bootstrap applied to {proxy.rewritten} worker command(s)",
                file=sys.stderr,
                flush=True,
            )
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
