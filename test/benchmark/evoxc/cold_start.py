"""Cold-start column: wall time from process start to the FIRST fitness value.

What this measures
------------------
The number a user actually feels: how long a fresh Python process takes, on the
DEFAULT route (``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` unset), to return a
first ``Forest.SR_fitness`` value for one cell -- i.e. a cold compile, a cold
host-side decode and a cold first graph call, all on an empty in-memory cache.

Why one process per population
------------------------------
The engine's compile cache and executable memo live in PROCESS memory
(``evoxc.CompileCache`` + ``evoxc_engine._EXECUTABLE_MEMO``), so a cold number
only exists in a process that has not compiled anything yet.  A loop over ``pop``
inside one process would measure the first population cold and every later one
warm, so this script measures exactly ONE population per invocation and the
caller runs one process per population::

    PY=/tmp/genesis/task_967a300e3bce7f35/evogp-venv/bin/python
    export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
    export EVOXC_DISABLE_LOCATIONS=1

    # the default cell, one process
    $PY test/benchmark/evoxc/cold_start.py --cell p256-L64-i32-N1024

    # the pop column: one process per population (never inside one process)
    for pop in 16 64 256 1000; do
        $PY test/benchmark/evoxc/cold_start.py --pop "$pop"
    done

    # opt-in disk artifacts (src/evogp/tree/artifact_cache.py): this process is
    # the DRIVER -- it spawns two child processes, one that populates the
    # directory (cold artifact write) and one that re-loads it (warm artifact)
    $PY test/benchmark/evoxc/cold_start.py --artifact-dir "$TMPDIR/evogp_artifacts"

    # replay against a frozen snapshot, apples to apples
    $PY test/benchmark/evoxc/cold_start.py --tree-root /tmp/evogp_preopt

``--repeat N`` re-issues the same first call N times INSIDE the one process.  No
cache is cleared between repeats, so only repeat 1 is cold; repeats 2..N are
warm-cache repeats and are recorded as such (``cold: false``) -- they are there to
show what the first call bought, never to be averaged with it.

The recorded headline is ``wall_to_first_fitness_ms``: ``time.perf_counter()``
sampled at module import (``_T0``) to the moment the first fitness tensor is back
and the device is synchronized.  It therefore includes the interpreter's import
of torch/evogp, the workload build (which also creates the CUDA context), and the
first engine call; those three are also recorded separately
(``breakdown.import_ms`` / ``workload_build_ms`` / ``first_call_ms``) with the
remainder as ``breakdown.other_ms``.  Interpreter start-up BEFORE this module's
first line is the one piece not visible from here -- it is stated in ``notes``.

``--tree-root DIR`` behaves exactly like ``worker.py``: ``<root>/src`` (or
``<root>`` when it has no ``src/``) is put first on ``sys.path`` before ``evogp``
is imported, so a git-archive snapshot times like this checkout.

Record: one JSON per invocation at ``--out`` (default
``test/benchmark/evoxc/results/final/cold/<cell>-pop<pop>[-artifact].json``),
schema ``evogp-bench-evoxc-v1`` in ``env``, plus the per-phase numbers and
``notes[]``.  In artifact driver mode ``wall_to_first_fitness_ms`` is a
per-phase dict (``populate`` / ``reload``) and the two child records are kept
next to it as ``<out stem>.<phase>.json``.

Nothing is written into ``src/`` and no library state is monkeypatched: the only
switch this script sets is ``EVOGP_EVOXC_ARTIFACT_DIR`` (``--artifact-dir``), the
documented opt-in knob.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

#: Origin of every ``wall_to_first_fitness_ms`` in this process: sampled at import,
#: before torch/evogp are imported (see the module docstring).
_T0 = time.perf_counter()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)

DEFAULT_CELL = "p256-L64-i32-N1024"
DEFAULT_SEED = 0
#: The opt-in disk layer of ``src/evogp/tree/artifact_cache.py``.  The constant is
#: also exported by ``evoxc_engine``; the literals here are what pre-optimization
#: trees (which lack the module) are compared against.
ARTIFACT_DIR_ENV = "EVOGP_EVOXC_ARTIFACT_DIR"
ARTIFACT_SUFFIX = ".evoxcart"
ARTIFACT_MODULE_PATH = os.path.join("evogp", "tree", "artifact_cache.py")
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "final", "cold")
PHASES = ("populate", "reload")


# ---------------------------------------------------------------------------
# plumbing (mirrors worker.py so both scripts switch trees the same way)
# ---------------------------------------------------------------------------
def install_tree_root(tree_root: str) -> str:
    """Put ``<tree_root>/src`` (or ``tree_root``) first on ``sys.path``; return it."""
    root = os.path.abspath(tree_root)
    src = os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root
    if src not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, src)
    return src


def artifact_feature_present(src: str) -> bool:
    """Does the imported tree ship the opt-in disk layer?  (File check: no import.)

    A pre-optimization snapshot has no ``evogp/tree/artifact_cache.py``; asking by
    path keeps this check usable before ``evogp`` is imported at all.
    """
    return os.path.isfile(os.path.join(src, ARTIFACT_MODULE_PATH))


def effective_cell(args: argparse.Namespace) -> Dict[str, Any]:
    """The cell actually measured (``--pop`` overrides the id's population)."""
    spec = common.parse_cell(args.cell)
    if args.pop is not None:
        spec = common.cell(args.pop, spec["tree_len"], spec["input_len"], spec["rows"])
    return spec


def default_out(cell: Dict[str, Any], artifact: bool) -> str:
    name = f"{cell['id']}-pop{cell['pop']}" + ("-artifact" if artifact else "") + ".json"
    return os.path.join(DEFAULT_OUT_DIR, name)


def resolve_out(args: argparse.Namespace, cell: Dict[str, Any], artifact: bool) -> str:
    return args.out or default_out(cell, artifact)


def phase_path(out: str, phase: str) -> str:
    """``<out stem>.<phase>.json`` -- a driver record never overwrites its children."""
    stem, ext = os.path.splitext(out if out.endswith(".json") else out + ".json")
    return f"{stem}.{phase}{ext}"


def dir_usage(path: Optional[str], suffix: str = ARTIFACT_SUFFIX) -> Dict[str, Any]:
    """File count / byte total of the artifacts in ``path`` (missing dir -> zeros)."""
    usage: Dict[str, Any] = {
        "dir": os.path.abspath(path) if path else None,
        "suffix": suffix,
        "files": 0,
        "bytes": 0,
        "unreadable": 0,
    }
    if not path or not os.path.isdir(path):
        return usage
    for root, _dirs, names in os.walk(path):
        for name in names:
            if not name.endswith(suffix):
                continue
            try:
                size = os.path.getsize(os.path.join(root, name))
            except OSError:
                usage["unreadable"] += 1
                continue
            usage["files"] += 1
            usage["bytes"] += size
    return usage


def stat_delta(before: Optional[Dict[str, int]], after: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """``after - before`` per counter (``None`` when either snapshot is missing)."""
    if not before or not after:
        return None
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


# ---------------------------------------------------------------------------
# the measurement (one process, one cell, one population)
# ---------------------------------------------------------------------------
def measure(args: argparse.Namespace, phase: str, artifact_absent: bool) -> int:
    cell = effective_cell(args)
    artifact_requested = bool(args.artifact_dir)

    import torch

    common.import_evogp()
    from evogp.tree import dense_engine, evoxc_engine

    import arms as arms_mod  # noqa: E402  (harness arm namespace: env application)
    import worker as worker_mod  # noqa: E402  (forest_stats: the structure context)

    # the DEFAULT route: both route knobs unset, whatever the caller exported
    inherited = {key: os.environ.get(key) for key in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND")}
    knobs = arms_mod.apply_arm_env("evoxc-auto")
    artifact_enabled = artifact_requested and not artifact_absent
    dir_env = getattr(evoxc_engine, "ARTIFACT_DIR_ENV", ARTIFACT_DIR_ENV)
    suffix = getattr(evoxc_engine, "ARTIFACT_SUFFIX", ARTIFACT_SUFFIX)
    if artifact_enabled:
        os.environ[dir_env] = os.path.abspath(args.artifact_dir)

    import_ms = (time.perf_counter() - _T0) * 1e3

    # -- workload (deterministic: torch.manual_seed(seed) + random_generate) ----
    build_start = time.perf_counter()
    workload = common.build_workload(cell, args.seed)
    torch.cuda.synchronize()
    workload_build_ms = (time.perf_counter() - build_start) * 1e3
    forest = workload["forest"]

    artifact_stats_fn = getattr(evoxc_engine, "artifact_cache_stats", None)
    artifact_before = artifact_stats_fn() if artifact_stats_fn else None
    dir_before = dir_usage(args.artifact_dir, suffix) if artifact_enabled else None

    # -- the first call, then the (warm-cache) repeats -------------------------
    runs: List[Dict[str, Any]] = []
    first_wall_ms = float("nan")
    for index in range(max(1, args.repeat)):
        cache_before = evoxc_engine.cache_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = forest.SR_fitness(workload["x"], workload["labels"])
        torch.cuda.synchronize()
        call_ms = (time.perf_counter() - started) * 1e3
        wall_ms = (time.perf_counter() - _T0) * 1e3
        cache_after = evoxc_engine.cache_stats()
        if index == 0:
            first_wall_ms = wall_ms
        runs.append(
            {
                "index": index + 1,
                "cold": index == 0,
                "first_call_ms": round(call_ms, 4),
                "wall_to_first_fitness_ms": round(wall_ms, 4),
                "cache": {
                    "before": cache_before,
                    "after": cache_after,
                    "compile_misses": cache_after["misses"] - cache_before["misses"],
                },
                "artifact": artifact_stats_fn() if artifact_stats_fn else None,
                "loss_digest": common.digest(loss),
            }
        )
        del loss

    dir_after = dir_usage(args.artifact_dir, suffix) if artifact_enabled else None

    record: Dict[str, Any] = {
        "env": common.env_block(
            args.tree_root,
            "cold-start",
            extra={
                "applied_knobs": knobs,
                "cell": cell,
                "phase": phase,
                "artifact_dir": os.path.abspath(args.artifact_dir) if artifact_requested else None,
            },
        ),
        "cell": cell,
        "measurement": "cold-start",
        "phase": phase,
        "protocol": {
            "one_process_per_population": True,
            "route": "default (EVOGP_EVAL_MODE / EVOGP_EVOXC_BACKEND unset)",
            "t0": "module import (cold_start.py top level)",
            "repeat": max(1, args.repeat),
            "repeats_share_the_process_cache": True,
            "seed": args.seed,
            "sync_bracketed": True,
            "workload": "sr_fitness",
        },
        "wall_to_first_fitness_ms": round(first_wall_ms, 4),
        "breakdown": {
            "import_ms": round(import_ms, 4),
            "workload_build_ms": round(workload_build_ms, 4),
            "first_call_ms": runs[0]["first_call_ms"],
            "other_ms": round(first_wall_ms - (import_ms + workload_build_ms + runs[0]["first_call_ms"]), 4),
        },
        "runs": runs,
        "cache": {
            "before": runs[0]["cache"]["before"],
            "after_first_call": runs[0]["cache"]["after"],
            "after": runs[-1]["cache"]["after"],
            "compile_misses_first_call": runs[0]["cache"]["compile_misses"],
            "compile_misses_total": sum(run["cache"]["compile_misses"] for run in runs),
        },
        "routes": {"sr_fitness": common.route_decision(cell["pop"], cell["rows"], knobs["eval_mode"], knobs["backend"])},
        "dense_policy": {
            "auto_min_pop": dense_engine.resolve_auto_min_pop(),
            "auto_max_rows": dense_engine.resolve_auto_max_rows(),
        },
        "notes": [],
    }
    record["structures"] = worker_mod.forest_stats(forest)

    if artifact_absent:
        record["artifact_cache"] = "absent"
        record["notes"].append(
            "the imported tree has no evogp/tree/artifact_cache.py (pre-optimization "
            "snapshot): the artifact half is skipped and EVOGP_EVOXC_ARTIFACT_DIR is "
            "left unset"
        )
    elif not artifact_enabled:
        record["artifact_cache"] = {
            "feature": "present",
            "enabled": False,
            "reason": "no --artifact-dir (EVOGP_EVOXC_ARTIFACT_DIR unset)",
        }
    else:
        record["artifact_cache"] = {
            "feature": "present",
            "enabled": True,
            "env": dir_env,
            "dir": os.path.abspath(args.artifact_dir),
            "suffix": suffix,
            "before": artifact_before,
            "after_first_call": runs[0]["artifact"],
            "after": runs[-1]["artifact"],
            "first_call_deltas": stat_delta(artifact_before, runs[0]["artifact"]),
            "dir_before": dir_before,
            "dir_after": dir_after,
        }

    if knobs["eval_mode"] is None and knobs["backend"] is None:
        record["notes"].append(
            "the DEFAULT route is measured: EVOGP_EVAL_MODE / EVOGP_EVOXC_BACKEND are "
            f"cleared in this process (inherited values were {inherited}), so an inherited "
            "route cannot leak into a cold number"
        )
    record["notes"].append(
        "the CUDA context is created during workload construction (the data tensors are "
        "built on the device), so first_call_ms excludes context creation while "
        "wall_to_first_fitness_ms includes it"
    )
    record["notes"].append(
        "wall_to_first_fitness_ms starts at this module's first line; interpreter "
        "start-up before it is not visible from inside the process"
    )
    if args.repeat > 1:
        record["notes"].append(
            f"--repeat {args.repeat}: repeats 2..{args.repeat} run on the same process cache "
            "(no cache is cleared), so only run 1 is cold -- never average them together"
        )
    if phase in PHASES:
        record["notes"].append(
            f"artifact phase {phase!r}: {dir_env}={os.path.abspath(args.artifact_dir)} was set "
            "before the first call"
        )
    if args.note:
        record["notes"].append(args.note)

    out = args.out or default_out(cell, artifact=artifact_requested)
    common.write_json(out, record)
    print(
        f"[cold-start] {cell['id']} pop={cell['pop']} phase={phase} "
        f"wall_to_first_fitness={first_wall_ms:.1f}ms "
        f"(import {import_ms:.1f} + build {workload_build_ms:.1f} + "
        f"first_call {runs[0]['first_call_ms']:.1f}) "
        f"compile_misses={runs[0]['cache']['compile_misses']} -> {out}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# artifact driver: two child processes (populate, then reload)
# ---------------------------------------------------------------------------
def run_artifact_driver(args: argparse.Namespace) -> int:
    """Populate the artifact dir in one process, re-load it in a second one."""
    cell = effective_cell(args)
    out = resolve_out(args, cell, artifact=True)
    before = dir_usage(args.artifact_dir)
    records: Dict[str, Dict[str, Any]] = {}

    for phase in PHASES:
        child_out = phase_path(out, phase)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--cell", args.cell,
            "--pop", str(cell["pop"]),
            "--tree-root", args.tree_root,
            "--seed", str(args.seed),
            "--repeat", str(max(1, args.repeat)),
            "--artifact-dir", os.path.abspath(args.artifact_dir),
            "--phase", phase,
            "--out", child_out,
        ]
        if args.note:
            cmd += ["--note", args.note]
        print(f"[artifact] {phase}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            raise SystemExit(f"artifact phase {phase!r} failed with exit {proc.returncode}")
        records[phase] = common.read_json(child_out)

    after = dir_usage(args.artifact_dir)
    populate, reload = records["populate"], records["reload"]

    env = dict(populate["env"])
    env["artifact_measurement"] = {
        "dir": os.path.abspath(args.artifact_dir),
        "phases": list(PHASES),
        "one_process_per_phase": True,
    }
    record: Dict[str, Any] = {
        "env": env,
        "cell": cell,
        "measurement": "cold-start-artifact",
        "phase": "artifact",
        "protocol": populate.get("protocol"),
        "wall_to_first_fitness_ms": {
            phase: records[phase]["wall_to_first_fitness_ms"] for phase in PHASES
        },
        "breakdown": {phase: records[phase]["breakdown"] for phase in PHASES},
        "artifact": {
            "feature": "present",
            "dir": os.path.abspath(args.artifact_dir),
            "suffix": after["suffix"],
            "before": before,
            "after": after,
            "files_added": after["files"] - before["files"],
            "bytes_added": after["bytes"] - before["bytes"],
        },
        "phases": {
            phase: {
                "out": phase_path(out, phase),
                "wall_to_first_fitness_ms": records[phase]["wall_to_first_fitness_ms"],
                "first_call_ms": records[phase]["breakdown"]["first_call_ms"],
                "compile_misses_first_call": records[phase]["runs"][0]["cache"]["compile_misses"],
                "artifact_first_call_deltas": records[phase]["artifact_cache"].get("first_call_deltas")
                if isinstance(records[phase].get("artifact_cache"), dict)
                else None,
                "artifact_dir_before": records[phase]["artifact_cache"].get("dir_before")
                if isinstance(records[phase].get("artifact_cache"), dict)
                else None,
                "artifact_dir_after": records[phase]["artifact_cache"].get("dir_after")
                if isinstance(records[phase].get("artifact_cache"), dict)
                else None,
            }
            for phase in PHASES
        },
        "notes": [
            "artifact mode is two child processes of this same script, one per phase: "
            "the in-memory compile cache is per process, so only a second process can "
            "show what the on-disk artifact buys",
            "phase 'populate' compiles and writes the artifacts; phase 'reload' must "
            "report artifact_hits >= 1 on its first call (artifact_first_call_deltas) "
            "for the number to mean anything",
            "each phase's wall_to_first_fitness_ms is that child's own full cold-start "
            "number (module import -> first synchronized fitness), not a delta",
            "artifact copies: the child records sit next to this one as "
            f"{os.path.basename(phase_path(out, 'populate'))} / "
            f"{os.path.basename(phase_path(out, 'reload'))}",
        ],
    }
    if args.note:
        record["notes"].append(args.note)

    reload_deltas = record["phases"]["reload"]["artifact_first_call_deltas"] or {}
    if not reload_deltas.get("artifact_hits"):
        record["notes"].append(
            "WARNING: the reload phase recorded no artifact hit on its first call "
            "(artifact_first_call_deltas.artifact_hits is 0/absent): the second process "
            "did not re-load what the first one stored, so its timing is not a warm-artifact "
            "number -- read the artifact_hits/artifact_misses counters before quoting it"
        )
    if before["files"]:
        record["notes"].append(
            "WARNING: the artifact dir was not empty before the populate phase "
            f"({before['files']} files, {before['bytes']} B): 'populate' may have re-loaded "
            "existing artifacts instead of writing them -- use a fresh directory for a true "
            "cold artifact write"
        )

    common.write_json(out, record)
    print(
        f"[cold-start] artifact dir={os.path.abspath(args.artifact_dir)} "
        f"populate={populate['wall_to_first_fitness_ms']:.1f}ms "
        f"reload={reload['wall_to_first_fitness_ms']:.1f}ms "
        f"files {before['files']}->{after['files']} bytes {before['bytes']}->{after['bytes']} "
        f"-> {out}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cell", default=DEFAULT_CELL, help="cell id, e.g. p256-L64-i32-N1024")
    parser.add_argument(
        "--pop", type=int, default=None,
        help="override the cell's population (one process per population, never a loop)",
    )
    parser.add_argument("--tree-root", default="src", help="tree root whose src/ holds evogp")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch.manual_seed for the workload")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="re-issue the same first call N times in this process (no cache reset: "
             "only repeat 1 is cold)",
    )
    parser.add_argument(
        "--artifact-dir", default=None,
        help=f"opt-in disk artifacts ({ARTIFACT_DIR_ENV}); runs two child processes, "
             "populate then reload",
    )
    parser.add_argument(
        "--phase", choices=("auto", "populate", "reload"), default="auto",
        help="internal: which artifact half this process performs (default: drive both)",
    )
    parser.add_argument("--out", default=None, help=f"JSON record path (default under {DEFAULT_OUT_DIR})")
    parser.add_argument("--note", default=None, help="free-form note stored in the record")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.phase != "auto" and not args.artifact_dir:
        raise SystemExit("--phase populate/reload requires --artifact-dir")

    src = install_tree_root(args.tree_root)

    if args.artifact_dir and args.phase == "auto":
        if artifact_feature_present(src):
            return run_artifact_driver(args)
        # the imported tree predates the disk layer: one cold number, "absent" noted
        return measure(args, phase="auto", artifact_absent=True)
    return measure(args, phase=args.phase, artifact_absent=False)


if __name__ == "__main__":
    raise SystemExit(main())
