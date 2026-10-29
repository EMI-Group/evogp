"""Batch-route compile economics: what ONE merged graph per population costs, and how often it is rebuilt.

What this measures
------------------
``EVOGP_EVAL_MODE=batch`` (``src/evogp/tree/batch_engine.py``) compiles ONE merged graph per
population, keyed on the population's STRUCTURE MULTISET -- not one graph per tree structure
(``specialized``) and not one graph per population shape (``dense``).  Its economics are therefore
decided by a question no other route asks: does the NEXT generation reuse a merged graph this process
has already compiled?

One child process measures ONE (cell, route) and records ``first_call`` (ONE sync-bracketed
``Forest.SR_fitness`` on the freshly built population: the merged-graph compile plus the cold host
decode for a batch route, the dense-graph compile for ``dense`` -- EXPECTED to be large, in seconds;
reading it as a defect is exactly the mistake this script exists to prevent), ``sr_fitness`` (the
warm/steady-state medians of the same call on the same population, via ``common.time_call``),
``generations`` (a REAL ``StandardPipeline.step()`` loop with the population's structure-multiset
signature snapshotted BEFORE every step -- that IS the population the generation evaluates -- both
cache stats around the step and the wall/eval split; deliberately NOT warmed up, so generation 1 is
the honest first generation) and ``summary`` (hit rate, recompile count, warm-vs-cold median
evaluation, compile-attributed share -- all derived from the recorded rows, never asserted).

The honest question, in one paragraph
-------------------------------------
Crossover and mutation replace most of a population every generation, so the structure MULTISET
changes on almost every step and a merged graph compiled for generation ``g`` may be useless for
``g + 1``.  If the route recompiles every generation, its warm-runtime advantage over ``dense`` is
eaten by compile; if elites and duplicates keep the multiset stable, the merged-graph cache hits and
the route wins.  Nobody knows which happens on a real run, so this script MEASURES the hit rate across
consecutive generations instead of asserting one.  ``dense`` goes through the same code path so the two
columns are directly comparable: its per-generation compile counter is the evoxc engine's own, and it
has no merged-graph cache at all (those keys are absent, not null).

Why one process per (cell, route)
---------------------------------
The evoxc compile cache, the executable memo and the batch route's merged-graph cache all live in
PROCESS memory: a fresh process per config is the only way each record pays its own cold compile, and
the only way the per-generation hit rate describes a process that really started empty.  The driver
spawns them one at a time; the child measures exactly ONE cell on ONE route.

Generation 1 is expected to HIT
-------------------------------
``StandardPipeline.step()`` evaluates ``algorithm.forest`` before selecting, so generation 1 evaluates
the initial population -- the very population whose graph the timed ``first_call`` already compiled.
That is recorded, not "fixed" by a warm-up step: the interesting numbers are generations 2..N.

Reproduce on the GPU box
------------------------
::

    PY=<venv>/bin/python
    export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
    export EVOXC_DISABLE_LOCATIONS=1

    # what would run, and whether the tree even ships the route (torch-free)
    python3 test/benchmark/evoxc/batch_economics.py --dry-run --routes batch,dense,batch-flat

    # the three large cells x two routes, one child process per config
    $PY test/benchmark/evoxc/batch_economics.py

    # one config by hand (a child is a normal, self-contained record producer)
    $PY test/benchmark/evoxc/batch_economics.py --phase child \
        --cell p1000-L32-i8-N1024 --route batch --out "$TMPDIR/batch.json" --generations 8

``--tree-root DIR`` selects the checkout under measurement exactly like ``worker.py`` /
``cold_start.py`` (``<root>/src`` first on ``sys.path``).  ``--out-dir`` defaults to a FRESH
``results/batch/`` and refuses ``results/final`` and ``results/baseline`` (frozen evidence).  Nothing
here writes into ``src/``; the only library state touched is the route's documented env knobs
(``arms.apply_arm_env``) plus ``EVOXC_DISABLE_LOCATIONS``, defaulted to ``1`` when the caller left it
unset (the driver records which of the two happened).

Record layout: one JSON per (cell, route) at ``<out-dir>/<cell>/<route>.json`` plus the child's
``.log`` and one driver record ``<out-dir>/economics.json`` (schema
``evogp-bench-evoxc-v1-batch-economics``).  ``--dry-run`` builds no ``env_block`` at all, so it runs
on a box without torch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)
from cold_start import install_tree_root  # noqa: E402  (the one tree-switch helper)

#: The three "large" cells the question is about: pop 256 / 1000 on the two built-in widths, where a
#: merged graph is big enough that its compile is worth measuring.
DEFAULT_CELLS = "p256-L32-i8-N1024,p1000-L32-i8-N1024,p1000-L64-i32-N1024"
DEFAULT_ROUTES = "batch,dense"
DEFAULT_TREE_ROOT = "src"
DEFAULT_EVOXC_ROOT = "/root/evoxc"
DEFAULT_GENERATIONS = 8
DEFAULT_WARMUP = 2
DEFAULT_REPS = 5
DEFAULT_SEED = 0
#: A merged-graph compile is minutes, not milliseconds: the per-config wall cap is deliberately
#: generous and a child that exceeds it is recorded as ``timeout`` (evidence, never a retry).
DEFAULT_TIMEOUT = 3600
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "batch")
DISABLE_LOCATIONS_ENV = "EVOXC_DISABLE_LOCATIONS"
FROZEN_RESULT_DIRS = ("final", "baseline")  # frozen evidence: never written in place
MEASUREMENT = "batch-compile-economics"

#: Route token -> registry arm name.  The ONLY route table in this file: the arm names are
#: ``common.EVOXC_ARMS`` keys, the knobs are the arms' own specs, and "is this a batch route" is
#: answered by ``common.is_batch_arm`` -- nothing about the route is re-declared here.
ROUTES: Dict[str, str] = {
    "batch": "evoxc-batch-iree", "batch-flat": "evoxc-batch-iree-flat", "dense": "evoxc-dense-iree",
}

#: Environment variables the driver forwards verbatim to every child (on top of the inherited
#: environment), so a record shows the configuration window it ran in.  The batch knobs are named
#: explicitly: the child clears and re-applies them per arm, but an operator's export must still reach
#: the process to be visible in its records.
FORWARDED_ENV = (
    "EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_EVOXC_DEVICE", "EVOGP_EVOXC_OPT_LEVEL",
    "EVOGP_EVOXC_CACHE_SIZE", "EVOGP_DENSE_AUTO_MIN_POP", "EVOGP_DENSE_AUTO_MAX_ROWS",
    "EVOGP_EVOXC_ARTIFACT_DIR",
) + common.BATCH_ENV_VARS


# --- selections (pure: no torch, no evogp) ----------------------------------
def _tokens(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_cells(text: str) -> List[Dict[str, Any]]:
    """``"p256-L32-i8-N1024,p1000-..."`` -> the cell dicts, in the given order."""
    cells: List[Dict[str, Any]] = []
    for token in _tokens(text):
        spec = common.parse_cell(token)
        if spec["id"] not in [cell["id"] for cell in cells]:
            cells.append(spec)
    if not cells:
        raise SystemExit("--cells selected nothing")
    return cells


def resolve_routes(text: str) -> List[str]:
    """``"batch,dense"`` -> route tokens, validated against :data:`ROUTES`."""
    routes: List[str] = []
    for token in _tokens(text):
        if token not in ROUTES:
            raise SystemExit(
                f"unknown route {token!r}; known: {', '.join(ROUTES)} "
                f"(arms: {', '.join(sorted(set(ROUTES.values())))})"
            )
        if token not in routes:
            routes.append(token)
    if not routes:
        raise SystemExit("--routes selected nothing")
    return routes


def route_is_batch(route: str) -> bool:
    """Does this route token measure the batch engine?  (The arm's spec decides.)"""
    return common.is_batch_arm(ROUTES[route])


# --- batch-route preflight (pure: the message is testable without torch) ----
def batch_preflight(route: str, policy: Dict[str, Any], src: str) -> Optional[str]:
    """The refusal paragraph for a batch route this tree cannot run (``None`` = OK).

    Pure on purpose: it consumes the report ``common.batch_policy(src)`` already builds, so the
    stub-tree smoke test can exercise the exact text on a box without torch and the child phase only
    decides whether to raise ``SystemExit`` with it.  A route this tree cannot run must fail here
    LOUDLY: a record of some other route written under a batch label would be worse than none.
    """
    if not route_is_batch(route):
        return None
    if policy.get("module_present") and policy.get("mode_supported"):
        return None
    return (
        f"route {route!r} ({ROUTES[route]}): EVOGP_EVAL_MODE=batch needs BOTH a module at "
        f"<src>/evogp/tree/batch_engine.py AND 'batch' declared in "
        f"evogp.tree.dense_engine.EVAL_MODES, and this tree does not provide the pair -- found "
        f"module_present={policy.get('module_present')} (expected {policy.get('module')}, "
        f"src={src}), declared_modes={policy.get('declared_modes')!r} "
        f"(mode_supported={policy.get('mode_supported')}). NO record was written for this config: "
        f"the batch route cannot run on this tree, and a record of a different route would be worse "
        f"than a missing one."
    )


#: Keys the dry run renders from ``common.batch_policy`` (order is the report's own order).
POLICY_FIELDS = ("band_size", "bands", "alignment", "switch_ops", "min_pop", "module_present",
                 "mode_supported")


def format_policy(policy: Dict[str, Any]) -> str:
    """One-line render of ``common.batch_policy`` (the dry run's evidence line)."""
    fields = " ".join(f"{key}={policy.get(key)}" for key in POLICY_FIELDS)
    return f"{fields} declared_modes={policy.get('declared_modes')!r}"


def batch_policy_for(arm: str, src: str) -> Dict[str, Any]:
    """``common.batch_policy(src)`` with the arm's own knob spec applied around it.

    The ``-flat`` variant differs from ``batch`` only by ``EVOGP_BATCH_BANDS=flat``, so the resolution
    has to happen with that arm's knobs in the environment for the dry run to report the bands the
    child will really use.  The environment is restored afterwards: the dry run resolves several
    routes in one process.
    """
    knobs = dict((common.EVOXC_ARMS.get(arm) or {}).get("knobs") or {})
    previous = {key: os.environ.get(key) for key in knobs}
    os.environ.update({key: str(value) for key, value in knobs.items()})
    try:
        return common.batch_policy(src)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- record paths / output-directory policy ---------------------------------
def resolve_src(tree_root: str) -> str:
    """``<tree_root>/src`` when it exists, else ``tree_root`` (same rule as worker.py)."""
    root = os.path.abspath(tree_root)
    return os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root


def record_path(out_dir: str, cell_id: str, route: str) -> str:
    return os.path.join(out_dir, cell_id, f"{route}.json")


def log_path(out_dir: str, cell_id: str, route: str) -> str:
    return os.path.join(out_dir, cell_id, f"{route}.log")


def assert_fresh_out(out_dir: str) -> None:
    """Refuse to write into the frozen evidence directories (they are never rewritten in place)."""
    resolved = os.path.abspath(out_dir)
    for name in FROZEN_RESULT_DIRS:
        frozen = os.path.join(_HERE, "results", name)
        if resolved == frozen or resolved.startswith(frozen + os.sep):
            raise SystemExit(
                f"--out-dir {resolved} is inside the frozen evidence directory {frozen}: "
                "results/final and results/baseline are never rewritten in place -- point "
                "--out-dir at a fresh directory (default: results/batch)"
            )


# --- environment plumbing (driver side) -------------------------------------
def child_env(args: argparse.Namespace, src: str) -> Dict[str, str]:
    """The environment every child runs with: PYTHONPATH, the tracing switch, the route knobs.

    ``EVOXC_DISABLE_LOCATIONS`` is defaulted to ``1`` (tracing is ~4x faster; it is not part of the
    engine's cache key) but never overrides an explicit caller choice --
    :func:`disable_locations_provenance` records which of the two happened.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(src), os.path.abspath(args.evoxc_root), _HERE]
    )
    env[DISABLE_LOCATIONS_ENV] = os.environ.get(DISABLE_LOCATIONS_ENV, "1")
    for key in FORWARDED_ENV:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def disable_locations_provenance(env: Dict[str, str]) -> Dict[str, Any]:
    """State, in the record, that (and how) the tracing switch was applied."""
    return {
        "env": DISABLE_LOCATIONS_ENV,
        "value": env.get(DISABLE_LOCATIONS_ENV),
        "source": "caller" if os.environ.get(DISABLE_LOCATIONS_ENV) is not None else "driver-default",
        "note": "honoured by every child: it speeds evoxc tracing up ~4x and is not part of the "
                "engine's cache key",
    }


def driver_env_block(args: argparse.Namespace, src: str) -> Optional[Dict[str, Any]]:
    """The driver's env block, or ``None`` on a box where torch cannot be imported.

    ``common.env_block`` imports torch/numpy, and the driver is supposed to run on a torch-free box,
    so the failure is a reportable state (the children build their own blocks).
    """
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - a torch-free driver box is a supported box
        return None
    return common.env_block(
        args.tree_root, "driver",
        extra={"note": args.note, "pythonpath": child_env(args, src)["PYTHONPATH"]},
    )


# --- derived statistics (pure: exercised by the stub smoke test) ------------
def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _median(values: Sequence[float]) -> Optional[float]:
    """Median of a possibly empty list (``None`` instead of a ZeroDivisionError)."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def compile_hit(misses: Sequence[Optional[int]]) -> Optional[bool]:
    """``True`` = nothing compiled, ``False`` = at least one recompile, ``None`` = unknown.

    ``None`` the moment one cache the route relies on has no counter: a MISSING counter must never
    read as "no compile" (0), because that number -- was the merged graph rebuilt -- is exactly what
    this script measures.
    """
    if any(value is None for value in misses):
        return None
    return sum(int(value) for value in misses) == 0


def generation_summary(
    generations: Sequence[Dict[str, Any]],
    first_call_ms: float,
    warm_median_ms: Optional[float],
    merged_graph_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The derived block, computed ONLY from the recorded per-generation rows.

    ``compile_ms_share`` is an ATTRIBUTION, not a timer: the wall time of the generations that
    recompiled divided by the loop's total wall time.  A recompiling generation also evaluates and
    mutates, so the share is the honest UPPER BOUND of "how much of the loop the compile generations
    own", never a measurement of compile alone.

    ``generation_hit_rate`` is ``hits / generations`` per the protocol, with ``generations_unknown``
    next to it so a reader sees how many generations had no counter to judge (a batch route without
    ``batch_cache_stats``).
    """
    rows = list(generations)
    count = len(rows)
    hits = [row for row in rows if row["compile_hit"] is True]
    recompiled = [row for row in rows if row["compile_hit"] is False]
    sum_total = sum(float(row["total_ms"]) for row in rows)
    summary: Dict[str, Any] = {
        "generations": count,
        "generations_with_compile": len(recompiled),
        "generations_unknown": count - len(hits) - len(recompiled),
        "generation_hit_rate": (len(hits) / count) if count else None,
        "distinct_population_signatures": len({row["population_signature"] for row in rows}),
        "median_warm_eval_ms": _round(_median([row["eval_ms"] for row in hits])),
        "median_cold_eval_ms": _round(_median([row["eval_ms"] for row in recompiled])),
        "sum_total_ms": _round(sum_total),
        "sum_eval_ms": _round(sum(float(row["eval_ms"]) for row in rows)),
        "compile_ms_share": _round(sum(float(row["total_ms"]) for row in recompiled) / sum_total)
        if sum_total else None,
        "first_call_ms": _round(first_call_ms),
        "warm_median_ms": _round(warm_median_ms),
    }
    # Absent (not null) on the dense route: it owns no merged-graph cache, and a null would invite
    # reading it as "the cache had no misses".
    if merged_graph_cache is not None:
        summary["merged_graph_cache"] = merged_graph_cache
    return summary


# --- dry run (torch-free by construction: no env_block) ---------------------
def dry_run(
    args: argparse.Namespace, cells: Sequence[Dict[str, Any]], routes: Sequence[str], src: str
) -> int:
    """Print every planned config, then where the batch routes resolve on THIS tree."""
    for cell in cells:
        for route in routes:
            print("would run", cell["id"], route, f"({ROUTES[route]})",
                  "->", record_path(args.out_dir, cell["id"], route))
    batch_routes = [route for route in routes if route_is_batch(route)]
    if not batch_routes:
        print("no batch route selected: nothing to resolve")
        return 0
    policy: Dict[str, Any] = {}
    for route in batch_routes:
        policy = batch_policy_for(ROUTES[route], src)
        print("batch policy", route, f"({ROUTES[route]})", format_policy(policy))
    # The point of a dry run is to learn this BEFORE a multi-minute compile: the resolved tree may
    # not ship the route yet, and then every batch config would fail its preflight.
    if not common.batch_engine_present(src):
        print(f"WARNING: {src} has no {common.BATCH_MODULE_REL} -- EVOGP_EVAL_MODE=batch cannot run "
              "here: every batch config would fail its preflight and write no record (the dense "
              "configs still run)")
    elif not policy["mode_supported"]:
        print(f"WARNING: {src} ships {common.BATCH_MODULE_REL} but its dense_engine.EVAL_MODES "
              f"({policy['declared_modes']!r}) does not declare 'batch' -- every batch config would "
              "fail its preflight and write no record")
    print(f"{len(cells) * len(routes)} configs planned -> {os.path.abspath(args.out_dir)}")
    return 0


# --- driver (stdlib + common only: no torch, one child process per config) --
def run_driver(args: argparse.Namespace) -> int:
    cells = resolve_cells(args.cells)
    routes = resolve_routes(args.routes)
    src = resolve_src(args.tree_root)
    if args.dry_run:
        return dry_run(args, cells, routes, src)

    assert_fresh_out(args.out_dir)
    env = child_env(args, src)
    os.makedirs(args.out_dir, exist_ok=True)
    driver_path = os.path.join(args.out_dir, "economics.json")
    driver_record: Dict[str, Any] = {
        "schema": common.SCHEMA + "-batch-economics",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "driver_env": driver_env_block(args, src),
        "protocol": {
            "one_process_per_config": True, "one_cell_per_process": True, "sequential": True,
            "cells": [cell["id"] for cell in cells], "routes": list(routes),
            "route_arms": {route: ROUTES[route] for route in routes},
            "generations": args.generations, "warmup": args.warmup, "reps": args.reps,
            "seed": args.seed, "timeout_s": args.timeout,
            "tree_root": os.path.abspath(args.tree_root), "src": src,
            "pythonpath": env["PYTHONPATH"],
            "evoxc_disable_locations": disable_locations_provenance(env),
            "batch_engine_present": common.batch_engine_present(src),
            "out_dir": os.path.abspath(args.out_dir),
        },
        "configs": [],
        "notes": [
            "one child process per (cell, route): the merged-graph cache, the evoxc compile cache "
            "and the executable memo are per process, so a route loop inside one process would "
            "measure warm caches only",
        ],
    }
    if driver_record["driver_env"] is None:
        driver_record["driver_env_note"] = (
            "torch is not importable in the driver process, so no env block was built (the driver "
            "itself only needs stdlib + common; every child builds its own)"
        )
    if args.note:
        driver_record["notes"].append(args.note)
    common.write_json(driver_path, driver_record)

    configs = [(cell, route) for cell in cells for route in routes]
    failures = 0
    for index, (cell, route) in enumerate(configs, 1):
        cell_id, arm = cell["id"], ROUTES[route]
        out_path, log = record_path(args.out_dir, cell_id, route), log_path(args.out_dir, cell_id, route)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [
            sys.executable, os.path.abspath(__file__), "--phase", "child",
            "--cell", cell_id, "--route", route, "--out", out_path,
            "--tree-root", args.tree_root, "--generations", str(args.generations),
            "--warmup", str(args.warmup), "--reps", str(args.reps), "--seed", str(args.seed),
        ]
        if args.note:
            cmd += ["--note", args.note]
        started = time.time()
        status, detail = "ok", ""
        with open(log, "w", encoding="utf-8") as handle:
            logged = {key: value for key, value in env.items()
                      if key.startswith(("EVOGP_", "EVOXC_")) or key == "PYTHONPATH"}
            handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
            handle.write("env: " + json.dumps(logged) + "\n\n")
            handle.flush()
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                detail = (f"killed after --timeout {args.timeout}s: a merged-graph compile may "
                          "exceed the cap on a large cell -- see the log for how far it got")
                for stream in (exc.stdout, exc.stderr):
                    if isinstance(stream, str):
                        handle.write(stream)
            else:
                handle.write(proc.stdout)
                if proc.stderr:
                    handle.write(proc.stderr)
                if proc.returncode != 0:
                    status = "failed"
                    tail = [ln for ln in (proc.stderr or proc.stdout or "").splitlines() if ln.strip()]
                    detail = tail[-1] if tail else f"exit {proc.returncode}"
                elif not os.path.isfile(out_path):
                    status, detail = "failed", "the child exited 0 but wrote no record"
        wall_s = round(time.time() - started, 2)
        if status != "ok":
            failures += 1
        driver_record["configs"].append({
            "cell": cell_id, "route": route, "arm": arm, "status": status, "wall_s": wall_s,
            "record": out_path if os.path.isfile(out_path) else None, "log": log, "detail": detail,
        })
        # Written after every config: a crash later must not lose the earlier evidence.
        common.write_json(driver_path, driver_record)
        print(f"[{index}/{len(configs)}] {cell_id} {route} ({arm}) -> {status} ({wall_s}s)"
              + (f" -- {detail}" if detail else ""), flush=True)
    print(f"[batch-econ] {len(configs)} configs, {len(configs) - failures} ok, "
          f"{failures} failed/timeout -> {driver_path}", flush=True)
    return 1 if failures else 0


# --- child phase: ONE cell, ONE route, ONE process --------------------------
def run_child(args: argparse.Namespace) -> int:
    """Measure one (cell, route) and write its record.

    Imports ``torch``/``evogp`` here and nowhere else, so ``--help``, ``--dry-run`` and the driver
    stay runnable on a box without torch.  The batch preflight runs BEFORE the torch import: a tree
    without the route fails fast with one readable paragraph.
    """
    for flag, value in (("--cell", args.cell), ("--route", args.route), ("--out", args.out)):
        if not value:
            raise SystemExit(f"--phase child needs {flag}")
    if args.route not in ROUTES:
        raise SystemExit(f"unknown route {args.route!r}; known: {', '.join(ROUTES)}")

    src = install_tree_root(args.tree_root)
    cell = common.parse_cell(args.cell)
    arm = ROUTES[args.route]

    import arms as arms_mod  # noqa: E402  (harness arm namespace: env application)
    import worker as worker_mod  # noqa: E402  (timed fitness + structure context)

    knobs = arms_mod.apply_arm_env(arm)  # one arm per process; the arm's knobs win
    is_batch = common.is_batch_arm(arm)
    batch_policy = common.batch_policy(src) if is_batch else None
    if batch_policy is not None:
        problem = batch_preflight(args.route, batch_policy, src)
        if problem:
            raise SystemExit(problem)

    import torch

    common.import_evogp()
    from evogp.tree import dense_engine, evoxc_engine

    # The batch engine owns its counter contract, so ask it directly (getattr-probed: a route
    # without ``batch_cache_stats`` is a reportable state -- null counters, never a fabricated 0).
    batch_stats = None
    if is_batch:
        try:
            from evogp.tree import batch_engine as batch_module
        except ImportError:  # the route can be absent on an older tree
            batch_module = None
        candidate = getattr(batch_module, "batch_cache_stats", None)
        batch_stats = candidate if callable(candidate) else None

    def batch_snapshot() -> Optional[Dict[str, Any]]:
        return batch_stats() if batch_stats is not None else None

    def misses(before: Any, after: Any) -> Optional[int]:
        # worker.py already owns the tolerant counter delta: an absent/renamed key reads as unknown
        # (None), never as 0.
        return worker_mod._stats_delta(before, after, "misses")

    # determinant workload: torch.manual_seed(seed) + Forest.random_generate
    workload = common.build_workload(cell, args.seed)
    forest = workload["forest"]
    structures = worker_mod.forest_stats(forest)
    arm_obj = arms_mod.make_arm(arm, workload)

    # (b) the cold first call: the merged-graph (batch) or dense-graph compile
    evoxc_before, batch_before = evoxc_engine.cache_stats(), batch_snapshot()
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = arm_obj.sr_fitness()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1e3
    evoxc_after, batch_after = evoxc_engine.cache_stats(), batch_snapshot()
    first_call: Dict[str, Any] = {
        "first_call_ms": round(first_call_ms, 4),
        "evoxc_cache": {"before": evoxc_before, "after": evoxc_after,
                        "compile_misses": misses(evoxc_before, evoxc_after)},
        "digest": common.digest(loss),
    }
    del loss
    if is_batch:
        first_call["merged_graph_cache"] = {
            "available": batch_stats is not None,
            "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
            "before": batch_before, "after": batch_after,
            "compile_misses": misses(batch_before, batch_after),
        }

    # (c) warm medians on the same population (the compile is already paid).  The batch probe
    # snapshots the merged-graph cache the instant time_call's untimed call returns -- the same
    # helper worker.py uses to separate the cold compile from the warmup+reps total.
    probe = (worker_mod.BatchStatsProbe(arm_obj.sr_fitness, batch_stats)
             if is_batch and batch_stats is not None else None)
    warm_before, batch_warm_before = evoxc_engine.cache_stats(), batch_snapshot()
    timing = common.time_call(probe if probe is not None else arm_obj.sr_fitness,
                              warmup=args.warmup, reps=args.reps, stats=evoxc_engine.cache_stats)
    warm_after = evoxc_engine.cache_stats()
    warm_result = timing.pop("_out")
    warm_after_first = timing.pop("cache_after_first_call")
    sr_fitness: Dict[str, Any] = {
        "warmup": timing["warmup"], "reps": timing["reps"],
        "samples_ms": [_round(value) for value in timing["samples_ms"]],
        # time_call's own first call is a WARM one here (the compile happened above); the cold
        # number is ``first_call.first_call_ms`` and lives only there.
        "warmup_call_ms": _round(timing["first_call_ms"]),
        "median_ms": _round(timing["median_ms"]), "min_ms": _round(timing["min_ms"]),
        "max_ms": _round(timing["max_ms"]), "mean_ms": _round(timing["mean_ms"]),
        "cache": {"before": warm_before, "after_first_call": warm_after_first, "after": warm_after,
                  "compile_misses_first_call": misses(warm_before, warm_after_first),
                  "compile_misses_this_workload": misses(warm_before, warm_after)},
        "digest": common.digest(warm_result),
    }
    del warm_result
    if is_batch:
        batch_warm_after_first = probe.first if probe is not None else None
        batch_warm_after = batch_snapshot()
        # the same before -> after_first_call -> after chain as the evoxc block above, so each
        # delta in this entry is checkable against its own snapshots
        sr_fitness["cache"]["batch_before"] = batch_warm_before
        sr_fitness["cache"]["batch_after_first_call"] = batch_warm_after_first
        sr_fitness["cache"]["batch_after"] = batch_warm_after
        sr_fitness["cache"]["batch_compile_misses_first_call"] = misses(
            batch_warm_before, batch_warm_after_first)
        sr_fitness["cache"]["batch_compile_misses_this_workload"] = misses(
            batch_warm_before, batch_warm_after)
    warm_median_ms = sr_fitness["median_ms"]

    # (d) consecutive real generations: the merged-graph cache's measured hit rate
    timed = worker_mod.TimedFitness(arm_obj.fitness)
    pipeline = arms_mod.make_pipeline(workload, timed)
    # Signatures this process has compiled, seeded with the initial population (the timed first call
    # above compiled exactly it -- generation 1 evaluates that same population).
    compiled_signatures = {structures["structure_signature"]}
    generations: List[Dict[str, Any]] = []
    for index in range(1, args.generations + 1):
        # BEFORE the step: this is the population the generation is about to evaluate
        signature = common.population_signature(pipeline.algorithm.forest)
        seen_before = signature["signature"] in compiled_signatures
        evoxc_gen_before, batch_gen_before = evoxc_engine.cache_stats(), batch_snapshot()
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipeline.step()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - started) * 1e3
        evoxc_gen_after, batch_gen_after = evoxc_engine.cache_stats(), batch_snapshot()
        entry: Dict[str, Any] = {
            "g": index, "population_signature": signature["signature"],
            "distinct_structures": signature["distinct_structures"],
            "signature_seen_before": seen_before, "total_ms": _round(total_ms),
            "eval_ms": _round(timed.last_ms), "ops_ms": _round(total_ms - timed.last_ms),
            "compile_misses": misses(evoxc_gen_before, evoxc_gen_after),
            "pop_size": signature["pop_size"], "best_fitness": float(pipeline.best_fitness),
        }
        route_misses: List[Optional[int]] = [entry["compile_misses"]]
        if is_batch:
            entry["batch_compile_misses"] = misses(batch_gen_before, batch_gen_after)
            route_misses.append(entry["batch_compile_misses"])
        entry["compile_hit"] = compile_hit(route_misses)
        compiled_signatures.add(signature["signature"])
        generations.append(entry)
    batch_final = batch_snapshot()

    # (e) the derived block, computed only from the recorded rows
    merged_cache = None
    if is_batch:
        merged_cache = {
            "available": batch_stats is not None,
            "provenance": {
                "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
                "key": "sorted structure-multiset fingerprint "
                       "(common.population_signature.signature)",
                "note": "counters are null (NOT zero) when the route exposes no stats callable: a "
                        "missing counter must never read as 'no compile'",
            },
            "stats_before_first_call": batch_before, "stats_after_first_call": batch_after,
            "stats_after_generations": batch_final,
            "compile_misses_first_call": misses(batch_before, batch_after),
            "compile_misses_generations": misses(batch_after, batch_final),
            "compile_misses_total": misses(batch_before, batch_final),
        }
    summary = generation_summary(generations, first_call_ms, warm_median_ms,
                                 merged_graph_cache=merged_cache)

    # (f) the record
    record: Dict[str, Any] = {
        "env": common.env_block(args.tree_root, arm,
                                extra={"applied_knobs": knobs, "cell": cell,
                                       "route": args.route, "arm": arm}),
        "cell": cell, "route": args.route, "arm": arm, "measurement": MEASUREMENT,
        "protocol": {
            "one_process_per_config": True, "one_cell_per_process": True,
            "generations": args.generations, "warmup": args.warmup, "reps": args.reps,
            "seed": args.seed, "sync_bracketed": True,
            "evoxc_disable_locations": os.environ.get(DISABLE_LOCATIONS_ENV),
            "workload": "sr_fitness", "route_is_batch": is_batch, "pipeline_warmup": False,
        },
        "structures": structures,
        "routes": {"sr_fitness": common.route_decision(cell["pop"], cell["rows"],
                                                      knobs["eval_mode"], knobs["backend"])},
        "dense_policy": {"auto_min_pop": dense_engine.resolve_auto_min_pop(),
                         "auto_max_rows": dense_engine.resolve_auto_max_rows()},
        "first_call": first_call, "sr_fitness": sr_fitness, "generations": generations,
        "summary": summary, "engine_cache_final": evoxc_engine.cache_stats(),
        "notes": ["generation 1 evaluates the INITIAL population, whose graph the timed first call "
                  "already compiled, so generation 1 is expected to hit; it is recorded as measured "
                  "(not warmed up away) and the interesting number is generations 2..N"],
    }
    if is_batch:
        record["batch_policy"] = batch_policy
        record["notes"].append(
            "batch route: ONE merged graph per population structure multiset, so a multi-second "
            "first_call_ms is the expected cold cost of the route, recorded as measured and never "
            "treated as a defect or retried away")
        if batch_stats is None:
            record["notes"].append(
                "batch route: evogp.tree.batch_engine.batch_cache_stats() is unavailable on this "
                "tree, so every merged-graph counter is null (NOT zero) and compile_hit is null "
                "wherever only those counters could tell; the compile cost can then only be read "
                "off first_call_ms")
    else:
        record["notes"].append(
            "dense route (comparison column): no merged-graph cache exists, so batch_* counters are "
            "ABSENT rather than null; its per-generation compile misses are the evoxc engine's own "
            "and its first call is the dense-graph compile")
    if args.note:
        record["notes"].append(args.note)

    common.write_json(args.out, record)
    hit_rate = summary["generation_hit_rate"]
    warm_text = "n/a" if warm_median_ms is None else f"{warm_median_ms:.3f}"
    hit_text = "n/a" if hit_rate is None else f"{hit_rate:.2f}"
    print(f"[batch-econ] {cell['id']} {args.route} ({arm}) "
          f"first_call={summary['first_call_ms']:.1f}ms warm={warm_text}ms "
          f"recompiles={summary['generations_with_compile']}/{summary['generations']} "
          f"hit_rate={hit_text} -> {args.out}", flush=True)
    return 0


# --- entry point ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default=DEFAULT_CELLS, help="comma-separated cell ids (driver)")
    parser.add_argument("--routes", default=DEFAULT_ROUTES,
                        help=f"comma-separated route tokens: {', '.join(ROUTES)}")
    parser.add_argument("--tree-root", default=DEFAULT_TREE_ROOT, help="tree root whose src/ holds evogp")
    parser.add_argument("--evoxc-root", default=DEFAULT_EVOXC_ROOT, help="evoxc checkout on PYTHONPATH")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="fresh result directory (never results/final or results/baseline)")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="StandardPipeline.step() iterations (the generations for the hit rate)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warm sr_fitness calls")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="timed sr_fitness samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch.manual_seed for the workload")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-config wall cap in seconds (a merged-graph compile may take minutes)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--note", default=None, help="free-form note stored in every record")
    parser.add_argument("--phase", choices=("driver", "child"), default="driver",
                        help="internal: driver spawns children; child measures one (cell, route)")
    parser.add_argument("--cell", default=None, help="child phase: the ONE cell id")
    parser.add_argument("--route", default=None, help="child phase: the ONE route token")
    parser.add_argument("--out", default=None, help="child phase: the JSON record path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")
    if args.phase == "child":
        return run_child(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
