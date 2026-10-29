"""Warm band-size sweep of the opt-in batch route (``EVOGP_EVAL_MODE=batch``).
What this measures
------------------
``EVOGP_BATCH_BAND_SIZE`` decides how many deduped programs one merge band holds; the band
count is then the engine's own ``ceil(n_functions / band_size)`` (clamped into
``1..n_functions``), and ``EVOGP_BATCH_BANDS=none`` / ``flat`` is the other half of the same
knob table: ONE unbanded merge.  A single warm per-call number cannot price that knob, so
this round sweeps it on FIXED cells: population, data and seed are identical across configs
(``common.build_workload``: ``torch.manual_seed(0)`` + ``Forest.random_generate``), so the
ONLY thing that changes between two records of a cell is the route environment.
Every config runs on the SINGLE-BODY path (``EVOGP_BATCH_COMPILE_WORKERS=1``) on purpose:
chunking changes the merge SHAPE (one compiled body per chunk, each with its own band
partition), which would confound the band knob with the pool width.  ``--workers`` is
therefore asserted to be 1 -- a pool-width sweep belongs to ``batch_economics.py --knob
EVOGP_BATCH_COMPILE_WORKERS=N``.
Per (cell, config) the child measures three value workloads in a FIXED order --
``sr_fitness`` (the training hot path), ``batch_forward`` (pop, N, out), ``forward``
(pop, out) -- recording per workload the cold first call (``first_call_ms``: the
merged-graph compile of that plan), the warm ``samples_ms`` with their median/min/max/mean,
the value ``digest``, the merged-graph counter deltas, the ten-stage split and the
``band_report`` of the plan that served it.
Where the time goes, and what the merge did
-------------------------------------------
``stage_ms`` / ``stage_first_call`` come from ``stages.py`` (the single owner of the ten-stage
vocabulary: decode, dedup, trace, merge, lower, compile, load, upload, run, gather plus the
compile pool's wall time and job count); ``stage_ms`` is the per-call median over the timed
samples (reset-bracketed when the route ships ``batch_stage_reset``), ``stage_first_call`` is
the cold call's own block, and ``stage_ms.plan`` carries ``describe_plan``'s
``compile_workers`` / ``chunk_count`` / ``chunk_sizes``.  Every read is getattr-probed: a
tree without ``batch_stage_stats()`` records ``available: false`` with EVERY counter ``null``
and a note, never a fabricated ``0``.
``common.band_report`` reads the plan's own report (``plan.report`` =
``merged.batch_report.as_dict()``) next to the ``describe_plan`` view, because
``describe_plan`` exposes neither ``bands`` nor the two cost estimates.  Those estimates are
``None`` unless ``EVOGP_BATCH_BANDS=auto`` ran the static cost model, and this sweep pins a
band SIZE or the flat merge, so they read ``null`` (never ``0``) here: the expected value,
not a missing measurement.  A chunked plan's aggregate report nulls ``bands`` /
``band_sizes`` / ``band_strategy`` and that null is kept as recorded; this round is
single-body, so every band field is populated on a config that merged at all.
Numeric validation, per config
------------------------------
Every workload is diffed against the independent CPU oracle
(``src/evogp/tests/reference_eval.py``) with ``validate.compare`` -- the ONE comparator this
harness already trusts -- built exactly as ``validate.parse_case`` builds it (same slicing,
same ``out_len``, same label reshape).  ``record['validation']`` is keyed by workload and
keeps ``max_abs`` / ``max_rel`` / ``bitwise_equal`` / ``nan_mismatch`` / ``inf_mismatch`` /
``finite_points`` plus the ORACLE digest, so cross-config agreement can be judged afterwards:
two configs of one cell must produce the same digest, because the band partition is a
scheduling decision and may not change the value.
Why one process per (cell, config)
----------------------------------
The evoxc compile cache, the executable memo and the batch route's merged-graph tables all
live in PROCESS memory, so a fresh process per config is the only way each record pays its
own cold compile and inherits no other config's warm cache.  The driver spawns them ONE AT A
TIME and re-checks that the device is idle (``campaign.guard``) before every config,
recording the verdict in the record and in ``sweep.json``.
Reproduce on the GPU box
------------------------
::
    PY=/tmp/rocvenv/bin/python
    export EVOXC_DISABLE_LOCATIONS=1
    # what would run, with the exact route env of every (cell, config, workload) (torch-free)
    $PY test/benchmark/evoxc/band_sweep.py --dry-run
    # the full sweep: 3 cells x 5 configs, one child process each
    $PY test/benchmark/evoxc/band_sweep.py
    # one config by hand (a child is a normal, self-contained record producer)
    $PY test/benchmark/evoxc/band_sweep.py --child p64-L64-i32-N64 --config b32 \\
        --workloads sr_fitness --warmup 1 --reps 2 --out "$TMPDIR/band.json"
``--tree-root DIR`` selects the checkout under measurement exactly like ``worker.py`` /
``batch_economics.py`` (``<root>/src`` first on ``sys.path``); the child's ``PYTHONPATH`` is
exactly ``[<src>, <evoxc-root>, <this node dir>]`` -- the foreign ``/root/evoxc`` checkout is
deliberately NOT on it, so the sweep runs against ``--evoxc-root`` only.
``EVOXC_DISABLE_LOCATIONS`` is defaulted to ``1`` (tracing ~4x faster, not part of the
engine's cache key) and the driver records which of the two happened.
Record layout
-------------
One JSON per (cell, config) at ``<out-dir>/<cell>/<config>.json`` (schema
``evogp-bench-evoxc-v1-band-sweep``) plus the child's ``.log`` and one driver index
``<out-dir>/sweep.json`` (schema ``...-band-sweep-driver``).  ``results/final``,
``results/baseline``, ``results/batch`` and ``results/batch_opt_round1`` are frozen evidence
and refused as ``--out-dir``.  A config the route cannot run (absent/undeclared route, a
compile/binding failure, a ``BatchUnsupported``) still writes a record: ``status: "failed"``
plus ``error`` / ``error_type`` / ``traceback`` -- a failure is a recorded outcome, never a
crash and never a silent skip.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)
import counter_blocks  # noqa: E402  (torch-free: the ONE getattr probe for optional counters)
import stages  # noqa: E402  (torch-free, evogp-free: the stage vocabulary + derived blocks)
import worker as worker_mod  # noqa: E402  (counter deltas + plan facts; torch-free at import)
from batch_economics import (  # noqa: E402  (the generic driver helpers of this node)
    _round,
    disable_locations_provenance,
    format_policy,
    log_path,
    record_path,
    resolve_cells,
    resolve_src,
)
from cold_start import install_tree_root  # noqa: E402  (the one tree-switch helper)

#: The three cells of the round: two pop-256 cells at the two built-in widths (where a
#: merged graph is expensive enough to price) and one cheap pop-64 cell (the fixed-population
#: regime the route is documented to win, and exactly ON the ``EVOGP_BATCH_MIN_POP`` floor).
DEFAULT_CELLS = "p256-L32-i8-N1024,p256-L64-i32-N1024,p64-L64-i32-N64"
DEFAULT_CONFIGS = "b8,b16,b32,b64,flat"
DEFAULT_WORKLOADS = "sr_fitness,batch_forward,forward"
#: The measurement order is FIXED (never the caller's order): every record of a cell then
#: pays for its three plans in the same sequence, so two configs are comparable call by call.
WORKLOAD_ORDER: Tuple[str, ...] = ("sr_fitness", "batch_forward", "forward")
DEFAULT_TREE_ROOT = "src"
#: The evoxc checkout this round runs against (NOT ``/root/evoxc``: the child's PYTHONPATH is
#: built from this value, so an implicit foreign checkout can never slip in).
DEFAULT_EVOXC_ROOT = "/tmp/evoxc_r2"
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "batch_opt_round2", "warm")
DEFAULT_WARMUP = 1
DEFAULT_REPS = 5
DEFAULT_SEED = 0
#: The sweep runs the SINGLE-BODY path: ``EVOGP_BATCH_COMPILE_WORKERS`` is part of the
#: config env, and any other worker count would confound the band knob with the pool width.
DEFAULT_WORKERS = 1
#: Trees handed to the CPU oracle per workload (``0`` = all).  The oracle is a plain-torch
#: python walk, so a full pop-256 comparison would dominate the record's wall time; the
#: engine's own output is digested in full regardless.
DEFAULT_MAX_ORACLE = 16
#: A merged-graph compile is seconds to minutes: the per-config wall cap is generous, and a
#: child that exceeds it is recorded as ``timeout`` (evidence, never a retry).
DEFAULT_TIMEOUT = 3600
DEFAULT_GUARD_WAIT = 120.0
DISABLE_LOCATIONS_ENV = "EVOXC_DISABLE_LOCATIONS"
#: Frozen evidence: never written in place (this round's own ``results/batch_opt_round2``).
FROZEN_RESULT_DIRS = ("final", "baseline", "batch", "batch_opt_round1")
MEASUREMENT = "batch-band-sweep"
SCHEMA = common.SCHEMA + "-band-sweep"
SCHEMA_DRIVER = SCHEMA + "-driver"

#: ``b<N>`` configs pin the band SIZE; the engine then resolves the band count as
#: ``ceil(n_functions / N)`` clamped to ``1..n_functions`` (``batch_knobs.resolve_band_count``),
#: so one token sweeps every population without a per-population table here.
BAND_SIZE_CONFIGS: Dict[str, int] = {token: int(token[1:]) for token in ("b8", "b16", "b32", "b64")}
#: ``flat`` pins the band COUNT to one unbanded merge: the engine resolves the flat
#: spellings (``none`` / ``flat`` / ``0``) to ``K = 1``, i.e. ``band_strategy == "flat"`` and
#: ``band_sizes == [n_functions]``.
FLAT_CONFIG = "flat"
CONFIGS: Tuple[str, ...] = tuple(BAND_SIZE_CONFIGS) + (FLAT_CONFIG,)

#: The route every config of this round runs.  The band knob is added per config, and every
#: other ``common.ROUTE_ENV_VARS`` entry is explicitly UNSET (see :func:`config_env`).
BASE_ENV: Dict[str, str] = {
    "EVOGP_EVAL_MODE": "batch",
    "EVOGP_EVOXC_BACKEND": "iree",
    "EVOGP_BATCH_COMPILE_WORKERS": "1",
}

#: The oracle entry point of each workload (``reference_eval``, imported from the tree under
#: measurement): recorded next to every validation block so the diff is reproducible.
ORACLE_FNS: Dict[str, str] = {
    "sr_fitness": "sr_fitness_from_rows",
    "batch_forward": "batch_forward_from_rows",
    "forward": "forward_from_rows",
}


class RouteUnavailable(RuntimeError):
    """The tree under measurement cannot run ``EVOGP_EVAL_MODE=batch``.

    Raised BEFORE any timing exists, so the child records one readable paragraph as a
    ``status: "failed"`` record instead of crashing (a missing route must never look like a
    measured one).
    """


# --- selections (pure: no torch, no evogp) ----------------------------------
def _tokens(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def resolve_configs(text: str) -> List[str]:
    """``"b8,b32,flat"`` -> config tokens, validated against :data:`CONFIGS`."""
    configs: List[str] = []
    for token in _tokens(text):
        if token not in CONFIGS:
            raise SystemExit(
                f"unknown config {token!r}; known: {', '.join(CONFIGS)} "
                "(b<N> pins EVOGP_BATCH_BAND_SIZE=N, flat pins EVOGP_BATCH_BANDS=none)"
            )
        if token not in configs:
            configs.append(token)
    if not configs:
        raise SystemExit("--configs selected nothing")
    return configs


def resolve_workloads(text: str) -> List[str]:
    """The requested workloads in the FIXED measurement order (:data:`WORKLOAD_ORDER`)."""
    requested = set(_tokens(text))
    unknown = sorted(requested - set(WORKLOAD_ORDER))
    if unknown:
        raise SystemExit(
            f"unknown workload(s) {', '.join(unknown)}; this sweep measures "
            f"{', '.join(WORKLOAD_ORDER)} (gp_step is batch_economics.py's column, not a "
            "warm per-call number)"
        )
    workloads = [name for name in WORKLOAD_ORDER if name in requested]
    if not workloads:
        raise SystemExit("--workloads selected nothing")
    return workloads


def check_workers(workers: int) -> int:
    """Assert the single-body path: the pool width is NOT part of this sweep."""
    if int(workers) != 1:
        raise SystemExit(
            f"--workers {workers}: this sweep runs the SINGLE-BODY path "
            "(EVOGP_BATCH_COMPILE_WORKERS=1) only, because chunking changes the merge shape "
            "(one compiled body per chunk, each with its own band partition) and would "
            "confound the band knob with the pool width -- sweep the pool with "
            "batch_economics.py --knob EVOGP_BATCH_COMPILE_WORKERS=N instead"
        )
    return 1


# --- the route environment of one config (pure) -----------------------------
def config_env(config: str) -> Dict[str, Optional[str]]:
    """The exact route env one config runs with (``None`` = the child must UNSET it).

    EVERY ``common.ROUTE_ENV_VARS`` entry is named, so the child clears the whole table
    before writing its own value: a ``EVOGP_BATCH_*`` or ``EVOGP_DENSE_MAX_TREE_LEN`` value
    exported by the caller can never leak into a config.  ``EVOGP_BATCH_MIN_POP`` stays
    UNSET on purpose -- the route's own floor (default 64) is what the sweep is priced
    against, and the ``p64`` cell sits exactly ON it (raising the floor would silently turn
    that cell into the ``specialized`` fallback).
    """
    if config not in CONFIGS:
        raise SystemExit(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
    env: Dict[str, Optional[str]] = {name: None for name in common.ROUTE_ENV_VARS}
    env.update(BASE_ENV)
    if config == FLAT_CONFIG:
        env["EVOGP_BATCH_BANDS"] = "none"
    else:
        env["EVOGP_BATCH_BAND_SIZE"] = str(BAND_SIZE_CONFIGS[config])
    return env


def apply_env(mapping: Dict[str, Optional[str]]) -> None:
    """Write ``mapping`` into ``os.environ`` (a ``None`` value UNSETS the variable)."""
    for name, value in mapping.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def env_text(mapping: Dict[str, Optional[str]]) -> str:
    """One-line render of a config env (``<unset>`` for the variables it clears)."""
    return " ".join(
        f"{name}={value if value is not None else '<unset>'}" for name, value in sorted(mapping.items())
    )


def config_policy(config: str, src: str) -> Dict[str, Any]:
    """``common.batch_policy(src)`` with this config's env applied (restored afterwards).

    Same precedence the child runs with, so the dry run and the driver index report the bands
    the child will really use.  ``common.batch_policy`` guards its imports, so this stays
    runnable on a box without torch (it then records ``module_present: false``).
    """
    mapping = config_env(config)
    previous = {name: os.environ.get(name) for name in mapping}
    apply_env(mapping)
    try:
        return common.batch_policy(src)
    finally:
        apply_env(previous)


# --- output-directory policy ------------------------------------------------
def _checkout_root() -> Optional[str]:
    """The checkout that contains this node (``<root>/src/evogp``), found by walking up."""
    path = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(path, "src", "evogp")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def resolve_tree_root(tree_root: str) -> str:
    """The tree root to measure: ``--tree-root`` as given, else this node's own checkout.

    ``batch_economics.py`` / ``worker.py`` resolve a relative ``--tree-root`` against the CWD
    (the documented way to run them is from the repository root).  This script is also run
    from inside ``test/benchmark/evoxc``, where the shipped default ``src`` does not exist,
    so the default falls back to the checkout containing this node instead of measuring a
    non-existent tree; an explicit non-default path is always used verbatim (a typo must fail
    loudly, not be redirected to another checkout).
    """
    candidate = os.path.abspath(tree_root)
    if os.path.isdir(os.path.join(candidate, "src")) or os.path.isdir(
        os.path.join(candidate, "evogp")
    ):
        return candidate
    if os.path.abspath(tree_root) == os.path.abspath(DEFAULT_TREE_ROOT):
        checkout = _checkout_root()
        if checkout is not None:
            return checkout
    return candidate


def install_evoxc_root(evoxc_root: str) -> str:
    """Put the sweep's ``evoxc`` checkout on ``sys.path`` (the driver side of the child PYTHONPATH).

    ``common.batch_policy`` asks the tree's own knob resolvers, and ``dense_engine`` imports
    ``evoxc``; without this the driver would report ``declared_modes`` ``None`` and warn that
    the tree cannot run the route, while the child -- whose PYTHONPATH carries ``--evoxc-root``
    -- runs it fine.  sys.path only, no import: on a box without ``evoxc``/torch the policy
    resolution stays guarded and reports the import error instead of raising here.
    """
    root = os.path.abspath(evoxc_root)
    if root not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, root)
    return root


def assert_fresh_out(out_dir: str) -> None:
    """Refuse to write into the frozen evidence directories (never rewritten in place)."""
    resolved = os.path.abspath(out_dir)
    for name in FROZEN_RESULT_DIRS:
        frozen = os.path.join(_HERE, "results", name)
        if resolved == frozen or resolved.startswith(frozen + os.sep):
            raise SystemExit(
                f"--out-dir {resolved} is inside the frozen evidence directory {frozen}: "
                f"results/{', results/'.join(FROZEN_RESULT_DIRS)} are never rewritten in "
                "place -- point --out-dir at this round's own tree "
                f"(default: {DEFAULT_OUT_DIR})"
            )


# --- dry run (torch-free by construction: no env_block) ---------------------
def dry_run(
    args: argparse.Namespace,
    cells: List[Dict[str, Any]],
    configs: List[str],
    workloads: List[str],
    src: str,
) -> int:
    """Print every planned (cell, config, workload) with the exact env and resolved policy."""
    policies: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        for config in configs:
            mapping = config_env(config)
            if config not in policies:
                policies[config] = config_policy(config, src)
            policy = policies[config]
            for workload in workloads:
                print(
                    f"would run {cell['id']:<22} {config:<5} {workload:<13} "
                    f"-> {record_path(args.out_dir, cell['id'], config)} "
                    f"env[{env_text(mapping)}] policy[{format_policy(policy)}]"
                )
    print(
        "band count model: b<N> -> K = ceil(n_functions / N) clamped into 1..n_functions; "
        "flat -> K = 1 (one unbanded merge).  n_functions is the population's DEDUPED row "
        "count, known only after the population is built, so it is recorded per workload "
        "(band_report.n_functions / band_report.bands) instead of printed here."
    )
    # The point of a dry run is to learn this BEFORE a multi-second-to-minute compile.
    policy = policies[configs[0]]
    if not common.batch_engine_present(src):
        print(
            f"WARNING: {src} has no {common.BATCH_MODULE_REL} -- EVOGP_EVAL_MODE=batch cannot "
            "run here: every config would write a failed record (status 'failed', no timings)"
        )
    elif not policy.get("mode_supported"):
        print(
            f"WARNING: {src} ships {common.BATCH_MODULE_REL} but its dense_engine.EVAL_MODES "
            f"({policy.get('declared_modes')!r}) does not declare 'batch' -- every config "
            "would write a failed record"
        )
    if any(int(cell["pop"]) < int(policy.get("min_pop") or 0) for cell in cells):
        print(
            "WARNING: a selected cell is BELOW EVOGP_BATCH_MIN_POP "
            f"({policy.get('min_pop')}): the route declines it to specialized and the record "
            "will say so in routes[*].declines_to_specialized (p64 sits exactly ON the floor "
            "and is served by the batch route)"
        )
    print(f"{len(cells) * len(configs)} configs x {len(workloads)} workloads planned "
          f"-> {os.path.abspath(args.out_dir)}")
    return 0


# --- environment plumbing (driver side) -------------------------------------
def child_env(
    args: argparse.Namespace, src: str, config: str, dis_source: str
) -> Dict[str, str]:
    """The environment one child runs with: PYTHONPATH, the tracing switch, its config env.

    ``PYTHONPATH`` is exactly ``[<src>, <evoxc-root>, <this node dir>]``: the node dir carries
    ``common``/``worker``/``validate``, and ``/root/evoxc`` is deliberately absent so only
    ``--evoxc-root`` can provide ``evoxc``.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(src), os.path.abspath(args.evoxc_root), _HERE]
    )
    env[DISABLE_LOCATIONS_ENV] = os.environ.get(DISABLE_LOCATIONS_ENV, "1")
    mapping = config_env(config)
    for name, value in mapping.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    env["EVOGP_BAND_SWEEP_DIS_SOURCE"] = dis_source
    return env


DIS_SOURCE_ENV = "EVOGP_BAND_SWEEP_DIS_SOURCE"


def _driver_env_block(
    args: argparse.Namespace, tree_root: str, src: str
) -> Optional[Dict[str, Any]]:
    """The driver's env block, or ``None`` on a box where torch cannot be imported."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - a torch-free driver box is a supported box
        return None
    return common.env_block(
        tree_root, "band-sweep-driver",
        extra={"note": args.note, "pythonpath": child_env(args, src, CONFIGS[0], "n/a")["PYTHONPATH"]},
    )


# --- driver ----------------------------------------------------------------
def run_driver(args: argparse.Namespace) -> int:
    """One child process per (cell, config), strictly sequential, one record each."""
    cells = resolve_cells(args.cells)
    configs = resolve_configs(args.configs)
    workloads = resolve_workloads(args.workloads)
    check_workers(args.workers)
    tree_root = resolve_tree_root(args.tree_root)
    src = resolve_src(tree_root)
    # Put the measured tree first on sys.path so ``common.batch_policy`` (the knob resolvers)
    # and ``common.env_block`` (``evogp_file``) describe THAT checkout, not whatever the
    # driver's own PYTHONPATH happens to carry.  sys.path only: no evogp import here, so the
    # dry run stays runnable on a box without torch.
    install_tree_root(tree_root)
    install_evoxc_root(args.evoxc_root)
    # the frozen-directory refusal covers the dry run too: it prints the exact record paths,
    # and a path this script would refuse to write must not look planned
    assert_fresh_out(args.out_dir)
    if args.dry_run:
        return dry_run(args, cells, configs, workloads, src)
    os.makedirs(args.out_dir, exist_ok=True)
    base_env = child_env(args, src, configs[0], "n/a")
    dis_source = disable_locations_provenance(base_env)["source"]
    index_path = os.path.join(args.out_dir, "sweep.json")
    index: Dict[str, Any] = {
        "schema": SCHEMA_DRIVER,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "driver_env": _driver_env_block(args, tree_root, src),
        "protocol": {
            "one_process_per_config": True, "one_config_per_process": True, "sequential": True,
            "cells": [cell["id"] for cell in cells],
            "configs": list(configs),
            "workloads": list(workloads),
            "config_env": {config: config_env(config) for config in configs},
            "warmup": args.warmup, "reps": args.reps, "seed": args.seed,
            "workers": args.workers,
            "max_oracle_trees": args.max_oracle,
            "tree_root": os.path.abspath(tree_root), "tree_root_arg": args.tree_root,
            "src": src,
            "evoxc_root": os.path.abspath(args.evoxc_root),
            "pythonpath": base_env["PYTHONPATH"],
            "evoxc_disable_locations": disable_locations_provenance(base_env),
            "guard_wait_s": args.guard_wait, "timeout_s": args.timeout,
            "batch_engine_present": common.batch_engine_present(src),
            "out_dir": os.path.abspath(args.out_dir),
            "frozen_result_dirs": list(FROZEN_RESULT_DIRS),
        },
        "policies": {},
        "configs": [],
        "notes": [
            "one child process per (cell, config): the merged-graph cache, the evoxc compile "
            "cache and the executable memo are per process, so a config loop inside one "
            "process would measure warm caches only",
            "the band knob is the ONLY thing that changes between two configs of a cell: "
            "same seed, same population, same data (common.build_workload)",
            "single-body path: EVOGP_BATCH_COMPILE_WORKERS=1 in every config env, so a "
            "difference between configs is the band partition and never the pool width",
        ],
    }
    if index["driver_env"] is None:
        index["driver_env_note"] = (
            "torch is not importable in the driver process, so no env block was built (the "
            "driver only needs stdlib + common; every child builds its own)"
        )
    for config in configs:
        try:
            index["policies"][config] = config_policy(config, src)
        except Exception as exc:  # noqa: BLE001 - a broken resolver must not lose the index
            index["policies"][config] = {"error": repr(exc)}
    if args.note:
        index["notes"].append(args.note)
    common.write_json(index_path, index)

    import campaign as campaign_mod  # the ONE idle-device guard (stdlib + common)

    plan = [(cell, config) for cell in cells for config in configs]
    failures, skipped = 0, 0
    started_all = time.time()
    for position, (cell, config) in enumerate(plan, 1):
        cell_id = cell["id"]
        out_path = record_path(args.out_dir, cell_id, config)
        log = log_path(args.out_dir, cell_id, config)
        if args.resume and os.path.isfile(out_path):
            skipped += 1
            index["configs"].append({
                "cell": cell_id, "config": config, "status": "skipped",
                "record": out_path, "log": log, "wall_s": 0.0,
                "detail": "--resume: the record already existed and was not re-measured",
            })
            common.write_json(index_path, index)
            print(f"[{position}/{len(plan)}] skip  {cell_id} {config} (record exists)", flush=True)
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        env = child_env(args, src, config, dis_source)
        state = campaign_mod.guard(args.guard_wait)
        if state.get("contaminated"):
            print(f"[{position}/{len(plan)}] WARNING busy box for {cell_id} {config}: {state}",
                  flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--child", cell_id, "--config", config, "--out", out_path,
            "--tree-root", tree_root, "--evoxc-root", args.evoxc_root,
            "--cells", args.cells, "--configs", args.configs,
            "--workloads", ",".join(workloads),
            "--warmup", str(args.warmup), "--reps", str(args.reps), "--seed", str(args.seed),
            "--workers", str(args.workers), "--max-oracle", str(args.max_oracle),
            "--guard", json.dumps(state),
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
                          "exceed the cap -- see the log for how far it got")
                for stream in (exc.stdout, exc.stderr):
                    if isinstance(stream, str):
                        handle.write(stream)
            else:
                handle.write(proc.stdout)
                if proc.stderr:
                    handle.write("\n--- stderr ---\n" + proc.stderr)
                if proc.returncode != 0:
                    status = "failed"
                    tail = [ln for ln in (proc.stderr or proc.stdout or "").splitlines() if ln.strip()]
                    detail = tail[-1] if tail else f"exit {proc.returncode}"
                elif not os.path.isfile(out_path):
                    status, detail = "failed", "the child exited 0 but wrote no record"
        wall_s = round(time.time() - started, 2)
        if status != "ok":
            failures += 1
        index["configs"].append({
            "cell": cell_id, "config": config, "status": status, "wall_s": wall_s,
            "record": out_path if os.path.isfile(out_path) else None, "log": log,
            "detail": detail, "guard": state,
        })
        # Written after every config: a crash later must not lose the earlier evidence.
        common.write_json(index_path, index)
        print(f"[{position}/{len(plan)}] {status:<8} {cell_id:<22} {config:<5} {wall_s:7.1f}s "
              f"(gpu_idle={state.get('ok')})" + (f" -- {detail}" if detail else ""), flush=True)
    index["elapsed_s"] = round(time.time() - started_all, 1)
    index["failures"] = failures
    index["skipped"] = skipped
    common.write_json(index_path, index)
    print(f"[band-sweep] {len(plan)} configs, {len(plan) - failures - skipped} ok, {failures} "
          f"failed/timeout, {skipped} skipped, {index['elapsed_s']}s -> {index_path}", flush=True)
    return 0 if failures == 0 else 1


# --- child helpers ---------------------------------------------------------
def _guard_block(raw: Optional[str]) -> Dict[str, Any]:
    """The driver's idle-device verdict, as the child records it (never raises)."""
    if not raw:
        return {"available": False, "note": "no --guard verdict was handed to this child"}
    try:
        state = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - a malformed verdict is a reportable state
        return {"available": False, "unparsed": str(raw)[:400], "error": repr(exc)}
    if not isinstance(state, dict):
        return {"available": False, "unparsed": str(state)[:400]}
    return {"available": True, "state": state}


def _env_block(
    tree_root: str, cell: Dict[str, Any], config: str,
    mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """``common.env_block`` plus the exact env this config set (never raises)."""
    extra = {
        "applied_knobs": dict(mapping),
        "knob_source": "band_sweep-config",
        "cell": cell, "config": config,
    }
    try:
        return common.env_block(tree_root, "band-sweep", extra=extra)
    except Exception as exc:  # noqa: BLE001 - a torch-free child box is reportable
        block: Dict[str, Any] = {
            "schema": common.SCHEMA, "tree_root": os.path.abspath(tree_root),
            "note": f"env block unavailable: {exc!r}",
        }
        block.update(extra)
        return block


def workload_fn(name: str, workload: Dict[str, Any]):
    """The PUBLIC-API call of one workload (no harness arm stands in between)."""
    forest = workload["forest"]
    if name == "sr_fitness":
        x, labels = workload["x"], workload["labels"]
        return lambda: forest.SR_fitness(x, labels)
    if name == "batch_forward":
        x = workload["x"]
        return lambda: forest.batch_forward(x)
    x_per_row = workload["x_per_row"]
    return lambda: forest.forward(x_per_row)


def workload_plan(name: str, batch_module: Any) -> Tuple[str, bool]:
    """The ``(variant, per_row)`` pair the workload's merged plan is keyed on."""
    value = str(getattr(batch_module, "VALUE_VARIANT", "value"))
    loss = str(getattr(batch_module, "LOSS_VARIANT", "loss"))
    pairs = {
        "sr_fitness": (loss, False),      # the loss variant is per_row=False by contract
        "batch_forward": (value, False),
        "forward": (value, True),         # tree i sees row i
    }
    return pairs[name]


def plan_for(facts: Dict[str, Any], wanted: Tuple[str, bool]) -> Tuple[Optional[Dict[str, Any]], Any]:
    """The ``describe_plan`` view and the ``BatchPlan`` of one workload (``None`` if absent).

    ``worker.batch_plan_facts`` keeps ``plans`` (describe_plan views) and ``objects`` (the
    plan objects, parallel to ``plans``) so the band report can read the plan's OWN report --
    the only place the two cost-model estimates live -- without a second memo walk here.
    """
    plans = facts.get("plans") or []
    objects = facts.get("objects") or []
    for position, described in enumerate(plans):
        if (described.get("variant"), bool(described.get("per_row"))) == wanted:
            return described, (objects[position] if position < len(objects) else None)
    return None, None


def _oracle_module():
    """Import the independent CPU oracle from the tree under measurement."""
    oracle_dir = os.path.join(common.src_root(), "evogp", "tests")
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    import reference_eval

    return reference_eval


def _oracle_tensor(
    name: str, reference_eval: Any, rows: Dict[str, Any], workload: Dict[str, Any],
    out_len: int,
):
    """One workload's CPU oracle, built exactly as ``validate.parse_case`` builds it."""
    if name == "sr_fitness":
        return reference_eval.sr_fitness_from_rows(
            rows["node_value"], rows["node_type"], rows["subtree_size"],
            workload["x"].to("cpu"),
            workload["labels"].to("cpu").reshape(rows["n_rows"], -1),
            out_len=out_len,
        )
    if name == "batch_forward":
        return reference_eval.batch_forward_from_rows(
            rows["node_value"], rows["node_type"], rows["subtree_size"],
            workload["x"].to("cpu"), out_len=out_len,
        )
    return reference_eval.forward_from_rows(
        rows["node_value"], rows["node_type"], rows["subtree_size"],
        workload["x_per_row"][: rows["trees"]].to("cpu"), out_len=out_len,
    )


def _validation_block(name: str, reference, candidate, trees: int) -> Dict[str, Any]:
    """``validate.compare`` (the ONE comparator) of the engine output against the oracle."""
    import validate  # torch at import: child-side only, never the driver

    block = validate.compare(reference, candidate[: int(reference.shape[0])])
    block.update({
        "oracle": f"evogp/tests/reference_eval.py:{ORACLE_FNS[name]}",
        "trees_compared": int(reference.shape[0]),
        "trees_pop": int(trees),
        "oracle_digest": common.digest(reference),
    })
    return block


# --- child phase: ONE cell, ONE config, ONE process -------------------------
def run_child(args: argparse.Namespace) -> int:
    """Measure one (cell, config) and write its record.

    The config env is applied to ``os.environ`` BEFORE ``evogp``/``torch`` are imported, so
    the engine resolves exactly the knobs this record states.  A route that cannot run, a
    compile/binding failure or any other exception is written into the record as
    ``status: "failed"`` (with ``error`` / ``error_type`` / ``traceback``) and exits non-zero:
    the failure is an outcome of this config, never a crash and never a silent skip.
    """
    if not args.child:
        raise SystemExit("child mode needs --child <cell id>")
    config = (args.config or "").strip()
    if config not in CONFIGS:
        raise SystemExit(f"child mode needs --config (one of {', '.join(CONFIGS)}), got {config!r}")
    if not args.out:
        raise SystemExit("child mode needs --out <record path>")

    check_workers(args.workers)
    workloads = resolve_workloads(args.workloads)
    cell = common.parse_cell(args.child)
    tree_root = resolve_tree_root(args.tree_root)  # --tree-root, as worker.py resolves it
    src = install_tree_root(tree_root)  # sys.path only: no evogp import yet
    mapping = config_env(config)
    apply_env(mapping)  # BEFORE the torch/evogp import, by construction

    plan_cells = [spec["id"] for spec in resolve_cells(args.cells)] if args.cells else [cell["id"]]
    plan_configs = resolve_configs(args.configs) if args.configs else [config]
    evoxc_version, evogp_file, evogp_head = _identity()
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "status": "ok",
        "error": None,
        "error_type": None,
        "cell": cell,
        "config": config,
        "env": _env_block(tree_root, cell, config, mapping),
        "guard": _guard_block(args.guard),
        "protocol": {
            "cells": plan_cells, "configs": plan_configs,
            "cell": cell["id"], "config": config,
            "workloads": workloads,
            "warmup": args.warmup, "reps": args.reps, "seed": args.seed,
            "workers": args.workers, "max_oracle_trees": args.max_oracle,
            "config_env": mapping,
            "one_process_per_config": True, "one_config_per_process": True,
            "sequential": True, "sync_bracketed": True,
            "tree_root": os.path.abspath(tree_root), "tree_root_arg": args.tree_root,
            "src": src,
            "evoxc_root": os.path.abspath(args.evoxc_root),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "evoxc_version": evoxc_version,
            "evogp_file": evogp_file,
            "evogp_head": evogp_head,
            "evoxc_disable_locations": {
                "env": DISABLE_LOCATIONS_ENV,
                "value": os.environ.get(DISABLE_LOCATIONS_ENV),
                "source": os.environ.get(DIS_SOURCE_ENV, "unknown"),
                "note": "the driver records whether it defaulted the value (sweep.json "
                        "protocol.evoxc_disable_locations); it is not part of the engine's "
                        "cache key",
            },
            "note": args.note,
        },
        "batch_policy": None,
        "routes": {},
        "workloads": {},
        "validation": {},
        "notes": [
            "warm medians only: this child's first call is the COLD merged-graph compile of "
            "that workload's plan (first_call_ms) and never enters median_ms",
            "single-body path (EVOGP_BATCH_COMPILE_WORKERS=1): one compiled body per merged "
            "graph, so band_count is the whole n_functions partition",
            "record['validation'] is keyed by workload name: one compare(oracle, engine) "
            "block per workload, computed AFTER the timed window on the first --max-oracle "
            "rows (oracle = src/evogp/tests/reference_eval.py), so a cross-config comparison "
            "reads validation[workload].bitwise_equal / max_abs / oracle_digest",
            "band_report.cost_estimate / flat_estimate read null unless "
            "EVOGP_BATCH_BANDS=auto ran the static cost model -- a pinned band size or the "
            "flat merge never runs it, so null (NOT 0) is the expected value here",
        ],
    }
    if args.note:
        record["notes"].append(args.note)
    try:
        _measure(record, args, cell, config, src, workloads)
    except Exception as exc:  # noqa: BLE001 - a failure is a RECORDED outcome
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_type"] = type(exc).__name__
        record["traceback"] = traceback.format_exc()[-4000:]
        record["notes"].append(
            "this config did NOT produce timings: the record exists so the failure is an "
            "outcome of the sweep rather than a crash or a silent skip"
        )
        common.write_json(args.out, record)
        print(f"[band-sweep] FAILED {cell['id']} {config}: {record['error']} -> {args.out}",
              flush=True)
        return 1
    common.write_json(args.out, record)
    warm = "n/a"
    first = record["workloads"].get(workloads[0], {})
    if first.get("median_ms") is not None:
        warm = f"{first['median_ms']:.3f}"
    print(f"[band-sweep] {cell['id']} {config} workloads={len(record['workloads'])} "
          f"first_workload_median={warm}ms bands="
          f"{ {name: entry['band_report'].get('bands') for name, entry in record['workloads'].items()} } "
          f"-> {args.out}", flush=True)
    return 0


def _identity() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(evoxc_version, evogp_file, evogp_head)`` -- best effort, never raises."""
    evoxc_version = None
    evogp_file = None
    evogp_head = None
    try:
        import evoxc

        evoxc_version = getattr(evoxc, "__version__", None)
    except Exception:  # noqa: BLE001 - provenance only
        evoxc_version = None
    try:
        import evogp

        evogp_file = getattr(evogp, "__file__", None)
    except Exception:  # noqa: BLE001 - provenance only
        evogp_file = None
    try:
        evogp_head = common._git_head(common.checkout_root())
    except Exception:  # noqa: BLE001 - a snapshot without .git is a legal tree root
        evogp_head = None
    return evoxc_version, evogp_file, evogp_head


def _measure(
    record: Dict[str, Any], args: argparse.Namespace, cell: Dict[str, Any], config: str,
    src: str, workloads: List[str],
) -> None:
    """Fill ``record`` from one freshly built population (imports torch/evogp here only)."""
    import torch

    common.import_evogp()
    from evogp.tree import evoxc_engine

    policy = common.batch_policy(src)
    record["batch_policy"] = policy
    problem = _route_preflight(policy, src)
    if problem:
        raise RouteUnavailable(problem)
    try:
        from evogp.tree import batch_engine as batch_module
    except ImportError as exc:  # the route can be absent on an older tree
        raise RouteUnavailable(
            f"evogp.tree.batch_engine could not be imported from {src}: {exc!r}"
        ) from exc

    # The batch engine owns its counter contract, so ask it directly (getattr-probed: a route
    # without ``batch_cache_stats`` is a reportable state -- null counters, never a 0).
    batch_stats = counter_blocks.accessor(batch_module, "batch_cache_stats")
    # The stage split and its reset are a SECOND optional contract on the same module.
    stage_stats = stages.stage_stats_fn(batch_module)
    stage_reset = stages.stage_reset_fn(batch_module)

    def batch_snapshot() -> Optional[Dict[str, Any]]:
        return batch_stats() if batch_stats is not None else None

    workload = common.build_workload(cell, args.seed)
    forest = workload["forest"]
    signature = common.population_signature(forest)
    for name in workloads:
        rows = 1 if name == "forward" else int(cell["rows"])
        record["routes"][name] = common.route_decision(
            cell["pop"], rows, BASE_ENV["EVOGP_EVAL_MODE"], BASE_ENV["EVOGP_EVOXC_BACKEND"]
        )

    # The oracle is built from a CPU slice of the SAME population, sliced exactly like
    # validate.parse_case does, so the diff is the harness's own comparator.
    trees = (
        min(int(forest.pop_size), int(args.max_oracle))
        if int(args.max_oracle) > 0
        else int(forest.pop_size)
    )
    n_rows = int(workload["x"].shape[0])
    oracle_rows = {
        "trees": trees,
        "n_rows": n_rows,
        "node_value": forest.batch_node_value[:trees].to("cpu"),
        "node_type": forest.batch_node_type[:trees].to("cpu"),
        "subtree_size": forest.batch_subtree_size[:trees].to("cpu"),
    }
    reference_eval = _oracle_module()

    facts: Dict[str, Any] = {"available": False, "note": None, "plans": [], "summary": {}}
    for name in workloads:
        fn = workload_fn(name, workload)
        cache_before = batch_snapshot()
        # The probe brackets every call (reset + snapshot) when the route ships the reset, so
        # ``stage_ms`` below is a per-call block rather than a cumulative one.
        probe = stages.StageStatsProbe(fn, stage_stats, stage_reset)
        timing = common.time_call(probe, warmup=args.warmup, reps=args.reps, stats=batch_stats)
        cache_after_first = timing.pop("cache_after_first_call")
        cache_after = batch_snapshot()
        result = timing.pop("_out")
        entry: Dict[str, Any] = {
            "warmup": timing["warmup"], "reps": timing["reps"],
            "samples_ms": [_round(value) for value in timing["samples_ms"]],
            # this child has no separate warm first call: time_call's untimed call IS the cold
            # one, i.e. where the merged-graph compile of this workload's plan lands
            "first_call_ms": _round(timing["first_call_ms"]),
            "median_ms": _round(timing["median_ms"]),
            "min_ms": _round(timing["min_ms"]),
            "max_ms": _round(timing["max_ms"]),
            "mean_ms": _round(timing["mean_ms"]),
            "digest": common.digest(result),
            "cache": {
                "available": batch_stats is not None,
                "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
                "before": cache_before,
                "after_first_call": cache_after_first,
                "after": cache_after,
                "stats_keys": worker_mod.stats_keys(cache_before, cache_after),
                "batch_exec_misses_first_call": worker_mod.batch_miss_delta(
                    cache_before, cache_after_first),
                "batch_exec_misses_this_workload": worker_mod.batch_miss_delta(
                    cache_before, cache_after),
                "batch_plan_misses_first_call": worker_mod.batch_plan_miss_delta(
                    cache_before, cache_after_first),
                "batch_plan_misses_this_workload": worker_mod.batch_plan_miss_delta(
                    cache_before, cache_after),
            },
        }
        sample_blocks = probe.blocks[-args.reps:] if probe.blocks else []
        entry["stage_ms"] = stages.median_block(sample_blocks, samples=args.reps)
        entry["stage_first_call"] = probe.first
        facts = worker_mod.batch_plan_facts(batch_module)
        entry["stage_ms"]["plan"] = stages.plan_block(facts)
        described, plan = plan_for(facts, workload_plan(name, batch_module))
        entry["band_report"] = common.band_report(plan, described)
        # the oracle + the comparator run AFTER the timed window: CPU host work never times a
        # call, and the value is kept as a block rather than as a tensor
        reference = _oracle_tensor(name, reference_eval, oracle_rows, workload, int(forest.output_len))
        record["validation"][name] = _validation_block(name, reference, result, int(forest.pop_size))
        del reference
        del result
        record["workloads"][name] = entry
        print(f"[band-sweep]   {cell['id']} {config} {name}: first={entry['first_call_ms']:.1f}ms "
              f"median={entry['median_ms']:.3f}ms bands={entry['band_report'].get('bands')} "
              f"n_functions={entry['band_report'].get('n_functions')}", flush=True)

    final = batch_snapshot()
    record["batch_cache_final"] = {
        "available": batch_stats is not None,
        "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
        "stats": final,
        "stats_keys": worker_mod.stats_keys(final),
    }
    facts = worker_mod.batch_plan_facts(batch_module)
    record["batch_plans"] = facts["plans"]
    record["batch_plan_summary"] = (
        facts["summary"] if facts["available"]
        else {"available": False, "note": facts["note"],
              "band_report": common.band_report(None)}
    )
    record["cell"].update({
        "population_signature": signature["signature"],
        "distinct_structures": int(signature["distinct_structures"]),
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "output_len": int(forest.output_len),
        "n_functions": {
            key: (value or {}).get("n_functions")
            for key, value in (facts["summary"] or {}).items()
        },
    })
    record["engine_cache_final"] = evoxc_engine.cache_stats()
    if batch_stats is None:
        record["notes"].append(
            "evogp.tree.batch_engine.batch_cache_stats() is unavailable on this tree, so every "
            "merged-graph counter is null (NOT zero) and the compile cost can only be read off "
            "first_call_ms"
        )
    if stage_stats is None:
        record["notes"].append(
            f"{stages.STATS_FN_PATH}() is unavailable on this tree, so every stage_ms counter "
            "is null (NOT zero) with available=false: the split was not measured, which is not "
            "the same as any stage costing nothing"
        )
    elif stage_reset is None:
        record["notes"].append(
            f"{stages.MODULE_NAME}.{stages.RESET_FN_NAME}() is unavailable, so each stage block "
            "is cumulative rather than per-call; a max is then only a process-lifetime maximum"
        )
    if not facts["available"]:
        record["notes"].append(
            "the merged-graph plan facts are unavailable (no BATCH_PLAN_MEMO / describe_plan), "
            "so band_report is available=false with every band field null (NOT zero)"
        )


def _route_preflight(policy: Dict[str, Any], src: str) -> Optional[str]:
    """The refusal paragraph for a tree that cannot run the batch route (``None`` = OK)."""
    if policy.get("module_present") and policy.get("mode_supported"):
        return None
    return (
        "EVOGP_EVAL_MODE=batch needs BOTH a <src>/evogp/tree/batch_engine.py module AND "
        "'batch' declared in evogp.tree.dense_engine.EVAL_MODES, and this tree does not "
        f"provide the pair -- module_present={policy.get('module_present')} "
        f"(expected {policy.get('module')}, src={src}), "
        f"declared_modes={policy.get('declared_modes')!r} "
        f"(mode_supported={policy.get('mode_supported')}). No timing was attempted."
    )


# --- entry point ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cells", default=DEFAULT_CELLS, help="comma-separated cell ids (driver)")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS,
                        help=f"comma-separated band configs: {', '.join(CONFIGS)}")
    parser.add_argument("--workloads", default=DEFAULT_WORKLOADS,
                        help=f"subset of {', '.join(WORKLOAD_ORDER)} (measured in that order)")
    parser.add_argument("--tree-root", default=DEFAULT_TREE_ROOT,
                        help="tree root whose src/ holds evogp (any checkout)")
    parser.add_argument("--evoxc-root", default=DEFAULT_EVOXC_ROOT,
                        help="evoxc checkout on the child's PYTHONPATH (never /root/evoxc by "
                             "default: the sweep pins the checkout it measured)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="round evidence directory (never results/final|baseline|batch|"
                             "batch_opt_round1)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warm calls per workload")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="timed samples per workload")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="torch.manual_seed of the population AND the data")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="must be 1: this sweep runs the single-body path only")
    parser.add_argument("--max-oracle", type=int, default=DEFAULT_MAX_ORACLE,
                        help="trees handed to the CPU oracle per workload (0 = all)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-config wall cap in seconds (a merged-graph compile takes time)")
    parser.add_argument("--guard-wait", type=float, default=DEFAULT_GUARD_WAIT,
                        help="seconds to wait for an idle device before each config")
    parser.add_argument("--resume", action="store_true",
                        help="skip configs whose record already exists (recorded as skipped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print every planned (cell, config, workload) with its env, run nothing")
    parser.add_argument("--note", default=None, help="free-form note stored in every record")
    parser.add_argument("--child", default=None, metavar="CELL",
                        help="internal: measure ONE cell (with --config) and write --out")
    parser.add_argument("--config", default=None, help="internal: the ONE band config")
    parser.add_argument("--out", default=None, help="internal: the child's JSON record path")
    parser.add_argument("--guard", default=None, help="internal: the driver's guard verdict (JSON)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")
    if args.guard_wait < 0:
        raise SystemExit("--guard-wait must be >= 0")
    if args.max_oracle < 0:
        raise SystemExit("--max-oracle must be >= 0 (0 = the whole population)")
    if args.child:
        return run_child(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
