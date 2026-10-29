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

This file is the DRIVER: CLI, cell/route resolution, the environment layers, the idle-config
bookkeeping, subprocess launching and ``economics.json``.  The child-side loop internals (the cold
call / warm medians / per-generation record assembly, the three counter-delta helpers and the summary
derivation) live in the torch-free sibling ``econ_child.py``, which this module imports.

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

Named composition routes (round 2, work item 3)
-----------------------------------------------
A route is no longer only an arm name: :data:`ROUTE_SPECS` gives every route token its own
``{"arm", "knobs"}`` spec, so THREE batch configurations can be measured in one driver run.  The
round-2 generation-loop column registers:

============================  =========================  ==========================================
token                         arm                        route knobs
============================  =========================  ==========================================
``batch``                     ``evoxc-batch-iree``       (none: the arm's own spec)
``batch-flat``                ``evoxc-batch-iree-flat``  (none: the arm pins ``BANDS=flat``)
``dense``                     ``evoxc-dense-iree``       (none: the comparison column)
``batch-banded``              ``evoxc-batch-iree``       ``BANDS=auto``, ``COMPILE_WORKERS=1``
``batch-flat-chunked``        ``evoxc-batch-iree``       ``BANDS=none``, ``COMPILE_WORKERS=4``
``batch-auto-chunked``        ``evoxc-batch-iree``       ``BANDS=auto``, ``CHUNK_BANDS=1``,
                                                         ``COMPILE_WORKERS=4``
``batch-b16-banded``          ``evoxc-batch-iree``       ``BAND_SIZE=16``, ``COMPILE_WORKERS=1``
``batch-b16-chunked``         ``evoxc-batch-iree``       ``BAND_SIZE=16``, ``CHUNK_BANDS=1``,
                                                         ``COMPILE_WORKERS=4``
============================  =========================  ==========================================

(``BAND_SIZE``/``BANDS``/``CHUNK_BANDS``/``COMPILE_WORKERS`` are ``EVOGP_BATCH_BAND_SIZE`` /
``EVOGP_BATCH_BANDS`` / ``EVOGP_BATCH_CHUNK_BANDS`` / ``EVOGP_BATCH_COMPILE_WORKERS``.)
``batch-flat-chunked`` deliberately
leaves ``EVOGP_BATCH_CHUNK_BANDS`` UNSET: off, a band request forces the single-body path, so that
route is the unbanded-per-chunk control next to ``batch-auto-chunked``.

The env layers are applied in ONE order, by :func:`apply_route_env`, for the dry run and the child
alike: ``arms.apply_arm_env`` CLEARS every route knob (``common.ROUTE_ENV_VARS``: the ``EVOGP_BATCH_*``
set -- ``EVOGP_BATCH_CHUNK_BANDS`` included -- plus the dense trace cap) and writes the ARM's own
spec, then the ROUTE's own spec, then the explicit ``--knob`` override.  So a variable nobody sets is
really unset, never inherited, and ``protocol.route_knobs`` states, per variable, which layer won:
``arm`` / ``route`` / ``harness-knob`` / ``env`` for a value, or the engine's own ``module`` resolver
/ ``harness-default`` / ``unset`` when the environment carries none.

The band/merge verdict
----------------------
Each batch record also carries ``common.band_report``'s block -- the SAME block ``band_sweep.py``
writes -- for the plan(s) that served the route (``band_report.sr_fitness``, plus every distinct plan
under ``band_reports``): the measured ``bands``/``band_sizes``/``band_strategy``, the static cost
model's ``cost_estimate``/``flat_estimate``, the policy verdict (``band_requested``,
``band_requested_bands``, ``band_clamped``, ``band_note``, ``band_policy``) and a chunked plan's
per-body ``chunk_bands``.  That is what makes a composition record state whether the clamp fired and
what band counts the banded+chunked merge produced, instead of only that it ran.  Null-never-zero: a
plan nobody can describe records ``available: false`` with every field ``null`` plus a note.

Where the time goes / where the reuse went
------------------------------------------
The ten-stage split (``stages.py``) and the content-addressed plan-key blocks (``content.py``) are
assembled by ``econ_child.py`` and shipped exactly as before: ``first_call.stage_ms`` /
``sr_fitness.stage_ms`` / ``generations[i].stage_ms`` / ``summary.{cold_stage_ms,warm_stage_ms,
compile_share}`` and the ``content`` blocks with their ``attempts``/``hit_rate``.  Every read is
getattr-probed: a tree without ``batch_stage_stats()`` / ``batch_content_stats()`` records
``available: false`` with EVERY counter ``null`` and a note -- never a fabricated ``0``.

Sweeping a knob without a new arm: ``--knob NAME=VALUE``
-------------------------------------------------------
``--knob`` (repeatable) overrides one route knob the harness owns (``common.ROUTE_ENV_VARS``) AFTER
both the arm's spec and the named route's spec, and BEFORE the policy is recorded, so a route's own
spec is overridable for the SAME variable (the explicit override wins, and the record says so).  That
is how the compiled-graph pool is swept: ``--knob EVOGP_BATCH_COMPILE_WORKERS=1`` and a companion run
with ``4`` (one config each -- the knob is per process).  An exported ``EVOGP_BATCH_COMPILE_WORKERS``
is cleared by the arm spec like every other route knob: use ``--knob``.

Generation loop
---------------
``--generations`` (default 8) is the number of real ``StandardPipeline.step()`` iterations, so the
round-2 protocol (``>= 6`` generations for a cold/warm/generation comparison) is expressible without
any further flag; the count is recorded per child and in the driver's ``protocol``.

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

    # the round-2 composition column: three batch configurations, one driver run
    $PY test/benchmark/evoxc/batch_economics.py --cells p256-L32-i8-N1024 \
        --routes batch-banded,batch-flat-chunked,batch-auto-chunked --generations 6 \
        --out-dir results/batch_opt_round2/composition

    # one config by hand (a child is a normal, self-contained record producer)
    $PY test/benchmark/evoxc/batch_economics.py --phase child \
        --cell p1000-L32-i8-N1024 --route batch --out "$TMPDIR/batch.json" --generations 8

``--tree-root DIR`` selects the checkout under measurement exactly like ``worker.py`` /
``cold_start.py`` (``<root>/src`` first on ``sys.path``).  ``--out-dir`` defaults to a FRESH
``results/batch/`` and refuses ``results/final`` and ``results/baseline`` (frozen evidence).  Nothing
here writes into ``src/``; the only library state touched is the route's documented env knobs
(``arms.apply_arm_env`` + the route's own spec) plus ``EVOXC_DISABLE_LOCATIONS``, defaulted to ``1``
when the caller left it unset (the driver records which of the two happened).

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
import econ_child  # noqa: E402  (the child-side loop internals; torch-free at import)
# ``_round`` is re-exported here on purpose: ``band_sweep.py`` imports it from this module (the
# generic driver helpers of this node live here), and the one implementation is econ_child's.
from econ_child import MEASUREMENT, _round  # noqa: E402,F401
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
#: The round-2 generation-loop protocol: a cold/warm/generation comparison needs at least this many
#: real ``StandardPipeline.step()`` iterations (``--generations`` expresses any count >= 1; the
#: default of 8 covers it).
ROUND2_MIN_GENERATIONS = 6

#: Route token -> ``{"arm", "knobs"}``.  The ONLY route table in this file: ``arm`` is a
#: ``common.EVOXC_ARMS`` key (its own registry spec is applied first), ``knobs`` is the ROUTE's own
#: extra env layer (applied after the arm's, before an explicit ``--knob``), and "is this a batch
#: route" stays answered by ``common.is_batch_arm`` -- nothing about the engine is re-declared here.
#: An empty ``knobs`` mapping means "the arm's own spec is the whole route", i.e. the tokens that
#: existed before the composition round keep their exact behaviour.
ROUTE_SPECS: Dict[str, Dict[str, Any]] = {
    "batch": {"arm": "evoxc-batch-iree", "knobs": {}},
    "batch-flat": {"arm": "evoxc-batch-iree-flat", "knobs": {}},
    "dense": {"arm": "evoxc-dense-iree", "knobs": {}},
    # round 2, work item 3 -- the composition column
    "batch-banded": {
        "arm": "evoxc-batch-iree",
        "knobs": {"EVOGP_BATCH_BANDS": "auto", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    },
    "batch-flat-chunked": {
        "arm": "evoxc-batch-iree",
        # CHUNK_BANDS deliberately UNSET: off, a band request forces the single-body path, so this
        # is the per-chunk-unbanded control next to batch-auto-chunked.
        "knobs": {"EVOGP_BATCH_BANDS": "none", "EVOGP_BATCH_COMPILE_WORKERS": "4"},
    },
    "batch-auto-chunked": {
        "arm": "evoxc-batch-iree",
        "knobs": {"EVOGP_BATCH_BANDS": "auto", "EVOGP_BATCH_CHUNK_BANDS": "1",
                  "EVOGP_BATCH_COMPILE_WORKERS": "4"},
    },
    # round 2, work item 3 -- the PINNED-SIZE composition: the same banded+chunked shape as
    # batch-auto-chunked with the ``auto`` band policy replaced by a fixed band size of 16 (the
    # round-2 minimax), so the pair separates the band policy from the composition itself.
    "batch-b16-banded": {
        "arm": "evoxc-batch-iree",
        # The single-run numeric-default baseline for the loop: pinned band size 16, single body.
        "knobs": {"EVOGP_BATCH_BAND_SIZE": "16", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    },
    "batch-b16-chunked": {
        "arm": "evoxc-batch-iree",
        # The candidate composition default: pinned band size 16 + the composition opt-in + pool 4.
        "knobs": {"EVOGP_BATCH_BAND_SIZE": "16", "EVOGP_BATCH_CHUNK_BANDS": "1",
                  "EVOGP_BATCH_COMPILE_WORKERS": "4"},
    },
}
#: Route token -> registry arm name (derived, so the two can never disagree).
ROUTES: Dict[str, str] = {token: spec["arm"] for token, spec in ROUTE_SPECS.items()}

#: Environment variables the driver forwards verbatim to every child (on top of the inherited
#: environment), so a record shows the configuration window it ran in.  The batch knobs are named
#: explicitly: the child clears and re-applies them per route, but an operator's export must still
#: reach the process to be visible in its records.
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


def route_spec(route: str) -> Dict[str, Any]:
    """The named route's spec, validated against the harness's own housekeeping table.

    A spec that pins a variable outside :data:`common.ROUTE_ENV_VARS` is a harness bug (that set is
    exactly what ``arms.apply_arm_env`` clears, so an unlisted name would leak between routes): it is
    reported by name here instead of silently measuring an inherited value.
    """
    spec = ROUTE_SPECS[route]
    for name in spec.get("knobs") or {}:
        if name not in common.ROUTE_ENV_VARS:
            raise SystemExit(
                f"route {route!r} pins {name!r}, which is not a route knob this harness clears; "
                f"known knobs: {', '.join(common.ROUTE_ENV_VARS)}"
            )
    return spec


def resolve_routes(text: str) -> List[str]:
    """``"batch,dense"`` -> route tokens, validated against :data:`ROUTE_SPECS`."""
    routes: List[str] = []
    for token in _tokens(text):
        if token not in ROUTE_SPECS:
            raise SystemExit(
                f"unknown route {token!r}; known: {', '.join(ROUTE_SPECS)} "
                f"(arms: {', '.join(sorted(set(ROUTES.values())))})"
            )
        route_spec(token)  # a spec naming an unmanaged variable fails before any child exists
        if token not in routes:
            routes.append(token)
    if not routes:
        raise SystemExit("--routes selected nothing")
    return routes


def route_is_batch(route: str) -> bool:
    """Does this route token measure the batch engine?  (The arm's spec decides.)"""
    return common.is_batch_arm(ROUTES[route])


def route_knobs_of(route: str) -> Dict[str, str]:
    """The route's own env layer (``{}`` when the route is just its arm)."""
    return {name: str(value) for name, value in (route_spec(route).get("knobs") or {}).items()}


# --- route-knob overrides (pure: validated without torch) -------------------
#: The batch knob spec key the compile-pool sweep uses (``common.BATCH_KNOBS`` owns the
#: env-var name, so the sweep command and the record cannot drift apart).
COMPILE_WORKERS_KNOB = "compile_workers"


def knob_env_name(spec_key: str) -> Optional[str]:
    """The env-var name ``common.BATCH_KNOBS`` gives ``spec_key`` (``None`` when unregistered)."""
    for entry in common.BATCH_KNOBS:
        if entry[0] == spec_key:
            return entry[1]
    return None


def parse_knobs(texts: Optional[Sequence[str]]) -> Dict[str, str]:
    """``["EVOGP_BATCH_COMPILE_WORKERS=4"]`` -> ``{"EVOGP_BATCH_COMPILE_WORKERS": "4"}``.

    Pure on purpose (torch-free, evogp-free): the driver validates the override before spawning a
    child that would pay a multi-minute compile.  A name outside :data:`common.ROUTE_ENV_VARS` is
    rejected by name (the known names are listed), a token without ``=`` is rejected, and an empty
    value is rejected too -- the engine would read ``""`` as unset, which silently measures the
    default instead of the sweep the caller asked for.
    """
    knobs: Dict[str, str] = {}
    for text in texts or ():
        name, sep, value = str(text).partition("=")
        name, value = name.strip(), value.strip()
        if not sep:
            raise SystemExit(
                f"--knob {text!r} is not NAME=VALUE; known names: {', '.join(common.ROUTE_ENV_VARS)}"
            )
        if name not in common.ROUTE_ENV_VARS:
            raise SystemExit(
                f"--knob {text!r}: {name!r} is not a route knob this harness may override; known "
                f"names: {', '.join(common.ROUTE_ENV_VARS)}"
            )
        if not value:
            raise SystemExit(
                f"--knob {text!r}: empty value -- the engine would read it as unset and measure the "
                "default, so an empty override is rejected instead of silently ignored"
            )
        knobs[name] = value
    return knobs


# --- the ONE env layer order (used by the child AND by the dry run) ---------
def apply_route_env(
    arm: str,
    route_knobs: Optional[Dict[str, str]] = None,
    override: Optional[Dict[str, str]] = None,
    cell: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Clear every route knob, then apply ``arm -> route -> --knob`` (the ONE precedence).

    ``arms.apply_arm_env`` is the clearing step: it pops every :data:`common.ROUTE_ENV_VARS` entry
    (so an exported -- or leftover -- ``EVOGP_BATCH_*`` / ``EVOGP_DENSE_MAX_TREE_LEN`` can never
    decide what this route measures) and writes the ARM's own spec.  The named route's spec is
    written next, and the explicit ``--knob`` override last, so an operator's explicit override wins
    for the SAME variable.  ``knobs`` in the return value is the resolved mapping that actually
    reached the engine (arm + route + override), which is what ``env.applied_knobs`` records.
    """
    import arms as arms_mod  # noqa: PLC0415 - torch-free at import, but only needed here

    applied_spec = arms_mod.apply_arm_env(arm, cell)
    applied: Dict[str, str] = dict(applied_spec.get("knobs") or {})
    for layer in (route_knobs or {}, override or {}):
        for name, value in layer.items():
            os.environ[name] = str(value)
            applied[name] = str(value)
    return {"eval_mode": applied_spec["eval_mode"], "backend": applied_spec["backend"],
            "knobs": applied}


def route_knob_table(
    arm: str,
    route_knobs: Optional[Dict[str, str]] = None,
    override: Optional[Dict[str, str]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per route variable: its value, the value the environment carries, and WHO set it.

    The source vocabulary is the layer order of :func:`apply_route_env`, most explicit first:
    ``harness-knob`` (the ``--knob`` override) > ``route`` (the named route's own spec) > ``arm``
    (the registry arm's knobs) > ``env`` (a value nobody in the harness wrote) -- and, for a variable
    the environment does not carry at all, whatever ``common.batch_policy`` decided:
    ``module`` (the engine's own resolver) / ``harness-default`` (the documented default) /
    ``unset``.  ``value`` is the RESOLVED value (so an unset ``EVOGP_BATCH_COMPILE_WORKERS`` still
    states the pool width the engine used), ``env_value`` is what the process environment holds
    (``None`` = really unset), and this table is the ONE provenance block a record carries.
    """
    arm_knobs = {name: str(value)
                 for name, value in ((common.EVOXC_ARMS.get(arm) or {}).get("knobs") or {}).items()}
    route_knobs = {str(key): str(value) for key, value in (route_knobs or {}).items()}
    override = {str(key): str(value) for key, value in (override or {}).items()}
    sources = (policy or {}).get("knobs_source") or {}
    spec_keys = {env: key for key, env, _default, _candidates in common.BATCH_KNOBS}
    defaults = {env: default for _key, env, default, _candidates in common.BATCH_KNOBS}
    table: Dict[str, Dict[str, Any]] = {}
    for name in common.ROUTE_ENV_VARS:
        env_value = os.environ.get(name)
        if name in override:
            source = "harness-knob"
        elif name in route_knobs:
            source = "route"
        elif name in arm_knobs and arm_knobs[name] == env_value:
            source = "arm"
        elif env_value is not None:
            source = "env"
        else:
            source = sources.get(spec_keys.get(name)) or "unset"
        spec_key = spec_keys.get(name)
        value = (policy or {}).get(spec_key) if spec_key else None
        if value is None:
            value = env_value if env_value is not None else defaults.get(name)
        entry: Dict[str, Any] = {"env": name, "value": value, "env_value": env_value,
                                 "source": source}
        if spec_key:
            entry["spec_key"] = spec_key
        table[name] = entry
    return table


def compile_workers_block(route_knobs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The resolved compile-pool knob, as the child's ``protocol`` records it (batch routes only).

    A view of :func:`route_knob_table`, so ``source`` is the layer that really set it -- a
    ``harness-default``/``module`` source means no route spec and no override supplied the value,
    never a fabricated one.
    """
    name = knob_env_name(COMPILE_WORKERS_KNOB)
    entry = (route_knobs or {}).get(name)
    return dict(entry) if entry else {"env": name, "value": None, "source": None}


def format_route_env(route_knobs: Dict[str, Dict[str, Any]]) -> str:
    """One-line render of the resolved route environment (the dry run's per-config evidence)."""
    set_parts, unset = [], []
    for name, entry in route_knobs.items():
        value = entry.get("env_value")
        if value is None:
            value = entry.get("value")
        if value is None:
            unset.append(name)
        else:
            set_parts.append(f"{name}={value}({entry.get('source')})")
    text = " ".join(set_parts) or "(no route knob carries a value)"
    return text + (f" | unset: {', '.join(unset)}" if unset else "")


def format_knob_spec(route: str) -> str:
    """``[EVOGP_BATCH_BANDS=auto, EVOGP_BATCH_COMPILE_WORKERS=1]`` for a route's own spec."""
    knobs = route_knobs_of(route)
    return "[" + ", ".join(f"{name}={value}" for name, value in knobs.items()) + "]"


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
POLICY_FIELDS = ("band_size", "bands", "alignment", "switch_ops", "min_pop", "compile_workers",
                 "module_present", "mode_supported")


def format_policy(policy: Dict[str, Any]) -> str:
    """One-line render of ``common.batch_policy`` (the dry run's evidence line)."""
    fields = " ".join(f"{key}={policy.get(key)}" for key in POLICY_FIELDS)
    return f"{fields} declared_modes={policy.get('declared_modes')!r}"


def policy_for_route(
    route: str,
    src: str,
    cell: Optional[Dict[str, Any]] = None,
    override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """``common.batch_policy(src)`` resolved with the route's own env layers applied around it.

    It goes through :func:`apply_route_env` -- the SAME clearing + layering the child performs -- so
    the dry run reports the bands / pool width / chunk-bands the child will really read, including a
    route spec and any ``--knob`` override.  The environment is restored afterwards (minus whatever
    the caller already had unset): the dry run resolves several routes in one process.
    """
    spec = route_spec(route)
    saved = {name: os.environ.get(name) for name in common.ROUTE_ENV_VARS}
    try:
        apply_route_env(spec["arm"], route_knobs_of(route), override, cell)
        policy = common.batch_policy(src) if route_is_batch(route) else None
        table = route_knob_table(spec["arm"], route_knobs_of(route), override, policy)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return {"policy": policy, "route_knobs": table}


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


# --- dry run (torch-free by construction: no env_block) ---------------------
def dry_run(
    args: argparse.Namespace,
    cells: Sequence[Dict[str, Any]],
    routes: Sequence[str],
    src: str,
    override: Optional[Dict[str, str]] = None,
) -> int:
    """Print every planned config, the knob override, then where the batch routes resolve here.

    One line per (cell, route) states the record path and the route's OWN knob spec, followed by the
    resolved route environment (every variable the child clears, its value and the layer that set it
    -- so ``bands`` / ``compile_workers`` / ``chunk_bands`` are readable per route) and, for a batch
    route, the engine policy the child will record.
    """
    warnings_seen: List[str] = []
    for cell in cells:
        for route in routes:
            print("would run", cell["id"], route, f"({ROUTES[route]})",
                  f"route-knobs={format_knob_spec(route)}",
                  "->", record_path(args.out_dir, cell["id"], route))
            resolved = policy_for_route(route, src, cell, override)
            print("  route env", route, format_route_env(resolved["route_knobs"]))
            policy = resolved["policy"]
            if policy is not None:
                print("  batch policy", route, f"({ROUTES[route]})", format_policy(policy))
                # The point of a dry run is to learn this BEFORE a multi-minute compile: the
                # resolved tree may not ship the route yet, and then every batch config fails.
                if not common.batch_engine_present(src):
                    warnings_seen.append(
                        f"WARNING: {src} has no {common.BATCH_MODULE_REL} -- EVOGP_EVAL_MODE=batch "
                        "cannot run here: every batch config would fail its preflight and write no "
                        "record (the dense configs still run)")
                elif not policy["mode_supported"]:
                    warnings_seen.append(
                        f"WARNING: {src} ships {common.BATCH_MODULE_REL} but its "
                        f"dense_engine.EVAL_MODES ({policy['declared_modes']!r}) does not declare "
                        "'batch' -- every batch config would fail its preflight and write no record")
    if override:
        # Stated as its own fact: the override is applied AFTER the arm's spec AND the named
        # route's spec, so it can move the policy a line above -- and it is the only way to sweep
        # the compile pool.
        for name, value in override.items():
            print("knob override", f"{name}={value}",
                  "(applied after the arm's own spec and the route's spec, before the policy is "
                  "recorded: it wins for the SAME variable)")
    for message in dict.fromkeys(warnings_seen):  # one per (cell, route) is noise
        print(message)
    if not any(route_is_batch(route) for route in routes):
        print("no batch route selected: nothing to resolve")
    print(f"{len(cells) * len(routes)} configs planned -> {os.path.abspath(args.out_dir)}")
    return 0


# --- driver (stdlib + common only: no torch, one child process per config) --
def run_driver(args: argparse.Namespace) -> int:
    cells = resolve_cells(args.cells)
    routes = resolve_routes(args.routes)
    override = parse_knobs(getattr(args, "knob", None))
    src = resolve_src(args.tree_root)
    if args.dry_run:
        return dry_run(args, cells, routes, src, override)

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
            # the named routes' own env layers, next to the arm each token resolves to
            "route_specs": {route: route_knobs_of(route) for route in routes},
            "generations": args.generations, "warmup": args.warmup, "reps": args.reps,
            "seed": args.seed, "timeout_s": args.timeout,
            "generation_loop": f"StandardPipeline.step() x --generations (>= {ROUND2_MIN_GENERATIONS} "
                               "for the round-2 cold/warm/generation comparison)",
            "tree_root": os.path.abspath(args.tree_root), "src": src,
            "pythonpath": env["PYTHONPATH"],
            "evoxc_disable_locations": disable_locations_provenance(env),
            "batch_engine_present": common.batch_engine_present(src),
            # The --knob override forwarded to EVERY child (empty when none was given): it is
            # applied after the arm's own spec and the route's spec, so it is named here rather than
            # only in a record.
            "knobs_override": dict(override),
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
    if override:
        driver_record["notes"].append(
            "knob override " + ", ".join(f"{key}={value}" for key, value in override.items())
            + ": forwarded to every child as repeated --knob args and applied AFTER the arm's own "
            "knobs and the named route's spec, so the recorded batch_policy shows the value the "
            "engine really read")
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
        for name, value in override.items():
            cmd += ["--knob", f"{name}={value}"]
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

    The env layers are ``arm -> named route -> --knob`` (:func:`apply_route_env`), all applied BEFORE
    the policy is resolved, so every recorded knob block carries the value the engine really read and
    ``protocol.route_knobs`` states which layer set it.  The measurement itself -- cold call, warm
    medians, the generation loop, the record -- is ``econ_child.measure``.
    """
    for flag, value in (("--cell", args.cell), ("--route", args.route), ("--out", args.out)):
        if not value:
            raise SystemExit(f"--phase child needs {flag}")
    if args.route not in ROUTE_SPECS:
        raise SystemExit(f"unknown route {args.route!r}; known: {', '.join(ROUTE_SPECS)}")

    override = parse_knobs(getattr(args, "knob", None))
    arm = ROUTES[args.route]
    spec_knobs = route_knobs_of(args.route)
    src = install_tree_root(args.tree_root)
    cell = common.parse_cell(args.cell)

    knobs = apply_route_env(arm, spec_knobs, override, cell)  # one route per process
    is_batch = common.is_batch_arm(arm)
    batch_policy = common.batch_policy(src) if is_batch else None
    if batch_policy is not None:
        problem = batch_preflight(args.route, batch_policy, src)
        if problem:
            raise SystemExit(problem)
    route_knobs = route_knob_table(arm, spec_knobs, override, batch_policy)

    import torch  # noqa: F401  (the child needs a device; econ_child/evogp import it too)

    common.import_evogp()
    from evogp.tree import dense_engine, evoxc_engine

    batch_module = None
    if is_batch:
        try:
            from evogp.tree import batch_engine as batch_module
        except ImportError:  # the route can be absent on an older tree
            batch_module = None

    record = econ_child.measure(econ_child.Measurement(
        cell=cell, route=args.route, arm=arm, tree_root=args.tree_root,
        generations=args.generations, warmup=args.warmup, reps=args.reps, seed=args.seed,
        is_batch=is_batch, batch_policy=batch_policy, applied_knobs=knobs["knobs"],
        route_knobs=route_knobs, route_spec_knobs=spec_knobs, override=override,
        eval_mode=knobs["eval_mode"], backend=knobs["backend"],
        evoxc_engine=evoxc_engine, dense_engine=dense_engine, batch_module=batch_module,
        surfaces=econ_child.CounterSurfaces(batch_module, is_batch), note=args.note,
    ))
    if is_batch:
        record["batch_policy"] = batch_policy
        record["protocol"]["batch_compile_workers"] = compile_workers_block(route_knobs)

    common.write_json(args.out, record)
    summary, warm_median_ms = record["summary"], record["summary"]["warm_median_ms"]
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
                        help=f"comma-separated route tokens: {', '.join(ROUTE_SPECS)}")
    parser.add_argument("--tree-root", default=DEFAULT_TREE_ROOT, help="tree root whose src/ holds evogp")
    parser.add_argument("--evoxc-root", default=DEFAULT_EVOXC_ROOT, help="evoxc checkout on PYTHONPATH")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="fresh result directory (never results/final or results/baseline)")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="StandardPipeline.step() iterations (the generations for the hit rate); "
                             "the round-2 comparison needs >= %d" % ROUND2_MIN_GENERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warm sr_fitness calls")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="timed sr_fitness samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="torch.manual_seed for the workload")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-config wall cap in seconds (a merged-graph compile may take minutes)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--knob", action="append", default=None, metavar="NAME=VALUE",
                        help="route-knob override, repeatable; names are validated against "
                             "common.ROUTE_ENV_VARS (the EVOGP_BATCH_* set plus "
                             "EVOGP_DENSE_MAX_TREE_LEN) and the value is applied AFTER the arm's own "
                             "spec and the named route's spec, before the policy is recorded. This is "
                             "how the compiled-graph pool is swept without a new arm: --knob "
                             "EVOGP_BATCH_COMPILE_WORKERS=1, then a second run with 4 (one config "
                             "per value: the knob is per process)")
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
