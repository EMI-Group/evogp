"""Band-size / merge-composition sweep of the batch route (``EVOGP_EVAL_MODE=batch``).

What this measures
------------------
``EVOGP_BATCH_BAND_SIZE`` decides how many deduped programs one merge band holds (the band count
is then the engine's own ``ceil(n_functions / band_size)``, clamped into ``1..n_functions``), and
``EVOGP_BATCH_BANDS`` is the other half of the same knob table: an explicit count ``K``,
``none``/``flat`` (ONE unbanded merge) or ``auto`` (evoxc's static cost model picks ``K``, which
the binding policy may then CLAMP on iree/rocm -- ``K <= 15`` because one exported dispatch binds
at most 16 tensors, and a band carries ``>= 8`` functions).  A single per-call number cannot price
that knob, so the sweep runs it on FIXED cells: population, data and seed are identical across
configs (``common.build_workload``), so the ONLY thing that changes between two records of a cell
is the route environment.

A config is a knob SPEC (:data:`CONFIG_KNOBS`), not a hard-coded band size, so the sweep also
covers the compiled-graph POOL and the banded+chunked COMPOSITION.  ``b8``/``b16``/``b32``/``b64``/
``flat``/``auto`` are the BAND-PARTITION configs and stay single-body
(``EVOGP_BATCH_COMPILE_WORKERS=1``) on purpose: chunking changes the merge SHAPE (one compiled body
per chunk, each with its own band partition) and would confound the band knob with the pool width.
``auto-cw4`` (``auto`` + the ``EVOGP_BATCH_CHUNK_BANDS=1`` composition opt-in + pool 4),
``b16-cw4`` (a PINNED band size 16 + the composition opt-in + pool 4, i.e. ``auto-cw4`` with the
band POLICY removed) and ``flat-cw4`` (flat merge + pool 4, no composition opt-in: an unbanded
chunked plan needs none) run the pool deliberately, so ``--workers`` is asserted only for the
band-partition set.  A pure pool-width sweep stays
``batch_economics.py --knob EVOGP_BATCH_COMPILE_WORKERS=N``.

The BAND-COUNT pins (round 3) are the other arm of the same question: ``b<N>`` asks the size MODEL
for a count, while ``k15`` / ``k7`` pin the count ITSELF at what the device can load
(``EVOGP_BATCH_BANDS=15`` / ``=7``, single-body), and ``k15-cw4`` / ``k7-cw4`` run the same pin
through the banded+chunked composition.  An explicit ``K`` is the ONE band request the engine
VALIDATES HARD: :func:`evogp.tree.batch_bands.plan_band_spec` raises ``ValueError`` for a count the
device cannot load (it never clamps an explicit ``K``; ``auto`` and the size model are clamped), so
a pin is only runnable where the deduped function count supports it -- the policy's cap is
``min(max_bands, n_functions, n_functions // min_band_size)``, i.e. ``K <= 15`` bands of
``>= 8`` merged functions on the CUDA/ROCm HAL, so ``K`` needs ``n_functions >= 8 * K``.  That makes
the pins POPULATION-SCOPED: on this grid ``n_functions`` is 225/232/245 (the two pop-256 cells) and
63/64 (``p64-L64-i32-N64``), so ``k15`` is legal only on a pop-256 (or larger) cell, while ``k7``
is legal on all three.  :func:`check_cell_legality` states that per (cell, config) pair -- a
WARNING in ``--dry-run``, a refusal before any child starts otherwise -- instead of paying a
compile for a record the engine is guaranteed to reject.

Per (cell, config) the child measures three value workloads in a FIXED order -- ``sr_fitness``,
``batch_forward``, ``forward`` -- recording per workload the cold first call (``first_call_ms``:
the merged-graph compile of that plan), the warm ``samples_ms`` with their median/min/max/mean, the
value ``digest``, the merged-graph counter deltas, the ten-stage split (``stage_ms`` /
``stage_first_call`` / ``stage_ms.plan`` from ``stages.py``, null-never-zero when the route does
not expose the counters) and the ``band_report`` of the plan that served it -- the merge shape
next to what the cost model chose (``band_requested`` / ``band_requested_bands`` = the ``K``
BEFORE clamping / ``band_clamped`` / ``band_note`` / ``band_policy``), with the two cost estimates
``null`` unless ``auto`` ran the model.  Every workload is also diffed against the independent CPU
oracle (``src/evogp/tests/reference_eval.py``) with ``validate.compare`` into
``record['validation']``, so cross-config agreement is checkable: the band partition is a
scheduling decision and may not change the value.

That child half -- the measurement loop, the oracle wiring and the record skeleton -- lives in
``band_measure.py``; this file is the DRIVER: CLI, cell/config resolution, the route env of each
config, the idle-device guard, one child process per (cell, config) and ``sweep.json``.  A process
per config is mandatory (the compile cache, the executable memo and the merged-graph tables are
per process), and the children run ONE AT A TIME behind ``campaign.guard``.

``--cold-only`` (the round-3 re-measure path) measures ONLY the cold first call per workload: one
synchronized call, no warmup, no timed reps, and every warm field recorded ``null``
(``samples_ms`` an empty list) instead of a fabricated median.

Cells come from :func:`common.parse_cell`, so ANY ``p<pop>-L<tree_len>-i<input_len>-N<rows>`` id
(plus an optional ``-d<depth>`` suffix) is runnable without registering it -- the round-3 population
column is addressed by its own id, e.g. ``--cold-only --cells p1000-L32-i8-N1024 --configs b16-cw4``.

Reproduce on the GPU box
------------------------
::
    PY=/tmp/rocvenv/bin/python
    export EVOXC_DISABLE_LOCATIONS=1
    $PY test/benchmark/evoxc/band_sweep.py --dry-run     # planned env of every (cell, config, workload)
    $PY test/benchmark/evoxc/band_sweep.py               # default plan: 3 cells x 9 configs into
                                                         # results/batch_opt_round3/band, evoxc_r4
    $PY test/benchmark/evoxc/band_sweep.py --cold-only --cells p64-L64-i32-N64 --configs auto
    $PY test/benchmark/evoxc/band_sweep.py --child p64-L64-i32-N64 --config b32 \
        --workloads sr_fitness --warmup 1 --reps 2 --out "$TMPDIR/band.json"
    # round 3: the band-count pins at the binding cap (k15* only on the pop-256 cells)
    $PY test/benchmark/evoxc/band_sweep.py --cells p256-L32-i8-N1024,p256-L64-i32-N1024 \
        --configs k15,k15-cw4 --out-dir results/batch_opt_round3/warm --evoxc-root /tmp/evoxc_r3
    $PY test/benchmark/evoxc/band_sweep.py --cold-only --cells p1000-L32-i8-N1024 \
        --configs b16-cw4 --out-dir results/batch_opt_round3/cold --evoxc-root /tmp/evoxc_r3

``--tree-root DIR`` selects the checkout under measurement like ``worker.py`` / ``batch_economics.py``
(``<root>/src`` first on ``sys.path``); the child's ``PYTHONPATH`` is exactly ``[<src>,
<evoxc-root>, <this node dir>]``, so the foreign ``/root/evoxc`` checkout is deliberately absent.
``EVOXC_DISABLE_LOCATIONS`` is defaulted to ``1`` (tracing ~4x faster, not part of the engine's
cache key) and the driver records which of the two happened.

Record layout: one JSON per (cell, config) at ``<out-dir>/<cell>/<config>.json`` (schema
``evogp-bench-evoxc-v1-band-sweep``) plus the child's ``.log`` and one driver index
``<out-dir>/sweep.json`` (``...-band-sweep-driver``); ``results/{final,baseline,batch,
batch_opt_round1}`` are frozen evidence and refused as ``--out-dir``.  A config the route cannot
run still writes a record -- ``status: "failed"`` with ``error`` / ``error_type`` /
``traceback`` -- because a failure is a recorded outcome, never a crash and never a silent skip.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import band_measure  # noqa: E402  (the child half: measurement + record skeleton, torch-free)
import common  # noqa: E402  (stdlib-only at import time)
from batch_economics import (  # noqa: E402  (the generic driver helpers of this node)
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
#: The round-2 config set (the band partition, the ``auto`` policy and the composition).  The
#: round-3 band-count pins (:data:`BAND_CAP_CONFIGS`) are deliberately NOT defaulted: they are
#: population-scoped (``k15``/``k15-cw4`` need a pop-256-or-larger cell), so a default run over
#: the default cells would select an illegal pair -- the round passes ``--configs k15,k7,...``
#: per cell set instead, which :func:`check_cell_legality` validates.
DEFAULT_CONFIGS = "b8,b16,b32,b64,flat,auto,auto-cw4,b16-cw4,flat-cw4"
DEFAULT_WORKLOADS = "sr_fitness,batch_forward,forward"
#: The measurement order is FIXED (never the caller's order): every record of a cell then
#: pays for its three plans in the same sequence, so two configs are comparable call by call.
WORKLOAD_ORDER: Tuple[str, ...] = ("sr_fitness", "batch_forward", "forward")
DEFAULT_TREE_ROOT = "src"
#: The evoxc checkout this round runs against (NOT ``/root/evoxc``: the child's PYTHONPATH is
#: built from this value, so an implicit foreign checkout can never slip in).  Round 3's snapshot.
DEFAULT_EVOXC_ROOT = "/tmp/evoxc_r4"
#: Round 3's own band-sweep tree: ``results/batch_opt_round2`` is frozen evidence, so the
#: no-``--out-dir`` default must not resolve into it.
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "batch_opt_round3", "band")
DEFAULT_WARMUP = 1
DEFAULT_REPS = 5
DEFAULT_SEED = 0
#: ``--workers`` is the expected pool width of the BAND-PARTITION configs (:data:`BAND_PARTITION_CONFIGS`),
#: which stay single-body; the composition configs pin their own pool in :data:`CONFIG_KNOBS`.
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
#: Frozen evidence: never written in place (round 2's own ``results/batch_opt_round2`` included).
FROZEN_RESULT_DIRS = ("final", "baseline", "batch", "batch_opt_round1", "batch_opt_round2")
MEASUREMENT = band_measure.MEASUREMENT
SCHEMA = band_measure.SCHEMA
SCHEMA_DRIVER = SCHEMA + "-driver"

#: Config token -> the route knobs it sets (everything it does not name is UNSET).  A token is a
#: SPEC, not a band size, so one table covers the band partition, the ``auto`` band policy and
#: the banded+chunked composition.  Every entry states its pool width explicitly: the band
#: partition stays single-body (``EVOGP_BATCH_COMPILE_WORKERS=1``) so a difference between two
#: of its records is the band partition alone, while the ``-cw4`` composition configs run the
#: compiled-graph pool on purpose.  ``b<N>`` pins the band SIZE (the engine resolves
#: ``K = ceil(n_functions / N)``, clamped into ``1..n_functions``); ``flat`` / ``flat-cw4`` pin
#: ``EVOGP_BATCH_BANDS=none`` (one unbanded merge, ``band_strategy == "flat"``); ``auto`` /
#: ``auto-cw4`` hand the partition to evoxc's static cost model (``band_requested == "auto"``,
#: then possibly clamped).  ``EVOGP_BATCH_CHUNK_BANDS`` is set by ``auto-cw4`` and ``b16-cw4``
#: (the two banded+chunked compositions): an unbanded chunked plan needs no composition opt-in,
#: and an explicit ``EVOGP_BATCH_BANDS`` spelling without it forces the single-body path (a band
#: SIZE does not: a chunked plan keeps its band size per chunk).
CONFIG_KNOBS: Dict[str, Dict[str, str]] = {
    "b8": {"EVOGP_BATCH_BAND_SIZE": "8", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "b16": {"EVOGP_BATCH_BAND_SIZE": "16", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "b32": {"EVOGP_BATCH_BAND_SIZE": "32", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "b64": {"EVOGP_BATCH_BAND_SIZE": "64", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "flat": {"EVOGP_BATCH_BANDS": "none", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "auto": {"EVOGP_BATCH_BANDS": "auto", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    "auto-cw4": {
        "EVOGP_BATCH_BANDS": "auto",
        "EVOGP_BATCH_CHUNK_BANDS": "1",
        "EVOGP_BATCH_COMPILE_WORKERS": "4",
    },
    "flat-cw4": {"EVOGP_BATCH_BANDS": "none", "EVOGP_BATCH_COMPILE_WORKERS": "4"},
    #: The composition with the band POLICY removed: pinned band size 16 (the minimax band size
    #: of round 2) + the composition opt-in + pool 4, so a difference against ``auto-cw4`` is the
    #: ``auto`` band policy alone and a difference against ``b16`` is the composition alone.
    "b16-cw4": {
        "EVOGP_BATCH_BAND_SIZE": "16",
        "EVOGP_BATCH_CHUNK_BANDS": "1",
        "EVOGP_BATCH_COMPILE_WORKERS": "4",
    },
    #: ---- round 3: the band COUNT pinned at the device's binding cap -------------------------
    #: ``k15`` pins ``EVOGP_BATCH_BANDS=15`` -- the largest count the CUDA/ROCm HAL can load (one
    #: exported dispatch binds <= 16 tensors, the measured safe count is 15) -- instead of asking
    #: the size model for one.  It needs ``n_functions >= 15 * 8`` deduped programs, i.e. a pop-256
    #: cell (measured n = 225 / 232 / 245 here); the engine RAISES on an unloadable explicit ``K``,
    #: so :func:`check_cell_legality` keeps it off the pop-64 cells.
    "k15": {"EVOGP_BATCH_BANDS": "15", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    #: ``k7`` pins ``EVOGP_BATCH_BANDS=7`` -- the cap of a 63-function population (``63 // 8 = 7``),
    #: so it is the largest pin that is legal on EVERY cell of the round, and on a pop-256 cell it
    #: prices the same partition as the size model at band size ~32 (``b32``).
    "k7": {"EVOGP_BATCH_BANDS": "7", "EVOGP_BATCH_COMPILE_WORKERS": "1"},
    #: The same pins through the banded+chunked composition.  ``EVOGP_BATCH_BANDS`` stays a GLOBAL
    #: count in a chunked plan (each chunk receives ``ceil(K * chunk_n / n)``, capped per chunk by
    #: the same policy: 4 chunks of ~56 functions -> at most 7 bands each), so ``k15-cw4`` is the
    #: plan-level cap at pool 4 and ``k7-cw4`` splits to ~2 bands per chunk.  Both are validated at
    #: the PLAN's ``n_functions`` (the chunk split clamps, it never raises).
    "k15-cw4": {
        "EVOGP_BATCH_BANDS": "15",
        "EVOGP_BATCH_CHUNK_BANDS": "1",
        "EVOGP_BATCH_COMPILE_WORKERS": "4",
    },
    "k7-cw4": {
        "EVOGP_BATCH_BANDS": "7",
        "EVOGP_BATCH_CHUNK_BANDS": "1",
        "EVOGP_BATCH_COMPILE_WORKERS": "4",
    },
}
#: The config tokens in registry (render) order.  Appending a token here is the only edit a new
#: config needs: the dry run, the driver and the child all read :data:`CONFIG_KNOBS`.
CONFIGS: Tuple[str, ...] = tuple(CONFIG_KNOBS)
#: The BAND-PARTITION configs: single-body, no composition opt-in, so two of their records of a
#: cell differ by the band partition alone.
BAND_PARTITION_CONFIGS: Tuple[str, ...] = ("b8", "b16", "b32", "b64", "flat", "auto", "k15", "k7")
#: The COMPOSITION configs: they run the compiled-graph pool (``auto-cw4`` / ``b16-cw4``
#: additionally opt into banding each chunk), which is a different question from the band size.
COMPOSITION_CONFIGS: Tuple[str, ...] = ("auto-cw4", "flat-cw4", "b16-cw4", "k15-cw4", "k7-cw4")
#: The configs that pin an explicit band COUNT (the ``k*`` family).  They are the only ones with a
#: population floor: the engine validates an explicit ``K`` hard and raises instead of clamping.
BAND_CAP_CONFIGS: Tuple[str, ...] = ("k15", "k7", "k15-cw4", "k7-cw4")
#: The band knob of each config (exactly one per config, asserted in :func:`check_config_specs`).
#: Of the remaining ``common.ROUTE_ENV_VARS`` entries a spec may set the pool width
#: (``EVOGP_BATCH_COMPILE_WORKERS``) and the composition opt-in (``EVOGP_BATCH_CHUNK_BANDS``); every
#: other one stays UNSET, which :func:`config_env` asserts by requiring all of them by name.
BAND_KNOBS: Tuple[str, ...] = ("EVOGP_BATCH_BAND_SIZE", "EVOGP_BATCH_BANDS")

#: The route every config of this round runs.  The band/pool knob is added per config, and every
#: other ``common.ROUTE_ENV_VARS`` entry is explicitly UNSET (see :func:`config_env`).
BASE_ENV: Dict[str, str] = {
    "EVOGP_EVAL_MODE": band_measure.ROUTE_EVAL_MODE,
    "EVOGP_EVOXC_BACKEND": band_measure.ROUTE_BACKEND,
}

#: The oracle entry point of each workload -- single-sourced in ``band_measure`` (the child that
#: diffs against it); re-exported here because the record documents it per workload.
ORACLE_FNS: Dict[str, str] = band_measure.ORACLE_FNS

#: Raised by the child before any timing exists when the tree cannot run the route.
RouteUnavailable = band_measure.RouteUnavailable

#: The a-priori device binding cap the ``k*`` pins are written against, MIRRORED from the engine
#: (``src/evogp/tree/batch_bands.py``: ``SAFE_BAND_COUNTS`` = 15 and ``MIN_BAND_SIZE`` = 8 on both
#: ``cuda`` and ``rocm``; ``local_task`` is 31/1 and the numpy interpreter is uncapped).
#: :func:`band_cap_limits` refreshes it from an importable tree and falls back to this mirror, so a
#: dry run stays runnable on a box without torch/evoxc -- it then REPORTS ``source: "mirror"``
#: instead of pretending it read the policy.
BAND_CAP_MIRROR: Dict[str, Any] = {
    "device_kind": "cuda/rocm",
    "max_bands": 15,
    "min_band_size": 8,
}


# --- selections (pure: no torch, no evogp) ----------------------------------
def _tokens(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def band_cap_limits() -> Dict[str, Any]:
    """``{device_kind, max_bands, min_band_size, source}`` -- the GPU dispatch-binding cap.

    The engine's own table (``evogp.tree.batch_bands``) when the tree under measurement is
    importable, else :data:`BAND_CAP_MIRROR`; both GPU HAL kinds (``cuda``, ``rocm``) share one
    row, so the probe takes the tightest of whichever it finds.  A tree that cannot be imported is
    a REPORTED state (``source``/``note``), never a raise: this is a planning gate, not the policy.
    """
    try:
        from evogp.tree.batch_bands import MIN_BAND_SIZE, SAFE_BAND_COUNTS  # noqa: PLC0415

        kinds = [
            kind
            for kind in ("cuda", "rocm")
            if SAFE_BAND_COUNTS.get(kind) is not None
        ]
        if kinds:
            return {
                "device_kind": "/".join(kinds),
                "max_bands": min(int(SAFE_BAND_COUNTS[kind]) for kind in kinds),
                "min_band_size": max(int(MIN_BAND_SIZE.get(kind, 1)) for kind in kinds),
                "source": "evogp.tree.batch_bands",
            }
        return {
            **BAND_CAP_MIRROR,
            "source": "mirror",
            "note": "evogp.tree.batch_bands carries no cuda/rocm binding row",
        }
    except Exception as exc:  # noqa: BLE001 - an unimportable tree is a reported state
        return {**BAND_CAP_MIRROR, "source": "mirror", "note": repr(exc)}


def config_pinned_bands(config: str) -> Optional[int]:
    """The explicit band COUNT a config pins, or ``None`` (flat / ``auto`` / a band SIZE).

    Only an explicit ``EVOGP_BATCH_BANDS=K`` is validated HARD by the engine
    (``batch_bands.plan_band_spec`` raises for a count the device cannot load); the band-size
    model, ``auto`` and the flat spelling are clamped instead, so they have no population floor
    and no legality question to answer.
    """
    raw = CONFIG_KNOBS[config].get("EVOGP_BATCH_BANDS")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("", "none", "flat", "auto"):
        return None
    try:
        return int(text)
    except ValueError:
        raise SystemExit(
            f"config {config!r} pins EVOGP_BATCH_BANDS={raw!r}, which is neither an integer count "
            "nor one of the flat/none/auto spellings"
        )


def config_pop_floor(
    config: str, limits: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """The smallest population at which this config's pinned ``K`` can load (``None`` = no floor).

    ``pop >= K * min_band_size`` is NECESSARY, not sufficient: the policy's cap is
    ``min(max_bands, n_functions, n_functions // min_band_size)`` and ``n_functions`` is the DEDUPED
    count, which only exists once the population is built (measured: 225/232/245 at the two pop-256
    cells, 63/64 at ``p64-L64-i32-N64``).  So this gates what can never work, and the record states
    the rest (``cell.n_functions`` next to the plan's ``band_report``).
    """
    pinned = config_pinned_bands(config)
    if pinned is None:
        return None
    limits = band_cap_limits() if limits is None else limits
    return int(pinned) * int(limits["min_band_size"])


def cell_legality(
    cell: Dict[str, Any], config: str, limits: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Why this (cell, config) pair cannot run on this device, else ``None``."""
    pinned = config_pinned_bands(config)
    if pinned is None:
        return None
    limits = band_cap_limits() if limits is None else limits
    if pinned > int(limits["max_bands"]):
        return (
            f"{config} pins EVOGP_BATCH_BANDS={pinned} above the binding cap of "
            f"{limits['max_bands']} bands on {limits['device_kind']}: the engine raises for an "
            "explicit K it cannot load"
        )
    floor = int(pinned) * int(limits["min_band_size"])
    if int(cell["pop"]) < floor:
        return (
            f"{config} pins EVOGP_BATCH_BANDS={pinned} and needs {floor} merged functions "
            f"({pinned} bands x >= {limits['min_band_size']} per band on {limits['device_kind']}), "
            f"but {cell['id']} is a pop-{int(cell['pop'])} cell -- n_functions <= pop, so the "
            "engine's plan_band_spec RAISES ValueError on this explicit K (it never clamps one) "
            f"and the child would only write a failed record; run {config} on a cell with "
            f"pop >= {floor} (the p64 cells keep k7)"
        )
    return None


def check_cell_legality(
    cells: List[Dict[str, Any]], configs: List[str]
) -> List[str]:
    """Every illegal (cell, config) pair of one plan, as human-readable reasons.

    The gate is the engine's own explicit-``K`` rule, so an empty list means "no pair is refusable
    on the binding cap alone" -- the exact ``n_functions`` test stays with the engine and lands in
    the record if a population dedups below the floor.
    """
    limits = band_cap_limits()
    problems: List[str] = []
    for cell in cells:
        for config in configs:
            reason = cell_legality(cell, config, limits)
            if reason is not None:
                problems.append(f"{cell['id']} + {config}: {reason}")
    return problems


def resolve_configs(text: str) -> List[str]:
    """``"b8,auto,flat-cw4"`` -> config tokens, validated against :data:`CONFIGS`."""
    configs: List[str] = []
    for token in _tokens(text):
        if token not in CONFIG_KNOBS:
            raise SystemExit(
                f"unknown config {token!r}; known: {', '.join(CONFIGS)} (b<N> pins "
                "EVOGP_BATCH_BAND_SIZE=N, k<K> pins EVOGP_BATCH_BANDS=K -- an explicit band count "
                "the device must be able to load, flat* EVOGP_BATCH_BANDS=none, auto* =auto, "
                "-cw4 runs the pool at EVOGP_BATCH_COMPILE_WORKERS=4)")
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
        raise SystemExit(f"unknown workload(s) {', '.join(unknown)}; this sweep measures "
                         f"{', '.join(WORKLOAD_ORDER)} (gp_step is batch_economics.py's column)")
    workloads = [name for name in WORKLOAD_ORDER if name in requested]
    if not workloads:
        raise SystemExit("--workloads selected nothing")
    return workloads


def check_workers(workers: int) -> int:
    """The BAND-PARTITION configs run single-body: ``--workers`` is their expected pool width.

    The pool width is a CONFIG knob now, so this flag describes only the band-partition set (whose
    whole point is that the band partition alone differs between its records); ``auto-cw4`` /
    ``b16-cw4`` / ``flat-cw4`` run the pool and take their width from their own knob spec.
    """
    if int(workers) != 1:
        pools = ", ".join(f"{token}={CONFIG_KNOBS[token]['EVOGP_BATCH_COMPILE_WORKERS']}"
                          for token in COMPOSITION_CONFIGS)
        raise SystemExit(
            f"--workers {workers}: the BAND-PARTITION configs ({', '.join(BAND_PARTITION_CONFIGS)}) "
            "run the SINGLE-BODY path (EVOGP_BATCH_COMPILE_WORKERS=1) on purpose, because "
            "chunking changes the merge shape (one compiled body per chunk with its own band "
            "partition) and would confound the band knob with the pool width.  The pool width is "
            f"a CONFIG knob ({pools}); sweep it on its own with batch_economics.py --knob "
            "EVOGP_BATCH_COMPILE_WORKERS=N, or time a composition config"
        )
    return 1


def check_config_specs(configs: List[str]) -> None:
    """Assert each selected config's knob spec against the sweep's split (pure, no torch).

    Five invariants, all derived from the table itself: exactly ONE band knob and a pool width per
    config; a band-partition config is single-body with no composition opt-in; a composition config
    runs a pool (``> 1``); only a config that runs a pool may opt into ``EVOGP_BATCH_CHUNK_BANDS``
    (banding a single chunk is meaningless); and a pinned band COUNT (the ``k*`` family) must fit the
    device's binding cap (``K <= max_bands``), which no population can rescue.
    """
    limits = band_cap_limits()
    for config in configs:
        spec = CONFIG_KNOBS[config]
        band = [name for name in BAND_KNOBS if name in spec]
        pool = int(spec.get("EVOGP_BATCH_COMPILE_WORKERS", 0) or 0)
        chunk = spec.get("EVOGP_BATCH_CHUNK_BANDS")
        pinned = config_pinned_bands(config)
        if len(band) != 1:
            raise SystemExit(f"config {config!r} must name exactly one of {', '.join(BAND_KNOBS)},"
                             f" got {band}")
        if pool < 1:
            raise SystemExit(f"config {config!r} must state its pool width "
                             "(EVOGP_BATCH_COMPILE_WORKERS), so a record never inherits the caller's")
        if pinned is not None and pinned > int(limits["max_bands"]):
            raise SystemExit(
                f"config {config!r} pins EVOGP_BATCH_BANDS={pinned}, above the {limits['max_bands']}"
                f"-band binding cap of {limits['device_kind']} ({limits['source']}): the engine "
                "raises for an explicit K it cannot load, so no cell can rescue this config -- pin "
                f"{limits['max_bands']} or less")
        if config in BAND_PARTITION_CONFIGS and (pool != 1 or chunk is not None):
            raise SystemExit(f"config {config!r} is a BAND-PARTITION config, so it must stay "
                             f"single-body without the composition opt-in; got pool={pool}, "
                             f"chunk_bands={chunk!r}")
        if config in COMPOSITION_CONFIGS and pool <= 1:
            raise SystemExit(f"config {config!r} is a COMPOSITION config, so it must run the "
                             f"compiled-graph pool (EVOGP_BATCH_COMPILE_WORKERS > 1); got {pool}")
        if config not in BAND_PARTITION_CONFIGS + COMPOSITION_CONFIGS:
            raise SystemExit(f"config {config!r} is in neither BAND_PARTITION_CONFIGS nor "
                             "COMPOSITION_CONFIGS: register it in one of the two sets")
        if chunk is not None and pool <= 1:
            raise SystemExit(f"config {config!r} sets EVOGP_BATCH_CHUNK_BANDS={chunk} with no pool: "
                             "banding a chunk is only meaningful when something chunks")


# --- the route environment of one config (pure) -----------------------------
def config_env(config: str) -> Dict[str, Optional[str]]:
    """The exact route env one config runs with (``None`` = the child must UNSET it).

    EVERY ``common.ROUTE_ENV_VARS`` entry is named (band size / bands / switch ops / alignment /
    cache size / min pop / compile workers / chunk bands, plus ``EVOGP_DENSE_MAX_TREE_LEN``), so
    the child clears the whole table before writing its own value: a caller-exported knob can
    never leak into a config, and a knob the config does not set records ``<unset>``.  The two
    role knobs come from :data:`BASE_ENV`, the rest from the config's own spec;
    ``EVOGP_BATCH_MIN_POP`` stays UNSET on purpose -- the route's own floor (64) is what the sweep
    is priced against, and the ``p64`` cell sits exactly ON it.
    """
    if config not in CONFIG_KNOBS:
        raise SystemExit(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
    env: Dict[str, Optional[str]] = {name: None for name in common.ROUTE_ENV_VARS}
    env.update(BASE_ENV)
    env.update(CONFIG_KNOBS[config])
    missing = [name for name in common.ROUTE_ENV_VARS if name not in env]
    if missing:  # the table above is the only source: a name it forgot must not go unset
        raise SystemExit(
            f"config {config!r} does not name every common.ROUTE_ENV_VARS entry "
            f"(missing {', '.join(missing)}): an unset knob must be stated explicitly"
        )
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


def config_text(config: str) -> str:
    """One-line render of a config's knob spec (the dry run's table row)."""
    spec = CONFIG_KNOBS[config]
    band = ", ".join(f"{name}={spec[name]}" for name in BAND_KNOBS if name in spec)
    return (f"{config} -> {band}, pool={spec['EVOGP_BATCH_COMPILE_WORKERS']}, "
            f"chunk_bands={spec.get('EVOGP_BATCH_CHUNK_BANDS', '<unset>')}")


def config_policy(config: str, src: str) -> Dict[str, Any]:
    """``common.batch_policy(src)`` with this config's env applied (restored afterwards).

    Same precedence the child runs with, so the dry run and the driver index report the bands the
    child will really use.  ``common.batch_policy`` guards its imports, so this stays runnable on
    a box without torch (it then records ``module_present: false``).
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

    ``worker.py`` / ``batch_economics.py`` resolve a relative ``--tree-root`` against the CWD.
    This script is also run from inside ``test/benchmark/evoxc``, where the shipped default
    ``src`` does not exist, so the default falls back to the checkout containing this node; an
    explicit non-default path is used verbatim (a typo must fail loudly, never be redirected).
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
    """Put the sweep's ``evoxc`` checkout on ``sys.path``.

    ``common.batch_policy`` asks the tree's own knob resolvers and ``dense_engine`` imports
    ``evoxc``, so without this the driver would report ``declared_modes`` ``None`` and warn that
    the tree cannot run the route while the child (whose PYTHONPATH carries ``--evoxc-root``) runs
    it fine.  sys.path only, no import: on a box without ``evoxc``/torch the policy resolution
    stays guarded and reports the import error instead of raising here.
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
                f"results/{', results/'.join(FROZEN_RESULT_DIRS)} are never rewritten in place "
                f"-- point --out-dir at this round's own tree (default: {DEFAULT_OUT_DIR})")


# --- dry run (torch-free by construction: no env_block) ---------------------
def dry_run(
    args: argparse.Namespace, cells: List[Dict[str, Any]], configs: List[str],
    workloads: List[str], src: str,
) -> int:
    """Print every planned (cell, config, workload) with the exact env and resolved policy."""
    check_config_specs(configs)
    policies: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        for config in configs:
            mapping = config_env(config)
            if config not in policies:
                policies[config] = config_policy(config, src)
            policy = policies[config]
            for workload in workloads:
                print(
                    f"would run {cell['id']:<22} {config:<9} {workload:<13} "
                    f"-> {record_path(args.out_dir, cell['id'], config)} "
                    f"env[{env_text(mapping)}] policy[{format_policy(policy)}]"
                )
    print("config table: " + "; ".join(config_text(config) for config in configs))
    print(f"route env table: every config names all {len(common.ROUTE_ENV_VARS)} "
          "common.ROUTE_ENV_VARS entries (<unset> = the child clears it), so a caller-exported "
          "EVOGP_BATCH_* or EVOGP_DENSE_MAX_TREE_LEN value can never leak into a config")
    print("band count model: b<N> -> K = ceil(n_functions / N) clamped into 1..n_functions; "
          "k<K> -> K pinned as the band COUNT (an explicit EVOGP_BATCH_BANDS, which the engine "
          "validates HARD: it raises instead of clamping, and a chunked plan splits it "
          "proportionally per chunk); flat/flat-cw4 -> K = 1; auto/auto-cw4 -> the static cost "
          "model picks K and the binding policy may clamp it (K <= 15, a band >= 8 functions).  "
          "n_functions is the DEDUPED row count, known only after the population is built, so "
          "what the model chose is recorded per workload (band_report.n_functions / "
          "band_requested / band_requested_bands / band_clamped / band_note) instead of printed "
          "here.")
    limits = band_cap_limits()
    pinned_configs = [config for config in configs if config_pinned_bands(config) is not None]
    if pinned_configs:
        print("binding cap: " + ", ".join(
            f"{config} -> EVOGP_BATCH_BANDS={config_pinned_bands(config)} needs "
            f"pop >= {config_pop_floor(config, limits)}"
            for config in pinned_configs
        ) + f" ({limits['max_bands']} bands max, >= {limits['min_band_size']} merged functions "
            f"per band on {limits['device_kind']}, source={limits['source']})")
    if getattr(args, "cold_only", False):
        print("cold-only (--cold-only): ONE synchronized call per workload is measured, so "
              f"--warmup {args.warmup} and --reps {args.reps} are SKIPPED -- every warm field "
              "(samples_ms/median_ms/min_ms/max_ms/mean_ms, stage_ms) is recorded null or [] and "
              "never a fabricated 0")
    # The point of a dry run is to learn this BEFORE a multi-second-to-minute compile.
    policy = policies[configs[0]]
    if not common.batch_engine_present(src):
        print(f"WARNING: {src} has no {common.BATCH_MODULE_REL} -- EVOGP_EVAL_MODE=batch cannot "
              "run here: every config would write a failed record (status 'failed', no timings)")
    elif not policy.get("mode_supported"):
        print(f"WARNING: {src} ships {common.BATCH_MODULE_REL} but its dense_engine.EVAL_MODES "
              f"({policy.get('declared_modes')!r}) does not declare 'batch' -- every config "
              "would write a failed record")
    if any(int(cell["pop"]) < int(policy.get("min_pop") or 0) for cell in cells):
        print("WARNING: a selected cell is BELOW EVOGP_BATCH_MIN_POP "
              f"({policy.get('min_pop')}): the route declines it to specialized (p64 sits exactly "
              "ON the floor and is served by the batch route)")
    # A pinned band count is the one band request the engine REFUSES instead of clamping, so an
    # illegal pair is knowable here -- before a multi-second-to-minute compile is spent on it.
    problems = check_cell_legality(cells, configs)
    for reason in problems:
        print(f"WARNING: illegal plan (a real run refuses this pair): {reason}")
    if problems:
        print(f"WARNING: {len(problems)} of {len(cells) * len(configs)} planned (cell, config) "
              "pairs pin a band count this device cannot load: --dry-run lists them, a real run "
              "exits before its first child (restrict --cells per config family)")
    print(f"{len(cells) * len(configs)} configs x {len(workloads)} workloads planned "
          f"-> {os.path.abspath(args.out_dir)}")
    return 0


# --- environment plumbing (driver side) -------------------------------------
DIS_SOURCE_ENV = "EVOGP_BAND_SWEEP_DIS_SOURCE"


def child_env(
    args: argparse.Namespace, src: str, config: str, dis_source: str
) -> Dict[str, str]:
    """The environment one child runs with: PYTHONPATH, the tracing switch, its config env.

    ``PYTHONPATH`` is exactly ``[<src>, <evoxc-root>, <this node dir>]`` (the node dir carries
    ``common``/``band_measure``/``worker``/``validate``), so only ``--evoxc-root`` provides
    ``evoxc``.
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
    env[DIS_SOURCE_ENV] = dis_source
    return env


def _driver_env_block(
    args: argparse.Namespace, tree_root: str, src: str
) -> Optional[Dict[str, Any]]:
    """The driver's env block, or ``None`` on a box where torch cannot be imported."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - a torch-free driver box is a supported box
        return None
    pythonpath = child_env(args, src, CONFIGS[0], "n/a")["PYTHONPATH"]
    return common.env_block(tree_root, "band-sweep-driver",
                            extra={"note": args.note, "pythonpath": pythonpath})


# --- driver ----------------------------------------------------------------
def run_driver(args: argparse.Namespace) -> int:
    """One child process per (cell, config), strictly sequential, one record each."""
    cells = resolve_cells(args.cells)
    configs = resolve_configs(args.configs)
    workloads = resolve_workloads(args.workloads)
    check_workers(args.workers)
    check_config_specs(configs)
    tree_root = resolve_tree_root(args.tree_root)
    src = resolve_src(tree_root)
    # Put the measured tree first on sys.path so the knob resolvers and ``common.env_block``
    # describe THAT checkout.  sys.path only: no evogp import, so --dry-run needs no torch.
    install_tree_root(tree_root)
    install_evoxc_root(args.evoxc_root)
    # the frozen-directory refusal covers the dry run too: it prints the exact record paths,
    # and a path this script would refuse to write must not look planned
    assert_fresh_out(args.out_dir)
    if args.dry_run:
        return dry_run(args, cells, configs, workloads, src)
    # A pinned band count this device cannot load would be refused by the engine itself, and only
    # AFTER the child paid its population build: refuse the whole plan before the first child, so
    # a doomed config never holds the device and never leaves a half-written evidence directory.
    problems = check_cell_legality(cells, configs)
    if problems:
        raise SystemExit(
            "refusing this plan -- " + str(len(problems)) + " (cell, config) pair(s) pin a band "
            "count the device cannot load (an explicit EVOGP_BATCH_BANDS is validated hard by the "
            "engine and never clamped):\n  " + "\n  ".join(problems) + "\n"
            "Restrict --cells per config family (e.g. the k15* pins on the pop-256-or-larger "
            "cells, k7* on every cell) -- --dry-run lists the same pairs as WARNINGs.")
    os.makedirs(args.out_dir, exist_ok=True)
    base_env = child_env(args, src, configs[0], "n/a")
    dis_source = disable_locations_provenance(base_env)["source"]
    index_path = os.path.join(args.out_dir, "sweep.json")
    notes = [
        "one child process per (cell, config): the merged-graph cache, the evoxc compile cache "
        "and the executable memo are per process, so a config loop inside one process would "
        "measure warm caches only",
        "one knob SPEC per config: the band knob plus the pool width and the banded+chunked "
        "composition, so a record states the merge shape it ran",
        f"band-partition configs ({', '.join(BAND_PARTITION_CONFIGS)}) are single-body, so a "
        "difference between two of them is the band partition alone; the composition configs "
        f"({', '.join(COMPOSITION_CONFIGS)}) deliberately run the pool",
    ]
    if args.cold_only:
        notes.append(
            "cold-only (--cold-only): one synchronized call per workload, no warmup and no "
            "timed reps -- every warm field is null (NOT zero) and samples_ms is []"
        )
    index: Dict[str, Any] = {
        "schema": SCHEMA_DRIVER,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "driver_env": _driver_env_block(args, tree_root, src),
        "protocol": {
            "one_process_per_config": True, "one_config_per_process": True, "sequential": True,
            "cold_only": bool(args.cold_only),
            "cells": [cell["id"] for cell in cells],
            "configs": list(configs),
            "config_knobs": {config: dict(CONFIG_KNOBS[config]) for config in configs},
            "band_partition_configs": list(BAND_PARTITION_CONFIGS),
            "composition_configs": list(COMPOSITION_CONFIGS),
            # What each config pinning an explicit band COUNT needs from the population, next to
            # the cap it was vetted against: the binding constants are the device's, not the cell's.
            "band_cap_configs": {
                config: {
                    "pinned_bands": config_pinned_bands(config),
                    "pop_floor": config_pop_floor(config),
                }
                for config in configs if config_pinned_bands(config) is not None
            },
            "band_cap_limits": band_cap_limits(),
            "workloads": list(workloads),
            "config_env": {config: config_env(config) for config in configs},
            "warmup": 0 if args.cold_only else args.warmup,
            "reps": 0 if args.cold_only else args.reps,
            "seed": args.seed, "workers": args.workers,
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
        "notes": notes,
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
        if args.cold_only:
            cmd.append("--cold-only")
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
        print(f"[{position}/{len(plan)}] {status:<8} {cell_id:<22} {config:<9} {wall_s:7.1f}s "
              f"(gpu_idle={state.get('ok')})" + (f" -- {detail}" if detail else ""), flush=True)
    index["elapsed_s"] = round(time.time() - started_all, 1)
    index["failures"] = failures
    index["skipped"] = skipped
    common.write_json(index_path, index)
    print(f"[band-sweep] {len(plan)} configs, {len(plan) - failures - skipped} ok, {failures} "
          f"failed/timeout, {skipped} skipped, {index['elapsed_s']}s -> {index_path}", flush=True)
    return 0 if failures == 0 else 1


# --- child phase: ONE cell, ONE config, ONE process -------------------------
def run_child(args: argparse.Namespace) -> int:
    """Resolve this child's cell/config/tree+env, then hand the measurement to ``band_measure``.

    The config env is applied to ``os.environ`` BEFORE ``evogp``/``torch`` are imported (inside
    ``band_measure.measure``), so the engine resolves exactly the knobs the record states.
    """
    if not args.child:
        raise SystemExit("child mode needs --child <cell id>")
    config = (args.config or "").strip()
    if config not in CONFIG_KNOBS:
        raise SystemExit(f"child mode needs --config (one of {', '.join(CONFIGS)}), got {config!r}")
    if not args.out:
        raise SystemExit("child mode needs --out <record path>")
    check_workers(args.workers)
    check_config_specs([config])
    workloads = resolve_workloads(args.workloads)
    cell = common.parse_cell(args.child)
    # A direct `--child` invocation gets the same legality gate as the driver: an explicit band
    # count the device cannot load is refused by the engine, and this refuses it before the
    # population build instead of after it.
    illegal = cell_legality(cell, config)
    if illegal is not None:
        raise SystemExit(f"refusing this child: {illegal}")
    tree_root = resolve_tree_root(args.tree_root)  # --tree-root, as worker.py resolves it
    src = install_tree_root(tree_root)  # sys.path only: no evogp import yet
    mapping = config_env(config)
    apply_env(mapping)  # BEFORE the torch/evogp import, by construction
    plan_cells = [spec["id"] for spec in resolve_cells(args.cells)] if args.cells else [cell["id"]]
    plan_configs = resolve_configs(args.configs) if args.configs else [config]
    return band_measure.run_child(
        args, cell, config, src, tree_root, mapping, workloads, plan_cells, plan_configs
    )


# --- entry point ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", default=DEFAULT_CELLS, help="comma-separated cell ids (driver)")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS,
                        help=f"comma-separated config tokens: {', '.join(CONFIGS)}")
    parser.add_argument("--workloads", default=DEFAULT_WORKLOADS,
                        help=f"subset of {', '.join(WORKLOAD_ORDER)} (measured in that order)")
    parser.add_argument("--tree-root", default=DEFAULT_TREE_ROOT,
                        help="tree root whose src/ holds evogp (any checkout)")
    parser.add_argument("--evoxc-root", default=DEFAULT_EVOXC_ROOT,
                        help="evoxc checkout on the child's PYTHONPATH (pinned, not /root/evoxc)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="evidence directory (never results/final|baseline|batch|batch_opt_round1)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="warm calls per workload")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="timed samples per workload")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="torch.manual_seed of the population AND the data")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="must be 1: the pool width of a record is its config's own knob")
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
    parser.add_argument("--cold-only", action="store_true",
                        help="measure ONLY the cold first call per workload (no warmup, no reps)")
    parser.add_argument("--note", default=None, help="free-form note stored in every record")
    parser.add_argument("--child", default=None, metavar="CELL",
                        help="internal: measure ONE cell (with --config) and write --out")
    parser.add_argument("--config", default=None, help="internal: the ONE config token")
    parser.add_argument("--out", default=None, help="internal: the child's JSON record path")
    parser.add_argument("--guard", default=None, help="internal: the driver's guard verdict (JSON)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.reps < 1 and not args.cold_only:
        raise SystemExit("--reps must be >= 1 (or pass --cold-only: reps are skipped there)")
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
