"""Shared plumbing for the ``test/benchmark/evoxc`` campaign.

This package times the CURRENT (evoxc-based) EvoGP engine against **the old
plain-PyTorch way of evaluating the same forests** -- no evoxc, no jit -- on
identical populations, in the same session, with a warm/steady-state protocol.

What lives here and nowhere else:

* the cell grid (population shape x tree length x input length x data rows) and
  the deterministic population builder every arm must share;
* the arm registry (torch arms + the evoxc engine routes) and the environment
  knobs each arm sets -- including the batch route's own knob table
  (``BATCH_KNOBS``: band size/bands/switch ops/alignment/cache size/min pop, the
  compiled-graph pool width ``EVOGP_BATCH_COMPILE_WORKERS``, the banded+chunked
  composition opt-in ``EVOGP_BATCH_CHUNK_BANDS`` and the merged compile's
  ``EVOGP_BATCH_DISPATCH_FORMATION``), which is also the list
  ``batch_economics.py --knob NAME=VALUE`` validates against;
* the disk tier's env var (``ARTIFACT_ENV_VARS``: ``EVOGP_EVOXC_ARTIFACT_DIR``, ONE
  name for the dense AND the batch route) plus this harness's own hook for where it
  points (``ARTIFACT_DIR_ENV`` = ``EVOGP_BENCH_ARTIFACT_DIR``, default
  ``DEFAULT_ARTIFACT_DIR``): ``arm_knob_env``/``apply_arm_env`` resolve the
  ``{artifact_dir}`` placeholder from it, so several processes can share one cache;
* the timing helper (sync-bracketed, warmup + reps, median/min/max/samples);
* the value digest used to prove cross-process numeric agreement;
* the environment block written into every record.

The band/merge report of one merged plan lives one module over, in ``band_report.py``
(torch-free, evogp-free: the ONE owner of ``BAND_REPORT_KEYS`` and of the block), whose
public names this module re-exports: ``band_report`` states the merge shape that
``describe_plan`` exposes, the two cost-model estimates only the plan's own report
carries, the band-policy verdict (``band_requested`` / ``band_requested_bands`` /
``band_clamped`` / ``band_note`` / ``band_policy``) and a chunked plan's per-body
``chunk_bands`` -- read null-never-zero, so a band sweep can state what it merged and
what the cost model chose.

The batch route's per-stage cost split lives one module over, in ``stages.py``
(torch-free, evogp-free): it owns the stage vocabulary and the null-never-zero rule
an absent ``batch_stage_stats()`` accessor is reported under.

Nothing here imports ``evogp`` at module level: the campaign re-points
``PYTHONPATH`` at a tree root (``--tree-root``), so the import has to happen
inside the worker process after that root is in place.  ``import_evogp()`` is the
one accessor.

Run the campaign with ``python test/benchmark/evoxc/campaign.py --help``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The band/merge report of ONE merged plan lives in its own torch-free module (the ONE
# owner of ``BAND_REPORT_KEYS`` and of the null-never-zero contract); it is re-exported
# here so every caller keeps reading ``common.band_report(...)``.
from band_report import (  # noqa: E402  (sibling harness module, same directory)
    BAND_ABSENT_NOTE,
    BAND_COST_NOTE,
    BAND_REPORT_KEYS,
    band_report,
)

SCHEMA = "evogp-bench-evoxc-v1"

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
#: The evoxc engine route of each arm, as ``(EVOGP_EVAL_MODE, EVOGP_EVOXC_BACKEND)``.
#: ``None`` means "leave the environment alone" (the library default).
EVOXC_ARMS: Dict[str, Dict[str, Optional[str]]] = {
    # the shipped default: EVOGP_EVAL_MODE=auto, EVOGP_EVOXC_BACKEND=numpy
    "evoxc-auto": {"eval_mode": None, "backend": None, "family": "evoxc"},
    # auto routing with the GPU backend opted in (the interesting auto case)
    "evoxc-auto-iree": {"eval_mode": None, "backend": "iree", "family": "evoxc"},
    # one compiled evoxc graph per tree structure
    "evoxc-specialized-numpy": {
        "eval_mode": "specialized",
        "backend": "numpy",
        "family": "evoxc",
    },
    "evoxc-specialized-iree": {
        "eval_mode": "specialized",
        "backend": "iree",
        "family": "evoxc",
    },
    # one dense all-ops graph per population shape
    "evoxc-dense-iree": {"eval_mode": "dense", "backend": "iree", "family": "evoxc"},
    # one merged graph per population (structure multiset), via evoxc.batch_functions
    "evoxc-batch-iree": {"eval_mode": "batch", "backend": "iree", "family": "evoxc"},
    # the same route with flat (unbanded) merging: O(N^2) in the population
    "evoxc-batch-iree-flat": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_BATCH_BANDS": "flat"},
    },
    # dense with its trace cap pinned to the CELL's tree length: the cap is a
    # ``{tree_len}`` template resolved by ``arms.apply_arm_env(arm, cell)``, so on a
    # main-grid cell (L <= 64) this is exactly ``evoxc-dense-iree``, and on a
    # long-tree cell it is the same route with the cap raised so dense actually
    # serves the population instead of declining to ``specialized``.
    "evoxc-dense-iree-raised": {
        "eval_mode": "dense",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_DENSE_MAX_TREE_LEN": "{tree_len}"},
    },
    # the batch route with the merged compile's aggressive dispatch formation pinned
    # (``EVOGP_BATCH_DISPATCH_FORMATION=aggressive``): on an iree GPU target the
    # route's ``auto`` default already resolves there, so this arm exists to price the
    # spelling where ``auto`` does NOT take it (a non-GPU compile target) and to state
    # the effective value a record's plan report carries when it is pinned rather
    # than derived.
    "evoxc-batch-iree-aggressive": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_BATCH_DISPATCH_FORMATION": "aggressive"},
    },
    # the batch route on the persistent on-disk executable tier (the SAME
    # ``EVOGP_EVOXC_ARTIFACT_DIR`` the dense route uses).  ``{artifact_dir}`` is
    # resolved by ``arms.apply_arm_env`` from ``ARTIFACT_DIR_ENV`` -- so an operator
    # points every process at ONE directory and a later process reloads instead of
    # recompiling.  This is the only arm whose cold column can change by reloading;
    # every other arm clears the variable (``ROUTE_ENV_VARS``).
    "evoxc-batch-iree-artifact": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_EVOXC_ARTIFACT_DIR": "{artifact_dir}"},
    },
    # the PRE-CHANGE control for the merged compile: the batch route with the
    # dispatch-formation request pinned to ``default``, i.e. NO
    # ``--iree-dispatch-creation-enable-aggressive-fusion`` -- the spelling the route
    # derived before ``EVOGP_BATCH_DISPATCH_FORMATION`` grew its ``auto`` default.
    # ``evoxc-batch-iree`` stays the shipped default: it sets no value, so it resolves
    # ``auto`` (``aggressive`` on a rocm/cuda compile target), and this arm is the arm
    # to read next to it when a record has to say what that default changed.
    "evoxc-batch-iree-default": {
        "eval_mode": "batch",
        "backend": "iree",
        "family": "evoxc",
        "knobs": {"EVOGP_BATCH_DISPATCH_FORMATION": "default"},
    },
}
TORCH_ARMS: Dict[str, Dict[str, Any]] = {
    "torch-per-tree": {"impl": "per_tree", "family": "torch"},
    "torch-batched": {"impl": "batched", "family": "torch"},
}
ALL_ARMS: List[str] = list(TORCH_ARMS) + list(EVOXC_ARMS)

#: The knob table of the batch route (``EVOGP_EVAL_MODE=batch``), the ONE place
#: the harness names them: ``(spec key, env var, documented default, candidate
#: resolver attribute names on ``evogp.tree.batch_engine``)``.  ``batch_policy()``
#: is the only reader, so an arm's ``knobs`` mapping, an exported variable and the
#: engine's own default are never spelled out twice.
#:
#: ``EVOGP_BATCH_BANDS`` accepts an explicit band count ``K`` or ``none``/``flat``
#: (one unbanded merge, O(N^2) in the population -- the ``-flat`` arm times it);
#: ``EVOGP_BATCH_ALIGNMENT`` defaults to ``variant`` (band variants are aligned
#: before the merge instead of being padded per call);
#: ``EVOGP_BATCH_COMPILE_WORKERS`` is the compiled-graph pool width (``batch_knobs.
#: resolve_compile_workers``), i.e. how many chunk compiles the merged graph may run
#: at once -- it moves the COLD cost of the route, which is why the harness sweeps
#: it with ``batch_economics.py --knob EVOGP_BATCH_COMPILE_WORKERS=N`` instead of
#: registering one arm per value.
#: ``EVOGP_BATCH_CHUNK_BANDS`` is the banded+chunked composition opt-in
#: (``batch_knobs.resolve_chunk_bands``), default OFF: with it off a band request
#: (``EVOGP_BATCH_BANDS``) forces the single-body path, so a chunked plan's bodies
#: are unbanded.
#: ``EVOGP_BATCH_DISPATCH_FORMATION`` is the merged compile's evoxc compile option
#: (``batch_knobs.resolve_dispatch_formation``): ``auto`` (docs default) enables
#: ``aggressive`` only for an iree backend whose compile target is a GPU
#: (``rocm``/``cuda``) and whose installed evoxc declares the option, ``default`` /
#: ``off`` derives no flag, and an explicit ``aggressive`` is honoured even where the
#: target is not a GPU (it may be slower there).  The REQUEST is what this table
#: resolves; the EFFECTIVE spelling is on the plan (``BatchPlan.dispatch_formation``,
#: surfaced by ``describe_plan`` and therefore by ``band_report``) and the arm
#: ``evoxc-batch-iree-aggressive`` pins the request.
BATCH_KNOBS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    # (spec key, env var, documented default, candidate resolver attribute names)
    ("band_size", "EVOGP_BATCH_BAND_SIZE", "32", ("resolve_band_size",)),
    ("bands", "EVOGP_BATCH_BANDS", "default", ("resolve_bands", "resolve_band_count")),
    ("switch_ops", "EVOGP_BATCH_SWITCH_OPS", "off", ("resolve_switch_ops", "uses_switch_ops")),
    ("alignment", "EVOGP_BATCH_ALIGNMENT", "variant", ("resolve_alignment",)),
    ("cache_size", "EVOGP_BATCH_CACHE_SIZE", "unset", ("resolve_cache_size",)),
    ("min_pop", "EVOGP_BATCH_MIN_POP", "64", ("resolve_min_pop",)),
    ("compile_workers", "EVOGP_BATCH_COMPILE_WORKERS", "default", ("resolve_compile_workers",)),
    ("chunk_bands", "EVOGP_BATCH_CHUNK_BANDS", "off", ("resolve_chunk_bands",)),
    # ``auto`` / ``aggressive`` / ``default`` (alias ``off``); the resolver answers the
    # REQUEST, and the plan's own ``dispatch_formation`` the effective spelling
    ("dispatch_formation", "EVOGP_BATCH_DISPATCH_FORMATION", "auto",
     ("resolve_dispatch_formation",)),
)
#: Every env var the table above owns -- the arms clear all of them per process.
BATCH_ENV_VARS: Tuple[str, ...] = tuple(entry[1] for entry in BATCH_KNOBS)
#: The DENSE route's knobs (single-sourced here so the harness names them once):
#: ``EVOGP_DENSE_MAX_TREE_LEN`` (``dense_engine.MAX_TREE_LEN_ENV``) is the longest
#: trace ``L = min(cap, max_tree_len)`` the dense evaluator will build, default 64 --
#: above it the route declines with ``DenseUnsupported`` and the public API falls back
#: to specialized; ``EVOGP_DENSE_AUTO_BUDGET_MB`` (``dense_engine.AUTO_BUDGET_MB_ENV``,
#: default 8192) is the memory envelope of ``auto``'s rows rule
#: (``pop_size * tree_len * rows * 4`` bytes), and
#: ``EVOGP_DENSE_AUTO_MAX_ROWS`` (``dense_engine.AUTO_MAX_ROWS_ENV``, default 0 =
#: unpinned) is now only an optional TIGHTENING pin on that envelope's allowance.
DENSE_ENV_VARS: Tuple[str, ...] = (
    "EVOGP_DENSE_MAX_TREE_LEN",
    "EVOGP_DENSE_AUTO_BUDGET_MB",
    "EVOGP_DENSE_AUTO_MAX_ROWS",
)
#: The opt-in persistent on-disk executable tier: ONE env var, and it is the DENSE
#: route's AND the batch route's (the batch route keeps its bodies in the ``batch/``
#: subdirectory of that directory).  Named here so it is cleared like every other route
#: knob -- an inherited directory would silently make an arm a disk-cached run -- and
#: so only ``evoxc-batch-iree-artifact`` turns it on.
ARTIFACT_ENV_VARS: Tuple[str, ...] = ("EVOGP_EVOXC_ARTIFACT_DIR",)
#: Every env var the arms clear before applying their own knob spec, so a route knob
#: can never leak between arms (``BATCH_ENV_VARS`` + ``DENSE_ENV_VARS`` + the disk
#: tier's ``ARTIFACT_ENV_VARS``).
ROUTE_ENV_VARS: Tuple[str, ...] = BATCH_ENV_VARS + DENSE_ENV_VARS + ARTIFACT_ENV_VARS
#: The counter names the batch route's compile misses can live under, in preference
#: order.  ``evogp.tree.batch_engine.batch_cache_stats()`` names its two front tables
#: explicitly: ``batch_exec_*`` describes ``BATCH_EXEC_MEMO`` -- the real compile
#: cache, since a hit never reaches ``evoxc.lower`` / ``compile`` / ``load`` -- and
#: ``batch_plan_*`` describes ``BATCH_PLAN_MEMO`` (host decode + traces + merge).
#: A bare ``misses`` is this harness's own older spelling: a fallback only.
BATCH_EXEC_MISS_KEYS: Tuple[str, ...] = ("batch_exec_misses", "misses")
#: The plan table's counter: a plan miss re-traces and re-merges, it never recompiles.
BATCH_PLAN_MISS_KEYS: Tuple[str, ...] = ("batch_plan_misses",)
#: The exec-table counter name -- the one a record's ``stats_keys`` must carry.
BATCH_EXEC_MISS_KEY = BATCH_EXEC_MISS_KEYS[0]
#: Location of the batch engine relative to the ``src`` directory (import-free check).
BATCH_MODULE_REL = os.path.join("evogp", "tree", "batch_engine.py")

#: This harness's own hook for the disk tier's directory: the operator exports it once
#: (or ``campaign.py --artifact-dir`` writes it into every child's environment) and
#: EVERY process of a round then points ``EVOGP_EVOXC_ARTIFACT_DIR`` at that one path,
#: so a later process reloads instead of recompiling.  It is NOT an engine variable:
#: the engine only ever sees ``EVOGP_EVOXC_ARTIFACT_DIR``.
ARTIFACT_DIR_ENV = "EVOGP_BENCH_ARTIFACT_DIR"
#: Where the artifact arm points the disk tier when nothing is exported: a fixed
#: per-box scratch directory (never the temp-file namespace of one process, so two
#: runs can share it).  ``campaign.py`` defaults it to ``<out>/artifacts`` instead --
#: a per-RUN directory -- and exports ``ARTIFACT_DIR_ENV``, so the default only
#: applies to a bare ``worker.py`` invocation.
DEFAULT_ARTIFACT_DIR = os.path.join(tempfile.gettempdir(), "evogp_bench_artifacts")
#: The placeholder an arm's ``knobs`` value uses to mean "the resolved disk directory"
#: (``arms.apply_arm_env`` substitutes it; a literal path works there too).
ARTIFACT_DIR_PLACEHOLDER = "{artifact_dir}"


def artifact_dir() -> str:
    """The absolute directory the disk tier points at (``ARTIFACT_DIR_ENV``, else default).

    Read per call, so a campaign driver that exports the hook before spawning a child
    and a hand-run worker agree without any further wiring.  An empty export reads as
    unset (the engine treats ``EVOGP_EVOXC_ARTIFACT_DIR=""`` as off, so an empty
    directory must never be handed to an arm).
    """
    value = os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    return os.path.abspath(value) if value else DEFAULT_ARTIFACT_DIR


def arm_knob_env(arm: str) -> Dict[str, str]:
    """The env vars an arm's ``knobs`` spec sets, with ``{artifact_dir}`` resolved.

    A read-only view for a dry run and for a test, so an arm's environment can be stated
    without a device: the disk tier's placeholder is resolved from :func:`artifact_dir`
    (it is the harness's hook, not a property of a cell), while a per-CELL template such
    as ``{tree_len}`` is left verbatim -- ``campaign.py`` already prints that one per
    scheduled cell, and resolving it here would need a cell this helper does not take.
    An arm with no knobs (and an unknown arm) reads ``{}``.
    """
    knobs = dict((EVOXC_ARMS.get(arm) or {}).get("knobs") or {})
    return {
        env_name: artifact_dir() if value == ARTIFACT_DIR_PLACEHOLDER else str(value)
        for env_name, value in knobs.items()
    }


def is_batch_arm(arm: str) -> bool:
    """Is this arm's route the batch engine (one merged graph per population)?

    One definition for ``worker.py`` / ``check.py`` / the economics script: an arm
    is a batch arm exactly when its registry entry asks for ``EVOGP_EVAL_MODE=batch``,
    so a future batch variant (another band policy) is picked up with no extra rule.
    """
    return (EVOXC_ARMS.get(arm) or {}).get("eval_mode") == "batch"

#: Workloads. ``sr_fitness`` is the training hot path, ``batch_forward`` is the
#: public ``Forest.batch_forward`` value path (``(pop, N, out)``), ``forward`` is
#: the per-row value path (``(pop, out)``: tree ``i`` sees row ``i``), ``gp_step``
#: is one full GP generation (evaluate + select + crossover + mutate).
WORKLOADS = ("sr_fitness", "batch_forward", "forward", "gp_step")


# ---------------------------------------------------------------------------
# cell grid
# ---------------------------------------------------------------------------
POPS = (16, 64, 256, 1000)
TREE_LENS = (32, 64)
#: the long-tree column's tree lengths -- both above the dense route's default trace
#: cap (``dense_engine.DEFAULT_MAX_TREE_LEN`` = 64, ``EVOGP_DENSE_MAX_TREE_LEN``)
LONG_TREE_LENS = (128, 256)
INPUT_LENS = (8, 32)
MAIN_ROWS = 1024
#: the second data-row value ("one N=64 column"): L=64/input_len=32 at every pop
N64_ROWS = 64
#: the large column's population scale (``--cells large``): the three populations
#: above the ``main_cells()`` ceiling of 1000, i.e. the region no record covers.
LARGE_POPS = (2048, 4096, 8192)
#: the large column's data-row scale: the rows axis has never been measured above
#: ``MAIN_ROWS`` = 1024, and it is the axis a per-dispatch overhead amortizes on.
LARGE_ROWS = (4096, 16384)
#: The memory rule of ONE cell: the dense route materializes ``pop * tree_len * rows``
#: float32 values in its largest intermediate (``pop * tree_len * rows * 4`` bytes, see
#: :func:`cell_tensor_bytes`) and the torch arms hold several such tensors at once, so a
#: cell is only schedulable while that product stays inside this per-cell ceiling.
#: ``large_cells()``'s worst cells are ``p1000-L64-i32-N16384`` at 4.19 GB and
#: ``p8192-L64-i32-N1024`` / ``p2048-L256-i32-N1024-d8`` at 2.15 GB, against this box's
#: 191.7 GB HBM -- 8 GiB leaves the whole column 2x clear and keeps a new cell a
#: one-line check (``cell_tensor_bytes(cell(...)) <= CELL_TENSOR_CEILING_BYTES``).
CELL_TENSOR_CEILING_BYTES = 8 * 2**30


def cell(
    pop: int, tree_len: int, input_len: int, rows: int, max_layer_cnt: int = 5
) -> Dict[str, Any]:
    """One cell: the population shape plus the generation depth cap.

    ``tree_len`` is the row stride (the descriptor's ``max_tree_len``) and
    ``max_layer_cnt`` is the depth cap that governs how many nodes the generated
    trees actually carry (a full tree is ``2 ** d - 1`` nodes).  The id keeps the
    historical spelling for the default cap (``max_layer_cnt == 5``) and gains a
    ``-d<k>`` suffix otherwise, so every existing cell id stays byte-identical.
    """
    suffix = "" if int(max_layer_cnt) == 5 else f"-d{int(max_layer_cnt)}"
    return {
        "id": f"p{pop}-L{tree_len}-i{input_len}-N{rows}{suffix}",
        "pop": pop,
        "tree_len": tree_len,
        "input_len": input_len,
        "rows": rows,
        "max_layer_cnt": int(max_layer_cnt),
    }


def main_cells() -> List[Dict[str, Any]]:
    """The 4 x 2 x 2 grid at ``N = 1024`` (16 cells)."""
    return [
        cell(pop, tree_len, input_len, MAIN_ROWS)
        for pop in POPS
        for tree_len in TREE_LENS
        for input_len in INPUT_LENS
    ]


def n64_cells() -> List[Dict[str, Any]]:
    """The ``N = 64`` column: every pop at tree_len 64 / input_len 32."""
    return [cell(pop, 64, 32, N64_ROWS) for pop in POPS]


def long_tree_cells() -> List[Dict[str, Any]]:
    """The LONG-TREE column at ``N = 1024`` (``--cells long``): six cells.

    The column has TWO axes and the cells separate them, because in this library
    they are different quantities:

    * ``tree_len`` is the descriptor's ``max_tree_len`` -- the row stride of the
      tree tensors and the length the dense route traces.  The dense route
      declines every population whose longest tree exceeds
      ``EVOGP_DENSE_MAX_TREE_LEN`` (default 64), so above that cap the public API
      silently falls back to the per-tree ``specialized`` route (one compile per
      distinct structure).  A cell therefore has to be priced with the cap both at
      its default (``evoxc-dense-iree``, measuring the fallback) and raised to the
      cell's own tree length (``evoxc-dense-iree-raised``).
    * ``max_layer_cnt`` is what actually decides how many nodes a generated tree
      carries (a full tree is ``2 ** d - 1`` nodes), so it -- not ``tree_len`` --
      governs the evaluation work.  Measured on this box with the
      :func:`build_workload` recipe (seed 0, pop 64), the node count is identical
      at L=32, L=64 and L=128 for one depth cap and grows with the cap:
      ``d=5`` mean 15.8 / max 29 nodes, ``d=6`` mean 27.1 / max 57, ``d=7``
      mean 35.5 / max 101, ``d=8`` mean 69.9 / max 187.

    Hence the pair of cells per tree length: the cap-5 cell is the stock
    generation with only the row stride made long, and each ``-d<k>`` sibling
    raises the depth cap (the real node-count axis) at that same stride.

    ``tree_len`` 128/256 and the caps up to 8 are legal descriptor settings
    (``MAX_STACK`` = 1024 is not a constraint at these lengths, and
    ``2 ** d - 1 <= tree_len`` holds for every cell below), and the population
    comes from the same :func:`build_workload` recipe as every other column --
    ``torch.manual_seed(0)`` + ``Forest.random_generate`` -- so only the shape is
    new.
    """
    return [
        cell(256, 128, 32, MAIN_ROWS, 5),
        cell(256, 128, 32, MAIN_ROWS, 6),
        cell(256, 128, 32, MAIN_ROWS, 7),
        cell(1000, 128, 32, MAIN_ROWS, 7),
        cell(256, 256, 32, MAIN_ROWS, 5),
        cell(256, 256, 32, MAIN_ROWS, 8),
    ]


def large_cells() -> List[Dict[str, Any]]:
    """The LARGE-SCALE column (``--cells large``): ten cells, in the order below.

    Every other column stops at pop 1000 and ``N`` = 1024 (``main_cells``) or at
    ``N`` = 64 (``n64_cells``), and the long-tree column moves only the node-count axis
    at pop 256/1000 -- so the grid below adds the four axes those columns cannot
    answer, one group of cells per axis:

    * **population scale** -- ``p2048``/``p4096``/``p8192`` at tree_len 64 /
      input_len 32 / ``N`` = 1024.  This is where the host glue that was 15.3 % of a
      warm call at pop 256 either amortizes against the graph run or does not, and it
      is the region no record covers.
    * **row scale** -- ``N`` 4096 and 16384 at pop 256 and pop 1000 (tree_len 64 /
      input_len 32): the rows axis nobody measured.  A per-dispatch overhead is paid
      once per call and a per-row one is paid ``rows`` times, so raising ``rows`` is
      what makes the first disappear from a ratio -- the dense route's own work grows
      with ``pop * tree_len * rows`` while the torch per-tree arm's grows with the
      node walk over the same rows.
    * **node-count scale at large pop** -- ``p2048-L128-i32-N1024-d7`` and
      ``p2048-L256-i32-N1024-d8``: the long-tree column's depth caps (mean 35.5 /
      69.9 nodes, max 101 / 187) at 2048 trees, i.e. real trees far past the dense
      route's default trace cap of 64.  ``evoxc-dense-iree`` therefore measures the
      cap decline into the per-tree ``specialized`` fallback and
      ``evoxc-dense-iree-raised`` measures dense itself at the cell's tree length.
    * **input width** -- ``p4096-L64-i64-N1024``: ``input_len`` 64 instead of 32 at a
      fixed population, tree length and row count, i.e. the only cell in the harness
      where the input plane per row is twice the ``main_cells`` one, so an evaluation
      cost that tracks ``input_len`` (the leaf reads) shows up here and nowhere else.

    **Memory rule.**  The dense route's largest intermediate is one
    ``pop * tree_len * rows * 4`` byte tensor (:func:`cell_tensor_bytes`), so the
    grid's worst cell is ``p1000-L64-i32-N16384`` at ~4.19 GB, with
    ``p8192-L64-i32-N1024`` and ``p2048-L256-i32-N1024-d8`` at ~2.15 GB; the torch
    arms hold several such tensors at once.  Every cell here is checked against
    ``CELL_TENSOR_CEILING_BYTES`` (8 GiB) -- the formula is the whole rule, so a cell
    that breaks it needs a smaller ``rows`` or ``pop``, not a bigger box.
    ``input_len`` does not enter the product but does enter the ``x`` plane, which is
    what the ``-i64`` cell prices.

    The populations come from the same :func:`build_workload` recipe as every other
    column (``torch.manual_seed(0)`` + ``Forest.random_generate``), so only the shape
    is new; the two ``-d<k>`` cells reuse ``LONG_TREE_LENS`` and
    ``2 ** d - 1 <= tree_len`` holds for both, and the ids keep the historical
    spelling (``-d<k>`` only above the default cap of 5).

    No report of record covers this column yet -- ``FINAL.md`` and ``BATCH.md`` stop
    at pop 1000 / ``N`` = 1024 -- so a run of it writes into a fresh result directory
    (the frozen ones are never rewritten in place).
    """
    cells = [cell(pop, 64, 32, MAIN_ROWS) for pop in LARGE_POPS]
    cells += [cell(pop, 64, 32, rows) for pop in (256, 1000) for rows in LARGE_ROWS]
    cells += [
        cell(2048, LONG_TREE_LENS[0], 32, MAIN_ROWS, 7),
        cell(2048, LONG_TREE_LENS[1], 32, MAIN_ROWS, 8),
    ]
    cells += [cell(4096, 64, 64, MAIN_ROWS)]
    return cells


def cell_tensor_bytes(cell_dict: Dict[str, Any]) -> int:
    """Bytes of one cell's largest intermediate on the dense route: ``pop * tree_len * rows * 4``.

    This is the ONE statement of the memory rule ``CELL_TENSOR_CEILING_BYTES`` enforces,
    so a future cell is checked rather than guessed at.  ``input_len`` deliberately does
    not enter it -- it sizes the input plane (``pop * rows * input_len``), not the tree
    planes the route indexes by the row stride -- and the torch arms hold several of
    these tensors at once, which is why the ceiling is 8 GiB and not the 4.19 GB worst
    cell of :func:`large_cells`.
    """
    return int(cell_dict["pop"]) * int(cell_dict["tree_len"]) * int(cell_dict["rows"]) * 4


def parse_cell(text: str) -> Dict[str, Any]:
    """``p256-L64-i32-N1024`` -> the cell dict (``-d7`` sets the depth cap)."""
    parts = dict(part.split("=", 1) if "=" in part else (part[0], part[1:]) for part in text.split("-"))
    try:
        return cell(
            int(parts["p"]),
            int(parts["L"]),
            int(parts["i"]),
            int(parts["N"]),
            int(parts.get("d", 5)),
        )
    except Exception as exc:  # noqa: BLE001 - a bad id is a usage error
        raise SystemExit(f"bad cell id {text!r}: {exc}")


# ---------------------------------------------------------------------------
# environment / imports
# ---------------------------------------------------------------------------
def import_evogp():
    """Import the library from whatever ``PYTHONPATH`` the caller set up."""
    import evogp.tree as tree  # noqa: F401  (the package under test)
    import evogp.algorithm as algorithm  # noqa: F401
    import evogp.pipeline as pipeline  # noqa: F401
    import evogp.problem as problem  # noqa: F401

    return tree, algorithm, pipeline, problem


def _git_head(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort provenance
        pass
    return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def env_knobs() -> Dict[str, str]:
    """Every ``EVOGP_*`` / ``EVOXC_*`` variable the engine can read, as set."""
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("EVOGP_") or key.startswith("EVOXC_")
    }


def env_block(tree_root: str, arm: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Environment provenance written at the head of every record."""
    import numpy as np
    import torch

    _, _, _, _ = (None, None, None, None)
    block: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree_root": os.path.abspath(tree_root),
        "tree_root_sha": _git_head(os.path.abspath(tree_root)),
        "cwd": os.getcwd(),
        "arm": arm,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "iree_base_compiler": _pkg_version("iree-base-compiler"),
        "iree_base_runtime": _pkg_version("iree-base-runtime"),
        "evoxc_path": None,
        "evoxc_sha": _git_head("/root/evoxc"),
        "evogp_file": None,
        "gpu": None,
        "gpu_total_memory_gb": None,
        "knobs": env_knobs(),
    }
    try:
        import evoxc

        block["evoxc_path"] = getattr(evoxc, "__file__", None)
        block["evoxc_version"] = getattr(evoxc, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        block["evoxc_import_error"] = repr(exc)
    try:
        import evogp

        block["evogp_file"] = getattr(evogp, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        block["evogp_import_error"] = repr(exc)
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            block["gpu"] = props.name
            block["gpu_total_memory_gb"] = round(props.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        block["gpu_error"] = repr(exc)
    if extra:
        block.update(extra)
    return block


# ---------------------------------------------------------------------------
# workload construction (ONE definition, shared by every arm)
# ---------------------------------------------------------------------------
#: The function set every population is generated from.  Restricted to the four
#: correctly-rounded arithmetic operators on purpose: with them the plain
#: float32 torch arms and the reference oracle agree bit-exactly (see
#: ``test/benchmark/evoxc/README.md`` "Numeric validation"), so a mismatch is a
#: real bug and not one-ulp transcendental noise.
def src_root() -> str:
    """The ``src`` directory of the *imported* evogp checkout (the ``--tree-root`` switch)."""
    import evogp

    return os.path.dirname(os.path.dirname(os.path.abspath(evogp.__file__)))


def checkout_root() -> str:
    """The git checkout that contains :func:`src_root` (``/tmp/evogp_preopt`` for a snapshot)."""
    return os.path.dirname(src_root())


DESCRIPTOR_FUNCS = ("+", "-", "*", "/")


def make_descriptor(tree_len: int, input_len: int, output_len: int = 1, using_funcs=None, max_layer_cnt: int = 5):
    """The canonical generation descriptor (mirrors ``src/evogp/sr_test.py``)."""
    from evogp.tree import GenerateDescriptor

    return GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=input_len,
        output_len=output_len,
        const_prob=0.5,
        out_prob=0.5,
        using_funcs=list(using_funcs or DESCRIPTOR_FUNCS),
        layer_leaf_prob=0.2,
        const_range=(-5, 5),
        sample_cnt=8,
        max_layer_cnt=max_layer_cnt,
    )


def build_workload(
    cell_spec: Dict[str, Any],
    seed: int = 0,
    using_funcs=None,
    max_layer_cnt: Optional[int] = None,
) -> Dict[str, Any]:
    """Deterministically build the population and data of one cell.

    ``torch.manual_seed(seed)`` + ``Forest.random_generate`` is the documented
    reproducibility recipe, so every arm in every process sees the same trees.
    The generation depth cap comes from the CELL (``max_layer_cnt``, default 5)
    unless the caller passes one explicitly, so a cell's id and its population
    always agree on the node budget.
    """
    import torch
    from evogp.tree import Forest

    if max_layer_cnt is None:
        max_layer_cnt = int(cell_spec.get("max_layer_cnt", 5))
    torch.manual_seed(seed)
    descriptor = make_descriptor(
        cell_spec["tree_len"], cell_spec["input_len"], using_funcs=using_funcs, max_layer_cnt=max_layer_cnt
    )
    forest = Forest.random_generate(cell_spec["pop"], descriptor)
    rows = cell_spec["rows"]
    x = torch.rand(rows, cell_spec["input_len"], device="cuda")
    labels = torch.rand(rows, 1, device="cuda")
    # per_row value path: one input row per tree
    x_per_row = torch.rand(cell_spec["pop"], cell_spec["input_len"], device="cuda")
    return {
        "forest": forest,
        "descriptor": descriptor,
        "x": x,
        "labels": labels,
        "x_per_row": x_per_row,
        "cell": cell_spec,
        "seed": seed,
        "using_funcs": list(using_funcs or DESCRIPTOR_FUNCS),
    }


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def time_call(fn, warmup: int = 2, reps: int = 5, stats=None) -> Dict[str, Any]:
    """Warm/steady-state timing of ``fn`` (sync-bracketed, per-call samples).

    The first call is timed separately and never counted in the warm statistics:
    that is where a cold compile lands.  ``stats`` (a zero-arg callable, e.g.
    ``evoxc_engine.cache_stats``) is sampled right after it, so the compile cost
    and the compile *misses* are both attributable to that one call.
    """
    import torch

    # one untimed call: this is where a first-call compile lands
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3
    after_first = stats() if stats is not None else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    ordered = sorted(samples)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "first_call_ms": first_ms,
        "cache_after_first_call": after_first,
        "warmup": warmup,
        "reps": reps,
        "samples_ms": samples,
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": sum(samples) / len(samples),
        "_out": out,
    }


# ---------------------------------------------------------------------------
# digests / records
# ---------------------------------------------------------------------------
def digest(value) -> Dict[str, Any]:
    """Shape + value fingerprint of a result tensor (cross-process comparison)."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().to("cpu", torch.float32).numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.ascontiguousarray(array)
    nan = int(np.isnan(array).sum())
    posinf = int(np.isposinf(array).sum())
    neginf = int(np.isneginf(array).sum())
    safe = np.where(np.isfinite(array), array, np.float32(0.0))
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest()[:32],
        "finite": int(np.isfinite(array).sum()),
        "nan": nan,
        "posinf": posinf,
        "neginf": neginf,
        "abs_sum": float(np.abs(safe.astype(np.float64)).sum()),
    }


def population_signature(forest) -> Dict[str, Any]:
    """Structure-multiset fingerprint of a population (the batch route's cache key).

    The merged batch graph is keyed on the *multiset* of tree structures, not on
    the row order, so the per-row hashes are collected into a set and sorted
    before the population hash: a permuted population must report the same
    ``signature``.  ``distinct_structures == 1`` means every row is the same
    structure (elites and duplicates dominate), which is where one merged graph
    wins by the widest margin; ``distinct_structures == pop_size`` means a fresh
    population of all-distinct trees, i.e. the batch route's worst case.

    Read through ``Forest.batch_*`` only: no torch import, and every tensor is
    moved to CPU with ``np.ascontiguousarray`` first (they live on the device and
    are not guaranteed contiguous after generation/mutation).
    """
    import numpy as np

    value = np.ascontiguousarray(forest.batch_node_value.detach().to("cpu").numpy())
    node_type = np.ascontiguousarray(forest.batch_node_type.detach().to("cpu").numpy())
    subtree = np.ascontiguousarray(forest.batch_subtree_size.detach().to("cpu").numpy())

    rows = set()
    for row in range(int(forest.pop_size)):
        length = int(subtree[row, 0])
        payload = (
            np.ascontiguousarray(value[row, :length]).tobytes()
            + np.ascontiguousarray(node_type[row, :length]).tobytes()
            + np.ascontiguousarray(subtree[row, :length]).tobytes()
        )
        rows.add(hashlib.sha256(payload).hexdigest())
    return {
        "distinct_structures": len(rows),
        "signature": hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()[:16],
        "pop_size": int(forest.pop_size),
    }


def public_timing(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the private result tensor before serializing."""
    out = dict(record)
    out.pop("_out", None)
    return out


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# route introspection
# ---------------------------------------------------------------------------
def _requested_eval_mode(eval_mode: Optional[str]) -> Optional[str]:
    """The route name the caller actually asked for, env fallback included.

    Stated once so ``route_decision`` can special-case a route before handing the
    request to the dense policy: an explicit argument wins over
    ``EVOGP_EVAL_MODE``, and neither case nor surrounding whitespace matters.
    ``None`` means "no route pinned" (the library default).
    """
    raw = eval_mode if eval_mode is not None else os.environ.get("EVOGP_EVAL_MODE")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _source_root_candidate() -> Optional[str]:
    """The ``src`` directory to inspect, resolved WITHOUT importing ``evogp``.

    ``src_root()`` is authoritative when ``evogp`` is already imported (that is
    the ``--tree-root`` checkout under measurement); before that import -- e.g. in
    ``check.py``, which is stdlib-only -- the harness's own checkout is found by
    walking up from this file for ``src/evogp``.  ``None`` when neither exists.
    """
    if "evogp" in sys.modules:
        try:
            return src_root()
        except Exception:  # noqa: BLE001 - a broken import must not raise here
            pass
    path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(path, "src", "evogp")):
            return os.path.join(path, "src")
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def batch_engine_present(src: Optional[str] = None) -> bool:
    """Does the tree under test ship ``evogp/tree/batch_engine.py``?  (No import.)

    Mirrors ``cold_start.artifact_feature_present``: a path check keeps the answer
    available before ``evogp`` is imported (a snapshot or a pre-batch tree simply
    has no such file), and this function never imports ``evogp`` and never raises.
    """
    if src is None:
        src = _source_root_candidate()
    if not src:
        return False
    try:
        return os.path.isfile(os.path.join(src, BATCH_MODULE_REL))
    except Exception:  # noqa: BLE001 - a bad path is "absent", never a crash
        return False


def _registered_knob_value(env_name: str, value: Optional[str] = None) -> Optional[str]:
    """The value an arm of the registry would write for ``env_name`` (or None).

    Lets ``batch_policy`` label a variable as arm-supplied instead of merely
    inherited: arms write their ``knobs`` mapping into ``os.environ`` before the
    engine is imported, so the registry is the only record of who set it.

    Two arms may register the SAME variable with different values (the batch route's
    dispatch-formation pins ``aggressive`` and ``default``), so a caller that passes the
    observed ``value`` gets the value an arm really wrote -- ``"arm"`` then means "one
    of the registry's arms wrote exactly this".  Without one, the first registered value
    is returned (the historical reading).
    """
    values = [
        str(spec["knobs"][env_name])
        for spec in EVOXC_ARMS.values()
        if (spec.get("knobs") or {}).get(env_name) is not None
    ]
    if value is not None and str(value) in values:
        return str(value)
    return values[0] if values else None


def _resolve_batch_knob(
    module, spec_key: str, env_name: str, default: str, candidates: Tuple[str, ...], errors: Dict[str, str]
) -> Tuple[str, str]:
    """``(value, source)`` for one batch knob, in the documented precedence order.

    arm-supplied env value > inherited env value > the engine's own resolver >
    the documented default.  A failing resolver is recorded in ``errors`` and the
    next candidate (then the default) is used, so a policy report is never lost to
    one broken accessor.  A resolver that does not exist YET on the tree under
    measurement (``resolve_compile_workers`` before ``batch_knobs`` ships it) is not
    an error at all: it falls through to the documented default with the source
    labelled ``harness-default``, which is exactly how a missing knob is reported.
    """
    env_value = os.environ.get(env_name)
    if env_value is not None:
        arm_value = _registered_knob_value(env_name, env_value)
        return env_value, "arm" if arm_value == env_value else "env"
    if module is not None:
        for attr in candidates:
            resolver = getattr(module, attr, None)
            if callable(resolver):
                try:
                    return str(resolver()), "module"
                except Exception as exc:  # noqa: BLE001 - report, then try/fall back
                    errors[spec_key] = repr(exc)
    return default, "harness-default"


def batch_policy(src: Optional[str] = None) -> Dict[str, Any]:
    """Everything a record must state about the batch route as THIS process sees it.

    Import-light by construction: ``evogp.tree.dense_engine`` is consulted only for
    its declared ``EVAL_MODES`` (``mode_supported`` tells whether today's tree even
    accepts ``EVOGP_EVAL_MODE=batch``), and ``evogp.tree.batch_engine`` only for the
    knob resolvers named in :data:`BATCH_KNOBS`.  Both imports are guarded, so a
    GPU-free / evogp-free caller (``check.py``, ``summarize.py``) still gets a full
    report with ``module_present: False`` instead of an exception.
    """
    present = batch_engine_present(src)
    resolved_src = src if src is not None else _source_root_candidate()
    module_path = os.path.join(resolved_src, BATCH_MODULE_REL) if resolved_src else None
    policy: Dict[str, Any] = {
        "module_present": present,
        "module": module_path,
        "eval_mode": "batch",
        "backend": "iree with numpy fallback",
        "declared_modes": None,
        "mode_supported": False,
    }
    errors: Dict[str, str] = {}

    # Does the installed policy know the route at all?  Its EVAL_MODES is the
    # authority; on today's tree ("dense", "specialized", "auto") it does not.
    try:
        from evogp.tree import dense_engine

        declared = list(getattr(dense_engine, "EVAL_MODES", ()) or ())
        policy["declared_modes"] = declared or None
        policy["mode_supported"] = "batch" in declared
    except Exception as exc:  # noqa: BLE001 - absence is a reportable state
        policy["declared_modes_error"] = repr(exc)

    module = None
    try:
        from evogp.tree import batch_engine as module
    except Exception as exc:  # noqa: BLE001 - the route can be absent
        policy["module_error"] = repr(exc)

    knobs: Dict[str, Optional[str]] = {}
    sources: Dict[str, str] = {}
    values: Dict[str, str] = {}
    defaults = {entry[0]: entry[2] for entry in BATCH_KNOBS}
    for spec_key, env_name, default, candidates in BATCH_KNOBS:
        value, source = _resolve_batch_knob(module, spec_key, env_name, default, candidates, errors)
        values[spec_key] = value
        sources[spec_key] = source
        knobs[env_name] = os.environ.get(env_name)
    if errors:
        policy["resolver_errors"] = errors

    # The two numeric knobs are coerce-only: a record should carry ints, and a
    # bad value falls back to the documented default with the reason recorded.
    for spec_key in ("min_pop", "band_size"):
        try:
            policy[spec_key] = int(values[spec_key])
        except (TypeError, ValueError) as exc:
            policy.setdefault("resolver_errors", {})[spec_key] = repr(exc)
            policy[spec_key] = int(defaults[spec_key])
    for spec_key in ("bands", "switch_ops", "alignment", "cache_size", "compile_workers",
                     "dispatch_formation"):
        policy[spec_key] = str(values[spec_key])
    policy["knobs"] = knobs
    policy["knobs_source"] = sources
    return policy


# ---------------------------------------------------------------------------
# band / merge report (torch-free, getattr-probed: null never zero)
# ---------------------------------------------------------------------------
#: The block builder and its key list live in ``band_report.py`` -- the ONE owner of
#: ``BAND_REPORT_KEYS`` and of the null-never-zero contract -- and are re-exported at the
#: top of this module, so every caller keeps reading ``common.band_report(...)``.
#: The block carries the measured merge shape (``bands`` / ``band_sizes`` /
#: ``band_strategy``), the two static cost-model estimates, and the policy VERDICT
#: (``band_requested`` / ``band_requested_bands`` / ``band_clamped`` / ``band_note`` /
#: ``band_policy``) plus a chunked plan's per-body ``chunk_bands`` -- what the cost model
#: chose next to what was measured.


def pop_min_declines(pop: int, min_pop: int) -> Optional[str]:
    """The batch route's documented decline rule: reason string below the floor.

    Contract agreed with the engine, which owns the real decision: below
    ``EVOGP_BATCH_MIN_POP`` (default 64) a merged graph cannot amortize its build,
    so the route declines to ``specialized``.  This mirrors that decision for the
    route summary; it never decides anything itself.
    """
    if int(pop) < int(min_pop):
        return (
            f"pop {int(pop)} < EVOGP_BATCH_MIN_POP {int(min_pop)}: one merged graph "
            "cannot amortize its build, so the route declines to specialized"
        )
    return None


def batch_route_decision(pop: int, rows: Optional[int] = None) -> Dict[str, Any]:
    """The batch analogue of :func:`route_decision`, in the same shape family.

    ``summarize.route_text`` reads ``eval_mode`` / ``backend`` / ``would_use_dense``
    from it, so those keys are always present (the batch route is never dense and
    its ``kind`` says so explicitly).  ``would_use_batch`` folds the three gates
    together -- module present, mode declared by the tree, population above the
    floor -- and ``rows`` is echoed only because every other route decision takes
    it: the batch route is not rows-gated.
    """
    policy = batch_policy()
    decline = pop_min_declines(pop, policy["min_pop"])
    would_use = bool(policy["module_present"] and policy["mode_supported"] and decline is None)
    if not policy["module_present"]:
        state = "batch_engine.py absent on this tree"
    elif not policy["mode_supported"]:
        state = "the tree's EVAL_MODES does not declare 'batch'"
    elif decline is not None:
        state = "declines to specialized below the population floor"
    else:
        state = "merged graph per population structure multiset"
    return {
        "eval_mode": "batch",
        "backend": "iree",
        "kind": "batch",
        "would_use_dense": False,
        "would_use_batch": would_use,
        "declines_to_specialized": decline is not None,
        "decline_reason": decline,
        "min_pop": policy["min_pop"],
        "band_size": policy["band_size"],
        "bands": policy["bands"],
        "alignment": policy["alignment"],
        "switch_ops": policy["switch_ops"],
        # the REQUESTED spelling only: the effective one is on the plan
        # (``batch_plan_summary.band_report.dispatch_formation``), because ``auto``
        # resolves per compile target and ``batch_policy`` never builds a plan
        "dispatch_formation": policy.get("dispatch_formation"),
        "module_present": policy["module_present"],
        "mode_supported": policy["mode_supported"],
        "rows": rows,
        "note": f"batch/iree: {state}",
    }


def route_decision(
    pop: int,
    rows: int,
    eval_mode: Optional[str],
    backend: Optional[str],
    *,
    tree_len: Optional[int] = None,
) -> Dict[str, Any]:
    """What ``Forest.forward``/``SR_fitness`` would do for this cell, per the policy.

    ``should_use_dense`` is the ONE policy function every wired call site asks,
    so this reports the real decision (not a re-derivation).  ``rows`` is the
    data-row count for the ``per_row=False`` workloads, ``1`` for the per-row
    one (which is never rows-gated).  ``pop``/``rows``/``tree_len`` are the
    population SHAPE and are passed through as ``pop_size=``/``rows=``/``tree_len=``,
    which is what the rows half needs: ``auto``'s rule is a memory envelope
    (``pop_size * tree_len * rows * 4`` bytes against ``EVOGP_DENSE_AUTO_BUDGET_MB``,
    tightened by an optional ``EVOGP_DENSE_AUTO_MAX_ROWS`` pin), so without the
    shape the report could only apply the pin and would under-state a decline --
    exactly the shape-blind reading the engine itself has where the shape is known.
    """
    # The batch route is intercepted BEFORE the dense policy is imported: on the
    # current tree dense_engine.EVAL_MODES == ("dense", "specialized", "auto"), so
    # resolve_eval_mode("batch") raises ValueError and the batch decision must come
    # from its own module (batch_route_decision), never from that call.
    if _requested_eval_mode(eval_mode) == "batch":
        return batch_route_decision(pop, rows)

    from evogp.tree import dense_engine

    shape = {"pop_size": pop, "tree_len": tree_len}
    decision = {
        "eval_mode": dense_engine.resolve_eval_mode(eval_mode),
        "backend": dense_engine.resolve_backend_name(backend, None),
    }
    # The wired call sites pass pop_size only; evaluate_dense/sr_fitness_dense
    # enforce the rows half, so both are reported -- the pop-only view unchanged,
    # the shape-aware one next to it.
    decision["should_use_dense_pop_only"] = bool(
        dense_engine.should_use_dense(pop, eval_mode=eval_mode, backend=backend)
    )
    decision["should_use_dense_with_rows"] = bool(
        dense_engine.should_use_dense(
            pop, eval_mode=eval_mode, backend=backend, rows=rows, tree_len=tree_len
        )
    )
    decision["would_use_dense"] = decision["should_use_dense_pop_only"] and not bool(
        dense_engine.auto_rows_decline(
            rows, eval_mode=decision["eval_mode"], **shape
        )
    )
    return decision
