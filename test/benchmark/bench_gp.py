#!/usr/bin/env python
"""``bench_gp.py`` -- side-agnostic benchmark harness for the legacy-CUDA vs evoxc comparison.

One file, two checkouts: it drives *either* engine through the same public
``evogp`` API, selected by ``PYTHONPATH``:

* NEW = this repository at HEAD (pure-torch/evoxc "true jit" engine,
  ``evogp/tree/evoxc_engine.py``), default ``"numpy"`` backend;
* OLD = the legacy custom-CUDA checkout at ``b613503`` (``evogp.evogp_cuda`` /
  ``src/evogp/cuda/*.cu``; no CPU fallback there).

The methodology, workloads W1-W3 and the C0-C7 config matrix are frozen in
``test/benchmark/SPEC.md``; this script implements them (see ``test/benchmark/new/NOTES.md``
for the run log, per-rep semantics and caveats).

Usage (from the repository root, i.e. the checkout root)::

    PYTHONPATH=src python test/benchmark/bench_gp.py --task sr --pop 200 --gens 100 \
        --tree-len 64 --tag C1 --reps 2 --json-out test/benchmark/new/results/C1.json

    PYTHONPATH=src python test/benchmark/bench_gp.py --task eval64 --eval-reps 25 \
        --tag C5 --json-out test/benchmark/new/results/C5.json

Tasks: ``sr`` (W1: synthetic SR training), ``xor`` (W2: parity-table training),
``eval64`` (W3: repeated ``evaluate`` on the forest rebuilt from a committed
fixture) and ``evalfresh`` (W3': repeated ``evaluate`` on a freshly generated
random population).

Environment contract (set *before* importing evogp, as the SPEC requires):
``EVOXC_DISABLE_LOCATIONS=1`` by default (``=0`` with ``--no-disable-locations``);
``EVOGP_EVOXC_BACKEND`` / ``EVOGP_EVOXC_IREE_DEVICE`` / ``EVOGP_EVOXC_IREE_OPT_LEVEL``
are read by the new engine only and are simply inert on the old one.

R2 additions -- every one of them is opt-in; with none of them passed the CLI,
the code path and the JSON keys are exactly the pre-R2 ones plus additive keys::

    --backend {default,numpy,iree}   ``default`` (default value) sets nothing;
                                     ``numpy``/``iree`` set ``EVOGP_EVOXC_BACKEND``
                                     (and the two IREE settings) in ``apply_env``,
                                     i.e. *before* evogp is imported
    --iree-device I                  ``EVOGP_EVOXC_IREE_DEVICE`` (default 0)
    --iree-opt-level {O0,O1}         ``EVOGP_EVOXC_IREE_OPT_LEVEL`` (default O1)
    --data-rows N                    eval64 only: tile the fixture ``x``/``labels``
                                     up to N rows (N == the fixture row count is a
                                     byte-identical no-op, so prior C5 stays
                                     reproducible); ignored (with a warning) by any
                                     other task
    --mem-guard-mb M                 default 8192: before any heavy step, compare M
                                     with the free device memory reported by
                                     ``torch.cuda.mem_get_info()``
    --max-cold-builds N              default 300: stop when the cumulative evoxc
                                     compile misses exceed N -- enforced only on the
                                     NEW side with ``--backend iree`` (a numpy run
                                     compiles no GPU program per structure and the
                                     legacy engine has no ``evoxc_engine`` module, so
                                     both are exempt)
    --allow-cold-builds              disable that guard
    --stage-probe                    decompose every ``problem.evaluate`` call
                                     (eval tasks) / training generation into the
                                     engine's stages -- ``decode_ms``,
                                     ``build_hit_ms``/``build_misses``,
                                     ``run_ms``/``run_calls``, ``readback_ms``,
                                     ``upload_ms``, ``loss_ms``, ``misc_ms`` plus
                                     the H2D/D2H counters -- and add the per-rep
                                     cold/warm build totals to the JSON under a
                                     new top-level ``stages`` key (``null`` when
                                     the flag is off).  NEW side only: the old
                                     side warns and no-ops.  Implemented by
                                     runtime monkeypatching (``src/`` untouched),
                                     see ``test/benchmark/stage_probe.py``

Both guards stop *cleanly*: the JSON is written with ``skipped: true``,
``skip_reason`` set and ``complete: true``, the completed steps are kept
(``partial: true`` for the cold-build guard) and the process exits 3.

R4 addition -- ``--task eval-pure``, the *pure* forward measurement (measurement
type 2) next to the existing end-to-end eval tasks (measurement type 1)::

    --task eval-pure                 new side only; one "invocation" = one full-population
                                     forward on the fixed ``x`` with ``per_row=False``,
                                     readback / host assembly / key derivation excluded
    --mode {auto,dense,specialized,dynamic}
                                     the route to measure: an explicit value wins, ``auto``
                                     asks the engine's own policy (``EVOGP_EVAL_MODE``,
                                     then ``dense_engine.should_use_dense``), ``dynamic``
                                     is a dense alias.  Both ``mode_requested`` and
                                     ``mode_resolved`` are recorded
    --invocations N                  invocations per timed block (default 50)
    --warmup W                       untimed warmup invocations after the construction (10)
    --e2e-companion K                after the pure reps, K synchronized
                                     ``problem.evaluate(forest)`` calls in the same process
                                     (the end-to-end / pure delta); 0 = off
    --pure-runner {auto,frozen,fallback}
                                     ``frozen`` = the (later) frozen
                                     ``build_{dense,specialized}_runner`` API, ``fallback`` =
                                     this harness' own re-implementation, ``auto`` = frozen
                                     when present else fallback; the JSON records which ran
    --output-checksum                after ALL timing (reps + e2e companion), ONE extra
                                     out-of-band invocation is captured to the host and
                                     fingerprinted (sha256 + shapes/dtypes/NaN/+-inf census)
                                     under ``pure.output_checksum``; no timed number ever
                                     includes it
    --save-output-dir DIR            implies ``--output-checksum`` and additionally writes
                                     the captured blocks to ``<DIR>/<TAG>_output.npz``
                                     (keys ``block_0..block_{k-1}``), recorded as
                                     ``pure.output_checksum.saved``

``eval-pure`` reuses ``--fixture`` (with the ``eval64`` ``--data-rows`` tiling), the
memory / cold-build guards, the FD + peak-memory telemetry and the JSON conventions; it
does not use the stage probe (the probe wraps ``evoxc.run`` and would tax a tight loop) and
it skips cleanly (exit 3) on the legacy side, which has no separable pure forward.  The
runner/timing core lives in ``test/benchmark/pure_forward.py``.

Exit codes: ``0`` = complete, ``3`` = skipped / stopped by a guard, ``130`` =
interrupted (``KeyboardInterrupt``), ``1`` = crash (the traceback is deliberately
left uncaught so the failure stays diagnosable).
The fixture (``--fixture``, default ``test/goldens/forest_random.npz`` relative to
the cwd) is only read by ``eval64``/``evalfresh``.  The legacy ``b613503`` checkout
has no ``test/goldens/``, so the old side must pass ``--fixture <absolute path>``.

Old-compat audit (every evogp symbol/kwarg this harness uses, checked against the
``b613503`` blobs; ``OK`` = present with a call-compatible signature)::

    symbol / call form                              b613503 file:line        result
    Forest(input_len, output_len, batch_node_value, src/evogp/tree/forest.py OK (ctor byte-identical)
           batch_node_type, batch_subtree_size)               :13
    Forest.random_generate(pop_size=, descriptor=)  src/evogp/tree/forest.py OK (staticmethod; body
                                                                      :43    dispatches evogp_cuda)
    GenerateDescriptor(max_tree_len, input_len,     src/evogp/tree/descriptor.py OK (same kwargs;
      output_len, const_prob, out_prob, using_funcs,               :31      const_range/sample_cnt
      layer_leaf_prob, const_range, sample_cnt,                             used only when
      max_layer_cnt, const_samples)                                        const_samples is None)
    GenerateDescriptor.update(**kwargs)            descriptor.py:186         OK (returns a new one)
    GeneticProgramming(initial_forest, crossover,   algorithm/                 OK
      mutation, selection, enable_pareto_front)     genetic_programming.py:21
    DefaultCrossover()                              algorithm/crossover/      OK
                                                    default.py:13
    DefaultMutation(mutation_rate=, descriptor=)    algorithm/mutation/        OK
                                                    default.py:19
    DefaultSelection(survival_rate=, elite_rate=)   algorithm/selection/       OK
                                                    default.py:15
    SymbolicRegression(func=, num_inputs=,          problem/                   OK
      num_data=, lower_bounds=, upper_bounds=)      symbolic_regression.py:11
    SymbolicRegression(datapoints=, labels=)        (same ctor)               OK
    SymbolicRegression.evaluate(forest)             (same file)               OK (default use_MSE=True;
      .problem_dim / .solution_dim                                            old side only adds a
                                                                              CombinedForest assert)
    StandardPipeline(algorithm, problem,            pipeline/standard.py:7    OK (byte-identical
      generation_limit=, is_show_details=)                                    between revisions)
    StandardPipeline.step() / .best_fitness         (same file)               OK (best_fitness is a
                                                                              float -inf, then a 0-dim
                                                                              CPU tensor -> float())
    import evogp / evogp.tree / evogp.algorithm /   src/evogp/__init__.py     OK (old evogp/tree/
      evogp.problem / evogp.pipeline                + package __init__ files   __init__.py imports
                                                                              evogp.evogp_cuda, i.e.
                                                                              the CUDA extension is a
                                                                              hard prerequisite there)
    torch.manual_seed / torch.cuda.manual_seed /    --                        OK (SPEC-wide)
      torch.cuda.synchronize / torch.cuda.get_device_name

Known asymmetric bits (do not fail the old side): ``enable_pareto_front`` is only
passed for ``--task xor`` (as in ``example/basic.py``); ``EVOXC_*`` env vars are inert
on the old side (C7 therefore degenerates to a plain legacy run); the old side reads
its RNG stream from the kernels, so population structures/fitness differ from the new
side -- compare *times*, use fitness only as a convergence sanity check.
"""

import argparse
import importlib.machinery
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# task table (defaults follow SPEC.md; ``--flag`` always wins)
# ---------------------------------------------------------------------------

TRAIN_TASKS = ("sr", "xor")
EVAL_TASKS = ("eval64", "evalfresh")
#: R4: the pure-forward task (measurement type 2 -- the internal graph forward pass with the
#: readback/host assembly/key derivation excluded).  New side only; see ``pure_forward.py``.
PURE_TASKS = ("eval-pure",)
TASKS = TRAIN_TASKS + EVAL_TASKS + PURE_TASKS

#: ``None`` means "not applicable / required from the workload definition".
TASK_DEFAULTS = {
    "sr": {"pop": 200, "gens": 100, "tree_len": 64, "eval_reps": None, "reps": 1},
    "xor": {"pop": 50, "gens": 100, "tree_len": 64, "eval_reps": None, "reps": 1},
    "eval64": {"pop": None, "gens": None, "tree_len": None, "eval_reps": 25, "reps": 1},
    "evalfresh": {"pop": 1000, "gens": None, "tree_len": 64, "eval_reps": 10, "reps": 1},
    "eval-pure": {"pop": None, "gens": None, "tree_len": None, "eval_reps": None, "reps": 5},
}

DEFAULT_FIXTURE = os.path.join("test", "goldens", "forest_random.npz")

#: top-level JSON fields that mirror rep 0
TRAIN_MIRROR = (
    "wall_s",
    "generation_loop_s",
    "cold_first_gen_ms",
    "best_fitness_final",
    "partial",
    "projection",
    "construct_s",
    "complete",
)
EVAL_MIRROR = (
    "wall_s",
    "first_call_ms",
    "median_warm_ms",
    "partial",
    "projection",
    "construct_s",
    "complete",
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unspecified workload parameters fall back to the SPEC.md defaults "
        "for the chosen task; see TASK_DEFAULTS in this file.",
    )
    parser.add_argument("--task", choices=TASKS, required=True, help="workload")
    parser.add_argument("--pop", type=int, default=None, help="population size")
    parser.add_argument("--gens", type=int, default=None, help="generations to run")
    parser.add_argument("--tree-len", type=int, default=None, help="max_tree_len")
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed")
    parser.add_argument("--tag", default=None, help="label for logs/JSON (default: task)")
    parser.add_argument(
        "--json-out", default=None, help="JSON output path (rewritten after every step)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="repeat the timed workload N times (same seed); task default from TASK_DEFAULTS "
        "(1 everywhere, 5 for --task eval-pure)",
    )
    parser.add_argument(
        "--eval-reps", type=int, default=None, help="repeated evaluate() calls (eval tasks)"
    )
    parser.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        help=f"npz fixture for the eval tasks (default: {DEFAULT_FIXTURE}, cwd-relative)",
    )
    parser.add_argument(
        "--no-disable-locations",
        action="store_true",
        help="force EVOXC_DISABLE_LOCATIONS=0 (verbose evoxc call-site capture)",
    )
    parser.add_argument(
        "--backend",
        choices=("default", "numpy", "iree"),
        default="default",
        help="evoxc backend for the NEW engine: 'default' (default) leaves the environment "
        "alone; 'numpy'/'iree' export EVOGP_EVOXC_BACKEND (plus the IREE settings) before "
        "evogp is imported.  Inert on the old side",
    )
    parser.add_argument(
        "--iree-device",
        type=int,
        default=0,
        help="IREE device index -> EVOGP_EVOXC_IREE_DEVICE (only meaningful with --backend iree)",
    )
    parser.add_argument(
        "--iree-opt-level",
        choices=("O0", "O1"),
        default="O1",
        help="IREE optimization level -> EVOGP_EVOXC_IREE_OPT_LEVEL (only with --backend iree)",
    )
    parser.add_argument(
        "--data-rows",
        type=int,
        default=None,
        help="eval64 only: tile the fixture x/labels up to N rows (np.tile + slice, "
        "contiguous copy); N equal to the fixture row count is a no-op.  Ignored by "
        "the other tasks",
    )
    parser.add_argument(
        "--mem-guard-mb",
        type=int,
        default=8192,
        help="stop cleanly (exit 3) before any heavy step when the device free memory "
        "reported by torch.cuda.mem_get_info() is below M MiB (default: 8192)",
    )
    parser.add_argument(
        "--max-cold-builds",
        type=int,
        default=300,
        help="stop cleanly (exit 3) when the cumulative evoxc compile misses "
        "(evoxc_engine.cache_stats()['misses']) exceed N (default: 300); enforced only on "
        "the NEW side with --backend iree",
    )
    parser.add_argument(
        "--allow-cold-builds",
        action="store_true",
        help="disable the --max-cold-builds guard (a new-side iree run is otherwise "
        "capped at --max-cold-builds)",
    )
    parser.add_argument(
        "--stage-probe",
        action="store_true",
        help="decompose every problem.evaluate call / training generation into engine "
        "stages (decode / build hit+miss / run / readback / upload / loss / misc) and "
        "record them under the new top-level 'stages' key (null when off).  NEW side "
        "only: the old side warns and no-ops.  Runtime monkeypatching only "
        "(test/benchmark/stage_probe.py)",
    )
    parser.add_argument(
        "--max-wall-s",
        type=float,
        default=None,        help="soft wall-clock cap per rep, checked after every generation",
    )
    parser.add_argument(
        "--min-gens",
        type=int,
        default=0,
        help="never stop early (--max-wall-s) before this many completed generations",
    )
    # ---- R4: --task eval-pure flags (inert for every other task) --------------
    parser.add_argument(
        "--mode",
        choices=("auto", "dense", "specialized", "dynamic"),
        default="auto",
        help="eval-pure only: the route to measure.  'dense'/'specialized' are explicit, "
        "'auto' asks the engine's own policy (EVOGP_EVAL_MODE, then "
        "dense_engine.should_use_dense(pop, backend)), 'dynamic' is an alias of 'dense'.  "
        "Both mode_requested and mode_resolved are recorded",
    )
    parser.add_argument(
        "--invocations",
        type=int,
        default=50,
        help="eval-pure only: full-population forwards per timed block (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="eval-pure only: untimed warmup invocations right after the construction "
        "(default: 10)",
    )
    parser.add_argument(
        "--e2e-companion",
        type=int,
        default=0,
        help="eval-pure only: after the pure reps, run K synchronized problem.evaluate(forest) "
        "calls in the same process (the end-to-end companion; 0 = off).  Set "
        "EVOGP_EVAL_MODE to the pure --mode so both numbers are on the same route",
    )
    parser.add_argument(
        "--pure-runner",
        choices=("auto", "frozen", "fallback"),
        default="auto",
        help="eval-pure only: which constructor to use -- 'frozen' = the frozen "
        "build_{dense,specialized}_runner API (hard error when absent), 'fallback' = this "
        "harness' own re-implementation (forces the fallback even when the frozen API "
        "exists), 'auto' = frozen when present else fallback.  Recorded as "
        "pure.runner_source",
    )
    parser.add_argument(
        "--output-checksum",
        action="store_true",
        help="eval-pure only: after ALL timing (the reps and the e2e companion), capture ONE "
        "extra out-of-band invocation to the host and record its sha256 + block shapes / "
        "dtypes / NaN / +-inf census under pure.output_checksum.  The capture (and its "
        "readback) is never part of any timed number.  Absent -> the JSON keeps exactly today's "
        "schema",
    )
    parser.add_argument(
        "--save-output-dir",
        default=None,
        help="eval-pure only: implies --output-checksum and additionally writes the captured "
        "blocks to <DIR>/<TAG>_output.npz (keys block_0..block_{k-1}, TAG = --tag); the file "
        "is recorded under pure.output_checksum.saved (relative to the cwd when it is inside it)",
    )
    parser.add_argument("--notes", default="", help="free-form text stored in the JSON")
    args = parser.parse_args(argv)

    for key, value in TASK_DEFAULTS[args.task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.tag is None:
        args.tag = args.task

    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.min_gens < 0:
        parser.error("--min-gens must be >= 0")
    if args.mem_guard_mb < 0:
        parser.error("--mem-guard-mb must be >= 0")
    if args.max_cold_builds < 0:
        parser.error("--max-cold-builds must be >= 0")
    if args.data_rows is not None and args.data_rows < 1:
        parser.error("--data-rows must be >= 1")
    if args.task in TRAIN_TASKS:
        for key in ("pop", "gens", "tree_len"):
            if getattr(args, key) is None or getattr(args, key) < 1:
                parser.error(f"--{key.replace('_', '-')} is required (and > 0) for --task {args.task}")
    if args.task == "evalfresh":
        for key in ("pop", "tree_len"):
            if getattr(args, key) is None or getattr(args, key) < 1:
                parser.error(f"--{key.replace('_', '-')} is required (and > 0) for --task evalfresh")
    if args.task in EVAL_TASKS and (args.eval_reps is None or args.eval_reps < 1):        parser.error(f"--eval-reps must be >= 1 for --task {args.task}")
    if args.task in PURE_TASKS:
        if args.invocations < 1:
            parser.error("--invocations must be >= 1")
        if args.warmup < 0:
            parser.error("--warmup must be >= 0")
        if args.e2e_companion < 0:
            parser.error("--e2e-companion must be >= 0")
        # ``--stage-probe`` stays accepted for CLI compatibility: install_stage_probe()
        # warns and never installs it for this task (it would tax the tight pure loop).
    return args


def apply_env(args):
    """Fix the engine environment.  MUST run before evogp (or torch) is imported."""
    if args.no_disable_locations:
        os.environ["EVOXC_DISABLE_LOCATIONS"] = "0"
    else:
        os.environ.setdefault("EVOXC_DISABLE_LOCATIONS", "1")
    # Old side: every EVOGP_EVOXC_* var below is inert (there is no ``evoxc_engine``
    # module there).  New side: ``--backend default`` forces nothing, so an
    # externally exported EVOGP_EVOXC_BACKEND keeps winning, while an explicit
    # ``--backend numpy|iree`` writes all three settings (the two IREE ones are
    # only read by the iree backend, and only there do they matter).
    if args.backend != "default":
        os.environ["EVOGP_EVOXC_BACKEND"] = args.backend
        os.environ["EVOGP_EVOXC_IREE_DEVICE"] = str(args.iree_device)
        os.environ["EVOGP_EVOXC_IREE_OPT_LEVEL"] = args.iree_opt_level
    return {k: v for k, v in os.environ.items() if k.startswith(("EVOXC_", "EVOGP_"))}

# ---------------------------------------------------------------------------
# environment / identity snapshots
# ---------------------------------------------------------------------------


def import_modules():
    """Import evogp *after* ``apply_env`` and return the symbols the harness uses."""
    import evogp
    from evogp.algorithm import (
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    return argparse.Namespace(
        evogp=evogp,
        Forest=Forest,
        GenerateDescriptor=GenerateDescriptor,
        GeneticProgramming=GeneticProgramming,
        DefaultCrossover=DefaultCrossover,
        DefaultMutation=DefaultMutation,
        DefaultSelection=DefaultSelection,
        StandardPipeline=StandardPipeline,
        SymbolicRegression=SymbolicRegression,
    )


def evogp_package_dir():
    """Directory of the ``evogp`` package this process resolves to (``None`` when absent).

    Import-free: the already-imported module wins (``import_modules()`` runs first in
    :func:`main`), otherwise ``importlib.machinery.PathFinder`` walks ``sys.path`` for the
    spec.  Unlike ``importlib.util.find_spec("evogp.tree.evoxc_engine")`` -- which *imports*
    ``evogp`` and ``evogp.tree``, and with them the legacy ``evogp.evogp_cuda`` extension --
    nothing here can fail because a checkout is not importable.
    """
    spec = getattr(sys.modules.get("evogp"), "__spec__", None)
    if spec is None:
        spec = importlib.machinery.PathFinder.find_spec("evogp")
    if spec is None:
        return None
    if spec.origin and os.path.isfile(spec.origin):  # regular package: <dir>/evogp/__init__.py
        return os.path.dirname(os.path.abspath(spec.origin))
    # namespace portion (no ``__init__.py``): the search locations are the *parents*
    for container in spec.submodule_search_locations or ():
        nested = os.path.join(os.path.abspath(container), "evogp")
        if os.path.isdir(nested):
            return nested
    return None


def detect_side():
    """``"new"`` when the ``evogp`` checkout in use carries the evoxc engine, ``"old"`` when it is a
    legacy-layout checkout, ``"unknown"`` only when no ``evogp`` checkout can be located.

    The test is a filesystem fact about the *checkout*, not environment-level import success:
    ``"new"`` iff the resolved package directory has ``tree/evoxc_engine.py`` (a legacy checkout
    has ``cuda/`` instead).  Nothing is imported, so a legacy checkout whose ``evogp_cuda``
    extension is not built -- e.g. a bare ``git archive b613503 src`` materialization, where the
    old env-level probe raised and read ``"unknown"`` -- still reads ``"old"``.

    This is the *engine family*, not the checkout identity: every evoxc-era checkout (this tree,
    ``/tmp/base_850db27``, a ``NEW_PYTHONPATH`` shadow run, ...) reads ``"new"`` by design.
    Checkout identity comes from the payload fields instead (see :func:`repo_snapshot`):
    ``evogp_file`` is the resolved package path -- distinct per checkout -- and ``evogp_commit``
    is ``git rev-parse HEAD`` of the cwd.  Only the new side has the module the side-gated
    paths need, so this label drives :func:`cold_build_guard_enforced` and
    :func:`install_stage_probe`.

    Evidence matrix (fresh processes, this file loaded by path, then ``import_modules()`` ->
    ``detect_side()``; ``before -> after`` the fix)::

        worktree/HEAD, PYTHONPATH=src                        new     -> new
        worktree/HEAD + /mnt/local-ssd/bchuang/evoxc entry   new     -> new
        /tmp/evogp_legacy_bench (built ext)                   old     -> old
        b613503 ``git archive`` layout, no built ext          unknown -> old   <- the bug
        /tmp/base_850db27 (evoxc-era base checkout)          new     -> new
        no ``evogp`` anywhere on ``sys.path``                 unknown -> unknown
    """
    try:
        package_dir = evogp_package_dir()
    except Exception:
        return "unknown"
    if package_dir is None:
        return "unknown"
    return "new" if os.path.isfile(os.path.join(package_dir, "tree", "evoxc_engine.py")) else "old"


def _guarded(label, fn, default=None):
    try:
        return fn()
    except Exception as exc:  # never let metadata collection kill a run
        return default if default is not None else f"<{label}: {type(exc).__name__}: {exc}>"


def etl_version():
    """Version of the ``evoxc`` package (the payload key stays ``etl_version``, the name the
    frozen R1-R6 comparators read); guarded so the old side can never crash on it."""
    try:
        from importlib.metadata import version as _md_version

        return _md_version("evoxc")
    except Exception:
        pass
    try:
        import evoxc  # noqa: F401

        return getattr(evoxc, "__version__", "unknown")
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def repo_snapshot(evogp_file):
    info = {
        "cwd": os.getcwd(),
        "evogp_file": evogp_file,
        "evogp_commit": None,
        "evogp_commit_error": None,
    }
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if done.returncode == 0:
            info["evogp_commit"] = done.stdout.strip()
        else:
            info["evogp_commit_error"] = (done.stderr or "").strip() or f"git exit {done.returncode}"
    except Exception as exc:
        info["evogp_commit_error"] = f"{type(exc).__name__}: {exc}"
    return info


def gpu_snapshot():
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    info["device_count"] = _guarded("device_count", torch.cuda.device_count)
    info["name"] = _guarded("name", lambda: torch.cuda.get_device_name(0))
    info["total_memory_bytes"] = _guarded(
        "total_memory", lambda: int(torch.cuda.get_device_properties(0).total_memory)
    )
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    return info


def env_snapshot():
    env = {k: v for k, v in os.environ.items() if k.startswith(("EVOXC_", "EVOGP_"))}
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return env


# ---------------------------------------------------------------------------
# JSON writing (atomic + incremental)
# ---------------------------------------------------------------------------


def write_json_atomic(path, payload):
    """Rewrite the whole JSON atomically, so an external kill still leaves usable data."""
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".bench_gp_", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o644)  # mkstemp creates 0600; keep results shareable
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# workload construction (public evogp API only, identical on both sides)
# ---------------------------------------------------------------------------


def sr_target(x):
    """W1 target function (verbatim from ``src/evogp/sr_test.py``)."""
    val = x[0] ** 4 / (x[0] ** 4 + 1) + x[1] ** 4 / (x[1] ** 4 + 1)
    return val.reshape(-1)


def w1_problem(mods):
    """W1 problem: ints for the bounds (the fixed-width scalar check quirk)."""
    return mods.SymbolicRegression(
        func=sr_target, num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5
    )


def w1_descriptor(mods, tree_len):
    return mods.GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=2,
        output_len=1,
        const_prob=0.5,
        out_prob=0.5,
        using_funcs=["+", "-", "*", "/"],
        layer_leaf_prob=0.2,
        const_range=(-5, 5),
        sample_cnt=8,
        max_layer_cnt=5,
    )


XOR_INPUTS = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]
XOR_OUTPUTS = [[0], [1], [1], [0], [1], [0], [0], [1]]


def w2_problem(mods):
    """W2 problem: the inline 8x3 parity table from ``example/basic.py`` (on cuda)."""
    datapoints = torch.tensor(XOR_INPUTS, dtype=torch.float, device="cuda")
    labels = torch.tensor(XOR_OUTPUTS, dtype=torch.float, device="cuda")
    return mods.SymbolicRegression(datapoints=datapoints, labels=labels)


def w2_descriptor(mods, problem, tree_len):
    return mods.GenerateDescriptor(
        max_tree_len=tree_len,
        input_len=problem.problem_dim,
        output_len=problem.solution_dim,
        using_funcs=["+", "-", "*", "/"],
        max_layer_cnt=5,
        const_samples=[-1, 0, 1],
    )


def build_problem_and_descriptor(mods, args):
    """(problem, descriptor, one-line description) for a training task."""
    if args.task == "sr":
        problem = w1_problem(mods)
        descriptor = w1_descriptor(mods, args.tree_len)
        desc = (
            f"W1 synthetic SR: SymbolicRegression(func=x0**4/(x0**4+1)+x1**4/(x1**4+1), "
            f"num_inputs=2, num_data=1000, lower_bounds=-5, upper_bounds=5); "
            f"GenerateDescriptor(max_tree_len={args.tree_len}, input_len=2, output_len=1, "
            f"const_prob=0.5, out_prob=0.5, using_funcs=['+','-','*','/'], layer_leaf_prob=0.2, "
            f"const_range=(-5,5), sample_cnt=8, max_layer_cnt=5)"
        )
    elif args.task == "xor":
        problem = w2_problem(mods)
        descriptor = w2_descriptor(mods, problem, args.tree_len)
        desc = (
            f"W2 parity: SymbolicRegression(datapoints=8x3 table, labels=8x1); "
            f"GenerateDescriptor(max_tree_len={args.tree_len}, input_len=problem.problem_dim=3, "
            f"output_len=problem.solution_dim=1, using_funcs=['+','-','*','/'], max_layer_cnt=5, "
            f"const_samples=[-1,0,1]); enable_pareto_front=True"
        )
    else:
        raise AssertionError(f"not a training task: {args.task}")
    return problem, descriptor, desc


def build_algorithm(mods, args, descriptor, initial_forest):
    kwargs = {}
    if args.task == "xor":  # mirrors example/basic.py
        kwargs["enable_pareto_front"] = True
    return mods.GeneticProgramming(
        initial_forest=initial_forest,
        crossover=mods.DefaultCrossover(),
        mutation=mods.DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=mods.DefaultSelection(survival_rate=0.3, elite_rate=0.01),
        **kwargs,
    )


def to_cuda(array, dtype):
    """Copy a fixture array into a writable contiguous torch tensor on cuda."""
    return torch.from_numpy(np.array(array, dtype=dtype, copy=True)).to("cuda")


def tile_rows(array, rows):
    """Tile ``array`` along axis 0 up to exactly ``rows`` rows (contiguous copy).

    ``rows`` smaller than the source is *also* a plain slice, so the result never
    exceeds the requested row count.
    """
    array = np.asarray(array)
    src_rows = int(array.shape[0])
    if src_rows < 1:
        raise ValueError("cannot tile an array without rows")
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    repeats = -(-rows // src_rows)  # ceil(rows / src_rows)
    tiled = np.tile(array, (repeats,) + (1,) * (array.ndim - 1))[:rows]
    return np.ascontiguousarray(tiled)


def build_eval_case(mods, args):
    """(problem, forest, eval_kind, description) for an eval-only task.

    ``--task eval-pure`` reuses the ``eval64`` fixture path verbatim -- same fixture, same
    ``--data-rows`` tiling, same ``Forest`` + ``SymbolicRegression`` wiring -- because the
    pure task needs exactly the same ``(forest, x)`` pair; only what is timed afterwards
    differs (``build_eval_case`` itself is byte-identical for ``eval64``).
    """
    data = np.load(args.fixture)
    if args.task == "eval64" or args.task in PURE_TASKS:
        rows = int(np.asarray(data["x"]).shape[0])
        audit_rows = args.data_rows is not None and int(args.data_rows) != rows
        if audit_rows:  # ``N == fixture rows`` must stay a byte-identical no-op
            data_x = tile_rows(data["x"], int(args.data_rows))
            data_labels = tile_rows(data["labels"], int(args.data_rows))
        else:
            data_x, data_labels = data["x"], data["labels"]
        x = to_cuda(data_x, np.float32)
        labels = to_cuda(data_labels, np.float32)
        forest = mods.Forest(
            input_len=int(x.shape[1]),
            output_len=int(labels.shape[1]),
            batch_node_value=to_cuda(data["forest__node_value"], np.float32),
            batch_node_type=to_cuda(data["forest__node_type"], np.int16),
            batch_subtree_size=to_cuda(data["forest__subtree_size"], np.int16),
        )
        problem = mods.SymbolicRegression(datapoints=x, labels=labels)
        desc = (
            f"W3 fixed fixture: Forest(input_len={forest.input_len}, output_len={forest.output_len}, "
            f"pop_size={forest.pop_size}, max_tree_len={forest.max_tree_len}) rebuilt from "
            f"{args.fixture}[forest__node_value/_node_type/_subtree_size] via the class constructor + "
            f"SymbolicRegression(datapoints=x[{tuple(x.shape)}], labels=labels[{tuple(labels.shape)}])"
        )
        if audit_rows:
            desc += (
                f"; --data-rows={int(args.data_rows)}: x and labels tiled from the fixture's "
                f"{rows} rows (np.tile to {int(args.data_rows)} rows, contiguous copy)"
            )
        return problem, forest, "fixture", desc
    # evalfresh: fresh random population, fixture inputs/labels
    x = to_cuda(data["x"], np.float32)
    labels = to_cuda(data["labels"], np.float32)
    problem = mods.SymbolicRegression(datapoints=x, labels=labels)
    descriptor = w1_descriptor(mods, args.tree_len)
    forest = mods.Forest.random_generate(pop_size=args.pop, descriptor=descriptor)
    desc = (
        f"W3' fresh population: Forest.random_generate(pop_size={args.pop}, "
        f"descriptor=W1-style GenerateDescriptor(max_tree_len={args.tree_len})) + "
        f"SymbolicRegression(datapoints=x[{tuple(x.shape)}], labels=labels[{tuple(labels.shape)}]) "
        f"from {args.fixture}"
    )
    return problem, forest, "fresh", desc


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def fmt_ms(value, spec=".1f"):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return f"{value:{spec}}"


def fmt_fit(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def mean_or_none(values):
    return statistics.fmean(values) if values else None


def new_rep_record(rep, kind):
    record = {
        "rep": rep,
        "construct_s": None,
        "wall_s": None,
        "partial": False,
        "projection": None,
        # `complete` = "the harness finished writing this record".  It stays False
        # only when the process died hard (SIGKILL / crash), leaving a truncated
        # snapshot: the JSON itself is still valid thanks to the incremental writes.
        "complete": False,
    }
    if kind == "train":
        record.update(
            per_gen_ms=[],
            per_gen_best_fitness=[],
            generation_loop_s=None,
            cold_first_gen_ms=None,
            best_fitness_final=None,
        )
    else:
        record.update(per_call_ms=[], first_call_ms=None, median_warm_ms=None)
    return record


def train_projection(per_gen_ms, gens_done, gens_requested):
    mean_ms = mean_or_none(per_gen_ms)
    return {
        "per_gen_ms_mean": mean_ms,
        "gens_completed": gens_done,
        "gens_requested": gens_requested,
        "projected_total_s_for_requested_gens": (
            None if mean_ms is None else mean_ms * gens_requested / 1000.0
        ),
    }


def eval_projection(per_call_ms, calls_done, calls_requested):
    mean_ms = mean_or_none(per_call_ms)
    return {
        "per_call_ms_mean": mean_ms,
        "calls_completed": calls_done,
        "calls_requested": calls_requested,
        "projected_total_s_for_requested_calls": (
            None if mean_ms is None else mean_ms * calls_requested / 1000.0
        ),
    }


def mirror_rep0(payload, record, keys):
    if record["rep"] != 0:
        return
    for key in keys:
        payload[key] = record.get(key)


def summary_training(tag, side, record, reps):
    per_gen = record["per_gen_ms"]
    gens = len(per_gen)
    wall = record["wall_s"]
    flag = " (PARTIAL)" if record["partial"] else ""
    return (
        f"[{tag}] side={side} rep{record['rep']}/{reps}: {gens} gens in {fmt_ms(wall)}s, "
        f"per-gen median {fmt_ms(statistics.median(per_gen) if per_gen else None)}ms, "
        f"first-gen {fmt_ms(record['cold_first_gen_ms'])}ms, "
        f"best_fitness_final={fmt_fit(record['best_fitness_final'])}{flag}"
    )


def summary_eval(tag, side, record, reps):
    calls = record["per_call_ms"]
    flag = " (PARTIAL)" if record["partial"] else ""
    return (
        f"[{tag}] side={side} rep{record['rep']}/{reps}: first {fmt_ms(record['first_call_ms'])}ms, "
        f"warm-median {fmt_ms(record['median_warm_ms'])}ms ({len(calls)} calls){flag}"
    )


# ---------------------------------------------------------------------------
# R4 pure-forward reporting helpers (``--task eval-pure``)
# ---------------------------------------------------------------------------


def import_pure_forward():
    """Import ``pure_forward.py`` from this script's directory.

    The module imports only numpy/torch at import time (``evogp``/``evoxc`` are imported
    lazily inside its functions), so this is safe on either side.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:  # the script directory is not always sys.path[0]
        sys.path.insert(0, here)
    import pure_forward

    return pure_forward


def compile_block():
    """Skeleton of the ``compile`` block (cold and separate from every pure number)."""
    return {
        "decode_plan_ms": None,
        "build_ms": None,
        "construct_total_ms": None,
        "cache_hits": None,
        "cache_misses": None,
        "path": None,
    }


def pure_block(args):
    """Skeleton of the ``pure`` block: static keys + empty per-rep timing lists."""
    return {
        "mode_requested": None,
        "mode_resolved": None,
        "runner_source": None,
        "backend": None,
        "per_row": False,
        "pop": None,
        "n_trees": None,
        "n_const_trees": None,
        "n_graph_calls_per_run": None,
        "n_graph_calls_source": None,
        "invocations": int(args.invocations),
        "warmup": int(args.warmup),
        "reps": int(args.reps),
        "per_invocation_ms": [],
        "block_ms": [],
        "median_per_invocation_ms": None,
        "min_per_invocation_ms": None,
        "mean_per_invocation_ms": None,
        "per_graph_call_ms": None,
        "per_tree_us": None,
        "per_population_evaluate_ms": None,
        "conventions": import_pure_forward().CONVENTIONS,
    }


def new_pure_rep_record(rep):
    """Per-rep record of the ``eval-pure`` task (one timed block of invocations)."""
    return {
        "rep": rep,
        "construct_s": None,
        "wall_s": None,
        "partial": False,
        "projection": None,
        "complete": False,
        "mode_resolved": None,
        "runner_source": None,
        "block_ms": None,
        "per_invocation_ms": None,
    }


def pure_projection(per_invocation_ms, reps_done, reps_requested):
    mean_ms = mean_or_none(per_invocation_ms)
    return {
        "per_invocation_ms_mean": mean_ms,
        "reps_completed": reps_done,
        "reps_requested": reps_requested,
        "projected_total_s_for_requested_reps": (
            None if mean_ms is None else mean_ms * reps_requested / 1000.0
        ),
    }


def summary_pure(tag, side, record, reps):
    flag = " (PARTIAL)" if record["partial"] else ""
    return (
        f"[{tag}] side={side} rep{record['rep']}/{reps}: per-invocation "
        f"{fmt_ms(record['per_invocation_ms'])}ms (block {fmt_ms(record['block_ms'])}ms){flag}"
    )


def save_output_blocks(blocks, directory, tag):
    """Write captured host blocks to ``<directory>/<tag>_output.npz``; returns the path.

    One ``block_<i>`` array per captured block, in capture order (dense: a single
    ``(pop_size, n_rows)`` block; specialized: one ``(n_rows, output_len)`` block per
    non-constant tree).  ``--save-output-dir`` only; the directory is created when missing.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{tag}_output.npz")
    np.savez(path, **{f"block_{index}": block for index, block in enumerate(blocks)})
    return path


def relative_to_cwd(path):
    """``path`` relative to the cwd when it lives inside it, else the absolute path."""
    absolute = os.path.abspath(path)
    relative = os.path.relpath(absolute, os.getcwd())
    return absolute if relative.startswith(os.pardir) else relative


# ---------------------------------------------------------------------------
# R2 accounting: file descriptors, device memory, cold-build guard
# ---------------------------------------------------------------------------


def fd_count():
    """Open file descriptors of this process (Linux ``/proc/self/fd``).

    An error is returned as a ``"<...>"`` string rather than raised: the value is
    pure telemetry and may never abort a timed run.
    """
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def nofile_rlimit():
    """``[soft, hard]`` of ``RLIMIT_NOFILE`` (``-1`` = RLIM_INFINITY, kept verbatim)."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return [int(soft), int(hard)]
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def visible_physical_gpu_index():
    """Physical GPU index named by ``CUDA_VISIBLE_DEVICES``, else ``None``.

    Only a bare single numeric index (``"6"``) is resolved; a list, a UUID/MIG id
    or an unset variable returns ``None`` and the nvidia-smi part is skipped.
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0])
    return None


def nvidia_smi_used_mb(index):
    """``memory.used`` (MiB) of physical GPU ``index``; ``None``/``"<...>"`` on failure."""
    if index is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return f"<nvidia-smi unavailable: {type(exc).__name__}: {exc}>"
    if done.returncode != 0:
        return f"<nvidia-smi exit {done.returncode}: {(done.stderr or '').strip()}>"
    raw = (done.stdout or "").strip().splitlines()
    if not raw:
        return "<nvidia-smi: empty output>"
    try:
        return int(raw[0].strip())
    except ValueError as exc:
        return f"<nvidia-smi parse error: {type(exc).__name__}: raw={raw[0]!r}>"


def cuda_free_mb():
    """``{"free_mb": ..., "total_mb": ...}`` from ``torch.cuda.mem_get_info()``.

    Returns ``None`` (after printing why) when the device cannot report memory: the
    memory guard then degrades to "unguarded" and the run continues.
    """
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        print(
            f"[bench_gp] torch.cuda.mem_get_info() failed, the --mem-guard-mb check is "
            f"skipped for this run ({type(exc).__name__}: {exc})",
            flush=True,
        )
        return None
    return {
        "free_mb": int(free_bytes) // (1024 * 1024),
        "total_mb": int(total_bytes) // (1024 * 1024),
    }


def cold_build_guard_enforced(args, side):
    """Is the ``--max-cold-builds`` guard active for this run?

    Only a new-side run that explicitly asked for the IREE backend builds a GPU
    program per tree structure, so numpy runs and the legacy side are exempt.
    """
    return bool(side == "new" and args.backend == "iree" and not args.allow_cold_builds)


class ColdBuildGuard:
    """Cumulative evoxc compile-miss guard (``--max-cold-builds``).

    Sampling is a no-op unless :func:`cold_build_guard_enforced` holds, and it never
    raises: a failing counter read is printed and ignored so the guard can never
    mask the real run.  Tripping it marks the payload ``partial``/``skipped`` (the
    completed steps stay in the JSON) and the caller stops the run and exits 3.
    """

    def __init__(self, args, payload, enforced):
        self.tag = args.tag
        self.limit = int(args.max_cold_builds)
        self.enforced = bool(enforced)
        self.payload = payload
        self.tripped = False
        self._engine = None

    def sample(self):
        """Sample ``evoxc_engine.cache_stats()['misses']``; True when the guard trips."""
        if not self.enforced:
            return False
        try:
            misses = self._read_misses()
        except Exception as exc:  # telemetry must never abort a run
            print(
                f"[{self.tag}] cold-build guard: cache_stats() read failed at this checkpoint, "
                f"checkpoint ignored ({type(exc).__name__}: {exc})",
                flush=True,
            )
            return False
        record = self.payload["cold_builds"]
        record["misses_last"] = misses
        if record["misses_start"] is None:
            record["misses_start"] = misses
        record["misses_max"] = misses if record["misses_max"] is None else max(record["misses_max"], misses)
        if misses > self.limit:
            self.tripped = True
            record["exceeded"] = True
            self.payload["partial"] = True
            self.payload["skipped"] = True
            self.payload["skip_reason"] = f"cold-builds-exceeded:{misses}>{self.limit}"
            print(
                f"[{self.tag}] cold-build guard: {misses} cumulative evoxc compile misses > "
                f"--max-cold-builds {self.limit}; stopping cleanly (exit 3, JSON partial/skipped)",
                flush=True,
            )
            return True
        return False

    def _read_misses(self):
        if self._engine is None:
            import evogp.tree.evoxc_engine as engine  # new side only (``enforced``)

            self._engine = engine
        return int(self._engine.cache_stats()["misses"])


def mark_cold_checkpoint(payload):
    """Record the post-construction FD count (after the first evaluate/generation)."""
    if payload["fds"]["after_cold"] is None:
        payload["fds"]["after_cold"] = fd_count()


# ---------------------------------------------------------------------------
# R2 stage probe (``test/benchmark/stage_probe.py``; new side only)
# ---------------------------------------------------------------------------


def install_stage_probe(args, side, payload):
    """Install the R2 stage probe, wire ``payload["stages"]``; never raises.

    Returns the ``stage_probe`` module (to be handed to the runners) or ``None``
    when the probe is off / unavailable.  The old side has no ``evoxc_engine``
    module to instrument, so ``--stage-probe`` there warns and no-ops (leaving
    ``stages`` at ``null``) instead of failing the run.

    ``--task eval-pure`` never installs it either: the probe wraps ``evoxc.run`` and
    books every call into stages, i.e. it would tax exactly the tight loop the pure
    task measures -- so the flag warns and no-ops there too (``stages`` stays
    ``null``; the pure numbers come from ``pure``/``compile``/``e2e_companion``).
    """
    if not args.stage_probe:
        return None
    if args.task in PURE_TASKS:
        print(
            f"[{args.tag}] warning: --stage-probe is not installed for --task eval-pure "
            f"(the probe wraps evoxc.run and would tax the timed pure loop); continuing with "
            f"stages=null -- the pure measurement is reported under 'pure'/'compile'",
            flush=True,
        )
        return None
    if side != "new":
        print(
            f"[{args.tag}] warning: --stage-probe is new-side only (side={side}: there is no "
            f"evoxc_engine to instrument); continuing with stages=null",
            flush=True,
        )
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:  # the script directory is not always sys.path[0]
        sys.path.insert(0, here)
    try:
        import stage_probe

        import evogp.tree.evoxc_engine as engine
    except Exception as exc:
        print(
            f"[{args.tag}] warning: --stage-probe could not be installed "
            f"({type(exc).__name__}: {exc}); continuing with stages=null",
            flush=True,
        )
        return None
    status = stage_probe.install(engine)
    payload["stages"] = stage_probe.container(side=side)
    print(f"[{args.tag}] {stage_probe.status_line()}", flush=True)
    return stage_probe


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def run_training(mods, args, payload, flush, guard=None, probe=None):
    """Manual ``pipeline.step()`` loop -- the same code path as ``pipeline.run()``."""
    interrupted = False
    for rep in range(args.reps):
        record = new_rep_record(rep, "train")
        payload["per_rep"].append(record)
        if probe is not None:  # R2 stage probe: one block per rep, reset here
            probe.begin_rep(rep, "train")
        if rep == 0:  # top-level arrays share rep 0's list objects
            payload["per_gen_ms"] = record["per_gen_ms"]
            payload["per_gen_best_fitness"] = record["per_gen_best_fitness"]
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()  # wall_s starts just before construction
        problem, descriptor, description = build_problem_and_descriptor(mods, args)
        initial_forest = mods.Forest.random_generate(pop_size=args.pop, descriptor=descriptor)
        algorithm = build_algorithm(mods, args, descriptor, initial_forest)
        pipeline = mods.StandardPipeline(
            algorithm,
            problem,
            generation_limit=args.gens,
            is_show_details=False,
        )
        torch.cuda.synchronize()
        record["construct_s"] = time.perf_counter() - wall_t0
        payload["problem"] = description
        mirror_rep0(payload, record, TRAIN_MIRROR)
        flush()

        try:
            for gen in range(args.gens):
                if probe is not None:
                    probe.begin()  # R2 stage probe: one record per generation
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pipeline.step()
                torch.cuda.synchronize()
                record["per_gen_ms"].append((time.perf_counter() - t0) * 1000.0)
                record["per_gen_best_fitness"].append(float(pipeline.best_fitness))
                record["cold_first_gen_ms"] = record["per_gen_ms"][0]
                record["best_fitness_final"] = record["per_gen_best_fitness"][-1]
                mirror_rep0(payload, record, TRAIN_MIRROR)
                if probe is not None:
                    probe.snapshot(i=gen, outer_ms=record["per_gen_ms"][-1])
                flush()  # a hard kill still leaves every finished generation
                if gen == 0:
                    mark_cold_checkpoint(payload)  # FD/cold-build checkpoint: first gen
                if guard is not None and guard.sample():
                    record["partial"] = True
                    break
                if (
                    args.max_wall_s is not None
                    and time.perf_counter() - wall_t0 >= args.max_wall_s
                    and gen + 1 >= args.min_gens
                ):
                    record["partial"] = True
                    record["projection"] = train_projection(
                        record["per_gen_ms"], gen + 1, args.gens
                    )
                    print(
                        f"[{args.tag}] side={payload['side']} soft wall cap "                        f"--max-wall-s={args.max_wall_s}s reached after {gen + 1}/{args.gens} gens",
                        flush=True,
                    )
                    break
        except KeyboardInterrupt:
            interrupted = True
            record["partial"] = True
            print(f"[{args.tag}] interrupted after {len(record['per_gen_ms'])} gens", flush=True)

        torch.cuda.synchronize()
        record["wall_s"] = time.perf_counter() - wall_t0
        record["generation_loop_s"] = sum(record["per_gen_ms"]) / 1000.0
        record["complete"] = True
        mirror_rep0(payload, record, TRAIN_MIRROR)
        if probe is not None:  # R2 stage probe: per-rep totals + one summary line
            line = probe.finish_rep()
            if line:
                print(f"[{args.tag}] {line}", flush=True)
        flush()
        print(summary_training(args.tag, payload["side"], record, args.reps), flush=True)
        if interrupted:
            break
        if guard is not None and guard.tripped:  # stopped by the cold-build guard
            break
    return interrupted


def run_eval(mods, args, payload, flush, guard=None, probe=None):
    """Repeated ``problem.evaluate(forest)`` on a fixed population."""
    interrupted = False
    run_t0 = time.perf_counter()
    calls_requested = args.reps * args.eval_reps
    for rep in range(args.reps):
        record = new_rep_record(rep, "eval")
        payload["per_rep"].append(record)
        if rep == 0:
            payload["per_call_ms"] = record["per_call_ms"]
        if probe is not None:  # R2 stage probe: one block per rep, reset here
            probe.begin_rep(rep, "eval")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        problem, forest, eval_kind, description = build_eval_case(mods, args)
        torch.cuda.synchronize()
        record["construct_s"] = time.perf_counter() - wall_t0
        payload["problem"] = description
        payload["eval_kind"] = eval_kind
        payload["config"]["pop"] = int(forest.pop_size)
        payload["config"]["tree_len"] = int(forest.max_tree_len)
        mirror_rep0(payload, record, EVAL_MIRROR)
        flush()

        try:
            for call in range(args.eval_reps):
                if probe is not None:
                    probe.begin()  # R2 stage probe: one record per evaluate call
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                problem.evaluate(forest)
                torch.cuda.synchronize()
                record["per_call_ms"].append((time.perf_counter() - t0) * 1000.0)
                if probe is not None:
                    probe.snapshot(i=call, outer_ms=record["per_call_ms"][-1])
                mirror_rep0(payload, record, EVAL_MIRROR)
                flush()
                if call == 0:
                    mark_cold_checkpoint(payload)  # FD/cold-build checkpoint: first eval
                if guard is not None and guard.sample():
                    record["partial"] = True
                    break
        except KeyboardInterrupt:
            interrupted = True
            record["partial"] = True
            print(f"[{args.tag}] interrupted after {len(record['per_call_ms'])} calls", flush=True)
        torch.cuda.synchronize()
        record["wall_s"] = time.perf_counter() - wall_t0
        record["first_call_ms"] = record["per_call_ms"][0] if record["per_call_ms"] else None
        warm = record["per_call_ms"][1:]
        record["median_warm_ms"] = statistics.median(warm) if len(warm) >= 2 else None
        record["complete"] = True
        mirror_rep0(payload, record, EVAL_MIRROR)
        if probe is not None:  # R2 stage probe: per-rep totals + one summary line
            line = probe.finish_rep()
            if line:
                print(f"[{args.tag}] {line}", flush=True)
        flush()
        print(summary_eval(args.tag, payload["side"], record, args.reps), flush=True)

        if interrupted:
            break
        if guard is not None and guard.tripped:  # stopped by the cold-build guard
            break
        calls_done = sum(len(r["per_call_ms"]) for r in payload["per_rep"])
        if (
            args.max_wall_s is not None
            and time.perf_counter() - run_t0 >= args.max_wall_s
            and calls_done >= args.min_gens
        ):
            projection = eval_projection(
                [ms for r in payload["per_rep"] for ms in r["per_call_ms"]],
                calls_done,
                calls_requested,
            )
            record["partial"] = True
            record["projection"] = projection
            payload["partial"] = True
            payload["projection"] = projection
            print(
                f"[{args.tag}] side={payload['side']} soft wall cap "
                f"--max-wall-s={args.max_wall_s}s reached after {calls_done}/{calls_requested} calls",
                flush=True,
            )
            break
    return interrupted


def run_pure(mods, args, payload, flush, guard=None, probe=None):
    """``--task eval-pure``: the pure forward measurement on a fixed population (new side).

    One invocation = one full-population forward on the fixed ``x`` with ``per_row=False``,
    with the readback / host assembly / key derivation excluded (see ``pure_forward.py`` and
    ``test/benchmark/r4/README.md``).  The runner is constructed ONCE for the whole task
    (timed separately into ``compile``, never mixed into a pure number) and warmed up once,
    then ``--reps`` timed blocks of ``--invocations`` invocations each; the reps only repeat
    the timed block (this path consumes no RNG, so the per-rep reseed of the train/eval
    runners would be pure noise here).  ``--e2e-companion K`` then runs the real end-to-end
    ``problem.evaluate(forest)`` K times in the same process, with the compile cache already
    warm from the construction, so ``E2E - pure`` prices the per-call host glue.

    ``probe`` is always ``None`` for this task (:func:`install_stage_probe` refuses to
    install it: the probe wraps ``evoxc.run`` and would tax exactly the loop being measured).
    """
    interrupted = False
    if payload["side"] != "new":
        reason = (
            "eval-pure is a new-side measurement only: the legacy CUDA checkout has no "
            "separable pure forward (one fused kernel call per evaluate, with the readback "
            "and the host assembly inside it)"
        )
        payload["pure"] = None
        payload["skipped"] = True
        payload["skip_reason"] = reason
        payload["complete"] = True
        flush()
        print(f"[{args.tag}] SKIPPED (exit 3): {reason}", flush=True)
        return interrupted

    pure = payload["pure"]
    pure_forward = import_pure_forward()
    wall_t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.synchronize()
    problem, forest, eval_kind, description = build_eval_case(mods, args)
    x = problem.datapoints
    payload["problem"] = description
    payload["eval_kind"] = eval_kind
    payload["config"]["pop"] = int(forest.pop_size)
    payload["config"]["tree_len"] = int(forest.max_tree_len)
    payload["config"]["n_rows"] = int(x.shape[0])

    backend_name = pure_forward.resolve_backend_name(args.backend)
    resolved = pure_forward.resolve_mode(args.mode, int(forest.pop_size), backend_name)
    pure["mode_requested"] = resolved["mode_requested"]
    pure["mode_resolved"] = resolved["mode_resolved"]
    pure["backend"] = backend_name
    pure["pop"] = int(forest.pop_size)
    flush()
    print(
        f"[{args.tag}] pure: pop={forest.pop_size} n_rows={int(x.shape[0])} "
        f"mode_requested={pure['mode_requested']} mode_resolved={pure['mode_resolved']} "
        f"backend={backend_name} runner={args.pure_runner} invocations={args.invocations} "
        f"warmup={args.warmup} reps={args.reps} e2e_companion={args.e2e_companion}",
        flush=True,
    )

    # ---- construct once: the cold, separately reported compile block ----------
    runner, compile_record = pure_forward.build_pure_runner(
        forest,
        x,
        mode=pure["mode_resolved"],
        runner_source=args.pure_runner,
        backend=args.backend,
        tag=args.tag,
    )
    construct_s = time.perf_counter() - wall_t0
    pure.update(runner.describe())
    payload["compile"] = compile_record
    flush()
    print(
        f"[{args.tag}] constructed pure runner: source={runner.source} "
        f"n_trees={runner.n_trees} n_const_trees={runner.n_const_trees} "
        f"graph_calls/invocation={runner.n_graph_calls_per_run} "
        f"({runner.n_graph_calls_source}); compile {fmt_ms(compile_record['construct_total_ms'])}ms "
        f"(decode_plan {fmt_ms(compile_record['decode_plan_ms'])}ms + build "
        f"{fmt_ms(compile_record['build_ms'])}ms; {compile_record['cache_misses']} cache misses)",
        flush=True,
    )

    # ---- one untimed warmup block, then the cold/FD checkpoints --------------
    if args.warmup > 0:
        pure_forward.time_block(runner, args.warmup)  # measurement intentionally discarded
    mark_cold_checkpoint(payload)  # FD/cold-build checkpoint: after the construction
    if guard is not None and guard.sample():
        payload["partial"] = True
        payload["median_per_invocation_ms"] = pure["median_per_invocation_ms"]
        return interrupted  # payload["skipped"] is set -> main exits 3

    # ---- R timed blocks ------------------------------------------------------
    reps_requested = int(args.reps)
    blocks_done = 0
    for rep in range(reps_requested):
        record = new_pure_rep_record(rep)
        payload["per_rep"].append(record)
        record["mode_resolved"] = pure["mode_resolved"]
        record["runner_source"] = pure["runner_source"]
        if rep == 0:
            record["construct_s"] = construct_s
        rep_t0 = wall_t0 if rep == 0 else time.perf_counter()
        try:
            record["block_ms"] = pure_forward.time_block(runner, args.invocations)
        except KeyboardInterrupt:
            interrupted = True
            record["partial"] = True
            payload["partial"] = True
            record["wall_s"] = time.perf_counter() - rep_t0
            flush()
            print(
                f"[{args.tag}] interrupted inside rep {rep}'s timed block "
                f"({blocks_done} block(s) completed)",
                flush=True,
            )
            break
        record["per_invocation_ms"] = record["block_ms"] / int(args.invocations)
        record["wall_s"] = time.perf_counter() - rep_t0
        record["complete"] = True
        pure["block_ms"].append(record["block_ms"])
        pure["per_invocation_ms"].append(record["per_invocation_ms"])
        pure.update(
            pure_forward.summarize(
                pure["per_invocation_ms"],
                n_graph_calls_per_run=pure["n_graph_calls_per_run"],
                n_trees=pure["n_trees"],
            )
        )
        payload["median_per_invocation_ms"] = pure["median_per_invocation_ms"]
        blocks_done = rep + 1
        flush()
        print(summary_pure(args.tag, payload["side"], record, reps_requested), flush=True)
        if guard is not None and guard.sample():  # stopped by the cold-build guard
            record["partial"] = True
            break
        if args.max_wall_s is not None and time.perf_counter() - wall_t0 >= args.max_wall_s:
            projection = pure_projection(pure["per_invocation_ms"], blocks_done, reps_requested)
            record["partial"] = True
            record["projection"] = projection
            payload["partial"] = True
            payload["projection"] = projection
            print(
                f"[{args.tag}] side={payload['side']} soft wall cap "
                f"--max-wall-s={args.max_wall_s}s reached after {blocks_done}/{reps_requested} "
                f"blocks",
                flush=True,
            )
            break

    # ---- end-to-end companion (same process, warm cache) ---------------------
    if args.e2e_companion > 0 and not interrupted and not payload["skipped"] and blocks_done:
        try:
            payload["e2e_companion"] = pure_forward.time_e2e_companion(
                problem, forest, args.e2e_companion
            )
        except KeyboardInterrupt:
            interrupted = True
            payload["partial"] = True
            payload["e2e_companion"] = None
            print(f"[{args.tag}] interrupted inside the e2e companion", flush=True)
        else:
            companion = payload["e2e_companion"]
            print(
                f"[{args.tag}] e2e companion: first "
                f"{fmt_ms(companion['first_call_ms'])}ms, warm median "
                f"{fmt_ms(companion['median_warm_ms'])}ms over {companion['eval_reps']} "
                f"problem.evaluate() calls (pure median "
                f"{fmt_ms(pure['median_per_invocation_ms'])}ms)",
                flush=True,
            )
        flush()
    # ---- output capture (out of band: never inside a timed region) -----------
    # Deliberately LAST: the capture performs one extra invocation plus a consumer-side
    # readback, so it must sit after the timed reps AND after the e2e companion to stay out
    # of every timed number (see ``pure_forward.PureRunner.capture_outputs``).  A capture
    # failure propagates: a record claiming a checksum it could not produce is worse than a
    # failed run.  Interrupted / guard-skipped runs keep the frozen schema (no key at all)
    # rather than a digest of a capture that never happened.
    if (args.output_checksum or args.save_output_dir) and not interrupted and not payload["skipped"]:
        blocks = runner.capture_outputs()
        digest = pure_forward.digest_blocks(blocks)
        if args.save_output_dir:
            digest["saved"] = relative_to_cwd(
                save_output_blocks(blocks, args.save_output_dir, args.tag)
            )
        digest["note"] = pure_forward.OUTPUT_CHECKSUM_NOTE
        pure["output_checksum"] = digest
        flush()
        print(
            f"[{args.tag}] output capture (out of band, after all timing): "
            f"n_blocks={digest['n_blocks']} shapes={digest['shapes']} "
            f"total_elems={digest['total_elems']} nan={digest['nan_count']} "
            f"inf={digest['inf_count']} sha256={digest['sha256']}"
            + (f" saved={digest['saved']}" if "saved" in digest else ""),
            flush=True,
        )
    elif args.output_checksum or args.save_output_dir:
        print(
            f"[{args.tag}] output capture skipped (interrupted={interrupted} "
            f"skipped={payload['skipped']}): no pure.output_checksum key is written",
            flush=True,
        )
    return interrupted


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def r2_config(args):
    """R2 additions to ``config`` (recorded for both the train and the eval tasks)."""
    return {
        "backend": args.backend,
        "iree_device": args.iree_device,
        "iree_opt_level": args.iree_opt_level,
        "data_rows": args.data_rows,
        "mem_guard_mb": args.mem_guard_mb,
        "max_cold_builds": args.max_cold_builds,
        "allow_cold_builds": args.allow_cold_builds,
        "stage_probe": args.stage_probe,
    }


def r2_payload_defaults(args):
    """R2 top-level keys: guards/exit state, FD accounting, compile-miss counters."""
    return {
        "skipped": False,
        "skip_reason": None,
        "fds": {"start": None, "after_cold": None, "end": None},
        "nofile_rlimit": None,
        #: ``stages`` is filled by the ``--stage-probe`` hooks (one block per rep);
        #: it stays ``null`` whenever the probe is off or unavailable.
        "stages": None,
        "cold_builds": {
            "enforced": False,  # re-derived in build_payload once the side is known
            "limit": args.max_cold_builds,
            "allow_cold_builds": args.allow_cold_builds,
            "misses_start": None,
            "misses_last": None,
            "misses_max": None,
            "exceeded": False,
        },
    }


def build_payload(args, side, mods, started_at):
    if args.task in TRAIN_TASKS:
        payload = {
            "tag": args.tag,
            "side": side,
            "task": args.task,
            "evogp_file": None,
            "evogp_commit": None,
            "etl_version": None,
            "problem": None,
            "config": {
                "pop": args.pop,
                "gens": args.gens,
                "tree_len": args.tree_len,
                "seed": args.seed,
                "reps": args.reps,
                **r2_config(args),
            },
            "env": {},
            "gpu": {},
            "wall_s": None,
            "generation_loop_s": None,
            "per_gen_ms": [],
            "per_gen_best_fitness": [],
            "cold_first_gen_ms": None,
            "best_fitness_final": None,
            "partial": False,
            "projection": None,
            "complete": False,
            "per_rep": [],
            "notes": args.notes,
            "started_at": started_at,
            "finished_at": None,
        }
    elif args.task in PURE_TASKS:
        payload = {
            "tag": args.tag,
            "side": side,
            "task": args.task,
            "eval_kind": None,
            "fixture": os.path.abspath(args.fixture),
            "evogp_file": None,
            "evogp_commit": None,
            "etl_version": None,
            "problem": None,
            "config": {
                "pop": args.pop,
                "tree_len": args.tree_len,
                "seed": args.seed,
                "reps": args.reps,
                "eval_reps": args.eval_reps,
                "mode": args.mode,
                "pure_runner": args.pure_runner,
                "invocations": args.invocations,
                "warmup": args.warmup,
                "e2e_companion": args.e2e_companion,
                **r2_config(args),
            },
            "env": {},
            "gpu": {},
            #: The pure measurement: static keys + one entry per rep.  ``None`` on the old
            #: side, which skips cleanly (it has no separable pure forward).
            "pure": None if side != "new" else pure_block(args),
            #: The cold construction, always reported separately from the pure numbers.
            "compile": compile_block(),
            #: The end-to-end companion (``--e2e-companion``), or ``null``.
            "e2e_companion": None,
            #: Running mirror of ``pure.median_per_invocation_ms`` (the
            #: ``median_warm_ms`` convention of the eval tasks).
            "median_per_invocation_ms": None,
            "wall_s": None,
            "partial": False,
            "projection": None,
            "complete": False,
            "per_rep": [],
            "notes": args.notes,
            "started_at": started_at,
            "finished_at": None,
        }
    else:
        payload = {
            "tag": args.tag,
            "side": side,
            "task": args.task,
            "eval_kind": None,
            "fixture": os.path.abspath(args.fixture),
            "evogp_file": None,
            "evogp_commit": None,
            "etl_version": None,
            "problem": None,
            "config": {
                "pop": args.pop,
                "tree_len": args.tree_len,
                "seed": args.seed,
                "reps": args.reps,
                "eval_reps": args.eval_reps,
                **r2_config(args),
            },
            "env": {},
            "gpu": {},
            "per_call_ms": [],
            "first_call_ms": None,
            "median_warm_ms": None,
            "wall_s": None,
            "partial": False,
            "projection": None,
            "complete": False,
            "per_rep": [],
            "notes": args.notes,
            "started_at": started_at,
            "finished_at": None,
        }
    repo = repo_snapshot(getattr(mods.evogp, "__file__", None))
    payload["evogp_file"] = repo["evogp_file"]
    payload["evogp_commit"] = repo["evogp_commit"]
    payload["evogp_commit_error"] = repo["evogp_commit_error"]
    payload["cwd"] = repo["cwd"]
    payload["etl_version"] = etl_version()
    payload["env"] = env_snapshot()
    payload["gpu"] = gpu_snapshot()
    payload["harness"] = os.path.abspath(__file__)
    payload.update(r2_payload_defaults(args))
    payload["cold_builds"]["enforced"] = cold_build_guard_enforced(args, side)
    return payload


def main(argv=None):
    args = parse_args(argv)
    apply_env(args)
    started_at = now_iso()
    t0 = time.perf_counter()
    mods = import_modules()
    side = detect_side()
    payload = build_payload(args, side, mods, started_at)
    payload["fds"]["start"] = fd_count()  # right after the imports, before any work
    payload["nofile_rlimit"] = nofile_rlimit()

    json_out = args.json_out

    def flush():
        write_json_atomic(json_out, payload)

    # ---- R2 guards: device free memory, then the compile-miss guard -----------
    gpu_index = visible_physical_gpu_index()
    payload["gpu"]["nvidia_smi_index"] = gpu_index
    payload["gpu"]["nvidia_smi_used_mb_start"] = nvidia_smi_used_mb(gpu_index)
    guard = ColdBuildGuard(args, payload, payload["cold_builds"]["enforced"])
    free = cuda_free_mb()
    payload["gpu"]["free_mb_start"] = None if free is None else free["free_mb"]
    if free is None:
        print(
            f"[{args.tag}] warning: --mem-guard-mb={args.mem_guard_mb} could not be evaluated "
            f"(no free-memory reading); continuing unguarded",
            flush=True,
        )
    elif free["free_mb"] < args.mem_guard_mb:
        reason = f"free_mem_below_guard:{free['free_mb']}MiB<{args.mem_guard_mb}MiB"
        payload["skipped"] = True
        payload["skip_reason"] = reason
        payload["finished_at"] = now_iso()
        payload["harness_wall_s"] = time.perf_counter() - t0
        payload["complete"] = True
        flush()
        print(f"[{args.tag}] SKIPPED (exit 3): {reason} -- no heavy step was started", flush=True)
        return 3
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        print(
            f"[{args.tag}] warning: torch.cuda.reset_peak_memory_stats() failed "
            f"({type(exc).__name__}: {exc}); the reported peak covers the whole process",
            flush=True,
        )
    if guard.enforced:
        print(
            f"[{args.tag}] cold-build guard ACTIVE: stop when the cumulative evoxc compile "
            f"misses exceed --max-cold-builds={guard.limit}",
            flush=True,
        )
    elif args.backend == "iree" and not args.allow_cold_builds:
        print(
            f"[{args.tag}] cold-build guard inactive (side={side}): only a new-side iree run "
            f"is capped",
            flush=True,
        )
    guard.sample()  # startup sample -> cold_builds.misses_start
    if args.data_rows is not None and args.task != "eval64" and args.task not in PURE_TASKS:
        print(
            f"[{args.tag}] warning: --data-rows is only applied by --task eval64 / eval-pure; "
            f"ignored for --task {args.task}",
            flush=True,
        )
    if (args.output_checksum or args.save_output_dir) and args.task not in PURE_TASKS:
        print(
            f"[{args.tag}] warning: --output-checksum / --save-output-dir are only applied by "
            f"--task eval-pure; ignored for --task {args.task}",
            flush=True,
        )
    # R2 stage probe: install the monkeypatches before the first timed call
    probe = install_stage_probe(args, side, payload)

    flush()  # metadata on disk before the first heavy step
    print(
        f"[{args.tag}] harness=bench_gp.py side={side} task={args.task} "
        f"pop={args.pop} gens={args.gens} tree_len={args.tree_len} seed={args.seed} "
        f"reps={args.reps} eval_reps={args.eval_reps} "
        f"EVOXC_DISABLE_LOCATIONS={payload['env'].get('EVOXC_DISABLE_LOCATIONS')} "
        f"EVOGP_EVOXC_BACKEND={payload['env'].get('EVOGP_EVOXC_BACKEND')} "
        f"CUDA_VISIBLE_DEVICES={payload['env'].get('CUDA_VISIBLE_DEVICES')} "
        f"gpu={payload['gpu'].get('name')} evogp={payload['evogp_file']} "
        f"commit={payload['evogp_commit']} etl={payload['etl_version']} "
        f"json_out={json_out}",
        flush=True,
    )

    interrupted = False
    try:
        if args.task in TRAIN_TASKS:
            interrupted = run_training(mods, args, payload, flush, guard, probe)
        elif args.task in PURE_TASKS:
            interrupted = run_pure(mods, args, payload, flush, guard, probe)
        else:
            interrupted = run_eval(mods, args, payload, flush, guard, probe)
    except KeyboardInterrupt:
        interrupted = True
        payload["partial"] = True
        print(f"[{args.tag}] interrupted outside a timed step", flush=True)

    # ---- end-of-run telemetry (R2): FD end, peaks, nvidia-smi ---------------
    payload["fds"]["end"] = fd_count()
    payload["gpu"]["peak_allocated_bytes"] = _guarded(
        "peak_allocated_bytes", lambda: int(torch.cuda.max_memory_allocated())
    )
    payload["gpu"]["peak_reserved_bytes"] = _guarded(
        "peak_reserved_bytes", lambda: int(torch.cuda.max_memory_reserved())
    )
    payload["gpu"]["nvidia_smi_used_mb_end"] = nvidia_smi_used_mb(gpu_index)

    payload["finished_at"] = now_iso()
    payload["harness_wall_s"] = time.perf_counter() - t0
    payload["complete"] = True  # this is the harness' final, deliberate write
    flush()
    print(
        f"[{args.tag}] done side={side} interrupted={interrupted} "
        f"harness_wall_s={payload['harness_wall_s']:.1f} json_out={json_out}",
        flush=True,
    )
    if json_out is None:
        print(json.dumps(payload, indent=2, allow_nan=True), flush=True)
    if payload["skipped"]:
        print(
            f"[{args.tag}] stopped by a guard (exit 3): {payload['skip_reason']} "
            f"(completed steps kept, partial={payload['partial']})",
            flush=True,
        )
        return 3
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
