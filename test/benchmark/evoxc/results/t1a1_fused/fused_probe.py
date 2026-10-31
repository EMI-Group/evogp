"""Per-cell evidence probe for the FUSED (Triton) evaluation route -- one process per cell.

The campaign records a timing and a digest; this probe records what makes a fused
number *believable* on one cell: that the ROUTE POLICY really names the fused engine,
that the facade really SERVED each call (kernel launches, not just attempts), that the
answer equals the harness's own ``torch-batched`` arm and the CPU oracle, where the
warm call's time goes (kernel vs host glue vs the transpose), how much of it is cold,
and -- with ``--gp`` -- that a changing population re-plans nothing (Triton
specializations stay put while the transpose table keeps hitting).

Usage (repo root; the two route knobs come from the ENVIRONMENT -- the probe never sets
them, so a record answers "what did this environment measure", not "what did the probe
ask for")::

    TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache \\
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \\
    EVOGP_EVAL_MODE=fused EVOGP_EVOXC_BACKEND=iree \\
        /tmp/rocvenv/bin/python test/benchmark/evoxc/results/t1a1_fused/fused_probe.py \\
        --cell p256-L32-i8-N1024 --out test/benchmark/evoxc/results/t1a1_fused/probes/p256-L32-i8-N1024.json

``--help`` needs nothing but the standard library: every heavy import (torch, the
harness, evogp) happens inside :func:`main` after the paths are set, so the CLI can be
inspected on a box with no device at all.

One process per cell is the harness's protocol (a warm median is only comparable if every
process sees one shape class, and Triton's JIT cache / the engine's tables are per
process).  ``--cold-cache`` narrows that further: each surface is called EXACTLY once
(no warmup, no reps), so a driver can run it against a fresh ``TRITON_CACHE_DIR`` and get
the honest cold number -- in that mode the stage split is skipped (it would re-time the
facade on an already warm process) and the record says ``cold_cache: true``.

What the record holds
---------------------
``cell`` / ``arm`` / ``probe``
    Identity, interpreter / torch / GPU / triton-distribution versions and the route
    knobs the process inherited (``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` /
    ``EVOGP_FUSED_*`` / ``TRITON_CACHE_DIR``), read from ``os.environ``.
``route``
    ``dense_engine.resolve_route(pop, rows=...)`` for the three surfaces -- the
    explicit proof the POLICY names fused (``engine`` / ``use_fused`` / ``backend`` /
    ``fallback_backend`` / ``reason``), not an inference from the timings.
``structures``
    ``worker.forest_stats`` (pop, tree lengths, distinct structures + signature).
``workloads``
    Per surface (``batch_forward`` / ``sr_fitness`` / ``forward``) the harness's
    ``common.time_call`` block minus ``_out`` (cold first call, warm samples, median /
    min / max / mean) plus ``fused_delta`` (the 13 ``fused_*`` counters across the whole
    window) and ``served`` -- ``fused_kernel_launches == 1 + warmup + reps`` AND
    ``fused_declines == 0``, i.e. a launch per call rather than a silent fallback.
``numeric``
    Per surface, ``validate.compare`` (reference = the harness's ``torch-batched`` arm
    on the SAME tensors, candidate = the fused result) plus, with
    ``--oracle-trees K``, an ``oracle`` sub-block comparing the first ``K`` trees
    against ``src/evogp/tests/reference_eval.py`` (CPU, sliced: it is a slow Python
    per-tree loop).  Each block states ``trees_compared`` / ``rows``.
``stage_split``
    Warm medians (2 warmup + 10 reps) of the full facade call, the raw
    ``fused_kernels.launch_value`` / ``launch_loss`` launch and the un-memoized
    ``x.t().contiguous()`` transpose, with ``kernel_ms``, ``host_ms = facade - kernel``,
    ``transpose_ms`` and the kernel's share of the call -- the recipe of
    ``fused_check._stage_split``, which this mirrors deliberately (same protocol, same
    three legs) instead of inventing a second definition of host glue.
``cold``
    Always the three ``first_call_ms`` values (the compile-bearing first call of each
    surface), i.e. the honest cold column even in a warm run.
``gp_step``
    With ``--gp``: ``arms.make_pipeline(workload, timed_fitness)`` stepped
    ``1`` warmup + ``5`` reps, each step split into ``eval_ms`` (the in-place
    ``SR_fitness`` call) and ``ops_ms`` (the operators), with the per-step counter delta
    and the aggregate ``per_generation`` block -- the evidence that a fresh population
    costs no new specialization (``kernel_compiles_total`` stays 0, the transpose table
    hits keep growing).

Floats are rounded to 4 decimals and no tensor is stored, so a record stays small.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
import time

#: ``<repo>`` -- found by :func:`_find_repo_root` from this file's own location (it
#: lives INSIDE the harness now), so no hop count is baked in.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start: str) -> str:
    """Resolve ``<repo>`` from this file's location, without naming a worktree.

    This file lives INSIDE the harness now (``<repo>/test/benchmark/evoxc/results/
    t1a1_fused/``), so the scan looks for the harness directory itself -- the one that
    holds ``campaign.py`` and ``common.py`` -- by ascending from this file, and takes the
    three hops back up from it.  A bounded scan instead of a fixed hop count is what keeps
    a copy placed one level deeper finding the checkout instead of mis-resolving.
    """
    probe = os.path.abspath(start)
    for _ in range(8):
        if (os.path.isfile(os.path.join(probe, "campaign.py"))
                and os.path.isfile(os.path.join(probe, "common.py"))):
            return os.path.abspath(os.path.join(probe, "..", "..", ".."))
        probe = os.path.dirname(probe)
    return os.path.abspath(os.path.join(start, "..", "..", "..", "..", ".."))


_REPO_ROOT = _find_repo_root(_HERE)

#: The library under test, the harness and the evaluation-engine checkout.
SRC_DIR = os.path.join(_REPO_ROOT, "src")
HARNESS_DIR = os.path.join(_REPO_ROOT, "test", "benchmark", "evoxc")
EVOXC_ROOT = "/root/evoxc"

#: The arm this probe is the evidence for (the harness's ``EVOXC_ARMS`` spelling: an
#: explicit ``iree`` backend makes the fused route resolve unconditionally).
ARM = "evoxc-fused-iree"
SCHEMA = "evogp-fused-probe-v1"

#: The three public surfaces, in record order.
SURFACES = ("batch_forward", "sr_fitness", "forward")
#: Data rows each surface is evaluated over: ``forward`` is the per-row path (one input
#: row per tree), so its "rows" axis is 1.
SURFACE_ROWS = {"batch_forward": None, "sr_fitness": None, "forward": 1}

#: The ``EVOGP_FUSED_*`` knobs recorded verbatim (unset reads as ``None``).
FUSED_KNOBS = (
    "EVOGP_FUSED_BLOCK_ROWS",
    "EVOGP_FUSED_NUM_WARPS",
    "EVOGP_FUSED_MAX_SCRATCH_MB",
    "EVOGP_FUSED_HOST_MEMO",
    "EVOGP_FUSED_MIN_POP",
)

#: The GP-step protocol (the campaign's own defaults) and the stage split's.
GP_WARMUP = 1
GP_REPS = 5
STAGE_WARMUP = 2
STAGE_REPS = 10


def bootstrap_paths() -> list:
    """Put the library, the harness and the evoxc checkout first on ``sys.path``."""
    added = []
    for path in (HARNESS_DIR, SRC_DIR, EVOXC_ROOT):
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added


# --------------------------------------------------------------------------- #
# small value helpers
# --------------------------------------------------------------------------- #
def _num(value, ndigits: int = 4):
    """``round(float(value), ndigits)``, or ``None`` for a null reading."""
    if value is None:
        return None
    return round(float(value), ndigits)


def _package_version(name: str):
    """The installed distribution version of ``name`` (``None`` when not installed).

    Read from the metadata rather than by importing the module: importing ``triton``
    here would move ~57 ms of the route's cold cost out of the first timed call, and
    the cold column is supposed to carry it.
    """
    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001 - a missing optional package is a reportable null
        return None


def _snapshot(fused_engine) -> dict:
    """One ``fused_cache_stats()`` snapshot as plain ints."""
    return {key: int(value) for key, value in fused_engine.fused_cache_stats().items()}


def _delta(before: dict, after: dict) -> dict:
    """Per-key counter delta of two snapshots (the 13 ``fused_*`` keys)."""
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _time_single(fn, torch):
    """One sync-bracketed call -> ``(out, ms)`` (the ``--cold-cache`` leg)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1e3


def _median(values):
    """Median of a non-empty list of numbers (``None`` for an empty one)."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


# --------------------------------------------------------------------------- #
# the four recorded sections
# --------------------------------------------------------------------------- #
def probe_block(torch) -> dict:
    """Interpreter / device / package provenance plus the route knobs as inherited."""
    gpu, gpu_total_gb = None, None
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = props.name
            gpu_total_gb = round(props.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001 - provenance must never fail the probe
        gpu = f"unavailable: {exc!r}"[:120]
    env = {
        name: os.environ.get(name)
        for name in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", *FUSED_KNOBS, "TRITON_CACHE_DIR")
    }
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "gpu": gpu,
        "gpu_total_memory_gb": gpu_total_gb,
        "triton": _package_version("triton"),
        "iree_base_compiler": _package_version("iree-base-compiler"),
        "env": env,
        "one_process_per_cell": True,
    }


def route_block(dense_engine, cell: dict) -> dict:
    """``resolve_route`` for each surface: the policy's own answer, per call shape."""
    pop = int(cell["pop"])
    block = {}
    for name in SURFACES:
        rows = SURFACE_ROWS[name]
        rows = int(cell["rows"]) if rows is None else int(rows)
        route = dense_engine.resolve_route(pop, rows=rows, tree_len=int(cell["tree_len"]))
        block[name] = {
            "rows": rows,
            "engine": route.engine,
            "use_fused": bool(route.use_fused),
            "backend": route.backend,
            "fallback_backend": route.fallback_backend,
            "reason": route.reason,
        }
    return block


def workload_block(args, common, fused_engine, calls: dict):
    """Time the three surfaces and prove each was SERVED (not fallen back).

    Returns ``(blocks, outputs)``: the per-surface record entries and the result
    tensors the numeric section compares (the timed call's own output, so the number and
    the value always belong to the same invocation).
    """
    import torch

    blocks, outputs = {}, {}
    for name in SURFACES:
        fn = calls[name]
        before = _snapshot(fused_engine)
        if args.cold_cache:
            # ONE call per surface: this is where Triton's JIT lands, and the whole
            # point of the mode is to keep it there.
            expected = 1
            out, first_ms = _time_single(fn, torch)
            entry = {
                "warmup": 0,
                "reps": 0,
                "first_call_ms": _num(first_ms),
                "samples_ms": [],
                "median_ms": None,
                "min_ms": None,
                "max_ms": None,
                "mean_ms": None,
            }
        else:
            expected = 1 + int(args.warmup) + int(args.reps)
            timing = common.time_call(
                fn,
                warmup=int(args.warmup),
                reps=int(args.reps),
                stats=fused_engine.fused_cache_stats,
            )
            out = timing.pop("_out")
            entry = common.public_timing(timing)
            for key in ("first_call_ms", "median_ms", "min_ms", "max_ms", "mean_ms"):
                entry[key] = _num(entry.get(key))
            entry["samples_ms"] = [_num(value) for value in entry.get("samples_ms") or []]
        after = _snapshot(fused_engine)
        delta = _delta(before, after)
        entry.setdefault("cache_after_first_call", after)
        entry["cold_cache"] = bool(args.cold_cache)
        entry["calls_expected"] = expected
        entry["fused_delta"] = delta
        # A served call launches the kernel ONCE per invocation and declines never; a
        # silent fallback leaves the launches at zero (or moves ``fused_declines``), so
        # the assertion is on LAUNCHES, never on ``fused_calls`` alone (an attempt is
        # bumped before the try in the facade).
        entry["served"] = (
            delta.get("fused_kernel_launches", 0) == expected
            and delta.get("fused_declines", 0) == 0
        )
        blocks[name] = entry
        outputs[name] = out
    return blocks, outputs


def numeric_block(args, common, harness_validate, arm, outputs: dict, forest, x, labels, x_per_row):
    """Fused vs the harness's ``torch-batched`` arm, and (sliced) vs the CPU oracle."""
    pop = int(forest.pop_size)
    rows = int(x.shape[0])
    numeric = {}
    references = {
        "batch_forward": lambda: arm.batch_forward(x),
        "sr_fitness": lambda: arm.sr_fitness(x, labels),
        "forward": lambda: arm.forward(x_per_row),
    }
    for name in SURFACES:
        reference = references[name]()
        block = harness_validate.compare(reference, outputs[name])
        block["reference"] = "torch-batched"
        block["trees_compared"] = pop
        block["rows"] = rows if SURFACE_ROWS[name] is None else SURFACE_ROWS[name]
        numeric[name] = block

    if int(args.oracle_trees) <= 0:
        return numeric

    oracle_dir = os.path.join(common.src_root(), "evogp", "tests")
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    from reference_eval import (  # noqa: PLC0415  (the oracle; sliced + CPU only)
        batch_forward_from_rows,
        forward_from_rows,
        sr_fitness_from_rows,
    )

    # The oracle is a Python per-tree loop, so it is handed only the first K trees --
    # on CPU, which is where it computes.
    k = min(int(args.oracle_trees), pop)
    node_value = forest.batch_node_value[:k].detach().to("cpu")
    node_type = forest.batch_node_type[:k].detach().to("cpu")
    subtree_size = forest.batch_subtree_size[:k].detach().to("cpu")
    x_host = x.detach().to("cpu")
    labels_host = labels.detach().to("cpu")
    x_per_row_host = x_per_row[:k].detach().to("cpu")

    expected = {
        "batch_forward": batch_forward_from_rows(
            node_value, node_type, subtree_size, x_host, out_len=1
        ),
        "sr_fitness": sr_fitness_from_rows(
            node_value, node_type, subtree_size, x_host, labels_host, use_mse=True, out_len=1
        ),
        "forward": forward_from_rows(node_value, node_type, subtree_size, x_per_row_host, out_len=1),
    }
    for name in SURFACES:
        block = harness_validate.compare(expected[name], outputs[name][:k])
        block["reference"] = "reference_eval"
        block["trees_compared"] = k
        block["rows"] = rows if SURFACE_ROWS[name] is None else SURFACE_ROWS[name]
        numeric[name]["oracle"] = block
    return numeric


def stage_split_block(common, fused_kernels, fused_engine, forest, x, labels, cell: dict) -> dict:
    """Warm medians of facade / raw kernel / transpose (the ``fused_check`` recipe).

    This MIRRORS ``fused_check._stage_split`` on purpose: ``kernel_ms`` is the Triton
    launch alone (the facade's validation, memo lookup and output write excluded, so
    ``facade - kernel`` is exactly the host glue) and ``transpose_ms`` is the
    un-memoized ``x.t().contiguous()`` -- the ONE prep step of the batched path, which
    the identity memo makes a steady-state no-op and which a GP run still pays once per
    fresh dataset tensor.
    """
    node_value = forest.batch_node_value
    node_type = forest.batch_node_type
    subtree_size = forest.batch_subtree_size
    tree_len = int(node_value.shape[1])
    input_len = int(cell["input_len"])
    rows = int(cell["rows"])
    block_rows = fused_engine.resolve_block_rows()
    num_warps = fused_engine.resolve_num_warps()
    xt = x.t().contiguous()

    kernel_value = common.time_call(
        lambda: fused_kernels.launch_value(
            node_value,
            node_type,
            subtree_size,
            xt,
            input_len=input_len,
            tree_len=tree_len,
            rows=rows,
            per_row=False,
            block_rows=block_rows,
            num_warps=num_warps,
        ),
        warmup=STAGE_WARMUP,
        reps=STAGE_REPS,
    )
    kernel_loss = common.time_call(
        lambda: fused_kernels.launch_loss(
            node_value,
            node_type,
            subtree_size,
            xt,
            labels.view(-1),
            input_len=input_len,
            tree_len=tree_len,
            rows=rows,
            use_mse=True,
            block_rows=block_rows,
            num_warps=num_warps,
        ),
        warmup=STAGE_WARMUP,
        reps=STAGE_REPS,
    )
    transpose = common.time_call(
        lambda: x.t().contiguous(), warmup=STAGE_WARMUP, reps=STAGE_REPS
    )
    facade_value = common.time_call(
        lambda: forest.batch_forward(x), warmup=STAGE_WARMUP, reps=STAGE_REPS
    )
    facade_loss = common.time_call(
        lambda: forest.SR_fitness(x, labels), warmup=STAGE_WARMUP, reps=STAGE_REPS
    )

    def _split(facade: dict, kernel: dict) -> dict:
        facade_ms = float(facade["median_ms"])
        kernel_ms = float(kernel["median_ms"])
        return {
            "facade_ms": _num(facade_ms),
            "kernel_ms": _num(kernel_ms),
            "host_ms": _num(facade_ms - kernel_ms),
            "kernel_share_pct": _num(100.0 * kernel_ms / facade_ms if facade_ms else 0.0, 2),
            "transpose_ms": _num(transpose["median_ms"]),
        }

    return {
        "warmup": STAGE_WARMUP,
        "reps": STAGE_REPS,
        "geometry": {
            "block_rows": int(block_rows),
            "num_warps": int(num_warps),
            "n_rb": int(
                fused_kernels.plan(
                    rows, per_row=False, block_rows=block_rows, num_warps=num_warps
                )["n_rb"]
            ),
        },
        "batch_forward": _split(facade_value, kernel_value),
        "sr_fitness": _split(facade_loss, kernel_loss),
        "note": (
            "kernel = the raw fused_kernels launch (median of "
            f"{STAGE_WARMUP} warmup + {STAGE_REPS} reps); host = the full facade call "
            "minus that kernel; transpose = the un-memoized x.t().contiguous()"
        ),
    }


def gp_step_block(common, harness_arms, harness_worker, fused_engine, workload, x, labels) -> dict:
    """One GP generation loop: eval in place, operator cost as the difference."""
    import torch

    timed = harness_worker.TimedFitness(lambda forest, report=None: forest.SR_fitness(x, labels))
    pipeline = harness_arms.make_pipeline(workload, timed)
    for _ in range(GP_WARMUP):
        pipeline.step()
    torch.cuda.synchronize()

    steps = []
    for _ in range(GP_REPS):
        before = _snapshot(fused_engine)
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipeline.step()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) * 1e3
        after = _snapshot(fused_engine)
        steps.append(
            {
                "total_ms": _num(total_ms),
                "eval_ms": _num(timed.last_ms),
                "ops_ms": _num(total_ms - timed.last_ms),
                "fused": _delta(before, after),
            }
        )

    def _total(key: str) -> int:
        return int(sum(int(step["fused"].get(key, 0)) for step in steps))

    return {
        "warmup": GP_WARMUP,
        "reps": GP_REPS,
        "steps": steps,
        "median_total_ms": _num(_median([step["total_ms"] for step in steps])),
        "median_eval_ms": _num(_median([step["eval_ms"] for step in steps])),
        "median_ops_ms": _num(_median([step["ops_ms"] for step in steps])),
        "best_fitness": _num(float(pipeline.best_fitness)),
        # A fresh population is a new STRUCTURE set, so a route that re-planned would
        # show compiles here; the fused route compiles per (PER_ROW, REDUCE, USE_MSE,
        # block, warps) specialization, i.e. once per process shape, and its transpose
        # table keeps hitting on the unchanged dataset tensor.
        "per_generation": {
            "kernel_launches_per_step": [
                int(step["fused"].get("fused_kernel_launches", 0)) for step in steps
            ],
            "kernel_compiles_total": _total("fused_kernel_compiles"),
            "xt_hits_total": _total("fused_xt_hits"),
            "xt_misses_total": _total("fused_xt_misses"),
            "scratch_allocs_total": _total("fused_scratch_allocs"),
            "scratch_reuses_total": _total("fused_scratch_reuses"),
            "declines_total": _total("fused_declines"),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The CLI (stdlib only: ``--help`` must work with no device and no harness)."""
    parser = argparse.ArgumentParser(
        prog="python test/benchmark/evoxc/results/t1a1_fused/fused_probe.py",
        description=(
            "One fused-route evidence record for ONE cell: route policy, served-call "
            "proof, numeric parity (torch-batched + CPU oracle), kernel/host/transpose "
            "split, cold first calls and (with --gp) a generation loop."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EVOGP_EVAL_MODE=fused and EVOGP_EVOXC_BACKEND=iree are read from the "
            "environment and never set here; TRITON_CACHE_DIR must be writable."
        ),
    )
    parser.add_argument("--cell", required=True, help="cell id, e.g. p256-L32-i8-N1024")
    parser.add_argument("--out", required=True, help="JSON record path")
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls per surface")
    parser.add_argument("--reps", type=int, default=5, help="warm reps per surface")
    parser.add_argument("--seed", type=int, default=0, help="workload seed (population + data)")
    parser.add_argument(
        "--oracle-trees",
        type=int,
        default=32,
        help="trees handed to the CPU oracle (0 disables it; it is a Python per-tree loop)",
    )
    parser.add_argument(
        "--gp",
        action="store_true",
        help="also run one GP generation loop (1 warmup + 5 reps) through the pipeline",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="call each surface exactly once (no warmup/reps) and skip the stage split, "
             "so a fresh TRITON_CACHE_DIR yields the honest cold number",
    )
    parser.add_argument("--note", default="", help="free-form note stored in the record")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_paths()

    import torch  # noqa: PLC0415  (every heavy import waits for the CLI to be parsed)
    import common  # noqa: PLC0415
    import worker as harness_worker  # noqa: PLC0415
    import validate as harness_validate  # noqa: PLC0415
    import arms as harness_arms  # noqa: PLC0415
    import arms_torch  # noqa: PLC0415
    from evogp.tree import dense_engine, fused_engine, fused_kernels  # noqa: PLC0415

    cell = common.parse_cell(args.cell)
    workload = common.build_workload(cell, seed=int(args.seed))
    forest = workload["forest"]
    x, labels, x_per_row = workload["x"], workload["labels"], workload["x_per_row"]

    record = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": cell["id"],
        "arm": ARM,
        "note": args.note,
        "probe": probe_block(torch),
        "protocol": {
            "one_process_per_cell": True,
            "sync_bracketed": True,
            "warmup": int(args.warmup),
            "reps": int(args.reps),
            "seed": int(args.seed),
            "oracle_trees": int(args.oracle_trees),
            "cold_cache": bool(args.cold_cache),
            "stage_split": {"warmup": STAGE_WARMUP, "reps": STAGE_REPS},
            "gp_step": {"warmup": GP_WARMUP, "reps": GP_REPS} if args.gp else None,
        },
        "route": route_block(dense_engine, cell),
        "structures": harness_worker.forest_stats(forest),
        "workloads": {},
        "numeric": {},
        "stage_split": None,
        "cold": {},
        "cold_cache": bool(args.cold_cache),
        "gp_step": None,
    }
    record["route_all_fused"] = all(
        entry["engine"] == "fused" for entry in record["route"].values()
    )
    if not record["route_all_fused"]:
        print(
            "WARNING: the route policy does not name the fused engine for every surface "
            f"({ {name: entry['engine'] for name, entry in record['route'].items()} }); "
            "export EVOGP_EVAL_MODE=fused (and EVOGP_EVOXC_BACKEND=iree) before running "
            "this probe -- the record below is what THIS environment measured",
            file=sys.stderr,
            flush=True,
        )

    calls = {
        "batch_forward": lambda: forest.batch_forward(x),
        "sr_fitness": lambda: forest.SR_fitness(x, labels),
        "forward": lambda: forest.forward(x_per_row),
    }
    blocks, outputs = workload_block(args, common, fused_engine, calls)
    record["workloads"] = blocks
    record["cold"] = {name: blocks[name].get("first_call_ms") for name in SURFACES}

    arm = arms_torch.make_torch_arm("batched", forest)
    record["numeric"] = numeric_block(
        args, common, harness_validate, arm, outputs, forest, x, labels, x_per_row
    )

    if args.cold_cache:
        record["stage_split"] = None
    else:
        record["stage_split"] = stage_split_block(
            common, fused_kernels, fused_engine, forest, x, labels, cell
        )

    if args.gp:
        record["gp_step"] = gp_step_block(
            common, harness_arms, harness_worker, fused_engine, workload, x, labels
        )

    out_path = os.path.abspath(args.out)
    common.write_json(out_path, record)
    print(
        f"{cell['id']:<26} arm={ARM} "
        + " ".join(
            f"{name}={blocks[name]['median_ms'] if blocks[name]['median_ms'] is not None else blocks[name]['first_call_ms']}"
            f"ms(cold={blocks[name]['first_call_ms']}ms,served={blocks[name]['served']})"
            for name in SURFACES
        ),
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    return 0 if all(blocks[name]["served"] for name in SURFACES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
