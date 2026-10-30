"""Envelope, degeneracy, min-pop and geometry measurements for the FUSED (Triton) route.

``__main__``-guarded dev CLI (not pytest, imported by nothing).  The route under test is
the whole-population Triton evaluator :mod:`evogp.tree.fused_engine`, driven through the
PUBLIC API (``Forest.batch_forward`` / ``Forest.SR_fitness`` / ``Forest.forward``) so a
number includes the route decision and the facade -- exactly the calls a user makes.

STAGES (every stage writes a JSON record; ``--json PATH`` overrides the file)
---------------------------------------------------------------------------
``envelope``    one record per cell of :data:`ENVELOPE_CELLS`: fused value/loss medians vs
                the harness's ``torch-batched`` arm (the ratio denominator), the
                ``specialized`` fallback cost, the ``dense`` cost, and what makes a fused
                number believable -- ``scratch_bytes`` (``pop * tree_len * rows * 4``)
                against ``EVOGP_FUSED_MAX_SCRATCH_MB``, the resolved ``n_rb``, the peak
                device memory of the fused window, the ``fused_*`` counter delta (one
                launch per call, so the kernel ran instead of falling back) and the
                deviation of the output that was just timed.
``degenerate``  the same over hand-built shapes a generator does not produce: ``rows_1``, a
                row count that is not a multiple of ``BLOCK_R``, ``input_len 1``, an
                all-constant population and a single-node population; parity against
                :mod:`reference_eval` (the independent oracle) as well as the arm.
``minpop``      pop 1..128 at the shipped geometry: the kernel's own cost next to
                ``specialized`` and the torch arm, plus what the policy does
                (``EVOGP_FUSED_MIN_POP``) -- whether the threshold leaves performance on
                the table below it.
``sweep``       the ``EVOGP_FUSED_BLOCK_ROWS`` x ``EVOGP_FUSED_NUM_WARPS`` grid on
                :data:`SWEEP_CELLS`, one process per (config, cell), with a ``worst``
                column (that config's slowest cell relative to the cell's best config).
``headline``    the six cells of ``FUSED.md`` through the public API with the SHIPPED
                geometry (value + loss + per-row), written under ``fused_results/main/``.

PROTOCOL (the contract a reader can rely on)
--------------------------------------------
* ONE process per (cell, config) for every GPU measurement, spawned with
  ``subprocess.run([sys.executable, "-m", ...], env=<same env + the per-config knobs>)``
  and strictly sequential: never two configurations in one process, never two at once.
  The per-config knobs (``EVOGP_FUSED_BLOCK_ROWS`` / ``EVOGP_FUSED_NUM_WARPS`` /
  ``EVOGP_EVAL_MODE``) are set in the CHILD environment, so Triton's JIT cache and the
  engine's tables belong to one geometry.  Within a cell the ARMS (``fused`` /
  ``specialized`` / ``dense``) run sequentially in one process, because an arm switch is a
  route-policy flip of ``EVOGP_EVAL_MODE`` (read per call), not a second geometry.
* Warm medians: 2 warmup + 5 sync-bracketed reps of ``common.time_call``; the cold first
  call is recorded separately and never enters the median.
* Every worker prints ONE ``RESULT {...json...}`` line per record and the drivers parse it.
  A worker that fails records ``{"status": "error", "error": ...}`` and the driver
  continues -- but exits non-zero, so a failure is never silent.  A record whose fused
  calls were not served (``served: false``) keeps its numbers and also fails.
* ``fused.served_via`` says which surface produced the fused number: ``public_api`` when
  the route policy serves fused for this cell, ``direct`` when ``pop_size <
  EVOGP_FUSED_MIN_POP`` (the policy answers ``specialized`` there, so timing the public
  API would report the FALLBACK under the fused label; the kernel is asked through
  ``evaluate_fused`` / ``sr_fitness_fused`` instead and the record carries the policy's
  own answer under ``fused.policy``).
* Every record republishes the resolved geometry (``EVOGP_FUSED_*``, ``EVOGP_EVAL_MODE``,
  backend) so a reader can prove which configuration produced the number.

HOW TO RUN -- from the repository root, with ``TRITON_CACHE_DIR`` writable (the default
``~/.triton`` is not, in the sandbox)::

    TRITON_CACHE_DIR=$TMPDIR/fused/tc \\
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \\
    /tmp/rocvenv/bin/python -m evogp.tree.fused_envelope --stage envelope
    ... --stage degenerate | minpop | sweep | headline

``--help`` imports nothing heavy: torch, ``evogp`` and the harness are imported inside the
worker functions, so the CLI is inspectable on a box with no device at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The module lives in <repo>/src/evogp/tree, so src/ and the test oracle are two levels
# up.  Appending (not prepending) keeps `PYTHONPATH=src` authoritative for evogp itself.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(os.path.dirname(_HERE), "tests")
_RESULTS_ROOT = os.path.join(_HERE, "fused_results")
if _TESTS_DIR not in sys.path:
    sys.path.append(_TESTS_DIR)

SCHEMA = "evogp-fused-envelope-v1"          # record schema tag of every written file
WARMUP = 2                                  # untimed warm calls before the reps
REPS = 5                                    # sync-bracketed samples of the median
DEFAULT_OUT_DIR = os.path.join(_RESULTS_ROOT, "envelope")   # default ``--out-dir``
#: The route knobs every record republishes, so a number is attributable to a geometry.
_ROUTE_ENV_NAMES = (
    "EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_FUSED_BLOCK_ROWS",
    "EVOGP_FUSED_NUM_WARPS", "EVOGP_FUSED_MAX_SCRATCH_MB", "EVOGP_FUSED_MIN_POP",
    "EVOGP_FUSED_HOST_MEMO", "EVOGP_DENSE_MAX_TREE_LEN", "EVOGP_EVOXC_ARTIFACT_DIR",
    "TRITON_CACHE_DIR",
)

# The cell columns: each entry is EXACTLY the positional argument tuple of the harness's
# ``common.cell(pop, tree_len, input_len, rows[, max_layer_cnt])``.  The envelope column
# spans population, row and node-count scale; the sweep's cells are one population-scale,
# one row-scale and one long-tree cell.
ENVELOPE_CELLS: List[Tuple[int, ...]] = [
    (16, 64, 32, 1024), (64, 64, 32, 1024), (256, 64, 32, 1024), (1000, 64, 32, 1024),
    (256, 64, 32, 4096), (256, 64, 32, 16384), (1000, 64, 32, 4096),
    (2048, 64, 32, 1024), (4096, 64, 32, 1024), (8192, 64, 32, 1024),
    (2048, 128, 32, 1024, 7), (2048, 256, 32, 1024, 8),
]
SWEEP_CELLS: List[Tuple[int, ...]] = [
    (256, 64, 32, 1024), (4096, 64, 32, 1024), (256, 64, 32, 4096), (2048, 128, 32, 1024, 7),
]
SWEEP_BLOCKS: Tuple[int, ...] = (128, 256, 512, 1024)
SWEEP_WARPS: Tuple[int, ...] = (1, 2, 4)
#: The six headline cells of ``FUSED.md`` (same ids, so a record is addressable).
HEADLINE_CELLS: List[Tuple[int, ...]] = [
    (256, 32, 8, 1024), (256, 64, 32, 1024), (256, 64, 32, 4096),
    (1000, 64, 32, 1024), (4096, 64, 32, 1024), (2048, 128, 32, 1024, 7),
]
#: The min-pop column: pop 1..128 at a fixed shape (``L=64``, ``i32``, ``N=1024``).
MINPOP_POPS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
MINPOP_CELLS: List[Tuple[int, ...]] = [(pop, 64, 32, 1024) for pop in MINPOP_POPS]
#: The degenerate cases: label -> cell args, population builder, why the shape matters.
DEGENERATE_CASES: Dict[str, Dict[str, Any]] = {
    "rows_1": dict(cell=(256, 64, 32, 1), population="generated",
                   note="one data row: a single row lane, so n_rb is 1 and the launch is "
                        "pure per-tree work"),
    "rows_1000": dict(cell=(256, 64, 32, 1000), population="generated",
                      note="1000 rows is not a multiple of the 256-lane block: n_rb 4 with "
                           "a partial last block, i.e. the masked tail of the row axis"),
    "input_len_1": dict(cell=(256, 64, 1, 1024), population="generated",
                        note="input_len 1: every VAR node reads the same single input row"),
    "all_const_pop": dict(cell=(256, 64, 32, 1024), population="all_const",
                          note="Forest.zero_generate: every tree is the single CONST 0 "
                               "node, so the walk never reads an input row and every tree "
                               "is structurally identical"),
    "single_node_pop": dict(cell=(256, 64, 32, 1024), population="single_node",
                            note="single-node CONST trees with constants spread over "
                                 "[-1, 1), so the population is not degenerate in VALUE "
                                 "(differs from all_const_pop)"),
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _json_line(payload: Dict[str, Any]) -> None:
    """One machine-readable line for the driver stages to parse."""
    print("RESULT " + json.dumps(payload, default=str), flush=True)


def _harness():
    """The harness ``common`` module (cells, workloads, ``time_call``)."""
    try:
        import common  # type: ignore
    except Exception as exc:  # noqa: BLE001 - an environment error, never a silent pass
        raise SystemExit(f"cannot import the harness (common): {exc!r}\n  add "
                         "test/benchmark/evoxc to PYTHONPATH")
    return common


def _comparator():
    """The harness's ``validate.compare`` (its deviation report)."""
    try:
        from validate import compare  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot import validate.compare: {exc!r}\n  add "
                         "test/benchmark/evoxc to PYTHONPATH")
    return compare


def _cell_id(cell_args: Sequence[int]) -> str:
    """The harness's cell id, computed locally so a driver needs no harness (see _cell_of)."""
    pop, tree_len, input_len, rows = (int(value) for value in cell_args[:4])
    depth = int(cell_args[4]) if len(cell_args) > 4 else 5
    return f"p{pop}-L{tree_len}-i{input_len}-N{rows}" + ("" if depth == 5 else f"-d{depth}")


def _cell_of(cell_args: Sequence[int]) -> Dict[str, Any]:
    """Build one harness cell and prove the locally computed id still matches it."""
    cell = _harness().cell(*cell_args)
    if cell["id"] != _cell_id(cell_args):
        raise SystemExit(f"cell id drift: {_cell_id(cell_args)!r} vs {cell['id']!r}")
    return cell


def _cell_args(cells: Sequence[Tuple[int, ...]], cell_id: str) -> Tuple[int, ...]:
    for cell_args in cells:
        if _cell_id(cell_args) == cell_id:
            return cell_args
    raise SystemExit(f"unknown cell {cell_id!r}; known: {[_cell_id(c) for c in cells]}")


def _select(args: argparse.Namespace, cells: Sequence[Tuple[int, ...]]) -> List[str]:
    """The cell ids of one column, restricted to ``--only`` when it was given."""
    return [cell_id for cell_id in map(_cell_id, cells)
            if not args.only or cell_id in args.only]


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """``numerator / denominator``, or ``None`` when a side is missing/zero (a null)."""
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _ms(value: Optional[float]) -> str:
    return "     n/a" if value is None else f"{float(value):8.3f}"


def _set_mode(mode: Optional[str]) -> None:
    """Pin ``EVOGP_EVAL_MODE`` for the next calls (every route knob is read per call)."""
    if mode is None:
        os.environ.pop("EVOGP_EVAL_MODE", None)
    else:
        os.environ["EVOGP_EVAL_MODE"] = str(mode)


def _stage_out(args: argparse.Namespace, sub: str, name: str) -> str:
    """The default record path of one stage (``--out-dir`` wins; ``--json`` still wins)."""
    base = _RESULTS_ROOT if args.out_dir == DEFAULT_OUT_DIR else args.out_dir
    return os.path.join(base, *([sub] if sub else []), name)


def _write_json(path: str, payload: Any) -> None:
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(f"{out}.tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    os.replace(f"{out}.tmp", out)
    print(f"wrote {out}", flush=True)


def _spawn(extra: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], int, str]:
    """Run one worker leg in its OWN process (sequential) and parse its RESULT line."""
    argv = [sys.executable, "-m", "evogp.tree.fused_envelope", *extra]
    proc = subprocess.run(argv, capture_output=True, text=True, env=dict(os.environ))
    payload: Optional[Dict[str, Any]] = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
    tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
    return payload, int(proc.returncode), tail


def _run_children(
    pairs: Sequence[Tuple[str, Sequence[str]]]
) -> Tuple[List[Dict[str, Any]], int]:
    """Spawn one child per ``(label, argv)`` pair, sequentially; never raise.

    A child that produced no ``RESULT`` line, reported ``status: error`` or exited
    non-zero (which includes a record whose fused calls were not served) counts as a
    failure -- the record is kept either way and the driver keeps going.
    """
    records: List[Dict[str, Any]] = []
    failures = 0
    for label, extra in pairs:
        payload, code, tail = _spawn(extra)
        if payload is None:
            payload = {"schema": SCHEMA, "status": "error", "kind": label,
                       "error": tail or f"exit {code} without a RESULT line"}
        if payload.get("status") == "error" or code != 0:
            payload.setdefault("status", "not-served")
            payload["exit_code"] = code
            failures += 1
        else:
            payload["status"] = "ok"
        records.append(payload)
    return records, failures


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], notes: Sequence[str] = ()):
    """Print a pipe-delimited table of pre-formatted cells plus footnote lines."""
    print("| " + " | ".join(head) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    for note in notes:
        print("  " + note)


def _drive(args: argparse.Namespace, worker: str, children: Sequence[Tuple[str, Sequence[str]]],
           head: Sequence[str], row: Callable[[Dict[str, Any]], Sequence[str]], out_path: str,
           *, title: str, notes: Sequence[str] = (), **extra: Any) -> int:
    """Run one child per entry, print its table (``row`` per record) and write the file."""
    print("=" * 116)
    print(title)
    print("=" * 116)
    if not children:
        print("nothing selected")
        return 1
    records, failures = _run_children(
        [(label, ["--stage", worker, *argv]) for label, argv in children])
    print()
    _table(head, [row(record) for record in records], notes)
    _write_json(args.json or out_path,
                {"schema": SCHEMA, "kind": worker.rsplit("-", 1)[0],
                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "protocol": {"warmup": WARMUP, "reps": REPS, "one_process_per_cell": True},
                 "failures": failures, "records": records, **extra})
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# environment / route / geometry blocks
# --------------------------------------------------------------------------- #
def _resolved_env(fused_engine, dense_engine, *, eval_mode: Optional[str]) -> Dict[str, Any]:
    """The geometry a record was produced under (resolved values + the raw environment)."""
    return {
        "requested_eval_mode": eval_mode,
        "raw": {name: os.environ.get(name) for name in _ROUTE_ENV_NAMES},
        "resolved": {
            "block_rows_cap": int(fused_engine.resolve_block_rows()),
            "num_warps": int(fused_engine.resolve_num_warps()),
            "max_scratch_mb": int(fused_engine.resolve_max_scratch_mb()),
            "min_pop": int(fused_engine.resolve_min_pop()),
            "evoxc_backend": dense_engine.resolve_backend_name(None, None),
        },
        "fused_backend": "torch device (cuda/rocm); the fused route builds no evoxc graph",
        "fused_available": bool(fused_engine.fused_available()),
    }


def _route(dense_engine, cell: Dict[str, Any], *, eval_mode: Optional[str]) -> Dict[str, Any]:
    """``resolve_route`` for one call shape: the policy's own answer, not an inference."""
    route = dense_engine.resolve_route(int(cell["pop"]), rows=int(cell["rows"]),
                                       tree_len=int(cell["tree_len"]), eval_mode=eval_mode)
    return {"eval_mode": eval_mode, "engine": route.engine,
            "use_fused": bool(route.use_fused), "use_dense": bool(route.use_dense),
            "backend": route.backend, "fallback_backend": route.fallback_backend,
            "reason": route.reason}


def _routes(dense_engine, cell: Dict[str, Any]) -> Dict[str, Any]:
    """The inherited default policy next to an explicit ``fused`` request."""
    return {"inherited": _route(dense_engine, cell, eval_mode=None),
            "fused_requested": _route(dense_engine, cell, eval_mode="fused")}


def _dense_cap(dense_engine) -> int:
    """The base dense trace cap (``EVOGP_DENSE_MAX_TREE_LEN``), read the engine's way.

    ``dense_engine._resolve_max_tree_len`` is private, so the two-line rule is repeated
    here: unset/empty means :data:`dense_engine.DEFAULT_MAX_TREE_LEN`.
    """
    raw = os.environ.get(dense_engine.MAX_TREE_LEN_ENV, "").strip()
    return int(raw) if raw else int(dense_engine.DEFAULT_MAX_TREE_LEN)


def _snapshot(fused_engine) -> Dict[str, int]:
    """One ``fused_cache_stats()`` snapshot as plain ints."""
    return {key: int(value) for key, value in fused_engine.fused_cache_stats().items()}


def _delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    """Per-key counter delta of two snapshots."""
    return {key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in sorted(set(before) | set(after))}


def _launches(delta: Dict[str, int]) -> Tuple[int, str]:
    """The route's launch count over a window, and which counter it came from.

    ``fused_kernel_launches`` if the facade reports one; otherwise ``fused_calls``,
    which counts ATTEMPTS -- every attempt that is not a decline launches the kernel,
    so an attempt count with ``fused_declines == 0`` is the same proof.
    """
    for key in ("fused_kernel_launches", "fused_calls"):
        if key in delta:
            return int(delta[key]), key
    return 0, "missing"


def _geometry(common, fused_kernels, fused_engine, cell: Dict[str, Any]) -> Dict[str, Any]:
    """Scratch envelope, resolved launch geometry and the memory rule of one cell."""
    cell_bytes = int(common.cell_tensor_bytes(cell))
    budget_mb = int(fused_engine.resolve_max_scratch_mb())
    launch = fused_kernels.plan(int(cell["rows"]), per_row=False,
                                block_rows=int(fused_engine.resolve_block_rows()),
                                num_warps=int(fused_engine.resolve_num_warps()))
    return {"scratch_bytes": cell_bytes, "scratch_mib": cell_bytes / 2**20,
            "max_scratch_mb": budget_mb, "envelope_ok": cell_bytes <= budget_mb * 2**20,
            "block_rows_resolved": int(launch["block_rows"]),
            "num_warps_resolved": int(launch["num_warps"]), "n_rb": int(launch["n_rb"])}


def _header(kind: str, cell: Dict[str, Any], fused_engine, dense_engine, *,
            eval_mode: Optional[str], note: Optional[str] = None) -> Dict[str, Any]:
    """The identity + protocol + geometry header every record starts with."""
    env = _resolved_env(fused_engine, dense_engine, eval_mode=eval_mode)
    resolved = env["resolved"]
    print(f"ENV {cell['id']}: mode={eval_mode} B={resolved['block_rows_cap']} "
          f"W={resolved['num_warps']} scratch_mb={resolved['max_scratch_mb']} "
          f"min_pop={resolved['min_pop']} evoxc={resolved['evoxc_backend']} "
          f"fused_available={env['fused_available']}", flush=True)
    return {"schema": SCHEMA, "kind": kind,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cell": cell["id"], "pop": int(cell["pop"]), "rows": int(cell["rows"]),
            "tree_len": int(cell["tree_len"]), "input_len": int(cell["input_len"]),
            "max_layer_cnt": int(cell["max_layer_cnt"]),
            "protocol": {"one_process_per_cell": True, "arms_sequential_in_process": True,
                         "sync_bracketed": True, "warmup": WARMUP, "reps": REPS},
            "env": env, "note": note}


# --------------------------------------------------------------------------- #
# the measurement blocks (shared by every worker)
# --------------------------------------------------------------------------- #
def _timed(common, fn: Callable[[], Any]) -> Tuple[Dict[str, Any], Any]:
    """One warm-median measurement (the harness protocol) + the timed call's output."""
    record = common.time_call(fn, warmup=WARMUP, reps=REPS)
    return common.public_timing(record), record.get("_out")


def _time_calls(common, calls: Dict[str, Callable[[], Any]]):
    """Time every named callable in order; returns ``(timings, outputs)``."""
    timings: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    for name, fn in calls.items():
        timings[name], outputs[name] = _timed(common, fn)
    return timings, outputs


def _public_block(common, mode: Optional[str], calls: Dict[str, Callable[[], Any]]):
    """Time the public surfaces with ``EVOGP_EVAL_MODE`` pinned to ``mode``."""
    _set_mode(mode)
    return _time_calls(common, calls)


def _fused_block(common, torch, fused_engine, calls: Dict[str, Callable[[], Any]]):
    """Time the fused route in ONE window: medians, served proof, memory peak."""
    _set_mode("fused")
    before = _snapshot(fused_engine)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)
    timings, outputs = _time_calls(common, calls)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20)
    delta = _delta(before, _snapshot(fused_engine))
    launches, counter = _launches(delta)
    expected = len(calls) * (1 + WARMUP + REPS)
    return {
        "timings": timings, "outputs": outputs, "launches": launches,
        "launch_counter": counter, "expected_launches": expected, "stats_delta": delta,
        "served": launches == expected and int(delta.get("fused_declines", 0)) == 0,
        "memory": {"baseline_allocated_mib": base[0], "baseline_reserved_mib": base[1],
                   "peak_allocated_mib": peak[0], "peak_reserved_mib": peak[1],
                   "delta_allocated_mib": peak[0] - base[0],
                   "delta_reserved_mib": peak[1] - base[1]},
    }


def _fused_direct(common, fused_engine, forest, x, labels, input_len: int, *,
                  per_row_x=None, torch=None):
    """The facade's kernel cost WITHOUT the route policy (served at any ``pop >= 1``).

    What ``minpop`` and ``sweep`` need: below ``EVOGP_FUSED_MIN_POP`` the public API would
    answer ``specialized``, so the kernel's own cost is only visible through
    ``evaluate_fused`` / ``sr_fitness_fused``.  ``per_row_x`` adds the per-row value
    surface (one input row per tree), which is the only per-row entry point the facade
    has.  Returns ``(record_fragment, outputs)``.
    """
    _set_mode("fused")
    nv, nt, ss = (forest.batch_node_value, forest.batch_node_type,
                  forest.batch_subtree_size)
    calls: Dict[str, Callable[[], Any]] = {
        "value": lambda: fused_engine.evaluate_fused(nv, nt, ss, input_len, 1, x,
                                                     per_row=False),
        "loss": lambda: fused_engine.sr_fitness_fused(nv, nt, ss, input_len, 1, x, labels,
                                                      use_MSE=True)}
    if per_row_x is not None:
        calls["per_row"] = lambda: fused_engine.evaluate_fused(
            nv, nt, ss, input_len, 1, per_row_x, per_row=True)
    before = _snapshot(fused_engine)
    if torch is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = (torch.cuda.memory_allocated() / 2**20,
                torch.cuda.memory_reserved() / 2**20)
    timings, outputs = _time_calls(common, calls)
    memory: Dict[str, float] = {}
    if torch is not None:
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() / 2**20,
                torch.cuda.max_memory_reserved() / 2**20)
        memory = {"baseline_allocated_mib": base[0], "baseline_reserved_mib": base[1],
                  "peak_allocated_mib": peak[0], "peak_reserved_mib": peak[1],
                  "delta_allocated_mib": peak[0] - base[0],
                  "delta_reserved_mib": peak[1] - base[1]}
    delta = _delta(before, _snapshot(fused_engine))
    launches, counter = _launches(delta)
    expected = len(calls) * (1 + WARMUP + REPS)
    return {
        "timings": timings, "launches": launches, "launch_counter": counter,
        "expected_launches": expected, "stats_delta": delta, "memory": memory,
        "served": launches == expected and int(delta.get("fused_declines", 0)) == 0,
        "note": "evaluate_fused / sr_fitness_fused: the kernel without the route policy",
    }, outputs


def _arm(forest):
    """The ``torch-batched`` reference arm (whole-population plain-torch implementation)."""
    from arms_torch import make_torch_arm  # type: ignore

    return make_torch_arm("batched", forest)


def _arm_calls(arm, x, labels, per_row_x=None) -> Dict[str, Callable[[], Any]]:
    """The torch arm's surfaces (the denominator of every ratio)."""
    calls: Dict[str, Callable[[], Any]] = {"value": lambda: arm.batch_forward(x),
                                          "loss": lambda: arm.sr_fitness(x, labels)}
    if per_row_x is not None:
        calls["per_row"] = lambda: arm.forward(per_row_x)
    return calls


def _dense_block(common, dense_engine, forest, x, labels, cell, *, reference):
    """The ``dense`` arm, or a recorded ``dense_expected_decline``.

    An explicit ``EVOGP_EVAL_MODE=dense`` gets the BASE trace cap (no ``auto`` raise), so a
    cell whose row stride exceeds it declines every tree into ``specialized`` and its
    timing would be the fallback's -- it is marked instead of measured.
    """
    cap = _dense_cap(dense_engine)
    if int(cell["tree_len"]) > cap:
        return {"max_tree_len": cap, "status": "skipped", "served": False,
                "dense_expected_decline": True,
                "note": f"tree_len {int(cell['tree_len'])} > "
                        f"{dense_engine.MAX_TREE_LEN_ENV}={cap}: an explicit dense call "
                        "declines into specialized, so timing it would report the "
                        "fallback, not dense"}
    before = {key: int(value) for key, value in dense_engine.dense_cache_stats().items()}
    timings, outputs = _public_block(common, "dense", {
        "value": lambda: forest.batch_forward(x),
        "loss": lambda: forest.SR_fitness(x, labels)})
    delta = _delta(before, dense_engine.dense_cache_stats())
    # A DenseUnsupported fallback is invisible through the public API; the dense front
    # tables only move when the route really planned/executed a graph -- the one
    # observable proof of service this CLI has.
    moves = int(delta.get("dense_plan_misses", 0)) + int(delta.get("dense_exec_misses", 0))
    return {"max_tree_len": cap, "status": "served" if moves else "unproven",
            "served": moves > 0, "dense_expected_decline": False, "timings": timings,
            "stats_delta": delta, "parity_value_vs_reference": reference(outputs["value"]),
            "note": "served is read from the dense front tables (a plan/exec miss); a "
                    "DenseUnsupported fallback is invisible through the public API, and "
                    "EVOGP_DENSE_HOST_MEMO=0 disables those tables"}


def _fused_arm(common, torch, fused_engine, dense_engine, forest, x, labels, cell, calls,
               per_row_x=None):
    """The fused measurement of one cell: the public API when the policy serves fused.

    ``EVOGP_EVAL_MODE=fused`` still answers ``specialized`` below
    ``EVOGP_FUSED_MIN_POP`` (that is the POLICY, not a decline), so timing the public API
    there would put the per-tree fallback under the ``fused`` label.  The kernel is asked
    directly instead and ``served_via`` says which surface produced the number.
    """
    route = _route(dense_engine, cell, eval_mode="fused")
    if route["engine"] == "fused":
        block = _fused_block(common, torch, fused_engine, calls)
        block["served_via"] = "public_api"
        return block, block["outputs"]
    block, outputs = _fused_direct(common, fused_engine, forest, x, labels,
                                   int(cell["input_len"]), per_row_x=per_row_x, torch=torch)
    block["served_via"] = "direct"
    block["policy"] = route
    return block, outputs


def _measure_public(common, torch, fused_engine, dense_engine, forest, x, labels, cell,
                    *, per_row_x=None, fallback: bool = False, dense_arm: bool = False,
                    direct: bool = False):
    """Every public-API arm of one cell: fused, torch-batched, specialized, dense.

    ``direct`` adds the policy-free kernel block (``fused_direct``) next to the public
    one, so a number can be attributed to the facade/route layer rather than to the
    kernel.  Returns ``(block, outputs)``: the JSON-able record fragment and the result
    tensors a caller may compare further (device tensors, never serialized).
    """
    calls: Dict[str, Callable[[], Any]] = {"value": lambda: forest.batch_forward(x),
                                          "loss": lambda: forest.SR_fitness(x, labels)}
    if per_row_x is not None:
        calls["per_row"] = lambda: forest.forward(per_row_x)
    fused, fused_outputs = _fused_arm(common, torch, fused_engine, dense_engine, forest,
                                      x, labels, cell, calls, per_row_x)
    arm = _arm(forest)
    arm_timings, arm_outputs = _time_calls(common,
                                           _arm_calls(arm, x, labels, per_row_x))
    compare = _comparator()
    block: Dict[str, Any] = {
        "fused": {key: fused[key] for key in ("timings", "served", "served_via",
                                             "launches", "launch_counter",
                                             "expected_launches", "stats_delta",
                                             "memory")},
        "torch_batched": {"timings": arm_timings},
        "parity": {f"{name}_vs_torch_batched":
                   compare(arm_outputs[name], fused_outputs[name])
                   for name in fused_outputs},
    }
    if "policy" in fused:
        block["fused"]["policy"] = fused["policy"]
    if direct:
        direct_block, direct_outputs = _fused_direct(
            common, fused_engine, forest, x, labels, int(cell["input_len"]),
            per_row_x=per_row_x, torch=torch)
        block["fused_direct"] = direct_block
        block["parity"]["fused_direct_vs_fused"] = {
            name: compare(fused_outputs[name], direct_outputs[name])
            for name in direct_outputs if name in fused_outputs}
    if fallback:
        spec_timings, spec_outputs = _public_block(
            common, "specialized", {"value": calls["value"], "loss": calls["loss"]})
        block["specialized"] = {
            "timings": spec_timings,
            "parity_value_vs_torch_batched": compare(arm_outputs["value"],
                                                     spec_outputs["value"])}
    if dense_arm:
        block["dense"] = _dense_block(common, dense_engine, forest, x, labels, cell,
                                      reference=lambda out: compare(arm_outputs["value"], out))
    return block, {"fused": fused_outputs, "torch_batched": arm_outputs}


def _ratios(record: Dict[str, Any], base: str = "fused",
            arms: Sequence[str] = ("torch_batched", "specialized", "dense"),
            ) -> Dict[str, Optional[float]]:
    """Per-surface speedup of ``base`` over each arm: ``arm_ms / base_ms`` (>1 wins).

    ``base`` is the arm under test (``fused`` for a public-API worker, ``fused_direct``
    for the policy-free ones), so one key always reads "fused is N x cheaper/faster".
    """
    denom = (record.get(base) or {}).get("timings") or {}
    out: Dict[str, Optional[float]] = {}
    for name in arms:
        for surface, entry in ((record.get(name) or {}).get("timings") or {}).items():
            if surface in denom:
                out[f"{surface}_vs_{name}"] = _ratio(entry["median_ms"],
                                                     denom[surface]["median_ms"])
    return out


def _surface_ms(record: Dict[str, Any], arm: str, surface: str = "value") -> Optional[float]:
    """``record[arm]["timings"][surface]["median_ms"]`` or ``None`` when absent."""
    entry = (record.get(arm, {}).get("timings") or {}).get(surface)
    return None if entry is None else entry["median_ms"]


def _degenerate_forest(name: str, cell: Dict[str, Any], forest, torch):
    """The hand-built populations (``generated`` keeps the cell's own generated one)."""
    from evogp.tree import Forest

    pop, tree_len, input_len = (int(cell[key]) for key in ("pop", "tree_len", "input_len"))
    if name == "all_const_pop":
        return Forest.zero_generate(pop, tree_len, input_len, 1)
    if name == "single_node_pop":
        built = Forest.zero_generate(pop, tree_len, input_len, 1)
        built.batch_node_value[:, 0] = torch.linspace(
            -1.0, 1.0, pop, dtype=torch.float32, device=built.batch_node_value.device)
        return built
    return forest


def _public_worker(args: argparse.Namespace, kind: str, cells: Sequence[Tuple[int, ...]], *,
                   per_row: bool = False, fallback: bool = False, dense_arm: bool = False,
                   forest_of: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
                   note: Optional[str] = None, cell_id: Optional[str] = None,
                   direct: bool = False):
    """Shared body of the public-API workers (envelope / headline / degenerate).

    ``cell_id`` overrides the ``--cell`` lookup, so a worker whose ``--cell`` argument is
    a DEGENERATE LABEL (``all_const_pop``) can still name the harness cell it measures.
    """
    import torch

    common = _harness()
    from evogp.tree import dense_engine, fused_engine, fused_kernels

    cell = _cell_of(_cell_args(cells, cell_id or args.cell))
    workload = common.build_workload(cell)
    forest = (forest_of(workload, cell) if forest_of else workload["forest"])
    record = _header(kind, cell, fused_engine, dense_engine, eval_mode="fused", note=note)
    record["geometry"] = _geometry(common, fused_kernels, fused_engine, cell)
    record["route"] = _routes(dense_engine, cell)
    block, outputs = _measure_public(
        common, torch, fused_engine, dense_engine, forest, workload["x"],
        workload["labels"], cell, per_row_x=workload["x_per_row"] if per_row else None,
        fallback=fallback, dense_arm=dense_arm, direct=direct)
    record.update(block)
    record["ratios"] = _ratios(record)
    return record, outputs, forest, cell, workload


def _direct_worker(args: argparse.Namespace, kind: str, cells: Sequence[Tuple[int, ...]], *,
                   note: Optional[str] = None):
    """Shared body of the direct-engine workers (minpop / sweep): kernel + torch arm."""
    import torch

    common = _harness()
    from evogp.tree import dense_engine, fused_engine, fused_kernels

    cell = _cell_of(_cell_args(cells, args.cell))
    workload = common.build_workload(cell)
    forest, x, labels = workload["forest"], workload["x"], workload["labels"]
    record = _header(kind, cell, fused_engine, dense_engine, eval_mode="fused", note=note)
    record["geometry"] = _geometry(common, fused_kernels, fused_engine, cell)
    record["route"] = _routes(dense_engine, cell)
    record["fused_direct"], direct_out = _fused_direct(
        common, fused_engine, forest, x, labels, int(cell["input_len"]), torch=torch)
    arm = _arm(forest)
    arm_timings, arm_outputs = _time_calls(common, _arm_calls(arm, x, labels))
    record["torch_batched"] = {"timings": arm_timings}
    compare = _comparator()
    record["parity"] = {
        "value_vs_torch_batched": compare(arm_outputs["value"], direct_out["value"]),
        "loss_vs_torch_batched": compare(arm_outputs["loss"], direct_out["loss"])}
    record["ratios"] = _ratios(record, "fused_direct", ("torch_batched",))
    return record, arm_timings, cell, workload


# --------------------------------------------------------------------------- #
# stage 1: the envelope column
# --------------------------------------------------------------------------- #
def stage_envelope_one(args: argparse.Namespace) -> int:
    """One envelope cell: fused / torch-batched / specialized / dense + the envelope."""
    record, _, _, _, _ = _public_worker(
        args, "envelope", ENVELOPE_CELLS,
        fallback=not args.skip_fallback, dense_arm=not args.skip_fallback)
    geometry = record["geometry"]
    print(f"{record['cell']:<28} fused={_ms(_surface_ms(record, 'fused'))} "
          f"loss={_ms(_surface_ms(record, 'fused', 'loss'))} "
          f"torch={_ms(_surface_ms(record, 'torch_batched'))} "
          f"({record['ratios'].get('value_vs_torch_batched') or float('nan'):.2f}x) "
          f"scratch={geometry['scratch_mib']:.1f}MiB ok={geometry['envelope_ok']} "
          f"n_rb={geometry['n_rb']} served={record['fused']['served']} "
          f"via={record['fused'].get('served_via')}", flush=True)
    _json_line(record)
    return 0 if record["fused"]["served"] and geometry["envelope_ok"] else 1


def stage_envelope(args: argparse.Namespace) -> int:
    """Driver: one subprocess per envelope cell, then the envelope table."""
    passthrough = ["--skip-fallback"] if args.skip_fallback else []
    return _drive(
        args, "envelope-one",
        [(cell_id, ["--cell", cell_id, *passthrough])
         for cell_id in _select(args, ENVELOPE_CELLS)],
        ["cell", "fused", "loss", "torch", "ratio", "spec", "dense", "scr", "ok", "n_rb",
         "via", "served"],
        lambda rec: [rec.get("kind", "?"), f"ERROR: {rec['error'][:70]}"]
        if rec.get("status") == "error" else [
            f"{rec['cell']:<26}", _ms(_surface_ms(rec, "fused")),
            _ms(_surface_ms(rec, "fused", "loss")), _ms(_surface_ms(rec, "torch_batched")),
            f"{rec['ratios'].get('value_vs_torch_batched') or float('nan'):>6.2f}",
            _ms(_surface_ms(rec, "specialized")), _ms(_surface_ms(rec, "dense")),
            f"{rec['geometry']['scratch_mib']:>5.0f}MiB",
            f"{str(rec['geometry']['envelope_ok'])[:5]:>5}",
            f"{rec['geometry']['n_rb']:>4}",
            f"{str(rec['fused'].get('served_via'))[:9]:>9}",
            f"{str(rec['fused']['served'])[:6]:>6}"],
        _stage_out(args, "", "envelope.json"),
        title="ENVELOPE: fused vs torch-batched / specialized / dense "
              f"(warm medians, {WARMUP} warmup + {REPS} reps, one process per cell)",
        notes=["'spec'/'dense' n/a = skipped (see the record: dense_expected_decline / "
               "--skip-fallback).",
               "'ok' = the cell's scratch (pop*tree_len*rows*4) fits "
               "EVOGP_FUSED_MAX_SCRATCH_MB; 'served' = one kernel launch per call.",
               "'via' = public_api when the route policy serves fused, direct when "
               "pop_size < EVOGP_FUSED_MIN_POP (the policy answers specialized, so the "
               "kernel is asked through evaluate_fused instead)."])


# --------------------------------------------------------------------------- #
# stage 2: degenerate populations / data
# --------------------------------------------------------------------------- #
def stage_degenerate_one(args: argparse.Namespace) -> int:
    """One degenerate case: fused value + loss + per-row, torch arm, oracle parity."""
    import torch

    import reference_eval as R  # type: ignore

    label = args.cell
    if label not in DEGENERATE_CASES:
        raise SystemExit(f"unknown degenerate case {label!r}; known: "
                         f"{sorted(DEGENERATE_CASES)}")
    case = DEGENERATE_CASES[label]
    record, outputs, forest, cell, workload = _public_worker(
        args, "degenerate", [case["cell"]], per_row=True, note=case["note"],
        cell_id=_cell_id(case["cell"]),
        forest_of=lambda wl, built: _degenerate_forest(case["population"], built,
                                                       wl["forest"], torch))
    record.update({"degenerate_case": label, "population": case["population"],
                   "reference": "torch-batched arm and src/evogp/tests/reference_eval.py "
                                "(imported as reference_eval; an independent oracle fed "
                                "the population rows directly, so it shares neither the "
                                "engine's decode nor the kernel)"})
    # The oracle: rows straight in, so it is independent of the engine's decode as well as
    # of the kernel -- the parity a fused number is judged on.  ``reference_eval`` is
    # CPU-only (its constants are CPU tensors), so it is fed CPU COPIES of the input
    # planes; ``validate.compare`` moves the device results itself.
    x, labels, x_per_row = (workload[name].detach().cpu() for name in
                            ("x", "labels", "x_per_row"))
    nv, nt, ss = (forest.batch_node_value, forest.batch_node_type,
                  forest.batch_subtree_size)
    compare = _comparator()
    record["fused"]["parity"] = {
        **record["parity"],
        "value_vs_reference_eval": compare(R.batch_forward_from_rows(nv, nt, ss, x),
                                           outputs["fused"]["value"]),
        "loss_vs_reference_eval": compare(R.sr_fitness_from_rows(nv, nt, ss, x, labels),
                                          outputs["fused"]["loss"]),
        "per_row_vs_reference_eval": compare(R.forward_from_rows(nv, nt, ss, x_per_row),
                                             outputs["fused"]["per_row"])}
    print(f"{label:<16} {cell['id']:<26} fused={_ms(_surface_ms(record, 'fused'))} "
          f"loss={_ms(_surface_ms(record, 'fused', 'loss'))} "
          f"torch={_ms(_surface_ms(record, 'torch_batched'))} "
          f"served={record['fused']['served']} oracle_max_rel="
          f"{record['fused']['parity']['value_vs_reference_eval'].get('max_rel', 0.0):.2e}",
          flush=True)
    _json_line(record)
    return 0 if record["fused"]["served"] else 1


def stage_degenerate(args: argparse.Namespace) -> int:
    """Driver: one subprocess per degenerate case."""
    return _drive(
        args, "degenerate-one",
        [(label, ["--cell", label]) for label in DEGENERATE_CASES
         if not args.only or label in args.only],
        ["case", "cell", "fused", "loss", "torch", "served", "oracle rel"],
        lambda rec: [rec.get("kind", "?"), f"ERROR: {rec['error'][:90]}"]
        if rec.get("status") == "error" else [
            f"{rec['degenerate_case']:<15}", f"{rec['cell']:<26}",
            _ms(_surface_ms(rec, "fused")), _ms(_surface_ms(rec, "fused", "loss")),
            _ms(_surface_ms(rec, "torch_batched")),
            f"{str(rec['fused']['served'])[:5]:>5}",
            f"{rec['fused']['parity']['value_vs_reference_eval'].get('max_rel', 0.0):.2e}"],
        _stage_out(args, "", "degenerate.json"),
        title="DEGENERATE: hand-built populations/data "
              f"(warmup {WARMUP}, reps {REPS}, one process per case)",
        notes=["parity is stated against BOTH the torch-batched arm and the "
               "reference_eval oracle."])


# --------------------------------------------------------------------------- #
# stage 3: the min-pop column
# --------------------------------------------------------------------------- #
def stage_minpop_one(args: argparse.Namespace) -> int:
    """One population size: the kernel's own cost, the public route's answer, the policy."""
    common = _harness()
    from evogp.tree import fused_engine

    record, arm_timings, cell, workload = _direct_worker(args, "minpop", MINPOP_CELLS)
    forest, x, labels = workload["forest"], workload["x"], workload["labels"]
    calls = {"value": lambda: forest.batch_forward(x),
             "loss": lambda: forest.SR_fitness(x, labels)}
    record["fused_api"] = {
        "timings": _public_block(common, "fused", calls)[0],
        "note": "Forest.batch_forward / SR_fitness with EVOGP_EVAL_MODE=fused: whatever "
                "the route policy serves at this pop_size"}
    record["specialized"] = {"timings": _public_block(common, "specialized", calls)[0]}
    min_pop = int(fused_engine.resolve_min_pop())
    direct_ms = _surface_ms(record, "fused_direct")
    record["min_pop_gate"] = {
        "min_pop": min_pop, "below_min_pop": int(cell["pop"]) < min_pop,
        "policy_engine": record["route"]["fused_requested"]["engine"],
        "fused_vs_specialized": _ratio(_surface_ms(record, "specialized"), direct_ms),
        "api_vs_direct": _ratio(_surface_ms(record, "fused_api"), direct_ms),
        "speedup_vs_torch": _ratio(arm_timings["value"]["median_ms"], direct_ms),
        "note": "fused_direct is the kernel at this pop_size; below EVOGP_FUSED_MIN_POP "
                "the policy answers specialized (fused_api == specialized), so a "
                "fused_vs_specialized > 1 below the gate is performance the shipped "
                "threshold does not collect"}
    print(f"p{record['pop']:<6} direct={_ms(direct_ms)} "
          f"spec={_ms(_surface_ms(record, 'specialized'))} "
          f"torch={_ms(arm_timings['value']['median_ms'])} "
          f"fused/spec={record['min_pop_gate']['fused_vs_specialized'] or float('nan'):.2f}x "
          f"served={record['fused_direct']['served']}", flush=True)
    _json_line(record)
    return 0 if record["fused_direct"]["served"] else 1


def stage_minpop(args: argparse.Namespace) -> int:
    """Driver: one subprocess per population size of the min-pop column."""
    return _drive(
        args, "minpop-one",
        [(cell_id, ["--cell", cell_id]) for cell_id in _select(args, MINPOP_CELLS)],
        ["pop", "fused direct", "fused api", "spec", "torch", "fused/spec", "vs torch",
         "engine"],
        lambda rec: ["?", f"ERROR: {rec['error'][:70]}"]
        if rec.get("status") == "error" else [
            f"{rec['pop']:>6}", _ms(_surface_ms(rec, "fused_direct")),
            _ms(_surface_ms(rec, "fused_api")), _ms(_surface_ms(rec, "specialized")),
            _ms(_surface_ms(rec, "torch_batched")),
            f"{rec['min_pop_gate']['fused_vs_specialized'] or float('nan'):>10.2f}",
            f"{rec['min_pop_gate']['speedup_vs_torch'] or float('nan'):>8.2f}",
            f"{rec['min_pop_gate']['policy_engine']:>12}"],
        _stage_out(args, "", "minpop.json"),
        title=f"MINPOP: pop 1..128 at L=64/i32/N=1024 (warmup {WARMUP}, reps {REPS})",
        notes=["'fused api' is what the policy serves with EVOGP_EVAL_MODE=fused; below "
               "EVOGP_FUSED_MIN_POP it is the specialized route."])


# --------------------------------------------------------------------------- #
# stage 4: the BLOCK_ROWS x NUM_WARPS sweep
# --------------------------------------------------------------------------- #
def stage_sweep_one(args: argparse.Namespace) -> int:
    """One (config, cell) in this process; the two knobs arrive through the environment."""
    os.environ["EVOGP_FUSED_BLOCK_ROWS"] = str(int(args.block))
    os.environ["EVOGP_FUSED_NUM_WARPS"] = str(int(args.warps))
    record, _, _, _ = _direct_worker(
        args, "sweep", SWEEP_CELLS,
        note=f"BLOCK_ROWS={int(args.block)} NUM_WARPS={int(args.warps)}")
    record["config"] = {"block_rows": int(args.block), "num_warps": int(args.warps)}
    print(f"B={int(args.block):<5} W={int(args.warps):<2} {record['cell']:<26} "
          f"value={_ms(_surface_ms(record, 'fused_direct'))} "
          f"loss={_ms(_surface_ms(record, 'fused_direct', 'loss'))} "
          f"n_rb={record['geometry']['n_rb']} "
          f"served={record['fused_direct']['served']}", flush=True)
    _json_line(record)
    return 0 if record["fused_direct"]["served"] else 1


def stage_sweep(args: argparse.Namespace) -> int:
    """Driver: one subprocess per (config, cell), then the minimax table."""

    def value_of(record: Dict[str, Any]) -> Optional[float]:
        """This record's fused value median, or None for a failed config."""
        return None if record.get("status") == "error" else _surface_ms(record,
                                                                       "fused_direct")

    cell_ids = _select(args, SWEEP_CELLS)
    children = [
        (f"B{block}-W{warps}-{cell_id}",
         ["--cell", cell_id, "--block", str(block), "--warps", str(warps)])
        for block in SWEEP_BLOCKS for warps in SWEEP_WARPS for cell_id in cell_ids]
    print("=" * 116)
    print(f"SWEEP: BLOCK_ROWS x NUM_WARPS over {len(cell_ids)} cells "
          f"(warmup {WARMUP}, reps {REPS}, one process per (config, cell))")
    print("=" * 116)
    if not children:
        print("nothing selected")
        return 1
    records, failures = _run_children(
        [(label, ["--stage", "sweep-one", *argv]) for label, argv in children])
    best: Dict[str, Optional[float]] = {}
    for cell_id in cell_ids:
        values = [value for value in (value_of(rec) for rec in records
                                      if rec.get("cell") == cell_id) if value is not None]
        best[cell_id] = min(values) if values else None
    rows: List[Dict[str, Any]] = []
    table: List[List[str]] = []
    for block in SWEEP_BLOCKS:
        for warps in SWEEP_WARPS:
            per_cell: Dict[str, float] = {}
            worst = 0.0
            for rec in records:
                if rec.get("status") == "error" or rec.get("config") != {
                        "block_rows": block, "num_warps": warps}:
                    continue
                value = value_of(rec)
                if value is None:
                    continue
                per_cell[rec["cell"]] = float(value)
                if best.get(rec["cell"]):
                    worst = max(worst, float(value) / float(best[rec["cell"]]))
            rows.append({"cfg": f"B{block}-W{warps}", "worst": worst})
            table.append([f"B{block}-W{warps:<8}"]
                         + [f"{_ms(per_cell.get(c)):>27}" for c in cell_ids]
                         + [f"{worst:>6.2f}x"])
    minimax = min(rows, key=lambda item: item["worst"])
    print()
    print("VALUE path (ms), warm median; 'worst' = this config's slowest cell relative to "
          "that cell's best config")
    _table(["cfg"] + cell_ids + ["worst"], table, notes=[
        f"minimax config: {minimax['cfg']} at {minimax['worst']:.2f}x",
        f"{failures} failing config(s)" if failures else "every config served"])
    _write_json(args.json or _stage_out(args, "sweep", "sweep.json"),
                {"schema": SCHEMA, "kind": "sweep",
                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "protocol": {"warmup": WARMUP, "reps": REPS,
                              "one_process_per_config_and_cell": True,
                              "blocks": list(SWEEP_BLOCKS), "warps": list(SWEEP_WARPS)},
                 "best_value_ms_per_cell": best,
                 "config_worst_ratio": {r["cfg"]: r["worst"] for r in rows},
                 "failures": failures, "records": records})
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# stage 5: the headline cells of FUSED.md
# --------------------------------------------------------------------------- #
def stage_headline_one(args: argparse.Namespace) -> int:
    """One headline cell through the public API with the SHIPPED geometry.

    ``direct=True`` adds the policy-free kernel block (``fused_direct``) beside the
    public call, so the record can separate the facade/route layer from the kernel.
    """
    record, _, _, _, _ = _public_worker(args, "headline", HEADLINE_CELLS, per_row=True,
                                        direct=True)
    geometry = record["geometry"]
    print(f"{record['cell']:<28} value={_ms(_surface_ms(record, 'fused'))} "
          f"loss={_ms(_surface_ms(record, 'fused', 'loss'))} "
          f"per_row={_ms(_surface_ms(record, 'fused', 'per_row'))} "
          f"kernel={_ms(_surface_ms(record, 'fused_direct'))} "
          f"({record['ratios'].get('value_vs_torch_batched') or float('nan'):.2f}x torch) "
          f"B={geometry['block_rows_resolved']} W={geometry['num_warps_resolved']} "
          f"served={record['fused']['served']}", flush=True)
    _json_line(record)
    return 0 if record["fused"]["served"] else 1


def stage_headline(args: argparse.Namespace) -> int:
    """Driver: one subprocess per headline cell, one timestamped record file."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return _drive(
        args, "headline-one",
        [(cell_id, ["--cell", cell_id]) for cell_id in _select(args, HEADLINE_CELLS)],
        ["cell", "value", "loss", "per_row", "kernel", "torch", "ratio", "B", "W",
         "served"],
        lambda rec: ["?", f"ERROR: {rec['error'][:70]}"]
        if rec.get("status") == "error" else [
            f"{rec['cell']:<26}", _ms(_surface_ms(rec, "fused")),
            _ms(_surface_ms(rec, "fused", "loss")),
            _ms(_surface_ms(rec, "fused", "per_row")),
            _ms(_surface_ms(rec, "fused_direct")),
            _ms(_surface_ms(rec, "torch_batched")),
            f"{rec['ratios'].get('value_vs_torch_batched') or float('nan'):>6.2f}",
            f"{rec['geometry']['block_rows_resolved']:>4}",
            f"{rec['geometry']['num_warps_resolved']:>2}",
            f"{str(rec['fused']['served'])[:6]:>6}"],
        _stage_out(args, "main", f"headline_{stamp}.json"),
        title="HEADLINE: the six cells of FUSED.md through the public API, shipped "
              f"geometry (warmup {WARMUP}, reps {REPS}); 'kernel' = fused_direct, the "
              "facade without the route policy",
        arm="evoxc-fused-iree")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
#: Stages that measure exactly ONE cell and therefore require ``--cell``.
_ONE_STAGES = ("envelope-one", "degenerate-one", "minpop-one", "sweep-one", "headline-one")

_DRIVERS = {"envelope": stage_envelope, "degenerate": stage_degenerate,
            "minpop": stage_minpop, "sweep": stage_sweep, "headline": stage_headline}
_WORKERS = {"envelope-one": stage_envelope_one, "degenerate-one": stage_degenerate_one,
            "minpop-one": stage_minpop_one, "sweep-one": stage_sweep_one,
            "headline-one": stage_headline_one}
STAGES = (*_DRIVERS, "all", *_ONE_STAGES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evogp.tree.fused_envelope",
        description="Envelope / degenerate / min-pop / geometry measurements of the "
                    "fused (Triton) evaluation route.")
    parser.add_argument("--stage", default="envelope", choices=STAGES,
                        help="which leg to run (default: envelope; the -one spellings "
                             "are the single-process workers the drivers spawn)")
    parser.add_argument("--cell", default=None,
                        help="cell id of a single-process stage (e.g. "
                             "p256-L64-i32-N1024, or a degenerate label such as "
                             "all_const_pop)")
    parser.add_argument("--block", type=int, default=256,
                        help="EVOGP_FUSED_BLOCK_ROWS (sweep-one)")
    parser.add_argument("--warps", type=int, default=1,
                        help="EVOGP_FUSED_NUM_WARPS (sweep-one)")
    parser.add_argument("--only", default=None,
                        help="comma-separated cell ids (or degenerate labels) to "
                             "restrict a driver to")
    parser.add_argument("--skip-fallback", action="store_true",
                        help="skip the specialized/dense arms of --stage envelope (the "
                             "specialized leg runs one evoxc graph per tree, so it is "
                             "the slow one on a large population)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="directory of the envelope/degenerate/minpop record files "
                             f"(default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--json", default=None, help="write the stage's records here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.only:
        args.only = [item.strip() for item in args.only.split(",") if item.strip()]
    if args.stage in _ONE_STAGES and not args.cell:
        parser.error(f"--stage {args.stage} needs --cell")
    status = 0
    if args.stage == "all":
        if args.json:
            parser.error("--json and --stage all are mutually exclusive: each stage "
                         "writes its own record file")
        for driver in _DRIVERS.values():
            status |= driver(args)
    elif args.stage in _DRIVERS:
        status |= _DRIVERS[args.stage](args)
    else:
        status |= _WORKERS[args.stage](args)
    print(f"\nfused_envelope: exit {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
