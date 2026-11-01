"""The fused route's timing and tuning stages (``--stage timing`` / ``--stage tuning``).

This module is the ECONOMICS half of the battery in :mod:`evogp.fused.check`; it is
imported by ``check.py`` and driven through ``python -m evogp.fused.check``.

WHAT IT MEASURES
----------------
1. **``--stage timing``** (driver + ``--stage time-one`` child).  Six cells, warm
   medians (2 warmup + 10 sync-bracketed reps) of ``evaluate_fused``,
   ``sr_fitness_fused`` and the per-tree value path, the cold first call (Triton
   import + JIT, or a cache LOAD), a stage split (kernel-only launch vs the rest of
   the call, plus the un-memoized transpose) and the SERVED PROOF (counters: a
   number from a decline-free path, not a fallback).  One process per cell, because
   a warm median is only comparable if every process sees one shape class.
2. **``--stage tuning``** (driver + ``--stage tune-one`` child).
   ``EVOGP_FUSED_BLOCK_ROWS`` in {64,128,256,512} x ``EVOGP_FUSED_NUM_WARPS`` in
   {1,2,4} on the same cells, ranked by each config's WORST per-cell ratio against
   the SHIPPED default (256 rows, 1 warp) measured in the same sweep: the route
   ships one default, so the minimax answer is the one that is never much dearer
   than what ships.

ABSOLUTE MILLISECONDS ONLY
--------------------------
The ``torch-batched`` reference arm lived in the benchmark harness
(``test/benchmark/evoxc``), which this package deliberately does not import, so
neither stage prints a speedup column: compare against a number you measured
yourself, on the same box, in the same session.  The tuning ranking normalises each
cell by the shipped default from the SAME sweep, which is an in-process comparison
and stays valid.

The child legs print one ``RESULT {...json...}`` line each; that is the whole
driver/child protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from . import engine as fused_engine
from . import kernels as fused_kernels
from .cases import time_call, workload

# --------------------------------------------------------------------------- #
# the cells (shared by both drivers and their single-process children)
# --------------------------------------------------------------------------- #
#: The six timing cells (the shapes the route's records use).  The ids are the
#: campaign's own spellings, so a number printed here is addressable by name.
CELL_SPECS: List[Dict[str, Any]] = [
    dict(pop=256, tree_len=32, input_len=8, rows=1024, max_layer_cnt=5),
    dict(pop=256, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=256, tree_len=64, input_len=32, rows=4096, max_layer_cnt=5),
    dict(pop=1000, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=4096, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=2048, tree_len=128, input_len=32, rows=1024, max_layer_cnt=7),
]

#: The shipped geometry, i.e. the baseline the tuning stage ranks against.
SHIPPED_BLOCK_ROWS = fused_kernels.DEFAULT_BLOCK_ROWS
SHIPPED_NUM_WARPS = fused_kernels.DEFAULT_NUM_WARPS


def _cell_id(spec: Dict[str, Any]) -> str:
    """``p<pop>-L<tree_len>-i<input_len>-N<rows>[-d<depth>]`` (``-d`` only off 5)."""
    cell = (
        f"p{int(spec['pop'])}-L{int(spec['tree_len'])}-i{int(spec['input_len'])}"
        f"-N{int(spec['rows'])}"
    )
    depth = int(spec["max_layer_cnt"])
    return cell if depth == 5 else f"{cell}-d{depth}"


def _cells() -> List[Dict[str, Any]]:
    """The six timing cells, each with its ``id`` resolved."""
    return [dict(spec, id=_cell_id(spec)) for spec in CELL_SPECS]


def _cell_by_id(cell_id: str) -> Dict[str, Any]:
    for cell in _cells():
        if cell["id"] == cell_id:
            return cell
    raise SystemExit(f"unknown cell id {cell_id!r}; known: {[c['id'] for c in _cells()]}")



def _json_line(payload: Dict[str, Any]) -> None:
    """One machine-readable line for the driver stages to parse."""
    print("RESULT " + json.dumps(payload, default=str), flush=True)
# --------------------------------------------------------------------------- #
# stage 3: timings (driver + one-cell child)
# --------------------------------------------------------------------------- #
def _stage_split(
    node_value, node_type, subtree_size, x, xt, labels, *, input_len, rows, block_rows, num_warps
):
    """Kernel-only and transpose-only timings for the value and the loss pass.

    ``kernel_ms`` is the Triton launch alone (the facade's validation, memo lookup
    and ``contiguous()`` are excluded, so ``total - kernel`` is exactly the host
    glue), and ``transpose_ms`` is the un-memoized ``x.t().contiguous()`` -- the ONE
    prep step of the batched path, which the identity memo makes a steady-state
    no-op and which a GP run still pays once per fresh dataset tensor.
    """
    tree_len = int(node_value.shape[1])
    kernel = time_call(
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
        warmup=2,
        reps=10,
    )
    loss_kernel = time_call(
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
        warmup=2,
        reps=10,
    )
    transpose = time_call(lambda: x.t().contiguous(), warmup=2, reps=10)
    return {
        "kernel_ms": kernel["median_ms"],
        "loss_kernel_ms": loss_kernel["median_ms"],
        "transpose_ms": transpose["median_ms"],
    }


def stage_time_one(args: argparse.Namespace) -> int:
    """One cell in this process: warm medians, cold first call, stage split.

    ABSOLUTE milliseconds only: the ``torch-batched`` reference arm lived in the
    benchmark harness this package does not reach into, so the payload carries the
    fused numbers, their cold first call and the kernel/host split, plus the proof
    that the timed calls were SERVED (counters, no decline) rather than fallbacks.
    """
    import torch

    cell = _cell_by_id(args.cell)
    built = workload(cell)
    node_value, node_type, subtree_size = built["rows"]
    x, labels, x_per_row = built["x"], built["labels"], built["x_per_row"]
    input_len = int(cell["input_len"])
    rows = int(cell["rows"])
    pop = int(cell["pop"])
    block_rows = fused_engine.resolve_block_rows()
    num_warps = fused_engine.resolve_num_warps()

    import_ms = 0.0
    if "triton" not in sys.modules:
        torch.cuda.synchronize()
        start = __import__("time").perf_counter()
        import triton  # noqa: F401
        import_ms = (__import__("time").perf_counter() - start) * 1e3

    before = fused_engine.fused_cache_stats()
    value = time_call(
        lambda: fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, x, per_row=False
        ),
        warmup=2,
        reps=10,
    )
    loss = time_call(
        lambda: fused_engine.sr_fitness_fused(
            node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=True
        ),
        warmup=2,
        reps=10,
    )
    per_row = time_call(
        lambda: fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, x_per_row, per_row=True
        ),
        warmup=2,
        reps=10,
    )
    after = fused_engine.fused_cache_stats()
    xt = x.t().contiguous()
    split = _stage_split(
        node_value,
        node_type,
        subtree_size,
        x,
        xt,
        labels,
        input_len=input_len,
        rows=rows,
        block_rows=block_rows,
        num_warps=num_warps,
    )
    stats = fused_engine.fused_cache_stats()
    payload = {
        "kind": "timing",
        "cell": cell["id"],
        "pop": pop,
        "rows": rows,
        "tree_len": int(cell["tree_len"]),
        "block_rows": int(block_rows),
        "num_warps": int(num_warps),
        "n_rb": int(
            fused_kernels.plan(
                rows, per_row=False, block_rows=block_rows, num_warps=num_warps
            )["n_rb"]
        ),
        "fused_value_ms": value["median_ms"],
        "fused_loss_ms": loss["median_ms"],
        "fused_per_row_ms": per_row["median_ms"],
        "fused_value_first_ms": value["first_call_ms"],
        "fused_loss_first_ms": loss["first_call_ms"],
        "fused_per_row_first_ms": per_row["first_call_ms"],
        "import_ms": import_ms,
        "kernel_ms": split["kernel_ms"],
        "loss_kernel_ms": split["loss_kernel_ms"],
        "transpose_ms": split["transpose_ms"],
        "served": dict(
            calls=int(after["fused_calls"] - before["fused_calls"]),
            declines=int(after["fused_declines"] - before["fused_declines"]),
            launches=int(after["fused_kernel_launches"] - before["fused_kernel_launches"]),
        ),
        "stats": stats,
    }
    print(
        f"{cell['id']:<28} value={value['median_ms']:.3f} ms "
        f"loss={loss['median_ms']:.3f} ms per_row={per_row['median_ms']:.3f} ms "
        f"cold(value)={value['first_call_ms']:.1f} ms "
        f"served={payload['served']['launches']} launches / "
        f"{payload['served']['declines']} declines",
        flush=True,
    )
    _json_line(payload)
    return 0


def _spawn(extra: List[str]) -> Optional[Dict[str, Any]]:
    """Run one child leg of this battery in its OWN process and parse its RESULT."""
    argv = [sys.executable, "-m", "evogp.fused.check"] + extra
    proc = subprocess.run(argv, capture_output=True, text=True, env=dict(os.environ))
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT ") :])
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        print(f"  [FAIL] {' '.join(extra)} (exit {proc.returncode}): " + " | ".join(tail))
    return payload


def stage_timing(args: argparse.Namespace) -> int:
    """Driver: one subprocess per cell, then the absolute-millisecond table."""
    print("=" * 100)
    print("TIMING: the fused route in ABSOLUTE milliseconds (warm medians, 2 warmup +")
    print("        10 sync-bracketed reps; one process per cell)")
    print("  NO RATIO COLUMN: the torch-batched reference arm lived in the benchmark")
    print("  harness (test/benchmark/evoxc), which this package does not import.  Compare")
    print("  against a number you measured yourself on this box, in this session.")
    print("=" * 100)
    records = []
    for cell in _cells():
        if args.only and cell["id"] not in args.only:
            continue
        payload = _spawn(["--stage", "time-one", "--cell", cell["id"]])
        if payload:
            records.append(payload)
    if not records:
        print("no cells measured")
        return 1
    print()
    header = (
        f"| {'cell':<26} | {'value ms':>9} | {'loss ms':>9} | {'per-row ms':>11} | "
        f"{'cold ms':>8} | {'B':>4} | {'W':>2} | {'n_rb':>4} |"
    )
    rule = (
        "|" + "-" * 28 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 13 + "|"
        + "-" * 10 + "|" + "-" * 6 + "|" + "-" * 4 + "|" + "-" * 6 + "|"
    )
    print(header)
    print(rule)
    for rec in records:
        print(
            f"| {rec['cell']:<26} | {rec['fused_value_ms']:>6.3f} ms | "
            f"{rec['fused_loss_ms']:>6.3f} ms | {rec['fused_per_row_ms']:>8.3f} ms | "
            f"{rec['fused_value_first_ms']:>6.1f} | {rec['block_rows']:>4} | "
            f"{rec['num_warps']:>2} | {rec['n_rb']:>4} |"
        )
    print()
    print("COLD FIRST CALL per surface (Triton import + JIT on an empty on-disk cache, or a")
    print("cache LOAD once the same (per-row, reduce, use_mse, block, warps) specialization")
    print("was compiled in an earlier run):")
    for rec in records:
        print(
            f"  {rec['cell']:<26} value={rec['fused_value_first_ms']:>9.1f} ms "
            f"loss={rec['fused_loss_first_ms']:>9.1f} ms "
            f"per_row={rec['fused_per_row_first_ms']:>9.1f} ms "
            f"(triton import {rec['import_ms']:.1f} ms)"
        )
    print()
    print("STAGE SPLIT (ms): kernel = the Triton launch alone, host = the rest of the call")
    print(
        "  NOTE: a kernel-only launch pays its own binding/prepare, so on a route whose")
    print(
        "  whole call is already cheaper than one independently timed launch the split")
    print(
        "  does not decompose it (the host column reads n/a there): only the TOTALS above")
    print("  are quotable.")
    print(
        f"  {'cell':<26} {'total':>8} {'kernel':>8} {'host':>7} {'transpose(cold)':>16} "
        f"{'loss total':>11} {'loss kernel':>12} {'loss host':>10}"
    )
    for rec in records:
        host = rec["fused_value_ms"] - rec["kernel_ms"]
        loss_host = rec["fused_loss_ms"] - rec["loss_kernel_ms"]
        host_text = f"{host:>7.3f}" if host > 0 else f"{'n/a':>7}"
        loss_host_text = f"{loss_host:>10.3f}" if loss_host > 0 else f"{'n/a':>10}"
        print(
            f"  {rec['cell']:<26} {rec['fused_value_ms']:>8.3f} {rec['kernel_ms']:>8.3f} "
            f"{host_text} {rec['transpose_ms']:>16.3f} {rec['fused_loss_ms']:>11.3f} "
            f"{rec['loss_kernel_ms']:>12.3f} {loss_host_text}"
        )
    print()
    print("SERVED PROOF (a number from a fallback would be worthless):")
    for rec in records:
        served = rec["served"]
        print(
            f"  {rec['cell']:<26} calls={served['calls']} launches={served['launches']} "
            f"declines={served['declines']} compiles={rec['stats']['fused_kernel_compiles']} "
            f"scratch_allocs={rec['stats']['fused_scratch_allocs']} "
            f"scratch_reuses={rec['stats']['fused_scratch_reuses']} "
            f"xt_hits={rec['stats']['fused_xt_hits']} "
            f"xt_misses={rec['stats']['fused_xt_misses']}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0 if all(rec["served"]["declines"] == 0 for rec in records) else 1


# --------------------------------------------------------------------------- #
# stage 4: the BLOCK_ROWS x NUM_WARPS sweep
# --------------------------------------------------------------------------- #
def stage_tune_one(args: argparse.Namespace) -> int:
    """One (config, cell) in this process; the knobs are read per call."""
    os.environ["EVOGP_FUSED_BLOCK_ROWS"] = str(int(args.block))
    os.environ["EVOGP_FUSED_NUM_WARPS"] = str(int(args.warps))
    cell = _cell_by_id(args.cell)
    built = workload(cell)
    node_value, node_type, subtree_size = built["rows"]
    x, labels = built["x"], built["labels"]
    input_len = int(cell["input_len"])
    value = time_call(
        lambda: fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, x, per_row=False
        ),
        warmup=2,
        reps=10,
    )
    loss = time_call(
        lambda: fused_engine.sr_fitness_fused(
            node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=True
        ),
        warmup=2,
        reps=10,
    )
    _json_line(
        {
            "kind": "tuning",
            "cell": cell["id"],
            "block_rows": int(args.block),
            "num_warps": int(args.warps),
            "fused_value_ms": value["median_ms"],
            "fused_loss_ms": loss["median_ms"],
            "cold_ms": value["first_call_ms"],
        }
    )
    return 0


def stage_tuning(args: argparse.Namespace) -> int:
    """Driver: one subprocess per (config, cell), then the minimax decision.

    Absolute milliseconds, ranked by the WORST per-cell ratio against the SHIPPED
    default (``BLOCK_ROWS`` 256, 1 warp) measured in the same sweep -- a route ships
    one default, so the question is which config is never much dearer than what
    ships, not which wins one cell.
    """
    print("=" * 100)
    print("TUNING: EVOGP_FUSED_BLOCK_ROWS x EVOGP_FUSED_NUM_WARPS (one process per config+cell)")
    print("  ABSOLUTE milliseconds; the ranking normalises each cell by the SHIPPED default")
    print(f"  ({SHIPPED_BLOCK_ROWS} rows, {SHIPPED_NUM_WARPS} warp) measured in this same sweep.")
    print("=" * 100)
    blocks = [int(b) for b in str(args.blocks).split(",") if b.strip()]
    warps = [int(w) for w in str(args.warps_list).split(",") if w.strip()]
    cells = [c for c in _cells() if not args.only or c["id"] in args.only]
    records: List[Dict[str, Any]] = []
    for block in blocks:
        for warp in warps:
            for cell in cells:
                payload = _spawn(
                    [
                        "--stage",
                        "tune-one",
                        "--cell",
                        cell["id"],
                        "--block",
                        str(block),
                        "--warps",
                        str(warp),
                    ]
                )
                if payload:
                    records.append(payload)
                    print(
                        f"  b{block:<4} w{warp} {cell['id']:<26} "
                        f"value={payload['fused_value_ms']:>8.3f} ms "
                        f"loss={payload['fused_loss_ms']:>8.3f} ms "
                        f"cold={payload['cold_ms']:>7.1f} ms",
                        flush=True,
                    )
    if not records:
        print("no configurations measured")
        return 1
    shipped_rows = [
        rec
        for rec in records
        if (rec["block_rows"], rec["num_warps"]) == (SHIPPED_BLOCK_ROWS, SHIPPED_NUM_WARPS)
    ]
    baseline = {
        rec["cell"]: rec["fused_value_ms"] for rec in shipped_rows
    }
    if not baseline:
        print(
            f"  (the shipped default {SHIPPED_BLOCK_ROWS}/{SHIPPED_NUM_WARPS} was not in the "
            "grid, so the ranking is by absolute ms only)"
        )
    by_config: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in records:
        by_config.setdefault((rec["block_rows"], rec["num_warps"]), []).append(rec)
    print()
    print(
        f"| {'BLOCK_R':>7} | {'warps':>5} | {'worst cell ms':>13} | {'mean cell ms':>12} | "
        f"{'worst vs shipped':>16} | {'worst cell':<26} |"
    )
    print(
        "|" + "-" * 9 + "|" + "-" * 7 + "|" + "-" * 15 + "|" + "-" * 14 + "|"
        + "-" * 18 + "|" + "-" * 28 + "|"
    )
    ranked = []
    for (block, warp), group in sorted(by_config.items()):
        costs = [(float(rec["fused_value_ms"]), rec["cell"]) for rec in group]
        worst_ms, worst_cell = max(costs, key=lambda item: item[0])
        ratios = [
            (float(rec["fused_value_ms"]) / baseline[rec["cell"]], rec["cell"])
            for rec in group
            if rec["cell"] in baseline and baseline[rec["cell"]]
        ]
        worst_ratio, ratio_cell = (max(ratios, key=lambda item: item[0]) if ratios else (1.0, "n/a"))
        ranked.append((worst_ratio, block, warp, group))
        print(
            f"| {block:>7} | {warp:>5} | {worst_ms:>10.3f} ms | "
            f"{sum(cost for cost, _ in costs) / len(costs):>9.3f} ms | "
            f"{worst_ratio:>16.3f} | {ratio_cell if ratios else worst_cell:<26} |"
        )
    ranked.sort()
    best_ratio, best_block, best_warp, _ = ranked[0]
    print()
    print(
        f"minimax choice (lowest WORST-cell ratio against the shipped default): "
        f"EVOGP_FUSED_BLOCK_ROWS={best_block} EVOGP_FUSED_NUM_WARPS={best_warp} "
        f"(worst {best_ratio:.3f}x)"
    )
    if baseline:
        shipped_worst = max(
            ratios[0][0]
            for ratios in (
                [
                    (float(rec["fused_value_ms"]) / baseline[rec["cell"]], rec["cell"])
                    for rec in shipped_rows
                    if rec["cell"] in baseline
                ],
            )
            if ratios
        )
        print(
            f"shipped default ({SHIPPED_BLOCK_ROWS} rows, {SHIPPED_NUM_WARPS} warp): "
            f"worst {shipped_worst:.3f}x the shipped default (by construction 1.000x)"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


