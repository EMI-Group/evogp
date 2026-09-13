#!/usr/bin/env python
"""``trace_analyze.py`` -- R5 Q2(d): the timing facts inside a kineto chrome trace.

The kineto attempt (``kineto_attempt.py``) captures the IREE-cuda kernels of the
compiled dense graph.  This script turns the (large) chrome trace into the Q2
numbers: per-invocation dispatch/memcpy counts, kernel durations, device-busy
time, the idle gaps between consecutive device ops, the launch-side
(``cuda_driver``) activity and the grid/occupancy census.

Method: read the trace (``.json`` or ``.json.gz``), keep the ``kernel`` /
``gpu_memcpy`` runtime events and the ``cuda_driver`` host events, then

* group the device events into invocations by splitting on the largest gaps
  (``--invocations`` N gives the number of blocks to cut; default: derive from
  the kernel count / the dump's dispatch count);
* report per-group span (first start -> last end), device-busy time
  (sum of durations), the idle gap total and its distribution, and the same for
  the launch-side events;
* census the grid/block shapes, "est. achieved occupancy" and shared-memory use
  of the kernels.

Pure stdlib: runs without torch/GPU, e.g.

    python3 test/benchmark/r5/gpu_probe/trace_analyze.py \
        --trace test/benchmark/r5/results/q2_gpu/kineto_f64/kineto_trace.json.gz \
        --invocations 30 --json-out test/benchmark/r5/results/q2_gpu/kineto_f64/trace_facts.json \
        --text-out test/benchmark/r5/results/q2_gpu/kineto_f64/trace_facts.txt
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import statistics
import sys


def load_events(path):
    """The ``traceEvents`` list of a chrome trace (plain ``.json`` or ``.json.gz``)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)["traceEvents"]


def spans(events):
    """(count, total_us, min/median/mean/max, list of durations) of X-phase events."""
    durs = [float(e.get("dur") or 0.0) for e in events if e.get("ph") == "X"]
    if not durs:
        return {"count": 0, "total_us": 0.0, "median_us": None, "mean_us": None,
                "min_us": None, "max_us": None, "durations": []}
    return {
        "count": len(durs),
        "total_us": sum(durs),
        "median_us": statistics.median(durs),
        "mean_us": statistics.fmean(durs),
        "min_us": min(durs),
        "max_us": max(durs),
        "durations": durs,
    }


def group_by_gaps(events, n_groups):
    """Split the time-ordered device events into ``n_groups`` runs at the largest gaps."""
    events = sorted(events, key=lambda e: float(e["ts"]))
    if n_groups <= 1 or len(events) < 2:
        return [events]
    gaps = []
    for prev, cur in zip(events, events[1:]):
        end = float(prev["ts"]) + float(prev.get("dur") or 0.0)
        gaps.append((float(cur["ts"]) - end, cur["ts"]))
    cuts = sorted(range(len(gaps)), key=lambda i: -gaps[i][0])[: n_groups - 1]
    bounds = sorted(cuts)
    groups, start = [], 0
    for cut in bounds:
        groups.append(events[start : cut + 1])
        start = cut + 1
    groups.append(events[start:])
    return groups


def stats_block(events):
    """Span / busy / gap statistics of one ordered event list."""
    if not events:
        return {}
    start = float(events[0]["ts"])
    end = max(float(e["ts"]) + float(e.get("dur") or 0.0) for e in events)
    busy = sum(float(e.get("dur") or 0.0) for e in events)
    gaps = []
    cursor = None
    for event in events:
        ev_start = float(event["ts"])
        ev_end = ev_start + float(event.get("dur") or 0.0)
        if cursor is not None and ev_start > cursor:
            gaps.append(ev_start - cursor)
        cursor = ev_end if cursor is None else max(cursor, ev_end)
    return {
        "n_events": len(events),
        "span_us": end - start,
        "busy_us": busy,
        "busy_pct": 100.0 * busy / (end - start) if end > start else None,
        "gap_total_us": sum(gaps),
        "gap_count": len(gaps),
        "gap_median_us": statistics.median(gaps) if gaps else 0.0,
        "gap_mean_us": statistics.fmean(gaps) if gaps else 0.0,
        "gap_max_us": max(gaps) if gaps else 0.0,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="R5 Q2(d) chrome-trace analysis (stdlib)")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--invocations", type=int, default=0,
                        help="number of timed invocations to cut the timeline into")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--text-out", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    events = load_events(args.trace)
    kernels = [e for e in events if e.get("cat") in ("kernel", "Kernel") and e.get("ph") == "X"]
    memcpys = [e for e in events if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X"]
    driver = [e for e in events if e.get("cat") == "cuda_driver" and e.get("ph") == "X"]
    device = sorted(kernels + memcpys, key=lambda e: float(e["ts"]))

    facts = {
        "trace": os.path.abspath(args.trace),
        "n_events": len(events),
        "kernel_calls": len(kernels),
        "memcpy_calls": len(memcpys),
        "driver_api_calls": len(driver),
        "kernel_us": {k: v for k, v in spans(kernels).items() if k != "durations"},
        "memcpy_us": {k: v for k, v in spans(memcpys).items() if k != "durations"},
        "driver_us": {k: v for k, v in spans(driver).items() if k != "durations"},
        "device": stats_block(device),
    }

    grids = collections.Counter(
        f"{tuple(e['args'].get('grid', []))}x{tuple(e['args'].get('block', []))}"
        for e in kernels
    )
    occupancy = [float(e["args"]["est. achieved occupancy %"]) for e in kernels
                 if e.get("args", {}).get("est. achieved occupancy %") is not None]
    shared = [float(e["args"].get("shared memory", 0) or 0) for e in kernels]
    distinct_kernels = collections.Counter(e.get("name") for e in kernels)
    facts["kernel_census"] = {
        "distinct_names": len(distinct_kernels),
        "calls_per_name_histogram": dict(collections.Counter(distinct_kernels.values())),
        "grid_x_block_histogram": {str(k): v for k, v in grids.most_common(8)},
        "occupancy_pct": {
            "count": len(occupancy),
            "min": min(occupancy) if occupancy else None,
            "median": statistics.median(occupancy) if occupancy else None,
            "max": max(occupancy) if occupancy else None,
        },
        "shared_memory_bytes": {"max": max(shared) if shared else None,
                                "nonzero_calls": sum(1 for s in shared if s)},
        "top_names": distinct_kernels.most_common(5),
    }

    groups = group_by_gaps(device, args.invocations) if args.invocations else group_by_gaps(device, 1)
    facts["per_invocation"] = [stats_block(group) for group in groups]
    if args.invocations and len(groups) == args.invocations:
        facts["per_invocation_summary"] = {
            "span_us_median": statistics.median(g["span_us"] for g in facts["per_invocation"]),
            "busy_us_median": statistics.median(g["busy_us"] for g in facts["per_invocation"]),
            "busy_pct_median": statistics.median(g["busy_pct"] for g in facts["per_invocation"]),
            "gap_total_us_median": statistics.median(g["gap_total_us"] for g in facts["per_invocation"]),
            "gap_median_us_median": statistics.median(g["gap_median_us"] for g in facts["per_invocation"]),
            "n_events_per_group": facts["per_invocation"][0]["n_events"],
        }

    lines = [f"=== trace facts: {facts['trace']} ==="]
    for key in ("n_events", "kernel_calls", "memcpy_calls", "driver_api_calls"):
        lines.append(f"{key}: {facts[key]}")
    for key in ("kernel_us", "memcpy_us", "driver_us", "device"):
        lines.append(f"{key}: {facts[key]}")
    lines.append(f"kernel_census: {json.dumps(facts['kernel_census'])}")
    if facts.get("per_invocation_summary"):
        lines.append(f"per_invocation: {json.dumps(facts['per_invocation_summary'])}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.text_out:
        with open(args.text_out, "w") as handle:
            handle.write(text)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(facts, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
