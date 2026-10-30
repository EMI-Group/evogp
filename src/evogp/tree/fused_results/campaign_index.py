"""Index a FUSED campaign directory and its probe directory into ONE JSON + a summary.

The harness writes **one JSON per (cell, arm)**; the fused round additionally writes
**one probe record per cell**.  Comparing those two by hand -- and answering "which
(cell, arm, workload) triples do I actually have a number for?" -- is what this script
does once, mechanically:

    python src/evogp/tree/fused_results/campaign_index.py \\
        --results src/evogp/tree/fused_results/main \\
        --probes  src/evogp/tree/fused_results/probes

It writes ``campaign.json`` (schema ``evogp-fused-campaign-v1``) next to a human
summary (``campaign.md``) in ``--out``, default ``src/evogp/tree/fused_results`` -- the
node's own result root, NOT the campaign directory: the harness's driver record is
``<results>/campaign.json`` already, so an index written there would overwrite it.  The
index is therefore ``<node>/fused_results/campaign.json``, sitting next to ``main/`` and
``probes/``.

What the index holds
--------------------
``configs[]``
    One entry per **(cell, arm, workload)** -- the granularity a report needs -- even
    though the harness stores one JSON per (cell, arm).  Each entry carries the record
    path, ``status`` (``ok`` / ``failed`` / ``missing``), ``median_ms``,
    ``first_call_ms`` and ``samples_ms``, plus the driver's own status for that config
    so "scheduled but never written" (``missing``), "the worker exited non-zero"
    (``failed``) and "measured" (``ok``) are never confused.  A workload the record did
    not time is ``missing`` too, with a ``detail`` saying so; a ``gp_step`` block is
    indexed as its own workload from ``record["gp_step"]`` (median of step totals, no
    cold column) because the harness keeps it outside ``record["workloads"]``.
``probes[]``
    One entry per probe record: the cell, the record path, ``status``, the route engine
    the probe's own policy block reports, ``fused_served`` (every recorded surface was
    served -- kernel launches, not just attempts) and ``median_ms_by_workload``.
``cells`` / ``arms``
    The union over configs and probes, so a report's coverage statement is checkable.
``counts``
    ``ok`` / ``failed`` / ``missing`` over the ``configs[]`` entries.  Missing configs
    are RECORDED as missing, never dropped.

Everything is read-only and stdlib-only: no torch, no evogp, no harness import, so the
index can be rebuilt on any box from the JSON alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

#: This node's result root (``src/evogp/tree/fused_results``): the index and the
#: summary are written here, next to the campaign (``main/``) and probe (``probes/``)
#: directories.
_HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA = "evogp-fused-campaign-v1"
DEFAULT_RESULTS = os.path.join(_HERE, "main")
DEFAULT_PROBES = os.path.join(_HERE, "probes")
#: The workloads the harness times by default (``gp_step`` is kept separately: it is a
#: generation loop, not a per-call surface).
DEFAULT_WORKLOADS = ("sr_fitness", "batch_forward", "forward")
#: Files in a campaign directory that are NOT per-config records.
NOT_RECORDS = ("campaign.json", "campaign_index.json")


def _num(value, ndigits: int = 4):
    """``round(float(value), ndigits)``, or ``None`` for a null/missing reading."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _load(path: str):
    """``(payload, error)`` -- a JSON file, or ``(None, repr(exc))``."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - a broken record is reported, never raised
        return None, repr(exc)


def _discover_records(results: str) -> dict:
    """``{(cell, arm): path}`` for every ``<results>/<cell>/<arm>.json``."""
    found = {}
    if not os.path.isdir(results):
        return found
    for cell in sorted(os.listdir(results)):
        cell_dir = os.path.join(results, cell)
        if not os.path.isdir(cell_dir):
            continue
        for name in sorted(os.listdir(cell_dir)):
            if not name.endswith(".json") or name in NOT_RECORDS:
                continue
            found[(cell, name[: -len(".json")])] = os.path.join(cell_dir, name)
    return found


def _driver_state(results: str):
    """``(protocol, {(cell, arm): entry}, schema, error)`` from the driver record."""
    path = os.path.join(results, "campaign.json")
    if not os.path.isfile(path):
        # No driver record is a legitimate state (a probe-only round, or a campaign that
        # has not run yet), not an error: configs[] then rests on the records on disk.
        return {}, {}, None, None
    driver, error = _load(path)
    if not isinstance(driver, dict):
        return {}, {}, None, error
    protocol = driver.get("protocol") if isinstance(driver.get("protocol"), dict) else {}
    statuses = {}
    for entry in driver.get("configs") or []:
        if not isinstance(entry, dict):
            continue
        cell, arm = entry.get("cell"), entry.get("arm")
        if not cell or not arm:
            continue
        statuses[(str(cell), str(arm))] = {
            "status": str(entry.get("status") or "unknown"),
            "detail": entry.get("detail") or "",
            "wall_s": entry.get("wall_s"),
            "record": entry.get("record"),
        }
    return protocol, statuses, driver.get("schema"), None


def _workload_names(protocol: dict, record) -> list:
    """The workload surfaces this config should have: the protocol's, plus any extra."""
    names = [
        str(name)
        for name in (protocol.get("workloads") or DEFAULT_WORKLOADS)
        if name and str(name) != "gp_step"
    ]
    if not names:
        names = list(DEFAULT_WORKLOADS)
    for name in (record or {}).get("workloads") or {}:
        if name not in names:
            names.append(name)
    return names


def collect_configs(results: str, protocol: dict, statuses: dict, found: dict):
    """One ``configs[]`` entry per (cell, arm, workload), missing ones included."""
    configs = []
    counts = {"ok": 0, "failed": 0, "missing": 0}
    for cell, arm in sorted(set(statuses) | set(found)):
        path = found.get((cell, arm))
        record, error = (_load(path) if path else (None, None))
        driver = statuses.get((cell, arm)) or {}
        driver_status = driver.get("status")
        names = _workload_names(protocol, record)
        if isinstance(record, dict) and record.get("gp_step"):
            names = names + ["gp_step"]
        if not path:
            names = names or list(DEFAULT_WORKLOADS)

        for name in names:
            entry = {
                "cell": cell,
                "arm": arm,
                "workload": name,
                "record": path or driver.get("record"),
                "status": "missing",
                "median_ms": None,
                "first_call_ms": None,
                "samples_ms": [],
                "driver_status": driver_status,
                "detail": "",
            }
            if record is None:
                entry["detail"] = error or driver.get("detail") or "no record for this config"
                entry["status"] = "failed" if driver_status in ("failed", "timeout") else "missing"
            elif name == "gp_step":
                block = record.get("gp_step") if isinstance(record.get("gp_step"), dict) else None
                if block:
                    entry["status"] = "ok"
                    entry["median_ms"] = _num(block.get("median_total_ms"))
                    entry["samples_ms"] = [
                        _num(step.get("total_ms")) for step in block.get("steps") or []
                    ]
                    entry["detail"] = (
                        "gp_step block: median of the step totals (the generation loop has "
                        "no cold/warm split)"
                    )
                else:
                    entry["detail"] = "record carries no gp_step block"
            else:
                block = (record.get("workloads") or {}).get(name)
                if isinstance(block, dict):
                    entry["status"] = "ok"
                    entry["median_ms"] = _num(block.get("median_ms"))
                    entry["first_call_ms"] = _num(block.get("first_call_ms"))
                    entry["samples_ms"] = [_num(value) for value in block.get("samples_ms") or []]
                elif isinstance(record.get("workloads"), dict):
                    entry["detail"] = "workload not recorded for this config"
                else:
                    entry["detail"] = "record has no workloads block"
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            configs.append(entry)
    return configs, counts


def _discover_probes(probes_dir: str) -> list:
    paths = []
    if os.path.isdir(probes_dir):
        for root, _dirs, files in os.walk(probes_dir):
            for name in sorted(files):
                if name.endswith(".json") and name not in NOT_RECORDS:
                    paths.append(os.path.join(root, name))
    return sorted(paths)


def collect_probes(probes_dir: str) -> list:
    """One ``probes[]`` entry per probe record (an unreadable one is reported as such)."""
    probes = []
    for path in _discover_probes(probes_dir):
        record, error = _load(path)
        entry = {
            "cell": None,
            "arm": None,
            "record": path,
            "status": "unreadable",
            "route_engine": None,
            "fused_served": None,
            "median_ms_by_workload": {},
            "cold_ms_by_workload": {},
            "detail": error or "",
        }
        if not isinstance(record, dict):
            probes.append(entry)
            continue
        entry["cell"] = record.get("cell")
        entry["arm"] = record.get("arm")
        route = (record.get("route") or {}).get("sr_fitness") or {}
        entry["route_engine"] = route.get("engine")
        served = []
        for name, block in (record.get("workloads") or {}).items():
            if not isinstance(block, dict):
                continue
            entry["median_ms_by_workload"][name] = _num(block.get("median_ms"))
            entry["cold_ms_by_workload"][name] = _num(block.get("first_call_ms"))
            if "served" in block:
                served.append(bool(block["served"]))
        entry["fused_served"] = bool(served) and all(served)
        entry["status"] = "ok" if entry["fused_served"] else "failed"
        if not entry["fused_served"]:
            entry["detail"] = (
                "at least one surface was not served (kernel launches != calls, or a "
                "decline); the route fell back -- see the probe's workloads[].fused_delta"
            )
        probes.append(entry)
    return probes


def _summary_markdown(index: dict) -> str:
    """The human summary written next to ``campaign.json``."""
    lines = [
        "# Fused campaign index",
        "",
        f"- schema: `{index['schema']}`",
        f"- created: {index['created_utc']}",
        f"- results: `{index['protocol']['results_dir']}`",
        f"- probes: `{index['protocol']['probes_dir']}`",
        f"- counts: ok={index['counts']['ok']} failed={index['counts']['failed']} "
        f"missing={index['counts']['missing']}",
        f"- cells: {', '.join(f'`{cell}`' for cell in index['cells']) or '-'}",
        f"- arms: {', '.join(f'`{arm}`' for arm in index['arms']) or '-'}",
        "",
        "## Configs (one row per cell x arm x workload)",
        "",
        "| cell | arm | workload | status | median ms | cold ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in index["configs"]:
        median = "-" if entry["median_ms"] is None else f"{entry['median_ms']:.4f}"
        cold = "-" if entry["first_call_ms"] is None else f"{entry['first_call_ms']:.4f}"
        detail = f" ({entry['detail']})" if entry["status"] != "ok" and entry["detail"] else ""
        lines.append(
            f"| {entry['cell']} | {entry['arm']} | {entry['workload']} | "
            f"{entry['status']}{detail} | {median} | {cold} |"
        )
    lines += [
        "",
        "## Probes",
        "",
        "| cell | route engine | served | workload medians (ms) | cold (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for probe in index["probes"]:
        medians = ", ".join(
            f"{name}={value}" for name, value in sorted(probe["median_ms_by_workload"].items())
        )
        cold = ", ".join(
            f"{name}={value}" for name, value in sorted(probe["cold_ms_by_workload"].items())
        )
        lines.append(
            f"| {probe['cell'] or '-'} | {probe['route_engine'] or '-'} | "
            f"{probe['fused_served']} | {medians or '-'} | {cold or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_index(results: str, probes_dir: str) -> dict:
    """Read both directories into the one index document."""
    found = _discover_records(results)
    protocol, statuses, driver_schema, driver_error = _driver_state(results)
    configs, counts = collect_configs(results, protocol, statuses, found)
    probes = collect_probes(probes_dir)
    cells = sorted(
        {entry["cell"] for entry in configs}
        | {probe["cell"] for probe in probes if probe["cell"]}
    )
    # The arm set is the union of what the configs name and what the probe records
    # themselves say (a probe file is named after its CELL, not after an arm).
    arms = sorted(
        {entry["arm"] for entry in configs}
        | {probe["arm"] for probe in probes if probe["arm"]}
    )
    return {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "one_process_per_config": protocol.get("one_process_per_config", True),
            "warmup": protocol.get("warmup"),
            "reps": protocol.get("reps"),
            "gp_warmup": protocol.get("gp_warmup"),
            "gp_reps": protocol.get("gp_reps"),
            "seed": protocol.get("seed"),
            "workloads": protocol.get("workloads"),
            "driver_schema": driver_schema,
            "results_dir": os.path.abspath(results),
            "probes_dir": os.path.abspath(probes_dir),
            "one_entry_per_workload": True,
            "driver_error": driver_error,
            "notes": [
                "the harness writes ONE JSON per (cell, arm); configs[] expands each into "
                "one entry per workload",
                "a missing record is status 'missing' (or 'failed' when the driver saw a "
                "non-zero exit / timeout), never dropped",
                "gp_step is indexed from the record's own block (median of step totals); "
                "the generation loop has no cold/warm split",
                "probes[] is the fused route's own evidence: route engine and the "
                "served-call proof (kernel launches per invocation, declines)",
            ],
        },
        "cells": cells,
        "arms": arms,
        "configs": configs,
        "probes": probes,
        "counts": counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python src/evogp/tree/fused_results/campaign_index.py",
        description=(
            "Index a fused campaign (one JSON per cell x arm) and its probe records into "
            "one campaign.json (one entry per cell x arm x workload) plus a summary."
        ),
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="campaign directory (the harness's --out), default "
                             "src/evogp/tree/fused_results/main")
    parser.add_argument("--probes", default=DEFAULT_PROBES,
                        help="probe directory (fused_probe.py --out files), default "
                             "src/evogp/tree/fused_results/probes")
    parser.add_argument("--out", default=_HERE,
                        help="where campaign.json is written; default this node's "
                             "fused_results directory (NOT --results: the harness's own "
                             "driver record is <results>/campaign.json)")
    parser.add_argument("--summary", default=None,
                        help="summary path (default <out>/campaign.md)")
    parser.add_argument("--no-summary", action="store_true", help="skip the markdown summary")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    index = build_index(args.results, args.probes)

    os.makedirs(os.path.abspath(args.out), exist_ok=True)
    out_path = os.path.join(os.path.abspath(args.out), "campaign.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1, sort_keys=False)
        handle.write("\n")

    summary_path = None
    if not args.no_summary:
        summary_path = os.path.abspath(args.summary or os.path.join(args.out, "campaign.md"))
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(_summary_markdown(index))

    print(
        f"[index] {len(index['configs'])} configs "
        f"(ok={index['counts']['ok']} failed={index['counts']['failed']} "
        f"missing={index['counts']['missing']}), {len(index['probes'])} probes, "
        f"{len(index['cells'])} cells, {len(index['arms'])} arms",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    if summary_path:
        print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
