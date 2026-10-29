"""Tabulate one ``evoxc`` benchmark result directory into ONE markdown document.

Input is what :mod:`worker` writes (``<results>/<cell>/<arm>.json``, one record per
(cell, arm)) plus the driver record ``<results>/campaign.json``.  The output holds,
in order: the protocol statement from ``campaign.json``, the environment table, one
warm-timing table per workload (``median_ms`` with the ``min_ms`` in parentheses)
plus the ``auto route``/``fastest`` columns and -- only when a batch arm is present
-- one line naming the batch route's knobs, the per-cell speedup against the two
torch baselines, the ``gp_step`` total with its eval/ops split, the excluded cold
compile call, the cross-arm digest agreement, and a caveats block.

The batch arms (``common.is_batch_arm``) are rendered by the same rules as the other
evoxc arms: their ``routes[...]`` carries ``kind: "batch"`` and is quoted as such,
their ``first_call_ms`` is the merged-graph compile for the whole population (so a
caveat says so), and their knobs are read from the record's ``batch_policy`` with
the registry's ``knobs`` as the fallback.

Two rules it obeys: it never re-derives a number (every table cell is a field of a
record, so a missing record, a ``status != "ok"`` config or a contaminated guard
stays visible instead of being averaged away), and it never needs torch, the GPU or
``evogp`` -- the records are plain JSON and the only import is :mod:`common`.

One deliberate exception to the first rule: the ``gp_step`` block is read from its
own ``median_total_ms`` / ``median_eval_ms`` / ``median_ops_ms`` fields, and when a
record carries ``steps[]`` but not those aggregate keys (a schema drift, a
hand-trimmed record) the value is re-derived from the upper median of
``steps[].total_ms`` / ``steps[].eval_ms`` -- exactly the statistic
:mod:`worker` computes -- and marked with ``*`` in the cell and in the caption.
A record with raw steps must never blank a ``gp_step`` cell.

``--compare-results DIR --compare-label NAME`` (default label ``before``) renders one
extra table per workload plus one for the ``gp_step`` total: the ratio of the
primary directory's median to the compare directory's median for the same cell and
arm, both directories measured in the same session.  Without the flag the document
is byte-identical to the single-directory render.

Usage (repository root, stdlib only, GPU-free)::

    python3 test/benchmark/evoxc/summarize.py \\
        --results test/benchmark/evoxc/results/baseline \\
        --out test/benchmark/evoxc/results/baseline/SUMMARY.md

    # one workload only, and only two of the tables
    python3 test/benchmark/evoxc/summarize.py --results results/w1 --out /tmp/w1.md \\
        --workloads sr_fitness --markdown-tables workloads,speedup

    # before/after: the primary dir against a second dir from the same session
    python3 test/benchmark/evoxc/summarize.py --results results/final/after \\
        --compare-results results/final/before --compare-label before \\
        --out results/final/SUMMARY.md

``--markdown-tables`` takes ``all`` (default) or a comma-separated subset of
``env,workloads,speedup,gp_step,cold,digest,compare``.  A partially populated result
directory is the normal state while a campaign runs: absent records render as
``-`` and are listed under ``## Caveats``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402

MISSING = "-"
DEFAULT_WORKLOADS = ("sr_fitness", "batch_forward", "forward")
#: the value paths whose digests must agree bitwise across arms
VALUE_WORKLOADS = ("batch_forward", "forward")
SECTIONS = ("env", "workloads", "speedup", "gp_step", "cold", "digest", "compare")
#: Presentation order of the batch knobs.  The TABLE itself is ``common.BATCH_KNOBS``
#: (the single owner); this only says in which order its keys are quoted, and any
#: knob ``common`` gains later is appended instead of silently disappearing.
_BATCH_KNOB_ORDER = ("band_size", "bands", "alignment", "switch_ops", "min_pop", "cache_size")
BATCH_KNOB_KEYS = tuple(
    [key for key in _BATCH_KNOB_ORDER
     if key in {entry[0] for entry in common.BATCH_KNOBS}]
    + [entry[0] for entry in common.BATCH_KNOBS if entry[0] not in _BATCH_KNOB_ORDER]
)
#: ``EVOGP_BATCH_BANDS`` -> ``bands``: an arm's registry ``knobs`` are env-var keyed,
#: a record's ``batch_policy`` is spec-keyed, and the caption quotes the latter.
BATCH_KNOB_BY_ENV = {entry[1]: entry[0] for entry in common.BATCH_KNOBS}



# --- small formatting / file helpers -------------------------------------------
def load_json(path: str) -> Tuple[Any, Optional[str]]:
    """``(payload, error)`` -- an unreadable file is a caveat, never a crash."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - report the file, do not die on it
        return None, f"{type(exc).__name__}: {exc}"


def fmt_ms(value: Any) -> str:
    """3 decimals for a millisecond field, ``-`` when absent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return MISSING
    return f"{value:.3f}"


def fmt_ratio(numerator: Any, denominator: Any) -> str:
    """2 decimals for a speedup ratio, ``-`` when either side is absent/zero."""
    for value in (numerator, denominator):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return MISSING
    if not denominator:
        return MISSING
    return f"{numerator / denominator:.2f}x"


def as_number(value: Any) -> Optional[float]:
    """The value when it is a real number, else ``None`` (``True`` is not a time)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fmt_fitness(value: Any) -> str:
    """6 decimals for a fitness value, ``-`` when absent."""
    number = as_number(value)
    return MISSING if number is None else f"{number:.6f}"


def share_pct(part: Any, whole: Any) -> Optional[float]:
    """``100 * part / whole`` when both are usable numbers, else ``None``."""
    part, whole = as_number(part), as_number(whole)
    if part is None or whole is None or whole <= 0:
        return None
    return 100.0 * part / whole


def step_median(steps: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Upper median of one ``gp_step.steps[]`` field -- the statistic ``worker.py`` stores.

    ``worker.py`` takes ``sorted(values)[len(values) // 2]`` (the upper median, not
    the mean of the two middle values), so a fallback that uses this rule reproduces
    what the record's own ``median_*_ms`` aggregate would have said.
    """
    ordered = sorted(number for number in (as_number(step.get(key)) for step in steps)
                     if number is not None)
    return ordered[len(ordered) // 2] if ordered else None


def table(header: List[str], rows: List[List[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


# --- ordering: the frozen cell grid, then the frozen arm registry --------------
def cell_grid() -> List[Dict[str, Any]]:
    """``main_cells()`` then ``n64_cells()``, de-duplicated, in that order."""
    grid: List[Dict[str, Any]] = []
    seen = set()
    for spec in list(common.main_cells()) + list(common.n64_cells()):
        if spec["id"] not in seen:
            seen.add(spec["id"])
            grid.append(spec)
    return grid


def order_cells(cell_ids) -> List[str]:
    """Grid order first; a cell outside the grid (an ad-hoc ``--cells`` run) last."""
    rank = {spec["id"]: index for index, spec in enumerate(cell_grid())}
    known = sorted((cid for cid in cell_ids if cid in rank), key=lambda cid: rank[cid])
    return known + sorted(cid for cid in cell_ids if cid not in rank)


def order_arms(arms) -> List[str]:
    known = [arm for arm in common.ALL_ARMS if arm in arms]
    return known + sorted(arm for arm in arms if arm not in common.ALL_ARMS)


def is_evoxc_arm(arm: str) -> bool:
    return arm in common.EVOXC_ARMS or arm not in common.TORCH_ARMS


# --- discovery -----------------------------------------------------------------
def load_results(results_dir: str):
    """Load ``campaign.json`` plus every record found on disk.

    Returns ``(campaign, campaign_error, configs, records, problems)`` where
    ``configs`` maps ``(cell, arm) -> campaign entry``, ``records`` maps
    ``(cell, arm) -> record`` and ``problems`` is a list of
    ``(cell, arm, reason, campaign entry)`` for absent/unreadable records.
    """
    campaign, campaign_error = load_json(os.path.join(results_dir, "campaign.json"))
    if campaign is not None and not isinstance(campaign, dict):
        campaign_error, campaign = "campaign.json is not a JSON object", None
    configs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in (campaign or {}).get("configs") or []:
        if isinstance(entry, dict) and entry.get("cell") and entry.get("arm"):
            configs[(entry["cell"], entry["arm"])] = entry

    pairs = set(configs)
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            sub = os.path.join(results_dir, name)
            if not os.path.isdir(sub):
                continue
            for fname in os.listdir(sub):
                if fname.endswith(".json"):
                    pairs.add((name, fname[: -len(".json")]))

    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    problems: List[Tuple[str, str, str, Optional[Dict[str, Any]]]] = []
    for cell, arm in sorted(pairs):
        entry = configs.get((cell, arm)) or {}
        path = os.path.join(results_dir, cell, f"{arm}.json")
        recorded = entry.get("record")
        # the campaign's own path, but only when it lives inside this directory: a
        # copied or partly cleaned result dir must not borrow another one's record
        inside = bool(recorded) and os.path.abspath(recorded).startswith(
            os.path.abspath(results_dir) + os.sep)
        if not os.path.isfile(path) and inside and os.path.isfile(recorded):
            path = recorded
        if not os.path.isfile(path):
            problems.append((cell, arm, "record file missing", entry))
            continue
        payload, error = load_json(path)
        if not isinstance(payload, dict):
            problems.append((cell, arm, f"unreadable record ({error})", entry))
            continue
        records[(cell, arm)] = payload
    return campaign, campaign_error, configs, records, problems


def workload_entry(record: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    entry = (record.get("workloads") or {}).get(name)
    return entry if isinstance(entry, dict) else None


def median_of(records, cell: str, arm: str, name: str) -> Any:
    entry = workload_entry(records.get((cell, arm)), name)
    return entry.get("median_ms") if entry else None


def fastest_arm(records, cell: str, arms, name: str) -> str:
    """The arm with the smallest ``median_ms`` for this cell/workload (``-`` if none)."""
    best, best_median = MISSING, None
    for arm in arms:
        median = as_number(median_of(records, cell, arm, name))
        if median is not None and (best_median is None or median < best_median):
            best, best_median = arm, median
    return best


def route_text(records, cell: str, workload: str) -> str:
    """``eval_mode/backend -> kind`` from the cell's route record.

    The shipped default (``evoxc-auto``) is preferred, then ``evoxc-auto-iree``; only
    when neither is recorded does the first evoxc arm present supply the cell, and it
    is then named in parentheses.  The batch arms are registered last in
    ``common.EVOXC_ARMS``, so they never shadow the default policy: they supply this
    column only for a directory that recorded nothing else.  ``kind`` is a field when
    the record states one (a batch route carries ``kind: "batch"``); otherwise the
    dense-vs-specialized half is derived from ``would_use_dense`` exactly as before.
    """
    source = None
    for arm in ("evoxc-auto", "evoxc-auto-iree"):
        if (cell, arm) in records:
            source = arm
            break
    if source is None:
        for arm in common.EVOXC_ARMS:
            if (cell, arm) in records:
                source = arm
                break
    if source is None:
        return MISSING
    route = (records[(cell, source)].get("routes") or {}).get(workload)
    if not isinstance(route, dict):
        return MISSING
    mode = route.get("eval_mode") or "?"
    backend = route.get("backend") or "?"
    kind = route.get("kind")
    if kind is None:
        kind = "dense" if route.get("would_use_dense") else "specialized"
    text = f"{mode}/{backend} → {kind}"
    annotations: List[str] = []
    if kind == "batch":
        # the flat-vs-banded distinction is the one knob that changes the merge
        # cost by an order of magnitude, so it belongs in the routing column;
        # the default banding is the unannotated case
        bands = route.get("bands")
        if bands not in (None, "", "default"):
            annotations.append(f"bands={bands}")
    if source != "evoxc-auto":
        annotations.append(source)
    if annotations:
        text += f" ({', '.join(annotations)})"
    return text


# --- sections ------------------------------------------------------------------
def protocol_lines(results_dir, campaign, campaign_error, configs, records, cells, arms) -> List[str]:
    lines = [f"Source: `{results_dir}` — **{len(records)} records**, {len(cells)} cells x "
             f"{len(arms)} arms.", ""]
    if campaign is None:
        lines.append(
            f"No `campaign.json` could be read ({campaign_error or 'missing'}), so the protocol "
            "statement is unavailable: every table below is built from the record files that "
            "exist, which is a partially populated result directory."
        )
        lines.append("")
        return lines
    proto = campaign.get("protocol") or {}
    failures = campaign.get("failures")
    elapsed = campaign.get("elapsed_s")
    dropped = proto.get("scope_dropped") or []
    scope_text = "; ".join(f"`{arm}` on pops {list(pops)}"
                           for arm, pops in (proto.get("scope_rule") or {}).items())
    measured = ((proto.get("specialized_iree_compile_ms_per_structure") or {})
                .get("measured_range") or [None, None])
    lines.append(
        f"The campaign ran **{len(configs)} configs** sequenced one at a time "
        f"(`one_process_per_config: {proto.get('one_process_per_config')}`, "
        f"`sequential: {proto.get('sequential')}`): every (cell, arm) pair gets its own process, so "
        "each record pays its own cold compile and inherits no other arm's compile cache.  "
        f"Elapsed **{elapsed} s**, failures **{failures if failures is not None else '?'}**."
    )
    lines.append(
        f"Per workload one cold first call is timed and reported separately (never part of the "
        f"medians), followed by `warmup={proto.get('warmup')}` untimed calls and "
        f"`reps={proto.get('reps')}` sync-bracketed samples; the GP generation step used "
        f"`gp_warmup={proto.get('gp_warmup')}` + `gp_reps={proto.get('gp_reps')}`; seed "
        f"`{proto.get('seed')}`, workloads `{', '.join(proto.get('workloads') or [])}`, "
        f"guard wait `{proto.get('guard_wait')} s`, scope rule "
        f"{scope_text or 'unrestricted'} — **{len(dropped)} config(s) dropped** from the schedule "
        f"(`specialized` iree measured at {measured[0]}–{measured[1]} ms per tree structure, so a "
        "fresh 256/1000-tree population is minutes of compile per evaluation; the dropped pairs are "
        "listed under `## Caveats`)."
    )
    lines.append("")
    return lines


def env_lines(campaign, records) -> List[str]:
    driver_env = (campaign or {}).get("driver_env")
    note = None
    if not isinstance(driver_env, dict):
        first = next(iter(records.values()), None)
        driver_env = (first or {}).get("env") or {}
        note = ("`campaign.json` carries no `driver_env`; this table is the environment block of "
                "the first record instead." if first else
                "No environment block is available: no `campaign.json` and no record to read.")
    knobs = driver_env.get("knobs") or {}
    rows: List[List[Any]] = []

    def row(label, value):
        rows.append([label, MISSING if value in (None, "") else value])

    row("created_utc", driver_env.get("created_utc"))
    row("git SHA (`tree_root_sha`)", driver_env.get("tree_root_sha"))
    row("tree_root", f"`{driver_env.get('tree_root')}`" if driver_env.get("tree_root") else None)
    for key in ("python", "torch", "numpy", "hip", "iree_base_compiler",
                "iree_base_runtime", "evoxc_version", "evoxc_path", "gpu",
                "gpu_total_memory_gb"):
        row(key, driver_env.get(key))
    for key, value in sorted(knobs.items()):
        if key == "EVOXC_DISABLE_LOCATIONS":
            continue  # reported on its own row right below, as the protocol requires
        row(f"knob `{key}`", f"`{value}`")
    row("`EVOXC_DISABLE_LOCATIONS`", f"`{knobs['EVOXC_DISABLE_LOCATIONS']}`"
        if "EVOXC_DISABLE_LOCATIONS" in knobs else MISSING)
    if not knobs:
        rows.append(["knob", "none set"])
    lines = ["## Environment", ""]
    if note:
        lines.extend([note, ""])
    lines.extend(table(["key", "value"], rows))
    lines.append("")
    return lines


def batch_arm_policy(records, arm: str) -> Optional[Dict[str, Any]]:
    """The ``batch_policy`` block of the first record of ``arm`` (``None`` if none)."""
    for (_cell, name), record in sorted(records.items()):
        if name != arm:
            continue
        policy = record.get("batch_policy")
        if isinstance(policy, dict):
            return policy
    return None


def batch_arm_knobs(records, arm: str) -> str:
    """``band_size=32, bands=flat, ...`` for one batch arm, record first, registry second.

    The record's ``batch_policy`` holds the RESOLVED values; the arm's registry
    ``knobs`` says which variables it overrides, so an arm that changes one knob is
    quoted as that one knob and an arm that changes none is quoted in full.  Neither
    is a knob table of its own: keys, defaults and env-var names all come from
    ``common.BATCH_KNOBS``.
    """
    spec = common.EVOXC_ARMS.get(arm) or {}
    # the arm's registry overrides, as spec keys (``EVOGP_BATCH_BANDS`` -> ``bands``)
    by_key = {BATCH_KNOB_BY_ENV[env]: str(value)
              for env, value in (spec.get("knobs") or {}).items()
              if env in BATCH_KNOB_BY_ENV}
    policy = batch_arm_policy(records, arm)
    if policy is not None:
        if by_key:
            # this arm differs from the default policy in these knobs only, so quote
            # exactly those; the record's resolved value wins over the registry literal
            return ", ".join(f"{key}={policy.get(key) if policy.get(key) is not None else by_key[key]}"
                             for key in BATCH_KNOB_KEYS if key in by_key)
        defaults = {entry[0]: entry[2] for entry in common.BATCH_KNOBS}
        pairs = [f"{key}={policy.get(key)}" for key in BATCH_KNOB_KEYS
                 if policy.get(key) is not None
                 and not (key == "cache_size" and str(policy.get(key)) == defaults[key])]
        if policy.get("mode_supported") is False:
            pairs.append("mode_supported=false")
        return ", ".join(pairs) or "unknown"
    # no record-level policy: the registry is the last source (an override is a fact,
    # and the documented default is labelled rather than presented as measured)
    if by_key:
        return ", ".join(f"{key}={by_key[key]}" for key in BATCH_KNOB_KEYS if key in by_key)
    return ", ".join(f"{entry[0]}={entry[2]} (documented default)"
                     for entry in common.BATCH_KNOBS
                     if entry[0] != "cache_size")


def batch_arm_note(arms, records) -> str:
    """ONE caption line naming the batch route's knobs, or ``""`` when no batch arm ran.

    Emitting nothing for a batch-free directory is what keeps those renders
    byte-identical, and those are the only ones with a frozen report of record.
    """
    present = [arm for arm in arms if common.is_batch_arm(arm)]
    if not present:
        return ""
    named = ", ".join(f"`{arm}` ({batch_arm_knobs(records, arm)})" for arm in present)
    return ("Batch route arms: " + named + " — one merged graph per population, so "
            "`first_call_ms` is a compile and the warm columns are the ones to compare.")


def batch_first_call_present(cells, arms, records) -> bool:
    """Did any batch arm record a cold first call (i.e. a merged-graph compile)?"""
    for cell in cells:
        for arm in arms:
            if not common.is_batch_arm(arm):
                continue
            entry = workload_entry(records.get((cell, arm)), "sr_fitness")
            if entry is not None and as_number(entry.get("first_call_ms")) is not None:
                return True
    return False


def workload_lines(name, cells, arms, records) -> List[str]:
    lines = [f"## Workload: {name}", "",
             "Warm/steady-state `median_ms` with the `min_ms` in parentheses.", ""]
    rows: List[List[Any]] = []
    for cell in cells:
        row: List[Any] = [f"`{cell}`"]
        for arm in arms:
            entry = workload_entry(records.get((cell, arm)), name)
            row.append(MISSING if entry is None else
                       f"{fmt_ms(entry.get('median_ms'))} ({fmt_ms(entry.get('min_ms'))})")
        row.extend([route_text(records, cell, name), fastest_arm(records, cell, arms, name)])
        rows.append(row)
    lines.extend(table(["cell"] + list(arms) + ["auto route", "fastest"], rows))
    lines.append("")
    lines.append("`auto route` is read from the cell's `evoxc-auto` record (`routes[...]`: "
                 "`eval_mode`/`backend` and `dense`|`specialized` from `would_use_dense`); when that "
                 "arm is absent the arm the route was read from is named in parentheses.")
    lines.append("")
    note = batch_arm_note(arms, records)
    if note:
        lines.extend([note, ""])
    return lines


def speedup_lines(name, cells, arms, records) -> List[str]:
    lines = [f"### Speedup vs torch ({name})", "",
             "`torch-*` median divided by the arm's median, same cell, same session (>1 = the "
             "torch baseline is slower).", ""]
    rows: List[List[Any]] = []
    for cell in cells:
        per_tree = median_of(records, cell, "torch-per-tree", name)
        batched = median_of(records, cell, "torch-batched", name)
        for arm in arms:
            if not is_evoxc_arm(arm):
                continue
            median = median_of(records, cell, arm, name)
            rows.append([f"`{cell}`", arm, fmt_ratio(per_tree, median), fmt_ratio(batched, median)])
    lines.extend(table(["cell", "arm", "torch-per-tree / arm", "torch-batched / arm"], rows))
    lines.append("")
    return lines


def gp_step_values(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """``total``/``eval``/``ops``/``fitness``/``source`` for one ``gp_step`` block.

    The aggregates ``median_total_ms`` / ``median_eval_ms`` / ``median_ops_ms`` are
    the record's own numbers and win whenever they are present.  Whatever is absent
    is re-derived from ``steps[]`` (:func:`step_median`, the same upper-median rule
    ``worker.py`` uses; ``ops`` from ``total - eval``), and ``source`` becomes
    ``"steps"`` so the caller can mark the cell instead of blanking it.
    """
    if not record:
        return None
    block = record.get("gp_step")
    if not isinstance(block, dict):
        return None
    steps = [step for step in (block.get("steps") or []) if isinstance(step, dict)]
    total = as_number(block.get("median_total_ms"))
    eval_ms = as_number(block.get("median_eval_ms"))
    ops_ms = as_number(block.get("median_ops_ms"))
    source = "median"
    if steps:
        if total is None:
            total = step_median(steps, "total_ms")
            source = "steps" if total is not None else source
        if eval_ms is None:
            eval_ms = step_median(steps, "eval_ms")
            source = "steps" if eval_ms is not None else source
        if ops_ms is None and total is not None and eval_ms is not None:
            ops_ms = total - eval_ms
            source = "steps"
    if total is None and eval_ms is None and ops_ms is None:
        return None
    return {"total": total, "eval": eval_ms, "ops": ops_ms,
            "fitness": as_number(block.get("best_fitness")), "source": source}


def gp_step_total_of(records, cell: str, arm: str) -> Optional[float]:
    values = gp_step_values(records.get((cell, arm)))
    return None if values is None else values["total"]


def gp_step_mark(values: Optional[Dict[str, Any]]) -> str:
    """``*`` when the cell was re-derived from ``steps[]`` (a footnote marker)."""
    return "*" if values is not None and values["source"] == "steps" else ""


def gp_step_lines(cells, arms, records) -> List[str]:
    totals: List[List[Any]] = []
    splits: List[List[Any]] = []
    fitness: List[List[Any]] = []
    speedups: List[List[Any]] = []
    any_fitness = False
    for cell in cells:
        total_row: List[Any] = [f"`{cell}`"]
        split_row: List[Any] = [f"`{cell}`"]
        fitness_row: List[Any] = [f"`{cell}`"]
        for arm in arms:
            values = gp_step_values(records.get((cell, arm)))
            mark = gp_step_mark(values)
            total_row.append(MISSING if values is None else f"{fmt_ms(values['total'])}{mark}")
            if values is None:
                split_row.append(MISSING)
            else:
                split = f"{fmt_ms(values['eval'])} / {fmt_ms(values['ops'])}{mark}"
                share = share_pct(values["ops"], values["total"])
                split_row.append(split if share is None else f"{split} (op {share:.0f}%)")
            fitness_row.append(fmt_fitness(None if values is None else values["fitness"]))
            any_fitness = any_fitness or (values is not None and values["fitness"] is not None)
        totals.append(total_row)
        splits.append(split_row)
        fitness.append(fitness_row)
        per_tree = gp_step_total_of(records, cell, "torch-per-tree")
        batched = gp_step_total_of(records, cell, "torch-batched")
        for arm in arms:
            if not is_evoxc_arm(arm):
                continue
            total = gp_step_total_of(records, cell, arm)
            speedups.append([f"`{cell}`", arm, fmt_ratio(per_tree, total),
                             fmt_ratio(batched, total)])
    lines = ["## gp_step", "",
             "One full GP generation (evaluate + select + crossover + mutate), warm.", "",
             "### Total", "",
             "Every cell is the record's own `median_total_ms`.  A `*` marks a cell the "
             "aggregate key could not supply, re-derived from the upper median of that "
             "record's `gp_step.steps[].total_ms` (the statistic `worker.py` stores), "
             "so a record with raw steps is never blanked.", ""]
    lines.extend(table(["cell"] + list(arms), totals))
    lines.extend(["", "### Eval / ops split", "",
                  "`median_eval_ms / median_ops_ms` with the operator share of the generation "
                  "in parentheses (`ops / total`); the operator part is the in-place difference "
                  "on the same step, not a cross-population comparison.  A `*` marks a value "
                  "re-derived from `gp_step.steps[]` (a missing `median_*` aggregate key).", ""])
    lines.extend(table(["cell"] + list(arms), splits))
    lines.extend(["", "### Speedup vs torch (gp_step)", "",
                  "`torch-*` `gp_step` total divided by the arm's total, same cell, same session "
                  "(>1 = the torch baseline is slower, i.e. the arm is faster).", ""])
    lines.extend(table(["cell", "arm", "torch-per-tree / arm", "torch-batched / arm"], speedups))
    if any_fitness:
        lines.extend(["", "### Best fitness after the timed steps", "",
                      "`gp_step.best_fitness` of the pipeline that ran the timed steps "
                      "(higher is better; `-` when the record has none).", ""])
        lines.extend(table(["cell"] + list(arms), fitness))
    lines.append("")
    return lines


# --- before/after comparison against a second result directory ------------------
def shared_cells(cells, other_records) -> List[str]:
    """The cells this document and the compare directory both measured."""
    other_cells = {cell for cell, _ in other_records}
    return [cell for cell in cells if cell in other_cells]


def compare_source_line(primary_dir: str, compare: Dict[str, Any]) -> str:
    return (f"primary `{primary_dir}` / `{compare['label']}` `{compare['dir']}` "
            "(both measured in the same session; one process per config on each side).")


def compare_rows(cells, arms, primary_records, other_records, value_of, label) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for cell in cells:
        row: List[Any] = [f"`{cell}`"]
        for arm in arms:
            now = value_of(primary_records, cell, arm)
            was = value_of(other_records, cell, arm)
            if now is not None and was is not None:
                row.append(fmt_ratio(now, was))
            elif was is not None:
                # the primary directory has no such arm: quote the compare side, marked
                row.append(f"{MISSING} ({label} {fmt_ms(was)})")
            else:
                row.append(MISSING)
        rows.append(row)
    return rows


def compare_workload_lines(name, cells, arms, records, primary_dir, compare) -> List[str]:
    """Ratio of this directory's median to the compare directory's, per cell and arm."""
    other = compare["records"]
    label = compare["label"]
    cells = shared_cells(cells, other)
    if not cells:
        return []
    rows = compare_rows(cells, arms, records, other,
                        lambda src, cell, arm: as_number(median_of(src, cell, arm, name)), label)
    lines = [f"### Compare vs {label}: {name}", "", compare_source_line(primary_dir, compare), "",
             f"Median ratio `primary / {label}` per arm, same cell: >1 means the {label} run was "
             f"faster (the primary got slower), <1 means it was slower.  A pair missing on the "
             f"primary side is quoted as `- ({label} <ms>)`; a pair missing on the {label} side "
             f"stays `-`.  Only cells present in both directories are listed.", ""]
    lines.extend(table(["cell"] + [f"{arm} / {arm}({label})" for arm in arms], rows))
    lines.append("")
    return lines


def compare_gp_step_lines(cells, arms, records, primary_dir, compare) -> List[str]:
    """Same ratio table for the ``gp_step`` total (the full generation)."""
    other = compare["records"]
    label = compare["label"]
    cells = shared_cells(cells, other)
    if not cells:
        return []
    rows = compare_rows(cells, arms, records, other, gp_step_total_of, label)
    lines = [f"### Compare vs {label}: gp_step total", "",
             compare_source_line(primary_dir, compare), "",
             f"`gp_step` total ratio `primary / {label}` per arm, same cell (>1 = the {label} "
             f"generation was faster).  Values carry the `gp_step` fallback: a record without "
             f"`median_total_ms` is read from the upper median of its `steps[].total_ms` on "
             f"either side.  Only cells present in both directories are listed.", ""]
    lines.extend(table(["cell"] + [f"{arm} / {arm}({label})" for arm in arms], rows))
    lines.append("")
    return lines


def cold_lines(cells, arms, records) -> List[str]:
    rows: List[List[Any]] = []
    for cell in cells:
        for arm in arms:
            if not is_evoxc_arm(arm):
                continue
            entry = workload_entry(records.get((cell, arm)), "sr_fitness")
            if entry is None:
                rows.append([f"`{cell}`", arm, MISSING, MISSING, MISSING])
                continue
            cache = entry.get("cache") or {}
            misses = cache.get("compile_misses_first_call")
            rows.append([f"`{cell}`", arm, fmt_ms(entry.get("first_call_ms")),
                         MISSING if misses is None else misses, fmt_ms(entry.get("median_ms"))])
    lines = ["## Cold compile (excluded from medians)", "",
             "The `sr_fitness` first call per (cell, evoxc arm): it contains any compile, so it is "
             "quoted here next to the warm median it must never be compared with.", ""]
    lines.extend(table(["cell", "arm", "first_call_ms", "compile_misses_first_call",
                        "warm median_ms"], rows))
    lines.append("")
    return lines


def digest_of(records, cell: str, arm: str, name: str) -> Dict[str, Any]:
    entry = workload_entry(records.get((cell, arm)), name)
    return (entry or {}).get("digest") or {}


def digest_lines(cells, arms, records) -> List[str]:
    rows: List[List[Any]] = []
    sr_counts: List[str] = []
    for cell in cells:
        for name in VALUE_WORKLOADS:
            groups: Dict[str, List[str]] = {}
            nan_of: Dict[str, Any] = {}
            for arm in arms:
                digest = digest_of(records, cell, arm, name)
                sha = digest.get("sha256")
                if not sha:
                    continue
                groups.setdefault(sha, []).append(arm)
                nan_of.setdefault(sha, digest.get("nan"))
            for sha, members in groups.items():
                rows.append([f"`{cell}`", name, f"`{sha}`", ", ".join(members),
                             MISSING if nan_of[sha] is None else nan_of[sha]])
        shas = [digest_of(records, cell, arm, "sr_fitness").get("sha256") for arm in arms]
        shas = [sha for sha in shas if sha]
        if shas:
            sr_counts.append(f"{cell}: {len(set(shas))}/{len(shas)}")
    lines = ["### Digest agreement", "",
             "Distinct `digest.sha256` values per cell and the arms that share them, for the two "
             "value paths (`batch_forward`, `forward`); a single shared sha per cell is the "
             "expected, numerically-consistent case.", ""]
    lines.extend(table(["cell", "workload", "sha256", "arms", "nan"], rows))
    lines.append("")
    lines.append(
        "`sr_fitness` digests legitimately differ between arms (the loss reduction order differs), "
        "so they are not compared across arms; distinct/records per cell: "
        + (", ".join(sr_counts) if sr_counts else MISSING) + "."
    )
    lines.append("")
    return lines


def caveat_lines(campaign, configs, problems, batch_cold: bool = False) -> List[str]:
    lines = ["## Caveats", ""]
    bad = [entry for entry in configs.values() if entry.get("status") != "ok"]
    contaminated = [entry for entry in configs.values()
                    if isinstance(entry.get("guard"), dict) and entry["guard"].get("contaminated")]
    if batch_cold:
        # a batch arm's cold call is a compile of the whole population's structure
        # multiset: quoting it next to a warm median is the reading error to prevent
        lines.append("- A **batch arm**'s `first_call_ms` is a merged-graph compile for the whole "
                     "population multiset (expect seconds to minutes) and is excluded from the warm "
                     "medians by protocol — compare the warm columns only.")
    if campaign is None:
        lines.append("- No `campaign.json`: schedule completeness, the drop list and the guard "
                     "record could not be checked; only the record files present on disk are "
                     "tabulated above.")
    elif not bad:
        lines.append("- Every scheduled config in `campaign.json` finished `ok`.")
    else:
        lines.append(f"- **{len(bad)} non-`ok` config(s)** in `campaign.json`: "
                     "the record for these is absent or partial and no table cell above is a "
                     "finished measurement.")
        lines.append("")
        lines.extend(table(
            ["cell", "arm", "status", "detail", "wall_s"],
            [[f"`{entry.get('cell')}`", entry.get("arm"), entry.get("status"),
              f"`{entry.get('detail')}`" if entry.get("detail") else MISSING,
              fmt_ms(entry.get("wall_s"))] for entry in bad],
        ))
        lines.append("")
    if problems:
        lines.append(f"- **{len(problems)} record file(s) absent or unreadable** "
                     "(rendered as `-` above):")
        lines.append("")
        lines.extend(table(
            ["cell", "arm", "reason", "campaign status"],
            [[f"`{cell}`", arm, reason,
              (entry or {}).get("status", MISSING)] for cell, arm, reason, entry in problems],
        ))
        lines.append("")
    if contaminated:
        lines.append(f"- **{len(contaminated)} config(s) ran on a contaminated box** "
                     "(`guard.contaminated: true`, i.e. a foreign compute process or a busy device "
                     "was seen inside the guard wait); their timings are upper bounds:")
        lines.append("")
        for entry in contaminated:
            guard = entry.get("guard") or {}
            pids = (guard.get("gpu_pids") or {}).get("pids") or []
            lines.append(f"  - `{entry.get('cell')}` / {entry.get('arm')}: waited "
                         f"{guard.get('waited_s')} s, gpu pids {pids or '-'}, "
                         f"python {guard.get('python') or '-'}")
        lines.append("")
    scope_dropped = ((campaign or {}).get("protocol") or {}).get("scope_dropped") or []
    if scope_dropped:
        lines.append(f"- **{len(scope_dropped)} config(s) never ran**: excluded by the routing "
                     "scope rule for compile cost, so they have no row anywhere above.")
        lines.append("")
        lines.extend(table(
            ["cell", "arm", "reason"],
            [[f"`{entry.get('cell')}`", entry.get("arm"), "dropped by the routing scope rule"]
             for entry in scope_dropped if isinstance(entry, dict)],
        ))
        lines.append("")
    lines.append("")
    return lines


# --- driver --------------------------------------------------------------------
def parse_sections(text: str) -> List[str]:
    if not text or text.strip() == "all":
        return list(SECTIONS)
    wanted = [token.strip() for token in text.split(",") if token.strip()]
    unknown = [token for token in wanted if token not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown --markdown-tables value(s) {unknown}; known: {list(SECTIONS)}")
    return [name for name in SECTIONS if name in wanted]


def load_compare(compare_dir: str, label: str) -> Dict[str, Any]:
    """The second result directory of a before/after render (records only)."""
    if not os.path.isdir(compare_dir):
        raise SystemExit(f"--compare-results {compare_dir!r} is not a directory")
    _, error, _, records, problems = load_results(compare_dir)
    return {"dir": compare_dir, "label": label, "records": records,
            "error": error, "problems": problems}


def build(args) -> Tuple[List[str], Dict[str, Any]]:
    """Render the document and hand the loaded data back for the console summary."""
    campaign, campaign_error, configs, records, problems = load_results(args.results)
    cells = order_cells({cell for cell, _ in records})
    arms = order_arms({arm for _, arm in records})
    workloads = [name.strip() for name in (args.workloads or "").split(",")
                 if name.strip() and name.strip() != "gp_step"]
    sections = parse_sections(args.markdown_tables)
    compare = (load_compare(args.compare_results, args.compare_label)
               if args.compare_results else None)
    compare_arms = arms if compare is None else order_arms(
        set(arms) | {arm for _, arm in compare["records"]})

    lines = ["# evoxc benchmark summary: " + os.path.basename(os.path.normpath(args.results)), ""]
    lines.extend(protocol_lines(args.results, campaign, campaign_error, configs, records, cells, arms))
    if "env" in sections:
        lines.extend(env_lines(campaign, records))
    if "compare" in sections and compare is not None and not shared_cells(cells, compare["records"]):
        lines.extend([f"## Compare vs {compare['label']}", "",
                      f"No cell is present in both `{args.results}` and "
                      f"`{compare['dir']}`, so no compare table is rendered.", ""])
    for name in workloads:
        if "workloads" in sections:
            lines.extend(workload_lines(name, cells, arms, records))
        if "speedup" in sections:
            lines.extend(speedup_lines(name, cells, arms, records))
        if "compare" in sections and compare is not None:
            lines.extend(compare_workload_lines(name, cells, compare_arms, records,
                                                args.results, compare))
    if "gp_step" in sections:
        lines.extend(gp_step_lines(cells, arms, records))
        if "compare" in sections and compare is not None:
            lines.extend(compare_gp_step_lines(cells, compare_arms, records,
                                               args.results, compare))
    if "cold" in sections:
        lines.extend(cold_lines(cells, arms, records))
    if "digest" in sections:
        lines.extend(digest_lines(cells, arms, records))
    # caveats are never optional: they are the honesty of the report
    lines.extend(caveat_lines(campaign, configs, problems,
                              batch_first_call_present(cells, arms, records)))
    data = {"campaign": campaign, "campaign_error": campaign_error, "configs": configs,
            "records": records, "problems": problems, "cells": cells, "arms": arms,
            "compare": compare}
    return lines, data


def console_summary(args, data: Dict[str, Any], line_count: int) -> List[str]:
    campaign, configs, records = data["campaign"], data["configs"], data["records"]
    cells, arms = data["cells"], data["arms"]
    out = [f"[summarize] {len(records)} records / {len(cells)} cells / {len(arms)} arms "
           f"from {args.results}"]
    if data["campaign_error"]:
        out.append(f"[summarize] WARNING no usable campaign.json: {data['campaign_error']}")
    if isinstance(campaign, dict):
        out.append(f"[summarize] campaign: {len(configs)} configs, {campaign.get('failures')} "
                   f"failures, {campaign.get('elapsed_s')} s")
    bad = [entry for entry in configs.values() if entry.get("status") != "ok"]
    contaminated = [entry for entry in configs.values()
                    if isinstance(entry.get("guard"), dict) and entry["guard"].get("contaminated")]
    out.append(f"[summarize] caveats: {len(bad)} non-ok config(s), {len(data['problems'])} absent "
               f"record(s), {len(contaminated)} contaminated guard(s)")
    compare = data.get("compare")
    if compare is not None:
        shared = shared_cells(cells, compare["records"])
        out.append(f"[summarize] compare vs {compare['label']} ({compare['dir']}): "
                   f"{len(compare['records'])} records, {len(shared)}/{len(cells)} cell(s) shared"
                   + (" — no cell in both, compare tables skipped" if not shared else ""))
    for name in DEFAULT_WORKLOADS:
        counts: Dict[str, int] = {}
        for cell in cells:
            best = fastest_arm(records, cell, arms, name)
            if best != MISSING:
                counts[best] = counts.get(best, 0) + 1
        if counts:
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            out.append(f"[summarize] {name}: fastest per cell -> "
                       + ", ".join(f"{arm} {count}" for arm, count in ranked))
    out.append(f"[summarize] wrote {args.out} ({line_count} lines)")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True, help="result directory (per-cell JSON records)")
    parser.add_argument("--out", required=True, help="markdown file to write")
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS),
                        help=f"comma-separated workload sections (default {','.join(DEFAULT_WORKLOADS)})")
    parser.add_argument("--markdown-tables", default="all",
                        help=f"'all' or a comma-separated subset of {','.join(SECTIONS)}")
    parser.add_argument("--compare-results", default=None,
                        help="second result directory to render the before/after ratio tables "
                             "against (same session; the primary is --results)")
    parser.add_argument("--compare-label", default="before",
                        help="name the compare directory carries in the tables (default 'before')")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.results):
        raise SystemExit(f"--results {args.results!r} is not a directory")
    lines, data = build(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip("\n") + "\n")
    for line in console_summary(args, data, len(lines)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
