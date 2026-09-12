#!/usr/bin/env python3
"""compare_rounds.py -- R2-vs-R3 table generator for the round-3 benchmark records.

Round 3 re-measured the R2 headline configs with the R2 methodology (same harness, flags,
fixtures, wiring), and the R3 ids intentionally match their R2 counterparts, so the two rounds
can be diffed id by id.  This script is the *sourcing tool for ``SUMMARY.md``*: it prints

1. the core **eval** configs -- ``first_call_ms`` (compile/first call, reported separately) and the
   warm per-call median;
2. the core **training** configs (``sr``) -- per-rep warm per-generation medians (rep0 "churning",
   rep1 "fully warm", the REPORT.md convention) plus each rep's cold first generation;
3. the **wave-attribution** table -- the three ``*_spec`` arms (``EVOGP_EVAL_MODE=specialized``,
   i.e. the R2-era route) next to their R2 counterpart and their R3 ``auto`` (dense-route) twin.

Metric rules are the round's own (identical to the R2/R3 validators): the warm median is
``median_warm_ms`` when present, else the median of ``per_call_ms[1:]``; a training rep's warm
figure is the median of ``per_gen_ms[1:]`` of that rep, and its cold figure is that rep's
``cold_first_gen_ms`` (else ``per_gen_ms[0]``).  Nothing is recomputed from scratch and nothing is
asserted here -- for the pass/fail gates see ``check_r3.py`` and ``check_agreement_r3.py``.

Stdlib only (no torch, no numpy, no GPU, no evogp).  Missing records print as ``MISSING`` in the
cells and never abort the report (the round may still be mid-flight); the exit code is 0 unless
nothing could be read at all.

Usage (paths are resolved from this file, so any cwd works)::

    python3 test/benchmark/r2/round3/compare_rounds.py              # plain aligned text
    python3 test/benchmark/r2/round3/compare_rounds.py --markdown   # markdown tables for SUMMARY.md
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir, os.pardir))

#: where each (round, side) keeps its records, relative to the repository root
ROUND_DIRS = {
    ("r2", "new"): os.path.join("test", "benchmark", "new", "results", "r2"),
    ("r2", "leg"): os.path.join("test", "benchmark", "legacy", "results", "r2"),
    ("r3", "new"): os.path.join("test", "benchmark", "r2", "round3", "results", "new"),
    ("r3", "leg"): os.path.join("test", "benchmark", "r2", "round3", "results", "legacy"),
}

#: the R3 config table, in run order -- (id, side) where side is the PYTHONPATH side
CORE_IDS = (
    ("leg_c5r2", "leg"),
    ("leg_fixture200_n64", "leg"),
    ("leg_sr_a", "leg"),
    ("np_c5r2", "new"),
    ("np_fixture200_n64", "new"),
    ("np_sr_a", "new"),
    ("iree_fixture64_n64", "new"),
    ("iree_fixture200_n64", "new"),
    ("iree_sr_a", "new"),
)

#: the wave-attribution arms -- (spec id, side, R3 `auto` twin = the id minus the `_spec` suffix)
SPEC_IDS = (
    ("iree_fixture64_n64_spec", "new", "iree_fixture64_n64"),
    ("iree_fixture200_n64_spec", "new", "iree_fixture200_n64"),
    ("iree_sr_a_spec", "new", "iree_sr_a"),
)

TRAIN_REP_LABELS = {0: "churning", 1: "fully warm"}


# ---------------------------------------------------------------------------
# loading / metric extraction (never raises)
# ---------------------------------------------------------------------------
def record_path(round_id, side, id_):
    return os.path.join(REPO_ROOT, ROUND_DIRS[(round_id, side)], f"{id_}.json")


def load(round_id, side, id_):
    """(payload, status) with status in ok|skipped|MISSING|MALFORMED."""
    path = record_path(round_id, side, id_)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None, "MISSING"
    except (OSError, ValueError):
        return None, "MALFORMED"
    if not isinstance(payload, dict):
        return None, "MALFORMED"
    if payload.get("skipped"):
        return payload, "skipped"
    return payload, "ok"


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def median_of(values):
    numbers = [v for v in (values or []) if is_number(v)]
    return statistics.median(numbers) if numbers else None


def median_warm(values):
    """The validators' rule: median of ``values[1:]``, only when >= 2 warm entries exist."""
    warm = [v for v in (values or [])[1:] if is_number(v)]
    return statistics.median(warm) if len(warm) >= 2 else None


def task_kind(payload):
    task = payload.get("task")
    if task in ("sr", "xor"):
        return "train"
    if task in ("eval64", "evalfresh"):
        return "eval"
    if "per_gen_ms" in payload:
        return "train"
    if "per_call_ms" in payload:
        return "eval"
    return None


def eval_metrics(payload):
    """(first_call_ms, warm_ms) for an eval record, by the check scripts' rule."""
    first = payload.get("first_call_ms")
    if not is_number(first):
        calls = payload.get("per_call_ms") or []
        first = calls[0] if calls and is_number(calls[0]) else None
    warm = payload.get("median_warm_ms")
    if not is_number(warm):
        warm = median_warm(payload.get("per_call_ms"))
    return first, warm


def train_reps(payload):
    """[{'rep', 'label', 'warm', 'cold'}] -- one entry per rep found in the record."""
    reps = payload.get("per_rep")
    entries = []
    if isinstance(reps, list) and reps:
        for index, rep in enumerate(reps):
            if not isinstance(rep, dict):
                continue
            gens = rep.get("per_gen_ms") or []
            cold = rep.get("cold_first_gen_ms")
            if not is_number(cold):
                cold = gens[0] if gens and is_number(gens[0]) else None
            warm = median_of(gens[1:])
            rep_no = rep.get("rep", index)
            entries.append({"rep": rep_no, "label": TRAIN_REP_LABELS.get(rep_no, f"rep{rep_no}"),
                            "warm": warm, "cold": cold})
    if not entries:                      # fallback: the top-level arrays describe rep 0
        gens = payload.get("per_gen_ms") or []
        cold = payload.get("cold_first_gen_ms")
        if not is_number(cold):
            cold = gens[0] if gens and is_number(gens[0]) else None
        entries.append({"rep": 0, "label": TRAIN_REP_LABELS[0], "warm": median_of(gens[1:]),
                        "cold": cold})
    return entries


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def num(value, digits=2):
    if not is_number(value):
        return "-"
    if abs(value) >= 10000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def delta(new, old):
    """Signed percentage change new-vs-old (``-`` when either side is unknown)."""
    if not is_number(new) or not is_number(old) or old == 0:
        return "-"
    return f"{(new - old) / old * 100.0:+.1f}%"


def ratio(new, old):
    if not is_number(new) or not is_number(old) or old == 0:
        return "-"
    return f"{new / old:.2f}x"


def cell(payload, status, extract):
    """extract(payload) for an ok record, else the status word."""
    if status != "ok" or payload is None:
        return status
    return extract(payload)


def render(title, headers, rows, markdown):
    print()
    print(f"## {title}" if markdown else f"== {title} ==")
    if markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(str(c) for c in row) + " |")
        return
    widths = [max([len(str(headers[i]))] + [len(str(row[i])) for row in rows])
              for i in range(len(headers))]
    header = "  ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def eval_table(markdown, missing):
    headers = ("id", "side", "R2 first_ms", "R2 warm_ms", "R3 first_ms", "R3 warm_ms",
               "warm delta", "R2 status", "R3 status")
    rows = []
    for id_, side in CORE_IDS:
        r2, r2_status = load("r2", side, id_)
        r3, r3_status = load("r3", side, id_)
        if r2 is None:
            missing.append(f"{id_}: R2 {r2_status} ({record_path('r2', side, id_)})")
        if r3 is None:
            missing.append(f"{id_}: R3 {r3_status} ({record_path('r3', side, id_)})")
        kind = task_kind(r2) if r2 is not None else None
        if r2 is not None and kind == "train":
            continue                     # training ids live in their own table
        if r2 is None and id_.endswith("_sr_a"):
            continue                     # keeps the split stable when only R3 is missing
        r2_first, r2_warm = (None, None) if r2 is None else eval_metrics(r2)
        r3_first, r3_warm = (None, None) if r3 is None else eval_metrics(r3)
        rows.append((id_, side, num(r2_first), num(r2_warm), num(r3_first), num(r3_warm),
                     delta(r3_warm, r2_warm), r2_status, r3_status))
    render("core eval configs -- first call (compile) and warm per-call median [ms]", headers,
           rows, markdown)


def train_table(markdown):
    headers = ("id", "side", "rep", "label", "R2 warm/gen_ms", "R3 warm/gen_ms", "warm delta",
               "R2 cold_1st_gen_ms", "R3 cold_1st_gen_ms")
    rows = []
    for id_, side in CORE_IDS:
        r2, r2_status = load("r2", side, id_)
        r3, r3_status = load("r3", side, id_)
        if r2 is not None and task_kind(r2) != "train":
            continue
        if r2 is None and not id_.endswith("_sr_a"):
            continue
        r2_reps = train_reps(r2) if r2 is not None else []
        r3_reps = train_reps(r3) if r3 is not None else []
        reps = sorted({entry["rep"] for entry in r2_reps + r3_reps})
        if not reps:
            rows.append((id_, side, "?", "-", r2_status, r3_status, "-", "-", "-"))
            continue
        for rep in reps:
            r2_rep = next((e for e in r2_reps if e["rep"] == rep), None)
            r3_rep = next((e for e in r3_reps if e["rep"] == rep), None)
            rows.append((
                id_, side, rep,
                TRAIN_REP_LABELS.get(rep, f"rep{rep}"),
                num(r2_rep["warm"] if r2_rep else None),
                num(r3_rep["warm"] if r3_rep else None),
                delta(r3_rep["warm"] if r3_rep else None, r2_rep["warm"] if r2_rep else None),
                num(r2_rep["cold"] if r2_rep else None),
                num(r3_rep["cold"] if r3_rep else None),
            ))
    render("core training configs (sr) -- warm per-generation median per rep [ms]", headers, rows,
           markdown)


def spec_table(markdown):
    headers = ("spec id", "metric", "R2 auto (base id)", "R3 auto", "R3 spec (specialized)",
               "spec/auto R3")
    rows = []
    for spec_id, side, auto_id in SPEC_IDS:
        r2_base, _ = load("r2", side, auto_id)
        r3_auto, _ = load("r3", side, auto_id)
        r3_spec, spec_status = load("r3", side, spec_id)
        kinds = [task_kind(p) for p in (r2_base, r3_auto, r3_spec) if p is not None]
        kind = "train" if "train" in kinds else ("eval" if kinds else None)
        if kind == "train":
            for index in (0, 1):
                def warm(entry_payload):
                    if entry_payload is None:
                        return None
                    reps = [e for e in train_reps(entry_payload) if e["rep"] == index]
                    return reps[0]["warm"] if reps else None
                r2_warm, auto_warm, spec_warm = warm(r2_base), warm(r3_auto), None
                if r3_spec is not None:
                    spec_warm = warm(r3_spec)
                rows.append((spec_id, f"rep{index} warm/gen_ms",
                             num(r2_warm) if r2_base is not None else "MISSING",
                             num(auto_warm) if r3_auto is not None else "MISSING",
                             num(spec_warm) if r3_spec is not None else spec_status,
                             ratio(spec_warm, auto_warm)))
        else:
            for label, extract in (("first_call_ms", lambda p: eval_metrics(p)[0]),
                                   ("warm_ms", lambda p: eval_metrics(p)[1])):
                r2_value = cell(r2_base, "ok" if r2_base is not None else "MISSING", extract)
                auto_value = cell(r3_auto, "ok" if r3_auto is not None else "MISSING", extract)
                spec_value = cell(r3_spec, spec_status, extract)
                rows.append((spec_id, label,
                             num(r2_value), num(auto_value), num(spec_value),
                             ratio(spec_value, auto_value)))
    render("wave attribution -- forced specialized (R2-era route) vs the R3 auto route", headers,
           rows, markdown)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Print R2-vs-R3 comparison tables for the round-3 benchmark records "
                    "(stdlib only, no torch/numpy/GPU).",
        epilog="Missing records print as MISSING; the script never fails on them.",
    )
    parser.add_argument("--markdown", action="store_true",
                        help="emit markdown tables (for SUMMARY.md) instead of aligned text")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    missing = []
    print("compare_rounds.py -- R2 vs R3 (R2 records are read-only history; R3 lives under round3/)")
    print(f"  repo root: {REPO_ROOT}")
    for key in ("r2", "r3"):
        for side in ("new", "leg"):
            print(f"  {key}/{side:<3}: {ROUND_DIRS[(key, side)]}")
    print("  warm (eval)  = median_warm_ms, else median(per_call_ms[1:])")
    print("  warm (train) = median(per_gen_ms[1:]) per rep; cold = that rep's cold_first_gen_ms")

    eval_table(args.markdown, missing)
    train_table(args.markdown)
    spec_table(args.markdown)

    print()
    print("== notes ==")
    print("  * rep0 is the churning population (it repays the per-structure builds), rep1 is the")
    print("    fully warm one; the R2 headline uses the warm (steady-state) numbers.")
    print("  * compile / first-call cost is reported separately and never mixed into a warm")
    print("    median; a '-' cell means the value is absent from the record, 'MISSING' that the")
    print("    whole record is not on disk yet.")
    if missing:
        print("  * records not readable yet (the round may be mid-flight):")
        for item in sorted(set(missing)):
            print(f"      {item}")
    else:
        print("  * every R2 and R3 record of the 12 ids was readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
