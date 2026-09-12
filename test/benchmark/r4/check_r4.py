#!/usr/bin/env python3
"""check_r4.py -- stdlib-only validator for the R4 pure-forward benchmark JSONs.

Validates the ``bench_gp.py --task eval-pure`` artifacts written by
``test/benchmark/r4/run_pure.sh`` (**no torch, no GPU, no evogp import** -- safe to run
anywhere with python3).  Per record:

* the file parses as a JSON object with ``task == "eval-pure"`` and ``side == "new"``
  (a guard-skipped record -- ``skipped`` / ``skip_reason`` -- passes and is reported SKIP,
  which is how the legacy side would appear if one were ever pointed at this task);
* with ``--manifest``: every expected id exists, its path is the manifest's path, and
  ``pure.mode_resolved`` / ``pure.backend`` match the manifest's mode/backend columns;
* the ``pure`` block is structurally complete: ``invocations > 0``, ``reps >= 1``,
  ``per_invocation_ms`` / ``block_ms`` are lists of ``reps`` positive floats that agree with
  each other (``per_invocation_ms[i] == block_ms[i] / invocations``), and the summary
  numbers (``median`` / ``min`` / ``mean``) are positive with ``min <= median <= mean``
  **and equal to the same statistics recomputed from the list**;
* ``per_population_evaluate_ms == median_per_invocation_ms`` (within 1e-9 relative by
  definition: one invocation IS one population evaluate), and the two normalizations
  (``per_graph_call_ms`` = median / ``n_graph_calls_per_run``,
  ``per_tree_us`` = median * 1000 / ``n_trees``) agree with the median;
* ``n_graph_calls_per_run >= 1``, ``pure.runner_source`` is one of the four recognized
  labels (frozen dense / frozen specialized / fallback dense / fallback specialized),
  ``compile.construct_total_ms > 0``;
* with ``--require-companion``: ``e2e_companion.median_warm_ms > 0`` (and the companion's
  ``per_call_ms`` list has ``eval_reps`` positive entries).

Partial records (``complete`` false or ``partial`` true, e.g. a guard stop) get a note
instead of a failure, but every structural check that can still be made is -- with the
``reps``-length equality relaxed to "no more than ``reps``" blocks.

Usage (from the repository root)::

    python3 test/benchmark/r4/check_r4.py --manifest test/benchmark/r4/manifest.txt --require-companion
    python3 test/benchmark/r4/check_r4.py                     # scan the r4 results dir
    python3 test/benchmark/r4/check_r4.py test/benchmark/r4/results/smoke \
        --manifest test/benchmark/r4/results/smoke/manifest_smoke.txt

With a manifest and no positional paths, exactly the manifest's files are checked; with
positional paths, every ``*.json`` under them is checked as well (recursively, hidden files
skipped) and manifest-less files are reported as ``extra``.

Exit code: 0 when nothing failed, 1 otherwise.
"""

import argparse
import json
import os
import statistics
import sys

TASK = "eval-pure"
SIDE = "new"
MODES = ("dense", "specialized")
MODES_REQUESTED = ("auto", "dense", "specialized", "dynamic")
RUNNER_SOURCES = (
    "frozen-api:build_dense_runner",
    "frozen-api:build_specialized_runner",
    "local-fallback:dense",
    "local-fallback:specialized",
)
N_GRAPH_CALLS_SOURCES = ("measured", "introspected", "assumed-1")

PURE_REQUIRED = (
    "mode_requested",
    "mode_resolved",
    "runner_source",
    "backend",
    "per_row",
    "pop",
    "n_trees",
    "n_const_trees",
    "n_graph_calls_per_run",
    "n_graph_calls_source",
    "invocations",
    "warmup",
    "reps",
    "per_invocation_ms",
    "block_ms",
    "median_per_invocation_ms",
    "min_per_invocation_ms",
    "mean_per_invocation_ms",
    "per_graph_call_ms",
    "per_tree_us",
    "per_population_evaluate_ms",
    "conventions",
)
COMPILE_REQUIRED = (
    "decode_plan_ms",
    "build_ms",
    "construct_total_ms",
    "cache_hits",
    "cache_misses",
    "path",
)

DEFAULT_DIRS = ("test/benchmark/r4/results",)
DEFAULT_MANIFEST = "test/benchmark/r4/manifest.txt"

#: where the pre-campaign plumbing records live; excluded from directory scans (they are
#: validated through results/smoke/manifest_smoke.txt instead)
SMOKE_DIR = "smoke"


# ---------------------------------------------------------------------------
# tiny check accumulator + numeric helpers
# ---------------------------------------------------------------------------


class Checker:
    def __init__(self):
        self.checks = 0
        self.failures = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return bool(condition)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def positive(value):
    return is_number(value) and float(value) > 0.0


def rel_close(a, b, tol=1e-9):
    """``a == b`` within ``tol`` relative (``|a-b| <= tol * max(1, |a|, |b|)``)."""
    if not (is_number(a) and is_number(b)):
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def same_list(values, derived, tol=1e-9):
    """Element-wise ``rel_close`` of two equally long numeric lists (no NaNs allowed)."""
    if not isinstance(values, list) or len(values) != len(derived):
        return False
    return all(rel_close(a, b, tol) for a, b in zip(values, derived))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class Entry:
    __slots__ = ("id", "mode", "backend", "path", "line")

    def __init__(self, id_, mode, backend, path, line):
        self.id = id_
        self.mode = mode
        self.backend = backend
        self.path = path
        self.line = line


def load_manifest(path, checker):
    entries = []
    seen = {}
    with open(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 4:
                checker.check(
                    False,
                    f"{path}:{lineno}: expected 4 TAB-separated fields "
                    f"(id, mode, backend, json path), got {len(fields)}",
                )
                continue
            id_, mode, backend, json_path = (field.strip() for field in fields)
            if mode not in MODES:
                checker.check(
                    False, f"{path}:{lineno}: mode must be dense|specialized (got {mode!r})"
                )
                continue
            if backend not in ("numpy", "iree"):
                checker.check(
                    False, f"{path}:{lineno}: backend must be numpy|iree (got {backend!r})"
                )
                continue
            if id_ in seen:
                checker.check(
                    False,
                    f"{path}:{lineno}: duplicate id {id_!r} (first at line {seen[id_]})",
                )
                continue
            seen[id_] = lineno
            if not os.path.isabs(json_path):
                json_path = os.path.normpath(json_path)
            entries.append(Entry(id_, mode, backend, json_path, lineno))
    checker.check(bool(entries), f"{path}: manifest lists at least one config")
    return entries


def discover(paths, checker):
    """``*.json`` files under the given dirs/files (recursive, hidden files skipped).

    ``smoke/`` is skipped in a directory scan: the pre-campaign plumbing records are
    validated through their own scratch manifest (``results/smoke/manifest_smoke.txt``).
    """
    found = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and not (d == SMOKE_DIR and root == path)
                ]
                for name in sorted(files):
                    if name.startswith(".") or not name.endswith(".json"):
                        continue
                    found.append(os.path.join(root, name))
        elif os.path.isfile(path):
            found.append(path)
        else:
            checker.check(False, f"{path}: no such file or directory")
    return found


# ---------------------------------------------------------------------------
# record checks
# ---------------------------------------------------------------------------


def is_skipped(payload):
    if payload.get("skipped"):
        return True
    reason = payload.get("skip_reason") or payload.get("skipped_reason")
    return bool(reason) and not payload.get("pure")


def check_pure_block(pure, complete, checker, rid):
    """Structural checks of the ``pure`` block; returns a diagnostics dict."""
    info = {"median": None, "graph_ms": None, "runner": None, "source": None, "mode": None}
    missing = [key for key in PURE_REQUIRED if key not in pure]
    checker.check(not missing, f"{rid}: pure block missing keys {missing}")

    info["mode"] = pure.get("mode_resolved")
    info["runner"] = pure.get("runner_source")
    info["source"] = pure.get("n_graph_calls_source")
    info["median"] = pure.get("median_per_invocation_ms")

    checker.check(
        pure.get("mode_requested") in MODES_REQUESTED,
        f"{rid}: pure.mode_requested in {list(MODES_REQUESTED)} "
        f"(got {pure.get('mode_requested')!r})",
    )
    checker.check(
        pure.get("mode_resolved") in MODES,
        f"{rid}: pure.mode_resolved in {list(MODES)} (got {pure.get('mode_resolved')!r})",
    )
    checker.check(
        pure.get("runner_source") in RUNNER_SOURCES,
        f"{rid}: pure.runner_source recognized {list(RUNNER_SOURCES)} "
        f"(got {pure.get('runner_source')!r})",
    )
    checker.check(
        pure.get("backend") in ("numpy", "iree"),
        f"{rid}: pure.backend numpy|iree (got {pure.get('backend')!r})",
    )
    checker.check(
        pure.get("per_row") is False,
        f"{rid}: pure.per_row is false (the pure task is defined per_row=False; "
        f"got {pure.get('per_row')!r})",
    )
    checker.check(
        is_number(pure.get("invocations")) and int(pure["invocations"]) > 0,
        f"{rid}: pure.invocations > 0 (got {pure.get('invocations')!r})",
    )
    checker.check(
        is_number(pure.get("reps")) and int(pure["reps"]) >= 1,
        f"{rid}: pure.reps >= 1 (got {pure.get('reps')!r})",
    )
    checker.check(
        is_number(pure.get("warmup")) and int(pure["warmup"]) >= 0,
        f"{rid}: pure.warmup >= 0 (got {pure.get('warmup')!r})",
    )
    checker.check(
        is_number(pure.get("n_graph_calls_per_run")) and int(pure["n_graph_calls_per_run"]) >= 1,
        f"{rid}: pure.n_graph_calls_per_run >= 1 (got {pure.get('n_graph_calls_per_run')!r})",
    )
    checker.check(
        pure.get("n_graph_calls_source") in N_GRAPH_CALLS_SOURCES,
        f"{rid}: pure.n_graph_calls_source in {list(N_GRAPH_CALLS_SOURCES)} "
        f"(got {pure.get('n_graph_calls_source')!r})",
    )
    checker.check(
        isinstance(pure.get("conventions"), str) and bool(pure.get("conventions")),
        f"{rid}: pure.conventions is a non-empty string",
    )

    # -- per-rep timing lists ------------------------------------------------
    invocations = pure.get("invocations") if is_number(pure.get("invocations")) else None
    reps = pure.get("reps") if is_number(pure.get("reps")) else None
    per_invocation = pure.get("per_invocation_ms")
    blocks = pure.get("block_ms")
    checker.check(
        isinstance(per_invocation, list),
        f"{rid}: pure.per_invocation_ms is a list (got {type(per_invocation).__name__})",
    )
    checker.check(
        isinstance(blocks, list),
        f"{rid}: pure.block_ms is a list (got {type(blocks).__name__})",
    )
    if not (isinstance(per_invocation, list) and isinstance(blocks, list)):
        return info
    entry = f"{rid}: pure.per_invocation_ms and pure.block_ms have equal length"
    checker.check(len(per_invocation) == len(blocks), entry)
    if reps is not None:
        if complete:
            checker.check(
                len(per_invocation) == int(reps),
                f"{rid}: {len(per_invocation)} per-invocation entries for pure.reps={int(reps)} "
                "(complete record must carry every rep)",
            )
        else:
            checker.check(
                0 < len(per_invocation) <= int(reps),
                f"{rid}: {len(per_invocation)} per-invocation entries for pure.reps={int(reps)} "
                "(a partial record carries at most one per finished block)",
            )
    checker.check(
        bool(per_invocation) and all(positive(v) for v in per_invocation),
        f"{rid}: every pure.per_invocation_ms value is a positive number "
        f"(got {per_invocation[:4]!r}...)",
    )
    checker.check(
        bool(blocks) and all(positive(v) for v in blocks),
        f"{rid}: every pure.block_ms value is a positive number (got {blocks[:4]!r}...)",
    )
    if invocations and len(per_invocation) == len(blocks):
        checker.check(
            same_list(per_invocation, [float(b) / int(invocations) for b in blocks], 1e-6),
            f"{rid}: pure.per_invocation_ms[i] == pure.block_ms[i] / invocations",
        )

    # -- summaries ------------------------------------------------------------
    median = pure.get("median_per_invocation_ms")
    info["median"] = median
    info["graph_ms"] = pure.get("per_graph_call_ms")
    checker.check(
        positive(median),
        f"{rid}: pure.median_per_invocation_ms is a positive number (got {median!r})",
    )
    checker.check(
        positive(pure.get("min_per_invocation_ms")),
        f"{rid}: pure.min_per_invocation_ms is a positive number "
        f"(got {pure.get('min_per_invocation_ms')!r})",
    )
    checker.check(
        positive(pure.get("mean_per_invocation_ms")),
        f"{rid}: pure.mean_per_invocation_ms is a positive number "
        f"(got {pure.get('mean_per_invocation_ms')!r})",
    )
    if positive(median) and positive(pure.get("min_per_invocation_ms")) and positive(
        pure.get("mean_per_invocation_ms")
    ):
        checker.check(
            float(pure["min_per_invocation_ms"]) <= float(median) <= float(pure["mean_per_invocation_ms"]),
            f"{rid}: min <= median <= mean of the per-invocation numbers "
            f"({pure['min_per_invocation_ms']} / {median} / {pure['mean_per_invocation_ms']})",
        )
    if complete and per_invocation:
        values = [float(v) for v in per_invocation]
        checker.check(
            rel_close(median, statistics.median(values)),
            f"{rid}: pure.median_per_invocation_ms == median(per_invocation_ms) "
            f"({median} vs {statistics.median(values)})",
        )
        checker.check(
            rel_close(pure.get("min_per_invocation_ms"), min(values)),
            f"{rid}: pure.min_per_invocation_ms == min(per_invocation_ms) "
            f"({pure.get('min_per_invocation_ms')} vs {min(values)})",
        )
        checker.check(
            rel_close(pure.get("mean_per_invocation_ms"), statistics.fmean(values)),
            f"{rid}: pure.mean_per_invocation_ms == mean(per_invocation_ms) "
            f"({pure.get('mean_per_invocation_ms')} vs {statistics.fmean(values)})",
        )

    # -- derived normalizations + the per_population_evaluate identity --------
    checker.check(
        rel_close(pure.get("per_population_evaluate_ms"), median),
        f"{rid}: pure.per_population_evaluate_ms == pure.median_per_invocation_ms "
        f"(got {pure.get('per_population_evaluate_ms')!r} vs {median!r}; one invocation IS one "
        f"population evaluate)",
    )
    calls = pure.get("n_graph_calls_per_run")
    if positive(median) and is_number(calls) and int(calls) >= 1:
        checker.check(
            rel_close(pure.get("per_graph_call_ms"), float(median) / int(calls), 1e-6),
            f"{rid}: pure.per_graph_call_ms == median / n_graph_calls_per_run "
            f"({pure.get('per_graph_call_ms')!r} vs {float(median) / int(calls)!r})",
        )
    trees = pure.get("n_trees")
    if positive(median) and is_number(trees) and int(trees) >= 1:
        checker.check(
            rel_close(pure.get("per_tree_us"), float(median) * 1000.0 / int(trees), 1e-6),
            f"{rid}: pure.per_tree_us == median * 1000 / n_trees "
            f"({pure.get('per_tree_us')!r} vs {float(median) * 1000.0 / int(trees)!r})",
        )
    else:
        checker.check(False, f"{rid}: pure.n_trees >= 1 (got {trees!r})")
    return info


def check_record(path, expected, require_companion, checker, notes):
    """Validate one JSON file; return the row dict for the report table."""
    row = {
        "id": expected.id if expected else os.path.splitext(os.path.basename(path))[0],
        "mode": expected.mode if expected else "?",
        "backend": expected.backend if expected else "?",
        "status": "FAIL",
        "median_ms": None,
        "graph_ms": None,
        "runner": None,
        "manifest": bool(expected),
        "path": path,
    }
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        checker.check(False, f"{row['id']}: missing JSON {path}")
        row["status"] = "MISSING"
        return row
    except (OSError, ValueError) as exc:
        checker.check(False, f"{row['id']}: cannot parse {path} ({type(exc).__name__}: {exc})")
        row["status"] = "MALFORMED"
        return row

    if not isinstance(payload, dict):
        checker.check(False, f"{row['id']}: top-level JSON value is not an object ({path})")
        row["status"] = "MALFORMED"
        return row

    # -- guard / skip records pass (they carry no pure block) ------------------
    if is_skipped(payload):
        row["status"] = "skipped"
        reason = payload.get("skip_reason") or payload.get("skipped_reason") or "skipped"
        notes.append(f"{row['id']}: SKIP -- {reason}")
        checker.check(payload.get("tag") is not None, f"{row['id']}: skipped record has a tag")
        checker.check(
            payload.get("task") == TASK,
            f"{row['id']}: task == {TASK!r} (got {payload.get('task')!r})",
        )
        return row

    # -- identity -------------------------------------------------------------
    checker.check(
        payload.get("task") == TASK,
        f"{row['id']}: task == {TASK!r} (got {payload.get('task')!r})",
    )
    checker.check(
        payload.get("side") == SIDE,
        f"{row['id']}: side == {SIDE!r} (got {payload.get('side')!r})",
    )
    if expected:
        checker.check(
            os.path.abspath(path) == os.path.abspath(expected.path),
            f"{row['id']}: path is the manifest's ({path} vs {expected.path})",
        )
    if payload.get("tag") is not None and str(payload.get("tag")).upper() != str(row["id"]).upper():
        notes.append(
            f"{row['id']}: JSON tag {payload.get('tag')!r} != 'ID.upper()' -- not written by "
            f"the current r4 runner (prior-round artifact?)"
        )

    # -- completeness ---------------------------------------------------------
    complete = payload.get("complete") is True
    partial = bool(payload.get("partial")) or not complete
    row["status"] = "partial" if partial else "complete"
    if partial:
        notes.append(
            f"{row['id']}: PARTIAL (complete={payload.get('complete')!r}, "
            f"partial={bool(payload.get('partial'))}) -- structural checks still applied"
        )
    if payload.get("skipped"):
        notes.append(f"{row['id']}: skipped flag set (detail in skip_reason)")

    # -- pure block -----------------------------------------------------------
    pure = payload.get("pure")
    checker.check(
        isinstance(pure, dict),
        f"{row['id']}: pure block is an object (got {type(pure).__name__})",
    )
    if isinstance(pure, dict):
        info = check_pure_block(pure, complete, checker, row["id"])
        row["median_ms"] = info["median"]
        row["graph_ms"] = info["graph_ms"]
        row["runner"] = info["runner"]
        if expected:
            checker.check(
                pure.get("mode_resolved") == expected.mode,
                f"{row['id']}: pure.mode_resolved == {expected.mode!r} (manifest mode; "
                f"got {pure.get('mode_resolved')!r})",
            )
            checker.check(
                pure.get("backend") == expected.backend,
                f"{row['id']}: pure.backend == {expected.backend!r} (manifest backend; "
                f"got {pure.get('backend')!r})",
            )
            if pure.get("mode_requested") == "auto":
                notes.append(
                    f"{row['id']}: mode_requested=auto (the engine's policy resolved "
                    f"{pure.get('mode_resolved')})"
                )

    # -- compile block --------------------------------------------------------
    compile_block = payload.get("compile")
    checker.check(
        isinstance(compile_block, dict),
        f"{row['id']}: compile block is an object (got {type(compile_block).__name__})",
    )
    if isinstance(compile_block, dict):
        missing = [key for key in COMPILE_REQUIRED if key not in compile_block]
        checker.check(not missing, f"{row['id']}: compile block missing keys {missing}")
        checker.check(
            positive(compile_block.get("construct_total_ms")),
            f"{row['id']}: compile.construct_total_ms > 0 "
            f"(got {compile_block.get('construct_total_ms')!r})",
        )
        checker.check(
            compile_block.get("path") in ("frozen", "fallback", None),
            f"{row['id']}: compile.path frozen|fallback|null (got {compile_block.get('path')!r})",
        )

    # -- end-to-end companion -------------------------------------------------
    companion = payload.get("e2e_companion")
    if companion is not None:
        checker.check(
            isinstance(companion, dict),
            f"{row['id']}: e2e_companion is an object or null "
            f"(got {type(companion).__name__})",
        )
        if isinstance(companion, dict):
            calls = companion.get("per_call_ms")
            reps = companion.get("eval_reps")
            checker.check(
                isinstance(calls, list) and all(positive(v) for v in calls),
                f"{row['id']}: e2e_companion.per_call_ms is a list of positive numbers "
                f"(got {calls!r})",
            )
            if isinstance(calls, list):
                checker.check(
                    is_number(reps) and len(calls) == int(reps),
                    f"{row['id']}: e2e_companion.per_call_ms has eval_reps entries "
                    f"({len(calls)} vs {reps!r})",
                )
                checker.check(
                    rel_close(companion.get("first_call_ms"), calls[0] if calls else None),
                    f"{row['id']}: e2e_companion.first_call_ms == per_call_ms[0]",
                )
            checker.check(
                positive(companion.get("median_warm_ms")),
                f"{row['id']}: e2e_companion.median_warm_ms > 0 "
                f"(got {companion.get('median_warm_ms')!r})",
            )
            checker.check(
                isinstance(companion.get("note"), str) and bool(companion.get("note")),
                f"{row['id']}: e2e_companion.note is a non-empty string",
            )
    elif require_companion:
        checker.check(
            False,
            f"{row['id']}: --require-companion but e2e_companion is null "
            f"(the config did not pass --e2e-companion K?)",
        )
    return row


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def fmt_ms(value):
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def print_table(rows):
    widths = {
        key: max(len(key), *(len(str(row[key])) for row in rows))
        for key in ("id", "mode", "backend", "status", "runner")
    }
    header = (
        f"{'id':<{widths['id']}}  {'mode':<{widths['mode']}}  "
        f"{'backend':<{widths['backend']}}  {'status':<{widths['status']}}  "
        f"{'median_ms':>10}  {'graph_call_ms':>13}  {'runner':<{widths['runner']}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:<{widths['id']}}  {row['mode']:<{widths['mode']}}  "
            f"{row['backend']:<{widths['backend']}}  {row['status']:<{widths['status']}}  "
            f"{fmt_ms(row['median_ms']):>10}  {fmt_ms(row['graph_ms']):>13}  "
            f"{str(row['runner'] or '-'):<{widths['runner']}}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the R4 pure-forward benchmark JSONs (stdlib only, no torch/GPU).",
        epilog="With --manifest and no paths, exactly the manifest's files are checked.",
    )
    parser.add_argument("paths", nargs="*", help="result dirs and/or JSON files to check")
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"expected-id manifest (4 TAB-separated fields: id, mode, backend, path; "
        f"{DEFAULT_MANIFEST} is the round's canonical list)",
    )
    parser.add_argument(
        "--require-companion",
        action="store_true",
        help="require a non-null e2e_companion with median_warm_ms > 0 (the config ran "
        "--e2e-companion K>0)",
    )
    args = parser.parse_args(argv)

    checker = Checker()
    notes = []

    entries = []
    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"check_r4.py: no such manifest: {args.manifest}", file=sys.stderr)
            return 1
        entries = load_manifest(args.manifest, checker)
        print(f"manifest: {args.manifest} ({len(entries)} expected ids)")
    elif args.require_companion:
        print(
            "check_r4.py: WARNING -- --require-companion without --manifest: no config is "
            "known to expect a companion, so nothing is enforced beyond the records present"
        )

    paths = list(args.paths)
    if not paths:
        paths = [] if args.manifest else list(DEFAULT_DIRS)

    targets = []  # (expected | None, path)
    covered = set()
    for entry in entries:
        targets.append((entry, entry.path))
        covered.add(os.path.abspath(entry.path))
    for path in discover(paths, checker):
        if os.path.abspath(path) not in covered:
            targets.append((None, path))

    if not targets:
        print(
            f"check_r4.py: nothing to check (no manifest, and no JSON under: "
            f"{', '.join(paths) or 'the default dirs'})"
        )
        return 1

    rows = []
    for expected, path in targets:
        rows.append(check_record(path, expected, args.require_companion, checker, notes))

    print()
    print_table(rows)

    complete = sum(1 for r in rows if r["status"] == "complete")
    partial = sum(1 for r in rows if r["status"] == "partial")
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    missing = sum(1 for r in rows if r["status"] == "MISSING")
    malformed = sum(1 for r in rows if r["status"] == "MALFORMED")
    extras = sum(1 for r in rows if not r["manifest"])

    print()
    print(
        f"{len(rows)} record(s): {complete} complete, {partial} partial, {skipped} skipped, "
        f"{missing} missing, {malformed} malformed"
        + (f" ({extras} not in the manifest)" if entries else "")
    )
    print(f"{checker.checks} check(s), {len(checker.failures)} failure(s)")

    if notes:
        print("\nnotes (not failures):")
        for note in notes:
            print(f"  {note}")

    for failure in checker.failures:
        print(f"  FAIL {failure}")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
