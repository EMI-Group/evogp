#!/usr/bin/env python3
"""check_r2.py -- stdlib-only validator for the R2 GPU-round benchmark JSONs.

Validates the ``bench_gp.py`` artifacts written by ``test/benchmark/r2/run_all.sh``
(**no torch, no GPU, no evogp import** -- safe to run anywhere with python3):

* every file parses as a JSON object;
* the required keys are present for the record's task kind
  (train: ``tag, side, task, config, per_gen_ms, per_gen_best_fitness, complete``;
  eval: ``tag, side, task, config, per_call_ms, first_call_ms, complete``);
* a guard/skip record (``skipped`` / ``skip_reason``) passes and is reported as ``SKIP``;
* with ``--manifest``: every expected id exists (missing id = failure), its path is the
  manifest's path and its ``side`` matches (manifest ``leg`` -> JSON ``side: "old"``);
* with ``--require-stages``: configs the manifest marks ``stages`` (i.e. they run
  ``bench_gp.py --stage-probe``) must carry a non-empty ``stages`` block.

Usage (from the repository root)::

    python3 test/benchmark/r2/check_r2.py                        # scan both r2 result dirs
    python3 test/benchmark/r2/check_r2.py --manifest test/benchmark/r2/manifest.txt --require-stages
    python3 test/benchmark/r2/check_r2.py test/benchmark/new/results      # any dir(s)/file(s)

With a manifest and no positional paths, exactly the manifest's files are checked; with
positional paths, every ``*.json`` under them is checked as well (recursively, hidden
files skipped) and manifest-less files are reported as ``extra``.

Exit code: 0 when nothing failed, 1 otherwise.
"""

import argparse
import json
import os
import re
import statistics
import sys

TRAIN_TASKS = ("sr", "xor")
EVAL_TASKS = ("eval64", "evalfresh")
TRAIN_REQUIRED = ("tag", "side", "task", "config", "per_gen_ms", "per_gen_best_fitness", "complete")
EVAL_REQUIRED = ("tag", "side", "task", "config", "per_call_ms", "first_call_ms", "complete")
BASE_REQUIRED = ("tag", "side", "task", "config", "complete")

DEFAULT_DIRS = ("test/benchmark/new/results/r2", "test/benchmark/legacy/results/r2")
DEFAULT_MANIFEST = "test/benchmark/r2/manifest.txt"
MANIFEST_SIDE = {"leg": "old", "new": "new"}  # manifest side -> JSON `side` field

# tokens that mean "this leaf is a duration / a ratio, not a counter"
TIME_TOKENS = {"ms", "us", "ns", "sec", "secs", "second", "seconds", "time", "times",
               "dur", "duration"}
RATIO_TOKENS = {"rate", "ratio", "pct", "percent", "frac", "fraction", "mean", "median",
                "avg", "average"}
BUILD_TOKENS = {"build", "builds", "built", "compile", "compiles", "compiled"}
MISS_TOKENS = {"miss", "misses", "missed"}


# ---------------------------------------------------------------------------
# tiny check accumulator
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


def _stem(token):
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _stems(path):
    return {_stem(tok) for tok in re.findall(r"[a-z0-9]+", path.lower())}


# ---------------------------------------------------------------------------
# best-effort stage-block reading (the JSON schema of `stages` is the harness's)
# ---------------------------------------------------------------------------


def _leaves(obj, prefix=""):
    """``(path, value)`` pairs; a list of scalars collapses to its length (a counter)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(value, child)
    elif isinstance(obj, (list, tuple)):
        scalars = [i for i in obj if isinstance(i, (int, float)) and not isinstance(i, bool)]
        if not obj or len(scalars) == len(obj):
            yield prefix, len(obj)
        else:
            for index, item in enumerate(obj):
                yield from _leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, obj


def _as_count(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def stage_counters(stages):
    """Extract ``(cold_builds, warm_misses)`` from an unknown-shape ``stages`` block.

    Returns ``((path, count) | None, (path, count) | None, keys)`` where ``keys`` is the
    flattened leaf-path list (for diagnostics).  Never raises.
    """
    if not isinstance(stages, (dict, list, tuple)) or not stages:
        return None, None, []

    cold = miss = None
    cold_score = miss_score = 0
    keys = []
    for path, value in _leaves(stages):
        keys.append(path)
        count = _as_count(value)
        if count is None:
            continue
        stems = _stems(path)
        if stems & TIME_TOKENS or stems & RATIO_TOKENS:
            continue
        if stems & BUILD_TOKENS:
            if "cold" in stems:
                score = 3 + len(stems & {"n", "num", "total", "count"})
            elif not stems & {"warm", "hit"}:
                score = 1  # fallback: an unqualified build counter
            else:
                score = 0
            if score > cold_score:
                cold, cold_score = (path, count), score
        if stems & MISS_TOKENS:
            score = 3 if "warm" in stems else 2
            if score > miss_score:
                miss, miss_score = (path, count), score
    return cold, miss, keys


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class Entry:
    __slots__ = ("id", "side", "path", "stages", "line")

    def __init__(self, id_, side, path, stages, line):
        self.id = id_
        self.side = side
        self.path = path
        self.stages = stages
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
            if len(fields) not in (3, 4):
                checker.check(False, f"{path}:{lineno}: expected 3 or 4 TAB-separated fields, "
                                     f"got {len(fields)}")
                continue
            id_, side, json_path = fields[0].strip(), fields[1].strip(), fields[2].strip()
            stages = len(fields) == 4 and bool(fields[3].strip())
            if side not in MANIFEST_SIDE:
                checker.check(False, f"{path}:{lineno}: side must be leg|new (got {side!r})")
                continue
            if id_ in seen:
                checker.check(False, f"{path}:{lineno}: duplicate id {id_!r} "
                                     f"(first at line {seen[id_]})")
                continue
            seen[id_] = lineno
            if not os.path.isabs(json_path):
                json_path = os.path.normpath(json_path)
            entries.append(Entry(id_, side, json_path, stages, lineno))
    checker.check(bool(entries), f"{path}: manifest lists at least one config")
    return entries


def discover(paths, checker):
    """``*.json`` files under the given dirs/files (recursive, hidden files skipped)."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # ``backend_agreement/`` holds pre-flight backend-agreement records; not R2 config results.
                dirs[:] = [d for d in dirs
                           if not d.startswith(".") and d != "backend_agreement"]
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


def task_kind(payload):
    task = payload.get("task")
    if task in TRAIN_TASKS:
        return "train"
    if task in EVAL_TASKS:
        return "eval"
    if "per_gen_ms" in payload:
        return "train"
    if "per_call_ms" in payload:
        return "eval"
    return None


def is_skipped(payload, kind):
    if payload.get("skipped"):
        return True
    reason = payload.get("skip_reason") or payload.get("skipped_reason")
    timings = payload.get("per_call_ms") or payload.get("per_gen_ms")
    return bool(reason) and not timings


def median_warm(values):
    warm = [v for v in (values or [])[1:] if isinstance(v, (int, float))]
    if len(warm) >= 2:
        return statistics.median(warm)
    return None


def first_of(payload, kind):
    if kind == "eval":
        first = payload.get("first_call_ms")
        if isinstance(first, (int, float)):
            return first
        calls = payload.get("per_call_ms") or []
    else:
        first = payload.get("cold_first_gen_ms")
        if isinstance(first, (int, float)):
            return first
        calls = payload.get("per_gen_ms") or []
    return calls[0] if calls and isinstance(calls[0], (int, float)) else None


def warm_of(payload, kind):
    if kind == "eval":
        warm = payload.get("median_warm_ms")
        if isinstance(warm, (int, float)):
            return warm
        return median_warm(payload.get("per_call_ms"))
    return median_warm(payload.get("per_gen_ms"))


def missing_messages(payload, kind):
    required = EVAL_REQUIRED if kind == "eval" else (TRAIN_REQUIRED if kind == "train" else BASE_REQUIRED)
    return [key for key in required if key not in payload]


def check_record(path, expected, require_stages, checker, notes):
    """Validate one JSON file; return the row dict for the report table."""
    row = {
        "id": expected.id if expected else os.path.splitext(os.path.basename(path))[0],
        "side": expected.side if expected else "?",
        "task": "?",
        "status": "FAIL",
        "first_ms": None,
        "warm_ms": None,
        "cold": None,
        "miss": None,
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

    kind = task_kind(payload)
    row["task"] = str(payload.get("task"))
    row["side"] = str(payload.get("side", row["side"]))

    # -- expected side (manifest only) ---------------------------------------
    if expected:
        checker.check(
            payload.get("side") == MANIFEST_SIDE[expected.side],
            f"{row['id']}: side == {MANIFEST_SIDE[expected.side]!r} for a {expected.side} config "
            f"(got {payload.get('side')!r})",
        )

    # -- guard / skip records pass (they carry no timings) --------------------
    if is_skipped(payload, kind):
        row["status"] = "skipped"
        reason = payload.get("skip_reason") or payload.get("skipped_reason") or "skipped"
        notes.append(f"{row['id']}: SKIP -- {reason}")
        checker.check(payload.get("tag") is not None, f"{row['id']}: skipped record has a tag")
        return row

    # -- required keys --------------------------------------------------------
    if kind is None:
        checker.check(False, f"{row['id']}: unrecognizable record -- unknown task "
                             f"{payload.get('task')!r} and neither per_gen_ms nor per_call_ms "
                             f"({path})")
        row["status"] = "MALFORMED"
        return row
    missing = missing_messages(payload, kind)
    checker.check(not missing, f"{row['id']}: missing required {kind} keys {missing} ({path})")

    # -- timing summary -------------------------------------------------------
    row["first_ms"] = first_of(payload, kind)
    row["warm_ms"] = warm_of(payload, kind)
    complete = payload.get("complete") is True
    partial = bool(payload.get("partial")) or not complete
    row["status"] = "partial" if partial else "complete"

    if kind == "eval":
        calls = payload.get("per_call_ms")
        checker.check(isinstance(calls, list), f"{row['id']}: per_call_ms is a list")
        if not calls:
            checker.check(partial, f"{row['id']}: no per_call_ms entries but partial is not set")
    else:
        gens = payload.get("per_gen_ms")
        best = payload.get("per_gen_best_fitness")
        checker.check(isinstance(gens, list), f"{row['id']}: per_gen_ms is a list")
        checker.check(isinstance(best, list), f"{row['id']}: per_gen_best_fitness is a list")
        if isinstance(gens, list) and isinstance(best, list):
            checker.check(len(gens) == len(best),
                          f"{row['id']}: per_gen_ms and per_gen_best_fitness have equal length")

    if partial:
        notes.append(f"{row['id']}: PARTIAL (complete={payload.get('complete')!r}, "
                     f"partial={bool(payload.get('partial'))})")
    if payload.get("tag") is not None and str(payload.get("tag")).upper() != row["id"].upper():
        notes.append(f"{row['id']}: JSON tag {payload.get('tag')!r} != 'ID.upper()' -- "
                     f"not written by the current r2 runner (prior-round artifact?)")

    # -- stages block ---------------------------------------------------------
    stages = payload.get("stages")
    if stages:
        cold, miss, keys = stage_counters(stages)
        row["cold"] = f"{cold[1]} ({cold[0]})" if cold else "?"
        row["miss"] = f"{miss[1]} ({miss[0]})" if miss else "?"
        if cold is None or miss is None:
            notes.append(f"{row['id']}: stage-counter extraction incomplete "
                         f"(cold={row['cold']}, misses={row['miss']}); stages keys: "
                         f"{', '.join(keys[:12])}{'...' if len(keys) > 12 else ''}")
    elif require_stages and expected is not None and expected.stages:
        checker.check(False, f"{row['id']}: manifest expects a 'stages' block "
                             f"(--stage-probe run) but the JSON has none ({path})")
    if stages and expected is not None and not expected.stages and require_stages:
        notes.append(f"{row['id']}: has a stages block although the manifest marks no probe "
                     f"(A/B control mix-up?)")
    return row


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def fmt_ms(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "-"


def print_table(rows, with_stages):
    id_w = max([len("id")] + [len(r["id"]) for r in rows])
    side_w = max([len("side")] + [len(r["side"]) for r in rows])
    task_w = max([len("task")] + [len(r["task"]) for r in rows])
    status_w = max([len("status")] + [len(r["status"]) for r in rows])
    header = (f"{'id':<{id_w}}  {'side':<{side_w}}  {'task':<{task_w}}  "
              f"{'status':<{status_w}}  {'first_ms':>9}  {'warm_ms':>9}")
    if with_stages:
        header += f"  {'cold_builds':>12}  {'warm_misses':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (f"{row['id']:<{id_w}}  {row['side']:<{side_w}}  {row['task']:<{task_w}}  "
                f"{row['status']:<{status_w}}  {fmt_ms(row['first_ms']):>9}  "
                f"{fmt_ms(row['warm_ms']):>9}")
        if with_stages:
            line += f"  {str(row['cold'] or '-'):>12}  {str(row['miss'] or '-'):>12}"
        print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the R2 benchmark JSONs (stdlib only, no torch/GPU).",
        epilog="With --manifest and no paths, exactly the manifest's files are checked.",
    )
    parser.add_argument("paths", nargs="*", help="result dirs and/or JSON files to check")
    parser.add_argument("--manifest", default=None,
                        help=f"expected-id manifest (default: none; {DEFAULT_MANIFEST} for the "
                             f"R2 round)")
    parser.add_argument("--require-stages", action="store_true",
                        help="require a `stages` block for manifest entries marked `stages`")
    parser.add_argument("--verbose", action="store_true",
                        help="also dump the flattened `stages` keys of every probed record")
    args = parser.parse_args(argv)

    checker = Checker()
    notes = []

    entries = []
    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"check_r2.py: no such manifest: {args.manifest}", file=sys.stderr)
            return 1
        entries = load_manifest(args.manifest, checker)
        if args.require_stages:
            probed = sum(1 for e in entries if e.stages)
            print(f"manifest: {args.manifest} ({len(entries)} expected ids, {probed} expect stages)")
    elif args.require_stages:
        print("check_r2.py: WARNING -- --require-stages without --manifest: no config is known to "
              "expect a probe, so nothing is enforced")

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
        print(f"check_r2.py: nothing to check (no manifest, and no JSON under: "
              f"{', '.join(paths) or 'the default dirs'})")
        return 1

    rows = []
    for expected, path in targets:
        rows.append(check_record(path, expected, args.require_stages, checker, notes))

    print()
    with_stages = any(row["cold"] or row["miss"] for row in rows)
    print_table(rows, with_stages)

    complete = sum(1 for r in rows if r["status"] == "complete")
    partial = sum(1 for r in rows if r["status"] == "partial")
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    missing = sum(1 for r in rows if r["status"] == "MISSING")
    malformed = sum(1 for r in rows if r["status"] == "MALFORMED")
    extras = sum(1 for r in rows if not r["manifest"])

    print()
    print(f"{len(rows)} record(s): {complete} complete, {partial} partial, {skipped} skipped, "
          f"{missing} missing, {malformed} malformed"
          + (f" ({extras} not in the manifest)" if entries else ""))
    print(f"{checker.checks} check(s), {len(checker.failures)} failure(s)")

    if args.verbose:
        for expected, path in targets:
            if not expected:
                continue
            try:
                with open(path) as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                continue
            stages = payload.get("stages")
            if stages:
                _, _, keys = stage_counters(stages)
                print(f"  stages[{expected.id}]: {', '.join(keys) or '<no leaves>'}")
    elif any(row["cold"] == "?" or row["miss"] == "?" for row in rows):
        print("  (stage-counter columns show '?' when the harness's `stages` key names are "
              "unrecognized; re-run with --verbose to dump them)")

    if notes:
        print("\nnotes (not failures):")
        for note in notes:
            print(f"  {note}")

    for failure in checker.failures:
        print(f"  FAIL {failure}")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
