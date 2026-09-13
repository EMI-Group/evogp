#!/usr/bin/env python3
"""check_e2e_r4.py -- stdlib-only validator for the R4 after-round END-TO-END records.

Validates the ``bench_gp.py --task eval64`` artifacts written by
``test/benchmark/r4/run_e2e.sh`` (measurement type 1: the public-API
``problem.evaluate(forest)`` on a fixed population, 25 warmed calls per record) with
**no torch, no numpy, no GPU and no evogp import** -- safe to run anywhere with python3.

Per record (the 14 ids listed in ``test/benchmark/r4/e2e_manifest.txt``):

* the file exists, parses as a JSON object and is the E2E eval task of the new side
  (``task == "eval64"``, ``side == "new"``; a harness-guard record -- ``skipped`` -- is a
  failure, because a skipped record is not a measurement);
* the timed result is structurally complete: ``per_call_ms`` carries the round's 25
  ``--eval-reps`` numbers (all positive), ``first_call_ms == per_call_ms[0]``,
  ``median_warm_ms > 0`` **and equal to ``median(per_call_ms[1:])``** (the harness' own
  warm-median definition) and ``config.eval_reps == 25``;
* identity blocks/fields are present (``evogp_commit``, ``etl_version``, ``config`` / ``env``
  / ``gpu``), the recorded environment matches the manifest's backend/mode columns
  (``env.EVOGP_ETL_BACKEND``; ``env.EVOGP_EVAL_MODE`` -- an ABSENT value is accepted only for
  the ``auto`` cells, where it is the engine default), ``ETL_DISABLE_LOCATIONS == "1"``, and
  the IREE cells record the round's ``EVOGP_ETL_IREE_DEVICE=0`` / ``EVOGP_ETL_IREE_OPT_LEVEL=O1``
  (O2/O3 miscompile CUDA -- anything else is not quotable);
* the record measures the fixture the manifest claims (the ``fixture`` basename plus
  ``config.pop`` = 64 / 200), and the ``--max-cold-builds`` budget is the round's: 400 on the
  fixture200 iree cells, the harness default 300 everywhere else;
* **kind consistency** -- a ``shadow`` record must carry a recorded PYTHONPATH naming ``src``
  AND the patched-ETL shadow checkout; a ``stock`` record must carry exactly ``src`` and no
  shadow path.  Two channels are read, because the harness' JSON has no PYTHONPATH field: the
  record's free-form ``notes`` (``run_e2e.sh`` writes ``... PYTHONPATH=<effective>`` there) and
  the sibling ``<id>.log``'s ``etl:`` identity line (repo, rev, ``etl.__file__``, PYTHONPATH).
  At least one channel must carry the PYTHONPATH, and EVERY recorded value must agree with the
  manifest's kind column.  ``etl_version`` is ``"0.1.0"`` for both checkouts, so it can never
  be the discriminator (the validator prints that as a note).
* cross-record consistency: every present record reports the SAME ``evogp_commit`` (one run
  window = one evogp checkout).  ``etl_version`` may differ (and is reported).

A guard-skipped config leaves no JSON behind, so it is reported MISSING (exit 1) -- re-run
``run_e2e.sh`` once the missing shadow checkout / fixture exists.

Usage (from the repository root)::

    python3 test/benchmark/r4/check_e2e_r4.py
    python3 test/benchmark/r4/check_e2e_r4.py --dir test/benchmark/r4/results/after/e2e
    python3 test/benchmark/r4/check_e2e_r4.py --manifest test/benchmark/r4/e2e_manifest.txt
    python3 test/benchmark/r4/check_e2e_r4.py --shadow-path /tmp/evogp_etl_patched_87c7562

``--dir`` / ``--manifest`` default to this round's canonical paths, resolved from this file's
repository root, so the validator works from any cwd; an explicitly passed relative value is
resolved against the cwd.  Manifest paths are repository-root-relative (the runner's
convention); absolute paths in a manifest are used as they are.  With a manifest, exactly the
manifest's records are checked and any additional ``*.json`` in ``--dir`` is reported and
checked as ``extra``.

Exit code: 0 when nothing failed, 1 otherwise.
"""

import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

TASK = "eval64"
SIDE = "new"
BACKENDS = ("numpy", "iree")
MODES = ("auto", "dynamic", "specialized")
KINDS = ("shadow", "stock")
#: fixture column -> (expected `fixture` basename, expected population = config.pop)
FIXTURES = {
    "fixture64": ("forest_random.npz", 64),
    "fixture200": ("forest200_tree64_random.npz", 200),
}
#: the round's `--eval-reps 25` (the R1-R3 protocol; one process, 25 synchronized evaluates)
EVAL_REPS = 25
#: `--max-cold-builds`: raised only on the fixture200 iree cells (R3 mirror)
COLD_BUILD_BUDGET = 400
COLD_BUILD_DEFAULT = 300

DEFAULT_DIR = os.path.join(REPO_ROOT, "test", "benchmark", "r4", "results", "after", "e2e")
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "test", "benchmark", "r4", "e2e_manifest.txt")
#: the patched-ETL checkout the shadow cells import (run_e2e.sh's SHADOW_DIR default)
DEFAULT_SHADOW_PATH = "/tmp/evogp_etl_patched_87c7562"
#: generic marker of "a patched ETL checkout is on the PYTHONPATH" (the R3 Arm B / R4
#: materialization convention: `evogp_etl_patched_<rev>`).  A stock record must never carry it.
SHADOW_MARKER = "evogp_etl_patched"

PYTHONPATH_RE = re.compile(r"pythonpath\s*=\s*([^\s;'\"]+)", re.IGNORECASE)
ETL_LINE_RE = re.compile(r"^\s*etl:.*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# tiny check accumulator + helpers
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


def fmt_ms(value):
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "-"


def as_dict(value):
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class Entry:
    __slots__ = ("id", "fixture", "backend", "mode", "kind", "path", "line")

    def __init__(self, id_, fixture, backend, mode, kind, path, line):
        self.id = id_
        self.fixture = fixture
        self.backend = backend
        self.mode = mode
        self.kind = kind
        self.path = path
        self.line = line


def load_manifest(path, checker):
    """Read ``id<TAB>fixture<TAB>backend<TAB>mode<TAB>kind<TAB>path`` lines.

    ``#`` comments and blank lines are skipped.  Every malformed line is a failure (the
    manifest is the expectation list -- a line that cannot be parsed is a broken expectation).
    """
    entries = []
    seen = {}
    with open(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 6:
                checker.check(
                    False,
                    f"{path}:{lineno}: expected 6 TAB-separated fields "
                    f"(id, fixture, backend, mode, kind, json path), got {len(fields)}",
                )
                continue
            id_, fixture, backend, mode, kind, json_path = (field.strip() for field in fields)
            ok = True
            if fixture not in FIXTURES:
                checker.check(
                    False,
                    f"{path}:{lineno}: fixture must be one of {list(FIXTURES)} (got {fixture!r})",
                )
                ok = False
            if backend not in BACKENDS:
                checker.check(
                    False,
                    f"{path}:{lineno}: backend must be numpy|iree (got {backend!r})",
                )
                ok = False
            if mode not in MODES:
                checker.check(
                    False, f"{path}:{lineno}: mode must be auto|dynamic|specialized (got {mode!r})"
                )
                ok = False
            if kind not in KINDS:
                checker.check(False, f"{path}:{lineno}: kind must be shadow|stock (got {kind!r})")
                ok = False
            if id_ in seen:
                checker.check(
                    False, f"{path}:{lineno}: duplicate id {id_!r} (first at line {seen[id_]})"
                )
                ok = False
            if not json_path:
                checker.check(False, f"{path}:{lineno}: empty json path")
                ok = False
            if not ok:
                continue
            seen[id_] = lineno
            if os.path.basename(json_path) != f"{id_}.json":
                checker.check(
                    False,
                    f"{path}:{lineno}: path basename must be '<id>.json' "
                    f"(id {id_!r}, path {json_path!r})",
                )
                continue
            if not os.path.isabs(json_path):
                json_path = os.path.normpath(os.path.join(REPO_ROOT, json_path))
            entries.append(Entry(id_, fixture, backend, mode, kind, json_path, lineno))
    checker.check(bool(entries), f"{path}: manifest lists at least one record")
    return entries


def discover(dir_path, checker):
    """``*.json`` files directly under ``dir_path`` (hidden files skipped)."""
    found = []
    try:
        names = sorted(os.listdir(dir_path))
    except OSError as exc:
        checker.check(False, f"{dir_path}: cannot list the results dir ({type(exc).__name__}: {exc})")
        return found
    for name in names:
        if name.startswith(".") or not name.endswith(".json"):
            continue
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# record checks
# ---------------------------------------------------------------------------


def fixture_of(payload):
    """Manifest fixture column implied by the record's ``fixture`` path, or ``None``."""
    base = os.path.basename(str(payload.get("fixture") or ""))
    for name, (filename, _pop) in FIXTURES.items():
        if base == filename:
            return name
    return None


def kind_evidence(payload, path):
    """Every identity string this record carries that may hold a PYTHONPATH.

    Returns ``(texts, log_path, log_found)``: the record's ``notes`` (when set) plus the
    ``etl:`` lines of the sibling ``<id>.log`` (when that log exists).  A record without a log
    is normal for a hand-made file -- then only the notes channel is available.
    """
    texts = []
    notes_value = payload.get("notes")
    if isinstance(notes_value, str) and notes_value.strip():
        texts.append(notes_value)
    log_path = os.path.splitext(path)[0] + ".log"
    log_found = os.path.isfile(log_path)
    if log_found:
        try:
            with open(log_path, errors="replace") as handle:
                log_text = handle.read()
        except OSError:
            log_text = ""
        texts.extend(ETL_LINE_RE.findall(log_text))
    return texts, log_path, log_found


def check_kind(payload, expected, checker, notes, path, shadow_path):
    """Verify the recorded PYTHONPATH (notes + ``etl:`` log line) matches the kind column."""
    rid = expected.id
    texts, log_path, log_found = kind_evidence(payload, path)
    recorded = []
    for text in texts:
        recorded.extend(match.group(1) for match in PYTHONPATH_RE.finditer(text))
    if not log_found:
        notes.append(
            f"{rid}: no sibling log {os.path.basename(log_path)} -- the kind was checked against "
            f"the record's notes only"
        )
    if expected.kind == "shadow":
        evidence = (
            "the record's notes (run_e2e.sh writes 'PYTHONPATH=<effective>') and/or the sibling "
            f"log's 'etl:' identity line must name BOTH 'src' and {shadow_path!r}"
        )
        if not recorded:
            checker.check(False, f"{rid}: kind=shadow but no recorded PYTHONPATH found -- {evidence}")
            return
        components = [[part for part in value.split(":") if part] for value in recorded]
        checker.check(
            all(shadow_path in parts and "src" in parts for parts in components),
            f"{rid}: kind=shadow and every recorded PYTHONPATH is 'src:{shadow_path}' "
            f"({evidence}; got {recorded})",
        )
    else:
        evidence = (
            "the record's notes and/or the sibling log's 'etl:' identity line must record exactly "
            "'src' (= the editable stock etl install) and no patched-ETL path"
        )
        if not recorded:
            checker.check(False, f"{rid}: kind=stock but no recorded PYTHONPATH found -- {evidence}")
            return
        components = [[part for part in value.split(":") if part] for value in recorded]
        checker.check(
            all(parts == ["src"] for parts in components),
            f"{rid}: kind=stock and every recorded PYTHONPATH is exactly 'src' ({evidence}; "
            f"got {recorded})",
        )
        checker.check(
            all(not any(SHADOW_MARKER in part for part in parts) for parts in components),
            f"{rid}: kind=stock but a patched-ETL checkout ({SHADOW_MARKER!r}) is on the recorded "
            f"PYTHONPATH ({recorded}) -- the record would not be a stock number",
        )


def check_record(path, expected, checker, notes, shadow_path):
    """Validate one JSON file; return the row dict for the report table."""
    rid = expected.id if expected else os.path.splitext(os.path.basename(path))[0]
    row = {
        "id": rid,
        "kind": expected.kind if expected else "?",
        "fixture": expected.fixture if expected else "?",
        "backend": expected.backend if expected else "?",
        "mode": expected.mode if expected else "?",
        "status": "FAIL",
        "median_ms": None,
        "first_ms": None,
        "commit": None,
        "etl_version": None,
        "manifest": bool(expected),
        "path": path,
    }

    try:
        with open(path) as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        checker.check(
            False,
            f"{rid}: missing JSON {path} (a guard-skipped record leaves no file -- re-run "
            f"run_e2e.sh once its shadow checkout / fixture exists)",
        )
        row["status"] = "MISSING"
        return row
    except (OSError, ValueError) as exc:
        checker.check(False, f"{rid}: cannot parse {path} ({type(exc).__name__}: {exc})")
        row["status"] = "MALFORMED"
        return row
    if not isinstance(payload, dict):
        checker.check(False, f"{rid}: top-level JSON value is not an object ({path})")
        row["status"] = "MALFORMED"
        return row

    if not row["manifest"]:
        notes.append(f"{rid}: not a manifest id (extra file) -- only the generic checks apply")

    # -- task / side / identity ----------------------------------------------
    checker.check(
        payload.get("task") == TASK, f"{rid}: task == {TASK!r} (got {payload.get('task')!r})"
    )
    checker.check(
        payload.get("side") == SIDE, f"{rid}: side == {SIDE!r} (got {payload.get('side')!r})"
    )
    commit = payload.get("evogp_commit")
    row["commit"] = commit.strip() if isinstance(commit, str) and commit.strip() else None
    checker.check(bool(row["commit"]), f"{rid}: evogp_commit is a non-empty string (got {commit!r})")
    etl_version = payload.get("etl_version")
    checker.check(
        isinstance(etl_version, str) and bool(etl_version.strip()),
        f"{rid}: etl_version is a non-empty string (got {etl_version!r})",
    )
    row["etl_version"] = etl_version.strip() if isinstance(etl_version, str) else None
    tag = payload.get("tag")
    if tag is not None and str(tag).upper() != rid.upper():
        notes.append(
            f"{rid}: JSON tag {tag!r} != 'ID.upper()' -- the record was written with a custom "
            f"--tag (run_e2e.sh uses UPPER(id))"
        )
    if payload.get("skipped") or payload.get("skip_reason"):
        checker.check(
            False,
            f"{rid}: the harness skipped this record (skipped={payload.get('skipped')!r}, "
            f"skip_reason={payload.get('skip_reason')!r}) -- a skipped record is not a measurement",
        )
        row["status"] = "skipped"
        return row

    config = as_dict(payload.get("config"))
    env = as_dict(payload.get("env"))
    gpu = as_dict(payload.get("gpu"))
    checker.check(
        isinstance(payload.get("config"), dict),
        f"{rid}: config block is an object (got {type(payload.get('config')).__name__})",
    )
    checker.check(
        isinstance(payload.get("env"), dict),
        f"{rid}: env block is an object (got {type(payload.get('env')).__name__})",
    )
    checker.check(
        isinstance(payload.get("gpu"), dict),
        f"{rid}: gpu block is an object (got {type(payload.get('gpu')).__name__})",
    )

    # -- completeness ---------------------------------------------------------
    complete = payload.get("complete") is True
    partial = bool(payload.get("partial")) or not complete
    row["status"] = "partial" if partial else "complete"
    if partial:
        notes.append(
            f"{rid}: PARTIAL (complete={payload.get('complete')!r}, partial={bool(payload.get('partial'))}) "
            f"-- structural checks still applied"
        )

    # -- the timed result -----------------------------------------------------
    median = payload.get("median_warm_ms")
    row["median_ms"] = float(median) if is_number(median) else None
    checker.check(positive(median), f"{rid}: median_warm_ms is a positive number (got {median!r})")

    calls = payload.get("per_call_ms")
    checker.check(
        isinstance(calls, list), f"{rid}: per_call_ms is a list (got {type(calls).__name__})"
    )
    if isinstance(calls, list):
        checker.check(
            len(calls) == EVAL_REPS and all(positive(v) for v in calls),
            f"{rid}: per_call_ms carries the round's {EVAL_REPS} --eval-reps numbers, all "
            f"positive (got {len(calls)} entries, first {calls[:3]!r})",
        )
        if calls and is_number(calls[0]):
            row["first_ms"] = float(calls[0])
        checker.check(
            rel_close(payload.get("first_call_ms"), calls[0] if calls else None),
            f"{rid}: first_call_ms == per_call_ms[0] "
            f"({payload.get('first_call_ms')!r} vs {calls[0] if calls else None!r})",
        )
        warm = [float(v) for v in calls[1:] if is_number(v)]
        if is_number(median) and len(warm) >= 2 and len(warm) == len(calls) - 1:
            checker.check(
                rel_close(median, statistics.median(warm)),
                f"{rid}: median_warm_ms == median(per_call_ms[1:]) ({median!r} vs "
                f"{statistics.median(warm)!r}; that is the harness' warm definition)",
            )
        else:
            checker.check(
                False,
                f"{rid}: cannot recompute the warm median from per_call_ms (need >= 2 numeric "
                f"warm entries, got {len(warm)} of {len(calls)})",
            )
    checker.check(
        is_number(config.get("eval_reps")) and int(config["eval_reps"]) == EVAL_REPS,
        f"{rid}: config.eval_reps == {EVAL_REPS} (the round ran --eval-reps {EVAL_REPS}; "
        f"got {config.get('eval_reps')!r})",
    )

    # -- environment vs the manifest ------------------------------------------
    recorded_backend = env.get("EVOGP_ETL_BACKEND")
    recorded_mode = env.get("EVOGP_EVAL_MODE")
    if expected:
        checker.check(
            recorded_backend == expected.backend,
            f"{rid}: env.EVOGP_ETL_BACKEND == {expected.backend!r} (manifest backend; "
            f"got {recorded_backend!r})",
        )
        if recorded_mode is None or str(recorded_mode).strip() == "":
            notes.append(
                f"{rid}: EVOGP_EVAL_MODE is unset -- read as the engine default 'auto' "
                f"(run_e2e.sh sets it explicitly on every record)"
            )
            checker.check(
                expected.mode == "auto",
                f"{rid}: EVOGP_EVAL_MODE unset (engine default 'auto') but the manifest says "
                f"mode={expected.mode!r}",
            )
        else:
            checker.check(
                str(recorded_mode).strip().lower() == expected.mode,
                f"{rid}: env.EVOGP_EVAL_MODE == {expected.mode!r} (manifest mode; "
                f"got {recorded_mode!r})",
            )
    else:
        checker.check(
            recorded_backend in BACKENDS,
            f"{rid}: env.EVOGP_ETL_BACKEND is numpy|iree (got {recorded_backend!r})",
        )
        checker.check(
            str(recorded_mode if recorded_mode is not None else "auto").strip().lower() in MODES,
            f"{rid}: env.EVOGP_EVAL_MODE is auto|dynamic|specialized (got {recorded_mode!r})",
        )

    checker.check(
        str(env.get("ETL_DISABLE_LOCATIONS")) == "1",
        f"{rid}: env.ETL_DISABLE_LOCATIONS == '1' (every record sets it; got "
        f"{env.get('ETL_DISABLE_LOCATIONS')!r})",
    )
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    devices = [part.strip() for part in str(cuda).split(",")] if isinstance(cuda, str) else []
    checker.check(
        bool(devices) and all(part not in ("", "7") for part in devices),
        f"{rid}: env.CUDA_VISIBLE_DEVICES is set and never GPU 7 (got {cuda!r}; GPU 7 has a "
        f"hardware ECC fault)",
    )
    checker.check(
        gpu.get("cuda_visible_devices") == cuda,
        f"{rid}: gpu.cuda_visible_devices == env.CUDA_VISIBLE_DEVICES "
        f"({gpu.get('cuda_visible_devices')!r} vs {cuda!r})",
    )
    effective_backend = (
        expected.backend if expected else (recorded_backend if recorded_backend in BACKENDS else None)
    )
    if effective_backend == "iree":
        checker.check(
            str(env.get("EVOGP_ETL_IREE_DEVICE")) == "0",
            f"{rid}: env.EVOGP_ETL_IREE_DEVICE == '0' (the round's iree-device; got "
            f"{env.get('EVOGP_ETL_IREE_DEVICE')!r})",
        )
        checker.check(
            str(env.get("EVOGP_ETL_IREE_OPT_LEVEL")) == "O1",
            f"{rid}: env.EVOGP_ETL_IREE_OPT_LEVEL == 'O1' (O2/O3 miscompile CUDA; got "
            f"{env.get('EVOGP_ETL_IREE_OPT_LEVEL')!r})",
        )

    # -- which fixture / which cold-build budget -------------------------------
    record_fixture = fixture_of(payload)
    if expected:
        checker.check(
            record_fixture == expected.fixture,
            f"{rid}: fixture is the manifest's {expected.fixture!r} "
            f"(basename {os.path.basename(str(payload.get('fixture') or ''))!r})",
        )
    else:
        checker.check(
            record_fixture is not None,
            f"{rid}: fixture basename is one of "
            f"{sorted(filename for filename, _pop in FIXTURES.values())} "
            f"(got {os.path.basename(str(payload.get('fixture') or ''))!r})",
        )
    if record_fixture is not None:
        _filename, pop = FIXTURES[record_fixture]
        checker.check(
            is_number(config.get("pop")) and int(config["pop"]) == pop,
            f"{rid}: config.pop == {pop} (the {record_fixture} tree count; "
            f"got {config.get('pop')!r})",
        )
    budget = (
        COLD_BUILD_BUDGET
        if (effective_backend == "iree" and (expected.fixture if expected else record_fixture) == "fixture200")
        else COLD_BUILD_DEFAULT
    )
    checker.check(
        is_number(config.get("max_cold_builds")) and int(config["max_cold_builds"]) == budget,
        f"{rid}: config.max_cold_builds == {budget} (--max-cold-builds {COLD_BUILD_BUDGET} is the "
        f"round's budget on the fixture200 iree cells; {COLD_BUILD_DEFAULT} is the harness default "
        f"everywhere else; got {config.get('max_cold_builds')!r})",
    )

    # -- kind: which etl checkout did this record actually import? -------------
    if expected:
        check_kind(payload, expected, checker, notes, path, shadow_path)
    else:
        notes.append(
            f"{rid}: kind unknown (not in the manifest) -- the recorded PYTHONPATH was not checked"
        )
    return row


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def print_table(rows):
    widths = {
        key: max(len(key), *(len(str(row[key])) for row in rows))
        for key in ("id", "kind", "fixture", "backend", "mode", "status")
    }
    header = (
        f"{'id':<{widths['id']}}  {'kind':<{widths['kind']}}  {'fixture':<{widths['fixture']}}  "
        f"{'backend':<{widths['backend']}}  {'mode':<{widths['mode']}}  "
        f"{'status':<{widths['status']}}  {'median_warm_ms':>14}  {'first_call_ms':>13}  commit"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:<{widths['id']}}  {row['kind']:<{widths['kind']}}  "
            f"{row['fixture']:<{widths['fixture']}}  {row['backend']:<{widths['backend']}}  "
            f"{row['mode']:<{widths['mode']}}  {row['status']:<{widths['status']}}  "
            f"{fmt_ms(row['median_ms']):>14}  {fmt_ms(row['first_ms']):>13}  "
            f"{(row['commit'] or '-')[:12]}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the R4 after-round END-TO-END (eval64) records written by "
        "test/benchmark/r4/run_e2e.sh (stdlib only, no torch/numpy/GPU).",
        epilog="With --manifest, exactly the manifest's records are checked; extra *.json in "
        "--dir are reported and checked as 'extra'.",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help=f"results dir holding the records (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"expected-id manifest (6 TAB-separated fields: id, fixture, backend, mode, kind, "
        f"path; default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--shadow-path",
        default=DEFAULT_SHADOW_PATH,
        help=f"the patched-ETL checkout the shadow records must name in their recorded "
        f"PYTHONPATH (default: {DEFAULT_SHADOW_PATH})",
    )
    args = parser.parse_args(argv)

    checker = Checker()
    notes = []
    result_dir = os.path.abspath(args.dir) if args.dir else DEFAULT_DIR
    manifest_path = os.path.abspath(args.manifest) if args.manifest else DEFAULT_MANIFEST

    targets = []  # (expected | None, path)
    covered = set()
    if not os.path.isfile(manifest_path):
        print(f"check_e2e_r4.py: no such manifest: {manifest_path}", file=sys.stderr)
        return 1
    entries = load_manifest(manifest_path, checker)
    print(f"manifest: {manifest_path} ({len(entries)} expected ids)")
    for entry in entries:
        targets.append((entry, entry.path))
        covered.add(os.path.abspath(entry.path))

    print(f"records dir: {result_dir}" + ("" if os.path.isdir(result_dir) else "  (missing)"))
    if not os.path.isdir(result_dir):
        checker.check(
            False,
            f"{result_dir}: no such results directory (run test/benchmark/r4/run_e2e.sh first)",
        )
    else:
        for path in discover(result_dir, checker):
            if os.path.abspath(path) not in covered:
                targets.append((None, path))

    if not targets:
        print(f"check_e2e_r4.py: nothing to check (no record in {result_dir})")
        print(f"{checker.checks} check(s), {len(checker.failures)} failure(s)")
        if checker.failures:
            print(f"\nfailures ({len(checker.failures)}):")
            for failure in checker.failures:
                print(f"  FAIL {failure}")
        return 1

    rows = []
    for expected, path in targets:
        rows.append(check_record(path, expected, checker, notes, args.shadow_path))

    # -- cross-record consistency ---------------------------------------------
    commits = {}
    for row in rows:
        if row["commit"]:
            commits.setdefault(row["commit"], []).append(row["id"])
    if len(commits) > 1:
        detail = "; ".join(
            f"{commit[:12]} <- {', '.join(ids)}" for commit, ids in sorted(commits.items())
        )
        checker.check(
            False,
            f"cross-record: every present record reports the same evogp_commit (got {detail})",
        )
    else:
        only = next(iter(commits), None)
        checker.check(
            True,
            "cross-record: every present record reports the same evogp_commit"
            + (f" ({only[:12]})" if only else " (no record present)"),
        )
    versions = sorted({str(row["etl_version"]) for row in rows if row["etl_version"]})
    kinds = {row["kind"] for row in rows if row["kind"] in KINDS}
    present = any(row["status"] in ("complete", "partial") for row in rows)
    if versions:
        notes.append(
            f"etl_version recorded by the present records: {', '.join(versions)} -- identical for "
            f"the stock and shadow checkouts, so the kind check reads the recorded PYTHONPATH / "
            f"etl identity instead"
        )
    if present and kinds == set(KINDS):
        notes.append(
            "both kinds present: the stock and shadow cells ran in one window (the stock cells "
            "follow their shadow twins in run_e2e.sh's table order)"
        )

    print()
    print_table(rows)

    complete = sum(1 for row in rows if row["status"] == "complete")
    partial = sum(1 for row in rows if row["status"] == "partial")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    missing = sum(1 for row in rows if row["status"] == "MISSING")
    malformed = sum(1 for row in rows if row["status"] == "MALFORMED")
    extras = sum(1 for row in rows if not row["manifest"])

    print()
    print(
        f"{len(rows)} record(s): {complete} complete, {partial} partial, {skipped} skipped, "
        f"{missing} missing, {malformed} malformed"
        + (f" ({extras} not in the manifest)" if extras else "")
    )
    print(f"{checker.checks} check(s), {len(checker.failures)} failure(s)")

    if notes:
        print("\nnotes (not failures):")
        for note in notes:
            print(f"  {note}")

    if checker.failures:
        print(f"\nfailures ({len(checker.failures)}):")
        for failure in checker.failures:
            print(f"  FAIL {failure}")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
