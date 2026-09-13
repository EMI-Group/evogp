#!/usr/bin/env python3
"""check_r6.py -- stdlib-only, GPU-free acceptance gate for the R6 Triton round.

Recomputes every R6 headline from the committed records (never from the prose), validates
the round's envelope / provenance identities, and gates the numeric-validation evidence
against the *working tree* -- with **no torch, no numpy, no triton, no GPU** and no import
of evogp: safe to run anywhere with ``python3``::

    python3 test/benchmark/r6/check_r6.py            # from the repo root, or any cwd
    python3 test/benchmark/r6/check_r6.py --verbose   # + the per-cell number table

Record sets gated (all immutable, read-only; nothing here is written):

* **primary window** ``results/w1d_20260914/`` -- 10 measurement cells (``leg_*``,
  ``dense_*``, ``triton_*``, ``densepure_*``, ``tritonpure_*``) + ``window_meta.txt`` +
  ``check_deps.txt`` + per-cell ``.log`` (+ the triton driver's ``<cell>.json.log``);
* **anomaly window** ``results/w1c_20260914/`` -- the same 10 cells, structurally validated,
  and the round's *rejection property* recomputed from its records (legacy w1c/w1d > 2x);
* **numeric validation** ``results/validation/validation.json`` (run 1 = record of record for
  the committed kernel revision) and ``results/validation/run2/validation.json`` (replicate);
* **P1 extras** ``results/w1d_20260914/extra/{kernel_host,compile_econ}/``.

House counting: the summary's ``N records`` counts the 14 primary-window JSON records + the
two validation records... precisely: 10 primary cells + 2 validation + 2 compile-econ = 14;
the 10 anomaly cells, the 5 kernel-host fact sheets and the logs/window files are gated too
and reported on the second summary line.

**One documented provenance finding (NOT a failure).** ``validation/validation.json`` (run 1)
pins ``triton_eval.py`` at ``163b5ffd...``, which is exactly the working-tree file's sha256 --
the committed kernel revision is the validated one. ``run2/validation.json`` declares a
*different* ``triton_eval.py`` sha256 (``e155dd13...``) that exists in no committed object, and
its per-op evidence corroborates a semantically different kernel (the ``zero_sign`` mismatch
class is gone, ``denormal_result`` drops 1792 -> 832, and a new ``dev-const-neglzero`` deviation
appears).  The gate therefore asserts what is true and checkable about run 2 -- a well-formed
declared revision that is NOT the committed one, the recorded signature of that divergence, and
bit-identical fixture/goldens headline numbers -- and prints the finding as a note.  See
``provenance:run2_*`` below.

Exit code: 0 when every check passed, 1 otherwise (the failing check id + path + values are
printed).
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
RESULTS = os.path.join(HERE, "results")
PRIMARY = "w1d_20260914"
ANOMALY = "w1c_20260914"
W1D = os.path.join(RESULTS, PRIMARY)
W1C = os.path.join(RESULTS, ANOMALY)
VALIDATION = os.path.join(RESULTS, "validation")
VALIDATION2 = os.path.join(VALIDATION, "run2")
EXTRAS = os.path.join(W1D, "extra")
KERNEL_HOST = os.path.join(EXTRAS, "kernel_host")
COMPILE_ECON = os.path.join(EXTRAS, "compile_econ")
RUNLOG = os.path.join(HERE, "RUNLOG.md")
R5_WINDOW = os.path.abspath(os.path.join(HERE, os.pardir, "r5", "results", "q4_window"))
TRITON_EVAL = os.path.join(HERE, "triton_eval.py")
VALIDATOR = os.path.join(HERE, "validate_triton.py")
ORACLE = os.path.join(REPO_ROOT, "src", "evogp", "tests", "reference_eval.py")
#: the sha256 of triton_eval.py at the validated revision (run 1's declaration + the working tree)
TRITON_EVAL_SHA = "163b5ffdb48b0bde219acd27e8b0eb7761e56a028bd96ec93f129fbb3937c96f"
#: run 2's declared triton_eval.py sha256 -- a different, uncommitted kernel revision
RUN2_TRITON_EVAL_SHA = "e155dd13c03ac4d8a5fb5bf9adb4129324e87f02cd3c3c08a76b56e9233b9c5d"

EVAL_REPS = 25
PURE_BLOCKS = 5
PURE_INVOCATIONS = 100
PURE_WARMUP = 10
COMPANION_REPS = 11
TRITON_E2E_CALLS = 25
TRITON_PURE_CALLS = 11
TRITON_PURE_RUNS = 521
SEAM = "evogp.tree.dense_engine.sr_fitness_dense"
STAGING = "dense_engine.prepare_dense"
TRITON_CACHE = "/tmp/evogp_r6_triton_cache"
LEGACY_ROOT = "/tmp/evogp_legacy_bench"
OP_COVERAGE_LEN = 31
EXACT_OPS = ["VAR", "CONST", "+", "-", "*", "/"]
USED_OPS = ["*", "+", "-", "/", "CONST", "VAR"]
RTOL = 1e-4
ATOL = 1e-4
#: every timestamp in this round is HKT; the triton driver's stamps carry no offset
CAMPAIGN_TZ = dt.timezone(dt.timedelta(hours=8))

FIXTURES = {
    "fixture64": {
        "basename": "forest_random.npz",
        "rel": "test/goldens/forest_random.npz",
        "pop": 64,
        "tree_len": 63,
        "n_rows": 64,
        "values_n": 4096,
        "nan": 24,
    },
    "fixture200": {
        "basename": "forest200_tree64_random.npz",
        "rel": "test/benchmark/fixtures/forest200_tree64_random.npz",
        "pop": 200,
        "tree_len": 64,
        "n_rows": 64,
        "values_n": 12800,
        "nan": 376,
    },
}
#: id -> (side, group, kind, fixture); kind = t1_legacy | t1_dense | t1_triton | t2_dense | t2_triton
CELLS = [
    ("leg_f64", "old", "legacy", "t1_legacy", "fixture64"),
    ("leg_f200", "old", "legacy", "t1_legacy", "fixture200"),
    ("dense_f64", "new", "dense", "t1_dense", "fixture64"),
    ("dense_f200", "new", "dense", "t1_dense", "fixture200"),
    ("triton_f64", "new", "triton", "t1_triton", "fixture64"),
    ("triton_f200", "new", "triton", "t1_triton", "fixture200"),
    ("densepure_f64", "new", "pure", "t2_dense", "fixture64"),
    ("densepure_f200", "new", "pure", "t2_dense", "fixture200"),
    ("tritonpure_f64", "new", "triton", "t2_triton", "fixture64"),
    ("tritonpure_f200", "new", "triton", "t2_triton", "fixture200"),
]
CELL_IDS = [row[0] for row in CELLS]
KIND = {row[0]: row[3] for row in CELLS}
GROUP = {row[0]: row[2] for row in CELLS}
SIDE = {row[0]: row[1] for row in CELLS}
FIXTURE = {row[0]: row[4] for row in CELLS}
#: the six r5 same-window anchors the RUNLOG quotes (r5 record file, r6 cell id, anchor ms)
ANCHORS = [
    ("leg_f64", "q4_leg64.json", 0.6461385637521744, "0.6461", "-1.5"),
    ("leg_f200", "q4_leg200.json", 0.6371848285198212, "0.6372", "-1.6"),
    ("dense_f64", "q4_eval64_iree.json", 5.382280796766281, "5.3823", "-11.1"),
    ("dense_f200", "q4_eval200_iree.json", 5.791207775473595, "5.7912", "+9.1"),
    ("densepure_f64", "q4_pure64_iree.json", 1.2142077, "1.2142", "+9.9"),
    ("densepure_f200", "q4_pure200_iree.json", 1.4612331, "1.4612", "+1.3"),
]
#: headline medians of record, as the RUNLOG prints them (bold, 4 or 5 decimals)
RUNLOG_MEDIAN_FMT = {
    "leg_f64": "%.4f",
    "leg_f200": "%.4f",
    "dense_f64": "%.4f",
    "dense_f200": "%.4f",
    "triton_f64": "%.4f",
    "triton_f200": "%.4f",
    "densepure_f64": "%.4f",
    "densepure_f200": "%.4f",
    "tritonpure_f64": "%.5f",
    "tritonpure_f200": "%.5f",
}
#: unit test: the runlog's primary/anomaly window lines (label -> regex on the RUNLOG)
WINDOW_LINE_RE = re.compile(
    r"\*\*(?P<kind>primary|anomaly) window\*\*[^|]*\|\s*`(?P<label>[^`]+)`\s*\u2014\s*"
    r"(?P<start>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s*\u2192\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})"
)
META_ROW_RE = re.compile(
    r"^#\s+(?P<id>\S+)\s+(?P<group>\S+)\s+(?P<side>\S+)\s+(?P<type>\d+)\s+(?P<fixture>\S+)"
    r"\s+(?P<start>\S+)\s+(?P<end>\S+)\s+(?P<exit>\d+)\s+(?P<status>\S+)\s+(?P<out>\S+)\s*$"
)
DEPS_ROW_RE = re.compile(r"^(?P<probe>[A-Za-z0-9_.-]+)\s+(?P<state>OK|WARN|MISSING|FAIL)\s+(?P<detail>.*)$")


class Checker:
    """Counts checks, collects labelled failures and notes (the ../r4 + ../r5 pattern)."""

    def __init__(self):
        self.checks = 0
        self.failures = []
        self.notes = []

    def check(self, condition, cid, path, message):
        self.checks += 1
        if not condition:
            self.failures.append(f"{cid}: {path}: {message}")
        return bool(condition)

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)


def rel(path):
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:  # pragma: no cover - different drive on exotic setups
        return path


def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite(value):
    return is_num(value) and math.isfinite(float(value))


def positive(value):
    return finite(value) and float(value) > 0.0


def eq(a, b):
    return a == b


def close(a, b, tol=1e-9):
    if not (finite(a) and finite(b)):
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def load_json(checker, path, cid):
    if not os.path.isfile(path):
        checker.check(False, cid, rel(path), "record is missing")
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        checker.check(False, cid, rel(path), f"unreadable JSON ({type(exc).__name__}: {exc})")
        return None
    if not isinstance(data, dict):
        checker.check(False, cid, rel(path), f"top-level JSON must be an object (got {type(data).__name__})")
        return None
    return data


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def sha256_file(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def parse_stamp(text):
    """Parse a record timestamp into an aware datetime.

    ``bench_gp.py`` writes ``%Y-%m-%dT%H:%M:%S+08:00``; the triton driver writes the local
    (HKT) wall clock without an offset.  The whole round ran in ``+08:00``, so a naive stamp
    is anchored to the campaign offset before any comparison (never silently mixed).
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            stamp = dt.datetime.strptime(str(text).strip(), fmt)
        except (ValueError, TypeError):
            continue
        return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=CAMPAIGN_TZ)
    return None


def med(values):
    return statistics.median(values)


def pct_tokens(text):
    """Every ``+/-X.X %`` magnitude quoted in ``text`` (U+2212 normalised)."""
    out = []
    for token in re.findall(r"([\u2212+-]?\d+(?:\.\d+)?)\s*%", text):
        try:
            out.append(abs(float(token.replace("\u2212", "-"))))
        except ValueError:
            continue
    return out


def ratio_tokens(text):
    """Every ``X.XXx`` / ``X.Xx`` magnitude quoted in ``text``."""
    out = []
    for token in re.findall(r"(\d+(?:\.\d+)?)\s*\u00d7", text):
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def quoted_pct(text, value, tol=0.15):
    return any(abs(token - abs(value)) <= tol for token in pct_tokens(text))


def quoted_ratio(text, value, tol=0.005):
    return any(abs(token - value) <= tol * max(1.0, value) for token in ratio_tokens(text))


def ref_value(record, keys):
    """First present, non-None key of ``keys`` in ``record`` (nested via dotted names)."""
    for key in keys:
        node = record
        for part in key.split("."):
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if node is not None:
            return node
    return None


def envelope_ms(record):
    """The headline median of a record: type-1 warm median, or the type-2 per-invocation median."""
    return ref_value(record, ("median_warm_ms", "median_per_invocation_ms", "pure.median_per_invocation_ms"))


# ---------------------------------------------------------------------------
# per-cell checks
# ---------------------------------------------------------------------------
def check_common(rec, cell_id, checker, path, commit):
    """Provenance / identity checks shared by both record shapes."""
    fixture = FIXTURE[cell_id]
    spec = FIXTURES[fixture]
    tag = rec.get("tag")
    checker.check(isinstance(tag, str) and tag == cell_id.upper(), "cell.tag", path,
                  f"tag must be {cell_id.upper()!r} (got {tag!r})")
    side = rec.get("side")
    checker.check(side == SIDE[cell_id], "cell.side", path,
                  f"side must be {SIDE[cell_id]!r} (got {side!r})")
    recorded = rec.get("fixture")
    checker.check(isinstance(recorded, str) and os.path.basename(recorded) == spec["basename"],
                  "cell.fixture.basename", path,
                  f"fixture must be {spec['basename']} (got {recorded!r})")
    norm = os.path.normpath(recorded) if isinstance(recorded, str) else ""
    checker.check(norm.endswith(spec["rel"]), "cell.fixture.path", path,
                  f"fixture path must end with {spec['rel']} (got {recorded!r})")
    commit_value = rec.get("evogp_commit")
    checker.check(isinstance(commit_value, str) and re.fullmatch(r"[0-9a-f]{40}", commit_value) is not None,
                  "cell.evogp_commit.hex", path, f"evogp_commit must be 40 hex chars (got {commit_value!r})")
    checker.check(commit_value == commit, "cell.evogp_commit.equal", path,
                  f"evogp_commit must be the window commit {commit} (got {commit_value})")
    evogp_file = rec.get("evogp_file")
    if SIDE[cell_id] == "old":
        checker.check(isinstance(evogp_file, str)
                      and os.path.normpath(evogp_file).startswith(LEGACY_ROOT),
                      "cell.evogp_file.legacy", path,
                      f"legacy evogp_file must live under {LEGACY_ROOT} (got {evogp_file!r})")
    else:
        checker.check(isinstance(evogp_file, str)
                      and os.path.normpath(evogp_file).endswith(os.path.join("src", "evogp", "__init__.py")),
                      "cell.evogp_file.new", path,
                      f"new-side evogp_file must be <root>/src/evogp/__init__.py (got {evogp_file!r})")
        checker.check(isinstance(evogp_file, str) and not os.path.normpath(evogp_file).startswith(LEGACY_ROOT),
                      "cell.evogp_file.not_legacy", path,
                      f"new-side evogp_file must not be the legacy checkout (got {evogp_file!r})")
    checker.check(rec.get("complete") is True, "cell.complete", path,
                  f"complete must be true (got {rec.get('complete')!r})")
    gpu = rec.get("gpu") if isinstance(rec.get("gpu"), dict) else {}
    cvd = gpu.get("cuda_visible_devices")
    checker.check(cvd == "6", "cell.gpu.cuda_visible_devices", path,
                  f"gpu.cuda_visible_devices must be '6' (got {cvd!r})")
    checker.check(cvd != "7", "cell.gpu.not_gpu7", path,
                  f"GPU 7 carries a hardware ECC fault and must never be used (got {cvd!r})")
    env = rec.get("env") if isinstance(rec.get("env"), dict) else {}
    checker.check(env.get("CUDA_VISIBLE_DEVICES") == "6", "cell.env.CUDA_VISIBLE_DEVICES", path,
                  f"env.CUDA_VISIBLE_DEVICES must be '6' (got {env.get('CUDA_VISIBLE_DEVICES')!r})")
    checker.check(env.get("ETL_DISABLE_LOCATIONS") == "1", "cell.env.ETL_DISABLE_LOCATIONS", path,
                  f"env.ETL_DISABLE_LOCATIONS must be '1' (got {env.get('ETL_DISABLE_LOCATIONS')!r})")
    stamp = parse_stamp(rec.get("started_at"))
    checker.check(stamp is not None, "cell.started_at", path,
                  f"started_at must be a parseable timestamp (got {rec.get('started_at')!r})")
    kind = KIND[cell_id]
    wall = rec.get("wall_s")
    checker.check(wall is None or positive(wall), "cell.wall_s", path,
                  f"wall_s must be null or finite and > 0 (got {wall!r})")
    if kind in ("t1_legacy", "t1_dense"):
        checker.check(positive(wall) and positive(rec.get("harness_wall_s")), "cell.timing", path,
                      f"a type-1 bench_gp cell must record wall_s and harness_wall_s "
                      f"(got {wall!r}/{rec.get('harness_wall_s')!r})")
        checker.check(finite(rec.get("construct_s")) and float(rec["construct_s"]) >= 0.0,
                      "cell.construct_s", path,
                      f"construct_s must be finite and >= 0 (got {rec.get('construct_s')!r})")
    elif kind in ("t1_triton", "t2_triton"):
        checker.check(positive(wall) and positive(rec.get("elapsed_s_total")), "cell.timing", path,
                      f"the triton driver must record wall_s and elapsed_s_total "
                      f"(got {wall!r}/{rec.get('elapsed_s_total')!r})")
    else:  # t2_dense: the pure record keeps the construction cost on its rep, not at top level
        checker.check(positive(rec.get("harness_wall_s")), "cell.harness_wall_s", path,
                      f"a type-2 bench_gp cell must record harness_wall_s (got {rec.get('harness_wall_s')!r})")
        reps = rec.get("per_rep") if isinstance(rec.get("per_rep"), list) else []
        rep0 = reps[0] if reps and isinstance(reps[0], dict) else {}
        checker.check(positive(rep0.get("wall_s")) and positive(rep0.get("construct_s")), "cell.rep_timing", path,
                      f"the pure rep must carry the build-once cost (got wall_s={rep0.get('wall_s')!r}, "
                      f"construct_s={rep0.get('construct_s')!r})")


def check_type1(rec, cell_id, checker, path):
    """Type-1 envelope: 25 sync-bracketed calls, call 0 cold, warm median = median(calls[1:])."""
    triton = KIND[cell_id] == "t1_triton"
    key = "calls_ms" if triton else "per_call_ms"
    calls = rec.get(key)
    checker.check(isinstance(calls, list), f"t1.{key}.present", path,
                  f"{key} must be a JSON array (got {type(calls).__name__})")
    if not isinstance(calls, list):
        return
    checker.check(len(calls) == EVAL_REPS, f"t1.{key}.length", path,
                  f"{key} must hold the round's {EVAL_REPS} calls (got {len(calls)})")
    for index, value in enumerate(calls):
        checker.check(positive(value), f"t1.{key}[{index}]", path,
                      f"call {index} must be finite and > 0 (got {value!r})")
    if not calls or not all(finite(v) for v in calls):
        return
    cold = ref_value(rec, ("cold_call_ms", "first_call_ms"))
    checker.check(finite(cold) and cold == calls[0], "t1.cold_call", path,
                  f"cold/first_call_ms must equal {key}[0]={calls[0]!r} (got {cold!r})")
    warm_median = rec.get("median_warm_ms")
    checker.check(finite(warm_median) and warm_median == med(calls[1:]), "t1.median_warm_ms", path,
                  f"median_warm_ms must be exactly median({key}[1:])={med(calls[1:])!r} (got {warm_median!r})")
    checker.check(positive(warm_median) and finite(cold) and warm_median < cold, "t1.cold_over_warm", path,
                  f"call 0 must be colder than the warm median (cold={cold!r}, warm={warm_median!r})")
    if triton:
        checker.check(rec.get("mean_warm_ms") is not None and close(rec["mean_warm_ms"], statistics.fmean(calls[1:])),
                      "t1.mean_warm_ms", path,
                      f"mean_warm_ms must be mean({key}[1:])={statistics.fmean(calls[1:])!r} (got {rec.get('mean_warm_ms')!r})")
        checker.check(rec.get("min_warm_ms") == min(calls[1:]), "t1.min_warm_ms", path,
                      f"min_warm_ms must be min({key}[1:])={min(calls[1:])!r} (got {rec.get('min_warm_ms')!r})")
        checker.check(rec.get("max_warm_ms") == max(calls[1:]), "t1.max_warm_ms", path,
                      f"max_warm_ms must be max({key}[1:])={max(calls[1:])!r} (got {rec.get('max_warm_ms')!r})")
        checker.check(rec.get("stdev_warm_ms") is not None
                      and close(rec["stdev_warm_ms"], statistics.stdev(calls[1:]), 1e-12),
                      "t1.stdev_warm_ms", path,
                      f"stdev_warm_ms must be the sample stdev of {key}[1:] (got {rec.get('stdev_warm_ms')!r})")
    else:
        cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
        checker.check(cfg.get("eval_reps") == EVAL_REPS, "t1.config.eval_reps", path,
                      f"config.eval_reps must be {EVAL_REPS} (got {cfg.get('eval_reps')!r})")
        checker.check(cfg.get("pop") == FIXTURES[FIXTURE[cell_id]]["pop"], "t1.config.pop", path,
                      f"config.pop must be {FIXTURES[FIXTURE[cell_id]]['pop']} (got {cfg.get('pop')!r})")
        checker.check(cfg.get("tree_len") == FIXTURES[FIXTURE[cell_id]]["tree_len"], "t1.config.tree_len", path,
                      f"config.tree_len must be {FIXTURES[FIXTURE[cell_id]]['tree_len']} (got {cfg.get('tree_len')!r})")
        checker.check(rec.get("skipped") is False, "t1.skipped", path,
                      f"skipped must be false (got {rec.get('skipped')!r})")
        checker.check(rec.get("skip_reason") is None, "t1.skip_reason", path,
                      f"skip_reason must be null (got {rec.get('skip_reason')!r})")
        checker.check(rec.get("partial") is False, "t1.partial", path,
                      f"partial must be false (got {rec.get('partial')!r})")
        checker.check(rec.get("projection") is None, "t1.projection", path,
                      f"projection must be null (got {rec.get('projection')!r})")
        checker.check(rec.get("evogp_commit_error") is None, "t1.evogp_commit_error", path,
                      f"evogp_commit_error must be null (got {rec.get('evogp_commit_error')!r})")
        checker.check(rec.get("nofile_rlimit") == [65536, 65536], "t1.nofile_rlimit", path,
                      f"nofile_rlimit must be [65536, 65536] (got {rec.get('nofile_rlimit')!r})")
        cold_builds = rec.get("cold_builds") if isinstance(rec.get("cold_builds"), dict) else {}
        checker.check(cold_builds.get("exceeded") is False, "t1.cold_builds.exceeded", path,
                      f"cold_builds.exceeded must be false (got {cold_builds.get('exceeded')!r})")
        checker.check(positive(rec.get("harness_wall_s")), "t1.harness_wall_s", path,
                      f"harness_wall_s must be finite and > 0 (got {rec.get('harness_wall_s')!r})")
        checker.check(finite(rec.get("construct_s")) and float(rec.get("construct_s", -1)) >= 0.0,
                      "t1.construct_s", path, f"construct_s must be >= 0 (got {rec.get('construct_s')!r})")
        reps = rec.get("per_rep")
        checker.check(isinstance(reps, list) and len(reps) == 1, "t1.per_rep.length", path,
                      f"a type-1 record holds exactly 1 rep (got {len(reps) if isinstance(reps, list) else reps!r})")
        if isinstance(reps, list) and reps and isinstance(reps[0], dict):
            rep = reps[0]
            checker.check(rep.get("per_call_ms") == calls, "t1.per_rep.calls", path,
                          "per_rep[0].per_call_ms must equal the top-level per_call_ms array")
            checker.check(rep.get("median_warm_ms") == warm_median, "t1.per_rep.median", path,
                          f"per_rep[0].median_warm_ms must equal {warm_median!r} (got {rep.get('median_warm_ms')!r})")
            checker.check(rep.get("first_call_ms") == cold, "t1.per_rep.cold", path,
                          f"per_rep[0].first_call_ms must equal {cold!r} (got {rep.get('first_call_ms')!r})")
    notes = rec.get("notes")
    if isinstance(notes, str):
        for needle, cid in ((f"window={PRIMARY}", "t1.notes.window"), (f"cell={cell_id}", "t1.notes.cell"),
                            (f"group={GROUP[cell_id]}", "t1.notes.group"),
                            (f"fixture={FIXTURE[cell_id]}", "t1.notes.fixture"),
                            ("type=1", "t1.notes.type")):
            checker.check(needle in notes, cid, path, f"notes must carry {needle!r} (got {notes!r})")


def check_triton_identity(rec, cell_id, checker, path):
    """The triton envelope identities: seam, staging, counters, coverage, op table."""
    spec = FIXTURES[FIXTURE[cell_id]]
    pure = KIND[cell_id] == "t2_triton"
    checker.check(rec.get("route") == "triton", "triton.route", path,
                  f"route must be 'triton' (got {rec.get('route')!r})")
    checker.check(rec.get("result_mode") == "loss", "triton.result_mode", path,
                  f"result_mode must be 'loss' (got {rec.get('result_mode')!r})")
    checker.check(rec.get("seam_patched") is True, "triton.seam_patched", path,
                  f"seam_patched must be true (got {rec.get('seam_patched')!r})")
    checker.check(rec.get("patch_target") == SEAM, "triton.patch_target", path,
                  f"patch_target must be {SEAM!r} (got {rec.get('patch_target')!r})")
    checker.check(SEAM in str(rec.get("binding_style", "")), "triton.binding_style", path,
                  f"binding_style must name the rebound seam (got {rec.get('binding_style')!r})")
    checker.check(rec.get("staging") == STAGING, "triton.staging", path,
                  f"staging must be {STAGING!r} (got {rec.get('staging')!r})")
    checker.check(rec.get("launches_per_call") == 1, "triton.launches_per_call", path,
                  f"launches_per_call must be 1 (got {rec.get('launches_per_call')!r})")
    checker.check(rec.get("kernel_name") == "sr_eval_kernel", "triton.kernel_name", path,
                  f"kernel_name must be 'sr_eval_kernel' (got {rec.get('kernel_name')!r})")
    checker.check(str(rec.get("triton_version", "")).startswith("3."), "triton.version", path,
                  f"triton_version must be 3.x (got {rec.get('triton_version')!r})")
    checker.check(rec.get("host") and rec.get("python"), "triton.host_python", path,
                  f"host={rec.get('host')!r} / python={rec.get('python')!r} must be recorded")
    gate = rec.get("counter_gate") if isinstance(rec.get("counter_gate"), dict) else {}
    checker.check(gate.get("passed") is True, "triton.counter_gate.passed", path,
                  f"counter_gate.passed must be true (got {gate.get('passed')!r})")
    checker.check(gate.get("problems") == [], "triton.counter_gate.problems", path,
                  f"counter_gate.problems must be empty (got {gate.get('problems')!r})")
    calls_exp = TRITON_PURE_CALLS if pure else TRITON_E2E_CALLS
    runs_exp = TRITON_PURE_RUNS if pure else TRITON_E2E_CALLS
    hits_exp = TRITON_PURE_CALLS if pure else TRITON_E2E_CALLS - 1
    for field, expected in (("triton_call_count", calls_exp), ("expected_triton_call_count", calls_exp),
                            ("triton_runs", runs_exp), ("expected_triton_runs", runs_exp),
                            ("builds", 1), ("cache_hits", hits_exp), ("data_uploads", 1),
                            ("fallback_count", 0)):
        checker.check(rec.get(field) == expected, f"triton.{field}", path,
                      f"{field} must be {expected} (got {rec.get(field)!r})")
    coverage = rec.get("op_coverage")
    checker.check(isinstance(coverage, list) and len(coverage) == OP_COVERAGE_LEN, "triton.op_coverage.length", path,
                  f"op_coverage must list the round's {OP_COVERAGE_LEN} declarations "
                  f"(got {len(coverage) if isinstance(coverage, list) else coverage!r})")
    checker.check(rec.get("exact_ops") == EXACT_OPS, "triton.exact_ops", path,
                  f"exact_ops must be {EXACT_OPS} (got {rec.get('exact_ops')!r})")
    kernel = rec.get("kernel") if isinstance(rec.get("kernel"), dict) else {}
    if not kernel:
        checker.check(False, "triton.kernel.block", path, "kernel block is missing")
        return
    checker.check(kernel.get("op_coverage") == coverage, "triton.kernel.op_coverage", path,
                  "kernel.op_coverage must equal the top-level op_coverage")
    checker.check(sorted(kernel.get("used_ops") or []) == sorted(USED_OPS), "triton.kernel.used_ops", path,
                  f"used_ops must be exactly {sorted(USED_OPS)} (got {kernel.get('used_ops')!r})")
    checker.check(kernel.get("fallthrough_nodes") == {}, "triton.kernel.fallthrough_nodes", path,
                  f"fallthrough_nodes must be empty (got {kernel.get('fallthrough_nodes')!r})")
    checker.check(kernel.get("uncovered_ops") == [], "triton.kernel.uncovered_ops", path,
                  f"uncovered_ops must be empty (got {kernel.get('uncovered_ops')!r})")
    checker.check(kernel.get("staging") == STAGING, "triton.kernel.staging", path,
                  f"kernel.staging must be {STAGING!r} (got {kernel.get('staging')!r})")
    checker.check(kernel.get("launches_per_call") == 1, "triton.kernel.launches_per_call", path,
                  f"kernel.launches_per_call must be 1 (got {kernel.get('launches_per_call')!r})")
    checker.check(kernel.get("output") == "loss" and kernel.get("use_MSE") is True, "triton.kernel.contract", path,
                  f"kernel must emit the positive MSE loss (got output={kernel.get('output')!r}, "
                  f"use_MSE={kernel.get('use_MSE')!r})")
    checker.check(kernel.get("pop") == spec["pop"], "triton.kernel.pop", path,
                  f"kernel.pop must be {spec['pop']} (got {kernel.get('pop')!r})")
    checker.check(kernel.get("trace_len") == spec["tree_len"], "triton.kernel.trace_len", path,
                  f"kernel.trace_len must be {spec['tree_len']} (got {kernel.get('trace_len')!r})")
    checker.check(kernel.get("n_rows") == spec["n_rows"] and kernel.get("input_len") == 2
                  and kernel.get("output_len") == 1, "triton.kernel.shape", path,
                  f"kernel shape must be n_rows={spec['n_rows']}, input_len=2, output_len=1 "
                  f"(got {kernel.get('n_rows')!r}, {kernel.get('input_len')!r}, {kernel.get('output_len')!r})")
    checker.check(is_num(kernel.get("stage_upload_ms")) and float(kernel["stage_upload_ms"]) >= 0.0,
                  "triton.kernel.stage_upload_ms", path,
                  f"kernel.stage_upload_ms must be >= 0 (got {kernel.get('stage_upload_ms')!r})")
    compile_block = rec.get("compile") if isinstance(rec.get("compile"), dict) else {}
    checker.check(compile_block.get("n_builds") == 1, "triton.compile.n_builds", path,
                  f"compile.n_builds must be 1 (got {compile_block.get('n_builds')!r})")
    checker.check(positive(compile_block.get("compile_ms")), "triton.compile.compile_ms", path,
                  f"compile.compile_ms must be > 0 (got {compile_block.get('compile_ms')!r})")
    checker.check(finite(compile_block.get("stage_upload_ms")) and float(compile_block["stage_upload_ms"]) >= 0.0,
                  "triton.compile.stage_upload_ms", path,
                  f"compile.stage_upload_ms must be >= 0 (got {compile_block.get('stage_upload_ms')!r})")
    if finite(compile_block.get("construct_total_ms")) and finite(compile_block.get("compile_ms")) \
            and finite(compile_block.get("stage_upload_ms")):
        checker.check(close(compile_block["construct_total_ms"],
                            compile_block["compile_ms"] + compile_block["stage_upload_ms"], 1e-6),
                      "triton.compile.construct_total_ms", path,
                      f"construct_total_ms must be compile_ms + stage_upload_ms "
                      f"({compile_block['compile_ms'] + compile_block['stage_upload_ms']!r}, got "
                      f"{compile_block.get('construct_total_ms')!r})")
    if pure:
        checker.check(rec.get("mode") == "pure" and rec.get("task") == "eval-pure", "triton.pure.mode", path,
                      f"mode/task must be pure/eval-pure (got {rec.get('mode')!r}/{rec.get('task')!r})")
        checker.check(compile_block.get("first_call_ms") is None, "triton.pure.first_call_ms", path,
                      f"a pure record has no timed cold call (got {compile_block.get('first_call_ms')!r})")
        checker.check(rec.get("reps_requested") == PURE_BLOCKS, "triton.pure.reps_requested", path,
                      f"reps_requested must be {PURE_BLOCKS} (got {rec.get('reps_requested')!r})")
        checker.check(rec.get("invocations") == PURE_INVOCATIONS, "triton.pure.invocations", path,
                      f"invocations must be {PURE_INVOCATIONS} (got {rec.get('invocations')!r})")
        warmup = rec.get("warmup") if isinstance(rec.get("warmup"), dict) else {}
        checker.check(warmup.get("invocations") == PURE_WARMUP and warmup.get("timed") is False,
                      "triton.pure.warmup", path,
                      f"warmup must be {PURE_WARMUP} discarded invocations (got {warmup!r})")
        blocks = rec.get("blocks_ms")
        per_inv = rec.get("per_invocation_ms")
        checker.check(isinstance(blocks, list) and len(blocks) == PURE_BLOCKS, "triton.pure.blocks_ms.length", path,
                      f"blocks_ms must hold {PURE_BLOCKS} blocks (got {blocks!r})")
        checker.check(isinstance(per_inv, list) and len(per_inv) == PURE_BLOCKS,
                      "triton.pure.per_invocation_ms.length", path,
                      f"per_invocation_ms must hold {PURE_BLOCKS} entries (got {per_inv!r})")
        if isinstance(blocks, list) and isinstance(per_inv, list) and len(blocks) == len(per_inv) == PURE_BLOCKS:
            for index in range(PURE_BLOCKS):
                checker.check(close(blocks[index] / PURE_INVOCATIONS, per_inv[index], 1e-12),
                              f"triton.pure.block[{index}]", path,
                              f"block {index} must be {PURE_INVOCATIONS} x per-invocation "
                              f"({blocks[index] / PURE_INVOCATIONS!r} vs {per_inv[index]!r})")
            checker.check(rec.get("median_block_ms") == med(blocks), "triton.pure.median_block_ms", path,
                          f"median_block_ms must be median(blocks_ms)={med(blocks)!r} "
                          f"(got {rec.get('median_block_ms')!r})")
            checker.check(rec.get("min_block_ms") == min(blocks) and rec.get("mean_block_ms") is not None
                          and close(rec["mean_block_ms"], statistics.fmean(blocks)), "triton.pure.block_stats", path,
                          "min_block_ms/mean_block_ms must be recomputable from blocks_ms "
                          f"(got {rec.get('min_block_ms')!r}/{rec.get('mean_block_ms')!r})")
            checker.check(rec.get("median_per_invocation_ms") == med(per_inv),
                          "triton.pure.median_per_invocation_ms", path,
                          f"median_per_invocation_ms must be median(per_invocation_ms)={med(per_inv)!r} "
                          f"(got {rec.get('median_per_invocation_ms')!r})")
            checker.check(rec.get("min_per_invocation_ms") == min(per_inv)
                          and rec.get("mean_per_invocation_ms") is not None
                          and close(rec["mean_per_invocation_ms"], statistics.fmean(per_inv), 1e-12),
                          "triton.pure.invocation_stats", path,
                          "min/mean per-invocation ms must be recomputable from per_invocation_ms "
                          f"(got {rec.get('min_per_invocation_ms')!r}/{rec.get('mean_per_invocation_ms')!r})")
            checker.check(rec.get("median_warm_ms") == rec.get("median_per_invocation_ms"),
                          "triton.pure.median_warm_ms", path,
                          "median_warm_ms must alias median_per_invocation_ms on a pure record "
                          f"(got {rec.get('median_warm_ms')!r} vs {rec.get('median_per_invocation_ms')!r})")
            spread = max(blocks) / min(blocks) - 1.0
            checker.check(spread <= 0.10, "triton.pure.block_spread", path,
                          f"block spread must be <= 10 % (got {spread * 100:.2f} % over {blocks!r})")
        companion = rec.get("e2e_companion") if isinstance(rec.get("e2e_companion"), dict) else {}
        comp_calls = companion.get("calls_ms")
        checker.check(companion.get("reps") == COMPANION_REPS, "triton.pure.companion.reps", path,
                      f"e2e_companion.reps must be {COMPANION_REPS} (got {companion.get('reps')!r})")
        checker.check(isinstance(comp_calls, list) and len(comp_calls) == COMPANION_REPS,
                      "triton.pure.companion.calls", path,
                      f"e2e_companion.calls_ms must hold {COMPANION_REPS} calls (got {comp_calls!r})")
        if isinstance(comp_calls, list) and len(comp_calls) == COMPANION_REPS and all(finite(v) for v in comp_calls):
            checker.check(companion.get("cold_call_ms") == comp_calls[0], "triton.pure.companion.cold", path,
                          f"companion cold_call_ms must equal calls_ms[0]={comp_calls[0]!r} "
                          f"(got {companion.get('cold_call_ms')!r})")
            checker.check(companion.get("median_warm_ms") == med(comp_calls[1:]),
                          "triton.pure.companion.median", path,
                          f"companion median must be median(calls_ms[1:])={med(comp_calls[1:])!r} "
                          f"(got {companion.get('median_warm_ms')!r})")
            checker.check(companion.get("mean_warm_ms") is not None
                          and close(companion["mean_warm_ms"], statistics.fmean(comp_calls[1:]), 1e-12)
                          and companion.get("min_warm_ms") == min(comp_calls[1:]),
                          "triton.pure.companion.stats", path,
                          "companion mean/min must be recomputable from calls_ms[1:]")
            pure_med = rec.get("median_per_invocation_ms")
            checker.check(finite(pure_med) and finite(companion.get("median_warm_ms"))
                          and companion["median_warm_ms"] > pure_med, "triton.pure.companion_over_pure", path,
                          f"the companion (e2e) median must exceed the pure median "
                          f"({companion.get('median_warm_ms')!r} vs {pure_med!r})")
    else:
        checker.check(rec.get("mode") == "e2e" and rec.get("task") == "eval64", "triton.e2e.mode", path,
                      f"mode/task must be e2e/eval64 (got {rec.get('mode')!r}/{rec.get('task')!r})")
        checker.check(rec.get("reps_requested") == TRITON_E2E_CALLS, "triton.e2e.reps_requested", path,
                      f"reps_requested must be {TRITON_E2E_CALLS} (got {rec.get('reps_requested')!r})")
        checker.check(positive(rec.get("compile", {}).get("first_call_ms")), "triton.e2e.compile.first_call_ms", path,
                      f"compile.first_call_ms must be > 0 (got {compile_block.get('first_call_ms')!r})")
        checker.check(finite(compile_block.get("first_call_ms")) and finite(compile_block.get("compile_ms"))
                      and compile_block["first_call_ms"] > compile_block["compile_ms"],
                      "triton.e2e.first_call_over_compile", path,
                      f"the cold call must include the compile (first_call_ms {compile_block.get('first_call_ms')!r} "
                      f"> compile_ms {compile_block.get('compile_ms')!r})")


def check_pure_dense(rec, cell_id, checker, path):
    """The bench_gp type-2 (pure dense-iree) envelope: 5 blocks x 100 invocations + companion."""
    spec = FIXTURES[FIXTURE[cell_id]]
    pure = rec.get("pure") if isinstance(rec.get("pure"), dict) else {}
    if not pure:
        checker.check(False, "t2d.pure.block", path, "pure block is missing")
        return
    checker.check(rec.get("task") == "eval-pure", "t2d.task", path,
                  f"task must be eval-pure (got {rec.get('task')!r})")
    checker.check(pure.get("mode_requested") == "dense" and pure.get("mode_resolved") == "dense",
                  "t2d.mode", path,
                  f"mode must resolve to dense (got {pure.get('mode_requested')!r}/"
                  f"{pure.get('mode_resolved')!r})")
    checker.check(str(pure.get("runner_source", "")).startswith("frozen-api:"), "t2d.runner_source", path,
                  f"runner_source must be the frozen build-once/run-many API (got {pure.get('runner_source')!r})")
    checker.check(pure.get("backend") == "iree", "t2d.backend", path,
                  f"pure.backend must be 'iree' (got {pure.get('backend')!r})")
    checker.check(pure.get("per_row") is False, "t2d.per_row", path,
                  f"per_row must be false (got {pure.get('per_row')!r})")
    checker.check(pure.get("pop") == spec["pop"] and pure.get("n_trees") == spec["pop"],
                  "t2d.population", path,
                  f"pop/n_trees must be {spec['pop']} (got {pure.get('pop')!r}/{pure.get('n_trees')!r})")
    checker.check(pure.get("n_const_trees") == 0, "t2d.n_const_trees", path,
                  f"n_const_trees must be 0 on these fixtures (got {pure.get('n_const_trees')!r})")
    checker.check(pure.get("n_graph_calls_per_run") == 1, "t2d.n_graph_calls_per_run", path,
                  f"n_graph_calls_per_run must be 1 (got {pure.get('n_graph_calls_per_run')!r})")
    checker.check(pure.get("n_graph_calls_source") == "introspected", "t2d.n_graph_calls_source", path,
                  f"n_graph_calls_source must be 'introspected' (got {pure.get('n_graph_calls_source')!r})")
    checker.check(pure.get("invocations") == PURE_INVOCATIONS, "t2d.invocations", path,
                  f"invocations must be {PURE_INVOCATIONS} (got {pure.get('invocations')!r})")
    checker.check(pure.get("warmup") == PURE_WARMUP, "t2d.warmup", path,
                  f"warmup must be {PURE_WARMUP} (got {pure.get('warmup')!r})")
    checker.check(pure.get("reps") == PURE_BLOCKS, "t2d.reps", path,
                  f"reps must be {PURE_BLOCKS} (got {pure.get('reps')!r})")
    blocks = pure.get("block_ms")
    per_inv = pure.get("per_invocation_ms")
    checker.check(isinstance(blocks, list) and len(blocks) == PURE_BLOCKS, "t2d.block_ms.length", path,
                  f"pure.block_ms must hold {PURE_BLOCKS} blocks of {PURE_INVOCATIONS} invocations (got {blocks!r})")
    checker.check(isinstance(per_inv, list) and len(per_inv) == PURE_BLOCKS, "t2d.per_invocation_ms.length", path,
                  f"pure.per_invocation_ms must hold {PURE_BLOCKS} entries (got {per_inv!r})")
    if isinstance(blocks, list) and isinstance(per_inv, list) and len(blocks) == len(per_inv) == PURE_BLOCKS:
        for index in range(PURE_BLOCKS):
            checker.check(positive(blocks[index]) and positive(per_inv[index])
                          and close(blocks[index] / PURE_INVOCATIONS, per_inv[index], 1e-12),
                          f"t2d.block[{index}]", path,
                          f"block {index} must be {PURE_INVOCATIONS} x per-invocation "
                          f"({blocks[index] / PURE_INVOCATIONS!r} vs {per_inv[index]!r})")
        checker.check(pure.get("median_per_invocation_ms") == med(per_inv), "t2d.median_per_invocation_ms", path,
                      f"median_per_invocation_ms must be median(per_invocation_ms)={med(per_inv)!r} "
                      f"(got {pure.get('median_per_invocation_ms')!r})")
        checker.check(pure.get("min_per_invocation_ms") == min(per_inv)
                      and pure.get("mean_per_invocation_ms") is not None
                      and close(pure["mean_per_invocation_ms"], statistics.fmean(per_inv), 1e-12),
                      "t2d.invocation_stats", path,
                      "pure min/mean per-invocation ms must be recomputable from per_invocation_ms "
                      f"(got {pure.get('min_per_invocation_ms')!r}/{pure.get('mean_per_invocation_ms')!r})")
        steady = blocks[2:]
        spread = max(steady) / min(steady) - 1.0
        checker.check(spread <= 0.10, "t2d.steady_block_spread", path,
                      f"the steady (last 3) blocks must be within 10 % (got {spread * 100:.2f} % over {steady!r})")
        decay = max(blocks) / min(blocks) - 1.0
        checker.check(decay <= 0.40, "t2d.block_spread", path,
                      f"the whole pure block set must stay within 40 % (post-compile warm-up decays; "
                      f"got {decay * 100:.2f} % over {blocks!r})")
        checker.check(blocks[0] >= min(blocks), "t2d.first_block_warmup", path,
                      f"block 0 must not be the fastest block (got {blocks[0]!r} vs min {min(blocks)!r})")
    checker.check(rec.get("median_per_invocation_ms") == pure.get("median_per_invocation_ms"),
                  "t2d.top_level_median", path,
                  "the top-level median_per_invocation_ms must alias the pure block's "
                  f"({rec.get('median_per_invocation_ms')!r} vs {pure.get('median_per_invocation_ms')!r})")
    for key, cid in (("per_graph_call_ms", "t2d.per_graph_call_ms"),
                     ("per_population_evaluate_ms", "t2d.per_population_evaluate_ms")):
        checker.check(pure.get(key) == pure.get("median_per_invocation_ms"), cid, path,
                      f"pure.{key} must equal the median per-invocation ({pure.get(key)!r} vs "
                      f"{pure.get('median_per_invocation_ms')!r})")
    checker.check(finite(pure.get("per_tree_us")) and finite(pure.get("median_per_invocation_ms"))
                  and close(pure["per_tree_us"], pure["median_per_invocation_ms"] * 1000.0 / spec["pop"], 1e-6),
                  "t2d.per_tree_us", path,
                  f"per_tree_us must be median_ms x 1000 / n_trees "
                  f"({pure.get('median_per_invocation_ms', 0) * 1000.0 / spec['pop']!r}, got {pure.get('per_tree_us')!r})")
    reps = rec.get("per_rep")
    checker.check(isinstance(reps, list) and len(reps) == PURE_BLOCKS, "t2d.per_rep.length", path,
                  f"per_rep must hold {PURE_BLOCKS} block entries (got {reps!r})")
    if isinstance(reps, list) and len(reps) == PURE_BLOCKS:
        for index, rep in enumerate(reps):
            rep = rep if isinstance(rep, dict) else {}
            checker.check(rep.get("rep") == index and rep.get("mode_resolved") == "dense"
                          and isinstance(blocks, list) and rep.get("block_ms") == blocks[index],
                          f"t2d.per_rep[{index}]", path,
                          f"per_rep[{index}] must repeat block_ms[{index}] and resolve to dense "
                          f"(got {rep.get('rep')!r}/{rep.get('mode_resolved')!r}/{rep.get('block_ms')!r})")
            checker.check(isinstance(per_inv, list) and rep.get("per_invocation_ms") == per_inv[index]
                          and positive(rep.get("wall_s")), f"t2d.per_rep[{index}].timing", path,
                          f"per_rep[{index}] must repeat per_invocation_ms[{index}] and carry wall_s > 0 "
                          f"(got {rep.get('per_invocation_ms')!r}/{rep.get('wall_s')!r})")
        payload = rec.get("compile") if isinstance(rec.get("compile"), dict) else {}
        checker.check(positive(reps[0].get("construct_s")) and positive(payload.get("construct_total_ms"))
                      and abs(payload["construct_total_ms"] - reps[0]["construct_s"] * 1000.0)
                      <= 0.01 * payload["construct_total_ms"],
                      "t2d.per_rep[0].construct_s", path,
                      f"per_rep[0].construct_s (s) must price the same construction as "
                      f"compile.construct_total_ms within 1 % (got {reps[0].get('construct_s')!r} vs "
                      f"{payload.get('construct_total_ms')!r} ms; the type-2 record leaves the top-level "
                      f"construct_s null = {rec.get('construct_s')!r})")
    checksum = pure.get("output_checksum") if isinstance(pure.get("output_checksum"), dict) else {}
    sha = checksum.get("sha256")
    checker.check(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
                  "t2d.checksum.sha256", path, f"output_checksum.sha256 must be 64 hex chars (got {sha!r})")
    checker.check(checksum.get("total_elems") == spec["values_n"]
                  and checksum.get("shapes") == [[spec["pop"], spec["n_rows"]]],
                  "t2d.checksum.shape", path,
                  f"output_checksum must frame one ({spec['pop']}, {spec['n_rows']}) block "
                  f"(got {checksum.get('shapes')!r}/{checksum.get('total_elems')!r})")
    checker.check(checksum.get("nan_count") == spec["nan"] and checksum.get("inf_count") == 0,
                  "t2d.checksum.specials", path,
                  f"output_checksum must report {spec['nan']} NaN / 0 inf values "
                  f"(got {checksum.get('nan_count')!r}/{checksum.get('inf_count')!r})")
    compile_block = rec.get("compile") if isinstance(rec.get("compile"), dict) else {}
    checker.check(compile_block.get("path") == "frozen", "t2d.compile.path", path,
                  f"compile.path must be 'frozen' for the frozen-API runner (got {compile_block.get('path')!r})")
    checker.check(compile_block.get("cache_misses") == 1 and compile_block.get("cache_hits") == 0,
                  "t2d.compile.cache", path,
                  f"one frozen build, no in-process cache hits (got misses={compile_block.get('cache_misses')!r}, "
                  f"hits={compile_block.get('cache_hits')!r})")
    checker.check(positive(compile_block.get("construct_total_ms")), "t2d.compile.construct_total_ms", path,
                  f"compile.construct_total_ms must be > 0 (got {compile_block.get('construct_total_ms')!r})")
    cold_builds = rec.get("cold_builds") if isinstance(rec.get("cold_builds"), dict) else {}
    checker.check(cold_builds.get("enforced") is True and cold_builds.get("limit") == 400
                  and cold_builds.get("exceeded") is False and is_num(cold_builds.get("misses_last")),
                  "t2d.cold_builds", path,
                  f"the pure cells must run the enforced --max-cold-builds 400 guard (got {cold_builds!r})")
    companion = rec.get("e2e_companion") if isinstance(rec.get("e2e_companion"), dict) else {}
    comp_calls = companion.get("per_call_ms")
    checker.check(companion.get("eval_reps") == COMPANION_REPS
                  and isinstance(comp_calls, list) and len(comp_calls) == COMPANION_REPS,
                  "t2d.companion.length", path,
                  f"e2e_companion must hold {COMPANION_REPS} calls (got {companion.get('eval_reps')!r}/"
                  f"{len(comp_calls) if isinstance(comp_calls, list) else comp_calls!r})")
    if isinstance(comp_calls, list) and len(comp_calls) == COMPANION_REPS and all(finite(v) for v in comp_calls):
        checker.check(companion.get("first_call_ms") == comp_calls[0], "t2d.companion.cold", path,
                      f"companion first_call_ms must equal per_call_ms[0]={comp_calls[0]!r} "
                      f"(got {companion.get('first_call_ms')!r})")
        checker.check(companion.get("median_warm_ms") == med(comp_calls[1:]), "t2d.companion.median", path,
                      f"companion median_warm_ms must be median(per_call_ms[1:])={med(comp_calls[1:])!r} "
                      f"(got {companion.get('median_warm_ms')!r})")
        checker.check(finite(pure.get("median_per_invocation_ms"))
                      and companion["median_warm_ms"] > pure["median_per_invocation_ms"],
                      "t2d.companion_over_pure", path,
                      f"the e2e companion must be slower than the pure block "
                      f"({companion.get('median_warm_ms')!r} vs {pure.get('median_per_invocation_ms')!r})")
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    for key, expected, cid in (("mode", "dense", "t2d.config.mode"), ("invocations", PURE_INVOCATIONS, "t2d.config.invocations"),
                               ("warmup", PURE_WARMUP, "t2d.config.warmup"),
                               ("e2e_companion", COMPANION_REPS, "t2d.config.e2e_companion"),
                               ("backend", "iree", "t2d.config.backend"),
                               ("max_cold_builds", 400, "t2d.config.max_cold_builds"),
                               ("reps", PURE_BLOCKS, "t2d.config.reps"),
                               ("pop", spec["pop"], "t2d.config.pop")):
        checker.check(cfg.get(key) == expected, cid, path,
                      f"config.{key} must be {expected!r} (got {cfg.get(key)!r})")
    checker.check(rec.get("wall_s") is None, "t2d.wall_s_absent", path,
                  f"the pure envelope records no single wall_s (got {rec.get('wall_s')!r})")
    checker.check(positive(rec.get("harness_wall_s")), "t2d.harness_wall_s", path,
                  f"harness_wall_s must be finite and > 0 (got {rec.get('harness_wall_s')!r})")
    checker.check(rec.get("skipped") is False and rec.get("partial") is False, "t2d.not_skipped", path,
                  f"skipped/partial must be false (got {rec.get('skipped')!r}/{rec.get('partial')!r})")
    checker.check(rec.get("nofile_rlimit") == [65536, 65536], "t2d.nofile_rlimit", path,
                  f"nofile_rlimit must be [65536, 65536] (got {rec.get('nofile_rlimit')!r})")


def check_env_roles(rec, cell_id, checker, path):
    """The per-group environment contract (which EVOGP_* knobs each cell must run with)."""
    env = rec.get("env") if isinstance(rec.get("env"), dict) else {}
    group = GROUP[cell_id]
    if group == "legacy":
        checker.check("EVOGP_ETL_BACKEND" not in env and "EVOGP_EVAL_MODE" not in env,
                      "env.legacy_inert", path,
                      f"the legacy side is inert to ETL/EVOGP knobs (got {sorted(env)})")
    if group == "dense":
        for key, expected in (("EVOGP_ETL_BACKEND", "iree"), ("EVOGP_EVAL_MODE", "auto"),
                              ("EVOGP_ETL_IREE_DEVICE", "0"), ("EVOGP_ETL_IREE_OPT_LEVEL", "O1")):
            checker.check(env.get(key) == expected, f"env.{key}", path,
                          f"{key} must be {expected!r} (got {env.get(key)!r})")
    if group == "pure":
        for key, expected in (("EVOGP_ETL_BACKEND", "iree"), ("EVOGP_EVAL_MODE", "dense"),
                              ("EVOGP_ETL_IREE_DEVICE", "0"), ("EVOGP_ETL_IREE_OPT_LEVEL", "O1")):
            checker.check(env.get(key) == expected, f"env.{key}", path,
                          f"{key} must be {expected!r} (got {env.get(key)!r})")
    if group == "triton":
        checker.check(env.get("EVOGP_EVAL_MODE") == "dense", "env.EVOGP_EVAL_MODE", path,
                      f"EVOGP_EVAL_MODE must be 'dense' (the seam is the dense route) "
                      f"(got {env.get('EVOGP_EVAL_MODE')!r})")
        checker.check(env.get("TRITON_CACHE_DIR") == TRITON_CACHE, "env.TRITON_CACHE_DIR", path,
                      f"TRITON_CACHE_DIR must be {TRITON_CACHE!r} (got {env.get('TRITON_CACHE_DIR')!r})")
        checker.check("python3.11" in str(env.get("CPATH", "")), "env.CPATH", path,
                      f"CPATH must carry the session-scoped Python.h tree (got {env.get('CPATH')!r})")
        checker.check(env.get("EVOGP_ETL_BACKEND") is None, "env.no_etl_backend", path,
                      f"the triton route bypasses ETL and must not select a backend (got {env.get('EVOGP_ETL_BACKEND')!r})")
    if group != "triton":
        checker.check("TRITON_CACHE_DIR" not in env and "CPATH" not in env, "env.triton_knobs_absent", path,
                      f"the triton-only session knobs must not leak into a {group} cell (got {sorted(env)})")


def check_cell(rec, cell_id, window, checker, commit):
    """Dispatch the per-kind cell checks (and the log/identity layer) for one record."""
    path = rel(os.path.join(RESULTS, window, cell_id + ".json"))
    check_common(rec, cell_id, checker, path, commit)
    check_env_roles(rec, cell_id, checker, path)
    kind = KIND[cell_id]
    if kind in ("t1_legacy", "t1_dense", "t1_triton"):
        check_type1(rec, cell_id, checker, path)
    elif kind == "t2_dense":
        check_pure_dense(rec, cell_id, checker, path)
    if kind in ("t1_triton", "t2_triton"):
        check_triton_identity(rec, cell_id, checker, path)
    log_dir = os.path.join(RESULTS, window)
    log = os.path.join(log_dir, cell_id + ".log")
    checker.check(os.path.isfile(log), "cell.log.present", rel(log), "every cell must keep its runner log")
    text = read_text(log)
    checker.check(f"R6 window cell {cell_id} " in text, "cell.log.header", rel(log),
                  f"log must carry the cell header for {cell_id}")
    checker.check("exit code: 0 (OK)" in text, "cell.log.exit0", rel(log),
                  "log must record 'exit code: 0 (OK)' (no timeout / non-zero exit)")
    checker.check("interrupted=True" not in text, "cell.log.interrupted", rel(log),
                  "log records interrupted=True (a timeout artifact)")
    if "triton" in kind or cell_id.startswith("triton"):
        side_log = os.path.join(log_dir, cell_id + ".json.log")
        checker.check(os.path.isfile(side_log), "cell.log.driver", rel(side_log),
                      "the triton driver must keep its own <cell>.json.log")
        driver = read_text(side_log)
        checker.check("counter gate PASSED" in driver, "cell.log.counter_gate", rel(side_log),
                      "the driver log must record the counter gate PASSED line")


# ---------------------------------------------------------------------------
# window-level checks
# ---------------------------------------------------------------------------
def check_window(window, checker, primary):
    """Window files, the window_meta.txt rows, and the check_deps.txt pre-flight verdict."""
    wdir = os.path.join(RESULTS, window)
    meta_path = os.path.join(wdir, "window_meta.txt")
    deps_path = os.path.join(wdir, "check_deps.txt")
    checker.check(os.path.isdir(wdir), "window.dir", rel(wdir), "window result dir is missing")
    checker.check(os.path.isfile(meta_path), "window.meta.present", rel(meta_path),
                  "window_meta.txt is missing")
    checker.check(os.path.isfile(deps_path), "window.deps.present", rel(deps_path),
                  "check_deps.txt is missing")
    meta = read_text(meta_path)
    deps = read_text(deps_path)
    checker.check(f"# window:     {window}" in meta, "window.meta.label", rel(meta_path),
                  f"window_meta.txt must declare window {window}")
    checker.check("ok=10 skipped=0 failed=0" in meta, "window.meta.counts", rel(meta_path),
                  "window_meta.txt must end with ok=10 skipped=0 failed=0")
    checker.check("CUDA_VISIBLE_DEVICES=6" in meta and "# gpu (start): 6," in meta,
                  "window.meta.gpu6", rel(meta_path), "window_meta.txt must record GPU 6")
    start_line = re.search(r"^# start:\s*(\S+)\s*$", meta, re.MULTILINE)
    end_line = re.search(r"^# end:\s*(\S+)\s*$", meta, re.MULTILINE)
    start = parse_stamp(start_line.group(1)) if start_line else None
    end = parse_stamp(end_line.group(1)) if end_line else None
    checker.check(start is not None and end is not None and start <= end, "window.meta.timestamps",
                  rel(meta_path),
                  f"# start: / # end: must parse and be ordered (got {start_line and start_line.group(1)!r} / "
                  f"{end_line and end_line.group(1)!r})")
    rows = [m.groupdict() for m in (META_ROW_RE.match(line) for line in meta.splitlines()) if m]
    checker.check(len(rows) == len(CELL_IDS), "window.meta.rows", rel(meta_path),
                  f"window_meta.txt must list the {len(CELL_IDS)} campaign cells (got {len(rows)})")
    checker.check([r["id"] for r in rows] == CELL_IDS, "window.meta.row_order", rel(meta_path),
                  f"window_meta.txt rows must follow the runner's table order (got "
                  f"{[r['id'] for r in rows]!r})")
    for row in rows:
        cid = row["id"]
        checker.check(row["exit"] == "0" and row["status"] == "OK", "window.meta.exit", rel(meta_path),
                      f"cell {cid} must be exit 0 / OK (got {row['exit']} / {row['status']})")
        checker.check(row["out"].endswith(f"results/{window}/{cid}.json"), "window.meta.out", rel(meta_path),
                      f"cell {cid} must point at its own json (got {row['out']!r})")
        row_start = parse_stamp(row["start"])
        row_end = parse_stamp(row["end"])
        checker.check(row_start is not None and row_end is not None and row_start <= row_end
                      and (start is None or row_start >= start) and (end is None or row_end <= end),
                      "window.meta.row_window", rel(meta_path),
                      f"cell {cid} must run inside [{row['start']}, {row['end']}] of the window "
                      f"[{start}, {end}]")
        cell_start = parse_stamp(ref_value(load_json(checker, os.path.join(wdir, cid + ".json"), "window.cell.json") or {},
                                           ("started_at",)))
        # the record's stamp is written after the interpreter/device comes up, so it trails the
        # shell-side row start by a few seconds and must still land inside the row's own window
        lead = (cell_start - row_start).total_seconds() if (cell_start and row_start) else None
        checker.check(lead is not None and 0.0 <= lead <= 60.0
                      and (row_end is None or (cell_start <= row_end)),
                      "window.meta.row_matches_cell", rel(meta_path),
                      f"cell {cid}: the record's started_at {cell_start} must trail window_meta start "
                      f"{row['start']} by 0-60 s (lead={lead}) and precede the row end {row['end']}")
    # the deps pre-flight
    checker.check("READY" in deps, "window.deps.verdict", rel(deps_path),
                  "the dependency pre-flight must print a READY verdict")
    checker.check("20 ok / 0 warn / 0 missing" in deps, "window.deps.counts", rel(deps_path),
                  "the dependency pre-flight must be 20 ok / 0 warn / 0 missing")
    deps_rows = [m.groupdict()
                 for m in (DEPS_ROW_RE.match(line) for line in deps.split("## raw probe")[0].splitlines()) if m]
    checker.check(len(deps_rows) == 20, "window.deps.rows", rel(deps_path),
                  f"the pre-flight must list its 20 probes (got {len(deps_rows)})")
    bad = [(r["probe"], r["state"]) for r in deps_rows if r["state"] != "OK"]
    checker.check(not bad, "window.deps.all_ok", rel(deps_path), f"every probe must be OK (got {bad})")
    checker.check("evogp-commit" in deps and "triton-driver" in deps and "fixture200" in deps,
                  "window.deps.probes", rel(deps_path),
                  "the pre-flight must cover the commit, the triton driver and both fixtures")
    if primary:
        doc = read_text(RUNLOG)
        match = None
        for candidate in WINDOW_LINE_RE.finditer(doc):
            if candidate.group("kind") == "primary" and candidate.group("label") == window:
                match = candidate
                break
        checker.check(match is not None, "window.runlog.primary_line", rel(RUNLOG),
                      f"RUNLOG.md must document {window} as the primary window with its bounds")
        if match and start_line and end_line:
            checker.check(start_line.group(1) == match.group("start"), "window.runlog.start", rel(RUNLOG),
                          f"window_meta start {start_line.group(1)!r} must equal the RUNLOG's "
                          f"{match.group('start')!r}")
            checker.check(end_line.group(1)[11:] == match.group("end"), "window.runlog.end", rel(RUNLOG),
                          f"window_meta end {end_line.group(1)!r} must equal the RUNLOG's {match.group('end')!r}")
    return rows


def check_anomaly_window(checker, primary_rows):
    """w1c is kept as evidence: structurally valid and recomputably a superseded window."""
    rows = check_window(ANOMALY, checker, primary=False)
    checker.check(len(rows) == len(CELL_IDS), "anomaly.rows", "results/" + ANOMALY,
                  f"the anomaly window must also hold {len(CELL_IDS)} cells (got {len(rows)})")
    ratios = {}
    for cid in CELL_IDS:
        rec = load_json(checker, os.path.join(W1C, cid + ".json"), f"anomaly[{cid}].json")
        if rec is None:
            continue
        main = envelope_ms(rec)
        if not positive(main):
            checker.check(False, f"anomaly[{cid}].median", rel(os.path.join(W1C, cid + ".json")),
                          f"anomaly median must be finite and > 0 (got {main!r})")
            continue
        ratios[cid] = main
    anomaly = ratios
    for cid in ("leg_f64", "leg_f200"):
        if cid in ratios and cid in primary_rows:
            ratio = ratios[cid] / primary_rows[cid]
            checker.check(ratio > 2.0, f"anomaly[{cid}].rejection_ratio", "results/" + ANOMALY,
                          f"the superseded window must be > 2x the primary window on the legacy cells "
                          f"(got {ratio:.3f}x on {cid})")
            checker.check(quoted_pct(read_text(RUNLOG), (ratio - 1.0) * 100.0), f"anomaly[{cid}].doc_delta",
                          rel(RUNLOG),
                          f"RUNLOG.md must quote the recomputed w1c/w1d delta of {cid} "
                          f"({(ratio - 1.0) * 100.0:+.1f} %)")
    for cid, value in sorted(anomaly.items()):
        ratio = value / primary_rows[cid]
        checker.check(0.5 < ratio < 2.0, f"anomaly[{cid}].same_order", "results/" + ANOMALY,
                      f"a non-legacy cell must stay inside [0.5, 2.0]x of the primary window -- the "
                      f">2x rejection rule is legacy-specific (got {ratio:.3f}x on {cid})")
    doc = read_text(RUNLOG)
    section = doc.find("## 3. Anomaly window")
    checker.check(section > 0, "anomaly.runlog.section", rel(RUNLOG),
                  "RUNLOG.md must carry the anomaly-window section")
    checker.check("superseded" in doc[section:section + 200] if section > 0 else False,
                  "anomaly.runlog.superseded", rel(RUNLOG),
                  "the anomaly section must mark the window as superseded")
    for cid in ("leg_f64", "leg_f200"):
        for form in ("%.4f" % ratios.get(cid, 0.0),):
            index = doc.find(form)
            checker.check(index > section >= 0, f"anomaly[{cid}].not_primary", rel(RUNLOG),
                          f"the anomaly median {form} must never appear before the anomaly section "
                          f"(found at {index}, section at {section})")
    return ratios


# ---------------------------------------------------------------------------
# headline / cross-record checks
# ---------------------------------------------------------------------------
def check_headlines(checker, medians):
    """Recompute every headline ratio from the records and reconcile them with the RUNLOG."""
    doc = read_text(RUNLOG)
    pairs = [
        ("dense/legacy", "dense_f64", "leg_f64", 7.52, "dense_f200", "leg_f200", 10.08),
        ("triton/legacy", "triton_f64", "leg_f64", 0.207, "triton_f200", "leg_f200", 0.220),
        ("dense/triton", "dense_f64", "triton_f64", 36.42, "dense_f200", "triton_f200", 45.81),
        ("densepure/tritonpure", "densepure_f64", "tritonpure_f64", 24.39, "densepure_f200", "tritonpure_f200", 27.04),
        ("tritonpure/legacy", "tritonpure_f64", "leg_f64", 0.0859, "tritonpure_f200", "leg_f200", 0.0873),
    ]
    for label, num64, den64, want64, num200, den200, want200 in pairs:
        for num, den, want, fixture in ((num64, den64, want64, "fixture64"), (num200, den200, want200, "fixture200")):
            if num not in medians or den not in medians:
                checker.check(False, f"headline[{label}].records", "results/" + PRIMARY,
                              f"{num}/{den} must both be present to recompute the {label} ratio")
                continue
            ratio = medians[num] / medians[den]
            checker.check(abs(ratio - want) <= 0.005 * want, f"headline[{label}].{fixture}", "results/" + PRIMARY,
                          f"{label} on {fixture} must recompute to {want}x (got {ratio:.4f}x from "
                          f"{medians[num]!r}/{medians[den]!r})")
            checker.check(quoted_ratio(doc, ratio), f"headline[{label}].doc.{fixture}", rel(RUNLOG),
                          f"RUNLOG.md must quote the recomputed {label} ratio {ratio:.2f}x")
    for cell_id, value in sorted(medians.items()):
        text = RUNLOG_MEDIAN_FMT[cell_id] % value
        checker.check(f"**{text}**" in doc, f"headline[{cell_id}].doc", rel(RUNLOG),
                      f"RUNLOG.md must quote the recomputed median {text} ms as the headline of record")
    checker.check(medians["triton_f64"] < medians["leg_f64"] and medians["triton_f200"] < medians["leg_f200"],
                  "headline.triton_beats_legacy", "results/" + PRIMARY,
                  f"the triton route must be faster than the legacy engine on both fixtures "
                  f"({medians['triton_f64']!r}/{medians['leg_f64']!r}, {medians['triton_f200']!r}/{medians['leg_f200']!r})")
    checker.check(medians["dense_f64"] > medians["densepure_f64"] and medians["dense_f200"] > medians["densepure_f200"],
                  "headline.dense_e2e_over_pure", "results/" + PRIMARY,
                  "the dense e2e envelope must be slower than the dense pure block on both fixtures")
    for fixture, e2e, pure in (("fixture64", "triton_f64", "tritonpure_f64"), ("fixture200", "triton_f200", "tritonpure_f200")):
        checker.check(medians[e2e] > medians[pure], f"headline.triton_e2e_over_pure.{fixture}",
                      "results/" + PRIMARY,
                      f"{e2e} must exceed {pure} (got {medians[e2e]!r} vs {medians[pure]!r})")
    checker.check(medians["dense_f64"] > medians["triton_f64"] * 10 and medians["dense_f200"] > medians["triton_f200"] * 10,
                  "headline.dense_order_of_magnitude", "results/" + PRIMARY,
                  "the triton fused evaluator must be an order of magnitude faster than the dense graph")


def check_r5_anchors(checker, medians):
    """Recompute the six RUNLOG anchor deltas against the r5 same-window records."""
    doc = read_text(RUNLOG)
    for cell_id, anchor_file, anchor_value, quoted, delta_quoted in ANCHORS:
        path = os.path.join(R5_WINDOW, anchor_file)
        anchor = load_json(checker, path, f"anchor[{cell_id}].json")
        if anchor is None:
            continue
        value = envelope_ms(anchor)
        checker.check(positive(value), f"anchor[{cell_id}].median", rel(path),
                      f"the r5 anchor median must be finite and > 0 (got {value!r})")
        checker.check(close(value, anchor_value, 1e-4), f"anchor[{cell_id}].value", rel(path),
                      f"the r5 anchor of record must be {anchor_value} ms (got {value!r})")
        checker.check(quoted in doc, f"anchor[{cell_id}].doc", rel(RUNLOG),
                      f"RUNLOG.md must quote the r5 anchor {quoted} ms")
        if cell_id in medians and positive(value):
            delta = (medians[cell_id] - value) / value * 100.0
            checker.check(abs(delta - float(delta_quoted)) <= 0.15, f"anchor[{cell_id}].delta", rel(RUNLOG),
                          f"the r6-vs-r5 delta on {cell_id} must be {delta_quoted} % (got {delta:+.2f} %)")
            checker.check(quoted_pct(doc, delta), f"anchor[{cell_id}].delta_doc", rel(RUNLOG),
                          f"RUNLOG.md must quote the recomputed delta {delta:+.1f} %")


def check_runlog(checker, w1d_rows):
    """The operator log's own immutability / evidence statements."""
    doc = read_text(RUNLOG)
    checker.check(len(doc) > 10000, "runlog.size", rel(RUNLOG), f"RUNLOG.md looks truncated (got {len(doc)} chars)")
    for needle, cid in (("immutable evidence", "runlog.immutable"),
                        ("w1c_20260914", "runlog.anomaly_label"),
                        ("w1d_20260914", "runlog.primary_label"),
                        ("counter gate", "runlog.counter_gate"),
                        (">2×", "runlog.anomaly_rule"),
                        ("GPU 7 never touched", "runlog.gpu7"),
                        ("fcntl.flock", "runlog.lock")):
        checker.check(needle in doc, cid, rel(RUNLOG), f"RUNLOG.md must state {needle!r}")
    for cell_id in CELL_IDS:
        checker.check(f"`{cell_id}`" in doc, f"runlog.cell[{cell_id}]", rel(RUNLOG),
                      f"RUNLOG.md must discuss cell {cell_id}")
    checker.check("check_r6.py" in doc, "runlog.gate_mention", rel(RUNLOG),
                  "RUNLOG.md must reference the round's gate")
    for row in w1d_rows:
        checker.check(f"`{row['id']}` | 0 |" in doc or f"| `{row['id']}` |" in doc,
                      f"runlog.row[{row['id']}]", rel(RUNLOG),
                      f"RUNLOG.md must table the primary window's {row['id']} row")


# ---------------------------------------------------------------------------
# validation evidence
# ---------------------------------------------------------------------------
def check_validation_record(rec, checker, path, run, checker_names=None):
    """Structural + numeric checks for one numeric-validation record."""
    cid = f"val{run}"
    checker.check(rec.get("schema") == "r6_validate_triton/1", cid + ".schema", path,
                  f"schema must be 'r6_validate_triton/1' (got {rec.get('schema')!r})")
    expected_tag = "r6_triton_validation" if run == 1 else "r6_triton_validation_run2"
    checker.check(rec.get("tag") == expected_tag, cid + ".tag", path,
                  f"tag must be {expected_tag!r} (got {rec.get('tag')!r})")
    checker.check(rec.get("route") == "triton" and rec.get("side") == "new", cid + ".route", path,
                  f"route/side must be triton/new (got {rec.get('route')!r}/{rec.get('side')!r})")
    checker.check(rec.get("staging") == STAGING, cid + ".staging", path,
                  f"staging must be {STAGING!r} (got {rec.get('staging')!r})")
    checker.check(re.fullmatch(r"[0-9a-f]{40}", str(rec.get("evogp_commit"))) is not None, cid + ".commit", path,
                  f"evogp_commit must be 40 hex chars (got {rec.get('evogp_commit')!r})")
    declaration = rec.get("declaration") if isinstance(rec.get("declaration"), dict) else {}
    checker.check(declaration.get("staging") == STAGING and declaration.get("exact_ops") == EXACT_OPS,
                  cid + ".declaration", path,
                  "the declaration must repeat the round's staging + exact_ops")
    checker.check(isinstance(declaration.get("op_coverage"), list)
                  and len(declaration["op_coverage"]) == OP_COVERAGE_LEN, cid + ".declaration.op_coverage", path,
                  f"the declaration must cover {OP_COVERAGE_LEN} ops "
                  f"(got {len(declaration.get('op_coverage') or [])})")
    checker.check(sha256_file(ORACLE) == declaration.get("oracle", {}).get("sha256"), cid + ".oracle.sha256", path,
                  f"the oracle sha256 must match the working-tree {rel(ORACLE)} "
                  f"(recorded {(declaration.get('oracle') or {}).get('sha256')!r})")
    checker.check(sha256_file(VALIDATOR) == declaration.get("validator", {}).get("sha256"), cid + ".validator.sha256", path,
                  f"the validator sha256 must match the working-tree {rel(VALIDATOR)} "
                  f"(recorded {(declaration.get('validator') or {}).get('sha256')!r})")
    for name, fixture in (("fixture64", "fixture64"), ("fixture200", "fixture200")):
        entry = (declaration.get("fixtures") or {}).get(name) or {}
        target = os.path.join(REPO_ROOT, FIXTURES[fixture]["rel"])
        checker.check(sha256_file(target) == entry.get("sha256"), f"{cid}.fixtures.{name}.sha256", path,
                      f"{name} sha256 must match the working-tree {FIXTURES[fixture]['rel']} "
                      f"(recorded {entry.get('sha256')!r})")
        checker.check(entry.get("path") == FIXTURES[fixture]["rel"], f"{cid}.fixtures.{name}.path", path,
                      f"{name} must be declared at {FIXTURES[fixture]['rel']!r} (got {entry.get('path')!r})")
    goldens = (declaration.get("fixtures") or {}).get("legacy_goldens") or {}
    checker.check(sha256_file(os.path.join(REPO_ROOT, "test", "goldens", "sr_fitness.npz")) == goldens.get("sha256"),
                  cid + ".fixtures.goldens.sha256", path,
                  f"the legacy-goldens sha256 must match test/goldens/sr_fitness.npz (recorded {goldens.get('sha256')!r})")
    method = rec.get("method") if isinstance(rec.get("method"), dict) else {}
    tol = method.get("tolerances") if isinstance(method.get("tolerances"), dict) else {}
    checker.check(tol.get("rtol") == RTOL and tol.get("atol") == ATOL and tol.get("equal_nan") is True,
                  cid + ".method.tolerances", path,
                  f"the declared tolerances must be rtol=atol={ATOL} with equal_nan (got {tol!r})")
    checks = rec.get("checks")
    checker.check(isinstance(checks, list) and len(checks) == 208, cid + ".checks.length", path,
                  f"the record must hold its 208 checks (got {len(checks) if isinstance(checks, list) else checks!r})")
    if isinstance(checks, list) and checks:
        names = [c.get("name") for c in checks]
        checker.check(len(set(names)) == len(names), cid + ".checks.unique", path,
                      "every check name must be unique")
        gate = [c for c in checks if c.get("severity") == "gate"]
        documented = [c for c in checks if c.get("severity") == "documented_deviation"]
        checker.check(len(gate) == 198 and len(documented) == 10, cid + ".checks.severity", path,
                      f"the severity split must be 198 gate / 10 documented_deviation "
                      f"(got {len(gate)}/{len(documented)})")
        failed_gate = [c.get("name") for c in gate if c.get("passed") is not True]
        checker.check(not failed_gate, cid + ".checks.gate_passed", path,
                      f"every gate-severity check must have passed (failed: {failed_gate})")
        if checker_names is not None:
            checker.check([c.get("name") for c in gate] == checker_names, cid + ".checks.names", path,
                          "the gate-severity check names must match run 1's (same validator)")
    policy = rec.get("gate_policy") if isinstance(rec.get("gate_policy"), dict) else {}
    checker.check(policy.get("passed") is True and policy.get("gate_severity_checks") == 198
                  and policy.get("documented_deviation_checks") == 10, cid + ".gate_policy", path,
                  f"gate_policy must report 198 gate checks passed (got {policy!r})")
    downgraded = rec.get("downgraded_checks")
    checker.check(isinstance(downgraded, list) and len(downgraded) == 10
                  and policy.get("passed_strict") is False, cid + ".downgraded_checks", path,
                  f"the 10 documented-deviation checks must be listed and passed_strict false "
                  f"(got {len(downgraded) if isinstance(downgraded, list) else downgraded!r}/"
                  f"{policy.get('passed_strict')!r})")
    checker.check(rec.get("passed") is True and rec.get("exit_code") == 0, cid + ".verdict", path,
                  f"passed must be true with exit_code 0 (got {rec.get('passed')!r}/{rec.get('exit_code')!r})")
    checker.check(rec.get("results_traceback") is None, cid + ".traceback", path,
                  f"no traceback may be recorded (got {rec.get('results_traceback')!r})")
    checker.check(positive(rec.get("elapsed_s")), cid + ".elapsed_s", path,
                  f"elapsed_s must be finite and > 0 (got {rec.get('elapsed_s')!r})")
    # fixture losses / values / goldens: the numeric gate
    for fixture, spec in FIXTURES.items():
        for metric in ("mse", "mae"):
            entry = ((rec.get("fixture_losses") or {}).get(fixture) or {}).get(metric) or {}
            cmp_ = entry.get("triton_vs_oracle") if isinstance(entry.get("triton_vs_oracle"), dict) else {}
            path_lbl = f"{cid}.{fixture}.{metric}"
            checker.check(cmp_.get("n") == spec["pop"], path_lbl + ".n", path,
                          f"the loss comparison must cover the whole population {spec['pop']} (got {cmp_.get('n')!r})")
            checker.check(cmp_.get("allclose") is True, path_lbl + ".allclose", path,
                          f"the {fixture} {metric} loss must be allclose at rtol=atol=1e-4 (got {cmp_.get('allclose')!r})")
            checker.check(cmp_.get("rtol") == RTOL and cmp_.get("atol") == ATOL, path_lbl + ".tolerance", path,
                          f"the comparison must use rtol=atol=1e-4 (got {cmp_.get('rtol')!r}/{cmp_.get('atol')!r})")
            checker.check(cmp_.get("max_rel_diff") is not None and float(cmp_["max_rel_diff"]) <= RTOL,
                          path_lbl + ".max_rel_diff", path,
                          f"the relative loss difference must be <= 1e-4 (got {cmp_.get('max_rel_diff')!r})")
            checker.check(is_num(cmp_.get("max_ulp")) and float(cmp_["max_ulp"]) <= 4, path_lbl + ".max_ulp", path,
                          f"the loss arrays must agree within a few ULP (got {cmp_.get('max_ulp')!r})")
            checker.check(cmp_.get("n_special_mismatch") == 0
                          and cmp_.get("n_nan_a") == cmp_.get("n_nan_b") == spec["nan"], path_lbl + ".specials", path,
                          f"NaN placement must match exactly ({spec['nan']} NaNs, no special mismatch; "
                          f"got mismatch={cmp_.get('n_special_mismatch')!r}, nan="
                          f"{cmp_.get('n_nan_a')!r}/{cmp_.get('n_nan_b')!r})")
            contract = entry.get("contract") if isinstance(entry.get("contract"), dict) else {}
            checker.check(contract.get("dtype") == "float32" and contract.get("shape") == [spec["pop"]]
                          and contract.get("launches_per_call") == 1 and contract.get("negative_finite") == 0,
                          path_lbl + ".contract", path,
                          f"the loss contract must be a positive float32 ({spec['pop']},) with one launch "
                          f"(got {contract!r})")
        entry = ((rec.get("fixture_values") or {}).get(fixture) or {})
        cmp_ = entry.get("triton_vs_oracle") if isinstance(entry.get("triton_vs_oracle"), dict) else {}
        path_lbl = f"{cid}.{fixture}.values"
        checker.check(cmp_.get("n") == spec["values_n"] and cmp_.get("allclose") is True, path_lbl + ".allclose", path,
                      f"the {spec['values_n']} decoded values must be allclose (got n={cmp_.get('n')!r}, "
                      f"allclose={cmp_.get('allclose')!r})")
        checker.check(finite(cmp_.get("max_abs_diff")) and float(cmp_["max_abs_diff"]) <= ATOL
                      and float(cmp_.get("max_rel_diff", 1.0)) <= RTOL, path_lbl + ".diffs", path,
                      f"the value arrays must agree within atol=1e-4 / rtol=1e-4 "
                      f"(got max|d|={cmp_.get('max_abs_diff')!r}, max rel={cmp_.get('max_rel_diff')!r})")
        checker.check(cmp_.get("max_ulp") == 0 and cmp_.get("n_bitexact_or_both_nan") == cmp_.get("n"),
                      path_lbl + ".bitexact", path,
                      f"the value arrays must be bit-exact (or both NaN) (got {cmp_.get('n_bitexact_or_both_nan')!r}"
                      f"/{cmp_.get('n')!r}, max_ulp={cmp_.get('max_ulp')!r})")
        contract = entry.get("contract") if isinstance(entry.get("contract"), dict) else {}
        checker.check(contract.get("shape") == [spec["pop"], spec["n_rows"]] and contract.get("dtype") == "float32"
                      and contract.get("n_nan") == contract.get("oracle_n_nan") == spec["nan"]
                      and contract.get("n_inf") == 0, path_lbl + ".contract", path,
                      f"the value contract must be ({spec['pop']}, {spec['n_rows']}) float32 with "
                      f"{spec['nan']} NaNs and no inf (got {contract!r})")
        decl = (rec.get("fixture_declaration") or {}).get(fixture) or {}
        checker.check(decl.get("pop") == spec["pop"] and decl.get("n_rows") == spec["n_rows"]
                      and decl.get("input_len") == 2 and decl.get("output_len") == 1, path_lbl + ".declaration", path,
                      f"the fixture declaration must be pop={spec['pop']}, n_rows={spec['n_rows']}, "
                      f"2 inputs / 1 output (got {decl!r})")
        checker.check(set((decl.get("used_op_ids") or {})) <= {"VAR", "CONST", "BFUNC:1", "BFUNC:2", "BFUNC:3", "BFUNC:4"},
                      path_lbl + ".op_ids", path,
                      f"the fixture may only use VAR/CONST and the four BFUNC ids (got {decl.get('used_op_ids')!r})")
        scan = (rec.get("fixture_special_value_scan") or {}).get(fixture) or {}
        for field in ("x", "labels", "consts"):
            block = scan.get(field) or {}
            checker.check(block.get("n_denormal") == block.get("n_inf") == block.get("n_nan") == 0,
                          f"{path_lbl}.scan.{field}", path,
                          f"the fixtures must contain no denormal/inf/NaN {field} values (got {block!r})")
    goldens = rec.get("legacy_goldens") if isinstance(rec.get("legacy_goldens"), dict) else {}
    comparisons = goldens.get("comparisons") if isinstance(goldens.get("comparisons"), dict) else {}
    checker.check(len(comparisons) == 12, cid + ".goldens.count", path,
                  f"all 12 frozen legacy arrays must be compared (got {len(comparisons)})")
    not_close = [name for name, cmp_ in comparisons.items() if cmp_.get("allclose") is not True]
    checker.check(not not_close, cid + ".goldens.allclose", path,
                  f"every legacy-goldens comparison must be allclose at 1e-4 (not: {not_close})")
    bad_tol = [name for name, cmp_ in comparisons.items()
               if cmp_.get("rtol") != RTOL or cmp_.get("atol") != ATOL]
    checker.check(not bad_tol, cid + ".goldens.tolerance", path,
                  f"every goldens comparison must use rtol=atol=1e-4 (not: {bad_tol})")
    cross = rec.get("dense_crosscheck") if isinstance(rec.get("dense_crosscheck"), dict) else {}
    checker.check(cross.get("run") is True and cross.get("backend") == "numpy", cid + ".crosscheck.run", path,
                  f"the dense-engine cross-check must have run on numpy (got {cross.get('run')!r}/"
                  f"{cross.get('backend')!r})")
    for fixture in FIXTURES:
        block = cross.get(fixture) if isinstance(cross.get(fixture), dict) else {}
        for name in ("engine_vs_oracle", "triton_vs_engine"):
            cmp_ = block.get(name) if isinstance(block.get(name), dict) else {}
            checker.check(cmp_.get("allclose") is True, f"{cid}.crosscheck.{fixture}.{name}", path,
                          f"{name} on {fixture} must be allclose (got {cmp_.get('allclose')!r})")
        checker.check(positive(block.get("wall_s")), f"{cid}.crosscheck.{fixture}.wall_s", path,
                      f"the cross-check wall_s must be finite and > 0 (got {block.get('wall_s')!r})")
    summary = rec.get("kernel_coverage_summary") if isinstance(rec.get("kernel_coverage_summary"), dict) else {}
    checker.check(summary.get("uncovered_ops") == [] and summary.get("fallthrough_nodes") == 0
                  and summary.get("never_exercised") == [] and summary.get("exercised_but_undeclared") == [],
                  cid + ".coverage", path,
                  f"every declared op must be exercised with no fallthrough (got {summary!r})")
    checker.check(len(summary.get("declared_ops") or []) == OP_COVERAGE_LEN
                  and sorted(summary.get("used_ops") or []) == sorted(USED_OPS + [
                      "<", "<=", ">", ">=", "abs", "cos", "cosh", "exp", "if", "inv", "log", "loose_div",
                      "loose_inv", "loose_log", "loose_pow", "loose_sqrt", "max", "min", "neg", "pow", "sin",
                      "sinh", "sqrt", "tan", "tanh"]), cid + ".coverage.ops", path,
                  f"the coverage summary must cover the declared {OP_COVERAGE_LEN} ops and the 31 exercised "
                  f"opcodes (got {len(summary.get('declared_ops') or [])}/"
                  f"{len(summary.get('used_ops') or [])})")
    per_op = rec.get("per_op") if isinstance(rec.get("per_op"), dict) else {}
    checker.check(len(per_op) >= OP_COVERAGE_LEN and set(summary.get("used_ops") or []) <= set(per_op),
                  cid + ".per_op", path,
                  f"the per-op matrix must cover the exercised opcodes (got {len(per_op)} entries)")
    edge = rec.get("edge_case_summary") if isinstance(rec.get("edge_case_summary"), dict) else {}
    checker.check(edge.get("fail") == [] and len(edge.get("pass") or []) == 24, cid + ".edge_cases", path,
                  f"all 24 edge cases must pass (got pass={len(edge.get('pass') or [])}, fail={edge.get('fail')!r})")
    checker.check(len(rec.get("edge_cases") or {}) == 24, cid + ".edge_case_rows", path,
                  f"the 24 edge-case rows must be recorded (got {len(rec.get('edge_cases') or {})})")
    deviations = rec.get("deviations") if isinstance(rec.get("deviations"), list) else []
    checker.check(bool(deviations) and all(isinstance(d.get("id"), str) and isinstance(d.get("root_cause"), str)
                                           for d in deviations), cid + ".deviations.shape", path,
                  "every deviation must carry an id and a root cause")
    checker.check(all(d.get("affects_fixtures") is False for d in deviations), cid + ".deviations.fixtures", path,
                  "no recorded deviation may affect the fixtures (the fixtures are in tolerance)")
    checker.check(len(rec.get("diagnostics") or []) >= 60, cid + ".diagnostics", path,
                  f"the diagnostics list looks truncated (got {len(rec.get('diagnostics') or [])})")


def check_validation(checker):
    """Run 1 (record of record) + run 2 (replicate) + their identity relation."""
    path1 = os.path.join(VALIDATION, "validation.json")
    path2 = os.path.join(VALIDATION2, "validation.json")
    run1 = load_json(checker, path1, "val1.json")
    run2 = load_json(checker, path2, "val2.json")
    if run1 is None or run2 is None:
        return
    check_validation_record(run1, checker, rel(path1), 1)
    gate_names = [c.get("name") for c in run1.get("checks", []) if c.get("severity") == "gate"]
    check_validation_record(run2, checker, rel(path2), 2, checker_names=gate_names)
    # the kernel-revision identity of the record of record
    working_sha = sha256_file(TRITON_EVAL)
    checker.check(working_sha == TRITON_EVAL_SHA, "provenance:triton_eval.sha256_constant", rel(TRITON_EVAL),
                  f"the committed kernel revision must still hash to {TRITON_EVAL_SHA} (got {working_sha})")
    recorded1 = (run1.get("declaration") or {}).get("triton_eval") or {}
    checker.check(recorded1.get("sha256") == working_sha, "provenance:run1_kernel_revision", rel(path1),
                  f"run 1 must pin the working-tree {rel(TRITON_EVAL)} sha256 ({working_sha}); "
                  f"recorded {recorded1.get('sha256')!r}")
    checker.check(str(recorded1.get("path", "")).endswith("test/benchmark/r6/triton_eval.py"),
                  "provenance:run1_path", rel(path1),
                  f"run 1 must name the round's triton_eval.py (got {recorded1.get('path')!r})")
    recorded2 = (run2.get("declaration") or {}).get("triton_eval") or {}
    sha2 = recorded2.get("sha256")
    checker.check(isinstance(sha2, str) and re.fullmatch(r"[0-9a-f]{64}", sha2) is not None,
                  "provenance:run2_declared_revision", rel(path2),
                  f"run 2 must declare a well-formed triton_eval.py sha256 (got {sha2!r})")
    checker.check(sha2 == RUN2_TRITON_EVAL_SHA, "provenance:run2_revision_of_record", rel(path2),
                  f"run 2's kernel revision is {RUN2_TRITON_EVAL_SHA} (got {sha2!r})")
    checker.check(sha2 != working_sha, "provenance:run2_not_committed_revision", rel(path2),
                  f"run 2's declared revision must NOT be the committed one ({working_sha}): the replicate "
                  f"ran a different, uncommitted triton_eval.py")
    checker.check(str(recorded2.get("path", "")).endswith("test/benchmark/r6/triton_eval.py"),
                  "provenance:run2_path", rel(path2),
                  f"run 2 must name its triton_eval.py (got {recorded2.get('path')!r})")
    checker.check(run1.get("evogp_commit") != run2.get("evogp_commit"), "provenance:two_worktrees", rel(path2),
                  f"run 1 and run 2 ran in different worktrees (got {run1.get('evogp_commit')!r} vs "
                  f"{run2.get('evogp_commit')!r})")
    # the divergence is corroborated by the recorded per-op signature (not a hash typo)
    key = "max per-op mismatch classes (all ops): "
    head1, head2 = str(run1.get("headline", "")), str(run2.get("headline", ""))
    seg1 = next((s for s in head1.split(" | ") if s.startswith(key)), "")
    seg2 = next((s for s in head2.split(" | ") if s.startswith(key)), "")
    checker.check(seg1 and seg2, "provenance:run2_mismatch_classes", rel(path1),
                  f"both headlines must carry the mismatch-class tail (got {seg1!r} / {seg2!r})")
    if seg1 and seg2:
        checker.check("'zero_sign'" in seg1 and "'zero_sign'" not in seg2, "provenance:run2_zero_sign", rel(path1),
                      "run 1 must record the zero_sign class and run 2 must not (the divergence is behavioural)")
    ids1 = {d.get("id") for d in run1.get("deviations") or []}
    ids2 = {d.get("id") for d in run2.get("deviations") or []}
    checker.check("dev-zero-sign" in ids1 and "dev-zero-sign" not in ids2, "provenance:run2_deviations", rel(path1),
                  f"run 1 must carry dev-zero-sign, run 2 not (got {sorted(ids1)} / {sorted(ids2)})")
    checker.check("dev-const-neglzero" in ids2 and "dev-const-neglzero" not in ids1,
                  "provenance:run2_new_deviation", rel(path1),
                  f"run 2 must carry the new dev-const-neglzero deviation (got {sorted(ids2)})")
    checker.note(
        "provenance: validation run 2 declares triton_eval.py sha256 %s, which is NOT the committed "
        "revision (%s) and exists in no git object; its per-op evidence corroborates a different kernel "
        "(zero_sign class gone, denormal_result 1792 -> 832, new dev-const-neglzero). run 1 is the "
        "record of record for the committed kernel; run 2 is a same-headline replicate on a different, "
        "uncommitted revision -- the run2 provenance checks above assert exactly that."
        % (RUN2_TRITON_EVAL_SHA[:12], TRITON_EVAL_SHA[:12]))
    # headline equality (the replicate must reproduce the fixture/golden numbers of record)
    def cmp_block(rec, *keys):
        node = rec
        for key in keys:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        return node if isinstance(node, dict) else {}

    def strip_compile(block, keys):
        out = {}
        for name in keys:
            node = block.get(name)
            if isinstance(node, dict):
                out[name] = {k: v for k, v in node.items() if k != "compile_ms"}
            else:
                out[name] = node
        return out

    for fixture in FIXTURES:
        metrics = ("mse", "mae")
        a = strip_compile(cmp_block(run1, "fixture_losses", fixture), metrics)
        b = strip_compile(cmp_block(run2, "fixture_losses", fixture), metrics)
        checker.check(a == b, f"valcross.{fixture}.losses", rel(path2),
                      f"run 2 must reproduce run 1's {fixture} loss comparisons exactly "  # (compile_ms excluded)
                      f"(run1={a!r}, run2={b!r})")
        a = {k: v for k, v in cmp_block(run1, "fixture_values", fixture).items()}
        b = {k: v for k, v in cmp_block(run2, "fixture_values", fixture).items()}
        for side in ("triton_vs_oracle", "contract"):
            va = dict(a.get(side) or {})
            vb = dict(b.get(side) or {})
            va.pop("compile_ms", None)
            vb.pop("compile_ms", None)
            checker.check(va == vb, f"valcross.{fixture}.values.{side}", rel(path2),
                          f"run 2 must reproduce run 1's {fixture} value comparisons on {side}")
        for name, cmp1 in (cmp_block(run1, "legacy_goldens", "comparisons")).items():
            cmp2 = cmp_block(run2, "legacy_goldens", "comparisons").get(name)
            checker.check(cmp1 is not None and cmp1 == cmp2, f"valcross.goldens.{name}", rel(path2),
                          f"run 2 must reproduce run 1's legacy-goldens comparison {name}")
    for key in ("method", "edge_case_summary", "kernel_coverage_summary", "fixture_special_value_scan"):
        checker.check(run1.get(key) == run2.get(key), f"valcross.{key}", rel(path2),
                      f"run 2 must repeat run 1's {key} block byte-for-byte")


# ---------------------------------------------------------------------------
# P1 extras
# ---------------------------------------------------------------------------
def check_kernel_host(checker):
    """The kineto kernel-vs-host decomposition fact sheets."""
    expectations = {
        "triton_pure_f64": ("triton-pure", "fixture64", 1.0, 0.056204162538051605, 5.8246599999999855),
        "triton_pure_f200": ("triton-pure", "fixture200", 1.0, 0.05460929125547409, 9.100540000000018),
        "triton_e2e_f64": ("triton-e2e", "fixture64", 3.0, 0.117361880838871, 8.242520000000004),
        "legacy_e2e_f64": ("legacy-e2e", "fixture64", 5.0, 0.6318847835063934, 14.707090000000516),
        "legacy_e2e_f200": ("legacy-e2e", "fixture200", 5.0, 0.6017780303955078, 25.30443000000254),
    }
    for name, (target, fixture, kernels_per_inv, clean_ms, leaf_us) in expectations.items():
        target_dir = os.path.join(KERNEL_HOST, name)
        facts_path = os.path.join(target_dir, "facts.json")
        cid = f"extra[{name}]"
        facts = load_json(checker, facts_path, cid + ".facts")
        for extra in ("kineto_key_averages.txt", "extra_trace.log", "kineto_trace.json.gz"):
            path = os.path.join(target_dir, extra)
            checker.check(os.path.isfile(path) and os.path.getsize(path) > 0, cid + ".file", rel(path),
                          f"{extra} must exist and be non-empty")
        if facts is None:
            continue
        spec = FIXTURES[fixture]
        checker.check(facts.get("target") == target, cid + ".target", rel(facts_path),
                      f"target must be {target!r} (got {facts.get('target')!r})")
        checker.check(os.path.basename(str(facts.get("fixture"))) == spec["basename"]
                      and facts.get("pop_size") == spec["pop"], cid + ".fixture", rel(facts_path),
                      f"the fact sheet must measure {spec['basename']} pop={spec['pop']} "
                      f"(got {facts.get('fixture')!r}/{facts.get('pop_size')!r})")
        checker.check(facts.get("invocations") == 100, cid + ".invocations", rel(facts_path),
                      f"invocations must be 100 (got {facts.get('invocations')!r})")
        checker.check(facts.get("kernels_per_invocation") == kernels_per_inv
                      and facts.get("leaf_kernels_per_invocation") == kernels_per_inv, cid + ".kernels_per_inv", rel(facts_path),
                      f"kernels/invocation must be {kernels_per_inv} (got {facts.get('kernels_per_invocation')!r}/"
                      f"{facts.get('leaf_kernels_per_invocation')!r})")
        checker.check(facts.get("kernel_count") == facts.get("leaf_kernel_count") == 100 * kernels_per_inv,
                      cid + ".kernel_count", rel(facts_path),
                      f"the profiled block must launch {int(100 * kernels_per_inv)} kernels "
                      f"(got {facts.get('kernel_count')!r}/{facts.get('leaf_kernel_count')!r})")
        checker.check(close(facts.get("clean_per_invocation_ms"), clean_ms, 1e-9)
                      and close(facts.get("kernel_time_leaf_per_invocation_us"), leaf_us, 1e-9),
                      cid + ".recorded_values", rel(facts_path),
                      f"the fact sheet must still read {clean_ms} ms clean / {leaf_us} us leaf device "
                      f"(got {facts.get('clean_per_invocation_ms')!r}/{facts.get('kernel_time_leaf_per_invocation_us')!r})")
        device_us = facts.get("kernel_time_leaf_per_invocation_us")
        residual = facts.get("leaf_residual_per_invocation_us")
        checker.check(finite(device_us) and finite(residual)
                      and close(facts.get("clean_per_invocation_ms") * 1000.0, device_us + residual, 1e-6),
                      cid + ".residual_recompute", rel(facts_path),
                      f"clean_us must equal leaf device_us + leaf residual "
                      f"({facts.get('clean_per_invocation_ms', 0) * 1000.0!r} vs "
                      f"{(device_us or 0) + (residual or 0)!r})")
        checker.check(positive(residual) and finite(device_us) and facts.get("clean_per_invocation_ms") * 1000.0 > device_us,
                      cid + ".host_bound", rel(facts_path),
                      f"the clean call must exceed the leaf device time (host/launch-bound; "
                      f"got clean={facts.get('clean_per_invocation_ms')!r} ms, device={device_us!r} us)")
        checker.check(is_num(facts.get("profiler_overhead_ms"))
                      and close(facts.get("profiled_per_invocation_ms") - facts.get("clean_per_invocation_ms"),
                                facts["profiler_overhead_ms"], 1e-6), cid + ".overhead", rel(facts_path),
                      f"profiler_overhead_ms must be profiled - clean (got {facts.get('profiler_overhead_ms')!r})")
        rows = facts.get("leaf_rows")
        checker.check(isinstance(rows, list) and sum(r.get("count", 0) for r in rows) == facts.get("leaf_kernel_count"),
                      cid + ".leaf_rows", rel(facts_path),
                      f"the leaf rows must sum to the leaf kernel count (got {facts.get('leaf_rows')!r})")
        if target == "triton-pure":
            checker.check(len(rows or []) == 1 and str((rows or [{}])[0].get("key", "")).endswith("_sr_eval_kernel"),
                          cid + ".single_kernel", rel(facts_path),
                          f"a pure triton invocation must launch exactly its own kernel (got {facts.get('leaf_rows')!r})")
            checker.check(positive(facts.get("median_per_invocation_ms")), cid + ".pure_median", rel(facts_path),
                          f"the pure median must be recorded (got {facts.get('median_per_invocation_ms')!r})")
        if target == "legacy-e2e":
            keys = " ".join(str(r.get("key")) for r in rows or [])
            checker.check("treeGPRegressionFitnessKernel" in keys and "averageFitnessValueKernel" in keys
                          and "Memset" in keys, cid + ".legacy_kernels", rel(facts_path),
                          f"the legacy decomposition must show the tree-walk, averaging and memset kernels (got {keys[:200]!r})")
    doc = read_text(RUNLOG)
    for token, cid in (("56.20", "extra.f64_clean_doc"), ("54.61", "extra.f200_clean_doc"),
                       ("117.36", "extra.e2e_clean_doc"), ("631.88", "extra.legacy_f64_doc"),
                       ("601.78", "extra.legacy_f200_doc"), ("5.825", "extra.leaf_f64_doc"),
                       ("9.101", "extra.leaf_f200_doc"), ("8.243", "extra.e2e_leaf_doc"),
                       ("14.707", "extra.legacy_leaf_f64_doc"), ("25.304", "extra.legacy_leaf_f200_doc"),
                       ("**1.00**", "extra.triton_kernels_doc"), ("5.00", "extra.legacy_kernels_doc")):
        checker.check(token in doc, cid, rel(RUNLOG), f"RUNLOG.md must quote the decomposed value {token}")


def check_compile_econ(checker):
    """The warm-vs-fresh TRITON_CACHE_DIR economics records."""
    warm_path = os.path.join(COMPILE_ECON, "warm_cache", "triton_f64_warmcache.json")
    fresh_path = os.path.join(COMPILE_ECON, "fresh_cache", "triton_f64_freshcache.json")
    counts_path = os.path.join(COMPILE_ECON, "cache_filecounts.txt")
    warm = load_json(checker, warm_path, "extra[compile_econ.warm]")
    fresh = load_json(checker, fresh_path, "extra[compile_econ.fresh]")
    checker.check(os.path.isfile(counts_path), "extra[compile_econ.counts].present", rel(counts_path),
                  "cache_filecounts.txt must be committed with the extras")
    main_cell = load_json(checker, os.path.join(W1D, "triton_f64.json"), "extra[compile_econ.main]")
    for name, rec, path in (("warm", warm, warm_path), ("fresh", fresh, fresh_path)):
        if rec is None:
            continue
        cid = f"extra[compile_econ.{name}]"
        checker.check(rec.get("route") == "triton" and rec.get("seam_patched") is True
                      and rec.get("staging") == STAGING, cid + ".identity", rel(path),
                      f"the extra must run the same triton seam/staging (got {rec.get('route')!r}/"
                      f"{rec.get('staging')!r})")
        checker.check((rec.get("counter_gate") or {}).get("passed") is True
                      and (rec.get("counter_gate") or {}).get("problems") == [], cid + ".counter_gate", rel(path),
                      f"the extra's counter gate must pass (got {rec.get('counter_gate')!r})")
        for field, expected in (("triton_call_count", TRITON_E2E_CALLS), ("triton_runs", TRITON_E2E_CALLS),
                                ("builds", 1), ("cache_hits", TRITON_E2E_CALLS - 1), ("data_uploads", 1),
                                ("fallback_count", 0), ("reps_requested", EVAL_REPS)):
            checker.check(rec.get(field) == expected, cid + "." + field, rel(path),
                          f"{field} must be {expected} (got {rec.get(field)!r})")
        calls = rec.get("calls_ms")
        checker.check(isinstance(calls, list) and len(calls) == EVAL_REPS, cid + ".calls_ms", rel(path),
                      f"calls_ms must hold {EVAL_REPS} calls (got {len(calls) if isinstance(calls, list) else calls!r})")
        if isinstance(calls, list) and len(calls) == EVAL_REPS and all(finite(v) for v in calls):
            checker.check(rec.get("cold_call_ms") == calls[0] and rec.get("median_warm_ms") == med(calls[1:]),
                          cid + ".timing", rel(path),
                          f"cold_call_ms/median_warm_ms must be recomputable from calls_ms "
                          f"(got {rec.get('cold_call_ms')!r}/{rec.get('median_warm_ms')!r})")
        compile_block = rec.get("compile") if isinstance(rec.get("compile"), dict) else {}
        checker.check(positive(compile_block.get("first_call_ms")) and positive(compile_block.get("compile_ms"))
                      and compile_block.get("n_builds") == 1, cid + ".compile", rel(path),
                      f"the extra must record its own compile economics (got {compile_block!r})")
        checker.check(is_num(compile_block.get("stage_upload_ms")) and float(compile_block["stage_upload_ms"]) >= 0.0,
                      cid + ".stage_upload_ms", rel(path),
                      f"stage_upload_ms must be >= 0 (got {compile_block.get('stage_upload_ms')!r})")
        checker.check(positive(rec.get("wall_s")) and positive(rec.get("elapsed_s_total")), cid + ".wall", rel(path),
                      f"wall_s/elapsed_s_total must be finite and > 0 (got {rec.get('wall_s')!r}/"
                      f"{rec.get('elapsed_s_total')!r})")
    if warm is None or fresh is None or main_cell is None:
        return
    warm_env = warm.get("env") if isinstance(warm.get("env"), dict) else {}
    fresh_env = fresh.get("env") if isinstance(fresh.get("env"), dict) else {}
    main_env = main_cell.get("env") if isinstance(main_cell.get("env"), dict) else {}
    checker.check(warm_env.get("TRITON_CACHE_DIR") == TRITON_CACHE == main_env.get("TRITON_CACHE_DIR"),
                  "extra[compile_econ.warm].cache_dir", rel(warm_path),
                  f"the warm re-run must reuse the campaign cache {TRITON_CACHE} (got {warm_env.get('TRITON_CACHE_DIR')!r})")
    checker.check(fresh_env.get("TRITON_CACHE_DIR") not in (None, TRITON_CACHE), "extra[compile_econ.fresh].cache_dir",
                  rel(fresh_path),
                  f"the fresh run must use a different, empty cache (got {fresh_env.get('TRITON_CACHE_DIR')!r})")
    warm_first = (warm.get("compile") or {}).get("first_call_ms")
    fresh_first = (fresh.get("compile") or {}).get("first_call_ms")
    main_first = ref_value(main_cell, ("compile.first_call_ms", "cold_call_ms"))
    checker.check(finite(warm_first) and finite(fresh_first) and fresh_first > warm_first,
                  "extra[compile_econ.fresh_over_warm]", rel(fresh_path),
                  f"a cold cache must inflate the first call (fresh {fresh_first!r} <= warm {warm_first!r})")
    if finite(warm_first) and finite(fresh_first) and fresh_first > warm_first:
        speedup = fresh_first / warm_first
        checker.check(1.2 <= speedup <= 3.0, "extra[compile_econ.fresh_ratio]", rel(fresh_path),
                      f"the cold-cache penalty must be the documented factor-of-~2 "
                      f"(got {speedup:.3f}x = {100 * (speedup - 1):.0f} %)")
        # the RUNLOG prices the penalty against the campaign cell's first call (the fresh run's
        # own `compile_ms` delta is quoted as the absolute figure): recompute both
        doc = read_text(RUNLOG)
        doc_speedup = fresh_first / main_first if positive(main_first) else None
        compile_delta = (fresh.get("compile") or {}).get("compile_ms", 0.0) \
            - (main_cell.get("compile") or {}).get("compile_ms", 0.0)
        band = [100 * (speedup - 1), 100 * (doc_speedup - 1) if doc_speedup else None,
                100 * ((fresh.get("compile") or {}).get("compile_ms", 0.0)
                       / (warm.get("compile") or {}).get("compile_ms", 1.0) - 1)]
        tokens = pct_tokens(doc)
        checker.check(doc_speedup is not None and quoted_ratio(doc, doc_speedup),
                      "extra[compile_econ.doc_speedup]", rel(RUNLOG),
                      f"RUNLOG.md must quote the recomputed cold-cache speed-up "
                      f"{doc_speedup:.2f}x (fresh first call / campaign first call)")
        checker.check(any(abs(token - value) <= 1.0 for token in tokens for value in band if value is not None),
                      "extra[compile_econ.doc_ratio]", rel(RUNLOG),
                      f"RUNLOG.md must quote a cold-cache penalty inside the recomputable band "
                      f"{[round(v, 1) for v in band if v is not None]} % (quoted tokens: {tokens})")
        checker.check(quoted_ratio(doc, abs(compile_delta) / 1000.0) or f"{abs(compile_delta):.1f}" in doc
                      or f"{abs(compile_delta):,.1f}".replace(",", " ") in doc,
                      "extra[compile_econ.doc_delta]", rel(RUNLOG),
                      f"RUNLOG.md must quote the recomputed compile_ms delta {abs(compile_delta):.1f} ms")
    checker.check(finite(warm_first) and finite(main_first)
                  and abs(warm_first - main_first) <= 0.10 * max(main_first, 1e-9),
                  "extra[compile_econ.warm_matches_campaign]", rel(warm_path),
                  f"the warm re-run must reproduce the campaign cell's first call within 10 % "
                  f"({warm_first!r} vs {main_first!r})")
    counts = read_text(counts_path)
    fields = dict(re.findall(r"(dirs_before|files_before|dirs_after|files_after)=(\d+)", counts))
    warm_line = "warm_cache=" in counts and f"dirs_before={fields.get('dirs_before')}" in counts
    checker.check(bool(fields) and warm_line, "extra[compile_econ.counts.parse]", rel(counts_path),
                  f"cache_filecounts.txt must carry dirs_before/files_before/dirs_after/files_after "
                  f"(parsed {fields!r})")
    if fields:
        before, after = int(fields["dirs_before"]), int(fields["dirs_after"])
        fb, fa = int(fields["files_before"]), int(fields["files_after"])
        checker.check(after == before and fa == fb and fa > 0, "extra[compile_econ.warm_counts]", rel(counts_path),
                      f"a fully warm compile must add nothing to the cache (got {before}/{fb} -> {after}/{fa})")
    doc = read_text(RUNLOG)
    for token, cid in (("732.7", "extra[compile_econ.warm_doc]"), ("**1785.4**", "extra[compile_econ.fresh_doc]"),
                       ("41 dirs / 251 files", "extra[compile_econ.warm_counts_doc]"),
                       ("0 → 3 dirs / 9 files", "extra[compile_econ.fresh_counts_doc]")):
        checker.check(token in doc, cid, rel(RUNLOG), f"RUNLOG.md must quote the compile-economics value {token}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def collect_primary(checker):
    """Gate the 10 primary-window cells and return their headline medians."""
    commits = set()
    for cell_id in CELL_IDS:
        rec = load_json(checker, os.path.join(W1D, cell_id + ".json"), f"cell[{cell_id}].json") or {}
        value = rec.get("evogp_commit")
        if isinstance(value, str):
            commits.add(value)
    checker.check(len(commits) == 1, "primary.commit", "results/" + PRIMARY,
                  f"every primary cell must record the same evogp_commit (got {sorted(commits)})")
    commit = sorted(commits)[0] if len(commits) == 1 else None
    checker.check(commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
                  "primary.commit.hex", "results/" + PRIMARY,
                  f"the window must record a 40-hex evogp_commit (got {commit!r})")
    medians = {}
    for cell_id in CELL_IDS:
        rec = load_json(checker, os.path.join(W1D, cell_id + ".json"), f"cell[{cell_id}].json")
        if rec is None:
            continue
        check_cell(rec, cell_id, PRIMARY, checker, commit)
        value = envelope_ms(rec)
        if positive(value):
            medians[cell_id] = float(value)
        else:
            checker.check(False, f"cell[{cell_id}].median", rel(os.path.join(W1D, cell_id + ".json")),
                          f"the headline median must be finite and > 0 (got {value!r})")
    checker.check(len(medians) == len(CELL_IDS), "primary.medians", "results/" + PRIMARY,
                  f"all {len(CELL_IDS)} cells must expose a headline median (got {len(medians)})")
    anomaly_commits = set()
    for cell_id in CELL_IDS:
        rec = load_json(checker, os.path.join(W1C, cell_id + ".json"), f"anomaly[{cell_id}].json") or {}
        if isinstance(rec.get("evogp_commit"), str):
            anomaly_commits.add(rec["evogp_commit"])
    checker.check(anomaly_commits == {commit}, "crosswindow.commit", "results/" + ANOMALY,
                  f"both windows must run one checkout: {sorted(anomaly_commits)} vs {commit}")
    return medians


def main(argv=None):
    parser = argparse.ArgumentParser(description="R6 Triton-round acceptance gate (stdlib-only, GPU-free)")
    parser.add_argument("--verbose", action="store_true", help="print the recomputed per-cell table too")
    args = parser.parse_args(argv)
    checker = Checker()
    print(f"check_r6.py: round tree = {rel(HERE)}")
    print(f"check_r6.py: primary = {PRIMARY}, anomaly = {ANOMALY}, validation = {rel(VALIDATION)}")
    medians = collect_primary(checker)
    rows = check_window(PRIMARY, checker, primary=True)
    check_anomaly_window(checker, medians)
    check_headlines(checker, medians)
    check_r5_anchors(checker, medians)
    check_runlog(checker, rows)
    check_validation(checker)
    check_kernel_host(checker)
    check_compile_econ(checker)
    if args.verbose:
        print()
        for cell_id in CELL_IDS:
            value = medians.get(cell_id)
            print(f"  {cell_id:<18} {value if value is None else '%.6f ms' % value}")
    print()
    if checker.failures:
        print(f"failures ({len(checker.failures)}):")
        for failure in checker.failures:
            print(f"  FAIL {failure}")
        print()
    else:
        print("all checks passed")
    for note in checker.notes:
        print(f"  NOTE {note}")
    print()
    print(f"check_r6: 14 records / {checker.checks} checks / {len(checker.failures)} failures")
    print(f"check_r6: +10 anomaly cells, +5 kernel-host fact sheets, +2 window_meta, +2 check_deps, "
          f"+30 cell logs, +1 RUNLOG")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
