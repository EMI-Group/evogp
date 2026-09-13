#!/usr/bin/env python3
"""``check_r5.py`` -- stdlib-only, GPU-free validator for the R5 RCA evidence records.

R5 (``test/benchmark/r5/``) is an **analysis-only** round: it measures why the new ETL
"true jit" engine's dense route is ~8-11x slower than the legacy CUDA engine on the
type-1 end-to-end ``problem.evaluate(forest)`` call.  Its evidence lives as immutable
JSON/log/text records under ``results/`` (see ``README.md`` section "Evidence map").

This checker validates those records without importing anything from ``evogp``, ``etl``,
``torch`` or ``numpy`` -- only ``argparse`` / ``json`` / ``pathlib`` / ``sys`` /
``statistics``-style arithmetic.  It resolves every path relative to its own location, so
it runs from any CWD with the system ``python3`` (no venv needed, no GPU, no torch)::

    python3 test/benchmark/r5/check_r5.py                  # the committed record tree
    python3 test/benchmark/r5/check_r5.py --results DIR    # a same-layout tree

What it asserts (hard checks; a failure means "the record no longer says what the round
says it says"):

* existence of every expected record (window cells, decomposition, floors, A/B, minimal
  loop, dispatch dumps / row series / kineto facts, agreement pre-flight, env + tool recon);
* structural stamps per record family (evogp commit, ETL identity, env knobs,
  ``CUDA_VISIBLE_DEVICES=6``, ``ETL_DISABLE_LOCATIONS=1``, task/side, call shapes) and the
  per-cell medians recomputed from the stored per-call arrays;
* the **Q4 headline recomputed from the JSONs**: 4-dp agreement with the quoted medians
  (0.6461 / 0.6372 / 5.3823 / 5.7912 / 1.2142 / 1.4612 ms) and the ratios e2e/legacy
  8.33x / 9.09x, pure/legacy 1.88x / 2.29x within +-2 %, plus ``window_meta.txt`` showing
  9/9 cells exit 0 and a nil ``--stage-probe`` tax;
* the Q1 decomposition reconciles (>= 8 buckets, sum of exclusive medians == the recorded
  ``sum_exclusive_ms``, the sum inside 0.90-1.10 of the call total, recorded reconciliation
  >= 99 %, the dominant buckets present) and Q7 shows no warm compile churn;
* the Q6 floor matrix has all six arms with only the *safe* inequalities (``M0 >= M3``,
  ``M0 >= M4``, ``M0 >= M5``, ``M5 < M0``, and upload removal ``M3 < M2``), the floors and
  all stored deltas recomputed, equal fitness digests across the equivalent arms;
* the Q3 A/B presents ``A1-A4, L1, L2, E1`` and the **no-engine-tax** claim
  (``|A2 - A3| <= 0.35 ms`` on the total and on the ``etl.run`` span), freshness
  ``L1 - L2 > 1 ms``, ``E1 > A3``, and every stored delta recomputed;
* the Q2 GPU records: dump files on disk, dispatch/executable/copy counts (249/125/61 for
  the 64-tree fixture, 253/127/62 for the 200-tree fixture), the kineto facts
  (7470 kernels / 1830 memcpys / 30 invocations x 310 events), and the row series' schema;
* the Q8 minimal loop lands inside the Q4 band (+-35 % of the same-window e2e median).

**Soft checks** (``RCA.md`` / ``README.md`` presence and the RCA section headers) are printed
as ``NOTE`` lines and never affect the exit code, so this stays green while the write-up is
still being written.

Exit code: 0 when no hard check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

R5_DIR = Path(__file__).resolve().parent
REPO_ROOT = R5_DIR.parents[2]  # <repo>/test/benchmark/r5 -> <repo>
DEFAULT_RESULTS = R5_DIR / "results"

# ---------------------------------------------------------------------------
# round constants (the claims the records are checked against)
# ---------------------------------------------------------------------------
ETL_REV = "96df7da8f2ba39f52cf2f2b61888e67f70471bc9"
ETL_SHADOW_DIR = "/tmp/evogp_etl_patched_87c7562"
GPU_INDEX = "6"
LOCK_PATH = "/tmp/gpbench_critical.lock"

FIXTURE64 = "test/goldens/forest_random.npz"
FIXTURE200 = "test/benchmark/fixtures/forest200_tree64_random.npz"
FIXTURE64_SHA = "0bfb0bbf48cd65832b2c024480bc572de5121675946d3611a2c7a5b234ff9ca0"
FIXTURE200_SHA = "6789ef61ecf8898b9b2a00774ee655e20b0b5be7c72eb861afc8d6b8799741b9"

#: the 9 Q4 window cells, in table order
Q4_CELLS = (
    "q4_leg64",
    "q4_leg200",
    "q4_eval64_iree",
    "q4_eval200_iree",
    "q4_pure64_iree",
    "q4_pure200_iree",
    "q4_eval64_iree_probe",
    "q4_leg64_bracket",
    "q4_eval64_iree_shadow87c7562",
)

#: the six headline medians of the same-window table (warm ms), quoted to 4 dp
Q4_ANCHORS = {
    "q4_leg64": 0.6461,
    "q4_leg200": 0.6372,
    "q4_eval64_iree": 5.3823,
    "q4_eval200_iree": 5.7912,
    "q4_pure64_iree": 1.2142,
    "q4_pure200_iree": 1.4612,
}
#: ratio -> documented value (checked within +-2 %)
Q4_RATIOS = {
    "e2e64/leg64": ("q4_eval64_iree", "q4_leg64", 8.33),
    "e2e200/leg200": ("q4_eval200_iree", "q4_leg200", 9.09),
    "pure64/leg64": ("q4_pure64_iree", "q4_leg64", 1.88),
    "pure200/leg200": ("q4_pure200_iree", "q4_leg200", 2.29),
}

Q4_RERUN_CELLS = ("q4_leg64", "q4_eval64_iree", "q4_eval64_iree_probe")

RUNNER_SOURCES = (
    "frozen-api:build_dense_runner",
    "frozen-api:build_specialized_runner",
    "local-fallback:dense",
    "local-fallback:specialized",
)

#: Q2(a) dispatch dumps: id -> (mode, data rows, dispatches, DtoD copies, executables)
DUMPS = {
    "f64_pure": ("pure", None, 249, 61, 125),
    "f64_e2e": ("e2e", None, 249, 61, 125),
    "f200_pure": ("pure", None, 253, 62, 127),
    "f64_r8": ("pure", 8, 249, 61, 125),
    "f64_r256": ("pure", 256, 249, 61, 125),
    "f200_r128": ("pure", 128, 253, 62, 127),
}
DUMP_FILES = ("dispatch_graph.dot", "scheduling_stats.csv", "dump.log")

#: Q2(c) row-scaling series
ROW_CELLS = ("f64_r8", "f64_r16", "f64_r32", "f64_r64", "f64_r128", "f64_r256", "f200_r64", "f200_r128")
ROW_RERUN_CELLS = ("f64_r8b", "f64_r64b", "f200_r64b", "f200_r96", "f200_r128b")

#: Q6 floor matrix arms (the six measured rungs; the run also records M0R/M3R/M0P controls)
FLOOR_ARMS = ("M0", "M1", "M2", "M3", "M4", "M5")
FLOOR_CALLS = 25

#: Q3 arms r5 asserts (the frozen controls R1/R2/A3_plain_control are checked too, softer)
AB_ARMS = (
    "A1_engine_fresh",
    "A2_engine_staged",
    "A3_frozen_runner",
    "A4_frozen_fresh",
    "L1_engine_fresh_loss",
    "L2_engine_staged_loss",
    "E1_e2e_in_process",
)
AB_EXTRA_ARMS = ("A3_plain_control", "R1_repl_fresh_loss", "R2_repl_fresh_drained")
AB_TAX_LIMIT_MS = 0.35  # |A2 - A3| <= 0.35 ms (the no-engine-tax claim)

#: Q2(d) kineto facts
KINETO_KERNELS = 7470
KINETO_MEMCPYS = 1830
KINETO_INVOCATIONS = 30
KINETO_EVENTS_PER_INVOCATION = 310
KINETO_DISTINCT_KERNELS = 125

#: agreement pre-flight record stems (six arms: 2 fixtures x {numpy, iree, iree_spec})
AGREEMENT_STEMS = (
    "fixture64_numpy",
    "fixture64_iree",
    "fixture64_iree_spec",
    "fixture200_numpy",
    "fixture200_iree",
    "fixture200_iree_spec",
)

#: soft doc checks: RCA.md section headers expected in the round's deliverable
RCA_HEADERS = (
    "TL;DR",
    "Method",
    "anchor",
    "decomposition",
    "GPU",
    "A/B",
    "Floor",
    "Ranked",
    "What would close",
    "Open",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Checker:
    """Counts checks and collects failures / notes (the ``../r4/check_r4.py`` pattern)."""

    def __init__(self):
        self.checks = 0
        self.failures = []
        self.notes = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return bool(condition)

    def note(self, label):
        if label not in self.notes:
            self.notes.append(label)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def positive(value):
    return is_number(value) and float(value) > 0.0


def median(values):
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def rel_close(a, b, tol=1e-6):
    if not (is_number(a) and is_number(b)):
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def near(a, b, tol):
    """Absolute-tolerance comparison (``b`` may be an expected constant)."""
    return is_number(a) and is_number(b) and abs(float(a) - float(b)) <= tol


def read_json(path, checker, label):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except FileNotFoundError:
        checker.check(False, f"{label}: record missing ({path})")
        return None
    except (OSError, ValueError) as exc:
        checker.check(False, f"{label}: unreadable/invalid JSON ({path}): {exc}")
        return None
    if not isinstance(data, dict):
        checker.check(False, f"{label}: top-level JSON value is not an object ({path})")
        return None
    return data


def read_text(path, checker, label):
    try:
        return Path(path).read_text()
    except OSError as exc:
        checker.check(False, f"{label}: unreadable text record ({path}): {exc}")
        return ""


def exists(path, checker, label):
    ok = Path(path).is_file()
    checker.check(ok, f"{label}: expected file missing ({path})")
    return ok


def resolve_record_path(raw, results):
    """Resolve a path stored inside a record (repo-relative or absolute) or return None."""
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (REPO_ROOT, results, R5_DIR):
        probe = base / candidate
        if probe.is_file():
            return probe
    return None


def window_meta_cells(text):
    """Return {cell id: exit code} from a ``window_probe.sh`` meta record's cell table."""
    cells = {}
    for line in text.splitlines():
        if not line or line.startswith((" ", "#")):
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        if not fields[0].startswith(("q4_", "smoke_")):
            continue
        try:
            cells[fields[0]] = int(fields[5])
        except ValueError:
            continue
    return cells


def check_bench_record(rec, checker, label, task=None, side=None, fixture_suffix=None):
    """Structural checks shared by every ``bench_gp.py`` record (eval64 and eval-pure)."""
    checker.check(rec.get("side") in ("new", "old"), f"{label}: side must be new|old")
    if side is not None:
        checker.check(rec.get("side") == side, f"{label}: side must be {side!r}")
    checker.check(rec.get("task") in ("eval64", "eval-pure"), f"{label}: task must be eval64|eval-pure")
    if task is not None:
        checker.check(rec.get("task") == task, f"{label}: task must be {task!r}")
    commit = rec.get("evogp_commit")
    checker.check(isinstance(commit, str) and len(commit) >= 7, f"{label}: evogp_commit stamped")
    checker.check(isinstance(rec.get("etl_version"), str), f"{label}: etl_version stamped")
    checker.check(rec.get("complete") is True, f"{label}: record marked complete")
    env = rec.get("env") or {}
    gpu = rec.get("gpu") or {}
    checker.check(env.get("CUDA_VISIBLE_DEVICES") == GPU_INDEX, f"{label}: env CUDA_VISIBLE_DEVICES=6")
    checker.check(gpu.get("cuda_visible_devices") == GPU_INDEX, f"{label}: gpu block pinned to GPU 6")
    checker.check(env.get("ETL_DISABLE_LOCATIONS") == "1", f"{label}: ETL_DISABLE_LOCATIONS=1")
    if rec.get("side") == "new":
        checker.check(env.get("EVOGP_ETL_BACKEND") == "iree", f"{label}: new side ran on the iree backend")
        checker.check(env.get("EVOGP_ETL_IREE_OPT_LEVEL") == "O1", f"{label}: IREE opt level O1")
        expected_mode = "dense" if rec.get("task") == "eval-pure" else "auto"
        checker.check(
            env.get("EVOGP_EVAL_MODE") == expected_mode,
            f"{label}: EVOGP_EVAL_MODE={expected_mode} for task {rec.get('task')}",
        )
    if fixture_suffix is not None:
        fixture = rec.get("fixture") or ""
        checker.check(
            isinstance(fixture, str) and fixture.endswith(fixture_suffix),
            f"{label}: fixture is {fixture_suffix} (path may be absolute)",
        )


def check_eval64_body(rec, checker, label, reps):
    """``--task eval64`` body: ``reps`` calls, call 0 cold, warm median recomputable."""
    checker.check(rec.get("eval_kind") == "fixture", f"{label}: fixed-fixture eval kind")
    cfg = rec.get("config") or {}
    checker.check(cfg.get("eval_reps") == reps, f"{label}: eval_reps == {reps}")
    calls = rec.get("per_call_ms")
    if not checker.check(isinstance(calls, list) and len(calls) == reps, f"{label}: {reps} per_call_ms entries"):
        return
    checker.check(all(positive(v) for v in calls), f"{label}: every per-call time is a positive number")
    med = rec.get("median_warm_ms")
    checker.check(positive(med), f"{label}: median_warm_ms is positive")
    checker.check(rel_close(med, median(calls[1:]), 1e-6), f"{label}: median_warm_ms == median of warm calls")
    first = rec.get("first_call_ms")
    checker.check(positive(first), f"{label}: first_call_ms is positive")
    checker.check(is_number(first) and is_number(med) and first > med, f"{label}: call 0 (cold) is the slowest")


def check_pure_body(rec, checker, label, reps, invocations):
    """``--task eval-pure`` body: the frozen pure-forward block."""
    pure = rec.get("pure")
    if not checker.check(isinstance(pure, dict), f"{label}: pure block present"):
        return
    checker.check(pure.get("runner_source") in RUNNER_SOURCES, f"{label}: recognized runner_source")
    checker.check(pure.get("mode_resolved") == "dense", f"{label}: mode resolves to dense")
    checker.check(pure.get("backend") == "iree", f"{label}: pure block ran on iree")
    checker.check(pure.get("invocations") == invocations, f"{label}: {invocations} invocations per rep")
    checker.check(pure.get("reps") == reps, f"{label}: {reps} timed reps")
    checker.check(pure.get("n_graph_calls_per_run") == 1, f"{label}: one graph call per invocation")
    per_rep = pure.get("per_invocation_ms")
    blocks = pure.get("block_ms")
    if not checker.check(isinstance(per_rep, list) and len(per_rep) == reps, f"{label}: {reps} per-rep medians"):
        return
    checker.check(isinstance(blocks, list) and len(blocks) == reps, f"{label}: {reps} block totals")
    if isinstance(blocks, list) and len(blocks) == reps:
        checker.check(
            all(rel_close(b, p * invocations, 1e-6) for p, b in zip(per_rep, blocks)),
            f"{label}: block_ms[i] == per_invocation_ms[i] * invocations",
        )
    med = pure.get("median_per_invocation_ms")
    checker.check(positive(med), f"{label}: median_per_invocation_ms is positive")
    checker.check(rel_close(med, median(per_rep), 1e-6), f"{label}: median_per_invocation_ms recomputes")
    checker.check(
        rel_close(pure.get("per_population_evaluate_ms"), med, 1e-9),
        f"{label}: per_population_evaluate_ms == the median (one invocation is one evaluate)",
    )
    checksum = pure.get("output_checksum") or {}
    sha = checksum.get("sha256")
    checker.check(isinstance(sha, str) and len(sha) == 64, f"{label}: pure output_checksum.sha256 stamped")


def check_identity_stamp(rec, checker, label):
    """The probe-family identity stamp (``probe_lib.identity_stamp``)."""
    stamp = rec.get("stamp")
    if not checker.check(isinstance(stamp, dict), f"{label}: stamp block present"):
        return
    for key in ("when", "cwd", "evogp_commit", "evogp_file", "etl_repo", "etl_repo_head", "etl_file", "python"):
        checker.check(isinstance(stamp.get(key), str) and bool(stamp.get(key)), f"{label}: stamp.{key} stamped")
    checker.check(stamp.get("etl_repo_head") == ETL_REV, f"{label}: stamp.etl_repo_head == {ETL_REV[:8]}")
    checker.check(stamp.get("cuda_visible_devices") == GPU_INDEX, f"{label}: stamp.cuda_visible_devices == 6")
    knobs = stamp.get("env_knobs")
    checker.check(isinstance(knobs, dict), f"{label}: stamp.env_knobs present")
    if isinstance(knobs, dict):
        checker.check(knobs.get("ETL_DISABLE_LOCATIONS") == "1", f"{label}: stamp env ETL_DISABLE_LOCATIONS=1")
        checker.check(knobs.get("EVOGP_ETL_BACKEND") == "iree", f"{label}: stamp env EVOGP_ETL_BACKEND=iree")
    gpu = stamp.get("gpu")
    checker.check(isinstance(gpu, (dict, str)) and bool(gpu), f"{label}: stamp.gpu sample present")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def check_environment(results, checker, notes):
    """P0: the environment freeze and the tool reconnaissance."""
    env = results / "env_probe.txt"
    if exists(env, checker, "results/env_probe.txt"):
        text = read_text(env, checker, "env_probe.txt")
        checker.check(ETL_REV in text, "env_probe.txt: ETL revision 96df7da8 pinned")
        checker.check("evogp_commit" in text, "env_probe.txt: evogp commit recorded")
        checker.check("535.247.01" in text, "env_probe.txt: host driver recorded")
        checker.check("gpu7" in text.lower(), "env_probe.txt: the GPU-7 ECC exclusion is recorded")
        checker.check(LOCK_PATH in text, "env_probe.txt: the shared benchmark lock is recorded")
        checker.check(FIXTURE64_SHA in text and FIXTURE200_SHA in text, "env_probe.txt: fixture sha256 pins")

    recon = results / "tool_recon"
    log = recon / "import_log.txt"
    if exists(log, checker, "results/tool_recon/import_log.txt"):
        text = read_text(log, checker, "tool_recon/import_log.txt")
        checker.check(
            "Unknown runtime API function index: 440" in text,
            "tool_recon/import_log.txt: the nsys QDSTRM import failure is the recorded evidence",
        )
    readme = recon / "README.md"
    if exists(readme, checker, "results/tool_recon/README.md"):
        text = read_text(readme, checker, "tool_recon/README.md")
        checker.check("ncu" in text, "tool_recon/README.md: ncu state recorded")
        checker.check("2022.4.2.1" in text, "tool_recon/README.md: nsys version recorded")


def check_backend_agreement(results, checker, notes):
    """The six-arm IREE agreement pre-flight re-run (the gate on every IREE number)."""
    agree = results / "backend_agreement"
    for stem in AGREEMENT_STEMS:
        exists(agree / f"{stem}.json", checker, f"backend_agreement/{stem}.json")
        exists(agree / f"{stem}.log", checker, f"backend_agreement/{stem}.log")
        exists(agree / f"{stem}_values.npy", checker, f"backend_agreement/{stem}_values.npy")
        rec = read_json(agree / f"{stem}.json", checker, f"backend_agreement/{stem}.json")
        if not rec:
            continue
        expected_count = 64 if stem.startswith("fixture64") else 200
        expected_sha = FIXTURE64_SHA if stem.startswith("fixture64") else FIXTURE200_SHA
        checker.check(rec.get("count") == expected_count, f"{stem}: count == {expected_count}")
        checker.check(rec.get("fixture_sha256") == expected_sha, f"{stem}: fixture sha256 matches the npz")
        checker.check(rec.get("nan_count") is not None, f"{stem}: nan_count recorded")
        sha = rec.get("values_sha256")
        checker.check(isinstance(sha, str) and len(sha) == 64, f"{stem}: values_sha256 stamped")
        checker.check(rec.get("iree_opt_level") == "O1", f"{stem}: IREE opt level O1")
        checker.check(isinstance(rec.get("per_call_ms"), list) and rec["per_call_ms"], f"{stem}: per_call_ms recorded")

    gate = agree / "check_agreement_r4.out.txt"
    if exists(gate, checker, "backend_agreement/check_agreement_r4.out.txt"):
        text = read_text(gate, checker, "check_agreement_r4.out.txt")
        checker.check("gate: PASS" in text, "agreement gate: PASS recorded")
        checker.check("0 failure(s)" in text, "agreement gate: zero failures recorded")
        checker.check("PASS-with-note" in text, "agreement gate: the documented PASS-with-note verdict")
    stdout = agree / "run_agreement_r4.stdout.txt"
    if exists(stdout, checker, "backend_agreement/run_agreement_r4.stdout.txt"):
        text = read_text(stdout, checker, "run_agreement_r4.stdout.txt")
        checker.check("records=6" in text, "agreement runner: ran all six arms")


def check_smoke(results, checker, notes):
    """The 3 short pre-window sanity cells."""
    smoke = results / "smoke"
    meta = smoke / "window_meta.txt"
    if exists(meta, checker, "smoke/window_meta.txt"):
        text = read_text(meta, checker, "smoke/window_meta.txt")
        checker.check("ok=3 skipped=0 failed=0" in text, "smoke/window_meta.txt: 3/3 smoke cells OK")
        cells = window_meta_cells(text)
        for cell in ("smoke_leg64", "smoke_eval64_iree", "smoke_pure64_iree"):
            checker.check(cells.get(cell) == 0, f"smoke/window_meta.txt: {cell} exit 0")
    for cell, task in (
        ("smoke_leg64", "eval64"),
        ("smoke_eval64_iree", "eval64"),
        ("smoke_pure64_iree", "eval-pure"),
    ):
        rec = read_json(smoke / f"{cell}.json", checker, f"smoke/{cell}.json")
        exists(smoke / f"{cell}.log", checker, f"smoke/{cell}.log")
        if not rec:
            continue
        suffix = FIXTURE64 if "64" in cell else FIXTURE200
        check_bench_record(rec, checker, cell, task=task, fixture_suffix=suffix)
        if task == "eval64":
            reps = (rec.get("config") or {}).get("eval_reps")
            if checker.check(isinstance(reps, int) and reps >= 1, f"{cell}: smoke eval_reps recorded"):
                check_eval64_body(rec, checker, cell, reps)
        else:
            reps = (rec.get("config") or {}).get("reps")
            invocations = (rec.get("config") or {}).get("invocations")
            if checker.check(
                isinstance(reps, int) and reps >= 1 and isinstance(invocations, int) and invocations >= 1,
                f"{cell}: smoke pure reps/invocations recorded",
            ):
                check_pure_body(rec, checker, cell, reps, invocations)


def check_q4_window(results, checker, notes):
    """Q4: the same-window 3-way anchor, its ratio arithmetic, the probe bridge, the control."""
    window = results / "q4_window"
    meta = window / "window_meta.txt"
    if exists(meta, checker, "q4_window/window_meta.txt"):
        text = read_text(meta, checker, "q4_window/window_meta.txt")
        checker.check("ok=9 skipped=0 failed=0" in text, "window_meta.txt: 9/9 window cells OK")
        checker.check(f"rev={ETL_REV}" in text, "window_meta.txt: ETL identity stamped")
        checker.check(f"CUDA_VISIBLE_DEVICES={GPU_INDEX}" in text, "window_meta.txt: GPU 6 pinned")
        checker.check("## deviations and protocol notes" in text, "window_meta.txt: deviations section present")
        cells = window_meta_cells(text)
        for cell in Q4_CELLS:
            checker.check(cell in cells, f"window_meta.txt: {cell} in the cell table")
            checker.check(cells.get(cell) == 0, f"window_meta.txt: {cell} exit 0")
            checker.check(f"{cell}.json" in text, f"window_meta.txt: {cell} json path recorded")
    exists(window / "SUMMARY.md", checker, "q4_window/SUMMARY.md")

    medians = {}
    for cell in Q4_CELLS:
        path = window / f"{cell}.json"
        rec = read_json(path, checker, f"q4_window/{cell}.json")
        exists(window / f"{cell}.log", checker, f"q4_window/{cell}.log")
        if not rec:
            continue
        task = rec.get("task")
        # cell ids embed the fixture size (``...64...`` vs ``...200...``); the record stores the path
        # exactly as it was passed on the command line (absolute, if the cell ran in another worktree),
        # so only the basename can be compared.
        suffix = FIXTURE200 if "200" in cell else FIXTURE64
        check_bench_record(rec, checker, cell, fixture_suffix=suffix)
        if task == "eval64":
            check_eval64_body(rec, checker, cell, 25)
            medians[cell] = rec.get("median_warm_ms")
        else:
            check_pure_body(rec, checker, cell, 5, 100)
            pure = rec.get("pure") or {}
            medians[cell] = pure.get("median_per_invocation_ms")

    # the six headline medians, to 4 dp, and the four ratios
    for cell, expected in Q4_ANCHORS.items():
        got = medians.get(cell)
        checker.check(
            is_number(got) and round(float(got), 4) == expected,
            f"{cell}: medians to 4 dp quote {expected} ms (got {got})",
        )
    for name, (hi, lo, expected_ratio) in Q4_RATIOS.items():
        a, b = medians.get(hi), medians.get(lo)
        if not (positive(a) and positive(b)):
            checker.check(False, f"ratio {name}: both medians present")
            continue
        ratio = float(a) / float(b)
        checker.check(
            abs(ratio - expected_ratio) <= 0.02 * expected_ratio,
            f"ratio {name}: recorded medians give {ratio:.3f}x, expected {expected_ratio}x +-2 %",
        )

    # the --stage-probe bridge cell: the probe's own tax must be nil
    probe = read_json(window / "q4_eval64_iree_probe.json", checker, "q4_window/q4_eval64_iree_probe.json")
    if probe:
        stages = probe.get("stages")
        if checker.check(isinstance(stages, dict), "q4 probe: stages block present"):
            reps = stages.get("reps") or []
            records = (reps[0] or {}).get("records") if reps and isinstance(reps[0], dict) else None
            if checker.check(isinstance(records, list) and len(records) == 25, "q4 probe: 25 instrumented records"):
                warm_deltas = [
                    float(r["probe_span_ms"]) - float(r["total_ms"])
                    for r in records[1:]
                    if is_number(r.get("probe_span_ms")) and is_number(r.get("total_ms"))
                ]
                if checker.check(bool(warm_deltas), "q4 probe: warm span-vs-total deltas computable"):
                    tax = median(warm_deltas)
                    checker.check(
                        abs(tax) <= 0.5,
                        f"q4 probe: stage-probe tax {tax:+.3f} ms/call is nil (<= 0.5 ms)",
                    )

    shadow_log = window / "q4_eval64_iree_shadow87c7562.log"
    if shadow_log.is_file():
        text = read_text(shadow_log, checker, "q4 shadow cell log")
        checker.check(ETL_SHADOW_DIR in text, "q4 shadow cell: the 87c7562 shadow checkout is recorded in the log")

    # the post-window control (drift answer + probe-tax A/B)
    rerun = window / "rerun_1"
    meta = rerun / "window_meta.txt"
    if exists(meta, checker, "q4_window/rerun_1/window_meta.txt"):
        text = read_text(meta, checker, "q4_window/rerun_1/window_meta.txt")
        checker.check("ok=3 skipped=0 failed=0" in text, "rerun_1/window_meta.txt: 3/3 control cells OK")
        checker.check("DEVIATION_NOTE" in text or "POST-WINDOW CONTROL" in text, "rerun_1: control purpose stamped")
        cells = window_meta_cells(text)
        for cell in Q4_RERUN_CELLS:
            checker.check(cells.get(cell) == 0, f"rerun_1/window_meta.txt: {cell} exit 0")
    rerun_med = {}
    for cell in Q4_RERUN_CELLS:
        rec = read_json(rerun / f"{cell}.json", checker, f"q4_window/rerun_1/{cell}.json")
        exists(rerun / f"{cell}.log", checker, f"q4_window/rerun_1/{cell}.log")
        if not rec:
            continue
        check_bench_record(rec, checker, f"rerun_1/{cell}", fixture_suffix=FIXTURE64)
        check_eval64_body(rec, checker, f"rerun_1/{cell}", 25)
        rerun_med[cell] = rec.get("median_warm_ms")
    if positive(rerun_med.get("q4_leg64")) and positive(medians.get("q4_leg64")):
        drift = float(rerun_med["q4_leg64"]) / float(medians["q4_leg64"]) - 1.0
        checker.check(abs(drift) <= 0.20, f"rerun_1: legacy anchor repeats within +-20 % (got {drift:+.1%})")
    if positive(rerun_med.get("q4_eval64_iree")) and positive(rerun_med.get("q4_eval64_iree_probe")):
        probe_delta = float(rerun_med["q4_eval64_iree_probe"]) / float(rerun_med["q4_eval64_iree"]) - 1.0
        checker.check(
            abs(probe_delta) <= 0.20,
            f"rerun_1: probe cell matches its plain twin within +-20 % (got {probe_delta:+.1%})",
        )

    # cross-node: the pure forward reproduces the r4 after-record bit-for-bit
    r4_after = REPO_ROOT / "test/benchmark/r4/results/after"
    for cell, r4_name in (("q4_pure64_iree", "pure64_dense_iree.json"), ("q4_pure200_iree", "pure200_dense_iree.json")):
        rec = read_json(window / f"{cell}.json", checker, f"q4_window/{cell}.json")
        ref_path = r4_after / r4_name
        if rec is None:
            continue
        if not ref_path.is_file():
            notes.append(f"{cell}: r4 after-record {r4_name} absent -> bit-equality cross-check skipped")
            continue
        ref = read_json(ref_path, checker, f"r4 after-record {r4_name}")
        if ref is None:
            continue
        mine = ((rec.get("pure") or {}).get("output_checksum") or {}).get("sha256")
        theirs = ((ref.get("pure") or {}).get("output_checksum") or {}).get("sha256")
        checker.check(
            isinstance(mine, str) and mine == theirs,
            f"{cell}: pure output_checksum.sha256 bit-equal to the r4 after-record",
        )


def check_q1_decomp(results, checker, notes):
    """Q1/Q5/Q7: the per-call decomposition, its reconciliation and the cache behaviour."""
    decomp_dir = results / "q1_decomp"
    exists(decomp_dir / "SUMMARY.md", checker, "q1_decomp/SUMMARY.md")
    fitness_shas = {}
    for fx in ("f64", "f200"):
        label = f"q1_decomp/decomp_{fx}.json"
        rec = read_json(decomp_dir / f"decomp_{fx}.json", checker, label)
        exists(decomp_dir / f"decomp_{fx}.log", checker, f"q1_decomp/decomp_{fx}.log")
        if not rec:
            continue
        check_identity_stamp(rec, checker, label)
        stamp = rec.get("stamp") or {}
        checker.check(stamp.get("etl_head_probe") == ETL_REV, f"{label}: stamp.etl_head_probe matches ETL HEAD")
        route = rec.get("route") or {}
        checker.check(route.get("should_use_dense") is True, f"{label}: route resolves to the dense evaluator")
        warm = rec.get("warm") or {}
        table = warm.get("table") or {}
        # the route counter spans the whole probe (measured calls + the control runs + the cold rebuild),
        # so it must at least cover the measured table
        checker.check(
            is_number(route.get("dense_calls")) and route["dense_calls"] >= table.get("calls", 0),
            f"{label}: the dense route served every measured call",
        )
        checker.check(table.get("calls") == 25, f"{label}: 25 measured calls")
        checker.check(table.get("warm_calls") == 24, f"{label}: 24 warm calls (call 0 cold)")
        checker.check(positive(table.get("median_total_ms")), f"{label}: median_total_ms positive")
        checker.check(positive(table.get("median_call_ms")), f"{label}: median_call_ms positive")
        checker.check(
            is_number(table.get("first_call_ms")) and table["first_call_ms"] > table["median_total_ms"],
            f"{label}: the first call pays the cold compile",
        )
        rows = warm.get("rows")
        checker.check(isinstance(rows, list) and len(rows) == 25, f"{label}: 25 per-call rows recorded")

        components = warm.get("components") or []
        if not checker.check(isinstance(components, list) and len(components) >= 8, f"{label}: >= 8 exclusive buckets"):
            continue
        names = {c.get("component") for c in components}
        for wanted in ("etl_run", "etl_tensor_to_payload", "prepare_dense_self", "run_loss_plan_self"):
            checker.check(wanted in names, f"{label}: bucket {wanted} present")
        shares = {c.get("component"): c for c in components}
        summed = sum(float(c.get("median_self_ms") or 0.0) for c in components)
        checker.check(
            rel_close(summed, warm.get("sum_exclusive_ms"), 1e-6),
            f"{label}: sum of exclusive medians == sum_exclusive_ms",
        )
        checker.check(
            0.90 <= summed / float(table["median_total_ms"]) <= 1.10,
            f"{label}: exclusive sum within 0.90-1.10 of the call total (the wrap set is complete)",
        )
        recon = (warm.get("reconciliation") or {}).get("median_pct")
        checker.check(is_number(recon) and float(recon) >= 99.0, f"{label}: recorded reconciliation >= 99 %")
        per_call = (warm.get("reconciliation") or {}).get("per_call_pct") or []
        checker.check(len(per_call) == 24, f"{label}: per-call reconciliation recorded for 24 warm calls")
        both = sum(
            float(shares[k].get("median_self_ms") or 0.0)
            for k in ("etl_run", "etl_tensor_to_payload")
            if k in shares
        )
        checker.check(
            both / float(table["median_total_ms"]) >= 0.50,
            f"{label}: graph call + payload uploads are >= 50 % of the warm call",
        )
        counts = warm.get("counts") or {}
        # the counters are whole-probe counters: the 25 measured calls + the 5 region-sync control
        # calls + the deliberate cold rebuild (= route.dense_calls), not just the warm table.
        probe_calls = route.get("dense_calls")
        if checker.check(
            is_number(probe_calls) and probe_calls >= table["calls"],
            f"{label}: route.dense_calls covers the measured table",
        ):
            checker.check(counts.get("etl_run") == probe_calls, f"{label}: one etl.run per dense call")
            checker.check(
                counts.get("etl_tensor_to_payload") == 10 * probe_calls,
                f"{label}: ten payload uploads per dense call",
            )
            checker.check(
                counts.get("build_dense_build_cached") == probe_calls,
                f"{label}: every call touches the build_cached prologue (no compile churn)",
            )
            checker.check(counts.get("prepare_dense_self") == probe_calls, f"{label}: one dense decode per call")
        controls = rec.get("controls") or {}
        plain = controls.get("plain") or {}
        cold = controls.get("cold") or {}
        checker.check((plain.get("table") or {}).get("calls") == 5, f"{label}: the uninstrumented control ran 5 calls")
        checker.check((plain.get("cache") or {}).get("misses") == 1, f"{label}: warm cache had exactly one compile")
        checker.check((plain.get("cache") or {}).get("hits") >= 20, f"{label}: the warm-cache hits are recorded")
        checker.check((cold.get("cache") or {}).get("hits") == 0, f"{label}: the cold control starts from an empty cache")
        checker.check(
            is_number((cold.get("table") or {}).get("first_call_ms")) and cold["table"]["first_call_ms"] > 1000.0,
            f"{label}: the deliberate cold rebuild costs seconds",
        )
        checker.check(
            (controls.get("sync_region") or {}).get("table", {}).get("calls") == 5,
            f"{label}: the region-sync control ran 5 calls",
        )
        install = rec.get("install_status") or {}
        checker.check(len(install.get("wrapped") or []) >= 8, f"{label}: >= 8 dense call sites wrapped")
        fitness = rec.get("fitness") or {}
        sha = fitness.get("sha256")
        checker.check(isinstance(sha, str) and len(sha) == 64, f"{label}: fitness sha256 stamped")
        checker.check(fitness.get("all_equal") is True, f"{label}: the warm calls all reproduce the same fitness")
        if isinstance(sha, str):
            fitness_shas[fx] = sha

    cprofile = read_json(decomp_dir / "cprofile_f64.json", checker, "q1_decomp/cprofile_f64.json")
    if cprofile:
        stamp = cprofile.get("stamp") or {}
        checker.check(stamp.get("etl_repo_head") == ETL_REV, "cprofile_f64.json: ETL identity stamped")
        checker.check(isinstance(stamp.get("evogp_commit"), str), "cprofile_f64.json: evogp commit stamped")
        checker.check(
            isinstance(cprofile.get("cprofile_text_written"), str),
            "cprofile_f64.json: points at the cProfile text table",
        )
    txt = decomp_dir / "cprofile_f64.txt"
    if exists(txt, checker, "q1_decomp/cprofile_f64.txt"):
        checker.check(txt.stat().st_size > 500, "cprofile_f64.txt: the cProfile table has content")
    return fitness_shas


def check_q8_minimal(results, checker, notes, q4_medians):
    """Q8: the minimal loop (no harness, no probe) lands in the Q4 band."""
    q8 = results / "q8_minimal"
    exists(q8 / "SUMMARY.md", checker, "q8_minimal/SUMMARY.md")
    for fx, suffix in (("f64", FIXTURE64), ("f200", FIXTURE200)):
        label = f"q8_minimal/minimal_{fx}.json"
        rec = read_json(q8 / f"minimal_{fx}.json", checker, label)
        exists(q8 / f"minimal_{fx}.log", checker, f"q8_minimal/minimal_{fx}.log")
        if not rec:
            continue
        checker.check(rec.get("probe") == "r5/minimal_loop.py", f"{label}: written by minimal_loop.py")
        cfg = rec.get("config") or {}
        checker.check(isinstance(cfg.get("fixture"), str) and cfg["fixture"].endswith(suffix), f"{label}: fixture path")
        checker.check(cfg.get("reps") == 25, f"{label}: 25 measured calls")
        calls = rec.get("per_call_ms")
        if not checker.check(isinstance(calls, list) and len(calls) == 25, f"{label}: 25 per-call times"):
            continue
        med = rec.get("median_warm_ms")
        checker.check(positive(med), f"{label}: median_warm_ms positive")
        checker.check(rel_close(med, median(calls[1:]), 1e-6), f"{label}: median recomputes from the warm calls")
        checker.check(
            is_number(rec.get("first_call_ms")) and rec["first_call_ms"] > 1000.0,
            f"{label}: the first call is the cold compile",
        )
        stamp = rec.get("stamp") or {}
        checker.check(stamp.get("etl_repo_head") == ETL_REV, f"{label}: stamp.etl_repo_head matches ETL HEAD")
        checker.check(isinstance(stamp.get("evogp_commit"), str), f"{label}: evogp commit stamped")
        checker.check(stamp.get("cuda_visible_devices") == GPU_INDEX, f"{label}: GPU 6 stamped")
        etl_file = rec.get("etl_file")
        checker.check(isinstance(etl_file, str) and etl_file.endswith(("etl/__init__.py", "etl.py")), f"{label}: etl file recorded")
        cell = "q4_eval64_iree" if fx == "f64" else "q4_eval200_iree"
        reference = q4_medians.get(cell)
        if positive(reference) and positive(med):
            delta = float(med) / float(reference) - 1.0
            checker.check(
                abs(delta) <= 0.35,
                f"{label}: inside the Q4 same-window band (+-35 % of {cell}, got {delta:+.1%})",
            )
        else:
            notes.append(f"{label}: Q4 reference median unavailable -> band check skipped")


def check_q2_gpu(results, checker, notes):
    """Q2: the compile-side dispatch dumps, the row series and the kineto facts."""
    q2 = results / "q2_gpu"
    exists(q2 / "SUMMARY.md", checker, "q2_gpu/SUMMARY.md")
    for cell, (mode, rows, dispatches, copies, executables) in DUMPS.items():
        label = f"q2_gpu/dumps/{cell}"
        rec = read_json(q2 / "dumps" / f"{cell}.json", checker, f"{label}.json")
        for name in DUMP_FILES:
            exists(q2 / "dumps" / cell / name, checker, f"{label}/{name}")
        exists(q2 / "dumps" / cell / "executables" / "manifest.txt", checker, f"{label}/executables/manifest.txt")
        if not rec:
            continue
        check_identity_stamp(rec, checker, label)
        cfg = rec.get("config") or {}
        checker.check(cfg.get("mode") == mode, f"{label}: mode {mode}")
        if rows is not None:
            checker.check(cfg.get("data_rows") == rows, f"{label}: --data-rows {rows}")
        graph = rec.get("dispatch_graph") or {}
        checker.check(graph.get("exists") is True, f"{label}: dispatch_graph.dot exists")
        checker.check(graph.get("dispatch_nodes") == dispatches, f"{label}: {dispatches} dispatch nodes")
        checker.check(
            is_number(graph.get("lines")) and graph["lines"] > 1000 and is_number(graph.get("bytes")) and graph["bytes"] > 1000,
            f"{label}: the raw .dot is non-trivial",
        )
        checker.check(resolve_record_path(graph.get("path"), results) is not None, f"{label}: .dot resolves on disk")
        stats = rec.get("scheduling_stats") or {}
        aggregate = stats.get("aggregate") or {}
        checker.check(_as_int(aggregate.get("Dispatches")) == dispatches, f"{label}: aggregate Dispatches == {dispatches}")
        checker.check(_as_int(aggregate.get("Copies")) == copies, f"{label}: {copies} DtoD copies")
        checker.check(_as_int(aggregate.get("Executables")) == executables, f"{label}: {executables} executables")
        exe = rec.get("executables") or {}
        checker.check(bool(exe.get("files")), f"{label}: linked executable artifact listed")
        checker.check(
            is_number(rec.get("compile_call_ms")) and rec["compile_call_ms"] > 1000.0,
            f"{label}: the cold compile costs seconds",
        )
        checker.check(isinstance(rec.get("warm_calls_ms"), list) and rec["warm_calls_ms"], f"{label}: warm calls recorded")

    for cell in ROW_CELLS:
        label = f"q2_gpu/rows/{cell}.json"
        rec = read_json(q2 / "rows" / f"{cell}.json", checker, label)
        exists(q2 / "rows" / f"{cell}.log", checker, f"q2_gpu/rows/{cell}.log")
        if rec:
            check_row_record(rec, checker, label)
    for cell in ROW_RERUN_CELLS:
        label = f"q2_gpu/rows/rerun_1/{cell}.json"
        rec = read_json(q2 / "rows" / "rerun_1" / f"{cell}.json", checker, label)
        exists(q2 / "rows" / "rerun_1" / f"{cell}.log", checker, f"q2_gpu/rows/rerun_1/{cell}.log")
        if rec:
            check_row_record(rec, checker, label)
    meta = q2 / "rows" / "rows_meta.txt"
    if exists(meta, checker, "q2_gpu/rows/rows_meta.txt"):
        text = read_text(meta, checker, "rows_meta.txt")
        checker.check(ETL_REV in text, "rows_meta.txt: ETL identity stamped")
        checker.check(f"CUDA_VISIBLE_DEVICES={GPU_INDEX}" in text, "rows_meta.txt: GPU 6 pinned")
    manifest = q2 / "rows" / "rows_manifest.txt"
    if exists(manifest, checker, "q2_gpu/rows/rows_manifest.txt"):
        text = read_text(manifest, checker, "rows_manifest.txt")
        for cell in ROW_CELLS:
            checker.check(f"{cell} " in text or f"{cell}\t" in text, f"rows_manifest.txt: {cell} listed with an exit code")

    # the row-scaling claim that is safe on both record generations: the f200 series rises r64 -> r128
    # (the ``*b`` cells of the re-run live in ``rows/rerun_1/``, not next to the primary cells).
    row_series = (
        (
            "q2_gpu/rows/f200_r64.json",
            q2 / "rows" / "f200_r64.json",
            "q2_gpu/rows/f200_r128.json",
            q2 / "rows" / "f200_r128.json",
        ),
        (
            "q2_gpu/rows/rerun_1/f200_r64b.json",
            q2 / "rows" / "rerun_1" / "f200_r64b.json",
            "q2_gpu/rows/rerun_1/f200_r128b.json",
            q2 / "rows" / "rerun_1" / "f200_r128b.json",
        ),
    )
    for base_label, base_path, top_label, top_path in row_series:
        base = read_json(base_path, checker, base_label)
        top = read_json(top_path, checker, top_label)
        if base and top:
            checker.check(
                float(top["median_per_invocation_ms"]) > float(base["median_per_invocation_ms"]),
                f"{top_label}: the f200 series rises from the r64 cell (the recorded row dependence)",
            )

    kineto = read_json(q2 / "kineto_f64.json", checker, "q2_gpu/kineto_f64.json")
    if kineto:
        check_identity_stamp(kineto, checker, "q2_gpu/kineto_f64")
        checker.check(
            "iree kernels visible" in str(kineto.get("verdict")),
            "kineto_f64.json: verdict says the IREE kernels are visible",
        )
        checker.check(is_number(kineto.get("construct_ms")) and kineto["construct_ms"] > 1000.0, "kineto: cold compile recorded")
    for name in ("trace_facts.json", "trace_facts.txt", "kineto_trace.json.gz", "kineto_key_averages.txt", "kineto.log"):
        exists(q2 / "kineto_f64" / name, checker, f"q2_gpu/kineto_f64/{name}")
    facts = read_json(q2 / "kineto_f64" / "trace_facts.json", checker, "q2_gpu/kineto_f64/trace_facts.json")
    if facts:
        checker.check(facts.get("kernel_calls") == KINETO_KERNELS, f"trace_facts: {KINETO_KERNELS} kernel calls")
        checker.check(facts.get("memcpy_calls") == KINETO_MEMCPYS, f"trace_facts: {KINETO_MEMCPYS} memcpy calls")
        checker.check(facts.get("driver_api_calls") == KINETO_KERNELS, "trace_facts: one driver launch per kernel")
        per_invocation = facts.get("per_invocation") or []
        checker.check(len(per_invocation) == KINETO_INVOCATIONS, f"trace_facts: {KINETO_INVOCATIONS} invocations grouped")
        if per_invocation:
            checker.check(
                per_invocation[0].get("n_events") == KINETO_EVENTS_PER_INVOCATION,
                f"trace_facts: {KINETO_EVENTS_PER_INVOCATION} device events per invocation",
            )
        census = facts.get("kernel_census") or {}
        checker.check(
            census.get("distinct_names") == KINETO_DISTINCT_KERNELS,
            f"trace_facts: {KINETO_DISTINCT_KERNELS} distinct kernels",
        )
        kernel_us = facts.get("kernel_us") or {}
        checker.check(
            is_number(kernel_us.get("median_us")) and 1.0 <= float(kernel_us["median_us"]) <= 3.0,
            "trace_facts: median kernel duration in the 1-3 us launch-bound range",
        )
        device = facts.get("device") or {}
        checker.check(
            is_number(device.get("busy_pct")) and 5.0 <= float(device["busy_pct"]) <= 50.0,
            "trace_facts: device busy fraction is a minority of the span (idle-gap shape)",
        )
        checker.check(is_number(device.get("gap_count")) and device["gap_count"] >= 300, "trace_facts: idle gaps recorded")


def check_row_record(rec, checker, label):
    """A row-series record is a frozen ``--task eval-pure`` cell with ``--data-rows N``."""
    check_bench_record(rec, checker, label, task="eval-pure", side="new")
    cfg = rec.get("config") or {}
    checker.check(cfg.get("n_rows") == cfg.get("data_rows"), f"{label}: n_rows == data_rows")
    checker.check(isinstance(cfg.get("n_rows"), int) and cfg["n_rows"] > 0, f"{label}: row count recorded")
    checker.check(positive(rec.get("median_per_invocation_ms")), f"{label}: warm median positive")
    compile_block = rec.get("compile") or {}
    checker.check(
        is_number(compile_block.get("construct_total_ms")) and compile_block["construct_total_ms"] > 1000.0,
        f"{label}: the row change paid a fresh cold compile",
    )


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_q6_floors(results, checker, notes, fitness_shas):
    """Q6: the floor matrix, its recomputed deltas, the safe inequalities and the controls."""
    floors_dir = results / "q6_floors"
    exists(floors_dir / "SUMMARY.md", checker, "q6_floors/SUMMARY.md")
    exists(floors_dir / "floor_f64.json", checker, "q6_floors/floor_f64.json (contention control)")
    exists(floors_dir / "floor_f64.log", checker, "q6_floors/floor_f64.log")
    for fx in ("f64", "f200"):
        label = f"q6_floors/rerun_1/floor_{fx}.json"
        rec = read_json(floors_dir / "rerun_1" / f"floor_{fx}.json", checker, label)
        exists(floors_dir / "rerun_1" / f"floor_{fx}.log", checker, f"q6_floors/rerun_1/floor_{fx}.log")
        if not rec:
            continue
        checker.check(rec.get("probe") == "r5/floor_probe.py", f"{label}: written by floor_probe.py")
        check_identity_stamp(rec, checker, label)
        for key in ("loadavg", "cpu_busy_pct"):
            checker.check(
                isinstance((rec.get("host_load_start") or {}).get(key), (str, int, float)),
                f"{label}: host_load_start.{key} stamped",
            )
            checker.check(
                isinstance((rec.get("host_load_end") or {}).get(key), (str, int, float)),
                f"{label}: host_load_end.{key} stamped",
            )
        route = (rec.get("route") or {})
        checker.check(route.get("should_use_dense") is True, f"{label}: measured on the dense route")
        arms = rec.get("arms") or {}
        for arm in FLOOR_ARMS:
            stats = (arms.get(arm) or {}).get("stats") or {}
            checker.check(stats.get("calls") == FLOOR_CALLS, f"{label}: {arm} measured over {FLOOR_CALLS} calls")
            checker.check(stats.get("warm_calls") == FLOOR_CALLS - 1, f"{label}: {arm} has {FLOOR_CALLS - 1} warm calls")
            med = stats.get("median_warm_ms")
            checker.check(positive(med), f"{label}: {arm} median_warm_ms positive")
            checker.check(
                is_number(stats.get("min_warm_ms")) and is_number(stats.get("max_warm_ms"))
                and stats["min_warm_ms"] <= stats["median_warm_ms"] <= stats["max_warm_ms"],
                f"{label}: {arm} min <= median <= max",
            )
        if not all(arm in arms for arm in FLOOR_ARMS):
            continue
        med = {arm: float(arms[arm]["stats"]["median_warm_ms"]) for arm in FLOOR_ARMS}
        # only the safe inequalities (M1/M2 ordering and M3-vs-M4 are NOT invariants)
        checker.check(med["M0"] >= med["M3"], f"{label}: M0 >= M3 (host glue removed is cheaper)")
        checker.check(med["M0"] >= med["M4"], f"{label}: M0 >= M4 (glue + readback removed)")
        checker.check(med["M0"] >= med["M5"], f"{label}: M0 >= M5 (pure-only floor is below the call)")
        checker.check(med["M5"] < med["M0"], f"{label}: M5 < M0 (the floor is strictly cheaper)")
        checker.check(med["M3"] < med["M2"], f"{label}: M3 < M2 (the payload upload step is the dominant removable cost)")

        stored_floors = rec.get("floors") or {}
        checker.check(rel_close(stored_floors.get("M0_full_evaluate_ms"), med["M0"], 1e-9), f"{label}: floors.M0 == arm M0")
        checker.check(rel_close(stored_floors.get("M3_host_glue_removed_ms"), med["M3"], 1e-9), f"{label}: floors.M3 == arm M3")
        checker.check(
            rel_close(stored_floors.get("M4_host_glue_and_readback_removed_ms"), med["M4"], 1e-9),
            f"{label}: floors.M4 == arm M4",
        )
        checker.check(rel_close(stored_floors.get("M5_pure_only_frozen_runner_ms"), med["M5"], 1e-9), f"{label}: floors.M5 == arm M5")
        checker.check(
            rel_close(stored_floors.get("M0_minus_M5_ms"), med["M0"] - med["M5"], 1e-6),
            f"{label}: floors.M0_minus_M5 == M0 - M5",
        )

        deltas = rec.get("deltas") or []
        checker.check(len(deltas) == 6, f"{label}: six removal deltas recorded")
        # delta steps name the readback-free / value-variant arms too (M4V), so resolve them against
        # every measured arm, not just the six rungs of the headline matrix.
        measured = {
            arm: float((entry.get("stats") or {}).get("median_warm_ms"))
            for arm, entry in arms.items()
            if positive(((entry.get("stats") or {}).get("median_warm_ms")))
        }
        for delta in deltas:
            step = str(delta.get("step"))
            parts = step.split("-")
            if len(parts) != 2:
                checker.check(False, f"{label}: delta step {step!r} is 'high-low'")
                continue
            high, low = parts
            if high not in measured or low not in measured:
                checker.check(False, f"{label}: delta step {step} names two measured arms")
                continue
            checker.check(
                rel_close(delta.get("median_high_ms"), measured[high], 1e-9),
                f"{label}: delta {step} median_high == arm {high}",
            )
            checker.check(
                rel_close(delta.get("median_low_ms"), measured[low], 1e-9),
                f"{label}: delta {step} median_low == arm {low}",
            )
            checker.check(
                near(delta.get("delta_ms"), measured[high] - measured[low], 0.05),
                f"{label}: delta {step} recomputes from the stored arm medians",
            )

        # correctness: every equivalent arm reproduces the same fitness digest
        digests = {
            arm: (arms[arm].get("fitness_sha256") or arms[arm].get("verify_fitness_sha256"))
            for arm in arms
        }
        for arm in ("M1", "M2", "M3", "M4", "M0R", "M3R", "M0P"):
            if arms.get(arm, {}).get("fitness_sha256"):
                checker.check(
                    arms[arm]["fitness_sha256"] == arms["M0"].get("fitness_sha256"),
                    f"{label}: {arm} reproduces the M0 fitness digest",
                )
        if arms["M4"].get("verify_fitness_sha256"):
            checker.check(
                arms["M4"]["verify_fitness_sha256"] == arms["M0"].get("fitness_sha256"),
                f"{label}: M4's verification digest matches M0",
            )
        for arm in ("M4V", "M5"):
            checker.check(arms[arm].get("fitness_sha256") is None, f"{label}: {arm} is readback-free (no digest)")
        for arm in ("M0", "M1", "M2", "M3", "M4"):
            checker.check((arms[arm].get("cache") or {}).get("misses") == 1, f"{label}: {arm} compiled once (no churn)")
        checker.check((arms["M5"].get("cache") or {}).get("misses") == 2, f"{label}: M5 added only the value-variant compile")
        m5 = route.get("m5_identity") or {}
        checker.check(m5.get("executable_is_m4v_artifact") is True, f"{label}: M5 reuses the M4V artifact")
        checker.check(m5.get("n_graph_calls_per_run") == 1, f"{label}: M5 is one graph call per run")
        drift = rec.get("drift") or {}
        for key in ("M0_repeat_delta_pct", "M3_repeat_delta_pct", "M0_plain_overhead_pct"):
            checker.check(is_number(drift.get(key)), f"{label}: drift.{key} recorded")
        micro = rec.get("upload_micro") or {}
        checker.check(isinstance(micro.get("payloads"), list) and micro["payloads"], f"{label}: upload micro-probe payloads")
        checker.check(positive(micro.get("batch_median_ms")), f"{label}: upload micro-probe batch median")

        # cross-record: the Q1 decomposition's fitness digest is the floor matrix's M0 digest
        if fx in fitness_shas:
            checker.check(
                fitness_shas[fx] == arms["M0"].get("fitness_sha256"),
                f"{label}: M0 fitness digest == the Q1 decomposition digest ({fx})",
            )

    # the contended first attempt stays as the control, not the record of record
    control = read_json(floors_dir / "floor_f64.json", checker, "q6_floors/floor_f64.json")
    if control:
        checker.check(
            "M0" in (control.get("arms") or {}) and positive(((control["arms"]["M0"].get("stats") or {}).get("median_warm_ms"))),
            "q6_floors/floor_f64.json: the contended control still carries the M0 arm",
        )


def check_q3_ab(results, checker, notes):
    """Q3: engine seam vs frozen runner on one artifact -- the no-engine-tax claim."""
    ab_dir = results / "q3_ab"
    exists(ab_dir / "SUMMARY.md", checker, "q3_ab/SUMMARY.md")
    for fx in ("f64", "f200"):
        label = f"q3_ab/ab_{fx}.json"
        rec = read_json(ab_dir / f"ab_{fx}.json", checker, label)
        exists(ab_dir / f"ab_{fx}.log", checker, f"q3_ab/ab_{fx}.log")
        if not rec:
            continue
        check_identity_stamp(rec, checker, label)
        route = rec.get("route") or {}
        checker.check(route.get("should_use_dense") is True, f"{label}: measured on the dense route")
        arms = rec.get("arms") or {}
        for arm in AB_ARMS:
            stats = (arms.get(arm) or {}).get("stats") or {}
            checker.check(stats.get("calls") == 25, f"{label}: {arm} measured over 25 calls")
            checker.check(stats.get("warm_calls") == 24, f"{label}: {arm} has 24 warm calls")
            checker.check(positive(stats.get("median_warm_ms")), f"{label}: {arm} median_warm_ms positive")
        for arm in AB_EXTRA_ARMS:
            checker.check(arm in arms, f"{label}: control arm {arm} present")
            stats = (arms.get(arm) or {}).get("stats") or {}
            checker.check(positive(stats.get("median_warm_ms")), f"{label}: {arm} median_warm_ms positive")
            checker.check(is_number(stats.get("calls")) and stats["calls"] > 1, f"{label}: {arm} measured calls recorded")
        if not all(arm in arms for arm in AB_ARMS):
            continue
        totals = {arm: float(arms[arm]["stats"]["median_warm_ms"]) for arm in AB_ARMS}
        # the control arms (A3_plain_control / R1 / R2) are resolved from the same dict
        all_totals = {
            arm: float((entry.get("stats") or {}).get("median_warm_ms"))
            for arm, entry in arms.items()
            if positive(((entry.get("stats") or {}).get("median_warm_ms")))
        }
        spans = {arm: float(arms[arm].get("etl_run_span_ms")) for arm in AB_ARMS if is_number(arms[arm].get("etl_run_span_ms"))}

        # THE claim: on the same staged payloads and artifact, the engine seam and the frozen run()
        # are indistinguishable (no run-path tax).
        checker.check(
            abs(totals["A2_engine_staged"] - totals["A3_frozen_runner"]) <= AB_TAX_LIMIT_MS,
            f"{label}: |A2 total - A3 total| <= {AB_TAX_LIMIT_MS} ms (no engine run-path tax)",
        )
        if "A2_engine_staged" in spans and "A3_frozen_runner" in spans:
            checker.check(
                abs(spans["A2_engine_staged"] - spans["A3_frozen_runner"]) <= AB_TAX_LIMIT_MS,
                f"{label}: |A2 etl.run span - A3 etl.run span| <= {AB_TAX_LIMIT_MS} ms",
            )
        # freshness: the same artifact fed with fresh payloads costs milliseconds more
        fresh = totals["L1_engine_fresh_loss"] - totals["L2_engine_staged_loss"]
        checker.check(
            fresh > 1.0,
            f"{label}: L1 - L2 = {fresh:+.2f} ms > 1 ms (fresh payloads are the dominant host cost)",
        )
        frozen_fresh = totals["A4_frozen_fresh"] - totals["A3_frozen_runner"]
        checker.check(
            frozen_fresh > 1.0,
            f"{label}: A4 - A3 = {frozen_fresh:+.2f} ms > 1 ms (the freshness cost is payload-, not engine-side)",
        )
        checker.check(
            totals["E1_e2e_in_process"] > totals["A3_frozen_runner"],
            f"{label}: E1 (in-process evaluate) > A3 (pure run)",
        )
        plain = (arms.get("A3_plain_control") or {}).get("stats") or {}
        if positive(plain.get("median_warm_ms")):
            overhead = totals["A3_frozen_runner"] - float(plain["median_warm_ms"])
            checker.check(
                abs(overhead) <= 0.75,
                f"{label}: probe instrumentation overhead on the frozen path is <= 0.75 ms (got {overhead:+.2f})",
            )
        if "R1_repl_fresh_loss" in all_totals:
            replication = abs(all_totals["R1_repl_fresh_loss"] - totals["L1_engine_fresh_loss"])
            checker.check(replication <= 0.50, f"{label}: R1 replicates L1 within 0.5 ms (got {replication:.2f})")
        else:
            checker.check(False, f"{label}: the R1 replication control has a median")
        if "R2_repl_fresh_drained" in all_totals:
            fence = abs(all_totals["R2_repl_fresh_drained"] - all_totals.get("R1_repl_fresh_loss", 0.0))
            checker.check(
                fence <= 0.75,
                f"{label}: an explicit staging fence changes nothing on the warm median (|R2 - R1| = {fence:.2f} ms)",
            )
        else:
            checker.check(False, f"{label}: the R2 drained replication control has a median")

        deltas = rec.get("deltas") or []
        checker.check(len(deltas) >= 12, f"{label}: the delta table is recorded ({len(deltas)} entries)")
        for delta in deltas:
            high, low = delta.get("high"), delta.get("low")
            if high not in arms or low not in arms:
                checker.check(False, f"{label}: delta {delta.get('name')!r} names two known arms")
                continue
            metric = delta.get("metric")
            if metric == "median_warm_ms":
                hi = float(arms[high]["stats"]["median_warm_ms"])
                lo = float(arms[low]["stats"]["median_warm_ms"])
            elif metric == "etl_run":
                hi, lo = arms[high].get("etl_run_span_ms"), arms[low].get("etl_run_span_ms")
                if not (is_number(hi) and is_number(lo)):
                    checker.check(False, f"{label}: delta {delta.get('name')!r} etl_run spans present")
                    continue
                hi, lo = float(hi), float(lo)
            else:
                checker.check(False, f"{label}: delta {delta.get('name')!r} uses a known metric")
                continue
            checker.check(
                near(delta.get("median_high"), hi, 1e-6) and near(delta.get("median_low"), lo, 1e-6),
                f"{label}: delta {delta.get('name')!r} stored operands match the arm records",
            )
            checker.check(
                near(delta.get("delta_ms"), hi - lo, 0.05),
                f"{label}: delta {delta.get('name')!r} recomputes from the stored arms",
            )


def check_docs(results, checker, notes):
    """Soft: the round's write-up deliverables (never affect the exit code)."""
    readme = R5_DIR / "README.md"
    if readme.is_file():
        text = readme.read_text()
        notes.append(f"README.md present ({len(text.splitlines())} lines) -- protocol of the round")
    else:
        notes.append("README.md missing -- the r5 protocol document is not written yet")

    rca = R5_DIR / "RCA.md"
    if not rca.is_file():
        notes.append("RCA.md missing -- the round's analysis of record is still in flight (soft check)")
        return
    text = rca.read_text()
    notes.append(f"RCA.md present ({len(text.splitlines())} lines)")
    lowered = text.lower()
    missing = [header for header in RCA_HEADERS if header.lower() not in lowered]
    if missing:
        notes.append("RCA.md section headers not found yet: " + ", ".join(missing))
    else:
        notes.append("RCA.md carries every expected section header")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
SECTIONS = (
    ("environment", check_environment, False),
    ("agreement", check_backend_agreement, False),
    ("smoke", check_smoke, False),
    ("q4_window", check_q4_window, True),
    ("q1_decomp", check_q1_decomp, True),
    ("q8_minimal", check_q8_minimal, True),
    ("q2_gpu", check_q2_gpu, True),
    ("q6_floors", check_q6_floors, True),
    ("q3_ab", check_q3_ab, True),
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the R5 RCA evidence records (stdlib only, no torch/GPU).",
        epilog="Exit code 0 when no hard check failed; RCA.md/README.md checks are soft notes.",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help=f"the results tree to validate (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=[name for name, _, _ in SECTIONS] + ["docs"],
        help="run only these sections (repeatable; default: all)",
    )
    args = parser.parse_args(argv)

    results = Path(args.results).resolve()
    selected = set(args.section) if args.section else {name for name, _, _ in SECTIONS} | {"docs"}

    checker = Checker()
    print(f"check_r5.py: results tree = {results}")
    if not results.is_dir():
        print(f"check_r5.py: no such results directory: {results}", file=sys.stderr)
        return 1

    q4_state = {}
    fitness_shas = {}
    for name, function, _needs_state in SECTIONS:
        if name not in selected:
            continue
        try:
            if name == "q4_window":
                function(results, checker, checker.notes)
                # the Q8 band reference is the same-window *dense-iree e2e* cell, not the legacy one
                leg = read_json(results / "q4_window" / "q4_leg64.json", checker, "q4_window/q4_leg64.json")
                if leg:
                    q4_state["q4_leg64"] = leg.get("median_warm_ms")
                cell64 = read_json(results / "q4_window" / "q4_eval64_iree.json", checker, "q4_window/q4_eval64_iree.json")
                if cell64:
                    q4_state["q4_eval64_iree"] = cell64.get("median_warm_ms")
                cell200 = read_json(results / "q4_window" / "q4_eval200_iree.json", checker, "q4_window/q4_eval200_iree.json")
                if cell200:
                    q4_state["q4_eval200_iree"] = cell200.get("median_warm_ms")
            elif name == "q1_decomp":
                fitness_shas = function(results, checker, checker.notes) or {}
            elif name == "q8_minimal":
                function(results, checker, checker.notes, q4_state)
            elif name == "q6_floors":
                function(results, checker, checker.notes, fitness_shas)
            else:
                function(results, checker, checker.notes)
        except Exception as exc:  # a validator crash is a hard failure, never a silent pass
            checker.check(False, f"section {name}: validator raised {type(exc).__name__}: {exc}")
    if "docs" in selected:
        check_docs(results, checker, checker.notes)

    print()
    for note in checker.notes:
        print(f"  NOTE {note}")
    for failure in checker.failures:
        print(f"  FAIL {failure}")
    print()
    print(f"{checker.checks} check(s), {len(checker.failures)} failure(s)")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
