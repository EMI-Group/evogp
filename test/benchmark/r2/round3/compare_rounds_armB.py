#!/usr/bin/env python3
"""compare_rounds_armB.py -- R2 / Arm A / Arm B table generator for the Arm B records.

Arm B re-measured the **nine new-side round-3 configs** with the R2 method unchanged, against a
*patched* ETL checkout shadowed onto ``PYTHONPATH`` (``src:/tmp/evogp_etl_patched_51a7141``).  Its ids
are the Arm A ids with an ``armB_`` prefix, and the workload is the same, so every Arm B record diffs
directly against its Arm A twin and against untouched R2 history.  This script is the *sourcing tool
for ``SUMMARY_armB.md``*: it prints

1. the **per-workload warm medians** -- R2 / Arm A / Arm B W1 / Arm B W2 side by side, with the
   A->B and R2->B deltas (eval: warm per-call median; ``sr``: per-rep warm per-generation median,
   rep0 "churning" / rep1 "fully warm", the ``../REPORT.md`` convention);
2. the **first call / cold-build table** -- compile cost and compile traffic per config, reported
   separately and never mixed into a warm median;
3. the **stage table** -- per-rep stage build counters (the probe's cold/warm miss split) plus the
   warm per-rep ms split of the specialized arms, so the dense-route probe degeneracy is visible
   rather than hidden;
4. the **environment table** -- the ``env`` block of every Arm B config (backend / eval policy /
   cache size / device), which is how the shadow and the route are read off the records;
5. the **``iree_sr_b2`` re-attempt table** and the **robustness cross-checks** -- the values that are
   identical across A / B W1 / B W2 (fitness trajectories, build traffic, fd telemetry).

Metric rules and parsing are **not** duplicated here: everything comes from ``compare_rounds.py``
(``load`` / ``eval_metrics`` / ``train_reps`` / ``num`` / ``delta`` / ``ratio`` / ``render``), which is
imported by path -- so the R2-vs-R3 tables and the R2-vs-Arm-B tables cannot drift apart.  The Arm B
record sources are:

* **W2 (records of record)** -- ``results/armB/new/<id>.json`` on disk (the committed W2 snapshots);
* **W1 (superseded, window-paired context)** -- the same paths at the pre-W2 git tip
  (``git show <W1_COMMIT>:test/benchmark/r2/round3/results/armB/new/<id>.json``);
* **Arm A** -- ``results/{new,legacy}/<id>.json``; **R2** -- ``test/benchmark/{new,legacy}/results/r2``.

Stdlib only (no torch, no numpy, no GPU, no evogp); git is read through ``git show`` and any failure
there degrades the cell to ``MISSING``/``NOGIT`` instead of aborting.  Nothing is asserted here -- for
the pass/fail gates see ``check_r3.py --manifest manifest_armB.txt`` and ``check_agreement_armB.py``.

Usage (paths are resolved from this file, so any cwd works)::

    python3 test/benchmark/r2/round3/compare_rounds_armB.py              # plain aligned text
    python3 test/benchmark/r2/round3/compare_rounds_armB.py --markdown   # markdown for SUMMARY_armB.md
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: the git tip whose ``results/armB/new/*.json`` are the superseded W1 snapshots
W1_COMMIT = "b728466"
W1_REPO_PATH = "test/benchmark/r2/round3/results/armB/new"
ARM_B_DIR = os.path.join(HERE, "results", "armB", "new")


def _import_compare_rounds():
    """Import the Arm A generator by path and return the module (never edited by Arm B)."""
    path = os.path.join(HERE, "compare_rounds.py")
    spec = importlib.util.spec_from_file_location("compare_rounds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cr = _import_compare_rounds()
REPO_ROOT = cr.REPO_ROOT

#: the three Arm B workloads -- (label, side, R2 id, Arm A id, Arm B id).  The legacy side has no Arm
#: B record by design (it is the frozen reference), so its Arm B id is None.  R2's ``auto`` *was* the
#: specialized engine, which is why the ``_spec`` rows read their R2 column off the base id.
WORKLOADS = (
    {
        "title": "workload 1/3 -- fixture64 x n64 (eval64 default fixture): warm per-call median [ms]",
        "kind": "eval",
        "rows": (
            ("legacy (CUDA)", "leg", "leg_c5r2", "leg_c5r2", None),
            ("numpy (default)", "new", "np_c5r2", "np_c5r2", "armB_np_c5r2"),
            ("iree auto (dense)", "new", "iree_fixture64_n64", "iree_fixture64_n64",
             "armB_iree_fixture64_n64"),
            ("iree spec (forced specialized)", "new", "iree_fixture64_n64",
             "iree_fixture64_n64_spec", "armB_iree_fixture64_n64_spec"),
        ),
    },
    {
        "title": "workload 2/3 -- fixture200 x n64: warm per-call median [ms]",
        "kind": "eval",
        "rows": (
            ("legacy (CUDA)", "leg", "leg_fixture200_n64", "leg_fixture200_n64", None),
            ("numpy (default)", "new", "np_fixture200_n64", "np_fixture200_n64",
             "armB_np_fixture200_n64"),
            ("iree auto (dense)", "new", "iree_fixture200_n64", "iree_fixture200_n64",
             "armB_iree_fixture200_n64"),
            ("iree spec (forced specialized)", "new", "iree_fixture200_n64",
             "iree_fixture200_n64_spec", "armB_iree_fixture200_n64_spec"),
        ),
    },
    {
        "title": "workload 3/3 -- sr_a pop200 / 10 gens / tree 64: warm per-generation median per rep [ms]",
        "kind": "train",
        "rows": (
            ("legacy (CUDA)", "leg", "leg_sr_a", "leg_sr_a", None),
            ("numpy (default)", "new", "np_sr_a", "np_sr_a", "armB_np_sr_a"),
            ("iree auto (dense)", "new", "iree_sr_a", "iree_sr_a", "armB_iree_sr_a"),
            ("iree spec (forced specialized)", "new", "iree_sr_a", "iree_sr_a_spec",
             "armB_iree_sr_a_spec"),
        ),
    },
)

#: the nine Arm B matrix ids, in the manifest's group order (iree-spec -> iree -> numpy)
CONFIG_IDS = tuple(
    b_id for workload in WORKLOADS for _, _, _, _, b_id in workload["rows"] if b_id
)

#: the non-matrix b2 re-attempt records (never manifest ids) plus the R2 crash skeleton
B2_IDS = ("armB_iree_sr_b2", "armB_iree_sr_b2_spec", "armB_iree_sr_b2_spec_botched_dense_route")


# ---------------------------------------------------------------------------
# loading (never raises; W1 comes from git).  Every loader is memoized so the
# report reads each record exactly once (a `git show` per W1 id).
# ---------------------------------------------------------------------------
_MEMO = {}


def _memoized(key, loader):
    if key not in _MEMO:
        _MEMO[key] = loader()
    return _MEMO[key]


def load_file(path):
    """(payload, status) with status in ok|skipped|MISSING|MALFORMED -- same contract as cr.load."""
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


def load_arm_b_w2(id_):
    """The committed W2 record of record for an Arm B id."""
    return _memoized(("w2", id_), lambda: load_file(os.path.join(ARM_B_DIR, f"{id_}.json")))


def load_arm_b_w1(id_):
    """The superseded W1 snapshot of the same path, read out of git (never off disk)."""
    def read():
        repo_path = f"{W1_REPO_PATH}/{id_}.json"
        try:
            blob = subprocess.run(("git", "show", f"{W1_COMMIT}:{repo_path}"), cwd=REPO_ROOT,
                                  capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None, "NOGIT"
        try:
            payload = json.loads(blob)
        except ValueError:
            return None, "MALFORMED"
        return (payload, "skipped") if payload.get("skipped") else (payload, "ok")
    return _memoized(("w1", id_), read)


def load_arm_a(side, id_):
    return _memoized(("a", side, id_), lambda: cr.load("r3", side, id_))


def load_r2(side, id_):
    return _memoized(("r2", side, id_), lambda: cr.load("r2", side, id_))


# ---------------------------------------------------------------------------
# metric extraction (delegates every rule to compare_rounds.py)
# ---------------------------------------------------------------------------
def warm_value(payload, workload_kind, rep=1):
    """The workload's warm figure: per-call median (eval) or per-rep per-generation median (train)."""
    if payload is None:
        return None
    if workload_kind == "eval":
        return cr.eval_metrics(payload)[1]
    entries = [entry for entry in cr.train_reps(payload) if entry["rep"] == rep]
    return entries[0]["warm"] if entries else None


def first_value(payload):
    """``first_call_ms`` (eval) or rep0's cold first generation (train) -- never a warm metric."""
    if payload is None:
        return None
    if cr.task_kind(payload) == "train":
        entries = cr.train_reps(payload)
        return entries[0]["cold"] if entries else None
    return cr.eval_metrics(payload)[0]


def reps_of(payload):
    """The reps to report: both for a training record, the single rep otherwise."""
    return (0, 1) if payload is not None and cr.task_kind(payload) == "train" else (0,)


def device_of(payload):
    """The device the record ran on, as recorded by the record itself ('' when absent)."""
    if payload is None:
        return "-"
    gpu = payload.get("gpu") or {}
    device = gpu.get("cuda_visible_devices")
    if device in (None, ""):
        device = (payload.get("env") or {}).get("CUDA_VISIBLE_DEVICES", "")
    return str(device) if device not in (None, "") else "-"


def stage_totals(payload, rep=0):
    """The probe's aggregate counters for one rep (empty dict when absent/degenerate)."""
    if payload is None:
        return {}
    stages = payload.get("stages") or {}
    for entry in stages.get("reps") or []:
        if entry.get("rep") == rep:
            return entry.get("totals") or {}
    return {}


def json_misses(payload):
    """The JSON-level cumulative compile-miss counter the --max-cold-builds guard samples."""
    if payload is None:
        return None
    return (payload.get("cold_builds") or {}).get("misses_max")


def build_traffic(payload, rep=0):
    """(stage cold misses, stage warm misses, stage build calls, json cumulative misses_max)."""
    totals = stage_totals(payload, rep)
    return (totals.get("cold_builds"), totals.get("warm_misses"), totals.get("build_calls"),
            json_misses(payload))


def traffic_cell(payload, rep=0):
    """``cold+warm (json N)`` -- the stage-sourced split with the JSON-level guard counter beside it."""
    if payload is None:
        return "-"
    cold, warm, calls, json_misses = build_traffic(payload, rep)
    if cold is None and warm is None and calls is None:
        return cr.cell(payload, "ok", lambda _p: "-")
    json_text = "-" if not cr.is_number(json_misses) else str(json_misses)
    return f"{cr.num(cold, 0)}+{cr.num(warm, 0)} misses / {cr.num(calls, 0)} calls ({json_text})"


def is_dense_block(payload, rep=0):
    """True when the stage block carries no specialized work (the documented dense-route gap)."""
    if payload is None:
        return None
    totals = stage_totals(payload, rep)
    if not totals:
        return None
    return not any(cr.is_number(totals.get(key)) and totals[key] for key in
                   ("build_calls", "build_misses", "build_hits", "decode_calls", "sr_fitness_calls"))


def route_of(payload, rep=0):
    """The evaluation route read off the record (env + stage fingerprint), never assumed."""
    if payload is None:
        return "-"
    env = payload.get("env") or {}
    backend = env.get("EVOGP_ETL_BACKEND", "numpy")
    mode = env.get("EVOGP_EVAL_MODE", "auto")
    if is_dense_block(payload, rep):
        return "dense (probe-degenerate)"
    if backend == "iree" and mode != "specialized":
        return "dense eval / specialized fallback"
    return "specialized"


def fds_cell(payload):
    if payload is None:
        return "-"
    fds = payload.get("fds") or {}
    parts = [fds.get("start"), fds.get("after_cold"), fds.get("end")]
    return " -> ".join("null" if part is None else str(part) for part in parts)


def trajectory(payload, rep):
    """The per-generation best-fitness trajectory of one rep ([] when absent)."""
    if payload is None:
        return []
    for entry in payload.get("per_rep") or []:
        if entry.get("rep") == rep:
            return entry.get("per_gen_best_fitness") or []
    return (payload.get("per_gen_best_fitness") or []) if rep == 0 else []


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def warm_tables(markdown, missing):
    headers = ("arm", "R2", "A (R3)", "A dev", "B W1 (GPU 6)", "B W2 (GPU 0)", "A->B", "R2->B")
    for workload in WORKLOADS:
        rows = []
        for label, side, r2_id, a_id, b_id in workload["rows"]:
            r2, r2_status = load_r2(side, r2_id)
            arm_a, a_status = load_arm_a(side, a_id)
            arm_b_w1, w1_status = load_arm_b_w1(b_id) if b_id else (None, "-")
            arm_b_w2, w2_status = load_arm_b_w2(b_id) if b_id else (None, "-")
            if r2 is None:
                missing.append(f"{r2_id}: R2 {r2_status}")
            if arm_a is None:
                missing.append(f"{a_id}: Arm A {a_status}")
            if b_id and arm_b_w2 is None:
                missing.append(f"{b_id}: Arm B W2 {w2_status}")
            if b_id and arm_b_w1 is None:
                missing.append(f"{b_id}: Arm B W1 {w1_status}")
            if workload["kind"] == "eval":
                values = [warm_value(payload, "eval") for payload in (r2, arm_a, arm_b_w1, arm_b_w2)]
                rows.append((label, cr.num(values[0]), cr.num(values[1]), device_of(arm_a),
                             cr.num(values[2]), cr.num(values[3]),
                             cr.delta(values[3], values[1]), cr.delta(values[3], values[0])))
            else:
                for rep in (0, 1):
                    values = [warm_value(payload, "train", rep) for payload in
                              (r2, arm_a, arm_b_w1, arm_b_w2)]
                    rows.append((f"{label} rep{rep} {cr.TRAIN_REP_LABELS[rep]}",
                                 cr.num(values[0]), cr.num(values[1]), device_of(arm_a),
                                 cr.num(values[2]), cr.num(values[3]),
                                 cr.delta(values[3], values[1]), cr.delta(values[3], values[0])))
        cr.render(workload["title"], headers, rows, markdown)
    print()
    print("  R2 = frozen R2 records; A (R3) = the committed Arm A record (its own window, `A dev`); "
          "B W1/B W2 = the Arm B record in the")
    print("  W1 and W2 windows. The `_spec` rows read their R2 column off the base id: in R2 the "
          "`auto` route WAS the specialized")
    print("  engine, so that cell is the R2 number for the same code path. `-` = not measured "
          "(the legacy side has no Arm B record).")


def compile_table(markdown, missing):
    headers = ("config (armB id)", "kind", "first_ms A", "first_ms B W1", "first_ms B W2",
               "json cold misses A / W1 / W2", "fds A", "fds B W2")
    rows = []
    for id_ in CONFIG_IDS:
        arm_a, _ = load_arm_a("new", id_[len("armB_"):])
        arm_b_w1, w1_status = load_arm_b_w1(id_)
        arm_b_w2, w2_status = load_arm_b_w2(id_)
        if arm_b_w2 is None:
            missing.append(f"{id_}: Arm B W2 {w2_status}")
        kind = cr.task_kind(arm_b_w2) if arm_b_w2 is not None else "-"
        misses = " / ".join(
            "-" if not cr.is_number(json_misses(payload)) else cr.num(json_misses(payload), 0)
            for payload in (arm_a, arm_b_w1, arm_b_w2))
        rows.append((id_, kind, cr.num(first_value(arm_a)), cr.num(first_value(arm_b_w1)),
                     cr.num(first_value(arm_b_w2)), misses, fds_cell(arm_a), fds_cell(arm_b_w2)))
    cr.render("compile / cold first step and compile traffic -- reported separately from warm "
              "medians", headers, rows, markdown)
    print()
    print("  first_ms = `first_call_ms` (eval64) or rep0's `cold_first_gen_ms` (sr); the miss "
          "counter is the JSON `cold_builds.misses_max`")
    print("  (the same numbers the --max-cold-builds guard enforces; '-' = no budget on the numpy "
          "configs). fds = start -> after cold -> end.")


def stage_table(markdown, missing):
    headers = ("config (rep)", "route", "A build calls / hits / misses",
               "B W1 build calls / hits / misses", "B W2 build calls / hits / misses",
               "B W2 stage cold / warm misses (ms)", "B W2 probe span in misc [ms]")
    rows = []
    for id_ in CONFIG_IDS:
        arm_a, _ = load_arm_a("new", id_[len("armB_"):])
        arm_b_w1, _ = load_arm_b_w1(id_)
        arm_b_w2, w2_status = load_arm_b_w2(id_)
        if arm_b_w2 is None:
            missing.append(f"{id_}: Arm B W2 {w2_status}")
        for rep in reps_of(arm_b_w2):
            route = route_of(arm_b_w2, rep)
            cells = []
            for payload in (arm_a, arm_b_w1, arm_b_w2):
                totals = stage_totals(payload, rep)
                calls = totals.get("build_calls")
                cells.append("-" if calls is None else
                             f"{cr.num(calls, 0)} / {cr.num(totals.get('build_hits'), 0)} / "
                             f"{cr.num(totals.get('build_misses'), 0)}")
            totals = stage_totals(arm_b_w2, rep)
            rows.append((f"{id_} (rep{rep})", route, cells[0], cells[1], cells[2],
                         f"{cr.num(totals.get('cold_builds'), 0)} ({cr.num(totals.get('cold_build_ms'), 0)}) "
                         f"/ {cr.num(totals.get('warm_misses'), 0)} ({cr.num(totals.get('warm_build_miss_ms'), 0)})",
                         cr.num(totals.get("misc_ms"))))
    cr.render("stage probe -- build traffic per rep (identical across A / W1 / W2) and the route "
              "fingerprint", headers, rows, markdown)
    print()
    print("  The probe wraps the SPECIALIZED engine's call sites only: a dense-route eval call "
          "reports 0 build/decode/")
    print("  sr_fitness calls, only `run_calls` non-zero and the whole probe span in the `misc` "
          "bucket -- documented in")
    print("  `./README.md`, not a gate failure. Per-structure compile ms is not reproducible on this "
          "box (spread of a factor of a few).")

    headers = ("config (rep)", "metric", "A [ms]", "B W1 [ms]", "B W2 [ms]")
    rows = []
    for id_ in CONFIG_IDS:
        arm_a, _ = load_arm_a("new", id_[len("armB_"):])
        arm_b_w1, _ = load_arm_b_w1(id_)
        arm_b_w2, _ = load_arm_b_w2(id_)
        for rep in reps_of(arm_b_w2):
            for key in ("build_hit_ms", "build_miss_ms", "run_ms", "readback_ms", "decode_ms"):
                rows.append((f"{id_} (rep{rep})", key,
                             cr.num(stage_totals(arm_a, rep).get(key)),
                             cr.num(stage_totals(arm_b_w1, rep).get(key)),
                             cr.num(stage_totals(arm_b_w2, rep).get(key))))
    cr.render("stage probe -- the specialized arms' per-stage ms split (window noise, not a lever "
              "measurement)", headers, rows, markdown)

    headers = ("config (fully warm rep)", "metric", "A [ms]", "B W1 [ms]", "B W2 [ms]")
    rows = []
    for id_ in ("np_sr_a", "iree_sr_a", "iree_sr_a_spec"):
        arm_a, _ = load_arm_a("new", id_)
        arm_b_w1, _ = load_arm_b_w1(f"armB_{id_}")
        arm_b_w2, _ = load_arm_b_w2(f"armB_{id_}")
        for key in ("build_hit_ms", "run_ms", "readback_ms", "total_ms"):
            rows.append((f"{id_} (rep1)", key, cr.num(stage_totals(arm_a, 1).get(key)),
                         cr.num(stage_totals(arm_b_w1, 1).get(key)),
                         cr.num(stage_totals(arm_b_w2, 1).get(key))))
    cr.render("stage probe -- the fully warm rep of the training ids (every build call hits the "
              "cache): the path a patched lookup sits on", headers, rows, markdown)


def env_table(markdown, missing):
    keys = ("EVOGP_ETL_BACKEND", "EVOGP_EVAL_MODE", "EVOGP_ETL_CACHE_SIZE",
            "EVOGP_ETL_IREE_DEVICE", "EVOGP_ETL_IREE_OPT_LEVEL", "ETL_DISABLE_LOCATIONS",
            "CUDA_VISIBLE_DEVICES")
    headers = ("config", "window", *keys, "route taken (stage fingerprint)")
    rows = []
    for id_ in CONFIG_IDS:
        for window, (payload, status) in (("W1", load_arm_b_w1(id_)),
                                          ("W2", load_arm_b_w2(id_))):
            if payload is None:
                missing.append(f"{id_}: Arm B {window} {status}")
                continue
            env = payload.get("env") or {}
            rows.append((id_, window, *(env.get(key, "(unset)") for key in keys),
                         route_of(payload)))
    cr.render("environment of every Arm B record -- the shadow lives in PYTHONPATH, not in the JSON",
              headers, rows, markdown)
    print()
    print("  PYTHONPATH is not recorded by the harness (and `etl_version` is \"0.1.0\" for BOTH "
          "checkouts), so the shadow is")
    print("  evidenced by `results/armB/etl_import_probe.txt` plus the `etl:` header line of every "
          "`results/armB/new/<id>.log`.")


def b2_table(markdown, missing):
    records = (("R2 twin `../new/results/r2/iree_sr_b2` (stock ETL, pre-dense era)", None,
                load_file(os.path.join(REPO_ROOT, "test", "benchmark", "new", "results", "r2",
                                       "iree_sr_b2.json"))),
               ("B attempt 1 -- `armB_iree_sr_b2` (auto => dense)", "dense", load_arm_b_w2(B2_IDS[0])),
               ("B attempt 2 (superseded) -- `..._spec_botched_dense_route` (auto => dense)",
                "dense", load_arm_b_w2(B2_IDS[2])),
               ("B attempt 2 -- `armB_iree_sr_b2_spec` (pinned specialized)", "specialized",
                load_arm_b_w2(B2_IDS[1])))
    headers = ("record", "route taken", "complete", "harness_wall_s",
               "cold+warm traffic (json misses)", "fds start->cold->end",
               "per-rep best-fitness trajectory")
    rows = []
    for label, route, (payload, status) in records:
        if payload is None:
            missing.append(f"b2: {label} {status}")
            continue
        complete = payload.get("complete")
        rows.append((label, route or "specialized (only route then)",
                     "-" if complete is None else ("complete" if complete else "skeleton (crash)"),
                     cr.num(payload.get("harness_wall_s"), 0), traffic_cell(payload),
                     fds_cell(payload),
                     str(trajectory(payload, 0) or "-")))
    cr.render("`iree_sr_b2` re-attempt (pop 1000 / gens 2 / tree 128) -- non-manifest ids",
              headers, rows, markdown)


def robust_checks():
    """The claims that must hold across A / W1 / W2 regardless of window or device."""
    print()
    print("== robustness cross-checks (computed here, not quoted) ==")
    for id_, label in (("iree_sr_a", "iree auto (dense)"), ("iree_sr_a_spec", "iree spec"),
                       ("np_sr_a", "numpy")):
        arm_a, _ = load_arm_a("new", id_)
        arm_b_w1, _ = load_arm_b_w1(f"armB_{id_}")
        arm_b_w2, _ = load_arm_b_w2(f"armB_{id_}")
        for rep in (0, 1):
            trails = [trajectory(payload, rep) for payload in (arm_a, arm_b_w1, arm_b_w2)]
            same = all(trail == trails[0] for trail in trails[1:]) and bool(trails[0])
            print(f"  {label:<20} rep{rep}: per_gen_best_fitness A vs W1 vs W2 = "
                  f"{'IDENTICAL' if same else 'DIFFER'} "
                  f"({len(trails[0])} gens, final {trails[0][-1] if trails[0] else '-'})")
        traffic = [build_traffic(payload, 0)[:3] for payload in (arm_a, arm_b_w1, arm_b_w2)]
        fds = [fds_cell(payload) for payload in (arm_a, arm_b_w1, arm_b_w2)]
        print(f"  {label:<20}       rep0 stage build traffic A / W1 / W2 identical = "
              f"{'YES' if traffic[0] == traffic[1] == traffic[2] else 'NO'} {traffic[0]}; "
              f"fds A {fds[0]} | B {fds[2]}")
    print("  (the legacy side is frozen: `leg_*` has an Arm A record and no Arm B record by design)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Print R2 / Arm A / Arm B comparison tables for the Arm B benchmark records "
                    "(stdlib only, no torch/numpy/GPU; Arm B W1 is read through `git show`).",
        epilog="Missing records print as MISSING; the script never fails on them.")
    parser.add_argument("--markdown", action="store_true",
                        help="emit markdown tables (for SUMMARY_armB.md) instead of aligned text")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    missing = []
    print("compare_rounds_armB.py -- R2 vs Arm A vs Arm B (R2 and Arm A records are read-only; "
          "Arm B lives under round3/results/armB/)")
    print(f"  repo root     : {REPO_ROOT}")
    print(f"  Arm B W2      : {os.path.relpath(ARM_B_DIR, REPO_ROOT)} (records of record, on disk)")
    print(f"  Arm B W1      : git show {W1_COMMIT}:{W1_REPO_PATH}/<id>.json (superseded snapshots)")
    print(f"  Arm A         : {cr.ROUND_DIRS[('r3', 'new')]} + {cr.ROUND_DIRS[('r3', 'leg')]}")
    print(f"  R2            : {cr.ROUND_DIRS[('r2', 'new')]} + {cr.ROUND_DIRS[('r2', 'leg')]}")
    print("  warm (eval)   = median_warm_ms, else median(per_call_ms[1:])")
    print("  warm (train)  = median(per_gen_ms[1:]) per rep; cold = that rep's cold_first_gen_ms")

    warm_tables(args.markdown, missing)
    compile_table(args.markdown, missing)
    stage_table(args.markdown, missing)
    env_table(args.markdown, missing)
    b2_table(args.markdown, missing)
    robust_checks()

    print()
    print("== notes ==")
    print("  * the warm median is the headline; first call / cold builds are reported separately "
          "(never mixed in).")
    print("  * Arm B W2 (GPU 0) is the record of record; W1 (GPU 6, contended) is window-paired "
          "context -- see the runlog.")
    print("  * the patched ETL changes no fitness vector (agreement gate: 6/6 Arm A "
          "`values_sha256` bit-identical).")
    print("  * `iree_sr_b2*` ids are deliberately NOT manifest ids -- `check_r3.py --manifest "
          "manifest_armB.txt` stays at 9 records.")
    if missing:
        print("  * records not readable:")
        for item in sorted(set(missing)):
            print(f"      {item}")
    else:
        print("  * every R2, Arm A, Arm B W1 and Arm B W2 record of the 9 ids + 3 b2 records was "
              "readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
