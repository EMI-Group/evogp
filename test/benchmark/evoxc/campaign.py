"""Sequential campaign driver: one subprocess per (cell, arm), one config at a time.

The engine keeps its compile cache and executable memo in process memory, so a
fresh process per config is the only way each record pays its own cold compile
and no arm benefits from another arm's warm cache.  This driver therefore *is*
the protocol: it never runs two configs at once, it re-checks that the GPU is
idle (and that no foreign python/torch process is resident) before every timed
config, and it records the guard result in the log line.

Two things it deliberately does NOT do: sum/average anything (the records are the
evidence, ``summarize.py`` only tabulates them) and decide the route (each worker
records what ``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` actually resolved).

Usage (from the repository root, project venv)::

    # the baseline campaign on this checkout
    PY=/tmp/.../evogp-venv/bin/python
    $PY test/benchmark/evoxc/campaign.py --tree-root src \\
        --out test/benchmark/evoxc/results/baseline

    # the same campaign against a git-archive snapshot, apples to apples
    $PY test/benchmark/evoxc/campaign.py --tree-root /tmp/evogp_preopt/src \\
        --out test/benchmark/evoxc/results/replay

    # what the batch route would do on the tree under test, BEFORE launching it
    # (torch-free: resolves the policy from the files, no device, no evogp import)
    python3 test/benchmark/evoxc/campaign.py --out "$TMPDIR/dry" \\
        --cells main --arms batch --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402


# --- guards -----------------------------------------------------------------
#: cmdline markers of a process that could actually touch the device: the project
#: venv, or an evogp / torch / benchmark entry point.  Housekeeping daemons that
#: merely run a system python (``unattended-upgrades``, ``fail2ban``) are noise.
GPU_MARKERS = ("evogp-venv", "/evoxc/", "import torch", "evogp", "bench", "pytest")


def self_pids() -> set:
    """This driver plus its whole ancestor chain (the driver itself imports torch).

    ``common.env_block`` reads ``torch.__version__`` and the device properties, so
    the driver legitimately holds a HIP context and shows up in the KFD process
    list; that is the guard's own footprint, not contamination.
    """
    pids = set()
    pid = os.getpid()
    for _ in range(64):
        if pid <= 1 or pid in pids:
            break
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat", "r") as handle:
                stat = handle.read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:  # noqa: BLE001
            break
    return pids


def foreign_python() -> List[str]:
    """GPU-capable python processes other than this driver and its own shell."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "python"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:  # noqa: BLE001 - no pgrep -> cannot check, say so
        return ["pgrep-unavailable"]
    mine = self_pids()
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        if not pid.isdigit() or int(pid) in mine:
            continue
        if "pgrep -af python" in cmd:
            continue
        if not any(marker in cmd for marker in GPU_MARKERS):
            continue
        rows.append(f"{pid}: {cmd[:120]}")
    return rows


def gpu_compute_pids() -> Dict[str, Any]:
    """Compute processes currently resident on the device (``rocm-smi --showpids``)."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    if "No KFD PIDs" in out or "No processes" in out:
        return {"pids": [], "idle": True}
    mine = self_pids()
    pids, own = [], []
    for line in out.splitlines():
        for token in line.replace("|", " ").split():
            if token.isdigit() and int(token) > 100:
                target = own if int(token) in mine else pids
                if int(token) not in target:
                    target.append(int(token))
    return {
        "pids": pids,
        "own_pids": own,
        "idle": not pids,
        "raw_lines": [ln for ln in out.splitlines() if ln.strip()][:20],
    }


def gpu_use() -> Dict[str, Any]:
    """Parse ``rocm-smi --showuse`` for the card utilisation percentages."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    uses = {}
    for line in out.splitlines():
        if "GPU use" not in line:
            continue
        key, _, rest = line.partition(":")
        value = "".join(ch for ch in rest if ch.isdigit() or ch == ".").split(".")[0]
        try:
            uses[key.strip()] = float(value)
        except ValueError:
            continue
    return {"gpu_use_pct": uses}


def guard(wait_s: float, poll_s: float = 5.0) -> Dict[str, Any]:
    """Wait for an idle device, then report what was actually seen.

    The primary signal is ``rocm-smi --showpids`` (a real compute process on the
    card); the python scan is a hint for a stale benchmark process that has not
    touched the device yet.  Only *this* driver's own worker may be resident, and
    the guard runs before the worker is spawned, so any PID on the card is foreign.
    """
    deadline = time.time() + wait_s
    started = time.time()
    while True:
        state = {
            "python": foreign_python(),
            "gpu": gpu_use(),
            "gpu_pids": gpu_compute_pids(),
            "waited_s": round(time.time() - started, 1),
        }
        # the device process list is the real signal: a foreign process on the
        # card is contamination, while a short-lived python that merely imports
        # torch (a probe, a test run) is only a hint -- blocking on it made every
        # config pay the whole guard window
        busy = bool(state["gpu_pids"].get("pids"))
        if not busy or time.time() >= deadline:
            state["contaminated"] = bool(busy)
            state["ok"] = not busy
            return state
        time.sleep(poll_s)


# --- schedule ---------------------------------------------------------------
#: Coverage policy.  ``specialized``-iree compiles one graph per DISTINCT tree
#: structure, and a measured iree compile is ~272-284 ms per structure on this
#: box (p16: 4.26 s / 15 misses, p64: 13.3 s / 49 misses), so a fresh 256- or
#: 1000-tree population costs 1-5 minutes of compile per evaluation and a GP step
#: re-pays it every generation.  That route is therefore timed on the small
#: populations only (where it is affordable) and the excluded cells are stated in
#: the driver record instead of being silently dropped.
#:
#: The BATCH arms (``evoxc-batch-iree`` / ``-flat``) deliberately get NO rule here:
#: nobody has measured the merged-graph compile yet, so excluding cells would be an
#: unfounded guess (and a wrong exclusion silently hides the route's real cost).
#: The first measurement round decides the scoping FROM the ``first_call_ms`` and
#: ``batch_compile_misses_first_call`` it records; until then every cell is timed.
SCOPE = {"evoxc-specialized-iree": {"pops": (16, 64)}}


def in_scope(arm: str, cell: Dict[str, Any], full: bool = False) -> bool:
    rule = SCOPE.get(arm)
    if rule is None or full:
        return True
    return cell["pop"] in rule["pops"]
def schedule(cells: List[Dict[str, Any]], arms: List[str]) -> List[Dict[str, Any]]:
    """Cells outer, arms inner: the cheap torch arms precede the compile-heavy ones."""
    order = [arm for arm in arms if arm in common.TORCH_ARMS] + [
        arm for arm in arms if arm in common.EVOXC_ARMS
    ]
    return [{"cell": cell, "arm": arm} for cell in cells for arm in order]


def record_path(out_dir: str, cell: str, arm: str) -> str:
    return os.path.join(out_dir, cell, f"{arm}.json")


def log_path(out_dir: str, cell: str, arm: str) -> str:
    return os.path.join(out_dir, cell, f"{arm}.log")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree-root", default="src", help="src root holding evogp (or a snapshot's src/)")
    ap.add_argument("--evoxc-root", default="/root/evoxc", help="foreign repo holding the evoxc package")
    ap.add_argument("--out", required=True, help="result directory (per-cell subdirs of JSON)")
    ap.add_argument("--cells", default="main,n64", help="'main', 'n64', or a comma-separated cell list")
    ap.add_argument("--arms", default="all",
                    help="'all', 'torch', 'evoxc', 'batch' (the batch-route arms only, in "
                         "registry order), or a comma-separated arm list")
    ap.add_argument("--workloads", default="sr_fitness,batch_forward,forward,gp_step")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gp-warmup", type=int, default=1)
    ap.add_argument("--gp-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard-wait", type=float, default=120.0, help="seconds to wait for an idle GPU")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-config wall-clock cap")
    ap.add_argument("--resume", action="store_true", help="skip configs whose record already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule only")
    ap.add_argument("--full-specialized-iree", action="store_true",
                    help="also time specialized-iree on the large populations (1-5 min of compile each)")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    cells: List[Dict[str, Any]] = []
    for token in args.cells.split(","):
        token = token.strip()
        if token == "main":
            cells.extend(common.main_cells())
        elif token == "n64":
            cells.extend(common.n64_cells())
        elif token:
            cells.append(common.parse_cell(token))
    arms: List[str] = []
    for token in args.arms.split(","):
        token = token.strip()
        if token == "all":
            arms = list(common.ALL_ARMS)
        elif token == "torch":
            arms.extend(a for a in common.TORCH_ARMS if a not in arms)
        elif token == "evoxc":
            arms.extend(a for a in common.EVOXC_ARMS if a not in arms)
        elif token == "batch":
            # registry order, driven by the ONE predicate -- a future batch variant
            # (another band policy) is picked up with no edit here
            arms.extend(
                a for a in common.EVOXC_ARMS if common.is_batch_arm(a) and a not in arms
            )
        elif token:
            arms.append(token)

    configs = schedule(cells, arms)
    kept, dropped = [], []
    for cfg in configs:
        (kept if in_scope(cfg["arm"], cfg["cell"], args.full_specialized_iree) else dropped).append(cfg)
    configs = kept
    src = os.path.abspath(os.path.join(args.tree_root, "src")) if os.path.isdir(
        os.path.join(args.tree_root, "src")
    ) else os.path.abspath(args.tree_root)
    pythonpath = os.pathsep.join([src, os.path.abspath(args.evoxc_root), _HERE])
    # inherit the caller's environment (ROCm needs its library paths) and only
    # override what defines the config
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env["EVOXC_DISABLE_LOCATIONS"] = os.environ.get("EVOXC_DISABLE_LOCATIONS", "1")
    for key in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_EVOXC_DEVICE",
                "EVOGP_EVOXC_OPT_LEVEL", "EVOGP_EVOXC_CACHE_SIZE",
                "EVOGP_DENSE_AUTO_MIN_POP", "EVOGP_DENSE_AUTO_MAX_ROWS"):
        if key in os.environ:
            env[key] = os.environ[key]
    # the batch route's knobs travel the same way, so an operator's explicit override
    # reaches the worker and is visible in its record.  This only forwards what the
    # caller exported: inside the child, ``arms.apply_arm_env`` clears every batch
    # knob for every arm and then writes the arm's own ``knobs`` spec, which wins.
    for key in common.BATCH_ENV_VARS:
        if key in os.environ:
            env[key] = os.environ[key]

    os.makedirs(args.out, exist_ok=True)
    driver_path = os.path.join(args.out, "campaign.json")

    if args.dry_run:
        for cfg in configs:
            print("would run", cfg["cell"]["id"], cfg["arm"])
        # ... and for the batch arms, WHERE the route resolves on the tree being
        # measured -- the point of the dry run is to see that before a multi-minute
        # merged-graph compile is paid.  The arm's own knob spec is applied around the
        # resolution (restored afterwards) so the ``-flat`` variant reports bands=flat.
        for arm in [a for a in arms if common.is_batch_arm(a)]:
            arm_knobs = dict((common.EVOXC_ARMS.get(arm) or {}).get("knobs") or {})
            previous = {key: os.environ.get(key) for key in arm_knobs}
            os.environ.update({key: str(value) for key, value in arm_knobs.items()})
            try:
                policy = common.batch_policy(src)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            print(
                "batch policy", arm,
                f"band_size={policy['band_size']}",
                f"bands={policy['bands']}",
                f"alignment={policy['alignment']}",
                f"switch_ops={policy['switch_ops']}",
                f"min_pop={policy['min_pop']}",
                f"declared_modes={policy['declared_modes']!r}",
                f"module_present={policy['module_present']}",
                f"mode_supported={policy['mode_supported']}",
            )
        return 0

    driver_record = {
        "schema": common.SCHEMA + "-campaign",
        "driver_env": common.env_block(os.path.abspath(args.tree_root), arm="driver",
                                       extra={"note": args.note, "pythonpath": pythonpath}),
        "protocol": {
            "one_process_per_config": True,
            "sequential": True,
            "workloads": args.workloads.split(","),
            "warmup": args.warmup,
            "reps": args.reps,
            "gp_warmup": args.gp_warmup,
            "gp_reps": args.gp_reps,
            "seed": args.seed,
            "guard_wait": args.guard_wait,
            "scope_rule": {arm: list(rule["pops"]) for arm, rule in SCOPE.items()},
            "scope_dropped": [{"cell": cfg["cell"]["id"], "arm": cfg["arm"]} for cfg in dropped],
            "specialized_iree_compile_ms_per_structure": {"measured_range": [271.6, 283.9],
                                                          "source": "probe p16/p64, this box"},
            "batch_arms": {
                arm: dict(common.EVOXC_ARMS[arm])
                for arm in common.ALL_ARMS
                if common.is_batch_arm(arm)
            },
            "batch_engine_present": common.batch_engine_present(src),
            "batch_route_note": (
                "the batch route compiles ONE merged graph per population structure "
                "multiset, so its compile is expected to take minutes and a batch-arm "
                "overrun shows up as status 'timeout' with the worker log as evidence, "
                "not as a defect to be retried away"
            ),
        },
        "configs": [],
    }
    common.write_json(driver_path, driver_record)

    t_start = time.time()
    failures = 0
    for index, cfg in enumerate(configs, 1):
        cell_id, arm = cfg["cell"]["id"], cfg["arm"]
        out_path = record_path(args.out, cell_id, arm)
        if args.resume and os.path.isfile(out_path):
            print(f"[{index}/{len(configs)}] skip {cell_id} {arm} (record exists)", flush=True)
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        state = guard(args.guard_wait)
        if state.get("contaminated"):
            print(f"[{index}/{len(configs)}] WARNING busy box for {cell_id} {arm}: {state}", flush=True)
        cmd = [
            sys.executable,
            os.path.join(_HERE, "worker.py"),
            "--cell", cell_id,
            "--arm", arm,
            "--out", out_path,
            "--tree-root", args.tree_root,
            "--workloads", args.workloads,
            "--warmup", str(args.warmup),
            "--reps", str(args.reps),
            "--gp-warmup", str(args.gp_warmup),
            "--gp-reps", str(args.gp_reps),
            "--seed", str(args.seed),
        ]
        if args.note:
            cmd += ["--note", args.note]
        started = time.time()
        status, detail = "ok", ""
        with open(log_path(args.out, cell_id, arm), "w") as log:
            log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
            log.write("env: " + json.dumps({k: v for k, v in env.items() if v}) + "\n\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=args.timeout
                )
                log.write(proc.stdout)
                if proc.stderr:
                    log.write("\n--- stderr ---\n" + proc.stderr)
                if proc.returncode != 0:
                    status = "failed"
                    detail = f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                detail = f">{args.timeout:.0f}s"
        wall = time.time() - started
        if status != "ok":
            failures += 1
        entry = {
            "index": index,
            "cell": cell_id,
            "arm": arm,
            "record": out_path,
            "log": log_path(args.out, cell_id, arm),
            "status": status,
            "detail": detail,
            "wall_s": round(wall, 2),
            "guard": state,
        }
        driver_record["configs"].append(entry)
        driver_record["elapsed_s"] = round(time.time() - t_start, 1)
        common.write_json(driver_path, driver_record)
        print(
            f"[{index}/{len(configs)}] {status:<8} {cell_id:<22} {arm:<24} {wall:6.1f}s "
            f"(gpu_idle={state.get('ok')})",
            flush=True,
        )
    driver_record["elapsed_s"] = round(time.time() - t_start, 1)
    driver_record["failures"] = failures
    common.write_json(driver_path, driver_record)
    print(f"[campaign] {len(configs)} configs, {failures} failures, {driver_record['elapsed_s']}s", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
