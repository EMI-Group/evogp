"""The measured basis of ``EVOGP_FUSED_MIN_POP``, re-taken on the current tree.

``src/evogp/tree/FUSED.md`` §13 calibrated the floor on a pop 1..128 column whose fused
numbers predate the pre-bound dispatch (``EVOGP_FUSED_DISPATCH=bound``) and the plan fast
path, so the quoted "0.0651-0.0684 ms over pop 16-128" is no longer what the shipped route
costs.  This driver re-takes the column at the shipped geometry (``L=64`` / ``i32`` /
``N=1024``) on the current checkout and adds the two readings the floor is a statement
about: ``specialized`` (the engine ``auto`` falls back to) and the classic
``torch-per-tree`` walk, plus the harness's ``torch-batched`` arm as the habitual
denominator.

Protocol (``FUSED.md``'s, tightened):
* ONE process per (pop, arm), strictly sequential, spawned with the same env plus the
  arm's ``EVOGP_EVAL_MODE``; never two configs in one process, never two at once.
* 5 untimed warmups + 20 sync-bracketed reps, warm MEDIAN; the cold first call is
  recorded separately and never enters the median.
* Idle-device guard before every config (``campaign.guard``: ``rocm-smi --showpids``).
* ``eval_mode=fused`` below ``EVOGP_FUSED_MIN_POP`` is the FALLBACK, not the kernel, so
  that arm records the policy's own answer and the kernel is measured policy-free
  through ``evaluate_fused`` / ``sr_fitness_fused`` (the ``fused_direct`` arm) at every
  pop.
* A fused number is only kept as served when the ``fused_*`` counter delta shows one
  kernel launch per call and no decline.

Usage (from the repository root)::

    TRITON_CACHE_DIR=$TMPDIR/triton-cache EVOXC_DISABLE_LOCATIONS=1 \\
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc \\
    /tmp/rocvenv/bin/python test/benchmark/evoxc/minpop_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SCHEMA = "evogp.minpop_probe/1"
#: The shipped geometry of the column (``FUSED.md`` §13).
TREE_LEN, INPUT_LEN, ROWS, DEPTH = 64, 32, 1024, 5
#: Every population the floor is a statement about: below it (1-8), at it (16) and up.
POPS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
ARMS: Tuple[str, ...] = ("fused", "fused_direct", "specialized", "torch_batched",
                         "torch_per_tree")
WARMUP, REPS = 5, 20
DEFAULT_OUT = os.path.join(_HERE, "results", "minpop_r3")


# --------------------------------------------------------------------------- #
# child: one (pop, arm)
# --------------------------------------------------------------------------- #
def _mode_of(arm: str) -> Any:
    """``EVOGP_EVAL_MODE`` this arm pins (``None`` = whatever the tree inherits)."""
    if arm in ("fused", "fused_direct"):
        return "fused"
    if arm == "specialized":
        return "specialized"
    return None


def _route(dense_engine, pop: int, eval_mode: str) -> Dict[str, Any]:
    """The route policy's own answer for this call shape (``eval_mode`` passed in)."""
    route = dense_engine.resolve_route(pop, rows=ROWS, tree_len=TREE_LEN,
                                       output_len=1, eval_mode=eval_mode)
    return {"eval_mode": eval_mode, "engine": route.engine,
            "use_fused": bool(route.use_fused), "use_dense": bool(route.use_dense),
            "backend": route.backend, "fallback_backend": route.fallback_backend,
            "reason": route.reason}


def worker(args: argparse.Namespace) -> int:
    import common
    from evogp.tree import dense_engine, fused_engine, fused_kernels

    pop, arm = int(args.pop), args.arm
    cell = common.cell(pop, TREE_LEN, INPUT_LEN, ROWS, DEPTH)
    workload = common.build_workload(cell)
    forest, x, labels = workload["forest"], workload["x"], workload["labels"]
    record: Dict[str, Any] = {
        "schema": SCHEMA, "kind": "minpop", "created_utc":
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cell": cell["id"], "pop": pop, "arm": arm, "rows": ROWS,
        "tree_len": TREE_LEN, "input_len": INPUT_LEN, "max_layer_cnt": DEPTH,
        "protocol": {"one_process_per_config": True, "sync_bracketed": True,
                     "warmup": WARMUP, "reps": REPS},
        "route_fused_request": _route(dense_engine, pop, "fused"),
        "route_inherited": _route(dense_engine, pop, None),
        "geometry": {"fused_block_rows": int(fused_engine.resolve_block_rows()),
                     "fused_num_warps": int(fused_engine.resolve_num_warps()),
                     "fused_min_pop": int(fused_engine.resolve_min_pop()),
                     "fused_max_scratch_mb": int(fused_engine.resolve_max_scratch_mb()),
                     "n_rb": -(-ROWS // int(fused_engine.resolve_block_rows()))},
        "dispatch": {k: int(v) for k, v in fused_kernels.dispatch_stats().items()},
        "fused_available": bool(fused_engine.fused_available()),
        "triton": {"cache_dir": os.environ.get("TRITON_CACHE_DIR"),
                   "specializations": int(fused_engine.fused_cache_stats()
                                          .get("fused_kernel_compiles", 0))},
    }
    mode = _mode_of(arm)
    if mode is None:
        os.environ.pop("EVOGP_EVAL_MODE", None)
    else:
        os.environ["EVOGP_EVAL_MODE"] = mode

    before = {k: int(v) for k, v in fused_engine.fused_cache_stats().items()}
    if arm in ("torch_batched", "torch_per_tree"):
        from arms_torch import make_torch_arm
        evaluator = make_torch_arm("batched" if arm == "torch_batched" else "per_tree",
                                   forest)
        calls: Dict[str, Any] = {"value": lambda: evaluator.batch_forward(x),
                                 "loss": lambda: evaluator.sr_fitness(x, labels)}
    elif arm == "fused_direct":
        nv, nt, ss = (forest.batch_node_value, forest.batch_node_type,
                      forest.batch_subtree_size)
        calls = {
            "value": lambda: fused_engine.evaluate_fused(nv, nt, ss, INPUT_LEN, 1, x,
                                                        per_row=False),
            "loss": lambda: fused_engine.sr_fitness_fused(nv, nt, ss, INPUT_LEN, 1, x,
                                                         labels, use_MSE=True),
        }
    else:
        calls = {"value": lambda: forest.batch_forward(x),
                 "loss": lambda: forest.SR_fitness(x, labels)}

    timings: Dict[str, Any] = {}
    for name, fn in calls.items():
        timings[name] = common.public_timing(common.time_call(fn, warmup=WARMUP,
                                                             reps=REPS))
    record["timings"] = timings
    record["policy_engine"] = _route(dense_engine, pop, _mode_of(arm))["engine"]
    record["arm_note"] = {
        "fused": "public API with EVOGP_EVAL_MODE=fused: whatever the policy serves "
                 "(below EVOGP_FUSED_MIN_POP that is the specialized FALLBACK, and "
                 "route_fused_request.use_fused is false)",
        "fused_direct": "evaluate_fused / sr_fitness_fused: the kernel without the "
                        "route policy, so it is measurable below the floor",
        "specialized": "public API with EVOGP_EVAL_MODE=specialized (auto's fallback)",
        "torch_batched": "the harness's whole-population plain-torch arm",
        "torch_per_tree": "the classic per-tree plain-torch walk (one op cluster per "
                          "node, one tree at a time)",
    }[arm]
    if arm in ("fused", "fused_direct"):
        after = {k: int(v) for k, v in fused_engine.fused_cache_stats().items()}
        delta = {k: after.get(k, 0) - before.get(k, 0)
                 for k in sorted(set(before) | set(after))}
        launches = int(delta.get("fused_kernel_launches", 0))
        expected = len(calls) * (1 + WARMUP + REPS)
        record["fused_cache_delta"] = {k: v for k, v in delta.items() if v}
        record["launches"] = launches
        record["expected_launches"] = expected
        record["served"] = bool(launches == expected
                                and int(delta.get("fused_declines", 0)) == 0)
    print("RESULT " + json.dumps(record), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _env(arm: str) -> Dict[str, str]:
    """The child environment: the harness presets plus this arm's route pin."""
    env = dict(os.environ)
    env.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(),
                                                   "evogp_minpop_triton"))
    env.setdefault("EVOXC_DISABLE_LOCATIONS", "1")
    parts = [os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src"),
             "/root/evoxc", _HERE]
    env["PYTHONPATH"] = os.pathsep.join(parts + [p for p in
                                                env.get("PYTHONPATH", "").split(os.pathsep)
                                                if p and p not in parts])
    env.pop("EVOGP_EVAL_MODE", None)
    mode = _mode_of(arm)
    if mode is not None:
        env["EVOGP_EVAL_MODE"] = mode
    return env


def _record_path(out: str, pop: int, arm: str) -> str:
    return os.path.join(out, "records", f"p{pop}-{arm}.json")


def _median(record: Dict[str, Any], surface: str) -> Any:
    """One arm's warm median (ms) for a surface, or ``None`` when it was not measured."""
    return (record.get("timings", {}).get(surface) or {}).get("median_ms")


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _ratio(numerator: Any, denominator: Any) -> Any:
    """``denominator / numerator`` (the FUSED.md convention: > 1 means the arm wins)."""
    if numerator in (None, 0) or denominator is None:
        return None
    return float(denominator) / float(numerator)


def render(out: str) -> int:
    """The column as markdown, from the records alone (no device, no torch)."""
    records: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for arm in ARMS:
        for pop in POPS:
            path = _record_path(out, pop, arm)
            if os.path.isfile(path):
                with open(path) as handle:
                    records[(pop, arm)] = json.load(handle)
    pops = sorted({pop for pop, _ in records})
    lines: List[str] = [
        "# `EVOGP_FUSED_MIN_POP` column, re-taken on the current tree",
        "",
        f"Cells `p<pop>-L{TREE_LEN}-i{INPUT_LEN}-N{ROWS}`, warm medians over {WARMUP} warmups + "
        f"{REPS} sync-bracketed reps, one process per (pop, arm), idle-device guard per config.",
        "`fused` = `Forest.batch_forward` / `SR_fitness` with `EVOGP_EVAL_MODE=fused` (the policy's "
        "answer, so below the floor it IS the specialized fallback);",
        "`kernel` = `fused_engine.evaluate_fused` / `sr_fitness_fused` (policy-free, served at any pop); "
        "`spec` = `EVOGP_EVAL_MODE=specialized`; `torch-1` = per-tree torch walk; `torch-N` = the "
        "harness's batched arm.",
        "Ratios are `reference / fused-kernel` (higher = the kernel wins).",
        "",
        "## Value surface (`batch_forward`), ms",
        "",
        "| pop | fused | kernel | spec | torch-1 | torch-N | spec/kernel | torch-1/kernel "
        "| torch-N/kernel | kernel served | fused served |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    losses: List[str] = [
        "",
        "## Loss surface (`SR_fitness`), ms",
        "",
        "| pop | fused | kernel | spec | torch-1 | torch-N | spec/kernel | torch-1/kernel "
        "| torch-N/kernel |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pop in pops:
        row = {arm: records.get((pop, arm), {}) for arm in ARMS}
        kernel_v = _median(row["fused_direct"], "value")
        kernel_l = _median(row["fused_direct"], "loss")
        served = row["fused_direct"].get("served")
        api_served = row["fused"].get("served")
        lines.append(
            f"| {pop} | {_fmt(_median(row['fused'], 'value'))} | {_fmt(kernel_v)} "
            f"| {_fmt(_median(row['specialized'], 'value'))} "
            f"| {_fmt(_median(row['torch_per_tree'], 'value'))} "
            f"| {_fmt(_median(row['torch_batched'], 'value'))} "
            f"| {_fmt(_ratio(kernel_v, _median(row['specialized'], 'value')), 2)} "
            f"| {_fmt(_ratio(kernel_v, _median(row['torch_per_tree'], 'value')), 2)} "
            f"| {_fmt(_ratio(kernel_v, _median(row['torch_batched'], 'value')), 2)} "
            f"| {'yes' if served else ('no' if served is False else 'n/a')} "
            f"| {'yes' if api_served else ('no' if api_served is False else 'n/a')} |")
        losses.append(
            f"| {pop} | {_fmt(_median(row['fused'], 'loss'))} | {_fmt(kernel_l)} "
            f"| {_fmt(_median(row['specialized'], 'loss'))} "
            f"| {_fmt(_median(row['torch_per_tree'], 'loss'))} "
            f"| {_fmt(_median(row['torch_batched'], 'loss'))} "
            f"| {_fmt(_ratio(kernel_l, _median(row['specialized'], 'loss')), 2)} "
            f"| {_fmt(_ratio(kernel_l, _median(row['torch_per_tree'], 'loss')), 2)} "
            f"| {_fmt(_ratio(kernel_l, _median(row['torch_batched'], 'loss')), 2)} |")
    lines.extend(losses)
    lines += ["", "## Route policy per population", "",
              "| pop | `auto` | `EVOGP_EVAL_MODE=fused` |", "|---|---|---|"]
    for pop in pops:
        inherited = records.get((pop, "fused_direct"), {}).get("route_inherited", {})
        requested = records.get((pop, "fused_direct"), {}).get("route_fused_request", {})
        lines.append(f"| {pop} | {inherited.get('engine', 'n/a')} "
                     f"| {requested.get('engine', 'n/a')} |")
    lines += ["", "## What the column says", ""]
    kernel_values = [_median(records[(pop, "fused_direct")], "value") for pop in pops
                     if (pop, "fused_direct") in records]
    kernel_losses = [_median(records[(pop, "fused_direct")], "loss") for pop in pops
                     if (pop, "fused_direct") in records]
    if kernel_values:
        lines.append(
            f"- The kernel is flat over pop {pops[0]}-{pops[-1]}: **{min(kernel_values):.3f}-"
            f"{max(kernel_values):.3f} ms** on the value surface and **{min(kernel_losses):.3f}-"
            f"{max(kernel_losses):.3f} ms** on the loss surface.")
    for reference in ("specialized", "torch_per_tree", "torch_batched"):
        crossing = next((pop for pop in pops
                         if (pop, reference) in records
                         and (records[(pop, "fused_direct")].get("timings", {})
                              .get("value") or {}).get("median_ms") is not None
                         and _median(records[(pop, reference)], "value") is not None
                         and _median(records[(pop, "fused_direct")], "value") is not None
                         and _median(records[(pop, "fused_direct")], "value")
                         < _median(records[(pop, reference)], "value")), None)
        lines.append(f"- Warm value crossover against `{reference}`: "
                     + (f"pop **{crossing}** and up." if crossing is not None
                        else "none in this column."))
    first = pops[0]
    if (first, "fused_direct") in records and (first, "specialized") in records:
        cold = (records[(first, "fused_direct")].get("timings", {}).get("value") or {}
                ).get("first_call_ms")
        saving = (_median(records[(first, "specialized")], "value")
                  - (_median(records[(first, "fused_direct")], "value") or 0))
        if cold and saving and saving > 0:
            lines.append(
                f"- At pop {first} the kernel's first call is **{cold:.0f} ms** (a Triton JIT: "
                "`FUSED.md` records ~1.6 s for the value kernel and ~0.94-0.99 s for the loss "
                "kernel from an empty cache, and ~2 ms once the disk cache is warm), against a "
                f"{saving:.3f} ms/call warm saving over `specialized`: a one-shot call needs "
                f"**≈{cold / saving:,.0f} calls** to repay it.")
    lines.append("- Route policy: `auto` and an explicit `EVOGP_EVAL_MODE=fused` both answer "
                 f"`fused` from pop {next((pop for pop in pops if (pop, 'fused_direct') in records and records[(pop, 'fused_direct')].get('route_inherited', {}).get('engine') == 'fused'), None)} "
                 "up and `specialized` below it, so the `fused` column below the floor IS the "
                 "fallback (0 launches).")
    path = os.path.join(out, "SUMMARY.md")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


def driver(args: argparse.Namespace) -> int:
    from campaign import guard
    pops: List[int] = [int(p) for p in args.pops.split(",") if p.strip()]
    arms: List[str] = [a.strip() for a in args.arms.split(",") if a.strip()]
    os.makedirs(args.out, exist_ok=True)
    driver_record: Dict[str, Any] = {
        "schema": SCHEMA, "kind": "minpop_driver",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {"one_process_per_config": True, "arms_sequential": False,
                     "sync_bracketed": True, "warmup": WARMUP, "reps": REPS},
        "geometry": {"tree_len": TREE_LEN, "input_len": INPUT_LEN, "rows": ROWS,
                     "max_layer_cnt": DEPTH},
        "configs": [], "failures": 0,
    }
    failures = 0
    for pop in pops:
        for arm in arms:
            state = guard(args.wait)
            out_path = _record_path(args.out, pop, arm)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--child",
                   "--pop", str(pop), "--arm", arm]
            started = time.time()
            proc = subprocess.run(cmd, env=_env(arm), capture_output=True, text=True,
                                 timeout=args.timeout)
            status, detail = "ok", ""
            line = next((ln for ln in proc.stdout.splitlines()
                         if ln.startswith("RESULT ")), None)
            if proc.returncode != 0 or line is None:
                status = "failed"
                detail = (proc.stderr or proc.stdout or "")[-800:]
                failures += 1
            else:
                record = json.loads(line[len("RESULT "):])
                record["guard"] = state
                with open(out_path, "w") as handle:
                    json.dump(record, handle, indent=2, sort_keys=True)
                    handle.write("\n")
            wall = time.time() - started
            driver_record["configs"].append({
                "pop": pop, "arm": arm, "record": out_path, "status": status,
                "detail": detail, "wall_s": round(wall, 2), "guard": state})
            driver_record["failures"] = failures
            print(f"[{len(driver_record['configs'])}] {status:<7}p{pop:<5} {arm:<16}"
                  f"{wall:6.1f}s (gpu_idle={state.get('ok')})", flush=True)
    with open(os.path.join(args.out, "campaign.json"), "w") as handle:
        json.dump(driver_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[driver] {len(driver_record['configs'])} configs, {failures} failures",
          flush=True)
    return 0 if failures == 0 else 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", action="store_true", help="measure ONE (pop, arm)")
    parser.add_argument("--render", action="store_true",
                        help="tabulate the records already on disk (no device, no torch)")
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--arm", default="fused", choices=list(ARMS))
    parser.add_argument("--pops", default=",".join(str(p) for p in POPS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--wait", type=float, default=180.0, help="idle-guard window")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.render:
        return render(args.out)
    return worker(args) if args.child else driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
