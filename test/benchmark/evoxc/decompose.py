"""Per-call cost decomposition of one representative cell, per evaluation route.

This is the optimisation target list: for one cell it breaks a warm engine call
into the phases the engine actually spends time in, with call counts, and it
counts the seams that dominate the residual (host<->device copies, ``evoxc.run``
dispatches, host readbacks).

Method
------
* One (cell, route, workload) per measurement, warm: the cold call is timed
  separately (``cold_ms`` + the compile misses it caused) and never enters the
  phases.
* Phases come from wrapping the engine's own module-level functions
  (``evogp.tree.evoxc_engine`` / ``evogp.tree.dense_engine``) with a
  ``perf_counter_ns`` timer.  Only functions that are called O(population) times
  or fewer are wrapped -- wrapping a per-node function would cost more than the
  phase it measures -- and the *uninstrumented* median of the same call is
  recorded next to the instrumented one so the instrumentation overhead is
  visible instead of being smuggled into the numbers.
* Times are INCLUSIVE per function.  ``total`` is the whole workload call, so a
  phase's ``residual`` (total minus its instrumented children) is printed as its
  own row: for ``sr_fitness`` that residual is the host-side loss assembly and
  the loss upload; for ``forward`` it is the output assembly.
* Counters count what the µs table cannot show: ``evoxc.run`` invocations (one
  per non-constant tree on the specialized route), ``evoxc.Tensor.to`` (host ->
  device uploads), ``evoxc.Tensor.numpy`` (device -> host readbacks) and
  ``evoxc.from_numpy`` (host wraps).  ``cache_stats()`` deltas give compile
  misses / memo hits per call.

Usage (repo root, project venv, one route per process is NOT required here --
every route runs sequentially in this one process, and each route is warmed
before it is measured)::

    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \
        $PY test/benchmark/evoxc/decompose.py \
        --cell p256-L64-i32-N1024 --out test/benchmark/evoxc/results/decompose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402

#: route name -> (EVOGP_EVAL_MODE, EVOGP_EVOXC_BACKEND)
ROUTES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "auto": (None, None),
    "auto-iree": (None, "iree"),
    "specialized-numpy": ("specialized", "numpy"),
    "specialized-iree": ("specialized", "iree"),
    "dense-iree": ("dense", "iree"),
}

#: functions whose inclusive cost is reported, per engine module
PHASES: Dict[str, List[str]] = {
    "evoxc_engine": [
        "_require_population",
        "decode_population",
        "_input_to_numpy",
        "_as_numpy",
        "_evaluate_rows",
        "_executable",
        "_run_tree",
        "evaluate",
        "sr_fitness",
    ],
    "dense_engine": [
        "prepare_dense",
        "plan_dense",
        "build_dense",
        "run_loss_plan",
        "run_plan",
        "evaluate_dense",
        "sr_fitness_dense",
        "_pad_rows",
        "_prep_columns",
        "_decode_columns",
        "_to_host",
    ],
}

#: top-level child of the workload call, per workload and route family: the
#: residual of the workload is reported against these
TOP_CHILDREN = {
    ("specialized", "sr_fitness"): ["_require_population", "_input_to_numpy", "_as_numpy", "_evaluate_rows"],
    ("specialized", "forward"): ["_require_population", "_input_to_numpy", "_evaluate_rows"],
    ("specialized", "batch_forward"): ["_require_population", "_input_to_numpy", "_evaluate_rows"],
    ("dense", "sr_fitness"): ["prepare_dense", "plan_dense", "build_dense", "run_loss_plan"],
    ("dense", "forward"): ["prepare_dense", "plan_dense", "build_dense", "run_plan"],
    ("dense", "batch_forward"): ["prepare_dense", "plan_dense", "build_dense", "run_plan"],
}


class Instrument:
    """Phase timers + seam counters, installed on the engine modules."""

    def __init__(self) -> None:
        self.phases: Dict[str, Dict[str, float]] = {}
        self.counters: Dict[str, int] = {}
        self._undo: List[Tuple[Any, str, Any]] = []

    # -- introspection ------------------------------------------------------
    def reset(self) -> None:
        self.phases = {}
        self.counters = {key: 0 for key in self.counters}

    def _phase(self, name: str) -> Dict[str, float]:
        return self.phases.setdefault(name, {"calls": 0, "ns": 0.0})

    def _count(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by

    # -- installation -------------------------------------------------------
    def _wrap_phase(self, module: Any, name: str) -> None:
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                record = self._phase(name)
                record["calls"] += 1
                record["ns"] += time.perf_counter_ns() - start

        setattr(module, name, wrapper)
        self._undo.append((module, name, original))

    def _wrap_counter(self, module: Any, name: str) -> None:
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            self._count(name)
            return original(*args, **kwargs)

        setattr(module, name, wrapper)
        self._undo.append((module, name, original))

    def install(self) -> None:
        # engine handles resolved through ``sys.modules`` (see ``common.installed_module``)
        dense_engine = common.installed_module("evogp.tree.dense_engine")
        evoxc_engine = common.installed_module("evogp.tree.evoxc_engine")

        for module, names in ((evoxc_engine, PHASES["evoxc_engine"]), (dense_engine, PHASES["dense_engine"])):
            for name in names:
                if hasattr(module, name):
                    self._wrap_phase(module, name)
        evoxc = evoxc_engine.evoxc
        for name in ("run", "from_numpy"):
            if hasattr(evoxc, name):
                self._wrap_counter(evoxc, name)
        tensor_cls = getattr(getattr(evoxc, "core", None), "Tensor", None)
        if tensor_cls is not None:
            for name in ("to", "numpy"):
                if hasattr(tensor_cls, name):
                    self._wrap_counter(tensor_cls, name)

    def remove(self) -> None:
        for module, name, original in reversed(self._undo):
            setattr(module, name, original)
        self._undo = []

    # -- reporting ----------------------------------------------------------
    def rows(self, total_ms: float, top: List[str]) -> List[Dict[str, Any]]:
        total_us = total_ms * 1e3
        calls_per_call = max((self.phases.get(name, {}).get("calls", 0) for name in top), default=0)
        rows: List[Dict[str, Any]] = []
        announced = set()
        for name in top:
            record = self.phases.get(name)
            if not record or not record["calls"]:
                continue
            announced.add(name)
            rows.append(
                {
                    "phase": name,
                    "calls": record["calls"],
                    "us": round(record["ns"] / 1e3 / max(1, calls_per_call), 3),
                    "percent": round(record["ns"] / 1e3 / total_us * 100.0, 1) if total_us else 0.0,
                }
            )
        accounted = sum(row["us"] for row in rows)
        rows.append(
            {
                "phase": "residual (glue: loss assembly, conversions, uploads)",
                "calls": 1,
                "us": round(total_us - accounted, 3),
                "percent": round((total_us - accounted) / total_us * 100.0, 1) if total_us else 0.0,
            }
        )
        for name, record in sorted(self.phases.items()):
            if name in announced or not record["calls"]:
                continue
            rows.append(
                {
                    "phase": f"  .{name} (nested)",
                    "calls": record["calls"],
                    "us": round(record["ns"] / 1e3 / max(1, calls_per_call), 3),
                    "percent": round(record["ns"] / 1e3 / total_us * 100.0, 1) if total_us else 0.0,
                }
            )
        return rows


def _set_route(eval_mode: Optional[str], backend: Optional[str]) -> None:
    for key, value in (("EVOGP_EVAL_MODE", eval_mode), ("EVOGP_EVOXC_BACKEND", backend)):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def measure(
    workload_obj: Any,
    workload: str,
    route: str,
    family: str,
    warmup: int,
    reps: int,
) -> Dict[str, Any]:
    evoxc_engine = common.installed_module("evogp.tree.evoxc_engine")

    call = getattr(workload_obj, workload)
    stats = evoxc_engine.cache_stats

    # -- cold call + warmup (never part of the phases) ----------------------
    cache_before = stats()
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    call()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1e3
    cache_after_cold = stats()
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    def timed(rep: int) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e3

    plain = sorted(timed(rep) for rep in range(reps))

    # -- instrumented reps --------------------------------------------------
    instrument = Instrument()
    instrument.install()
    try:
        instrument.reset()
        instrumented_calls = timed(0)
        phase_rows = instrument.rows(instrumented_calls, TOP_CHILDREN[(family, workload)])
        counters_first = dict(instrument.counters)
        instrumented = []
        for rep in range(1, reps):
            instrument.reset()
            instrumented.append(timed(rep))
            counters_sample = dict(instrument.counters)
        counters = counters_sample if reps > 1 else counters_first
    finally:
        instrument.remove()
    cache_after = stats()

    return {
        "route": route,
        "family": family,
        "workload": workload,
        "cold_ms": round(cold_ms, 3),
        "cold_compile_misses": cache_after_cold["misses"] - cache_before["misses"],
        "plain_ms": {"median": round(plain[len(plain) // 2], 3), "min": round(plain[0], 3), "samples": [round(v, 3) for v in plain]},
        "instrumented_ms": {"median": round(sorted(instrumented)[len(instrumented) // 2], 3), "samples": [round(v, 3) for v in instrumented]},
        "phases": phase_rows,
        "counters_per_call": counters,
        "cache_after": {
            key: cache_after[key] - cache_before[key]
            for key in ("hits", "misses", "memo_hits", "memo_misses")
        },
        "structures": None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="p256-L64-i32-N1024")
    ap.add_argument("--routes", default="specialized-numpy,specialized-iree,dense-iree,auto")
    ap.add_argument("--workloads", default="sr_fitness,forward")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="", help="result directory (one JSON per route)")
    ap.add_argument("--tree-root", default="src")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.tree_root)
    src = os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root
    if src not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, src)

    cell = common.parse_cell(args.cell)
    results: List[Dict[str, Any]] = []
    for route in [token.strip() for token in args.routes.split(",") if token.strip()]:
        if route not in ROUTES:
            raise SystemExit(f"unknown route {route!r}; known: {sorted(ROUTES)}")
        eval_mode, backend = ROUTES[route]
        _set_route(eval_mode, backend)
        # rebuild per route so no route benefits from another's decoded state
        built = common.build_workload(cell, seed=0)
        forest = built["forest"]
        target = type("Target", (), {})()
        target.sr_fitness = lambda f=forest, b=built: f.SR_fitness(b["x"], b["labels"])
        target.forward = lambda f=forest, b=built: f.forward(b["x_per_row"])
        target.batch_forward = lambda f=forest, b=built: f.batch_forward(b["x"])
        family = "dense" if common.route_decision(
            cell["pop"], cell["rows"], eval_mode, backend
        )["would_use_dense"] else "specialized"
        for workload in [token.strip() for token in args.workloads.split(",") if token.strip()]:
            entry = measure(target, workload, route, family, args.warmup, args.reps)
            entry["cell"] = cell["id"]
            entry["route_env"] = {"EVOGP_EVAL_MODE": eval_mode, "EVOGP_EVOXC_BACKEND": backend}
            entry["route_resolved"] = common.route_decision(
                cell["pop"], cell["rows"], eval_mode, backend
            )
            results.append(entry)
            print(
                f"[decompose] {cell['id']:<22} {route:<19} {workload:<14} "
                f"plain={entry['plain_ms']['median']:8.3f}ms "
                f"instr={entry['instrumented_ms']['median']:8.3f}ms "
                f"cold={entry['cold_ms']:8.3f}ms ({family})",
                flush=True,
            )
            for row in entry["phases"]:
                print(
                    f"    {row['phase']:<52} calls={row['calls']:<6} "
                    f"{row['us']:9.1f}us  {row['percent']:5.1f}%",
                    flush=True,
                )
            print(f"    counters: {entry['counters_per_call']}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = {
            "schema": common.SCHEMA + "-decompose",
            "env": common.env_block(
                common.checkout_root(), arm="decompose",
                extra={
                    "note": args.note,
                    "protocol": {
                        "warmup": args.warmup,
                        "reps": args.reps,
                        "method": "module-level inclusive perf_counter_ns phase timers + seam counters",
                        "cold_call_excluded": True,
                    },
                },
            ),
            "cell": cell,
            "entries": results,
        }
        common.write_json(os.path.join(args.out, f"{cell['id']}.json"), payload)
        print(f"[decompose] wrote {os.path.join(args.out, cell['id'] + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
