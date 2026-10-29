"""The child side of the band sweep: measure ONE (cell, config) and assemble its record.

``band_sweep.py`` is the DRIVER (CLI, cell/config resolution, the route env of each config,
the idle-device guard, one child process per (cell, config), ``sweep.json``); this module is
what that child runs -- the per-workload measurement loop, the oracle/validation wiring and the
record skeleton -- so neither file has to carry both concerns.

**Torch-free at import.**  Everything that needs ``torch`` / ``evogp`` imports it *inside*
:func:`measure`, i.e. after the child has applied its config env and put the measured tree
first on ``sys.path``.  A record is therefore never built from a tree the caller did not name,
and importing this module costs no GPU.

Per (cell, config) the workloads are measured in the driver's FIXED order (``sr_fitness`` --
the training hot path -- then ``batch_forward``, then the per-row ``forward``).  Per workload
the record keeps the cold first call (``first_call_ms``: the merged-graph compile of that
plan), the warm ``samples_ms`` with their median/min/max/mean, the value ``digest``, the
merged-graph counter deltas, the ten-stage split (``stage_ms`` / ``stage_first_call``, from
``stages.py``) and the ``band_report`` of the plan that served it (from ``common.band_report``,
i.e. ``band_report.py``), plus one ``validation`` block per workload: the engine output diffed
against the independent CPU oracle (``src/evogp/tests/reference_eval.py``) with
``validate.compare``, the ONE comparator this harness already trusts.

``--cold-only`` (round 3's re-measure path) measures ONLY that first synchronized call per
workload: no warmup, no timed reps.  The absent warm fields are recorded null-never-zero
(``samples_ms: []``, ``median_ms``/``min_ms``/``max_ms``/``mean_ms``: ``null``, ``warmup`` /
``reps``: ``0``, ``stage_ms`` an ``stages.unavailable_block`` with a note) -- never a
fabricated median and never a ``0`` that would read as "it took no time".

A route that cannot run, a compile/binding failure or any other exception is written into the
record as ``status: "failed"`` (with ``error`` / ``error_type`` / ``traceback``) and exits
non-zero: a failure is an OUTCOME of this config, never a crash and never a silent skip.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)
import counter_blocks  # noqa: E402  (torch-free: the ONE getattr probe for optional counters)
import stages  # noqa: E402  (torch-free, evogp-free: the stage vocabulary + derived blocks)
import worker as worker_mod  # noqa: E402  (counter deltas + plan facts; torch-free at import)
from batch_economics import _round  # noqa: E402  (the generic helpers of this node)

MEASUREMENT = "batch-band-sweep"
SCHEMA = common.SCHEMA + "-band-sweep"

#: The route every config of the sweep runs; a config's own spec only adds its band/pool knobs
#: (``band_sweep.config_env`` is the ONE place that turns a config token into that env).
ROUTE_EVAL_MODE = "batch"
ROUTE_BACKEND = "iree"

#: The oracle entry point of each workload (``reference_eval``, imported from the tree under
#: measurement): recorded next to every validation block so the diff is reproducible.
ORACLE_FNS: Dict[str, str] = {
    "sr_fitness": "sr_fitness_from_rows",
    "batch_forward": "batch_forward_from_rows",
    "forward": "forward_from_rows",
}

#: The two env vars the record reports the tracing switch through (the driver's child env sets
#: the second one to state whether it defaulted ``EVOXC_DISABLE_LOCATIONS``).
DISABLE_LOCATIONS_ENV = "EVOXC_DISABLE_LOCATIONS"
DIS_SOURCE_ENV = "EVOGP_BAND_SWEEP_DIS_SOURCE"

#: Why ``stage_ms`` is an unavailable block under ``--cold-only`` (null, never a 0 per stage).
COLD_ONLY_STAGE_NOTE = (
    "cold-only: one synchronized call per workload was measured, so there is no warm window to "
    "take a per-call stage median over -- the warm split was NOT measured (which is not the same "
    "as any stage costing nothing); stage_first_call carries that one call's own block"
)
#: The record-level statement of what ``--cold-only`` did (and did not) measure.
COLD_ONLY_NOTE = (
    "cold-only (--cold-only): ONE synchronized call per workload -- the merged-graph compile of "
    "that workload's plan -- with no warmup and no timed reps, so every warm field is null and "
    "samples_ms is [] (NOT a measured 0): median_ms / min_ms / max_ms / mean_ms and stage_ms were "
    "not measured, first_call_ms / digest / stage_first_call / band_report / validation were"
)


class RouteUnavailable(RuntimeError):
    """The tree under measurement cannot run ``EVOGP_EVAL_MODE=batch``.

    Raised BEFORE any timing exists, so the child records one readable paragraph as a
    ``status: "failed"`` record instead of crashing (a missing route must never look like a
    measured one).
    """


# --- workload plumbing ------------------------------------------------------
def workload_fn(name: str, workload: Dict[str, Any]):
    """The PUBLIC-API call of one workload (no harness arm stands in between)."""
    forest = workload["forest"]
    if name == "sr_fitness":
        x, labels = workload["x"], workload["labels"]
        return lambda: forest.SR_fitness(x, labels)
    if name == "batch_forward":
        x = workload["x"]
        return lambda: forest.batch_forward(x)
    x_per_row = workload["x_per_row"]
    return lambda: forest.forward(x_per_row)


def workload_plan(name: str, batch_module: Any) -> Tuple[str, bool]:
    """The ``(variant, per_row)`` pair the workload's merged plan is keyed on."""
    value = str(getattr(batch_module, "VALUE_VARIANT", "value"))
    loss = str(getattr(batch_module, "LOSS_VARIANT", "loss"))
    pairs = {
        "sr_fitness": (loss, False),      # the loss variant is per_row=False by contract
        "batch_forward": (value, False),
        "forward": (value, True),         # tree i sees row i
    }
    return pairs[name]


def plan_for(
    facts: Dict[str, Any], wanted: Tuple[str, bool]
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """The ``describe_plan`` view and the ``BatchPlan`` of one workload (``None`` if absent).

    ``worker.batch_plan_facts`` keeps ``plans`` (describe_plan views) and ``objects`` (the
    plan objects, parallel to ``plans``) so the band report can read the plan's OWN report --
    the only place the two cost-model estimates live -- without a second memo walk here.
    """
    plans = facts.get("plans") or []
    objects = facts.get("objects") or []
    for position, described in enumerate(plans):
        if (described.get("variant"), bool(described.get("per_row"))) == wanted:
            return described, (objects[position] if position < len(objects) else None)
    return None, None


# --- oracle / validation ----------------------------------------------------
def _oracle_module():
    """Import the independent CPU oracle from the tree under measurement."""
    oracle_dir = os.path.join(common.src_root(), "evogp", "tests")
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    import reference_eval

    return reference_eval


def _oracle_tensor(
    name: str, reference_eval: Any, rows: Dict[str, Any], workload: Dict[str, Any],
    out_len: int,
):
    """One workload's CPU oracle, built exactly as ``validate.parse_case`` builds it."""
    if name == "sr_fitness":
        return reference_eval.sr_fitness_from_rows(
            rows["node_value"], rows["node_type"], rows["subtree_size"],
            workload["x"].to("cpu"),
            workload["labels"].to("cpu").reshape(rows["n_rows"], -1),
            out_len=out_len,
        )
    if name == "batch_forward":
        return reference_eval.batch_forward_from_rows(
            rows["node_value"], rows["node_type"], rows["subtree_size"],
            workload["x"].to("cpu"), out_len=out_len,
        )
    return reference_eval.forward_from_rows(
        rows["node_value"], rows["node_type"], rows["subtree_size"],
        workload["x_per_row"][: rows["trees"]].to("cpu"), out_len=out_len,
    )


def validation_block(name: str, reference, candidate, trees: int) -> Dict[str, Any]:
    """``validate.compare`` (the ONE comparator) of the engine output against the oracle."""
    import validate  # torch at import: child-side only, never the driver

    block = validate.compare(reference, candidate[: int(reference.shape[0])])
    block.update({
        "oracle": f"evogp/tests/reference_eval.py:{ORACLE_FNS[name]}",
        "trees_compared": int(reference.shape[0]),
        "trees_pop": int(trees),
        "oracle_digest": common.digest(reference),
    })
    return block


# --- provenance / preflight -------------------------------------------------
def identity() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(evoxc_version, evogp_file, evogp_head)`` -- best effort, never raises."""
    evoxc_version = None
    evogp_file = None
    evogp_head = None
    try:
        import evoxc

        evoxc_version = getattr(evoxc, "__version__", None)
    except Exception:  # noqa: BLE001 - provenance only
        evoxc_version = None
    try:
        import evogp

        evogp_file = getattr(evogp, "__file__", None)
    except Exception:  # noqa: BLE001 - provenance only
        evogp_file = None
    try:
        evogp_head = common._git_head(common.checkout_root())
    except Exception:  # noqa: BLE001 - a snapshot without .git is a legal tree root
        evogp_head = None
    return evoxc_version, evogp_file, evogp_head


def route_preflight(policy: Dict[str, Any], src: str) -> Optional[str]:
    """The refusal paragraph for a tree that cannot run the batch route (``None`` = OK)."""
    if policy.get("module_present") and policy.get("mode_supported"):
        return None
    return (
        "EVOGP_EVAL_MODE=batch needs BOTH a <src>/evogp/tree/batch_engine.py module AND "
        "'batch' declared in evogp.tree.dense_engine.EVAL_MODES, and this tree does not "
        f"provide the pair -- module_present={policy.get('module_present')} "
        f"(expected {policy.get('module')}, src={src}), "
        f"declared_modes={policy.get('declared_modes')!r} "
        f"(mode_supported={policy.get('mode_supported')}). No timing was attempted."
    )


# --- the record skeleton ----------------------------------------------------
def guard_block(raw: Optional[str]) -> Dict[str, Any]:
    """The driver's idle-device verdict, as the child records it (never raises)."""
    import json

    if not raw:
        return {"available": False, "note": "no --guard verdict was handed to this child"}
    try:
        state = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - a malformed verdict is a reportable state
        return {"available": False, "unparsed": str(raw)[:400], "error": repr(exc)}
    if not isinstance(state, dict):
        return {"available": False, "unparsed": str(state)[:400]}
    return {"available": True, "state": state}


def env_block(
    tree_root: str, cell: Dict[str, Any], config: str, mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """``common.env_block`` plus the exact env this config set (never raises)."""
    extra = {
        "applied_knobs": dict(mapping),
        "knob_source": "band_sweep-config",
        "cell": cell, "config": config,
    }
    try:
        return common.env_block(tree_root, "band-sweep", extra=extra)
    except Exception as exc:  # noqa: BLE001 - a torch-free child box is reportable
        block: Dict[str, Any] = {
            "schema": common.SCHEMA, "tree_root": os.path.abspath(tree_root),
            "note": f"env block unavailable: {exc!r}",
        }
        block.update(extra)
        return block


def build_record(
    args: Any, cell: Dict[str, Any], config: str, src: str, tree_root: str,
    mapping: Dict[str, Optional[str]], workloads: List[str],
    plan_cells: List[str], plan_configs: List[str],
) -> Dict[str, Any]:
    """The record skeleton one child fills (``measure`` raises into ``status: failed``).

    ``cold_only`` (a top-level key and a protocol key) appears ONLY on a ``--cold-only`` record:
    a warm record keeps the shape the previous round's records have, so the same config token
    stays byte-comparable across rounds.
    """
    cold_only = bool(getattr(args, "cold_only", False))
    evoxc_version, evogp_file, evogp_head = identity()
    notes = [
        "warm medians only: this child's first call is the COLD merged-graph compile of "
        "that workload's plan (first_call_ms) and never enters median_ms",
        "the pool width and the banded+chunked composition are CONFIG knobs "
        "(EVOGP_BATCH_COMPILE_WORKERS / EVOGP_BATCH_CHUNK_BANDS), so each record states the "
        "merge shape it ran instead of inheriting one",
        "record['validation'] is keyed by workload name: one compare(oracle, engine) "
        "block per workload, computed AFTER the timed window on the first --max-oracle "
        "rows (oracle = src/evogp/tests/reference_eval.py), so a cross-config comparison "
        "reads validation[workload].bitwise_equal / max_abs / oracle_digest",
        "band_report.cost_estimate / flat_estimate read null unless EVOGP_BATCH_BANDS=auto "
        "ran the static cost model -- a pinned band size or the flat merge never runs it, "
        "so null (NOT 0) is the expected value there, and band_requested / "
        "band_requested_bands / band_clamped / band_note / band_policy state what the model "
        "chose and whether the binding policy lowered it",
    ]
    if cold_only:
        notes[0] = COLD_ONLY_NOTE
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "status": "ok",
        "error": None,
        "error_type": None,
        "cell": cell,
        "config": config,
        "env": env_block(tree_root, cell, config, mapping),
        "guard": guard_block(args.guard),
        "protocol": {
            "cells": plan_cells, "configs": plan_configs,
            "cell": cell["id"], "config": config,
            "workloads": workloads,
            # a --cold-only record measured no warm window, so it states 0 rather than echoing
            # a --warmup/--reps request it deliberately skipped
            "warmup": 0 if cold_only else args.warmup,
            "reps": 0 if cold_only else args.reps,
            "seed": args.seed,
            "workers": args.workers, "max_oracle_trees": args.max_oracle,
            "config_env": mapping,
            "one_process_per_config": True, "one_config_per_process": True,
            "sequential": True, "sync_bracketed": True,
            "tree_root": os.path.abspath(tree_root), "tree_root_arg": args.tree_root,
            "src": src,
            "evoxc_root": os.path.abspath(args.evoxc_root),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "evoxc_version": evoxc_version,
            "evogp_file": evogp_file,
            "evogp_head": evogp_head,
            "evoxc_disable_locations": {
                "env": DISABLE_LOCATIONS_ENV,
                "value": os.environ.get(DISABLE_LOCATIONS_ENV),
                "source": os.environ.get(DIS_SOURCE_ENV, "unknown"),
                "note": "the driver records whether it defaulted the value (sweep.json "
                        "protocol.evoxc_disable_locations); it is not part of the engine's "
                        "cache key",
            },
            "note": args.note,
        },
        "batch_policy": None,
        "routes": {},
        "workloads": {},
        "validation": {},
        "notes": notes,
    }
    if cold_only:
        record["cold_only"] = True
        record["protocol"]["cold_only"] = True
    if args.note:
        record["notes"].append(args.note)
    return record


# --- child phase: ONE cell, ONE config, ONE process -------------------------
def run_child(
    args: Any, cell: Dict[str, Any], config: str, src: str, tree_root: str,
    mapping: Dict[str, Optional[str]], workloads: List[str],
    plan_cells: List[str], plan_configs: List[str],
) -> int:
    """Assemble + write one ``(cell, config)`` record (returns a process exit code).

    The driver has already resolved the cell/config, put the measured tree first on
    ``sys.path`` and applied the config env; this function only records and measures, and it
    turns every exception into a ``status: "failed"`` record rather than a crash.
    """
    if not args.out:
        raise SystemExit("child mode needs --out <record path>")
    record = build_record(
        args, cell, config, src, tree_root, mapping, workloads, plan_cells, plan_configs
    )
    try:
        measure(record, args, cell, config, src, workloads)
    except Exception as exc:  # noqa: BLE001 - a failure is a RECORDED outcome
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_type"] = type(exc).__name__
        record["traceback"] = traceback.format_exc()[-4000:]
        record["notes"].append(
            "this config did NOT produce timings: the record exists so the failure is an "
            "outcome of the sweep rather than a crash or a silent skip"
        )
        common.write_json(args.out, record)
        print(f"[band-sweep] FAILED {cell['id']} {config}: {record['error']} -> {args.out}",
              flush=True)
        return 1
    common.write_json(args.out, record)
    first = record["workloads"].get(workloads[0], {})
    if first.get("median_ms") is not None:
        headline = f"first_workload_median={first['median_ms']:.3f}ms"
    else:
        headline = f"first_workload_first_call={first.get('first_call_ms')}ms (cold-only)"
    print(f"[band-sweep] {cell['id']} {config} workloads={len(record['workloads'])} "
          f"{headline} bands="
          f"{ {name: entry['band_report'].get('bands') for name, entry in record['workloads'].items()} } "
          f"-> {args.out}", flush=True)
    return 0


# --- the measurement --------------------------------------------------------
def _cache_entry(
    available: bool, before: Any, after_first: Any, after: Any
) -> Dict[str, Any]:
    """The merged-graph counter deltas of one workload (``None`` counters when absent)."""
    return {
        "available": available,
        "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
        "before": before,
        "after_first_call": after_first,
        "after": after,
        "stats_keys": worker_mod.stats_keys(before, after),
        "batch_exec_misses_first_call": worker_mod.batch_miss_delta(before, after_first),
        "batch_exec_misses_this_workload": worker_mod.batch_miss_delta(before, after),
        "batch_plan_misses_first_call": worker_mod.batch_plan_miss_delta(before, after_first),
        "batch_plan_misses_this_workload": worker_mod.batch_plan_miss_delta(before, after),
    }


def _cold_call(probe, batch_stats):
    """ONE sync-bracketed call of ``probe``: ``(first_ms, out, cache_after)``.

    The same bracket ``common.time_call`` puts around its untimed first call, without the
    warmup and the timed reps: that first call is where the merged-graph compile lands, which
    is exactly what ``--cold-only`` re-measures.
    """
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    out = probe()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1e3
    return first_ms, out, (batch_stats() if batch_stats is not None else None)


def measure(
    record: Dict[str, Any], args: Any, cell: Dict[str, Any], config: str, src: str,
    workloads: List[str],
) -> None:
    """Fill ``record`` from one freshly built population (imports torch/evogp here only)."""
    import torch  # noqa: F401  (the public API relocates every tensor to the device)

    common.import_evogp()
    from evogp.tree import evoxc_engine

    cold_only = bool(getattr(args, "cold_only", False))
    policy = common.batch_policy(src)
    record["batch_policy"] = policy
    problem = route_preflight(policy, src)
    if problem:
        raise RouteUnavailable(problem)
    try:
        from evogp.tree import batch_engine as batch_module
    except ImportError as exc:  # the route can be absent on an older tree
        raise RouteUnavailable(
            f"evogp.tree.batch_engine could not be imported from {src}: {exc!r}"
        ) from exc

    # The batch engine owns its counter contract, so ask it directly (getattr-probed: a route
    # without ``batch_cache_stats`` is a reportable state -- null counters, never a 0).
    batch_stats = counter_blocks.accessor(batch_module, "batch_cache_stats")
    # The stage split and its reset are a SECOND optional contract on the same module.
    stage_stats = stages.stage_stats_fn(batch_module)
    stage_reset = stages.stage_reset_fn(batch_module)

    def batch_snapshot() -> Optional[Dict[str, Any]]:
        return batch_stats() if batch_stats is not None else None

    workload = common.build_workload(cell, args.seed)
    forest = workload["forest"]
    signature = common.population_signature(forest)
    for name in workloads:
        rows = 1 if name == "forward" else int(cell["rows"])
        record["routes"][name] = common.route_decision(
            cell["pop"], rows, ROUTE_EVAL_MODE, ROUTE_BACKEND
        )

    # The oracle is built from a CPU slice of the SAME population, sliced exactly like
    # validate.parse_case does, so the diff is the harness's own comparator.
    trees = (
        min(int(forest.pop_size), int(args.max_oracle))
        if int(args.max_oracle) > 0
        else int(forest.pop_size)
    )
    n_rows = int(workload["x"].shape[0])
    oracle_rows = {
        "trees": trees,
        "n_rows": n_rows,
        "node_value": forest.batch_node_value[:trees].to("cpu"),
        "node_type": forest.batch_node_type[:trees].to("cpu"),
        "subtree_size": forest.batch_subtree_size[:trees].to("cpu"),
    }
    reference_eval = _oracle_module()

    facts: Dict[str, Any] = {"available": False, "note": None, "plans": [], "summary": {}}
    for name in workloads:
        fn = workload_fn(name, workload)
        cache_before = batch_snapshot()
        # The probe brackets every call (reset + snapshot) when the route ships the reset, so
        # ``stage_ms`` below is a per-call block rather than a cumulative one.
        probe = stages.StageStatsProbe(fn, stage_stats, stage_reset)
        if cold_only:
            first_ms, result, cache_after_first = _cold_call(probe, batch_stats)
            cache_after = batch_snapshot()
            entry: Dict[str, Any] = {
                "cold_only": True,
                "warmup": 0, "reps": 0, "samples_ms": [],
                "first_call_ms": _round(first_ms),
                # null, never 0: no warm window was measured, so there is no median to report
                "median_ms": None, "min_ms": None, "max_ms": None, "mean_ms": None,
                "digest": common.digest(result),
                "cache": _cache_entry(
                    batch_stats is not None, cache_before, cache_after_first, cache_after),
            }
            entry["stage_ms"] = stages.unavailable_block(note=COLD_ONLY_STAGE_NOTE)
            entry["stage_first_call"] = probe.first
        else:
            timing = common.time_call(probe, warmup=args.warmup, reps=args.reps, stats=batch_stats)
            cache_after_first = timing.pop("cache_after_first_call")
            cache_after = batch_snapshot()
            result = timing.pop("_out")
            entry = {
                "warmup": timing["warmup"], "reps": timing["reps"],
                "samples_ms": [_round(value) for value in timing["samples_ms"]],
                # this child has no separate warm first call: time_call's untimed call IS the
                # cold one, i.e. where the merged-graph compile of this workload's plan lands
                "first_call_ms": _round(timing["first_call_ms"]),
                "median_ms": _round(timing["median_ms"]),
                "min_ms": _round(timing["min_ms"]),
                "max_ms": _round(timing["max_ms"]),
                "mean_ms": _round(timing["mean_ms"]),
                "digest": common.digest(result),
                "cache": _cache_entry(
                    batch_stats is not None, cache_before, cache_after_first, cache_after),
            }
            sample_blocks = probe.blocks[-args.reps:] if probe.blocks else []
            entry["stage_ms"] = stages.median_block(sample_blocks, samples=args.reps)
            entry["stage_first_call"] = probe.first
        facts = worker_mod.batch_plan_facts(batch_module)
        entry["stage_ms"]["plan"] = stages.plan_block(facts)
        described, plan = plan_for(facts, workload_plan(name, batch_module))
        entry["band_report"] = common.band_report(plan, described)
        # the oracle + the comparator run AFTER the timed window: CPU host work never times a
        # call, and the value is kept as a block rather than as a tensor
        reference = _oracle_tensor(
            name, reference_eval, oracle_rows, workload, int(forest.output_len))
        record["validation"][name] = validation_block(
            name, reference, result, int(forest.pop_size))
        del reference
        del result
        record["workloads"][name] = entry
        median = entry["median_ms"]
        print(f"[band-sweep]   {cell['id']} {config} {name}: "
              f"first={entry['first_call_ms']:.1f}ms "
              f"median={'n/a' if median is None else round(median, 3)}ms "
              f"bands={entry['band_report'].get('bands')} "
              f"n_functions={entry['band_report'].get('n_functions')}", flush=True)

    final = batch_snapshot()
    record["batch_cache_final"] = {
        "available": batch_stats is not None,
        "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
        "stats": final,
        "stats_keys": worker_mod.stats_keys(final),
    }
    facts = worker_mod.batch_plan_facts(batch_module)
    record["batch_plans"] = facts["plans"]
    record["batch_plan_summary"] = (
        facts["summary"] if facts["available"]
        else {"available": False, "note": facts["note"],
              "band_report": common.band_report(None)}
    )
    record["cell"].update({
        "population_signature": signature["signature"],
        "distinct_structures": int(signature["distinct_structures"]),
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "output_len": int(forest.output_len),
        "n_functions": {
            key: (value or {}).get("n_functions")
            for key, value in (facts["summary"] or {}).items()
        },
    })
    record["engine_cache_final"] = evoxc_engine.cache_stats()
    if batch_stats is None:
        record["notes"].append(
            "evogp.tree.batch_engine.batch_cache_stats() is unavailable on this tree, so every "
            "merged-graph counter is null (NOT zero) and the compile cost can only be read off "
            "first_call_ms"
        )
    if stage_stats is None:
        record["notes"].append(
            f"{stages.STATS_FN_PATH}() is unavailable on this tree, so every stage_ms counter "
            "is null (NOT zero) with available=false: the split was not measured, which is not "
            "the same as any stage costing nothing"
        )
    elif stage_reset is None:
        record["notes"].append(
            f"{stages.MODULE_NAME}.{stages.RESET_FN_NAME}() is unavailable, so each stage block "
            "is cumulative rather than per-call; a max is then only a process-lifetime maximum"
        )
    if not facts["available"]:
        record["notes"].append(
            "the merged-graph plan facts are unavailable (no BATCH_PLAN_MEMO / describe_plan), "
            "so band_report is available=false with every band field null (NOT zero)"
        )
