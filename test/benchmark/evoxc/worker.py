"""One process, one (cell, arm): the measured unit of the campaign.

Why a process per config: the engine's per-structure compile cache and its
executable memo live in process memory, so a fresh process per (cell, arm) is
the only way every record pays its own cold compile and no arm is helped by
another arm's warm cache.  The driver (``campaign.py``) spawns these one at a
time.

What one record contains: the environment block, the cell and arm, the route the
policy actually resolved, the per-workload timings (``first_call_ms`` = cold,
then warmup + reps -> median/min/max/samples), the cache deltas (compile
misses), value digests (cross-process numeric agreement) and the GP generation
step split into eval / operator parts with its per-generation cache misses.

What a BATCH arm (``EVOGP_EVAL_MODE=batch``) adds on top of that -- and only a
batch arm: ``batch_policy`` (whether the tree under test ships the module and
declares the mode, plus the band knobs and where each value came from),
``batch_cache`` (the merged-graph cache counters before the workloads and after
the whole config, which counter names the route exposes, and the first-call and
total deltas of ``batch_exec_misses`` -- the real compile cache, a hit never
reaches ``evoxc.lower``/``compile``/``load`` -- and of ``batch_plan_misses`` --
host decode + traces + merge) and, per workload / per generation, the batch
deltas next to the evoxc ones.  That route compiles ONE merged graph per
population structure multiset, so a multi-second ``first_call_ms`` and a large
``compile_misses_first_call`` are the expected cost of the route -- recorded as
measured, never retried away as a failure.

The per-stage split (``stages.py``: decode, dedup, trace, merge, lower, compile,
load, upload, run, gather, plus the compile pool's wall time and job count) is a
SECOND optional contract on the same module.  ``batch_cache.stage_before_workloads``
/ ``stage_after`` / ``stage_keys`` / ``stage_totals`` window the whole config (what
the merged-graph compile and the warm calls spent their time on) and every
``gp_step`` step carries its own ``stage_ms`` delta.  It is getattr-probed like the
cache counters: a tree without ``batch_engine.batch_stage_stats()`` records
``available: false`` with EVERY stage counter ``null`` plus a note -- never a
fabricated ``0``, because "this stage took no time" and "this stage was not
measured" must never look alike.  ``batch_plan_summary`` also keeps
``describe_plan``'s ``compile_workers`` / ``chunk_count`` / ``chunk_sizes`` (null
when the report predates them): the compiled-graph pool width and how the merge was
split for it.

The content-addressed plan-key table (``content.py``: ``batch_content_hits`` /
``_misses`` / ``_rejects`` / ``_entries`` / ``_maxsize``) is a THIRD optional contract
on the same module: ``batch_cache.content_before_workloads`` / ``content_after`` /
``content_keys`` / ``content_totals`` window exactly the span the stage split does, and
they price the surface a REBUILT-but-equal population hits (an unchanged ``Forest``
object hits the identity tables and leaves the content table untouched).  It is
getattr-probed the same way: a tree without ``batch_engine.batch_content_stats()``
records ``content_totals`` with ``available: false`` and EVERY counter ``null`` plus a
note -- never a fabricated ``0``.

The persistent on-disk executable tier (``artifact.py``: ``batch_artifact_enabled`` /
``_files`` / ``_hits`` / ``_misses`` / ``_load_failures`` / ``_stores`` /
``_save_failures``, switched by ``EVOGP_EVOXC_ARTIFACT_DIR``) is a FOURTH optional
contract on the same module, windowed at the same two points into
``batch_cache.artifact_before_workloads`` / ``artifact_after`` / ``artifact_keys`` /
``artifact_totals``.  It is the one counter surface that can turn the cold call into a
reload -- a disk hit skips ``lower`` / ``compile`` / ``load`` -- so a record whose
``artifact_totals.available`` is true and whose ``batch_artifact_enabled`` is 0 was
measured with the tier OFF, which is a different statement from ``available: false``
(the tree has no tier at all).  ``batch_cache.artifact_dir`` states the directory the
arm actually pointed ``EVOGP_EVOXC_ARTIFACT_DIR`` at.

Two keys answer a question a timing alone cannot. ``batch_plans`` /
``batch_plan_summary`` (batch arms) are what the route actually merged -- the row
dedupe, the band shape and the emitted op count, read off ``BATCH_PLAN_MEMO`` via
``batch_engine.describe_plan``, which compiles nothing. ``dense_route`` (explicit
``EVOGP_EVAL_MODE=dense`` arms only) is the existence proof for the dense route on
that cell: the trace cap ``EVOGP_DENSE_MAX_TREE_LEN`` makes the public API fall
back to per-tree ``specialized`` silently, so the record states whether
``sr_fitness_dense`` served the population (``supported``) or declined it
(``reason``), probed after all timings so it can never re-compile.

Usage (the driver sets ``PYTHONPATH``; run it from the repository root)::

    python test/benchmark/evoxc/worker.py \
        --cell p256-L64-i32-N1024 --arm evoxc-dense-iree \
        --out test/benchmark/evoxc/results/w1/p256-L64-i32-N1024/evoxc-dense-iree.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402
import arms as arms_mod  # noqa: E402
import stages  # noqa: E402  (torch-free, evogp-free: the stage vocabulary + derived blocks)
import content  # noqa: E402  (torch-free, evogp-free: the content-key vocabulary + derived blocks)
import artifact  # noqa: E402  (torch-free, evogp-free: the disk-tier vocabulary + derived blocks)
import counter_blocks  # noqa: E402  (torch-free, evogp-free: the strict number/block helpers)


class TimedFitness:
    """``fitness(forest)`` wrapper that records the cost of the call it wraps.

    The generation step is timed end-to-end AND its evaluation sub-call is timed
    in place (same population, same step), so the operator part is a real
    difference and not a cross-population comparison.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last_ms = 0.0
        self.last_setup_ms = 0.0

    def __call__(self, forest, report: Optional[Dict[str, Any]] = None):
        import torch

        scratch: Dict[str, Any] = {}
        start = time.perf_counter()
        loss = self.inner(forest, scratch)
        torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - start) * 1e3
        self.last_setup_ms = float(scratch.get("setup_ms", 0.0))
        if report is not None:
            report.update(scratch)
            report["eval_ms"] = self.last_ms
        return loss


def _stat_value(stats: Any, key: str) -> Optional[int]:
    """One int counter out of an OPTIONAL stats dict (``None`` when unavailable).

    The batch engine owns its counter contract, so this is deliberately tolerant:
    an absent key reads as ``0`` instead of raising, because a benchmark record must
    never be lost to a counter the route renamed or has not implemented yet.
    """
    if not isinstance(stats, dict):
        return None
    try:
        return int(stats.get(key, 0) or 0)
    except (TypeError, ValueError):
        return None


def _stats_delta(before: Any, after: Any, key: str) -> Optional[int]:
    """``after[key] - before[key]`` across two optional snapshots (``None`` if unknown)."""
    start, end = _stat_value(before, key), _stat_value(after, key)
    if start is None or end is None:
        return None
    return end - start


def _has_stat_key(before: Any, after: Any, key: str) -> bool:
    """Does at least one of the two snapshots actually carry ``key``?"""
    return any(isinstance(stats, dict) and key in stats for stats in (before, after))


def _optional_policy_value(module: Any, name: str) -> Optional[int]:
    """``module.<name>()`` as an int, or ``None`` when this tree does not expose it.

    A policy accessor the tree under measurement predates (``resolve_auto_budget_mb``
    before ``auto``'s rows rule became a memory envelope) is a REPORTED state: the
    record reads ``null``, never a fabricated default that the engine did not apply.
    """
    getter = getattr(module, name, None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:  # noqa: BLE001 - a broken accessor must not cost the record
        return None


#: The counter names the batch route's compile misses can live under, in preference
#: order (single-sourced in :mod:`common`).  ``batch_engine.batch_cache_stats()``
#: (eight keys) names its two front tables explicitly: ``batch_exec_*`` describes
#: ``BATCH_EXEC_MEMO`` -- the real compile cache, since a hit never reaches
#: ``evoxc.lower`` / ``compile`` / ``load`` -- and ``batch_plan_*`` describes
#: ``BATCH_PLAN_MEMO`` (host decode + traces + merge).  A bare ``misses`` is the
#: older spelling this harness used to read; it is a fallback, never the primary.
BATCH_EXEC_MISS_KEYS: Tuple[str, ...] = common.BATCH_EXEC_MISS_KEYS
#: The plan-table counter: a plan miss re-traces and re-merges, it does not recompile.
BATCH_PLAN_MISS_KEYS: Tuple[str, ...] = common.BATCH_PLAN_MISS_KEYS


def _snapshot_key(snapshot: Any, candidates: Tuple[str, ...]) -> Optional[str]:
    """The first of ``candidates`` this snapshot carries (``None`` when it carries none)."""
    if not isinstance(snapshot, dict):
        return None
    for key in candidates:
        if key in snapshot:
            return key
    return None


def batch_miss_key(snapshot: Any) -> Optional[str]:
    """Which counter holds the batch route's real compile misses in ``snapshot``.

    ``batch_exec_misses`` on the engine route we measure (``BATCH_EXEC_MEMO``: a hit
    never reaches ``evoxc.lower`` / ``compile`` / ``load``), the older bare ``misses``
    when that is all a snapshot carries, and ``None`` when the snapshot has neither --
    an unknown counter must read as null, never as "no compile".
    """
    return _snapshot_key(snapshot, BATCH_EXEC_MISS_KEYS)


def batch_plan_miss_key(snapshot: Any) -> Optional[str]:
    """Which counter holds the batch route's plan misses in ``snapshot`` (``None`` if none)."""
    return _snapshot_key(snapshot, BATCH_PLAN_MISS_KEYS)


def _stats_delta_for(before: Any, after: Any, resolver) -> Optional[int]:
    """Delta of the counter ``resolver`` names on either snapshot (``None`` if unknown).

    Strict on purpose: a snapshot that does not carry the resolved key reads as
    UNKNOWN, never as ``0``, so a null delta always means "this counter was not
    available", which is exactly what ``check.py`` and the renderer need to see.
    """
    key = resolver(before) or resolver(after)
    if key is None:
        return None
    for snapshot in (before, after):
        if not (isinstance(snapshot, dict) and key in snapshot):
            return None
    return _stats_delta(before, after, key)


def batch_miss_delta(before: Any, after: Any) -> Optional[int]:
    """``batch_exec_misses`` delta across two batch-cache snapshots (``None`` if unknown)."""
    return _stats_delta_for(before, after, batch_miss_key)


def batch_plan_miss_delta(before: Any, after: Any) -> Optional[int]:
    """``batch_plan_misses`` delta across two batch-cache snapshots (``None`` if unknown)."""
    return _stats_delta_for(before, after, batch_plan_miss_key)


def stats_keys(*snapshots: Any) -> List[str]:
    """Sorted counter names of the FIRST snapshot that has any (``[]`` when none does).

    A record must state which counters the route actually exposes: that is what makes
    a null delta checkable ("the key was absent") instead of ambiguous.
    """
    for snapshot in snapshots:
        if isinstance(snapshot, dict) and snapshot:
            return sorted(snapshot)
    return []


#: The fused route's 15 counters (``fused_engine.fused_cache_stats()``); six are GAUGES.
FUSED_STATS_KEYS: Tuple[str, ...] = (
    "fused_calls", "fused_declines", "fused_xt_hits", "fused_xt_misses",
    "fused_xt_entries", "fused_xt_maxsize", "fused_kernel_launches",
    "fused_kernel_compiles", "fused_scratch_allocs", "fused_scratch_reuses",
    "fused_scratch_evictions", "fused_scratch_entries", "fused_scratch_bytes",
    "fused_block_rows", "fused_num_warps",
)
#: The ones that are GAUGES (an AFTER-state), never a delta.
FUSED_GAUGE_KEYS: Tuple[str, ...] = (
    "fused_xt_entries", "fused_xt_maxsize", "fused_scratch_entries",
    "fused_scratch_bytes", "fused_block_rows", "fused_num_warps",
)
#: The accessor the fused counters come from (recorded in every fused block).
FUSED_STATS_FN_PATH = "evogp.tree.fused_engine.fused_cache_stats"
#: The note a fused block carries when the accessor or its window is unavailable: the
#: counters are then null (NEVER zero), because "the kernel was not launched" and "the
#: route was never asked" must not look alike.
FUSED_ABSENT_NOTE = (
    f"the fused route exposes no {FUSED_STATS_FN_PATH}(), so every fused_* counter is "
    "null (NOT zero): this record cannot say whether a Triton launch served the call"
)


def fused_window_block(before: Any, after: Any) -> Dict[str, Any]:
    """The fused route's before -> after counter window (``FUSED_STATS_KEYS``).

    Nine counters are RUNNING TOTALS, so they are after-minus-before
    (``counter_blocks.delta_int``); the six :data:`FUSED_GAUGE_KEYS` describe the route's
    STATE and are read off the AFTER snapshot.  A half-read or absent window reads
    ``available: false`` with every counter ``None`` -- never a fabricated ``0``, which
    is why a fused record can be told apart from one whose counters were not measured.
    """
    if not (counter_blocks.has_any(before, FUSED_STATS_KEYS)
            and counter_blocks.has_any(after, FUSED_STATS_KEYS)):
        note = (
            "fused window unavailable: both the before and the after snapshot must be "
            "dicts carrying fused_* counters -- a half-read window is null (NOT a "
            "partial delta)"
        )
        return counter_blocks.block(
            False, FUSED_STATS_FN_PATH, note, FUSED_STATS_KEYS, {}
        )
    values: Dict[str, Any] = {}
    for key in FUSED_STATS_KEYS:
        if key in FUSED_GAUGE_KEYS:
            values[key] = counter_blocks.integer((after or {}).get(key))
        else:
            values[key] = counter_blocks.delta_int(before, after, key)
    return counter_blocks.block(True, FUSED_STATS_FN_PATH, None, FUSED_STATS_KEYS, values)


def fused_served(block: Dict[str, Any], calls: Optional[int]) -> Optional[bool]:
    """Did ``calls`` calls reach a Triton launch?  ``fused_calls`` counts ATTEMPTS.

    The fused facade is silent on a decline and never compiles, so
    ``cache.compile_misses_first_call`` is ``0`` by construction and proves nothing: the
    only existence proof is that every attempt of the window actually launched a kernel
    (``fused_calls - fused_declines == calls`` with at least that many
    ``fused_kernel_launches``).  ``None`` when the counters or the call count are
    unknown, so "not proven" never reads as "served" or as "declined".
    """
    if not isinstance(block, dict):
        return None
    attempts = block.get("fused_calls")
    declines = block.get("fused_declines")
    launches = block.get("fused_kernel_launches")
    if None in (attempts, declines, launches) or not calls:
        return None
    return int(attempts) - int(declines) == int(calls) and int(launches) >= int(calls)


def stage_window_block(before: Any, after: Any) -> Dict[str, Any]:
    """One before -> after STAGE window: ``stages.delta_block`` plus its compact stage table.

    The block carries ``available``/``stats_fn``/``note`` and every counter key (per-stage
    ``calls``/``total_ms``/``max_ms``, ``batch_compile_wall_ms``, ``batch_compile_jobs``), then
    ``stage_ms`` -- ``{stage: total_ms}`` off ``stages.stage_totals`` -- so the split is readable
    without re-walking the flat keys.  ``max_ms`` is the AFTER snapshot's PROCESS-LIFETIME maximum
    (a max is not delta-able); ``calls``/``total_ms`` and the two ``batch_compile_*`` counters are
    after-minus-before.  An absent accessor or a half-read window reads as ``available: false``
    with every counter ``None`` plus a note -- never a ``0``.
    """
    block = stages.delta_block(before, after)
    block["stage_ms"] = stages.stage_totals(block)
    return block


class BatchStatsProbe:
    """Sample the batch cache the moment ``time_call``'s untimed first call returns.

    ``common.time_call`` only exposes one post-sequence sample point (it samples the
    ``stats`` callable right after the cold call), and the batch route's compile
    lands entirely in that first call -- so wrapping the callable is the only way to
    report ``compile_misses_first_call`` separately from the warmup+reps total.  When
    the route has no stats callable the snapshot stays ``None``; every other arm is
    handed the arm's own bound method, i.e. unchanged behaviour.
    """

    def __init__(self, fn, stats):
        self.fn = fn
        self.stats = stats
        self.first: Optional[Dict[str, Any]] = None

    def __call__(self):
        out = self.fn()
        if self.first is None and self.stats is not None:
            self.first = self.stats()
        return out


def forest_stats(forest) -> Dict[str, Any]:
    """Structure context the compile counts are read against.

    ``distinct_structures`` / ``structure_signature`` come from
    ``common.population_signature`` -- the structure-multiset fingerprint the batch
    route's merged-graph cache is keyed on -- so a record states the compile story in
    the cache's own terms instead of maintaining a second, subtly different hash.
    """
    import numpy as np

    signature = common.population_signature(forest)
    sizes = forest.batch_subtree_size[:, 0].detach().to("cpu").numpy().astype(np.int64)
    return {
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "tree_len_mean": float(sizes.mean()),
        "tree_len_median": float(np.median(sizes)),
        "tree_len_max": int(sizes.max()),
        "distinct_structures": int(signature["distinct_structures"]),
        "structure_signature": signature["signature"],
    }


def dense_trace_cap(dense_engine) -> int:
    """The dense route's trace cap ``L`` as the engine would resolve it right now.

    ``dense_engine._resolve_max_tree_len`` is private, so this reads the public pair
    instead: the env var ``dense_engine.MAX_TREE_LEN_ENV`` when it is set and the
    module's own ``DEFAULT_MAX_TREE_LEN`` when it is not.  Read once, early, before
    any timing exists: an unparseable value is a wiring bug and must fail the config
    there rather than after the numbers have been paid for.
    """
    env_name = getattr(dense_engine, "MAX_TREE_LEN_ENV", "EVOGP_DENSE_MAX_TREE_LEN")
    default = int(getattr(dense_engine, "DEFAULT_MAX_TREE_LEN", 64))
    raw = os.environ.get(env_name, "").strip()
    return int(raw) if raw else default


def dense_route_probe(
    dense_engine,
    workload: Dict[str, Any],
    cell: Dict[str, Any],
    tree_len_max: int,
    cap: int,
) -> Dict[str, Any]:
    """Did the dense route actually SERVE this cell, or did its trace cap decline it?

    Existence proof for an explicit-dense arm: ``Forest.SR_fitness`` falls back to the
    per-tree ``specialized`` route silently whenever ``dense_engine`` raises
    ``DenseUnsupported``, so a warm median alone cannot tell "dense served this
    population" from "dense declined and specialized did the work".  This calls
    ``sr_fitness_dense`` directly and reports the outcome next to the numbers.

    It runs AFTER every timed workload on purpose: on a cell the workloads already
    evaluated through dense this is a memo hit (same population, same data, same
    shape -- no host crossing, no compile), and when the cap declines the population
    ``prepare_dense`` raises before any host work, so the probe can neither re-compile
    nor distort a number that is already in the record.
    """
    import torch

    forest = workload["forest"]
    entry: Dict[str, Any] = {
        "max_tree_len_cap": int(cap),
        "tree_len_max": int(tree_len_max),
        "supported": False,
        "reason": None,
        "probe_ms": None,
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        dense_engine.sr_fitness_dense(
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
            int(cell["input_len"]),
            # the sr_fitness/batch_forward label width: this cell's labels are (N, out)
            int(workload["labels"].shape[-1]),
            workload["x"],
            workload["labels"],
        )
    except dense_engine.DenseUnsupported as exc:
        # the one decline that matters here: the cap is what priced the fallback
        entry["reason"] = str(exc.reason)
    except Exception as exc:  # noqa: BLE001 - a probe must never lose the record
        entry["reason"] = repr(exc)
    else:
        entry["supported"] = True
    torch.cuda.synchronize()
    entry["probe_ms"] = round((time.perf_counter() - start) * 1e3, 4)
    return entry


#: The merge-shape fields ``stages.plan_fields`` owns: the compiled-graph pool width
#: (``EVOGP_BATCH_COMPILE_WORKERS`` as the route resolved it) and how the merge was split
#: for it.  A ``describe_plan`` report that predates them reads ``None`` (NOT ``0``).
PLAN_MERGE_SHAPE_KEYS: Tuple[str, ...] = ("compile_workers", "chunk_count", "chunk_sizes")
#: The ``batch_engine.describe_plan`` fields a ``batch_plan_summary`` keeps per variant:
#: the merge shape, the row dedupe and the graph I/O, i.e. everything that explains
#: the merged-graph compile (the per-spec shapes/classes are in ``batch_plans``).
BATCH_PLAN_SUMMARY_KEYS: Tuple[str, ...] = (
    "emitted_ops",
    "band_count",
    "band_sizes",
    "band_strategy",
    "n_functions",
    "functions",
    "row_index_collisions",
    "pop_size",
    "input_len",
    "output_len",
    "per_row",
    "backend",
    "device",
) + PLAN_MERGE_SHAPE_KEYS


def batch_plan_facts(batch_engine_module) -> Dict[str, Any]:
    """The merged-graph PLANS a batch config had, read off ``BATCH_PLAN_MEMO``.

    ``describe_plan`` reads the stored report only -- nothing is decoded, traced or
    compiled -- so this is a compile-free window on what the route actually merged:
    how many programs survived the row dedupe (``functions`` vs ``pop_size``), the
    band shape (``band_count``/``band_sizes``/``band_strategy``) and the emitted op
    count, which is what a long-tree population spends its compile on.

    Never raises: a tree without ``BATCH_PLAN_MEMO`` / ``describe_plan``, an absent
    entry attribute or a memo that refuses iteration is a reportable state
    (``available: False`` + a note), not a crash -- a missing counter must never cost
    a whole measured record.
    """
    memo = getattr(batch_engine_module, "BATCH_PLAN_MEMO", None)
    describe = getattr(batch_engine_module, "describe_plan", None)
    facts: Dict[str, Any] = {
        "available": False, "note": None, "plans": [], "summary": {},
        # the BatchPlan objects, PARALLEL to ``plans``: ``common.band_report`` reads the
        # two cost-model estimates off ``plan.report``, which ``describe_plan`` does not
        # expose.  Never serialized (a plan holds graphs), only introspected.
        "objects": [],
    }
    if memo is None or not callable(describe):
        absent = [
            name
            for name, value in (("BATCH_PLAN_MEMO", memo), ("describe_plan", describe))
            if value is None
        ]
        facts["note"] = (
            "evogp.tree.batch_engine exposes no "
            + " and no ".join(absent or ["plan introspection"])
            + ", so the merged-graph plan facts are unavailable (NOT empty)"
        )
        return facts
    try:
        entries = list(memo.values())
    except Exception as exc:  # noqa: BLE001 - optional introspection only
        facts["note"] = f"BATCH_PLAN_MEMO could not be read: {exc!r}"
        return facts
    seen = set()
    for entry in entries:
        plan = getattr(entry, "plan", None)
        if plan is None:
            continue
        try:
            described = describe(plan)
        except Exception as exc:  # noqa: BLE001 - optional introspection only
            facts["note"] = f"describe_plan failed for one entry: {exc!r}"
            continue
        key = (described.get("variant"), bool(described.get("per_row")))
        if key in seen:  # one plan per (variant, per_row): the rest are re-bound copies
            continue
        seen.add(key)
        facts["plans"].append(described)
        facts["objects"].append(plan)
    counts: Dict[str, int] = {}
    for plan in facts["plans"]:
        counts[str(plan.get("variant"))] = counts.get(str(plan.get("variant")), 0) + 1
    for index, plan in enumerate(facts["plans"]):
        variant = str(plan.get("variant"))
        # ``value`` is planned twice in a normal config (batch_forward per_row=False,
        # forward per_row=True), so disambiguate only when the variant is not unique
        key = variant if counts[variant] == 1 else f"{variant}[per_row={int(bool(plan.get('per_row')))}]"
        entry = {name: plan.get(name) for name in BATCH_PLAN_SUMMARY_KEYS}
        # the merge-shape fields have ONE resolution (``stages.plan_fields``); a missing one is
        # ``None`` here, never 0 -- the pool width is part of what the compile cost is priced against
        entry.update(stages.plan_fields(plan))
        # the band/merge shape plus the static cost model's own verdict (``common.band_report``):
        # ``describe_plan`` does not expose ``bands`` / ``cost_estimate`` / ``flat_cost_estimate``,
        # so the plan's own report is read for them and they stay null (never 0) unless
        # ``EVOGP_BATCH_BANDS=auto`` ran the model.  Additive: every key above is unchanged.
        entry["band_report"] = common.band_report(facts["objects"][index], plan)
        facts["summary"][key] = entry
    facts["available"] = bool(facts["plans"])
    if not facts["available"] and facts["note"] is None:
        facts["note"] = (
            "BATCH_PLAN_MEMO holds no entry carrying a ``plan`` -- the batch route "
            "never planned a population in this process"
        )
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="cell id, e.g. p256-L64-i32-N1024")
    parser.add_argument("--arm", required=True, help=f"one of {common.ALL_ARMS}")
    parser.add_argument("--out", required=True, help="JSON record path")
    parser.add_argument("--tree-root", default="src", help="tree root whose src/ holds evogp")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gp-warmup", type=int, default=1)
    parser.add_argument("--gp-reps", type=int, default=5)
    parser.add_argument(
        "--workloads", default="sr_fitness,batch_forward,forward,gp_step"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", default=None, help="free-form note stored in the record")
    args = parser.parse_args()

    # the --tree-root switch: import evogp from <tree_root>/src (a git-archive
    # snapshot times exactly like this checkout) instead of PYTHONPATH order
    root = os.path.abspath(args.tree_root)
    src = os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root
    if src not in [os.path.abspath(entry) for entry in sys.path]:
        sys.path.insert(0, src)

    cell = common.parse_cell(args.cell)
    # the cell dict resolves a templated knob (``EVOGP_DENSE_MAX_TREE_LEN={tree_len}``);
    # arms with purely literal knobs ignore it
    knobs = arms_mod.apply_arm_env(args.arm, cell)

    # -- batch support preflight ---------------------------------------------
    # EVOGP_EVAL_MODE=batch is only meaningful when the tree under test ships the
    # module AND its policy declares the mode.  A missing route must fail here, with
    # one readable paragraph, instead of writing a record that silently measured some
    # other route (or dying deep inside the tree layer): the driver logs
    # ``status: failed`` plus this message, and no record exists for the cell.
    is_batch = common.is_batch_arm(args.arm)
    batch_policy = common.batch_policy(src) if is_batch else None
    if is_batch and not (batch_policy["module_present"] and batch_policy["mode_supported"]):
        raise SystemExit(
            f"arm {args.arm}: EVOGP_EVAL_MODE=batch needs BOTH a "
            f"<src>/evogp/tree/batch_engine.py module and 'batch' declared in "
            f"evogp.tree.dense_engine.EVAL_MODES, and this tree does not provide the "
            f"pair -- found module_present={batch_policy['module_present']} "
            f"(expected {batch_policy['module']}), "
            f"declared_modes={batch_policy['declared_modes']!r} "
            f"(mode_supported={batch_policy['mode_supported']}) for src={src}. "
            f"NO record was written for cell {args.cell} arm {args.arm}: the batch "
            f"route cannot run here, and a record of a different route would be worse "
            f"than a missing one."
        )

    common.import_evogp()

    # -- the batch route's own cache counters (optional, batch arms only) -----
    # The merged-graph cache may or may not be evoxc_engine's, so ask the batch
    # engine itself.  Resolved only for batch arms: importing that module can then
    # never break a torch/evoxc arm, and a missing ``batch_cache_stats`` is a
    # reportable state (recorded as null), not a crash.
    batch_stats = None
    batch_engine_module = None
    stage_stats = None
    stage_reset = None
    content_stats = None
    artifact_stats = None
    if is_batch:
        try:
            from evogp.tree import batch_engine as _batch_engine_probe
        except ImportError:  # noqa: F401 - the route is optional on older trees
            _batch_engine_probe = None
        batch_engine_module = _batch_engine_probe
        batch_stats = getattr(_batch_engine_probe, "batch_cache_stats", None)
        if not callable(batch_stats):
            batch_stats = None
        # the per-stage split is a SECOND optional contract on the same module: resolved here
        # (getattr-probed, ``None`` when absent) so every block below reads the same accessors
        stage_stats = stages.stage_stats_fn(_batch_engine_probe)
        stage_reset = stages.stage_reset_fn(_batch_engine_probe)
        # the content-addressed plan-key table is a THIRD optional contract on the same module:
        # a rebuilt-but-equal population's reuse surface, resolved the same getattr-probed way
        content_stats = content.stats_fn(_batch_engine_probe)
        # the persistent on-disk executable tier is a FOURTH: a disk hit skips
        # lower/compile/load, so it is the one counter that can move a cold column
        artifact_stats = artifact.stats_fn(_batch_engine_probe)

    # -- the fused (Triton) route's own cache counters (optional, fused arms only) ----
    # The facade is silent on a decline and never compiles, so
    # ``cache.compile_misses_first_call`` is 0 by construction and proves NOTHING here:
    # ``fused_kernel_launches`` is the existence proof, and it lives on the route's own
    # 15-counter surface.  Resolved only for a fused arm (getattr-probed, ``None`` when
    # absent), so importing that module can never break a torch / evoxc / batch arm and a
    # tree without the route records nulls plus a note instead of a fabricated 0.
    is_fused = (common.EVOXC_ARMS.get(args.arm) or {}).get("eval_mode") == "fused"
    fused_stats = None
    if is_fused:
        try:
            from evogp.tree import fused_engine as _fused_engine_probe
        except ImportError:  # noqa: F401 - the route is optional on older trees
            _fused_engine_probe = None
        fused_stats = counter_blocks.accessor(_fused_engine_probe, "fused_cache_stats")

    import torch
    from evogp.tree import dense_engine, evoxc_engine

    # the dense trace cap as the engine resolves it for THIS process (the arm's spec
    # already wrote ``EVOGP_DENSE_MAX_TREE_LEN``): read before the timings so a broken
    # value fails the config here instead of after the numbers were paid for
    dense_cap = dense_trace_cap(dense_engine)

    workload = common.build_workload(cell, args.seed)
    forest = workload["forest"]
    structure = forest_stats(forest)

    record: Dict[str, Any] = {
        "env": common.env_block(
            args.tree_root,
            args.arm,
            extra={"applied_knobs": knobs, "cell": cell, "arm_spec": common.EVOXC_ARMS.get(args.arm, common.TORCH_ARMS.get(args.arm))},
        ),
        "cell": cell,
        "arm": args.arm,
        "family": "torch" if args.arm in common.TORCH_ARMS else "evoxc",
        "protocol": {
            "warmup": args.warmup,
            "reps": args.reps,
            "gp_warmup": args.gp_warmup,
            "gp_reps": args.gp_reps,
            "workloads": args.workloads.split(","),
            "seed": args.seed,
            "sync_bracketed": True,
        },
        "structures": structure,
        "routes": {},
        "workloads": {},
        "gp_step": None,
        "notes": [],
    }
    if args.note:
        record["notes"].append(args.note)

    # -- batch route provenance / merged-graph cache counters ------------------
    # These keys exist ONLY for batch arms: "not a batch route" must stay
    # distinguishable from "batch route whose counters are unavailable", so no
    # None-noise is added to the existing record schema.
    if is_batch:
        record["batch_policy"] = batch_policy
        # the stage window opens here, i.e. before ``make_arm`` and before the first workload:
        # exactly where ``stats_before_workloads`` is taken, so the two windows are the same span
        stage_before_workloads = stage_stats() if stage_stats is not None else None
        # the content window opens at exactly the same point, so it spans the same span
        content_before_workloads = content_stats() if content_stats is not None else None
        # ... and so does the disk tier's, so ``hits``/``misses`` are this config's own
        artifact_before_workloads = (
            artifact_stats() if artifact_stats is not None else None
        )
        record["batch_cache"] = {
            "available": batch_stats is not None,
            "stats_before_workloads": batch_stats() if batch_stats is not None else None,
            "stats_after": None,
            "stats_keys": [],
            "compile_misses_first_call": None,
            "compile_misses_total": None,
            "plan_misses_first_call": None,
            "plan_misses_total": None,
            # the per-stage split (``stages.py``): a second optional counter contract, so its
            # absence is recorded as ``available: false`` + null counters, never as 0
            "stage_before_workloads": stage_before_workloads,
            "stage_after": None,
            "stage_keys": [],
            "stage_totals": None,
            # the content-addressed plan-key table (``content.py``): a third optional counter
            # contract -- the reuse surface a REBUILT-but-equal population hits
            "content_before_workloads": content_before_workloads,
            "content_after": None,
            "content_keys": [],
            "content_totals": None,
            # the persistent on-disk executable tier (``artifact.py``): the FOURTH optional
            # counter contract -- a hit skips lower/compile/load, so this is the block that
            # says whether a cold call was a compile or a reload
            "artifact_before_workloads": artifact_before_workloads,
            "artifact_after": None,
            "artifact_keys": [],
            "artifact_totals": None,
            "artifact_dir": os.environ.get(common.ARTIFACT_ENV_VARS[0]),
        }
        record["notes"].append(
            "batch arm: one merged graph per population structure multiset, so a "
            "multi-second first_call_ms and a large compile_misses_first_call are the "
            "expected cold cost of the route and are recorded as measured, never retried"
        )
        if batch_stats is None:
            record["notes"].append(
                "batch arm: evogp.tree.batch_engine.batch_cache_stats() is unavailable on "
                "this tree, so every merged-graph cache counter is null (NOT zero) and "
                "the compile cost can only be read off first_call_ms"
            )
        if stage_stats is None:
            record["notes"].append(
                f"batch arm: {stages.STATS_FN_PATH}() is unavailable on this tree, so every "
                "stage counter in batch_cache.stage_totals / gp_step.steps[].stage_ms is null "
                "(NOT zero) with available=false: the split was not measured, which is not the "
                "same as any stage costing nothing -- read the cost off first_call_ms instead"
            )
        elif stage_reset is None:
            record["notes"].append(
                f"batch arm: {stages.MODULE_NAME}.{stages.RESET_FN_NAME}() is unavailable, so the "
                "stage counters are cumulative from process start: every block here is a "
                "before -> after DELTA (never a per-call block), and batch_<stage>_max_ms is a "
                "process-lifetime maximum"
            )
        if content_stats is None:
            record["notes"].append(
                f"batch arm: {content.STATS_FN_PATH}() is unavailable on this tree, so every "
                "content counter in batch_cache.content_totals is null (NOT zero) with "
                "available=false: plan-key reuse from a rebuilt-but-equal population was not "
                "measured, which is not the same as it never happening -- the reuse surface this "
                "record can show is the identity tables' (batch_exec_misses / batch_plan_misses)"
            )
        if artifact_stats is None:
            record["notes"].append(
                f"batch arm: {artifact.STATS_FN_PATH}() is unavailable on this tree, so every "
                "disk-tier counter in batch_cache.artifact_totals is null (NOT zero) with "
                "available=false: this tree has no persistent artifact tier for the merged "
                "bodies, so a cold call always compiles -- a zero here would mean 'the tier "
                "served nothing', which this record cannot claim"
            )

    # -- fused route provenance / kernel-launch counters ----------------------
    # Same rule as the batch block above: these keys exist ONLY for fused arms, so "not the
    # fused route" stays distinguishable from "fused route whose proof was not recorded".
    served_flags: List[Optional[bool]] = []
    fused_before_workloads: Optional[Dict[str, Any]] = None
    fused_after_workloads: Optional[Dict[str, Any]] = None
    if is_fused:
        # the window opens where the first workload's own window opens, so the totals below
        # are this config's workload span and never a cumulative-from-import reading
        fused_before_workloads = fused_stats() if fused_stats is not None else None
        record["notes"].append(
            "fused arm: the route compiles no evoxc graph, so "
            "cache.compile_misses_first_call is 0 by construction -- the existence proof is "
            "cache.fused_served / fused_cache.served_every_workload (fused_kernel_launches)"
        )
        if fused_stats is None:
            record["notes"].append("fused arm: " + FUSED_ABSENT_NOTE)

    # -- what the policy resolves for this cell (per workload row count) -----
    # ``tree_len`` is the population's row stride (the cell's), i.e. the shape term of
    # ``auto``'s memory envelope: passing it makes the reported rows half the one the
    # engine really applies instead of a pin-only reading.
    for name, rows in (
        ("sr_fitness", cell["rows"]),
        ("batch_forward", cell["rows"]),
        ("forward", 1),
    ):
        record["routes"][name] = common.route_decision(
            cell["pop"], rows, knobs["eval_mode"], knobs["backend"],
            tree_len=cell["tree_len"],
        )

    # -- what ``auto``'s trace-cap raise resolves for this cell --------------
    # Under ``auto`` the dense trace cap is NOT the pinned one: an over-cap population
    # is served at ``min(longest, min(EVOGP_DENSE_AUTO_MAX_TREE_LEN, envelope))``
    # instead of falling back to the per-tree ``specialized`` route, so an ``auto``
    # warm median can be dense-at-a-RASIED cap.  A record must state which cap it
    # actually ran at, per workload (the envelope's row axis differs: ``forward`` sees
    # one row per tree), or the number cannot be read against a pinned-cap dense arm.
    # A tree without the accessor records nulls plus a note, never a fabricated cap.
    auto_cap_fn = getattr(dense_engine, "auto_trace_cap", None)
    pinned_fn = getattr(dense_engine, "trace_cap_pinned", None)
    auto_cap_block: Dict[str, Any] = {
        "available": callable(auto_cap_fn),
        "auto_max_tree_len": _optional_policy_value(
            dense_engine, "resolve_auto_max_tree_len"
        ),
        "tree_len_max": int(structure["tree_len_max"]),
        "row_len": int(cell["tree_len"]),
        "cap_pinned": bool(pinned_fn()) if callable(pinned_fn) else None,
        "base_cap": int(dense_cap),
        "caps": {},
    }
    if callable(auto_cap_fn):
        for name, rows in (
            ("sr_fitness", cell["rows"]),
            ("batch_forward", cell["rows"]),
            ("forward", 1),
        ):
            entry = {}
            for mode in ("auto", "dense"):
                try:
                    entry[mode] = int(
                        auto_cap_fn(
                            structure["tree_len_max"],
                            cell["tree_len"],
                            eval_mode=mode,
                            pop_size=cell["pop"],
                            rows=rows,
                        )
                    )
                except Exception:  # noqa: BLE001 - a broken policy accessor must not cost the record
                    entry[mode] = None
            auto_cap_block["caps"][name] = entry
    else:
        auto_cap_block["note"] = (
            "evogp.tree.dense_engine.auto_trace_cap() is unavailable on this tree, so "
            "the effective trace cap under auto is null: the pinned base cap above is "
            "what BOTH modes ran at"
        )
        record["notes"].append("auto trace cap: " + auto_cap_block["note"])
    record["auto_trace_cap"] = auto_cap_block
    # -- evaluation workloads ------------------------------------------------
    arm_obj = arms_mod.make_arm(args.arm, workload)
    record["setup_ms"] = round(arm_obj.setup_ms, 4)
    if arm_obj.family == "torch":
        record["notes"].append(
            "torch arm: the tree decode / batched plan is built once, outside the "
            "timed region (setup_ms); per-call cost is torch ops only"
        )

    requested = [name for name in args.workloads.split(",") if name]
    batch_first_call_misses = None
    batch_first_call_plan_misses = None
    for name in requested:
        if name == "gp_step":
            continue
        fn = getattr(arm_obj, name)
        cache_before = evoxc_engine.cache_stats()
        batch_before = batch_stats() if batch_stats is not None else None
        fused_before = fused_stats() if fused_stats is not None else None
        # A batch arm's compile lands in time_call's untimed first call; the probe
        # snapshots the merged-graph cache the instant that call returns.  A fused arm
        # gets the same treatment for its launch counters (the kernel JIT lands in that
        # same first call).  Every other arm (and one with no stats callable) gets its
        # own bound method untouched, i.e. no extra call in the timed loop.
        probe = None
        if is_batch and batch_stats is not None:
            probe = BatchStatsProbe(fn, batch_stats)
        elif is_fused and fused_stats is not None:
            probe = BatchStatsProbe(fn, fused_stats)
        timing = common.time_call(
            probe if probe is not None else fn,
            warmup=args.warmup,
            reps=args.reps,
            stats=evoxc_engine.cache_stats,
        )
        cache_after = evoxc_engine.cache_stats()
        result = timing.pop("_out")
        after_first = timing.pop("cache_after_first_call")
        entry = common.public_timing(timing)
        for key in ("first_call_ms", "median_ms", "min_ms", "max_ms", "mean_ms"):
            entry[key] = round(entry[key], 4)
        entry["samples_ms"] = [round(v, 4) for v in entry["samples_ms"]]
        entry["cache"] = {
            "before": cache_before,
            "after_first_call": after_first,
            "after": cache_after,
            "compile_misses_this_workload": cache_after["misses"] - cache_before["misses"],
            "compile_misses_first_call": (
                after_first["misses"] - cache_before["misses"] if after_first else None
            ),
        }
        if is_batch:
            # the existing keys above stay the evoxc-engine view (check.py reads them);
            # the batch counters sit next to it so the two caches are never conflated
            batch_after_first = probe.first if probe is not None else None
            batch_after = batch_stats() if batch_stats is not None else None
            entry["cache"]["batch_after_first_call"] = batch_after_first
            entry["cache"]["batch_after"] = batch_after
            entry["cache"]["batch_compile_misses_first_call"] = batch_miss_delta(
                batch_before, batch_after_first
            )
            entry["cache"]["batch_compile_misses_this_workload"] = batch_miss_delta(
                batch_before, batch_after
            )
            entry["cache"]["batch_plan_misses_first_call"] = batch_plan_miss_delta(
                batch_before, batch_after_first
            )
            entry["cache"]["batch_plan_misses_this_workload"] = batch_plan_miss_delta(
                batch_before, batch_after
            )
            if batch_first_call_misses is None:
                batch_first_call_misses = entry["cache"]["batch_compile_misses_first_call"]
            if batch_first_call_plan_misses is None:
                batch_first_call_plan_misses = entry["cache"]["batch_plan_misses_first_call"]
        if is_fused:
            # the route compiles nothing, so the PROOF is that this workload's calls all
            # reached a Triton launch: attempts - declines must equal the calls ``time_call``
            # made (one cold + warmup + reps), and the launches must cover them
            fused_after_first = probe.first if probe is not None else None
            fused_after = fused_stats() if fused_stats is not None else None
            entry["cache"]["fused"] = fused_window_block(fused_before, fused_after)
            entry["cache"]["fused_first_call"] = fused_window_block(
                fused_before, fused_after_first
            )
            entry["cache"]["fused_after_first_call"] = fused_after_first
            entry["cache"]["fused_after"] = fused_after
            entry["cache"]["fused_calls_timed"] = 1 + int(args.warmup) + int(args.reps)
            entry["cache"]["fused_served"] = fused_served(
                entry["cache"]["fused"], entry["cache"]["fused_calls_timed"]
            )
            served_flags.append(entry["cache"]["fused_served"])
        entry["digest"] = common.digest(result)
        record["workloads"][name] = entry
        del result

    if is_fused:
        # close the fused counter window over exactly the workload span: the GP loop below
        # makes calls of its own, which the whole-config totals must never claim
        fused_after_workloads = fused_stats() if fused_stats is not None else None

    # -- one full GP generation step ----------------------------------------
    if "gp_step" in requested:
        timed = TimedFitness(arm_obj.fitness)
        pipeline = arms_mod.make_pipeline(workload, timed)
        for _ in range(args.gp_warmup):
            pipeline.step()
        torch.cuda.synchronize()
        steps = []
        for _ in range(args.gp_reps):
            cache_before = evoxc_engine.cache_stats()
            batch_before = batch_stats() if batch_stats is not None else None
            stage_step_before = stage_stats() if stage_stats is not None else None
            torch.cuda.synchronize()
            start = time.perf_counter()
            pipeline.step()
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1e3
            cache_after = evoxc_engine.cache_stats()
            batch_after = batch_stats() if batch_stats is not None else None
            stage_step_after = stage_stats() if stage_stats is not None else None
            step = {
                "total_ms": round(total_ms, 4),
                "eval_ms": round(timed.last_ms, 4),
                "ops_ms": round(total_ms - timed.last_ms, 4),
                "torch_setup_ms": round(timed.last_setup_ms, 4),
                "compile_misses": cache_after["misses"] - cache_before["misses"],
                "memo_hits": cache_after["memo_hits"] - cache_before["memo_hits"],
            }
            if is_batch:
                # a fresh population is a new structure multiset, i.e. a new merged
                # graph: this is the counter that explains a slow batch GP step.
                # ``batch_compile_misses`` is the EXEC table (a miss recompiles),
                # ``batch_plan_misses`` the PLAN table (re-decode/re-trace/re-merge).
                step["batch_compile_misses"] = batch_miss_delta(batch_before, batch_after)
                step["batch_plan_misses"] = batch_plan_miss_delta(batch_before, batch_after)
                # this generation's own per-stage split: the same before -> after bracket as the
                # two cache counters above, so a recompiling step says WHERE its compile went
                # (trace/merge vs lower/compile/load) instead of only that it happened
                step["stage_ms"] = stage_window_block(stage_step_before, stage_step_after)
                if _has_stat_key(batch_before, batch_after, "memo_hits"):
                    step["batch_memo_hits"] = _stats_delta(batch_before, batch_after, "memo_hits")
            steps.append(step)
        totals = sorted(step["total_ms"] for step in steps)
        evals = sorted(step["eval_ms"] for step in steps)
        mid = len(totals) // 2
        record["gp_step"] = {
            "warmup": args.gp_warmup,
            "reps": args.gp_reps,
            "steps": steps,
            "median_total_ms": totals[mid],
            "min_total_ms": totals[0],
            "max_total_ms": totals[-1],
            "median_eval_ms": evals[mid],
            "median_ops_ms": round(totals[mid] - evals[mid], 4),
            "compile_misses_total": sum(step["compile_misses"] for step in steps),
            "final_pop_size": int(pipeline.algorithm.forest.pop_size),
            "best_fitness": float(pipeline.best_fitness),
        }
        if is_batch:
            misses = [step["batch_compile_misses"] for step in steps]
            record["gp_step"]["batch_compile_misses_total"] = (
                None if any(value is None for value in misses) else sum(misses)
            )
            plan_misses = [step["batch_plan_misses"] for step in steps]
            record["gp_step"]["batch_plan_misses_total"] = (
                None if any(value is None for value in plan_misses) else sum(plan_misses)
            )

    if is_batch:
        # close the batch cache story for the whole config: the deltas above are per
        # workload, this is the one number a reader can price the process against
        stats_after = batch_stats() if batch_stats is not None else None
        record["batch_cache"]["stats_after"] = stats_after
        record["batch_cache"]["stats_keys"] = stats_keys(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )
        record["batch_cache"]["compile_misses_first_call"] = batch_first_call_misses
        record["batch_cache"]["compile_misses_total"] = batch_miss_delta(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )
        record["batch_cache"]["plan_misses_first_call"] = batch_first_call_plan_misses
        record["batch_cache"]["plan_misses_total"] = batch_plan_miss_delta(
            record["batch_cache"]["stats_before_workloads"], stats_after
        )
        # -- the same window for the per-stage split ---------------------------
        # Where the merged-graph compile and the warm calls actually spent their time,
        # over exactly the span the cache deltas above describe.  ``available: false``
        # + null counters when the route ships no ``batch_stage_stats()``.
        stage_after = stage_stats() if stage_stats is not None else None
        record["batch_cache"]["stage_after"] = stage_after
        record["batch_cache"]["stage_keys"] = stats_keys(
            record["batch_cache"]["stage_before_workloads"], stage_after
        )
        record["batch_cache"]["stage_totals"] = (
            stages.unavailable_block()
            if stage_stats is None
            else stage_window_block(record["batch_cache"]["stage_before_workloads"], stage_after)
        )
        # -- the same window for the content-addressed plan-key table -------------
        # ``content.window_block`` never fabricates a 0: an absent accessor (or a half-read
        # window) yields the ``available: false`` block with every counter null plus a note.
        content_after = content_stats() if content_stats is not None else None
        record["batch_cache"]["content_after"] = content_after
        record["batch_cache"]["content_keys"] = stats_keys(
            record["batch_cache"]["content_before_workloads"], content_after
        )
        record["batch_cache"]["content_totals"] = content.window_block(
            record["batch_cache"]["content_before_workloads"], content_after
        )
        # -- the same window for the persistent on-disk executable tier -----------
        # The only counter surface that can turn a cold compile into a reload: a
        # ``available: false`` + null block means the tree has no tier, while a served
        # block with ``batch_artifact_enabled`` 0 means the tier exists and was OFF for
        # this arm (``EVOGP_EVOXC_ARTIFACT_DIR`` unset -- every arm but
        # ``evoxc-batch-iree-artifact``).  The two must not look alike.
        artifact_after = artifact_stats() if artifact_stats is not None else None
        record["batch_cache"]["artifact_after"] = artifact_after
        record["batch_cache"]["artifact_keys"] = stats_keys(
            record["batch_cache"]["artifact_before_workloads"], artifact_after
        )
        record["batch_cache"]["artifact_totals"] = artifact.window_block(
            record["batch_cache"]["artifact_before_workloads"], artifact_after
        )
        # what the route actually MERGED, read off the plan memo (compile-free): the
        # row dedupe and the band shape are what a long-tree population pays its
        # merged-graph compile for.  A tree without the introspection records
        # ``available: false`` + a note instead of an empty (i.e. lying) summary.
        plan_facts = batch_plan_facts(batch_engine_module)
        record["batch_plans"] = plan_facts["plans"]
        if plan_facts["available"]:
            record["batch_plan_summary"] = plan_facts["summary"]
        else:
            record["batch_plan_summary"] = {
                "available": False,
                "note": plan_facts["note"],
                # the band/merge sub-block keeps its shape when there is no plan to read:
                # every field null with the reason, never a 0 (common.band_report)
                "band_report": common.band_report(None),
            }
            record["notes"].append(
                "batch arm: " + str(plan_facts["note"])
            )

    # -- did the dense route serve this cell, or did its trace cap decline it? ----
    # Only an explicit-dense arm can be asked this, and only after the timings: when
    # the sr_fitness workload already ran through dense the probe is a plan/executable
    # memo hit, and when the cap declines the population it raises before any host
    # work -- so it never re-compiles a plan the timings already paid for.
    if knobs["eval_mode"] == "dense" and not is_batch:
        record["dense_route"] = dense_route_probe(
            dense_engine, workload, cell, structure["tree_len_max"], dense_cap
        )
        record["notes"].append(
            "dense arm: dense_route is probed AFTER the timed workloads, so the probe "
            "is a plan/executable memo hit when the sr_fitness workload itself went "
            "through dense and raises immediately (before any host work) when the trace "
            "cap declines the population -- it never changes a number above"
        )
        if not record["dense_route"]["supported"]:
            record["notes"].append(
                "dense arm: the dense evaluator DECLINED this population, so the warm "
                "medians above are the per-tree specialized fallback, not the dense "
                "route -- see dense_route.reason"
            )

    record["engine_cache_final"] = evoxc_engine.cache_stats()
    if is_fused:
        # the route's own counter surface, beside the evoxc engine's (never folded into it):
        # ``available: false`` with null counters means this tree has no fused counters, which
        # is NOT the same as a route that never launched a kernel
        served_every = (
            None if not served_flags or any(flag is None for flag in served_flags)
            else all(served_flags)
        )
        record["fused_cache"] = {
            "available": fused_stats is not None,
            "stats_fn": FUSED_STATS_FN_PATH,
            "stats_before_workloads": fused_before_workloads,
            "stats_after": fused_after_workloads,
            "stats_keys": stats_keys(fused_before_workloads, fused_after_workloads),
            "totals": fused_window_block(fused_before_workloads, fused_after_workloads),
            "served_every_workload": served_every,
        }
        if served_every is False:
            record["notes"].append(
                "fused arm: NOT every workload reached a Triton launch "
                f"(cache.fused_served per workload: {served_flags}), so the median of a "
                "declined workload is the route's specialized+numpy FALLBACK, not a fused "
                "measurement -- read that workload's cache.fused.fused_declines"
            )
        elif served_every is None:
            record["notes"].append(
                "fused arm: the kernel-launch proof is unavailable (null counters), so this "
                "record says the fused route was ASKED for, never that it served a call"
            )
    record["dense_policy"] = {
        "auto_min_pop": dense_engine.resolve_auto_min_pop(),
        # ``0`` = unpinned: the memory envelope (``auto_budget_mb``) decides the allowance
        "auto_max_rows": dense_engine.resolve_auto_max_rows(),
        # the envelope ``auto``'s rows rule applies: pop_size * tree_len * rows * 4 bytes
        # must fit it (``None`` on a tree that predates the memory rule)
        "auto_budget_mb": _optional_policy_value(dense_engine, "resolve_auto_budget_mb"),
        # the trace cap this record's process ran with (the arm's knob, else the
        # engine default): what ``dense_route`` is measured against
        "max_tree_len_cap": int(dense_cap),
    }
    common.write_json(args.out, record)
    summary = " ".join(
        f"{name}={entry['median_ms']:.3f}ms" for name, entry in record["workloads"].items()
    )
    if record["gp_step"]:
        summary += f" gp={record['gp_step']['median_total_ms']:.3f}ms"
    print(f"[record] {args.cell} {args.arm} -> {args.out} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
