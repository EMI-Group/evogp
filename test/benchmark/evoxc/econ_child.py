"""The child-side measurement internals of ``batch_economics.py`` (torch-free at import).

One child process of the compile-economics column measures ONE ``(cell, route)``: a cold first call,
the warm ``sr_fitness`` medians, a real ``StandardPipeline.step()`` generation loop and the derived
summary.  That machinery lives here, so ``batch_economics.py`` stays the DRIVER (CLI, cell/route
resolution, env construction, guard, subprocess launching, ``economics.json``) and both files stay
readable.  Nothing in this module imports ``torch`` / ``evogp`` at import time: the caller imports
them and hands the module objects in (:class:`Measurement`), exactly as it did before the split.

**No recorded key and no formula changed by the split.**  ``first_call`` / ``sr_fitness`` /
``generations[i]`` / ``summary`` / ``notes`` are assembled here byte-for-byte as they were in the
driver, and every counter read stays getattr-probed and null-never-zero:

* :class:`CounterSurfaces` resolves the route's THREE optional counter contracts once
  (``batch_engine.batch_cache_stats``, ``batch_stage_stats``/``batch_stage_reset`` and
  ``batch_content_stats``) and owns the one tolerant delta helper for each, so a route that does not
  ship a counter records ``null`` (never a fabricated ``0``).
* :func:`generation_entry` assembles one generation row: the population's structure signature
  (sampled BEFORE the step), ``signature_seen_before``, wall/eval/ops ms, the compile misses of BOTH
  front tables (exec + plan) and ``compile_hit`` -- ``None`` the moment a counter is missing.
* :func:`generation_summary` derives ``generation_hit_rate`` / ``generations_with_compile`` /
  ``distinct_population_signatures`` / the warm-vs-cold eval medians / ``compile_ms_share`` from the
  recorded rows only; :func:`summary_block` adds the ten-stage tables (``stages.summary_block``) and
  the plan-key reuse block (``content.summary_block``) without touching those formulas.

The band/merge shape of the plan(s) that served the route is read off ``worker.batch_plan_facts``
(compile-free ``describe_plan``) and reported through ``common.band_report`` -- the same block
``band_sweep.py`` writes -- so a composition record states whether the band clamp fired and what
per-chunk band counts the banded+chunked composition produced.  A plan that cannot be described
records ``available: false`` with every field ``null`` plus a note, never ``0``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)
import content  # noqa: E402  (torch-free, evogp-free: the content-key vocabulary)
import stages  # noqa: E402  (torch-free, evogp-free: the stage vocabulary)
import worker as worker_mod  # noqa: E402  (counter deltas + plan facts; torch-free at import)

#: The record's measurement tag (the driver writes the same string into ``economics.json``).
MEASUREMENT = "batch-compile-economics"
#: The tracing switch the child records (it is applied by the driver, never here).
DISABLE_LOCATIONS_ENV = "EVOXC_DISABLE_LOCATIONS"


# --- derived statistics (pure) ---------------------------------------------
def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _median(values: Sequence[float]) -> Optional[float]:
    """Median of a possibly empty list (``None`` instead of a ZeroDivisionError)."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def compile_hit(misses: Sequence[Optional[int]]) -> Optional[bool]:
    """``True`` = nothing compiled, ``False`` = at least one recompile, ``None`` = unknown.

    ``None`` the moment one cache the route relies on has no counter: a MISSING counter must never
    read as "no compile" (0), because that number -- was the merged graph rebuilt -- is exactly what
    this script measures.
    """
    if any(value is None for value in misses):
        return None
    return sum(int(value) for value in misses) == 0


def generation_summary(
    generations: Sequence[Dict[str, Any]],
    first_call_ms: float,
    warm_median_ms: Optional[float],
    merged_graph_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The derived block, computed ONLY from the recorded per-generation rows.

    ``compile_ms_share`` is an ATTRIBUTION, not a timer: the wall time of the generations that
    recompiled divided by the loop's total wall time.  A recompiling generation also evaluates and
    mutates, so the share is the honest UPPER BOUND of "how much of the loop the compile generations
    own", never a measurement of compile alone.

    ``generation_hit_rate`` is ``hits / generations`` per the protocol, with ``generations_unknown``
    next to it so a reader sees how many generations had no counter to judge (a batch route without
    ``batch_cache_stats``).
    """
    rows = list(generations)
    count = len(rows)
    hits = [row for row in rows if row["compile_hit"] is True]
    recompiled = [row for row in rows if row["compile_hit"] is False]
    sum_total = sum(float(row["total_ms"]) for row in rows)
    summary: Dict[str, Any] = {
        "generations": count,
        "generations_with_compile": len(recompiled),
        "generations_unknown": count - len(hits) - len(recompiled),
        "generation_hit_rate": (len(hits) / count) if count else None,
        "distinct_population_signatures": len({row["population_signature"] for row in rows}),
        "median_warm_eval_ms": _round(_median([row["eval_ms"] for row in hits])),
        "median_cold_eval_ms": _round(_median([row["eval_ms"] for row in recompiled])),
        "sum_total_ms": _round(sum_total),
        "sum_eval_ms": _round(sum(float(row["eval_ms"]) for row in rows)),
        "compile_ms_share": _round(sum(float(row["total_ms"]) for row in recompiled) / sum_total)
        if sum_total else None,
        "first_call_ms": _round(first_call_ms),
        "warm_median_ms": _round(warm_median_ms),
    }
    # Absent (not null) on the dense route: it owns no merged-graph cache, and a null would invite
    # reading it as "the cache had no misses".
    if merged_graph_cache is not None:
        summary["merged_graph_cache"] = merged_graph_cache
    return summary


# --- the route's optional counter surfaces ---------------------------------
class CounterSurfaces:
    """The route's THREE optional counter accessors, resolved ONCE per child process.

    Each is getattr-probed on ``evogp.tree.batch_engine`` (the module that owns all three
    contracts) and every helper below is tolerant: an absent accessor, an absent key or a
    half-read window reads ``None``, never ``0`` -- "this counter was not measured" must never
    look like "this counter did not move".  The dense column owns none of them (``is_batch``
    False), so its ``batch_*`` keys stay ABSENT rather than null.
    """

    def __init__(self, batch_module: Any = None, is_batch: bool = False) -> None:
        candidate = getattr(batch_module, "batch_cache_stats", None) if is_batch else None
        self.batch_stats = candidate if callable(candidate) else None
        self.stage_stats = stages.stage_stats_fn(batch_module) if is_batch else None
        self.stage_reset = stages.stage_reset_fn(batch_module) if is_batch else None
        self.content_stats = content.stats_fn(batch_module) if is_batch else None

    def batch(self) -> Optional[Dict[str, Any]]:
        """One merged-graph cache snapshot (``None`` when the route exposes no callable)."""
        return self.batch_stats() if self.batch_stats is not None else None

    def stage(self) -> Optional[Dict[str, Any]]:
        """One stage snapshot (``None`` when the route exposes no ``batch_stage_stats``)."""
        return self.stage_stats() if self.stage_stats is not None else None

    def content(self) -> Optional[Dict[str, Any]]:
        """One content snapshot (``None`` when the route exposes no ``batch_content_stats``)."""
        return self.content_stats() if self.content_stats is not None else None

    @staticmethod
    def misses(before: Any, after: Any) -> Optional[int]:
        # worker.py already owns the tolerant counter delta: an absent/renamed key reads as unknown
        # (None), never as 0.
        return worker_mod._stats_delta(before, after, "misses")

    @staticmethod
    def batch_misses(before: Any, after: Any) -> Optional[int]:
        # The batch route's compile counter is ``batch_exec_misses`` on the engine we measure (a hit
        # never recompiles); ``worker.batch_miss_key`` is the ONE resolution of that name, and an
        # unknown counter stays None so ``compile_hit`` can never read "no compile" from a missing key.
        return worker_mod.batch_miss_delta(before, after)

    @staticmethod
    def plan_misses(before: Any, after: Any) -> Optional[int]:
        # The plan table's misses: a re-decode/re-trace/re-merge, never a recompile.
        return worker_mod.batch_plan_miss_delta(before, after)


# --- per-generation record assembly ----------------------------------------
def generation_entry(
    index: int,
    signature: Dict[str, Any],
    seen_before: bool,
    total_ms: float,
    eval_ms: float,
    before: Dict[str, Any],
    after: Dict[str, Any],
    surfaces: CounterSurfaces,
    is_batch: bool,
    best_fitness: float = 0.0,
) -> Dict[str, Any]:
    """One generation row, assembled from that step's own before -> after snapshots.

    ``before``/``after`` carry the four snapshot pairs the step was bracketed with
    (``evoxc``, ``batch``, ``stage``, ``content``); every delta below is computed from them, so a
    reader can re-derive any number in the row.  The batch-only keys are ABSENT on the dense route.
    """
    entry: Dict[str, Any] = {
        "g": index, "population_signature": signature["signature"],
        "distinct_structures": signature["distinct_structures"],
        "signature_seen_before": seen_before, "total_ms": _round(total_ms),
        "eval_ms": _round(eval_ms), "ops_ms": _round(total_ms - eval_ms),
        "compile_misses": surfaces.misses(before["evoxc"], after["evoxc"]),
        "pop_size": signature["pop_size"], "best_fitness": float(best_fitness),
    }
    route_misses: List[Optional[int]] = [entry["compile_misses"]]
    if is_batch:
        entry["batch_compile_misses"] = surfaces.batch_misses(before["batch"], after["batch"])
        route_misses.append(entry["batch_compile_misses"])
        # the PLAN table's misses for this generation: a re-decode/re-trace/re-merge, never a
        # recompile -- recorded next to the exec table's misses instead of only in total
        entry["batch_plan_misses"] = surfaces.plan_misses(before["batch"], after["batch"])
        # this generation's stage deltas, including batch_compile_wall_ms / batch_compile_jobs
        entry["stage_ms"] = stages.delta_block(before["stage"], after["stage"])
        # and this generation's plan-key reuse: hit (an already merged plan served), miss
        # (consulted, re-merged) or no_attempt (the identity fast path answered)
        entry["content"] = content.window_block(before["content"], after["content"])
    entry["compile_hit"] = compile_hit(route_misses)
    return entry


def merged_cache_block(
    surfaces: CounterSurfaces,
    before: Any,
    after_first: Any,
    final: Any,
) -> Dict[str, Any]:
    """The merged-graph cache block the summary carries (batch routes only)."""
    return {
        "available": surfaces.batch_stats is not None,
        "provenance": {
            "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
            "key": "sorted structure-multiset fingerprint "
                   "(common.population_signature.signature)",
            "counter": "the exec table's misses (worker.batch_miss_key: "
                       "'batch_exec_misses', falling back to a bare 'misses'), i.e. the "
                       "table whose miss really recompiles; the plan table's "
                       "'batch_plan_misses' is a re-decode/re-merge, not a recompile",
            "stats_keys": worker_mod.stats_keys(before, after_first, final),
            "note": "counters are null (NOT zero) when the route exposes no stats callable: a "
                    "missing counter must never read as 'no compile'; the plan table's misses "
                    "are now recorded per generation as generations[i].batch_plan_misses (a "
                    "re-decode/re-trace/re-merge, never a recompile)",
        },
        "stats_before_first_call": before, "stats_after_first_call": after_first,
        "stats_after_generations": final,
        "compile_misses_first_call": surfaces.batch_misses(before, after_first),
        "compile_misses_generations": surfaces.batch_misses(after_first, final),
        "compile_misses_total": surfaces.batch_misses(before, final),
        # the PLAN table's misses over the generation loop: the same span as
        # compile_misses_generations, off the table whose miss re-merges instead of recompiling
        "plan_misses_generations": surfaces.plan_misses(after_first, final),
    }


def band_reports(batch_module: Any) -> Dict[str, Any]:
    """The band/merge shape of every merged plan this process built (compile-free).

    ``worker.batch_plan_facts`` walks ``BATCH_PLAN_MEMO`` through ``describe_plan`` (nothing is
    decoded, traced or compiled) and keeps the ``BatchPlan`` objects, so ``common.band_report``
    can read the plan's OWN report -- the only place ``bands`` and the two static cost-model
    estimates live.  ``plans`` is keyed by the same ``(variant, per_row)`` label
    ``worker.batch_plan_facts`` uses; ``sr_fitness`` is the loss-variant entry when present.
    A memo that cannot be read (or a tree without the route's introspection) records
    ``available: false`` with every band field ``null`` plus a note, never ``0``.
    """
    facts = worker_mod.batch_plan_facts(batch_module)
    by_label = {
        label: (entry or {}).get("band_report") or common.band_report(None)
        for label, entry in (facts.get("summary") or {}).items()
    }
    primary: Any = None
    for label, block in by_label.items():
        if label.startswith("loss"):
            primary = block
            break
    if primary is None and by_label:
        primary = next(iter(by_label.values()))
    if primary is None:
        primary = common.band_report(None)
    return {
        "available": bool(facts.get("available")),
        "note": facts.get("note"),
        "sr_fitness": primary,
        "plans": by_label,
    }


# --- the measurement -------------------------------------------------------
@dataclass
class Measurement:
    """Everything one child needs to measure ONE ``(cell, route)``.

    The caller (``batch_economics.run_child``) owns the parts that must happen BEFORE the torch
    import -- the env layers, the batch preflight, the route spec -- and hands the resolved result
    in, so this module never re-derives a knob or a policy.
    """

    cell: Dict[str, Any]
    route: str
    arm: str
    tree_root: str
    generations: int
    warmup: int
    reps: int
    seed: int
    is_batch: bool
    batch_policy: Optional[Dict[str, Any]]
    applied_knobs: Dict[str, str]
    route_knobs: Dict[str, Any]
    route_spec_knobs: Dict[str, str]
    override: Dict[str, str]
    eval_mode: Optional[str]
    backend: Optional[str]
    evoxc_engine: Any
    dense_engine: Any
    batch_module: Any
    surfaces: CounterSurfaces
    note: Optional[str] = None


def measure(ctx: Measurement) -> Dict[str, Any]:
    """Run the cold call, the warm medians, the generation loop and the record assembly.

    Imports nothing heavy itself: the caller has already imported ``torch`` / ``evogp`` and passed
    the engine modules in, which is what keeps this module torch-free at import.
    """
    import torch  # noqa: PLC0415 - imported by the caller already; this is the measurement body

    import arms as arms_mod  # noqa: PLC0415 - the harness arm namespace (torch-free at import)

    evoxc_engine = ctx.evoxc_engine
    surfaces = ctx.surfaces
    is_batch = ctx.is_batch

    # determinant workload: torch.manual_seed(seed) + Forest.random_generate
    workload = common.build_workload(ctx.cell, ctx.seed)
    forest = workload["forest"]
    structures = worker_mod.forest_stats(forest)
    arm_obj = arms_mod.make_arm(ctx.arm, workload)

    # (b) the cold first call: the merged-graph (batch) or dense-graph compile
    evoxc_before, batch_before = evoxc_engine.cache_stats(), surfaces.batch()
    stage_before = surfaces.stage()
    content_before = surfaces.content()
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = arm_obj.sr_fitness()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1e3
    evoxc_after, batch_after = evoxc_engine.cache_stats(), surfaces.batch()
    stage_after = surfaces.stage()
    content_after = surfaces.content()
    first_call: Dict[str, Any] = {
        "first_call_ms": round(first_call_ms, 4),
        "evoxc_cache": {"before": evoxc_before, "after": evoxc_after,
                        "compile_misses": surfaces.misses(evoxc_before, evoxc_after)},
        "digest": common.digest(loss),
    }
    del loss
    if is_batch:
        first_call["merged_graph_cache"] = {
            "available": surfaces.batch_stats is not None,
            "stats_fn": "evogp.tree.batch_engine.batch_cache_stats",
            "before": batch_before, "after": batch_after,
            "compile_misses": surfaces.batch_misses(batch_before, batch_after),
        }
        # What the cold call spent its seconds on (decode/trace/merge vs lower/compile/load) plus
        # the merge shape that explains it.
        first_call["stage_ms"] = stages.delta_block(stage_before, stage_after)
        first_call["stage_ms"]["plan"] = stages.plan_block(
            worker_mod.batch_plan_facts(ctx.batch_module))
        # The plan-key REUSE surface over the cold call: did the content table serve an already
        # merged plan, or was this population merged from scratch?  Null (never 0) when the route
        # ships no ``batch_content_stats()``.
        first_call["content"] = content.window_block(content_before, content_after)

    # (c) warm medians on the same population (the compile is already paid).  The batch probe
    # snapshots the merged-graph cache the instant time_call's untimed call returns -- the same
    # helper worker.py uses to separate the cold compile from the warmup+reps total.
    probe = (worker_mod.BatchStatsProbe(arm_obj.sr_fitness, surfaces.batch_stats)
             if is_batch and surfaces.batch_stats is not None else None)
    # Wraps whatever time_call would otherwise call (never changing time_call itself): the reset
    # bracket is what makes each block a PER-CALL block rather than a cumulative one.
    stage_probe = (stages.StageStatsProbe(probe if probe is not None else arm_obj.sr_fitness,
                                          surfaces.stage_stats, surfaces.stage_reset)
                   if is_batch else None)
    warm_before, batch_warm_before = evoxc_engine.cache_stats(), surfaces.batch()
    content_warm_before = surfaces.content()
    timing = common.time_call(
        stage_probe if stage_probe is not None else (probe if probe is not None
                                                     else arm_obj.sr_fitness),
        warmup=ctx.warmup, reps=ctx.reps, stats=evoxc_engine.cache_stats)
    warm_after = evoxc_engine.cache_stats()
    content_warm_after = surfaces.content()
    warm_result = timing.pop("_out")
    warm_after_first = timing.pop("cache_after_first_call")
    sr_fitness: Dict[str, Any] = {
        "warmup": timing["warmup"], "reps": timing["reps"],
        "samples_ms": [_round(value) for value in timing["samples_ms"]],
        # time_call's own first call is a WARM one here (the compile happened above); the cold
        # number is ``first_call.first_call_ms`` and lives only there.
        "warmup_call_ms": _round(timing["first_call_ms"]),
        "median_ms": _round(timing["median_ms"]), "min_ms": _round(timing["min_ms"]),
        "max_ms": _round(timing["max_ms"]), "mean_ms": _round(timing["mean_ms"]),
        "cache": {"before": warm_before, "after_first_call": warm_after_first, "after": warm_after,
                  "compile_misses_first_call": surfaces.misses(warm_before, warm_after_first),
                  "compile_misses_this_workload": surfaces.misses(warm_before, warm_after)},
        "digest": common.digest(warm_result),
    }
    del warm_result
    if is_batch:
        batch_warm_after_first = probe.first if probe is not None else None
        batch_warm_after = surfaces.batch()
        # the same before -> after_first_call -> after chain as the evoxc block above, so each
        # delta in this entry is checkable against its own snapshots
        sr_fitness["cache"]["batch_before"] = batch_warm_before
        sr_fitness["cache"]["batch_after_first_call"] = batch_warm_after_first
        sr_fitness["cache"]["batch_after"] = batch_warm_after
        sr_fitness["cache"]["batch_compile_misses_first_call"] = surfaces.batch_misses(
            batch_warm_before, batch_warm_after_first)
        sr_fitness["cache"]["batch_compile_misses_this_workload"] = surfaces.batch_misses(
            batch_warm_before, batch_warm_after)
        # time_call calls fn once untimed, then warmup times, then reps times: the LAST reps blocks
        # are the timed samples, block 0 is the untimed warm call (kept as ``first_call``).
        sample_blocks = stage_probe.blocks[-ctx.reps:] if stage_probe is not None else []
        sr_fitness["stage_ms"] = stages.median_block(sample_blocks, samples=ctx.reps)
        sr_fitness["stage_ms"]["first_call"] = (stage_probe.first
                                                if stage_probe is not None else None)
        sr_fitness["stage_ms"]["plan"] = stages.plan_block(
            worker_mod.batch_plan_facts(ctx.batch_module))
        # the same before -> after span as the warm cache deltas above: what the warm calls asked
        # the plan-key table for, and how often it answered from content instead of re-merging
        sr_fitness["content"] = content.window_block(content_warm_before, content_warm_after)
    warm_median_ms = sr_fitness["median_ms"]

    # (d) consecutive real generations: the merged-graph cache's measured hit rate
    timed = worker_mod.TimedFitness(arm_obj.fitness)
    pipeline = arms_mod.make_pipeline(workload, timed)
    # Signatures this process has compiled, seeded with the initial population (the timed first call
    # above compiled exactly it -- generation 1 evaluates that same population).
    compiled_signatures = {structures["structure_signature"]}
    generations: List[Dict[str, Any]] = []
    for index in range(1, ctx.generations + 1):
        # BEFORE the step: this is the population the generation is about to evaluate
        signature = common.population_signature(pipeline.algorithm.forest)
        seen_before = signature["signature"] in compiled_signatures
        before = {
            "evoxc": evoxc_engine.cache_stats(), "batch": surfaces.batch(),
            "stage": surfaces.stage(), "content": surfaces.content(),
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipeline.step()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - started) * 1e3
        after = {
            "evoxc": evoxc_engine.cache_stats(), "batch": surfaces.batch(),
            "stage": surfaces.stage(), "content": surfaces.content(),
        }
        entry = generation_entry(index, signature, seen_before, total_ms, timed.last_ms,
                                 before, after, surfaces, is_batch,
                                 best_fitness=pipeline.best_fitness)
        compiled_signatures.add(signature["signature"])
        generations.append(entry)
    batch_final = surfaces.batch()
    content_final = surfaces.content()

    # (e) the derived block, computed only from the recorded rows
    merged_cache = (merged_cache_block(surfaces, batch_before, batch_after, batch_final)
                    if is_batch else None)
    summary = generation_summary(generations, first_call_ms, warm_median_ms,
                                 merged_graph_cache=merged_cache)
    if is_batch:
        # The stage tables + the compile stage's real share of the loop: derived from the rows above
        # only (stages.summary_block never asserts a number it was not given).
        summary.update(stages.summary_block(first_call.get("stage_ms"),
                                            sr_fitness.get("stage_ms"), generations))
        # The content table over the WHOLE measurement (cold call + warm calls + the loop) plus how
        # many recorded generations had a plan-key decision to make at all (content.row_state).
        summary["content"] = content.summary_block(content_before, content_final, generations)

    # (f) the record
    record: Dict[str, Any] = {
        "env": common.env_block(ctx.tree_root, ctx.arm,
                                extra={"applied_knobs": ctx.applied_knobs, "cell": ctx.cell,
                                       "route": ctx.route, "arm": ctx.arm}),
        "cell": ctx.cell, "route": ctx.route, "arm": ctx.arm, "measurement": MEASUREMENT,
        "protocol": {
            "one_process_per_config": True, "one_cell_per_process": True,
            "generations": ctx.generations, "warmup": ctx.warmup, "reps": ctx.reps,
            "seed": ctx.seed, "sync_bracketed": True,
            "evoxc_disable_locations": os.environ.get(DISABLE_LOCATIONS_ENV),
            "workload": "sr_fitness", "route_is_batch": is_batch, "pipeline_warmup": False,
            "route_spec": {"token": ctx.route, "arm": ctx.arm,
                           "knobs": dict(ctx.route_spec_knobs)},
            "route_knobs": ctx.route_knobs,
        },
        "structures": structures,
        "routes": {"sr_fitness": common.route_decision(ctx.cell["pop"], ctx.cell["rows"],
                                                      ctx.eval_mode, ctx.backend)},
        "dense_policy": {"auto_min_pop": ctx.dense_engine.resolve_auto_min_pop(),
                         "auto_max_rows": ctx.dense_engine.resolve_auto_max_rows()},
        "first_call": first_call, "sr_fitness": sr_fitness, "generations": generations,
        "summary": summary, "engine_cache_final": evoxc_engine.cache_stats(),
        "notes": ["generation 1 evaluates the INITIAL population, whose graph the timed first call "
                  "already compiled, so generation 1 is expected to hit; it is recorded as measured "
                  "(not warmed up away) and the interesting number is generations 2..N"],
    }
    if is_batch:
        # The band/merge shape of the plan(s) that served this route, read off BATCH_PLAN_MEMO
        # (compile-free): the clamp verdict of a named composition route above, and the per-body
        # band counts a banded+chunked merge produced.  Null-never-zero: a plan nobody can describe
        # records ``available: false`` + nulls + a note (common.band_report).
        bands = band_reports(ctx.batch_module)
        record["band_report"] = bands["sr_fitness"]
        record["band_reports"] = bands["plans"]
        record["band_report_note"] = bands["note"]
    record["notes"].extend(record_notes(ctx, surfaces))
    if ctx.override:
        record["knobs_override"] = dict(ctx.override)
        record["notes"].append(
            "knob override " + ", ".join(f"{key}={value}" for key, value in ctx.override.items())
            + ": applied AFTER the arm's own knob spec (and after the named route's own spec) and "
            "BEFORE the policy was resolved, so batch_policy / env.applied_knobs / "
            "protocol.route_knobs / this block all carry the value the engine read")
    if ctx.note:
        record["notes"].append(ctx.note)
    return record


def record_notes(ctx: Measurement, surfaces: CounterSurfaces) -> List[str]:
    """The route-specific caveats a record carries (never a number, only provenance)."""
    notes: List[str] = []
    if ctx.is_batch:
        notes.append(
            "batch route: ONE merged graph per population structure multiset, so a multi-second "
            "first_call_ms is the expected cold cost of the route, recorded as measured and never "
            "treated as a defect or retried away")
        if ctx.route_spec_knobs:
            notes.append(
                "named composition route: the route's own knob spec "
                + ", ".join(f"{name}={value}" for name, value in ctx.route_spec_knobs.items())
                + " is applied AFTER arms.apply_arm_env cleared the table and wrote the arm's own "
                "knobs, so protocol.route_knobs states which layer set each variable")
        if surfaces.batch_stats is None:
            notes.append(
                "batch route: evogp.tree.batch_engine.batch_cache_stats() is unavailable on this "
                "tree, so every merged-graph counter is null (NOT zero) and compile_hit is null "
                "wherever only those counters could tell; the compile cost can then only be read "
                "off first_call_ms")
        if surfaces.stage_stats is None:
            notes.append(
                f"batch route: {stages.STATS_FN_PATH}() is unavailable on this tree, so every "
                "stage_ms counter is null (NOT zero) with available=false: the split was not "
                "measured, which is not the same as any stage costing nothing")
        if surfaces.stage_reset is None:
            notes.append(
                f"batch route: {stages.MODULE_NAME}.{stages.RESET_FN_NAME}() is unavailable, so each "
                "stage block is cumulative rather than per-call; a max is then only a "
                "process-lifetime maximum")
        if surfaces.content_stats is None:
            notes.append(
                f"batch route: {content.STATS_FN_PATH}() is unavailable on this tree, so every "
                "content counter is null (NOT zero) with available=false: plan-key reuse from a "
                "rebuilt-but-equal population was not measured, which is not the same as it never "
                "happening -- read first_call_ms / generations[i].batch_compile_misses instead")
        else:
            notes.append(
                "batch route: the content blocks (first_call / sr_fitness / generations[i] / "
                "summary) price the plan-key REUSE surface, not the compile itself. The identity "
                "table is consulted first, so a repeated call on the same Forest object AND the same "
                "plan parameters leaves the content table untouched (attempts 0, which is not a "
                "content miss); a first call for a NEW plan shape is an identity miss even on an "
                "unchanged object, so it does probe the content table (one attempt, usually a miss "
                "that stores the entry), and only a REBUILT-but-equal population can produce a "
                "content hit -- which means the merge was served instead of re-decoded, re-traced "
                "and re-merged")
    else:
        notes.append(
            "dense route (comparison column): no merged-graph cache exists, so batch_* counters are "
            "ABSENT rather than null; its per-generation compile misses are the evoxc engine's own "
            "and its first call is the dense-graph compile")
    return notes
