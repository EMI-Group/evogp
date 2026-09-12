#!/usr/bin/env python
"""Validate ``bench_gp.py`` JSON outputs (used for the Phase-1 smoke suite).

Checks, per JSON file:

* ``side`` is the expected side (default ``new``) and the base SPEC fields are present;
* ``config.reps`` matches ``len(per_rep)`` and every rep record carries the task's
  timing keys, with rep 0 mirrored into the top-level fields;
* training: ``len(per_gen_ms) == config.gens``, the best-fitness series is
  non-decreasing (the pipeline is elitist), ``best_fitness_final`` and
  ``cold_first_gen_ms`` agree with the arrays;
* eval: ``len(per_call_ms) == config.eval_reps``, ``first_call_ms`` /
  ``median_warm_ms`` agree with the array;
* the ``partial`` flag is consistent with the number of completed steps, and a
  soft-wall-cap stop carries a ``projection``;
* a few per-tag domain checks (``ETL_DISABLE_LOCATIONS`` value, the W1 reference
  fitness of the full C1-scale smoke run).

Usage::

    PYTHONPATH=src python test/benchmark/new/check_smoke.py [JSON ...]

With no arguments it validates ``test/benchmark/new/results/smoke/*.json``.
Exits non-zero when any check fails.
"""

import argparse
import glob
import json
import os
import statistics
import sys

DEFAULT_GLOB = os.path.join("test", "benchmark", "new", "results", "smoke", "*.json")

TRAIN_KEYS = {
    "tag",
    "side",
    "task",
    "evogp_file",
    "evogp_commit",
    "etl_version",
    "problem",
    "config",
    "env",
    "gpu",
    "wall_s",
    "generation_loop_s",
    "per_gen_ms",
    "per_gen_best_fitness",
    "cold_first_gen_ms",
    "best_fitness_final",
    "partial",
    "projection",
    "complete",
    "per_rep",
    "notes",
    "started_at",
    "finished_at",
}
EVAL_KEYS = {
    "tag",
    "side",
    "task",
    "eval_kind",
    "fixture",
    "evogp_file",
    "evogp_commit",
    "etl_version",
    "problem",
    "config",
    "env",
    "gpu",
    "per_call_ms",
    "first_call_ms",
    "median_warm_ms",
    "wall_s",
    "partial",
    "projection",
    "complete",
    "per_rep",
    "notes",
    "started_at",
    "finished_at",
}
TRAIN_REP_KEYS = {
    "rep",
    "wall_s",
    "generation_loop_s",
    "per_gen_ms",
    "per_gen_best_fitness",
    "cold_first_gen_ms",
    "best_fitness_final",
    "partial",
    "projection",
    "complete",
}
EVAL_REP_KEYS = {
    "rep",
    "wall_s",
    "per_call_ms",
    "first_call_ms",
    "median_warm_ms",
    "partial",
    "projection",
    "complete",
}
EVAL_TASKS = ("eval64", "evalfresh")

#: tag -> (ETL_DISABLE_LOCATIONS, reference best fitness or None)
TAG_EXPECTATIONS = {
    "SMOKE_C0": ("0", None),
    "SMOKE_C1FULL": ("1", -0.1720),  # src/evogp/sr_test.py module docstring
}


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return bool(condition)

    def close(self, value, reference, tol, label):
        ok = isinstance(value, (int, float)) and abs(value - reference) <= tol
        return self.check(ok, f"{label}: {value!r} vs reference {reference!r} (tol {tol})")

    def steps(self, got, want, relaxed, label):
        """Exact match for a completed run; ``<= want`` when partial or truncated."""
        self.checks += 1
        ok = got <= want if relaxed else got == want
        if not ok:
            self.failures.append(f"{label}: got {got}, want {want} (relaxed={relaxed})")
        return ok


def check_file(path, expected_side, checker):
    with open(path) as handle:
        payload = json.load(handle)
    tag = payload.get("tag", os.path.basename(path))
    task = payload.get("task")
    is_eval = task in EVAL_TASKS
    required = EVAL_KEYS if is_eval else TRAIN_KEYS
    missing = sorted(required - set(payload))
    checker.check(not missing, f"{tag}: missing top-level keys {missing}")
    checker.check(
        payload.get("side") == expected_side,
        f"{tag}: side == {expected_side!r} (got {payload.get('side')!r})",
    )
    config = payload.get("config") or {}
    reps = config.get("reps")
    per_rep = payload.get("per_rep") or []
    checker.check(
        reps == len(per_rep), f"{tag}: config.reps {reps} == len(per_rep) {len(per_rep)}"
    )

    summary = {"tag": tag, "task": task, "steps": None, "wall_s": payload.get("wall_s")}

    for rep_record in per_rep:
        index = rep_record.get("rep")
        rep_keys = EVAL_REP_KEYS if is_eval else TRAIN_REP_KEYS
        rep_missing = sorted(rep_keys - set(rep_record))
        checker.check(not rep_missing, f"{tag} rep{index}: missing keys {rep_missing}")
        checker.check(isinstance(index, int), f"{tag}: rep index is an int")

    if is_eval:
        requested = config.get("eval_reps")
        relaxed = bool(payload.get("partial")) or not bool(payload.get("complete", True))
        for rep_record in per_rep:
            calls = rep_record.get("per_call_ms") or []
            index = rep_record.get("rep")
            checker.steps(
                len(calls),
                requested,
                relaxed,
                f"{tag} rep{index}: per_call_ms vs eval_reps {requested}",
            )
            checker.check(
                rep_record.get("first_call_ms") == (calls[0] if calls else None),
                f"{tag} rep{index}: first_call_ms == per_call_ms[0]",
            )
            warm = calls[1:]
            expected_median = statistics.median(warm) if len(warm) >= 2 else None
            checker.check(
                rep_record.get("median_warm_ms") == expected_median,
                f"{tag} rep{index}: median_warm_ms == median(per_call_ms[1:])",
            )
            if rep_record.get("wall_s") is not None:
                checker.check(
                    rep_record["wall_s"] >= 0,
                    f"{tag} rep{index}: rep wall_s is a non-negative number",
                )
        if per_rep:
            first = per_rep[0]
            checker.check(
                payload.get("per_call_ms") == first.get("per_call_ms"),
                f"{tag}: top-level per_call_ms mirrors rep 0",
            )
            checker.check(
                payload.get("first_call_ms") == first.get("first_call_ms"),
                f"{tag}: top-level first_call_ms mirrors rep 0",
            )
            checker.check(
                payload.get("median_warm_ms") == first.get("median_warm_ms"),
                f"{tag}: top-level median_warm_ms mirrors rep 0",
            )
            checker.check(
                payload.get("wall_s") == first.get("wall_s"),
                f"{tag}: top-level wall_s mirrors rep 0",
            )
            checker.check(
                len(payload.get("per_call_ms") or []) == len(first["per_call_ms"]),
                f"{tag}: top-level per_call_ms has rep 0's length",
            )
        calls_total = sum(len(r.get("per_call_ms") or []) for r in per_rep)
        requested_total = (requested or 0) * (reps or 0)
        partial = bool(payload.get("partial"))
        complete = bool(payload.get("complete", True))
        if calls_total < requested_total:
            checker.check(
                partial or not complete,
                f"{tag}: {requested_total - calls_total} call(s) missing -> partial is set "
                f"or the file is truncated (complete=false)",
            )
        else:
            checker.check(not partial, f"{tag}: no calls missing -> partial is false")
            checker.check(complete, f"{tag}: full run -> complete is true")
        summary["truncated"] = not complete
        if partial and payload.get("projection") is not None:
            projection = payload["projection"]
            checker.check(
                projection.get("calls_completed") == calls_total,
                f"{tag}: projection.calls_completed == {calls_total}",
            )
        summary["steps"] = calls_total
        warm = payload.get("median_warm_ms")
        first = payload.get("first_call_ms")
        summary["extra"] = (
            f"first={'n/a' if first is None else f'{first:.1f}ms'} "
            f"warm_med={'n/a' if warm is None else f'{warm:.2f}ms'} "
            f"({calls_total} calls)"
            if per_rep
            else "no reps"
        )
        return summary

    gens = config.get("gens")
    relaxed = bool(payload.get("partial")) or not bool(payload.get("complete", True))
    for rep_record in per_rep:
        times = rep_record.get("per_gen_ms") or []
        best = rep_record.get("per_gen_best_fitness") or []
        index = rep_record.get("rep")
        checker.steps(
            len(times), gens, relaxed, f"{tag} rep{index}: per_gen_ms vs gens {gens}"
        )
        checker.check(
            len(best) == len(times),
            f"{tag} rep{index}: per_gen_best_fitness same length as per_gen_ms",
        )
        checker.check(
            all(b >= a for a, b in zip(best, best[1:])),
            f"{tag} rep{index}: best fitness is non-decreasing (elitist pipeline)",
        )
        if times:
            checker.check(
                rep_record.get("cold_first_gen_ms") == times[0],
                f"{tag} rep{index}: cold_first_gen_ms == per_gen_ms[0]",
            )
            checker.check(
                rep_record.get("best_fitness_final") == best[-1],
                f"{tag} rep{index}: best_fitness_final == per_gen_best_fitness[-1]",
            )
            if rep_record.get("generation_loop_s") is not None:
                loop = sum(times) / 1000.0
                checker.close(
                    rep_record["generation_loop_s"], loop, 1e-6,
                    f"{tag} rep{index}: generation_loop_s == sum(per_gen_ms)/1000",
                )
            if rep_record.get("wall_s") is not None:
                checker.check(
                    rep_record["wall_s"] >= rep_record.get("construct_s", 0),
                    f"{tag} rep{index}: wall_s >= construct_s",
                )
    if per_rep:
        first = per_rep[0]
        checker.check(
            payload.get("per_gen_ms") == first.get("per_gen_ms"),
            f"{tag}: top-level per_gen_ms mirrors rep 0",
        )
        checker.check(
            payload.get("per_gen_best_fitness") == first.get("per_gen_best_fitness"),
            f"{tag}: top-level per_gen_best_fitness mirrors rep 0",
        )
        checker.check(
            payload.get("best_fitness_final") == first.get("best_fitness_final"),
            f"{tag}: top-level best_fitness_final mirrors rep 0",
        )
        checker.check(
            payload.get("cold_first_gen_ms") == first.get("cold_first_gen_ms"),
            f"{tag}: top-level cold_first_gen_ms mirrors rep 0",
        )
    gens_total = sum(len(r.get("per_gen_ms") or []) for r in per_rep)
    requested_total = (gens or 0) * (reps or 0)
    partial = bool(payload.get("partial"))
    complete = bool(payload.get("complete", True))
    if gens_total < requested_total:
        checker.check(
            partial or not complete,
            f"{tag}: {requested_total - gens_total} generation(s) missing -> partial is set "
            f"or the file is truncated (complete=false)",
        )
        projection = payload.get("projection")
        if partial and projection is not None:
            checker.check(
                projection.get("gens_completed") == gens_total,
                f"{tag}: projection.gens_completed == {gens_total}",
            )
            checker.check(
                projection.get("gens_requested") == gens,
                f"{tag}: projection.gens_requested == {gens}",
            )
            if projection.get("per_gen_ms_mean") is not None:
                checker.check(
                    projection.get("projected_total_s_for_requested_gens")
                    == projection["per_gen_ms_mean"] * gens / 1000.0,
                    f"{tag}: projected_total_s_for_requested_gens is mean*gens/1000",
                )
    else:
        checker.check(not partial, f"{tag}: nothing missing -> partial is false")
        checker.check(complete, f"{tag}: full run -> complete is true")
    summary["truncated"] = not complete
    if per_rep and per_rep[0].get("per_gen_ms"):
        times = per_rep[0]["per_gen_ms"]
        summary["steps"] = len(times)
        summary["extra"] = (
            f"median={statistics.median(times):.1f}ms first={times[0]:.1f}ms "
            f"best={payload.get('best_fitness_final')}"
        )

    expected = TAG_EXPECTATIONS.get(tag)
    if expected:
        locations, reference_fitness = expected
        checker.check(
            payload["env"].get("ETL_DISABLE_LOCATIONS") == locations,
            f"{tag}: ETL_DISABLE_LOCATIONS == {locations!r} "
            f"(got {payload['env'].get('ETL_DISABLE_LOCATIONS')!r})",
        )
        if reference_fitness is not None:
            checker.close(
                payload.get("best_fitness_final"),
                reference_fitness,
                1e-3,
                f"{tag}: best_fitness_final matches the sr_test reference",
            )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help=f"JSON files (default: {DEFAULT_GLOB})")
    parser.add_argument("--side", default="new", help="expected side (default: new)")
    args = parser.parse_args(argv)

    paths = args.paths or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"no JSON files found ({DEFAULT_GLOB})")
        return 1

    checker = Checker()
    summaries = []
    for path in paths:
        summaries.append(check_file(path, args.side, checker))

    print(f"{'tag':<16} {'task':<10} {'steps':>6} {'wall_s':>9}  details")
    truncated = 0
    for summary in summaries:
        wall = summary["wall_s"]
        wall_text = f"{wall:.2f}" if isinstance(wall, (int, float)) else str(wall)
        marker = " [TRUNCATED: hard kill]" if summary.get("truncated") else ""
        truncated += 1 if summary.get("truncated") else 0
        print(
            f"{summary['tag']:<16} {summary['task']:<10} {str(summary['steps']):>6} "
            f"{wall_text:>9}  {summary.get('extra', '')}{marker}"
        )
    print(
        f"\n{len(paths)} JSON file(s) ({truncated} truncated by a hard kill), "
        f"{checker.checks} checks, {len(checker.failures)} failure(s)"
    )
    for failure in checker.failures:
        print(f"  FAIL {failure}")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
