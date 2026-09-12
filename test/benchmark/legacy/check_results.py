#!/usr/bin/env python3
"""Post-run validator for the OLD-side records of `test/benchmark/legacy/run_matrix.py`.

`test/benchmark/legacy/check_results.py` — pure-stdlib (numpy only for the optional `--golden` cross
check, torch is never imported), so it can be run while the benchmark lock is held by somebody else:

    python check_results.py results/C5.json
    python check_results.py results/*.json --golden ../../goldens/sr_fitness.npz

Checks per JSON (SPEC.md "Required outputs" + the consistency the harness promises):

* required SPEC keys present (pipeline schema for C1/C2/C2b/C4, eval-only schema for C5/C5b),
* `side == "old"` and `evogp_file` living inside the legacy checkout (`--legacy-root`,
  default `/tmp/evogp_legacy_bench`),
* `env`: the two ETL variables are `null` (the legacy engine ignores them) and
  `CUDA_VISIBLE_DEVICES` matches `--gpu` when given,
* pipeline: `len(per_gen_ms) == len(per_gen_best_fitness) == config.gens`,
  `per_gen_best_fitness` monotone non-decreasing (`pipeline.best_fitness` is a running max),
  `best_fitness_final == per_gen_best_fitness[-1]`, `cold_first_gen_ms == per_gen_ms[0]`,
* eval-only: `first_call_ms == per_call_ms[0]`, `median_warm_ms == median(per_call_ms[1:])`.

It also prints the per-config timing stats (median/min/max of the warm generations or calls) and flags
a contention-warped profile (max > 10x median of the measured region). With `--golden`, a C5 record is
compared against the frozen `test/goldens/sr_fitness.npz` capture: `nan_count == 3` and
`fitness_mean` == -nanmean(finite entries) of `forest_sr_fitness__hybrid_parallel__mse`, rtol=atol=1e-4.

Exit code 0 = every check held, 1 = at least one FAIL (each one printed).
"""
import argparse
import glob
import json
import os
import statistics
import sys

ENV_KEYS = ("ETL_DISABLE_LOCATIONS", "EVOGP_ETL_BACKEND", "CUDA_VISIBLE_DEVICES")
PIPELINE_REQUIRED = (
    "tag", "side", "evogp_file", "evogp_commit", "problem", "config", "env", "gpu", "wall_s",
    "per_gen_ms", "per_gen_best_fitness", "best_fitness_final", "cold_first_gen_ms", "notes",
)
PIPELINE_CONFIG_REQUIRED = ("pop", "gens", "tree_len", "seed")
EVAL_REQUIRED = (
    "tag", "side", "evogp_file", "evogp_commit", "eval_kind", "fixture", "env", "gpu", "wall_s",
    "per_call_ms", "first_call_ms", "median_warm_ms", "fitness_mean", "nan_count", "notes",
)
GOLDEN_KEY = "forest_sr_fitness__hybrid_parallel__mse"
DEFAULT_LEGACY_ROOT = "/tmp/evogp_legacy_bench"


def golden_expectation(path):
    """-nanmean(finite entries) + NaN count of the frozen legacy SR-fitness capture."""
    import numpy

    arr = numpy.load(path)[GOLDEN_KEY].astype(numpy.float64)
    finite = arr[numpy.isfinite(arr)]
    return -float(finite.mean()), int(numpy.isnan(arr).sum()), len(arr)


def close(a, b, rtol=1e-4, atol=1e-4):
    return abs(a - b) <= atol + rtol * abs(b)


def check_one(path, legacy_root, gpu, commit, golden_path, failures):
    """Validate one JSON; append FAIL strings to `failures`, return the record (or None)."""
    tag = os.path.splitext(os.path.basename(path))[0]

    def fail(msg):
        failures.append(f"{tag}: {msg}")

    with open(path) as handle:
        rec = json.load(handle)

    eval_only = "eval_kind" in rec
    missing = [k for k in (EVAL_REQUIRED if eval_only else PIPELINE_REQUIRED) if k not in rec]
    if missing:
        fail(f"missing required keys {missing}")
        return rec
    if rec.get("tag") != tag:
        fail(f"tag {rec['tag']!r} != filename stem {tag!r}")
    if rec["side"] != "old":
        fail(f"side {rec['side']!r} != 'old' (legacy checkout not winning on PYTHONPATH?)")
    if commit and rec.get("evogp_commit") != commit:
        fail(f"evogp_commit {rec.get('evogp_commit')!r} != {commit!r}")
    evogp_file = os.path.realpath(rec["evogp_file"])
    if not evogp_file.startswith(os.path.realpath(legacy_root) + os.sep):
        fail(f"evogp_file {evogp_file!r} is not inside {legacy_root!r}")
    env = rec.get("env") or {}
    for key in ENV_KEYS[:2]:
        if env.get(key) is not None:
            fail(f"env[{key!r}] = {env[key]!r}, expected null (legacy engine ignores ETL vars)")
    if gpu is not None and env.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        fail(f"env['CUDA_VISIBLE_DEVICES'] = {env.get('CUDA_VISIBLE_DEVICES')!r} != {str(gpu)!r}")

    if eval_only:
        per_call = rec["per_call_ms"]
        warm = per_call[1:]
        if not per_call:
            fail("per_call_ms is empty")
            return rec
        if rec["first_call_ms"] != per_call[0]:
            fail(f"first_call_ms {rec['first_call_ms']} != per_call_ms[0] {per_call[0]}")
        expect_median = statistics.median(warm) if warm else None
        if expect_median is None or rec["median_warm_ms"] != expect_median:
            fail(f"median_warm_ms {rec['median_warm_ms']} != median(per_call_ms[1:]) {expect_median}")
        warm = warm or per_call
        print(
            f"  {tag}: evals={len(per_call)} wall={rec['wall_s']:.3f}s first_call_ms={per_call[0]:.3f} "
            f"warm median/min/max={statistics.median(warm):.3f}/{min(warm):.3f}/{max(warm):.3f} ms "
            f"nan_count={rec['nan_count']} fitness_mean={rec['fitness_mean']!r}"
        )
    else:
        gens = rec["config"]["gens"]
        missing_cfg = [k for k in PIPELINE_CONFIG_REQUIRED if k not in rec["config"]]
        if missing_cfg:
            fail(f"config missing keys {missing_cfg}")
        per_gen, best = rec["per_gen_ms"], rec["per_gen_best_fitness"]
        if len(per_gen) != gens:
            fail(f"len(per_gen_ms) {len(per_gen)} != config.gens {gens}")
        if len(best) != gens:
            fail(f"len(per_gen_best_fitness) {len(best)} != config.gens {gens}")
        if any(b < a for a, b in zip(best, best[1:])):
            bad = next(i for i in range(1, len(best)) if best[i] < best[i - 1])
            fail(f"per_gen_best_fitness decreases at gen {bad}: {best[bad - 1]} -> {best[bad]}")
        if best and rec["best_fitness_final"] != best[-1]:
            fail(f"best_fitness_final {rec['best_fitness_final']} != per_gen_best_fitness[-1] {best[-1]}")
        if per_gen and rec["cold_first_gen_ms"] != per_gen[0]:
            fail(f"cold_first_gen_ms {rec['cold_first_gen_ms']} != per_gen_ms[0] {per_gen[0]}")
        warm = per_gen[1:] or per_gen
        med, mx = statistics.median(warm), max(warm)
        warp = "  [WARPED? max > 10x median]" if mx > 10 * med else ""
        print(
            f"  {tag}: gens={gens} wall={rec['wall_s']:.3f}s gen0={per_gen[0]:.3f} ms warm "
            f"median/min/max={med:.3f}/{min(warm):.3f}/{mx:.3f} ms best_final={rec['best_fitness_final']:.6f}"
            f"{warp}"
        )

    if golden_path and eval_only and rec.get("eval_kind") == "fixed_fixture_forest":
        want_mean, want_nan, n = golden_expectation(golden_path)
        got_mean, got_nan = rec["fitness_mean"], rec["nan_count"]
        for label, ok, detail in (
            ("nan_count", got_nan == want_nan, f"{got_nan} vs golden {want_nan}"),
            (
                "fitness_mean",
                close(got_mean, want_mean),
                f"{got_mean!r} vs golden -nanmean(finite)={want_mean!r} (rtol=atol=1e-4, n={n})",
            ),
        ):
            print(f"  {tag}: GOLDEN {label} {'PASS' if ok else 'FAIL'} — {detail}")
            if not ok:
                fail(f"golden cross-check {label}: {detail}")
    return rec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("json", nargs="+", help="run record JSON file(s); globs are expanded")
    parser.add_argument("--legacy-root", default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--gpu", type=int, default=None, help="CUDA_VISIBLE_DEVICES the runs were given")
    parser.add_argument("--commit", default="b613503")
    parser.add_argument("--golden", default=None, help="frozen capture for the C5 cross-check, e.g. "
                                                      "../../goldens/sr_fitness.npz")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.json:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits or [pattern])

    failures = []
    for path in paths:
        print(f"checking {path}")
        try:
            check_one(path, args.legacy_root, args.gpu, args.commit, args.golden, failures)
        except Exception as exc:  # a malformed/unreadable record is a FAIL too
            failures.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"FAIL — {len(failures)} problem(s):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"PASS — {len(paths)} record(s) checked, all SPEC consistency checks held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
