"""Gate over one ``evoxc`` benchmark result directory (stdlib only, exits 1 on failure).

The records in ``<results>/<cell>/<arm>.json`` are written by :mod:`worker` and the
schedule/guard evidence by :mod:`campaign`; this script is the machine check over
both, so a record that was hand-edited, half-written, or measured under a bad
protocol cannot slip into a table.  It verifies:

* every ``campaign.json`` config with ``status == "ok"`` has a present, parsing
  record (and, with ``--expect-arms``, that every expected arm is present per cell);
* ``env.schema`` is the frozen schema and ``tree_root_sha`` / ``torch`` / ``gpu``
  are non-empty (engine provenance, not decoration);
* per workload: at least 5 warm samples, ``len(samples_ms) == reps`` (so the cold
  first call cannot sit inside the warm samples) and ``first_call_ms > 0``;
* ``min_ms <= median_ms <= max_ms``, and ``median_ms`` recomputed from
  ``samples_ms`` matches the stored value (a hand-edited median is caught);
* for the evoxc arms: ``routes`` is non-empty and every workload's
  ``cache.compile_misses_first_call`` was recorded (not ``None``);
* digest agreement: all arms of one cell share one ``batch_forward`` and one
  ``forward`` sha256, and every workload's ``nan``/``finite`` counts agree across
  arms (``--allow-digest-mismatch`` downgrades the sha256 half to a warning);
* ``gp_step`` sanity: ``median_total_ms >= median_eval_ms``, ``median_ops_ms >= 0``
  and ``reps == len(steps)``;
* guard honesty: a config whose ``guard.contaminated`` is true is reported as a
  ``WARN`` (never a failure -- the box is shared) and the number of configs that
  ran on an idle device is printed.

Two things it deliberately does not soften: the ``median_ms`` re-computation allows
only the 4-decimal rounding ``worker.py`` applies to ``samples_ms``/``median_ms``
(1e-6 relative, floored at 1.5e-4 ms), and an empty ``env.tree_root_sha`` fails --
a ``git-archive`` snapshot tree root has no HEAD, so a replay campaign must be
quoted with that known caveat.

Usage (repository root, stdlib only, GPU-free)::

    python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/baseline

    python3 test/benchmark/evoxc/check.py --results results/w1 \\
        --expect-arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree \\
        --require-digest-agreement

A partially populated result directory is the normal state while a campaign is
still running: pass ``--allow-missing`` to only warn about the records the driver
has not reached yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402

#: frozen record schema (``common.SCHEMA``; spelled out so a schema edit here is a
#: deliberate act rather than a silent follow)
SCHEMA = "evogp-bench-evoxc-v1"
VALUE_WORKLOADS = ("batch_forward", "forward")
MIN_SAMPLES = 5
#: ``worker.py`` rounds ``samples_ms`` and every statistic to 4 decimals, so a
#: re-computed median can differ from the stored one by up to ~1e-4 ms.
ROUNDING_FLOOR_MS = 1.5e-4
MEDIAN_RTOL = 1e-6


class Checker:
    """Counts checks and collects failures/warnings (one line per failure)."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def check(self, ok: bool, message: str) -> bool:
        self.checks += 1
        if not ok:
            self.failures.append(message)
        return bool(ok)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def load_json(path: str) -> Tuple[Any, Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - an unreadable record is a failure to report
        return None, f"{type(exc).__name__}: {exc}"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_evoxc_arm(arm: str) -> bool:
    return arm in common.EVOXC_ARMS or arm not in common.TORCH_ARMS


def median(samples: List[float]) -> Optional[float]:
    ordered = sorted(value for value in samples if is_number(value))
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def discover(results_dir: str):
    """``(campaign, campaign_error, configs, pairs)`` -- disk scan plus the schedule."""
    campaign, campaign_error = load_json(os.path.join(results_dir, "campaign.json"))
    if campaign is not None and not isinstance(campaign, dict):
        campaign_error, campaign = "campaign.json is not a JSON object", None
    configs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in (campaign or {}).get("configs") or []:
        if isinstance(entry, dict) and entry.get("cell") and entry.get("arm"):
            configs[(entry["cell"], entry["arm"])] = entry
    pairs = set(configs)
    for name in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, name)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if fname.endswith(".json"):
                pairs.add((name, fname[: -len(".json")]))
    return campaign, campaign_error, configs, pairs


def record_path(results_dir: str, cell: str, arm: str, entry: Optional[Dict[str, Any]]) -> str:
    """``<results>/<cell>/<arm>.json``; the campaign's own path only if it lives inside.

    The campaign stores the path it wrote, which can be absolute.  Trusting it as a
    fallback would let a gated directory silently borrow a record from another one,
    so a path outside ``--results`` is ignored and the derived path is used.
    """
    path = os.path.join(results_dir, cell, f"{arm}.json")
    recorded = (entry or {}).get("record")
    if recorded and not os.path.isfile(path):
        inside = os.path.abspath(recorded).startswith(os.path.abspath(results_dir) + os.sep)
        if inside and os.path.isfile(recorded):
            return recorded
    return path


# ---------------------------------------------------------------------------
# per-record checks
# ---------------------------------------------------------------------------
def check_env(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    env = record.get("env") or {}
    checker.check(env.get("schema") == SCHEMA,
                  f"{card}: env.schema={env.get('schema')!r} != {SCHEMA!r}")
    for key in ("tree_root_sha", "torch", "gpu"):
        value = env.get(key)
        checker.check(bool(value), f"{card}: env.{key} is empty ({value!r})")


def check_workload(checker: Checker, card: str, name: str, entry: Dict[str, Any]) -> None:
    label = f"{card}/{name}"
    samples = entry.get("samples_ms")
    samples = samples if isinstance(samples, list) else []
    reps = entry.get("reps")
    checker.check(len(samples) >= MIN_SAMPLES,
                  f"{label}: {len(samples)} warm samples, need >= {MIN_SAMPLES}")
    checker.check(reps == len(samples),
                  f"{label}: reps={reps!r} != len(samples_ms)={len(samples)} "
                  "(the cold first call must not sit in the warm samples)")
    first = entry.get("first_call_ms")
    checker.check(is_number(first) and first > 0,
                  f"{label}: first_call_ms={first!r} is not a positive cold-call time")
    low, mid, high = entry.get("min_ms"), entry.get("median_ms"), entry.get("max_ms")
    if not all(is_number(value) for value in (low, mid, high)):
        checker.check(False, f"{label}: min/median/max not all numeric "
                             f"({low!r}, {mid!r}, {high!r})")
    else:
        checker.check(low <= mid <= high,
                      f"{label}: median_ms={mid!r} outside [min_ms={low!r}, max_ms={high!r}]")
    recomputed = median(samples)
    if recomputed is None:
        checker.check(False, f"{label}: no numeric samples_ms to re-derive the median from")
    elif is_number(mid):
        tolerance = max(MEDIAN_RTOL * abs(recomputed), ROUNDING_FLOOR_MS)
        checker.check(abs(mid - recomputed) <= tolerance,
                      f"{label}: median_ms={mid!r} != median(samples_ms)={recomputed!r} "
                      f"(tolerance {tolerance:.2e} ms)")


def check_cache(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    routes = record.get("routes") or {}
    checker.check(bool(routes), f"{card}: routes is empty for an evoxc arm")
    workloads = record.get("workloads") or {}
    for name, entry in workloads.items():
        if not isinstance(entry, dict):
            continue
        cache = entry.get("cache") or {}
        checker.check(cache.get("compile_misses_first_call") is not None,
                      f"{card}/{name}: cache.compile_misses_first_call is None "
                      "(the cold compile is not attributable)")


def check_gp_step(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    block = record.get("gp_step")
    protocol = record.get("protocol") or {}
    if not isinstance(block, dict):
        if "gp_step" in (protocol.get("workloads") or []):
            checker.check(False, f"{card}: gp_step was requested but the record has none")
        return
    total, eval_ms, ops_ms = (block.get("median_total_ms"), block.get("median_eval_ms"),
                              block.get("median_ops_ms"))
    if all(is_number(value) for value in (total, eval_ms, ops_ms)):
        checker.check(total >= eval_ms,
                      f"{card}/gp_step: median_total_ms={total!r} < median_eval_ms={eval_ms!r}")
        checker.check(ops_ms >= 0, f"{card}/gp_step: median_ops_ms={ops_ms!r} is negative")
    else:
        checker.check(False, f"{card}/gp_step: median total/eval/ops not all numeric "
                             f"({total!r}, {eval_ms!r}, {ops_ms!r})")
    steps = block.get("steps")
    steps = steps if isinstance(steps, list) else []
    checker.check(block.get("reps") == len(steps),
                  f"{card}/gp_step: reps={block.get('reps')!r} != len(steps)={len(steps)}")


def check_digests(checker: Checker, cells, records, allow_mismatch: bool,
                  require_agreement: bool) -> None:
    for cell in cells:
        present = [arm for (cid, arm) in records if cid == cell]
        for name in VALUE_WORKLOADS:
            digests = {}
            for arm in present:
                entry = (records[(cell, arm)].get("workloads") or {}).get(name) or {}
                sha = (entry.get("digest") or {}).get("sha256")
                if sha:
                    digests[arm] = sha
            if not digests:
                checker.check(False, f"{cell}/{name}: no digest anywhere in the cell")
                continue
            if require_agreement and len(digests) < 2:
                checker.check(False,
                              f"{cell}/{name}: digest agreement not demonstrated "
                              f"(only {sorted(digests)} carries one)")
                continue
            distinct = sorted(set(digests.values()))
            message = (f"{cell}/{name}: {len(distinct)} distinct sha256 across arms: "
                       + ", ".join(f"{sha[:12]}={sorted(a for a, s in digests.items() if s == sha)}"
                                   for sha in distinct))
            if allow_mismatch:
                checker.check(True, message)
                if len(distinct) > 1:
                    checker.warn(message + " (allowed by --allow-digest-mismatch)")
            else:
                checker.check(len(distinct) == 1, message)
        # nan / finite placement must agree on EVERY workload, sr_fitness included
        workloads = set()
        for arm in present:
            workloads.update((records[(cell, arm)].get("workloads") or {}).keys())
        for name in sorted(workloads):
            nans, finites = {}, {}
            for arm in present:
                entry = (records[(cell, arm)].get("workloads") or {}).get(name) or {}
                digest = entry.get("digest") or {}
                nans[arm] = digest.get("nan")
                finites[arm] = digest.get("finite")
            checker.check(len(set(nans.values())) == 1 and len(set(finites.values())) == 1,
                          f"{cell}/{name}: nan/finite counts differ across arms "
                          f"(nan={nans}, finite={finites})")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True, help="result directory to gate")
    parser.add_argument("--expect-arms", default=None,
                        help="comma-separated arms every cell must have "
                             "(default: the arms in campaign.json)")
    parser.add_argument("--require-digest-agreement", action="store_true",
                        help="demand >= 2 arms per cell so agreement is demonstrated, not vacuous")
    parser.add_argument("--allow-missing", action="store_true",
                        help="warn instead of failing on a scheduled record that is not written yet")
    parser.add_argument("--allow-digest-mismatch", action="store_true",
                        help="warn instead of failing when arms disagree on a value-path sha256")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.results):
        raise SystemExit(f"--results {args.results!r} is not a directory")

    checker = Checker()
    campaign, campaign_error, configs, pairs = discover(args.results)
    checker.check(campaign is not None,
                  f"campaign.json unreadable or absent ({campaign_error or 'missing'})")
    if campaign_error:
        checker.warn(f"campaign.json: {campaign_error}")

    # -- the schedule: an ok config must have its record --------------------
    scheduled = {key for key, entry in configs.items() if entry.get("status") == "ok"}
    not_ok = [(key, entry) for key, entry in configs.items() if entry.get("status") != "ok"]
    for (cell, arm), entry in not_ok:
        checker.warn(f"{cell}/{arm}: campaign status {entry.get('status')!r} "
                     f"({entry.get('detail') or 'no detail'}) — excluded from the gate")

    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cell, arm in sorted(pairs):
        entry = configs.get((cell, arm))
        status = (entry or {}).get("status")
        path = record_path(args.results, cell, arm, entry)
        if not os.path.isfile(path):
            message = f"{cell}/{arm}: record file missing ({path})"
            if (cell, arm) in scheduled and not args.allow_missing:
                checker.check(False, message)
            else:
                # not a gate failure: either the caller allowed it or the config
                # itself is not an ``ok`` one (its status is warned about above)
                checker.check(True, message)
                checker.warn(f"{message} ({'allowed' if args.allow_missing else status})")
            continue
        payload, error = load_json(path)
        if not isinstance(payload, dict):
            checker.check(False, f"{cell}/{arm}: record does not parse ({error})")
            continue
        records[(cell, arm)] = payload
        if (cell, arm) in scheduled:
            checker.check(True, f"{cell}/{arm}: scheduled ok config has a record")
        if status and status != "ok":
            checker.warn(f"{cell}/{arm}: a record exists although campaign status is {status!r}")

    checker.check(bool(records), f"no records in {args.results}")

    # -- every expected arm, per cell ---------------------------------------
    expected = None
    if args.expect_arms:
        expected = [arm.strip() for arm in args.expect_arms.split(",") if arm.strip()]
    elif configs:
        expected = [arm for arm in common.ALL_ARMS
                    if any(key[1] == arm for key in configs)]
        expected += sorted({key[1] for key in configs} - set(common.ALL_ARMS))
    if expected:
        cells = sorted({cell for cell, _ in records} | {cell for cell, _ in pairs})
        for cell in cells:
            have = {arm for cid, arm in records if cid == cell}
            missing = [arm for arm in expected if arm not in have]
            statuses = {arm: (configs.get((cell, arm)) or {}).get("status", "never scheduled")
                        for arm in missing}
            if not missing:
                checker.check(True, f"{cell}: all {len(expected)} expected arms present")
            elif args.allow_missing:
                checker.check(True, f"{cell}: missing arms {missing}")
                checker.warn(f"{cell}: missing arms (allowed) {missing} — campaign status: "
                             f"{statuses}")
            else:
                checker.check(False, f"{cell}: missing arms {missing} — campaign status: "
                                     f"{statuses} (present: {sorted(have)})")

    # -- per record ---------------------------------------------------------
    for (cell, arm), record in sorted(records.items()):
        card = f"{cell}/{arm}"
        check_env(checker, card, record)
        workloads = record.get("workloads") or {}
        if not workloads:
            checker.check(False, f"{card}: no workloads recorded")
        for name, entry in workloads.items():
            if isinstance(entry, dict):
                check_workload(checker, card, name, entry)
            else:
                checker.check(False, f"{card}/{name}: workload block is not an object")
        if is_evoxc_arm(arm):
            check_cache(checker, card, record)
        check_gp_step(checker, card, record)

    # -- cross-arm agreement and guard honesty -------------------------------
    check_digests(checker, sorted({cell for cell, _ in records}), records,
                  args.allow_digest_mismatch, args.require_digest_agreement)

    contaminated = [(key, entry) for key, entry in configs.items()
                    if isinstance(entry.get("guard"), dict) and entry["guard"].get("contaminated")]
    for (cell, arm), entry in contaminated:
        guard = entry.get("guard") or {}
        checker.warn(f"{cell}/{arm}: contaminated box (waited {guard.get('waited_s')} s, "
                     f"python={guard.get('python')}, "
                     f"gpu pids={(guard.get('gpu_pids') or {}).get('pids')})")
    guarded = [entry for entry in configs.values() if isinstance(entry.get("guard"), dict)]
    idle = [entry for entry in guarded if entry["guard"].get("ok")]

    for message in checker.warnings:
        print(f"WARN {message}")
    for message in checker.failures:
        print(f"FAIL {message}")
    print(f"[check] results={args.results}")
    if campaign is not None:
        print(f"[check] campaign: {len(configs)} configs, {len(scheduled)} ok, "
              f"{len(not_ok)} not ok, failures={campaign.get('failures')}, "
              f"elapsed={campaign.get('elapsed_s')} s")
    print(f"[check] guard: {len(idle)}/{len(guarded)} configs on an idle device, "
          f"{len(contaminated)} contaminated")
    print(f"{len(records)} records / {checker.checks} checks / {len(checker.failures)} failures")
    return 0 if not checker.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
