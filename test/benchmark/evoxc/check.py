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
* for the batch arms (``common.is_batch_arm``): ``routes["sr_fitness"]`` reports
  ``eval_mode == "batch"`` with ``kind == "batch"``, ``batch_policy.mode_supported``
  is true and ``batch_cache`` carries a boolean ``available``.  A first call that no
  cache counter attributes to a compile, a tree whose batch engine exposes no
  counters at all, or a ``batch_cache.stats_keys`` without the exec table's
  ``batch_exec_misses``, is a ``WARN`` -- one merged graph per population legitimately
  costs seconds to minutes to compile, so its magnitude is the measurement itself
  and is never bounded here;
* the batch route's per-stage blocks (``stages.py``) are self-consistent: a
  ``stage_ms``/``stage_totals`` block that says ``available: true`` while every stage
  counter is null, a block that says ``available: false`` while counters carry numbers,
  or a ``summary.compile_share`` without its ``available`` flag is a ``WARN``.  This is
  a consistency check on what the record claims, never a bound on the numbers (which
  are the measurement's business), and it stays silent for every record that carries
  no stage keys -- every frozen record, and every torch/dense/specialized arm;
* the batch route's content-table blocks (``content.py``: ``first_call.content``,
  ``sr_fitness.content``, ``generations[i].content``, ``summary.content`` and
  ``batch_cache.content_totals``) are self-consistent in the same way: an
  ``available: true`` block whose five ``batch_content_*`` counters are all null, or an
  ``available: false`` block whose counters carry numbers, is a ``WARN``.  Also an
  optional surface (null, never zero), so it too stays silent for a record with no
  content keys and never fails a gate;
* digest agreement: all arms of one cell share one ``batch_forward`` and one
  ``forward`` sha256, and every workload's ``nan``/``finite`` counts agree across
  arms (``--allow-digest-mismatch`` downgrades the sha256 half to a warning);
* ``gp_step`` sanity: ``median_total_ms >= median_eval_ms``, ``median_ops_ms >= 0``,
  ``reps == len(steps)`` and ``median_total_ms`` re-derived as the upper median of
  ``steps[].total_ms`` (the statistic ``worker.py`` stores) matches the stored value;
* guard honesty: a config whose ``guard.contaminated`` is true is reported as a
  ``WARN`` (never a failure -- the box is shared) and the number of configs that
  ran on an idle device is printed.

Two things it deliberately does not soften: the ``median_ms`` re-computation allows
only the 4-decimal rounding ``worker.py`` applies to ``samples_ms``/``median_ms``
(1e-6 relative, floored at 1.5e-4 ms), and an empty ``env.tree_root_sha`` fails --
a ``git-archive`` snapshot tree root has no HEAD, so a replay campaign must be
quoted with that known caveat, or gated with ``--allow-snapshot`` (warn instead of
fail, but only when the recorded ``env.tree_root`` really has no ``.git``).

``--results`` takes either a single result directory (``<results>/<cell>/<arm>.json``
plus ``campaign.json``) or one that groups them: a directory whose subdirectories are
themselves result directories (``results/final/{after,before}/``) is gated directory
by directory in one run, and the exit code is 1 if any of them fails.

Usage (repository root, stdlib only, GPU-free)::

    python3 test/benchmark/evoxc/check.py --results test/benchmark/evoxc/results/baseline

    python3 test/benchmark/evoxc/check.py --results results/w1 \\
        --expect-arms torch-per-tree,torch-batched,evoxc-auto,evoxc-dense-iree \\
        --require-digest-agreement

    # a before/after round: every subdirectory that holds records is gated
    python3 test/benchmark/evoxc/check.py --results results/final --allow-snapshot

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
import stages  # noqa: E402
import content  # noqa: E402

#: frozen record schema (``common.SCHEMA``; spelled out so a schema edit here is a
#: deliberate act rather than a silent follow)
SCHEMA = "evogp-bench-evoxc-v1"
VALUE_WORKLOADS = ("batch_forward", "forward")
MIN_SAMPLES = 5
#: ``worker.py`` rounds ``samples_ms`` and every statistic to 4 decimals, so a
#: re-computed median can differ from the stored one by up to ~1e-4 ms.
ROUNDING_FLOOR_MS = 1.5e-4
MEDIAN_RTOL = 1e-6
#: The batch route's exec-table counter (``common.BATCH_EXEC_MISS_KEY``): the one whose
#: miss really recompiles, so a record's ``batch_cache.stats_keys`` should carry it.
BATCH_EXEC_MISS_KEY = common.BATCH_EXEC_MISS_KEY
#: ``stages.py`` owns the per-stage vocabulary: a stage block carries either the counter
#: keys (``batch_<stage>_{calls,total_ms,max_ms}`` plus the compile pool's two) with an
#: ``available`` flag, or -- for ``summary.compile_share`` -- a ``share`` next to the
#: compile stage's and the loop's own ms.  Read off ``stages`` so the names are never
#: spelled out twice.
STAGE_COUNTER_KEYS = stages.COUNTER_KEYS
COMPILE_SHARE_KEYS = ("share", "compile_ms", "loop_ms")
#: ``content.py`` owns the content-addressed plan-key vocabulary (``batch_content_hits`` /
#: ``_misses`` / ``_rejects`` / ``_entries`` / ``_maxsize``); the walker below reads the names off
#: it so a block is recognized by the counters it carries, never by the path it sits at.
CONTENT_COUNTER_KEYS = content.COUNTER_KEYS


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


def upper_median(values) -> Optional[float]:
    """``sorted(values)[len(values) // 2]`` -- the median statistic ``worker.py`` stores.

    ``gp_step``'s ``median_*_ms`` fields (and the warm ``median_ms``) are the upper
    median of the per-step values, not the mean of the two middle ones, so the
    re-derivation has to use the same rule.
    """
    ordered = sorted(value for value in values if is_number(value))
    return ordered[len(ordered) // 2] if ordered else None


def tree_root_is_snapshot(tree_root: Any) -> bool:
    """A ``git archive`` export has no ``.git``, so it has no HEAD to name."""
    if not isinstance(tree_root, str) or not tree_root:
        return False
    return not os.path.exists(os.path.join(tree_root, ".git"))


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


def holds_records(path: str) -> bool:
    """``<cell>/<arm>.json`` under ``path``: a subdirectory that is not itself a result dir."""
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return False
    for name in names:
        sub = os.path.join(path, name)
        if not os.path.isdir(sub) or os.path.isfile(os.path.join(sub, "campaign.json")):
            continue
        try:
            if any(fname.endswith(".json") for fname in os.listdir(sub)):
                return True
        except OSError:
            continue
    return False


def is_result_dir(path: str) -> bool:
    """A result directory holds ``campaign.json`` or ``<cell>/<arm>.json`` records.

    A ``<results>/<cell>/`` directory holds ``<arm>.json`` files directly and no
    subdirectories, and a directory that only groups result directories
    (``results/final/{after,before}/``) holds no records of its own, so this is what
    tells a result directory apart from its cells and from its container.
    """
    if not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "campaign.json")) or holds_records(path)


def gate_targets(results_dir: str) -> List[str]:
    """The directories one ``--results`` argument gates.

    ``<results>`` itself when it is a result directory (the single-directory layout,
    today's behaviour).  A directory that merely *groups* result directories
    (``results/final/{after,before}/``) is not gated itself -- it holds no records --
    and every immediate subdirectory that is a result directory is gated instead.
    """
    children = [os.path.join(results_dir, name)
                for name in sorted(os.listdir(results_dir))
                if is_result_dir(os.path.join(results_dir, name))]
    if is_result_dir(results_dir) or not children:
        return [results_dir]
    return children


# ---------------------------------------------------------------------------
# per-record checks
# ---------------------------------------------------------------------------
def check_env(checker: Checker, card: str, record: Dict[str, Any],
              allow_snapshot: bool = False) -> None:
    env = record.get("env") or {}
    checker.check(env.get("schema") == SCHEMA,
                  f"{card}: env.schema={env.get('schema')!r} != {SCHEMA!r}")
    tree_root, sha = env.get("tree_root"), env.get("tree_root_sha")
    if sha:
        checker.check(True, f"{card}: env.tree_root_sha={sha}")
    elif allow_snapshot and tree_root_is_snapshot(tree_root):
        # a git-archive export has no HEAD: the provenance is the archived SHA,
        # which the record cannot know -- warn so it is quoted, never fail it
        checker.check(True, f"{card}: env.tree_root_sha is null for the snapshot tree root")
        checker.warn(f"{card}: no env.tree_root_sha (tree_root={tree_root!r} has no .git) — "
                     "a snapshot's numbers must be quoted with the archived SHA "
                     "(--allow-snapshot)")
    else:
        hint = (" — the tree root is a git-archive snapshot, rerun with --allow-snapshot"
                if tree_root_is_snapshot(tree_root) else "")
        checker.check(False, f"{card}: env.tree_root_sha is empty ({sha!r}){hint}")
    for key in ("torch", "gpu"):
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


def check_batch(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    """The invariants that make a batch-route record trustworthy (batch arms only).

    A batch record is the only one whose cold first call is a merged-graph compile
    for the whole population, so the magnitude of ``first_call_ms`` and of the miss
    counters is the measurement, never a defect: this checks provenance (the cell
    really resolved to the batch route and the tree really supports it) and
    attribution (the merged-graph compile shows up in at least one counter), and it
    only ``warn``s when attribution is missing.  Called exclusively for
    ``common.is_batch_arm`` records, so it is a no-op for every other arm.
    """
    policy = record.get("batch_policy")
    supported = isinstance(policy, dict) and policy.get("mode_supported") is True
    declared = policy.get("declared_modes") if isinstance(policy, dict) else None
    checker.check(
        supported,
        f"{card}: batch arm but batch_policy.mode_supported is not True — this record "
        f"cannot be a batch-route measurement.  The route needs EVOGP_EVAL_MODE=batch, "
        f"src/evogp/tree/batch_engine.py present, and 'batch' declared in "
        f"evogp.tree.dense_engine.EVAL_MODES (declared_modes={declared!r}, "
        f"module_present={(policy or {}).get('module_present')!r})",
    )

    route = (record.get("routes") or {}).get("sr_fitness")
    route = route if isinstance(route, dict) else {}
    checker.check(route.get("eval_mode") == "batch",
                  f"{card}: routes.sr_fitness.eval_mode={route.get('eval_mode')!r} "
                  "!= 'batch' (the cell did not resolve to the batch route)")
    checker.check(route.get("kind") == "batch",
                  f"{card}: routes.sr_fitness.kind={route.get('kind')!r} != 'batch' "
                  "(a batch route decision must say so explicitly)")

    cache = record.get("batch_cache")
    checker.check(isinstance(cache, dict) and isinstance(cache.get("available"), bool),
                  f"{card}: batch_cache is not an object with a boolean 'available' "
                  f"({cache!r})")

    # -- cold-compile attribution: WARN only ---------------------------------
    # The merged-graph compile must be visible in the evoxc cache counter, the
    # batch engine's own counter, or both; a route whose counters do not cover it
    # is still a valid record (``first_call_ms`` carries the cost), so this never
    # fails the gate and never bounds the magnitude of a miss count.
    if not isinstance(cache, dict) or cache.get("available") is False:
        checker.warn(f"{card}: batch_cache.available is false — this tree's "
                     "evogp.tree.batch_engine exposes no merged-graph cache counters, so "
                     "the compile is visible only through first_call_ms (NOT zero misses)")
        return
    misses = 0
    for entry in (record.get("workloads") or {}).values():
        if not isinstance(entry, dict):
            continue
        counters = entry.get("cache") or {}
        for key in ("compile_misses_first_call", "batch_compile_misses_first_call"):
            value = counters.get(key)
            if is_number(value) and value > 0:
                misses += int(value)
    if misses < 1:
        checker.warn(f"{card}: 0 first-call compile misses across both caches — the "
                     "merged-graph compile was not attributed to any cache counter "
                     "(the counters may not cover this route); first_call_ms still carries it")

    # -- does the record say which counters the route exposes? (WARN only) ---
    # ``batch_cache.stats_keys`` is the record's own statement of the counter names
    # ``batch_cache_stats()`` returned.  The exec table (``batch_exec_misses``) is the
    # one that really recompiles, so a key set without it means the compile delta was
    # read off something else (or not at all) -- reportable, never a gate failure.
    keys = cache.get("stats_keys")
    if isinstance(keys, list) and keys and BATCH_EXEC_MISS_KEY not in keys:
        checker.warn(f"{card}: batch_cache.stats_keys={keys!r} carries no "
                     f"{BATCH_EXEC_MISS_KEY!r} — the exec table (the counter whose miss "
                     "recompiles) is not among the counters this route exposes, so the "
                     "compile deltas are read off a fallback key or are null")


def iter_dicts(node: Any, path: str = ""):
    """``(path, dict)`` for every object in a record, depth first (the path names it in a WARN)."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from iter_dicts(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_dicts(value, f"{path}[{index}]")


def stage_blocks(record: Dict[str, Any]):
    """Every per-stage block a record carries: ``(path, block, kind)``, ``kind`` a ``stages.py`` shape.

    The two shapes are the counter block (an ``available`` flag plus ``batch_<stage>_*``
    counters -- ``batch_cache.stage_totals``, ``gp_step.steps[i].stage_ms``,
    ``first_call.stage_ms``, ``sr_fitness.stage_ms`` and its ``.first_call``) and the
    compile-share block (``summary.compile_share``).  A record carrying neither -- every
    frozen record, and every torch/dense/specialized arm, which record no stage keys at
    all -- yields nothing, which is what keeps :func:`check_stages` silent there.
    """
    for path, block in iter_dicts(record):
        if "available" in block and any(key in block for key in STAGE_COUNTER_KEYS):
            yield path, block, "counters"
        elif all(key in block for key in COMPILE_SHARE_KEYS):
            yield path, block, "share"


def check_stages(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    """Self-consistency of a batch record's per-stage blocks (``stages.py``); WARN only.

    Modelled on :func:`check_batch`: the split is an OPTIONAL second counter contract, so
    a tree without ``batch_stage_stats()`` legitimately records ``available: false`` with
    every counter null -- that is never a failure, and a block's magnitudes are the
    measurement's business.  What is checked is that a block does not contradict itself:

    * ``available: true`` while every stage counter is null -- "measured" and "nothing
      measured" in one block;
    * ``available: false`` while counters carry numbers -- a measurement the block
      declares it does not have;
    * a ``compile_share`` block without its boolean ``available`` flag, i.e. a share
      whose provenance the record does not state.

    Deliberately no ``checker.check`` call: the gate's check count must stay exactly what
    it was for a directory of frozen records, which carry no stage keys at all.
    """
    for path, block, kind in stage_blocks(record):
        where = f"{card}.{path}" if path else card
        if kind == "share":
            if not isinstance(block.get("available"), bool):
                checker.warn(f"{where}: compile_share carries no boolean 'available' "
                             f"flag ({block.get('available')!r}) — a share whose provenance "
                             "the record does not state")
            continue
        available = block.get("available")
        numbers = [key for key in STAGE_COUNTER_KEYS if is_number(block.get(key))]
        if available is True and not numbers:
            checker.warn(f"{where}: available is true but every stage counter is null "
                         f"— the block claims a split it does not carry (null, NOT zero: it "
                         "should read available: false with a note)")
        elif available is False and numbers:
            checker.warn(f"{where}: available is false but {len(numbers)} stage "
                         f"counter(s) carry numbers ({sorted(numbers)[:3]}...) — the block "
                         "declares no split while its counters measured one")


def content_blocks(record: Dict[str, Any]):
    """Every content-table block a record carries: ``(path, block)``, ``content.py``'s shape.

    A content block is an ``available`` flag plus at least one ``batch_content_*`` counter --
    ``batch_cache.content_totals``, ``gp_step`` rows that carry one, ``first_call.content``,
    ``sr_fitness.content``, ``generations[i].content`` and ``summary.content`` (which also
    carries its ``generations`` outcome counts).  A record carrying none -- every frozen
    record, and every torch/dense/specialized arm, which record no content keys at all --
    yields nothing, which is what keeps :func:`check_content` silent there.
    """
    for path, block in iter_dicts(record):
        if "available" in block and any(key in block for key in CONTENT_COUNTER_KEYS):
            yield path, block


def check_content(checker: Checker, card: str, record: Dict[str, Any]) -> None:
    """Self-consistency of a batch record's content-table blocks (``content.py``); WARN only.

    Modelled on :func:`check_stages`: the content-addressed plan-key table is an OPTIONAL third
    counter contract, so a tree without ``batch_content_stats()`` legitimately records
    ``available: false`` with every counter null -- that is never a failure, and a block's
    magnitudes (a hit rate of 0 is a measurement) are the record's business.  What is checked
    is that a block does not contradict itself:

    * ``available: true`` while every content counter is null -- "measured" and "nothing
      measured" in one block (null, NOT zero: it should read ``available: false`` + a note);
    * ``available: false`` while counters carry numbers -- a measurement the block declares
      it does not have.

    Deliberately no ``checker.check`` call: the gate's check count must stay exactly what it
    was for a directory of frozen records, which carry no content keys at all.
    """
    for path, block in content_blocks(record):
        where = f"{card}.{path}" if path else card
        available = block.get("available")
        numbers = [key for key in CONTENT_COUNTER_KEYS if is_number(block.get(key))]
        if available is True and not numbers:
            checker.warn(f"{where}: available is true but every content counter is null "
                         f"— the block claims a measured plan-key table it does not carry "
                         "(null, NOT zero: it should read available: false with a note)")
        elif available is False and numbers:
            checker.warn(f"{where}: available is false but {len(numbers)} content "
                         f"counter(s) carry numbers ({sorted(numbers)[:3]}...) — the block "
                         "declares no content table while its counters measured one")


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
    derived_total = upper_median(step.get("total_ms") for step in steps if isinstance(step, dict))
    if derived_total is None:
        checker.check(False, f"{card}/gp_step: no numeric steps[].total_ms to re-derive "
                             "median_total_ms from")
    elif is_number(total):
        tolerance = max(MEDIAN_RTOL * abs(derived_total), ROUNDING_FLOOR_MS)
        checker.check(abs(total - derived_total) <= tolerance,
                      f"{card}/gp_step: median_total_ms={total!r} != "
                      f"median(steps[].total_ms)={derived_total!r} "
                      f"(tolerance {tolerance:.2e} ms)")


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
def gate_dir(results_dir: str, args) -> Tuple[Checker, Dict[str, Any]]:
    """Run every check over ONE result directory; return its checker and its facts."""
    checker = Checker()
    campaign, campaign_error, configs, pairs = discover(results_dir)
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
        path = record_path(results_dir, cell, arm, entry)
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

    checker.check(bool(records), f"no records in {results_dir}")

    # -- every expected arm, per cell ---------------------------------------
    expected = None
    if args.expect_arms:
        expected = [arm.strip() for arm in args.expect_arms.split(",") if arm.strip()]
    elif configs:
        # The expected set is the campaign's OWN scheduled arms, never the whole
        # registry: appending an arm to ``common.ALL_ARMS`` is therefore backwards
        # compatible, because an older campaign that never scheduled it keeps gating
        # exactly the arms it ran instead of failing on a record that cannot exist.
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
        check_env(checker, card, record, args.allow_snapshot)
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
        if common.is_batch_arm(arm):
            check_batch(checker, card, record)
            check_stages(checker, card, record)
            check_content(checker, card, record)
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

    facts = {"campaign": campaign is not None, "configs": len(configs),
             "scheduled": len(scheduled), "not_ok": len(not_ok),
             "campaign_failures": (campaign or {}).get("failures"),
             "elapsed_s": (campaign or {}).get("elapsed_s"),
             "guarded": len(guarded), "idle": len(idle), "contaminated": len(contaminated),
             "records": len(records)}
    return checker, facts


def print_block(results_dir: str, checker: Checker, facts: Dict[str, Any],
                header: Optional[str] = None) -> None:
    """One directory's WARN/FAIL lines and its summary (the single-dir output format)."""
    if header:
        print(header)
    for message in checker.warnings:
        print(f"WARN {message}")
    for message in checker.failures:
        print(f"FAIL {message}")
    print(f"[check] results={results_dir}")
    if facts["campaign"]:
        print(f"[check] campaign: {facts['configs']} configs, {facts['scheduled']} ok, "
              f"{facts['not_ok']} not ok, failures={facts['campaign_failures']}, "
              f"elapsed={facts['elapsed_s']} s")
    print(f"[check] guard: {facts['idle']}/{facts['guarded']} configs on an idle device, "
          f"{facts['contaminated']} contaminated")
    print(f"{facts['records']} records / {checker.checks} checks / "
          f"{len(checker.failures)} failures")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", required=True,
                        help="result directory to gate (a directory grouping result "
                             "directories is gated subdirectory by subdirectory)")
    parser.add_argument("--expect-arms", default=None,
                        help="comma-separated arms every cell must have "
                             "(default: the arms in campaign.json)")
    parser.add_argument("--require-digest-agreement", action="store_true",
                        help="demand >= 2 arms per cell so agreement is demonstrated, not vacuous")
    parser.add_argument("--allow-missing", action="store_true",
                        help="warn instead of failing on a scheduled record that is not written yet")
    parser.add_argument("--allow-digest-mismatch", action="store_true",
                        help="warn instead of failing when arms disagree on a value-path sha256")
    parser.add_argument("--allow-snapshot", action="store_true",
                        help="warn instead of failing on a null env.tree_root_sha when the "
                             "recorded tree root is a git-archive snapshot (no .git)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.results):
        raise SystemExit(f"--results {args.results!r} is not a directory")

    targets = gate_targets(args.results)
    multi = len(targets) > 1
    total_checks = total_records = total_failures = 0
    for target in targets:
        checker, facts = gate_dir(target, args)
        total_checks += checker.checks
        total_records += facts["records"]
        total_failures += len(checker.failures)
        print_block(target, checker, facts,
                    header=f"[check] == {target}" if multi else None)
    if multi:
        print(f"[check] {len(targets)} directories under {args.results} gated: "
              f"{total_records} records / {total_checks} checks / {total_failures} failures")
    return 0 if not total_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
