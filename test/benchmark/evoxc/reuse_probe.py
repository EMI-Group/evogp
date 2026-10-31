"""Plan-key REUSE of the batch route: does a rebuilt-but-equal population re-merge or get served?

What this measures
------------------
``EVOGP_EVAL_MODE=batch`` keys its merged plans on two surfaces (``src/evogp/tree/batch_keys.py``):
the population's tensor IDENTITY tokens and the population's program CONTENT digest.  A timing number
cannot tell which of them answered a call, and that is exactly the question this script asks:

* call 1 evaluates ``Forest`` A (cold): the merged graph is compiled -- the expected multi-second cost
  of the route, recorded as measured;
* call 2 evaluates the SAME ``Forest`` object again: an IDENTITY fast path must answer it, so the
  content table is never consulted (``attempts`` 0, not "a content miss");
* call 3 evaluates ``Forest`` B -- a DIFFERENT object built from the same seed, i.e. the same
  evolution in fresh storage with no shared tensor -- against the same dataset.  On a tree with the
  content-addressed plan key this is an identity MISS that the content table answers: one decode to
  prove the programs, then no re-trace, no re-merge and no recompile;
* call 4 evaluates B again: the identity fast path must now answer for B as well.

Every call records both counter surfaces before -> after (``batch_cache_stats``: exec/plan;
``content.py``: the content table), the value digest of the result, the sync-bracketed call time and
-- when the route ships ``batch_stage_stats`` -- the per-stage split, so "it skipped trace/merge/
compile" is a measured statement (those stages did not run) rather than an inference from a fast call.
The ``verification`` block states each expectation as a derived fact, with ``null`` (never ``false``)
wherever the route exposes no counter that could decide it.

Run it against any tree
-----------------------
The whole point is a before/after that is two RECORDS, not two assertions: ``--tree-root`` selects the
checkout under measurement exactly like ``worker.py`` / ``batch_economics.py``, so the pre-round tree
(no content table, no stage timers) records ``content`` blocks with ``available: false`` and every
counter ``null`` while its exec/plan counters still show the rebuilt population paying a plan MISS.

One process per measurement
---------------------------
The compile cache, the executable memo and both plan tables live in PROCESS memory, so a sequence
measured twice in one process is not the same sequence: the driver spawns one child per (tree,
cell, row count) and the child measures exactly one sequence.  The default output directory is
``results/batch_opt_round1/reuse`` (this round's own evidence tree); ``results/final``,
``results/baseline`` and ``results/batch`` are frozen and refused.

Reproduce on the GPU box
------------------------
::

    PY=<venv>/bin/python
    export PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc
    export EVOXC_DISABLE_LOCATIONS=1

    # HEAD (the content-addressed plan key)
    $PY test/benchmark/evoxc/reuse_probe.py --label after

    # the pre-round tree, extracted outside the repo at commit 619cbf7
    $PY test/benchmark/evoxc/reuse_probe.py --label base --tree-root "$TMPDIR/evogp_preopt_619cbf7"

    # one config by hand (a child is a normal, self-contained record producer)
    $PY test/benchmark/evoxc/reuse_probe.py --phase child --cell p64-L64-i32-N64 \\
        --out "$TMPDIR/reuse.json"

``--dry-run`` prints the planned sequence without importing torch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common  # noqa: E402  (stdlib-only at import time)
import content  # noqa: E402  (torch-free, evogp-free: the content-key vocabulary)
import stages  # noqa: E402  (torch-free, evogp-free: the ten-stage vocabulary)
import worker as worker_mod  # noqa: E402  (counter deltas + structure context; torch-free at import)
from cold_start import install_tree_root  # noqa: E402  (the one tree-switch helper)

#: The cheap cell the round measures: pop 64 (one merge of 63 distinct functions, so chunking is inert
#: and the cold call is ~2.2 s), 64 data rows, 32 inputs -- small enough that the whole sequence is a
#: handful of seconds, wide enough that every structure is distinct.
DEFAULT_CELL = "p64-L64-i32-N64"
DEFAULT_TREE_ROOT = "src"
DEFAULT_EVOXC_ROOT = "/root/evoxc"
DEFAULT_SEED = 0
DEFAULT_LABEL = "after"
DEFAULT_OUT_DIR = os.path.join(_HERE, "results", "batch_opt_round1", "reuse")
DEFAULT_TIMEOUT = 900
DISABLE_LOCATIONS_ENV = "EVOXC_DISABLE_LOCATIONS"
#: Frozen evidence: never written in place (the round's own ``reuse/`` tree is the target).
FROZEN_RESULT_DIRS = ("final", "baseline", "batch")
MEASUREMENT = "batch-plan-key-reuse"
#: The arm whose route this measures: the shipped ``batch`` arm (iree, banded, default pool).
ARM = "evoxc-batch-iree"
SCHEMA = common.SCHEMA + "-batch-reuse"

#: The call sequence. ``(forest label, what this call is proving)`` -- fixed, so two columns are
#: comparable call by call and a reader never has to guess which population a counter delta belongs to.
SEQUENCE: Tuple[Tuple[str, str], ...] = (
    ("A", "cold"),
    ("A", "identity"),
    ("B", "rebuilt"),
    ("B", "identity"),
)
#: For every call index: what the counters must show if the route behaves as this round claims.
#: ``None`` in :func:`verdict` means "no counter could decide it" -- never ``False``.
EXPECTATIONS: Tuple[str, ...] = (
    "cold_compiles",
    "identity_call_skips_compile",
    "identity_call_leaves_content_untouched",
    "rebuilt_call_answers_identically",
    "rebuilt_call_reuses_a_plan",
    "rebuilt_call_skips_trace_merge_compile",
)

FORWARDED_ENV = (
    "EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_EVOXC_DEVICE", "EVOGP_EVOXC_OPT_LEVEL",
    "EVOGP_EVOXC_CACHE_SIZE", "EVOGP_DENSE_AUTO_MIN_POP", "EVOGP_DENSE_AUTO_MAX_ROWS",
    "EVOGP_EVOXC_ARTIFACT_DIR",
) + common.BATCH_ENV_VARS


# --- paths / output policy (pure: no torch, no evogp) -----------------------
def resolve_src(tree_root: str) -> str:
    """``<tree_root>/src`` when it exists, else ``tree_root`` (same rule as worker.py)."""
    root = os.path.abspath(tree_root)
    return os.path.join(root, "src") if os.path.isdir(os.path.join(root, "src")) else root


def record_path(out_dir: str, label: str) -> str:
    return os.path.join(out_dir, label, "reuse.json")


def log_path(out_dir: str, label: str) -> str:
    return os.path.join(out_dir, label, "reuse.log")


def assert_fresh_out(out_dir: str) -> None:
    """Refuse to write into the frozen evidence directories (they are never rewritten in place)."""
    resolved = os.path.abspath(out_dir)
    for name in FROZEN_RESULT_DIRS:
        frozen = os.path.join(_HERE, "results", name)
        if resolved == frozen or resolved.startswith(frozen + os.sep):
            raise SystemExit(
                f"--out-dir {resolved} is inside the frozen evidence directory {frozen}: those "
                "directories are never rewritten in place -- point --out-dir at the round's own "
                f"tree (default: {DEFAULT_OUT_DIR})"
            )


# --- the derived verdict (pure: exercised by the dry run and by the record) --
def _count(entry: Any, path: Sequence[str]) -> Optional[int]:
    """An int counter at ``path`` inside one call entry (``None`` when unmeasured)."""
    node: Any = entry
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return None if node is None or isinstance(node, bool) else int(node)


def _call(calls: Sequence[Any], index: int) -> Any:
    """The ``index``-th recorded call (``None`` when the sequence was cut short)."""
    return calls[index] if 0 <= index < len(calls) else None


def verdict(calls: Sequence[Any]) -> Dict[str, Any]:
    """Each expectation as a measured fact: ``True`` / ``False`` / ``None`` (= no counter could tell).

    ``None`` is the honest third answer: a pre-round tree ships no content table and no stage timers,
    so "content was reused" is not FALSE there -- it is unmeasurable, and saying ``False`` would read
    as "the route was asked and refused".  The distinguishing evidence for such a tree is the plan
    table's miss count, which is exactly what ``rebuilt_call_plan_misses`` states.
    """
    cold, identity, rebuilt, identity_b = (_call(calls, i) for i in range(4))
    result: Dict[str, Any] = {name: None for name in EXPECTATIONS}
    result["rebuilt_call_plan_misses"] = _count(rebuilt, ("cache", "batch", "batch_plan_misses"))
    result["rebuilt_call_compile_misses"] = _count(rebuilt, ("cache", "batch", "batch_exec_misses"))
    result["identity_call_plan_misses"] = _count(identity, ("cache", "batch", "batch_plan_misses"))
    compile_cold = _count(cold, ("cache", "batch", "batch_exec_misses"))
    if compile_cold is not None:
        result["cold_compiles"] = compile_cold > 0
    misses = result["identity_call_plan_misses"]
    if misses is not None:
        result["identity_call_skips_compile"] = misses == 0
    attempts = _count(identity, ("content", "attempts"))
    if attempts is not None:
        result["identity_call_leaves_content_untouched"] = attempts == 0
    cold_digest = (cold or {}).get("digest", {}).get("sha256")
    rebuilt_digest = (rebuilt or {}).get("digest", {}).get("sha256")
    if cold_digest is not None and rebuilt_digest is not None:
        result["rebuilt_call_answers_identically"] = cold_digest == rebuilt_digest
    content_hits = _count(rebuilt, ("content", content.HITS_KEY))
    content_misses = _count(rebuilt, ("content", content.MISSES_KEY))
    if content_hits is not None and content_misses is not None:
        result["rebuilt_call_reuses_a_plan"] = content_hits > 0 and content_misses == 0
    # A content hit means the plan was served: the trace/merge/compile stages must not have run.  Read
    # off the per-stage CALL counters (a call count, not a duration), so a fast call is not the claim.
    stage_calls = {stage: _count(rebuilt, ("stage_ms", f"batch_{stage}_calls"))
                   for stage in ("trace", "merge", "compile")}
    if result["rebuilt_call_reuses_a_plan"] is True and all(
            value is not None for value in stage_calls.values()):
        result["rebuilt_call_skips_trace_merge_compile"] = all(
            value == 0 for value in stage_calls.values())
    result["rebuilt_call_stage_calls"] = stage_calls
    result["rebuilt_call_measured"] = {
        "content_available": (rebuilt or {}).get("content", {}).get("available"),
        "stage_available": (rebuilt or {}).get("stage_ms", {}).get("available"),
        "identity_call_on_b": _count(identity_b, ("cache", "batch", "batch_plan_misses")),
    }
    return result


# --- dry run (torch-free by construction) -----------------------------------
def dry_run(args: argparse.Namespace, src: str) -> int:
    """Print the sequence, the record path and whether this tree ships the route at all."""
    for index, (forest, kind) in enumerate(SEQUENCE, 1):
        print(f"would call {index}/4: Forest {forest} ({kind}) -> same x/labels")
    print("record:", record_path(args.out_dir, args.label))
    print("cell:", args.cell, "seed:", args.seed, "arm:", ARM, "src:", src)
    if not common.batch_engine_present(src):
        print(f"WARNING: {src} has no {common.BATCH_MODULE_REL} -- EVOGP_EVAL_MODE=batch cannot run "
              "here and the child will fail its preflight without writing a record")
    return 0


# --- environment plumbing (driver side) -------------------------------------
def child_env(args: argparse.Namespace, src: str) -> Dict[str, str]:
    """The environment the child runs with: PYTHONPATH, the tracing switch, the route knobs."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath(src), os.path.abspath(args.evoxc_root), _HERE]
    )
    env[DISABLE_LOCATIONS_ENV] = os.environ.get(DISABLE_LOCATIONS_ENV, "1")
    for key in FORWARDED_ENV:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_driver(args: argparse.Namespace) -> int:
    """One child process, one record (``<out-dir>/<label>/reuse.json``)."""
    src = resolve_src(args.tree_root)
    if args.dry_run:
        return dry_run(args, src)
    assert_fresh_out(args.out_dir)
    env = child_env(args, src)
    out_path, log = record_path(args.out_dir, args.label), log_path(args.out_dir, args.label)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        sys.executable, os.path.abspath(__file__), "--phase", "child",
        "--cell", args.cell, "--out", out_path, "--tree-root", args.tree_root,
        "--seed", str(args.seed),
    ]
    if args.note:
        cmd += ["--note", args.note]
    started = time.time()
    status, detail = "ok", ""
    with open(log, "w", encoding="utf-8") as handle:
        logged = {key: value for key, value in env.items()
                  if key.startswith(("EVOGP_", "EVOXC_")) or key == "PYTHONPATH"}
        handle.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        handle.write("env: " + json.dumps(logged) + "\n\n")
        handle.flush()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
            detail = f"killed after --timeout {args.timeout}s -- see the log for how far it got"
        else:
            handle.write(proc.stdout)
            if proc.stderr:
                handle.write(proc.stderr)
            if proc.returncode != 0:
                status = "failed"
                tail = [ln for ln in (proc.stderr or proc.stdout or "").splitlines() if ln.strip()]
                detail = tail[-1] if tail else f"exit {proc.returncode}"
            elif not os.path.isfile(out_path):
                status, detail = "failed", "the child exited 0 but wrote no record"
            else:
                print(proc.stdout.strip(), flush=True)
    print(f"[reuse] {args.label} {args.cell} ({src}) -> {status} "
          f"({round(time.time() - started, 2)}s)" + (f" -- {detail}" if detail else ""), flush=True)
    return 1 if status != "ok" else 0


# --- child phase: ONE tree, ONE cell, ONE sequence --------------------------
def _forest_block(label: str, forest, signature: Dict[str, Any]) -> Dict[str, Any]:
    """Identity of one population object: its structure fingerprint and its storage addresses."""
    import numpy as np

    sizes = forest.batch_subtree_size[:, 0].detach().to("cpu").numpy().astype(np.int64)
    return {
        "label": label,
        "pop_size": int(forest.pop_size),
        "max_tree_len": int(forest.max_tree_len),
        "tree_len_mean": float(sizes.mean()),
        "tree_len_max": int(sizes.max()),
        "distinct_structures": int(signature["distinct_structures"]),
        "structure_signature": signature["signature"],
        # the identity tokens the route's first plan table is keyed on: two DISTINCT objects have
        # distinct buffers, which is why a rebuilt-but-equal population is an identity MISS
        "storage": {
            name: int(getattr(forest, name).data_ptr())
            for name in ("batch_node_value", "batch_node_type", "batch_subtree_size")
        },
    }


def _same_content(first, second) -> Dict[str, Any]:
    """Do two populations carry the same evolution in different storage?  (Measured, not assumed.)"""
    import torch

    fields = ("batch_node_value", "batch_node_type", "batch_subtree_size")
    equal = {name: bool(torch.equal(getattr(first, name), getattr(second, name)))
             for name in fields}
    distinct = {name: (int(getattr(first, name).data_ptr())
                       != int(getattr(second, name).data_ptr())) for name in fields}
    return {
        "fields_equal": equal,
        "fields_in_distinct_storage": distinct,
        "identical_content": all(equal.values()),
        "no_shared_storage": all(distinct.values()),
    }


def run_child(args: argparse.Namespace) -> int:
    """Measure the four-call reuse sequence on one tree and write its record.

    Imports ``torch``/``evogp`` here and nowhere else, so ``--help``, ``--dry-run`` and the driver stay
    runnable on a box without torch.  The batch preflight runs BEFORE the torch import.
    """
    if not args.out:
        raise SystemExit("--phase child needs --out")
    src = install_tree_root(args.tree_root)
    cell = common.parse_cell(args.cell)

    import arms as arms_mod  # noqa: E402  (harness arm namespace: env application)

    knobs = arms_mod.apply_arm_env(ARM, cell)  # one arm per process; the arm's knobs win
    is_batch = common.is_batch_arm(ARM)
    policy = common.batch_policy(src) if is_batch else None
    if policy is not None and not (policy.get("module_present") and policy.get("mode_supported")):
        raise SystemExit(
            f"EVOGP_EVAL_MODE=batch needs BOTH a <src>/evogp/tree/batch_engine.py module and 'batch' "
            f"in evogp.tree.dense_engine.EVAL_MODES; this tree provides "
            f"module_present={policy.get('module_present')} "
            f"declared_modes={policy.get('declared_modes')!r}. NO record was written: a record of "
            "another route under the batch label would be worse than a missing one."
        )

    import torch

    common.import_evogp()
    try:
        batch_module = common.installed_module("evogp.tree.batch_engine")
    except ImportError:  # the route can be absent on an older tree
        batch_module = None
    batch_stats = getattr(batch_module, "batch_cache_stats", None)
    batch_stats = batch_stats if callable(batch_stats) else None
    content_stats = content.stats_fn(batch_module)
    stage_stats = stages.stage_stats_fn(batch_module)
    stage_reset = stages.stage_reset_fn(batch_module)

    def batch_snapshot() -> Optional[Dict[str, Any]]:
        return batch_stats() if batch_stats is not None else None

    def content_snapshot() -> Optional[Dict[str, Any]]:
        return content_stats() if content_stats is not None else None

    def stage_snapshot() -> Optional[Dict[str, Any]]:
        return stage_stats() if stage_stats is not None else None

    # -- two populations with the same content, built independently ----------
    # ``torch.manual_seed(seed)`` + ``Forest.random_generate`` is the documented recipe, so building
    # the same cell twice yields the SAME evolution in two independent allocations -- which is the
    # only way to ask the question: an object that was never seen before, carrying a plan the route
    # has already merged.
    workload_a = common.build_workload(cell, args.seed)
    workload_b = common.build_workload(cell, args.seed)
    forest_a, forest_b = workload_a["forest"], workload_b["forest"]
    signature = common.population_signature(forest_a)
    # ONE dataset object for every call: the comparison must not be confounded by new data tensors.
    x, labels = workload_a["x"], workload_a["labels"]
    dataset = {
        "shape": list(x.shape), "input_len": int(cell["input_len"]), "rows": int(cell["rows"]),
        "shared_tensor_object": True,
        "x_data_ptr": int(x.data_ptr()), "labels_data_ptr": int(labels.data_ptr()),
        "identical_to_build_b": bool(torch.equal(x, workload_b["x"])),
    }

    calls: List[Dict[str, Any]] = []
    forests = {"A": forest_a, "B": forest_b}
    for index, (label, kind) in enumerate(SEQUENCE, 1):
        forest = forests[label]
        batch_before, content_before, stage_before = (
            batch_snapshot(), content_snapshot(), stage_snapshot())
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = forest.SR_fitness(x, labels)
        torch.cuda.synchronize()
        call_ms = (time.perf_counter() - started) * 1e3
        batch_after, content_after, stage_after = (
            batch_snapshot(), content_snapshot(), stage_snapshot())
        entry: Dict[str, Any] = {
            "call": index, "forest": label, "kind": kind,
            "call_ms": round(call_ms, 4),
            "digest": common.digest(loss),
            "cache": {"batch": {
                "before": batch_before, "after": batch_after,
                "batch_exec_hits": worker_mod._stats_delta(batch_before, batch_after,
                                                           "batch_exec_hits"),
                "batch_exec_misses": worker_mod.batch_miss_delta(batch_before, batch_after),
                "batch_plan_hits": worker_mod._stats_delta(batch_before, batch_after,
                                                           "batch_plan_hits"),
                "batch_plan_misses": worker_mod.batch_plan_miss_delta(batch_before, batch_after),
            }},
            "content": content.window_block(content_before, content_after),
            "stage_ms": stages.delta_block(stage_before, stage_after),
        }
        del loss
        calls.append(entry)

    # -- the two surfaces' counter names, so a null delta is checkable -------
    cache_keys = {
        "batch_cache_stats": list(worker_mod.stats_keys(
            calls[0]["cache"]["batch"]["before"], calls[0]["cache"]["batch"]["after"])),
        "batch_content_stats": list(content.COUNTER_KEYS) if content_stats is not None else [],
        "batch_stage_stats": list(stages.COUNTER_KEYS) if stage_stats is not None else [],
    }

    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measurement": MEASUREMENT,
        "env": common.env_block(args.tree_root, ARM,
                                extra={"applied_knobs": knobs.get("knobs") or {}, "cell": cell,
                                       "arm": ARM}),
        "cell": cell, "arm": ARM, "seed": args.seed,
        "protocol": {
            "one_process_per_measurement": True, "sync_bracketed": True,
            "sequence": [{"call": index, "forest": label, "kind": kind}
                         for index, (label, kind) in enumerate(SEQUENCE, 1)],
            "dataset": "one x/labels object shared by every call",
            "evoxc_disable_locations": os.environ.get(DISABLE_LOCATIONS_ENV),
            "content_stats_fn": content.STATS_FN_PATH,
            "content_stats_available": content_stats is not None,
            "stage_stats_fn": stages.STATS_FN_PATH,
            "stage_stats_available": stage_stats is not None,
            "stage_reset_available": stage_reset is not None,
        },
        "routes": {"sr_fitness": common.route_decision(
            cell["pop"], cell["rows"], knobs["eval_mode"], knobs["backend"])},
        "batch_policy": policy,
        "structures": signature,
        "forests": {
            "A": _forest_block("A", forest_a, signature),
            "B": _forest_block("B", forest_b, common.population_signature(forest_b)),
            "same_content": _same_content(forest_a, forest_b),
        },
        "dataset": dataset,
        "cache_keys": cache_keys,
        "calls": calls,
        "verification": verdict(calls),
        "notes": [
            "calls 1 and 3 are the two populations' FIRST evaluation (A cold, B rebuilt-but-equal); "
            "calls 2 and 4 are the identity fast path doing its job on each, and an identity hit "
            "leaves the content table untouched (attempts 0 -- not a content miss)",
            "a content hit is served only after ONE decode, which is what the content key needs to "
            "prove the programs; the merge, the trace and the compile are what it skips",
            "null, never false: an expectation the route exposes no counter for stays None "
            "(a pre-round tree has no content table and no stage timers)",
        ],
    }
    if content_stats is None:
        record["notes"].append(
            f"{content.STATS_FN_PATH}() is unavailable on this tree: every content counter is null "
            "(NOT zero) and the reuse verdict stays None for the content-keyed expectations. The "
            "measured before/after evidence is then the PLAN table: the rebuilt population pays a "
            "plan MISS (re-decode, re-trace, re-merge) instead of being served"
        )
    if stage_stats is None:
        record["notes"].append(
            f"{stages.STATS_FN_PATH}() is unavailable on this tree: every stage counter is null "
            "(NOT zero), so 'it skipped trace/merge/compile' cannot be read off stage call counts "
            "here -- the merge work is visible only through the plan/exec counters and call_ms"
        )
    if args.note:
        record["notes"].append(args.note)

    common.write_json(args.out, record)
    facts = record["verification"]
    print(f"[reuse] {cell['id']} content_available={content_stats is not None} "
          f"cold={calls[0]['call_ms']:.1f}ms identity={calls[1]['call_ms']:.1f}ms "
          f"rebuilt={calls[2]['call_ms']:.1f}ms identity_b={calls[3]['call_ms']:.1f}ms "
          f"reuse={facts['rebuilt_call_reuses_a_plan']} "
          f"plan_misses(rebuilt)={facts['rebuilt_call_plan_misses']} -> {args.out}", flush=True)
    return 0


# --- entry point ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", default=DEFAULT_CELL, help="the ONE cell to measure")
    parser.add_argument("--tree-root", default=DEFAULT_TREE_ROOT,
                        help="tree root whose src/ holds evogp (any checkout, e.g. the pre-round tree)")
    parser.add_argument("--evoxc-root", default=DEFAULT_EVOXC_ROOT, help="evoxc checkout on PYTHONPATH")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="round evidence directory (never results/final|baseline|batch); one "
                             "subdirectory per --label")
    parser.add_argument("--label", default=DEFAULT_LABEL,
                        help="column label for this tree (e.g. after, base) -> <out-dir>/<label>/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="torch.manual_seed for BOTH populations (identical content by design)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="child wall cap (s)")
    parser.add_argument("--dry-run", action="store_true", help="print the sequence, run nothing")
    parser.add_argument("--note", default=None, help="free-form note stored in the record")
    parser.add_argument("--phase", choices=("driver", "child"), default="driver",
                        help="internal: driver spawns the child; child measures the sequence")
    parser.add_argument("--out", default=None, help="child phase: the JSON record path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")
    if args.phase == "child":
        return run_child(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
