#!/usr/bin/env python
"""``dispatch_dump.py`` -- R5 Q2(a): the compiled dense route's GPU-side structure.

The question Q2 answers: the dense-iree **pure** forward of the 64/200-tree fixture
costs ~1.2-1.5 ms warm (Q4 same-window, ``results/q4_window/``) while the legacy
CUDA engine's *entire* evaluate is 0.65 ms -- what does the compiled graph
actually put on the GPU?  This probe produces the IREE-side evidence: the
**dispatch graph** and the **scheduling statistics** of the freshly compiled
dense route.

Method (analysis-only; ``src/evogp`` and the foreign ``etl`` are never edited):

* set ``ETL_IREE_COMPILE_ARGS`` (the documented etl env var -> the iree compile
  stage's ``iree_compile_args``; see ``etl/pipeline_options.py:123`` and
  ``etl/backends/adapters/iree.py:345/953``) to the dump flags, so the *first*
  graph call (which compiles) writes its dispatch graph, its scheduling
  statistics CSV and its per-executable binaries into ``--out-dir``;
* build the harness' exact route (``probe_lib.build_eval_case`` =
  ``bench_gp.py:768-806``; optional ``--data-rows`` tiling identical to
  ``bench_gp.py:760-768`` ``tile_rows``) and time the compile call + a few
  invocations, in the mode asked for:
    - ``--mode e2e``  : ``problem.evaluate(forest)`` (the headline route);
    - ``--mode pure`` : ``dense_engine.build_dense_runner(forest, x).run()``
      (the frozen pure runner of ``../pure_forward.py``);
* parse the dump files (edge/node counts, dispatch labels, csv header + row
  count) into a compact record so the raw dumps can stay small on disk.

Usage (repo root, project venv, GPU; route envs come from the shell):

    PYTHONPATH=src ETL_DISABLE_LOCATIONS=1 CUDA_VISIBLE_DEVICES=6 \\
      EVOGP_ETL_BACKEND=iree EVOGP_EVAL_MODE=dense EVOGP_ETL_IREE_DEVICE=0 \\
      EVOGP_ETL_IREE_OPT_LEVEL=O1 \\
      <venv>/bin/python test/benchmark/r5/gpu_probe/dispatch_dump.py \\
        --fixture test/goldens/forest_random.npz --mode pure --invocations 3 \\
        --out-dir test/benchmark/r5/results/q2_gpu/dumps/f64_pure \\
        --json-out test/benchmark/r5/results/q2_gpu/dumps/f64_pure.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.dirname(HERE)
if R5 not in sys.path:
    sys.path.insert(0, R5)

import probe_lib as pl  # noqa: E402

PROBE = "r5/gpu_probe/dispatch_dump.py"

#: The iree-compile flags the dump rides on.  All three are plain pass-through
#: flags for etl (``_IREE_DENIED_COMPILE_FLAGS`` denies only the f64-demotion and
#: the llvm-cpu flags), validated by the compiler itself.
DUMP_FLAGS = (
    "--iree-flow-dump-dispatch-graph",
    "--iree-flow-dump-dispatch-graph-output-file={dumps}/dispatch_graph.dot",
    "--iree-scheduling-dump-statistics-file={dumps}/scheduling_stats.csv",
    "--iree-scheduling-dump-statistics-format=csv",
    "--iree-hal-dump-executable-binaries-to={dumps}/executables",
)


def _tile_rows(array, rows):
    """``bench_gp.py:760-768`` ``tile_rows`` -- ceil-tile to exactly ``rows``."""
    import numpy as np

    array = np.asarray(array)
    src_rows = int(array.shape[0])
    if src_rows < 1:
        raise ValueError("cannot tile an array without rows")
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    repeats = -(-rows // src_rows)
    tiled = np.tile(array, (repeats,) + (1,) * (array.ndim - 1))[:rows]
    return np.ascontiguousarray(tiled)


def build_case(fixture, data_rows=None):
    """``probe_lib.build_eval_case`` + the harness' optional ``--data-rows`` tiling."""
    import numpy as np
    import torch

    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest

    if data_rows is None:
        return pl.build_eval_case(fixture)
    data = np.load(fixture)
    src_rows = int(np.asarray(data["x"]).shape[0])
    if int(data_rows) == src_rows:  # ``N == fixture rows`` stays a no-op (bench_gp policy)
        return pl.build_eval_case(fixture)
    x = pl.to_cuda(_tile_rows(data["x"], int(data_rows)), np.float32)
    labels = pl.to_cuda(_tile_rows(data["labels"], int(data_rows)), np.float32)
    forest = Forest(
        input_len=int(x.shape[1]),
        output_len=int(labels.shape[1]),
        batch_node_value=pl.to_cuda(data["forest__node_value"], np.float32),
        batch_node_type=pl.to_cuda(data["forest__node_type"], np.int16),
        batch_subtree_size=pl.to_cuda(data["forest__subtree_size"], np.int16),
    )
    problem = SymbolicRegression(datapoints=x, labels=labels)
    del torch  # imported for the side effect of the cuda context only
    return problem, forest, x, labels


def _node_labels(text):
    """Every quoted label of a DOT dump, in file order (deduplicated)."""
    labels = []
    for match in re.finditer(r'label\s*=\s*"((?:[^"\\]|\\.)*)"', text):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    return labels


_NODE_RE = re.compile(r'^\s*(v\d+)\s*\[[^\]]*label\s*=\s*"((?:[^"\\]|\\.)*)"', re.M)
_EDGE_RE = re.compile(r'^\s*(v\d+)\s*->\s*(v\d+)\s*\[', re.M)
_DISPATCH_SYMBOL_RE = re.compile(r"@([A-Za-z0-9_]+)::@([A-Za-z0-9_]+)")


def _longest_dispatch_chain(nodes, edges):
    """Longest path over the DAG counting only ``flow.dispatch`` nodes.
    The dumped graph is acyclic (SSA use-def), so a topological relaxation is exact.
    """
    order = {name: idx for idx, name in enumerate(nodes)}
    incoming = {name: [] for name in nodes}
    for src, dst in edges:
        if src in incoming and dst in incoming:
            incoming[dst].append(src)
    best = {}
    for name in sorted(nodes, key=lambda n: order[n]):
        weight = 1 if "flow.dispatch" in nodes[name] else 0
        best[name] = max((best[src] for src in incoming[name]), default=0) + weight
    return max(best.values(), default=0)


def parse_dispatch_graph(path):
    """Compact facts about the dispatch graph dump (raw file stays on disk)."""
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    lines = [line.rstrip("\n") for line in text.splitlines()]
    nodes = {name: label for name, label in _NODE_RE.findall(text)}
    edges = _EDGE_RE.findall(text)
    dispatch_nodes = {name: label for name, label in nodes.items() if "flow.dispatch" in label}
    symbols = []
    for label in dispatch_nodes.values():
        match = _DISPATCH_SYMBOL_RE.search(label)
        if match:
            symbols.append(f"@{match.group(2)}")
    return {
        "path": path,
        "exists": True,
        "bytes": os.path.getsize(path),
        "lines": len(lines),
        "nodes": len(nodes),
        "edges": len(edges),
        "dispatch_nodes": len(dispatch_nodes),
        "dispatch_symbols": symbols,
        "longest_dispatch_chain": _longest_dispatch_chain(nodes, edges),
        "node_label_sample": [label for label in dispatch_nodes.values()][:12],
        "head": "\n".join(lines[:18]),
    }


def parse_scheduling_stats(path):
    """Aggregate statistics + the per-command execution rows of the stats CSV."""
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    import csv
    import io

    sections = {}
    section = None
    header = None
    with open(path, "r", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(";"):
                if line.startswith("; ") and not line.startswith("; builtin"):
                    section = line[2:].strip()
                    header = None
                continue
            fields = next(csv.reader(io.StringIO(line)))
            if header is None:
                header = fields
                sections.setdefault(section, [])
                continue
            sections.setdefault(section, []).append(dict(zip(header, fields)))
    aggregate = (sections.get("Aggregate Statistics (static, whole-program)") or [{}])[0]
    execution = sections.get("Execution") or []
    return {
        "path": path,
        "exists": True,
        "bytes": os.path.getsize(path),
        "sections": {name: len(rows) for name, rows in sections.items() if name},
        "aggregate": aggregate,
        "execution_rows": len(execution),
        "execution": execution[:24],
    }


def list_executables(dir_path, budget_bytes=4 * 1024 * 1024):
    """Per-executable binary sizes; drop the payloads when they exceed the budget."""
    if not os.path.isdir(dir_path):
        return {"path": dir_path, "exists": False}
    files = []
    for root, _dirs, names in os.walk(dir_path):
        for name in sorted(names):
            full = os.path.join(root, name)
            files.append({"name": os.path.relpath(full, dir_path), "bytes": os.path.getsize(full)})
    total = sum(item["bytes"] for item in files)
    manifest = os.path.join(dir_path, "manifest.txt")
    with open(manifest, "w") as handle:
        for item in files:
            handle.write(f"{item['bytes']:>12d}  {item['name']}\n")
        handle.write(f"{total:>12d}  TOTAL ({len(files)} files)\n")
    pruned = False
    if total > budget_bytes:
        for item in files:
            os.remove(os.path.join(dir_path, item["name"]))
        pruned = True
    return {
        "path": dir_path,
        "exists": True,
        "files": files,
        "total_bytes": total,
        "pruned": pruned,
        "manifest": manifest,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="R5 Q2(a) dense-route dispatch dumps")
    parser.add_argument("--fixture", required=True, help="the eval fixture .npz")
    parser.add_argument("--out-dir", required=True, help="where the dump files are written")
    parser.add_argument("--mode", choices=("e2e", "pure"), default="pure")
    parser.add_argument("--invocations", type=int, default=3, help="timed calls after the compile")
    parser.add_argument("--data-rows", type=int, default=None, help="tile x/labels to N rows")
    parser.add_argument("--compile-args", default=None, help="override ETL_IREE_COMPILE_ARGS")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--log-out", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    compile_args = args.compile_args or " ".join(
        flag.format(dumps=os.path.abspath(args.out_dir)) for flag in DUMP_FLAGS
    )
    os.environ["ETL_IREE_COMPILE_ARGS"] = compile_args

    log_path = args.log_out or os.path.join(args.out_dir, "dump.log")
    with open(log_path, "w") as out:
        out.write(f"=== {PROBE} ===\n")
        out.write(f"ETL_IREE_COMPILE_ARGS={os.environ['ETL_IREE_COMPILE_ARGS']}\n")

        import etl
        import torch

        from evogp.tree import dense_engine

        problem, forest, x, labels = build_case(args.fixture, args.data_rows)
        torch.cuda.synchronize()
        out.write(
            f"route      : backend={dense_engine.resolve_backend_name()} "
            f"eval_mode={dense_engine.resolve_eval_mode()} "
            f"should_use_dense({int(forest.pop_size)})="
            f"{dense_engine.should_use_dense(int(forest.pop_size))}\n"
        )
        out.write(
            f"case       : fixture={args.fixture} pop={int(forest.pop_size)} "
            f"max_tree_len={int(forest.max_tree_len)} rows={int(x.shape[0])} "
            f"input_len={int(x.shape[1])} output_len={int(labels.shape[1])} mode={args.mode}\n"
        )

        compile_ms = None
        calls = []
        if args.mode == "e2e":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            problem.evaluate(forest)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000.0
            for _ in range(args.invocations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                problem.evaluate(forest)
                torch.cuda.synchronize()
                calls.append((time.perf_counter() - t0) * 1000.0)
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            runner = dense_engine.build_dense_runner(forest, x)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000.0
            for _ in range(args.invocations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                runner.run()
                torch.cuda.synchronize()
                calls.append((time.perf_counter() - t0) * 1000.0)
            out.write(f"runner     : invocations/graph-calls per run = {runner.invocations}\n")
            try:
                plan = runner.plan
                out.write(
                    f"plan       : trace_len={plan.trace_len} pop_size={plan.pop_size} "
                    f"n_rows={plan.n_rows} peak_bytes={getattr(plan, 'peak_bytes', '<n/a>')} "
                    f"variant={getattr(plan, 'variant', '<n/a>')}\n"
                )
            except Exception as exc:  # diagnostics must never fail a dump
                out.write(f"plan       : <{type(exc).__name__}: {exc}>\n")

        out.write(f"compile call (first, cold): {compile_ms:.3f} ms\n")
        out.write("warm calls ms: " + " ".join(f"{v:.3f}" for v in calls) + "\n")

        graph = parse_dispatch_graph(os.path.join(args.out_dir, "dispatch_graph.dot"))
        stats = parse_scheduling_stats(os.path.join(args.out_dir, "scheduling_stats.csv"))
        executables = list_executables(os.path.join(args.out_dir, "executables"))

        out.write("\n--- dispatch graph ---\n")
        for key in ("path", "exists", "bytes", "lines", "nodes", "edges", "dispatch_nodes",
                    "longest_dispatch_chain"):
            out.write(f"{key}: {graph.get(key)}\n")
        if graph.get("exists"):
            out.write("dispatch symbols:\n")
            for symbol in graph["dispatch_symbols"]:
                out.write(f"  {symbol}\n")
            out.write("dispatch node labels:\n")
            for label in graph["node_label_sample"]:
                out.write(f"  {label!r}\n")
        out.write("\n--- scheduling statistics ---\n")
        out.write(f"sections: {stats.get('sections')}\n")
        out.write(f"aggregate: {stats.get('aggregate')}\n")
        out.write(f"execution rows: {stats.get('execution_rows')}\n")
        for row in stats.get("execution", [])[:24]:
            out.write(f"  {row}\n")
        out.write("\n--- hal executables ---\n")
        for item in executables.get("files", []):
            out.write(f"  {item['bytes']:>10d}  {item['name']}\n")
        out.write(f"total {executables.get('total_bytes')} bytes, pruned={executables.get('pruned')}\n")

        payload = {
            "probe": PROBE,
            "tag": args.tag,
            "config": {
                "fixture": os.path.abspath(args.fixture),
                "mode": args.mode,
                "data_rows": args.data_rows,
                "pop_size": int(forest.pop_size),
                "max_tree_len": int(forest.max_tree_len),
                "rows": int(x.shape[0]),
                "input_len": int(x.shape[1]),
                "output_len": int(labels.shape[1]),
                "invocations": args.invocations,
                "etl_iree_compile_args": compile_args,
            },
            "compile_call_ms": compile_ms,
            "warm_calls_ms": calls,
            "dispatch_graph": graph,
            "scheduling_stats": stats,
            "executables": executables,
            "stamp": pl.identity_stamp(
                etl,
                extra={
                    "probe": PROBE,
                    "tag": args.tag,
                    "etl_head_probe": pl._run(["git", "rev-parse", "HEAD"], cwd=pl.ETL_REPO),
                },
            ),
        }
        out.write(f"\nlog: {log_path}\n")
        if args.json_out:
            pl.write_json(args.json_out, payload)
            out.write(f"record: {args.json_out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
