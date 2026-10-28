"""Numeric validation of every benchmark arm against the independent CPU oracle.

The oracle is ``src/evogp/tests/reference_eval.py`` (the plain-torch reference the
pytest suite diffs against).  For every validation cell this script evaluates the
same population with

  * ``evoxc``   -- ``Forest.batch_forward`` / ``Forest.forward`` / ``Forest.SR_fitness``
                   (the route selected by ``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND``),
  * ``per-tree``-- the ``torch-per-tree`` arm (Python tree walk, torch ops per node),
  * ``batched`` -- the ``torch-batched`` arm (one vectorized pass over the population),

and reports, per workload, the deviation of every arm against the oracle and against
the evoxc result: max abs / max rel deviation, NaN and Inf placement mismatches and a
bitwise-equality flag.  A non-bitwise-equal arithmetic-only result is a failing signal,
not something to smooth over -- the record is written verbatim.

Usage (repo root, project venv)::

    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \
        $PY test/benchmark/evoxc/validate.py --out test/benchmark/evoxc/records/validation.json

``--max-oracle N`` caps how many trees of a population are handed to the CPU oracle
(the population is sliced, so the record states how many trees were actually compared).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import common
import arms_torch


# --- comparison helpers -----------------------------------------------------
def compare(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    """Deviation of ``candidate`` from ``reference`` (both ``(pop, ...)`` float tensors)."""
    ref = reference.detach().to("cpu", torch.float32)
    cand = candidate.detach().to("cpu", torch.float32)
    out: Dict[str, Any] = {"shape": list(ref.shape)}
    if tuple(ref.shape) != tuple(cand.shape):
        out.update({"shape_mismatch": True, "candidate_shape": list(cand.shape)})
        return out
    both_nan = torch.isnan(ref) & torch.isnan(cand)
    nan_mismatch = int((torch.isnan(ref) != torch.isnan(cand)).sum().item())
    ref_inf, cand_inf = torch.isinf(ref), torch.isinf(cand)
    inf_mismatch = int(
        (ref_inf != cand_inf).sum().item()
        + (ref_inf & cand_inf & (ref != cand)).sum().item()
    )
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    n_finite = int(finite.sum().item())
    if n_finite:
        diff = (cand[finite] - ref[finite]).abs()
        denom = ref[finite].abs()
        rel = diff / torch.where(denom > 0, denom, torch.ones_like(denom))
        out["max_abs"] = float(diff.max().item())
        out["max_rel"] = float(rel.max().item())
        out["ref_abs_max"] = float(denom.max().item())
    else:
        out["max_abs"] = 0.0
        out["max_rel"] = 0.0
        out["ref_abs_max"] = 0.0
    out.update(
        {
            "finite_points": n_finite,
            "nan_points": int(torch.isnan(ref).sum().item()),
            "nan_mismatch": nan_mismatch,
            "inf_mismatch": inf_mismatch,
            "bitwise_equal": bool(
                nan_mismatch == 0
                and inf_mismatch == 0
                and bool(torch.equal(torch.where(both_nan, ref, cand), ref))
            ),
        }
    )
    return out


def _checks(prefix: str, reference: torch.Tensor, candidates: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return {f"{prefix}:{name}": compare(reference, value) for name, value in candidates.items()}


# --- one validation case ----------------------------------------------------
def validate_case(
    cell: Dict[str, Any],
    func_names: List[str],
    label: str,
    max_oracle: int,
    seed: int = 0,
) -> Dict[str, Any]:
    oracle_dir = os.path.join(common.src_root(), "evogp", "tests")
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    from reference_eval import (
        batch_forward_from_rows,
        forward_from_rows,
        sr_fitness_from_rows,
    )

    t0 = time.perf_counter()
    # a ternary function set needs a deeper node budget: the descriptor asserts
    # ``max_tree_len >=`` the fully-branching size, so shrink the layer count
    layer_cnt = 5 if "if" not in func_names else (3 if cell["tree_len"] >= 64 else 2)
    work = common.build_workload(
        cell, seed=seed, using_funcs=func_names, max_layer_cnt=layer_cnt
    )
    forest, x, labels, x_per_row = (
        work["forest"],
        work["x"],
        work["labels"],
        work["x_per_row"],
    )
    pop = forest.pop_size
    n_rows = int(x.shape[0])
    k = min(pop, max_oracle) if max_oracle > 0 else pop

    node_value = forest.batch_node_value[:k].to("cpu")
    node_type = forest.batch_node_type[:k].to("cpu")
    subtree_size = forest.batch_subtree_size[:k].to("cpu")
    x_cpu = x.to("cpu")
    x_row_cpu = x_per_row[:k].to("cpu")
    lab_cpu = labels.to("cpu").reshape(n_rows, -1)

    # ---- oracle (CPU, plain torch) ----
    oracle: Dict[str, torch.Tensor] = {}
    oracle["batch_forward"] = batch_forward_from_rows(
        node_value, node_type, subtree_size, x_cpu, out_len=forest.output_len
    )
    oracle["forward"] = forward_from_rows(
        node_value, node_type, subtree_size, x_row_cpu, out_len=forest.output_len
    )
    oracle["sr_fitness"] = sr_fitness_from_rows(
        node_value, node_type, subtree_size, x_cpu, lab_cpu, out_len=forest.output_len
    )

    # ---- the two pure-torch arms ----
    per = arms_torch.make_torch_arm("per_tree", forest)
    bat = arms_torch.make_torch_arm("batched", forest)
    arms: Dict[str, Dict[str, torch.Tensor]] = {
        "per-tree": {
            "batch_forward": per.batch_forward(x)[:k],
            "forward": per.forward(x_per_row)[:k],
            "sr_fitness": per.sr_fitness(x, labels)[:k],
        },
        "batched": {
            "batch_forward": bat.batch_forward(x)[:k],
            "forward": bat.forward(x_per_row)[:k],
            "sr_fitness": bat.sr_fitness(x, labels)[:k],
        },
    }
    setup_ms = {
        "per-tree": per.setup_ms,
        "batched": bat.setup_ms,
    }

    # ---- the evoxc engine through the public API ----
    engine: Dict[str, torch.Tensor] = {
        "batch_forward": forest.batch_forward(x)[:k],
        "forward": forest.forward(x_per_row)[:k],
        "sr_fitness": forest.SR_fitness(x, labels)[:k],
    }

    route = common.route_decision(pop, n_rows, os.environ.get("EVOGP_EVAL_MODE"), os.environ.get("EVOGP_EVOXC_BACKEND"))
    report: Dict[str, Any] = {
        "cell": cell["id"],
        "label": label,
        "funcs": list(func_names),
        "pop_size": pop,
        "trees_compared": k,
        "max_tree_len": int(forest.max_tree_len),
        "rows": n_rows,
        "output_len": int(forest.output_len),
        "route": route,
        "setup_ms": setup_ms,
        "checks": {},
    }
    checks = report["checks"]
    for workload in ("batch_forward", "forward", "sr_fitness"):
        ref = oracle[workload]
        checks.update(
            _checks(
                f"vs-oracle-{workload}",
                ref,
                {
                    "engine": engine[workload],
                    "per-tree": arms["per-tree"][workload],
                    "batched": arms["batched"][workload],
                },
            )
        )
        checks.update(
            _checks(
                f"vs-engine-{workload}",
                engine[workload],
                {
                    "per-tree": arms["per-tree"][workload],
                    "batched": arms["batched"][workload],
                },
            )
        )
    report["seconds"] = round(time.perf_counter() - t0, 3)
    return report


# --- campaign ---------------------------------------------------------------
DEFAULT_CELLS = ("p16-L32-i8-N1024", "p256-L64-i32-N1024", "p1000-L64-i32-N1024", "p256-L64-i32-N64")
FUNC_SETS = {
    # the campaign's function set: float32 +,-,*,/ are correctly rounded, so every
    # arm must agree with the oracle BITWISE
    "arith": ["+", "-", "*", "/"],
    # a wider set covering every op class (unary/ternary/comparison/loose op);
    # transcendental float32 kernels legitimately differ in the last bit
    "rich": ["+", "-", "*", "/", "max", "min", "<", "sin", "exp", "log", "abs", "sqrt", "loose_div", "loose_inv"],
    # ternary selection: the ``if`` node needs its own (shorter-layer) descriptor
    "ternary": ["if", "+", "-", "*", "/"],
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS), help="comma-separated cell names")
    ap.add_argument("--func-sets", default="arith,rich", help="comma-separated function-set names")
    ap.add_argument("--max-oracle", type=int, default=64, help="trees handed to the CPU oracle (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write the JSON record here")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    reports = []
    for name in args.cells.split(","):
        cell = common.parse_cell(name.strip())
        for fs_name in args.func_sets.split(","):
            fs_name = fs_name.strip()
            funcs = FUNC_SETS[fs_name]
            rep = validate_case(cell, funcs, fs_name, args.max_oracle, seed=args.seed)
            reports.append(rep)
            worst = max(
                (c.get("max_abs", 0.0) for c in rep["checks"].values() if isinstance(c, dict)),
                default=0.0,
            )
            bad = [k for k, c in rep["checks"].items() if not c.get("bitwise_equal", False)]
            print(
                f"[validate] {cell['id']:<22} {fs_name:<6} route={rep['route']['eval_mode']}/"
                f"{rep['route']['backend']} trees={rep['trees_compared']:<5} "
                f"worst|abs|={worst:.3e} non-bitwise={len(bad)} ({rep['seconds']:.1f}s)",
                flush=True,
            )
            for key in bad:
                c = rep["checks"][key]
                print(
                    f"    ! {key}: max_abs={c.get('max_abs', float('nan')):.3e} "
                    f"max_rel={c.get('max_rel', float('nan')):.3e} "
                    f"nan_mismatch={c.get('nan_mismatch')} inf_mismatch={c.get('inf_mismatch')}",
                    flush=True,
                )

    record = {
        "schema": common.SCHEMA + "-validation",
        "env": common.env_block(common.checkout_root(), arm="validate", extra={"note": args.note}),
        "protocol": {
            "oracle": "src/evogp/tests/reference_eval.py (plain torch, CPU)",
            "max_oracle_trees": args.max_oracle,
            "func_sets": {name: FUNC_SETS[name] for name in args.func_sets.split(",")},
            "seed": args.seed,
        },
        "cases": reports,
    }
    if args.out:
        common.write_json(args.out, record)
        print(f"[validate] wrote {args.out}")
    failures = sum(
        1
        for rep in reports
        for key, c in rep["checks"].items()
        if not c.get("bitwise_equal", False) and rep["label"] == "arith"
    )
    print(f"[validate] arith-set non-bitwise checks: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
