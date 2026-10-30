"""Dev battery for the FUSED (Triton) evaluation route — parity, goldens, timings, tuning.

This is a ``__main__``-guarded CLI, not a pytest module: it needs a GPU, it imports
the *harness* (``test/benchmark/evoxc``) and it is the only place where the fused
route's numeric contract and its economics are both stated.  Nothing imports it.

WHAT IT CHECKS
--------------
1. **Parity against the oracle.**  ``src/evogp/tests/reference_eval.py`` is the
   numeric contract of record (pure torch/numpy, written independently of the
   engines).  Four random populations cover the function-table families — the
   benchmark arithmetic set, the arithmetic+ternary set, the loose/protected set
   and a ``rich`` transcendental set — and a set of handmade adversarial rows
   covers the shapes a random generator does not produce: single-node leaves, a
   pure constant tree, a 40-deep left chain, a 40-deep right chain, full ternary
   ``if`` trees, a real tree followed by garbage padding, and single-output rows
   carrying ``OUT_NODE``-tagged nodes (which fall through to the ternary branch,
   exactly like the oracle's ``_kind_from_type``).  Each population is compared on
   all three surfaces: the batched value path (``per_row=False``), the per-tree
   value path (``per_row=True``) and the SR loss (MSE and MAE).
   Assertions: **bitwise** on the arithmetic-only set's value paths (that is the
   claim ``fused_kernels.py``'s ``enable_fp_fusion=False`` buys), ``<= 1e-6``
   relative everywhere else on the value path, ``<= 1e-5`` relative on the loss
   path (the kernel reduces in float32 per row-block, the oracle reduces in one
   float32 ``sum``, so the two differ by accumulation order only).
2. **Goldens.**  The two frozen fixture families of ``test/equivalence/test_goldens``
   (``src/evogp/tests/goldens/*.npz`` = this package's engine schema, and
   ``test/goldens/*.npz`` = outputs of the deleted CUDA kernels) at the documented
   ``1e-4`` rtol/atol.  Multi-output fixtures are expected to DECLINE (the fused
   route implements the single-output protocol only) and are reported as such.
3. **Timings.**  Six cells, warm medians (2 warmup + 10 sync-bracketed reps) of
   ``evaluate_fused`` / ``sr_fitness_fused`` against the ``torch-batched``
   reference arm (``test/benchmark/evoxc/arms_torch.BatchedEvaluator``), the cold
   first call (Triton import + JIT) and a stage split (kernel vs host glue vs the
   cold transpose).  One process per cell (``--stage timing`` spawns them), which
   is the harness's own protocol: a warm median is only comparable across
   processes if every process sees one shape class.
4. **Tuning.**  ``EVOGP_FUSED_BLOCK_ROWS`` in {64,128,256,512} x
   ``EVOGP_FUSED_NUM_WARPS`` in {1,2,4} on the same six cells, one process per
   (config, cell).  The default is chosen by the WORST cell (minimax warm ratio
   against the torch arm), because a route ships one default, not six.
5. **The route wiring.**  ``--stage wired`` drives the PUBLIC API
   (``Forest.forward`` / ``Forest.batch_forward`` / ``Forest.SR_fitness``) with
   ``EVOGP_EVAL_MODE=fused`` set in-process on three small cells and asserts the
   numbers equal the ``specialized`` reference (value bitwise or ``<= 1e-6``
   relative, loss ``<= 1e-5``), that the facade SERVED each call (its counters
   move), that ``resolve_route(pop, eval_mode="fused")`` names the fused engine on
   ``iree`` with a ``numpy`` fallback while ``pop_size == 1``, a population below
   ``EVOGP_FUSED_MIN_POP`` and a box without a device stay ``specialized``+``numpy``,
   that ``auto`` never selects it and that a multi-output population declines into
   the fallback without launching a kernel.  ``--stage timing --wired`` re-times
   the six cells through the same public calls.

HOW TO RUN
----------
From the repository root, with the torch-carrying interpreter and the harness on
the path (``TRITON_CACHE_DIR`` must be writable — the default ``~/.triton`` is not,
in the sandbox)::

    TRITON_CACHE_DIR=$TMPDIR/fused/tc \
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \
    /tmp/rocvenv/bin/python -m evogp.tree.fused_check --stage parity
    ... --stage goldens
    ... --stage declines
    ... --stage wired           # EVOGP_EVAL_MODE=fused through the public API
    ... --stage timing          # six cells, one subprocess each
    ... --stage timing --wired  # the same six cells, through the public API
    ... --stage tuning          # 4 x 3 configs x six cells
    ... --stage all

``--stage time-one --cell <id>`` / ``--stage tune-one --cell <id> --block B
--warps W`` are the single-process legs the two drivers spawn; they print one
``RESULT {...json...}`` line and are not meant to be run by hand except when a
cell misbehaves.

DEVIATIONS FROM THE ORACLE THAT ARE NOT BUGS
--------------------------------------------
* The oracle raises on a VAR index outside ``[0, input_len)``; the kernel cannot
  (a device read out of bounds would be undefined), so it masks the load with
  ``other=0.0``.  Every handmade row here keeps its VAR indices in range, which is
  what generation/splicing guarantee anyway.
* ``rows == 0`` and ``pop == 0`` are declines, not empty results — the fused route
  refuses to launch a zero-sized grid.
* ``FusedUnsupported`` is a *decline* (the facade's contract), so a cell that
  cannot be served reports a decline instead of a number and never a wrong number.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# paths: the module lives in <repo>/src/evogp/tree, so src/ and the test oracle
# are two levels up.  Adding them here keeps `python -m evogp.tree.fused_check`
# usable with only PYTHONPATH=src, while PYTHONPATH still wins for evogp itself.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TESTS_DIR = os.path.join(os.path.dirname(_HERE), "tests")
_REPO_ROOT = os.path.dirname(_SRC_ROOT)
for _path in (_TESTS_DIR,):
    if _path not in sys.path:
        sys.path.append(_path)

#: The six timing cells.  Ids come from the harness's own ``common.cell`` so a
#: number recorded here is addressable by the same name as every other round.
CELL_SPECS: List[Dict[str, Any]] = [
    dict(pop=256, tree_len=32, input_len=8, rows=1024, max_layer_cnt=5),
    dict(pop=256, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=256, tree_len=64, input_len=32, rows=4096, max_layer_cnt=5),
    dict(pop=1000, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=4096, tree_len=64, input_len=32, rows=1024, max_layer_cnt=5),
    dict(pop=2048, tree_len=128, input_len=32, rows=1024, max_layer_cnt=7),
]

#: ``using_funcs`` sets of the parity stage.  ``arithmetic`` is the only one whose
#: value path must be bitwise (no transcendental emits an fp64 round trip there).
PARITY_SETS: Dict[str, Dict[str, Any]] = {
    "arithmetic": dict(
        using_funcs=("+", "-", "*", "/"),
        input_len=8,
        max_layer_cnt=4,
        bitwise=True,
    ),
    "ternary": dict(
        using_funcs=("+", "-", "*", "/", "if"),
        input_len=4,
        max_layer_cnt=3,
        bitwise=False,
    ),
    "loose": dict(
        using_funcs=(
            "/",
            "loose_div",
            "loose_pow",
            "loose_log",
            "loose_inv",
            "loose_sqrt",
            "inv",
            "sqrt",
            "log",
        ),
        input_len=8,
        max_layer_cnt=4,
        bitwise=False,
    ),
    "rich": dict(
        using_funcs=(
            "sin",
            "cos",
            "tan",
            "tanh",
            "exp",
            "pow",
            "max",
            "min",
            "<",
            ">=",
            "+",
            "*",
        ),
        input_len=8,
        max_layer_cnt=4,
        bitwise=False,
    ),
}

#: Parity tolerances (see the module docstring).
VALUE_RTOL = 1e-6
LOSS_RTOL = 1e-5
GOLDEN_RTOL = 1e-4
GOLDEN_ATOL = 1e-4


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _json_line(payload: Dict[str, Any]) -> None:
    """One machine-readable line for the driver stages to parse."""
    print("RESULT " + json.dumps(payload, default=str), flush=True)


def _comparator():
    """The harness's ``validate.compare`` (the campaign's own deviation report).

    Imported lazily: it pulls the whole harness (``common``), which imports torch
    and evaluates cells at import time — irrelevant for the driver stages, so the
    child processes pay for it and the drivers do not.
    """
    try:
        from validate import compare  # type: ignore
    except Exception as exc:  # noqa: BLE001 - an environment error, never a silent pass
        raise SystemExit(
            "cannot import the harness comparison helper (validate.compare): "
            f"{exc!r}\n  add test/benchmark/evoxc to PYTHONPATH "
            "(e.g. PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc)"
        )
    return compare


def _harness():
    """The harness ``common`` module (cells, workloads, ``time_call``)."""
    try:
        import common  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cannot import the benchmark harness (common): {exc!r}\n  add "
            "test/benchmark/evoxc to PYTHONPATH"
        )
    return common


def _cells() -> List[Dict[str, Any]]:
    """The six timing cells, built through the harness so the ids are canonical."""
    common = _harness()
    return [
        common.cell(
            spec["pop"],
            spec["tree_len"],
            spec["input_len"],
            spec["rows"],
            spec["max_layer_cnt"],
        )
        for spec in CELL_SPECS
    ]


def _cell_by_id(cell_id: str) -> Dict[str, Any]:
    for cell in _cells():
        if cell["id"] == cell_id:
            return cell
    raise SystemExit(f"unknown cell id {cell_id!r}; known: {[c['id'] for c in _cells()]}")


def _report(tag: str, stats: Dict[str, Any], limit: float, *, key: str) -> bool:
    """Print one comparison row and return whether it stayed inside ``limit``."""
    ok = (stats.get("nan_mismatch", 1) == 0) and (stats.get("inf_mismatch", 1) == 0)
    if "shape_mismatch" in stats:
        ok = False
    rel = float(stats.get("max_rel", 0.0))
    ok = ok and rel <= limit
    mark = "ok " if ok else "FAIL"
    print(
        f"  [{mark}] {tag:<34} bitwise={str(stats.get('bitwise_equal')):<5} "
        f"max_abs={stats.get('max_abs', 0.0):.3e} max_rel={rel:.3e} "
        f"nan_mismatch={stats.get('nan_mismatch', 0)} "
        f"inf_mismatch={stats.get('inf_mismatch', 0)} "
        f"({key} <= {limit:g})",
        flush=True,
    )
    return ok


# --------------------------------------------------------------------------- #
# handmade trees (the adversarial rows of the parity stage)
# --------------------------------------------------------------------------- #
def _arity(function_id: int) -> int:
    import reference_eval as R

    return int(R.arity_of(function_id))


def _node_type_for(function_id: int) -> int:
    arity = _arity(function_id)
    return 2 if arity == 1 else (3 if arity == 2 else 4)


def build_row(spec: Sequence[Any]) -> Tuple[List[float], List[int], List[int]]:
    """Encode one nested node spec into ``(values, types, sizes)`` prefix lists.

    Spec grammar (deliberately tiny — the battery must not depend on the
    generator for the shapes it wants):
    ``("var", index)``, ``("const", value)``, ``(function_id, (child, ...))`` and
    ``("raw", value, node_type, (child, ...))`` for a hand-chosen node type (that is
    the only way to build an ``OUT_NODE``-tagged row, whose semantics are the
    oracle's ``_kind_from_type`` fall-through, not an operator).
    """
    values: List[float] = []
    types: List[int] = []
    sizes: List[int] = []

    def emit(node: Sequence[Any]) -> None:
        index = len(values)
        values.append(0.0)
        types.append(0)
        sizes.append(0)
        head = node[0]
        if head == "var":
            values[index] = float(node[1])
            types[index] = 0
        elif head == "const":
            values[index] = float(node[1])
            types[index] = 1
        elif head == "raw":
            values[index] = float(node[1])
            types[index] = int(node[2])
            for child in node[3]:
                emit(child)
        else:
            values[index] = float(head)
            types[index] = _node_type_for(int(head))
            for child in node[1]:
                emit(child)
        sizes[index] = len(values) - index

    emit(spec)
    return values, types, sizes


def pack_rows(
    rows: Sequence[Sequence[Any]],
    *,
    pad_to: int = 0,
    garbage_tail: bool = False,
):
    """Pack nested specs into the population tensors (``[pop, L]``) on the GPU.

    ``garbage_tail=True`` fills every dead cell behind ``subtree_size[0]`` with a
    NaN value, a nonsense type and a nonsense size — the kernel must never read it
    (it walks ``range(0, subtree_size[0])`` only), and the oracle's ``decode_row``
    does not either, so a divergence would be a real bug rather than a rounding.
    """
    import torch

    encoded = [build_row(row) for row in rows]
    length = max(len(values) for values, _, _ in encoded)
    length = max(length, int(pad_to), 1)
    pop = len(encoded)
    node_value = torch.zeros((pop, length), dtype=torch.float32)
    node_type = torch.zeros((pop, length), dtype=torch.int16)
    subtree_size = torch.zeros((pop, length), dtype=torch.int16)
    for row, (values, types, sizes) in enumerate(encoded):
        size = len(values)
        node_value[row, :size] = torch.tensor(values, dtype=torch.float32)
        node_type[row, :size] = torch.tensor(types, dtype=torch.int16)
        subtree_size[row, :size] = torch.tensor(sizes, dtype=torch.int16)
        if garbage_tail and size < length:
            node_value[row, size:] = float("nan")
            node_type[row, size:] = 5
            subtree_size[row, size:] = 1
    return node_value.cuda(), node_type.cuda(), subtree_size.cuda()


def _chain(depth: int, *, right: bool, input_len: int) -> Sequence[Any]:
    """A ``depth``-deep ``+`` chain over ``VAR`` leaves (right-leaning if asked).

    A left chain keeps one pending operand and a right chain keeps ``depth`` of
    them, so the pair pins the kernel's stack discipline from both sides.
    """
    from evogp.tree.utils import Func

    add = int(Func.ADD)
    node: Any = ("var", 0)
    for level in range(1, depth + 1):
        leaf = ("var", level % input_len)
        node = (add, (node, leaf)) if not right else (add, (leaf, node))
    return node


def _full_ternary(depth: int, input_len: int) -> Sequence[Any]:
    """A complete ternary ``if`` tree of ``depth`` levels (``(3**d - 1) / 2`` nodes)."""
    from evogp.tree.utils import Func

    if_node = int(Func.IF)
    if depth <= 1:
        return ("var", 0)
    return (
        if_node,
        (
            _full_ternary(depth - 1, input_len),
            ("var", (depth + 1) % input_len),
            _full_ternary(depth - 1, input_len),
        ),
    )


def adversarial_cases() -> List[Dict[str, Any]]:
    """Handmade populations: the shapes a random generator never produces."""
    from evogp.tree.utils import Func

    add, mul, div, sqrt, log_op = (
        int(Func.ADD),
        int(Func.MUL),
        int(Func.DIV),
        int(Func.SQRT),
        int(Func.LOG),
    )
    const_tree = (add, (("const", 1.5), (mul, (("const", 2.0), ("const", 3.0)))))
    mixed = (
        add,
        (
            (div, (("var", 0), (("const", 0.0)))),
            (mul, ((sqrt, (("var", 1),)), (log_op, (("var", 2),)))),
        ),
    )
    # OUT_NODE-tagged nodes: every tagged type (0x80|0 .. 0x80|4) must follow the
    # oracle's TFUNC fall-through, whatever its packed value means.
    # Every OUT_NODE-tagged type falls through to TFUNC, i.e. it is a TERNARY node
    # whatever its tag said -- so each one below carries three children, exactly as
    # the oracle's `_kind_from_type` demands.
    out_tagged = (
        "raw",
        1.5e9,
        0x82,
        (
            # a FINITE garbage value: the oracle casts a TFUNC node's value to int
            # (``int(nan)`` raises), and the whole point is that the value is ignored
            ("raw", 1234.5, 0x80, (("var", 0), ("var", 1), ("var", 2))),
            ("raw", 0.0, 0x84, (("const", 0.5), ("const", 0.25), ("var", 3))),
            ("raw", 3.0, 0x84, (("var", 0), ("var", 1), ("var", 2))),
        ),
    )
    return [
        dict(
            name="leaf_var",
            rows=[("var", 0)],
            input_len=4,
            note="single VAR root",
        ),
        dict(
            name="leaf_const",
            rows=[("const", 2.5)],
            input_len=4,
            note="single CONST root (never compiled on the specialized route)",
        ),
        dict(
            name="const_tree",
            rows=[const_tree],
            input_len=4,
            note="pure constant subtree",
        ),
        dict(
            name="left_chain_40",
            rows=[_chain(40, right=False, input_len=8)],
            input_len=8,
            note="40-deep left chain, one pending operand",
        ),
        dict(
            name="right_chain_40",
            rows=[_chain(40, right=True, input_len=8)],
            input_len=8,
            note="40-deep right chain, 40 pending operands",
        ),
        dict(
            name="full_if_3",
            rows=[_full_ternary(3, 4)],
            input_len=4,
            note="complete ternary if tree (13 nodes)",
        ),
        dict(
            name="padded_garbage",
            rows=[(add, (("var", 0), ("const", 1.25)))],
            input_len=4,
            pad_to=64,
            garbage_tail=True,
            note="5-node tree, NaN/nonsense padding behind it",
        ),
        dict(
            name="out_node_tags",
            rows=[out_tagged],
            input_len=4,
            note="OUT_NODE-tagged nodes -> TFUNC fall-through (out_len 1)",
        ),
        dict(
            name="mixed_deep",
            rows=[mixed],
            input_len=4,
            note="DIV by const 0 (NaN per the emitter), SQRT/LOG of a var",
        ),
        dict(
            name="mixed_bag",
            rows=[
                ("var", 0),
                ("const", -0.5),
                const_tree,
                _chain(12, right=False, input_len=4),
                _chain(12, right=True, input_len=4),
                _full_ternary(2, 4),
                out_tagged,
                mixed,
            ],
            input_len=4,
            pad_to=32,
            garbage_tail=True,
            note="all of the above in ONE population (pop 8)",
        ),
    ]


# --------------------------------------------------------------------------- #
# stage 1: parity vs the oracle
# --------------------------------------------------------------------------- #
def _oracle(rows: Tuple[Any, Any, Any], x, out_len: int = 1):
    import reference_eval as R

    node_value, node_type, subtree_size = rows
    return R.batch_forward_from_rows(
        node_value.cpu(), node_type.cpu(), subtree_size.cpu(), x.cpu(), out_len=out_len
    )


def _oracle_per_row(rows: Tuple[Any, Any, Any], x, out_len: int = 1):
    import reference_eval as R

    node_value, node_type, subtree_size = rows
    return R.forward_from_rows(
        node_value.cpu(), node_type.cpu(), subtree_size.cpu(), x.cpu(), out_len=out_len
    )


def _oracle_loss(rows: Tuple[Any, Any, Any], x, labels, use_mse: bool):
    import reference_eval as R

    node_value, node_type, subtree_size = rows
    return R.sr_fitness_from_rows(
        node_value.cpu(),
        node_type.cpu(),
        subtree_size.cpu(),
        x.cpu(),
        labels.cpu(),
        use_mse=use_mse,
        out_len=1,
    )


def _check_population(
    tag: str,
    rows: Tuple[Any, Any, Any],
    input_len: int,
    *,
    batch_x=None,
    per_row_x=None,
    labels=None,
    value_limit: float,
    note: str = "",
) -> Tuple[int, int]:
    """All three surfaces of one population.  Returns ``(passed, total)``."""
    import torch

    from evogp.tree import fused_engine

    compare = _comparator()
    node_value, node_type, subtree_size = rows
    passed = total = 0
    if note:
        print(f"  {tag}: {note}", flush=True)
    if batch_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, batch_x, per_row=False
        )
        want = _oracle(rows, batch_x)
        total += 1
        passed += _report(
            f"{tag}/batch_forward",
            compare(want, got),
            value_limit,
            key="value",
        )
        if labels is not None:
            for use_mse in (True, False):
                got_l = fused_engine.sr_fitness_fused(
                    node_value, node_type, subtree_size, input_len, 1, batch_x, labels,
                    use_MSE=use_mse,
                )
                want_l = _oracle_loss(rows, batch_x, labels, use_mse)
                total += 1
                passed += _report(
                    f"{tag}/sr_fitness{'MSE' if use_mse else 'MAE'}",
                    compare(want_l, got_l),
                    LOSS_RTOL,
                    key="loss",
                )
    if per_row_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, per_row_x, per_row=True
        )
        want = _oracle_per_row(rows, per_row_x)
        total += 1
        passed += _report(
            f"{tag}/forward(per_row)",
            compare(want, got),
            value_limit,
            key="value",
        )
    return passed, total


def stage_parity(args: argparse.Namespace) -> int:
    """Random op-set populations + the handmade adversarial rows."""
    import torch

    common = _harness()
    print("=" * 78)
    print("PARITY: fused engine vs src/evogp/tests/reference_eval.py")
    print("=" * 78)
    passed = total = 0
    for name, spec in PARITY_SETS.items():
        cell = {
            "pop": int(args.parity_pop),
            "tree_len": 32,
            "input_len": int(spec["input_len"]),
            "rows": int(args.parity_rows),
            "max_layer_cnt": int(spec["max_layer_cnt"]),
        }
        workload = common.build_workload(cell, using_funcs=list(spec["using_funcs"]))
        forest = workload["forest"]
        rows = (
            forest.batch_node_value,
            forest.batch_node_type,
            forest.batch_subtree_size,
        )
        print(f"\n[{name}] using_funcs={list(spec['using_funcs'])}")
        limit = 0.0 if spec["bitwise"] else VALUE_RTOL
        p, t = _check_population(
            name,
            rows,
            cell["input_len"],
            batch_x=workload["x"],
            per_row_x=workload["x_per_row"],
            labels=workload["labels"],
            value_limit=limit,
            note=(
                "bitwise required (arithmetic set)"
                if spec["bitwise"]
                else f"value <= {VALUE_RTOL:g} rel"
            ),
        )
        passed += p
        total += t
    print("\n[adversarial]")
    for case in adversarial_cases():
        node_value, node_type, subtree_size = pack_rows(
            case["rows"],
            pad_to=int(case.get("pad_to", 0)),
            garbage_tail=bool(case.get("garbage_tail", False)),
        )
        input_len = int(case["input_len"])
        pop = int(node_value.shape[0])
        torch.manual_seed(1234)
        batch_x = torch.rand(int(args.parity_rows), input_len, device="cuda")
        labels = torch.rand(int(args.parity_rows), 1, device="cuda")
        per_row_x = torch.rand(pop, input_len, device="cuda")
        p, t = _check_population(
            case["name"],
            (node_value, node_type, subtree_size),
            input_len,
            batch_x=batch_x,
            per_row_x=per_row_x,
            labels=labels,
            value_limit=VALUE_RTOL,
            note=case["note"],
        )
        passed += p
        total += t
    print("\n" + "-" * 78)
    print(f"PARITY: {passed}/{total} comparisons inside tolerance")
    print("-" * 78)
    return 0 if passed == total else 1


# --------------------------------------------------------------------------- #
# stage 2: goldens
# --------------------------------------------------------------------------- #
_ENGINE_KEYS = ("node_value", "node_type", "subtree_size")
_LEGACY_TREE = ("forest__node_value", "forest__node_type", "forest__subtree_size")


def _golden_dirs() -> List[str]:
    directories = [os.path.join(_REPO_ROOT, "test", "goldens"), os.path.join(_TESTS_DIR, "goldens")]
    extra = os.environ.get("EVOGP_GOLDEN_DIR")
    if extra:
        directories.append(extra)
    return [d for d in directories if os.path.isdir(d)]


def stage_goldens(args: argparse.Namespace) -> int:
    """Engine-schema and legacy-capture fixtures at 1e-4 rtol/atol."""
    import numpy as np
    import torch

    from evogp.tree import fused_engine

    compare = _comparator()
    passed = total = 0
    print("=" * 78)
    print("GOLDENS: fused engine vs the frozen fixtures (1e-4 rtol/atol)")
    print("=" * 78)
    for directory in _golden_dirs():
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".npz"):
                continue
            path = os.path.join(directory, name)
            with np.load(path) as data:
                keys = set(data.files)
                if set(_ENGINE_KEYS) <= keys:
                    passed, total = _golden_engine(
                        path, data, compare, fused_engine, torch, passed, total
                    )
                elif set(_LEGACY_TREE) <= keys:
                    passed, total = _golden_legacy(
                        path, data, compare, fused_engine, torch, passed, total
                    )
                else:
                    print(f"  [skip] {name}: unrecognized schema ({sorted(keys)[:6]}...)")
    print("-" * 78)
    print(f"GOLDENS: {passed}/{total} comparisons inside 1e-4")
    print("-" * 78)
    return 0 if passed == total else 1


def _as_cuda(values, dtype):
    import torch

    return torch.as_tensor(values, dtype=dtype).cuda()


def _golden_engine(path, data, compare, fused_engine, torch, passed, total):
    """``src/evogp/tests/goldens/*.npz``: batch_forward / SR_fitness expectations."""
    name = os.path.basename(path)
    x = _as_cuda(data["x"], torch.float32)
    node_value = _as_cuda(data["node_value"], torch.float32)
    node_type = _as_cuda(data["node_type"], torch.int16)
    subtree_size = _as_cuda(data["subtree_size"], torch.int16)
    input_len = int(x.shape[1])
    multi_output = "expected" in data and int(np_shape_last(data["expected"])) > 1
    if multi_output:
        print(f"  [skip] {name}: multi-output fixture (the fused route declines it)")
        return passed, total
    if "expected" in data:
        want = _as_cuda(data["expected"], torch.float32)
        try:
            got = fused_engine.evaluate_fused(
                node_value, node_type, subtree_size, input_len, 1, x, per_row=False
            )
        except fused_engine.FusedUnsupported as exc:
            print(f"  [FAIL] {name}/expected: declined ({exc})")
            total += 1
            return passed, total
        total += 1
        passed += _golden_report(f"{name}/expected", compare(want, got), name)
    if "expected_fitness" in data:
        labels = _as_cuda(data["labels"], torch.float32)
        use_mse = bool(int(np_scalar(data["use_mse"]))) if "use_mse" in data else True
        want = _as_cuda(data["expected_fitness"], torch.float32)
        try:
            got = fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1, x, labels,
                use_MSE=use_mse,
            )
        except fused_engine.FusedUnsupported as exc:
            print(f"  [FAIL] {name}/expected_fitness: declined ({exc})")
            total += 1
            return passed, total
        total += 1
        passed += _golden_report(
            f"{name}/expected_fitness(mse={use_mse})", compare(want, got), name
        )
    return passed, total


def _golden_legacy(path, data, compare, fused_engine, torch, passed, total):
    """``test/goldens/*.npz``: the frozen outputs of the deleted CUDA kernels."""
    name = os.path.basename(path)
    node_value = _as_cuda(data["forest__node_value"], torch.float32)
    node_type = _as_cuda(data["forest__node_type"], torch.int16)
    subtree_size = _as_cuda(data["forest__subtree_size"], torch.int16)
    x = _as_cuda(data["x"], torch.float32)
    input_len = int(x.shape[1])
    labels = _as_cuda(data["labels"], torch.float32) if "labels" in data else None
    checks = []
    if "forest_batch_forward" in data and "x_small" in data:
        checks.append(
            (
                f"{name}/forest_batch_forward",
                _as_cuda(data["forest_batch_forward"], torch.float32),
                _as_cuda(data["x_small"], torch.float32),
                False,
            )
        )
    if "forest_forward" in data:
        checks.append(
            (f"{name}/forest_forward", _as_cuda(data["forest_forward"], torch.float32), x, True)
        )
    for tag, want, xa, per_row in checks:
        try:
            got = fused_engine.evaluate_fused(
                node_value, node_type, subtree_size, input_len, 1, xa, per_row=per_row
            )
        except fused_engine.FusedUnsupported as exc:
            print(f"  [FAIL] {tag}: declined ({exc})")
            total += 1
            continue
        total += 1
        passed += _golden_report(tag, compare(want, got), name)
    if labels is not None and "forest_sr_fitness__auto__mse" in data:
        for key, use_mse in (
            ("forest_sr_fitness__auto__mse", True),
            ("forest_sr_fitness__auto__nonmse", False),
        ):
            if key not in data:
                continue
            want = _as_cuda(data[key], torch.float32)
            got = fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=use_mse
            )
            total += 1
            passed += _golden_report(f"{name}/{key}", compare(want, got), name)
    return passed, total


def _golden_report(tag: str, stats: Dict[str, Any], fixture: str) -> int:
    """1e-4 rtol/atol — the legacy capture used fast-math, so no bitwise claim."""
    ok = (
        (stats.get("nan_mismatch", 1) == 0)
        and (stats.get("inf_mismatch", 1) == 0)
        and "shape_mismatch" not in stats
        and float(stats.get("max_abs", 1e9)) <= GOLDEN_ATOL + GOLDEN_RTOL * float(
            stats.get("ref_abs_max", 0.0)
        )
    )
    rel = float(stats.get("max_rel", 0.0))
    absd = float(stats.get("max_abs", 0.0))
    print(
        f"  [{'ok ' if ok else 'FAIL'}] {tag:<44} max_abs={absd:.3e} max_rel={rel:.3e} "
        f"bitwise={stats.get('bitwise_equal')} nan_mismatch={stats.get('nan_mismatch', 0)}",
        flush=True,
    )
    return 1 if ok else 0


def np_shape_last(array) -> int:
    import numpy as np

    return int(np.asarray(array).shape[-1])


def np_scalar(array) -> float:
    import numpy as np

    return float(np.asarray(array).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# stage 2b: the decline matrix and the knob/stat contracts (cheap, no tuning)
# --------------------------------------------------------------------------- #
def stage_declines(args: argparse.Namespace) -> int:
    """``FusedUnsupported`` on every unsupported call, and the stats/knob contracts.

    The route's whole safety story is "decline, never a wrong number", so every
    decline reason is exercised here (one case per reason), together with the
    POSITIVE controls (a legal call served at ``pop == 1`` and with the ``iree``
    backend name) and the invariants the rest of the package relies on:
    ``fused_cache_stats()`` is all ``fused_*`` keys, ``cache_stats()`` stays at its
    pinned 8 keys, and ``evoxc_engine.clear_cache()`` resets the fused tables too.
    """
    import torch

    from evogp.tree import evoxc_engine
    from evogp.tree import fused_engine

    checks: List[Tuple[str, bool, str]] = []

    def expect(name: str, fn, kind) -> None:
        try:
            fn()
        except kind as exc:
            checks.append((name, True, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - the wrong type is a failure
            checks.append((name, False, f"raised {type(exc).__name__}: {exc}"))
        else:
            checks.append((name, False, "did NOT raise"))

    node_value, node_type, subtree_size = pack_rows([("var", 0)], pad_to=4)
    x = torch.rand(8, 4, device="cuda")
    labels = torch.rand(8, 1, device="cuda")
    call = dict(
        batch_node_value=node_value,
        batch_node_type=node_type,
        batch_subtree_size=subtree_size,
        input_len=4,
        output_len=1,
    )
    expect(
        "decline: backend=numpy",
        lambda: fused_engine.evaluate_fused(**call, x=x, per_row=False, backend="numpy"),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: output_len=2 (multi-output)",
        lambda: fused_engine.evaluate_fused(
            **{**call, "output_len": 2}, x=x, per_row=False
        ),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: pop_size=0",
        lambda: fused_engine.evaluate_fused(
            **{**call, "batch_node_value": node_value[:0],
               "batch_node_type": node_type[:0], "batch_subtree_size": subtree_size[:0]},
            x=x,
            per_row=False,
        ),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: rows=0",
        lambda: fused_engine.evaluate_fused(**call, x=x[:0], per_row=False),
        fused_engine.FusedUnsupported,
    )
    expect(
        "decline: x is a numpy array (never staged host memory)",
        lambda: fused_engine.evaluate_fused(**call, x=x.cpu().numpy(), per_row=False),
        fused_engine.FusedUnsupported,
    )
    # The envelope is a budget on ``pop * tree_len * rows * 4`` BYTES, so the
    # check is landed with a legal but tiny budget (a non-positive one is a knob
    # error, raised as ``ValueError`` above) on the ``p256-L64-i32-N1024`` shape.
    expect(
        "decline: scratch envelope (1 MiB on p256-L64-N1024)",
        lambda: fused_engine._check_envelope(256, 64, 1024, 1),
        fused_engine.FusedUnsupported,
    )
    checks.append(
        (
            "served: same shape inside an 8192 MiB envelope",
            fused_engine._check_envelope(256, 64, 1024, 8192) is None,
            "ok",
        )
    )
    expect(
        "raise: labels rows != inputs rows",
        lambda: fused_engine.sr_fitness_fused(**call, inputs=x, labels=labels[:4]),
        ValueError,
    )

    # POSITIVE controls: the same population IS served at pop 1 and with the
    # iree backend name (the fused route has no evoxc backend dependency).
    for tag, kwargs in (
        ("served: pop_size=1", {}),
        ("served: backend=iree", {"backend": "iree"}),
    ):
        try:
            out = fused_engine.evaluate_fused(**call, x=x, per_row=False, **kwargs)
            ok = tuple(out.shape) == (1, 8, 1) and out.dtype == torch.float32
            checks.append((tag, ok, f"shape {tuple(out.shape)} dtype {out.dtype}"))
        except Exception as exc:  # noqa: BLE001
            checks.append((tag, False, f"raised {type(exc).__name__}: {exc}"))

    # per_row=True with one mis-sized row block is a ValueError, not a decline.
    expect(
        "raise: per_row x rows != pop",
        lambda: fused_engine.evaluate_fused(**call, x=x, per_row=True),
        ValueError,
    )

    # knob readers
    for name, reader, env, bad in (
        ("knob: BLOCK_ROWS", fused_engine.resolve_block_rows, "EVOGP_FUSED_BLOCK_ROWS", "abc"),
        ("knob: NUM_WARPS", fused_engine.resolve_num_warps, "EVOGP_FUSED_NUM_WARPS", "0"),
        ("knob: MAX_SCRATCH_MB", fused_engine.resolve_max_scratch_mb, "EVOGP_FUSED_MAX_SCRATCH_MB", "-1"),
        ("knob: MIN_POP", fused_engine.resolve_min_pop, "EVOGP_FUSED_MIN_POP", "0"),
    ):
        old = os.environ.get(env)
        os.environ[env] = bad
        try:
            expect(f"{name} rejects {bad!r}", lambda r=reader: r(), ValueError)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old
    checks.append(
        (
            "knob: resolve_min_pop default",
            fused_engine.resolve_min_pop() == 64,
            str(fused_engine.resolve_min_pop()),
        )
    )
    checks.append(
        (
            "knob: resolve_block_rows default",
            fused_engine.resolve_block_rows() == 256,
            str(fused_engine.resolve_block_rows()),
        )
    )

    # counters: own namespace, pinned key count of cache_stats(), shared reset
    stats = fused_engine.fused_cache_stats()
    checks.append(
        (
            "stats: every key is fused_*",
            all(key.startswith("fused_") for key in stats),
            ", ".join(sorted(stats)),
        )
    )
    checks.append(
        (
            "stats: cache_stats() still has 8 keys",
            len(evoxc_engine.cache_stats()) == 8,
            str(len(evoxc_engine.cache_stats())),
        )
    )
    before = fused_engine.fused_cache_stats()
    evoxc_engine.clear_cache()
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "clear_cache() resets the fused front tables",
            after["fused_xt_entries"] == 0,
            f"xt_entries {before['fused_xt_entries']} -> {after['fused_xt_entries']}",
        )
    )
    fused_engine.clear_fused_cache()
    checks.append(
        (
            "clear_fused_cache() zeroes its own counters",
            fused_engine.fused_cache_stats()["fused_calls"] == 0,
            str(fused_engine.fused_cache_stats()["fused_calls"]),
        )
    )

    print("=" * 78)
    print("DECLINES & CONTRACTS: fused route")
    print("=" * 78)
    passed = 0
    for name, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<42} {detail}")
        passed += 1 if ok else 0
    print("-" * 78)
    print(f"DECLINES: {passed}/{len(checks)} checks passed")
    print("-" * 78)
    return 0 if passed == len(checks) else 1


# --------------------------------------------------------------------------- #
# stage 2c: the ROUTE WIRING, driven through the public API only
# --------------------------------------------------------------------------- #
def _set_eval_mode(mode: Optional[str]) -> None:
    """Set ``EVOGP_EVAL_MODE`` in-process (the wired stage drives the public API)."""
    if mode is None:
        os.environ.pop("EVOGP_EVAL_MODE", None)
    else:
        os.environ["EVOGP_EVAL_MODE"] = mode


def _out_node_value(function_id: int, out_index: int) -> float:
    """float32 bit-reinterpretation of ``struct OutNodeValue{function, outIndex}``.

    The multi-output protocol packs the output index into a node's ``node_value``
    (``outIndex`` in the high 16 bits, the function id in the low ones), so a
    hand-built multi-output row needs the raw bit pattern rather than a number.
    """
    import struct

    bits = ((int(out_index) & 0xFFFF) << 16) | (int(function_id) & 0xFF)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _wired_cells() -> List[Dict[str, Any]]:
    """The wired stage's cells: two timing cells plus the harness's ``p64`` niche.

    All three are small enough that the specialized+numpy REFERENCE they are
    compared against costs a few tens of milliseconds, and the ``p64`` cell is
    exactly the smallest population the fused route serves at the shipped
    ``EVOGP_FUSED_MIN_POP`` (64).
    """
    common = _harness()
    wanted = {"p256-L32-i8-N1024", "p256-L64-i32-N1024"}
    cells = [cell for cell in _cells() if cell["id"] in wanted]
    cells.append(common.cell(64, 64, 32, 64))
    return cells


def _close(stats: Dict[str, Any], limit: float) -> bool:
    """Value/loss closeness of the wired stage: bitwise OR ``max_rel <= limit``."""
    if "shape_mismatch" in stats:
        return False
    if stats.get("nan_mismatch", 0) or stats.get("inf_mismatch", 0):
        return False
    if stats.get("bitwise_equal"):
        return True
    return float(stats.get("max_rel", 0.0)) <= limit


def stage_wired(args: argparse.Namespace) -> int:
    """``EVOGP_EVAL_MODE=fused`` end to end, through the PUBLIC API only.

    The parity/goldens stages call :func:`fused_engine.evaluate_fused` directly, so
    they say nothing about the POLICY.  This stage drives ``Forest.forward`` /
    ``Forest.batch_forward`` / ``Forest.SR_fitness`` with ``EVOGP_EVAL_MODE`` set
    in-process and asserts the two things a wiring bug would break:

    1. **The numbers.**  The same three calls under ``fused`` must equal the same
       calls under ``specialized`` (the reference): the value path bitwise or
       ``<= VALUE_RTOL`` relative, the loss ``<= LOSS_RTOL`` relative.  The fused
       facade's counters prove the calls were actually SERVED by this route
       (``fused_calls`` moves by exactly three per cell) rather than silently
       falling back, which a passing comparison alone would not show.
    2. **The decision.**  ``resolve_route(pop, eval_mode="fused")`` names the
       fused engine on ``iree`` with a ``numpy`` fallback on a device box, is
       ``specialized``+``numpy`` for ``pop_size == 1`` (a single ``Tree``) and for
       any population below ``EVOGP_FUSED_MIN_POP``, is ``specialized``+``numpy``
       when ``device_available()`` is False, and is NEVER produced by ``auto``;
       ``should_use_dense`` is False for it and the ``Route`` invariant refuses
       ``use_dense=True`` with ``engine="fused"``.

    A multi-output population (``output_len=2``) must DECLINE into the
    specialized fallback: same numbers, one ``fused_declines``, and no kernel
    compiled or scratch buffer allocated (that is what "no fused launch" means --
    the facade counts the attempt in ``fused_calls`` before it declines).
    """
    import torch

    from evogp.tree import Forest
    from evogp.tree import dense_engine, fused_engine

    common = _harness()
    compare = _comparator()
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    saved = {
        name: os.environ.get(name)
        for name in ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_FUSED_MIN_POP")
    }
    # Hermetic: the reference leg must be the specialized engine on numpy, so an
    # ambient backend spelling goes away for the whole stage.
    os.environ.pop("EVOGP_EVOXC_BACKEND", None)
    os.environ.pop("EVOGP_FUSED_MIN_POP", None)
    try:
        # ---- 1. the policy surface -------------------------------------- #
        minimum = dense_engine.resolve_fused_min_pop()
        check(
            "policy: resolve_fused_min_pop delegates to the one parser",
            minimum == fused_engine.resolve_min_pop() == 64,
            f"{minimum} vs fused_engine {fused_engine.resolve_min_pop()}",
        )
        route = dense_engine.resolve_route(256, eval_mode="fused")
        check(
            "policy: resolve_route(256, fused).engine == 'fused'",
            route.engine == "fused",
            route.reason,
        )
        check(
            "policy: fused route is iree with a numpy fallback",
            (route.backend, route.fallback_backend) == ("iree", "numpy"),
            f"{route.backend} / {route.fallback_backend}",
        )
        check(
            "policy: fused route is use_fused and not use_dense",
            route.use_fused and not route.use_dense,
            f"use_fused={route.use_fused} use_dense={route.use_dense}",
        )
        for pop in (1, minimum - 1):
            below = dense_engine.resolve_route(pop, eval_mode="fused")
            check(
                f"policy: pop_size={pop} < {minimum} -> specialized+numpy",
                (below.engine, below.backend, below.fallback_backend)
                == ("specialized", "numpy", "numpy"),
                below.reason,
            )
        at_min = dense_engine.resolve_route(minimum, eval_mode="fused")
        check(
            f"policy: pop_size={minimum} == EVOGP_FUSED_MIN_POP -> fused",
            at_min.engine == "fused",
            at_min.reason,
        )
        explicit = dense_engine.resolve_route(256, backend="numpy", eval_mode="fused")
        check(
            "policy: an explicit backend spelling is honoured (and declines)",
            explicit.engine == "fused" and explicit.backend == "numpy",
            f"{explicit.engine} on {explicit.backend}: {explicit.reason}",
        )
        # A CPU box: the monkeypatched probe is the battery's stand-in for
        # unplugging the GPU.
        original_probe = dense_engine.device_available
        dense_engine.device_available = lambda: False
        try:
            cpu_route = dense_engine.resolve_route(256, eval_mode="fused")
        finally:
            dense_engine.device_available = original_probe
        check(
            "policy: no usable device -> specialized+numpy (engine stays default)",
            (cpu_route.engine, cpu_route.backend, cpu_route.fallback_backend)
            == ("specialized", "numpy", "numpy"),
            cpu_route.reason,
        )
        engines = set()
        for pop in (1, 16, 63, 64, 256, 1024, 2048):
            for backend in (None, "numpy", "iree"):
                for rows in (None, 1024):
                    engines.add(
                        dense_engine.resolve_route(
                            pop, backend=backend, rows=rows, eval_mode="auto"
                        ).engine
                    )
        check(
            "policy: auto NEVER selects fused",
            "fused" not in engines,
            "engines seen: " + ", ".join(sorted(engines)),
        )
        for pop in (1, minimum, 256):
            check(
                f"policy: should_use_dense({pop}, fused) is False",
                dense_engine.should_use_dense(pop, eval_mode="fused") is False,
            )
        try:
            dense_engine.Route(True, "iree", "numpy", engine="fused")
        except ValueError as exc:
            check(
                "policy: Route(use_dense=True, engine='fused') is refused",
                "fused" in str(exc),
                str(exc),
            )
        else:
            check(
                "policy: Route(use_dense=True, engine='fused') is refused",
                False,
                "did NOT raise",
            )
        # The knob has ONE parser, so the policy reader must refuse a bad value too.
        os.environ["EVOGP_FUSED_MIN_POP"] = "0"
        try:
            dense_engine.resolve_fused_min_pop()
        except ValueError as exc:
            check(
                "policy: EVOGP_FUSED_MIN_POP=0 refused by the policy reader",
                "EVOGP_FUSED_MIN_POP" in str(exc),
                str(exc),
            )
        else:
            check(
                "policy: EVOGP_FUSED_MIN_POP=0 refused by the policy reader",
                False,
                "did NOT raise",
            )
        finally:
            os.environ.pop("EVOGP_FUSED_MIN_POP", None)

        # ---- 2. the numbers, through the public API --------------------- #
        for cell in _wired_cells():
            workload = common.build_workload(cell)
            forest = workload["forest"]
            x = workload["x"]
            labels = workload["labels"]
            x_per_row = workload["x_per_row"]
            _set_eval_mode("specialized")
            ref_batch = forest.batch_forward(x)
            ref_row = forest.forward(x_per_row)
            ref_loss = forest.SR_fitness(x, labels)
            before = fused_engine.fused_cache_stats()
            _set_eval_mode("fused")
            got_batch = forest.batch_forward(x)
            got_row = forest.forward(x_per_row)
            got_loss = forest.SR_fitness(x, labels)
            after = fused_engine.fused_cache_stats()
            for tag, ref, got, limit in (
                ("batch_forward (per_row=False)", ref_batch, got_batch, VALUE_RTOL),
                ("forward (per_row=True)", ref_row, got_row, VALUE_RTOL),
                ("SR_fitness (MSE)", ref_loss, got_loss, LOSS_RTOL),
            ):
                stats = compare(ref, got)
                check(
                    f"wired {cell['id']}: {tag} == specialized",
                    _close(stats, limit),
                    f"bitwise={stats.get('bitwise_equal')} "
                    f"max_rel={stats.get('max_rel', 0.0):.3e} (<= {limit:g})",
                )
            served = after["fused_calls"] - before["fused_calls"]
            declined = after["fused_declines"] - before["fused_declines"]
            check(
                f"wired {cell['id']}: the fused engine served all three calls",
                served == 3 and declined == 0,
                f"fused_calls +{served} declines +{declined}",
            )

        # ---- 3. a multi-output population: decline, never launch -------- #
        from evogp.tree.utils import Func

        # 64 rows so the population clears EVOGP_FUSED_MIN_POP: below it the route
        # IS specialized, and the decline this check is about would never happen.
        multi_rows: List[Any] = [
            (
                "raw",
                _out_node_value(int(Func.ADD), 0),
                0x80 | 3,
                (("var", 0), ("var", 1)),
            ),
            (
                "raw",
                _out_node_value(int(Func.MUL), 1),
                0x80 | 3,
                (("var", 2), ("const", 2.0)),
            ),
        ] * 32
        multi = Forest(3, 2, *pack_rows(multi_rows, pad_to=8))
        mx = torch.rand(16, 3, device="cuda")
        _set_eval_mode("specialized")
        ref_multi = multi.batch_forward(mx)
        before = fused_engine.fused_cache_stats()
        _set_eval_mode("fused")
        got_multi = multi.batch_forward(mx)
        after = fused_engine.fused_cache_stats()
        stats = compare(ref_multi, got_multi)
        check(
            "wired output_len=2: declines into specialized+numpy (same values)",
            _close(stats, VALUE_RTOL),
            f"bitwise={stats.get('bitwise_equal')} "
            f"max_rel={stats.get('max_rel', 0.0):.3e} (<= {VALUE_RTOL:g})",
        )
        check(
            "wired output_len=2: NO fused launch (no kernel, no scratch)",
            after["fused_kernel_compiles"] == before["fused_kernel_compiles"]
            and after["fused_scratch_allocs"] == before["fused_scratch_allocs"],
            "compiles +"
            f"{after['fused_kernel_compiles'] - before['fused_kernel_compiles']} "
            f"scratch_allocs +{after['fused_scratch_allocs'] - before['fused_scratch_allocs']}",
        )
        check(
            "wired output_len=2: the facade was consulted and declined",
            after["fused_declines"] == before["fused_declines"] + 1,
            f"declines +{after['fused_declines'] - before['fused_declines']}",
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("=" * 96)
    print(
        "WIRED: EVOGP_EVAL_MODE=fused through Forest.forward / batch_forward / "
        "SR_fitness"
    )
    print("=" * 96)
    passed = 0
    for name, ok, detail in checks:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<62} {detail}")
        passed += 1 if ok else 0
    print("-" * 96)
    print(f"WIRED: {passed}/{len(checks)} checks passed")
    print("-" * 96)
    return 0 if passed == len(checks) else 1


# --------------------------------------------------------------------------- #
# stage 3: timings (driver + one-cell child)
# --------------------------------------------------------------------------- #
def _arm(forest):
    """The ``torch-batched`` reference arm (whole-population torch implementation)."""
    from arms_torch import BatchedEvaluator  # type: ignore

    return BatchedEvaluator(forest)


def _stage_split(
    node_value, node_type, subtree_size, x, xt, labels, *, input_len, rows, block_rows, num_warps
):
    """Kernel-only and transpose-only timings for the value and the loss pass.

    ``kernel_ms`` is the Triton launch alone (the facade's validation, memo lookup
    and ``contiguous()`` are excluded, so ``total - kernel`` is exactly the host
    glue), and ``transpose_ms`` is the un-memoized ``x.t().contiguous()`` — the ONE
    prep step of the batched path, which the identity memo makes a steady-state
    no-op and which a GP run still pays once per fresh dataset tensor.
    """
    import torch

    from evogp.tree import fused_kernels

    common = _harness()
    tree_len = int(node_value.shape[1])
    kernel = common.time_call(
        lambda: fused_kernels.launch_value(
            node_value,
            node_type,
            subtree_size,
            xt,
            input_len=input_len,
            tree_len=tree_len,
            rows=rows,
            per_row=False,
            block_rows=block_rows,
            num_warps=num_warps,
        ),
        warmup=2,
        reps=10,
    )
    loss_kernel = common.time_call(
        lambda: fused_kernels.launch_loss(
            node_value,
            node_type,
            subtree_size,
            xt,
            labels.view(-1),
            input_len=input_len,
            tree_len=tree_len,
            rows=rows,
            use_mse=True,
            block_rows=block_rows,
            num_warps=num_warps,
        ),
        warmup=2,
        reps=10,
    )
    transpose = common.time_call(lambda: x.t().contiguous(), warmup=2, reps=10)
    return {
        "kernel_ms": kernel["median_ms"],
        "loss_kernel_ms": loss_kernel["median_ms"],
        "transpose_ms": transpose["median_ms"],
    }


def stage_time_one(args: argparse.Namespace) -> int:
    """One cell in this process: warm medians, cold first call, stage split.

    ``--wired`` times the same three surfaces through the PUBLIC API
    (``Forest.batch_forward`` / ``Forest.SR_fitness`` / ``Forest.forward``) with
    ``EVOGP_EVAL_MODE=fused`` set in-process, so the number includes the route
    decision and the facade instead of a direct :func:`fused_engine.evaluate_fused`
    call.  Everything else (the torch arm, the warmup/rep protocol, the cold
    first-call column, the stage split) is identical.
    """
    import torch

    common = _harness()
    from evogp.tree import fused_engine, fused_kernels

    if args.wired:
        os.environ["EVOGP_EVAL_MODE"] = "fused"
    cell = _cell_by_id(args.cell)
    workload = common.build_workload(cell)
    forest = workload["forest"]
    x = workload["x"]
    labels = workload["labels"]
    x_per_row = workload["x_per_row"]
    node_value = forest.batch_node_value
    node_type = forest.batch_node_type
    subtree_size = forest.batch_subtree_size
    input_len = int(cell["input_len"])
    rows = int(cell["rows"])
    pop = int(cell["pop"])
    block_rows = fused_engine.resolve_block_rows()
    num_warps = fused_engine.resolve_num_warps()

    # The torch arm is built BEFORE the fused cold call so the cold number is the
    # Triton import + JIT only (the arm's own construction is host code).
    arm = _arm(forest)
    arm_value = common.time_call(lambda: arm.batch_forward(x), warmup=2, reps=10)
    arm_loss = common.time_call(lambda: arm.sr_fitness(x, labels), warmup=2, reps=10)
    arm_per_row = common.time_call(lambda: arm.forward(x_per_row), warmup=2, reps=10)

    import_ms = 0.0
    if "triton" not in sys.modules:
        torch.cuda.synchronize()
        start = time.perf_counter()
        import triton  # noqa: F401
        import_ms = (time.perf_counter() - start) * 1e3

    if args.wired:
        # The public API: the route decision and the facade are inside the number.
        fused_value = common.time_call(
            lambda: forest.batch_forward(x), warmup=2, reps=10
        )
        fused_loss = common.time_call(
            lambda: forest.SR_fitness(x, labels), warmup=2, reps=10
        )
        fused_per_row = common.time_call(
            lambda: forest.forward(x_per_row), warmup=2, reps=10
        )
    else:
        fused_value = common.time_call(
            lambda: fused_engine.evaluate_fused(
                node_value, node_type, subtree_size, input_len, 1, x, per_row=False
            ),
            warmup=2,
            reps=10,
        )
        fused_loss = common.time_call(
            lambda: fused_engine.sr_fitness_fused(
                node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=True
            ),
            warmup=2,
            reps=10,
        )
        fused_per_row = common.time_call(
            lambda: fused_engine.evaluate_fused(
                node_value, node_type, subtree_size, input_len, 1, x_per_row, per_row=True
            ),
            warmup=2,
            reps=10,
        )
    xt = x.t().contiguous()
    split = _stage_split(
        node_value,
        node_type,
        subtree_size,
        x,
        xt,
        labels,
        input_len=input_len,
        rows=rows,
        block_rows=block_rows,
        num_warps=num_warps,
    )
    stats = fused_engine.fused_cache_stats()
    payload = {
        "kind": "timing",
        "wired": bool(args.wired),
        "cell": cell["id"],
        "pop": pop,
        "rows": rows,
        "tree_len": int(cell["tree_len"]),
        "block_rows": int(block_rows),
        "num_warps": int(num_warps),
        "n_rb": int(fused_kernels.plan(rows, per_row=False, block_rows=block_rows, num_warps=num_warps)["n_rb"]),
        "arm_value_ms": arm_value["median_ms"],
        "arm_loss_ms": arm_loss["median_ms"],
        "arm_per_row_ms": arm_per_row["median_ms"],
        "fused_value_ms": fused_value["median_ms"],
        "fused_loss_ms": fused_loss["median_ms"],
        "fused_per_row_ms": fused_per_row["median_ms"],
        "fused_value_first_ms": fused_value["first_call_ms"],
        "fused_loss_first_ms": fused_loss["first_call_ms"],
        "import_ms": import_ms,
        "kernel_ms": split["kernel_ms"],
        "loss_kernel_ms": split["loss_kernel_ms"],
        "transpose_ms": split["transpose_ms"],
        "stats": stats,
    }
    print(
        f"{cell['id']:<28} arm={arm_value['median_ms']:.3f} "
        f"fused{'(wired)' if args.wired else ''}={fused_value['median_ms']:.3f} ms "
        f"({arm_value['median_ms'] / fused_value['median_ms']:.2f}x) "
        f"cold={fused_value['first_call_ms']:.1f} ms",
        flush=True,
    )
    # Cross-check the TIMED population against the torch arm: the parity stage
    # uses the harness's small populations, and a timing number over a wrong answer
    # would be worthless, so the exact workload that was just timed is compared too.
    compare = _comparator()
    arm_out = arm.batch_forward(x)
    if args.wired:
        fused_out = forest.batch_forward(x)
    else:
        fused_out = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, x, per_row=False
        )
    value_check = compare(arm_out, fused_out)
    arm_loss_out = arm.sr_fitness(x, labels)
    if args.wired:
        fused_loss_out = forest.SR_fitness(x, labels)
    else:
        fused_loss_out = fused_engine.sr_fitness_fused(
            node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=True
        )
    loss_check = compare(arm_loss_out, fused_loss_out)
    payload["arm_vs_fused_value"] = value_check
    payload["arm_vs_fused_loss"] = loss_check
    print(
        f"  cross-check vs the torch arm on this population: "
        f"value max_abs={value_check.get('max_abs', 0.0):.3e} "
        f"max_rel={value_check.get('max_rel', 0.0):.3e} "
        f"nan_mismatch={value_check.get('nan_mismatch', 0)} | "
        f"loss max_rel={loss_check.get('max_rel', 0.0):.3e} "
        f"nan_mismatch={loss_check.get('nan_mismatch', 0)}",
        flush=True,
    )
    _json_line(payload)
    return 0


def _spawn(extra: List[str]) -> Optional[Dict[str, Any]]:
    """Run one child leg of this battery in its OWN process and parse its RESULT."""
    env = dict(os.environ)
    argv = [sys.executable, "-m", "evogp.tree.fused_check"] + extra
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT ") :])
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        print(f"  [FAIL] {' '.join(extra)} (exit {proc.returncode}): " + " | ".join(tail))
    return payload


def stage_timing(args: argparse.Namespace) -> int:
    """Driver: one subprocess per cell, then the honest comparison table."""
    print("=" * 96)
    if args.wired:
        print(
            "TIMING (WIRED): EVOGP_EVAL_MODE=fused through Forest.batch_forward / "
            "Forest.SR_fitness / Forest.forward"
        )
        print(
            "  vs torch-batched (warm medians, 2 warmup + 10 sync-bracketed reps)"
        )
    else:
        print(
            "TIMING: fused vs torch-batched (warm medians, 2 warmup + 10 "
            "sync-bracketed reps)"
        )
    print("=" * 96)
    records = []
    for cell in _cells():
        if args.only and cell["id"] not in args.only:
            continue
        extra = ["--stage", "time-one", "--cell", cell["id"]]
        if args.wired:
            extra.append("--wired")
        payload = _spawn(extra)
        if payload:
            records.append(payload)
    if not records:
        print("no cells measured")
        return 1
    print()
    header = (
        f"| {'cell':<26} | {'torch-batched':>13} | {'fused':>10} | {'speedup':>7} | "
        f"{'cold ms':>8} | {'B':>4} | {'W':>2} | {'n_rb':>4} |"
    )
    print(header)
    print("|" + "-" * 28 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 6 + "|" + "-" * 4 + "|" + "-" * 6 + "|")
    worst = 1e9
    for rec in records:
        speedup = rec["arm_value_ms"] / rec["fused_value_ms"]
        worst = min(worst, speedup)
        print(
            f"| {rec['cell']:<26} | {rec['arm_value_ms']:>10.3f} ms | {rec['fused_value_ms']:>7.3f} ms | "
            f"{speedup:>6.2f}x | {rec['fused_value_first_ms']:>6.1f} | {rec['block_rows']:>4} | "
            f"{rec['num_warps']:>2} | {rec['n_rb']:>4} |"
        )
    print()
    print("LOSS workload (MSE) -- the cold column is the first loss call of a fresh")
    print("process: Triton import + JIT on a cold on-disk cache, a cache LOAD once the")
    print("same (BLOCK_R, warps, flags) specialization was compiled in an earlier run.")
    print("LOSS workload (MSE)")
    print(header.replace("cell", "cell").replace("torch-batched", "torch-batched"))
    print("|" + "-" * 28 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 6 + "|" + "-" * 4 + "|" + "-" * 6 + "|")
    worst_loss = 1e9
    for rec in records:
        speedup = rec["arm_loss_ms"] / rec["fused_loss_ms"]
        worst_loss = min(worst_loss, speedup)
        print(
            f"| {rec['cell']:<26} | {rec['arm_loss_ms']:>10.3f} ms | {rec['fused_loss_ms']:>7.3f} ms | "
            f"{speedup:>6.2f}x | {rec['fused_loss_first_ms']:>6.1f} | {rec['block_rows']:>4} | "
            f"{rec['num_warps']:>2} | {rec['n_rb']:>4} |"
        )
    print()
    print("PER-ROW workload (one input row per tree)")
    for rec in records:
        speedup = rec["arm_per_row_ms"] / rec["fused_per_row_ms"]
        print(
            f"  {rec['cell']:<26} arm={rec['arm_per_row_ms']:>8.3f} ms fused={rec['fused_per_row_ms']:>7.3f} ms "
            f"({speedup:>5.2f}x)"
        )
    print()
    print("STAGE SPLIT of the value call (ms): kernel = the Triton launch alone, host = the rest")
    print(
        f"  {'cell':<26} {'total':>8} {'kernel':>8} {'host':>7} {'transpose(cold)':>16} "
        f"{'loss total':>11} {'loss kernel':>12} {'loss host':>10}"
    )
    for rec in records:
        host = rec["fused_value_ms"] - rec["kernel_ms"]
        loss_host = rec["fused_loss_ms"] - rec["loss_kernel_ms"]
        print(
            f"  {rec['cell']:<26} {rec['fused_value_ms']:>8.3f} {rec['kernel_ms']:>8.3f} {host:>7.3f} "
            f"{rec['transpose_ms']:>16.3f} {rec['fused_loss_ms']:>11.3f} {rec['loss_kernel_ms']:>12.3f} "
            f"{loss_host:>10.3f}"
        )
    print()
    print("CROSS-CHECK of the timed populations vs the torch arm (deviation of the number")
    print("that was just timed, not of a different population):")
    for rec in records:
        value = rec.get("arm_vs_fused_value", {})
        loss = rec.get("arm_vs_fused_loss", {})
        print(
            f"  {rec['cell']:<26} value max_abs={value.get('max_abs', 0.0):.3e} "
            f"max_rel={value.get('max_rel', 0.0):.3e} bitwise={value.get('bitwise_equal')} "
            f"nan_mismatch={value.get('nan_mismatch', 0)} | loss max_rel="
            f"{loss.get('max_rel', 0.0):.3e} nan_mismatch={loss.get('nan_mismatch', 0)}"
        )
    print()
    for rec in records:
        stats = rec["stats"]
        print(
            f"  {rec['cell']:<26} stats: compiles={stats['fused_kernel_compiles']} "
            f"scratch_allocs={stats['fused_scratch_allocs']} scratch_reuses={stats['fused_scratch_reuses']} "
            f"xt_hits={stats['fused_xt_hits']} xt_misses={stats['fused_xt_misses']} "
            f"calls={stats['fused_calls']} declines={stats['fused_declines']}"
        )
    print()
    print(f"worst value speedup vs torch-batched: {worst:.2f}x   worst loss speedup: {worst_loss:.2f}x")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


# --------------------------------------------------------------------------- #
# stage 4: the BLOCK_ROWS x NUM_WARPS sweep
# --------------------------------------------------------------------------- #
def stage_tune_one(args: argparse.Namespace) -> int:
    """One (config, cell) in this process; the knobs are read per call."""
    os.environ["EVOGP_FUSED_BLOCK_ROWS"] = str(int(args.block))
    os.environ["EVOGP_FUSED_NUM_WARPS"] = str(int(args.warps))
    import torch

    common = _harness()
    from evogp.tree import fused_engine

    cell = _cell_by_id(args.cell)
    workload = common.build_workload(cell)
    forest = workload["forest"]
    x, labels = workload["x"], workload["labels"]
    node_value = forest.batch_node_value
    node_type = forest.batch_node_type
    subtree_size = forest.batch_subtree_size
    input_len = int(cell["input_len"])
    arm = _arm(forest)
    arm_value = common.time_call(lambda: arm.batch_forward(x), warmup=2, reps=10)
    arm_loss = common.time_call(lambda: arm.sr_fitness(x, labels), warmup=2, reps=10)
    value = common.time_call(
        lambda: fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, 1, x, per_row=False
        ),
        warmup=2,
        reps=10,
    )
    loss = common.time_call(
        lambda: fused_engine.sr_fitness_fused(
            node_value, node_type, subtree_size, input_len, 1, x, labels, use_MSE=True
        ),
        warmup=2,
        reps=10,
    )
    _json_line(
        {
            "kind": "tuning",
            "cell": cell["id"],
            "block_rows": int(args.block),
            "num_warps": int(args.warps),
            "arm_value_ms": arm_value["median_ms"],
            "arm_loss_ms": arm_loss["median_ms"],
            "fused_value_ms": value["median_ms"],
            "fused_loss_ms": loss["median_ms"],
            "cold_ms": value["first_call_ms"],
        }
    )
    return 0


def stage_tuning(args: argparse.Namespace) -> int:
    """Driver: one subprocess per (config, cell), then the minimax decision."""
    print("=" * 96)
    print("TUNING: EVOGP_FUSED_BLOCK_ROWS x EVOGP_FUSED_NUM_WARPS (one process per config+cell)")
    print("=" * 96)
    blocks = [int(b) for b in str(args.blocks).split(",") if b.strip()]
    warps = [int(w) for w in str(args.warps_list).split(",") if w.strip()]
    cells = [c for c in _cells() if not args.only or c["id"] in args.only]
    records: List[Dict[str, Any]] = []
    for block in blocks:
        for warp in warps:
            for cell in cells:
                payload = _spawn(
                    [
                        "--stage",
                        "tune-one",
                        "--cell",
                        cell["id"],
                        "--block",
                        str(block),
                        "--warps",
                        str(warp),
                    ]
                )
                if payload:
                    records.append(payload)
                    print(
                        f"  b{block:<4} w{warp} {cell['id']:<26} "
                        f"value={payload['fused_value_ms']:>8.3f} ms "
                        f"({payload['arm_value_ms'] / payload['fused_value_ms']:>5.2f}x) "
                        f"loss={payload['fused_loss_ms']:>8.3f} ms "
                        f"({payload['arm_loss_ms'] / payload['fused_loss_ms']:>5.2f}x)",
                        flush=True,
                    )
    if not records:
        print("no configurations measured")
        return 1
    by_config: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in records:
        by_config.setdefault((rec["block_rows"], rec["num_warps"]), []).append(rec)
    print()
    print(
        f"| {'BLOCK_R':>7} | {'warps':>5} | {'worst value ratio':>18} | {'mean value ratio':>17} | "
        f"{'worst loss ratio':>17} | {'worst cell':<26} |"
    )
    print("|" + "-" * 9 + "|" + "-" * 7 + "|" + "-" * 20 + "|" + "-" * 19 + "|" + "-" * 19 + "|" + "-" * 28 + "|")
    ranked = []
    for (block, warp), rows in sorted(by_config.items()):
        ratios = [(r["fused_value_ms"] / r["arm_value_ms"], r["cell"]) for r in rows]
        worst_ratio, worst_cell = max(ratios, key=lambda item: item[0])
        loss_ratios = [r["fused_loss_ms"] / r["arm_loss_ms"] for r in rows]
        ranked.append((worst_ratio, block, warp, rows))
        print(
            f"| {block:>7} | {warp:>5} | {worst_ratio:>18.3f} | "
            f"{sum(r for r, _ in ratios) / len(ratios):>17.3f} | {max(loss_ratios):>17.3f} | {worst_cell:<26} |"
        )
    ranked.sort()
    best_worst, best_block, best_warp, best_rows = ranked[0]
    print()
    print(
        f"minimax choice (lowest WORST-cell fused/torch ratio): "
        f"EVOGP_FUSED_BLOCK_ROWS={best_block} EVOGP_FUSED_NUM_WARPS={best_warp} "
        f"(worst {best_worst:.3f} on {max(((r['fused_value_ms'] / r['arm_value_ms'], r['cell']) for r in best_rows))[1]})"
    )
    shipped = (256, 1)
    shipped_rows = by_config.get(shipped, [])
    if shipped_rows:
        shipped_worst = max(r["fused_value_ms"] / r["arm_value_ms"] for r in shipped_rows)
        print(
            f"shipped default (BLOCK_R 256, 1 warp): worst {shipped_worst:.3f} "
            f"({shipped_worst / best_worst:.2f}x the minimax ratio)"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evogp.tree.fused_check",
        description="Dev battery of the fused (Triton) evaluation route.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=(
            "parity",
            "goldens",
            "declines",
            "wired",
            "timing",
            "tuning",
            "all",
            "time-one",
            "tune-one",
        ),
        help="which leg to run (default: all)",
    )
    parser.add_argument("--cell", default=None, help="cell id for the single-process legs")
    parser.add_argument(
        "--wired",
        action="store_true",
        help="time through the public API with EVOGP_EVAL_MODE=fused (--stage timing)",
    )
    parser.add_argument("--block", type=int, default=256, help="EVOGP_FUSED_BLOCK_ROWS for tune-one")
    parser.add_argument("--warps", type=int, default=4, help="EVOGP_FUSED_NUM_WARPS for tune-one")
    parser.add_argument(
        "--blocks", default="64,128,256,512", help="BLOCK_ROWS grid of the tuning driver"
    )
    parser.add_argument("--warps-list", default="1,2,4", help="num_warps grid of the tuning driver")
    parser.add_argument("--parity-pop", type=int, default=32, help="pop of a parity population")
    parser.add_argument("--parity-rows", type=int, default=128, help="data rows of a parity set")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated cell ids to restrict the driver stages to",
    )
    parser.add_argument("--json", default=None, help="write the stage's records to this file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.only:
        args.only = [item.strip() for item in args.only.split(",") if item.strip()]
    if args.stage in ("time-one", "tune-one") and not args.cell:
        parser.error(f"--stage {args.stage} needs --cell")
    status = 0
    if args.stage in ("parity", "all"):
        status |= stage_parity(args)
    if args.stage in ("goldens", "all"):
        status |= stage_goldens(args)
    if args.stage in ("declines", "all"):
        status |= stage_declines(args)
    if args.stage in ("wired", "all"):
        status |= stage_wired(args)
    if args.stage in ("timing", "all"):
        status |= stage_timing(args)
    if args.stage in ("tuning", "all"):
        status |= stage_tuning(args)
    if args.stage == "time-one":
        status |= stage_time_one(args)
    if args.stage == "tune-one":
        status |= stage_tune_one(args)
    print(f"\nfused_check: exit {status}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
