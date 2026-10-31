"""Dev battery for the FUSED (Triton) evaluation route — parity, goldens, timings, tuning.

This is a ``__main__``-guarded CLI, not a pytest module: it needs a GPU, it imports
the *harness* (``test/benchmark/evoxc``) and it is the only place where the fused
route's numeric contract and its economics are both stated.  Nothing imports it.

WHAT IT CHECKS
--------------
1. **Parity against the oracle.**  ``src/evogp/tests/reference_eval.py`` is the
   numeric contract of record (pure torch/numpy, written independently of the
   engines).  Five random populations cover the function-table families — the
   benchmark arithmetic set, the arithmetic+ternary set, the loose/protected set,
   a ``rich`` transcendental set and a ``compare_unary`` set (``>``, ``<=``,
   ``neg``, ``abs``, ``sinh``, ``cosh``, ``+``, ``*``) that is the ONLY place the
   six operators no other set draws are exercised — the hand-written
   ``sinh``/``cosh`` ``(exp(x) -+ exp(-x)) * 0.5`` compositions included — and a
   set of handmade adversarial rows covers the shapes a random generator does not
   produce: single-node leaves, a pure constant tree, a 40-deep left chain, a
   40-deep right chain, full ternary ``if`` trees and a real tree followed by
   garbage padding.  A single-output row carrying ``OUT_NODE``-tagged nodes is NOT
   a parity case: the oracle's ``_kind_from_type`` folds such a type into its
   ternary branch, and serving it would return a plausible wrong number, so every
   engine REFUSES it host-side (``multi_output.check_multi_output``) and
   ``--stage declines`` pins that refusal (``out_node_tagged_row``).
   Each population is compared on all three surfaces: the batched value path
   (``per_row=False``), the per-tree value path (``per_row=True``) and the SR loss
   (MSE and MAE).
   Assertions: every ``PARITY_SETS`` entry carries its own MEASURED ``value_tol``
   (``0.0`` = the value path must be bitwise, which is what ``fused_kernels.py``'s
   ``enable_fp_fusion=False`` buys — measured for ``arithmetic`` and
   ``compare_unary``, ``1e-7`` relative for the other three, i.e. the measured
   ``max_rel`` of 0 with a margin); every adversarial case carries its own
   measured ``bitwise`` flag (``False`` only where a NaN payload / signed zero
   makes the byte comparison meaningless, and its ``note`` says which); the loss
   path is ``<= 1e-5`` relative everywhere (the kernel reduces in float32 per
   row-block, the oracle reduces in one float32 ``sum``, so the two differ by
   accumulation order only).
2. **Goldens.**  The two frozen fixture families of ``test/equivalence/test_goldens``
   (``src/evogp/tests/goldens/*.npz`` = this package's engine schema, and
   ``test/goldens/*.npz`` = outputs of the deleted CUDA kernels) at the documented
   ``1e-4`` rtol/atol.  Multi-output fixtures are still SKIPPED here -- this stage
   is the single-output golden gate, and serving them would move its recorded
   count -- so the multi-output protocol is verified by ``--stage multi`` against
   the oracle instead.
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
   that ``auto`` selects it from ``EVOGP_FUSED_MIN_POP`` up on a device box and keeps the
   pre-fused policy below the floor / without a device, and that a multi-output population is SERVED by
   the multi-output kernel (same numbers as the specialized reference, one
   ``fused_calls`` and no ``fused_declines``).  ``--stage timing --wired`` re-times
   the six cells through the same public calls.
6. **The multi-output protocol** (``out_len >= 2``, ``--stage multi``).  Handmade
   ``OUT_NODE``-tagged populations (``multi_adversarial_cases``) at
   ``2 <= out_len <= fused_engine.MAX_OUT_LEN``: both outputs of a deep/shallow
   pair, a tagged root ternary, a single tagged output (the others must stay
   exactly ``0.0``), two nodes accumulating into ONE output, tagged leaves (which
   accumulate nothing), an out-of-range ``outIndex`` (dropped), one tagged node per
   arity, a tree with no tag at all (every output ``0.0``, so the loss is the label
   norm), a packed ``MUL`` beside a packed ``ADD`` on the same operands (the
   extraction has to change the number), and one population evaluated at
   ``out_len`` 2 AND 3 (the tag on index 2 changes meaning).  Every case runs the
   batched value path, the per-tree value path and BOTH loss paths (MSE and MAE)
   against the oracle, and the value path must stay **bitwise** -- the multi-output
   kernel reuses the single-output per-op emitters and ``enable_fp_fusion=False``.

HOW TO RUN
----------
From the repository root, with the torch-carrying interpreter and the harness on
the path (``TRITON_CACHE_DIR`` must be writable — the default ``~/.triton`` is not,
in the sandbox)::

    TRITON_CACHE_DIR=$TMPDIR/fused/tc \
    PYTHONPATH=src:/root/evoxc:test/benchmark/evoxc EVOXC_DISABLE_LOCATIONS=1 \
    /tmp/rocvenv/bin/python -m evogp.tree.fused_check --stage parity
    ... --stage multi           # the multi-output (OUT_NODE) protocol
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
# are two levels up.  ``main`` adds the oracle's directory (see ``_setup_paths``),
# which keeps `python -m evogp.tree.fused_check` usable with only PYTHONPATH=src
# while PYTHONPATH still wins for evogp itself.  Deliberately NOT at import time:
# importing this module must not mutate ``sys.path``.  Every ``-m`` child leg the
# driver stages spawn re-enters through ``main``, so the child sees the same path.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.dirname(os.path.dirname(_HERE))
_TESTS_DIR = os.path.join(os.path.dirname(_HERE), "tests")
_REPO_ROOT = os.path.dirname(_SRC_ROOT)

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

#: ``using_funcs`` sets of the parity stage.  ``value_tol`` is the MEASURED bound
#: of the value paths (``0.0`` = bitwise): ``arithmetic`` and ``compare_unary`` are
#: bitwise (no transcendental round trip, and no NaN payload / signed zero), the
#: other three measured ``max_rel`` 0 as well but keep a 1e-7 margin because a
#: transcendental emitter could move by an ulp.  The loss bound stays ``LOSS_RTOL``
#: for every set (accumulation order, not semantics).
PARITY_SETS: Dict[str, Dict[str, Any]] = {
    "arithmetic": dict(
        using_funcs=("+", "-", "*", "/"),
        input_len=8,
        max_layer_cnt=4,
        value_tol=0.0,
    ),
    "ternary": dict(
        using_funcs=("+", "-", "*", "/", "if"),
        input_len=4,
        max_layer_cnt=3,
        value_tol=1e-7,
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
        value_tol=1e-7,
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
        value_tol=1e-7,
    ),
    # The six operators no other set draws: ``GT``(11) / ``LE``(12) from the
    # comparison family and ``SINH``(17) / ``COSH``(18) / ``NEG``(25) / ``ABS``(26)
    # from the unary one.  ``sinh``/``cosh`` are the hand-written
    # ``(exp(x) -+ exp(-x)) * 0.5`` compositions (``utils`` has no native branch
    # for a device ``sinh``), i.e. exactly where divergence from the oracle's
    # ``torch.sinh``/``torch.cosh`` would hide.  Measured BITWISE on both value
    # paths, hence ``value_tol=0.0``.
    "compare_unary": dict(
        using_funcs=(">", "<=", "neg", "abs", "sinh", "cosh", "+", "*"),
        input_len=8,
        max_layer_cnt=4,
        value_tol=0.0,
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


def _report(
    tag: str,
    stats: Dict[str, Any],
    limit: float,
    *,
    key: str,
    sink: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Print one comparison row and return whether it stayed inside ``limit``.

    ``sink`` collects the same row for a stage that wants to aggregate it (the
    ``--stage multi`` table); the parity stage passes ``None`` and prints exactly
    what it always printed.
    """
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
    if sink is not None:
        sink.append(dict(stats, tag=tag, key=key, limit=float(limit), ok=bool(ok)))
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


def out_node_tagged_row() -> Any:
    """One single-output row whose nodes carry the multi-output ``OUT_NODE`` tag.

    The population a MIS-WIRED ``output_len`` produces: built for
    ``output_len >= 2``, evaluated as single-output.  The oracle's
    ``_kind_from_type`` folds every tagged type into its TERNARY branch (so the
    kernel's own fall-through is defined and would return a number), which is
    exactly why the facade must refuse it before any launch -- the shared
    ``multi_output.check_multi_output`` rule, applied in
    ``fused_engine._prepare``.  ``--stage declines`` is where the refusal is
    pinned; every tagged type (``0x80|0 .. 0x80|4``) is covered, and each one
    carries three children because the oracle would evaluate it as a ternary.
    """
    return (
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


def adversarial_cases() -> List[Dict[str, Any]]:
    """Handmade populations: the shapes a random generator never produces.

    Every case carries a MEASURED ``bitwise`` flag: ``True`` means the value paths
    of this row are already bitwise against the oracle and the stage enforces
    ``value_limit=0.0``, ``False`` means the case has a documented reason (a NaN
    payload or a signed zero) for which a byte comparison says nothing, so the
    stage falls back to ``VALUE_RTOL`` -- and the ``note`` spells the reason out.
    """
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
    return [
        dict(
            name="leaf_var",
            rows=[("var", 0)],
            input_len=4,
            bitwise=True,
            note="single VAR root",
        ),
        dict(
            name="leaf_const",
            rows=[("const", 2.5)],
            input_len=4,
            bitwise=True,
            note="single CONST root (never compiled on the specialized route)",
        ),
        dict(
            name="const_tree",
            rows=[const_tree],
            input_len=4,
            bitwise=True,
            note="pure constant subtree",
        ),
        dict(
            name="left_chain_40",
            rows=[_chain(40, right=False, input_len=8)],
            input_len=8,
            bitwise=True,
            note="40-deep left chain, one pending operand",
        ),
        dict(
            name="right_chain_40",
            rows=[_chain(40, right=True, input_len=8)],
            input_len=8,
            bitwise=True,
            note="40-deep right chain, 40 pending operands",
        ),
        dict(
            name="full_if_3",
            rows=[_full_ternary(3, 4)],
            input_len=4,
            bitwise=True,
            note="complete ternary if tree (13 nodes)",
        ),
        dict(
            name="padded_garbage",
            rows=[(add, (("var", 0), ("const", 1.25)))],
            input_len=4,
            pad_to=64,
            garbage_tail=True,
            bitwise=True,
            note="5-node tree, NaN/nonsense padding behind it",
        ),
        dict(
            name="mixed_deep",
            rows=[mixed],
            input_len=4,
            # MEASURED: the values agree (max_abs 0.0) but the bytes do not, because
            # the DIV-by-a-literal-zero branch is a NaN and the two evaluators write
            # different NaN payloads -- a byte comparison is meaningless here.
            bitwise=False,
            note="DIV by const 0 (NaN per the emitter), SQRT/LOG of a var; "
            "NOT bitwise: NaN payload",
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
                mixed,
            ],
            input_len=4,
            pad_to=32,
            garbage_tail=True,
            # MEASURED: the same rows as above, so the same NaN-payload split.
            bitwise=False,
            note="all of the above in ONE population (pop 7); "
            "NOT bitwise: the NaN payload of the DIV-by-zero row",
        ),
    ]


def multi_adversarial_cases() -> List[Dict[str, Any]]:
    """Handmade MULTI-OUTPUT populations (``out_len >= 2``) for ``--stage multi``.

    Spec grammar is :func:`build_row`'s, plus one idiom: a node is tagged by giving
    it the raw form ``("raw", _out_node_value(function_id, out_index), 0x80 | base,
    children)``.  ``base`` (not the function id) fixes the ARITY, exactly as the
    protocol's decode does -- ``node_type & 0x7F`` selects the kind, so ``0x80|0`` is
    a tagged VAR, ``0x80|1`` a tagged CONST, ``0x80|2`` a tagged UFUNC (one child),
    ``0x80|3`` a tagged BFUNC (two), ``0x80|4`` a tagged TFUNC (three, and always the
    ternary ``IF``), while the operator itself comes out of the packed value bits
    (low half ``function``, high half ``outIndex``).

    The specs assume ``x`` is drawn in ``[0, 1)`` (so a VAR is positive and ``IF``'s
    ``cond > 0`` branch is exercised on both sides) and stay inside the arithmetic
    and ternary function table, which is the bitwise claim the kernel makes.

    Per-case keys beyond ``build_row``'s: ``out_len`` (or ``out_lens`` for a case
    evaluated at several), ``ops`` (the reported operator set), ``expect_zero``
    (output indices that must come out EXACTLY ``0.0``, a tuple or a
    ``{out_len: tuple}`` map), ``expect_loss_is_label_norm`` (every prediction is
    ``0.0``, so the MSE is ``sum(labels**2) / rows``) and ``expect_rows_differ``
    (two population rows that must produce different numbers, i.e. the case really
    exercises the operator it packs).
    """
    from evogp.tree.utils import Func

    add, sub, mul, neg, div = (
        int(Func.ADD),
        int(Func.SUB),
        int(Func.MUL),
        int(Func.NEG),
        int(Func.DIV),
    )
    if_op = int(Func.IF)

    def tagged(function_id: int, out_index: int, base: int, children: Sequence[Any]):
        """One ``OUT_NODE``-tagged node: ``out_index`` in the high, op id in the low half."""
        return ("raw", _out_node_value(function_id, out_index), 0x80 | base, tuple(children))

    # Two tagged outputs, one of them 7 nodes deep and one on the root's right arm.
    two_outputs = (
        add,
        (
            tagged(neg, 0, 2, (_chain(6, right=False, input_len=4),)),
            tagged(add, 1, 3, (("var", 1), ("var", 2))),
        ),
    )
    # The tag on the ROOT: a tagged TFUNC (always the ternary IF, arity 3).
    root_if = tagged(
        if_op,
        1,
        4,
        (
            ("var", 0),
            (add, (("var", 1), ("const", 0.5))),
            (mul, (("var", 2), ("var", 3))),
        ),
    )
    # Only output 1 is tagged; 0 and 2 must stay exactly 0.0.
    one_output = (
        add,
        (
            tagged(mul, 1, 3, (("var", 0), ("const", 2.5))),
            (sub, (("var", 1), ("var", 2))),
        ),
    )
    # Two nodes accumulate into the SAME output (out0 = var0 - 2*var1).
    same_output = tagged(add, 0, 3, (("var", 0), tagged(neg, 0, 2, (("var", 1),))))
    # Tagged LEAVES: a VAR and a CONST.  They decode as their untagged selves and
    # accumulate nothing (the oracle returns before it reads an out index).
    tagged_leaves = (
        add,
        (
            ("raw", 2.0, 0x80 | 0, ()),
            tagged(add, 1, 3, (("raw", 0.75, 0x80 | 1, ()), ("var", 0))),
        ),
    )
    # An outIndex of 5 at out_len 3 is dropped; output 0 is genuinely tagged.
    out_of_range = (
        add,
        (
            tagged(add, 5, 3, (("var", 0), ("var", 1))),
            tagged(neg, 0, 2, (("var", 2),)),
        ),
    )
    # One tagged node per arity, each with a real operator id in the packed half.
    three_arities = (
        add,
        (
            tagged(neg, 0, 2, (("var", 0),)),
            (
                add,
                (
                    tagged(sub, 1, 3, (("var", 1), ("const", 0.25))),
                    tagged(if_op, 2, 4, (("var", 2), ("var", 3), ("const", 1.5))),
                ),
            ),
        ),
    )
    # No tag at all: every output is 0.0, so the loss is exactly the label norm.
    no_tags = (add, (("var", 0), (div, (("var", 1), ("const", 2.0)))))
    # Same operands, different packed operator: the extraction has to matter.
    mul_row = tagged(mul, 0, 3, (("var", 0), ("var", 1)))
    add_row = tagged(add, 0, 3, (("var", 0), ("var", 1)))
    # The same population at out_len 2 and 3: the tag on index 2 is dropped at 2.
    index_two = (
        add,
        (
            tagged(neg, 0, 2, (("var", 0),)),
            tagged(add, 2, 3, (("var", 1), ("const", 0.5))),
        ),
    )
    # The widest served output count: one tagged output per index.
    from evogp.tree import fused_engine

    max_out_len = int(fused_engine.MAX_OUT_LEN)
    all_outputs: Any = ("var", 0)
    for index in range(max_out_len - 1, -1, -1):
        all_outputs = tagged(add, index, 3, (("var", index % 4), all_outputs))

    return [
        dict(
            name="two_outputs_deep_shallow",
            rows=[two_outputs],
            input_len=4,
            out_len=2,
            ops="ADD,NEG",
            note="two tagged outputs, one 7 nodes deep, one on the root's right arm",
        ),
        dict(
            name="root_tagged_ternary",
            rows=[root_if],
            input_len=4,
            out_len=3,
            ops="IF,ADD,MUL",
            expect_zero=(0, 2),
            note="the OUT_NODE as the ROOT (a tagged ternary IF); outputs 0 and 2 stay 0.0",
        ),
        dict(
            name="one_output_tagged",
            rows=[one_output],
            input_len=4,
            out_len=3,
            ops="ADD,MUL,SUB",
            expect_zero=(0, 2),
            note="only output 1 tagged; outputs 0 and 2 must be exactly 0.0",
        ),
        dict(
            name="same_output_twice",
            rows=[same_output],
            input_len=4,
            out_len=2,
            ops="ADD,NEG",
            note="two nodes tag the SAME output (out0 = var0 - 2*var1): accumulation",
        ),
        dict(
            name="tagged_leaves",
            rows=[tagged_leaves],
            input_len=4,
            out_len=2,
            ops="ADD",
            expect_zero=(0,),
            note="OUT_NODE-tagged VAR and CONST leaves: behave untagged, accumulate nothing",
        ),
        dict(
            name="out_index_out_of_range",
            rows=[out_of_range],
            input_len=4,
            out_len=3,
            ops="ADD,NEG",
            expect_zero=(1, 2),
            note="outIndex 5 at out_len 3 is dropped; outputs 1 and 2 stay 0.0",
        ),
        dict(
            name="three_arities",
            rows=[three_arities],
            input_len=4,
            out_len=3,
            ops="ADD,SUB,NEG,IF",
            note="one tagged node per arity: unary NEG, binary SUB, ternary IF",
        ),
        dict(
            name="no_tagged_node",
            rows=[no_tags],
            input_len=4,
            out_len=3,
            ops="ADD,DIV",
            expect_zero=(0, 1, 2),
            expect_loss_is_label_norm=True,
            note="no tag anywhere: all outputs 0.0, so MSE = sum(labels**2)/rows",
        ),
        dict(
            name="mul_vs_add_packed",
            rows=[mul_row, add_row],
            input_len=4,
            out_len=2,
            ops="ADD,MUL",
            expect_rows_differ=(0, 1),
            note="same operands, packed MUL vs ADD: the function extraction must matter",
        ),
        dict(
            name="tag_at_index_two",
            rows=[index_two],
            input_len=4,
            out_len=3,
            out_lens=(2, 3),
            ops="ADD,NEG",
            expect_zero={2: (1,), 3: ()},
            note="same population at out_len 2 and 3: the index-2 tag is dropped vs accumulated",
        ),
        dict(
            name="all_outputs",
            rows=[all_outputs],
            input_len=4,
            out_len=max_out_len,
            ops="ADD",
            note=f"one tagged output per index, the accumulator at its full width ({max_out_len})",
        ),
        dict(
            name="padded_garbage_multi",
            rows=[
                (add, (tagged(mul, 0, 3, (("var", 0), ("var", 1))), (neg, (("var", 2),))))
            ],
            input_len=4,
            out_len=2,
            ops="ADD,MUL,NEG",
            pad_to=64,
            garbage_tail=True,
            note="16-node multi-output tree, NaN/nonsense padding behind it",
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


def _oracle_loss(rows: Tuple[Any, Any, Any], x, labels, use_mse: bool, out_len: int = 1):
    import reference_eval as R

    node_value, node_type, subtree_size = rows
    return R.sr_fitness_from_rows(
        node_value.cpu(),
        node_type.cpu(),
        subtree_size.cpu(),
        x.cpu(),
        labels.cpu(),
        use_mse=use_mse,
        out_len=out_len,
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
    out_len: int = 1,
    sink: Optional[List[Dict[str, Any]]] = None,
    capture: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """All three surfaces of one population.  Returns ``(passed, total)``.

    ``out_len`` selects the protocol: ``1`` (the default, and every call site of
    the parity stage) is the single-output one, ``>= 2`` the ``OUT_NODE`` protocol
    the multi-output kernel implements.  ``sink`` collects the comparison rows and
    ``capture`` the raw ``(got, want)`` tensors, both for ``--stage multi``.
    """
    import torch

    from evogp.tree import fused_engine

    compare = _comparator()
    node_value, node_type, subtree_size = rows
    passed = total = 0
    if note:
        print(f"  {tag}: {note}", flush=True)
    if batch_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, out_len, batch_x, per_row=False
        )
        want = _oracle(rows, batch_x, out_len)
        if capture is not None:
            capture["batch_forward"] = (got, want)
        total += 1
        passed += _report(
            f"{tag}/batch_forward",
            compare(want, got),
            value_limit,
            key="value",
            sink=sink,
        )
        if labels is not None:
            for use_mse in (True, False):
                got_l = fused_engine.sr_fitness_fused(
                    node_value, node_type, subtree_size, input_len, out_len, batch_x, labels,
                    use_MSE=use_mse,
                )
                want_l = _oracle_loss(rows, batch_x, labels, use_mse, out_len=out_len)
                if capture is not None:
                    capture["sr_fitnessMSE" if use_mse else "sr_fitnessMAE"] = (
                        got_l,
                        want_l,
                        labels,
                    )
                total += 1
                passed += _report(
                    f"{tag}/sr_fitness{'MSE' if use_mse else 'MAE'}",
                    compare(want_l, got_l),
                    LOSS_RTOL,
                    key="loss",
                    sink=sink,
                )
    if per_row_x is not None:
        got = fused_engine.evaluate_fused(
            node_value, node_type, subtree_size, input_len, out_len, per_row_x, per_row=True
        )
        want = _oracle_per_row(rows, per_row_x, out_len)
        if capture is not None:
            capture["forward(per_row)"] = (got, want)
        total += 1
        passed += _report(
            f"{tag}/forward(per_row)",
            compare(want, got),
            value_limit,
            key="value",
            sink=sink,
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
        limit = float(spec["value_tol"])
        p, t = _check_population(
            name,
            rows,
            cell["input_len"],
            batch_x=workload["x"],
            per_row_x=workload["x_per_row"],
            labels=workload["labels"],
            value_limit=limit,
            note=(
                f"bitwise required (measured value_tol={limit:g})"
                if limit == 0.0
                else f"value <= {limit:g} rel (measured 0 with a margin)"
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
        print(f"\n[{case['name']}]")
        p, t = _check_population(
            case["name"],
            (node_value, node_type, subtree_size),
            input_len,
            batch_x=batch_x,
            per_row_x=per_row_x,
            labels=labels,
            value_limit=0.0 if case["bitwise"] else VALUE_RTOL,
            note=case["note"],
        )
        passed += p
        total += t
    print("\n" + "-" * 78)
    print(f"PARITY: {passed}/{total} comparisons inside tolerance")
    print("-" * 78)
    return 0 if passed == total else 1


# --------------------------------------------------------------------------- #
# stage 1b: the multi-output (OUT_NODE) protocol
# --------------------------------------------------------------------------- #
def _check_line(tag: str, ok: bool, detail: str = "") -> int:
    """Print one structural expectation of ``--stage multi``; ``1`` when it holds."""
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:<40} {detail}", flush=True)
    return 1 if ok else 0


def _multi_case_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a case's comparison rows: the value bounds and the loss bound."""
    value = [row for row in rows if row["key"] == "value"]
    loss = [row for row in rows if row["key"] == "loss"]
    return dict(
        value_max_abs=max((float(row.get("max_abs", 0.0)) for row in value), default=0.0),
        value_max_rel=max((float(row.get("max_rel", 0.0)) for row in value), default=0.0),
        value_bitwise=bool(value) and all(bool(row.get("bitwise_equal")) for row in value),
        loss_max_rel=max((float(row.get("max_rel", 0.0)) for row in loss), default=0.0),
        loss_max_abs=max((float(row.get("max_abs", 0.0)) for row in loss), default=0.0),
        nan_mismatch=sum(int(row.get("nan_mismatch", 0)) for row in rows),
        inf_mismatch=sum(int(row.get("inf_mismatch", 0)) for row in rows),
    )


def stage_multi(args: argparse.Namespace) -> int:
    """The multi-output protocol (``out_len`` 2..``MAX_OUT_LEN``) vs the oracle.

    The route serves ``output_len`` 1..:data:`fused_engine.MAX_OUT_LEN`: ``out_len
    == 1`` runs the original single-output kernel (what ``--stage parity`` pins),
    ``out_len >= 2`` the second kernel that resolves a tagged node's operator from
    the packed value bits, accumulates into ``out[outIndex]``, drops an index
    ``>= out_len``, ignores tagged leaves and leaves untagged outputs at ``0.0``.

    This stage runs :func:`multi_adversarial_cases` -- the shapes a random
    generator does not produce for that protocol -- on all four surfaces: the
    batched value path, the per-tree value path and BOTH loss paths (MSE and MAE),
    each against ``src/evogp/tests/reference_eval.py``.  The value path must be
    BITWISE (``value_limit = 0.0``: the multi-output kernel reuses the single-output
    per-op emitters) and the loss ``<= LOSS_RTOL`` relative.

    Cases also carry structural expectations that a numeric comparison alone would
    not pin: outputs no node tags must be EXACTLY ``0.0`` (an untagged or dropped
    index leaves the accumulator at its zeroed value), a population with no tag
    must lose exactly ``sum(labels**2) / rows``, and two rows that differ only in
    their packed operator must actually produce different numbers.
    """
    import torch

    print("=" * 78)
    print("MULTI: the multi-output (OUT_NODE) protocol vs "
          "src/evogp/tests/reference_eval.py")
    print("=" * 78)
    passed = total = 0
    records: List[Dict[str, Any]] = []
    for case in multi_adversarial_cases():
        rows = pack_rows(
            case["rows"],
            pad_to=int(case.get("pad_to", 0)),
            garbage_tail=bool(case.get("garbage_tail", False)),
        )
        input_len = int(case["input_len"])
        pop = int(rows[0].shape[0])
        out_lens = tuple(int(item) for item in case.get("out_lens") or (case["out_len"],))
        for out_len in out_lens:
            tag = f"{case['name']}/out{out_len}"
            torch.manual_seed(1234)
            batch_x = torch.rand(int(args.parity_rows), input_len, device="cuda")
            labels = torch.rand(int(args.parity_rows), out_len, device="cuda")
            per_row_x = torch.rand(pop, input_len, device="cuda")
            sink: List[Dict[str, Any]] = []
            capture: Dict[str, Any] = {}
            print(f"\n[{tag}] ops={case['ops']}")
            p, t = _check_population(
                tag,
                rows,
                input_len,
                batch_x=batch_x,
                per_row_x=per_row_x,
                labels=labels,
                value_limit=float(case.get("value_limit", 0.0)),
                note=case["note"],
                out_len=out_len,
                sink=sink,
                capture=capture,
            )
            passed += p
            total += t
            batch_got = capture["batch_forward"][0]
            per_row_got = capture["forward(per_row)"][0]
            # 1. every output no node tags must be EXACTLY 0.0 in BOTH value paths.
            expect_zero = case.get("expect_zero", ())
            if isinstance(expect_zero, dict):
                expect_zero = expect_zero.get(out_len, ())
            for index in expect_zero:
                batch_max = float(batch_got[:, :, index].abs().max())
                per_row_max = float(per_row_got[:, index].abs().max())
                total += 1
                passed += _check_line(
                    f"{tag}/out{index}==0.0",
                    max(batch_max, per_row_max) == 0.0,
                    f"max|batch|={batch_max:.3e} max|per_row|={per_row_max:.3e}",
                )
            # 2. a population with no tag at all predicts 0.0 everywhere, so its
            #    MSE is exactly the label norm (an independent, oracle-free cross-check).
            if case.get("expect_loss_is_label_norm"):
                got_loss = capture["sr_fitnessMSE"][0]
                expected = float((labels.to(torch.float32) ** 2).sum()) / float(
                    labels.shape[0]
                )
                ok = bool(
                    torch.allclose(
                        got_loss.to(torch.float32),
                        torch.full_like(got_loss, expected),
                        rtol=LOSS_RTOL,
                        atol=0.0,
                    )
                )
                total += 1
                passed += _check_line(
                    f"{tag}/loss==label_norm",
                    ok,
                    f"expected {expected:.6e}, got {float(got_loss.min()):.6e}.."
                    f"{float(got_loss.max()):.6e}",
                )
            # 3. two rows that differ only in their packed operator must differ.
            if case.get("expect_rows_differ"):
                left, right = case["expect_rows_differ"]
                delta = float((batch_got[left] - batch_got[right]).abs().max())
                total += 1
                passed += _check_line(
                    f"{tag}/rows {left} vs {right} differ",
                    delta > 0.0,
                    f"max|delta|={delta:.3e}",
                )
            aggregate = _multi_case_stats(sink)
            line_ok = all(bool(row["ok"]) for row in sink)
            print(
                f"  [{'ok ' if line_ok else 'FAIL'}] CASE {tag:<38} out_len={out_len} "
                f"ops={case['ops']:<16} max_abs={aggregate['value_max_abs']:.3e} "
                f"max_rel={aggregate['value_max_rel']:.3e} "
                f"bitwise={str(aggregate['value_bitwise']):<5} "
                f"loss_max_rel={aggregate['loss_max_rel']:.3e} "
                f"nan_mismatch={aggregate['nan_mismatch']}",
                flush=True,
            )
            records.append(
                dict(
                    case=case["name"],
                    out_len=out_len,
                    ops=case["ops"],
                    note=case["note"],
                    comparisons=len(sink),
                    comparisons_ok=line_ok,
                    rows=[
                        dict(
                            tag=row["tag"],
                            key=row["key"],
                            limit=row["limit"],
                            ok=row["ok"],
                            max_abs=row.get("max_abs", 0.0),
                            max_rel=row.get("max_rel", 0.0),
                            bitwise_equal=row.get("bitwise_equal"),
                            nan_mismatch=row.get("nan_mismatch", 0),
                            inf_mismatch=row.get("inf_mismatch", 0),
                        )
                        for row in sink
                    ],
                    **aggregate,
                )
            )
    print("\n" + "-" * 78)
    print(f"MULTI: {passed}/{total} comparisons inside tolerance")
    print("-" * 78)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.json}")
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
        print(
            f"  [skip] {name}: multi-output fixture (this stage is the single-output "
            "golden gate; --stage multi covers the protocol)"
        )
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
# stage 2b helpers: the regression contracts of the hardened kernel and facade
#
# Each helper appends ``(name, ok, detail)`` rows to the stage's ``checks`` list and
# is RUN WRAPPED (see ``stage_declines``), so a crash reports as one FAIL line
# instead of killing the matrix.  They reuse the battery's own machinery
# (``pack_rows`` / ``_full_ternary`` / ``_expect``) on purpose: a second way of
# building a population would be a second thing to get wrong.
# --------------------------------------------------------------------------- #
def _expect(tag: str, fn, kind, needle: Optional[str] = None) -> Tuple[str, bool, str]:
    """One "this call must raise ``kind``" expectation, as a ``checks`` row."""
    try:
        fn()
    except kind as exc:
        detail = f"{type(exc).__name__}: {exc}"
        ok = needle is None or needle in str(exc)
        if not ok:
            detail = f"{detail} (does not name {needle!r})"
        return (tag, ok, detail)
    except Exception as exc:  # noqa: BLE001 - the wrong type is a failure
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    return (tag, False, "did NOT raise")


def _dev_x(rows: int, input_len: int):
    """A deterministic ``(rows, input_len)`` device block (same values every call)."""
    import torch

    torch.manual_seed(4321)
    return torch.rand(int(rows), int(input_len), device="cuda")


def _fused_value(
    rows_specs: Sequence[Sequence[Any]],
    *,
    pad_to: int,
    input_len: int,
    data_rows: int,
    size0: Optional[int] = None,
    size_row: int = 0,
):
    """``evaluate_fused`` on a hand-built population, optionally with a forced size.

    ``size0`` overrides ``batch_subtree_size[size_row, 0]`` AFTER packing, which is
    the only way to build the malformed rows the kernel's clamp has to survive
    (``pack_rows`` always writes the honest subtree sizes).
    """
    from evogp.tree import fused_engine

    node_value, node_type, subtree_size = pack_rows(rows_specs, pad_to=pad_to)
    if size0 is not None:
        subtree_size = subtree_size.clone()
        subtree_size[int(size_row), 0] = int(size0)
    return fused_engine.evaluate_fused(
        node_value,
        node_type,
        subtree_size,
        int(input_len),
        1,
        _dev_x(data_rows, input_len),
        per_row=False,
    )


def _bitwise_row(tag: str, fn_got, fn_want, *, note: str = "") -> Tuple[str, bool, str]:
    """Compare two call results BITWISE (the contract of a layout/out_device leg)."""
    import torch

    try:
        got = fn_got()
        want = fn_want()
    except Exception as exc:  # noqa: BLE001
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    same = tuple(got.shape) == tuple(want.shape) and torch.equal(got, want)
    detail = f"bitwise={same} shape {tuple(got.shape)}"
    if not same and tuple(got.shape) == tuple(want.shape):
        detail += f" max|delta|={float((got.float() - want.float()).abs().max()):.3e}"
    return (tag, bool(same), detail + ((" " + note) if note else ""))


def _stats_row(tag: str, key: str) -> Tuple[str, bool, str]:
    """A stats snapshot must NEVER raise, whatever the ambient knob says (S7)."""
    from evogp.tree import fused_engine

    try:
        value = fused_engine.fused_cache_stats().get(key)
    except Exception as exc:  # noqa: BLE001
        return (tag, False, f"raised {type(exc).__name__}: {exc}")
    return (tag, value == -1, f"{key}={value} (-1 = present but unusable)")


def _check_row_length_clamp(checks: List[Tuple[str, bool, str]]) -> None:
    """B1: ``subtree_size[0]`` is clamped to the row (no spill, no out-of-bounds).

    The rows below are padded to EXACTLY their node count, so the clamp
    ``min(max(subtree_size[0], 0), tree_len)`` IS the honest length and the honest
    value is the assertion: a row that walked past its own cells would read the
    NEXT row's nodes (or the memory behind the last row) and produce a different
    number.  ``-5`` is the floor side of the same clamp: it floors to 0, i.e. the
    walk does nothing and the value is exactly 0.0.
    """
    import torch

    from evogp.tree.utils import Func

    add, mul = int(Func.ADD), int(Func.MUL)
    tree = _full_ternary(3, 4)  # exactly 13 nodes
    other = (mul, (("var", 0), (add, (("const", 1.5), ("var", 1)))))  # 5 nodes
    honest = _fused_value([tree], pad_to=13, input_len=4, data_rows=8)
    for size0 in (30000, 32767):
        forced = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=size0)
        checks.append(
            (
                f"B1: subtree_size[0]={size0} == the honest 13-node row",
                torch.equal(forced, honest),
                f"clamped to tree_len 13; "
                f"max|delta|={float((forced - honest).abs().max()):.3e}",
            )
        )
    floored = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=-5)
    empty = _fused_value([tree], pad_to=13, input_len=4, data_rows=8, size0=0)
    checks.append(
        (
            "B1: subtree_size[0]=-5 floors to 0 (no crash, exactly 0.0)",
            torch.equal(floored, empty) and float(floored.abs().max()) == 0.0,
            f"max|value|={float(floored.abs().max()):.3e}, same as size 0: "
            f"{torch.equal(floored, empty)}",
        )
    )
    spilled = _fused_value([tree, other], pad_to=13, input_len=4, data_rows=8, size0=32767)
    honest_pair = _fused_value([tree, other], pad_to=13, input_len=4, data_rows=8)
    checks.append(
        (
            "B1: 32767 on a row FOLLOWED by another tree does not spill",
            torch.equal(spilled, honest_pair),
            f"max|delta|={float((spilled - honest_pair).abs().max()):.3e} "
            "(row 1 holds a different tree, so an overshoot would show)",
        )
    )


def _check_underflow_determinism(checks: List[Tuple[str, bool, str]]) -> None:
    """S8: a malformed tree popping an EMPTY stack yields a defined, stable value.

    ``MUL(VAR0, <nothing>)`` is two nodes: the walk reaches the binary root with an
    EMPTY stack.  The scratch block is shared per shape class, so the two junk
    populations below leave DIFFERENT bytes in the slot that pop would read -- one
    masked operand (``other=0.0``) or the other, the malformed result must be the
    same.  Before the mask it was the PREVIOUS call's leftover.
    """
    import torch

    from evogp.tree.utils import Func

    add, mul = int(Func.ADD), int(Func.MUL)

    def junk(value: float):
        """A right-leaning const chain: it leaves a ``2 * value``-ish sum on the stack."""
        node: Any = ("const", value)
        for _ in range(3):
            node = (add, (("const", value), node))
        return node

    def run(spec) -> Any:
        return _fused_value([spec], pad_to=8, input_len=4, data_rows=8)

    malformed = (mul, (("var", 0),))
    first_junk = run(junk(5.0))
    first = run(malformed)
    second_junk = run(junk(-7.0))
    second = run(malformed)
    checks.append(
        (
            "S8: an empty-stack pop is deterministic across junk fills",
            torch.equal(first, second),
            f"junk max {float(first_junk.max()):.3e} vs "
            f"{float(second_junk.max()):.3e} -> value "
            f"{float(first.min()):.3e} / {float(second.min()):.3e}",
        )
    )
    checks.append(
        (
            "S8: the masked (missing) operand is exactly 0.0",
            float(first.abs().max()) == 0.0 and float(second.abs().max()) == 0.0,
            f"max|value|={float(first.abs().max()):.3e} / "
            f"{float(second.abs().max()):.3e}",
        )
    )


def _check_geometry_knobs(checks: List[Tuple[str, bool, str]]) -> None:
    """S3/S7: an off-power-of-two block cap is floored, and a bad knob never raises.

    ``tl.arange`` needs a power of two, so ``EVOGP_FUSED_BLOCK_ROWS=100`` used to
    fail INSIDE Triton; the resolver and ``plan`` now floor it to 64.  An unusable
    spelling still refuses the call (``ValueError``) while a stats SNAPSHOT -- the
    one thing a driver prints unconditionally -- reports ``-1`` instead.
    """
    import torch

    from evogp.tree import fused_engine, fused_kernels

    node_value, node_type, subtree_size = pack_rows([("var", 0)], pad_to=4)
    x = torch.rand(8, 4, device="cuda")
    call = dict(
        batch_node_value=node_value,
        batch_node_type=node_type,
        batch_subtree_size=subtree_size,
        input_len=4,
        output_len=1,
    )
    names = ("EVOGP_FUSED_BLOCK_ROWS", "EVOGP_FUSED_NUM_WARPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["EVOGP_FUSED_BLOCK_ROWS"] = "100"
        try:
            out = fused_engine.evaluate_fused(**call, x=x, per_row=False)
            served, detail = tuple(out.shape) == (1, 8, 1), f"shape {tuple(out.shape)}"
        except Exception as exc:  # noqa: BLE001
            served, detail = False, f"raised {type(exc).__name__}: {exc}"
        checks.append(("S3: BLOCK_ROWS=100 is SERVED (floored to 64)", served, detail))
        geom = fused_kernels.plan(1024, per_row=False, block_rows=100, num_warps=1)
        checks.append(
            ("S3: plan(block_rows=100) floors the block to 64", int(geom["block_rows"]) == 64,
             f"block_rows={geom['block_rows']} n_rb={geom['n_rb']}")
        )
        reported = fused_engine.fused_cache_stats()["fused_block_rows"]
        checks.append(
            ("S3: stats report the floored cap (64, not 100)", reported == 64, str(reported))
        )
        os.environ["EVOGP_FUSED_BLOCK_ROWS"] = "abc"
        checks.append(_stats_row("S7: fused_cache_stats() with BLOCK_ROWS=abc", "fused_block_rows"))
        checks.append(
            _expect(
                "S3: resolve_block_rows() still refuses 'abc'",
                fused_engine.resolve_block_rows,
                ValueError,
                "EVOGP_FUSED_BLOCK_ROWS",
            )
        )
        os.environ.pop("EVOGP_FUSED_BLOCK_ROWS", None)
        os.environ["EVOGP_FUSED_NUM_WARPS"] = "0"
        checks.append(_stats_row("S7: fused_cache_stats() with NUM_WARPS=0", "fused_num_warps"))
        checks.append(
            _expect(
                "S7: resolve_num_warps() still refuses '0'",
                fused_engine.resolve_num_warps,
                ValueError,
                "EVOGP_FUSED_NUM_WARPS",
            )
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _check_scratch_budget(checks: List[Tuple[str, bool, str]]) -> None:
    """S1: the scratch table is bounded by BYTES, and a smaller budget evicts.

    It used to be bounded by an entry COUNT (8), i.e. up to 8x the envelope.  Two
    2 MiB shapes are allocated, then a call with a 1 MiB budget has to EVICT them
    rather than add to them.
    """
    import torch

    from evogp.tree import fused_engine

    def serve(pop: int, tree_len: int, rows: int, **kwargs) -> None:
        node_value, node_type, subtree_size = pack_rows(
            [("var", 0)] * int(pop), pad_to=int(tree_len)
        )
        fused_engine.evaluate_fused(
            node_value,
            node_type,
            subtree_size,
            4,
            1,
            torch.rand(int(rows), 4, device="cuda"),
            per_row=False,
            **kwargs,
        )

    serve(64, 32, 256)  # 2 MiB of scratch
    serve(128, 16, 256)  # another 2 MiB
    stats = fused_engine.fused_cache_stats()
    budget = fused_engine.resolve_max_scratch_mb() * 2**20
    checks.append(
        (
            "S1: fused_scratch_bytes stays inside the resolved budget",
            stats["fused_scratch_bytes"] <= budget and stats["fused_scratch_entries"] >= 2,
            f"{stats['fused_scratch_bytes'] / 2**20:.2f} MiB in "
            f"{stats['fused_scratch_entries']} entries <= {budget / 2**20:.0f} MiB",
        )
    )
    before = fused_engine.fused_cache_stats()
    # A shape class the table has never seen (a reused one would be a HIT and prove
    # nothing) with a budget the entries above cannot fit in.
    serve(3, 5, 8, max_scratch_mb=1)
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "S1: a shrinking budget EVICTS instead of growing the table",
            after["fused_scratch_evictions"] > before["fused_scratch_evictions"]
            and after["fused_scratch_bytes"] <= 2**20,
            f"evictions +{after['fused_scratch_evictions'] - before['fused_scratch_evictions']}, "
            f"bytes {before['fused_scratch_bytes'] / 2**20:.2f} -> "
            f"{after['fused_scratch_bytes'] / 2**20:.2f} MiB",
        )
    )


def _check_operator_table(checks: List[Tuple[str, bool, str]]) -> None:
    """S5: the kernels' emitted operator table IS ``utils.FUNCS``, split by arity.

    A function id with no branch evaluates to ``0.0`` silently, so this assertion is
    the battery's half of the guard (the import-time check in ``fused_kernels`` is
    the other half): removing a branch or adding an operator has to fail here.
    """
    from evogp.tree import fused_kernels
    from evogp.tree.utils import FUNCS, Func

    declared = fused_kernels.emitted_function_ids()
    binary = tuple(f for f in FUNCS if Func.BF_START <= f < Func.UF_START)
    unary = tuple(f for f in FUNCS if Func.UF_START <= f < Func.END)
    checks.append(
        (
            "S5: emitted_function_ids() == utils.FUNCS split by arity",
            tuple(declared["binary"]) == binary and tuple(declared["unary"]) == unary,
            f"binary {len(declared['binary'])}/{len(binary)} ids "
            f"{sorted(binary)[0]}..{sorted(binary)[-1]}, "
            f"unary {len(declared['unary'])}/{len(unary)} ids "
            f"{sorted(unary)[0]}..{sorted(unary)[-1]}",
        )
    )


def _check_noncontiguous(checks: List[Tuple[str, bool, str]]) -> None:
    """B2: a non-contiguous input / label / population block is read correctly.

    The kernel dereferences raw pointers derived from the SHAPE, so a column slice
    of a padded ``x`` (or a stride-2 population table) would be read with the wrong
    strides and yield plausible garbage -- no exception, no decline, a wrong answer.
    Every leg therefore compares against the same call on ``.contiguous()`` copies,
    bitwise, through the PUBLIC API.
    """
    import torch

    from evogp.tree import Forest
    from evogp.tree.utils import Func

    add = int(Func.ADD)
    pop, input_len, tree_len, rows = 64, 4, 8, 32
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * pop, pad_to=tree_len
    )
    forest = Forest(input_len, 1, node_value, node_type, subtree_size)
    saved = os.environ.get("EVOGP_EVAL_MODE")
    os.environ["EVOGP_EVAL_MODE"] = "fused"  # >= EVOGP_FUSED_MIN_POP, so it is SERVED
    try:
        torch.manual_seed(99)
        padded = torch.rand(rows, input_len + 8, device="cuda")
        sliced = padded[:, :input_len]
        checks.append(
            _bitwise_row(
                "B2: batch_forward on a column slice == contiguous",
                lambda: forest.batch_forward(sliced),
                lambda: forest.batch_forward(sliced.contiguous()),
                note=f"(contiguous={sliced.is_contiguous()})",
            )
        )
        padded_row = torch.rand(pop, input_len + 8, device="cuda")
        row_slice = padded_row[:, :input_len]
        checks.append(
            _bitwise_row(
                "B2: forward(per_row) on a column slice == contiguous",
                lambda: forest.forward(row_slice),
                lambda: forest.forward(row_slice.contiguous()),
                note=f"(contiguous={row_slice.is_contiguous()})",
            )
        )
        labels = torch.rand(rows, 3, device="cuda")[:, :1]
        checks.append(
            _bitwise_row(
                "B2: SR_fitness with sliced inputs/labels == contiguous",
                lambda: forest.SR_fitness(sliced, labels),
                lambda: forest.SR_fitness(sliced.contiguous(), labels.contiguous()),
                note=f"(labels contiguous={labels.is_contiguous()})",
            )
        )

        def strided(table):
            """The same values in a stride-2 view (the population tables themselves)."""
            wide = torch.zeros(
                (table.shape[0], 2 * table.shape[1]),
                dtype=table.dtype,
                device=table.device,
            )
            wide[:, ::2] = table
            return wide[:, ::2]

        sliced_forest = Forest(
            input_len, 1, strided(node_value), strided(node_type), strided(subtree_size)
        )
        plain_forest = Forest(
            input_len, 1, node_value, node_type, subtree_size
        )
        checks.append(
            _bitwise_row(
                "B2: a stride-2 population equals its contiguous copy",
                lambda: sliced_forest.batch_forward(sliced),
                lambda: plain_forest.batch_forward(sliced),
                note="(population tables non-contiguous="
                f"{not sliced_forest.batch_subtree_size.is_contiguous()})",
            )
        )
    finally:
        if saved is None:
            os.environ.pop("EVOGP_EVAL_MODE", None)
        else:
            os.environ["EVOGP_EVAL_MODE"] = saved


def _check_out_device(checks: List[Tuple[str, bool, str]]) -> None:
    """``out_device``: a host request returns the SAME numbers; a same-device one is a no-op."""
    import torch

    from evogp.tree import fused_engine
    from evogp.tree.utils import Func

    add = int(Func.ADD)
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * 4, pad_to=8
    )
    rows = (node_value, node_type, subtree_size)
    x = _dev_x(16, 4)
    labels = _dev_x(16, 1)
    device = fused_engine.evaluate_fused(*rows, 4, 1, x, per_row=False)
    host = fused_engine.evaluate_fused(*rows, 4, 1, x, per_row=False, out_device="cpu")
    checks.append(
        (
            "out_device='cpu': evaluate_fused values are unchanged",
            host.device.type == "cpu" and torch.equal(host, device.cpu()),
            f"device {host.device} bitwise={torch.equal(host, device.cpu())}",
        )
    )
    same = fused_engine.evaluate_fused(
        *rows, 4, 1, x, per_row=False, out_device=device.device
    )
    checks.append(
        (
            "out_device == the computation device: a no-op",
            same.device == device.device and torch.equal(same, device),
            f"device {same.device} bitwise={torch.equal(same, device)}",
        )
    )
    loss = fused_engine.sr_fitness_fused(*rows, 4, 1, x, labels)
    loss_host = fused_engine.sr_fitness_fused(*rows, 4, 1, x, labels, out_device="cpu")
    close = bool(
        torch.allclose(loss_host, loss.cpu(), rtol=LOSS_RTOL, atol=0.0)
        if not torch.equal(loss_host, loss.cpu())
        else True
    )
    checks.append(
        (
            "out_device='cpu': sr_fitness_fused values are unchanged",
            loss_host.device.type == "cpu" and close,
            f"device {loss_host.device} "
            f"bitwise={torch.equal(loss_host, loss.cpu())}",
        )
    )


def _check_envelope_bytes(checks: List[Tuple[str, bool, str]]) -> None:
    """S2: the envelope charges the bytes a launch ACTUALLY allocates.

    ``p2048-L128-i32-N1024`` at ``out_len=8`` is the shipped ``d7`` shape: the OLD
    ``pop * tree_len * rows * 4`` formula reads 1 024 MiB and fits a 1 050 MiB
    budget, while the scratch (1 024 MiB, rounded to 4 blocks of 256 lanes) plus the
    ``(pop, rows, out_len)`` value buffer (64 MiB) is 1 088 MiB and must decline.
    """
    from evogp.tree import fused_engine, fused_kernels

    pop, tree_len, rows, out_len, budget_mb = 2048, 128, 1024, 8, 1050
    geom = fused_kernels.plan(
        rows,
        per_row=False,
        block_rows=fused_engine.resolve_block_rows(),
        num_warps=fused_engine.resolve_num_warps(),
    )
    scratch = pop * int(geom["n_rb"]) * tree_len * int(geom["block_rows"]) * 4
    values = pop * rows * out_len * 4
    old = pop * tree_len * rows * 4
    checks.append(
        (
            "S2: the shape fits the PRE-FIX formula but not the real bytes",
            old <= budget_mb * 2**20 < scratch + values,
            f"old {old / 2**20:.0f} MiB <= {budget_mb} MiB < scratch "
            f"{scratch / 2**20:.0f} + values {values / 2**20:.0f} MiB",
        )
    )
    checks.append(
        _expect(
            "S2: p2048-L128 out_len=8 declines at 1050 MiB",
            lambda: fused_engine._check_envelope(
                pop, tree_len, rows, budget_mb, out_len=out_len
            ),
            fused_engine.FusedUnsupported,
            "value buffer",
        )
    )
    budget = fused_engine._check_envelope(pop, tree_len, rows, 8192)
    checks.append(
        (
            "S2: p2048-L128-i32-N1024-d7 returns the budget at 8192",
            budget == 8192,
            f"returned {budget!r}",
        )
    )


def _check_decline_fallback(checks: List[Tuple[str, bool, str]]) -> None:
    """B3: a DECLINED fused call lands on specialized+``numpy`` -- the explicit arm too.

    ``EVOGP_EVAL_MODE=fused`` + ``EVOGP_EVOXC_BACKEND=iree`` used to keep ``iree`` as
    the fallback, i.e. a decline paid a per-tree iree compile.  The proof is the
    ``backend=`` keyword the specialized engine actually receives: a monkeypatched
    ``evoxc_engine.evaluate`` records it (restored in a ``finally``).
    """
    import torch

    from evogp.tree import Forest, dense_engine, evoxc_engine, fused_engine
    from evogp.tree.utils import Func

    route = dense_engine.resolve_route(256, backend="iree", eval_mode="fused")
    checks.append(
        (
            "B3: resolve_route(256, iree, fused) falls back to 'numpy'",
            (route.backend, route.fallback_backend) == ("iree", "numpy"),
            f"backend {route.backend} / fallback {route.fallback_backend}",
        )
    )
    add = int(Func.ADD)
    pop, input_len, tree_len, rows = 32, 4, 64, 1024
    node_value, node_type, subtree_size = pack_rows(
        [(add, (("var", 0), ("var", 1)))] * pop, pad_to=tree_len
    )
    forest = Forest(input_len, 1, node_value, node_type, subtree_size)
    x = _dev_x(rows, input_len)
    names = ("EVOGP_EVAL_MODE", "EVOGP_EVOXC_BACKEND", "EVOGP_FUSED_MAX_SCRATCH_MB")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["EVOGP_EVAL_MODE"] = "fused"
    os.environ["EVOGP_EVOXC_BACKEND"] = "iree"
    os.environ["EVOGP_FUSED_MAX_SCRATCH_MB"] = "1"  # 8 MiB of scratch: a real decline
    recorded: Dict[str, Any] = {}
    original = evoxc_engine.evaluate

    def spy(population, inputs, **kwargs):
        recorded["backend"] = kwargs.get("backend")
        return original(population, inputs, **kwargs)

    before = fused_engine.fused_cache_stats()
    evoxc_engine.evaluate = spy
    try:
        out = forest.batch_forward(x)
        served, detail = tuple(out.shape) == (pop, rows, 1), f"shape {tuple(out.shape)}"
    except Exception as exc:  # noqa: BLE001
        served, detail = False, f"raised {type(exc).__name__}: {exc}"
    finally:
        evoxc_engine.evaluate = original
        after = fused_engine.fused_cache_stats()
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    checks.append(
        ("B3: the declined call is still SERVED by the fallback", served, detail)
    )
    checks.append(
        (
            "B3: the fallback ran on the specialized engine's numpy backend",
            recorded.get("backend") == "numpy",
            f"recorded backend={recorded.get('backend')!r} "
            "(EVOGP_EVOXC_BACKEND=iree, EVOGP_FUSED_MAX_SCRATCH_MB=1)",
        )
    )
    checks.append(
        (
            "B3: the fused route did DECLINE (the fallback is what served it)",
            after["fused_declines"] == before["fused_declines"] + 1,
            f"declines +{after['fused_declines'] - before['fused_declines']}",
        )
    )


#: The hardened-contract helpers ``stage_declines`` runs (wrapped, one FAIL each).
CONTRACT_CHECKS = (
    _check_row_length_clamp,
    _check_underflow_determinism,
    _check_geometry_knobs,
    _check_operator_table,
    _check_scratch_budget,
    _check_envelope_bytes,
    _check_noncontiguous,
    _check_out_device,
    _check_decline_fallback,
)


# --------------------------------------------------------------------------- #
def stage_declines(args: argparse.Namespace) -> int:
    """``FusedUnsupported`` on every unsupported call, and the stats/knob contracts.

    The route's whole safety story is "decline, never a wrong number", so every
    decline reason is exercised here (one case per reason), together with the
    POSITIVE controls (a legal call served at ``pop == 1``, with the ``iree``
    backend name and -- since the multi-output kernel landed -- at ``output_len 2``,
    whose counters are checked too) and the invariants the rest of the package
    relies on: ``fused_cache_stats()`` is all ``fused_*`` keys, ``cache_stats()``
    stays at its pinned 8 keys, and ``evoxc_engine.clear_cache()`` resets the fused
    tables too.

    The multi-output matrix is explicitly TWO-SIDED: ``output_len 2`` is SERVED
    (it used to be the decline case) while ``MAX_OUT_LEN + 1`` declines with a
    reason naming ``MAX_OUT_LEN`` and ``output_len 0`` is a ``ValueError`` (an
    unusable argument, not a decline).

    The second half of the stage is the set of REGRESSION CONTRACTS the hardened
    kernel/facade earned (``CONTRACT_CHECKS``): an oversized / negative
    ``subtree_size[0]`` is clamped to the row (B1), an empty-stack pop is a
    deterministic masked ``0.0`` rather than the previous call's scratch bytes
    (S8), an off-power-of-two ``EVOGP_FUSED_BLOCK_ROWS`` is floored to 64 and an
    unusable knob reports ``-1`` from the stats snapshot instead of raising (S3 /
    S7), the scratch table is bounded by BYTES and evicts on a smaller budget
    (S1), ``emitted_function_ids()`` is exactly ``utils.FUNCS`` by arity (S5), the
    envelope charges the scratch + value buffer it really allocates (S2),
    non-contiguous inputs / labels / population tables give the same BITWISE
    numbers as their ``.contiguous()`` copies (B2), ``out_device`` changes
    placement only (never the numbers) and a DECLINED fused call on the explicit
    ``iree`` spelling still lands on specialized+``numpy`` (B3).
    """
    import torch

    from evogp.tree import evoxc_engine
    from evogp.tree import fused_engine
    from evogp.tree import multi_output

    checks: List[Tuple[str, bool, str]] = []

    def expect(name: str, fn, kind, needle: Optional[str] = None) -> None:
        checks.append(_expect(name, fn, kind, needle))

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
    # The multi-output protocol is SERVED (out_len 2..MAX_OUT_LEN): the shape and
    # the counters, not a decline, are what this matrix pins now.
    before = fused_engine.fused_cache_stats()
    try:
        multi = fused_engine.evaluate_fused(**{**call, "output_len": 2}, x=x, per_row=False)
        shape_ok = tuple(multi.shape) == (1, 8, 2) and multi.dtype == torch.float32
        checks.append(
            (
                "served: output_len=2 (multi-output)",
                shape_ok,
                f"shape {tuple(multi.shape)} dtype {multi.dtype}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            ("served: output_len=2 (multi-output)", False, f"raised {type(exc).__name__}: {exc}")
        )
    after = fused_engine.fused_cache_stats()
    checks.append(
        (
            "served: output_len=2 counted as a CALL, not a decline",
            after["fused_calls"] == before["fused_calls"] + 1
            and after["fused_declines"] == before["fused_declines"],
            f"calls +{after['fused_calls'] - before['fused_calls']} "
            f"declines +{after['fused_declines'] - before['fused_declines']}",
        )
    )
    expect(
        f"decline: output_len=MAX_OUT_LEN+1 ({fused_engine.MAX_OUT_LEN + 1})",
        lambda: fused_engine.evaluate_fused(
            **{**call, "output_len": fused_engine.MAX_OUT_LEN + 1}, x=x, per_row=False
        ),
        fused_engine.FusedUnsupported,
        needle="MAX_OUT_LEN",
    )
    expect(
        "raise: output_len=0",
        lambda: fused_engine.evaluate_fused(**{**call, "output_len": 0}, x=x, per_row=False),
        ValueError,
        needle="output_len",
    )
    # The single-output protocol refuses the multi-output (OUT_NODE) tag BEFORE any
    # launch: the kernel's type dispatch would otherwise fold a tagged type into its
    # TERNARY branch and return a plausible wrong number.  The refusal is the shared
    # ``multi_output.check_multi_output`` rule -- the same message the dense and
    # specialized decoders raise -- on BOTH entry points, and it costs no decline.
    tagged_rows = pack_rows([out_node_tagged_row()], pad_to=8)
    before_tag = fused_engine.fused_cache_stats()
    for tag, refused in (
        (
            "raise: single-output value path over an OUT_NODE-tagged population",
            lambda: fused_engine.evaluate_fused(*tagged_rows, 4, 1, x, per_row=False),
        ),
        (
            "raise: single-output loss path over an OUT_NODE-tagged population",
            lambda: fused_engine.sr_fitness_fused(*tagged_rows, 4, 1, x, labels),
        ),
    ):
        expect(tag, refused, multi_output.MultiOutputError, needle="OUT_NODE")
    after_tag = fused_engine.fused_cache_stats()
    checks.append(
        (
            "raise: the OUT_NODE refusal is not a DECLINE and launches nothing",
            after_tag["fused_declines"] == before_tag["fused_declines"]
            and after_tag["fused_kernel_launches"] == before_tag["fused_kernel_launches"],
            f"declines +{after_tag['fused_declines'] - before_tag['fused_declines']} "
            f"launches +{after_tag['fused_kernel_launches'] - before_tag['fused_kernel_launches']}",
        )
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
    # The envelope is a budget over the bytes a launch actually allocates (scratch
    # + value buffer + loss chart), so the check is landed with a legal but tiny
    # budget (a non-positive one is a knob error, raised as ``ValueError`` above) on
    # the ``p256-L64-i32-N1024`` shape: it must name the components it summed.
    expect(
        "decline: scratch envelope (1 MiB on p256-L64-N1024)",
        lambda: fused_engine._check_envelope(256, 64, 1024, 1),
        fused_engine.FusedUnsupported,
        needle="value buffer",
    )
    # The one-sided control: the SAME shape inside the budget RETURNS the resolved
    # budget in MiB (the entry points pass it down as the scratch table's budget;
    # it used to return ``None``).
    served_budget = fused_engine._check_envelope(256, 64, 1024, 8192)
    checks.append(
        (
            "served: same shape inside an 8192 MiB envelope",
            served_budget == 8192,
            f"returned {served_budget!r} MiB",
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
            fused_engine.resolve_min_pop() == 16,
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

    # the hardened kernel/facade contracts (B1, B2, B3, S1, S2, S3, S5, S7, S8).
    # Wrapped: a helper that raises is ONE failed line, not a dead matrix.
    for helper in CONTRACT_CHECKS:
        try:
            helper(checks)
        except Exception as exc:  # noqa: BLE001
            checks.append(
                (
                    f"{helper.__name__} did not run",
                    False,
                    f"raised {type(exc).__name__}: {exc}",
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
    above the shipped ``EVOGP_FUSED_MIN_POP`` (16) with plenty of headroom.
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
       any population below ``EVOGP_FUSED_MIN_POP``, and is ``specialized``+``numpy``
       when ``device_available()`` is False.  ``auto`` -- the shipped default --
       SELECTS it from ``EVOGP_FUSED_MIN_POP`` up when the box can run it (no
       explicit backend), and an explicit backend keeps the historical rule;
       ``should_use_dense`` is False for it and the ``Route`` invariant refuses
       ``use_dense=True`` with ``engine="fused"``.

    A multi-output population (``output_len=2``) must now be SERVED by the
    multi-output kernel rather than declined: the same numbers as the specialized
    fallback, one ``fused_calls``, no ``fused_declines``, and a real launch (the
    kernel and the scratch buffer both move).  ``--stage multi`` is the numeric
    gate of that protocol; this stage only proves the POLICY reaches it.
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
            minimum == fused_engine.resolve_min_pop() == 16,
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
        # The gate's boundary is spelled out: the shipped default threshold is 16,
        # so pop 15 is the last population the specialized fallback serves, pop 16
        # the first one the route serves, and pop 0 (the degenerate empty
        # population) declines instead of launching a kernel.
        for pop in (0, 1, 15):
            below = dense_engine.resolve_route(pop, eval_mode="fused")
            check(
                f"policy: pop_size={pop} < 16 -> specialized+numpy",
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
        # A CPU box: the pinned probe SNAPSHOT is the battery's stand-in for
        # unplugging the GPU.  ``device_available()`` resolves ``_DEVICE_AVAILABLE``
        # once per process (``test_route_policy`` pins the snapshot for the same
        # reason), so replacing the FUNCTION only simulates a CPU box while that
        # snapshot is still ``None`` -- i.e. it was order-dependent: a stage that
        # had already resolved the probe (or a warm route-memo entry keyed on the
        # resolved value) made the patched call answer "device available" anyway.
        # Pinning both keeps the simulation real in any order.
        original_probe = dense_engine.device_available
        original_snapshot = dense_engine._DEVICE_AVAILABLE
        dense_engine.device_available = lambda: False
        dense_engine._DEVICE_AVAILABLE = False
        try:
            cpu_route = dense_engine.resolve_route(256, eval_mode="fused")
        finally:
            dense_engine.device_available = original_probe
            dense_engine._DEVICE_AVAILABLE = original_snapshot
        check(
            "policy: no usable device -> specialized+numpy (engine stays default)",
            (cpu_route.engine, cpu_route.backend, cpu_route.fallback_backend)
            == ("specialized", "numpy", "numpy"),
            cpu_route.reason,
        )
        engines = {}
        for pop in (1, 16, 63, 64, 256, 1024, 2048):
            for backend in (None, "numpy", "iree"):
                for rows in (None, 1024):
                    route = dense_engine.resolve_route(
                        pop, backend=backend, rows=rows, eval_mode="auto"
                    )
                    engines[(pop, backend)] = route.engine
        if fused_engine.fused_available():
            # The fused route is the DEFAULT here: `auto` picks it from
            # EVOGP_FUSED_MIN_POP up whenever it owns the backend, and an explicit
            # backend keeps the historical dense/specialized rule.
            served = sorted(
                pop for (pop, backend), engine in engines.items()
                if backend is None and engine == "fused"
            )
            check(
                "policy: auto selects fused from EVOGP_FUSED_MIN_POP up",
                served and min(served) >= minimum
                and all(engines[(pop, None)] == "fused" for pop in served)
                and engines[(1, None)] != "fused",
                "fused at pop: " + ", ".join(str(pop) for pop in served),
            )
            check(
                "policy: an explicit backend never reaches fused",
                all(
                    engine != "fused"
                    for (pop, backend), engine in engines.items()
                    if backend is not None
                ),
                "explicit-backend engines: "
                + ", ".join(
                    sorted(
                        {
                            engine
                            for (pop, backend), engine in engines.items()
                            if backend is not None
                        }
                    )
                ),
            )
        else:
            check(
                "policy: auto never selects fused without triton/device",
                all(engine != "fused" for engine in engines.values()),
                "engines seen: " + ", ".join(sorted(set(engines.values()))),
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

        # ---- 3. a multi-output population: served by the second kernel --- #
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
            "wired output_len=2: SERVED by the multi-output kernel (same values)",
            _close(stats, VALUE_RTOL),
            f"bitwise={stats.get('bitwise_equal')} "
            f"max_rel={stats.get('max_rel', 0.0):.3e} (<= {VALUE_RTOL:g})",
        )
        check(
            "wired output_len=2: a real fused launch (kernel + scratch)",
            after["fused_kernel_launches"] > before["fused_kernel_launches"]
            and after["fused_scratch_allocs"] >= before["fused_scratch_allocs"],
            "launches +"
            f"{after['fused_kernel_launches'] - before['fused_kernel_launches']} "
            f"scratch_allocs +{after['fused_scratch_allocs'] - before['fused_scratch_allocs']}",
        )
        check(
            "wired output_len=2: served, not declined",
            after["fused_calls"] == before["fused_calls"] + 1
            and after["fused_declines"] == before["fused_declines"],
            f"calls +{after['fused_calls'] - before['fused_calls']} "
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
            "multi",
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
    # ``fused_kernels.DEFAULT_NUM_WARPS`` is the SHIPPED default (1); the parser is
    # built before any ``evogp`` import (so ``python src/evogp/tree/fused_check.py``
    # stays usable), hence the lazy resolution in ``main``.
    parser.add_argument(
        "--warps",
        type=int,
        default=None,
        help="EVOGP_FUSED_NUM_WARPS for tune-one (default: fused_kernels.DEFAULT_NUM_WARPS)",
    )
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


def _setup_paths() -> None:
    """Make the in-repo oracle (``src/evogp/tests``) importable.

    Called from :func:`main` — and therefore from every ``-m`` child leg the driver
    stages spawn — so merely IMPORTING this module never mutates ``sys.path``.
    """
    if _TESTS_DIR not in sys.path:
        sys.path.append(_TESTS_DIR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_paths()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.only:
        args.only = [item.strip() for item in args.only.split(",") if item.strip()]
    if args.stage in ("time-one", "tune-one") and not args.cell:
        parser.error(f"--stage {args.stage} needs --cell")
    if args.stage == "tune-one" and args.warps is None:
        # ``--warps``' default IS the shipped kernel default (single-sourced).
        from evogp.tree import fused_kernels

        args.warps = int(fused_kernels.DEFAULT_NUM_WARPS)
    status = 0
    if args.stage in ("parity", "all"):
        status |= stage_parity(args)
    if args.stage in ("multi", "all"):
        status |= stage_multi(args)
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
