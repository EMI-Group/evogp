"""Population builders and the comparison helper of the ``evogp.fused`` battery.

:mod:`evogp.fused.check` is the ``__main__``-guarded dev battery of the fused
(Triton) route.  This module holds the half of it that is reusable and pure
host code: how a population and its data are BUILT and how a result is
COMPARED.  It was split out of the 3 230-line ``fused_check.py`` so the CLI
stays readable; nothing here imports the route, prints, or parses arguments.

What lives here
---------------
* :func:`build_row` / :func:`pack_rows` -- the tiny nested-spec DSL
  (``("var", i)``, ``("const", v)``, ``(function_id, kids)`` and the
  ``("raw", value, node_type, kids)`` escape hatch that is the only way to build
  an ``OUT_NODE``-tagged row) and its encoding into the legacy population triple
  ``(node_value float32, node_type int16, subtree_size int16)``, each
  ``(pop, max_tree_len)`` in prefix order, on the device.
* :func:`random_population` -- the random populations the parity stage needs
  (op-set coverage), generated from Python's :mod:`random` so the battery is
  reproducible WITHOUT touching torch's global RNG and without the deleted
  ``Forest.random_generate``.
* the handmade adversarial rows (:func:`chain`, :func:`full_ternary`,
  :func:`out_node_tagged_row`, :func:`adversarial_cases`,
  :func:`multi_adversarial_cases`) and the :data:`PARITY_SETS` table.
* :func:`compare` -- the deviation report (``max_abs`` / ``max_rel`` /
  ``nan_mismatch`` / ``inf_mismatch`` / ``bitwise_equal``) every stage grades
  its comparisons with; formerly the benchmark harness's ``validate.compare``,
  which no longer exists.
* :func:`time_call` -- the warm-median protocol (one untimed first call, then
  ``warmup`` untimed calls, then ``reps`` sync-bracketed samples), formerly the
  harness's ``common.time_call``.

Node ids come from the torch-free :mod:`evogp._tables` (``Func``, ``FUNCS``,
``FUNC_ARITY``, ``FUNC_TYPE``, ``OUT_NODE``, ``MAX_OUT_LEN``), never from a
hand-copied number, so this module cannot disagree with the kernels about what
an id means.
"""

from __future__ import annotations

import random
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._tables import (
    FUNC_ARITY,
    FUNC_IDS,
    FUNC_TYPE,
    MAX_OUT_LEN,
    NType,
    OUT_NODE,
)

#: ``(using_funcs, value_tol)`` per parity set.  ``value_tol = 0.0`` means the
#: route is BITWISE against the oracle on the value paths there (re-measured with
#: this battery's own populations -- see ``tests/CONTEXT.md``); the ``1e-7`` sets
#: keep a margin because a transcendental emitter may agree in value on a
#: different byte.
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
    # ``(exp(x) -+ exp(-x)) * 0.5`` compositions (there is no native device branch),
    # i.e. exactly where divergence from the oracle's ``torch.sinh``/``torch.cosh``
    # would hide.
    "compare_unary": dict(
        using_funcs=(">", "<=", "neg", "abs", "sinh", "cosh", "+", "*"),
        input_len=8,
        max_layer_cnt=4,
        value_tol=0.0,
    ),
}

#: Tolerances of the staged comparisons (see the module docstring of ``check``).
VALUE_RTOL = 1e-6
LOSS_RTOL = 1e-5
GOLDEN_RTOL = 1e-4
GOLDEN_ATOL = 1e-4

#: ``using_funcs`` of the timing/tuning cells (the arithmetic set the campaign
#: of record timed).
TIMING_FUNCS: Tuple[str, ...] = ("+", "-", "*", "/")
#: Data rows / population of a cell when the caller does not override them.
DEFAULT_PARITY_POP = 32
DEFAULT_PARITY_ROWS = 128


# --------------------------------------------------------------------------- #
# the nested-spec DSL and its encoding
# --------------------------------------------------------------------------- #
def arity(function_id: int) -> int:
    """Number of children of a function node with that id (1, 2 or 3)."""
    return int(FUNC_ARITY[int(function_id)])


def node_type_of(function_id: int) -> int:
    """The ``NType`` marker a function id implies (arity 1/2/3 -> UFUNC/BFUNC/TFUNC)."""
    return int(FUNC_TYPE[int(function_id)])


def function_id(name: str) -> int:
    """The id of a function by its table name (``"+"``, ``"loose_div"``, ...)."""
    return int(FUNC_IDS[str(name)])


def out_node_value(function_id: int, out_index: int) -> float:
    """``float32`` bit pattern of ``struct OutNodeValue { function, outIndex }``."""
    bits = ((int(out_index) & 0xFFFF) << 16) | (int(function_id) & 0xFF)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def build_row(spec: Sequence[Any]) -> Tuple[List[float], List[int], List[int]]:
    """Encode one nested node spec into ``(values, types, sizes)`` prefix lists.

    Spec grammar (deliberately tiny -- the battery must not depend on a generator
    for the shapes it wants): ``("var", index)``, ``("const", value)``,
    ``(function_id, (child, ...))`` and ``("raw", value, node_type, (child, ...))``
    for a hand-chosen node type (that is the only way to build an
    ``OUT_NODE``-tagged row, whose semantics are the oracle's ``_kind_from_type``
    fall-through, not an operator).
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
            types[index] = int(NType.VAR)
        elif head == "const":
            values[index] = float(node[1])
            types[index] = int(NType.CONST)
        elif head == "raw":
            values[index] = float(node[1])
            types[index] = int(node[2])
            for child in node[3]:
                emit(child)
        else:
            values[index] = float(head)
            types[index] = node_type_of(int(head))
            for child in node[1]:
                emit(child)
        sizes[index] = len(values) - index

    emit(spec)
    return values, types, sizes


def node_count(spec: Sequence[Any]) -> int:
    """Number of prefix nodes a spec encodes to (``build_row``'s list length)."""
    return int(build_row(spec)[2][0])


def pack_rows(
    rows: Sequence[Sequence[Any]],
    *,
    pad_to: int = 0,
    garbage_tail: bool = False,
):
    """Pack nested specs into the population triple (``[pop, L]``) on the device.

    ``garbage_tail=True`` fills every dead cell behind ``subtree_size[0]`` with a
    NaN value, a nonsense type and a nonsense size -- the kernel must never read it
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


# --------------------------------------------------------------------------- #
# random populations (the parity stage's op-set coverage)
# --------------------------------------------------------------------------- #
def _leaf(rng: random.Random, input_len: int, *, const_prob: float,
          const_range: Tuple[float, float]) -> Sequence[Any]:
    """One random leaf: a CONST in ``const_range`` or a VAR over ``input_len``."""
    if rng.random() < const_prob:
        return ("const", round(rng.uniform(*const_range), 6))
    return ("var", rng.randrange(int(input_len)))


def _split_budget(rng: random.Random, total: int, parts: int) -> List[int]:
    """Hand ``total`` nodes to ``parts`` children, every child at least one."""
    if parts <= 1:
        return [int(total)]
    shares = [1] * parts
    left = int(total) - parts
    while left > 0:
        take = rng.randint(1, left)
        shares[rng.randrange(parts)] += take
        left -= take
    return shares


def random_row(
    rng: random.Random,
    using_funcs: Sequence[str],
    *,
    input_len: int,
    max_nodes: int,
    max_layer_cnt: int,
    layer_leaf_prob: float = 0.2,
    const_prob: float = 0.5,
    const_range: Tuple[float, float] = (0.25, 1.25),
) -> Sequence[Any]:
    """One random tree over ``using_funcs``, at most ``max_nodes`` nodes deep/wide.

    A recursive grow: a node is a leaf when the depth budget or the node budget is
    spent (or on the ``layer_leaf_prob`` coin), otherwise a uniformly drawn
    function from ``using_funcs`` whose arity fits the remaining budget, with the
    rest of the budget split over its children.  Nothing about this distribution is
    meant to match the GP generator -- it only has to reach every operator of a set
    with a realistic depth/width mix.
    """
    funcs = [function_id(name) for name in using_funcs]

    def grow(budget: int, depth_left: int) -> Sequence[Any]:
        if depth_left <= 0 or budget <= 1 or rng.random() < layer_leaf_prob:
            return _leaf(rng, input_len, const_prob=const_prob, const_range=const_range)
        candidates = [fid for fid in funcs if arity(fid) + 1 <= budget]
        if not candidates:
            return _leaf(rng, input_len, const_prob=const_prob, const_range=const_range)
        chosen = rng.choice(candidates)
        shares = _split_budget(rng, budget - 1, arity(chosen))
        return (chosen, tuple(grow(shares[i], depth_left - 1) for i in range(len(shares))))

    return grow(int(max_nodes), int(max_layer_cnt))


def random_population(
    *,
    pop: int,
    tree_len: int,
    input_len: int,
    using_funcs: Sequence[str],
    max_layer_cnt: int,
    seed: int = 0,
    pad_to: Optional[int] = None,
):
    """``seed`` -> a population of ``pop`` random trees over ``using_funcs``.

    Generated from Python's :mod:`random` (never torch's global RNG), so the same
    seed reproduces the same population in any process and after any other torch
    work.
    """
    rng = random.Random(int(seed))
    rows = [
        random_row(
            rng,
            using_funcs,
            input_len=int(input_len),
            max_nodes=int(tree_len),
            max_layer_cnt=int(max_layer_cnt),
        )
        for _ in range(int(pop))
    ]
    for index, row in enumerate(rows):
        length = node_count(row)
        if length > int(tree_len):
            raise AssertionError(
                f"row {index} has {length} nodes > tree_len={int(tree_len)}"
            )
    return pack_rows(rows, pad_to=int(tree_len) if pad_to is None else int(pad_to))


def workload(
    cell: Dict[str, Any],
    *,
    using_funcs: Optional[Sequence[str]] = None,
    max_layer_cnt: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """The population AND data of one timing/parity cell (deterministic).

    ``torch.rand`` draws the data (positive in ``[0, 1)``, so a LOG or a SQRT of a
    VAR stays finite), the population comes from :func:`random_population`.
    """
    import torch

    funcs = list(using_funcs or TIMING_FUNCS)
    depth = int(cell.get("max_layer_cnt", 5)) if max_layer_cnt is None else int(max_layer_cnt)
    node_value, node_type, subtree_size = random_population(
        pop=int(cell["pop"]),
        tree_len=int(cell["tree_len"]),
        input_len=int(cell["input_len"]),
        using_funcs=funcs,
        max_layer_cnt=depth,
        seed=seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7)
    rows = int(cell["rows"])
    input_len = int(cell["input_len"])
    return {
        "rows": (node_value, node_type, subtree_size),
        "x": torch.rand(rows, input_len, generator=generator).cuda(),
        "labels": torch.rand(rows, 1, generator=generator).cuda(),
        "x_per_row": torch.rand(int(cell["pop"]), input_len, generator=generator).cuda(),
        "cell": dict(cell),
        "using_funcs": funcs,
    }


# --------------------------------------------------------------------------- #
# handmade trees (the adversarial rows of the parity stage)
# --------------------------------------------------------------------------- #
def chain(depth: int, *, right: bool, input_len: int) -> Sequence[Any]:
    """A ``depth``-deep ``+`` chain over ``VAR`` leaves (right-leaning if asked).

    A left chain keeps one pending operand and a right chain keeps ``depth`` of
    them, so the pair pins the kernel's stack discipline from both sides.
    """
    add = function_id("+")
    node: Any = ("var", 0)
    for level in range(1, depth + 1):
        leaf = ("var", level % input_len)
        node = (add, (node, leaf)) if not right else (add, (leaf, node))
    return node


def full_ternary(depth: int, input_len: int) -> Sequence[Any]:
    """A complete ternary ``if`` tree of ``depth`` levels (``(3**d - 1) / 2`` nodes)."""
    if_node = function_id("if")
    if depth <= 1:
        return ("var", 0)
    return (
        if_node,
        (
            full_ternary(depth - 1, input_len),
            ("var", (depth + 1) % input_len),
            full_ternary(depth - 1, input_len),
        ),
    )


def out_node_tagged_row() -> Any:
    """One single-output row whose nodes carry the multi-output ``OUT_NODE`` tag.

    The population a MIS-WIRED ``output_len`` produces: built for
    ``output_len >= 2``, evaluated as single-output.  The oracle's
    ``_kind_from_type`` folds every tagged type into its TERNARY branch (so the
    kernel's own fall-through is defined and would return a number), which is
    exactly why the facade must refuse it before any launch -- the shared
    ``evogp._multi_output.check_multi_output`` rule.  ``--stage declines`` is where
    the refusal is pinned; every tagged type (``0x80|0 .. 0x80|4``) is covered, and
    each one carries three children because the oracle would evaluate it as a
    ternary.
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
    add, mul, div, sqrt, log_op = (
        function_id("+"),
        function_id("*"),
        function_id("/"),
        function_id("sqrt"),
        function_id("log"),
    )
    const_tree = (add, (("const", 1.5), (mul, (("const", 2.0), ("const", 3.0)))))
    mixed = (
        add,
        (
            (div, (("var", 0), ("const", 0.0))),
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
            note="single CONST root (a pure constant tree)",
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
            rows=[chain(40, right=False, input_len=8)],
            input_len=8,
            bitwise=True,
            note="40-deep left chain, one pending operand",
        ),
        dict(
            name="right_chain_40",
            rows=[chain(40, right=True, input_len=8)],
            input_len=8,
            bitwise=True,
            note="40-deep right chain, 40 pending operands",
        ),
        dict(
            name="full_if_3",
            rows=[full_ternary(3, 4)],
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
                chain(12, right=False, input_len=4),
                chain(12, right=True, input_len=4),
                full_ternary(2, 4),
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
    it the raw form ``("raw", out_node_value(function_id, out_index), 0x80 | base,
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
    add, sub, mul, neg, div = (
        function_id("+"),
        function_id("-"),
        function_id("*"),
        function_id("neg"),
        function_id("/"),
    )
    if_op = function_id("if")

    def tagged(function_id: int, out_index: int, base: int, children: Sequence[Any]):
        """One ``OUT_NODE``-tagged node: ``out_index`` in the high, op id in the low half."""
        return ("raw", out_node_value(function_id, out_index), 0x80 | base, tuple(children))

    # Two tagged outputs, one of them 7 nodes deep and one on the root's right arm.
    two_outputs = (
        add,
        (
            tagged(neg, 0, 2, (chain(6, right=False, input_len=4),)),
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
    widest = int(MAX_OUT_LEN)
    all_outputs: Any = ("var", 0)
    for index in range(widest - 1, -1, -1):
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
            out_len=widest,
            ops="ADD",
            note=f"one tagged output per index, the accumulator at its full width ({widest})",
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
# comparison and timing helpers
# --------------------------------------------------------------------------- #
def compare(reference, candidate) -> Dict[str, Any]:
    """Deviation of ``candidate`` from ``reference`` (both float tensors).

    NaN and inf PLACEMENT is always exact; a NaN *payload* is deliberately not
    part of the claim (the loose surface may write a different payload for the same
    NaN), so ``bitwise_equal`` compares finite values after folding NaNs.
    """
    import torch

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


def time_call(fn, warmup: int = 2, reps: int = 10) -> Dict[str, Any]:
    """Warm/steady-state timing of ``fn`` (sync-bracketed, per-call samples).

    The first call is timed separately and never counted in the warm statistics:
    that is where a cold compile lands (Triton's JIT on an empty cache, or a
    specialization LOAD on a warm disk cache).
    """
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1e3
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(int(reps)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(samples)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "first_call_ms": first_ms,
        "warmup": int(warmup),
        "reps": int(reps),
        "samples_ms": samples,
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": sum(samples) / len(samples),
        "_out": out,
    }


__all__ = [
    "DEFAULT_PARITY_POP",
    "DEFAULT_PARITY_ROWS",
    "GOLDEN_ATOL",
    "GOLDEN_RTOL",
    "LOSS_RTOL",
    "MAX_OUT_LEN",
    "OUT_NODE",
    "PARITY_SETS",
    "TIMING_FUNCS",
    "VALUE_RTOL",
    "adversarial_cases",
    "arity",
    "build_row",
    "chain",
    "compare",
    "full_ternary",
    "function_id",
    "multi_adversarial_cases",
    "node_count",
    "node_type_of",
    "out_node_tagged_row",
    "out_node_value",
    "pack_rows",
    "random_population",
    "random_row",
    "time_call",
    "workload",
]
