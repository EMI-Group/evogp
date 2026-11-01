"""Regression net for the Triton ``fused`` evaluation route (``evogp.fused``).

This is the SUITE's net for the route: seconds of assertions that fail loudly when
the Triton evaluator (:mod:`evogp.fused.kernels`), its facade
(:mod:`evogp.fused.engine`) or its front tables (:mod:`evogp.fused.memo`) regress.
``python -m evogp.fused.check`` is the route's EXHAUSTIVE battery (parity, multi,
goldens, declines, timing, tuning) and stays the authority; it is a dev CLI and is
NOT pytest-collected.

The route is called the way its only in-tree caller calls it: DIRECTLY, on the
legacy population triple ``(batch_node_value float32, batch_node_type int16,
batch_subtree_size int16)``, each ``(pop_size, max_tree_len)`` in prefix order on a
device -- ``evaluate_fused`` for the value paths, ``sr_fitness_fused`` for the SR
loss.  There is no ``Forest``/``Tree`` container and no route policy any more (the
``tree/`` stack is gone), so nothing here drives an ``EVOGP_EVAL_MODE`` or a
``EVOGP_FUSED_MIN_POP``; a case that used to prove a call was ROUTED here now
proves it was SERVED here, from the route's own counters.

Oracle: ``test/oracle/reference_eval.py`` (dev-only, imported by PATH through the
``oracle`` fixture of ``conftest.py``).  The three dispatch/cache cases need no
oracle and keep running where ``test/oracle/`` is absent.

HERMETIC BY CONSTRUCTION: the suite SKIPS (never fails) when the route cannot exist
-- no CUDA/HIP device visible to torch, ``triton`` not importable (the session
``fused_route_guard`` fixture of ``conftest.py``), or no oracle next to the tests.
A writable ``TRITON_CACHE_DIR`` is provided by that fixture when the ambient one is
unusable, because the first launch JIT-compiles the kernels.

RUN (repo root, torch-carrying venv, from a checkout whose ``test/oracle/`` is
present)::

    PYTHONPATH=src:/root/evoxc:/root/evox/src EVOXC_DISABLE_LOCATIONS=1 \\
      TRITON_CACHE_DIR=$TMPDIR/triton-fused-cache \\
      /tmp/rocvenv/bin/python -m pytest src/evogp/fused/tests -q

The per-case verdict table (kept / rewritten / dropped-with-reason) is in
``src/evogp/fused/tests/CONTEXT.md``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evogp._multi_output import MultiOutputError  # noqa: E402
from evogp._tables import NType, OUT_NODE  # noqa: E402
from evogp.fused import engine as fused_engine  # noqa: E402
from evogp.fused import kernels as fused_kernels  # noqa: E402
from evogp.fused.cases import MAX_OUT_LEN, random_population  # noqa: E402
from support import (  # noqa: E402  (the tests directory is on sys.path)
    DEVICE,
    assert_same,
    chain,
    data,
    declined,
    deltas,
    function_id,
    oracle_loss,
    oracle_value,
    population,
    served,
    tagged,
)

#: Node kinds (``evogp._tables.NType``) and the function ids the cases name.
VAR = int(NType.VAR)
CONST = int(NType.CONST)
(
    IF,
    ADD,
    SUB,
    MUL,
    DIV,
    LOOSE_DIV,
    POW,
    LOOSE_POW,
    MAX,
    MIN,
    LT,
    GT,
    LE,
    GE,
    SIN,
    COS,
    TAN,
    SINH,
    COSH,
    TANH,
    LOG,
    LOOSE_LOG,
    EXP,
    INV,
    LOOSE_INV,
    NEG,
    ABS,
    SQRT,
    LOOSE_SQRT,
) = (
    function_id(name)
    for name in (
        "if",
        "+",
        "-",
        "*",
        "/",
        "loose_div",
        "pow",
        "loose_pow",
        "max",
        "min",
        "<",
        ">",
        "<=",
        ">=",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "log",
        "loose_log",
        "exp",
        "inv",
        "loose_inv",
        "neg",
        "abs",
        "sqrt",
        "loose_sqrt",
    )
)

#: ``(using_funcs, rtol)`` per function set.  ``rtol = 0.0`` means the route is
#: BITWISE against the oracle there (re-measured for this module's own populations;
#: the numbers are in ``tests/CONTEXT.md``); the ``1e-6`` sets are the
#: transcendental / loose surfaces, where an emitter may agree in value on a
#: different byte.
VALUE_SETS = {
    "arithmetic": (["+", "-", "*", "/"], 0.0),
    "ternary": (["if", "+", "*"], 0.0),
    "compare_unary": (["<", ">", "<=", ">=", "neg", "abs", "sqrt"], 0.0),
    "transcendental": (["sin", "cos", "tan", "exp", "log", "+", "*"], 1e-6),
    "loose": (["loose_div", "loose_pow", "loose_log", "loose_inv", "loose_sqrt"], 1e-6),
}


# --- populations, data ------------------------------------------------------------ #
def _random_rows(using_funcs, *, pop=24, input_len=4, output_len=1, tree_len=64, seed=0):
    """``seed -> population`` (Python's RNG: reproducible without touching torch)."""
    max_layer_cnt = 3 if "if" in using_funcs else 4
    return random_population(
        pop=pop,
        tree_len=tree_len,
        input_len=input_len,
        using_funcs=list(using_funcs),
        max_layer_cnt=max_layer_cnt,
        seed=seed,
    )


def _const_rows(pop_size=16, value=1.5, pad_to=32):
    """A population of one-node CONSTANT trees: ``batch_node_value[row, 0]`` is it."""
    return population([("const", value)] * pop_size, pad_to=pad_to)


def _serve(rows, input_len, x, *, output_len=1, per_row=False):
    """``evaluate_fused`` on a triple (the route's value path, verbatim)."""
    return fused_engine.evaluate_fused(
        *rows, input_len, output_len, x, per_row=per_row
    )


def _serve_loss(rows, input_len, inputs, labels, *, output_len=1, use_mse=True):
    """``sr_fitness_fused`` on a triple (the route's loss path, verbatim)."""
    return fused_engine.sr_fitness_fused(
        *rows, input_len, output_len, inputs, labels, use_MSE=use_mse
    )


# --- fixtures --------------------------------------------------------------------- #
@pytest.fixture(scope="module", autouse=True)
def _warm_jit():
    """Compile the four single-output Triton specializations once for the module.

    The JIT cost is per ``(per_row, reduce, use_mse, block, warps)`` combination
    (~0.4 s the first time in a process, ~2 ms from a warm disk cache).  The
    multi-output kernels are compiled by their own case.
    """
    rows = population([("fn", ADD, (("var", 0), ("var", 1)))] * 16, pad_to=8)
    broadcast, labels, per_row = data(8, 4, seed=1), data(8, 1, seed=3), data(16, 4, 2)
    _serve(rows, 4, broadcast)
    _serve(rows, 4, per_row, per_row=True)
    for use_mse in (True, False):
        _serve_loss(rows, 4, broadcast, labels, use_mse=use_mse)
    yield


# --- 1. value-path parity --------------------------------------------------------- #
@pytest.mark.parametrize("per_row", [False, True], ids=["batch_forward", "forward"])
@pytest.mark.parametrize("name", sorted(VALUE_SETS))
def test_value_parity(name, per_row, oracle):
    """The batched / per-tree value paths against the oracle, per function set."""
    using_funcs, rtol = VALUE_SETS[name]
    rows = _random_rows(using_funcs, seed=11 if name == "arithmetic" else 0)
    pop_size = int(rows[0].shape[0])
    x = data(pop_size if per_row else 16, 4, seed=13 if per_row else 11)
    context = f"{'forward' if per_row else 'batch_forward'}[{name}]"
    got = served(lambda: _serve(rows, 4, x, per_row=per_row), context)
    want = oracle_value(oracle, rows, x, per_row=per_row)
    assert_same(got, want, context, rtol=None if rtol == 0.0 else rtol)


def test_single_node_and_constant_populations(oracle):
    """A VAR root, a CONST root and a fully constant population."""
    x = data(8, 4)
    cases = {
        "var_root": [("var", 1)],
        "const_root": [("const", 2.5)],
        "constant_population": [
            ("fn", ADD, (("const", 1.5), ("fn", MUL, (("const", 2.0), ("const", 3.0)))))
        ],
    }
    for name, specs in cases.items():
        rows = population(specs * 16, pad_to=8)
        context = f"batch_forward[{name}]"
        got = served(lambda: _serve(rows, 4, x), context)
        assert_same(got, oracle_value(oracle, rows, x), context)
        per_row = data(16, 4, seed=21)
        assert_same(
            _serve(rows, 4, per_row, per_row=True),
            oracle_value(oracle, rows, per_row, per_row=True),
            f"forward[{name}]",
        )


def test_deep_chain(oracle):
    """An 81-node deep chain (well past 40): the stack discipline under load."""
    spec = chain(40, right=False, input_len=4)
    rows = population([spec] * 16, pad_to=128)
    length = int(rows[2][0, 0])
    assert length >= 40, f"the deep-chain case regressed to {length} nodes"
    x = data(8, 4)
    context = f"batch_forward[deep_chain {length} nodes]"
    got = served(lambda: _serve(rows, 4, x), context)
    assert_same(got, oracle_value(oracle, rows, x), context)
    per_row = data(16, 4, seed=22)
    assert_same(
        _serve(rows, 4, per_row, per_row=True),
        oracle_value(oracle, rows, per_row, per_row=True),
        "forward[deep_chain]",
    )


def test_protected_div_and_nan_inf_placement(oracle):
    """DIV by zero, LOG(0), SQRT(-1), an overflow -- and the NaN/Inf PLACEMENT rules."""
    edge = [
        ("fn", DIV, (("var", 0), ("const", 0.0))),  # +-inf, NaN when 0/0
        ("fn", DIV, (("const", 0.0), ("const", 0.0))),  # NaN (payload may differ)
        ("fn", DIV, (("const", 1.0), ("const", -0.0))),  # -inf
        ("fn", INV, (("const", 0.0),)),
        ("fn", LOG, (("const", 0.0),)),  # -inf
        ("fn", LOG, (("const", -1.0),)),  # NaN
        ("fn", SQRT, (("const", -1.0),)),  # NaN
        ("fn", LOOSE_DIV, (("const", 1.0), ("const", 0.0))),  # 1e9, never unbounded
        ("fn", EXP, (("const", 100.0),)),  # +inf
        ("fn", NEG, (("fn", DIV, (("var", 1), ("const", 0.0))),)),
        ("fn", ADD, (("fn", DIV, (("const", 0.0), ("const", 0.0))), ("const", 1.0))),
        ("var", 0),
        ("const", 2.5),
        ("fn", IF, (("fn", DIV, (("var", 1), ("const", 0.0))), ("var", 1), ("var", 2))),
    ]
    rows = population([spec for spec in edge for _ in range(2)], pad_to=32)
    x = data(8, 4)
    got = served(lambda: _serve(rows, 4, x), "batch_forward[edge]")
    want = oracle_value(oracle, rows, x)
    assert int(torch.isnan(want).sum()) > 0, (
        "the edge population produces no NaN any more, so it stopped exercising NaN placement"
    )
    assert_same(got, want, "batch_forward[edge: div/0, log(0), sqrt(-1), overflow]")
    per_row = data(int(rows[0].shape[0]), 4, seed=23)
    assert_same(
        _serve(rows, 4, per_row, per_row=True),
        oracle_value(oracle, rows, per_row, per_row=True),
        "forward[edge]",
    )


# --- 2. the loss path and its result contract ------------------------------------- #
@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
@pytest.mark.parametrize("name", ["arithmetic", "loose"])
def test_sr_fitness_matches_oracle(name, use_mse, oracle):
    """The SR loss against the oracle's reduction, for both loss metrics."""
    rows = _random_rows(VALUE_SETS[name][0], seed=5 if name == "loose" else 3)
    inputs, labels = data(16, 4), data(16, 1, seed=31)
    context = f"SR_fitness[{name}, mse={use_mse}]"
    loss = served(lambda: _serve_loss(rows, 4, inputs, labels, use_mse=use_mse), context)
    assert_same(
        loss,
        oracle_loss(oracle, rows, inputs, labels, use_mse=use_mse),
        context,
        rtol=1e-6,
    )


def test_sr_fitness_result_contract():
    """``(pop,)`` float32, contiguous, no grad, on the device, and a FRESH buffer."""
    rows = _random_rows(["+", "*"], seed=7)
    pop_size = int(rows[0].shape[0])
    inputs, labels = data(16, 4), data(16, 1, seed=41)
    loss = served(lambda: _serve_loss(rows, 4, inputs, labels), "SR_fitness[contract]")
    assert tuple(loss.shape) == (pop_size,), f"shape {tuple(loss.shape)}"
    assert loss.dtype == torch.float32, f"dtype {loss.dtype}"
    assert loss.is_contiguous(), "the caller mutates the loss in place: it must be contiguous"
    assert loss.requires_grad is False, "the loss must not carry an autograd graph"
    assert loss.device.type == "cuda", f"the loss lives on {loss.device}"
    snapshot = loss.clone()
    loss.fill_(-123.0)  # the caller writes into the result; the next call must be clean
    assert_same(_serve_loss(rows, 4, inputs, labels), snapshot, "SR_fitness[fresh buffer]")


# --- 3. multi-output (OUT_NODE-tagged) populations --------------------------------- #
def _multi_population():
    """Four tagged shapes x 4: two outputs, a tagged ROOT, one tagged, same output twice."""
    two_outputs = (
        "fn",
        ADD,
        (
            tagged(NEG, 0, (("var", 0),)),
            tagged(ADD, 1, (("var", 1), ("var", 2))),
        ),
    )
    root_if = tagged(
        IF,
        1,
        (
            ("var", 0),
            ("fn", ADD, (("var", 1), ("const", 0.5))),
            ("fn", MUL, (("var", 2), ("var", 3))),
        ),
    )
    one_tagged = (
        "fn",
        ADD,
        (
            tagged(MUL, 1, (("var", 0), ("const", 2.5))),
            ("fn", SUB, (("var", 1), ("var", 2))),
        ),
    )
    same_output = tagged(ADD, 0, (("var", 0), tagged(NEG, 0, (("var", 1),))))
    return [two_outputs, root_if, one_tagged, same_output] * 4


@pytest.mark.parametrize("out_len", [2, 3])
def test_multi_output_value_parity(out_len, oracle):
    """Both value paths at ``output_len`` 2 and 3, BITWISE against the oracle.

    The multi-output path accumulates every tagged node's contribution into the
    ``(pop, rows, out_len)`` float32 buffer, while the oracle writes each out-node's
    value once -- yet the two agree to the BIT on this box at the default block
    geometry, re-measured for this module and for all twelve multi-output cases of
    ``evogp.fused.cases`` (``output_len`` 2 / 3 / 8, both value paths; see
    ``tests/CONTEXT.md``).  What is NOT bitwise is the multi-output LOSS (its in-kernel
    fp32 reduction order), which the case below pins at ``1e-6`` relative.
    """
    rows = population(_multi_population(), pad_to=32, output_len=out_len)
    x, context = data(16, 4), f"multi_batch_forward[out_len={out_len}]"
    got = served(lambda: _serve(rows, 4, x, output_len=out_len), context)
    assert_same(got, oracle_value(oracle, rows, x, output_len=out_len), context)
    per_row = data(int(rows[0].shape[0]), 4, seed=33)
    assert_same(
        _serve(rows, 4, per_row, output_len=out_len, per_row=True),
        oracle_value(oracle, rows, per_row, per_row=True, output_len=out_len),
        f"forward[multi, out_len={out_len}]",
    )


@pytest.mark.parametrize("use_mse", [True, False], ids=["mse", "mae"])
@pytest.mark.parametrize("out_len", [2, 3])
def test_multi_output_loss_parity(out_len, use_mse, oracle):
    """The SR loss on tagged trees: the documented ``<= 1e-6`` relative, NOT bitwise.

    The in-kernel reduction sums the per-output errors in fp32 in its own order
    while the oracle sums with torch, so the two differ by accumulation order only.
    """
    rows = population(_multi_population(), pad_to=32, output_len=out_len)
    inputs, labels = data(16, 4), data(16, out_len, seed=35)
    context = f"SR_fitness[multi, out_len={out_len}, mse={use_mse}]"
    loss = served(
        lambda: _serve_loss(rows, 4, inputs, labels, output_len=out_len, use_mse=use_mse),
        context,
    )
    assert tuple(loss.shape) == (int(rows[0].shape[0]),), f"shape {tuple(loss.shape)}"
    assert_same(
        loss,
        oracle_loss(oracle, rows, inputs, labels, use_mse=use_mse, output_len=out_len),
        context,
        rtol=1e-6,
    )


# --- 3b. the OUT_NODE refusal (a single-output request over tagged rows) ----------- #
def _miswired_rows():
    """REAL ``output_len = 2`` rows presented as a single-output population.

    This is what a wrong ``output_len`` at generation time produces: the tags on
    the rows belong to the out-nodes.  Before the shared rule
    (``evogp._multi_output``) the fused kernel read such a type as a ternary ``if``
    and returned a plausible wrong number; the facade now refuses it host-side.
    """
    genuine = population(_multi_population(), pad_to=32, output_len=2)
    return genuine


def test_refuse_out_node_tags_in_single_output_mode():
    """The facade refuses, and it is NOT a decline: nothing is launched."""
    rows = _miswired_rows()
    assert bool((rows[1] & OUT_NODE).any()), "the fixture lost its tags"
    broadcast, labels = data(16, 4), data(16, 1, seed=41)
    per_row = data(int(rows[0].shape[0]), 4, seed=43)
    before = fused_engine.fused_cache_stats()
    for context, thunk in (
        ("evaluate_fused", lambda: _serve(rows, 4, broadcast)),
        ("evaluate_fused[per_row]", lambda: _serve(rows, 4, per_row, per_row=True)),
        ("sr_fitness_fused", lambda: _serve_loss(rows, 4, broadcast, labels)),
    ):
        with pytest.raises(ValueError, match="OUT_NODE") as caught:
            thunk()
        assert str(caught.value).startswith("row 0: node "), (
            f"{context}: the refusal must name row and node, got {str(caught.value)!r}"
        )
    declines, launches = deltas(before, fused_engine.fused_cache_stats())
    assert (declines, launches) == (0, 0), (
        "the OUT_NODE refusal must happen before any launch and must not be counted as a "
        f"DECLINE (a decline is a route answer, not a caller error): declines={declines} "
        f"launches={launches}"
    )


def test_refuse_out_node_tags_on_every_entry_point_and_name_the_row():
    """The refusal is the shared ``MultiOutputError`` and reports the FIRST tagged row.

    Row 0 is a clean single-output tree, so a refusal that reported ``row 0`` would
    be reading the wrong population; the tagged node lives at the root of row 1.
    """
    rows = population(
        [("fn", ADD, (("var", 0), ("var", 1))), tagged(ADD, 0, (("var", 0), ("var", 1)))],
        pad_to=16,
        output_len=2,
    )
    assert not bool((rows[1][0] & OUT_NODE).any()), "row 0 must stay untagged"
    broadcast, labels = data(16, 4), data(16, 1, seed=47)
    per_row = data(int(rows[0].shape[0]), 4, seed=49)
    for context, thunk in (
        ("evaluate_fused", lambda: _serve(rows, 4, broadcast)),
        ("evaluate_fused[per_row]", lambda: _serve(rows, 4, per_row, per_row=True)),
        ("sr_fitness_fused", lambda: _serve_loss(rows, 4, broadcast, labels)),
    ):
        with pytest.raises(MultiOutputError, match="OUT_NODE") as caught:
            thunk()
        assert str(caught.value).startswith("row 1: node 0: "), (
            f"{context}: the refusal must name the FIRST tagged row and node, got "
            f"{str(caught.value)!r}"
        )


def test_the_same_rows_are_served_once_output_len_says_two(oracle):
    """The refusal is scoped to ``output_len == 1``: the tagged rows still evaluate."""
    rows = _miswired_rows()
    x = data(16, 4)
    got = served(lambda: _serve(rows, 4, x, output_len=2), "batch_forward[multi, out_len=2]")
    assert_same(
        got,
        oracle_value(oracle, rows, x, output_len=2),
        "batch_forward[multi, out_len=2]",
    )


# --- 4. the served-call proof ------------------------------------------------------ #
def test_served_call_increments_launches_only():
    """A served call bumps ``fused_kernel_launches`` and ``fused_calls`` only."""
    rows = _random_rows(["+", "*"], pop=32, seed=17)
    x = data(8, 4)
    before = fused_engine.fused_cache_stats()
    _serve(rows, 4, x)
    after = fused_engine.fused_cache_stats()
    calls = int(after["fused_calls"] - before["fused_calls"])
    assert calls == 1, f"one facade call must be exactly one fused call, counted {calls}"
    assert deltas(before, after) == (0, 1), (
        f"one served call must launch once and never decline, counted {deltas(before, after)}"
    )


# --- 5. declines: the documented "decline, never a wrong number" contract ---------- #
def test_decline_wide_output_len(oracle):
    """``output_len`` above ``MAX_OUT_LEN`` declines; the same rows serve at the cap.

    The decline is the route's answer, not a crash and not a number: the caller owns
    the fallback (the gp stack's in-graph / host evaluator), so what the facade owes
    is a named :class:`FusedUnsupported` and no launch -- and the narrower width of
    the very same population must still be served correctly.
    """
    out_len = int(MAX_OUT_LEN) + 1
    rows = population(
        [tagged(ADD, 0, (("var", 0), ("var", 1)))] * 16,
        pad_to=32,
        output_len=out_len,
    )
    x = data(8, 4)
    with pytest.raises(fused_engine.FusedUnsupported) as caught:
        declined(lambda: _serve(rows, 4, x, output_len=out_len), f"out_len={out_len}")
    assert "MAX_OUT_LEN" in str(caught.value), (
        f"the decline must name the constant, got {str(caught.value)!r}"
    )
    got = served(lambda: _serve(rows, 4, x, output_len=int(MAX_OUT_LEN)),
                 f"out_len={int(MAX_OUT_LEN)}")
    assert_same(
        got,
        oracle_value(oracle, rows, x, output_len=int(MAX_OUT_LEN)),
        f"decline[out_len={out_len}] -> served at {int(MAX_OUT_LEN)}",
    )


def test_decline_tiny_scratch_budget(oracle):
    """A scratch budget below the launch's real bytes declines; the envelope agrees.

    The envelope is the bytes one launch ALLOCATES: ``pop * n_rb * tree_len * block
    * 4`` with ``block`` = the largest power of two ``<= min(rows, cap)``, never
    ``rows`` itself.  32 trees x 64 cells at 160 data rows is 2 blocks of 128 lanes
    -> 32 * 2 * 64 * 128 * 4 B = 4 MiB of scratch, comfortably above the 1 MiB
    budget below (16 rows would be a 16-lane block, i.e. 128 KiB, and would be
    SERVED).
    """
    rows = _random_rows(["+", "*"], pop=32, seed=19)
    x = data(160, 4)
    baseline = served(lambda: _serve(rows, 4, x), "batch_forward[budget=default]")
    with pytest.raises(fused_engine.FusedUnsupported):
        declined(
            lambda: fused_engine.evaluate_fused(
                *rows, 4, 1, x, per_row=False, max_scratch_mb=1
            ),
            "batch_forward[budget=1MB]",
        )
    verdict = fused_engine.envelope_decline(32, 64, 160, max_scratch_mb=1)
    assert verdict is not None, (
        "envelope_decline() must predict the decline the call raises (it is the "
        "non-raising half of the same verdict)"
    )
    assert_same(
        served(lambda: _serve(rows, 4, x), "batch_forward[budget=default, again]"),
        baseline,
        "decline[scratch budget] leaves the served numbers alone",
    )


# --- 6. guards for the defects the hardening rounds fixed ------------------------- #
def test_noncontiguous_inputs_are_read_correctly():
    """A column slice of a padded ``x`` (and of the labels) equals its contiguous copy."""
    rows = _random_rows(["+", "*"], seed=23)
    pop_size = int(rows[0].shape[0])
    sliced = data(16, 12)[:, :4]
    assert not sliced.is_contiguous(), "the case stopped being a non-contiguous view"
    assert_same(
        _serve(rows, 4, sliced),
        _serve(rows, 4, sliced.contiguous()),
        "batch_forward[non-contiguous x]",
    )
    row_slice = data(pop_size, 12, seed=17)[:, :4]
    assert not row_slice.is_contiguous()
    assert_same(
        _serve(rows, 4, row_slice, per_row=True),
        _serve(rows, 4, row_slice.contiguous(), per_row=True),
        "forward[non-contiguous x]",
    )
    labels = data(16, 3, seed=19)[:, :1]
    assert not labels.is_contiguous()
    assert_same(
        _serve_loss(rows, 4, sliced, labels),
        _serve_loss(rows, 4, sliced.contiguous(), labels.contiguous()),
        "SR_fitness[non-contiguous inputs/labels]",
    )


def test_noncontiguous_population_equals_contiguous():
    """A stride-2 population table equals its contiguous copy, bitwise."""
    rows = _random_rows(["+", "*"], seed=29)
    x = data(8, 4)

    def strided(table):
        wide = torch.zeros(
            (table.shape[0], 2 * table.shape[1]), dtype=table.dtype, device=table.device
        )
        wide[:, ::2] = table
        return wide[:, ::2]

    view = tuple(strided(table) for table in rows)
    assert not view[2].is_contiguous(), "the case stopped being a non-contiguous population"
    got = served(lambda: _serve(view, 4, x), "batch_forward[stride-2 population]")
    assert_same(got, _serve(rows, 4, x), "batch_forward[stride-2 population]")


def test_subtree_size_clamp():
    """An oversized / negative ``subtree_size[0]`` must not crash or spill."""
    spec = ("fn", ADD, (("fn", MUL, (("var", 0), ("const", 1.5))), ("var", 1)))
    honest = population([spec] * 16, pad_to=16)
    x = data(8, 4)
    honest_values = _serve(honest, 4, x)

    oversized = honest[2].clone()
    oversized[0, 0] = 30000  # clamps to tree_len: the honest row's own answer
    assert_same(
        _serve((honest[0], honest[1], oversized), 4, x),
        honest_values,
        "clamp[subtree_size[0]=30000]",
    )

    negative = honest[2].clone()
    negative[0, 0] = -5  # floors to 0: the walk does nothing, the value is exactly 0
    floored_values = _serve((honest[0], honest[1], negative), 4, x)
    assert float(floored_values[0].abs().max()) == 0.0, (
        "a negative subtree_size[0] must floor to an empty walk (0.0), got "
        f"max|value|={float(floored_values[0].abs().max()):.3e}"
    )
    assert_same(
        floored_values[1:].unsqueeze(0),
        honest_values[1:].unsqueeze(0),
        "clamp[subtree_size[0]=-5] leaves the other rows alone",
    )


def test_clear_fused_cache_resets_every_table_and_counter(oracle):
    """``clear_fused_cache()`` is the ONE in-memory reset entry of the route."""
    rows = _random_rows(["+", "*"], seed=31)
    x = data(8, 4)
    _serve(rows, 4, x)  # make sure every counter has moved
    fused_engine.clear_fused_cache()
    for key in (
        "fused_calls",
        "fused_declines",
        "fused_kernel_launches",
        "fused_xt_entries",
        "fused_plan_hits",
        "fused_plan_misses",
    ):
        assert fused_engine.fused_cache_stats()[key] == 0, f"{key} survived clear_fused_cache()"
    got = served(lambda: _serve(rows, 4, x), "batch_forward[after clear_fused_cache]")
    assert_same(got, oracle_value(oracle, rows, x), "batch_forward[after clear_fused_cache]")


# --- 7. the launch path (``EVOGP_FUSED_DISPATCH``: pre-bound vs ``JITFunction.run``) --- #
# The route's per-call cost at the small cells is dominated by Triton's PYTHON
# dispatcher, not by the kernel, so the shipped path binds the ``CompiledKernel``
# once per prepared call and serves every later call directly.  These cases pin the
# contract that makes that safe: the knob's spellings, the counters, bitwise
# equality with the historical dispatch, the re-arm point and the two declines.
def test_resolve_dispatch_spellings(monkeypatch):
    """``bound`` is the default, ``jit`` the A/B arm, and a typo is a caller error."""
    monkeypatch.delenv(fused_kernels.DISPATCH_ENV, raising=False)
    assert fused_kernels.resolve_dispatch() == "bound", "the shipped default moved"
    assert fused_kernels.resolve_dispatch("") == "bound", "an empty value means the default"
    assert fused_kernels.resolve_dispatch(" BOUND ") == "bound", "spellings are normalised"
    assert fused_kernels.resolve_dispatch("jit") == "jit"
    monkeypatch.setenv(fused_kernels.DISPATCH_ENV, "jit")
    assert fused_kernels.resolve_dispatch() == "jit", "the env knob is not read"
    with pytest.raises(ValueError) as excinfo:
        fused_kernels.resolve_dispatch("gpu-graph")
    message = str(excinfo.value)
    assert fused_kernels.DISPATCH_ENV in message and "bound" in message and "jit" in message, (
        f"the ValueError must name the knob and both spellings, got {message!r}"
    )


def test_jit_mode_serves_every_call_through_the_dispatcher(monkeypatch, oracle):
    """``EVOGP_FUSED_DISPATCH=jit`` reproduces the pre-change dispatch exactly."""
    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "jit")
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*"], pop=16, seed=37)
    x = data(8, 4)
    got = [_serve(rows, 4, x) for _ in range(3)]
    for index, value in enumerate(got):
        assert_same(value, oracle_value(oracle, rows, x), f"batch_forward[jit mode, call {index}]")
    stats, launches = fused_kernels.dispatch_stats(), fused_engine.fused_cache_stats()
    assert stats["bound_launches"] == 0, (
        f"jit mode must never serve a pre-bound launch, counted {stats}"
    )
    assert stats["jit_launches"] == int(launches["fused_kernel_launches"]) == 3, (
        "every call of a jit-mode plan pays the dispatcher (3 calls, 3 dispatches); "
        f"dispatch={stats} launches={launches['fused_kernel_launches']}"
    )


def test_bound_mode_serves_warm_calls_after_one_dispatch(oracle):
    """The default binds the kernel on the first call and skips the dispatcher after."""
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*"], pop=16, seed=41)
    x = data(8, 4)
    outputs = [_serve(rows, 4, x) for _ in range(4)]
    stats = fused_kernels.dispatch_stats()
    assert stats["declines"] == 0, (
        f"the pre-bound dispatch declined on this box: {stats} -- the route silently "
        "served the historical path, so the timings would be the old ones"
    )
    assert (stats["jit_launches"], stats["bound_launches"]) == (1, 3), (
        "one prepared call = ONE dispatcher pass, then the pre-bound kernel; counted "
        f"{stats}"
    )
    for index, value in enumerate(outputs):
        assert_same(value, oracle_value(oracle, rows, x), f"batch_forward[bound mode, call {index}]")
    assert stats["bound_launches"] + stats["jit_launches"] == int(
        fused_engine.fused_cache_stats()["fused_kernel_launches"]
    ), "every served call is exactly one launch, bound or dispatched"


def test_bound_and_jit_agree_bitwise(monkeypatch):
    """The two launch paths are the SAME device code: value, loss and per-row agree."""
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*", "sin"], seed=43)
    pop_size = int(rows[0].shape[0])
    x, labels = data(12, 4), data(12, 1, seed=5)
    per_row = data(pop_size, 4, seed=11)

    def run_all():
        # TWICE: one prepared call per surface, then one WARM call per surface -- the
        # warm half is where the two launch paths differ.
        run_all_once()
        return run_all_once()

    def run_all_once():
        return (
            _serve(rows, 4, x),
            _serve_loss(rows, 4, x, labels),
            _serve(rows, 4, per_row, per_row=True),
        )

    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "jit")
    fused_engine.clear_fused_cache()
    jit_value, jit_loss, jit_per_row = run_all()
    jit_stats = fused_kernels.dispatch_stats()

    monkeypatch.setenv("EVOGP_FUSED_DISPATCH", "bound")
    fused_engine.clear_fused_cache()
    bound_value, bound_loss, bound_per_row = run_all()
    bound_stats = fused_kernels.dispatch_stats()

    assert jit_stats["bound_launches"] == 0 and bound_stats["bound_launches"] > 0, (
        f"the two arms did not take different paths: jit={jit_stats} bound={bound_stats}"
    )
    for name, want, got in (
        ("batch_forward", jit_value, bound_value),
        ("SR_fitness", jit_loss, bound_loss),
        ("forward", jit_per_row, bound_per_row),
    ):
        assert torch.equal(want, got), (
            f"{name}: the pre-bound launch changed a value (max|delta|="
            f"{float((want - got).abs().max())!r}) -- it must be the same device code"
        )


def test_clear_fused_cache_rearms_the_dispatch():
    """``clear_fused_cache()`` drops the bound kernel with the plan (the invalidation)."""
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*"], pop=16, seed=47)
    x = data(8, 4)
    first = _serve(rows, 4, x)
    _serve(rows, 4, x)
    before = fused_kernels.dispatch_stats()
    assert (before["jit_launches"], before["bound_launches"]) == (1, 1), before

    fused_engine.clear_fused_cache()
    after_clear = fused_kernels.dispatch_stats()
    assert after_clear == {"bound_launches": 0, "jit_launches": 0, "declines": 0}, (
        f"the launch-path counters must reset with the route's tables, got {after_clear}"
    )
    again = _serve(rows, 4, x)
    assert_same(again, first, "batch_forward[after clear_fused_cache]")
    assert fused_kernels.dispatch_stats()["jit_launches"] == 1, (
        "a cleared plan must re-arm (one dispatcher pass) instead of reusing the "
        "previous binding"
    )


# --- 3c. the single-slot fast path's invalidation contract ------------------------- #
# The plan fast path (`fused_engine._LAST`) serves a warm call off tensor IDENTITY
# plus the `_version` vector, so the one way a stale plan can be served is a write
# torch's version counter does not see (a numpy view of the same memory, a foreign
# kernel) -- which is what `clear_fused_cache()` is for.  These cases pin both
# halves: an in-place write (through the tensor or through a VIEW of it) is a MISS
# with correct numbers, and the reset drops the fast path and the geometry memo with
# the tables.
def test_an_in_place_channel_mutation_misses_the_plan_fast_path():
    """A channel written in place is a NEW call: no plan hit, and the numbers follow."""
    fused_engine.clear_fused_cache()
    rows, x = _const_rows(), data(8, 4)
    node_value = rows[0]
    first = served(lambda: _serve(rows, 4, x), "batch_forward[const population]")
    assert_same(first, _serve(rows, 4, x), "batch_forward[const population, repeat]")
    _serve(rows, 4, x)  # the warm repeat: armed the fast path
    before = fused_engine.fused_cache_stats()
    assert int(before["fused_plan_hits"]) >= 1, (
        "the warm repeat must be served by the plan fast path (or the PLAN_MEMO table "
        f"behind it); counted {before['fused_plan_hits']} plan hits"
    )
    with torch.no_grad():
        node_value[3, 0] = 2.5  # row 3 is now the constant 2.5
    mutated = served(lambda: _serve(rows, 4, x), "batch_forward[after the in-place write]")
    after = fused_engine.fused_cache_stats()
    assert int(after["fused_plan_hits"]) == int(before["fused_plan_hits"]), (
        "an in-place channel write must MISS the plan fast path (its `_version` moved), "
        f"but the call was served from it: {before['fused_plan_hits']} -> "
        f"{after['fused_plan_hits']} plan hits"
    )
    assert int(after["fused_plan_misses"]) > int(before["fused_plan_misses"]), (
        "the mutated call must re-prepare (a plan MISS): "
        f"{before['fused_plan_misses']} -> {after['fused_plan_misses']}"
    )
    assert float(mutated[3, 0]) == pytest.approx(2.5), (
        f"row 3 must answer the NEW constant, got {float(mutated[3, 0])!r}"
    )
    assert not torch.equal(mutated[3], first[3]), "the numbers must move with the write"
    assert torch.equal(mutated[4], first[4]), "and only the written row may move"


def test_an_in_place_mutation_through_a_view_misses_the_plan_fast_path():
    """A write through a VIEW shares the version counter -- so it must miss too."""
    fused_engine.clear_fused_cache()
    rows, x = _const_rows(), data(8, 4)
    node_value = rows[0]
    first = served(lambda: _serve(rows, 4, x), "batch_forward[const population]")
    assert_same(_serve(rows, 4, x), first, "batch_forward[warm repeat]")
    before = fused_engine.fused_cache_stats()
    view = node_value[3:5]  # a VIEW: same storage, same `_version`
    assert view.data_ptr() == node_value.data_ptr() + 3 * node_value.stride(0) * (
        node_value.element_size()
    ), "the case is only about views if this is one"
    with torch.no_grad():
        view[0, 0] = 4.25  # writes into row 3 of the population tensor
    mutated = served(lambda: _serve(rows, 4, x), "batch_forward[after the view write]")
    after = fused_engine.fused_cache_stats()
    assert int(after["fused_plan_hits"]) == int(before["fused_plan_hits"]), (
        "a write through a VIEW of a channel must MISS the plan fast path (the version "
        f"counter is shared), but it was served: {before['fused_plan_hits']} -> "
        f"{after['fused_plan_hits']} plan hits"
    )
    assert float(mutated[3, 0]) == pytest.approx(4.25), (
        f"row 3 must answer the value written through the view, got {float(mutated[3, 0])!r}"
    )
    assert float(mutated[4, 0]) == pytest.approx(1.5), "the un-written row must not move"


def test_the_served_fast_path_is_one_launch_per_call():
    """The fast path changes the PLAN KEY, never the launch: one launch per call."""
    fused_engine.clear_fused_cache()
    rows, x = _const_rows(), data(8, 4)
    launches = []
    for _ in range(3):
        stats_before = fused_kernels.dispatch_stats()
        facade_before = fused_engine.fused_cache_stats()
        _serve(rows, 4, x)
        stats_after = fused_kernels.dispatch_stats()
        facade_after = fused_engine.fused_cache_stats()
        launches.append(
            (
                int(stats_after["bound_launches"] - stats_before["bound_launches"])
                + int(stats_after["jit_launches"] - stats_before["jit_launches"]),
                int(facade_after["fused_kernel_launches"] - facade_before["fused_kernel_launches"]),
            )
        )
    assert launches == [(1, 1), (1, 1), (1, 1)], (
        f"every served call must be exactly one dispatch and one launch, counted {launches}"
    )


def test_clear_fused_cache_drops_the_fast_path_and_the_geometry_memo():
    """The reset reaches the two in-memory tables a warm call leaves behind."""
    rows, x = _const_rows(), data(8, 4)
    fused_engine.clear_fused_cache()
    first = fused_engine.fused_cache_stats()
    assert int(first["fused_plan_hits"]) == 0 and int(first["fused_plan_misses"]) == 0
    served_values = served(lambda: _serve(rows, 4, x), "batch_forward[clear_fused_cache -> first]")
    _serve(rows, 4, x)
    assert fused_engine._LAST is not None, (
        "the two calls above must have armed the single-slot fast path"
    )
    assert fused_engine._GEOMETRY_MEMO, "the calls above must have filled the geometry memo"
    fused_engine.clear_fused_cache()
    assert fused_engine._LAST is None, "clear_fused_cache() must drop the single-slot fast path"
    assert not fused_engine._GEOMETRY_MEMO, "clear_fused_cache() must drop the geometry memo"
    again = served(lambda: _serve(rows, 4, x), "batch_forward[clear_fused_cache -> re-served]")
    assert_same(again, served_values, "batch_forward[numbers before/after the reset]")


def test_active_launch_hooks_decline(monkeypatch, oracle):
    """A registered Triton launch hook keeps the historical dispatch (profilers)."""
    triton_knobs = pytest.importorskip("triton.knobs")
    hook = triton_knobs.HookChain()
    seen = []
    hook.add(lambda *args, **kwargs: seen.append(1))
    monkeypatch.setattr(triton_knobs.runtime, "launch_enter_hook", hook)
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*"], pop=16, seed=53)
    x = data(8, 4)
    got = [_serve(rows, 4, x) for _ in range(3)]
    stats = fused_kernels.dispatch_stats()
    assert stats["declines"] >= 1, (
        "an ACTIVE launch hook must decline the pre-bound path (it would skip the "
        f"hook); counted {stats}"
    )
    assert stats["bound_launches"] == 0, stats
    assert seen, "the hook itself must have run (the C launcher still calls it)"
    for index, value in enumerate(got):
        assert_same(value, oracle_value(oracle, rows, x), f"batch_forward[hooked, call {index}]")


def test_unusable_launcher_declines(monkeypatch, oracle):
    """A backend launcher of another shape keeps the historical dispatch, silently."""
    monkeypatch.setattr(fused_kernels, "_launcher_takes_launch_args", lambda launcher: False)
    fused_engine.clear_fused_cache()
    rows = _random_rows(["+", "*"], pop=16, seed=59)
    x = data(8, 4)
    got = [_serve(rows, 4, x) for _ in range(3)]
    stats = fused_kernels.dispatch_stats()
    assert stats["bound_launches"] == 0 and stats["declines"] >= 1, stats
    for index, value in enumerate(got):
        assert_same(value, oracle_value(oracle, rows, x), f"batch_forward[declined, call {index}]")
